#include <SE.Math.hpp>
#include <SE.HSL.hpp>
using namespace SIByL;
using namespace SIByL::Math;
using namespace SIByL::HSL;

sampler2D hi_z;

inline auto clamp(vec2 v, float min, float max) noexcept -> vec2 {
  return vec2{clamp(v.x, min, max), clamp(v.y, min, max)};
}

vec2 saturate(vec2 v) { return clamp(v, 0.0, 1.0); }

// Returns the number of cells in the quad tree at the given level.
vec2 getCellCount(in const int mipLevel) {
  return vec2(textureSize(hi_z, mipLevel));
}

// Returns the 2D integer index of the cell that contains the given 2D position
// within it.
vec2 getCell(in const vec2 pos, in const vec2 cell_count) {
  return vec2(floor(pos * cell_count));
}

// Returns the 2D integer index of the cell that contains the given 2D position
// within it.
vec2 getCellCoord(in const vec2 pos, in const vec2 cell_count) {
  return vec2(pos * cell_count);
}

vec3 intersectDepthPlane(in const vec3 o, in const vec3 d, float z) {
  return o + d * z;
}

vec3 intersectCellBoundary(
    in const vec3 o, in const vec3 d, in const vec2 cell,
    in const vec2 cell_count, in const vec2 crossStep,
    in const vec2 crossOffset) {  // crossStep is added to the current cell to
                                  // get the next cell index
  vec2 index = cell + crossStep;
  // By dividing the cell index by cell count, we can get the position
  // of the boundaries between the current cell and the new cell .
  vec2 boundary = index / cell_count;
  // crossOffset is used to push the position just a tiny bit further
  // to make sure the new position is not right on the boundary.
  boundary += crossOffset;

  // delta between the new position and the origin
  vec2 delta = boundary - o.xy;
  // After the division, the x and y component in delta will have value between
  // 0 to 1 which represents how far the delta position is from the origin of
  // the ray.
  delta /= d.xy;
  // take the minimum of the two components,
  // x and y of delta because we want to cross the nearest boundary
  float t = std::min(delta.x, delta.y);

  return intersectDepthPlane(o, d, t);
}

bool crossedCellBoundary(in const vec2 cellIdxOne, in const vec2 cellIdxTwo) {
  return floor(cellIdxOne.x) != floor(cellIdxTwo.x) ||
         floor(cellIdxOne.y) != floor(cellIdxTwo.y);
}

/**
 * Do DDA over the 2D plane.
 * @ref: https://www.shadertoy.com/view/Nl23RV
 */
struct Trace2D_Linear {
  vec2 ro;   // ray origin
  vec2 rd;   // ray direction, normalized
  vec2 dd;   // 1/abs(rd)
  ivec2 sd;  // sign(rd)
  float t0;  // range start
  float t1;  // range end
};

struct DDA2D_Linear {
  vec2 st;   // t of next axis boundary
  ivec2 mp;  // current cell index
  float t;   // current trace at
};

struct DebugPack {
  ivec2 qi;
  vec2 pixel;
  float cd;
  bool celltouchesline;
};

void setup(vec2 pa, vec2 pb, DDA2D_Linear* dda, Trace2D_Linear* tr) {
  // Trace2D_Linear setup
  tr->ro = pa;
  tr->rd = pb - pa;
  vec2 ad = abs(tr->rd);
  // sign() is somewhat buggy, can sometimes produce values *slightly* below 1.0
  // in magnitude, which convert to integer zero... many workarounds
  tr->sd = ivec2(sign(tr->rd) * 1.001);
  tr->t1 = length(tr->rd);  // could use ad but abs isn't necessary here
  tr->t0 = 0.;
  tr->rd /= tr->t1;  // tr.rd = normalize(tr.rd);
  ad /= tr->t1;
  tr->dd = vec2(1) / max(ad, Math::vec2{1e-32});  // will be effectively
                                                      // infinity
                                                 // where rd was
                                  // 0, so
                           // only use when actually crossing between cells
                           // DDA2D_Linear setup
  dda->t = 0.;
  dda->mp = ivec2(floor(tr->ro));  // current cell index
  dda->st = tr->ro - vec2(dda->mp);  // fract(tr.ro);
  dda->st = (vec2(.5) - dda->st) * vec2(tr->sd) + .5;
  // compensate when starting precisely on an integer (grid cell boundary) going
  // in negative direction, would be otherwise off by one
  ivec2 sot =
      ivec2(equal(dda->st, vec2(0)));    // mask for on cell boundary (integer)
  dda->mp += sot * min(tr->sd, ivec2(0));  // factor mask for negative directions
  dda->st += vec2(
      sot);  // step cannot be zero.  If start on boundary, just go to next one!
  dda->st *= tr->dd;
}

bool traverse(inout DDA2D_Linear dda, inout Trace2D_Linear tr
) {
  return dda.t < tr.t1;
}

int nextIsect(inout DDA2D_Linear& dda, in const Trace2D_Linear tr) {
  int s = dda.st.x <= dda.st.y ? 0 : 1;  // which cell boundary is closer?
  dda.t = dda.st[s];                     // update total distance traversed
  dda.st[s] += tr.dd[s];                 // step to next boundary
  dda.mp[s] += tr.sd[s];                 // move, which way? to next cell
  return s;
}

vec2 Scan(inout DDA2D_Linear& dda, Trace2D_Linear tr, out ivec2& n
) {
  float ot = dda.t;
  int s;
  while (traverse(dda, tr)) {
    vec2 pos = tr.ro + tr.rd* dda.t;
    s = nextIsect(dda, tr);
  }
  n = ivec2(0);
  n[s] = -tr.sd[s];  // avoid negating zeroes
  return vec2(ot, dda.t);
}

int main() {
    const vec2 cellCount = {1280, 720};
    const vec2 m0 = vec2(2, 1);
    const vec2 m1 = vec2(0, 0);

    mat4 proj = perspective(45, 1, 0.01, 100).m;

    vec3 p1 = vec3(-1, 0, 1);
    vec3 p2 = vec3(1, 0, 50);
    vec3 pm = (p1 + p2) / 2;

    vec4 p1_proj = proj * vec4(p1, 1);
    p1_proj /= p1_proj.w;
    vec4 p2_proj = proj * vec4(p2, 1);
    p2_proj /= p2_proj.w;
    vec4 pm_proj = proj * vec4(pm, 1);
    pm_proj /= pm_proj.w;

    vec3 ro = {0, 0, 0};
    vec3 rd = {0, 0, 1};
    {
        vec3 points[4];
        points[0] = vec3(0, 0, 1);
        points[1] = vec3(1, 0, 1);
        points[2] = vec3(0, 1, 1);
        points[3] = vec3(1, 1, 1);
        vec3 ro = {0.25, 0.25, 0};
        vec3 rd = {0, 0, 1};

    bool isected = ray_triangle_intersection_lightweight(
            ro, rd, points[0], points[1], points[2]);
        ro = {0.75, 0.75, 0};
        rd = {0, 0, 1};

    bool isected2 = ray_triangle_intersection_lightweight(
            ro, rd, points[2], points[1], points[3]);
        //bool isected2 = ray_triangle_intersection_lightweight(ro, rd, A, B, C);
        float a = 1.f;
    }
    {
        vec3 A = {1, 0, 1};
        vec3 B = {2, +1, 1};
        vec3 C = {2, -1, 1};
        bool isected = ray_triangle_intersection_lightweight(ro, rd, A, B, C);
        float a = 1.f;
    }

    DDA2D_Linear dda;
    Trace2D_Linear tr;
    setup(m0, m1, &dda, &tr);
    ivec2 n;
    Scan(dda, tr, n);
    //const vec2 cellCount_0 = getCellCount(0);


    //const vec3 vReflDirInTS = normalize(vec3(1, 0.5, 0.2));

    //vec2 cellCount = {512, 512};

    //vec2 crossStep =
    //    vec2(vReflDirInTS.x >= 0 ? 1 : -1, vReflDirInTS.y >= 0 ? 1 : -1);
    //vec2 crossOffset = crossStep / vec2(512) / 128;
    //crossStep = saturate(crossStep);

    //vec3 ray = vec3{0.5, 0.5, 0};
    //vec3 o = ray;
    //vec3 d = vReflDirInTS;
    //vec3 test = ray * 512;
    //for (int i = 0; i < 100; ++i) {
    //  const vec2 oldCellIdx = getCell(ray.xy, cellCount);
    //  ray = intersectCellBoundary(o, d, oldCellIdx, cellCount, crossStep,
    //                              crossOffset);
    //  test = ray * 512;    
    //}

    return 0; }