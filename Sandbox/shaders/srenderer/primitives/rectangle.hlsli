#ifndef _SRENDERER_RECTANGLE_PRIMITIVE_
#define _SRENDERER_RECTANGLE_PRIMITIVE_

#include "common/math.hlsli"
#include "srenderer/spt.hlsli"

/*****************************************************************************************
 * This code is adapted from shadertoy @koiava
 * @ref: https://www.shadertoy.com/view/ltsyzl
 *****************************************************************************************/

/**
 * Check intersection between a ray and a axis-aligned (xy) quad.
 * @param ray: the ray
 * @param pmin: the minimum point of the quad (xy only) (in object space)
 * @param pmax: the maximum point of the quad (xy only) (in object space)
 * @param t: the distance from the ray origin to the intersection point
 * @return: true if the ray intersects the quad, false otherwise
 */
bool RayAAPlaneIntersection_Visibility(
    in_ref(Ray) ray,
    in_ref(float2) pmin, in_ref(float2) pmax,
    out float t)
{
    t = -1.0;
    if (abs(ray.direction.z) < 1e-6)
        return false;
    t = (-ray.origin.z) / ray.direction.z;
    const float3 pos = ray.origin + ray.direction * t;
    if ((pos.x < pmin.x) ||
        (pos.x > pmax.x) ||
        (pos.y < pmin.y) ||
        (pos.y > pmax.y))
        return false;

    return true;
}

struct SphQuad {
    float3 o, x, y, z;     // local reference system ’R’
    float z0, z0sq;        //
    float x0, y0, y0sq;    // rectangle coords in ’R’
    float x1, y1, y1sq;    //
    float b0, b1, b0sq, k; // misc precomputed constants
    float S;               // solid angle of ’Q’
};

/**
 * Initialize a spherical quad.
 * @param s: the left bottom corner of the quad (in object space)
 * @param ex: the vector from s to the right bottom corner of the quad (in object space)
 * @param ey: the vector from s to the left top corner of the quad (in object space)
 * @param o: the position of the reference point (in object space)
 * @param squad: the description of the spherical quad
 */
void SphQuadInit(float3 s, float3 ex, float3 ey, float3 o, out SphQuad squad) {
    squad.o = o;
    float exl = length(ex);
    float eyl = length(ey);
    // compute local reference system ’R’
    squad.x = ex / exl;
    squad.y = ey / eyl;
    squad.z = cross(squad.x, squad.y);
    // compute rectangle coords in local reference system
    float3 d = s - o;
    squad.z0 = dot(d, squad.z);
    // flip ’z’ to make it point against ’Q’
    if (squad.z0 > 0.) {
        squad.z *= -1.;
        squad.z0 *= -1.;
    }
    squad.z0sq = squad.z0 * squad.z0;
    squad.x0 = dot(d, squad.x);
    squad.y0 = dot(d, squad.y);
    squad.x1 = squad.x0 + exl;
    squad.y1 = squad.y0 + eyl;
    squad.y0sq = squad.y0 * squad.y0;
    squad.y1sq = squad.y1 * squad.y1;
    // create vectors to four vertices
    float3 v00 = float3(squad.x0, squad.y0, squad.z0);
    float3 v01 = float3(squad.x0, squad.y1, squad.z0);
    float3 v10 = float3(squad.x1, squad.y0, squad.z0);
    float3 v11 = float3(squad.x1, squad.y1, squad.z0);
    // compute normals to edges
    float3 n0 = normalize(cross(v00, v10));
    float3 n1 = normalize(cross(v10, v11));
    float3 n2 = normalize(cross(v11, v01));
    float3 n3 = normalize(cross(v01, v00));
    // compute internal angles (gamma_i)
    float g0 = acos(-dot(n0, n1));
    float g1 = acos(-dot(n1, n2));
    float g2 = acos(-dot(n2, n3));
    float g3 = acos(-dot(n3, n0));
    // compute predefined constants
    squad.b0 = n0.z;
    squad.b1 = n2.z;
    squad.b0sq = squad.b0 * squad.b0;
    squad.k = 2. * k_pi - g2 - g3;
    // compute solid angle from internal angles
    squad.S = g0 + g1 - squad.k;
}

/**
 * Draw a sample from a quad, uniformly distributed over the spherical solid angle.
 * @param pos: the position of the sampling point (in object space)
 * @param squad: the description of the spherical quad
 * @param uv: the sample point in [0, 1]^2
 * @param w: the sampled direction
 * @param pdfW: the probability density of the sampled direction
 */
#define EPSILON 0.0001
void SampleSphQuad(
    in_ref(float3) pos, in_ref(SphQuad) squad,
    in_ref(float2) uv,
    out_ref(float3) w, out_ref(float) pdf)
{
    // 1. compute ’cu’
    float au = uv.x * squad.S + squad.k;
    float fu = (cos(au) * squad.b0 - squad.b1) / sin(au);
    float cu = 1. / sqrt(fu * fu + squad.b0sq) * (fu > 0. ? +1. : -1.);
    cu = clamp(cu, -1., 1.); // avoid NaNs
    // 2. compute ’xu’
    float xu = -(cu * squad.z0) / sqrt(1. - cu * cu);
    xu = clamp(xu, squad.x0, squad.x1); // avoid Infs
    // 3. compute ’yv’
    float d = sqrt(xu * xu + squad.z0sq);
    float h0 = squad.y0 / sqrt(d * d + squad.y0sq);
    float h1 = squad.y1 / sqrt(d * d + squad.y1sq);
    float hv = h0 + uv.y * (h1 - h0);
    float hv2 = hv * hv;
    float yv = (hv2 < 1. - EPSILON) ? (hv * d) / sqrt(1. - hv2) : squad.y1;
    // 4. transform (xu,yv,z0) to world coords
    float3 p = (squad.o + xu * squad.x + yv * squad.y + squad.z0 * squad.z);
    w = normalize(p - pos);
    pdf = 1. / squad.S;
}
#undef EPSILON

/**
 * Draw a sample from a quad, uniformly distributed over the surface area.
 * @param pos: the position of the sampling point (in object space)
 * @param pmin: the minimum point of the quad (xy only) (in object space)
 * @param pmax: the maximum point of the quad (xy only) (in object space)
 * @param uv: the sample point in [0, 1]^2
 * @param w: the sampled direction
 * @param p: the sampled point
 * @param pdfW: the probability density of the sampled direction
 */
void SampleQuadArea(
    in_ref(float3) pos, in_ref(float2) pmin, in_ref(float2) pmax,
    in_ref(float2) uv,
    out_ref(float3) w, out_ref(float3) p, out_ref(float) pdfW)
{
    const float2 size = pmax - pmin;
    const float pdfA = 1.0 / (size.x * size.y);
    p = float3(pmin + size * uv, 0.0);
    w = p - pos;
    const float d2 = dot(w, w);
    w /= sqrt(d2);
    float aCosThere = max(0.0, -w.z);
    pdfW = PdfAtoW(pdfA, d2, aCosThere);
}

/**
 * Create a spherical quad from a square quad.
 * @param local The local coordinate of the reference point position.
 * @param extend The extend of the square quad.
 * @return The spherical quad.
 */
SphQuad CreateSphQuad(in_ref(float3) local, float extend) {
    const float extend2 = extend + extend;
    SphQuad squad;
    SphQuadInit(float3(-extend, -extend, 0), float3(extend2, 0, 0), float3(0, extend2, 0), local, squad);
    return squad;
}

SphQuad CreateSphQuad(in_ref(float3) local, float2 extend) {
    const float2 extend2 = extend + extend;
    SphQuad squad;
    SphQuadInit(float3(-extend, 0), float3(extend2.x, 0, 0), float3(0, extend2.y, 0), local, squad);
    // if (local.z < 0) squad.S = 0;
    return squad;
}

#endif // _SRENDERER_RECTANGLE_PRIMITIVE_