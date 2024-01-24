#include <map>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>
#include "../Public/SE.MeSh.Simplify.hpp"

#pragma warning(disable:4996)

template <>
struct std::hash<Math::vec3> {
  std::size_t operator()(Math::vec3 const& k) const {
    return ((hash<float>()(k.x) ^ (hash<float>()(k.y) << 1)) >> 1) ^
           (hash<int>()(k.z) << 1);
  }
};

namespace QuadricSimplify {
/*
 * The implementation adapted from the following github repo:
 * @url: https://github.com/sp4cerat/Fast-Quadric-Mesh-Simplification
 * With original copyright as follows:
 *  ////////////////////////////////////////////////////////////////
 *  // Mesh Simplification Tutorial
 *  // (C) by Sven Forstmann in 2014
 *  // License : MIT
 *  // http://opensource.org/licenses/MIT
 *  // https://github.com/sp4cerat/Fast-Quadric-Mesh-Simplification
 *  // 5/2016: Chris Rorden created minimal version for OSX/Linux/Windows
 *compile
 *  ////////////////////////////////////////////////////////////////
 **/

#define loopi(start_l, end_l) for (int i = start_l; i < end_l; ++i)
#define loopi(start_l, end_l) for (int i = start_l; i < end_l; ++i)
#define loopj(start_l, end_l) for (int j = start_l; j < end_l; ++j)
#define loopk(start_l, end_l) for (int k = start_l; k < end_l; ++k)
    
class SymetricMatrix {
 public:
  // Constructor
  SymetricMatrix(double c = 0) { loopi(0, 10) m[i] = c; }
  SymetricMatrix(double m11, double m12, double m13, double m14, double m22,
                 double m23, double m24, double m33, double m34, double m44) {
    m[0] = m11; m[1] = m12; m[2] = m13; m[3] = m14; m[4] = m22;
    m[5] = m23; m[6] = m24; m[7] = m33; m[8] = m34; m[9] = m44; }
  // Make plane
  SymetricMatrix(double a, double b, double c, double d) {
    m[0] = a * a; m[1] = a * b; m[2] = a * c; m[3] = a * d; m[4] = b * b;
    m[5] = b * c; m[6] = b * d; m[7] = c * c; m[8] = c * d; m[9] = d * d;
  }
  double operator[](int c) const { return m[c]; }
  // Determinant
  double det(int a11, int a12, int a13, int a21, int a22, int a23, int a31,
             int a32, int a33) {
    return m[a11] * m[a22] * m[a33] + m[a13] * m[a21] * m[a32] +
           m[a12] * m[a23] * m[a31] - m[a13] * m[a22] * m[a31] -
           m[a11] * m[a23] * m[a32] - m[a12] * m[a21] * m[a33];
  }

  const SymetricMatrix operator+(const SymetricMatrix &n) const {
    return SymetricMatrix(m[0] + n[0], m[1] + n[1], m[2] + n[2], m[3] + n[3],
                          m[4] + n[4], m[5] + n[5], m[6] + n[6], m[7] + n[7],
                          m[8] + n[8], m[9] + n[9]);
  }

  SymetricMatrix &operator+=(const SymetricMatrix &n) {
    m[0] += n[0]; m[1] += n[1]; m[2] += n[2]; m[3] += n[3]; m[4] += n[4];
    m[5] += n[5]; m[6] += n[6]; m[7] += n[7]; m[8] += n[8]; m[9] += n[9];
    return *this;
  }

  double m[10] = {0.};
};

// Global Variables & Strctures
struct Triangle {
  int v[3];
  double err[4];
  int deleted = 0, dirty, attr;
  Math::dvec3 n;
  Math::vec2 uvs[3];
  int material;
  // skeleton data
  Math::Vector4<size_t> joints[3];
  Math::vec4 weights[3];
};

struct Vertex {
  Math::dvec3 p;
  int tstart = 0, tcount = 0;
  SymetricMatrix q;
  int border = 0;
};

struct Ref {
  int tid, tvertex;
};

struct Mesh {
  std::vector<Triangle> triangles;
  std::vector<Vertex> vertices;
  std::vector<Ref> refs;
  bool use_skinning = false;
};


// Error between vertex and Quadric
double vertex_error(SymetricMatrix q, double x, double y, double z) {
  return q[0] * x * x + 2 * q[1] * x * y + 2 * q[2] * x * z + 2 * q[3] * x +
         q[4] * y * y + 2 * q[5] * y * z + 2 * q[6] * y + q[7] * z * z +
         2 * q[8] * z + q[9];
}

// Error for one edge
double calculate_error(Mesh &mesh, int id_v1, int id_v2, Math::dvec3& p_result) {
  // compute interpolated vertex
  SymetricMatrix q = mesh.vertices[id_v1].q + mesh.vertices[id_v2].q;
  bool border = mesh.vertices[id_v1].border & mesh.vertices[id_v2].border;
  double error = 0;
  double det = q.det(0, 1, 2, 1, 4, 5, 2, 5, 7);
  if (det != 0 && !border) {
    // q_delta is invertible
    p_result.x = -1. / det * (q.det(1, 2, 3, 4, 5, 6, 5, 7, 8)); // vx = A41/det(q_delta)
    p_result.y = 1. / det * (q.det(0, 2, 3, 1, 5, 6, 2, 7, 8));  // vy = A42/det(q_delta)
    p_result.z = -1. / det * (q.det(0, 1, 3, 1, 4, 6, 2, 5, 8)); // vz = A43/det(q_delta)
    error = vertex_error(q, p_result.x, p_result.y, p_result.z);
  } else {
    // det = 0 -> try to find best result
    Math::dvec3 p1 = mesh.vertices[id_v1].p;
    Math::dvec3 p2 = mesh.vertices[id_v2].p;
    Math::dvec3 p3 = (p1 + p2) / 2;
    double error1 = vertex_error(q, p1.x, p1.y, p1.z);
    double error2 = vertex_error(q, p2.x, p2.y, p2.z);
    double error3 = vertex_error(q, p3.x, p3.y, p3.z);
    error = std::min(error1, std::min(error2, error3));
    if (error1 == error) p_result = p1;
    if (error2 == error) p_result = p2;
    if (error3 == error) p_result = p3;
  }
  return error;
}

// compact triangles, compute edge error and build reference list
void update_mesh(Mesh &mesh, int iteration) {
  // compact triangles
  if (iteration > 0) {
    int dst = 0;
    loopi(0, mesh.triangles.size()) 
      if (!mesh.triangles[i].deleted)
        mesh.triangles[dst++] = mesh.triangles[i];
    mesh.triangles.resize(dst);
  }
  // Init Reference ID list
  loopi(0, mesh.vertices.size()) {
    mesh.vertices[i].tstart = 0;
    mesh.vertices[i].tcount = 0;
  }
  loopi(0, mesh.triangles.size()) {
    Triangle &t = mesh.triangles[i];
    loopj(0, 3) mesh.vertices[t.v[j]].tcount++;
  }
  int tstart = 0;
  loopi(0, mesh.vertices.size()) {
    Vertex &v = mesh.vertices[i];
    v.tstart = tstart;
    tstart += v.tcount;
    v.tcount = 0;
  }
  // Write References
  mesh.refs.resize(mesh.triangles.size() * 3);
  loopi(0, mesh.triangles.size()) {
    Triangle &t = mesh.triangles[i];
    loopj(0, 3) {
      Vertex &v = mesh.vertices[t.v[j]];
      mesh.refs[v.tstart + v.tcount].tid = i;
      mesh.refs[v.tstart + v.tcount].tvertex = j;
      v.tcount++;
    }
  }

  // Init Quadrics by Plane & Edge Errors
  //
  // required at the beginning ( iteration == 0 )
  // recomputing during the simplification is not required,
  // but mostly improves the result for closed meshes
  //
  if (iteration == 0) {
    // Identify boundary : vertices[].border=0,1
    std::vector<int> vcount, vids;
    loopi(0, mesh.vertices.size()) mesh.vertices[i].border = 0;
    loopi(0, mesh.vertices.size()) {
      Vertex &v = mesh.vertices[i];
      vcount.clear();
      vids.clear();
      loopj(0, v.tcount) {
        int k = mesh.refs[v.tstart + j].tid;
        Triangle &t = mesh.triangles[k];
        loopk(0, 3) {
          int ofs = 0, id = t.v[k];
          while (ofs < vcount.size()) {
            if (vids[ofs] == id) break;
            ofs++;
          }
          if (ofs == vcount.size()) {
            vcount.push_back(1);
            vids.push_back(id);
          } else
            vcount[ofs]++;
        }
      }
      loopj(0, vcount.size()) if (vcount[j] == 1) 
          mesh.vertices[vids[j]].border = 1;
    }
    // initialize errors
    loopi(0, mesh.vertices.size()) mesh.vertices[i].q = SymetricMatrix(0.0);

    loopi(0, mesh.triangles.size()) {
      Triangle &t = mesh.triangles[i];
      Math::dvec3 n, p[3];
      loopj(0, 3) p[j] = mesh.vertices[t.v[j]].p;
      n = Math::cross(p[1] - p[0], p[2] - p[0]);
      n = Math::normalize(n);
      t.n = n;
      loopj(0, 3) mesh.vertices[t.v[j]].q =
          mesh.vertices[t.v[j]].q +
          SymetricMatrix(n.x, n.y, n.z, -Math::dot(n, p[0]));
    }
    loopi(0, mesh.triangles.size()) {
      // Calc Edge Error
      Triangle &t = mesh.triangles[i];
      Math::dvec3 p;
      loopj(0, 3) t.err[j] = calculate_error(mesh, t.v[j], t.v[(j + 1) % 3], p);
      t.err[3] = std::min(t.err[0], std::min(t.err[1], t.err[2]));
    }
  }
}

// Check if a triangle flips when this edge is removed
bool flipped(Mesh &mesh, Math::dvec3 p, int i0, int i1, Vertex &v0, Vertex &v1,
             std::vector<int> &deleted) {
  loopk(0, v0.tcount) {
    Triangle &t = mesh.triangles[mesh.refs[v0.tstart + k].tid];
    if (t.deleted) continue;
    int s = mesh.refs[v0.tstart + k].tvertex;
    int id1 = t.v[(s + 1) % 3];
    int id2 = t.v[(s + 2) % 3];
    // delete ?
    if (id1 == i1 || id2 == i1) {
      deleted[k] = 1;
      continue;
    }
    Math::dvec3 d1 = mesh.vertices[id1].p - p;
    d1 = Math::normalize(d1);
    Math::dvec3 d2 = mesh.vertices[id2].p - p;
    d2 = Math::normalize(d2);
    if (std::fabs(Math::dot(d1, d2)) > 0.999) return true;
    Math::dvec3 n;
    n = Math::cross(d1, d2);
    n = Math::normalize(n);
    deleted[k] = 0;
    if (Math::dot(n, t.n) < 0.2) return true;
  }
  return false;
}

// Update triangle connections and edge error after a edge is collapsed

 void update_triangles(Mesh &mesh, int i0, Vertex &v, 
   std::vector<int> &deleted, int &deleted_triangles) {
  Math::dvec3 p;
  loopk(0, v.tcount) {
    Ref &r = mesh.refs[v.tstart + k];
    Triangle& t = mesh.triangles[r.tid];
    if (t.deleted) continue;
    if (deleted[k]) {
      t.deleted = 1;
      deleted_triangles++;
      continue;
    }
    t.v[r.tvertex] = i0;
    t.dirty = 1;
    t.err[0] = calculate_error(mesh, t.v[0], t.v[1], p);
    t.err[1] = calculate_error(mesh, t.v[1], t.v[2], p);
    t.err[2] = calculate_error(mesh, t.v[2], t.v[0], p);
    t.err[3] = std::min(t.err[0], std::min(t.err[1], t.err[2]));
    mesh.refs.push_back(r);
  }
}

 // Finally compact mesh before exiting
void compact_mesh(Mesh &mesh) {
  int dst = 0;
  loopi(0, mesh.vertices.size()) { mesh.vertices[i].tcount = 0; }
  loopi(0, mesh.triangles.size()) if (!mesh.triangles[i].deleted) {
    Triangle &t = mesh.triangles[i];
    mesh.triangles[dst++] = t;
    loopj(0, 3) mesh.vertices[t.v[j]].tcount = 1;
  }
  mesh.triangles.resize(dst);
  dst = 0;
  loopi(0, mesh.vertices.size()) if (mesh.vertices[i].tcount) {
    mesh.vertices[i].tstart = dst;
    mesh.vertices[dst].p = mesh.vertices[i].p;
    dst++;
  }
  loopi(0, mesh.triangles.size()) {
    Triangle &t = mesh.triangles[i];
    loopj(0, 3) t.v[j] = mesh.vertices[t.v[j]].tstart;
  }
  mesh.vertices.resize(dst);
}

Math::dvec3 barycentric(Math::dvec3 const &p, Math::dvec3 const &a,
                        Math::dvec3 const &b, Math::dvec3 const &c) {
  Math::dvec3 v0 = b - a;
  Math::dvec3 v1 = c - a;
  Math::dvec3 v2 = p - a;
  double d00 = Math::dot(v0, v0);
  double d01 = Math::dot(v0, v1);
  double d11 = Math::dot(v1, v1);
  double d20 = Math::dot(v2, v0);
  double d21 = Math::dot(v2, v1);
  double denom = d00 * d11 - d01 * d01;
  double v = (d11 * d20 - d01 * d21) / denom;
  double w = (d00 * d21 - d01 * d20) / denom;
  double u = 1.0 - v - w;
  return Math::dvec3(u, v, w);
}

template<class T>
T interpolate(Math::dvec3 const& bary, T attrs[3]) {
  T out;
  out = out + attrs[0] * bary.x;
  out = out + attrs[1] * bary.y;
  out = out + attrs[2] * bary.z;
  return out;
}

void update_attributes(Mesh &mesh, int i0, Vertex const &v,
                       Math::dvec3 const &p, std::vector<int> &deleted) {
  loopk(0,v.tcount) {
    Ref &r = mesh.refs[v.tstart + k];
    Triangle &t = mesh.triangles[r.tid];
    if (t.deleted) continue;
    if (deleted[k]) continue;
    Math::dvec3 p1 = mesh.vertices[t.v[0]].p;
    Math::dvec3 p2 = mesh.vertices[t.v[1]].p;
    Math::dvec3 p3 = mesh.vertices[t.v[2]].p;

    Math::dvec3 const& bary = barycentric(p, p1, p2, p3);
    t.uvs[r.tvertex] = interpolate(bary, t.uvs);
    /*
    if (mesh.use_skinning) {
      if (v1.joints != v0.joints) {
        bool test = 1.f;
      }
      if (v1.weights != v0.weights) {
        bool test = 1.f;
      }
    }*/

  }
}

// Main simplification function
// target_count  : target nr. of triangles
// agressiveness : sharpness to increase the threshold.
//                 5..8 are good numbers
//                 more iterations yield higher quality
void simplify_mesh(Mesh& mesh, int target_count, 
  double agressiveness = 7, bool verbose = false) {
  // init
  loopi(0, mesh.triangles.size()) { mesh.triangles[i].deleted = 0; }
  // main iteration loop
  int deleted_triangles = 0;
  std::vector<int> deleted0, deleted1;
  int triangle_count = mesh.triangles.size();
  // take iterations that
  for (int iteration = 0; iteration < 100; iteration++) {
    if (triangle_count - deleted_triangles <= target_count) break;
    // update mesh once in a while
    if (iteration % 5 == 0) {
      update_mesh(mesh, iteration);
    }
    // clear dirty flag
    loopi(0, mesh.triangles.size()) mesh.triangles[i].dirty = 0;
    // All triangles with edges below the threshold will be removed
    // The following numbers works well for most models.
    // If it does not, try to adjust the 3 parameters
    double threshold = 0.000000001 * pow(double(iteration + 3), agressiveness);
    // target number of triangles reached ? Then break
    if ((verbose) && (iteration % 5 == 0)) {
      Core::LogManager::Debug(std::format("iteration {0} - triangles {1} threshold {2}", 
        iteration, triangle_count - deleted_triangles, threshold));
    }
    // remove vertices & mark deleted triangles
    loopi(0, mesh.triangles.size()) {
      Triangle &t = mesh.triangles[i];
      if (t.err[3] > threshold) continue;
      if (t.deleted) continue;
      if (t.dirty) continue;

      loopj(0, 3) if (t.err[j] < threshold) {
        int i0 = t.v[j];
        Vertex &v0 = mesh.vertices[i0];
        int i1 = t.v[(j + 1) % 3];
        Vertex &v1 = mesh.vertices[i1];
        // Border check
        if (v0.border != v1.border) continue;

        // Compute vertex to collapse to
        Math::dvec3 p;
        calculate_error(mesh, i0, i1, p);
        deleted0.resize(v0.tcount);  // normals temporarily
        deleted1.resize(v1.tcount);  // normals temporarily
        // don't remove if flipped
        if (flipped(mesh, p, i0, i1, v0, v1, deleted0)) continue;
        if (flipped(mesh, p, i1, i0, v1, v0, deleted1)) continue;

        // update UV and more attributes information
        update_attributes(mesh, i0, v0, p, deleted0);
        update_attributes(mesh, i0, v1, p, deleted1);
        
        // not flipped, so remove edge
        v0.p = p;
        v0.q = v1.q + v0.q;
        int tstart = mesh.refs.size();
        update_triangles(mesh, i0, v0, deleted0, deleted_triangles);
        update_triangles(mesh, i0, v1, deleted1, deleted_triangles);

        int tcount = mesh.refs.size() - tstart;
        if (tcount <= v0.tcount) {
          // save ram
          if (tcount)
            memcpy(&mesh.refs[v0.tstart], &mesh.refs[tstart], tcount * sizeof(Ref));
        } else // append
          v0.tstart = tstart;
        v0.tcount = tcount;
        break;
      }
      // done?
      if (triangle_count - deleted_triangles <= target_count) break;
    }
  }
  // clean up mesh
  compact_mesh(mesh);
}

QuadricSimplify::Mesh to_quadric_mesh(GFX::Mesh const &mesh) {
  QuadricSimplify::Mesh quadmesh;
  std::span<float> const vertex_buffer(
      (float*)mesh.vertexBuffer_host.data,
      mesh.vertexBuffer_host.size / sizeof(float));
  std::span<float> const position_buffer(
      (float*)mesh.positionBuffer_host.data,
      mesh.positionBuffer_host.size / sizeof(float));
  std::span<uint32_t> const indices_buffer(
      (uint32_t*)mesh.indexBuffer_host.data,
      mesh.indexBuffer_host.size / sizeof(uint32_t));
  std::span<Math::Vector4<size_t>> const joints_buffer(
      (Math::Vector4<size_t>*)mesh.jointIndexBuffer_host.data,
      mesh.jointIndexBuffer_host.size / sizeof(Math::Vector4<size_t>));
  std::span<Math::vec4> const weights_buffer(
      (Math::vec4 *)mesh.jointWeightBuffer_host.data,
      mesh.jointWeightBuffer_host.size / sizeof(Math::vec4));
  quadmesh.use_skinning = joints_buffer.size() > 0;

  size_t vertex_offset = 0;
  for (auto &submesh : mesh.submeshes) {
    std::unordered_map<Math::vec3, uint32_t> map;
    std::vector<uint32_t> mapping;
    // find the scope of vertex/position buffer
    uint32_t vid_min = std::numeric_limits<uint32_t>::max(); uint32_t vid_max = 0;
    for (size_t i = submesh.offset; i < submesh.offset + submesh.size; ++i) {
      vid_min = std::min(vid_min, indices_buffer[i]);
      vid_max = std::max(vid_max, indices_buffer[i]);
    }
    // load the vertices
    for (size_t i = submesh.baseVertex; i <= submesh.baseVertex + vid_max; i++) {
      size_t pi = i * 3;
      Math::vec3 pos = {position_buffer[pi], position_buffer[pi + 1], position_buffer[pi + 2]};
      auto iter = map.find(pos);
      if (iter == map.end()) {
        mapping.push_back(map.size());
        map[pos] = map.size();
        Vertex vertex = {Math::dvec3{pos}};
        quadmesh.vertices.push_back(vertex);
      } else {
        mapping.push_back(iter->second);
      }
    }

    for (size_t i = submesh.offset; i < submesh.offset + submesh.size; i += 3) {
      Triangle t;
      t.v[0] = (int)(vertex_offset + mapping[indices_buffer[i + 0]]);
      t.v[1] = (int)(vertex_offset + mapping[indices_buffer[i + 1]]);
      t.v[2] = (int)(vertex_offset + mapping[indices_buffer[i + 2]]);
      //t.uvs[0] = {vertex_buffer[indices_buffer[i + 0] * 11 + 9],
      //            vertex_buffer[indices_buffer[i + 0] * 11 + 10]};
      //t.uvs[1] = {vertex_buffer[indices_buffer[i + 1] * 11 + 9],
      //            vertex_buffer[indices_buffer[i + 1] * 11 + 10]};
      //t.uvs[2] = {vertex_buffer[indices_buffer[i + 2] * 11 + 9],
      //            vertex_buffer[indices_buffer[i + 2] * 11 + 10]};

      // add skinning information
      if (quadmesh.use_skinning) {
        t.joints[0] = joints_buffer[submesh.baseVertex + indices_buffer[i + 0]];
        t.joints[1] = joints_buffer[submesh.baseVertex + indices_buffer[i + 1]];
        t.joints[2] = joints_buffer[submesh.baseVertex + indices_buffer[i + 2]];
        t.weights[0] = weights_buffer[submesh.baseVertex + indices_buffer[i + 0]];
        t.weights[1] = weights_buffer[submesh.baseVertex + indices_buffer[i + 1]];
        t.weights[2] = weights_buffer[submesh.baseVertex + indices_buffer[i + 2]];
      }

      t.attr = 0;
      quadmesh.triangles.push_back(t);
    }
    vertex_offset += map.size();
  }
  return quadmesh;
}

GFX::Mesh to_gfx_mesh(QuadricSimplify::Mesh const& mesh) {
  GFX::Mesh gfxmesh;
  std::vector<uint32_t> index_buffer;
  std::vector<float> vertex_buffer;
  std::vector<float> position_buffer;

  std::vector<Math::Vector4<uint64_t>> joints_buffer;
  std::vector<Math::vec4> weights_buffer;

  // stacking vertex positions
  loopi(0, mesh.vertices.size()) {
    position_buffer.push_back(mesh.vertices[i].p.x);
    position_buffer.push_back(mesh.vertices[i].p.y);
    position_buffer.push_back(mesh.vertices[i].p.z);

    vertex_buffer.push_back(mesh.vertices[i].p.x);
    vertex_buffer.push_back(mesh.vertices[i].p.y);
    vertex_buffer.push_back(mesh.vertices[i].p.z);
    for (int i = 0; i < 8; ++i) {
      vertex_buffer.push_back(0.f);
    }

    if (mesh.use_skinning) {
      joints_buffer.push_back({});
      weights_buffer.push_back({});
    }
  }
  // stacking indices
  loopi(0, mesh.triangles.size()) if (!mesh.triangles[i].deleted) {
    index_buffer.push_back(mesh.triangles[i].v[0]);
    index_buffer.push_back(mesh.triangles[i].v[1]);
    index_buffer.push_back(mesh.triangles[i].v[2]);

    vertex_buffer[mesh.triangles[i].v[0] * 11 + 9] = mesh.triangles[i].uvs[0].x;
    vertex_buffer[mesh.triangles[i].v[0] * 11 + 10] = mesh.triangles[i].uvs[0].y;
    vertex_buffer[mesh.triangles[i].v[1] * 11 + 9] = mesh.triangles[i].uvs[1].x;
    vertex_buffer[mesh.triangles[i].v[1] * 11 + 10] = mesh.triangles[i].uvs[1].y;
    vertex_buffer[mesh.triangles[i].v[2] * 11 + 9] = mesh.triangles[i].uvs[2].x;
    vertex_buffer[mesh.triangles[i].v[2] * 11 + 10] = mesh.triangles[i].uvs[2].y;

    joints_buffer[mesh.triangles[i].v[0]] = mesh.triangles[i].joints[0];
    joints_buffer[mesh.triangles[i].v[1]] = mesh.triangles[i].joints[1];
    joints_buffer[mesh.triangles[i].v[2]] = mesh.triangles[i].joints[2];

    weights_buffer[mesh.triangles[i].v[0]] = mesh.triangles[i].weights[0];
    weights_buffer[mesh.triangles[i].v[1]] = mesh.triangles[i].weights[1];
    weights_buffer[mesh.triangles[i].v[2]] = mesh.triangles[i].weights[2];
  }

  gfxmesh.vertexBuffer_host = Core::Buffer(vertex_buffer.size()*sizeof(float));
  gfxmesh.positionBuffer_host = Core::Buffer(position_buffer.size()*sizeof(float));
  gfxmesh.indexBuffer_host = Core::Buffer(index_buffer.size()*sizeof(uint32_t));
  memcpy(gfxmesh.vertexBuffer_host.data, vertex_buffer.data(), gfxmesh.vertexBuffer_host.size);
  memcpy(gfxmesh.positionBuffer_host.data, position_buffer.data(), gfxmesh.positionBuffer_host.size);
  memcpy(gfxmesh.indexBuffer_host.data, index_buffer.data(), gfxmesh.indexBuffer_host.size);

  if (mesh.use_skinning) {
    gfxmesh.jointIndexBuffer_host = Core::Buffer(joints_buffer.size()*sizeof(Math::Vector4<uint64_t>));
    gfxmesh.jointWeightBuffer_host = Core::Buffer(weights_buffer.size()*sizeof(Math::vec4));
    memcpy(gfxmesh.jointIndexBuffer_host.data, joints_buffer.data(), gfxmesh.jointIndexBuffer_host.size);
    memcpy(gfxmesh.jointWeightBuffer_host.data, weights_buffer.data(), gfxmesh.jointWeightBuffer_host.size);

    gfxmesh.jointIndexBufferInfo = {(uint32_t)gfxmesh.jointIndexBuffer_host.size, true, false};
    gfxmesh.jointWeightBufferInfo = {(uint32_t)gfxmesh.jointWeightBuffer_host.size, true, false};
  }

  gfxmesh.vertexBufferInfo = {(uint32_t)gfxmesh.vertexBuffer_host.size, true, false};
  gfxmesh.positionBufferInfo = {(uint32_t)gfxmesh.positionBuffer_host.size, true, false};
  gfxmesh.indexBufferInfo = {(uint32_t)gfxmesh.indexBuffer_host.size, true, false};

  gfxmesh.submeshes.push_back(GFX::Mesh::Submesh{0, uint32_t(index_buffer.size()), 0, 0});
  return gfxmesh;
}

// Option : Load OBJ
QuadricSimplify::Mesh load_obj(const char *filename) {
  QuadricSimplify::Mesh mesh;
  mesh.vertices.clear();
  mesh.triangles.clear();
  // printf ( "Loading Objects %s ... \n",filename);
  FILE *fn;
  if (filename == NULL) return mesh;
  if ((char)filename[0] == 0) return mesh;
  if ((fn = fopen(filename, "rb")) == NULL) {
    printf("File %s not found!\n", filename);
    return mesh;
  }
  char line[1000];
  memset(line, 0, 1000);
  int vertex_cnt = 0;
  int material = -1;
  std::map<std::string, int> material_map;
  std::vector<Math::dvec3> uvs;
  std::vector<std::vector<int> > uvMap;

  while (fgets(line, 1000, fn) != NULL) {
    Vertex v;
    Math::dvec3 uv;

    if (line[0] == 'v' && line[1] == 't') {
      if (line[2] == ' ')
        if (sscanf(line, "vt %lf %lf", &uv.x, &uv.y) == 2) {
          uv.z = 0;
          uvs.push_back(uv);
        } else if (sscanf(line, "vt %lf %lf %lf", &uv.x, &uv.y, &uv.z) == 3) {
          uvs.push_back(uv);
        }
    } else if (line[0] == 'v') {
      if (line[1] == ' ')
        if (sscanf(line, "v %lf %lf %lf", &v.p.x, &v.p.y, &v.p.z) == 3) {
          mesh.vertices.push_back(v);
        }
    }
    int integers[9];
    if (line[0] == 'f') {
      Triangle t;
      bool tri_ok = false;
      bool has_uv = false;

      if (sscanf(line, "f %d %d %d", &integers[0], &integers[1],
                 &integers[2]) == 3) {
        tri_ok = true;
      } else if (sscanf(line, "f %d// %d// %d//", &integers[0], &integers[1],
                        &integers[2]) == 3) {
        tri_ok = true;
      } else if (sscanf(line, "f %d//%d %d//%d %d//%d", &integers[0],
                        &integers[3], &integers[1], &integers[4], &integers[2],
                        &integers[5]) == 6) {
        tri_ok = true;
      } else if (sscanf(line, "f %d/%d/%d %d/%d/%d %d/%d/%d", &integers[0],
                        &integers[6], &integers[3], &integers[1], &integers[7],
                        &integers[4], &integers[2], &integers[8],
                        &integers[5]) == 9) {
        tri_ok = true;
        has_uv = true;
      } else  // Add Support for v/vt only meshes
        if (sscanf(line, "f %d/%d %d/%d %d/%d", &integers[0], &integers[6],
                   &integers[1], &integers[7], &integers[2],
                   &integers[8]) == 6) {
          tri_ok = true;
          has_uv = true;
        } else {
          printf("unrecognized sequence\n");
          printf("%s\n", line);
          while (1)
            ;
        }
      if (tri_ok) {
        t.v[0] = integers[0] - 1 - vertex_cnt;
        t.v[1] = integers[1] - 1 - vertex_cnt;
        t.v[2] = integers[2] - 1 - vertex_cnt;
        t.attr = 0;
        t.material = material;
        mesh.triangles.push_back(t);
      }
    }
  }

  fclose(fn);
  return mesh;
}

void write_obj(QuadricSimplify::Mesh const &mesh, const char *filename) {
  FILE *file = fopen(filename, "w");
  int cur_material = -1;
  if (!file) {
    printf("write_obj: can't write data file \"%s\".\n", filename);
    exit(0);
  }
  loopi(0, mesh.vertices.size()) {
    fprintf(file, "v %g %g %g\n", mesh.vertices[i].p.x, mesh.vertices[i].p.y,
            mesh.vertices[i].p.z);  // more compact: remove trailing zeros
  }
  loopi(0, mesh.triangles.size()) if (!mesh.triangles[i].deleted) {
    fprintf(file, "f %d %d %d\n", mesh.triangles[i].v[0] + 1,
            mesh.triangles[i].v[1] + 1, mesh.triangles[i].v[2] + 1);
  }
  fclose(file);
}
}// namespace QuadricSimplify

namespace SIByL::MeSh {
auto MeshSimplifier::quadric_simplify(GFX::Mesh &mesh, int target_count,
  float target_ratio, double agressiveness) noexcept -> GFX::Mesh {
  QuadricSimplify::Mesh quadmesh = QuadricSimplify::to_quadric_mesh(mesh);
  QuadricSimplify::write_obj(quadmesh, "what.obj");
  target_ratio = std::clamp(target_ratio, 0.f, 1.f);
  if (target_count <= 0) target_count = std::numeric_limits<int>::max();
  target_count = std::min(target_count, int(target_ratio * quadmesh.triangles.size()));
  target_count = std::max(target_count, 1);
  QuadricSimplify::simplify_mesh(quadmesh, target_count, agressiveness, false);
  QuadricSimplify::write_obj(quadmesh, "what_simp.obj");
  GFX::Mesh gfxmesh = QuadricSimplify::to_gfx_mesh(quadmesh);
  gfxmesh.primitiveState = mesh.primitiveState;
  gfxmesh.vertexBufferLayout = mesh.vertexBufferLayout;
  if (mesh.jointInvMatBuffer_host.size > 0) {
    gfxmesh.jointWeightBuffer_host = Core::Buffer(mesh.jointInvMatBuffer_host.size);
    memcpy(gfxmesh.jointWeightBuffer_host.data,
           mesh.jointInvMatBuffer_host.data, mesh.jointInvMatBuffer_host.size);
  }
  gfxmesh.serialize();
  return gfxmesh;
}
}