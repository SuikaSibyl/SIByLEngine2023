#include "../Public/SE.MeSh.MeshCluster.hpp"
#include <algorithm>
#include <Print/SE.Core.Log.hpp>

namespace SIByL::MeSh {
auto FaceTree::cluster_is_root(uint32_t i) const noexcept -> bool {
  return clusters[i].parent == NilID;
}
auto FaceTree::cluster_is_leaf(uint32_t i) const const noexcept -> bool {
  return clusters[i].child[0] == NilID;
}
auto FaceTree::add_cluster() noexcept -> uint32_t {
  uint32_t id = clusters.size();
  clusters.emplace_back(FaceCluster{});
  cluster_marks.emplace_back(0x0);
  cluster_flags.emplace_back(0x0);
  return id;
}
auto FaceTree::merge_clusters(uint32_t x, uint32_t y) noexcept -> uint32_t {
  // Make sure we're dealing with top-level clusters
  x = find_root_cluster(x);
  y = find_root_cluster(y);
  SE_ASSERT(x != y, "SE::MeSh::MeshCluster::FaceTree::merge_cluster receive similar clusters.");
  // Allocate and link a new cluster
  uint32_t z = add_cluster();
  FaceCluster& current = cluster(z);
  FaceCluster& left    = cluster(x);
  FaceCluster& right   = cluster(y);
  current.child[0] = x;
  current.child[1] = y;
  left.parent = right.parent = z;

  compute_face_list(z);
  // Set up avg & total normals
  current.clear_normal();
  current.add_normal(left.total_normal());
  current.add_normal(right.total_normal());
  current.finalize_normal();
  // !!BUG: Set up the frame?
  return z;
}
auto FaceTree::find_root_cluster(uint32_t id) noexcept -> uint32_t {
  while (!cluster_is_root(id))
      id = clusters[id].parent;
  return id;
}
auto FaceTree::compute_face_list(uint32_t id) noexcept -> bool {
  FaceCluster &current = cluster(id), &left = cluster(current.child[0]),
              &right = cluster(current.child[1]);
  current.nfaces = left.nfaces + right.nfaces;
  if (right.first_face == left.first_face + left.nfaces)
    current.first_face = left.first_face;
  else if (left.first_face == right.first_face + right.nfaces)
    current.first_face = right.first_face;
  else {
    current.first_face = NilID;
    return false;
  }
  return true;
}

auto DualMesh::contract(DualContraction const& conx) noexcept -> void {
  
}

auto MeshClusterFacotry::init(StdMesh* mesh) noexcept -> void {
  target_edges.resize(dual.edge_count());

  
  // Construct the initial heap
  for (uint32_t i = 0; i < dual.edge_count(); ++i) {
    DualHierarchEdge& t = target_edges[i];
    t = i;  // set index correctly
    compute_edge_info(t);
  }
}

auto MeshClusterFacotry::aggregate(uint32_t target) noexcept -> bool {
  while (root_cluster_count > target) {

  }
  return true;
}

auto MeshClusterFacotry::contract(DualHierarchEdge const& e) noexcept -> void {
  DualEdge& edge = dual.edges[e];
  if (edge.v1 == NilID && edge.v2 == NilID) return; // not a feasible edge

  DualHierarchNode& n1 = node_info(edge.v1);
  DualHierarchNode& n2 = node_info(edge.v2);
  uint32_t i;

  n1.qdir += n2.qdir;
  n1.qfit += n2.qfit;
  n1.nverts += n2.nverts;
  n1.perimeter += n2.perimeter - 2 * edge.border_length;

  // ??TODO: Remove duplicate quadrics from Q_fit

  // update face list
  n1.faces.insert(n1.faces.end(), n2.faces.begin(), n2.faces.end());

  DualContraction conx;
  conx.n1 = edge.v1;
  conx.n2 = edge.v2;
  dual.contract(conx);
  uint32_t new_cluster = tree->merge_clusters(conx.n1, conx.n2);
  --root_cluster_count;

  if (!tree->cluster(new_cluster).compute_frame(n1.qfit, n1.nverts)) {
    //!!BUG: Again, we need a fallback policy for finding a frame
    Core::LogManager::Error("MeSh::BUG -- Can't proceed without valid frame.");
  }

  if (will_maintain_bounds)
      update_node_bounds(conx.n1);

  //// remove dead edges from heap
  //for (i = 0; i < conx.dead_edges.size(); i++)
  //  heap.remove(get_edge(conx.dead_edges[i]));

  //// recompute edge info for all edges linked to n1
  //for (i = 0; i < dual->node_edges(conx.n1).length(); i++)
  //  compute_edge_info(get_edge(dual->node_edges(conx.n1)[i]));
}

auto MeshClusterFacotry::update_node_bounds(uint32_t id) noexcept -> void {
  DualHierarchNode& n = node_info(id);
  FaceCluster& c = tree->cluster(tree->find_root_cluster(id));

  c.reset_bounds();
  update_frame_bounds(c, n.faces);
}

auto MeshClusterFacotry::compute_edge_info(DualHierarchEdge e) noexcept
    -> bool {
  DualEdge& edge = dual.edge(e);
  DualHierarchNode& n1 = node_info(edge.v1);
  DualHierarchNode& n2 = node_info(edge.v2);
  //MxFaceCluster& c1 = tree->cluster(tree->find_root_cluster(edge.v1));
  //MxFaceCluster& c2 = tree->cluster(tree->find_root_cluster(edge.v2));



  return true;
}
}