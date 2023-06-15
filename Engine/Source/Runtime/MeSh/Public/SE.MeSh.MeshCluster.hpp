#pragma once
#include "SE.MeSh.StdMesh.hpp"
#include "SE.MeSh.GeoPrims.hpp"
#include "SE.MeSh.Metric.hpp"
#include "SE.MeSh.Utility.hpp"
#include <array>
#include <queue>

namespace SIByL::MeSh {
struct FaceCluster : public FitFrame {
  uint32_t parent;
  uint32_t child[2];
  uint32_t first_face, nfaces;
};

struct FaceTree {
  std::vector<FaceCluster> clusters;
  std::vector<uint16_t> cluster_marks;
  std::vector<uint16_t> cluster_flags;

  auto cluster_count() const noexcept -> size_t { return clusters.size(); }
  auto cluster(uint32_t i) noexcept -> FaceCluster& { return clusters[i]; }
  auto cluster(uint32_t i) const noexcept -> FaceCluster const& { return clusters[i]; }

  auto cluster_is_root(uint32_t i) const noexcept -> bool;
  auto cluster_is_leaf(uint32_t i) const const noexcept -> bool;
  /** Given a cluster, find its root.
  * For anything other than a complete tree, this is not unique. */
  auto find_root_cluster(uint32_t id) noexcept -> uint32_t;

  auto add_cluster() noexcept -> uint32_t;
  auto merge_clusters(uint32_t a, uint32_t b) noexcept -> uint32_t;

  auto compute_face_list(uint32_t cluster) noexcept -> bool;
};

struct DualEdge : public Edge {
  float border_length;  // edge length
  bool operator<(DualEdge const& x) const {
    return border_length < x.border_length;
  }
};

using DualEdgeTriplet = std::array<DualEdge, 3>;
using DualEdgeList = std::vector<DualEdge>;

struct DualContraction {
  uint32_t n1, n2;
  DualEdgeList dead_edges;
};

struct DualMesh {
  // All edges in the mesh
  std::vector<DualEdge> edges;
  // Edge links for each node.
  // Each node is a triangle, so has at most 3 neighbors.
  std::vector<DualEdgeTriplet> edge_links;
  uint32_t edge_count() const { return static_cast<uint32_t>(edges.size()); }
  DualEdge& edge(uint32_t i) { return edges[i]; }
  DualEdge const& edge(uint32_t i) const { return edges[i]; }

  auto contract(DualContraction const& conx) noexcept -> void;
};

using DualHierarchEdge = uint32_t;
struct DualHierarchNode{
  Quadric3 qdir;	// error form for normal orientation
  Quadric3 qfit;	// error form for planarity
  FaceList faces;	// All the faces in this cluster
  float perimeter;  // The length of the cluster's perimeter
  uint32_t nverts;	// The number of vertices in this cluster
};

struct MeshClusterFacotry {
  // init the mesh cluster constructor
  auto init(StdMesh* mesh) noexcept -> void;
  auto aggregate(uint32_t target) noexcept -> bool;

  auto compute_edge_info(DualHierarchEdge e) noexcept -> bool;

  DualHierarchNode& node_info(uint32_t i) { return face_nodes[i]; }
  const DualHierarchNode& node_info(uint32_t i) const { return face_nodes[i]; }

  DualMesh dual;	// dual mesh
 private:
  FaceTree* tree;
  uint32_t root_cluster_count;
  std::priority_queue<DualEdge> edges;
  std::vector<DualHierarchEdge> target_edges;
  std::vector<DualHierarchNode> face_nodes;

  bool will_maintain_bounds;

 auto update_node_bounds(uint32_t id) noexcept -> void;

  auto contract(DualHierarchEdge const& e) noexcept -> void;
};
}  // namespace SIByL::MeSh