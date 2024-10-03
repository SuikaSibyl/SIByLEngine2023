#define DLIB_EXPORT
#include <se.gfx.hpp>
#undef DLIB_EXPORT

namespace se::gfx {
struct DirectionCone {
  vec3 w;
  float cosTheta;
  DirectionCone() { w = vec3(0, 0, 0); cosTheta = std::numeric_limits<float>::infinity(); }
  DirectionCone(vec3 w, float cosTheta) { this->w = w; this->cosTheta = cosTheta; }
  auto is_empty() -> bool { return cosTheta == std::numeric_limits<float>::infinity(); }
  static DirectionCone entire_sphere() { return DirectionCone(vec3(0, 0, 1), -1); }
  static DirectionCone bound_subtended_directions(bounds3 b, vec3 p) {
    // Compute bounding sphere for b and check if p is inside
    float radius; point3 pCenter;
    b.boundingSphere(&pCenter, &radius);
    if (distanceSquared(p, pCenter) < sqr(radius))
      return DirectionCone::entire_sphere();
    // Compute and return DirectionCone for bounding sphere
    vec3 w = normalize(pCenter - p);
    float sin2ThetaMax = sqr(radius) / distanceSquared(pCenter, p);
    float cosThetaMax = safe_sqrt(1 - sin2ThetaMax);
    return DirectionCone(w, cosThetaMax);
  }
};

struct LightBounds {
	bounds3 bounds;
	float phi = 0;
	vec3 w;
	float cosTheta_o;
  vec3 rgb;
	float cosTheta_e;
	bool twoSided;

  static float cosSubClamped(
    float sinTheta_a, float cosTheta_a,
    float sinTheta_b, float cosTheta_b) {
    if (cosTheta_a > cosTheta_b) return 1;
    return cosTheta_a * cosTheta_b + sinTheta_a * sinTheta_b;
  };

  static float sinSubClamped(
    float sinTheta_a, float cosTheta_a,
    float sinTheta_b, float cosTheta_b) {
    if (cosTheta_a > cosTheta_b) return 0;
    return sinTheta_a * cosTheta_b - cosTheta_a * sinTheta_b;
  };

  auto centroid() const -> vec3 const { return (bounds.pMax + bounds.pMin) / 2; }
};

DirectionCone Union (DirectionCone a, DirectionCone b) {
  // Handle the cases where one or both cones are empty
  if (a.is_empty()) return b;
  if (b.is_empty()) return a;
  // Handle the cases where one cone is inside the other
  float theta_a = safe_acos(a.cosTheta);
  float theta_b = safe_acos(b.cosTheta);
  float theta_d = angle_between(a.w, b.w);
  if (std::min(theta_d + theta_b, k_pi) <= theta_a) return a;
  if (std::min(theta_d + theta_a, k_pi) <= theta_b) return b;
  // Compute the spread angle of the merged cone, theta_o
  float theta_o = (theta_a + theta_d + theta_b) / 2;
  if (theta_o >= k_pi)
    return DirectionCone::entire_sphere();
  // Find the merged cone¡¯s axis and return cone union
  float theta_r = theta_o - theta_a;
  vec3 wr = cross(a.w, b.w);
  if (wr.lengthSquared() == 0)
    return DirectionCone::entire_sphere();
  vec3 w = rotate(degrees(theta_r), wr) * (a.w);
  return DirectionCone(w, std::cos(theta_o));
}

LightBounds Union (LightBounds a, LightBounds b) {
  // If one LightBounds has zero power, return the other
  if (a.phi == 0) return b;
  if (b.phi == 0) return a;
  // Find average direction and updated angles for LightBounds
  DirectionCone cone = Union(DirectionCone(a.w, a.cosTheta_o),
    DirectionCone(b.w, b.cosTheta_o));
  float cosTheta_o = cone.cosTheta;
  float cosTheta_e = std::min(a.cosTheta_e, b.cosTheta_e);
  // Return final LightBounds union
  LightBounds lb;
  lb.bounds = unionBounds(a.bounds, b.bounds);
  lb.w = cone.w;
  lb.phi = a.phi + b.phi;
  lb.rgb = a.rgb + b.rgb;
  lb.cosTheta_o = cosTheta_o;
  lb.cosTheta_e = cosTheta_e;
  lb.twoSided = a.twoSided || b.twoSided;
  return lb;
}

inline vec2 octWrap(vec2 const& v) {
  return (vec2(1.f) - abs(vec2(v.y, v.x))) *
    select(vec2(v.x, v.y) >= vec2(0), vec2(1), vec2(-1));
}

inline vec2 UnitVectorToSignedOctahedron(vec3 normal) {
  // Project the sphere onto the octahedron (|x|+|y|+|z| = 1)
  normal.xy /= dot(vec3(1), abs(normal));
  // Then project the octahedron onto the xy-plane
  return (normal.z < 0) ? octWrap(normal.xy) : normal.xy;
}

inline uint32_t UnitVectorToUnorm32Octahedron(vec3 normal) {
  vec2 p = UnitVectorToSignedOctahedron(normal);
  p = clamp_vec2(vec2(p.x, p.y) * 0.5f + vec2(0.5f), vec2(0.f), vec2(1.f));
  return uint32_t(p.x * 0xfffe) | (uint32_t(p.y * 0xfffe) << 16);
}

inline vec3 SignedOctahedronToUnitVector(vec2 oct) {
  vec3 normal = vec3(oct, 1 - dot(vec2(1), abs(oct)));
  const float t = std::max(-normal.z, 0.f);
  normal.xy += select(normal.xy >= vec2(0), vec2(-t), vec2(t));
  return normalize(normal);
}

inline vec3 Unorm32OctahedronToUnitVector(uint32_t pUnorm) {
  vec2 p;
  p.x = std::clamp(float(pUnorm & 0xffff) / 0xfffe, 0.f, 1.f);
  p.y = std::clamp(float(pUnorm >> 16) / 0xfffe, 0.f, 1.f);
  p = p * 2.0 - vec2(1.0);
  return SignedOctahedronToUnitVector(p);
}

struct CompactLightBounds {
  float phi;
  uint32_t w;
  uint32_t bitfield;
  uint32_t qb_0;
  uint32_t qb_1;
  uint32_t qb_2;
  half coloru;
  half colorv;

  static uint32_t quantize_cos(float c) {
    return uint32_t(std::floor(32767.f * ((c + 1) / 2)));
  }

  // remaps a coordinate value c between min and max to the range
  // [0, 2^16 - 1] range of values that an unsigned 16-bit integer can store.
  static float quantize_bounds(float c, float min, float max) {
    if (min == max) return 0;
    return 65535.f * clamp((c - min) / (max - min), 0.f, 1.f);
  }

  CompactLightBounds() = default;
  CompactLightBounds(LightBounds const& lb, bounds3 const& allb) {
    uint32_t qCosTheta_o = quantize_cos(lb.cosTheta_o);
    uint32_t qCosTheta_e = quantize_cos(lb.cosTheta_e);
    uint32_t twoSided = lb.twoSided ? 1 : 0;


    bitfield = (qCosTheta_o << 17) | (qCosTheta_e << 2) | twoSided;
    phi = lb.phi;
    w = UnitVectorToUnorm32Octahedron(normalize(lb.w));
    // Quantize bounding box into qb
    uint32_t qb[3];
    for (int c = 0; c < 3; ++c) {
      uint32_t qb_lc = uint32_t(std::floor(quantize_bounds(
        lb.bounds.pMin[c], allb.pMin[c], allb.pMax[c])));
      uint32_t qb_rc = uint32_t(std::ceil(quantize_bounds(
        lb.bounds.pMax[c], allb.pMin[c], allb.pMax[c])));
      qb[c] = (qb_lc << 16) | qb_rc;
    }
    coloru = half(lb.rgb.g);
    colorv = half(lb.rgb.b);
    qb_0 = qb[0];
    qb_1 = qb[1];
    qb_2 = qb[2];
  }
};

struct SampledLight {

};

struct LightSampleContext {
  point3 p;
  normal3 ns;
};

struct LightBVHNode {
  CompactLightBounds cb;
  struct {
		unsigned int childOrLightIndex : 31;
		unsigned int isLeaf : 1;
	};

	static auto makeLeaf(unsigned int lightIndex,
		const CompactLightBounds& cb) -> LightBVHNode;
	static auto makeInterior(unsigned int child1Index,
		const CompactLightBounds& cb) -> LightBVHNode;


  bounds3 bounds(bounds3 allb) const {
    bounds3 b;
    b.pMin = vec3(
      lerp((cb.qb_0 >> 16) / 65535.f, allb.pMin.x, allb.pMax.x),
      lerp((cb.qb_1 >> 16) / 65535.f, allb.pMin.y, allb.pMax.y),
      lerp((cb.qb_2 >> 16) / 65535.f, allb.pMin.z, allb.pMax.z));
    b.pMax = vec3(
      lerp((cb.qb_0 & 0xffff) / 65535.f, allb.pMin.x, allb.pMax.x),
      lerp((cb.qb_1 & 0xffff) / 65535.f, allb.pMin.y, allb.pMax.y),
      lerp((cb.qb_2 & 0xffff) / 65535.f, allb.pMin.z, allb.pMax.z));
    return b;
  }

  float cos_theta_o() const  { return 2 * ((cb.bitfield >> 17) / 32767.f) - 1; }
  float cos_theta_e() const { return 2 * (((cb.bitfield >> 2) & 0x7FFF) / 32767.f) - 1; }
  bool two_sided() const { return (cb.bitfield & 1) != 0; }

  float importance(vec3 p, vec3 n, bounds3 allb) const {
    bounds3 b = bounds(allb);
    float cosTheta_o = cos_theta_o();
    float cosTheta_e = cos_theta_e();
    // Return importance for light bounds at reference point
    // Compute clamped squared distance to reference point
    vec3 pc = (b.pMin + b.pMax) / 2;
    float d2 = distanceSquared(p, pc);
    d2 = std::max(d2, length(b.diagonal()) / 2);
    // Compute sine and cosine of angle to vector w, theta_w
    vec3 wi = normalize(p - pc);
    vec3 w = Unorm32OctahedronToUnitVector(cb.w);
    float cosTheta_w = dot(w, wi);
    if (two_sided()) cosTheta_w = std::abs(cosTheta_w);
    float sinTheta_w = safe_sqrt(1 - sqr(cosTheta_w));
    // Compute cos theta_b for reference point
    float cosTheta_b = DirectionCone::bound_subtended_directions(b, p).cosTheta;
    float sinTheta_b = safe_sqrt(1 - sqr(cosTheta_b));
    // Compute cos theta' and test against cos theta_e
    float sinTheta_o = safe_sqrt(1 - sqr(cosTheta_o));
    float cosTheta_x = LightBounds::cosSubClamped(
      sinTheta_w, cosTheta_w, sinTheta_o, cosTheta_o);
    float sinTheta_x = LightBounds::sinSubClamped(
      sinTheta_w, cosTheta_w, sinTheta_o, cosTheta_o);
    float cosThetap = LightBounds::cosSubClamped(
      sinTheta_x, cosTheta_x, sinTheta_b, cosTheta_b);
    if (cosThetap <= cosTheta_e) return 0;
    // Return final importance at reference point
    float importance = cb.phi * cosThetap / d2;
    // Account for cos theta_i in importance at surfaces
    if (n.x != 0 || n.y != 0 || n.z != 0) {
      float cosTheta_i = std::abs(dot(wi, n));
      float sinTheta_i = safe_sqrt(1 - sqr(cosTheta_i));
      float cosThetap_i = LightBounds::cosSubClamped(
        sinTheta_i, cosTheta_i, sinTheta_b, cosTheta_b);
      importance *= cosThetap_i;
    }
    return importance;
  }
};

auto LightBVHNode::makeLeaf(unsigned int lightIndex,
  const CompactLightBounds& cb) -> LightBVHNode {
  return LightBVHNode{ cb, {lightIndex, 1} };
}

auto LightBVHNode::makeInterior(unsigned int child1Index,
  const CompactLightBounds& cb) -> LightBVHNode {
  return LightBVHNode{ cb, {child1Index, 0} };
}

struct BVHLightSampler {
	std::vector<Light> lights;
	std::vector<Light> infiniteLights;
	bounds3 allLightBounds;
	std::vector<LightBVHNode> nodes;
  std::vector<uint32_t> lightToBitTrail;

	BVHLightSampler(std::span<Light::LightPacket> lights);
	auto buildBVH(std::vector<std::pair<int, LightBounds>>& bvhLights,
		int start, int end, uint32_t bitTrail, int depth
	) noexcept -> std::pair<int, LightBounds>;
	auto evaluateCost(const LightBounds& b, 
		const bounds3& bounds, int dim) const -> float;

  auto sample(const LightSampleContext& ctx, float u) const ->std::optional<SampledLight>;
  auto pmf(const LightSampleContext& ctx, uint32_t lightID) const -> float;
};

BVHLightSampler::BVHLightSampler(std::span<Light::LightPacket> lights) {
  std::vector<std::pair<int, LightBounds>> bvhLights;
  for (size_t i = 0; i < lights.size(); ++i) {
    // Store th light in either infiniteLights or bvhLights
    Light::LightPacket light = lights[i];
    // handle light bounds
    LightBounds lb;
    lb.bounds = { light.floatvec_1.xyz(), light.floatvec_2.xyz() };
    lb.phi = light.floatvec_0.x;
    lb.w = { light.floatvec_0.w, light.floatvec_1.w, light.floatvec_2.w };
    lb.cosTheta_o = 1;
    lb.rgb = light.floatvec_0.xyz();
    lb.cosTheta_e = std::cos(k_pi / 2);
    lb.twoSided = false;

    //if (!lightBounds)
    //  infiniteLights.push_back(light);
    if (lb.phi > 0) {
      bvhLights.push_back(std::make_pair(i, lb));
      allLightBounds = unionBounds(allLightBounds, lb.bounds);
    }
  }

  lightToBitTrail.resize(lights.size());
  if (!bvhLights.empty())
    buildBVH(bvhLights, 0, bvhLights.size(), 0, 0);
}

auto BVHLightSampler::buildBVH(
	std::vector<std::pair<int, LightBounds>>& bvhLights, 
	const int start, const int end,
	uint32_t bitTrail, int depth
) noexcept -> std::pair<int, LightBounds> {
	// Initialize leaf node if only a single light remains
	if (end - start == 1) {
		int nodeIndex = nodes.size();
		CompactLightBounds cb(bvhLights[start].second, allLightBounds);
		int lightIndex = bvhLights[start].first;
		nodes.push_back(LightBVHNode::makeLeaf(lightIndex, cb));
		lightToBitTrail[lightIndex] = bitTrail;
		return { nodeIndex, bvhLights[start].second };
  }

  // Choose split dimension and position using modified SAH
  // Compute bounds and centroid bounds for lights
  bounds3 bounds, centroidBounds;
  for (int i = start; i < end; ++i) {
    const LightBounds& lb = bvhLights[i].second;
    bounds = unionBounds(bounds, lb.bounds);
    centroidBounds = unionPoint(centroidBounds, point3(lb.centroid()));
  }

  float minCost = std::numeric_limits<float>::max();
  int minCostSplitBucket = -1, minCostSplitDim = -1;
  constexpr int nBuckets = 12;
  for (int dim = 0; dim < 3; ++dim) {
    // Compute minimum cost bucket for splitting along dimension _dim_
    if (centroidBounds.pMax[dim] == centroidBounds.pMin[dim])
      continue;
    // Compute _LightBounds_ for each bucket
    LightBounds bucketLightBounds[nBuckets];
    for (int i = start; i < end; ++i) {
      point3 pc = bvhLights[i].second.centroid();
      int b = nBuckets * centroidBounds.offset(pc)[dim];
      if (b == nBuckets)
        b = nBuckets - 1;
      bucketLightBounds[b] = Union(bucketLightBounds[b], bvhLights[i].second);
    }

    // Compute costs for splitting lights after each bucket
    float cost[nBuckets - 1];
    for (int i = 0; i < nBuckets - 1; ++i) {
      // Find _LightBounds_ for lights below and above bucket split
      LightBounds b0, b1;
      for (int j = 0; j <= i; ++j)
        b0 = Union(b0, bucketLightBounds[j]);
      for (int j = i + 1; j < nBuckets; ++j)
        b1 = Union(b1, bucketLightBounds[j]);

      // Compute final light split cost for bucket
      cost[i] = evaluateCost(b0, bounds, dim) + evaluateCost(b1, bounds, dim);
    }

    // Find light split that minimizes SAH metric
    for (int i = 1; i < nBuckets - 1; ++i) {
      if (cost[i] > 0 && cost[i] < minCost) {
        minCost = cost[i];
        minCostSplitBucket = i;
        minCostSplitDim = dim;
      }
    }
  }

  // Partition lights according to chosen split
  int mid;
  if (minCostSplitDim == -1)
    mid = (start + end) / 2;
  else {
    const auto* pmid = std::partition(
      &bvhLights[start], &bvhLights[end - 1] + 1,
      [=](const std::pair<int, LightBounds>& l) {
        int b = nBuckets *
        centroidBounds.offset(l.second.centroid())[minCostSplitDim];
    if (b == nBuckets)
      b = nBuckets - 1;
    return b <= minCostSplitBucket;
      });
    mid = pmid - &bvhLights[0];
    if (mid == start || mid == end)
      mid = (start + end) / 2;
  }

  // Allocate interior _LightBVHNode_ and recursively initialize children
  int nodeIndex = nodes.size();
  nodes.push_back(LightBVHNode());
  std::pair<int, LightBounds> child0 =
    buildBVH(bvhLights, start, mid, bitTrail, depth + 1);
  std::pair<int, LightBounds> child1 =
    buildBVH(bvhLights, mid, end, bitTrail | (1u << depth), depth + 1);

  // Initialize interior node and return node index and bounds
  LightBounds lb = Union(child0.second, child1.second);
  CompactLightBounds cb(lb, allLightBounds);
  nodes[nodeIndex] = LightBVHNode::makeInterior(child1.first, cb);
  return { nodeIndex, lb };
}

// evaluate the cost model for the two LightBounds for each split candidate
auto BVHLightSampler::evaluateCost(const LightBounds& b,
	const bounds3& bounds, int dim) const -> float {
	// Evaluate direction bounds measure for LightBounds
	float theta_o = std::acos(b.cosTheta_o), theta_e = std::acos(b.cosTheta_e);
	float theta_w = std::min(theta_o + theta_e, float_Pi);
	float sinTheta_o = safe_sqrt(1 - sqr(b.cosTheta_o));
	float M_omega = 2 * float_Pi * (1 - b.cosTheta_o) +
		float_Pi / 2 * (2 * theta_w * sinTheta_o - std::cos(theta_o - 2 * theta_w) -
			2 * theta_o * sinTheta_o + b.cosTheta_o);
	// Return complete cost estimate for LightBounds
	float Kr = maxComponent(bounds.diagonal()) / bounds.diagonal()[dim];
	return b.phi * M_omega * Kr * b.bounds.surfaceArea();
}

static constexpr float FloatOneMinusEpsilon = 0x1.fffffep-1;

inline auto SampleDiscrete(float weights[2], float u, float& pmf, float& uRemapped) -> int {
  return 1;
}

auto BVHLightSampler::sample(
  const LightSampleContext& ctx, float u
) const ->std::optional<SampledLight> {
  //  light BVH to sample light
  if (nodes.empty()) return {};
  // Declare common variables for light BVH traversal
  point3 p = ctx.p;
  normal3 n = ctx.ns;

  int nodeIndex = 0;
  float pmf = 1;
  
  while (true) {
    // Process light BVH node for light sampling
    LightBVHNode node = nodes[nodeIndex];
    if (!node.isLeaf) {
      // Compute light BVH child node importances
      const LightBVHNode * children[2] = { &nodes[nodeIndex + 1],
                                           &nodes[node.childOrLightIndex] };
      float ci[2] = { children[0]->importance(p, n, allLightBounds),
                      children[1]->importance(p, n, allLightBounds) };
      if (ci[0] == 0 && ci[1] == 0)
        return {};
      // Randomly sample light BVH child node
      float nodePMF;
      int child = SampleDiscrete(ci, u, nodePMF, u);
      pmf *= nodePMF;
      nodeIndex = (child == 0) ? (nodeIndex + 1) : node.childOrLightIndex;
    }
    else {
      // Confirm light has nonzero importance before returning light sample
      if (nodeIndex > 0 ||
        node.importance(p, n, allLightBounds) > 0)
        //return SampledLight{ lights[node.childOrLightIndex], pmf };
        return SampledLight{};
      return {};
    }
  }
}

float BVHLightSampler::pmf(const LightSampleContext& ctx, uint32_t lightID) const {
  // Handle infinite light PMF computation
  //if (!lightToBitTrail.HasKey(light))
  //  return 1.f / (infiniteLights.size() + (nodes.empty() ? 0 : 1));

  // Initialize local variables for BVH traversal for PMF computation
  uint32_t bitTrail = lightToBitTrail[lightID];
  point3 p = ctx.p;
  normal3 n = ctx.ns;
  //// Compute infinite light sampling probability pInfinite
  //  float pInfinite = float(infiniteLights.size()) /
  //    float(infiniteLights.size() + (nodes.empty() ? 0 : 1));

  float pmf = 1;
  int nodeIndex = 0;

  // Compute light¡¯s PMF by walking down tree nodes to the light
  while (true) {
    const LightBVHNode* node = &nodes[nodeIndex];
    if (node->isLeaf)
      return pmf;
    // Compute child importances and update PMF for current node
    const LightBVHNode* child0 = &nodes[nodeIndex + 1];
    const LightBVHNode* child1 = &nodes[node->childOrLightIndex];
    float ci[2] = { child0->importance(p, n, allLightBounds),
                    child1->importance(p, n, allLightBounds) };
    pmf *= ci[bitTrail & 1] / (ci[0] + ci[1]);
    
    // Use bitTrail to find next node index and update its value
    nodeIndex = (bitTrail & 1) ? node->childOrLightIndex : (nodeIndex + 1);
    bitTrail >>= 1;
  }
}

auto Scene::GPUScene::build_light_bvh() noexcept -> void {
  std::span<Light::LightPacket> light_span = {
    (Light::LightPacket*)light_buffer->host.data(),
    light_buffer->host.size() / sizeof(Light::LightPacket)
  };
  BVHLightSampler sampler(light_span);
  lbvh.light_bvh_buffer->host.resize(sampler.nodes.size() * sizeof(LightBVHNode));
  memcpy((LightBVHNode*)lbvh.light_bvh_buffer->host.data(), sampler.nodes.data(),
    sampler.nodes.size() * sizeof(LightBVHNode));
  lbvh.light_trail_buffer->host.resize(sampler.lightToBitTrail.size() * sizeof(uint32_t));
  memcpy((LightBVHNode*)lbvh.light_trail_buffer->host.data(), sampler.lightToBitTrail.data(),
    sampler.lightToBitTrail.size() * sizeof(uint32_t));
  lbvh.light_bvh_buffer->host_stamp++;
  lbvh.light_trail_buffer->host_stamp++;
  lbvh.all_light_bounds = sampler.allLightBounds;

  LightSampleContext ctx;
  ctx.p = vec3(0, 0, 0);
  ctx.ns = vec3(0, 1, 0);
  sampler.sample(ctx, 0.45);
}
}