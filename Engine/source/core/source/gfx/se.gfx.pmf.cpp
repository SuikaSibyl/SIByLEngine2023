#define DLIB_EXPORT
#include <se.gfx.hpp>
#undef DLIB_EXPORT

namespace se::gfx {
PMFDataPack PMFConstructor::datapack;

PMFDataPack::PMFDataPack() {
  buffer.usages =
	(uint32_t)rhi::BufferUsageBit::COPY_DST |
	(uint32_t)rhi::BufferUsageBit::STORAGE;
}

PMFConstructor::PiecewiseConstant1D 
PMFConstructor::build_piecewise_constant_1d(
  std::span<float> f, float min, float max) {
	// take absolute value of f
	for (float& val : f) val = std::abs(val);
	// Compute integral of step function at xi
	size_t n = f.size();
	const int current_offset = datapack.buffer.host.size() / 4;
	datapack.buffer.host.resize((current_offset + n + 1) * 4);
	std::span<float> cdf{ (float*)&datapack.buffer.host[current_offset * 4], (n + 1) };
	cdf[0] = 0;
	for (size_t i = 1; i < n + 1; ++i)
		cdf[i] = cdf[i - 1] + f[i - 1] * (max - min) / n;
	// Transform step function integral into CDF
	float funcInt = cdf[n];
	if (funcInt == 0) for (size_t i = 1; i < n + 1; ++i)
		cdf[i] = float(i) / float(n);
	else for (size_t i = 1; i < n + 1; ++i)
		cdf[i] /= funcInt;
	PiecewiseConstant1D constant_1d;
	constant_1d.min = min;
	constant_1d.max = max;
	constant_1d.offset = current_offset;
	constant_1d.size = n + 1;
	constant_1d.func_int = funcInt;
	datapack.buffer.host_stamp++;
	return constant_1d;
}

float PMFConstructor::PiecewiseConstant1D::sample(float u, float& pdf, int& offset) {
  std::span<float> cdf{ (float*)&datapack.buffer.host[this->offset * 4], size };
  // Find surrounding CDF segments and offset
  int size = (int)(this->size) - 2;
  int first = 1;
  while (size > 0) {
	// Evaluate predicate at midpoint and update first and size
	int half = (uint32_t)size >> 1;
	int middle = first + half;
	bool predResult = cdf[middle] <= u;
	first = predResult ? middle + 1 : first;
	size = predResult ? size - (half + 1) : half;
  }
  int o = (uint32_t)std::clamp((int)first - 1, 0, (int)(this->size) - 2);
  offset = o;
  // Compute offset along CDF segment
  float du = u - cdf[o];
  if (cdf[o + 1] - cdf[o] > 0)
	  du /= cdf[o + 1] - cdf[o];
  // Compute PDF for sampled offset
  pdf = cdf[o + 1] - cdf[o];
  return std::lerp(0, 1, (o + du) / (this->size - 1));
}

PMFConstructor::PiecewiseConstant2D 
PMFConstructor::build_piecewise_constant_2d(
  std::span<float> f, int nu, int nv,
  vec2 min, vec2 max) {
  std::vector<PMFConstructor::PiecewiseConstant1D> conditionalV;
  for (int v = 0; v < nv; ++v) {
	// Compute conditional sampling distribution for v
	conditionalV.emplace_back(build_piecewise_constant_1d(
		f.subspan(v * nu, nu), min.x, max.x));
  }
  // Compute marginal sampling distribution 
  std::vector<float> marginalFunc;
  for (int v = 0; v < nv; ++v)
	marginalFunc.push_back(conditionalV[v].func_int);
  PMFConstructor::PiecewiseConstant1D marginal =
	build_piecewise_constant_1d(marginalFunc, min.x, max.x);
  PiecewiseConstant2D constant_2d;
  constant_2d.condition_offset = conditionalV[0].offset;
  constant_2d.condition_size = conditionalV[0].size;
  constant_2d.marginal_offset = marginal.offset;
  constant_2d.marginal_size = marginal.size;
  constant_2d.min = min;
  constant_2d.max = max;
  constant_2d.func_int = marginal.func_int;
  datapack.buffer.host_stamp++;
  return constant_2d;
}

auto PMFConstructor::upload_datapack() noexcept -> void {
  datapack.buffer.hostToDevice();
}

auto PMFConstructor::clear_datapack() noexcept -> void {
  datapack.buffer.buffer = nullptr;
  datapack.buffer.previous = nullptr;
}

auto PMFConstructor::binding_resource_buffer() noexcept -> rhi::BindingResource {
  return rhi::BindingResource{ {datapack.buffer.buffer.get(), 0, datapack.buffer.buffer->size()} };
}
}