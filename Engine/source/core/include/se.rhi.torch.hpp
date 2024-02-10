#pragma once
#include <torch/extension.h>
#include <se.rhi.hpp>
#include <se.core.hpp>

namespace se::rhi {
auto SIByL_API toTensor(se::rhi::CUDABuffer* cudaBuffer, std::vector<int64_t> const& dimension) noexcept -> torch::Tensor;
}