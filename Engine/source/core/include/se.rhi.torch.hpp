#pragma once
#include <torch/extension.h>
#include <se.rhi.hpp>
#include <se.core.hpp>

namespace se::rhi {
enum struct DataType {
  Float16,
  Float32,
  Float64,
  UINT8,
  INT8,
  INT16,
  INT32,
  INT64,
};

auto SIByL_API toTensor(se::rhi::CUDABuffer* cudaBuffer, std::vector<int64_t> const& dimension, DataType type = DataType::Float32) noexcept -> torch::Tensor;
}