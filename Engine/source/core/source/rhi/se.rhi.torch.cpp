#define DLIB_EXPORT
#include <se.rhi.torch.hpp>
#undef DLIB_EXPORT

namespace se::rhi {
c10::ScalarType toScalarType(DataType type) noexcept {
  switch (type) {
  case se::rhi::DataType::Float16:  return c10::ScalarType::Half;
  case se::rhi::DataType::Float32:  return c10::ScalarType::Float;
  case se::rhi::DataType::Float64:  return c10::ScalarType::Double;
  case se::rhi::DataType::UINT8:    return c10::ScalarType::Byte;
  case se::rhi::DataType::INT8:     return c10::ScalarType::Char;
  case se::rhi::DataType::INT16:    return c10::ScalarType::Short;
  case se::rhi::DataType::INT32:    return c10::ScalarType::Int;
  case se::rhi::DataType::INT64:    return c10::ScalarType::Long;
  return c10::ScalarType::Float;
  }
}

auto toTensor(se::rhi::CUDABuffer* cudaBuffer, std::vector<int64_t> const& dimension, DataType type) noexcept -> torch::Tensor {
  torch::IntArrayRef dims(dimension.data(), dimension.size());
  torch::TensorOptions option = torch::TensorOptions(toScalarType(type));
  return torch::from_blob(cudaBuffer->ptr(), dims, option.device(torch::kCUDA));
}
}