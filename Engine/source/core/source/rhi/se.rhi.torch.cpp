#define DLIB_EXPORT
#include <se.rhi.torch.hpp>
#undef DLIB_EXPORT

namespace se::rhi {
auto toTensor(se::rhi::CUDABuffer* cudaBuffer, std::vector<int64_t> const& dimension) noexcept -> torch::Tensor {
  torch::IntArrayRef dims(dimension.data(), dimension.size());
  return torch::from_blob(cudaBuffer->ptr(), dims, torch::TensorOptions().device(torch::kCUDA));
}
}