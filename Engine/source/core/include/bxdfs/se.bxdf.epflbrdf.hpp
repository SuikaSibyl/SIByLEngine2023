#pragma once
#include <se.rhi.hpp>
#include <se.gfx.hpp>

namespace se {
struct EPFLBRDFData {
  // ndf tensor // --- 0
  uint32_t ndf_offset; 
  uint32_t ndf_shape_0;
  uint32_t ndf_shape_1;
  // sigma tensor
  uint32_t sigma_offset; // --- 1
  uint32_t sigma_shape_0;
  uint32_t sigma_shape_1;
  // vndf tensor
  uint32_t vndf_offset;
  uint32_t vndf_shape_0; // --- 2
  uint32_t vndf_shape_1;
  uint32_t vndf_param_size_0;
  uint32_t vndf_param_size_1;
  uint32_t vndf_param_stride_0; // --- 3
  uint32_t vndf_param_stride_1;
  uint32_t vndf_param_offset_0;
  uint32_t vndf_param_offset_1;
  uint32_t vndf_marginal_offset; // --- 4
  uint32_t vndf_conditional_offset;
  // luminance tensor
  uint32_t luminance_offset;
  uint32_t luminance_shape_0;
  uint32_t luminance_shape_1; // --- 5
  uint32_t luminance_param_size_0;
  uint32_t luminance_param_size_1;
  uint32_t luminance_param_stride_0;
  uint32_t luminance_param_stride_1; // --- 6
  uint32_t luminance_param_offset_0;
  uint32_t luminance_param_offset_1;
  uint32_t luminance_marginal_offset;
  uint32_t luminance_conditional_offset; // --- 7
  // rgb tensor
  uint32_t rgb_offset;
  uint32_t rgb_shape_0;
  uint32_t rgb_shape_1;
  uint32_t rgb_param_size_0; // --- 8
  uint32_t rgb_param_size_1;
  uint32_t rgb_param_size_2;
  uint32_t rgb_param_stride_0;
  uint32_t rgb_param_stride_1; // --- 9
  uint32_t rgb_param_stride_2;
  uint32_t rgb_param_offset_0;
  uint32_t rgb_param_offset_1;
  uint32_t rgb_param_offset_2; // --- 10
  // other parameters
  uint32_t isotropic;
  uint32_t jacobian; // --- 6
  uint32_t normalizer_offset;
  uint32_t padding_1;
};

struct SIByL_API EPFLBrdfDataPack {
  EPFLBrdfDataPack();
  gfx::Buffer buffer;
  gfx::Buffer brdfs;
  std::unordered_map<std::string, size_t> counters;
  std::unordered_map<std::string, size_t> maps;
};

struct SIByL_API EPFLBrdf :public gfx::IBxDF {
  EPFLBrdf(std::string const& filepath);
  virtual ~EPFLBrdf();
  std::string name;
  static EPFLBrdfDataPack datapack;
  static auto updateGPUResource() noexcept -> void;
  static auto bindingResourceBuffer() noexcept -> rhi::BindingResource;
  static auto bindingResourceBRDFs() noexcept -> rhi::BindingResource;
};

}