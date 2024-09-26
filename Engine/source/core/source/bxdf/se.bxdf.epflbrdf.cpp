#define DLIB_EXPORT
#include "bxdfs/se.bxdf.epflbrdf.hpp"
#undef DLIB_EXPORT
#define POWITACQ_IMPLEMENTATION
#include "ex.bxdf.powitacq.h"
#include <filesystem>

namespace se {
EPFLBrdfDataPack::EPFLBrdfDataPack() {
  buffer.usages =
    (uint32_t)rhi::BufferUsageBit::COPY_DST |
    (uint32_t)rhi::BufferUsageBit::STORAGE;
  brdfs.usages =
    (uint32_t)rhi::BufferUsageBit::COPY_DST |
    (uint32_t)rhi::BufferUsageBit::STORAGE;
}

EPFLBrdfDataPack EPFLBrdf::datapack;

EPFLBrdf::EPFLBrdf(std::string const& filepath) {
  name = std::filesystem::path(filepath).stem().string();
  auto find = datapack.counters.find(name);
  if (find == datapack.counters.end()) {
    datapack.counters[name] = 1;
  } else {
    find->second++;
    return;
  }

  powitacq_rgb::Tensor tf = powitacq_rgb::Tensor(filepath);
  auto& theta_i = tf.field("theta_i");
  auto& phi_i = tf.field("phi_i");
  auto& ndf = tf.field("ndf");
  auto& sigma = tf.field("sigma");
  auto& vndf = tf.field("vndf");
  auto& rgb = tf.field("rgb");
  auto& luminance = tf.field("luminance");
  auto& description = tf.field("description");
  auto& jacobian = tf.field("jacobian");
  
  if (!(description.shape.size() == 1 &&
    description.dtype == powitacq_rgb::Tensor::UInt8 &&
    
    theta_i.shape.size() == 1 &&
    theta_i.dtype == powitacq_rgb::Tensor::Float32 &&
    
    phi_i.shape.size() == 1 &&
    phi_i.dtype == powitacq_rgb::Tensor::Float32 &&
    
    ndf.shape.size() == 2 &&
    ndf.dtype == powitacq_rgb::Tensor::Float32 &&
    
    sigma.shape.size() == 2 &&
    sigma.dtype == powitacq_rgb::Tensor::Float32 &&
    
    vndf.shape.size() == 4 &&
    vndf.dtype == powitacq_rgb::Tensor::Float32 &&
    vndf.shape[0] == phi_i.shape[0] &&
    vndf.shape[1] == theta_i.shape[0] &&
    
    luminance.shape.size() == 4 &&
    luminance.dtype == powitacq_rgb::Tensor::Float32 &&
    luminance.shape[0] == phi_i.shape[0] &&
    luminance.shape[1] == theta_i.shape[0] &&
    luminance.shape[2] == luminance.shape[3] &&
    
    rgb.dtype == powitacq_rgb::Tensor::Float32 &&
    rgb.shape.size() == 5 &&
    rgb.shape[0] == phi_i.shape[0] &&
    rgb.shape[1] == theta_i.shape[0] &&
    rgb.shape[2] == 3 &&
    rgb.shape[3] == luminance.shape[2] &&
    
    luminance.shape[2] == rgb.shape[3] &&
    luminance.shape[3] == rgb.shape[4] &&
    
    jacobian.shape.size() == 1 &&
    jacobian.shape[0] == 1 &&
    jacobian.dtype == powitacq_rgb::Tensor::UInt8))
    root::print::error("Invalid file structure: " + tf.to_string());
    
  EPFLBRDFData data;
  data.isotropic = phi_i.shape[0] <= 2;
  data.jacobian = ((uint8_t*)jacobian.data.get())[0];
  
  if (!data.isotropic) {
    float* phi_i_data = (float*)phi_i.data.get();
    int reduction = (int)std::rint((2 * se::float_Pi) /
        (phi_i_data[phi_i.shape[0] - 1] - phi_i_data[0]));
    if (reduction != 1)
      root::print::error("reduction != 1, not supported by this implementation");
  }

  auto m_data = std::unique_ptr<powitacq_rgb::BRDF::Data>(new powitacq_rgb::BRDF::Data());
    
  auto push_back_data = [&](float* data, uint32_t byte_size) -> uint32_t {
    uint32_t byte_address = datapack.buffer.host.size();
    datapack.buffer.host.resize(byte_address + byte_size);
    memcpy(&datapack.buffer.host[byte_address], data, byte_size);
    return byte_address / 4;
  };

  /* Construct NDF interpolant data structure */
  m_data->ndf = powitacq_rgb::Warp2D0(
    powitacq_rgb::Vector2u(ndf.shape[1], ndf.shape[0]),
    (float*)ndf.data.get(),
    { }, { }, false, false
  );

  // Construct NDF interpolant data structure
  data.ndf_shape_0 = ndf.shape[1];
  data.ndf_shape_1 = ndf.shape[0];
  data.ndf_offset = push_back_data(
    m_data->ndf.m_data.data(),
    m_data->ndf.m_data.size() * sizeof(float));

  /* Construct projected surface area interpolant data structure */
  m_data->sigma = powitacq_rgb::Warp2D0(
    powitacq_rgb::Vector2u(sigma.shape[1], sigma.shape[0]),
    (float*)sigma.data.get(),
    { }, { }, false, false
  );

  // Construct NDF interpolant data structure
  data.sigma_shape_0 = sigma.shape[1];
  data.sigma_shape_1 = sigma.shape[0];
  data.sigma_offset = push_back_data(
    m_data->sigma.m_data.data(),
    m_data->sigma.m_data.size() * sizeof(float));

  /* Construct VNDF warp data structure */
  m_data->vndf = powitacq_rgb::Warp2D2(
    powitacq_rgb::Vector2u(vndf.shape[3], vndf.shape[2]),
    (float*)vndf.data.get(),
    { { (uint32_t)phi_i.shape[0],
       (uint32_t)theta_i.shape[0] } },
    { { (const float*)phi_i.data.get(),
       (const float*)theta_i.data.get() } }
  );
  data.vndf_shape_0 = vndf.shape[3];
  data.vndf_shape_1 = vndf.shape[2];
  data.vndf_offset = push_back_data(
    m_data->vndf.m_data.data(),
    m_data->vndf.m_data.size() * sizeof(float));
  data.vndf_param_size_0 = m_data->vndf.m_param_size[0];
  data.vndf_param_size_1 = m_data->vndf.m_param_size[1];
  data.vndf_param_stride_0 = m_data->vndf.m_param_strides[0];
  data.vndf_param_stride_1 = m_data->vndf.m_param_strides[1];
  data.vndf_param_offset_0 = push_back_data(
    m_data->vndf.m_param_values[0].data(),
    m_data->vndf.m_param_values[0].size() * sizeof(float));
  data.vndf_param_offset_1 = push_back_data(
    m_data->vndf.m_param_values[1].data(),
    m_data->vndf.m_param_values[1].size() * sizeof(float));
  data.vndf_marginal_offset = push_back_data(
    m_data->vndf.m_marginal_cdf.data(),
    m_data->vndf.m_marginal_cdf.size() * sizeof(float));
  data.vndf_conditional_offset = push_back_data(
    m_data->vndf.m_conditional_cdf.data(),
    m_data->vndf.m_conditional_cdf.size() * sizeof(float));

  /* Construct Luminance warp data structure */
  m_data->luminance = powitacq_rgb::Warp2D2(
    powitacq_rgb::Vector2u(luminance.shape[3], luminance.shape[2]),
    (float*)luminance.data.get(),
    { { (uint32_t)phi_i.shape[0],
       (uint32_t)theta_i.shape[0] } },
    { { (const float*)phi_i.data.get(),
       (const float*)theta_i.data.get() } }
  );
  data.luminance_shape_0 = luminance.shape[3];
  data.luminance_shape_1 = luminance.shape[2];
  data.luminance_offset = push_back_data(
    m_data->luminance.m_data.data(),
    m_data->luminance.m_data.size() * sizeof(float));
  data.luminance_param_size_0 = m_data->luminance.m_param_size[0];
  data.luminance_param_size_1 = m_data->luminance.m_param_size[1];
  data.luminance_param_stride_0 = m_data->luminance.m_param_strides[0];
  data.luminance_param_stride_1 = m_data->luminance.m_param_strides[1];
  data.luminance_param_offset_0 = push_back_data(
    m_data->luminance.m_param_values[0].data(),
    m_data->luminance.m_param_values[0].size() * sizeof(float));
  data.luminance_param_offset_1 = push_back_data(
    m_data->luminance.m_param_values[1].data(),
    m_data->luminance.m_param_values[1].size() * sizeof(float));
  data.luminance_marginal_offset = push_back_data(
    m_data->luminance.m_marginal_cdf.data(),
    m_data->luminance.m_marginal_cdf.size() * sizeof(float));
  data.luminance_conditional_offset = push_back_data(
    m_data->luminance.m_conditional_cdf.data(),
    m_data->luminance.m_conditional_cdf.size() * sizeof(float));


  /* Construct spectral interpolant */
  const float channels[] = { 0.0f, 1.0f, 2.0f };
  m_data->rgb = powitacq_rgb::Warp2D3(
    powitacq_rgb::Vector2u(rgb.shape[4], rgb.shape[3]),
    (float*)rgb.data.get(),
    { { (uint32_t)phi_i.shape[0],
       (uint32_t)theta_i.shape[0],
       (uint32_t)3 } },
    { { (const float*)phi_i.data.get(),
       (const float*)theta_i.data.get(),
       (const float*)channels } },
      false, false, true
  );

  data.rgb_shape_0 = rgb.shape[4];
  data.rgb_shape_1 = rgb.shape[3];
  data.rgb_offset = push_back_data(
    m_data->rgb.m_data.data(),
    m_data->rgb.m_data.size() * sizeof(float));
  data.rgb_param_size_0 = m_data->rgb.m_param_size[0];
  data.rgb_param_size_1 = m_data->rgb.m_param_size[1];
  data.rgb_param_size_2 = m_data->rgb.m_param_size[2];
  data.rgb_param_stride_0 = m_data->rgb.m_param_strides[0];
  data.rgb_param_stride_1 = m_data->rgb.m_param_strides[1];
  data.rgb_param_stride_2 = m_data->rgb.m_param_strides[2];
  data.rgb_param_offset_0 = push_back_data(
    m_data->rgb.m_param_values[0].data(),
    m_data->rgb.m_param_values[0].size() * sizeof(float));
  data.rgb_param_offset_1 = push_back_data(
    m_data->rgb.m_param_values[1].data(),
    m_data->rgb.m_param_values[1].size() * sizeof(float));
  data.rgb_param_offset_2 = push_back_data(
    m_data->rgb.m_param_values[2].data(),
    m_data->rgb.m_param_values[2].size() * sizeof(float));
  data.normalizer_offset = push_back_data(
    m_data->rgb.m_normalizer.data(),
    m_data->rgb.m_normalizer.size() * sizeof(float));
  se::root::print::debug("base size:" + std::to_string(data.normalizer_offset / 4));
  se::root::print::debug("additional size:" + std::to_string(m_data->rgb.m_normalizer.size()));

  size_t mat_idx = datapack.brdfs.host.size();
  datapack.maps[name] = mat_idx / sizeof(EPFLBRDFData);
  datapack.brdfs.host.resize(mat_idx + sizeof(EPFLBRDFData));
  memcpy(&datapack.brdfs.host[mat_idx], &data, sizeof(EPFLBRDFData));
}

EPFLBrdf::~EPFLBrdf() {
  auto find = datapack.counters.find(name);
  find->second--;
  if (find->second == 0) {
    datapack.maps.erase(name);
    datapack.counters.erase(name);
  }
  if (datapack.maps.empty()) {
    datapack.buffer.buffer = nullptr;
    datapack.buffer.previous = nullptr;
    datapack.brdfs.buffer = nullptr;
    datapack.brdfs.previous = nullptr;
  }
}

auto EPFLBrdf::updateGPUResource() noexcept -> void {
  datapack.buffer.hostToDevice();
  datapack.brdfs.hostToDevice();
}

auto EPFLBrdf::bindingResourceBuffer() noexcept -> rhi::BindingResource {
  return rhi::BindingResource{ {datapack.buffer.buffer.get(), 0, datapack.buffer.buffer->size()} };
}

auto EPFLBrdf::bindingResourceBRDFs() noexcept -> rhi::BindingResource {
  return rhi::BindingResource{ {datapack.brdfs.buffer.get(), 0, datapack.brdfs.buffer->size()} };
}
}