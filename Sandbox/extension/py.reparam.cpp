#include <se.core.hpp>
#include <se.math.hpp>
#include <se.rhi.hpp>
#include <se.rhi.torch.hpp>
#include <se.gfx.hpp>
#include <se.rdg.hpp>

struct ForwardPass : public se::rdg::ComputePass {
  ForwardPass() {
    auto [comp] = se::gfx::GFXContext::load_shader_slang(
      "S:/SIByL2024/Sandbox/shaders/neuralmat/forward.slang",
      std::array<std::pair<std::string, se::rhi::ShaderStageBit>, 1>{
      std::make_pair("ComputeMain", se::rhi::ShaderStageBit::COMPUTE), });
    se::rdg::ComputePass::init(comp.get());
  }

  virtual auto reflect() noexcept -> se::rdg::PassReflection override {
    se::rdg::PassReflection reflector;
    return reflector;
  }
  
  virtual auto execute(se::rdg::RenderContext* context,
    se::rdg::RenderData const& renderData) noexcept -> void override {
    updateBindings(context, {
      {"u_image", se::rhi::BindingResource{{image_buffer, 0, image_buffer->size()}}},
      {"u_trans", se::rhi::BindingResource{{tran_buffer, 0, tran_buffer->size()}}},
    });
    
    se::rhi::ComputePassEncoder* encoder = beginPass(context);
    encoder->dispatchWorkgroups(32, 32, 1);
    encoder->end();
  }

  se::rhi::Buffer* image_buffer;
  se::rhi::Buffer* tran_buffer;
};

struct BackwardPass : public se::rdg::ComputePass {
  BackwardPass() {
    auto [comp] = se::gfx::GFXContext::load_shader_slang(
      "S:/SIByL2024/Sandbox/shaders/reparam/backward.slang",
      std::array<std::pair<std::string, se::rhi::ShaderStageBit>, 1>{
      std::make_pair("ComputeMain", se::rhi::ShaderStageBit::COMPUTE), }, {}, true);
    se::rdg::ComputePass::init(comp.get());
  }

  virtual auto reflect() noexcept -> se::rdg::PassReflection override {
    se::rdg::PassReflection reflector;
    return reflector;
  }
  
  virtual auto execute(se::rdg::RenderContext* context,
    se::rdg::RenderData const& renderData) noexcept -> void override {
    updateBindings(context, {
      {"u_gradient", se::rhi::BindingResource{{grad_buffer, 0, grad_buffer->size()}}},
      {"u_trans", se::rhi::BindingResource{{tran_buffer, 0, tran_buffer->size()}}},
      {"u_trans_gradient", se::rhi::BindingResource{{trangrad_buffer, 0, trangrad_buffer->size()}}},
    });
    
    se::rhi::ComputePassEncoder* encoder = beginPass(context);
    encoder->pushConstants(&seed, (uint32_t)se::rhi::ShaderStageBit::COMPUTE, 0, sizeof(seed));
    encoder->dispatchWorkgroups(32, 32, 1);
    encoder->end();
  }

  void setSeed(int64_t i) {
    seed = uint32_t(i);
  }

  uint32_t seed;
  se::rhi::Buffer* grad_buffer;
  se::rhi::Buffer* tran_buffer;
  se::rhi::Buffer* trangrad_buffer;
};

struct TinyMLPForwardPass : public se::rdg::ComputePass {
  TinyMLPForwardPass() {
    auto [comp] = se::gfx::GFXContext::load_shader_slang(
      "S:/SIByL2024/Sandbox/shaders/tinymlp/forward.slang",
      std::array<std::pair<std::string, se::rhi::ShaderStageBit>, 1>{
        std::make_pair("ComputeMain", se::rhi::ShaderStageBit::COMPUTE), }, {}, true);
    se::rdg::ComputePass::init(comp.get());
  }

  virtual auto reflect() noexcept -> se::rdg::PassReflection override {
    se::rdg::PassReflection reflector;
    return reflector;
  }
  
  virtual auto execute(se::rdg::RenderContext* context,
    se::rdg::RenderData const& renderData) noexcept -> void override {
    updateBindings(context, {
      {"output_texture", se::rhi::BindingResource{{output_buffer, 0, output_buffer->size()}}},
      {"_primal_buffer", se::rhi::BindingResource{{primal_buffer, 0, primal_buffer->size()}}},
      {"_gradient_buffer", se::rhi::BindingResource{{grad_buffer, 0, grad_buffer->size()}}},
    });

    struct PushConstant {
      se::ivec2 resolution;
      float gradient_scale;
    } pConst = { se::ivec2{ 512,512 }, this->gradient_scale };

    se::rhi::ComputePassEncoder* encoder = beginPass(context);
    //encoder->pushConstants(&pConst, (uint32_t)se::rhi::ShaderStageBit::COMPUTE, 0, sizeof(pConst));
    encoder->dispatchWorkgroups(16, 128, 1);
    encoder->end();
  }

  float gradient_scale = 0.1f;
  se::rhi::Buffer* output_buffer;
  se::rhi::Buffer* primal_buffer;
  se::rhi::Buffer* grad_buffer;
};

struct TinyMLPBackwardPass : public se::rdg::ComputePass {
  TinyMLPBackwardPass() {
    auto [comp] = se::gfx::GFXContext::load_shader_slang(
      "S:/SIByL2024/Sandbox/shaders/tinymlp/backward.slang",
      std::array<std::pair<std::string, se::rhi::ShaderStageBit>, 1>{
        std::make_pair("ComputeMain", se::rhi::ShaderStageBit::COMPUTE), }, {}, true);
    se::rdg::ComputePass::init(comp.get());
  }

  virtual auto reflect() noexcept -> se::rdg::PassReflection override {
    se::rdg::PassReflection reflector;
    return reflector;
  }
  
  virtual auto execute(se::rdg::RenderContext* context,
    se::rdg::RenderData const& renderData) noexcept -> void override {
    updateBindings(context, {
      {"grad_texture", se::rhi::BindingResource{{grad_texture, 0, grad_texture->size()}}},
      {"_primal_buffer", se::rhi::BindingResource{{primal_buffer, 0, primal_buffer->size()}}},
      {"_gradient_buffer", se::rhi::BindingResource{{grad_buffer, 0, grad_buffer->size()}}},
    });
    
    se::ivec2 resolution = { 512,512 };
    se::rhi::ComputePassEncoder* encoder = beginPass(context);
    encoder->pushConstants(&resolution, (uint32_t)se::rhi::ShaderStageBit::COMPUTE, 0, sizeof(resolution));
    encoder->dispatchWorkgroups(16, 128, 1);
    encoder->end();
  }

  se::rhi::Buffer* grad_texture;
  se::rhi::Buffer* primal_buffer;
  se::rhi::Buffer* grad_buffer;
};

PYBIND11_MODULE(pyapp, m) {
  m.doc() = "SIByLpy :: core module";  

  py::class_<ForwardPass> class_reapramFwdPass(m, "ReparamFwdPass");
  class_reapramFwdPass
    .def(py::init<>())
    .def("execute", &ForwardPass::execute)
    .def_readwrite("image_buffer", &ForwardPass::image_buffer)
    .def_readwrite("tran_buffer", &ForwardPass::tran_buffer);

  py::class_<BackwardPass> class_reapramBwdPass(m, "ReparamBwdPass");
  class_reapramBwdPass
    .def(py::init<>())
    .def("execute", &BackwardPass::execute)
    .def("setSeed", &BackwardPass::setSeed)
    .def_readwrite("grad_buffer", &BackwardPass::grad_buffer)
    .def_readwrite("tran_buffer", &BackwardPass::tran_buffer)
    .def_readwrite("trangrad_buffer", &BackwardPass::trangrad_buffer);

  py::class_<TinyMLPForwardPass> class_tinymlpFwdPass(m, "TinyMLPFwdPass");
  class_tinymlpFwdPass
    .def(py::init<>())
    .def("execute", &TinyMLPForwardPass::execute)
    .def_readwrite("output_buffer", &TinyMLPForwardPass::output_buffer)
    .def_readwrite("primal_buffer", &TinyMLPForwardPass::primal_buffer)
    .def_readwrite("grad_buffer", &TinyMLPForwardPass::grad_buffer);

  py::class_<TinyMLPBackwardPass> class_tinymlpBwdPass(m, "TinyMLPBwdPass");
  class_tinymlpBwdPass
    .def(py::init<>())
    .def("execute", &TinyMLPBackwardPass::execute)
    .def_readwrite("grad_texture", &TinyMLPBackwardPass::grad_texture)
    .def_readwrite("primal_buffer", &TinyMLPBackwardPass::primal_buffer)
    .def_readwrite("grad_buffer", &TinyMLPBackwardPass::grad_buffer);
}