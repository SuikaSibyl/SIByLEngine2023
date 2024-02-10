#include <se.core.hpp>
#include <se.math.hpp>
#include <se.rhi.hpp>
#include <se.rhi.torch.hpp>
#include <se.gfx.hpp>
#include <se.rdg.hpp>

struct ForwardPass : public se::rdg::ComputePass {
  ForwardPass() {
    auto [comp] = se::gfx::GFXContext::load_shader_slang(
      "S:/SIByL2024/Sandbox/shaders/reparam/forward.slang",
      std::array<std::pair<std::string, se::rhi::ShaderStageBit>, 1>{
      std::make_pair("ComputeMain", se::rhi::ShaderStageBit::COMPUTE), });
    se::rdg::ComputePass::init(comp.handle().get());
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
    se::rdg::ComputePass::init(comp.handle().get());
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

void FwdExe(ForwardPass& pass, se::rdg::RenderContext& renderContext, se::rdg::RenderData& renderData) {
  auto* device = se::gfx::GFXContext::device;
  std::unique_ptr<se::rhi::CommandEncoder> commandEncoder = device->createCommandEncoder({ nullptr });
  renderContext.cmdEncoder = commandEncoder.get();
  renderContext.flightIdx = 0;
  pass.execute(&renderContext, renderData);
  device->getGraphicsQueue()->submit({ commandEncoder->finish() });
  device->waitIdle();
}

void BwdExe(BackwardPass& pass, se::rdg::RenderContext& renderContext, se::rdg::RenderData& renderData) {
  auto* device = se::gfx::GFXContext::device;
  std::unique_ptr<se::rhi::CommandEncoder> commandEncoder = device->createCommandEncoder({ nullptr });
  renderContext.cmdEncoder = commandEncoder.get();
  renderContext.flightIdx = 0;
  pass.execute(&renderContext, renderData);
  device->getGraphicsQueue()->submit({ commandEncoder->finish() });
  device->waitIdle();
}

struct A {
    ~A(){
        se::root::print::debug("hello world!");
    }
};

struct AWarpper {
    static auto create() noexcept -> AWarpper {
        return AWarpper{std::make_unique<A>()};
    }
    auto get() noexcept -> A* { return ptr_a.get(); }
    std::unique_ptr<A> ptr_a;
};

PYBIND11_MODULE(pyapp, m) {
  m.doc() = "SIByLpy :: core module";

  m.def("FwdExe", &FwdExe);
  m.def("BwdExe", &BwdExe);
  
  py::class_<A> class_a(m, "A");
  class_a.def(py::init<>());
  py::class_<AWarpper> class_awarpper(m, "AWarpper");
  class_awarpper.def("get", &AWarpper::get, py::return_value_policy::reference)
      .def_static("create", &AWarpper::create);

  py::class_<ForwardPass> class_fwdPass(m, "FwdPass");
  class_fwdPass
    .def(py::init<>())
    .def("execute", &ForwardPass::execute)
    .def_readwrite("image_buffer", &ForwardPass::image_buffer)
    .def_readwrite("tran_buffer", &ForwardPass::tran_buffer);

  py::class_<BackwardPass> class_bwdPass(m, "BwdPass");
  class_bwdPass
    .def(py::init<>())
    .def("execute", &BackwardPass::execute)
    .def("setSeed", &BackwardPass::setSeed)
    .def_readwrite("grad_buffer", &BackwardPass::grad_buffer)
    .def_readwrite("tran_buffer", &BackwardPass::tran_buffer)
    .def_readwrite("trangrad_buffer", &BackwardPass::trangrad_buffer);
}