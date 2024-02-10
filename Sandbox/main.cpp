#include <se.core.hpp>
#include <se.math.hpp>
#include <se.rhi.hpp>
#include <se.rhi.torch.hpp>
#include <se.gfx.hpp>
#include <se.rdg.hpp>
#include "extension/py.reparam.cpp"
#include <iostream>

int main() {
  //std::unique_ptr<se::window> window = se::window::create(se::window::WindowOptions{
  //  se::window::Vendor::GLFW, L"SIByL 2024",
  //  1280, 720, se::window::Properties::VULKAN_CONTEX});
  
  std::unique_ptr<se::rhi::Context> context = 
	se::rhi::Context::create(se::rhi::Context::Backend::Vulkan);
  //context->init(window.get(), se::rhi::ContextExtensions(
  context->init(nullptr, se::rhi::ContextExtensions(
    se::rhi::ContextExtensionBit::RAY_TRACING |
    se::rhi::ContextExtensionBit::BINDLESS_INDEXING |
    se::rhi::ContextExtensionBit::FRAGMENT_BARYCENTRIC |
    se::rhi::ContextExtensionBit::CONSERVATIVE_RASTERIZATION |
    se::rhi::ContextExtensionBit::COOPERATIVE_MATRIX |
    se::rhi::ContextExtensionBit::CUDA_INTEROPERABILITY |
    se::rhi::ContextExtensionBit::ATOMIC_FLOAT));

  std::unique_ptr<se::rhi::Adapter> adapter = context->requestAdapter({});
  std::unique_ptr<se::rhi::Device> device = adapter->requestDevice();
  se::rhi::CUDAContext::initialize(device.get());
  se::gfx::GFXContext::initialize(device.get());

  std::unique_ptr<se::rhi::Buffer> prim_buffer = device->createBuffer(se::rhi::BufferDescriptor{
    512*512 * sizeof(float), (uint32_t)se::rhi::BufferUsageBit::STORAGE,
    se::rhi::BufferShareMode::EXCLUSIVE, 
    (uint32_t)se::rhi::MemoryPropertyBit::DEVICE_LOCAL_BIT,});
  std::unique_ptr<se::rhi::Buffer> tans_buffer = device->createBuffer(se::rhi::BufferDescriptor{
    3 * sizeof(float), (uint32_t)se::rhi::BufferUsageBit::STORAGE,
    se::rhi::BufferShareMode::EXCLUSIVE, 
    (uint32_t)se::rhi::MemoryPropertyBit::DEVICE_LOCAL_BIT,});
  std::unique_ptr<se::rhi::Buffer> grad_buffer = device->createBuffer(se::rhi::BufferDescriptor{
    512 * 512 * sizeof(float), (uint32_t)se::rhi::BufferUsageBit::STORAGE,
    se::rhi::BufferShareMode::EXCLUSIVE, 
    (uint32_t)se::rhi::MemoryPropertyBit::DEVICE_LOCAL_BIT,});

  BackwardPass bwdpass;

  std::unique_ptr<se::rhi::CUDABuffer> prim_cu = se::rhi::CUDAContext::toCUDABuffer(prim_buffer.get());
  std::unique_ptr<se::rhi::CUDABuffer> grad_cu = se::rhi::CUDAContext::toCUDABuffer(grad_buffer.get());
  std::unique_ptr<se::rhi::CUDABuffer> trans_cu = se::rhi::CUDAContext::toCUDABuffer(tans_buffer.get());
  
  torch::Tensor prim_tensor = se::rhi::toTensor(prim_cu.get(), { 512, 512 });
  torch::Tensor grad_tensor = se::rhi::toTensor(grad_cu.get(), { 512, 512 });
  torch::Tensor trans_tensor = se::rhi::toTensor(trans_cu.get(), { 512, 512 });
  
  ForwardPass pass;
  pass.image_buffer = prim_buffer.get();
  pass.tran_buffer = tans_buffer.get();
  se::rdg::RenderContext renderContext;
  se::rdg::RenderData renderData;

  for (int i = 0; i < 10; ++i) {
    std::unique_ptr<se::rhi::CommandEncoder> commandEncoder = device->createCommandEncoder({ nullptr });
    renderContext.cmdEncoder = commandEncoder.get();
    renderContext.flightIdx = 0;
    pass.execute(&renderContext, renderData);
    
    device->getGraphicsQueue()->submit({ commandEncoder->finish() });
    device->waitIdle();
  }

  torch::Tensor prim_cpu = prim_tensor.cpu();
  std::cout << prim_cpu << std::endl;


  se::gfx::GFXContext::finalize();

  //se::gfx::GFXContext::shaders.load(ex::id_type(0), se::gfx::ShaderModuleLoader::from_spirv_tag{}, nullptr, se::rhi::ShaderStageBit::COMPUTE);

  //ex::resource<se::gfx::ShaderModule> a = se::gfx::GFXContext::shaders[ex::id_type(0)];



  //using my_cache = entt::resource_cache<my_resource, my_loader>;
  //my_cache cache{};

  //{
  //    cache.load(0,0);
  //}

  //float ssa = 0;
  //ex::resource<my_resource> load0 = cache[0];
  //my_resource* raw = load0.handle().get();

  //bool should_exit = false;
  //// run the main loop
  //while (!should_exit) {
  //  //  fetch main window events
  //  window->fetchEvents();
  //  window->endFrame();
  //  // Update window status, to check whether should exit
  //  should_exit |= !window->isRunning();
  //}
  //window->destroy();

  return 0;
}