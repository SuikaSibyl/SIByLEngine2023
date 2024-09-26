#include <se.core.hpp>
#include <seditor-base.hpp>
#include <cnpy.h>
#include <se.image.hpp>
#include <bxdfs/se.bxdf.epflbrdf.hpp>
#include <lights/se.lights.envmap.hpp>

int main() {
std::unique_ptr<se::window> window = se::window::create(se::window::WindowOptions{
    se::window::Vendor::GLFW, L"SIByL 2024",
    1280, 720, se::window::Properties::VULKAN_CONTEX });

 std::unique_ptr<se::rhi::Context> context = 
   se::rhi::Context::create(se::rhi::Context::Backend::Vulkan);
 context->init(window.get(), se::rhi::ContextExtensions(
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
 se::gfx::GFXContext::createFlights(MULTIFRAME_FLIGHTS_COUNT, nullptr);
 se::editor::ImGuiContext::initialize(device.get());
 se::editor::EditorContext::initialize();
 ImGui::SetCurrentContext(se::editor::ImGuiContext::getRawCtx());


 //std::string path = "S:/SIByL2024/Sandbox/examples/lighting/_data/test.exr";
 //std::unique_ptr<se::EnvmapLight> envlight = std::make_unique<se::EnvmapLight>(path);
 //envlight = nullptr;
 //se::gfx::PMFConstructor::upload_datapack();
 //se::gfx::PMFConstructor::clear_datapack();
 std::string path = "S:/SIByL2024/Sandbox/examples/glt/_data/cbox.scene";
 auto scene = se::gfx::GFXContext::load_scene_gltf(path);
 scene->updateGPUScene();
 //std::vector<float> pmf = { 0.1,0.2,0.3,0.4 };
 //auto pc1d = se::gfx::PMFConstructor::build_piecewise_constant_1d(pmf, 0,1);
 //float pdf; int offset;
 //float a = pc1d.sample(0.09, pdf, offset);


 device->waitIdle();
 se::editor::EditorBase::finalize();
 se::editor::ImGuiContext::finalize();
 se::gfx::GFXContext::finalize();
 window->destroy();
}