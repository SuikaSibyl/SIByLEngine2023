#include <se.core.hpp>
#include <se.rhi.hpp>
#include <se.rhi.torch.hpp>
#include <se.gfx.hpp>
#include "py.rdg.hpp"
#include <se.rdg.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/pybind11.h>
//#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>
//#include "include.hpp"
//#include <cuda_runtime.h>
namespace py = pybind11;
using namespace se;
#pragma once

//
//void torch_launch_add2(
//    torch::Tensor& c,
//    const torch::Tensor& a,
//    const torch::Tensor& b,
//    int n) {
//    launch_add2((float*)c.data_ptr(),
//        (const float*)a.data_ptr(),
//        (const float*)b.data_ptr(),
//        n);
//}
//
////#include "se.test.cuh"
////
//////void torch_launch_add2(torch::Tensor& c,
//////    const torch::Tensor& a,
//////    const torch::Tensor& b,
//////    int64_t n) {
//////    //launch_add2((float*)c.data_ptr(),
//////    //    (const float*)a.data_ptr(),
//////    //    (const float*)b.data_ptr(),
//////    //    n);
//////}
//////
////
////void test() {
////
////}
////
////at::Tensor add_one_fn(const at::Tensor& input) {
////    auto output = torch::zeros_like(input);
////    return output;
////}
//
//int a() {
//	return 2;
//}
//
//torch::Tensor trilinear_interpolation(
//	torch::Tensor feats,
//	torch::Tensor points
//) {
//	return feats;
//}

struct rhi_namescope {};
struct gfx_namescope {};
struct rdg_namescope {};

void set_null(void* ptr) {
  ptr = nullptr;
}

PYBIND11_MODULE(pycore, m) {
  m.doc() = "SIByLpy :: core module";
  m.def("SetNullptr", &set_null);

  // Export root struct
  // ------------------------------------------------------------------------
  // add root class definition
  py::class_<root> class_root(m, "root");
  // add print module to root
  py::class_<root::print>(class_root, "print")
    .def(py::init<>())
    .def_static("debug", &root::print::debug)
    .def_static("log", &root::print::log)
    .def_static("warning", &root::print::warning)
    .def_static("error", &root::print::error)
    .def_static("correct", &root::print::correct);

  // Export window struct
  // ------------------------------------------------------------------------
  py::class_<se::window::WindowOptions> class_WindowOption(m, "WindowOption");
  class_WindowOption.def(py::init<se::window::Vendor, 
    std::wstring const&, uint32_t, uint32_t, se::window::Properties>())
    .def_readwrite("vendor", &se::window::WindowOptions::vendor)
    .def_readwrite("title", &se::window::WindowOptions::title)
    .def_readwrite("width", &se::window::WindowOptions::width)
    .def_readwrite("height", &se::window::WindowOptions::height)
    .def_readwrite("properties", &se::window::WindowOptions::properties);
  py::class_<se::window> class_window(m, "window");
  py::enum_<se::window::Vendor>(class_window, "EnumVendor")
    .value("GLFW", se::window::Vendor::GLFW)
    .value("Win64", window::Vendor::WIN_64);
  py::enum_<se::window::Properties>(class_window, "EnumProperty")
    .value("None", se::window::Properties::NONE)
    .value("VulkanContext", se::window::Properties::VULKAN_CONTEX)
    .value("OpenGLContext", se::window::Properties::OPENGL_CONTEX);
  class_window.def_static("create", &se::window::create);
  class_window.def("fetchEvents", &se::window::fetchEvents)
    .def("endFrame", &se::window::endFrame)
    .def("isRunning", &se::window::isRunning);

  // Export rhi namespace
  // ------------------------------------------------------------------------
  py::class_<rhi_namescope> namespace_rhi(m, "rhi");
  py::enum_<se::rhi::ContextExtensionBit>(namespace_rhi, "EnumContextExtension")
    .value("None", se::rhi::ContextExtensionBit::NONE)
    .value("DebugUtils", se::rhi::ContextExtensionBit::DEBUG_UTILS)
    .value("MeshShader", se::rhi::ContextExtensionBit::MESH_SHADER)
    .value("FragmentBarycentric", se::rhi::ContextExtensionBit::FRAGMENT_BARYCENTRIC)
    .value("SamplerFilterMinMax", se::rhi::ContextExtensionBit::SAMPLER_FILTER_MIN_MAX)
    .value("RayTracing", se::rhi::ContextExtensionBit::RAY_TRACING)
    .value("ShaderNonSemanticInfo", se::rhi::ContextExtensionBit::SHADER_NON_SEMANTIC_INFO)
    .value("BindlessIndexing", se::rhi::ContextExtensionBit::BINDLESS_INDEXING)
    .value("AtomicFloat", se::rhi::ContextExtensionBit::ATOMIC_FLOAT)
    .value("ConservativeRasterization", se::rhi::ContextExtensionBit::CONSERVATIVE_RASTERIZATION)
    .value("CooperativeMatrix", se::rhi::ContextExtensionBit::COOPERATIVE_MATRIX)
    .value("CudaInteroperability", se::rhi::ContextExtensionBit::CUDA_INTEROPERABILITY);
  py::enum_<se::rhi::MemoryPropertyBit>(namespace_rhi, "EnumMemoryProperty")
    .value("DEVICE_LOCAL_BIT", se::rhi::MemoryPropertyBit::DEVICE_LOCAL_BIT)
    .value("HOST_VISIBLE_BIT", se::rhi::MemoryPropertyBit::HOST_VISIBLE_BIT)
    .value("HOST_COHERENT_BIT", se::rhi::MemoryPropertyBit::HOST_COHERENT_BIT)
    .value("HOST_CACHED_BIT", se::rhi::MemoryPropertyBit::HOST_CACHED_BIT)
    .value("LAZILY_ALLOCATED_BIT", se::rhi::MemoryPropertyBit::LAZILY_ALLOCATED_BIT)
    .value("PROTECTED_BIT", se::rhi::MemoryPropertyBit::PROTECTED_BIT)
    .value("FLAG_BITS_MAX_ENUM", se::rhi::MemoryPropertyBit::FLAG_BITS_MAX_ENUM);
  
  // Export rhi::context structures
  // ------------------------------------------------------------------------
  py::class_<se::rhi::Context> class_rhi_context(namespace_rhi, "Context");
  py::enum_<se::rhi::Context::Backend>(class_rhi_context, "EnumBackend")
    .value("Vulkan", se::rhi::Context::Backend::Vulkan);
  class_rhi_context.def_static("create", &se::rhi::Context::create);
  class_rhi_context.def("init", &se::rhi::Context::init)
    .def("requestAdapter", &se::rhi::Context::requestAdapter);
  // Export rhi::adapter structures
  // ------------------------------------------------------------------------
  py::class_<se::rhi::AdapterDescriptor> class_rhi_adapter_options(
    namespace_rhi, "RequestAdapterOptions");
  py::enum_<se::rhi::AdapterDescriptor::PowerPreference>(class_rhi_adapter_options, "EnumPowerPreference")
    .value("LOW_POWER", se::rhi::AdapterDescriptor::PowerPreference::LOW_POWER)
    .value("HIGH_PERFORMANCE", se::rhi::AdapterDescriptor::PowerPreference::HIGH_PERFORMANCE);
  class_rhi_adapter_options.def(py::init<se::rhi::AdapterDescriptor::PowerPreference, bool>(),
    py::arg("powerPerference")= se::rhi::AdapterDescriptor::PowerPreference::HIGH_PERFORMANCE,
    py::arg("forceFallbackAdapter")=false)
    .def_readwrite("powerPerference", &se::rhi::AdapterDescriptor::powerPerference)
    .def_readwrite("forceFallbackAdapter", &se::rhi::AdapterDescriptor::forceFallbackAdapter);
  py::class_<se::rhi::Adapter> class_rhi_adapter(namespace_rhi, "Adapter");
  class_rhi_adapter.def("requestDevice", &se::rhi::Adapter::requestDevice);
  // Export rhi::device structures
  // ------------------------------------------------------------------------
  py::class_<se::rhi::Queue> class_rhi_queue(namespace_rhi, "Queue");
  class_rhi_queue.def("submit", py::overload_cast<std::vector<se::rhi::CommandBuffer*> const&>(&se::rhi::Queue::submit));

  py::class_<se::rhi::Device> class_rhi_device(namespace_rhi, "Device");
  class_rhi_device.def("createBuffer", &se::rhi::Device::createBuffer);
  class_rhi_device.def("waitIdle", &se::rhi::Device::waitIdle);
  class_rhi_device.def("getGraphicsQueue", &se::rhi::Device::getGraphicsQueue, py::return_value_policy::reference);

  // Export rhi::buffer structures
  // ------------------------------------------------------------------------
  py::enum_<se::rhi::BufferUsageBit>(namespace_rhi, "EnumBufferUsage")
    .value("MAP_READ", se::rhi::BufferUsageBit::MAP_READ)
    .value("MAP_WRITE", se::rhi::BufferUsageBit::MAP_WRITE)
    .value("COPY_SRC", se::rhi::BufferUsageBit::COPY_SRC)
    .value("COPY_DST", se::rhi::BufferUsageBit::COPY_DST)
    .value("INDEX", se::rhi::BufferUsageBit::INDEX)
    .value("VERTEX", se::rhi::BufferUsageBit::VERTEX)
    .value("UNIFORM", se::rhi::BufferUsageBit::UNIFORM)
    .value("STORAGE", se::rhi::BufferUsageBit::STORAGE)
    .value("INDIRECT", se::rhi::BufferUsageBit::INDIRECT)
    .value("QUERY_RESOLVE", se::rhi::BufferUsageBit::QUERY_RESOLVE)
    .value("SHADER_DEVICE_ADDRESS", se::rhi::BufferUsageBit::SHADER_DEVICE_ADDRESS)
    .value("ACCELERATION_STRUCTURE_STORAGE", se::rhi::BufferUsageBit::ACCELERATION_STRUCTURE_STORAGE)
    .value("ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY", se::rhi::BufferUsageBit::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY)
    .value("SHADER_BINDING_TABLE", se::rhi::BufferUsageBit::SHADER_BINDING_TABLE)
    .value("CUDA_ACCESS", se::rhi::BufferUsageBit::CUDA_ACCESS);
  py::enum_<se::rhi::BufferShareMode>(namespace_rhi, "EnumBufferShareMode")
    .value("CONCURRENT", se::rhi::BufferShareMode::CONCURRENT)
    .value("EXCLUSIVE", se::rhi::BufferShareMode::EXCLUSIVE);
  py::class_<se::rhi::BufferDescriptor> class_BufferDescriptor(namespace_rhi, "BufferDescriptor");
  class_BufferDescriptor.def(py::init<size_t,
    se::rhi::BufferUsages, se::rhi::BufferShareMode, se::rhi::MemoryProperties, bool, int>(),
    py::arg("size"), py::arg("usage"), py::arg("shareMode") = se::rhi::BufferShareMode::EXCLUSIVE, 
    py::arg("memoryProperties") = 0, py::arg("mappedAtCreation") = false, py::arg("minimumAlignment") = -1)
    .def_readwrite("size", &se::rhi::BufferDescriptor::size)
    .def_readwrite("usage", &se::rhi::BufferDescriptor::usage)
    .def_readwrite("shareMode", &se::rhi::BufferDescriptor::shareMode)
    .def_readwrite("memoryProperties", &se::rhi::BufferDescriptor::memoryProperties)
    .def_readwrite("mappedAtCreation", &se::rhi::BufferDescriptor::mappedAtCreation)
    .def_readwrite("minimumAlignment", &se::rhi::BufferDescriptor::minimumAlignment);
  py::class_<se::rhi::Buffer>(namespace_rhi, "Buffer");

  // Export rhi::CommandEncoder structures
  // ------------------------------------------------------------------------
  py::class_<se::rhi::CommandEncoderDescriptor> class_CommandEncoderDescriptor(namespace_rhi, "CommandEncoderDescriptor");
  class_CommandEncoderDescriptor.def(py::init<>());

  py::class_<se::rhi::CommandEncoder> class_CommandEncoder(namespace_rhi, "CommandEncoder");
  class_rhi_device.def("createCommandEncoder", &se::rhi::Device::createCommandEncoder);
  class_CommandEncoder.def("finish", &se::rhi::CommandEncoder::finish, py::return_value_policy::reference);
  
  py::class_<se::rhi::CommandBuffer> class_CommandBuffer(namespace_rhi, "CommandBuffer");

  // Export rhi::cudacontext structures
  // ------------------------------------------------------------------------
  py::class_<se::rhi::CUDABuffer> class_cudaBuffer(namespace_rhi, "CUDABuffer");
  py::class_<se::rhi::CUDAContext> class_cudacontext(namespace_rhi, "CUDAContext");
  class_cudacontext.def_static("initialize", py::overload_cast<se::rhi::Device*>(&se::rhi::CUDAContext::initialize));
  class_cudacontext.def_static("synchronize", &se::rhi::CUDAContext::synchronize);
  class_cudacontext.def_static("toCUDABuffer", &se::rhi::CUDAContext::toCUDABuffer);
  class_cudacontext.def_static("allocCUDABuffer", &se::rhi::CUDAContext::allocCUDABuffer);

  // Export rhi::pytorch structures
  // ------------------------------------------------------------------------
  namespace_rhi.def("toTensor", &se::rhi::toTensor);

  // Export gfx:: structures
  // ------------------------------------------------------------------------
  py::class_<gfx_namescope> namespace_gfx(m, "gfx");
  py::class_<se::gfx::GFXContext> class_gfx_context(namespace_gfx, "Context");
  class_gfx_context.def_static("initialize", py::overload_cast<se::rhi::Device*>(&se::gfx::GFXContext::initialize));
  class_gfx_context.def_static("finalize", &se::gfx::GFXContext::finalize);

  // Export gfx:: structures
  // ------------------------------------------------------------------------
  py::class_<rdg_namescope> namespace_rdg(m, "rdg");

  py::class_<se::rdg::RenderContext> class_render_context(namespace_rdg, "RenderContext");
  class_render_context.def(py::init<>())
    .def_readwrite("cmdEncoder", &se::rdg::RenderContext::cmdEncoder)
    .def_readwrite("flightIdx", &se::rdg::RenderContext::flightIdx);

  py::class_<se::rdg::RenderData> class_render_data(namespace_rdg, "RenderData");
  class_render_data.def(py::init<>());
}