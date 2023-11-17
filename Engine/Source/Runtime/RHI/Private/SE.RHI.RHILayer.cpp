#pragma once

#include <Memory>
#include <Print/SE.Core.Log.hpp>
#include <System/SE.Core.System.hpp>
#include <string>
#include "../../Application/Public/SE.Application.Config.h"
#include "SE.RHI-Interface.hpp"
#include "SE.RHI-VK.hpp"
#include <SE.RHI.RHILayer.hpp>
#include <GFSDK_Aftermath.h>
#include <GFSDK_Aftermath_GpuCrashDump.h>
#include <IO/SE.Core.IO.hpp>
#include <Misc/SE.Core.Misc.hpp>

namespace SIByL::RHI {
RHILayer* RHILayer::singleton = nullptr;

namespace Aftermath {
std::mutex m_mutex;
// Static wrapper for the GPU crash dump handler. See the 'Handling GPU crash
// dump Callbacks' section for details.
void GpuCrashDumpCallback(const void* pGpuCrashDump,
                          const uint32_t gpuCrashDumpSize, void* pUserData) {
  //// Make sure only one thread at a time...
   std::lock_guard<std::mutex> lock(m_mutex);
  //  Write to file for later in-depth analysis.
  Core::Buffer buffer;
  buffer.isReference = true;
  buffer.data = (void*)pGpuCrashDump;
  buffer.size = gpuCrashDumpSize;
  std::string timestamp = Core::WorldTimePoint::get().to_string();
  Core::syncWriteFile(std::string("./dmp/" + timestamp + ".nv-gpudmp").c_str(), buffer);
}

// Static wrapper for the shader debug information handler. See the 'Handling
// Shader Debug Information callbacks' section for details.
void ShaderDebugInfoCallback(const void* pShaderDebugInfo,
                             const uint32_t shaderDebugInfoSize,
                             void* pUserData) {
  Core::LogManager::Error("ShaderDebugInfoCallback");
}

// Static wrapper for the GPU crash dump description handler. See the
// 'Handling GPU Crash Dump Description Callbacks' section for details.
void CrashDumpDescriptionCallback(
    PFN_GFSDK_Aftermath_AddGpuCrashDumpDescription addDescription,
    void* pUserData) {
  // GpuCrashTracker* pGpuCrashTracker =
  //     reinterpret_cast<GpuCrashTracker*>(pUserData);
  // pGpuCrashTracker->OnDescription(addDescription);

  Core::LogManager::Error("CrashDumpDescriptionCallback");
}

// Static wrapper for the resolve marker handler. See the 'Handling Marker
// Resolve Callbacks' section for details.
void ResolveMarkerCallback(const void* pMarkerData,
                           const uint32_t markerDataSize, void* pUserData,
                           void** ppResolvedMarkerData,
                           uint32_t* pResolvedMarkerDataSize) {
  Core::LogManager::Error("ResolveMarkerCallback");
  // GpuCrashTracker* pGpuCrashTracker =
  // reinterpret_cast<GpuCrashTracker*>(pUserData);
  // pGpuCrashTracker->OnResolveMarker(pMarkerData, markerDataSize,
  // ppResolvedMarkerData, pResolvedMarkerDataSize);
}
}

RHILayer::RHILayer(RHILayerDescriptor const& desc) : desc(desc) {
  singleton = this;
  if (desc.backend == RHIBackend::Vulkan) {
    if (desc.useAftermath) {
      // Enable GPU crash dumps and register callbacks.
      GFSDK_Aftermath_EnableGpuCrashDumps(
          GFSDK_Aftermath_Version_API,
          GFSDK_Aftermath_GpuCrashDumpWatchedApiFlags_Vulkan,
          GFSDK_Aftermath_GpuCrashDumpFeatureFlags_Default,  // Default behavior.
          Aftermath::GpuCrashDumpCallback,  // Register callback for GPU crash dumps.
          Aftermath::ShaderDebugInfoCallback,  // Register callback for shader debug information.
          Aftermath::CrashDumpDescriptionCallback,  // Register callback for GPU crash dump description.
          Aftermath::ResolveMarkerCallback,  // Register callback for marker resolution (R495 or later NVIDIA graphics driver).
          nullptr);               // Set the GpuCrashTracker object as user data passed back by the above callbacks.
    }
    context = std::make_unique<Context_VK>();
    context->init(desc.windowBinded, desc.extensions);
    adapter = context->requestAdapter({});
    device = adapter->requestDevice();
    multiFrameFlights = device->createMultiFrameFlights(
        {MULTIFRAME_FLIGHTS_COUNT, swapChain.get()});
    if (!desc.useImGui) {
      swapChain = device->createSwapChain({});
      desc.windowBinded->connectResizeEvent(
          [&](size_t w, size_t h) -> void { swapChain->recreate(); });
    }
  } else if (desc.backend == RHIBackend::OpenGL) {
    Core::LogManager::Error("RHILayer :: OpenGL backend is not supported yet.");
  } else if (desc.backend == RHIBackend::None) {
    Core::LogManager::Error(
        "RHILayer :: No backend is selected for RHILayer, soft fallback is not "
        "supported yet.");
  }
}

RHILayer::~RHILayer() {
  if (desc.useAftermath) {
    // Disable GPU crash dump creation.
    GFSDK_Aftermath_DisableGpuCrashDumps();
  }
  if (desc.backend == RHIBackend::Vulkan) {
    static_cast<Context_VK*>(context.get())->getVkSurfaceKHR() = {};
  }
}
}  // namespace SIByL::RHI