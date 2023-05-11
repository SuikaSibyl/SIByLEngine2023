#pragma once

#include <Memory>
#include <Print/SE.Core.Log.hpp>
#include <System/SE.Core.System.hpp>
#include <string>
#include "../../Application/Public/SE.Application.Config.h"
#include "SE.RHI-Interface.hpp"
#include "SE.RHI-VK.hpp"
#include <SE.RHI.RHILayer.hpp>

namespace SIByL::RHI {
RHILayer* RHILayer::singleton = nullptr;

RHILayer::RHILayer(RHILayerDescriptor const& desc) : desc(desc) {
  singleton = this;
  if (desc.backend == RHIBackend::Vulkan) {
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
  if (desc.backend == RHIBackend::Vulkan) {
    static_cast<Context_VK*>(context.get())->getVkSurfaceKHR() = {};
  }
}
}  // namespace SIByL::RHI