#pragma once

#include <Memory>
#include <string>
#include "../../Application/Public/SE.Application.Config.h"
#include "SE.RHI-Interface.hpp"
#include <Print/SE.Core.Log.hpp>
#include <System/SE.Core.System.hpp>
import SE.Platform.Window;

namespace SIByL::RHI {
SE_EXPORT enum struct RHIBackend {
  None,
  OpenGL,
  Vulkan,
};

SE_EXPORT struct RHILayerDescriptor {
  RHIBackend backend = RHIBackend::None;
  ContextExtensionsFlags extensions = {};
  Platform::Window* windowBinded = nullptr;
  bool useImGui = false;
  bool useAftermath = false;
};

SE_EXPORT struct RHILayer : public Core::Layer {
  /** initialzier */
  RHILayer(RHILayerDescriptor const& desc);
  /** virtual destructor */
  virtual ~RHILayer();
  /** get rhi context */
  auto getContext() noexcept -> RHI::Context* { return context.get(); }
  /** get rhi adapter */
  auto getAdapter() noexcept -> RHI::Adapter* { return adapter.get(); }
  /** get rhi device */
  auto getDevice() noexcept -> RHI::Device* { return device.get(); }
  /** get swapChain device */
  auto getSwapChain() noexcept -> RHI::SwapChain* { return swapChain.get(); }
  /** get multi frame flights device */
  auto getMultiFrameFlights() noexcept -> RHI::MultiFrameFlights* {
    return multiFrameFlights.get();
  }
  /** get descriptor */
  auto getRHILayerDescriptor() const noexcept -> RHILayerDescriptor const& {
    return desc;
  }
  /** get singleton */
  static auto get() noexcept -> RHILayer* { return singleton; }

 private:
  RHILayerDescriptor const desc;
  /** rhi context */
  std::unique_ptr<RHI::Context> context = nullptr;
  /** rhi adapter */
  std::unique_ptr<RHI::Adapter> adapter = nullptr;
  /** rhi device */
  std::unique_ptr<RHI::Device> device = nullptr;
  /** swapChain device */
  std::unique_ptr<RHI::SwapChain> swapChain = nullptr;
  /** multi frame flights device */
  std::unique_ptr<RHI::MultiFrameFlights> multiFrameFlights = nullptr;
  /** singleton */
  static RHILayer* singleton;
};
}  // namespace SIByL::RHI