#pragma once
#include "FLIP.hpp"
#include "FLIPRender.hpp"

namespace FLIP {
 struct FLIPDemo {
  FLIPDemo();
  ~FLIPDemo();

  std::unique_ptr<FLIP2D_host> flip2d = nullptr;

   auto onStart() noexcept -> void;
   auto Exit() noexcept -> void;
   auto onUpdate(RHI::CommandEncoder* encoder, uint32_t idx) noexcept -> void;

   Platform::Input* input;
   FLIP2DDeviceData device_data;
   std::unique_ptr<RDG::Pipeline> pipeline;
   ParticleDrawPass* particle_draw_pass;
 };
}