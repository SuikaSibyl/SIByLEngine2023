#include "FLIPDemo.hpp"

namespace FLIP {
FLIPDemo::FLIPDemo() {}

FLIPDemo::~FLIPDemo() {}

auto FLIPDemo::onStart() noexcept -> void {
  double const tank_width = 4;
  double const tank_height = 3;
  uint32_t const resolution = 100;
  double const cell_size = tank_height / resolution;
  double rel_water_height = 0.8, rel_water_width = 0.6;
  double radius = 0.3 * cell_size;

  FLIP2DDescriptor desc;
  // fluid info
  desc.density = 1000.0f;
  // grid infomation compute
  desc.cellX = std::floor(tank_width / cell_size) + 1;
  desc.cellY = std::floor(tank_height / cell_size) + 1;
  desc.cellCount = desc.cellX * desc.cellY;
  desc.cell_size = cell_size;
  desc.h = std::max(tank_width / desc.cellX, tank_height / desc.cellY);
  desc.cellInvSpacing = 1.0 / desc.h;
  // particles infomation compute
  desc.particle_radius = 0.3 * cell_size;  // particle radius w.r.t. cell size
  desc.dx = 2.0 * desc.particle_radius;
  desc.dy = std::sqrt(3.0f) / 2.0f * desc.dx;
  desc.numX = std::floor(
      (rel_water_width * tank_width - 2.0 * cell_size - 2.0 * radius) /
      desc.dx);
  desc.numY = std::floor(
      (rel_water_height * tank_height - 2.0 * cell_size - 2.0 * radius) /
      desc.dy);
  desc.particle_num = desc.numX * desc.numY;
  //
  double h = 1.0 / desc.cellInvSpacing;
  double minX = h + desc.particle_radius;
  double maxX = (desc.cellX - 1) * h - desc.particle_radius;
  double minY = h + desc.particle_radius;
  double maxY = (desc.cellY - 1) * h - desc.particle_radius;

  desc.bound.pMin = Math::dvec2{minX, minY};
  desc.bound.pMax = Math::dvec2{maxX, maxY};
  // grid data
  desc.gridInvSpacing = 1.0 / (2.2 * desc.particle_radius);
  desc.gridX = std::floor(tank_width * desc.gridInvSpacing) + 1;
  desc.gridY = std::floor(tank_height * desc.gridInvSpacing) + 1;
  desc.grid_num = desc.gridX * desc.gridY;

  flip2d = std::make_unique<FLIP2D_host>();
  flip2d->init(desc);

  pipeline = std::make_unique<SRP::FLIPRendererPipeline>(desc);
  SRP::FLIPRendererPipeline* pip =
      reinterpret_cast<SRP::FLIPRendererPipeline*>(pipeline.get());
  pip->graph.particle_pass->data = &device_data;
  pip->graph.cell_pass->data = &device_data;
  pipeline->build();
}

auto FLIPDemo::Exit() noexcept -> void {
  flip2d = nullptr;
  pipeline = nullptr;
}

auto FLIPDemo::onUpdate(RHI::CommandEncoder* encoder, uint32_t idx) noexcept
    -> void {
  static bool isStart = false;
  if (input->isKeyPressed(Platform::SIByL_KEY_SPACE)) {
    isStart = true;
  }
  if (isStart)
    flip2d->simulate(1.0 / 60.0);
  device_data = flip2d->composeCmd(encoder, idx);
}
}  // namespace FLIP