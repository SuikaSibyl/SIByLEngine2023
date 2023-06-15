#pragma once
#include <cstdint>
#include <vector>
#include <SE.Math.Geometric.hpp>
#include <SE.RHI.hpp>
#include <SE.PhysicS.SpatialHashing.hpp>

using namespace ::SIByL;

#define float double

namespace FLIP {
enum struct CellType : uint32_t {
    Solid = 0,
    Fluid = 1,
    Air   = 2,
};

struct FLIP2DDeviceData {
  uint32_t particle_count;
  float particle_radius;
  RHI::Buffer* particle_pos_buffer;
  RHI::Buffer* particle_col_buffer;
  RHI::Buffer* cell_color_buffer;
};

struct FLIP2DDescriptor {
  // fluid info
  float density;
  Math::bounds2 bound;
  // cell info
  uint32_t cellX, cellY, cellCount;
  float cell_size;
  float h;
  float cellInvSpacing; 
  // particle info
  float particle_radius;
  float dx, dy;
  uint32_t numX, numY;
  uint32_t particle_num;
  // grid info
  uint32_t gridX, gridY;
  uint32_t grid_num;
  float gridInvSpacing; 
  // solver info
  float dt = 1.0 / 60.0;
  uint32_t num_pressure_iters = 50;
  uint32_t num_particle_iters = 2;
};

struct FLIPInterface {
  virtual auto composeCmd(RHI::CommandEncoder* cmdEncoder,
                          size_t flightIdx) noexcept -> FLIP2DDeviceData = 0;
};

struct FLIP2D_host : public FLIPInterface {
  virtual auto composeCmd(RHI::CommandEncoder* cmdEncoder,
                          size_t flightIdx) noexcept
      -> FLIP2DDeviceData override;

  auto init(FLIP2DDescriptor const& desc) noexcept -> void;
  auto simulate(float dt) noexcept -> void;
  auto push_particle_apart() noexcept -> void;
  struct SoA {
    std::vector<Math::vec2> vel;
    std::vector<Math::vec2> pos;
    std::vector<Math::vec3> col;
  } dataParticle;
  struct SoACell {
    std::vector<float> u;   // x-axis velocity in stagger grid
    std::vector<float> v;   // y-axis velocity in stagger grid
    std::vector<float> du;  // sum of weights of u
    std::vector<float> dv;  // sum of weights of v
    std::vector<float> prev_u;  // previous u for FLIP
    std::vector<float> prev_v;  // previous v for FLIP
    std::vector<float> p;
    std::vector<float> s;
    std::vector<float> particle_density;
    std::vector<CellType>   type;
    std::vector<Math::vec3> color;
    float rest_density;
  } cellData;
  struct DeviceData {
    std::unique_ptr<RHI::Buffer> particle_pos_buffers[2] = {nullptr, nullptr};
    std::unique_ptr<RHI::Buffer> particle_color_buffers[2] = {nullptr, nullptr};
    std::unique_ptr<RHI::Buffer> cell_color_buffers[2] = {nullptr, nullptr};
    std::unique_ptr<RHI::Buffer> particle_pos_back;
    std::unique_ptr<RHI::Buffer> particle_color_back;
    std::unique_ptr<RHI::Buffer> cell_color_back;
  } deviceData;
 private:
  PhysicS::SpatialHashing2D spatial_hashing;
  FLIP2DDescriptor desc;
  uint32_t particle_num;
  auto forward(float dt) noexcept -> void;
  auto collision() noexcept -> void;
  auto transfer_vel_to_cell() noexcept -> void;
  auto transfer_vel_to_particle(float flipRatio) noexcept -> void;
  auto update_density() noexcept -> void;
  auto solve_incompressibility(float dt, float over_relaxation,
                               bool compensate_drift) noexcept -> void;
  auto update_color() noexcept -> void;
};
}

#undef float