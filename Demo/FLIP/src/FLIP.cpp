#include "FLIP.hpp"
#include <time.h>

#define float double

namespace FLIP {
const float gravity_constant = -9.81f;

auto FLIP2D_host::composeCmd(
    RHI::CommandEncoder* cmdEncoder, size_t i) noexcept
    -> FLIP2DDeviceData {
  FLIP2DDeviceData retData;
  RHI::Device* device = RHI::RHILayer::get()->getDevice();

  RHI::BarrierDescriptor barrier{
      (uint32_t)RHI::PipelineStages::ALL_GRAPHICS_BIT,
      (uint32_t)RHI::PipelineStages::ALL_GRAPHICS_BIT,
      0,
      // Optional (Memory Barriers)
      {},
      {RHI::BufferMemoryBarrierDescriptor{
          nullptr,
          (uint32_t)RHI::AccessFlagBits::MEMORY_WRITE_BIT,
          (uint32_t)RHI::AccessFlagBits::MEMORY_READ_BIT,
      }},
      {}};
  std::vector<RHI::BarrierDescriptor> barriers;

  deviceData.particle_pos_buffers[i] = device->createDeviceLocalBuffer(
      dataParticle.pos.data(), dataParticle.pos.size() * sizeof(Math::vec2),
      (uint32_t)RHI::BufferUsage::STORAGE);

  deviceData.cell_color_buffers[i] = device->createDeviceLocalBuffer(
      cellData.color.data(), cellData.color.size() * sizeof(Math::vec3),
      (uint32_t)RHI::BufferUsage::STORAGE);

  deviceData.particle_color_buffers[i] = device->createDeviceLocalBuffer(
      dataParticle.col.data(), dataParticle.col.size() * sizeof(Math::vec3),
      (uint32_t)RHI::BufferUsage::STORAGE);

  retData.cell_color_buffer = deviceData.cell_color_buffers[i].get();
  retData.particle_pos_buffer = deviceData.particle_pos_buffers[i].get();
  retData.particle_col_buffer = deviceData.particle_color_buffers[i].get();
  retData.particle_count = desc.particle_num;
  retData.particle_radius = desc.particle_radius;

  return retData;
}

auto FLIP2D_host::init(FLIP2DDescriptor const& desc) noexcept -> void {
  this->desc = desc;
  // alloc cell data
  cellData.u.resize(desc.cellCount);
  cellData.v.resize(desc.cellCount);
  cellData.du.resize(desc.cellCount);
  cellData.dv.resize(desc.cellCount);
  cellData.prev_u.resize(desc.cellCount);
  cellData.prev_v.resize(desc.cellCount);
  cellData.p.resize(desc.cellCount);
  cellData.s.resize(desc.cellCount);
  cellData.type.resize(desc.cellCount);
  cellData.color.resize(desc.cellCount);
  cellData.particle_density.resize(desc.cellCount);
  cellData.rest_density = 0.f;
  // alloc particle data
  dataParticle.pos.resize(desc.particle_num);
  dataParticle.vel.resize(desc.particle_num);
  dataParticle.col.resize(desc.particle_num);
  // init particle data
  uint32_t p = 0;
  for (uint32_t i = 0; i < desc.numX; i++) {
    for (uint32_t j = 0; j < desc.numY; j++) {
      dataParticle.vel[p] = Math::vec2(0);
      dataParticle.col[p] = Math::vec3(0, 0, 1);
      dataParticle.pos[p++] =
          Math::vec2(desc.cell_size + desc.particle_radius + desc.dx * i +
                         (j % 2 == 0 ? 0.0 : desc.particle_radius),
                     desc.cell_size + desc.particle_radius + desc.dy * j);
    }
  }
  for (uint32_t i = 0; i < desc.cellX; i++) {
    for (uint32_t j = 0; j < desc.cellY; j++) {
      CellType type = CellType::Fluid;
      if (i == 0 || i == desc.cellX - 1 || j == 0) type = CellType::Solid;
      cellData.s[i * desc.cellY + j] = uint32_t(type);
      cellData.type[i * desc.cellY + j] = type;
    }
  }
  // create device buffer
  RHI::Device* device = RHI::RHILayer::get()->getDevice();
  for (int i = 0; i < 2; ++i) {
    deviceData.cell_color_buffers[i] = device->createDeviceLocalBuffer(
        cellData.color.data(), cellData.color.size() * sizeof(Math::vec3),
        (uint32_t)RHI::BufferUsage::STORAGE);
    deviceData.particle_pos_buffers[i] = device->createDeviceLocalBuffer(
        dataParticle.pos.data(), dataParticle.pos.size() * sizeof(Math::vec2),
        (uint32_t)RHI::BufferUsage::STORAGE);
  }

  // init spatial hashing
  spatial_hashing.init_bounded(Math::uvec2{desc.gridX, desc.gridY},
                               1.f / desc.gridInvSpacing, desc.particle_num);
}

auto FLIP2D_host::forward(float dt) noexcept -> void {
  // Semi-implicit Euler
  for (uint32_t i = 0; i < desc.particle_num; ++i) {
    dataParticle.vel[i] += Math::vec2(0, dt * gravity_constant);
    dataParticle.pos[i] += dataParticle.vel[i] * dt;
  }
}

auto FLIP2D_host::collision() noexcept -> void {
  for (uint32_t i = 0; i < desc.particle_num; ++i) {
    // wall collision
    if (dataParticle.pos[i].x < desc.bound.pMin.x) {
      dataParticle.pos[i].x = desc.bound.pMin.x;
      dataParticle.vel[i].x = 0;
    }
    if (dataParticle.pos[i].x > desc.bound.pMax.x) {
      dataParticle.pos[i].x = desc.bound.pMax.x;
      dataParticle.vel[i].x = 0;
    }
    if (dataParticle.pos[i].y < desc.bound.pMin.y) {
      dataParticle.pos[i].y = desc.bound.pMin.y;
      dataParticle.vel[i].y = 0;
    }
    if (dataParticle.pos[i].y > desc.bound.pMax.y) {
      dataParticle.pos[i].y = desc.bound.pMax.y;
      dataParticle.vel[i].y = 0;
    }
  }
}

auto FLIP2D_host::push_particle_apart() noexcept -> void {
  float colorDiffusionCoeff = 0.001;
  float const minDist = 2.0 * desc.particle_radius;
  float const minDist2 = minDist * minDist;
  spatial_hashing.prepare(dataParticle.pos);
  int numIters = 2;
  // push particles apart
  for (int iter = 0; iter < numIters; iter++) {
    for (int i = 0; i < desc.particle_num; i++) {
       Math::vec2 pos = dataParticle.pos[i];
       int pxi = std::floor(pos.x * desc.gridInvSpacing);
       int pyi = std::floor(pos.y * desc.gridInvSpacing);
       int x0 = std::max(pxi - 1, 0);
       int y0 = std::max(pyi - 1, 0);
       int x1 = std::min(pxi + 1, int(desc.gridX) - 1);
       int y1 = std::min(pyi + 1, int(desc.gridY) - 1);
       for (int xi = x0; xi <= x1; xi++) {
         for (int yi = y0; yi <= y1; yi++) {
           int cellNr = xi * desc.gridY + yi;
           int first = spatial_hashing.grid.partial_sums[cellNr];
           int last = spatial_hashing.grid.partial_sums[cellNr + 1];
           for (int j = first; j < last; j++) {
             int id = spatial_hashing.grid.dense_ids[j];
            if (id == i) continue;
            Math::vec2 q = dataParticle.pos[id];
            Math::vec2 d = q - pos;
            float d2 = d.x * d.x + d.y * d.y;
            if (d2 > minDist2 || d2 == 0.0) continue;
            float dist = std::sqrt(d2);
            float s = 0.5 * (minDist - dist) / dist;
            d *= s;
            dataParticle.pos[i] -= d;
            dataParticle.pos[id] += d;

            // diffuse colors

            for (int k = 0; k < 3; k++) {
              float color0 = dataParticle.col[i].data[k];
              float color1 = dataParticle.col[id].data[k];
              float color = (color0 + color1) * 0.5;
              dataParticle.col[i].data[k] =
                  color0 + (color - color0) * colorDiffusionCoeff;
              dataParticle.col[id].data[k] =
                  color1 + (color - color1) * colorDiffusionCoeff;
            }
          }
        }
      }
    }
  }
}

auto FLIP2D_host::simulate(float dt) noexcept -> void {
  uint32_t sub_step_num = 1;
  float sdt = dt / sub_step_num;
  for (uint32_t i = 0; i < sub_step_num; ++i) {
    forward(sdt);   // forward simulate particles
    push_particle_apart();
    collision();    // collision handling
    transfer_vel_to_cell();
    update_density();
    solve_incompressibility(sdt, 1.9, true);
    transfer_vel_to_particle(0.9);
  }
  update_color();
}

auto FLIP2D_host::transfer_vel_to_cell() noexcept -> void {
  const uint32_t n = desc.cellY;
  const float h = desc.h;
  const float h1 = desc.cellInvSpacing;
  const float h2 = 0.5 * h;
  // store previous cell velocity for FLIP
  cellData.prev_u = cellData.u;
  cellData.prev_v = cellData.v;
  // clean up velocity data
  std::fill(cellData.du.begin(), cellData.du.end(), 0);
  std::fill(cellData.dv.begin(), cellData.dv.end(), 0);
  std::fill(cellData.u.begin(), cellData.u.end(), 0);
  std::fill(cellData.v.begin(), cellData.v.end(), 0);
  // Set solid cells to solid, non-solid cells to air
  // If a non-solid cell has a particle in it, set to fluid
  for (int i = 0; i < desc.cellCount; i++)
    cellData.type[i] = (cellData.s[i] == 0.0 ? CellType::Solid : CellType::Air);
  for (int i = 0; i < desc.particle_num; i++) {
    Math::vec2 const& pos = dataParticle.pos[i];
    int xi = std::clamp(int(std::floor(pos.x * h1)), 0, int(desc.cellX) - 1);
    int yi = std::clamp(int(std::floor(pos.y * h1)), 0, int(desc.cellY) - 1);
    int cellNr = xi * n + yi;
    if (cellData.type[cellNr] == CellType::Air) {
      cellData.type[cellNr] = CellType::Fluid;
      if (yi == desc.cellY - 1u) {
        float a = 1.;
      }
    }
  }
  // Handle u and v respectively
  for (uint32_t component = 0; component < 2; component++) {
    // Defining the offset needed to find the stagged cell id
    float dx = component == 0 ? 0.0 : h2;
    float dy = component == 0 ? h2 : 0.0;
    std::vector<float>& f = component == 0 ? cellData.u : cellData.v;
    std::vector<float>& d = component == 0 ? cellData.du : cellData.dv;
    // iterating through each particle to do Particle2Grid
    for (uint32_t i = 0; i < desc.particle_num; i++) {
      // clamp the position to valid pos
      float x = dataParticle.pos[i].x;
      float y = dataParticle.pos[i].y;
      x = std::clamp(x, h, (desc.cellX - 1) * h);
      y = std::clamp(y, h, (desc.cellY - 1) * h);
      // get the corresponding stagged grid id
      const int x0 =
          std::min(int(std::floor((x - dx) * h1)), int(desc.cellX) - 2);
      const int x1 = std::min(x0 + 1, int(desc.cellX) - 2);
      const int y0 =
          std::min(int(std::floor((y - dy) * h1)), int(desc.cellY) - 2);
      const int y1 = std::min(int(y0 + 1), int(desc.cellY) - 2);
      // compute bilinear interpolation weights
      const float tx = ((x - dx) - x0 * h) * h1;
      const float ty = ((y - dy) - y0 * h) * h1;
      const float sx = 1.0 - tx;
      const float sy = 1.0 - ty;
      const float d0 = sx * sy;
      const float d1 = tx * sy;
      const float d2 = tx * ty;
      const float d3 = sx * ty;
      // cell ids
      const int nr0 = x0 * n + y0;
      const int nr1 = x1 * n + y0;
      const int nr2 = x1 * n + y1;
      const int nr3 = x0 * n + y1;
      // splat the velocity to neighboring cells
      const float pv = dataParticle.vel[i].data[component];
      f[nr0] += pv * d0;    d[nr0] += d0;
      f[nr1] += pv * d1;    d[nr1] += d1;
      f[nr2] += pv * d2;    d[nr2] += d2;
      f[nr3] += pv * d3;    d[nr3] += d3;
    }
    // For each cell divide the sum of weights 
    for (uint32_t i = 0; i < f.size(); i++) {
      if (d[i] > 0.0) f[i] /= d[i];
    }
    // restore solid cells if it and its stagger grid neighbor are both solid
    for (uint32_t i = 0; i < desc.cellX; i++) {
      for (uint32_t j = 0; j < desc.cellY; j++) {
        bool solid = cellData.type[i * n + j] == CellType::Solid;
        if (solid ||
            (i > 0 && cellData.type[(i - 1) * n + j] == CellType::Solid))
          cellData.u[i * n + j] = cellData.prev_u[i * n + j];
        if (solid || (j > 0 && cellData.type[i * n + j - 1] == CellType::Solid))
          cellData.v[i * n + j] = cellData.prev_v[i * n + j];
      }
    }
  }
}

auto FLIP2D_host::transfer_vel_to_particle(float flipRatio) noexcept -> void {
  // Interpolate the cell velocity to renew particle velocities
  const uint32_t n = desc.cellY;
  const float h = desc.h;
  const float h1 = desc.cellInvSpacing;
  const float h2 = 0.5 * h;
  for (uint32_t component = 0; component < 2; component++) {
    // Defining the offset needed to find the stagged cell id
    const float dx = component == 0 ? 0.0 : h2;
    const float dy = component == 0 ? h2 : 0.0;
    // handle u & v respectively
    std::vector<float>& f = component == 0 ? cellData.u : cellData.v;
    std::vector<float>& prevF =
        component == 0 ? cellData.prev_u : cellData.prev_v;
    // iterating through each particle to do Grid2Particle
    for (uint32_t i = 0; i < desc.particle_num; i++) {
      // clamp the position to valid pos
      float x = dataParticle.pos[i].x;
      float y = dataParticle.pos[i].y;
      x = std::clamp(x, h, (desc.cellX - 1) * h);
      y = std::clamp(y, h, (desc.cellY - 1) * h);
      // get the corresponding stagged grid id
      const int x0 =
          std::min(int(std::floor((x - dx) * h1)), int(desc.cellX - 2));
      const int x1 = std::min(x0 + 1, int(desc.cellX) - 2);
      const int y0 =
          std::min(int(std::floor((y - dy) * h1)), int(desc.cellY) - 2);
      const int y1 = std::min(int(y0 + 1), int(desc.cellY) - 2);
      // compute bilinear interpolation weights
      const float tx = ((x - dx) - x0 * h) * h1;
      const float ty = ((y - dy) - y0 * h) * h1;
      const float sx = 1.0 - tx;
      const float sy = 1.0 - ty;
      const float d0 = sx * sy;
      const float d1 = tx * sy;
      const float d2 = tx * ty;
      const float d3 = sx * ty;
      // cell ids
      const int nr0 = x0 * n + y0;
      const int nr1 = x1 * n + y0;
      const int nr2 = x1 * n + y1;
      const int nr3 = x0 * n + y1;
      // validate the weights by checking whether the cell is valid or undefined.
      // staggered grid need at least one of the adjecent cell be valid
      const int offset = component == 0 ? n : 1;
      const float w0 = d0 * (cellData.type[nr0] != CellType::Air ||
                               cellData.type[nr0 - offset] != CellType::Air
                           ? 1.0 : 0.0);
      const float w1 = d1 * (cellData.type[nr1] != CellType::Air ||
                               cellData.type[nr1 - offset] != CellType::Air
                           ? 1.0 : 0.0);
      const float w2 = d2 * (cellData.type[nr2] != CellType::Air ||
                               cellData.type[nr2 - offset] != CellType::Air
                           ? 1.0 : 0.0);
      const float w3 = d3 * (cellData.type[nr3] != CellType::Air ||
                               cellData.type[nr3 - offset] != CellType::Air
                           ? 1.0 : 0.0);
      const float d = w0 + w1 + w2 + w3;
      // update velocity of each particle by transfer from grid
      const float v = dataParticle.vel[i].data[component];
      if (d > 0.0) {    // if at least one cell is valid, do updating
        const float picV =
            (w0 * f[nr0] + w1 * f[nr1] + w2 * f[nr2] + w3 * f[nr3]) / d;
        const float corr =
            (w0 * (f[nr0] - prevF[nr0]) + w1 * (f[nr1] - prevF[nr1]) +
                      w2 * (f[nr2] - prevF[nr2]) + w3 * (f[nr3] - prevF[nr3])) /
                     d;
        const float flipV = v + corr;
        if (picV > 0) {
          float a = 1.f;
        }
        // blending pic and flip results with flipRatio
        dataParticle.vel[i].data[component] =
            (1.0 - flipRatio) * picV + flipRatio * flipV;
      }
    }
  }
}

auto FLIP2D_host::update_density() noexcept -> void {
  uint32_t n = desc.cellY;
  float h = desc.h;
  float h1 = desc.cellInvSpacing;
  float h2 = 0.5 * h;

  std::vector<float>& d = cellData.particle_density;
  std::fill(d.begin(), d.end(), 0);

  for (uint32_t i = 0; i < desc.particle_num; i++) {
    Math::vec2 const& pos = dataParticle.pos[i];
    float x = std::clamp(float(pos.x), h, (desc.cellX - 1) * h);
    float y = std::clamp(float(pos.y), h, (desc.cellY - 1) * h);

    float x0 = std::floor((x - h2) * h1);
    float tx = ((x - h2) - x0 * h) * h1;
    float x1 = std::min(float(x0 + 1.f), float(desc.cellX - 2.f));

    float y0 = std::floor((y - h2) * h1);
    float ty = ((y - h2) - y0 * h) * h1;
    float y1 = std::min(float(y0 + 1), float(desc.cellY - 2.f));

    float sx = 1.0 - tx;
    float sy = 1.0 - ty;

    if (x0 < desc.cellX && y0 < desc.cellY) d[x0 * n + y0] += sx * sy;
    if (x1 < desc.cellX && y0 < desc.cellY) d[x1 * n + y0] += tx * sy;
    if (x1 < desc.cellX && y1 < desc.cellY) d[x1 * n + y1] += tx * ty;
    if (x0 < desc.cellX && y1 < desc.cellY) d[x0 * n + y1] += sx * ty;
  }

  if (cellData.rest_density == 0.0) {
    float sum = 0.0;
    float numFluidCells = 0;
    for (uint32_t i = 0; i < desc.cellCount; i++) {
      if (cellData.type[i] == CellType::Fluid) {
        sum += d[i];
        numFluidCells++;
      }
    }

    if (numFluidCells > 0)
        cellData.rest_density = sum / numFluidCells;
  }
}

auto FLIP2D_host::solve_incompressibility(float dt, float over_relaxation,
                                          bool compensate_drift) noexcept
    -> void {
  // make the grid velocities incompressible
  std::fill(cellData.p.begin(), cellData.p.end(), 0);
  cellData.prev_u = cellData.u;
  cellData.prev_v = cellData.v;

  uint32_t n = desc.cellY;
  float cp = desc.density * desc.h / dt;

  for (uint32_t iter = 0; iter < desc.num_pressure_iters; iter++) {
    for (uint32_t i = 1; i < desc.cellX - 1; i++) {
      for (uint32_t j = 1; j < desc.cellY - 1; j++) {
        if (cellData.type[i * n + j] != CellType::Fluid) continue;

        int center = i * n + j;
        int left = (i - 1) * n + j;
        int right = (i + 1) * n + j;
        int bottom = i * n + j - 1;
        int top = i * n + j + 1;

        float s = cellData.s[center];
        float sx0 = cellData.s[left];
        float sx1 = cellData.s[right];
        float sy0 = cellData.s[bottom];
        float sy1 = cellData.s[top];
        s = sx0 + sx1 + sy0 + sy1;
        if (s == 0.0) continue;

        float div = cellData.u[right] - cellData.u[center] + cellData.v[top] -
                    cellData.v[center];

        if (cellData.rest_density > 0.0 && compensate_drift) {
          float k = 1.0;
          float compression = cellData.particle_density[i * n + j] - cellData.rest_density;
          if (compression > 0.0) div = div - k * compression;
        }

        float p = -div / s;
        p *= over_relaxation;
        cellData.p[center] += cp * p;

        cellData.u[center] -= sx0 * p;
        cellData.u[right] += sx1 * p;
        cellData.v[center] -= sy0 * p;
        cellData.v[top] += sy1 * p;
      }
    }
  }
}

auto FLIP2D_host::update_color() noexcept -> void {
  float const h1 = desc.cellInvSpacing;
  for (uint32_t i = 0; i < desc.particle_num; i++) {
    float s = 0.01f;
    dataParticle.col[i].x = Math::clamp(dataParticle.col[i].x - s, 0.0, 1.0);
    dataParticle.col[i].y = Math::clamp(dataParticle.col[i].y - s, 0.0, 1.0);
    dataParticle.col[i].z = Math::clamp(dataParticle.col[i].z + s, 0.0, 1.0);

    int xi = std::clamp(int(std::floor(dataParticle.pos[i].x * h1)), 1,
                        int(desc.cellX) - 1);
    int yi = std::clamp(int(std::floor(dataParticle.pos[i].y * h1)), 1,
                        int(desc.cellY) - 1);
    int cellNr = xi * desc.cellY + yi;

    float d0 = cellData.rest_density;
    if (d0 > 0.0) {
      float relDensity = cellData.particle_density[cellNr] / d0;
      if (relDensity < 0.7) {
        float s = 0.8;
        dataParticle.col[i] = Math::vec3(s, s, 1.0);
      }
    }
  }
  for (uint32_t i = 0; i < desc.cellCount; ++i) {
    if (cellData.type[i] == CellType::Solid) {
      cellData.color[i] = Math::vec3(0.5);
    } else if (cellData.type[i] == CellType::Fluid) {
      //var d = this.particleDensity[i];
      //if (this.particleRestDensity > 0.0) d /= this.particleRestDensity;
      //this.setSciColor(i, d, 0.0, 2.0);
      cellData.color[i] = Math::vec3(0.1, 0.1, 0.5);
    } else {
      // var d = this.particleDensity[i];
      // if (this.particleRestDensity > 0.0) d /= this.particleRestDensity;
      // this.setSciColor(i, d, 0.0, 2.0);
      cellData.color[i] = Math::vec3(0);
    }

  }
}
}  // namespace FLIP
#undef float