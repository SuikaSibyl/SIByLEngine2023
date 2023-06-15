#pragma once
#include <array>
#include <vector>
#include <algorithm>
#include <filesystem>
#include <istream>
#include <streambuf>
#include <IO/SE.Core.IO.hpp>
#include <SE.Math.Geometric.hpp>
#include <SE.GFX.hpp>

using namespace SIByL;

namespace ElasticDemo {
struct Vertex {
  Math::vec3 X;
  Math::vec3 x;
  Math::vec3 v;
  Math::vec3 f;
  float mass;
};

struct Tetrahedral {
  std::array<uint32_t, 4> indices;
  Math::mat4 invDm;
  float volume;
};

struct membuf : std::streambuf {
  membuf(char* begin, char* end) { this->setg(begin, begin, end); }
};

struct DemoScript {
  void onStart(GFX::Scene* scene);

  Platform::Input* input;

  double mass = 1.f;

  void Timestep(double delta);

  float muN = 0.5f;
  float muT = 0.5f;

  float lambda = 20000.f;
  float mu = 5000.f;
  float damp = 0.999f;
  float floor_y = -3;
  float timestep = 0.001;

  Math::mat4 EdgeMatrix(Tetrahedral const& tet);

  void onUpdate(int i);

  void Exit() {
    vertex_buffers[0] = nullptr;
    vertex_buffers[1] = nullptr;
  }

  std::unique_ptr<RHI::Buffer> vertex_buffers[2];

  GFX::Mesh mesh;
  std::vector<Math::vec3> vertex_buffer_cpu;
  std::vector<Vertex> vertices;
  std::vector<Tetrahedral> tetrahedrals;

  size_t vertex_count;
};
}