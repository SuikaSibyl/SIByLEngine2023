#pragma once
#include <vector>
#include <SE.RHI.hpp>

namespace SIByL::GFX {
/** Mesh data layout */
SE_EXPORT struct MeshDataLayout {
  /** info types in vertex */
  enum struct VertexInfo {
    POSITION,
    NORMAL,
    TANGENT,
    UV,
    COLOR,
    CUSTOM,
  };
  /** an entry of the layout */
  struct Entry {
    RHI::VertexFormat format;
    VertexInfo info;
  };
  /* the list of vertex layout */
  std::vector<Entry> layout;
  /* index format */
  RHI::IndexFormat format;
};

/** A setting config to guide loading of mesh resource */
SE_EXPORT struct MeshLoaderConfig {
  MeshDataLayout layout = {};
  bool usePositionBuffer = true;
  bool residentOnHost = true;
  bool residentOnDevice = false;
  bool deduplication = true;
};

/** A setting config contained in GFXManager. Guide:
 * 1. the loading setting of mesh resource. */
SE_EXPORT struct GFXConfig {
  /** config to load a mesh */
  MeshLoaderConfig meshLoaderConfig = {};
  /** global gfx configuration */
  static GFXConfig* globalConfig;
};
}  // namespace SIByL::GFX