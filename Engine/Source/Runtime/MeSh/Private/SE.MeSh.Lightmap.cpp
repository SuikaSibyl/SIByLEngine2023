#include "../Public/SE.MeSh.Lightmap.hpp"
#include <xatlas.h>

namespace SIByL::MeSh {
class Stopwatch {
 public:
  Stopwatch() { reset(); }
  void reset() { m_start = clock(); }
  double elapsed() const {
    return (clock() - m_start) * 1000.0 / CLOCKS_PER_SEC;
  }

 private:
  clock_t m_start;
};

// May be called from any thread.
static bool ProgressCallback(xatlas::ProgressCategory category, int progress, void *userData) {
  Stopwatch *stopwatch = (Stopwatch *)userData;
  static std::mutex progressMutex;
  std::unique_lock<std::mutex> lock(progressMutex);
  if (progress == 0) stopwatch->reset();
  printf("\r   %s [", xatlas::StringForEnum(category));
  for (int i = 0; i < 10; i++) printf(progress / ((i + 1) * 10) ? "*" : " ");
  printf("] %d%%", progress);
  fflush(stdout);
  if (progress == 100)
    printf("\n      %.2f seconds (%g ms) elapsed\n",
           stopwatch->elapsed() / 1000.0, stopwatch->elapsed());
  return true;
}

auto LightmapBuilder::build(GFX::Scene &scene) noexcept -> void { 
  // Create empty atlas.
  xatlas::Atlas *atlas = xatlas::Create();
  // Set progress callback.
  Stopwatch globalStopwatch, stopwatch;
  xatlas::SetProgressCallback(atlas, ProgressCallback, &stopwatch);
  // Counting the number of meshes
  uint32_t mesh_count = 0;
  for (auto go_handle : scene.gameObjects) {
    auto *go = scene.getGameObject(go_handle.first);
    auto *meshref = go->getEntity().getComponent<GFX::MeshReference>();
    auto *meshrenderer = go->getEntity().getComponent<GFX::MeshRenderer>();
    if (meshref == nullptr || meshrenderer == nullptr) continue;
    mesh_count += meshref->mesh->submeshes.size(); }
  // Add meshes to atlas.
  uint32_t totalVertices = 0, totalFaces = 0;
  for (auto go_handle : scene.gameObjects) {
    auto *go = scene.getGameObject(go_handle.first);
    auto *tagref = go->getEntity().getComponent<GFX::TagComponent>();
    auto *meshref = go->getEntity().getComponent<GFX::MeshReference>();
    auto *meshrenderer = go->getEntity().getComponent<GFX::MeshRenderer>();
    if (meshref == nullptr || meshrenderer == nullptr) continue;
    auto &vertex_buffer = meshref->mesh->vertexBuffer_host;
    auto &index_buffer = meshref->mesh->indexBuffer_host;
    // for every submesh create an mesh object
    for (auto &submesh : meshref->mesh->submeshes) {
      const size_t vdata_offset = size_t(vertex_buffer.data) + submesh.baseVertex * 44;
      const size_t idata_offset = size_t(index_buffer.data) + submesh.offset * 4;
      // create xatlas::MeshDecl object
      xatlas::MeshDecl meshDecl;
      meshDecl.vertexCount = vertex_buffer.size / 44;
      meshDecl.vertexPositionData = (void *)vdata_offset;
      meshDecl.vertexPositionStride = 44;
      meshDecl.vertexNormalData = (void *)(vdata_offset + 12);
      meshDecl.vertexNormalStride = 44;
      meshDecl.vertexUvData = (void *)(vdata_offset + 36);
      meshDecl.vertexUvStride = 44;
      meshDecl.indexCount = index_buffer.size / sizeof(uint32_t);
      meshDecl.indexData = (void *)idata_offset;
      meshDecl.indexFormat = xatlas::IndexFormat::UInt32;
      
      auto error = xatlas::AddMesh(atlas, meshDecl, mesh_count);
      if (error != xatlas::AddMeshError::Success) {
        xatlas::Destroy(atlas);
        Core::LogManager::Error("rError adding mesh: " + 
            tagref->name+ xatlas::StringForEnum(error));
        return;
      }
      
      totalVertices += meshDecl.vertexCount;
      if (meshDecl.faceCount > 0) totalFaces += meshDecl.faceCount;
      else totalFaces += meshDecl.indexCount / 3;  
      // Assume triangles if MeshDecl::faceCount not specified.
    }
  }
  // Not necessary. Only called here so geometry totals are
  // printed after the AddMesh progress indicator.
  xatlas::AddMeshJoin(atlas);
  printf("   %u total vertices\n", totalVertices);
  printf("   %u total faces\n", totalFaces);
  // Generate atlas.
  printf("Generating atlas\n");
  xatlas::PackOptions pack_options{};
  pack_options.padding = 4;
  xatlas::Generate(atlas, {}, pack_options);
  printf("   %d charts\n", atlas->chartCount);
  printf("   %d atlases\n", atlas->atlasCount);
  for (uint32_t i = 0; i < atlas->atlasCount; i++)
    printf("      %d: %0.2f%% utilization\n", i, atlas->utilization[i] * 100.0f);
  printf("   %ux%u resolution\n", atlas->width, atlas->height);
  totalVertices = 0;
  for (uint32_t i = 0; i < atlas->meshCount; i++) {
    const xatlas::Mesh &mesh = atlas->meshes[i];
    totalVertices += mesh.vertexCount;
  }
  printf("   %u total vertices\n", totalVertices);
  printf("%.2f seconds (%g ms) elapsed total\n",
  globalStopwatch.elapsed() / 1000.0, globalStopwatch.elapsed());
  // Write meshes.
  uint32_t submeshID = 0;
  uint32_t firstVertex = 0;
  for (auto go_handle : scene.gameObjects) {
    auto *go = scene.getGameObject(go_handle.first);
    auto *tagref = go->getEntity().getComponent<GFX::TagComponent>();
    auto *meshref = go->getEntity().getComponent<GFX::MeshReference>();
    auto *meshrenderer = go->getEntity().getComponent<GFX::MeshRenderer>();
    std::vector<float> new_vertex_buffer;
    std::vector<float> new_pos_buffer;
    std::vector<float> uv2_buffer;
    std::vector<uint32_t> indices_buffer;
    if (meshref == nullptr || meshrenderer == nullptr) continue;
    // for every submesh create an mesh object
    for (auto &submesh : meshref->mesh->submeshes) {
      const xatlas::Mesh &mesh = atlas->meshes[submeshID++];
      const size_t offset = new_vertex_buffer.size();
      const size_t input = (size_t)meshref->mesh->vertexBuffer_host.data +
                           submesh.baseVertex * sizeof(float) * 11;
      new_vertex_buffer.resize(offset + mesh.vertexCount * 11);
      new_pos_buffer.resize(offset + mesh.vertexCount * 3);
      for (uint32_t v = 0; v < mesh.vertexCount; v++) {
        const xatlas::Vertex &vertex = mesh.vertexArray[v];
        memcpy(&new_vertex_buffer[offset + vertex.xref * 11], 
               (void *)(input + vertex.xref * sizeof(float) * 11),
               sizeof(float) * 11);
        memcpy(&new_pos_buffer[offset + vertex.xref * 3],
               (void *)(input + vertex.xref * sizeof(float) * 11),
               sizeof(float) * 3);
        uv2_buffer.push_back(vertex.uv[0] / atlas->width);
        uv2_buffer.push_back(vertex.uv[1] / atlas->height);
      }
      for (uint32_t f = 0; f < mesh.indexCount; f += 3) {
        for (uint32_t j = 0; j < 3; j++) {
          const uint32_t index = mesh.indexArray[f + j];  // 1-indexed
          indices_buffer.emplace_back(index);
        }
      }
    }
    // copy uv2 buffer
    meshref->mesh->uv2Buffer_host = Core::Buffer(
      uv2_buffer.size() * sizeof(float));
    memcpy(meshref->mesh->uv2Buffer_host.data, uv2_buffer.data(),
      uv2_buffer.size() * sizeof(float));
    // copy index buffer
    meshref->mesh->indexBuffer_host =
      Core::Buffer(indices_buffer.size() * sizeof(uint32_t));
    memcpy(meshref->mesh->indexBuffer_host.data, indices_buffer.data(),
      indices_buffer.size() * sizeof(uint32_t));
    // copy vertex buffer
    meshref->mesh->vertexBuffer_host =
      Core::Buffer(new_vertex_buffer.size() * sizeof(float));
    memcpy(meshref->mesh->vertexBuffer_host.data, new_vertex_buffer.data(),
      new_vertex_buffer.size() * sizeof(float));
    // copy vertex buffer
    meshref->mesh->positionBuffer_host =
      Core::Buffer(new_pos_buffer.size() * sizeof(float));
    memcpy(meshref->mesh->positionBuffer_host.data, new_pos_buffer.data(),
      new_pos_buffer.size() * sizeof(float));
    // save
    meshref->mesh->serialize();
  }
  // Cleanup.
  xatlas::Destroy(atlas);
}
}