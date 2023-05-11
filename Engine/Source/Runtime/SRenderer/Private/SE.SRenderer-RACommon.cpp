#include <Core/SE.SRenderer-RACommon.hpp>

namespace SIByL {
auto RACommon::DrawcallData::buildIndirectDrawcalls() noexcept -> void {
  // release the previous buffer
  if (all_drawcall_device) all_drawcall_device->release();
  all_drawcall_host.clear();
  bsdfs_drawcalls.clear();
  // opaque pass
  opaque_drawcall = IndirectDrawcall{
      (uint32_t)(all_drawcall_host.size() * sizeof(DrawIndexedIndirectEX)),
      (uint32_t)opaque_drawcalls_host.size()};
  all_drawcall_host.insert(all_drawcall_host.end(),
                           opaque_drawcalls_host.begin(),
                           opaque_drawcalls_host.end());
  // opaque pass
  alphacut_drawcall = IndirectDrawcall{
      (uint32_t)(all_drawcall_host.size() * sizeof(DrawIndexedIndirectEX)),
      (uint32_t)alphacut_drawcalls_host.size()};
  all_drawcall_host.insert(all_drawcall_host.end(),
                           alphacut_drawcalls_host.begin(),
                           alphacut_drawcalls_host.end());
  // bsdf passes
  for (auto const& iter : bsdf_drawcalls_host) {
    std::vector<DrawIndexedIndirectEX> const& drawcall_host = iter.second;
    IndirectDrawcall bsdf_drawcall = IndirectDrawcall{
        (uint32_t)(all_drawcall_host.size() * sizeof(DrawIndexedIndirectEX)),
        (uint32_t)drawcall_host.size()};
    all_drawcall_host.insert(all_drawcall_host.end(), drawcall_host.begin(),
                             drawcall_host.end());
    bsdfs_drawcalls[iter.first] = bsdf_drawcall;
  }
  // create buffer
  // all_drawcall_device
  Core::GUID guid =
      Core::ResourceManager::get()->requestRuntimeGUID<GFX::Buffer>();
  GFX::GFXManager::get()->registerBufferResource(
      guid, all_drawcall_host.data(),
      all_drawcall_host.size() * sizeof(DrawIndexedIndirectEX),
      (uint32_t)RHI::BufferUsage::STORAGE |
          (uint32_t)RHI::BufferUsage::INDIRECT);
  all_drawcall_device =
      Core::ResourceManager::get()->getResource<GFX::Buffer>(guid);
}
RACommon* RACommon::singleton = nullptr;
}