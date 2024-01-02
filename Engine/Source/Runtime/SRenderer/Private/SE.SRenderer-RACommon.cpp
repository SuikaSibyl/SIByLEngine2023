#include <Core/SE.SRenderer-RACommon.hpp>

namespace SIByL {
RACommon::RACommon() { 
  singleton = this; 
  structured_drawcalls.all_drawcalls.usage = 
    (uint32_t)RHI::BufferUsage::STORAGE |
    (uint32_t)RHI::BufferUsage::INDIRECT;
}

auto RACommon::DrawcallData::invalidIndirectDrawcalls() noexcept -> void {
  if (!isDirty) return;
  else isDirty = false;
  // release the previous buffer
  all_drawcalls.buffer_host.clear();
  bsdfs_drawcalls.clear();
  // opaque pass
  opaque_drawcall = IndirectDrawcall{
      (uint32_t)(all_drawcalls.buffer_host.size() * sizeof(DrawIndexedIndirectEX)),
      (uint32_t)opaque_drawcalls_host.size()};
  all_drawcalls.buffer_host.insert(all_drawcalls.buffer_host.end(),
    opaque_drawcalls_host.begin(), opaque_drawcalls_host.end());
  // opaque pass
  alphacut_drawcall = IndirectDrawcall{(uint32_t)(all_drawcalls.buffer_host.size() *
    sizeof(DrawIndexedIndirectEX)), (uint32_t)alphacut_drawcalls_host.size()};
  all_drawcalls.buffer_host.insert(all_drawcalls.buffer_host.end(),
    alphacut_drawcalls_host.begin(), alphacut_drawcalls_host.end());
  // bsdf passes
  for (auto const& iter : bsdf_drawcalls_host) {
    std::vector<DrawIndexedIndirectEX> const& drawcall_host = iter.second;
    IndirectDrawcall bsdf_drawcall = IndirectDrawcall{
        (uint32_t)(all_drawcalls.buffer_host.size() * sizeof(DrawIndexedIndirectEX)),
        (uint32_t)drawcall_host.size()};
    all_drawcalls.buffer_host.insert(all_drawcalls.buffer_host.end(), drawcall_host.begin(),
                             drawcall_host.end());
    bsdfs_drawcalls[iter.first] = bsdf_drawcall;
  }
  // create buffer
  // all_drawcall_device
  all_drawcalls.stamp++;
  all_drawcalls.buffer_device.swap();
  RHI::Device* device = GFX::GFXManager::get()->rhiLayer->getDevice();
  all_drawcalls.update_to_device(device);
}

RACommon* RACommon::singleton = nullptr;
}