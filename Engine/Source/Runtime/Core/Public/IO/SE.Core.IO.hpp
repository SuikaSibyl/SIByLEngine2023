#pragma once

#include <format>
#include <fstream>
#include <string>
#include <vector>

#include <Print/SE.Core.Log.hpp>
#include <Memory/SE.Core.Memory.hpp>
#include <IO/SE.Core.IO.hpp>

namespace SIByL::Core {
SE_EXPORT inline auto syncReadFile(char const* path, Buffer& buffer) noexcept
    -> bool {
  LogManager::Log(std::string(path));
  std::ifstream ifs(path, std::ifstream::binary);
  if (ifs.is_open()) {
    ifs.seekg(0, std::ios::end);
    size_t size = size_t(ifs.tellg());
    buffer = Buffer(size + 1);
    buffer.size = size;
    ifs.seekg(0);
    ifs.read(reinterpret_cast<char*>(buffer.data), size);
    ((char*)buffer.data)[size] = '\0';
    ifs.close();
  } else {
    LogManager::Error(std::format(
        "Core.IO:SyncRW::syncReadFile() failed, file \'{}\' not found.", path));
  }
  return false;
}

SE_EXPORT inline auto syncWriteFile(char const* path, Buffer& buffer) noexcept
    -> bool {
  std::ofstream ofs(path, std::ios::out | std::ios::binary);
  if (ofs.is_open()) {
    ofs.write((char*)buffer.data, buffer.size);
    ofs.close();
  } else {
    LogManager::Error(std::format(
        "Core.IO:SyncRW::syncWriteFile() failed, file \'{}\' open failed.",
        path));
  }
  return false;
}

SE_EXPORT inline auto syncWriteFile(char const* path,
                                 std::vector<Buffer*> const& buffers) noexcept
    -> bool {
  std::ofstream ofs(path, std::ios::out | std::ios::binary);
  if (ofs.is_open()) {
    for (auto* buffer : buffers) ofs.write((char*)buffer->data, buffer->size);
    ofs.close();
  } else {
    LogManager::Error(std::format(
        "Core.IO:SyncRW::syncWriteFile() failed, file \'{}\' open failed.",
        path));
  }
  return false;
}
}  // namespace SIByL::Core