#pragma once

#include <cstdint>
#include <format>
#include <future>
#include <optional>

#include <Print/SE.Core.Log.hpp>
#include <Misc/SE.Core.Misc.hpp>
#include <Memory/SE.Core.Memory.hpp>
#include <Resource/SE.Core.Resource.hpp>
#include <SE.RHI.hpp>
#include <SE.GFX.hpp>

namespace SIByL::Video {
struct AVOpaqueData;

SE_EXPORT struct VideoDecoder {
  VideoDecoder();
  ~VideoDecoder();

  VideoDecoder(VideoDecoder const&) = delete;
  auto operator=(VideoDecoder const&) -> VideoDecoder& = delete;
  VideoDecoder(VideoDecoder&&);
  auto operator=(VideoDecoder&&) -> VideoDecoder&;

  auto open(char const* filepath) noexcept -> bool;
  auto readFrame() noexcept -> bool;
  auto close() noexcept -> bool;

  uint32_t width, height;
  Core::Buffer data;

  AVOpaqueData* av_opaque = nullptr;
  int video_stream_index;

  GFX::Buffer* device_buffer;
  GFX::Texture* device_texture;
  std::unique_ptr<Core::Timer> timer;

  int repeat = -1;
  int64_t curr_pts;
  double pt_in_seconds;
  double actual_seconds;
};
}  // namespace SIByL::Video

namespace SIByL::GFX {
SE_EXPORT struct VideoClip : public Core::Resource {
  /** ctors & rval copies */
  VideoClip() = default;
  VideoClip(VideoClip&& vc) = default;
  VideoClip(VideoClip const& vc) = delete;
  auto operator=(VideoClip&& vc) -> VideoClip& = default;
  auto operator=(VideoClip const& vc) -> VideoClip& = delete;
  /** serialize */
  auto serialize() noexcept -> void;
  /** deserialize */
  auto deserialize(RHI::Device* device, Core::ORID orid) noexcept
      -> void;
  /** get name */
  virtual auto getName() const noexcept -> char const* override {
    return name.c_str();
  }
  /** active */
  bool active = false;
  /** resrouce GUID */
  Core::GUID guid;
  /** resrouce ORID */
  Core::ORID orid = Core::INVALID_ORID;
  /** name */
  std::string name;
  /** path string */
  std::optional<std::string> resourcePath;
  /** the video decoder */
  Video::VideoDecoder decoder;
};

SE_EXPORT struct VideExtension : public GFX::Extension {
  auto registerVideoClipResource(char const* filepath) noexcept -> Core::GUID;
  auto requestOfflineVideoClipResource(Core::ORID orid) noexcept -> Core::GUID;
  virtual auto foo(uint32_t id, void* data) noexcept -> void*;
 protected:
  virtual auto startUp() noexcept -> void override;
  virtual auto onUpdate() noexcept -> void override;
};
}  // namespace SIByL::Video