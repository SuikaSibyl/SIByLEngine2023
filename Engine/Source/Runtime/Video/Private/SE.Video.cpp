#include "../Public/SE.Video.hpp"
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
}
#include <IO/SE.Core.IO.hpp>

namespace SIByL::Video {
struct AVOpaqueData {
  AVRational time_base;
  AVFormatContext* av_format_ctx = nullptr;
  AVCodecContext* av_codec_ctx = nullptr;
  AVFrame* av_frame = nullptr;
  AVPacket* av_packet = nullptr;
  SwsContext* sws_scaler_ctx = nullptr;
};

VideoDecoder::VideoDecoder() { 
	av_opaque = new AVOpaqueData(); }

VideoDecoder::~VideoDecoder() {
  if (av_opaque) delete av_opaque;
}

VideoDecoder::VideoDecoder(VideoDecoder&& a) {
  width = a.width;
  height = a.height;
  data = std::move(a.data);
  av_opaque = a.av_opaque;
  a.av_opaque = nullptr;
  video_stream_index = video_stream_index;
  device_buffer = a.device_buffer;
  device_texture = a.device_texture;
  timer = std::move(timer);
  repeat = a.repeat;
  curr_pts = a.curr_pts;
  pt_in_seconds = a.pt_in_seconds;
  actual_seconds = a.actual_seconds;
}

auto VideoDecoder::operator=(VideoDecoder&& a) -> VideoDecoder& {
  width = a.width;
  height = a.height;
  data = std::move(a.data);
  av_opaque = a.av_opaque;
  a.av_opaque = nullptr;
  video_stream_index = video_stream_index;
  device_buffer = a.device_buffer;
  device_texture = a.device_texture;
  timer = std::move(timer);
  repeat = a.repeat;
  curr_pts = a.curr_pts;
  pt_in_seconds = a.pt_in_seconds;
  actual_seconds = a.actual_seconds;
  return *this;
}

auto VideoDecoder::open(char const* filepath) noexcept -> bool {
  // Open the file using liv_avformat
  av_opaque->av_format_ctx = avformat_alloc_context();
  if (!av_opaque->av_format_ctx) {
    Core::LogManager::Error(
        "Video::Decoder::ffmpeg:: could not create AVFormatContext");
    return false;
  }
  if (avformat_open_input(&av_opaque->av_format_ctx, filepath, nullptr,
                          nullptr) != 0) {
    Core::LogManager::Error(
        "Video::Decoder::ffmpeg:: could not open video file");
    return false;
  }

  // Find the first valid video stream inside the file
  video_stream_index = -1;
  AVCodecParameters* av_codec_params = nullptr;
  AVCodec const* av_codec = nullptr;
  for (uint32_t i = 0; i < av_opaque->av_format_ctx->nb_streams; ++i) {
    auto& stream = av_opaque->av_format_ctx->streams[i];
    av_codec_params = av_opaque->av_format_ctx->streams[i]->codecpar;
    av_codec = avcodec_find_decoder(av_codec_params->codec_id);

    if (!av_codec) {
      continue;
    }
    if (av_codec->type == AVMEDIA_TYPE_VIDEO) {
      video_stream_index = i;
      width = av_codec_params->width;
      height = av_codec_params->height;
      av_opaque->time_base = av_opaque->av_format_ctx->streams[i]->time_base;
      curr_pts = 0;
      pt_in_seconds = 0.;
      actual_seconds = 0.;
      break;
    }
  }

  if (video_stream_index == -1) {
    Core::LogManager::Error(
        "Video::Decoder::ffmpeg:: Couldn't find valid video streamm inside "
        "file!");
    return false;
  }

  // Set up a codec context for the decoder
  av_opaque->av_codec_ctx = avcodec_alloc_context3(av_codec);
  if (!av_opaque->av_codec_ctx) {
    Core::LogManager::Error(
        "Video::Decoder::ffmpeg:: Couldn't create AVCodecContext!");
    return false;
  }
  if (avcodec_parameters_to_context(av_opaque->av_codec_ctx, av_codec_params)) {
    Core::LogManager::Error(
        "Video::Decoder::ffmpeg:: Couldn't initialize AVCodecContext!");
    return false;
  }
  if (avcodec_open2(av_opaque->av_codec_ctx, av_codec, NULL) < 0) {
    Core::LogManager::Error("Video::Decoder::ffmpeg:: Couldn't open codec!");
    return false;
  }

  av_opaque->av_frame = av_frame_alloc();
  if (!av_opaque->av_frame) {
    Core::LogManager::Error(
        "Video::Decoder::ffmpeg:: Couldn't allocate AVFrame!");
    return false;
  }
  av_opaque->av_packet = av_packet_alloc();
  if (!av_opaque->av_packet) {
    Core::LogManager::Error(
        "Video::Decoder::ffmpeg:: Couldn't allocate AVPacket!");
    return false;
  }

  // set up data
  data = Core::Buffer(width * height * 4 * sizeof(uint8_t));
  timer = std::make_unique<Core::Timer>();
  if (true) {
    Core::GUID guid =
        Core::ResourceManager::get()->requestRuntimeGUID<GFX::Texture>();
    GFX::GFXManager::get()->registerTextureResource(
        guid,
        RHI::TextureDescriptor{RHI::Extend3D{width, height, 1},
                               1,
                               1,
                               1,
                               RHI::TextureDimension::TEX2D,
                               RHI::TextureFormat::RGBA8_UNORM,
                               uint32_t(RHI::TextureUsage::COPY_DST) |
                                   uint32_t(RHI::TextureUsage::TEXTURE_BINDING),
                               {},
                               RHI::TextureFlags::HOSTI_VISIBLE});
    device_texture =
        Core::ResourceManager::get()->getResource<GFX::Texture>(guid);
  }

  return true;
}

auto VideoDecoder::readFrame() noexcept -> bool {
  if (actual_seconds == 0) {
    timer->update();
    actual_seconds += 0.0000001;
  }
  timer->update();
  actual_seconds += timer->deltaTime();
  if (actual_seconds < pt_in_seconds) {
    return true;
  }

  // Decode one frame
  int error, response;
  while (true) {
    error = av_read_frame(av_opaque->av_format_ctx, av_opaque->av_packet);
    if (error == AVERROR_EOF) {
      if (repeat == -1) {
        av_packet_unref(av_opaque->av_packet);
        auto stream = av_opaque->av_format_ctx->streams[video_stream_index];
        avio_seek(av_opaque->av_format_ctx->pb, 0, SEEK_SET);
        avformat_seek_file(av_opaque->av_format_ctx, video_stream_index, 0, 0,
                           stream->duration, 0);
        av_seek_frame(av_opaque->av_format_ctx, video_stream_index, 0, 0);

        curr_pts = 0;
        pt_in_seconds = 0.;
        actual_seconds = 0.;

        continue;
      }
      avcodec_send_packet(av_opaque->av_codec_ctx, av_opaque->av_packet);
      break;
    } else if (error < 0) {
      break;
    }

    if (av_opaque->av_packet->stream_index != video_stream_index) {
      av_packet_unref(av_opaque->av_packet);
      continue;
    }

    response =
        avcodec_send_packet(av_opaque->av_codec_ctx, av_opaque->av_packet);
    if (response < 0) {
      char error_str[AV_ERROR_MAX_STRING_SIZE];
      av_make_error_string(error_str, AV_ERROR_MAX_STRING_SIZE, response);
      Core::LogManager::Error(std::format(
          "Video::Decoder::ffmpeg:: Faled  to decode packet! Error: {0}",
          error_str));
      return false;
    }

    response =
        avcodec_receive_frame(av_opaque->av_codec_ctx, av_opaque->av_frame);
    if (response == AVERROR(EAGAIN) || response == AVERROR_EOF) {
      av_packet_unref(av_opaque->av_packet);
      continue;
    } else if (response < 0) {
      char error_str[AV_ERROR_MAX_STRING_SIZE];
      av_make_error_string(error_str, AV_ERROR_MAX_STRING_SIZE, response);
      Core::LogManager::Error(std::format(
          "Video::Decoder::ffmpeg:: Faled  to decode packet! Error: {0}",
          error_str));
      return false;
    }

    av_packet_unref(av_opaque->av_packet);
    break;
  }

  // Set up sws ctx
  if (!av_opaque->sws_scaler_ctx) {
    av_opaque->sws_scaler_ctx = sws_getContext(
        av_opaque->av_frame->width, av_opaque->av_frame->height,
        av_opaque->av_codec_ctx->pix_fmt, av_opaque->av_frame->width,
        av_opaque->av_frame->height, AV_PIX_FMT_RGB0,
        SWS_BILINEAR, nullptr, nullptr, nullptr);
  }
  if (!av_opaque->sws_scaler_ctx) {
    Core::LogManager::Error(
        "Video::Decoder::ffmpeg:: Couldn't initialize sw scalar.");
    return false;
  }

  uint8_t* dest[4] = {reinterpret_cast<uint8_t*>(data.data), nullptr, nullptr,
                      nullptr};
  int dest_linesize[4] = {av_opaque->av_frame->width * 4, 0, 0, 0};
  sws_scale(av_opaque->sws_scaler_ctx, av_opaque->av_frame->data,
            av_opaque->av_frame->linesize, 0, av_opaque->av_frame->height, dest,
            dest_linesize);
  curr_pts = av_opaque->av_frame->pts;
  pt_in_seconds = curr_pts * (double)av_opaque->time_base.num /
                  (double)av_opaque->time_base.den;

  // Copy to device texture
  if (device_texture) {
    std::future<bool> mapped =
        device_texture->texture->mapAsync(0, 0, data.size);
    if (mapped.get()) {
      void* mapdata = device_texture->texture->getMappedRange(0, data.size);
      memcpy(mapdata, data.data, (size_t)data.size);
      device_texture->texture->unmap();
    }
  }
  return true;
}

auto VideoDecoder::close() noexcept -> bool {
  sws_freeContext(av_opaque->sws_scaler_ctx);
  av_frame_free(&av_opaque->av_frame);
  av_packet_free(&av_opaque->av_packet);
  avformat_close_input(&av_opaque->av_format_ctx);
  avformat_free_context(av_opaque->av_format_ctx);
  avcodec_free_context(&av_opaque->av_codec_ctx);
  return true;
}
}

namespace SIByL::GFX {

auto VideoClip::serialize() noexcept -> void {
  // only serialize if has orid
  if (orid != Core::INVALID_ORID && resourcePath.has_value()) {
    std::filesystem::path metadata_path =
        "./bin/" + std::to_string(orid) + ".meta";
    // handle metadata
    {
      YAML::Emitter out;
      out << YAML::BeginMap;
      // output type
      out << YAML::Key << "ResourceType" << YAML::Value << "VideoClip";
      out << YAML::Key << "Name" << YAML::Value << name;
      out << YAML::Key << "ORID" << YAML::Value << orid;
      out << YAML::Key << "path" << YAML::Value << resourcePath.value();
      out << YAML::Key << "End" << YAML::Value << "TRUE";
      out << YAML::EndMap;
      Core::Buffer vc_proxy;
      vc_proxy.data = (void*)out.c_str();
      vc_proxy.size = out.size();
      Core::syncWriteFile(metadata_path.string().c_str(), vc_proxy);
      vc_proxy.data = nullptr;
    }
  }
}

auto VideoClip::deserialize(RHI::Device* device, Core::ORID ORID) noexcept
    -> void {
  orid = ORID;
  std::filesystem::path metadata_path =
      "./bin/" + std::to_string(ORID) + ".meta";
  Core::Buffer metadata;
  Core::syncReadFile(metadata_path.string().c_str(), metadata);
  YAML::NodeAoS data = YAML::Load(reinterpret_cast<char*>(metadata.data));
  // check scene name
  if (data["ResourceType"].as<std::string>() != "VideoClip") {
    Core::LogManager::Error(std::format(
        "GFX :: VideoClip resource not found when deserializing, ORID: {0}",
        std::to_string(orid)));
    return;
  }
  name = data["Name"].as<std::string>();
  resourcePath = data["path"].as<std::string>();
}

auto VideExtension::registerVideoClipResource(char const* filepath) noexcept
    -> Core::GUID {
  Core::ORID orid =
      Core::ResourceManager::get()->database.mapResourcePath(filepath);
  Core::GUID guid =
      Core::ResourceManager::get()->requestRuntimeGUID<GFX::VideoClip>();
  Core::ResourceManager::get()->database.registerResource(orid, guid);
  Core::ResourceManager::get()->addResource(guid, std::move(GFX::VideoClip{}));
  GFX::VideoClip* videoClip =
      Core::ResourceManager::get()->getResource<GFX::VideoClip>(guid);
  videoClip->orid = orid;
  videoClip->guid = guid;
  videoClip->resourcePath = std::string(filepath);
  videoClip->serialize();
  videoClip->decoder.open(filepath);
  return guid;
}

auto VideExtension::requestOfflineVideoClipResource(Core::ORID orid) noexcept
    -> Core::GUID {
  Core::GUID guid = Core::ResourceManager::get()->database.findResource(orid);
  // if not loaded
  if (guid == Core::INVALID_GUID) {
    guid = Core::ResourceManager::get()->requestRuntimeGUID<GFX::VideoClip>();
    GFX::VideoClip vc;
    vc.deserialize(GFX::GFXManager::get()->rhiLayer->getDevice(), orid);
    vc.decoder.open(vc.resourcePath.value().c_str());
    Core::ResourceManager::get()->addResource(guid, std::move(vc));
    Core::ResourceManager::get()->database.registerResource(orid, guid);
  }
  return guid;
}

auto VideExtension::foo(uint32_t id, void* data) noexcept -> void* {
  if (id == 0) {
    Core::ORID orid = *(reinterpret_cast<Core::ORID*>(data));
    Core::GUID guid = requestOfflineVideoClipResource(orid);
    GFX::VideoClip* videoClip =
        Core::ResourceManager::get()->getResource<GFX::VideoClip>(guid);
    videoClip->active = true;
    return &(videoClip->decoder.device_texture->guid);
  }
  return nullptr;
}

auto VideExtension::startUp() noexcept -> void {
  Core::ResourceManager::get()->registerResource<GFX::VideoClip>();
}

auto VideExtension::onUpdate() noexcept -> void {
  for (auto& pair : Core::ResourceManager::get()
                        ->getResourcePool<GFX::VideoClip>()
                        ->getPool())
    if (pair.second.active) pair.second.decoder.readFrame();
}

}