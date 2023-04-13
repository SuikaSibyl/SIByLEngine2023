module;
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
}
#include <cstdint>
#include <future>
#include <format>
#include <optional>
export module SE.Video:Decoder;
import SE.Core.Log;
import SE.Core.Misc;
import SE.Core.Memory;
import SE.Core.Resource;
import SE.RHI;
import SE.GFX;

namespace SIByL::Video
{
	export struct VideoDecoder {
		
		auto open(char const* filepath) noexcept -> bool;

		auto readFrame() noexcept -> bool;

		auto close() noexcept -> bool;

		uint32_t width, height;
		AVRational time_base;
		Core::Buffer data;

		AVFormatContext* av_format_ctx	= nullptr;
		AVCodecContext* av_codec_ctx	= nullptr;
		AVFrame* av_frame				= nullptr;
		AVPacket* av_packet				= nullptr;
		SwsContext* sws_scaler_ctx		= nullptr;
		int video_stream_index;

		GFX::Buffer*	device_buffer;
		GFX::Texture*	device_texture;
		std::unique_ptr<Core::Timer> timer;

		int repeat = -1;
		int64_t curr_pts;
		double pt_in_seconds;
		double actual_seconds;
	};

	auto VideoDecoder::open(char const* filepath) noexcept -> bool {
		// Open the file using liv_avformat
		av_format_ctx = avformat_alloc_context();
		if (!av_format_ctx) {
			Core::LogManager::Error("Video::Decoder::ffmpeg:: could not create AVFormatContext");
			return false;
		}
		if (avformat_open_input(&av_format_ctx, filepath, nullptr, nullptr) != 0) {
			Core::LogManager::Error("Video::Decoder::ffmpeg:: could not open video file");
			return false;
		}

		// Find the first valid video stream inside the file
		video_stream_index = -1;
		AVCodecParameters* av_codec_params = nullptr;
		AVCodec const* av_codec = nullptr;
		for (int i = 0; i < av_format_ctx->nb_streams; ++i) {
			auto& stream = av_format_ctx->streams[i];
			av_codec_params = av_format_ctx->streams[i]->codecpar;
			av_codec = avcodec_find_decoder(av_codec_params->codec_id);

			if (!av_codec) {
				continue;
			}
			if (av_codec->type == AVMEDIA_TYPE_VIDEO) {
				video_stream_index = i;
				width = av_codec_params->width;
				height = av_codec_params->height;
				time_base = av_format_ctx->streams[i]->time_base;
				curr_pts = 0;
				pt_in_seconds = 0.;
				actual_seconds = 0.;
				break;
			}
		}

		if (video_stream_index == -1) {
			Core::LogManager::Error("Video::Decoder::ffmpeg:: Couldn't find valid video streamm inside file!");
			return false;
		}

		// Set up a codec context for the decoder
		av_codec_ctx = avcodec_alloc_context3(av_codec);
		if (!av_codec_ctx) {
			Core::LogManager::Error("Video::Decoder::ffmpeg:: Couldn't create AVCodecContext!");
			return false;
		}
		if (avcodec_parameters_to_context(av_codec_ctx, av_codec_params)) {
			Core::LogManager::Error("Video::Decoder::ffmpeg:: Couldn't initialize AVCodecContext!");
			return false;
		}
		if (avcodec_open2(av_codec_ctx, av_codec, NULL) < 0) {
			Core::LogManager::Error("Video::Decoder::ffmpeg:: Couldn't open codec!");
			return false;
		}

		av_frame = av_frame_alloc();
		if (!av_frame) {
			Core::LogManager::Error("Video::Decoder::ffmpeg:: Couldn't allocate AVFrame!");
			return false;
		}
		av_packet = av_packet_alloc();
		if (!av_packet) {
			Core::LogManager::Error("Video::Decoder::ffmpeg:: Couldn't allocate AVPacket!");
			return false;
		}

		// set up data
		data = Core::Buffer(width * height * 4 * sizeof(uint8_t));
		timer = std::make_unique<Core::Timer>();
		if (true) {
			Core::GUID guid = Core::ResourceManager::get()->requestRuntimeGUID<GFX::Texture>();
			GFX::GFXManager::get()->registerTextureResource(guid, RHI::TextureDescriptor{
				RHI::Extend3D{width, height, 1},
				1, 1, RHI::TextureDimension::TEX2D,
				RHI::TextureFormat::RGBA8_UNORM,
				uint32_t(RHI::TextureUsage::COPY_DST) | uint32_t(RHI::TextureUsage::TEXTURE_BINDING),
				{}, RHI::TextureFlags::HOSTI_VISIBLE
			});
			device_texture = Core::ResourceManager::get()->getResource<GFX::Texture>(guid);
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
			error = av_read_frame(av_format_ctx, av_packet);
			if (error == AVERROR_EOF) {
				if (repeat == -1) {
					av_packet_unref(av_packet);
					auto stream = av_format_ctx->streams[video_stream_index];
					avio_seek(av_format_ctx->pb, 0, SEEK_SET);
					avformat_seek_file(av_format_ctx, video_stream_index, 0, 0, stream->duration, 0);
					av_seek_frame(av_format_ctx, video_stream_index, 0, 0);

					curr_pts = 0;
					pt_in_seconds = 0.;
					actual_seconds = 0.;

					continue;
				}
				avcodec_send_packet(av_codec_ctx, av_packet);
				break;
			}
			else if (error < 0) {
				break;
			}

			if (av_packet->stream_index != video_stream_index) {
				av_packet_unref(av_packet);
				continue;
			}

			response = avcodec_send_packet(av_codec_ctx, av_packet);
			if (response < 0) {
				char error_str[AV_ERROR_MAX_STRING_SIZE];
				av_make_error_string(error_str, AV_ERROR_MAX_STRING_SIZE, response);
				Core::LogManager::Error(std::format("Video::Decoder::ffmpeg:: Faled  to decode packet! Error: {0}", error_str));
				return false;
			}

			response = avcodec_receive_frame(av_codec_ctx, av_frame);
			if (response == AVERROR(EAGAIN) || response == AVERROR_EOF) {
				av_packet_unref(av_packet);
				continue;
			}
			else if (response < 0) {
				char error_str[AV_ERROR_MAX_STRING_SIZE];
				av_make_error_string(error_str, AV_ERROR_MAX_STRING_SIZE, response);
				Core::LogManager::Error(std::format("Video::Decoder::ffmpeg:: Faled  to decode packet! Error: {0}", error_str));
				return false;
			}

			av_packet_unref(av_packet);
			break;
		}

		// Set up sws ctx
		if (!sws_scaler_ctx) {
			sws_scaler_ctx = sws_getContext(
				av_frame->width,
				av_frame->height,
				av_codec_ctx->pix_fmt,
				av_frame->width,
				av_frame->height,
				AV_PIX_FMT_RGB0,
				SWS_BILINEAR,
				nullptr,
				nullptr,
				nullptr);
		}
		if (!sws_scaler_ctx) {
			Core::LogManager::Error("Video::Decoder::ffmpeg:: Couldn't initialize sw scalar.");
			return false;
		}

		uint8_t* dest[4] = { reinterpret_cast<uint8_t*>(data.data), nullptr, nullptr, nullptr};
		int dest_linesize[4] = { av_frame->width * 4,0,0,0 };
		sws_scale(sws_scaler_ctx, av_frame->data, av_frame->linesize, 0, av_frame->height, dest, dest_linesize);
		curr_pts = av_frame->pts;
		pt_in_seconds = curr_pts * (double)time_base.num / (double)time_base.den;

		// Copy to device texture
		if (device_texture) {
			std::future<bool> mapped = device_texture->texture->mapAsync(0, 0, data.size);
			if (mapped.get()) {
				void* mapdata = device_texture->texture->getMappedRange(0, data.size);
				memcpy(mapdata, data.data, (size_t)data.size);
				device_texture->texture->unmap();
			}
		}
		return true;
	}

	auto VideoDecoder::close() noexcept -> bool {
		sws_freeContext(sws_scaler_ctx);
		av_frame_free(&av_frame);
		av_packet_free(&av_packet);
		avformat_close_input(&av_format_ctx);
		avformat_free_context(av_format_ctx);
		avcodec_free_context(&av_codec_ctx);
		return true;
	}

}