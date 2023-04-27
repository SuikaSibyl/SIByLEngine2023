module;
#include <string>
#include <cstdint>
#include <future>
#include <format>
#include <optional>
#include <filesystem>
#include <yaml-cpp/yaml.h>
#include <yaml-cpp/node/node.h>
export module SE.Video:VideoClip;
import SE.Core.IO;
import SE.Core.Log;
import SE.Core.Misc;
import SE.Core.Memory;
import SE.Core.Resource;
import SE.RHI;
import SE.GFX;
import :Decoder;

namespace SIByL::GFX
{
	export struct VideoClip :public Core::Resource {
		/** ctors & rval copies */
		VideoClip() = default;
		VideoClip(VideoClip&& vc) = default;
		VideoClip(VideoClip const& vc) = delete;
		auto operator=(VideoClip&& vc)->VideoClip & = default;
		auto operator=(VideoClip const& vc)->VideoClip & = delete;
		/** serialize */
		inline auto serialize() noexcept -> void;
		/** deserialize */
		inline auto deserialize(RHI::Device* device, Core::ORID orid) noexcept -> void;
		/** get name */
		virtual auto getName() const noexcept -> char const* override { return name.c_str(); }
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

	export struct VideExtension :public GFX::Extension {
		auto registerVideoClipResource(char const* filepath) noexcept -> Core::GUID {
			Core::ORID orid = Core::ResourceManager::get()->database.mapResourcePath(filepath);
			Core::GUID guid = Core::ResourceManager::get()->requestRuntimeGUID<GFX::VideoClip>();
			Core::ResourceManager::get()->database.registerResource(orid, guid);
			Core::ResourceManager::get()->addResource(guid, std::move(GFX::VideoClip{}));
			GFX::VideoClip* videoClip = Core::ResourceManager::get()->getResource<GFX::VideoClip>(guid);
			videoClip->orid = orid;
			videoClip->guid = guid;
			videoClip->resourcePath = std::string(filepath);
			videoClip->serialize();
			videoClip->decoder.open(filepath);
			return guid;
		}
		auto requestOfflineVideoClipResource(Core::ORID orid) noexcept -> Core::GUID {
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
		virtual auto foo(uint32_t id, void* data) noexcept -> void* {
			if (id == 0) {
				Core::ORID orid = *(reinterpret_cast<Core::ORID*>(data));
				Core::GUID guid = requestOfflineVideoClipResource(orid);
				GFX::VideoClip* videoClip = Core::ResourceManager::get()->getResource<GFX::VideoClip>(guid);
				videoClip->active = true;
				return &(videoClip->decoder.device_texture->guid);
			}
			return nullptr;
		}
	protected:
		virtual auto startUp() noexcept -> void override {
			Core::ResourceManager::get()->registerResource<GFX::VideoClip>();
		}
		virtual auto onUpdate() noexcept -> void override {
			for (auto& pair : Core::ResourceManager::get()->getResourcePool<GFX::VideoClip>()->getPool())
				if (pair.second.active)
					pair.second.decoder.readFrame();
		}
	};

#pragma region IMPL_VIDEO_CLIP

	inline auto VideoClip::serialize() noexcept -> void {
		// only serialize if has orid
		if (orid != Core::INVALID_ORID && resourcePath.has_value()) {
			std::filesystem::path metadata_path = "./bin/" + std::to_string(orid) + ".meta";
			// handle metadata
			{	YAML::Emitter out;
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

	inline auto VideoClip::deserialize(RHI::Device* device, Core::ORID ORID) noexcept -> void {
		orid = ORID;
		std::filesystem::path metadata_path = "./bin/" + std::to_string(ORID) + ".meta";
		Core::Buffer metadata;
		Core::syncReadFile(metadata_path.string().c_str(), metadata);
		YAML::NodeAoS data = YAML::Load(reinterpret_cast<char*>(metadata.data));
		// check scene name
		if (data["ResourceType"].as<std::string>() != "VideoClip") {
			Core::LogManager::Error(std::format("GFX :: VideoClip resource not found when deserializing, ORID: {0}", std::to_string(orid)));
			return;
		}
		name = data["Name"].as<std::string>();
		resourcePath = data["path"].as<std::string>();
	}

#pragma endregion

}