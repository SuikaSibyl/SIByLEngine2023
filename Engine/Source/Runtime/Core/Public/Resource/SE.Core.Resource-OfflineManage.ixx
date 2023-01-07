module;
#include <unordered_map>
#include <filesystem>
#include <random>
#include <functional>
#include <yaml-cpp/yaml.h>
#include <yaml-cpp/node/node.h>
export module SE.Core.Resource:OfflineManage;
import :GUID;
import SE.Core.IO;
import SE.Core.Log;
import SE.Core.Memory;

namespace SIByL::Core
{
	/** Offline Resource ID */
	export using ORID = uint64_t;
	export constexpr inline ORID ORID_NONE = uint64_t(-1);

	export inline auto requestORID() noexcept -> ORID {
		static std::default_random_engine e;
		static std::uniform_int_distribution<uint64_t> u(0, 0X3FFFFF);

		ORID id = 0;
		time_t now = time(0);
		tm ltm;
		localtime_s(&ltm, &now);
		id += (uint64_t(ltm.tm_year - 100) & 0xFF) << 56;
		id += (uint64_t(ltm.tm_mon) & 0xF) << 52;
		id += (uint64_t(ltm.tm_mday) & 0x1F) << 47;
		id += (uint64_t(ltm.tm_hour) & 0x1F) << 42;
		id += (uint64_t(ltm.tm_min) & 0x3F) << 36;
		id += (uint64_t(ltm.tm_sec) & 0x3F) << 30;

		std::thread::id tid = std::this_thread::get_id();
		unsigned int nId = *(unsigned int*)((char*)&tid);
		id += (uint64_t(nId) & 0xFF) << 22;
		id += u(e);
		return id;
	}

	export struct ResourceDatabase {
		struct Entry {
			Core::ORID orid;	// offline resource id
			Core::GUID guid = Core::ORID_NONE;	// runtime resource id
		};
		/** register a resourece to databse as loaded */
		auto registerResource(Core::ORID orid, Core::GUID guid) noexcept -> void {
			mapper[orid] = guid;
		}
		/** register a resourece to databse as loaded */
		auto findResource(Core::ORID orid) noexcept -> Core::GUID {
			auto iter = mapper.find(orid);
			if (iter == mapper.end()) return Core::ORID_NONE;
			else return iter->second;
		}
		/** find ORID for a path */
		auto findResourcePath(char const* path_c) noexcept -> Core::ORID {
			std::filesystem::path path(path_c);
			std::filesystem::path current_path = std::filesystem::current_path();
			std::filesystem::path relative_path = std::filesystem::relative(path, current_path);
			auto iter = resource_mapper.find(relative_path.string());
			if (iter == resource_mapper.end()) {
				return Core::ORID_NONE;
			}
			else return iter->second;
		}
		/** find ORID for a path or create one */
		auto mapResourcePath(char const* path_c) noexcept -> Core::ORID {
			std::filesystem::path path(path_c);
			std::filesystem::path current_path = std::filesystem::current_path();
			std::filesystem::path relative_path = std::filesystem::relative(path, current_path);
			auto iter = resource_mapper.find(relative_path.string());
			if (iter == resource_mapper.end()) {
				Core::ORID orid = requestORID();;
				resource_mapper[relative_path.string()] = orid;
				return orid;
			}
			else return iter->second;
		}
		std::unordered_map<Core::ORID, Core::GUID> mapper;
		std::unordered_map<std::string, Core::ORID> resource_mapper;
		/** serialize */
		auto serialize() noexcept -> void {
			std::filesystem::path path = "./bin/.adb";
			YAML::Emitter out;
			out << YAML::BeginMap;
			out << YAML::Key << "Prefix" << YAML::Value << "AssetDatabase";
			out << YAML::Key << "Entries" << YAML::Value << YAML::BeginSeq;
			for (auto& [name, ORID] : resource_mapper) {
				out << YAML::BeginMap;
				out << YAML::Key << "PATH" << YAML::Value << name;
				out << YAML::Key << "ORID" << YAML::Value << ORID;
				out << YAML::EndMap;
			}
			out << YAML::EndSeq;
			out << YAML::Key << "End" << YAML::Value << "TRUE";
			Core::Buffer adb_proxy;
			adb_proxy.data = (void*)out.c_str();
			adb_proxy.size = out.size();
			Core::syncWriteFile(path, adb_proxy);
			adb_proxy.data = nullptr;
		}
		/** deserialize */
		auto deserialize() noexcept -> void {
			std::filesystem::path path = "./bin/.adb";
			Core::Buffer adb_proxy;
			Core::syncReadFile(path, adb_proxy);
			if (adb_proxy.size != 0) {
				YAML::NodeAoS data = YAML::Load(reinterpret_cast<char*>(adb_proxy.data));
				// check scene name
				if (!data["Prefix"]) {
					Core::LogManager::Error("GFX :: Asset Database not found when deserializing {0}");
					return;
				}
				auto entries = data["Entries"];
				for (auto node : entries) {
					resource_mapper[node["PATH"].as<std::string>()] = node["ORID"].as<Core::ORID>();
				}
			}
		}
	};
}