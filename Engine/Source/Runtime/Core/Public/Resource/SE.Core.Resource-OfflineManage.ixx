module;
#include <unordered_map>
#include <filesystem>
#include <random>
#include <functional>
export module SE.Core.Resource:OfflineManage;
import :GUID;

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
		std::unordered_map<Core::ORID, Core::GUID> mapper;
	};
}