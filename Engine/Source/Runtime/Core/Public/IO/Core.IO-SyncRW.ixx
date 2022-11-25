module;
#include <format>
#include <filesystem>
#include <fstream>
export module Core.IO:SyncRW;
import Core.Log;
import Core.Memory;
import :FileSystem;

namespace SIByL::Core
{
	export inline auto syncReadFile(filepath const& path, Buffer& buffer) noexcept -> bool {
		std::ifstream ifs(path.string().c_str(), std::ifstream::binary);
		if (ifs.is_open()) {
			ifs.seekg(0, std::ios::end);
			buffer = Buffer(ifs.tellg());
			ifs.seekg(0);
			ifs.read(reinterpret_cast<char*>(buffer.data), buffer.size);
			ifs.close();
		}
		else {
			LogManager::Error(std::format("Core.IO:SyncRW::syncReadFile() failed, file \'{}\' not found.", path.string().c_str()));
		}
		return false;
	}

	export inline auto syncWriteFile(filepath const& path, Buffer& buffer) noexcept -> bool {
		std::ofstream ofs(path.string().c_str(), std::ios::out | std::ios::binary);
		if (ofs.is_open()) {
			ofs.write((char*)buffer.data, buffer.size);
			ofs.close();
		}
		else {
			LogManager::Error(std::format("Core.IO:SyncRW::syncWriteFile() failed, file \'{}\' open failed.", path.string().c_str()));
		}
		return false;
	}
}