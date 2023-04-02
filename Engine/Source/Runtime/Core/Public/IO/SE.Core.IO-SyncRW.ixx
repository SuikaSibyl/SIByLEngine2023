module;
#include <vector>
#include <format>
#include <filesystem>
#include <fstream>
export module SE.Core.IO:SyncRW;
import SE.Core.Log;
import SE.Core.Memory;
import :FileSystem;

namespace SIByL::Core
{
	export inline auto syncReadFile(filepath const& path, Buffer& buffer) noexcept -> bool {
		std::ifstream ifs(path.string().c_str(), std::ifstream::binary);
		if (ifs.is_open()) {
			ifs.seekg(0, std::ios::end);
			size_t size = size_t(ifs.tellg());
			buffer = Buffer(size + 1);
			buffer.size = size;
			ifs.seekg(0);
			ifs.read(reinterpret_cast<char*>(buffer.data), size);
			((char*)buffer.data)[size] = '\0';
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

	export inline auto syncWriteFile(filepath const& path, std::vector<Buffer*> const& buffers) noexcept -> bool {
		std::ofstream ofs(path.string().c_str(), std::ios::out | std::ios::binary);
		if (ofs.is_open()) {
			for (auto* buffer : buffers)
				ofs.write((char*)buffer->data, buffer->size);
			ofs.close();
		}
		else {
			LogManager::Error(std::format("Core.IO:SyncRW::syncWriteFile() failed, file \'{}\' open failed.", path.string().c_str()));
		}
		return false;
	}
}