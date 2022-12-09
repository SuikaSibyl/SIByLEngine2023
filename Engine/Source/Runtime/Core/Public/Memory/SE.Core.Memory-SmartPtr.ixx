module;
#include <cstdint>
#include <utility>
#include <memory>
#include <type_traits>
export module SE.Core.Memory:SmartPtr;
import :Allocator;
import :MemoryManager;

namespace SIByL::Core
{
	export template<class T>
	using Scope = std::unique_ptr<T>;
}

export using SIByL::Core::Scope;