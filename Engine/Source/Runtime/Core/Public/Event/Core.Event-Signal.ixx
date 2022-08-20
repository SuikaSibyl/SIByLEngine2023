module;
#include <functional>
#include <vector>
export module Core.Event:Signal;

namespace SIByL::Core
{
	export template<class ...T>
	struct EventSignal
	{
		using Slot = std::function<void(T...)>;
		auto connect(Slot const& slot) noexcept -> void;

		template<class ...U>
		auto emit(U&&... args) noexcept -> void;
	private:
		std::vector<Slot> connectedSlots;
	};

	template<class ...T>
	auto EventSignal<T...>::connect(Slot const& slot) noexcept -> void {
		connectedSlots.push_back(slot);
	}

	template<class ...T>
	template<class ...U>
	auto EventSignal<T...>::emit(U&&... args) noexcept -> void {
		for (auto& slot : connectedSlots)
			slot(std::forward<U>(args)...);
	}
}