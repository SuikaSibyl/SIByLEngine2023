export module SE.Core.System:Manager;

namespace SIByL::Core
{
	export struct Manager
	{
		Manager() = default;
		virtual ~Manager() = default;

		virtual auto startUp() noexcept -> void {}
		virtual auto shutDown() noexcept -> void {}
	};
}