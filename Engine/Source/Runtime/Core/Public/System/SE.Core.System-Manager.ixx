export module SE.Core.System:Manager;

namespace SIByL::Core
{
	/** An interface for singleton manager for engine*/
	export struct Manager {
		Manager() = default;
		virtual ~Manager() = default;
		/* start up the manager */
		virtual auto startUp() noexcept -> void {}
		/* shut down the manager */
		virtual auto shutDown() noexcept -> void {}
	};
}