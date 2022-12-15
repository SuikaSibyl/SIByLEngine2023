module;
#include <memory>
#include <utility>
export module SE.GFX.Core:Sampler;
import SE.Core.Resource;
import SE.RHI;

namespace SIByL::GFX
{
	export struct Sampler :public Core::Resource {
		/** ctors & rval copies */
		Sampler() = default;
		Sampler(Sampler&& sampler) = default;
		Sampler(Sampler const& sampler) = delete;
		auto operator=(Sampler&& sampler) -> Sampler & = default;
		auto operator=(Sampler const& sampler) ->Sampler & = delete;
		/* rhi sampler */
		std::unique_ptr<RHI::Sampler> sampler = nullptr;
		/** get name */
		virtual auto getName() const noexcept -> char const* {
			return sampler->getName().c_str();
		}
	};
}