module;
#include <memory>
#include <utility>
export module GFX.Resource:Sampler;
import RHI;

namespace SIByL::GFX
{
	export struct Sampler {
		/** ctors & rval copies */
		Sampler() = default;
		Sampler(Sampler&& sampler) = default;
		Sampler(Sampler const& sampler) = delete;
		auto operator=(Sampler&& sampler) -> Sampler & = default;
		auto operator=(Sampler const& sampler) ->Sampler & = delete;
		/* rhi sampler */
		std::unique_ptr<RHI::Sampler> sampler = nullptr;
	};
}