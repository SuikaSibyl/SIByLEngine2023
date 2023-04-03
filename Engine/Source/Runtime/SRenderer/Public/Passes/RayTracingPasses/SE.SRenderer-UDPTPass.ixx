module;
#include <cstdint>
export module SE.SRenderer.UDPTPass;
import SE.Core.Resource;
import SE.RHI;
import SE.GFX.Core;
import SE.GFX.RDG;
import SE.RDG;

namespace SIByL::SRenderer
{
	export struct UDPTPass :public RDG::Pass {

		virtual auto reflect() noexcept -> RDG::PassReflection override {
			RDG::PassReflection reflector;

			reflector.addOutput("Color")
				.isTexture()
				.withFormat(RHI::TextureFormat::RGBA32_FLOAT)
				.withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING);

			return reflector;
		}
	};
}