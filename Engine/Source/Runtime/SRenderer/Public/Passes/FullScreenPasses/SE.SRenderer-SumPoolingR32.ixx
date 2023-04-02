module;
#include <cmath>
#include <cstdint>
export module SE.SRenderer.SumPoolingR32;
import SE.Core.Resource;
import SE.RHI;
import SE.GFX.Core;
import SE.GFX.RDG;
import SE.RDG;

namespace SIByL::SRenderer
{
	export struct SumPoolingR32 :public RDG::Pass {

		SumPoolingR32(size_t size)
			:texture_size(size)
		{

		}

		virtual auto reflect() noexcept -> RDG::PassReflection override {
			RDG::PassReflection reflector;
			reflector.addInputOutput("R32HiR")
				.isTexture()
				.withSize(Math::ivec3(texture_size, texture_size, 1))
				.withLevels(std::log2(texture_size))
				.withFormat(RHI::TextureFormat::R32_FLOAT)
				.withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING);
			return reflector;
		}

		virtual auto execute(RDG::RenderContext* context, RDG::RenderData const& renderData) noexcept -> void override {
			GFX::Texture* input = renderData.getTexture("R32HiR");

		}

		size_t texture_size;
	};
}