module;
#include <memory>
export module SE.SRenderer:Config;
import SE.Math.Geometric;
import SE.RHI;
import SE.GFX.Core;
import SE.GFX.RDG;

import :SRenderer;
import :Raster.Albedo;


namespace SIByL {
	export struct SRendererRegister {
		static auto registerSRenderer(SRenderer* srenderer) noexcept -> void {
			// register common textures
			srenderer->textures.emplace_back(SRenderer::TextureRegisterInfo{
				"RasterizerTarget_Color",
				GFX::RDGTexture::Desc{
					{800,600,1},
					1, 1, RHI::TextureDimension::TEX2D,
					RHI::TextureFormat::RGBA32_FLOAT }
				});
			srenderer->textures.emplace_back(SRenderer::TextureRegisterInfo{
				"RasterizerTarget_Depth",
				GFX::RDGTexture::Desc{
					{800,600,1},
					1, 1, RHI::TextureDimension::TEX2D,
					RHI::TextureFormat::DEPTH32_FLOAT }
				});
			// register passes
			srenderer->passes.emplace_back(std::make_unique<AlbedoOnlyPass>());
		}
	};
}