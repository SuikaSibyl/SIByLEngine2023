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
			GFX::RDGraph* rdg = srenderer->rdgraph;
			// register common textures
			GFX::RDGTexture* texRes_rasterizer_target_color = rdg->createTexture(
				"RasterizerTarget_Color",
				GFX::RDGTexture::Desc{
					{800,600,1},
					1, 1, RHI::TextureDimension::TEX2D,
					RHI::TextureFormat::RGBA32_FLOAT }
			);
			GFX::RDGTexture* texRes_rasterizer_target_depth = rdg->createTexture(
				"RasterizerTarget_Depth",
				GFX::RDGTexture::Desc{
					{800,600,1},
					1, 1, RHI::TextureDimension::TEX2D,
					RHI::TextureFormat::DEPTH32_FLOAT }
			);
			// register passes
			srenderer->passes.emplace_back(std::make_unique<AlbedoOnlyPass>());
			for (auto& pass : srenderer->passes) {
				pass->loadShaders();
				pass->registerPass(srenderer);
			}
		}
	};
}