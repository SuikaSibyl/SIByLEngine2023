module;
#include <cmath>
#include <memory>
#include <string>
#include <cstdint>
export module SE.SRenderer.MMLTPipeline;
import SE.Core.Resource;
import SE.RHI;
import SE.GFX.Core;
import SE.GFX.RDG;
import SE.RDG;

import SE.SRenderer.ClearI32RGBA;
import SE.SRenderer.CombineI32RGBA;

namespace SIByL::SRP
{
	export struct MMLTPipeline :public RDG::Graph {

		MMLTPipeline() {

			addPass(std::make_unique<ClearI32RGBAPass>(), "Clear-I32RGBA Pass");
			addPass(std::make_unique<CombineI32RGBAPass>(), "Combine-I32RGBA Pass");

			addEdge("Clear-I32RGBA Pass", "I32RGBA", "Combine-I32RGBA Pass", "I32RGBA");

			markOutput("Combine-I32RGBA Pass", "HDRAccum");
		}
	};
}