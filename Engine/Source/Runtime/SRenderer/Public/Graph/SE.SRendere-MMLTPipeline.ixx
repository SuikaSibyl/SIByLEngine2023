module;
#include <cmath>
#include <memory>
#include <string>
#include <cstdint>
export module SE.SRenderer.MMLTPipeline;
import SE.SRenderer.AlbedoPass;
import SE.SRenderer.PreZPass;
import SE.Core.Resource;
import SE.RHI;
import SE.GFX.Core;
import SE.GFX.RDG;
import SE.RDG;

namespace SIByL::SRP
{
	export struct MMLTPipeline :public RDG::Graph {

		MMLTPipeline() {

			//addPass(std::make_unique<PreZPass>(), "Pre-Z Pass");
			//addPass(std::make_unique<AlbedoPass>(), "Albedo Pass");

			//addEdge("Pre-Z Pass", "Depth", "Albedo Pass", "Depth");

			//markOutput("Albedo Pass", "Color");
		}
	};
}