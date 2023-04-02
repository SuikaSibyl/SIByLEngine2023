module;
#include <cmath>
#include <memory>
#include <string>
#include <cstdint>
export module SE.SRenderer.ForwardPipeline;
import SE.Core.Resource;
import SE.RHI;
import SE.GFX.Core;
import SE.GFX.RDG;
import SE.RDG;

import SE.SRenderer.AlbedoPass;
import SE.SRenderer.PreZPass;
import SE.SRenderer.ACEsPass;

namespace SIByL::SRP
{
	export struct ForwardPipeline :public RDG::Graph {

		ForwardPipeline() {
			addPass(std::make_unique<PreZPass>(), "Pre-Z Pass");
			addPass(std::make_unique<AlbedoPass>(), "Albedo Pass");
			addPass(std::make_unique<ACEsPass>(), "ACEs Pass");

			addEdge("Pre-Z Pass", "Depth", "Albedo Pass", "Depth");
			addEdge("Albedo Pass", "Color", "ACEs Pass", "HDR");

			markOutput("ACEs Pass", "LDR");
		}

	};
}