module;
#include <cmath>
#include <memory>
#include <string>
#include <cstdint>
export module SE.SRenderer.ForwardPipeline;
import SE.Core.Resource;
import SE.RHI;
import SE.GFX;
import SE.RDG;

import SE.SRenderer.AlbedoPass;
import SE.SRenderer.PreZPass;
import SE.SRenderer.ACEsPass;
import SE.SRenderer.MIPMinPoolingPass;

namespace SIByL::SRP
{
	export struct ForwardGraph :public RDG::Graph {
		ForwardGraph() {
			addPass(std::make_unique<PreZPass>(), "Pre-Z Pass");
			addPass(std::make_unique<AlbedoPass>(), "Albedo Pass");
			addPass(std::make_unique<ACEsPass>(), "ACEs Pass");

			addSubgraph(std::make_unique<MIPMinPoolingPass>(1280, 720), "HiZ-Gen Pass");

			addEdge("Pre-Z Pass", "Depth", "Albedo Pass", "Depth");
			addEdge("Pre-Z Pass", "Depth", "HiZ-Gen Pass", "Input");
			addEdge("Albedo Pass", "Color", "ACEs Pass", "HDR");

			markOutput("HiZ-Gen Pass", "Output");
		}
	};

	export struct ForwardPipeline :public RDG::SingleGraphPipeline {
		ForwardPipeline() { pGraph = &graph; }
		ForwardGraph graph;
	};
}