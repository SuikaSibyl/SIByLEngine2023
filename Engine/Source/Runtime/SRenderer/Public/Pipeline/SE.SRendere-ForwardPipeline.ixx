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
import SE.SRenderer.GBufferPass;
import SE.SRenderer.MIPMinPoolingPass;
import SE.SRenderer.MIPAddPoolingPass;
import SE.SRenderer.FooPass;
import SE.SRenderer.BarPass;

namespace SIByL::SRP
{
	export struct ForwardGraph :public RDG::Graph {
		ForwardGraph() {
			addPass(std::make_unique<PreZPass>(), "Pre-Z Pass");
			addPass(std::make_unique<GBufferPass>(), "GBuffer Pass");
			addPass(std::make_unique<FooPass>(), "Foo Pass");
			addSubgraph(std::make_unique<MIPAddPoolingPass>(512), "LuminAddPooling");
			addPass(std::make_unique<BarPass>(), "Bar Pass");
			addSubgraph(std::make_unique<MIPMinPoolingPass>(512, 512), "HiZ-Gen Pass");

			addEdge("Pre-Z Pass", "Depth", "HiZ-Gen Pass", "Input");
			addEdge("Pre-Z Pass", "Depth", "GBuffer Pass", "Depth");
			addEdge("Foo Pass", "LightProjLumMIP", "LuminAddPooling", "Input");

			addEdge("GBuffer Pass", "Color", "Bar Pass", "BaseColor");
			addEdge("HiZ-Gen Pass", "Output", "Bar Pass", "HiZ");
			addEdge("LuminAddPooling", "Output", "Bar Pass", "HiLumin");
			addEdge("Foo Pass", "LightProjectionDepth", "Bar Pass", "DepthLumin");
			addEdge("GBuffer Pass", "wNormal", "Bar Pass", "NormalWS");
			addEdge("Foo Pass", "LightProjection", "Bar Pass", "LightProjection");

			markOutput("Bar Pass", "Combined");
		}
	};

	export struct ForwardPipeline :public RDG::SingleGraphPipeline {
		ForwardPipeline() { pGraph = &graph; }
		ForwardGraph graph;
	};
}