#include <Resource/SE.Core.Resource.hpp>
#include <SE.GFX.hpp>
#include <SE.RDG.hpp>
#include <SE.RHI.hpp>
#include <cmath>
#include <cstdint>
#include <memory>
#include <string>

#include "../Passes/FullScreenPasses/MIP/SE.SRenderer-MIPAddPoolingPass.hpp"
#include "../Passes/FullScreenPasses/MIP/SE.SRenderer-MIPMinPoolingPass.hpp"
#include "../Passes/FullScreenPasses/MIP/SE.SRenderer-MIPSSLCPass.hpp"
#include "../Passes/FullScreenPasses/SE.SRenderer-ACEsPass.hpp"
#include "../Passes/FullScreenPasses/SE.SRenderer-AccumulatePass.hpp"
#include "../Passes/FullScreenPasses/ScreenSpace/SE.SRenderer-BarLCPass.hpp"
#include "../Passes/FullScreenPasses/ScreenSpace/SE.SRenderer-BarPass.hpp"
#include "../Passes/RasterizerPasses/SE.SRenderer-AlbedoPass.hpp"
#include "../Passes/RasterizerPasses/SE.SRenderer-CascadeShadowmapPass.hpp"
#include "../Passes/RasterizerPasses/SE.SRenderer-FooPass.hpp"
#include "../Passes/RasterizerPasses/SE.SRenderer-ForwardPass.hpp"
#include "../Passes/RasterizerPasses/SE.SRenderer-GBufferPass.hpp"
#include "../Passes/RasterizerPasses/SE.SRenderer-PreZPass.hpp"
#include "../Passes/RasterizerPasses/SE.SRenderer-ShadowmapPass.hpp"

namespace SIByL::SRP
{
	SE_EXPORT struct ForwardGraph :public RDG::Graph {
		ForwardGraph() {
			//addPass(std::make_unique<PreZPass>(), "Pre-Z Pass");
			//addPass(std::make_unique<GBufferPass>(), "GBuffer Pass");
			//addPass(std::make_unique<FooPass>(), "Foo Pass");
			//addSubgraph(std::make_unique<MIPAddPoolingPass>(512), "LuminAddPooling");
			//addPass(std::make_unique<BarPass>(), "Bar Pass");
			//addSubgraph(std::make_unique<MIPMinPoolingPass>(512, 512), "HiZ-Gen Pass");

			//addEdge("Pre-Z Pass", "Depth", "HiZ-Gen Pass", "Input");
			//addEdge("Pre-Z Pass", "Depth", "GBuffer Pass", "Depth");
			//addEdge("Foo Pass", "LightProjLumMIP", "LuminAddPooling", "Input");

			//addEdge("GBuffer Pass", "Color", "Bar Pass", "BaseColor");
			//addEdge("HiZ-Gen Pass", "Output", "Bar Pass", "HiZ");
			//addEdge("LuminAddPooling", "Output", "Bar Pass", "HiLumin");
			//addEdge("Foo Pass", "LightProjectionDepth", "Bar Pass", "DepthLumin");
			//addEdge("GBuffer Pass", "wNormal", "Bar Pass", "NormalWS");
			//addEdge("Foo Pass", "LightProjection", "Bar Pass", "LightProjection");

			//markOutput("Bar Pass", "Combined");

			addSubgraph(std::make_unique<PreZPass>(), "Pre-Z Pass");
			addPass(std::make_unique<GBufferPass>(), "GBuffer Pass");
			addSubgraph(std::make_unique<CascadeShadowmapPass>(), "Shadowmap Pass");
			addPass(std::make_unique<ForwardPass>(), "Forward Pass");

			addSubgraph(std::make_unique<MIPMinPoolingPass>(512, 512), "HiZ-Gen Pass");
			addSubgraph(std::make_unique<MIPSSLCPass>(512, 512), "MIPSSLC Pass");
			addPass(std::make_unique<MISCompensationDiffPass>(), "MISC Pass");
			addSubgraph(std::make_unique<MIPAddPoolingPass>(512), "AddPooling Pass");


			addPass(std::make_unique<BarLCPass>(), "BarLC Pass");
			addPass(std::make_unique<AccumulatePass>(), "Accumulate Pass");

			addEdge("Pre-Z Pass", "Depth", "Forward Pass", "Depth");
			addEdge("Shadowmap Pass", "Depth", "Forward Pass", "Shadowmap");

			addEdge("Pre-Z Pass", "Depth", "HiZ-Gen Pass", "Input");
			addEdge("Pre-Z Pass", "Depth", "GBuffer Pass", "Depth");

			addEdge("Forward Pass", "Color", "MIPSSLC Pass", "Color");
			addEdge("GBuffer Pass", "wNormal", "MIPSSLC Pass", "Normal");
			addEdge("Pre-Z Pass", "Depth", "MIPSSLC Pass", "Depth");


			addEdge("Forward Pass", "Color", "BarLC Pass", "DI");
			addEdge("GBuffer Pass", "Color", "BarLC Pass", "BaseColor");
			addEdge("HiZ-Gen Pass", "Output", "BarLC Pass", "HiZ");
			addEdge("GBuffer Pass", "wNormal", "BarLC Pass", "NormalWS");

			addEdge("MIPSSLC Pass", "ImportanceMIP", "MISC Pass", "R32");
			addEdge("MISC Pass", "R32", "AddPooling Pass", "Input");


			addEdge("AddPooling Pass", "Output", "BarLC Pass", "ImportanceMIP");
			addEdge("MIPSSLC Pass", "BoundingBoxMIP", "BarLC Pass", "BoundingBoxMIP");
			addEdge("MIPSSLC Pass", "BBNCPackMIP", "BarLC Pass", "BBNCPackMIP");
			addEdge("MIPSSLC Pass", "NormalConeMIP", "BarLC Pass", "NormalConeMIP");

			addEdge("BarLC Pass", "Combined", "Accumulate Pass", "Input");

			markOutput("Accumulate Pass", "Output");
		}
	};

	SE_EXPORT struct ForwardPipeline : public RDG::SingleGraphPipeline {
		ForwardPipeline() { pGraph = &graph; }
		ForwardGraph graph;
	};
}