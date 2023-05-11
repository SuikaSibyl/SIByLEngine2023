#include <cmath>
#include <memory>
#include <string>
#include <cstdint>
#include <Resource/SE.Core.Resource.hpp>
#include <SE.RHI.hpp>
#include <SE.GFX.hpp>
#include <SE.RDG.hpp>
#include "../Passes/RayTracingPasses/SE.SRenderer-MMLTPass.hpp"
#include "../Passes/FullScreenPasses/SE.SRenderer-ClearI32RGBA.hpp"
#include "../Passes/FullScreenPasses/SE.SRenderer-CombineI32RGBA.hpp"
#include "../Passes/FullScreenPasses/SE.SRenderer-Blit.hpp"
#include "../Passes/FullScreenPasses/MIP/SE.SRenderer-MIPAddPoolingPass.hpp"
#include "../SE.SRenderer.hpp"

namespace SIByL::SRP
{
	SE_EXPORT struct MMLTBoostrapGraph :public RDG::Graph {
		MMLTBoostrapGraph() {
			addPass(std::make_unique<ClearI32RGBAPass>(), "Clean Pass");
			addPass(std::make_unique<MMLTBoostrapPass>(), "Boostrap Pass");

			addEdge("Clean Pass", "Boostrap Pass");

			markOutput("Boostrap Pass", "BoostrapLuminance");
		}
	};

	SE_EXPORT struct MIPDummy :public RDG::DummyPass {
		MIPDummy() {
			RDG::Pass::init();
		}

		virtual auto reflect() noexcept -> RDG::PassReflection override {
			RDG::PassReflection reflector;

			reflector.addOutput("R32")
				.isTexture()
				.withSize(Math::ivec3{ 512,512,1 })
				.withLevels(std::log2(512) + 1)
				.withFormat(RHI::TextureFormat::R32_FLOAT);

			return reflector;
		}
	};
    SE_EXPORT struct MMLTInitializeGraph : public RDG::Graph {
		MMLTInitializeGraph() {
			addPass(std::make_unique<MIPDummy>(), "InputDummy");
			addSubgraph(std::make_unique<MIPAddPoolingPass>(512), "BoostrapMIP");
			addEdge("InputDummy", "R32", "BoostrapMIP", "Input");
			markOutput("BoostrapMIP", "Output");
		}

		auto setSource(GFX::Texture* ref) noexcept -> void {
			setExternal("InputDummy", "R32", ref);
		}
	};

	SE_EXPORT struct MMLTMutationGraph : public RDG::Graph {
		MMLTMutationGraph() {
			addPass(std::make_unique<MIPDummy>(), "InputDummy");
			addPass(std::make_unique<MMLTMutationPass>(), "Mutation Pass");
			addPass(std::make_unique<MMLTCombinePass>(), "Combine Pass");

			addEdge("InputDummy", "R32", "Mutation Pass", "boostrapMIP");
			addEdge("Mutation Pass", "atomicRGBA", "Combine Pass", "I32RGBA");
			addEdge("InputDummy", "R32", "Combine Pass", "BoostrapLuminance");

			markOutput("Combine Pass", "HDRAccum");
		}

		auto setSource(GFX::Texture* mip, GFX::Texture* rgba) noexcept -> void {
			setExternal("InputDummy", "R32", mip);
			setExternal("Mutation Pass", "atomicRGBA", rgba);
		}
	};

	SE_EXPORT struct MMLTPipeline : public RDG::Pipeline {
		
		MMLTPipeline() {}

		virtual auto build() noexcept -> void {
			boostrapPhase.build();
			initializePhase.setSource(boostrapPhase.getOutput());
			initializePhase.build();
			mutationPhase.setSource(boostrapPhase.getOutput(), boostrapPhase.getTextureResource("Clean Pass", "I32RGBA"));
			mutationPhase.build();
		}

		virtual auto execute(RHI::CommandEncoder* encoder) noexcept -> void {
			auto graphs = getActiveGraphs();
			for(auto* graph: graphs)
				graph->execute(encoder);
		}

		virtual auto getActiveGraph() noexcept -> RDG::Graph* {
			uint32_t batchID = RTCommon::get()->accumIDX;
			if (batchID < 4)
				return &boostrapPhase;
			else if (batchID == 4)
				return &initializePhase;
			else 
				return &mutationPhase;
		}

		virtual auto getOutput() noexcept -> GFX::Texture* {
			return mutationPhase.getOutput();
		}

		MMLTBoostrapGraph	boostrapPhase;
		MMLTInitializeGraph initializePhase;
		MMLTMutationGraph	mutationPhase;
	};


}