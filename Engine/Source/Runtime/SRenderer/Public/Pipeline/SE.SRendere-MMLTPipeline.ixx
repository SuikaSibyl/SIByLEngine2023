module;
#include <cmath>
#include <memory>
#include <string>
#include <cstdint>
export module SE.SRenderer.MMLTPipeline;
import SE.Core.Resource;
import SE.RHI;
import SE.GFX;
import SE.RDG;

import SE.SRenderer;
import SE.SRenderer.MMLTPass;
import SE.SRenderer.ClearI32RGBA;
import SE.SRenderer.CombineI32RGBA;
import SE.SRenderer.MIPAddPoolingPass;
import SE.SRenderer.BlitPass;

namespace SIByL::SRP
{
	export struct MMLTBoostrapGraph :public RDG::Graph {
		MMLTBoostrapGraph() {
			addPass(std::make_unique<ClearI32RGBAPass>(), "Clean Pass");
			addPass(std::make_unique<MMLTBoostrapPass>(), "Boostrap Pass");

			addEdge("Clean Pass", "Boostrap Pass");

			markOutput("Boostrap Pass", "BoostrapLuminance");
		}
	};

	export struct MIPDummy :public RDG::DummyPass {
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
	export struct MMLTInitializeGraph :public RDG::Graph {
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

	export struct MMLTMutationGraph :public RDG::Graph {
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

	export struct MMLTPipeline : public RDG::Pipeline {
		
		MMLTPipeline() {}

		virtual auto build() noexcept -> void {
			boostrapPhase.build();
			initializePhase.setSource(boostrapPhase.getOutput());
			initializePhase.build();
			mutationPhase.setSource(boostrapPhase.getOutput(), boostrapPhase.getTextureResource("Clean Pass", "I32RGBA"));
			mutationPhase.build();
		}

		virtual auto execute(RHI::CommandEncoder* encoder) noexcept -> void {
			getActiveGraph()->execute(encoder);
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