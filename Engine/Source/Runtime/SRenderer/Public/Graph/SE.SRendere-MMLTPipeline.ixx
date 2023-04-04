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
import SE.SRenderer.ClearI32RGBA;
import SE.SRenderer.CombineI32RGBA;
import SE.SRenderer.MIPAddPoolingPass;
import SE.SRenderer.BlitPass;

namespace SIByL::SRP
{
	export struct MMLTBoostrapPass :public RDG::RayTracingPass {

		MMLTBoostrapPass() {
			mmlt_boostrap_rgen = GFX::GFXManager::get()->registerShaderModuleResource("../Engine/Binaries/Runtime/spirv/SRenderer/raytracer/mmlt/mmlt_boostrap_pass_rgen.spv", { nullptr, RHI::ShaderStages::RAYGEN });

			GFX::SBTsDescriptor sbt = RTCommon::get()->getSBTDescriptor();
			sbt.rgenSBT = GFX::SBTsDescriptor::RayGenerationSBT{ { Core::ResourceManager::get()->getResource<GFX::ShaderModule>(mmlt_boostrap_rgen) } };

			RDG::RayTracingPass::init(sbt, 1);
		}

		virtual auto reflect() noexcept -> RDG::PassReflection override {
			RDG::PassReflection reflector;

			reflector.addOutput("Color")
				.isTexture()
				.withFormat(RHI::TextureFormat::RGBA32_FLOAT)
				.withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING);

			return reflector;
		}

		virtual auto execute(RDG::RenderContext* context, RDG::RenderData const& renderData) noexcept -> void override {

			GFX::Texture* color = renderData.getTexture("Color");

			std::vector<RHI::BindGroupEntry>* set_0_entries = renderData.getBindGroupEntries("CommonScene");
			getBindGroup(context, 0)->updateBinding(*set_0_entries);

			RHI::RayTracingPassEncoder* encoder = beginPass(context);
			encoder->end();
		}

		Core::GUID mmlt_boostrap_rgen;
	};

	export struct MMLTMetropolisPass :public RDG::RayTracingPass {

		MMLTMetropolisPass() {
			//RDG::RayTracingPass::init();
		}

		virtual auto reflect() noexcept -> RDG::PassReflection override {
			RDG::PassReflection reflector;

			reflector.addOutput("Color")
				.isTexture()
				.withFormat(RHI::TextureFormat::RGBA32_FLOAT)
				.withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING);

			return reflector;
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

	export struct MMLTPipeline :public RDG::Graph {

		MMLTPipeline() {

			addPass(std::make_unique<MIPDummy>(), "Dummy Input");
			addSubgraph(std::make_unique<MIPAddPoolingPass>(512), "Build Boostrap MIP");
			addPass(std::make_unique<BlitPass>(BlitPass::Descriptor(
				RDG::TextureInfo{}.withFormat(RHI::TextureFormat::RGBA32_FLOAT).withSize(Math::ivec3{ 512,512,1 })
			)), "Blit Result");

			addEdge("Dummy Input", "R32", "Build Boostrap MIP", "Input");
			addEdge("Build Boostrap MIP", "Output", "Blit Result", "Input");

			markOutput("Blit Result", "Output");

			//addPass(std::make_unique<ClearI32RGBAPass>(), "Clear-I32RGBA Pass");
			//addPass(std::make_unique<CombineI32RGBAPass>(), "Combine-I32RGBA Pass");



			//addEdge("Clear-I32RGBA Pass", "I32RGBA", "Combine-I32RGBA Pass", "I32RGBA");

			//markOutput("Combine-I32RGBA Pass", "HDRAccum");
			//markOutput("Build Boostrap MIP", "R32");
		}
	};
}