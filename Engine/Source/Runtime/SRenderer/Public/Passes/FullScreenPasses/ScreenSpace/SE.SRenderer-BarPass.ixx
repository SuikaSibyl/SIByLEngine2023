module;
#include <array>
#include <memory>
#include <compare>
#include <typeinfo>
#include <filesystem>
#include <functional>
#include <imgui.h>
#include <imgui_internal.h>
#include "../../../../../Application/Public/SE.Application.Config.h"
export module SE.SRenderer.BarPass;
import SE.SRenderer;
import SE.Math.Geometric;
import SE.Core.Resource;
import SE.RHI;
import SE.GFX;
import SE.RDG;

namespace SIByL
{
	export struct BarPass :public RDG::FullScreenPass {

		BarPass() {
			frag = GFX::GFXManager::get()->registerShaderModuleResource(
				"../Engine/Binaries/Runtime/spirv/SRenderer/rasterizer/foo_bar/bar_frag.spv",
				{ nullptr, RHI::ShaderStages::FRAGMENT });
			RDG::FullScreenPass::init(Core::ResourceManager::get()->getResource<GFX::ShaderModule>(frag));
		}

		virtual auto reflect() noexcept -> RDG::PassReflection {
			RDG::PassReflection reflector;

			reflector.addInput("BaseColor")
				.isTexture()
				.withUsages((uint32_t)RHI::TextureUsage::TEXTURE_BINDING)
				.consume(RDG::TextureInfo::ConsumeEntry{ RDG::TextureInfo::ConsumeType::TextureBinding }
					.setSubresource(0, 1, 0, 1)
					.addStage((uint32_t)RHI::PipelineStages::FRAGMENT_SHADER_BIT));

			reflector.addInput("HiZ")
				.isTexture()
				.withUsages((uint32_t)RHI::TextureUsage::TEXTURE_BINDING)
				.consume(RDG::TextureInfo::ConsumeEntry{ RDG::TextureInfo::ConsumeType::TextureBinding }
					.setSubresource(0, RDG::MaxPossible, 0, 1)
					.addStage((uint32_t)RHI::PipelineStages::FRAGMENT_SHADER_BIT));

			reflector.addInput("LightProjection")
				.isTexture()
				.withUsages((uint32_t)RHI::TextureUsage::TEXTURE_BINDING)
				.consume(RDG::TextureInfo::ConsumeEntry{ RDG::TextureInfo::ConsumeType::TextureBinding }
					.setSubresource(0, RDG::MaxPossible, 0, 1)
					.addStage((uint32_t)RHI::PipelineStages::FRAGMENT_SHADER_BIT));

			reflector.addInput("HiLumin")
				.isTexture()
				.withUsages((uint32_t)RHI::TextureUsage::TEXTURE_BINDING)
				.consume(RDG::TextureInfo::ConsumeEntry{ RDG::TextureInfo::ConsumeType::TextureBinding }
					.setSubresource(0, RDG::MaxPossible, 0, 1)
					.addStage((uint32_t)RHI::PipelineStages::FRAGMENT_SHADER_BIT));
			
			reflector.addInput("DepthLumin")
				.isTexture()
				.withUsages((uint32_t)RHI::TextureUsage::TEXTURE_BINDING)
				.consume(RDG::TextureInfo::ConsumeEntry{ RDG::TextureInfo::ConsumeType::TextureBinding }
					.setSubresource(0, 1, 0, 1)
					.addStage((uint32_t)RHI::PipelineStages::FRAGMENT_SHADER_BIT));

			reflector.addInput("NormalWS")
				.isTexture()
				.withUsages((uint32_t)RHI::TextureUsage::TEXTURE_BINDING)
				.consume(RDG::TextureInfo::ConsumeEntry{ RDG::TextureInfo::ConsumeType::TextureBinding }
					.setSubresource(0, 1, 0, 1)
					.addStage((uint32_t)RHI::PipelineStages::FRAGMENT_SHADER_BIT));

			reflector.addOutput("Combined")
				.isTexture()
				.withUsages((uint32_t)RHI::TextureUsage::COLOR_ATTACHMENT)
				.withFormat(RHI::TextureFormat::RGBA32_FLOAT)
				.consume(RDG::TextureInfo::ConsumeEntry{ RDG::TextureInfo::ConsumeType::ColorAttachment }
					.setSubresource(0, 1, 0, 1)
					.enableDepthWrite(false)
					.setAttachmentLoc(0)
					.setDepthCompareFn(RHI::CompareFunction::ALWAYS));

			return reflector;
		}

		struct PushConstant {
			Math::vec2	view_size;
			int			hiz_mip_levels;
			uint32_t	max_iteration = 100;
			int			strategy = 0;
			int			sample_batch;
			int			hilum_mip_levels;
			float		max_thickness = 0.001;
			Math::mat4	InvProjMat;
			Math::mat4	ProjMat;
			Math::mat4	TransInvViewMat;
		} pConst;

		virtual auto renderUI() noexcept -> void override {
			{	int strategy = pConst.strategy;
				ImGui::DragInt("Strategy", &strategy, 1, 0, 4);
				pConst.strategy = strategy;
			}
			{	int max_iteration = pConst.max_iteration;
				ImGui::DragInt("Max Iteration", &max_iteration, 1, 0, 1000);
				pConst.max_iteration = max_iteration;
			}
			{	float max_thickness = pConst.max_thickness;
				ImGui::DragFloat("Max Thickness", &max_thickness, 0.01);
				pConst.max_thickness = max_thickness;
			}
		}

		virtual auto execute(RDG::RenderContext* context, RDG::RenderData const& renderData) noexcept -> void {

			GFX::Texture* base_color = renderData.getTexture("BaseColor");
			GFX::Texture* light_projection = renderData.getTexture("LightProjection");
			GFX::Texture* hi_lumin = renderData.getTexture("HiLumin");
			GFX::Texture* hi_z = renderData.getTexture("HiZ");
			GFX::Texture* out = renderData.getTexture("Combined");
			GFX::Texture* depthLum = renderData.getTexture("DepthLumin");
			GFX::Texture* normalWS = renderData.getTexture("NormalWS");

			renderPassDescriptor = {
				{ RHI::RenderPassColorAttachment{
					out->getRTV(0, 0, 1),
					nullptr, {0,0,0,1}, RHI::LoadOp::CLEAR, RHI::StoreOp::STORE }},
				RHI::RenderPassDepthStencilAttachment{},
			};

			if (hi_lumin_sampler == nullptr) {
				Core::GUID hil_sampler, hiz_sampler, basecolor;
				RHI::SamplerDescriptor hil_desc, hiz_desc, basecolor_desc;
				hil_desc.maxLod = hi_lumin->texture->mipLevelCount();
				hiz_desc.maxLod = hi_z->texture->mipLevelCount();
				hiz_desc.magFilter = RHI::FilterMode::NEAREST;
				hiz_desc.minFilter = RHI::FilterMode::NEAREST;
				hiz_desc.mipmapFilter = RHI::MipmapFilterMode::NEAREST;
				basecolor_desc.magFilter = RHI::FilterMode::LINEAR;
				basecolor_desc.minFilter = RHI::FilterMode::LINEAR;
				hil_sampler = Core::ResourceManager::get()->requestRuntimeGUID<GFX::Sampler>();
				hiz_sampler = Core::ResourceManager::get()->requestRuntimeGUID<GFX::Sampler>();
				basecolor = Core::ResourceManager::get()->requestRuntimeGUID<GFX::Sampler>();
				GFX::GFXManager::get()->registerSamplerResource(hil_sampler, hil_desc);
				GFX::GFXManager::get()->registerSamplerResource(hiz_sampler, hiz_desc);
				GFX::GFXManager::get()->registerSamplerResource(basecolor, basecolor_desc);
				hi_lumin_sampler = Core::ResourceManager::get()->getResource<GFX::Sampler>(hil_sampler);
				hi_z_sampler = Core::ResourceManager::get()->getResource<GFX::Sampler>(hiz_sampler);
				basecolor_sampler = Core::ResourceManager::get()->getResource<GFX::Sampler>(basecolor);
			}

			RHI::RenderPassEncoder* encoder = beginPass(context, out);

			RHI::Sampler* defaultSampler = Core::ResourceManager::get()->getResource<GFX::Sampler>(GFX::GFXManager::get()->commonSampler.defaultSampler)->sampler.get();

			getBindGroup(context, 0)->updateBinding(std::vector<RHI::BindGroupEntry>{
				RHI::BindGroupEntry{ 0,RHI::BindingResource(base_color->getSRV(0,1,0,1), basecolor_sampler->sampler.get()) },
				RHI::BindGroupEntry{ 1,RHI::BindingResource(depthLum->getSRV(0,1,0,1), defaultSampler) },
				RHI::BindGroupEntry{ 2,RHI::BindingResource(normalWS->getSRV(0,1,0,1), defaultSampler) }, 
			});
			getBindGroup(context, 1)->updateBinding(std::vector<RHI::BindGroupEntry>{
				RHI::BindGroupEntry{ 0,RHI::BindingResource(hi_lumin->getSRV(0,hi_lumin->texture->mipLevelCount(),0,1), hi_lumin_sampler->sampler.get()) },
				RHI::BindGroupEntry{ 1,RHI::BindingResource(hi_z->getSRV(0,hi_z->texture->mipLevelCount(),0,1), hi_z_sampler->sampler.get()) },
				RHI::BindGroupEntry{ 2,RHI::BindingResource(light_projection->getSRV(0,1,0,1), defaultSampler) },
			});

			{
				std::vector<RHI::BindGroupEntry>* set_0_entries = renderData.getBindGroupEntries("CommonScene");
				pConst.view_size = Math::vec2(base_color->texture->width(), base_color->texture->height());
				pConst.hiz_mip_levels = hi_z->texture->mipLevelCount();
				pConst.hilum_mip_levels = hi_lumin->texture->mipLevelCount();
				pConst.sample_batch = renderData.getUInt("AccumIdx");
				
				SRenderer::CameraData* cd = reinterpret_cast<SRenderer::CameraData*>(renderData.getPtr("CameraData"));
				pConst.InvProjMat = Math::inverse((cd->projMat));
				pConst.ProjMat = cd->projMat;
				pConst.TransInvViewMat = Math::transpose(Math::inverse(cd->viewMat));
			}
			encoder->pushConstants(&pConst, (uint32_t)RHI::ShaderStages::FRAGMENT, 0, sizeof(PushConstant));

			dispatchFullScreen(context);

			encoder->end();
		}

		Core::GUID vert, frag;
		GFX::Sampler* hi_lumin_sampler = nullptr;
		GFX::Sampler* hi_z_sampler = nullptr;
		GFX::Sampler* basecolor_sampler = nullptr;
	};
}