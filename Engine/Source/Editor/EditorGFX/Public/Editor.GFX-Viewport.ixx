module;
#include <string>
#include <typeinfo>
#include <imgui.h>
#include <imgui_internal.h>
export module SE.Editor.GFX:Viewport;
import :Utils;
import :TextureFragment;
import SE.Editor.Core;
import SE.Core.Resource;
import SE.Core.Misc;
import SE.Core.Log;
import SE.RHI;
import SE.Image;
import SE.GFX;
import SE.Math.Geometric;
import SE.Platform.Window;

namespace SIByL::Editor
{
	auto captureImage(Core::GUID src) noexcept -> void {
		RHI::RHILayer* rhiLayer = ImGuiLayer::get()->rhiLayer;
		Platform::Window* mainWindow = rhiLayer->getRHILayerDescriptor().windowBinded;
		GFX::Texture* tex = Core::ResourceManager::get()->getResource<GFX::Texture>(src);
		size_t width = tex->texture->width();
		size_t height = tex->texture->height();

		RHI::TextureFormat format;
		size_t pixelSize;
		if (tex->texture->format() == RHI::TextureFormat::RGBA32_FLOAT) {
			format = RHI::TextureFormat::RGBA32_FLOAT;
			pixelSize = sizeof(Math::vec4);
		}
		else if (tex->texture->format() == RHI::TextureFormat::RGBA8_UNORM) {
			format = RHI::TextureFormat::RGBA8_UNORM;
			pixelSize = sizeof(uint8_t) * 4;
		}
		else {
			Core::LogManager::Error("Editor :: ViewportWidget :: captureImage() :: Unsupported format to capture.");
			return;
		}

		std::unique_ptr<RHI::CommandEncoder> commandEncoder = rhiLayer->getDevice()->createCommandEncoder({});

		static Core::GUID copyDst = 0;
		if (copyDst == 0) {
			copyDst = Core::ResourceManager::get()->requestRuntimeGUID<GFX::Texture>();
			RHI::TextureDescriptor desc{
				{width,height,1},
				1, 1, RHI::TextureDimension::TEX2D,
				format,
				(uint32_t)RHI::TextureUsage::COPY_DST | (uint32_t)RHI::TextureUsage::TEXTURE_BINDING,
				{ format },
				RHI::TextureFlags::HOSTI_VISIBLE
			};
			GFX::GFXManager::get()->registerTextureResource(copyDst, desc);

			GFX::Texture* copydst = Core::ResourceManager::get()->getResource<GFX::Texture>(copyDst);
			commandEncoder->pipelineBarrier(RHI::BarrierDescriptor{
				(uint32_t)RHI::PipelineStages::ALL_GRAPHICS_BIT,
				(uint32_t)RHI::PipelineStages::TRANSFER_BIT,
				(uint32_t)RHI::DependencyType::NONE,
				{}, {},
				{ RHI::TextureMemoryBarrierDescriptor{
					copydst->texture.get(),
					RHI::ImageSubresourceRange{(uint32_t)RHI::TextureAspect::COLOR_BIT, 0,1,0,1},
					(uint32_t)RHI::AccessFlagBits::NONE,
					(uint32_t)RHI::AccessFlagBits::TRANSFER_WRITE_BIT,
					RHI::TextureLayout::SHADER_READ_ONLY_OPTIMAL,
					RHI::TextureLayout::TRANSFER_DST_OPTIMAL
				}}
				});
		}
		rhiLayer->getDevice()->waitIdle();
		commandEncoder->pipelineBarrier(RHI::BarrierDescriptor{
			(uint32_t)RHI::PipelineStages::FRAGMENT_SHADER_BIT,
			(uint32_t)RHI::PipelineStages::TRANSFER_BIT,
			(uint32_t)RHI::DependencyType::NONE,
			{}, {},
			{ RHI::TextureMemoryBarrierDescriptor{
				Core::ResourceManager::get()->getResource<GFX::Texture>(src)->texture.get(),
				RHI::ImageSubresourceRange{(uint32_t)RHI::TextureAspect::COLOR_BIT, 0,1,0,1},
				(uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT | (uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT,
				(uint32_t)RHI::AccessFlagBits::TRANSFER_READ_BIT,
				RHI::TextureLayout::SHADER_READ_ONLY_OPTIMAL,
				RHI::TextureLayout::TRANSFER_SRC_OPTIMAL
			}}
			});
		commandEncoder->copyTextureToTexture(
			RHI::ImageCopyTexture{
				Core::ResourceManager::get()->getResource<GFX::Texture>(src)->texture.get()
			},
			RHI::ImageCopyTexture{
				Core::ResourceManager::get()->getResource<GFX::Texture>(copyDst)->texture.get()
			},
			RHI::Extend3D{ uint32_t(width), uint32_t(height), 1 }
		);
		commandEncoder->pipelineBarrier(RHI::BarrierDescriptor{
			(uint32_t)RHI::PipelineStages::TRANSFER_BIT,
			(uint32_t)RHI::PipelineStages::FRAGMENT_SHADER_BIT,
			(uint32_t)RHI::DependencyType::NONE,
			{}, {},
			{ RHI::TextureMemoryBarrierDescriptor{
				Core::ResourceManager::get()->getResource<GFX::Texture>(src)->texture.get(),
				RHI::ImageSubresourceRange{(uint32_t)RHI::TextureAspect::COLOR_BIT, 0,1,0,1},
				(uint32_t)RHI::AccessFlagBits::TRANSFER_READ_BIT,
				(uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT | (uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT,
				RHI::TextureLayout::TRANSFER_SRC_OPTIMAL,
				RHI::TextureLayout::SHADER_READ_ONLY_OPTIMAL
			}}
			});
		commandEncoder->pipelineBarrier(RHI::BarrierDescriptor{
			(uint32_t)RHI::PipelineStages::TRANSFER_BIT,
			(uint32_t)RHI::PipelineStages::HOST_BIT,
			(uint32_t)RHI::DependencyType::NONE,
			{}, {},
			{ RHI::TextureMemoryBarrierDescriptor{
				Core::ResourceManager::get()->getResource<GFX::Texture>(copyDst)->texture.get(),
				RHI::ImageSubresourceRange{(uint32_t)RHI::TextureAspect::COLOR_BIT, 0,1,0,1},
				(uint32_t)RHI::AccessFlagBits::TRANSFER_WRITE_BIT,
				(uint32_t)RHI::AccessFlagBits::HOST_READ_BIT,
				RHI::TextureLayout::TRANSFER_DST_OPTIMAL,
				RHI::TextureLayout::TRANSFER_DST_OPTIMAL
			}}
			});
		rhiLayer->getDevice()->getGraphicsQueue()->submit({ commandEncoder->finish({}) });
		rhiLayer->getDevice()->waitIdle();
		std::future<bool> mapped = Core::ResourceManager::get()->getResource<GFX::Texture>(copyDst)->texture->mapAsync(
			(uint32_t)RHI::MapMode::READ, 0, width * height * pixelSize);
		if (mapped.get()) {
			void* data = Core::ResourceManager::get()->getResource<GFX::Texture>(copyDst)->texture->getMappedRange(0, width * height * pixelSize);
			if (tex->texture->format() == RHI::TextureFormat::RGBA32_FLOAT) {
				std::string filepath = mainWindow->saveFile("", Core::WorldTimePoint::get().to_string() + ".hdr");
				Image::HDR::writeHDR(filepath, width, height, 4, reinterpret_cast<float*>(data));
			}
			else if (tex->texture->format() == RHI::TextureFormat::RGBA8_UNORM) {
				std::string filepath = mainWindow->saveFile("", Core::WorldTimePoint::get().to_string() + ".bmp");
				Image::BMP::writeBMP(filepath, width, height, 4, reinterpret_cast<float*>(data));
			}
			Core::ResourceManager::get()->getResource<GFX::Texture>(copyDst)->texture->unmap();
		}
	}

	export struct ViewportWidget :public Widget {

		auto setTarget(std::string const& name, GFX::Texture* tex) noexcept -> void {
			this->name = name;
			texture = tex;
		}

		GFX::Texture* texture;
		std::string name = "Viewport";

		/** draw gui*/
		virtual auto onDrawGui() noexcept -> void override {
			ImGui::Begin(name.c_str() , 0, ImGuiWindowFlags_MenuBar);

			ImGui::PushItemWidth(ImGui::GetFontSize() * -12);
			if (ImGui::BeginMenuBar()) {
				if (ImGui::Button("capture")) {
					if (texture) {
						captureImage(texture->guid);
					}
				}
				//menuBarSize = ImGui::GetWindowSize();
				ImGui::EndMenuBar();
			}
			ImGui::PopItemWidth();
			commonOnDrawGui();
			auto currPos = ImGui::GetCursorPos();
			info.mousePos = ImGui::GetMousePos();
			info.mousePos.x -= info.windowPos.x + currPos.x;
			info.mousePos.y -= info.windowPos.y + currPos.y;
			//info.contentPos = info.windowPos;
			//info.contentPos.x += currPos.x;
			//info.contentPos.y += currPos.y;

			if (texture) {
				ImGui::Image(
					Editor::TextureUtils::getImGuiTexture(texture->guid)->getTextureID(),
					{ (float)texture->texture->width(),(float)texture->texture->height() },
					{ 0,0 }, { 1, 1 });
			}

			ImGui::End();
		}
	};
}
