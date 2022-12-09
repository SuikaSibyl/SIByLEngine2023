module;
#include <vector>
#include <array>
#include <unordered_map>
export module SE.RHI:Utility;
import :Interface;

namespace SIByL::RHI
{
	export struct ResourceBookKeeper {

		struct BufferConsumeDescriptor {
			Buffer* buffer;
			PipelineStageFlags pipelineStageFlags;
			AccessFlags accessFlags;
		};

		auto consumeBuffer(BufferConsumeDescriptor const& desc) noexcept -> void {
			
		}

		struct TextureConsumeDescriptor {
			Texture* texture;
			ImageSubresourceRange subresourceRange;
			PipelineStageFlags pipelineStageFlags;
			AccessFlags accessFlags;
			TextureLayout layout;
		};

		auto consumeTexture() {

		}

		auto beginNewPage() noexcept -> void {
			bufferBookKeeping.clear();
			textureBookKeeping.clear();
		}

		auto beginPipelineDescript() noexcept -> void {

		}

		auto endPipelineDescript() noexcept -> void {

		}

		auto forwardStatus() noexcept -> void {
		}

		auto generateBarrierDescriptors() noexcept -> std::vector<BarrierDescriptor> {
			// Create multiple barrier descriptor according to src stage maks.
			std::unordered_map<PipelineStageFlags, BarrierDescriptor> barrierDescriptorMap;
			for (auto bufferConsumePair : bufferBookKeeping) {
				// if current consume state has a null buffer, means the buffer is
				// not consumed in this pipeline, so just omit it.
				if (bufferConsumePair.second[1].buffer == nullptr)
					continue;
				auto descriptorMapIter = barrierDescriptorMap.find(bufferConsumePair.second[0].pipelineStageFlags);
				if (descriptorMapIter == barrierDescriptorMap.end()) {
					barrierDescriptorMap[bufferConsumePair.second[0].pipelineStageFlags] = BarrierDescriptor{
						bufferConsumePair.second[0].pipelineStageFlags, // srcStage
						bufferConsumePair.second[1].pipelineStageFlags, // dstStage
						(DependencyTypeFlags)DependencyType::NONE,		// dependencyType
						{}, {}, {}
					};
					descriptorMapIter = barrierDescriptorMap.find(bufferConsumePair.second[0].pipelineStageFlags);
				}
				BarrierDescriptor& barrierDesc = descriptorMapIter->second;
				barrierDesc.bufferMemoryBarriers.emplace_back(BufferMemoryBarrierDescriptor{
					bufferConsumePair.second[1].buffer, // buffer
					bufferConsumePair.second[0].accessFlags,
					bufferConsumePair.second[1].accessFlags,
					});
			}
			for (auto textureConsumePair : textureBookKeeping) {
				// if current consume state has a null texture, means the texture is
				// not consumed in this pipeline, so just omit it.
				if (textureConsumePair.second[1].texture == nullptr)
					continue;
				auto descriptorMapIter = barrierDescriptorMap.find(textureConsumePair.second[0].pipelineStageFlags);
				if (descriptorMapIter == barrierDescriptorMap.end()) {
					barrierDescriptorMap[textureConsumePair.second[0].pipelineStageFlags] = BarrierDescriptor{
						textureConsumePair.second[0].pipelineStageFlags, // srcStage
						textureConsumePair.second[1].pipelineStageFlags, // dstStage
						(DependencyTypeFlags)DependencyType::NONE,		// dependencyType
						{}, {}, {}
					};
					descriptorMapIter = barrierDescriptorMap.find(textureConsumePair.second[0].pipelineStageFlags);
				}
				BarrierDescriptor& barrierDesc = descriptorMapIter->second;
				barrierDesc.textureMemoryBarriers.emplace_back(TextureMemoryBarrierDescriptor{
					textureConsumePair.second[1].texture, // texture
					textureConsumePair.second[1].subresourceRange, // TODO: different subresource
					textureConsumePair.second[0].accessFlags,
					textureConsumePair.second[1].accessFlags,
					textureConsumePair.second[0].layout,
					textureConsumePair.second[1].layout
					});
			}
			// Push all descriptors in the map to the vector.
			std::vector<BarrierDescriptor> barrierDescriptors(barrierDescriptorMap.size());
			for (auto& pair : barrierDescriptorMap)
				barrierDescriptors.emplace_back(pair.second);
			return barrierDescriptors;
		}

		std::unordered_map<Buffer*, std::array<BufferConsumeDescriptor, 2>> bufferBookKeeping;
		std::unordered_map<Texture*, std::array<TextureConsumeDescriptor, 2>> textureBookKeeping;
	};
}