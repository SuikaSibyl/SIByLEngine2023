module;
#include <vector>
#include <functional>
#include <filesystem>
#include <spirv_cross/spirv_glsl.hpp>
#include <spirv_cross/spirv_cross.hpp>
export module SE.GFX.Core:ShaderLoader;
import SE.Core.Log;
import SE.Core.Memory;
import SE.Core.Resource;
import SE.RHI;

namespace SIByL::GFX
{
	export enum struct ShaderLang {
		UNKNOWN,
		SPIRV,
		GLSL,
		HLSL,
		SLANG,
	};

	export struct ShaderReflection {
		enum struct ResourceType {
			Undefined,
			UniformBuffer,
			StorageBuffer,
			StorageImages,
			SampledImages,
			AccelerationStructure,
		};
		enum struct ResourceFlag :uint32_t {
			None = 0,
			NotReadable = 1 << 0,
			NotWritable = 1 << 1,
		};
		using ResourceFlags = uint32_t;
		struct ResourceEntry {
			ResourceType type = ResourceType::Undefined;
			ResourceFlags flags = 0;
			RHI::ShaderStagesFlags stages	= 0;
		};
		struct PushConstantEntry {
			uint32_t index	= -1;
			uint32_t offset = -1;
			uint32_t range	= -1;
			RHI::ShaderStagesFlags stages	= 0;
		};
		std::vector<PushConstantEntry> pushConstant;
		std::vector<std::vector<ResourceEntry>> bindings;

		static auto toBindGroupLayoutDescriptor(std::vector<ResourceEntry> const& bindings) noexcept -> RHI::BindGroupLayoutDescriptor;
		auto operator+(ShaderReflection const& reflection) const ->ShaderReflection;
	};

	inline auto combineResourceFlags(ShaderReflection::ResourceFlags a, ShaderReflection::ResourceFlags b) noexcept -> ShaderReflection::ResourceFlags {
		ShaderReflection::ResourceFlags r = 0;
		if ((a | uint32_t(ShaderReflection::ResourceFlag::NotReadable)) != 0 && (b | uint32_t(ShaderReflection::ResourceFlag::NotReadable)) != 0)
			r |= uint32_t(ShaderReflection::ResourceFlag::NotReadable);
		if ((a | uint32_t(ShaderReflection::ResourceFlag::NotWritable)) != 0 && (b | uint32_t(ShaderReflection::ResourceFlag::NotWritable)) != 0)
			r |= uint32_t(ShaderReflection::ResourceFlag::NotWritable);
		return r;
	}

	inline auto SPIRV_TO_Reflection(Core::Buffer* code, RHI::ShaderStages stage) noexcept -> ShaderReflection;

	export struct ShaderLoader {

	};

	export struct ShaderLoader_SPIRV {

	};

	export struct ShaderLoader_GLSL {

	};

	export struct ShaderLoader_HLSL {

	};

	export struct ShaderLoader_SLANG {

	};

#pragma region IMPL_SPIRV_2_REFLECTION

	auto SPIRV_TO_Reflection(Core::Buffer* code, RHI::ShaderStages stage) noexcept -> ShaderReflection {
		ShaderReflection reflection = {};
		std::vector<uint32_t> spirv_binary(code->size / sizeof(uint32_t));
		memcpy(spirv_binary.data(), code->data, code->size);
		spirv_cross::CompilerGLSL glsl(std::move(spirv_binary));
		// The SPIR-V is now parsed, and we can perform reflection on it.
		spirv_cross::ShaderResources resources = glsl.get_shader_resources();
		// add resource entry
		auto addResourceEntry = [&](ShaderReflection::ResourceEntry const& entry, int set, int binding) {
			if (reflection.bindings.size() <= set)
				reflection.bindings.resize(set + 1);
			if (reflection.bindings[set].size() <= binding)
				reflection.bindings[set].resize(binding + 1);
			reflection.bindings[set][binding] = entry;
		};
		// Get all uniform buffers in the shader.
		for (auto& resource : resources.uniform_buffers) {
			unsigned set = glsl.get_decoration(resource.id, spv::DecorationDescriptorSet);
			unsigned binding = glsl.get_decoration(resource.id, spv::DecorationBinding);
			ShaderReflection::ResourceFlags flag = uint32_t(ShaderReflection::ResourceFlag::None);
			if (glsl.get_decoration(resource.id, spv::DecorationNonReadable))
				flag |= uint32_t(ShaderReflection::ResourceFlag::NotReadable);
			if (glsl.get_decoration(resource.id, spv::DecorationNonWritable))
				flag |= uint32_t(ShaderReflection::ResourceFlag::NotWritable);
			addResourceEntry({ ShaderReflection::ResourceType::UniformBuffer, flag, uint32_t(stage) }, set, binding);
		}
		// Get all storage buffers in the shader.
		for (auto& resource : resources.storage_buffers) {
			unsigned set = glsl.get_decoration(resource.id, spv::DecorationDescriptorSet);
			unsigned binding = glsl.get_decoration(resource.id, spv::DecorationBinding);
			ShaderReflection::ResourceFlags flag = uint32_t(ShaderReflection::ResourceFlag::None);
			if (glsl.get_decoration(resource.id, spv::DecorationNonReadable))
				flag |= uint32_t(ShaderReflection::ResourceFlag::NotReadable);
			if (glsl.get_decoration(resource.id, spv::DecorationNonWritable))
				flag |= uint32_t(ShaderReflection::ResourceFlag::NotWritable);
			addResourceEntry({ ShaderReflection::ResourceType::StorageBuffer, flag, uint32_t(stage) }, set, binding);
		}
		// Get all storage images in the shader.
		for (auto& resource : resources.storage_images) {
			unsigned set = glsl.get_decoration(resource.id, spv::DecorationDescriptorSet);
			unsigned binding = glsl.get_decoration(resource.id, spv::DecorationBinding);
			ShaderReflection::ResourceFlags flag = uint32_t(ShaderReflection::ResourceFlag::None);
			if (glsl.get_decoration(resource.id, spv::DecorationNonReadable))
				flag |= uint32_t(ShaderReflection::ResourceFlag::NotReadable);
			if (glsl.get_decoration(resource.id, spv::DecorationNonWritable))
				flag |= uint32_t(ShaderReflection::ResourceFlag::NotWritable);
			addResourceEntry({ ShaderReflection::ResourceType::StorageImages, flag, uint32_t(stage) }, set, binding);
		}
		// Get all sampled images in the shader.
		for (auto& resource : resources.sampled_images) {
			unsigned set = glsl.get_decoration(resource.id, spv::DecorationDescriptorSet);
			unsigned binding = glsl.get_decoration(resource.id, spv::DecorationBinding);
			ShaderReflection::ResourceFlags flag = uint32_t(ShaderReflection::ResourceFlag::None);
			if (glsl.get_decoration(resource.id, spv::DecorationNonReadable))
				flag |= uint32_t(ShaderReflection::ResourceFlag::NotReadable);
			if (glsl.get_decoration(resource.id, spv::DecorationNonWritable))
				flag |= uint32_t(ShaderReflection::ResourceFlag::NotWritable);
			addResourceEntry({ ShaderReflection::ResourceType::SampledImages, flag, uint32_t(stage) }, set, binding);
		}
		// Get all accleration structures in the shader.
		for (auto& resource : resources.acceleration_structures) {
			unsigned set = glsl.get_decoration(resource.id, spv::DecorationDescriptorSet);
			unsigned binding = glsl.get_decoration(resource.id, spv::DecorationBinding);
			ShaderReflection::ResourceFlags flag = uint32_t(ShaderReflection::ResourceFlag::None);
			if (glsl.get_decoration(resource.id, spv::DecorationNonReadable))
				flag |= uint32_t(ShaderReflection::ResourceFlag::NotReadable);
			if (glsl.get_decoration(resource.id, spv::DecorationNonWritable))
				flag |= uint32_t(ShaderReflection::ResourceFlag::NotWritable);
			addResourceEntry({ ShaderReflection::ResourceType::AccelerationStructure, flag, uint32_t(stage) }, set, binding);
		}
		// Get all push constants in the shader.
		for (auto& resource : resources.push_constant_buffers) {
			auto ranges = glsl.get_active_buffer_ranges(resource.id);
			for (auto& range : ranges)
				reflection.pushConstant.emplace_back(ShaderReflection::PushConstantEntry{ range.index, uint32_t(range.offset), uint32_t(range.range), (uint32_t)stage });
		}
		return reflection;
	}

#pragma endregion

#pragma region IMPL_SHADER_REFLECTION
	
	auto ShaderReflection::toBindGroupLayoutDescriptor(std::vector<ResourceEntry> const& bindings) noexcept -> RHI::BindGroupLayoutDescriptor {
		RHI::BindGroupLayoutDescriptor descriptor;
		for (uint32_t i = 0; i < bindings.size(); ++i) {
			auto const& bind = bindings[i];
			if (bind.type == ResourceType::UniformBuffer)
				descriptor.entries.push_back(RHI::BindGroupLayoutEntry{ i, bind.stages, RHI::BufferBindingLayout{RHI::BufferBindingType::UNIFORM} });
			else if (bind.type == ResourceType::StorageBuffer)
				descriptor.entries.push_back(RHI::BindGroupLayoutEntry{ i, bind.stages, RHI::BufferBindingLayout{RHI::BufferBindingType::STORAGE} });
			else if (bind.type == ResourceType::StorageImages)
				descriptor.entries.push_back(RHI::BindGroupLayoutEntry{ i, bind.stages, RHI::StorageTextureBindingLayout{} });
			else if (bind.type == ResourceType::AccelerationStructure)
				descriptor.entries.push_back(RHI::BindGroupLayoutEntry{ i, bind.stages, RHI::AccelerationStructureBindingLayout{} });
			else if (bind.type == ResourceType::SampledImages)
				descriptor.entries.push_back(RHI::BindGroupLayoutEntry{ i, bind.stages, RHI::BindlessTexturesBindingLayout{} });
		}
		return descriptor;
	}

	auto ShaderReflection::operator+(ShaderReflection const& reflection) const ->ShaderReflection {
		ShaderReflection added = *this;
		for (int set = 0; set < reflection.bindings.size(); ++set) {
			if (added.bindings.size() <= set) {
				added.bindings.resize(set + 1);
				added.bindings[set] = reflection.bindings[set];
			}
			else {
				for (int binding = 0; binding < reflection.bindings[set].size(); ++binding) {
					if (added.bindings[set].size() <= binding) {
						added.bindings[set].resize(binding + 1);
						added.bindings[set][binding] = reflection.bindings[set][binding];
					}
					else {
						assert(added.bindings[set][binding].type == reflection.bindings[set][binding].type);
						added.bindings[set][binding].stages |= reflection.bindings[set][binding].stages;
						added.bindings[set][binding].flags = combineResourceFlags(added.bindings[set][binding].flags, reflection.bindings[set][binding].flags);
					}
				}
			}
		}
		int this_id = 0;
		for (int i = 0; i < reflection.pushConstant.size(); ++i) {
			if (this_id == added.pushConstant.size())
				added.pushConstant.push_back(reflection.pushConstant[i]);
			else if (added.pushConstant[this_id].offset == added.pushConstant[i].offset) {
				added.pushConstant[this_id].stages |= added.pushConstant[i].stages;
			}
		}
		return added;
	}

#pragma endregion

}