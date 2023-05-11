#pragma once

#include <SE.GFX-Loader.ShaderLoader.hpp>
#include <Print/SE.Core.Log.hpp>
#include <Memory/SE.Core.Memory.hpp>
#include <Resource/SE.Core.Resource.hpp>
#include <vector>
#include <functional>
#include <spirv_cross/spirv_cross.hpp>
#include <spirv_cross/spirv_glsl.hpp>

namespace SIByL::GFX {
inline auto combineResourceFlags(ShaderReflection::ResourceFlags a,
                                 ShaderReflection::ResourceFlags b) noexcept
    -> ShaderReflection::ResourceFlags {
  ShaderReflection::ResourceFlags r = 0;
  if ((a | uint32_t(ShaderReflection::ResourceFlag::NotReadable)) != 0 &&
      (b | uint32_t(ShaderReflection::ResourceFlag::NotReadable)) != 0)
    r |= uint32_t(ShaderReflection::ResourceFlag::NotReadable);
  if ((a | uint32_t(ShaderReflection::ResourceFlag::NotWritable)) != 0 &&
      (b | uint32_t(ShaderReflection::ResourceFlag::NotWritable)) != 0)
    r |= uint32_t(ShaderReflection::ResourceFlag::NotWritable);
  return r;
}

auto SPIRV_TO_Reflection(Core::Buffer* code, RHI::ShaderStages stage) noexcept
    -> ShaderReflection {
  ShaderReflection reflection = {};
  std::vector<uint32_t> spirv_binary(code->size / sizeof(uint32_t));
  memcpy(spirv_binary.data(), code->data, code->size);
  spirv_cross::CompilerGLSL glsl(std::move(spirv_binary));
  // The SPIR-V is now parsed, and we can perform reflection on it.
  spirv_cross::ShaderResources resources = glsl.get_shader_resources();
  // add resource entry
  auto addResourceEntry = [&](ShaderReflection::ResourceEntry const& entry,
                              int set, int binding) {
    if (reflection.bindings.size() <= set) reflection.bindings.resize(set + 1);
    if (reflection.bindings[set].size() <= binding)
      reflection.bindings[set].resize(binding + 1);
    reflection.bindings[set][binding] = entry;
  };
  // Get all uniform buffers in the shader.
  for (auto& resource : resources.uniform_buffers) {
    unsigned set =
        glsl.get_decoration(resource.id, spv::DecorationDescriptorSet);
    unsigned binding = glsl.get_decoration(resource.id, spv::DecorationBinding);
    ShaderReflection::ResourceFlags flag =
        uint32_t(ShaderReflection::ResourceFlag::None);
    if (glsl.get_decoration(resource.id, spv::DecorationNonReadable))
      flag |= uint32_t(ShaderReflection::ResourceFlag::NotReadable);
    if (glsl.get_decoration(resource.id, spv::DecorationNonWritable))
      flag |= uint32_t(ShaderReflection::ResourceFlag::NotWritable);
    addResourceEntry(
        {ShaderReflection::ResourceType::UniformBuffer, flag, uint32_t(stage)},
        set, binding);
  }
  // Get all storage buffers in the shader.
  for (auto& resource : resources.storage_buffers) {
    unsigned set =
        glsl.get_decoration(resource.id, spv::DecorationDescriptorSet);
    unsigned binding = glsl.get_decoration(resource.id, spv::DecorationBinding);
    ShaderReflection::ResourceFlags flag =
        uint32_t(ShaderReflection::ResourceFlag::None);
    if (glsl.get_decoration(resource.id, spv::DecorationNonReadable))
      flag |= uint32_t(ShaderReflection::ResourceFlag::NotReadable);
    if (glsl.get_decoration(resource.id, spv::DecorationNonWritable))
      flag |= uint32_t(ShaderReflection::ResourceFlag::NotWritable);
    addResourceEntry(
        {ShaderReflection::ResourceType::StorageBuffer, flag, uint32_t(stage)},
        set, binding);
  }
  // Get all storage images in the shader.
  for (auto& resource : resources.storage_images) {
    unsigned set =
        glsl.get_decoration(resource.id, spv::DecorationDescriptorSet);
    unsigned binding = glsl.get_decoration(resource.id, spv::DecorationBinding);
    ShaderReflection::ResourceFlags flag =
        uint32_t(ShaderReflection::ResourceFlag::None);
    if (glsl.get_decoration(resource.id, spv::DecorationNonReadable))
      flag |= uint32_t(ShaderReflection::ResourceFlag::NotReadable);
    if (glsl.get_decoration(resource.id, spv::DecorationNonWritable))
      flag |= uint32_t(ShaderReflection::ResourceFlag::NotWritable);
    addResourceEntry(
        {ShaderReflection::ResourceType::StorageImages, flag, uint32_t(stage)},
        set, binding);
  }
  // Get all sampled images in the shader.
  for (auto& resource : resources.sampled_images) {
    unsigned set =
        glsl.get_decoration(resource.id, spv::DecorationDescriptorSet);
    unsigned binding = glsl.get_decoration(resource.id, spv::DecorationBinding);
    ShaderReflection::ResourceFlags flag =
        uint32_t(ShaderReflection::ResourceFlag::None);
    if (glsl.get_decoration(resource.id, spv::DecorationNonReadable))
      flag |= uint32_t(ShaderReflection::ResourceFlag::NotReadable);
    if (glsl.get_decoration(resource.id, spv::DecorationNonWritable))
      flag |= uint32_t(ShaderReflection::ResourceFlag::NotWritable);
    addResourceEntry(
        {ShaderReflection::ResourceType::SampledImages, flag, uint32_t(stage)},
        set, binding);
  }
  // Get all accleration structures in the shader.
  for (auto& resource : resources.acceleration_structures) {
    unsigned set =
        glsl.get_decoration(resource.id, spv::DecorationDescriptorSet);
    unsigned binding = glsl.get_decoration(resource.id, spv::DecorationBinding);
    ShaderReflection::ResourceFlags flag =
        uint32_t(ShaderReflection::ResourceFlag::None);
    if (glsl.get_decoration(resource.id, spv::DecorationNonReadable))
      flag |= uint32_t(ShaderReflection::ResourceFlag::NotReadable);
    if (glsl.get_decoration(resource.id, spv::DecorationNonWritable))
      flag |= uint32_t(ShaderReflection::ResourceFlag::NotWritable);
    addResourceEntry({ShaderReflection::ResourceType::AccelerationStructure,
                      flag, uint32_t(stage)},
                     set, binding);
  }
  // Get all push constants in the shader.
  for (auto& resource : resources.push_constant_buffers) {
    auto ranges = glsl.get_active_buffer_ranges(resource.id);
    for (auto& range : ranges)
      reflection.pushConstant.emplace_back(ShaderReflection::PushConstantEntry{
          range.index, uint32_t(range.offset), uint32_t(range.range),
          (uint32_t)stage});
  }
  return reflection;
}

auto ShaderReflection::toBindGroupLayoutDescriptor(
    std::vector<ResourceEntry> const& bindings) noexcept
    -> RHI::BindGroupLayoutDescriptor {
  RHI::BindGroupLayoutDescriptor descriptor;
  for (uint32_t i = 0; i < bindings.size(); ++i) {
    auto const& bind = bindings[i];
    if (bind.type == ResourceType::UniformBuffer)
      descriptor.entries.push_back(RHI::BindGroupLayoutEntry{
          i, bind.stages,
          RHI::BufferBindingLayout{RHI::BufferBindingType::UNIFORM}});
    else if (bind.type == ResourceType::StorageBuffer)
      descriptor.entries.push_back(RHI::BindGroupLayoutEntry{
          i, bind.stages,
          RHI::BufferBindingLayout{RHI::BufferBindingType::STORAGE}});
    else if (bind.type == ResourceType::StorageImages)
      descriptor.entries.push_back(RHI::BindGroupLayoutEntry{
          i, bind.stages, RHI::StorageTextureBindingLayout{}});
    else if (bind.type == ResourceType::AccelerationStructure)
      descriptor.entries.push_back(RHI::BindGroupLayoutEntry{
          i, bind.stages, RHI::AccelerationStructureBindingLayout{}});
    else if (bind.type == ResourceType::SampledImages)
      descriptor.entries.push_back(RHI::BindGroupLayoutEntry{
          i, bind.stages, RHI::BindlessTexturesBindingLayout{}});
  }
  return descriptor;
}

auto ShaderReflection::operator+(ShaderReflection const& reflection) const
    -> ShaderReflection {
  ShaderReflection added = *this;
  for (int set = 0; set < reflection.bindings.size(); ++set) {
    if (added.bindings.size() <= set) {
      added.bindings.resize(set + 1);
      added.bindings[set] = reflection.bindings[set];
    } else {
      for (int binding = 0; binding < reflection.bindings[set].size();
           ++binding) {
        if (added.bindings[set].size() <= binding) {
          added.bindings[set].resize(binding + 1);
          added.bindings[set][binding] = reflection.bindings[set][binding];
        } else {
          assert(added.bindings[set][binding].type ==
                 reflection.bindings[set][binding].type);
          added.bindings[set][binding].stages |=
              reflection.bindings[set][binding].stages;
          added.bindings[set][binding].flags =
              combineResourceFlags(added.bindings[set][binding].flags,
                                   reflection.bindings[set][binding].flags);
        }
      }
    }
  }
  int this_id = 0;
  for (int i = 0; i < reflection.pushConstant.size(); ++i) {
    // if added_pushconstants is smaller
    if (this_id >= added.pushConstant.size())
      added.pushConstant.push_back(reflection.pushConstant[i]);
    else if (added.pushConstant[this_id].offset ==
             reflection.pushConstant[i].offset) {
      added.pushConstant[this_id].stages |= reflection.pushConstant[i].stages;
    } else {
      added.pushConstant.push_back(reflection.pushConstant[i]);
    }
  }
  return added;
}
}  // namespace SIByL::GFX