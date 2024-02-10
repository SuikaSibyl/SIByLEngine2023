#define DLIB_EXPORT
#include <se.gfx.hpp>
#undef DLIB_EXPORT
#include "spirv_reflect/spirv_reflect.h"
#include <slang.h>
#include <slang-com-ptr.h>
#include <filesystem>

namespace se::gfx {
// ===========================================================================
// Base class definition


// Base class definition
// ===========================================================================
// Shader Resource

inline auto combineResourceFlags(
  ShaderReflection::ResourceFlags a,
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

inline auto compare_pushconstant(
  ShaderReflection::PushConstantEntry& a,
  ShaderReflection::PushConstantEntry& b) noexcept -> bool { 
  return a.offset < b.offset;
}

inline auto rearrange_pushconstant(
  ShaderReflection& reflection) noexcept
    -> void {
    if (reflection.pushConstant.size() == 0) return;
    std::sort(reflection.pushConstant.begin(), reflection.pushConstant.end(),
              compare_pushconstant);
    while (true) {
      bool should_break = false;
      for (auto iter = reflection.pushConstant.begin();
        iter != reflection.pushConstant.end();) {
        // break when reach the end of the push constants
        auto iter_next = iter + 1;
        if (iter_next == reflection.pushConstant.end()) {
          should_break = true;
          break;
        } else {
          if (iter->offset + iter->range <= iter_next->offset) {
            iter->range = iter_next->offset + iter_next->range - iter->offset;
            reflection.pushConstant.erase(iter_next);
          } else {
            iter++;
          }
        }
      }
      if (should_break) break;
    }

    reflection.pushConstant[0].range += reflection.pushConstant[0].offset;
    reflection.pushConstant[0].offset = 0;
}

auto SPIRV_TO_Reflection(se::buffer* code, rhi::ShaderStageBit stage) noexcept
  -> ShaderReflection {
  ShaderReflection reflection = {};
  // add resource entry
  auto addResourceEntry = [&](ShaderReflection::ResourceEntry const& entry,
                              int set, int binding) {
    if (reflection.bindings.size() <= set) reflection.bindings.resize(set + 1);
    if (reflection.bindings[set].size() <= binding)
      reflection.bindings[set].resize(binding + 1);
    reflection.bindings[set][binding] = entry;
  };

  // Generate reflection data for a shader
  SpvReflectShaderModule module;
  SpvReflectResult result =
      spvReflectCreateShaderModule(code->size, code->data, &module);
  assert(result == SPV_REFLECT_RESULT_SUCCESS);
  
  // Enumerate and extract shader's input variables
  uint32_t var_count = 0;
  result = spvReflectEnumerateInputVariables(&module, &var_count, NULL);
  assert(result == SPV_REFLECT_RESULT_SUCCESS);
  SpvReflectInterfaceVariable** input_vars =
      (SpvReflectInterfaceVariable**)malloc(
          var_count * sizeof(SpvReflectInterfaceVariable*));
  result = spvReflectEnumerateInputVariables(&module, &var_count, input_vars);
  assert(result == SPV_REFLECT_RESULT_SUCCESS);
  
  // Output variables, descriptor bindings, descriptor sets, and push
  // constants can be enumerated and extracted using a similar mechanism.
  for (int i = 0; i < module.descriptor_binding_count; ++i) {
    auto const& desc_set = module.descriptor_sets[i];
    for (int j = 0; j < desc_set.binding_count; ++j) {
      auto const& binding = desc_set.bindings[j];
      ShaderReflection::ResourceFlags flag =
          uint32_t(ShaderReflection::ResourceFlag::None);
      ShaderReflection::ResourceEntry entry;
      switch (binding->descriptor_type) {
        case SPV_REFLECT_DESCRIPTOR_TYPE_SAMPLER:
          entry.type = ShaderReflection::ResourceType::Sampler; break;
        case SPV_REFLECT_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
          entry.type = ShaderReflection::ResourceType::SampledImages; break;
        case SPV_REFLECT_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
          entry.type = ShaderReflection::ResourceType::ReadonlyImage; break;
        case SPV_REFLECT_DESCRIPTOR_TYPE_STORAGE_IMAGE:
          entry.type = ShaderReflection::ResourceType::StorageImages; break;
        case SPV_REFLECT_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
          entry.type = ShaderReflection::ResourceType::UniformBuffer; break;
        case SPV_REFLECT_DESCRIPTOR_TYPE_STORAGE_BUFFER:
          entry.type = ShaderReflection::ResourceType::StorageBuffer; break;
        case SPV_REFLECT_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
          entry.type = ShaderReflection::ResourceType::UniformBuffer; break;
        case SPV_REFLECT_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC:
          entry.type = ShaderReflection::ResourceType::StorageBuffer; break;
        case SPV_REFLECT_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR:
          entry.type = ShaderReflection::ResourceType::AccelerationStructure; break;
        case SPV_REFLECT_DESCRIPTOR_TYPE_INPUT_ATTACHMENT:
        case SPV_REFLECT_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
        case SPV_REFLECT_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
        default: 
          root::print::error("SPIRV-Reflect :: Unexpected resource type");  
          break;
      }
      if (desc_set.bindings[j]->array.dims_count >= 1) {
        entry.arraySize = 1000;
      }
      entry.flags = flag;
      entry.stages = uint32_t(stage);
      addResourceEntry(entry, binding->set, binding->binding);
    }
  }
  // Push constants
  for (uint32_t i = 0; i < module.push_constant_block_count; ++i) {
    auto const& block = module.push_constant_blocks[i];
    reflection.pushConstant.emplace_back(
        ShaderReflection::PushConstantEntry{
            i, uint32_t(block.offset), uint32_t(block.size), (uint32_t)stage});
  }
  // Destroy the reflection data when no longer required.
  spvReflectDestroyShaderModule(&module);
  // rearrange push constants
  rearrange_pushconstant(reflection);
  return reflection;
}

auto ShaderReflection::toBindGroupLayoutDescriptor(
  std::vector<ResourceEntry> const& bindings) noexcept
  -> rhi::BindGroupLayoutDescriptor {
  rhi::BindGroupLayoutDescriptor descriptor;
  for (uint32_t i = 0; i < bindings.size(); ++i) {
    auto const& bind = bindings[i];
    if (bind.type == ResourceType::UniformBuffer)
      descriptor.entries.push_back(rhi::BindGroupLayoutEntry{
          i, bind.stages,
          rhi::BufferBindingLayout{rhi::BufferBindingType::UNIFORM}});
    else if (bind.type == ResourceType::StorageBuffer)
      descriptor.entries.push_back(rhi::BindGroupLayoutEntry{
          i, bind.stages,
          rhi::BufferBindingLayout{rhi::BufferBindingType::STORAGE}});
    else if (bind.type == ResourceType::StorageImages) {
      rhi::BindGroupLayoutEntry entry{i, bind.stages, rhi::StorageTextureBindingLayout{}};
      entry.array_size = bind.arraySize;
      descriptor.entries.emplace_back(entry);
    }
    else if (bind.type == ResourceType::AccelerationStructure)
      descriptor.entries.push_back(rhi::BindGroupLayoutEntry{
          i, bind.stages, rhi::AccelerationStructureBindingLayout{}});
    else if (bind.type == ResourceType::SampledImages)
      descriptor.entries.push_back(rhi::BindGroupLayoutEntry{
          i, bind.stages, rhi::BindlessTexturesBindingLayout{}});
    else if (bind.type == ResourceType::ReadonlyImage) {
      rhi::BindGroupLayoutEntry entry{i, bind.stages,
                                      rhi::TextureBindingLayout{}};
      entry.array_size = bind.arraySize;
      descriptor.entries.emplace_back(entry);
    }
    else if (bind.type == ResourceType::Sampler)
      descriptor.entries.push_back(rhi::BindGroupLayoutEntry{
          i, bind.stages, rhi::SamplerBindingLayout{}});
  }
  return descriptor;
}

auto ShaderReflection::operator+(
  ShaderReflection const& reflection) const
    -> ShaderReflection {
  ShaderReflection added = *this;
  added.bindingInfo.insert(reflection.bindingInfo.begin(),
                           reflection.bindingInfo.end());
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
        } 
        else if (reflection.bindings[set][binding].type ==
                   ResourceType::Undefined) {
            // SKIP
        } else if (added.bindings[set][binding].type ==
                   ResourceType::Undefined) {
          added.bindings[set][binding] = reflection.bindings[set][binding];
        }
        else {
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

auto ShaderModule::getName() const noexcept -> char const* {
    return shaderModule->getName().c_str();
}

// loader
// -------------------------------------------------

ShaderModuleLoader::result_type ShaderModuleLoader::operator()(
  ShaderModuleLoader::from_spirv_tag, 
  se::buffer* buffer,
  rhi::ShaderStageBit stage) {
  rhi::ShaderModuleDescriptor desc;
  desc.code = buffer;
  desc.name = "main";
  desc.stage = stage;
  auto ptr = std::make_shared<ShaderModule>();
  ptr->shaderModule = GFXContext::device->createShaderModule(desc);
  ptr->reflection = SPIRV_TO_Reflection(desc.code, desc.stage);
  return ptr;
}

ShaderModuleLoader::result_type ShaderModuleLoader::operator()(
  ShaderModuleLoader::from_glsl_tag,
  std::string const& filepath, rhi::ShaderStageBit stage,
  std::vector<std::string> const& argv) {
  std::filesystem::path path(filepath);
  std::filesystem::path filename = path.filename();
  std::filesystem::path extension = path.extension();
  filename = std::filesystem::path(filename.string().substr(
      0, filename.string().size() - extension.string().size()));
  std::string extension_str =
      extension.string().substr(1, extension.string().length());
  std::filesystem::path parent_path = path.parent_path();
  std::string cmdLine("glslangValidator --target-env vulkan1.2");
  std::string spirv_path = (parent_path / filename).string() + "_" + extension_str + ".spv";
  cmdLine += " " + std::string(filepath);
  cmdLine += " -o " + spirv_path + " ";
  for (int i = 2; i < argv.size(); ++i) {
    cmdLine += " " + std::string(argv[i]);
  }
  system(cmdLine.c_str());
  // load the compiled spir-v
  se::buffer spirv_code;
  se::syncReadFile(spirv_path.c_str(), spirv_code);
  return this->operator()(from_spirv_tag{}, &spirv_code, stage);
}

namespace slang_inline {
using Slang::ComPtr;

struct SlangSession {
  SlangSession(
    std::string const& filepath,
    std::vector<std::pair<char const*, char const*>> const& macros = {},
    bool use_glsl_intermediate = false);

  auto load(std::vector<std::pair<std::string, rhi::ShaderStageBit>> const&
    entrypoints) noexcept -> std::vector<ShaderHandle>;

  std::unordered_map<std::string, ShaderReflection::BindingInfo> bindingInfo;
 private:
  slang::SessionDesc sessionDesc = {};
  slang::TargetDesc targetDesc = {};
  slang::IModule* slangModule = nullptr;
  ComPtr<slang::ISession> session;
  bool use_glsl_intermediate;
  std::string filepath;
};

struct SlangManager {
  SINGLETON(SlangManager, {
    SlangResult result = slang::createGlobalSession(slangGlobalSession.writeRef());
    if (result != 0)
      root::print::error("GFX::SLANG::Global session create failed.");
  });
  auto getGlobalSession() noexcept -> slang::IGlobalSession* {
    return slangGlobalSession.get();
  }
  ComPtr<slang::IGlobalSession> slangGlobalSession;
};

inline void diagnoseIfNeeded(slang::IBlob* diagnosticsBlob) {
  if (diagnosticsBlob != nullptr) {
    root::print::error(std::string((const char*)diagnosticsBlob->getBufferPointer()));
  }
}

SlangSession::SlangSession(
    std::string const& filepath,
    std::vector<std::pair<char const*, char const*>> const& macros,
    bool use_glsl_intermediate)
    : use_glsl_intermediate(use_glsl_intermediate), filepath(filepath) {
  std::filesystem::path path(filepath);
  SlangManager* manager = Singleton<SlangManager>::instance();
  slang::IGlobalSession* globalSession = manager->getGlobalSession();
  if (use_glsl_intermediate) {
    // set target to spirv glsl460
    targetDesc.format = SLANG_GLSL;
    targetDesc.profile = globalSession->findProfile("glsl460");  
  } else {
    // set target to spirv glsl460
    targetDesc.format = SLANG_SPIRV;
    targetDesc.profile = globalSession->findProfile("glsl460");  
    //targetDesc.flags = SLANG_TARGET_FLAG_GENERATE_SPIRV_DIRECTLY;
  }
  sessionDesc.targets = &targetDesc;
  sessionDesc.targetCount = 1;
  // set search path
  std::string parent_path = path.parent_path().string();
  char const* search_path = parent_path.c_str();
  sessionDesc.searchPaths = &search_path;
  sessionDesc.searchPathCount = 1;
  // push pre-defined macros
  std::vector<slang::PreprocessorMacroDesc> macro_list;
  for (auto const& macro : macros)
    macro_list.emplace_back(
        slang::PreprocessorMacroDesc{macro.first, macro.second});
  sessionDesc.preprocessorMacros = macro_list.data();
  sessionDesc.preprocessorMacroCount = macro_list.size();
  // create slang session
  SlangResult result =
      globalSession->createSession(sessionDesc, session.writeRef());
  if (result != 0) {
    root::print::error("GFX::SLANG::Session create failed.");
    return;
  }
  // load module
  ComPtr<slang::IBlob> diagnosticBlob;
  std::string stem = path.stem().string();
  slangModule = session->loadModule(stem.c_str(),
                                    diagnosticBlob.writeRef());
  diagnoseIfNeeded(diagnosticBlob);
  if (!slangModule) return;

  slang::ShaderReflection* shaderReflection = slangModule->getLayout();

  unsigned parameterCount = shaderReflection->getParameterCount();
  for (unsigned pp = 0; pp < parameterCount; pp++) {
    ShaderReflection::BindingInfo bindinfo = {
        ShaderReflection::ResourceType::Undefined, 0, 0};
    slang::VariableLayoutReflection* parameter =
        shaderReflection->getParameterByIndex(pp);
    char const* parameterName = parameter->getName();
    slang::ParameterCategory category = parameter->getCategory();
    unsigned index = parameter->getBindingIndex();
    unsigned space = parameter->getBindingSpace();
    bindinfo.binding = index;
    bindinfo.set = space;
    slang::TypeLayoutReflection* typeLayout = parameter->getTypeLayout();
    slang::TypeReflection::Kind kind = typeLayout->getKind();
    switch (kind) {
      case slang::TypeReflection::Kind::None:
        break;
      case slang::TypeReflection::Kind::Struct:
        break;
      case slang::TypeReflection::Kind::Array:
        break;
      case slang::TypeReflection::Kind::Matrix:
        break;
      case slang::TypeReflection::Kind::Vector:
        break;
      case slang::TypeReflection::Kind::Scalar:
        break;
      case slang::TypeReflection::Kind::ConstantBuffer:
        break;
      case slang::TypeReflection::Kind::Resource:
        break;
      case slang::TypeReflection::Kind::SamplerState:
        break;
      case slang::TypeReflection::Kind::TextureBuffer:
        break;
      case slang::TypeReflection::Kind::ShaderStorageBuffer:
        break;
      case slang::TypeReflection::Kind::ParameterBlock:
        break;
      case slang::TypeReflection::Kind::GenericTypeParameter:
        break;
      case slang::TypeReflection::Kind::Interface:
        break;
      case slang::TypeReflection::Kind::OutputStream:
        break;
      case slang::TypeReflection::Kind::Specialized:
        break;
      case slang::TypeReflection::Kind::Feedback:
        break;
      default:
        break;
    }
    slang::BindingType type = typeLayout->getDescriptorSetDescriptorRangeType(0, 0);
    switch (type) {
      case slang::BindingType::PushConstant:
        bindinfo.type = ShaderReflection::ResourceType::Undefined;
        break;
      case slang::BindingType::Unknown:
        bindinfo.type = ShaderReflection::ResourceType::Undefined;
        break;
      case slang::BindingType::CombinedTextureSampler:
        bindinfo.type = ShaderReflection::ResourceType::SampledImages;
        break;
      case slang::BindingType::RayTracingAccelerationStructure:
        bindinfo.type = ShaderReflection::ResourceType::AccelerationStructure;
        break;
      case slang::BindingType::ConstantBuffer:
        bindinfo.type = ShaderReflection::ResourceType::UniformBuffer;
        break;
      case slang::BindingType::RawBuffer:
      case slang::BindingType::MutableRawBuffer:
        bindinfo.type = ShaderReflection::ResourceType::StorageBuffer;
        break;
      case slang::BindingType::MutableTexture:
        bindinfo.type = ShaderReflection::ResourceType::StorageImages;
        break;
      case slang::BindingType::Texture:
        bindinfo.type = ShaderReflection::ResourceType::ReadonlyImage;
        break;
      case slang::BindingType::Sampler:
        bindinfo.type = ShaderReflection::ResourceType::Sampler;
        break;
      case slang::BindingType::ParameterBlock:
      case slang::BindingType::TypedBuffer:
      case slang::BindingType::InputRenderTarget:
      case slang::BindingType::InlineUniformData:
      case slang::BindingType::VaryingInput:
      case slang::BindingType::VaryingOutput:
      case slang::BindingType::ExistentialValue:
      case slang::BindingType::MutableFlag:
      case slang::BindingType::MutableTypedBuffer:
      case slang::BindingType::BaseMask:
      case slang::BindingType::ExtMask:
      default: root::print::error("SLANG :: Binding not valid");
      break;
    }
    bindingInfo[std::string(parameterName)] = bindinfo;
    float a = 1.f;
    // ...
  }
}

auto SlangSession::load(
    std::vector<std::pair<std::string, rhi::ShaderStageBit>> const&
    entrypoints) noexcept -> std::vector<ShaderHandle> {
  std::vector<ShaderHandle> sms(entrypoints.size());
  // add all entrypoints
  std::vector<ComPtr<slang::IEntryPoint>> entryPointsPtrs(entrypoints.size());
  std::vector<slang::IComponentType*> componentTypes;
  componentTypes.push_back(slangModule);
  for (size_t i = 0; i < entrypoints.size(); ++i) {
    SlangInt32 w = slangModule->getDefinedEntryPointCount();
    SlangResult result = slangModule->findEntryPointByName(
        entrypoints[i].first.c_str(), entryPointsPtrs[i].writeRef());
    if (result != 0) {
      root::print::error("GFX::SLANG::cannot find entrypoint \"" + entrypoints[i].first + "\"");
      return sms;
    }
    componentTypes.push_back(entryPointsPtrs[i]);
  }
  // compile the session
  ComPtr<slang::IBlob> diagnosticBlob;
  ComPtr<slang::IComponentType> composedProgram;
  SlangResult result = session->createCompositeComponentType(
      componentTypes.data(), componentTypes.size(), composedProgram.writeRef(),
      diagnosticBlob.writeRef());
  diagnoseIfNeeded(diagnosticBlob);
  if (result != 0) {
    root::print::error("GFX::SLANG::createCompositeComponentType() failed.");
    return sms;
  }
  ComPtr<slang::IBlob> compiledCode;
  for (size_t i = 0; i < entrypoints.size(); ++i) {
    ComPtr<slang::IBlob> diagnosticsBlob;

    SlangResult result = composedProgram->getEntryPointCode(
        i, 0, compiledCode.writeRef(), diagnosticsBlob.writeRef());
    diagnoseIfNeeded(diagnosticsBlob);
    if (result != 0) {
      root::print::error("GFX::SLANG::getEntryPointCode() failed.");
      return sms;
    }
    se::buffer spirvcode;
    if (use_glsl_intermediate) {
      // compile SPIR-V from glsl
      se::buffer glslcode;
      glslcode.isReference = true;
      glslcode.data = (void*)(compiledCode->getBufferPointer());
      glslcode.size = compiledCode->getBufferSize();
      std::string suffix = "glsl";
      switch (entrypoints[i].second) {
        case rhi::ShaderStageBit::VERTEX:       suffix = "vert"; break;
        case rhi::ShaderStageBit::FRAGMENT:     suffix = "frag"; break;
        case rhi::ShaderStageBit::COMPUTE:      suffix = "comp"; break;
        case rhi::ShaderStageBit::GEOMETRY:     suffix = "geom"; break;
        case rhi::ShaderStageBit::RAYGEN:       suffix = "rgen"; break;
        case rhi::ShaderStageBit::MISS:         suffix = "rmiss"; break;
        case rhi::ShaderStageBit::CLOSEST_HIT:  suffix = "rchit"; break;
        case rhi::ShaderStageBit::INTERSECTION: suffix = "rint"; break;
        case rhi::ShaderStageBit::ANY_HIT:      suffix = "rahit"; break;
        case rhi::ShaderStageBit::CALLABLE:     suffix = "rcall"; break;
        case rhi::ShaderStageBit::TASK:         suffix = "task"; break;
        case rhi::ShaderStageBit::MESH:         suffix = "mesh"; break;
        default: break;
      }
      std::string glsl_path = filepath.substr(0, filepath.find_last_of('.')+1) + suffix;
      se::syncWriteFile(glsl_path.c_str(), glslcode);
      // create shader module
      sms[i] = GFXContext::load_shader_glsl(glsl_path.c_str(), entrypoints[i].second);
    } else {
      // directly use the compiled SPIR-V
      spirvcode.isReference = true;
      spirvcode.data = (void*)(compiledCode->getBufferPointer());
      spirvcode.size = compiledCode->getBufferSize();
      // create shader module
      sms[i] = GFXContext::load_shader_spirv(&spirvcode, entrypoints[i].second);
    }
  }
  for (size_t i = 0; i < entrypoints.size(); ++i) {
    sms[i]->reflection.bindingInfo = bindingInfo;
  }
  return sms;
}
}

//ShaderModuleLoader::result_type ShaderModuleLoader::operator()(
//  ShaderModuleLoader::from_slang_tag, 
//  se::buffer* buffer,
//  rhi::ShaderStageBit stage) {
//  
//}

auto Buffer::getName() const noexcept -> char const* {
  return buffer->getName().c_str();
}

bool ViewIndex::operator==(ViewIndex const& p) const {
  return type == p.type && mostDetailedMip == p.mostDetailedMip &&
    mipCount == p.mipCount && firstArraySlice == p.firstArraySlice &&
    arraySize == p.arraySize;
}

auto Texture::getName() const noexcept -> char const* {
    return texture->getName().c_str();
}

auto findDimension(rhi::TextureDimension dim, uint32_t arraySize) noexcept
    -> rhi::TextureViewDimension {
    rhi::TextureViewDimension dimension;
  switch (dim) {
    case rhi::TextureDimension::TEX1D:
      dimension = (arraySize > 1) ? rhi::TextureViewDimension::TEX1D_ARRAY
                                  : rhi::TextureViewDimension::TEX1D;
      break;
    case rhi::TextureDimension::TEX2D:
      dimension = (arraySize > 1) ? rhi::TextureViewDimension::TEX2D_ARRAY
                                  : rhi::TextureViewDimension::TEX2D;
      break;
    case rhi::TextureDimension::TEX3D:
      dimension = (arraySize > 1) ? rhi::TextureViewDimension::TEX3D_ARRAY
                                  : rhi::TextureViewDimension::TEX3D;
      break;
    default:
      break;
  }
  return dimension;
}

auto Texture::getUAV(uint32_t mipLevel, uint32_t firstArraySlice,
                     uint32_t arraySize) noexcept -> rhi::TextureView* {
  ViewIndex idx = {rhi::TextureViewType::UAV, mipLevel, 0, firstArraySlice,
                   arraySize};
  rhi::TextureViewDimension dimension =
      findDimension(texture->dimension(), arraySize);
  auto find = viewPool.find(idx);
  if (find == viewPool.end()) {
    viewPool[idx] = texture->createView(rhi::TextureViewDescriptor{
        texture->format(), dimension, (uint32_t)rhi::TextureAspectBit::COLOR_BIT,
        mipLevel, 1, firstArraySlice, arraySize});
    find = viewPool.find(idx);
  }
  return find->second.get();
}

auto Texture::getRTV(uint32_t mipLevel, uint32_t firstArraySlice,
                     uint32_t arraySize) noexcept -> rhi::TextureView* {
  ViewIndex idx = { rhi::TextureViewType::RTV, mipLevel, 0, firstArraySlice,
                   arraySize};
  rhi::TextureViewDimension dimension =
      findDimension(texture->dimension(), arraySize);
  auto find = viewPool.find(idx);
  if (find == viewPool.end()) {
    viewPool[idx] = texture->createView(rhi::TextureViewDescriptor{
        texture->format(), dimension, (uint32_t)rhi::TextureAspectBit::COLOR_BIT,
        mipLevel, 1, firstArraySlice, arraySize});
    find = viewPool.find(idx);
  }
  return find->second.get();
}

auto Texture::getDSV(uint32_t mipLevel, uint32_t firstArraySlice,
                     uint32_t arraySize) noexcept -> rhi::TextureView* {
  ViewIndex idx = { rhi::TextureViewType::DSV, mipLevel, 0, firstArraySlice,
                   arraySize};
  rhi::TextureViewDimension dimension =
      findDimension(texture->dimension(), arraySize);
  auto find = viewPool.find(idx);
  if (find == viewPool.end()) {
    viewPool[idx] = texture->createView(rhi::TextureViewDescriptor{
        texture->format(), dimension, (uint32_t)rhi::TextureAspectBit::DEPTH_BIT,
        mipLevel, 1, firstArraySlice, arraySize});
    find = viewPool.find(idx);
  }
  return find->second.get();
}

auto Texture::getSRV(uint32_t mostDetailedMip, uint32_t mipCount,
                     uint32_t firstArraySlice, uint32_t arraySize) noexcept
    -> rhi::TextureView* {
  ViewIndex idx = { rhi::TextureViewType::RTV, mostDetailedMip, mipCount,
                   firstArraySlice, arraySize};
  rhi::TextureViewDimension dimension =
      findDimension(texture->dimension(), arraySize);
  uint32_t aspect = (uint32_t)rhi::TextureAspectBit::COLOR_BIT;
  if (rhi::hasDepthBit(texture->format()))
    aspect = (uint32_t)rhi::TextureAspectBit::DEPTH_BIT;
  if (rhi::hasStencilBit(texture->format()))
    aspect |= (uint32_t)rhi::TextureAspectBit::STENCIL_BIT;

  auto find = viewPool.find(idx);
  if (find == viewPool.end()) {
    viewPool[idx] = texture->createView(rhi::TextureViewDescriptor{
        texture->format(), dimension, aspect, mostDetailedMip, mipCount,
        firstArraySlice, arraySize});
    find = viewPool.find(idx);
  }
  return find->second.get();
}

// Base class definition
// ===========================================================================
// GFX Context definition

rhi::Device* GFXContext::device = nullptr;
ex::resource_cache<ShaderModule, ShaderModuleLoader> GFXContext::shaders = {};

auto GFXContext::initialize(rhi::Device* device) noexcept -> void {
  GFXContext::device = device;
}

auto GFXContext::finalize() noexcept -> void {
  shaders.clear();
}

auto GFXContext::load_shader_spirv(se::buffer* buffer, rhi::ShaderStageBit stage) noexcept -> ShaderHandle {
  RUID const ruid = root::resource::queryRUID();
  auto ret = shaders.load(ruid, ShaderModuleLoader::from_spirv_tag{}, buffer, stage);
  return ret.first->second;
}

auto GFXContext::load_shader_glsl(const char* path, rhi::ShaderStageBit stage) noexcept -> ex::resource<se::gfx::ShaderModule> {
  RUID const ruid = root::resource::queryRUID();
  auto ret = shaders.load(ruid, ShaderModuleLoader::from_glsl_tag{}, std::string(path), stage);
  return ret.first->second;
}

auto GFXContext::load_shader_slang(
  std::string const& path,
  std::vector<std::pair<std::string, rhi::ShaderStageBit>> const& entrypoints,
  std::vector<std::pair<char const*, char const*>> const& macros,
  bool glsl_intermediate
) noexcept -> std::vector<ex::resource<se::gfx::ShaderModule>> {
  slang_inline::SlangSession session(path, macros, glsl_intermediate);
  return session.load(entrypoints);
}
}