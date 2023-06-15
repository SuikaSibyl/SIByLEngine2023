#include "SE.SRenderer-RTRSMPass.hpp"
#include "../RasterizerPasses/SE.SRenderer-ShadowMapPass.hpp"

namespace SIByL::SRP {
DirectRSMPass::DirectRSMPass(uint32_t width, uint32_t height,
                             RSMShareInfo* info)
    : width(width), height(height), info(info) {
  rsm_rgen = GFX::GFXManager::get()->registerShaderModuleResource(
      "../Engine/Binaries/Runtime/spirv/SRenderer/raytracer/rsm/"
      "directional_light_rsm_rgen.spv",
      {nullptr, RHI::ShaderStages::RAYGEN});

  GFX::SBTsDescriptor sbt = RTCommon::get()->getSBTDescriptor();
  sbt.rgenSBT = GFX::SBTsDescriptor::RayGenerationSBT{
      {Core::ResourceManager::get()->getResource<GFX::ShaderModule>(rsm_rgen)}};

  RayTracingPass::init(sbt, 3);
}

auto DirectRSMPass::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;

  reflector.addOutput("PixImportance")
      .isTexture()
      .withSize(Math::ivec3(width, height, 1))
      .withLevels(RDG::MaxPossible)
      .withFormat(RHI::TextureFormat::R32_FLOAT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(
          RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage(
                  (uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));

  reflector.addOutput("NormalCone")
      .isTexture()
      .withSize(Math::ivec3(width, height, 1))
      .withLevels(RDG::MaxPossible)
      .withFormat(RHI::TextureFormat::RGBA32_FLOAT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(
          RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage(
                  (uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));

  reflector.addOutput("AABBXY")
      .isTexture()
      .withSize(Math::ivec3(width, height, 1))
      .withLevels(RDG::MaxPossible)
      .withFormat(RHI::TextureFormat::RGBA32_FLOAT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(
          RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage(
                  (uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));

  reflector.addOutput("AABBZ")
      .isTexture()
      .withSize(Math::ivec3(width, height, 1))
      .withLevels(RDG::MaxPossible)
      .withFormat(RHI::TextureFormat::RG32_FLOAT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(
          RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage(
                  (uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));

  reflector.addInputOutput("WeightImg")
      .isTexture()
      .withSize(Math::ivec3(512, 512, 1))
      .withFormat(RHI::TextureFormat::R32_FLOAT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(
          RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage(
                  (uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));

  return reflector;
}

Math::vec3 CS2WS(const Math::vec2 uv, Math::mat4 const& inv_proj,
                 Math::mat4 const& inv_view) {
  const Math::vec4 posInCS = Math::vec4(uv * 2 - 1.0f, 0, 1);
  Math::vec4 posInVS = inv_proj * posInCS;
  posInVS /= posInVS.w;
  const Math::vec4 posInWS =
      inv_view * Math::vec4(posInVS.x, posInVS.y, posInVS.z, 1.0);
  return Math::vec3(posInWS.x, posInWS.y, posInWS.z);
}

auto DirectRSMPass::renderUI() noexcept -> void {
  ImGui::DragFloat("Scaling X", &scaling_x, 0.01);
  ImGui::DragFloat("Scaling Y", &scaling_y, 0.01);
  ImGui::Checkbox("Use Weight", &use_weight);
}

auto DirectRSMPass::execute(RDG::RenderContext* context,
                     RDG::RenderData const& renderData) noexcept
    -> void {
  GFX::Texture* pix_importance = renderData.getTexture("PixImportance");
  GFX::Texture* normal_cone = renderData.getTexture("NormalCone");
  GFX::Texture* aabb_xy = renderData.getTexture("AABBXY");
  GFX::Texture* aabb_z = renderData.getTexture("AABBZ");
  GFX::Texture* weight_img = renderData.getTexture("WeightImg");
  
  std::vector<RHI::BindGroupEntry>* set_0_entries =
      renderData.getBindGroupEntries("CommonScene");
  getBindGroup(context, 0)->updateBinding(*set_0_entries);
  std::vector<RHI::BindGroupEntry>* set_1_entries =
      renderData.getBindGroupEntries("CommonRT");
  getBindGroup(context, 1)->updateBinding(*set_1_entries);
  getBindGroup(context, 2)
      ->updateBinding(
          {RHI::BindGroupEntry{0, RHI::BindingResource{pix_importance->getUAV(0, 0, 1)}},
           RHI::BindGroupEntry{1, RHI::BindingResource{normal_cone->getUAV(0, 0, 1)}},
           RHI::BindGroupEntry{2, RHI::BindingResource{aabb_xy->getUAV(0, 0, 1)}},
           RHI::BindGroupEntry{3, RHI::BindingResource{aabb_z->getUAV(0, 0, 1)}},
           RHI::BindGroupEntry{4, RHI::BindingResource{weight_img->getUAV(0, 0, 1)}},
          });

  RHI::RayTracingPassEncoder* encoder = beginPass(context);

  uint32_t batchIdx = renderData.getUInt("AccumIdx");

  Math::mat4 view;
  Math::mat4 proj;
  Math::mat4 inv_view;
  Math::mat4 inv_proj;
  Math::vec3 direction;

  if (RACommon::get()->mainDirectionalLight.has_value()) {
    Math::mat4 transform = RACommon::get()->mainDirectionalLight->transform;
    direction = Math::Transform(transform) * Math::vec3(0, 0, 1);
    direction = Math::normalize(direction);
    Math::mat4 w2l =
        Math::lookAt(Math::vec3{0}, direction, Math::vec3{0, 1, 0}).m;
    proj = fitToScene(RACommon::get()->sceneAABB, w2l);
    view = w2l;

    proj.data[0][0] *= scaling_x;
    proj.data[1][1] *= scaling_y;

    inv_proj = Math::inverse(proj);
    inv_view = Math::inverse(view);

    Math::mat4 view_porj = proj * view;
    Math::vec2 xy = Math::vec2(1, 1) * 2 - 1.0f;
    const Math::vec4 posInCS =
        Math::vec4(xy.x,xy.y, 0.f, 1.f) *
        Math::vec4(1.f, 1.f, 1.f, 1.f);
    Math::vec4 posInVS = inv_proj * posInCS;
    float a = 1.f;
    posInVS /= posInVS.w;
    const Math::vec4 posInWS =
        inv_view * Math::vec4(posInVS.x, posInVS.y, posInVS.z, 1.0);
    
    const Math::vec3 wsPos_do = CS2WS(Math::vec2(0, 0), inv_proj, inv_view);
    const Math::vec3 wsPos_dx = CS2WS(Math::vec2(1, 0), inv_proj, inv_view);
    const Math::vec3 wsPos_dy = CS2WS(Math::vec2(0, 1), inv_proj, inv_view);
    const float area = length(cross(wsPos_dx - wsPos_do, wsPos_dy - wsPos_do));
    info->area = area;
  }

  inv_proj = Math::transpose(inv_proj);
  inv_view = Math::transpose(inv_view);

  info->inv_proj = inv_proj;
  info->inv_view = inv_view;
  info->proj_view = Math::transpose(proj * view);
  info->direction = direction;

  PushConstant pConst = {inv_view, inv_proj, direction, use_weight ? 1 : 0,
                         width,    height,   batchIdx};
  encoder->pushConstants(&pConst, (uint32_t)RHI::ShaderStages::RAYGEN, 0,
                         sizeof(PushConstant));
  encoder->traceRays(width, height, 1);

  encoder->end();
}
}