#include "SE.SRenderer-MIPSSLCPass.hpp"

namespace SIByL {
MIPSSLCInitSubPass::MIPSSLCInitSubPass(size_t size) : size(size) {
  frag = GFX::GFXManager::get()->registerShaderModuleResource(
      "../Engine/Binaries/Runtime/spirv/SRenderer/rasterizer/fullscreen_pass/"
      "sslt_mip_init_frag.spv",
      {nullptr, RHI::ShaderStages::FRAGMENT});
  RDG::FullScreenPass::init(
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(frag));
}

auto MIPSSLCInitSubPass::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;

  reflector.addInput("Color")
      .isTexture()
      .withUsages((uint32_t)RHI::TextureUsage::TEXTURE_BINDING)
      .consume(
          RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::TextureBinding}
              .setSubresource(0, 1, 0, 1)
              .addStage((uint32_t)RHI::PipelineStages::FRAGMENT_SHADER_BIT));

  reflector.addInput("Normal")
      .isTexture()
      .withUsages((uint32_t)RHI::TextureUsage::TEXTURE_BINDING)
      .consume(
          RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::TextureBinding}
              .setSubresource(0, 1, 0, 1)
              .addStage((uint32_t)RHI::PipelineStages::FRAGMENT_SHADER_BIT));

  reflector.addInput("Depth")
      .isTexture()
      .withUsages((uint32_t)RHI::TextureUsage::TEXTURE_BINDING)
      .consume(
          RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::TextureBinding}
              .setSubresource(0, 1, 0, 1)
              .addStage((uint32_t)RHI::PipelineStages::FRAGMENT_SHADER_BIT));

  reflector.addOutput("ImportanceMIP")
      .isTexture()
      .withSize(Math::ivec3{size, size, 1})
      .withLevels(RDG::MaxPossible)
      .withUsages((uint32_t)RHI::TextureUsage::COLOR_ATTACHMENT)
      .withFormat(RHI::TextureFormat::R32_FLOAT)
      .consume(
          RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::ColorAttachment}
              .setSubresource(0, 1, 0, 1)
              .setAttachmentLoc(0)
              .addStage(
                  (uint32_t)RHI::PipelineStages::COLOR_ATTACHMENT_OUTPUT_BIT));

  reflector.addOutput("BoundingBoxMIP")
      .isTexture()
      .withSize(Math::ivec3{size, size, 1})
      .withLevels(RDG::MaxPossible)
      .withUsages((uint32_t)RHI::TextureUsage::COLOR_ATTACHMENT)
      .withFormat(RHI::TextureFormat::RGBA32_FLOAT)
      .consume(
          RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::ColorAttachment}
              .setSubresource(0, 1, 0, 1)
              .setAttachmentLoc(1)
              .addStage(
                  (uint32_t)RHI::PipelineStages::COLOR_ATTACHMENT_OUTPUT_BIT));

  reflector.addOutput("BBNCPackMIP")
      .isTexture()
      .withSize(Math::ivec3{size, size, 1})
      .withLevels(RDG::MaxPossible)
      .withUsages((uint32_t)RHI::TextureUsage::COLOR_ATTACHMENT)
      .withFormat(RHI::TextureFormat::RGBA32_FLOAT)
      .consume(
          RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::ColorAttachment}
              .setSubresource(0, 1, 0, 1)
              .setAttachmentLoc(2)
              .addStage(
                  (uint32_t)RHI::PipelineStages::COLOR_ATTACHMENT_OUTPUT_BIT));

  reflector.addOutput("NormalConeMIP")
      .isTexture()
      .withSize(Math::ivec3{size, size, 1})
      .withLevels(RDG::MaxPossible)
      .withUsages((uint32_t)RHI::TextureUsage::COLOR_ATTACHMENT)
      .withFormat(RHI::TextureFormat::RGB10A2_UNORM)
      .consume(
          RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::ColorAttachment}
              .setSubresource(0, 1, 0, 1)
              .setAttachmentLoc(3)
              .addStage(
                  (uint32_t)RHI::PipelineStages::COLOR_ATTACHMENT_OUTPUT_BIT));

  return reflector;
}

auto MIPSSLCInitSubPass::renderUI() noexcept -> void {
  {  // Select an item type
    const char* item_names[] = {"Luminance", "Average", "Maximum", "Uniform"};
    int importance_operator = pConst.importance_operator;
    ImGui::Combo("Mode", &importance_operator, item_names,
                 IM_ARRAYSIZE(item_names), IM_ARRAYSIZE(item_names));
    pConst.importance_operator = importance_operator;
  }
  {
    int modulateJacobian = pConst.modulateJacobian;
    ImGui::DragInt("Modulate Jacobian", &modulateJacobian, 1, 0, 1);
    pConst.modulateJacobian = modulateJacobian;
  }
}

auto MIPSSLCInitSubPass::execute(
    RDG::RenderContext* context,
                     RDG::RenderData const& renderData) noexcept -> void {
  GFX::Texture* color = renderData.getTexture("Color");
  GFX::Texture* normal = renderData.getTexture("Normal");
  GFX::Texture* depth = renderData.getTexture("Depth");

  GFX::Texture* importance_mip = renderData.getTexture("ImportanceMIP");
  GFX::Texture* boundingbox_mip = renderData.getTexture("BoundingBoxMIP");
  GFX::Texture* bbncpack_mip = renderData.getTexture("BBNCPackMIP");
  GFX::Texture* normalcone_mip = renderData.getTexture("NormalConeMIP");

  renderPassDescriptor = {
      {RHI::RenderPassColorAttachment{importance_mip->getRTV(0, 0, 1),
                                      nullptr,
                                      {0, 0, 0, 1},
                                      RHI::LoadOp::CLEAR,
                                      RHI::StoreOp::STORE},
       RHI::RenderPassColorAttachment{boundingbox_mip->getRTV(0, 0, 1),
                                      nullptr,
                                      {0, 0, 0, 1},
                                      RHI::LoadOp::CLEAR,
                                      RHI::StoreOp::STORE},
       RHI::RenderPassColorAttachment{bbncpack_mip->getRTV(0, 0, 1),
                                      nullptr,
                                      {0, 0, 0, 1},
                                      RHI::LoadOp::CLEAR,
                                      RHI::StoreOp::STORE},
       RHI::RenderPassColorAttachment{normalcone_mip->getRTV(0, 0, 1),
                                      nullptr,
                                      {0, 0, 0, 1},
                                      RHI::LoadOp::CLEAR,
                                      RHI::StoreOp::STORE}},
      RHI::RenderPassDepthStencilAttachment{},
  };

  std::vector<RHI::BindGroupEntry>* set_0_entries =
      renderData.getBindGroupEntries("CommonScene");
  auto entry = (*set_0_entries)[0];
  entry.binding = 3;

  auto* defaul_sampler =
      Core::ResourceManager::get()
          ->getResource<GFX::Sampler>(
              GFX::GFXManager::get()->commonSampler.clamp_nearest)
          ->sampler.get();
  getBindGroup(context, 0)
      ->updateBinding(std::vector<RHI::BindGroupEntry>{
          {0, RHI::BindingResource(color->getSRV(0, 1, 0, 1), defaul_sampler)},
          {1, RHI::BindingResource(depth->getSRV(0, 1, 0, 1), defaul_sampler)},
          {2, RHI::BindingResource(normal->getSRV(0, 1, 0, 1), defaul_sampler)},
          entry});

  SRenderer::CameraData* cd =
      reinterpret_cast<SRenderer::CameraData*>(renderData.getPtr("CameraData"));
  pConst.inv_proj = Math::inverse((cd->projMat));
  pConst.trans_inv_view = Math::transpose(Math::inverse(cd->viewMat));
  pConst.resolution = {(int)color->texture->width(),
                       (int)color->texture->height()};
  RHI::RenderPassEncoder* encoder = beginPass(context, size, size);
  encoder->pushConstants(&pConst, (uint32_t)RHI::ShaderStages::FRAGMENT, 0,
                         sizeof(PushConstant));

  dispatchFullScreen(context);

  encoder->end();
}

 MIPSSLCSubPass::MIPSSLCSubPass(size_t mip_offset) : mipOffset(mip_offset) {
    frag = GFX::GFXManager::get()->registerShaderModuleResource(
        "../Engine/Binaries/Runtime/spirv/SRenderer/rasterizer/fullscreen_pass/"
        "sslt_mip_frag.spv",
        {nullptr, RHI::ShaderStages::FRAGMENT});
    RDG::FullScreenPass::init(
        Core::ResourceManager::get()->getResource<GFX::ShaderModule>(frag));
  }

auto MIPSSLCSubPass::reflect() noexcept -> RDG::PassReflection {
    RDG::PassReflection reflector;

    reflector.addInputOutput("ImportanceMIP")
        .isTexture()
        .withFormat(RHI::TextureFormat::R32_FLOAT)
        .withUsages((uint32_t)RHI::TextureUsage::COLOR_ATTACHMENT |
                    (uint32_t)RHI::TextureUsage::TEXTURE_BINDING)
        .consume(
            RDG::TextureInfo::ConsumeEntry{
                RDG::TextureInfo::ConsumeType::TextureBinding}
                .setSubresource(mipOffset, mipOffset + 1, 0, 1)
                .addStage((uint32_t)RHI::PipelineStages::FRAGMENT_SHADER_BIT))
        .consume(
            RDG::TextureInfo::ConsumeEntry{
                RDG::TextureInfo::ConsumeType::ColorAttachment}
                .setSubresource(mipOffset + 1, mipOffset + 2, 0, 1)
                .setAttachmentLoc(0)
                .addStage((
                    uint32_t)RHI::PipelineStages::COLOR_ATTACHMENT_OUTPUT_BIT));

    reflector.addInputOutput("BoundingBoxMIP")
        .isTexture()
        .withUsages((uint32_t)RHI::TextureUsage::COLOR_ATTACHMENT |
                    (uint32_t)RHI::TextureUsage::TEXTURE_BINDING)
        .consume(
            RDG::TextureInfo::ConsumeEntry{
                RDG::TextureInfo::ConsumeType::TextureBinding}
                .setSubresource(mipOffset, mipOffset + 1, 0, 1)
                .addStage((uint32_t)RHI::PipelineStages::FRAGMENT_SHADER_BIT))
        .consume(
            RDG::TextureInfo::ConsumeEntry{
                RDG::TextureInfo::ConsumeType::ColorAttachment}
                .setSubresource(mipOffset + 1, mipOffset + 2, 0, 1)
                .setAttachmentLoc(1)
                .addStage((
                    uint32_t)RHI::PipelineStages::COLOR_ATTACHMENT_OUTPUT_BIT));

    reflector.addInputOutput("BBNCPackMIP")
        .isTexture()
        .withUsages((uint32_t)RHI::TextureUsage::COLOR_ATTACHMENT |
                    (uint32_t)RHI::TextureUsage::TEXTURE_BINDING)
        .consume(
            RDG::TextureInfo::ConsumeEntry{
                RDG::TextureInfo::ConsumeType::TextureBinding}
                .setSubresource(mipOffset, mipOffset + 1, 0, 1)
                .addStage((uint32_t)RHI::PipelineStages::FRAGMENT_SHADER_BIT))
        .consume(
            RDG::TextureInfo::ConsumeEntry{
                RDG::TextureInfo::ConsumeType::ColorAttachment}
                .setSubresource(mipOffset + 1, mipOffset + 2, 0, 1)
                .setAttachmentLoc(2)
                .addStage((
                    uint32_t)RHI::PipelineStages::COLOR_ATTACHMENT_OUTPUT_BIT));

    reflector.addInputOutput("NormalConeMIP")
        .isTexture()
        .withUsages((uint32_t)RHI::TextureUsage::COLOR_ATTACHMENT |
                    (uint32_t)RHI::TextureUsage::TEXTURE_BINDING)
        .consume(
            RDG::TextureInfo::ConsumeEntry{
                RDG::TextureInfo::ConsumeType::TextureBinding}
                .setSubresource(mipOffset, mipOffset + 1, 0, 1)
                .addStage((uint32_t)RHI::PipelineStages::FRAGMENT_SHADER_BIT))
        .consume(
            RDG::TextureInfo::ConsumeEntry{
                RDG::TextureInfo::ConsumeType::ColorAttachment}
                .setSubresource(mipOffset + 1, mipOffset + 2, 0, 1)
                .setAttachmentLoc(3)
                .addStage((
                    uint32_t)RHI::PipelineStages::COLOR_ATTACHMENT_OUTPUT_BIT));

    return reflector;
 }

auto MIPSSLCSubPass::execute(
     RDG::RenderContext* context,
                      RDG::RenderData const& renderData) noexcept -> void {
    GFX::Texture* importance_mip = renderData.getTexture("ImportanceMIP");
    GFX::Texture* boundingbox_mip = renderData.getTexture("BoundingBoxMIP");
    GFX::Texture* bbncpack_mip = renderData.getTexture("BBNCPackMIP");
    GFX::Texture* normalcone_mip = renderData.getTexture("NormalConeMIP");

    renderPassDescriptor = {
        {RHI::RenderPassColorAttachment{
             importance_mip->getRTV(mipOffset + 1, 0, 1),
             nullptr,
             {0, 0, 0, 1},
             RHI::LoadOp::CLEAR,
             RHI::StoreOp::STORE},
         RHI::RenderPassColorAttachment{
             boundingbox_mip->getRTV(mipOffset + 1, 0, 1),
             nullptr,
             {0, 0, 0, 1},
             RHI::LoadOp::CLEAR,
             RHI::StoreOp::STORE},
         RHI::RenderPassColorAttachment{
             bbncpack_mip->getRTV(mipOffset + 1, 0, 1),
             nullptr,
             {0, 0, 0, 1},
             RHI::LoadOp::CLEAR,
             RHI::StoreOp::STORE},
         RHI::RenderPassColorAttachment{
             normalcone_mip->getRTV(mipOffset + 1, 0, 1),
             nullptr,
             {0, 0, 0, 1},
             RHI::LoadOp::CLEAR,
             RHI::StoreOp::STORE}},
        RHI::RenderPassDepthStencilAttachment{},
    };

    auto* defaul_sampler =
        Core::ResourceManager::get()
            ->getResource<GFX::Sampler>(
                GFX::GFXManager::get()->commonSampler.defaultSampler)
            ->sampler.get();
    getBindGroup(context, 0)
        ->updateBinding(std::vector<RHI::BindGroupEntry>{
            {0, RHI::BindingResource(importance_mip->getSRV(mipOffset, 1, 0, 1),
                                     defaul_sampler)},
            {1,
             RHI::BindingResource(boundingbox_mip->getSRV(mipOffset, 1, 0, 1),
                                  defaul_sampler)},
            {2, RHI::BindingResource(bbncpack_mip->getSRV(mipOffset, 1, 0, 1),
                                     defaul_sampler)},
            {3, RHI::BindingResource(normalcone_mip->getSRV(mipOffset, 1, 0, 1),
                                     defaul_sampler)},
        });

    int32_t src_width = importance_mip->getRTV(mipOffset, 0, 1)->getWidth();
    int32_t src_height = importance_mip->getRTV(mipOffset, 0, 1)->getHeight();

    struct PushConstant {
    int32_t src_width;
    int32_t src_height;
    int32_t dst_width;
    int32_t dst_height;
    } ps = {src_width, src_height, std::max(src_width >> 1, 1),
            std::max(src_height >> 1, 1)};

    RHI::RenderPassEncoder* encoder =
        beginPass(context, ps.dst_width, ps.dst_height);
    encoder->pushConstants(&ps, (uint32_t)RHI::ShaderStages::FRAGMENT, 0,
                           sizeof(PushConstant));

    dispatchFullScreen(context);

    encoder->end();
 }

  MIPSSLCPass::MIPSSLCPass(size_t width, size_t height)
     : width(width), height(height) {
    mipCount = std::log2(std::max(width, height)) + 1;
 }

   auto MIPSSLCPass::alias() noexcept -> RDG::AliasDict {
    RDG::AliasDict dict;
    dict.addAlias("Color", CONCAT("Init"), "Color");
    dict.addAlias("Normal", CONCAT("Init"), "Normal");
    dict.addAlias("Depth", CONCAT("Init"), "Depth");

    dict.addAlias("ImportanceMIP",
                  CONCAT("MinPool Pass " + std::to_string(mipCount - 2)),
                  "ImportanceMIP");
    dict.addAlias("BoundingBoxMIP",
                  CONCAT("MinPool Pass " + std::to_string(mipCount - 2)),
                  "BoundingBoxMIP");
    dict.addAlias("BBNCPackMIP",
                  CONCAT("MinPool Pass " + std::to_string(mipCount - 2)),
                  "BBNCPackMIP");
    dict.addAlias("NormalConeMIP",
                  CONCAT("MinPool Pass " + std::to_string(mipCount - 2)),
                  "NormalConeMIP");
    return dict;
  }

  auto MIPSSLCPass::onRegister(RDG::Graph* graph) noexcept -> void {
    graph->addPass(std::make_unique<MIPSSLCInitSubPass>(width), CONCAT("Init"));
    graph->addPass(std::make_unique<MIPSSLCSubPass>(0),
                   CONCAT("MinPool Pass " + std::to_string(0)));

    graph->addEdge(CONCAT("Init"), "ImportanceMIP", CONCAT("MinPool Pass 0"),
                   "ImportanceMIP");
    graph->addEdge(CONCAT("Init"), "BoundingBoxMIP", CONCAT("MinPool Pass 0"),
                   "BoundingBoxMIP");
    graph->addEdge(CONCAT("Init"), "BBNCPackMIP", CONCAT("MinPool Pass 0"),
                   "BBNCPackMIP");
    graph->addEdge(CONCAT("Init"), "NormalConeMIP", CONCAT("MinPool Pass 0"),
                   "NormalConeMIP");

    for (size_t i = 1; i < mipCount - 1; ++i) {
    graph->addPass(std::make_unique<MIPSSLCSubPass>(i),
                   CONCAT("MinPool Pass " + std::to_string(i)));
    graph->addEdge(CONCAT("MinPool Pass " + std::to_string(i - 1)),
                   "ImportanceMIP", CONCAT("MinPool Pass " + std::to_string(i)),
                   "ImportanceMIP");
    graph->addEdge(
        CONCAT("MinPool Pass " + std::to_string(i - 1)), "BoundingBoxMIP",
        CONCAT("MinPool Pass " + std::to_string(i)), "BoundingBoxMIP");
    graph->addEdge(CONCAT("MinPool Pass " + std::to_string(i - 1)),
                   "BBNCPackMIP", CONCAT("MinPool Pass " + std::to_string(i)),
                   "BBNCPackMIP");
    graph->addEdge(CONCAT("MinPool Pass " + std::to_string(i - 1)),
                   "NormalConeMIP", CONCAT("MinPool Pass " + std::to_string(i)),
                   "NormalConeMIP");
    }
  }

   MISCompensationDiffPass::MISCompensationDiffPass() {
    comp = GFX::GFXManager::get()->registerShaderModuleResource(
        "../Engine/Binaries/Runtime/spirv/SRenderer/rasterizer/fullscreen_pass/"
        "misc_diff_comp.spv",
        {nullptr, RHI::ShaderStages::COMPUTE});
    RDG::ComputePass::init(
        Core::ResourceManager::get()->getResource<GFX::ShaderModule>(comp));
  }

   auto MISCompensationDiffPass::reflect() noexcept
      -> RDG::PassReflection {
    RDG::PassReflection reflector;

    reflector.addInputOutput("R32")
        .isTexture()
        .withFormat(RHI::TextureFormat::R32_FLOAT)
        .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
        .consume(
            RDG::TextureInfo::ConsumeEntry{
                RDG::TextureInfo::ConsumeType::StorageBinding}
                .setSubresource(9, 10, 0, 1)
                .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT))
        .consume(
            RDG::TextureInfo::ConsumeEntry{
                RDG::TextureInfo::ConsumeType::StorageBinding}
                .setSubresource(0, 1, 0, 1)
                .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));

    return reflector;
  }

   auto MISCompensationDiffPass::renderUI() noexcept -> void {
    {
    float weight = pConst.diffWeight;
    ImGui::DragFloat("Diff Weight", &weight, 0.01);
    pConst.diffWeight = weight;
    }
   }

   auto MISCompensationDiffPass::execute(
       RDG::RenderContext* context,
                        RDG::RenderData const& renderData) noexcept -> void {
    GFX::Texture* input = renderData.getTexture("R32");

    getBindGroup(context, 0)
        ->updateBinding(std::vector<RHI::BindGroupEntry>{
            RHI::BindGroupEntry{0,
                                RHI::BindingResource{input->getUAV(0, 0, 1)}},
            RHI::BindGroupEntry{1,
                                RHI::BindingResource{input->getUAV(9, 0, 1)}}});

    RHI::ComputePassEncoder* encoder = beginPass(context);

    uint32_t width = input->texture->width();
    uint32_t height = input->texture->height();

    prepareDispatch(context);

    pConst.resolution = Math::uvec2{width, height};
    encoder->pushConstants(&pConst, (uint32_t)RHI::ShaderStages::COMPUTE, 0,
                           sizeof(PushConstant));
    encoder->dispatchWorkgroups((width + 15) / 16, (height + 15) / 16, 1);

    encoder->end();
   }
  }  // namespace SIByL