#include "SE.SRendere-SSRXPipeline.hpp"

namespace SIByL::SRP {

SSRXForwardGraph::SSRXForwardGraph() {
  addSubgraph(std::make_unique<PreZPass>(), "Pre-Z Pass");
  addPass(std::make_unique<GBufferPass>(), "GBuffer Pass");
  addSubgraph(std::make_unique<CascadeShadowmapPass>(), "Shadowmap Pass");
  addPass(std::make_unique<ForwardPass>(), "Forward Pass");

  addSubgraph(std::make_unique<MIPMinPoolingPass>(512, 512), "HiZ-Gen Pass");
  addSubgraph(std::make_unique<MIPSSLCPass>(512, 512), "MIPSSLC Pass");
  addPass(std::make_unique<MISCompensationDiffPass>(), "MISC Pass");
  addSubgraph(std::make_unique<MIPAddPoolingPass>(512), "AddPooling Pass");

  addPass(std::make_unique<BarLCPass>(), "BarLC Pass");
  addPass(std::make_unique<AccumulatePass>(), "Accumulate Pass");

  addEdge("Pre-Z Pass", "Depth", "Forward Pass", "Depth");
  addEdge("Shadowmap Pass", "Depth", "Forward Pass", "Shadowmap");

  addEdge("Pre-Z Pass", "Depth", "HiZ-Gen Pass", "Input");
  addEdge("Pre-Z Pass", "Depth", "GBuffer Pass", "Depth");

  addEdge("Forward Pass", "Color", "MIPSSLC Pass", "Color");
  addEdge("GBuffer Pass", "wNormal", "MIPSSLC Pass", "Normal");
  addEdge("Pre-Z Pass", "Depth", "MIPSSLC Pass", "Depth");

  addEdge("Forward Pass", "Color", "BarLC Pass", "DI");
  addEdge("GBuffer Pass", "Color", "BarLC Pass", "BaseColor");
  addEdge("HiZ-Gen Pass", "Output", "BarLC Pass", "HiZ");
  addEdge("GBuffer Pass", "wNormal", "BarLC Pass", "NormalWS");

  addEdge("MIPSSLC Pass", "ImportanceMIP", "MISC Pass", "R32");
  addEdge("MISC Pass", "R32", "AddPooling Pass", "Input");

  addEdge("AddPooling Pass", "Output", "BarLC Pass", "ImportanceMIP");
  addEdge("MIPSSLC Pass", "BoundingBoxMIP", "BarLC Pass", "BoundingBoxMIP");
  addEdge("MIPSSLC Pass", "BBNCPackMIP", "BarLC Pass", "BBNCPackMIP");
  addEdge("MIPSSLC Pass", "NormalConeMIP", "BarLC Pass", "NormalConeMIP");

  addEdge("BarLC Pass", "Combined", "Accumulate Pass", "Input");

  markOutput("Accumulate Pass", "Output");
}

auto SSRXPipeline::build() noexcept -> void {
  forwardGraph.build();
  GFX::Texture* depth = forwardGraph.getTextureResource("Pre-Z Pass", "Depth");
  geoReconstrGraph.setExternal("Reconstr Pass", "Depth", depth);
  geoReconstrGraph.build();

  geoRenderGraph.setExternal(
      "SSRXGT Pass", "DI",
      forwardGraph.getTextureResource("BarLC Pass", "DI"));
  geoRenderGraph.setExternal(
      "SSRXGT Pass", "BaseColor",
      forwardGraph.getTextureResource("BarLC Pass", "BaseColor"));
  geoRenderGraph.setExternal(
      "SSRXGT Pass", "HiZ",
      forwardGraph.getTextureResource("BarLC Pass", "HiZ"));
  geoRenderGraph.setExternal(
      "SSRXGT Pass", "NormalWS",
      forwardGraph.getTextureResource("BarLC Pass", "NormalWS"));
  geoRenderGraph.setExternal(
      "SSRXGT Pass", "ImportanceMIP",
      forwardGraph.getTextureResource("BarLC Pass", "ImportanceMIP"));
  geoRenderGraph.setExternal(
      "SSRXGT Pass", "BoundingBoxMIP",
      forwardGraph.getTextureResource("BarLC Pass", "BoundingBoxMIP"));
  geoRenderGraph.setExternal(
      "SSRXGT Pass", "BBNCPackMIP",
      forwardGraph.getTextureResource("BarLC Pass", "BBNCPackMIP"));
  geoRenderGraph.setExternal(
      "SSRXGT Pass", "NormalConeMIP",
      forwardGraph.getTextureResource("BarLC Pass", "NormalConeMIP"));
  geoRenderGraph.build();
}

auto SSRXPipeline::renderUI() noexcept -> void {
    // Select an state type
    const char* item_names[] = {"Forward", "Geometry Reconstr",
                                "Geometry Render"};
    int debug_mode = static_cast<int>(state);
    ImGui::Combo("Mode", &debug_mode, item_names, IM_ARRAYSIZE(item_names),
                 IM_ARRAYSIZE(item_names));
    state = static_cast<State>(debug_mode);

    if (ImGui::Button("Capture Geometry")) {
        
    }
}

std::unique_ptr<RHI::BLAS> createBLAS(GFX::Buffer* vb, GFX::Buffer* ib) {
    RHI::Device* device = GFX::GFXManager::get()->rhiLayer->getDevice();
    return device->createBLAS(RHI::BLASDescriptor{
        std::vector<RHI::BLASTriangleGeometry>{RHI::BLASTriangleGeometry{
            vb->buffer.get(),
            ib->buffer.get(),
            nullptr,
            RHI::IndexFormat::UINT32_T,
            uint32_t(vb->buffer->size() / sizeof(uint32_t)),
            0,
            uint32_t(ib->buffer->size() / (3*sizeof(uint32_t))),
            0,
            RHI::AffineTransformMatrix{}
        }}});
}

std::unique_ptr<RHI::TLAS> createTLAS(RHI::BLAS* blas) {
    RHI::Device* device = GFX::GFXManager::get()->rhiLayer->getDevice();
    return device->createTLAS(RHI::TLASDescriptor{
        std::vector<RHI::BLASInstance>{RHI::BLASInstance{blas, Math::mat4{}}}});
}

auto SSRXPipeline::execute(RHI::CommandEncoder* encoder) noexcept -> void {
    if (to_build) {
      RHI::Device* device = GFX::GFXManager::get()->rhiLayer->getDevice();
      device->waitIdle();
      if (blas) blas_back = std::move(blas);
      if (tlas) tlas_back = std::move(tlas);
      blas = createBLAS(vertex_buffer, index_buffer);
      tlas = createTLAS(blas.get());
      device->waitIdle();
      to_build = false;  
      geoRenderGraph.gtPass->vb = vertex_buffer;
      geoRenderGraph.gtPass->ib = index_buffer;
      geoRenderGraph.gtPass->tlas = tlas.get();
    }
    switch (state) {
      case SIByL::SRP::SSRXPipeline::State::Forward:
        forwardGraph.execute(encoder);
        return;
      case SIByL::SRP::SSRXPipeline::State::GeoReconstr: {
        {
          forwardGraph.execute(encoder);
          geoReconstrGraph.execute(encoder);
          vertex_buffer = geoReconstrGraph.getBufferResource("Reconstr Pass", "VertexBuffer");
          index_buffer = geoReconstrGraph.getBufferResource("Reconstr Pass", "IndicesBuffer");
          to_build = true;
          state = SIByL::SRP::SSRXPipeline::State::GeoRender;
          }
        return;
      }
        break;
      case SIByL::SRP::SSRXPipeline::State::GeoRender:
        geoRenderGraph.execute(encoder);
        return;
      default:
        break;
    }
  return forwardGraph.execute(encoder);
}

auto SSRXPipeline::getActiveGraphs() noexcept -> std::vector<RDG::Graph*> {
  switch (state) {
    case State::Forward: return {&forwardGraph};
    case State::GeoReconstr: return {&forwardGraph, &geoReconstrGraph};
    case State::GeoRender: return {&geoRenderGraph};
    default:
      break;
  }
}

auto SSRXPipeline::getOutput() noexcept -> GFX::Texture* {
  switch (state) {
    case State::Forward:
      return forwardGraph.getOutput();
    case State::GeoReconstr:
      return forwardGraph.getOutput();
    case State::GeoRender:
      return geoRenderGraph.getOutput();
    default:
      break;
  }
}
}