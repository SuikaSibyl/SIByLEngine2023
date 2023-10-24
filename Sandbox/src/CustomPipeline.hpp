#pragma once

#include <SE.Addon.GBuffer.hpp>
#include <Passes/RasterizerPasses/SE.SRenderer-PreZPass.hpp>
#include <Passes/FullScreenPasses/SE.SRenderer-AccumulatePass.hpp>
#include <Passes/RasterizerPasses/SE.SRenderer-GeometryInspectorPass.hpp>
#include <Passes/FullScreenPasses/SE.SRenderer-Blit.hpp>
#include <SE.Addon.SemiNee.hpp>
#include <SE.Addon.BitonicSort.hpp>
#include <SE.Addon.VXGI.hpp>
#include <SE.Addon.VXGuiding.hpp>
#include <SE.Addon.SLC.hpp>
#include <SE.Addon.VBuffer.hpp>
#include <SE.Addon.gSLICr.hpp>
#include <SE.Addon.SSGuiding.hpp>
#include <SE.Addon.ASVGF.hpp>
#include <SE.Addon.RestirGI.hpp>
#include <SE.Addon.Postprocess.hpp>
#include <SE.Addon.Differentiable.hpp>

namespace SIByL {
SE_EXPORT struct CustomGraph : public RDG::Graph {
  CustomGraph() {
    // Create VBuffer + GBuffer, as well as history GBuffer, A-SVGF buffers
    addPass(std::make_unique<Addon::VBuffer::RayTraceVBuffer>(), "VBuffer Pass");
    addPass(std::make_unique<Addon::VBuffer::VBuffer2GBufferPass>(), "VBuffer2GBuffer Pass");
    addEdge("VBuffer Pass", "VBuffer", "VBuffer2GBuffer Pass", "VBuffer");
    addPass(std::make_unique<Addon::GBufferHolderSource>(), "GBufferPrev Pass");
    addPass(std::make_unique<Addon::ASVGF::Prelude>(), "ASVGF-Prelude Pass");

    {
      //// Sanity check: Consume current & previous GBuffer
      //addPass(std::make_unique<Addon::GBufferTemporalInspectorPass>(), "GBuffer Inspect Pass"); 
      //Addon::GBufferUtils::addGBufferEdges(this, "VBuffer2GBuffer Pass", "GBuffer Inspect Pass");
      //Addon::GBufferUtils::addPrevGBufferEdges(this, "GBufferPrev Pass", "GBuffer Inspect Pass");
      
      // ASVGF :: Gradient Reprojection
      addPass(std::make_unique<Addon::ASVGF::GradientReprojection>(), "ASVGF-GradProj Pass");
      addEdge("VBuffer Pass", "VBuffer", "ASVGF-GradProj Pass", "VBuffer");
      addEdge("ASVGF-Prelude Pass", "GradSamplePosPrev", "ASVGF-GradProj Pass", "GradSamplePosPrev");
      addEdge("ASVGF-Prelude Pass", "HFPrev", "ASVGF-GradProj Pass", "HFPrev");
      addEdge("ASVGF-Prelude Pass", "SpecPrev", "ASVGF-GradProj Pass", "SpecPrev");
      addEdge("ASVGF-Prelude Pass", "VBufferPrev", "ASVGF-GradProj Pass", "VBufferPrev");
      addEdge("ASVGF-Prelude Pass", "RandPrev", "ASVGF-GradProj Pass", "RandPrev");
      Addon::GBufferUtils::addGBufferEdges(this, "VBuffer2GBuffer Pass", "ASVGF-GradProj Pass");
      Addon::GBufferUtils::addPrevGBufferEdges(this, "GBufferPrev Pass", "ASVGF-GradProj Pass");

      // GBuffer Shading
      addPass(std::make_unique<Addon::GBufferShading>(), "GBufferShading Pass");
      Addon::GBufferUtils::addGBufferEdges(this, "ASVGF-GradProj Pass", "GBufferShading Pass");
      //Addon::GBufferUtils::addPrevGBufferEdges(this, "ASVGF-GradProj Pass", "GBufferShading Pass");
      addEdge("ASVGF-GradProj Pass", "RandSeed", "GBufferShading Pass", "RandSeed");
      addEdge("ASVGF-GradProj Pass", "RandPrev", "GBufferShading Pass", "RandPrev");
      addEdge("ASVGF-GradProj Pass", "Debug", "GBufferShading Pass", "Indicator");

      // Now blit HF, Spec, GradSamplePos and VBuffer img for next frame
      addPass(std::make_unique<BlitPass>(BlitPass::Descriptor{0, 0, 0, 0, BlitPass::SourceType::UINT}), "ASVGF Blit Diffuse");
      addEdge("GBufferShading Pass", "Diffuse", "ASVGF Blit Diffuse", "Source");
      addEdge("ASVGF-GradProj Pass", "HFPrev", "ASVGF Blit Diffuse", "Target");
      addPass(std::make_unique<BlitPass>(BlitPass::Descriptor{0, 0, 0, 0, BlitPass::SourceType::UINT}), "ASVGF Blit Specular");
      addEdge("GBufferShading Pass", "Specular", "ASVGF Blit Specular", "Source");
      addEdge("ASVGF-GradProj Pass", "SpecPrev", "ASVGF Blit Specular", "Target");
      addPass(std::make_unique<BlitPass>(BlitPass::Descriptor{0, 0, 0, 0, BlitPass::SourceType::UINT}), "ASVGF Blit GradSamplePos");
      addEdge("ASVGF-GradProj Pass", "GradSamplePos", "ASVGF Blit GradSamplePos", "Source");
      addEdge("ASVGF-GradProj Pass", "GradSamplePosPrev", "ASVGF Blit GradSamplePos", "Target");
      addPass(std::make_unique<BlitPass>(BlitPass::Descriptor{0, 0, 0, 0, BlitPass::SourceType::UINT4}), "ASVGF Blit VBuffer");
      addEdge("VBuffer Pass", "VBuffer", "ASVGF Blit VBuffer", "Source");
      addEdge("ASVGF-GradProj Pass", "VBufferPrev", "ASVGF Blit VBuffer", "Target");
      
      // create gradient image
      addPass(std::make_unique<Addon::ASVGF::GradientImagePass>(), "ASVGF-GradImg Pass");
      addEdge("ASVGF-GradProj Pass", "GradSamplePos", "ASVGF-GradImg Pass", "GradSamplePos");
      addEdge("ASVGF-GradProj Pass", "HfSpecLumPrev", "ASVGF-GradImg Pass", "HfSpecLumPrev");
      addEdge("GBufferShading Pass", "Diffuse", "ASVGF-GradImg Pass", "HF");
      addEdge("GBufferShading Pass", "Specular", "ASVGF-GradImg Pass", "Spec");
    
      // atrous gradient image
      addPass(std::make_unique<Addon::ASVGF::GradientAtrousPass>(0), "ASVGF-GradAtrous-0 Pass");
      addEdge("ASVGF-GradImg Pass", "GradHFSpec", "ASVGF-GradAtrous-0 Pass", "GradHFSpecPing");
      addEdge("ASVGF-GradImg Pass", "GradHFSpecBack", "ASVGF-GradAtrous-0 Pass", "GradHFSpecPong");
      addPass(std::make_unique<Addon::ASVGF::GradientAtrousPass>(1), "ASVGF-GradAtrous-1 Pass");
      addEdge("ASVGF-GradAtrous-0 Pass", "GradHFSpecPong", "ASVGF-GradAtrous-1 Pass", "GradHFSpecPing");
      addEdge("ASVGF-GradAtrous-0 Pass", "GradHFSpecPing", "ASVGF-GradAtrous-1 Pass", "GradHFSpecPong");
      addPass(std::make_unique<Addon::ASVGF::GradientAtrousPass>(2), "ASVGF-GradAtrous-2 Pass");
      addEdge("ASVGF-GradAtrous-1 Pass", "GradHFSpecPong", "ASVGF-GradAtrous-2 Pass", "GradHFSpecPing");
      addEdge("ASVGF-GradAtrous-1 Pass", "GradHFSpecPing", "ASVGF-GradAtrous-2 Pass", "GradHFSpecPong");
      
      // temporal accumulate pass
      addPass(std::make_unique<Addon::ASVGF::TemporalPass>(), "ASVGF-Temporal Pass");
      Addon::GBufferUtils::addGBufferEdges(this, "GBufferShading Pass", "ASVGF-Temporal Pass");
      Addon::GBufferUtils::addPrevGBufferEdges(this, "ASVGF-GradProj Pass", "ASVGF-Temporal Pass");
      addEdge("ASVGF-GradAtrous-2 Pass", "GradHFSpecPong", "ASVGF-Temporal Pass", "GradHF");
      addEdge("GBufferShading Pass", "Diffuse", "ASVGF-Temporal Pass", "HF");
      addEdge("GBufferShading Pass", "Specular", "ASVGF-Temporal Pass", "Spec");
      
      // blit pass - two moments and color history
      addPass(std::make_unique<BlitPass>(BlitPass::Descriptor{0, 0, 0, 0, BlitPass::SourceType::FLOAT4}), "Blit MomentsHistlenHF");
      addEdge("ASVGF-Temporal Pass", "MomentsHistlenHF", "Blit MomentsHistlenHF", "Source");
      addEdge("ASVGF-Temporal Pass", "MomentsHistlenHFPrev", "Blit MomentsHistlenHF", "Target");
      addPass(std::make_unique<BlitPass>(BlitPass::Descriptor{0, 0, 0, 0, BlitPass::SourceType::FLOAT4}), "Blit ColorHistlenSpec");
      addEdge("ASVGF-Temporal Pass", "ColorHistlenSpec", "Blit ColorHistlenSpec", "Source");
      addEdge("ASVGF-Temporal Pass", "ColorHistlenSpecPrev", "Blit ColorHistlenSpec", "Target");

      // 
      addPass(std::make_unique<Addon::ASVGF::AtrousPass>(0), "ASVGF-Atrous-0 Pass");
      Addon::GBufferUtils::addGBufferEdges(this, "ASVGF-Temporal Pass", "ASVGF-Atrous-0 Pass");
      addEdge("ASVGF-Temporal Pass", "Composite", "ASVGF-Atrous-0 Pass", "Composite");
      addEdge("ASVGF-Temporal Pass", "MomentsHistlenHF", "ASVGF-Atrous-0 Pass", "MomentsHistlenHF");
      addEdge("ASVGF-Temporal Pass", "AtrousHF", "ASVGF-Atrous-0 Pass", "AtrousHFPing");
      addEdge("ASVGF-Temporal Pass", "AtrousSpec", "ASVGF-Atrous-0 Pass", "AtrousSpecPing");
      addEdge("ASVGF-Temporal Pass", "AtrousMoments", "ASVGF-Atrous-0 Pass", "AtrousMomentPing");
      addEdge("ASVGF-Temporal Pass", "AtrousHFBack", "ASVGF-Atrous-0 Pass", "AtrousHFPong");
      addEdge("ASVGF-Temporal Pass", "AtrousSpecBack", "ASVGF-Atrous-0 Pass", "AtrousSpecPong");
      addEdge("ASVGF-Temporal Pass", "AtrousMomentsBack", "ASVGF-Atrous-0 Pass", "AtrousMomentPong");
      addEdge("ASVGF-GradProj Pass", "IsCorrelated", "ASVGF-Atrous-0 Pass", "IsCorrelated");
      
      addPass(std::make_unique<BlitPass>(BlitPass::Descriptor{0, 0, 0, 0, BlitPass::SourceType::UINT}), "Blit HFFiltered");
      addEdge("ASVGF-Atrous-0 Pass", "AtrousHFPong", "Blit HFFiltered", "Source");
      addEdge("ASVGF-Temporal Pass", "HFFilteredPrev", "Blit HFFiltered", "Target");
      
      addPass(std::make_unique<Addon::ASVGF::AtrousPass>(1), "ASVGF-Atrous-1 Pass");
      Addon::GBufferUtils::addGBufferEdges(this, "ASVGF-Atrous-0 Pass", "ASVGF-Atrous-1 Pass");
      addEdge("Blit HFFiltered", "Target", "ASVGF-Atrous-1 Pass", "AtrousHFPing");
      addEdge("ASVGF-Temporal Pass", "Composite", "ASVGF-Atrous-1 Pass", "Composite");
      addEdge("ASVGF-Temporal Pass", "MomentsHistlenHF", "ASVGF-Atrous-1 Pass", "MomentsHistlenHF");
      addEdge("ASVGF-Atrous-0 Pass", "AtrousSpecPong", "ASVGF-Atrous-1 Pass", "AtrousSpecPing");
      addEdge("ASVGF-Atrous-0 Pass", "AtrousMomentPong", "ASVGF-Atrous-1 Pass", "AtrousMomentPing");
      addEdge("ASVGF-Atrous-0 Pass", "AtrousHFPing", "ASVGF-Atrous-1 Pass", "AtrousHFPong");
      addEdge("ASVGF-Atrous-0 Pass", "AtrousSpecPing", "ASVGF-Atrous-1 Pass", "AtrousSpecPong");
      addEdge("ASVGF-Atrous-0 Pass", "AtrousMomentPing", "ASVGF-Atrous-1 Pass", "AtrousMomentPong");
      addEdge("ASVGF-Atrous-0 Pass", "IsCorrelated", "ASVGF-Atrous-1 Pass", "IsCorrelated");
      
      addPass(std::make_unique<Addon::ASVGF::AtrousPass>(2), "ASVGF-Atrous-2 Pass");
      Addon::GBufferUtils::addGBufferEdges(this, "ASVGF-Atrous-1 Pass", "ASVGF-Atrous-2 Pass");
      addEdge("ASVGF-Temporal Pass", "Composite", "ASVGF-Atrous-2 Pass", "Composite");
      addEdge("ASVGF-Temporal Pass", "MomentsHistlenHF", "ASVGF-Atrous-2 Pass", "MomentsHistlenHF");
      addEdge("ASVGF-Atrous-1 Pass", "AtrousHFPong", "ASVGF-Atrous-2 Pass", "AtrousHFPing");
      addEdge("ASVGF-Atrous-1 Pass", "AtrousSpecPong", "ASVGF-Atrous-2 Pass", "AtrousSpecPing");
      addEdge("ASVGF-Atrous-1 Pass", "AtrousMomentPong", "ASVGF-Atrous-2 Pass", "AtrousMomentPing");
      addEdge("ASVGF-Atrous-1 Pass", "AtrousHFPing", "ASVGF-Atrous-2 Pass", "AtrousHFPong");
      addEdge("ASVGF-Atrous-1 Pass", "AtrousSpecPing", "ASVGF-Atrous-2 Pass", "AtrousSpecPong");
      addEdge("ASVGF-Atrous-1 Pass", "AtrousMomentPing", "ASVGF-Atrous-2 Pass", "AtrousMomentPong");
      addEdge("ASVGF-Atrous-1 Pass", "IsCorrelated", "ASVGF-Atrous-2 Pass", "IsCorrelated");
      
      addPass(std::make_unique<Addon::ASVGF::AtrousPass>(3), "ASVGF-Atrous-3 Pass");
      Addon::GBufferUtils::addGBufferEdges(this, "ASVGF-Atrous-2 Pass", "ASVGF-Atrous-3 Pass");
      addEdge("ASVGF-Temporal Pass", "Composite", "ASVGF-Atrous-3 Pass", "Composite");
      addEdge("ASVGF-Temporal Pass", "MomentsHistlenHF", "ASVGF-Atrous-3 Pass", "MomentsHistlenHF");
      addEdge("ASVGF-Atrous-2 Pass", "AtrousHFPong", "ASVGF-Atrous-3 Pass", "AtrousHFPing");
      addEdge("ASVGF-Atrous-2 Pass", "AtrousSpecPong", "ASVGF-Atrous-3 Pass", "AtrousSpecPing");
      addEdge("ASVGF-Atrous-2 Pass", "AtrousMomentPong", "ASVGF-Atrous-3 Pass", "AtrousMomentPing");
      addEdge("ASVGF-Atrous-2 Pass", "AtrousHFPing", "ASVGF-Atrous-3 Pass", "AtrousHFPong");
      addEdge("ASVGF-Atrous-2 Pass", "AtrousSpecPing", "ASVGF-Atrous-3 Pass", "AtrousSpecPong");
      addEdge("ASVGF-Atrous-2 Pass", "AtrousMomentPing", "ASVGF-Atrous-3 Pass", "AtrousMomentPong");
      addEdge("ASVGF-Atrous-2 Pass", "IsCorrelated", "ASVGF-Atrous-3 Pass", "IsCorrelated");

    }

    // Blit history GBuffer
    addSubgraph(std::make_unique<Addon::GBufferHolderGraph>(), "GBuffer Blit Pass");
    Addon::GBufferUtils::addBlitPrevGBufferEdges(this, "GBufferShading Pass", "ASVGF-Temporal Pass", "GBuffer Blit Pass");
    
    //markOutput("GBufferShading Pass", "Debug");

    addPass(std::make_unique<AccumulatePass>(), "Accum Pass");
    addEdge("ASVGF-Atrous-3 Pass", "Composite", "Accum Pass", "Input");
    
    addPass(std::make_unique<Addon::Postprocess::ToneMapperPass>(), "ToneMapper Pass");
    addEdge("Accum Pass", "Output", "ToneMapper Pass", "Input");

    markOutput("ToneMapper Pass", "Output");
  }
};

SE_EXPORT struct CustomPipeline : public RDG::SingleGraphPipeline {
  CustomPipeline() { pGraph = &graph; }
  CustomGraph graph;
};

SE_EXPORT struct RestirGIGraph : public RDG::Graph {
  RestirGIGraph() {    
    restirgi_param = Addon::RestirGI::InitializeParameters(1280, 720);

    addPass(std::make_unique<Addon::VBuffer::RayTraceVBuffer>(), "VBuffer Pass");
    addPass(std::make_unique<Addon::VBuffer::VBuffer2GBufferPass>(), "VBuffer2GBuffer Pass");
    addEdge("VBuffer Pass", "VBuffer", "VBuffer2GBuffer Pass", "VBuffer");
    
    // Hold history GBuffer
    addPass(std::make_unique<Addon::GBufferHolderSource>(), "GBufferPrev Pass");
    
    addPass(std::make_unique<Addon::RestirGI::InitialSample>(&restirgi_param), "InitialSample Pass");
    Addon::GBufferUtils::addGBufferEdges(this, "VBuffer2GBuffer Pass", "InitialSample Pass");

    // Execute temporal resampling
    addPass(std::make_unique<Addon::RestirGI::TemporalResampling>(&restirgi_param), "TemporalResampling Pass");
    Addon::GBufferUtils::addGBufferEdges(this, "VBuffer2GBuffer Pass", "TemporalResampling Pass");
    Addon::GBufferUtils::addPrevGBufferEdges(this, "GBufferPrev Pass", "TemporalResampling Pass");
    addEdge("InitialSample Pass", "GIReservoir", "TemporalResampling Pass", "GIReservoir");
    
    // Execute spatial resampling
    addPass(std::make_unique<Addon::RestirGI::SpatialResampling>(&restirgi_param), "SpatialResampling Pass");
    Addon::GBufferUtils::addGBufferEdges(this, "VBuffer2GBuffer Pass", "SpatialResampling Pass");
    addEdge("TemporalResampling Pass", "GIReservoir", "SpatialResampling Pass", "GIReservoir");

    addPass(std::make_unique<Addon::RestirGI::FinalShading>(&restirgi_param), "FinalShading Pass");
    Addon::GBufferUtils::addGBufferEdges(this, "VBuffer2GBuffer Pass", "FinalShading Pass");
    addEdge("SpatialResampling Pass", "GIReservoir", "FinalShading Pass", "GIReservoir");

    addPass(std::make_unique<AccumulatePass>(), "Accum Pass");
    addEdge("FinalShading Pass", "Diffuse", "Accum Pass", "Input");
    
    addSubgraph(std::make_unique<Addon::GBufferHolderGraph>(), "GBuffer Blit Pass");
    Addon::GBufferUtils::addBlitPrevGBufferEdges(
        this, "VBuffer2GBuffer Pass", "TemporalResampling Pass", "GBuffer Blit Pass");
    
    addPass(std::make_unique<Addon::Postprocess::ToneMapperPass>(), "ToneMapper Pass");
    addEdge("Accum Pass", "Output", "ToneMapper Pass", "Input");

    markOutput("ToneMapper Pass", "Output");
  }
  Addon::RestirGI::GIResamplingRuntimeParameters restirgi_param;
};

SE_EXPORT struct RestirGIPipeline : public RDG::SingleGraphPipeline {
  RestirGIPipeline() { pGraph = &graph; }
  RestirGIGraph graph;
};

SE_EXPORT struct SemiNEEGraph : public RDG::Graph {
  SemiNEEGraph() {
    sort_setting.element_count = 1280 * 720;
    
    addPass(std::make_unique<Addon::SemiNEE::InitialSamplePass>(),
            "Initial Sample Pass");
    addPass(std::make_unique<Addon::SemiNEE::LeafEncodePass>(),
            "Leaf Encode Pass");

    addSubgraph(
        std::make_unique<Addon::BitonicSort::BitonicSort>(&sort_setting),
        "Sort Pass");

    addEdge("Initial Sample Pass", "DiffuseVPLs", "Leaf Encode Pass", "DiffuseVPLs");

    addEdge("Leaf Encode Pass", "LeafCodes", "Sort Pass", "Input");

    addPass(std::make_unique<Addon::SemiNEE::TreeInitPass>(), "TreeInit Pass");
    addEdge("Sort Pass", "Output", "TreeInit Pass", "LeafCodesSorted");
    addPass(std::make_unique<Addon::SemiNEE::TreeLeavesPass>(),
            "TreeLeaves Pass");
    addEdge("TreeInit Pass", "TreeNodes", "TreeLeaves Pass", "TreeNodes");
    addEdge("TreeInit Pass", "IndirectArgs", "TreeLeaves Pass", "IndirectArgs");
    addEdge("Sort Pass", "Output", "TreeLeaves Pass", "LeafCodesSorted");
    addPass(std::make_unique<Addon::SemiNEE::TreeInternalPass>(),
            "TreeInternal Pass");
    addEdge("TreeLeaves Pass", "TreeNodes", "TreeInternal Pass", "TreeNodes");
    addEdge("TreeInit Pass", "IndirectArgs", "TreeInternal Pass", "IndirectArgs");
    addEdge("Sort Pass", "Output", "TreeInternal Pass", "LeafCodesSorted");

    addPass(std::make_unique<Addon::SemiNEE::TreeMergePass>(),
            "TreeMerge Pass");
    addEdge("TreeInternal Pass", "TreeNodes", "TreeMerge Pass", "TreeNodes");
    addEdge("Initial Sample Pass", "DiffuseVPLs", "TreeMerge Pass", "VPLData");
    addEdge("TreeInit Pass", "VPLMerges", "TreeMerge Pass", "VPLMerges");
    addEdge("TreeInit Pass", "IndirectArgs", "TreeMerge Pass", "IndirectArgs");

    addPass(std::make_unique<Addon::SemiNEE::TileBasedDistPass>(),
            "TileDist Build Pass");
    addEdge("TreeMerge Pass", "VPLData", "TileDist Build Pass", "VPLData");
    
    addPass(std::make_unique<Addon::SemiNEE::TileDistExchangePass>(),
            "TileDist Exchange Pass");
    addEdge("TreeMerge Pass", "VPLData", "TileDist Exchange Pass", "VPLData");
    addEdge("TileDist Build Pass", "VPLSelection", "TileDist Exchange Pass", "VPLSelection");
    
    addPass(std::make_unique<Addon::SemiNEE::TileDistVisualizePass>(),
            "TileDist Visualize Pass");
    addEdge("TreeMerge Pass", "VPLData", "TileDist Visualize Pass", "VPLData");
    addEdge("TileDist Exchange Pass", "VPLSelection", "TileDist Visualize Pass", "SelectedVPL");

    addPass(std::make_unique<Addon::SemiNEE::TileDistSamplePass>(),
        "TileDistSample Pass");
    addEdge("TreeMerge Pass", "VPLData", "TileDistSample Pass", "VPLData");
    addEdge("TileDist Exchange Pass", "VPLSelection", "TileDistSample Pass",
            "VPLSelection");
     //markOutput("TileDistSample Pass", "Color");

    addPass(std::make_unique<Addon::SemiNEE::DVPLVisualizePass>(),
        "DVPL Visualize Pass");
    addEdge("TreeMerge Pass", "VPLData", "DVPL Visualize Pass", "DiffuseVPLs");

    addPass(std::make_unique<Addon::SemiNEE::TileDistPerPixelVisPass>(),
        "TileDistPixVis Pass");
    //addEdge("TreeMerge Pass", "VPLData", "DVPL Visualize Pass", "DiffuseVPLs");
    //markOutput("DVPL Visualize Pass", "Color");

    addPass(std::make_unique<Addon::SemiNEE::TreeVisualizePass>(),
            "TreeVisualize Pass");
    addEdge("TreeMerge Pass", "VPLData", "TreeVisualize Pass", "DiffuseVPLs");
    addEdge("TreeMerge Pass", "TreeNodes", "TreeVisualize Pass", "TreeNodes");
    addEdge("TreeInit Pass", "IndirectArgs", "TreeVisualize Pass", "IndirectArgs");

    //markOutput("TreeVisualize Pass", "Color");
    addPass(std::make_unique<AccumulatePass>(), "Accum Pass");

    addEdge("TileDistSample Pass", "Color", "Accum Pass", "Input");


    markOutput("Accum Pass", "Output");

    //addPass(std::make_unique<Addon::SemiNEE::TestQuadSamplePass>(),
    //        "TestQuadSample Pass");
    //addPass(std::make_unique<AccumulatePass>(), "Accumulate Pass");
    //addEdge("TestQuadSample Pass", "Color", "Accumulate Pass", "Input");

    //markOutput("Accumulate Pass", "Output");
  }

  Addon::BitonicSort::BitonicSortSetting sort_setting;
};

SE_EXPORT struct SemiNEEPipeline : public RDG::SingleGraphPipeline {
  SemiNEEPipeline() { pGraph = &graph; }
  SemiNEEGraph graph;
};

SE_EXPORT struct GTGraph : public RDG::Graph {
  GTGraph() {
    addPass(std::make_unique<Addon::SemiNEE::GroundTruthPass>(), "UDPT Pass");
    addPass(std::make_unique<AccumulatePass>(), "Accum Pass");
    addEdge("UDPT Pass", "Color", "Accum Pass", "Input");

    addPass(std::make_unique<Addon::VBuffer::RayTraceVBuffer>(), "VBuffer Pass");
    //addPass(std::make_unique<Addon::SSGuiding::PdfNormalize_ClearPass>(), "PdfClear Pass");
    //addPass(std::make_unique<Addon::SSGuiding::PdfNormalize_TestPass>(), "PdfTest Pass");
    //addEdge("VBuffer Pass", "VBuffer", "PdfTest Pass", "VBuffer");
    //addEdge("PdfClear Pass", "PdfNormalizing", "PdfTest Pass", "PdfNormalizing");
    //addPass(std::make_unique<Addon::SSGuiding::PdfNormalize_SumPass>(), "PdfNormalize Pass");
    //addEdge("PdfClear Pass", "PdfNormalizingInfo", "PdfNormalize Pass", "PdfNormalizingInfo");
    //addEdge("PdfTest Pass", "PdfNormalizing", "PdfNormalize Pass", "PdfNormalizing");
    //addPass(std::make_unique<Addon::SSGuiding::PdfNormalize_ViewerPass>(), "PdfViewer Pass");
    //addEdge("PdfNormalize Pass", "PdfNormalizingInfo", "PdfViewer Pass", "PdfNormalizingInfo");
    //addEdge("PdfNormalize Pass", "PdfNormalizing", "PdfViewer Pass", "PdfNormalizing");
    //
    addPass(std::make_unique<Addon::Postprocess::ToneMapperPass>(), "ToneMapper Pass");
    addEdge("Accum Pass", "Output", "ToneMapper Pass", "Input");


    markOutput("Accum Pass", "Output");
  }
};

SE_EXPORT struct GTPipeline : public RDG::SingleGraphPipeline {
  GTPipeline() { pGraph = &graph; }
  GTGraph graph;
};

SE_EXPORT struct CDQGraph : public RDG::Graph {
  CDQGraph() {
    addPass(std::make_unique<Addon::SemiNEE::GroundTruthPass>(), "UDPT Pass");
    addPass(std::make_unique<AccumulatePass>(), "Accum Pass");
    addEdge("UDPT Pass", "Color", "Accum Pass", "Input");

    addPass(std::make_unique<Addon::VBuffer::RayTraceVBuffer>(), "VBuffer Pass");
    addPass(std::make_unique<Addon::SSGuiding::PdfNormalize_ClearPass>(), "PdfClear Pass");
    addPass(std::make_unique<Addon::SSGuiding::CDQ_PresamplePass>(), "CDQPresample Pass");
    addEdge("VBuffer Pass", "VBuffer", "CDQPresample Pass", "VBuffer");
    addPass(std::make_unique<Addon::SSGuiding::CDQ_AdaptionPass>(), "CDQAdaptive Pass");
    addEdge("CDQPresample Pass", "PresampleList", "CDQAdaptive Pass", "PresampleList");
    addPass(std::make_unique<Addon::SSGuiding::CDQ_VisualizePass>(), "PdfTest Pass");
    addEdge("CDQAdaptive Pass", "CDQBuffer", "PdfTest Pass", "CDQBuffer");
    addEdge("PdfClear Pass", "PdfNormalizing", "PdfTest Pass", "PdfNormalizing");
    addPass(std::make_unique<Addon::SSGuiding::PdfNormalize_SumPass>(), "PdfNormalize Pass");
    addEdge("PdfClear Pass", "PdfNormalizingInfo", "PdfNormalize Pass", "PdfNormalizingInfo");
    addEdge("PdfTest Pass", "PdfNormalizing", "PdfNormalize Pass", "PdfNormalizing");
    addPass(std::make_unique<Addon::SSGuiding::PdfNormalize_ViewerPass>(), "PdfViewer Pass");
    addEdge("PdfNormalize Pass", "PdfNormalizingInfo", "PdfViewer Pass", "PdfNormalizingInfo");
    addEdge("PdfNormalize Pass", "PdfNormalizing", "PdfViewer Pass", "PdfNormalizing");
    
    markOutput("Accum Pass", "Output");
  }
};

SE_EXPORT struct CDQPipeline : public RDG::SingleGraphPipeline {
  CDQPipeline() { pGraph = &graph; }
  CDQGraph graph;
};

SE_EXPORT struct VXDIGraph : public RDG::Graph {
  Addon::VXGI::VXGISetting setting;
  Addon::VXGuiding::DITestSetting diTestSetting;

  VXDIGraph() {
    setting.clipmapSetting.mip = 6;

    addPass(std::make_unique<Addon::VXGuiding::VoxelClear6DPass>(&setting),
            "Voxelize6DClear Pass");

    addPass(std::make_unique<Addon::VXGuiding::Voxelize6DPass>(&setting),
                "Voxelize6D Pass");
    for (int i = 0; i < 6; ++i)
        addEdge("Voxelize6DClear Pass", "RadOpaVox6DTex" +
        std::to_string(i),
                "Voxelize6D Pass", "RadOpaVox6DTex" + std::to_string(i));

    addPass(std::make_unique<Addon::VXGuiding::DITestInjectPass>(&diTestSetting,
                                                                 &setting),
            "DITestInject Pass");
    addEdge("Voxelize6D Pass", "Depth", "DITestInject Pass", "Depth");
    for (int i = 0; i < 6; ++i)
        addEdge("Voxelize6D Pass", "RadOpaVox6DTex" + std::to_string(i),
                "DITestInject Pass", "RadOpaVox6DTex" + std::to_string(i));
   
    // Create mipmap
    addPass(std::make_unique<Addon::VXGuiding::VoxelMip6DPass>(&setting, false),
            "VoxelMIP6D 1st Pass");
    for (int i = 0; i < 6; ++i)
        addEdge("DITestInject Pass", "RadOpaVox6DTex" + std::to_string(i),
                "VoxelMIP6D 1st Pass", "RadOpaVox6DTex" + std::to_string(i));
    addPass(std::make_unique<Addon::VXGuiding::VoxelMip6DPass>(&setting, true),
            "VoxelMIP6D 2nd Pass");
    for (int i = 0; i < 6; ++i)
        addEdge("VoxelMIP6D 1st Pass", "RadOpaVox6DTex" + std::to_string(i),
                "VoxelMIP6D 2nd Pass", "RadOpaVox6DTex" + std::to_string(i));

    addPass(std::make_unique<Addon::VXGuiding::DITestVoxelCheckPass>(
                &diTestSetting, &setting),
            "Vox6dVisualize Pass");
    for (int i = 0; i < 6; ++i)
        addEdge("VoxelMIP6D 2nd Pass", "RadOpaVox6DTex" + std::to_string(i),
                "Vox6dVisualize Pass", "RadOpaVox6DTex" + std::to_string(i));

    addPass(std::make_unique<Addon::VXGuiding::DITestPass>(&diTestSetting,
                                                           &setting),
            "DITestComp Pass");
    for (int i = 0; i < 6; ++i)
        addEdge("VoxelMIP6D 2nd Pass", "RadOpaVox6DTex" + std::to_string(i),
                "DITestComp Pass", "RadOpaVox6DTex" + std::to_string(i));

    addPass(std::make_unique<AccumulatePass>(), "Accum Pass");
    addEdge("DITestComp Pass", "Color", "Accum Pass", "Input");

    markOutput("Accum Pass", "Output");
  }
};

SE_EXPORT struct VXDIPipeline : public RDG::SingleGraphPipeline {
  VXDIPipeline() { pGraph = &graph; }
  VXDIGraph graph;
};

SE_EXPORT struct VXGIGraph : public RDG::Graph {
  Addon::VXGI::VXGISetting setting;
  Addon::VXGuiding::DITestSetting diTestSetting;

  VXGIGraph() {
    setting.clipmapSetting.mip = 6;

    addPass(std::make_unique<Addon::VXGuiding::VoxelClear6DPass>(&setting),
            "Voxelize6DClear Pass");

    addPass(std::make_unique<Addon::VXGuiding::Voxelize6DPass>(&setting),
            "Voxelize6D Pass");
    for (int i = 0; i < 6; ++i)
      addEdge("Voxelize6DClear Pass", "RadOpaVox6DTex" + std::to_string(i),
              "Voxelize6D Pass", "RadOpaVox6DTex" + std::to_string(i));

    // Injection direct light by ray tracing (?)
    addPass(std::make_unique<Addon::VXGuiding::Voxel6DRTInjection>(&setting),
            "RTInject Pass");
    for (int i = 0; i < 6; ++i)
      addEdge("Voxelize6D Pass", "RadOpaVox6DTex" + std::to_string(i),
              "RTInject Pass", "RadOpaVox6DTex" + std::to_string(i));
    
    // Create mipmap
    addPass(std::make_unique<Addon::VXGuiding::VoxelMip6DPass>(&setting, false),
            "VoxelMIP6D 1st Pass");
    for (int i = 0; i < 6; ++i)
      addEdge("RTInject Pass", "RadOpaVox6DTex" + std::to_string(i),
              "VoxelMIP6D 1st Pass", "RadOpaVox6DTex" + std::to_string(i));
    addPass(std::make_unique<Addon::VXGuiding::VoxelMip6DPass>(&setting, true),
            "VoxelMIP6D 2nd Pass");
    for (int i = 0; i < 6; ++i)
      addEdge("VoxelMIP6D 1st Pass", "RadOpaVox6DTex" + std::to_string(i),
              "VoxelMIP6D 2nd Pass", "RadOpaVox6DTex" + std::to_string(i));

    addPass(std::make_unique<Addon::VXGuiding::Voxel6DVisualizePass>(&setting),
            "Vox6dVisualize Pass");
    for (int i = 0; i < 6; ++i)
      addEdge("VoxelMIP6D 2nd Pass", "RadOpaVox6DTex" + std::to_string(i),
              "Vox6dVisualize Pass", "RadOpaVox6DTex" + std::to_string(i));

    addPass(std::make_unique<Addon::VXGuiding::GITestPass>(&setting),
            "GITestComp Pass");
    for (int i = 0; i < 6; ++i)
      addEdge("VoxelMIP6D 2nd Pass", "RadOpaVox6DTex" + std::to_string(i),
              "GITestComp Pass", "RadOpaVox6DTex" + std::to_string(i));

    addPass(std::make_unique<Addon::VXGI::ConeTraceDebuggerPass>(&setting),
            "ConetraceDebugger Pass");
    addEdge("GITestComp Pass", "Color", "ConetraceDebugger Pass", "Color");
    for (int i = 0; i < 6; ++i)
      addEdge("VoxelMIP6D 2nd Pass", "RadOpaVox6DTex" + std::to_string(i),
              "ConetraceDebugger Pass", "RadOpaVox6DTex" + std::to_string(i));

    addPass(std::make_unique<Addon::VXGuiding::ImportInjectPass>(&setting),
            "Importance Pass");
    for (int i = 0; i < 6; ++i)
      addEdge("VoxelMIP6D 2nd Pass", "RadOpaVox6DTex" + std::to_string(i),
              "Importance Pass", "RadOpaVox6DTex" + std::to_string(i));
    addPass(std::make_unique<Addon::VXGuiding::Voxel6DVisualizePass>(&setting, true),
            "ImportanceVis Pass");
    addEdge("Importance Pass", "ImportTex",
            "ImportanceVis Pass", "RadOpaVox6DTex");

    addPass(std::make_unique<AccumulatePass>(), "Accum Pass");
    addEdge("ConetraceDebugger Pass", "Color", "Accum Pass", "Input");

    markOutput("Accum Pass", "Output");
  }
};

SE_EXPORT struct SSPGGraph : public RDG::Graph {
  SSPGGraph() {
    // Get the gbuffer
    addPass(std::make_unique<Addon::VBuffer::RayTraceVBuffer>(), "VBuffer Pass");
    addPass(std::make_unique<Addon::VBuffer::VBuffer2GBufferPass>(), "VBuffer2GBuffer Pass");
    addEdge("VBuffer Pass", "VBuffer", "VBuffer2GBuffer Pass", "VBuffer");
    addPass(std::make_unique<Addon::GBufferHolderSource>(), "GBufferPrev Pass");
    // Screen space pth guiding
    addPass(std::make_unique<Addon::SSGuiding::SSPGvMF_ClearPass>(), "GuiderClear Pass");
    addPass(std::make_unique<Addon::SSGuiding::SSPGvMF_SamplePass>(), "Sample Pass");
    Addon::GBufferUtils::addGBufferEdges(this, "VBuffer2GBuffer Pass", "Sample Pass");
    Addon::GBufferUtils::addPrevGBufferEdges(this, "GBufferPrev Pass", "Sample Pass");
    addEdge("GuiderClear Pass", "vMFStatistics", "Sample Pass", "vMFStatistics");
    addEdge("GuiderClear Pass", "EpochCounter", "Sample Pass", "EpochCounter");
    addEdge("GuiderClear Pass", "vMFStatisticsPrev", "Sample Pass", "vMFStatisticsPrev");
    addEdge("GuiderClear Pass", "EpochCounterPrev", "Sample Pass", "EpochCounterPrev");

    addPass(std::make_unique<Addon::SSGuiding::SSPGvMF_LearnPass>(), "Learn Pass");
    addEdge("Sample Pass", "vMFStatistics", "Learn Pass", "vMFStatistics");
    addEdge("Sample Pass", "EpochCounter", "Learn Pass", "EpochCounter");
    addEdge("Sample Pass", "VPL0", "Learn Pass", "VPL0");
    addEdge("Sample Pass", "VPL1", "Learn Pass", "VPL1");
    addEdge("VBuffer Pass", "VBuffer", "Learn Pass", "VBuffer");
    // Gbuffer temporal copy
    addSubgraph(std::make_unique<Addon::GBufferHolderGraph>(), "GBuffer Blit Pass");
    Addon::GBufferUtils::addBlitPrevGBufferEdges(
        this, "VBuffer2GBuffer Pass", "Sample Pass", "GBuffer Blit Pass");
    // Post processing
    addPass(std::make_unique<AccumulatePass>(), "Accum Pass");
    addEdge("Sample Pass", "Color", "Accum Pass", "Input");
    addPass(std::make_unique<Addon::Postprocess::ToneMapperPass>(), "ToneMapper Pass");
    addEdge("Accum Pass", "Output", "ToneMapper Pass", "Input");

    addPass(std::make_unique<BlitPass>(BlitPass::Descriptor{0, 0, 0, 0, BlitPass::SourceType::FLOAT4}), "BlitPack0");
    addEdge("Learn Pass", "vMFStatistics", "BlitPack0", "Source");
    addEdge("Sample Pass", "vMFStatisticsPrev", "BlitPack0", "Target");
    addPass(std::make_unique<BlitPass>(BlitPass::Descriptor{0, 0, 0, 0, BlitPass::SourceType::UINT}), "BlitPack1");
    addEdge("Learn Pass", "EpochCounter", "BlitPack1", "Source");
    addEdge("Sample Pass", "EpochCounterPrev", "BlitPack1", "Target");

    // output the accum result
    markOutput("ToneMapper Pass", "Output");
  }
};

SE_EXPORT struct SSPGPipeline : public RDG::SingleGraphPipeline {
  SSPGPipeline() { pGraph = &graph; }
  SSPGGraph graph;
};

SE_EXPORT struct SSPGReSTIRGraph : public RDG::Graph {
  SSPGReSTIRGraph() {
    restirgi_param = Addon::RestirGI::InitializeParameters(1280, 720);

    // Get the gbuffer
    addPass(std::make_unique<Addon::VBuffer::RayTraceVBuffer>(), "VBuffer Pass");
    addPass(std::make_unique<Addon::VBuffer::VBuffer2GBufferPass>(), "VBuffer2GBuffer Pass");
    addEdge("VBuffer Pass", "VBuffer", "VBuffer2GBuffer Pass", "VBuffer");
    addPass(std::make_unique<Addon::GBufferHolderSource>(), "GBufferPrev Pass");
    // Screen space pth guiding
    addPass(std::make_unique<Addon::SSGuiding::SSPGvMF_ClearPass>(), "GuiderClear Pass");
    addPass(std::make_unique<Addon::SSGuiding::SSPGvMF_SampleReSTIRPass>(&restirgi_param), "Sample Pass");
    Addon::GBufferUtils::addGBufferEdges(this, "VBuffer2GBuffer Pass", "Sample Pass");
    Addon::GBufferUtils::addPrevGBufferEdges(this, "GBufferPrev Pass", "Sample Pass");
    addEdge("GuiderClear Pass", "vMFStatistics", "Sample Pass", "vMFStatistics");
    addEdge("GuiderClear Pass", "EpochCounter", "Sample Pass", "EpochCounter");
    addEdge("GuiderClear Pass", "vMFStatisticsPrev", "Sample Pass", "vMFStatisticsPrev");
    addEdge("GuiderClear Pass", "EpochCounterPrev", "Sample Pass", "EpochCounterPrev");

    addPass(std::make_unique<Addon::SSGuiding::SSPGvMF_LearnPass>(), "Learn Pass");
    addEdge("Sample Pass", "vMFStatistics", "Learn Pass", "vMFStatistics");
    addEdge("Sample Pass", "EpochCounter", "Learn Pass", "EpochCounter");
    addEdge("Sample Pass", "VPL0", "Learn Pass", "VPL0");
    addEdge("Sample Pass", "VPL1", "Learn Pass", "VPL1");
    addEdge("VBuffer Pass", "VBuffer", "Learn Pass", "VBuffer");
    
    addPass(std::make_unique<BlitPass>(BlitPass::Descriptor{0, 0, 0, 0, BlitPass::SourceType::FLOAT4}), "BlitPack0");
    addEdge("Learn Pass", "vMFStatistics", "BlitPack0", "Source");
    addEdge("Sample Pass", "vMFStatisticsPrev", "BlitPack0", "Target");
    addPass(std::make_unique<BlitPass>(BlitPass::Descriptor{0, 0, 0, 0, BlitPass::SourceType::UINT}), "BlitPack1");
    addEdge("Learn Pass", "EpochCounter", "BlitPack1", "Source");
    addEdge("Sample Pass", "EpochCounterPrev", "BlitPack1", "Target");

    // ReSTIR process
    // --------------------------------------------------------------------------------
    // Execute temporal resampling
    addPass(std::make_unique<Addon::RestirGI::TemporalResampling>(&restirgi_param), "TemporalResampling Pass");
    Addon::GBufferUtils::addGBufferEdges(this, "VBuffer2GBuffer Pass", "TemporalResampling Pass");
    Addon::GBufferUtils::addPrevGBufferEdges(this, "GBufferPrev Pass", "TemporalResampling Pass");
    addEdge("Sample Pass", "GIReservoir", "TemporalResampling Pass", "GIReservoir");

    // Execute spatial resampling
    addPass(std::make_unique<Addon::RestirGI::SpatialResampling>(&restirgi_param), "SpatialResampling Pass");
    Addon::GBufferUtils::addGBufferEdges(this, "VBuffer2GBuffer Pass", "SpatialResampling Pass");
    addEdge("TemporalResampling Pass", "GIReservoir", "SpatialResampling Pass", "GIReservoir");

    // Execute final shading
    addPass(std::make_unique<Addon::RestirGI::FinalShading>(&restirgi_param), "FinalShading Pass");
    Addon::GBufferUtils::addGBufferEdges(this, "VBuffer2GBuffer Pass", "FinalShading Pass");
    addEdge("SpatialResampling Pass", "GIReservoir", "FinalShading Pass", "GIReservoir");

    // All post-processing stuff ...
    addPass(std::make_unique<AccumulatePass>(), "Accum Pass");
    addEdge("FinalShading Pass", "Diffuse", "Accum Pass", "Input");
    addSubgraph(std::make_unique<Addon::GBufferHolderGraph>(), "GBuffer Blit Pass");
    Addon::GBufferUtils::addBlitPrevGBufferEdges(this, "VBuffer2GBuffer Pass", "TemporalResampling Pass", "GBuffer Blit Pass");
    addPass(std::make_unique<Addon::Postprocess::ToneMapperPass>(), "ToneMapper Pass");
    addEdge("Accum Pass", "Output", "ToneMapper Pass", "Input");

    markOutput("ToneMapper Pass", "Output");
  }

  Addon::RestirGI::GIResamplingRuntimeParameters restirgi_param;
};

SE_EXPORT struct SSPGReSTIRPipeline : public RDG::SingleGraphPipeline {
  SSPGReSTIRPipeline() { pGraph = &graph; }
  SSPGReSTIRGraph graph;
};

SE_EXPORT struct SSPG_GMM_Graph : public RDG::Graph {
  SSPG_GMM_Graph() {
    // Get the gbuffer
    addPass(std::make_unique<Addon::VBuffer::RayTraceVBuffer>(), "VBuffer Pass");
    addPass(std::make_unique<Addon::VBuffer::VBuffer2GBufferPass>(), "VBuffer2GBuffer Pass");
    addEdge("VBuffer Pass", "VBuffer", "VBuffer2GBuffer Pass", "VBuffer");
    addPass(std::make_unique<Addon::GBufferHolderSource>(), "GBufferPrev Pass");
    // Screen space pth guiding
    addPass(std::make_unique<Addon::SSGuiding::SSPGGMM_ClearPass>(), "GuiderClear Pass");
    addPass(std::make_unique<Addon::SSGuiding::SSPGGMM_SamplePass>(), "Sample Pass");
    Addon::GBufferUtils::addGBufferEdges(this, "VBuffer2GBuffer Pass", "Sample Pass");
    Addon::GBufferUtils::addPrevGBufferEdges(this, "GBufferPrev Pass", "Sample Pass");
    addEdge("GuiderClear Pass", "GMMStatisticsPack0", "Sample Pass", "GMMStatisticsPack0");
    addEdge("GuiderClear Pass", "GMMStatisticsPack1", "Sample Pass", "GMMStatisticsPack1");
    addEdge("GuiderClear Pass", "GMMStatisticsPack0Prev", "Sample Pass", "GMMStatisticsPack0Prev");
    addEdge("GuiderClear Pass", "GMMStatisticsPack1Prev", "Sample Pass", "GMMStatisticsPack1Prev");
    addPass(std::make_unique<Addon::SSGuiding::SSPGGMM_LearnPass>(), "Learn Pass");
    Addon::GBufferUtils::addGBufferEdges(this, "VBuffer2GBuffer Pass", "Learn Pass");
    addEdge("Sample Pass", "GMMStatisticsPack0", "Learn Pass", "GMMStatisticsPack0");
    addEdge("Sample Pass", "GMMStatisticsPack1", "Learn Pass", "GMMStatisticsPack1");
    addEdge("Sample Pass", "VPLs", "Learn Pass", "VPLs");
    // Gbuffer temporal copy
    addSubgraph(std::make_unique<Addon::GBufferHolderGraph>(), "GBuffer Blit Pass");
    Addon::GBufferUtils::addBlitPrevGBufferEdges(this, "VBuffer2GBuffer Pass", "Sample Pass", "GBuffer Blit Pass");
    // Post processing
    addPass(std::make_unique<AccumulatePass>(), "Accum Pass");
    addEdge("Sample Pass", "Color", "Accum Pass", "Input");
    addPass(std::make_unique<Addon::Postprocess::ToneMapperPass>(), "ToneMapper Pass");
    addEdge("Accum Pass", "Output", "ToneMapper Pass", "Input");

    addPass(std::make_unique<BlitPass>(BlitPass::Descriptor{0, 0, 0, 0, BlitPass::SourceType::FLOAT4}), "BlitPack0");
    addEdge("Learn Pass", "GMMStatisticsPack0", "BlitPack0", "Source");
    addEdge("Sample Pass", "GMMStatisticsPack0Prev", "BlitPack0", "Target");
    addPass(std::make_unique<BlitPass>(BlitPass::Descriptor{0, 0, 0, 0, BlitPass::SourceType::FLOAT4}), "BlitPack1");
    addEdge("Learn Pass", "GMMStatisticsPack1", "BlitPack1", "Source");
    addEdge("Sample Pass", "GMMStatisticsPack1Prev", "BlitPack1", "Target");

    // output the accum result
    markOutput("Accum Pass", "Output");
    // Visualize screen space path guding
    bool vis_gmm = false;
    if (vis_gmm) {
      addPass(std::make_unique<Addon::SSGuiding::PdfNormalize_ClearPass>(), "PdfClear Pass");
      addPass(std::make_unique<Addon::SSGuiding::SSPGGMM_VisPass>(), "Vis Pass");
      addEdge("VBuffer Pass", "VBuffer", "Vis Pass", "VBuffer");
      addEdge("Sample Pass", "GMMStatisticsPack0", "Vis Pass", "GMMStatisticsPack0");
      addEdge("Sample Pass", "GMMStatisticsPack1", "Vis Pass", "GMMStatisticsPack1");
      addEdge("PdfClear Pass", "PdfNormalizing", "Vis Pass", "PdfNormalizing");
      addPass(std::make_unique<Addon::SSGuiding::PdfNormalize_SumPass>(), "PdfNormalize Pass");
      addEdge("PdfClear Pass", "PdfNormalizingInfo", "PdfNormalize Pass", "PdfNormalizingInfo");
      addEdge("Vis Pass", "PdfNormalizing", "PdfNormalize Pass", "PdfNormalizing");
      addPass(std::make_unique<Addon::SSGuiding::PdfNormalize_ViewerPass>(), "PdfViewer Pass");
      addEdge("PdfNormalize Pass", "PdfNormalizingInfo", "PdfViewer Pass", "PdfNormalizingInfo");
      addEdge("PdfNormalize Pass", "PdfNormalizing", "PdfViewer Pass", "PdfNormalizing");
      //addPass(std::make_unique<Addon::SSGuiding::PdfAccum_ClearPass>(), "PdfAccumClear Pass");
      //addPass(std::make_unique<Addon::SSGuiding::SSPGGMM_TestPass>(), "SSPGGMTest Pass");
      //addEdge("PdfAccumClear Pass", "PdfAccumulator", "SSPGGMTest Pass", "PdfAccumulator");
      //addEdge("PdfAccumClear Pass", "PdfAccumulatorInfo", "SSPGGMTest Pass", "PdfAccumulatorInfo");
      //addEdge("Sample Pass", "GMMStatisticsPack0", "SSPGGMTest Pass", "GMMStatisticsPack0");
      //addEdge("Sample Pass", "GMMStatisticsPack1", "SSPGGMTest Pass", "GMMStatisticsPack1");
      //addPass(std::make_unique<Addon::SSGuiding::PdfAccum_ViewerPass>(), "PdfAccumViewer Pass");
      //addEdge("SSPGGMTest Pass", "PdfAccumulator", "PdfAccumViewer Pass", "PdfAccumulator");
      //addEdge("SSPGGMTest Pass", "PdfAccumulatorInfo", "PdfAccumViewer Pass", "PdfAccumulatorInfo");
    }
  }
};

SE_EXPORT struct SSPGP_GMM_Pipeline : public RDG::SingleGraphPipeline {
  SSPGP_GMM_Pipeline() { pGraph = &graph; }
  SSPG_GMM_Graph graph;
};

SE_EXPORT struct VXGuidingPrebakeGraph : public RDG::Graph {
  Addon::VXGI::VXGISetting setting;
  VXGuidingPrebakeGraph() {
    addPass(std::make_unique<Addon::VXGuiding::VXGuiderBakeCleanPass>(), "PrebakeClean Pass");
    addPass(std::make_unique<Addon::VXGuiding::VXGuiderGeometryBakePass>(&setting), "Prebake Pass");
    addEdge("PrebakeClean Pass", "AABBMin", "Prebake Pass", "AABBMin");
    addEdge("PrebakeClean Pass", "AABBMax", "Prebake Pass", "AABBMax");
  }
};

SE_EXPORT struct VXGuidingASVGFGraph : public RDG::Graph {
  Addon::VXGI::VXGISetting setting;
  Addon::VXGuiding::DITestSetting diTestSetting;
  Addon::gSLICr::gSLICrSetting slicSetting;
  Addon::VXGuiding::VXGuidingSetting vxgSetting;
  bool need_rebake = false;

  VXGuidingASVGFGraph() {
    setting.clipmapSetting.mip = 6;
    slicSetting.img_size = {1280, 720};
    slicSetting.map_size = {(1280 + 31) / 32, (720 + 31) / 32};
    slicSetting.number_iter = 5;
    slicSetting.spixel_size = 32;
    vxgSetting.use_average_irradiance = true;

    // prepare stages
    addPass(std::make_unique<Addon::VBuffer::RayTraceVBuffer>(), "VBuffer Pass");
    addPass(std::make_unique<Addon::VXGuiding::PrebakeDummyPass>(need_rebake), "Prebake Pass");
    addPass(std::make_unique<Addon::VXGuiding::VXGuiderClearPass>(&setting, &vxgSetting), "GuiderClear Pass");
    addEdge("Prebake Pass", "AABBMin", "GuiderClear Pass", "AABBMinPrebake");
    addEdge("Prebake Pass", "AABBMax", "GuiderClear Pass", "AABBMaxPrebake");
    // Create VBuffer + GBuffer, as well as history GBuffer, A-SVGF buffers
    addPass(std::make_unique<Addon::VBuffer::VBuffer2GBufferPass>(), "VBuffer2GBuffer Pass");
    addEdge("VBuffer Pass", "VBuffer", "VBuffer2GBuffer Pass", "VBuffer");
    addPass(std::make_unique<Addon::GBufferHolderSource>(), "GBufferPrev Pass");
    addPass(std::make_unique<Addon::ASVGF::Prelude>(), "ASVGF-Prelude Pass");

    {
      // ASVGF :: Gradient Reprojection
      addPass(std::make_unique<Addon::ASVGF::GradientReprojection>(), "ASVGF-GradProj Pass");
      addEdge("VBuffer Pass", "VBuffer", "ASVGF-GradProj Pass", "VBuffer");
      addEdge("ASVGF-Prelude Pass", "GradSamplePosPrev", "ASVGF-GradProj Pass", "GradSamplePosPrev");
      addEdge("ASVGF-Prelude Pass", "HFPrev", "ASVGF-GradProj Pass", "HFPrev");
      addEdge("ASVGF-Prelude Pass", "SpecPrev", "ASVGF-GradProj Pass", "SpecPrev");
      addEdge("ASVGF-Prelude Pass", "VBufferPrev", "ASVGF-GradProj Pass", "VBufferPrev");
      addEdge("ASVGF-Prelude Pass", "RandPrev", "ASVGF-GradProj Pass", "RandPrev");
      Addon::GBufferUtils::addGBufferEdges(this, "VBuffer2GBuffer Pass", "ASVGF-GradProj Pass");
      Addon::GBufferUtils::addPrevGBufferEdges(this, "GBufferPrev Pass", "ASVGF-GradProj Pass");
    }

    //addPass(std::make_unique<Addon::VXGuiding::VXGuiderLightInjection>(&setting, &vxgSetting), "ImportonInjection Pass");
    addPass(std::make_unique<Addon::VXGuiding::VXGuider1stBounceInjection>(&setting, &vxgSetting), "ImportonInjection Pass");
    addEdge("VBuffer Pass", "VBuffer", "ImportonInjection Pass", "VBuffer");
    addEdge("GuiderClear Pass", "Irradiance", "ImportonInjection Pass", "Irradiance");
    addEdge("GuiderClear Pass", "VPLCount", "ImportonInjection Pass", "VPLCount");

    // geometry injection of dynamic objects
    addPass(std::make_unique<Addon::VXGuiding::VXGuiderGeometryPass>(&setting), "GuiderGeom Pass");
    addEdge("GuiderClear Pass", "AABBMin", "GuiderGeom Pass", "AABBMin");
    addEdge("GuiderClear Pass", "AABBMax", "GuiderGeom Pass", "AABBMax");
    addEdge("ImportonInjection Pass", "Irradiance", "GuiderGeom Pass", "Irradiance");

    addPass(std::make_unique<Addon::VXGuiding::VXGuiderCompactPass>(&setting), "VXGuiderCompact Pass");
    addEdge("GuiderClear Pass", "CounterBuffer", "VXGuiderCompact Pass", "CounterBuffer");
    addEdge("ImportonInjection Pass", "Irradiance", "VXGuiderCompact Pass", "Irradiance");

    // Superpixel clustering
    addPass(std::make_unique<Addon::gSLICr::InitClusterCenterPass>(&slicSetting), "InitClusterCenter Pass");
    addEdge("ImportonInjection Pass", "ShadingPoints", "InitClusterCenter Pass",  "Color");
    addPass(std::make_unique<Addon::gSLICr::FindCenterAssociationPass>(&slicSetting), "FindCenterAssociation Pass");
    addEdge("InitClusterCenter Pass", "Color", "FindCenterAssociation Pass", "Color");
    addEdge("InitClusterCenter Pass", "SPixelInfo", "FindCenterAssociation Pass", "SPixelInfo");
    addEdge("InitClusterCenter Pass", "IndexImage", "FindCenterAssociation Pass", "IndexImage");

    bool use_mcrs_cluster = true;
    // mrcs based clustering
    if (use_mcrs_cluster) {
      addPass(std::make_unique<Addon::VXGuiding::VXInfoClearPass>(&setting), "RCClear Pass");
      addEdge("GuiderGeom Pass", "AABBMin", "RCClear Pass", "AABBMin");
      addEdge("GuiderGeom Pass", "AABBMax", "RCClear Pass", "AABBMax");
      addEdge("VXGuiderCompact Pass", "CompactIndices", "RCClear Pass", "CompactIndices");
      addEdge("VXGuiderCompact Pass", "CounterBuffer", "RCClear Pass", "CounterBuffer");
      addEdge("ImportonInjection Pass", "Irradiance", "RCClear Pass", "Irradiance");
      addEdge("ImportonInjection Pass", "VPLCount", "RCClear Pass", "VPLCount");

      addPass(std::make_unique<Addon::VXGuiding::RowColumnPresamplePass>(&setting), "RCPresample Pass");
      addEdge("VBuffer Pass", "VBuffer", "RCPresample Pass", "VBuffer");
      addEdge("VXGuiderCompact Pass", "CounterBuffer", "RCPresample Pass", "VXCounter");
      addEdge("VXGuiderCompact Pass", "InverseIndex", "RCPresample Pass", "VXInverseIndex");
      addEdge("ImportonInjection Pass", "Positions", "RCPresample Pass", "VPLPositions");
      addEdge("RCClear Pass", "BoundMin", "RCPresample Pass", "BoundMin");
      addEdge("RCClear Pass", "BoundMax", "RCPresample Pass", "BoundMax");
      //addEdge("RCClear Pass", "DirectionalHash", "RCPresample Pass", "DirectionalHash");

      addPass(std::make_unique<Addon::VXGuiding::RowVisibilityPass>(), "RowVisibility Pass");
      addEdge("RCPresample Pass", "RepresentPixel", "RowVisibility Pass", "RepresentPixel");
      addEdge("RCPresample Pass", "RepresentVPL", "RowVisibility Pass", "RepresentVPL");
      addEdge("RCPresample Pass", "IndirectArgs", "RowVisibility Pass", "IndirectArgs");
    
      addPass(std::make_unique<Addon::VXGuiding::RowKmppCenterPass>(), "RowKmppCenter Pass");
      addEdge("RCPresample Pass", "IndirectArgs", "RowKmppCenter Pass", "IndirectArgs");
      addEdge("RowVisibility Pass", "RowVisibility", "RowKmppCenter Pass", "RowVisibility");
    
      addPass(std::make_unique<Addon::VXGuiding::RowFindCenterPass>(), "RowFindCenter Pass");
      addEdge("RCPresample Pass", "IndirectArgs", "RowFindCenter Pass", "IndirectArgs");
      addEdge("RowVisibility Pass", "RowVisibility", "RowFindCenter Pass", "RowVisibility");
      addEdge("RowKmppCenter Pass", "RowClusterInfo", "RowFindCenter Pass", "RowClusterInfo");
      addEdge("VXGuiderCompact Pass", "CompactIndices", "RowFindCenter Pass", "CompactIndices");
    }

    // BL - VXTree building
    {
      addPass(std::make_unique<Addon::VXGuiding::VXTreeEncodePass>(&setting), "TreeEncode Pass");
      addEdge("VXGuiderCompact Pass", "CounterBuffer", "TreeEncode Pass", "CounterBuffer");
      addEdge("VXGuiderCompact Pass", "CompactIndices", "TreeEncode Pass", "CompactIndices");
      if (use_mcrs_cluster) {
        addEdge("RowFindCenter Pass", "Clusters", "TreeEncode Pass", "VXClusterAssociation");
        addEdge("RCPresample Pass", "IndirectArgs", "TreeEncode Pass", "DispatchIndirectArgs");
      } else {
        addEdge("VXClusterFindAssociate2 Pass", "AssociateBuffer", "TreeEncode Pass", "VXClusterAssociation");
        addEdge("VXClusterInitCenter Pass", "DispatchIndirectArgs", "TreeEncode Pass", "DispatchIndirectArgs");
      }
      
      sort_setting.dispath = SIByL::Addon::BitonicSort::BitonicSortSetting::DispathType::DYNAMIC_INDIRECT;
      sort_setting.element_count = 65536;
      addSubgraph(std::make_unique<Addon::BitonicSort::BitonicSort>(&sort_setting),"Sort Pass");
      addEdge("VXGuiderCompact Pass", "CounterBuffer", "Sort Pass", "CounterBuffer");
      addEdge("TreeEncode Pass", "Code", "Sort Pass", "Input");
      
      addPass(std::make_unique<Addon::VXGuiding::VXTreeIIntializePass>(&setting),"TreeInitialize Pass");
      addEdge("Sort Pass", "Output", "TreeInitialize Pass", "Code");
      addEdge("VXGuiderCompact Pass", "CompactIndices", "TreeInitialize Pass", "CompactIndices");
      if (use_mcrs_cluster)  {
        addEdge("RowFindCenter Pass", "Clusters", "TreeInitialize Pass", "VXClusterAssociation");
        addEdge("RCClear Pass", "VXPremulIrradiance", "TreeInitialize Pass", "Irradiance");
      } else {
        addEdge("VXClusterFindAssociate2 Pass", "AssociateBuffer", "TreeInitialize Pass", "VXClusterAssociation");
        addEdge("VXClusterComputeInfo Pass", "VXPremulIrradiance", "TreeInitialize Pass", "Irradiance");
      }
      addEdge("TreeEncode Pass", "IndirectArgs", "TreeInitialize Pass", "IndirectArgs");
      
      addPass(std::make_unique<Addon::VXGuiding::VXTreeInternalPass>(), "TreeInternal Pass");
      addEdge("Sort Pass", "Output", "TreeInternal Pass", "Code");
      addEdge("TreeInitialize Pass", "Node", "TreeInternal Pass", "Node");
      addEdge("TreeEncode Pass", "IndirectArgs", "TreeInternal Pass", "IndirectArgs");
      addEdge("TreeInitialize Pass", "Descendant", "TreeInternal Pass", "Descendant");
      
      addPass(std::make_unique<Addon::VXGuiding::VXTreeMergePass>(), "TreeMerge Pass");
      addEdge("TreeInternal Pass", "Node", "TreeMerge Pass", "Node");
      addEdge("TreeInitialize Pass", "ClusterRoots", "TreeMerge Pass", "ClusterRoots");
      addEdge("TreeEncode Pass", "IndirectArgs", "TreeMerge Pass", "IndirectArgs");
    }
    
    // Rearrange vx info for further visibility check
    if (use_mcrs_cluster) {
      addPass(std::make_unique<Addon::VXGuiding::VXInfoRearrangePass>(), "VXInfoRearrange Pass");
      addEdge("GuiderGeom Pass", "AABBMin", "VXInfoRearrange Pass", "AABBMin");
      addEdge("GuiderGeom Pass", "AABBMax", "VXInfoRearrange Pass", "AABBMax");
      addEdge("RCPresample Pass", "BoundMin", "VXInfoRearrange Pass", "BoundMin");
      addEdge("RCPresample Pass", "BoundMax", "VXInfoRearrange Pass", "BoundMax");
      //addEdge("RCPresample Pass", "DirectionalHash", "VXInfoRearrange Pass", "DirectionalHash");
      addEdge("TreeInitialize Pass", "Compact2Leaf", "VXInfoRearrange Pass", "Compact2Leaf");
      addEdge("RCPresample Pass", "IndirectArgs", "VXInfoRearrange Pass", "IndirectArgs");
    }
    
    addPass(std::make_unique<Addon::VXGuiding::SPixelClearPass>(), "VisibilityClear Pass");
    
    addPass(std::make_unique<Addon::VXGuiding::SPixelGatherPass>(&setting), "VisibilityGather Pass");
    addEdge("VisibilityClear Pass", "SPixelVisibility", "VisibilityGather Pass", "SPixelVisibility");
    addEdge("VisibilityClear Pass", "SPixelCounter", "VisibilityGather Pass", "SPixelCounter");
    addEdge("VisibilityClear Pass", "ClusterCounter", "VisibilityGather Pass", "ClusterCounter");
    addEdge("FindCenterAssociation Pass", "IndexImage", "VisibilityGather Pass", "SPixelIndexImage");
    addEdge("ImportonInjection Pass", "Positions", "VisibilityGather Pass", "VPLPositions");
    addEdge("VXGuiderCompact Pass", "InverseIndex", "VisibilityGather Pass", "VXInverseIndex");
    if (use_mcrs_cluster) addEdge("RowFindCenter Pass", "Clusters", "VisibilityGather Pass", "VXClusterAssociation");
    else addEdge("VXClusterFindAssociate2 Pass", "AssociateBuffer", "VisibilityGather Pass", "VXClusterAssociation");

    addPass(std::make_unique<Addon::VXGuiding::SPixelVisibilityEXPass>(&setting), "VisibilityAdditional Pass");
    addEdge("VisibilityGather Pass", "SPixelGathered", "VisibilityAdditional Pass", "SPixelGathered");
    addEdge("VisibilityGather Pass", "ClusterGathered", "VisibilityAdditional Pass", "ClusterGathered");
    addEdge("VisibilityGather Pass", "SPixelCounter", "VisibilityAdditional Pass", "SPixelCounter");
    addEdge("VisibilityGather Pass", "ClusterCounter", "VisibilityAdditional Pass", "ClusterCounter");
    addEdge("VisibilityGather Pass", "SPixelVisibility", "VisibilityAdditional Pass", "SPixelVisibility");
    addEdge("VBuffer Pass", "VBuffer", "VisibilityAdditional Pass", "VBuffer");
    addEdge("VisibilityClear Pass", "SPixelAvgVisibility", "VisibilityAdditional Pass", "SPixelAvgVisibility");
    //if (use_mcrs_cluster) {
    //  addEdge("VXInfoRearrange Pass", "BoundMinOrdered", "VisibilityAdditional Pass", "BoundMinOrdered");
    //  addEdge("VXInfoRearrange Pass", "BoundMaxOrdered", "VisibilityAdditional Pass", "BoundMaxOrdered");
    //  addEdge("VXInfoRearrange Pass", "DirectionalHashOrdered", "VisibilityAdditional Pass", "DirectionalHashOrdered");
    //  addEdge("TreeInternal Pass", "Descendant", "VisibilityAdditional Pass", "Descendant");
    //  addEdge("TreeMerge Pass", "ClusterRoots", "VisibilityAdditional Pass", "ClusterRoots");
    //}

    addPass(std::make_unique<Addon::VXGuiding::VXTreeTopLevelPass>(), "TreeTopLevel Pass");
    addEdge("TreeMerge Pass", "Node", "TreeTopLevel Pass", "Node");
    addEdge("TreeMerge Pass", "ClusterRoots", "TreeTopLevel Pass", "ClusterRoots");
    addEdge("VisibilityAdditional Pass", "SPixelVisibility", "TreeTopLevel Pass", "SPixelVisibility");
    addEdge("VisibilityAdditional Pass", "SPixelAvgVisibility", "TreeTopLevel Pass", "SPixelAvgVisibility");

    addPass(std::make_unique<Addon::VXGuiding::VXGuiderGIASVGFPass>(&setting), "VXGuiderGI Pass");
    addEdge("VXGuiderCompact Pass", "CounterBuffer", "VXGuiderGI Pass", "CounterBuffer");
    addEdge("VXGuiderCompact Pass", "CompactIndices", "VXGuiderGI Pass", "CompactIndices");
    addEdge("VXInfoRearrange Pass", "AABBMin", "VXGuiderGI Pass", "AABBMin");
    addEdge("VXInfoRearrange Pass", "AABBMax", "VXGuiderGI Pass", "AABBMax");
    addEdge("ImportonInjection Pass", "Color", "VXGuiderGI Pass", "Color");
    addEdge("ImportonInjection Pass", "Positions", "VXGuiderGI Pass", "Positions");
    addEdge("ImportonInjection Pass", "Irradiance", "VXGuiderGI Pass", "Irradiance");
    addEdge("ImportonInjection Pass", "Intensity", "VXGuiderGI Pass", "Intensity");
    addEdge("VBuffer Pass", "VBuffer", "VXGuiderGI Pass", "VBuffer");
    addEdge("TreeMerge Pass", "Node", "VXGuiderGI Pass", "Node");
    addEdge("TreeMerge Pass", "ClusterRoots", "VXGuiderGI Pass", "ClusterRoots");
    addEdge("TreeTopLevel Pass", "TopLevelTree", "VXGuiderGI Pass", "TopLevelTree");
    addEdge("FindCenterAssociation Pass", "IndexImage", "VXGuiderGI Pass", "SPixelIndexImage");
    addEdge("FindCenterAssociation Pass", "FuzzyWeight", "VXGuiderGI Pass", "FuzzyWeight");
    addEdge("FindCenterAssociation Pass", "FuzzyIndex", "VXGuiderGI Pass", "FuzzyIndex");
    addEdge("VXGuiderCompact Pass", "InverseIndex", "VXGuiderGI Pass", "InverseIndex");
    addEdge("TreeInitialize Pass", "Compact2Leaf", "VXGuiderGI Pass", "Compact2Leaf");
    if (use_mcrs_cluster) addEdge("RowFindCenter Pass", "Clusters", "VXGuiderGI Pass", "AssociateBuffer");
    else addEdge("VXClusterFindAssociate2 Pass", "AssociateBuffer", "VXGuiderGI Pass", "AssociateBuffer");
    Addon::GBufferUtils::addGBufferEdges(this, "ASVGF-GradProj Pass", "VXGuiderGI Pass");
    addEdge("ASVGF-GradProj Pass", "RandSeed", "VXGuiderGI Pass", "RandSeed");
    addEdge("ASVGF-GradProj Pass", "RandPrev", "VXGuiderGI Pass", "RandPrev");

    {
      // Now blit HF, Spec, GradSamplePos and VBuffer img for next frame
      addPass(std::make_unique<BlitPass>(BlitPass::Descriptor{0, 0, 0, 0, BlitPass::SourceType::UINT}), "ASVGF Blit Diffuse");
      addEdge("VXGuiderGI Pass", "DiffuseBSDF", "ASVGF Blit Diffuse", "Source");
      addEdge("ASVGF-GradProj Pass", "HFPrev", "ASVGF Blit Diffuse", "Target");
      addPass(std::make_unique<BlitPass>(BlitPass::Descriptor{0, 0, 0, 0, BlitPass::SourceType::UINT}), "ASVGF Blit Specular");
      addEdge("VXGuiderGI Pass", "SpecularBSDF", "ASVGF Blit Specular", "Source");
      addEdge("ASVGF-GradProj Pass", "SpecPrev", "ASVGF Blit Specular", "Target");
      addPass(std::make_unique<BlitPass>(BlitPass::Descriptor{0, 0, 0, 0, BlitPass::SourceType::UINT}), "ASVGF Blit GradSamplePos");
      addEdge("ASVGF-GradProj Pass", "GradSamplePos", "ASVGF Blit GradSamplePos", "Source");
      addEdge("ASVGF-GradProj Pass", "GradSamplePosPrev", "ASVGF Blit GradSamplePos", "Target");
      addPass(std::make_unique<BlitPass>(BlitPass::Descriptor{0, 0, 0, 0, BlitPass::SourceType::UINT4}), "ASVGF Blit VBuffer");
      addEdge("VBuffer Pass", "VBuffer", "ASVGF Blit VBuffer", "Source");
      addEdge("ASVGF-GradProj Pass", "VBufferPrev", "ASVGF Blit VBuffer", "Target");
      
      // create gradient image
      addPass(std::make_unique<Addon::ASVGF::GradientImagePass>(), "ASVGF-GradImg Pass");
      addEdge("ASVGF-GradProj Pass", "GradSamplePos", "ASVGF-GradImg Pass", "GradSamplePos");
      addEdge("ASVGF-GradProj Pass", "HfSpecLumPrev", "ASVGF-GradImg Pass", "HfSpecLumPrev");
      addEdge("VXGuiderGI Pass", "DiffuseBSDF", "ASVGF-GradImg Pass", "HF");
      addEdge("VXGuiderGI Pass", "SpecularBSDF", "ASVGF-GradImg Pass", "Spec");
    
      // atrous gradient image
      addPass(std::make_unique<Addon::ASVGF::GradientAtrousPass>(0), "ASVGF-GradAtrous-0 Pass");
      addEdge("ASVGF-GradImg Pass", "GradHFSpec", "ASVGF-GradAtrous-0 Pass", "GradHFSpecPing");
      addEdge("ASVGF-GradImg Pass", "GradHFSpecBack", "ASVGF-GradAtrous-0 Pass", "GradHFSpecPong");
      addPass(std::make_unique<Addon::ASVGF::GradientAtrousPass>(1), "ASVGF-GradAtrous-1 Pass");
      addEdge("ASVGF-GradAtrous-0 Pass", "GradHFSpecPong", "ASVGF-GradAtrous-1 Pass", "GradHFSpecPing");
      addEdge("ASVGF-GradAtrous-0 Pass", "GradHFSpecPing", "ASVGF-GradAtrous-1 Pass", "GradHFSpecPong");
      addPass(std::make_unique<Addon::ASVGF::GradientAtrousPass>(2), "ASVGF-GradAtrous-2 Pass");
      addEdge("ASVGF-GradAtrous-1 Pass", "GradHFSpecPong", "ASVGF-GradAtrous-2 Pass", "GradHFSpecPing");
      addEdge("ASVGF-GradAtrous-1 Pass", "GradHFSpecPing", "ASVGF-GradAtrous-2 Pass", "GradHFSpecPong");
      
      // temporal accumulate pass
      addPass(std::make_unique<Addon::ASVGF::TemporalPass>(), "ASVGF-Temporal Pass");
      Addon::GBufferUtils::addGBufferEdges(this, "VXGuiderGI Pass", "ASVGF-Temporal Pass");
      Addon::GBufferUtils::addPrevGBufferEdges(this, "ASVGF-GradProj Pass", "ASVGF-Temporal Pass");
      addEdge("ASVGF-GradAtrous-2 Pass", "GradHFSpecPong", "ASVGF-Temporal Pass", "GradHF");
      addEdge("VXGuiderGI Pass", "Diffuse", "ASVGF-Temporal Pass", "HF");
      addEdge("VXGuiderGI Pass", "Specular", "ASVGF-Temporal Pass", "Spec");
      
      // blit pass - two moments and color history
      addPass(std::make_unique<BlitPass>(BlitPass::Descriptor{0, 0, 0, 0, BlitPass::SourceType::FLOAT4}), "Blit MomentsHistlenHF");
      addEdge("ASVGF-Temporal Pass", "MomentsHistlenHF", "Blit MomentsHistlenHF", "Source");
      addEdge("ASVGF-Temporal Pass", "MomentsHistlenHFPrev", "Blit MomentsHistlenHF", "Target");
      addPass(std::make_unique<BlitPass>(BlitPass::Descriptor{0, 0, 0, 0, BlitPass::SourceType::FLOAT4}), "Blit ColorHistlenSpec");
      addEdge("ASVGF-Temporal Pass", "ColorHistlenSpec", "Blit ColorHistlenSpec", "Source");
      addEdge("ASVGF-Temporal Pass", "ColorHistlenSpecPrev", "Blit ColorHistlenSpec", "Target");

      // 
      addPass(std::make_unique<Addon::ASVGF::AtrousPass>(0), "ASVGF-Atrous-0 Pass");
      Addon::GBufferUtils::addGBufferEdges(this, "ASVGF-Temporal Pass", "ASVGF-Atrous-0 Pass");
      addEdge("ASVGF-Temporal Pass", "Composite", "ASVGF-Atrous-0 Pass", "Composite");
      addEdge("ASVGF-Temporal Pass", "MomentsHistlenHF", "ASVGF-Atrous-0 Pass", "MomentsHistlenHF");
      addEdge("ASVGF-Temporal Pass", "AtrousHF", "ASVGF-Atrous-0 Pass", "AtrousHFPing");
      addEdge("ASVGF-Temporal Pass", "AtrousSpec", "ASVGF-Atrous-0 Pass", "AtrousSpecPing");
      addEdge("ASVGF-Temporal Pass", "AtrousMoments", "ASVGF-Atrous-0 Pass", "AtrousMomentPing");
      addEdge("ASVGF-Temporal Pass", "AtrousHFBack", "ASVGF-Atrous-0 Pass", "AtrousHFPong");
      addEdge("ASVGF-Temporal Pass", "AtrousSpecBack", "ASVGF-Atrous-0 Pass", "AtrousSpecPong");
      addEdge("ASVGF-Temporal Pass", "AtrousMomentsBack", "ASVGF-Atrous-0 Pass", "AtrousMomentPong");
      addEdge("ASVGF-GradProj Pass", "IsCorrelated", "ASVGF-Atrous-0 Pass", "IsCorrelated");
      
      addPass(std::make_unique<BlitPass>(BlitPass::Descriptor{0, 0, 0, 0, BlitPass::SourceType::UINT}), "Blit HFFiltered");
      addEdge("ASVGF-Atrous-0 Pass", "AtrousHFPong", "Blit HFFiltered", "Source");
      addEdge("ASVGF-Temporal Pass", "HFFilteredPrev", "Blit HFFiltered", "Target");
      
      addPass(std::make_unique<Addon::ASVGF::AtrousPass>(1), "ASVGF-Atrous-1 Pass");
      Addon::GBufferUtils::addGBufferEdges(this, "ASVGF-Atrous-0 Pass", "ASVGF-Atrous-1 Pass");
      addEdge("Blit HFFiltered", "Target", "ASVGF-Atrous-1 Pass", "AtrousHFPing");
      addEdge("ASVGF-Temporal Pass", "Composite", "ASVGF-Atrous-1 Pass", "Composite");
      addEdge("ASVGF-Temporal Pass", "MomentsHistlenHF", "ASVGF-Atrous-1 Pass", "MomentsHistlenHF");
      addEdge("ASVGF-Atrous-0 Pass", "AtrousSpecPong", "ASVGF-Atrous-1 Pass", "AtrousSpecPing");
      addEdge("ASVGF-Atrous-0 Pass", "AtrousMomentPong", "ASVGF-Atrous-1 Pass", "AtrousMomentPing");
      addEdge("ASVGF-Atrous-0 Pass", "AtrousHFPing", "ASVGF-Atrous-1 Pass", "AtrousHFPong");
      addEdge("ASVGF-Atrous-0 Pass", "AtrousSpecPing", "ASVGF-Atrous-1 Pass", "AtrousSpecPong");
      addEdge("ASVGF-Atrous-0 Pass", "AtrousMomentPing", "ASVGF-Atrous-1 Pass", "AtrousMomentPong");
      addEdge("ASVGF-Atrous-0 Pass", "IsCorrelated", "ASVGF-Atrous-1 Pass", "IsCorrelated");
      
      addPass(std::make_unique<Addon::ASVGF::AtrousPass>(2), "ASVGF-Atrous-2 Pass");
      Addon::GBufferUtils::addGBufferEdges(this, "ASVGF-Atrous-1 Pass", "ASVGF-Atrous-2 Pass");
      addEdge("ASVGF-Temporal Pass", "Composite", "ASVGF-Atrous-2 Pass", "Composite");
      addEdge("ASVGF-Temporal Pass", "MomentsHistlenHF", "ASVGF-Atrous-2 Pass", "MomentsHistlenHF");
      addEdge("ASVGF-Atrous-1 Pass", "AtrousHFPong", "ASVGF-Atrous-2 Pass", "AtrousHFPing");
      addEdge("ASVGF-Atrous-1 Pass", "AtrousSpecPong", "ASVGF-Atrous-2 Pass", "AtrousSpecPing");
      addEdge("ASVGF-Atrous-1 Pass", "AtrousMomentPong", "ASVGF-Atrous-2 Pass", "AtrousMomentPing");
      addEdge("ASVGF-Atrous-1 Pass", "AtrousHFPing", "ASVGF-Atrous-2 Pass", "AtrousHFPong");
      addEdge("ASVGF-Atrous-1 Pass", "AtrousSpecPing", "ASVGF-Atrous-2 Pass", "AtrousSpecPong");
      addEdge("ASVGF-Atrous-1 Pass", "AtrousMomentPing", "ASVGF-Atrous-2 Pass", "AtrousMomentPong");
      addEdge("ASVGF-Atrous-1 Pass", "IsCorrelated", "ASVGF-Atrous-2 Pass", "IsCorrelated");
      
      addPass(std::make_unique<Addon::ASVGF::AtrousPass>(3), "ASVGF-Atrous-3 Pass");
      Addon::GBufferUtils::addGBufferEdges(this, "ASVGF-Atrous-2 Pass", "ASVGF-Atrous-3 Pass");
      addEdge("ASVGF-Temporal Pass", "Composite", "ASVGF-Atrous-3 Pass", "Composite");
      addEdge("ASVGF-Temporal Pass", "MomentsHistlenHF", "ASVGF-Atrous-3 Pass", "MomentsHistlenHF");
      addEdge("ASVGF-Atrous-2 Pass", "AtrousHFPong", "ASVGF-Atrous-3 Pass", "AtrousHFPing");
      addEdge("ASVGF-Atrous-2 Pass", "AtrousSpecPong", "ASVGF-Atrous-3 Pass", "AtrousSpecPing");
      addEdge("ASVGF-Atrous-2 Pass", "AtrousMomentPong", "ASVGF-Atrous-3 Pass", "AtrousMomentPing");
      addEdge("ASVGF-Atrous-2 Pass", "AtrousHFPing", "ASVGF-Atrous-3 Pass", "AtrousHFPong");
      addEdge("ASVGF-Atrous-2 Pass", "AtrousSpecPing", "ASVGF-Atrous-3 Pass", "AtrousSpecPong");
      addEdge("ASVGF-Atrous-2 Pass", "AtrousMomentPing", "ASVGF-Atrous-3 Pass", "AtrousMomentPong");
      addEdge("ASVGF-Atrous-2 Pass", "IsCorrelated", "ASVGF-Atrous-3 Pass", "IsCorrelated");

    }

    // Blit history GBuffer
    addSubgraph(std::make_unique<Addon::GBufferHolderGraph>(), "GBuffer Blit Pass");
    Addon::GBufferUtils::addBlitPrevGBufferEdges(this, "VXGuiderGI Pass", "ASVGF-Temporal Pass", "GBuffer Blit Pass");
    
    addPass(std::make_unique<AccumulatePass>(), "Accum Pass");
    addEdge("ASVGF-Atrous-3 Pass", "Composite", "Accum Pass", "Input");
    
    addPass(std::make_unique<Addon::Postprocess::ToneMapperPass>(), "ToneMapper Pass");
    addEdge("Accum Pass", "Output", "ToneMapper Pass", "Input");

    markOutput("ToneMapper Pass", "Output");
  }

  Addon::BitonicSort::BitonicSortSetting sort_setting;
};

SE_EXPORT struct VXPGASVGFPipeline : public RDG::Pipeline {
  VXPGASVGFPipeline() {}
  enum struct Phase {
    GeometryBaking,
    Runtime,
  } curr_phase = Phase::GeometryBaking;

  virtual auto build() noexcept -> void {
    prebake_graph.build();
    GFX::Buffer* prebake_geom_min =
        prebake_graph.getBufferResource("Prebake Pass", "AABBMin");
    GFX::Buffer* prebake_geom_max =
        prebake_graph.getBufferResource("Prebake Pass", "AABBMax");
    runtime_graph.setExternal("Prebake Pass", "AABBMin", prebake_geom_min);
    runtime_graph.setExternal("Prebake Pass", "AABBMax", prebake_geom_max);
    runtime_graph.build();
  }

  virtual auto execute(RHI::CommandEncoder* encoder) noexcept -> void {
    auto graphs = getActiveGraphs();
    for (auto* graph : graphs) graph->execute(encoder);
    if (curr_phase == Phase::GeometryBaking) {
      curr_phase = Phase::Runtime;
    }
    if (runtime_graph.need_rebake) {
      curr_phase = Phase::GeometryBaking;
      runtime_graph.need_rebake = false;
    }
  }

  virtual auto getActiveGraphs() noexcept -> std::vector<RDG::Graph*> {
    if (curr_phase == Phase::GeometryBaking)
      return {&prebake_graph};
    else if (curr_phase == Phase::Runtime)
      return {&runtime_graph};
    else
      return {};
  }

  virtual auto getOutput() noexcept -> GFX::Texture* {
    return runtime_graph.getOutput();
  }

  VXGuidingPrebakeGraph prebake_graph;
  VXGuidingASVGFGraph runtime_graph;
};


SE_EXPORT struct VXGuidingGraph : public RDG::Graph {
  Addon::VXGI::VXGISetting setting;
  Addon::VXGuiding::DITestSetting diTestSetting;
  Addon::gSLICr::gSLICrSetting slicSetting;
  Addon::VXGuiding::VXGuidingSetting vxgSetting;
  bool need_rebake = false;

  VXGuidingGraph() {
    setting.clipmapSetting.mip = 6;
    slicSetting.img_size = {1280, 720};
    slicSetting.map_size = {(1280 + 31) / 32, (720 + 31) / 32};
    slicSetting.number_iter = 5;
    slicSetting.spixel_size = 32;
    vxgSetting.use_average_irradiance = true;

    // prepare stages
    addPass(std::make_unique<Addon::VBuffer::RayTraceVBuffer>(), "VBuffer Pass");
    addPass(std::make_unique<Addon::VXGuiding::PrebakeDummyPass>(need_rebake), "Prebake Pass");
    addPass(std::make_unique<Addon::VXGuiding::VXGuiderClearPass>(&setting, &vxgSetting), "GuiderClear Pass");
    addEdge("Prebake Pass", "AABBMin", "GuiderClear Pass", "AABBMinPrebake");
    addEdge("Prebake Pass", "AABBMax", "GuiderClear Pass", "AABBMaxPrebake");

    addPass(std::make_unique<Addon::VXGuiding::VXGuiderLightInjection>(&setting, &vxgSetting), "ImportonInjection Pass");
    //addPass(std::make_unique<Addon::VXGuiding::VXGuider1stBounceInjection>(&setting, &vxgSetting), "ImportonInjection Pass");
    addEdge("VBuffer Pass", "VBuffer", "ImportonInjection Pass", "VBuffer");
    addEdge("GuiderClear Pass", "Irradiance", "ImportonInjection Pass", "Irradiance");
    addEdge("GuiderClear Pass", "VPLCount", "ImportonInjection Pass", "VPLCount");

    // geometry injection of dynamic objects
    addPass(std::make_unique<Addon::VXGuiding::VXGuiderGeometryPass>(&setting), "GuiderGeom Pass");
    addEdge("GuiderClear Pass", "AABBMin", "GuiderGeom Pass", "AABBMin");
    addEdge("GuiderClear Pass", "AABBMax", "GuiderGeom Pass", "AABBMax");
    addEdge("ImportonInjection Pass", "Irradiance", "GuiderGeom Pass", "Irradiance");

    addPass(std::make_unique<Addon::VXGuiding::VXGuiderCompactPass>(&setting), "VXGuiderCompact Pass");
    addEdge("GuiderClear Pass", "CounterBuffer", "VXGuiderCompact Pass", "CounterBuffer");
    addEdge("ImportonInjection Pass", "Irradiance", "VXGuiderCompact Pass", "Irradiance");

    // Superpixel clustering
    addPass(std::make_unique<Addon::gSLICr::InitClusterCenterPass>(&slicSetting), "InitClusterCenter Pass");
    addEdge("ImportonInjection Pass", "ShadingPoints", "InitClusterCenter Pass",  "Color");
    addPass(std::make_unique<Addon::gSLICr::FindCenterAssociationPass>(&slicSetting), "FindCenterAssociation Pass");
    addEdge("InitClusterCenter Pass", "Color", "FindCenterAssociation Pass", "Color");
    addEdge("InitClusterCenter Pass", "SPixelInfo", "FindCenterAssociation Pass", "SPixelInfo");
    addEdge("InitClusterCenter Pass", "IndexImage", "FindCenterAssociation Pass", "IndexImage");

    bool use_mcrs_cluster = true;
    // geometry based clustering
    if (!use_mcrs_cluster) {
      addPass(std::make_unique<Addon::VXGuiding::VXClusterComputeInfoPass>(&setting), "VXClusterComputeInfo Pass");
      addEdge("GuiderGeom Pass", "AABBMin", "VXClusterComputeInfo Pass", "AABBMin");
      addEdge("GuiderGeom Pass", "AABBMax", "VXClusterComputeInfo Pass", "AABBMax");
      addEdge("VXGuiderCompact Pass", "CompactIndices", "VXClusterComputeInfo Pass", "CompactIndices");
      addEdge("ImportonInjection Pass", "Irradiance", "VXClusterComputeInfo Pass", "Irradiance");
      addEdge("VXGuiderCompact Pass", "CounterBuffer", "VXClusterComputeInfo Pass", "CounterBuffer");

      addPass(std::make_unique<Addon::VXGuiding::VXClusterSeedingPass>(), "VXGuiderSeed Pass");
      addEdge("VXGuiderCompact Pass", "CompactIndices", "VXGuiderSeed Pass", "CompactIndices");
      addEdge("VXGuiderCompact Pass", "CounterBuffer", "VXGuiderSeed Pass", "CounterBuffer");
      addEdge("VXClusterComputeInfo Pass", "VXNormal", "VXGuiderSeed Pass", "VXNormal");

      addPass(std::make_unique<Addon::VXGuiding::VXClusterInitCenterPass>(&setting), "VXClusterInitCenter Pass");
      addEdge("VXGuiderCompact Pass", "CompactIndices", "VXClusterInitCenter Pass", "CompactIndices");
      addEdge("VXGuiderCompact Pass", "CounterBuffer", "VXClusterInitCenter Pass", "CounterBuffer");
      addEdge("VXClusterComputeInfo Pass", "VXNormal", "VXClusterInitCenter Pass", "VXNormal");
      addEdge("VXGuiderSeed Pass", "ClusterSeeds", "VXClusterInitCenter Pass", "ClusterSeeds");

      addPass(std::make_unique<Addon::VXGuiding::VXClusterFindAssociatePass>(&setting), "VXClusterFindAssociate Pass");
      addEdge("VXGuiderCompact Pass", "CounterBuffer", "VXClusterFindAssociate Pass", "CounterBuffer");
      addEdge("VXClusterInitCenter Pass", "GridClusterCount", "VXClusterFindAssociate Pass", "GridClusterCount");
      addEdge("VXClusterInitCenter Pass", "GridClusterIndices", "VXClusterFindAssociate Pass", "GridClusterIndices");
      addEdge("VXClusterComputeInfo Pass", "AssociateBuffer", "VXClusterFindAssociate Pass", "AssociateBuffer");
      addEdge("VXGuiderCompact Pass", "CompactIndices", "VXClusterFindAssociate Pass", "CompactIndices");
      addEdge("VXClusterComputeInfo Pass", "VXNormal", "VXClusterFindAssociate Pass", "VXNormal");
      addEdge("VXClusterInitCenter Pass", "SVXInfo", "VXClusterFindAssociate Pass", "SVXInfo");
      addEdge("VXClusterInitCenter Pass", "SVXAccumInfo", "VXClusterFindAssociate Pass", "SVXAccumInfo");
      addEdge("VXClusterInitCenter Pass", "DispatchIndirectArgs", "VXClusterFindAssociate Pass", "DispatchIndirectArgs");

      addPass(std::make_unique<Addon::VXGuiding::VXClusterUpdateCenterPass>(&setting), "VXClusterUpdate Pass");
      addEdge("VXClusterInitCenter Pass", "SVXInfo", "VXClusterUpdate Pass", "SVXInfo");
      addEdge("VXClusterFindAssociate Pass", "SVXAccumInfo", "VXClusterUpdate Pass", "SVXAccumInfo");
      addEdge("VXClusterInitCenter Pass", "GridClusterCount", "VXClusterUpdate Pass", "GridClusterCount");
      addEdge("VXClusterInitCenter Pass", "GridClusterIndices", "VXClusterUpdate Pass", "GridClusterIndices");

      addPass(std::make_unique<Addon::VXGuiding::VXClusterFindAssociatePass>(&setting), "VXClusterFindAssociate2 Pass");
      addEdge("VXGuiderCompact Pass", "CounterBuffer", "VXClusterFindAssociate2 Pass", "CounterBuffer");
      addEdge("VXClusterUpdate Pass", "GridClusterCount", "VXClusterFindAssociate2 Pass", "GridClusterCount");
      addEdge("VXClusterUpdate Pass", "GridClusterIndices", "VXClusterFindAssociate2 Pass", "GridClusterIndices");
      addEdge("VXClusterFindAssociate Pass", "AssociateBuffer", "VXClusterFindAssociate2 Pass", "AssociateBuffer");
      addEdge("VXGuiderCompact Pass", "CompactIndices", "VXClusterFindAssociate2 Pass", "CompactIndices");
      addEdge("VXClusterComputeInfo Pass", "VXNormal", "VXClusterFindAssociate2 Pass", "VXNormal");
      addEdge("VXClusterUpdate Pass", "SVXInfo", "VXClusterFindAssociate2 Pass", "SVXInfo");
      addEdge("VXClusterUpdate Pass", "SVXAccumInfo", "VXClusterFindAssociate2 Pass", "SVXAccumInfo");
      addEdge("VXClusterInitCenter Pass", "DispatchIndirectArgs", "VXClusterFindAssociate2 Pass", "DispatchIndirectArgs"); 
    }
    // mrcs based clustering
    if (use_mcrs_cluster) {
      addPass(std::make_unique<Addon::VXGuiding::VXInfoClearPass>(&setting), "RCClear Pass");
      addEdge("GuiderGeom Pass", "AABBMin", "RCClear Pass", "AABBMin");
      addEdge("GuiderGeom Pass", "AABBMax", "RCClear Pass", "AABBMax");
      addEdge("VXGuiderCompact Pass", "CompactIndices", "RCClear Pass", "CompactIndices");
      addEdge("VXGuiderCompact Pass", "CounterBuffer", "RCClear Pass", "CounterBuffer");
      addEdge("ImportonInjection Pass", "Irradiance", "RCClear Pass", "Irradiance");
      addEdge("ImportonInjection Pass", "VPLCount", "RCClear Pass", "VPLCount");

      addPass(std::make_unique<Addon::VXGuiding::RowColumnPresamplePass>(&setting), "RCPresample Pass");
      addEdge("VBuffer Pass", "VBuffer", "RCPresample Pass", "VBuffer");
      addEdge("VXGuiderCompact Pass", "CounterBuffer", "RCPresample Pass", "VXCounter");
      addEdge("VXGuiderCompact Pass", "InverseIndex", "RCPresample Pass", "VXInverseIndex");
      addEdge("ImportonInjection Pass", "Positions", "RCPresample Pass", "VPLPositions");
      addEdge("RCClear Pass", "BoundMin", "RCPresample Pass", "BoundMin");
      addEdge("RCClear Pass", "BoundMax", "RCPresample Pass", "BoundMax");
      //addEdge("RCClear Pass", "DirectionalHash", "RCPresample Pass", "DirectionalHash");

      addPass(std::make_unique<Addon::VXGuiding::RowVisibilityPass>(), "RowVisibility Pass");
      addEdge("RCPresample Pass", "RepresentPixel", "RowVisibility Pass", "RepresentPixel");
      addEdge("RCPresample Pass", "RepresentVPL", "RowVisibility Pass", "RepresentVPL");
      addEdge("RCPresample Pass", "IndirectArgs", "RowVisibility Pass", "IndirectArgs");
    
      addPass(std::make_unique<Addon::VXGuiding::RowKmppCenterPass>(), "RowKmppCenter Pass");
      addEdge("RCPresample Pass", "IndirectArgs", "RowKmppCenter Pass", "IndirectArgs");
      addEdge("RowVisibility Pass", "RowVisibility", "RowKmppCenter Pass", "RowVisibility");
    
      addPass(std::make_unique<Addon::VXGuiding::RowFindCenterPass>(), "RowFindCenter Pass");
      addEdge("RCPresample Pass", "IndirectArgs", "RowFindCenter Pass", "IndirectArgs");
      addEdge("RowVisibility Pass", "RowVisibility", "RowFindCenter Pass", "RowVisibility");
      addEdge("RowKmppCenter Pass", "RowClusterInfo", "RowFindCenter Pass", "RowClusterInfo");
      addEdge("VXGuiderCompact Pass", "CompactIndices", "RowFindCenter Pass", "CompactIndices");
    }

    // BL - VXTree building
    {
      addPass(std::make_unique<Addon::VXGuiding::VXTreeEncodePass>(&setting), "TreeEncode Pass");
      addEdge("VXGuiderCompact Pass", "CounterBuffer", "TreeEncode Pass", "CounterBuffer");
      addEdge("VXGuiderCompact Pass", "CompactIndices", "TreeEncode Pass", "CompactIndices");
      if (use_mcrs_cluster) {
        addEdge("RowFindCenter Pass", "Clusters", "TreeEncode Pass", "VXClusterAssociation");
        addEdge("RCPresample Pass", "IndirectArgs", "TreeEncode Pass", "DispatchIndirectArgs");
      } else {
        addEdge("VXClusterFindAssociate2 Pass", "AssociateBuffer", "TreeEncode Pass", "VXClusterAssociation");
        addEdge("VXClusterInitCenter Pass", "DispatchIndirectArgs", "TreeEncode Pass", "DispatchIndirectArgs");
      }
      
      sort_setting.dispath = SIByL::Addon::BitonicSort::BitonicSortSetting::DispathType::DYNAMIC_INDIRECT;
      sort_setting.element_count = 65536;
      addSubgraph(std::make_unique<Addon::BitonicSort::BitonicSort>(&sort_setting),"Sort Pass");
      addEdge("VXGuiderCompact Pass", "CounterBuffer", "Sort Pass", "CounterBuffer");
      addEdge("TreeEncode Pass", "Code", "Sort Pass", "Input");
      
      addPass(std::make_unique<Addon::VXGuiding::VXTreeIIntializePass>(&setting),"TreeInitialize Pass");
      addEdge("Sort Pass", "Output", "TreeInitialize Pass", "Code");
      addEdge("VXGuiderCompact Pass", "CompactIndices", "TreeInitialize Pass", "CompactIndices");
      if (use_mcrs_cluster)  {
        addEdge("RowFindCenter Pass", "Clusters", "TreeInitialize Pass", "VXClusterAssociation");
        addEdge("RCClear Pass", "VXPremulIrradiance", "TreeInitialize Pass", "Irradiance");
      } else {
        addEdge("VXClusterFindAssociate2 Pass", "AssociateBuffer", "TreeInitialize Pass", "VXClusterAssociation");
        addEdge("VXClusterComputeInfo Pass", "VXPremulIrradiance", "TreeInitialize Pass", "Irradiance");
      }
      addEdge("TreeEncode Pass", "IndirectArgs", "TreeInitialize Pass", "IndirectArgs");
      
      addPass(std::make_unique<Addon::VXGuiding::VXTreeInternalPass>(), "TreeInternal Pass");
      addEdge("Sort Pass", "Output", "TreeInternal Pass", "Code");
      addEdge("TreeInitialize Pass", "Node", "TreeInternal Pass", "Node");
      addEdge("TreeEncode Pass", "IndirectArgs", "TreeInternal Pass", "IndirectArgs");
      addEdge("TreeInitialize Pass", "Descendant", "TreeInternal Pass", "Descendant");
      
      addPass(std::make_unique<Addon::VXGuiding::VXTreeMergePass>(), "TreeMerge Pass");
      addEdge("TreeInternal Pass", "Node", "TreeMerge Pass", "Node");
      addEdge("TreeInitialize Pass", "ClusterRoots", "TreeMerge Pass", "ClusterRoots");
      addEdge("TreeEncode Pass", "IndirectArgs", "TreeMerge Pass", "IndirectArgs");
    }
    
    // Rearrange vx info for further visibility check
    if (use_mcrs_cluster) {
      addPass(std::make_unique<Addon::VXGuiding::VXInfoRearrangePass>(), "VXInfoRearrange Pass");
      addEdge("GuiderGeom Pass", "AABBMin", "VXInfoRearrange Pass", "AABBMin");
      addEdge("GuiderGeom Pass", "AABBMax", "VXInfoRearrange Pass", "AABBMax");
      addEdge("RCPresample Pass", "BoundMin", "VXInfoRearrange Pass", "BoundMin");
      addEdge("RCPresample Pass", "BoundMax", "VXInfoRearrange Pass", "BoundMax");
      //addEdge("RCPresample Pass", "DirectionalHash", "VXInfoRearrange Pass", "DirectionalHash");
      addEdge("TreeInitialize Pass", "Compact2Leaf", "VXInfoRearrange Pass", "Compact2Leaf");
      addEdge("RCPresample Pass", "IndirectArgs", "VXInfoRearrange Pass", "IndirectArgs");
    }
    
    addPass(std::make_unique<Addon::VXGuiding::SPixelClearPass>(), "VisibilityClear Pass");
    
    addPass(std::make_unique<Addon::VXGuiding::SPixelGatherPass>(&setting), "VisibilityGather Pass");
    addEdge("VisibilityClear Pass", "SPixelVisibility", "VisibilityGather Pass", "SPixelVisibility");
    addEdge("VisibilityClear Pass", "SPixelCounter", "VisibilityGather Pass", "SPixelCounter");
    addEdge("VisibilityClear Pass", "ClusterCounter", "VisibilityGather Pass", "ClusterCounter");
    addEdge("FindCenterAssociation Pass", "IndexImage", "VisibilityGather Pass", "SPixelIndexImage");
    addEdge("ImportonInjection Pass", "Positions", "VisibilityGather Pass", "VPLPositions");
    addEdge("VXGuiderCompact Pass", "InverseIndex", "VisibilityGather Pass", "VXInverseIndex");
    if (use_mcrs_cluster) addEdge("RowFindCenter Pass", "Clusters", "VisibilityGather Pass", "VXClusterAssociation");
    else addEdge("VXClusterFindAssociate2 Pass", "AssociateBuffer", "VisibilityGather Pass", "VXClusterAssociation");

    addPass(std::make_unique<Addon::VXGuiding::SPixelVisibilityEXPass>(&setting), "VisibilityAdditional Pass");
    addEdge("VisibilityGather Pass", "SPixelGathered", "VisibilityAdditional Pass", "SPixelGathered");
    addEdge("VisibilityGather Pass", "ClusterGathered", "VisibilityAdditional Pass", "ClusterGathered");
    addEdge("VisibilityGather Pass", "SPixelCounter", "VisibilityAdditional Pass", "SPixelCounter");
    addEdge("VisibilityGather Pass", "ClusterCounter", "VisibilityAdditional Pass", "ClusterCounter");
    addEdge("VisibilityGather Pass", "SPixelVisibility", "VisibilityAdditional Pass", "SPixelVisibility");
    addEdge("VBuffer Pass", "VBuffer", "VisibilityAdditional Pass", "VBuffer");
    addEdge("VisibilityClear Pass", "SPixelAvgVisibility", "VisibilityAdditional Pass", "SPixelAvgVisibility");
    //if (use_mcrs_cluster) {
    //  addEdge("VXInfoRearrange Pass", "BoundMinOrdered", "VisibilityAdditional Pass", "BoundMinOrdered");
    //  addEdge("VXInfoRearrange Pass", "BoundMaxOrdered", "VisibilityAdditional Pass", "BoundMaxOrdered");
    //  addEdge("VXInfoRearrange Pass", "DirectionalHashOrdered", "VisibilityAdditional Pass", "DirectionalHashOrdered");
    //  addEdge("TreeInternal Pass", "Descendant", "VisibilityAdditional Pass", "Descendant");
    //  addEdge("TreeMerge Pass", "ClusterRoots", "VisibilityAdditional Pass", "ClusterRoots");
    //}

    addPass(std::make_unique<Addon::VXGuiding::VXTreeTopLevelPass>(), "TreeTopLevel Pass");
    addEdge("TreeMerge Pass", "Node", "TreeTopLevel Pass", "Node");
    addEdge("TreeMerge Pass", "ClusterRoots", "TreeTopLevel Pass", "ClusterRoots");
    addEdge("VisibilityAdditional Pass", "SPixelVisibility", "TreeTopLevel Pass", "SPixelVisibility");
    addEdge("VisibilityAdditional Pass", "SPixelAvgVisibility", "TreeTopLevel Pass", "SPixelAvgVisibility");

    // visualize the bounding voxels by ray marching
    bool useGuiderViewer = false;
    if (useGuiderViewer) {
      addPass(std::make_unique<Addon::VXGuiding::VXGuiderViewPass>(&setting), "GuiderViewer Pass");
      addEdge("ImportonInjection Pass", "Irradiance", "GuiderViewer Pass", "Irradiance");
      addEdge("VXGuiderCompact Pass", "InverseIndex", "GuiderViewer Pass", "InverseIndex");
      addEdge("VisibilityAdditional Pass", "SPixelVisibility", "GuiderViewer Pass", "SPixelVisibility");
      addEdge("FindCenterAssociation Pass", "IndexImage", "GuiderViewer Pass", "SPixelIndexImage");
      if (use_mcrs_cluster) {
        addEdge("RowFindCenter Pass", "Clusters", "GuiderViewer Pass", "AssociateBuffer");
        addEdge("RCClear Pass", "BoundMin", "GuiderViewer Pass", "VXNormal");
        addEdge("RCClear Pass", "VXPremulIrradiance", "GuiderViewer Pass", "VXPremulIrradiance");
      } else {
        addEdge("VXClusterFindAssociate2 Pass", "AssociateBuffer", "GuiderViewer Pass", "AssociateBuffer");
        addEdge("VXClusterComputeInfo Pass", "VXNormal", "GuiderViewer Pass", "VXNormal");
        addEdge("VXClusterComputeInfo Pass", "VXPremulIrradiance", "GuiderViewer Pass", "VXPremulIrradiance");
      }
    }
    
    addPass(std::make_unique<Addon::VXGuiding::VXGuiderGIPass>(&setting), "VXGuiderGI Pass");
    addEdge("VXGuiderCompact Pass", "CounterBuffer", "VXGuiderGI Pass", "CounterBuffer");
    addEdge("VXGuiderCompact Pass", "CompactIndices", "VXGuiderGI Pass", "CompactIndices");
    addEdge("VXInfoRearrange Pass", "AABBMin", "VXGuiderGI Pass", "AABBMin");
    addEdge("VXInfoRearrange Pass", "AABBMax", "VXGuiderGI Pass", "AABBMax");
    addEdge("ImportonInjection Pass", "Color", "VXGuiderGI Pass", "Color");
    addEdge("ImportonInjection Pass", "Positions", "VXGuiderGI Pass", "Positions");
    addEdge("ImportonInjection Pass", "Irradiance", "VXGuiderGI Pass", "Irradiance");
    addEdge("ImportonInjection Pass", "Intensity", "VXGuiderGI Pass", "Intensity");
    addEdge("VBuffer Pass", "VBuffer", "VXGuiderGI Pass", "VBuffer");
    addEdge("TreeMerge Pass", "Node", "VXGuiderGI Pass", "Node");
    addEdge("TreeMerge Pass", "ClusterRoots", "VXGuiderGI Pass", "ClusterRoots");
    addEdge("TreeTopLevel Pass", "TopLevelTree", "VXGuiderGI Pass", "TopLevelTree");
    addEdge("FindCenterAssociation Pass", "IndexImage", "VXGuiderGI Pass", "SPixelIndexImage");
    addEdge("FindCenterAssociation Pass", "FuzzyWeight", "VXGuiderGI Pass", "FuzzyWeight");
    addEdge("FindCenterAssociation Pass", "FuzzyIndex", "VXGuiderGI Pass", "FuzzyIndex");
    addEdge("VXGuiderCompact Pass", "InverseIndex", "VXGuiderGI Pass", "InverseIndex");
    addEdge("TreeInitialize Pass", "Compact2Leaf", "VXGuiderGI Pass", "Compact2Leaf");
    if (use_mcrs_cluster) addEdge("RowFindCenter Pass", "Clusters", "VXGuiderGI Pass", "AssociateBuffer");
    else addEdge("VXClusterFindAssociate2 Pass", "AssociateBuffer", "VXGuiderGI Pass", "AssociateBuffer");

    // visualize the VXPG distribution via accumulation
    bool accumPdf = false;
    if (accumPdf) {
      addPass(std::make_unique<Addon::SSGuiding::PdfAccum_ClearPass>(), "PdfAccumClear Pass");
      addPass(std::make_unique<Addon::VXGuiding::VXGuiderAccumPass>(&setting), "VXGuiderGIAccum Pass");
      addEdge("PdfAccumClear Pass", "PdfAccumulator", "VXGuiderGIAccum Pass", "PdfAccumulator");
      addEdge("PdfAccumClear Pass", "PdfAccumulatorInfo", "VXGuiderGIAccum Pass", "PdfAccumulatorInfo");
      addEdge("VXGuiderCompact Pass", "CounterBuffer", "VXGuiderGIAccum Pass", "CounterBuffer");
      addEdge("VXGuiderCompact Pass", "CompactIndices", "VXGuiderGIAccum Pass", "CompactIndices");
      addEdge("VXInfoRearrange Pass", "AABBMin", "VXGuiderGIAccum Pass", "AABBMin");
      addEdge("VXInfoRearrange Pass", "AABBMax", "VXGuiderGIAccum Pass", "AABBMax");
      addEdge("ImportonInjection Pass", "Color", "VXGuiderGIAccum Pass", "Color");
      addEdge("ImportonInjection Pass", "Positions", "VXGuiderGIAccum Pass", "Positions");
      addEdge("ImportonInjection Pass", "Irradiance", "VXGuiderGIAccum Pass", "Irradiance");
      addEdge("VBuffer Pass", "VBuffer", "VXGuiderGIAccum Pass", "VBuffer");
      addEdge("TreeMerge Pass", "Node", "VXGuiderGIAccum Pass", "Node");
      addEdge("TreeMerge Pass", "ClusterRoots", "VXGuiderGIAccum Pass", "ClusterRoots");
      addEdge("TreeTopLevel Pass", "TopLevelTree", "VXGuiderGIAccum Pass", "TopLevelTree");
      addEdge("FindCenterAssociation Pass", "IndexImage", "VXGuiderGIAccum Pass", "SPixelIndexImage");
      addEdge("FindCenterAssociation Pass", "FuzzyWeight", "VXGuiderGIAccum Pass", "FuzzyWeight");
      addEdge("FindCenterAssociation Pass", "FuzzyIndex", "VXGuiderGIAccum Pass", "FuzzyIndex");
      addEdge("VXGuiderCompact Pass", "InverseIndex", "VXGuiderGIAccum Pass", "InverseIndex");
      addEdge("TreeInitialize Pass", "Compact2Leaf", "VXGuiderGIAccum Pass", "Compact2Leaf");
      if (use_mcrs_cluster) addEdge("RowFindCenter Pass", "Clusters", "VXGuiderGIAccum Pass", "AssociateBuffer");
      else addEdge("VXClusterFindAssociate2 Pass", "AssociateBuffer", "VXGuiderGIAccum Pass", "AssociateBuffer");

      addPass(std::make_unique<Addon::SSGuiding::PdfAccum_ViewerPass>(), "PdfAccumViewer Pass");
      addEdge("VXGuiderGIAccum Pass", "PdfAccumulator", "PdfAccumViewer Pass", "PdfAccumulator");
      addEdge("VXGuiderGIAccum Pass", "PdfAccumulatorInfo", "PdfAccumViewer Pass", "PdfAccumulatorInfo");
    }

    // visualize the vpls in primary sampling
    bool viewPrimaryVPLs = false;
    if (viewPrimaryVPLs) {
       addPass(std::make_unique<Addon::VXGuiding::VPLVisualizePass>(), "VPLVisualize Pass"); 
       addEdge("VisibilityGather Pass", "ClusterGathered", "VPLVisualize Pass", "ClusterGathered");
       addEdge("VisibilityGather Pass", "ClusterCounter", "VPLVisualize Pass", "ClusterCounter");
    }

    addPass(std::make_unique<AccumulatePass>(), "Accum Pass");
    addEdge("VXGuiderGI Pass", "Color", "Accum Pass", "Input");
    
    addPass(std::make_unique<Addon::Postprocess::ToneMapperPass>(), "ToneMapper Pass");
    addEdge("Accum Pass", "Output", "ToneMapper Pass", "Input");

    bool visualizeVXs = false;
    if (visualizeVXs) {
       addSubgraph(std::make_unique<PreZPass>(), "Pre-Z Pass");
       // addPass(std::make_unique<GeometryInspectorPass>(), "GeoInspect Pass");
       // addEdge("Pre-Z Pass", "Depth", "GeoInspect Pass", "Depth");
       addPass(std::make_unique<Addon::VXGuiding::VXGuiderVisualizePass>(&setting), "VXGuiderVIS Pass");
       addEdge("Pre-Z Pass", "Depth", "VXGuiderVIS Pass", "Depth");
       addEdge("Accum Pass", "Output", "VXGuiderVIS Pass", "Color");
       addEdge("GuiderGeom Pass", "AABBMin", "VXGuiderVIS Pass", "AABBMin");
       addEdge("GuiderGeom Pass", "AABBMax", "VXGuiderVIS Pass", "AABBMax");
       addEdge("ImportonInjection Pass", "Irradiance", "VXGuiderVIS Pass", "Irradiance");
    }

    bool visualizeSPixels = false;
    if (visualizeSPixels) {
      addPass(std::make_unique<Addon::gSLICr::VisualizeSPixelPass>(slicSetting), "VisualizeSPixel Pass");
      addEdge("FindCenterAssociation Pass", "IndexImage", "VisualizeSPixel Pass", "IndexImage");
      addEdge("FindCenterAssociation Pass", "FuzzyWeight", "VisualizeSPixel Pass", "FuzzyWeight");
      addEdge("FindCenterAssociation Pass", "FuzzyIndex", "VisualizeSPixel Pass", "FuzzyIndex");
      addEdge("VXGuiderVIS Pass", "Color", "VisualizeSPixel Pass", "Color");
      markOutput("VisualizeSPixel Pass", "Color");
    } else {
      markOutput("ToneMapper Pass", "Output");
    }
  }

  Addon::BitonicSort::BitonicSortSetting sort_setting;
};

SE_EXPORT struct VXGuidingPipeline : public RDG::SingleGraphPipeline {
  VXGuidingPipeline() { pGraph = &graph; }
  VXGuidingGraph graph;
};

SE_EXPORT struct VXPGPipeline : public RDG::Pipeline {
  VXPGPipeline() {}
  enum struct Phase {
      GeometryBaking,
      Runtime,
  } curr_phase = Phase::GeometryBaking;

  virtual auto build() noexcept -> void {
    prebake_graph.build();
    GFX::Buffer* prebake_geom_min = prebake_graph.getBufferResource("Prebake Pass", "AABBMin");
    GFX::Buffer* prebake_geom_max = prebake_graph.getBufferResource("Prebake Pass", "AABBMax");
    runtime_graph.setExternal("Prebake Pass", "AABBMin", prebake_geom_min);
    runtime_graph.setExternal("Prebake Pass", "AABBMax", prebake_geom_max);
    runtime_graph.build();
  }

  virtual auto execute(RHI::CommandEncoder* encoder) noexcept -> void {
    auto graphs = getActiveGraphs();
    for (auto* graph : graphs) graph->execute(encoder);
    if (curr_phase == Phase::GeometryBaking) {
      curr_phase = Phase::Runtime;
    }
    if (runtime_graph.need_rebake) {
      curr_phase = Phase::GeometryBaking;
      runtime_graph.need_rebake = false;
    }
  }

  virtual auto getActiveGraphs() noexcept -> std::vector<RDG::Graph*> {
    if (curr_phase == Phase::GeometryBaking) return { &prebake_graph };
    else if (curr_phase == Phase::Runtime) return { &runtime_graph };
    else return {};
  }

  virtual auto getOutput() noexcept -> GFX::Texture* {
    return runtime_graph.getOutput();
  }

  VXGuidingPrebakeGraph prebake_graph;
  VXGuidingGraph runtime_graph;
};

SE_EXPORT struct VXPGRestirGIGraph : public RDG::Graph {
  VXPGRestirGIGraph() {
    restirgi_param = Addon::RestirGI::InitializeParameters(1280, 720);
    setting.clipmapSetting.mip = 6;
    slicSetting.img_size = {1280, 720};
    slicSetting.map_size = {(1280 + 31) / 32, (720 + 31) / 32};
    slicSetting.number_iter = 5;
    slicSetting.spixel_size = 32;
    vxgSetting.use_average_irradiance = true;
    
    // trace v-buffer
    addPass(std::make_unique<Addon::VBuffer::RayTraceVBuffer>(), "VBuffer Pass");
    addPass(std::make_unique<Addon::VBuffer::VBuffer2GBufferPass>(), "VBuffer2GBuffer Pass");
    addEdge("VBuffer Pass", "VBuffer", "VBuffer2GBuffer Pass", "VBuffer");
    // Hold history GBuffer
    addPass(std::make_unique<Addon::GBufferHolderSource>(), "GBufferPrev Pass");
    
    // VXPG process
    // --------------------------------------------------------------------------------
    // prepare stages
    addPass(std::make_unique<Addon::VXGuiding::PrebakeDummyPass>(need_rebake), "Prebake Pass");
    addPass(std::make_unique<Addon::VXGuiding::VXGuiderClearPass>(&setting, &vxgSetting), "GuiderClear Pass");
    addEdge("Prebake Pass", "AABBMin", "GuiderClear Pass", "AABBMinPrebake");
    addEdge("Prebake Pass", "AABBMax", "GuiderClear Pass", "AABBMaxPrebake");

    addPass(std::make_unique<Addon::VXGuiding::VXGuiderLightInjection>(&setting, &vxgSetting), "ImportonInjection Pass");
    addEdge("VBuffer Pass", "VBuffer", "ImportonInjection Pass", "VBuffer");
    addEdge("GuiderClear Pass", "Irradiance", "ImportonInjection Pass", "Irradiance");
    addEdge("GuiderClear Pass", "VPLCount", "ImportonInjection Pass", "VPLCount");

    // geometry injection of dynamic objects
    addPass(std::make_unique<Addon::VXGuiding::VXGuiderGeometryPass>(&setting), "GuiderGeom Pass");
    addEdge("GuiderClear Pass", "AABBMin", "GuiderGeom Pass", "AABBMin");
    addEdge("GuiderClear Pass", "AABBMax", "GuiderGeom Pass", "AABBMax");
    addEdge("ImportonInjection Pass", "Irradiance", "GuiderGeom Pass", "Irradiance");

    addPass(std::make_unique<Addon::VXGuiding::VXGuiderCompactPass>(&setting), "VXGuiderCompact Pass");
    addEdge("GuiderClear Pass", "CounterBuffer", "VXGuiderCompact Pass", "CounterBuffer");
    addEdge("ImportonInjection Pass", "Irradiance", "VXGuiderCompact Pass", "Irradiance");

    // Superpixel clustering
    addPass(std::make_unique<Addon::gSLICr::InitClusterCenterPass>(&slicSetting), "InitClusterCenter Pass");
    addEdge("ImportonInjection Pass", "ShadingPoints", "InitClusterCenter Pass",  "Color");
    addPass(std::make_unique<Addon::gSLICr::FindCenterAssociationPass>(&slicSetting), "FindCenterAssociation Pass");
    addEdge("InitClusterCenter Pass", "Color", "FindCenterAssociation Pass", "Color");
    addEdge("InitClusterCenter Pass", "SPixelInfo", "FindCenterAssociation Pass", "SPixelInfo");
    addEdge("InitClusterCenter Pass", "IndexImage", "FindCenterAssociation Pass", "IndexImage");

    bool use_mcrs_cluster = true;
    // mrcs based clustering
    if (use_mcrs_cluster) {
      addPass(std::make_unique<Addon::VXGuiding::VXInfoClearPass>(&setting), "RCClear Pass");
      addEdge("GuiderGeom Pass", "AABBMin", "RCClear Pass", "AABBMin");
      addEdge("GuiderGeom Pass", "AABBMax", "RCClear Pass", "AABBMax");
      addEdge("VXGuiderCompact Pass", "CompactIndices", "RCClear Pass", "CompactIndices");
      addEdge("VXGuiderCompact Pass", "CounterBuffer", "RCClear Pass", "CounterBuffer");
      addEdge("ImportonInjection Pass", "Irradiance", "RCClear Pass", "Irradiance");
      addEdge("ImportonInjection Pass", "VPLCount", "RCClear Pass", "VPLCount");

      addPass(std::make_unique<Addon::VXGuiding::RowColumnPresamplePass>(&setting), "RCPresample Pass");
      addEdge("VBuffer Pass", "VBuffer", "RCPresample Pass", "VBuffer");
      addEdge("VXGuiderCompact Pass", "CounterBuffer", "RCPresample Pass", "VXCounter");
      addEdge("VXGuiderCompact Pass", "InverseIndex", "RCPresample Pass", "VXInverseIndex");
      addEdge("ImportonInjection Pass", "Positions", "RCPresample Pass", "VPLPositions");
      addEdge("RCClear Pass", "BoundMin", "RCPresample Pass", "BoundMin");
      addEdge("RCClear Pass", "BoundMax", "RCPresample Pass", "BoundMax");
      //addEdge("RCClear Pass", "DirectionalHash", "RCPresample Pass", "DirectionalHash");

      addPass(std::make_unique<Addon::VXGuiding::RowVisibilityPass>(), "RowVisibility Pass");
      addEdge("RCPresample Pass", "RepresentPixel", "RowVisibility Pass", "RepresentPixel");
      addEdge("RCPresample Pass", "RepresentVPL", "RowVisibility Pass", "RepresentVPL");
      addEdge("RCPresample Pass", "IndirectArgs", "RowVisibility Pass", "IndirectArgs");
    
      addPass(std::make_unique<Addon::VXGuiding::RowKmppCenterPass>(), "RowKmppCenter Pass");
      addEdge("RCPresample Pass", "IndirectArgs", "RowKmppCenter Pass", "IndirectArgs");
      addEdge("RowVisibility Pass", "RowVisibility", "RowKmppCenter Pass", "RowVisibility");
    
      addPass(std::make_unique<Addon::VXGuiding::RowFindCenterPass>(), "RowFindCenter Pass");
      addEdge("RCPresample Pass", "IndirectArgs", "RowFindCenter Pass", "IndirectArgs");
      addEdge("RowVisibility Pass", "RowVisibility", "RowFindCenter Pass", "RowVisibility");
      addEdge("RowKmppCenter Pass", "RowClusterInfo", "RowFindCenter Pass", "RowClusterInfo");
      addEdge("VXGuiderCompact Pass", "CompactIndices", "RowFindCenter Pass", "CompactIndices");
    }

    // BL - VXTree building
    {
      addPass(std::make_unique<Addon::VXGuiding::VXTreeEncodePass>(&setting), "TreeEncode Pass");
      addEdge("VXGuiderCompact Pass", "CounterBuffer", "TreeEncode Pass", "CounterBuffer");
      addEdge("VXGuiderCompact Pass", "CompactIndices", "TreeEncode Pass", "CompactIndices");
      if (use_mcrs_cluster) {
        addEdge("RowFindCenter Pass", "Clusters", "TreeEncode Pass", "VXClusterAssociation");
        addEdge("RCPresample Pass", "IndirectArgs", "TreeEncode Pass", "DispatchIndirectArgs");
      } else {
        addEdge("VXClusterFindAssociate2 Pass", "AssociateBuffer", "TreeEncode Pass", "VXClusterAssociation");
        addEdge("VXClusterInitCenter Pass", "DispatchIndirectArgs", "TreeEncode Pass", "DispatchIndirectArgs");
      }
      
      sort_setting.dispath = SIByL::Addon::BitonicSort::BitonicSortSetting::DispathType::DYNAMIC_INDIRECT;
      sort_setting.element_count = 65536;
      addSubgraph(std::make_unique<Addon::BitonicSort::BitonicSort>(&sort_setting),"Sort Pass");
      addEdge("VXGuiderCompact Pass", "CounterBuffer", "Sort Pass", "CounterBuffer");
      addEdge("TreeEncode Pass", "Code", "Sort Pass", "Input");
      
      addPass(std::make_unique<Addon::VXGuiding::VXTreeIIntializePass>(&setting),"TreeInitialize Pass");
      addEdge("Sort Pass", "Output", "TreeInitialize Pass", "Code");
      addEdge("VXGuiderCompact Pass", "CompactIndices", "TreeInitialize Pass", "CompactIndices");
      if (use_mcrs_cluster)  {
        addEdge("RowFindCenter Pass", "Clusters", "TreeInitialize Pass", "VXClusterAssociation");
        addEdge("RCClear Pass", "VXPremulIrradiance", "TreeInitialize Pass", "Irradiance");
      } else {
        addEdge("VXClusterFindAssociate2 Pass", "AssociateBuffer", "TreeInitialize Pass", "VXClusterAssociation");
        addEdge("VXClusterComputeInfo Pass", "VXPremulIrradiance", "TreeInitialize Pass", "Irradiance");
      }
      addEdge("TreeEncode Pass", "IndirectArgs", "TreeInitialize Pass", "IndirectArgs");
      
      addPass(std::make_unique<Addon::VXGuiding::VXTreeInternalPass>(), "TreeInternal Pass");
      addEdge("Sort Pass", "Output", "TreeInternal Pass", "Code");
      addEdge("TreeInitialize Pass", "Node", "TreeInternal Pass", "Node");
      addEdge("TreeEncode Pass", "IndirectArgs", "TreeInternal Pass", "IndirectArgs");
      addEdge("TreeInitialize Pass", "Descendant", "TreeInternal Pass", "Descendant");
      
      addPass(std::make_unique<Addon::VXGuiding::VXTreeMergePass>(), "TreeMerge Pass");
      addEdge("TreeInternal Pass", "Node", "TreeMerge Pass", "Node");
      addEdge("TreeInitialize Pass", "ClusterRoots", "TreeMerge Pass", "ClusterRoots");
      addEdge("TreeEncode Pass", "IndirectArgs", "TreeMerge Pass", "IndirectArgs");
    }
    
    // Rearrange vx info for further visibility check
    if (use_mcrs_cluster) {
      addPass(std::make_unique<Addon::VXGuiding::VXInfoRearrangePass>(), "VXInfoRearrange Pass");
      addEdge("GuiderGeom Pass", "AABBMin", "VXInfoRearrange Pass", "AABBMin");
      addEdge("GuiderGeom Pass", "AABBMax", "VXInfoRearrange Pass", "AABBMax");
      addEdge("RCPresample Pass", "BoundMin", "VXInfoRearrange Pass", "BoundMin");
      addEdge("RCPresample Pass", "BoundMax", "VXInfoRearrange Pass", "BoundMax");
      addEdge("TreeInitialize Pass", "Compact2Leaf", "VXInfoRearrange Pass", "Compact2Leaf");
      addEdge("RCPresample Pass", "IndirectArgs", "VXInfoRearrange Pass", "IndirectArgs");
    }
    
    addPass(std::make_unique<Addon::VXGuiding::SPixelClearPass>(), "VisibilityClear Pass");
    
    addPass(std::make_unique<Addon::VXGuiding::SPixelGatherPass>(&setting), "VisibilityGather Pass");
    addEdge("VisibilityClear Pass", "SPixelVisibility", "VisibilityGather Pass", "SPixelVisibility");
    addEdge("VisibilityClear Pass", "SPixelCounter", "VisibilityGather Pass", "SPixelCounter");
    addEdge("VisibilityClear Pass", "ClusterCounter", "VisibilityGather Pass", "ClusterCounter");
    addEdge("FindCenterAssociation Pass", "IndexImage", "VisibilityGather Pass", "SPixelIndexImage");
    addEdge("ImportonInjection Pass", "Positions", "VisibilityGather Pass", "VPLPositions");
    addEdge("VXGuiderCompact Pass", "InverseIndex", "VisibilityGather Pass", "VXInverseIndex");
    addEdge("RowFindCenter Pass", "Clusters", "VisibilityGather Pass", "VXClusterAssociation");

    addPass(std::make_unique<Addon::VXGuiding::SPixelVisibilityEXPass>(&setting), "VisibilityAdditional Pass");
    addEdge("VisibilityGather Pass", "SPixelGathered", "VisibilityAdditional Pass", "SPixelGathered");
    addEdge("VisibilityGather Pass", "ClusterGathered", "VisibilityAdditional Pass", "ClusterGathered");
    addEdge("VisibilityGather Pass", "SPixelCounter", "VisibilityAdditional Pass", "SPixelCounter");
    addEdge("VisibilityGather Pass", "ClusterCounter", "VisibilityAdditional Pass", "ClusterCounter");
    addEdge("VisibilityGather Pass", "SPixelVisibility", "VisibilityAdditional Pass", "SPixelVisibility");
    addEdge("VBuffer Pass", "VBuffer", "VisibilityAdditional Pass", "VBuffer");
    addEdge("VisibilityClear Pass", "SPixelAvgVisibility", "VisibilityAdditional Pass", "SPixelAvgVisibility");

    addPass(std::make_unique<Addon::VXGuiding::VXTreeTopLevelPass>(), "TreeTopLevel Pass");
    addEdge("TreeMerge Pass", "Node", "TreeTopLevel Pass", "Node");
    addEdge("TreeMerge Pass", "ClusterRoots", "TreeTopLevel Pass", "ClusterRoots");
    addEdge("VisibilityAdditional Pass", "SPixelVisibility", "TreeTopLevel Pass", "SPixelVisibility");
    addEdge("VisibilityAdditional Pass", "SPixelAvgVisibility", "TreeTopLevel Pass", "SPixelAvgVisibility");
    
    addPass(std::make_unique<Addon::VXGuiding::VXGuiderReSTIRGIPass>(&setting, &restirgi_param), "InitialSample Pass");
    addEdge("VXGuiderCompact Pass", "CounterBuffer", "InitialSample Pass", "CounterBuffer");
    addEdge("VXGuiderCompact Pass", "CompactIndices", "InitialSample Pass", "CompactIndices");
    addEdge("VXInfoRearrange Pass", "AABBMin", "InitialSample Pass", "AABBMin");
    addEdge("VXInfoRearrange Pass", "AABBMax", "InitialSample Pass", "AABBMax");
    addEdge("ImportonInjection Pass", "Color", "InitialSample Pass", "Color");
    addEdge("ImportonInjection Pass", "Positions", "InitialSample Pass", "Positions");
    addEdge("ImportonInjection Pass", "Irradiance", "InitialSample Pass", "Irradiance");
    addEdge("ImportonInjection Pass", "Intensity", "InitialSample Pass", "Intensity");
    addEdge("VBuffer Pass", "VBuffer", "InitialSample Pass", "VBuffer");
    addEdge("TreeMerge Pass", "Node", "InitialSample Pass", "Node");
    addEdge("TreeMerge Pass", "ClusterRoots", "InitialSample Pass", "ClusterRoots");
    addEdge("TreeTopLevel Pass", "TopLevelTree", "InitialSample Pass", "TopLevelTree");
    addEdge("FindCenterAssociation Pass", "IndexImage", "InitialSample Pass", "SPixelIndexImage");
    addEdge("FindCenterAssociation Pass", "FuzzyWeight", "InitialSample Pass", "FuzzyWeight");
    addEdge("FindCenterAssociation Pass", "FuzzyIndex", "InitialSample Pass", "FuzzyIndex");
    addEdge("VXGuiderCompact Pass", "InverseIndex", "InitialSample Pass", "InverseIndex");
    addEdge("TreeInitialize Pass", "Compact2Leaf", "InitialSample Pass", "Compact2Leaf");
    addEdge("RowFindCenter Pass", "Clusters", "InitialSample Pass", "AssociateBuffer");

    // ReSTIR process
    // --------------------------------------------------------------------------------
    // Execute temporal resampling
    addPass(std::make_unique<Addon::RestirGI::TemporalResampling>(&restirgi_param), "TemporalResampling Pass");
    Addon::GBufferUtils::addGBufferEdges(this, "VBuffer2GBuffer Pass", "TemporalResampling Pass");
    Addon::GBufferUtils::addPrevGBufferEdges(this, "GBufferPrev Pass", "TemporalResampling Pass");
    addEdge("InitialSample Pass", "GIReservoir", "TemporalResampling Pass", "GIReservoir");

    // Execute spatial resampling
    addPass(std::make_unique<Addon::RestirGI::SpatialResampling>(&restirgi_param), "SpatialResampling Pass");
    Addon::GBufferUtils::addGBufferEdges(this, "VBuffer2GBuffer Pass", "SpatialResampling Pass");
    addEdge("TemporalResampling Pass", "GIReservoir", "SpatialResampling Pass", "GIReservoir");
    
    // Execute final shading
    addPass(std::make_unique<Addon::RestirGI::FinalShading>(&restirgi_param), "FinalShading Pass");
    Addon::GBufferUtils::addGBufferEdges(this, "VBuffer2GBuffer Pass", "FinalShading Pass");
    addEdge("SpatialResampling Pass", "GIReservoir", "FinalShading Pass", "GIReservoir");

    // All post-processing stuff ...
    addPass(std::make_unique<AccumulatePass>(), "Accum Pass");
    addEdge("FinalShading Pass", "Diffuse", "Accum Pass", "Input");
    addSubgraph(std::make_unique<Addon::GBufferHolderGraph>(), "GBuffer Blit Pass");
    Addon::GBufferUtils::addBlitPrevGBufferEdges(this, "VBuffer2GBuffer Pass", "TemporalResampling Pass", "GBuffer Blit Pass");
    addPass(std::make_unique<Addon::Postprocess::ToneMapperPass>(), "ToneMapper Pass");
    addEdge("Accum Pass", "Output", "ToneMapper Pass", "Input");

    markOutput("Accum Pass", "Output");
  }

  Addon::RestirGI::GIResamplingRuntimeParameters restirgi_param;
  Addon::VXGI::VXGISetting setting;
  Addon::VXGuiding::DITestSetting diTestSetting;
  Addon::gSLICr::gSLICrSetting slicSetting;
  Addon::VXGuiding::VXGuidingSetting vxgSetting;
  Addon::BitonicSort::BitonicSortSetting sort_setting;
  bool need_rebake = false;
};

SE_EXPORT struct VXPGReSTIRPipeline : public RDG::Pipeline {
  VXPGReSTIRPipeline() {}
  enum struct Phase {
    GeometryBaking,
    Runtime,
  } curr_phase = Phase::GeometryBaking;

  virtual auto build() noexcept -> void {
    prebake_graph.build();
    GFX::Buffer* prebake_geom_min =
        prebake_graph.getBufferResource("Prebake Pass", "AABBMin");
    GFX::Buffer* prebake_geom_max =
        prebake_graph.getBufferResource("Prebake Pass", "AABBMax");
    runtime_graph.setExternal("Prebake Pass", "AABBMin", prebake_geom_min);
    runtime_graph.setExternal("Prebake Pass", "AABBMax", prebake_geom_max);
    runtime_graph.build();
  }

  virtual auto execute(RHI::CommandEncoder* encoder) noexcept -> void {
    auto graphs = getActiveGraphs();
    for (auto* graph : graphs) graph->execute(encoder);
    if (curr_phase == Phase::GeometryBaking) {
      curr_phase = Phase::Runtime;
    }
    if (runtime_graph.need_rebake) {
      curr_phase = Phase::GeometryBaking;
      runtime_graph.need_rebake = false;
    }
  }

  virtual auto getActiveGraphs() noexcept -> std::vector<RDG::Graph*> {
    if (curr_phase == Phase::GeometryBaking)
      return {&prebake_graph};
    else if (curr_phase == Phase::Runtime)
      return {&runtime_graph};
    else
      return {};
  }

  virtual auto getOutput() noexcept -> GFX::Texture* {
    return runtime_graph.getOutput();
  }

  VXGuidingPrebakeGraph prebake_graph;
  VXPGRestirGIGraph runtime_graph;
};

SE_EXPORT struct ADGTGraph : public RDG::Graph {
  Addon::VXGI::VXGISetting setting;
  ADGTGraph() {
    addPass(std::make_unique<Addon::Differentiable::TestGTPass>(), "TestGT Pass");
    addPass(std::make_unique<AccumulatePass>(), "Accum Pass");
    addEdge("TestGT Pass", "Output", "Accum Pass", "Input");
    
    markOutput("Accum Pass", "Output");
  }
};

SE_EXPORT struct ADGTPipeline : public RDG::SingleGraphPipeline {
  ADGTPipeline() { pGraph = &graph; }
  ADGTGraph graph;
};
}  // namespace SIByL