#pragma once

#include <SE.Addon.GBuffer.hpp>
#include <Passes/RasterizerPasses/SE.SRenderer-PreZPass.hpp>
#include <Passes/FullScreenPasses/SE.SRenderer-AccumulatePass.hpp>
#include <Passes/RasterizerPasses/SE.SRenderer-GeometryInspectorPass.hpp>
#include <SE.Addon.SemiNee.hpp>
#include <SE.Addon.BitonicSort.hpp>
#include <SE.Addon.VXGI.hpp>
#include <SE.Addon.VXGuiding.hpp>
#include <SE.Addon.SLC.hpp>
#include <SE.Addon.VBuffer.hpp>
#include <SE.Addon.gSLICr.hpp>
#include <SE.Addon.SSGuiding.hpp>
#include <SE.Addon.ASVGF.hpp>

namespace SIByL {
SE_EXPORT struct CustomGraph : public RDG::Graph {
  CustomGraph() {
    addPass(std::make_unique<Addon::VBuffer::RayTraceVBuffer>(), "VBuffer Pass");
    addPass(std::make_unique<Addon::VBuffer::VBuffer2GBufferPass>(), "VBuffer2GBuffer Pass");
    addEdge("VBuffer Pass", "VBuffer", "VBuffer2GBuffer Pass", "VBuffer");
    // alternative: use rasterizer g-buffer
    //addSubgraph(std::make_unique<PreZPass>(), "Pre-Z Pass");
    //addPass(std::make_unique<Addon::RasterizedGBufferPass>(), "GBuffer Pass");
    //addEdge("Pre-Z Pass", "Depth", "GBuffer Pass", "Depth");

    // Hold history GBuffer
    addPass(std::make_unique<Addon::GBufferHolderSource>(), "GBufferPrev Pass");

    addPass(std::make_unique<Addon::GBufferTemporalInspectorPass>(), "GBuffer Inspect Pass");
    Addon::GBufferUtils::addGBufferEdges(this, "VBuffer2GBuffer Pass", "GBuffer Inspect Pass");
    Addon::GBufferUtils::addPrevGBufferEdges(this, "GBufferPrev Pass", "GBuffer Inspect Pass");

    // Blit history GBuffer
    addSubgraph(std::make_unique<Addon::GBufferHolderGraph>(), "GBuffer Blit Pass");
    Addon::GBufferUtils::addBlitPrevGBufferEdges(
        this, "VBuffer2GBuffer Pass", "GBuffer Inspect Pass", "GBuffer Blit Pass");
    
    markOutput("GBuffer Inspect Pass", "Output");
  }
};

SE_EXPORT struct CustomPipeline : public RDG::SingleGraphPipeline {
  CustomPipeline() { pGraph = &graph; }
  CustomGraph graph;
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

    markOutput("Accum Pass", "Output");
  }
};

SE_EXPORT struct GTPipeline : public RDG::SingleGraphPipeline {
  GTPipeline() { pGraph = &graph; }
  GTGraph graph;
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
    addPass(std::make_unique<Addon::VBuffer::RayTraceVBuffer>(),
            "VBuffer Pass");
    addPass(std::make_unique<Addon::SSGuiding::SSPGvMF_ClearPass>(),
            "GuiderClear Pass");
    addPass(std::make_unique<Addon::SSGuiding::SSPGvMF_SamplePass>(),
            "Sample Pass");
    addEdge("VBuffer Pass", "VBuffer", "Sample Pass", "VBuffer");
    addEdge("GuiderClear Pass", "vMFStatistics", "Sample Pass", "vMFStatistics");
    addEdge("GuiderClear Pass", "EpochCounter", "Sample Pass", "EpochCounter");
    addPass(std::make_unique<Addon::SSGuiding::SSPGvMF_VisPass>(), "Vis Pass");
    addEdge("VBuffer Pass", "VBuffer", "Vis Pass", "VBuffer");
    addEdge("Sample Pass", "vMFStatistics", "Vis Pass", "vMFStatistics");

    addPass(std::make_unique<AccumulatePass>(), "Accum Pass");
    addEdge("Sample Pass", "Color", "Accum Pass", "Input");

    markOutput("Accum Pass", "Output");
  }
};

SE_EXPORT struct SSPGPipeline : public RDG::SingleGraphPipeline {
  SSPGPipeline() { pGraph = &graph; }
  SSPGGraph graph;
};

SE_EXPORT struct VXGuidingGraph : public RDG::Graph {
  Addon::VXGI::VXGISetting setting;
  Addon::VXGuiding::DITestSetting diTestSetting;
  Addon::gSLICr::gSLICrSetting slicSetting;
  Addon::VXGuiding::VXGuidingSetting vxgSetting;

  VXGuidingGraph() {
    setting.clipmapSetting.mip = 6;
    slicSetting.img_size = {1280, 720};
    slicSetting.map_size = {(1280 + 31) / 32, (720 + 31) / 32};
    slicSetting.number_iter = 5;
    slicSetting.spixel_size = 32;

    //addPass(std::make_unique<Addon::VXGuiding::VXGuiderClearPass>(&setting),
    //        "GuiderClear Pass");
    //addPass(std::make_unique<Addon::VXGuiding::VXGuiderDIInjection>(
    //            &diTestSetting, &setting),
    //        "GuiderGeom Pass");
    //addEdge("GuiderClear Pass", "AABBMin", "GuiderGeom Pass", "AABBMin");
    //addEdge("GuiderClear Pass", "AABBMax", "GuiderGeom Pass", "AABBMax");
    //addEdge("GuiderClear Pass", "Irradiance", "GuiderGeom Pass", "Irradiance");

    //addSubgraph(std::make_unique<PreZPass>(), "Pre-Z Pass");
    //addPass(std::make_unique<GeometryInspectorPass>(), "GeoInspect Pass");
    //addEdge("Pre-Z Pass", "Depth", "GeoInspect Pass", "Depth");

    //addPass(std::make_unique<Addon::VXGuiding::VXGuiderCompactPass>(&setting),
    //        "VXGuiderCompact Pass");
    //addEdge("GuiderClear Pass", "CounterBuffer", "VXGuiderCompact Pass", "CounterBuffer");
    //addEdge("GuiderGeom Pass", "Irradiance", "VXGuiderCompact Pass", "Irradiance");

    //addPass(std::make_unique<Addon::VXGuiding::DITestPass>(&diTestSetting,
    //                                                       &setting),
    //        "DITestComp Pass");
    //addEdge("VXGuiderCompact Pass", "CounterBuffer", "DITestComp Pass", "CounterBuffer");
    //addEdge("VXGuiderCompact Pass", "CompactIndices", "DITestComp Pass", "CompactIndices");
    //addEdge("GuiderGeom Pass", "AABBMin", "DITestComp Pass", "AABBMin");
    //addEdge("GuiderGeom Pass", "AABBMax", "DITestComp Pass", "AABBMax");

    //addPass(std::make_unique<AccumulatePass>(), "Accum Pass");
    //addEdge("DITestComp Pass", "Color", "Accum Pass", "Input");

    //addPass(std::make_unique<Addon::VXGuiding::VXGuiderVisualizePass>(&setting),
    //        "VXGuiderVIS Pass");
    //addEdge("Pre-Z Pass", "Depth", "VXGuiderVIS Pass", "Depth");
    //addEdge("Accum Pass", "Output", "VXGuiderVIS Pass", "Color");
    //addEdge("GuiderGeom Pass", "AABBMin", "VXGuiderVIS Pass", "AABBMin");
    //addEdge("GuiderGeom Pass", "AABBMax", "VXGuiderVIS Pass", "AABBMax");

    //markOutput("VXGuiderVIS Pass", "Color");

    addPass(std::make_unique<Addon::VBuffer::RayTraceVBuffer>(),
        "VBuffer Pass");
    addPass(std::make_unique<Addon::VXGuiding::VXGuiderClearPass>(&setting,
                                                                  &vxgSetting),
            "GuiderClear Pass");


    addPass(std::make_unique<Addon::VXGuiding::VXGuider1stBounceInjection>(
                &setting, &vxgSetting),
            "ImportonInjection Pass");
    addEdge("VBuffer Pass", "VBuffer", "ImportonInjection Pass", "VBuffer");
    addEdge("GuiderClear Pass", "Irradiance", "ImportonInjection Pass", "Irradiance");
    
    addPass(std::make_unique<Addon::VXGuiding::VXGuiderGeometryPass>(&setting),
            "GuiderGeom Pass");
    addEdge("GuiderClear Pass", "AABBMin", "GuiderGeom Pass", "AABBMin");
    addEdge("GuiderClear Pass", "AABBMax", "GuiderGeom Pass", "AABBMax");
    addEdge("ImportonInjection Pass", "Irradiance", "GuiderGeom Pass",
            "Irradiance");

    addPass(std::make_unique<Addon::VXGuiding::VXGuiderCompactPass>(&setting),
            "VXGuiderCompact Pass");
    addEdge("GuiderClear Pass", "CounterBuffer", "VXGuiderCompact Pass",
            "CounterBuffer");
    addEdge("GuiderGeom Pass", "Irradiance", "VXGuiderCompact Pass",
            "Irradiance");

    addSubgraph(std::make_unique<PreZPass>(), "Pre-Z Pass");
    //addPass(std::make_unique<GeometryInspectorPass>(), "GeoInspect Pass");
    //addEdge("Pre-Z Pass", "Depth", "GeoInspect Pass", "Depth");

    addPass(
        std::make_unique<Addon::gSLICr::InitClusterCenterPass>(&slicSetting),
        "InitClusterCenter Pass");
    addEdge("ImportonInjection Pass", "ShadingPoints", "InitClusterCenter Pass",
            "Color");
    addPass(std::make_unique<Addon::gSLICr::FindCenterAssociationPass>(
                &slicSetting),
            "FindCenterAssociation Pass");
    addEdge("InitClusterCenter Pass", "Color", "FindCenterAssociation Pass",
            "Color");
    addEdge("InitClusterCenter Pass", "SPixelInfo",
            "FindCenterAssociation Pass", "SPixelInfo");
    addEdge("InitClusterCenter Pass", "IndexImage",
            "FindCenterAssociation Pass", "IndexImage");


    addPass(
        std::make_unique<Addon::VXGuiding::VXClusterComputeInfoPass>(&setting),
        "VXClusterComputeInfo Pass");
    addEdge("GuiderGeom Pass", "AABBMin", "VXClusterComputeInfo Pass", "AABBMin");
    addEdge("GuiderGeom Pass", "AABBMax", "VXClusterComputeInfo Pass", "AABBMax");
    addEdge("VXGuiderCompact Pass", "CompactIndices",
            "VXClusterComputeInfo Pass", "CompactIndices");
    addEdge("GuiderGeom Pass", "Irradiance", "VXClusterComputeInfo Pass",
            "Irradiance");
    addEdge("VXGuiderCompact Pass", "CounterBuffer",
            "VXClusterComputeInfo Pass", "CounterBuffer");

    addPass(std::make_unique<Addon::VXGuiding::VXClusterSeedingPass>(),
            "VXGuiderSeed Pass");
    addEdge("VXGuiderCompact Pass", "CompactIndices",
            "VXGuiderSeed Pass", "CompactIndices");
    addEdge("VXGuiderCompact Pass", "CounterBuffer", "VXGuiderSeed Pass",
            "CounterBuffer");
    addEdge("VXClusterComputeInfo Pass", "VXNormal", "VXGuiderSeed Pass",
            "VXNormal");

    addPass(
        std::make_unique<Addon::VXGuiding::VXClusterInitCenterPass>(&setting),
        "VXClusterInitCenter Pass");
    addEdge("VXGuiderCompact Pass", "CompactIndices",
            "VXClusterInitCenter Pass", "CompactIndices");
    addEdge("VXGuiderCompact Pass", "CounterBuffer",
            "VXClusterInitCenter Pass", "CounterBuffer");
    addEdge("VXClusterComputeInfo Pass", "VXNormal",
            "VXClusterInitCenter Pass", "VXNormal");
    addEdge("VXGuiderSeed Pass", "ClusterSeeds",
            "VXClusterInitCenter Pass", "ClusterSeeds");

    addPass(std::make_unique<Addon::VXGuiding::VXClusterFindAssociatePass>(
                &setting),
            "VXClusterFindAssociate Pass");
    addEdge("VXGuiderCompact Pass", "CounterBuffer",
            "VXClusterFindAssociate Pass", "CounterBuffer");
    addEdge("VXClusterInitCenter Pass", "GridClusterCount",
            "VXClusterFindAssociate Pass", "GridClusterCount");
    addEdge("VXClusterInitCenter Pass", "GridClusterIndices",
            "VXClusterFindAssociate Pass", "GridClusterIndices");
    addEdge("VXClusterComputeInfo Pass", "AssociateBuffer",
            "VXClusterFindAssociate Pass", "AssociateBuffer");
    addEdge("VXGuiderCompact Pass", "CompactIndices",
            "VXClusterFindAssociate Pass", "CompactIndices");
    addEdge("VXClusterComputeInfo Pass", "VXNormal",
            "VXClusterFindAssociate Pass", "VXNormal");
    addEdge("VXClusterInitCenter Pass", "SVXInfo",
            "VXClusterFindAssociate Pass", "SVXInfo");
    addEdge("VXClusterInitCenter Pass", "SVXAccumInfo",
            "VXClusterFindAssociate Pass", "SVXAccumInfo");
    addEdge("VXClusterInitCenter Pass", "DispatchIndirectArgs",
            "VXClusterFindAssociate Pass", "DispatchIndirectArgs");

    addPass(
        std::make_unique<Addon::VXGuiding::VXClusterUpdateCenterPass>(&setting),
        "VXClusterUpdate Pass");
    addEdge("VXClusterInitCenter Pass", "SVXInfo", "VXClusterUpdate Pass",
            "SVXInfo");
    addEdge("VXClusterFindAssociate Pass", "SVXAccumInfo",
            "VXClusterUpdate Pass", "SVXAccumInfo");
    addEdge("VXClusterInitCenter Pass", "GridClusterCount",
            "VXClusterUpdate Pass", "GridClusterCount");
    addEdge("VXClusterInitCenter Pass", "GridClusterIndices",
            "VXClusterUpdate Pass", "GridClusterIndices");

    
    addPass(std::make_unique<Addon::VXGuiding::VXClusterFindAssociatePass>(
                &setting),
            "VXClusterFindAssociate2 Pass");
    addEdge("VXGuiderCompact Pass", "CounterBuffer",
            "VXClusterFindAssociate2 Pass", "CounterBuffer");
    addEdge("VXClusterUpdate Pass", "GridClusterCount",
            "VXClusterFindAssociate2 Pass", "GridClusterCount");
    addEdge("VXClusterUpdate Pass", "GridClusterIndices",
            "VXClusterFindAssociate2 Pass", "GridClusterIndices");
    addEdge("VXClusterFindAssociate Pass", "AssociateBuffer",
            "VXClusterFindAssociate2 Pass", "AssociateBuffer");
    addEdge("VXGuiderCompact Pass", "CompactIndices",
            "VXClusterFindAssociate2 Pass", "CompactIndices");
    addEdge("VXClusterComputeInfo Pass", "VXNormal",
            "VXClusterFindAssociate2 Pass", "VXNormal");
    addEdge("VXClusterUpdate Pass", "SVXInfo",
            "VXClusterFindAssociate2 Pass", "SVXInfo");
    addEdge("VXClusterUpdate Pass", "SVXAccumInfo",
            "VXClusterFindAssociate2 Pass", "SVXAccumInfo");
    addEdge("VXClusterInitCenter Pass", "DispatchIndirectArgs",
            "VXClusterFindAssociate2 Pass", "DispatchIndirectArgs");
    
    addPass(std::make_unique<Addon::VXGuiding::SPixelClearPass>(),
            "VisibilityClear Pass");
    addPass(std::make_unique<Addon::VXGuiding::SPixelGatherPass>(&setting),
            "VisibilityGather Pass");
    addEdge("VisibilityClear Pass", "SPixelVisibility", "VisibilityGather Pass",
            "SPixelVisibility");
    addEdge("VisibilityClear Pass", "SPixelCounter", "VisibilityGather Pass",
            "SPixelCounter");
    addEdge("VisibilityClear Pass", "ClusterCounter", "VisibilityGather Pass",
            "ClusterCounter");
    addEdge("FindCenterAssociation Pass", "IndexImage", "VisibilityGather Pass",
            "SPixelIndexImage");
    addEdge("ImportonInjection Pass", "Positions", "VisibilityGather Pass",
            "VPLPositions");
    addEdge("VXGuiderCompact Pass", "InverseIndex", "VisibilityGather Pass",
            "VXInverseIndex");
    addEdge("VXClusterFindAssociate2 Pass", "AssociateBuffer",
            "VisibilityGather Pass", "VXClusterAssociation");
    addPass(std::make_unique<Addon::VXGuiding::SPixelVisibilityPass>(),
            "VisibilityAdditional Pass");
    addEdge("VisibilityGather Pass", "SPixelGathered", "VisibilityAdditional Pass",
            "SPixelGathered");
    addEdge("VisibilityGather Pass", "ClusterGathered", "VisibilityAdditional Pass",
            "ClusterGathered");
    addEdge("VisibilityGather Pass", "SPixelCounter", "VisibilityAdditional Pass",
            "SPixelCounter");
    addEdge("VisibilityGather Pass", "ClusterCounter", "VisibilityAdditional Pass",
            "ClusterCounter");
    addEdge("VisibilityGather Pass", "SPixelVisibility", "VisibilityAdditional Pass",
            "SPixelVisibility");
    addEdge("VBuffer Pass", "VBuffer", "VisibilityAdditional Pass", "VBuffer");


    addPass(std::make_unique<Addon::VXGuiding::VXTreeEncodePass>(&setting),
            "TreeEncode Pass");
    addEdge("VXGuiderCompact Pass", "CounterBuffer",
            "TreeEncode Pass", "CounterBuffer");
    addEdge("VXGuiderCompact Pass", "CompactIndices", "TreeEncode Pass",
            "CompactIndices");
    addEdge("VXClusterFindAssociate2 Pass", "AssociateBuffer",
            "TreeEncode Pass", "VXClusterAssociation");
    addEdge("VXClusterInitCenter Pass", "DispatchIndirectArgs",
            "TreeEncode Pass", "DispatchIndirectArgs");
    
    sort_setting.dispath = SIByL::Addon::BitonicSort::BitonicSortSetting::
        DispathType::DYNAMIC_INDIRECT;
    sort_setting.element_count = 65536;
    addSubgraph(
        std::make_unique<Addon::BitonicSort::BitonicSort>(&sort_setting),
        "Sort Pass");
    addEdge("VXGuiderCompact Pass", "CounterBuffer", "Sort Pass",
            "CounterBuffer");
    addEdge("TreeEncode Pass", "Code", "Sort Pass", "Input");

    addPass(std::make_unique<Addon::VXGuiding::VXTreeIIntializePass>(&setting),
            "TreeInitialize Pass");
    addEdge("Sort Pass", "Output", "TreeInitialize Pass", "Code");
    addEdge("VXGuiderCompact Pass", "CompactIndices", "TreeInitialize Pass",
            "CompactIndices");
    addEdge("VXClusterComputeInfo Pass", "VXPremulIrradiance",
            "TreeInitialize Pass", "Irradiance");
    addEdge("VXClusterFindAssociate2 Pass", "AssociateBuffer",
            "TreeInitialize Pass", "VXClusterAssociation");
    addEdge("TreeEncode Pass", "IndirectArgs", "TreeInitialize Pass",
            "IndirectArgs");

    addPass(std::make_unique<Addon::VXGuiding::VXTreeInternalPass>(),
            "TreeInternal Pass");
    addEdge("Sort Pass", "Output", "TreeInternal Pass", "Code");
    addEdge("TreeInitialize Pass", "Node", "TreeInternal Pass", "Node");
    addEdge("TreeEncode Pass", "IndirectArgs", "TreeInternal Pass",
            "IndirectArgs");
    
    addPass(std::make_unique<Addon::VXGuiding::VXTreeMergePass>(),
            "TreeMerge Pass");
    addEdge("TreeInternal Pass", "Node", "TreeMerge Pass", "Node");
    addEdge("TreeInitialize Pass", "ClusterRoots", "TreeMerge Pass", "ClusterRoots");
    addEdge("TreeEncode Pass", "IndirectArgs", "TreeMerge Pass",
            "IndirectArgs");
    
    addPass(std::make_unique<Addon::VXGuiding::VXTreeTopLevelPass>(),
            "TreeTopLevel Pass");
    addEdge("TreeMerge Pass", "Node", "TreeTopLevel Pass", "Node");
    addEdge("TreeMerge Pass", "ClusterRoots", "TreeTopLevel Pass", "ClusterRoots");
    addEdge("VisibilityAdditional Pass", "SPixelVisibility", "TreeTopLevel Pass",
            "SPixelVisibility");

    addPass(std::make_unique<Addon::VXGuiding::VXGuiderViewPass>(&setting),
            "GuiderViewer Pass");
    addEdge("GuiderGeom Pass", "Irradiance", "GuiderViewer Pass", "Irradiance");
    addEdge("VXClusterFindAssociate2 Pass", "AssociateBuffer", "GuiderViewer Pass", "AssociateBuffer");
    addEdge("VXGuiderCompact Pass", "InverseIndex", "GuiderViewer Pass", "InverseIndex");
    addEdge("VXClusterComputeInfo Pass", "VXNormal", "GuiderViewer Pass",
            "VXNormal");
    addEdge("VisibilityAdditional Pass", "SPixelVisibility", "GuiderViewer Pass",
            "SPixelVisibility");
    addEdge("FindCenterAssociation Pass", "IndexImage", "GuiderViewer Pass",
            "SPixelIndexImage");
    addEdge("VXClusterComputeInfo Pass", "VXPremulIrradiance",
            "GuiderViewer Pass", "VXPremulIrradiance");

    
    addPass(std::make_unique<Addon::VXGuiding::VXGuiderGIPass>(&setting),
            "VXGuiderGI Pass");
    addEdge("VXGuiderCompact Pass", "CounterBuffer", "VXGuiderGI Pass",
            "CounterBuffer");
    addEdge("VXGuiderCompact Pass", "CompactIndices", "VXGuiderGI Pass",
            "CompactIndices");
    addEdge("GuiderGeom Pass", "AABBMin", "VXGuiderGI Pass", "AABBMin");
    addEdge("GuiderGeom Pass", "AABBMax", "VXGuiderGI Pass", "AABBMax");
    addEdge("ImportonInjection Pass", "Color", "VXGuiderGI Pass", "Color");
    addEdge("ImportonInjection Pass", "Positions", "VXGuiderGI Pass",
            "Positions");
    addEdge("GuiderGeom Pass", "Irradiance", "VXGuiderGI Pass", "Irradiance");
    addEdge("VBuffer Pass", "VBuffer", "VXGuiderGI Pass", "VBuffer");
    addEdge("TreeMerge Pass", "Node", "VXGuiderGI Pass", "Node");
    addEdge("TreeMerge Pass", "ClusterRoots", "VXGuiderGI Pass",
            "ClusterRoots");
    addEdge("TreeTopLevel Pass", "TopLevelTree", "VXGuiderGI Pass",
            "TopLevelTree");
    addEdge("FindCenterAssociation Pass", "IndexImage", "VXGuiderGI Pass",
            "SPixelIndexImage");
    addEdge("FindCenterAssociation Pass", "FuzzyWeight", "VXGuiderGI Pass",
            "FuzzyWeight");
    addEdge("FindCenterAssociation Pass", "FuzzyIndex", "VXGuiderGI Pass",
            "FuzzyIndex");
    addEdge("VXClusterFindAssociate2 Pass", "AssociateBuffer",
            "VXGuiderGI Pass", "AssociateBuffer");
    addEdge("VXGuiderCompact Pass", "InverseIndex", "VXGuiderGI Pass",
            "InverseIndex");
    addEdge("TreeInitialize Pass", "Compact2Leaf", "VXGuiderGI Pass",
            "Compact2Leaf");

    //addPass(std::make_unique<Addon::VXGuiding::VPLVisualizePass>(), "VPLVisualize Pass");
    //addEdge("VisibilityGather Pass", "ClusterGathered",
    //        "VPLVisualize Pass", "ClusterGathered");
    //addEdge("VisibilityGather Pass", "ClusterCounter",
    //        "VPLVisualize Pass", "ClusterCounter");

    addPass(std::make_unique<AccumulatePass>(), "Accum Pass");
    addEdge("VXGuiderGI Pass", "Color", "Accum Pass", "Input");

    addPass(std::make_unique<Addon::VXGuiding::VXGuiderVisualizePass>(&setting),
            "VXGuiderVIS Pass");
    addEdge("Pre-Z Pass", "Depth", "VXGuiderVIS Pass", "Depth");
    addEdge("Accum Pass", "Output", "VXGuiderVIS Pass", "Color");
    addEdge("GuiderGeom Pass", "AABBMin", "VXGuiderVIS Pass", "AABBMin");
    addEdge("GuiderGeom Pass", "AABBMax", "VXGuiderVIS Pass", "AABBMax");

    addPass(std::make_unique<Addon::gSLICr::VisualizeSPixelPass>(slicSetting),
            "VisualizeSPixel Pass");
    addEdge("FindCenterAssociation Pass", "IndexImage", "VisualizeSPixel Pass",
            "IndexImage");
    addEdge("FindCenterAssociation Pass", "FuzzyWeight", "VisualizeSPixel Pass",
            "FuzzyWeight");
    addEdge("FindCenterAssociation Pass", "FuzzyIndex", "VisualizeSPixel Pass",
            "FuzzyIndex");

    addEdge("VXGuiderVIS Pass", "Color", "VisualizeSPixel Pass", "Color");

    markOutput("VisualizeSPixel Pass", "Color");
  }

  Addon::BitonicSort::BitonicSortSetting sort_setting;
};

SE_EXPORT struct VXGuidingPipeline : public RDG::SingleGraphPipeline {
  VXGuidingPipeline() { pGraph = &graph; }
  VXGuidingGraph graph;
};
}