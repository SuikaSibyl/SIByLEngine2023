#pragma once
#include <Resource/SE.Core.Resource.hpp>
#include <SE.GFX.hpp>
#include <SE.RDG.hpp>
#include <SE.RHI.hpp>
#include <cmath>
#include <cstdint>
#include <memory>
#include <string>

#include "../Passes/RayTracingPasses/SE.SRenderer-RTRSMPass.hpp"
#include "../Passes/FullScreenPasses/MIP/SE.SRenderer-MIPSLCPoolingPass.hpp"
#include "../Passes/FullScreenPasses/SE.SRenderer-AccumulatePass.hpp"

namespace SIByL::SRP {
SE_EXPORT struct RTGIGraph : public RDG::Graph {
  RTGIGraph() {
    addSubgraph(std::make_unique<MIPTiledVisPass>(16, 16, 1280, 720),
                "TV MIP Pass");
    addPass(std::make_unique<DirectRSMPass>(512, 512, &rsm_info),
            "DirectRS Pass");
    addSubgraph(std::make_unique<MIPSLCPass>(512, 512), "RSM MIP Pass");
    addPass(std::make_unique<RSMGIPass>(&rsm_info), "RSM GI Pass");
    addPass(std::make_unique<AccumulatePass>(), "Accumulate Pass");

    addEdge("TV MIP Pass",  "ImportanceSplatting",  "DirectRS Pass", "WeightImg");

    addEdge("DirectRS Pass", "PixImportance", "RSM MIP Pass", "PixImportance");
    addEdge("DirectRS Pass", "NormalCone", "RSM MIP Pass", "NormalCone");
    addEdge("DirectRS Pass", "AABBXY", "RSM MIP Pass", "AABBXY");
    addEdge("DirectRS Pass", "AABBZ", "RSM MIP Pass", "AABBZ");

    addEdge("TV MIP Pass",  "Output",               "RSM GI Pass", "TVIn");
    addEdge("TV MIP Pass",  "Input",                "RSM GI Pass", "TVOut");
    addEdge("RSM MIP Pass", "PixImportanceOut",     "RSM GI Pass", "PixImportance");
    addEdge("RSM MIP Pass", "NormalConeOut",        "RSM GI Pass", "NormalCone");
    addEdge("RSM MIP Pass", "AABBXYOut",            "RSM GI Pass", "AABBXY");
    addEdge("RSM MIP Pass", "AABBZOut",             "RSM GI Pass", "AABBZ");
    addEdge("DirectRS Pass",  "WeightImg",           "RSM GI Pass", "ImportanceSplatting");
    
    addEdge("RSM GI Pass", "Color", "Accumulate Pass", "Input");

    markOutput("Accumulate Pass", "Output");
  }

  RSMShareInfo rsm_info;
};

SE_EXPORT struct RTGIPipeline : public RDG::SingleGraphPipeline {
  RTGIPipeline() { pGraph = &graph; }
  RTGIGraph graph;
};
}  // namespace SIByL::SRP