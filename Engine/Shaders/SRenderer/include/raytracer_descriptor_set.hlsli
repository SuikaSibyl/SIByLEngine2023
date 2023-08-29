#ifndef _SRENDERER_COMMON_RAY_TRACER_DESCRIPOTR_SET_HEADER_
#define _SRENDERER_COMMON_RAY_TRACER_DESCRIPOTR_SET_HEADER_

[[vk::binding(0, 1)]]
RaytracingAccelerationStructure SceneBVH;
[[vk::binding(1, 1)]]
RaytracingAccelerationStructure PrevSceneBVH;

#endif // !_SRENDERER_COMMON_RAY_TRACER_DESCRIPOTR_SET_HEADER_