#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : require

// #include "ao_shared.h"
#include "raycommon.glsl"
#include "host_device.h"
// layout(location = 0) rayPayloadInEXT RayPayload pay;
layout(location = 0) rayPayloadInEXT hitPayload prd;

layout(push_constant) uniform _PushConstantRay
{
  PushConstantRay pcRay;
};

void main()
{
  prd.hitValue = pcRay.clearColor.xyz * 0.8;
}