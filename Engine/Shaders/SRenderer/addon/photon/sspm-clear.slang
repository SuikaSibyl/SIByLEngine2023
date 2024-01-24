#include "../vbuffer/vbuffer.hlsli"
#include "common.hlsli"

const Texture2D<uint4> u_vBuffer;

RWByteAddressBuffer u_photon_map;
RWTexture2D<float2> u_depth;

[[vk::push_constant]]
cbuffer PushConstants {
    int2 resolution;
};

float2 pixel_depth_area(CameraData camera, float3 position) {
    const float x = resolution.y * camera.focalDistance / (2 * length(camera.cameraV));
    const float dist = distance(camera.posW, position);
    const float area = (dist * dist) / (x * x);
    return float2(dist, area);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void ComputeMain(uint3 DTid: SV_DispatchThreadID) {
    const int2 pixel = int2(DTid.xy);
    if (any(pixel >= resolution)) return;
    // Clear the screen space photon map
    const uint flatten = FlattensPixelToIndex(pixel, resolution.x);
    u_photon_map.Store<float3>(flatten * sizeof(float3), float3(0));
    // unpack visibility buffer
    const Ray primaryRay = generateRay(pixel, resolution, false, globalUniform.cameraData);
    const VBufferData vhit = UnpackVBufferData(u_vBuffer[pixel]);
    const GeometryHit hit = GetGeometryHit(vhit, primaryRay);
    
    if (!IsInvalid(vhit)) {
        u_depth[pixel] = pixel_depth_area(globalUniform.cameraData, hit.position);
    } else {
        u_depth[pixel] = float2(0, 0);
    }
}