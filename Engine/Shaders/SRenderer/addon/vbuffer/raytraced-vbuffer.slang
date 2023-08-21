#include "../../raytracer/spt_interface.hlsli"
#include "../../include/common/random.hlsli"
#include "../../include/common/sampling.hlsli"
#include "../../include/raytracer_descriptor_set.hlsli"
#include "../../include/scene_descriptor_set.hlsli"
#include "vbuffer.hlsli"

[[vk::push_constant]]
cbuffer PushConstants {
    struct PushConstants {
        uint2 resolution;
        uint sample_batch;
    } pConst;
};

RWTexture2D<uint4> u_vBuffer;

[shader("raygeneration")]
void RgenMain() {
    // The resolution of the image, which is the same as the launch size:
    const int2 resolution = int2(pConst.resolution);
    const int2 pixel = int2(DispatchRaysIndex().xy);
    // If the pixel is outside of the image, don't do anything:
    if(any(pixel >= resolution)) {
        return;
    }
    RandomSamplerState RNG = InitRandomSampler(pixel, pConst.sample_batch);
    
    Ray primaryRay = generateRay(
        pixel,
        resolution,
        false,
        globalUniform.cameraData
    );
    
    PrimaryPayload payload;
    payload.RND = RNG;
    
    uint instanceMask = 0xff;
    uint rayFlags = RAY_FLAG_NONE;
    TraceRay(SceneBVH, rayFlags, instanceMask, 0, 0, 0, ToRayDesc(primaryRay), payload);
    RNG = payload.RND;

    VBufferData vbufferData;

    if (HasHit(payload.hit) == false) {
        // set to invalid
        vbufferData.primitiveID = 0xFFFFFFFF;
    }
    else {
        vbufferData.geometryID = payload.hit.geometryID;
        vbufferData.primitiveID = payload.hit.primitiveID;
        vbufferData.barycentric = payload.hit.barycentric;
        vbufferData.primitiveType = 0;
    }
    u_vBuffer[pixel] = PackVBufferData(vbufferData);
    return;
}