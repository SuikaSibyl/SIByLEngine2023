#include "../../include/common/cylindrical.hlsli"
#include "../../include/common/light_impl.hlsli"
#include "../../include/common/random.hlsli"
#include "../../include/common/sampling.hlsli"
#include "../../include/common/vmf.hlsli"
#include "../../include/raytracer_descriptor_set.hlsli"
#include "../../include/scene_descriptor_set.hlsli"
#include "../../raytracer/spt_interface.hlsli"
#include "../vbuffer/vbuffer.hlsli"
#include "include/pathguiding.hlsli"

[[vk::push_constant]]
cbuffer PushConstants {
    int2 resolution;
    int2 debugPixel;
    int mode;
    float scalar;
};

Texture2D<float4> u_vMFStatistics;
Texture2D<uint4> u_vBuffer;

RWTexture2D<float> u_PdfNormalizing;

[shader("raygeneration")]
void RgenMain() {
    // load the first hit from the vbuffer
    const int2 pixel = int2(DispatchRaysIndex().xy);
    const VBufferData vhit = UnpackVBufferData(u_vBuffer[debugPixel]);
    if (IsInvalid(vhit)) {
        u_PdfNormalizing[pixel] = 0.f; // error color
        return;
    }

    const float2 uv = (float2(pixel) + 0.5f) / float2(512);
    const float3 dir = CylindricalToUnitVector(UnitToCylindrical(uv));
    
    const vMFMLEStatistics vMFS = vMFMLEStatistics(u_vMFStatistics[debugPixel]);
    const vMFDistribution vMF = vMFDistribution(vMFS);
    const float importance = vMF.Pdf(dir);
    
    u_PdfNormalizing[pixel] = importance;
}

