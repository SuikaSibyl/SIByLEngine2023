#include "../../include/common/camera.hlsli"
#include "gbuffer_interface.hlsli"
#include "gbuffer_prev_interface.hlsli"

enum ShowEnum {
    DiffuseAlbedo,
    GeometryNormal,
    ShadingNormal,
    MotionVector,
    TemporalDiffuseAlbedo,
    AlbedoDiff,
};

[[vk::push_constant]]
cbuffer PushConstants {
    struct PushConstants {
        ShowEnum showEnum;
    } pConst;
};

cbuffer CameraDataUniforms {
    CameraData gCamera;
};
cbuffer PrevCameraDataUniforms {
    CameraData gPrevCamera;
};

[shader("fragment")]
float4 fragmentMain(
    in float2 uv: Coord,
    in float3 bary: SV_Barycentrics) : SV_Target
{
    int2 pixelPosition = int2(uv * getViewportSize(gCamera));
    float3 motionVector = GetMotionVector(pixelPosition);
    int2 prevPixelPosition = pixelPosition + int2(motionVector.xy);

    ShadingSurface surface = GetGBufferSurface(pixelPosition, gCamera);
    ShadingSurface prevSurface = GetPrevGBufferSurface(prevPixelPosition, gPrevCamera);

    float3 color = float3(0.0);
    if (pConst.showEnum == ShowEnum::DiffuseAlbedo) {
        color = surface.diffuseAlbedo;
        if (!IsShadingSurfaceValid(surface))
            color = float3(0.0);
    } else if (pConst.showEnum == ShowEnum::GeometryNormal) {
        color = surface.geoNormal * 0.5 + 0.5;
        if (!IsShadingSurfaceValid(surface))
            color = float3(0.0);
    } else if (pConst.showEnum == ShowEnum::ShadingNormal) {
        color = surface.normal * 0.5 + 0.5;
        if (!IsShadingSurfaceValid(surface))
            color = float3(0.0);
    } else if (pConst.showEnum == ShowEnum::MotionVector) {
        color = GetMotionVector(pixelPosition) * 1000;
        if (!IsShadingSurfaceValid(surface))
            color = float3(0.0);
    } else if (pConst.showEnum == ShowEnum::TemporalDiffuseAlbedo) {
        color = (IsShadingSurfaceValid(surface) & IsShadingSurfaceValid(prevSurface)) ? prevSurface.diffuseAlbedo : float3(0., 0., 0);
        if (!IsShadingSurfaceValid(surface))
            color = float3(0.0);
    } else if (pConst.showEnum == ShowEnum::AlbedoDiff) {
        float3 valid_this = IsShadingSurfaceValid(surface) ? surface.diffuseAlbedo : float3(0, 0, 0);
        float3 valid_prev = (IsShadingSurfaceValid(surface) & IsShadingSurfaceValid(prevSurface)) ? prevSurface.diffuseAlbedo : float3(0, 0, 0);
        color = abs(valid_this - valid_prev) * 10;
    }

    // float3 motionVector = GetMotionVector(pixelPosition);
    // motionVector = convertMotionVectorToPixelSpace(g_Const.view, g_Const.prevView, pixelPosition, motionVector);

    return float4(color, 1.0);
}