#include "../../include/common/camera.hlsli"
#include "gbuffer_interface.hlsli"

enum ShowEnum {
    DiffuseAlbedo,
    GeometryNormal,
    ShadingNormal,
    MotionVector,
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

[shader("fragment")]
float4 fragmentMain(
    in float2 uv: Coord,
    in float3 bary: SV_Barycentrics) : SV_Target
{
    int2 pixelPosition = int2(uv * getViewportSize(gCamera));

    ShadingSurface surface = GetGBufferSurface(pixelPosition, gCamera);

    float3 color = float3(0.0);
    if (pConst.showEnum == ShowEnum::DiffuseAlbedo) {
        color = surface.diffuseAlbedo;
    } else if (pConst.showEnum == ShowEnum::GeometryNormal) {
        color = surface.geometryNormal * 0.5 + 0.5;
    } else if (pConst.showEnum == ShowEnum::ShadingNormal) {
        color = surface.normal * 0.5 + 0.5;
    } else if (pConst.showEnum == ShowEnum::MotionVector) {
        color = GetMotionVector(pixelPosition) * 1000;
    }

    if (!IsShadingSurfaceValid(surface)) {
        color = float3(0.0);
    }

    // float3 motionVector = GetMotionVector(pixelPosition);
    // motionVector = convertMotionVectorToPixelSpace(g_Const.view, g_Const.prevView, pixelPosition, motionVector);

    return float4(color, 1.0);
}