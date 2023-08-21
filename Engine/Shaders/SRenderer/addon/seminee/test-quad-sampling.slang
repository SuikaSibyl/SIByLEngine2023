#include "../../include/common/geometry.hlsli"
#include "../../include/common/sampling.hlsli"
#include "../ltc/ltc_interface.hlsli"
#include "seminee-config.hlsli"
#include "seminee_interface.hlsli"
#include "seminee_internal.hlsli"

RWTexture2D<float4> u_color;
Sampler2D<float> u_HoriClipLUT;

[[vk::push_constant]]
cbuffer PushConstants {
    struct PushConstants {
        uint2 resolution;
        uint sample_batch;
        uint padding;
        float3 pos_emitter;
        uint separator;
        float3 dir_emitter;
        int sampleStragegy;
    } pConst;
};

cbuffer CameraBuffer {
    CameraData gCamera;
};

[shader("compute")]
[numthreads(16, 16, 1)]
void ComputeMain(uint3 gid: SV_DispatchThreadID) {
    // The resolution of the image, which is the same as the launch size:
    const int2 resolution = int2(pConst.resolution);
    const int2 pixel = int2(gid.xy);
    
    RandomSamplerState RNG = InitRandomSampler(pixel, pConst.sample_batch);

    Ray primaryRay = generateRay(
        pixel,
        resolution,
        false,
        gCamera
    );

    QuadObject plane;
    plane.pMin = float2(-15.0f, -15.0f);
    plane.pMax = float2(15.0f, 15.0f);
    plane.rotation = getRotationMatrixFromAToB(float3(0, 1, 0), float3(0.f, 0.f, 1.f));
    plane.translation = float3(0);
    plane.intensity = 1.f;

    QuadObject plane_emitter;
    plane_emitter.pMin = float2(-3.0f, -3.0f);
    plane_emitter.pMax = float2(3.0f, 3.0f);
    plane_emitter.rotation = getRotationMatrixFromAToB(normalize(pConst.dir_emitter), float3(0.f, 0.f, 1.f));
    plane_emitter.translation = pConst.pos_emitter;
    plane_emitter.intensity = 1.f;

    QuadObject scene[4];
    scene[0] = plane;
    scene[1] = plane_emitter;
    scene[2].intensity = 0;
    scene[3].intensity = 0;
    
    float t = -1;
    float3 color = float3(0);
    
    int hitID = HitQuads4(scene, primaryRay, t);
    
    if (hitID == 0) {
        const float3 hitPoint = primaryRay.origin + primaryRay.direction * t;
        float2 uv = float2(GetNextRandom(RNG), GetNextRandom(RNG));
        
        const float3 localHitPoint = plane_emitter.toLocal(hitPoint);
        const float3 localNormal = plane_emitter.toLocalDir(float3(0, 1, 0));

        if (localHitPoint.z > 0) {
            if (pConst.sampleStragegy >= 2) {
                update_x_bounds(localHitPoint, localNormal, plane_emitter);
                update_y_bounds(localHitPoint, localNormal, plane_emitter);
            }

            if (pixel.x < pConst.separator) {
                // Area sampling
                float3 dir;
                float pdf;
                float3 pos;
                SampleQuadArea(localHitPoint, plane_emitter.pMin, plane_emitter.pMax, uv, dir, pos, pdf);
                if (pdf > 0.f) {
                    // float3 dirW = plane_emitter.toWorldDir(dir);
                    // float3 originW = hitPoint + float3(0, 0.002, 0);
                    // Ray shadowRay = { originW, 0, dirW.x, 1000 };
                    color += max(dot(normalize(dir), localNormal), 0) / pdf;
                }
                
                // Hemisphere sampling
                // const float3 rayDir = normalize(float3(0, 1, 0) + randomPointInSphere(float2(GetNextRandom(RNG), GetNextRandom(RNG))));
                // Ray secondaryRay = { hitPoint, 0,
                //                      rayDir, 1000 };
                // float t;
                // if (RayQuadObjectIntersection(secondaryRay, plane_emitter, t1)) {
                //     if (t1 > 0 && t1 < k_inf) {
                //         color += k_pi;
                //     }
                // }
            }
            else {
                // // Spherical sampling
                // SphQuad sphQuad = CreateSphQuad(plane_emitter, localHitPoint);
                // float3 dir;
                // float pdf;
                // SampleSphQuad(localHitPoint, sphQuad, uv, dir, pdf);
                // if (pdf > 0.f) {
                //     // float3 dirW = plane_emitter.toWorldDir(dir);
                //     // float3 originW = hitPoint + float3(0, 0.002, 0);
                //     // Ray shadowRay = { originW, 0, dirW.x, 1000 };
                //     color += max(dot(normalize(dir), localNormal), 0) / pdf;
                // }
                float3x3 Minv = {
                    1, 0, 0,
                    0, 1, 0,
                    0, 0, 1
                };
                float3 points[4];
                points[0] = float3(-3, -3, 0);
                points[1] = float3(+3, -3, 0);
                points[2] = float3(+3, +3, 0);
                points[3] = float3(-3, +3, 0);
                
                float3 view = normalize(gCamera.posW - hitPoint);
                color += LTCEvaluate_Quad(u_HoriClipLUT,
                                          hitPoint, float3(0, 1, 0), view,
                                          Minv, points, false);
            }
        }
    }
    else if (hitID == 1) {
        color = float3(1, 1, 1);
    }
    
    u_color[pixel] = float4(color, 1);
}