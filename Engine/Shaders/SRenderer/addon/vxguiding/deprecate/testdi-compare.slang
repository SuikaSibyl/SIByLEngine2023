#include "../../include/common/random.hlsli"
#include "../../include/common/sampling.hlsli"
#include "../../include/raytracer_descriptor_set.hlsli"
#include "../../include/scene_descriptor_set.hlsli"
#include "../../raytracer/primitives/quad.hlsli"
#include "../../raytracer/primitives/sphere.hlsli"
#include "../../raytracer/spt_interface.hlsli"
#include "../vxgi/include/conetrace_utils.hlsli"
#include "include/vxguiding_interface.hlsli"
#include "testdi_interface.hlsli"


[[vk::push_constant]]
cbuffer PushConstants {
    struct PushConstants {
        uint2 resolution;
        uint sample_batch;
        uint sampleStragegy;
    } pConst;
};

[[vk::binding(0, 2)]]
RWTexture2D<float4> u_Color;

RWTexture2D<float4> u_Debug;


cbuffer TestDISettingUniform {
    TestDISetting uTestDISetting;
}

Texture3D<float2> uRadopaVox6D[6];

cbuffer VoxelizerUniform {
    VoxerlizerData gVD;
}

float middle(float3 vector) {
    float smallestElement = min(vector.x, min(vector.y, vector.z));
    float largestElement = max(vector.x, max(vector.y, vector.z));
    float middleElement = vector.x + vector.y + vector.z - smallestElement - largestElement;
    return middleElement;
}

bool IsEdge(float3 pos, float threshold) {
    float3 offset = abs(fmod(pos, 1));
    offset = (offset < float3(0.5)) ? offset : 1 - offset;
    if (middle(offset) < threshold) {
        return true;
    }
    return false;
}

float3 VS2WS(int3 voxel, float extentMax, float3 center, int mipLevel) {
    float3 cs = ((voxel + float3(0.5)) / (gVD.voxelSize >> mipLevel)) * 2 - 1;
    return cs * extentMax + center;
}

int2 WS2SS(float3 position) {
    float4 cs = mul(float4(position, 1), globalUniform.cameraData.viewProjMat);
    cs.xyz /= cs.w;
    return int2((cs.xy + 1) * 0.5 * pConst.resolution);
}

float3 unidirectional_path_tracing(in_ref(Ray) primaryRay, inout_ref(RandomSamplerState) RNG) {
    RayDesc ray = ToRayDesc(primaryRay);
    
    PrimaryPayload payload;
    payload.RND = RNG;
    // payload.occluded = false;

    uint instanceMask = 0xff;
    uint rayFlags = RAY_FLAG_NONE;
    TraceRay(SceneBVH, rayFlags, instanceMask, 0, 0, 0, ray, payload);
    RNG = payload.RND;

    float3 radiance = float3(0, 0, 0);
    float3 throughput = float3(1, 1, 1);

    float t;
    if (RayQuadIntersection(primaryRay, uTestDISetting, t)) {
        if (HasHit(payload.hit) && t > 0) {
            float tMin = distance(payload.hit.position, primaryRay.origin);
            if (t < tMin) {
                return float3(1, 1, 1);
            }
        }
    }
    
    if (HasHit(payload.hit) == false)
        return radiance;

    VoxelTexInfo info = CreateVoxelTexInfo(gVD);

    // If primary ray hit scene geometry.
    float2 uv = float2(GetNextRandom(RNG), GetNextRandom(RNG));
    if (pConst.sampleStragegy == 0) {
        // NEE
        const float3 localHitPoint = mul(float4(payload.hit.position, 1.), uTestDISetting.localMat).xyz;
        const float3 localNormal = mul(float4(payload.hit.geometryNormal, 0.), uTestDISetting.localMat).xyz;

        float3 dir;
        float3 pos;
        float pdf;
        SampleQuadArea(localHitPoint, uTestDISetting.pMin, uTestDISetting.pMax, uv, dir, pos, pdf);
        if (pdf > 0.f) {
            float3 shadowDir = mul(float4(dir, 0.), uTestDISetting.worldMat).xyz;
            float3 shadowPos = mul(float4(pos, 1.), uTestDISetting.worldMat).xyz;
            Ray shadowRay = SpawnRay(payload.hit, shadowDir);
            shadowRay.tMax = clamp(distance(shadowRay.origin, shadowPos) - 0.01f, 0.0f, k_inf);
            bool occluded = TraceOccludeRay(shadowRay, RNG, SceneBVH);
            float visibility = occluded ? 0.0f : 1.0f;

            radiance += max(dot(normalize(dir), localNormal), 0) * visibility * uTestDISetting.intensity / pdf;
        }
    }
    else if (pConst.sampleStragegy == 1) {
        // BSDF
        float3 shadowDir = normalize(payload.hit.geometryNormal + randomPointInSphere(RNG));
        Ray shadowRay = SpawnRay(payload.hit, shadowDir);
        TraceRay(SceneBVH, rayFlags, instanceMask, 0, 0, 0, ToRayDesc(shadowRay), payload);
        float tQuad;
        if (RayQuadIntersection(shadowRay, uTestDISetting, tQuad) && tQuad > 0) {
            if (!HasHit(payload.hit) || tQuad < distance(payload.hit.position, shadowRay.origin)) {
                radiance += uTestDISetting.intensity * k_pi;
            }
        }
    }
    else if (pConst.sampleStragegy == 2) {
        // Semi NEE - sphere
        const float3 sphereCenter = mul(float4(0, 0, 0, 1.), uTestDISetting.worldMat).xyz;
        const float sphereRadius = length(uTestDISetting.pMax);
        // // Sample from sphere
        const float3 sphPosition = sampleSphere(sphereCenter, sphereRadius, payload.hit.position, uv);
        const float3 shadowDir = normalize(sphPosition - payload.hit.position);
        const float pdf = sampleSpherePdf(sphereCenter, sphereRadius, payload.hit.position, sphPosition);

        const float3 geometryNormal = normalize(payload.hit.geometryNormal);
        Ray shadowRay = SpawnRay(payload.hit, shadowDir);
        TraceRay(SceneBVH, rayFlags, instanceMask, 0, 0, 0, ToRayDesc(shadowRay), payload);
        float tQuad;
        if (RayQuadIntersection(shadowRay, uTestDISetting, tQuad) && tQuad > 0) {
            if (!HasHit(payload.hit) || tQuad < distance(payload.hit.position, shadowRay.origin)) {
                radiance += uTestDISetting.intensity * saturate(dot(geometryNormal, shadowDir)) / pdf;
            }
        }
    }
    else if (pConst.sampleStragegy == 3) {
        // Semi NEE - voxel2sphere
        int mipLevelShown = 5;
        const int3 voxelSize = gVD.voxelSize >> mipLevelShown;
        const float3 extent = gVD.aabbMax - gVD.aabbMin;
        const float extentMax = max(extent.x, max(extent.y, extent.z)) * 0.5 + 0.01;
        const float3 center = (gVD.aabbMax + gVD.aabbMin) * 0.5;

        float pdf = 1.f;
        int3 selectedVox = SampleByLuminance(uRadopaVox6D, payload.hit.position.xyz, mipLevelShown, info, RNG, pdf);
        if (all(selectedVox == int3(-1))) return float3(1, 0, 1); // invalid voxel
        u_Debug[DispatchRaysIndex().xy] = float4(selectedVox, pdf);

        float voxelExtent = extentMax / 64;
        float3 voxelCenter = VS2WS(selectedVox, extentMax, center, 0);
        const float sphereRadius = length(float3(voxelExtent));
        const float3 sphPosition = sampleSphere(voxelCenter, sphereRadius, payload.hit.position, uv);
        AABB aabb = { voxelCenter - voxelExtent, voxelCenter + voxelExtent };
        const float3 shadowDir = normalize(sphPosition - payload.hit.position);
        const float sph_pdf = sampleSpherePdf(voxelCenter, sphereRadius, payload.hit.position, sphPosition);

        const float3 geometryNormal = normalize(payload.hit.geometryNormal);
        Ray shadowRay = SpawnRay(payload.hit, shadowDir);
        TraceRay(SceneBVH, rayFlags, instanceMask, 0, 0, 0, ToRayDesc(shadowRay), payload);
        float tQuad;
        if (RayQuadIntersection(shadowRay, uTestDISetting, tQuad) && tQuad > 0) {
            if (!HasHit(payload.hit) || tQuad < distance(payload.hit.position, shadowRay.origin)) {
                float3 hitQuadPos = shadowRay.origin + shadowRay.direction * tQuad;
                if (all(hitQuadPos >= aabb.min) && all(hitQuadPos <= aabb.max)) {
                    radiance += uTestDISetting.intensity * saturate(dot(geometryNormal, shadowDir)) / (pdf * sph_pdf);
                }
            }
        }

        if (hitSphere(voxelCenter, sphereRadius, primaryRay.origin, primaryRay.direction) > 0) {
            radiance = lerp(radiance, float3(1, 0, 0), 0.5);
        }
    }
    else if (pConst.sampleStragegy == 4) {
        // Semi NEE - voxel2cube
        int mipLevelShown = 5;
        const int3 voxelSize = gVD.voxelSize >> mipLevelShown;
        const float3 extent = gVD.aabbMax - gVD.aabbMin;
        const float extentMax = max(extent.x, max(extent.y, extent.z)) * 0.5 + 0.01;
        const float3 center = (gVD.aabbMax + gVD.aabbMin) * 0.5;
        const float3 rayPos = ((payload.hit.position.xyz - center) / extentMax + 1) * 0.5 * voxelSize; // [-1, 1]^3
        
        float pdf = 1.f;
        int3 selectedVox = SampleByLuminance(uRadopaVox6D, payload.hit.position.xyz, mipLevelShown, info, RNG, pdf);

        if (all(selectedVox == int3(-1))) return float3(1, 0, 1); // invalid voxel
        u_Debug[DispatchRaysIndex().xy] = float4(selectedVox, pdf);

        float voxelExtent = extentMax / 64;
        float3 voxelCenter = VS2WS(selectedVox, extentMax, center, 0);
        AABB aabb = { voxelCenter - voxelExtent, voxelCenter + voxelExtent };

        float sph_pdf;
        float3 shadowDir = SampleSphericalVoxel(
            voxelCenter,
            voxelExtent,
            payload.hit.position,
            float3(GetNextRandom(RNG), uv),
            sph_pdf);
        
        const float3 geometryNormal = normalize(payload.hit.geometryNormal);
        Ray shadowRay = SpawnRay(payload.hit, shadowDir);
        TraceRay(SceneBVH, rayFlags, instanceMask, 0, 0, 0, ToRayDesc(shadowRay), payload);
        float tQuad;
        if (RayQuadIntersection(shadowRay, uTestDISetting, tQuad) && tQuad > 0) {
            if (!HasHit(payload.hit) || tQuad < distance(payload.hit.position, shadowRay.origin)) {
                float3 hitQuadPos = shadowRay.origin + shadowRay.direction * tQuad;
                if (all(hitQuadPos >= aabb.min) && all(hitQuadPos <= aabb.max)) {
                    radiance += uTestDISetting.intensity * saturate(dot(geometryNormal, shadowDir)) / (pdf * sph_pdf);
                }
            }
        }
        
        // if (hitSphere(voxelCenter, sphereRadius, primaryRay.origin, primaryRay.direction) > 0) {
        //     radiance = lerp(radiance, float3(1, 0, 0), 0.5);
        // }
    }
    // u_Color[pixel]
    
    return radiance;
}

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

    float3 pixelColor = unidirectional_path_tracing(primaryRay, RNG);
    u_Color[pixel] = float4(pixelColor, 1);
}