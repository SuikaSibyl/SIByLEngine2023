#include "../../raytracer/spt_interface.hlsli"
#include "../../raytracer/primitives/cone.hlsli"
#include "../../raytracer/primitives/sphere.hlsli"

#include "../../include/common/bitfield.hlsli"
#include "../../include/common/geometry.hlsli"
#include "../../include/common/random.hlsli"
#include "../../include/common/sampling.hlsli"
#include "../../include/raytracer_descriptor_set.hlsli"
#include "../../include/scene_descriptor_set.hlsli"
#include "include/conetrace_utils.hlsli"
#include "include/vxgi_interface.hlsli"


[[vk::push_constant]]
cbuffer PushConstants {
    struct PushConstants {
        uint2 resolution;
        uint sample_batch;
        uint flag;
        int2 debug_pixel;
        float coneLength;
        int voxelShown;
        int4 voxelChoosen;
        int debug_mode;
        float maximum_tan;
    } pConst;
};

RWTexture2D<float4> u_Color;
RWTexture2D<float4> u_Debug;

Texture3D<float2> uRadopaVox6D[6];
SamplerState uTex3DSampler;

struct DebugUniform {
    float3 position;
    uint geometryID;
    float3 direction;
    uint32_t validFlags;
    int coneID;
};
RWStructuredBuffer<DebugUniform> u_DebugUniform;

cbuffer VoxelizerUniform {
    VoxerlizerData gVD;
}

float deLine(in_ref(float3) ro, in_ref(float3) rd, in_ref(float3) a, in_ref(float3) b) {
    // http://www.sousakuba.com/Programming/gs_two_lines_intersect.html
    const float3 ab = normalize(b - a); const float3 ao = a - ro;
    const float d0 = dot(rd, ab); const float d1 = dot(rd, ao); const float d2 = dot(ab, ao);
    float len = (d0 * d1 - d2) / (1.0 - d0 * d0);
    len = clamp(len, 0.0, length(b - a));
    const float3 p = a + ab * len;
    return length(cross(p - ro, rd));
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

    if (HasHit(payload.hit) == false)
        return radiance;
    else {
        // GeometryInfo geometry = geometries[payload.hit.geometryID];
        // MaterialInfo material = materials[geometry.materialID];
        // float4 albedo = textures[material.baseOrDiffuseTextureIndex].Sample(payload.hit.texcoord, 0);

        // const PolymorphicLightInfo light = lights[0];
        // float3 lightColor = unpackLightColor(light);
        // float3 lightDir = Unorm32OctahedronToUnitVector(light.databyte0);

        // Ray shadowRay = SpawnRay(payload.hit, -lightDir);
        // bool occluded = TraceOccludeRay(shadowRay, RNG, SceneBVH);
        // float visibility = occluded ? 0.0f : 1.0f;
        // lightDir

        // radiance += albedo.xyz * lightColor * saturate(dot(payload.hit.geometryNormal, -lightDir)) * visibility;
        // throughput *= albedo.xyz;
    }

    // cosineSampleHemisphere
    // GeometryInfo geometry = geometries[payload.hit.geometryID];
    // MaterialInfo material = materials[geometry.materialID];
    
    float3 dir = normalize(payload.hit.geometryNormal + randomPointInSphere(RNG));
    Ray bounceRay = SpawnRay(payload.hit, dir);
    ray = ToRayDesc(bounceRay);
    payload.RND = RNG;
    TraceRay(SceneBVH, rayFlags, instanceMask, 0, 0, 0, ray, payload);
    
    if (HasHit(payload.hit) == false)
        return radiance;
    else {
        GeometryInfo geometry = geometries[payload.hit.geometryID];
        MaterialInfo material = materials[geometry.materialID];
        const float4 albedo = textures[material.baseOrDiffuseTextureIndex].Sample(payload.hit.texcoord);

        const PolymorphicLightInfo light = lights[0];
        float3 lightColor = unpackLightColor(light);
        float3 lightDir = Unorm32OctahedronToUnitVector(light.databyte0);

        Ray shadowRay = SpawnRay(payload.hit, -lightDir);
        bool occluded = TraceOccludeRay(shadowRay, RNG, SceneBVH);
        float visibility = occluded ? 0.0f : 1.0f;
        // // lightDir
        
        radiance += albedo.xyz * lightColor * saturate(dot(payload.hit.geometryNormal, -lightDir)) * visibility * k_pi;
    }
    return radiance;
}

float3 MatcapShading(in_ref(float3) normal) {
    const float dif = clamp(dot(normal, float3(0.57703)), 0.0, 1.0);
    const float amb = 0.5 + 0.5 * dot(normal, float3(0.0, 1.0, 0.0));
    return float3(0.2, 0.3, 0.4) * amb + float3(0.8, 0.7, 0.5) * dif;
}

float3 pattern(in float2 uv) {
    float3 col = float3(0.6);
    col += 0.4 * smoothstep(-0.01, 0.01, cos(uv.x * 0.5) * cos(uv.y * 0.5));
    col *= smoothstep(-1.0, -0.98, cos(uv.x)) * smoothstep(-1.0, -0.98, cos(uv.y));
    return col;
}

VOXEL_TYPE TraceConeDdebug(
    out int4 hitVoxels[8], out float4 hitPositions[8],
    in_ref(Texture3D<VOXEL_TYPE>) tex[6],
    in_ref(SamplerState) sampler,
    in_ref(float3) position,
    in_ref(float3) normal,
    in_ref(float3) direction,
    in_ref(float) tan_half_aperture,
    in_ref(VoxelTexInfo) info,
    in_ref(float) maxDistWS,
    in_ref(float) samplingFactor = 1.0f,
) {
    for (int i = 0; i < 8; ++i) {
        hitVoxels[i] = int4(-1);
    }
    // Compute visible face indices
    const int3 visibleFace = ComputeVoxelFaceIndices(direction);
    // world space grid voxel size
    float voxelWorldSize = 1.0 / (info.voxelScale * info.volumeDimension);
    // weight per axis for aniso sampling
    const float3 weight = direction * direction;
    // move further to avoid self collision
    float dist = voxelWorldSize; // move 1 voxel size to avoid self collision
    const float3 startPosition = position + normal * dist * 2;
    // final results
    float2 coneSample = float2(0.0f);
    float occlusion = 0.0f;
    float maxDistance = maxDistWS / info.voxelScale;
    // out of boundaries check
    float enter = 0.0;
    float leave = 0.0;
    // ray marching loop
    int i = 0;

    const int2 pixel = DispatchRaysIndex().xy;
    float4 debugInfo = float4(0);

    while (coneSample.y < 1.0f && dist <= maxDistance) {
        const float3 conePosition = startPosition + direction * dist;
        // cone expansion and respective mip level based on diameter
        const float diameter = 2.0f * tan_half_aperture * dist;
        const float mipLevel = log2(diameter / voxelWorldSize);
        // convert position to texture coord
        const float3 coord = WorldToVoxel(conePosition, info);
        // get directional sample from anisotropic representation
        VOXEL_TYPE anisoSample = AnistropicSample(tex, sampler, coord, weight, visibleFace, mipLevel);
        // front to back composition, also accumulate occlusion
        coneSample += (1.0f - coneSample[ALPHA_CHANNEL]) * anisoSample;
        // move further into volume
        dist += diameter * samplingFactor;
        // Add debug info
        if (pixel.y % 2 == 0) {
            if (pixel.x % 8 == i)
                debugInfo = float4(anisoSample.y);
        }
        else if (pixel.y % 2 == 1)
            if (pixel.x % 8 == i)
                debugInfo = float4(coord, mipLevel);

        if (i < 8) {
            hitPositions[i] = float4(conePosition, coneSample.y);
            hitVoxels[i++] = int4(int3(coord * (info.volumeDimension >> int(mipLevel))), int(mipLevel));
        }
    }
    // u_Debug[DispatchRaysIndex().xy] = debugInfo;

    return coneSample;
}

void DrawHitVoxels(
    in_ref(int4) hitVoxels[8],
    in_ref(Ray) primaryRay,
    inout_ref(float3) color,
    float4 hitPositions[8],
    in_ref(VoxelTexInfo) info
) {
    const float3 extent = gVD.aabbMax - gVD.aabbMin;
    const float extentMax = max(extent.x, max(extent.y, extent.z)) * 0.5 + 0.01;
    const float3 center = (gVD.aabbMax + gVD.aabbMin) * 0.5;

    Ray voxelRay = primaryRay;
    voxelRay.origin = ((primaryRay.origin.xyz - center) / extentMax + 1) * 0.5 * gVD.voxelSize; // [0, 64]^3

    float3 colors[8] = {
        float3(1, 0, 0),
        float3(0, 1, 0),
        float3(0, 0, 1),
        float3(1, 1, 0),
        float3(1, 0, 1),
        float3(0, 1, 1),
        float3(1, 0, 1),
        float3(0),
    };

    float voxelWorldSize = 1.0 / (info.voxelScale * info.volumeDimension);

    for (int i = 0; i < 8; ++i) {
        if (all(hitVoxels[i] == int4(-1))) break;
        // if (hitVoxels[i].w >= 6) break;
        // int3 coord = hitVoxels[i].xyz >> hitVoxels[i].w;
        // coord = coord << hitVoxels[i].w;
        // const AABB aabb = { coord, coord + (1 << hitVoxels[i].w) };
        // if (intersectTMin(voxelRay, aabb) > 0) {
        //     color += colors[hitVoxels[i].w] * 0.3;
        // }
        AABB aabb;
        if (pConst.voxelShown == 1) {
            const float extent = voxelWorldSize * (1 << hitVoxels[i].w) / 2;
            aabb.min = hitPositions[i].xyz - float3(extent);
            aabb.max = hitPositions[i].xyz + float3(extent);
        }
        else {
            aabb = VoxelToBound(hitVoxels[i].xyz, hitVoxels[i].w, info);
        }
        if (intersectTMin(primaryRay, aabb) > 0) {
            color += colors[hitVoxels[i].w] * 0.3;
        }
        
        if (hitSphere(hitPositions[i].xyz, 0.05f, primaryRay.origin, primaryRay.direction) > 0) {
            color = float3(hitPositions[i].w);
            break;
        }
    }
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
    float3 pixelColor = u_Color[pixel].rgb;
    RandomSamplerState RNG = InitRandomSampler(pixel, pConst.sample_batch);

    bool hasHit = false;
    float3 position;
    float3 normal;
    if ((pConst.flag & 0x3) == 1) {
        Ray debugRay = generateRay(
            pConst.debug_pixel,
            resolution,
            false,
            globalUniform.cameraData
        );
        RayDesc ray = ToRayDesc(debugRay);
        PrimaryPayload payload;
        payload.RND = RNG;
        uint instanceMask = 0xff;
        uint rayFlags = RAY_FLAG_NONE;
        TraceRay(SceneBVH, rayFlags, instanceMask, 0, 0, 0, ray, payload);
        RNG = payload.RND;
        position = payload.hit.position;
        normal = payload.hit.geometryNormal;
        hasHit = HasHit(payload.hit);
        // save debug hit info
        u_DebugUniform[0].position = position;
        u_DebugUniform[0].geometryID = payload.hit.geometryID;
        u_DebugUniform[0].direction = normal;
        u_DebugUniform[0].validFlags = SetBit(u_DebugUniform[0].validFlags, 0, hasHit);
    }
    else {
        // load debug hit info
        position = u_DebugUniform[0].position;
        normal = u_DebugUniform[0].direction;
        hasHit = GetBit(u_DebugUniform[0].validFlags, 0);
    }

    // Has hit some point
    if (hasHit == false) {
        u_Color[pixel] = float4(pixelColor, 1);
        return;
    }

    Ray primaryRay = generateRay(
        pixel,
        resolution,
        false,
        globalUniform.cameraData
    );

    float3x3 TBN = createONB(normal);
    // float3 diffuseConeNormals[5] = {
    //     mul(float3(0, 0, 1), TBN),
    //     mul(float3(+0.7071, 0, +0.7071), TBN),
    //     mul(float3(-0.7071, 0, +0.7071), TBN),
    //     mul(float3(0, +0.7071, +0.7071), TBN),
    //     mul(float3(0, -0.7071, +0.7071), TBN)
    // };
    float3 diffuseConeNormals[6] = {
        mul(float3(0, 0, 1), TBN),
        mul(float3(0.8660254f * +1.000000000000f, 0.8660254f * +0.000000000000f, +0.5), TBN),
        mul(float3(0.8660254f * +0.309016994375f, 0.8660254f * +0.951056516295f, +0.5), TBN),
        mul(float3(0.8660254f * -0.809016994370f, 0.8660254f * +0.587785252290f, +0.5), TBN),
        mul(float3(0.8660254f * -0.809016994370f, 0.8660254f * -0.587785252290f, +0.5), TBN),
        mul(float3(0.8660254f * +0.309016994370f, 0.8660254f * -0.951056516295f, +0.5), TBN),
    };

    Cone cones[6];
    for (int i = 0; i < 6; ++i) {
        cones[i] = {
            position,
            0.8660254f, // cos(30)
            diffuseConeNormals[i],
            pConst.coneLength};
    }
    
    int hitConeIndex = -1;
    float4 hitScene = float4(-1, 0, 0, 0);
    for (int i = 0; i < 6; ++i) {
        float4 hitCone = RayConeIntersection(cones[i], primaryRay);
        if(hitCone.x > 0 && (hitScene.x < 0 || hitCone.x < hitScene.x)) {
            hitScene = hitCone;
            hitConeIndex = i;
        }
    }
    if (hitScene.x > 0) {
        pixelColor = MatcapShading(hitScene.yzw);
        // grid pattern
        const float3 pos = primaryRay.origin + hitScene.x * primaryRay.direction;
        const float3x3 obn = createONB(cones[hitConeIndex].axis);
        const float3 q = mul(obn, (pos - cones[hitConeIndex].position));
        pixelColor *= pattern(float2(16.0, 64.0) * float2(atan2(q.y, q.x), q.z));
    }

    // Updating selecting cones
    // The selected cone is highlighted by red color
    // ------------------------------------------------------------------------
    if ((pConst.flag & 0x3) == 2 && all(pConst.debug_pixel == pixel && hitConeIndex >= 0)) {
        u_DebugUniform[0].validFlags = SetBit(u_DebugUniform[0].validFlags, 1, true);
        u_DebugUniform[0].coneID = hitConeIndex;
    }
    const int selectedConeID = GetBit(u_DebugUniform[0].validFlags, 1) ?
        clamp(u_DebugUniform[0].coneID, 0, 5) : 0;
    if (hitConeIndex == selectedConeID) {
        pixelColor.r = 1;
    }

    float3 selectedConeNormal = cones[selectedConeID].axis;
    float tan_half_aperture = 0.57735026919f;
    float de = deLine(primaryRay.origin, primaryRay.direction, position, position + selectedConeNormal * 5);
    pixelColor = lerp(float3(1.0, 0.1, 0.1), pixelColor, smoothstep(0.01, 0.01, de));

    VoxelTexInfo info = CreateVoxelTexInfo(gVD);

    if (pConst.debug_mode == 0) {
        // difuse cone
    }
    else if (pConst.debug_mode == 1) {
    }
    else if (pConst.debug_mode == 2) {
        // to voxel
        AABB aabb = VoxelToBound(pConst.voxelChoosen.xyz, pConst.voxelChoosen.w, info);
        u_Debug[DispatchRaysIndex().xy] = float4(pConst.voxelChoosen.xyz, pConst.voxelChoosen.w);
        float3 dir = (aabb.max + aabb.min) * 0.5 - position;
        const float dist = length(dir);
        selectedConeNormal = dir / dist;
        float local_radius = distance(aabb.max, aabb.min) * 0.81649658 * 0.5;
        tan_half_aperture = min(local_radius / dist, pConst.maximum_tan);
    }

    int4 hitVoxels[8];
    float4 hitPositions[8];
    TraceConeDdebug(hitVoxels, hitPositions,
                    uRadopaVox6D, uTex3DSampler, position,
                    normal, selectedConeNormal,
                    tan_half_aperture, info, 50.0f);

    // Restore primary ray
    Ray primaryRay2 = generateRay(
        pixel,
        resolution,
        false,
        globalUniform.cameraData
    );
    // primaryRay2.origin = primaryRay.origin;
    // primaryRay2.direction = primaryRay.direction;
    DrawHitVoxels(hitVoxels, primaryRay2, pixelColor, hitPositions, info);

    // Cone cone = generateCone(
    //     pixel,
    //     resolution,
    //     globalUniform.cameraData
    // );

    // float3 pixelColor = unidirectional_path_tracing(primaryRay, RNG);
    u_Color[pixel] = float4(pixelColor, 1);
}