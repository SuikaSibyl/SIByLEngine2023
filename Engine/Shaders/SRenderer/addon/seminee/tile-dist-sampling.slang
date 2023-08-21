#include "../../include/common/random.hlsli"
#include "../../include/common/sampling.hlsli"
#include "../../include/raytracer_descriptor_set.hlsli"
#include "../../include/scene_descriptor_set.hlsli"
#include "../../raytracer/spt_interface.hlsli"
#include "seminee-config.hlsli"
#include "seminee_interface.hlsli"
#include "seminee_internal.hlsli"

[[vk::push_constant]]
cbuffer PushConstants {
    struct PushConstants {
        uint2 resolution;
        uint sample_batch;
        int sample_hint;
    } pConst;
};

RWTexture2D<float4> u_Color;
RWTexture2D<float4> u_Debug;
RWStructuredBuffer<VPLData> u_VPLData;
RWStructuredBuffer<int4> u_SelectedVPL;

// bool DrawSampleFromQQ(
//     in_ref(QuadObject) quads[4],
//     in_ref(float3) position, in_ref(float3) normal,
//     out_ref(float3) dir, out_ref(float) pdf,
//     inout_ref(RandomSamplerState) RNG)
// {
//     SphQuad sphQuads[4];

//     float4 pmf = float4(0, 0, 0, 0);
//     for (uint i = 0; i < 4; i++) {
//         // skip invalid quad
//         if (quads[i].intensity < 0) continue;
//         // May alternatively use LTC here
//         const float3 localPosition = quads[i].toLocal(position);
//         const float3 localNormal = quads[i].toLocalDir(normal);
//         update_x_bounds(localPosition, localNormal, quads[i]);
//         update_y_bounds(localPosition, localNormal, quads[i]);
//         sphQuads[i] = CreateSphQuad(quads[i], localPosition);
//         pmf[i] = quads[i].intensity > 0 ? quads[i].intensity * sphQuads[i].S : 0;
//     }
//     float4 cdf = pmf;
//     for (int i = 0; i < 4; i++) {
//         if (i > 0) {
//             cdf[i] += cdf[i - 1];
//         }
//     }
//     pmf /= cdf[3];
//     cdf /= cdf[3];

//     u_Debug[int2(DispatchRaysIndex().xy)] = float4(pmf);

//     int selected_quad = -1;
//     float r = GetNextRandom(RNG);
//     for (int i = 0; i < 4; i++) {
//         if (cdf[i] > r) {
//             selected_quad = i;
//         }
//     }

//     if (selected_quad == -1 || pmf[selected_quad] == 0) {
//         return false;
//     }

//     selected_quad = 0;
//     const float3 localPosition = quads[selected_quad].toLocal(position);
//     const float3 localNormal = quads[selected_quad].toLocalDir(normal);
//     if (localPosition.z > 0) {
//         float2 uv = float2(GetNextRandom(RNG), GetNextRandom(RNG));
//         if (pConst.sample_hint != 0) {
//             SampleSphQuad(localPosition, sphQuads[selected_quad], uv, dir, pdf);
//         }
//         else {
//             SampleQuadArea(localPosition, quads[selected_quad].pMin, quads[selected_quad].pMax, uv, dir, pdf);
//         }
//         dir = quads[selected_quad].toWorldDir(dir);
//         // pdf *= pmf[selected_quad];
//         return true;
//     }
//     return false;
// }

bool DrawSampleFromQQ(
    inout_ref(QuadObject) quads[4],
    in_ref(float3) position, in_ref(float3) normal,
    out_ref(float3) dir, out_ref(float) pdf,
    inout_ref(RandomSamplerState) RNG,
    bool useVPLMIS,
    in_ref(float3) vplDir, out_ref(float) vplPdf)
{
    SphQuad sphQuads[4];
    float4 cdf = float4(0, 0, 0, 0);
    for (uint i = 0; i < 4; i++) {
        // skip invalid quad
        if (quads[i].intensity < 0) continue;
        // May alternatively use LTC here
        const float3 localPosition = quads[i].toLocal(position);
        const float3 localNormal = quads[i].toLocalDir(normal);
        update_x_bounds(localPosition, localNormal, quads[i]);
        update_y_bounds(localPosition, localNormal, quads[i]);
        sphQuads[i] = CreateSphQuad(quads[i], localPosition);
        if (isnan(sphQuads[i].S)) quads[i].intensity = 0;
        const bool valid = sphQuads[i].S > 0 && (!isnan(sphQuads[i].S));
        cdf[i] = quads[i].intensity > 0 ? quads[i].intensity * sphQuads[i].S : 0;
        if (!valid) cdf[i] = 0;
    }
    for (uint i = 0; i < 4; i++) {
        if (i > 0) {
            cdf[i] += cdf[i - 1];
        }
    }
    for (int i = 0; i < 4; i++) {
        cdf[i] /= cdf[3];
    }

    int selected_quad = -1;
    float r = GetNextRandom(RNG);
    for (int i = 0; i < 4; i++) {
        if (cdf[i] > r) {
            selected_quad = i;
            break;
        }
    }

    if (useVPLMIS) {
        float tmin;
        Ray vplRay = { position, 0, vplDir, k_inf };
        int hitID = HitQuads4(quads, vplRay, tmin);
        if (hitID == -1) vplPdf = 0;
        else {
            if (cdf[hitID] == 0) vplPdf = 0;
            // TODO
            // now only consider sph samp first
            else {
                vplPdf = cdf[hitID] / sphQuads[hitID].S;
            }
        }
    }

    if (selected_quad == -1 || cdf[selected_quad] == 0) {
        return false;
    }

    // selected_quad = 0;
    QuadObject theQuad = quads[selected_quad];
    const float3 localPosition = theQuad.toLocal(position);
    const float3 localNormal = normalize(theQuad.toLocalDir(normal));

    // if (pConst.sample_hint == 2) {
    //     update_x_bounds(localPosition, localNormal, theQuad);
    //     update_y_bounds(localPosition, localNormal, theQuad);
    // }
    // else if (pConst.sample_hint > 2) {
    //     update_x_bounds(localPosition, -localNormal, theQuad);
    //     update_y_bounds(localPosition, -localNormal, theQuad);
    // }

    if (localPosition.z > 0) {
        float2 uv = float2(GetNextRandom(RNG), GetNextRandom(RNG));
        if (pConst.sample_hint != 0) {
            SphQuad sphQuad = CreateSphQuad(theQuad, localPosition);
            SampleSphQuad(localPosition, sphQuad, uv, dir, pdf);
        }
        else {
            float3 pos;
            SampleQuadArea(localPosition, theQuad.pMin, theQuad.pMax, uv, dir, pos, pdf);
        }
        dir = theQuad.toWorldDir(dir);

        Ray ray = { position, 0, dir, k_inf };
        float tmin;
        int hitID = HitQuads4(quads, ray, tmin);
        if (hitID != selected_quad) {
            return false;
        }

        const float tmp = pdf;
        pdf *= cdf[selected_quad];

        return true;
    }
    return false;
}

void LoadQQ(in_ref(int4) selected, out_ref(QuadObject) quads[4]) {
    if (selected.y == -1) {
        u_Debug[int2(DispatchRaysIndex().xy)] = float4(1, 0, 1, 1);
    }
    else {
        u_Debug[int2(DispatchRaysIndex().xy)] = float4(1);
    }
    for (int i = 0; i < 4; ++i) {
        int id = selected[i];
        if (id == -1) {
            // set invalid quad
            quads[i].intensity = -1;
            continue;
        }
        else {
            const VPLData vpl = u_VPLData[id];
            const float3 center = (vpl.getAABBMin() + vpl.getAABBMax()) * 0.5f;
            const float3 variance = vpl.getVariance();
            const float3 normal = vpl.getNormW();
            const float3 position = vpl.getPosW();

            //   const float2 xy = variance.xy * rect[assembledVertex.vertexId].xy;
            const float3x3 R = getRotationMatrixFromAToB(normal, float3(0.f, 0.f, 1.f));
            const float ex = vpl.ex;
            const float ey = vpl.ey;

            quads[i].translation = position;
            quads[i].rotation = R;
            quads[i].pMin = float2(-ex, -ey);
            quads[i].pMax = float2(+ex, +ey);
            quads[i].intensity = luminance(vpl.getColor());
        }
    }
}

float3 unidirectional_path_tracing(in_ref(Ray) primaryRay, inout_ref(RandomSamplerState) RNG) {
    const int2 resolution = int2(pConst.resolution);
    const int2 pixel = int2(DispatchRaysIndex().xy);

    int2 tileID = computeTileID(pixel, TILE_SIZE);
    int2 tileResolution = computeTileResolution(int2(1280, 720), TILE_SIZE);
    int tileIndex = tileID.y * tileResolution.x + tileID.x;

    QuadObject quads[4];
    const int4 selected = u_SelectedVPL[tileIndex];
    LoadQQ(selected, quads);

    float3 DebugColor = float3(0);
    if (all(selected == int4(-1))) {
        DebugColor = float3(1, 0, 1);
    }
    else {
        DebugColor = float3(0.5);
    }
    u_Debug[int2(DispatchRaysIndex().xy)] = float4(DebugColor, 1.);

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

    if (HasHit(payload.hit) == false) {
        return radiance;
    }
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
        // // lightDir

        // radiance += albedo.xyz * lightColor * saturate(dot(payload.hit.geometryNormal, -lightDir)) * visibility;
        // throughput *= albedo.xyz;
    }

    float3 dir;
    float pdf;
    PrimaryPayload ob_payload;
    float3 gi_luminance = float3(0);
    float3 value = float3(0);

    // Load VPL
    int tileOffset = tileIndex * TILE_SIZE * TILE_SIZE;
    int2 subtileID = computeSubtileID(pixel, TILE_SIZE);
    int subtileOffset = subtileID.y * TILE_SIZE + subtileID.x;
    int globalOffset = tileOffset + subtileOffset;
    VPLData vpl = u_VPLData[globalOffset];

    float vplPdf = 0;
    if (DrawSampleFromQQ(quads, payload.hit.position, payload.hit.geometryNormal, dir, pdf, RNG,
                         !vpl.isInvalid(), normalize(vpl.getPosW() - payload.hit.position), vplPdf) && pdf > 0) {
        Ray secondaryRay = SpawnRay(payload.hit, dir);
        TraceRay(SceneBVH, rayFlags, instanceMask, 0, 0, 0, ToRayDesc(secondaryRay), ob_payload);
        if (HasHit(ob_payload.hit) == false) {
            u_Debug[int2(DispatchRaysIndex().xy)] = float4(1, 1, 0, 1);
        }
        else {
            GeometryInfo geometry = geometries[ob_payload.hit.geometryID];
            MaterialInfo material = materials[geometry.materialID];
            float4 albedo = textures[material.baseOrDiffuseTextureIndex].Sample(ob_payload.hit.texcoord, 0);

            const PolymorphicLightInfo light = lights[0];
            float3 lightColor = unpackLightColor(light);
            float3 lightDir = Unorm32OctahedronToUnitVector(light.databyte0);

            Ray shadowRay = SpawnRay(ob_payload.hit, -lightDir);
            bool occluded = TraceOccludeRay(shadowRay, RNG, SceneBVH);
            float visibility = occluded ? 0.0f : 1.0f;
            // lightDir

            float dotNL = saturate(dot(payload.hit.geometryNormal, dir));
            gi_luminance = albedo.xyz * lightColor * saturate(dot(ob_payload.hit.geometryNormal, -lightDir)) * visibility;
            float p1 = pdf;
            float p2 = dotNL / k_pi;
            float w2 = p1 * p1 / (p1 * p1 + p2 * p2);
            value += gi_luminance * dotNL / pdf;
        }
    }

    // if (!vpl.isInvalid()) {
    //     if (vplPdf <= 0) {
    //         value += vpl.getColor() * k_pi;
    //         u_Debug[int2(DispatchRaysIndex().xy)] = float4(1, 0, 0, 1);
    //     }
    //     else {
    //         float p1 = vplPdf;
    //         float p2 = 1. / k_pi;
    //         float w1 = p2 * p2 / (p1 * p1 + p2 * p2);
    //         if (isnan(p1)) {
    //             u_Debug[int2(DispatchRaysIndex().xy)] = float4(1, 0, 1, 1);
    //         }
    //         value += vpl.getColor() * k_pi * w1;
    //     }
    // }
    // else {
    //     u_Debug[int2(DispatchRaysIndex().xy)] = float4(0, 0, 0, 1);
    // }

    // else {
    //     u_Debug[int2(DispatchRaysIndex().xy)] = float4(0, 1, 1, 1);
    // }

    // if (all(gi_luminance == float3(0))) {
    //     u_Debug[int2(DispatchRaysIndex().xy)] = float4(1, 0, 0, 1);
    // }
    // else if (dot(payload.hit.geometryNormal, dir) <= 0) {
    //     u_Debug[int2(DispatchRaysIndex().xy)] = float4(0, 1, 0, 1);
    // }

    // float3 value = gi_luminance * dot(payload.hit.geometryNormal, dir) / pdf;
    // if(selected.y == -1) {
    //     value = float3(1,0,1);
    // }
    // if (all(value == float3(0))) {
    //     u_Debug[int2(DispatchRaysIndex().xy)] = float4(dot(payload.hit.geometryNormal, dir), pdf, 0, 1);
    // }
    return value;
}

[shader("raygeneration")]
void RaygenMain() {
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