#include "../../../include/common/octahedral.hlsli"
#include "../../../include/common/random.hlsli"
#include "../../../include/raytracer_descriptor_set.hlsli"
#include "../../../include/scene_descriptor_set.hlsli"
#include "../../../raytracer/spt_interface.hlsli"
#include "../../vbuffer/vbuffer.hlsli"
#include "../../vxgi/include/conetrace_utils.hlsli"
#include "../../vxgi/include/vxgi_interface.hlsli"
#include "../include/vxguiding_interface.hlsli"

void EvalLambertian(inout_ref(BSDFEvalQuery) cBSDFEvalQuery) {
    if (dot(cBSDFEvalQuery.geometric_normal, cBSDFEvalQuery.dir_in) < 0 ||
        dot(cBSDFEvalQuery.geometric_normal, cBSDFEvalQuery.dir_out) < 0) {
        // No light below the surface
        cBSDFEvalQuery.bsdf = float3(0);
        cBSDFEvalQuery.dir_out = float3(0); // override specular
        return;
    }
    // Making sure the shading frame is consistent with the view direction.
    float3x3 frame = cBSDFEvalQuery.frame;
    const QueryBitfield bitfield = UnpackQueryBitfield(cBSDFEvalQuery.misc_flag);

    // Evaluate bsdf
    float3 albedo;
    if (cBSDFEvalQuery.mat_id == 0xFFFFFFFF) {
        albedo = UnpackRGBE(asuint(cBSDFEvalQuery.bsdf.x));
    }
    else {
        const MaterialInfo material = materials[cBSDFEvalQuery.mat_id];
        const float3 texAlbedo = textures[material.baseOrDiffuseTextureIndex]
                                     .Sample(cBSDFEvalQuery.uv, 0) .xyz;
        albedo = material.baseOrDiffuseColor * texAlbedo;
    }
    const float3 demodulate = saturate(dot(frame[2], cBSDFEvalQuery.dir_out)) / k_pi;
    cBSDFEvalQuery.bsdf = bitfield.split_query ? demodulate : albedo * demodulate;
    cBSDFEvalQuery.dir_out = float3(0); // override specular
}

void EvalRoughPlastic(inout_ref(BSDFEvalQuery) cBSDFEvalQuery) {
    // First load the material info
    // -------------------------------------------------------------
    float3 Kd;
    float3 Ks;
    float eta;
    float roughness;
    if (cBSDFEvalQuery.mat_id == 0xFFFFFFFF) {
        // info is already packed in the query
        Kd = UnpackRGBE(asuint(cBSDFEvalQuery.bsdf.x));
        Ks = UnpackRGBE(asuint(cBSDFEvalQuery.uv.y));
        eta = cBSDFEvalQuery.bsdf.y;
        roughness = cBSDFEvalQuery.uv.x;
    } else { // load info from the material buffer
        const MaterialInfo material = materials[cBSDFEvalQuery.mat_id];
        const float3 texAlbedo = textures[material.baseOrDiffuseTextureIndex]
                                     .Sample(cBSDFEvalQuery.uv, 0) .xyz;
        Kd = material.baseOrDiffuseColor * texAlbedo;
        Ks = material.specularColor;
        eta = material.transmissionFactor;
        roughness = material.roughness;
    }
    // Clamp roughness to avoid numerical issues.
    roughness = clamp(roughness, 0.01f, 1.f);
    const QueryBitfield bitfield = UnpackQueryBitfield(cBSDFEvalQuery.misc_flag);

    // Then evaluate the BSDF
    // -------------------------------------------------------------
    cBSDFEvalQuery.bsdf = float3(0);
    if (dot(cBSDFEvalQuery.geometric_normal, cBSDFEvalQuery.dir_in) < 0 ||
        dot(cBSDFEvalQuery.geometric_normal, cBSDFEvalQuery.dir_out) < 0) {
        // No light below the surface
        cBSDFEvalQuery.dir_out = float3(0);
        return;
    }
    // Making sure the shading frame is consistent with the view direction.
    float3x3 frame = cBSDFEvalQuery.frame;
    if (dot(frame[2], cBSDFEvalQuery.dir_in) < 0) {
        frame = -frame;
    }

    const float3 half_vector = normalize(cBSDFEvalQuery.dir_in + cBSDFEvalQuery.dir_out);
    const float n_dot_h = dot(frame[2], half_vector);
    const float n_dot_in = dot(frame[2], cBSDFEvalQuery.dir_in);
    const float n_dot_out = dot(frame[2], cBSDFEvalQuery.dir_out);
    if (n_dot_out <= 0 || n_dot_h <= 0) {
        cBSDFEvalQuery.dir_out = float3(0);
        return;
    }

    // dielectric layer:
    // F_o is the reflection percentage.
    const float F_o = FresnelDielectric(dot(half_vector, cBSDFEvalQuery.dir_out), eta);
    const float D = GTR2_NDF(n_dot_h, roughness);
    const float G = IsotropicGGX_Masking(to_local(frame, cBSDFEvalQuery.dir_in), roughness) *
                    IsotropicGGX_Masking(to_local(frame, cBSDFEvalQuery.dir_out), roughness);
    const float3 spec_contrib = Ks * (G * F_o * D) / (4 * n_dot_in * n_dot_out);
    // diffuse layer:
    // In order to reflect from the diffuse layer,
    // the photon needs to bounce through the dielectric layers twice.
    // The transmittance is computed by 1 - fresnel.
    const float F_i = FresnelDielectric(dot(half_vector, cBSDFEvalQuery.dir_in), eta);
    // Multiplying with Fresnels leads to an overly dark appearance at the
    // object boundaries. Disney BRDF proposes a fix to this -- we will implement this in problem set 1.
    const float3 diffuse_contrib = (1.f - F_o) * (1.f - F_i) / k_pi;
    if (bitfield.split_query) {
        cBSDFEvalQuery.bsdf = diffuse_contrib * n_dot_out;
        cBSDFEvalQuery.dir_out = spec_contrib * n_dot_out;
    } else {
        cBSDFEvalQuery.bsdf = (spec_contrib + Kd * diffuse_contrib) * n_dot_out;
    }
    return;
}

Texture2D<int2> u_spixel_gathered;
Texture2D<uint> u_spixel_counter;
StructuredBuffer<uint> u_cluster_counter;
StructuredBuffer<float4> u_cluster_gathered;

Texture2D<uint4> u_vBuffer;
RWTexture2D<uint> u_spixel_visibility;
RWTexture2D<float4> u_Debug;

RWStructuredBuffer<float> u_spixel_avg_visibility;

[[vk::push_constant]]
cbuffer PushConstants {
    int2 resolution;
    int2 map_size;
    int seed;
    bool use_bsdf;
    bool use_distance;
};

groupshared float sharedVisibility[32];
groupshared uint groupVisibility;

bool HandleVisibilityCheck(
    int2 spixelID,
    int2 dispatchID,
    int2 taskOffset,
    int clusterToCheck,
    out float bsdf
) {
    bsdf = 0.f;
    RandomSamplerState RNG = InitRandomSampler(dispatchID, seed);

    // Grab from vsibility buffer
    const uint spixel_count = clamp(u_spixel_counter[spixelID], 0, 1024);
    if (spixel_count == 0) {
        return false;
    }
    const int subtaskFlatten = clamp(int(spixel_count * GetNextRandom(RNG)), 0, spixel_count - 1);
    const int2 subtaskID = int2(subtaskFlatten % 32, subtaskFlatten / 32);
    const int2 pixelID = u_spixel_gathered[taskOffset + subtaskID];

    const Ray primaryRay = generateRay(
        pixelID,
        resolution,
        false,
        globalUniform.cameraData
    );
    
    const VBufferData vhit = UnpackVBufferData(u_vBuffer[pixelID]);
    if (IsInvalid(vhit)) {
        return false;
    }
    const GeometryHit hit = GetGeometryHit(vhit, primaryRay);

    // Grab a valid vpl from the voxel buffer
    const int clusterVPLCount = clamp(u_cluster_counter[clusterToCheck], 0, 1024);
    if (clusterVPLCount == 0) return false;
    // if ((u_spixel_visibility[spixelID] & uint(1 << clusterToCheck)) != 0) return false;
    const int subtaskFlatten_vpl = clamp(int(clusterVPLCount * GetNextRandom(RNG)), 0, clusterVPLCount - 1);
    const float4 vpl_posnorm = u_cluster_gathered[clusterToCheck * 1024 + subtaskFlatten_vpl];
    const float3 vpl_normal = Unorm32OctahedronToUnitVector(asuint(vpl_posnorm.w));
    float3 dir = vpl_posnorm.xyz - hit.position;
    const float distannce = length(dir);
    dir /= distannce;
    if (dot(-dir, vpl_normal) <= 0.0001 || dot(dir, hit.geometryNormal) <= 0.0001) return false;
    
    // Visibility test
    bool visible = true;
    Ray visiblityRay = SpawnRay(hit, dir);
    visiblityRay.tMax = distannce - 0.01;

    // Ray query to check visibility
    { RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> q;
        q.TraceRayInline(
            SceneBVH,
            0, 0xff,
            ToRayDesc(visiblityRay));
        if (q.Proceed()) {
            visible = false;
        }
    }

    if (visible) {
        if (use_bsdf) {
            uint materialID = geometries[hit.geometryID].materialID;
            uint bsdf_type = materials[materialID].bsdfID;
            if (bsdf_type == 0) {
                bsdf = 1.f;
            } else if (bsdf_type == 1) {
                BSDFEvalQuery cBSDFEvalQuery;
                cBSDFEvalQuery.dir_in = -primaryRay.direction;
                cBSDFEvalQuery.dir_out = visiblityRay.direction;
                cBSDFEvalQuery.mat_id = materialID;
                cBSDFEvalQuery.geometric_normal = hit.geometryNormal;
                cBSDFEvalQuery.uv = hit.texcoord;
                cBSDFEvalQuery.frame = createONB(hit.shadingNormal);
                QueryBitfield flag;
                flag.transport_mode = 0;
                flag.face_forward = IsFaceForward(hit);
                flag.split_query = false;
                cBSDFEvalQuery.misc_flag = PackQueryBitfield(flag);
                EvalRoughPlastic(cBSDFEvalQuery);
                bsdf = maxComponent(cBSDFEvalQuery.bsdf);
            }
        } 
        else {
            bsdf = 1.f;
        }
        if (use_distance) {
            bsdf /= (distannce * distannce);
        }
    } 

    return visible;
}

[shader("compute")]
[numthreads(32, 8, 1)]
void ComputeMain(
    int3 DTid: SV_DispatchThreadID,
    int gidx: SV_GroupIndex,
) {
    // each block is handling the sample superpixel
    const int2 dispatchID = int2(DTid.xy);
    const int2 spixelID = int2(dispatchID / 32);
    const int2 task = int2(dispatchID % 32);
    const int2 taskOffset = spixelID * 32;
    const int clusterToCheck = task.y;

    if (all(dispatchID < int2(1280, 720)))
        u_Debug[dispatchID] = float4(WaveGetLaneIndex(), 0, 0, 1);

    const int clusterMod = clusterToCheck % 8;
    if (gidx < 32) {
        groupVisibility = 0;
        sharedVisibility[gidx] = 0;
    }
    GroupMemoryBarrierWithGroupSync();

    float bsdf;
    bool visible = HandleVisibilityCheck(spixelID, dispatchID, taskOffset, clusterToCheck, bsdf);
    sharedVisibility[clusterToCheck] = WaveActiveSum(visible ? bsdf : 0);
    
    GroupMemoryBarrierWithGroupSync();
    if (gidx < 32) {
        float visibility = sharedVisibility[gidx];
        const uint spixelIDFlatten = spixelID.y * map_size.x + spixelID.x;
        const int cluster_offset = (clusterToCheck / 8) * 8;
        if (gidx >= cluster_offset && gidx < (cluster_offset + 8)) {
            u_spixel_avg_visibility[spixelIDFlatten * 32 + gidx] = visibility;
        }
        if (visibility != 0) {
            InterlockedOr(groupVisibility, uint(1 << gidx));
        } 
    }
    if (gidx == 0) {
        InterlockedOr(u_spixel_visibility[spixelID], groupVisibility);
    }
}