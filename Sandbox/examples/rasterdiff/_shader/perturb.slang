#include "common/random.hlsli"
#include "common.hlsli"

// Albedo
RWTexture2D<float4> Albedo;
RWTexture2D<float4> AlbedoPlus;
RWTexture2D<float4> AlbedoMinus;
RWByteAddressBuffer dAlbedo;
// Position
RWByteAddressBuffer Position;
RWByteAddressBuffer PositionPlus;
RWByteAddressBuffer PositionMinus;
RWByteAddressBuffer dPosition;
// Position Average
RWByteAddressBuffer PositionAvg;
RWByteAddressBuffer PositionCount;

[[vk::push_constant]]
cbuffer PushConstants {
    int vertex_num;
    int random_seed;
    float epsilon_albedo;
    float epsilon_pos;
    int initialize;
    float learning_rate;
    float learning_rate2;
    float smooth_rate;
};

[shader("compute")]
[numthreads(512, 1, 1)]
void ComputeMain(
    int3 dtid: SV_DispatchThreadID,
    int3 gtid: SV_GroupThreadID,
    int3 gid: SV_GroupID,
    int gi: SV_GroupIndex
) {
    const int globalID = dtid.x;
    const int texel_num = 1024 * 1024;
    RandomSamplerState RNG = InitRandomSampler(globalID, random_seed);
    const float3 pos_perturb = epsilon_pos * float2sign(GetNextRandomFloat3(RNG));
    const float3 albedo_perturb = epsilon_albedo * float2sign(GetNextRandomFloat3(RNG));
    
    if (globalID < vertex_num) {
        // fetch the position
        const int pos_offset = 4 * 3 * globalID;
        float3 pos = Position.Load<float3>(pos_offset);
        if (initialize != 1) {
            int count = PositionCount.Load<uint>(globalID * 4);
            const float3 smooth_pos = PositionAvg.Load<float3>(pos_offset) / count;
            pos = lerp(pos, smooth_pos, smooth_rate);
            const float3 d_posistion = dPosition.Load<float3>(pos_offset);
            pos = pos - learning_rate * d_posistion;
            Position.Store<float3>(pos_offset, pos);
        }
        // store the perturbed position
        PositionPlus.Store<float3>(pos_offset, float3(pos + pos_perturb));
        PositionMinus.Store<float3>(pos_offset, float3(pos - pos_perturb));
        dPosition.Store<float3>(pos_offset, float3(0.0f, 0.0f, 0.0f));
        PositionAvg.Store<float3>(pos_offset, float3(0.0f, 0.0f, 0.0f));
        PositionCount.Store<uint>(globalID * 4, 0);
    }

    if (globalID < texel_num) {
        // fetch the albedo
        const int2 pixel = int2(globalID % 1024, globalID / 1024);
        float3 albedo = (initialize == 1) ? float3(0.5) : Albedo[pixel].rgb;
        if (initialize == 1) Albedo[pixel] = float4(albedo, 1.f);
        else {
            const int albedo_offset = 4 * 3 * globalID;
            float3 d_albedo = dAlbedo.Load<float3>(albedo_offset);
            if (any(isnan(d_albedo))) d_albedo = float3(0.0f, 0.0f, 0.0f);
            albedo = albedo - learning_rate2 * d_albedo;
            albedo = clamp(albedo, 0.0f, 1.0f);
            Albedo[pixel] = float4(albedo, 1.f);
        }
        // store the perturbed albedo
        AlbedoPlus[pixel] = float4(albedo + albedo_perturb, 1.f);
        AlbedoMinus[pixel] = float4(albedo - albedo_perturb, 1.f);
        dAlbedo.Store<float3>(globalID * 4 * 3, float3(0.0f, 0.0f, 0.0f));
    }
}