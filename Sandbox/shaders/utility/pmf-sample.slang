#include "../common/random.hlsli"

[[vk::push_constant]]
cbuffer PushConstants { int rand_seed; };

const StructuredBuffer<float> sampler_buffer;
RWStructuredBuffer<int> sample_buffer;
RWStructuredBuffer<float> pdf_buffer;

groupshared float shared_buffer[2048];

[shader("compute")]
[numthreads(16, 16, 1)]
void ComputeMain(
    int3 dtid: SV_DispatchThreadID,
    int3 gtid: SV_GroupThreadID,
    int3 gctid: SV_GroupID,
    int gidx: SV_GroupIndex
) {
    const int superpixel_index = gctid.x + gctid.y * 33;
    const int superpixel_offset = 2048 * superpixel_index;

    RandomSamplerState RNG = InitRandomSampler(uint2(dtid.xy), rand_seed);
    float rnd = GetNextRandom(RNG);

    // preload the sampler buffer
    for (int i = 0; i < 2048; i += 256) {
        shared_buffer[i + gidx] = sampler_buffer[superpixel_offset + i + gidx];
    }

    GroupMemoryBarrierWithGroupSync();
    
    double nprob = 1.0;

    int nid = 1;
    while (nid < 1024) {
        int c0_id = nid << 1;  // left child
        int c1_id = c0_id + 1; // right child

        float prob0 = shared_buffer[c0_id] / shared_buffer[nid];
        if (rnd < prob0) {
            nid = c0_id;
            rnd /= prob0;
            nprob *= double(prob0);
        }
        else {
            nid = c1_id;
            rnd = (rnd - prob0) / (1 - prob0);
            nprob *= double(1 - prob0);
        }
    }
    
    nid -= 1024;
    int sample_x = nid % 32;
    int sample_y = nid / 32;
    int sample = (sample_x << 16) + sample_y;

    const int sample_pixel = dtid.x + dtid.y * (33 * 16);
    sample_buffer[sample_pixel] = sample;
    float weight = 0.f;
    if (nprob > 0.0) {
        weight = float((double(32 * 32 * 1.) / nprob));
    }
    pdf_buffer[sample_pixel] = weight;
}