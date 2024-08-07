#include "srenderer/materials/rglbrdf.hlsli"

RWTexture2D<float3> output;

[shader("compute")]
[numthreads(16, 16, 1)]
void ComputeMain(
    int3 dtid: SV_DispatchThreadID,
    int3 gtid: SV_GroupThreadID,
    int3 gid: SV_GroupID,
    int gi: SV_GroupIndex
) {
    RGLBRDF bsdf = RGLBRDF(0);
    ibsdf::sample_in sample_in;
    sample_in.wi = normalize(float3(1, 1, 1));
    sample_in.u = float2(0.99334f, 0.231f);
    ibsdf::sample_out o = bsdf.sample(sample_in);

    ibsdf::pdf_in pdf_in;
    pdf_in.wi = sample_in.wi;
    pdf_in.wo = o.wo;
    float pdf = bsdf.pdf(pdf_in);
    
    output[dtid.xy] = float3(pdf);
}