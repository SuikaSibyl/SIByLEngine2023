#include "common/random.hlsli"
#include "common/sampling.hlsli"
#include "common/camera.hlsli"
#include "common/rotation.hlsli"
#include "srenderer/rayquery.hlsli"
#include "srenderer/materials/lambertian.hlsli"
#include "srenderer/shapes/rectangle.hlsli"

[[vk::push_constant]] cbuffer PushConstants { 
    float3 position;
    int random_seed;
    float3 euler;
    int camera_index;
    float width;
    float height;
    int render_mode;
};

RWTexture2D<float4> u_image;
RWTexture2D<float> u_pmf_ideal;
RWTexture2D<float> u_pmf_actual;

[shader("compute")]
[numthreads(32, 4, 1)]
void ComputeMain(
    int3 dtid: SV_DispatchThreadID,
    int3 gtid: SV_GroupThreadID,
    int3 gid: SV_GroupID,
    int gi: SV_GroupIndex
) {
    // The resolution of the image, which is the same as the launch size:
    const int2 resolution = int2(1024, 1024);
    const int2 pixel = int2(dtid.xy);
    // If the pixel is outside of the image, don't do anything:
    if (any(pixel >= resolution)) return;
    RandomSamplerState RNG = InitRandomSampler(dtid.xy, random_seed);

    float4x4 rot = rotate_euler(euler);
    float3 u = mul(transpose(rot), float4(1, 0, 0, 0)).xyz * width;
    float3 v = mul(transpose(rot), float4(0, 1, 0, 0)).xyz * height;
    float3 n = mul(transpose(rot), float4(0, 0, 1, 0)).xyz;
    Rectangle shape = Rectangle(
        position - 0.5 * u - 0.5 * v,
        n, u, v
    );
    
    float3 radiance = float3(0);

    const float2 jitter = GetNextRandomFloat2(RNG) - 0.5f;
    Ray ray = generateRay(pixel + jitter, resolution, false, GPUScene_camera[camera_index]);

    PrimaryPayload payload = PrimaryRayQuery(ray);
    bool hit_scene = HasHit(payload.hit);
    PrimaryPayload shape_payload;
    if (shape.intersect(ray, shape_payload)) {
        if ((!hit_scene) || distance(payload.hit.position, ray.origin) > 
            distance(shape_payload.hit.position, ray.origin)) {
            radiance = float3(1.0f);
            hit_scene = false;
        }
    }

    LambertianBRDF brdf = LambertianBRDF(float3(1.0f));
    float3 throughput = float3(1.0f);

    if (hit_scene) {
        if (render_mode == 0) {
            ibsdf::sample_in i;
            i.u = GetNextRandomFloat2(RNG);
            i.wi = -ray.direction;
            i.geometric_normal = payload.hit.geometryNormal;
            i.shading_frame = Frame(createFrame(payload.hit.shadingNormal));
            ibsdf::sample_out sample_o = brdf.sample(i);
            ray = SpawnRay(payload.hit, sample_o.wo);
            throughput *= sample_o.bsdf;
        } else {
            const float3 ray_in = -ray.direction;
            // sample the light shape
            ishape::sample_in i;
            i.uv = GetNextRandomFloat2(RNG);
            i.position = payload.hit.position;
            i.normal = payload.hit.geometryNormal;
            i.flag = render_mode - 1;
            ishape::sample sample_o = shape.sample(i);
            float3 direction = normalize(sample_o.position - payload.hit.position);
            ray = SpawnRay(payload.hit, direction);
            // evaluate the brdf
            ibsdf::eval_in eval_i;
            eval_i.wi = ray_in;
            eval_i.wo = direction;
            eval_i.geometric_normal = payload.hit.geometryNormal;
            eval_i.shading_frame = Frame(createFrame(payload.hit.shadingNormal));

            float3 bsdf = brdf.eval(eval_i);
            throughput *= bsdf / sample_o.pdf;
        }
        
        // Add direct lighting
        if (shape.intersect(ray, shape_payload)) {
            throughput *= float3(1);
            radiance += throughput;
        }

        if (dot(payload.hit.position.xz, payload.hit.position.xz) < 0.02) {
            radiance = float3(1.0f, 0, 1.0f);
        }
    }

    if (all(pixel < int2(512))) {
        const float2 uv = (pixel.xy + float2(0.5f)) / 512.0f;
        ishape::sample_in i;
        i.uv = uv;
        i.position = float3(0.);
        i.normal = float3(0, 1, 0);
        
        let sr = SphericalRectangle(shape.p, shape.u, shape.v, i.position);
        ishape::sample o = sr.sample(i);
        float3 direction = normalize(o.position - i.position);
        
        u_pmf_ideal[pixel] = max(0., dot(direction, i.normal));

        float3 wp[4];
        wp[0] = shape.p;
        wp[1] = shape.p + shape.u;
        wp[2] = shape.p + shape.v;
        wp[3] = shape.p + shape.u + shape.v;
        float4 prob;
        for (int j = 0; j < 4; j++) {
            float3 wpd = wp[j] - i.position;
            float3 wpn = normalize(wpd);
            // cosine term of ray with illuminated surface
            prob[j] = max(0., dot(wpn, i.normal));
        }
        let bw = BilinearWarp(prob[0], prob[2], prob[1], prob[3]);
        u_pmf_actual[pixel] = bw.pdf(uv);
    }

    u_image[pixel] = float4(radiance, 1.0f);
}