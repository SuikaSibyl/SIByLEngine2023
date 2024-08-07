#include "common/random.hlsli"
#include "common/sampling.hlsli"
#include "common/camera.hlsli"
#include "common/rotation.hlsli"
#include "srenderer/rayquery.hlsli"
#include "srenderer/materials/lambertian.hlsli"
#include "srenderer/shapes/rectangle.hlsli"

[[vk::push_constant]] cbuffer PushConstants { 
    int random_seed;
    int camera_index;
    int render_mode;
};

RWTexture2D<float4> u_image;

Rectangle create_transformed_rectangle(float3 position, float3 euler, float width, float height) {
    float4x4 rot = rotate_euler(euler);
    float3 u = mul(transpose(rot), float4(1, 0, 0, 0)).xyz * width;
    float3 v = mul(transpose(rot), float4(0, 1, 0, 0)).xyz * height;
    float3 n = mul(transpose(rot), float4(0, 0, 1, 0)).xyz;
    return Rectangle(
        position - 0.5 * u - 0.5 * v,
        n, u, v
    );}

struct RectangleLight {
    Rectangle shape;
    float3 emission;
};

void SceneHit(RectangleLight lights[3], Ray ray, out float nearest_distance, out int nearest_primitve) {
    nearest_distance = 1e20;
    nearest_primitve = -1;
    PrimaryPayload shape_payload;
    for (int i = 0; i < 3; ++i) {
        if (lights[i].shape.intersect(ray, shape_payload)) {
            const float dist = distance(shape_payload.hit.position, ray.origin);
            if (dist < nearest_distance) {
                nearest_primitve = i;
                nearest_distance = dist;
            }
        }
    }
}

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

    
    float3 radiance = float3(0);

    const float2 jitter = GetNextRandomFloat2(RNG) - 0.5f;
    Ray ray = generateRay(pixel + jitter, resolution, false, GPUScene_camera[camera_index]);

    RectangleLight lights[3] = {
        { create_transformed_rectangle(float3(4, 8, -6), float3(0, 2, 0), 5, 5), float3(1.0f, 0.f, 0.f) },
        { create_transformed_rectangle(float3(4, 10, 0), float3(0, 1.57, 0), 5, 5), float3(0.0f, 1.f, 0.f) },
        { create_transformed_rectangle(float3(4, 8, 6), float3(0, -2, 0), 5, 5), float3(0.0f, 0.f, 1.f) },
    };
    
    PrimaryPayload payload = PrimaryRayQuery(ray);
    bool hit_scene = HasHit(payload.hit);
    float prim_distance; int primitive;
    SceneHit(lights, ray, prim_distance, primitive);

    // If the ray directly hits a light, return the light's emission
    if (primitive != -1) {
        if ((!hit_scene) || prim_distance < distance(payload.hit.position, ray.origin)) {
            radiance = lights[primitive].emission;
            hit_scene = false;
        }
    }

    LambertianBRDF brdf = LambertianBRDF(float3(1.0f));
    Ray primary_ray = ray;
    if (hit_scene) {
        for (int spp = 0; spp < 3; ++spp) {
            // reset
            float3 throughput = float3(1.0f);
            ray = primary_ray;

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
                int random_index = GetNextRandomUint(RNG) % 3;
                ishape::sample sample_o = lights[random_index].shape.sample(i);
                float3 direction = normalize(sample_o.position - payload.hit.position);
                ray = SpawnRay(payload.hit, direction);
                // evaluate the brdf
                ibsdf::eval_in eval_i;
                eval_i.wi = ray_in;
                eval_i.wo = direction;
                eval_i.geometric_normal = payload.hit.geometryNormal;
                eval_i.shading_frame = Frame(createFrame(payload.hit.shadingNormal));

                float3 bsdf = brdf.eval(eval_i);
                throughput *= bsdf / sample_o.pdf * 3;
            }
            
            // Add direct lighting
            float distance; int primitive;
            SceneHit(lights, ray, distance, primitive);
            if (primitive != -1) {
                throughput *= lights[primitive].emission;
                radiance += throughput;
            }
        }
        radiance /= 3;
    }

    u_image[pixel] = float4(radiance, 1.0f);
}