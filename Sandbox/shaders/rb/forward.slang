#include "../srenderer/rayquery.hlsli"
#include "../srenderer/materials/lambertian.hlsli"
#include "../srenderer/materials/roughplastic.hlsli"

[[vk::push_constant]]
cbuffer PushConstants {
    int camera_index;
    int random_seed;
};

RWStructuredBuffer<float> u_image;
RWStructuredBuffer<float> u_basecolor;
RWStructuredBuffer<float> u_roughness;

float3 LoadAlbedoTexture(no_diff float2 uv) {
    int2 texel = int2(uv * 512);
    float3 color = float3(0);
    color.r = u_basecolor[(texel.x + texel.y * 512) * 3 + 0];
    color.g = u_basecolor[(texel.x + texel.y * 512) * 3 + 1];
    color.b = u_basecolor[(texel.x + texel.y * 512) * 3 + 2];
    return color;
}

float LoadRoughnessTexture(no_diff float2 uv) {
    int2 texel = int2(uv * 512);
    return u_roughness[texel.x + texel.y * 512];
}

void set_image(RWStructuredBuffer<float> image, int2 pixel, int2 resolution, float3 color) {
    image[(pixel.x + pixel.y * resolution.x) * 3 + 0] = color.x;
    image[(pixel.x + pixel.y * resolution.x) * 3 + 1] = color.y;
    image[(pixel.x + pixel.y * resolution.x) * 3 + 2] = color.z;
}

float3 EvalBSDFValue(
    no_diff float2 uv,
    no_diff BSDFEvalGeometry geom,
    no_diff float weight
) {
    RoughPlasticMaterial material;
    material.Kd = LoadAlbedoTexture(uv);
    material.Ks = float3(1.f);
    material.roughness = LoadRoughnessTexture(uv);
    material.eta = 2.0f;
    return EvalRoughPlastic(material, geom) * weight;
    // LambertMaterial material;
    // material.albedo = LoadAlbedoTexture(uv);
    // return EvalLambertian(material, geom) * weight;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void ComputeMain(int3 dtid: SV_DispatchThreadID) {
    // The resolution of the image, which is the same as the launch size:
    const int2 resolution = int2(512, 512);
    const int2 pixel = int2(dtid.xy);
    RandomSamplerState RNG = InitRandomSampler(pixel, random_seed);

    // If the pixel is outside of the image, don't do anything:
    if (any(pixel >= resolution)) return;

    float2 jitter = GetNextRandomFloat2(RNG) - 0.5f;
    const Ray primaryRay = generateRay(
        pixel + jitter,
        resolution,
        false,
        GPUScene_camera[camera_index]
    );
    float2 p = (float2(pixel) + float2(0.5f, 0.5f)) / float2(resolution);
    
    // Ray query to check visibility
    float3 output = float3(0.0f, 0.0f, 0.0f);
    
    PrimaryPayload payload = PrimaryRayQuery(primaryRay);
    if (HasHit(payload.hit)) {
        // output = payload.hit.position * 0.5f + 0.5f;
        // output = float3(payload.hit.texcoord, 0);
        float3 point_light = float3(0, 15, 10);
        float3 light_dir = normalize(point_light - payload.hit.position);
        BSDFEvalGeometry geom = createBSDFEvalGeometry(payload.hit, -primaryRay.direction, light_dir);
        float visible = VisibilityRayQuery(SpawnVisibilityRay(payload.hit, point_light)) ? 1 : 0;
        // visible = 1;
        float light_intensity = 3;
        output = EvalBSDFValue(payload.hit.texcoord, geom, visible * light_intensity);
        // output = float3(payload.hit.texcoord, 1);
        if(any(isnan(output))) {
            output = float3(0, 0, 0);
        }
    }

    set_image(u_image, pixel, resolution, output);
}