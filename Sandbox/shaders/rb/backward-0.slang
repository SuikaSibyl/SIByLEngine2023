#include "../srenderer/materials/lambertian.hlsli"
#include "../srenderer/materials/roughplastic.hlsli"
#include "../srenderer/rayquery.hlsli"

[[vk::push_constant]]
cbuffer PushConstants {
    int camera_index;
    int random_seed;
};

RWStructuredBuffer<float> u_image_grad;
RWStructuredBuffer<float> u_basecolor;
RWStructuredBuffer<float> u_roughness;
RWByteAddressBuffer u_basecolor_grad;
RWByteAddressBuffer u_roughness_grad;
RWByteAddressBuffer u_spp;

[BackwardDerivative(LoadAlbedoTextureBwd)]
float3 LoadAlbedoTexture(no_diff float2 uv) {
    int2 texel = int2(uv * 512);
    float3 color = float3(0);
    color.r = u_basecolor[(texel.x + texel.y * 512) * 3 + 0];
    color.g = u_basecolor[(texel.x + texel.y * 512) * 3 + 1];
    color.b = u_basecolor[(texel.x + texel.y * 512) * 3 + 2];
    AddSpp(texel);
    return color;
}

void AddSpp(int2 pixel) {
    u_spp.InterlockedAddF32((pixel.x + pixel.y * 512) * 4, 1);
}

void LoadAlbedoTextureBwd(no_diff float2 uv, in float3.Differential dcolor) {
    const int2 texel = int2(uv * 512);
    u_basecolor_grad.InterlockedAddF32(((texel.x + texel.y * 512) * 3 + 0) * 4, dcolor.r);
    u_basecolor_grad.InterlockedAddF32(((texel.x + texel.y * 512) * 3 + 1) * 4, dcolor.g);
    u_basecolor_grad.InterlockedAddF32(((texel.x + texel.y * 512) * 3 + 2) * 4, dcolor.b);
}

[BackwardDerivative(LoadRoughnessTextureBwd)]
float LoadRoughnessTexture(no_diff float2 uv) {
    int2 texel = int2(uv * 512);
    return u_roughness[texel.x + texel.y * 512];
}

void LoadRoughnessTextureBwd(no_diff float2 uv, in float.Differential dRoughness) {
    const int2 texel = int2(uv * 512);
    u_roughness_grad.InterlockedAddF32((texel.x + texel.y * 512) * 4, dRoughness);
}

float3 load_image_grad(RWStructuredBuffer<float> image_grad, int2 pixel, int2 resolution) {
    float3 color_grad;
    color_grad.r = image_grad[(pixel.x + pixel.y * resolution.x) * 3 + 0];
    color_grad.g = image_grad[(pixel.x + pixel.y * resolution.x) * 3 + 1];
    color_grad.b = image_grad[(pixel.x + pixel.y * resolution.x) * 3 + 2];
    return color_grad;
}

[Differentiable]
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
    float3 pixel_grad = load_image_grad(u_image_grad, pixel, resolution);
    
    PrimaryPayload payload = PrimaryRayQuery(primaryRay);
    if (HasHit(payload.hit)) {
        // output = payload.hit.position * 0.5f + 0.5f;
        // output = float3(payload.hit.texcoord, 0);
        float3 point_light = float3(0, 15, 10);
        float3 light_dir = normalize(point_light - payload.hit.position);
        BSDFEvalGeometry geom = createBSDFEvalGeometry(payload.hit, -primaryRay.direction, light_dir);
        float visible = VisibilityRayQuery(SpawnVisibilityRay(payload.hit, point_light)) ? 1 : 0;
        float light_intensity = 3;
        bwd_diff(EvalBSDFValue)(payload.hit.texcoord, geom, visible * light_intensity, pixel_grad);
    }
}