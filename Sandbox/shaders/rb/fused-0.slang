#include "../srenderer/materials/lambertian.hlsli"
#include "../srenderer/materials/roughplastic.hlsli"
#include "../srenderer/rayquery.hlsli"

RWStructuredBuffer<float4> u_inverse;
RWStructuredBuffer<float> u_image_grad;
RWStructuredBuffer<float> u_groundtruth;
RWStructuredBuffer<float> u_image;
RWStructuredBuffer<float> u_basecolor;
RWStructuredBuffer<float> u_roughness;
RWByteAddressBuffer u_basecolor_grad;
RWByteAddressBuffer u_roughness_grad;

float Jacobian(CameraData cameraData, float2 sample_uv, float3 pos, float3 normal) {
    float3 pix_position = cameraData.cameraW
        + (-0.5 + sample_uv.x) * 2 * cameraData.cameraU
        + (-0.5 + sample_uv.y) * 2 * cameraData.cameraV;
    float3 pix_dir = normalize(pix_position);
    float cos_0 = max(dot(-pix_dir, -normalize(cameraData.cameraW)), 0);
    float r_0 = length(pix_position);
    float area_0 = 4 * length(cameraData.cameraU) * length(cameraData.cameraV);
    float jacobian_0 = cos_0 * area_0 / (r_0 * r_0);

    float dist_1 = length(pos - cameraData.posW);
    float3 dir_1 = normalize(pos - cameraData.posW);
    float cos_1 = max(dot(normal, -dir_1), 0);
    return jacobian_0 * (dist_1 * dist_1) / cos_1;
}

float3 load_ground_truth(int2 pixel, int view) {
    int index[4] = { 3, 0, 1, 2 };
    float3 color;
    color.x = u_groundtruth[(index[view]*512*512*3) + (pixel.x + pixel.y * 512) * 3 + 0];
    color.y = u_groundtruth[(index[view]*512*512*3) + (pixel.x + pixel.y * 512) * 3 + 1];
    color.z = u_groundtruth[(index[view]*512*512*3) + (pixel.x + pixel.y * 512) * 3 + 2];
    return color;
}

GeometryHit LoadInverseBuffer(int2 pixel) {
    GeometryHit hit;
    const float4 val = u_inverse[(pixel.x + pixel.y * 512)];
    if (all(val == float4(0))) {
        SetHit(hit, false);
    } else {
        RawPayload payload;
        payload.primitiveID = asuint(val.x);
        payload.geometryID = asuint(val.y);
        payload.barycentric = val.zw;
        const float2 bary = payload.barycentric;
        const float3 barycentrics = float3(1 - bary.x - bary.y, bary.x, bary.y);
        hit = fetchTrimeshGeometryHit(payload.geometryID, barycentrics, payload.primitiveID);
        SetHit(hit, true);
    }
    return hit;
}

[BackwardDerivative(LoadAlbedoTextureBwd)]
float3 LoadAlbedoTexture(no_diff float2 uv) {
    int2 texel = int2(uv * 512);
    float3 color = float3(0);
    color.r = u_basecolor[(texel.x + texel.y * 512) * 3 + 0];
    color.g = u_basecolor[(texel.x + texel.y * 512) * 3 + 1];
    color.b = u_basecolor[(texel.x + texel.y * 512) * 3 + 2];
    return color;
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

void set_image(RWStructuredBuffer<float> image, int2 pixel, int2 resolution, float3 color) {
    image[(pixel.x + pixel.y * resolution.x) * 3 + 0] = color.x;
    image[(pixel.x + pixel.y * resolution.x) * 3 + 1] = color.y;
    image[(pixel.x + pixel.y * resolution.x) * 3 + 2] = color.z;
}

[Differentiable]
float L2(no_diff float3 ref, float3 pred) {
    return dot(ref - pred, ref - pred) / (3 * 512 * 512);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void ComputeMain(int3 dtid: SV_DispatchThreadID) {
    // The resolution of the image, which is the same as the launch size:
    const int2 resolution = int2(512, 512);
    const int2 texel = int2(dtid.xy);

    // If the pixel is outside of the image, don't do anything:
    if (any(texel >= resolution)) return;

    GeometryHit texelHit = LoadInverseBuffer(texel);
    float3 output = float3(0);
    if (HasHit(texelHit)) {
        for (int camera_index = 1; camera_index < 2; ++camera_index) {
            const float3 cameraPos = GPUScene_camera[camera_index].posW;
            float visible = VisibilityRayQuery(SpawnVisibilityRay(texelHit, cameraPos)) ? 1 : 0;
            if (visible > 0) {
                const float2 pixel = GetScreenPixelPos(GPUScene_camera[camera_index], texelHit.position);
                // const float3 ground_truth = load_ground_truth(int2(pixel), 3 - camera_index);
                const float3 ground_truth = load_ground_truth(int2(pixel), camera_index);
                
                float3 point_light = float3(0, 15, 10);
                float3 light_dir = normalize(point_light - texelHit.position);
                BSDFEvalGeometry geom = createBSDFEvalGeometry(texelHit, normalize(cameraPos - texelHit.position), light_dir);
                float visible = VisibilityRayQuery(SpawnVisibilityRay(texelHit, point_light)) ? 1 : 0;
                float light_intensity = 3;
                float3 output = EvalBSDFValue(texelHit.texcoord, geom, visible * light_intensity);

                // var pred = diffPair(output);
                // bwd_diff(L2)(ground_truth, pred, 1.0);
                const float3 pixel_grad = (output - ground_truth) * 2. / (512 * 512 * 3);

                const float j = Jacobian(GPUScene_camera[camera_index], pixel / float2(resolution), texelHit.position, texelHit.geometryNormal);
                bwd_diff(EvalBSDFValue)(texelHit.texcoord, geom, visible * light_intensity, pixel_grad);

                set_image(u_image, texel, resolution, ground_truth);
                // set_image(u_image, texel, resolution, ground_truth);
            }
            else {
                // set_image(u_image, texel, resolution, float3(0));
            }
            return;
        }
    }
    // set_image(u_image, texel, resolution, float3(0));
}