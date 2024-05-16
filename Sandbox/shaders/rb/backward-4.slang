#include "../srenderer/materials/lambertian.hlsli"
#include "../srenderer/materials/roughplastic.hlsli"
#include "../srenderer/rayquery.hlsli"

[[vk::push_constant]]
cbuffer PushConstants {
    int random_seed;
};

RWStructuredBuffer<float> u_image;
RWStructuredBuffer<float> u_groundtruth;
RaytracingAccelerationStructure TexCoordScene_tlas;
RWByteAddressBuffer u_spp;
RWStructuredBuffer<uint32_t> u_index;
RWStructuredBuffer<float> u_index_pmf;
RWStructuredBuffer<float> u_camera_weight;


void AddSpp(int2 pixel) {
    u_spp.InterlockedAddF32((pixel.x + pixel.y * 512) * 4, 1);
}

struct TexcoordSamplePayload {
    GeometryHit hit;
    float pdf;
};

float4 fetchCameraImportance(int2 texel) {
    float4 weights;
    weights[0] = u_camera_weight[(texel.x + texel.y * 512) + 512 * 512 * 0];
    weights[1] = u_camera_weight[(texel.x + texel.y * 512) + 512 * 512 * 1];
    weights[2] = u_camera_weight[(texel.x + texel.y * 512) + 512 * 512 * 2];
    weights[3] = u_camera_weight[(texel.x + texel.y * 512) + 512 * 512 * 3];
    // float sum = weights[0] + weights[1] + weights[2] + weights[3];
    // weights /= (sum+1e-6f);
    return weights;
}

TexcoordSamplePayload TexcoordRayQuery(in const Ray ray) {
    // Initialize the payload
    TexcoordSamplePayload payload;
    payload.pdf = 0.f;
    SetHit(payload.hit, false);
    // Apply the ray query
    RayQuery<RAY_FLAG_NONE> q;
    q.TraceRayInline(TexCoordScene_tlas, 0, 0xff, ToRayDesc(ray));
    while (q.Proceed()) {
        switch (q.CandidateType()) {
        case CANDIDATE_NON_OPAQUE_TRIANGLE: {
            if (true) {
                q.CommitNonOpaqueTriangleHit();
            }
            if (false) {
                q.Abort();
            }
            break;
        }
        }
    }
    // Process the result
    switch (q.CommittedStatus()) {
    case COMMITTED_TRIANGLE_HIT: {
        SetHit(payload.hit, true);
        // Fetch the hit information
        const int primitiveID = q.CommittedPrimitiveIndex();
        const int geometryID = q.CommittedInstanceIndex() + q.CommittedGeometryIndex();
        const float2 bary = q.CommittedTriangleBarycentrics();
        const float3 barycentrics = float3(1 - bary.x - bary.y, bary.x, bary.y);
        const GeometryData geometry = GPUScene_geometry[geometryID];
        const int3 index = int3(fetchTriangleIndices(geometry, primitiveID));

        float3 vertexPositions[3];
        vertexPositions[0] = fetchVertexPosition(index[0] + int(geometry.vertexOffset));
        vertexPositions[1] = fetchVertexPosition(index[1] + int(geometry.vertexOffset));
        vertexPositions[2] = fetchVertexPosition(index[2] + int(geometry.vertexOffset));

        const float4x4 o2w = ObjectToWorld(geometry);
        const float3 positionOS = interpolate(vertexPositions, barycentrics);
        const float3 positionWS = mul(float4(positionOS, 1.0), o2w).xyz;
        payload.hit.position = positionWS;

        float2 vertexUVs[3];
        vertexUVs[0] = fetchVertexTexCoord(index[0] + int(geometry.vertexOffset));
        vertexUVs[1] = fetchVertexTexCoord(index[1] + int(geometry.vertexOffset));
        vertexUVs[2] = fetchVertexTexCoord(index[2] + int(geometry.vertexOffset));
        float2 uv = interpolate(vertexUVs, barycentrics);
        payload.hit.texcoord = uv;

        float3 objectSpaceFlatNormal = normalize(cross(
            vertexPositions[1] - vertexPositions[0],
            vertexPositions[2] - vertexPositions[0]));

        float4x4 o2wn = ObjectToWorldNormal(geometry);
        const float3 flatNormal = normalize(mul(float4(objectSpaceFlatNormal, 0.0), o2wn).xyz);
        payload.hit.geometryNormal = flatNormal;

        float3 normals[3];
        normals[0] = fetchVertexNormal(index[0] + int(geometry.vertexOffset));
        normals[1] = fetchVertexNormal(index[1] + int(geometry.vertexOffset));
        normals[2] = fetchVertexNormal(index[2] + int(geometry.vertexOffset));
        float3 vertexNormalOS = interpolate(normals, barycentrics);
        float3 gvertexNormalWS = normalize(mul(float4(vertexNormalOS, 0.0), o2wn).xyz);
        payload.hit.shadingNormal = gvertexNormalWS;

        float3 tangents[3];
        tangents[0] = fetchVertexTangent(index[0] + int(geometry.vertexOffset));
        tangents[1] = fetchVertexTangent(index[1] + int(geometry.vertexOffset));
        tangents[2] = fetchVertexTangent(index[2] + int(geometry.vertexOffset));
        float3 tangentOS = interpolate(tangents, barycentrics);
        float4 tangentWS = float4(normalize(mul(float4(tangentOS, 0), o2w).xyz), geometry.oddNegativeScaling);
        payload.hit.tangent = tangentWS;
        
        // transform vertexPositions to world space
        vertexPositions[0] = mul(float4(vertexPositions[0], 1.0), o2w).xyz;
        vertexPositions[1] = mul(float4(vertexPositions[1], 1.0), o2w).xyz;
        vertexPositions[2] = mul(float4(vertexPositions[2], 1.0), o2w).xyz;

        // compute the 2d area of vertexUVs triangle
        const float texcoord_area = 0.5 * abs((vertexUVs[0].x * (vertexUVs[1].y - vertexUVs[2].y) 
            + vertexUVs[1].x * (vertexUVs[2].y - vertexUVs[0].y) + vertexUVs[2].x * (vertexUVs[0].y - vertexUVs[1].y)));
        // compute the 3d surface area of positionWS triangle
        const float3 edge1 = vertexPositions[1] - vertexPositions[0];
        const float3 edge2 = vertexPositions[2] - vertexPositions[0];
        const float3 normal = cross(edge1, edge2);
        const float area = 0.5 * length(normal);
        payload.pdf = texcoord_area / area;
        
        break;
    }
    case COMMITTED_PROCEDURAL_PRIMITIVE_HIT: {
        break;
    }
    case COMMITTED_NOTHING: {
        break;
    }
    }

    return payload;
}

RWStructuredBuffer<float> sampleBuffer;
RWStructuredBuffer<float> debugBuffer;

void SetSampleBuffer(float3 sample, int2 pixel) {
    sampleBuffer[(pixel.y * 512 + pixel.x) * 3 + 0] = sample.x;
    sampleBuffer[(pixel.y * 512 + pixel.x) * 3 + 1] = sample.y;
    sampleBuffer[(pixel.y * 512 + pixel.x) * 3 + 2] = sample.z;
}

RWStructuredBuffer<float> u_image_grad;
RWStructuredBuffer<float> u_basecolor;
RWStructuredBuffer<float> u_roughness;
RWByteAddressBuffer u_basecolor_grad;
RWByteAddressBuffer u_roughness_grad;

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

float3 load_ground_truth(int2 pixel, int view) {
    // int index[4] = { 3, 0, 1, 2 };
    int index[4] = { 0, 1, 2, 3 };
    float3 color;
    color.x = u_groundtruth[(index[view] * 512 * 512 * 3) + (pixel.x + pixel.y * 512) * 3 + 0];
    color.y = u_groundtruth[(index[view] * 512 * 512 * 3) + (pixel.x + pixel.y * 512) * 3 + 1];
    color.z = u_groundtruth[(index[view] * 512 * 512 * 3) + (pixel.x + pixel.y * 512) * 3 + 2];
    return color;
}

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

void set_image(RWStructuredBuffer<float> image, int2 pixel, int2 resolution, float3 color) {
    image[(pixel.x + pixel.y * resolution.x) * 3 + 0] = color.x;
    image[(pixel.x + pixel.y * resolution.x) * 3 + 1] = color.y;
    image[(pixel.x + pixel.y * resolution.x) * 3 + 2] = color.z;
}

int SampleWeight4(float4 weights, float rand, out float pdf) {
    if (rand < weights[0]) {
        pdf = weights[0];
        return 0;
    }
    else if (rand < weights[0] + weights[1]) {
        pdf = weights[1];
        return 1;
    }
    else if (rand < weights[0] + weights[1] + weights[2]) {
        pdf = weights[2];
        return 2;
    }
    else {
        pdf = weights[3];
        return 3;
    }
}

[shader("compute")]
[numthreads(16, 16, 1)]
void ComputeMain(int3 dtid: SV_DispatchThreadID) {
    // const int2 texel = dtid.xy;
    // RandomSamplerState RNG = InitRandomSampler(dtid.xy, random_seed);
    // const int globalIDX = dtid.x + dtid.y * 512;
    // const int indexIDX = u_index[globalIDX];
    // const bool invalidIDX = indexIDX == -1;
    // int2 pixelIDX = invalidIDX ? int2(0, 0) : int2(indexIDX >> 16, indexIDX & 0xFFFF);
    // const float adaptiveProb = u_index_pmf[globalIDX];

    // float2 jitter = GetNextRandomFloat2(RNG);
    // Ray texSampleRay;
    // texSampleRay.origin = float3((pixelIDX + jitter) / 512, -1);
    // texSampleRay.direction = float3(0, 0, 1);
    // texSampleRay.tMin = 0;
    // texSampleRay.tMax = 1000;
    // TexcoordSamplePayload samplePayload = TexcoordRayQuery(texSampleRay);
    // if (samplePayload.pdf > 0) {
    //     SetSampleBuffer(samplePayload.hit.position, dtid.xy);
    //     debugBuffer[dtid.y * 512 + dtid.x] = samplePayload.pdf;
    // }
    // else {
    //     SetSampleBuffer(float3(dtid.x, 0, dtid.y) / float3(512), dtid.xy);
    //     debugBuffer[dtid.y * 512 + dtid.x] = samplePayload.pdf;
    // }
    
    // const int2 resolution = int2(512, 512);

    // float4 camera_weights = fetchCameraImportance(pixelIDX);
    // float sum_weights = dot(camera_weights, float4(1, 1, 1, 1));
    // camera_weights /= sum_weights;
    // float camera_pdf = 1.f;
    // int camera_index = SampleWeight4(camera_weights, GetNextRandom(RNG), camera_pdf);
    // // float camera_pdf = 0.25;
    // // int camera_index = GetNextRandomUint(RNG) % 4;
    

    // if (samplePayload.pdf > 0) {
    //     if (true) {
    //         const float3 cameraPos = GPUScene_camera[camera_index].posW;
    //         float visible = VisibilityRayQuery(SpawnVisibilityRay(samplePayload.hit, cameraPos)) ? 1 : 0;
    //         if (visible > 0) {
    //             const float2 pixel = GetScreenPixelPos(GPUScene_camera[camera_index], samplePayload.hit.position);
    //             const float3 ground_truth = load_ground_truth(int2(pixel), camera_index);
                
    //             float3 point_light = float3(0, 15, 10);
    //             float3 light_dir = normalize(point_light - samplePayload.hit.position);
    //             BSDFEvalGeometry geom = createBSDFEvalGeometry(samplePayload.hit, normalize(cameraPos - samplePayload.hit.position), light_dir);
    //             float visible = VisibilityRayQuery(SpawnVisibilityRay(samplePayload.hit, point_light)) ? 1 : 0;
    //             float light_intensity = 3;
    //             // float3 output = EvalBSDFValue(samplePayload.hit.texcoord, geom, visible * light_intensity);
                
    //             // var pred = diffPair(output);
    //             // bwd_diff(L2)(ground_truth, pred, 1.0);
    //             // const float3 pixel_grad = (output - ground_truth) * 2. / (512 * 512 * 3);

    //             const float j = Jacobian(GPUScene_camera[camera_index], pixel / float2(resolution), samplePayload.hit.position, samplePayload.hit.geometryNormal);
    //             const float3 grad_scaled = ground_truth / (samplePayload.pdf * j * adaptiveProb * camera_pdf);
    //             if (any(isnan(grad_scaled) || isinf(grad_scaled))) {
    //             } else {
    //                 bwd_diff(EvalBSDFValue)(texSampleRay.origin.xy, geom, visible * light_intensity, grad_scaled);
    //             }

    //             // set_image(u_image, texel, resolution, ground_truth);
    //             // set_image(u_image, texel, resolution, ground_truth);
    //         }
    //         else {
    //             // set_image(u_image, texel, resolution, float3(0, 0, 0));
    //         }
    //     }
    // }
    // else {
    //     // set_image(u_image, texel, resolution, float3(0, 0, 0));
    // }

    // The resolution of the image, which is the same as the launch size:
    const int2 resolution = int2(512, 512);
    // If the pixel is outside of the image, don't do anything:
    if (any(dtid.xy >= resolution)) return;

    RandomSamplerState RNG = InitRandomSampler(dtid.xy, random_seed);
    const int2 pixel = int2(GetNextRandomUint(RNG) % 512, GetNextRandomUint(RNG) % 512);
    const float2 jitter = GetNextRandomFloat2(RNG) - 0.5f;
    const int camera_index = GetNextRandomUint(RNG) % 4;

    const Ray primaryRay = generateRay(
        pixel + jitter,
        resolution,
        false,
        GPUScene_camera[camera_index]
    );
    float2 p = (float2(pixel) + float2(0.5f, 0.5f)) / float2(resolution);
    
    // Ray query to check visibility
    // float3 pixel_grad = load_image_grad(u_image_grad, pixel, resolution);
    const float3 pixel_grad = load_ground_truth(int2(pixel), camera_index);
    
    PrimaryPayload payload = PrimaryRayQuery(primaryRay);
    if (HasHit(payload.hit)) {
        // output = payload.hit.position * 0.5f + 0.5f;
        // output = float3(payload.hit.texcoord, 0);
        float3 point_light = float3(0, 15, 10);
        float3 light_dir = normalize(point_light - payload.hit.position);
        BSDFEvalGeometry geom = createBSDFEvalGeometry(payload.hit, -primaryRay.direction, light_dir);
        float visible = VisibilityRayQuery(SpawnVisibilityRay(payload.hit, point_light)) ? 1 : 0;
        float light_intensity = 3;
        bwd_diff(EvalBSDFValue)(payload.hit.texcoord, geom, visible * light_intensity, pixel_grad * 4);
    }
}