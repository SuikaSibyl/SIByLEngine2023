#include "../../../include/common/light_impl.hlsli"
#include "../../../include/common/sampling.hlsli"
#include "../../../include/common/raycone.hlsli"
#include "../../../include/diff_descriptor_set.hlsli"
#include "../../../raytracer/spt_interface.hlsli"
#include "../../../raytracer/spt_differentiable.hlsli"
#include "../../lightmap/lightmap_common.hlsli"

[[vk::push_constant]]
cbuffer PushConstants {
    uint rand_seed;
    uint max_depth;
    uint spp;
};

// ---------------- input ----------------
const Texture2D<float4> u_deltaY; // delta Y a.k.a. adjoint rendering
// ---------------- output ----------------
RWTexture2D<float4> u_color; // output color, probably not used
// ------------------------------------------------------------
RWTexture2D<float4> u_debug;

float3 RadiativeBackpropagation(in_ref(Ray) primary_ray, in_ref(float3) delta, inout_ref(RandomSamplerState) RNG) {
    float3 radiance = float3(0);
    PrimaryPayload primary_payload;
    Intersection(primary_ray, SceneBVH, primary_payload, RNG);

    RayCone root_rc;
    root_rc.width = 0;
    root_rc.spreadAngle = pixelSpreadAngle(globalUniform.cameraData);
    float t = distance(primary_ray.origin, primary_payload.hit.position);
    root_rc = propagate(root_rc, 0, t);
    float lod = computeLOD(root_rc, primary_payload.hit.lambda, primary_ray.direction, primary_payload.hit.shadingNormal);

    float lambda = 0.5 * log2(512 * 512) + lod;
    // return t.SampleLevel(s, uv, lambda);
    u_debug[DispatchRaysIndex().xy] = float4(lambda, lod, primary_payload.hit.lambda, 1);
    
    for (uint i = 0; i < spp; ++i) {
        Ray ray = primary_ray;
        PrimaryPayload payload = primary_payload;
        RayCone rc = root_rc;
        float3 throughput = float3(1);
        
        // Iterate over the path, and do radiative backpropagation
        for (uint depth = 0; depth < max_depth; ++depth) {
            // TODO (optional): Russian roulette
            // // If we hit nothing, just output the background color:
            // if (!HasHit(payload.hit)) {
            //     u_output[pixel] = float4(float3(0), 1);
            //     return;
            // }
            if (!HasHit(payload.hit)) {
                break;
            }
            
            // DI Evaluation
            const PolymorphicLightInfo light = lights[0];
            const LightSample lightSample = SampleLight(payload.hit, light);
            Ray shadowRay = SpawnRay(payload.hit, lightSample.wi);
            shadowRay.tMax = distance(lightSample.position, payload.hit.position) - 0.01;
            const bool occluded = TraceOccludeRay(shadowRay, RNG, SceneBVH);
            const float visibility = occluded ? 0.0f : 1.0f;
            const float3 bsdf = EvalBsdf(payload.hit, -ray.direction, lightSample.wi);
            const uint materialID = geometries[payload.hit.geometryID].materialID;
            const MaterialInfo material = materials[materialID];
            const float3 emission = material.emissiveColor;
            radiance += (lightSample.radiance * bsdf * visibility / lightSample.pdf + emission) * throughput;
            // if (depth == 1) {
            //     u_debug[DispatchRaysIndex().xy] = float4(debug, 1);
            // }
            // BSDF derivatives
            // --------------------------------------------
            float lod = computeLOD(rc, payload.hit.lambda, ray.direction, payload.hit.shadingNormal);

            float bsdf_pdf; float3 debug;
            Ray bsdf_ray = SpawnBsdfRay(payload.hit, -ray.direction, RNG, bsdf_pdf);
            float3 bsdf_next = EvalBsdfDiff(payload.hit, -ray.direction, bsdf_ray.direction, delta * throughput / bsdf_pdf, debug, lod);
            throughput *= bsdf_next / bsdf_pdf;
            ray = bsdf_ray;

            Intersection(ray, SceneBVH, payload, RNG);
            float t = distance(ray.origin, payload.hit.position);
            rc = propagate(rc, 0, t);
        }
    }

    // TODO (optional): envmap
    return radiance / spp;
}

[shader("raygeneration")]
void RgenMain() {
    // The resolution of the image, which is the same as the launch size:
    const int2 resolution = int2(DispatchRaysDimensions().xy);
    const int2 pixel = int2(DispatchRaysIndex().xy);
    // If the pixel is outside of the image, don't do anything:
    if (any(pixel >= resolution)) return;
    RandomSamplerState RNG = InitRandomSampler(pixel, rand_seed);
    const float2 jitter_pixel = pixel + GetNextRandomFloat2(RNG) - float2(0.5); // manually jitter
    const Ray primaryRay = generateRay(jitter_pixel, resolution, false, globalUniform.cameraData);
    
    // load deltaY from the gradient buffer
    const float3 delta = u_deltaY[pixel].xyz;
    // Do radiative backgropagation, and get the radiance
    const float3 radiance = RadiativeBackpropagation(primaryRay, delta, RNG); 
}