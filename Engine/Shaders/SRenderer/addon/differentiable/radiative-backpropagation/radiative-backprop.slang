#include "../../../include/common/light_impl.hlsli"
#include "../../../include/common/sampling.hlsli"
#include "../../../include/diff_descriptor_set.hlsli"
#include "../../../raytracer/spt_interface.hlsli"
#include "../../../raytracer/spt_differentiable.hlsli"

[[vk::push_constant]]
cbuffer PushConstants {
    uint rand_seed;
    uint max_depth;
};

// ---------------- input ----------------
const Texture2D<float4> u_deltaY; // delta Y a.k.a. adjoint rendering
// ---------------- output ----------------
RWTexture2D<float4> u_color; // output color, probably not used
// ------------------------------------------------------------

RWTexture2D<float4> u_debug;

float3 RadiativeBackpropagation(in_ref(Ray) ray, in_ref(float3) delta, inout_ref(RandomSamplerState) RNG) {
    float3 radiance = float3(0);
    float3 throughput = float3(1);
    PrimaryPayload payload;
    u_debug[DispatchRaysIndex().xy] = float4(float3(0), 1);
    // Iterate over the path, and do radiative backpropagation
    for (uint depth = 0; depth < max_depth; ++depth) {
        // TODO (optional): Russian roulette
        Intersection(ray, SceneBVH, payload, RNG);

        // // If we hit nothing, just output the background color:
        // if (!HasHit(payload.hit)) {
        //     u_output[pixel] = float4(float3(0), 1);
        //     return;
        // }
        if (!HasHit(payload.hit)) {
            return radiance;
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
        
        float bsdf_pdf; float3 debug;
        Ray bsdf_ray = SpawnBsdfRay(payload.hit, -ray.direction, RNG, bsdf_pdf);
        float3 bsdf_next = EvalBsdfDiff(payload.hit, -ray.direction, bsdf_ray.direction, delta * throughput / bsdf_pdf, debug);
        throughput *= bsdf_next / bsdf_pdf;
        ray = bsdf_ray;
    }
    // TODO (optional): envmap
    return radiance;
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
    // Output the radiance of the pixel
    // In a biased estimator, it could be used 
    // in gradient evaluation of the next frame.
    u_color[pixel] = float4(radiance, 1);
}