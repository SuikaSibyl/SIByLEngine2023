#ifndef _SRENDERER_LIGHT_IMPL_HEADER_
#define _SRENDERER_LIGHT_IMPL_HEADER_

#include "../../raytracer/spt_interface.hlsli"
#include "../../include/common/sampling.hlsli"
#include "geometry.hlsli"
#include "light.hlsli"

struct LightSample {
    float3 wi;
    float pdf;
    float3 radiance;
    float pdf_direction;
    float3 position;
    float3 normal;
};

struct LightRaySample {
    Ray ray;
    float3 radiance;
    float pdf_position;
    float3 normal;
    float pdf_direction;
};

LightRaySample EmptyLightRaySample() {
    LightRaySample sample;
    sample.radiance = float3(0.f);
    sample.pdf_position = 0.f;
    sample.normal = float3(0.f);
    sample.pdf_direction = 0.f;
    return sample;
}

/**********************************************************************
** Directional Light
**********************************************************************/
struct DirectionalLight {
    float3 intensity; // intensity of the directional light
    float3 direction; // direction of the directional light
};

/**
 * Pack the directional light
 * @param light the directional light
 * @return the unpacked directional light
 */
DirectionalLight unpackDirectionalLight(in_ref(PolymorphicLightInfo) light) {
    DirectionalLight dlight;
    dlight.intensity = unpackLightColor(light);
    dlight.direction = Unorm32OctahedronToUnitVector(light.databyte0);
    return dlight;
}

/**
 * Sample a point on the directional light
 * @param hit the hit point
 * @param light the directional light
 * @return the light sample
 */
LightSample sample_li(
    in_ref(GeometryHit) hit,
    in_ref(DirectionalLight) light
) {
    LightSample sample;
    sample.wi = -light.direction;
    sample.pdf = 1.f;
    sample.radiance = light.intensity;
    sample.position = hit.position + sample.wi * 1e5f;
    return sample;
}

/**
 * Sample a point light with a random point and direction
 */
LightRaySample sample_le(
    in_ref(DirectionalLight) light,
    inout_ref(RandomSamplerState) RNG,
    in_ref(float4) boundingSphere
) {
    const float3x3 obn = createONB(light.direction);
    const float2 cd = uniformSampleDisk(GetNextRandomFloat2(RNG));
    const float3 origin = boundingSphere.xyz - boundingSphere.w * light.direction 
                       + boundingSphere.w * (cd.x * obn[0] + cd.y * obn[1]);
    // Generate the light ray sample
    LightRaySample sample;
    sample.ray = SpawnRay(origin, light.direction, light.direction);
    sample.normal = light.direction;
    sample.pdf_position = 1. / (k_pi * boundingSphere.w * boundingSphere.w);
    sample.pdf_direction = 1.f;
    sample.radiance = light.intensity;
    return sample;
}

/**********************************************************************
** Point Light
**********************************************************************/
struct PointLight {
    float3 intensity;
    float3 position;
};

/**
 * Pack the point light
 * @param light the point light
 * @return the unpacked point light
 */
PointLight unpackPointLight(in_ref(PolymorphicLightInfo) light) {
    PointLight plight;
    plight.intensity = unpackLightColor(light);
    plight.position = light.center;
    return plight;
}

/**
 * Sample a point light
 * @param hit the geometry hit
 * @param light the point light
 * @return the light sample
 */
LightSample sample_li(
    in_ref(GeometryHit) hit,
    in_ref(PointLight) light
) {
    LightSample sample;
    sample.wi = normalize(light.position - hit.position);
    sample.pdf = 1.f;
    sample.radiance = light.intensity / distance_squared(light.position, hit.position);
    sample.position = light.position;
    return sample;
}

/**
 * Sample a point light with a random point and direction
 */
LightRaySample sample_le(
    in_ref(PointLight) light,
    inout_ref(RandomSamplerState) RNG
) {
    const float3 direction = UniformOnSphere(GetNextRandomFloat2(RNG));
    LightRaySample sample;
    sample.ray = SpawnRay(light.position, direction, direction);
    sample.normal = direction;
    sample.pdf_position = 1.f;
    sample.pdf_direction = PdfUniformOnSphere();
    sample.radiance = light.intensity;
    return sample;
}

/**
 * Pdf of sampling a point light with a random point and direction
 */
// float2 pdf_le(
//     in_ref(PointLight) light
// ) {
// }

/**********************************************************************
** Spot Light
**********************************************************************/
struct SpotLight {
    float3 intensity;      // intensity of the spot light
    float cosTotalWidth;   // cosine of the total width of the spot light
    float3 position;       // position of the spot light
    float cosFalloffStart; // cosine of the falloff start of the spot light
    float3 direction;      // direction of the spot light
};

/**
 * Compute the falloff of the spot light
 * @param light the spot light
 * @param w the direction to the light
 * @return the falloff of the spot light
 */
float spot_light_falloff(in_ref(SpotLight) light, in_ref(float3) w) {
    const float3x3 w2l = createONB(light.direction);
    const float3 wl = to_local(w2l, w);
    const float cosTheta = wl.z;
    if (cosTheta < light.cosTotalWidth) return 0.f;
    if (cosTheta > light.cosFalloffStart) return 1.f;
    // Compute falloff inside spotlight cone
    const float delta = (cosTheta - light.cosTotalWidth) /
                        (light.cosFalloffStart - light.cosTotalWidth);
    return (delta * delta) * (delta * delta);
}

/**
 * Pack the spot light
 * @param light the spot light
 * @return the unpacked spot light
 */
SpotLight unpackSpotLight(in_ref(PolymorphicLightInfo) light) {
    SpotLight slight;
    slight.intensity = unpackLightColor(light);
    slight.position = light.center;
    slight.direction = Unorm32OctahedronToUnitVector(light.databyte0);
    slight.cosTotalWidth = asfloat(light.databyte1);
    slight.cosFalloffStart = asfloat(light.databyte2);
    return slight;
}

/**
 * Sample a point on the spot light
 * @param hit the hit point
 * @param light the spot light
 * @return the light sample
 */
LightSample sample_li(
    in_ref(GeometryHit) hit,
    in_ref(SpotLight) light
) {
    LightSample sample;
    sample.wi = normalize(light.position - hit.position);
    sample.pdf = 1.f;
    sample.radiance = light.intensity * spot_light_falloff(light, -sample.wi) 
        / distance_squared(light.position, hit.position);
    sample.position = light.position;
    return sample;
}

/**
 * Sample a point light with a random point and direction
 */
LightRaySample sample_le(
    in_ref(SpotLight) light,
    inout_ref(RandomSamplerState) RNG
) {
    // Uniformly sample a direction of the spot light
    const float3x3 w2l = createONB(light.direction);
    const float3 dir_ls = SampleUniformCone(GetNextRandomFloat2(RNG), light.cosTotalWidth);
    const float3 dir_ws = to_world(w2l, dir_ls);
    // Generate the light sample
    LightRaySample sample;
    sample.ray = SpawnRay(light.position, dir_ws, dir_ws);
    sample.pdf_position = 1.f;
    sample.pdf_direction = PdfUniformCone(light.cosTotalWidth);
    sample.radiance = light.intensity * spot_light_falloff(light, dir_ws);
    sample.normal = dir_ws;
    return sample;
}

/**********************************************************************
** Polymorphic Light
**********************************************************************/
LightSample SampleLight(
    in_ref(GeometryHit) hit,
    in_ref(PolymorphicLightInfo) light
) {
    const PolymorphicLightType type = getLightType(light);
    switch (type) {
    case PolymorphicLightType::kDirectional:
        return sample_li(hit, unpackDirectionalLight(light));
    case PolymorphicLightType::kPoint:
        return sample_li(hit, unpackPointLight(light));
    case PolymorphicLightType::kSpot:
        return sample_li(hit, unpackSpotLight(light));
    }
}

struct LightSampleInfo {
    float4 boundingSphere;
};

LightRaySample SampleLightRay(
    in_ref(PolymorphicLightInfo) light,
    in_ref(LightSampleInfo) info,
    inout_ref(RandomSamplerState) RNG
) {
    const PolymorphicLightType type = getLightType(light);
    switch (type) {
    case PolymorphicLightType::kDirectional:
        return sample_le(unpackDirectionalLight(light), RNG, info.boundingSphere);
    case PolymorphicLightType::kPoint:
        return sample_le(unpackPointLight(light), RNG);
    case PolymorphicLightType::kSpot:
        return sample_le(unpackSpotLight(light), RNG);
    }
    return EmptyLightRaySample();
}

float3 EvaluateDirectLight(
    in_ref(Ray) previousRay,
    in_ref(GeometryHit) hit,
    in_ref(PolymorphicLightInfo) light,
    inout_ref(RandomSamplerState) RNG
) {
    const LightSample lightSample = SampleLight(hit, light);
    Ray shadowRay = SpawnRay(hit, lightSample.wi);
    shadowRay.tMax = distance(lightSample.position, hit.position) - 0.01;
    const bool occluded = TraceOccludeRay(shadowRay, RNG, SceneBVH);
    const float visibility = occluded ? 0.0f : 1.0f;
    const float3 bsdf = EvalBsdf(hit, -previousRay.direction, lightSample.wi);
    return lightSample.radiance * bsdf * visibility / lightSample.pdf;
}

float3 EvaluateDirectLight(
    in_ref(Ray) previousRay,
    in_ref(ShadingSurface) surface,
    in_ref(PolymorphicLightInfo) light,
    inout_ref(RandomSamplerState) RNG
) {
    const GeometryHit hit = CreateGeometryHit(surface);
    const LightSample lightSample = SampleLight(hit, light);
    Ray shadowRay = SpawnRay(surface, lightSample.wi);
    shadowRay.tMax = distance(lightSample.position, surface.worldPos) - 0.01;
    const bool occluded = TraceOccludeRay(shadowRay, RNG, SceneBVH);
    const float visibility = occluded ? 0.0f : 1.0f;
    const float3 bsdf = EvalBsdf(surface, -previousRay.direction, lightSample.wi);
    return lightSample.radiance * bsdf * visibility / lightSample.pdf;
}

#endif // _SRENDERER_LIGHT_IMPL_HEADER_