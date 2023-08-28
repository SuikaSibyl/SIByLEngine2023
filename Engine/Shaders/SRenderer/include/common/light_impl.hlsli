#ifndef _SRENDERER_LIGHT_IMPL_HEADER_
#define _SRENDERER_LIGHT_IMPL_HEADER_

#include "../../raytracer/spt_interface.hlsli"
#include "geometry.hlsli"
#include "light.hlsli"

struct LightSample {
    float3 wi;
    float pdf;
    float3 radiance;
    float3 position;
};

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