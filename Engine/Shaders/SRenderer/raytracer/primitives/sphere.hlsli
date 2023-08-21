#ifndef SRENDERER_RAYTRACER_PLUGIN_SPHERE_PRIMITIVE_
#define SRENDERER_RAYTRACER_PLUGIN_SPHERE_PRIMITIVE_

#include "../../include/common/cpp_compatible.hlsli"
#include "../../include/common/geometry.hlsli"
#include "../../include/common/math.hlsli"
#include "../../include/common/sampling.hlsli"
#include "../../include/common/raycast.hlsli"

/**
 * Compute intersection of a ray and a sphere
 * @ref: http://viclw17.github.io/2018/07/16/raytracing-ray-sphere-intersection
 * @ref: <ray tracing gems> https://link.springer.com/content/pdf/10.1007/978-1-4842-4427-2_7.pdf
 */
float hitSphere(
    in float3 center, in float radius,
    in float3 origin, in float3 direction)
{
    const float3 oc = origin - center;
    const float a = dot(direction, direction);
    const float b = 2.0 * dot(oc, direction);
    const float c = dot(oc, oc) - radius * radius;
    // Degenerated case
    if (a == 0) {
        if (b == 0)
            return -1.0; // if no solve
        return -c / b;
    }
    float3 term = oc - (dot(oc, direction) * direction);
    float discriminant = 4 * a * (radius * radius - dot(term, term));
    if (discriminant < 0)
        return -1.0; // if no solve
    const float root_discriminant = sqrt(discriminant);
    float t0, t1;
    if (b >= 0) {
        t0 = (-b - root_discriminant) / (2 * a);
        t1 = 2 * c / (-b - root_discriminant);
    } else {
        t0 = 2 * c / (-b + root_discriminant);
        t1 = (-b + root_discriminant) / (2 * a);
    }
    float t = -1;
    if (t0 >= 0 && t0 < k_inf)
        t = t0;
    if (t1 >= 0 && t1 < k_inf && t < 0)
        t = t1;
    return t;
}

float3 sampleSphere(
    in float3 center, in float radius, 
    in float3 ref_point, in float2 uv)
{
    bool validRef = validRefPoint(ref_point);
    // If the reference point is inside the sphere, just sample the whole sphere uniformly
    if (!validRef || distance_squared(ref_point, center) < radius * radius) {
        float z = 1 - 2 * uv.x;
        float r_ = sqrt(max(0., 1 - z * z));
        float phi = 2 * k_pi * uv.y;
        float3 offset = normalize(float3(r_ * cos(phi), r_ * sin(phi), z));
        return center + radius * offset;
    }
    // Otherwise sample a ray inside a cone towards the sphere center.
    // Build a coordinate system with n pointing towards the sphere
    float3 dir_to_center = normalize(center - ref_point);
    float3x3 frame = createONB(dir_to_center);
    float sin_theta_max_sq = radius * radius / distance_squared(ref_point, center);
    float cos_theta_max = sqrt(max(0.f, 1 - sin_theta_max_sq));
    float3 n_on_sphere;
    if (cos_theta_max > 0.999999) {
        n_on_sphere = -normalize(mul(frame, cosineSampleHemisphere(uv)));
    }
    else {
        // Uniformly interpolate between 1 (angle 0) and max
        float cos_theta = (1 - uv[0]) + uv[0] * cos_theta_max;
        float sin_theta = sqrt(max(0.f, 1 - cos_theta * cos_theta));
        float phi = uv[1] * 2 * k_pi;
        // Compute angle from center of sphere to sampled point on surface
        float dc = distance(ref_point, center);
        float ds = dc * cos_theta - sqrt(max(0.f, radius * radius - dc * dc * sin_theta * sin_theta));
        float cos_alpha = (dc * dc + radius * radius - ds * ds) / (2 * dc * radius);
        float sin_alpha = sqrt(max(0.f, 1 - cos_alpha * cos_alpha));
        // get the normal of the sample
        n_on_sphere = -normalize(mul(float3(sin_alpha * cos(phi), sin_alpha * sin(phi), cos_alpha), frame));
    }
    // Compute surface normal and sampled point on sphere
    // Add negative sign since normals point outwards
    float3 p_on_sphere = radius * n_on_sphere + center;
    return p_on_sphere;
}

float surface_area_sphere(in float radius) {
    return 4 * k_pi * radius * radius;
}

float sampleSpherePdf(
    in float3 center, in float r,
    in float3 ref_point, in float3 sample_pos)
{
    bool validRef = validRefPoint(ref_point);
    const float3 p_on_sphere = sample_pos;
    const float3 n_on_sphere = normalize(sample_pos - center);
    const float3 dir = normalize(p_on_sphere - ref_point);

    if (!validRef || distance_squared(ref_point, center) < r * r) {
        // If the reference point is inside the sphere, just sample the whole sphere uniformly
        return 1. / surface_area_sphere(r)
               * distance_squared(ref_point, p_on_sphere) / abs(dot(n_on_sphere, dir));
    }
    float sin_theta_max_sq = r * r / distance_squared(ref_point, center);
    float cos_theta_max = sqrt(max(0.f, 1 - sin_theta_max_sq));
    if (cos_theta_max > 0.999999) {
        float3 mdir_to_center = normalize(ref_point - center);
        return cosineHemispherePdf(abs(dot(mdir_to_center, normalize(sample_pos - center)))) 
            * distance_squared(ref_point, p_on_sphere) / abs(dot(n_on_sphere, dir));
    }
    else {
        // Uniform sampling PDF of a cone.
        float pdf_solid_angle = 1 / (2 * k_pi * (1 - cos_theta_max));
        // Convert it back to area measure
        // Output the result
        return pdf_solid_angle;
    }
}

// ray tracing apis
#ifdef _SRENDERER_SPT_INTERFACEC_HEADER_
#include "../../include/scene_descriptor_set.hlsli"
#include "../spt_implement.hlsli"
#include "../spt_interface.hlsli"

[shader("intersection")]
void SphereIntersection() {
    // ray data
    const float3 ray_origin = WorldRayOrigin();
    const float3 ray_direction = WorldRayDirection();
    // Sphere data
    const int geometryID = int(InstanceID() + GeometryIndex());
    const GeometryInfo geometryInfo = geometries[geometryID];
    const float4x4 o2w = ObjectToWorld(geometryInfo);
    const float3 sphere_center = mul(float4(0, 0, 0, 1), o2w).xyz;
    const float sphere_radius = length(mul(float4(1, 0, 0, 1), o2w).xyz - sphere_center);
    // Sphere intersection
    const float tHit = hitSphere(sphere_center, sphere_radius, ray_origin, ray_direction);
    if (tHit > 0) ReportHit(tHit, 1, 1);
}

[shader("closesthit")]
void SphereClosestHit(
    inout_ref(PrimaryPayload) payload: SV_RayPayload)
{
    const uint primitiveID = PrimitiveIndex();
    const uint geometryID = InstanceID() + GeometryIndex();

    const float3 ray_origin = WorldRayOrigin();
    const float3 ray_direction = WorldRayDirection();

    // Sphere data
    const GeometryInfo geometryInfo = geometries[geometryID];
    const float4x4 o2w = ObjectToWorld(geometryInfo);
    const float4x4 o2wn = ObjectToWorldNormal(geometryInfo);
    const float3 sphere_center = mul(float4(0, 0, 0, 1), o2w).xyz;
    const float sphere_radius = length(mul(float4(1, 0, 0, 1), o2w).xyz - sphere_center);
    // Record the intersection
    // *: we should re-normalize it, using tHit is super unstable
    const float3 hitPoint = sphere_center + sphere_radius * normalize(ray_origin + ray_direction * RayTCurrent() - sphere_center);
    const float3 geometric_normal = normalize(hitPoint - sphere_center);
    const float3 cartesian = normalize(mul(o2wn, float4(geometric_normal, 0)).xyz);
    // We use the spherical coordinates as uv
    // We use the convention that y is up axis.
    // https://en.wikipedia.org/wiki/Spherical_coordinate_system#Cartesian_coordinates
    const float elevation = acos(clamp(cartesian.y, -1., 1.));
    const float azimuth = atan2(cartesian.z, cartesian.x);
    float3x3 TBN = createFrame(geometric_normal);

    payload.hit.position = hitPoint;
    payload.hit.geometryID = geometryID;
    payload.hit.flatNormal = geometric_normal;
    payload.hit.primitiveID = primitiveID;
    payload.hit.geometryNormal = geometric_normal;
    payload.hit.barycentric = float2(0, 0);
    payload.hit.texcoord = float2(-azimuth * k_inv_2_pi, elevation * k_inv_pi);
    payload.hit.tangent = float4(TBN[0], 0);
    SetHit(payload.hit, true);
}

[shader("anyhit")]
void SphereAnyHit(
    inout_ref(PrimaryPayload) payload: SV_RayPayload)
{
    const uint primitiveID = PrimitiveIndex();
    const uint geometryID = InstanceID() + GeometryIndex();

    const float3 ray_origin = WorldRayOrigin();
    const float3 ray_direction = WorldRayDirection();

    // Sphere data
    const GeometryInfo geometryInfo = geometries[geometryID];
    const float4x4 o2w = ObjectToWorld(geometryInfo);
    const float4x4 o2wn = ObjectToWorldNormal(geometryInfo);
    const float3 sphere_center = mul(float4(0, 0, 0, 1), o2w).xyz;
    const float sphere_radius = length(mul(float4(1, 0, 0, 1), o2w).xyz - sphere_center);
    // Record the intersection
    // *: we should re-normalize it, using tHit is super unstable
    const float3 hitPoint = sphere_center + sphere_radius * normalize(ray_origin + ray_direction * RayTCurrent() - sphere_center);
    const float3 geometric_normal = normalize(hitPoint - sphere_center);
    const float3 cartesian = normalize(mul(o2wn, float4(geometric_normal, 0)).xyz);
    // We use the spherical coordinates as uv
    // We use the convention that y is up axis.
    // https://en.wikipedia.org/wiki/Spherical_coordinate_system#Cartesian_coordinates
    const float elevation = acos(clamp(cartesian.y, -1., 1.));
    const float azimuth = atan2(cartesian.z, cartesian.x);

    float2 uv = float2(-azimuth * k_inv_2_pi, elevation * k_inv_pi);
    const MaterialInfo material = materials[geometryInfo.materialID];
    const float alpha = textures[material.baseOrDiffuseTextureIndex].SampleLevel(uv, 0).w;

    const float rnd = GetNextRandom(payload.RND);
    if (rnd > alpha) {
        IgnoreHit();
    }
}

[shader("anyhit")]
void SphereShadowRayAnyHit(
    inout_ref(ShadowPayload) payload: SV_RayPayload)
{
    const uint primitiveID = PrimitiveIndex();
    const uint geometryID = InstanceID() + GeometryIndex();

    const float3 ray_origin = WorldRayOrigin();
    const float3 ray_direction = WorldRayDirection();

    // Sphere data
    const GeometryInfo geometryInfo = geometries[geometryID];
    const float4x4 o2w = ObjectToWorld(geometryInfo);
    const float4x4 o2wn = ObjectToWorldNormal(geometryInfo);
    const float3 sphere_center = mul(float4(0, 0, 0, 1), o2w).xyz;
    const float sphere_radius = length(mul(float4(1, 0, 0, 1), o2w).xyz - sphere_center);
    // Record the intersection
    // *: we should re-normalize it, using tHit is super unstable
    const float3 hitPoint = sphere_center + sphere_radius * normalize(ray_origin + ray_direction * RayTCurrent() - sphere_center);
    const float3 geometric_normal = normalize(hitPoint - sphere_center);
    const float3 cartesian = normalize(mul(o2wn, float4(geometric_normal, 0)).xyz);
    // We use the spherical coordinates as uv
    // We use the convention that y is up axis.
    // https://en.wikipedia.org/wiki/Spherical_coordinate_system#Cartesian_coordinates
    const float elevation = acos(clamp(cartesian.y, -1., 1.));
    const float azimuth = atan2(cartesian.z, cartesian.x);

    float2 uv = float2(-azimuth * k_inv_2_pi, elevation * k_inv_pi);
    const MaterialInfo material = materials[geometryInfo.materialID];
    const float alpha = textures[material.baseOrDiffuseTextureIndex].SampleLevel(uv, 0).w;

    const float rnd = GetNextRandom(payload.RND);
    if (rnd > alpha) {
        IgnoreHit();
    }
}

/**
 * Sample a point on a sphere light.
 * @ref: https://www.pbr-book.org/3ed-2018/Light_Transport_I_Surface_Reflection/Sampling_Light_Sources#x2-SamplingSpheres
 */
[shader("callable")]
void sampleSphere(inout_ref(ShapeSampleQuery) cSampleQuery)
{
    const uint geometryID = cSampleQuery.geometry_id;
    const GeometryInfo geometryInfo = geometries[geometryID];
    const float4x4 o2w = ObjectToWorld(geometryInfo);
    const float3 center = mul(float4(0, 0, 0, 1), o2w).xyz;
    const float r = length(mul(float4(1, 0, 0, 1), o2w).xyz - center);

    bool validRef = validRefPoint(cSampleQuery.ref_point);

    // If the reference point is inside the sphere, just sample the whole sphere uniformly
    if (!validRef || distance_squared(cSampleQuery.ref_point, center) < r * r) {
        float z = 1 - 2 * cSampleQuery.uv.x;
        float r_ = sqrt(max(0., 1 - z * z));
        float phi = 2 * k_pi * cSampleQuery.uv.y;
        float3 offset = normalize(float3(r_ * cos(phi), r_ * sin(phi), z));
        cSampleQuery.position = center + r * offset;
        cSampleQuery.normal = validRef ? -offset : offset;
        return;
    }
    // Otherwise sample a ray inside a cone towards the sphere center.
    // Build a coordinate system with n pointing towards the sphere
    float3 dir_to_center = normalize(center - cSampleQuery.ref_point);
    float3x3 frame = createONB(dir_to_center);
    float sin_theta_max_sq = r * r / distance_squared(cSampleQuery.ref_point, center);
    float cos_theta_max = sqrt(max(0.f, 1 - sin_theta_max_sq));
    float3 n_on_sphere;
    if (cos_theta_max > 0.999999) {
        n_on_sphere = -normalize(mul(frame, cosineSampleHemisphere(cSampleQuery.uv)));
    }
    else {
        // Uniformly interpolate between 1 (angle 0) and max
        float cos_theta = (1 - cSampleQuery.uv[0]) + cSampleQuery.uv[0] * cos_theta_max;
        float sin_theta = sqrt(max(0.f, 1 - cos_theta * cos_theta));
        float phi = cSampleQuery.uv[1] * 2 * k_pi;
        // Compute angle from center of sphere to sampled point on surface
        float dc = distance(cSampleQuery.ref_point, center);
        float ds = dc * cos_theta - sqrt(max(0.f, r * r - dc * dc * sin_theta * sin_theta));
        float cos_alpha = (dc * dc + r * r - ds * ds) / (2 * dc * r);
        float sin_alpha = sqrt(max(0.f, 1 - cos_alpha * cos_alpha));
        // get the normal of the sample
        n_on_sphere = -normalize(mul(float3(sin_alpha * cos(phi), sin_alpha * sin(phi), cos_alpha), frame));
    }
    // Compute surface normal and sampled point on sphere
    // Add negative sign since normals point outwards
    float3 p_on_sphere = r * n_on_sphere + center;
    cSampleQuery.position = p_on_sphere;
    cSampleQuery.normal = n_on_sphere;
    return;
}

/**
 * Get the pdf of a sampled point on a sphere light.
 * @ref: https://www.pbr-book.org/3ed-2018/Light_Transport_I_Surface_Reflection/Sampling_Light_Sources#x2-SamplingSpheres
 */
[shader("callable")]
void sampleSpherePdf(inout_ref(ShapeSamplePdfQuery) cSamplePdfQuery)
{
    const uint geometryID = cSamplePdfQuery.geometry_id;
    const GeometryInfo geometryInfo = geometries[geometryID];
    const float4x4 o2w = ObjectToWorld(geometryInfo);
    const float3 center = mul(float4(0, 0, 0, 1), o2w).xyz;
    const float r = length(mul(float4(1, 0, 0, 1), o2w).xyz - center);

    bool validRef = validRefPoint(cSamplePdfQuery.ref_point);

    if (!validRef || distance_squared(cSamplePdfQuery.ref_point, center) < r * r) {
        // If the reference point is inside the sphere, just sample the whole sphere uniformly
        cSamplePdfQuery.pdf = 1. / surface_area_sphere(r);
        return;
    }
    float sin_theta_max_sq = r * r / distance_squared(cSamplePdfQuery.ref_point, center);
    float cos_theta_max = sqrt(max(0.f, 1 - sin_theta_max_sq));
    if (cos_theta_max > 0.999999) {
        float3 mdir_to_center = normalize(cSamplePdfQuery.ref_point - center);
        cSamplePdfQuery.pdf = cosineHemispherePdf(abs(dot(mdir_to_center, cSamplePdfQuery.sample_normal)));
    }
    else {
        // Uniform sampling PDF of a cone.
        float pdf_solid_angle = 1 / (2 * k_pi * (1 - cos_theta_max));
        // Convert it back to area measure
        float3 p_on_sphere = cSamplePdfQuery.sample_position;
        float3 n_on_sphere = cSamplePdfQuery.sample_normal;
        float3 dir = normalize(p_on_sphere - cSamplePdfQuery.ref_point);
        // Output the result
        cSamplePdfQuery.pdf = pdf_solid_angle * abs(dot(n_on_sphere, dir)) /
                              distance_squared(cSamplePdfQuery.ref_point, p_on_sphere);
    }

    return;
}
#endif  // _SRENDERER_SPT_INTERFACEC_HEADER_

#endif // SRENDERER_RAYTRACER_PLUGIN_SPHERE_PRIMITIVE_