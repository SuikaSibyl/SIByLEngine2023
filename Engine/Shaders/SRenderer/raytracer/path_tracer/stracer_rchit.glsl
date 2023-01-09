#include "../include/common_hit.h"
#include "../include/common_sample_shape.h"
#include "../../../Utility/random.h"

layout(location = 0) rayPayloadInEXT PrimaryPayload primaryPld;;
layout(location = 1) rayPayloadEXT   bool  hitOccluder;

layout(location = 0) callableDataEXT SampleQuery cSampleQuery;

void main()
{
    setIntersected(primaryPld.flags, true);

    HitGeometry geoInfo = getHitGeometry();
    MaterialData material = materials[geoInfo.matID];
    vec3 base_color = texture(textures[material.basecolor_opacity_tex], geoInfo.uv).rgb;
    vec3 normal = texture(textures[material.normal_bump_tex], geoInfo.uv).rgb;
    normal = vec3(0.5,0.5,1);
    normal = normalize(normal * 2.0 - 1);
    normal = geoInfo.TBN * normal;

    cSampleQuery.ref_point = geoInfo.worldPosition;
    cSampleQuery.geometry_id = 2;
    cSampleQuery.uv = vec2(stepAndOutputRNGFloat(primaryPld.rngState), stepAndOutputRNGFloat(primaryPld.rngState));
    executeCallableEXT(0, 0);
    const vec3 lightPos = cSampleQuery.position;
    const vec3 lightNormal = cSampleQuery.normal;

    float weight = 0.f;
    float color = 0.f;
    const vec3 shadowRayOrigin = offsetPositionAlongNormal(geoInfo.worldPosition, normal);
    const vec3 lightSample = lightPos;
    const vec3 lightDir = normalize(lightSample - shadowRayOrigin);
    const float NdL = dot(normal, lightDir);

    if(NdL > 0) {
        weight += 1;
        const float lightDist = length(lightSample - shadowRayOrigin) - 0.001;
        hitOccluder = false;            
        traceRayEXT(tlas,           // Top-level acceleration structure
            gl_RayFlagsOpaqueEXT,   // Ray flags, here saying "treat all geometry as opaque"
            0xFF,                   // 8-bit instance mask, here saying "trace against all instances"
            PRIMITIVE_TYPE_COUNT,   // SBT record offset
            0,                      // SBT record stride for offset
            1,                      // Miss index
            shadowRayOrigin,        // Ray origin
            0.0,                    // Minimum t-value
            lightDir,               // Ray direction
            lightDist,              // Maximum t-value
            1);                     // Location of payload
        if(!hitOccluder) {
            color += NdL;
        }
    }

    primaryPld.baseColor = base_color * color;
}