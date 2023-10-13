#ifndef _SRENDERER_ADDON_VXGUIDING_TESTDI_INTERFACE_HEADER_
#define _SRENDERER_ADDON_VXGUIDING_TESTDI_INTERFACE_HEADER_

#include "../../raytracer/primitives/quad.hlsli"

struct TestDISetting {
    float4x4 worldMat;
    float4x4 localMat;
    float2 pMin;
    float2 pMax;
    float intensity;
};

bool RayQuadIntersection(in_ref(Ray) ray, in_ref(TestDISetting) setting, out float t) {
    t = 0.f;
    Ray objRay = ray;
    objRay.direction = mul(float4(ray.direction, 0), setting.localMat).xyz;
    objRay.origin = mul(float4(ray.origin, 1), setting.localMat).xyz;
    if (objRay.origin.z < 0) return false;
    return RayAAPlaneIntersection_Visibility(objRay, setting.pMin, setting.pMax, t);
}

#endif // _SRENDERER_ADDON_VXGUIDING_TESTDI_INTERFACE_HEADER_