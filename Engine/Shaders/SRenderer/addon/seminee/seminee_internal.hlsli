#ifndef _SRENDERER_ADDON_SEMINEE_INTERNAL_HEADER_
#define _SRENDERER_ADDON_SEMINEE_INTERNAL_HEADER_

#include "../../raytracer/primitives/quad.hlsli"

struct QuadObject {
    float2 pMin;
    float2 pMax;
    float3 translation;
    float intensity;
    float3x3 rotation;

    inline bool isValid() { return (intensity > 0) && all(pMax - pMin > 0); }
    inline float3 toLocal(float3 pos) {
        return mul(rotation, pos - translation);
    }
    inline float3 toLocalDir(float3 dir) {
        return mul(rotation, dir);
    }
    inline float3 toWorldDir(float3 dir) {
        return mul(dir, rotation);
    }

    inline float3 toWorld(float3 pos) {
        return mul(pos, rotation) + translation;
    }
};

bool RayQuadObjectIntersection(in_ref(Ray) ray, in_ref(QuadObject) quad, out float t) {
    Ray objRay = ray;
    objRay.direction = mul(quad.rotation, ray.direction);
    objRay.origin = ray.origin - quad.translation;
    objRay.origin = mul(quad.rotation, objRay.origin);
    return RayAAPlaneIntersection_Visibility(objRay, quad.pMin, quad.pMax, t);
}

SphQuad CreateSphQuad(in_ref(QuadObject) quad, in_ref(float3) position) {
    float3 s = float3(quad.pMin, 0.0);
    float3 ex = float3(quad.pMax.x - quad.pMin.x, 0., 0.);
    float3 ey = float3(0., quad.pMax.y - quad.pMin.y, 0.);
    SphQuad squad;
    SphQuadInit(s, ex, ey, position, squad);
    return squad;
}

void update_x_bounds(in_ref(float3) pos, in_ref(float3) n, inout_ref(QuadObject) quad) {
    float2 p1 = quad.pMin;
    float2 p2 = float2(quad.pMin.x, quad.pMax.y);
    if (n.x != 0.0) {
        const float tmin = (pos.z * n.z - (p1.y - pos.y) * n.y) / n.x - p1.x + pos.x;
        const float tmax = (pos.z * n.z - (p2.y - pos.y) * n.y) / n.x - p2.x + pos.x;
        const float minx = clamp(quad.pMin.x + tmin, quad.pMin.x, quad.pMax.x);
        const float maxx = clamp(quad.pMin.x + tmax, quad.pMin.x, quad.pMax.x);
        if (n.x > 0.0) {
            quad.pMin.x = min(minx, maxx);
        } else {
            quad.pMax.x = max(minx, maxx);
        }
    }
}

void update_y_bounds(in_ref(float3) pos, in_ref(float3) n, inout_ref(QuadObject) quad) {
    float2 p1 = quad.pMin;
    float2 p2 = float2(quad.pMax.x, quad.pMin.y);
    // pos = pos.xzy;
    if (n.y != 0.0) {
        float tmin = (pos.z * n.z - (p1.x - pos.x) * n.x) / n.y - p1.y + pos.y;
        float tmax = (pos.z * n.z - (p2.x - pos.x) * n.x) / n.y - p2.y + pos.y;
        float miny = clamp(quad.pMin.y + tmin, quad.pMin.y, quad.pMax.y);
        float maxy = clamp(quad.pMin.y + tmax, quad.pMin.y, quad.pMax.y);
        if (n.y > 0.0) {
            quad.pMin.y = min(miny, maxy);
        } else {
            quad.pMax.y = max(miny, maxy);
        }
    }
}

int HitQuads4(in_ref(QuadObject) quads[4],
              in_ref(Ray) ray, out_ref(float) tmin) {
    tmin = -1;
    int hitID = -1;
    float t;
    if (quads[0].isValid() && RayQuadObjectIntersection(ray, quads[0], t)) {
        if (t > 0 && t < k_inf) {
            hitID = 0;
            tmin = t;
        }
    }
    if (quads[1].isValid() && RayQuadObjectIntersection(ray, quads[1], t)) {
        if (t > 0 && t < k_inf && (hitID == -1 || t < tmin)) {
            if (hitID == -1 || t < tmin) {
                hitID = 1;
                tmin = t;
            }
        }
    }
    if (quads[2].isValid() && RayQuadObjectIntersection(ray, quads[2], t)) {
        if (t > 0 && t < k_inf && (hitID == -1 || t < tmin)) {
            if (hitID == -1 || t < tmin) {
                hitID = 2;
                tmin = t;
            }
        }
    }
    if (quads[3].isValid() && RayQuadObjectIntersection(ray, quads[3], t)) {
        if (t > 0 && t < k_inf && (hitID == -1 || t < tmin)) {
            if (hitID == -1 || t < tmin) {
                hitID = 3;
                tmin = t;
            }
        }
    }
    return hitID;
}

#endif // _SRENDERER_ADDON_SEMINEE_INTERNAL_HEADER_