#ifndef _SRENDERER_ADDON_DIFFERENTIABLE_COMMON_SIMPLETESTUTILS_
#define _SRENDERER_ADDON_DIFFERENTIABLE_COMMON_SIMPLETESTUTILS_

// Triangle intersection. Returns { t, u, v }
float3 triIntersect(
    in float3 ro, in float3 direction,
    in float3 v0, in float3 v1, in float3 v2
) {
    float3 v1v0 = v1 - v0;
    float3 v2v0 = v2 - v0;
    float3 rov0 = ro - v0;

    float3 n = cross(v1v0, v2v0);
    float3 q = cross(rov0, direction);
    float d = 1.0 / dot(direction, n);
    float u = d * dot(-q, v2v0);
    float v = d * dot(q, v1v0);
    float t = d * dot(-n, rov0);

    if (u < 0.0 || v < 0.0 || (u + v) > 1.0) t = -1.0;

    return float3(t, u, v);
}

struct Ray : IDifferentiable {
    __init(float3 origin, float3 direction) {
        this.origin = origin;
        this.direction = direction;
    }
    float3 origin;
    float3 direction;
};

struct RayDifferential {
    float3 dodx;
    float3 dody;
    float3 dddx;
    float3 dddy;
};

Ray generateRay(int2 pixel, int2 resolution) {
    Ray ray;
    float2 p = (-resolution.xy + 2.0 * pixel) / resolution.y;
    ray.origin = float3(0.0, 0.0, 4.0);
    ray.direction = normalize(float3(p, -2.0));
    return ray;
}

RayDifferential generateRayDifferential(int2 pixel, int2 resolution) {
    RayDifferential raydiff;
    Ray rx = generateRay(pixel + int2(1, 0), resolution);
    Ray ry = generateRay(pixel + int2(0, 1), resolution);
    raydiff.dodx = rx.origin;
    raydiff.dody = ry.origin;
    raydiff.dddx = rx.direction;
    raydiff.dddy = ry.direction;
    return raydiff;
}

struct HitAD : IDifferentiable {
    float3 position;
    no_diff float3 normal;
    no_diff float3 color;
    no_diff uint geometryID;
    no_diff bool hasHit;
    no_diff float2 uv;
    no_diff float2 bary;
};

bool HasHit(HitAD hit) {
    return hit.hasHit;
}

[Differentiable]
float3 loadPosition(inout HitAD hit, float3 translation = float3(0)) {
    if (!hit.hasHit) return float3(0);
    float3 v1 = cos(float3(2.0, 1.0, 1.0) + 1.0) * float3(1.5, 1.0, 1.0) + translation;
    float3 v2 = cos(float3(5.0, 2.0, 3.0) + 2.0) * float3(1.5, 1.0, 1.0) + translation;
    float3 v3 = cos(float3(1.0, 3.0, 5.0) + 4.0) * float3(1.5, 1.0, 1.0) + translation;
    float3 v = hit.uv.x * v2 + hit.uv.y * v3 + (1.0 - hit.uv.x - hit.uv.y) * v1;
    hit.position = v;
    return v;
}

[Differentiable]
float3 loadPosition(no_diff float2 uv, float3 translation = float3(0)) {
    float3 v1 = cos(float3(2.0, 1.0, 1.0) + 1.0) * float3(1.5, 1.0, 1.0) + translation;
    float3 v2 = cos(float3(5.0, 2.0, 3.0) + 2.0) * float3(1.5, 1.0, 1.0) + translation;
    float3 v3 = cos(float3(1.0, 3.0, 5.0) + 4.0) * float3(1.5, 1.0, 1.0) + translation;
    float3 v = uv.x * v2 + uv.y * v3 + (1.0 - uv.x - uv.y) * v1;
    return v;
}

HitAD ray_tracing(Ray ray, float3 translation = float3(0)) {
    // triangle animation
    float3 v1 = cos(float3(2.0, 1.0, 1.0) + 1.0) * float3(1.5, 1.0, 1.0) + translation;
    float3 v2 = cos(float3(5.0, 2.0, 3.0) + 2.0) * float3(1.5, 1.0, 1.0) + translation;
    float3 v3 = cos(float3(1.0, 3.0, 5.0) + 4.0) * float3(1.5, 1.0, 1.0) + translation;

    float3 col = float3(0.08) + 0.02 * ray.direction.y;

    float tmin = 1e10;

    col *= 1.0 - 0.3;

    HitAD hit;

    float3 res = triIntersect(ray.origin, ray.direction, v1, v2, v3);
    float t2 = res.x;
    if (t2 > 0.0 && t2 < tmin) {
        tmin = t2;
        float t = t2;
        float3 pos = ray.origin + t * ray.direction;
        float3 nor = normalize(cross(v2 - v1, v3 - v1));
        col = float3(0.8, 0.4, 0.6);
        col *= 0.55 + 0.45 * faceforward(-nor, -ray.direction, nor).y;

        hit.hasHit = true;
        hit.position = ray.origin + tmin * ray.direction;
        hit.uv = res.yz;
        hit.normal = normalize(cross(v2 - v1, v3 - v1));
    }
    else {
        hit.hasHit = false;
    }

    hit.color = col;
    return hit;
}

#endif // _SRENDERER_ADDON_DIFFERENTIABLE_COMMON_SIMPLETESTUTILS_