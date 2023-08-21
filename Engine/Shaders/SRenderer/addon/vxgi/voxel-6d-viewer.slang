#include "../../include/common/camera.hlsli"
#include "../../include/common/geometry.hlsli"
#include "../../include/common/random.hlsli"
#include "../../include/common/raycast.hlsli"
#include "include/conetrace_utils.hlsli"
#include "include/vxgi_interface.hlsli"

[[vk::push_constant]]
cbuffer PushConstantBuffer {
    int2 resolution;
    float edgeThreshold;
    int mipLevelShown;
    int2 padding;
    int showWhat;
    float scaling;
    int useSampler;
};

cbuffer CameraBuffer {
    CameraData gCamera;
};

cbuffer VoxelizerUniform {
    VoxerlizerData gVD;
}

#ifdef USE_ISOTROPIC_VOXEL
Texture3D<float2> uRadOpaVox6D;
#else
Texture3D<float2> uRadOpaVox6D[6];
#endif

SamplerState uTex3DSampler;

float2 getVoxelOpacity(int3 c, int faceID) {
    if (any(c < 0 || c >= (gVD.voxelSize >> mipLevelShown))) return -1.f;
#ifdef USE_ISOTROPIC_VOXEL
    else return uRadOpaVox6D.Load(int4(c, mipLevelShown));
#else
    else return uRadOpaVox6D[faceID].Load(int4(c, mipLevelShown));
#endif
}

float middle(float3 vector) {
    float smallestElement = min(vector.x, min(vector.y, vector.z));
    float largestElement = max(vector.x, max(vector.y, vector.z));
    float middleElement = vector.x + vector.y + vector.z - smallestElement - largestElement;
    return middleElement;
}

bool IsEdge(float3 pos, float threshold) {
    float3 offset = abs(fmod(pos, 1));
    offset = (offset < float3(0.5)) ? offset : 1 - offset;
    if (middle(offset) < threshold) {
        return true;
    }
    return false;
}

int GetFaceId(int3 step) {
    if (step.x == -1) return 0;
    if (step.x == +1) return 1;
    if (step.y == -1) return 2;
    if (step.y == +1) return 3;
    if (step.z == -1) return 4;
    if (step.z == +1) return 5;
    return 0;
}

[shader("fragment")]
float4 fragmentMain(in float2 uv: Coord) : SV_Target {
    const int2 pixel = clamp(int2(uv * resolution), 0, resolution - 1);
    // RandomSamplerState RNG = InitRandomSampler(pixel, pConst.sample_batch);

    Ray primaryRay = generateRay(
        pixel,
        resolution,
        false,
        gCamera
    );
    
    const int3 voxelSize = gVD.voxelSize >> mipLevelShown;
    const float3 extent = gVD.aabbMax - gVD.aabbMin;
    const float extentMax = max(extent.x, max(extent.y, extent.z)) * 0.5 + 0.01;
    const float3 center = (gVD.aabbMax + gVD.aabbMin) * 0.5;
    float3 rayPos = ((primaryRay.origin.xyz - center) / extentMax + 1) * 0.5 * voxelSize; // [-1, 1]^3
    primaryRay.origin = rayPos;
    
    // if the ray is outside the voxel grid, we need to clip it to the voxel grid
    if (any(rayPos < 0 || rayPos >= voxelSize)) {
        AABB aabb = { float3(0), float3(voxelSize) };
        const float tMin = intersectTMin(primaryRay, aabb);
        if (tMin < 0) {
            return float4(0, 0, 0, 1);
        }
        else {
            if (IsEdge((rayPos + primaryRay.direction * tMin) / voxelSize, 0.005)) {
                return float4(1, 0, 0, 1);
            }
            rayPos = rayPos + primaryRay.direction * (tMin - 0.002);
        }
    }
    
    int3 mapPos = int3(floor(rayPos + 0.));
    const int3 startPos = mapPos;

    const float3 deltaDist = abs(float3(length(primaryRay.direction)) / primaryRay.direction);
    const int3 rayStep = int3(sign(primaryRay.direction));
    float3 sideDist = (sign(primaryRay.direction) * (float3(mapPos) - rayPos) + (sign(primaryRay.direction) * 0.5) + 0.5) * deltaDist;

    bool3 mask = false;
    bool hit = false;
    bool edge = false;
    const int MAX_RAY_STEPS = 1000;

    float show_value = 0.f;

    for (int i = 0; i < MAX_RAY_STEPS; i++) {
        float2 value;
        value = getVoxelOpacity(mapPos, GetFaceId(int3(float3(mask)) * rayStep));
        // Skip the voxel that the camera is in, so that we could observe occluded voxels if necessary
        if (value.y > 0.f && !all(mapPos == startPos) && (showWhat != 3 || value.x > 0.f)) {
            hit = true;
            if (showWhat == 0 || showWhat == 3)
                show_value = value.x;
            else if (showWhat == 1 || showWhat == 2)
                show_value = value.y;
            // check whether is edge
            const AABB aabb = { float3(mapPos), float3(mapPos + 1) };
            const float tMin = intersectTMin(primaryRay, aabb);
            if (tMin > 0) {
                const float3 hitPoint = primaryRay.origin + primaryRay.direction * tMin;
                float3 offset = abs(fmod(hitPoint, 1));
                offset = (offset < float3(0.5)) ? offset : 1 - offset;
                if (middle(offset) < edgeThreshold) {
                    edge = true;
                }
            }
            break;
        }
        else if (i > 3 && value.y < 0.f) {
            break;
        }

        mask = sideDist.xyz <= min(sideDist.yzx, sideDist.zxy);
        // All components of mask are false except for the corresponding largest component
        // of sideDist, which is the axis along which the ray should be incremented.
        sideDist += float3(mask) * deltaDist;
        mapPos += int3(float3(mask)) * rayStep;
    }

    float3 color = float3(0.0);
    show_value = show_value * scaling;
    float3 base_color = float3(show_value);
    if (showWhat == 2) {
        base_color = show_value == 1 ? float3(1.0, 1.0, 0) : float3(show_value);
    }

    if (hit) {
        if (mask.x) {
            color = float3(0.9) * base_color;
        }
        if (mask.y) {
            color = float3(1.0) * base_color;
        }
        if (mask.z) {
            color = float3(0.8) * base_color;
        }

        if (edge) {
            color = 0.5;
        }
    }
    
    // fragColor.rgb = color;
    // // fragColor.rgb = vec3(0.1 * noiseDeriv);

    return float4(color, 1);
}