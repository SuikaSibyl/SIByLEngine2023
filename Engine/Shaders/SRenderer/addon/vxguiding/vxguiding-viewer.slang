#include "../../include/common/camera.hlsli"
#include "../../include/common/geometry.hlsli"
#include "../../include/common/random.hlsli"
#include "../../include/common/raycast.hlsli"
#include "../../include/common/packing.hlsli"
#include "../../include/common/colormaps.hlsli"
#include "../vxgi/include/vxgi_interface.hlsli"
#include "clustering/voxel_cluster.hlsli"

[[vk::push_constant]]
cbuffer PushConstantBuffer {
    int2 resolution;
    float edgeThreshold;
    int showWhat;
    int2 map_size;
    int2 debugPixel;
    float scalar;
};

cbuffer CameraBuffer {
    CameraData gCamera;
};

cbuffer VoxelizerUniform {
    VoxerlizerData gVD;
}

const Texture3D<uint> uIrradianceMip;
const Texture3D<int> uInverseIndex;
const StructuredBuffer<int> u_associate;
StructuredBuffer<float4> u_vxNormal;
StructuredBuffer<float> u_premulIrr;

const Texture2D<uint> uSPixelVisibility;
const Texture2D<int> uSPixelIndex;

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

/** Load the 27 neighbors of the voxel at the given position. */
void LoadNeighbors(inout_ref(uint) neighbors[3][3][3], int3 pos) {
    int3 offset = int3(0);
    for (offset.z = -1; offset.z <= 1; offset.z++) {
        for (offset.y = -1; offset.y <= 1; offset.y++) {
            for (offset.x = -1; offset.x <= 1; offset.x++) {
                int3 neighborPos = pos + offset;
                if (any(neighborPos < 0 || neighborPos >= 64)) {
                    neighbors[offset.x + 1][offset.y + 1][offset.z + 1] = 0;
                }
                else {
                    neighbors[offset.x + 1][offset.y + 1][offset.z + 1] = uIrradianceMip[neighborPos];
                }
            }
        }
    }
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
    
    const int3 voxelSize = gVD.voxelSize >> 0;
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
    
    const float3 deltaDist = abs(float3(length(primaryRay.direction)) / primaryRay.direction);
    const int3 rayStep = int3(sign(primaryRay.direction));
    float3 sideDist = (sign(primaryRay.direction) * (float3(mapPos) - rayPos) + (sign(primaryRay.direction) * 0.5) + 0.5) * deltaDist;

    bool3 mask = false;
    bool hit = false;
    bool edge = false;
    const int MAX_RAY_STEPS = 1000;

    for (int i = 0; i < MAX_RAY_STEPS; i++) {
        const float opacity = uIrradianceMip[mapPos] == 0 ? 0.f : 1.f;
        if (opacity > 0.f) {
            hit = true;
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
        else if (i > 3 && opacity < 0.f) {
            break;
        }

        mask = sideDist.xyz <= min(sideDist.yzx, sideDist.zxy);
        // All components of mask are false except for the corresponding largest component
        // of sideDist, which is the axis along which the ray should be incremented.
        sideDist += float3(mask) * deltaDist;
        mapPos += int3(float3(mask)) * rayStep;
    }

    float3 color = float3(0);
    if (hit) {
        int index = uInverseIndex[mapPos];
        if (showWhat == 0) {
            color = u_vxNormal[index].xyz;
        }
        else if (showWhat == 3) {
            color = float3(u_premulIrr[index] * scalar);
        }
        else if (showWhat >= 1) {
            if (index >= 0) {
                const int id = u_associate[index];
                if (id == -1) {
                    color = float3(1, 0, 1);
                }
                else {
                    uint colorPacked = Hash_murmur(id);
                    color = UNPACK_FLOAT3_UINT(colorPacked);

                    if (showWhat >= 2) {
                        const int spixelID = uSPixelIndex[debugPixel];
                        if (spixelID != -1) {
                            const int2 spixelID2D = int2(spixelID % map_size.x, spixelID / map_size.x);
                            const uint visibility = uSPixelVisibility[spixelID2D];
                            if ((visibility & uint(1 << id)) == 0) {
                                color = float3(0.1);
                            }
                        }
                    }
                }
            }
        }

        if (mask.x) {
            color *= float3(0.5);
        }
        if (mask.y) {
            color *= float3(1.0);
        }
        if (mask.z) {
            color *= float3(0.75);
        }
        if (edge) {
            color *= 0.5;
        }
    }
    // fragColor.rgb = color;
    // // fragColor.rgb = vec3(0.1 * noiseDeriv);

    return float4(color, 1);
}