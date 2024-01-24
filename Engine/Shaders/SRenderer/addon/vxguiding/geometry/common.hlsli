#ifndef _SRENDERER_ADDON_VXPG_GEOMETRY_HEADER_
#define _SRENDERER_ADDON_VXPG_GEOMETRY_HEADER_

#include "../../../include/common/hashing.hlsli"
#include "../../vxgi/include/conetrace_utils.hlsli"

cbuffer GeometryUniform {
    struct GeometryConfig {
        float3 aabbMin;  // min of world AABB
        uint32_t size;   // length of one dim over world AABB
        float3 aabbMax;  // max of world AABB
        int HashmapSize; // hash map size
        uint32_t mode;
        uint32_t bucket_mode;
        uint32_t bucket_size;
        bool use_compact;
        bool z_conservative;
        bool clipping;
    } u_config;
}

VoxelTexInfo CreateVoxelTexInfo(in_ref(GeometryConfig) data) {
    const float3 extent = data.aabbMax - data.aabbMin;
    const float extentMax = max(extent.x, max(extent.y, extent.z)) * 0.5 + 0.01;
    const float3 center = (data.aabbMax + data.aabbMin) * 0.5;
    VoxelTexInfo info;
    info.minPoint = center - float3(extentMax);
    info.voxelScale = 1.0f / (extentMax * 2.0f);
    info.volumeDimension = u_config.size;
    return info;
}

VoxelTexInfo GetVoxelTexInfo() {
    return CreateVoxelTexInfo(u_config);
}

struct VoxelDataConf {
    uint32_t num_max_bucket;
};

uint32_t QueryBucketID(int3 voxelID, in_ref(GeometryConfig) data) {
    if (data.mode == 0) {
        return murmur3(uint3(voxelID)) % data.HashmapSize;
    } else {
        const uint32_t dimension = data.size;
        return voxelID.x + voxelID.y * dimension 
            + voxelID.z * dimension * dimension;
    }
}

uint32_t QueryKey(int3 voxelID, int qn = 0) {
    return ((uint32_t(qn) & 0x7) << 24)
         | ((uint32_t(voxelID.x) & 0xff) << 16)
         | ((uint32_t(voxelID.y) & 0xff) << 8)
         | ((uint32_t(voxelID.z) & 0xff) << 0);
}

int QueryCellID(uint32_t bucketID, uint32_t key, in_ref(GeometryConfig) data,
                RWStructuredBuffer<uint> hashRecord) {
    if (data.mode == 0) {
        const uint32_t offset = bucketID * data.bucket_size;
        for (int i = 0; i < data.bucket_size; ++i) {
            uint32_t originfp = 0;
            InterlockedCompareExchange(hashRecord[offset + i], 0xffffffffu, key, originfp);
            if (originfp == 0xffffffffu || originfp == key) {
                return i;
            }
        }
        return -1;
    } else {
        return 0;
    }
}

int FindCellID(uint32_t bucketID, uint32_t key, in_ref(GeometryConfig) data,
                RWStructuredBuffer<uint> hashRecord) {
    if (data.mode == 0) {
        const uint32_t offset = bucketID * data.bucket_size;
        for (int i = 0; i < data.bucket_size; ++i) {
            uint32_t originfp = hashRecord[offset + i];
            if (originfp == key) {
                return i;
            }
        }
        return -1;
    } else {
        return 0;
    }
}

int FindCellID(uint32_t bucketID, uint32_t key, in_ref(GeometryConfig) data,
               StructuredBuffer<uint> hashRecord) {
    if (data.mode == 0) {
        const uint32_t offset = bucketID * data.bucket_size;
        for (int i = 0; i < data.bucket_size; ++i) {
            uint32_t originfp = hashRecord[offset + i];
            if (originfp == key) {
                return i;
            }
        }
        return -1;
    } else {
        return 0;
    }
}

int GetGlobalID(uint32_t bucketID, uint32_t cellID, in_ref(GeometryConfig) data) {
    if (data.mode == 0) {
        return bucketID * u_config.bucket_size + cellID;
    } else {
        return bucketID;
    }
}

int3 ReconstructIndex(uint32_t flattenID, in_ref(GeometryConfig) data,
                      StructuredBuffer<uint> hashRecord) {
    if (data.mode == 0) {
        const uint32_t fingerprint = hashRecord[flattenID];
        if (fingerprint == 0xffffffffu) return int3(-1);
        return int3((uint32_t(fingerprint >> 16) & 0xff))
         | ((uint32_t(fingerprint >> 8) & 0xff))
         | ((uint32_t(fingerprint >> 0) & 0xff));
    } else {
        const uint32_t dimension = data.size;
        return int3(flattenID % dimension, (flattenID / dimension) % dimension, 
                flattenID / (dimension * dimension));
    }
}

#endif // _SRENDERER_ADDON_VXPG_GEOMETRY_HEADER_