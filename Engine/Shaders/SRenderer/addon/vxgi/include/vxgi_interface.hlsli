#ifndef _SRENDERER_ADDON_VOXELIZING_INTERFACE_HEADER_
#define _SRENDERER_ADDON_VOXELIZING_INTERFACE_HEADER_

struct VoxerlizerData {
    float3 aabbMin;
    int voxelSize;
    float3 aabbMax;
    int padding;
};

int FlatIndex(int3 voxelID, int dimension) {
    return voxelID.x + voxelID.y * dimension + voxelID.z * dimension * dimension;
}

int3 ReconstructIndex(int flattenID, int dimension) {
    return int3(flattenID % dimension, (flattenID / dimension) % dimension, flattenID / (dimension * dimension));
}

#endif // !_SRENDERER_ADDON_VOXELIZING_INTERFACE_HEADER_