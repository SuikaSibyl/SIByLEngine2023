#ifndef _SRENDERER_ADDON_VXGTUIDING_CLUSTERING_HEADER_
#define _SRENDERER_ADDON_VXGTUIDING_CLUSTERING_HEADER_

#include "../../../include/common/cpp_compatible.hlsli"

struct svoxel_info {
    float3 color_info; // avg color of the superpixel
    int no_voxels;     // number of voxels
    float3 center;     // center (avg pixel pos) of the supervoxel
    int id;            // superpixel id
};

float3 GetNormal(in_ref(uint) neighbors[3][3][3]) {
    int count_x = 0;
    int count_y = 0;
    int count_z = 0;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            count_x += neighbors[0][i][j] > 0 ? 1 : 0;
            count_x += neighbors[2][i][j] > 0 ? 1 : 0;
            count_y += neighbors[i][0][j] > 0 ? 1 : 0;
            count_y += neighbors[i][2][j] > 0 ? 1 : 0;
            count_z += neighbors[i][j][0] > 0 ? 1 : 0;
            count_z += neighbors[i][j][2] > 0 ? 1 : 0;
        }
    }
    float x = (18 - count_x) / 18.f;
    float y = (18 - count_y) / 18.f;
    float z = (18 - count_z) / 18.f;
    if (count_x == 0 && count_y != 0 && count_z != 0) return float3(1, 0, 0);
    else if (count_x != 0 && count_y == 0 && count_z != 0) return float3(0, 1, 0);
    else if (count_x != 0 && count_y != 0 && count_z == 0) return float3(0, 0, 1);
    return float3(x, y, z);
}

#endif // !_SRENDERER_ADDON_VXGTUIDING_CLUSTERING_HEADER_