#ifndef _SRENDERER_VXGI_VOXELIZER_UTILS_HEADER_
#define _SRENDERER_VXGI_VOXELIZER_UTILS_HEADER_

#include "../../include/common/cpp_compatible.hlsli"
#include "../../include/common/geometry.hlsli"

/***********************************************************************
 * Voxelizer Utils
 ***********************************************************************
 * This file contains some utility functions for voxelization.
 * The voxelization algorithm is based on the tutorial
 * "The Basics of GPU Voxelization" by Nvidia.
 * It dependes on conservative rasterization.
 * My implementation take advantages of hardware conservative rasterization.
 * You could also implement and use software conservative rasterization.
 * @ref: https://developer.nvidia.com/content/basics-gpu-voxelization
***********************************************************************/

/***********************************************************************
 * In this part, we define some utility functions for depth conservatie
 * rasterization. By using hardware conservative rasterization, we can
 * get axises along the image plane become conservative.
 * However, in the depth / dominant axis direction, one pixel invocation
 * may need to fill in multiple voxels.
 * According to "The Basics of GPU Voxelization", it could at most
 * be 3 voxels (shading point, back voxel, rear voxel).
 *
 * To check whether to fill extra voxels, we first do a AABB test.
 * a.k.a. "AABBConditionTest";
 * If the triangle's AABB passes, we do a more accurate test.
 * a.k.a. "EdgeVoxelConditionTest"; (separate axis theorems)
 * If the triangle passes both tests, we fill in the voxels.
************************************************************************/

/**
* Computes the axis-aligned bounding box of a triangle.
* @param v1 The first vertex of the triangle.
* @param v2 The second vertex of the triangle.
* @param v3 The third vertex of the triangle.
* @return The axis-aligned bounding box of the triangle.
*/
AABB GetTriangleAABB(
    in_ref(float3) v1, in_ref(float3) v2, in_ref(float3) v3
) {
    AABB aabb;
    aabb.min = min(min(v1, v2), v3);
    aabb.max = max(max(v1, v2), v3);
    return aabb;
}

/**
* Compute whether a triangle AABB intersects a voxel.
* This is a fast test, but it is not accurate. Only do this before the edge test.
* @param voxelID The voxel's ID.
* @param primMin The minimum corner of the triangle's AABB.
* @param primMax The maximum corner of the triangle's AABB.
*/
bool AABBConditionTest(
    in_ref(int3) voxelID,
    in_ref(AABB) primAABB
) {
    const float3 voxelMin = float3(voxelID);
    const float3 voxelMax = float3(voxelID + 1);
    if ((primAABB.max.x - voxelMin.x) * (primAABB.min.x - voxelMax.x) >= 0.0 ||
        (primAABB.max.y - voxelMin.y) * (primAABB.min.y - voxelMax.y) >= 0.0 ||
        (primAABB.max.z - voxelMin.z) * (primAABB.min.z - voxelMax.z) >= 0.0)
        return false;
    return true;
}

/**
 * Compute whether a triangle intersects a voxel.
 * This is a accurate test, but it is slow. Only do this after the AABB test.
 * @param voxPos The positions of the voxel's corners.
 * @param voxelCenter The center of the voxel.
*/
bool EdgeVoxelConditionTest(in_ref(float3) voxPos[3], in_ref(float3) voxelCenter) {
    // calculate edge vectors in voxel coordinate space
    float3 e0 = voxPos[1] - voxPos[0];
    float3 e1 = voxPos[2] - voxPos[1];
    float3 e2 = voxPos[0] - voxPos[2];
    float3 planeNormal = cross(e0, e1);

    {   // for testing in XY plane projection
        float isFront = -sign(planeNormal.z);
        // compute the 2d space normal of each edges.
        float2 eNrm[3];
        eNrm[0] = float2(e0.y, -e0.x) * isFront;
        eNrm[1] = float2(e1.y, -e1.x) * isFront;
        eNrm[2] = float2(e2.y, -e2.x) * isFront;
        // absolute value for each normal.
        float2 an[3];
        an[0] = abs(eNrm[0]);
        an[1] = abs(eNrm[1]);
        an[2] = abs(eNrm[2]);
        // calculate signed distance offset from a voxel center
        // to the voxel vertex which has maximum signed distance value.
        float3 eOfs;
        eOfs.x = (an[0].x + an[0].y) * 0.5;
        eOfs.y = (an[1].x + an[1].y) * 0.5;
        eOfs.z = (an[2].x + an[2].y) * 0.5;
        // calculate signed distance of each edges.
        float3 ef;
        ef.x = eOfs.x - dot(voxPos[0].xy - voxelCenter.xy, eNrm[0]);
        ef.y = eOfs.y - dot(voxPos[1].xy - voxelCenter.xy, eNrm[1]);
        ef.z = eOfs.z - dot(voxPos[2].xy - voxelCenter.xy, eNrm[2]);
        // test is passed if all of signed distances are positive.
        if (ef.x < 0 || ef.y < 0 || ef.z < 0)
            return false;
    }
    { // for testing in YZ plane projection
        float isFront = -sign(planeNormal.x);
        // compute the 2d space normal of each edges.
        float2 eNrm[3];
        eNrm[0] = float2(e0.z, -e0.y) * isFront;
        eNrm[1] = float2(e1.z, -e1.y) * isFront;
        eNrm[2] = float2(e2.z, -e2.y) * isFront;
        // absolute value for each normal.
        float2 an[3];
        an[0] = abs(eNrm[0]);
        an[1] = abs(eNrm[1]);
        an[2] = abs(eNrm[2]);
        // calculate signed distance offset from a voxel center
        // to the voxel vertex which has maximum signed distance value.
        float3 eOfs;
        eOfs.x = (an[0].x + an[0].y) * 0.5;
        eOfs.y = (an[1].x + an[1].y) * 0.5;
        eOfs.z = (an[2].x + an[2].y) * 0.5;
        // calculate signed distance of each edges.
        float3 ef;
        ef.x = eOfs.x - dot(voxPos[0].yz - voxelCenter.yz, eNrm[0]);
        ef.y = eOfs.y - dot(voxPos[1].yz - voxelCenter.yz, eNrm[1]);
        ef.z = eOfs.z - dot(voxPos[2].yz - voxelCenter.yz, eNrm[2]);
        // test is passed if all of signed distances are positive.
        if (ef.x < 0 || ef.y < 0 || ef.z < 0)
            return false;
    }
    { // for testing in ZX plane projection
        float isFront = -sign(planeNormal.y);
        // compute the 2d space normal of each edges.
        float2 eNrm[3];
        eNrm[0] = float2(e0.x, -e0.z) * isFront;
        eNrm[1] = float2(e1.x, -e1.z) * isFront;
        eNrm[2] = float2(e2.x, -e2.z) * isFront;
        // absolute value for each normal.
        float2 an[3];
        an[0] = abs(eNrm[0]);
        an[1] = abs(eNrm[1]);
        an[2] = abs(eNrm[2]);
        // calculate signed distance offset from a voxel center
        // to the voxel vertex which has maximum signed distance value.
        float3 eOfs;
        eOfs.x = (an[0].x + an[0].y) * 0.5;
        eOfs.y = (an[1].x + an[1].y) * 0.5;
        eOfs.z = (an[2].x + an[2].y) * 0.5;
        // calculate signed distance of each edges.
        float3 ef;
        ef.x = eOfs.x - dot(voxPos[0].zx - voxelCenter.zx, eNrm[0]);
        ef.y = eOfs.y - dot(voxPos[1].zx - voxelCenter.zx, eNrm[1]);
        ef.z = eOfs.z - dot(voxPos[2].zx - voxelCenter.zx, eNrm[2]);
        // test is passed if all of signed distances are positive.
        if (ef.x < 0 || ef.y < 0 || ef.z < 0)
            return false;
    }
    return true;
}

/**
 * Get the dominant axis of a triangle, that should be projected to.
 * @param pos0 The first vertex of the triangle.
 * @param pos1 The second vertex of the triangle.
 * @param pos2 The third vertex of the triangle.
 * @return The dominant axis of the triangle.
 */
int GetDominantAxis(in_ref(float3) pos0, in_ref(float3) pos1, in_ref(float3) pos2) {
    const float3 normal = abs(cross(pos1 - pos0, pos2 - pos0));
    return (normal.x > normal.y && normal.x > normal.z) ? 0 : 
			(normal.y > normal.z) ? 1 : 2;
}

/**
 * Project a vertex to a 2D plane.
 * @param vertex The vertex to project.
 * @param axis The axis to project along.
 * @return The projected vertex.
 */
float2 ProjectAlongAxis(in_ref(float3) vertex, uint axis) {
    return axis == 0 ? vertex.yz : (axis == 1 ? vertex.xz : vertex.xy);
}

#endif // _SRENDERER_VXGI_VOXELIZER_UTILS_HEADER_