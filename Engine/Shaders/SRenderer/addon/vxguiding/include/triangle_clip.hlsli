#ifndef _SRENDERER_VXGUIDING_ADDON_TRIANGLE_CLIP_HEADER_
#define _SRENDERER_VXGUIDING_ADDON_TRIANGLE_CLIP_HEADER_

#include "../../../include/common/cpp_compatible.hlsli"
#include "../../../include/common/geometry.hlsli"

#define PLANE_THICKNESS_EPSILON 0.00001f

/**
 * Classify a given vertex against an axis-aligned plane
 * @param sign        min/max clipping plane
 * @param axis        axis of the clipping plane
 * @param c_v         one vertex of the clipping plane
 * @param p_v         vertex to classify
 * @return            classification of the vertex
 */
inline int Classify(
    int sign,
    int axis,
    float3 c_v,
    float3 p_v)
{
    const double d = sign * (p_v[axis] - c_v[axis]);
    if (d > PLANE_THICKNESS_EPSILON) return 1;
    else if (d < -PLANE_THICKNESS_EPSILON) return -1;
    else return 0;
}

// Clip the given polygon against an axis-aligned plane
//
// @param p_vs        polygon before clipping as a sequence of vertices
// @param nb_p_vs     number of vertices before clipping
// @param sign        min/max clipping plane
// @param axis        axis of the clipping plane
// @param c_v         one vertex of the clipping plane
//
// @return p_vs       polygon after clipping as a sequence of vertices
// @return nb_p_vs    number of vertices after clipping
void ClipTrianglePlane(
    inout_ref(float3) p_vs[9],
    inout_ref(int) nb_p_vs,
    int sign,
    int axis,
    float3 c_v
) {
    int nb = nb_p_vs;
    if (nb <= 1) return;

    float3 new_p_vs[9];
    int k = 0;
    bool b = true; // polygon is fully located on clipping plane

    float3 p_v1 = p_vs[nb - 1];
    int d1 = Classify(sign, axis, c_v, p_v1);
    for (int j = 0; j < nb; ++j) {
        float3 p_v2 = p_vs[j];
        int d2 = Classify(sign, axis, c_v, p_v2);
        if (d2 == 0) {
            if (d1 != 0)
                new_p_vs[k++] = p_v2;
        } else { // d2 < 0 OR d2 > 0
            b = false;
            // test for d1 == 0 first, because if that is true,
            // none of the expressions in the else statement can be true.
            if (d1 == 0) {
                // You might want to consider special casing k == 0 outside of the
                // loop so you can avoid the check for k==0 in here which is only
                // true once.
                if (k == 0 || any(new_p_vs[k - 1] != p_v1))
                    new_p_vs[k++] = p_v1;
            }
            else if ((d2 < 0 && d1 > 0) || (d2 > 0 && d1 < 0)) {
                const float alpha = (p_v2[axis] - c_v[axis]) / (p_v2[axis] - p_v1[axis]);
                new_p_vs[k++] = lerp(p_v2, p_v1, alpha);
            }

            if (d2 > 0)
                new_p_vs[k++] = p_v2;
        }

        p_v1 = p_v2;
        d1 = d2;
    }

    if (b) return;
    // store results
    nb_p_vs = k;
    for (int j = 0; j < k; ++j)
        p_vs[j] = new_p_vs[j];
}

/**
 * Clip a given triangle against an axis-aligned bounding box
 * @param p_vs        polygon before/after clipping as a sequence of vertices
 * @param nb_p_vs     number of vertices before/after clipping
 * @param clipper     axis-aligned bounding box used for clipping
 */
void ClipTriangleAgainstAABB(
    inout_ref(float3) p_vs[9],
    inout_ref(int) nb_p_vs,
    in_ref(AABB) clipper
) {
    for (int axis = 0; axis < 3; ++axis) {
        ClipTrianglePlane(p_vs, nb_p_vs, 1, axis, clipper.min);
        ClipTrianglePlane(p_vs, nb_p_vs, -1, axis, clipper.max);
    }
}

#undef PLANE_THICKNESS_EPSILON
#endif // _SRENDERER_VXGUIDING_ADDON_TRIANGLE_CLIP_HEADER_