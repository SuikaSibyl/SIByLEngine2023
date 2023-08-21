#ifndef _SRENDERER_COMMON_PCA_HEADER_
#define _SRENDERER_COMMON_PCA_HEADER_

#include "cpp_compatible.hlsli"

void EigenDecomposition(
    in_ref(float3x3) matrix,
    out_ref(float3) eigenvalues,
    out_ref(float3x3) eigenvectors)
{
    // Calculate coefficients for the characteristic polynomial
    float a = -1.0f;
    float b = matrix._11 + matrix._22 + matrix._33;
    float c = matrix._12 * matrix._21 + matrix._13 * matrix._31 + matrix._23 * matrix._32;
    float d = matrix._11 * matrix._22 + matrix._11 * matrix._33 + matrix._22 * matrix._33 - matrix._12 * matrix._21 - matrix._13 * matrix._31 - matrix._23 * matrix._32;
    // Solve cubic equation for eigenvalues
    float3 pqr = float3(c / a, d / a, 0.0f);
    float3 roots = solveCubic(pqr);
    // Store eigenvalues
    eigenvalues = roots;
    // Compute eigenvectors using eigenvalues and matrix
    eigenvectors._11 = matrix._11 - roots.x;
    eigenvectors._21 = matrix._21;
    eigenvectors._31 = matrix._31;
    eigenvectors._12 = matrix._12;
    eigenvectors._22 = matrix._22 - roots.x;
    eigenvectors._32 = matrix._32;
    eigenvectors._13 = matrix._13;
    eigenvectors._23 = matrix._23;
    eigenvectors._33 = matrix._33 - roots.x;
    float3 nullSpaceVector = normalize(cross(eigenvectors._12, eigenvectors._13));
    eigenvectors._12 = cross(eigenvectors._13, nullSpaceVector);
    eigenvectors._13 = cross(nullSpaceVector, eigenvectors._12);
    eigenvectors._12 = normalize(eigenvectors._12);
    eigenvectors._13 = normalize(eigenvectors._13);
}

#endif // _SRENDERER_COMMON_PCA_HEADER_