#ifndef _SRENDERER_LINEAR_ALGEBRA_HEADER_
#define _SRENDERER_LINEAR_ALGEBRA_HEADER_

#include "cpp_compatible.hlsli"
#include "math.hlsli"

// Function to calculate the inverse of a 2x2 matrix
float2x2 Inverse2x2(in_ref(float2x2) matrix) {
    const float determinant = matrix._11 * matrix._22 - matrix._12 * matrix._21;
    const float reciprocalDeterminant = 1.0 / determinant;
    float2x2 inverseMatrix;
    inverseMatrix._11 = matrix._22 * reciprocalDeterminant;
    inverseMatrix._12 = -matrix._12 * reciprocalDeterminant;
    inverseMatrix._21 = -matrix._21 * reciprocalDeterminant;
    inverseMatrix._22 = matrix._11 * reciprocalDeterminant;
    return inverseMatrix;
}

// Function to calculate the inverse of a 2x2 matrix
float2x2 Inverse2x2(in_ref(float2x2) matrix, float determinant) {
    const float reciprocalDeterminant = 1.0 / determinant;
    float2x2 inverseMatrix;
    inverseMatrix._11 = matrix._22 * reciprocalDeterminant;
    inverseMatrix._12 = -matrix._12 * reciprocalDeterminant;
    inverseMatrix._21 = -matrix._21 * reciprocalDeterminant;
    inverseMatrix._22 = matrix._11 * reciprocalDeterminant;
    return inverseMatrix;
}

// Function to perform Cholesky decomposition on a 2x2 matrix
float2x2 CholeskyDecomposition2x2(float2x2 matrix) {
    float2x2 result;
    result._11 = sqrt(matrix._11);
    result._12 = matrix._12 / result._11;
    result._21 = 0.f;
    result._22 = sqrt(matrix._22 - result._12 * result._12);
    return result;
}

#endif // !_SRENDERER_LINEAR_ALGEBRA_HEADER_