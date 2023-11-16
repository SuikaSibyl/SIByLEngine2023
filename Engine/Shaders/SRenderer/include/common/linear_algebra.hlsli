#ifndef _SRENDERER_LINEAR_ALGEBRA_HEADER_
#define _SRENDERER_LINEAR_ALGEBRA_HEADER_

#include "cpp_compatible.hlsli"
#include "math.hlsli"

// Function to calculate the inverse of a 2x2 matrix
float2x2 Inverse2x2(in_ref(float2x2) matrix) {
    const float determinant = matrix._11 * matrix._22 - matrix._12 * matrix._21;
    const float reciprocalDeterminant = 1.0 / determinant;
    float2x2 inverseMatrix;
    inverseMatrix[0][0] = matrix[1][1] * reciprocalDeterminant;
    inverseMatrix[0][1] = -matrix[0][1] * reciprocalDeterminant;
    inverseMatrix[1][0] = -matrix[1][0] * reciprocalDeterminant;
    inverseMatrix[1][1] = matrix[0][0] * reciprocalDeterminant;
    return inverseMatrix;
}

// Function to calculate the inverse of a 2x2 matrix
float2x2 Inverse2x2(in_ref(float2x2) matrix, float determinant) {
    const float reciprocalDeterminant = 1.0 / determinant;
    float2x2 inverseMatrix;
    inverseMatrix[0][0] = matrix._22 * reciprocalDeterminant;
    inverseMatrix[0][1] = -matrix._12 * reciprocalDeterminant;
    inverseMatrix[1][0] = -matrix._21 * reciprocalDeterminant;
    inverseMatrix[1][1] = matrix._11 * reciprocalDeterminant;
    return inverseMatrix;
}

// Function to calculate the inverse of a 2x2 matrix
double2x2 Inverse2x2(in_ref(double2x2) matrix, double determinant) {
    const double reciprocalDeterminant = 1.0 / determinant;
    double2x2 inverseMatrix;
    inverseMatrix[0][0] = matrix._22 * reciprocalDeterminant;
    inverseMatrix[0][1] = -matrix._12 * reciprocalDeterminant;
    inverseMatrix[1][0] = -matrix._21 * reciprocalDeterminant;
    inverseMatrix[1][1] = matrix._11 * reciprocalDeterminant;
    return inverseMatrix;
}

// Function to perform Cholesky decomposition on a 2x2 matrix
float2x2 CholeskyDecomposition2x2(in_ref(float2x2) matrix) {
    float2x2 result;
    result[0][0] = sqrt(matrix._11);
    result[0][1] = matrix._12 / result._11;
    result[1][0] = 0.f;
    result[1][1] = sqrt(matrix._22 - result._12 * result._12);
    return result;
}

// Function to perform Cholesky decomposition on a 2x2 matrix
double2x2 CholeskyDecomposition2x2(in_ref(double2x2) matrix) {
    double2x2 result;
    result[0][0] = sqrt(matrix._11);
    result[0][1] = matrix._12 / result._11;
    result[1][0] = 0.;
    result[1][1] = sqrt(matrix._22 - result._12 * result._12);
    return result;
}

float QuadraticForm(in_ref(float2x2) matrix, in_ref(float2) vector) {
    return dot(vector, mul(matrix, vector));
}

// Function to calculate the inverse of a 3x3 matrix.
[Differentiable]
float3x3 Inverse3x3(in_ref(float3x3) m) {
    const float oneOverDet = 1.f / determinant(m);
    float3x3 result;
    result[0][0] = +(m[1][1] * m[2][2] - m[1][2] * m[2][1]) * oneOverDet;
    result[0][1] = -(m[0][1] * m[2][2] - m[0][2] * m[2][1]) * oneOverDet;
    result[0][2] = +(m[0][1] * m[1][2] - m[0][2] * m[1][1]) * oneOverDet;
    result[1][0] = -(m[1][0] * m[2][2] - m[1][2] * m[2][0]) * oneOverDet;
    result[1][1] = +(m[0][0] * m[2][2] - m[0][2] * m[2][0]) * oneOverDet;
    result[1][2] = -(m[0][0] * m[1][2] - m[0][2] * m[1][0]) * oneOverDet;
    result[2][0] = +(m[1][0] * m[2][1] - m[1][1] * m[2][0]) * oneOverDet;
    result[2][1] = -(m[0][0] * m[2][1] - m[0][1] * m[2][0]) * oneOverDet;
    result[2][2] = +(m[0][0] * m[1][1] - m[0][1] * m[1][0]) * oneOverDet;
    return result;
}

[Differentiable]
float3x3 identity3x3() {
    return float3x3(1, 0, 0,
                    0, 1, 0,
                    0, 0, 1);
}

[Differentiable]
float3x3 outer_product(float3 v0, float3 v1) {
    return float3x3(
        v0 * v1.x,
        v0 * v1.y,
        v0 * v1.z);
}

#endif // !_SRENDERER_LINEAR_ALGEBRA_HEADER_