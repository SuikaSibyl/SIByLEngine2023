#ifndef _SRENDERER_ADDON_HALF_TINYNN_FEATUREGRID_HLSLI_HEADER_
#define _SRENDERER_ADDON_HALF_TINYNN_FEATUREGRID_HLSLI_HEADER_

#include "tinynn-half-linear.hlsli"

struct FeatureGrid2DIndex {
    uint2 cellId;
    float2 weights;
    __init(uint2 frameDim, int2 pixelId, int2 feature_grid) {
        const float2 cellSize = float2(frameDim.x * 1.f / feature_grid.x, frameDim.y * 1.f / feature_grid.y);
        this.cellId = uint2(floor(pixelId / cellSize));
        this.weights = (float2(pixelId) / cellSize) - float2(cellId);
    }
};

[Differentiable]
HalfFeature<32> computeInterpolatedFeature(
    no_diff TensorView featureGrid,
    no_diff FeatureGrid2DIndex index
) {
    const uint2 cellId = index.cellId;
    const float2 weights = index.weights;
    HalfFeature<32> feature;
    [ForceUnroll]
    for (int i = 0; i < 16 - 2; i++) {
        float loadedf = featureGrid.load_prim(cellId.x, cellId.y, i) * (1 - weights.x) * (1 - weights.y) +
                        featureGrid.load_prim(cellId.x + 1, cellId.y, i) * weights.x * (1 - weights.y) +
                        featureGrid.load_prim(cellId.x, cellId.y + 1, i) * (1 - weights.x) * weights.y +
                        featureGrid.load_prim(cellId.x + 1, cellId.y + 1, i) * weights.x * weights.y;
        feature.vals[i] = float16_t(loadedf);
    }

    feature.vals[16 - 2] = float16_t(weights.y);
    feature.vals[16 - 1] = float16_t(weights.x);
    feature.vals[16 + 0] = sin(float16_t(weights.x) * float16_t(3.1415926f * 1));
    feature.vals[16 + 1] = cos(float16_t(weights.x) * float16_t(3.1415926f * 1));
    feature.vals[16 + 2] = sin(float16_t(weights.y) * float16_t(3.1415926f * 1));
    feature.vals[16 + 3] = cos(float16_t(weights.y) * float16_t(3.1415926f * 1));

    feature.vals[16 + 4] = sin(float16_t(weights.x) * float16_t(3.1415926f * 2));
    feature.vals[16 + 5] = cos(float16_t(weights.x) * float16_t(3.1415926f * 2));
    feature.vals[16 + 6] = sin(float16_t(weights.y) * float16_t(3.1415926f * 2));
    feature.vals[16 + 7] = cos(float16_t(weights.y) * float16_t(3.1415926f * 2));

    feature.vals[16 + 8] = sin(float16_t(weights.x) * float16_t(3.1415926f * 4));
    feature.vals[16 + 9] = cos(float16_t(weights.x) * float16_t(3.1415926f * 4));
    feature.vals[16 + 10] = sin(float16_t(weights.y) * float16_t(3.1415926f * 4));
    feature.vals[16 + 11] = cos(float16_t(weights.y) * float16_t(3.1415926f * 4));

    feature.vals[16 + 12] = sin(float16_t(weights.x) * float16_t(3.1415926f * 8));
    feature.vals[16 + 13] = cos(float16_t(weights.x) * float16_t(3.1415926f * 8));
    feature.vals[16 + 14] = sin(float16_t(weights.y) * float16_t(3.1415926f * 8));
    feature.vals[16 + 15] = cos(float16_t(weights.y) * float16_t(3.1415926f * 8));

    return feature;
}

[Differentiable]
float load_prim_PMF(
    no_diff TensorView featureGrid,
    no_diff uint2 cellId,
    no_diff int i,
    no_diff int2 pixel
) {
    const int2 lower = (int2(cellId) - 1) * 16;
    const int2 offset = pixel - lower;
    const int2 index = cellId * 32 + offset;
    const int grad_index = ((index.x + index.y * 33*32) * 14 + i);
    return featureGrid.load_prim_grad_id(cellId.x, cellId.y, i, grad_index);
}

[Differentiable]
HalfFeature<32> computeInterpolatedFeature_PMF(
    no_diff TensorView featureGrid,
    no_diff FeatureGrid2DIndex index,
    no_diff int2 pixel
) {
    const uint2 cellId = index.cellId;
    const float2 weights = index.weights;
    HalfFeature<32> feature;
    [ForceUnroll]
    for (int i = 0; i < 16 - 2; i++) {
        float loadedf = load_prim_PMF(featureGrid, uint2(cellId.x, cellId.y), i, pixel) * (1 - weights.x) * (1 - weights.y) +
                        load_prim_PMF(featureGrid, uint2(cellId.x + 1, cellId.y), i, pixel) * weights.x * (1 - weights.y) +
                        load_prim_PMF(featureGrid, uint2(cellId.x, cellId.y + 1), i, pixel) * (1 - weights.x) * weights.y +
                        load_prim_PMF(featureGrid, uint2(cellId.x + 1, cellId.y + 1), i, pixel) * weights.x * weights.y;
        feature.vals[i] = float16_t(loadedf);
    }
    
    feature.vals[16 - 2] = float16_t(weights.y);
    feature.vals[16 - 1] = float16_t(weights.x);
    feature.vals[16 + 0] = sin(float16_t(weights.x) * float16_t(3.1415926f * 1));
    feature.vals[16 + 1] = cos(float16_t(weights.x) * float16_t(3.1415926f * 1));
    feature.vals[16 + 2] = sin(float16_t(weights.y) * float16_t(3.1415926f * 1));
    feature.vals[16 + 3] = cos(float16_t(weights.y) * float16_t(3.1415926f * 1));

    feature.vals[16 + 4] = sin(float16_t(weights.x) * float16_t(3.1415926f * 2));
    feature.vals[16 + 5] = cos(float16_t(weights.x) * float16_t(3.1415926f * 2));
    feature.vals[16 + 6] = sin(float16_t(weights.y) * float16_t(3.1415926f * 2));
    feature.vals[16 + 7] = cos(float16_t(weights.y) * float16_t(3.1415926f * 2));

    feature.vals[16 + 8] = sin(float16_t(weights.x) * float16_t(3.1415926f * 4));
    feature.vals[16 + 9] = cos(float16_t(weights.x) * float16_t(3.1415926f * 4));
    feature.vals[16 + 10] = sin(float16_t(weights.y) * float16_t(3.1415926f * 4));
    feature.vals[16 + 11] = cos(float16_t(weights.y) * float16_t(3.1415926f * 4));

    feature.vals[16 + 12] = sin(float16_t(weights.x) * float16_t(3.1415926f * 8));
    feature.vals[16 + 13] = cos(float16_t(weights.x) * float16_t(3.1415926f * 8));
    feature.vals[16 + 14] = sin(float16_t(weights.y) * float16_t(3.1415926f * 8));
    feature.vals[16 + 15] = cos(float16_t(weights.y) * float16_t(3.1415926f * 8));

    return feature;
}

[Differentiable]
float load_prim_Narrow(
    no_diff TensorView featureGrid,
    no_diff uint2 cellId,
    no_diff int i,
    no_diff int2 cellSampled
) {
    const int grad_index = all(int2(cellId) == cellSampled) ? -1 : -2;
    return featureGrid.load_prim_gradid(cellId.x, cellId.y, i, grad_index);
}

[Differentiable]
HalfFeature<32> computeInterpolatedFeature_Narrow(
    no_diff TensorView featureGrid,
    no_diff FeatureGrid2DIndex index,
    no_diff int2 cellSampled
) {
    const uint2 cellId = index.cellId;
    const float2 weights = index.weights;
    HalfFeature<32> feature;
    [ForceUnroll]
    for (int i = 0; i < 16 - 2; i++) {
        float loadedf = load_prim_Narrow(featureGrid, uint2(cellId.x, cellId.y), i, cellSampled) * (1 - weights.x) * (1 - weights.y) +
                        load_prim_Narrow(featureGrid, uint2(cellId.x + 1, cellId.y), i, cellSampled) * weights.x * (1 - weights.y) +
                        load_prim_Narrow(featureGrid, uint2(cellId.x, cellId.y + 1), i, cellSampled) * (1 - weights.x) * weights.y +
                        load_prim_Narrow(featureGrid, uint2(cellId.x + 1, cellId.y + 1), i, cellSampled) * weights.x * weights.y;
        feature.vals[i] = float16_t(loadedf);
    }
    
    feature.vals[16 - 2] = float16_t(weights.y);
    feature.vals[16 - 1] = float16_t(weights.x);
    feature.vals[16 + 0] = sin(float16_t(weights.x) * float16_t(3.1415926f * 1));
    feature.vals[16 + 1] = cos(float16_t(weights.x) * float16_t(3.1415926f * 1));
    feature.vals[16 + 2] = sin(float16_t(weights.y) * float16_t(3.1415926f * 1));
    feature.vals[16 + 3] = cos(float16_t(weights.y) * float16_t(3.1415926f * 1));

    feature.vals[16 + 4] = sin(float16_t(weights.x) * float16_t(3.1415926f * 2));
    feature.vals[16 + 5] = cos(float16_t(weights.x) * float16_t(3.1415926f * 2));
    feature.vals[16 + 6] = sin(float16_t(weights.y) * float16_t(3.1415926f * 2));
    feature.vals[16 + 7] = cos(float16_t(weights.y) * float16_t(3.1415926f * 2));

    feature.vals[16 + 8] = sin(float16_t(weights.x) * float16_t(3.1415926f * 4));
    feature.vals[16 + 9] = cos(float16_t(weights.x) * float16_t(3.1415926f * 4));
    feature.vals[16 + 10] = sin(float16_t(weights.y) * float16_t(3.1415926f * 4));
    feature.vals[16 + 11] = cos(float16_t(weights.y) * float16_t(3.1415926f * 4));

    feature.vals[16 + 12] = sin(float16_t(weights.x) * float16_t(3.1415926f * 8));
    feature.vals[16 + 13] = cos(float16_t(weights.x) * float16_t(3.1415926f * 8));
    feature.vals[16 + 14] = sin(float16_t(weights.y) * float16_t(3.1415926f * 8));
    feature.vals[16 + 15] = cos(float16_t(weights.y) * float16_t(3.1415926f * 8));

    return feature;
}

[Differentiable]
float load_prim_MIS(
    no_diff TensorView featureGrid,
    no_diff uint2 cellId,
    no_diff int i,
    no_diff int2 pixel,
) {
    float mis = 1.f;
    if (any(pixel < 0)) {
        mis = 0.f;
    }
    // if (pixel.x % 16 == 0 && pixel.y % 16 == 0) {
    //     mis = 1.f;
    // }
    // else if (pixel.x % 16 == 0 || pixel.y % 16 == 0) {
    //     mis = 1.f / 2.f;
    // }
    else {
        mis = 1.f / 4.f;
    }
    // else if (any(pixel < 0)) {
    //     mis = 0.f;
    // }
    // else {
    //     mis = 0.f;
    // }
    
    return featureGrid.load_prim_gscalar(cellId.x, cellId.y, i, mis);
}

[Differentiable]
HalfFeature<32> computeInterpolatedFeature_MIS(
    no_diff TensorView featureGrid,
    no_diff FeatureGrid2DIndex index,
    no_diff int2 pixel
) {
    const uint2 cellId = index.cellId;
    const float2 weights = index.weights;
    HalfFeature<32> feature;
    [ForceUnroll]
    for (int i = 0; i < 16 - 2; i++) {
        float loadedf = load_prim_MIS(featureGrid, int2(cellId.x, cellId.y), i, pixel) * (1 - weights.x) * (1 - weights.y) +
                        load_prim_MIS(featureGrid, int2(cellId.x + 1, cellId.y), i, pixel) * weights.x * (1 - weights.y) +
                        load_prim_MIS(featureGrid, int2(cellId.x, cellId.y + 1), i, pixel) * (1 - weights.x) * weights.y +
                        load_prim_MIS(featureGrid, int2(cellId.x + 1, cellId.y + 1), i, pixel) * weights.x * weights.y;
        feature.vals[i] = float16_t(loadedf);
    }
    
    feature.vals[16 - 2] = float16_t(weights.y);
    feature.vals[16 - 1] = float16_t(weights.x);
    feature.vals[16 + 0] = sin(float16_t(weights.x) * float16_t(3.1415926f * 1));
    feature.vals[16 + 1] = cos(float16_t(weights.x) * float16_t(3.1415926f * 1));
    feature.vals[16 + 2] = sin(float16_t(weights.y) * float16_t(3.1415926f * 1));
    feature.vals[16 + 3] = cos(float16_t(weights.y) * float16_t(3.1415926f * 1));

    feature.vals[16 + 4] = sin(float16_t(weights.x) * float16_t(3.1415926f * 2));
    feature.vals[16 + 5] = cos(float16_t(weights.x) * float16_t(3.1415926f * 2));
    feature.vals[16 + 6] = sin(float16_t(weights.y) * float16_t(3.1415926f * 2));
    feature.vals[16 + 7] = cos(float16_t(weights.y) * float16_t(3.1415926f * 2));

    feature.vals[16 + 8] = sin(float16_t(weights.x) * float16_t(3.1415926f * 4));
    feature.vals[16 + 9] = cos(float16_t(weights.x) * float16_t(3.1415926f * 4));
    feature.vals[16 + 10] = sin(float16_t(weights.y) * float16_t(3.1415926f * 4));
    feature.vals[16 + 11] = cos(float16_t(weights.y) * float16_t(3.1415926f * 4));

    feature.vals[16 + 12] = sin(float16_t(weights.x) * float16_t(3.1415926f * 8));
    feature.vals[16 + 13] = cos(float16_t(weights.x) * float16_t(3.1415926f * 8));
    feature.vals[16 + 14] = sin(float16_t(weights.y) * float16_t(3.1415926f * 8));
    feature.vals[16 + 15] = cos(float16_t(weights.y) * float16_t(3.1415926f * 8));

    return feature;
}

#endif // !_SRENDERER_ADDON_HALF_TINYNN_FEATUREGRID_HLSLI_HEADER_