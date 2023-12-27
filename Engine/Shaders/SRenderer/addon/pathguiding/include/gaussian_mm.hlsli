#ifndef _SRENDERER_ADDON_PATHGUIDING_GAUSSIAN_EM_HEADER_
#define _SRENDERER_ADDON_PATHGUIDING_GAUSSIAN_EM_HEADER_

#include "../../../include/common/cpp_compatible.hlsli"
#include "../../../include/common/gaussian.hlsli"
#include "../../../include/common/math.hlsli"
#include "pathguiding.hlsli"

/**
 * Compute the covariance matrix of a 2D variable pair.
 * @param ex The expected value of the first variable.
 * @param ey The expected value of the second variable.
 * @param ex2 The expected value of the first variable squared.
 * @param ey2 The expected value of the second variable squared.
 * @param exy The expected value of the product of the two variables.
 * @return The covariance matrix.
*/
float2x2 Covariance2x2(float ex, float ey, float ex2, float ey2, float exy) {
    const float vx = ex2 - ex * ex;  // variance of x
    const float vy = ey2 - ey * ey;  // variance of y
    const float vxy = exy - ex * ey; // covariance of x and y
    return float2x2(vx, vxy, vxy, vy);
}

struct GMMStatictics {
    float ex;
    float ey;
    float ex2;
    float ey2;
    float exy;
};

GMMStatictics UnpackStatistics(in_ref(float4) pack0, in_ref(float4) pack1) {
    const float weightSum = pack1.y;
    GMMStatictics stat;
    stat.ex = pack0.x / weightSum;
    stat.ey = pack0.y / weightSum;
    stat.ex2 = pack0.z / weightSum;
    stat.ey2 = pack0.w / weightSum;
    stat.exy = pack1.x / weightSum;
    return stat;
}

float2x2 Covariance2x2(in_ref(GMMStatictics) statistics) {
    return Covariance2x2(statistics.ex, statistics.ey, statistics.ex2, statistics.ey2, statistics.exy);
}

MultivariateGaussian2D CreateDistribution(in_ref(GMMStatictics) statistics) {
    float2x2 covariance = Covariance2x2(statistics);
    return MultivariateGaussian2D(float2(statistics.ex, statistics.ey), covariance);
}

struct GMM2D {
    MultivariateGaussian2D lobes[4];
    float4 sufficientStats0[4];
    float4 sufficientStats1[4];
    int epoch_cap;
    
    /** Compute the responsibility of a point for a given lobe.
     * @param h The index of the lobe.
     * @param point The point. */
    float responsibility(int h, float2 point) { return sufficientStats1[h].w * lobes[h].Pdf(point); }
    
    float2 DrawSample(in_ref(float3) uv) {
        float pdf = 0.f;
        for (int h = 0; h < 4; ++h) {
            pdf += sufficientStats1[h].w;
            if (uv.z < pdf || h == 4 - 1)
                return lobes[h].DrawSample(uv.xy);
        }
        return float2(0);
    }
    [mutating]
    void build() {
        for (int h = 0; h < 4; ++h) {
            const GMMStatictics GMMstat = UnpackStatistics(sufficientStats0[h], sufficientStats1[h]);
            lobes[h] = MultivariateGaussian2D(float2(GMMstat.ex, GMMstat.ey), Covariance2x2(GMMstat));
        }
    }

    float Pdf(float2 point) {
        float pdf = 0.f;
        for (int h = 0; h < 4; ++h) {
            pdf += responsibility(h, point);
        }
        return pdf;
    }

    [mutating]
    void update(
        int h,
        float2 square_coord,
        float sumWeight,
        float exponential_factor,
    ) {
        if (sumWeight > 0) {
            // exponential smoothing vMF
            const float4 pack0 = sufficientStats0[h];
            const float4 pack1 = sufficientStats1[h];
            uint epoch_count = uint(pack1.z);
            const float alpha = pow(exponential_factor, epoch_count);
            epoch_count = clamp(epoch_count, 0, epoch_cap);
            const float4 new_pack0 = sumWeight * float4(square_coord.x, square_coord.y, square_coord.x * square_coord.x, square_coord.y * square_coord.y);
            const float2 new_pack1 = sumWeight * float2(square_coord.x * square_coord.y, 1);
            const float4 update_pack0 = ExponentialSmooth(pack0, new_pack0, alpha);
            const float2 update_pack1 = ExponentialSmooth(pack1.xy, new_pack1, alpha);
            epoch_count += 1;
            sufficientStats0[h] = update_pack0;
            sufficientStats1[h] = float4(update_pack1, epoch_count, pack1.w);
        }
    }

    [mutating]
    void stepwiseEM(
        in_ref(float) sumWeight,
        in_ref(float2) square_coord,
        in_ref(float) exponential_factor,
    ) {
        if (sumWeight <= 0) return;
        // E-step: update sufficient statistics
        vector<float, 4> pdf;
        float denom = 0.f;
        float2 x = float2(0, 0);
        for (int h = 0; h < 4; ++h) {
            pdf[h] = responsibility(h, x);
            if (isnan(pdf[h])) pdf[h] = 0.001;
            if (isinf(pdf[h])) pdf[h] = 10000;
            if (pdf[h] <= 0) pdf[h] = 0.001;
            denom += pdf[h];
        }
        // M-step: update model
        vector<float, 4> weights;
        float w_denom = 0.f;
        for (int h = 0; h < 4; ++h) {
            float posterior = pdf[h] / denom;
            update(h, square_coord, sumWeight * posterior, exponential_factor);
            weights[h] = sufficientStats1[h].y;
            w_denom += weights[h];
        }
        // normalize weights
        for (int h = 0; h < 4; ++h) {
            sufficientStats1[h].w = weights[h] / w_denom;
        }

        // sufficientStats1[0].w = 1;
        // sufficientStats1[1].w = 0;
        // sufficientStats1[2].w = 0;
        // sufficientStats1[3].w = 0;
    }
};

#endif // !_SRENDERER_ADDON_PATHGUIDING_GAUSSIAN_EM_HEADER_