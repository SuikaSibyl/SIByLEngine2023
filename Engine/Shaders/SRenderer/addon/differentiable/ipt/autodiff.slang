#include "../../../include/common/light_impl.hlsli"
#include "../../../include/common/raydifferential.hlsli"
#include "../../../include/common/sampling.hlsli"
#include "../../../include/common/vmf.hlsli"
#include "../../../include/diff_descriptor_set.hlsli"
#include "../../../raytracer/spt_differentiable.hlsli"
#include "../../../raytracer/spt_interface.hlsli"
#include "reparam.hlsli"

[[vk::push_constant]]
cbuffer PushConstants {
    uint rand_seed;
    uint max_depth;
    uint spp;
};

RWTexture2D<float4> u_color;

// Convert square samples to Gaussian samples
float2 sampleGaussian(float2 xy, float sigma) {
    return (sqrt(-2.f * log(xy.x)) * float2(cos(M_2PI * xy.y), sin(M_2PI * xy.y))) * sigma;
}

// Evaluate Gaussian at xy
[Differentiable]
[PreferRecompute]
float evalGaussian(float2 xy, no_diff float sigma) {
    const float alpha = (-1.f / (2.f * sigma * sigma));
    return exp(alpha * dot(xy, xy)) * (1.f / (2.f * M_PI * sigma * sigma));
}

[Differentiable]
float3 path_tracing(
    SceneQueryAD sceneQuery, float2 pixel, 
    no_diff float2 pixelCenter, inout SampleGenerator sg
) {
    PathData pathData = PathData();
    IntersectionAD isect = IntersectionAD();

    // Warped-area reparameterization for primary rays.
    if (kUseWAR)
    {
        float weight = reparameterizeScreenSample(sceneQuery, gDiffPTData.params.frameDim, sg, pixel);
        pathData.thp *= weight;
    }

    // Use Gaussian pixel filter.
    float pixelWeight = evalGaussian(pixel - pixelCenter, kPixelGaussianSigma);
    pathData.thp *= pixelWeight / detach(pixelWeight);

    float3 primaryRayOrigin = sceneQuery.loadCameraPosition();
    float3 primaryRayDirection = sceneQuery.computeCameraRayDirection(pixel, gDiffPTData.params.frameDim);
    RayAD ray = RayAD(detach(primaryRayOrigin), primaryRayDirection, sceneQuery.gradInfo.pixel);

    if (!sceneQuery.traceRayInlineAD(ray, isect, SceneQueryAD.DiffIntersectionMode.AttachToRay))
        return pathData.radiance;

    [MaxIters(MAX_BOUNCES + 2)]
    while (!pathData.terminated)
    {
        handleHit(sceneQuery, isect, gDiffPTData.emissiveSampler, pathData, ray, sg, kUseWAR);

        if (pathData.terminated)
            break;

        if (!sceneQuery.traceRayInlineAD(ray, isect, SceneQueryAD.DiffIntersectionMode.AttachToRay))
            break;
    }

    return pathData.radiance;
}

[Differentiable]
float3 path_tracing(SceneQueryAD sceneQuery, uint2 pixel) {
    float3 result = float3(0.f);

    var params = gDiffPTData.params;
    uint frameSeed = bool(params.useFixedSeed) ? params.fixedSeed : params.frameCount;

    // Antithetic sampling for differentiable rendering (interior term).
    SampleGenerator sg = SampleGenerator(pixel, frameSeed / 2);

    // Throw away some samples.
    sampleNext2D(sg);
    sampleNext2D(sg);
    sampleNext2D(sg);

    // Use Gaussian pixel filter.
    float2 pixelf = pixel + float2(0.5f, 0.5f);

    float2 uniform2D = no_diff sampleNext2D(sg);
    float2 jitter2D = no_diff sampleGaussian(uniform2D, kPixelGaussianSigma);

    // Antithetic sampling for differentiable rendering (interior term).
    if (kUseAntitheticSampling && frameSeed % 2 == 0)
        jitter2D = -jitter2D;

    // Trace an interior path.
    result += tracePath(sceneQuery, pixelf + jitter2D, pixelf, sg);

    return result;
}

[shader("raygeneration")]
void RgenMain() {
    // The resolution of the image, which is the same as the launch size:
    const int2 resolution = int2(DispatchRaysDimensions().xy);
    const int2 pixel = int2(DispatchRaysIndex().xy);
    // If the pixel is outside of the image, don't do anything:
    if (any(pixel >= resolution)) return;
    RandomSamplerState RNG = InitRandomSampler(pixel, rand_seed);
    const float2 jitter_pixel = pixel + GetNextRandomFloat2(RNG) - float2(0.5); // manually jitter
    const Ray primaryRay = generateRay(jitter_pixel, resolution, false, globalUniform.cameraData);

    // kDiffMode == DiffMode::Primal
    if (true) {
        // Primal rendering.
        float3 color = path_tracing(sceneQuery, pixel);
        u_color[pixel] = float4(color, 1.f);
    }
    else if (kDiffMode == DiffMode::BackwardDiff) {
        var params = gDiffPTData.params;
        float3 color = float3(0.f);
        if (params.runBackward == 0) {
            color = tracePaths(sceneQuery, pixel);
            gOutputColor[pixel] = float4(color, 1.f);
        }
        else {
            // Set differential query mode to `Scene` for visualizing gradient images.
            SceneQueryAD.Differential dQuery;
            dQuery.gradInfo = { SceneGradientFlag.make(GradientMode.Scene) };

            DifferentialPair<SceneQueryAD> dpQuery = diffPair(sceneQuery, dQuery);

            uint pixelIdx = pixel.y * params.frameDim.x + pixel.x;
            bwd_diff(tracePaths)(dpQuery, pixel, dLdI[pixelIdx]);
        }
    }
    else if (kDiffMode == DiffMode::ForwardDiffDebug) {
        // Forward-mode differentiable rendering for visualizing gradients.
        // Set differential query mode to `ForwardDebug` for visualizing gradient images.
        SceneQueryAD.Differential dQuery;
        dQuery.gradInfo = { SceneGradientFlag.make(GradientMode.ForwardDebug) };

        DifferentialPair<float3> dpColor = fwd_diff(path_tracing)(diffPair(sceneQuery, dQuery), pixel);
        gOutputColor[pixel] = float4(dpColor.p, 1.f);

        // Clamp gradient values.
        float maxValue = gDiffPTData.params.assertThreshold;
        bool resIsFinite = (dpColor.d[0] >= -maxValue && dpColor.d[0] <= maxValue) &&
                           (dpColor.d[1] >= -maxValue && dpColor.d[1] <= maxValue) &&
                           (dpColor.d[2] >= -maxValue && dpColor.d[2] <= maxValue);
        if (resIsFinite)
            gOutputDColor[pixel] = float4(dpColor.d[0], 0.f, 0.f, 1.f); // Only show the red channel.
    }
    // // kDiffMode == DiffMode::BackwardDiffDebug
    // else if (kDiffMode == DiffMode::BackwardDiffDebug) {
    //     // Reverse-mode differentiable rendering for visualizing gradients.
    //     // Set differential query mode to `Scene` for visualizing gradient images.
    //     SceneQueryAD.Differential dQuery;
    //     dQuery.gradInfo = { SceneGradientFlag.make(GradientMode.Scene) };

    //     DifferentialPair<SceneQueryAD> dpQuery = diffPair(sceneQuery, dQuery);
    //     bwd_diff(path_tracing)(dpQuery, pixel, float3(1.f, 0.f, 0.f));
    // }
    
}