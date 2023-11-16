#ifndef _SRENDERER_ADDON_DIFFERENTIABLE_AUTODIFF_SCENE_IO_HEADER_
#define _SRENDERER_ADDON_DIFFERENTIABLE_AUTODIFF_SCENE_IO_HEADER_

// __exported import DiffRendering.SceneGradientInfo;
// import DiffRendering.DiffDebug;

RWTexture2D<float4> gOutputDColor;

// Avoid compilation warnings if not running in a differentiable mode.
#ifndef DIFF_MODE
#define DIFF_MODE 0
#endif

// Scene IO wrapper for propagating gradients.
struct DiffSceneIO {
    // Vertex position
    [ForwardDerivative(fwd_loadVertexPositionsW)]
    [BackwardDerivative(bwd_loadVertexPositionsW)]
    [PreferRecompute]
    void loadVertexPositionsW(
        SceneGradientInfo gradInfo, 
        GeometryInstanceID instanceID, 
        uint triangleID, 
        out float3 p[3]
    ) {
        no_diff gScene.getVertexPositionsW(instanceID, triangleID, p);
    }

    [Differentiable]
    [PreferRecompute]
    void fwd_loadVertexPositionsW(
        DifferentialPair<SceneGradientInfo> dpGradInfo,
        GeometryInstanceID instanceID,
        uint triangleID,
        out DifferentialPair<float3[3]> dpPos
    ) {
        float3 p[3];
        loadVertexPositionsW(dpGradInfo.p, instanceID, triangleID, p);

        float3.Differential dPos[3];
        [ForceUnroll]
        for (uint i = 0; i < 3; i++)
            dPos[i] = float3(0.f);

#if DIFF_MODE == 2 // ForwardDiffDebug
        // For visualizing a gradient image.
        if (dpGradInfo.d.flag.gradMode == GradientMode.ForwardDebug && gDiffDebug.varType == DiffVariableType::GeometryTranslation &&
            gDiffDebug.id.x == instanceID.index)
        {
            [ForceUnroll]
            for (uint i = 0; i < 3; i++)
                dPos[i] = gDiffDebug.grad.xyz;
        }
#endif

        dpPos = diffPair(p, dPos);
    }
    
    [Differentiable]
    [PreferRecompute]
    void bwd_loadVertexPositionsW(
        inout DifferentialPair<SceneGradientInfo> dpGradInfo,
        GeometryInstanceID instanceID,
        uint triangleID,
        in float3.Differential dPos[3]
    )
    {
#if DIFF_MODE == 3 // BackwardDiffDebug
        // For visualizing a gradient image.
        if (dpGradInfo.d.flag.gradMode == GradientMode.Scene && 
            gDiffDebug.varType == DiffVariableType::GeometryTranslation &&
            gDiffDebug.id.x == instanceID.index)
        {
            float3 velocity = gDiffDebug.grad.xyz;
            float value = dot(velocity, dPos[0]) + dot(velocity, dPos[1]) + dot(velocity, dPos[2]);
            gOutputDColor[dpGradInfo.p.pixel] += float4(value, 0.f, 0.f, 0.f);
        }
#endif
    }

    // TODO: Add custom derivatives for the following functions.

    // Vertex normal
    [Differentiable]
    void loadVertexNormalsW(SceneGradientInfo gradInfo, GeometryInstanceID instanceID, uint triangleID, out float3 n[3])
    {
        uint3 indices = gScene.getIndices(instanceID, triangleID);
        float3x3 mat = no_diff gScene.getInverseTransposeWorldMatrix(instanceID);

        [ForceUnroll]
        for (int i = 0; i < 3; i++)
        {
            var v = no_diff gScene.getVertex(indices[i]);
            n[i] = normalize(mul(mat, v.normal));
        }
    }

    // Camera position
    [Differentiable]
    float3 loadCameraPositionW(SceneGradientInfo gradInfo) { return no_diff gScene.camera.getPosition(); }
};

#endif // _SRENDERER_ADDON_DIFFERENTIABLE_AUTODIFF_SCENE_IO_HEADER_