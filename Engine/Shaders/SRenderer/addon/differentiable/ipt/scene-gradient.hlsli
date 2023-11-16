#ifndef _SRENDERER_ADDON_DIFFERENTIABLE_SCENE_GRADIENT_HEADER_
#define _SRENDERER_ADDON_DIFFERENTIABLE_SCENE_GRADIENT_HEADER_

enum GradientMode {
    None,         // No gradients are propagated (in either direction) for this pass.
    Scene,        // Pass gradients to scene parameters normally.
    ForwardDebug, // Use forward-mode gradients to debug.
};

// Create a differentiable flag for higher-order differentiation.
// It is mainly used for:
// 1. Set dFlag = { GradientMode.None } to zero out gradients from/to scene parameters for the Jacobian computation.
// 2. Set dFlag = { GradientMode.ForwardDebug } to pass in forward-mode gradients for debugging.
struct SceneGradientFlag : IDifferentiable {
    GradientMode gradMode;

    typealias Differential = SceneGradientFlag;

    [ForwardDerivative(__fwd_make)]
    [BackwardDerivative(__bwd_make)]
    static SceneGradientFlag make(GradientMode mode)
    {
        SceneGradientFlag info = { mode };
        return info;
    }

    static DifferentialPair<SceneGradientFlag> __fwd_make(GradientMode mode)
    {
        SceneGradientFlag flag = { mode };
        return diffPair(flag, flag);
    }

    static void __bwd_make(GradientMode mode, SceneGradientFlag.Differential dFlag)
    {
        // Do nothing.
    }

    [Differentiable]
    static SceneGradientFlag dadd(SceneGradientFlag a, SceneGradientFlag b)
    {
        // Default to all gradients.
        return SceneGradientFlag.make(GradientMode.Scene);
    }

    [Differentiable]
    static SceneGradientFlag dmul(SceneGradientFlag a, SceneGradientFlag b)
    {
        // Default to all gradients.
        return SceneGradientFlag.make(GradientMode.Scene);
    }

    [Differentiable]
    static SceneGradientFlag dzero()
    {
        // Default to all gradients.
        return SceneGradientFlag.make(GradientMode.Scene);
    }
};

struct SceneGradientInfo : IDifferentiable
{
    // Differentiable flag for higher-order differentiation.
    SceneGradientFlag flag;

    // Extra data.
    uint2 pixel;

    [Differentiable]
    __init(SceneGradientFlag _flag, uint2 _pixel)
    {
        flag = _flag;
        pixel = _pixel;
    }
};

#endif // !_SRENDERER_ADDON_DIFFERENTIABLE_SCENE_GRADIENT_HEADER_