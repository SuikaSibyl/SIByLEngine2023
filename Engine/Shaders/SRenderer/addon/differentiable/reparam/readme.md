# Differentiable: Reparam Submodule

This submodule implements the paper "Reparameterizing Discontinuous Integrands for Differentiable Rendering".

`Discontinuity`, `Biased`

`Key idea`: Assuming integral over a small projected solid angle, the discontinuities typically consist of the sihoutte of a single object. And this displacement can be well-approximated using spherical rotations.

`How to find the rotation`: 