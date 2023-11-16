# Notation for “Unbiased Warped-Area Sampling for Differentiable Rendering ”



Divergence Theorem

- $V$ is a region in space
- $\part V$ is the boundary of $V$

The volume integral of the divergence $\nabla F$ over a closed in
$$
\int_V (\nabla\cdot \mathbb{F})\mathrm{d}V=\int_{\part V}\mathbb{F}\cdot \mathrm{d}a
$$


Define the vector field $\vec{\mathcal{V}}$

- continuous: for divergence exists
- boundary consistent: 





## Algorithm Derivation


$$
\mathbb{I}_B=\oint_{\part D}f(s;\theta)
$$





$$
\widehat{\part_\theta^b\mathbb{I}}\leftarrow \left< \part_\omega L, \hat{\mathcal{V}}_\theta(\omega) \right> + L\nabla_{\omega}\cdot \hat{\mathcal{V}}_\theta(\omega)
$$



$$
\begin{aligned}
\mathbb{I}_B&=\oint_{\partial \mathrm{D}} f(\omega;\theta)(\partial_{\theta}\omega\cdot \hat{\mathbf{n}})\mathrm{d}\partial\mathrm{D}_i(\theta)\\
&=\iint_{D'}\nabla_\omega\cdot(f(\omega;\theta)\vec{\mathcal{V}}_\theta(\omega))\mathrm{d}\omega
\end{aligned}
$$
According to the 2nd property of divergence:
$$
\iint_{D'}\left(\partial_\omega f(\omega;\theta)\right)\cdot\vec{\mathcal{V}_\theta(\omega)\mathrm{d}\omega}+ \iint_{D'}(\nabla_\omega\cdot\mathcal{V}_\theta(\omega))f(\omega;\theta)\mathrm{d}\omega
$$
