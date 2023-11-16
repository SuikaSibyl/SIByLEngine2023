# Notation for "Reparameterizing Discontinuous Integrands for Differentiable Rendering"



Consider a very simple case
$$
\int_0^x 1 \mathrm{d}t
$$
Mathematically, it is a definite integral with a variable on upper limit, and thus
$$
\frac{\mathrm{d}}{\mathrm{d}x}\int_0^xf(t)\mathrm{d}t=f(x)
$$
More generally, we have
$$
\int_{g(x)}^{h(t)}f(x)\mathrm{d}x=h'(t)\cdot f(h(t))-g'(t)\cdot f(g(t))
$$
Leibniz Integral Rule [https://mathworld.wolfram.com/LeibnizIntegralRule.html]:
$$
\frac{\partial}{\partial z}\int_{a(z)}^{b(z)}f(x,z)\mathrm{d}x =\int_{a(z)}^{b(z)} \frac{\partial f}{\partial z}\mathrm{d}x+f(b(z),z)\frac{\partial b}{\partial z} - f(a(z),z)\frac{\partial a}{\partial z}
$$
The parametrization trick essentially put the discontinuity from $f(x,z)$ onto the limits.



In forward Monte Carlo pass, we are computing $f(x,z)$, and the derivative is some how $f(b(z),z)\frac{\partial b}{\partial z}$.



The clever part is, here, we can compute the forward Monte Carlo thing correctly, while its AD is exactly the term we need.

## 2. Control Variates


$$
\alpha = -\frac{\mathrm{Cov}(E,F)}{\mathrm{Var}(F)}
$$
**Intuition:** $\alpha$ should be negative in the case of a positive correlation of $E$ and $F$, as the estimation of (E-F) is likely to be closer to 0 and has lower variance. 

The observation in the paper points out that the differentiation of the estimator (the gradient of the parameters) is correlated with the differentiation of the vMF weights.

Furthermore, we know the expected value of the estimator $\frac{\partial F}{\partial \theta}$ is zero.



 
