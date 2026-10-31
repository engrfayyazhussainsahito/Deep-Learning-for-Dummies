---
title: "Generative Models II: Diffusion, Flows, and Autoregressive Models"
phase: Advanced Topics
summary: Denoising diffusion from first principles (forward process, reverse process, the variational bound and the epsilon-prediction loss), noise schedules and v-prediction, score matching and the SDE/probability-flow view, DDIM and fast samplers, classifier and classifier-free guidance, latent diffusion with U-Net and DiT backbones, normalizing flows and autoregressive models, a comparison of all generative families, and the societal risks of synthetic media. The lab trains a small class-conditional DDPM on MNIST and explores sampling steps and guidance.
---

:::key Learning objectives
By the end of this week you should be able to:
- Write the DDPM forward process, derive the closed form $q(x_t\mid x_0)$, and explain why it makes training cheap.
- Derive the reverse-process posterior $q(x_{t-1}\mid x_t,x_0)$, the variational bound, and the simplified $\epsilon$-prediction loss, and say what the simplification throws away.
- Compare linear and cosine noise schedules through $\bar\alpha_t$ and the signal-to-noise ratio, and convert between $\epsilon$-, $x_0$-, and $v$-parameterizations.
- Define the score function, derive denoising score matching, and show that $\epsilon$-prediction is score estimation up to a scale.
- State the forward SDE, the reverse-time SDE and the probability-flow ODE, and explain how DDPM and DDIM are discretizations of them.
- Derive classifier guidance and classifier-free guidance and explain the quality/diversity trade-off controlled by the guidance scale.
- Explain latent diffusion, cross-attention conditioning, and the architectural differences between U-Net and DiT denoisers.
- Describe normalizing flows (change of variables, coupling layers) and autoregressive models (PixelCNN, WaveNet, Transformers over tokens), and place all five generative families in one comparison.
- Discuss deepfakes, provenance (C2PA), watermarking and other risks of generative media with technical precision.
- Train a class-conditional DDPM on MNIST and run sampling-step and guidance experiments.
:::

[Week 12](week-12.html) ended with a tension. VAEs train stably and cover the data, but their samples are blurry; GANs make sharp samples but are hard to train and can drop modes. Between roughly 2020 and 2022 a third family closed most of that gap and then took over image, audio and video generation: **diffusion models**. The idea is almost embarrassingly simple. Take a data point and add a little Gaussian noise, again and again, until nothing but noise is left. Then train a neural network to undo *one small step* of that corruption. Generating data means starting from pure noise and applying the learned denoiser many times.

Why does that work so much better than asking a network to produce an image in one shot? Because the hard problem (go from noise to a sharp, globally coherent image) is split into hundreds of easy ones (remove a little noise from a slightly-noisy image), and each easy problem is a plain regression with a stable, well-behaved loss. This week derives the model from three angles that turn out to be the same thing: a **variational latent-variable model** (a hierarchical VAE with a fixed encoder), a **denoising / score-matching model** (a network that points toward higher-density regions), and a **differential equation** (a continuous flow that carries noise to data). Then we make the sampler fast (DDIM), make it controllable (guidance, text conditioning), make it scale (latent diffusion, Transformers), look at the two other likelihood-based families (flows and autoregressive models), compare everything, and finish with the societal side. The lab trains a small DDPM on MNIST from scratch.

## The diffusion idea in one picture {#idea}

{{figure: week13-forward-reverse | Diffusion as a Markov chain. The forward process $q$ gradually adds noise and has no parameters; the reverse process $p_\theta$ is learned and removes noise one step at a time. The dashed arc is the closed-form shortcut $q(x_t\mid x_0)$.}}

Two Markov chains, one fixed and one learned:

- The **forward (noising) process** $q(x_t\mid x_{t-1})$ for $t=1,\dots,T$ turns a data point $x_0\sim p_\text{data}$ into a nearly-pure-Gaussian $x_T$. It is designed by hand and has **no learnable parameters**.
- The **reverse (denoising) process** $p_\theta(x_{t-1}\mid x_t)$ goes from $x_T\sim\mathcal N(0,I)$ back to $x_0$. A neural network parameterizes it. If the forward steps are small, the true reverse step is also (close to) a Gaussian, which is why a Gaussian with a learned mean is a *reasonable* model of it, not just a convenient one.

The model was introduced by Sohl-Dickstein, Weiss, Maheswaranathan and Ganguli in 2015 (inspired by non-equilibrium thermodynamics), then sat nearly unnoticed until Ho, Jain and Abbeel (2020) showed with a simple loss, a U-Net and a careful schedule that it could rival GANs on CIFAR-10 and 256x256 LSUN images. In parallel, Song and Ermon (2019) developed noise-conditional score networks from the score-matching tradition. Song et al. (2021) then showed the two lines are one theory.

:::history Two roads to the same place
The **thermodynamic** road: Sohl-Dickstein et al. (2015) proposed learning to reverse a diffusion process. The **score-matching** road: Hyvarinen (2005) introduced score matching, Vincent (2011) connected it to denoising autoencoders, and Song and Ermon (2019) trained a network on many noise levels and sampled with Langevin dynamics. Ho et al. (2020) unified the practicalities (a simple $\epsilon$-prediction loss, a U-Net backbone), and Song et al. (2021) framed both with stochastic differential equations. Modern systems (Stable Diffusion, Imagen, DALL-E 2 and 3, video models) are descendants of this merged line. Keep both vocabularies in your head: papers use them interchangeably.
:::

## The forward process and its closed form {#forward}

### One step of noising {#forward-step}

Fix a **variance schedule** $\beta_1,\dots,\beta_T\in(0,1)$ (small numbers, typically from $10^{-4}$ to $0.02$ in the original paper). The forward step is

$$
q(x_t\mid x_{t-1})=\mathcal N\!\big(x_t;\ \sqrt{1-\beta_t}\,x_{t-1},\ \beta_t I\big),
\qquad\text{i.e.}\qquad
x_t=\sqrt{1-\beta_t}\,x_{t-1}+\sqrt{\beta_t}\,\epsilon_{t},\quad \epsilon_t\sim\mathcal N(0,I).
$$

Intuition: shrink the signal slightly (by $\sqrt{1-\beta_t}$) and add a little fresh noise (variance $\beta_t$). The shrinking matters. If you only added noise the variance would grow without bound; with the shrink factor, the process is **variance preserving**: if $x_{t-1}$ has unit variance per coordinate, so does $x_t$, because $(1-\beta_t)\cdot 1+\beta_t=1$. Eventually the stationary distribution is $\mathcal N(0,I)$, whatever the data looked like.

Write $\alpha_t=1-\beta_t$ and $\bar\alpha_t=\prod_{s=1}^t\alpha_s$.

### The closed-form noising distribution {#closed-form}

:::math Derivation: jumping straight to step $t$
Substitute the one-step recursion into itself:

$$
\begin{aligned}
x_t&=\sqrt{\alpha_t}\,x_{t-1}+\sqrt{1-\alpha_t}\,\epsilon_t\\
&=\sqrt{\alpha_t}\big(\sqrt{\alpha_{t-1}}\,x_{t-2}+\sqrt{1-\alpha_{t-1}}\,\epsilon_{t-1}\big)+\sqrt{1-\alpha_t}\,\epsilon_t\\
&=\sqrt{\alpha_t\alpha_{t-1}}\,x_{t-2}+\underbrace{\sqrt{\alpha_t(1-\alpha_{t-1})}\,\epsilon_{t-1}+\sqrt{1-\alpha_t}\,\epsilon_t}_{\text{sum of two independent Gaussians}} .
\end{aligned}
$$

The sum of independent zero-mean Gaussians is a zero-mean Gaussian whose variance is the sum of the variances: $\alpha_t(1-\alpha_{t-1})+(1-\alpha_t)=1-\alpha_t\alpha_{t-1}$. So the noise term equals $\sqrt{1-\alpha_t\alpha_{t-1}}\,\bar\epsilon$ for a fresh $\bar\epsilon\sim\mathcal N(0,I)$. Repeating the argument down to $x_0$ (a one-line induction) gives

$$
\boxed{\,q(x_t\mid x_0)=\mathcal N\!\big(x_t;\ \sqrt{\bar\alpha_t}\,x_0,\ (1-\bar\alpha_t)I\big),\qquad x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\epsilon,\ \ \epsilon\sim\mathcal N(0,I).\,}
$$
:::

Three things to appreciate:

1. **Training is cheap.** To get a training example at a random step $t$ you do not simulate $t$ steps; you draw $\epsilon$ and apply one formula. This is what makes diffusion training a plain supervised regression.
2. **$\bar\alpha_t$ is the whole story.** The "signal" weight is $\sqrt{\bar\alpha_t}$ and the "noise" weight is $\sqrt{1-\bar\alpha_t}$. Their squares sum to one (variance preserving). At $\bar\alpha_t\approx1$ you see almost clean data; at $\bar\alpha_t\approx0$ almost pure noise. Many papers drop $\beta_t$ entirely and define a model directly through $\bar\alpha_t$.
3. **Signal-to-noise ratio.** $\mathrm{SNR}(t)=\bar\alpha_t/(1-\bar\alpha_t)$ is the ratio of signal power to noise power. It decreases monotonically in $t$. Its logarithm will turn out to be the natural "time" variable ([later](#snr-weighting)).

The NumPy code below builds the original linear schedule, checks the closed form against explicitly simulating $t$ steps, and prints how fast the signal dies.

```python:forward_closed_form.py
import numpy as np

rng = np.random.default_rng(0)
T = 1000
betas = np.linspace(1e-4, 0.02, T)           # linear schedule from Ho et al. (2020)
alphas = 1.0 - betas
alpha_bar = np.cumprod(alphas)               # alpha_bar[t-1] is \bar{alpha}_t

# 1) closed form vs explicit simulation, for a 1-D data point x0 = 2.0 at t = 300
x0, t, n = 2.0, 300, 200_000
x = np.full(n, x0)
for s in range(t):                           # t explicit noising steps
    x = np.sqrt(alphas[s]) * x + np.sqrt(betas[s]) * rng.standard_normal(n)
ab = alpha_bar[t - 1]
print(f"simulated   mean={x.mean():.4f} var={x.var():.4f}")
print(f"closed form mean={np.sqrt(ab) * x0:.4f} var={1 - ab:.4f}")

# 2) how quickly does the signal die?
for t in [1, 100, 250, 500, 750, 1000]:
    ab = alpha_bar[t - 1]
    print(f"t={t:4d}  alpha_bar={ab:.5f}  signal std={np.sqrt(ab):.3f}  noise std={np.sqrt(1-ab):.3f}  SNR={ab/(1-ab):.4g}")
```

```text
simulated   mean=1.2619 var=0.6051
closed form mean=1.2592 var=0.6036
t=   1  alpha_bar=0.99990  signal std=1.000  noise std=0.010  SNR=9999
t= 100  alpha_bar=0.89702  signal std=0.947  noise std=0.321  SNR=8.71
t= 250  alpha_bar=0.52409  signal std=0.724  noise std=0.690  SNR=1.101
t= 500  alpha_bar=0.07859  signal std=0.280  noise std=0.960  SNR=0.08529
t= 750  alpha_bar=0.00335  signal std=0.058  noise std=0.998  SNR=0.003362
t=1000  alpha_bar=0.00004  signal std=0.006  noise std=1.000  SNR=4.036e-05
```

{{demo: week13-forward-noising | Forward diffusion explorer: noise a 2-D point cloud, compare schedules | 520}}

## The reverse process and its posterior {#reverse}

Sampling needs the *reverse* conditionals $q(x_{t-1}\mid x_t)$. Applying Bayes' rule to them requires the marginal over all possible data $x_0$, which is exactly the intractable thing we want to learn. Two facts rescue us.

**Fact 1 (small steps).** When $\beta_t$ is small, the true reverse step $q(x_{t-1}\mid x_t)$ is approximately Gaussian (Feller, 1949, for diffusion processes; this is what makes the whole construction legitimate). So we model $p_\theta(x_{t-1}\mid x_t)=\mathcal N(x_{t-1};\mu_\theta(x_t,t),\Sigma_\theta(x_t,t))$.

**Fact 2 (conditioning on $x_0$ makes it exact).** If we *also* condition on the clean data point, the reverse step is tractable and exactly Gaussian. We will use it as the training target.

:::math Derivation: the posterior $q(x_{t-1}\mid x_t,x_0)$
By Bayes and the Markov property,
$$
q(x_{t-1}\mid x_t,x_0)=\frac{q(x_t\mid x_{t-1})\,q(x_{t-1}\mid x_0)}{q(x_t\mid x_0)}\ \propto\ q(x_t\mid x_{t-1})\,q(x_{t-1}\mid x_0).
$$
Both factors on the right are Gaussians in $x_{t-1}$, so the product is Gaussian. Collect the exponent (all terms in $x_{t-1}$, per coordinate):
$$
-\tfrac12\Big[\tfrac{(x_t-\sqrt{\alpha_t}x_{t-1})^2}{\beta_t}+\tfrac{(x_{t-1}-\sqrt{\bar\alpha_{t-1}}x_0)^2}{1-\bar\alpha_{t-1}}\Big]
=-\tfrac12\Big[\underbrace{\Big(\tfrac{\alpha_t}{\beta_t}+\tfrac1{1-\bar\alpha_{t-1}}\Big)}_{1/\tilde\beta_t}x_{t-1}^2-2\Big(\tfrac{\sqrt{\alpha_t}}{\beta_t}x_t+\tfrac{\sqrt{\bar\alpha_{t-1}}}{1-\bar\alpha_{t-1}}x_0\Big)x_{t-1}+\text{const}\Big].
$$
Matching to $-\tfrac1{2\tilde\beta}(x_{t-1}-\tilde\mu)^2$ gives the variance and mean
$$
\tilde\beta_t=\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\,\beta_t,\qquad
\tilde\mu_t(x_t,x_0)=\frac{\sqrt{\bar\alpha_{t-1}}\,\beta_t}{1-\bar\alpha_t}\,x_0+\frac{\sqrt{\alpha_t}\,(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}\,x_t .
$$
(The simplification of $1/\tilde\beta_t$ uses $\alpha_t(1-\bar\alpha_{t-1})+\beta_t=1-\bar\alpha_t$.) Now substitute $x_0=\big(x_t-\sqrt{1-\bar\alpha_t}\,\epsilon\big)/\sqrt{\bar\alpha_t}$ from the closed form; after algebra the $x_t$ coefficient collapses and
$$
\boxed{\ \tilde\mu_t=\frac1{\sqrt{\alpha_t}}\Big(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\epsilon\Big)\ }
$$
where $\epsilon$ is the noise that was used to produce $x_t$ from $x_0$.
:::

Read the boxed formula as a recipe: *to step backwards, take $x_t$, subtract the (scaled) noise that was added, and rescale.* The only thing the network needs to supply is that noise $\epsilon$. This is the origin of **$\epsilon$-prediction**.

Let us check the posterior coefficients numerically before building on them.

```python:posterior_check.py
import numpy as np
T = 1000
betas = np.linspace(1e-4, 0.02, T); alphas = 1 - betas
ab = np.cumprod(alphas); ab_prev = np.concatenate([[1.0], ab[:-1]])

t = 400 - 1                                   # 0-based index of step 400
rng = np.random.default_rng(1)
x0 = rng.standard_normal(5); eps = rng.standard_normal(5)
xt = np.sqrt(ab[t]) * x0 + np.sqrt(1 - ab[t]) * eps

# posterior mean from the (x0, xt) form
c0 = np.sqrt(ab_prev[t]) * betas[t] / (1 - ab[t])
ct = np.sqrt(alphas[t]) * (1 - ab_prev[t]) / (1 - ab[t])
mu_a = c0 * x0 + ct * xt
# posterior mean from the epsilon form
mu_b = (xt - betas[t] / np.sqrt(1 - ab[t]) * eps) / np.sqrt(alphas[t])
print(np.abs(mu_a - mu_b).max())              # ~1e-16: the two forms agree
print("posterior variance beta_tilde =", (1 - ab_prev[t]) / (1 - ab[t]) * betas[t], " vs beta_t =", betas[t])
```

## The variational bound and the simplified loss {#elbo}

Diffusion is a latent-variable model $p_\theta(x_0)=\int p_\theta(x_{0:T})\,dx_{1:T}$ whose latents $x_{1:T}$ have the same dimension as the data. The inference ("encoder") distribution is the fixed forward process $q(x_{1:T}\mid x_0)$. Exactly as for the VAE of [Week 12](week-12.html#elbo), the negative log-likelihood is bounded by the ELBO.

:::math Derivation: the diffusion ELBO
Let $p_\theta(x_{0:T})=p(x_T)\prod_{t=1}^Tp_\theta(x_{t-1}\mid x_t)$ with $p(x_T)=\mathcal N(0,I)$. Jensen's inequality gives
$$
-\log p_\theta(x_0)\le \E_{q(x_{1:T}\mid x_0)}\Big[-\log\frac{p_\theta(x_{0:T})}{q(x_{1:T}\mid x_0)}\Big]=:L .
$$
Insert the factorizations, and rewrite each forward factor with Bayes' rule using $q(x_t\mid x_{t-1})=q(x_t\mid x_{t-1},x_0)=\frac{q(x_{t-1}\mid x_t,x_0)\,q(x_t\mid x_0)}{q(x_{t-1}\mid x_0)}$ for $t\ge2$. The marginal ratios telescope, leaving
$$
L=\underbrace{\KL\!\big(q(x_T\mid x_0)\,\|\,p(x_T)\big)}_{L_T}
+\sum_{t=2}^{T}\underbrace{\E_q\,\KL\!\big(q(x_{t-1}\mid x_t,x_0)\,\|\,p_\theta(x_{t-1}\mid x_t)\big)}_{L_{t-1}}
\ \underbrace{-\ \E_q\log p_\theta(x_0\mid x_1)}_{L_0}.
$$
- $L_T$ has no parameters; it is tiny when $\bar\alpha_T\approx0$.
- $L_0$ is a reconstruction (decoder) term for the last step.
- Each $L_{t-1}$ is a KL between **two Gaussians**: the exact posterior $\mathcal N(\tilde\mu_t,\tilde\beta_tI)$ and the model $\mathcal N(\mu_\theta,\sigma_t^2I)$.
:::

If we fix the model variance to $\sigma_t^2$ (Ho et al. use $\sigma_t^2=\beta_t$ or $\tilde\beta_t$; the two behave nearly the same), the KL between equal-covariance Gaussians is a squared distance between means (compare the closed-form KL in [Week 12](week-12.html#gaussian-kl)):

$$
L_{t-1}=\E_{x_0,\epsilon}\Big[\frac1{2\sigma_t^2}\big\|\tilde\mu_t(x_t,x_0)-\mu_\theta(x_t,t)\big\|^2\Big]+\text{const}.
$$

Now *choose* the parameterization of $\mu_\theta$ to mirror $\tilde\mu_t$, with the network predicting the noise:

$$
\mu_\theta(x_t,t)=\frac1{\sqrt{\alpha_t}}\Big(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\epsilon_\theta(x_t,t)\Big)
\ \Longrightarrow\
L_{t-1}=\E_{x_0,\epsilon}\Big[\frac{\beta_t^2}{2\sigma_t^2\,\alpha_t\,(1-\bar\alpha_t)}\,\big\|\epsilon-\epsilon_\theta(\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\,\epsilon,\ t)\big\|^2\Big].
$$

The $x_t$ terms cancel inside $\tilde\mu_t-\mu_\theta$, leaving only the difference in the noise predictions. This is a **weighted** sum of denoising errors, with a $t$-dependent weight $\lambda_t=\beta_t^2/(2\sigma_t^2\alpha_t(1-\bar\alpha_t))$. Ho et al. found that **dropping the weights** and sampling $t$ uniformly works better in practice for sample quality:

$$
\boxed{\ \mathcal L_\text{simple}(\theta)=\E_{x_0\sim p_\text{data},\ t\sim\mathcal U\{1..T\},\ \epsilon\sim\mathcal N(0,I)}\Big[\big\|\epsilon-\epsilon_\theta\big(\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\epsilon,\ t\big)\big\|^2\Big]\ }
$$

:::note What the simplification costs
The true ELBO weights are large at small $t$ (low noise) and small at large $t$. $\mathcal L_\text{simple}$ *down-weights* the easy, low-noise steps relative to the ELBO, which frees capacity for the harder high-noise denoising tasks that determine global structure. The price is that $\mathcal L_\text{simple}$ is **not** a bound on the likelihood: models trained with it have noticeably worse log-likelihoods (in bits per dimension) than models trained with the true bound, even though the samples look better. Nichol and Dhariwal (2021) recover good likelihoods by also learning the variances $\Sigma_\theta$ and adding a small weight on the true bound ("hybrid loss"); Kingma et al. (2021, Variational Diffusion Models) learn the schedule and optimize the bound directly.
:::

The training algorithm is therefore four lines: sample a data point, a random step $t$ and noise $\epsilon$; build $x_t$ with the closed form; predict the noise; take a gradient step on the squared error. Sampling (Ho et al., Algorithm 2) starts from $x_T\sim\mathcal N(0,I)$ and for $t=T,\dots,1$ draws

$$
x_{t-1}=\frac1{\sqrt{\alpha_t}}\Big(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\epsilon_\theta(x_t,t)\Big)+\sigma_t z,\qquad z\sim\mathcal N(0,I)\ \ (z=0\text{ at }t=1).
$$

This is the **ancestral sampler**: a learned deterministic drift followed by injected noise. It needs $T$ sequential network evaluations (1000 in the original), the main cost of diffusion.

### Parameterizations: epsilon, x0, and v {#parameterizations}

Because $x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon$ is one linear equation in two unknowns, predicting $\epsilon$ is equivalent to predicting $x_0$ **given $x_t$ and $t$**:

$$
\hat x_0=\frac{x_t-\sqrt{1-\bar\alpha_t}\,\hat\epsilon}{\sqrt{\bar\alpha_t}} .
$$

So there are several "equivalent" prediction targets, and they differ only in the implicit **loss weighting** and in numerical conditioning:

| Target | Network output | Best when | Weakness |
| --- | --- | --- | --- |
| $\epsilon$ | the added noise | moderate to low noise; the DDPM default | At $t\to T$, $\hat x_0$ divides by $\sqrt{\bar\alpha_t}\approx0$, amplifying errors |
| $x_0$ | the clean data | very high noise | Blurry predictions at low noise; weak signal for fine detail |
| $v$ | $v=\sqrt{\bar\alpha_t}\,\epsilon-\sqrt{1-\bar\alpha_t}\,x_0$ | all noise levels; few-step distillation | Slightly less intuitive |

**v-prediction** (Salimans and Ho, 2022, in the progressive-distillation paper) is a rotation. In the $(x_0,\epsilon)$ plane, $x_t$ is a point on a circle of "angle" $\phi_t$ with $\cos\phi_t=\sqrt{\bar\alpha_t}$, $\sin\phi_t=\sqrt{1-\bar\alpha_t}$, and $v$ is the velocity of $x_t$ as $\phi$ increases ($v=\partial x_t/\partial\phi$). You can recover both ingredients by rotating back:

$$
\hat x_0=\sqrt{\bar\alpha_t}\,x_t-\sqrt{1-\bar\alpha_t}\,\hat v,\qquad \hat\epsilon=\sqrt{1-\bar\alpha_t}\,x_t+\sqrt{\bar\alpha_t}\,\hat v .
$$

(Check the first: $\sqrt{\bar\alpha}x_t-\sqrt{1-\bar\alpha}\,v=\bar\alpha x_0+\sqrt{\bar\alpha(1-\bar\alpha)}\epsilon-\sqrt{\bar\alpha(1-\bar\alpha)}\epsilon+(1-\bar\alpha)x_0=x_0$.) Its main virtue is that the target stays well scaled (unit variance) at *both* ends of the schedule, whereas $\epsilon$-prediction becomes ill-conditioned at the nearly-pure-noise end. The next code block implements the conversions.

```python:parameterizations.py
import torch

def to_x0(x_t, out, ab, kind):
    """Convert a network output to an x0 estimate. ab = alpha_bar_t, broadcastable to x_t."""
    if kind == "eps":
        return (x_t - (1 - ab).sqrt() * out) / ab.sqrt()
    if kind == "x0":
        return out
    if kind == "v":
        return ab.sqrt() * x_t - (1 - ab).sqrt() * out
    raise ValueError(kind)

def to_eps(x_t, out, ab, kind):
    if kind == "eps":
        return out
    if kind == "x0":                                  # solve x_t = sqrt(ab) x0 + sqrt(1-ab) eps for eps
        return (x_t - ab.sqrt() * out) / (1 - ab).sqrt()
    if kind == "v":
        return (1 - ab).sqrt() * x_t + ab.sqrt() * out
    raise ValueError(kind)

def training_target(x0, eps, ab, kind):
    return {"eps": eps, "x0": x0, "v": ab.sqrt() * eps - (1 - ab).sqrt() * x0}[kind]

# self-test: all conversions must be mutually consistent
torch.manual_seed(0)
x0, eps = torch.randn(4, 3, 8, 8), torch.randn(4, 3, 8, 8)
ab = torch.tensor([0.9, 0.5, 0.1, 0.01]).view(-1, 1, 1, 1)
x_t = ab.sqrt() * x0 + (1 - ab).sqrt() * eps
for kind in ["eps", "x0", "v"]:
    out = training_target(x0, eps, ab, kind)
    print(kind, (to_x0(x_t, out, ab, kind) - x0).abs().max().item(), (to_eps(x_t, out, ab, kind) - eps).abs().max().item())
```

### Loss weighting and the SNR {#snr-weighting}

Everything above can be restated as "what weight does each noise level get in the $x_0$-reconstruction error?". Using $\hat\epsilon-\epsilon=\sqrt{\mathrm{SNR}}\,(x_0-\hat x_0)$ (this follows from holding $x_t$ fixed in the closed form), we get:

$$
\|\epsilon-\hat\epsilon\|^2=\mathrm{SNR}(t)\,\|x_0-\hat x_0\|^2,\qquad
\|v-\hat v\|^2=\big(1+\mathrm{SNR}(t)\big)\,\|x_0-\hat x_0\|^2 .
$$

So the $\epsilon$-loss is a *SNR-weighted* $x_0$-loss, and the $v$-loss is the same plus a constant, which keeps a nonvanishing weight at high noise. The chart shows the weights.

```chart
{
  "type": "line",
  "caption": "Effective weight on the x0-reconstruction error, log10 scale, as a function of t for the linear schedule (computed from the closed-form alpha-bar). x0-prediction has weight 1 (log10 = 0). The epsilon-loss weight is SNR(t); the v-loss weight is 1 + SNR(t). At high noise (large t) epsilon-prediction nearly ignores x0 accuracy while v-prediction keeps weight about 1.",
  "height": 340,
  "data": { "labels": [0,50,100,150,200,250,300,350,400,450,500,550,600,650,700,750,800,850,900,950],
    "datasets": [
      {"label": "epsilon-loss: log10 SNR", "data": [4.0,1.5089,0.9313,0.5647,0.281,0.0372,-0.187,-0.4033,-0.6197,-0.8418,-1.0739,-1.3194,-1.5811,-1.8608,-2.1602,-2.48,-2.8211,-3.1837,-3.5681,-3.9745], "borderColor": "var(--c-blue)", "pointRadius": 0, "tension": 0.2},
      {"label": "v-loss: log10 (1 + SNR)", "data": [4.0,1.5222,0.9794,0.6694,0.4639,0.32,0.2175,0.1446,0.0934,0.0584,0.0352,0.0203,0.0112,0.0059,0.003,0.0014,0.0007,0.0003,0.0001,0.0], "borderColor": "var(--c-accent)", "pointRadius": 0, "tension": 0.2},
      {"label": "x0-loss: weight 1", "data": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], "borderColor": "var(--c-muted)", "borderDash": [6,4], "pointRadius": 0}
    ] },
  "options": { "scales": { "x": {"title": {"display": true, "text": "diffusion step t (linear schedule, T = 1000)"}}, "y": {"title": {"display": true, "text": "log10 weight on ||x0 - x0_hat||^2"}} } }
}
```

## Noise schedules {#schedules}

The schedule decides how the finite budget of $T$ steps is spent across noise levels. Two schedules dominate the literature.

**Linear** (Ho et al., 2020): $\beta_t$ increases linearly from $10^{-4}$ to $0.02$ with $T=1000$. This was tuned for $32\times32$ images.

**Cosine** (Nichol and Dhariwal, 2021): define $\bar\alpha_t$ directly,
$$
\bar\alpha_t=\frac{f(t)}{f(0)},\qquad f(t)=\cos^2\!\Big(\frac{t/T+s}{1+s}\cdot\frac\pi2\Big),\quad s=0.008,
$$
and clip each implied $\beta_t=1-\bar\alpha_t/\bar\alpha_{t-1}$ to at most $0.999$ to avoid singularities near $t=T$. The offset $s$ keeps $\beta_t$ from being too tiny at the start.

```chart
{
  "type": "line",
  "caption": "Signal fraction alpha-bar_t for the linear (beta from 1e-4 to 0.02) and cosine (s = 0.008) schedules with T = 1000, computed from the definitions above. The linear schedule destroys signal quickly: alpha-bar is about 0.08 at t = 500 and about 0.0005 by t = 850, so many late steps denoise inputs that are already near pure noise. The cosine schedule decays gently and spends more steps at intermediate noise.",
  "height": 340,
  "data": { "labels": [0,50,100,150,200,250,300,350,400,450,500,550,600,650,700,750,800,850,900,950,1000],
    "datasets": [
      {"label": "linear", "data": [1.0,0.971,0.897,0.7883,0.659,0.5241,0.3964,0.2852,0.1951,0.127,0.0786,0.0462,0.0259,0.0138,0.007,0.0034,0.0015,0.0007,0.0003,0.0001,0.0], "borderColor": "var(--c-blue)", "pointRadius": 0, "tension": 0.25},
      {"label": "cosine", "data": [0.9999,0.992,0.9721,0.9407,0.8987,0.847,0.7869,0.7199,0.6475,0.5715,0.4938,0.4163,0.3408,0.2692,0.2031,0.1443,0.094,0.0537,0.0241,0.0061,0.0], "borderColor": "var(--c-accent)", "pointRadius": 0, "tension": 0.25}
    ] },
  "options": { "scales": { "x": {"title": {"display": true, "text": "step t"}}, "y": {"title": {"display": true, "text": "alpha-bar_t (fraction of signal variance kept)"}} } }
}
```

A cleaner way to compare schedules is the **log-SNR** $\lambda_t=\log\frac{\bar\alpha_t}{1-\bar\alpha_t}$. Any schedule is a monotone curve $\lambda(t)$; what the model sees at a given noise level does not depend on $t$ itself, only on $\lambda$. Equal spacing in $t$ translates into a particular *density of training samples along the $\lambda$ axis*.

```chart
{
  "type": "line",
  "caption": "Log-SNR versus step for the two schedules (computed). The cosine schedule is close to linear in log-SNR over most of the range (a roughly constant drop per step), meaning each step covers a similar change of noise level; the linear-beta schedule falls steeply late, wasting steps at extreme noise (log-SNR below -6). Values at t = 10, 60, ..., 960.",
  "height": 340,
  "data": { "labels": [10,60,110,160,210,260,310,360,410,460,510,560,610,660,710,760,810,860,910,960],
    "datasets": [
      {"label": "linear", "data": [6.2667,3.1668,1.9703,1.1727,0.541,-0.0095,-0.5209,-1.018,-1.5177,-2.0325,-2.572,-3.1436,-3.7533,-4.4054,-5.103,-5.8482,-6.6424,-7.4864,-8.3806,-9.3254], "borderColor": "var(--c-blue)", "pointRadius": 0, "tension": 0.25},
      {"label": "cosine", "data": [7.3671,4.4957,3.3688,2.6362,2.0816,1.626,1.2311,0.8748,0.543,0.2252,-0.087,-0.4013,-0.7258,-1.0697,-1.4453,-1.8704,-2.3744,-3.0127,-3.9153,-5.5478], "borderColor": "var(--c-accent)", "pointRadius": 0, "tension": 0.25}
    ] },
  "options": { "scales": { "x": {"title": {"display": true, "text": "step t"}}, "y": {"title": {"display": true, "text": "log-SNR  = log(alpha-bar / (1 - alpha-bar))"}} } }
}
```

Practical consequences:

- **Resolution matters.** At higher resolution neighboring pixels are redundant, so the same noise variance destroys *less* information; high-resolution models need a schedule shifted toward more noise (a lower SNR at every $t$). Hoogeboom et al. (2023, "simple diffusion") and Stable Diffusion 3 both rescale the schedule with image size. Treat a schedule tuned for 32x32 as a starting point, not a law.
- **Zero terminal SNR.** Both schedules leave a small leftover signal at $t=T$ ($\bar\alpha_T>0$; for the cosine schedule it is clipped to avoid exactly zero). At sampling time the model is then started from pure noise that does not match what it saw at $t=T$, which can cause a bias toward medium brightness in generated images. Lin et al. (2024) analyze this and recommend enforcing $\bar\alpha_T=0$ together with $v$-prediction.
- **Schedules are part of the model.** Changing the schedule at sampling time without retraining is possible only through careful re-timing (see Karras et al. below), because the network was trained for a specific mapping from $t$ to noise level.
- **Karras et al. (2022, "EDM")** argue for abandoning discrete steps and designing the noise level $\sigma$, the loss weighting, the network preconditioning and the sampler separately; their recipe is a good example of how much of "diffusion quality" is in these design choices rather than in the core idea.

## Score matching and the denoising view {#score}

### The score function {#score-def}

The **score** of a density $p$ is the gradient of its log-density with respect to the *data*, not the parameters:
$$
s(x)=\nabla_x\log p(x).
$$
It is a vector field that points uphill on the log-density landscape; for a Gaussian $\mathcal N(\mu,\sigma^2I)$ it is $-(x-\mu)/\sigma^2$, an arrow toward the mean whose length grows with distance. Two properties make scores attractive for generative modeling. First, the score does not depend on the normalizing constant: $\nabla_x\log\big(\tilde p(x)/Z\big)=\nabla_x\log\tilde p(x)$, so you can model an unnormalized density without computing $Z$. Second, given a score you can *sample* by following it with noise (**Langevin dynamics**):
$$
x_{k+1}=x_k+\tfrac{\eta}{2}\,\nabla_x\log p(x_k)+\sqrt{\eta}\,z_k,\qquad z_k\sim\mathcal N(0,I),
$$
which under mild conditions converges to samples from $p$ as $\eta\to0$ and $k\to\infty$.

{{figure: week13-score-field | The score field of a Gaussian mixture at two noise levels. At small noise the arrows point sharply toward the nearest mode and vanish between modes (where the estimate is hard); at large noise the blobs merge and the field is smooth and global.}}

### Why one noise level is not enough {#multi-noise}

Estimating $\nabla_x\log p_\text{data}$ directly has two problems on real data. The data lie near a low-dimensional manifold, so the density is essentially zero (and its score undefined) off the manifold; and in the *low-density regions between modes* there are almost no training samples, so the score estimate there is poor, and that is exactly where Langevin chains start. Song and Ermon (2019) fixed both by **perturbing the data with Gaussian noise at many scales** $\sigma_1>\sigma_2>\dots>\sigma_L$ and learning one network conditioned on the noise level. Heavy noise fills space with probability mass and gives the network a signal everywhere; light noise restores the fine structure. Sampling then goes from the largest to the smallest noise level (**annealed Langevin dynamics**). This is already the structure of diffusion.

### Denoising score matching {#dsm}

We cannot regress on $\nabla\log p_\sigma(x)$ directly because we do not know it. Vincent (2011) showed that for a *Gaussian corruption kernel* the unknown quantity can be replaced by a known one.

:::math Derivation: denoising score matching (DSM)
Let $\tilde x=x+\sigma\epsilon$ with $x\sim p_\text{data}$, $\epsilon\sim\mathcal N(0,I)$, and $p_\sigma(\tilde x)=\int p_\text{data}(x)\,q_\sigma(\tilde x\mid x)\,dx$. The *explicit* score matching loss would be
$$
J_\text{ESM}(\theta)=\tfrac12\E_{\tilde x\sim p_\sigma}\big\|s_\theta(\tilde x)-\nabla_{\tilde x}\log p_\sigma(\tilde x)\big\|^2,
$$
whose target is unknown. The *denoising* version uses the known conditional score $\nabla_{\tilde x}\log q_\sigma(\tilde x\mid x)=-(\tilde x-x)/\sigma^2=-\epsilon/\sigma$:
$$
J_\text{DSM}(\theta)=\tfrac12\E_{x,\tilde x}\big\|s_\theta(\tilde x)-\nabla_{\tilde x}\log q_\sigma(\tilde x\mid x)\big\|^2 .
$$
**Claim:** $J_\text{ESM}$ and $J_\text{DSM}$ differ by a constant independent of $\theta$, so they have the same minimizer. Expand both squares. The terms $\|s_\theta\|^2$ coincide. For the cross terms,
$$
\E_{\tilde x\sim p_\sigma}\big\langle s_\theta(\tilde x),\nabla\log p_\sigma(\tilde x)\big\rangle=\int\langle s_\theta,\nabla p_\sigma\rangle\,d\tilde x=\int\!\!\int p_\text{data}(x)\langle s_\theta(\tilde x),\nabla_{\tilde x}q_\sigma(\tilde x\mid x)\rangle\,dx\,d\tilde x=\E_{x,\tilde x}\big\langle s_\theta(\tilde x),\nabla\log q_\sigma(\tilde x\mid x)\big\rangle,
$$
using $\nabla p_\sigma=\int p_\text{data}\nabla q_\sigma$ and $\nabla q=q\nabla\log q$. The remaining terms do not involve $\theta$. $\blacksquare$

The minimizer of $J_\text{DSM}$ is therefore $s^*(\tilde x)=\nabla\log p_\sigma(\tilde x)$, and equivalently (the conditional expectation of the known target) $s^*(\tilde x)=\E[-\epsilon/\sigma\mid\tilde x]$.
:::

### Epsilon-prediction is score estimation {#eps-is-score}

Now apply DSM to the diffusion kernel $q(x_t\mid x_0)=\mathcal N(\sqrt{\bar\alpha_t}x_0,(1-\bar\alpha_t)I)$. Its conditional score is
$$
\nabla_{x_t}\log q(x_t\mid x_0)=-\frac{x_t-\sqrt{\bar\alpha_t}x_0}{1-\bar\alpha_t}=-\frac{\epsilon}{\sqrt{1-\bar\alpha_t}} .
$$
So training a network $s_\theta(x_t,t)$ by DSM is the same as training $\epsilon_\theta$ by $\mathcal L_\text{simple}$, with
$$
\boxed{\ s_\theta(x_t,t)=-\frac{\epsilon_\theta(x_t,t)}{\sqrt{1-\bar\alpha_t}}\ }.
$$
In words: **the noise that was added points away from the data; the score points toward it.** The DDPM denoiser is a score estimator at every noise level, and the weights of the simplified loss correspond to the DSM weighting $\lambda(\sigma)\propto\sigma^2$ used by Song and Ermon. There is also a neat identity named after Tweedie (and Robbins): the posterior mean of the clean data given the noisy point is
$$
\E[x_0\mid x_t]=\frac{x_t+(1-\bar\alpha_t)\,\nabla_{x_t}\log p_t(x_t)}{\sqrt{\bar\alpha_t}},
$$
which is the $\hat x_0$ of [parameterizations](#parameterizations) once $\hat\epsilon=-\sqrt{1-\bar\alpha_t}\,s$ is substituted. The denoiser *is* the score, up to an affine change.

:::intuition What the network actually learns
At noise level $\sigma$ the best possible guess of the clean point is the *average* of all clean points that could have produced $x_t$, weighted by how likely each is. At high noise that average is a blurry blob near the data mean; at low noise it snaps to the nearest data point. The score field is "where that average sits, relative to me". Sampling is a long walk guided by this arrow field, starting where the arrows are smooth and global and ending where they are sharp and local.
:::

{{demo: week13-score-sampler | Score field and reverse process on a 2-D Gaussian mixture (exact score) | 560}}

## The SDE view and the probability-flow ODE {#sde}

The discrete chain with $T=1000$ is a time discretization of a continuous-time process. Song et al. (2021) took the limit $T\to\infty$.

### Forward SDE {#forward-sde}

A **stochastic differential equation** describes a process with a drift $f$ and a noise term driven by Brownian motion $W_t$:
$$
dx=f(x,t)\,dt+g(t)\,dW_t .
$$
The DDPM chain is the Euler-Maruyama discretization of the **variance-preserving (VP) SDE**
$$
dx=-\tfrac12\beta(t)\,x\,dt+\sqrt{\beta(t)}\,dW_t,
$$
where $\beta(t)$ is the continuous-time analogue of $\beta_t$ (so $\beta_t\approx\beta(t)\Delta t$). Its marginals are exactly the Gaussian form we derived, $x_t=\sqrt{\bar\alpha(t)}x_0+\sqrt{1-\bar\alpha(t)}\epsilon$ with $\bar\alpha(t)=\exp(-\int_0^t\beta(s)\,ds)$. The older score-network line (NCSN) corresponds to the **variance-exploding (VE) SDE** $dx=\sqrt{d\sigma^2(t)/dt}\,dW_t$, with no shrinking drift and noise that grows without bound.

### Reverse-time SDE {#reverse-sde}

A theorem of Anderson (1982) says that the time-reversal of this process is again a diffusion, with the same noise scale but a drift corrected by the score:
$$
dx=\Big[f(x,t)-g(t)^2\,\nabla_x\log p_t(x)\Big]dt+g(t)\,d\bar W_t,
$$
where time runs backward from $T$ to $0$ and $\bar W$ is a Brownian motion in reverse time. **This is the entire mathematical content of diffusion sampling**: knowing the score of every marginal $p_t$ gives you a generative process. A network trained by denoising score matching supplies $s_\theta\approx\nabla\log p_t$, and any SDE solver produces samples. The DDPM ancestral sampler is one particular discretization of the reverse VP-SDE.

### The probability-flow ODE {#pf-ode}

Remarkably, the same family of marginals $\{p_t\}$ is produced by a **deterministic** process:
$$
\frac{dx}{dt}=f(x,t)-\tfrac12\,g(t)^2\,\nabla_x\log p_t(x).
$$
The reason, in one line, is that the Fokker-Planck equation (which governs how the density $p_t$ evolves) only depends on the combination of drift and diffusion, and the term $\tfrac12g^2\nabla\log p_t$ can be moved between the "drift" and "diffusion" columns without changing $p_t$. The ODE has no randomness, so:

- Each noise sample $x_T$ maps to **exactly one** image $x_0$ (a smooth, invertible map): you can **encode** an image to noise by integrating forward. This gives meaningful latent interpolation and image editing (SDEdit, inversion).
- You can use **fast, high-order ODE solvers** (Runge-Kutta, multistep) instead of 1000 stochastic steps. This is the theoretical foundation for DDIM, DPM-Solver and the EDM samplers.
- With a neural score you can compute exact likelihoods via the instantaneous change-of-variables formula, as in continuous normalizing flows (see [flows](#flows)).

:::note Stochastic versus deterministic samplers
Both families target the same marginals when the score is exact, but they behave differently with an *imperfect* network. Stochastic samplers can correct earlier errors (each step re-randomizes) and often give slightly better quality at many steps; deterministic ODE samplers are much more efficient at few steps and give a one-to-one noise-to-image map. In practice you will see both: ancestral samplers for quality at 50-250 steps, deterministic solvers for 10-30 steps.
:::

## DDIM and fast sampling {#ddim}

Sampling with 1000 sequential network calls is the practical bottleneck: an image takes seconds to minutes. Song, Meng and Ermon (2021, **DDIM**, "Denoising Diffusion Implicit Models") observed that the DDPM *training objective* depends only on the marginals $q(x_t\mid x_0)$, not on the joint Markov structure. Therefore many different (non-Markovian) forward processes share the same marginals and the same trained network, and some of them admit *deterministic, much shorter* reverse chains.

:::math The DDIM update
Given the network output at a step $t$, form the clean estimate $\hat x_0=(x_t-\sqrt{1-\bar\alpha_t}\,\epsilon_\theta)/\sqrt{\bar\alpha_t}$. For any earlier step $s<t$ (not necessarily $t-1$) the DDIM update is
$$
x_s=\sqrt{\bar\alpha_s}\ \hat x_0+\sqrt{1-\bar\alpha_s-\sigma^2}\ \epsilon_\theta(x_t,t)+\sigma\,z,\qquad
\sigma=\eta\sqrt{\tfrac{1-\bar\alpha_s}{1-\bar\alpha_t}}\sqrt{1-\tfrac{\bar\alpha_t}{\bar\alpha_s}},
$$
with $z\sim\mathcal N(0,I)$. Read it as: *go to the predicted clean image, then re-noise it to level $s$ using a mix of the predicted noise direction and fresh noise.*
- $\eta=1$ recovers DDPM ancestral sampling (when $s=t-1$).
- $\eta=0$ gives $\sigma=0$: a **deterministic** map, an Euler discretization of the probability-flow ODE written in the variable $x/\sqrt{\bar\alpha}$ versus $\sqrt{(1-\bar\alpha)/\bar\alpha}$.
- Any decreasing subsequence $T=\tau_S>\dots>\tau_0=0$ can be used: $S=20$ or $50$ network evaluations instead of 1000.
:::

The code below implements both samplers for a one-dimensional two-Gaussian mixture where the optimal noise predictor is known in closed form, so any error you see comes from the *sampler*, not from a learned network. It produces the numbers in the next chart.

```python:ddim_toy_exact_score.py
import numpy as np

# Data: 1-D mixture 0.5 N(-2, 0.5^2) + 0.5 N(+2, 0.5^2).  Linear schedule, T = 1000.
T = 1000
betas = np.linspace(1e-4, 0.02, T)
ab = np.concatenate([[1.0], np.cumprod(1 - betas)])          # ab[k] = alpha_bar at step k, ab[0] = 1
mus, sg, wts = np.array([-2.0, 2.0]), 0.5, np.array([0.5, 0.5])

def eps_star(x, k):
    """Exact optimal noise predictor E[eps | x_t] = -sqrt(1-ab) * score_k(x)."""
    a = ab[k]
    m = np.sqrt(a) * mus[None, :]                              # component means at noise level k
    v = a * sg**2 + 1 - a                                      # component variance at noise level k
    logp = np.log(wts)[None, :] - 0.5 * (x[:, None] - m) ** 2 / v - 0.5 * np.log(v)
    p = np.exp(logp - logp.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)   # responsibilities
    score = -(p * (x[:, None] - m) / v).sum(1)
    return -np.sqrt(1 - a) * score

def sample(n_steps, eta, N=20000, seed=1):
    rng = np.random.default_rng(seed)
    ks = np.round(np.linspace(T, 0, n_steps + 1)).astype(int)   # decreasing time grid
    x = rng.standard_normal(N)
    for t, s in zip(ks[:-1], ks[1:]):
        e = eps_star(x, t)
        x0 = (x - np.sqrt(1 - ab[t]) * e) / np.sqrt(ab[t])
        sig = eta * np.sqrt((1 - ab[s]) / (1 - ab[t])) * np.sqrt(1 - ab[t] / ab[s])
        x = np.sqrt(ab[s]) * x0 + np.sqrt(max(1 - ab[s] - sig**2, 0)) * e + sig * rng.standard_normal(N)
    return x

def w1_to_truth(x, rng=np.random.default_rng(0)):
    ref = np.sort(np.concatenate([rng.normal(-2, .5, 200000), rng.normal(2, .5, 200000)]))
    q = np.quantile(ref, np.linspace(0, 1, len(x) + 2)[1:-1])
    return np.abs(np.sort(x) - q).mean()                         # 1-D Wasserstein-1 distance

for n in [5, 20, 100, 1000]:
    print(n, "DDIM", round(w1_to_truth(sample(n, 0.0)), 4), "DDPM", round(w1_to_truth(sample(n, 1.0)), 4))
```

```chart
{
  "type": "line",
  "caption": "Sample error (1-D Wasserstein-1 distance to the true two-mode mixture) versus the number of sampling steps for DDIM (eta = 0) and DDPM-style ancestral sampling (eta = 1), using the exact score so that all error comes from time discretization (computed with the NumPy code above; 20,000 samples). The floor of about 0.007 for DDPM is Monte Carlo error. DDIM wins at few steps and loses slightly at many steps in this run because it cannot correct the small mismatch between N(0, 1) and the schedule's final distribution (alpha-bar_T = 4e-5).",
  "height": 340,
  "data": { "labels": ["2","3","5","10","20","50","100","250","1000"],
    "datasets": [
      {"label": "DDIM (eta = 0)", "data": [1.1814,0.629,0.2785,0.1381,0.0745,0.0379,0.0282,0.0231,0.021], "borderColor": "var(--c-accent)", "tension": 0.2},
      {"label": "DDPM (eta = 1)", "data": [1.189,0.7314,0.3521,0.1498,0.09,0.0378,0.031,0.0315,0.0071], "borderColor": "var(--c-blue)", "tension": 0.2}
    ] },
  "options": { "scales": { "x": {"title": {"display": true, "text": "number of sampling steps"}}, "y": {"type": "logarithmic", "title": {"display": true, "text": "Wasserstein-1 error (log scale)"}} } }
}
```

The chart shows the main trend of the field: a 20-50 step deterministic sampler already reaches an error within a factor of a few of the 1000-step floor. In real image models the picture is similar, and later solvers (DPM-Solver, Lu et al. 2022; the EDM Heun sampler, Karras et al. 2022; UniPC) push good samples down to 10-20 evaluations by using higher-order ODE solvers, because the probability-flow trajectory is smooth.

### Beyond solvers: distillation {#distillation}

Solver improvements reduce the number of steps to roughly 10-20. To go lower, you change the *model*. **Progressive distillation** (Salimans and Ho, 2022) trains a student to match two teacher steps with one, halving the step count repeatedly (8, 4, 2 steps). **Consistency models** (Song et al., 2023) train a network that maps *any* point on a probability-flow trajectory to the trajectory's endpoint, enabling one- or two-step generation. **Adversarial distillation** variants (for example, SDXL-Turbo, 2023) add a GAN loss to the few-step student. Distilled models trade some diversity and flexibility (for instance, worse behavior under strong guidance) for speed. The distinction to remember: *solvers* change how a fixed network is used; *distillation* changes the network.

### Flow matching and rectified flows {#flow-matching}

A closely related recipe, **flow matching** (Lipman et al., 2023) and **rectified flow** (Liu et al., 2023), trains a network to predict a *velocity field* along a straight-line interpolation $x_t=(1-t)\,x_0+t\,\epsilon$ between data and noise, with the regression target $\epsilon-x_0$. Sampling integrates an ODE; no SDE or variance schedule is needed. It is a $v$-prediction-like objective with a different path, and it underlies several recent large models (for instance Stable Diffusion 3 uses a rectified-flow formulation). Everything in this week transfers: only the interpolation path and the time parameterization change.

## Diffusion as a hierarchical VAE {#vae-link}

{{figure: week13-vae-hierarchy | Left: a standard VAE with one bottlenecked latent and a learned encoder. Right: diffusion as a hierarchical VAE whose latents $x_1,\dots,x_T$ are noisy copies of the data and whose encoder is fixed.}}

The ELBO we derived above is formally the ELBO of a **Markovian hierarchical VAE**: the latents are $z_t=x_t$, the encoder $q(x_{1:T}\mid x_0)$ is the forward process, the decoder chain is $p_\theta(x_{t-1}\mid x_t)$. Comparing with [Week 12](week-12.html#vae):

| | Standard VAE | Diffusion model |
| --- | --- | --- |
| Latent dimension | much smaller than the data | the same as the data, $T$ copies |
| Encoder | learned $q_\phi(z\mid x)$ | fixed, Gaussian noising, no parameters |
| Decoder | one network call | $T$ network calls, each a small denoising step |
| Posterior collapse | a real risk | impossible: the latents are fixed noisy copies of the input |
| KL term | per-sample KL to the prior, trades off with reconstruction | sum of $T$ KLs between Gaussians, each a denoising regression |
| Blurriness | mean-prediction of one big step is blurry | each step is tiny, so predicting the mean is accurate |

This explains why diffusion succeeds where a one-step Gaussian decoder fails. The blurriness of a VAE comes from asking a unimodal Gaussian decoder to capture a multimodal $p(x\mid z)$. In diffusion each *step* really is nearly Gaussian (a small perturbation of $x_t$), so the Gaussian assumption is nearly true, and multimodality is resolved gradually across steps. It also explains why the objective is stable: with the encoder frozen, there is no adversary and no joint optimization of encoder and decoder. The cost is sampling speed: instead of one decoder pass you need many. A recurring theme is to combine the two: a VAE compresses images into a short, perceptually meaningful code, and diffusion models that code, which is exactly latent diffusion ([below](#latent-diffusion)).

## Conditioning and guidance {#guidance}

An unconditional sampler is a curiosity; useful systems let you say *what* to generate: a class label, a text prompt, a sketch, a low-resolution image. Mathematically we want $p(x\mid c)$. The score decomposes by Bayes' rule:

$$
\nabla_{x_t}\log p_t(x_t\mid c)=\nabla_{x_t}\log p_t(x_t)+\nabla_{x_t}\log p_t(c\mid x_t).
$$

The first term is the unconditional score we already have; the second pushes the sample toward regions where the condition $c$ is likely.

### Classifier guidance {#classifier-guidance}

Dhariwal and Nichol (2021) trained a separate *noisy-image classifier* $p_\phi(c\mid x_t)$ (a classifier trained on images corrupted at every noise level) and added its gradient to the sampler, scaled by a **guidance scale** $w$:
$$
\tilde s(x_t,c)=\nabla\log p(x_t)+w\,\nabla_{x_t}\log p_\phi(c\mid x_t)
\quad\Longleftrightarrow\quad
\tilde\epsilon=\epsilon_\theta(x_t,t)-w\sqrt{1-\bar\alpha_t}\,\nabla_{x_t}\log p_\phi(c\mid x_t).
$$
With $w=1$ this samples from the true conditional. With $w>1$ it samples from the sharpened distribution $p(x)\,p(c\mid x)^w$, which concentrates on points the classifier is very sure about. Their "ADM-G" model, with guidance, outperformed the best GANs of the time on ImageNet (FID), the paper's title ("Diffusion Models Beat GANs on Image Synthesis") being a landmark. The cost: you need to train and run a second network on noisy images, the classifier gradient can be adversarial (the sample may fool the classifier rather than look like class $c$), and it only works for conditions you can classify.

### Classifier-free guidance {#cfg}

Ho and Salimans (2022) removed the classifier. The trick is to train **one** network on both the conditional and the unconditional task by randomly replacing the condition with a null token $\varnothing$ (typically 10-20% of the time), so that $\epsilon_\theta(x_t,t,c)$ and $\epsilon_\theta(x_t,t,\varnothing)$ are two heads of one set of weights. Then the classifier gradient is *implicit*.

:::math Derivation: classifier-free guidance
Rearrange Bayes' rule for the score: $\nabla\log p(c\mid x_t)=\nabla\log p(x_t\mid c)-\nabla\log p(x_t)$. Using $\epsilon=-\sqrt{1-\bar\alpha_t}\,s$, with $\epsilon_c=\epsilon_\theta(x_t,t,c)$ and $\epsilon_\varnothing=\epsilon_\theta(x_t,t,\varnothing)$,
$$
\nabla\log p(c\mid x_t)\ \approx\ -\frac{1}{\sqrt{1-\bar\alpha_t}}\big(\epsilon_c-\epsilon_\varnothing\big).
$$
Plug into the classifier-guided score $\nabla\log p(x_t)+w\,\nabla\log p(c\mid x_t)$ and convert back to a noise prediction:
$$
\boxed{\ \tilde\epsilon=\epsilon_\varnothing+w\,\big(\epsilon_c-\epsilon_\varnothing\big)=(1-w)\,\epsilon_\varnothing+w\,\epsilon_c\ }
$$
The effective density being sampled is $\tilde p(x\mid c)\propto p(x\mid c)^{\,w}\,p(x)^{\,1-w}\propto p(x)\,p(c\mid x)^{w}$.
:::

{{figure: week13-cfg | Classifier-free guidance at one sampling step: two forward passes of the same network, one with the real condition and one with the null condition, extrapolated along their difference.}}

**Convention warning.** Ho and Salimans write the guided prediction as $(1+w)\,\epsilon_c-w\,\epsilon_\varnothing$, so their $w=0$ is the plain conditional model. The formula above uses $w_\text{here}=1+w_\text{theirs}$. Libraries are not consistent either: the "guidance_scale" in Hugging Face diffusers equals $w_\text{here}$ (1 means no guidance; typical text-to-image values are 5 to 9). Always check which convention a codebase uses.

**What guidance does.** The difference $\epsilon_c-\epsilon_\varnothing$ is "the part of the denoising direction that is *due to the condition*". Multiplying it by $w>1$ exaggerates the prompt. The effect is *not* just sharper images but a distribution change: samples move toward high-likelihood, prototypical examples of the class, away from the boundary with other classes. You get higher fidelity and prompt adherence at the price of diversity and, at very high $w$, over-saturated, high-contrast, artifact-prone images. The next demo computes the idealized guided density exactly for two Gaussian classes, and the chart summarizes a sampling experiment.

{{demo: week13-cfg-explorer | Classifier-free guidance on two Gaussian classes: sharper and less diverse as w grows | 480}}

```chart
{
  "type": "line",
  "caption": "Classifier-free guidance on a 1-D two-class problem (classes N(-1, 0.8^2) and N(+1, 0.8^2), condition = class B), sampled with 100-step DDIM and the exact conditional and unconditional noise predictors (computed with the NumPy code in the exercise on guidance). Mean moves from 0.0 (unconditional) through 1.0 (true class mean at w = 1) to 2.8 at w = 8, while the sample standard deviation shrinks from 1.26 to 0.38: guidance overshoots the class mean and collapses diversity.",
  "height": 340,
  "data": { "labels": ["0","0.5","1","2","3","4","6","8"],
    "datasets": [
      {"label": "sample mean", "data": [0.01,0.586,1.0,1.506,1.828,2.078,2.472,2.79], "borderColor": "var(--c-accent)", "tension": 0.25, "yAxisID": "y"},
      {"label": "sample std", "data": [1.26,0.998,0.782,0.57,0.493,0.452,0.408,0.384], "borderColor": "var(--c-blue)", "tension": 0.25, "yAxisID": "y"}
    ] },
  "options": { "scales": { "x": {"title": {"display": true, "text": "guidance scale w"}}, "y": {"title": {"display": true, "text": "mean / std of generated samples"}} } }
}
```

:::warning The price of high guidance
Because the guided density is *not* the true conditional, high $w$ gives non-physical samples: with the same $w$ the guided distribution has a different mean and a smaller variance than the data, so metrics like FID typically follow a U-shape (best at moderate $w$) while Inception Score and CLIP score keep improving. Remedies in the literature include *guidance intervals* (apply guidance only in the middle of the trajectory), *dynamic thresholding* (Imagen, Saharia et al. 2022: clip and rescale the predicted $x_0$ to avoid saturation) and rescaling the guided prediction to match the standard deviation of $\epsilon_c$.
:::

### The training change is tiny {#cfg-training}

Classifier-free guidance requires almost no extra code in training: randomly drop the condition. The following fragment shows the two places it touches a codebase (training with dropout, sampling with two passes).

```python:cfg_core.py
import torch

NULL_LABEL = 10                                      # MNIST has classes 0..9; index 10 means "no condition"

def drop_condition(y, p_uncond=0.1):
    """Training: replace labels with the null token with probability p_uncond."""
    mask = torch.rand(y.shape[0], device=y.device) < p_uncond
    return torch.where(mask, torch.full_like(y, NULL_LABEL), y)

@torch.no_grad()
def guided_eps(model, x_t, t, y, w):
    """Sampling: two passes of the same network, then extrapolate (w = 1: plain conditional model)."""
    if w == 1.0:
        return model(x_t, t, y)
    # batch the two passes together for efficiency
    x2, t2 = torch.cat([x_t, x_t]), torch.cat([t, t])
    y2 = torch.cat([y, torch.full_like(y, NULL_LABEL)])
    eps_c, eps_u = model(x2, t2, y2).chunk(2)
    return eps_u + w * (eps_c - eps_u)
```

### Text-to-image conditioning with cross-attention {#cross-attention}

For text, the condition is a *sequence* of token embeddings, not a single label. Systems encode the prompt with a frozen language model (CLIP's text encoder in Stable Diffusion 1.x and 2.x, [Week 11](week-11.html#clip); T5 in Imagen; combinations in later systems) into a matrix $C\in\R^{L\times d_c}$ (for example $L=77$ tokens). The denoiser injects it through **cross-attention** layers ([Week 9](week-09.html#cross-attention)): the queries come from the image features, the keys and values from the text tokens,
$$
\mathrm{CrossAttn}(F,C)=\softmax\!\Big(\frac{(FW_Q)(CW_K)^\top}{\sqrt{d}}\Big)\,(CW_V),
$$
with $F\in\R^{N\times d_f}$ the $N$ spatial positions of a feature map. Each spatial position decides which words to read from, and the attention maps are interpretable: the map for the word "dog" highlights where the dog is (the basis of prompt-to-prompt editing). Timestep information is injected separately (see the architecture section). Classifier-free guidance is used for text exactly as above, with the empty prompt as the null condition; *negative prompts* simply replace $\epsilon_\varnothing$ by the prediction for the negative text, so that the extrapolation moves away from the undesired concept.

```python:cross_attention.py
import torch, torch.nn as nn, torch.nn.functional as F

class CrossAttention(nn.Module):
    """Image tokens (queries) attend to text tokens (keys/values).
    x: (B, N, d_model)  image features, N = H*W positions
    ctx: (B, L, d_ctx)  text token embeddings, L tokens
    """
    def __init__(self, d_model, d_ctx, n_heads=8):
        super().__init__()
        assert d_model % n_heads == 0
        self.h, self.dh = n_heads, d_model // n_heads
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_ctx, d_model, bias=False)
        self.v = nn.Linear(d_ctx, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x, ctx, ctx_mask=None):
        B, N, _ = x.shape
        L = ctx.shape[1]
        q = self.q(x).view(B, N, self.h, self.dh).transpose(1, 2)    # (B, h, N, dh)
        k = self.k(ctx).view(B, L, self.h, self.dh).transpose(1, 2)  # (B, h, L, dh)
        v = self.v(ctx).view(B, L, self.h, self.dh).transpose(1, 2)  # (B, h, L, dh)
        attn_mask = None if ctx_mask is None else ctx_mask[:, None, None, :]   # True = keep
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)       # softmax(qk^T/sqrt(dh)) v
        o = o.transpose(1, 2).reshape(B, N, -1)
        return self.out(o)

x = torch.randn(2, 32 * 32, 320)       # a 32x32 feature map with 320 channels, flattened
ctx = torch.randn(2, 77, 768)          # 77 text tokens from a 768-d text encoder
print(CrossAttention(320, 768)(x, ctx).shape)    # torch.Size([2, 1024, 320])
```

Other conditioning mechanisms are variations on the theme: **concatenating** a low-resolution or masked image to the input channels (super-resolution, inpainting); **adaLN/FiLM** modulation of normalization layers by a global embedding (class labels, timestep; see DiT below); and **adapters** such as ControlNet (Zhang et al., 2023), which add a trainable copy of the encoder that injects spatial controls (edges, depth, pose) into a frozen model, and LoRA ([Week 10](week-10.html)) fine-tuning for subjects and styles.

## Latent diffusion {#latent-diffusion}

Running a denoiser at full pixel resolution is expensive: a $512\times512\times3$ image has 786,432 values and the network must be evaluated dozens of times. Rombach et al. (2022, **Latent Diffusion Models**, the basis of Stable Diffusion) observed that most pixel bits encode imperceptible high-frequency detail. Their recipe has two stages:

1. **Perceptual compression.** Train an autoencoder (a VAE with a small KL penalty and/or a VQ layer, plus an LPIPS perceptual loss and a patch GAN loss; see [Week 12](week-12.html#vq-vae)) that maps an image $x\in\R^{3\times512\times512}$ to a latent $z=E(x)\in\R^{4\times64\times64}$ with spatial downsampling factor $f=8$ and decodes it back with $D$.
2. **Latent diffusion.** Train the diffusion model on $z$ instead of $x$ (the loss is the same $\mathcal L_\text{simple}$, with $z_0=E(x_0)$). Generation: sample $z_T$, denoise to $z_0$, decode $x=D(z_0)$.

{{figure: week13-latent-diffusion | Latent diffusion. A frozen text encoder produces tokens that condition the denoiser through cross-attention; denoising runs in the compressed latent space, and a VAE decoder maps the final latent back to pixels.}}

The saving is large: $3\cdot512\cdot512=786{,}432$ pixel values versus $4\cdot64\cdot64=16{,}384$ latent values, a $48\times$ reduction, and self-attention cost (quadratic in positions) falls by a factor of $64^2=4096$ between $512^2$ pixel positions and $64^2$ latent positions. The autoencoder is trained once and reused. Two caveats: the autoencoder's reconstruction ceiling caps the quality (small text and fine facial detail are often lost; newer systems use 16-channel latents), and the latent scale must be normalized to roughly unit variance (Stable Diffusion multiplies latents by a fixed constant, 0.18215 for version 1) so that the noise schedule is meaningful.

:::note Why not just make the VAE strong and skip diffusion?
A VAE's decoder cannot model the global structure of a scene in one Gaussian step (previous section); diffusion handles global structure with a series of small steps. Latent diffusion splits the labor: the autoencoder removes *perceptually irrelevant* detail, diffusion models the *semantic* structure that remains. It is the same division of labor as VQ-GAN plus a Transformer prior, with a different prior.
:::

## Backbones: U-Net and DiT {#backbones}

The network $\epsilon_\theta(x_t,t,c)$ must map a tensor to a tensor of the same shape, using $t$ and $c$ as side information. Two families are common.

### The U-Net {#unet}

{{figure: week13-unet | A diffusion U-Net: a convolutional encoder-decoder with skip connections, ResBlocks conditioned on the timestep embedding, self-attention at low resolution, and cross-attention to text tokens.}}

The U-Net (Ronneberger et al., 2015, for biomedical segmentation) pairs a **downsampling path** that builds a multi-scale representation with an **upsampling path** that restores resolution, joined by **skip connections** that carry fine detail across. Diffusion U-Nets (Ho et al., 2020, building on PixelCNN++ and Wide-ResNet blocks) add:

- **Time conditioning.** $t$ (or $\log$-SNR) is mapped through a sinusoidal embedding (as in Transformer positional encodings, [Week 9](week-09.html#sinusoidal)) and an MLP, then added into every residual block (or used for FiLM scale-and-shift).
- **GroupNorm and SiLU** activations throughout, rather than BatchNorm, because the per-sample noise level varies and batch statistics would leak across samples.
- **Self-attention** at the low-resolution stages (for example 16x16 and 8x8) to capture global structure, and **cross-attention** blocks for text.
- **Output of the same shape as the input**, plus optional variance channels.

U-Nets have a strong inductive bias (locality, multi-scale), are efficient at resolutions up to 64x64 in latent space, and remain the backbone of many deployed systems.

### Diffusion Transformers (DiT) {#dit}

{{figure: week13-dit | A DiT block: latent patches become tokens, and the timestep and class embedding modulate each block through adaptive LayerNorm. Zero-initialized gates make every block start as an identity map.}}

Peebles and Xie (2023, "Scalable Diffusion Models with Transformers") replaced the U-Net by a plain Vision Transformer ([Week 11](week-11.html#vit)) on latent patches. The latent $z\in\R^{4\times32\times32}$ is cut into $p\times p$ patches ($p=2$ gives 256 tokens), linearly embedded, and given positional embeddings. The conditioning (timestep and class) is injected by **adaptive layer norm**: a small MLP maps the conditioning embedding to per-block scales and shifts $(\gamma,\beta)$ and a gate $\alpha$, so that a block computes $x\leftarrow x+\alpha_1\,\mathrm{Attn}\big(\gamma_1\cdot\mathrm{LN}(x)+\beta_1\big)$ and similarly for the MLP. In **adaLN-Zero** the gates are initialized to zero, so each block is initially the identity, which stabilizes training of deep stacks. The key empirical finding was a clean **scaling law**: FID improves smoothly as Gflops grow (by more tokens or a larger model), just as for language Transformers ([Week 10](week-10.html#scaling-laws)). Later systems (Stable Diffusion 3, with multimodal Transformer blocks that process text and image tokens jointly, and several video models) use Transformer backbones.

| | U-Net | DiT |
| --- | --- | --- |
| Inductive bias | convolutional, multi-scale, local | none beyond patches and position embeddings |
| Scaling | works, but architecture-specific tuning | smooth, predictable scaling in Gflops |
| Resolution handling | fully convolutional, any size | needs positional-embedding interpolation |
| Cost at high resolution | cheap in conv layers; attention only at low-res | quadratic attention in tokens |
| Conditioning | additive timestep, cross-attention for text | adaLN for global, joint or cross attention for text |
| Reuse of Transformer tooling | no | yes (FlashAttention, tensor parallelism, etc.) |

Neither is "better" in an absolute sense; a Transformer shines when data and compute are plentiful, and the U-Net's bias helps in the small-data regime of your lab. You will build a small U-Net below.

## Normalizing flows {#flows}

Diffusion's probability-flow ODE is one instance of a broader idea: build a generative model by *transforming* a simple density through an invertible map and track how density changes. This is the **normalizing flow** family, and unlike VAEs and GANs, flows give the *exact* log-likelihood.

### Change of variables {#change-of-variables}

Let $z\sim p_Z$ (typically $\mathcal N(0,I)$) and let $x=g_\theta(z)$ with $g_\theta$ invertible and differentiable, $f_\theta=g_\theta^{-1}$. Then
$$
\log p_X(x)=\log p_Z\big(f_\theta(x)\big)+\log\Big|\det\frac{\partial f_\theta(x)}{\partial x}\Big| .
$$
The derivation is one line from probability conservation, $p_X(x)\,|dx|=p_Z(z)\,|dz|$: a map that stretches space must lower the density by the same factor. A deep flow composes $K$ maps $f=f_K\circ\dots\circ f_1$, and the log-determinants add up. Training maximizes the exact log-likelihood of the data; sampling draws $z$ and applies $g_\theta=f_1^{-1}\circ\dots\circ f_K^{-1}$.

For a general $D\times D$ Jacobian the determinant costs $O(D^3)$ and the inverse is not available, so a flow layer must be designed so that both are cheap. Invertibility plus an easy determinant is the design constraint that limits flows' architectural freedom: you cannot use arbitrary networks or bottlenecks (the dimension must be preserved).

### Coupling layers {#coupling}

{{figure: week13-coupling | An affine coupling layer: half of the coordinates pass through unchanged and parameterize an elementwise scale and shift of the other half. The Jacobian is triangular, so its log-determinant is a sum; the inverse needs no matrix inversion.}}

The influential trick (NICE, Dinh et al. 2015; **RealNVP**, Dinh et al. 2017) splits $x=(x_a,x_b)$ and sets
$$
y_a=x_a,\qquad y_b=x_b\odot\exp\big(s(x_a)\big)+t(x_a).
$$
The functions $s,t$ can be **arbitrary neural networks**, because they are only ever evaluated in the forward direction. The Jacobian is block-triangular,
$$
\frac{\partial y}{\partial x}=\begin{pmatrix}I&0\\ \partial y_b/\partial x_a&\mathrm{diag}(e^{s(x_a)})\end{pmatrix},\qquad
\log\Big|\det\frac{\partial y}{\partial x}\Big|=\sum_j s_j(x_a),
$$
and the inverse is $x_b=(y_b-t(y_a))\odot e^{-s(y_a)}$ with $x_a=y_a$. Stacking layers that alternate which half is transformed (checkerboard or channel masks for images) lets every coordinate be transformed. **Glow** (Kingma and Dhariwal, 2018) added invertible $1\times1$ convolutions and actnorm, and produced high-resolution faces. Here is a compact coupling-layer flow with exact likelihood on a 2-D toy problem.

```python:realnvp_coupling.py
import torch, torch.nn as nn

class AffineCoupling(nn.Module):
    """y_a = x_a;  y_b = x_b * exp(s(x_a)) + t(x_a).  `flip` swaps which half is conditioned on."""
    def __init__(self, dim, hidden=128, flip=False):
        super().__init__()
        self.flip, self.d = flip, dim // 2
        self.net = nn.Sequential(nn.Linear(self.d, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 2 * (dim - self.d)))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)   # start as the identity map

    def _split(self, x):
        a, b = x[:, :self.d], x[:, self.d:]
        return (b, a) if self.flip else (a, b)

    def forward(self, x):                        # data -> latent, returns (y, log|det J|)
        a, b = self._split(x)
        s, t = self.net(a).chunk(2, dim=1)
        s = torch.tanh(s)                        # bound the log-scale for numerical stability
        y_b = b * torch.exp(s) + t
        y = torch.cat([y_b, a], 1) if self.flip else torch.cat([a, y_b], 1)
        return y, s.sum(1)

    def inverse(self, y):                        # latent -> data
        a, y_b = self._split(y)
        s, t = self.net(a).chunk(2, dim=1)
        s = torch.tanh(s)
        b = (y_b - t) * torch.exp(-s)
        return torch.cat([b, a], 1) if self.flip else torch.cat([a, b], 1)

class Flow(nn.Module):
    def __init__(self, dim=2, n_layers=8):
        super().__init__()
        self.layers = nn.ModuleList([AffineCoupling(dim, flip=(i % 2 == 1)) for i in range(n_layers)])
        self.register_buffer("log_norm", torch.tensor(-0.5 * dim * torch.log(torch.tensor(2 * torch.pi)).item()))

    def log_prob(self, x):
        logdet = torch.zeros(x.shape[0], device=x.device)
        for layer in self.layers:
            x, ld = layer(x); logdet = logdet + ld
        return self.log_norm - 0.5 * (x ** 2).sum(1) + logdet     # log N(z; 0, I) + sum log|det|

    @torch.no_grad()
    def sample(self, n, dim=2):
        z = torch.randn(n, dim, device=self.log_norm.device)
        for layer in reversed(self.layers):
            z = layer.inverse(z)
        return z

flow = Flow()
x = torch.randn(256, 2)
y = x
for l in flow.layers: y, _ = l(y)
for l in reversed(flow.layers): y = l.inverse(y)
print("invertibility error:", (x - y).abs().max().item())      # ~0 (exactly 0 at init)
loss = -flow.log_prob(x).mean()                                 # train with Adam on this NLL
print(loss.item())
```

The strengths of flows are exact likelihood, a single-pass sampler and a meaningful latent space; the weaknesses are architectural rigidity (full dimension preserved, constrained layers), parameter inefficiency and sample quality that lagged GANs and diffusion. Their mathematical ideas live on: continuous-time flows ("neural ODEs", FFJORD) define the density change through $\partial_t\log p=-\nabla\cdot v$, and the probability-flow ODE of diffusion and the velocity field of flow matching are exactly such continuous flows, trained without needing exact Jacobians.

## Autoregressive models {#autoregressive}

The other exact-likelihood family writes the data distribution as a product of conditionals by the chain rule of probability,
$$
p_\theta(x)=\prod_{i=1}^{D}p_\theta\big(x_i\mid x_{<i}\big),
$$
and trains by maximizing $\sum_i\log p_\theta(x_i\mid x_{<i})$, i.e. ordinary cross-entropy (or a mixture likelihood for continuous values), with **teacher forcing**: all conditionals can be computed in one parallel pass if the architecture enforces causality. Sampling is sequential: draw $x_1$, then $x_2\mid x_1$, and so on. You have already met the dominant instance: the GPT-style language models of [Week 10](week-10.html). The question for images and audio is how to impose an ordering and a causal architecture.

### PixelRNN and PixelCNN {#pixelcnn}

van den Oord, Kalchbrenner and Kavukcuoglu (2016, PixelRNN/PixelCNN) order pixels in raster scan and model each pixel's three colour channels as a 256-way softmax conditioned on all previous pixels. A **masked convolution** zeroes the kernel weights that would look at the current pixel (mask type A, first layer) or at future pixels (mask type B, subsequent layers), so a standard convolutional stack computes all conditionals in parallel during training. Later versions (Gated PixelCNN, PixelCNN++) improved the receptive field (the plain stack has a blind spot to the upper right) and used a discretized logistic mixture output.

```python:masked_conv.py
import torch, torch.nn as nn

class MaskedConv2d(nn.Conv2d):
    """Mask 'A': centre pixel excluded (first layer). Mask 'B': centre included (later layers)."""
    def __init__(self, mask_type, *args, **kw):
        super().__init__(*args, **kw)
        assert mask_type in ("A", "B")
        kH, kW = self.kernel_size
        mask = torch.ones_like(self.weight)                  # (out, in, kH, kW)
        mask[:, :, kH // 2, kW // 2 + (mask_type == "B"):] = 0   # centre row: block right of (or at) centre
        mask[:, :, kH // 2 + 1:, :] = 0                          # all rows below centre
        self.register_buffer("mask", mask)

    def forward(self, x):
        return nn.functional.conv2d(x, self.weight * self.mask, self.bias,
                                    self.stride, self.padding, self.dilation, self.groups)

# check causality: the output at position (i, j) must not depend on pixels at or after (i, j) for mask A
net = nn.Sequential(MaskedConv2d("A", 1, 8, 7, padding=3), nn.ReLU(), MaskedConv2d("B", 8, 8, 3, padding=1))
x = torch.randn(1, 1, 8, 8, requires_grad=True)
net(x)[0, :, 3, 4].sum().backward()
g = x.grad[0, 0] != 0
print("influenced pixels are all before (3,4) in raster order:", bool(g.flatten()[3 * 8 + 4:].sum() == 0))
```

### WaveNet and audio {#wavenet}

For audio the sequence is a waveform at 16-24 kHz, so a model needs huge receptive fields. **WaveNet** (van den Oord et al., 2016) uses stacks of **dilated causal convolutions** with dilation $1,2,4,\dots,512$ repeated in blocks, so the receptive field grows exponentially with depth, gated activations, and a 256-way softmax over $\mu$-law-quantized samples. It set the quality bar for text-to-speech but was painfully slow to sample (one network pass per audio sample); fast variants (Parallel WaveNet via distillation, WaveRNN, GAN vocoders) followed, and modern speech systems combine neural codecs that compress audio into discrete tokens with Transformers or diffusion models.

### Transformers over tokens {#ar-transformers}

Today autoregressive generation of images and audio usually works on **discrete tokens** from a learned tokenizer ([VQ-VAE and VQ-GAN](week-12.html#vq-vae)) rather than raw pixels, and uses a Transformer to model the token sequence. Examples: Image GPT (Chen et al., 2020, a GPT over low-resolution pixels), DALL-E (Ramesh et al., 2021, a Transformer over text tokens followed by image tokens), Parti (Yu et al., 2022), and audio models over neural-codec tokens. The tokenization trades pixel-level exactness for sequences short enough to model, and it lets all the language-model machinery (scaling laws, KV-cache, speculative decoding, RLHF) apply. The strengths are exact likelihood, a stable cross-entropy objective, scalability and a unified interface for multiple modalities. The weaknesses are slow, sequential sampling, an *arbitrary* ordering imposed on inherently 2-D data and error accumulation (exposure bias) in long sequences. Hybrid models such as masked-token generators (MaskGIT, Chang et al., 2022) decode many tokens in parallel in a few iterations, resembling diffusion on discrete tokens.

## Comparing the generative families {#comparison}

You now have all the major families. The table compares them on the properties that decide which one to use. Ratings reflect the typical state of the art around 2023-2025 and are a guide, not a law: each family has been pushed beyond its stereotype by clever engineering.

| | VAE | GAN | Normalizing flow | Autoregressive | Diffusion / flow matching |
| --- | --- | --- | --- | --- | --- |
| Training objective | ELBO (reconstruction + KL) | minimax adversarial | exact log-likelihood | exact log-likelihood (cross-entropy) | denoising regression (variational bound or reweighted) |
| Exact likelihood | no (lower bound) | no | **yes** | **yes** | bound; exact via ODE (expensive) |
| Latent code | yes, low-dim, learned encoder | yes (no encoder) | yes, same dimension, exact inverse | none (tokens only) | noise, deterministic via ODE |
| Sample quality | blurry in pixel space | sharp, fast | moderate | high (with tokens and scale) | **state of the art** |
| Mode coverage | good | prone to collapse | good | good | good |
| Sampling cost | one pass | one pass | one pass | $D$ sequential passes (or tokens) | 10-1000 passes (fewer with distillation) |
| Training stability | stable | unstable | stable | stable | stable |
| Typical failure | blur, posterior collapse | mode collapse, instability | poor fine detail, rigid architecture | slow sampling, ordering artifacts | slow sampling, high-guidance artifacts |
| Main uses today | compression, latent spaces for other models | real-time synthesis, super-resolution, vocoders, distillation targets | density estimation, scientific sampling | language, code, audio tokens, multimodal LLMs | images, video, audio, 3D, molecules, robotics policies |

A few observations for choosing in practice. If you need **likelihoods** (compression, anomaly detection, scientific models), prefer autoregressive models or flows. If you need **real-time** synthesis, prefer a GAN or a distilled diffusion model. If you want the **best images and controllability**, a latent diffusion model with guidance is the default. If you want one **unified interface** with language, an autoregressive Transformer is attractive. And note the convergence: latent diffusion uses a VAE; many tokenizers are VQ-VAEs trained with GAN losses; the best few-step samplers are adversarially distilled; diffusion Transformers borrow everything from language models. The families are components, not rival camps.

## Applications {#applications}

Diffusion-style generators are now general-purpose density learners for continuous signals, and a short survey shows how the ideas of this week recombine.

- **Images.** Text-to-image systems (DALL-E 2, Imagen, Stable Diffusion, Midjourney, and many others) combine a text encoder, latent diffusion and classifier-free guidance. Conditioning variants cover inpainting and outpainting, super-resolution (cascaded or latent), image editing by inversion and re-sampling (SDEdit, prompt-to-prompt), and fine-grained control (ControlNet, IP-Adapter-style image prompts, LoRA personalization such as DreamBooth).
- **Video.** Video diffusion models add a time axis to the latent (3-D convolutions, temporal attention, or spacetime patches in a DiT), and are trained on enormous caption/video corpora; quality is limited by temporal consistency and the cost of long sequences, and many systems generate short clips and extend them.
- **Audio and music.** Diffusion operates on spectrograms or on neural-codec latents (text-to-music, text-to-audio, speech synthesis); autoregressive token models remain strong competitors, especially for speech.
- **Science and engineering.** Diffusion on 3-D coordinates and rotation groups for protein structure and molecule generation, conditional weather and trajectory generation, and **diffusion policies** for robot control (learning a distribution over action sequences). These connect directly to the geometric ideas of [Week 14](week-14.html).
- **Inverse problems.** Because a diffusion model is a learned prior, one can sample from the posterior $p(x\mid y)$ for a measurement $y=Ax+\text{noise}$ (denoising, deblurring, MRI reconstruction) by adding the measurement likelihood gradient to the score: the same decomposition as classifier guidance, with $\log p(y\mid x_t)$ in place of $\log p(c\mid x_t)$.

## Risks, provenance and watermarking {#risks}

High-quality, cheap generation changes the information environment, and responsible practitioners need to understand the technical facts as well as the ethics.

### What can go wrong {#harms}

- **Deepfakes and impersonation.** Synthetic faces, voices and video of real people enable fraud (voice cloning in phone scams), harassment (non-consensual intimate imagery, a major real-world harm that disproportionately affects women), and political disinformation. A related harm is the "liar's dividend": real evidence can be dismissed as fake.
- **Training data, copyright and consent.** Models are trained on web-scale images and text, many protected by copyright or containing personal data. The legal status varies by jurisdiction and is actively litigated; it is unresolved at the time of writing. Carlini et al. (2023) demonstrated that diffusion models can **memorize** and regurgitate individual training images, particularly ones duplicated many times in the data, which makes deduplication and membership audits a technical necessity, not only a legal one.
- **Bias and representation.** Models reproduce and can amplify stereotypes present in training captions (occupations, nationalities, skin tones). Guidance makes prototypical (hence stereotypical) outputs more likely: a diversity-versus-fidelity trade-off with social consequences.
- **Misuse of open weights.** Releasing weights enables research, auditing and local use, and also removes the ability to enforce safety filters. Safety classifiers, prompt filters, concept erasure and fine-tuning to remove a concept are partial mitigations; fine-tuning can usually undo them for open models.
- **Environmental and labor costs.** Training and serving cost energy, and data labeling and content moderation rely on human work that is often poorly paid.

### Detection, provenance and watermarking {#provenance}

Three technical ideas are used, each with different guarantees. Keep them separate.

1. **Passive detection** trains a classifier to tell real from generated images using statistical artifacts. It works against the generators it was trained on and degrades on new generators, compression and editing: it is an arms race, and false positives (accusing a real photo) are costly. Treat a detector score as weak evidence.
2. **Content provenance (C2PA).** The *Coalition for Content Provenance and Authenticity* defines an open standard, "Content Credentials", for attaching **cryptographically signed metadata** to media: who or what produced it (a camera, an editing tool, a generative model), and a record of edits. Verification checks the signature chain. It does not detect fakes. It lets a trustworthy producer *prove* origin, and its absence means "unknown", not "fake". Weaknesses: metadata is easily stripped by screenshots or re-encoding, so it needs platform support and complementary signals (soft binding to a fingerprint or watermark) to survive.
3. **Watermarking.** The generator embeds a signal in the output that survives common edits and can be detected by an algorithm. Post-hoc watermarks modify pixels or frequencies after generation. **Generation-time watermarks** are woven into the sampling process: for example, *Tree-Ring watermarks* (Wen et al., 2023) embed a pattern in the Fourier spectrum of the initial noise and detect it by inverting a (deterministic DDIM) sampler back to the noise; production systems such as Google DeepMind's SynthID embed imperceptible signals during generation. For language models, Kirchenbauer et al. (2023) bias token sampling toward a pseudo-random "green list". Limits: a watermark only identifies content from cooperating providers that embed it; adversaries can attempt removal (rewriting, regeneration, adding noise) or forgery; open-weight models can simply skip it.

:::note A practical stance
No single technique solves authenticity. Defence in depth combines: signed provenance for content that wants to be trusted, watermarks for large providers' outputs, platform policy and labeling, legal and procedural norms (for example, consent requirements for synthetic likenesses), and user education. As an engineer, document training-data sources, evaluate memorization, red-team the system, log provenance metadata by default, and be clear about which claims your safeguards do and do not support.
:::

## Lab: a class-conditional DDPM on MNIST {#lab}

You will build and train a small DDPM from scratch, conditioning on the digit label with classifier-free guidance, then run two experiments: **number of sampling steps** (DDPM vs DDIM) and **guidance scale**. A free Colab GPU trains it in 5-10 minutes (about 10 epochs); a CPU works but is slow. Fix a single device and seed at the top and keep every tensor on it.

### Part 1: the diffusion process {#lab-process}

```python:lab_diffusion.py
import math, torch, torch.nn as nn, torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

def extract(a, t, x_shape):
    """Pick a[t] for each batch element and reshape to broadcast against x of shape x_shape."""
    out = a.gather(0, t)                                   # (B,)
    return out.view(-1, *([1] * (len(x_shape) - 1)))       # (B, 1, 1, 1)

class Diffusion:
    def __init__(self, T=1000, schedule="linear", device=device):
        self.T = T
        if schedule == "linear":
            betas = torch.linspace(1e-4, 0.02, T)
        elif schedule == "cosine":
            s = 0.008
            u = torch.arange(T + 1) / T
            f = torch.cos((u + s) / (1 + s) * math.pi / 2) ** 2
            ab = f / f[0]
            betas = (1 - ab[1:] / ab[:-1]).clamp(max=0.999)
        else:
            raise ValueError(schedule)
        betas = betas.double()
        alphas = 1.0 - betas
        ab = torch.cumprod(alphas, 0)
        ab_prev = torch.cat([torch.ones(1, dtype=torch.double), ab[:-1]])
        f32 = lambda v: v.float().to(device)
        # index i (0-based) corresponds to step t = i + 1 in the notation of the text
        self.betas, self.alphas, self.ab, self.ab_prev = f32(betas), f32(alphas), f32(ab), f32(ab_prev)
        self.post_var = f32(betas * (1 - ab_prev) / (1 - ab))      # beta_tilde_t

    def q_sample(self, x0, t, noise=None):
        """x_t = sqrt(ab) x0 + sqrt(1-ab) eps (the closed form)."""
        if noise is None:
            noise = torch.randn_like(x0)
        return extract(self.ab, t, x0.shape).sqrt() * x0 + extract(1 - self.ab, t, x0.shape).sqrt() * noise, noise

    def loss(self, model, x0, y):
        t = torch.randint(0, self.T, (x0.shape[0],), device=x0.device)
        x_t, eps = self.q_sample(x0, t)
        return F.mse_loss(model(x_t, t, y), eps)                    # L_simple

diff = Diffusion()
x0 = torch.randn(4, 1, 28, 28, device=device)
t = torch.tensor([0, 100, 500, 999], device=device)
x_t, eps = diff.q_sample(x0, t)
print("x_t shape:", x_t.shape, "| signal fraction sqrt(ab):", diff.ab[t].sqrt().tolist())
print("var of x_t at t=999:", x_t[-1].var().item())                 # ~ 1 (pure noise)
```

### Part 2: the network {#lab-network}

```python:lab_unet.py
import math, torch, torch.nn as nn, torch.nn.functional as F

class SinusoidalEmb(nn.Module):
    def __init__(self, dim):
        super().__init__(); self.dim = dim
    def forward(self, t):                                            # t: (B,) integer steps
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
        args = t.float()[:, None] * freqs[None, :]
        return torch.cat([args.sin(), args.cos()], dim=1)            # (B, dim)

class ResBlock(nn.Module):
    def __init__(self, c_in, c_out, emb_dim):
        super().__init__()
        self.n1 = nn.GroupNorm(8, c_in);  self.c1 = nn.Conv2d(c_in, c_out, 3, padding=1)
        self.emb = nn.Linear(emb_dim, c_out)
        self.n2 = nn.GroupNorm(8, c_out); self.c2 = nn.Conv2d(c_out, c_out, 3, padding=1)
        self.skip = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()
    def forward(self, x, emb):
        h = self.c1(F.silu(self.n1(x)))
        h = h + self.emb(F.silu(emb))[:, :, None, None]              # inject timestep + label
        h = self.c2(F.silu(self.n2(h)))
        return h + self.skip(x)

class SelfAttention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.norm = nn.GroupNorm(8, c); self.qkv = nn.Conv2d(c, 3 * c, 1); self.proj = nn.Conv2d(c, c, 1)
    def forward(self, x):
        B, C, H, W = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(B, 3, C, H * W).transpose(2, 3).unbind(1)   # each (B, HW, C)
        o = F.scaled_dot_product_attention(q[:, None], k[:, None], v[:, None])[:, 0]         # (B, HW, C)
        return x + self.proj(o.transpose(1, 2).reshape(B, C, H, W))

class UNet(nn.Module):
    """28x28 -> 14x14 -> 7x7 -> back. Input channel 1. Labels 0..9 plus null label 10."""
    def __init__(self, ch=64, n_classes=10):
        super().__init__()
        emb = ch * 4
        self.t_mlp = nn.Sequential(SinusoidalEmb(ch), nn.Linear(ch, emb), nn.SiLU(), nn.Linear(emb, emb))
        self.y_emb = nn.Embedding(n_classes + 1, emb)                 # last index = null condition
        self.inc = nn.Conv2d(1, ch, 3, padding=1)
        self.d1 = ResBlock(ch, ch, emb);          self.ds1 = nn.Conv2d(ch, ch, 3, stride=2, padding=1)         # 28 -> 14
        self.d2 = ResBlock(ch, 2 * ch, emb);      self.ds2 = nn.Conv2d(2 * ch, 2 * ch, 3, stride=2, padding=1)  # 14 -> 7
        self.m1, self.ma, self.m2 = ResBlock(2 * ch, 2 * ch, emb), SelfAttention(2 * ch), ResBlock(2 * ch, 2 * ch, emb)
        self.us1 = nn.Sequential(nn.Upsample(scale_factor=2), nn.Conv2d(2 * ch, 2 * ch, 3, padding=1))   # 7 -> 14
        self.u1 = ResBlock(4 * ch, 2 * ch, emb)                       # concat with skip from d2
        self.us2 = nn.Sequential(nn.Upsample(scale_factor=2), nn.Conv2d(2 * ch, ch, 3, padding=1))       # 14 -> 28
        self.u2 = ResBlock(2 * ch, ch, emb)                           # concat with skip from d1
        self.out = nn.Sequential(nn.GroupNorm(8, ch), nn.SiLU(), nn.Conv2d(ch, 1, 3, padding=1))
        nn.init.zeros_(self.out[-1].weight); nn.init.zeros_(self.out[-1].bias)   # start by predicting 0

    def forward(self, x, t, y):
        emb = self.t_mlp(t) + self.y_emb(y)                           # (B, 4ch)
        h0 = self.inc(x)                                              # (B, ch, 28, 28)
        h1 = self.d1(h0, emb)                                         # (B, ch, 28, 28)
        h2 = self.d2(self.ds1(h1), emb)                               # (B, 2ch, 14, 14)
        m = self.m2(self.ma(self.m1(self.ds2(h2), emb)), emb)         # (B, 2ch, 7, 7)
        u = self.u1(torch.cat([self.us1(m), h2], 1), emb)             # (B, 2ch, 14, 14)
        u = self.u2(torch.cat([self.us2(u), h1], 1), emb)             # (B, ch, 28, 28)
        return self.out(u)                                            # (B, 1, 28, 28): predicted noise

net = UNet()
print(sum(p.numel() for p in net.parameters()) / 1e6, "M parameters")
x = torch.randn(5, 1, 28, 28); t = torch.randint(0, 1000, (5,)); y = torch.randint(0, 11, (5,))
print(net(x, t, y).shape)                                             # torch.Size([5, 1, 28, 28])
```

### Part 3: training with condition dropout and an EMA {#lab-train}

```python:lab_train.py
import copy, torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
# assumes Diffusion, UNet, device from the previous parts (and drop_condition from cfg_core.py)

NULL_LABEL = 10
def drop_condition(y, p_uncond=0.1):
    mask = torch.rand(y.shape[0], device=y.device) < p_uncond
    return torch.where(mask, torch.full_like(y, NULL_LABEL), y)

tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])   # pixels to [-1, 1]
train = datasets.MNIST("data", train=True, download=True, transform=tf)
loader = DataLoader(train, batch_size=128, shuffle=True, drop_last=True, num_workers=2)

diff = Diffusion(T=1000, schedule="linear")
model = UNet().to(device)
ema = copy.deepcopy(model).eval().requires_grad_(False)                # exponential moving average of weights
opt = torch.optim.AdamW(model.parameters(), lr=2e-4)

@torch.no_grad()
def ema_update(ema, model, decay=0.999):
    for pe, pm in zip(ema.parameters(), model.parameters()):
        pe.lerp_(pm, 1 - decay)                                        # pe = decay * pe + (1 - decay) * pm

for epoch in range(10):
    model.train(); total = 0.0
    for x0, y in loader:
        x0, y = x0.to(device), drop_condition(y.to(device))
        loss = diff.loss(model, x0, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); ema_update(ema, model)
        total += loss.item()
    print(f"epoch {epoch + 1:2d}  mean loss {total / len(loader):.4f}")   # roughly 0.05-0.03 and slowly falling
torch.save(ema.state_dict(), "ddpm_mnist_ema.pt")
```

The loss is a mean-squared error on unit-variance noise: at initialization (predicting zero) it equals about 1.0, and it should drop quickly to a few hundredths. Do not expect it to keep falling dramatically: the irreducible part (the noise at high $t$ that truly cannot be predicted) sets a floor, which is why **sample quality, not loss, is the metric you must look at**.

### Part 4: samplers (DDPM and DDIM) with guidance {#lab-sample}

```python:lab_sample.py
import torch
# assumes Diffusion `diff`, trained EMA model `ema`, device, NULL_LABEL

@torch.no_grad()
def eps_guided(model, x, t, y, w):
    if w == 1.0:
        return model(x, t, y)
    e_c, e_u = model(torch.cat([x, x]), torch.cat([t, t]), torch.cat([y, torch.full_like(y, NULL_LABEL)])).chunk(2)
    return e_u + w * (e_c - e_u)

@torch.no_grad()
def sample_ddpm(model, diff, y, w=1.0, shape=(1, 28, 28), seed=0):
    """Ancestral sampling, T network evaluations (uses sigma_t^2 = beta_tilde_t)."""
    g = torch.Generator(device=y.device).manual_seed(seed)
    x = torch.randn(len(y), *shape, device=y.device, generator=g)
    for i in reversed(range(diff.T)):
        t = torch.full((len(y),), i, device=y.device, dtype=torch.long)
        eps = eps_guided(model, x, t, y, w)
        mean = (x - diff.betas[i] / (1 - diff.ab[i]).sqrt() * eps) / diff.alphas[i].sqrt()
        noise = torch.randn(x.shape, device=y.device, generator=g) if i > 0 else 0.0
        x = mean + diff.post_var[i].sqrt() * noise
    return x.clamp(-1, 1)

@torch.no_grad()
def sample_ddim(model, diff, y, steps=50, w=1.0, eta=0.0, shape=(1, 28, 28), seed=0):
    """DDIM over a subsequence of `steps` timesteps. eta = 0 is deterministic."""
    g = torch.Generator(device=y.device).manual_seed(seed)
    x = torch.randn(len(y), *shape, device=y.device, generator=g)
    ts = torch.linspace(diff.T - 1, 0, steps).round().long().tolist()          # e.g. 999, ..., 0
    for j, i in enumerate(ts):
        t = torch.full((len(y),), i, device=y.device, dtype=torch.long)
        eps = eps_guided(model, x, t, y, w)
        ab_t = diff.ab[i]
        ab_s = diff.ab[ts[j + 1]] if j + 1 < len(ts) else torch.tensor(1.0, device=y.device)   # alpha_bar at the next step
        x0 = ((x - (1 - ab_t).sqrt() * eps) / ab_t.sqrt()).clamp(-1, 1)         # clip the x0 estimate to the data range
        eps = (x - ab_t.sqrt() * x0) / (1 - ab_t).sqrt()                       # re-derive eps consistent with the clipped x0
        sigma = eta * ((1 - ab_s) / (1 - ab_t)).sqrt() * (1 - ab_t / ab_s).clamp(min=0).sqrt()
        noise = torch.randn(x.shape, device=y.device, generator=g) if eta > 0 else 0.0
        x = ab_s.sqrt() * x0 + (1 - ab_s - sigma ** 2).clamp(min=0).sqrt() * eps + sigma * noise
    return x.clamp(-1, 1)

ema.load_state_dict(torch.load("ddpm_mnist_ema.pt", map_location=device)); ema.to(device)
y = torch.arange(10, device=device).repeat_interleave(8)                       # 8 samples of each digit
imgs = sample_ddim(ema, diff, y, steps=50, w=3.0)
from torchvision.utils import save_image
save_image((imgs + 1) / 2, "samples_ddim50_w3.png", nrow=8)
print(imgs.shape, imgs.min().item(), imgs.max().item())
```

### Part 5: experiments {#lab-experiments}

**Experiment A: steps.** Compare DDIM at 5, 10, 20, 50 and 200 steps and DDPM at 1000 steps, with the same seed (so the same starting noise), at $w=1$. Look at the grids first, then measure quality with the quick classifier below. Expect: with 5 steps images are blobby; at 20-50 DDIM steps digits are clearly legible; DDPM at 1000 steps is the quality reference but 20-50 times slower.

**Experiment B: guidance.** Fix 50 DDIM steps and sweep $w\in\{0,1,2,4,8\}$. Expect $w=0$ (unconditional) to ignore the label, $w=1$ to follow it mostly, and $w\ge4$ to make digits bolder and more uniform in style (less diversity), with artifacts and saturation by $w=8$.

```python:lab_eval.py
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
# assumes sample_ddim, sample_ddpm, ema, diff, device from the previous parts

# 1) a tiny MNIST classifier (about 99% test accuracy after 3 epochs) to judge sample legibility
tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
tr = DataLoader(datasets.MNIST("data", train=True, transform=tf), batch_size=256, shuffle=True)
clf = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                    nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                    nn.Flatten(), nn.Linear(64 * 7 * 7, 128), nn.ReLU(), nn.Linear(128, 10)).to(device)
opt = torch.optim.Adam(clf.parameters(), 1e-3)
for _ in range(3):
    for x, y in tr:
        loss = F.cross_entropy(clf(x.to(device)), y.to(device)); opt.zero_grad(); loss.backward(); opt.step()
clf.eval()

# 2) metrics on 500 samples per setting: label accuracy and within-class diversity
@torch.no_grad()
def evaluate(sampler, n_per_class=50, **kw):
    y = torch.arange(10, device=device).repeat_interleave(n_per_class)
    imgs = sampler(ema, diff, y, **kw)
    acc = (clf(imgs).argmax(1) == y).float().mean().item()
    flat = imgs.flatten(1).view(10, n_per_class, -1)
    div = torch.stack([torch.pdist(c).mean() for c in flat]).mean().item()   # mean pairwise L2 within class
    return acc, div

for steps in [5, 10, 20, 50, 200]:
    acc, div = evaluate(sample_ddim, steps=steps, w=1.0)
    print(f"DDIM {steps:4d} steps   accuracy {acc:.3f}   diversity {div:.2f}")
for w in [0, 1, 2, 4, 8]:
    acc, div = evaluate(sample_ddim, steps=50, w=float(w))
    print(f"guidance w={w}   accuracy {acc:.3f}   diversity {div:.2f}")
```

You should see class accuracy rise with $w$ (from chance, about 0.1, at $w=0$) and saturate near 1, while the diversity number falls: the same trade-off as the toy chart above. If accuracy at $w=1$ is already near 1, push the model harder by training for fewer epochs, or compare at $w=0.5$.

:::tip Extensions
(1) Switch to the cosine schedule and compare 10-step samples. (2) Change the network target to $v$-prediction using `training_target` from the parameterization code, and convert with `to_eps` inside the sampler. (3) Add a progress plot of $\hat x_0$ along the DDIM trajectory to see coarse-to-fine generation: blobs first, then strokes. (4) Train on Fashion-MNIST or CIFAR-10 (3 channels, 32x32: change `inc`/`out`, raise `ch` to 128 and add one more resolution level).
:::

## Common mistakes and debugging tips {#mistakes}

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| Samples are pure noise or grey blobs, loss barely moves from 1.0 | data not scaled to $[-1,1]$, so the signal-to-noise assumptions break; or timestep not passed to the network | normalize with mean 0.5, std 0.5; check `t` enters every block |
| Loss is fine, samples are very over-saturated or burnt | sampler uses wrong indices (off-by-one between 0-based arrays and $t$ in 1..$T$), or a missing clamp of $\hat x_0$ | print `ab[0]` and `ab[T-1]`; use the same index convention in training and sampling; clamp $\hat x_0$ to the data range |
| Samples bright or dim on average, never very dark or light | non-zero terminal SNR: model trained with $\bar\alpha_T>0$ and sampled from pure noise | zero-terminal-SNR rescaling plus $v$-prediction; or at least a schedule with $\bar\alpha_T\approx0$ |
| Guided samples ignore the label | the null label was never used in training (no condition dropout), or sampling uses the wrong null index | `p_uncond` about 0.1; same `NULL_LABEL` in train and sample; check $\epsilon_c\ne\epsilon_\varnothing$ |
| Guided samples saturated, texture-less | guidance scale too high; high-guidance artifacts | lower $w$, apply guidance only in mid-trajectory, rescale or threshold $\hat x_0$ |
| DDIM with few steps looks worse than expected | timestep subsequence skipped the high-noise region, or eta nonzero with few steps | include $t=T-1$, space steps evenly (or in log-SNR), use $\eta=0$ for few steps |
| Training spikes or diverges | high learning rate, no gradient clipping, or a bad init | lr about 1e-4 to 2e-4, clip grad norm to 1, zero-init the output layer, use an EMA for sampling |
| EMA model gives worse samples than the raw model early on | EMA decay too slow relative to the number of steps | use `decay=0.99` for short runs or warm up the decay; always sample from the EMA once it has seen enough steps |
| Different results on every run despite a fixed seed | sampling noise from the global RNG or non-deterministic ops | pass a `torch.Generator` to every `randn` (as in the lab) |
| Latent diffusion produces blurry or banded images | wrong latent scaling factor, or the decoder is too weak | scale latents to unit variance; check the autoencoder's reconstructions first |
| Memory blow-up in attention at higher resolution | self-attention at 32x32 or 64x64 maps ($N^2$ cost) | use attention only at 16x16 or lower; use flash attention; use latent diffusion |

A general debugging habit: **test the pieces separately.** (1) Check that `q_sample` with $t=0$ returns almost the input and with $t=T-1$ is almost pure noise. (2) Overfit a single mini-batch: loss should fall far below the usual floor. (3) Evaluate the denoiser by plotting $\hat x_0$ at several fixed $t$, which shows blurry means at high noise and sharp digits at low noise. (4) Use the toy exact-score sampler of this page to validate your sampler code, since it removes the network as a source of error.

## Exercises {#exercises}

:::exercise Exercise 1: variance preservation and the closed form
(a) Show that if $x_{t-1}$ has mean $m$ and variance $v$ per coordinate, then $x_t$ has mean $\sqrt{\alpha_t}\,m$ and variance $\alpha_t v+\beta_t$. (b) Use this to show that if the data have unit variance and zero mean, then every $x_t$ has zero mean and unit variance. (c) For data with variance $\sigma_0^2$ and mean $m_0$, find the mean and variance of $x_t$ in terms of $\bar\alpha_t$.
:::details Solution
(a) $x_t=\sqrt{\alpha_t}x_{t-1}+\sqrt{\beta_t}\epsilon$ with $\epsilon$ independent: mean $\sqrt{\alpha_t}m$, variance $\alpha_tv+\beta_t$. (b) With $m=0$, $v=1$: variance $=\alpha_t+\beta_t=1$ at every step, by induction. (c) By the closed form $x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon$: mean $\sqrt{\bar\alpha_t}m_0$, variance $\bar\alpha_t\sigma_0^2+1-\bar\alpha_t$. This tends to $1$ as $\bar\alpha_t\to0$ regardless of $\sigma_0$, but it only *equals* 1 at every step if $\sigma_0=1$. This is why data are normalized before diffusion and why latent-diffusion systems rescale their latents.
:::
:::

:::exercise Exercise 2: where is the signal-to-noise ratio equal to one?
Using the NumPy code of the [forward process](#closed-form), find the first step $t$ at which $\mathrm{SNR}(t)<1$ (i.e. $\bar\alpha_t<0.5$) for the linear schedule and for the cosine schedule with $T=1000$. What does the result say about how each schedule spends its steps?
:::details Solution
For the linear schedule, `np.argmax(ab < 0.5) + 1` returns $t=260$ ($\bar\alpha_{260}\approx0.4976$). For the cosine schedule the crossing is at $t\approx497$, almost exactly the midpoint. So the linear schedule reaches equal signal and noise power after only a quarter of the steps and then spends the remaining three quarters (SNR from 1 down to $4\times10^{-5}$) on very noisy inputs, where the network has little to work with. The cosine schedule is much more evenly spread (see the log-SNR chart). This is the quantitative argument for the cosine schedule on small images.
:::
:::

:::exercise Exercise 3: the SNR-weighted view of the epsilon-loss
Show that $\|\epsilon-\hat\epsilon\|^2=\mathrm{SNR}(t)\,\|x_0-\hat x_0\|^2$ and $\|v-\hat v\|^2=(1+\mathrm{SNR}(t))\,\|x_0-\hat x_0\|^2$, where $\hat x_0$ and $\hat\epsilon$ are the estimates implied by the same $x_t$.
:::details Solution
Both estimates satisfy $\sqrt{\bar\alpha}\hat x_0+\sqrt{1-\bar\alpha}\hat\epsilon=x_t=\sqrt{\bar\alpha}x_0+\sqrt{1-\bar\alpha}\epsilon$. So $\sqrt{1-\bar\alpha}(\epsilon-\hat\epsilon)=\sqrt{\bar\alpha}(\hat x_0-x_0)$, which gives $\|\epsilon-\hat\epsilon\|^2=\frac{\bar\alpha}{1-\bar\alpha}\|x_0-\hat x_0\|^2$. For $v=\sqrt{\bar\alpha}\epsilon-\sqrt{1-\bar\alpha}x_0$, the error is $\hat v-v=\sqrt{\bar\alpha}(\hat\epsilon-\epsilon)-\sqrt{1-\bar\alpha}(\hat x_0-x_0)$. Substituting $\hat\epsilon-\epsilon=\sqrt{\bar\alpha/(1-\bar\alpha)}\,(x_0-\hat x_0)$ gives $\big(\tfrac{\bar\alpha}{\sqrt{1-\bar\alpha}}+\sqrt{1-\bar\alpha}\big)(x_0-\hat x_0)=\tfrac{1}{\sqrt{1-\bar\alpha}}(x_0-\hat x_0)$, so the squared error is $\frac1{1-\bar\alpha}\|x_0-\hat x_0\|^2=(1+\mathrm{SNR})\|x_0-\hat x_0\|^2$ because $1+\frac{\bar\alpha}{1-\bar\alpha}=\frac1{1-\bar\alpha}$.
:::
:::

:::exercise Exercise 4: the optimal denoiser for Gaussian data (derivation)
Let $x_0\sim\mathcal N(0,\sigma_0^2)$ in one dimension. (a) Find the marginal of $x_t$. (b) Find the optimal noise predictor $\epsilon^*(x_t,t)=\E[\epsilon\mid x_t]$ and the score $\nabla\log p_t$, and verify that $\epsilon^*=-\sqrt{1-\bar\alpha_t}\,\nabla\log p_t$. (c) Verify Tweedie's formula for $\E[x_0\mid x_t]$.
:::details Solution
(a) $x_t\sim\mathcal N(0,v_t)$ with $v_t=\bar\alpha_t\sigma_0^2+1-\bar\alpha_t$. (b) $x_t$ and $\epsilon$ are jointly Gaussian with $\mathrm{Cov}(x_t,\epsilon)=\sqrt{1-\bar\alpha_t}$, so $\E[\epsilon\mid x_t]=\frac{\sqrt{1-\bar\alpha_t}}{v_t}x_t$. The score is $\nabla\log p_t(x_t)=-x_t/v_t$, so $-\sqrt{1-\bar\alpha_t}\nabla\log p_t=\sqrt{1-\bar\alpha_t}\,x_t/v_t$, matching. (c) $\E[x_0\mid x_t]=\frac{\sqrt{\bar\alpha_t}\sigma_0^2}{v_t}x_t$ from the joint Gaussian with $\mathrm{Cov}(x_t,x_0)=\sqrt{\bar\alpha_t}\sigma_0^2$. Tweedie: $\big(x_t+(1-\bar\alpha_t)(-x_t/v_t)\big)/\sqrt{\bar\alpha_t}=x_t\,\frac{v_t-(1-\bar\alpha_t)}{v_t\sqrt{\bar\alpha_t}}=x_t\frac{\bar\alpha_t\sigma_0^2}{v_t\sqrt{\bar\alpha_t}}$, identical. Note also what it says: at high noise ($v_t\to1$) the best guess of $x_0$ shrinks to the data mean; at $\bar\alpha_t\to1$ it returns $x_t$.
:::
:::

:::exercise Exercise 5: v-prediction conversions (coding)
Starting from the `parameterizations.py` code, train a tiny MLP to denoise a 2-D two-moons dataset with each of the three targets ($\epsilon$, $x_0$, $v$) on the linear schedule. For each, report the final loss, and the error in $\hat x_0$ for 100 test points at $t=50$, $t=500$ and $t=950$. Which target gives the most uniform behavior across $t$, and why?
:::details Solution
A reference implementation: build `x_t` with the closed form, feed `[x_t, t/T]` to a 3-layer MLP, regress the chosen target with MSE, and for evaluation convert the output to $\hat x_0$ with `to_x0`. Typical outcome: $\epsilon$-prediction gives small $\hat x_0$ errors at low $t$ but large ones at $t=950$ (the conversion divides by $\sqrt{\bar\alpha_t}\approx0.04$, amplifying network error by about 25); $x_0$-prediction is best at high $t$ but blurry at low $t$ (it must recover fine noise-free detail from a nearly clean input without ever being told which part is noise); $v$ is the most uniform, since by Exercise 3 its implied $x_0$-weight $1+\mathrm{SNR}$ never vanishes and never explodes relative to the target scale. Your numbers will vary with seed and architecture; check the trend, not the absolute values.
:::
:::

:::exercise Exercise 6: classifier-free guidance on a toy problem (coding, challenging)
Reproduce the guidance chart. Use the exact-score machinery of `ddim_toy_exact_score.py` with two classes $N(-1,0.8^2)$ and $N(+1,0.8^2)$, the unconditional model being their equal-weight mixture and the conditional model the single Gaussian of class B. Sample with 100-step DDIM and $\tilde\epsilon=\epsilon_\varnothing+w(\epsilon_c-\epsilon_\varnothing)$ for $w\in\{0,1,2,4,8\}$. (a) Report the sample mean and standard deviation. (b) Explain why the mean at $w=8$ exceeds the class mean $+1$. (c) What would the guided density be if both classes were *exactly* separated and Gaussian with the same variance? Derive it.
:::details Solution
```python
import numpy as np
T = 1000
ab = np.concatenate([[1.0], np.cumprod(1 - np.linspace(1e-4, 0.02, T))])
mc, sc = np.array([-1.0, 1.0]), 0.8

def eps_star(x, k, mus, sigs, wts):
    a = ab[k]; m = np.sqrt(a) * mus[None, :]; v = a * sigs**2 + 1 - a
    lp = np.log(wts)[None, :] - 0.5 * (x[:, None] - m) ** 2 / v - 0.5 * np.log(v)
    p = np.exp(lp - lp.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    return np.sqrt(1 - a) * (p * (x[:, None] - m) / v).sum(1)          # = -sqrt(1-ab) * score

def run(w, n=100, N=20000, seed=2):
    rng = np.random.default_rng(seed); ks = np.round(np.linspace(T, 0, n + 1)).astype(int)
    x = rng.standard_normal(N)
    for t, s in zip(ks[:-1], ks[1:]):
        ec = eps_star(x, t, mc[1:], sc, np.array([1.0]))               # class B only
        eu = eps_star(x, t, mc, sc, np.array([.5, .5]))                # unconditional mixture
        e = eu + w * (ec - eu)
        x0 = (x - np.sqrt(1 - ab[t]) * e) / np.sqrt(ab[t])
        x = np.sqrt(ab[s]) * x0 + np.sqrt(1 - ab[s]) * e
    return x.mean(), x.std()

for w in [0, 1, 2, 4, 8]:
    print(w, run(w))
```
(a) Approximately: $w=0$: mean 0.01, std 1.26 (the unconditional mixture); $w=1$: mean 1.00, std 0.78 (class B); $w=2$: 1.51, 0.57; $w=4$: 2.08, 0.45; $w=8$: 2.79, 0.38. (b) The guided density is $\propto p(x)\,p(B\mid x)^w$. Since $p(B\mid x)$ is a logistic function of $x$ that keeps increasing toward $x\to+\infty$, raising it to a large power drags mass toward larger $x$, away from the class boundary and *past* the class mean: guidance optimizes "how clearly class B" and not "how typical of class B". (c) For $p(x)=\tfrac12\mathcal N(\mu_A,\sigma^2)+\tfrac12\mathcal N(\mu_B,\sigma^2)$ the guided density $p(x\mid B)^w p(x)^{1-w}$ for $w$ greater than 1 is not Gaussian: with well-separated classes the term $p(x)^{1-w}$ is a negative power of a mixture, which is large between and outside the modes. In the region near mode B, $p(x)\approx\frac12 p(x\mid B)$, so the density is $\propto p(x\mid B)\,(1/2)^{1-w}$: nearly unchanged. So *for well-separated classes guidance does almost nothing*; it matters exactly when the classes overlap, which is the regime of real images and text.
:::
:::

:::exercise Exercise 7: is DDIM marginal-preserving? (what-if)
Take Gaussian data $N(0,\sigma_0^2)$ with $\sigma_0=2$ and the exact optimal $\epsilon^*$ from Exercise 4. Apply one deterministic DDIM step ($\eta=0$) from $\bar\alpha_t=0.3$ to $\bar\alpha_s=0.6$. Compute the variance of the output and compare with the target marginal variance $\bar\alpha_s\sigma_0^2+1-\bar\alpha_s=2.8$. Repeat with $\bar\alpha_s=0.35$ and $0.301$. What do you conclude?
:::details Solution
With $\epsilon^*=\sqrt{1-\bar\alpha_t}\,x/v_t$ and $\hat x_0=\sqrt{\bar\alpha_t}\sigma_0^2x/v_t$, the step is linear: $x_s=c\,x$ with $c=\big(\sqrt{\bar\alpha_t\bar\alpha_s}\sigma_0^2+\sqrt{(1-\bar\alpha_t)(1-\bar\alpha_s)}\big)/v_t$. Output variance $c^2v_t$: for $\bar\alpha_s=0.6$ it is $2.608$ (target $2.8$); for $0.35$, $2.044$ (target $2.05$); for $0.301$, $1.9030$ (target $1.9030$ to four digits). A single large DDIM step does **not** preserve the marginal; the error shrinks quickly (roughly quadratically in the step size) as $s\to t$, where DDIM becomes the exact probability-flow ODE. This is why few-step sampling has an error floor, why higher-order solvers help, and why the "DDIM 1000 steps" line in the chart plateaus above Monte Carlo noise.
:::
:::

:::exercise Exercise 8: change of variables and coupling layers
(a) A 1-D flow is $x=az+b$ with $z\sim\mathcal N(0,1)$. Compute $\log p_X(x)$ by the change-of-variables formula and check it against the Gaussian density. (b) For the affine coupling layer with $x\in\R^2$, $y_1=x_1$, $y_2=x_2e^{s(x_1)}+t(x_1)$, write the Jacobian and its determinant. (c) Why would using $s(x_1,x_2)$ in place of $s(x_1)$ break the efficiency of the layer?
:::details Solution
(a) $f(x)=(x-b)/a$, $|f'|=1/|a|$, so $\log p_X(x)=\log\mathcal N\big((x-b)/a;0,1\big)-\log|a|=-\tfrac12\frac{(x-b)^2}{a^2}-\tfrac12\log2\pi-\log|a|$, which is the log-density of $\mathcal N(b,a^2)$. (b) $J=\begin{pmatrix}1&0\\ \partial y_2/\partial x_1&e^{s(x_1)}\end{pmatrix}$, $\det J=e^{s(x_1)}$, $\log|\det J|=s(x_1)$. (c) With $s(x_1,x_2)$ the entry $\partial y_2/\partial x_2$ would pick up a $x_2\,\partial s/\partial x_2$ term, and the map would no longer be invertible in closed form (solving $y_2=x_2e^{s(x_1,x_2)}+t$ for $x_2$ is an implicit equation); the Jacobian would stop being triangular with an easy diagonal. The whole point of coupling is that $s,t$ depend only on the unchanged half.
:::
:::

:::exercise Exercise 9: what would happen if...
(a) You trained a conditional diffusion model with $p_\text{uncond}=0$ and then sampled with $w=3$ using a null label. (b) You trained with $p_\text{uncond}=0.9$. (c) You used BatchNorm instead of GroupNorm in the U-Net and sampled one image at a time at inference. (d) You sampled a model trained for $T=1000$ steps using only the timesteps $\{999, 500, 0\}$ with DDIM.
:::details Solution
(a) The null embedding is an untrained, random vector; $\epsilon_\varnothing$ is then an arbitrary extrapolation and guided samples are corrupted (typically artifacts or noise-like images). Guidance needs both tasks to be trained. (b) The conditional model sees the label only 10% of the time and learns it poorly; guidance can compensate partly (the unconditional branch is excellent), but the conditional branch is weak, so prompt fidelity is bad and you waste capacity. Typical values are 0.05 to 0.2. (c) BatchNorm statistics at training time come from batches of mixed noise levels and at inference from a batch of one at one noise level: train/test mismatch gives poor samples, and it couples samples within a batch. GroupNorm and LayerNorm are per-sample. (d) Three steps is far too coarse: the model has an error floor (Exercise 7 and the chart, 3-step error of about 0.63 versus 0.02 at 1000 steps) and the result would be blurry digit-like blobs. The first step from pure noise to the model's $\hat x_0$ is essentially the posterior mean, the blurry average of all digits.
:::
:::

:::exercise Exercise 10: the probability-flow ODE (derivation, challenging)
Using the Fokker-Planck equation $\partial_tp_t=-\nabla\cdot(fp_t)+\tfrac12g^2\Delta p_t$ for $dx=f\,dt+g\,dW$, show that the deterministic ODE $\dot x=f-\tfrac12g^2\nabla\log p_t$ has the same marginals $p_t$. Then explain why the reverse SDE has drift $f-g^2\nabla\log p_t$ but the ODE uses only half of the score term.
:::details Solution
Use $\Delta p=\nabla\cdot\nabla p=\nabla\cdot(p\,\nabla\log p)$. Then $\partial_tp_t=-\nabla\cdot\big(fp_t\big)+\tfrac12g^2\nabla\cdot(p_t\nabla\log p_t)=-\nabla\cdot\Big(\big[f-\tfrac12g^2\nabla\log p_t\big]p_t\Big)$. This is the continuity (Liouville) equation for a deterministic flow with velocity $\tilde f=f-\tfrac12g^2\nabla\log p_t$, so particles following $\dot x=\tilde f$ have marginals $p_t$. For the reverse-time SDE with noise $g$ we need two things: time reversal flips the sign of the drift of the (forward) *diffusion* contribution, and the stochastic term itself diffuses mass; to cancel that diffusion and reverse the density evolution we need an extra score term. Concretely, running the Fokker-Planck equation backward for a process with diffusion $g$ requires the full $g^2\nabla\log p_t$, half of which is "undoing" the diffusion that the added noise $g\,d\bar W$ does again. In the ODE there is no noise term to undo, so half the score is sufficient. In general one can interpolate: $dx=\big[f-\tfrac{1+\lambda^2}{2}g^2\nabla\log p_t\big]dt+\lambda g\,d\bar W$ has marginals $p_t$ for any $\lambda\ge0$ (with $\lambda=0$ the ODE and $\lambda=1$ the reverse SDE), which is the continuous analogue of DDIM's $\eta$.
:::
:::

:::exercise Exercise 11: provenance, watermarking and detection
You are asked to design the authenticity strategy for a photo-generation product. (a) What does a C2PA manifest prove, and what does it fail to prove? (b) Name two ways a watermark can fail. (c) Why is a classifier-based "AI image detector" a poor sole basis for moderation or academic-integrity decisions?
:::details Solution
(a) It proves that a specific signing party (identified by a certificate chain) asserted particular claims (capture device, edits, generation tool) about a particular asset hash, and that the asset and manifest have not been modified since signing. It does not prove the *claims* are true (a signer could lie), does not survive stripping (absence of a manifest tells you nothing), and does not detect fakes made without it. (b) Removal or degradation by re-generation, heavy compression, cropping, noise or paraphrase attacks (an adversary with access to the detector may optimize against it), and non-adoption (open-weight models and other providers simply do not embed one); also forgery of a watermark on real content if the key leaks. (c) Detectors have non-zero false-positive rates that hit real people (and have shown bias against some groups, for example non-native writers for text detectors), they generalize poorly to new generators and edits, and their scores are not calibrated for individual decisions. Use them as one weak signal alongside provenance and human review.
:::
:::

## Key takeaways

:::key Key takeaways
- Diffusion has a fixed forward noising chain and a learned reverse chain. The closed form $x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\,\epsilon$ turns training into cheap regression at a random noise level.
- The ELBO is a sum of KLs between Gaussians; with the $\epsilon$-parameterization it reduces to weighted noise prediction, and dropping the weights ($\mathcal L_\text{simple}$) improves samples but is no longer a likelihood bound.
- $\epsilon$-, $x_0$- and $v$-prediction are linear reparameterizations of each other; they differ in implicit loss weighting (SNR versus $1+$SNR versus 1) and numerical conditioning. $v$-prediction behaves well at every noise level.
- The network is a score estimator: $s_\theta=-\epsilon_\theta/\sqrt{1-\bar\alpha_t}$. Denoising score matching gives the loss; the reverse SDE and the deterministic probability-flow ODE generate samples from it.
- DDIM and higher-order ODE solvers reduce 1000 steps to 10-50 because the probability-flow trajectory is smooth; distillation and consistency training go lower by changing the model.
- Classifier-free guidance, $\tilde\epsilon=\epsilon_\varnothing+w(\epsilon_c-\epsilon_\varnothing)$, samples from $p(x)p(c\mid x)^w$: higher fidelity and prompt adherence, lower diversity, artifacts at large $w$.
- Latent diffusion runs diffusion in a VAE latent space (48 times fewer values at $f=8$) with cross-attention to text tokens; U-Nets and DiTs are interchangeable denoisers, DiTs scale predictably.
- Flows give exact likelihood through invertible layers with cheap Jacobians (coupling layers); autoregressive models factorize by the chain rule and now operate on learned tokens. Each family trades quality, likelihood, speed and stability differently.
- Generative media bring real risks (deepfakes, memorization, bias). Provenance (C2PA), watermarking and detection are complementary and none is sufficient alone.
:::

## Further reading {#further-reading}

**Essential**
- Ho, Jain and Abbeel, "Denoising Diffusion Probabilistic Models", NeurIPS 2020 (arXiv:2006.11239). The paper that made diffusion practical: read the forward process, the simplified loss and the U-Net.
- Song, Sohl-Dickstein, Kingma, Kumar, Ermon and Poole, "Score-Based Generative Modeling through Stochastic Differential Equations", ICLR 2021 (arXiv:2011.13456). The SDE/ODE unification.
- Rombach, Blattmann, Lorenz, Esser and Ommer, "High-Resolution Image Synthesis with Latent Diffusion Models", CVPR 2022 (arXiv:2112.10752). The basis of Stable Diffusion.
- Ho and Salimans, "Classifier-Free Diffusion Guidance", NeurIPS 2021 Workshop on Deep Generative Models, 2021/2022 (arXiv:2207.12598).

**Foundations and theory**
- Sohl-Dickstein, Weiss, Maheswaranathan and Ganguli, "Deep Unsupervised Learning using Nonequilibrium Thermodynamics", ICML 2015 (arXiv:1503.03585).
- Vincent, "A Connection Between Score Matching and Denoising Autoencoders", Neural Computation, 2011.
- Song and Ermon, "Generative Modeling by Estimating Gradients of the Data Distribution", NeurIPS 2019 (arXiv:1907.05600).
- Song, Meng and Ermon, "Denoising Diffusion Implicit Models", ICLR 2021 (arXiv:2010.02502).
- Nichol and Dhariwal, "Improved Denoising Diffusion Probabilistic Models", ICML 2021 (arXiv:2102.09672): cosine schedule, learned variances.
- Kingma, Salimans, Poole and Ho, "Variational Diffusion Models", NeurIPS 2021 (arXiv:2107.00630).
- Karras, Aittala, Aila and Laine, "Elucidating the Design Space of Diffusion-Based Generative Models", NeurIPS 2022 (arXiv:2206.00364).
- Dhariwal and Nichol, "Diffusion Models Beat GANs on Image Synthesis", NeurIPS 2021 (arXiv:2105.05233).
- Salimans and Ho, "Progressive Distillation for Fast Sampling of Diffusion Models", ICLR 2022 (arXiv:2202.00512): $v$-prediction.
- Lipman, Chen, Ben-Hamu, Nickel and Le, "Flow Matching for Generative Modeling", ICLR 2023 (arXiv:2210.02747).

**Architectures and systems**
- Ronneberger, Fischer and Brox, "U-Net: Convolutional Networks for Biomedical Image Segmentation", MICCAI 2015 (arXiv:1505.04597).
- Peebles and Xie, "Scalable Diffusion Models with Transformers", ICCV 2023 (arXiv:2212.09748).
- Saharia et al., "Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding" (Imagen), NeurIPS 2022 (arXiv:2205.11487).
- Zhang, Rao and Agrawala, "Adding Conditional Control to Text-to-Image Diffusion Models" (ControlNet), ICCV 2023 (arXiv:2302.05543).

**Flows and autoregressive models**
- Dinh, Sohl-Dickstein and Bengio, "Density Estimation Using Real NVP", ICLR 2017 (arXiv:1605.08803).
- Kingma and Dhariwal, "Glow: Generative Flow with Invertible 1x1 Convolutions", NeurIPS 2018 (arXiv:1807.03039).
- van den Oord, Kalchbrenner and Kavukcuoglu, "Pixel Recurrent Neural Networks", ICML 2016 (arXiv:1601.06759).
- van den Oord et al., "WaveNet: A Generative Model for Raw Audio", 2016 (arXiv:1609.03499).
- Chang, Zhang, Jiang, Liu and Freeman, "MaskGIT: Masked Generative Image Transformer", CVPR 2022 (arXiv:2202.04200).

**Risks and provenance**
- Carlini et al., "Extracting Training Data from Diffusion Models", USENIX Security 2023 (arXiv:2301.13188).
- Wen, Kirchenbauer, Geiping and Goldstein, "Tree-Ring Watermarks: Fingerprints for Diffusion Images that are Invisible and Robust", NeurIPS 2023 (arXiv:2305.20030).
- Kirchenbauer et al., "A Watermark for Large Language Models", ICML 2023 (arXiv:2301.10226).
- C2PA, "Content Credentials" technical specification (c2pa.org).
- Lin, Liu, Li and Yang, "Common Diffusion Noise Schedules and Sample Steps are Flawed", WACV 2024 (arXiv:2305.08891).

**Tutorials**
- Lilian Weng, "What are Diffusion Models?" (lilianweng.github.io, 2021); Yang Song, "Generative Modeling by Estimating Gradients of the Data Distribution" (blog, 2021); Calvin Luo, "Understanding Diffusion Models: A Unified Perspective" (arXiv:2208.11970).
