---
title: Optimization for Deep Learning
phase: Foundations
summary: Why training deep networks is a hard non-convex problem and how the practical toolbox (SGD, momentum, Adam/AdamW, learning-rate schedules, careful initialization, clipping and normalization layers) makes it work.
---

:::key Learning objectives
By the end of this week you should be able to:
- Describe what the loss surface of a neural network looks like (saddles, plateaus, flat and sharp minima) and what that implies for optimizers.
- Explain the bias-variance structure of mini-batch gradients and use the linear scaling rule and the gradient noise scale to reason about batch size.
- Derive and implement momentum, Nesterov momentum, AdaGrad, RMSProp, Adam (with bias correction) and AdamW (with decoupled weight decay) from scratch.
- Choose and implement learning-rate schedules: step decay, cosine annealing, linear warmup, cyclical rates and the one-cycle policy.
- Derive the Xavier/Glorot and He initialization variances, and diagnose vanishing and exploding gradients, including the use of gradient clipping.
- Write the forward and backward pass of BatchNorm, explain train versus eval behavior, and compare BatchNorm, LayerNorm and GroupNorm.
- Run a controlled experiment comparing optimizers, initializations and learning rates, and read the resulting loss curves.
:::

[Week 2](week-02.html) gave you the machinery to compute gradients with backpropagation. This week is about what you do with them. Training a neural network means minimizing a function of millions or billions of parameters that is non-convex, noisy (because you only ever see a small batch of data per step), badly conditioned and full of strange geometry. Classical optimization theory says very little that is comforting about this setting. And yet a handful of simple, well-tuned ingredients make training reliably succeed: mini-batch stochastic gradient descent, momentum or Adam-style adaptive scaling, a learning-rate schedule with warmup, an initialization that preserves signal scale, gradient clipping for the cases where it does not, and normalization layers that keep activations well behaved.

You will meet each ingredient in the same way: first the intuition, then the math, then a few lines of code, then the failure modes. By the end you will have a mental checklist for the single most common debugging situation in deep learning: "my loss is not going down, or it is going down too slowly, or it blew up."

## The geometry of neural network loss surfaces {#loss-landscapes}

Let $\mathcal{L}(\boldsymbol{\theta})$ be the training loss as a function of all parameters $\boldsymbol{\theta}\in\R^d$. For linear regression with squared error $\mathcal{L}$ is a convex quadratic: one basin, one minimum, and gradient descent with a small enough step is guaranteed to find it. A neural network breaks this in two ways. First, the composition of weight matrices and nonlinearities makes $\mathcal{L}$ non-convex. Second, the parameterization has enormous symmetries: permute the hidden units of a layer (together with the matching rows and columns of neighbouring matrices) and you get exactly the same function, so a minimum always comes with at least $\prod_\ell n_\ell!$ copies. Non-convexity is therefore not an accident but a structural feature.

It is worth being precise about what "non-convex" implies and does not imply. It does *not* imply that gradient descent is hopeless. What matters is which kinds of critical points ($\nabla\mathcal{L}=\mathbf{0}$) exist and how much of the surface is easy to descend.

{{figure: week03-landscape | A cartoon of a non-convex loss along a single direction in weight space. The sharp minimum has slightly lower training loss but a large gap to the (shifted) test loss; the flat minimum is more tolerant of the shift between training and test distributions. Cartoon, not measured data.}}

### Critical points: minima, maxima and saddles {#saddle-points}

At a critical point the behavior is governed by the Hessian $\mathbf{H}=\nabla^2\mathcal{L}(\boldsymbol{\theta})\in\R^{d\times d}$, which is symmetric and therefore has real eigenvalues $\lambda_1\ge\dots\ge\lambda_d$. Writing a small displacement $\boldsymbol{\delta}$ in the eigenbasis, the second-order Taylor expansion at a critical point is

$$
\mathcal{L}(\boldsymbol{\theta}+\boldsymbol{\delta}) \approx \mathcal{L}(\boldsymbol{\theta}) + \tfrac12\sum_{i=1}^d \lambda_i\, c_i^2,
$$

where $c_i$ is the coordinate of $\boldsymbol{\delta}$ along eigenvector $i$. If every $\lambda_i>0$ you are at a local minimum, if every $\lambda_i<0$ at a local maximum, and if the signs are mixed at a **saddle point**: the loss goes up in some directions and down in others.

Now count. In one dimension a critical point is a minimum or a maximum. In $d$ dimensions, if you imagine the signs of the eigenvalues as independent coin flips, the probability that all $d$ are positive is $2^{-d}$. For $d=10^6$ essentially every critical point is a saddle, and local minima with a high loss are exponentially rare compared with saddles. This heuristic (from random matrix theory and spin-glass models) was made influential for deep learning by Dauphin et al. (2014) and Choromanska et al. (2015): in high dimensions the obstacle to optimization is not bad local minima but saddle points and the flat regions around them, where the gradient is tiny and progress is slow. Empirically, on real networks, the local minima that gradient methods reach tend to have training loss close to the global minimum.

:::warning The coin-flip argument is a heuristic
Hessian eigenvalues are not independent, and real losses are not random Gaussian fields. The argument explains why you should expect saddles to outnumber minima; it does not prove that every network trains well. Poor conditioning, dead ReLUs, bad initialization and wrong learning rates produce plenty of failures that have nothing to do with local minima.
:::

{{figure: week03-saddle | A saddle point: curving up along one axis and down along the other. Gradient descent started on the stable axis slows to a crawl near the saddle and may take thousands of steps to leave.}}

Why do saddles hurt gradient descent? Near the saddle the gradient is small, so the step $-\eta\nabla\mathcal{L}$ is small. In the negative-curvature direction the escape is exponential, $c(t)\sim c(0)\,(1+\eta|\lambda_{\min}|)^t$, but it starts from whatever tiny component of the iterate lies along that direction. If you start almost exactly on the stable manifold, you can sit near the saddle for a very long time. The noise in stochastic gradients is a *feature* here: it perturbs the iterate off the stable manifold almost immediately. You can see this in the optimizer race below, where momentum and adaptive methods escape the saddle faster than plain gradient descent.

### Plateaus, ravines and ill-conditioning {#ill-conditioning}

Even when there is a clean minimum, descent can be slow because of the shape of the basin. Consider the quadratic $f(\boldsymbol{\theta})=\tfrac12\boldsymbol{\theta}^\top\mathbf{H}\boldsymbol{\theta}$ with Hessian eigenvalues between $\lambda_{\min}>0$ and $\lambda_{\max}$. Gradient descent $\boldsymbol{\theta}\leftarrow\boldsymbol{\theta}-\eta\mathbf{H}\boldsymbol{\theta}$ multiplies the coordinate along eigenvector $i$ by $(1-\eta\lambda_i)$ every step. Two facts follow immediately.

1. It converges only if $|1-\eta\lambda_i|<1$ for all $i$, i.e. $\eta<2/\lambda_{\max}$. The steepest direction limits the learning rate.
2. The slowest direction contracts by $1-\eta\lambda_{\min}$. With the best possible step $\eta^\star=2/(\lambda_{\max}+\lambda_{\min})$ the per-step contraction factor is
$$
\rho=\frac{\lambda_{\max}-\lambda_{\min}}{\lambda_{\max}+\lambda_{\min}}=\frac{\kappa-1}{\kappa+1},\qquad \kappa=\frac{\lambda_{\max}}{\lambda_{\min}}.
$$

The **condition number** $\kappa$ is the ratio of the steepest to the shallowest curvature. If $\kappa=1000$ you need roughly $\kappa\ln(1/\varepsilon)/2\approx 500\ln(1/\varepsilon)$ steps to shrink the error by a factor $\varepsilon$. Neural network Hessians are famously ill-conditioned: a few outlier eigenvalues (often about as many as there are classes) are far larger than a huge bulk of near-zero ones. This is why plain gradient descent zigzags across steep ravine walls while crawling along the floor, and why the optimizers in the next sections exist.

### Flat versus sharp minima {#flat-vs-sharp}

Different minima can have identical training loss and very different generalization. The idea, going back to Hochreiter and Schmidhuber (1997), is that a **flat** minimum (small Hessian eigenvalues, a wide basin) is better than a **sharp** one (large eigenvalues, narrow basin), because the test loss surface is a slightly shifted version of the training loss surface (see the dashed curve in the landscape figure): at a flat minimum the shift costs little, at a sharp one it costs a lot. A measurement often used is the loss increase under a small random perturbation, or the largest Hessian eigenvalue $\lambda_{\max}(\mathbf{H})$.

Keskar et al. (2017) reported that large-batch training tends to find sharper minima than small-batch training, and that this correlates with a generalization gap. This fed a popular picture: gradient noise from small batches "bounces" the iterate out of sharp basins and into flat ones. The picture is useful but has two caveats you should remember.

- **Sharpness depends on parameterization.** Dinh et al. (2017) showed that for ReLU networks you can rescale one layer by $\alpha$ and the next by $1/\alpha$ without changing the function, which changes the Hessian arbitrarily. A raw $\lambda_{\max}$ is thus not by itself a reliable predictor of generalization; scale-invariant sharpness measures are needed.
- **The batch-size gap can often be closed** by retuning the learning rate and schedule (Goyal et al., 2017; Shallue et al., 2019), so the story is not "large batch = bad".

There is also a link between learning rate and sharpness that is purely dynamical. GD with step $\eta$ on a quadratic diverges once $\lambda_{\max}>2/\eta$, so with a large learning rate the iterates can only live in regions where the sharpness is below $2/\eta$ (observed by Cohen et al. 2021 under the name "edge of stability"). A large learning rate acts as an implicit constraint that steers training toward flatter regions.

:::intuition What to carry forward
Deep learning loss surfaces are mostly saddles and long, curved, ill-conditioned valleys with many roughly equivalent minima. The practical goals are therefore (i) make progress along shallow directions without diverging along steep ones (momentum, adaptivity, normalization), (ii) avoid starting in a region where gradients vanish or explode (initialization), and (iii) end in a wide, low basin (learning-rate schedule, noise, regularization).
:::

{{demo: week03-optimizer-race | Optimizer race on 2D loss surfaces | 560}}

## Gradient descent variants and batch size {#gd-variants}

The training objective is an average over $N$ examples, $\mathcal{L}(\boldsymbol{\theta})=\frac1N\sum_{i=1}^N \ell_i(\boldsymbol{\theta})$, where $\ell_i$ is the loss on example $i$. The three classical variants differ only in how many examples they use to estimate $\nabla\mathcal{L}$.

| Variant | Examples per step | Gradient | Per-step cost | Noise |
| --- | --- | --- | --- | --- |
| **Batch** (full-batch) GD | all $N$ | exact $\nabla\mathcal{L}$ | $O(N)$ | none |
| **Stochastic** GD (SGD, $B=1$) | 1 | $\nabla\ell_i$ | $O(1)$ | very high |
| **Mini-batch** SGD | $B$ (e.g. 32 to 4096) | $\frac1B\sum_{i\in\mathcal{B}}\nabla\ell_i$ | $O(B)$ | moderate, $\propto 1/B$ |

In modern usage "SGD" almost always means mini-batch SGD: sample a batch $\mathcal{B}$ of size $B$ without replacement (one pass through a shuffled dataset is an **epoch**), compute the average gradient, step.

$$
\boldsymbol{\theta}_{t+1}=\boldsymbol{\theta}_t-\eta\,\boldsymbol{g}_t,\qquad \boldsymbol{g}_t=\frac1B\sum_{i\in\mathcal{B}_t}\nabla\ell_i(\boldsymbol{\theta}_t).
$$

Mini-batches win for three reasons: hardware (a GPU computes 256 examples in barely more time than 8, so per-example cost falls), statistics (the gradient of a random subset is an unbiased estimate of the full gradient, so you get a good direction at a fraction of the price) and generalization (the noise is helpful, see above).

### Mini-batch gradients as noisy estimates {#gradient-noise}

Let $\boldsymbol{G}=\nabla\mathcal{L}(\boldsymbol{\theta})$ be the true gradient and let $\boldsymbol{\Sigma}$ be the covariance of the per-example gradients $\nabla\ell_i$ (over a random example). For a batch of $B$ examples drawn independently,

$$
\E[\boldsymbol{g}_B]=\boldsymbol{G},\qquad \Cov[\boldsymbol{g}_B]=\frac{1}{B}\boldsymbol{\Sigma}.
$$

(When sampling without replacement from a finite dataset there is an extra factor $(N-B)/(N-1)$, which makes the noise vanish when $B=N$.) The expected squared error $\E\norm{\boldsymbol{g}_B-\boldsymbol{G}}^2=\tr(\boldsymbol{\Sigma})/B$ falls like $1/B$. The chart below shows this for a small network: the measured error follows the $1/B$ line and is much larger than the gradient itself for tiny batches.

```chart
{"type": "line", "caption": "Squared error of the mini-batch gradient relative to the full-batch gradient at initialization, versus batch size (200 random batches each, three-spiral model). The error falls as 1/B and is exactly zero when the batch is the full dataset (not shown on the log axis).", "height": 320, "data": {"labels": ["1", "2", "4", "8", "16", "32", "64", "128"], "datasets": [{"label": "measured E||g_B - g||^2", "data": [13.8603, 7.3217, 3.307, 1.7633, 0.7682, 0.4031, 0.1816, 0.0629], "borderColor": "var(--c-blue)", "tension": 0.1}, {"label": "ideal 13.86 / B", "data": [13.86, 6.93, 3.465, 1.7325, 0.8662, 0.4331, 0.2166, 0.1083], "borderColor": "var(--c-muted)", "borderDash": [5, 4], "pointRadius": 0}]}, "options": {"scales": {"x": {"title": {"display": true, "text": "batch size B"}}, "y": {"type": "logarithmic", "title": {"display": true, "text": "gradient error"}}}}}
```

The code below measures this for any PyTorch model: it takes many random batches at a fixed parameter value and reports the mean squared deviation from the full-batch gradient.

```python:measure_gradient_noise.py
import torch, torch.nn as nn, torch.nn.functional as F

torch.manual_seed(0)
N = 300
X = torch.randn(N, 2)
y = (X[:, 0] * X[:, 1] > 0).long()           # XOR-like 2-class toy labels
model = nn.Sequential(nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 2))

def flat_grad(xb, yb):
    """Gradient of the mean loss on (xb, yb), flattened into one vector."""
    model.zero_grad()
    F.cross_entropy(model(xb), yb).backward()
    return torch.cat([p.grad.flatten() for p in model.parameters()]).clone()

g_full = flat_grad(X, y)                      # "true" gradient (full batch)
print(f"||G||^2 = {g_full.pow(2).sum():.4f}")

for B in [1, 4, 16, 64, 256]:
    errs = []
    for _ in range(300):
        idx = torch.randperm(N)[:B]           # sample without replacement
        errs.append((flat_grad(X[idx], y[idx]) - g_full).pow(2).sum())
    print(f"B={B:4d}  E||g_B - G||^2 = {torch.stack(errs).mean():8.4f}")
```

Running it you should see the error shrink roughly in proportion to $1/B$ (a little faster for $B$ close to $N$, because of the finite-population correction).

### The effect of batch size {#batch-size-effects}

If you double the batch size, each step costs about twice the compute but its gradient has half the noise variance. Whether that is a good deal depends on whether noise or curvature is the bottleneck, and the **gradient noise scale** makes this precise. McCandlish et al. (2018) define (in its simple form)

$$
\mathcal{B}_{\text{noise}}=\frac{\tr(\boldsymbol{\Sigma})}{\norm{\boldsymbol{G}}^2},
$$

the batch size at which the noise in the gradient estimate is about as large as the signal. The qualitative prediction is a two-regime picture.

- For $B\ll\mathcal{B}_{\text{noise}}$ the gradient is mostly noise, and averaging more examples helps almost linearly: doubling $B$ lets you double the learning rate and halve the number of steps for the same progress, at the same total compute. Large batches give near-perfect parallel speed-up.
- For $B\gg\mathcal{B}_{\text{noise}}$ the gradient is already accurate. More examples give a slightly better estimate and no extra progress per step: you pay twice the compute for nothing, and the number of steps stops falling. There are **diminishing returns**.

The noise scale is not a constant. It is small early in training, when gradients all point the same way, and grows by orders of magnitude as the loss falls and the signal $\norm{\boldsymbol{G}}$ shrinks. This is why people use small batches or progressive batch-size growth for the early phase and why harder tasks tolerate larger batches.

:::math Derivation: the linear scaling rule
Compare $k$ consecutive SGD steps with batch size $B$ and learning rate $\eta$ to a single step with batch size $kB$ and learning rate $\eta'$. Assume the parameters hardly move over $k$ steps, so every step sees essentially the same expected gradient $\boldsymbol{G}$:
$$
\boldsymbol{\theta}_{t+k}\approx\boldsymbol{\theta}_t-\eta\sum_{j=0}^{k-1}\boldsymbol{g}_{t+j}\approx \boldsymbol{\theta}_t - k\eta\,\boldsymbol{G}-\eta\sum_j\boldsymbol{\xi}_j,
$$
where $\boldsymbol{\xi}_j$ is zero-mean noise with covariance $\boldsymbol{\Sigma}/B$. A single large-batch step gives $\boldsymbol{\theta}_t-\eta'\boldsymbol{G}-\eta'\boldsymbol{\xi}$ with $\Cov[\boldsymbol{\xi}]=\boldsymbol{\Sigma}/(kB)$. The drift terms match if $\eta'=k\eta$. The noise covariance then is $\eta'^2\boldsymbol{\Sigma}/(kB)=k\eta^2\boldsymbol{\Sigma}/B$ for the large batch and $k\,\eta^2\boldsymbol{\Sigma}/B$ for the $k$ small steps: they match as well. So: **multiply the batch size by $k$, multiply the learning rate by $k$** and training dynamics over the same number of epochs are approximately preserved. (With Adam the corresponding heuristic is a square-root rule, $\eta'\approx\sqrt{k}\,\eta$, because the update is normalized by the gradient scale.)
:::

The assumption that the parameters hardly move is exactly what fails at the start of training, when the weights change rapidly, and at very large batch sizes. Goyal et al. (2017) trained ResNet-50 on ImageNet with batch size 8192 using the linear scaling rule *plus a gradual warmup* of the learning rate during the first few epochs, and matched the accuracy of the small-batch baseline; beyond roughly this batch size they saw the accuracy degrade. The chart below reproduces the rule in miniature.

```chart
{"type": "bar", "caption": "Steps needed to reach full-data loss below 0.35 with plain mini-batch SGD when the learning rate is scaled linearly with batch size (lr = 0.05 at B = 32). Doubling the batch halves the steps: the linear scaling rule holds in this small-batch regime. Smaller batches (B = 8, 16) did not reach the target within 600 steps at their scaled learning rates.", "height": 300, "data": {"labels": ["B=32 (lr 0.05)", "B=64 (lr 0.1)", "B=128 (lr 0.2)", "B=300 (lr 0.469)"], "datasets": [{"label": "steps to loss < 0.35", "data": [315, 158, 81, 41], "backgroundColor": "var(--c-blue)"}]}, "options": {"scales": {"y": {"title": {"display": true, "text": "steps"}}}}}
```

:::note Rules of thumb for choosing a batch size
Pick the largest batch that fits in memory and keeps accelerator utilization high, scale the learning rate with it (linearly for SGD, roughly as the square root for Adam), add warmup, and verify on validation accuracy. If you cannot afford the memory, use **gradient accumulation**: run $k$ micro-batches, sum the gradients, and call `optimizer.step()` once. This is mathematically the same as one batch of $kB$ (except for BatchNorm, whose statistics are computed per micro-batch).
:::

## Momentum and Nesterov accelerated gradient {#momentum}

On a ravine, plain SGD oscillates across the walls (large gradient components with alternating sign) and makes slow progress along the floor (small gradient components with a consistent sign). **Momentum** (Polyak, 1964, "heavy ball") keeps an exponentially decaying running sum of past gradients, the velocity $\boldsymbol{v}$. Components that alternate in sign cancel in that sum; components that keep their sign accumulate.

$$
\boldsymbol{v}_{t}=\beta\,\boldsymbol{v}_{t-1}+\boldsymbol{g}_t,\qquad \boldsymbol{\theta}_{t}=\boldsymbol{\theta}_{t-1}-\eta\,\boldsymbol{v}_{t},
$$

with $\beta\in[0,1)$, typically $0.9$. (PyTorch's `SGD(momentum=...)` uses exactly this form. Some texts write $\boldsymbol{v}_t=\beta\boldsymbol{v}_{t-1}-\eta\boldsymbol{g}_t$ and $\boldsymbol{\theta}_t=\boldsymbol{\theta}_{t-1}+\boldsymbol{v}_t$; the two are equivalent after rescaling $\boldsymbol{v}$ by $\eta$ when $\eta$ is constant.) Unrolling the recursion shows that the update is $-\eta\sum_{s\le t}\beta^{t-s}\boldsymbol{g}_s$: a weighted sum of past gradients with an effective memory of about $1/(1-\beta)$ steps. For a constant gradient $\boldsymbol{g}$ the velocity saturates at $\boldsymbol{g}/(1-\beta)$, so momentum 0.9 gives an effective learning rate ten times larger along consistent directions. When you add momentum, you should usually lower $\eta$ accordingly.

{{figure: week03-momentum | Gradient descent and heavy-ball momentum on the same ill-conditioned quadratic. GD zigzags across the steep axis; momentum damps the oscillation while retaining speed along the shallow axis.}}

:::math Why momentum accelerates: the quadratic case
On $f=\tfrac12\lambda\theta^2$ (one eigendirection of a quadratic) the heavy-ball iteration in the form $\theta_{t+1}=\theta_t-\eta\lambda\theta_t+\beta(\theta_t-\theta_{t-1})$ is the linear recurrence $\theta_{t+1}=(1+\beta-\eta\lambda)\theta_t-\beta\theta_{t-1}$. Its characteristic polynomial is $z^2-(1+\beta-\eta\lambda)z+\beta=0$. When the roots are complex, which happens when $(1+\beta-\eta\lambda)^2<4\beta$, both roots have modulus exactly $\sqrt{\beta}$ regardless of $\lambda$. So for every curvature in that range the error decays at the same rate $\sqrt\beta$ per step (with oscillation). The range covers all $\lambda\in[\lambda_{\min},\lambda_{\max}]$ if $\eta$ and $\beta$ are tuned as
$$
\eta=\frac{4}{(\sqrt{\lambda_{\max}}+\sqrt{\lambda_{\min}})^2},\qquad \beta=\left(\frac{\sqrt\kappa-1}{\sqrt\kappa+1}\right)^2,
$$
giving a per-step rate of $\frac{\sqrt\kappa-1}{\sqrt\kappa+1}$. Compare that with $\frac{\kappa-1}{\kappa+1}$ for gradient descent: the number of iterations falls from $O(\kappa)$ to $O(\sqrt\kappa)$. With $\kappa=10^4$ that is the difference between 10,000 and 100 steps. This is the best possible rate for any first-order method on quadratics. Real networks are not quadratics, and the tuned values are unknowable in advance, so in practice $\beta=0.9$ is simply a good default.
:::

### Nesterov accelerated gradient {#nesterov}

Nesterov (1983) proposed a subtle change: evaluate the gradient not at the current position but at the **look-ahead** point where momentum alone is about to carry you.

$$
\begin{aligned}
\boldsymbol{v}_{t}&=\beta\,\boldsymbol{v}_{t-1}+\nabla\mathcal{L}(\boldsymbol{\theta}_{t-1}-\eta\beta\,\boldsymbol{v}_{t-1}),\\
\boldsymbol{\theta}_{t}&=\boldsymbol{\theta}_{t-1}-\eta\,\boldsymbol{v}_{t}.
\end{aligned}
$$

{{figure: week03-nesterov | Classical momentum takes the gradient at the current point; Nesterov takes it at the look-ahead point, so the correction already reflects what happens after the momentum step.}}

If the momentum step is about to overshoot, the look-ahead gradient already points back and reduces the velocity: it is a built-in brake. For smooth convex problems Nesterov's method achieves the optimal $O(1/t^2)$ rate versus $O(1/t)$ for plain gradient descent; for strongly convex quadratics the improvement over heavy ball is in the constants. In deep learning the gain is modest but consistent, and Sutskever et al. (2013) popularized it. In PyTorch, set `nesterov=True`, which implements a reformulation that needs only the gradient at the current iterate, so there is no extra backward pass.

```python:momentum_from_scratch.py
import numpy as np

# Ill-conditioned quadratic f(x, y) = 0.5 * (30 x^2 + y^2); curvatures 30 and 1.
H = np.array([30.0, 1.0])
grad = lambda th: H * th
f = lambda th: 0.5 * np.sum(H * th**2)

def run(kind, lr, beta=0.9, steps=60, start=(-1.6, 1.5)):
    th = np.array(start); v = np.zeros(2); hist = [f(th)]
    for _ in range(steps):
        if kind == "sgd":
            th = th - lr * grad(th)
        elif kind == "momentum":
            v = beta * v + grad(th)
            th = th - lr * v
        elif kind == "nesterov":
            v = beta * v + grad(th - lr * beta * v)   # gradient at the look-ahead point
            th = th - lr * v
        hist.append(f(th))
    return np.array(hist)

for kind, lr in [("sgd", 0.06), ("momentum", 0.015), ("nesterov", 0.015)]:
    h = run(kind, lr)
    print(f"{kind:9s} lr={lr:<6} loss after 20 steps {h[20]:.2e}, after 60 steps {h[60]:.2e}")
```

Note the largest stable plain-GD step is $2/30\approx0.067$, which is why the SGD run uses 0.06 and still zigzags: it contracts the steep coordinate by $|1-0.06\cdot30|=0.8$ per step while the shallow coordinate shrinks by only $0.94$ per step. Try the "Optimizer race" demo above with $\beta=0$ and $\beta=0.9$ to feel this difference.

## Adaptive learning rates: AdaGrad, RMSProp, Adam, AdamW {#adaptive-methods}

Momentum fixes the *direction* but still uses one global learning rate for every parameter. In a network, different parameters live at wildly different gradient scales (embedding rows of rare words versus common words, early layers versus late layers, biases versus weights). **Adaptive methods** keep a per-parameter running estimate of the typical gradient magnitude and divide the step by it, so every coordinate moves at a comparable speed.

### AdaGrad {#adagrad}

Duchi, Hazan and Singer (2011) accumulate the sum of squared gradients for each coordinate:

$$
\boldsymbol{s}_t=\boldsymbol{s}_{t-1}+\boldsymbol{g}_t\odot\boldsymbol{g}_t,\qquad \boldsymbol{\theta}_t=\boldsymbol{\theta}_{t-1}-\frac{\eta}{\sqrt{\boldsymbol{s}_t}+\varepsilon}\odot\boldsymbol{g}_t,
$$

where $\odot$ is elementwise multiplication and all operations on vectors are elementwise. Coordinates with large past gradients get smaller steps; rare features whose gradients are mostly zero keep a large step on the few occasions they appear. That is a wonderful property for sparse problems (bag-of-words models, embeddings). The weakness is that $\boldsymbol{s}_t$ only grows, so the effective learning rate $\eta/\sqrt{s_t}$ decays monotonically, roughly like $1/\sqrt{t}$ in the constant-gradient case, and in non-convex deep learning training can stall long before it converges.

### RMSProp {#rmsprop}

Geoffrey Hinton proposed, in a 2012 Coursera lecture (never formally published), replacing the sum by an exponential moving average so the denominator forgets old gradients:

$$
\boldsymbol{s}_t=\rho\,\boldsymbol{s}_{t-1}+(1-\rho)\,\boldsymbol{g}_t\odot\boldsymbol{g}_t,\qquad \boldsymbol{\theta}_t=\boldsymbol{\theta}_{t-1}-\frac{\eta}{\sqrt{\boldsymbol{s}_t}+\varepsilon}\odot\boldsymbol{g}_t,
$$

with $\rho\approx0.9$ (or 0.99). Now $\sqrt{\boldsymbol{s}_t}$ approximates the recent root-mean-square (RMS) gradient per coordinate, so the update has magnitude about $\eta$ per coordinate whatever the gradient scale. This is a diagonal approximation to a second-order method: dividing by the RMS gradient crudely rescales each coordinate toward unit curvature.

### Adam {#adam}

Adam (Kingma and Ba, 2014; ICLR 2015) combines momentum on the gradient (first moment) with RMSProp scaling (second moment), plus a bias correction:

{{figure: week03-adam-flow | One Adam step for a parameter tensor: update the two moving averages, correct their initialization bias, then take a per-coordinate step.}}

$$
\begin{aligned}
\boldsymbol{m}_t&=\beta_1\boldsymbol{m}_{t-1}+(1-\beta_1)\boldsymbol{g}_t,\\
\boldsymbol{v}_t&=\beta_2\boldsymbol{v}_{t-1}+(1-\beta_2)\,\boldsymbol{g}_t\odot\boldsymbol{g}_t,\\
\hat{\boldsymbol{m}}_t&=\frac{\boldsymbol{m}_t}{1-\beta_1^t},\qquad \hat{\boldsymbol{v}}_t=\frac{\boldsymbol{v}_t}{1-\beta_2^t},\\
\boldsymbol{\theta}_t&=\boldsymbol{\theta}_{t-1}-\eta\,\frac{\hat{\boldsymbol{m}}_t}{\sqrt{\hat{\boldsymbol{v}}_t}+\varepsilon}.
\end{aligned}
$$

The defaults $\beta_1=0.9$, $\beta_2=0.999$, $\varepsilon=10^{-8}$ and $\eta=10^{-3}$ are remarkably robust. The first moment $\boldsymbol{m}$ averages roughly the last 10 gradients (momentum), the second moment $\boldsymbol{v}$ the last 1000 (a smooth estimate of the gradient variance plus mean-square).

:::math Derivation: why the bias correction is needed
Initialize $\boldsymbol{m}_0=\mathbf{0}$. Unrolling the recursion gives $\boldsymbol{m}_t=(1-\beta_1)\sum_{i=1}^t\beta_1^{t-i}\boldsymbol{g}_i$. Suppose the gradients are drawn from a stationary distribution with mean $\boldsymbol{\mu}=\E[\boldsymbol{g}_i]$. Then
$$
\E[\boldsymbol{m}_t]=\boldsymbol{\mu}\,(1-\beta_1)\sum_{i=1}^t\beta_1^{t-i}=\boldsymbol{\mu}\,(1-\beta_1^t).
$$
So $\boldsymbol{m}_t$ is a biased (shrunk) estimate of $\boldsymbol{\mu}$ by the factor $1-\beta_1^t$, which is tiny at the start ($0.1$ at $t=1$ for $\beta_1=0.9$) and tends to 1 later. Dividing by it, $\hat{\boldsymbol{m}}_t=\boldsymbol{m}_t/(1-\beta_1^t)$, makes the estimator unbiased. For $\boldsymbol{v}$ the effect is far larger: with $\beta_2=0.999$ the factor is $0.001$ at $t=1$, so *without* correction $\sqrt{\boldsymbol{v}_1}=\sqrt{0.001}\,|\boldsymbol{g}_1|\approx0.03|\boldsymbol{g}_1|$ and the first step would be $\approx 30\times$ too large, since $\boldsymbol{m}_1/\sqrt{\boldsymbol{v}_1}=0.1\boldsymbol{g}_1/(0.032|\boldsymbol{g}_1|)\approx3.2$. With the correction, $\hat{\boldsymbol{m}}_1/\sqrt{\hat{\boldsymbol{v}}_1}=\boldsymbol{g}_1/|\boldsymbol{g}_1|=\pm1$, so the first update has magnitude exactly $\eta$ per coordinate. In general, with bias correction the per-coordinate step size is bounded by roughly $\eta$ (up to a factor related to $(1-\beta_1)/\sqrt{1-\beta_2}$ in the worst case), which is why Adam's learning rate has a direct meaning: "about how far each parameter may move per step".
:::

Because the update is a ratio $\hat m/\sqrt{\hat v}$ it is invariant to the overall scale of the gradient (multiply the loss by 1000 and Adam's trajectory is unchanged apart from $\varepsilon$). It behaves like a smoothed *sign* descent when gradients are noisy: if the mean $\mu$ is small relative to the RMS $\sqrt{\E g^2}$, the step shrinks automatically. This is a sensible signal-to-noise adaptation. Be aware of the known caveats: Reddi et al. (2018) constructed convex examples where Adam fails to converge (motivating AMSGrad), and Wilson et al. (2017) found that well-tuned SGD with momentum sometimes generalizes better than adaptive methods on image tasks. In practice Adam variants are the default for transformers, GANs, and most NLP, while SGD with momentum remains common in vision (ResNets).

```python:adam_from_scratch.py
import numpy as np

class Adam:
    """Adam (Kingma & Ba) with optional decoupled weight decay (AdamW).
    params: list of np.ndarray updated in place."""
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0, decoupled=True):
        self.p, self.lr, (self.b1, self.b2), self.eps = params, lr, betas, eps
        self.wd, self.decoupled, self.t = weight_decay, decoupled, 0
        self.m = [np.zeros_like(q) for q in params]
        self.v = [np.zeros_like(q) for q in params]

    def step(self, grads):
        self.t += 1
        for q, g, m, v in zip(self.p, grads, self.m, self.v):
            if self.wd and not self.decoupled:
                g = g + self.wd * q                    # classic L2: goes through the adaptive scaling
            m *= self.b1; m += (1 - self.b1) * g       # first moment (in place)
            v *= self.b2; v += (1 - self.b2) * g * g   # second moment
            m_hat = m / (1 - self.b1 ** self.t)        # bias corrections
            v_hat = v / (1 - self.b2 ** self.t)
            if self.wd and self.decoupled:
                q -= self.lr * self.wd * q             # AdamW: shrink weights directly
            q -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

# Sanity check: first step has size ~lr for every coordinate, whatever the gradient scale.
w = [np.zeros(3)]
opt = Adam(w, lr=0.1)
opt.step([np.array([1e-6, 1.0, 1e4])])
print(w[0])        # approximately [-0.1, -0.1, -0.1]
```

The check at the bottom shows scale invariance: gradients spanning ten orders of magnitude still produce steps of equal size, a defining property of Adam's first step. (For the $10^{-6}$ coordinate the value is slightly below $0.1$ because $\varepsilon=10^{-8}$ is no longer negligible compared with $|g|$.)

### AdamW: decoupled weight decay {#adamw}

**Weight decay** shrinks every weight a little at each step, $\boldsymbol{\theta}\leftarrow(1-\eta\lambda)\boldsymbol{\theta}$, to keep weights small. For plain SGD this is *identical* to adding an $\ell_2$ penalty $\frac\lambda2\norm{\boldsymbol{\theta}}^2$ to the loss, because the penalty's gradient is $\lambda\boldsymbol{\theta}$ and the step is $-\eta(\boldsymbol{g}+\lambda\boldsymbol{\theta})$. Early Adam implementations simply added $\lambda\boldsymbol{\theta}$ to the gradient, because people assumed the equivalence carries over. It does not.

:::math Why the equivalence breaks for Adam
With the $\ell_2$ penalty folded into the gradient, $\boldsymbol{g}_t'=\boldsymbol{g}_t+\lambda\boldsymbol{\theta}_{t-1}$, and the penalty term flows through both moment estimates. In the simplest case ($\beta_1=0$, ignoring $\varepsilon$) the update is
$$
\Delta\boldsymbol{\theta}=-\eta\,\frac{\boldsymbol{g}_t+\lambda\boldsymbol{\theta}}{\sqrt{\boldsymbol{v}_t'}}=-\eta\frac{\boldsymbol{g}_t}{\sqrt{\boldsymbol{v}_t'}}-\eta\lambda\frac{\boldsymbol{\theta}}{\sqrt{\boldsymbol{v}_t'}}.
$$
The effective decay of coordinate $j$ is $\eta\lambda/\sqrt{v'_{t,j}}$: parameters with large historical gradients (large $v$) are regularized *less*, while parameters with tiny gradients are regularized *more*. The strength of the regularizer is coupled to the gradient statistics of the loss, which has nothing to do with the intended penalty. Loshchilov and Hutter (2019) therefore **decouple** the two, applying the decay directly to the weights outside the adaptive pipeline:
$$
\boldsymbol{\theta}_t=\boldsymbol{\theta}_{t-1}-\eta\left(\frac{\hat{\boldsymbol{m}}_t}{\sqrt{\hat{\boldsymbol{v}}_t}+\varepsilon}+\lambda\,\boldsymbol{\theta}_{t-1}\right).
$$
Every weight is now shrunk by the same relative factor $1-\eta\lambda$ per step, independent of its gradient history. As a bonus, the best values of $\eta$ and $\lambda$ become less entangled, which makes hyperparameter search easier. (In the paper's formulation a schedule multiplier $\alpha_t$ multiplies both terms, so weight decay follows the schedule too.)
:::

AdamW is the default optimizer for training transformers and most modern models, typically with $\beta_2$ lowered to $0.95$ to $0.98$ for large language models and $\lambda\approx0.01$ to $0.1$. Weight decay should usually be *excluded* from biases and normalization parameters, as in the following PyTorch snippet.

```python:adamw_param_groups.py
import torch, torch.nn as nn

model = nn.Sequential(nn.Linear(128, 256), nn.LayerNorm(256), nn.GELU(), nn.Linear(256, 10))

decay, no_decay = [], []
for name, p in model.named_parameters():
    if not p.requires_grad:
        continue
    # biases and 1-D parameters (LayerNorm/BatchNorm gain and shift) are not decayed
    (no_decay if p.ndim < 2 else decay).append(p)

optimizer = torch.optim.AdamW(
    [{"params": decay, "weight_decay": 0.05},
     {"params": no_decay, "weight_decay": 0.0}],
    lr=3e-4, betas=(0.9, 0.999), eps=1e-8)

print([len(g["params"]) for g in optimizer.param_groups])   # [2, 3]: two weight matrices; two biases + LN gain/shift
```

:::history Where these methods came from
Momentum for optimization is due to Boris Polyak (1964); Yurii Nesterov's look-ahead method followed in 1983. AdaGrad (2011) was motivated by online learning and sparse features. RMSProp appeared in Hinton's 2012 Coursera lecture and was adopted everywhere without ever being written up as a paper. Adam was posted to arXiv in December 2014 (ICLR 2015) and became one of the most cited papers in machine learning. AdamW (arXiv 2017, ICLR 2019) fixed a subtle but important flaw that had been silently hurting regularization for years.
:::

## Learning-rate schedules {#lr-schedules}

The learning rate is the most important hyperparameter, and a constant value is almost never optimal. Early in training you want to move fast, late in training you want to settle into a minimum without bouncing around due to gradient noise. In the continuous-time picture, the stationary spread of the iterates around a minimum scales with $\eta/B$, so decaying $\eta$ is a way to shrink the noise ball and sharpen the final answer, which is why a sudden improvement in loss appears every time the learning rate drops.

```chart
{"type": "line", "caption": "Learning-rate multipliers over 1000 steps for six common schedules (computed with the formulas in this section).", "height": 320, "data": {"labels": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550, 560, 570, 580, 590, 600, 610, 620, 630, 640, 650, 660, 670, 680, 690, 700, 710, 720, 730, 740, 750, 760, 770, 780, 790, 800, 810, 820, 830, 840, 850, 860, 870, 880, 890, 900, 910, 920, 930, 940, 950, 960, 970, 980, 990], "datasets": [{"label": "constant", "data": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], "borderColor": "var(--c-blue)", "pointRadius": 0, "borderWidth": 2, "tension": 0}, {"label": "step", "data": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001], "borderColor": "var(--c-red)", "pointRadius": 0, "borderWidth": 2, "tension": 0}, {"label": "cosine", "data": [1.0, 0.9998, 0.999, 0.9978, 0.9961, 0.9938, 0.9911, 0.988, 0.9843, 0.9801, 0.9755, 0.9704, 0.9649, 0.9589, 0.9524, 0.9455, 0.9382, 0.9304, 0.9222, 0.9135, 0.9045, 0.8951, 0.8853, 0.8751, 0.8645, 0.8536, 0.8423, 0.8307, 0.8187, 0.8065, 0.7939, 0.781, 0.7679, 0.7545, 0.7409, 0.727, 0.7129, 0.6986, 0.6841, 0.6694, 0.6545, 0.6395, 0.6243, 0.6091, 0.5937, 0.5782, 0.5627, 0.5471, 0.5314, 0.5157, 0.5, 0.4843, 0.4686, 0.4529, 0.4373, 0.4218, 0.4063, 0.3909, 0.3757, 0.3605, 0.3455, 0.3306, 0.3159, 0.3014, 0.2871, 0.273, 0.2591, 0.2455, 0.2321, 0.219, 0.2061, 0.1935, 0.1813, 0.1693, 0.1577, 0.1464, 0.1355, 0.1249, 0.1147, 0.1049, 0.0955, 0.0865, 0.0778, 0.0696, 0.0618, 0.0545, 0.0476, 0.0411, 0.0351, 0.0296, 0.0245, 0.0199, 0.0157, 0.012, 0.0089, 0.0062, 0.0039, 0.0022, 0.001, 0.0002], "borderColor": "var(--c-green)", "pointRadius": 0, "borderWidth": 2, "tension": 0}, {"label": "warmup+cosine", "data": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.9997, 0.9988, 0.9973, 0.9951, 0.9924, 0.9891, 0.9851, 0.9806, 0.9755, 0.9698, 0.9636, 0.9568, 0.9494, 0.9415, 0.933, 0.924, 0.9145, 0.9045, 0.894, 0.883, 0.8716, 0.8597, 0.8473, 0.8346, 0.8214, 0.8078, 0.7939, 0.7796, 0.765, 0.75, 0.7347, 0.7192, 0.7034, 0.6873, 0.671, 0.6545, 0.6378, 0.621, 0.604, 0.5868, 0.5696, 0.5523, 0.5349, 0.5174, 0.5, 0.4826, 0.4651, 0.4477, 0.4304, 0.4132, 0.396, 0.379, 0.3622, 0.3455, 0.329, 0.3127, 0.2966, 0.2808, 0.2653, 0.25, 0.235, 0.2204, 0.2061, 0.1922, 0.1786, 0.1654, 0.1527, 0.1403, 0.1284, 0.117, 0.106, 0.0955, 0.0855, 0.076, 0.067, 0.0585, 0.0506, 0.0432, 0.0364, 0.0302, 0.0245, 0.0194, 0.0149, 0.0109, 0.0076, 0.0049, 0.0027, 0.0012, 0.0003], "borderColor": "var(--c-amber)", "pointRadius": 0, "borderWidth": 2, "tension": 0}, {"label": "one-cycle", "data": [0.1, 0.13, 0.16, 0.19, 0.22, 0.25, 0.28, 0.31, 0.34, 0.37, 0.4, 0.43, 0.46, 0.49, 0.52, 0.55, 0.58, 0.61, 0.64, 0.67, 0.7, 0.73, 0.76, 0.79, 0.82, 0.85, 0.88, 0.91, 0.94, 0.97, 1.0, 0.9995, 0.9982, 0.9959, 0.9928, 0.9887, 0.9838, 0.978, 0.9713, 0.9638, 0.9554, 0.9463, 0.9363, 0.9256, 0.9141, 0.9018, 0.8889, 0.8753, 0.861, 0.8461, 0.8306, 0.8145, 0.7979, 0.7808, 0.7632, 0.7452, 0.7269, 0.7081, 0.6891, 0.6697, 0.6501, 0.6304, 0.6104, 0.5903, 0.5702, 0.55, 0.5298, 0.5097, 0.4896, 0.4696, 0.4499, 0.4303, 0.4109, 0.3919, 0.3731, 0.3548, 0.3368, 0.3192, 0.3021, 0.2855, 0.2694, 0.2539, 0.239, 0.2247, 0.2111, 0.1982, 0.1859, 0.1744, 0.1637, 0.1537, 0.1446, 0.1362, 0.1287, 0.122, 0.1162, 0.1113, 0.1072, 0.1041, 0.1018, 0.1005], "borderColor": "var(--c-purple)", "pointRadius": 0, "borderWidth": 2, "tension": 0}, {"label": "triangular cyclic", "data": [0.1, 0.19, 0.28, 0.37, 0.46, 0.55, 0.64, 0.73, 0.82, 0.91, 1.0, 0.91, 0.82, 0.73, 0.64, 0.55, 0.46, 0.37, 0.28, 0.19, 0.1, 0.19, 0.28, 0.37, 0.46, 0.55, 0.64, 0.73, 0.82, 0.91, 1.0, 0.91, 0.82, 0.73, 0.64, 0.55, 0.46, 0.37, 0.28, 0.19, 0.1, 0.19, 0.28, 0.37, 0.46, 0.55, 0.64, 0.73, 0.82, 0.91, 1.0, 0.91, 0.82, 0.73, 0.64, 0.55, 0.46, 0.37, 0.28, 0.19, 0.1, 0.19, 0.28, 0.37, 0.46, 0.55, 0.64, 0.73, 0.82, 0.91, 1.0, 0.91, 0.82, 0.73, 0.64, 0.55, 0.46, 0.37, 0.28, 0.19, 0.1, 0.19, 0.28, 0.37, 0.46, 0.55, 0.64, 0.73, 0.82, 0.91, 1.0, 0.91, 0.82, 0.73, 0.64, 0.55, 0.46, 0.37, 0.28, 0.19], "borderColor": "var(--c-teal)", "pointRadius": 0, "borderWidth": 2, "tension": 0}]}, "options": {"scales": {"x": {"title": {"display": true, "text": "step"}}, "y": {"title": {"display": true, "text": "LR / peak LR"}}}}}
```

Before looking at specific schedules, here is why a schedule helps at all. Near a minimum with SGD noise, the stationary spread of the iterates scales roughly with $\eta/B$ (see the noise argument in the batch-size section): a large constant rate keeps the loss floating above the floor set by the noise. Decaying $\eta$ shrinks the noise ball and lets the model settle. Early on the opposite problem holds: you want the largest stable rate to move quickly through the flat, high-loss regions.

### The standard schedules

Write $\eta_0$ for the base (peak) rate, $t$ for the step and $T$ for the total number of steps.

- **Step decay.** $\eta_t=\eta_0\,\gamma^{\lfloor t/s\rfloor}$: multiply by $\gamma$ (commonly $0.1$) every $s$ steps. The classic ImageNet recipe drops the rate by 10x at epochs 30, 60 and 90. Simple and still competitive, but the drop points are two extra hyperparameters.
- **Cosine annealing** (Loshchilov and Hutter, 2017). $\eta_t=\eta_{\min}+\tfrac12(\eta_0-\eta_{\min})\big(1+\cos(\pi t/T)\big)$. It decays slowly at first, fastest in the middle, and gently toward the end, and needs only $T$. This is the dominant choice for modern vision and language training. (The original SGDR variant restarts the cosine periodically.)
- **Linear warmup.** $\eta_t=\eta_0\,t/T_w$ for $t<T_w$, usually followed by one of the decays above. Warmup is needed because at initialization the loss surface is at its sharpest and Adam's second-moment estimate is built from very few samples; a full-size step at $t=0$ can throw the weights into a region from which training never recovers. Transformers essentially always need it (a few hundred to a few thousand steps), and large-batch SGD needs it too (Goyal et al., 2017).
- **Cyclical learning rates** (Smith, 2017). The rate oscillates between $\eta_{\min}$ and $\eta_{\max}$, e.g. triangularly with half-period $s$: $\eta_t=\eta_{\min}+(\eta_{\max}-\eta_{\min})\max\!\big(0,\,1-|t/s-2k-1|\big)$ with $k=\lfloor t/(2s)\rfloor$. Periodically raising the rate helps cross saddles and flat regions; Smith also proposed the **LR range test**: sweep the rate exponentially upward during one short run and pick the range where the loss falls fastest.
- **One-cycle policy** (Smith and Topin, 2018, "super-convergence"). One triangle: warm up from $\eta_0/25$ (say) to a large peak over about 30 percent of training, then anneal (linearly or by cosine) to a rate far below the start, often $10^{-4}$ of the peak. Together with momentum cycled in the opposite direction, it can train some networks in a fraction of the usual epochs. PyTorch ships it as `OneCycleLR`.

{{demo: week03-lr-schedules | Learning-rate schedules: shape, warmup and the area under the curve | 560}}

All of these are a few lines of code. The version below returns a multiplier on the base rate, which is exactly what `torch.optim.lr_scheduler.LambdaLR` expects.

```python:lr_schedules.py
import math

def warmup_cosine(step, total, warmup, min_ratio=0.0):
    """LR multiplier: linear warmup for `warmup` steps, then cosine decay to min_ratio."""
    if step < warmup:
        return (step + 1) / warmup                       # never exactly 0, so the first step moves
    progress = (step - warmup) / max(1, total - warmup)  # 0 -> 1
    return min_ratio + 0.5 * (1 - min_ratio) * (1 + math.cos(math.pi * progress))

def step_decay(step, every, gamma=0.1):
    return gamma ** (step // every)

def triangular(step, half_period, lo=0.1):
    cycle = step // (2 * half_period)
    x = abs(step / half_period - 2 * cycle - 1)           # 1 -> 0 -> 1 within a cycle
    return lo + (1 - lo) * max(0.0, 1 - x)

T = 1000
for t in [0, 50, 99, 100, 300, 600, 999]:
    print(f"t={t:4d}  warmup+cosine {warmup_cosine(t, T, 100):.3f}  step {step_decay(t, 300):.3f}  tri {triangular(t, 100):.3f}")

# Using it in PyTorch (sketch):
#   opt   = torch.optim.AdamW(model.parameters(), lr=3e-4)
#   sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: warmup_cosine(s, T, 100))
#   ... each iteration:  opt.step(); sched.step()          # step the scheduler per iteration, not per epoch
```

:::tip Choosing a schedule
Default to **linear warmup followed by cosine decay** (to about 10 percent of peak or to zero) and pick the peak rate with a quick LR range test. Make the schedule a function of the *total* number of steps, so changing the training length rescales the whole curve. If you stop a cosine run early you stop at a large learning rate and the result is poor: decide the budget first.
:::

## Weight initialization {#initialization}

A network's first gradient step is determined entirely by its initial weights, and so is whether the signal survives the trip through dozens of layers. Initialization is therefore not a detail. The principle is simple: **choose the weight variance so that the scale of activations (forward) and of gradients (backward) stays roughly constant from layer to layer.** If each layer multiplies the variance by 2 the signal grows as $2^L$; by $\tfrac12$, it shrinks as $2^{-L}$. At $L=30$ that is a factor of a billion either way.

### Deriving Xavier/Glorot initialization

Consider a dense layer $z_j=\sum_{i=1}^{n_{\rm in}}w_{ij}a_i$ (ignore the bias, initialized at 0), with $a=\phi(z_{\rm prev})$. Assume the weights are i.i.d. with mean 0 and variance $\sigma_w^2$, independent of the inputs, and that the inputs $a_i$ are i.i.d.

:::math Forward variance
Because the terms are independent and zero-mean,
$$
\Var(z_j)=\sum_{i=1}^{n_{\rm in}}\Var(w_{ij}a_i)=n_{\rm in}\,\sigma_w^2\,\E[a_i^2].
$$
Here we used $\E[w_{ij}a_i]=0$ and $\Var(w_{ij}a_i)=\E[w^2]\E[a^2]$. For a symmetric activation that is approximately linear near zero (tanh, or no activation), $\E[a^2]\approx\Var(z_{\rm prev})$, so to keep the variance constant from layer to layer we need
$$
n_{\rm in}\,\sigma_w^2=1.
$$
:::

:::math Backward variance
In the backward pass the gradient with respect to a layer's input is $\bar a_i=\sum_{j=1}^{n_{\rm out}}w_{ij}\bar z_j$, a sum of $n_{\rm out}$ terms. The same computation gives $\Var(\bar a_i)=n_{\rm out}\,\sigma_w^2\,\Var(\bar z)$ (taking $\phi'\approx1$), so keeping gradient variance constant needs
$$
n_{\rm out}\,\sigma_w^2=1.
$$
:::

When $n_{\rm in}\neq n_{\rm out}$ we cannot satisfy both. Glorot and Bengio (2010) split the difference with the harmonic compromise

$$
\sigma_w^2=\frac{2}{n_{\rm in}+n_{\rm out}},\qquad\text{i.e. } W_{ij}\sim\mathcal N\!\Big(0,\tfrac{2}{n_{\rm in}+n_{\rm out}}\Big)\ \text{ or }\ \mathrm{Unif}\!\big[-a,a\big],\ a=\sqrt{\tfrac{6}{n_{\rm in}+n_{\rm out}}}.
$$

(The uniform bound follows from $\Var(\mathrm{Unif}[-a,a])=a^2/3$.) This is `nn.init.xavier_normal_` / `xavier_uniform_`. LeCun's earlier rule $\sigma_w^2=1/n_{\rm in}$ matches the forward condition only.

### Deriving He (Kaiming) initialization

ReLU breaks the "approximately linear" assumption: it zeroes half the pre-activations. If $z$ is symmetric around zero, then $\E[\relu(z)^2]=\tfrac12\E[z^2]=\tfrac12\Var(z)$. Redoing the forward computation:

:::math He variance
$$
\Var(z_j)=n_{\rm in}\,\sigma_w^2\,\E[a^2]=n_{\rm in}\,\sigma_w^2\cdot\tfrac12\Var(z_{\rm prev}).
$$
Constant variance requires $\tfrac12 n_{\rm in}\sigma_w^2=1$, that is
$$
\sigma_w^2=\frac{2}{n_{\rm in}}.
$$
The same argument applies backward (only half the units pass gradient), giving $2/n_{\rm out}$; He et al. (2015) show that either choice keeps the *product over layers* bounded, and use $n_{\rm in}$ ("fan-in" mode) by default.
:::

{{figure: week03-init-variance | Variance of activations through depth. If each layer scales the variance by a factor other than 1, the signal vanishes or explodes exponentially; Xavier and He choose the weight variance that makes the factor exactly 1 for tanh and ReLU respectively.}}

Here is that effect measured. The chart shows the activation standard deviation per layer in a 30-layer network for several choices:

```chart
{"type":"line","caption":"Standard deviation of the activations at each layer of a 30-layer, width-256 network with Gaussian weights, unit-variance Gaussian inputs, batch 512 (computed with NumPy, one seed). Only the variance-preserving choices (ReLU with He, tanh with Glorot-like or slightly larger scale) stay in a usable range; ReLU with a too-small variance collapses by many orders of magnitude, a factor 2 off per layer is already fatal at depth 30.","height":360,"data":{"labels":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30],"datasets":[{"label":"ReLU, He (var 2/n)","data":[0.8267,0.8329,0.8688,0.8523,0.8594,1.005,1.105,1.104,1.038,0.8872,0.8974,0.9753,0.9483,0.9256,0.8941,0.9122,0.995,1.016,1.134,1.05,1.104,1.144,1.082,1.154,1.017,1.075,1.186,1.237,1.212,1.212],"borderColor":"var(--c-blue)","borderWidth":2,"pointRadius":0},{"label":"ReLU, Glorot (var 1/n)","data":[0.5829,0.4014,0.2892,0.2028,0.1408,0.1047,0.06998,0.0495,0.03741,0.02471,0.01818,0.01172,0.007827,0.005923,0.004205,0.003273,0.002108,0.001527,0.001109,0.0007738,0.0004647,0.000292,0.0001946,0.0001294,0.0001019,7.911e-05,5.534e-05,4.353e-05,2.922e-05,1.961e-05],"borderColor":"var(--c-green)","borderWidth":2,"pointRadius":0},{"label":"ReLU, std 0.05","data":[0.4706,0.2659,0.1439,0.07905,0.04261,0.02522,0.01503,0.008094,0.004658,0.002458,0.001491,0.0008458,0.0005043,0.000283,0.0001721,9.798e-05,5.21e-05,3.15e-05,1.604e-05,8.909e-06,5.196e-06,2.637e-06,1.409e-06,8.743e-07,5.103e-07,2.706e-07,1.486e-07,8.853e-08,4.598e-08,2.811e-08],"borderColor":"var(--c-red)","borderWidth":2,"pointRadius":0},{"label":"tanh, Glorot (var 1/n)","data":[0.6275,0.4847,0.4054,0.3562,0.3191,0.2916,0.2732,0.2569,0.2444,0.2286,0.2154,0.2046,0.1978,0.191,0.185,0.1808,0.1744,0.1666,0.1605,0.1564,0.151,0.1464,0.1422,0.1367,0.1324,0.1305,0.1307,0.1305,0.1308,0.1303],"borderColor":"var(--c-purple)","borderWidth":2,"pointRadius":0},{"label":"tanh, std 0.05","data":[0.5605,0.3838,0.2829,0.2153,0.1669,0.129,0.1008,0.08031,0.0635,0.05059,0.04019,0.03212,0.02503,0.02031,0.01645,0.01297,0.01005,0.007923,0.006405,0.005173,0.004161,0.00327,0.002577,0.002045,0.001646,0.001309,0.001065,0.0008651,0.0006918,0.0005513],"borderColor":"var(--c-amber)","borderWidth":2,"pointRadius":0},{"label":"tanh, std 0.2","data":[0.8715,0.8525,0.8502,0.8493,0.8493,0.8481,0.8475,0.8497,0.8502,0.8518,0.8514,0.8506,0.8493,0.8487,0.8497,0.8493,0.8486,0.8475,0.8496,0.8477,0.8491,0.8499,0.8497,0.8482,0.8496,0.8491,0.8492,0.8502,0.8497,0.8496],"borderColor":"var(--c-teal)","borderWidth":2,"pointRadius":0}]},"options":{"scales":{"x":{"title":{"display":true,"text":"layer"}},"y":{"type":"logarithmic","title":{"display":true,"text":"activation std"}}}}}
```

{{demo: week03-init-explorer | Initialization explorer: activations and gradients versus depth | 560}}

```python:init_check.py
import torch, torch.nn as nn

def activation_stds(init, act, depth=30, width=256, batch=512):
    torch.manual_seed(0)
    x, stds = torch.randn(batch, width), []
    for _ in range(depth):
        lin = nn.Linear(width, width, bias=False)
        init(lin.weight)
        x = act(lin(x))
        stds.append(x.std().item())
    return stds

he = lambda w: nn.init.kaiming_normal_(w, nonlinearity="relu")   # std = sqrt(2 / fan_in)
glorot = lambda w: nn.init.xavier_normal_(w)                      # std = sqrt(2 / (fan_in + fan_out))
naive = lambda w: nn.init.normal_(w, 0.0, 0.05)

for name, init, act in [("relu + He", he, torch.relu), ("relu + naive 0.05", naive, torch.relu),
                        ("tanh + Glorot", glorot, torch.tanh), ("tanh + naive 0.05", naive, torch.tanh)]:
    s = activation_stds(init, act)
    print(f"{name:20s} layer 1: {s[0]:.2e}   layer 10: {s[9]:.2e}   layer 30: {s[29]:.2e}")
```

Practical defaults: **He** for ReLU-family layers, **Glorot** for tanh/sigmoid/linear, biases at zero, and for transformers a small normal (std about 0.02) with residual-branch outputs scaled down by $1/\sqrt{2L}$ so that the sum of many residual contributions does not grow with depth. PyTorch's `nn.Linear` default is a Kaiming-uniform with $a=\sqrt5$, which amounts to variance $1/(3n_{\rm in})$, deliberately conservative; it is fine for shallow nets but means the default is not He. Initialization derivations assume the layer inputs are roughly independent and zero-mean; normalization layers (below) make training far less sensitive to these choices.

## Vanishing and exploding gradients {#vanishing-exploding}

[Week 2](week-02.html#backpropagation-derived) showed that the gradient reaching layer $k$ is a product of per-layer Jacobians $\prod_{\ell>k}\diag(\phi'(\boldsymbol z_\ell))\mathbf W_\ell^\top$. The norm of a product of $L$ matrices tends to grow or shrink *exponentially* in $L$, governed by the typical singular value $\rho$ of each factor: the gradient scales like $\rho^{L-k}$. If $\rho<1$ the early layers get no learning signal (**vanishing**); if $\rho>1$ the gradient blows up and a single step can destroy the model (**exploding**).

Where each comes from:

- **Saturating activations.** $\sigma'\le0.25$ and $\tanh'\to0$ in the tails give factors below 1 (Hochreiter 1991; Bengio, Simard and Frasconi 1994).
- **Bad initialization.** Weight variance too small shrinks signals; too large inflates them and also saturates sigmoid/tanh units.
- **Depth and recurrence.** In an RNN the *same* $\mathbf W$ is applied at every time step, so the product is $\mathbf W^T$ and its behavior is set by the largest eigenvalue: above 1 explodes, below 1 vanishes (see [Week 7](week-07.html)).

How to diagnose it: log the gradient norm of each parameter group per step. A healthy network has comparable relative gradient magnitudes ($\norm{g}/\norm{\theta}$ around $10^{-3}$) across layers. Gradient norms that spike by orders of magnitude, or `nan`, indicate explosion; first-layer norms many orders below the last layer's indicate vanishing (compare the bar chart in [Week 2](week-02.html#gradient-flow-in-deep-networks)).

The toolbox, in order of impact: proper initialization; non-saturating activations (ReLU family, GELU); **normalization layers**; **residual connections** (the identity path gives a Jacobian factor near 1, which is why ResNets train at depth 100+, [Week 5](week-05.html)); gated recurrent cells (LSTM/GRU); and, for explosions, **gradient clipping**.

### Gradient clipping

Clipping bounds the size of an update without changing its direction. Two forms:

- **Clip by value:** clamp each component to $[-c,c]$. Cheap but distorts the direction.
- **Clip by global norm** (Pascanu, Mikolov and Bengio, 2013): treat all gradients as one vector $\boldsymbol g$ and rescale

$$
\boldsymbol g\leftarrow\boldsymbol g\cdot\min\!\Big(1,\ \frac{c}{\norm{\boldsymbol g}_2}\Big).
$$

If $\norm{\boldsymbol g}\le c$ nothing happens; otherwise the update has norm exactly $c$. A good threshold is a little above the typical gradient norm (log it for a few hundred steps and use something like the 90th percentile); $c=1.0$ is the near-universal default for transformers and RNNs. Clipping is a safety net against rare spikes, not a cure for a learning rate that is too high: if clipping is active on most steps the effective learning rate is simply $c/\norm{g}$ times smaller, and you should lower $\eta$ instead.

```python:grad_clipping.py
import torch, torch.nn as nn

torch.manual_seed(0)
model = nn.Sequential(nn.Linear(10, 32), nn.Tanh(), nn.Linear(32, 1))
opt = torch.optim.SGD(model.parameters(), lr=0.1)
x, y = torch.randn(64, 10), torch.randn(64, 1)
y[0] += 500.0                                     # one corrupted target: an outlier-driven gradient spike

for step in range(3):
    opt.zero_grad()
    loss = nn.functional.mse_loss(model(x), y)
    loss.backward()
    # returns the norm BEFORE clipping; use it for logging
    total_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    after = torch.sqrt(sum(p.grad.pow(2).sum() for p in model.parameters()))
    print(f"step {step}: loss {loss.item():9.2f}  grad norm before {total_norm:9.2f}  after {after:.2f}")
    opt.step()
```

Remember that with Adam, clipping interacts with the second-moment estimate (clipped gradients feed $v_t$), and that **clipping must happen after `backward()` and before `optimizer.step()`**. With mixed precision and gradient scaling, call `scaler.unscale_(optimizer)` first so you clip the true gradients.

## Normalization layers {#normalization}

Initialization sets the scale of activations at step 0, but nothing keeps it there as the weights change. **Normalization layers** fix this by explicitly standardizing activations inside the network. They are among the most effective training aids ever found: they allow learning rates several times larger, reduce sensitivity to initialization and make very deep networks trainable.

### Batch normalization {#batchnorm}

Ioffe and Szegedy (2015) normalize each feature using the statistics of the current mini-batch. For a layer output $\mathbf X\in\R^{B\times C}$ (for images, $B\times C\times H\times W$ with statistics taken over $B,H,W$ for each channel):

$$
\mu_c=\frac1B\sum_{n}x_{nc},\qquad \sigma_c^2=\frac1B\sum_n(x_{nc}-\mu_c)^2,\qquad
\hat x_{nc}=\frac{x_{nc}-\mu_c}{\sqrt{\sigma_c^2+\varepsilon}},\qquad y_{nc}=\gamma_c\,\hat x_{nc}+\beta_c .
$$

The learnable $\gamma_c,\beta_c$ let the network undo the normalization if that is best (so no expressive power is lost). Typically BN sits between the linear layer and the nonlinearity (original paper) or after the nonlinearity; both work, and with ReLU the former is the convention.

**Train versus eval.** During training the statistics come from the current batch, which makes each output depend on the other examples in the batch (a mild source of noise that also regularizes). At test time you want a deterministic function of one input, so BN uses **running averages** of the training means and variances, updated as $\mu_{\rm run}\leftarrow(1-m)\mu_{\rm run}+m\,\mu_{\rm batch}$ with momentum $m=0.1$ in PyTorch. Forgetting `model.eval()` is the classic BN bug.

{{figure: week03-norm-axes | Which axes each normalizer averages over for an activation tensor of shape $(N, C, H, W)$: BatchNorm over $N,H,W$ per channel; LayerNorm over $C,H,W$ per sample; GroupNorm over groups of channels (and $H,W$) per sample; InstanceNorm per sample and channel.}}

:::math Backward pass of BatchNorm
Let $\bar y$ denote $\partial\mathcal L/\partial y$ for one channel, and $s=1/\sqrt{\sigma^2+\varepsilon}$. Then $\bar\gamma=\sum_n\bar y_n\hat x_n$, $\bar\beta=\sum_n\bar y_n$ and $\bar{\hat x}_n=\gamma\bar y_n$. Since every $x_n$ affects $\hat x_m$ for all $m$ through $\mu$ and $\sigma^2$, three paths contribute:

- directly: $\bar{\hat x}_n\,s$;
- through $\mu$: $\partial\mu/\partial x_n=1/B$ and $\partial\hat x_m/\partial\mu=-s$ (plus a term through $\sigma^2$ that vanishes because $\sum_m(x_m-\mu)=0$), giving $-\frac sB\sum_m\bar{\hat x}_m$;
- through $\sigma^2$: $\partial\sigma^2/\partial x_n=\tfrac2B(x_n-\mu)$ and $\partial\hat x_m/\partial\sigma^2=-\tfrac12(x_m-\mu)s^3$, giving $-\frac sB\hat x_n\sum_m\bar{\hat x}_m\hat x_m$.

Adding the three terms,
$$
\bar x_n=\frac{s}{B}\Big(B\,\bar{\hat x}_n-\sum_m\bar{\hat x}_m-\hat x_n\sum_m\bar{\hat x}_m\hat x_m\Big).
$$
The gradient is the incoming gradient with its batch mean removed and its component along $\hat x$ removed: BN projects out the two directions (shift and scale) along which the output cannot change.
:::

```python:batchnorm_numpy.py
import numpy as np

class BatchNorm1d:
    """BatchNorm for (B, C) inputs: normalize each feature over the batch, then scale and shift."""
    def __init__(self, C, momentum=0.1, eps=1e-5):
        self.gamma, self.beta = np.ones(C), np.zeros(C)
        self.run_mean, self.run_var = np.zeros(C), np.ones(C)
        self.mom, self.eps = momentum, eps

    def forward(self, x, train=True):
        if train:
            mu, var = x.mean(axis=0), x.var(axis=0)             # biased variance (divide by B)
            self.run_mean = (1 - self.mom) * self.run_mean + self.mom * mu
            self.run_var = (1 - self.mom) * self.run_var + self.mom * var * len(x) / (len(x) - 1)  # unbiased for running
        else:
            mu, var = self.run_mean, self.run_var               # eval: fixed statistics
        self.inv_std = 1.0 / np.sqrt(var + self.eps)
        self.xhat = (x - mu) * self.inv_std
        return self.gamma * self.xhat + self.beta

    def backward(self, dy):
        B = len(dy)
        self.dgamma = (dy * self.xhat).sum(axis=0)
        self.dbeta = dy.sum(axis=0)
        dxhat = dy * self.gamma
        # compact form: dx = inv_std/B * (B*dxhat - sum(dxhat) - xhat * sum(dxhat * xhat))
        return self.inv_std / B * (B * dxhat - dxhat.sum(axis=0) - self.xhat * (dxhat * self.xhat).sum(axis=0))

rng = np.random.default_rng(0)
x = rng.normal(2.0, 3.0, (8, 5)); w = rng.normal(size=(8, 5))      # w makes the scalar "loss" non-trivial
bn = BatchNorm1d(5); bn.gamma = rng.normal(size=5); bn.beta = rng.normal(size=5)
loss = lambda x_: (bn.forward(x_) * w).sum()
bn.forward(x); dx = bn.backward(w)                                  # upstream gradient of that loss is w
num = np.zeros_like(x)
for i in np.ndindex(*x.shape):                                      # centered finite differences
    xp, xm = x.copy(), x.copy(); xp[i] += 1e-6; xm[i] -= 1e-6
    num[i] = (loss(xp) - loss(xm)) / 2e-6
print("max |analytic - numeric| =", np.abs(dx - num).max())        # about 1e-9
```

**Why does it help?** The original explanation, "reducing internal covariate shift" (the input distribution of each layer drifting as earlier layers change), is intuitive but was challenged: Santurkar et al. (2018) showed that injecting noise to restore covariate shift after BN does not remove the benefit, and argued that BN instead makes the loss landscape smoother (smaller Lipschitz constants of loss and gradients), which allows larger steps. A complementary view: BN makes the layer output invariant to the scale of the incoming weights, so the effective learning rate adapts automatically (the weight norm grows, the effective step shrinks; see Arora et al., 2019, and the reading list). Treat the covariate-shift story as historical motivation and the smoothing/scale-invariance story as the better-supported mechanism. What is *not* in dispute is the practical effect.

**When BN fails.** It needs reasonably large batches (with $B=2$ the statistics are mostly noise), it behaves differently in train and eval, it couples the examples in a batch (a problem for contrastive losses and for metric learning if statistics leak across samples), and it is awkward for sequence models with variable lengths. These problems motivate the alternatives.

### Layer normalization {#layernorm}

Ba, Kiros and Hinton (2016) normalize over the **features of a single example**, so nothing depends on the batch: $\mu_n=\frac1C\sum_c x_{nc}$, $\sigma_n^2=\frac1C\sum_c(x_{nc}-\mu_n)^2$, followed by a per-feature gain and bias. Train and eval behave identically, and it works for any batch size, including 1, and for sequences. It is the standard in transformers ([Week 9](week-09.html)), where it is applied per token over the model dimension. **RMSNorm** (Zhang and Sennrich, 2019) drops the mean subtraction and divides only by the root-mean-square, $x/\sqrt{\frac1C\sum_c x_c^2+\varepsilon}\cdot g$, which is a little cheaper and used in many recent large language models.

### Group normalization {#groupnorm}

Wu and He (2018) split the $C$ channels into $G$ groups and normalize within each group, per sample. $G=1$ recovers LayerNorm (over all of $C,H,W$), $G=C$ recovers InstanceNorm. With $G=32$ it matches BN at large batch sizes and, unlike BN, does not degrade with batch size 2 or 4; it is the usual choice for detection and segmentation with large images and tiny batches, and for generative models.

| Normalizer | Statistics over | Batch-dependent? | Typical home |
| --- | --- | --- | --- |
| BatchNorm | batch and spatial positions, per channel | yes (and train/eval differ) | CNNs with large batches |
| LayerNorm | all features of one sample (one token) | no | Transformers, RNNs |
| GroupNorm | channel groups of one sample | no | Small-batch vision, diffusion U-Nets |
| InstanceNorm | one channel of one sample | no | Style transfer |
| RMSNorm | features of one sample, no centering | no | Modern LLMs |

{{demo: week03-norm-explorer | Normalization explorer: which statistics each method uses | 560}}

```python:norm_layers.py
import torch, torch.nn as nn

x = torch.randn(8, 32, 14, 14) * 3 + 5               # (N, C, H, W) with non-zero mean and large scale

for name, layer in [("BatchNorm2d", nn.BatchNorm2d(32)),
                    ("LayerNorm  ", nn.LayerNorm([32, 14, 14])),
                    ("GroupNorm  ", nn.GroupNorm(8, 32))]:
    y = layer(x)
    # BN: each channel ~ N(0,1) over (N,H,W). LN/GN: each sample (or group) ~ N(0,1) over its own axes.
    print(name, "per-channel mean (c=0):", y[:, 0].mean().item(), " per-sample mean (n=0):", y[0].mean().item())

bn = nn.BatchNorm2d(32)
bn.train(); _ = bn(x)                                  # updates running_mean / running_var
print("running_mean[0] after one batch (momentum 0.1 -> ~0.1 * 5):", bn.running_mean[0].item())
bn.eval()
y1 = bn(x[:1]); y2 = bn(x[:1])                         # eval: deterministic, no dependence on other samples
print("eval is deterministic:", torch.equal(y1, y2))
```

## Lab: compare SGD, momentum and Adam {#lab}

The lab turns every idea of the week into a controlled experiment, in pure NumPy so you can see every line. We train the same 2-64-64-3 ReLU network (the one from [Week 2](week-02.html#lab-a-neural-network-from-scratch)) on a three-arm spiral with mini-batches of 64, change **one thing at a time**, and read the loss curves. Everything below, including the numbers and charts, is produced by this code.

```python:optimizer_lab.py
import numpy as np, json

def make_spiral(n=200, k=3, noise=0.2, seed=0):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for c in range(k):
        r = np.linspace(0.05, 1, n); t = np.linspace(c * 4, (c + 1) * 4, n) + rng.normal(0, noise, n)
        X.append(np.c_[r * np.sin(t), r * np.cos(t)]); y += [c] * n
    return np.vstack(X), np.array(y)

def init_params(sizes, scheme, seed=0):
    rng = np.random.default_rng(seed)
    Ws, bs = [], []
    for m, n in zip(sizes[:-1], sizes[1:]):
        std = {"he": np.sqrt(2 / m), "glorot": np.sqrt(2 / (m + n)),
               "small": 0.01, "large": 1.0}[scheme]
        Ws.append(rng.normal(0, std, (m, n))); bs.append(np.zeros(n))
    return Ws + bs                                   # flat list: weights then biases

def loss_and_grads(P, X, y):
    L = len(P) // 2; W, b = P[:L], P[L:]
    A = [X]; Z = []
    for l in range(L):
        Z.append(A[-1] @ W[l] + b[l]); A.append(np.maximum(Z[-1], 0) if l < L - 1 else Z[-1])
    s = A[-1] - A[-1].max(1, keepdims=True); p = np.exp(s); p /= p.sum(1, keepdims=True)
    loss = -np.log(p[np.arange(len(y)), y] + 1e-12).mean()
    d = p.copy(); d[np.arange(len(y)), y] -= 1; d /= len(y)
    gW, gb = [None] * L, [None] * L
    for l in reversed(range(L)):
        gW[l] = A[l].T @ d; gb[l] = d.sum(0)
        if l > 0: d = (d @ W[l].T) * (Z[l - 1] > 0)
    return loss, gW + gb

class SGD:
    def __init__(self, P, lr, momentum=0.0): self.P, self.lr, self.mu = P, lr, momentum; self.v = [np.zeros_like(p) for p in P]
    def step(self, G):
        for p, g, v in zip(self.P, G, self.v):
            v *= self.mu; v += g; p -= self.lr * v

class Adam:
    def __init__(self, P, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.P, self.lr, self.b1, self.b2, self.eps, self.t = P, lr, b1, b2, eps, 0
        self.m = [np.zeros_like(p) for p in P]; self.v = [np.zeros_like(p) for p in P]
    def step(self, G):
        self.t += 1
        for p, g, m, v in zip(self.P, G, self.m, self.v):
            m *= self.b1; m += (1 - self.b1) * g; v *= self.b2; v += (1 - self.b2) * g * g
            p -= self.lr * (m / (1 - self.b1 ** self.t)) / (np.sqrt(v / (1 - self.b2 ** self.t)) + self.eps)

def run(make_opt, scheme="he", epochs=60, bs=64, seed=0):
    X, y = make_spiral(); P = init_params([2, 64, 64, 3], scheme, seed); opt = make_opt(P)
    rng = np.random.default_rng(seed); curve = []
    for ep in range(epochs):
        perm = rng.permutation(len(X)); tot = 0.0
        for i in range(0, len(X), bs):
            idx = perm[i:i + bs]; loss, G = loss_and_grads(P, X[idx], y[idx]); opt.step(G); tot += loss * len(idx)
        curve.append(tot / len(X))
    return curve
```

Now the three experiments.

```python:run_lab.py
# Experiment 1: three optimizers, same init and data order.
runs = {
    "SGD lr 0.05":          run(lambda P: SGD(P, 0.05)),
    "SGD+momentum lr 0.05": run(lambda P: SGD(P, 0.05, momentum=0.9)),
    "Adam lr 0.003":        run(lambda P: Adam(P, 0.003)),
}
for k, c in runs.items():
    print(f"{k:22s} epoch 1: {c[0]:.3f}   epoch 10: {c[9]:.3f}   epoch 30: {c[29]:.3f}   epoch 60: {c[59]:.3f}")

# Experiment 2: initialization (SGD + momentum)
for scheme in ["he", "glorot", "small", "large"]:
    c = run(lambda P: SGD(P, 0.05, momentum=0.9), scheme=scheme)
    print(f"{scheme:7s} epoch 1: {c[0]:.3f}  epoch 10: {c[9]:.3f}  epoch 60: {c[59]:.3f}")

# Experiment 3: learning-rate sweep on a log grid
for lr in [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]:
    print(f"lr {lr:<7} momentum: {run(lambda P: SGD(P, lr, 0.9), epochs=30)[-1]:.4f}   adam: {run(lambda P: Adam(P, lr), epochs=30)[-1]:.4f}")
```

Typical output (one seed, so exact digits will differ across machines only in the last digits):

```text
SGD lr 0.05            epoch 1: 1.006   epoch 10: 0.630   epoch 30: 0.321   epoch 60: 0.126
SGD+momentum lr 0.05   epoch 1: 0.914   epoch 10: 0.085   epoch 30: 0.017   epoch 60: 0.006
Adam lr 0.003          epoch 1: 0.896   epoch 10: 0.224   epoch 30: 0.020   epoch 60: 0.005
he      epoch 1: 0.914  epoch 10: 0.085  epoch 60: 0.006
glorot  epoch 1: 1.072  epoch 10: 0.441  epoch 60: 0.012
small   epoch 1: 1.099  epoch 10: 1.100  epoch 60: 0.424
large   epoch 1: 2.507  epoch 10: 0.010  epoch 60: 0.001
```

```chart
{"type":"line","caption":"Mean training loss per epoch on the three-arm spiral (600 points, 2-64-64-3 ReLU MLP, He init, batch 64, 60 epochs, one seed; produced by the lab code above). Momentum and Adam reach a loss near 0.005 while plain SGD at the same rate is still at 0.13. A single seed is illustrative.","height":340,"data":{"labels":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60],"datasets":[{"label":"SGD lr 0.05","data":[1.0056,0.8499,0.7853,0.7498,0.7207,0.6993,0.6818,0.664,0.6457,0.6299,0.6118,0.5968,0.5799,0.5613,0.5443,0.5255,0.5092,0.4943,0.479,0.463,0.446,0.4311,0.416,0.4023,0.3859,0.3716,0.3594,0.3449,0.3338,0.3208,0.3096,0.2979,0.2871,0.2775,0.2664,0.2573,0.2485,0.2393,0.2313,0.2231,0.2168,0.2087,0.2032,0.1966,0.1895,0.1836,0.1782,0.1728,0.1675,0.164,0.1588,0.1544,0.1503,0.1463,0.1426,0.139,0.1358,0.1321,0.1291,0.1262],"borderColor":"var(--c-blue)","borderWidth":2,"pointRadius":0},{"label":"SGD+momentum lr 0.05","data":[0.9141,0.7005,0.5739,0.4649,0.3437,0.2476,0.1771,0.1299,0.1029,0.0852,0.0738,0.0648,0.0556,0.0516,0.0478,0.039,0.0361,0.0355,0.0293,0.0287,0.0251,0.0248,0.0222,0.0212,0.0198,0.0185,0.0175,0.0165,0.0166,0.017,0.0179,0.0143,0.0142,0.0134,0.0126,0.0119,0.0121,0.0123,0.0114,0.0115,0.0103,0.0095,0.009,0.0089,0.0088,0.0081,0.0081,0.0082,0.0076,0.0073,0.0077,0.0073,0.007,0.0064,0.0067,0.0065,0.0061,0.0061,0.0061,0.0056],"borderColor":"var(--c-green)","borderWidth":2,"pointRadius":0},{"label":"Adam lr 0.003","data":[0.8959,0.7124,0.6545,0.597,0.5396,0.4772,0.4121,0.3444,0.2769,0.2242,0.1821,0.1493,0.1242,0.1004,0.0861,0.0738,0.0642,0.0604,0.0507,0.046,0.0412,0.0384,0.034,0.0316,0.0294,0.0269,0.0248,0.0226,0.0216,0.0201,0.0196,0.0175,0.0169,0.0169,0.0146,0.0144,0.0139,0.0125,0.0118,0.0117,0.011,0.0099,0.0095,0.009,0.0088,0.0083,0.0079,0.0078,0.0073,0.007,0.0068,0.0065,0.0062,0.0059,0.0058,0.0056,0.0053,0.0052,0.005,0.0047],"borderColor":"var(--c-red)","borderWidth":2,"pointRadius":0}]},"options":{"scales":{"x":{"title":{"display":true,"text":"epoch"}},"y":{"type":"logarithmic","title":{"display":true,"text":"training loss"}}}}}
```

**Reading the optimizer curves.** Plain SGD needs the steepest-direction-limited learning rate, so progress along the shallow directions is slow; momentum at the same rate (an effective rate of $\eta/(1-\beta)=0.5$) is about 20x lower in loss by epoch 30. Adam starts slower here (its bias-corrected steps are small per coordinate with $\eta=0.003$) but catches up and matches momentum by epoch 60. Neither is universally better: SGD with momentum plus a tuned schedule remains the standard for many CNN recipes, Adam(W) for transformers. Compare optimizers at their *own* best learning rates, never at one shared value, or you are measuring your choice of rate.

```chart
{"type":"line","caption":"Effect of initialization with SGD + momentum (lr 0.05) on the same task and model (lab code, one seed). N(0, 0.01) is stuck on a plateau at loss ln 3 = 1.10 for more than 30 epochs; N(0, 1) begins at a loss of 2.5 yet trains fast here because the network is only three layers deep and the data are tiny, so it is a poor guide to what happens at depth 30.","height":340,"data":{"labels":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60],"datasets":[{"label":"He","data":[0.9141,0.7005,0.5739,0.4649,0.3437,0.2476,0.1771,0.1299,0.1029,0.0852,0.0738,0.0648,0.0556,0.0516,0.0478,0.039,0.0361,0.0355,0.0293,0.0287,0.0251,0.0248,0.0222,0.0212,0.0198,0.0185,0.0175,0.0165,0.0166,0.017,0.0179,0.0143,0.0142,0.0134,0.0126,0.0119,0.0121,0.0123,0.0114,0.0115,0.0103,0.0095,0.009,0.0089,0.0088,0.0081,0.0081,0.0082,0.0076,0.0073,0.0077,0.0073,0.007,0.0064,0.0067,0.0065,0.0061,0.0061,0.0061,0.0056],"borderColor":"var(--c-blue)","borderWidth":2,"pointRadius":0},{"label":"Glorot","data":[1.0723,0.9654,0.8344,0.7247,0.6694,0.6349,0.5857,0.529,0.4871,0.4405,0.3697,0.3302,0.2861,0.2599,0.2394,0.2003,0.1754,0.1535,0.1344,0.1187,0.1005,0.0974,0.084,0.0852,0.0759,0.0624,0.0597,0.0531,0.0513,0.0613,0.0572,0.0509,0.036,0.038,0.0413,0.0415,0.032,0.0275,0.0323,0.0594,0.0388,0.0314,0.0243,0.0207,0.0204,0.0244,0.0188,0.0288,0.026,0.0251,0.0254,0.0169,0.0133,0.0133,0.0117,0.0125,0.0122,0.0108,0.011,0.0121],"borderColor":"var(--c-green)","borderWidth":2,"pointRadius":0},{"label":"N(0, 0.01)","data":[1.0993,1.0991,1.1005,1.0986,1.1002,1.1007,1.0993,1.0991,1.0995,1.0998,1.0996,1.1008,1.0994,1.0995,1.0993,1.0991,1.0995,1.0989,1.0988,1.1013,1.0988,1.0991,1.0982,1.0981,1.0983,1.0985,1.0994,1.0979,1.0964,1.0955,1.0941,1.091,1.0832,1.0719,1.0461,1.0015,0.9399,0.8987,0.8769,0.8645,0.8271,0.7827,0.7318,0.7197,0.6877,0.6904,0.7079,0.6606,0.6376,0.6917,0.6415,0.6034,0.6081,0.6098,0.5862,0.5317,0.508,0.4679,0.4526,0.4236],"borderColor":"var(--c-red)","borderWidth":2,"pointRadius":0},{"label":"N(0, 1)","data":[2.5074,0.2677,0.1398,0.0583,0.033,0.0337,0.0417,0.0424,0.0145,0.0096,0.0074,0.0068,0.0051,0.0056,0.0061,0.0064,0.0044,0.0035,0.0035,0.0031,0.0031,0.0035,0.003,0.003,0.0028,0.0027,0.0025,0.0025,0.0026,0.0034,0.0036,0.0026,0.0023,0.0023,0.0021,0.0022,0.0025,0.0031,0.0027,0.0023,0.0019,0.0019,0.0019,0.0018,0.0017,0.0017,0.0016,0.0017,0.0016,0.0015,0.0016,0.0015,0.0015,0.0014,0.0015,0.0015,0.0014,0.0014,0.0014,0.0013],"borderColor":"var(--c-purple)","borderWidth":2,"pointRadius":0}]},"options":{"scales":{"x":{"title":{"display":true,"text":"epoch"}},"y":{"type":"logarithmic","title":{"display":true,"text":"training loss"}}}}}
```

**Reading the initialization curves.** Initialization changes the first epochs, not the destination: Glorot lags He slightly on a ReLU net (its variance is half as large), and a weight std of 0.01 leaves the network on a plateau at the chance loss $\ln3\approx1.10$ because the signal is too small to produce useful gradients, until it eventually breaks symmetry. The std-1 run is an honest surprise: it starts with a huge loss but trains fine in a 3-layer ReLU network, since ReLU is scale-equivariant and there is no saturation. The failure of bad initialization is a *depth* phenomenon (see the 30-layer chart above), which a shallow lab only hints at; try `init_params([2] + [64] * 12 + [3], ...)` to see it bite.

```chart
{"type":"line","caption":"Final training loss after 30 epochs versus learning rate (lab code, one seed; diverged runs are clipped at 5). Adam's good region sits around 0.001 to 0.1 and is wider than momentum SGD's, which needs 0.03 to 0.3. Both fall apart one step beyond their optimum: a learning-rate sweep on a log grid is the cheapest hyperparameter search there is.","height":340,"data":{"labels":["0.0003","0.001","0.003","0.01","0.03","0.1","0.3","1.0"],"datasets":[{"label":"SGD+momentum","data":[0.8225,0.6907,0.5029,0.1384,0.031,0.008,0.0022,1.1277],"borderColor":"var(--c-blue)","backgroundColor":"var(--c-blue)","borderWidth":2},{"label":"Adam","data":[0.5853,0.1398,0.0201,0.0031,0.0013,0.0005,0.377,0.9071],"borderColor":"var(--c-green)","backgroundColor":"var(--c-green)","borderWidth":2}]},"options":{"scales":{"x":{"title":{"display":true,"text":"learning rate"}},"y":{"type":"logarithmic","title":{"display":true,"text":"final training loss"}}}}}
```

**Reading the sweep.** The usable learning-rate window is a band, not a point, and its position differs by optimizer. This is why the first thing to do on a new problem is a coarse log-spaced learning-rate sweep, and why the LR range test of the schedules section is worth the five minutes.

:::exercise Lab extensions
1. Add Nesterov momentum and RMSProp classes and add them to experiment 1. 2. Add gradient clipping at norm 1.0 and rerun with the lr-1.0 SGD run: does it survive? 3. Add a warmup+cosine schedule to Adam and compare final loss at lr 0.01. 4. Replace the data with a 12-layer MLP and rerun experiment 2. 5. Re-implement experiment 1 in PyTorch with `torch.optim.SGD(momentum=0.9)` and `torch.optim.Adam` and confirm you get qualitatively the same curves.
:::

The PyTorch version of the comparison is short; here it is, so you can transfer the experiment to your own model:

```python:optimizer_lab_torch.py
import torch, torch.nn as nn, numpy as np

def make_spiral(n=200, k=3, noise=0.2, seed=0):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for c in range(k):
        r = np.linspace(0.05, 1, n); t = np.linspace(c * 4, (c + 1) * 4, n) + rng.normal(0, noise, n)
        X.append(np.c_[r * np.sin(t), r * np.cos(t)]); y += [c] * n
    return torch.tensor(np.vstack(X), dtype=torch.float32), torch.tensor(y)

def train(make_opt, epochs=60, bs=64, seed=0):
    torch.manual_seed(seed)
    X, y = make_spiral()
    model = nn.Sequential(nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 3))
    for m in model:
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu"); nn.init.zeros_(m.bias)
    opt = make_opt(model.parameters())
    curve = []
    for ep in range(epochs):
        perm, total = torch.randperm(len(X)), 0.0
        for i in range(0, len(X), bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = nn.functional.cross_entropy(model(X[idx]), y[idx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        curve.append(total / len(X))
    return curve

for name, mk in [("SGD", lambda p: torch.optim.SGD(p, lr=0.05)),
                 ("SGD+momentum", lambda p: torch.optim.SGD(p, lr=0.05, momentum=0.9)),
                 ("Nesterov", lambda p: torch.optim.SGD(p, lr=0.05, momentum=0.9, nesterov=True)),
                 ("RMSprop", lambda p: torch.optim.RMSprop(p, lr=1e-3)),
                 ("Adam", lambda p: torch.optim.Adam(p, lr=3e-3)),
                 ("AdamW", lambda p: torch.optim.AdamW(p, lr=3e-3, weight_decay=0.01))]:
    c = train(mk)
    print(f"{name:13s} loss at epochs 1/10/60: {c[0]:.3f} {c[9]:.3f} {c[59]:.3f}")
```

## Common mistakes and debugging tips

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| Loss is `nan` after a few steps | Learning rate too high; exploding gradient; `log(0)`; fp16 overflow | Lower lr by 3-10x, add warmup and clipping, use bf16 or a loss scaler |
| Loss flat at $\ln K$ from the start | Learning rate too small, init too small, dead ReLUs, labels not matching inputs | Run an lr sweep; check init stats; overfit a 32-example batch |
| Loss decreases then suddenly spikes | LR too high late in a long run, Adam $\varepsilon$/$\beta_2$ issues, a bad batch | Clip gradients; lower peak lr; use $\beta_2=0.95$ for large models; use cosine decay |
| Adam converges but generalizes worse than SGD | Weak or coupled weight decay | Use AdamW, tune $\lambda$, no decay on biases and norm parameters |
| Weight decay "does nothing" in Adam | Using `Adam(weight_decay=...)` (L2 inside the gradient) | Switch to `AdamW` |
| Training is fine, eval accuracy is erratic | BatchNorm in train mode at eval, or tiny batch statistics | `model.eval()`; use GroupNorm/LayerNorm for tiny batches |
| Moved to bigger batch, accuracy dropped | LR not rescaled, no warmup | Scale lr (linear for SGD, sqrt for Adam), warm up, retune the schedule |
| Scheduler seems to have no effect | `scheduler.step()` not called, called per epoch instead of per step, or called before `optimizer.step()` | Call `optimizer.step()` then `scheduler.step()`; check `optimizer.param_groups[0]["lr"]` |
| Gradients of first layers are tiny, last layers huge | Vanishing gradients: saturating activation, bad init, no normalization | He/Glorot init, ReLU/GELU, add norm layers and residual connections |
| Clipping fires every step | lr far too high | Lower lr; clipping is a safety net only |

Three habits catch most optimization bugs: (1) **log** training loss, gradient norm, the learning rate and update-to-weight ratio $\norm{\eta g}/\norm{\theta}$ (healthy is around $10^{-3}$); (2) **overfit one small batch** to prove the pipeline can learn; (3) change **one thing at a time** and always compare optimizers at their own tuned learning rates.

## Exercises

:::exercise Exercise 1: Stability of gradient descent on a quadratic
For $f(\theta)=\tfrac12\lambda\theta^2$ derive the condition on $\eta$ for gradient descent to converge, and the iterates after $t$ steps. What is the best learning rate for $f(\boldsymbol\theta)=\tfrac12(100\theta_1^2+\theta_2^2)$ and how many steps to shrink the error by $10^{-3}$?
:::details Solution
$\theta_{t+1}=\theta_t-\eta\lambda\theta_t=(1-\eta\lambda)\theta_t$, so $\theta_t=(1-\eta\lambda)^t\theta_0$, convergent iff $|1-\eta\lambda|<1$, i.e. $0<\eta<2/\lambda$. For the two-dimensional case the best rate is $\eta^\star=2/(100+1)\approx0.0198$ with rate $\rho=(100-1)/(100+1)\approx0.980$; $\rho^t=10^{-3}$ gives $t=\ln(10^{-3})/\ln(0.980)\approx340$ steps. With momentum at the optimal setting the rate becomes $(\sqrt{100}-1)/(\sqrt{100}+1)=9/11\approx0.82$, so about 35 steps.
:::
:::

:::exercise Exercise 2: Variance of the mini-batch gradient
Per-example gradients have mean $\boldsymbol G$ and covariance $\boldsymbol\Sigma$. Show that for $B$ independent samples, $\E\norm{\boldsymbol g_B-\boldsymbol G}^2=\tr(\boldsymbol\Sigma)/B$. If $\tr\boldsymbol\Sigma=50$ and $\norm{\boldsymbol G}^2=0.5$, what is the noise scale and what does it suggest for batch size?
:::details Solution
$\boldsymbol g_B-\boldsymbol G=\frac1B\sum_i(\boldsymbol\xi_i)$ with independent zero-mean $\boldsymbol\xi_i$, so the expected squared norm is $\frac1{B^2}\sum_i\E\norm{\boldsymbol\xi_i}^2=\tr(\boldsymbol\Sigma)/B$. Noise scale $=\tr\boldsymbol\Sigma/\norm{\boldsymbol G}^2=100$. Below about $B=100$ the gradient is noise-dominated and larger batches are nearly free in terms of steps; far above 100 extra examples buy little per step.
:::
:::

:::exercise Exercise 3: Bias correction by hand
Run two Adam steps ($\beta_1=0.9$, $\beta_2=0.999$, $\varepsilon=0$, $\eta=0.1$) on a scalar with constant gradient $g=2$. Compute $m,v,\hat m,\hat v$ and the updates, with and without bias correction.
:::details Solution
Step 1: $m_1=0.2$, $v_1=0.004$; $\hat m_1=0.2/0.1=2$, $\hat v_1=0.004/0.001=4$, update $=0.1\cdot2/2=0.1$. Without correction the update would be $0.1\cdot0.2/\sqrt{0.004}=0.316$ (too big, since $v$ is biased more than $m$). Step 2: $m_2=0.38$, $v_2=0.007996$; $\hat m_2=0.38/0.19=2$, $\hat v_2=0.007996/0.001999=4$, update $0.1$ again. With constant gradients the corrected update is always $\eta\,\mathrm{sign}(g)$.
:::
:::

:::exercise Exercise 4: Derive the He variance backward
Repeat the He derivation for the backward pass of a ReLU layer and show the variance condition is $\sigma_w^2=2/n_{\rm out}$. Why does it not matter much which one you pick?
:::details Solution
$\bar a_i=\sum_{j=1}^{n_{\rm out}}w_{ij}\bar z_j$ and $\bar z_j=\bar a'_j\,\mathbb 1[z_j>0]$, so $\E[\bar z_j^2]=\tfrac12\E[\bar a_j'^2]$. Then $\Var(\bar a_i)=n_{\rm out}\sigma_w^2\cdot\tfrac12\Var(\bar a')$, which is constant iff $\sigma_w^2=2/n_{\rm out}$. Over a whole network the fan-in and fan-out conditions differ by the ratio $n_{\rm in}/n_{\rm out}$ at each layer, and these ratios telescope along a chain of layers (the product of $n_{\ell-1}/n_\ell$ is $n_0/n_L$), so using either rule changes the overall scale only by a bounded factor, not an exponential one.
:::
:::

:::exercise Exercise 5: Why weight decay and L2 differ under Adam
For a scalar parameter with $g=0$ in the data term (so only the penalty acts), compare the Adam-with-L2 update and the AdamW update in steady state with $\theta=1$, $\lambda=0.1$, $\eta=0.01$.
:::details Solution
Adam+L2: the gradient is $\lambda\theta=0.1$, and in steady state $\hat m\approx0.1$, $\sqrt{\hat v}\approx0.1$, so the update is $\eta\cdot1=0.01$ in magnitude: the parameter loses 0.01 per step regardless of $\lambda$ (the scaling cancels $\lambda$!). If the data gradient were large with $\sqrt v\approx10$, the same penalty would move by only $0.01\cdot0.1/10=10^{-4}$: decay is switched off for parameters with big gradients. AdamW: the decay is $\eta\lambda\theta=10^{-3}$ per step in both cases. This is the difference: AdamW's decay strength is independent of gradient history.
:::
:::

:::exercise Exercise 6: BatchNorm gradient identities
Using the BatchNorm backward formula, show that $\sum_n\bar x_n=0$ and $\sum_n\bar x_n\hat x_n=0$.
:::details Solution
Let $u_n=\bar{\hat x}_n$. Then $\sum_n\bar x_n\propto B\sum u_n-B\sum u_m-(\sum_n\hat x_n)(\sum u_m\hat x_m)=0$ because $\sum_n\hat x_n=0$. For the second, $\sum_n\hat x_n\bar x_n\propto B\sum u_n\hat x_n-\sum\hat x_n\sum u_m-\sum_n\hat x_n^2\sum u_m\hat x_m$, and $\sum_n\hat x_n^2=B$ (up to $\varepsilon$), so the first and third terms cancel and the middle term is 0. So the gradient wrt the layer input is orthogonal to both the constant vector and $\hat x$: shifting or scaling the batch cannot change the loss.
:::
:::

:::exercise Exercise 7: Code: reproduce the linear scaling rule
Using the lab code, train with batch sizes 16, 32, 64, 128 on the spiral with $\eta=0.05\cdot B/32$ and count the optimizer steps needed to reach a training loss below 0.3. Do larger batches follow the rule? Where does it break?
:::details Solution
Wrap `run` so that it records the first step at which the epoch loss drops below 0.3 (modify it to evaluate the full-data loss every few steps). You should see the steps-to-target fall roughly in inverse proportion to $B$ for small $B$ (compare the chart earlier in the page, which used plain SGD). It breaks when $B$ approaches the noise scale or the dataset size: with $B=600$ (the full data) there is no noise left to average, the gradient is exact, and the step count is limited by curvature (the largest stable $\eta$), not by $B$. Raising $\eta$ proportionally past the stable value diverges; warmup or a lower multiplier fixes that.
:::
:::

:::exercise Exercise 8: What would happen if (challenging)
You train a ReLU network with He init and BatchNorm after every linear layer, but you then multiply every weight matrix by 100 at initialization. What changes in the forward pass, in the gradients, and in the effective learning rate?
:::details Solution
Forward: nothing, because BN removes the scale: $\mathrm{BN}(c\,\mathbf W\boldsymbol a)=\mathrm{BN}(\mathbf W\boldsymbol a)$ (up to $\varepsilon$). Gradients: the loss is invariant to the scale $c$ of $\mathbf W$, so the gradient is orthogonal to $\mathbf W$ and scales as $1/c$ (the function depends on the direction $\mathbf W/\norm{\mathbf W}$ only; by homogeneity $\nabla L(c\mathbf W)=\tfrac1c\nabla L(\mathbf W)$). A gradient step of size $\eta/c$ on a weight of norm $c\norm{\mathbf W}$ rotates the direction by an angle of order $\eta/c^2$: the *effective* learning rate on the direction is divided by $c^2=10^4$, and training effectively freezes. Conversely, weight decay shrinks $c$ and so *raises* the effective learning rate. This scale-invariance explains why BN networks tolerate poor init but interact strongly with weight decay.
:::
:::

:::exercise Exercise 9: Design a schedule (challenging)
You will train a transformer with AdamW for 100,000 steps at a peak rate of $6\times10^{-4}$. Specify a schedule and justify warmup length, final rate, weight-decay settings and clipping.
:::details Solution
A reasonable specification: linear warmup over 2,000 steps (about 2 percent), then cosine decay to $6\times10^{-5}$ (10 percent of peak) at step 100,000; AdamW with $\beta=(0.9,0.95)$, $\varepsilon=10^{-8}$, weight decay 0.1 on weight matrices only (not biases, LayerNorm gains or embeddings by convention), global-norm clipping at 1.0. Justification: warmup protects against early instability when the second-moment estimate and the attention logits are poorly calibrated; $\beta_2=0.95$ makes $v$ adapt within about 20 steps, which reacts faster to loss spikes; the nonzero final rate avoids over-committing to the last few steps and cosine is parameter-free apart from the horizon. These numbers are conventional starting points used by several open LLM recipes, not guarantees: tune the peak rate first.
:::
:::

## Key takeaways

:::key Key takeaways
- Neural loss surfaces are non-convex but mostly benign: high-dimensional critical points are overwhelmingly saddles, and the real difficulty is ill-conditioning (a large ratio of curvatures) rather than bad local minima. Flatness is suggestive but parameterization-dependent.
- Mini-batch gradients are unbiased with covariance $\boldsymbol\Sigma/B$; batch size trades steps for compute, and the linear scaling rule (with warmup) works until the batch exceeds the gradient noise scale.
- Momentum accelerates along consistent directions ($O(\kappa)\to O(\sqrt\kappa)$ on quadratics); adaptive methods rescale each coordinate; Adam adds bias-corrected first and second moments; AdamW decouples weight decay and is the default for transformers.
- Use warmup, then cosine (or step) decay; one-cycle and cyclical schedules are useful when training budgets are short.
- Initialize with $\sigma_w^2=2/n_{\rm in}$ for ReLU (He) and $2/(n_{\rm in}+n_{\rm out})$ for tanh (Glorot) so that activation and gradient variances stay constant with depth; clip gradients by global norm as a safety net.
- Normalization (BatchNorm for large-batch CNNs, LayerNorm/RMSNorm for transformers, GroupNorm for small batches) stabilizes scales, smooths the landscape and allows larger learning rates; BatchNorm behaves differently in train and eval.
- Compare optimizers only at tuned learning rates, change one variable at a time, log gradient norms, and overfit a small batch first.
:::

## Further reading

- **Essential.** Kingma & Ba (2015). *Adam: a method for stochastic optimization.* ICLR. arXiv:1412.6980.
- **Essential.** Ioffe & Szegedy (2015). *Batch normalization: accelerating deep network training by reducing internal covariate shift.* ICML. arXiv:1502.03167.
- **Essential.** Loshchilov & Hutter (2019). *Decoupled weight decay regularization.* ICLR. arXiv:1711.05101.
- Goodfellow, Bengio & Courville (2016). *Deep Learning*, Chapter 8: Optimization for training deep models. MIT Press.
- Duchi, Hazan & Singer (2011). *Adaptive subgradient methods for online learning and stochastic optimization.* JMLR 12.
- Polyak, B. (1964). *Some methods of speeding up the convergence of iteration methods.* USSR Computational Mathematics and Mathematical Physics 4(5).
- Sutskever, Martens, Dahl & Hinton (2013). *On the importance of initialization and momentum in deep learning.* ICML.
- Dauphin et al. (2014). *Identifying and attacking the saddle point problem in high-dimensional non-convex optimization.* NeurIPS. arXiv:1406.2572.
- Hochreiter & Schmidhuber (1997). *Flat minima.* Neural Computation 9(1).
- Keskar et al. (2017). *On large-batch training for deep learning: generalization gap and sharp minima.* ICLR. arXiv:1609.04836.
- Dinh, Pascanu, Bengio & Bengio (2017). *Sharp minima can generalize for deep nets.* ICML. arXiv:1703.04933.
- Goyal et al. (2017). *Accurate, large minibatch SGD: training ImageNet in 1 hour.* arXiv:1706.02677.
- McCandlish et al. (2018). *An empirical model of large-batch training.* arXiv:1812.06162.
- Shallue et al. (2019). *Measuring the effects of data parallelism on neural network training.* JMLR 20. arXiv:1811.03600.
- Glorot & Bengio (2010). *Understanding the difficulty of training deep feedforward neural networks.* AISTATS.
- He, Zhang, Ren & Sun (2015). *Delving deep into rectifiers: surpassing human-level performance on ImageNet classification.* ICCV. arXiv:1502.01852.
- Pascanu, Mikolov & Bengio (2013). *On the difficulty of training recurrent neural networks.* ICML. arXiv:1211.5063.
- Loshchilov & Hutter (2017). *SGDR: stochastic gradient descent with warm restarts.* ICLR. arXiv:1608.03983.
- Smith (2017). *Cyclical learning rates for training neural networks.* WACV. arXiv:1506.01186.
- Smith & Topin (2018). *Super-convergence: very fast training of neural networks using large learning rates.* arXiv:1708.07120.
- Ba, Kiros & Hinton (2016). *Layer normalization.* arXiv:1607.06450.
- Wu & He (2018). *Group normalization.* ECCV. arXiv:1803.08494.
- Zhang & Sennrich (2019). *Root mean square layer normalization.* NeurIPS. arXiv:1910.07467.
- Santurkar, Tsipras, Ilyas & Madry (2018). *How does batch normalization help optimization?* NeurIPS. arXiv:1805.11604.
- Arora, Li & Lyu (2019). *Theoretical analysis of auto rate-tuning by batch normalization.* ICLR. arXiv:1812.03981.
