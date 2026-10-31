---
title: "Trustworthy Deep Learning and Theory"
phase: Advanced Topics
summary: What it takes to trust a model, and what we do and do not understand about why deep learning works. Adversarial examples (FGSM and PGD derived, linearity explanation, threat models, adversarial training, certified defenses, transferability), distribution shift and OOD detection (max softmax, energy, Mahalanobis), uncertainty (aleatoric vs epistemic, ensembles, MC dropout, ECE, temperature scaling, conformal prediction), interpretability (saliency, SmoothGrad, Grad-CAM, Integrated Gradients, attention caveats, probing, mechanistic interpretability), fairness, privacy (membership inference, differential privacy, DP-SGD), safety and responsible release, theory highlights (randomization tests, implicit regularization, NTK, lottery tickets, grokking, double descent) and open problems. The lab attacks and defends a classifier, implements Grad-CAM, and measures calibration.
---

:::key Learning objectives
By the end of this week you should be able to:
- Derive FGSM and PGD from a first-order model of the loss, explain the linearity account of adversarial vulnerability, and state precisely what a threat model is.
- Implement adversarial training as a min-max problem, and explain why it costs more, why it trades off clean accuracy, and why certified defenses give weaker but provable guarantees.
- Classify distribution shifts (covariate, label, concept) and implement max-softmax, energy and Mahalanobis out-of-distribution detectors.
- Separate aleatoric from epistemic uncertainty, build deep ensembles and MC dropout, compute the expected calibration error, and fix miscalibration with temperature scaling.
- State and prove the marginal coverage guarantee of split conformal prediction.
- Derive Grad-CAM, SmoothGrad and Integrated Gradients (with its axioms), and explain why attention weights and probing accuracy are not explanations.
- Describe circuits, superposition and sparse autoencoders in mechanistic interpretability.
- Compute demographic parity and equalized odds gaps, explain why they cannot generally be satisfied together, and list mitigation strategies.
- Define membership inference, $(\varepsilon,\delta)$-differential privacy and DP-SGD, and reason about the privacy-utility trade-off.
- Explain what the randomization experiments, implicit regularization, the neural tangent kernel, the lottery ticket hypothesis, grokking and double descent each tell us (and do not tell us) about generalization.
:::

For seventeen weeks the question was "how do I get a network to minimize a loss on a dataset". This week the question is "should anyone rely on the result". A classifier with 99% test accuracy can be fooled by a change to its input that no human notices. A medical model that is accurate on the hospital that trained it can fail at the next hospital over. A model that says "97% confident" may be right only 60% of the time. A heat map that "explains" a prediction may be the same map you would get from a network with random weights. A model trained on private records can sometimes be made to reveal them. None of these failures show up in the number you were optimizing, which is why this material matters: **test accuracy under the training distribution is a necessary condition for a good model and a very weak sufficient one.**

The second half of the week turns around and asks the theorist's question: why does any of this work at all? Classical learning theory says a model with more parameters than training points should overfit catastrophically. Modern networks interpolate their training data, including random labels, and still generalize. We will look at what is understood (a surprising amount in simplified settings), what is conjectured, and what is simply unknown.

A note on tone. Trustworthiness is a field with many results that are *negative* (impossibility theorems, attacks that break defenses) and many that are *empirical and fragile* (a saliency method that works on one network). Throughout, I will separate what is proven, what is empirical, and what is folklore.

## Adversarial examples {#adversarial}

### The phenomenon

In 2013, Szegedy and colleagues reported something startling: for a trained image classifier you can find a tiny perturbation $\boldsymbol{\delta}$, invisible to a person, such that $f(\boldsymbol{x}+\boldsymbol{\delta})$ is confidently wrong. Biggio et al. had demonstrated related evasion attacks against classical classifiers in the same period. The perturbed input is an **adversarial example**. It is not noise: random perturbations of the same size almost never change the prediction, yet a carefully chosen one succeeds on essentially every input of an undefended network.

:::history
Szegedy et al., "Intriguing properties of neural networks" (2013/2014), found adversarial examples by box-constrained L-BFGS. Goodfellow, Shlens and Szegedy, "Explaining and harnessing adversarial examples" (ICLR 2015), proposed the one-step FGSM attack and the linearity hypothesis. Madry et al., "Towards deep learning models resistant to adversarial attacks" (ICLR 2018), framed robustness as a saddle-point problem and popularized PGD adversarial training. Athalye, Carlini and Wagner (ICML 2018) showed that most defenses published at ICLR 2018 relied on "obfuscated gradients" and could be broken, a lesson that still shapes how defenses are evaluated.
:::

### Threat models

A defense is meaningless without saying what attack it defends against. A **threat model** specifies four things.

1. **Goal.** *Untargeted*: make the prediction anything but the true class. *Targeted*: make it a chosen class $t$. Or other goals such as raising the loss, or inducing a high-confidence error.
2. **Capability (the perturbation set).** Most of the literature uses an $\ell_p$ ball $\mathcal{S}=\{\boldsymbol{\delta}:\norm{\boldsymbol{\delta}}_p\le\epsilon\}$, usually $p=\infty$ (each pixel moves at most $\epsilon$, typically $8/255$ on CIFAR-10), $p=2$, or $p=0$/$1$ (few pixels change). The $\ell_p$ ball is a *convenient proxy* for "imperceptible", not a faithful model of real attackers: rotations, patches, colour shifts, paraphrases and prompt injections are all valid attacks outside it.
3. **Knowledge.** *White-box*: the attacker knows weights and architecture and can compute gradients. *Black-box*: the attacker can only query (score-based or label-only), or only has a surrogate model (transfer attacks).
4. **Setting.** Digital (edit the pixels), physical (adversarial stickers, glasses frames, printed patches that survive camera capture), training time (poisoning, backdoors) versus test time (evasion).

{{figure: week17-threat-models | A threat model is a four-part specification. The $\ell_\infty$ and $\ell_2$ balls of the same radius contain different perturbations; a defense for one need not hold for the other.}}

### FGSM: the one-step attack

Let $\mathcal{L}(\boldsymbol{\theta},\boldsymbol{x},y)$ be the loss of a classifier with parameters $\boldsymbol{\theta}$ at input $\boldsymbol{x}$ and label $y$. Training differentiates with respect to $\boldsymbol{\theta}$; an attack differentiates with respect to $\boldsymbol{x}$, which you can do with the same backpropagation machinery. The attacker wants a perturbation $\boldsymbol{\delta}$ in the allowed set that maximizes the loss. Start with the first-order Taylor expansion around $\boldsymbol{x}$:

$$
\mathcal{L}(\boldsymbol{\theta},\boldsymbol{x}+\boldsymbol{\delta},y)\;\approx\;\mathcal{L}(\boldsymbol{\theta},\boldsymbol{x},y)+\boldsymbol{\delta}^\top \boldsymbol{g},\qquad \boldsymbol{g}=\nabla_{\boldsymbol{x}}\mathcal{L}(\boldsymbol{\theta},\boldsymbol{x},y).
$$

So within the ball of radius $\epsilon$ the attacker maximizes the linear function $\boldsymbol{\delta}^\top\boldsymbol{g}$. This is exactly the definition of a dual norm: $\max_{\norm{\boldsymbol{\delta}}_p\le\epsilon}\boldsymbol{\delta}^\top\boldsymbol{g}=\epsilon\norm{\boldsymbol{g}}_q$ with $1/p+1/q=1$.

:::math Derivation: the optimal first-order perturbation for $\ell_\infty$ and $\ell_2$
**$\ell_\infty$ ($p=\infty$, $q=1$).** The constraint is $\abs{\delta_i}\le\epsilon$ for every coordinate independently, so we maximize $\sum_i\delta_i g_i$ coordinatewise. Each term is largest when $\delta_i=\epsilon\,\mathrm{sign}(g_i)$. Hence
$$\boldsymbol{\delta}^\star=\epsilon\,\mathrm{sign}(\boldsymbol{g}),\qquad \boldsymbol{\delta}^{\star\top}\boldsymbol{g}=\epsilon\sum_i\abs{g_i}=\epsilon\norm{\boldsymbol{g}}_1.$$
This is the **Fast Gradient Sign Method**: $\boldsymbol{x}_{\text{adv}}=\boldsymbol{x}+\epsilon\,\mathrm{sign}(\nabla_{\boldsymbol{x}}\mathcal{L})$.

**$\ell_2$ ($p=2$, $q=2$).** By Cauchy-Schwarz, $\boldsymbol{\delta}^\top\boldsymbol{g}\le\norm{\boldsymbol{\delta}}_2\norm{\boldsymbol{g}}_2\le\epsilon\norm{\boldsymbol{g}}_2$, with equality iff $\boldsymbol{\delta}$ points along $\boldsymbol{g}$ and has length $\epsilon$:
$$\boldsymbol{\delta}^\star=\epsilon\,\frac{\boldsymbol{g}}{\norm{\boldsymbol{g}}_2}.$$
The $\ell_2$ attack is simply a normalized gradient step. Note the loss increase predicted by the linear model is $\epsilon\norm{\boldsymbol{g}}_1$ in the first case and $\epsilon\norm{\boldsymbol{g}}_2$ in the second; since $\norm{\boldsymbol{g}}_1$ can be as large as $\sqrt d\,\norm{\boldsymbol{g}}_2$, the $\ell_\infty$ attack is far more powerful per unit "per-coordinate" budget in high dimension. That observation is the heart of the next subsection.
:::

### Why linearity is enough: the high-dimensional argument

Goodfellow et al. argued that adversarial examples do not require the extreme non-linearity that many assumed; they are a consequence of the *near-linear* behaviour of networks (ReLU, maxout and tuned sigmoids are all piecewise or locally linear) in *high dimension*. Consider a linear score $s(\boldsymbol{x})=\boldsymbol{w}^\top\boldsymbol{x}$ and an $\ell_\infty$ perturbation of size $\epsilon$. The change in the score is

$$
\boldsymbol{w}^\top\boldsymbol{\delta}=\epsilon\,\boldsymbol{w}^\top\mathrm{sign}(\boldsymbol{w})=\epsilon\norm{\boldsymbol{w}}_1.
$$

If the weights have typical magnitude $m$ per coordinate and the input has $d$ dimensions, then $\norm{\boldsymbol{w}}_1\approx m\,d$. The perturbation is tiny in every coordinate ($\epsilon$), yet the score moves by $\epsilon\,m\,d$, growing **linearly in the dimension**. A 224 by 224 by 3 image has $d\approx 150{,}000$. Each pixel can move by less than the 8-bit quantization step and the logit can still move by a very large amount, because thousands of tiny aligned contributions add up. The perturbation is aligned with the weights; random noise is not, and adds up like $\sqrt d$ rather than $d$.

The following chart is a computation, not an illustration. I trained a logistic regression on two Gaussian classes whose means differ by a *fixed* small amount per coordinate, in $d=10$, $100$ and $1000$ dimensions, and attacked it with FGSM. Higher dimension gives a *more accurate* clean model (more evidence is available) and also a model that collapses at a fixed $\epsilon$-per-pixel once $\epsilon$ passes the per-coordinate signal.

```chart
{
  "type": "line",
  "caption": "FGSM accuracy of a logistic-regression classifier on two Gaussian classes (per-coordinate mean shift 0.08, unit noise) as the L-infinity budget grows, for three input dimensions. Computed with NumPy, 4000 training points, full-batch gradient descent. Higher dimension means higher clean accuracy but a sharper collapse at the same per-coordinate budget.",
  "height": 340,
  "data": {
    "labels": [0, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1],
    "datasets": [
      {"label": "d = 10", "data": [0.607, 0.592, 0.579, 0.553, 0.530, 0.502, 0.485], "borderColor": "var(--c-blue)", "tension": 0.2},
      {"label": "d = 100", "data": [0.805, 0.775, 0.743, 0.674, 0.593, 0.517, 0.439], "borderColor": "var(--c-amber)", "tension": 0.2},
      {"label": "d = 1000", "data": [1.0, 1.0, 1.0, 0.942, 0.741, 0.494, 0.280], "borderColor": "var(--c-red)", "tension": 0.2}
    ]
  },
  "options": { "scales": { "x": {"title": {"display": true, "text": "epsilon (L-infinity budget per coordinate)"}}, "y": {"title": {"display": true, "text": "accuracy under FGSM"}, "min": 0, "max": 1.05} } }
}
```

The linearity explanation predicts several things that experiments confirm: adversarial examples transfer between models (similar linear structure), they exist in all directions weakly correlated with the weights rather than in isolated pockets, and training procedures that make the function *more* linear (like Goodfellow's early observations on shallow models) did not remove them. It is not the whole story: Ilyas et al. (2019) argued that standard datasets contain *non-robust but predictive features* that models legitimately use, so adversarial vulnerability is also a property of the data, and Gilmer et al. (2018) and Shafahi et al. (2019) proved that in high dimension even small-error classifiers must be vulnerable under some assumptions about the data distribution. Treat "linearity" as one correct and useful lens among several.

{{figure: week17-fgsm-geometry | Geometry of FGSM: the loss surface is locally approximated by its tangent plane, and the sign of the gradient picks the corner of the $\ell_\infty$ box (the green point) that increases the loss most. The $\ell_2$ attack moves along the gradient to the circle instead.}}

### PGD: the iterated attack

FGSM trusts the linear model across a whole $\epsilon$-step. If the loss curves, a better strategy is to take several small steps and re-linearize each time. **Projected Gradient Descent** (really gradient *ascent* on the loss) with random start, as used by Madry et al.:

$$
\boldsymbol{x}^{(0)}=\boldsymbol{x}+\boldsymbol{u},\ \boldsymbol{u}\sim U[-\epsilon,\epsilon]^d,\qquad
\boldsymbol{x}^{(t+1)}=\Pi_{\mathcal{B}_\epsilon(\boldsymbol{x})}\Big(\boldsymbol{x}^{(t)}+\alpha\,\mathrm{sign}\big(\nabla_{\boldsymbol{x}}\mathcal{L}(\boldsymbol{\theta},\boldsymbol{x}^{(t)},y)\big)\Big),
$$

where $\Pi$ projects back into the allowed set (for $\ell_\infty$ this is a coordinatewise clip to $[\boldsymbol{x}-\epsilon,\boldsymbol{x}+\epsilon]$, and also into the valid pixel range $[0,1]$), and $\alpha$ is a step size, commonly $\epsilon/4$ for 10 steps or $2.5\epsilon/T$ for $T$ steps. The random start matters: it avoids the case where the gradient at the clean point is uninformative (a flat region of a saturated softmax) and gives the attack access to different local maxima.

FGSM is PGD with one step, $\alpha=\epsilon$ and no random start. PGD is considered the standard "first-order adversary": Madry et al. found that the loss values reached by many random restarts of PGD were similar, which suggests the local maxima are of comparable quality, and a defense that survives strong PGD is robust against a broad class of gradient-based attacks. Stronger evaluation uses an ensemble such as AutoAttack (Croce and Hein, 2020), which combines adaptive-step PGD variants with a margin-based loss and a black-box attack, precisely because weak attacks flatter weak defenses.

```python:attacks.py
import torch
import torch.nn.functional as F


def fgsm(model, x, y, eps):
    """One-step L-infinity attack. x in [0,1], shape (B, C, H, W)."""
    x = x.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(x), y)
    grad, = torch.autograd.grad(loss, x)          # d loss / d x, same shape as x
    x_adv = x + eps * grad.sign()
    return x_adv.clamp(0, 1).detach()


def pgd(model, x, y, eps, alpha, steps, random_start=True):
    """Iterated L-infinity attack with projection onto the eps-ball and the valid pixel range."""
    x_adv = x.clone().detach()
    if random_start:
        x_adv = (x_adv + torch.empty_like(x).uniform_(-eps, eps)).clamp(0, 1)
    for _ in range(steps):
        x_adv.requires_grad_(True)
        loss = F.cross_entropy(model(x_adv), y)
        grad, = torch.autograd.grad(loss, x_adv)
        x_adv = x_adv.detach() + alpha * grad.sign()
        x_adv = torch.min(torch.max(x_adv, x - eps), x + eps)   # project to the L-inf ball
        x_adv = x_adv.clamp(0, 1)                                # and to the image domain
    return x_adv.detach()


def pgd_l2(model, x, y, eps, alpha, steps):
    """L2 variant: normalized-gradient step, then rescale the perturbation to norm <= eps."""
    x_adv = x.clone().detach()
    for _ in range(steps):
        x_adv.requires_grad_(True)
        loss = F.cross_entropy(model(x_adv), y)
        grad, = torch.autograd.grad(loss, x_adv)
        g = grad / (grad.flatten(1).norm(dim=1).view(-1, 1, 1, 1) + 1e-12)
        x_adv = x_adv.detach() + alpha * g
        delta = x_adv - x
        norm = delta.flatten(1).norm(dim=1).view(-1, 1, 1, 1)
        delta = delta * torch.clamp(eps / (norm + 1e-12), max=1.0)
        x_adv = (x + delta).clamp(0, 1)
    return x_adv.detach()


@torch.no_grad()
def robust_accuracy(model, loader, attack, device="cpu", **kw):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.enable_grad():          # the attack needs gradients even inside a no_grad function
            x_adv = attack(model, x, y, **kw)
        correct += (model(x_adv).argmax(1) == y).sum().item()
        total += y.numel()
    return correct / total
```

Notice the model must be in `eval()` mode while attacking (so batch norm and dropout are deterministic) and that gradients are taken with respect to the *input*, which is why the input tensor needs `requires_grad_(True)` and `torch.autograd.grad` is cleaner than `loss.backward()` (which would also accumulate parameter gradients).

{{demo: week17-adversarial-playground | Adversarial playground: FGSM and PGD on a 2D logistic classifier | 520}}

### Adversarial training

The most reliable empirical defense is to train on adversarial examples. Madry et al. wrote robust training as a saddle-point problem:

$$
\min_{\boldsymbol{\theta}}\ \ \E_{(\boldsymbol{x},y)\sim\mathcal{D}}\Big[\max_{\boldsymbol{\delta}\in\mathcal{S}}\ \mathcal{L}(\boldsymbol{\theta},\boldsymbol{x}+\boldsymbol{\delta},y)\Big].
$$

The inner maximization is the attacker (approximated by PGD) and the outer minimization is the learner. **Danskin's theorem** says that, under mild conditions, the gradient of the inner maximum with respect to $\boldsymbol{\theta}$ equals the gradient of the loss evaluated at the maximizer $\boldsymbol{\delta}^\star$; so we can run PGD to get $\boldsymbol{\delta}^\star$, then take an ordinary SGD step on $\mathcal{L}(\boldsymbol{\theta},\boldsymbol{x}+\boldsymbol{\delta}^\star,y)$. (The theorem needs exact maximization and differentiability, so for ReLU networks and approximate PGD it is a heuristic justification, though one that works.)

```python:adv_train.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from attacks import pgd


def adversarial_epoch(model, loader, opt, eps=8 / 255, alpha=2 / 255, steps=7, device="cpu"):
    """One epoch of PGD adversarial training (Madry et al., 2018)."""
    model.train()
    total_loss = total_correct = n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        # inner maximization: craft the attack against the CURRENT weights.
        # (Batch norm: many implementations switch to eval() here; either way, be consistent.)
        model.eval()
        x_adv = pgd(model, x, y, eps=eps, alpha=alpha, steps=steps)
        model.train()
        # outer minimization: ordinary SGD step on the adversarial batch
        logits = model(x_adv)
        loss = F.cross_entropy(logits, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        total_loss += loss.item() * y.numel()
        total_correct += (logits.argmax(1) == y).sum().item()
        n += y.numel()
    return total_loss / n, total_correct / n
```

Adversarial training has well-known costs and phenomena.

- **Compute.** With $K$ PGD steps, each iteration costs about $K+1$ forward-backward passes, so training is roughly $8\times$ slower for $K=7$. Fast variants exist ("Fast is better than free", Wong, Rice and Kolter, 2020, used FGSM with a random start plus a cyclic learning rate, and Shafahi et al. 2019 recycled gradients), but single-step training can suffer **catastrophic overfitting**, where the model suddenly becomes robust to FGSM and loses all robustness to PGD.
- **Accuracy trade-off.** On CIFAR-10 at $\ell_\infty$ $\epsilon=8/255$, standard models reach about 95% clean accuracy and 0% robust accuracy; PGD-trained ResNets typically reach about 80-87% clean and about 45-55% robust accuracy depending on architecture and extra data. Tsipras et al. (2019) and Zhang et al. (TRADES, 2019) analysed this as an inherent tension, and TRADES adds an explicit KL term between clean and adversarial predictions to trade the two off with a weight $\beta$.
- **Capacity and data.** Robust training needs larger networks (Madry et al.) and benefits strongly from more data, including unlabeled or synthetic data (Carmon et al. 2019; Gowal et al. 2020, 2021).
- **Robust overfitting.** Robust test accuracy peaks early in training and then falls while robust *training* accuracy keeps rising (Rice, Wong and Kolter, 2020), so early stopping on a robust validation set matters.
- **Robustness is specific.** A model robust to $\ell_\infty$ at $8/255$ is not robust to $\ell_2$ at a matching radius, nor to rotations, nor to common corruptions beyond what it saw.

### Certified defenses (overview)

Empirical robustness has an uncomfortable property: "no attack I tried succeeded" proves nothing. A **certified defense** gives a guarantee: for input $\boldsymbol{x}$ the prediction is provably constant over the whole ball $\norm{\boldsymbol{\delta}}\le r$. Three families dominate.

- **Interval bound propagation (IBP) and convex relaxations.** Propagate lower and upper bounds on each activation through the network. If the lower bound on the true-class logit minus every other upper bound is positive, the point is certified. Tighter relaxations (CROWN, linear programming, semidefinite programs) cost more. Training with the bound as the loss (Gowal et al. 2018; Wong and Kolter 2018) yields models with certified accuracy but lower clean accuracy.
- **Lipschitz-constrained networks.** If every logit difference is $L$-Lipschitz in $\ell_2$ and the margin is $m=f_y-\max_{j\ne y}f_j$, then no perturbation of norm smaller than $m/(\sqrt2 L)$ can change the prediction. Spectral-normalized and orthogonal layers keep $L$ small by construction.
- **Randomized smoothing.** Cohen, Rosenfeld and Kolter (ICML 2019) define a smoothed classifier $g(\boldsymbol{x})=\arg\max_c\ \P_{\boldsymbol{\eta}\sim\mathcal{N}(0,\sigma^2 I)}[f(\boldsymbol{x}+\boldsymbol{\eta})=c]$. If the top class has probability at least $p_A$ and the runner-up at most $p_B$, then $g$ is constant within $\ell_2$ radius $R=\tfrac{\sigma}{2}\big(\Phi^{-1}(p_A)-\Phi^{-1}(p_B)\big)$, where $\Phi^{-1}$ is the standard normal quantile. Probabilities are estimated by Monte Carlo sampling with a confidence bound, so the certificate holds with high probability. It scales to ImageNet because it only needs black-box evaluations of $f$. The price is slow inference (hundreds to thousands of samples) and certificates that are small in the $\ell_\infty$ sense for high-dimensional inputs.

Certified accuracy is always below empirical robust accuracy for the same radius, because it must hold against *every* attack. The gap between what attacks achieve and what certificates prove is itself a measure of how much we do not know.

### Transferability and black-box attacks

An adversarial example crafted against model A often fools model B, even with a different architecture or training set. Papernot, McDaniel and Goodfellow (2016) and Liu et al. (2017) showed that this enables **transfer attacks**: train a surrogate, attack it with white-box methods, and send the result to the victim. Transfer is stronger when the models share training data, when the surrogate is simpler (smoother), and when attacks are regularized to avoid over-fitting to the surrogate (momentum, input diversity, ensembles of surrogates). Transfer is also why "hide the weights" is not a defense. Other black-box options are **score-based** attacks that estimate the gradient from queried probabilities (finite differences, NES, SPSA) and **decision-based** attacks such as the Boundary Attack (Brendel et al. 2018) that need only the predicted label.

:::warning Gradient masking
A defense that makes gradients useless (non-differentiable preprocessing, randomization, saturating activations, shattered gradients) makes gradient attacks fail without making the model robust. Warning signs: single-step attacks beat iterative ones, black-box or transfer attacks beat white-box ones, increasing $\epsilon$ does not monotonically lower accuracy, and unbounded attacks do not reach 0%. Test with adaptive attacks (BPDA, EOT) and a standard suite such as AutoAttack.
:::

:::note Beyond images
The same ideas apply to text (synonym swaps, character edits, universal suffixes that jailbreak language models, Zou et al. 2023), audio, malware and reinforcement-learning policies. Discrete inputs break the gradient step, so attacks use gradient-guided search over tokens. Prompt injection, where untrusted text carries instructions to an LLM agent, is the adversarial-example problem again, but with no tidy norm ball.
:::

## Distribution shift and out-of-distribution detection {#shift}

### What can change

Every guarantee in supervised learning assumes the test data are drawn from the same distribution as the training data. Deployment violates this constantly. Write the joint distribution as $p(\boldsymbol{x},y)=p(y\mid\boldsymbol{x})\,p(\boldsymbol{x})=p(\boldsymbol{x}\mid y)\,p(y)$. Different factorizations make different shifts easy to describe.

| Shift | What changes | What stays | Example | Standard fix |
| --- | --- | --- | --- | --- |
| **Covariate shift** | $p(\boldsymbol{x})$ | $p(y\mid\boldsymbol{x})$ | A camera is replaced; photos are brighter | Importance weighting $w(\boldsymbol{x})=p_{\text{test}}(\boldsymbol{x})/p_{\text{train}}(\boldsymbol{x})$, augmentation, domain adaptation |
| **Label (prior) shift** | $p(y)$ | $p(\boldsymbol{x}\mid y)$ | A disease becomes more prevalent | Re-estimate class priors (BBSE, EM of Saerens et al.), re-weight the posterior |
| **Concept shift** | $p(y\mid\boldsymbol{x})$ | maybe $p(\boldsymbol{x})$ | What counts as "spam" or "creditworthy" evolves | Monitoring, retraining, continual learning |
| **Subpopulation shift** | mixture weights of groups | within-group distributions | A face model is trained mostly on one demographic | Group-robust training (Group DRO), data collection |
| **Novel class / OOD input** | support of $\boldsymbol{x}$ | | A cat appears to a digit classifier | Detection, abstention |

Under covariate shift you can correct the loss: $\E_{p_{\text{test}}}[\ell]=\E_{p_{\text{train}}}\!\big[w(\boldsymbol{x})\,\ell\big]$, provided the support of $p_{\text{train}}$ covers that of $p_{\text{test}}$ and the weights do not have exploding variance. Weights are typically estimated with a **domain classifier**: train a model to tell train from test inputs, then $w(\boldsymbol{x})\propto \frac{c(\boldsymbol{x})}{1-c(\boldsymbol{x})}$ where $c$ is its output.

Shift is not hypothetical. Benchmarks such as ImageNet-C (Hendrycks and Dietterich, 2019: 15 corruption types at 5 severities), ImageNet-A and ImageNet-R, and WILDS (Koh et al., 2021, with real shifts across hospitals, cameras and time) show that accuracy collapses under shifts that humans shrug off. A recurring empirical finding (Recht et al. 2019, Taori et al. 2020, Miller et al. 2021) is that in-distribution accuracy and out-of-distribution accuracy are strongly (often linearly, after a probit transform) correlated across many models: better models are better out of distribution too. Robustness interventions that move a model *off* this line, such as training on far more diverse data, as with CLIP-style models, are the rare and interesting ones.

{{figure: week17-shift-types | Three canonical shifts in a 2D classification problem. Covariate shift moves the inputs but not the labelling rule; label shift changes the class proportions; concept shift changes the rule itself, which no amount of unlabeled data can detect.}}

### The OOD detection problem

A related, narrower task: given an input, decide whether it comes from the training distribution (**in-distribution**, ID) or from something the model was not trained for (**out-of-distribution**, OOD). We want a score $s(\boldsymbol{x})$ that is high on ID inputs and low on OOD inputs, thresholded to flag suspicious ones. Evaluation uses threshold-free metrics: **AUROC** of ID versus OOD, **AUPR**, and **FPR@95TPR** (the fraction of OOD inputs accepted when 95% of ID inputs are accepted). Choose the OOD set carefully: "CIFAR-10 versus SVHN" is easy; "CIFAR-10 versus CIFAR-100" (near-OOD) is much harder.

### Max softmax probability (MSP)

The simplest baseline (Hendrycks and Gimpel, 2017) uses the classifier's own confidence: $s_{\text{MSP}}(\boldsymbol{x})=\max_c\,\softmax(\boldsymbol{z})_c$ where $\boldsymbol{z}=f(\boldsymbol{x})$ are the logits. It works surprisingly well, and its main failure is structural: softmax is shift-invariant and its confidence can be made arbitrarily high far from the training data (a ReLU network is piecewise linear, so scaling the input scales the logits and the softmax saturates; Hein, Andriushchenko and Bitterwolf 2019 proved this). Overconfident OOD predictions are not a bug of a particular network, they are generic.

ODIN (Liang, Li and Srikant, 2018) improves MSP with temperature scaling of the softmax and a small gradient-based input perturbation that increases the max-softmax score; the perturbation helps ID inputs more than OOD ones, widening the gap.

### Energy score

Interpret the logits as defining an energy-based model: $E(\boldsymbol{x};f)=-T\log\sum_c e^{z_c/T}$ (negative temperature-scaled log-sum-exp). The Gibbs distribution gives $p(\boldsymbol{x})\propto e^{-E(\boldsymbol{x})/T}$, so *low energy* means *high density* under the model. Liu et al. (NeurIPS 2020) use $s_{\text{energy}}(\boldsymbol{x})=T\log\sum_c e^{z_c/T}$ as the ID score. Why does it beat MSP? The softmax normalizes away the overall magnitude of the logits: MSP depends only on *differences* between logits, while the log-sum-exp also reflects how large the logits are in absolute terms, and ID data tend to produce larger logits. Concretely, $\log\max_c p_c = z_{\max}-\log\sum_c e^{z_c}$, which couples a growing numerator and a growing denominator, so the two cancel to a large extent. The energy score keeps the denominator term alone.

### Mahalanobis distance

A feature-space approach (Lee et al., NeurIPS 2018) assumes that the penultimate-layer features $\boldsymbol{h}$ of each class are Gaussian with class mean $\boldsymbol{\mu}_c$ and a *shared* covariance $\Sigma$ (this is the assumption behind linear discriminant analysis). The score is the negative squared Mahalanobis distance to the closest class mean:

$$
s_{\text{Maha}}(\boldsymbol{x})=-\min_c\ (\boldsymbol{h}(\boldsymbol{x})-\boldsymbol{\mu}_c)^\top\Sigma^{-1}(\boldsymbol{h}(\boldsymbol{x})-\boldsymbol{\mu}_c).
$$

Parameters are estimated from the training features: $\boldsymbol{\mu}_c$ is the class mean and $\Sigma$ the within-class covariance averaged over classes. The Mahalanobis metric whitens the features, so directions with low variance (where a deviation is rare among ID data) count more than high-variance directions. It is a *generative* detector built on a discriminative network and it can use features from several layers. Its weakness is the Gaussian assumption and sensitivity to $\Sigma$ estimation in high dimension (shrinkage helps).

```python:ood_scores.py
import torch
import torch.nn.functional as F


def msp_score(logits):
    """Max softmax probability; high = in-distribution."""
    return F.softmax(logits, dim=1).max(dim=1).values


def energy_score(logits, T=1.0):
    """T * logsumexp(z / T); high = in-distribution (negative energy)."""
    return T * torch.logsumexp(logits / T, dim=1)


@torch.no_grad()
def fit_class_gaussians(feats, labels, num_classes):
    """Class means and a SHARED (tied) covariance from penultimate features (N, D)."""
    D = feats.shape[1]
    mus = torch.stack([feats[labels == c].mean(0) for c in range(num_classes)])  # (K, D)
    centered = feats - mus[labels]
    cov = centered.T @ centered / feats.shape[0]
    cov += 1e-4 * torch.eye(D, device=feats.device)                              # shrinkage for stability
    return mus, torch.linalg.inv(cov)


def mahalanobis_score(feats, mus, prec):
    """Negative minimum squared Mahalanobis distance to any class mean."""
    diff = feats[:, None, :] - mus[None, :, :]                  # (N, K, D)
    d2 = torch.einsum("nkd,de,nke->nk", diff, prec, diff)       # (N, K)
    return -d2.min(dim=1).values


def auroc(score_id, score_ood):
    """Probability that a random ID sample scores higher than a random OOD sample (rank statistic)."""
    s = torch.cat([score_id, score_ood])
    labels = torch.cat([torch.ones_like(score_id), torch.zeros_like(score_ood)])
    order = s.argsort()
    ranks = torch.empty_like(order, dtype=torch.float)
    ranks[order] = torch.arange(1, len(s) + 1, dtype=torch.float)
    n1, n0 = len(score_id), len(score_ood)
    return ((ranks[labels == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)).item()
```

:::warning What OOD detection cannot do
No detector is universally good. Without access to OOD data there is a fundamental ambiguity about what "different" means, and results are very dependent on the ID/OOD pair. Evaluations that tune the detector's hyperparameters on the test OOD set overstate performance. Detectors trained with **outlier exposure** (Hendrycks, Mazeika and Dietterich, 2019: train the classifier to output a uniform distribution on a large auxiliary set of outliers) generalize well to unseen OOD data in practice, but they are not guaranteed to.
:::

## Uncertainty estimation {#uncertainty}

### Aleatoric and epistemic uncertainty

A softmax output is a number between zero and one, but it hides two very different kinds of "I don't know".

- **Aleatoric uncertainty** (from Latin *alea*, dice) is irreducible noise in the data-generating process: two identical-looking X-rays with different diagnoses, a blurry digit between a 4 and a 9, sensor noise. More data does not reduce it; it only lets you estimate it better. It can depend on the input (**heteroscedastic**): a self-driving system should be less sure at night than at noon.
- **Epistemic uncertainty** (from *episteme*, knowledge) is uncertainty about the *model*: many functions fit the finite training set and they disagree away from it. More data in the right places reduces it. This is the uncertainty that matters for OOD inputs, active learning and exploration.

Bayesian reasoning draws the line cleanly. With a posterior $p(\boldsymbol{\theta}\mid\mathcal{D})$ over parameters, the predictive distribution is
$$
p(y\mid\boldsymbol{x},\mathcal{D})=\int p(y\mid\boldsymbol{x},\boldsymbol{\theta})\,p(\boldsymbol{\theta}\mid\mathcal{D})\,d\boldsymbol{\theta}\;\approx\;\frac1M\sum_{m=1}^M p(y\mid\boldsymbol{x},\boldsymbol{\theta}_m),\quad\boldsymbol{\theta}_m\sim p(\boldsymbol{\theta}\mid\mathcal{D}).
$$
For classification, the total predictive entropy decomposes (using the mutual information between $y$ and $\boldsymbol{\theta}$):
$$
\underbrace{\mathcal{H}\big[\bar p(y\mid\boldsymbol{x})\big]}_{\text{total}}=\underbrace{\E_{\boldsymbol{\theta}}\,\mathcal{H}\big[p(y\mid\boldsymbol{x},\boldsymbol{\theta})\big]}_{\text{aleatoric (expected entropy)}}+\underbrace{\mathcal{I}\big[y;\boldsymbol{\theta}\mid\boldsymbol{x},\mathcal{D}\big]}_{\text{epistemic (disagreement)}},
$$
where $\bar p=\E_{\boldsymbol{\theta}}\,p(y\mid\boldsymbol{x},\boldsymbol{\theta})$. If every ensemble member says "50-50", the epistemic term is zero and the aleatoric term is $\log 2$. If half say "100% A" and half say "100% B", each member has zero entropy (aleatoric zero), the mean is 50-50, and the whole entropy is epistemic. The decomposition is the main reason ensembles are used for uncertainty rather than a single softmax.

For regression one can use a network that outputs both a mean and a variance, trained with the Gaussian negative log-likelihood $\mathcal{L}=\tfrac12\log\sigma^2(\boldsymbol{x})+\tfrac{(y-\mu(\boldsymbol{x}))^2}{2\sigma^2(\boldsymbol{x})}$. The learned $\sigma^2(\boldsymbol{x})$ is aleatoric. (Kendall and Gal, 2017, discuss combining both kinds in vision.) Practical caution: jointly learning $\mu$ and $\sigma$ can ignore hard examples by inflating $\sigma$; a warm-up phase or $\beta$-NLL (Seitzer et al., 2022) mitigates this.

### Deep ensembles

Lakshminarayanan, Pritzel and Blundell (NeurIPS 2017) proposed the strikingly simple recipe: train $M$ networks (typically 5) from different random initializations (and shuffled data order), then average the predicted probabilities. No Bayesian machinery is needed. Ensembles improve accuracy, calibration and OOD detection together. Why do they work well? Fort, Hu and Lakshminarayanan (2019) argued that independent runs from different initializations land in *different modes* of the loss landscape (functions that are far apart in function space), whereas samples from a local Bayesian approximation stay within a single mode. Averaging across modes captures more of the posterior's diversity. The cost is $M$ times the training and inference compute and memory; cheaper variants include snapshot ensembles, SWAG, BatchEnsemble and distillation of an ensemble into one model.

### MC dropout

Gal and Ghahramani (ICML 2016) showed that a network with dropout before weight layers, trained with standard dropout, can be interpreted as an approximate variational Bayesian model. At test time you keep dropout *on* and average $T$ stochastic forward passes:

$$
\bar p(y\mid\boldsymbol{x})\approx\frac1T\sum_{t=1}^T\softmax\big(f_{\boldsymbol{\theta}\odot\boldsymbol{m}_t}(\boldsymbol{x})\big),\qquad \boldsymbol{m}_t\sim\text{Bernoulli}(1-p).
$$

It is almost free to adopt (no retraining) and decent for detecting inputs far from the data, but the approximation is crude: the posterior it represents is a restricted family tied to the dropout rate, and it is typically less diverse than a deep ensemble. Treat it as a cheap baseline.

```python:uncertainty.py
import torch
import torch.nn as nn
import torch.nn.functional as F


def enable_mc_dropout(model):
    """Put the model in eval mode, but keep dropout layers stochastic."""
    model.eval()
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d)):
            m.train()


@torch.no_grad()
def predictive_distribution(models_or_passes):
    """Stack a list of (B, K) probability tensors into (M, B, K) and summarize."""
    P = torch.stack(models_or_passes)                       # (M, B, K)
    mean = P.mean(0)                                        # (B, K) predictive distribution
    eps = 1e-12
    total = -(mean * (mean + eps).log()).sum(-1)            # entropy of the mean
    aleatoric = -(P * (P + eps).log()).sum(-1).mean(0)      # expected entropy
    epistemic = total - aleatoric                           # mutual information (>= 0)
    return mean, total, aleatoric, epistemic


@torch.no_grad()
def mc_dropout_predict(model, x, T=30):
    enable_mc_dropout(model)
    passes = [F.softmax(model(x), dim=1) for _ in range(T)]
    return predictive_distribution(passes)


@torch.no_grad()
def ensemble_predict(models, x):
    passes = []
    for m in models:
        m.eval()
        passes.append(F.softmax(m(x), dim=1))
    return predictive_distribution(passes)
```

### Calibration

A classifier is **calibrated** if its confidence matches its accuracy: among all predictions made with confidence 0.8, 80% should be correct. Formally, with prediction $\hat y$ and confidence $\hat p=\max_c p_c$, perfect calibration means $\P(\hat y=y\mid\hat p=p)=p$ for all $p\in[0,1]$. Calibration matters whenever the probability is *used*: to decide whether to defer to a human, to combine with costs, to feed into another model.

Guo, Pleiss, Sun and Weinberger (ICML 2017) made a point that surprised many: modern deep networks (ResNets, DenseNets) are much *more overconfident* than the shallower LeNet-era networks, even though they are more accurate. They traced the effect to model capacity, batch normalization, weight decay and, above all, training to near-zero cross-entropy: once the training set is separated the loss can only go down by inflating logit norms.

**Reliability diagram.** Bin test predictions by confidence into $B$ bins $I_1,\dots,I_B$ (equal width, e.g. 10 bins). For each bin $b$ compute accuracy $\mathrm{acc}(b)=\frac1{\abs{I_b}}\sum_{i\in I_b}\ind[\hat y_i=y_i]$ and mean confidence $\mathrm{conf}(b)$. Plot accuracy against confidence; the diagonal is perfect calibration; bars below the diagonal indicate overconfidence.

**Expected Calibration Error (ECE)** summarizes the diagram:
$$
\mathrm{ECE}=\sum_{b=1}^B\frac{\abs{I_b}}{n}\,\big|\mathrm{acc}(b)-\mathrm{conf}(b)\big|.
$$
ECE is a weighted gap between the bars and the diagonal. It has caveats: it depends on the number and placement of bins (use equal-mass bins or more samples to reduce bias), it only looks at the top-label confidence, and a classifier that always predicts the base rates can have ECE near zero while being useless. Report it together with accuracy, NLL and the Brier score. Proper scoring rules (NLL, Brier) reward calibration *and* sharpness (confident correct predictions), whereas ECE rewards calibration alone.

```python:calibration.py
import torch
import torch.nn.functional as F


def reliability_bins(probs, labels, n_bins=10):
    """Return per-bin (count, mean confidence, accuracy) and ECE for top-label confidence."""
    conf, pred = probs.max(dim=1)
    correct = (pred == labels).float()
    edges = torch.linspace(0, 1, n_bins + 1)
    ece, rows = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            acc_b, conf_b = correct[m].mean().item(), conf[m].mean().item()
            ece += m.float().mean().item() * abs(acc_b - conf_b)
            rows.append((int(m.sum()), conf_b, acc_b))
        else:
            rows.append((0, float("nan"), float("nan")))
    return rows, ece


def fit_temperature(logits, labels, steps=200, lr=0.05):
    """Learn a scalar T > 0 minimizing NLL of softmax(logits / T) on a held-out set."""
    log_t = torch.zeros(1, requires_grad=True)       # T = exp(log_t) keeps T positive
    opt = torch.optim.Adam([log_t], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), labels)
        loss.backward()
        opt.step()
    return log_t.exp().item()
```

**Temperature scaling.** The simplest post-hoc fix (Guo et al.): divide all logits by a single scalar $T>0$ learned on a held-out validation set by minimizing NLL, $\hat p=\softmax(\boldsymbol{z}/T)$. Because dividing all logits by the same positive number never changes their order, the **predicted class and accuracy are unchanged**; only the confidence moves. $T>1$ softens (reduces overconfidence), $T<1$ sharpens. The NLL is convex in $1/T$ (it is a log-sum-exp minus a linear term), so the fit is a one-dimensional convex problem and works with only a few hundred validation examples. More flexible variants (vector scaling, matrix scaling, Dirichlet calibration) need more data and often overfit. Temperature scaling fixes only *marginal* miscalibration; it does not help with distribution shift (calibration degrades there, Ovadia et al. 2019, where deep ensembles were the strongest method among those compared).

To make this concrete, I simulated a 10-class classifier whose logits are 2.5 times too large (a perfectly calibrated logit vector multiplied by 2.5), on 5000 samples. It is 52% accurate and its ECE is 0.254; temperature scaling with $T=2.5$ brings the ECE to 0.014 without changing a single prediction.

```chart
{
  "type": "bar",
  "caption": "Reliability diagram (accuracy per confidence bin) of the simulated overconfident 10-class classifier, versus its calibrated version after temperature scaling (T = 2.5). Bin centres are mean confidences; bars below the diagonal mean overconfidence. Computed with NumPy: ECE 0.254 before, 0.014 after.",
  "height": 340,
  "data": {
    "labels": ["0.27", "0.36", "0.46", "0.55", "0.65", "0.75", "0.85", "0.97"],
    "datasets": [
      {"label": "Accuracy, before (overconfident)", "data": [0.222, 0.259, 0.304, 0.390, 0.373, 0.464, 0.472, 0.712], "backgroundColor": "var(--c-red)"},
      {"label": "Perfect calibration (accuracy = confidence)", "data": [0.269, 0.359, 0.456, 0.551, 0.650, 0.751, 0.852, 0.971], "backgroundColor": "var(--c-muted)"}
    ]
  },
  "options": { "scales": { "x": {"title": {"display": true, "text": "mean confidence in the bin (before scaling)"}}, "y": {"title": {"display": true, "text": "accuracy"}, "min": 0, "max": 1} } }
}
```

```chart
{
  "type": "line",
  "caption": "Validation NLL of the same classifier as a function of the temperature T. The curve is smooth with a single minimum near T = 2.5, which is the temperature that undoes the logit inflation. Computed with NumPy.",
  "height": 300,
  "data": {
    "labels": [0.5, 1, 1.5, 2, 2.5, 3, 4],
    "datasets": [ {"label": "NLL", "data": [3.382, 1.916, 1.530, 1.410, 1.388, 1.406, 1.484], "borderColor": "var(--c-accent)", "tension": 0.3} ]
  },
  "options": { "scales": { "x": {"title": {"display": true, "text": "temperature T"}}, "y": {"title": {"display": true, "text": "negative log-likelihood"}} } }
}
```

{{demo: week17-calibration-explorer | Reliability diagram, ECE and temperature scaling | 560}}

### Conformal prediction

Calibration says "when I say 80%, I am right 80% of the time on average". It does not say anything about *this* input, and requires trusting the model. **Conformal prediction** (Vovk, Gammerman and Shafer; modern introduction by Angelopoulos and Bates, 2021) wraps any model in a procedure that outputs a **set** of labels $\mathcal{C}(\boldsymbol{x})$ with a finite-sample, distribution-free guarantee, assuming only that calibration and test points are exchangeable (e.g. i.i.d. from the same distribution).

**Split conformal recipe** for target miscoverage $\alpha$ (say $0.1$ for 90% coverage):

1. Hold out a calibration set $\{(\boldsymbol{x}_i,y_i)\}_{i=1}^n$ unused in training.
2. Define a **nonconformity score** $s(\boldsymbol{x},y)$, larger meaning "this label looks worse for this input". A common choice is $s(\boldsymbol{x},y)=1-p_y(\boldsymbol{x})$, one minus the softmax probability of the true label.
3. Compute scores $s_i=s(\boldsymbol{x}_i,y_i)$ on the calibration set and let $\hat q$ be the $\lceil(n+1)(1-\alpha)\rceil/n$ empirical quantile of $s_1,\dots,s_n$.
4. For a new input, output $\mathcal{C}(\boldsymbol{x}_{\text{new}})=\{y:\ s(\boldsymbol{x}_{\text{new}},y)\le\hat q\}$.

:::math Why the coverage guarantee holds
Let $s_{n+1}=s(\boldsymbol{x}_{n+1},y_{n+1})$ be the score of the (unseen) test point with its true label. The event "$y_{n+1}\in\mathcal{C}(\boldsymbol{x}_{n+1})$" is exactly "$s_{n+1}\le\hat q$". Since the $n+1$ scores $s_1,\dots,s_{n+1}$ are exchangeable, the rank of $s_{n+1}$ among them is uniform on $\{1,\dots,n+1\}$ (ignoring ties). The event $s_{n+1}\le\hat q$, where $\hat q$ is the $k$-th smallest calibration score with $k=\lceil(n+1)(1-\alpha)\rceil$, occurs iff the test score's rank is at most $k$, which has probability $k/(n+1)\ge 1-\alpha$. So
$$
1-\alpha\ \le\ \P\big(y_{n+1}\in\mathcal{C}(\boldsymbol{x}_{n+1})\big)\ \le\ 1-\alpha+\frac1{n+1}.
$$
The upper bound holds when scores are almost surely distinct. This is a **marginal** guarantee: it averages over the calibration draw and the test point. It does *not* promise $90\%$ coverage for each class, subgroup or individual input (conditional coverage is impossible to guarantee distribution-free), and it breaks under distribution shift because exchangeability fails.
:::

The size of the set carries the uncertainty: a confident model yields singletons, a confused one returns several labels, and a hopeless input returns many. Better models yield smaller sets at the same coverage. Adaptive scores (APS, Romano et al. 2020; RAPS, Angelopoulos et al. 2021) give sets whose size tracks difficulty better than $1-p_y$. For regression, **conformalized quantile regression** (Romano, Patterson and Candes, 2019) gives intervals with adaptive width.

```python:conformal.py
import numpy as np


def split_conformal_threshold(cal_probs, cal_labels, alpha=0.1):
    """cal_probs: (n, K) softmax outputs on a held-out calibration set; returns the quantile q_hat."""
    n = len(cal_labels)
    scores = 1.0 - cal_probs[np.arange(n), cal_labels]          # nonconformity of the true label
    k = int(np.ceil((n + 1) * (1 - alpha)))                      # finite-sample corrected rank
    k = min(k, n)
    return np.sort(scores)[k - 1]


def prediction_sets(probs, q_hat):
    """Boolean mask (m, K): label j is in the set iff 1 - p_j <= q_hat."""
    return (1.0 - probs) <= q_hat


def evaluate(sets, labels):
    covered = sets[np.arange(len(labels)), labels].mean()
    avg_size = sets.sum(1).mean()
    return covered, avg_size
```

```chart
{
  "type": "bar",
  "caption": "Empirical marginal coverage of split conformal prediction versus the nominal level (1 - alpha), averaged over 2000 random calibration sets of size n = 100 with exchangeable scores. Computed with NumPy. The guarantee says coverage is at least 1 - alpha and at most 1 - alpha + 1/(n+1).",
  "height": 300,
  "data": {
    "labels": ["alpha = 0.05", "alpha = 0.10", "alpha = 0.20", "alpha = 0.30"],
    "datasets": [
      {"label": "Nominal 1 - alpha", "data": [0.95, 0.90, 0.80, 0.70], "backgroundColor": "var(--c-muted)"},
      {"label": "Empirical coverage", "data": [0.954, 0.899, 0.794, 0.706], "backgroundColor": "var(--c-accent)"}
    ]
  },
  "options": { "scales": { "y": {"title": {"display": true, "text": "coverage"}, "min": 0.5, "max": 1} } }
}
```

{{figure: week17-conformal | Split conformal prediction. Scores on a calibration set fix a threshold $\hat q$; any label whose score for the new input falls below the threshold enters the prediction set.}}

## Interpretability and explainability {#interpretability}

### What we want from an explanation

An explanation should help a person do something: debug a model, decide whether to trust a prediction, discover a spurious cue, satisfy a regulator, or learn something about the world. Keep two properties apart. **Faithfulness** asks whether the explanation reflects what the model actually computes. **Plausibility** asks whether it looks reasonable to a human. They are different, and visually pleasing maps can be unfaithful. Adebayo et al. (NeurIPS 2018, "Sanity checks for saliency maps") showed that several popular methods produce nearly the same map when the network's weights are randomized, meaning the map reflected image edges more than the model. Always run such sanity checks: randomize the weights (a faithful explanation must change), and measure **deletion** and **insertion** curves (remove the most important pixels first; a faithful map makes the confidence fall fast).

### Gradient saliency and its relatives

The simplest attribution is the gradient of the class score with respect to the input, $S_c(\boldsymbol{x})=\partial z_c/\partial\boldsymbol{x}$ (Simonyan, Vedaldi and Zisserman, 2014, whose saliency map is the per-pixel maximum of the absolute value over colour channels). By the first-order Taylor expansion $z_c(\boldsymbol{x}+\boldsymbol{\delta})\approx z_c(\boldsymbol{x})+S_c^\top\boldsymbol{\delta}$, it measures *local sensitivity*: how much a small change at each pixel changes the score. This is the same quantity that FGSM uses, and the maps look just as noisy for the same reason: the gradient of a ReLU network fluctuates rapidly (it is piecewise constant), so neighbouring pixels can have unrelated gradients.

Related variants: **Gradient times input** ($\boldsymbol{x}\odot\partial z_c/\partial\boldsymbol{x}$), **guided backpropagation** (Springenberg et al. 2015, which zeroes negative gradients at ReLUs and is notorious for acting like an edge detector independent of the weights) and **DeconvNet**. The first-order view is also a limitation: gradients are *local*, so saturated units give zero gradient even when the feature is decisive.

### SmoothGrad

Smilkov et al. (2017) reduce the noise by averaging gradients of noisy copies of the input:
$$
\hat S_c(\boldsymbol{x})=\frac1N\sum_{n=1}^N\frac{\partial z_c}{\partial\boldsymbol{x}}\Big|_{\boldsymbol{x}+\boldsymbol{\eta}_n},\qquad\boldsymbol{\eta}_n\sim\mathcal{N}(0,\sigma^2 I).
$$
This is the gradient of the Gaussian-smoothed score $\E_{\boldsymbol{\eta}}[z_c(\boldsymbol{x}+\boldsymbol{\eta})]$, which removes the high-frequency fluctuations. The noise scale $\sigma$ is typically 10-20% of the input range and $N$ about 50. Smoothing is the same trick that underlies randomized smoothing for certified defenses, applied to a different goal. It improves visual quality; whether it improves *faithfulness* depends on the test used.

### Grad-CAM

Saliency maps are pixel-level and noisy. CNNs, though, build up *spatial* feature maps in their last convolutional layer, with each channel acting like a detector for some pattern, and the class decision is a function of those channels. **Grad-CAM** (Selvaraju et al., ICCV 2017) produces a coarse localization map for class $c$ from that layer. Let $A^k\in\R^{H\times W}$ be the activation of channel $k$ (there are $K$ channels) in the chosen convolutional layer, and $y^c$ the *pre-softmax* score of class $c$.

:::math Derivation of Grad-CAM from a global-average-pooling network
Start with the simplest architecture that supports a class activation map (CAM, Zhou et al. 2016): the last conv layer is followed by global average pooling and a linear layer,
$$
y^c=\sum_k w^c_k\,F^k,\qquad F^k=\frac1{Z}\sum_{i,j}A^k_{ij},\quad Z=HW.
$$
Substituting, $y^c=\sum_{i,j}\frac1Z\sum_k w^c_kA^k_{ij}$. So $y^c$ is a sum over spatial positions of $L^c_{ij}=\frac1Z\sum_kw^c_kA^k_{ij}$, and the **class activation map** $\sum_kw^c_kA^k_{ij}$ tells how much each position contributes. Now compute the gradient of $y^c$ with respect to the activation $A^k_{ij}$: since $A^k_{ij}$ appears only in $F^k$, $\frac{\partial y^c}{\partial A^k_{ij}}=\frac{\partial y^c}{\partial F^k}\cdot\frac{\partial F^k}{\partial A^k_{ij}}=w^c_k\cdot\frac1Z$. Summing over positions gives $\sum_{i,j}\frac{\partial y^c}{\partial A^k_{ij}}=Z\cdot\frac{w^c_k}{Z}=w^c_k$, so the weight of channel $k$ equals the **globally averaged gradient** up to the constant $Z$:
$$
\alpha^c_k=\frac1Z\sum_{i,j}\frac{\partial y^c}{\partial A^k_{ij}}\ \ \Longrightarrow\ \ \alpha^c_k\propto w^c_k.
$$
This identity is the key: in a GAP network, the CAM channel weights are the pooled gradients. Grad-CAM *defines* the weights by that formula for *any* differentiable network (with a conv layer, followed by arbitrary layers, e.g. fully connected heads, or captioning models), giving
$$
\boxed{\;L^c_{\text{Grad-CAM}}=\relu\Big(\sum_k\alpha^c_k A^k\Big),\qquad\alpha^c_k=\frac1Z\sum_{i,j}\frac{\partial y^c}{\partial A^k_{ij}}\;}
$$
The ReLU keeps only features with a *positive* influence on class $c$ (negative regions likely belong to other classes). The map has the resolution of the conv layer ($7\times7$ for ResNet-50's last block on a $224\times224$ image) and is upsampled bilinearly to the image size for display.
:::

**Practicalities.** Choose the last conv layer (deeper means more semantic, coarser; shallower means finer, less class-specific). Use the logit, not the softmax probability: the softmax gradient of a class is dominated by competition with other classes and can vanish when the class is already confident. Guided Grad-CAM multiplies the upsampled map by guided backpropagation for a high-resolution, class-discriminative picture, but inherits guided backprop's weight-insensitivity problem. Grad-CAM++ (Chattopadhay et al., 2018) uses higher-order terms to better handle multiple instances of a class.

{{figure: week17-gradcam | The Grad-CAM pipeline: forward to the last convolutional feature maps, backpropagate the class logit to those maps, pool the gradients spatially to get one weight per channel, take the weighted sum of channels, apply ReLU and upsample.}}

```python:gradcam.py
import torch
import torch.nn.functional as F
import torchvision


class GradCAM:
    """Grad-CAM for any CNN. `layer` is the nn.Module whose output (B, K, H, W) is explained."""

    def __init__(self, model, layer):
        self.model, self.acts = model.eval(), None
        layer.register_forward_hook(self._save_activation)

    def _save_activation(self, module, inp, out):
        self.acts = out
        out.retain_grad()                      # we want d y^c / d A for this non-leaf tensor

    def __call__(self, x, class_idx=None):
        logits = self.model(x)                 # (B, num_classes), pre-softmax scores
        if class_idx is None:
            class_idx = logits.argmax(1)       # explain the predicted class
        score = logits.gather(1, class_idx.view(-1, 1)).sum()
        self.model.zero_grad(set_to_none=True)
        score.backward()
        A, G = self.acts.detach(), self.acts.grad          # both (B, K, H, W)
        alpha = G.mean(dim=(2, 3), keepdim=True)           # (B, K, 1, 1): global-average-pooled gradients
        cam = F.relu((alpha * A).sum(dim=1, keepdim=True)) # (B, 1, H, W)
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam - cam.amin(dim=(2, 3), keepdim=True)
        cam = cam / (cam.amax(dim=(2, 3), keepdim=True) + 1e-8)   # normalize to [0, 1] per image
        return cam.squeeze(1), class_idx


if __name__ == "__main__":
    model = torchvision.models.resnet18(weights="IMAGENET1K_V1")
    cam_fn = GradCAM(model, model.layer4)       # last residual stage, output (B, 512, 7, 7)
    x = torch.randn(2, 3, 224, 224)             # replace by real, normalized images
    cam, cls = cam_fn(x)
    print(cam.shape, cls)                        # torch.Size([2, 224, 224])
```

{{demo: week17-saliency-gradcam | Saliency vs Grad-CAM on a synthetic image | 520}}

### Integrated Gradients

Plain gradients have a famous failure: **saturation**. If a feature pushes the logit to a plateau, the gradient there is zero even though the feature caused the output. Sundararajan, Taly and Yan (ICML 2017) proposed to accumulate gradients along a path from a **baseline** $\boldsymbol{x}'$ (a "no information" input such as a black image or zeros) to the input $\boldsymbol{x}$:

$$
\mathrm{IG}_i(\boldsymbol{x})=(x_i-x'_i)\int_0^1\frac{\partial F\big(\boldsymbol{x}'+t(\boldsymbol{x}-\boldsymbol{x}')\big)}{\partial x_i}\,dt,
$$

where $F$ is the scalar output being explained (e.g. the class logit or probability). In practice, a Riemann sum with $m$ steps (20 to 300) approximates the integral.

**Axioms.** The paper derives IG as the unique (up to path choice) method satisfying two desiderata, and it also satisfies several more.

- **Completeness** (sensitivity-a): attributions sum to the difference in output between input and baseline, $\sum_i\mathrm{IG}_i(\boldsymbol{x})=F(\boldsymbol{x})-F(\boldsymbol{x}')$. Proof: by the fundamental theorem of calculus applied to $g(t)=F(\boldsymbol{x}'+t(\boldsymbol{x}-\boldsymbol{x}'))$, $F(\boldsymbol{x})-F(\boldsymbol{x}')=\int_0^1g'(t)dt=\int_0^1\sum_i\frac{\partial F}{\partial x_i}(x_i-x'_i)\,dt=\sum_i\mathrm{IG}_i$. In code this is a free correctness check: if the sum is far from $F(\boldsymbol{x})-F(\boldsymbol{x}')$, increase the number of steps.
- **Sensitivity (b)**: if $F$ depends on feature $i$ and the input and baseline differ only in feature $i$ with different outputs, then $\mathrm{IG}_i\ne0$. Plain gradients violate this under saturation.
- **Implementation invariance**: two functionally equivalent networks get identical attributions, because IG depends only on gradients of the function, not the architecture. (Methods that propagate custom backward rules, such as DeepLIFT variants and LRP, can violate it.)
- **Linearity and symmetry-preservation**: attributions of a weighted sum of models are the weighted sum of attributions, and symmetric features get equal attributions.

:::warning The baseline is a modelling decision
The baseline defines what "absence" means, and attributions are always relative to it. A black baseline gives zero attribution to black pixels in the input regardless of their role. Alternatives are a blurred image, noise, the dataset mean, or averaging over several baselines (expected gradients, Erion et al. 2021). Never present an IG map without naming the baseline.
:::

```python:integrated_gradients.py
import torch


def integrated_gradients(model, x, target, baseline=None, steps=64):
    """x: (B, C, H, W); target: (B,) class indices. Returns attributions with the shape of x."""
    model.eval()
    if baseline is None:
        baseline = torch.zeros_like(x)
    # midpoint Riemann sum over alpha in (0, 1): shape (steps,) -> broadcast over the batch
    alphas = (torch.arange(steps, dtype=x.dtype, device=x.device) + 0.5) / steps
    total = torch.zeros_like(x)
    for a in alphas:
        xi = (baseline + a * (x - baseline)).requires_grad_(True)
        out = model(xi).gather(1, target.view(-1, 1)).sum()   # use the logit of the target class
        grad, = torch.autograd.grad(out, xi)
        total += grad
    attr = (x - baseline) * total / steps
    # completeness check: sum of attributions should be close to F(x) - F(baseline)
    with torch.no_grad():
        delta = (model(x).gather(1, target.view(-1, 1)) - model(baseline).gather(1, target.view(-1, 1))).squeeze(1)
    gap = (attr.flatten(1).sum(1) - delta).abs().max().item()
    print(f"max completeness error: {gap:.4f}  (increase `steps` if large)")
    return attr


def smoothgrad(model, x, target, n=50, sigma=0.15):
    """Average input-gradients over n noisy copies; sigma is a fraction of the input range."""
    model.eval()
    span = (x.max() - x.min()).item()
    total = torch.zeros_like(x)
    for _ in range(n):
        xn = (x + sigma * span * torch.randn_like(x)).requires_grad_(True)
        out = model(xn).gather(1, target.view(-1, 1)).sum()
        grad, = torch.autograd.grad(out, xn)
        total += grad
    return total / n
```

### Attention analysis and its caveats

Transformers expose attention weights $\mathrm{softmax}(QK^\top/\sqrt{d_k})$ that look like a ready-made explanation: "the model looked at these words". Jain and Wallace (2019, "Attention is not explanation") found that attention weights often correlate poorly with gradient-based importance and that very different attention distributions can give the same prediction; Wiegreffe and Pinter (2019, "Attention is not not explanation") replied that the answer depends on what claim you make and what you test. The caveats worth remembering:

- The attention output is a weighted sum of **value vectors**, which are then mixed by output projections, residual connections, MLPs and later layers. A large weight on a token with a small-norm value contributes little (Kobayashi et al. 2020 proposed norm-based analysis).
- Information is **mixed across layers**: by layer 6 the representation at position $i$ contains information from many positions, so raw attention in higher layers does not say where information came from. *Attention rollout* and *attention flow* (Abnar and Zuidema, 2020) multiply attention matrices across layers (with the residual identity added) to approximate the flow; they are heuristics.
- Multiple heads do different things, and many can be pruned (Michel et al. 2019; Voita et al. 2019), so a head's pattern is not necessarily important.
- Attention is a *mechanism*, not an *attribution*. Use it as a hypothesis generator, then test it by intervention (ablate or patch a head and see what changes).

### Probing and its pitfalls

A **probe** is a small classifier trained on frozen representations to predict a property (part of speech, syntactic depth, board state in a game, the colour of an object). If a linear probe predicts the property from layer $\ell$, the information is *linearly decodable* there. Alain and Bengio (2017) introduced linear classifier probes; Hewitt and Manning (2019) found structure in BERT's geometry reflecting syntax trees.

The pitfalls are significant and well documented.

1. **Probe capacity.** A powerful probe (an MLP) can learn the task itself from the raw input, so high accuracy says nothing about the representation. Hewitt and Liang (2019) proposed **control tasks**: the same probe on random labels per word type; *selectivity* is task accuracy minus control accuracy. Use simple (linear) probes, report baselines (probe on random-initialized network, on input embeddings), and compare with the best from-scratch model.
2. **Decodable is not used.** The model may encode a feature that the downstream computation ignores. To test *use*, intervene: edit the representation along the probe direction and see whether the output changes (Elazar et al., 2021, amnesic probing, removes the information and measures the effect; Ravfogel et al.'s INLP iteratively projects out linear directions).
3. **Correlation with confounds.** The probe may exploit a correlate (position, frequency, a shortcut) rather than the property.
4. **Mutual-information view.** Because accuracy depends on the probe's learning curve, information-theoretic probes (Voita and Titov, 2020, minimum description length) report the code length of the labels given the representations, which accounts for how much data the probe needs.

### Mechanistic interpretability (overview)

Mechanistic interpretability aims to reverse-engineer networks into human-understandable algorithms, treating the trained weights as a program to be understood rather than a black box to be summarized by correlations. The programme has a few central ideas.

- **Features and circuits.** Olah et al. (Distill, 2020, "Zoom in") argued that networks consist of **features** (directions in activation space that correspond to meaningful properties, such as curve detectors in vision models) connected by weights into **circuits**: subgraphs implementing an identifiable computation. In transformers, Elhage et al. (2021, "A mathematical framework for transformer circuits") decomposed the residual stream view, in which each attention head and MLP reads from and writes to a shared stream, and identified **induction heads** (Olsson et al. 2022): a pair of heads that implements "if token $A$ was followed by $B$ earlier in the context, and $A$ appears again, predict $B$", a mechanism behind in-context copying and linked to the sudden formation of in-context learning during training. Wang et al. (2022) traced a full circuit for indirect-object identification in GPT-2 small.
- **Causal tools.** Observing activations is not enough. **Activation patching** (also causal tracing, used by Meng et al., 2022, to locate factual associations in mid-layer MLPs) copies an activation from one run into another and measures the effect on the output. **Ablation** removes a component. **Logit lens** reads the residual stream through the unembedding at each layer. These interventions produce *causal* claims, unlike saliency.
- **Superposition.** If each neuron represented one feature, a network of $d$ neurons could represent only $d$ features. Real networks seem to represent more features than dimensions by storing them as *non-orthogonal* directions, relying on features being **sparse** (rarely active together) so interference is tolerable. Elhage et al. (2022, "Toy models of superposition") showed this in small ReLU networks: with sparse features, the optimal solution packs $n>d$ features into $d$ dimensions as geometric arrangements (pentagons in 2D). The consequence is **polysemantic neurons**: single neurons that respond to several unrelated things, making neuron-by-neuron analysis unreliable.
- **Sparse autoencoders (SAEs).** If activations $\boldsymbol{a}\in\R^d$ are sparse combinations of $n\gg d$ feature directions, learn them as a dictionary:
$$
\boldsymbol{f}=\relu(W_e(\boldsymbol{a}-\boldsymbol{b}_d)+\boldsymbol{b}_e)\in\R^n,\qquad\hat{\boldsymbol{a}}=W_d\boldsymbol{f}+\boldsymbol{b}_d,\qquad\mathcal{L}=\norm{\boldsymbol{a}-\hat{\boldsymbol{a}}}_2^2+\lambda\norm{\boldsymbol{f}}_1.
$$
The $\ell_1$ penalty encourages few active features per input, and each column of $W_d$ is a candidate feature direction. Bricken et al. (Anthropic, 2023, "Towards monosemanticity") and Cunningham et al. (2023) applied this to small language models and found many interpretable features; later work scaled SAEs to larger models (Templeton et al. 2024, "Scaling monosemanticity"; also top-$k$ and gated SAE variants). The picture is encouraging but incomplete: SAEs have reconstruction error that harms the model's loss when spliced in, features can split or merge as the dictionary grows, "dead" features occur, and it remains debated how faithfully features capture the model's own computation.

{{figure: week17-superposition | Superposition: with sparse features, a network can store more features than dimensions as non-orthogonal directions (left: five features in a 2D plane). A sparse autoencoder recovers a dictionary of such directions from activations (right).}}

```python:sparse_autoencoder.py
import torch
import torch.nn as nn


class SparseAutoencoder(nn.Module):
    """Overcomplete dictionary learning on activations a in R^d with n_features >> d."""

    def __init__(self, d, n_features):
        super().__init__()
        self.W_enc = nn.Linear(d, n_features)
        self.W_dec = nn.Linear(n_features, d, bias=False)
        self.b_dec = nn.Parameter(torch.zeros(d))
        with torch.no_grad():                       # unit-norm decoder columns
            self.W_dec.weight.copy_(nn.functional.normalize(self.W_dec.weight, dim=0))

    def forward(self, a):
        f = torch.relu(self.W_enc(a - self.b_dec))  # (B, n_features), sparse after training
        a_hat = self.W_dec(f) + self.b_dec          # (B, d)
        return a_hat, f

    @torch.no_grad()
    def renormalize(self):
        """Keep decoder columns at unit norm so the L1 penalty cannot be cheated by shrinking f."""
        self.W_dec.weight.copy_(nn.functional.normalize(self.W_dec.weight, dim=0))


def train_sae(acts, n_features=512, l1=1e-3, epochs=20, bs=256, lr=1e-3):
    sae = SparseAutoencoder(acts.shape[1], n_features)
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    for ep in range(epochs):
        perm = torch.randperm(len(acts))
        for i in range(0, len(acts), bs):
            a = acts[perm[i:i + bs]]
            a_hat, f = sae(a)
            loss = (a - a_hat).pow(2).sum(-1).mean() + l1 * f.abs().sum(-1).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sae.renormalize()
        l0 = (f > 0).float().sum(-1).mean().item()      # average number of active features
        print(f"epoch {ep}: loss {loss.item():.4f}  L0 {l0:.1f}")
    return sae
```

## Fairness and bias audits {#fairness}

### Where bias comes from

A model is not biased in a vacuum; harm arises along the whole pipeline. **Historical bias**: the world is unequal and data record it. **Representation bias**: some groups are underrepresented in the data (Buolamwini and Gebru, 2018, "Gender Shades", found that commercial gender classifiers had far higher error on darker-skinned women than on lighter-skinned men). **Measurement bias**: the label is a flawed proxy (arrests for crime, cost for health need; Obermeyer et al., 2019, showed a healthcare risk algorithm underestimated the needs of Black patients because cost was the target). **Aggregation bias**: one model for groups with different relationships. **Feedback loops**: model decisions change the data collected next. Language models inherit stereotypes from text (Bolukbasi et al. 2016 on word embeddings; Caliskan et al. 2017). Removing the protected attribute rarely helps ("fairness through unawareness") because other features are proxies for it.

### Group fairness metrics

Let $Y\in\{0,1\}$ be the true outcome, $\hat Y\in\{0,1\}$ the decision (e.g. approve), and $A$ the protected attribute (group). The three standard group criteria differ in what they condition on.

| Criterion | Requirement | Meaning |
| --- | --- | --- |
| **Demographic parity** (statistical parity) | $\P(\hat Y=1\mid A=a)=\P(\hat Y=1\mid A=b)$ | Same selection rate in every group |
| **Equalized odds** (Hardt, Price and Srebro, 2016) | $\P(\hat Y=1\mid Y=y,A=a)$ equal across $a$ for $y=0$ **and** $y=1$ | Same TPR and FPR in every group |
| **Equal opportunity** | equal TPR only ($y=1$) | Qualified people are accepted equally often |
| **Predictive parity / calibration within groups** | $\P(Y=1\mid\hat Y=1,A=a)$ (or $\P(Y=1\mid S=s,A=a)=s$) equal across groups | A score means the same thing in each group |

Demographic parity ignores the true outcome, so it can force unqualified acceptances if base rates differ, or be satisfied by a useless classifier. Equalized odds takes the labels as ground truth, which is only reasonable if the labels are not themselves biased. Choose the criterion by thinking about the harm: a false positive in recidivism (wrongly labeled high-risk) and a false negative in medical screening (missed disease) have different costs.

:::example A small audit by hand
A loan model on 1000 applicants per group. Group A: 500 repay, model approves 400 of them and 100 of the 500 who default. Group B: 200 repay, model approves 120 of them and 40 of the 800 who default.
- Approval rate: A $= 500/1000=0.50$, B $= 160/1000=0.16$. Demographic parity difference $=0.34$.
- TPR: A $=400/500=0.80$, B $=120/200=0.60$. FPR: A $=100/500=0.20$, B $=40/800=0.05$. Equalized-odds gaps $0.20$ and $0.15$.
- Precision (predictive parity): A $=400/500=0.80$, B $=120/160=0.75$.
Group B differs in base rate (20% versus 50%), which is exactly why the criteria disagree.
:::

### Impossibility results

One would like all of these at once. Kleinberg, Mullainathan and Raghavan (2017) and Chouldechova (2017) showed that this is mathematically impossible except in degenerate cases. A compact derivation: for group $a$ with base rate $p_a=\P(Y=1\mid A=a)$, positive predictive value $\mathrm{PPV}_a$, false positive rate $\mathrm{FPR}_a$ and false negative rate $\mathrm{FNR}_a$, simple Bayes algebra gives the identity

$$
\mathrm{FPR}_a=\frac{p_a}{1-p_a}\cdot\frac{1-\mathrm{PPV}_a}{\mathrm{PPV}_a}\cdot\big(1-\mathrm{FNR}_a\big).
$$

:::math Proof sketch
Write $\mathrm{PPV}=\frac{TP}{TP+FP}$, so $FP=TP\,\frac{1-\mathrm{PPV}}{\mathrm{PPV}}$. Divide both sides by the number of negatives $N_0=n(1-p)$ and write $TP=n\,p\,(1-\mathrm{FNR})$:
$$\mathrm{FPR}=\frac{FP}{N_0}=\frac{n\,p\,(1-\mathrm{FNR})}{n(1-p)}\cdot\frac{1-\mathrm{PPV}}{\mathrm{PPV}}.$$
Now suppose two groups have equal PPV (predictive parity) *and* equal FNR and FPR (equalized odds). The identity then forces $\frac{p_a}{1-p_a}=\frac{p_b}{1-p_b}$, i.e. equal base rates, unless the classifier is perfect ($\mathrm{PPV}=1$ and $\mathrm{FNR}=0$, making FPR $=0$). When base rates differ and the classifier is imperfect, you must give up at least one criterion.
:::

The practical reading: fairness is not a number that a training recipe can drive to zero. It requires a *normative choice* about which error to equalize, made with the stakeholders affected, and the existence of the trade-off is not evidence of a flawed model but of a real conflict (this is the core of the COMPAS controversy, ProPublica 2016 versus the vendor's reply). Individual fairness (Dwork et al., 2012: similar individuals receive similar outcomes under a task-appropriate similarity metric) and counterfactual fairness (Kusner et al., 2017: the decision would not change had the protected attribute been different in a causal model) offer alternative notions with their own requirements, namely a metric or a causal graph.

### Mitigation

Interventions are classified by where in the pipeline they act.

- **Pre-processing:** re-sample or re-weight (Kamiran and Calders), collect more data for under-represented groups (usually the highest-impact fix), repair features, remove proxies, and audit labels.
- **In-processing:** add a fairness penalty (e.g. difference in selection rates, or an adversary trying to predict $A$ from the representation, Zhang, Lemoine and Mitchell 2018; Edwards and Storkey 2016); Group DRO (Sagawa et al. 2020) minimizes the *worst-group* loss $\max_g\E_g[\ell]$ and needs strong regularization to avoid overfitting small groups.
- **Post-processing:** adjust decisions without retraining; Hardt et al. choose group-specific thresholds (possibly randomized) to equalize TPR and FPR. Legal and ethical constraints may forbid using the attribute at decision time, even when mitigating.

Each mitigation trades accuracy, one fairness criterion or another, and sometimes the other groups' welfare. Report subgroup metrics, *intersectional* groups (e.g. gender by skin type), and confidence intervals, since small groups give noisy estimates.

```python:fairness_audit.py
import numpy as np


def group_rates(y_true, y_pred, group):
    """Per-group selection rate, TPR, FPR and PPV. Inputs are 1D integer arrays."""
    out = {}
    for g in np.unique(group):
        m = group == g
        yt, yp = y_true[m], y_pred[m]
        tp = ((yp == 1) & (yt == 1)).sum()
        fp = ((yp == 1) & (yt == 0)).sum()
        fn = ((yp == 0) & (yt == 1)).sum()
        tn = ((yp == 0) & (yt == 0)).sum()
        out[g] = dict(
            n=int(m.sum()),
            base_rate=float(yt.mean()),
            selection=float(yp.mean()),
            tpr=tp / max(tp + fn, 1),
            fpr=fp / max(fp + tn, 1),
            ppv=tp / max(tp + fp, 1),
        )
    return out


def gaps(rates):
    """Largest between-group gap for each metric (0 means parity)."""
    keys = ["selection", "tpr", "fpr", "ppv"]
    return {k: max(r[k] for r in rates.values()) - min(r[k] for r in rates.values()) for k in keys}


def threshold_for_equal_tpr(scores, y_true, group, target_tpr=0.8):
    """Post-processing: per-group score threshold that achieves the same TPR in every group."""
    thr = {}
    for g in np.unique(group):
        s = scores[(group == g) & (y_true == 1)]
        thr[g] = np.quantile(s, 1 - target_tpr)     # accept the top `target_tpr` fraction of positives
    return thr


# Reproduces the worked example in the text (group A: 500/500 split, group B: 200/800).
yt = np.r_[np.ones(500), np.zeros(500), np.ones(200), np.zeros(800)].astype(int)
yp = np.r_[np.ones(400), np.zeros(100), np.ones(100), np.zeros(400), np.ones(120), np.zeros(80), np.ones(40), np.zeros(760)].astype(int)
grp = np.r_[np.zeros(1000), np.ones(1000)].astype(int)
r = group_rates(yt, yp, grp)
print(r)
print(gaps(r))
```

## Privacy {#privacy}

### What can leak

A trained model is a compressed, lossy summary of its training data, and summaries leak. Three increasingly serious attacks illustrate this.

- **Membership inference** (Shokri, Stronati, Song and Shmatikov, 2017): given a record and access to the model, decide whether the record was in the training set. The signal is the generalization gap: models are more confident, with lower loss, on points they trained on. The simplest attack thresholds the loss: predict "member" if $\mathcal{L}(f(\boldsymbol{x}),y)<\tau$. Shokri et al. trained shadow models to learn the decision rule; Carlini et al. (2022, LiRA) showed that comparing against *per-example* reference distributions, evaluated at low false-positive rates, is far more powerful than average-case accuracy suggests. Membership of a single patient in a "cancer clinic" training set is itself sensitive.
- **Model inversion and training-data extraction.** Fredrikson et al. (2015) reconstructed faces from a face-recognition model's confidence outputs. Carlini et al. (2021) extracted hundreds of verbatim training sequences from GPT-2 by sampling and filtering, and later work showed larger language models memorize more and that duplicated data is memorized most.
- **Memorization is not always overfitting.** Feldman (2020) argued that memorizing rare examples is *necessary* for optimal generalization on long-tailed data; a model can generalize well on average while memorizing individuals.

Empirically, membership attacks succeed more for small training sets, many epochs, large models and overfit training procedures. Regularization, early stopping and more data reduce, but do not eliminate, risk, and they give no *quantitative guarantee*. Differential privacy does.

### Differential privacy: the definition

Imagine two datasets $D$ and $D'$ that differ in exactly one person's record (**neighbouring datasets**). A randomized algorithm $\mathcal{M}$ is **$(\varepsilon,\delta)$-differentially private** (Dwork, McSherry, Nissim and Smith, 2006; the $\delta$ relaxation by Dwork et al., 2006) if for every pair of neighbours and every set of outputs $S$,

$$
\P\big[\mathcal{M}(D)\in S\big]\ \le\ e^{\varepsilon}\,\P\big[\mathcal{M}(D')\in S\big]+\delta.
$$

The intuition is that an observer looking at the output cannot tell much about whether any one person's data was included: the output distributions with and without that person are within a factor $e^\varepsilon$ of each other, except on an event of probability $\delta$. The smaller $\varepsilon$, the stronger the privacy. The guarantee holds against an adversary with *arbitrary side information* and is a property of the *algorithm*, not of one output. Key facts:

- **Post-processing:** any computation on the output of a DP algorithm, without access to the data, remains DP with the same parameters.
- **Composition:** running $k$ mechanisms consumes budget. Basic composition gives $(k\varepsilon,k\delta)$; advanced composition improves this to roughly $\varepsilon\sqrt{2k\ln(1/\delta')}+k\varepsilon(e^\varepsilon-1)$; modern *accountants* (moments accountant, Renyi DP, Gaussian DP, privacy-loss distributions) are tighter and are what libraries use.
- **Interpretation of $\varepsilon$:** a hypothesis-testing view says that $\varepsilon$ bounds how well any test can distinguish $D$ from $D'$; $\varepsilon\approx 1$ is considered strong, 8-10 or more weak-but-still-meaningful, and $\delta$ should be much smaller than $1/n$ (typically $10^{-5}$ or lower). These are conventions, not laws; and the guarantee is on *record-level* membership, not on harm in general (it does not stop inference about people who are not in the data).

**Gaussian mechanism.** To release a function $q(D)\in\R^d$ with $\ell_2$ **sensitivity** $\Delta_2=\max_{D\sim D'}\norm{q(D)-q(D')}_2$, add noise $\mathcal{N}(0,\sigma^2I)$. The classical analysis shows it is $(\varepsilon,\delta)$-DP for $\varepsilon\in(0,1)$ when $\sigma\ge\Delta_2\sqrt{2\ln(1.25/\delta)}/\varepsilon$. The noise is calibrated to how much one person *can* change the answer, not to how much they actually did. That is why DP needs bounded sensitivity (clipping).

```chart
{
  "type": "line",
  "caption": "Noise scale needed by the classical Gaussian mechanism (sensitivity 1, delta = 1e-5) as a function of epsilon: sigma = sqrt(2 ln(1.25/delta)) / epsilon. The formula is valid for epsilon < 1; beyond that it is only indicative (tighter analyses exist). Computed with NumPy.",
  "height": 300,
  "data": {
    "labels": [0.1, 0.25, 0.5, 0.75, 1.0],
    "datasets": [ {"label": "sigma", "data": [48.45, 19.38, 9.69, 6.46, 4.84], "borderColor": "var(--c-purple)", "tension": 0.3} ]
  },
  "options": { "scales": { "x": {"title": {"display": true, "text": "epsilon"}}, "y": {"title": {"display": true, "text": "noise standard deviation sigma"}, "type": "logarithmic"} } }
}
```

### DP-SGD

Abadi et al. (CCS 2016, "Deep learning with differential privacy") made DP practical for deep learning. The recipe changes plain SGD in two places, and each has a reason.

1. **Per-example gradient clipping.** Compute the gradient $\boldsymbol{g}_i$ for *each* example, and rescale it to norm at most $C$: $\bar{\boldsymbol{g}}_i=\boldsymbol{g}_i/\max(1,\norm{\boldsymbol{g}_i}_2/C)$. Now one example's contribution to the summed gradient has norm at most $C$: the **sensitivity** is bounded. Without clipping, one outlier could move the update arbitrarily far, and no finite amount of noise would hide it.
2. **Gaussian noise.** Sum the clipped gradients over a batch of expected size $B$ (sampled by *Poisson subsampling*: each example is included independently with probability $q=B/N$), add $\mathcal{N}(0,\sigma^2C^2I)$ and divide by $B$:
$$
\tilde{\boldsymbol{g}}=\frac1B\Big(\sum_{i\in\text{batch}}\bar{\boldsymbol{g}}_i+\mathcal{N}(0,\sigma^2C^2 I)\Big),\qquad\boldsymbol{\theta}\leftarrow\boldsymbol{\theta}-\eta\,\tilde{\boldsymbol{g}}.
$$
The privacy cost of $T$ steps is computed by an accountant, using *amplification by subsampling*: since each step touches only a fraction $q$ of the data, its privacy loss is smaller than that of a full-batch Gaussian mechanism. Given $q$, $\sigma$, $T$ and $\delta$, the accountant returns the total $\varepsilon$.

{{figure: week17-dpsgd | One DP-SGD step. Per-example gradients are clipped to norm $C$, summed, perturbed with Gaussian noise of standard deviation $\sigma C$, averaged and applied. The privacy accountant tracks the cumulative $\varepsilon$ over steps.}}

What drives the privacy-utility trade-off? The noise added to the *average* gradient has standard deviation $\sigma C/B$ per coordinate, so (i) larger batches reduce the noise relative to the signal (DP-SGD works best with very large batches), (ii) more noise per unit $\varepsilon$ is needed as $T$ grows, so training must be short, (iii) clipping introduces *bias* when $C$ is much smaller than typical gradient norms, and (iv) the noise is added in all $p$ dimensions, so the total noise norm scales like $\sigma C\sqrt p/B$, hurting large models (though gradients often concentrate in a low-dimensional subspace and fine-tuning small adapters can work well). Practical results: DP training on CIFAR-10 reached accuracies in the 60-70% range at $\varepsilon\approx 8$ with careful tuning in early work, and later work with large batches, augmentation multiplicity and pretrained backbones (De et al., 2022, and others) closed much of the gap. **DP fine-tuning of a model pretrained on public data** is the dominant practical recipe.

{{demo: week17-dp-explorer | Differential privacy: noise, clipping and epsilon | 520}}

```python:dp_sgd_manual.py
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)


def dp_sgd_step(model, loss_fn, xb, yb, lr, clip, sigma):
    """One DP-SGD step using torch.func for per-example gradients.
    (Poisson sampling of the batch is assumed to have happened outside, and the privacy
    accountant is a separate component, e.g. from Opacus.)"""
    from torch.func import functional_call, vmap, grad
    params = {k: v.detach() for k, v in model.named_parameters()}

    def per_example_loss(p, x, y):
        out = functional_call(model, p, (x.unsqueeze(0),))
        return loss_fn(out, y.unsqueeze(0))

    g = vmap(grad(per_example_loss), in_dims=(None, 0, 0))(params, xb, yb)   # dict of (B, *shape)
    norms = torch.sqrt(sum(v.flatten(1).pow(2).sum(1) for v in g.values()))  # (B,) per-example L2 norm
    scale = torch.clamp(clip / (norms + 1e-12), max=1.0)                     # = 1 / max(1, ||g|| / C)
    B = xb.shape[0]
    with torch.no_grad():
        for name, p in model.named_parameters():
            clipped_sum = (g[name] * scale.view(-1, *[1] * (g[name].dim() - 1))).sum(0)
            noise = torch.randn_like(p) * sigma * clip
            p -= lr * (clipped_sum + noise) / B


# A tiny demonstration on random data
model = nn.Sequential(nn.Linear(20, 32), nn.ReLU(), nn.Linear(32, 2))
X, Y = torch.randn(256, 20), torch.randint(0, 2, (256,))
for step in range(5):
    idx = torch.rand(256) < 0.25                       # Poisson subsampling with q = 0.25
    dp_sgd_step(model, F.cross_entropy, X[idx], Y[idx], lr=0.5, clip=1.0, sigma=1.0)
    print(step, F.cross_entropy(model(X), Y).item())
```

In production use a library: **Opacus** (PyTorch) wraps a model, optimizer and loader with a `PrivacyEngine`, computes per-sample gradients efficiently with hooks, and tracks $\varepsilon$ with an RDP or PRV accountant.

```python:opacus_example.py
# pip install opacus  (API of Opacus 1.x)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from opacus import PrivacyEngine

X, Y = torch.randn(2000, 20), torch.randint(0, 2, (2000,))
loader = DataLoader(TensorDataset(X, Y), batch_size=128)
model = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 2))
opt = torch.optim.SGD(model.parameters(), lr=0.1)

engine = PrivacyEngine()
model, opt, loader = engine.make_private_with_epsilon(
    module=model, optimizer=opt, data_loader=loader,
    target_epsilon=8.0, target_delta=1e-5, epochs=5, max_grad_norm=1.0)

for epoch in range(5):
    for xb, yb in loader:
        opt.zero_grad()
        nn.functional.cross_entropy(model(xb), yb).backward()
        opt.step()
    print(f"epoch {epoch}: epsilon so far = {engine.get_epsilon(delta=1e-5):.2f}")
```

:::warning What DP does and does not give you
DP bounds the influence of any one *record*. It does not hide properties of the population (a DP model can still learn that smoking predicts cancer), it does not protect an individual whose data appears many times (group privacy degrades as $k\varepsilon$ for a group of $k$), it protects only if the *implementation* is right (clipping, Poisson sampling, accounting, secure noise generation), and it has real accuracy costs, which fall hardest on rare subgroups and tail examples (Bagdasaryan et al., 2019, "Differential privacy has disparate impact on model accuracy"). It interacts with fairness. Related tools: **federated learning** keeps raw data on-device but gradients still leak without DP or secure aggregation; **machine unlearning** tries to remove a record's influence after training, and has no general efficient exact solution for deep networks.
:::

## Safety, red-teaming and responsible release {#safety}

### Thinking about harm

Safety in this setting covers technical failure modes (the model does something unintended), misuse (a person uses it to cause harm) and structural effects (concentration of power, labour, information ecosystems). A useful frame for any model is to enumerate **who is affected, how they could be harmed, how likely and how severe the harm is, and what mitigations exist** at each stage: data, training, evaluation, release, monitoring. Distinguish *capability* (what the model can do) from *propensity* (what it tends to do) and from *deployment context* (who can access it, with which guardrails).

### Model cards and documentation

Mitchell et al. (FAT* 2019, "Model cards for model reporting") proposed a short standard document that accompanies a model. Gebru et al. (2018/2021, "Datasheets for datasets") did the same for data. A good model card states:

- **Intended use** and out-of-scope uses; users and contexts it was evaluated for.
- **Training data** (sources, collection, licensing, known gaps) and preprocessing.
- **Evaluation**: metrics **disaggregated** by relevant groups and conditions (not just a global average), with confidence intervals, and the evaluation data.
- **Ethical considerations and risks**, known failure modes, and **limitations**.
- **Environment and compute**, version, contact, and update policy.

A card is useful only if it contains numbers a reader could be disappointed by. Treat it as part of the engineering process (written during development), not as post-hoc marketing.

### Red-teaming and evaluation

**Red-teaming** is adversarial testing by people (and increasingly models) whose job is to find failures before users do. It complements benchmarks: benchmarks measure known failure modes on fixed data, red teams search for unknown ones, adaptively. Practices that have emerged:

- Define threat scenarios first (e.g. generation of harmful instructions, privacy leakage, biased outputs, prompt injection through tool outputs, deception), then probe each with diverse testers, including domain experts and people from affected communities.
- Combine **manual** and **automated** red-teaming: Perez et al. (2022) used a language model to generate test cases for another; gradient-based or search-based attacks produce adversarial suffixes (Zou et al. 2023). Automated methods give coverage, humans give creativity.
- Record failures in a *reproducible, versioned* test set, fix, and re-test: a fix that moves the failure elsewhere is common. Measure the **attack success rate** at a fixed budget, and report the attacker's knowledge and effort.
- **Capability evaluations** for dangerous capabilities (cyber-offense, bio-chemical uplift, autonomous replication) are run by specialist teams before deployment, with thresholds that trigger stronger safeguards. This is an active, unsettled area.
- Safeguard layers: training-time alignment (RLHF and related methods, see [Week 10](week-10.html) and [Week 15](week-15.html)), input and output classifiers, rate limits, monitoring, abuse reporting, and the ability to revoke access (feasible for APIs, impossible for open weights).

### Misuse and responsible release

Generative models are **dual use**: the same image model produces art and non-consensual imagery; the same code model writes tools and malware. The release decision has a spectrum: *closed* (internal), *API access* (can monitor, filter, revoke), *gated weights* (research access) and *open weights* (maximal scientific and economic benefit, no revocation). There is no universal answer; factors include the marginal risk relative to what is available already ("uplift"), the severity and reversibility of potential harm, whether defenses benefit from openness (security research often does), and the capability level. Release practices that help: staged release, documentation (model card, data statement), watermarking or provenance metadata for synthetic media (C2PA, statistical watermarks for text, with known limits: they can be removed by paraphrasing), usage licenses (such as OpenRAIL, enforceable only where there is a way to enforce), and an incident-response process.

For *your* projects, a practical checklist: write down the intended use and who could be harmed; evaluate on subgroups and shifted data; measure calibration and set an abstention rule; decide what you log (privacy) and how a user can contest a decision; plan monitoring for drift; and say plainly what the model cannot do.

:::note Governance in brief
Regulation is evolving quickly (for instance the EU AI Act's risk-tiered obligations, sector-specific rules in health and finance, and voluntary frameworks such as the NIST AI Risk Management Framework). You do not need the details to start, but you should expect high-stakes deployments to require documentation, testing for bias and robustness, human oversight and traceability, which is why the tools in this chapter are becoming part of the job.
:::

## Deep learning theory highlights {#theory}

Theory for deep learning is in the odd position that practice runs far ahead of it. What we have is a collection of **partial explanations**, mostly for simplified models (linear, two-layer, infinite-width), plus empirical phenomena that any complete theory must explain. This section picks the results that every practitioner should know. For each, ask: what is proved, under which assumptions, and what does it not explain?

### Generalization of overparameterized models

Classical statistical learning theory bounds the gap between training and test error by the **capacity** of the hypothesis class: VC dimension, Rademacher complexity, or parameter count. A typical bound has the form $R(f)\le\hat R(f)+O\big(\sqrt{\mathrm{capacity}/n}\big)$. A modern network has more parameters than training points, can fit any labelling of the data, and so the bound is vacuous. Yet it generalizes. Something other than raw capacity must govern the outcome: the *algorithm* (SGD), the *data*, the *architecture*.

**The randomization experiments.** Zhang, Bengio, Hardt, Recht and Vinyals (ICLR 2017, "Understanding deep learning requires rethinking generalization", later published in CACM as "Understanding deep learning (still) requires rethinking generalization") trained standard CNNs on CIFAR-10 and ImageNet with the labels **replaced by random labels**. The networks reached zero training error, though more slowly. They also fit images replaced by random noise. Explicit regularizers (weight decay, dropout, data augmentation) neither were necessary for generalization on real labels nor prevented fitting random ones. The lesson is that the hypothesis class (these architectures) contains functions that fit pure noise, so *uniform* capacity-based bounds over the class cannot explain why the one solution SGD finds on real data generalizes. Any explanation must be data- and algorithm-dependent. They also showed a depth-two network with $2n+d$ parameters can fit any labelling of $n$ points in $d$ dimensions (a finite-sample expressivity result).

You can reproduce the effect at miniature scale. I trained a single-hidden-layer tanh network (512 units, plain gradient descent on logistic loss) on 200 points in 20 dimensions, whose true labels depend on two coordinates, with an increasing fraction of the labels randomized, and evaluated on fresh clean data.

```chart
{
  "type": "bar",
  "caption": "Miniature randomization experiment (NumPy, one seed, so noisy): a wide one-hidden-layer network always reaches 100% training accuracy, even with all labels random, while test accuracy on clean data falls toward chance (50%) as more labels are randomized. Training accuracy does not predict generalization; this is the point of Zhang et al. (2017).",
  "height": 320,
  "data": {
    "labels": ["0%", "25%", "50%", "75%", "100%"],
    "datasets": [
      {"label": "train accuracy", "data": [1.0, 1.0, 1.0, 1.0, 1.0], "backgroundColor": "var(--c-blue)"},
      {"label": "test accuracy (clean labels)", "data": [0.923, 0.828, 0.695, 0.719, 0.443], "backgroundColor": "var(--c-amber)"}
    ]
  },
  "options": { "scales": { "x": {"title": {"display": true, "text": "fraction of training labels randomized"}}, "y": {"title": {"display": true, "text": "accuracy"}, "min": 0, "max": 1.05} } }
}
```

**What has been proposed since.** (1) *Norm- and margin-based bounds* (Bartlett, Foster and Telgarsky, 2017, spectrally-normalized margin bounds) replace parameter count with products of layer norms; they are better than parameter counting but still loose or even *increase* with data on some tasks (Nagarajan and Kolter 2019 argued that uniform-convergence bounds may be inherently unable to explain deep-network generalization). (2) *PAC-Bayes bounds* (Dziugaite and Roy, 2017) computed non-vacuous bounds for small networks by optimizing a posterior over weights; the flatness of the found solution matters. (3) *Compression bounds* (Arora et al., 2018): if a trained network can be compressed to far fewer parameters with little change in output, it generalizes as if it had that smaller count. (4) *Algorithmic stability* (Hardt, Recht and Singer, 2016): SGD run for few steps on a smooth loss is stable, so generalizes. (5) *Benign overfitting* (Bartlett, Long, Lugosi and Tsigler, 2020): in linear regression one can interpolate noisy training labels and still have near-optimal test error when the data has many low-variance directions to absorb the noise. All are partial; there is no accepted complete explanation.

### Implicit regularization

When there are many global minima of the training loss, the **optimizer** chooses among them, and its choice is a regularizer nobody wrote down. The cleanest example is overparameterized linear regression.

:::math Gradient descent finds the minimum-norm interpolant
Let $X\in\R^{n\times p}$ with $p>n$ and full row rank, and $\boldsymbol{y}\in\R^n$. Run gradient descent on $\mathcal{L}(\boldsymbol{w})=\tfrac12\norm{X\boldsymbol{w}-\boldsymbol{y}}_2^2$ from $\boldsymbol{w}_0=\boldsymbol{0}$. Each update is $\boldsymbol{w}_{t+1}=\boldsymbol{w}_t-\eta X^\top(X\boldsymbol{w}_t-\boldsymbol{y})$, so by induction $\boldsymbol{w}_t$ always lies in the row space of $X$: $\boldsymbol{w}_t=X^\top\boldsymbol{a}_t$ for some $\boldsymbol{a}_t$. If it converges to a point with $X\boldsymbol{w}_\infty=\boldsymbol{y}$, then $\boldsymbol{w}_\infty=X^\top\boldsymbol{a}$ and $XX^\top\boldsymbol{a}=\boldsymbol{y}$, so $\boldsymbol{a}=(XX^\top)^{-1}\boldsymbol{y}$ and
$$\boldsymbol{w}_\infty=X^\top(XX^\top)^{-1}\boldsymbol{y}=X^{+}\boldsymbol{y},$$
the Moore-Penrose solution, which is the interpolating solution with the **smallest $\ell_2$ norm** (any other interpolant differs by a vector in the null space of $X$, orthogonal to the row space, and adds norm). No explicit penalty was used.
:::

Other results of this kind: on **separable data with logistic or cross-entropy loss**, gradient descent drives $\norm{\boldsymbol{w}}\to\infty$ while the *direction* converges to the **maximum-margin** SVM solution (Soudry, Hoffer, Nacson, Gunasekar and Srebro, 2018), which explains why test error can keep improving after training error reaches zero. In **matrix factorization** with small initialization, gradient descent is biased toward low nuclear norm or low rank (Gunasekar et al., 2017; Arora et al., 2019: deeper factorizations are biased more strongly toward low rank). The **initialization scale** changes the bias, from a "rich" feature-learning regime to a "kernel" regime (Woodworth et al., 2020). **SGD noise** is thought to prefer flat minima (Keskar et al. 2017; but flatness is not reparameterization-invariant, Dinh et al. 2017), and the *edge of stability* phenomenon (Cohen et al., 2021: full-batch GD's sharpness rises until $2/\eta$ and hovers) shows that large-step training dynamics self-regulate in ways classical analyses do not capture. Whether implicit regularization explains the generalization of real deep networks is still unproven; it explains toy models well.

### Neural tangent kernel and lazy training

Consider a network $f(\boldsymbol{x};\boldsymbol{\theta})$ with parameters $\boldsymbol{\theta}\in\R^p$. Gradient flow on the squared loss over training set $\{(\boldsymbol{x}_i,y_i)\}$ changes the function by
$$
\frac{d}{dt}f(\boldsymbol{x};\boldsymbol{\theta}_t)=\nabla_{\boldsymbol{\theta}}f(\boldsymbol{x})^\top\dot{\boldsymbol{\theta}}_t=-\sum_{i=1}^n\underbrace{\nabla_{\boldsymbol{\theta}}f(\boldsymbol{x})^\top\nabla_{\boldsymbol{\theta}}f(\boldsymbol{x}_i)}_{\Theta_t(\boldsymbol{x},\boldsymbol{x}_i)}\,\big(f(\boldsymbol{x}_i)-y_i\big),
$$
so the function evolves according to the **neural tangent kernel** $\Theta_t(\boldsymbol{x},\boldsymbol{x}')=\langle\nabla_{\boldsymbol{\theta}}f(\boldsymbol{x};\boldsymbol{\theta}_t),\nabla_{\boldsymbol{\theta}}f(\boldsymbol{x}';\boldsymbol{\theta}_t)\rangle$. In general this kernel changes during training, which is why training is hard to analyze. **Jacot, Gabriel and Hongler (NeurIPS 2018)** proved that for fully connected networks in the infinite-width limit (with the NTK parameterization, $1/\sqrt{\text{width}}$ output scaling), the kernel at initialization converges to a deterministic limit $\Theta_\infty$ and **stays constant** during training. Then $f$ evolves like kernel regression:

$$
f_t(\boldsymbol{x})=f_0(\boldsymbol{x})-\Theta_\infty(\boldsymbol{x},X)\,\Theta_\infty(X,X)^{-1}\big(I-e^{-\eta\,\Theta_\infty(X,X)\,t}\big)\big(f_0(X)-\boldsymbol{y}\big),
$$

a closed form (with $f_0(X)$ the vector of initial outputs on the training set). Consequences: (i) convergence to zero training loss at a linear rate whose speed is the smallest eigenvalue of the kernel matrix $\Theta_\infty(X,X)$ (this is the idea behind global-convergence proofs for overparameterized networks, Du et al. 2019; Allen-Zhu, Li and Song 2019), and (ii) the trained network is equivalent to a kernel machine with a *fixed* kernel.

**Lazy-training sketch.** Why does the kernel freeze? Chizat, Oyallon and Bach (2019) gave a general explanation: if the output is scaled by a large factor $\alpha$, $f_\alpha=\alpha\,(f(\boldsymbol{\theta})-f(\boldsymbol{\theta}_0))$, the loss gradient w.r.t. $\boldsymbol{\theta}$ is multiplied by $\alpha$, so achieving a fixed change of the *output* needs only a parameter change of order $1/\alpha$. The parameters stay close to $\boldsymbol{\theta}_0$, and the network is well approximated by its first-order Taylor expansion $f(\boldsymbol{x};\boldsymbol{\theta})\approx f(\boldsymbol{x};\boldsymbol{\theta}_0)+\nabla_{\boldsymbol{\theta}}f(\boldsymbol{x};\boldsymbol{\theta}_0)^\top(\boldsymbol{\theta}-\boldsymbol{\theta}_0)$, a model that is *linear in the parameters* with fixed features $\nabla_{\boldsymbol{\theta}}f(\cdot;\boldsymbol{\theta}_0)$. That is exactly kernel regression with the tangent kernel. The features never move, so the network never learns representations: it is "lazy".

```python:ntk_empirical.py
import torch
import torch.nn as nn
from torch.func import functional_call, vmap, jacrev

torch.manual_seed(0)
net = nn.Sequential(nn.Linear(2, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1))
params = {k: v.detach() for k, v in net.named_parameters()}


def f_single(p, x):
    return functional_call(net, p, (x.unsqueeze(0),)).squeeze()      # scalar output


def empirical_ntk(x1, x2):
    """Theta(x1, x2)_{ab} = <grad_theta f(x1_a), grad_theta f(x2_b)>, shape (n1, n2)."""
    j1 = vmap(jacrev(f_single), (None, 0))(params, x1)               # dict name -> (n1, *param_shape)
    j2 = vmap(jacrev(f_single), (None, 0))(params, x2)
    k = 0.0
    for name in params:
        a, b = j1[name].flatten(1), j2[name].flatten(1)
        k = k + a @ b.T
    return k


x = torch.randn(8, 2)
K = empirical_ntk(x, x)
print(K.shape, torch.allclose(K, K.T, atol=1e-4))
print("eigenvalues:", torch.linalg.eigvalsh(K))     # PSD; smallest eigenvalue controls the convergence rate
```

What the NTK does *not* explain: the kernel regime does not match the performance of practical networks, which *do* change their features (finite-width networks beat their NTK counterpart on CIFAR-10 by large margins, Arora et al. 2019, "On exact computation with an infinitely wide neural net"), so the NTK is a theory of one regime, not of deep learning. The competing view is the **mean-field** and **feature-learning** regime (Mei, Montanari and Nguyen 2018; Yang and Hu 2021, maximal-update parameterization, $\mu$P), where hidden representations evolve and hyperparameters (such as learning rate) transfer across widths, an idea with direct practical value for scaling models.

### The lottery ticket hypothesis

Pruning (see [Week 16](week-16.html#pruning)) can remove 90% or more of a trained network's weights with little loss, yet training the pruned architecture from a fresh random initialization usually works poorly. Frankle and Carbin (ICLR 2019, "The lottery ticket hypothesis") asked why, and proposed: *a randomly initialized dense network contains a sparse subnetwork (a "winning ticket") that, when trained in isolation **from its original initialization**, matches the full network's test accuracy in a comparable number of iterations.*

The procedure to find tickets is **iterative magnitude pruning (IMP)** with rewinding: (1) initialize $\boldsymbol{\theta}_0$, (2) train to get $\boldsymbol{\theta}_T$, (3) remove the smallest-magnitude $p\%$ of weights to define a binary mask $\boldsymbol{m}$, (4) *reset* the surviving weights to their values in $\boldsymbol{\theta}_0$ and train the masked network $\boldsymbol{m}\odot\boldsymbol{\theta}_0$, (5) repeat. Tickets for small networks (MNIST fully connected, small CIFAR convnets) were found at 1-10% of the original size. For deeper networks (ResNet-50, VGG on ImageNet), resetting to the exact initialization fails, but **late rewinding** works: reset to the weights from a *very early training iteration* $k$ (Frankle, Dziugaite, Roy and Carbin, 2020, linear mode connectivity and the lottery ticket hypothesis): the network has by then become stable to SGD noise, meaning that two runs from that point with different data order end up in the same linearly connected basin.

```python:imp.py
import copy
import torch
import torch.nn as nn


def make_masks(model):
    return {n: torch.ones_like(p) for n, p in model.named_parameters() if p.dim() > 1}   # prune weights, not biases


def apply_masks(model, masks):
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in masks:
                p.mul_(masks[n])


def prune_global(model, masks, frac):
    """Remove the `frac` smallest-magnitude surviving weights (global threshold across layers)."""
    alive = torch.cat([p.detach().abs()[masks[n].bool()].flatten() for n, p in model.named_parameters() if n in masks])
    k = int(frac * alive.numel())
    if k == 0:
        return masks
    thresh = alive.kthvalue(k).values
    return {n: (masks[n].bool() & (p.detach().abs() > thresh)).float() for n, p in model.named_parameters() if n in masks}


def lottery_ticket(model_fn, train_fn, rounds=5, frac=0.2, rewind_state=None):
    """Iterative magnitude pruning with rewinding. train_fn(model, masks) trains the model in place
    (and must re-apply the masks after every optimizer step so pruned weights stay zero)."""
    model = model_fn()
    init_state = copy.deepcopy(model.state_dict()) if rewind_state is None else rewind_state
    masks = make_masks(model)
    for r in range(rounds):
        train_fn(model, masks)
        masks = prune_global(model, masks, frac)
        model.load_state_dict(init_state)          # rewind surviving weights to initialization (or an early iterate)
        apply_masks(model, masks)
        sparsity = 1 - sum(m.sum() for m in masks.values()) / sum(m.numel() for m in masks.values())
        print(f"round {r}: sparsity {sparsity:.1%}")
    return model, masks
```

The hypothesis is a statement about *trainability of sparse networks*, not about better final accuracy. It connects pruning, initialization and optimization, and led to "pruning at initialization" methods (SNIP, GraSP, SynFlow), which on careful inspection often do no better than layer-wise random pruning at the same sparsity (Frankle et al. 2021). A practical takeaway is that sparse training from scratch is feasible, but the theory of *which* connections matter and *why* is still open.

### Grokking

Power et al. (2022, "Grokking: generalization beyond overfitting on small algorithmic datasets") trained small transformers on modular arithmetic tables (e.g. $a+b \bmod 97$) with a fraction of the table as training data. Training accuracy reached 100% quickly, while validation accuracy stayed near chance for a long time, then rose suddenly to near 100%, thousands of steps *after* the training loss was already almost zero. This delayed generalization is **grokking**.

Subsequent work gave a partial, mechanistic account. Nanda et al. (2023, "Progress measures for grokking via mechanistic interpretability") reverse-engineered a one-layer transformer trained on modular addition and found that it learns to represent the inputs as points on a circle using discrete Fourier features, and implements addition via trigonometric identities ($\cos(a+b)=\cos a\cos b-\sin a\sin b$). Training has three phases: **memorization** (fit the training set), **circuit formation** (the generalizing algorithm gradually forms while memorization still dominates the loss), and **cleanup** (weight decay removes the memorizing components, and test accuracy jumps). The key insight is that a smooth, hidden process underlies the apparent sudden transition, and that **weight decay** (which favours low-norm, hence more efficient, circuits) drives it. Liu, Michaud and Tegmark (2022) argued grokking requires an initialization and weight norm in a "Goldilocks zone", and Varma et al. (2023) explained grokking in terms of the relative efficiency of memorizing and generalizing circuits. The practical lesson: *a flat validation curve with zero training loss is not proof that generalization is impossible*, and conversely, reaching zero training loss is the beginning of a representation-learning process, not its end.

### Double descent

The classical **U-shaped bias-variance curve** says test error falls and then rises as model complexity grows. Belkin, Hsu, Ma and Mandal (PNAS 2019, "Reconciling modern machine-learning practice and the classical bias-variance trade-off") observed that if you keep increasing capacity beyond the **interpolation threshold**, where the model first can fit the training set exactly, test error falls *again*. Nakkiran et al. (ICLR 2020, "Deep double descent") showed the effect in CNNs, ResNets and transformers, as a function of model size, training epochs, and data size ("epoch-wise" and "sample-wise" double descent: more data can hurt near the threshold).

The cleanest way to see it is minimum-norm least squares with random features. I fit $n=40$ noisy training points with the first $p$ of 400 Gaussian features (noise variance 0.25), using the minimum-norm solution `pinv(A) @ y` and measuring mean-squared test error averaged over 60 random datasets.

```chart
{
  "type": "line",
  "caption": "Double descent in minimum-norm least squares (NumPy, n = 40 training points, p features used out of 400, label noise std 0.5, averaged over 60 draws). Test error explodes at the interpolation threshold p = n = 40 and decreases again as p grows. Log scale on the y axis.",
  "height": 340,
  "data": {
    "labels": [2, 5, 10, 15, 20, 25, 30, 35, 38, 40, 42, 45, 50, 60, 80, 120, 200, 400],
    "datasets": [ {"label": "test MSE", "data": [4.19, 4.54, 5.37, 6.30, 8.30, 11.7, 18.0, 33.0, 171.7, 110980, 153.5, 44.5, 23.4, 11.8, 7.24, 5.12, 4.16, 3.64], "borderColor": "var(--c-red)", "tension": 0.15, "pointRadius": 3} ]
  },
  "options": { "scales": { "x": {"title": {"display": true, "text": "number of features p (model size)"}}, "y": {"type": "logarithmic", "title": {"display": true, "text": "test mean squared error"}} } }
}
```

:::math Why the peak? The role of the smallest singular value
For $p\ge n$ the min-norm interpolant is $\hat{\boldsymbol{w}}=A^{+}\boldsymbol{y}$ with $A\in\R^{n\times p}$ the training feature matrix. Noise $\boldsymbol{\varepsilon}$ in $\boldsymbol{y}$ is mapped through $A^+$, so it contributes $A^{+}\boldsymbol{\varepsilon}$ to $\hat{\boldsymbol{w}}$, whose squared norm is at most $\norm{\boldsymbol{\varepsilon}}^2/\sigma_{\min}(A)^2$. At $p=n$, a square random matrix has a smallest singular value that is close to zero, so the noise is amplified enormously: the variance diverges. For $p\gg n$, $\sigma_{\min}(A)\approx\sqrt p-\sqrt n$ grows, and the minimum-norm solution spreads the fit over many weak directions, so the variance term shrinks again. Test error is a sum of a bias term (falls with $p$) and this variance term (peaks at $p=n$ then decays).
:::

Three cautions. Double descent is about *interpolation with noise*, so it is most visible with label noise or weak regularization. Proper regularization (optimal ridge, early stopping) smooths the peak away (Nakkiran et al., 2021, "Optimal regularization can mitigate double descent"). And the second descent shows that bigger models *can* generalize better, but it does not say that they always do; the right claim is that classical capacity measures fail, not that complexity is free.

:::note Putting the pieces together
Zhang et al. say capacity measures cannot explain generalization; implicit regularization says SGD plus parameterization picks a particular solution; the NTK says in one limit the picture is kernel regression; the lottery ticket says trainability concentrates in sparse subnetworks; grokking says representation-learning can continue after training loss vanishes; double descent says the classical U-curve is only part of a picture that includes interpolation. A unified theory is missing, and good research problems for MS students are available at the boundary of each.
:::

## Open problems and research directions {#open-problems}

Predictions about the future are unreliable, so this section lists *open questions*, each with enough context to find the literature. Treat it as a map of where a project could contribute.

- **Reasoning.** Language models solve many reasoning benchmarks, particularly with chain-of-thought prompting (Wei et al., 2022) and training on verifiable rewards, and still fail in ways that look unlike human errors: sensitivity to irrelevant changes, poor out-of-distribution length generalization, inconsistent answers. Open questions: how much of the reasoning in a chain of thought causally drives the answer (faithfulness of explanations, Turpin et al. 2023), how to train and verify long multi-step reasoning without a reliable ground truth, and whether current architectures can represent the needed algorithms (transformers have known expressivity limits at fixed depth, Merrill and Sabharwal 2023).
- **Agents.** Models that act in environments (browsing, coding, operating software) over many steps compound errors, are exposed to prompt injection from the content they read, and are hard to evaluate (long horizons, non-deterministic outcomes). Open: reliability over long horizons, safe tool use and sandboxing, measuring autonomy-related risk, credit assignment over very long trajectories, and cost.
- **Data efficiency.** Humans learn language from orders of magnitude fewer tokens than large language models. Scaling laws (Kaplan et al. 2020; Hoffmann et al. 2022) describe how loss falls with data and compute, and the supply of high-quality text is finite. Directions: better data curation and curricula, synthetic data and its risks (model collapse when training on one's own outputs, Shumailov et al. 2024), self-supervised objectives that extract more per example, few-shot and in-context adaptation, active learning, and inductive biases.
- **Continual learning.** Networks forget earlier tasks when trained on new ones (**catastrophic forgetting**; McCloskey and Cohen, 1989). Approaches are regularization (EWC, Kirkpatrick et al. 2017), replay, architectural expansion, and parameter-efficient modules. Open: a method that learns indefinitely from a stream, without storing all data, scaling to foundation models, and with a benchmark that reflects deployment (where distribution shift is gradual and unlabeled).
- **World models.** A model that predicts how an environment evolves given actions can support planning, imagination and sample-efficient control (Ha and Schmidhuber 2018; the Dreamer family; JEPA-style latent prediction proposed by LeCun, 2022; video generation models used as simulators). Open: whether learned predictors capture causal and physical structure rather than correlations, how to evaluate "understanding", long-horizon consistency, and how to plan in the latent space reliably.
- **Cross-cutting.** Reliable *evaluation* (contamination, benchmark saturation, measuring capabilities and risks), **scalable oversight** (supervising systems that may exceed our ability to check), **alignment** and its theory, interpretability that scales to frontier models, efficiency and the energy cost of training (see [Week 16](week-16.html)), and the science of *why* all of the above works (the previous section).

:::tip Choosing a research problem
A good project has a measurable claim, a baseline that is not a straw man, an ablation that isolates the cause, and an honest account of failure cases. Many of the best contributions in this field are **careful negative results and evaluation papers**: the sanity checks for saliency maps, the adaptive attacks on defenses, the retrospective on calibration, and the randomization test are all of that kind.
:::

## Lab: attack, defend, explain and calibrate {#lab}

In this lab you will take one small CNN on MNIST through the whole chapter: train a standard model, break it with FGSM and PGD, harden it with adversarial training, explain predictions with Grad-CAM and saliency, and measure and repair its calibration. Everything runs in a few minutes on a Colab GPU (CPU is fine but slower). Run the cells in order; each part reuses the previous variables. We use MNIST so you can see everything; the same code works for CIFAR-10 with a bigger network and $\epsilon=8/255$.

### Part 0: data and model

```python:lab_setup.py
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision, torchvision.transforms as T
from torch.utils.data import DataLoader, random_split

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

# No normalization transform: keep pixels in [0, 1] so that epsilon is in pixel units
tfm = T.ToTensor()
full = torchvision.datasets.MNIST("data", train=True, download=True, transform=tfm)
test = torchvision.datasets.MNIST("data", train=False, download=True, transform=tfm)
train, val = random_split(full, [50000, 10000], generator=torch.Generator().manual_seed(0))   # val is for calibration
train_dl = DataLoader(train, batch_size=128, shuffle=True)
val_dl = DataLoader(val, batch_size=1000)
test_dl = DataLoader(test, batch_size=1000)


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):                                   # x: (B, 1, 28, 28)
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)          # (B, 32, 14, 14)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)          # (B, 64, 7, 7)  <- last conv features (for Grad-CAM)
        x = F.relu(self.fc1(x.flatten(1)))
        return self.fc2(x)                                  # logits (B, 10)


def train_standard(model, epochs=3, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(epochs):
        model.train()
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
        print(f"epoch {ep}: clean test acc {accuracy(model, test_dl):.4f}")


@torch.no_grad()
def accuracy(model, dl):
    model.eval()
    c = n = 0
    for x, y in dl:
        x, y = x.to(device), y.to(device)
        c += (model(x).argmax(1) == y).sum().item(); n += len(y)
    return c / n


std_model = SmallCNN().to(device)
train_standard(std_model)
```

Expected: clean test accuracy around 0.99 after three epochs.

### Part 1: attacks

Reuse `fgsm` and `pgd` from the earlier code block (save them in `attacks.py`, or paste them into a cell), then sweep $\epsilon$.

```python:lab_attacks.py
from attacks import fgsm, pgd          # the functions defined in the "Adversarial examples" section

def robust_acc(model, dl, attack, **kw):
    model.eval()
    c = n = 0
    for x, y in dl:
        x, y = x.to(device), y.to(device)
        xa = attack(model, x, y, **kw)                       # needs gradients, computed inside
        with torch.no_grad():
            c += (model(xa).argmax(1) == y).sum().item(); n += len(y)
    return c / n

eps_list = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]
print("eps   FGSM    PGD-20")
for eps in eps_list:
    a_f = robust_acc(std_model, test_dl, fgsm, eps=eps)
    a_p = robust_acc(std_model, test_dl, pgd, eps=eps, alpha=eps / 8 if eps > 0 else 0.0, steps=20)
    print(f"{eps:.2f}  {a_f:.4f}  {a_p:.4f}")
```

You should see FGSM accuracy dropping steeply by $\epsilon=0.1$ to $0.2$ and PGD driving the standard model to near 0% by $\epsilon\approx0.2$ (PGD is always at least as strong as FGSM, up to the randomness of the start). A perturbation of $0.3$ on MNIST is large and visible; $\epsilon\le0.1$ is nearly invisible.

```python:lab_visualize.py
import matplotlib.pyplot as plt

x, y = next(iter(test_dl)); x, y = x[:8].to(device), y[:8].to(device)
xa = pgd(std_model, x, y, eps=0.15, alpha=0.02, steps=20)
with torch.no_grad():
    pa, pb = std_model(x).softmax(1), std_model(xa).softmax(1)
fig, ax = plt.subplots(3, 8, figsize=(14, 5.5))
for i in range(8):
    ax[0, i].imshow(x[i, 0].cpu(), cmap="gray"); ax[0, i].set_title(f"{pa[i].argmax().item()} ({pa[i].max():.2f})")
    ax[1, i].imshow((xa - x)[i, 0].cpu(), cmap="RdBu", vmin=-0.15, vmax=0.15); ax[1, i].set_title("perturbation")
    ax[2, i].imshow(xa[i, 0].cpu(), cmap="gray"); ax[2, i].set_title(f"{pb[i].argmax().item()} ({pb[i].max():.2f})")
    for r in range(3): ax[r, i].axis("off")
plt.tight_layout(); plt.show()
```

### Part 2: adversarial training and evaluation

```python:lab_advtrain.py
from adv_train import adversarial_epoch

adv_model = SmallCNN().to(device)
opt = torch.optim.Adam(adv_model.parameters(), lr=1e-3)
for ep in range(5):
    loss, acc_adv = adversarial_epoch(adv_model, train_dl, opt, eps=0.1, alpha=0.025, steps=7, device=device)
    print(f"epoch {ep}: adv-train loss {loss:.3f}  adv-train acc {acc_adv:.3f}  "
          f"clean test {accuracy(adv_model, test_dl):.4f}  PGD-20 test {robust_acc(adv_model, test_dl, pgd, eps=0.1, alpha=0.0125, steps=20):.4f}")

# Compare the two models under identical attacks (eps = 0.1)
for name, m in [("standard", std_model), ("adversarially trained", adv_model)]:
    print(name,
          "clean", round(accuracy(m, test_dl), 4),
          "FGSM", round(robust_acc(m, test_dl, fgsm, eps=0.1), 4),
          "PGD-20", round(robust_acc(m, test_dl, pgd, eps=0.1, alpha=0.0125, steps=20), 4))
```

Expect the adversarially trained model to keep roughly 90% or more of its accuracy under $\epsilon=0.1$ PGD (the exact numbers vary with seed and training length) at the price of a little clean accuracy. **Sanity checks you should run yourself:** (a) evaluate the adversarially trained model at $\epsilon=0.3$ and see that robustness is specific to the training radius; (b) check that PGD with more steps or restarts does not reduce accuracy much more (if it does, the evaluation was too weak); (c) try an $\ell_2$ attack with `pgd_l2` to see that $\ell_\infty$ training does not guarantee $\ell_2$ robustness.

### Part 3: explanations with Grad-CAM and saliency

```python:lab_explain.py
from gradcam import GradCAM               # from the Grad-CAM section; here layer = the second conv layer (7x7 after pooling)
import matplotlib.pyplot as plt

# The conv layer outputs 14x14 before the final max-pool; hook the module whose output is the 7x7 feature map.
# Easiest: wrap the pooled features as a module.
class Pool(nn.Module):
    def forward(self, x): return F.max_pool2d(x, 2)

class CNNWithPool(SmallCNN):                      # same weights, but the pooled map is an explicit module
    def __init__(self):
        super().__init__(); self.pool = Pool()
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        return self.fc2(F.relu(self.fc1(x.flatten(1))))

def with_pool(model):
    m = CNNWithPool().to(device); m.load_state_dict(model.state_dict()); return m.eval()

m_std, m_adv = with_pool(std_model), with_pool(adv_model)
x, y = next(iter(test_dl)); x, y = x[:6].to(device), y[:6].to(device)

def saliency(model, x, cls):
    x = x.clone().requires_grad_(True)
    score = model(x).gather(1, cls.view(-1, 1)).sum()
    g, = torch.autograd.grad(score, x)
    return g.abs().squeeze(1)

fig, ax = plt.subplots(4, 6, figsize=(12, 8))
for name_i, (name, m) in enumerate([("standard", m_std), ("adv-trained", m_adv)]):
    cam_fn = GradCAM(m, m.pool)       # note: m.pool is called twice; the hook fires twice and keeps the LAST output (7x7)
    cam, cls = cam_fn(x)
    sal = saliency(m, x, cls)
    for i in range(6):
        ax[2 * name_i, i].imshow(sal[i].detach().cpu(), cmap="magma"); ax[2 * name_i, i].set_title(f"{name} saliency")
        ax[2 * name_i + 1, i].imshow(x[i, 0].cpu(), cmap="gray"); ax[2 * name_i + 1, i].imshow(cam[i].cpu(), cmap="jet", alpha=0.45)
        ax[2 * name_i + 1, i].set_title(f"{name} Grad-CAM")
        ax[2 * name_i, i].axis("off"); ax[2 * name_i + 1, i].axis("off")
plt.tight_layout(); plt.show()

# Randomization sanity check: Grad-CAM from a model with re-initialized final layer should look different
m_rand = with_pool(std_model); nn.init.normal_(m_rand.fc2.weight, std=0.1)
cam_r, _ = GradCAM(m_rand, m_rand.pool)(x, cls)
print("mean |difference| after randomizing the classifier layer:", (cam_r.cpu() - cam.cpu()).abs().mean().item())
```

What to look for: the saliency map of the adversarially trained model is typically much cleaner and more digit-shaped than the standard model's noisy map (Tsipras et al. 2019 and Kaur et al. 2019 observed that robust models have more perceptually aligned gradients), which is a nice by-product, though it does not mean the standard model is "wrong". Grad-CAM at $7\times7$ resolution is blobby on MNIST because the digits are small relative to the grid, a useful reminder that its resolution is limited.

### Part 4: measure and repair calibration

```python:lab_calibration.py
from calibration import reliability_bins, fit_temperature

@torch.no_grad()
def collect_logits(model, dl):
    model.eval(); L, Y = [], []
    for x, y in dl:
        L.append(model(x.to(device)).cpu()); Y.append(y)
    return torch.cat(L), torch.cat(Y)

for name, m in [("standard", std_model), ("adv-trained", adv_model)]:
    Lv, Yv = collect_logits(m, val_dl); Lt, Yt = collect_logits(m, test_dl)
    ece_before = reliability_bins(Lt.softmax(1), Yt)[1]
    T = fit_temperature(Lv, Yv)                       # fit on validation, evaluate on test
    ece_after = reliability_bins((Lt / T).softmax(1), Yt)[1]
    nll_b = F.cross_entropy(Lt, Yt).item(); nll_a = F.cross_entropy(Lt / T, Yt).item()
    print(f"{name:12s}  T = {T:.2f}  ECE {ece_before:.4f} -> {ece_after:.4f}  NLL {nll_b:.4f} -> {nll_a:.4f}")

# Confidence of the standard model on an OOD input (Fashion-MNIST) vs MNIST: MSP and energy
fm = torchvision.datasets.FashionMNIST("data", train=False, download=True, transform=tfm)
Lo, _ = collect_logits(std_model, DataLoader(fm, batch_size=1000))
Li, _ = collect_logits(std_model, test_dl)
from ood_scores import msp_score, energy_score, auroc
print("AUROC MSP   :", auroc(msp_score(Li), msp_score(Lo)))
print("AUROC energy:", auroc(energy_score(Li), energy_score(Lo)))
```

Deliverables for the lab report: (1) the table of clean, FGSM and PGD accuracy for both models; (2) one figure of clean, perturbed and perturbation images with predictions; (3) the saliency and Grad-CAM figure with a paragraph interpreting differences, including the randomization check; (4) reliability diagrams and ECE/NLL before and after temperature scaling for both models, plus one sentence on whether adversarial training changes calibration; (5) the OOD AUROC for MSP versus energy.

## Exercises {#exercises}

:::exercise Exercise 1: FGSM for a linear model (derivation)
For a logistic regression $p(y=1\mid\boldsymbol{x})=\sigma(\boldsymbol{w}^\top\boldsymbol{x}+b)$ with label $y\in\{0,1\}$ and cross-entropy loss, (a) compute $\nabla_{\boldsymbol{x}}\mathcal{L}$, (b) give the FGSM adversarial example, and (c) show that the logit moves by exactly $\epsilon\norm{\boldsymbol{w}}_1$ in the direction that hurts, when $\boldsymbol{x}$ is correctly classified.
:::details Solution
(a) $\mathcal{L}=-y\log p-(1-y)\log(1-p)$ with $p=\sigma(z)$, $z=\boldsymbol{w}^\top\boldsymbol{x}+b$. Then $\partial\mathcal{L}/\partial z=p-y$ and $\nabla_{\boldsymbol{x}}\mathcal{L}=(p-y)\boldsymbol{w}$.
(b) $\boldsymbol{x}_{\text{adv}}=\boldsymbol{x}+\epsilon\,\mathrm{sign}\big((p-y)\boldsymbol{w}\big)$. If $y=1$ then $p-y<0$, so the step is $-\epsilon\,\mathrm{sign}(\boldsymbol{w})$; if $y=0$, $p-y>0$ and it is $+\epsilon\,\mathrm{sign}(\boldsymbol{w})$.
(c) For $y=1$: $z_{\text{adv}}=z-\epsilon\,\boldsymbol{w}^\top\mathrm{sign}(\boldsymbol{w})=z-\epsilon\sum_i\abs{w_i}=z-\epsilon\norm{\boldsymbol{w}}_1$. The classifier flips when $z<\epsilon\norm{\boldsymbol{w}}_1$, i.e. for $\epsilon>z/\norm{\boldsymbol{w}}_1$: the margin in units of the $\ell_1$-norm of the weights decides robustness. Note: for the *linear* model FGSM is exactly optimal for the $\ell_\infty$ ball (the loss is monotone in the logit).
:::
:::

:::exercise Exercise 2: Dimension dependence (conceptual and coding)
In the chart earlier, the same per-coordinate budget $\epsilon$ is more damaging in higher dimension. (a) Show that with $\boldsymbol{w}$ having all entries of magnitude $m$, the logit change under an $\ell_\infty$ attack is $\epsilon m d$ but under random sign noise it is about $\epsilon m\sqrt d$ in standard deviation. (b) Write NumPy code to verify numerically for $d=10,100,1000$.
:::details Solution
(a) The attack: $\boldsymbol{w}^\top\boldsymbol{\delta}=\epsilon\sum_i m=\epsilon m d$. Random: $\boldsymbol{\delta}=\epsilon\boldsymbol{s}$ with independent signs $s_i=\pm1$, so $\boldsymbol{w}^\top\boldsymbol{\delta}=\epsilon m\sum_is'_i$ with mean 0 and variance $\epsilon^2m^2d$, so the standard deviation is $\epsilon m\sqrt d$. The ratio of attack to noise effect is $\sqrt d$.
(b)
```python
import numpy as np
rng = np.random.default_rng(0)
for d in [10, 100, 1000]:
    w = np.full(d, 0.5); eps = 0.01
    attack = w @ (eps * np.sign(w))
    noise = np.std([w @ (eps * rng.choice([-1, 1], d)) for _ in range(5000)])
    print(d, attack, noise, attack / noise)    # ratio is close to sqrt(d)
```
:::
:::

:::exercise Exercise 3: PGD versus FGSM (what would happen if...)
You evaluate a defended model and find FGSM accuracy 62% and PGD-20 accuracy 41% at the same $\epsilon$. Then you find that a *black-box transfer attack* from an undefended surrogate gives 30% accuracy on it. What do you conclude, and what do you do next?
:::details Solution
FGSM above PGD is normal (PGD is stronger). The red flag is that a **transfer attack beats the white-box attack** (30% < 41%): a gradient-based white-box attack should be at least as strong as one crafted on a different model. That points to gradient masking or an under-powered attack (too few steps, a badly chosen step size, a saturating loss, non-differentiable preprocessing). Next: use more steps and restarts, switch to a margin loss instead of cross-entropy, run AutoAttack, use BPDA/EOT if the defense has non-differentiable or randomized parts, and check that accuracy goes to 0% at very large $\epsilon$. Until then, report the strongest number (30% or lower), not the PGD number.
:::
:::

:::exercise Exercise 4: Temperature scaling preserves accuracy (derivation and coding)
(a) Prove that dividing logits by $T>0$ never changes the argmax. (b) Show that the NLL of $\softmax(\boldsymbol{z}/T)$ on a dataset is a convex function of $\beta=1/T$. (c) Write a PyTorch snippet fitting $T$ by L-BFGS on validation logits.
:::details Solution
(a) $z_j/T>z_k/T\iff z_j>z_k$ for $T>0$, so the ordering is unchanged.
(b) $\mathrm{NLL}(\beta)=\frac1n\sum_i\big[\log\sum_je^{\beta z_{ij}}-\beta z_{iy_i}\big]$. The log-sum-exp of a linear function of $\beta$ is convex, and $-\beta z_{iy_i}$ is linear, so the sum is convex in $\beta$.
(c)
```python
import torch, torch.nn.functional as F
def fit_T_lbfgs(logits, labels):
    beta = torch.ones(1, requires_grad=True)                    # beta = 1/T
    opt = torch.optim.LBFGS([beta], lr=0.1, max_iter=100)
    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits * beta, labels)
        loss.backward()
        return loss
    opt.step(closure)
    return (1 / beta).item()
```
Because the loss is convex in $\beta$ the optimizer finds the global optimum; optimizing $1/T$ (or $\log T$) is better conditioned than optimizing $T$ directly.
:::
:::

:::exercise Exercise 5: Conformal coverage by hand (coding)
With a calibration set of $n=19$ scores and $\alpha=0.1$, which order statistic is used as $\hat q$? Then simulate 20,000 trials to confirm the coverage lies between $0.9$ and $0.9+1/(n+1)=0.95$.
:::details Solution
$k=\lceil(n+1)(1-\alpha)\rceil=\lceil20\times0.9\rceil=18$, so $\hat q$ is the 18th smallest of the 19 scores. The guarantee gives coverage between $18/20=0.90$ and $0.90$ here (since $k/(n+1)=0.9$ exactly), so coverage is exactly $0.9$ for continuous scores.
```python
import numpy as np
rng = np.random.default_rng(0)
n, alpha, trials = 19, 0.1, 20000
k = int(np.ceil((n + 1) * (1 - alpha)))
cov = np.mean([rng.random() <= np.sort(rng.random(n))[k - 1] for _ in range(trials)])
print(k, cov)       # ~0.90
```
:::
:::

:::exercise Exercise 6: Integrated Gradients completeness (challenging)
(a) Prove completeness from the fundamental theorem of calculus. (b) For $F(\boldsymbol{x})=\relu(x_1)\,x_2$ with baseline $\boldsymbol{0}$ and input $(2,3)$, compute IG analytically and verify the sum. (c) Compute the plain gradient at $(2,3)$ and at the point $(-1,3)$, and explain what IG does better.
:::details Solution
(a) See the derivation in the text: $F(\boldsymbol{x})-F(\boldsymbol{x}')=\int_0^1\frac{d}{dt}F(\boldsymbol{x}'+t(\boldsymbol{x}-\boldsymbol{x}'))dt=\sum_i(x_i-x'_i)\int_0^1\partial_iF\,dt$.
(b) Path $\boldsymbol{x}(t)=(2t,3t)$. $F=\relu(2t)\cdot3t=6t^2$. $\partial_1F=\ind[x_1>0]\,x_2=3t$, $\partial_2F=\relu(x_1)=2t$. $\mathrm{IG}_1=2\int_0^13t\,dt=2\cdot1.5=3$; $\mathrm{IG}_2=3\int_0^12t\,dt=3$. Sum $=6=F(2,3)-F(0,0)=6$. ✓.
(c) At $(2,3)$ the gradient is $(3,2)$, which multiplied by the input gives $(6,6)$ for gradient-times-input, summing to 12, double-counting the interaction (the output is 6). At $(-1,3)$: $F=0$ and the gradient is $(0,0)$ (ReLU inactive), so plain gradient says no feature matters, whereas IG integrates over the path from the baseline: here $F\equiv0$ along the path so IG is also 0, correctly (the output does not change). IG's gain appears in saturating cases such as $F=\min(x,1)$ at $x=5$: gradient 0, IG$=1$.
:::
:::

:::exercise Exercise 7: Fairness criteria (derivation and numbers)
Group A has base rate 0.5, group B has base rate 0.2. A classifier has FNR $=0.2$ and PPV $=0.8$ in both groups. (a) Compute the FPR in each group using the identity from the text. (b) Does equalized odds hold? (c) Which criterion would you equalize for a screening tool whose positive prediction triggers an invasive follow-up? Argue.
:::details Solution
(a) $\mathrm{FPR}=\frac{p}{1-p}\cdot\frac{1-\mathrm{PPV}}{\mathrm{PPV}}\cdot(1-\mathrm{FNR})$. Group A: $\frac{0.5}{0.5}\cdot\frac{0.2}{0.8}\cdot0.8=0.20$. Group B: $\frac{0.2}{0.8}\cdot0.25\cdot0.8=0.05$.
(b) FNR equal (0.2) but FPR differs (0.20 versus 0.05): equalized odds fails although predictive parity holds. Forcing equal FPR would require either dropping PPV in one group or changing FNR.
(c) An invasive follow-up makes false positives costly, which argues for equalizing the **FPR** (and for high PPV) so that healthy members of any group are equally unlikely to be subjected to it; one could still set the thresholds separately by group, subject to the legal and ethical context. There is no universally right answer, and the stakeholder discussion is part of the deliverable.
:::
:::

:::exercise Exercise 8: DP-SGD noise scale (derivation and design)
A DP-SGD run uses clip norm $C=1$, noise multiplier $\sigma=1.1$, expected batch size $B=256$, in a model with $p=10^6$ parameters. (a) What is the standard deviation of the noise per coordinate of the averaged gradient, and the expected norm of the total noise vector? (b) If a typical clipped per-example gradient has norm about $0.5$, estimate the norm of the averaged signal at best (all gradients aligned). (c) What do you change to improve the signal-to-noise ratio? List three levers and a downside for each.
:::details Solution
(a) Noise on the sum has std $\sigma C=1.1$ per coordinate; after dividing by $B$, std $=1.1/256\approx0.0043$. The noise vector norm is about $0.0043\sqrt{10^6}=4.3$.
(b) If all per-example gradients are aligned the average has norm $0.5$ at most (realistically much less, since gradients disagree). So the noise norm (4.3) can exceed the signal by an order of magnitude; progress is possible only because the signal accumulates across steps while the noise averages out, and because gradients live in a low-dimensional subspace.
(c) Levers: larger batch size (noise $\propto1/B$; downside: fewer steps per epoch or a bigger privacy cost per step as $q$ grows, and compute); smaller $\sigma$ (less noise; downside: larger $\varepsilon$); fewer parameters, for example fine-tune an adapter or last layer of a pretrained model (noise norm $\propto\sqrt p$; downside: lower capacity); tuning the clip norm $C$ (reduces bias when $C$ is near the median gradient norm; downside: it needs tuning, and a too-large $C$ means more noise relative to the signal).
:::
:::

:::exercise Exercise 9: Linear-mode connectivity of the NTK regime (challenging)
Consider the linearized model $f_{\text{lin}}(\boldsymbol{x};\boldsymbol{\theta})=f(\boldsymbol{x};\boldsymbol{\theta}_0)+\nabla_{\boldsymbol{\theta}}f(\boldsymbol{x};\boldsymbol{\theta}_0)^\top(\boldsymbol{\theta}-\boldsymbol{\theta}_0)$ trained by gradient flow on squared loss. (a) Show that the loss is convex in $\boldsymbol{\theta}$. (b) Derive the trajectory of the training-set predictions $\boldsymbol{u}(t)=f_{\text{lin}}(X;\boldsymbol{\theta}_t)$ and show $\boldsymbol{u}(t)-\boldsymbol{y}=e^{-t K}(\boldsymbol{u}(0)-\boldsymbol{y})$, where $K=JJ^\top$ with $J=\nabla_{\boldsymbol{\theta}}f(X;\boldsymbol{\theta}_0)\in\R^{n\times p}$. (c) What does the smallest eigenvalue of $K$ say about the rate?
:::details Solution
(a) $f_{\text{lin}}$ is affine in $\boldsymbol{\theta}$, so $\mathcal{L}=\frac12\norm{\boldsymbol{u}-\boldsymbol{y}}^2$ is a convex quadratic in $\boldsymbol{\theta}$.
(b) With $\boldsymbol{u}=\boldsymbol{u}_0+J(\boldsymbol{\theta}-\boldsymbol{\theta}_0)$, gradient flow gives $\dot{\boldsymbol{\theta}}=-J^\top(\boldsymbol{u}-\boldsymbol{y})$, so $\dot{\boldsymbol{u}}=J\dot{\boldsymbol{\theta}}=-JJ^\top(\boldsymbol{u}-\boldsymbol{y})=-K(\boldsymbol{u}-\boldsymbol{y})$, a linear ODE with solution $\boldsymbol{u}(t)-\boldsymbol{y}=e^{-tK}(\boldsymbol{u}(0)-\boldsymbol{y})$.
(c) In the eigenbasis of $K$ the residual component along eigenvector $\boldsymbol{v}_k$ decays as $e^{-\lambda_kt}$. The slowest component decays with $\lambda_{\min}$, so training time to reach a tolerance scales like $1/\lambda_{\min}$; if $\lambda_{\min}=0$ (e.g., duplicated or too few features, $p<n$), exact interpolation is impossible. This is the core of global convergence proofs for wide networks: show that $K$ stays positive definite near initialization.
:::
:::

:::exercise Exercise 10: Design a trustworthiness audit (open-ended)
You are handed a CNN that classifies chest X-rays for pneumonia, trained at Hospital A, to be deployed at Hospital B. Draft a one-page audit plan that covers: distribution shift, calibration, subgroup performance, explanations, privacy of the training data, and the model card. Name at least one concrete test, one metric and one acceptance criterion for each.
:::details Solution
One reasonable plan. **Shift:** collect an unlabeled sample from B, train a domain classifier (AUROC near 0.5 means little covariate shift), compare ID/OOD scores (energy), label a small B sample to measure accuracy; criterion: accuracy drop under 3 points or a recalibration/fine-tuning plan. **Calibration:** reliability diagram and ECE on labelled B data before and after temperature scaling on a B validation split; criterion: ECE below 0.03 and an abstention rule at a confidence threshold. **Subgroups:** TPR/FPR by sex, age band, scanner model, and intersections, with bootstrap confidence intervals; criterion: TPR gap below an agreed threshold, and no group with TPR below a clinical minimum. **Explanations:** Grad-CAM on a stratified sample read by a radiologist plus a randomization sanity check and a deletion curve; look for shortcut cues (text markers, tubes, scanner artefacts). **Privacy:** loss-threshold and LiRA membership tests against held-out patients; consider DP fine-tuning; criterion: attack TPR at 1% FPR close to chance. **Model card:** intended use, exclusion criteria (paediatric patients?), disaggregated metrics, known failures, monitoring and update plan, and who to contact. Add post-deployment drift monitoring.
:::
:::

## Common mistakes and debugging tips {#mistakes}

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| Adversarial attack does nothing (accuracy unchanged) | Model in `train()` mode with dropout, attack gradient is zero because the logits saturate, or input not in `requires_grad` | `model.eval()`, use logits-based margin loss, `x.requires_grad_(True)`, check `grad.abs().max()` |
| Adversarial examples exceed the budget | Forgot projection, or clipping to $[0,1]$ happens before adding the perturbation | Project onto $\epsilon$-ball *and* clamp to the valid range after every step |
| Epsilon is "8/255" but inputs were normalized by mean/std | Perturbation is applied in normalized space | Attack in pixel space, normalize inside the model (a first layer), or rescale $\epsilon$ by the std |
| Robust accuracy 0% at small $\epsilon$ after "adversarial training" | Catastrophic overfitting of single-step training, or training on perturbations of a stale model | Use PGD, random start, early stopping on a PGD validation set |
| "Defense" works against FGSM but not PGD, or black-box beats white-box | Gradient masking | Adaptive attacks, AutoAttack, check unbounded-$\epsilon$ attack reaches 0% |
| Evaluating robustness on the same samples used for attack tuning | Leaks into hyperparameters | Hold out an attack-evaluation split |
| ECE looks great but the model is useless | Always predicting base rates, or too few samples per bin | Report accuracy, NLL, Brier; use equal-mass bins; bootstrap |
| Temperature fitted on the test set | Optimistic calibration numbers | Fit on a validation split; evaluate on test |
| Calibration collapses after deployment | Distribution shift | Monitor ECE on a labelled sample; ensembles; recalibrate per domain |
| MSP OOD detector flags everything or nothing | Threshold chosen on one OOD set; detector given softmax of a different temperature | Fix a threshold from ID data only (e.g. 95% TPR) and test on several OOD sets |
| Saliency map looks nice but is unchanged after randomizing weights | The method (e.g. guided backprop) is mostly edge detection | Run the Adebayo et al. sanity checks; add deletion/insertion curves |
| Grad-CAM highlights the wrong region or is empty | Wrong layer, gradients of softmax instead of logit, or all-ReLU-negative CAM | Use the last conv layer, class logit, and check `cam.max()` before normalizing |
| IG attributions do not sum to $F(x)-F(x')$ | Too few steps, or baseline mismatch | Increase `steps`; verify completeness |
| Probe gets 95% accuracy on random-init network | Probe too powerful or label leakage via input | Linear probe, control task, selectivity |
| Group metrics based on tiny subgroups swing wildly | High variance | Confidence intervals; combine with other evidence; collect more data |
| DP-SGD training diverges or accuracy is near chance | Clip norm too small/large, noise too high, tiny batch | Tune $C$ near the median gradient norm, increase batch size, use pretrained features |
| Reported $\varepsilon$ cannot be trusted | Sampling without replacement or fixed-size batches while the accountant assumes Poisson; reused data across steps in ways the accountant ignores | Use the library's data loader; document the accountant and assumptions |
| Conformal sets have under-coverage in deployment | Exchangeability violated (shift), calibration data used in training | Recalibrate on recent data; use weighted/adaptive conformal methods; keep calibration disjoint |

:::tip A default workflow for evaluating trustworthiness
(1) Accuracy on in-distribution *and* shifted/corrupted data; (2) calibration (ECE, NLL, reliability diagram) and a rule for abstention; (3) robustness under a *named* threat model with an adaptive attack; (4) subgroup metrics with confidence intervals; (5) explanations validated by sanity checks and interventions; (6) a privacy review of what the model could reveal; (7) a model card. Skipping steps is fine if you say that you skipped them.
:::

## Key takeaways {#takeaways}

:::key Key takeaways
- **Adversarial examples** come from maximizing the loss in a small perturbation set. FGSM is the exact first-order solution for $\ell_\infty$ ($\epsilon\,\mathrm{sign}(\nabla_{\boldsymbol{x}}\mathcal{L})$); PGD iterates it with projection and is the standard evaluation. High dimension plus near-linearity makes small perturbations powerful.
- **A threat model** (goal, capability, knowledge, setting) is part of every robustness claim. Evaluate with adaptive attacks; gradient masking is the usual reason for a false sense of security.
- **Adversarial training** (a min-max problem) is the strongest empirical defense, costing about $(K+1)\times$ compute and some clean accuracy. **Certified** defenses (IBP, Lipschitz, randomized smoothing) give provable but smaller radii.
- **Distribution shift** (covariate, label, concept) is the norm; OOD detectors (MSP, energy, Mahalanobis) are useful but not universal.
- **Uncertainty:** aleatoric (irreducible) versus epistemic (reducible, disagreement among models). Deep ensembles are the strongest simple method; MC dropout is a cheap approximation. **Calibration** is measured by reliability diagrams and ECE and often fixed with a single temperature, which does not change accuracy.
- **Conformal prediction** wraps any model to give sets with marginal coverage $\ge1-\alpha$ under exchangeability, with a short proof; the guarantee is marginal, not conditional.
- **Explanations:** gradients are local and noisy; SmoothGrad averages; Grad-CAM weights feature maps by pooled gradients; Integrated Gradients satisfies completeness and sensitivity but depends on the baseline. Attention is not an explanation, and a probe shows decodability, not use. Validate with sanity checks and interventions. Mechanistic interpretability seeks causal circuits; superposition and sparse autoencoders are central ideas.
- **Fairness** criteria (demographic parity, equalized odds, predictive parity) conflict when base rates differ; choosing among them is a normative decision. Audit with disaggregated metrics.
- **Privacy:** models leak membership and training data; $(\varepsilon,\delta)$-DP bounds that leakage; DP-SGD clips per-example gradients and adds Gaussian noise, at a cost in accuracy that falls hardest on rare groups.
- **Responsible release** means model cards, red-teaming, staged access and monitoring. Documentation is only valuable if it contains uncomfortable numbers.
- **Theory:** deep networks fit random labels yet generalize (Zhang et al.), so capacity-based bounds fail; SGD has implicit biases (minimum norm, max margin); the NTK describes a lazy regime; lottery tickets, grokking and double descent each show that training dynamics matter. No unified theory exists.
- **Open problems** (reasoning, agents, data efficiency, continual learning, world models) are mostly questions of evaluation and reliability as much as of capability.
:::

## Further reading {#reading}

Essential (marked with a star) and supporting references. Check the venue or arXiv listing for the latest version.

**Robustness and adversarial examples**
- (Essential) Goodfellow, Shlens, Szegedy. *Explaining and Harnessing Adversarial Examples.* ICLR 2015 (arXiv:1412.6572).
- Szegedy et al. *Intriguing properties of neural networks.* ICLR 2014 (arXiv:1312.6199).
- (Essential) Madry, Makelov, Schmidt, Tsipras, Vladu. *Towards Deep Learning Models Resistant to Adversarial Attacks.* ICLR 2018 (arXiv:1706.06083).
- Athalye, Carlini, Wagner. *Obfuscated Gradients Give a False Sense of Security.* ICML 2018.
- Cohen, Rosenfeld, Kolter. *Certified Adversarial Robustness via Randomized Smoothing.* ICML 2019.
- Ilyas et al. *Adversarial Examples Are Not Bugs, They Are Features.* NeurIPS 2019.
- Croce and Hein. *Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks* (AutoAttack). ICML 2020.
- Zhang et al. *Theoretically Principled Trade-off between Robustness and Accuracy* (TRADES). ICML 2019.

**Shift and uncertainty**
- Hendrycks and Gimpel. *A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks.* ICLR 2017.
- Liu, Wang, Owens, Li. *Energy-based Out-of-distribution Detection.* NeurIPS 2020.
- Lee, Lee, Lee, Shin. *A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks.* NeurIPS 2018.
- Lakshminarayanan, Pritzel, Blundell. *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles.* NeurIPS 2017.
- Gal and Ghahramani. *Dropout as a Bayesian Approximation.* ICML 2016.
- (Essential) Guo, Pleiss, Sun, Weinberger. *On Calibration of Modern Neural Networks.* ICML 2017.
- Ovadia et al. *Can You Trust Your Model's Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift.* NeurIPS 2019.
- Angelopoulos and Bates. *A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification.* 2021 (arXiv:2107.07511).

**Interpretability**
- (Essential) Selvaraju et al. *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.* ICCV 2017.
- Sundararajan, Taly, Yan. *Axiomatic Attribution for Deep Networks.* ICML 2017.
- Smilkov et al. *SmoothGrad: removing noise by adding noise.* 2017 (arXiv:1706.03825).
- Adebayo et al. *Sanity Checks for Saliency Maps.* NeurIPS 2018.
- Jain and Wallace. *Attention is not Explanation.* NAACL 2019; Wiegreffe and Pinter. *Attention is not not Explanation.* EMNLP 2019.
- Hewitt and Liang. *Designing and Interpreting Probes with Control Tasks.* EMNLP 2019.
- Elhage et al. *A Mathematical Framework for Transformer Circuits.* Transformer Circuits Thread, 2021; *Toy Models of Superposition*, 2022.
- Bricken et al. *Towards Monosemanticity: Decomposing Language Models With Dictionary Learning.* Transformer Circuits Thread, 2023.

**Fairness, privacy, safety**
- Hardt, Price, Srebro. *Equality of Opportunity in Supervised Learning.* NeurIPS 2016.
- Kleinberg, Mullainathan, Raghavan. *Inherent Trade-Offs in the Fair Determination of Risk Scores.* ITCS 2017; Chouldechova. *Fair prediction with disparate impact.* Big Data 2017.
- Barocas, Hardt, Narayanan. *Fairness and Machine Learning: Limitations and Opportunities.* MIT Press, 2023 (free online).
- Buolamwini and Gebru. *Gender Shades.* FAT* 2018.
- Shokri, Stronati, Song, Shmatikov. *Membership Inference Attacks Against Machine Learning Models.* IEEE S&P 2017.
- Dwork and Roth. *The Algorithmic Foundations of Differential Privacy.* 2014.
- (Essential) Abadi et al. *Deep Learning with Differential Privacy.* CCS 2016.
- Mitchell et al. *Model Cards for Model Reporting.* FAT* 2019; Gebru et al. *Datasheets for Datasets.* CACM 2021.

**Theory**
- (Essential) Zhang, Bengio, Hardt, Recht, Vinyals. *Understanding deep learning requires rethinking generalization.* ICLR 2017.
- Jacot, Gabriel, Hongler. *Neural Tangent Kernel: Convergence and Generalization in Neural Networks.* NeurIPS 2018.
- Chizat, Oyallon, Bach. *On Lazy Training in Differentiable Programming.* NeurIPS 2019.
- (Essential) Frankle and Carbin. *The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks.* ICLR 2019.
- Belkin, Hsu, Ma, Mandal. *Reconciling modern machine-learning practice and the classical bias-variance trade-off.* PNAS 2019.
- Nakkiran et al. *Deep Double Descent.* ICLR 2020.
- Power et al. *Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets.* 2022 (arXiv:2201.02177); Nanda et al. *Progress measures for grokking via mechanistic interpretability.* ICLR 2023.
- Soudry et al. *The Implicit Bias of Gradient Descent on Separable Data.* JMLR 2018.
- Bartlett, Long, Lugosi, Tsigler. *Benign overfitting in linear regression.* PNAS 2020.
