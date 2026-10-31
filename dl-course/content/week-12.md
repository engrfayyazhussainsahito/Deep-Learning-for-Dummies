---
title: "Generative Models I: Autoencoders, VAEs, and GANs"
phase: Advanced Topics
summary: A taxonomy of deep generative models, then autoencoders (undercomplete, denoising, sparse), variational autoencoders with a full ELBO derivation, the reparameterization trick, beta-VAE, posterior collapse and VQ-VAE, and GANs from the minimax game and optimal discriminator to WGAN-GP, DCGAN, conditional GANs, CycleGAN, StyleGAN and evaluation with IS, FID and precision/recall. The lab builds a VAE and a DCGAN on MNIST.
---

:::key Learning objectives
By the end of this week you should be able to:
- Place a generative model in the taxonomy of explicit versus implicit density models and say what each family can and cannot compute.
- Explain what an autoencoder learns, how undercomplete, denoising and sparse variants regularize the code, and why a plain autoencoder is not a generative model.
- Derive the evidence lower bound (ELBO) two ways, derive the closed-form KL divergence between a diagonal Gaussian and the standard normal, and explain why the reparameterization trick is needed.
- Describe beta-VAE, posterior collapse and VQ-VAE, and diagnose collapse from training curves.
- Derive the optimal discriminator and show that the GAN objective equals a Jensen-Shannon divergence at the optimum; explain the non-saturating loss, mode collapse, and why training is unstable.
- State the Wasserstein distance, the WGAN objective, and the gradient penalty; sketch DCGAN, conditional GANs, CycleGAN and StyleGAN.
- Compute and interpret Inception Score, Frechet Inception Distance and precision/recall, including their weaknesses.
- Implement a VAE and a DCGAN on MNIST, and produce latent-space interpolations.
:::

:::note Logistics this week
**Assignment 3 is due at the end of Week 12.** **Project feedback meetings** also take place this week: bring your problem statement, your baseline results and the one question you most want answered. Generative models are a popular project topic, so expect questions about evaluation (how will you *know* your samples are good?), which is the last section of this page.
:::

Every model you have seen so far answers a *discriminative* question: given $x$, what is $y$? A **generative model** answers a different one: what does the data itself look like? It learns a distribution $p_\theta(x)$ close to the true data distribution $p_\text{data}(x)$ so that we can draw new samples (images, molecules, audio), evaluate how plausible an input is, learn compact representations, or fill in missing parts. [Week 11](week-11.html) used self-supervision to learn *features*. This week and the next use the data's own structure to learn the *distribution*.

The difficulty is geometric. An image with $64\times64\times3=12{,}288$ pixels lives in a space of enormous dimension, yet natural images occupy a tiny, curved, low-dimensional sliver of it (the *manifold hypothesis*). A generative model must find that sliver. The two families in this lesson attack the problem differently. **Variational autoencoders** write down a latent-variable model and maximize a tractable lower bound on the likelihood. **Generative adversarial networks** give up on the likelihood altogether and train a generator to fool a learned critic. [Week 13](week-13.html) will add diffusion models, which now dominate image generation, and the ideas here (latent spaces, KL divergences, adversarial losses, perceptual metrics) reappear there.

## A map of generative models {#taxonomy}

{{figure: week12-taxonomy | A taxonomy of deep generative models by how they represent the density. Explicit models can evaluate $p_\theta(x)$ exactly or bound it; implicit models only provide a sampler.}}

The first question to ask of any generative model is **what can you compute with it?**

- **Explicit, tractable density.** You can evaluate $p_\theta(x)$ exactly. *Autoregressive models* factor $p(x)=\prod_i p(x_i\mid x_{<i})$ (PixelCNN, the GPT family of [Week 10](week-10.html)). *Normalizing flows* push a simple density through an invertible map and track the change of variables. Exact likelihood is wonderful for model comparison, but the architectural constraints (ordering, invertibility) are costly, and autoregressive sampling is sequential.
- **Explicit, approximate density.** The model defines a density but you only evaluate a bound or an unnormalized value. *VAEs* maximize the ELBO. *Energy-based models* define $p(x)\propto e^{-E_\theta(x)}$ with an intractable normalizer.
- **Implicit density.** There is a sampler $x=G_\theta(z)$ but no way to evaluate $p_\theta(x)$. *GANs* are the main example. Sampling is a single forward pass, which is fast, but there is no likelihood to monitor and no guarantee that every part of the data is covered.
- **Iterative and score-based models.** *Diffusion models* (Week 13) learn to reverse a gradual noising process. They optimize a variational bound (so they are close cousins of VAEs) but generate by many small steps (so they behave like samplers).

Three desiderata pull against each other: **sample quality**, **mode coverage and likelihood evaluation**, and **sampling speed**. For several years the "generative trilemma" was a fair summary: GANs gave fast, sharp samples but poor coverage and unstable training; VAEs gave good coverage and stable training but blurry samples; autoregressive models gave exact likelihoods but slow sampling. Diffusion improved all three at the cost of speed. Knowing which axis a method sacrifices lets you choose sensibly.

:::history A short history
Restricted Boltzmann machines and deep belief nets (Hinton, Osindero and Teh, 2006) were the first modern deep generative models; Hinton and Salakhutdinov (Science, 2006) used deep autoencoders for dimensionality reduction. Denoising autoencoders (Vincent et al., 2008) tied representation learning to a noise-removal task. In 2013 and 2014, Kingma and Welling (and, independently, Rezende, Mohamed and Wierstra) introduced the VAE, and Goodfellow and colleagues introduced the GAN. From 2015 to 2019 GAN research moved fast: DCGAN (2015), Wasserstein GAN (2017), progressive growing (2018), BigGAN (2018) and StyleGAN (2018 and 2019), whose faces became the symbol of the field's progress.
:::

## Autoencoders {#autoencoders}

An **autoencoder** (AE) is a network trained to copy its input to its output through a narrow channel. It has an **encoder** $f_\phi:\R^d\to\R^k$ that maps $x$ to a code $z=f_\phi(x)$ and a **decoder** $g_\theta:\R^k\to\R^d$ that returns a reconstruction $\hat x=g_\theta(z)$. Training minimizes a reconstruction loss, usually squared error for real-valued data or binary cross-entropy for pixels in $[0,1]$:

$$
\mathcal{L}_\text{AE}(\phi,\theta)=\frac1N\sum_{n=1}^N \norm{x_n-g_\theta(f_\phi(x_n))}^2 .
$$

{{figure: week12-autoencoder | An autoencoder squeezes the input through a bottleneck code $z$ with $k<d$ units and reconstructs it. The loss compares $\hat x$ with $x$; no labels are needed.}}

Copying is trivial unless the network is prevented from learning the identity function. Every variant of the autoencoder is a different way of preventing this, and the regularizer shapes what the code means.

### Undercomplete autoencoders and PCA {#undercomplete}

The simplest constraint is a **bottleneck**: choose $k<d$ so the code cannot hold everything. With a *linear* encoder and decoder and squared error, the optimum has a closed form. The decoder spans the top-$k$ principal subspace, so a linear autoencoder learns exactly the same subspace as **principal component analysis** (Baldi and Hornik, 1989), although not necessarily the same basis within it. The reconstruction error is the sum of the discarded eigenvalues of the data covariance. The chart shows this on the $8\times8$ digits dataset from scikit-learn (1,797 images, $d=64$): the error falls quickly because digits are highly redundant.

```chart
{
  "type": "line",
  "caption": "Mean squared reconstruction error (summed over the 64 pixels, pixels scaled to [0,1]) of the optimal linear autoencoder, i.e. PCA, on the scikit-learn 8x8 digits dataset as a function of the code size k. Computed with an SVD in NumPy. Total variance is 4.69, and k = 16 already removes 85% of it.",
  "height": 300,
  "data": { "labels": [1, 2, 4, 8, 16, 32, 48, 64], "datasets": [ { "label": "reconstruction error (PCA, linear AE optimum)", "data": [3.994, 3.355, 2.407, 1.53, 0.707, 0.158, 0.007, 0.0], "borderColor": "var(--c-accent)", "backgroundColor": "var(--c-accent)", "pointRadius": 4, "tension": 0.2 } ] },
  "options": { "scales": { "x": {"title": {"display": true, "text": "code size k"}}, "y": {"min": 0, "title": {"display": true, "text": "squared error per image"}} } }
}
```

With nonlinear layers (convolutional or MLP encoders and decoders), the autoencoder can follow a *curved* manifold rather than a flat subspace, which is why a deep autoencoder with $k=2$ separates MNIST digits far better than PCA with $k=2$. Hinton and Salakhutdinov made that point in 2006.

```python:autoencoder.py
import torch, torch.nn as nn, torch.nn.functional as F

class AutoEncoder(nn.Module):
    def __init__(self, d=784, k=32, hidden=256):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, k))
        self.dec = nn.Sequential(nn.Linear(k, hidden), nn.ReLU(), nn.Linear(hidden, d))   # logits for pixels in [0, 1]

    def forward(self, x):
        z = self.enc(x)                       # (B, k) the code
        return self.dec(z), z                 # (B, d) logits, (B, k)

model = AutoEncoder()
x = torch.rand(16, 784)                       # stand-in for a batch of flattened 28x28 images
logits, z = model(x)
loss = F.binary_cross_entropy_with_logits(logits, x)    # per-pixel Bernoulli reconstruction loss
print(logits.shape, z.shape, loss.item())               # torch.Size([16, 784]) torch.Size([16, 32])
```

### Denoising autoencoders {#denoising}

A **denoising autoencoder** (Vincent, Larochelle, Bengio and Manzagol, ICML 2008) corrupts the input, $\tilde x\sim q(\tilde x\mid x)$ (Gaussian noise, random pixel dropout, or masking), and trains the network to reconstruct the *clean* $x$:

$$
\mathcal{L}_\text{DAE}=\E_{x\sim p_\text{data}}\,\E_{\tilde x\sim q(\cdot\mid x)}\norm{x-g_\theta(f_\phi(\tilde x))}^2 .
$$

Because the identity map no longer reduces the loss, the network must learn the *structure* of the data: which pixel patterns are plausible, and therefore which direction leads back to the manifold. A known result (Vincent, 2011; Alain and Bengio, 2014) says that for small Gaussian corruption $\tilde x=x+\sigma\epsilon$, the optimal denoiser satisfies

$$
g^*(\tilde x)-\tilde x\;\approx\;\sigma^2\,\nabla_{\tilde x}\log p_\sigma(\tilde x),
$$

i.e. the reconstruction residual points along the **score** of the (noised) data density. This connection is the seed of score-based and diffusion models in [Week 13](week-13.html), and the masked autoencoder of [Week 11](week-11.html#mae) is a denoising autoencoder whose "noise" is the deletion of 75% of the patches. Note that a denoising autoencoder can have a code *larger* than the input; the corruption alone does the regularizing.

```python:denoising_sparse.py
import torch, torch.nn as nn, torch.nn.functional as F

def denoising_step(model, x, opt, sigma=0.3):
    x_noisy = (x + sigma * torch.randn_like(x)).clamp(0, 1)      # corrupt the input
    logits, _ = model(x_noisy)
    loss = F.binary_cross_entropy_with_logits(logits, x)          # but reconstruct the CLEAN target
    opt.zero_grad(); loss.backward(); opt.step()
    return loss.item()

def sparse_loss(logits, x, z_act, l1=1e-3):
    """Sparse autoencoder: reconstruction + L1 penalty on the (non-negative) code activations."""
    rec = F.binary_cross_entropy_with_logits(logits, x)
    return rec + l1 * z_act.abs().mean()

class SparseAE(nn.Module):
    def __init__(self, d=784, k=512):                             # overcomplete: k > d is fine when the code is sparse
        super().__init__()
        self.enc, self.dec = nn.Linear(d, k), nn.Linear(k, d)
    def forward(self, x):
        a = F.relu(self.enc(x))                                   # (B, k) non-negative code
        return self.dec(a), a

m = SparseAE(); x = torch.rand(32, 784)
logits, a = m(x)
print(sparse_loss(logits, x, a).item(), (a > 0).float().mean().item())   # fraction of active units
```

### Sparse and contractive autoencoders {#sparse}

A **sparse autoencoder** may have *more* code units than inputs ($k>d$) but penalizes activity, either with an $L_1$ term $\lambda\norm{z}_1$ (as in the code above), or with a KL penalty that pushes each unit's average activation toward a small target $\rho$, or with a hard top-$k$ rule that keeps only the $k$ largest activations (Makhzani and Frey, 2013, "k-Sparse Autoencoders"). The code then behaves like a learned dictionary in which each input is explained by a few active atoms; individual units tend to become interpretable detectors. This idea has recently returned in the interpretability of large language models, where sparse autoencoders are trained on a model's hidden activations to find human-readable features. A **contractive autoencoder** (Rifai et al., 2011) penalizes the Frobenius norm of the encoder's Jacobian, $\norm{\partial f_\phi/\partial x}_F^2$, making the code insensitive to small input changes.

### Why an autoencoder is not a generative model {#ae-not-generative}

If you sample $z$ from some distribution and decode it, you will usually get garbage. The encoder maps the data to arbitrary islands in code space, with unused gaps between them and no rule for what the gaps mean. We have no distribution over $z$ to sample from, and decoding a point between two islands has no reason to produce a plausible image. A fix that comes to mind is to fit a Gaussian mixture to the codes after training. That works somewhat (and is a legitimate baseline), but the **VAE** builds the answer directly into training: it forces the code distribution to be close to a known prior so that sampling from the prior is meaningful.

## Variational autoencoders {#vae}

### The latent-variable model {#latent-variable}

A VAE (Kingma and Welling, ICLR 2014, "Auto-Encoding Variational Bayes") posits a simple generative story:

1. Draw a latent code $z\sim p(z)=\mathcal{N}(0,\mathbf{I})$ with $z\in\R^k$.
2. Draw the data $x\sim p_\theta(x\mid z)$, where the parameters of this conditional are produced by a neural network (the **decoder**): a Bernoulli over pixels with logits $g_\theta(z)$, or a Gaussian with mean $g_\theta(z)$ and fixed variance.

The marginal likelihood is $p_\theta(x)=\int p_\theta(x\mid z)p(z)\,dz$, an integral over all latent codes. For a neural-network decoder it has no closed form, and a naive Monte Carlo estimate $\frac1S\sum_s p_\theta(x\mid z_s)$ with $z_s\sim p(z)$ is hopeless in high dimension: almost every random $z$ explains $x$ terribly, so the estimator is dominated by rare lucky samples. The posterior $p_\theta(z\mid x)=p_\theta(x\mid z)p(z)/p_\theta(x)$, which would tell us which codes are plausible for a given $x$, is intractable for the same reason.

The variational trick is to introduce an **approximate posterior** $q_\phi(z\mid x)$ (the **encoder**), usually a diagonal Gaussian $\mathcal{N}(\mu_\phi(x),\diag(\sigma_\phi^2(x)))$, and to optimize a lower bound on $\log p_\theta(x)$ that involves $q_\phi$.

### Deriving the ELBO {#elbo}

:::math Derivation 1: from the log-evidence and an importance weight (Jensen)
Start from the quantity we want, $\log p_\theta(x)$. Multiply and divide by $q_\phi(z\mid x)$ inside the integral, so that it becomes an expectation under $q_\phi$:

$$
\log p_\theta(x)=\log\int p_\theta(x,z)\,dz=\log\int q_\phi(z\mid x)\frac{p_\theta(x,z)}{q_\phi(z\mid x)}\,dz=\log\E_{q_\phi(z\mid x)}\left[\frac{p_\theta(x,z)}{q_\phi(z\mid x)}\right].
$$

The logarithm is concave, so Jensen's inequality ($\log\E[Y]\ge\E[\log Y]$) gives

$$
\log p_\theta(x)\;\ge\;\E_{q_\phi(z\mid x)}\left[\log\frac{p_\theta(x,z)}{q_\phi(z\mid x)}\right]=:\mathcal{L}(\theta,\phi;x)\quad(\text{the ELBO}).
$$

Now split the joint, $p_\theta(x,z)=p_\theta(x\mid z)p(z)$:

$$
\mathcal{L}(\theta,\phi;x)=\underbrace{\E_{q_\phi(z\mid x)}\big[\log p_\theta(x\mid z)\big]}_{\text{reconstruction}}-\underbrace{\KL\big(q_\phi(z\mid x)\,\|\,p(z)\big)}_{\text{regularizer}} .
$$
:::

:::math Derivation 2: the gap is a KL divergence
Jensen's route shows the inequality but not how loose it is. Start instead from the true posterior. For any $z$, $\log p_\theta(x)=\log p_\theta(x,z)-\log p_\theta(z\mid x)$. Take the expectation under $q_\phi(z\mid x)$ (the left side does not depend on $z$) and add and subtract $\log q_\phi(z\mid x)$:

$$
\begin{aligned}
\log p_\theta(x)&=\E_q\big[\log p_\theta(x,z)-\log q_\phi(z\mid x)\big]+\E_q\big[\log q_\phi(z\mid x)-\log p_\theta(z\mid x)\big]\\
&=\mathcal{L}(\theta,\phi;x)+\KL\big(q_\phi(z\mid x)\,\|\,p_\theta(z\mid x)\big).
\end{aligned}
$$

Since KL is non-negative, $\mathcal{L}\le\log p_\theta(x)$, with equality exactly when $q_\phi(z\mid x)=p_\theta(z\mid x)$. For a fixed $\theta$, maximizing the ELBO over $\phi$ is the same as minimizing the KL from the approximate to the true posterior. Over $\theta$, it pushes up a lower bound on the log-likelihood. The gap is the quality of the encoder, and we can never evaluate it, because it involves the posterior we cannot compute.
:::

{{figure: week12-elbo-gap | The log-evidence is a constant for given $\theta$ and $x$. It splits into the ELBO plus a non-negative gap, the KL from the approximate posterior to the true posterior. Improving the encoder shrinks the gap; improving the decoder raises the whole bar.}}

Read the two ELBO terms as a *compression* trade-off. The **reconstruction term** asks the code to carry enough information about $x$ for the decoder to rebuild it. The **KL term** charges the encoder for every nat of information it puts in the code beyond what the prior already expects, and keeps the codes of different inputs overlapping and centred, so that a draw from $p(z)$ falls in a region the decoder has seen. A VAE is therefore an autoencoder with a *noise* injected in the code (the variance of $q$) and a *penalty* for deviating from the prior.

:::note Another view: what does the average KL measure?
Averaged over the data, $\E_x\KL(q_\phi(z\mid x)\|p(z))=I_q(x;z)+\KL(q_\phi(z)\|p(z))$ where $q_\phi(z)=\E_x q_\phi(z\mid x)$ is the *aggregate posterior* and $I_q$ is the mutual information between data and code under the encoder. The first piece is the information the code carries about the input (the "rate"); the second measures how well the aggregate posterior matches the prior (the "hole" problem). A good VAE needs both small, which is why "prior hole" analyses and rate-distortion views of VAEs are popular (Alemi et al., 2018, "Fixing a broken ELBO").
:::

### The closed-form KL divergence {#gaussian-kl}

For the standard choices, the KL term needs no sampling. First, the intuition: a diagonal Gaussian encoder against a standard-normal prior is a sum over independent latent dimensions, and each dimension pays a penalty for having a mean away from 0 or a variance away from 1.

:::math Derivation: $\KL(\mathcal{N}(\mu,\sigma^2)\,\|\,\mathcal{N}(0,1))$
By definition $\KL(q\|p)=\E_q[\log q(z)-\log p(z)]$. For $q=\mathcal{N}(\mu,\sigma^2)$ and $p=\mathcal{N}(0,1)$,

$$
\log q(z)=-\tfrac12\log(2\pi)-\tfrac12\log\sigma^2-\frac{(z-\mu)^2}{2\sigma^2},\qquad \log p(z)=-\tfrac12\log(2\pi)-\frac{z^2}{2}.
$$

Take the expectation under $q$, using $\E_q[(z-\mu)^2]=\sigma^2$ and $\E_q[z^2]=\mu^2+\sigma^2$:

$$
\KL=-\tfrac12\log\sigma^2-\frac{\sigma^2}{2\sigma^2}+\frac{\mu^2+\sigma^2}{2}=\tfrac12\big(\mu^2+\sigma^2-1-\log\sigma^2\big).
$$

For a $k$-dimensional diagonal Gaussian the KL is the sum over dimensions, $\tfrac12\sum_{j=1}^k(\mu_j^2+\sigma_j^2-1-\log\sigma_j^2)$. It is zero if and only if $\mu_j=0$ and $\sigma_j=1$ for all $j$. In terms of the log-variance $\ell=\log\sigma^2$ that networks usually output, it is $\tfrac12(\mu^2+e^{\ell}-1-\ell)$, which is convex in both arguments.
:::

{{demo: week12-kl-gaussians | KL divergence between Gaussians: asymmetry, closed form, and mode seeking | 520}}

The demo also shows why the *direction* of the KL matters: the VAE minimizes $\KL(q\|p_\theta(z\mid x))$ (reverse), which is **mode-seeking**: $q$ tolerates missing part of $p$ but is punished for putting mass where $p$ has none. A fit of a unimodal $q$ to a bimodal posterior locks onto one mode. Maximum-likelihood training, by contrast, minimizes the forward KL $\KL(p_\text{data}\|p_\theta)$, which is **mass-covering** and punishes missing modes. Keep both in mind for GANs.

### The reparameterization trick {#reparameterization}

To maximize the ELBO by gradient ascent, we need $\nabla_\phi\E_{q_\phi(z\mid x)}[\log p_\theta(x\mid z)]$. The expectation depends on $\phi$ through the *distribution* being sampled from, and a sample $z\sim q_\phi$ is a random draw: backpropagation cannot pass through "sampling". The generic score-function (REINFORCE) estimator works, but has high variance.

The **reparameterization trick** rewrites the sample as a deterministic, differentiable function of the parameters and an independent noise variable:

$$
z=\mu_\phi(x)+\sigma_\phi(x)\odot\epsilon,\qquad \epsilon\sim\mathcal{N}(0,\mathbf{I}).
$$

Now $z$ has exactly the distribution $q_\phi(z\mid x)$, but the randomness lives in $\epsilon$, which does not depend on $\phi$. The expectation becomes $\E_{\epsilon}[\log p_\theta(x\mid\mu+\sigma\odot\epsilon)]$, and the gradient moves inside the expectation:

$$
\nabla_\phi\E_{q_\phi}[f(z)]=\E_{\epsilon}\big[\nabla_\phi f(\mu_\phi+\sigma_\phi\odot\epsilon)\big]\approx\nabla_\phi f(\mu_\phi+\sigma_\phi\odot\epsilon_1),
$$

and a single draw per data point is enough in practice because the variance is low. The same idea works for any location-scale family, and (with extra machinery) for others; it does not work directly for discrete latents, which is the reason for the Gumbel-softmax relaxation and for the straight-through estimator in VQ-VAE, below.

{{figure: week12-vae | A VAE forward pass. The encoder outputs the mean and log-variance of $q_\phi(z\mid x)$. The sample $z=\mu+\sigma\odot\epsilon$ is a deterministic function of the encoder output and of external noise, so gradients flow back to $\mu$ and $\sigma$. The loss adds the reconstruction term and the closed-form KL.}}

:::intuition Why the noise matters
Imagine the encoder wants to be accurate, so it makes $\sigma$ tiny. Then the code is a deterministic point, the KL term $-\log\sigma^2$ becomes huge, and the model pays heavily. If instead it uses a large $\sigma$, the decoder sees a smeared code and reconstructs poorly. The optimum balances these. The effect is that **nearby points in latent space decode to similar images**, since the decoder has been trained on a cloud around each code, not on a single point. This smoothness is what makes interpolation and sampling work.
:::

### The VAE loss in practice {#vae-loss}

For a Bernoulli decoder on pixels in $[0,1]$, the single-sample estimate of the negative ELBO for one image is

$$
-\mathcal{L}(x)\approx\underbrace{-\sum_{i=1}^{d}\Big[x_i\log\hat x_i+(1-x_i)\log(1-\hat x_i)\Big]}_{\text{binary cross-entropy summed over pixels}}+\tfrac12\sum_{j=1}^k\big(\mu_j^2+e^{\ell_j}-1-\ell_j\big),
$$

where $\hat x=\operatorname{sigmoid}(g_\theta(z))$. For a Gaussian decoder with fixed variance $s^2$, the reconstruction term is $\frac{1}{2s^2}\norm{x-\hat x}^2+\text{const}$, so the ratio between the two terms is controlled by $s^2$. This is the key, often overlooked point: **a squared-error VAE with the "usual" unit weights is secretly choosing a decoder noise** and thereby how strongly the KL is weighted relative to reconstruction.

```python:vae_closed_form_check.py
import numpy as np
rng = np.random.default_rng(0)

mu, sigma = 0.7, 1.9
closed = 0.5 * (mu**2 + sigma**2 - 1 - np.log(sigma**2))

# Monte Carlo estimate of E_q[log q(z) - log p(z)] with z = mu + sigma * eps
eps = rng.standard_normal(2_000_000)
z = mu + sigma * eps
log_q = -0.5 * np.log(2 * np.pi * sigma**2) - (z - mu) ** 2 / (2 * sigma**2)
log_p = -0.5 * np.log(2 * np.pi) - z**2 / 2
print(closed, (log_q - log_p).mean())      # both about 1.0345

# for k dimensions: sum over the latent dimension, mean over the batch
mu_b, logvar_b = rng.standard_normal((64, 8)), rng.standard_normal((64, 8)) * 0.3
kl_per_example = 0.5 * (mu_b**2 + np.exp(logvar_b) - 1 - logvar_b).sum(axis=1)   # (64,)
print(kl_per_example.mean())
```

```python:vae.py
import torch, torch.nn as nn, torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, d=784, hidden=400, k=20):
        super().__init__()
        self.fc1 = nn.Linear(d, hidden)
        self.fc_mu = nn.Linear(hidden, k)
        self.fc_logvar = nn.Linear(hidden, k)          # predict log sigma^2: unconstrained and numerically stable
        self.fc3 = nn.Linear(k, hidden)
        self.fc4 = nn.Linear(hidden, d)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)        # each (B, k)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)                  # sigma = exp(logvar / 2)
        eps = torch.randn_like(std)                    # noise independent of the parameters
        return mu + std * eps                          # (B, k)

    def decode(self, z):
        return self.fc4(F.relu(self.fc3(z)))           # (B, d) logits

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def vae_loss(logits, x, mu, logvar, beta=1.0):
    """Negative ELBO summed over pixels / latent dims, averaged over the batch."""
    rec = F.binary_cross_entropy_with_logits(logits, x, reduction="none").sum(dim=1)     # (B,)
    kl = 0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar).sum(dim=1)                        # (B,)
    return (rec + beta * kl).mean(), rec.mean(), kl.mean()

model = VAE()
x = torch.rand(64, 784)
logits, mu, logvar = model(x)
loss, rec, kl = vae_loss(logits, x, mu, logvar)
print(logits.shape, mu.shape, loss.item(), rec.item(), kl.item())
```

:::warning Sum, do not average, over pixels and latent dimensions
The ELBO is a sum over pixels (the log-likelihood of the whole image) and a sum over latent dimensions. If you average over pixels with `reduction="mean"` but sum the KL, you have silently multiplied the KL weight by the number of pixels (784 on MNIST). The result is an over-regularized VAE with blurry outputs and a collapsed code. If you do use means, choose the weight explicitly as a $\beta$.
:::

{{demo: week12-vae-elbo | A 1-D VAE you can optimize: ELBO, KL weight and posterior collapse | 600}}

### Beta-VAE and the rate-distortion trade-off {#beta-vae}

Higgins et al. (ICLR 2017) proposed **$\beta$-VAE**, which multiplies the KL term:

$$
\mathcal{L}_\beta=\E_{q_\phi(z\mid x)}[\log p_\theta(x\mid z)]-\beta\,\KL\big(q_\phi(z\mid x)\,\|\,p(z)\big).
$$

With $\beta>1$ the model is pushed to use less information, and to align the code with the factorized prior, which empirically encourages **disentangled** factors (one latent per factor of variation such as object size, rotation or x-position) on simple synthetic data. Burgess et al. (2018) argued for increasing the capacity target gradually. The price is worse reconstructions. With $\beta<1$ you recover sharper reconstructions but a code that departs from the prior and is a poor sampler. The cleanest view is **rate and distortion**: the KL is the *rate* (nats to describe the code), the reconstruction error is the *distortion*, and $\beta$ is a Lagrange multiplier selecting a point on the optimal rate-distortion curve. We trained a small NumPy VAE (an MLP encoder and decoder with 32 tanh units, 4 latent dimensions, hand-written backprop and Adam, 300 epochs) on 4-D points that lie on a 2-D nonlinear manifold, once for each $\beta$:

```chart
{
  "type": "line",
  "caption": "Rate-distortion trade-off in a small VAE trained in NumPy on 4-D data from a 2-D manifold with a Gaussian decoder of variance 0.1 (beta values 0.1, 0.5, 1, 2, 4, 8, 16). As beta grows the KL (rate) falls and the reconstruction error (distortion) rises; at beta = 16 the KL is essentially 0 and every latent unit has collapsed.",
  "height": 340,
  "data": { "labels": [0.1, 0.5, 1, 2, 4, 8, 16], "datasets": [
    { "label": "KL (nats, rate)", "data": [7.454, 4.539, 3.661, 2.813, 1.865, 1.023, 0.005], "borderColor": "var(--c-blue)", "yAxisID": "y", "pointRadius": 4, "tension": 0.2 },
    { "label": "squared reconstruction error (distortion)", "data": [0.04, 0.172, 0.319, 0.572, 1.097, 1.918, 3.947], "borderColor": "var(--c-accent)", "yAxisID": "y1", "pointRadius": 4, "tension": 0.2 } ] },
  "options": { "scales": { "x": {"type": "logarithmic", "title": {"display": true, "text": "beta (log scale)"}}, "y": {"position": "left", "title": {"display": true, "text": "KL (nats)"}}, "y1": {"position": "right", "grid": {"drawOnChartArea": false}, "title": {"display": true, "text": "squared error (4 dims)"}} } }
}
```

### Posterior collapse {#posterior-collapse}

Sometimes the KL term wins completely: $q_\phi(z\mid x)\approx p(z)$ for every $x$, the code carries no information, and the decoder ignores $z$ and models the average data (a blur for images, a language model for text). This is **posterior collapse**, and you can see it at $\beta=16$ in the chart above and in the demo. It is most common when the decoder is **powerful enough to model the data without the latent** (an autoregressive decoder over pixels or words, as in the VAE language models of Bowman et al., 2016), when the KL weight is too large relative to the reconstruction scale, and early in training, when the encoder is still uninformative.

Standard remedies:

- **KL annealing**: warm up $\beta$ from 0 to 1 over the first epochs (Bowman et al., 2016), so the decoder first learns to use $z$.
- **Free bits** (Kingma et al., 2016): ignore the KL of a latent dimension until it exceeds a threshold, $\sum_j\max(\lambda,\mathrm{KL}_j)$.
- **Weaken the decoder** (smaller receptive field, dropout on inputs) or **strengthen the encoder**, so information in $z$ is useful.
- **Lower-capacity or discrete latents** (VQ-VAE), which sidesteps the issue because there is no KL term to win.

How to detect it: the KL per latent dimension (or the variance of $\mu_j$ across the data) is near zero for most dimensions, and the "active units" count (dimensions with $\mathrm{Var}_x[\mu_j(x)]>\epsilon$) is small. Always log the reconstruction and the KL **separately**, not just their sum.

### VQ-VAE: discrete latents {#vq-vae}

The **VQ-VAE** (van den Oord, Vinyals and Kavukcuoglu, NeurIPS 2017, "Neural Discrete Representation Learning") replaces the continuous Gaussian code by a **discrete** one drawn from a learned **codebook** of $K$ vectors $e_1,\dots,e_K\in\R^D$. The encoder outputs a grid of vectors $z_e(x)$ (for a $64\times64$ image with an 8x downsampling CNN, an $8\times8\times D$ grid), each of which is replaced by its **nearest codebook entry**,

$$
z_q(x)_{ij}=e_{c_{ij}},\qquad c_{ij}=\argmin_k\norm{z_e(x)_{ij}-e_k}_2 ,
$$

and the decoder reconstructs from $z_q$. An image becomes a small grid of integers.

{{figure: week12-vqvae | VQ-VAE. The encoder produces an $8\times8$ grid of $D$-dimensional vectors; each is snapped to its nearest entry in a codebook of $K$ vectors. The decoder reconstructs from the quantized grid, and a second model (for example an autoregressive Transformer) is trained on the grid of indices to enable sampling.}}

The argmin is not differentiable, so the decoder's gradient with respect to $z_q$ is copied directly to $z_e$ (the **straight-through estimator**). The codebook is trained with two auxiliary terms, with $\operatorname{sg}[\cdot]$ denoting stop-gradient:

$$
\mathcal{L}=\underbrace{\log p(x\mid z_q)}_{\text{reconstruction (negated)}}\;+\;\underbrace{\norm{\operatorname{sg}[z_e]-e}^2}_{\text{moves codebook toward encoder outputs}}\;+\;\beta\underbrace{\norm{z_e-\operatorname{sg}[e]}^2}_{\text{commitment: keeps the encoder near the codebook}} ,
$$

with $\beta=0.25$ in the paper. There is no sampling noise and no KL term, since the posterior is deterministic and the prior is uniform over a fixed number of codes; the model sidesteps posterior collapse. To *generate*, you train a separate autoregressive prior over the index grid (PixelCNN in the original paper; later, Transformers) and decode the sampled indices. VQ-VAE-2 (Razavi, van den Oord and Vinyals, NeurIPS 2019) used a hierarchy of such grids. The same tokenizer idea powers the image tokens of text-to-image models such as DALL-E (Ramesh et al., 2021), and the VQGAN variant (Esser, Rombach and Ommer, CVPR 2021) adds a perceptual and adversarial loss for sharper reconstructions.

```python:vector_quantizer.py
import torch, torch.nn as nn, torch.nn.functional as F

class VectorQuantizer(nn.Module):
    def __init__(self, num_codes=512, dim=64, beta=0.25):
        super().__init__()
        self.codebook = nn.Embedding(num_codes, dim)
        self.codebook.weight.data.uniform_(-1 / num_codes, 1 / num_codes)
        self.beta = beta

    def forward(self, z_e):                                    # z_e: (B, D, H, W) encoder output
        B, D, H, W = z_e.shape
        flat = z_e.permute(0, 2, 3, 1).reshape(-1, D)          # (B*H*W, D)
        d2 = (flat.pow(2).sum(1, keepdim=True) - 2 * flat @ self.codebook.weight.T
              + self.codebook.weight.pow(2).sum(1))            # squared distances to every code: (B*H*W, K)
        idx = d2.argmin(dim=1)                                 # (B*H*W,) nearest code per position
        z_q = self.codebook(idx).view(B, H, W, D).permute(0, 3, 1, 2)   # (B, D, H, W)
        # codebook loss moves e toward sg[z_e]; commitment loss moves z_e toward sg[e]
        loss = F.mse_loss(z_q, z_e.detach()) + self.beta * F.mse_loss(z_e, z_q.detach())
        z_q = z_e + (z_q - z_e).detach()                       # straight-through: forward = z_q, backward = identity
        return z_q, loss, idx.view(B, H, W)

vq = VectorQuantizer()
z_q, loss, idx = vq(torch.randn(2, 64, 8, 8))
print(z_q.shape, idx.shape, loss.item())                       # (2, 64, 8, 8) (2, 8, 8)
```

A common failure is **codebook collapse**: only a handful of codes are ever selected. Remedies include initializing the codebook from encoder outputs (k-means), re-initializing dead codes, an exponential-moving-average codebook update, and lowering the code dimension.

## Generative adversarial networks {#gan}

The VAE needs a likelihood, and with a Gaussian or Bernoulli decoder that likelihood treats pixels as independent given $z$, which is a major reason VAE samples are blurry: under squared error the best guess for an uncertain pixel is the average of the possibilities. Goodfellow, Pouget-Abadie, Mirza, Xu, Warde-Farley, Ozair, Courville and Bengio (NeurIPS 2014, "Generative Adversarial Nets") proposed to drop the likelihood and replace it by a *learned* measure of sample quality: a second network that tries to tell real from fake.

### The minimax game {#minimax}

Two networks play against each other. The **generator** $G_\theta:\R^k\to\R^d$ maps noise $z\sim p_z=\mathcal{N}(0,\mathbf{I})$ to a sample $G(z)$, implicitly defining a distribution $p_g$ over $x$. The **discriminator** $D_\psi:\R^d\to(0,1)$ outputs the probability that its input came from the data rather than from $G$. The value function is

$$
\min_G\max_D\;V(D,G)=\E_{x\sim p_\text{data}}\big[\log D(x)\big]+\E_{z\sim p_z}\big[\log\big(1-D(G(z))\big)\big].
$$

The discriminator maximizes $V$: it is a binary classifier with cross-entropy loss, labelling real data 1 and fakes 0. The generator minimizes $V$: it wants $D(G(z))$ to be large. In practice we alternate a few gradient steps on $D$ and one on $G$, each with a minibatch.

{{figure: week12-gan | The GAN game. The generator turns noise into samples; the discriminator scores real and generated samples. The generator is trained through the discriminator: the gradient of $D$'s output with respect to its input flows back through $G(z)$ to the generator's parameters.}}

### The optimal discriminator {#optimal-discriminator}

Why should this game produce a good generator? Analyse it in the idealized setting where $D$ can represent any function and is trained to optimality for a fixed $G$.

:::math Derivation: the optimal discriminator and the Jensen-Shannon connection
For fixed $G$, write the second expectation as an integral over $x$ using $x=G(z)\sim p_g$:

$$
V(D,G)=\int_x\Big[p_\text{data}(x)\log D(x)+p_g(x)\log\big(1-D(x)\big)\Big]dx .
$$

The integrand can be maximized pointwise in $D(x)$. For $a=p_\text{data}(x)$, $b=p_g(x)$, the function $y\mapsto a\log y+b\log(1-y)$ on $(0,1)$ has derivative $a/y-b/(1-y)$, which vanishes at $y=a/(a+b)$ and is concave, so

$$
D^*_G(x)=\frac{p_\text{data}(x)}{p_\text{data}(x)+p_g(x)} .
$$

Substitute back. Let $m=\tfrac12(p_\text{data}+p_g)$, so $p_\text{data}/(p_\text{data}+p_g)=p_\text{data}/(2m)$:

$$
\begin{aligned}
C(G)&=\max_D V(D,G)=\E_{p_\text{data}}\Big[\log\frac{p_\text{data}}{2m}\Big]+\E_{p_g}\Big[\log\frac{p_g}{2m}\Big]\\
&=-\log4+\KL\big(p_\text{data}\,\|\,m\big)+\KL\big(p_g\,\|\,m\big)\\
&=-\log4+2\,\mathrm{JS}\big(p_\text{data}\,\|\,p_g\big),
\end{aligned}
$$

where $\mathrm{JS}(p\|q)=\tfrac12\KL(p\|m)+\tfrac12\KL(q\|m)\in[0,\log2]$ is the Jensen-Shannon divergence. Since $\mathrm{JS}\ge0$ with equality iff $p_g=p_\text{data}$, the global minimum of $C(G)$ is $-\log4$, reached exactly when $p_g=p_\text{data}$. At that point $D^*(x)=\tfrac12$ everywhere: the discriminator cannot do better than guessing.
:::

So, in theory, training the generator against an optimal discriminator minimizes the JS divergence between the model and the data. Three caveats keep this from being a guarantee. The result is about *function space*, while we train finite networks by alternating stochastic gradient steps and the game need not converge; $D$ is never optimal; and, as we are about to see, the JS divergence is a poor training signal when $p_g$ and $p_\text{data}$ barely overlap.

### The non-saturating generator loss {#non-saturating}

Early in training, the generator is poor and the discriminator wins easily: $D(G(z))\approx0$. Write $D=\sigma(a)$ with logit $a$. The minimax generator loss $\log(1-D)=\log(1-\sigma(a))$ has gradient $\partial/\partial a=-\sigma(a)=-D$, which is *near zero* exactly when $D\approx0$: the generator gets no signal when it needs it most. Goodfellow et al. therefore recommended that the generator instead **maximize $\log D(G(z))$**, i.e. minimize $-\log D(G(z))$, whose gradient with respect to the logit is $-(1-D)$, close to $-1$ when $D$ is near 0.

```chart
{
  "type": "line",
  "caption": "Gradient magnitude of the two generator losses with respect to the discriminator logit a on a generated sample, where D(G(z)) = sigmoid(a). The minimax loss log(1 - D) has gradient magnitude D, which vanishes when the discriminator is confident the sample is fake (a very negative); the non-saturating loss -log D has gradient magnitude 1 - D, which stays near 1. Exact formulas, evaluated on a grid.",
  "height": 320,
  "data": { "labels": [-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4], "datasets": [
    { "label": "minimax (saturating): |d log(1-D) / da| = D", "data": [0.0003, 0.0009, 0.0025, 0.0067, 0.018, 0.0474, 0.1192, 0.2689, 0.5, 0.7311, 0.8808, 0.9526, 0.982], "borderColor": "var(--c-red)", "pointRadius": 3, "tension": 0.2 },
    { "label": "non-saturating: |d (-log D) / da| = 1 - D", "data": [0.9997, 0.9991, 0.9975, 0.9933, 0.982, 0.9526, 0.8808, 0.7311, 0.5, 0.2689, 0.1192, 0.0474, 0.018], "borderColor": "var(--c-green)", "pointRadius": 3, "tension": 0.2 } ] },
  "options": { "scales": { "x": {"title": {"display": true, "text": "discriminator logit a on a generated sample (negative = confident it is fake)"}}, "y": {"min": 0, "max": 1.05, "title": {"display": true, "text": "gradient magnitude"}} } }
}
```

The non-saturating loss changes the fixed points of the game (it is no longer a zero-sum game) and no longer corresponds exactly to the JS divergence; Arjovsky and Bottou (ICLR 2017) show that its gradient equals that of $\KL(p_g\|p_\text{data})-2\,\mathrm{JS}(p_g\|p_\text{data})$, the difference of a reverse KL (mode seeking: it punishes implausible samples heavily and dropped modes only mildly) and a JS term with the opposite sign, so it is not a divergence minimization in the strict sense. In practice it is the default.

```python:gan_losses.py
import torch, torch.nn.functional as F

def d_loss(D, real, fake):
    """Discriminator: maximize log D(x) + log(1 - D(G(z))) = minimize BCE with labels 1 (real) and 0 (fake)."""
    real_logits, fake_logits = D(real), D(fake.detach())          # detach: do not update G in the D step
    return (F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
            + F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits)))

def g_loss_nonsaturating(D, fake):
    """Generator: minimize -log D(G(z)), i.e. BCE with the *real* label on the fakes."""
    fake_logits = D(fake)
    return F.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits))

def g_loss_minimax(D, fake):
    """Saturating version, for comparison: minimize log(1 - D(G(z))) = -BCE(fake logits, label 0)."""
    fake_logits = D(fake)
    return -F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
```

### Why GAN training is hard {#instability}

The game is not an optimization problem with a single loss to decrease. Two networks with opposite goals chase each other, and **simultaneous gradient descent on a minimax game can cycle rather than converge**. The classic toy example is $\min_x\max_y xy$: gradient descent-ascent spirals outward around the saddle point $(0,0)$. The losses of $G$ and $D$ therefore do not need to fall, and, unlike a supervised loss, the training curves say little about sample quality: a generator loss that rises can mean the discriminator is improving, or the generator is failing. Typical problems:

- **Vanishing gradients.** If $D$ becomes too good, its output saturates and gives the generator nothing useful (above), and the deeper reason is below.
- **Non-overlapping supports.** Data and samples both lie on low-dimensional manifolds in a huge space, so initially the two distributions have (essentially) disjoint supports. A discriminator can separate them perfectly, and the JS divergence is the constant $\log2$ regardless of how far apart they are (see the chart below), so its gradient is uninformative. Arjovsky and Bottou (2017) analyse this carefully, and it motivates the Wasserstein distance.
- **Mode collapse.** The generator finds a few outputs that the current discriminator likes and keeps emitting them. The discriminator then learns to reject those, the generator jumps to another mode, and the process cycles without ever covering the data.
- **Oscillation and sensitivity.** Small changes in learning rates, normalization, initialization or architecture can decide between success and failure, and results vary widely between random seeds.
- **No convergence criterion.** You have to look at samples or use metrics such as FID (below).

### Mode collapse {#mode-collapse}

Mode collapse has a clean explanation in terms of the objective. The generator's loss only looks at samples $G(z)$ and asks whether they look real, *not* whether the whole data distribution is covered. Producing a single perfect-looking sample every time gets a good score from a discriminator that evaluates each sample **independently** (nothing in $D(x)$ looks at the batch as a whole). If you view the training as minimizing a divergence, the generator's solution is "reverse-KL-like" (mode seeking, the same effect as in the KL demo above). Mitigations include: **minibatch discrimination** and **minibatch standard deviation** features, which let $D$ see statistics of a batch and detect lack of diversity (Salimans et al., 2016; Karras et al., 2018); **unrolled GANs** (Metz et al., 2017), where $G$ looks ahead at $D$'s response; a better divergence (WGAN); conditional labels; and plain regularization of the discriminator. Try to produce it yourself in the demo below.

{{demo: week12-gan-playground | A 1-D GAN trained live: mode collapse, saturation and WGAN | 640}}

## Wasserstein GAN {#wgan}

Consider two unit-variance Gaussians whose means are a distance $d$ apart (a stand-in for a generator that has not yet reached the data). The Jensen-Shannon divergence saturates as soon as the distributions stop overlapping, while the Wasserstein-1 distance keeps growing in proportion to $d$:

```chart
{
  "type": "line",
  "caption": "Jensen-Shannon divergence (nats, numerically integrated) and Wasserstein-1 distance between N(0,1) and N(d,1) as the mean offset d grows. JS saturates at log 2 = 0.693, so its gradient with respect to the position of the generator vanishes, while W1 equals d exactly for equal-variance Gaussians and always points the right way.",
  "height": 320,
  "data": { "labels": [0, 0.5, 1, 2, 3, 4, 6, 8], "datasets": [
    { "label": "JS divergence (nats)", "data": [0.0, 0.0303, 0.1114, 0.3368, 0.5268, 0.6327, 0.6893, 0.6931], "borderColor": "var(--c-red)", "yAxisID": "y", "pointRadius": 3, "tension": 0.2 },
    { "label": "Wasserstein-1 distance", "data": [0, 0.5, 1, 2, 3, 4, 6, 8], "borderColor": "var(--c-green)", "yAxisID": "y1", "pointRadius": 3, "tension": 0.2 } ] },
  "options": { "scales": { "x": {"title": {"display": true, "text": "distance d between the two means"}}, "y": {"position": "left", "min": 0, "max": 0.8, "title": {"display": true, "text": "JS (nats)"}}, "y1": {"position": "right", "min": 0, "grid": {"drawOnChartArea": false}, "title": {"display": true, "text": "W1"}} } }
}
```

For distributions on low-dimensional manifolds that never overlap, the difference is dramatic: JS is exactly $\log2$ for any separation, while a Wasserstein distance (the cost of moving one pile of probability mass onto another, hence "earth mover's distance") decreases smoothly as the generated samples approach the data.

### The Wasserstein distance and the Kantorovich-Rubinstein dual {#kantorovich}

$$
W_1(p,q)=\inf_{\gamma\in\Pi(p,q)}\E_{(x,y)\sim\gamma}\norm{x-y},
$$

where $\Pi(p,q)$ is the set of joint distributions with marginals $p$ and $q$. Computing this directly is intractable, but the **Kantorovich-Rubinstein duality** turns it into a maximization over 1-Lipschitz functions:

$$
W_1(p_\text{data},p_g)=\sup_{\norm{f}_L\le1}\;\E_{x\sim p_\text{data}}[f(x)]-\E_{x\sim p_g}[f(x)] .
$$

Arjovsky, Chintala and Bottou (ICML 2017, "Wasserstein GAN") parameterize $f_w$ with a network called the **critic** (it outputs an unbounded real number and not a probability, so it is not a "discriminator"), constrain it to be (approximately) 1-Lipschitz, and train $G$ to minimize the critic's estimate of $W_1$. The losses are very simple:

$$
\mathcal{L}_\text{critic}=-\Big(\E_{x\sim p_\text{data}}[f_w(x)]-\E_{z}[f_w(G(z))]\Big),\qquad \mathcal{L}_G=-\E_z[f_w(G(z))].
$$

{{figure: week12-wgan-critic | With disjoint supports, a JS-based discriminator saturates (flat, zero gradient for the generator) while a 1-Lipschitz critic provides a slope that tells the generator which way to move.}}

The critic's value is a meaningful quantity: because it estimates a distance, the critic loss correlates with sample quality and can be plotted as a training curve, a rare convenience in the GAN world. Practical consequences of the WGAN framework: train the critic *more* than the generator (5 critic steps per generator step in the paper), use no sigmoid, and avoid momentum-based optimizers for the critic in the original version (RMSProp).

### Enforcing the Lipschitz constraint: clipping, then gradient penalty {#wgan-gp}

The original WGAN enforced the constraint by **clipping** the critic's weights to $[-c,c]$ (with $c=0.01$), which the authors themselves called a crude hack: it biases the critic towards simple functions, and gradients either vanish or explode depending on $c$. **WGAN-GP** (Gulrajani, Ahmed, Arjovsky, Dumoulin and Courville, NeurIPS 2017, "Improved Training of Wasserstein GANs") instead penalizes the critic's gradient norm. A differentiable 1-Lipschitz function has gradient norm at most 1 everywhere, and the optimal critic has gradient norm exactly 1 along straight lines between a real sample and its coupled generated sample, so the penalty is evaluated at random points $\hat x=\varepsilon x+(1-\varepsilon)G(z)$ with $\varepsilon\sim U[0,1]$:

$$
\mathcal{L}_\text{critic}=\E_{z}[f_w(G(z))]-\E_{x}[f_w(x)]+\lambda\,\E_{\hat x}\Big[\big(\norm{\nabla_{\hat x}f_w(\hat x)}_2-1\big)^2\Big],\qquad\lambda=10 .
$$

Because the penalty is computed per sample, **do not use batch normalization in the critic** (it couples samples in a batch); use layer normalization or none. Another way to restrict the Lipschitz constant is **spectral normalization** (Miyato, Kataoka, Koyama and Yoshida, ICLR 2018), which divides each weight matrix by its largest singular value (estimated by power iteration) and is cheap and effective for ordinary GANs too.

```python:wgan_gp.py
import torch

def gradient_penalty(critic, real, fake, lam=10.0):
    """WGAN-GP penalty on interpolates; real, fake: (B, C, H, W) or (B, d)."""
    B = real.size(0)
    shape = [B] + [1] * (real.dim() - 1)
    eps = torch.rand(shape, device=real.device)                      # one epsilon per sample
    x_hat = (eps * real + (1 - eps) * fake.detach()).requires_grad_(True)
    scores = critic(x_hat)                                           # (B,) or (B, 1)
    grads, = torch.autograd.grad(outputs=scores.sum(), inputs=x_hat,
                                 create_graph=True)                  # create_graph: we backprop through the penalty
    grad_norm = grads.reshape(B, -1).norm(2, dim=1)                  # (B,)
    return lam * ((grad_norm - 1) ** 2).mean()

def critic_loss(critic, real, fake):
    return critic(fake.detach()).mean() - critic(real).mean() + gradient_penalty(critic, real, fake)

def generator_loss(critic, fake):
    return -critic(fake).mean()

# check with a toy critic
critic = torch.nn.Sequential(torch.nn.Linear(2, 16), torch.nn.LeakyReLU(0.2), torch.nn.Linear(16, 1))
real, fake = torch.randn(32, 2) + 3, torch.randn(32, 2)
print(critic_loss(critic, real, fake).item())
```

## DCGAN: architecture rules that made GANs trainable {#dcgan}

The original GAN used fully connected networks on MNIST and small images. **DCGAN** (Radford, Metz and Chintala, ICLR 2016, "Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks") listed a handful of architectural guidelines that made convolutional GANs train reliably, and they remain a template:

- Replace pooling by **strided convolutions** in the discriminator and **transposed convolutions** (fractionally strided) in the generator, so the networks learn their own up- and down-sampling.
- Use **batch normalization** in both networks (except the generator's output layer and the discriminator's input layer).
- Remove fully connected hidden layers in deeper architectures.
- Generator: **ReLU** activations, **tanh** at the output (images scaled to $[-1,1]$). Discriminator: **LeakyReLU** (slope 0.2).
- Optimizer: Adam with learning rate $2\times10^{-4}$ and $\beta_1=0.5$ (the default $\beta_1=0.9$ makes training oscillate).
- Initialize weights from $\mathcal{N}(0,0.02^2)$.

The generator for $64\times64$ images projects $z\in\R^{100}$ to a $4\times4\times512$ tensor and upsamples with four stride-2 transposed convolutions ($4\to8\to16\to32\to64$) while halving channels ($512\to256\to128\to64\to3$). The paper also showed that the latent space is meaningful: **vector arithmetic** on faces (the average code of "smiling woman" minus "neutral woman" plus "neutral man" decodes to a smiling man), and that interpolating $z$ yields smooth transitions rather than fading, evidence that the generator was not memorizing. For MNIST we use a $28\times28$ variant in the lab below.

## Conditional GANs and image-to-image translation {#conditional}

### Conditional GANs {#cgan}

Unconditional GANs give you a random sample. A **conditional GAN** (Mirza and Osindero, 2014) feeds a label or other conditioning signal $y$ to both networks, modelling $p(x\mid y)$:

$$
\min_G\max_D\;\E_{x,y}[\log D(x\mid y)]+\E_{z,y}[\log(1-D(G(z\mid y)\mid y))].
$$

For class labels the usual implementation concatenates a one-hot (or learned embedding) of $y$ to $z$ for $G$ and to the input (or a feature map) for $D$. Better-performing variants use **projection discriminators** (Miyato and Koyama, ICLR 2018), conditional batch normalization and class-conditional spectral normalization, as in BigGAN (Brock, Donahue and Simonyan, ICLR 2019), which scaled class-conditional GANs to ImageNet at $512\times512$ and introduced the **truncation trick**: sample $z$ from a truncated normal and trade diversity for fidelity at test time. **pix2pix** (Isola, Zhu, Zhou and Efros, CVPR 2017) conditions on an *image* (edges to photo, day to night, labels to street scene): the generator is a U-Net, the discriminator judges local $70\times70$ patches ("PatchGAN"), and the loss adds an $L_1$ term between the output and the paired target, so the adversarial loss only has to supply sharp high-frequency detail.

### CycleGAN: unpaired translation {#cyclegan}

Paired data is rare (nobody has a photograph of the same scene as a horse and as a zebra). **CycleGAN** (Zhu, Park, Isola and Efros, ICCV 2017, "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks") learns two mappings $G:X\to Y$ and $F:Y\to X$ between two *unpaired* image collections, with two discriminators $D_Y$ and $D_X$. An adversarial loss on each mapping makes outputs look like the target domain, but that alone is under-constrained (any mapping that produces plausible zebras would do). The key idea is **cycle consistency**: translating and then translating back should reproduce the input.

{{figure: week12-cyclegan | CycleGAN. Two generators translate between domains and two discriminators judge realism in each. The cycle-consistency loss ties the two mappings together, which gives the unpaired problem enough structure.}}

$$
\mathcal{L}=\mathcal{L}_\text{GAN}(G,D_Y)+\mathcal{L}_\text{GAN}(F,D_X)+\lambda\Big(\E_x\norm{F(G(x))-x}_1+\E_y\norm{G(F(y))-y}_1\Big),\quad\lambda=10 .
$$

CycleGAN works best when the change is textural or local (horse to zebra, summer to winter, photo to painting style) and fails when geometry must change (cat to dog) or when the cycle loss lets the generator hide information in imperceptible high-frequency patterns (steganography) in order to "cheat" the reconstruction. The paper also uses an identity loss for colour preservation in some tasks, and a replay buffer of previously generated images for the discriminators.

## StyleGAN {#stylegan}

**ProGAN** (Karras, Aila, Laine and Lehtinen, ICLR 2018, "Progressive Growing of GANs") trained a GAN by starting at $4\times4$ and adding layers that raise the resolution during training, which stabilized the generation of $1024\times1024$ faces. **StyleGAN** (Karras, Laine and Aila, CVPR 2019, "A Style-Based Generator Architecture for Generative Adversarial Networks") rethought the *generator*:

1. A **mapping network** (an 8-layer MLP) transforms the noise $z\in\mathcal{Z}$ into an intermediate latent $w\in\mathcal{W}$. $\mathcal{Z}$ must follow a fixed distribution (a Gaussian), but the data distribution of factors such as "age" or "hair length" is not Gaussian; $\mathcal{W}$ is free to be warped to make factors more linear and less entangled.
2. The synthesis network starts from a **learned constant** $4\times4\times512$ tensor (not from $z$). At each resolution, $w$ is turned by a learned affine layer into a per-channel scale and bias that **modulate** the feature maps, originally by adaptive instance normalization (AdaIN) and in StyleGAN2 by a weight demodulation that removes the "droplet" artefacts AdaIN created (Karras et al., CVPR 2020).
3. **Noise inputs** (per-pixel Gaussian noise, scaled by learned per-channel weights) are added after each convolution to generate stochastic detail such as hair strands and pores that need not be encoded in $w$.

{{figure: week12-stylegan | The StyleGAN generator. A mapping MLP turns $z$ into $w$, which controls every block through learned affine styles; noise adds stochastic detail. Using $w_1$ for coarse blocks and $w_2$ for fine blocks mixes attributes of two faces.}}

Because each block receives its own style, **style mixing** works: use $w_1$ for the coarse blocks (pose, face shape, hair style) and $w_2$ for the fine blocks (colour, micro-texture), and you obtain a face that combines the two. A **truncation trick** in $\mathcal{W}$ (pulling $w$ toward its mean) trades diversity for quality. Later versions fixed artefacts (StyleGAN2), removed aliasing (StyleGAN3, Karras et al., NeurIPS 2021) and made training work on small datasets with adaptive discriminator augmentation (Karras et al., NeurIPS 2020). StyleGAN-type latent spaces remain popular for image editing by moving a code along a learned direction and for GAN inversion (finding the $w$ for a real photo).

## Evaluating generative models {#evaluation}

Likelihood-based models report **bits per dimension** or negative log-likelihood on held-out data. For implicit models there is no likelihood, and for GANs even a VAE-style bound does not help. Evaluation therefore uses *sample-based* metrics, and no single number captures the three things you want: **fidelity** (samples look real), **diversity** (they cover the data) and **novelty** (they are not copies of the training set). Human evaluation remains the gold standard and is expensive.

### Inception Score {#inception-score}

The **Inception Score** (Salimans, Goodfellow, Zaremba, Cheung, Radford and Chen, NeurIPS 2016, "Improved Techniques for Training GANs") feeds generated images to an ImageNet-trained Inception network and looks at the class posterior $p(y\mid x)$. A good generator should produce images that are **recognizable** (each $p(y\mid x)$ has low entropy) and **diverse** (the marginal $p(y)=\E_xp(y\mid x)$ has high entropy). Both are captured by

$$
\mathrm{IS}=\exp\Big(\E_{x\sim p_g}\KL\big(p(y\mid x)\,\|\,p(y)\big)\Big)=\exp\big(H(y)-\E_xH(y\mid x)\big).
$$

It ranges from 1 (all outputs the same, or unrecognizable) to the number of classes (1000 for ImageNet; confident and perfectly spread). **Weaknesses:** it never looks at real data, so a generator can obtain a high score while missing the real distribution; it depends on the ImageNet classes (meaningless for faces or medical images); it is blind to intra-class diversity (a generator that produces one perfect image per class gets the maximum); and it is sensitive to the network implementation and image resizing.

### Frechet Inception Distance {#fid}

The **FID** (Heusel, Ramsauer, Unterthiner, Nessler and Hochreiter, NeurIPS 2017, in the paper that also proposed the two time-scale update rule for GANs) compares real and generated images in the feature space of the Inception network (the 2048-dimensional pooling layer). Fit a Gaussian $\mathcal{N}(\mu_r,\Sigma_r)$ to the real features and $\mathcal{N}(\mu_g,\Sigma_g)$ to the generated features and compute the **Frechet (2-Wasserstein) distance** between the Gaussians:

$$
\mathrm{FID}=\norm{\mu_r-\mu_g}_2^2+\operatorname{Tr}\Big(\Sigma_r+\Sigma_g-2\big(\Sigma_r\Sigma_g\big)^{1/2}\Big).
$$

Lower is better, with 0 meaning the two Gaussians agree. The first term compares *means* (is the overall content right?) and the second compares *covariances* (is the spread and diversity right?), so FID reacts to both quality and mode dropping. It is the standard benchmark (CIFAR-10 FIDs fell from tens to low single digits during the GAN era), but it has limits: it assumes the features are Gaussian, is **biased by sample size** (always use at least 10,000, ideally 50,000, and report that number), depends on the exact Inception weights and image preprocessing, and can be hacked or misjudge images far from ImageNet. Each number must be paired with the protocol.

```chart
{
  "type": "line",
  "caption": "FID and Inception Score on a synthetic benchmark (illustrative; computed with NumPy): real data are 10 Gaussian clusters in a 16-D feature space, and the 'generator' perfectly samples only the first k clusters. A classifier posterior over the 10 clusters plays the role of Inception. Both metrics detect mode dropping: IS grows as the number of covered modes k (it equals k up to noise) and FID falls from 171 to 0.",
  "height": 340,
  "data": { "labels": [1, 2, 3, 5, 7, 10], "datasets": [
    { "label": "FID (lower is better)", "data": [171.0, 113.4, 73.2, 44.1, 22.9, 0.2], "borderColor": "var(--c-red)", "yAxisID": "y", "pointRadius": 4, "tension": 0.2 },
    { "label": "Inception Score (higher is better)", "data": [1.0, 2.0, 3.0, 5.0, 7.0, 9.99], "borderColor": "var(--c-blue)", "yAxisID": "y1", "pointRadius": 4, "tension": 0.2 } ] },
  "options": { "scales": { "x": {"title": {"display": true, "text": "number of modes the generator covers (out of 10)"}}, "y": {"position": "left", "min": 0, "title": {"display": true, "text": "FID"}}, "y1": {"position": "right", "min": 0, "grid": {"drawOnChartArea": false}, "title": {"display": true, "text": "Inception Score"}} } }
}
```

```python:fid.py
import numpy as np
from scipy import linalg

def frechet_distance(feats_r, feats_g):
    """feats_*: (N, d) arrays of Inception pool features (d = 2048 for the standard FID)."""
    mu_r, mu_g = feats_r.mean(0), feats_g.mean(0)
    cov_r, cov_g = np.cov(feats_r, rowvar=False), np.cov(feats_g, rowvar=False)
    covmean = linalg.sqrtm(cov_r @ cov_g)                     # matrix square root (can be complex numerically)
    covmean = np.real(covmean)
    return float(((mu_r - mu_g) ** 2).sum() + np.trace(cov_r) + np.trace(cov_g) - 2 * np.trace(covmean))

def inception_score(probs, eps=1e-12):
    """probs: (N, C) softmax outputs p(y|x) of the classifier on generated images."""
    p_y = probs.mean(0, keepdims=True)
    kl = (probs * (np.log(probs + eps) - np.log(p_y + eps))).sum(1)
    return float(np.exp(kl.mean()))

rng = np.random.default_rng(0)
a, b = rng.normal(size=(5000, 16)), rng.normal(size=(5000, 16)) + 0.5
print(frechet_distance(a, a[::-1]), frechet_distance(a, b))   # about 0 and about 16 * 0.25 = 4
```

### Precision and recall for distributions {#precision-recall}

FID collapses fidelity and diversity into one number, and a high FID does not say which one failed. Sajjadi, Bachem, Lucic, Bousquet and Gelly (NeurIPS 2018) and, in the form now commonly used, Kynkaanniemi, Karras, Laine, Lehtinen and Aila (NeurIPS 2019, "Improved Precision and Recall Metric for Assessing Generative Models") split it in two. Embed real and generated samples in a feature space and approximate each distribution's *support* by the union of balls around its samples, with each ball's radius set by the distance to its $k$-th nearest neighbour (typically $k=3$):

- **Precision** is the fraction of generated samples that fall inside the real manifold: a fidelity measure ("are the samples realistic?").
- **Recall** is the fraction of real samples that fall inside the generated manifold: a coverage measure ("does the generator reach everything?").

A mode-collapsed GAN has high precision and low recall; a blurry VAE tends to have lower precision but good recall. This pair makes the quality-diversity trade-off explicit (and shows the effect of the truncation trick, which raises precision and lowers recall). Other metrics worth knowing: **KID** (kernel inception distance, an unbiased alternative), **CLIP-based scores** ([Week 11](week-11.html#clip)) such as CLIPScore for text-conditional generation, **LPIPS** for perceptual diversity and similarity, and **nearest-neighbour checks** against the training set for memorization.

## Lab: a VAE and a DCGAN on MNIST {#lab}

In this lab you will train both models on MNIST and compare them. Everything runs in a few minutes on a free Colab GPU (the VAE is fast even on a CPU). The goals: (1) a working VAE, with reconstruction, sampling and latent **interpolation**; (2) a working DCGAN; (3) a side-by-side comparison of samples and interpolations.

### Part 1: data and the VAE {#lab-vae}

```python:lab_vae_train.py
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
tf = transforms.ToTensor()                                           # pixels in [0, 1]
train = datasets.MNIST("data", train=True, download=True, transform=tf)
loader = DataLoader(train, batch_size=128, shuffle=True, drop_last=True, num_workers=2)

class ConvVAE(nn.Module):
    def __init__(self, k=16):
        super().__init__()
        self.enc = nn.Sequential(                                    # (B, 1, 28, 28)
            nn.Conv2d(1, 32, 4, 2, 1), nn.ReLU(),                    # (B, 32, 14, 14)
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),                   # (B, 64, 7, 7)
            nn.Flatten())                                            # (B, 3136)
        self.mu, self.logvar = nn.Linear(64 * 7 * 7, k), nn.Linear(64 * 7 * 7, k)
        self.fc = nn.Linear(k, 64 * 7 * 7)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),          # (B, 32, 14, 14)
            nn.ConvTranspose2d(32, 1, 4, 2, 1))                      # (B, 1, 28, 28) logits

    def encode(self, x):
        h = self.enc(x)
        return self.mu(h), self.logvar(h)

    def decode(self, z):
        return self.dec(F.relu(self.fc(z)).view(-1, 64, 7, 7))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)      # reparameterization
        return self.decode(z), mu, logvar

vae = ConvVAE().to(device)
opt = torch.optim.Adam(vae.parameters(), lr=1e-3)
beta, epochs = 1.0, 10

for epoch in range(epochs):
    tot_rec = tot_kl = 0.0
    kl_weight = min(1.0, (epoch + 1) / 3) * beta                      # KL warm-up over the first 3 epochs
    for x, _ in loader:
        x = x.to(device)
        logits, mu, logvar = vae(x)
        rec = F.binary_cross_entropy_with_logits(logits, x, reduction="sum") / x.size(0)    # per image
        kl = (0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar).sum()) / x.size(0)              # per image
        loss = rec + kl_weight * kl
        opt.zero_grad(); loss.backward(); opt.step()
        tot_rec += rec.item(); tot_kl += kl.item()
    n = len(loader)
    print(f"epoch {epoch+1}: recon {tot_rec/n:.1f}  KL {tot_kl/n:.1f}  (-ELBO {(tot_rec+tot_kl)/n:.1f})")
```

```text
epoch 1: recon <number>  KL <number>  (-ELBO <number>)
...
epoch 10: recon <number>  KL <number>  (-ELBO <number>)
```

As a rough guide, a convolutional VAE like this one typically ends up with a negative ELBO of a little over 100 nats per image on MNIST (a rule of thumb, not a benchmark; it varies with the seed and the latent size). Always check *two* things: the reconstruction and the KL separately, and the number of active latent dimensions.

### Part 2: sampling, reconstruction and latent interpolation {#lab-interp}

```python:lab_vae_eval.py
@torch.no_grad()
def active_units(vae, loader, thresh=0.01):
    mus = torch.cat([vae.encode(x.to(device))[0].cpu() for x, _ in loader])
    return (mus.var(0) > thresh).sum().item(), mus.var(0)

n_active, var_per_dim = active_units(vae, loader)
print("active latent dimensions:", n_active, "of", var_per_dim.numel())

@torch.no_grad()
def sample_vae(n=64):
    z = torch.randn(n, 16, device=device)                              # draw from the PRIOR
    return torch.sigmoid(vae.decode(z))

@torch.no_grad()
def interpolate(vae, x_a, x_b, steps=10, spherical=True):
    """Decode points on a path between the codes of two images (use the posterior means)."""
    za, zb = vae.encode(x_a)[0], vae.encode(x_b)[0]                    # (1, k) each
    out = []
    for t in torch.linspace(0, 1, steps):
        if spherical:                                                  # slerp keeps the norm closer to a Gaussian shell
            cos = F.cosine_similarity(za, zb).clamp(-0.9999, 0.9999)
            om = torch.acos(cos)
            z = (torch.sin((1 - t) * om) * za + torch.sin(t * om) * zb) / torch.sin(om)
        else:
            z = (1 - t) * za + t * zb                                  # plain linear interpolation
        out.append(torch.sigmoid(vae.decode(z)))
    return torch.cat(out)                                              # (steps, 1, 28, 28)

x, y = next(iter(loader))
a = x[y == 3][:1].to(device); b = x[y == 8][:1].to(device)
path = interpolate(vae, a, b)
save_image(make_grid(path, nrow=10), "vae_interpolation.png")
save_image(make_grid(sample_vae(64), nrow=8), "vae_samples.png")
```

Look at the interpolation: a good VAE morphs the 3 into the 8 smoothly through plausible digits, since the decoder has been trained on neighbourhoods and the latent space is filled. Compare it with linear interpolation of the *pixels* (a cross-fade of two superimposed digits) and with a plain autoencoder's code space, where intermediate points often decode to garbage. Then train again with $\beta\in\{0.1,1,4,10\}$ and compare samples, the KL, and the number of active units.

### Part 3: the DCGAN {#lab-dcgan}

```python:lab_dcgan_train.py
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])   # to [-1, 1] to match tanh
loader = torch.utils.data.DataLoader(datasets.MNIST("data", train=True, download=True, transform=tf),
                                     batch_size=128, shuffle=True, drop_last=True, num_workers=2)
nz = 100

class Generator(nn.Module):
    def __init__(self, nz=100, ngf=64):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(nz, ngf * 4 * 7 * 7), nn.BatchNorm1d(ngf * 4 * 7 * 7), nn.ReLU(True))
        self.net = nn.Sequential(                                       # (B, 256, 7, 7)
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf * 2), nn.ReLU(True),   # (B, 128, 14, 14)
            nn.ConvTranspose2d(ngf * 2, 1, 4, 2, 1), nn.Tanh())         # (B, 1, 28, 28) in [-1, 1]
    def forward(self, z):
        return self.net(self.fc(z).view(z.size(0), -1, 7, 7))

class Discriminator(nn.Module):
    def __init__(self, ndf=64):
        super().__init__()
        self.net = nn.Sequential(                                       # (B, 1, 28, 28)
            nn.Conv2d(1, ndf, 4, 2, 1), nn.LeakyReLU(0.2, True),        # (B, 64, 14, 14)
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 2), nn.LeakyReLU(0.2, True),   # (B, 128, 7, 7)
            nn.Flatten(), nn.Linear(ndf * 2 * 7 * 7, 1))                # (B, 1) a logit (the sigmoid lives in the loss)
    def forward(self, x):
        return self.net(x).squeeze(1)

def weights_init(m):                                                    # DCGAN initialization N(0, 0.02)
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None: nn.init.zeros_(m.bias)

G, D = Generator(nz).to(device), Discriminator().to(device)
G.apply(weights_init); D.apply(weights_init)
optG = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
optD = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
bce = nn.BCEWithLogitsLoss()
fixed_z = torch.randn(64, nz, device=device)                            # fixed noise: watch the SAME codes improve

for epoch in range(15):
    for real, _ in loader:
        real = real.to(device); B = real.size(0)
        # ---- discriminator step: real -> 1, fake -> 0
        fake = G(torch.randn(B, nz, device=device))
        lossD = bce(D(real), torch.ones(B, device=device)) + bce(D(fake.detach()), torch.zeros(B, device=device))
        optD.zero_grad(); lossD.backward(); optD.step()
        # ---- generator step (non-saturating): make D say 1 on fakes
        lossG = bce(D(fake), torch.ones(B, device=device))
        optG.zero_grad(); lossG.backward(); optG.step()
    with torch.no_grad():
        G.eval(); grid = make_grid(G(fixed_z) * 0.5 + 0.5, nrow=8); G.train()
    save_image(grid, f"dcgan_epoch{epoch+1:02d}.png")
    print(f"epoch {epoch+1}: lossD {lossD.item():.3f}  lossG {lossG.item():.3f}")
```

Typical behaviour: the discriminator loss hovers around 1.0 to 1.4 (about $2\log2=1.39$ when it cannot tell real from fake), the generator loss around 0.7 to 2, and digits appear after 2 to 3 epochs. Neither loss decreases monotonically: that is **normal**, so judge by the saved image grids from the *fixed* noise, not by the curves.

### Part 4: latent interpolation in the GAN, and comparison {#lab-compare}

```python:lab_dcgan_interp.py
@torch.no_grad()
def slerp(a, b, t):
    """Spherical interpolation between two latent vectors a, b of shape (nz,)."""
    a_n, b_n = a / a.norm(), b / b.norm()
    om = torch.acos((a_n * b_n).sum().clamp(-0.9999, 0.9999))
    return (torch.sin((1 - t) * om) * a + torch.sin(t * om) * b) / torch.sin(om)

G.eval()
za, zb = torch.randn(nz, device=device), torch.randn(nz, device=device)
zs = torch.stack([slerp(za, zb, t) for t in torch.linspace(0, 1, 10)])       # (10, nz)
with torch.no_grad():
    imgs = G(zs) * 0.5 + 0.5                                                  # (10, 1, 28, 28) in [0, 1]
save_image(make_grid(imgs, nrow=10), "dcgan_interpolation.png")

# Mode coverage check: classify 5000 samples with a small pretrained MNIST classifier and look at the class histogram.
# A collapsed generator will leave several digit classes with almost no mass.
```

To compare the two models, produce 64 samples each and look at them side by side: the VAE's digits are smooth and slightly blurry but cover every class, while the DCGAN's are crisper but the class frequencies may be uneven. Quantify this: train a small MNIST classifier, compute the histogram of predicted classes over 5,000 samples from each model, and compute the Inception-score analogue (with this classifier in place of Inception) and the entropy of the class histogram.

## Common mistakes and debugging tips {#mistakes}

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| VAE samples are blurry, KL near 0 for every latent | Posterior collapse (KL weight too high relative to the reconstruction scale, e.g. averaging the pixel loss but summing the KL) | Sum the reconstruction over pixels, KL warm-up, free bits, lower $\beta$; log KL per dimension |
| VAE reconstructions are sharp but samples from the prior are garbage | KL too weak ($\beta\ll1$) so the aggregate posterior does not match the prior | Increase $\beta$, or fit a second-stage prior (a GMM or a small flow) on the codes |
| `nan` loss in a VAE | `exp(logvar)` overflow, or a log of 0 in a probability decoder | Clamp `logvar` to $[-10,10]$, use `binary_cross_entropy_with_logits`, lower the learning rate |
| Reconstruction loss does not decrease with a Gaussian decoder | Pixel scaling mismatched with the decoder variance | Scale data to $[0,1]$ or $[-1,1]$ and set the variance (equivalently $\beta$) deliberately |
| GAN: discriminator loss goes to 0, generator loss rises | The discriminator is overpowering the generator (saturation) | Use the non-saturating loss, smaller discriminator or learning rate, add noise/augmentation to the discriminator's inputs, spectral norm, label smoothing (0.9 for real) |
| GAN: all samples look the same (mode collapse) | The generator exploits the discriminator's per-sample judgement | Minibatch standard deviation, WGAN-GP, unrolling, larger and more diverse batches, lower learning rate |
| GAN: samples are noise after 50 epochs, losses are flat near 0.69 | The discriminator is too weak, or a bug (not detaching, wrong labels, tanh missing) | Verify $D$ can classify real vs fake noise on its own; check data range matches the generator's output range |
| `RuntimeError: Trying to backward through the graph a second time` | The fake batch is reused in both steps without `detach()` | Use `fake.detach()` in the D step; compute the G loss from `D(fake)` after the D update |
| WGAN-GP training diverges | Batch norm in the critic, or the penalty is computed on the wrong points | Use LayerNorm or none, sample $\hat x$ as per-sample interpolates, $\lambda=10$, `create_graph=True` |
| FID worse than expected despite good samples | Few samples, different image preprocessing or resizing, or the wrong Inception weights | Use 10k to 50k samples, the standard FID library and the same resize and dtype for real and fake |
| Interpolations pass through ugly images | Linear interpolation leaves the Gaussian shell, or a plain AE has no latent structure | Use slerp, use the posterior mean, prefer a VAE or $\mathcal{W}$-space in a StyleGAN |

## Exercises {#exercises}

:::exercise Exercise 1: KL by hand
Compute $\KL(\mathcal{N}(\mu,\sigma^2)\|\mathcal{N}(0,1))$ for $(\mu,\sigma)=(0,1)$, $(1,1)$, $(0,0.5)$ and $(0,2)$. Which of $\mu\neq0$ or $\sigma\neq1$ is penalized more, and in which direction is the variance penalty asymmetric?

:::details Solution
Use $\tfrac12(\mu^2+\sigma^2-1-\log\sigma^2)$. $(0,1)$: $0$. $(1,1)$: $\tfrac12(1)=0.5$. $(0,0.5)$: $\tfrac12(0.25-1-\log0.25)=\tfrac12(-0.75+1.386)=0.318$. $(0,2)$: $\tfrac12(4-1-\log4)=\tfrac12(3-1.386)=0.807$. Shrinking the variance to $0.25$ costs less than doubling the standard deviation costs, because the $-\log\sigma^2$ term grows slowly while $\sigma^2$ grows quickly: **too-large variance is penalized more than too-small by the same factor**. A mean of 1 costs $0.5$ nats.
:::
:::

:::exercise Exercise 2: ELBO as a bound (derivation)
Show that $\log p_\theta(x)-\mathcal{L}(\theta,\phi;x)=\KL(q_\phi(z\mid x)\|p_\theta(z\mid x))$ and deduce that if the encoder family can represent the true posterior exactly, then maximizing the ELBO over $\phi$ makes the bound tight. Why can the gap not be evaluated in a real VAE?

:::details Solution
Write $p_\theta(x)=p_\theta(x,z)/p_\theta(z\mid x)$, take $\log$ and the expectation under $q_\phi(z\mid x)$ (the left side does not depend on $z$): $\log p_\theta(x)=\E_q[\log p_\theta(x,z)-\log q]+\E_q[\log q-\log p_\theta(z\mid x)]=\mathcal{L}+\KL(q\|p_\theta(z\mid x))$. For fixed $\theta$ the left side is constant, so maximizing $\mathcal{L}$ over $\phi$ minimizes the KL; if $q$ can equal the true posterior the KL is 0 and $\mathcal{L}=\log p_\theta(x)$. The gap needs $p_\theta(z\mid x)=p_\theta(x\mid z)p(z)/p_\theta(x)$, which contains the intractable marginal $p_\theta(x)$ we are trying to avoid computing.
:::
:::

:::exercise Exercise 3: reparameterization (conceptual and short coding)
(a) Why can't we backpropagate through $z\sim\mathcal{N}(\mu,\sigma^2)$ written as a sampling operation? (b) Show that $\partial z/\partial\mu=1$ and $\partial z/\partial\sigma=\epsilon$ for $z=\mu+\sigma\epsilon$. (c) Check numerically, with PyTorch or NumPy, that the reparameterized gradient of $\E[z^2]$ with respect to $\mu$ matches the analytic value $2\mu$.

:::details Solution
(a) A sample is a draw from a distribution that depends on the parameters, but the draw itself is a random number: the map from the parameters to the value has no defined derivative, so the chain rule has nothing to differentiate. (b) Directly from $z=\mu+\sigma\epsilon$ with $\epsilon$ independent of $(\mu,\sigma)$. (c) $\E[z^2]=\mu^2+\sigma^2$, so $\partial/\partial\mu=2\mu$ and $\partial/\partial\sigma=2\sigma$. Estimate with samples:
```python
import numpy as np
rng = np.random.default_rng(0)
mu, sigma, n = 1.5, 0.8, 1_000_000
eps = rng.standard_normal(n)
z = mu + sigma * eps
print("d/dmu    :", (2 * z).mean(), "vs", 2 * mu)             # dz/dmu = 1  -> E[2 z * 1]
print("d/dsigma :", (2 * z * eps).mean(), "vs", 2 * sigma)    # dz/dsigma = eps -> E[2 z * eps]
```
Both estimates agree with the analytic values to about three decimals.
:::
:::

:::exercise Exercise 4: where does the KL weight hide?
A VAE for 28x28 images uses a Gaussian decoder $\mathcal{N}(\hat x,s^2\mathbf{I})$ and the loss $\norm{x-\hat x}^2+\KL$ (plain squared error, KL weight 1). Show that this equals the negative ELBO (up to a constant) of a VAE with $\beta$-weighting and a specific decoder variance, and compute that variance. What happens to the sharpness of the samples when $s^2$ increases?

:::details Solution
The Gaussian negative log-likelihood is $\frac{1}{2s^2}\norm{x-\hat x}^2+\frac d2\log(2\pi s^2)$. So the negative ELBO is $\frac{1}{2s^2}\norm{x-\hat x}^2+\KL+\text{const}$. Multiplying by $2s^2$ (which does not change the minimizer) gives $\norm{x-\hat x}^2+2s^2\KL$, so the loss $\norm{x-\hat x}^2+\KL$ corresponds to $2s^2=1$, i.e. $s^2=0.5$, equivalently a $\beta$-VAE with $\beta=2s^2$ relative to the squared error. A decoder variance of $0.5$ on pixels in $[0,1]$ is huge, a strong KL weight: expect blurry samples and many collapsed units. Increasing $s^2$ further increases the effective $\beta$ and the blur; decreasing it gives sharper but less regular codes.
:::
:::

:::exercise Exercise 5: coding a VAE on 2-D data (challenging)
Implement the VAE from the lab for a 2-D toy dataset (a ring or two moons from scikit-learn) with a 2-D latent space. Plot (i) the encoder means $\mu(x)$ coloured by the angle on the ring, (ii) a grid of decoded latent points. Train with $\beta=0.1$, $1$ and $5$ and describe the three pictures.

:::details Solution
```python
import torch, torch.nn as nn, numpy as np
from sklearn.datasets import make_moons

X = torch.tensor(make_moons(4000, noise=0.05, random_state=0)[0], dtype=torch.float32)
class V(nn.Module):
    def __init__(s, k=2, h=64):
        super().__init__()
        s.e = nn.Sequential(nn.Linear(2, h), nn.Tanh(), nn.Linear(h, 2 * k))
        s.d = nn.Sequential(nn.Linear(k, h), nn.Tanh(), nn.Linear(h, 2))
    def forward(s, x):
        mu, lv = s.e(x).chunk(2, dim=1)
        z = mu + torch.randn_like(mu) * (0.5 * lv).exp()
        return s.d(z), mu, lv

def train(beta, steps=3000, s2=0.05):
    m = V(); opt = torch.optim.Adam(m.parameters(), 2e-3)
    for _ in range(steps):
        x = X[torch.randint(0, len(X), (256,))]
        xh, mu, lv = m(x)
        rec = ((x - xh) ** 2).sum(1) / (2 * s2)
        kl = 0.5 * (mu ** 2 + lv.exp() - 1 - lv).sum(1)
        loss = (rec + beta * kl).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return m, rec.mean().item(), kl.mean().item()

for beta in [0.1, 1.0, 5.0]:
    m, rec, kl = train(beta)
    print(beta, round(rec, 2), round(kl, 2))
```
Expected picture: with $\beta=0.1$ the encoder means spread over a wide area, KL is large, reconstructions are accurate, but prior samples that fall between codes land off the moons. With $\beta=1$ the aggregate posterior roughly fills a unit Gaussian and decoding the grid reproduces both moons; the shape is a smooth deformation of the plane. With $\beta=5$ the code is squeezed, the two moons overlap in latent space, and decoded points blur toward the data mean (the KL is small, reconstructions poor).
:::
:::

:::exercise Exercise 6: the optimal discriminator in numbers
Let $p_\text{data}=\mathcal{N}(0,1)$ and $p_g=\mathcal{N}(2,1)$. Compute $D^*(x)$ analytically as a function of $x$, then (numerically) $C(G)=-\log4+2\,\mathrm{JS}$. What is $D^*(1)$, and what is the value of the game when $p_g=p_\text{data}$?

:::details Solution
$D^*(x)=\dfrac{e^{-x^2/2}}{e^{-x^2/2}+e^{-(x-2)^2/2}}=\dfrac{1}{1+e^{2x-2}}=\sigma(2-2x)$, since $(x-2)^2-x^2=4-4x$ and half of that is $2-2x$. So $D^*(1)=\sigma(0)=\tfrac12$: the point $x=1$ is equidistant from the two means. Numerically $\mathrm{JS}\approx0.337$ (see the chart), so $C(G)=-1.386+0.674=-0.712$, above the optimum $-\log4=-1.386$. When $p_g=p_\text{data}$, $D^*\equiv\tfrac12$ and $V=-\log4$.
:::
:::

:::exercise Exercise 7: JS is blind to distance (derivation)
Let $p$ be a point mass at $0$ and $q_\theta$ a point mass at $\theta\neq0$. Compute $\mathrm{JS}(p\|q_\theta)$, $\KL(p\|q_\theta)$ and $W_1(p,q_\theta)$. What does this say about the gradient of each with respect to $\theta$, and why does it matter for GANs?

:::details Solution
The two distributions have disjoint supports. The mixture $m=\tfrac12(p+q_\theta)$ puts mass $\tfrac12$ on each point, so $\KL(p\|m)=\log\frac{1}{1/2}=\log2$, the same for $q_\theta$, and $\mathrm{JS}=\log2$ for every $\theta\neq0$: constant, zero gradient. $\KL(p\|q_\theta)=+\infty$ (the density ratio is undefined on the support of $p$). $W_1=|\theta|$, with gradient $\mathrm{sign}(\theta)$: it always points toward $p$. Real data and generated images both concentrate near low-dimensional manifolds, which generically do not overlap early in training, so the GAN's JS-based signal is flat and the generator learns nothing from a perfect discriminator. The Wasserstein critic does not have this problem.
:::
:::

:::exercise Exercise 8: gradient descent-ascent cycles (coding, challenging)
Simulate simultaneous gradient descent-ascent on $V(x,y)=xy$ (minimize over $x$, maximize over $y$) with step size $0.1$ from $(1,1)$. Show that the iterates spiral away from the equilibrium $(0,0)$, then test (a) alternating updates, (b) the optimistic update $x\leftarrow x-\eta(2g_t-g_{t-1})$, and describe what happens. What does this tell you about GAN training?

:::details Solution
```python
import numpy as np

def run(mode, eta=0.1, steps=200):
    x, y = 1.0, 1.0
    gx_prev, gy_prev = y, x                     # gradients at the start: dV/dx = y, dV/dy = x
    r = []
    for _ in range(steps):
        gx, gy = y, x
        if mode == "simultaneous":
            x, y = x - eta * gx, y + eta * gy
        elif mode == "alternating":
            x = x - eta * gx
            y = y + eta * x                     # y sees the new x
        else:                                   # optimistic: extrapolate with the last gradient
            x, y = x - eta * (2 * gx - gx_prev), y + eta * (2 * gy - gy_prev)
        gx_prev, gy_prev = gx, gy
        r.append(np.hypot(x, y))
    return r[-1]

for m in ["simultaneous", "alternating", "optimistic"]:
    print(m, run(m))        # simultaneous: grows like (1+eta^2)^(t/2); alternating: bounded, orbits; optimistic: -> 0
```
Simultaneous updates multiply the radius by $\sqrt{1+\eta^2}$ each step, so it grows without bound. Alternating updates orbit at a roughly constant radius, and the optimistic update (Daskalakis et al., ICLR 2018, "Training GANs with Optimism") converges. The take-away is that the *dynamics* of a two-player game, not only the loss, determine whether training converges: this is why learning-rate ratios (TTUR), momentum settings, gradient penalties on $D$ (e.g. R1, Mescheder et al., ICML 2018) and optimistic methods matter for GANs.
:::
:::

:::exercise Exercise 9: reading FID
You compute FID between 5,000 real and 5,000 generated images and get 18.2; with 50,000 generated images you get 14.1. Give two explanations for the difference, and describe three experiments you would run before claiming that model A (FID 14.1) is better than model B (FID 14.9).

:::details Solution
FID is **biased upward for small samples**: the covariance estimate from few samples in 2048 dimensions is noisy and the trace of the matrix square root is underestimated, so FID decreases as the sample size grows; and 5,000 samples contain fewer rare modes, so the generated statistics are less representative. For A versus B: (1) fix the protocol (same number of samples, same Inception weights, same resizing); (2) repeat with several random seeds of the training run and of the sample draw to estimate the standard deviation (a difference of 0.8 is plausibly within the noise); (3) compute precision and recall, or KID, and look at samples to see whether the difference is fidelity or diversity.
:::
:::

:::exercise Exercise 10: designing a CycleGAN failure
Describe a concrete pair of image domains for which CycleGAN will not work well, explain which part of the loss is responsible, and suggest a remedy.

:::details Solution
Translating between domains that differ in **geometry or shape**, such as cats to dogs, or a photograph of a person to a different pose. The cycle-consistency loss rewards mappings that are nearly invertible pixel-wise, which encourages local texture changes that preserve structure, and the adversarial loss on its own is satisfied by any plausible image of the target domain. The generator is therefore pushed to change colours and textures and leave the shape. Another failure is **steganography**: the generator hides the information required for the reverse mapping in low-amplitude patterns. Remedies include paired data if any exists (pix2pix), shape-aware architectures with larger receptive fields, a perceptual or feature-space cycle loss, adding noise to the generator's outputs before the reverse mapping, or moving to diffusion-based translation methods.
:::
:::

## Key takeaways

:::key Key takeaways
- Generative models differ in what they let you compute: autoregressive models and flows give exact likelihoods, VAEs give a bound, GANs give only samples, and the choice trades off sample quality, coverage and speed.
- Autoencoders learn a code by reconstruction through a bottleneck (or noise, or sparsity); a linear autoencoder spans the PCA subspace. A plain autoencoder has no prior over codes, so it is not a generative model.
- The ELBO is $\E_q[\log p(x\mid z)]-\KL(q(z\mid x)\|p(z))$, a lower bound whose gap is $\KL(q(z\mid x)\|p(z\mid x))$. The KL to a standard normal is $\tfrac12\sum(\mu^2+\sigma^2-1-\log\sigma^2)$.
- The reparameterization trick, $z=\mu+\sigma\epsilon$, moves the randomness out of the computation graph and makes the expectation differentiable with a low-variance gradient.
- $\beta$ weights rate against distortion. Large $\beta$ or a strong decoder leads to posterior collapse; remedies are KL warm-up, free bits and weaker decoders. VQ-VAE uses discrete codes, a straight-through estimator and a separate prior, and avoids the KL altogether.
- For a fixed generator the optimal discriminator is $p_\text{data}/(p_\text{data}+p_g)$, and the minimax value is $-\log4+2\,\mathrm{JS}(p_\text{data}\|p_g)$. The non-saturating generator loss avoids vanishing gradients early on.
- GAN training is a game and may cycle; mode collapse, vanishing gradients and sensitivity to hyperparameters are typical. WGAN replaces JS with the Wasserstein distance, enforced via 1-Lipschitz critics (gradient penalty or spectral normalization).
- DCGAN's recipe (strided convolutions, batch norm, LeakyReLU, Adam with $\beta_1=0.5$) is the template for stable convolutional GANs; conditional GANs, pix2pix and CycleGAN add control, and StyleGAN adds a mapping network and per-layer styles.
- Evaluate with FID (fidelity and diversity, protocol-dependent), Inception Score (classifier-dependent, blind to real data), and precision and recall (which separate fidelity from coverage), and always look at samples.
- Assignment 3 is due at the end of this week, and project feedback meetings happen this week.
:::

## Further reading {#further-reading}

**Essential**
- Kingma and Welling, "Auto-Encoding Variational Bayes", ICLR 2014 (arXiv:1312.6114). **Essential.**
- Goodfellow et al., "Generative Adversarial Nets", NeurIPS 2014 (arXiv:1406.2661). **Essential.**
- Arjovsky, Chintala and Bottou, "Wasserstein GAN", ICML 2017 (arXiv:1701.07875). **Essential.**

**Autoencoders and VAEs**
- Doersch, "Tutorial on Variational Autoencoders", 2016 (arXiv:1606.05908).
- Kingma and Welling, "An Introduction to Variational Autoencoders", Foundations and Trends in Machine Learning, 2019 (arXiv:1906.02691).
- Rezende, Mohamed and Wierstra, "Stochastic Backpropagation and Approximate Inference in Deep Generative Models", ICML 2014 (arXiv:1401.4082).
- Vincent, Larochelle, Bengio and Manzagol, "Extracting and Composing Robust Features with Denoising Autoencoders", ICML 2008.
- Higgins et al., "beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework", ICLR 2017.
- Bowman et al., "Generating Sentences from a Continuous Space", CoNLL 2016 (arXiv:1511.06349).
- Alemi et al., "Fixing a Broken ELBO", ICML 2018 (arXiv:1711.00464).
- van den Oord, Vinyals and Kavukcuoglu, "Neural Discrete Representation Learning" (VQ-VAE), NeurIPS 2017 (arXiv:1711.00937).
- Esser, Rombach and Ommer, "Taming Transformers for High-Resolution Image Synthesis" (VQGAN), CVPR 2021 (arXiv:2012.09841).

**GANs**
- Goodfellow, "NIPS 2016 Tutorial: Generative Adversarial Networks", 2016 (arXiv:1701.00160).
- Radford, Metz and Chintala, "Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks" (DCGAN), ICLR 2016 (arXiv:1511.06434).
- Arjovsky and Bottou, "Towards Principled Methods for Training Generative Adversarial Networks", ICLR 2017 (arXiv:1701.04862).
- Gulrajani et al., "Improved Training of Wasserstein GANs", NeurIPS 2017 (arXiv:1704.00028).
- Miyato et al., "Spectral Normalization for Generative Adversarial Networks", ICLR 2018 (arXiv:1802.05957).
- Mescheder, Geiger and Nowozin, "Which Training Methods for GANs do actually Converge?", ICML 2018 (arXiv:1801.04406).
- Isola et al., "Image-to-Image Translation with Conditional Adversarial Networks" (pix2pix), CVPR 2017 (arXiv:1611.07004).
- Zhu et al., "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks", ICCV 2017 (arXiv:1703.10593).
- Brock, Donahue and Simonyan, "Large Scale GAN Training for High Fidelity Natural Image Synthesis" (BigGAN), ICLR 2019 (arXiv:1809.11096).
- Karras, Laine and Aila, "A Style-Based Generator Architecture for Generative Adversarial Networks", CVPR 2019 (arXiv:1812.04948).
- Karras et al., "Analyzing and Improving the Image Quality of StyleGAN" (StyleGAN2), CVPR 2020 (arXiv:1912.04958).

**Evaluation**
- Salimans et al., "Improved Techniques for Training GANs", NeurIPS 2016 (arXiv:1606.03498).
- Heusel et al., "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium", NeurIPS 2017 (arXiv:1706.08500).
- Kynkaanniemi et al., "Improved Precision and Recall Metric for Assessing Generative Models", NeurIPS 2019 (arXiv:1904.06991).
