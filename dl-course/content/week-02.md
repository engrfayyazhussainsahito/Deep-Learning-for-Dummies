---
title: Neural Networks and Backpropagation
phase: Foundations
summary: From the perceptron to multilayer networks, why they can approximate almost anything, the activation and loss functions that shape training, and a complete derivation of backpropagation and automatic differentiation, with a from-scratch NumPy lab.
---

:::key Learning objectives
By the end of this week you will be able to:
- Explain what a perceptron and a multilayer perceptron (MLP) compute, count their parameters, and track every tensor shape through a batched forward pass.
- State the universal approximation theorem, explain *why* ReLU units can fit any 1D function, and say precisely what the theorem does **not** promise.
- Compare sigmoid, tanh, ReLU, Leaky ReLU, GELU and SiLU by range, derivative, saturation and cost, and choose one for a given architecture.
- Derive the gradient of softmax cross-entropy ($\hat{\boldsymbol p}-\boldsymbol y$), and use MSE, hinge and focal losses appropriately.
- Draw the computational graph of an expression and run the forward and backward passes by hand.
- Derive backpropagation for a scalar chain, then for a vectorized, batched layer, and implement it from scratch with numerical gradient checking.
- Explain forward-mode versus reverse-mode automatic differentiation, how PyTorch's autograd records and replays a graph, and write a custom `torch.autograd.Function`.
:::

[Week 1](week-01.html) gave you the language: tensors, gradients, losses, gradient descent. This week we build the object that language was designed to describe. A neural network is a *parametrized function composed of simple layers*, and training it means computing the gradient of a loss with respect to millions of parameters and following it downhill. The algorithm that makes that gradient affordable, **backpropagation**, is the single most important computation in deep learning. Everything in the next sixteen weeks (convolutions, attention, transformers, diffusion) is trained by it.

We go in a deliberate order. First the neuron and its limits (why one is not enough), then layers and shapes, then the theory that justifies stacking them. Then the ingredients you choose when you design a network: activation functions at the hidden layers, and an output layer plus loss at the end. Finally the machinery: computational graphs, a full derivation of backpropagation, automatic differentiation, and a lab in which you build a network from nothing but NumPy and prove its gradients are right.

## The perceptron

### A neuron as a weighted vote

The simplest model of a neuron takes a vector of inputs $\boldsymbol x\in\R^d$, forms a weighted sum, adds a bias, and applies a nonlinearity:

$$
y=\phi(\boldsymbol w^\top\boldsymbol x+b)=\phi\Big(\sum_{j=1}^d w_jx_j+b\Big).
$$

The numbers $w_j$ are **weights** (how strongly input $j$ pushes the output), $b$ is the **bias** (a threshold shift: the neuron fires more or less easily), and $\phi$ is the **activation function**. For Rosenblatt's original **perceptron**, $\phi$ is a hard threshold: $y=1$ if $\boldsymbol w^\top\boldsymbol x+b>0$ and $y=0$ otherwise.

Geometrically, the equation $\boldsymbol w^\top\boldsymbol x+b=0$ describes a hyperplane. The vector $\boldsymbol w$ is perpendicular to it, and $b/\norm{\boldsymbol w}$ sets its signed distance from the origin. The neuron answers one question: *on which side of this hyperplane does the input lie?* A perceptron is therefore a **linear classifier**, exactly the model you met as logistic regression in Week 1 but with a hard decision instead of a probability.

{{figure: week02-perceptron | A perceptron computes a weighted sum plus bias, then a threshold. It can represent linearly separable functions (left panel) but no single neuron can represent XOR (right panel).}}

:::history From McCulloch-Pitts to the Mark I
Warren McCulloch and Walter Pitts (1943) proposed the first mathematical neuron: a binary threshold unit that, wired into networks, can compute logical functions. Frank Rosenblatt (1958) added what they lacked, a *learning rule*, and built the Mark I Perceptron, a room-sized machine that learned to classify simple images with motor-driven potentiometers as weights. The press was breathless. In 1969 Marvin Minsky and Seymour Papert published *Perceptrons*, a rigorous analysis of what single-layer perceptrons cannot do (parity, connectedness, XOR). Their book is often blamed for the first "AI winter" in neural network funding; the nuanced truth is that they also noted multilayer networks might be more powerful, but no one then had a practical way to train them. That way arrived with backpropagation (popularized in 1986, see below).
:::

### The perceptron learning rule

Rosenblatt's rule is delightfully simple. Cycle through the training examples $(\boldsymbol x_i,y_i)$ with labels $y_i\in\{-1,+1\}$. If the current prediction $\operatorname{sign}(\boldsymbol w^\top\boldsymbol x_i+b)$ is wrong, nudge the hyperplane toward the example:

$$
\boldsymbol w\leftarrow\boldsymbol w+y_i\boldsymbol x_i,\qquad b\leftarrow b+y_i.
$$

If the example was labeled $+1$ but scored too low, we add $\boldsymbol x_i$ to $\boldsymbol w$, which raises $\boldsymbol w^\top\boldsymbol x_i$. If it was labeled $-1$ but scored too high, we subtract it. Correct examples cause no update. The **perceptron convergence theorem** (Novikoff, 1962) guarantees that if the data are linearly separable with margin $\gamma$ and all points satisfy $\norm{\boldsymbol x_i}\le R$, the algorithm makes at most $(R/\gamma)^2$ mistakes and then stops. Notice that the bound depends on the margin, not on the dimension.

```python:perceptron.py
import numpy as np

def train_perceptron(X, y, epochs=50):
    """X: (N, d) float array, y: (N,) labels in {-1, +1}. Returns w (d,), b, mistakes per epoch."""
    N, d = X.shape
    w, b = np.zeros(d), 0.0
    history = []
    for _ in range(epochs):
        mistakes = 0
        for i in np.random.permutation(N):
            if y[i] * (X[i] @ w + b) <= 0:          # wrong side (or on the boundary)
                w += y[i] * X[i]                      # Rosenblatt update
                b += y[i]
                mistakes += 1
        history.append(mistakes)
        if mistakes == 0:                              # converged: all points classified correctly
            break
    return w, b, history

rng = np.random.default_rng(0)
# 1) linearly separable: AND-like data
X = rng.normal(size=(200, 2)); y = np.where(X[:, 0] + 0.5 * X[:, 1] > 0.2, 1, -1)
w, b, hist = train_perceptron(X, y)
print("separable:  mistakes per epoch", hist)

# 2) XOR: the sign of x1*x2 -- no hyperplane separates it
X = rng.normal(size=(200, 2)); y = np.where(X[:, 0] * X[:, 1] > 0, 1, -1)
w, b, hist = train_perceptron(X, y, epochs=50)
print("XOR:        mistakes in last epoch", hist[-1], "of", len(y), "(never reaches 0)")
```

The second experiment is the crux. On XOR-like data the perceptron cycles forever because no weight vector is correct on all points. Hold that thought: the fix is not a better learning rule, it is a better *model*.

### Why one neuron is not enough: features and depth

XOR is not separable in the raw coordinates $(x_1,x_2)$. But add one hand-made feature, $x_3=x_1x_2$, and it becomes trivially separable by a hyperplane in $(x_1,x_2,x_3)$. This is the central idea of representation learning: **a linear classifier on top of the right features can solve problems the same classifier cannot solve on the raw input**. Classical machine learning engineered those features by hand (SIFT, HOG, n-grams). A multilayer network *learns* them: each hidden neuron is a learned feature, and the output neuron is a linear classifier on those features. Making this precise is the job of the next two sections.

## Multilayer perceptrons

### Layers, notation and the forward pass

A **multilayer perceptron** (MLP, also *feedforward network* or *fully connected network*) stacks layers of neurons. Every neuron of layer $\ell$ receives the outputs of *all* neurons of layer $\ell-1$ (hence "fully connected" or "dense"), and nothing feeds back. We use the following notation throughout the course:

| Symbol | Meaning | Shape (one example) |
| --- | --- | --- |
| $\boldsymbol a^{(0)}=\boldsymbol x$ | input | $(n_0,)$ |
| $\mathbf W^{(\ell)}$ | weight matrix of layer $\ell$ | $(n_{\ell-1},n_\ell)$ |
| $\boldsymbol b^{(\ell)}$ | bias of layer $\ell$ | $(n_\ell,)$ |
| $\boldsymbol z^{(\ell)}$ | **pre-activation** (the "logits" of that layer) | $(n_\ell,)$ |
| $\boldsymbol a^{(\ell)}$ | **activation** (output of layer $\ell$) | $(n_\ell,)$ |
| $L$ | number of weight layers | |

The forward pass for one example is

$$
\boldsymbol z^{(\ell)}=\mathbf W^{(\ell)\top}\boldsymbol a^{(\ell-1)}+\boldsymbol b^{(\ell)},\qquad
\boldsymbol a^{(\ell)}=\phi^{(\ell)}\big(\boldsymbol z^{(\ell)}\big),\qquad \ell=1,\dots,L,
$$

and the network's output is $\boldsymbol a^{(L)}$ (often the last $\phi$ is the identity, and the loss handles the rest). A layer with $n_{\ell-1}$ inputs and $n_\ell$ outputs has $n_{\ell-1}n_\ell$ weights and $n_\ell$ biases.

For a **batch** of $B$ examples we stack them as rows: $\mathbf X\in\R^{B\times n_0}$, and the forward pass becomes

$$
\mathbf Z^{(\ell)}=\mathbf A^{(\ell-1)}\mathbf W^{(\ell)}+\boldsymbol 1\boldsymbol b^{(\ell)\top},\qquad \mathbf A^{(\ell)}=\phi(\mathbf Z^{(\ell)}),
$$

where the bias is broadcast across rows. This "row convention" matches how PyTorch and NumPy lay out data (first axis = batch), and all code in this course follows it. (PyTorch's `nn.Linear` stores its weight transposed, shape `(out, in)`, and computes `x @ W.T + b`; the mathematics is identical.)

{{figure: week02-mlp-shapes | A 784-128-64-10 MLP applied to a batch of B flattened MNIST images. Each arrow is one matrix multiply followed by a nonlinearity (except the last layer, whose softmax is usually folded into the loss).}}

Counting parameters is a useful sanity habit. For the network in the figure: $784\cdot128+128=100{,}480$, then $128\cdot64+64=8{,}256$, then $64\cdot10+10=650$, for $109{,}386$ in total. Notice that the first layer holds 92% of them: the cost of a dense network is dominated by the widest matrices, which is one reason convolutions (Week 5) are so valuable for images.

```python:mlp_shapes.py
import torch
from torch import nn

model = nn.Sequential(
    nn.Flatten(),                      # (B, 1, 28, 28) -> (B, 784)
    nn.Linear(784, 128), nn.ReLU(),
    nn.Linear(128, 64),  nn.ReLU(),
    nn.Linear(64, 10),                 # logits; softmax is folded into the loss
)

x = torch.randn(32, 1, 28, 28)         # a fake batch of 32 "images"
h = x
for layer in model:                    # inspect every intermediate shape
    h = layer(h)
    print(f"{layer.__class__.__name__:8s} -> {tuple(h.shape)}")

n_params = sum(p.numel() for p in model.parameters())
print("parameters:", n_params)         # 109386
print("first weight:", tuple(model[1].weight.shape), "(out, in) = PyTorch stores W transposed")
```

### Why nonlinearity is essential

Suppose we dropped every $\phi$. Then $\boldsymbol a^{(L)}=\mathbf W^{(L)\top}\cdots\mathbf W^{(1)\top}\boldsymbol x+\text{(bias terms)}$, a product of matrices, which is itself just one matrix: the whole deep network collapses to a single affine map with no more expressive power than linear regression. **Depth without nonlinearity buys nothing.** All of the expressive richness of a neural network comes from interleaving linear maps with nonlinearities. (The multilayer structure still changes *optimization*: a deep linear network has a non-convex loss with interesting dynamics, a favorite theory playground, but it cannot represent anything a single linear layer cannot.)

### Solving XOR by hand

Here is a two-layer network with two hidden ReLU units that computes XOR exactly on $\{0,1\}^2$:

$$
h_1=\relu(x_1+x_2),\quad h_2=\relu(x_1+x_2-1),\quad y=h_1-2h_2.
$$

Check: $(0,0)\mapsto0$; $(1,0)$ and $(0,1)\mapsto h_1=1,h_2=0\Rightarrow y=1$; $(1,1)\mapsto h_1=2,h_2=1\Rightarrow y=0$. The hidden layer has *folded* the plane so that the two classes become linearly separable in $(h_1,h_2)$. This folding picture generalizes: each hidden layer re-represents the data so the next layer finds the problem easier.

Use the playground to watch this happen. Pick **XOR blobs**, reduce the network to one hidden layer with two units, and train. Then switch to **Two spirals**, which needs more capacity: compare one layer against three. The code that trains the network, forward and backward, is written out in plain JavaScript in the demo; you will write its Python twin in the lab.

{{demo: week02-mlp-playground | Train a small MLP on 2D toy problems | 560}}

## The universal approximation theorem

### What the theorem says

It is natural to ask: how much can one hidden layer compute? A remarkable family of results answers: essentially *anything continuous*, if the layer is wide enough.

:::definition Universal approximation (informal statement)
Let $\sigma$ be a continuous, non-polynomial activation function (sigmoid, tanh, ReLU, GELU, ...). Let $K\subset\R^d$ be compact and $f:K\to\R$ continuous. Then for every $\varepsilon>0$ there exist a width $N$, weights $\boldsymbol w_i\in\R^d$, biases $b_i$, and output coefficients $c_i$ such that
$$
\sup_{\boldsymbol x\in K}\Big|f(\boldsymbol x)-\sum_{i=1}^{N}c_i\,\sigma(\boldsymbol w_i^\top\boldsymbol x+b_i)\Big|<\varepsilon .
$$
:::

Historically: George Cybenko (1989) proved it for sigmoidal activations using a functional-analysis argument (Hahn-Banach plus the Riesz representation theorem); Hornik, Stinchcombe and White (1989) proved a closely related result for general squashing functions; Hornik (1991) showed that what matters is the *architecture*, not the particular activation; and Leshno, Lin, Pinkus and Schocken (1993) showed that a continuous activation gives universality **if and only if it is not a polynomial**. That last condition makes sense: sums of polynomials of bounded degree stay polynomials of bounded degree, so they cannot approximate everything.

### Intuition: ReLU hinges build piecewise-linear functions

The cleanest way to *see* universality is in one dimension with ReLU. A single hidden unit $\relu(wx+b)$ is a **hinge**: zero until the point $x=-b/w$ (its *kink*), then a ramp with slope $w$. Adding a scaled hinge to a function changes its slope by a fixed amount at the kink and leaves everything to the left untouched.

{{figure: week02-relu-bumps | Left: each ReLU unit contributes a hinge. Right: a sum of hinges is a continuous piecewise-linear function, so $N$ hidden units can produce $N+1$ linear pieces and can follow a smooth target arbitrarily closely.}}

:::math Piecewise-linear interpolation as a one-hidden-layer ReLU network
Take a target $f$ on $[a,b]$ and knots $a=x_0<x_1<\dots<x_N=b$. Let $s_i=\dfrac{f(x_{i+1})-f(x_i)}{x_{i+1}-x_i}$ be the slope of the chord on $[x_i,x_{i+1}]$, for $i=0,\dots,N-1$. Define
$$
\hat f(x)=f(x_0)+\sum_{i=0}^{N-1}c_i\,\relu(x-x_i),\qquad c_0=s_0,\quad c_i=s_i-s_{i-1}\ (i\ge1).
$$
On $[x_k,x_{k+1}]$ exactly the hinges $i\le k$ are active, so the slope is $\sum_{i\le k}c_i=s_k$ (the sum telescopes), and $\hat f(x_0)=f(x_0)$. A continuous function with the right starting value and the right slope on each piece equals the chord interpolant, so $\hat f(x_k)=f(x_k)$ at every knot. This is a one-hidden-layer ReLU network with $N$ hidden units ($w_i=1$, $b_i=-x_i$, output weights $c_i$) plus an output bias.

*How good is it?* For $f\in C^2$ and uniform spacing $h=(b-a)/N$, the classical error bound for linear interpolation is
$$
\max_{x\in[a,b]}|f(x)-\hat f(x)|\le\frac{h^2}{8}\max|f''|.
$$
Doubling the number of hidden units cuts the worst-case error by about 4. Since continuous functions can be approximated uniformly by $C^2$ ones, universality in 1D follows.
:::

The chart verifies the bound numerically for $f(x)=\sin(3x)+0.3x$ on $[-1,1]$ (where $\max|f''|=9$). The bound is almost tight.

```chart
{
  "type": "line",
  "caption": "Worst-case error of the interpolating one-hidden-layer ReLU network for f(x)=sin(3x)+0.3x on [-1,1], measured on a 20,001-point grid, against the h^2/8 * max|f''| bound (computed with NumPy; both axes logarithmic). The slope of -2 on a log-log plot is the 1/N^2 rate.",
  "height": 320,
  "data": {"labels": [2, 4, 8, 16, 32, 64, 128], "datasets": [
    {"label": "measured max error", "data": [0.9272, 0.2108, 0.0663, 0.0174, 0.00439, 0.00110, 0.000275], "borderColor": "var(--c-accent)", "pointRadius": 4},
    {"label": "bound h^2/8 * 9", "data": [1.125, 0.28125, 0.0703, 0.01758, 0.004395, 0.0010986, 0.00027466], "borderColor": "var(--c-blue)", "borderDash": [6, 4], "pointRadius": 0}
  ]},
  "options": {"scales": {"x": {"type": "logarithmic", "title": {"display": true, "text": "hidden units N"}}, "y": {"type": "logarithmic", "title": {"display": true, "text": "max |f - fhat|"}}}}
}
```

Now try it yourself. The demo *trains* a one-hidden-layer ReLU network with Adam to fit several targets. Start with the target $|x|-0.5$ and a single unit: one hinge cannot make a V. Then try two units. Then switch to the fast wiggle and watch how many hinges are needed before the fit tracks it. Notice also that training does not always place the kinks where interpolation would; gradient descent finds *a* solution, not the tidy one from the proof.

{{demo: week02-relu-fit | Fit a 1D function with a handful of ReLU units | 560}}

```python:relu_fit.py
import numpy as np, torch
from torch import nn

torch.manual_seed(0)
x = torch.linspace(-1, 1, 200).unsqueeze(1)            # (200, 1)
y = torch.sin(3 * x) + 0.3 * x                         # target

def fit(width, steps=4000, lr=0.01):
    net = nn.Sequential(nn.Linear(1, width), nn.ReLU(), nn.Linear(width, 1))
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss = ((net(x) - y) ** 2).mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        return (net(x) - y).abs().max().item()

for width in [2, 4, 8, 16, 32, 64]:
    print(f"width {width:3d}: max error {fit(width):.4f}")
```

### What the theorem does *not* say

The theorem is an *existence* result about a particular function class, and it is easy to over-read. Be precise about its limits:

1. **It does not bound the width.** In $d$ dimensions the required number of units can grow exponentially in $d$ for generic continuous targets (the curse of dimensionality). Barron (1993) showed a dimension-independent $O(1/\sqrt N)$ rate for functions whose Fourier transform has a finite first moment, but that is a restricted class.
2. **It says nothing about learning.** That weights exist does not mean gradient descent finds them, nor that a finite training set pins them down. Existence is not a training algorithm.
3. **It says nothing about generalization.** A network with enough units can interpolate any finite training set, including pure noise (Zhang et al., 2017, which we revisit in Week 4). Fitting is cheap; fitting the *right* function from limited data is the real problem.
4. **It is silent on depth.** Shallow wide networks can in principle do everything, but they can need exponentially more units than deep ones for some functions.

### Why depth helps: counting linear regions

Depth gives *compositional* efficiency. Telgarsky (2016) constructed a ReLU network with $k$ layers and constant width whose output is a sawtooth with $2^k$ teeth; a network with a single hidden layer needs on the order of $2^k$ units to match it. Intuitively, each layer can *fold* the input space, and folds compound multiplicatively: fold a strip of paper once, twice, three times, and the creases double each time. Montufar, Pascanu, Cho and Bengio (2014) formalized this by showing that the number of linear regions of a deep ReLU network can grow exponentially with depth but only polynomially with width. Lu et al. (2017) proved the complementary statement that width-$(d+4)$ ReLU networks of arbitrary depth are universal on $\R^d$, so *depth alone* also suffices.

:::note Expressive power is not what limits you
In practice the limiting factors are data, optimization and generalization, not representational capacity. Universality tells you that the hypothesis class is not the bottleneck; it tells you nothing about which architecture to pick. The architectural inductive biases of later weeks (convolution, recurrence, attention) exist precisely because plain MLPs, although universal, are statistically inefficient for images, sequences and graphs.
:::

## Activation functions

The activation $\phi$ is applied elementwise after each linear map, and it determines three things at once: what the network can represent, how gradients flow backward through it, and how expensive and numerically stable it is. Because backpropagation multiplies by $\phi'$ at every layer, the *derivative* matters as much as the function itself.

### What makes a good activation?

Before the catalogue, here are the properties to look for.

- **Nonlinearity** (and, per Leshno et al., non-polynomial). Otherwise the network collapses to a linear map.
- **Gradient flow.** A derivative that is near zero over large input regions (**saturation**) starves earlier layers of gradient; a derivative that is exactly 1 over a large region passes gradient through unchanged.
- **Range and centering.** Bounded outputs keep activations tame but saturate; outputs centered near zero make the next layer's inputs better conditioned. Unbounded outputs (ReLU) can grow, which normalization layers ([Week 3](week-03.html)) then control.
- **Smoothness.** ReLU has a kink at 0; smooth functions (GELU, SiLU) have well-defined second derivatives, which matters for some optimizers, second-order analysis and double backpropagation.
- **Cost.** ReLU is a comparison; sigmoid and tanh need exponentials; GELU needs an error function or an approximation. On large models, cheap elementwise ops are often memory-bandwidth-bound anyway, so the real cost difference is small on GPUs, but fused kernels help.

### Sigmoid

$$
\sigma(z)=\frac1{1+e^{-z}},\qquad \sigma'(z)=\sigma(z)\big(1-\sigma(z)\big)\le\tfrac14.
$$

The logistic sigmoid squashes the real line into $(0,1)$ and was the default activation of early networks, partly because it resembles a firing rate and partly because it is smooth and differentiable. Its drawbacks are severe for *hidden* layers. (1) **Saturation:** for $|z|>5$ the derivative is below $0.007$, so units stuck at the extremes barely learn. (2) **Maximum slope $1/4$:** even in the best case each sigmoid layer shrinks the backward signal by 4, so a 10-layer sigmoid network shrinks gradients by up to $4^{10}\approx10^6$ before weights are even considered. (3) **Not zero-centered:** outputs are all positive, so the gradients on a neuron's incoming weights all share the sign of the neuron's error signal, forcing zig-zag updates. Sigmoid remains the right choice for *output* units that must be probabilities (binary classification, gates in LSTMs).

### Tanh

$$
\tanh(z)=\frac{e^z-e^{-z}}{e^z+e^{-z}}=2\sigma(2z)-1,\qquad \tanh'(z)=1-\tanh^2(z)\le1.
$$

Tanh is a rescaled, shifted sigmoid: zero-centered with maximum slope 1 at the origin. LeCun, Bottou, Orr and Müller's "Efficient BackProp" (1998) recommended it over sigmoid for exactly those reasons. It still saturates, so deep tanh networks need careful initialization (Xavier/Glorot, [Week 3](week-03.html)), but with that they train acceptably. Tanh is still standard inside recurrent networks.

### ReLU

$$
\relu(z)=\max(0,z),\qquad \relu'(z)=\begin{cases}1&z>0\\0&z<0\end{cases}
$$

(at exactly $z=0$ the function is not differentiable; frameworks pick a subgradient, PyTorch uses $0$). The rectified linear unit was used in neuroscience-inspired models (Hahnloser et al., 2000), popularized for deep learning by Nair and Hinton (2010) and Glorot, Bordes and Bengio (2011), and became the workhorse after AlexNet (Krizhevsky et al., 2012) showed it trained several times faster than tanh. Its virtues: the derivative is exactly 1 for positive inputs, so there is no saturation on that side and gradients do not shrink; it is trivially cheap; and it yields **sparse** activations (roughly half the units are exactly zero for centered inputs), which helps both efficiency and, empirically, generalization.

Its signature failure is the **dying ReLU**: a unit whose pre-activation is negative for *every* training input has zero gradient forever, since $\relu'=0$ blocks the whole backward path through it, and no weight update can revive it. This typically happens after a too-large learning rate throws the bias far negative, or with a poor initialization. A network can lose a large fraction of its units this way, and the symptom (a large share of activations always exactly zero) is cheap to measure; the lab pitfalls section shows how.

### Leaky ReLU, PReLU and ELU

The simplest repair is to give the negative side a small slope:

$$
\mathrm{LeakyReLU}_\alpha(z)=\begin{cases}z&z>0\\\alpha z&z\le0\end{cases}\qquad(\alpha\approx0.01\text{ to }0.2).
$$

Maas, Hannun and Ng (2013) introduced Leaky ReLU for acoustic models; He et al. (2015) made $\alpha$ a *learned* parameter (**PReLU**) in the same paper that introduced He initialization. Clevert, Unterthiner and Hochreiter (2016) proposed the **ELU**, $z$ for $z>0$ and $\alpha(e^z-1)$ otherwise, whose smooth negative saturation pushes mean activations toward zero. In practice these variants give small, task-dependent gains; plain ReLU remains a hard-to-beat default for convolutional networks.

### GELU

$$
\mathrm{GELU}(z)=z\,\Phi(z),\qquad \mathrm{GELU}'(z)=\Phi(z)+z\,\varphi(z),
$$

where $\Phi$ and $\varphi$ are the standard normal CDF and density. Hendrycks and Gimpel (2016) motivated it as a *stochastic gate*: multiply the input by a Bernoulli variable that is 1 with probability $\Phi(z)$, and take the expectation. Large positive inputs pass almost unchanged (like ReLU), large negative inputs are suppressed, and the transition is smooth. GELU is **non-monotonic**: it dips to a minimum of about $-0.17$ near $z\approx-0.75$ and then returns to 0. Implementations use either the exact form via the error function or the tanh approximation $0.5z\big(1+\tanh(\sqrt{2/\pi}(z+0.044715z^3))\big)$. GELU is the default in BERT and GPT-style transformers ([Week 9](week-09.html) and [Week 10](week-10.html)).

### SiLU / Swish

$$
\mathrm{SiLU}(z)=z\,\sigma(z),\qquad \mathrm{SiLU}'(z)=\sigma(z)\big(1+z(1-\sigma(z))\big).
$$

Elfwing, Uchibe and Doya (2018, arXiv 2017) introduced it for reinforcement learning; Ramachandran, Zoph and Le (2017) rediscovered it by *automated search* over activation functions and named it Swish (with an optional scale $\beta$: $z\sigma(\beta z)$; $\beta=1$ is SiLU). It looks like GELU: smooth, non-monotonic, with a minimum of about $-0.28$ near $z\approx-1.28$. It is common in modern vision networks (EfficientNet) and, in gated form (SwiGLU), in many recent LLMs ([Week 10](week-10.html)).

### Comparing them

| Activation | Range | Max slope | Saturates? | Zero-centered | Typical use |
| --- | --- | --- | --- | --- | --- |
| Sigmoid | $(0,1)$ | 0.25 | both sides | no | binary outputs, gates |
| Tanh | $(-1,1)$ | 1 | both sides | yes | RNNs, small nets |
| ReLU | $[0,\infty)$ | 1 | negative side (dead) | no | CNNs, default MLPs |
| Leaky ReLU | $\R$ | 1 | no | roughly | GANs, when dead units hurt |
| GELU | $\approx[-0.17,\infty)$ | about 1.13 | negative side (softly) | no | transformers |
| SiLU/Swish | $\approx[-0.28,\infty)$ | about 1.1 | negative side (softly) | no | modern CNNs, LLM MLPs |

```chart
{
  "type": "line",
  "caption": "The six activation functions on [-4, 4] (computed with NumPy; GELU uses the exact erf form, Leaky ReLU uses slope 0.1).",
  "height": 340,
  "data": {"labels":[-4.0,-3.9,-3.8,-3.7,-3.6,-3.5,-3.4,-3.3,-3.2,-3.1,-3.0,-2.9,-2.8,-2.7,-2.6,-2.5,-2.4,-2.3,-2.2,-2.1,-2.0,-1.9,-1.8,-1.7,-1.6,-1.5,-1.4,-1.3,-1.2,-1.1,-1.0,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8,2.9,3.0,3.1,3.2,3.3,3.4,3.5,3.6,3.7,3.8,3.9,4.0],"datasets":[{"label":"Sigmoid","data":[0.018,0.0198,0.0219,0.0241,0.0266,0.0293,0.0323,0.0356,0.0392,0.0431,0.0474,0.0522,0.0573,0.063,0.0691,0.0759,0.0832,0.0911,0.0998,0.1091,0.1192,0.1301,0.1419,0.1545,0.168,0.1824,0.1978,0.2142,0.2315,0.2497,0.2689,0.2891,0.31,0.3318,0.3543,0.3775,0.4013,0.4256,0.4502,0.475,0.5,0.525,0.5498,0.5744,0.5987,0.6225,0.6457,0.6682,0.69,0.7109,0.7311,0.7503,0.7685,0.7858,0.8022,0.8176,0.832,0.8455,0.8581,0.8699,0.8808,0.8909,0.9002,0.9089,0.9168,0.9241,0.9309,0.937,0.9427,0.9478,0.9526,0.9569,0.9608,0.9644,0.9677,0.9707,0.9734,0.9759,0.9781,0.9802,0.982],"borderColor":"var(--c-blue)","borderWidth":2,"tension":0},{"label":"Tanh","data":[-0.9993,-0.9992,-0.999,-0.9988,-0.9985,-0.9982,-0.9978,-0.9973,-0.9967,-0.9959,-0.9951,-0.994,-0.9926,-0.991,-0.989,-0.9866,-0.9837,-0.9801,-0.9757,-0.9705,-0.964,-0.9562,-0.9468,-0.9354,-0.9217,-0.9051,-0.8854,-0.8617,-0.8337,-0.8005,-0.7616,-0.7163,-0.664,-0.6044,-0.537,-0.4621,-0.3799,-0.2913,-0.1974,-0.0997,0.0,0.0997,0.1974,0.2913,0.3799,0.4621,0.537,0.6044,0.664,0.7163,0.7616,0.8005,0.8337,0.8617,0.8854,0.9051,0.9217,0.9354,0.9468,0.9562,0.964,0.9705,0.9757,0.9801,0.9837,0.9866,0.989,0.991,0.9926,0.994,0.9951,0.9959,0.9967,0.9973,0.9978,0.9982,0.9985,0.9988,0.999,0.9992,0.9993],"borderColor":"var(--c-green)","borderWidth":2,"tension":0},{"label":"ReLU","data":[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8,2.9,3.0,3.1,3.2,3.3,3.4,3.5,3.6,3.7,3.8,3.9,4.0],"borderColor":"var(--c-red)","borderWidth":2,"tension":0},{"label":"Leaky ReLU (0.1)","data":[-0.4,-0.39,-0.38,-0.37,-0.36,-0.35,-0.34,-0.33,-0.32,-0.31,-0.3,-0.29,-0.28,-0.27,-0.26,-0.25,-0.24,-0.23,-0.22,-0.21,-0.2,-0.19,-0.18,-0.17,-0.16,-0.15,-0.14,-0.13,-0.12,-0.11,-0.1,-0.09,-0.08,-0.07,-0.06,-0.05,-0.04,-0.03,-0.02,-0.01,0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8,2.9,3.0,3.1,3.2,3.3,3.4,3.5,3.6,3.7,3.8,3.9,4.0],"borderColor":"var(--c-purple)","borderWidth":2,"tension":0},{"label":"GELU","data":[-0.0001,-0.0002,-0.0003,-0.0004,-0.0006,-0.0008,-0.0011,-0.0016,-0.0022,-0.003,-0.004,-0.0054,-0.0072,-0.0094,-0.0121,-0.0155,-0.0197,-0.0247,-0.0306,-0.0375,-0.0455,-0.0546,-0.0647,-0.0758,-0.0877,-0.1002,-0.1131,-0.1258,-0.1381,-0.1492,-0.1587,-0.1657,-0.1695,-0.1694,-0.1646,-0.1543,-0.1378,-0.1146,-0.0841,-0.046,0.0,0.054,0.1159,0.1854,0.2622,0.3457,0.4354,0.5306,0.6305,0.7343,0.8413,0.9508,1.0619,1.1742,1.2869,1.3998,1.5123,1.6242,1.7353,1.8454,1.9545,2.0625,2.1694,2.2753,2.3803,2.4845,2.5879,2.6906,2.7928,2.8946,2.996,3.097,3.1978,3.2984,3.3989,3.4992,3.5994,3.6996,3.7997,3.8998,3.9999],"borderColor":"var(--c-amber)","borderWidth":2,"tension":0},{"label":"SiLU","data":[-0.0719,-0.0774,-0.0831,-0.0893,-0.0957,-0.1026,-0.1098,-0.1174,-0.1253,-0.1336,-0.1423,-0.1512,-0.1605,-0.17,-0.1798,-0.1896,-0.1996,-0.2096,-0.2195,-0.2291,-0.2384,-0.2472,-0.2553,-0.2626,-0.2688,-0.2736,-0.2769,-0.2784,-0.2778,-0.2747,-0.2689,-0.2601,-0.248,-0.2323,-0.2126,-0.1888,-0.1605,-0.1277,-0.09,-0.0475,0.0,0.0525,0.11,0.1723,0.2395,0.3112,0.3874,0.4677,0.552,0.6399,0.7311,0.8253,0.9222,1.0216,1.1231,1.2264,1.3312,1.4374,1.5447,1.6528,1.7616,1.8709,1.9805,2.0904,2.2004,2.3104,2.4202,2.53,2.6395,2.7488,2.8577,2.9664,3.0747,3.1826,3.2902,3.3974,3.5043,3.6107,3.7169,3.8226,3.9281],"borderColor":"var(--c-teal)","borderWidth":2,"tension":0}]},
  "options": {"scales": {"x": {"title": {"display": true, "text": "z"}}, "y": {"title": {"display": true, "text": "phi(z)"}, "min": -1.5, "max": 4}}, "elements": {"point": {"radius": 0}}}
}
```

```chart
{
  "type": "line",
  "caption": "Derivatives of the same activations (central finite differences). Sigmoid never exceeds 0.25 and vanishes in the tails; ReLU is exactly 0 or 1; GELU and SiLU overshoot slightly above 1 for moderate positive inputs.",
  "height": 340,
  "data": {"labels":[-4.0,-3.9,-3.8,-3.7,-3.6,-3.5,-3.4,-3.3,-3.2,-3.1,-3.0,-2.9,-2.8,-2.7,-2.6,-2.5,-2.4,-2.3,-2.2,-2.1,-2.0,-1.9,-1.8,-1.7,-1.6,-1.5,-1.4,-1.3,-1.2,-1.1,-1.0,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8,2.9,3.0,3.1,3.2,3.3,3.4,3.5,3.6,3.7,3.8,3.9,4.0],"datasets":[{"label":"Sigmoid","data":[0.0177,0.0194,0.0214,0.0235,0.0259,0.0285,0.0313,0.0343,0.0376,0.0412,0.0452,0.0494,0.054,0.059,0.0644,0.0701,0.0763,0.0828,0.0898,0.0972,0.105,0.1132,0.1217,0.1306,0.1398,0.1491,0.1587,0.1683,0.1779,0.1874,0.1966,0.2055,0.2139,0.2217,0.2288,0.235,0.2403,0.2445,0.2475,0.2494,0.25,0.2494,0.2475,0.2445,0.2403,0.235,0.2288,0.2217,0.2139,0.2055,0.1966,0.1874,0.1779,0.1683,0.1587,0.1491,0.1398,0.1306,0.1217,0.1132,0.105,0.0972,0.0898,0.0828,0.0763,0.0701,0.0644,0.059,0.054,0.0494,0.0452,0.0412,0.0376,0.0343,0.0313,0.0285,0.0259,0.0235,0.0214,0.0194,0.0177],"borderColor":"var(--c-blue)","borderWidth":2,"tension":0},{"label":"Tanh","data":[0.0013,0.0016,0.002,0.0024,0.003,0.0036,0.0044,0.0054,0.0066,0.0081,0.0099,0.012,0.0147,0.0179,0.0218,0.0266,0.0324,0.0394,0.0479,0.0582,0.0707,0.0856,0.1036,0.125,0.1505,0.1807,0.2162,0.2574,0.305,0.3592,0.42,0.4869,0.5591,0.6347,0.7116,0.7864,0.8556,0.9151,0.961,0.9901,1.0,0.9901,0.961,0.9151,0.8556,0.7864,0.7116,0.6347,0.5591,0.4869,0.42,0.3592,0.305,0.2574,0.2162,0.1807,0.1505,0.125,0.1036,0.0856,0.0707,0.0582,0.0479,0.0394,0.0324,0.0266,0.0218,0.0179,0.0147,0.012,0.0099,0.0081,0.0066,0.0054,0.0044,0.0036,0.003,0.0024,0.002,0.0016,0.0013],"borderColor":"var(--c-green)","borderWidth":2,"tension":0},{"label":"ReLU","data":[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.5,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0],"borderColor":"var(--c-red)","borderWidth":2,"tension":0},{"label":"Leaky ReLU (0.1)","data":[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.55,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0],"borderColor":"var(--c-purple)","borderWidth":2,"tension":0},{"label":"GELU","data":[-0.0005,-0.0007,-0.001,-0.0015,-0.002,-0.0028,-0.0039,-0.0052,-0.0069,-0.0092,-0.0119,-0.0154,-0.0196,-0.0247,-0.0307,-0.0376,-0.0455,-0.0544,-0.0641,-0.0745,-0.0852,-0.096,-0.1062,-0.1153,-0.1227,-0.1275,-0.1289,-0.126,-0.118,-0.104,-0.0833,-0.0554,-0.0199,0.0234,0.0743,0.1325,0.1973,0.2677,0.3425,0.4205,0.5,0.5795,0.6575,0.7323,0.8027,0.8675,0.9257,0.9766,1.0199,1.0554,1.0833,1.104,1.118,1.126,1.1289,1.1275,1.1227,1.1153,1.1062,1.096,1.0852,1.0745,1.0641,1.0544,1.0455,1.0376,1.0307,1.0247,1.0196,1.0154,1.0119,1.0092,1.0069,1.0052,1.0039,1.0028,1.002,1.0015,1.001,1.0007,1.0005],"borderColor":"var(--c-amber)","borderWidth":2,"tension":0},{"label":"SiLU","data":[-0.0527,-0.056,-0.0594,-0.063,-0.0666,-0.0703,-0.074,-0.0776,-0.0813,-0.0848,-0.0881,-0.0912,-0.094,-0.0963,-0.0982,-0.0994,-0.0998,-0.0994,-0.0978,-0.095,-0.0908,-0.0849,-0.0773,-0.0676,-0.0556,-0.0413,-0.0243,-0.0046,0.018,0.0436,0.0723,0.1041,0.1389,0.1766,0.2171,0.26,0.3052,0.3522,0.4007,0.4501,0.5,0.5499,0.5993,0.6478,0.6948,0.74,0.7829,0.8234,0.8611,0.8959,0.9277,0.9564,0.982,1.0046,1.0243,1.0413,1.0556,1.0676,1.0773,1.0849,1.0908,1.095,1.0978,1.0994,1.0998,1.0994,1.0982,1.0963,1.094,1.0912,1.0881,1.0848,1.0813,1.0776,1.074,1.0703,1.0666,1.063,1.0594,1.056,1.0527],"borderColor":"var(--c-teal)","borderWidth":2,"tension":0}]},
  "options": {"scales": {"x": {"title": {"display": true, "text": "z"}}, "y": {"title": {"display": true, "text": "phi'(z)"}, "min": -0.2, "max": 1.3}}, "elements": {"point": {"radius": 0}}}
}
```

The explorer lets you probe each function and its derivative, and also runs a crude **depth experiment**: draw pre-activations from $\mathcal N(0,\sigma^2)$, average $|\phi'|$, and raise it to the power $L$. This ignores the weight matrices, so it is a toy, but it makes the point that a per-layer factor below 1 (sigmoid: at most $0.25$) decays *exponentially* with depth.

{{demo: week02-activations | Activation explorer: function, derivative and the depth effect | 600}}

The effect is real and visible in training. We trained a three-weight-layer MLP (2-64-64-3) on a three-arm spiral dataset (600 points) with full-batch gradient descent at learning rate 0.5, identical initial seeds and a Glorot-style (sigmoid, tanh) or He-style (ReLU) variance. The sigmoid network is learning slowly when the ReLU network has almost solved the problem.

```chart
{
  "type": "line",
  "caption": "Training cross-entropy of a 2-64-64-3 MLP on a 3-arm spiral dataset (600 points, full-batch gradient descent, lr 0.5, 300 epochs, one seed; computed with NumPy, sampled every 6 epochs). Final losses: sigmoid 0.845, tanh 0.184, ReLU 0.017. A single seed is illustrative, not a benchmark, but the gap is large and typical.",
  "height": 340,
  "data": {"labels":[0,6,12,18,24,30,36,42,48,54,60,66,72,78,84,90,96,102,108,114,120,126,132,138,144,150,156,162,168,174,180,186,192,198,204,210,216,222,228,234,240,246,252,258,264,270,276,282,288,294,300],"datasets":[{"label":"sigmoid","data":[1.3929,1.6386,1.2923,1.1815,1.129,1.1061,1.0957,1.0904,1.087,1.0843,1.0819,1.0795,1.077,1.0745,1.0723,1.0711,1.0736,1.0849,1.0971,1.0918,1.0788,1.0681,1.061,1.057,1.0555,1.0546,1.0515,1.0458,1.0388,1.0317,1.0246,1.0172,1.0089,0.9997,0.9898,0.9792,0.968,0.9562,0.9441,0.9316,0.9189,0.9062,0.8936,0.8813,0.8694,0.858,0.8479,0.8459,0.8765,0.8806,0.8454],"borderColor":"var(--c-blue)","borderWidth":2},{"label":"tanh","data":[1.1058,0.8225,0.7685,0.7574,0.7547,0.7536,0.7527,0.7519,0.751,0.7501,0.7492,0.7481,0.747,0.7457,0.7443,0.7428,0.741,0.7389,0.7366,0.734,0.731,0.7276,0.7237,0.7193,0.7142,0.7081,0.7009,0.6919,0.6805,0.666,0.6481,0.6269,0.6034,0.5785,0.5535,0.529,0.5054,0.4828,0.461,0.4399,0.4189,0.3977,0.3755,0.3516,0.3258,0.2988,0.2718,0.2464,0.2232,0.2026,0.1844],"borderColor":"var(--c-green)","borderWidth":2},{"label":"relu","data":[1.2551,0.6938,0.6139,0.5415,0.4669,0.4036,0.38,0.2824,0.2322,0.2024,0.1654,0.1343,0.1164,0.1031,0.0924,0.0837,0.0764,0.0702,0.0649,0.0603,0.0562,0.0526,0.0494,0.0466,0.044,0.0416,0.0395,0.0376,0.0358,0.0342,0.0327,0.0313,0.03,0.0288,0.0277,0.0266,0.0257,0.0247,0.0239,0.0231,0.0223,0.0216,0.0209,0.0203,0.0197,0.0191,0.0185,0.018,0.0175,0.017,0.0166],"borderColor":"var(--c-red)","borderWidth":2}]},
  "options": {"scales": {"x": {"title": {"display": true, "text": "epoch"}}, "y": {"type": "logarithmic", "title": {"display": true, "text": "training loss"}}}, "elements": {"point": {"radius": 0}}}
}
```

```python:activations_autograd.py
import torch, math

z = torch.linspace(-4, 4, 9, requires_grad=True)

acts = {
    "sigmoid": torch.sigmoid,
    "tanh": torch.tanh,
    "relu": torch.relu,
    "leaky_relu": lambda t: torch.nn.functional.leaky_relu(t, 0.1),
    "gelu": torch.nn.functional.gelu,
    "silu": torch.nn.functional.silu,
}

# analytic derivatives to compare against autograd
Phi = lambda t: 0.5 * (1 + torch.erf(t / math.sqrt(2)))
phi = lambda t: torch.exp(-t**2 / 2) / math.sqrt(2 * math.pi)
analytic = {
    "sigmoid": lambda t: torch.sigmoid(t) * (1 - torch.sigmoid(t)),
    "tanh": lambda t: 1 - torch.tanh(t) ** 2,
    "relu": lambda t: (t > 0).float(),
    "leaky_relu": lambda t: torch.where(t > 0, torch.ones_like(t), torch.full_like(t, 0.1)),
    "gelu": lambda t: Phi(t) + t * phi(t),
    "silu": lambda t: torch.sigmoid(t) * (1 + t * (1 - torch.sigmoid(t))),
}

for name, f in acts.items():
    (g,) = torch.autograd.grad(f(z).sum(), z)        # sum -> scalar; elementwise so g_i = f'(z_i)
    err = (g - analytic[name](z.detach())).abs().max().item()
    print(f"{name:11s} max |autograd - analytic| = {err:.2e}")
```

:::tip Which activation should I use?
Start with the default for your architecture: **ReLU** for plain MLPs and CNNs, **GELU** or **SiLU** in transformers, **tanh/sigmoid** only where bounded outputs have a meaning (gates, probabilities). Change it only after the data pipeline, loss, initialization and learning rate are verified: the differences between ReLU-family activations are usually smaller than the differences caused by any of those. If you suspect dead units, try Leaky ReLU or a lower learning rate before anything exotic.
:::

## Output layers and loss functions

The last layer and the loss are a *pair*: the output layer produces numbers in a form that suits the task, and the loss measures their disagreement with the target. Choosing them is mostly a matter of matching the **probabilistic model** of your targets, and the right pairing produces gradients with a beautifully simple form.

### Regression: linear output and mean squared error

For a real-valued target, use a linear output layer ($\phi=$ identity) and the squared error

$$
\mathcal L_{\text{MSE}}=\frac1B\sum_{i=1}^B\frac12\norm{\hat{\boldsymbol y}_i-\boldsymbol y_i}^2,\qquad
\frac{\partial\mathcal L}{\partial\hat{\boldsymbol y}_i}=\frac1B(\hat{\boldsymbol y}_i-\boldsymbol y_i).
$$

(The $\frac12$ is a convention that cancels the 2 from differentiation; PyTorch's `nn.MSELoss` omits it, which only rescales the gradient by 2.) MSE is the **maximum likelihood** loss if the target is the model output plus Gaussian noise of fixed variance, $y\sim\mathcal N(\hat y,\sigma^2)$: the negative log-likelihood is $\frac1{2\sigma^2}(y-\hat y)^2+\text{const}$. That also tells you when it is wrong: for heavy-tailed noise or outliers, a few huge errors dominate the sum, and the Huber or L1 loss is more robust.

### Binary classification: sigmoid and binary cross-entropy

For $y\in\{0,1\}$, output one logit $z$ and set $p=\sigma(z)=P(y=1\mid\boldsymbol x)$. The negative log-likelihood of a Bernoulli variable is the **binary cross-entropy**

$$
\mathcal L=-\big[y\log p+(1-y)\log(1-p)\big],\qquad \frac{\partial\mathcal L}{\partial z}=p-y.
$$

(Check: $\partial\mathcal L/\partial p=-y/p+(1-y)/(1-p)$ and $\partial p/\partial z=p(1-p)$; multiply and simplify.) Always compute this from the **logit**, not from $p$: `F.binary_cross_entropy_with_logits` uses the identity $\log(1+e^{-|z|})+\max(z,0)-yz$ and never evaluates $\log(0)$.

:::tip Why not MSE for classification?
Suppose we put an MSE loss on a sigmoid output: $\mathcal L=\frac12(p-y)^2$. Then $\partial\mathcal L/\partial z=(p-y)\,p(1-p)$. When the model is confidently *wrong* ($y=1$ but $p\approx0$), the factor $p(1-p)\approx0$ kills the gradient, so the worst mistakes learn the slowest. Cross-entropy's gradient $p-y$ has no such factor: the more wrong the prediction, the larger the gradient. This is the cleanest argument for pairing each output nonlinearity with its *matching* log-likelihood loss.
:::

### Multiclass classification: softmax and cross-entropy

For $K$ classes the network outputs a vector of **logits** $\boldsymbol z\in\R^K$ and the **softmax** converts them to a probability distribution:

$$
p_k=\softmax(\boldsymbol z)_k=\frac{e^{z_k}}{\sum_{j=1}^Ke^{z_j}}.
$$

Softmax is invariant to adding the same constant to all logits, which is what makes the numerically stable implementation (subtract $\max_k z_k$ first) legitimate. With a one-hot target $\boldsymbol y$ (class $c$), the cross-entropy loss is $\mathcal L=-\log p_c$, i.e., the negative log-probability the model assigns to the correct class. It is also the maximum-likelihood loss for a categorical model, and equals the KL divergence between the target and predicted distributions up to a constant (the entropy of a one-hot target is zero); see [Week 1](week-01.html#math-refresher-iv-information-theory).

:::math The gradient of softmax cross-entropy
Write $\mathcal L=-\log p_c=-z_c+\log\sum_je^{z_j}$ (the second term is the *log-sum-exp*). Differentiate with respect to logit $z_k$:
$$
\frac{\partial\mathcal L}{\partial z_k}=-\mathbb 1[k=c]+\frac{e^{z_k}}{\sum_je^{z_j}}=p_k-y_k .
$$
In vector form, $\nabla_{\boldsymbol z}\mathcal L=\boldsymbol p-\boldsymbol y$. For a batch with mean reduction, $\partial\mathcal L/\partial\mathbf Z=\frac1B(\mathbf P-\mathbf Y)$.

*Reading the result.* The gradient is the *prediction error*: positive for wrongly favored classes (push their logits down), $p_c-1<0$ for the true class (push it up), and the components sum to zero (consistent with the shift-invariance of softmax). No saturation factor appears because the logarithm in the loss exactly cancels the exponential in the softmax.

*Via the Jacobian (a useful check).* The softmax Jacobian is $\partial p_i/\partial z_j=p_i(\delta_{ij}-p_j)$. Chain rule: $\partial\mathcal L/\partial z_j=\sum_i(-y_i/p_i)\,p_i(\delta_{ij}-p_j)=-y_j+p_j\sum_iy_i=p_j-y_j$. Same answer.
:::

```python:softmax_ce.py
import torch
import torch.nn.functional as F

torch.manual_seed(0)
B, K = 4, 5
z = torch.randn(B, K, requires_grad=True)         # logits
y = torch.tensor([0, 3, 1, 4])                    # class indices

# 1) the library loss (log-softmax + NLL, numerically stable, mean over batch)
loss = F.cross_entropy(z, y)
loss.backward()

# 2) by hand: gradient should be (softmax(z) - onehot(y)) / B
with torch.no_grad():
    p = torch.softmax(z, dim=1)
    onehot = F.one_hot(y, K).float()
    manual = (p - onehot) / B
print("max |autograd - (p - y)/B| =", (z.grad - manual).abs().max().item())   # ~1e-9
print("rows of gradient sum to zero:", z.grad.sum(dim=1))

# 3) why we never compute log(softmax(z)) naively
big = torch.tensor([[1000.0, 1001.0, 1002.0]])
print("naive softmax:", torch.exp(big) / torch.exp(big).sum())            # nan: inf / inf
print("stable       :", torch.softmax(big, dim=1))                         # [0.09, 0.245, 0.665]
print("log-softmax  :", F.log_softmax(big, dim=1))                         # finite
```

### Hinge loss and margins

Cross-entropy asks for calibrated probabilities. The **hinge loss** asks only for a *margin*: the correct class score should beat the others by at least a fixed amount, and once it does, the example contributes nothing. For binary labels $y\in\{-1,+1\}$ and score $s$,

$$
\mathcal L_{\text{hinge}}=\max(0,\,1-ys),\qquad \partial\mathcal L/\partial s=\begin{cases}-y&ys<1\\0&ys\ge1\end{cases}
$$

(a subgradient at the kink). This is the loss of the support vector machine. The multiclass version of Weston and Watkins (1999), the one popularized by Stanford's CS231n, sums violations over wrong classes: $\mathcal L=\sum_{j\ne c}\max(0,\,s_j-s_c+\Delta)$ with margin $\Delta$. The contrast with cross-entropy is instructive: hinge gradients are *exactly zero* for examples that are already classified with margin, so it focuses on hard examples and ignores easy ones; cross-entropy keeps pushing logits apart forever (its gradient never reaches 0), which tends to inflate weights and overconfidence unless regularized. Hinge outputs are not probabilities, so you lose calibration; deep learning therefore uses cross-entropy by default and reserves margin losses for metric learning and ranking (contrastive and triplet losses, [Week 8](week-08.html)).

### Focal loss: down-weighting the easy majority

In dense object detection a detector scores around $10^5$ candidate boxes per image, and nearly all are easy background. Their individual cross-entropy losses are tiny, but their *number* swamps the few hard positives. Lin, Goyal, Girshick, He and Dollár (2017) proposed the **focal loss** to fix this. Writing $p_t$ for the probability assigned to the true class,

$$
\mathrm{FL}(p_t)=-(1-p_t)^{\gamma}\log p_t,
$$

with a focusing parameter $\gamma\ge0$ (they found $\gamma=2$ to work well; an additional class-balancing weight $\alpha_t$ is often included). With $\gamma=0$ it is exactly cross-entropy. The factor $(1-p_t)^\gamma$ shrinks the loss of well-classified examples: at $p_t=0.9$ and $\gamma=2$ the loss is $100\times$ smaller than cross-entropy, while at $p_t=0.1$ it is only $0.81\times$ as large.

```chart
{
  "type": "line",
  "caption": "Focal loss -(1-p_t)^gamma log p_t versus the probability of the true class, for several gamma (computed with NumPy). gamma = 0 is ordinary cross-entropy; larger gamma suppresses the loss of confident, correct predictions.",
  "height": 340,
  "data": {"labels":[0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,0.2,0.21,0.22,0.23,0.24,0.25,0.26,0.27,0.28,0.29,0.3,0.31,0.32,0.33,0.34,0.35,0.36,0.37,0.38,0.39,0.4,0.41,0.42,0.43,0.44,0.45,0.46,0.47,0.48,0.49,0.5,0.51,0.52,0.53,0.54,0.55,0.56,0.57,0.58,0.59,0.6,0.61,0.62,0.63,0.64,0.65,0.66,0.67,0.68,0.69,0.7,0.71,0.72,0.73,0.74,0.75,0.76,0.77,0.78,0.79,0.8,0.81,0.82,0.83,0.84,0.85,0.86,0.87,0.88,0.89,0.9,0.91,0.92,0.93,0.94,0.95,0.96,0.97,0.98,0.99,1.0],"datasets":[{"label":"gamma = 0","data":[4.6052,3.912,3.5066,3.2189,2.9957,2.8134,2.6593,2.5257,2.4079,2.3026,2.2073,2.1203,2.0402,1.9661,1.8971,1.8326,1.772,1.7148,1.6607,1.6094,1.5606,1.5141,1.4697,1.4271,1.3863,1.3471,1.3093,1.273,1.2379,1.204,1.1712,1.1394,1.1087,1.0788,1.0498,1.0217,0.9943,0.9676,0.9416,0.9163,0.8916,0.8675,0.844,0.821,0.7985,0.7765,0.755,0.734,0.7133,0.6931,0.6733,0.6539,0.6349,0.6162,0.5978,0.5798,0.5621,0.5447,0.5276,0.5108,0.4943,0.478,0.462,0.4463,0.4308,0.4155,0.4005,0.3857,0.3711,0.3567,0.3425,0.3285,0.3147,0.3011,0.2877,0.2744,0.2614,0.2485,0.2357,0.2231,0.2107,0.1985,0.1863,0.1744,0.1625,0.1508,0.1393,0.1278,0.1165,0.1054,0.0943,0.0834,0.0726,0.0619,0.0513,0.0408,0.0305,0.0202,0.0101,-0.0],"borderColor":"var(--c-blue)","borderWidth":2},{"label":"gamma = 0.5","data":[4.5821,3.8727,3.4536,3.1538,2.9199,2.7277,2.5645,2.4226,2.297,2.1844,2.0823,1.989,1.903,1.8233,1.7491,1.6796,1.6143,1.5528,1.4947,1.4395,1.3871,1.3372,1.2896,1.2441,1.2006,1.1588,1.1187,1.0801,1.0431,1.0073,0.9729,0.9396,0.9075,0.8764,0.8464,0.8173,0.7892,0.7619,0.7354,0.7098,0.6848,0.6607,0.6372,0.6144,0.5922,0.5706,0.5497,0.5293,0.5094,0.4901,0.4713,0.4531,0.4353,0.4179,0.401,0.3846,0.3686,0.353,0.3378,0.3231,0.3087,0.2947,0.281,0.2678,0.2549,0.2423,0.2301,0.2182,0.2066,0.1954,0.1844,0.1738,0.1635,0.1535,0.1438,0.1344,0.1253,0.1165,0.108,0.0998,0.0919,0.0842,0.0768,0.0697,0.0629,0.0564,0.0502,0.0443,0.0386,0.0333,0.0283,0.0236,0.0192,0.0152,0.0115,0.0082,0.0053,0.0029,0.001,-0.0],"borderColor":"var(--c-green)","borderWidth":2},{"label":"gamma = 1","data":[4.5591,3.8338,3.4014,3.0901,2.8459,2.6446,2.4731,2.3237,2.1912,2.0723,1.9645,1.8658,1.775,1.6909,1.6126,1.5394,1.4707,1.4061,1.3452,1.2876,1.2329,1.181,1.1317,1.0846,1.0397,0.9968,0.9558,0.9165,0.8789,0.8428,0.8081,0.7748,0.7428,0.712,0.6824,0.6539,0.6264,0.5999,0.5744,0.5498,0.526,0.5032,0.4811,0.4597,0.4392,0.4193,0.4002,0.3817,0.3638,0.3466,0.3299,0.3139,0.2984,0.2834,0.269,0.2551,0.2417,0.2288,0.2163,0.2043,0.1928,0.1817,0.171,0.1607,0.1508,0.1413,0.1322,0.1234,0.115,0.107,0.0993,0.092,0.085,0.0783,0.0719,0.0659,0.0601,0.0547,0.0495,0.0446,0.04,0.0357,0.0317,0.0279,0.0244,0.0211,0.0181,0.0153,0.0128,0.0105,0.0085,0.0067,0.0051,0.0037,0.0026,0.0016,0.0009,0.0004,0.0001,-0.0],"borderColor":"var(--c-red)","borderWidth":2},{"label":"gamma = 2","data":[4.5135,3.7571,3.2993,2.9665,2.7036,2.4859,2.3,2.1378,1.994,1.8651,1.7484,1.6419,1.5442,1.4541,1.3707,1.2931,1.2207,1.153,1.0896,1.03,0.974,0.9212,0.8714,0.8243,0.7798,0.7377,0.6977,0.6599,0.624,0.5899,0.5576,0.5269,0.4977,0.4699,0.4435,0.4185,0.3946,0.3719,0.3504,0.3299,0.3104,0.2918,0.2742,0.2575,0.2415,0.2264,0.2121,0.1985,0.1855,0.1733,0.1617,0.1507,0.1402,0.1304,0.1211,0.1123,0.1039,0.0961,0.0887,0.0817,0.0752,0.069,0.0633,0.0578,0.0528,0.048,0.0436,0.0395,0.0357,0.0321,0.0288,0.0258,0.0229,0.0204,0.018,0.0158,0.0138,0.012,0.0104,0.0089,0.0076,0.0064,0.0054,0.0045,0.0037,0.003,0.0024,0.0018,0.0014,0.0011,0.0008,0.0005,0.0004,0.0002,0.0001,0.0001,0.0,0.0,0.0,-0.0],"borderColor":"var(--c-purple)","borderWidth":2},{"label":"gamma = 5","data":[4.3795,3.5362,3.0112,2.6246,2.318,2.0648,1.85,1.6647,1.5026,1.3597,1.2326,1.1189,1.0169,0.9249,0.8418,0.7664,0.698,0.6357,0.5791,0.5274,0.4802,0.4372,0.3978,0.3618,0.329,0.2989,0.2714,0.2463,0.2233,0.2024,0.1832,0.1657,0.1497,0.1351,0.1218,0.1097,0.0987,0.0886,0.0795,0.0713,0.0637,0.0569,0.0508,0.0452,0.0402,0.0357,0.0316,0.0279,0.0246,0.0217,0.019,0.0167,0.0146,0.0127,0.011,0.0096,0.0083,0.0071,0.0061,0.0052,0.0045,0.0038,0.0032,0.0027,0.0023,0.0019,0.0016,0.0013,0.0011,0.0009,0.0007,0.0006,0.0005,0.0004,0.0003,0.0002,0.0002,0.0001,0.0001,0.0001,0.0001,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-0.0],"borderColor":"var(--c-amber)","borderWidth":2}]},
  "options": {"scales": {"x": {"title": {"display": true, "text": "p_t (probability of the true class)"}}, "y": {"title": {"display": true, "text": "loss"}, "max": 4}}, "elements": {"point": {"radius": 0}}}
}
```

:::math Gradient of the binary focal loss with respect to the logit
For a positive example with $p_t=\sigma(z)$, use $\partial p_t/\partial z=p_t(1-p_t)$ and
$$
\frac{\partial\,\mathrm{FL}}{\partial p_t}=\gamma(1-p_t)^{\gamma-1}\log p_t-\frac{(1-p_t)^{\gamma}}{p_t}.
$$
Multiplying,
$$
\frac{\partial\,\mathrm{FL}}{\partial z}=(1-p_t)^{\gamma}\Big[\gamma\,p_t\log p_t-(1-p_t)\Big].
$$
For $\gamma=0$ this is $-(1-p_t)=p-y$, matching cross-entropy. For $\gamma>0$ the leading $(1-p_t)^\gamma$ factor makes the gradient vanish quickly as $p_t\to1$: easy examples stop contributing. (Negative examples follow by symmetry.)
:::

```python:losses.py
import torch
import torch.nn.functional as F

def focal_loss(logits, targets, gamma=2.0, alpha=None):
    """Multiclass focal loss. logits: (B, K); targets: (B,) class indices."""
    logp = F.log_softmax(logits, dim=1)                       # (B, K), stable
    logp_t = logp.gather(1, targets[:, None]).squeeze(1)      # log p_t, shape (B,)
    p_t = logp_t.exp()
    loss = -((1 - p_t) ** gamma) * logp_t                     # (B,)
    if alpha is not None:                                     # optional per-class weights (K,)
        loss = alpha[targets] * loss
    return loss.mean()

def multiclass_hinge(scores, targets, margin=1.0):
    """Weston-Watkins hinge. scores: (B, K); targets: (B,)."""
    correct = scores.gather(1, targets[:, None])              # (B, 1)
    viol = (scores - correct + margin).clamp(min=0)           # (B, K)
    viol = viol.scatter(1, targets[:, None], 0.0)             # zero out the correct class
    return viol.sum(dim=1).mean()

torch.manual_seed(0)
z = torch.randn(6, 4)
y = torch.randint(0, 4, (6,))
print("CE            :", F.cross_entropy(z, y).item())
print("focal gamma=0 :", focal_loss(z, y, gamma=0.0).item(), "(equals CE)")
print("focal gamma=2 :", focal_loss(z, y, gamma=2.0).item(), "(smaller: easy examples discounted)")
print("hinge         :", multiclass_hinge(z, y).item())
```

:::note Related tools
**Label smoothing** replaces the one-hot target by $(1-\varepsilon)\boldsymbol y+\varepsilon/K$, which caps the logit gaps and improves calibration; it is a one-argument change in PyTorch (`label_smoothing=0.1`). **Class weights** (`weight=` in `cross_entropy`) address imbalance more simply than focal loss and are the first thing to try. The loss functions for generative and self-supervised models appear in later weeks.
:::

## Computational graphs

Every differentiable program, however large, is a composition of small operations whose derivatives you already know: add, multiply, `exp`, `log`, matrix multiply, `max(0, x)`. A **computational graph** makes that composition explicit. Nodes are values (inputs, parameters, intermediate results, the final loss); edges say which values were used to compute which. The graph is a *directed acyclic graph* (DAG): you can order the nodes so that every node comes after its inputs, and that order is exactly the order of the forward pass.

Why draw it? Because it turns the chain rule into bookkeeping. To get the derivative of the loss with respect to any node, you walk the graph backwards and, at each node, multiply the incoming gradient by the node's **local derivative** (how its output changes with each input). Where two paths from a node to the loss merge, you **add** their contributions. That is all backpropagation is.

{{figure: week02-compgraph-scalar | The graph of a single sigmoid neuron with squared loss, evaluated at $x=1.5$, $w=0.8$, $b=-0.3$, target $t=1$. Black numbers are forward values, orange numbers are $\partial\mathcal L/\partial(\cdot)$ computed in the backward sweep. Addition passes the gradient through unchanged; multiplication swaps in the other operand.}}

Three local rules cover a surprising amount of deep learning:

| Operation | Forward | Backward (given upstream gradient $\bar{o}=\partial\mathcal L/\partial o$) |
| --- | --- | --- |
| Sum $o=u+v$ | $u+v$ | $\bar u=\bar o,\ \bar v=\bar o$ (copy) |
| Product $o=u\,v$ | $uv$ | $\bar u=\bar o\,v,\ \bar v=\bar o\,u$ (swap) |
| Fan-out (one value $u$ used by several nodes) | | $\bar u=\sum_k \bar{o}_k\,\partial o_k/\partial u$ (add) |
| Unary $o=\phi(u)$ | $\phi(u)$ | $\bar u=\bar o\,\phi'(u)$ |

Step through the example yourself. Change the sliders and watch which cached values the backward pass needs; this is the reason frameworks store intermediate activations, and why training needs far more memory than inference.

{{demo: week02-backprop-stepper | Step through the forward and backward pass of one neuron | 560}}

:::example Worked numbers
With $x=1.5,w=0.8,b=-0.3,t=1$: $p=xw=1.2$, $z=p+b=0.9$, $a=\sigma(0.9)=0.7109$, $\mathcal L=\tfrac12(a-t)^2=0.0418$. Backward: $\bar a=a-t=-0.2891$; $\sigma'(z)=a(1-a)=0.2055$ so $\bar z=-0.0594$; $\bar b=\bar p=\bar z=-0.0594$; $\bar w=x\bar p=-0.0891$; $\bar x=w\bar p=-0.0475$. A gradient step with $\eta=1$ moves $w$ to $0.889$: the neuron's output rises toward the target, as it should.
:::

## Backpropagation derived

We now derive backprop three times, at increasing generality: a scalar chain, a fully-connected layer for one example, and the batched matrix form you will implement in the lab.

### The scalar chain

Consider a "network" with one weight per layer: $a_0=x$, $z_\ell=w_\ell a_{\ell-1}$, $a_\ell=\phi(z_\ell)$ for $\ell=1,\dots,L$, and loss $\mathcal L=\ell(a_L,t)$. The chain rule gives

$$
\frac{\partial\mathcal L}{\partial w_k}
=\frac{\partial\mathcal L}{\partial a_L}\;
\underbrace{\prod_{\ell=k+1}^{L}\frac{\partial a_\ell}{\partial a_{\ell-1}}}_{\text{layers above }k}\;
\frac{\partial a_k}{\partial z_k}\frac{\partial z_k}{\partial w_k},
\qquad \frac{\partial a_\ell}{\partial a_{\ell-1}}=\phi'(z_\ell)\,w_\ell .
$$

Two things to notice. First, the gradient for the weight in layer $k$ contains a *product* of $L-k$ factors $\phi'(z_\ell)w_\ell$. If these are typically smaller than one the gradient vanishes exponentially with depth; if larger, it explodes. This single formula is the source of the vanishing/exploding gradient problem treated in [Week 3](week-03.html#vanishing-exploding). Second, the products for different $k$ **share** their tails. Computing each $\partial\mathcal L/\partial w_k$ from scratch would cost $O(L^2)$ multiplications; defining

$$
\delta_k \;\equiv\; \frac{\partial\mathcal L}{\partial z_k},\qquad
\delta_L=\frac{\partial\mathcal L}{\partial a_L}\phi'(z_L),\qquad
\delta_{k}=\delta_{k+1}\,w_{k+1}\,\phi'(z_k),
$$

reuses the running product, so all $L$ gradients cost $O(L)$ in total, and $\partial\mathcal L/\partial w_k=\delta_k\,a_{k-1}$. **Backpropagation is dynamic programming on the chain rule**: cache the forward values, sweep backward, reuse $\delta$.

### One layer, one example

For a layer $\boldsymbol z=\mathbf W^\top\boldsymbol a_{\rm in}+\boldsymbol b$, $\boldsymbol a=\phi(\boldsymbol z)$, with $\mathbf W\in\R^{n_{\rm in}\times n_{\rm out}}$, suppose the upstream gradient $\bar{\boldsymbol a}=\partial\mathcal L/\partial\boldsymbol a\in\R^{n_{\rm out}}$ is known. We want $\bar{\boldsymbol z}$, $\bar{\mathbf W}$, $\bar{\boldsymbol b}$ and $\bar{\boldsymbol a}_{\rm in}$ (to pass on).

:::math Index-by-index derivation
**Through the activation.** $a_j=\phi(z_j)$ depends only on $z_j$, so $\bar z_j=\bar a_j\,\phi'(z_j)$, i.e. $\bar{\boldsymbol z}=\bar{\boldsymbol a}\odot\phi'(\boldsymbol z)$ (elementwise product; the Jacobian is diagonal).

**Weights.** $z_j=\sum_i W_{ij}\,a_{{\rm in},i}+b_j$, so $\partial z_j/\partial W_{ij}=a_{{\rm in},i}$ and $\partial z_{j'}/\partial W_{ij}=0$ for $j'\ne j$. Hence
$$
\bar W_{ij}=\bar z_j\,a_{{\rm in},i}\quad\Longleftrightarrow\quad \bar{\mathbf W}=\boldsymbol a_{\rm in}\,\bar{\boldsymbol z}^\top\ \ (\text{an outer product}).
$$
**Bias.** $\partial z_j/\partial b_j=1$, so $\bar{\boldsymbol b}=\bar{\boldsymbol z}$.

**Inputs.** $a_{{\rm in},i}$ influences every $z_j$ with weight $W_{ij}$, so the contributions add (fan-out rule):
$$
\bar a_{{\rm in},i}=\sum_j \bar z_j\,W_{ij}\quad\Longleftrightarrow\quad \bar{\boldsymbol a}_{\rm in}=\mathbf W\,\bar{\boldsymbol z}.
$$
:::

A useful memory aid: **a gradient always has the shape of the thing it is the gradient of.** $\bar{\mathbf W}$ is $n_{\rm in}\times n_{\rm out}$, so it must be built from $\boldsymbol a_{\rm in}$ ($n_{\rm in}$) and $\bar{\boldsymbol z}$ ($n_{\rm out}$); there is only one way to do that with an outer product.

### The batched, vectorized form

With a batch $\mathbf A_{\rm in}\in\R^{B\times n_{\rm in}}$ the forward pass is $\mathbf Z=\mathbf A_{\rm in}\mathbf W+\boldsymbol 1\boldsymbol b^\top$, $\mathbf A=\phi(\mathbf Z)$. The loss is the **mean** of per-example losses, $\mathcal L=\frac1B\sum_{n}\mathcal L_n$. Each example contributes the outer product above, and parameter gradients sum over examples:

$$
\bar{\mathbf Z}=\bar{\mathbf A}\odot\phi'(\mathbf Z),\qquad
\bar{\mathbf W}=\mathbf A_{\rm in}^\top\,\bar{\mathbf Z},\qquad
\bar{\boldsymbol b}=\mathbf 1^\top\bar{\mathbf Z}\ \ (\text{sum over the batch axis}),\qquad
\bar{\mathbf A}_{\rm in}=\bar{\mathbf Z}\,\mathbf W^\top .
$$

The matrix product $\mathbf A_{\rm in}^\top\bar{\mathbf Z}$ *is* the sum over examples of outer products, which is why one GEMM does all the work. The factor $1/B$ enters once, in the seed gradient at the output (for softmax cross-entropy: $\bar{\mathbf Z}^{(L)}=(\mathbf P-\mathbf Y)/B$).

{{figure: week02-backprop-layer | One layer in the batched row convention. The forward pass caches $\mathbf A^{(\ell-1)}$ and $\mathbf Z^{(\ell)}$; the backward pass turns $\bar{\mathbf A}^{(\ell)}$ into $\bar{\mathbf Z}$, the parameter gradients, and $\bar{\mathbf A}^{(\ell-1)}$ for the layer below.}}

### The complete algorithm

:::definition Backpropagation for an MLP
**Forward:** $\mathbf A^{(0)}=\mathbf X$; for $\ell=1..L$: $\mathbf Z^{(\ell)}=\mathbf A^{(\ell-1)}\mathbf W^{(\ell)}+\boldsymbol b^{(\ell)}$, $\mathbf A^{(\ell)}=\phi^{(\ell)}(\mathbf Z^{(\ell)})$; compute $\mathcal L$. Cache every $\mathbf A$ and $\mathbf Z$.

**Backward:** $\bar{\mathbf Z}^{(L)}=\partial\mathcal L/\partial\mathbf Z^{(L)}$ (for softmax-CE: $(\mathbf P-\mathbf Y)/B$). For $\ell=L..1$:
1. $\bar{\mathbf W}^{(\ell)}=\mathbf A^{(\ell-1)\top}\bar{\mathbf Z}^{(\ell)}$, $\ \bar{\boldsymbol b}^{(\ell)}=\mathbf 1^\top\bar{\mathbf Z}^{(\ell)}$
2. if $\ell>1$: $\bar{\mathbf A}^{(\ell-1)}=\bar{\mathbf Z}^{(\ell)}\mathbf W^{(\ell)\top}$, then $\bar{\mathbf Z}^{(\ell-1)}=\bar{\mathbf A}^{(\ell-1)}\odot\phi'(\mathbf Z^{(\ell-1)})$.

**Cost:** the backward pass does two matrix products per layer (for $\bar{\mathbf W}$ and $\bar{\mathbf A}$) against one in the forward pass, so a full training step costs about **3x** a forward pass in FLOPs. **Memory:** all activations must be kept until used, so memory grows with depth and batch size.
:::

:::history Who invented backpropagation?
The reverse-mode idea (accumulate derivatives backward through a computation) was published by Seppo Linnainmaa in his 1970 master's thesis on rounding error, and applied to control and network-like systems by others in the 1960s and 70s (e.g. Paul Werbos proposed using it for neural networks in his 1974 thesis). It became central to the field after Rumelhart, Hinton and Williams showed in 1986 (*Nature*) that networks trained this way learn useful internal representations. LeCun's 1989 handwritten zip-code network made it practical. Credit is shared, and the history is genuinely contested; Schmidhuber's survey in the reading list gives one detailed account.
:::

## Automatic differentiation

Backprop as we derived it is a special case of a general technique: **automatic differentiation (AD)** computes exact derivatives of a program by applying the chain rule to its elementary operations. It is not numerical differentiation (finite differences, which are approximate and need one evaluation per parameter) and not symbolic differentiation (which manipulates formulas and can blow up in size). AD gives machine-precision derivatives at a small constant factor over the cost of the function itself.

### Forward mode versus reverse mode

Let $f:\R^n\to\R^m$ have Jacobian $\mathbf J\in\R^{m\times n}$. Evaluating the chain rule for a composition $f=f_K\circ\cdots\circ f_1$ means multiplying Jacobians, $\mathbf J=\mathbf J_K\cdots\mathbf J_1$, and the only freedom is the *order of multiplication*.

- **Forward mode** pushes a tangent vector $\boldsymbol v\in\R^n$ along with the computation, multiplying right to left: it returns the **Jacobian-vector product** $\mathbf J\boldsymbol v$ in one pass. A full Jacobian needs $n$ passes (one per input direction).
- **Reverse mode** first runs the forward pass, then pulls a cotangent $\boldsymbol u\in\R^m$ back, multiplying left to right: it returns the **vector-Jacobian product** $\boldsymbol u^\top\mathbf J$ in one pass. A full Jacobian needs $m$ passes.

A loss has $m=1$ and the parameters have $n$ in the millions. One reverse pass with $\boldsymbol u=1$ yields the entire gradient; forward mode would need $n$ passes. That asymmetry is why reverse mode (backprop) is the engine of deep learning. Forward mode wins when inputs are few and outputs many, and it needs no stored activations; it is used for Jacobian-vector products in some second-order and sensitivity methods.

{{figure: week02-autodiff-modes | What each AD mode extracts from the Jacobian per pass. Forward mode: one column ($\mathbf J\boldsymbol v$). Reverse mode: one row ($\boldsymbol u^\top\mathbf J$). For a scalar loss, reverse mode produces the full gradient in a single pass at the price of storing intermediates.}}

To make forward mode concrete, here is a tiny **dual number** implementation: each value carries its derivative with it, and every operation updates both by the chain rule.

```python:dual_numbers.py
import math

class Dual:
    """value + eps * derivative, with eps**2 = 0. Forward-mode AD in a dozen lines."""
    def __init__(self, v, d=0.0): self.v, self.d = v, d
    def __add__(s, o): o = o if isinstance(o, Dual) else Dual(o); return Dual(s.v + o.v, s.d + o.d)
    __radd__ = __add__
    def __mul__(s, o): o = o if isinstance(o, Dual) else Dual(o); return Dual(s.v * o.v, s.d * o.v + s.v * o.d)
    __rmul__ = __mul__
    def __sub__(s, o): o = o if isinstance(o, Dual) else Dual(o); return Dual(s.v - o.v, s.d - o.d)
    def __rtruediv__(s, o): return Dual(o) / s
    def __truediv__(s, o): o = o if isinstance(o, Dual) else Dual(o); return Dual(s.v / o.v, (s.d * o.v - s.v * o.d) / o.v**2)

def exp(x): return Dual(math.exp(x.v), math.exp(x.v) * x.d)
def sigmoid(x): return 1 / (1 + exp(Dual(0) - x))

# d/dw of  L = 0.5 * (sigmoid(x*w + b) - t)^2  at the numbers used in the figure
x, b, t = 1.5, -0.3, 1.0
w = Dual(0.8, 1.0)                    # seed: dw/dw = 1
a = sigmoid(x * w + b)
L = (a - t) * (a - t) * 0.5
print(f"L = {L.v:.4f}, dL/dw = {L.d:.4f}")     # L = 0.0418, dL/dw = -0.0891 (matches the graph)
```

One forward pass gave one directional derivative. To get gradients for 100 million parameters this way you would run it 100 million times; reverse mode does it in one sweep.

### How PyTorch autograd works

PyTorch implements **define-by-run** (dynamic) reverse-mode AD. When you run a forward computation on tensors with `requires_grad=True`, each operation does two things: it computes the result, and it records a **`grad_fn`** node pointing to the nodes that produced its inputs. The graph is built *as the code executes*, so ordinary Python control flow (loops, `if`) just works, and a different graph can be built every iteration.

Calling `loss.backward()` walks this graph from the loss in reverse topological order. Each `grad_fn` knows how to compute a vector-Jacobian product for its operation (it saved whatever tensors it needs from the forward pass). Gradients arriving at the same leaf from several paths are **accumulated** into its `.grad` attribute. After `backward()`, the graph's saved buffers are freed (unless `retain_graph=True`).

```python:autograd_inspect.py
import torch

x = torch.tensor(1.5)
w = torch.tensor(0.8, requires_grad=True)
b = torch.tensor(-0.3, requires_grad=True)
t = torch.tensor(1.0)

p = x * w
z = p + b
a = torch.sigmoid(z)
loss = 0.5 * (a - t) ** 2

# The graph is a linked list of grad_fn nodes, from the output back to the leaves
node = loss.grad_fn
while node is not None:
    print(type(node).__name__)
    node = node.next_functions[0][0] if node.next_functions else None

loss.backward()
print("dL/dw =", w.grad.item(), " dL/db =", b.grad.item())   # -0.0891, -0.0594
print("x.grad (not a leaf requiring grad):", x.grad)          # None: x.requires_grad is False

# Gradients ACCUMULATE: a second backward on a fresh graph adds to .grad
loss2 = 0.5 * (torch.sigmoid(x * w + b) - t) ** 2
loss2.backward()
print("after second backward, dL/dw =", w.grad.item())        # doubled: -0.1782
w.grad.zero_()                                                # this is why optimizers call zero_grad()
```

Details worth knowing:

- **Leaf tensors** (created by you, e.g. parameters) get a `.grad`; intermediate tensors do not unless you call `.retain_grad()`.
- `with torch.no_grad():` and `tensor.detach()` stop recording. Use them for evaluation and for parts of the computation you do not want differentiated; they also save memory.
- In-place operations (`x.relu_()`, `x += 1`) can overwrite values the backward pass needs; autograd tracks a version counter and raises an error if a saved tensor was modified.
- `torch.autograd.grad(outputs, inputs, create_graph=True)` returns gradients that are themselves differentiable, giving higher-order derivatives (Hessian-vector products, gradient penalties).
- `loss.backward()` on a non-scalar needs an explicit `gradient=` argument: it is the cotangent $\boldsymbol u$ in the VJP.

### Writing a custom autograd Function

Sometimes you need an operation whose backward is not the automatic one: a numerically better formula, a non-differentiable op with a surrogate gradient (straight-through estimators), or a fused kernel. Subclass `torch.autograd.Function` and implement `forward` and `backward`; `backward` receives the upstream gradient and must return one gradient per `forward` input.

```python:custom_function.py
import torch

class MyLinear(torch.autograd.Function):
    """y = x @ W^T + b with a hand-written backward (same math as the batched derivation)."""
    @staticmethod
    def forward(ctx, x, W, b):                  # x: (B, in), W: (out, in), b: (out,)
        ctx.save_for_backward(x, W)             # cache what backward will need
        return x @ W.t() + b

    @staticmethod
    def backward(ctx, grad_y):                  # grad_y: (B, out)
        x, W = ctx.saved_tensors
        grad_x = grad_y @ W                     # (B, in)
        grad_W = grad_y.t() @ x                 # (out, in)
        grad_b = grad_y.sum(dim=0)              # (out,)
        return grad_x, grad_W, grad_b

class ReLUSTE(torch.autograd.Function):
    """Hard threshold forward, 'straight-through' identity gradient backward."""
    @staticmethod
    def forward(ctx, x): return (x > 0).to(x.dtype)
    @staticmethod
    def backward(ctx, g): return g

# gradcheck compares the analytic backward to finite differences; it needs float64
x = torch.randn(4, 3, dtype=torch.double, requires_grad=True)
W = torch.randn(2, 3, dtype=torch.double, requires_grad=True)
b = torch.randn(2, dtype=torch.double, requires_grad=True)
print("gradcheck MyLinear:", torch.autograd.gradcheck(MyLinear.apply, (x, W, b)))   # True

y = ReLUSTE.apply(torch.randn(5, requires_grad=True))
y.sum().backward()                              # runs without error: gradient flows through the step
```

:::warning Gradient check precision
`gradcheck` uses finite differences, so always use `torch.double`. In float32 it will often fail even for a correct backward (the chart in the lab shows why).
:::

## Lab: a neural network from scratch

Time to build it. The lab has three parts: (1) a NumPy MLP whose backward pass is the algorithm above, (2) a **numerical gradient check** that proves the backward pass is right, (3) training on MNIST and comparing with the same network in PyTorch. Do the NumPy part by hand at least once: nothing teaches backprop like debugging your own.

### Part 1: the network

```python:mlp_numpy.py
import numpy as np

class MLP:
    """Fully connected net: ReLU hidden layers, softmax cross-entropy output. Row convention (B, features)."""
    def __init__(self, sizes, seed=0):
        rng = np.random.default_rng(seed)
        self.W = [rng.normal(0, np.sqrt(2.0 / m), (m, n)) for m, n in zip(sizes[:-1], sizes[1:])]  # He init
        self.b = [np.zeros(n) for n in sizes[1:]]

    def forward(self, X):
        self.A, self.Z = [X], []
        for l, (W, b) in enumerate(zip(self.W, self.b)):
            Z = self.A[-1] @ W + b
            self.Z.append(Z)
            self.A.append(np.maximum(Z, 0) if l < len(self.W) - 1 else Z)   # last layer: logits
        return self.A[-1]

    def loss(self, X, y):
        logits = self.forward(X)
        logits = logits - logits.max(axis=1, keepdims=True)                 # stability
        logp = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))   # log-softmax
        return -logp[np.arange(len(y)), y].mean()

    def backward(self, y):
        B = len(y)
        P = np.exp(self.A[-1] - self.A[-1].max(axis=1, keepdims=True))
        P /= P.sum(axis=1, keepdims=True)
        dZ = P; dZ[np.arange(B), y] -= 1; dZ /= B                           # (P - Y) / B
        gW, gb = [None] * len(self.W), [None] * len(self.W)
        for l in reversed(range(len(self.W))):
            gW[l] = self.A[l].T @ dZ
            gb[l] = dZ.sum(axis=0)
            if l > 0:
                dZ = (dZ @ self.W[l].T) * (self.Z[l - 1] > 0)               # through ReLU
        return gW, gb

def grad_check(net, X, y, eps=1e-5, n_checks=20, seed=0):
    net.loss(X, y); gW, gb = net.backward(y)
    rng = np.random.default_rng(seed)
    worst = 0.0
    for params, grads in ((net.W, gW), (net.b, gb)):
        for p, g in zip(params, grads):
            for _ in range(n_checks):
                idx = tuple(rng.integers(0, s) for s in p.shape)
                old = p[idx]
                p[idx] = old + eps; lp = net.loss(X, y)
                p[idx] = old - eps; lm = net.loss(X, y)
                p[idx] = old
                num = (lp - lm) / (2 * eps)
                rel = abs(num - g[idx]) / max(1e-8, abs(num) + abs(g[idx]))
                worst = max(worst, rel)
    return worst

def train(net, X, y, Xt, yt, epochs=5, lr=0.1, bs=128, seed=0):
    rng = np.random.default_rng(seed)
    for ep in range(epochs):
        perm = rng.permutation(len(X)); tot = 0.0
        for i in range(0, len(X), bs):
            idx = perm[i:i + bs]
            tot += net.loss(X[idx], y[idx]) * len(idx)
            gW, gb = net.backward(y[idx])
            for l in range(len(net.W)):
                net.W[l] -= lr * gW[l]; net.b[l] -= lr * gb[l]
        acc = (net.forward(Xt).argmax(1) == yt).mean()
        print(f"epoch {ep+1}: train loss {tot/len(X):.4f}  test acc {acc:.4f}")
```

Read `backward` next to the boxed algorithm: `dZ` starts as $(\mathbf P-\mathbf Y)/B$; each iteration forms `gW = A[l].T @ dZ`, `gb = dZ.sum(0)`, then moves the gradient down with `dZ @ W.T` and multiplies by the ReLU mask `(Z > 0)`. Note that `forward` caches `A` and `Z` on the object; `backward` is only valid right after the `forward` it belongs to.

### Part 2: gradient checking

If your backward is wrong, training will not crash; it will quietly do worse. The **centered finite difference** $\frac{\mathcal L(\theta_i+\varepsilon)-\mathcal L(\theta_i-\varepsilon)}{2\varepsilon}$ approximates $\partial\mathcal L/\partial\theta_i$ with error $O(\varepsilon^2)$ (the one-sided version is only $O(\varepsilon)$). Compare it with your analytic gradient through a **relative error**, never an absolute one, because gradients can be tiny.

```python:gradient_check.py
rng = np.random.default_rng(1)
X = rng.normal(size=(32, 20)); y = rng.integers(0, 3, 32)        # tiny fake batch
net = MLP([20, 32, 16, 3])
worst = grad_check(net, X, y, eps=1e-5)
print(f"worst relative error over sampled parameters: {worst:.2e}")   # expect < 1e-6
```

A rough guide for relative error in float64: below $10^{-7}$ is right, $10^{-7}$ to $10^{-4}$ deserves a look, above $10^{-3}$ is a bug. On our run the worst error over 20 sampled entries per parameter array was about $10^{-8}$. Why not always use $\varepsilon$ as small as possible? Because of floating-point cancellation, as the next chart shows.

```chart
{"type":"line","caption":"Gradient-check error as a function of the finite-difference step h, for a 5-8-3 tanh network with softmax cross-entropy, 16 examples (computed with NumPy, relative error of the whole gradient vector against the analytic backward). Float64 is a U shape with a floor near 5e-11: truncation error falls with h, rounding error grows. Float32 never gets below about 3e-5 and is useless for h below 1e-3.","height":340,"data":{"labels":["1e-01","3e-02","1e-02","3e-03","1e-03","3e-04","1e-04","3e-05","1e-05","3e-06","1e-06","3e-07","1e-07","3e-08","1e-08","3e-09","1e-09","3e-10","1e-10","3e-11","1e-11","3e-12","1e-12"],"datasets":[{"label":"float64","data":[0.002862,0.0002938,2.945e-05,2.946e-06,2.946e-07,2.946e-08,2.947e-09,2.962e-10,5.146e-11,1.351e-10,4.191e-10,1.517e-09,4.512e-09,1.445e-08,4.608e-08,1.344e-07,4.285e-07,1.28e-06,4.654e-06,1.393e-05,5.049e-05,0.0001538,0.0004606],"borderColor":"var(--c-blue)","borderWidth":2},{"label":"float32","data":[0.002697,0.0002727,3.828e-05,8.07e-05,0.0002238,0.0007356,0.002598,0.008405,0.02894,0.0875,0.2739,0.5164,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0],"borderColor":"var(--c-green)","borderWidth":2}]},"options":{"scales":{"x":{"title":{"display":true,"text":"step size h"}},"y":{"type":"logarithmic","title":{"display":true,"text":"relative error"}}},"elements":{"point":{"radius":2}}}}
```

Check gradients on a **tiny** network and batch, with float64, and remove stochastic layers (dropout) while checking. Kinks matter too: if a pre-activation sits within $\varepsilon$ of zero, the ReLU finite difference straddles the kink and disagrees with the (valid) subgradient. A single disagreeing coordinate in a ReLU net is not necessarily a bug; a consistent disagreement is.

### Part 3: train on MNIST

MNIST has 60,000 training and 10,000 test images of 28x28 digits. We flatten, scale to $[0,1]$ and standardize with the training mean and standard deviation.

```python:mnist_numpy.py
import numpy as np
from torchvision import datasets            # only used to download MNIST; the model is pure NumPy

tr = datasets.MNIST("data", train=True, download=True)
te = datasets.MNIST("data", train=False, download=True)
Xtr = tr.data.numpy().reshape(-1, 784).astype(np.float64) / 255.0
Xte = te.data.numpy().reshape(-1, 784).astype(np.float64) / 255.0
ytr, yte = tr.targets.numpy(), te.targets.numpy()
mu, sd = Xtr.mean(), Xtr.std()               # scalar statistics; (x - mu) / sd
Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
print(Xtr.shape, ytr.shape, Xtr.mean().round(3), Xtr.std().round(3))   # (60000, 784) (60000,) 0.0 1.0

net = MLP([784, 128, 64, 10], seed=0)
print("grad check:", grad_check(net, Xtr[:16], ytr[:16]))            # always check before training
train(net, Xtr, ytr, Xte, yte, epochs=5, lr=0.1, bs=128)
```

A 784-128-64-10 network trained for five epochs of mini-batch SGD at learning rate 0.1 typically reaches about 97 percent test accuracy (expect roughly 96.5 to 98; it depends on the seed). If you see 10 percent (chance), your gradients or learning rate are broken; if the loss prints `nan`, lower the learning rate and check your softmax stability.

### Part 4: the same network in PyTorch

We now reproduce the model with `nn.Linear`, copy in the *same* initial weights, and verify that PyTorch's autograd returns the gradients you computed by hand. Matching to about $10^{-12}$ in float64 means your derivation and the framework agree exactly.

```python:mnist_pytorch.py
import numpy as np, torch
from torch import nn

# reuse `net` (NumPy, freshly initialized) and Xtr, ytr from above
torch_model = nn.Sequential(nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 10)).double()
with torch.no_grad():
    for lin, W, b in zip([torch_model[0], torch_model[2], torch_model[4]], net.W, net.b):
        lin.weight.copy_(torch.from_numpy(W.T))          # nn.Linear stores (out, in): transpose ours
        lin.bias.copy_(torch.from_numpy(b))

xb = torch.from_numpy(Xtr[:128]); yb = torch.from_numpy(ytr[:128])
loss = nn.functional.cross_entropy(torch_model(xb), yb)   # mean reduction, like our 1/B
loss.backward()

net_loss = net.loss(Xtr[:128], ytr[:128]); gW, gb = net.backward(ytr[:128])
print("loss  numpy vs torch:", net_loss, loss.item())
for i, lin in enumerate([torch_model[0], torch_model[2], torch_model[4]]):
    err = np.abs(lin.weight.grad.numpy() - gW[i].T).max()
    print(f"layer {i+1}: max |grad_torch - grad_numpy| = {err:.2e}")     # ~1e-15 to 1e-12

# Now the idiomatic training loop (float32, GPU if available)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = nn.Sequential(nn.Flatten(), nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 10)).to(device)
opt = torch.optim.SGD(model.parameters(), lr=0.1)
Xt = torch.tensor(Xtr, dtype=torch.float32, device=device); Yt = torch.tensor(ytr, device=device)
Xv = torch.tensor(Xte, dtype=torch.float32, device=device); Yv = torch.tensor(yte, device=device)

for epoch in range(5):
    perm = torch.randperm(len(Xt), device=device)
    for i in range(0, len(Xt), 128):
        idx = perm[i:i + 128]
        opt.zero_grad()                                   # grads accumulate: reset every step
        loss = nn.functional.cross_entropy(model(Xt[idx]), Yt[idx])
        loss.backward()
        opt.step()
    with torch.no_grad():
        acc = (model(Xv).argmax(1) == Yv).float().mean().item()
    print(f"epoch {epoch+1}: test acc {acc:.4f}")
```

:::exercise Lab extensions
1. Add a momentum term to your NumPy `train` and compare speed of convergence. 2. Swap ReLU for tanh in your class (change both `forward` and the mask in `backward`) and re-run the gradient check. 3. Add L2 weight decay and verify the gradient check still passes. 4. Time your NumPy epoch against PyTorch on CPU.
:::

### Gradient flow in deep networks

The scalar-chain formula said gradients are products of per-layer factors. We can measure that directly.

```chart
{"type":"bar","caption":"Frobenius norm of the weight gradient in each layer of a 10-layer, width-100 MLP at initialization (computed with NumPy; batch 64 of Gaussian inputs, Gaussian upstream gradient at the output; weight variance 1/n for sigmoid and tanh, 2/n for ReLU). Sigmoid gradients shrink by roughly a factor of 2 to 5 per layer going down (log scale); tanh and ReLU stay roughly flat. One seed, illustrative.","height":340,"data":{"labels":["layer 1","layer 2","layer 3","layer 4","layer 5","layer 6","layer 7","layer 8","layer 9","layer 10"],"datasets":[{"label":"sigmoid","data":[4.49e-05,0.000105,0.0004191,0.001785,0.007271,0.03389,0.129,0.5444,2.576,11.66],"borderColor":"var(--c-blue)","backgroundColor":"var(--c-blue)"},{"label":"tanh","data":[26.46,26.83,24.77,24.21,23.97,23.1,22.97,22.93,22.31,22.46],"borderColor":"var(--c-green)","backgroundColor":"var(--c-green)"},{"label":"relu","data":[72.47,71.23,69.11,68.56,68.83,68.35,72.66,77.78,64.81,79.06],"borderColor":"var(--c-red)","backgroundColor":"var(--c-red)"}]},"options":{"scales":{"x":{"title":{"display":true,"text":"layer (1 = closest to the input)"}},"y":{"type":"logarithmic","title":{"display":true,"text":"gradient norm"}}}}}
```

This is the quantitative face of the vanishing gradient problem: the sigmoid network's first layer receives a gradient about five orders of magnitude smaller than its last layer, so with a shared learning rate the bottom layers barely move. Normalization layers, residual connections and careful initialization (all of [Week 3](week-03.html#initialization) and later) are the fixes.

## Common mistakes and debugging tips

| Symptom | Likely cause | What to do |
| --- | --- | --- |
| Loss stuck at $\log K$ (about 2.303 for 10 classes) | Network outputs near-uniform probabilities: learning rate too small, dead network, or labels shuffled relative to inputs | Overfit 10 examples first; print gradient norms; check data/label alignment |
| Loss becomes `nan` or `inf` | Learning rate too high; `log(0)`; `exp` overflow in a hand-written softmax | Lower lr; use `log_softmax`/`cross_entropy` on logits; add gradient clipping |
| Training accuracy at chance from the start | Applied `softmax` before `CrossEntropyLoss` (double softmax) or forgot to remove sigmoid/softmax from the last layer | Output **raw logits** to `cross_entropy` and `BCEWithLogitsLoss` |
| Gradients do not change between steps, or grow unboundedly | Forgot `opt.zero_grad()` | Zero (or `set_to_none`) every iteration before `backward()` |
| Loss decreases on train, test accuracy random | Forgot `model.eval()` / `torch.no_grad()` (matters once dropout/BatchNorm appear), or different preprocessing at test time | Reuse exactly the train-time normalization statistics |
| Many units output exactly zero (ReLU) | Dead ReLUs from a large learning rate or a large negative bias | Lower lr, He init, try Leaky ReLU, monitor the fraction of zero activations |
| Finite-difference check fails by a large factor | Wrong backward, or float32, or $\varepsilon$ too small, or a ReLU kink | Use float64, $\varepsilon\approx10^{-5}$, test a tiny net layer by layer |
| `RuntimeError: ... modified by an inplace operation` | In-place op overwrote a tensor needed by backward | Replace `x += y` / `relu_()` by out-of-place versions |
| `Trying to backward through the graph a second time` | Reused a graph whose buffers were freed (e.g. an RNN state kept across iterations) | `.detach()` the carried state, or `retain_graph=True` only if you truly need it |
| Memory blows up over iterations | Accumulating `loss` tensors (keeps graphs alive) | Accumulate `loss.item()` instead |

The most valuable single habit: **overfit a tiny batch** (say 16 to 64 examples) before any real training. A correct model with a correct loss should drive its training loss to nearly zero. If it cannot, the bug is in the model, loss or data wiring, not in your hyperparameters.

## Exercises

:::exercise Exercise 1: Parameter count and shapes
A network maps 3x32x32 images (flattened) through hidden layers of width 512 and 256 to 10 classes. (a) How many parameters? (b) For a batch of 64, give the shape of every $\mathbf Z^{(\ell)}$, $\bar{\mathbf W}^{(\ell)}$ and $\bar{\boldsymbol b}^{(\ell)}$.
:::details Solution
Input size 3072. Layer 1: $3072\cdot512+512=1{,}573{,}376$. Layer 2: $512\cdot256+256=131{,}328$. Layer 3: $256\cdot10+10=2{,}570$. Total $1{,}707{,}274$. Shapes: $\mathbf Z^{(1)}$ (64,512), $\mathbf Z^{(2)}$ (64,256), $\mathbf Z^{(3)}$ (64,10); $\bar{\mathbf W}$: (3072,512), (512,256), (256,10); $\bar{\boldsymbol b}$: (512,), (256,), (10,).
:::
:::

:::exercise Exercise 2: Softmax cross-entropy gradient
Derive $\partial\mathcal L/\partial z_k=p_k-y_k$ for $\mathcal L=-\sum_j y_j\log p_j$, $p=\softmax(\boldsymbol z)$, one-hot $\boldsymbol y$.
:::details Solution
$\mathcal L=-z_c+\log\sum_j e^{z_j}$ where $c$ is the true class. Differentiate: $\partial\mathcal L/\partial z_k=-\mathbb 1[k=c]+e^{z_k}/\sum_j e^{z_j}=p_k-y_k$. (Equivalently use the softmax Jacobian $\partial p_j/\partial z_k=p_j(\delta_{jk}-p_k)$ and $\sum_j y_j=1$.)
:::
:::

:::exercise Exercise 3: Backprop by hand
Let $\boldsymbol x=(1,2)$, $\mathbf W=\begin{pmatrix}1&-1\\0&2\end{pmatrix}$ (row $i$ = input $i$), $\boldsymbol b=(0,0)$, $\boldsymbol a=\relu(\boldsymbol x^\top\mathbf W)$, loss $\mathcal L=\tfrac12\norm{\boldsymbol a-\boldsymbol t}^2$ with $\boldsymbol t=(1,1)$. Compute $\partial\mathcal L/\partial\mathbf W$.
:::details Solution
$\boldsymbol z=\boldsymbol x^\top\mathbf W=(1\cdot1+2\cdot0,\ 1\cdot(-1)+2\cdot2)=(1,3)$, $\boldsymbol a=(1,3)$. $\bar{\boldsymbol a}=\boldsymbol a-\boldsymbol t=(0,2)$; ReLU mask $(1,1)$ so $\bar{\boldsymbol z}=(0,2)$. $\bar{\mathbf W}=\boldsymbol x\bar{\boldsymbol z}^\top=\begin{pmatrix}0&2\\0&4\end{pmatrix}$.
:::
:::

:::exercise Exercise 4: Dead network
Initialize all weights and biases of an MLP to zero (tanh hidden units, softmax output). What happens during gradient descent? What if only the biases are zero?
:::details Solution
All hidden units in a layer compute the same function and receive the same gradient (by symmetry), so they stay identical forever: a layer of width $n$ behaves like one unit. In fact with zero weights the gradient into earlier layers is exactly zero (it is multiplied by $\mathbf W^\top=0$), so only the last layer's bias moves at first. Zero biases alone are fine: random weights already break symmetry, and the biases then learn like any other parameter.
:::
:::

:::exercise Exercise 5: Why not finite differences for training?
For a network with $10^8$ parameters, estimate the cost of one gradient by centered finite differences versus backprop, in units of one forward pass.
:::details Solution
Centered differences need two forward passes per parameter: $2\times10^8$ forward passes. Backprop needs about 3 forward-pass equivalents in total (one forward, roughly two for the backward). The ratio is about $7\times10^7$. Finite differences are also inexact; they are used only to *test* backprop on small models.
:::
:::

:::exercise Exercise 6: Backward of vector normalization
Let $y=\boldsymbol x/\norm{\boldsymbol x}_2$ for $\boldsymbol x\in\R^n$. Derive the VJP: given $\bar{\boldsymbol y}$, find $\bar{\boldsymbol x}$, and check it is orthogonal to $\boldsymbol x$.
:::details Solution
Let $r=\norm{\boldsymbol x}$. $\partial y_i/\partial x_j=\delta_{ij}/r-x_ix_j/r^3$, so $\bar x_j=\sum_i\bar y_i\,\partial y_i/\partial x_j=\bar y_j/r-x_j(\boldsymbol x^\top\bar{\boldsymbol y})/r^3$, i.e. $\bar{\boldsymbol x}=\frac1r\big(\bar{\boldsymbol y}-\hat{\boldsymbol x}(\hat{\boldsymbol x}^\top\bar{\boldsymbol y})\big)$ with $\hat{\boldsymbol x}=\boldsymbol x/r$: the component of $\bar{\boldsymbol y}$ perpendicular to $\boldsymbol x$, scaled by $1/r$. Then $\boldsymbol x^\top\bar{\boldsymbol x}=0$: rescaling $\boldsymbol x$ does not change $y$.
:::
:::

:::exercise Exercise 7: Code: a custom Function with a surrogate gradient
Write a `torch.autograd.Function` for `clamp01(x)` (clamp to $[0,1]$) whose backward passes the gradient unchanged *everywhere* (straight-through), and show how its gradient differs from `torch.clamp`'s.
:::details Solution
```python
import torch

class ClampSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x): return x.clamp(0, 1)
    @staticmethod
    def backward(ctx, g): return g                      # ignore the clamp in backward

x = torch.tensor([-1.0, 0.5, 2.0], requires_grad=True)
ClampSTE.apply(x).sum().backward(); print(x.grad)       # tensor([1., 1., 1.])
x.grad = None
x.clamp(0, 1).sum().backward(); print(x.grad)           # tensor([0., 1., 0.]): zero outside [0, 1]
```
The built-in clamp gives zero gradient outside the interval, so saturated entries cannot recover; the straight-through version always pushes them. It is biased, but is the standard trick for quantization-aware training.
:::
:::

:::exercise Exercise 8: What would happen if (challenging)
You use sigmoid hidden units and initialize weights with standard deviation 1 in a network 20 layers deep and width 100. Estimate the magnitude of the gradient at layer 1 relative to layer 20, and explain how the picture changes with tanh and Glorot init.
:::details Solution
With standard deviation 1 and width 100, pre-activations have standard deviation around 10, so nearly every sigmoid is saturated and $\sigma'(z)$ is typically $10^{-3}$ or much smaller. Each backward step multiplies the gradient norm by about $\sigma'_{\rm typ}\cdot\sqrt n\,\sigma_w\approx10\,\sigma'_{\rm typ}$, a factor far below one, so after 19 layers the first layer's gradient is many orders of magnitude below the last layer's (often at the edge of float32 underflow); measure it by re-running the experiment behind the chart above with `std=1`. With Glorot init (variance $1/n$), $\sqrt n\sigma_w\approx1$ and $\sigma'\le 0.25$ gives a factor about 4 smaller per layer for sigmoid. For tanh ($\tanh'(0)=1$) the factor stays near 1 at small pre-activations, so gradients neither vanish nor explode, which is the point of Glorot/He initialization derived in [Week 3](week-03.html#initialization).
:::
:::

:::exercise Exercise 9: Memory of backprop (challenging)
A 50-layer MLP of width 4096 processes a batch of 256 in float32. Estimate the activation memory that backprop must hold, and explain how gradient checkpointing trades compute for memory.
:::details Solution
Each layer stores $\mathbf A$ (and $\mathbf Z$, or a ReLU mask): about $256\cdot4096\cdot4$ bytes $=4$ MiB per tensor, so roughly 8 MiB per layer for $\mathbf A,\mathbf Z$, times 50 layers, about 400 MiB. It grows linearly in depth and batch size. With **checkpointing** you store activations only at every $\sqrt{L}$-th layer and recompute the others during the backward pass: memory drops to $O(\sqrt L)$ layers at the cost of one extra forward pass (about 33 percent more compute). In PyTorch: `torch.utils.checkpoint.checkpoint`.
:::
:::

## Key takeaways

:::key Key takeaways
- A neuron is a weighted sum plus a nonlinearity; stacking layers without nonlinearity collapses to one linear map, and one hidden layer already suffices to approximate any continuous function on a compact set, but the theorem says nothing about learnability or width.
- Activations matter through their derivatives: saturation (sigmoid, tanh) shrinks gradients, ReLU is cheap but can die, GELU/SiLU are smooth modern defaults.
- Softmax cross-entropy has the clean gradient $\mathbf P-\mathbf Y$; always feed logits to the loss, never probabilities.
- Backprop is the chain rule plus dynamic programming on a computational graph: cache forward values, sweep backward, add where paths merge. A step costs about 3x a forward pass and memory proportional to depth.
- Reverse-mode AD returns a vector-Jacobian product per pass, which is ideal for scalar losses with huge parameter counts; PyTorch builds the graph dynamically and accumulates into `.grad`.
- Gradient-check every hand-written backward (float64, centered differences, relative error) and overfit a tiny batch before trusting a training run.
:::

## Further reading

- **Essential.** Rumelhart, Hinton & Williams (1986). *Learning representations by back-propagating errors.* Nature 323, 533-536.
- **Essential.** Baydin, Pearlmutter, Radul & Siskind (2018). *Automatic differentiation in machine learning: a survey.* JMLR 18(153). arXiv:1502.05767.
- **Essential.** Karpathy, A. *The spelled-out intro to neural networks and backpropagation: building micrograd* (video lecture, 2022) and the `micrograd` repository.
- Hornik, Stinchcombe & White (1989). *Multilayer feedforward networks are universal approximators.* Neural Networks 2(5).
- Cybenko (1989). *Approximation by superpositions of a sigmoidal function.* Mathematics of Control, Signals and Systems 2.
- Leshno, Lin, Pinkus & Schocken (1993). *Multilayer feedforward networks with a nonpolynomial activation function can approximate any function.* Neural Networks 6(6).
- Montufar, Pascanu, Cho & Bengio (2014). *On the number of linear regions of deep neural networks.* NeurIPS. arXiv:1402.1869.
- Glorot & Bengio (2010). *Understanding the difficulty of training deep feedforward neural networks.* AISTATS.
- Hendrycks & Gimpel (2016). *Gaussian Error Linear Units (GELUs).* arXiv:1606.08415.
- Ramachandran, Zoph & Le (2017). *Searching for activation functions.* arXiv:1710.05941.
- Lin, Goyal, Girshick, He & Dollar (2017). *Focal loss for dense object detection.* ICCV. arXiv:1708.02002.
- Paszke et al. (2019). *PyTorch: an imperative style, high-performance deep learning library.* NeurIPS. arXiv:1912.01703.
- Schmidhuber, J. (2015). *Deep learning in neural networks: an overview.* Neural Networks 61. arXiv:1404.7828.
- Chen, Xu, Zhang & Guestrin (2016). *Training deep nets with sublinear memory cost.* arXiv:1604.06174.
