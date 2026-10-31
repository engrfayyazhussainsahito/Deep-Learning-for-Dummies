---
title: Introduction, Mathematical Foundations, and Tooling
phase: Foundations
summary: What deep learning is and where it came from, a self-contained refresher of the linear algebra, calculus, probability and information theory you will use all semester, and a working PyTorch toolkit.
---

:::key Learning objectives
By the end of this week you should be able to:
- Explain what makes a model "deep", why depth helps, and how the field got from the perceptron to foundation models.
- Formalize learning as risk minimization: define loss, empirical risk, generalization gap, and the supervised, unsupervised and self-supervised settings.
- Use tensors fluently: shapes, matrix products, broadcasting, and the index notation that deep learning papers rely on.
- Compute gradients of scalar functions with respect to vectors and matrices, and apply the chain rule through compositions.
- Derive maximum likelihood estimators and show that squared error and cross-entropy are negative log-likelihoods.
- Define entropy, cross-entropy and KL divergence, and explain why minimizing cross-entropy is the same as fitting a model distribution to the data.
- Write reproducible PyTorch code: tensors, devices, autograd, and a training loop for linear regression.
:::

This course is about one idea applied with enormous persistence: build a flexible parametric function out of many simple differentiable pieces, define a number that says how wrong it is, and use calculus to nudge the parameters so that the number goes down. Everything else (convolutions, attention, diffusion, large language models) is a way of choosing the pieces, the number, or the nudging procedure.

That sounds simple, and in outline it is. The difficulty is that each of the three ingredients hides real mathematics, and the practical details (shapes, devices, random seeds, numerical stability) decide whether your experiment works at all. This first week therefore does two jobs. It gives you the big picture and the historical context, and it builds the mathematical and software foundation that every later week assumes. If you have a strong background you can skim the refreshers, but do not skip the exercises: the notation we fix here is the notation we use for the rest of the course.

## What is deep learning?

**Machine learning** is the study of algorithms that improve at a task through data instead of through hand-written rules. **Deep learning** is the subfield in which the learned function is a composition of many parametrized, differentiable layers, and in which the intermediate representations are themselves learned from data.

Consider recognizing a handwritten digit. A classical pipeline would have an engineer decide which features matter (stroke counts, edge histograms, loops), compute them, and train a simple classifier such as logistic regression or a support vector machine on top. The quality of the system was bounded by the quality of the hand-designed features. In a deep network, the input pixels go in at one end and class scores come out at the other; every stage in between is learned jointly, end to end, by gradient descent on a single objective.

{{figure: week01-paradigms | Classical machine learning separates feature engineering (hand-designed, fixed) from learning (a simple model on top). Deep learning learns a hierarchy of features and the classifier jointly, end to end.}}

A deep network is a function $f_{\boldsymbol\theta}:\mathcal{X}\to\mathcal{Y}$ written as a composition

$$
f_{\boldsymbol\theta}(\boldsymbol{x}) = f^{(L)}_{\boldsymbol\theta_L}\circ f^{(L-1)}_{\boldsymbol\theta_{L-1}}\circ\cdots\circ f^{(1)}_{\boldsymbol\theta_1}(\boldsymbol{x}),
$$

where each $f^{(\ell)}$ is a "layer" with its own parameters $\boldsymbol\theta_\ell$ (weights and biases). The number $L$ is the **depth**. Typical layers apply a linear map followed by a simple nonlinearity, but the layers can be convolutions, attention blocks, normalizations, or anything else that is differentiable almost everywhere. The only structural requirement is that we can compute the derivative of the final loss with respect to every parameter, which is what backpropagation (Week 2) provides.

### Why "representation learning" is the real story

The phrase to remember is **representation learning**. The first layers of a vision network respond to edges and color blobs; middle layers to textures and parts; late layers to object categories. Nobody wrote those features down. They emerge because they are useful for lowering the loss, and because composition lets later layers reuse earlier ones. The same pattern holds in language models, where early layers encode local syntax and later layers encode more abstract, task-relevant information. The practical consequence is that the same recipe (large model, large dataset, gradient descent) transfers across domains with remarkably little domain-specific engineering.

### Why depth matters

Why not simply use one very wide layer? In Week 2 we will see that a single hidden layer can approximate any continuous function on a compact set (universal approximation), so expressivity alone does not force depth. What depth buys is **efficiency** and **inductive bias**:

1. **Parameter efficiency.** There are functions that a deep network of modest size represents exactly but that a shallow network needs exponentially many units to match. A classical example is the parity of $n$ bits, and a more recent one is the sawtooth function obtained by composing a "tent" map $k$ times: a depth-$k$ ReLU network with a constant number of units per layer produces $2^k$ linear pieces, while a one-hidden-layer ReLU network needs on the order of $2^k$ units to produce the same number of pieces. Composition multiplies complexity; width merely adds it.
2. **Reuse and hierarchy.** Natural data are compositional: pixels form edges, edges form parts, parts form objects; characters form words, words form phrases. A layered model can share intermediate features across many higher-level concepts.
3. **Optimization behavior.** Deep, overparametrized networks are easier to optimize with simple gradient methods than their shallowness would suggest. This is empirical and only partly understood, and we return to it in [Week 3](week-03.html).

:::note A caution about "depth"
Depth is not free. Each extra layer is another factor in the chain rule, so gradients can vanish or explode, and the model can be harder to train. Much of the history of the field (careful initialization, normalization layers, residual connections) is the history of making depth trainable.
:::

### Scale and the modern picture

Since about 2012, three ingredients have grown together: **data** (ImageNet's 1.2 million labeled images, then web-scale text and image-text corpora), **compute** (GPUs, then TPUs and large clusters), and **model size**. The chart below shows published parameter counts for some landmark models. The values span almost six orders of magnitude, from tens of thousands of parameters in the 1998 digit recognizer to hundreds of billions in 2020.

```chart
{
  "type": "bar",
  "caption": "Approximate parameter counts of landmark models (published figures: LeNet-5 about 60 thousand; AlexNet about 60 million; VGG-16 about 138 million; ResNet-50 about 25.6 million; BERT-large 340 million; GPT-2 1.5 billion; GPT-3 175 billion). Logarithmic axis.",
  "height": 340,
  "data": {
    "labels": ["LeNet-5 (1998)", "AlexNet (2012)", "ResNet-50 (2015)", "VGG-16 (2014)", "BERT-large (2018)", "GPT-2 (2019)", "GPT-3 (2020)"],
    "datasets": [{ "label": "Parameters", "data": [60000, 60000000, 25600000, 138000000, 340000000, 1500000000, 175000000000], "backgroundColor": "var(--c-blue)" }]
  },
  "options": { "scales": { "y": { "type": "logarithmic", "title": { "display": true, "text": "parameters (log scale)" } }, "x": { "title": { "display": true, "text": "model" } } } }
}
```

## A short history, from the perceptron to foundation models

History matters here for a practical reason: many "new" ideas are old ideas that finally had enough data and compute, and knowing which is which helps you judge claims.

{{figure: week01-timeline | A condensed timeline of deep learning. Events are evenly spaced, not to scale in time.}}

:::history The long road
- **1943.** Warren McCulloch and Walter Pitts propose a logical model of a neuron as a threshold unit and show that networks of such units can compute logical functions.
- **1958.** Frank Rosenblatt introduces the **perceptron** and a learning rule for it, and builds the Mark I Perceptron hardware. Enthusiastic press coverage sets expectations the technology cannot meet.
- **1969.** Marvin Minsky and Seymour Papert's book *Perceptrons* proves that a single-layer perceptron cannot represent functions like XOR. Funding for neural networks shrinks; this is often counted as the start of the first "AI winter", although the book's effect on funding is debated by historians.
- **1970s to 1980s.** Backpropagation, the efficient application of the chain rule to layered networks, is discovered several times (Linnainmaa 1970 for reverse-mode automatic differentiation; Werbos 1974 for neural networks). Kunihiko Fukushima's Neocognitron (1980) anticipates convolutional networks.
- **1986.** Rumelhart, Hinton and Williams publish a clear and influential account of backpropagation and show that hidden layers learn useful internal representations. Multilayer networks become a serious research program again.
- **1989 to 1998.** Hornik, Stinchcombe and White (1989) and Cybenko (1989) prove universal approximation theorems. Yann LeCun and colleagues train convolutional networks with backpropagation to read zip codes, culminating in LeNet-5 (1998).
- **1997.** Hochreiter and Schmidhuber publish the LSTM, a recurrent architecture designed to fight vanishing gradients.
- **2000s.** Support vector machines, kernel methods and boosted trees dominate practical machine learning, as they are easier to train and tune and have good theory. A small community (Hinton, Bengio, LeCun and others) keeps working on deep models; layerwise pretraining (2006) shows that deep networks can be trained.
- **2012.** Krizhevsky, Sutskever and Hinton's AlexNet wins the ImageNet challenge with a top-5 error of 15.3%, against 26.2% for the runner-up, using GPUs, ReLU activations and dropout. This is the moment the field reorients.
- **2014 to 2016.** Generative adversarial networks (Goodfellow et al., 2014), Adam (Kingma and Ba, 2014), batch normalization (Ioffe and Szegedy, 2015), attention for translation (Bahdanau et al., 2014), and residual networks (He et al., 2015), which allow hundreds of layers.
- **2017.** Vaswani et al. introduce the Transformer, which replaces recurrence with attention and scales far better on parallel hardware.
- **2018 to now.** Pretrained Transformers (BERT, GPT series), diffusion models for images, and large multimodal models. Scaling laws (Kaplan et al., 2020) describe how loss falls predictably with compute, data and parameters. A single pretrained "foundation model" is adapted to many tasks.
:::

The ImageNet Large Scale Visual Recognition Challenge gives one of the cleanest records of progress. Winning top-5 error went from 28.2% in 2010 (before deep learning) to 3.6% in 2015 (ResNet), below the roughly 5% a trained human annotator achieved in a study by Russakovsky et al.

```chart
{
  "type": "line",
  "caption": "Winning top-5 error of the ImageNet ILSVRC classification challenge by year (published challenge results: 2010 NEC-UIUC 28.2%, 2011 XRCE 25.8%, 2012 AlexNet/SuperVision 15.3%, 2013 ZFNet/Clarifai 11.7%, 2014 GoogLeNet 6.7%, 2015 ResNet 3.6%).",
  "height": 320,
  "data": {
    "labels": ["2010", "2011", "2012", "2013", "2014", "2015"],
    "datasets": [{ "label": "winning top-5 error (%)", "data": [28.2, 25.8, 15.3, 11.7, 6.7, 3.6], "borderColor": "var(--c-accent)", "backgroundColor": "var(--c-accent)", "tension": 0.15 }]
  },
  "options": { "scales": { "x": { "title": { "display": true, "text": "challenge year" } }, "y": { "title": { "display": true, "text": "top-5 error (%)" }, "min": 0 } } }
}
```

:::intuition Why now?
Neural networks were not "wrong" in 1990; they were starved. A network with millions of parameters needs millions of examples to avoid memorizing, and needs trillions of floating-point operations to train. Large labeled datasets and GPU matrix multiplication supplied both, and a handful of algorithmic fixes (ReLU, better initialization, dropout, normalization) made optimization reliable. When someone says "it is just scaling", the accurate version is "scaling plus the fixes that make scaling work".
:::

## Machine learning recap

### The learning problem

We observe data and want a function that works well on data we have not seen. Formally, assume pairs $(\boldsymbol{x},y)$ are drawn independently from an unknown joint distribution $p_{\text{data}}$. We choose a **hypothesis class** $\mathcal{F}=\{f_{\boldsymbol\theta}\}$ (here, a neural architecture) and a **loss function** $\ell(f_{\boldsymbol\theta}(\boldsymbol{x}),y)\ge 0$ that measures the cost of predicting $f_{\boldsymbol\theta}(\boldsymbol{x})$ when the truth is $y$. The quantity we really care about is the **risk** (also called the population risk or generalization error):

$$
R(\boldsymbol\theta)=\E_{(\boldsymbol{x},y)\sim p_{\text{data}}}\big[\ell(f_{\boldsymbol\theta}(\boldsymbol{x}),y)\big].
$$

We cannot compute $R$, because we do not know $p_{\text{data}}$. We have only a finite training set $\{(\boldsymbol{x}_i,y_i)\}_{i=1}^N$, so we minimize the **empirical risk**

$$
\hat R(\boldsymbol\theta)=\frac1N\sum_{i=1}^N \ell(f_{\boldsymbol\theta}(\boldsymbol{x}_i),y_i).
$$

This is **empirical risk minimization (ERM)**. The difference $R(\hat{\boldsymbol\theta})-\hat R(\hat{\boldsymbol\theta})$ at the minimizer is the **generalization gap**. It is almost always positive, because the parameters were chosen to fit exactly these samples: the training loss is an optimistically biased estimate of the risk. That is why we measure performance on a held-out **validation** set (used for model selection) and a final **test** set (used once).

:::definition Training error vs. generalization
- **Underfitting:** the model class or the optimization is too weak; both training and validation loss are high.
- **Overfitting:** training loss is low but validation loss is much higher: the model has fit noise or idiosyncrasies of the training set.
- **Generalization gap:** validation (or test) loss minus training loss.
:::

Classical statistical learning theory said that overfitting is controlled by model capacity: too many parameters relative to $N$ means a large gap. Modern deep networks routinely have more parameters than training examples, fit the training set perfectly, and yet generalize. Understanding why (implicit regularization of gradient descent, the geometry of the loss landscape, "double descent") is an active research topic and we return to it in [Week 3](week-03.html) and [Week 4](week-04.html). For now, remember the protocol: **never select hyperparameters on the test set**.

### Supervised, unsupervised and self-supervised learning

{{figure: week01-learning-setups | The three learning settings differ in where the training signal comes from. Self-supervised learning manufactures labels from the data itself.}}

**Supervised learning** has labeled pairs $(\boldsymbol{x},y)$. When $y$ is a category it is *classification* (output a distribution over classes); when $y$ is a real number or vector it is *regression*. Most of the early successes of deep learning (ImageNet, speech recognition, translation with parallel text) were supervised. The cost is labels: humans must annotate, which does not scale to billions of examples.

**Unsupervised learning** has only inputs $\boldsymbol{x}$. The goal is to discover structure: clusters, low-dimensional manifolds, or the density $p(\boldsymbol{x})$ itself. Classical examples are $k$-means and PCA; deep examples are autoencoders, variational autoencoders and generative adversarial networks ([Week 13](week-13.html)).

**Self-supervised learning** is unsupervised data with supervised-style training: we create a prediction task whose labels come for free from the data. Predict the next word given the previous ones (GPT-style language modeling), predict masked words (BERT), predict one view of an image from another, or predict the missing part of an input. The model is trained with ordinary supervised losses, but on essentially unlimited unlabeled data. The resulting representations are then reused. This idea, more than any architecture, is why large language models exist.

Two more settings appear later: **reinforcement learning** (learning from reward signals by acting in an environment) and **semi-supervised learning** (few labels, many unlabeled examples). They are outside our weekly scope but use the same machinery.

### The losses you will meet

The loss encodes what "wrong" means. For regression with target $y\in\R$ and prediction $\hat y$, **squared error** $\ell=\tfrac12(\hat y-y)^2$ is standard. For classification with $K$ classes, the model outputs a probability vector $\hat{\boldsymbol p}$ and we use **cross-entropy** $\ell=-\log \hat p_y$, the negative log-probability of the true class. Both are negative log-likelihoods under simple noise models, as we derive in the maximum likelihood section below. The 0-1 loss (count of mistakes) is what we usually report but cannot optimize directly, since it is piecewise constant and has zero gradient almost everywhere. A good **surrogate loss** is smooth, upper-bounds the 0-1 loss or is consistent with it, and gives informative gradients. [Week 2](week-02.html) examines the options in detail.

### A bias-variance picture

For squared error and a fixed input, the expected prediction error splits into three terms:

$$
\E\big[(\hat f(\boldsymbol{x})-y)^2\big]=\underbrace{\big(\E\hat f(\boldsymbol{x})-f^\star(\boldsymbol{x})\big)^2}_{\text{bias}^2}+\underbrace{\Var\big(\hat f(\boldsymbol{x})\big)}_{\text{variance}}+\underbrace{\sigma^2}_{\text{noise}},
$$

where $f^\star(\boldsymbol{x})=\E[y\mid\boldsymbol{x}]$, the expectation is over training sets and label noise, and $\sigma^2$ is the irreducible noise. Simple models have high bias and low variance, flexible models the opposite. The decomposition is exact for squared error, but the classical "U-shaped" tradeoff curve it suggests is not the whole story for heavily overparametrized networks, so treat it as a useful way of thinking, not a law.

## Math refresher I: linear algebra for tensors

Deep learning is, computationally, a long sequence of tensor operations. You do not need abstract vector-space theory; you need to be fluent in shapes and in what a matrix product *does*.

### Scalars, vectors, matrices, tensors

A **tensor** is a multi-dimensional array of numbers. Its **rank** (or number of dimensions, "ndim") is the number of indices needed to address an element; its **shape** lists the size along each axis.

{{figure: week01-tensors | Tensors of increasing rank, with the conventional meaning of each axis in deep learning: a batch of RGB images is a rank-4 tensor of shape (B, C, H, W) in PyTorch.}}

Conventions used throughout the course: scalars are $a$, vectors are bold lowercase $\boldsymbol{x}\in\R^d$ and are **column vectors** in math, matrices are bold uppercase $\mathbf{W}\in\R^{m\times n}$, and $\mathbf{X}\in\R^{N\times d}$ stacks $N$ examples as **rows**. This row-per-example convention matches how PyTorch stores a batch, so a linear layer reads $\mathbf{Y}=\mathbf{X}\mathbf{W}^\top+\boldsymbol{b}$ in code with $\mathbf{W}\in\R^{d_{\text{out}}\times d_{\text{in}}}$. (Math texts that treat $\boldsymbol{x}$ as a column write $\boldsymbol{y}=\mathbf{W}\boldsymbol{x}+\boldsymbol{b}$; transposing both sides gives the batch form. Keep track of which one a paper uses.)

### Matrix multiplication, three ways

For $\mathbf{A}\in\R^{m\times k}$ and $\mathbf{B}\in\R^{k\times n}$ the product $\mathbf{C}=\mathbf{A}\mathbf{B}\in\R^{m\times n}$ has entries

$$
C_{ij}=\sum_{l=1}^{k}A_{il}B_{lj}.
$$

Hold three pictures in your head, because different problems call for different ones.

1. **Dot products.** $C_{ij}$ is the dot product of row $i$ of $\mathbf{A}$ with column $j$ of $\mathbf{B}$. This is the view for "similarity of every query with every key", as in attention.
2. **Linear combination of columns.** Column $j$ of $\mathbf{C}$ is $\mathbf{A}$ applied to column $j$ of $\mathbf{B}$, i.e. a combination of the columns of $\mathbf{A}$ weighted by column $j$ of $\mathbf{B}$. The map $\boldsymbol{x}\mapsto\mathbf{A}\boldsymbol{x}$ sends the $j$-th basis vector to the $j$-th column of $\mathbf{A}$.
3. **Sum of outer products.** $\mathbf{A}\mathbf{B}=\sum_{l}\mathbf{a}_{:,l}\,\mathbf{b}_{l,:}$, a sum of $k$ rank-one matrices. This is exactly the form that appears in weight gradients: $\partial\mathcal L/\partial\mathbf{W}$ is a sum over the batch of outer products of upstream gradient and input.

The cost of the product is $mkn$ multiply-adds, which is why modern hardware is judged in FLOP/s: a transformer's cost is dominated by matrix multiplications, and GPUs have dedicated units for them.

A matrix is a **linear map**: $f(a\boldsymbol{x}+b\boldsymbol{y})=af(\boldsymbol{x})+bf(\boldsymbol{y})$. Try the demo below to see how a $2\times2$ matrix deforms the plane, and what its singular values and eigenvectors mean geometrically. This geometric picture is the right way to think about a layer $\boldsymbol{x}\mapsto \mathbf{W}\boldsymbol{x}$ before the nonlinearity is applied.

{{demo: week01-linear-map | A 2x2 matrix as a linear map of the plane | 520}}

### Norms, inner products, and the geometry of similarity

The dot product $\boldsymbol{a}^\top\boldsymbol{b}=\sum_i a_ib_i=\norm{\boldsymbol a}\norm{\boldsymbol b}\cos\phi$ measures alignment. The **$\ell_2$ norm** $\norm{\boldsymbol x}_2=\sqrt{\boldsymbol x^\top\boldsymbol x}$ is Euclidean length; the $\ell_1$ norm $\sum_i|x_i|$ appears in sparsity-inducing penalties; the **Frobenius norm** $\norm{\mathbf{A}}_F=\sqrt{\sum_{ij}A_{ij}^2}$ is the $\ell_2$ norm of the flattened matrix. Cosine similarity $\cos\phi$ is the dot product of unit-normalized vectors and is the standard similarity measure for embeddings.

### Eigenvalues, singular values and why they matter for networks

A square matrix $\mathbf{A}$ has an **eigenvector** $\boldsymbol v\neq 0$ with eigenvalue $\lambda$ if $\mathbf{A}\boldsymbol v=\lambda\boldsymbol v$. For a **symmetric** matrix, eigenvalues are real and the eigenvectors can be chosen orthonormal, so $\mathbf{A}=\mathbf{Q}\boldsymbol\Lambda\mathbf{Q}^\top$. Every real matrix (square or not) has a **singular value decomposition** $\mathbf{A}=\mathbf{U}\boldsymbol\Sigma\mathbf{V}^\top$ with orthogonal $\mathbf U,\mathbf V$ and non-negative diagonal $\boldsymbol\Sigma$: the map rotates, stretches along orthogonal axes by the singular values $\sigma_i$, and rotates again. The largest singular value $\sigma_{\max}=\norm{\mathbf A}_2$ is the maximum factor by which $\mathbf{A}$ can stretch a vector.

Why this matters: in a deep linear chain $\mathbf{W}_L\cdots\mathbf{W}_1\boldsymbol{x}$, the signal's norm can grow or shrink by up to $\prod_\ell\sigma_{\max}(\mathbf{W}_\ell)$. If typical singular values are slightly above 1, activations and gradients explode exponentially with depth; slightly below 1, they vanish. The condition number $\kappa=\sigma_{\max}/\sigma_{\min}$ of the Hessian also governs how fast gradient descent converges, which we see in the optimization demo below and in [Week 3](week-03.html). For symmetric positive semidefinite $\mathbf{A}$, a positive-definite quadratic $\tfrac12\boldsymbol w^\top\mathbf A\boldsymbol w$ has level sets that are ellipses with axes along the eigenvectors and lengths inversely proportional to $\sqrt{\lambda_i}$.

### Broadcasting and shapes

Real code rarely matches shapes exactly, so libraries use **broadcasting**: when combining two tensors elementwise, shapes are aligned from the *trailing* axis; two axes are compatible if they are equal or one of them is 1 (or missing), and a size-1 axis is conceptually repeated to match. Broadcasting is how a bias vector of shape `(d,)` is added to a batch of shape `(N, d)` without a loop.

{{figure: week01-broadcast | Broadcasting a column (3, 1) against a row (1, 4) produces a (3, 4) result without materializing the repeated copies.}}

Broadcasting is also the largest single source of silent bugs. Subtracting a `(N,)` vector from an `(N, 1)` vector yields an `(N, N)` matrix, not an error. We will return to this under common mistakes.

```python:tensors_and_shapes.py
import torch

# Creation: from data, from factories, from random numbers
a = torch.tensor([[1., 2., 3.], [4., 5., 6.]])   # shape (2, 3), dtype float32
z = torch.zeros(2, 3)
r = torch.randn(4, 3)                            # standard normal samples
print(a.shape, a.dtype, a.ndim, a.numel())       # torch.Size([2, 3]) torch.float32 2 6

# Matrix product: (2, 3) @ (3, 4) -> (2, 4)
W = torch.randn(3, 4)
print((a @ W).shape)

# Broadcasting: (4, 3) + (3,) -> (4, 3);  (3, 1) * (1, 4) -> (3, 4)
bias = torch.tensor([10., 20., 30.])
print((r + bias).shape)
col = torch.arange(3.).reshape(3, 1)
row = torch.arange(4.).reshape(1, 4)
print((col * row).shape)                         # torch.Size([3, 4])

# The classic broadcasting trap: (N,) - (N, 1) is (N, N), not (N,)
y_hat = torch.randn(5)
y     = torch.randn(5, 1)
print((y_hat - y).shape)                         # torch.Size([5, 5])  <- usually a bug

# Reshaping and axis permutation: a batch of images (B, C, H, W) -> (B, H, W, C)
imgs = torch.randn(8, 3, 32, 32)
print(imgs.permute(0, 2, 3, 1).shape)            # torch.Size([8, 32, 32, 3])
flat = imgs.reshape(8, -1)                       # -1 infers 3*32*32 = 3072
print(flat.shape)                                # torch.Size([8, 3072])
```

### Einsum: index notation as code

Deep learning papers write operations with indices; `einsum` lets you write the same thing in code. Repeated indices on the input side are summed over, and the output indices are listed after `->`. A batched matrix multiply is `torch.einsum('bij,bjk->bik', A, B)`; the attention score computation in [Week 9](week-09.html) will be `'bqd,bkd->bqk'`. Once you can read einsum, you can read most tensor code unambiguously.

```python:einsum_examples.py
import torch

A = torch.randn(4, 2, 3)      # batch of 4 matrices (2 x 3)
B = torch.randn(4, 3, 5)      # batch of 4 matrices (3 x 5)

C1 = torch.einsum('bij,bjk->bik', A, B)   # batched matmul
C2 = A @ B                                # same thing
print(torch.allclose(C1, C2, atol=1e-6))  # True

x = torch.randn(7, 3)
W = torch.randn(5, 3)
# linear layer without bias: y[n, o] = sum_i x[n, i] * W[o, i]
y = torch.einsum('ni,oi->no', x, W)
print(torch.allclose(y, x @ W.T, atol=1e-6))

# outer product, trace, and sum over everything
u, v = torch.randn(3), torch.randn(4)
print(torch.einsum('i,j->ij', u, v).shape)     # (3, 4)
M = torch.randn(3, 3)
print(torch.einsum('ii->', M), M.trace())      # same number
```

## Math refresher II: matrix calculus and the chain rule

Training means computing derivatives of a scalar loss with respect to millions of parameters. This section fixes the conventions and derives the handful of gradient formulas that occur everywhere.

### Gradients, Jacobians and conventions

For a scalar function $f:\R^n\to\R$, the **gradient** $\grad_{\boldsymbol x}f\in\R^n$ collects the partial derivatives, $(\grad_{\boldsymbol x}f)_i=\partial f/\partial x_i$. It points in the direction of steepest ascent, and the first-order Taylor expansion is $f(\boldsymbol x+\boldsymbol\delta)\approx f(\boldsymbol x)+\grad f^\top\boldsymbol\delta$. Gradient descent moves along $-\grad f$ because that direction decreases $f$ fastest for a small step of fixed length.

For a vector-valued $\boldsymbol f:\R^n\to\R^m$ the derivative is the **Jacobian** $\mathbf J\in\R^{m\times n}$ with $J_{ij}=\partial f_i/\partial x_j$. It is the best linear approximation: $\boldsymbol f(\boldsymbol x+\boldsymbol\delta)\approx\boldsymbol f(\boldsymbol x)+\mathbf J\boldsymbol\delta$. For a scalar function, the Jacobian is the gradient written as a row vector. For a scalar loss and a matrix parameter, $\partial\mathcal L/\partial\mathbf W$ is the matrix of the same shape as $\mathbf W$ with entries $\partial\mathcal L/\partial W_{ij}$. That last convention (**gradient has the shape of the parameter**) is the one PyTorch uses: `W.grad.shape == W.shape` always.

The **Hessian** $\mathbf H\in\R^{n\times n}$, $H_{ij}=\partial^2f/\partial x_i\partial x_j$, collects second derivatives; it describes curvature and the second-order Taylor expansion $f(\boldsymbol x+\boldsymbol\delta)\approx f+\grad f^\top\boldsymbol\delta+\tfrac12\boldsymbol\delta^\top\mathbf H\boldsymbol\delta$.

### The chain rule

If $\boldsymbol y=\boldsymbol g(\boldsymbol x)$ and $\boldsymbol z=\boldsymbol f(\boldsymbol y)$, then the Jacobian of the composition is the product of Jacobians:

$$
\frac{\partial\boldsymbol z}{\partial\boldsymbol x}=\frac{\partial\boldsymbol z}{\partial\boldsymbol y}\,\frac{\partial\boldsymbol y}{\partial\boldsymbol x},\qquad
\mathbf J_{f\circ g}(\boldsymbol x)=\mathbf J_f(\boldsymbol g(\boldsymbol x))\;\mathbf J_g(\boldsymbol x).
$$

For a scalar loss $\mathcal L(\boldsymbol z)$ at the end of a chain, the row vector $\partial\mathcal L/\partial\boldsymbol z$ is multiplied from the left by successive Jacobians. Equivalently, in terms of gradient *column* vectors, the gradient is pulled back through the transpose:

$$
\grad_{\boldsymbol x}\mathcal L=\mathbf J_g^\top\,\grad_{\boldsymbol y}\mathcal L.
$$

This product of a transposed Jacobian with a vector is called a **vector-Jacobian product (VJP)**. It is the one primitive that backpropagation needs, and it is cheap: we never form the full Jacobian (which could be $10^6\times10^6$), only its action on a vector. We develop this in [Week 2](week-02.html).

### Four gradients worth memorizing

Here is a small toolkit. We derive each with index notation, which is the most reliable method when you are unsure: write the scalar as a sum, differentiate one component, and recognize the pattern.

:::math Gradient of a linear form and a quadratic form
**Linear:** $f(\boldsymbol x)=\boldsymbol a^\top\boldsymbol x=\sum_i a_ix_i$. Then $\partial f/\partial x_k=a_k$, so $\grad f=\boldsymbol a$.

**Quadratic:** $f(\boldsymbol x)=\boldsymbol x^\top\mathbf A\boldsymbol x=\sum_{i,j}x_iA_{ij}x_j$. Differentiating with respect to $x_k$, the variable appears once as $x_i$ (when $i=k$) and once as $x_j$ (when $j=k$):

$$
\frac{\partial f}{\partial x_k}=\sum_jA_{kj}x_j+\sum_ix_iA_{ik}=(\mathbf A\boldsymbol x)_k+(\mathbf A^\top\boldsymbol x)_k.
$$

Therefore $\grad f=(\mathbf A+\mathbf A^\top)\boldsymbol x$, which is $2\mathbf A\boldsymbol x$ when $\mathbf A$ is symmetric.
:::

:::math Gradient of the least-squares loss
Let $\mathbf X\in\R^{N\times d}$, $\boldsymbol y\in\R^N$, $\boldsymbol w\in\R^d$ and $\mathcal L(\boldsymbol w)=\frac1N\norm{\mathbf X\boldsymbol w-\boldsymbol y}^2$. Put $\boldsymbol r=\mathbf X\boldsymbol w-\boldsymbol y$ (the residual). Expand:

$$
\mathcal L=\tfrac1N\big(\boldsymbol w^\top\mathbf X^\top\mathbf X\boldsymbol w-2\boldsymbol y^\top\mathbf X\boldsymbol w+\boldsymbol y^\top\boldsymbol y\big).
$$

Using the two rules above with the symmetric matrix $\mathbf X^\top\mathbf X$,

$$
\grad_{\boldsymbol w}\mathcal L=\tfrac1N\big(2\mathbf X^\top\mathbf X\boldsymbol w-2\mathbf X^\top\boldsymbol y\big)=\tfrac2N\mathbf X^\top(\mathbf X\boldsymbol w-\boldsymbol y)=\tfrac2N\mathbf X^\top\boldsymbol r.
$$

Setting the gradient to zero gives the **normal equations** $\mathbf X^\top\mathbf X\boldsymbol w=\mathbf X^\top\boldsymbol y$, whose solution (when $\mathbf X^\top\mathbf X$ is invertible) is $\boldsymbol w^\star=(\mathbf X^\top\mathbf X)^{-1}\mathbf X^\top\boldsymbol y$. The Hessian is $\tfrac2N\mathbf X^\top\mathbf X\succeq0$, so the loss is convex and this stationary point is a global minimum.
:::

Notice the structure $\mathbf X^\top\boldsymbol r$: the gradient with respect to the weights is the transposed input matrix times the error. Every layer's weight gradient in a deep network has the same form (input activations transposed times upstream error), which is the main result of Week 2.

:::math Gradient of a matrix-vector product, and of the bilinear form
For $\boldsymbol y=\mathbf W\boldsymbol x$ and a downstream scalar $\mathcal L$ with $\boldsymbol g=\grad_{\boldsymbol y}\mathcal L$: since $y_i=\sum_jW_{ij}x_j$,

$$
\frac{\partial\mathcal L}{\partial W_{ij}}=\frac{\partial\mathcal L}{\partial y_i}\frac{\partial y_i}{\partial W_{ij}}=g_ix_j\quad\Longrightarrow\quad\grad_{\mathbf W}\mathcal L=\boldsymbol g\,\boldsymbol x^\top\ (\text{an outer product}),
$$

$$
\frac{\partial\mathcal L}{\partial x_j}=\sum_i\frac{\partial\mathcal L}{\partial y_i}W_{ij}\quad\Longrightarrow\quad\grad_{\boldsymbol x}\mathcal L=\mathbf W^\top\boldsymbol g.
$$

Shape check: $\boldsymbol g\in\R^m$, $\boldsymbol x\in\R^n$, so $\boldsymbol g\boldsymbol x^\top\in\R^{m\times n}=\text{shape}(\mathbf W)$, and $\mathbf W^\top\boldsymbol g\in\R^n$. Always check shapes: they catch most transposition errors.
:::

### Verifying derivatives numerically

Whenever you derive a gradient, test it. A central difference $\big(f(\boldsymbol x+\epsilon\boldsymbol e_i)-f(\boldsymbol x-\epsilon\boldsymbol e_i)\big)/2\epsilon$ approximates $\partial f/\partial x_i$ with error $O(\epsilon^2)$. In `float64`, $\epsilon\approx10^{-6}$ balances truncation and rounding error. We use the same idea for gradient checking in the Week 2 lab; here is a first look, comparing the formula above with PyTorch autograd.

```python:check_gradients.py
import torch

torch.manual_seed(0)
N, d = 50, 4
X = torch.randn(N, d, dtype=torch.float64)
y = torch.randn(N, dtype=torch.float64)
w = torch.randn(d, dtype=torch.float64, requires_grad=True)

loss = ((X @ w - y) ** 2).mean()
loss.backward()                                   # autograd gradient

analytic = 2 / N * X.T @ (X @ w.detach() - y)     # our derived formula

# central finite differences
eps = 1e-6
numeric = torch.zeros(d, dtype=torch.float64)
for i in range(d):
    e = torch.zeros(d, dtype=torch.float64); e[i] = eps
    f_plus  = (((X @ (w.detach() + e)) - y) ** 2).mean()
    f_minus = (((X @ (w.detach() - e)) - y) ** 2).mean()
    numeric[i] = (f_plus - f_minus) / (2 * eps)

print("autograd :", w.grad)
print("analytic :", analytic)
print("numeric  :", numeric)
print("max abs diff (autograd vs analytic):", (w.grad - analytic).abs().max().item())
print("max abs diff (numeric  vs analytic):", (numeric - analytic).abs().max().item())
```

## Math refresher III: probability and maximum likelihood

Why does everyone minimize squared error for regression and cross-entropy for classification? Because these are what maximum likelihood estimation gives under simple probabilistic assumptions. Probability is the language that explains loss functions.

### The essentials

A **random variable** $X$ takes values according to a distribution: a probability mass function $p(x)$ for discrete variables, a density for continuous ones. The **expectation** is $\E[g(X)]=\sum_xp(x)g(x)$ (or an integral), the **variance** is $\Var(X)=\E[(X-\E X)^2]=\E[X^2]-(\E X)^2$, and the **covariance** of two variables is $\Cov(X,Y)=\E[(X-\E X)(Y-\E Y)]$. Expectation is linear regardless of dependence; variance of a sum adds only for uncorrelated variables. **Conditional probability** is $p(y\mid x)=p(x,y)/p(x)$, and **Bayes' rule** is

$$
p(\boldsymbol\theta\mid\mathcal D)=\frac{p(\mathcal D\mid\boldsymbol\theta)\,p(\boldsymbol\theta)}{p(\mathcal D)}.
$$

Two distributions dominate deep learning:

- The **Bernoulli/categorical** distribution over $K$ classes with probabilities $\boldsymbol p$, $p(y=k)=p_k$, $\sum_kp_k=1$. A classifier's softmax output is a categorical distribution over labels.
- The **Gaussian** $\mathcal N(\mu,\sigma^2)$ with density $\frac{1}{\sqrt{2\pi\sigma^2}}\exp\!\big(-\frac{(x-\mu)^2}{2\sigma^2}\big)$, the default noise model for real-valued targets, and (by the central limit theorem) a good description of sums of many small independent effects. Multivariate version: $\mathcal N(\boldsymbol\mu,\boldsymbol\Sigma)$.

Two facts underpin stochastic training. The **law of large numbers** says sample averages converge to expectations, which is why the empirical risk approximates the true risk. And a minibatch average $\frac1B\sum_{i\in\text{batch}}g_i$ of independent samples is an **unbiased estimator** of the full average with variance $\sigma^2/B$ (where $\sigma^2$ is the per-sample variance), so doubling the batch size cuts the gradient noise standard deviation only by $\sqrt2$. This one line of statistics explains many facts about batch size in [Week 3](week-03.html).

### Maximum likelihood estimation

Given i.i.d. data $\mathcal D=\{z_1,\dots,z_N\}$ and a model $p(z;\boldsymbol\theta)$, the **likelihood** is $\prod_ip(z_i;\boldsymbol\theta)$ as a function of $\boldsymbol\theta$. The **maximum likelihood estimator (MLE)** picks the parameters under which the observed data are most probable:

$$
\hat{\boldsymbol\theta}_{\text{MLE}}=\argmax_{\boldsymbol\theta}\prod_{i=1}^Np(z_i;\boldsymbol\theta)=\argmin_{\boldsymbol\theta}\;-\frac1N\sum_{i=1}^N\log p(z_i;\boldsymbol\theta).
$$

We take logs because products of many small numbers underflow and sums are easier to differentiate; the log is monotonic so the maximizer is unchanged. The last expression, the **average negative log-likelihood (NLL)**, is the form that becomes a training loss.

:::example Coin flips by hand
You flip a coin 10 times and see 7 heads. With $p=\Pr(\text{heads})$ the log-likelihood is $\ell(p)=7\log p+3\log(1-p)$. Setting $\ell'(p)=7/p-3/(1-p)=0$ gives $7(1-p)=3p$, so $\hat p=7/10$. The curve below is $\ell(p)$ computed at 25 values of $p$; its peak is at 0.7 with value $\approx-6.11$. Notice how steeply it falls near 0 and 1: a model that assigns tiny probability to something that happened is punished heavily.
:::

```chart
{
  "type": "line",
  "caption": "Log-likelihood of 7 heads in 10 flips as a function of the head probability p, computed from l(p) = 7 log p + 3 log(1-p). The maximum is at p = 0.7.",
  "height": 300,
  "data": {
    "labels": [0.02, 0.06, 0.1, 0.14, 0.18, 0.22, 0.26, 0.3, 0.34, 0.38, 0.42, 0.46, 0.5, 0.54, 0.58, 0.62, 0.66, 0.7, 0.74, 0.78, 0.82, 0.86, 0.9, 0.94, 0.98],
    "datasets": [{ "label": "log-likelihood", "data": [-27.44, -19.88, -16.43, -14.22, -12.6, -11.34, -10.33, -9.5, -8.8, -8.21, -7.71, -7.28, -6.93, -6.64, -6.42, -6.25, -6.15, -6.11, -6.15, -6.28, -6.53, -6.95, -7.65, -8.87, -11.88], "borderColor": "var(--c-blue)", "pointRadius": 2, "tension": 0.25 }]
  },
  "options": { "scales": { "x": { "title": { "display": true, "text": "p (probability of heads)" } }, "y": { "title": { "display": true, "text": "log-likelihood" } } } }
}
```

:::math Squared error is Gaussian maximum likelihood
Assume the regression model $y=f_{\boldsymbol\theta}(\boldsymbol x)+\varepsilon$ with noise $\varepsilon\sim\mathcal N(0,\sigma^2)$, i.e. $p(y\mid\boldsymbol x;\boldsymbol\theta)=\mathcal N\big(y;f_{\boldsymbol\theta}(\boldsymbol x),\sigma^2\big)$. The negative log-likelihood of one example is

$$
-\log p(y\mid\boldsymbol x;\boldsymbol\theta)=\frac{(y-f_{\boldsymbol\theta}(\boldsymbol x))^2}{2\sigma^2}+\tfrac12\log(2\pi\sigma^2).
$$

For fixed $\sigma$ the second term is constant in $\boldsymbol\theta$, so minimizing the NLL over $\boldsymbol\theta$ is exactly minimizing the sum of squared errors. The squared error is therefore not an arbitrary choice: it encodes the assumption of constant-variance Gaussian noise. If you instead assume Laplace noise you obtain the absolute error (L1 loss), which is more robust to outliers. If the noise variance depends on $\boldsymbol x$ you obtain a heteroscedastic loss.
:::

:::math Cross-entropy is categorical maximum likelihood
For classification, let the model output a categorical distribution $\hat{\boldsymbol p}_{\boldsymbol\theta}(\boldsymbol x)\in\Delta^{K-1}$ and let the label be $y\in\{1,\dots,K\}$. Then $p(y\mid\boldsymbol x;\boldsymbol\theta)=\hat p_{y}$, and the NLL is $-\log\hat p_y$. Writing the label as a one-hot vector $\boldsymbol e_y$, this equals $-\sum_{k}e_{y,k}\log\hat p_k$, which is the **cross-entropy** between the one-hot "true" distribution and the model. Training a softmax classifier by minimizing cross-entropy is therefore maximum likelihood for a categorical model.
:::

**Regularization as a prior.** If we add a prior $p(\boldsymbol\theta)$ and maximize the posterior (MAP), we minimize $-\sum_i\log p(z_i;\boldsymbol\theta)-\log p(\boldsymbol\theta)$. A zero-mean Gaussian prior $\mathcal N(0,\tau^2\mathbf I)$ contributes $\norm{\boldsymbol\theta}^2/(2\tau^2)$, which is **L2 regularization** (weight decay, up to the distinction we make in Week 3 for adaptive optimizers), and a Laplace prior gives the L1 penalty. This is a useful *interpretation*; it does not mean that every regularizer in deep learning has a clean Bayesian reading.

## Math refresher IV: information theory

Information theory supplies the other half of the story: not just "which loss", but a quantitative way to compare distributions.

### Entropy, cross-entropy and KL divergence

The **information content** (surprisal) of an outcome with probability $p$ is $-\log p$: rare events are more surprising. The **entropy** of a distribution $p$ is its average surprisal,

$$
H(p)=-\sum_xp(x)\log p(x)=\E_{x\sim p}[-\log p(x)].
$$

With $\log_2$ it is measured in bits (the minimum average number of binary questions, or code length, needed to describe a sample); with the natural log, in nats. Deep learning code uses nats because that is what `torch.log` returns. Entropy is maximal for the uniform distribution ($\log K$) and zero for a deterministic one.

The **cross-entropy** of a model $q$ relative to a true distribution $p$ is the average surprisal when the data come from $p$ but we score them with $q$:

$$
H(p,q)=-\sum_xp(x)\log q(x)=\E_{x\sim p}[-\log q(x)].
$$

The **Kullback-Leibler divergence** is the extra surprisal paid for using the wrong distribution:

$$
\KL(p\,\|\,q)=\sum_xp(x)\log\frac{p(x)}{q(x)}=H(p,q)-H(p).
$$

{{figure: week01-info | Cross-entropy decomposes into entropy (the irreducible part that depends only on the data) plus KL divergence (the penalty for the model q being different from p).}}

:::math KL is non-negative (Gibbs' inequality)
Because $-\log$ is convex, Jensen's inequality ($\E[\phi(X)]\ge\phi(\E X)$ for convex $\phi$) gives

$$
\KL(p\|q)=\E_{x\sim p}\Big[-\log\frac{q(x)}{p(x)}\Big]\ \ge\ -\log\E_{x\sim p}\Big[\frac{q(x)}{p(x)}\Big]=-\log\sum_{x:p(x)>0}q(x)\ \ge\ -\log1=0,
$$

with equality iff $p=q$. Properties to remember: KL is **not symmetric** and is not a distance; $\KL(p\|q)=\infty$ if $q$ assigns zero probability where $p$ does not; and the direction matters. Minimizing $\KL(p\|q)$ over $q$ (the "forward" direction) forces $q$ to cover everything $p$ covers (mass-covering); minimizing $\KL(q\|p)$ makes $q$ concentrate on one mode of $p$ (mode-seeking). Variational inference ([Week 13](week-13.html)) uses the second.
:::

### Why minimizing cross-entropy is maximum likelihood

Let $p_{\text{data}}$ be the empirical distribution of the training set. Then

$$
H(p_{\text{data}},q_{\boldsymbol\theta})=-\frac1N\sum_{i=1}^N\log q_{\boldsymbol\theta}(z_i)
$$

is exactly the average NLL. Since $H(p_{\text{data}})$ does not depend on $\boldsymbol\theta$, minimizing the cross-entropy, minimizing $\KL(p_{\text{data}}\|q_{\boldsymbol\theta})$ and maximizing the likelihood are the **same optimization problem**. Three viewpoints, one objective. This also tells you what the best conceivable loss is: you can never get below $H(p_{\text{data}})$, which is the intrinsic uncertainty in the labels. A language model's per-token loss, for instance, is bounded below by the entropy of English given the context.

The plot below shows, for a true Bernoulli$(0.7)$ source, the cross-entropy and KL of a model $q$ with head probability $q$ (bits). The cross-entropy bottoms out at $H(0.7)\approx0.881$ bits, and the KL is the vertical gap above that floor, reaching zero at $q=0.7$.

```chart
{
  "type": "line",
  "caption": "Cross-entropy H(p, q) and KL(p || q) in bits for p = Bernoulli(0.7) as the model's head probability q varies (computed from the definitions). The entropy H(p) = 0.881 bits is the floor.",
  "height": 320,
  "data": {
    "labels": [0.01, 0.05, 0.09, 0.13, 0.17, 0.21, 0.26, 0.3, 0.34, 0.38, 0.42, 0.46, 0.5, 0.54, 0.58, 0.62, 0.66, 0.7, 0.74, 0.79, 0.83, 0.87, 0.91, 0.95, 0.99],
    "datasets": [
      { "label": "cross-entropy H(p, q)", "data": [4.655, 3.031, 2.455, 2.103, 1.852, 1.661, 1.507, 1.382, 1.277, 1.189, 1.115, 1.052, 1.0, 0.958, 0.924, 0.9, 0.886, 0.881, 0.889, 0.91, 0.951, 1.018, 1.131, 1.342, 2.003], "borderColor": "var(--c-blue)", "pointRadius": 0, "tension": 0.3 },
      { "label": "KL(p || q)", "data": [3.774, 2.15, 1.574, 1.221, 0.971, 0.779, 0.626, 0.501, 0.396, 0.308, 0.233, 0.171, 0.119, 0.076, 0.043, 0.019, 0.004, 0.0, 0.007, 0.029, 0.069, 0.137, 0.25, 0.461, 1.122], "borderColor": "var(--c-red)", "pointRadius": 0, "tension": 0.3 }
    ]
  },
  "options": { "scales": { "x": { "title": { "display": true, "text": "model probability q of heads" } }, "y": { "title": { "display": true, "text": "bits" } } } }
}
```

Explore the same quantities with arbitrary categorical distributions in the next demo: drag the bars of $p$ and $q$, and watch entropy, cross-entropy and KL change.

{{demo: week01-entropy-kl | Entropy, cross-entropy and KL divergence of two distributions | 520}}

### Numerical stability: log-sum-exp

Information-theoretic losses involve $\log$ of probabilities that come from a softmax, and naive code overflows. The **softmax** of logits $\boldsymbol z$ is $\softmax(\boldsymbol z)_k=e^{z_k}/\sum_je^{z_j}$. Subtracting $m=\max_jz_j$ changes nothing mathematically, since the factor $e^{-m}$ cancels, but makes the largest exponent zero. The log-softmax is $z_k-\log\sum_je^{z_j}=z_k-m-\log\sum_je^{z_j-m}$ (the **log-sum-exp trick**). Never compute `log(softmax(z))` directly: use `log_softmax`, and use `F.cross_entropy` on raw logits, which fuses both stably.

```python:stable_logsoftmax.py
import torch
import torch.nn.functional as F

z = torch.tensor([1000., 1001., 1002.])        # large logits

naive = torch.exp(z) / torch.exp(z).sum()       # exp(1000) overflows float32 -> inf/inf = nan
print("naive softmax :", naive)

m = z.max()
stable = torch.exp(z - m) / torch.exp(z - m).sum()
print("shifted       :", stable)
print("log_softmax   :", F.log_softmax(z, dim=0))
print("logsumexp     :", torch.logsumexp(z, dim=0))

# Cross-entropy on raw logits == NLL of log_softmax
logits = torch.randn(4, 10)
target = torch.tensor([3, 1, 0, 9])
ce1 = F.cross_entropy(logits, target)
ce2 = F.nll_loss(F.log_softmax(logits, dim=1), target)
print(ce1.item(), ce2.item())                   # identical
print("uniform guess loss log(10) =", torch.log(torch.tensor(10.)).item())
```

:::tip A free sanity check
A randomly initialized $K$-class classifier should start with cross-entropy near $\log K$ (about 2.30 for 10 classes), because its outputs are close to uniform. If your first loss is 50, or $0.01$, something is wrong before training even begins.
:::

## Gradient descent: the algorithm you will use all semester

We now have a loss $\mathcal L(\boldsymbol\theta)$ and its gradient. **Gradient descent** repeats the update

$$
\boldsymbol\theta_{t+1}=\boldsymbol\theta_t-\eta\,\grad_{\boldsymbol\theta}\mathcal L(\boldsymbol\theta_t),
$$

where $\eta>0$ is the **learning rate** (step size). The first-order Taylor expansion shows why this works: for a small step $-\eta\boldsymbol g$, $\mathcal L(\boldsymbol\theta-\eta\boldsymbol g)\approx\mathcal L(\boldsymbol\theta)-\eta\norm{\boldsymbol g}^2$, a decrease. "Small" is the catch: the approximation holds only if $\eta$ is small compared to the curvature scale.

:::math Convergence on a quadratic
Take $\mathcal L(w)=\tfrac12\lambda w^2$ with curvature $\lambda>0$ and minimizer $w^\star=0$. The update is $w_{t+1}=w_t-\eta\lambda w_t=(1-\eta\lambda)w_t$, so $w_t=(1-\eta\lambda)^tw_0$. This converges iff $|1-\eta\lambda|<1$, i.e. $0<\eta<2/\lambda$. At $\eta=1/\lambda$ it lands on the minimum in one step; for $1/\lambda<\eta<2/\lambda$ it overshoots and oscillates while converging; above $2/\lambda$ it diverges. For a multi-dimensional quadratic with Hessian eigenvalues $\lambda_1\ge\dots\ge\lambda_d>0$, each eigen-direction contracts by $(1-\eta\lambda_i)$, so stability needs $\eta<2/\lambda_{\max}$, while progress along the flattest direction is governed by $\eta\lambda_{\min}$. The best rate is limited by the **condition number** $\kappa=\lambda_{\max}/\lambda_{\min}$: ill-conditioned problems are slow. (You will derive the optimal step in the exercises.)
:::

For our linear regression loss $\frac1N\norm{\mathbf X\boldsymbol w-\boldsymbol y}^2$ the Hessian is $\frac2N\mathbf X^\top\mathbf X$. With standardized features its top eigenvalue is about 2, so we expect divergence for $\eta$ above about 1. The simulation below confirms this prediction.

```chart
{"type": "line", "caption": "Training loss of one-feature linear regression (N = 100, true w = 3, b = 2, noise sd 0.5) under full-batch gradient descent with four learning rates; computed with the NumPy equivalent of the lab code (seed 0). Mean-squared-error loss floors at about 0.23, the noise variance. The sweet spot is not the largest stable rate: lr = 0.9 oscillates before settling, and lr = 1.05 (not shown) diverges.", "height": 340, "data": {"labels": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39], "datasets": [{"label": "lr = 0.01", "data": [13.6321, 13.0852, 12.5608, 12.0577, 11.5753, 11.1126, 10.6688, 10.2432, 9.835, 9.4435, 9.068, 8.7079, 8.3625, 8.0311, 7.7134, 7.4086, 7.1163, 6.8359, 6.567, 6.309, 6.0616, 5.8243, 5.5967, 5.3783, 5.1689, 4.9681, 4.7754, 4.5906, 4.4133, 4.2433, 4.0801, 3.9237, 3.7736, 3.6296, 3.4915, 3.3591, 3.232, 3.1101, 2.9932, 2.881], "borderColor": "var(--c-muted)", "pointRadius": 0, "tension": 0.2}, {"label": "lr = 0.05", "data": [13.6321, 11.012, 8.9057, 7.212, 5.8499, 4.7544, 3.8731, 3.1639, 2.5932, 2.1339, 1.764, 1.4663, 1.2264, 1.0332, 0.8775, 0.7521, 0.651, 0.5694, 0.5037, 0.4506, 0.4078, 0.3733, 0.3454, 0.3229, 0.3047, 0.29, 0.2781, 0.2685, 0.2607, 0.2545, 0.2494, 0.2453, 0.2419, 0.2392, 0.2371, 0.2353, 0.2338, 0.2327, 0.2317, 0.231], "borderColor": "var(--c-blue)", "pointRadius": 0, "tension": 0.2}, {"label": "lr = 0.2", "data": [13.6321, 4.8661, 1.8442, 0.7957, 0.4292, 0.2999, 0.2538, 0.2373, 0.2312, 0.229, 0.2282, 0.2278, 0.2277, 0.2277, 0.2277, 0.2276, 0.2276, 0.2276, 0.2276, 0.2276, 0.2276, 0.2276, 0.2276, 0.2276, 0.2276, 0.2276, 0.2276, 0.2276, 0.2276, 0.2276, 0.2276, 0.2276, 0.2276, 0.2276, 0.2276, 0.2276, 0.2276, 0.2276, 0.2276, 0.2276], "borderColor": "var(--c-green)", "pointRadius": 0, "tension": 0.2}, {"label": "lr = 0.9", "data": [13.6321, 10.1832, 7.9603, 6.3576, 5.1303, 4.1636, 3.3926, 2.7743, 2.2773, 1.8776, 1.5558, 1.2969, 1.0884, 0.9206, 0.7855, 0.6767, 0.5891, 0.5187, 0.4619, 0.4162, 0.3795, 0.3499, 0.326, 0.3069, 0.2914, 0.279, 0.269, 0.2609, 0.2544, 0.2492, 0.245, 0.2416, 0.2389, 0.2367, 0.2349, 0.2335, 0.2324, 0.2314, 0.2307, 0.2301], "borderColor": "var(--c-amber)", "pointRadius": 0, "tension": 0.2}]}, "options": {"scales": {"x": {"title": {"display": true, "text": "step"}}, "y": {"type": "logarithmic", "title": {"display": true, "text": "training MSE (log scale)"}}}}}
```

Use the demo to build the same intuition in two dimensions. Pick a loss surface (a well-conditioned bowl, an elongated ill-conditioned valley, or a non-convex surface), click to set the starting point, and run gradient descent with different step sizes. Watch the zig-zag in the narrow valley: it is the signature of high condition number and the motivation for momentum and adaptive methods in [Week 3](week-03.html).

{{demo: week01-gd-playground | Gradient descent on 2D loss surfaces | 560}}

## Tensors, GPUs and the PyTorch programming model

### Why PyTorch, and what it gives you

PyTorch (Paszke et al., 2019) is the framework we use. It offers three things: (1) an **n-dimensional array** (`torch.Tensor`) with NumPy-like semantics that can live on a CPU or a GPU; (2) **automatic differentiation** (`autograd`), which records operations on tensors and computes gradients of any scalar with respect to any tensor that `requires_grad`; and (3) **building blocks** (`torch.nn` modules, `torch.optim` optimizers, `torch.utils.data` loaders) for assembling and training models. Its defining design choice is **define-by-run**: the computation graph is built dynamically as ordinary Python code executes, so you can use `if`, `for` and the debugger freely. We unpack how this works in [Week 2](week-02.html#how-pytorch-autograd-works).

### Tensor attributes: shape, dtype, device

Every tensor has a **dtype** (`float32` by default for floating point; `float64`, `float16`, `bfloat16`, `int64`, `bool`...), a **device** (`cpu`, `cuda:0`, `mps`), a **shape**, and **strides** describing how the multi-dimensional index maps to memory. Operations between tensors require compatible dtype (with some promotion rules) and **identical devices**. Mixed device operands raise an error, which is a feature.

Precision matters. `float32` has about 7 decimal digits of precision and a maximum near $3.4\times10^{38}$. `float16` has a 5-bit exponent (maximum 65504) and 10 mantissa bits, so it overflows easily; `bfloat16` keeps the 8-bit exponent of `float32` with only 7 mantissa bits, trading precision for range, which is why it is the default for large-model training on modern accelerators. Training in reduced precision is covered in [Week 17](week-17.html); for this course's small models and for gradient checking we sometimes use `float64`.

**Views versus copies.** `reshape`, `view`, `transpose`, `permute` and slicing usually return **views** that share memory with the original. Modifying one modifies the other. `clone()` makes a copy. A transposed tensor is non-contiguous; `view` fails on it, while `reshape` silently copies if needed. In-place operations (names ending in `_`, such as `add_`, or `+=`) mutate memory and can break autograd, which sometimes needs the overwritten value for the backward pass.

```python:autograd_basics.py
import torch

# 1. A tiny scalar graph: z = (x*y + sin(x))**2
x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)
z = (x * y + torch.sin(x)) ** 2
z.backward()                                   # populates x.grad and y.grad
# by hand: u = x*y + sin x;  dz/dx = 2u (y + cos x);  dz/dy = 2u x
u = (x * y + torch.sin(x)).item()
print(x.grad.item(), 2 * u * (3.0 + torch.cos(torch.tensor(2.0)).item()))
print(y.grad.item(), 2 * u * 2.0)

# 2. Gradients ACCUMULATE: calling backward twice adds into .grad
w = torch.tensor([1.0, 2.0], requires_grad=True)
for _ in range(2):
    loss = (w ** 2).sum()
    loss.backward()
print(w.grad)                                  # tensor([4., 8.]) not [2., 4.]
w.grad.zero_()                                 # reset before the next step

# 3. Stop tracking: for evaluation or when updating parameters manually
with torch.no_grad():
    w -= 0.1 * w.grad                          # in-place update outside the graph
y_pred = (w * 3).detach()                      # shares data, no graph
print(w.requires_grad, y_pred.requires_grad)   # True False

# 4. Non-scalar outputs need an explicit upstream gradient (a VJP)
v = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
out = v ** 2                                   # vector output
out.backward(torch.ones_like(out))             # computes (d out / d v)^T @ ones
print(v.grad)                                  # 2 v = tensor([2., 4., 6.])
```

### GPUs: why they matter and how to use them

A CPU has a few powerful cores optimized for fast sequential, branchy code. A GPU has thousands of simpler cores optimized for doing the *same* arithmetic on huge amounts of data in parallel, with far higher memory bandwidth. Matrix multiplication and convolution are exactly this kind of workload, so a GPU can be 10 to 100 times faster than a CPU for the matrix products that dominate deep learning.

{{figure: week01-gpu | CPU and GPU have separate memories connected by a comparatively slow bus. Data must be moved to the GPU, and the performance rule is to keep data on the device and minimize transfers and synchronization.}}

The practical rules are few:

1. Move the **model** and every **input batch** to the same device: `model.to(device)`, `x = x.to(device)`.
2. GPU kernels launch **asynchronously**: Python returns immediately while the GPU works. Timing code must call `torch.cuda.synchronize()` before reading the clock, and calls like `.item()`, `.cpu()` or `print(tensor)` force synchronization.
3. Avoid per-element Python loops; express the computation as batched tensor operations.
4. Beware the transfer cost: copying small tensors back and forth every step can dominate runtime. Accumulate metrics on the device and convert once per epoch.
5. GPU memory is limited. Activations saved for backward dominate it; batch size and model size trade off against each other.

```python:devices_and_timing.py
import time
import torch

# Pick the best available device; the same code then runs on laptop CPU or Colab GPU
if torch.cuda.is_available():
    device = torch.device("cuda")
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print("using", device)

n = 2048
A = torch.randn(n, n, device=device)           # create directly on the device (no copy)
B = torch.randn(n, n, device=device)

def bench(fn, reps=10):
    fn()                                       # warm-up (kernel selection, caches)
    if device.type == "cuda":
        torch.cuda.synchronize()               # wait for queued GPU work
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps

t = bench(lambda: A @ B)
flops = 2 * n ** 3                              # n^3 multiply-adds = 2 n^3 floating-point ops
print(f"matmul {n}x{n}: {t*1e3:.2f} ms  ->  {flops / t / 1e9:.1f} GFLOP/s")

# Moving data costs time; keep tensors on the device
x_cpu = torch.randn(n, n)
x_dev = x_cpu.to(device)                        # host -> device copy
y_cpu = (x_dev @ x_dev).cpu()                   # device -> host copy forces a sync
print(y_cpu.device, y_cpu.shape)
```

### Reproducibility and random seeds

Deep learning experiments are stochastic: weight initialization, data shuffling, dropout masks and data augmentation all draw random numbers. For debugging and for scientific honesty you want to be able to repeat a run, and to know how much of an observed difference is mere seed variance.

The practical recipe is to seed every random number generator you use, pass explicit generators to data loaders, and (when exact repeatability matters more than speed) request deterministic algorithms. Know the limits: bit-exact results are only guaranteed on the same hardware, library versions and settings, since floating-point addition is not associative and parallel reductions may sum in a different order. Some CUDA kernels are nondeterministic by default.

```python:reproducibility.py
import os, random
import numpy as np
import torch

def seed_everything(seed: int = 0, deterministic: bool = False) -> None:
    random.seed(seed)                          # Python's RNG
    np.random.seed(seed)                       # NumPy's global RNG
    torch.manual_seed(seed)                    # CPU and (all) CUDA devices
    torch.cuda.manual_seed_all(seed)           # explicit, harmless on CPU-only machines
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        # Raises an error if an op has no deterministic implementation
        torch.use_deterministic_algorithms(True)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    # DataLoader workers get their own seeds; make numpy/random follow torch's per-worker seed
    s = torch.initial_seed() % 2**32
    np.random.seed(s); random.seed(s)

seed_everything(42)
a = torch.randn(3)
seed_everything(42)
b = torch.randn(3)
print(torch.equal(a, b))                       # True: same seed, same stream

# A dedicated generator keeps data shuffling independent of other random calls
g = torch.Generator().manual_seed(123)
# loader = DataLoader(ds, batch_size=64, shuffle=True, generator=g, worker_init_fn=seed_worker)
```

:::tip Seeds are not a hyperparameter to tune
If a result only appears with one lucky seed, it is not a result. Report the mean and spread over several seeds (three to five for small experiments). A seed gives you *repeatability*; multiple seeds give you *reliability*.
:::

## Lab: linear regression with gradient descent in PyTorch

Linear regression is the smallest model that has every part of a deep learning training loop: parameters, a forward pass, a loss, a backward pass, an update. The computation graph for one loss evaluation is shown below; autograd will traverse it backwards to produce $\partial\mathcal L/\partial\boldsymbol w$ and $\partial\mathcal L/\partial b$.

{{figure: week01-graph-linreg | The computation graph of the mean-squared-error loss of a linear model. Solid arrows are the forward pass; accent-colored arrows are the gradient flowing backward during backward().}}

We will fit $y=3x_1-2x_2+1+\varepsilon$ with $\varepsilon\sim\mathcal N(0,0.1^2)$ in three ways: closed form, manual gradient descent with autograd, and the idiomatic `nn.Module` plus `optim` loop. All three should agree.

```python:lab_linreg_manual.py
import torch

torch.manual_seed(0)

# ---- 1. Synthetic data -------------------------------------------------
N, d = 200, 2
true_w = torch.tensor([3.0, -2.0])
true_b = 1.0
X = torch.randn(N, d)                                   # (200, 2)
y = X @ true_w + true_b + 0.1 * torch.randn(N)          # (200,)
print("X", X.shape, "y", y.shape)

# ---- 2. Closed form (normal equations) via least squares ----------------
X1 = torch.cat([X, torch.ones(N, 1)], dim=1)            # append a column of ones for the bias
sol = torch.linalg.lstsq(X1, y.unsqueeze(1)).solution.squeeze(1)
print("closed form  w, b:", sol[:d].tolist(), sol[d].item())

# ---- 3. Gradient descent with autograd ----------------------------------
w = torch.zeros(d, requires_grad=True)
b = torch.zeros(1, requires_grad=True)
lr = 0.1
losses = []
for step in range(200):
    y_hat = X @ w + b                                   # forward: (200,)
    loss = ((y_hat - y) ** 2).mean()                    # scalar MSE
    loss.backward()                                     # fills w.grad (2,), b.grad (1,)
    with torch.no_grad():                               # update outside the graph
        w -= lr * w.grad
        b -= lr * b.grad
    w.grad.zero_(); b.grad.zero_()                      # gradients accumulate: reset!
    losses.append(loss.item())
    if step % 50 == 0:
        print(f"step {step:3d}  loss {loss.item():.5f}")
print("gradient descent  w, b:", w.detach().tolist(), b.item())
```

```text
X torch.Size([200, 2]) y torch.Size([200])
closed form  w, b: [2.99..., -2.00...], 1.00...
step   0  loss 13.6...
step  50  loss 0.01...
gradient descent  w, b: [2.99..., -2.00...], 1.00...
```

(Your digits will differ slightly; the point is that all methods recover roughly $(3,-2,1)$ and the loss floors near the noise variance $0.1^2=0.01$.) Now the same model written the way you will write every model in the course:

```python:lab_linreg_module.py
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"

N, d = 1000, 2
X = torch.randn(N, d)
y = X @ torch.tensor([3.0, -2.0]) + 1.0 + 0.1 * torch.randn(N)

# Train/validation split
perm = torch.randperm(N)
tr, va = perm[:800], perm[800:]
train_ds = TensorDataset(X[tr], y[tr].unsqueeze(1))      # targets as (n, 1) to match model output
val_X, val_y = X[va].to(device), y[va].unsqueeze(1).to(device)
train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)

model = nn.Linear(d, 1).to(device)                       # weight (1, 2), bias (1,)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

for epoch in range(20):
    model.train()
    for xb, yb in train_dl:                              # xb (32, 2), yb (32, 1)
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()                            # 1. clear old gradients
        loss = criterion(model(xb), yb)                  # 2. forward + loss
        loss.backward()                                  # 3. backward
        optimizer.step()                                 # 4. parameter update
    model.eval()
    with torch.no_grad():
        val_loss = criterion(model(val_X), val_y).item()
    if epoch % 5 == 0 or epoch == 19:
        print(f"epoch {epoch:2d}  val MSE {val_loss:.4f}")

print("learned weight:", model.weight.data.cpu().flatten().tolist())
print("learned bias  :", model.bias.item())
```

Those four lines inside the loop (zero the gradients, forward, backward, step) are the skeleton of essentially all deep learning training code. Everything that follows in this course changes what `model` is, what `criterion` is, what `optimizer` is, or how the batches are produced.

:::note Experiments to run in the lab
1. Change `lr` to 0.9 and 1.5 in the manual loop and compare with the chart above. Which stays stable? Compute $\lambda_{\max}$ of $\frac2N\mathbf X^\top\mathbf X$ with `torch.linalg.eigvalsh` to check the $2/\lambda_{\max}$ prediction.
2. Multiply the second feature by 100 (`X[:, 1] *= 100`) and rerun with the old learning rate. Explain what happens using the condition number. Then standardize features and repeat.
3. Remove `w.grad.zero_()` and observe the effect on convergence.
4. Add label noise of scale 1.0 and see where the loss floors.
:::

## Putting it together: a map of what comes next

You now have the main ingredients of the course. The model $f_{\boldsymbol\theta}$ is a differentiable composition of layers; the loss is a negative log-likelihood, equivalently a cross-entropy or KL to the data distribution; training is gradient descent, with gradients computed by reverse-mode autodiff on a graph. [Week 2](week-02.html) builds the model (the multilayer perceptron), derives backpropagation in full, and implements it from scratch. [Week 3](week-03.html) makes the descent work well (optimizers, schedules, initialization, normalization). [Week 4](week-04.html) controls generalization (regularization, evaluation, experiment hygiene). Weeks 5 to 12 then specialize the architecture to images, sequences and attention.

## Common mistakes and debugging tips

These are the errors that cost beginners the most time. Each has a recognizable symptom.

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| Loss is `nan` after a few steps | Learning rate too high, `log(0)`, division by zero, overflow in `exp` | Lower the learning rate; use `log_softmax`/`cross_entropy` on logits; add `eps` inside logs and square roots; check inputs for `nan` |
| Loss does not decrease at all | Forgot `optimizer.step()`; parameters not passed to the optimizer; `requires_grad=False`; learning rate far too small | Print a parameter before and after one step; check `p.grad` is not `None` |
| Loss decreases then explodes | Learning rate above the stability threshold ($2/\lambda_{\max}$) | Reduce the rate, normalize inputs, clip gradients |
| Gradients keep growing, odd convergence | Forgot `zero_grad()`, so gradients accumulate across steps | Zero gradients each iteration (or set `set_to_none=True`) |
| Shape error or, worse, silent wrong results | Broadcasting an `(N,)` against `(N,1)` producing `(N,N)` | Assert shapes: `assert pred.shape == target.shape`; use `unsqueeze`/`squeeze` deliberately |
| `RuntimeError: Expected all tensors to be on the same device` | Model on GPU, batch on CPU, or the reverse | Move both with `.to(device)` |
| `RuntimeError: a leaf Variable that requires grad is being used in an in-place operation` | Updating a parameter in place outside `torch.no_grad()` | Wrap manual updates in `with torch.no_grad():` |
| Validation numbers are better than they should be | Test data leaked into training or normalization statistics; evaluating in `train()` mode | Split first, compute statistics on the training set only; call `model.eval()` and use `torch.no_grad()` |
| Memory grows every iteration | Storing `loss` (a tensor with a graph) in a list | Store `loss.item()` or `loss.detach()` |
| Results change from run to run | Unseeded randomness, nondeterministic kernels | Use the seeding recipe; compare over multiple seeds |
| Training is slow | Python loops over elements; data loading on the main thread; many tiny GPU ops; tensors bouncing between devices | Vectorize; use `num_workers`; move data once; profile |

A general principle: **overfit a tiny batch first**. If your model cannot drive the loss to nearly zero on 8 to 32 examples, there is a bug in the model, the loss or the data pipeline, and no amount of tuning on the full dataset will fix it. Also check the **initial loss** against what chance performance predicts ($\log K$ for $K$ classes).

## Exercises

:::exercise Exercise 1: matrix calculus (derivation)
Let $f(\boldsymbol x)=\norm{\mathbf A\boldsymbol x-\boldsymbol b}^2$ with $\mathbf A\in\R^{m\times n}$. Derive $\grad_{\boldsymbol x}f$ and the Hessian, and state when $f$ has a unique minimizer.

:::details Solution
Write $\boldsymbol r=\mathbf A\boldsymbol x-\boldsymbol b$, so $f=\boldsymbol r^\top\boldsymbol r$. Then $df=2\boldsymbol r^\top d\boldsymbol r=2\boldsymbol r^\top\mathbf A\,d\boldsymbol x$, so $\grad f=2\mathbf A^\top(\mathbf A\boldsymbol x-\boldsymbol b)$. Differentiating again, $\mathbf H=2\mathbf A^\top\mathbf A\succeq0$, so $f$ is convex. The minimizer is unique iff $\mathbf A^\top\mathbf A$ is positive definite, i.e. $\mathbf A$ has full column rank ($\operatorname{rank}\mathbf A=n$, which needs $m\ge n$). Then $\boldsymbol x^\star=(\mathbf A^\top\mathbf A)^{-1}\mathbf A^\top\boldsymbol b$. If the columns are dependent, the minimizers form an affine subspace.
:::
:::

:::exercise Exercise 2: shapes and broadcasting (conceptual)
Predict the shape of each result, then run it. (a) `torch.randn(8,1,5) + torch.randn(3,5)`; (b) `torch.randn(4) * torch.randn(4,1)`; (c) `torch.randn(2,3,4) @ torch.randn(4,5)`; (d) `torch.randn(2,3,4).sum(dim=1, keepdim=True)`; (e) `torch.randn(6,3)[:, None, :] - torch.randn(5,3)[None, :, :]`. Explain what (e) computes and why it is useful.

:::details Solution
(a) Align trailing axes: `(8,1,5)` with `(1,3,5)` gives `(8,3,5)`. (b) `(1,4)` with `(4,1)` gives `(4,4)`, an outer product. (c) matmul broadcasts the batch: `(2,3,5)`. (d) `(2,1,4)`. (e) `(6,1,3) - (1,5,3)` gives `(6,5,3)`: all pairwise difference vectors between 6 points and 5 points. Taking the squared norm over the last axis gives the `(6,5)` matrix of pairwise squared distances, the basis of nearest-neighbor and kernel computations without Python loops.
:::
:::

:::exercise Exercise 3: maximum likelihood of a Gaussian (derivation)
Given i.i.d. samples $x_1,\dots,x_N\sim\mathcal N(\mu,\sigma^2)$, derive the MLE of $\mu$ and $\sigma^2$. Is $\hat\sigma^2_{\text{MLE}}$ unbiased?

:::details Solution
$\ell(\mu,\sigma^2)=-\tfrac N2\log(2\pi\sigma^2)-\frac1{2\sigma^2}\sum_i(x_i-\mu)^2$. Setting $\partial\ell/\partial\mu=\frac1{\sigma^2}\sum_i(x_i-\mu)=0$ gives $\hat\mu=\frac1N\sum_ix_i$. Setting $\partial\ell/\partial\sigma^2=-\frac N{2\sigma^2}+\frac1{2\sigma^4}\sum_i(x_i-\mu)^2=0$ gives $\hat\sigma^2=\frac1N\sum_i(x_i-\hat\mu)^2$. This estimator is biased: $\E[\hat\sigma^2]=\frac{N-1}N\sigma^2$, because $\hat\mu$ was fit to the same data and so the residuals are slightly too small. The unbiased version divides by $N-1$ (Bessel's correction). The same effect is why training loss underestimates test loss.
:::
:::

:::exercise Exercise 4: KL is asymmetric (computation)
Let $p=(0.5,0.5)$ and $q=(0.9,0.1)$. Compute $\KL(p\|q)$ and $\KL(q\|p)$ in nats, and $H(p,q)$. Then let $q'=(1,0)$ and discuss $\KL(p\|q')$ versus $\KL(q'\|p)$.

:::details Solution
$\KL(p\|q)=0.5\ln\frac{0.5}{0.9}+0.5\ln\frac{0.5}{0.1}=0.5(-0.588)+0.5(1.609)=0.511$. $\KL(q\|p)=0.9\ln\frac{0.9}{0.5}+0.1\ln\frac{0.1}{0.5}=0.9(0.588)+0.1(-1.609)=0.368$. $H(p)=\ln2=0.693$, so $H(p,q)=0.693+0.511=1.204$. For $q'=(1,0)$: $\KL(p\|q')=\infty$, because $p$ puts mass 0.5 on an outcome that $q'$ says is impossible; but $\KL(q'\|p)=1\cdot\ln\frac1{0.5}=0.693$ is finite. This is why classifiers trained with cross-entropy must never output exactly zero probability for an outcome that can occur, and why label smoothing and numerical clamping exist.
:::
:::

:::exercise Exercise 5: coding a stable cross-entropy
Implement `my_cross_entropy(logits, targets)` for logits of shape `(B, K)` and integer targets of shape `(B,)` using only `torch.logsumexp` and indexing. Verify against `F.cross_entropy` and test with logits of magnitude 1000.

:::details Solution
```python
import torch
import torch.nn.functional as F

def my_cross_entropy(logits, targets):
    lse = torch.logsumexp(logits, dim=1)                          # (B,)
    picked = logits.gather(1, targets.unsqueeze(1)).squeeze(1)    # (B,) logit of the true class
    return (lse - picked).mean()                                  # -log softmax at the target

logits = torch.randn(5, 7) * 1000
t = torch.randint(0, 7, (5,))
print(my_cross_entropy(logits, t).item(), F.cross_entropy(logits, t).item())
```
The loss for one example is $-\log\softmax(\boldsymbol z)_y=\log\sum_je^{z_j}-z_y$, and `logsumexp` applies the max-shift internally. A naive `-torch.log(torch.softmax(z,1))` also works for moderate logits but produces `inf` when a probability underflows to zero.
:::
:::

:::exercise Exercise 6: optimal step size (challenging)
Gradient descent on $\mathcal L(\boldsymbol w)=\tfrac12\boldsymbol w^\top\mathbf H\boldsymbol w$, with $\mathbf H$ symmetric positive definite with eigenvalues in $[\lambda_{\min},\lambda_{\max}]$. (a) Show $\boldsymbol w_t=(\mathbf I-\eta\mathbf H)^t\boldsymbol w_0$ and find the range of $\eta$ for convergence. (b) Find the $\eta$ minimizing the worst-case contraction factor $\rho(\eta)=\max_i|1-\eta\lambda_i|$ and the resulting rate in terms of $\kappa=\lambda_{\max}/\lambda_{\min}$. (c) For $\kappa=100$ how many iterations are needed to reduce $\norm{\boldsymbol w}$ by a factor $10^{-3}$?

:::details Solution
(a) The gradient is $\mathbf H\boldsymbol w$, so $\boldsymbol w_{t+1}=(\mathbf I-\eta\mathbf H)\boldsymbol w_t$. In the eigenbasis each coordinate is multiplied by $1-\eta\lambda_i$ per step; convergence for all $i$ requires $|1-\eta\lambda_{\max}|<1$, i.e. $0<\eta<2/\lambda_{\max}$. (b) $\rho(\eta)=\max(|1-\eta\lambda_{\min}|,|1-\eta\lambda_{\max}|)$. The first term decreases in $\eta$, the second increases once $\eta>1/\lambda_{\max}$; the worst case is minimized where $1-\eta\lambda_{\min}=\eta\lambda_{\max}-1$, i.e. $\eta^\star=2/(\lambda_{\min}+\lambda_{\max})$, giving $\rho^\star=(\kappa-1)/(\kappa+1)$. (c) $\rho^\star=99/101\approx0.9802$; we need $\rho^t\le10^{-3}$, so $t\ge\ln(10^{-3})/\ln(0.9802)\approx-6.908/-0.02 \approx 342$ iterations. With $\kappa=10^4$ it would be about 34,500: ill-conditioning is brutal, which is why momentum (rate $\approx(\sqrt\kappa-1)/(\sqrt\kappa+1)$) and preconditioning matter.
:::
:::

:::exercise Exercise 7: what would happen if...? (conceptual)
You train the linear model of the lab but (a) forget `zero_grad()`; (b) shuffle labels `y` randomly before training; (c) use `float16` throughout with a learning rate of 1e-8. Predict the behaviour of each.

:::details Solution
(a) Gradients accumulate, so the update at step $t$ uses the sum of all gradients so far: an effective learning rate that grows with $t$, typically leading to oscillation or divergence (momentum-like but without damping). (b) The labels carry no information about $x$, so the best linear model predicts the label mean: weights go to about zero and the loss floors at the label variance. This is the "random labels" sanity check: a model that gets lower training loss than that is memorizing, which a linear model with $N\gg d$ cannot. (c) `float16` has about 3 decimal digits of precision; an update of size $10^{-8}\times$gradient is smaller than the spacing between representable numbers near the weights, so the parameters do not change (updates are rounded away); this is why mixed-precision training keeps a float32 master copy of the weights.
:::
:::

:::exercise Exercise 8: memory budget (computation)
A multilayer perceptron has layers $784\to4096\to4096\to10$ with biases. (a) Count its parameters. (b) How much memory do the parameters take in float32? (c) Training with Adam stores, per parameter, the weight, the gradient and two moment estimates. Estimate the total. (d) Compare with the activation memory for a batch of 256 (counting only the hidden layer outputs).

:::details Solution
(a) $784\cdot4096+4096=3{,}215{,}360$; $4096\cdot4096+4096=16{,}781{,}312$; $4096\cdot10+10=40{,}970$. Total $\approx20.04$ million. (b) $\times4$ bytes $\approx80.2$ MB. (c) Four copies: about $320.6$ MB. (d) Hidden outputs per example: $4096+4096=8192$ floats; for 256 examples about $2.1$ million floats $\approx8.4$ MB (and a similar amount again for pre-activations kept for backward). For this MLP, parameter-related memory dominates; for convolutional networks on large images the situation reverses and activations dominate.
:::
:::

:::exercise Exercise 9: Laplace noise and robust regression (challenging)
Suppose the noise in a regression problem is Laplace: $p(\varepsilon)=\frac1{2b}\exp(-|\varepsilon|/b)$. Derive the MLE loss, give its gradient with respect to the prediction, and explain in one paragraph when you would prefer it over squared error. Then implement both losses in PyTorch and fit a line to data with 5% outliers of magnitude 50, comparing the fitted slopes.

:::details Solution
The NLL of one example is $|y-\hat y|/b+\log2b$, so the MLE minimizes the sum of absolute errors (L1 loss). The gradient with respect to $\hat y$ is $\operatorname{sign}(\hat y-y)/b$: bounded, in contrast with $(\hat y-y)/\sigma^2$ for squared error, which grows linearly with the error. Large outliers therefore dominate the squared-error fit but have no more influence than moderate errors under L1. Prefer L1 (or Huber loss, which is quadratic near zero and linear in the tails) when the data contain outliers or heavy-tailed noise. The L1 gradient does not vanish near the optimum, so use a decaying learning rate. A short experiment: generate $y=2x+1$, add outliers to 5% of the targets, and run gradient descent with `F.mse_loss` and `F.l1_loss`; the squared-error slope is visibly biased by the outliers while the L1 slope stays near 2.
:::
:::

:::exercise Exercise 10: reproducibility audit
A colleague reports that two runs of "the same" script with seed 0 give different final accuracies (91.2 and 91.6). List at least four plausible causes and how you would find out which applies.

:::details Solution
Candidate causes: (1) only `torch.manual_seed` was set but not NumPy or Python seeds used by augmentation; (2) DataLoader workers use different random streams (fix with `worker_init_fn` and a generator); (3) nondeterministic GPU kernels (atomic adds in some backward passes, cuDNN autotuning with `benchmark=True`); (4) different hardware or library versions between runs; (5) data order depends on file-system listing order; (6) code path with dropout in evaluation due to a missing `model.eval()`. Diagnose by bisection: log the first batch's indices and a checksum of the initial weights (these should match exactly), then compare the loss at step 1, 10, 100 across runs to find where divergence begins, then enable `torch.use_deterministic_algorithms(True)` to see whether a nondeterministic op raises. Also remember that 0.4 points may be within ordinary seed-to-seed spread, so measure it over several seeds.
:::
:::

## Key takeaways

:::key Key takeaways
- Deep learning learns a composition of differentiable layers end to end; representation learning, scale and the tricks that make deep models trainable explain its success.
- Learning is risk minimization: we minimize empirical risk on training data and care about risk on unseen data. Always separate training, validation and test data.
- Squared error is Gaussian maximum likelihood; cross-entropy is categorical maximum likelihood; $H(p,q)=H(p)+\KL(p\|q)$, so minimizing cross-entropy fits the model distribution to the data distribution.
- Gradients have the shape of the parameter. For a linear layer, the weight gradient is an outer product (or its batched sum) of upstream gradient and input; the backward pass multiplies by the transposed Jacobian.
- Gradient descent is stable only for $\eta<2/\lambda_{\max}$, and its speed is limited by the condition number.
- In PyTorch, keep tensors on one device, zero gradients, wrap manual updates in `no_grad`, seed everything, and verify shapes. Use log-sum-exp and fused cross-entropy for stability.
:::

## Further reading

- **Essential.** Goodfellow, Bengio and Courville, *Deep Learning* (MIT Press, 2016), Chapters 1 to 5 (introduction, linear algebra, probability and information theory, numerical computation, machine learning basics). Free at deeplearningbook.org.
- **Essential.** Zhang, Lipton, Li and Smola, *Dive into Deep Learning* (d2l.ai), the "Preliminaries" chapter (tensors, calculus, autograd, probability).
- **Essential.** The PyTorch documentation and tutorials (pytorch.org/tutorials), in particular "Learn the Basics" and "Autograd: automatic differentiation". Paszke et al., "PyTorch: An Imperative Style, High-Performance Deep Learning Library", NeurIPS 2019.
- LeCun, Bengio and Hinton, "Deep learning", *Nature* 521, 2015: a short survey of the field's core ideas.
- Krizhevsky, Sutskever and Hinton, "ImageNet Classification with Deep Convolutional Neural Networks", NIPS 2012.
- Russakovsky et al., "ImageNet Large Scale Visual Recognition Challenge", *International Journal of Computer Vision*, 2015 (arXiv:1409.0575).
- Telgarsky, "Benefits of depth in neural networks", COLT 2016 (arXiv:1602.04485): the sawtooth depth-separation result mentioned above.
- Petersen and Pedersen, *The Matrix Cookbook* (2012): a reference table of matrix derivative identities.
- Cover and Thomas, *Elements of Information Theory*, 2nd ed. (Wiley, 2006), Chapter 2: entropy, KL divergence and Gibbs' inequality.
- Kaplan et al., "Scaling Laws for Neural Language Models", 2020 (arXiv:2001.08361).
