---
title: "The Transformer Architecture"
phase: Attention & Pretrained Models
summary: Self-attention from first principles, multi-head attention with exact tensor shapes, feed-forward layers, residuals and normalization, positional encodings from sinusoids to RoPE and ALiBi, encoder/decoder variants and causal masking, the quadratic cost and its cures, parameter and FLOP counting, training tricks, and a from-scratch GPT in PyTorch.
---

:::key Learning objectives
By the end of this week you should be able to:
- Derive scaled dot-product attention from the idea of a soft dictionary lookup, and prove why the $1/\sqrt{d_k}$ factor keeps the softmax out of its saturated regime.
- Write multi-head attention with exact tensor shapes at every step, and implement it in PyTorch without `nn.MultiheadAttention`.
- Explain the roles of the position-wise feed-forward network, residual connections, layer normalization and RMSNorm, and argue when pre-norm is preferable to post-norm.
- Compare sinusoidal, learned, relative, rotary (RoPE) and ALiBi position schemes, and derive why RoPE makes attention scores depend only on relative offsets.
- Distinguish encoder-only, decoder-only and encoder-decoder models, and implement causal and padding masks correctly.
- Count the parameters and FLOPs of a Transformer from its hyperparameters, explain the quadratic cost of attention, and compare sparse, sliding-window and linear attention.
- Apply the standard training recipe (AdamW, warmup, label smoothing, mixed precision) and train a GPT-style decoder from scratch on a tiny corpus, then sample from it.
:::

In [Week 7](week-07.html) you saw attention as an add-on to a recurrent encoder-decoder: a decoder looked back at the encoder's hidden states with a learned weighted average. It worked so well that a natural question arose. If attention lets every output position reach every input position directly, what is the recurrence still for? Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser and Polosukhin answered in 2017 with "Attention Is All You Need": nothing. Delete the recurrence, keep the attention, add some feed-forward layers and normalization, and you get a model that is easier to parallelize on GPUs, easier to optimize at depth, and that scales to sizes no recurrent model ever reached.

The Transformer is now the default architecture for language, and has become dominant in vision ([Week 11](week-11.html)), speech, protein modeling and more. This week is therefore the pivot of the course. We build it piece by piece, from first principles, and we are pedantic about shapes because most real-world Transformer bugs are shape or masking bugs. [Week 10](week-10.html) then asks what happens when you pretrain this architecture on a large fraction of the internet.

:::note A mental model to carry through the week
A Transformer layer is two operations alternated: **communication** (attention: tokens exchange information, a weighted average over positions) and **computation** (the feed-forward network: each token is transformed independently). Residual connections form a "highway" (the *residual stream*) that every sublayer reads from and writes to. Nearly everything in this chapter is a detail about one of those three ideas.
:::

## From recurrence to attention-only models {#history}

Recurrent networks have two structural problems, both visible in Week 7. First, they are **sequential**: the hidden state $\boldsymbol{h}_t$ needs $\boldsymbol{h}_{t-1}$, so a sequence of length $n$ requires $n$ dependent steps and a GPU sits mostly idle. Second, **information from step $i$ to step $j$ must pass through $|i-j|$ nonlinear transformations**, so long-range dependencies are hard to learn even with gating. Convolutional sequence models ([Week 7](week-07.html#tcn)) shorten that path to $O(\log_k n)$ with dilation, but still not to one.

Self-attention has a path length of exactly **one** between any pair of positions, and all positions are computed in parallel. The price is that it costs $O(n^2)$ time and memory in the sequence length, and that it is **blind to order** unless we inject position information separately. Both costs are real and shape most of this week.

| Layer type | Sequential ops | Max path length | Cost per layer |
| --- | --- | --- | --- |
| Recurrent | $O(n)$ | $O(n)$ | $O(n d^2)$ |
| Convolutional (kernel $k$) | $O(1)$ | $O(\log_k n)$ with dilation | $O(k n d^2)$ |
| Self-attention | $O(1)$ | $O(1)$ | $O(n^2 d + n d^2)$ |

:::history Where the pieces came from
Attention in neural translation is due to Bahdanau, Cho and Bengio (2014/2015) and Luong et al. (2015). Self-attention (a sequence attending to itself) appeared in Cheng et al. (2016), Parikh et al. (2016) and Lin et al. (2017), usually inside recurrent models. The 2017 paper combined self-attention with residual connections ([He et al. 2016](week-06.html)), layer normalization (Ba, Kiros and Hinton, 2016), multi-head projections and sinusoidal position signals, and trained it for translation in a few days on eight GPUs, beating the best recurrent and convolutional systems. The architecture was subsequently reused almost unchanged for GPT (2018), BERT (2018) and everything since.
:::

## Self-attention from first principles {#self-attention}

### Attention as a soft dictionary lookup {#soft-dictionary}

Start with something familiar: a Python dictionary. A lookup `d[query]` compares the query against each stored **key**; if one key matches exactly, it returns that key's **value**. This is hard, all-or-nothing retrieval.

Now make it soft. Suppose keys and queries are vectors, and "match" means a high dot product. Instead of returning one value, return a weighted average of *all* values, where the weights are larger for keys that match the query better. Weights must be non-negative and sum to one, which is exactly what the softmax produces:

$$
\text{attend}(\boldsymbol{q}; \{(\boldsymbol{k}_j, \boldsymbol{v}_j)\}_{j=1}^n) \;=\; \sum_{j=1}^{n} \alpha_j \boldsymbol{v}_j, \qquad \alpha_j = \frac{\exp(\boldsymbol{q}^\top \boldsymbol{k}_j)}{\sum_{l}\exp(\boldsymbol{q}^\top \boldsymbol{k}_l)}.
$$

That is the whole mechanism. When the query matches one key much better than the others, $\alpha$ is nearly one-hot and the result is nearly a hard lookup. When the query matches many keys similarly, the result is an average. The three roles are worth naming precisely:

- the **query** says "what am I looking for?"
- the **key** says "what do I contain, as far as matching is concerned?"
- the **value** says "what will I contribute if I am selected?"

### Where Q, K and V come from {#qkv}

In **self**-attention, queries, keys and values are all computed from the same sequence of token representations. Let $\mathbf{X}\in\R^{n\times d}$ be the matrix whose row $i$ is the vector $\boldsymbol{x}_i$ of token $i$ ($n$ tokens, model width $d = d_\text{model}$). Three learned linear maps produce the three roles:

$$
\mathbf{Q} = \mathbf{X}\mathbf{W}_Q,\qquad \mathbf{K} = \mathbf{X}\mathbf{W}_K,\qquad \mathbf{V} = \mathbf{X}\mathbf{W}_V,
$$

with $\mathbf{W}_Q,\mathbf{W}_K\in\R^{d\times d_k}$ and $\mathbf{W}_V\in\R^{d\times d_v}$. Why three different projections of the same input rather than using $\boldsymbol{x}_i$ directly? Because the three jobs differ. In the sentence "the animal didn't cross the street because it was too tired", the word "it" should *look for* a noun-like thing (a query feature), the word "animal" should *advertise* that it is a noun and an animate subject (a key feature), and what "animal" *passes along* once selected (the value) may be a different set of features again. Separate projections let the model decouple matching from content. In an RNN the single hidden state had to serve all roles at once.

{{figure: week09-attention-qkv | Self-attention on four tokens. Each input row is projected to a query, key and value. Row $i$ of the score matrix $\mathbf{S}=\mathbf{Q}\mathbf{K}^\top/\sqrt{d_k}$ holds the similarity of query $i$ to every key; a row-wise softmax turns it into weights, and the output row is the weighted average of the value rows.}}

### Scaled dot-product attention {#scaled-dot-product}

Stacking all $n$ queries gives the matrix form used everywhere:

$$
\text{Attention}(\mathbf{Q},\mathbf{K},\mathbf{V}) \;=\; \softmax\!\Big(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\Big)\mathbf{V}.
$$

The shapes are: $\mathbf{Q}\mathbf{K}^\top\in\R^{n\times n}$ (entry $(i,j)$ is the score between query $i$ and key $j$), the softmax is applied **along each row** (over keys), and multiplying by $\mathbf{V}\in\R^{n\times d_v}$ yields $\R^{n\times d_v}$: one output vector per token. Write the output for token $i$ explicitly:

$$
\boldsymbol{o}_i = \sum_{j=1}^n \alpha_{ij}\,\boldsymbol{v}_j,\qquad \alpha_{ij} = \frac{\exp\big(\boldsymbol{q}_i^\top\boldsymbol{k}_j/\sqrt{d_k}\big)}{\sum_{l=1}^n \exp\big(\boldsymbol{q}_i^\top\boldsymbol{k}_l/\sqrt{d_k}\big)}.
$$

Two properties are worth internalizing right away.

1. **Each output is a convex combination of value vectors.** Attention can only *route and average* existing information; non-linear processing happens elsewhere (the feed-forward layer).
2. **Self-attention is permutation-equivariant.** If you shuffle the rows of $\mathbf{X}$, the rows of the output are shuffled the same way and are otherwise unchanged. The layer has no idea which token came first. This is why position must be injected explicitly (see [Positional information](#positions)).

:::math Proof of permutation equivariance
Let $\mathbf{P}$ be a permutation matrix. Then $\mathbf{Q}' = \mathbf{P}\mathbf{X}\mathbf{W}_Q = \mathbf{P}\mathbf{Q}$, and likewise $\mathbf{K}'=\mathbf{P}\mathbf{K}$, $\mathbf{V}'=\mathbf{P}\mathbf{V}$. The score matrix becomes $\mathbf{P}\mathbf{Q}\mathbf{K}^\top\mathbf{P}^\top$, and a row-wise softmax commutes with simultaneous row and column permutation, so the weights are $\mathbf{P}\mathbf{A}\mathbf{P}^\top$. The output is $\mathbf{P}\mathbf{A}\mathbf{P}^\top\mathbf{P}\mathbf{V} = \mathbf{P}\mathbf{A}\mathbf{V}$, using $\mathbf{P}^\top\mathbf{P}=\mathbf{I}$. So permuting the input permutes the output. $\blacksquare$
:::

### A single attention head in NumPy {#attention-numpy}

Before PyTorch, the whole mechanism fits in a dozen lines. Read the shapes in the comments.

```python:attention_numpy.py
import numpy as np

def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)      # subtract max for numerical stability
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)

def attention(X, Wq, Wk, Wv, causal=False):
    """X: (n, d). Wq, Wk: (d, d_k). Wv: (d, d_v). Returns (out (n, d_v), A (n, n))."""
    Q, K, V = X @ Wq, X @ Wk, X @ Wv             # (n, d_k), (n, d_k), (n, d_v)
    S = Q @ K.T / np.sqrt(Q.shape[-1])            # (n, n) scaled scores
    if causal:                                    # forbid attending to the future
        S = np.where(np.triu(np.ones_like(S), k=1) == 1, -np.inf, S)
    A = softmax(S, axis=-1)                       # (n, n); every row sums to 1
    return A @ V, A                               # (n, d_v)

rng = np.random.default_rng(0)
n, d, dk = 5, 16, 8
X = rng.standard_normal((n, d))
Wq, Wk, Wv = (rng.standard_normal((d, dk)) / np.sqrt(d) for _ in range(3))
out, A = attention(X, Wq, Wk, Wv, causal=True)
print(out.shape, A.shape)                         # (5, 8) (5, 5)
print(np.round(A, 2))                             # lower-triangular rows summing to 1

# permutation equivariance check (non-causal): permuting inputs permutes outputs
perm = rng.permutation(n)
out_a, _ = attention(X, Wq, Wk, Wv)
out_b, _ = attention(X[perm], Wq, Wk, Wv)
print(np.allclose(out_a[perm], out_b))            # True
```

Use the demo below to build intuition. Type your own sentence; the weights are random (untrained), so what you are seeing is the geometry of the mechanism, not linguistic knowledge. Try the temperature slider and the causal mask now, and come back to the scaling toggle after the next section.

{{demo: week09-attention-explorer | Self-attention explorer: typed tokens, temperature, causal mask | 560}}

### Why divide by the square root of the key dimension {#scaling}

The factor $1/\sqrt{d_k}$ looks like a cosmetic constant. It is not. It is what keeps the softmax usable as $d_k$ grows, and the argument is a short variance computation.

Treat a query and a key as random vectors with independent components of mean 0 and variance 1 (this is roughly what a well-initialized network produces, cf. [Week 3](week-03.html#initialization)). The score is $s = \boldsymbol{q}^\top\boldsymbol{k} = \sum_{m=1}^{d_k} q_m k_m$.

:::math Variance of the dot product
Each product $q_mk_m$ has mean $\E[q_m]\E[k_m]=0$ and, by independence, variance $\E[q_m^2]\E[k_m^2] = 1\cdot 1 = 1$. The $d_k$ terms are independent, so variances add:

$$
\E[s] = 0,\qquad \Var(s) = d_k.
$$

The standard deviation is $\sqrt{d_k}$. Dividing by $\sqrt{d_k}$ gives $\Var(s/\sqrt{d_k}) = 1$ for every $d_k$. $\blacksquare$
:::

Why does the *size* of the logits matter? Consider the softmax $\alpha_j\propto e^{s_j}$. If the logits have standard deviation $\sigma$, then the gap between the largest and the typical logit is about $\sigma$ times a number of order one (growing slowly with the number of keys). With $\sigma=1$ the largest weight is moderate. With $\sigma=\sqrt{d_k}=8$ (as in $d_k=64$) the largest logit dominates by factors like $e^{8}\approx 3000$, and the softmax output is nearly one-hot.

That is a problem for **learning**, not only for expressiveness. The softmax Jacobian is

$$
\frac{\partial \alpha_i}{\partial s_j} = \alpha_i(\delta_{ij}-\alpha_j),
$$

and when $\boldsymbol{\alpha}$ is nearly one-hot, every entry is close to zero ($\alpha_i\approx1$ gives $\alpha_i(1-\alpha_i)\approx 0$; $\alpha_i\approx 0$ gives a tiny product). The gradient flowing back to $\mathbf{W}_Q$ and $\mathbf{W}_K$ vanishes, the heads freeze into arbitrary hard patterns early in training, and optimization stalls. You can see both effects in real numbers:

```chart
{"type": "line", "caption": "Variance of the dot product q.k for independent standard-normal q, k as the head width d_k grows (20,000 samples per point, NumPy). Unscaled variance equals d_k; dividing by sqrt(d_k) keeps it at 1.", "height": 320, "data": {"labels": [1, 4, 16, 64, 256, 1024], "datasets": [{"label": "Var(q.k), unscaled", "data": [1.01, 4.0, 16.11, 63.21, 256.6, 1026.4], "borderColor": "var(--c-red)", "tension": 0.2}, {"label": "Var(q.k / sqrt(d_k))", "data": [1.009, 1.0, 1.007, 0.988, 1.002, 1.002], "borderColor": "var(--c-blue)", "tension": 0.2}]}, "options": {"scales": {"x": {"title": {"display": true, "text": "d_k"}}, "y": {"type": "logarithmic", "title": {"display": true, "text": "variance (log scale)"}}}}}
```

```chart
{"type": "line", "caption": "Average largest softmax weight over 16 random keys (2,000 queries per point). Uniform attention would give 0.0625. Without scaling, large d_k drives the softmax to a near one-hot, where its gradient vanishes.", "height": 320, "data": {"labels": [1, 4, 16, 64, 256, 1024], "datasets": [{"label": "unscaled scores", "data": [0.21, 0.425, 0.683, 0.845, 0.921, 0.964], "borderColor": "var(--c-red)", "tension": 0.2}, {"label": "scaled by 1/sqrt(d_k)", "data": [0.21, 0.235, 0.243, 0.243, 0.246, 0.247], "borderColor": "var(--c-blue)", "tension": 0.2}]}, "options": {"scales": {"x": {"title": {"display": true, "text": "d_k"}}, "y": {"min": 0, "max": 1, "title": {"display": true, "text": "mean max attention weight"}}}}}
```

The second chart is the important one. With 16 random keys, a uniform softmax would put 0.0625 on each; unscaled logits at $d_k=1024$ give the largest key an average weight of 0.96, while scaled logits stay at about 0.25 regardless of $d_k$. (Vaswani et al. note the same effect; additive attention was reported to beat unscaled dot-product attention at large $d_k$ for this reason.)

:::warning Scaling is about the *initial* distribution, not a law
Once the model trains, the logits are no longer unit-variance, and that is fine: a sharp, nearly one-hot head is sometimes exactly what is needed. The $1/\sqrt{d_k}$ factor merely guarantees a healthy starting point. Some recent models add extra stabilizers such as **QK-norm** (normalizing queries and keys before the dot product) because logits can still grow uncontrollably at large scale, a failure sometimes called attention-logit growth or entropy collapse.
:::

### Temperature and the geometry of attention {#temperature}

The $1/\sqrt{d_k}$ is a fixed *temperature*. If you write $\alpha = \softmax(\mathbf{s}/\tau)$, then $\tau\to\infty$ gives uniform averaging, $\tau\to0$ gives hard argmax retrieval, and standard attention sits at $\tau=\sqrt{d_k}$ for raw scores. The explorer above exposes $\tau$ so you can see the interpolation: row entropy goes from $\log n$ (uniform) to $0$ (one-hot). Keep this picture; in [Week 10](week-10.html#decoding) the same knob appears again when we sample text.

## Multi-head attention {#mha}

One attention head produces one weighted average per token. But a token typically needs several different kinds of information at once: the previous token, the syntactic head, a coreferent noun, a matching bracket. A single softmax has to compromise. **Multi-head attention** runs $h$ attention operations in parallel in lower-dimensional subspaces and concatenates the results.

### Definition and the shape bookkeeping {#mha-shapes}

Let $d$ be the model width and $h$ the number of heads, with $d_k = d_v = d/h$. For head $r\in\{1,\dots,h\}$:

$$
\text{head}_r = \text{Attention}\big(\mathbf{X}\mathbf{W}_Q^{(r)},\;\mathbf{X}\mathbf{W}_K^{(r)},\;\mathbf{X}\mathbf{W}_V^{(r)}\big),\qquad
\text{MHA}(\mathbf{X}) = \big[\text{head}_1;\dots;\text{head}_h\big]\mathbf{W}_O,
$$

where $\mathbf{W}_Q^{(r)},\mathbf{W}_K^{(r)},\mathbf{W}_V^{(r)}\in\R^{d\times d/h}$ and $\mathbf{W}_O\in\R^{d\times d}$. The cost is the same as one big head: $h$ matrices of shape $d\times (d/h)$ hold the same number of weights as one $d\times d$ matrix. In practice one fuses them: compute $\mathbf{X}\mathbf{W}_Q$ with a single $d\times d$ matrix and then **reshape** the result into $h$ heads. Here are the exact tensor shapes with a batch dimension $B$, sequence length $T$:

{{figure: week09-multihead-shapes | Shape trace through multi-head attention for $d=512$, $h=8$, $d_k=64$. Heads are created by a view and transpose, not by separate matrices; the attention matrices are $T\times T$ per head and per batch element; the heads are merged by the inverse transpose and reshape and mixed by $\mathbf{W}_O$.}}

| Step | Operation | Shape |
| --- | --- | --- |
| input | $\mathbf{X}$ | `(B, T, d)` |
| project | `X @ W_Q` (also K, V) | `(B, T, d)` |
| split heads | `view(B, T, h, d_k)` then `transpose(1, 2)` | `(B, h, T, d_k)` |
| scores | `Q @ K.transpose(-2, -1) / sqrt(d_k)` | `(B, h, T, T)` |
| weights | `softmax(dim=-1)` | `(B, h, T, T)` |
| weighted values | `A @ V` | `(B, h, T, d_k)` |
| merge heads | `transpose(1, 2)` then `reshape(B, T, d)` | `(B, T, d)` |
| output projection | `@ W_O` | `(B, T, d)` |

The output has the same shape as the input, which is what lets us stack layers and add residual connections.

:::intuition Why subspaces rather than $h$ full-width heads?
If every head used the full width $d$, cost would grow $h$-fold. Splitting $d$ into $h$ slices of width $d/h$ keeps the budget constant while giving each head its own softmax, hence its own attention pattern. The trade-off is that each head's queries and keys live in a smaller space, which limits how many independent "match features" it can express. Empirically $d_k$ between 64 and 128 is the sweet spot, and most models scale heads with width ($d=768, h=12$; $d=12288, h=96$ gives $d_k=128$).
:::

### What do heads learn? {#head-roles}

Studies of trained models (Voita et al., 2019, on translation; Clark et al., 2019, on BERT; Olsson et al., 2022, "induction heads" in GPT-style models) find that heads specialize: some attend to the previous token, some to rare or syntactically salient tokens, some implement a copy-and-continue pattern ("... A B ... A" predicts "B") that appears to underlie in-context learning. Many heads can also be pruned with little loss, which tells you that head count is partly redundancy. Treat attention weights as a useful *diagnostic*, not a faithful explanation of the model's reasoning: a high weight does not guarantee a high influence on the output, because the value vector's norm and what later layers do with it also matter.

### Multi-head attention in PyTorch {#mha-torch}

This is a complete, mask-aware implementation. It uses `torch.nn.functional.scaled_dot_product_attention` when you want speed (it dispatches to fused FlashAttention-style kernels on supported GPUs), and an explicit path so you can see the math and inspect the weights.

```python:mha.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0, bias: bool = True):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.h, self.dk = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=bias)    # fused W_Q, W_K, W_V
        self.out = nn.Linear(d_model, d_model, bias=bias)         # W_O
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask=None, return_weights=False):
        """x: (B, T, d). mask: bool broadcastable to (B, h, T, T), True = may attend."""
        B, T, d = x.shape
        q, k, v = self.qkv(x).split(d, dim=-1)                    # each (B, T, d)
        # (B, T, d) -> (B, T, h, dk) -> (B, h, T, dk)
        q, k, v = (t.view(B, T, self.h, self.dk).transpose(1, 2) for t in (q, k, v))
        if return_weights:                                        # explicit path
            s = q @ k.transpose(-2, -1) / math.sqrt(self.dk)      # (B, h, T, T)
            if mask is not None:
                s = s.masked_fill(~mask, float("-inf"))
            a = self.drop(torch.softmax(s, dim=-1))
            y = a @ v                                             # (B, h, T, dk)
        else:                                                     # fused kernel path
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                               dropout_p=self.drop.p if self.training else 0.0)
            a = None
        y = y.transpose(1, 2).reshape(B, T, d)                    # merge heads -> (B, T, d)
        y = self.out(y)
        return (y, a) if return_weights else y

if __name__ == "__main__":
    B, T, d, h = 2, 6, 64, 8
    x = torch.randn(B, T, d)
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool))       # (T, T), True on/below diagonal
    mha = MultiHeadAttention(d, h)
    y, a = mha(x, mask=causal, return_weights=True)
    print(y.shape, a.shape)                                       # (2, 6, 64) (2, 8, 6, 6)
    print(a[0, 0].sum(-1))                                        # each row sums to 1
    print((a[0, 0].triu(1) == 0).all())                           # no weight on the future
    y2 = mha(x, mask=causal)                                      # fused path agrees
    print(torch.allclose(y, y2, atol=1e-5))
```

Notice that `scaled_dot_product_attention` interprets a **boolean** mask with `True` meaning "take part in attention", whereas a float mask is *added* to the scores. Mixing the two conventions is a classic bug (see [Common mistakes](#pitfalls)).

### Cross-attention {#cross-attention}

In the original encoder-decoder Transformer a third attention type appears: **cross-attention**. Its queries come from the decoder's states and its keys and values from the encoder's output ("memory"). Everything else is identical, except that the score matrix is $T_\text{dec}\times T_\text{enc}$ rather than square. This is the direct descendant of Bahdanau attention from Week 7: the decoder looks at the source sentence while generating each target word.

## The position-wise feed-forward network {#ffn}

After attention mixes information across positions, each token passes **independently** through the same two-layer MLP:

$$
\text{FFN}(\boldsymbol{x}) = \mathbf{W}_2\,\phi(\mathbf{W}_1\boldsymbol{x} + \boldsymbol{b}_1) + \boldsymbol{b}_2,\qquad \mathbf{W}_1\in\R^{d_\text{ff}\times d},\ \mathbf{W}_2\in\R^{d\times d_\text{ff}},
$$

with $d_\text{ff}=4d$ by convention (2048 for $d=512$ in the original paper) and $\phi$ originally ReLU, later GELU ([Hendrycks and Gimpel, 2016](week-02.html)) in BERT and GPT. "Position-wise" means the same weights are applied to every position; it is equivalent to two $1\times1$ convolutions.

### Why do we need it? {#why-ffn}

Without the FFN a stack of attention layers is surprisingly weak. Each attention output is a convex combination of linearly projected inputs; stacking just re-mixes linear functions of the token embeddings through input-dependent weights. The FFN supplies the **element-wise non-linearity** and the bulk of the capacity: it holds about two thirds of the parameters of a block ($8d^2$ versus $4d^2$ for attention, derived [below](#counting)).

A useful reading, from Geva, Schuster, Berant and Levy (2021), treats the FFN as a **key-value memory**: the rows of $\mathbf{W}_1$ act as keys that detect patterns in the residual stream, and the columns of $\mathbf{W}_2$ are the values written back when the corresponding hidden unit fires. Attention then moves information *between* tokens, and the FFN looks facts up *within* a token. This division (communication versus computation) is a good heuristic for where to look when you study what a model "knows".

### Gated variants: GLU, SwiGLU {#swiglu}

Shazeer (2020, "GLU Variants Improve Transformer") showed that replacing the first linear layer and activation by a **gated** product gives consistently lower loss at the same compute:

$$
\text{FFN}_\text{SwiGLU}(\boldsymbol{x}) = \mathbf{W}_2\big(\text{SiLU}(\mathbf{W}_g\boldsymbol{x})\odot(\mathbf{W}_u\boldsymbol{x})\big),
$$

where $\text{SiLU}(z)=z\,\sigma(z)$ and $\odot$ is element-wise multiplication. There are now three matrices instead of two, so to keep the parameter count equal models shrink the hidden width to roughly $\tfrac{8}{3}d$ (rounded to a hardware-friendly multiple). LLaMA-family and many other open models use this form. The gate lets each hidden unit decide how much of its input to pass, a mild form of multiplicative interaction.

```python:ffn.py
import torch, torch.nn as nn, torch.nn.functional as F

class FFN(nn.Module):                       # GPT-2 style
    def __init__(self, d, mult=4, dropout=0.0):
        super().__init__()
        self.w1, self.w2 = nn.Linear(d, mult * d), nn.Linear(mult * d, d)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):                   # (B, T, d) -> (B, T, d)
        return self.drop(self.w2(F.gelu(self.w1(x))))

class SwiGLU(nn.Module):                    # LLaMA style, no biases
    def __init__(self, d, hidden=None):
        super().__init__()
        hidden = hidden or int(8 * d / 3)
        self.gate, self.up, self.down = (nn.Linear(d, hidden, bias=False),
                                         nn.Linear(d, hidden, bias=False),
                                         nn.Linear(hidden, d, bias=False))
    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))

d = 512
n1 = sum(p.numel() for p in FFN(d).parameters())
n2 = sum(p.numel() for p in SwiGLU(d).parameters())
print(n1, n2)    # 2099712 vs 2096640: hidden = int(8d/3) = 1365 matches the 4x GELU FFN to within 0.2%
```

:::note Mixture of experts in one paragraph
Because the FFN holds most parameters and processes tokens independently, it is the natural place to add capacity cheaply. A **mixture-of-experts** (MoE) layer replaces one FFN by many (say 8 to 64), plus a small router that sends each token to its top-1 or top-2 experts (Shazeer et al., 2017; Fedus, Zoph and Shazeer, 2021, "Switch Transformer"). Parameters grow by the number of experts while per-token FLOPs stay about constant. The cost is memory, load-balancing losses and communication.
:::

## Residual connections and normalization {#norms}

### The residual stream {#residual-stream}

Each sublayer (attention or FFN) is wrapped in a residual connection: the output is $\boldsymbol{x} + \text{Sublayer}(\boldsymbol{x})$, as in [ResNets (Week 6)](week-06.html). In a Transformer this has a particularly clean interpretation due to Elhage et al. (2021): the vector at each position is a **residual stream**, a running sum to which every attention head and every FFN adds its contribution. Layer $\ell$ reads from the stream (through a projection), computes something, and writes the result back by addition. Nothing is ever overwritten; information from the embedding can flow unchanged to the last layer.

That has an optimization benefit identical to ResNets (the gradient of the loss with respect to an early layer contains a direct identity path), and a conceptual one: a 96-layer model is not a 96-step bottleneck but a shared workspace that layers incrementally edit.

### Layer normalization {#layernorm}

Batch normalization ([Week 3](week-03.html#normalization)) is awkward for sequences: batch statistics are noisy with variable lengths and padding, and in autoregressive decoding the batch may be a single sequence. **Layer normalization** (Ba, Kiros and Hinton, 2016) normalizes each token vector over its feature dimension, independently of the batch and of the other positions:

$$
\text{LN}(\boldsymbol{x}) = \boldsymbol{\gamma}\odot\frac{\boldsymbol{x}-\mu}{\sqrt{\sigma^2+\epsilon}} + \boldsymbol{\beta},\qquad \mu=\frac1d\sum_{m}x_m,\quad \sigma^2 = \frac1d\sum_m (x_m-\mu)^2,
$$

with learned gain $\boldsymbol{\gamma}$ and bias $\boldsymbol{\beta}$ in $\R^d$ and $\epsilon\approx10^{-5}$ for numerical safety. In PyTorch, `nn.LayerNorm(d)` normalizes over the last dimension: the statistics are computed *per token*, so `(B, T, d)` produces $B\cdot T$ separate means and variances.

### RMSNorm {#rmsnorm}

Zhang and Sennrich (2019) observed that re-centering (subtracting $\mu$) contributes little and that the benefit of LayerNorm comes mostly from re-scaling. **RMSNorm** drops the mean and the bias:

$$
\text{RMSNorm}(\boldsymbol{x}) = \boldsymbol{\gamma}\odot\frac{\boldsymbol{x}}{\sqrt{\tfrac1d\sum_m x_m^2+\epsilon}}.
$$

It is cheaper (one reduction instead of two, no bias) and works as well in practice; LLaMA, Mistral, Gemma and many others use it. Both normalizations are scale-invariant in their input: multiplying $\boldsymbol{x}$ by a positive constant leaves the output unchanged, which is a useful fact when thinking about what the residual stream's *norm* means.

```python:norms.py
import torch, torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))       # gamma
    def forward(self, x):                                # x: (..., d)
        # compute in float32 for stability even if x is bf16/fp16, then cast back
        xf = x.float()
        out = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (out.type_as(x)) * self.weight

x = torch.randn(4, 10, 64) * 5 + 3
ln, rms = nn.LayerNorm(64), RMSNorm(64)
print(ln(x).mean(-1).abs().max().item())                 # ~0: LN removes the per-token mean
print(rms(x).mean(-1).abs().max().item())                # not 0: RMSNorm keeps the mean
print(rms(x).pow(2).mean(-1).sqrt().mean().item())       # ~1: unit RMS per token
```

### Pre-norm versus post-norm {#pre-post-norm}

Where does the normalization go? The original Transformer used **post-norm**: apply the sublayer, add the residual, *then* normalize, $\boldsymbol{x}\leftarrow\text{LN}(\boldsymbol{x}+\text{Sublayer}(\boldsymbol{x}))$. GPT-2 and nearly every large model since use **pre-norm**: normalize first, $\boldsymbol{x}\leftarrow\boldsymbol{x}+\text{Sublayer}(\text{LN}(\boldsymbol{x}))$, with one extra final LayerNorm before the output head.

{{figure: week09-prenorm-postnorm | Post-norm (left) places LayerNorm on the main path after every addition; pre-norm (right) leaves the residual path untouched and normalizes only the sublayer's input. Pre-norm keeps a clean identity path from input to output.}}

The difference is about gradient flow. In post-norm, every residual addition is followed by a normalization, so the identity path is interrupted $2N$ times in an $N$-layer model, and gradient magnitudes at initialization are largest for the last layers (Xiong et al., 2020 analyze this for post-norm at initialization). The consequence is that post-norm models need a careful **learning-rate warmup** or they diverge; with pre-norm the gradient scale is roughly uniform across depth, training is stable with little or no warmup and larger learning rates, and 100-layer-plus models train reliably.

:::math A one-line view of the residual stream under pre-norm
Unrolling $\boldsymbol{x}_{\ell+1}=\boldsymbol{x}_\ell+F_\ell(\text{LN}(\boldsymbol{x}_\ell))$ gives
$$
\boldsymbol{x}_L = \boldsymbol{x}_0 + \sum_{\ell=0}^{L-1} F_\ell(\text{LN}(\boldsymbol{x}_\ell)),
$$
so $\partial\boldsymbol{x}_L/\partial\boldsymbol{x}_0 = \mathbf{I}+(\text{terms from the }F_\ell)$ always contains an identity term. Under post-norm each step applies a normalization to the whole sum, and no such clean expansion exists.
:::

The cost of pre-norm: the residual stream's norm grows with depth (each layer adds to it), so later layers' contributions are relatively smaller, and some experiments find post-norm reaches slightly better final quality when it can be made to train (DeepNorm, Wang et al. 2022, is a modified post-norm that trains 1000-layer models). The practical rule: **use pre-norm (with RMSNorm) unless you have a reason not to**, and always add the final norm.

## Positional information {#positions}

Since self-attention is permutation-equivariant, a model fed only token embeddings sees a bag of words. We must tell it where each token is. There are two broad strategies:

- **Absolute** schemes add (or concatenate) a vector that depends on the position $p$ to the token embedding, once, at the input.
- **Relative** schemes change the *attention score* so it depends on the offset $n-m$ between query position $m$ and key position $n$, applied in every layer.

Which is better is an empirical question that the field answered gradually in favor of relative schemes, especially RoPE. We cover each, with the derivations that explain their properties.

### Sinusoidal encodings {#sinusoidal}

The original paper used fixed (not learned) vectors $\text{PE}(p)\in\R^d$, with pairs of coordinates oscillating at geometrically spaced frequencies:

$$
\text{PE}(p)_{2i} = \sin(p\,\omega_i),\quad \text{PE}(p)_{2i+1} = \cos(p\,\omega_i),\qquad \omega_i = 10000^{-2i/d},\quad i=0,\dots,d/2-1.
$$

Think of it as a **bank of clocks**: pair 0 ticks fast (wavelength $2\pi\approx6.3$ positions), the last pair ticks very slowly (wavelength about $2\pi\cdot10000$). Reading all the clocks together identifies a position uniquely, like reading seconds, minutes and hours; and like a binary counter the low-order bits flip fast and the high-order bits slowly.

The encoding is added to the token embedding: $\boldsymbol{x}_p = \text{Emb}(w_p)+\text{PE}(p)$. Try the visualizer; the left columns of the matrix oscillate quickly, the right ones slowly.

{{demo: week09-positional-encoding | Position encodings: matrix, offset similarity, RoPE and ALiBi | 560}}

:::math Why sinusoids: shifts are linear maps, and dot products depend on the offset
Fix a frequency $\omega$. Write the pair at position $p$ as $\boldsymbol{u}_p=(\sin p\omega,\cos p\omega)^\top$. The angle-addition formulas give

$$
\begin{pmatrix}\sin (p+k)\omega\\ \cos (p+k)\omega\end{pmatrix} =
\begin{pmatrix}\cos k\omega & \sin k\omega\\ -\sin k\omega & \cos k\omega\end{pmatrix}
\begin{pmatrix}\sin p\omega\\ \cos p\omega\end{pmatrix}.
$$

So $\boldsymbol{u}_{p+k}=\mathbf{M}_k\boldsymbol{u}_p$, where the matrix $\mathbf{M}_k$ (a rotation by $-k\omega$ in this coordinate order) depends on the offset $k$ only, not on $p$. Hence for any fixed $k$, $\text{PE}(p+k)$ is a **linear function** of $\text{PE}(p)$, which a linear layer could in principle exploit to attend by offset. Moreover the dot product of two encodings is

$$
\text{PE}(p)^\top\text{PE}(p+k)=\sum_{i}\big[\sin p\omega_i\sin (p+k)\omega_i+\cos p\omega_i\cos(p+k)\omega_i\big]=\sum_{i=0}^{d/2-1}\cos(k\,\omega_i),
$$

using $\cos(a-b)=\cos a\cos b+\sin a\sin b$. It depends only on the offset $k$, is symmetric in $k$, and decays as $|k|$ grows. $\blacksquare$
:::

```chart
{"type": "line", "caption": "Normalized sinusoidal-PE similarity PE(p).PE(p+k) (d = 128, p = 100) and the RoPE score for q = k = ones (d = 64) as functions of offset k. Both depend only on the offset and decay, with high-frequency ripples.", "height": 320, "data": {"labels": [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100], "datasets": [{"label": "sinusoidal PE similarity", "data": [1.0, 0.897, 0.759, 0.735, 0.714, 0.669, 0.662, 0.649, 0.62, 0.619, 0.608, 0.586, 0.589, 0.579, 0.559, 0.566, 0.557, 0.534, 0.548, 0.543, 0.509, 0.53, 0.542, 0.488, 0.495, 0.546, 0.498, 0.449, 0.509, 0.533, 0.469, 0.444, 0.477, 0.506, 0.5, 0.447, 0.417, 0.465, 0.501, 0.476, 0.452, 0.421, 0.402, 0.458, 0.501, 0.455, 0.43, 0.432, 0.384, 0.389, 0.477], "borderColor": "var(--c-blue)", "pointRadius": 0}, {"label": "RoPE score (q = k = 1)", "data": [1.0, 0.884, 0.748, 0.736, 0.7, 0.658, 0.667, 0.631, 0.605, 0.634, 0.59, 0.549, 0.616, 0.598, 0.494, 0.533, 0.611, 0.561, 0.505, 0.486, 0.481, 0.559, 0.598, 0.497, 0.454, 0.49, 0.44, 0.435, 0.554, 0.576, 0.496, 0.463, 0.435, 0.423, 0.458, 0.414, 0.374, 0.498, 0.582, 0.5, 0.462, 0.465, 0.4, 0.383, 0.423, 0.406, 0.399, 0.393, 0.337, 0.4, 0.559], "borderColor": "var(--c-amber)", "pointRadius": 0}]}, "options": {"scales": {"x": {"title": {"display": true, "text": "offset k"}}, "y": {"title": {"display": true, "text": "normalized score"}}}}}
```

The chart shows this numerically for $d=128$. Note an important limitation: the dot product above is between *inputs*. After the learned projections $\mathbf{W}_Q,\mathbf{W}_K$ and after many layers the clean "function of offset only" property is no longer guaranteed, which is one motivation for relative schemes that build it into the score itself.

```python:sinusoidal.py
import math, torch

def sinusoidal_pe(T: int, d: int, base: float = 10000.0) -> torch.Tensor:
    """Returns (T, d). Even columns sin, odd columns cos (interleaved, as in the paper)."""
    pos = torch.arange(T, dtype=torch.float32)[:, None]                    # (T, 1)
    inv = torch.exp(-math.log(base) * torch.arange(0, d, 2).float() / d)   # (d/2,) = base^(-2i/d)
    pe = torch.zeros(T, d)
    pe[:, 0::2] = torch.sin(pos * inv)
    pe[:, 1::2] = torch.cos(pos * inv)
    return pe

pe = sinusoidal_pe(100, 128)
print(pe.shape)                                    # torch.Size([100, 128])
# dot product depends only on the offset: compare (10, 15) with (60, 65)
print(torch.allclose(pe[10] @ pe[15], pe[60] @ pe[65], atol=1e-3))   # True
```

### Learned absolute embeddings {#learned-pos}

BERT and GPT-2 instead learn a table `nn.Embedding(max_len, d)` and add the row for position $p$. It is the simplest possible scheme and works as well as sinusoids within the training length (the original paper reports near-identical translation quality). Its limits: positions beyond `max_len` do not exist, and the table cannot generalize to unseen lengths (a position never seen in training has an untrained embedding). It also spends $T_\text{max}\cdot d$ parameters (0.8M for GPT-2 small, small next to 124M).

### Relative position representations {#relative}

Intuition says language is mostly about *relative* position: the previous word, the word three back. **Shaw, Uszkoreit and Vaswani (2018)** modified attention so the score and the value aggregation both receive a learned vector $\boldsymbol{a}^K_{n-m},\boldsymbol{a}^V_{n-m}$ indexed by the clipped offset. **Transformer-XL** (Dai et al., 2019) reparameterized the score into content-content, content-position, global-content-bias and global-position-bias terms so that attention could span recurrently cached segments. **T5** (Raffel et al., 2020) used the simplest form: a learned scalar *bias* per head added to the logit, indexed by a bucketed offset $b(n-m)$ (exact buckets for small offsets, logarithmically larger buckets for distant ones):

$$
s_{mn} = \frac{\boldsymbol{q}_m^\top\boldsymbol{k}_n}{\sqrt{d_k}} + b^{(\text{head})}_{\,b(n-m)}.
$$

This is cheap (a small table per head) and extends to longer sequences by sharing the last bucket. The downside is that the bias must be materialized as an $n\times n$ addition to the scores, which complicates fused attention kernels.

### Rotary position embeddings (RoPE) {#rope}

**RoFormer** (Su, Lu, Pan, Murtadha, Wen and Liu, 2021) introduced rotary embeddings, which are now the default in LLaMA, Mistral, Qwen, Gemma and many other open models. The idea: *rotate* the query and key vectors by an angle proportional to their position, so that the dot product between a rotated query and a rotated key depends only on the angle *difference*, hence on the relative offset.

{{figure: week09-rope-rotation | RoPE in one two-dimensional pair of coordinates. The same vector at positions $m$, $m+1$, $m+2$ is rotated by successive multiples of $\theta_i$. The dot product of two rotated vectors depends on the angle between them, and that angle is the original angle plus $(n-m)\theta_i$.}}

:::math Deriving RoPE
We want functions $f_q(\boldsymbol{x},m)$ and $f_k(\boldsymbol{x},n)$ that inject positions $m,n$ such that the score depends only on $\boldsymbol{q},\boldsymbol{k}$ and $n-m$:
$$
\langle f_q(\boldsymbol{q},m),\,f_k(\boldsymbol{k},n)\rangle = g(\boldsymbol{q},\boldsymbol{k},n-m).
$$
Work in two dimensions first. Let $\mathbf{R}(\phi)=\begin{pmatrix}\cos\phi&-\sin\phi\\ \sin\phi&\cos\phi\end{pmatrix}$ be a rotation. Rotations are orthogonal, $\mathbf{R}(\phi)^\top=\mathbf{R}(-\phi)$, and compose by adding angles, $\mathbf{R}(a)\mathbf{R}(b)=\mathbf{R}(a+b)$. Choose
$$
f_q(\boldsymbol{q},m)=\mathbf{R}(m\theta)\boldsymbol{q},\qquad f_k(\boldsymbol{k},n)=\mathbf{R}(n\theta)\boldsymbol{k}.
$$
Then
$$
\big(\mathbf{R}(m\theta)\boldsymbol{q}\big)^\top\big(\mathbf{R}(n\theta)\boldsymbol{k}\big)=\boldsymbol{q}^\top\mathbf{R}(-m\theta)\mathbf{R}(n\theta)\boldsymbol{k}=\boldsymbol{q}^\top\mathbf{R}\big((n-m)\theta\big)\boldsymbol{k}.
$$
The absolute positions have cancelled; only $n-m$ survives. For a $d_k$-dimensional head (even $d_k$), split coordinates into $d_k/2$ pairs and rotate pair $i$ by its own frequency $\theta_i=10000^{-2i/d_k}$, i.e. use the block-diagonal rotation $\mathbf{R}_m=\text{diag}\big(\mathbf{R}(m\theta_0),\dots,\mathbf{R}(m\theta_{d_k/2-1})\big)$. Then $\langle\mathbf{R}_m\boldsymbol{q},\mathbf{R}_n\boldsymbol{k}\rangle=\boldsymbol{q}^\top\mathbf{R}_{n-m}\boldsymbol{k}$. $\blacksquare$

In complex notation, treat pair $i$ as $z=x_{2i}+\mathrm{i}\,x_{2i+1}$; rotation is multiplication by $e^{\mathrm{i}m\theta_i}$, and the pair's contribution to the score is $\mathrm{Re}\big[q_i\bar k_i\,e^{\mathrm{i}(m-n)\theta_i}\big]$.
:::

Properties that make RoPE attractive:

- **No extra parameters and no change to the residual stream**: it is applied to $\mathbf{Q}$ and $\mathbf{K}$ inside each attention layer, not added to the embeddings. Values are *not* rotated.
- **Relative by construction**, yet computed as a cheap element-wise operation (pairs of multiplications by precomputed cosines and sines), compatible with fused attention kernels and with KV caching (cached keys already carry their own position).
- **Decay with distance** of the score contributions of high-frequency pairs (the chart above), a useful inductive bias for locality.
- **Length extrapolation is imperfect**. Beyond the training length the rotation angles take values never seen, and quality degrades. The common remedies rescale positions or frequencies: **position interpolation** (Chen et al., 2023) compresses positions $m\to m\cdot L_\text{train}/L_\text{new}$ and a short fine-tune restores quality; NTK-aware scaling and YaRN (Peng et al., 2023) rescale frequencies non-uniformly.

```python:rope.py
import torch

def rope_cache(T, dk, base=10000.0, device=None):
    """cos, sin tables of shape (T, dk/2)."""
    inv = 1.0 / (base ** (torch.arange(0, dk, 2, device=device).float() / dk))   # (dk/2,) theta_i
    ang = torch.outer(torch.arange(T, device=device).float(), inv)               # (T, dk/2) m * theta_i
    return ang.cos(), ang.sin()

def apply_rope(x, cos, sin):
    """x: (B, h, T, dk). Rotates each consecutive pair (x[2i], x[2i+1]) by m * theta_i."""
    x1, x2 = x[..., 0::2], x[..., 1::2]                  # (B, h, T, dk/2) each
    c, s = cos[: x.shape[-2]], sin[: x.shape[-2]]        # (T, dk/2), broadcast over B, h
    r1 = x1 * c - x2 * s
    r2 = x1 * s + x2 * c
    return torch.stack((r1, r2), dim=-1).flatten(-2)     # interleave back -> (B, h, T, dk)

# test: scores depend only on the relative offset
torch.manual_seed(0)
dk, T = 64, 32
cos, sin = rope_cache(T, dk)
q0, k0 = torch.randn(dk), torch.randn(dk)
Q = q0.expand(1, 1, T, dk).clone()                       # same raw q at every position
K = k0.expand(1, 1, T, dk).clone()
Qr, Kr = apply_rope(Q, cos, sin), apply_rope(K, cos, sin)
s = Qr[0, 0] @ Kr[0, 0].T                                # (T, T) score between positions
print(torch.allclose(s[5, 3], s[20, 18], atol=1e-4))     # same offset (2): True
print(torch.allclose(s[5, 3], s[5, 2], atol=1e-4))       # different offset: False
```

### ALiBi {#alibi}

**ALiBi** (Press, Smith and Lewis, "Train Short, Test Long", 2021/2022) removes positional vectors entirely. It subtracts a penalty proportional to distance from each logit:

$$
s_{mn} = \frac{\boldsymbol{q}_m^\top\boldsymbol{k}_n}{\sqrt{d_k}} - \lambda_r\,(m-n),\qquad n\le m,\qquad \lambda_r = 2^{-8r/H}\ (r=1,\dots,H).
$$

Head $r$ has a fixed slope $\lambda_r$ (none are learned): steep slopes make a head local, shallow slopes let it see far. The authors showed that models trained on 1024 tokens could extrapolate to roughly twice that length at inference with unchanged perplexity, outperforming sinusoidal and learned encodings in that test. Empirically RoPE with scaling has been more widely adopted, but ALiBi (used in BLOOM and MPT) is the simplest scheme to implement: add a precomputed bias matrix to the scores.

### Choosing a scheme {#choosing-position}

| Scheme | Where applied | Params | Relative? | Extrapolates? | Used by |
| --- | --- | --- | --- | --- | --- |
| Sinusoidal | embeddings (input) | 0 | approximately | weak in practice | original Transformer |
| Learned absolute | embeddings (input) | $T_{\max} d$ | no | no | BERT, GPT-2 |
| Relative bias | attention scores | small | yes | moderate | T5, Transformer-XL |
| RoPE | $\mathbf{Q},\mathbf{K}$ each layer | 0 | yes | with scaling | LLaMA-family, Qwen, Gemma |
| ALiBi | attention scores | 0 | yes | good | BLOOM, MPT |

A modern default is RoPE. Note what is *not* here: the **causal mask** itself leaks position information (token $i$ sees exactly $i$ predecessors), and some decoder-only models with no explicit position encoding at all ("NoPE") still learn order from it, a fact worth remembering when you debug.

## Encoder-only, decoder-only and encoder-decoder models {#architectures}

The same block (attention, FFN, residuals, norms) can be assembled in three ways, distinguished by **which attention masks** they use and what they are trained to do.

{{figure: week09-masks-architectures | Attention masks of the three families. Rows are queries, columns are keys. Colored cells are allowed; gray cells are masked. The encoder-decoder uses all three kinds: bidirectional self-attention in the encoder, causal self-attention in the decoder, and unmasked cross-attention.}}

**Encoder-only (BERT, RoBERTa, DeBERTa).** Bidirectional self-attention: every token sees the whole input. Output: a contextual vector for each token. Ideal for understanding tasks where the full input is available (classification, tagging, retrieval embeddings). It cannot generate text left to right because it has no autoregressive factorization. Trained with masked language modeling ([Week 10](week-10.html#pretraining)).

**Decoder-only (GPT family, LLaMA, most chat models).** Causal self-attention: token $i$ sees tokens $\le i$. Trained to predict the next token, which gives a proper joint distribution $p(x_1,\dots,x_T)=\prod_t p(x_t\mid x_{<t})$. Generates by sampling one token at a time, feeding it back in. It is the simplest design, uses every position as a training signal, and scales well, which is why it dominates large language models.

**Encoder-decoder (original Transformer, T5, BART, many speech and translation systems).** The encoder reads the input bidirectionally; the decoder generates the output causally while cross-attending to the encoder output. Natural for sequence-to-sequence problems with a clear input/output split (translation, summarization). It spends parameters on two stacks and cross-attention.

Between them lies the **prefix-LM**: a single stack where the first part (the prompt or prefix) is attended bidirectionally and the rest causally (rightmost panel of the figure above). It gives an encoder's richer representation of the prompt at the cost of one shared stack.

{{figure: week09-block-stack | Left: the decoder-only stack as used by GPT-style models (embedding plus position, $N$ blocks of pre-norm attention and FFN, final norm, LM head). Right: the original encoder-decoder Transformer, with the encoder output feeding the decoder's cross-attention as keys and values.}}

### Causal masking in detail {#causal-masking}

For autoregressive training we want to compute the loss at all $T$ positions in one parallel pass, so position $i$ must not see tokens $j>i$. The trick: before the softmax, set the forbidden score entries to $-\infty$, so they receive weight exactly zero:

$$
\tilde s_{ij}=\begin{cases}s_{ij}&j\le i\\-\infty&j>i\end{cases}\qquad\Rightarrow\qquad \alpha_{ij}=0\ \text{for }j>i.
$$

Every row still has at least one finite entry (the diagonal), so no row is all $-\infty$ (which would produce NaNs). With a **padding mask** the situation is different: if a sequence is padded to the batch length, padded key positions must be masked for every query (a mask over the key axis, shape `(B, 1, 1, T)`), and padded *query* rows can be ignored in the loss. Combine masks with logical AND on booleans.

The causal mask is what makes **teacher forcing** efficient: given the whole target sequence, one forward pass produces the next-token predictions for every prefix at once. It is also why generation is different at inference time: new tokens arrive one at a time, and recomputing all earlier keys and values at each step would waste work, hence the **KV cache** (next section).

```python:masks.py
import torch

T, B = 5, 2
causal = torch.tril(torch.ones(T, T, dtype=torch.bool))              # (T, T); True = allowed
lengths = torch.tensor([5, 3])                                       # true lengths in the batch
keep = torch.arange(T)[None, :] < lengths[:, None]                   # (B, T); True = real token
pad_mask = keep[:, None, None, :]                                    # (B, 1, 1, T) over keys
mask = causal[None, None] & pad_mask                                 # (B, 1, T, T)
print(mask[1, 0].int())
# row i: ones up to min(i, 2) (columns 3, 4 are padding for the second sequence)
# every row still has a True on or left of the diagonal for real tokens
scores = torch.randn(B, 1, T, T)
weights = scores.masked_fill(~mask, float("-inf")).softmax(-1)       # padded query rows 3, 4:
print(torch.isnan(weights).any().item())                             # False here: causal keeps col 0
```

## Computational cost and efficient attention {#complexity}

### Counting the cost of a layer {#cost-derivation}

Count multiply-adds as 2 FLOPs. For sequence length $n$ and width $d$ (one sequence, ignoring biases and normalization):

:::math FLOPs of one Transformer layer
- Q, K, V projections: three matrix products $(n\times d)(d\times d)$: $3\cdot 2nd^2=6nd^2$.
- Scores $\mathbf{Q}\mathbf{K}^\top$ across all heads: $h$ products $(n\times d_k)(d_k\times n)$: $2n^2 d_k h = 2n^2d$.
- Weighted values $\mathbf{A}\mathbf{V}$: another $2n^2d$.
- Output projection: $2nd^2$.
- FFN with $d_\text{ff}=4d$: two products, $2\cdot 2n\cdot d\cdot4d=16nd^2$.

Total: $\;\underbrace{24\,n d^2}_{\text{linear in }n}+\underbrace{4\,n^2 d}_{\text{quadratic in }n}.$ The quadratic term equals the linear one when $4n^2d=24nd^2$, i.e. at $n=6d$. For $d=768$ that is $n=4608$; for $d=12288$ (GPT-3 scale) it is $n\approx 74{,}000$.
:::

So at the context lengths of the original paper ($n\approx 50$ to $500$, $d=512$) attention is a *minor* part of the FLOPs, and the FFN and projections dominate. The quadratic term only becomes the bottleneck at long contexts:

```chart
{"type": "line", "caption": "Share of a layer's forward FLOPs spent in the two n x n matrix products (QK^T and AV), for d_model = 768 and a 4x FFN. The crossover with the linear-in-n projection and FFN FLOPs is at n = 6d = 4608 (exact formulas derived in the text).", "height": 320, "data": {"labels": [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768], "datasets": [{"label": "attention-matrix share", "data": [0.027, 0.053, 0.1, 0.182, 0.308, 0.471, 0.64, 0.78, 0.877], "borderColor": "var(--c-accent)", "tension": 0.2}]}, "options": {"scales": {"x": {"title": {"display": true, "text": "sequence length n"}}, "y": {"min": 0, "max": 1, "title": {"display": true, "text": "fraction of layer FLOPs"}}}}}
```

But FLOPs understate the problem. Two further costs grow as $n^2$:

- **Memory for the score matrix.** Materializing $\mathbf{S}\in\R^{h\times n\times n}$ for $n=8192$, $h=32$ in 16-bit precision takes $32\cdot8192^2\cdot2\approx4.3$ GB *per layer per batch element* (plus the same again for the softmax output, plus saved activations for backward).
- **Memory traffic.** On a GPU, attention at moderate $d_k$ is limited by reading and writing these $n\times n$ tensors to slow high-bandwidth memory, not by arithmetic.

**FlashAttention** (Dao, Fu, Ermon, Rudra and Ré, 2022) attacks the second cost without approximation. It computes attention in tiles that fit in fast on-chip SRAM, maintaining a running maximum and running denominator for each row (the "online softmax" trick), so the $n\times n$ matrix is never written to main memory. The result is the **exact same output**, with memory linear in $n$ and a substantial wall-clock speedup. FLOPs are still $O(n^2d)$. In PyTorch you obtain it through `F.scaled_dot_product_attention`. This is the pragmatic first answer to "attention is quadratic": use an exact IO-aware kernel before reaching for approximations.

### The KV cache and the cost of generation {#kv-cache}

At inference a decoder generates token $t$ given $t-1$ previous tokens. The keys and values of earlier tokens do not change (the mask forbids them from seeing the future), so we **cache** them. Each new step computes $\boldsymbol{q}_t,\boldsymbol{k}_t,\boldsymbol{v}_t$ for the new token only, appends $\boldsymbol{k}_t,\boldsymbol{v}_t$ to the cache, and attends over all cached entries: $O(td)$ work per step rather than $O(t^2d)$. The cost is memory: $2\cdot L\cdot n\cdot d$ numbers. For $L=32$, $d=4096$, $n=4096$ in 16-bit precision that is $2\cdot32\cdot4096\cdot4096\cdot 2$ bytes $\approx 2.1$ GB **per sequence**, so the KV cache, not the weights, often limits serving throughput. **Multi-query attention** (Shazeer, 2019) shares one K/V head across all query heads, and **grouped-query attention** (Ainslie et al., 2023) shares one K/V head per group of query heads, shrinking the cache by the group factor at small quality cost; both are standard in recent models.

### Sparse and sliding-window attention {#sparse}

If each token attends to only $O(1)$ or $O(\sqrt n)$ others, cost drops from $n^2$ to $n$ or $n\sqrt n$. The question is which entries to keep.

{{figure: week09-attention-patterns | Six attention patterns on $n=12$ tokens. Dense and causal attention compute $\Theta(n^2)$ scores. Sliding-window, dilated and window-plus-global patterns compute $O(nw)$ (or $O(n\sqrt n)$ for strided sparse attention). Linear attention avoids the $n\times n$ matrix altogether by reordering the matrix products.}}

- **Sliding-window (local) attention**: token $i$ attends to the $w$ nearest tokens. Cost $O(nw)$. Locality matches most of language, and **stacking** layers grows the effective receptive field: after $L$ layers with a one-sided window $w$, information can travel $L(w-1)$ positions. Longformer (Beltagy, Peters and Cohan, 2020) and Mistral 7B (Jiang et al., 2023) use sliding windows.
- **Dilated / strided patterns**: attend to every $s$-th token plus a local neighborhood, so a few layers cover long distances. **Sparse Transformers** (Child, Gray, Radford and Sutskever, 2019) factorize attention into a local head and a strided head with $s\approx\sqrt n$, giving $O(n\sqrt n)$.
- **Global tokens**: a few special tokens attend to and are attended by everyone (a [CLS] token, or task-specific question tokens in Longformer), providing a shortcut for global information. **BigBird** (Zaheer et al., 2020) combines window, global and random connections and proves the pattern retains the expressive power of full attention under certain assumptions.
- **Hashing / clustering** (Reformer, Kitaev, Kaiser and Levskaya, 2020, using locality-sensitive hashing): attend only within buckets of similar queries and keys.

The practical price: custom sparse kernels, less hardware efficiency than dense attention (a sparse FLOP is slower than a dense FLOP), and, for fixed patterns, no way to attend to arbitrary far tokens in one layer. In practice many production LLMs keep full attention within a context length (extended by RoPE scaling) and use FlashAttention and GQA, while some interleave local and global layers.

### Linear attention {#linear-attention}

The softmax is what forces us to form the $n\times n$ matrix. If we replace $\exp(\boldsymbol{q}^\top\boldsymbol{k})$ with a **kernel** $\phi(\boldsymbol{q})^\top\phi(\boldsymbol{k})$ for a feature map $\phi:\R^{d_k}\to\R^{m}$, associativity lets us regroup the products.

:::math Linear attention and its recurrent form
Define $\alpha_{ij}=\phi(\boldsymbol{q}_i)^\top\phi(\boldsymbol{k}_j)\,/\,\sum_l\phi(\boldsymbol{q}_i)^\top\phi(\boldsymbol{k}_l)$. Then
$$
\boldsymbol{o}_i=\sum_j\alpha_{ij}\boldsymbol{v}_j=\frac{\phi(\boldsymbol{q}_i)^\top\Big(\sum_j\phi(\boldsymbol{k}_j)\boldsymbol{v}_j^\top\Big)}{\phi(\boldsymbol{q}_i)^\top\Big(\sum_l\phi(\boldsymbol{k}_l)\Big)}.
$$
The matrix $\mathbf{S}=\sum_j\phi(\boldsymbol{k}_j)\boldsymbol{v}_j^\top\in\R^{m\times d_v}$ and the vector $\boldsymbol{z}=\sum_l\phi(\boldsymbol{k}_l)\in\R^m$ are computed **once** and shared by all queries, so the total cost is $O(n\,m\,d_v)$, linear in $n$. For a causal model restrict the sums to $j\le i$:
$$
\mathbf{S}_i=\mathbf{S}_{i-1}+\phi(\boldsymbol{k}_i)\boldsymbol{v}_i^\top,\qquad \boldsymbol{z}_i=\boldsymbol{z}_{i-1}+\phi(\boldsymbol{k}_i),\qquad \boldsymbol{o}_i=\frac{\mathbf{S}_i^\top\phi(\boldsymbol{q}_i)}{\boldsymbol{z}_i^\top\phi(\boldsymbol{q}_i)}.
$$
This is a **recurrent network with a matrix-valued state** of fixed size $m\times d_v$, evaluated in parallel during training: the title "Transformers are RNNs" (Katharopoulos, Vyas, Pappas and Fleuret, 2020) refers to exactly this identity. Their feature map was $\phi(x)=\text{elu}(x)+1$ (positive, so the weights are valid); Performer (Choromanski et al., 2021) uses random features to approximate the softmax kernel.
:::

Linear attention achieves constant memory per generated token, but it compresses the entire past into a fixed-size state, so it is a lossy memory, and in practice it has trailed softmax attention in quality on tasks that need precise recall. Its descendants (state-space models such as Mamba, Gu and Dao 2023, and gated linear recurrences) revisit this trade-off; see the discussion in [Week 16](week-16.html). **Linformer** (Wang et al., 2020) takes a different route: project the $n$ keys and values to a fixed $k\ll n$ using learned low-rank maps. The summary: *exact + efficient kernels* (FlashAttention, KV cache, GQA) are the baseline; *approximate* patterns are chosen when context is very long or memory is tight.

{{demo: week09-mask-cost | Mask explorer: allowed score entries and the attention share of FLOPs | 520}}

## Counting parameters and FLOPs {#counting}

### Parameters per layer {#param-count}

Take a standard pre-norm decoder block with width $d$, FFN width $4d$, learned biases, and two LayerNorms:

- attention: $\mathbf{W}_Q,\mathbf{W}_K,\mathbf{W}_V,\mathbf{W}_O$: $4d^2$ weights plus $4d$ biases;
- FFN: $d\cdot4d+4d\cdot d=8d^2$ weights plus $4d+d=5d$ biases;
- two LayerNorms: $2\cdot2d=4d$.

So a block has $12d^2+13d$ parameters, and $L$ blocks give $12Ld^2+13Ld$. Add the token embedding $Vd$ (shared with the output head when weights are tied), a learned position table $T_{\max} d$ if used, and the final LayerNorm $2d$:

$$
N \;=\; 12Ld^2+13Ld+(V+T_{\max})d+2d.
$$

For GPT-2 small ($L=12$, $d=768$, $V=50257$, $T_{\max}=1024$): $12\cdot12\cdot768^2=84.9$M, plus $13\cdot12\cdot768\approx0.12$M, plus $51281\cdot768=39.4$M, giving 124.4M, the published 124M. The "$12Ld^2$" term is the one to remember: **non-embedding parameters $\approx 12Ld^2$**.

```chart
{"type": "bar", "caption": "Parameter counts of the four GPT-2 sizes computed with the formula 12Ld^2 + 13Ld + (V + T)d + 2d (V = 50,257, T = 1,024, tied embeddings). Totals: 124M, 355M, 774M, 1,558M, matching the published sizes.", "height": 340, "data": {"labels": ["small (12L, 768)", "medium (24L, 1024)", "large (36L, 1280)", "XL (48L, 1600)"], "datasets": [{"label": "transformer blocks (M)", "data": [85.1, 302.3, 708.4, 1475.6], "backgroundColor": "var(--c-blue)"}, {"label": "token + position embeddings (M)", "data": [39.4, 52.5, 65.6, 82.0], "backgroundColor": "var(--c-purple)"}]}, "options": {"scales": {"x": {"stacked": true}, "y": {"stacked": true, "title": {"display": true, "text": "parameters (millions)"}}}}}
```

Use the calculator to check your own arithmetic against the presets, and to see how vocabulary size and learned positions matter for small models.

{{demo: week09-transformer-calculator | Parameter and FLOP calculator for decoder-only Transformers | 520}}

### Training FLOPs: the 6N rule {#flops-rule}

A matrix multiply of an $m\times k$ matrix by a $k\times n$ matrix costs $2mkn$ FLOPs. A token passing through all weight matrices of the model does one multiply-add per weight, so the **forward pass costs about $2N$ FLOPs per token** ($N$ counting matrix-multiplied weights; embedding lookups are free but the output head $Vd$ is a real matmul). The backward pass computes two products per weight matrix (the gradient with respect to the input, and with respect to the weights), each the size of the forward product, so it costs about $4N$. Forward plus backward gives the widely used estimate

$$
C \;\approx\; 6\,N\,D\quad\text{FLOPs to train on }D\text{ tokens},
$$

which ignores the attention-score term $\approx 12\,L\,n\,d$ per token (forward and backward) that matters only when $n\gtrsim d$. GPT-2 small on 10 billion tokens costs about $6\cdot1.24\times10^8\cdot10^{10}\approx7.5\times10^{18}$ FLOPs: at a realistic 40% utilization of a 300 TFLOP/s accelerator (120 TFLOP/s effective) that is about 17 hours on one device. This rule is the basis of the scaling-law analysis in [Week 10](week-10.html#scaling-laws).

:::note Memory budget of training
For mixed-precision training with Adam, each parameter costs roughly 16 bytes: 2 for the bf16/fp16 weight, 2 for its gradient, and 12 for fp32 master weight, first moment and second moment (4 + 4 + 4). A 7-billion-parameter model therefore needs about 112 GB of state before activations, which is why large models shard optimizer state (ZeRO/FSDP, [Week 16](week-16.html)).
:::

## Training Transformers {#training}

Transformers are notoriously sensitive to optimization details. The recipe below is a compact summary of what practitioners converge on; each item has a reason, not only a tradition.

### Optimizer: AdamW and its betas {#adamw}

Use [Adam with decoupled weight decay](week-03.html#adaptive-methods) (AdamW, Loshchilov and Hutter, 2019). The original paper used $\beta_2=0.98$, $\epsilon=10^{-9}$; most LLM recipes use $\beta_2=0.95$ (a shorter memory of squared gradients reacts faster to the loss spikes that large models exhibit), $\epsilon\approx10^{-8}$, weight decay $0.1$ applied to weight matrices but **not** to biases, LayerNorm gains or embeddings, and gradient-norm clipping at 1.0.

### Learning-rate warmup {#warmup}

Almost every Transformer recipe increases the learning rate linearly from near zero over the first few hundred to few thousand steps. Why?

1. **Adam's second-moment estimate is unreliable at the start.** After a few steps $\hat v_t$ is computed from very few gradients, so the effective per-parameter step $\eta\,\hat m_t/\sqrt{\hat v_t}$ can be wildly too large for some parameters. Liu et al. (2020, "On the Variance of the Adaptive Learning Rate and Beyond", RAdam) give this analysis and show that warmup is a heuristic cure for the high variance of the adaptive learning rate early on.
2. **Post-norm layers have large initial gradients** near the output (Xiong et al., 2020). A big step in a badly conditioned region can push the model into a bad basin it never leaves. Pre-norm substantially reduces this need for warmup, though large models still use it.
3. **Attention is sharp at initialization for some heads,** and large early updates can saturate softmaxes.

The original paper's schedule, $\eta(t)=d_\text{model}^{-0.5}\min(t^{-0.5},\,t\cdot t_\text{warm}^{-1.5})$ with $t_\text{warm}=4000$, rises linearly and then decays as $1/\sqrt t$. Modern LLM recipes prefer linear warmup followed by **cosine decay** to about 10% of the peak ([Week 3](week-03.html#lr-schedules)).

```chart
{"type": "line", "caption": "Learning-rate schedule from the original Transformer paper (d_model = 512, 4000 warmup steps) against a linear-warmup, cosine-decay schedule with the same peak (computed here for 40,000 steps, illustrative).", "height": 340, "data": {"labels": [250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000, 3250, 3500, 3750, 4000, 4250, 4500, 4750, 5000, 5250, 5500, 5750, 6000, 6250, 6500, 6750, 7000, 7250, 7500, 7750, 8000, 8250, 8500, 8750, 9000, 9250, 9500, 9750, 10000, 10250, 10500, 10750, 11000, 11250, 11500, 11750, 12000, 12250, 12500, 12750, 13000, 13250, 13500, 13750, 14000, 14250, 14500, 14750, 15000, 15250, 15500, 15750, 16000, 16250, 16500, 16750, 17000, 17250, 17500, 17750, 18000, 18250, 18500, 18750, 19000, 19250, 19500, 19750, 20000, 20250, 20500, 20750, 21000, 21250, 21500, 21750, 22000, 22250, 22500, 22750, 23000, 23250, 23500, 23750, 24000, 24250, 24500, 24750, 25000, 25250, 25500, 25750, 26000, 26250, 26500, 26750, 27000, 27250, 27500, 27750, 28000, 28250, 28500, 28750, 29000, 29250, 29500, 29750, 30000, 30250, 30500, 30750, 31000, 31250, 31500, 31750, 32000, 32250, 32500, 32750, 33000, 33250, 33500, 33750, 34000, 34250, 34500, 34750, 35000, 35250, 35500, 35750, 36000, 36250, 36500, 36750, 37000, 37250, 37500, 37750, 38000, 38250, 38500, 38750, 39000, 39250, 39500, 39750, 40000], "datasets": [{"label": "inverse-square-root (Vaswani et al.)", "data": [0.0437, 0.0873, 0.131, 0.1747, 0.2184, 0.262, 0.3057, 0.3494, 0.3931, 0.4367, 0.4804, 0.5241, 0.5678, 0.6114, 0.6551, 0.6988, 0.6779, 0.6588, 0.6412, 0.625, 0.6099, 0.5959, 0.5828, 0.5705, 0.559, 0.5482, 0.5379, 0.5282, 0.519, 0.5103, 0.502, 0.4941, 0.4866, 0.4794, 0.4725, 0.4658, 0.4595, 0.4534, 0.4476, 0.4419, 0.4365, 0.4313, 0.4262, 0.4214, 0.4167, 0.4121, 0.4077, 0.4034, 0.3993, 0.3953, 0.3914, 0.3876, 0.3839, 0.3804, 0.3769, 0.3735, 0.3702, 0.367, 0.3639, 0.3608, 0.3579, 0.355, 0.3521, 0.3494, 0.3467, 0.3441, 0.3415, 0.339, 0.3365, 0.3341, 0.3317, 0.3294, 0.3271, 0.3249, 0.3227, 0.3206, 0.3185, 0.3165, 0.3145, 0.3125, 0.3106, 0.3087, 0.3068, 0.305, 0.3032, 0.3014, 0.2997, 0.298, 0.2963, 0.2946, 0.293, 0.2914, 0.2898, 0.2883, 0.2868, 0.2853, 0.2838, 0.2823, 0.2809, 0.2795, 0.2781, 0.2768, 0.2754, 0.2741, 0.2728, 0.2715, 0.2702, 0.269, 0.2677, 0.2665, 0.2653, 0.2641, 0.2629, 0.2618, 0.2606, 0.2595, 0.2584, 0.2573, 0.2562, 0.2552, 0.2541, 0.2531, 0.252, 0.251, 0.25, 0.249, 0.248, 0.2471, 0.2461, 0.2451, 0.2442, 0.2433, 0.2424, 0.2415, 0.2406, 0.2397, 0.2388, 0.2379, 0.2371, 0.2362, 0.2354, 0.2346, 0.2337, 0.2329, 0.2321, 0.2313, 0.2305, 0.2298, 0.229, 0.2282, 0.2275, 0.2267, 0.226, 0.2252, 0.2245, 0.2238, 0.2231, 0.2224, 0.2217, 0.221], "borderColor": "var(--c-blue)", "pointRadius": 0}, {"label": "linear warmup + cosine decay", "data": [0.0437, 0.0873, 0.131, 0.1747, 0.2184, 0.262, 0.3057, 0.3494, 0.3931, 0.4367, 0.4804, 0.5241, 0.5678, 0.6114, 0.6551, 0.6988, 0.6987, 0.6984, 0.698, 0.6974, 0.6967, 0.6958, 0.6947, 0.6935, 0.6921, 0.6905, 0.6888, 0.6869, 0.6848, 0.6826, 0.6802, 0.6777, 0.675, 0.6722, 0.6692, 0.666, 0.6627, 0.6593, 0.6557, 0.652, 0.6481, 0.6441, 0.6399, 0.6356, 0.6311, 0.6266, 0.6219, 0.617, 0.6121, 0.607, 0.6018, 0.5964, 0.591, 0.5854, 0.5798, 0.574, 0.5681, 0.5621, 0.556, 0.5498, 0.5435, 0.5371, 0.5306, 0.5241, 0.5174, 0.5107, 0.5039, 0.497, 0.4901, 0.4831, 0.476, 0.4689, 0.4617, 0.4544, 0.4472, 0.4398, 0.4324, 0.425, 0.4175, 0.4101, 0.4025, 0.395, 0.3874, 0.3798, 0.3722, 0.3646, 0.357, 0.3494, 0.3418, 0.3341, 0.3265, 0.3189, 0.3113, 0.3038, 0.2962, 0.2887, 0.2812, 0.2738, 0.2663, 0.259, 0.2516, 0.2443, 0.2371, 0.2299, 0.2228, 0.2157, 0.2087, 0.2017, 0.1949, 0.1881, 0.1813, 0.1747, 0.1681, 0.1617, 0.1553, 0.149, 0.1428, 0.1367, 0.1307, 0.1248, 0.119, 0.1133, 0.1078, 0.1023, 0.097, 0.0918, 0.0867, 0.0817, 0.0769, 0.0722, 0.0676, 0.0632, 0.0589, 0.0547, 0.0507, 0.0468, 0.0431, 0.0395, 0.036, 0.0327, 0.0296, 0.0266, 0.0238, 0.0211, 0.0185, 0.0162, 0.014, 0.0119, 0.01, 0.0083, 0.0067, 0.0053, 0.0041, 0.003, 0.0021, 0.0013, 0.0007, 0.0003, 0.0001, 0.0], "borderColor": "var(--c-green)", "pointRadius": 0}]}, "options": {"scales": {"x": {"title": {"display": true, "text": "step"}}, "y": {"title": {"display": true, "text": "learning rate (x 1e-3)"}}}}}
```

### Label smoothing {#label-smoothing}

For a vocabulary of $V$ classes the one-hot target pushes the model's logit for the correct class toward infinity. **Label smoothing** (Szegedy et al., 2016; used in the Transformer paper with $\epsilon=0.1$) replaces the target by $q_c=(1-\epsilon)\,\mathbb 1[c=y]+\epsilon/V$. The loss becomes a mix of the usual cross-entropy and a cross-entropy against the uniform distribution:

$$
\mathcal{L}=(1-\epsilon)\,\big(-\log p_y\big)+\epsilon\cdot\Big(-\frac1V\sum_c\log p_c\Big).
$$

The effect on a translation model was a *worse* perplexity (the model is deliberately less confident) but a *better* BLEU score, because better calibrated outputs make beam search work better. For modern pretraining of decoder-only LLMs it is usually omitted. In PyTorch: `F.cross_entropy(logits, y, label_smoothing=0.1)`.

### Mixed precision {#mixed-precision}

Matrix multiplications on modern GPUs run several times faster in 16-bit formats, and 16-bit tensors halve memory. There are two formats, and the difference matters:

| Format | Exponent bits | Mantissa bits | Range | Notes |
| --- | --- | --- | --- | --- |
| fp32 | 8 | 23 | about $10^{\pm38}$ | reference |
| fp16 | 5 | 10 | max 65504, min normal about $6\times10^{-5}$ | more precise but narrow range; needs **loss scaling** |
| bf16 | 8 | 7 | same range as fp32 | coarser precision, no loss scaling needed |

With fp16, gradients smaller than about $6\times10^{-8}$ underflow to zero; **dynamic loss scaling** (Micikevicius et al., 2018) multiplies the loss by a large factor before backward and divides the gradients afterwards, lowering the scale when an overflow (inf/NaN) is detected. bf16 has the exponent range of fp32 and therefore needs none of this, which is why it is the default on hardware that supports it (A100/H100, TPUs, recent consumer GPUs). The rule of thumb: keep **weights' master copy, optimizer state, softmax, normalization statistics and the loss in fp32**; run matmuls in 16-bit. PyTorch's `torch.autocast` automates the choices.

```python:amp.py
import torch

def train_step(model, opt, scaler, x, y, use_bf16=True):
    # autocast chooses 16-bit for matmuls and keeps reductions/softmax/loss in fp32
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    with torch.autocast(device_type="cuda", dtype=dtype):
        _, loss = model(x, y)
    if use_bf16:
        loss.backward()                                  # no scaler needed for bf16
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    else:
        scaler.scale(loss).backward()                    # fp16: scale to avoid underflow
        scaler.unscale_(opt)                             # unscale BEFORE clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)                                 # skips the step if grads have inf/NaN
        scaler.update()
    opt.zero_grad(set_to_none=True)
    return loss.item()

# scaler = torch.amp.GradScaler("cuda")   # only needed for fp16
```

### Dropout, initialization and other details {#training-details}

- **Dropout** of 0.1 on attention weights, sublayer outputs and embeddings was standard for small supervised models; large pretrained models trained for a single epoch on huge corpora often use none, because overfitting is not the issue ([Week 4](week-04.html)).
- **Initialization**: normal with standard deviation 0.02 for weights (GPT-2), and for the output projections that write into the residual stream, a scale of $0.02/\sqrt{2L}$ so that the stream's variance does not grow with depth (each of the $2L$ sublayers adds a term).
- **Batch size** in tokens: hundreds of thousands to millions for large runs; use gradient accumulation if memory-limited.
- **Loss spikes**: occasionally the loss jumps and may or may not recover. Remedies are lowering the learning rate, skipping the offending batch, restarting from an earlier checkpoint, QK-norm, and keeping softmax and logits in fp32.

## Lab: a GPT from scratch in PyTorch {#lab}

We now assemble every piece into a small GPT-style decoder (about 800k parameters), train it as a **character-level language model** on a tiny corpus, and sample text. The structure mirrors Karpathy's nanoGPT, which you should read afterwards. A GPU makes it fast (about two minutes on a Colab T4), but it also runs on CPU with smaller settings.

### Step 1: data {#lab-data}

```python:lab_data.py
import os, urllib.request, torch

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
if not os.path.exists("input.txt"):
    try:
        urllib.request.urlretrieve(URL, "input.txt")
    except Exception:                                   # offline fallback: tiny repeated corpus
        text = ("to be or not to be that is the question\n"
                "whether tis nobler in the mind to suffer\n") * 200
        open("input.txt", "w").write(text)
text = open("input.txt", encoding="utf-8").read()
chars = sorted(set(text))                                # character vocabulary
V = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: "".join(itos[i] for i in ids)

data = torch.tensor(encode(text), dtype=torch.long)
n_train = int(0.9 * len(data))
train_data, val_data = data[:n_train], data[n_train:]
print(f"{len(text):,} chars, vocab {V}")                 # ~1.1M chars, vocab 65 for tinyshakespeare

def get_batch(split, B, T, device):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - T - 1, (B,))             # random start offsets
    x = torch.stack([d[i : i + T] for i in ix])          # (B, T) inputs
    y = torch.stack([d[i + 1 : i + T + 1] for i in ix])  # (B, T) targets = inputs shifted by one
    return x.to(device), y.to(device)

xb, yb = get_batch("train", 4, 8, "cpu")
print(xb.shape, yb.shape)                                # torch.Size([4, 8]) torch.Size([4, 8])
```

### Step 2: the model {#lab-model}

```python:lab_model.py
import math
import torch, torch.nn as nn, torch.nn.functional as F

class CausalSelfAttention(nn.Module):
    def __init__(self, d, h, dropout):
        super().__init__()
        assert d % h == 0
        self.h, self.dk = h, d // h
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.dropout = dropout
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x):                                  # x: (B, T, d)
        B, T, d = x.shape
        q, k, v = self.qkv(x).split(d, dim=2)
        q, k, v = (t.view(B, T, self.h, self.dk).transpose(1, 2) for t in (q, k, v))  # (B,h,T,dk)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                           dropout_p=self.dropout if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(B, T, d)   # (B, T, d)
        return self.resid_drop(self.proj(y))

class MLP(nn.Module):
    def __init__(self, d, dropout):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(d, 4 * d, bias=False), nn.Linear(4 * d, d, bias=False)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        return self.drop(self.fc2(F.gelu(self.fc1(x))))

class Block(nn.Module):                                    # pre-norm Transformer block
    def __init__(self, d, h, dropout):
        super().__init__()
        self.ln1, self.attn = nn.LayerNorm(d), CausalSelfAttention(d, h, dropout)
        self.ln2, self.mlp = nn.LayerNorm(d), MLP(d, dropout)
    def forward(self, x):
        x = x + self.attn(self.ln1(x))                     # communicate
        x = x + self.mlp(self.ln2(x))                      # compute
        return x

class GPT(nn.Module):
    def __init__(self, vocab, block_size=128, n_layer=4, n_head=4, d=128, dropout=0.1):
        super().__init__()
        self.block_size = block_size
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(block_size, d)             # learned absolute positions
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(Block(d, n_head, dropout) for _ in range(n_layer))
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.tok.weight                 # weight tying
        self.apply(self._init)
        for name, p in self.named_parameters():            # scaled init for residual writers
            if name.endswith("proj.weight") or name.endswith("fc2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def forward(self, idx, targets=None):                  # idx: (B, T) int64
        B, T = idx.shape
        assert T <= self.block_size
        pos = torch.arange(T, device=idx.device)           # (T,)
        x = self.drop(self.tok(idx) + self.pos(pos))       # (B, T, d)
        for blk in self.blocks:
            x = blk(x)
        logits = self.head(self.ln_f(x))                   # (B, T, V)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        self.eval()
        for _ in range(max_new_tokens):
            ctx = idx[:, -self.block_size:]                # crop to the context window
            logits, _ = self(ctx)
            logits = logits[:, -1, :] / max(temperature, 1e-6)       # (B, V) last position only
            if top_k is not None:
                kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, [-1]]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx

model = GPT(vocab=65)
n_params = sum(p.numel() for p in model.parameters())
print(f"{n_params/1e6:.2f}M parameters")                   # ~0.82M (tied head counted once)
```

Two sanity checks before training. First, the initial loss should be close to $\ln V$ (about 4.17 for $V=65$), because a model with tiny initial weights predicts a nearly uniform distribution. Second, a **causality test**: change a future token and verify that earlier logits are unchanged.

```python:lab_checks.py
import math, torch
model = GPT(vocab=65).eval()
x = torch.randint(0, 65, (2, 16))
_, loss = model(x, x)
print(loss.item(), math.log(65))                         # both close to 4.17

x2 = x.clone()
x2[:, 10] = (x2[:, 10] + 1) % 65                         # corrupt token at position 10
l1, _ = model(x)
l2, _ = model(x2)
print(torch.allclose(l1[:, :10], l2[:, :10], atol=1e-5))  # True: positions < 10 unaffected
print(torch.allclose(l1[:, 10:], l2[:, 10:], atol=1e-5))  # False: positions >= 10 changed
```

### Step 3: training {#lab-train}

```python:lab_train.py
import math, time, torch

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(1337)
B, T = 64, 128
model = GPT(vocab=V, block_size=T, n_layer=4, n_head=4, d=128, dropout=0.1).to(device)

# weight decay only on matrices (ndim >= 2), not on biases/LayerNorm gains
decay = [p for p in model.parameters() if p.ndim >= 2]
no_decay = [p for p in model.parameters() if p.ndim < 2]
opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1},
                         {"params": no_decay, "weight_decay": 0.0}],
                        lr=3e-3, betas=(0.9, 0.95))

max_steps, warmup, lr_max, lr_min = 2000, 100, 3e-3, 3e-4
def lr_at(step):
    if step < warmup:
        return lr_max * (step + 1) / warmup              # linear warmup
    t = (step - warmup) / (max_steps - warmup)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * t))   # cosine decay

@torch.no_grad()
def evaluate(n_batches=20):
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(n_batches)
        for i in range(n_batches):
            x, y = get_batch(split, B, T, device)
            _, loss = model(x, y)
            losses[i] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

use_amp = device == "cuda"
t0 = time.time()
for step in range(max_steps + 1):
    for g in opt.param_groups:
        g["lr"] = lr_at(step)
    if step % 250 == 0:
        e = evaluate()
        print(f"step {step:4d} | train {e['train']:.3f} | val {e['val']:.3f} | {time.time()-t0:.0f}s")
    x, y = get_batch("train", B, T, device)
    with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_amp):
        _, loss = model(x, y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    opt.zero_grad(set_to_none=True)
```

On tinyshakespeare you should see the loss fall from about 4.2 at step 0 to roughly 1.6 to 1.8 on validation after 2000 steps (numbers vary with seed and hardware; a larger model with more steps reaches about 1.5). If your loss stays above 3, see the debugging checklist below.

```text
step    0 | train 4.213 | val 4.209 | 1s
step  250 | train 2.431 | val 2.452 | 14s
step  500 | train 2.127 | val 2.175 | 27s
...
step 2000 | train 1.620 | val 1.742 | 105s
```

(Illustrative output; your numbers will differ in the second decimal.)

### Step 4: generation and the effect of temperature {#lab-generate}

```python:lab_generate.py
prompt = torch.zeros((1, 1), dtype=torch.long, device=device)       # start from token id 0 (newline)
for temp, k in [(0.5, None), (1.0, None), (1.0, 10)]:
    out = model.generate(prompt, max_new_tokens=300, temperature=temp, top_k=k)
    print(f"--- temperature {temp}, top_k {k} ---")
    print(decode(out[0].tolist()))
```

Low temperature yields repetitive but locally well-formed text, temperature 1 samples from the model's actual distribution and produces more variety (and more typos), and top-$k$ trims the improbable tail. [Week 10](week-10.html#decoding) analyzes these decoding strategies carefully and gives an interactive playground.

### Step 5: speed up generation with a KV cache (extension) {#lab-kvcache}

The `generate` method above recomputes everything for the whole context at each step. As an extension, modify `CausalSelfAttention.forward` to accept and return a `(k, v)` cache: when generating, feed only the newest token, concatenate its key and value to the cached ones along the time axis, and compute attention with `is_causal=False` (a single query attending to all cached keys is already causal). Remember to pass the correct absolute position to the position embedding. Check your implementation by asserting that the logits agree with the uncached path to within $10^{-5}$.

## Common mistakes and debugging tips {#pitfalls}

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| Loss is NaN from the first steps | A row of the mask is all $-\infty$ (softmax of all $-\infty$); or fp16 overflow | Ensure every query can attend to at least one key; use bf16 or a loss scaler; compute softmax in fp32 |
| Loss decreases suspiciously fast to near zero | **Information leak**: missing or wrong causal mask, or targets not shifted | Run the causality test above; check `y = x` shifted by one |
| Loss stuck near $\ln V$ | Learning rate too high or too low, no warmup, wrong init, forgot to call `opt.step()` or zero grads | Overfit a single batch first; add warmup; lower LR |
| Loss good but generations are garbage | `model.eval()` not set during generation (dropout active); wrong context cropping; temperature bug | Call `eval()`; check that the last position's logits are sampled, not the first |
| Boolean/float mask confusion | `True` means *allowed* in `scaled_dot_product_attention` but *masked* in `nn.MultiheadAttention`'s `attn_mask` and `key_padding_mask` | Read the docs for each API; write a unit test with a tiny known case |
| Padding tokens influence outputs | Pad keys not masked | Build a key padding mask; test that appending pads does not change non-pad outputs |
| Shapes are right but results wrong after reshaping heads | Used `view(B, h, T, dk)` directly instead of `view(B, T, h, dk).transpose(1, 2)` | The head split must keep each token's features contiguous; transpose after view |
| Out-of-memory at longer sequence | $O(n^2)$ score matrices materialized | Use fused SDPA, gradient checkpointing, smaller batch, shorter context |
| Model cannot use word order | Position information missing, or added to the wrong tensor | Verify that permuting input tokens changes the output |
| Training unstable at larger scale | Post-norm without warmup, $\beta_2=0.999$, attention logit growth | Pre-norm, $\beta_2=0.95$, grad clipping, QK-norm, lower LR |

:::tip The three debugging tests every Transformer should pass
1. **Overfit one batch**: loss should go to about zero within a few hundred steps.
2. **Initial loss $\approx\ln V$**.
3. **Causality / invariance tests**: changing future tokens must not change past logits; appending pad tokens must not change non-pad outputs.
:::

## Exercises {#exercises}

:::exercise Exercise 1: variance and the scale factor (derivation)
Suppose queries and keys have i.i.d. components with mean 0 and variance $\sigma^2$, independent of each other. (a) What is $\Var(\boldsymbol{q}^\top\boldsymbol{k})$? (b) What scaling factor restores unit variance? (c) After training, LayerNorm'd inputs and learned projections often give $\sigma^2\ne1$. Why does the $1/\sqrt{d_k}$ factor still matter?
:::details Solution
(a) Each product $q_mk_m$ has variance $\E[q_m^2]\E[k_m^2]=\sigma^4$, and the $d_k$ terms are independent, so $\Var=d_k\sigma^4$. (b) Divide by $\sigma^2\sqrt{d_k}$; with $\sigma=1$ this is $1/\sqrt{d_k}$. (c) The factor fixes the *initial* statistics so that training starts in a well-conditioned regime; the network can then learn larger or smaller effective temperatures through the norms of $\mathbf{W}_Q,\mathbf{W}_K$. Without it, the scale at initialization grows with $d_k$ and the softmax saturates (gradients vanish) before the network has had a chance to adapt.
:::
:::

:::exercise Exercise 2: shapes of multi-head attention (conceptual)
A model has $d=768$, $h=12$. For batch size 4 and sequence length 100, give the shape of every tensor from input to output of one self-attention layer, the number of parameters in the layer (with biases), and the number of floats stored for the attention probabilities.
:::details Solution
$d_k=64$. Input `(4, 100, 768)`; after the fused QKV projection `(4, 100, 2304)`, split into three `(4, 100, 768)`; heads `(4, 12, 100, 64)`; scores and probabilities `(4, 12, 100, 100)`; weighted values `(4, 12, 100, 64)`; merged `(4, 100, 768)`; output `(4, 100, 768)`. Parameters: $4\cdot768^2+4\cdot768=2{,}362{,}368$. Attention probabilities: $4\cdot12\cdot100\cdot100=480{,}000$ floats per layer (about 0.96 MB in fp16), which for $n=8192$ would be $4\cdot 12\cdot 8192^2\approx3.2\times10^9$ floats: the quadratic cost in action.
:::
:::

:::exercise Exercise 3: implement masks (coding)
Write a function `make_mask(lengths, T)` returning a boolean `(B, 1, T, T)` tensor that combines a causal mask with a key padding mask for sequences of the given true lengths. Test that (i) future entries are False, (ii) padded keys are False for every query, and (iii) the output of `MultiHeadAttention` at non-pad positions is unchanged when you append extra pad tokens.
:::details Solution
```python
import torch
def make_mask(lengths, T):
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool))
    keep = torch.arange(T)[None, :] < lengths[:, None]       # (B, T)
    return causal[None, None] & keep[:, None, None, :]       # (B, 1, T, T)

lengths = torch.tensor([4, 2]); T = 6
m = make_mask(lengths, T)
assert not m[:, :, 0, 1].any()                               # future blocked
assert not m[1, 0, :, 2:].any()                              # padded keys blocked
# (iii) run MHA on a batch of length 4 and on the same batch padded to 6 with arbitrary
# values in the pad slots; the first 4 output rows must agree (atol 1e-5).
```
Note that query rows at pad positions may be all False when the sequence has zero real tokens before them; with a causal mask plus padding they still have key 0 allowed whenever the sequence has at least one real token, so no NaNs arise for non-empty sequences.
:::
:::

:::exercise Exercise 4: RoPE and relative position (derivation)
Show that for the block-diagonal rotation $\mathbf{R}_m$ of RoPE, the attention score between a query at position $m$ and a key at position $n$ equals $\boldsymbol{q}^\top\mathbf{R}_{n-m}\boldsymbol{k}$. Then show that the effect of this score on pair $i$ vanishes slowly for small $\theta_i$ (low frequency) and oscillates fast for large $\theta_i$, and explain why this gives a mixture of local and long-range sensitivity.
:::details Solution
$\mathbf{R}_m$ is orthogonal with $\mathbf{R}_m^\top=\mathbf{R}_{-m}$ and $\mathbf{R}_a\mathbf{R}_b=\mathbf{R}_{a+b}$ (block-wise rotations add angles). Therefore $(\mathbf{R}_m\boldsymbol{q})^\top(\mathbf{R}_n\boldsymbol{k})=\boldsymbol{q}^\top\mathbf{R}_{-m}\mathbf{R}_n\boldsymbol{k}=\boldsymbol{q}^\top\mathbf{R}_{n-m}\boldsymbol{k}$. Pair $i$ contributes $\|\boldsymbol{q}_i\|\|\boldsymbol{k}_i\|\cos(\angle_i+(n-m)\theta_i)$, where $\angle_i$ is the angle between the unrotated pair vectors. For $\theta_i=1$ the cosine cycles every $2\pi\approx6$ positions: it can encode the exact small offset but aliases at larger ones. For $\theta_i=10000^{-1}\approx10^{-4}$ the cosine changes appreciably only over thousands of positions: it carries coarse, long-range distance information. Together they are like the clocks in the sinusoidal encoding, now acting on the *score*.
:::
:::

:::exercise Exercise 5: count parameters (computation)
Compute the number of parameters of a decoder-only model with $L=24$, $d=2048$, $d_\text{ff}=4d$, vocabulary 32,000, tied embeddings, RoPE (no position parameters), no biases, RMSNorm with $d$ gains per norm and one final norm. Then compute its training FLOPs on 300 billion tokens using $6ND$.
:::details Solution
Per block: attention $4d^2$, FFN $8d^2$, two RMSNorms $2d$: $12d^2+2d=12\cdot4{,}194{,}304+4096=50{,}335{,}744$. For 24 blocks: $1{,}208{,}057{,}856$. Embeddings: $32000\cdot2048=65{,}536{,}000$. Final norm 2048. Total $\approx1.2736\times10^9$, about 1.27B. Training: $6\cdot1.27\times10^9\cdot3\times10^{11}\approx2.3\times10^{21}$ FLOPs (using the total; using only the matrix-multiplied parameters gives a similar number). At $10^{14}$ effective FLOP/s per GPU this is $2.3\times10^7$ GPU-seconds, about 265 GPU-days.
:::
:::

:::exercise Exercise 6: pre-norm versus post-norm experiment (coding)
Using the lab code, add a flag to switch `Block` to post-norm ($x\leftarrow\text{LN}(x+\text{Sublayer}(x))$). Train both variants with the same seed, six layers, a deliberately large learning rate of $3\times10^{-3}$ and no warmup, and then again with 200 warmup steps. Report the loss curves and explain what you see.
:::details Solution
Expected outcome (to be verified by running): with no warmup and a high learning rate the post-norm model typically trains poorly or diverges, while the pre-norm model trains; with warmup both train, with post-norm often ending slightly worse or similar at this tiny scale. The explanation is the gradient-scale difference of the two placements described in the text. A minimal post-norm block is:
```python
def forward(self, x):
    x = self.ln1(x + self.attn(x))
    return self.ln2(x + self.mlp(x))
```
(and drop the final `ln_f`, since each block already ends in a norm). Plot both curves with and without warmup and relate any divergence to the first 100 steps.
:::
:::

:::exercise Exercise 7: what would happen if... (conceptual)
For each modification, predict the effect on a trained decoder-only LM and on training: (a) remove the positional encoding; (b) remove the causal mask but still train on next-token prediction; (c) apply RoPE to the values as well as queries and keys; (d) share all of $\mathbf{W}_Q,\mathbf{W}_K,\mathbf{W}_V$ within a head.
:::details Solution
(a) Content-only attention is permutation-equivariant, but the causal mask alone provides some order information (token $i$ sees exactly $i$ predecessors), so the model can still learn to some degree; it will be weaker at tasks that depend on exact offsets. In an encoder without a mask, the model becomes a bag-of-words model. (b) Each position can read its own target token (the next token is in its context), so the loss collapses toward zero in training and the model learns to copy, not to predict: generation then fails. This is the "leak" described in the pitfalls table. (c) Rotating values would make the output depend on absolute positions (the weighted sum would mix vectors rotated by different angles) and break relative-only behavior; RoPE is deliberately applied to $\mathbf{Q}$ and $\mathbf{K}$ only. (d) With $\mathbf{W}_Q=\mathbf{W}_K$ the score matrix $\mathbf{X}\mathbf{W}\mathbf{W}^\top\mathbf{X}^\top$ is symmetric, so token $i$'s affinity for $j$ equals $j$'s for $i$ (before the softmax), and a token always has high affinity with itself. That removes useful asymmetric relations (such as a verb looking for its subject but not vice versa) and reduces capacity.
:::
:::

:::exercise Exercise 8: linear attention equals a recurrence (derivation, challenging)
Starting from $\boldsymbol{o}_i=\sum_{j\le i}\alpha_{ij}\boldsymbol{v}_j$ with $\alpha_{ij}=\phi(\boldsymbol{q}_i)^\top\phi(\boldsymbol{k}_j)/\sum_{l\le i}\phi(\boldsymbol{q}_i)^\top\phi(\boldsymbol{k}_l)$, derive the recurrent form and compute the time and memory cost of generating $T$ tokens, versus softmax attention with a KV cache. Then explain what information the fixed-size state cannot represent that a KV cache can.
:::details Solution
Factor the numerator: $\sum_{j\le i}\phi(\boldsymbol{q}_i)^\top\phi(\boldsymbol{k}_j)\boldsymbol{v}_j=\big(\sum_{j\le i}\boldsymbol{v}_j\phi(\boldsymbol{k}_j)^\top\big)\phi(\boldsymbol{q}_i)=\mathbf{S}_i^\top\phi(\boldsymbol{q}_i)$ with $\mathbf{S}_i=\sum_{j\le i}\phi(\boldsymbol{k}_j)\boldsymbol{v}_j^\top\in\R^{m\times d_v}$, and the denominator is $\boldsymbol{z}_i^\top\phi(\boldsymbol{q}_i)$ with $\boldsymbol{z}_i=\sum_{l\le i}\phi(\boldsymbol{k}_l)$. Both satisfy $\mathbf{S}_i=\mathbf{S}_{i-1}+\phi(\boldsymbol{k}_i)\boldsymbol{v}_i^\top$ and $\boldsymbol{z}_i=\boldsymbol{z}_{i-1}+\phi(\boldsymbol{k}_i)$. Each step costs $O(m d_v)$ time, and memory is $O(md_v)$ independent of $i$; generating $T$ tokens costs $O(Tmd_v)$ time and $O(md_v)$ memory. With a KV cache, step $t$ costs $O(td)$ and the cache holds $O(td)$ numbers, so total time $O(T^2d)$ and memory growing linearly. The state $\mathbf{S}$ is a sum of $T$ rank-one terms compressed into $m\times d_v$ numbers: when $T\gg m$ the individual $(\boldsymbol{k}_j,\boldsymbol{v}_j)$ pairs overlap in the state and exact recall of a specific old token (e.g. a rare name given 3000 tokens earlier) is not possible, whereas the KV cache keeps them all separate.
:::
:::

:::exercise Exercise 9: break the position encoding (challenging, coding)
Train the lab GPT on a synthetic task where the target is the input sequence reversed (feed a random string of 16 digits, then a separator, and train the model to output the string reversed). Compare (i) learned absolute positions, (ii) no positions, and (iii) RoPE (apply `apply_rope` to $\mathbf{Q}$, $\mathbf{K}$ in `CausalSelfAttention` and delete the position embedding). Which succeed, and why? Then test (i) and (iii) on strings of length 32.
:::details Solution
Expected behavior (verify by running): the reversal task requires the model to attend to position $2T-i$ when writing token $i$, a *positional* operation. With learned absolute positions or RoPE the model solves it for the training length; with no positions it can use only the causal-mask-induced counts and generally fails or does so slowly. At length 32, learned positions are undefined beyond the table (the table has only 16+1+16 entries unless you enlarge it, and untrained rows are random), so learned absolute positions cannot extrapolate. RoPE can generalize partially, but because the model learned frequencies for distances up to about 33 it can still degrade; this is the same length-extrapolation limitation discussed in the RoPE section.
:::
:::

## Key takeaways {#takeaways}

:::key Key takeaways
- **Attention is a soft, differentiable dictionary lookup**: output $=\softmax(\mathbf{Q}\mathbf{K}^\top/\sqrt{d_k})\mathbf{V}$, each token's output a convex combination of value vectors. It is permutation-equivariant, so positions must be injected.
- **The $1/\sqrt{d_k}$ factor** keeps the logit variance at 1 (the raw variance is $d_k$), preventing softmax saturation and vanishing gradients at initialization.
- **Multi-head attention** splits $d$ into $h$ subspaces by `view` + `transpose`: tensors are `(B, h, T, d_k)` and the score tensor is `(B, h, T, T)`. Heads are merged and mixed by $\mathbf{W}_O$.
- **Block = communication + computation**: attention moves information between tokens, the FFN (about two thirds of the parameters, $8d^2$) transforms each token, and residual connections form the stream both write to.
- **Pre-norm** (with RMSNorm) keeps an identity path and trains stably; post-norm needs warmup and care.
- **Position**: sinusoidal and learned absolute vectors are added at the input; relative biases, **RoPE** (rotate $\mathbf{Q},\mathbf{K}$ so scores depend on $n-m$) and ALiBi act on the scores. RoPE is the modern default, with scaling for longer contexts.
- **Architectures differ by mask**: encoder-only (full), decoder-only (causal), encoder-decoder (full + causal + cross). The causal mask enables parallel training and requires a KV cache for fast generation.
- **Cost**: a layer is $24nd^2+4n^2d$ FLOPs; parameters $\approx12Ld^2$; training $\approx6ND$. The quadratic term bites beyond $n\approx6d$ and in memory first. FlashAttention (exact), GQA and KV caching are the baseline fixes; sparse and linear variants are approximations with trade-offs.
- **Training recipe**: AdamW with $\beta_2\approx0.95$, linear warmup then cosine decay, gradient clipping, bf16 mixed precision (fp16 needs loss scaling), weight decay on matrices only, scaled residual-output init.
:::

## Further reading {#further-reading}

- **(Essential)** Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, Polosukhin. "Attention Is All You Need." NeurIPS 2017 (arXiv:1706.03762).
- **(Essential)** Jay Alammar, "The Illustrated Transformer" (blog, 2018): the best visual walkthrough.
- **(Essential)** Andrej Karpathy, "Let's build GPT: from scratch, in code, spelled out" (video, 2023) and the nanoGPT repository.
- Elhage et al. "A Mathematical Framework for Transformer Circuits." Anthropic, 2021: the residual-stream view.
- Xiong et al. "On Layer Normalization in the Transformer Architecture." ICML 2020 (arXiv:2002.04745): pre-norm versus post-norm.
- Su, Lu, Pan, Murtadha, Wen, Liu. "RoFormer: Enhanced Transformer with Rotary Position Embedding." 2021 (arXiv:2104.09864).
- Press, Smith, Lewis. "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation." ICLR 2022 (arXiv:2108.12409).
- Shaw, Uszkoreit, Vaswani. "Self-Attention with Relative Position Representations." NAACL 2018.
- Dao, Fu, Ermon, Rudra, Ré. "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness." NeurIPS 2022 (arXiv:2205.14135).
- Katharopoulos, Vyas, Pappas, Fleuret. "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention." ICML 2020 (arXiv:2006.16236).
- Child, Gray, Radford, Sutskever. "Generating Long Sequences with Sparse Transformers." 2019 (arXiv:1904.10509); Beltagy, Peters, Cohan. "Longformer." 2020 (arXiv:2004.05150).
- Zhang, Sennrich. "Root Mean Square Layer Normalization." NeurIPS 2019; Shazeer. "GLU Variants Improve Transformer." 2020 (arXiv:2002.05202).
- Geva, Schuster, Berant, Levy. "Transformer Feed-Forward Layers Are Key-Value Memories." EMNLP 2021.
- Radford et al. "Language Models are Unsupervised Multitask Learners" (GPT-2), 2019; Micikevicius et al. "Mixed Precision Training." ICLR 2018.
