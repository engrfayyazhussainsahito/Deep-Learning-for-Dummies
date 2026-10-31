---
title: "Sequence Models: RNNs, LSTMs, and the Road to Attention"
phase: Core Architectures
summary: How to model data that arrives in order. Recurrent networks, backpropagation through time and why gradients vanish or explode, LSTM and GRU gating, sequence-to-sequence learning with teacher forcing and beam search, and the attention mechanism that made recurrence optional.
---

:::key Learning objectives
By the end of this week you should be able to:
- Classify sequence problems (one-to-many, many-to-one, many-to-many) and formulate language modeling as next-token prediction, including cross-entropy, bits per character and perplexity.
- Write the equations of a vanilla RNN, unroll it, and implement the forward pass from scratch in NumPy and with `torch.nn.RNN`, with correct tensor shapes.
- Derive backpropagation through time, show how the gradient is a product of Jacobians, and use the spectral radius to explain vanishing and exploding gradients. Apply truncated BPTT and gradient clipping.
- Explain every gate of an LSTM and a GRU, and argue from the cell-state recursion why gating improves gradient flow.
- Build bidirectional and stacked recurrent networks and know when a bidirectional model is not allowed (autoregressive generation).
- Implement an encoder-decoder with teacher forcing, discuss exposure bias, and implement greedy and beam-search decoding.
- Derive Bahdanau (additive) and Luong (multiplicative) attention, and read an attention heatmap as a soft alignment.
- Compare recurrent models with 1D convolutional and temporal convolutional networks for translation, speech and time-series forecasting.
:::

So far every input in this course had a fixed size: a vector of features, an image of $224\times224$ pixels. Much of the world is not like that. A sentence has a variable number of words, an audio clip a variable number of samples, a patient record a variable number of visits, a stock price an ever-growing history. The *order* matters ("dog bites man" is not "man bites dog"), the length varies, and the relevant information may be separated by hundreds of steps.

This week is the story of how neural networks learned to handle such data. It begins with the recurrent neural network, an idea that is conceptually simple (apply the same function at every time step, carrying a memory forward) and mathematically revealing, because training it exposes the vanishing and exploding gradient problem in its purest form. We then follow the chain of fixes: gating (LSTM, GRU), reading in both directions, stacking, encoder-decoder models, and finally attention, the mechanism that lets a decoder look back at every input position directly. [Week 9](week-09.html) will take attention and remove the recurrence entirely, giving the Transformer. Understanding *why* people needed attention is the best way to understand why the Transformer looks the way it does.

If you have not already, skim [Week 2](week-02.html) (backpropagation as reverse-mode autodiff) and [Week 3](week-03.html#vanishing-exploding) (vanishing gradients and clipping); this week builds directly on both.

## Sequential data and the task zoo {#sequence-tasks}

A **sequence** is an ordered list $\boldsymbol{x}_1,\boldsymbol{x}_2,\dots,\boldsymbol{x}_T$ whose length $T$ may differ from example to example. Each element $\boldsymbol{x}_t$ might be a one-hot vector for a character, an index into a vocabulary of words, a frame of audio features, or a vector of sensor measurements. Three properties make sequences different from the fixed-size inputs of previous weeks:

1. **Variable length.** A model must accept $T=3$ and $T=3000$ with the same parameters.
2. **Order dependence.** Permuting the elements changes the meaning, so a model cannot treat the input as a bag.
3. **Long-range dependencies.** The right output at step $t$ may depend on something seen at step $t-200$: the subject of a sentence determines verb agreement many words later, a closing bracket must match an opening one far earlier.

Problems differ in how input and output sequences relate. It helps to name the common patterns, because each leads to a different wiring of the same recurrent building block.

{{figure: week07-seq-tasks | The four common wirings. Orange boxes are the same recurrent cell reused at every step; purple boxes in the last panel are a second network (the decoder) that starts from a summary of the first. The blue/green nodes show where inputs enter and outputs are read off.}}

- **One-to-many.** One input, a sequence out. Image captioning feeds one image embedding into a recurrent decoder that emits words. Music or text generation from a seed also fits.
- **Many-to-one.** A sequence in, a single label out. Sentiment classification, spam detection, classifying a time series (is this ECG abnormal?). You read the final hidden state, or pool over all of them.
- **Many-to-many, aligned.** One output per input step, same length. Part-of-speech tagging, named-entity recognition, frame-by-frame speech labeling, per-step forecasting.
- **Many-to-many, unaligned (sequence-to-sequence).** Input and output lengths differ and no step-by-step correspondence exists: machine translation, summarization, speech recognition. This needs the encoder-decoder design of [a later section](#seq2seq).

### Language modeling as the canonical sequence task {#language-modeling}

A **language model** assigns a probability to a sequence of tokens. Using the chain rule of probability (no approximation yet),

$$
p(w_1,\dots,w_T)=\prod_{t=1}^{T}p(w_t\mid w_1,\dots,w_{t-1}).
$$

Every factor is the probability of the next token given the whole past. An **autoregressive** model parameterizes exactly these conditionals with a network that reads the prefix $w_{<t}$ and outputs a distribution over the vocabulary of size $V$. Training is plain supervised learning: the input at step $t$ is $w_{t-1}$ (plus the hidden state that summarizes the older past), the target is $w_t$. Nobody has to label anything, which is why language modeling scales: any text is training data. This is the same objective that later powers GPT-style models ([Week 10](week-10.html)).

Before neural networks, the standard approach was the **$n$-gram model**, which truncates the history to the last $n-1$ tokens and estimates $p(w_t\mid w_{t-n+1:t-1})$ by counting. It fails in two ways. The table of counts grows as $V^{n}$, and most $n$-grams are never observed, so you need elaborate smoothing. And it cannot use context beyond $n-1$ tokens, however informative. A recurrent network replaces the explicit count table with a fixed-size learned state, so the number of parameters does not grow with the context length.

#### Cross-entropy, bits per character and perplexity

We train by maximizing the log-likelihood of the data, equivalently by minimizing the average cross-entropy over tokens:

$$
\mathcal{L}=-\frac{1}{T}\sum_{t=1}^{T}\log p_{\boldsymbol{\theta}}(w_t\mid w_{<t}).
$$

Reporting conventions differ by community. In nats (natural log) this is the loss PyTorch's `cross_entropy` returns. Character-level models are usually quoted in **bits per character** (BPC), the same quantity with $\log_2$, so $\text{BPC}=\mathcal{L}/\ln 2$. Word-level models are quoted as **perplexity**:

$$
\mathrm{PPL}=\exp(\mathcal{L})=\Big(\prod_{t=1}^T p(w_t\mid w_{<t})\Big)^{-1/T}.
$$

Perplexity has a clean interpretation: it is the *effective branching factor*. A model that is uniformly uncertain among $k$ equally likely next tokens has $\mathcal{L}=\ln k$ and perplexity exactly $k$. A model with perplexity 50 is, on average, as unsure as if it were choosing uniformly among 50 words. A uniform model over a vocabulary of $V$ words has perplexity $V$; that is the baseline you must beat, and a perfect model has perplexity 1.

:::warning Perplexities are only comparable under identical tokenization
A character-level model with 1.2 BPC and a word-level model with perplexity 100 cannot be compared directly, and two word-level models with different vocabularies or different handling of unknown words cannot either: the per-token loss is averaged over different units. Convert to a common unit (for example bits per byte of the raw text) before comparing. This matters again in [Week 8](week-08.html#subword-tokenization), where subword tokenizers change the number of tokens per sentence.
:::

Sanity check you should always run before training: with $V$ outputs and random initialization the initial loss should be close to $\ln V$ (for 65 characters, about 4.17 nats). If it is much larger your logits are too big; if the loss never leaves $\ln V$ your model is ignoring its input.

## The recurrent neural network {#rnn}

Here is the central idea. Instead of a function of the whole input, learn a function that consumes **one element at a time** and updates a fixed-size summary of everything seen so far, the **hidden state** $\boldsymbol{h}_t\in\R^{d}$. The same function (the same weights) is applied at every step.

The vanilla (Elman) recurrent network is

$$
\boldsymbol{h}_t=\phi\!\left(\mathbf{W}_{xh}\boldsymbol{x}_t+\mathbf{W}_{hh}\boldsymbol{h}_{t-1}+\boldsymbol{b}_h\right),\qquad
\boldsymbol{o}_t=\mathbf{W}_{hy}\boldsymbol{h}_t+\boldsymbol{b}_y,\qquad
\hat{\boldsymbol{y}}_t=\softmax(\boldsymbol{o}_t),
$$

with $\boldsymbol{x}_t\in\R^{m}$, $\mathbf{W}_{xh}\in\R^{d\times m}$, $\mathbf{W}_{hh}\in\R^{d\times d}$, $\mathbf{W}_{hy}\in\R^{V\times d}$, nonlinearity $\phi$ (usually $\tanh$) and initial state $\boldsymbol{h}_0$ (zeros, or learned). Read the first equation as: the new state is a squashed mixture of the current input and the previous state. The hidden state is the network's only memory.

{{figure: week07-rnn-unrolled | A vanilla RNN folded (left) and unrolled across four steps (right). Unrolling turns the recurrence into an ordinary deep feed-forward graph whose layers share weights. The depth equals the sequence length, which is the source of both the power and the trouble.}}

Two observations organize everything that follows.

**Parameter sharing across time.** The weights $\mathbf{W}_{xh},\mathbf{W}_{hh},\mathbf{W}_{hy}$ do not depend on $t$. This is the temporal analogue of the convolution's weight sharing in [Week 5](week-05.html): it encodes the assumption that the rule for updating memory is the same at every position, it lets the model handle any length, and it keeps the parameter count independent of $T$. A model that saw "the cat sat" at positions 1 to 3 automatically knows what to do with "the cat sat" at positions 51 to 53.

**Unrolled, an RNN is a very deep network.** A sequence of length 1000 produces a computation graph with 1000 layers, each using the same matrix $\mathbf{W}_{hh}$. [Week 3](week-03.html#vanishing-exploding) taught you what products of many matrices do to gradients. Recurrence is that problem with no escape hatch: you cannot choose a shallower network without ignoring part of the sequence.

### Why not just feed a window to an MLP? {#why-not-window}

A fixed-window MLP (concatenate the last $n$ embeddings and classify) is the neural version of the $n$-gram. It has the same defect: context size is frozen into the architecture, and the weights for position $t-3$ are unrelated to the weights for $t-2$, so nothing learned about "a noun two words back" transfers across positions. The RNN's state can, in principle, carry information from arbitrarily far back, and its weights are shared. Whether it *does* in practice is the subject of the next sections.

### Computing an RNN forward pass from scratch {#rnn-from-scratch}

The following NumPy implementation makes the shapes explicit. It runs a batch of $B$ sequences of length $T$ through a tanh RNN, and is the code we will differentiate by hand in the BPTT section.

```python:rnn_numpy.py
import numpy as np

rng = np.random.default_rng(0)
B, T, m, d, V = 4, 6, 10, 16, 10     # batch, steps, input dim, hidden dim, vocab/output dim

# Parameters (small random init; see the BPTT section for why scale matters)
Wxh = rng.normal(0, 1 / np.sqrt(m), (d, m))
Whh = rng.normal(0, 1 / np.sqrt(d), (d, d))
Why = rng.normal(0, 1 / np.sqrt(d), (V, d))
bh, by = np.zeros(d), np.zeros(V)

def rnn_forward(x, h0):
    """x: (B, T, m). Returns logits (B, T, V), hidden states (B, T+1, d)."""
    hs = [h0]
    logits = []
    for t in range(T):
        a = x[:, t] @ Wxh.T + hs[-1] @ Whh.T + bh   # (B, d) pre-activation
        h = np.tanh(a)                               # (B, d) new state
        hs.append(h)
        logits.append(h @ Why.T + by)                # (B, V) per-step output
    return np.stack(logits, axis=1), np.stack(hs, axis=1)

x = rng.normal(size=(B, T, m))
logits, hs = rnn_forward(x, np.zeros((B, d)))
print(logits.shape, hs.shape)        # (4, 6, 10) (4, 7, 16)
print(np.abs(hs).max())              # tanh keeps the state in (-1, 1)
```

The PyTorch equivalent is a single call. Note the default layout: `nn.RNN` expects `(T, B, m)` unless you pass `batch_first=True`, a classic source of silent bugs.

```python:rnn_torch.py
import torch
import torch.nn as nn

B, T, m, d = 4, 6, 10, 16
rnn = nn.RNN(input_size=m, hidden_size=d, num_layers=1, nonlinearity="tanh", batch_first=True)

x = torch.randn(B, T, m)               # (batch, time, features) because batch_first=True
h0 = torch.zeros(1, B, d)              # (num_layers * num_directions, batch, hidden)
out, h_n = rnn(x, h0)

print(out.shape)    # torch.Size([4, 6, 16])  hidden state at EVERY step
print(h_n.shape)    # torch.Size([1, 4, 16])  final state of each layer
# Sanity: the last output step equals the final hidden state
print(torch.allclose(out[:, -1], h_n[0]))   # True

# Parameter shapes: weight_ih = W_xh, weight_hh = W_hh, plus two bias vectors
for name, p in rnn.named_parameters():
    print(name, tuple(p.shape))
```

```text
weight_ih_l0 (16, 10)
weight_hh_l0 (16, 16)
bias_ih_l0 (16,)
bias_hh_l0 (16,)
```

PyTorch keeps two bias vectors ($\boldsymbol{b}_{ih}$ and $\boldsymbol{b}_{hh}$) where the equation above has one; they are redundant for the plain RNN and only matter for matching cuDNN's layout.

:::history Where recurrence came from
Recurrent connections were studied long before deep learning. Hopfield (1982) analysed symmetric recurrent networks as associative memories. Jeffrey Elman (1990, "Finding Structure in Time") trained a simple recurrent network with a hidden-state copy-back loop on small linguistic tasks and showed it learned word-class structure; that architecture is the "Elman RNN" above. Jordan networks (1986) fed the *output* back instead. The training algorithm, backpropagation through time, was described in the late 1980s (Werbos 1988 and Rumelhart, Hinton and Williams 1986 for the unrolling view). The vanishing gradient diagnosis arrived in 1991 to 1994 (Hochreiter's diploma thesis, then Bengio, Simard and Frasconi), and the LSTM (Hochreiter and Schmidhuber, 1997) was the answer.
:::

## Backpropagation through time {#bptt}

Training an RNN means computing $\partial\mathcal{L}/\partial\boldsymbol{\theta}$ for the shared parameters. Because unrolling gives an ordinary acyclic graph, you do not need a new algorithm: apply reverse-mode autodiff ([Week 2](week-02.html)) to the unrolled graph and sum the contributions from every time step where a weight was used. This is **backpropagation through time (BPTT)**. The derivation is worth doing in full, because the structure of the result explains the main pathology of recurrent networks.

### Setup and notation {#bptt-setup}

Use the pre-activation form. Let $\boldsymbol{a}_t=\mathbf{W}_{xh}\boldsymbol{x}_t+\mathbf{W}_{hh}\boldsymbol{h}_{t-1}+\boldsymbol{b}_h$ and $\boldsymbol{h}_t=\tanh(\boldsymbol{a}_t)$, with per-step loss $\mathcal{L}_t=\ell(\hat{\boldsymbol{y}}_t,y_t)$ and total loss $\mathcal{L}=\sum_{t=1}^{T}\mathcal{L}_t$ (sum or mean, a constant factor). Write $\boldsymbol{\delta}_t=\partial\mathcal{L}/\partial\boldsymbol{a}_t$ for the gradient at the pre-activation; it is the quantity that gets propagated backwards.

:::math Derivation: BPTT for a tanh RNN with softmax outputs
**Output layer.** For softmax cross-entropy the gradient of $\mathcal{L}_t$ with respect to the logits is $\boldsymbol{o}$-space error $\boldsymbol{e}_t=\hat{\boldsymbol{y}}_t-\boldsymbol{y}_t$ (predicted probabilities minus the one-hot target), exactly as in [Week 2](week-02.html). Therefore
$$
\frac{\partial\mathcal{L}}{\partial\mathbf{W}_{hy}}=\sum_{t=1}^T \boldsymbol{e}_t\boldsymbol{h}_t^\top,\qquad
\frac{\partial\mathcal{L}}{\partial\boldsymbol{h}_t}\Big|_{\text{direct}}=\mathbf{W}_{hy}^\top\boldsymbol{e}_t .
$$

**Hidden state: two routes.** The state $\boldsymbol{h}_t$ influences the loss in two ways: directly through its own output $\mathcal{L}_t$, and indirectly through the next state $\boldsymbol{h}_{t+1}$, which feeds every later loss. By the multivariate chain rule,
$$
\frac{\partial\mathcal{L}}{\partial\boldsymbol{h}_t}=\mathbf{W}_{hy}^\top\boldsymbol{e}_t+\left(\frac{\partial\boldsymbol{h}_{t+1}}{\partial\boldsymbol{h}_t}\right)^{\!\top}\frac{\partial\mathcal{L}}{\partial\boldsymbol{h}_{t+1}},
\qquad \frac{\partial\mathcal{L}}{\partial\boldsymbol{h}_{T}}=\mathbf{W}_{hy}^\top\boldsymbol{e}_T .
$$
Since $\boldsymbol{h}_{t+1}=\tanh(\mathbf{W}_{xh}\boldsymbol{x}_{t+1}+\mathbf{W}_{hh}\boldsymbol{h}_t+\boldsymbol{b}_h)$, the one-step Jacobian is
$$
\mathbf{J}_{t+1}\equiv\frac{\partial\boldsymbol{h}_{t+1}}{\partial\boldsymbol{h}_t}=\diag\!\big(1-\boldsymbol{h}_{t+1}^{\odot2}\big)\,\mathbf{W}_{hh}=\mathbf{D}_{t+1}\mathbf{W}_{hh},
$$
where $\mathbf{D}_{t+1}$ is diagonal with entries $\tanh'(a)=1-h^2\in(0,1]$.

**Backward recursion.** Multiplying through by the activation derivative gives a recursion for $\boldsymbol{\delta}_t$:
$$
\boldsymbol{\delta}_t=\mathbf{D}_t\!\left(\mathbf{W}_{hy}^\top\boldsymbol{e}_t+\mathbf{W}_{hh}^\top\boldsymbol{\delta}_{t+1}\right),\qquad \boldsymbol{\delta}_{T+1}=\mathbf{0}.
$$

**Parameter gradients.** Each weight is used at every step, so its gradient is a sum over time of the local outer products:
$$
\frac{\partial\mathcal{L}}{\partial\mathbf{W}_{hh}}=\sum_{t=1}^{T}\boldsymbol{\delta}_t\boldsymbol{h}_{t-1}^\top,\quad
\frac{\partial\mathcal{L}}{\partial\mathbf{W}_{xh}}=\sum_{t=1}^{T}\boldsymbol{\delta}_t\boldsymbol{x}_t^\top,\quad
\frac{\partial\mathcal{L}}{\partial\boldsymbol{b}_h}=\sum_{t=1}^{T}\boldsymbol{\delta}_t .
$$
:::

{{figure: week07-bptt-graph | The unrolled graph. Forward arrows (solid) carry states; the backward recursion (dashed red) sends gradient from later steps to earlier ones through the Jacobian $\mathbf{J}=\mathbf{D}\mathbf{W}_{hh}$. The gradient reaching $h_k$ is the sum of contributions from every loss $L_t$ with $t\ge k$.}}

Unfold the recursion. The contribution of the loss at step $t$ to the gradient with respect to an *earlier* state $\boldsymbol{h}_k$ ($k<t$) is

$$
\frac{\partial\mathcal{L}_t}{\partial\boldsymbol{h}_k}=\left(\frac{\partial\boldsymbol{h}_t}{\partial\boldsymbol{h}_k}\right)^{\!\top}\frac{\partial\mathcal{L}_t}{\partial\boldsymbol{h}_t},
\qquad
\frac{\partial\boldsymbol{h}_t}{\partial\boldsymbol{h}_k}=\prod_{j=k+1}^{t}\mathbf{J}_j=\prod_{j=k+1}^{t}\mathbf{D}_j\mathbf{W}_{hh}.
\tag{1}
$$

This product of $t-k$ Jacobians is the whole story. It is what lets the loss at step $t$ teach an input at step $k$ ("this early word should have been remembered"), and it is also exactly where learning breaks down.

### A NumPy BPTT implementation, with a gradient check {#bptt-code}

Implementing the recursion once, by hand, and verifying it numerically is the best way to make it concrete. The code below does a char-level RNN step with BPTT and checks one parameter's gradient against central finite differences.

```python:bptt_numpy.py
import numpy as np

rng = np.random.default_rng(1)
V, d, T = 5, 8, 7                       # vocab, hidden size, sequence length
Wxh = rng.normal(0, 0.3, (d, V)); Whh = rng.normal(0, 0.3, (d, d))
Why = rng.normal(0, 0.3, (V, d)); bh = np.zeros(d); by = np.zeros(V)
xs = rng.integers(0, V, T); ys = rng.integers(0, V, T)   # integer-coded input/target tokens

def loss_and_grads(Wxh, Whh, Why, bh, by, h0=None):
    h = np.zeros(d) if h0 is None else h0
    hs, ps, loss = {-1: h}, {}, 0.0
    for t in range(T):                                    # forward
        hs[t] = np.tanh(Wxh[:, xs[t]] + Whh @ hs[t - 1] + bh)   # one-hot x: column lookup
        o = Why @ hs[t] + by
        p = np.exp(o - o.max()); ps[t] = p / p.sum()
        loss += -np.log(ps[t][ys[t]])
    dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
    dbh, dby = np.zeros_like(bh), np.zeros_like(by)
    delta_next = np.zeros(d)                              # delta_{T+1} = 0
    for t in reversed(range(T)):                          # backward through time
        e = ps[t].copy(); e[ys[t]] -= 1                   # softmax-CE gradient wrt logits
        dWhy += np.outer(e, hs[t]); dby += e
        delta = (1 - hs[t] ** 2) * (Why.T @ e + Whh.T @ delta_next)   # recursion from the derivation
        dWhh += np.outer(delta, hs[t - 1]); dWxh[:, xs[t]] += delta; dbh += delta
        delta_next = delta
    return loss, (dWxh, dWhh, dWhy, dbh, dby)

loss, (dWxh, dWhh, dWhy, dbh, dby) = loss_and_grads(Wxh, Whh, Why, bh, by)

# Central finite differences on a few entries of Whh
eps = 1e-5
for (i, j) in [(0, 0), (3, 5), (7, 2)]:
    Wp = Whh.copy(); Wp[i, j] += eps
    Wm = Whh.copy(); Wm[i, j] -= eps
    num = (loss_and_grads(Wxh, Wp, Why, bh, by)[0] - loss_and_grads(Wxh, Wm, Why, bh, by)[0]) / (2 * eps)
    print(f"dWhh[{i},{j}]  analytic {dWhh[i, j]: .8f}  numeric {num: .8f}")
```

```text
dWhh[0,0]  analytic -0.14178676  numeric -0.14178676
dWhh[3,5]  analytic  0.15044806  numeric  0.15044806
dWhh[7,2]  analytic -0.02415699  numeric -0.02415699
```

(Analytic and numeric values must agree to about six places. If they do not, the bug is nearly always a missing term in `delta`, usually forgetting the `Whh.T @ delta_next` carry-over, or using `hs[t]` where `hs[t-1]` belongs in `dWhh`.)

## Vanishing and exploding gradients in time {#vanishing-exploding}

Look again at the product in equation (1). To understand its behavior, start with the simplest case: a **linear** RNN with $\phi=\mathrm{identity}$, where $\mathbf{J}_j=\mathbf{W}_{hh}$ for every $j$. Then $\partial\boldsymbol{h}_t/\partial\boldsymbol{h}_k=\mathbf{W}_{hh}^{\,t-k}$. Suppose $\mathbf{W}_{hh}$ is diagonalizable, $\mathbf{W}_{hh}=\mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^{-1}$. Then

$$
\mathbf{W}_{hh}^{\,n}=\mathbf{Q}\boldsymbol{\Lambda}^{n}\mathbf{Q}^{-1},
$$

and each eigen-direction is scaled by $\lambda_i^{\,n}$. If the largest eigenvalue magnitude, the **spectral radius** $\rho(\mathbf{W}_{hh})=\max_i|\lambda_i|$, is below 1 the product shrinks geometrically, like $\rho^{\,n}$; if it is above 1 it grows geometrically. A gradient that must travel $n=50$ steps back is multiplied by roughly $\rho^{50}$: with $\rho=0.9$ that is $0.005$, with $\rho=1.1$ it is $117$, with $\rho=0.5$ it is $9\times10^{-16}$.

:::math Bounding the gradient with the spectral norm
The spectral radius gives the asymptotic rate; a clean *bound* uses the induced 2-norm. Since $\|\mathbf{A}\mathbf{B}\|_2\le\|\mathbf{A}\|_2\|\mathbf{B}\|_2$ and the diagonal matrix $\mathbf{D}_j$ has $\|\mathbf{D}_j\|_2=\max_i|\phi'(a_{j,i})|\le\gamma$ ($\gamma=1$ for tanh, $\gamma=1/4$ for the sigmoid),
$$
\left\|\frac{\partial\boldsymbol{h}_t}{\partial\boldsymbol{h}_k}\right\|_2\le\prod_{j=k+1}^{t}\|\mathbf{D}_j\|_2\,\|\mathbf{W}_{hh}\|_2\le\left(\gamma\,\sigma_{\max}(\mathbf{W}_{hh})\right)^{t-k}.
$$
**Sufficient condition for vanishing:** if $\gamma\,\sigma_{\max}(\mathbf{W}_{hh})<1$, the influence of $\boldsymbol{h}_k$ on $\boldsymbol{h}_t$ decays at least exponentially in $t-k$. This is the Pascanu, Mikolov and Bengio (2013) argument. For tanh ($\gamma=1$) it means that a recurrent matrix with largest singular value below 1 *guarantees* vanishing gradients. The converse does not hold: $\sigma_{\max}>1$ does not guarantee explosion (the bound is an upper bound; the product can still shrink if the large singular directions do not align between steps), but $\rho(\mathbf{W}_{hh})>1$ is a *necessary* condition for exploding gradients in the linear case, and in the nonlinear case it gives growth along the directions where the activation is not saturated.
:::

Two caveats make the real picture richer than "$\rho<1$ vanishes, $\rho>1$ explodes".

1. **Saturation changes the picture.** With tanh, $\mathbf{D}_j$ has entries $\le 1$ and often much smaller: when a unit saturates ($|h|\approx1$), $1-h^2\approx0$ and that unit's gradient is killed. So a tanh RNN with $\rho$ moderately above 1 often *does not* explode, because saturation pulls the effective gain back down, as the measurements below show.
2. **Vanishing and exploding are two sides of one coin.** The product of $n$ random-ish matrices has singular values that spread out exponentially (a fact from random matrix theory: Lyapunov exponents). You almost never get a clean $\rho^n$; some directions blow up while others vanish. Practically, the network sits on the edge: it has a narrow band of weight scales where gradients are usable.

Let us look at measured data. We simulate a $64$-unit tanh RNN with $\mathbf{W}_{hh}=\rho\,\mathbf{Q}$ for a random orthogonal $\mathbf{Q}$ (so every singular value and every eigenvalue magnitude is exactly $\rho$), random Gaussian inputs, and compute the spectral norm of $\partial\boldsymbol{h}_T/\partial\boldsymbol{h}_{T-k}$ for $k$ up to 60, averaging $\log_{10}$ over ten random draws.

```chart
{"type": "line", "caption": "Measured log10 of the spectral norm of the Jacobian product dh_T/dh_(T-k) for a 64-unit tanh RNN with W = rho * (random orthogonal matrix), averaged over 10 random draws (NumPy simulation). rho = 0.9 collapses by eight orders of magnitude within 60 steps; rho = 2.5 explodes; rho = 1.5 hovers near 1 because tanh saturates.", "height": 340, "data": {"labels": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60], "datasets": [{"label": "rho = 0.9", "data": [-0.046, -0.119, -0.216, -0.324, -0.451, -0.568, -0.688, -0.828, -0.97, -1.104, -1.242, -1.39, -1.531, -1.663, -1.791, -1.93, -2.091, -2.239, -2.377, -2.511, -2.663, -2.807, -2.95, -3.101, -3.252, -3.386, -3.53, -3.684, -3.839, -3.982, -4.145, -4.29, -4.446, -4.597, -4.745, -4.904, -5.05, -5.198, -5.339, -5.489, -5.64, -5.787, -5.939, -6.092, -6.239, -6.383, -6.535, -6.698, -6.863, -7.028, -7.175, -7.334, -7.492, -7.642, -7.804, -7.958, -8.105, -8.265, -8.402, -8.521], "borderColor": "var(--c-blue)", "pointRadius": 0, "tension": 0.2}, {"label": "rho = 1.5", "data": [0.175, 0.283, 0.341, 0.382, 0.42, 0.434, 0.434, 0.465, 0.479, 0.48, 0.507, 0.5, 0.516, 0.533, 0.518, 0.519, 0.508, 0.507, 0.516, 0.503, 0.478, 0.454, 0.435, 0.42, 0.425, 0.421, 0.404, 0.405, 0.394, 0.387, 0.363, 0.343, 0.35, 0.345, 0.345, 0.333, 0.329, 0.317, 0.314, 0.285, 0.281, 0.279, 0.267, 0.259, 0.247, 0.243, 0.231, 0.228, 0.225, 0.221, 0.196, 0.19, 0.168, 0.128, 0.092, 0.086, 0.074, 0.088, 0.143, 0.255], "borderColor": "var(--c-green)", "pointRadius": 0, "tension": 0.2}, {"label": "rho = 2.5", "data": [0.396, 0.633, 0.762, 0.89, 1.025, 1.168, 1.292, 1.397, 1.532, 1.631, 1.72, 1.818, 1.882, 1.97, 2.113, 2.221, 2.347, 2.424, 2.507, 2.61, 2.717, 2.81, 2.917, 3.014, 3.106, 3.201, 3.282, 3.396, 3.472, 3.569, 3.685, 3.748, 3.832, 3.933, 3.996, 4.078, 4.127, 4.209, 4.307, 4.402, 4.534, 4.63, 4.716, 4.816, 4.892, 4.983, 5.054, 5.116, 5.224, 5.317, 5.423, 5.535, 5.631, 5.719, 5.804, 5.906, 5.986, 6.125, 6.322, 6.647], "borderColor": "var(--c-red)", "pointRadius": 0, "tension": 0.2}]}, "options": {"scales": {"x": {"title": {"display": true, "text": "steps back in time k"}}, "y": {"title": {"display": true, "text": "log10 || dh_T / dh_(T-k) ||"}}}}}
```

The data show all three regimes. At $\rho=0.9$ the sensitivity of the last state to a state 60 steps back is about $10^{-8.5}$: no amount of training signal will ever teach the network to use information from that far back. At $\rho=2.5$ it is about $10^{6.6}$, large enough to overflow quickly. And at $\rho=1.5$, a value where a *linear* network would explode, the tanh saturates and the norm stays near 1, a narrow "edge of chaos" band. You can explore this interactively.

{{demo: week07-vanishing-gradient | Vanishing and exploding gradient explorer | 460}}

:::intuition Why long dependencies are structurally hard
Think of the gradient as a message sent backward through $n$ relays. Each relay multiplies the message by a gain. A gain slightly below 1 at every relay erases the message; slightly above 1 amplifies the message and the noise alike. Training only needs a *sum* of many such messages with different path lengths, so the short paths (a few steps) dominate the sum when gains are below 1. The network learns easily from recent context and effectively cannot learn from distant context. Notice that this is a learning problem, not a representational one: an RNN *can* store information for hundreds of steps (with the right weights, e.g. $\mathbf{W}_{hh}=\mathbf{I}$ copies the state forever), but gradient descent from a standard initialization cannot find those weights.
:::

### Remedies, short of changing the architecture {#remedies}

Several tricks mitigate the problem without gating:

- **Gradient clipping** for the exploding side. Rescale the gradient when its norm exceeds a threshold $c$: $\boldsymbol{g}\leftarrow\boldsymbol{g}\cdot\min(1,c/\|\boldsymbol{g}\|)$. It keeps the direction and limits the step. This is cheap, almost universally used for RNNs, and was popularized for recurrent nets by Pascanu et al. (2013). It does **nothing** for vanishing gradients.
- **Careful initialization.** Initialize $\mathbf{W}_{hh}$ as an orthogonal matrix (all singular values exactly 1, `nn.init.orthogonal_`), or as the identity (the "IRNN" of Le, Jaitly and Hinton, 2015, with ReLU units). This puts the network at $\rho=1$ at the start, at the cost of no longer being stable once trained.
- **Regularize the gradient norm or the spectral norm** (soft constraints on $\mathbf{W}_{hh}$), or constrain it exactly to the orthogonal group (unitary RNNs, Arjovsky, Shah and Bengio, 2016). These help on synthetic memory tasks but have not displaced gating in practice.
- **Skip connections through time and ReLU-like activations.** The idea of giving gradients an additive route is exactly what the LSTM does, next.

#### A measured example of clipping {#clipping-measured}

To see clipping in action we trained the NumPy RNN from the BPTT section (48 hidden units, sequence chunks of 25 characters, learning rate 3.0, plain SGD) on a tiny repetitive corpus, twice with identical initialization: once unclipped and once with the global gradient norm clipped to 5. The learning rate is deliberately aggressive, which is when exploding gradients show up.

```chart
{"type": "line", "caption": "Training loss (mean per character, in nats, averaged over blocks of 10 SGD steps) for the same NumPy RNN with and without gradient-norm clipping at 5. The unclipped run suffers gradient norms above 10^5 (a spike of loss to roughly 60 nats, far above the ln(V) = 3.3 baseline) and stays at a high loss; the clipped run trains stably. Toy corpus, one seed; the qualitative picture, not the exact values, is the point.", "height": 340, "data": {"labels": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300], "datasets": [{"label": "no clipping", "data": [5.075, 33.102, 44.133, 45.733, 49.007, 57.739, 65.584, 59.858, 61.959, 18.304, 11.022, 13.999, 16.395, 15.474, 14.669, 15.924, 16.736, 14.929, 13.599, 12.531, 11.039, 9.799, 10.287, 9.228, 10.374, 10.374, 12.852, 11.645, 10.784, 10.13], "borderColor": "var(--c-red)", "pointRadius": 0, "tension": 0.2}, {"label": "clip norm = 5", "data": [2.822, 2.11, 1.485, 1.0, 0.644, 2.677, 2.69, 1.356, 1.245, 1.096, 0.383, 0.125, 0.038, 0.017, 2.704, 0.935, 0.14, 0.771, 0.249, 0.03, 0.009, 0.007, 0.032, 1.51, 0.029, 0.007, 0.442, 0.017, 0.006, 0.004], "borderColor": "var(--c-blue)", "pointRadius": 0, "tension": 0.2}]}, "options": {"scales": {"x": {"title": {"display": true, "text": "SGD step"}}, "y": {"type": "logarithmic", "title": {"display": true, "text": "loss per character (nats, log scale)"}}}}}
```

Implementing clipping in PyTorch takes one line between `backward()` and `step()`. The function returns the *pre-clipping* norm, which is a useful quantity to log: a sudden spike in it is the earliest warning that training is about to diverge.

```python:clip_torch.py
import torch
import torch.nn as nn

model = nn.GRU(input_size=32, hidden_size=64, batch_first=True)
head = nn.Linear(64, 10)
params = list(model.parameters()) + list(head.parameters())
opt = torch.optim.SGD(params, lr=0.1)

x = torch.randn(8, 100, 32)
y = torch.randint(0, 10, (8, 100))

out, _ = model(x)                              # (8, 100, 64)
loss = nn.functional.cross_entropy(head(out).reshape(-1, 10), y.reshape(-1))
opt.zero_grad()
loss.backward()
# Clip the GLOBAL norm of all gradients together (keeps the update direction)
total_norm = torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
print(f"grad norm before clipping: {total_norm:.3f}")
opt.step()
```

### Truncated BPTT {#truncated-bptt}

Full BPTT on a long sequence (a book of a million characters) is impossible in memory: it must store every intermediate state until the backward pass. **Truncated BPTT** (TBPTT) cuts the sequence into chunks of length $K$ and treats each chunk as a separate training example, with one crucial twist: the hidden state at the end of chunk $j$ is *carried forward* as the initial state of chunk $j+1$, but detached from the graph, so no gradient flows across the chunk boundary.

```text
chunk 1:  x_1 ... x_K          h_0 = 0          loss, backward within chunk
chunk 2:  x_{K+1} ... x_{2K}   h_0 = h_K.detach()   loss, backward within chunk
chunk 3:  ...
```

What this does and does not do:

- The forward pass still *uses* information from before the chunk (the carried state), so the model can in principle exploit long context at inference time.
- The backward pass cannot assign credit across the boundary, so dependencies longer than $K$ steps are never *learned* from gradients. $K$ is a hard cap on the credit-assignment horizon. Typical values for language modeling are $K=35$ to $200$.
- Memory and time per update scale with $K$ rather than with the sequence length, and the batch of chunks must be arranged so that row $b$ of consecutive minibatches continues the same stream (otherwise the carried state is meaningless).

In PyTorch the idiom is `h = h.detach()` after every optimizer step (for an LSTM, detach both $h$ and $c$). Forgetting it has two symptoms: memory grows without bound until you hit an out-of-memory error, and PyTorch raises "Trying to backward through the graph a second time". We will use it in the lab.

## LSTM: long short-term memory {#lstm}

The diagnosis in the previous section suggests the cure. The trouble is that the state is rewritten *multiplicatively* by a matrix and squashed at every step, so information and gradient both pass through a long chain of transformations. What if the memory had a route that was updated *additively*, with a gain near 1 by default? Hochreiter and Schmidhuber (1997) proposed exactly this, calling it the **constant error carousel**, and wrapped it in learned gates that decide what to write, keep and read.

The LSTM keeps two vectors per layer: the **cell state** $\boldsymbol{c}_t\in\R^d$ (long-term memory, the conveyor belt in the figure) and the **hidden state** $\boldsymbol{h}_t\in\R^d$ (the short-term output that other layers see). Let $\sigma$ be the logistic sigmoid and $[\boldsymbol{h}_{t-1};\boldsymbol{x}_t]\in\R^{d+m}$ the concatenation.

$$
\begin{aligned}
\boldsymbol{f}_t&=\sigma\big(\mathbf{W}_f[\boldsymbol{h}_{t-1};\boldsymbol{x}_t]+\boldsymbol{b}_f\big) &&\text{forget gate}\\
\boldsymbol{i}_t&=\sigma\big(\mathbf{W}_i[\boldsymbol{h}_{t-1};\boldsymbol{x}_t]+\boldsymbol{b}_i\big) &&\text{input gate}\\
\tilde{\boldsymbol{c}}_t&=\tanh\big(\mathbf{W}_g[\boldsymbol{h}_{t-1};\boldsymbol{x}_t]+\boldsymbol{b}_g\big) &&\text{candidate memory}\\
\boldsymbol{c}_t&=\boldsymbol{f}_t\odot\boldsymbol{c}_{t-1}+\boldsymbol{i}_t\odot\tilde{\boldsymbol{c}}_t &&\text{cell update}\\
\boldsymbol{o}_t&=\sigma\big(\mathbf{W}_o[\boldsymbol{h}_{t-1};\boldsymbol{x}_t]+\boldsymbol{b}_o\big) &&\text{output gate}\\
\boldsymbol{h}_t&=\boldsymbol{o}_t\odot\tanh(\boldsymbol{c}_t) &&\text{hidden state}
\end{aligned}
$$

{{figure: week07-lstm-cell | The LSTM cell. The thick horizontal line is the cell state. The forget gate multiplies it (erase), the input gate times the candidate is added to it (write), and the output gate controls how much of the squashed cell is exposed as $h_t$ (read). All gates see the previous hidden state and the current input.}}

### Every gate, explained {#lstm-gates}

**Gates are soft switches.** A sigmoid output lies in $(0,1)$ and is applied elementwise, so each coordinate of the memory gets its own dial between "completely closed" (0) and "completely open" (1). Gates are *learned functions of the context*, not fixed numbers: the network decides at every step, per unit, how much to erase and write.

- **Forget gate $\boldsymbol{f}_t$** (erase). Multiplies the old cell state. $f=1$ keeps a memory cell unchanged, $f=0$ wipes it. It lets the network reset memory at natural boundaries (a new sentence, a closed bracket). It was *not* in the original 1997 LSTM, which had a constant carry of 1 and so could never erase; Gers, Schmidhuber and Cummins (2000) added it, and it turned out to be the most important gate (Greff et al. 2017, "LSTM: a search space odyssey", found forget gate and output activation to be the critical components).
- **Input gate $\boldsymbol{i}_t$ and candidate $\tilde{\boldsymbol{c}}_t$** (write). The candidate proposes new content in $(-1,1)$; the input gate decides how much of it to admit. Separating "what to write" from "whether to write" lets the model ignore irrelevant inputs: $i\approx0$ makes the cell blind to the current token.
- **Cell state $\boldsymbol{c}_t$** (memory). The additive update $\boldsymbol{c}_t=\boldsymbol{f}_t\odot\boldsymbol{c}_{t-1}+\boldsymbol{i}_t\odot\tilde{\boldsymbol{c}}_t$ is the heart of the design. The old memory is *scaled and added to*, never pushed through a weight matrix and a squashing nonlinearity.
- **Output gate $\boldsymbol{o}_t$** (read). The cell may store information that is not needed right now. The output gate chooses which coordinates of $\tanh(\boldsymbol{c}_t)$ to expose to the rest of the network and to the next step's gates. The $\tanh$ bounds the exposed value even when the cell grows large.

Because everything is elementwise except the four matrix products, you can implement all four affine maps as **one** matrix multiplication of size $4d\times(d+m)$ and split the result. That is what `nn.LSTM` does; the parameter count of a single layer is $4\,(d\,m+d^2+2d)$ in PyTorch's layout (input weights, recurrent weights, two bias vectors).

### Why gating fixes (most of) the gradient problem {#lstm-gradient}

Differentiate the cell recursion with respect to the previous cell state. Holding the gates fixed (they also depend on $\boldsymbol{h}_{t-1}$, which we will come back to),

$$
\frac{\partial\boldsymbol{c}_t}{\partial\boldsymbol{c}_{t-1}}=\diag(\boldsymbol{f}_t)+\text{(terms through the gates)} .
$$

Compare with the vanilla RNN Jacobian $\mathbf{D}_t\mathbf{W}_{hh}$, a dense matrix that rotates and squashes. The LSTM's *direct* path is **diagonal** with entries $f_{t,i}\in(0,1)$ that the network controls. Over $n$ steps the direct path contributes

$$
\frac{\partial\boldsymbol{c}_t}{\partial\boldsymbol{c}_{t-n}}\approx\diag\Big(\prod_{j=t-n+1}^{t}\boldsymbol{f}_j\Big).
$$

If a unit learns to keep $f_{j,i}\approx1$ over the stretch where a dependency must be remembered, the gradient along that path neither shrinks nor grows: the error "rides the carousel". There is no weight matrix in this path, so there is no spectral radius to get wrong, and no activation derivative (the cell state is *not* squashed on the recurrent path). Information and gradient both have a highway.

:::intuition The gate is a learned per-unit decay rate
For a constant forget value $f$, a memory decays as $f^{k}$ after $k$ steps, a "half-life" of $\ln0.5/\ln f$ steps. With $f=0.9$ the half-life is $6.6$ steps; with $f=0.99$ it is $69$ steps. The LSTM does not remove the exponential: it lets the network choose the base per unit and per time step, so different units can have different time scales, and the base can be set to almost exactly 1 when the content must be preserved.
:::

The retention curves below show how sharply the choice of base matters. They are the direct-path gradient $f^k$ for constant $f=\sigma(b)$ with several forget-gate biases.

```chart
{"type": "line", "caption": "Direct-path gradient (and memory retention) f^k after k steps for a constant forget gate f = sigmoid(b). Computed exactly. With bias 0 the memory is gone in about 7 steps; with bias 4 it still retains 40% after 50 steps.", "height": 320, "data": {"labels": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50], "datasets": [{"label": "bias 0 (f = 0.500)", "data": [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.01562, 0.00781, 0.00391, 0.00195, 0.00098, 0.00049, 0.00024, 0.00012, 6e-05, 3e-05, 2e-05, 1e-05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "borderColor": "var(--c-red)", "pointRadius": 0, "tension": 0.2}, {"label": "bias 1 (f = 0.731)", "data": [1.0, 0.73106, 0.53445, 0.39071, 0.28563, 0.20881, 0.15266, 0.1116, 0.08159, 0.05964, 0.0436, 0.03188, 0.0233, 0.01704, 0.01245, 0.00911, 0.00666, 0.00487, 0.00356, 0.0026, 0.0019, 0.00139, 0.00102, 0.00074, 0.00054, 0.0004, 0.00029, 0.00021, 0.00016, 0.00011, 8e-05, 6e-05, 4e-05, 3e-05, 2e-05, 2e-05, 1e-05, 1e-05, 1e-05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "borderColor": "var(--c-amber)", "pointRadius": 0, "tension": 0.2}, {"label": "bias 2 (f = 0.881)", "data": [1.0, 0.8808, 0.7758, 0.68333, 0.60187, 0.53013, 0.46693, 0.41127, 0.36225, 0.31907, 0.28103, 0.24753, 0.21803, 0.19204, 0.16915, 0.14898, 0.13122, 0.11558, 0.1018, 0.08967, 0.07898, 0.06957, 0.06127, 0.05397, 0.04754, 0.04187, 0.03688, 0.03248, 0.02861, 0.0252, 0.0222, 0.01955, 0.01722, 0.01517, 0.01336, 0.01177, 0.01036, 0.00913, 0.00804, 0.00708, 0.00624, 0.00549, 0.00484, 0.00426, 0.00375, 0.00331, 0.00291, 0.00257, 0.00226, 0.00199, 0.00175], "borderColor": "var(--c-blue)", "pointRadius": 0, "tension": 0.2}, {"label": "bias 4 (f = 0.982)", "data": [1.0, 0.98201, 0.96435, 0.94701, 0.92997, 0.91325, 0.89682, 0.88069, 0.86485, 0.84929, 0.83402, 0.81902, 0.80429, 0.78982, 0.77562, 0.76166, 0.74797, 0.73451, 0.7213, 0.70833, 0.69559, 0.68308, 0.67079, 0.65873, 0.64688, 0.63524, 0.62382, 0.6126, 0.60158, 0.59076, 0.58013, 0.5697, 0.55945, 0.54939, 0.53951, 0.5298, 0.52028, 0.51092, 0.50173, 0.4927, 0.48384, 0.47514, 0.46659, 0.4582, 0.44996, 0.44187, 0.43392, 0.42611, 0.41845, 0.41092, 0.40353], "borderColor": "var(--c-green)", "pointRadius": 0, "tension": 0.2}]}, "options": {"scales": {"x": {"title": {"display": true, "text": "steps k"}}, "y": {"title": {"display": true, "text": "f^k"}, "min": 0, "max": 1}}}}
```

:::warning What the "constant error carousel" does not promise
The diagonal direct path is only part of the Jacobian. The gates and candidate depend on $\boldsymbol{h}_{t-1}$, and $\boldsymbol{h}_{t-1}=\boldsymbol{o}_{t-1}\odot\tanh(\boldsymbol{c}_{t-1})$ depends on $\boldsymbol{c}_{t-1}$, so there are additional paths through weight matrices; those paths can still vanish or explode, and gradient clipping remains standard for LSTMs. What the architecture guarantees is that a *good path exists* and that learning can find it by raising $f$. LSTMs mitigate vanishing gradients; they do not abolish them, and in practice a vanilla LSTM still struggles to use context beyond a few hundred steps, which is part of why attention won.
:::

### Initialization: the forget-gate bias trick {#forget-bias}

At the default initialization, $f_t\approx\sigma(0)=0.5$: the memory halves every step, so at initialization the LSTM has *the same vanishing-gradient problem as the vanilla RNN*, and long-range structure is learned only after the network has learned to open the forget gate. The remedy, advocated by Gers et al. (2000) and tested at scale by Jozefowicz, Zaremba and Sutskever (2015), is to initialize the forget-gate bias to a positive value (typically 1, sometimes 2 to 5 for very long dependencies) so that gates start mostly open. PyTorch does not do this for you; set it explicitly:

```python:lstm_forget_bias.py
import torch.nn as nn

lstm = nn.LSTM(input_size=32, hidden_size=64, num_layers=2, batch_first=True)
d = lstm.hidden_size
for name, p in lstm.named_parameters():
    if "bias" in name:
        # PyTorch stacks the gates in the order (input, forget, cell, output):
        # bias vector has length 4*d; the forget gate occupies [d : 2d].
        p.data.zero_()
        p.data[d:2 * d] = 1.0      # note: bias_ih and bias_hh are SUMMED, so total forget bias = 2.0 here
print(lstm.bias_ih_l0[d:2 * d].sum().item())   # 64.0
```

Both `bias_ih` and `bias_hh` contribute to the pre-activation, so setting both to 1 gives an effective forget bias of 2. That is fine, but know it is happening.

### Gate step-through {#lstm-demo}

The demo runs a tiny four-unit LSTM with fixed random weights on an input sequence of your choice. Step through time and watch which gates are open, and how the cell states respond to a pulse. Then vary the forget-gate bias.

{{demo: week07-lstm-gates | LSTM gate step-through | 560}}

### LSTM in PyTorch: shapes and the state tuple {#lstm-torch}

`nn.LSTM` returns the per-step outputs and a *tuple* `(h_n, c_n)` of final states; both have shape `(num_layers * num_directions, B, d)`.

```python:lstm_shapes.py
import torch
import torch.nn as nn

B, T, m, d, L = 4, 12, 10, 16, 2
lstm = nn.LSTM(m, d, num_layers=L, batch_first=True, dropout=0.1)   # dropout only BETWEEN layers
x = torch.randn(B, T, m)
out, (h_n, c_n) = lstm(x)               # states default to zeros
print(out.shape)                         # (4, 12, 16): top-layer hidden state at every step
print(h_n.shape, c_n.shape)              # (2, 4, 16) each: final state of each layer

# Verify the cell recursion by hand with a single-layer LSTM at step 0
one = nn.LSTM(m, d, batch_first=True)
out1, _ = one(x)
b = one.bias_ih_l0 + one.bias_hh_l0
z = x[:, 0] @ one.weight_ih_l0.T + b          # h_{-1} = 0, so the W_hh term vanishes
i, f, g, o = z.chunk(4, dim=1)                # PyTorch gate order: input, forget, cell(g), output
c1 = torch.sigmoid(i) * torch.tanh(g)         # c_{-1} = 0, so the forget gate is irrelevant at t=0
h1 = torch.sigmoid(o) * torch.tanh(c1)
print(torch.allclose(h1, out1[:, 0], atol=1e-6))   # True
```

## GRU: gated recurrent unit {#gru}

Cho et al. (2014) introduced a lighter gated cell while building an encoder-decoder for translation. The **GRU** merges the cell and hidden state into one vector and uses two gates instead of three:

$$
\begin{aligned}
\boldsymbol{z}_t&=\sigma\big(\mathbf{W}_z[\boldsymbol{h}_{t-1};\boldsymbol{x}_t]+\boldsymbol{b}_z\big) &&\text{update gate}\\
\boldsymbol{r}_t&=\sigma\big(\mathbf{W}_r[\boldsymbol{h}_{t-1};\boldsymbol{x}_t]+\boldsymbol{b}_r\big) &&\text{reset gate}\\
\tilde{\boldsymbol{h}}_t&=\tanh\big(\mathbf{W}_h[\boldsymbol{r}_t\odot\boldsymbol{h}_{t-1};\boldsymbol{x}_t]+\boldsymbol{b}_h\big) &&\text{candidate}\\
\boldsymbol{h}_t&=(1-\boldsymbol{z}_t)\odot\boldsymbol{h}_{t-1}+\boldsymbol{z}_t\odot\tilde{\boldsymbol{h}}_t &&\text{interpolation}
\end{aligned}
$$

{{figure: week07-gru-cell | The GRU cell. The update gate $z$ interpolates between keeping the old state and adopting the candidate; the reset gate $r$ controls how much of the old state is visible when proposing the candidate.}}

- **Update gate $\boldsymbol{z}_t$** plays the roles of both the LSTM forget and input gates, tied together: the amount of old state kept is $1-z$ and the amount of new content written is $z$ (a convex combination, so the state stays bounded). When $z\approx0$ the unit copies $\boldsymbol{h}_{t-1}$ forward unchanged, the same gradient highway as the LSTM, with $\partial\boldsymbol{h}_t/\partial\boldsymbol{h}_{t-1}\supseteq\diag(1-\boldsymbol{z}_t)$.
- **Reset gate $\boldsymbol{r}_t$** controls how much of the previous state is used to *compute the candidate*. With $r\approx0$ the candidate depends only on the current input, as if the unit were starting a new segment, which is useful for short-term dependencies. Without it, the candidate would always be contaminated by old state.

Be aware of a convention clash: some references (and the original paper) write $\boldsymbol{h}_t=\boldsymbol{z}_t\odot\boldsymbol{h}_{t-1}+(1-\boldsymbol{z}_t)\odot\tilde{\boldsymbol{h}}_t$, swapping the roles of $z$ and $1-z$. PyTorch follows the latter form and additionally applies the reset gate *after* the recurrent matrix product, $\tilde{\boldsymbol{h}}_t=\tanh(\mathbf{W}_{in}\boldsymbol{x}_t+\boldsymbol{b}_{in}+\boldsymbol{r}_t\odot(\mathbf{W}_{hn}\boldsymbol{h}_{t-1}+\boldsymbol{b}_{hn}))$. These variants are functionally equivalent in expressive power; just don't mix equations from different sources when debugging.

### LSTM versus GRU {#lstm-vs-gru}

| | LSTM | GRU |
|---|---|---|
| State | $(\boldsymbol{h}_t,\boldsymbol{c}_t)$ | $\boldsymbol{h}_t$ only |
| Gates | forget, input, output (+ candidate) | update, reset (+ candidate) |
| Parameters per layer | $4(dm+d^2+2d)$ in PyTorch | $3(dm+d^2+2d)$ in PyTorch |
| Output exposed | $o\odot\tanh(c)$, can hide memory | full state $h$ |
| Typical result | slightly better on tasks needing precise counting or long memory | similar accuracy, about 25% fewer parameters, a little faster |

Chung, Gulcehre, Cho and Bengio (2014) compared both and found them similar overall, with both clearly better than the plain tanh RNN. Greff et al. (2017) and Jozefowicz et al. (2015) reached the same conclusion: there is no universal winner, so try both on your task and choose with a validation set. Cost per step is dominated by the matrix multiplications, which cuDNN fuses, so wall-clock speed differences are modest.

## Bidirectional and stacked recurrent networks {#bi-stacked}

{{figure: week07-bi-stacked | Left: a bidirectional RNN runs one cell left to right and another right to left, and concatenates the two states at every position. Right: a stacked RNN feeds the hidden sequence of one layer as the input sequence to the next.}}

**Bidirectional RNNs** (Schuster and Paliwal, 1997). A unidirectional RNN at position $t$ has only seen $\boldsymbol{x}_{1:t}$. Many tasks, such as tagging a word as a person's name, depend on the words *after* it too ("Washington was founded in..." versus "Washington said..."). A bidirectional RNN runs two independent recurrent networks, a forward one producing $\overrightarrow{\boldsymbol{h}}_t$ from $\boldsymbol{x}_{1:t}$ and a backward one producing $\overleftarrow{\boldsymbol{h}}_t$ from $\boldsymbol{x}_{T:t}$, and concatenates $\boldsymbol{h}_t=[\overrightarrow{\boldsymbol{h}}_t;\overleftarrow{\boldsymbol{h}}_t]\in\R^{2d}$. Each position now has a summary of the full context. The cost is twice the parameters and compute.

The restriction to remember: a bidirectional model needs the **whole input** in advance. It is perfect for encoders (tagging, classification, the encoder of a translation model, the acoustic model in offline speech recognition) and *impossible* for autoregressive generation, where the future does not exist yet. Using a bidirectional layer in a next-token language model leaks the answer: the backward state at position $t$ has already seen $w_{t+1}$, the very token you are asked to predict, and your validation loss would look miraculous while the model is useless. This is the same issue that motivates masked language modeling in BERT ([Week 10](week-10.html)).

**Stacked (deep) RNNs.** The recurrence is already deep in time, but each step is a single shallow transformation. Stacking $L$ layers ($\boldsymbol{h}^{(\ell)}_t=\text{cell}^{(\ell)}(\boldsymbol{h}^{(\ell-1)}_t,\boldsymbol{h}^{(\ell)}_{t-1})$) adds depth *in the feature direction*, giving the model room for hierarchical processing: lower layers tend to capture local, syntactic structure and upper layers more abstract features. Two to four layers is typical for LSTM models (Google's 2016 neural translation system used eight, which required residual connections between layers to train). Beyond a couple of layers, add residual connections and dropout between layers (not on the recurrent connection, see the mistakes section).

In PyTorch both are arguments: `nn.LSTM(m, d, num_layers=2, bidirectional=True, batch_first=True)`. The output has shape `(B, T, 2*d)`, and `h_n` has shape `(num_layers*2, B, d)`, ordered layer by layer with the forward direction first. To get the final forward and backward states of the top layer you take `h_n[-2]` and `h_n[-1]`.

```python:bilstm_tagger.py
import torch
import torch.nn as nn

class BiLSTMTagger(nn.Module):
    """Per-token classifier: embeds tokens, runs a 2-layer BiLSTM, projects each position to tag logits."""
    def __init__(self, vocab_size, n_tags, emb=64, hid=128):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=0)
        self.rnn = nn.LSTM(emb, hid, num_layers=2, bidirectional=True, batch_first=True, dropout=0.3)
        self.out = nn.Linear(2 * hid, n_tags)

    def forward(self, tokens, lengths):
        x = self.emb(tokens)                                           # (B, T, emb)
        # Pack so the BACKWARD direction starts at each sequence's true last token, not at padding
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.rnn(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=tokens.size(1))
        return self.out(out)                                           # (B, T, n_tags)

model = BiLSTMTagger(vocab_size=1000, n_tags=9)
tokens = torch.randint(1, 1000, (3, 7)); tokens[1, 5:] = 0; tokens[2, 3:] = 0     # 0 = padding
lengths = torch.tensor([7, 5, 3])
logits = model(tokens, lengths)
print(logits.shape)    # torch.Size([3, 7, 9])
```

Padding deserves emphasis. If you batch sequences of different lengths by padding and run a bidirectional RNN naively, the backward direction starts by reading padding tokens, corrupting its state. `pack_padded_sequence` (or masking the loss and carefully gathering the last real state) fixes this. The rule: *never* take `out[:, -1]` as "the last hidden state" of a padded batch; index by `lengths - 1`, or use `h_n`.

## Sequence-to-sequence learning {#seq2seq}

Machine translation breaks the aligned many-to-many assumption: "je suis étudiant" has three tokens, "I am a student" has four, and the word order may differ. The **encoder-decoder** (or **seq2seq**) architecture, introduced independently by Cho et al. (2014) and Sutskever, Vinyals and Le (2014), solves this by factoring the problem into two networks:

1. The **encoder** RNN reads the source $\boldsymbol{x}_{1:S}$ and compresses it into a representation (in the simplest version, its final hidden state $\boldsymbol{h}_S$, the "context vector").
2. The **decoder** RNN is an autoregressive language model over the target, conditioned on that context: $p(y_{1:T}\mid\boldsymbol{x})=\prod_{t=1}^{T}p(y_t\mid y_{<t},\boldsymbol{x})$. Its initial state is set from the encoder's summary.

{{figure: week07-seq2seq | A sequence-to-sequence model. The encoder reads the source and passes its final state to the decoder, which generates the target token by token. Training feeds the gold previous token (teacher forcing); at test time the decoder feeds back its own predictions.}}

Special tokens make this work. A `<sos>` (start) token seeds the decoder, and `<eos>` (end) is a vocabulary item the decoder learns to emit when it is done; that is how a variable-length output is produced. Training minimizes the cross-entropy summed over target positions, backpropagating through the decoder, into the encoder, and into the embeddings. Sutskever et al. (2014) used a 4-layer deep LSTM, and reported a striking trick: reversing the source sentence improved results, because it shortens the distance between the first source words and the first target words and hands the optimization many short-term dependencies.

### The information bottleneck {#bottleneck}

The weakness is visible in the figure: *everything* the decoder knows about the source must pass through one fixed-size vector. Whether the source has 5 words or 50, the context is $\R^d$ with the same $d$. Cho et al. (2014, "On the properties of neural machine translation: encoder-decoder approaches") showed empirically that translation quality dropped sharply as the sentence length grew past those seen in training. This bottleneck is the problem attention solves, in the next-but-one section.

### Teacher forcing and exposure bias {#teacher-forcing}

How do we feed the decoder during training? Two options at step $t$:

- **Teacher forcing**: input the *ground-truth* previous token $y_{t-1}$ regardless of what the model would have predicted.
- **Free running**: input the model's own previous prediction $\hat y_{t-1}$.

Teacher forcing is the default, for good reasons. It makes every time step's loss independent of earlier *predictions*, so the whole target sequence can be processed in parallel for non-recurrent decoders, gradients are stable, and early training does not drown in garbage inputs. It also corresponds to maximizing the true likelihood $\prod p(y_t\mid y_{<t}^{\text{gold}},\boldsymbol{x})$.

The price is **exposure bias** (term from Ranzato et al., 2016): the model is trained only on gold prefixes, but at test time it conditions on its own, possibly erroneous, prefixes. One early mistake produces a prefix the model never saw in training, which raises the chance of a second mistake, and errors compound along the sequence. Train-test mismatch of this kind is a distribution shift of the decoder's inputs.

Mitigations, from simple to sophisticated:

- **Scheduled sampling** (Bengio, Vinyals, Jaitly and Shazeer, 2015): with probability $\epsilon_k$ feed the model's own prediction instead of the gold token, and anneal $\epsilon$ from 0 (pure teacher forcing) upward over training. It is easy to implement but optimizes a slightly inconsistent objective.
- **Sequence-level training** such as MIXER (Ranzato et al., 2016) or minimum risk training, which optimizes a sequence metric like BLEU via policy-gradient or sampling estimates; this is a precursor to RLHF-style ideas in [Week 15](week-12.html).
- **Better decoding** (beam search, next section) and **larger models**, which reduce the rate of first mistakes. In practice, modern Transformers trained with teacher forcing work extremely well, which suggests exposure bias is a manageable rather than fatal issue.

```python:teacher_forcing.py
import random
import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, vocab, emb, hid):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb)
        self.cell = nn.GRUCell(emb, hid)
        self.out = nn.Linear(hid, vocab)

    def forward(self, h, targets, sos_id, tf_ratio=1.0):
        """h: (B, hid) initial state from the encoder; targets: (B, T) gold tokens (incl. <eos>).
        tf_ratio = probability of feeding the gold token (1.0 = pure teacher forcing)."""
        B, T = targets.shape
        inp = torch.full((B,), sos_id, dtype=torch.long, device=targets.device)
        logits = []
        for t in range(T):
            h = self.cell(self.emb(inp), h)                      # (B, hid)
            step_logits = self.out(h)                            # (B, vocab)
            logits.append(step_logits)
            use_gold = random.random() < tf_ratio
            inp = targets[:, t] if use_gold else step_logits.argmax(dim=-1)   # next input
        return torch.stack(logits, dim=1)                        # (B, T, vocab)

dec = Decoder(vocab=20, emb=16, hid=32)
h0 = torch.zeros(5, 32)
tgt = torch.randint(3, 20, (5, 9))
logits = dec(h0, tgt, sos_id=1, tf_ratio=0.8)     # 80% teacher forcing, 20% self-feeding
loss = nn.functional.cross_entropy(logits.reshape(-1, 20), tgt.reshape(-1))
print(logits.shape, float(loss))
```

### Decoding: greedy versus beam search {#decoding}

After training, we want the *most probable output sequence* $\hat y=\arg\max_y\sum_t\log p(y_t\mid y_{<t},\boldsymbol{x})$. The number of candidate sequences is $V^{T}$, and because each factor depends on the previous choices the problem has no dynamic-programming shortcut for a recurrent decoder (the state depends on the entire prefix). We approximate.

**Greedy decoding** picks the argmax at every step. It is fast ($O(T)$ model calls) but myopic: a locally best token can lead into a poor continuation, exactly as in the toy bigram example in the demo below, where the most probable first word leads to a flat, high-entropy continuation.

**Beam search** keeps the $B$ best partial hypotheses (ranked by cumulative log-probability) at every step. At each step, every hypothesis is expanded by all $V$ possible next tokens, giving $BV$ candidates, and the top $B$ are retained. Hypotheses that emit `<eos>` are moved to a finished list. The algorithm stops when no live hypotheses are left or a length limit is reached, and returns the finished hypothesis with the best (normalized) score. Cost is $O(BTV)$ with batching across the beam; $B=1$ recovers greedy decoding.

{{figure: week07-beam-tree | Beam search with width 2 on a toy language model. At every step only the two best prefixes survive (solid); the rest are pruned (dashed). Greedy search would have followed only the single best node at each level.}}

Three practical details matter:

1. **Length bias.** The score of a hypothesis is a sum of negative numbers, so every extra token lowers it, and beam search systematically prefers short outputs, including empty or truncated translations. The standard fix is **length normalization**: rank finished hypotheses by $\frac{1}{|y|^{\alpha}}\sum_t\log p(y_t\mid\cdot)$ with $\alpha\in[0,1]$ ($\alpha=1$ is the per-token mean). Google's NMT system (Wu et al., 2016) used a slightly different penalty, plus a coverage term.
2. **Larger beams are not monotonically better.** More search finds higher-probability sequences, but the probability under the *model* is not the same as quality. Tuning shows BLEU for translation typically improves up to a beam of about 4 to 10 then flattens or even degrades; for open-ended generation (stories, chat), beam search yields bland, repetitive text, which is why sampling methods (top-$k$, nucleus sampling) are preferred there ([Week 10](week-10.html)).
3. **Beam search is a heuristic and can be wrong even for the model.** It is not guaranteed to find the true argmax.

We measured how close beam search gets to the exact optimum. For 60 random bigram language models (vocabulary of 12 words, sequences of length 10, concentrated Dirichlet-distributed transition rows), the true best sequence can be found exactly with the Viterbi algorithm, so we can measure the gap in log-probability between that optimum and what beam search returns.

```chart
{"type": "bar", "caption": "Mean gap in log-probability (nats) between the exact best sequence (Viterbi) and the beam-search result, for 60 random bigram models (V=12, length 10; NumPy experiment). Beam width 1 is greedy decoding. The fraction of models where beam search found the exact optimum was 33% (B=1), 58% (B=2), 73% (B=3), 80% (B=4), 82% (B=5), 90% (B=8), 97% (B=16).", "height": 320, "data": {"labels": ["B=1", "B=2", "B=3", "B=4", "B=5", "B=8", "B=16"], "datasets": [{"label": "mean log-prob gap to optimum", "data": [1.156, 0.384, 0.142, 0.108, 0.076, 0.021, 0.001], "backgroundColor": "var(--c-accent)"}]}, "options": {"scales": {"x": {"title": {"display": true, "text": "beam width"}}, "y": {"title": {"display": true, "text": "gap (nats)"}}}}}
```

Greedy search misses the optimum in two of three cases and loses over a nat on average (a factor of $e^{1.16}\approx3.2$ in probability); a width of only 4 recovers most of the gap, and returns diminish quickly. Try the visualizer on a tiny language model where you can see the mechanism.

{{demo: week07-beam-search | Beam search visualizer | 520}}

Here is a compact, batch-free beam search that works with *any* autoregressive model exposing a function `step(prefix) -> log-probs over the vocabulary`. It is deliberately simple so you can read it; production implementations batch the beams and cache decoder states.

```python:beam_search.py
import math
from typing import Callable, List, Tuple

def beam_search(step: Callable[[List[int]], List[float]], sos: int, eos: int,
                beam: int = 4, max_len: int = 20, alpha: float = 0.7) -> List[Tuple[List[int], float]]:
    """step(prefix) returns log p(next token | prefix) as a list of length V.
    Returns finished hypotheses (token list without sos, normalized score), best first."""
    live = [([sos], 0.0)]                    # (prefix, cumulative log-prob)
    finished = []
    for _ in range(max_len):
        candidates = []
        for prefix, score in live:
            logp = step(prefix)
            for tok, lp in enumerate(logp):
                candidates.append((prefix + [tok], score + lp))
        candidates.sort(key=lambda c: c[1], reverse=True)
        live = []
        for prefix, score in candidates[: beam * 2]:           # look a little past `beam` so eos does not starve the beam
            if prefix[-1] == eos:
                length = len(prefix) - 1                         # exclude sos
                finished.append((prefix[1:], score / (length ** alpha)))
            else:
                live.append((prefix, score))
            if len(live) == beam:
                break
        if not live:
            break
    if not finished:                                             # fell off max_len: return the live ones
        finished = [(p[1:], s / (max(1, len(p) - 1) ** alpha)) for p, s in live]
    return sorted(finished, key=lambda f: f[1], reverse=True)

# Toy check: a 3-token vocabulary {0: <eos>, 1: a, 2: b}; model prefers 'a' then 'b' then stops
def toy_step(prefix):
    table = {1: [0.05, 0.05, 0.9], 2: [0.9, 0.05, 0.05]}         # after 'a' -> 'b'; after 'b' -> <eos>
    probs = table.get(prefix[-1], [0.1, 0.8, 0.1])               # after <sos> (=3) -> 'a'
    return [math.log(p) for p in probs]

print(beam_search(toy_step, sos=3, eos=0, beam=2, max_len=6))   # [([1, 2, 0], ...)]
```

## Attention: removing the bottleneck {#attention}

Squeezing a 50-word sentence into one vector and asking a decoder to unfold it is cruel. When a human translates, they glance back at the *relevant source words* for each target word. **Attention** lets the decoder do the same: at every output step it computes a weighted average of *all* encoder states, with weights that depend on what it is about to produce. Bahdanau, Cho and Bengio (2015, "Neural machine translation by jointly learning to align and translate") introduced this for translation; the idea of a learned soft alignment is the direct ancestor of the attention in [Week 9](week-09.html).

### Bahdanau (additive) attention {#bahdanau}

Let the encoder produce one annotation per source position, $\boldsymbol{h}_1,\dots,\boldsymbol{h}_S$ (typically from a bidirectional RNN, so each $\boldsymbol{h}_i\in\R^{2d}$ summarizes the source *around* position $i$). Let $\boldsymbol{s}_{t-1}$ be the decoder state before producing output $t$. For each decoder step:

1. **Score** how well each source position matches the current decoding need with a small feed-forward network:
$$
e_{ti}=\boldsymbol{v}^\top\tanh\!\big(\mathbf{W}\boldsymbol{s}_{t-1}+\mathbf{U}\boldsymbol{h}_i\big),\qquad \boldsymbol{v}\in\R^{a},\ \mathbf{W}\in\R^{a\times d_s},\ \mathbf{U}\in\R^{a\times d_h}.
$$
2. **Normalize** the scores over source positions into weights: $\alpha_{ti}=\dfrac{\exp(e_{ti})}{\sum_{j=1}^{S}\exp(e_{tj})}$, so $\alpha_{ti}\ge0$ and $\sum_i\alpha_{ti}=1$.
3. **Pool**: the **context vector** is the expected annotation under that distribution, $\boldsymbol{c}_t=\sum_{i=1}^{S}\alpha_{ti}\boldsymbol{h}_i$.
4. **Decode**: use $\boldsymbol{c}_t$ in the recurrent update and the output, e.g. $\boldsymbol{s}_t=f(\boldsymbol{s}_{t-1},y_{t-1},\boldsymbol{c}_t)$ and $p(y_t\mid\cdot)=\softmax(g(\boldsymbol{s}_t,y_{t-1},\boldsymbol{c}_t))$.

{{figure: week07-attention | Additive attention at one decoder step. Scores are computed between the previous decoder state and every encoder annotation, softmaxed into weights, and used to form a weighted sum, the context, which feeds the next decoder state.}}

The key points are conceptual. The "alignment" is **soft** (a distribution, not a hard choice), which makes the whole mechanism differentiable: gradients flow from the loss through $\boldsymbol{c}_t$ into both the encoder states and the scoring network, and the alignment is learned *without any alignment labels*, purely from the translation loss. The context vector is **different at every step**: the fixed-size bottleneck is replaced by $S$ vectors the decoder can address, so memory grows with the source length. And the weights are interpretable: plotting $\alpha_{ti}$ as a matrix (target by source) shows what the model looked at, usually a near-diagonal band with local reorderings (adjective-noun swaps) that match linguistic intuition. Bahdanau et al. also reported that BLEU no longer collapsed for long sentences.

{{demo: week07-attention-heatmap | Attention alignment heatmap | 470}}

:::math Gradient flow through attention
Attention also shortens gradient paths. Without it, the gradient from a target word to the *first* source word travels back through the decoder steps and then through all encoder steps, a path of length about $S+T$ with the vanishing problems above. With attention, the loss at step $t$ reaches $\boldsymbol{h}_i$ in a **single hop**: $\partial\mathcal{L}_t/\partial\boldsymbol{h}_i=\alpha_{ti}\,\partial\mathcal{L}_t/\partial\boldsymbol{c}_t+(\text{terms through }\alpha_{t\cdot})$. The distance between any target position and any source position is constant. That is a second reason attention works, beyond removing the capacity bottleneck, and it is the property the Transformer pushes to its logical extreme by making *every* position attend to every other.
:::

### Luong (multiplicative) attention and score functions {#luong}

Luong, Pham and Manning (2015, "Effective approaches to attention-based neural machine translation") simplified and compared variants. Differences from Bahdanau's version:

- Use the **current** decoder state $\boldsymbol{s}_t$ (computed first, from the previous word and state) to attend, instead of $\boldsymbol{s}_{t-1}$; then combine context and state: $\tilde{\boldsymbol{s}}_t=\tanh(\mathbf{W}_c[\boldsymbol{c}_t;\boldsymbol{s}_t])$ and predict from $\tilde{\boldsymbol{s}}_t$.
- Cheaper **score functions** than the feed-forward network:

| Name | Score $e_{ti}$ |
|---|---|
| dot | $\boldsymbol{s}_t^\top\boldsymbol{h}_i$ |
| general (multiplicative) | $\boldsymbol{s}_t^\top\mathbf{W}_a\boldsymbol{h}_i$ |
| concat (additive, as Bahdanau) | $\boldsymbol{v}^\top\tanh(\mathbf{W}_a[\boldsymbol{s}_t;\boldsymbol{h}_i])$ |

- **Global versus local attention.** Global attends to all source positions (what we described). Local attention predicts an aligned position $p_t$ and attends only to a window around it, which reduces cost on long inputs.
- **Input feeding.** The previous attentional vector $\tilde{\boldsymbol{s}}_{t-1}$ is concatenated to the next decoder input so the model knows what it already attended to (a soft coverage mechanism).

The dot-product form is the seed of the Transformer's scaled dot-product attention: queries are decoder states, keys and values are encoder states. The Transformer adds $1/\sqrt{d_k}$ scaling because dot products of high-dimensional vectors have variance growing with $d_k$, which pushes the softmax into saturation ([Week 9](week-09.html#scaled-dot-product)). Dot-product attention is faster and uses highly optimized matrix multiplication, at the price of requiring query and key to have the same dimension; additive attention can work in mismatched spaces and was found to be better for large $d$ *without* scaling.

```python:attention_modules.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class AdditiveAttention(nn.Module):
    """Bahdanau: e_ti = v^T tanh(W s + U h_i)."""
    def __init__(self, dec_dim, enc_dim, att_dim):
        super().__init__()
        self.W = nn.Linear(dec_dim, att_dim, bias=False)
        self.U = nn.Linear(enc_dim, att_dim, bias=True)
        self.v = nn.Linear(att_dim, 1, bias=False)

    def forward(self, s, enc, mask):
        """s: (B, dec_dim) decoder state; enc: (B, S, enc_dim); mask: (B, S) True where real token."""
        scores = self.v(torch.tanh(self.W(s).unsqueeze(1) + self.U(enc))).squeeze(-1)   # (B, S)
        scores = scores.masked_fill(~mask, float("-inf"))                                # ignore padding
        alpha = F.softmax(scores, dim=-1)                                               # (B, S), rows sum to 1
        context = torch.bmm(alpha.unsqueeze(1), enc).squeeze(1)                          # (B, enc_dim)
        return context, alpha

class DotAttention(nn.Module):
    """Luong 'dot' (requires dec_dim == enc_dim) and 'general' (learned bilinear map) in one class."""
    def __init__(self, dec_dim, enc_dim, general=True):
        super().__init__()
        self.Wa = nn.Linear(enc_dim, dec_dim, bias=False) if general else nn.Identity()

    def forward(self, s, enc, mask):
        scores = torch.bmm(self.Wa(enc), s.unsqueeze(-1)).squeeze(-1)                   # (B, S)
        alpha = F.softmax(scores.masked_fill(~mask, float("-inf")), dim=-1)
        return torch.bmm(alpha.unsqueeze(1), enc).squeeze(1), alpha

B, S, de, dd = 3, 7, 20, 16
enc = torch.randn(B, S, de); s = torch.randn(B, dd)
mask = torch.ones(B, S, dtype=torch.bool); mask[1, 5:] = False       # second sequence has length 5
ctx, a = AdditiveAttention(dd, de, 24)(s, enc, mask)
print(ctx.shape, a.shape, a.sum(-1))        # (3, 20), (3, 7), all ones
print(a[1, 5:])                              # zeros: padded positions get no attention
```

:::history From alignment to attention to Transformers
Soft alignment had precedent in statistical MT, where IBM alignment models (1990s) estimated which source word generated which target word. Bahdanau et al. (arXiv 2014, ICLR 2015) learned alignment jointly with translation. Luong et al. (EMNLP 2015) simplified it. Xu et al. (2015, "Show, attend and tell") applied the same mechanism over image regions for captioning. Google's neural machine translation system (Wu et al., 2016) put an attentional LSTM seq2seq into production. In 2017, Vaswani et al. asked whether the recurrence was needed at all, and the answer ("Attention is all you need") is the subject of [Week 9](week-09.html).
:::

## Applications {#applications}

### Machine translation {#mt}

Neural machine translation was the application that drove this whole line of work. The pipeline: tokenize the source and target (subword units, see [Week 8](week-08.html#subword-tokenization)), embed, encode with a (bi)LSTM, decode with an attentional LSTM, and search with beam search. The 2014 to 2016 sequence of results is a good case study in how quickly the field moved. Sutskever et al. (2014) showed a plain LSTM seq2seq could approach phrase-based statistical MT on English to French; attention (Bahdanau 2015, Luong 2015) removed the length degradation; and Wu et al. (2016) deployed an eight-layer attentional LSTM with residual connections as Google Translate's production system, reporting large human-evaluated error reductions on several language pairs. The metric to know is **BLEU**: a geometric mean of $n$-gram precisions ($n=1..4$) against reference translations, multiplied by a brevity penalty that punishes outputs shorter than the reference. It is cheap, correlates only roughly with human judgment, and is only comparable when computed with the same tokenization and tooling (use `sacreBLEU`).

### Speech recognition and CTC {#speech-ctc}

In speech recognition, the input is a sequence of $T$ acoustic frames (typically 100 per second of audio, as log-mel filterbank features) and the output is a *shorter* character or word-piece sequence. Alignment between frames and letters is unknown. Attention-based seq2seq models can learn it (Chan et al. 2016, "Listen, attend and spell"), but a different loss, **connectionist temporal classification** (Graves, Fernandez, Gomez and Schmidhuber, 2006), is widely used because it needs no decoder at all and supports streaming.

The idea: let a (bi)RNN emit, at every frame, a distribution over the output alphabet plus a special **blank** symbol $\varnothing$. A frame-level labeling $\pi\in(\mathcal{A}\cup\{\varnothing\})^{T}$ is mapped to an output string by the many-to-one function $\mathcal{B}$ that first merges repeated symbols and then deletes blanks: $\mathcal{B}(\text{h h }\varnothing\text{ e l }\varnothing\text{ l l o})=\text{hello}$. (The blank between the two "l"s is what allows double letters.) The probability of a target string $\boldsymbol{y}$ is the sum over **all** frame-level paths that collapse to it, assuming frame-wise conditional independence given the input:

$$
p(\boldsymbol{y}\mid\boldsymbol{x})=\sum_{\pi:\ \mathcal{B}(\pi)=\boldsymbol{y}}\ \prod_{t=1}^{T}p(\pi_t\mid\boldsymbol{x}).
$$

The number of paths is exponential, but a forward dynamic program over the target string interleaved with blanks computes the sum in $O(T\cdot|\boldsymbol{y}|)$, and its gradient comes from the matching backward pass, so the loss $-\log p(\boldsymbol{y}\mid\boldsymbol{x})$ is trained end to end. At test time one can decode greedily (argmax per frame, then collapse) or with a beam search that merges paths with equal collapsed prefixes. CTC is the same marginalization idea as the latent-variable models of [Week 12](week-12.html), applied to alignments.

```python:ctc_loss.py
import torch
import torch.nn as nn

T, B, C = 50, 4, 28                  # frames, batch, classes (index 0 is the CTC blank; 1..27 = a-z and space)
log_probs = torch.randn(T, B, C).log_softmax(dim=2).requires_grad_()    # NOTE the layout: (T, B, C)
targets = torch.randint(1, C, (B, 12))                  # label sequences must not contain the blank (0)
input_lengths = torch.full((B,), T, dtype=torch.long)   # number of valid frames per example
target_lengths = torch.full((B,), 12, dtype=torch.long)

ctc = nn.CTCLoss(blank=0, zero_infinity=True)           # zero_infinity avoids inf when T is too short to emit the target
loss = ctc(log_probs, targets, input_lengths, target_lengths)
loss.backward()
print(loss.item(), log_probs.grad.shape)                # scalar, (50, 4, 28)
```

(Requirement to remember: the input must have at least as many frames as the target has symbols, plus one extra frame for every pair of adjacent identical symbols, otherwise no valid alignment exists and the loss is infinite.) Modern speech systems have moved to Transformers and RNN-transducers, but the CTC objective remains a standard building block.

### Time-series forecasting {#time-series}

Forecasting is a many-to-one or many-to-many problem on real-valued sequences: given the past $L$ observations (a *window*), predict the next $H$ values. Recurrent networks were the neural baseline for years, with caveats that experienced practitioners repeat constantly:

- **Always compare with simple baselines.** A "predict the last value" or "same value as one season ago" baseline is shockingly hard to beat for many series. Report MAE or RMSE relative to the baseline, not in isolation.
- **Split by time, never randomly.** A random train/validation split lets the model interpolate between neighbouring windows it has trained on and gives dishonestly good numbers. Train on the past, validate on a later period, test on the latest.
- **Normalize using training statistics only**, and consider per-window normalization (subtract the window's last value or mean) for non-stationary series.
- **Choose between one-step-ahead recursion and direct multi-horizon output.** Feeding predictions back (recursive forecasting) has the same exposure-bias issue as seq2seq decoding; predicting all $H$ steps at once avoids it at the cost of ignoring dependencies between outputs.

```python:forecast_lstm.py
import numpy as np
import torch
import torch.nn as nn

# Synthetic series: two seasonalities plus noise
rng = np.random.default_rng(0)
t = np.arange(3000)
series = np.sin(2 * np.pi * t / 50) + 0.5 * np.sin(2 * np.pi * t / 17) + 0.1 * rng.normal(size=t.size)

L, H = 60, 10                                           # window length, horizon
def windows(s):
    X = np.stack([s[i:i + L] for i in range(len(s) - L - H)])
    Y = np.stack([s[i + L:i + L + H] for i in range(len(s) - L - H)])
    return torch.tensor(X, dtype=torch.float32).unsqueeze(-1), torch.tensor(Y, dtype=torch.float32)

split = int(0.8 * len(series))                          # time-based split, no shuffling across the boundary
Xtr, Ytr = windows(series[:split])
Xva, Yva = windows(series[split:])
mu, sd = Xtr.mean(), Xtr.std()                          # normalise with TRAIN statistics only

class Forecaster(nn.Module):
    def __init__(self, hid=64, horizon=H):
        super().__init__()
        self.rnn = nn.LSTM(1, hid, batch_first=True)
        self.head = nn.Linear(hid, horizon)            # direct multi-horizon: all H steps at once
    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out[:, -1])                    # last hidden state -> (B, H)

model = Forecaster()
opt = torch.optim.Adam(model.parameters(), lr=3e-3)
for epoch in range(5):
    perm = torch.randperm(len(Xtr))
    for i in range(0, len(perm), 128):
        idx = perm[i:i + 128]
        pred = model((Xtr[idx] - mu) / sd) * sd + mu
        loss = nn.functional.mse_loss(pred, Ytr[idx])
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    with torch.no_grad():
        va = nn.functional.mse_loss(model((Xva - mu) / sd) * sd + mu, Yva).item()
        base = nn.functional.mse_loss(Xva[:, -1:, 0].expand(-1, H), Yva).item()   # "repeat last value" baseline
    print(f"epoch {epoch}  val MSE {va:.4f}  last-value baseline {base:.4f}")
```

### One-dimensional convolutions and temporal convolutional networks {#tcn}

Recurrence is not the only way to process a sequence. A **1D convolution** slides a kernel of width $k$ along the time axis: $\boldsymbol{y}_t=\sum_{j=0}^{k-1}\mathbf{W}_j\boldsymbol{x}_{t-j}$, exactly the operation of [Week 5](week-05.html) with one spatial dimension. Compared with an RNN it has three advantages: all positions are computed **in parallel** (no sequential dependency), gradients travel along paths whose length is the *network depth*, not the sequence length, and training is stable because there is no repeated multiplication by the same matrix. The disadvantage is a **fixed, finite receptive field**: a stack of $L$ layers with kernel $k$ sees only $1+L(k-1)$ steps.

**Causal convolutions** pad on the left only, so $\boldsymbol{y}_t$ never depends on $\boldsymbol{x}_{t+1},\dots$ (necessary for forecasting and autoregressive generation). **Dilated convolutions** skip inputs: with dilation $\delta$, taps are spaced $\delta$ apart, $\boldsymbol{y}_t=\sum_j\mathbf{W}_j\boldsymbol{x}_{t-\delta j}$. Doubling the dilation at each layer ($\delta=1,2,4,\dots,2^{L-1}$) makes the receptive field grow **exponentially** with depth, $1+(k-1)(2^{L}-1)$, at constant cost per layer.

{{figure: week07-tcn | Causal dilated convolutions with kernel 2 and dilations 1, 2, 4 (bottom to top). Highlighted nodes are the receptive field of the last output: eight time steps, all in the past. Doubling the dilation doubles the reach per layer.}}

WaveNet (van den Oord et al., 2016) used this stack for raw audio generation at 16 kHz, and Bai, Kolter and Koltun (2018, "An empirical evaluation of generic convolutional and recurrent networks for sequence modeling") packaged residual blocks of dilated causal convolutions as the **temporal convolutional network (TCN)** and showed that it matched or beat LSTMs on a range of standard sequence benchmarks, with faster training and longer effective memory. The lesson is nuanced: for fixed-horizon tasks where you can afford a receptive field covering the relevant history, TCNs are a strong, simple baseline; RNNs retain the advantage of constant memory at inference and unbounded (if fragile) context.

```python:tcn.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalConv1d(nn.Module):
    """Conv1d that only looks at the past: pad (k-1)*dilation zeros on the LEFT."""
    def __init__(self, c_in, c_out, k, dilation):
        super().__init__()
        self.pad = (k - 1) * dilation
        self.conv = nn.Conv1d(c_in, c_out, k, dilation=dilation)
    def forward(self, x):                               # x: (B, C, T)
        return self.conv(F.pad(x, (self.pad, 0)))       # output length stays T

class TCNBlock(nn.Module):
    def __init__(self, c, k, dilation, p=0.1):
        super().__init__()
        self.c1, self.c2 = CausalConv1d(c, c, k, dilation), CausalConv1d(c, c, k, dilation)
        self.drop = nn.Dropout(p)
    def forward(self, x):
        y = self.drop(F.relu(self.c1(x)))
        y = self.drop(F.relu(self.c2(y)))
        return F.relu(x + y)                            # residual connection

class TCN(nn.Module):
    def __init__(self, c_in, c_hidden, n_layers=6, k=3):
        super().__init__()
        self.inp = nn.Conv1d(c_in, c_hidden, 1)
        self.blocks = nn.Sequential(*[TCNBlock(c_hidden, k, 2 ** i) for i in range(n_layers)])
    def forward(self, x):                               # (B, C_in, T) -> (B, C_hidden, T)
        return self.blocks(self.inp(x))

net = TCN(c_in=1, c_hidden=32, n_layers=6, k=3)
x = torch.randn(4, 1, 200)
print(net(x).shape)                                      # (4, 32, 200)
# receptive field: 1 + 2 * (k-1) * (2^6 - 1) = 1 + 2*2*63 = 253 steps (two convs per block)
# Causality check: perturbing the future must not change earlier outputs
x2 = x.clone(); x2[:, :, 150:] += 1.0
print(torch.allclose(net.eval()(x)[:, :, :150], net.eval()(x2)[:, :, :150], atol=1e-6))   # True
```

:::note How this fits the bigger picture
Three families compete for sequence modeling: recurrent (constant memory, sequential), convolutional (parallel, finite receptive field), and attention-based (parallel, global receptive field, quadratic cost in length). Weeks 9 to 11 show attention overtaking the other two for most tasks at scale, but recurrent ideas are alive again in state-space models such as S4 and Mamba (see [Week 16](week-16.html)) that revive linear recurrences with careful parameterization to avoid the vanishing-gradient pitfall derived above.
:::

## Lab 1: a character-level language model with LSTM and GRU {#lab-char-lm}

Goal: train a character-level language model on Tiny Shakespeare (about 1.1 million characters), using truncated BPTT with a carried hidden state, gradient clipping, and a switchable LSTM/GRU core; evaluate in bits per character; and sample text at several temperatures. Run it on a Colab GPU; an epoch takes well under a minute.

**Step 1: data.** Encode characters as integers and arrange the stream into $B$ parallel columns, so that consecutive minibatches continue each column's text (required for carrying the hidden state).

```python:lab1_data.py
import urllib.request, math, torch

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
text = urllib.request.urlopen(URL).read().decode("utf-8")
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
V = len(chars)
print(len(text), V)                                   # about 1.1M characters, V = 65

data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
n_train = int(0.9 * len(data))
train_data, val_data = data[:n_train], data[n_train:]

def batchify(d, B):
    """Reshape a 1-D stream to (B, N): row b is a contiguous slice, so batches continue each row."""
    N = len(d) // B
    return d[: N * B].view(B, N)

device = "cuda" if torch.cuda.is_available() else "cpu"
B, K = 64, 128                                        # batch of 64 streams, BPTT window of 128 characters
train_b, val_b = batchify(train_data, B).to(device), batchify(val_data, B).to(device)

def get_chunk(src, i, K):
    """Inputs src[:, i:i+K], targets shifted by one character."""
    k = min(K, src.size(1) - 1 - i)
    return src[:, i : i + k], src[:, i + 1 : i + 1 + k]

x, y = get_chunk(train_b, 0, K)
print(x.shape, y.shape)                               # (64, 128) (64, 128)
print("".join(itos[int(c)] for c in x[0, :40]))         # first characters of stream 0
print("baseline loss ln(V) =", math.log(V))             # about 4.17 nats = 6.02 bits per character
```

**Step 2: model and utilities.**

```python:lab1_model.py
import torch, torch.nn as nn

class CharRNN(nn.Module):
    def __init__(self, V, emb=64, hid=512, layers=2, cell="lstm", dropout=0.2):
        super().__init__()
        self.emb = nn.Embedding(V, emb)
        rnn_cls = {"lstm": nn.LSTM, "gru": nn.GRU}[cell]
        self.rnn = rnn_cls(emb, hid, num_layers=layers, batch_first=True, dropout=dropout)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hid, V)
        self.hid, self.layers, self.cell = hid, layers, cell

    def forward(self, x, state=None):                 # x: (B, K) int64
        out, state = self.rnn(self.drop(self.emb(x)), state)    # out: (B, K, hid)
        return self.head(self.drop(out)), state       # logits: (B, K, V)

    def init_state(self, B, device):
        h = torch.zeros(self.layers, B, self.hid, device=device)
        return (h, h.clone()) if self.cell == "lstm" else h

def detach(state):
    """Cut the graph at the chunk boundary (truncated BPTT). Handles LSTM tuples and GRU tensors."""
    return tuple(s.detach() for s in state) if isinstance(state, tuple) else state.detach()
```

**Step 3: training and evaluation.** The BPC is the cross-entropy in nats divided by $\ln2$.

```python:lab1_train.py
import time, math
import torch, torch.nn as nn

def evaluate(model, src, K):
    model.eval()
    state, total, count = model.init_state(src.size(0), src.device), 0.0, 0
    with torch.no_grad():
        for i in range(0, src.size(1) - 1, K):
            x, y = get_chunk(src, i, K)
            logits, state = model(x, state)
            total += nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum").item()
            count += y.numel()
    nats = total / count
    return nats, nats / math.log(2), math.exp(nats)        # loss, bits per char, perplexity

def train(cell="lstm", epochs=5, lr=2e-3, clip=1.0):
    torch.manual_seed(0)
    model = CharRNN(V, cell=cell).to(device)
    # forget-gate bias trick for the LSTM (see the LSTM section)
    if cell == "lstm":
        for name, p in model.rnn.named_parameters():
            if "bias_ih" in name:
                p.data[model.hid : 2 * model.hid] = 1.0
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * (train_b.size(1) // K))
    for ep in range(epochs):
        model.train(); t0 = time.time()
        state = model.init_state(B, device)
        for i in range(0, train_b.size(1) - 1, K):
            x, y = get_chunk(train_b, i, K)
            state = detach(state)                             # carry state, cut gradients
            logits, state = model(x, state)
            loss = nn.functional.cross_entropy(logits.reshape(-1, V), y.reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step(); sched.step()
        _, bpc, ppl = evaluate(model, val_b, K)
        print(f"{cell} epoch {ep+1}  train loss {loss.item():.3f}  val BPC {bpc:.3f}  val ppl {ppl:.2f}  "
              f"last |grad| {gnorm:.2f}  {time.time()-t0:.0f}s")
    return model

lstm_model = train("lstm")
gru_model = train("gru")
```

Expect the validation BPC to drop from about 6 (the uniform baseline) to roughly 2.3 bits after the first epoch and to 1.7 to 1.9 after a handful of epochs; the exact numbers depend on hardware and seed, and the two cells usually finish within a few hundredths of each other. If your first-batch loss is far from $\ln65\approx4.17$, check the initialization and the target shift before anything else.

**Step 4: sampling.** Generation feeds the model's own output back, one character at a time. The temperature $\tau$ rescales the logits before the softmax: $p_\tau(c)\propto\exp(z_c/\tau)$. Low $\tau$ approaches greedy decoding (repetitive, safe), $\tau=1$ samples from the model, high $\tau$ flattens toward uniform (random, noisy).

```python:lab1_sample.py
import torch

@torch.no_grad()
def sample(model, prime="ROMEO:\n", n=400, temperature=0.8, top_k=None):
    model.eval()
    state = model.init_state(1, device)
    x = torch.tensor([[stoi[c] for c in prime]], device=device)
    out = list(prime)
    logits, state = model(x, state)                     # feed the whole prime at once
    for _ in range(n):
        z = logits[0, -1] / max(temperature, 1e-6)
        if top_k is not None:
            kth = torch.topk(z, top_k).values[-1]
            z = z.masked_fill(z < kth, float("-inf"))   # keep only the k most likely characters
        probs = torch.softmax(z, dim=-1)
        nxt = torch.multinomial(probs, 1)               # shape (1,)
        out.append(itos[int(nxt)])
        logits, state = model(nxt.view(1, 1), state)    # feed one character, keep the state
    return "".join(out)

for tau in (0.5, 0.8, 1.2):
    print(f"--- temperature {tau} ---")
    print(sample(lstm_model, temperature=tau))
```

**Experiments to run** (each answers a question from this week):
1. Set `clip=1e9` (no clipping) and `lr=2e-2`: do you see a spike in the printed gradient norm and then divergence or NaN? Compare with the clipped run.
2. Change the window `K` from 128 to 16 and to 512. How does validation BPC change, and what does that tell you about the useful credit-assignment horizon in this data?
3. Remove the forget-gate bias initialization. Does the early training curve differ?
4. Replace `nn.LSTM` by a vanilla `nn.RNN(nonlinearity="tanh")` with the same hidden size. How large is the gap?
5. Look at the samples at $\tau=0.5$ versus $1.2$. Which has better spelling, and which has more variety? What would a bits-per-character number tell you about each?

## Lab 2: sequence-to-sequence with attention on a toy translation task {#lab-seq2seq}

Real translation needs data and hours of GPU time. To see every mechanism in minutes, we invent a **toy language pair** whose structure is known: the source is a random string of 6 to 8 letters from an alphabet of 12 symbols; the target is the *reversed* string with each letter mapped through a fixed substitution cipher. The correct alignment is therefore the anti-diagonal. A model without attention must squeeze the whole string into one vector; a model with attention should recover the anti-diagonal in its attention map, which gives us a ground truth to compare against, something real translation never offers.

**Step 1: the data.**

```python:lab2_data.py
import random, torch

random.seed(0); torch.manual_seed(0)
PAD, SOS, EOS = 0, 1, 2
letters = list("abcdefghijkl")                      # source alphabet (12 symbols)
cipher = dict(zip(letters, list("lkjihgfedcba")))   # target symbol for each source symbol (fixed substitution)
src_vocab = ["<pad>", "<sos>", "<eos>"] + letters
tgt_vocab = ["<pad>", "<sos>", "<eos>"] + [cipher[c] for c in letters]
s2i = {c: i for i, c in enumerate(src_vocab)}
t2i = {c: i for i, c in enumerate(tgt_vocab)}

def make_pair(min_len=6, max_len=8):
    n = random.randint(min_len, max_len)
    src = [random.choice(letters) for _ in range(n)]
    tgt = [cipher[c] for c in reversed(src)]         # reversal + substitution
    return src, tgt

def encode(src, tgt, max_len=8):
    s = [s2i[c] for c in src] + [EOS]                                   # source ends with <eos>
    t = [t2i[c] for c in tgt] + [EOS]                                   # decoder target ends with <eos>
    s += [PAD] * (max_len + 1 - len(s)); t += [PAD] * (max_len + 1 - len(t))
    return s, t

def batch(n):
    pairs = [encode(*make_pair()) for _ in range(n)]
    S = torch.tensor([p[0] for p in pairs]); T = torch.tensor([p[1] for p in pairs])
    return S, T                                                          # both (n, 9)

S, T = batch(3)
print(S); print(T)
print([src_vocab[i] for i in S[0]], "->", [tgt_vocab[i] for i in T[0]])
```

**Step 2: the model.** A bidirectional GRU encoder, additive attention, and a GRU-cell decoder that, at each step, attends with its previous state, concatenates the context to the input embedding and predicts from state plus context.

```python:lab2_model.py
import random
import torch, torch.nn as nn, torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, V, emb, hid):
        super().__init__()
        self.emb = nn.Embedding(V, emb, padding_idx=PAD)
        self.rnn = nn.GRU(emb, hid, batch_first=True, bidirectional=True)
        self.bridge = nn.Linear(2 * hid, hid)           # maps final fwd+bwd states to decoder's initial state
    def forward(self, src):
        out, h_n = self.rnn(self.emb(src))              # out: (B, S, 2*hid); h_n: (2, B, hid)
        h0 = torch.tanh(self.bridge(torch.cat([h_n[0], h_n[1]], dim=-1)))   # (B, hid)
        return out, h0

class AttnDecoder(nn.Module):
    def __init__(self, V, emb, hid, enc_dim, att=64, use_attention=True):
        super().__init__()
        self.emb = nn.Embedding(V, emb, padding_idx=PAD)
        self.use_attention = use_attention
        self.W, self.U, self.v = nn.Linear(hid, att, bias=False), nn.Linear(enc_dim, att), nn.Linear(att, 1, bias=False)
        self.cell = nn.GRUCell(emb + (enc_dim if use_attention else 0), hid)
        self.out = nn.Linear(hid + (enc_dim if use_attention else 0), V)

    def step(self, tok, h, enc, mask):
        e = self.emb(tok)                                                      # (B, emb)
        alpha = None
        if self.use_attention:
            scores = self.v(torch.tanh(self.W(h).unsqueeze(1) + self.U(enc))).squeeze(-1)     # (B, S)
            alpha = F.softmax(scores.masked_fill(~mask, float("-inf")), dim=-1)
            ctx = torch.bmm(alpha.unsqueeze(1), enc).squeeze(1)                               # (B, enc_dim)
            h = self.cell(torch.cat([e, ctx], dim=-1), h)
            logits = self.out(torch.cat([h, ctx], dim=-1))
        else:
            h = self.cell(e, h)
            logits = self.out(h)
        return logits, h, alpha

class Seq2Seq(nn.Module):
    def __init__(self, use_attention=True, emb=32, hid=128):
        super().__init__()
        self.enc = Encoder(len(src_vocab), emb, hid)
        self.dec = AttnDecoder(len(tgt_vocab), emb, hid, 2 * hid, use_attention=use_attention)

    def forward(self, src, tgt, tf=1.0):
        enc, h = self.enc(src)
        mask = src != PAD
        tok = torch.full((src.size(0),), SOS, dtype=torch.long)
        logits = []
        for t in range(tgt.size(1)):
            lg, h, _ = self.dec.step(tok, h, enc, mask)
            logits.append(lg)
            tok = tgt[:, t] if random.random() < tf else lg.argmax(-1)       # teacher forcing / scheduled sampling
        return torch.stack(logits, dim=1)                                      # (B, T, V_tgt)

    @torch.no_grad()
    def greedy(self, src, max_len=9):
        enc, h = self.enc(src); mask = src != PAD
        tok = torch.full((src.size(0),), SOS, dtype=torch.long)
        outs, atts = [], []
        for _ in range(max_len):
            lg, h, a = self.dec.step(tok, h, enc, mask)
            tok = lg.argmax(-1); outs.append(tok); atts.append(a)
        return torch.stack(outs, 1), (torch.stack(atts, 1) if atts[0] is not None else None)   # (B, T), (B, T, S)
```

**Step 3: train with and without attention, and measure exact-match accuracy.**

```python:lab2_train.py
def train_model(use_attention, steps=1500, bs=64):
    torch.manual_seed(0)
    model = Seq2Seq(use_attention)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    for step in range(1, steps + 1):
        S, T = batch(bs)
        logits = model(S, T, tf=1.0 if step < steps // 2 else 0.8)         # mostly teacher forcing, a bit of self-feeding late
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), T.reshape(-1), ignore_index=PAD)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step % 300 == 0:
            print(f"attention={use_attention} step {step}  loss {loss.item():.4f}")
    return model

def exact_match(model, n=1000):
    model.eval(); S, T = batch(n)
    pred, _ = model.greedy(S)
    ok = 0
    for p, t in zip(pred, T):
        # compare up to and including the first <eos> of the target
        L = int((t == EOS).nonzero()[0]) + 1
        ok += int(torch.equal(p[:L], t[:L]))
    return ok / n

m_att, m_plain = train_model(True), train_model(False)
print("exact match with attention   :", exact_match(m_att))
print("exact match without attention:", exact_match(m_plain))
```

**Step 4: inspect the alignment.** With attention, the matrix of $\alpha_{ti}$ for a test sentence should light up the anti-diagonal.

```python:lab2_attention_plot.py
import matplotlib.pyplot as plt

S, T = batch(1)
pred, att = m_att.greedy(S)                       # att: (1, 9, 9) target steps x source positions
n = int((S[0] != PAD).sum())                      # real source length incl. <eos>
fig, ax = plt.subplots(figsize=(4.5, 4.5))
ax.imshow(att[0, :n, :n].numpy(), cmap="viridis")
ax.set_xticks(range(n)); ax.set_xticklabels([src_vocab[i] for i in S[0, :n]])
ax.set_yticks(range(n)); ax.set_yticklabels([tgt_vocab[i] for i in pred[0, :n]])
ax.set_xlabel("source"); ax.set_ylabel("generated target"); ax.set_title("Attention weights")
plt.show()
```

What to expect (do run it, and treat the following as a hypothesis to test rather than a promise): the attentional model should reach near-perfect exact-match accuracy within the 1500 steps, with the heatmap showing a clean anti-diagonal; the model without attention should do noticeably worse, and the gap should grow if you increase the string length to 15 to 20, because a fixed 128-dimensional vector must now carry more symbols. Try it: set `min_len=14, max_len=16` (and update the padding length) and compare. Then use the [attention demo](#attention) above to build intuition for what the weights look like when training is imperfect.


## Common mistakes and debugging tips {#pitfalls}

Recurrent models fail in characteristic ways. This table maps symptoms to the usual cause and the fix; most of them cost somebody a weekend.

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| Loss is `nan` after a few hundred steps, or spikes by orders of magnitude | Exploding gradients through time (spectral radius of the recurrent Jacobian above 1) | `clip_grad_norm_` with a global threshold (0.5 to 5), lower the learning rate, check the orthogonal or identity-like initialization of $\mathbf{W}_{hh}$ |
| Loss plateaus at roughly $\ln V$ (the unigram or uniform level) and never moves | Vanishing gradients, or a data bug (inputs and targets not shifted by one) | Print one decoded `(input, target)` pair; switch to LSTM/GRU, set the forget-gate bias to 1, shorten the BPTT window while debugging |
| Train loss very low, val loss poor in a language model | Hidden state carried across the train/val boundary, or overlap between train and val text | Re-initialize state at the start of evaluation; split the corpus by contiguous chunk, not by random window |
| Training loss is excellent under teacher forcing but free-running samples are repetitive or drift | Exposure bias | Scheduled sampling, lower decoding temperature, beam search with a length penalty, or more data; always report metrics from free-running decoding |
| Padding tokens change the result, or batch size changes the output | Padding is processed as real input, or loss is averaged over pad positions | Use `pack_padded_sequence` (or a mask), `ignore_index=PAD` in the loss, and take the hidden state at the true last position |
| `RuntimeError: Trying to backward through the graph a second time` | Hidden state carried across truncated-BPTT chunks without `detach()` | `h = h.detach()` (for LSTM, detach both $h$ and $c$) at the start of each chunk |
| Memory grows every step in a training loop | The same cause as above, in the other direction: the graph spans the whole epoch | Same fix; also do not accumulate `loss` tensors, accumulate `loss.item()` |
| Shapes silently wrong: model trains but is bad | Confusing `batch_first`; `nn.LSTM` defaults to `(T, B, F)` | Set `batch_first=True` consistently, assert shapes, and remember that `h_n` is always `(layers*dirs, B, H)` |
| Bidirectional model has implausibly good validation perplexity | Information leak: the backward direction saw the token being predicted | Never use bidirectional recurrence for next-token prediction; use it for tagging and encoding only |
| Beam search returns very short outputs | Log-probabilities are summed, so shorter hypotheses score higher | Length normalization or a length penalty; keep finished hypotheses and compare them fairly |
| Attention heatmap is uniform | Scores saturated or masked incorrectly, or the model does not need attention for the task | Check the mask (padding set to $-\infty$ before softmax), the scale of scores, and the learning rate |

A short debugging routine that works on almost every sequence model:

1. **Overfit ten examples.** If a model cannot drive the loss to near zero on ten sequences, there is a bug (shapes, shifts, masks, learning rate), not a capacity problem.
2. **Check the initial loss.** It should be close to $\ln V$ for a vocabulary of size $V$ (for $V=65$ characters, about 4.17). A very different value points at a wrong output scale or a wrong loss reduction.
3. **Log the gradient norm** before clipping, every step. A norm that is nearly always above the clip threshold means the threshold is acting as the learning rate; a norm of about $10^{-6}$ means vanishing.
4. **Look at outputs, not only the loss.** Sample text, print the greedy decode, plot the attention matrix.
5. **Test causality and masking explicitly** (perturb the future, perturb the padding) as in the TCN code above.

## Exercises {#exercises}

:::exercise Exercise 1: Perplexity and bits per character
A character-level model reaches a validation cross-entropy of 1.10 nats per character. (a) Convert it to bits per character and to perplexity. (b) The same text tokenized into words averages 5.6 characters per word including the space. If a word-level model has perplexity 140 per word, which model is better per character?
:::details Solution
(a) BPC $=1.10/\ln 2\approx1.587$ bits. Perplexity per character is $e^{1.10}\approx3.00$ (equivalently $2^{1.587}$). (b) Convert the word model to a per-character cross-entropy: $\ln 140/5.6\approx4.942/5.6\approx0.883$ nats per character, which is $\approx1.27$ bits. That is lower than $1.10$ nats $=1.587$ bits, so the word model is better per character here. The comparison is only valid when both models are measured over exactly the same text and the word model handles out-of-vocabulary words in the same way (or has no unknown tokens).
:::
:::

:::exercise Exercise 2: Gradient of a linear RNN
Consider the scalar linear RNN $h_t=w\,h_{t-1}+x_t$ with $h_0=0$ and loss $L=h_T$. (a) Compute $\partial L/\partial h_k$ for $k\le T$. (b) For $T=100$, what is the gradient with respect to $h_1$ when $w=0.9$, $w=1$, $w=1.1$? (c) What does this say about learning dependencies of length 100?
:::details Solution
(a) $h_T$ depends on $h_k$ through $T-k$ applications of $w$, so $\partial L/\partial h_k=w^{T-k}$. (b) With $T-k=99$: $0.9^{99}\approx2.9\times10^{-5}$, $1^{99}=1$, $1.1^{99}\approx1.2\times10^{4}$. (c) Only $|w|=1$ keeps the signal intact, and it is a knife edge: any mismatch decays or explodes exponentially in the distance. A vanilla RNN with a nonlinearity has the same structure with $w$ replaced by the Jacobian $\mathbf{D}_t\mathbf{W}_{hh}$, so its long-range gradients behave like $\rho^{T-k}$ for the relevant spectral radius $\rho$. This is exactly what the LSTM's additive cell state with a forget gate near 1 tries to realize.
:::
:::

:::exercise Exercise 3: Why clipping by norm and not by value?
Explain the difference between `clip_grad_value_` and `clip_grad_norm_` for a gradient $\boldsymbol{g}=(30,\,0.3)$ and a threshold of 1. Which preserves the update direction?
:::details Solution
Value clipping clamps each coordinate: $(1,\,0.3)$. The direction changed from almost exactly the first axis (angle $\approx0.6^\circ$) to about $17^\circ$ away from it, so the small coordinate now has a disproportionate influence. Norm clipping rescales the whole vector by $1/\lVert\boldsymbol{g}\rVert\approx1/30$: $(0.9999,\,0.01)$, which has the same direction and a norm of 1. Norm clipping keeps the descent direction and only shortens the step, which is why it is the default for RNNs. (Use the global norm across all parameters, not per-tensor norms, to preserve the direction of the full update.)
:::
:::

:::exercise Exercise 4: Count LSTM and GRU parameters
An `nn.LSTM(input_size=128, hidden_size=256, num_layers=2)` has biases. (a) Count its parameters. (b) How many would the same configuration of `nn.GRU` have? (c) What would a bidirectional version add?
:::details Solution
PyTorch's LSTM has four gate blocks per layer: weights $W_{ih}\in\R^{4H\times d_{in}}$, $W_{hh}\in\R^{4H\times H}$ and two bias vectors of size $4H$. Layer 1: $4\cdot256\cdot128+4\cdot256\cdot256+2\cdot4\cdot256=131{,}072+262{,}144+2{,}048=395{,}264$. Layer 2 (input size 256): $4\cdot256\cdot256+262{,}144+2{,}048=262{,}144+262{,}144+2{,}048=526{,}336$. Total: $921{,}600$. (b) A GRU has three blocks instead of four, so $\tfrac34$ of that: $691{,}200$. (c) A bidirectional model doubles the parameters of layer 1, and layer 2 receives $2H=512$ inputs, so layer 2 grows too: in total more than twice the unidirectional count. Check with `sum(p.numel() for p in model.parameters())`.
:::
:::

:::exercise Exercise 5: Derive the gradient along the cell state
For an LSTM, treat the gates as constants and show that $\partial c_t/\partial c_{t-1}=\diag(f_t)$. Then explain why $\partial c_T/\partial c_k=\diag(\prod_{j=k+1}^{T}f_j)$ can stay near the identity, and give one reason the full gradient is still not perfectly preserved.
:::details Solution
From $c_t=f_t\odot c_{t-1}+i_t\odot g_t$, the direct dependence of $c_t$ on $c_{t-1}$ is element-wise multiplication by $f_t$, hence the diagonal Jacobian $\diag(f_t)$. Chaining gives a product of diagonal matrices, i.e. the element-wise product of the forget gates. If the gates are close to 1 in some unit (forget bias initialized to 1 or higher, so $\sigma(1)\approx0.73$, $\sigma(3)\approx0.95$), the unit's gradient is multiplied by numbers near 1 rather than by a weight matrix with a possibly small spectral radius. Caveats: (i) the gates themselves depend on $h_{t-1}$, which depends on $c_{t-1}$, so there are additional gradient paths through the gates that are not identity-like; (ii) $f_t<1$ still shrinks exponentially, just at a rate the network can learn; (iii) over thousands of steps even $0.99^{T}$ vanishes. LSTMs make hundreds of steps feasible, not unlimited ranges.
:::
:::

:::exercise Exercise 6: Implement truncated BPTT correctly
The loop below has a bug that causes `Trying to backward through the graph a second time`. Find it and fix it. Also say what happens to the gradient across chunk boundaries after your fix.
```python
h = torch.zeros(1, B, H)
for x, y in chunks:                 # each chunk: (B, 35)
    out, h = rnn(x, h)
    loss = criterion(head(out).reshape(-1, V), y.reshape(-1))
    opt.zero_grad(); loss.backward(); opt.step()
```
:::details Solution
`h` returned by one chunk is still attached to that chunk's graph, which `backward()` has already freed. Fix: `out, h = rnn(x, h.detach())` (or `h = h.detach()` right after the step). After the fix the hidden state *value* carries forward, so the model can use context from earlier chunks at inference-like behavior, but gradients do not flow across chunk boundaries: the effective gradient horizon is the chunk length (35), while the state's memory can still be longer. For an LSTM, detach both elements of the `(h, c)` tuple, for example `h = tuple(s.detach() for s in h)`.
:::
:::

:::exercise Exercise 7: Beam search by hand
A model over $\{a,b,\text{eos}\}$ has $p(a\mid\text{start})=0.5$, $p(b\mid\text{start})=0.4$, $p(\text{eos}\mid\text{start})=0.1$. After $a$ it gives $p(a)=0.25$, $p(b)=0.25$, $p(\text{eos})=0.5$. After $b$ it gives $p(\text{eos})=1$. (a) What does greedy decoding return, and with what probability? (b) Run beam search with width 2. Which sequence wins, and why can greedy not find it?
:::details Solution
(a) Greedy takes $a$ (0.5), then eos (0.5): the sequence "a eos" with probability $0.5\cdot0.5=0.25$. (b) After step 1 the beam keeps $a$ (0.5) and $b$ (0.4). Expanding both: "a eos" 0.25, "a a" 0.125, "a b" 0.125, "b eos" $0.4\cdot1=0.40$. The best finished hypothesis is "b eos" with 0.40, beating greedy's 0.25. Greedy commits to the locally best first token and cannot recover. Beam search is still approximate: width 1 is greedy, and no finite width guarantees the global optimum.
:::
:::

:::exercise Exercise 8: Attention as a differentiable lookup (challenging)
Let encoder states be $\boldsymbol{h}_1,\dots,\boldsymbol{h}_S$ and the decoder query $\boldsymbol{s}$. Use dot-product scores $e_i=\boldsymbol{s}^\top\boldsymbol{h}_i$. (a) Write the context vector $\boldsymbol{c}$ and show that $\partial\boldsymbol{c}/\partial e_i=\alpha_i(\boldsymbol{h}_i-\boldsymbol{c})$. (b) What happens as $\lVert\boldsymbol{s}\rVert\to\infty$? (c) Why does the Transformer divide scores by $\sqrt{d_k}$ (preview of [Week 9](week-09.html))?
:::details Solution
(a) $\alpha_i=\softmax(e)_i$ and $\boldsymbol{c}=\sum_j\alpha_j\boldsymbol{h}_j$. The softmax derivative is $\partial\alpha_j/\partial e_i=\alpha_j(\delta_{ij}-\alpha_i)$. Then $\partial\boldsymbol{c}/\partial e_i=\sum_j\alpha_j(\delta_{ij}-\alpha_i)\boldsymbol{h}_j=\alpha_i\boldsymbol{h}_i-\alpha_i\sum_j\alpha_j\boldsymbol{h}_j=\alpha_i(\boldsymbol{h}_i-\boldsymbol{c})$. Raising score $i$ pulls the context toward $\boldsymbol{h}_i$, proportional to how much weight it already has. (b) Scores scale up, the softmax saturates to one-hot (hard attention) on the largest-score position, and $\alpha_i(\boldsymbol{h}_i-\boldsymbol{c})\to0$ for all $i$ since either $\alpha_i\to0$ or $\boldsymbol{c}\to\boldsymbol{h}_i$: the gradient vanishes. (c) If the entries of $\boldsymbol{s}$ and $\boldsymbol{h}_i$ are independent with zero mean and unit variance, $e_i$ has variance $d_k$; the scale grows with dimension, pushing the softmax into the saturated regime. Dividing by $\sqrt{d_k}$ restores unit variance.
:::
:::

:::exercise Exercise 9: Receptive field of a TCN (coding)
Write a function that returns the receptive field of a TCN with kernel size $k$, dilations $1,2,4,\dots,2^{L-1}$ and two convolutions per residual block, then find the smallest $L$ for $k=3$ that covers a history of 1,000 steps.
:::details Solution
Each convolution with dilation $d$ adds $(k-1)d$ to the receptive field, so with two per block: $\mathrm{RF}=1+2(k-1)\sum_{l=0}^{L-1}2^l=1+2(k-1)(2^L-1)$.
```python
def tcn_rf(k, L, convs_per_block=2):
    return 1 + convs_per_block * (k - 1) * (2 ** L - 1)

L = 1
while tcn_rf(3, L) < 1000:
    L += 1
print(L, tcn_rf(3, L))   # 8 blocks -> 1021
```
For $k=3$: $L=7$ gives $1+4\cdot127=509$, $L=8$ gives $1+4\cdot255=1021\ge1000$. So 8 blocks suffice, with a depth that grows only logarithmically in the history length.
:::
:::

:::exercise Exercise 10: Sequence labelling choice
For each task say whether you would use a unidirectional or bidirectional recurrent model, and which of many-to-one, many-to-many aligned, or encoder-decoder fits: (a) named entity recognition, (b) next-keystroke prediction on a phone, (c) sentiment of a movie review, (d) translating French to English, (e) transcribing speech with CTC.
:::details Solution
(a) Bidirectional, many-to-many aligned: the label of a word depends on both sides. (b) Unidirectional, autoregressive: only the past is available at prediction time. (c) Bidirectional encoder, many-to-one (pool or take final states). (d) Bidirectional encoder plus unidirectional attentional decoder (encoder-decoder), because the output length differs and generation is causal. (e) Bidirectional (or stacked) encoder with a CTC output layer for offline transcription, many-to-many with the alignment marginalized; streaming recognition must use unidirectional or limited-look-ahead layers.
:::
:::

## Key takeaways {#takeaways}

:::key Key takeaways
- A recurrent network applies the same cell at every step, carrying a state; unrolled, it is a deep network whose depth is the sequence length and whose layers share weights. BPTT is ordinary backprop on that graph, with gradients summed over time into the shared parameters.
- The gradient to a state $k$ steps back is a product of $k$ Jacobians; its size is governed by the spectral radius of $\mathbf{D}\mathbf{W}_{hh}$. Below 1 it vanishes (no long-range learning), above 1 it explodes. Clipping handles explosion; architecture handles vanishing.
- LSTM and GRU use additive, gated updates of the state so that the Jacobian along the memory path is a learned diagonal near 1 rather than a repeated dense matrix. Initialize the LSTM forget-gate bias to about 1.
- Truncated BPTT bounds memory and compute by detaching the state between chunks; the state carries forward but the gradient horizon is the chunk.
- Bidirectional and stacked RNNs enrich representations but a bidirectional model cannot be used to generate or to train a next-token objective.
- Seq2seq trains with teacher forcing and tests with self-feeding, which causes exposure bias; decode with beam search plus length normalization, and evaluate with free-running outputs.
- A single fixed vector is a bottleneck. Attention computes a weighted average of all encoder states with data-dependent weights, giving a short gradient path and an interpretable soft alignment. Bahdanau uses an additive MLP score, Luong a multiplicative one.
- 1D and temporal convolutions are parallel, stable alternatives with a receptive field that grows linearly (or exponentially with dilation); the next step, attention without recurrence, is [Week 9](week-09.html).
:::

## Further reading {#further-reading}

**Essential**

- Hochreiter, S. and Schmidhuber, J. (1997). *Long Short-Term Memory.* Neural Computation 9(8).
- Sutskever, I., Vinyals, O. and Le, Q. V. (2014). *Sequence to Sequence Learning with Neural Networks.* NeurIPS. arXiv:1409.3215.
- Bahdanau, D., Cho, K. and Bengio, Y. (2015). *Neural Machine Translation by Jointly Learning to Align and Translate.* ICLR. arXiv:1409.0473.

**Recommended**

- Goodfellow, I., Bengio, Y. and Courville, A. (2016). *Deep Learning*, Chapter 10: Sequence Modeling: Recurrent and Recursive Nets. MIT Press.
- Olah, C. (2015). *Understanding LSTM Networks.* colah.github.io.
- Bengio, Y., Simard, P. and Frasconi, P. (1994). *Learning long-term dependencies with gradient descent is difficult.* IEEE Transactions on Neural Networks 5(2).
- Pascanu, R., Mikolov, T. and Bengio, Y. (2013). *On the difficulty of training recurrent neural networks.* ICML. arXiv:1211.5063.
- Cho, K. et al. (2014). *Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation.* EMNLP. arXiv:1406.1078.
- Luong, M.-T., Pham, H. and Manning, C. D. (2015). *Effective Approaches to Attention-based Neural Machine Translation.* EMNLP. arXiv:1508.04025.
- Gers, F. A., Schmidhuber, J. and Cummins, F. (2000). *Learning to Forget: Continual Prediction with LSTM.* Neural Computation 12(10).
- Graves, A., Fernandez, S., Gomez, F. and Schmidhuber, J. (2006). *Connectionist Temporal Classification.* ICML.
- Bai, S., Kolter, J. Z. and Koltun, V. (2018). *An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling.* arXiv:1803.01271.
- Wu, Y. et al. (2016). *Google's Neural Machine Translation System.* arXiv:1609.08144.
- Bengio, S., Vinyals, O., Jaitly, N. and Shazeer, N. (2015). *Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks.* NeurIPS. arXiv:1506.03099.
