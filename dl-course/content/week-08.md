---
title: "Representation Learning, Embeddings, and Midterm"
phase: Core Architectures
summary: How to turn symbols, users and items into useful vectors. Distributed representations, word2vec (skip-gram, CBOW, negative sampling), GloVe and fastText, embedding geometry and bias, subword tokenization (BPE, WordPiece, Unigram), metric learning with contrastive, triplet and InfoNCE losses, plus a complete review of Weeks 1 to 7 and a practice midterm.
---

:::key Learning objectives
By the end of this week you should be able to:
- Explain why dense, learned, distributed representations beat one-hot and hand-built features, and state the distributional hypothesis.
- Derive the skip-gram objective with negative sampling from a binary-classification view, write its gradients, and implement it in NumPy and PyTorch.
- Contrast word2vec, GloVe and fastText, and state the sense in which skip-gram with negative sampling factorizes a shifted PMI matrix.
- Evaluate embeddings intrinsically (similarity, analogy) and extrinsically (a downstream task), and name the pitfalls of each.
- Measure and discuss social bias in embeddings with a projection test and WEAT, and say what debiasing can and cannot do.
- Implement byte-pair encoding from scratch and explain how WordPiece and SentencePiece/Unigram differ; reason about character- and byte-level models.
- View users, items, nodes and entities as embeddings, and see matrix factorization as an embedding model.
- Write and compare contrastive, triplet and InfoNCE losses, and explain the role of temperature and negatives.
- Revisit the course so far (Weeks 1 to 7), reproduce the key derivations, and complete a practice midterm.
:::

Up to now the inputs were already numbers: pixels, features, audio samples. Language, products, users and graph nodes are not. They are *discrete symbols*, and a neural network can only consume vectors. The bridge between the two is a **representation**: a map from each symbol to a point in $\R^d$ that the rest of the network can use. In [Week 7](week-07.html) you used an `nn.Embedding` layer without asking where its numbers come from. This week answers that question, and it turns out to be one of the central ideas of modern deep learning. Learn good vectors, and a simple classifier on top is often enough. This is the thread that runs from word2vec (2013) to the pretrained language models of [Weeks 9 to 11](week-10.html) and the contrastive image-text models you will meet later.

The week has two halves. The first covers representation learning: word vectors, their geometry and their biases, subword tokenization, embeddings of non-text objects, and metric learning. The second is a review of everything so far, a practice midterm, and the logistics for Assignment 2 and the final project proposal.

## Distributed representations {#distributed-representations}

Suppose your vocabulary has $V=50{,}000$ words. The most obvious representation is **one-hot**: word $i$ becomes the vector $\boldsymbol{e}_i\in\{0,1\}^V$. It is exact and it is useless for generalization. Every pair of distinct words is orthogonal and at the same distance, $\lVert\boldsymbol{e}_i-\boldsymbol{e}_j\rVert=\sqrt2$, so "cat" is exactly as far from "kitten" as it is from "carburetor". A model that learns something about "cat" learns nothing about "kitten". The input layer also has $V$ dimensions and grows with the vocabulary.

A **distributed representation** replaces this with a dense vector $\boldsymbol{v}_i\in\R^d$ with $d$ typically between 50 and 1000. "Distributed" means that a concept is spread across many coordinates and each coordinate takes part in representing many concepts, in contrast to a *local* (one-hot) code where one unit means one thing. Two properties follow.

1. **Similarity becomes geometry.** Words used in similar ways get nearby vectors, so what the model learns for one transfers to its neighbours.
2. **Compositional capacity.** $d$ real numbers can encode exponentially many distinguishable combinations of features, whereas $d$ one-hot units encode only $d$ items.

An embedding layer is mathematically trivial: with the embedding matrix $\mathbf{E}\in\R^{V\times d}$, the lookup for token $i$ is $\boldsymbol{e}_i^\top\mathbf{E}$, a row selection. It is the same as a linear layer applied to a one-hot input, implemented without ever materializing the one-hot vector. Everything interesting is in *how $\mathbf{E}$ is trained*.

### The distributional hypothesis {#distributional-hypothesis}

How can a machine learn what words mean from raw text, without labels? The linguist J. R. Firth's slogan (1957) is the standard answer: you shall know a word by the company it keeps. Words that occur in similar contexts tend to have similar meanings. This is the **distributional hypothesis**, and every method this week is an implementation of it. The only difference between methods is how "context" is defined and how the statistics are compressed into a low-dimensional vector.

The most direct implementation is count-based. Build a co-occurrence matrix $\mathbf{X}\in\R^{V\times V}$ where $X_{ij}$ counts how often word $j$ appears within a window of $m$ words of word $i$. Raw counts are dominated by frequent function words ("the", "of"), so reweight with **pointwise mutual information**,

$$
\mathrm{PMI}(i,j)=\log\frac{P(i,j)}{P(i)\,P(j)},
$$

which is positive when the two words co-occur more than independence predicts, and replace negative values by zero to obtain positive PMI (PPMI). Then compress the sparse $V\times V$ matrix with a truncated singular value decomposition, keeping the top $d$ singular directions: $\mathbf{X}_{\mathrm{PPMI}}\approx\mathbf{U}_d\boldsymbol{\Sigma}_d\mathbf{V}_d^\top$ and using rows of $\mathbf{U}_d\boldsymbol{\Sigma}_d^{1/2}$ as vectors. This is classical latent semantic analysis with a better weighting. The code below runs it on a tiny corpus so that you can see the mechanism end to end.

```python:ppmi_svd.py
import numpy as np

corpus = [
    "the cat chased the mouse", "the dog chased the cat", "the cat ate fish",
    "the dog ate meat", "a mouse ate cheese", "the king ruled the land",
    "the queen ruled the land", "the king loved the queen", "a dog loved a bone",
    "the cat loved fish", "the queen ate cheese", "the king ate meat",
]
tokens = [s.split() for s in corpus]
vocab = sorted({w for s in tokens for w in s})
idx = {w: i for i, w in enumerate(vocab)}
V, m = len(vocab), 2                                   # window: m words each side

X = np.zeros((V, V))
for s in tokens:
    for i, w in enumerate(s):
        for j in range(max(0, i - m), min(len(s), i + m + 1)):
            if j != i:
                X[idx[w], idx[s[j]]] += 1

total = X.sum()
p_ij = X / total
p_i = X.sum(1, keepdims=True) / total
p_j = X.sum(0, keepdims=True) / total
with np.errstate(divide="ignore"):
    pmi = np.log(p_ij / (p_i * p_j))
ppmi = np.maximum(pmi, 0)                              # -inf (never co-occur) and negatives -> 0

U, S, Vt = np.linalg.svd(ppmi)
d = 3
emb = U[:, :d] * np.sqrt(S[:d])                        # (V, d) word vectors
emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9

def neighbours(w, k=3):
    sims = emb @ emb[idx[w]]
    return [(vocab[i], round(float(sims[i]), 2)) for i in np.argsort(-sims)[1:k + 1]]

print("vocab size", V, "| matrix", ppmi.shape, "| embedding", emb.shape)
print("king  ->", neighbours("king"))
print("cat   ->", neighbours("cat"))
```

Expect output like `king -> [('queen', 1.0), ('ate', 0.98), ('cat', 0.93)]`. The corpus is far too small for the neighbours to be reliable (twelve sentences, three dimensions: "king" and "ate" being close is an artefact), but the structure shows through: "king" and "queen" share contexts (ruled, loved, the land) and come out as the nearest pair. Count-based methods have a clear cost, though. The matrix is $V\times V$, the SVD does not scale to a 1M-word vocabulary or stream over new data, and the whole pipeline is fixed rather than trained end to end for a task. **Prediction-based** methods, led by word2vec, learn the same kind of vectors with SGD on a simple prediction task and scale effortlessly. Levy and Goldberg later showed (derived below) that the two families are closer than they look.

:::history Neural language models and the 2013 breakthrough
Bengio, Ducharme, Vincent and Jauvin (2003) trained the first neural language model with a learned embedding table, but the network was expensive and the vectors a by-product. Collobert and Weston (2008) and Collobert et al. (2011) showed that embeddings trained on unlabeled text help many tagging tasks. Mikolov and colleagues at Google (2013) then stripped the model down to almost nothing (no hidden nonlinearity, a cheap output approximation) so that it could be trained on billions of words in hours. The speed, together with the surprising linear structure of the resulting space, made word2vec the first widely used pretrained representation in NLP.
:::

## word2vec {#word2vec}

Word2vec is not one model but a family of two architectures and two output approximations. All of them learn *two* embedding tables of size $V\times d$: input vectors $\boldsymbol{v}_w$ (used when $w$ is the centre or the "query") and output vectors $\boldsymbol{u}_w$ (used when $w$ is a target or a "context"). After training, people usually keep $\boldsymbol{v}_w$, or the average $(\boldsymbol{v}_w+\boldsymbol{u}_w)/2$.

{{figure: week08-word2vec-arch | The two word2vec architectures on the same window. Skip-gram (left) predicts every context word from the centre word, producing up to $2m$ training pairs per position. CBOW (right) averages the context embeddings and predicts the centre word once.}}

### Skip-gram {#skip-gram}

Slide a window of radius $m$ over a corpus $w_1,\dots,w_T$. At position $t$, the word $w_t$ is the **centre** $c$ and each of $w_{t+j}$ ($0<|j|\le m$) is a **context** word $o$. Skip-gram models the probability of a context word given the centre as a softmax over the whole vocabulary,

$$
P(o\mid c)=\frac{\exp(\boldsymbol{u}_o^\top\boldsymbol{v}_c)}{\sum_{w=1}^{V}\exp(\boldsymbol{u}_w^\top\boldsymbol{v}_c)},
$$

and maximizes the average log-likelihood of all (centre, context) pairs,

$$
\mathcal{L}_{\mathrm{SG}}=-\frac1T\sum_{t=1}^{T}\ \sum_{\substack{-m\le j\le m\\ j\ne0}}\log P(w_{t+j}\mid w_t).
$$

Reading the dot product $\boldsymbol{u}_o^\top\boldsymbol{v}_c$ as a compatibility score, training pushes the vectors of words that co-occur to have a large inner product. Two words that appear in the same contexts (cat and dog both co-occur with "pet", "fed", "barked at" or "purred at") are pulled toward the same context vectors and therefore end up close to each other. The model is a *log-bilinear* classifier: there is no hidden layer and no nonlinearity, which is why it is fast and why its geometry is so regular.

The catch is the denominator. Each pair costs $O(Vd)$ to evaluate and to differentiate. With $V=10^6$ and $d=300$ that is $3\times10^8$ multiply-adds per training pair, and a corpus has billions of pairs. Two remedies were proposed: **hierarchical softmax** (a binary tree over the vocabulary, $O(\log V)$ per pair) and **negative sampling**, which is the one that took over.

### CBOW {#cbow}

The **continuous bag of words** model flips the direction: it averages the input vectors of the context words and predicts the centre word,

$$
\bar{\boldsymbol{v}}=\frac1{2m}\sum_{0<|j|\le m}\boldsymbol{v}_{w_{t+j}},\qquad P(w_t\mid\text{context})=\softmax_w\!\big(\boldsymbol{u}_w^\top\bar{\boldsymbol{v}}\big).
$$

Because it makes one prediction per position (rather than up to $2m$), CBOW is faster per epoch, and averaging smooths over noise, which helps frequent words. Skip-gram sees each rare word many times as the centre with a fresh target each time, and in practice gives better vectors for rare words and small corpora. The common rule of thumb (a rule, not a theorem) is skip-gram for small data and rare words, CBOW for speed.

### Negative sampling: the objective, derived {#negative-sampling}

Negative sampling replaces the $V$-way classification with $K+1$ *binary* classifications. The idea is to teach the model to distinguish real (centre, context) pairs from fake ones. For a real pair $(c,o)$ you draw $K$ "noise" words $n_1,\dots,n_K$ from a noise distribution $P_n$ and train the logistic classifier

$$
P(D=1\mid c,w)=\sigma(\boldsymbol{u}_w^\top\boldsymbol{v}_c),\qquad \sigma(x)=\frac1{1+e^{-x}},
$$

to output 1 for the observed pair and 0 for each noise pair. Writing the negative log-likelihood of those $K+1$ labels gives the per-pair loss.

:::math Negative sampling loss and its gradients
For one observed pair $(c,o)$ with noise words $n_1,\dots,n_K$, the cross-entropy of the labels $(1,0,\dots,0)$ is

$$
J=-\log\sigma(\boldsymbol{u}_o^\top\boldsymbol{v}_c)-\sum_{k=1}^{K}\log\sigma(-\boldsymbol{u}_{n_k}^\top\boldsymbol{v}_c),
$$

where we used $1-\sigma(x)=\sigma(-x)$. Use $\sigma'(x)=\sigma(x)(1-\sigma(x))$, which gives $\frac{d}{dx}\log\sigma(x)=1-\sigma(x)=\sigma(-x)$. Define the scores $s_o=\boldsymbol{u}_o^\top\boldsymbol{v}_c$ and $s_k=\boldsymbol{u}_{n_k}^\top\boldsymbol{v}_c$. Then

$$
\begin{aligned}
\frac{\partial J}{\partial s_o}&=-(1-\sigma(s_o))=\sigma(s_o)-1,\\
\frac{\partial J}{\partial s_k}&=\sigma(s_k),
\end{aligned}
$$

and by the chain rule

$$
\begin{aligned}
\frac{\partial J}{\partial\boldsymbol{u}_o}&=(\sigma(s_o)-1)\,\boldsymbol{v}_c,\\
\frac{\partial J}{\partial\boldsymbol{u}_{n_k}}&=\sigma(s_k)\,\boldsymbol{v}_c,\\
\frac{\partial J}{\partial\boldsymbol{v}_c}&=(\sigma(s_o)-1)\,\boldsymbol{u}_o+\sum_{k=1}^{K}\sigma(s_k)\,\boldsymbol{u}_{n_k}.
\end{aligned}
$$

Intuition: the true context vector is pulled toward $\boldsymbol{v}_c$ with a force that shrinks as the model gets it right ($\sigma(s_o)\to1$); every noise vector is pushed away with a force proportional to how wrongly the model likes it. Only $K+2$ rows of the two tables are touched per pair, so the cost is $O(Kd)$ regardless of $V$.
:::

Typical values are $K=5$ to $20$ for small corpora and $K=2$ to $5$ for very large ones. The noise distribution matters. Sampling from the raw unigram distribution $f(w)$ gives too many "the" and "of" as negatives, which are uninformative; sampling uniformly over-represents rare words. Mikolov et al. used the **unigram distribution raised to the 3/4 power**,

$$
P_n(w)=\frac{f(w)^{3/4}}{\sum_{w'}f(w')^{3/4}},
$$

an empirical choice that flattens the head of the Zipf distribution without making it uniform. The chart shows the effect for a Zipfian vocabulary of 1000 words.

```chart
{
  "type": "line",
  "caption": "Noise distribution P_n(w) proportional to f(w)^alpha for the 20 most frequent words of an illustrative Zipfian vocabulary (1000 words, f(r) proportional to 1/r), computed with NumPy. alpha = 0.75 is the word2vec choice. Under alpha = 1 the top word alone supplies 13% of all negatives; under 0.75 it supplies 5%.",
  "height": 330,
  "data": {
    "labels": [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],
    "datasets": [
      {"label": "alpha = 1 (raw unigram)", "data": [0.13359,0.0668,0.04453,0.0334,0.02672,0.02227,0.01908,0.0167,0.01484,0.01336,0.01214,0.01113,0.01028,0.00954,0.00891,0.00835,0.00786,0.00742,0.00703,0.00668], "borderColor": "var(--c-red)", "backgroundColor": "var(--c-red)", "pointRadius": 3},
      {"label": "alpha = 0.75 (word2vec)", "data": [0.05248,0.0312,0.02302,0.01855,0.01569,0.01369,0.01219,0.01103,0.0101,0.00933,0.00869,0.00814,0.00767,0.00725,0.00689,0.00656,0.00627,0.00601,0.00577,0.00555], "borderColor": "var(--c-accent)", "backgroundColor": "var(--c-accent)", "pointRadius": 3},
      {"label": "alpha = 0 (uniform)", "data": [0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001], "borderColor": "var(--c-blue)", "backgroundColor": "var(--c-blue)", "pointRadius": 3}
    ]
  },
  "options": {"scales": {"x": {"title": {"display": true, "text": "word frequency rank"}}, "y": {"type": "logarithmic", "title": {"display": true, "text": "probability of being drawn as a negative"}}}}
}
```

{{figure: week08-negative-sampling | Negative sampling for one training pair. The centre vector $\boldsymbol{v}_c$ is scored against one true context vector and $K$ noise vectors, each by a sigmoid of a dot product. Only these $K+2$ rows are updated.}}

{{demo: week08-skipgram-window | Skip-gram windows and negative sampling | 520}}

Two more details of the original recipe are worth knowing, because they matter for quality as much as the loss does.

- **Frequent-word subsampling.** A word $w$ with corpus frequency $f(w)$ is discarded from a training window with probability $1-\sqrt{t/f(w)}$ (for $f(w)>t$), where $t\approx10^{-5}$. This throws away much of the redundant "the ... of" signal, speeds up training, and improves the vectors for mid-frequency words.
- **Dynamic window.** For each centre word, the effective radius is sampled uniformly from $1..m$, so nearer words are used more often than distant ones.

:::note What about the two tables?
Why keep both $\mathbf{V}$ (input) and $\mathbf{U}$ (output)? Using one table would force $\boldsymbol{v}_w^\top\boldsymbol{v}_w$, the score of a word with itself, to appear in the loss whenever $w$ is its own noise sample, and it couples the roles of "being described" and "describing". Separate tables are simpler to optimize. The output table is initialized to zero in the reference implementation, and the input table to small random values.
:::

### A skip-gram you can run: NumPy implementation {#skipgram-numpy}

To make the gradients concrete, here is a complete NumPy implementation that follows the derivation above exactly (mini-batched, with `np.add.at` for the scatter-updates). It trains on a small synthetic corpus with four topical groups (animals, food, tools, cities), so that you can check that nearest neighbours recover the groups. The loss and neighbours reported are from an actual run.

```python:sgns_numpy.py
import numpy as np
rng = np.random.default_rng(0)

# ---- a synthetic corpus with four topics, each with its own nouns, verbs and adjectives ----
groups = {"animals": "cat dog horse cow sheep bird fish".split(),
          "food":    "bread cheese apple rice soup pasta cake".split(),
          "tools":   "hammer saw drill wrench knife shovel axe".split(),
          "cities":  "paris berlin rome madrid vienna lisbon oslo".split()}
verbs = {"animals": "runs sleeps eats hunts".split(), "food": "tastes cooks burns spoils".split(),
         "tools": "cuts breaks hangs rusts".split(), "cities": "grows hosts attracts needs".split()}
adjs  = {"animals": "wild small hungry".split(), "food": "fresh hot sweet".split(),
         "tools": "heavy sharp old".split(), "cities": "large old busy".split()}
det = "the a one this".split()
sents = []
for _ in range(6000):
    g = rng.choice(list(groups)); n, v, a = rng.choice(groups[g]), rng.choice(verbs[g]), rng.choice(adjs[g])
    sents.append([rng.choice(det), a, n, v] if rng.random() < .5 else [rng.choice(det), n, v, "today"])

vocab = sorted({w for s in sents for w in s}); idx = {w: i for i, w in enumerate(vocab)}; V = len(vocab)
cnt = np.zeros(V)
for s in sents:
    for w in s: cnt[idx[w]] += 1

# ---- (centre, context) pairs from a window of radius 2 ----
pairs = []
for s in sents:
    ids = [idx[w] for w in s]
    for i, c in enumerate(ids):
        for j in range(max(0, i - 2), min(len(ids), i + 3)):
            if j != i: pairs.append((c, ids[j]))
pairs = np.array(pairs)                                  # (num_pairs, 2)

d, K, lr, B = 16, 5, 0.05, 256
W_in  = (rng.random((V, d)) - .5) / d                    # input vectors  v_w
W_out = np.zeros((V, d))                                 # output vectors u_w
p_noise = cnt ** .75; p_noise /= p_noise.sum()           # unigram^(3/4)
sig = lambda x: 1 / (1 + np.exp(-x))

losses = []
for epoch in range(30):
    rng.shuffle(pairs); total = 0.0
    for b in range(0, len(pairs), B):
        P = pairs[b:b + B]; c, o = P[:, 0], P[:, 1]
        neg = rng.choice(V, size=(len(P), K), p=p_noise)        # (B, K) noise words
        vc, uo, un = W_in[c], W_out[o], W_out[neg]              # (B,d) (B,d) (B,K,d)
        s_pos = (vc * uo).sum(1)                                # (B,)
        s_neg = np.einsum("bd,bkd->bk", vc, un)                 # (B,K)
        total += (-np.log(sig(s_pos) + 1e-9) - np.log(sig(-s_neg) + 1e-9).sum(1)).sum()
        g_pos = (sig(s_pos) - 1)[:, None]                       # dJ/ds_o
        g_neg = sig(s_neg)                                      # dJ/ds_k
        d_vc = g_pos * uo + np.einsum("bk,bkd->bd", g_neg, un)  # dJ/dv_c
        np.add.at(W_in, c, -lr * d_vc)
        np.add.at(W_out, o, -lr * g_pos * vc)
        np.add.at(W_out, neg.reshape(-1), -lr * (g_neg[:, :, None] * vc[:, None, :]).reshape(-1, d))
    losses.append(total / len(pairs))
    if epoch % 5 == 0: print(f"epoch {epoch:2d}  loss per pair {losses[-1]:.3f}")

E = W_in / np.linalg.norm(W_in, axis=1, keepdims=True)
def nn(w, k=4):
    s = E @ E[idx[w]]
    return [(str(vocab[i]), round(float(s[i]), 2)) for i in np.argsort(-s)[1:k + 1]]
for w in ["cat", "paris", "hammer", "cheese"]:
    print(w, "->", nn(w))
```

```text
epoch  0  loss per pair 2.476
epoch  5  loss per pair 2.322
...
cat    -> [('fish', 0.98), ('bird', 0.97), ('sheep', 0.97), ('horse', 0.96)]
paris  -> [('rome', 0.97), ('lisbon', 0.97), ('madrid', 0.96), ('oslo', 0.94)]
hammer -> [('saw', 0.98), ('drill', 0.97), ('knife', 0.97), ('axe', 0.96)]
cheese -> [('soup', 0.98), ('bread', 0.97), ('pasta', 0.97), ('cake', 0.96)]
```

The neighbours recover the four topics perfectly even though no word was ever labelled. The loss flattens early at about 2.3, and that is not a failure: the sentence structure makes the exact context word inherently unpredictable (any of seven nouns fits a slot), so the best achievable loss is far above zero. What the vectors capture is *which words are interchangeable*, and that is exactly the distributional hypothesis at work.

```chart
{
  "type": "line",
  "caption": "Skip-gram negative-sampling loss per pair (K = 5, d = 16) over 30 epochs on the synthetic corpus above (60,000 training pairs, 60-word vocabulary), from the NumPy code. The loss drops quickly and then plateaus at the noise floor set by the unpredictability of the corpus.",
  "height": 300,
  "data": {
    "labels": [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30],
    "datasets": [{"label": "NS loss per pair", "data": [2.4757, 2.3261, 2.322, 2.3172, 2.307, 2.3225, 2.3038, 2.3156, 2.3116, 2.3082, 2.3246, 2.3039, 2.3203, 2.3111, 2.3027, 2.3113, 2.3151, 2.318, 2.3174, 2.317, 2.3131, 2.3142, 2.3067, 2.3184, 2.3194, 2.3221, 2.3018, 2.314, 2.3073, 2.3093], "borderColor": "var(--c-accent)", "pointRadius": 0, "tension": 0.2}]
  },
  "options": {"scales": {"x": {"title": {"display": true, "text": "epoch"}}, "y": {"title": {"display": true, "text": "loss per (centre, context) pair"}, "min": 2.2}}}
}
```

### What is skip-gram really optimizing? {#pmi-factorization}

Negative sampling looks like a heuristic, but its optimum has a clean closed form, found by Levy and Goldberg (2014). Group the pairs by type. Let $\#(w,c)$ be the number of times the pair occurs in the training data, $\#(w)=\sum_c\#(w,c)$ and $|D|$ the total pair count. Here I follow the usual convention in that analysis, in which the word $w$ plays the role of the "centre" and $c$ of the context.

:::math The optimum of SGNS is a shifted PMI matrix
For a fixed pair $(w,c)$ with score $x=\boldsymbol{u}_c^\top\boldsymbol{v}_w$, the expected contribution to the (negated) objective is

$$
\ell(x)=\#(w,c)\log\sigma(x)+K\,\#(w)\,P_n(c)\,\log\sigma(-x),
$$

since the pair appears as a positive example $\#(w,c)$ times and as a noise example about $K\,\#(w)P_n(c)$ times. Maximize over $x$ (treating $x$ as free, which holds when $d$ is large enough):

$$
\ell'(x)=\#(w,c)\,\sigma(-x)-K\,\#(w)P_n(c)\,\sigma(x)=0
\ \Longrightarrow\ e^{x}=\frac{\#(w,c)}{K\,\#(w)\,P_n(c)}.
$$

If the noise distribution is the unigram distribution, $P_n(c)=\#(c)/|D|$, this becomes

$$
x^\star=\boldsymbol{u}_c^\top\boldsymbol{v}_w=\log\frac{\#(w,c)\,|D|}{\#(w)\,\#(c)}-\log K=\mathrm{PMI}(w,c)-\log K.
$$

So skip-gram with negative sampling implicitly factorizes the PMI matrix shifted by $\log K$, with the factors $\mathbf{V}\mathbf{U}^\top$ playing the role of the SVD above. (With $P_n\propto f^{3/4}$ the PMI is computed against the smoothed noise distribution.)
:::

Two consequences. First, count-based and prediction-based embeddings are two algorithms for the same family of objectives, and careful count-based pipelines (with the right weighting and smoothing) match word2vec on many benchmarks. The popular claim that "neural embeddings are magic and counts are not" does not survive this analysis. Second, pairs that never co-occur have $\mathrm{PMI}=-\infty$, but SGNS never sees an infinite target: its finite training sample keeps scores bounded. That is one reason it outperforms naive unshifted PMI factorization.

### GloVe: global vectors {#glove}

Pennington, Socher and Manning (2014) took the count-based view and made it a direct regression problem. Their observation: *ratios* of co-occurrence probabilities carry meaning. For probe words $k$ and a pair "ice" and "steam", $P(k\mid\text{ice})/P(k\mid\text{steam})$ is large for $k=$ "solid", small for $k=$ "gas" and near 1 for "water" or "fashion". They fit vectors so that dot products model *log* co-occurrence counts with a weighted least-squares loss,

$$
J=\sum_{i,j=1}^{V}f(X_{ij})\Big(\boldsymbol{w}_i^\top\tilde{\boldsymbol{w}}_j+b_i+\tilde b_j-\log X_{ij}\Big)^2,\qquad f(x)=\begin{cases}(x/x_{\max})^{3/4}&x<x_{\max}\\ 1&\text{otherwise,}\end{cases}
$$

with $x_{\max}=100$ in the paper. Only pairs with $X_{ij}>0$ enter the sum (so $\log X_{ij}$ is defined), and $f$ downweights rare, noisy pairs and caps the influence of very frequent ones. Training iterates over the nonzero entries of a pre-computed matrix instead of over the raw corpus, which makes each epoch cheap once $\mathbf{X}$ exists. Compare with the SGNS result above: GloVe regresses on $\log X_{ij}$ (plus biases that absorb the marginal terms of the PMI), SGNS on shifted PMI implicitly. In practice the two give vectors of similar quality, and published comparisons disagree about which wins, because both depend heavily on hyperparameters (window, subsampling, dimension, number of epochs). GloVe's final vector for a word is usually $\boldsymbol{w}_i+\tilde{\boldsymbol{w}}_i$.

### fastText: embeddings with subwords {#fasttext}

Word2vec and GloVe assign each word type an independent vector. That fails for morphologically rich languages (Finnish, Turkish and German have many inflected forms, most of which are rare) and for any word never seen in training: the model has no vector for "unfriendliest" even if it has seen "friend", "friendly" and "unfriendly". Bojanowski, Grave, Joulin and Mikolov (2017), the fastText embeddings, fix this by representing a word as a bag of its character $n$-grams. With boundary markers, the word "where" with $n=3$ becomes `<wh, whe, her, ere, re>` plus the whole word token `<where>`. The word vector is the sum of the vectors of its $n$-grams ($n$ from 3 to 6 by default, hashed into a fixed number of buckets):

$$
\boldsymbol{v}_w=\sum_{g\in\mathcal{G}_w}\boldsymbol{z}_g,
$$

and the same skip-gram negative-sampling loss is applied with this composed vector in place of $\boldsymbol{v}_w$. The payoff: rare and unseen words get a sensible vector from their pieces (suffixes like "-ness", prefixes like "un-"), and related word forms share parameters. The costs are a larger model (the $n$-gram table) and slower training. This is the first appearance of an idea that dominates the second half of this week: *do not make the word the atomic unit*.

### Training your own versus using pretrained {#pretrained-vs-own}

Training embeddings is cheap, but pretrained vectors (GloVe on Wikipedia and Gigaword, fastText on Common Crawl, word2vec on Google News) are trained on billions of tokens you do not have. For a small labelled dataset, initializing an embedding layer from pretrained vectors almost always helps. For a large dataset or a specialized domain (clinical notes, source code, legal text), vectors trained on in-domain text can win, and fine-tuning the table on the task is the standard compromise. All of these are *static* embeddings: one vector per word type, regardless of the sentence it appears in. "Bank" in "river bank" and in "bank loan" gets a single vector that blends both senses. **Contextual** embeddings (ELMo, 2018; BERT, 2018), where the vector depends on the whole sentence, remove that limit and are the subject of [Weeks 10 and 11](week-10.html). Everything in this week still applies to their *input* layer.

## Embedding geometry, analogies and evaluation {#geometry-evaluation}

### Cosine similarity and nearest neighbours {#cosine}

The standard way to compare two vectors is **cosine similarity**,
$\cos(\boldsymbol{a},\boldsymbol{b})=\boldsymbol{a}^\top\boldsymbol{b}/(\lVert\boldsymbol{a}\rVert\lVert\boldsymbol{b}\rVert)$. Raw dot products are dominated by vector length, and in word2vec-style models the length correlates with word frequency, so frequent words win just by being long. Cosine compares directions only; after normalizing every row of the embedding matrix to unit length, a matrix product gives all cosines at once. Always normalize before looking at neighbours.

```python:nearest.py
import numpy as np

def normalize(E, eps=1e-9):
    """Row-normalize an embedding matrix of shape (V, d)."""
    return E / (np.linalg.norm(E, axis=1, keepdims=True) + eps)

def most_similar(E_norm, vocab, idx, positive, negative=(), k=5):
    """Nearest neighbours of sum(positive) - sum(negative) by cosine. Inputs are excluded from the answer."""
    q = sum(E_norm[idx[w]] for w in positive) - sum(E_norm[idx[w]] for w in negative)
    q = q / (np.linalg.norm(q) + 1e-9)
    sims = E_norm @ q                                     # (V,) cosines
    banned = {idx[w] for w in list(positive) + list(negative)}
    order = [i for i in np.argsort(-sims) if i not in banned][:k]
    return [(vocab[i], float(sims[i])) for i in order]

# Usage (with a trained matrix E, vocab list and idx dict):
#   En = normalize(E)
#   most_similar(En, vocab, idx, positive=["king", "woman"], negative=["man"])      # analogy
#   most_similar(En, vocab, idx, positive=["paris"])                                  # neighbours
```

### Linear structure and the analogy test {#analogies}

The result that made word2vec famous is that relations between words appear as roughly constant *offsets*: $\boldsymbol{v}_{\text{king}}-\boldsymbol{v}_{\text{man}}+\boldsymbol{v}_{\text{woman}}\approx\boldsymbol{v}_{\text{queen}}$, and $\boldsymbol{v}_{\text{paris}}-\boldsymbol{v}_{\text{france}}+\boldsymbol{v}_{\text{germany}}\approx\boldsymbol{v}_{\text{berlin}}$. The analogy task "$a$ is to $b$ as $c$ is to $?$" is answered by

$$
d^\star=\argmax_{w\notin\{a,b,c\}}\ \cos\big(\boldsymbol{v}_w,\ \boldsymbol{v}_b-\boldsymbol{v}_a+\boldsymbol{v}_c\big).
$$

{{figure: week08-geometry | The parallelogram picture of analogies, with hand-drawn illustrative positions. If "woman minus man" and "queen minus king" are nearly the same offset, then adding that offset to king lands near queen. Capital-of is another approximately constant offset.}}

Why should this happen at all? The PMI derivation above gives the beginning of an answer. If $\boldsymbol{u}_c^\top\boldsymbol{v}_w\approx\mathrm{PMI}(w,c)-\log K$ and the PMI difference between "king" and "queen" is similar across context words $c$ for every context that distinguishes gender (she/he, her/his, mother/father...) and negligible for the others (ruled, crown, palace), then $\boldsymbol{v}_{\text{king}}-\boldsymbol{v}_{\text{queen}}$ must be nearly the same vector as $\boldsymbol{v}_{\text{man}}-\boldsymbol{v}_{\text{woman}}$ when tested against all contexts. Explicit regularities in co-occurrence statistics become linear regularities in the embedding.

Try the explorer below. It uses a hand-built 2D "embedding" so that you can see the geometry, including how the answer depends on excluding the input words and how quickly noise destroys a parallelogram.

{{demo: week08-analogy-explorer | Analogy explorer on a hand-built 2D embedding | 560}}

:::warning Analogies are a demo, not a benchmark of understanding
Three caveats that every practitioner should know. (1) The three input words are excluded from the candidate list; without that exclusion the nearest neighbour of $\boldsymbol{v}_b-\boldsymbol{v}_a+\boldsymbol{v}_c$ is frequently $b$ or $c$ itself, so part of the apparent success comes from the exclusion rule. Try it in the demo above. (2) Many relations are not one-to-one or not linear (the offset for "country to language" is not constant across countries), and accuracy drops sharply on the harder relations. (3) Results depend on frequency and on the particular dataset and are sensitive to hyperparameters. Levy and Goldberg (2014) introduced a multiplicative variant (3CosMul) and Linzen (2016) discussed the limits of the offset method. Treat analogies as a sanity check of the geometry, not as proof that embeddings "understand".
:::

### How to evaluate embeddings {#evaluating-embeddings}

There are two families of evaluation, and you need both.

**Intrinsic evaluation** probes the vectors directly.
- *Word similarity.* Human raters score pairs of words for similarity; you compute the Spearman rank correlation between the cosine similarities and the human scores. Classic datasets are WordSim-353 (Finkelstein et al., 2002), which mixes similarity and relatedness (coffee and cup are related, not similar), and SimLex-999 (Hill, Reichart and Korhonen, 2015), which was designed to measure true similarity. Because they are small, differences of a point or two are often noise.
- *Analogy accuracy* on the Google analogy set of Mikolov et al. (2013), with semantic (capital-country) and syntactic (tense, plural) questions.
- *Clustering and outlier detection:* do the neighbours of a word belong to the same category?

**Extrinsic evaluation** plugs the embeddings into a downstream model (a sentiment classifier, a tagger, a retrieval system) and measures task performance. This is the number that matters in practice, and the two kinds of evaluation do not always agree: embeddings that do well on similarity benchmarks are not guaranteed to help a classifier. The lab at the end of this week does both.

Other things worth checking by hand: neighbours of rare words (often noise), the effect of frequency (very frequent words are often "hubs" that appear in many neighbour lists), and whether vectors trained with different random seeds give the same neighbours. They usually agree on the closest few and diverge afterwards, so be careful with strong claims from a single run.

## Bias in embeddings {#embedding-bias}

Embeddings are fitted to text written by people, and text reflects the world, including its stereotypes and inequities. Bolukbasi et al. (2016) showed that word2vec vectors trained on news text encode gendered associations: the analogy "man is to computer programmer as woman is to ..." completes with homemaker, and occupations such as nurse or engineer sit measurably on one side of a "he minus she" direction. Caliskan, Bryson and Narayanan (2017) showed that associations between target words (flowers vs insects, European-American vs African-American names) and attribute words (pleasant vs unpleasant) in GloVe vectors closely mirror the classic human Implicit Association Test results. The embeddings did not invent these associations; they are compressed statistics of how words are used. But a model built on them can *replicate and amplify* them in applications such as résumé screening, search ranking and autocomplete.

### Measuring bias: projections and WEAT {#measuring-bias}

Two measurements are standard, and both are short to implement.

**Direction projection.** Estimate a "gender direction" $\boldsymbol{g}$ from definitional pairs (he-she, man-woman, his-her, ...), e.g. as the top principal component of the pair differences or simply the mean difference. Then the projection $\cos(\boldsymbol{v}_w,\boldsymbol{g})$ of a word that should be neutral with respect to gender (a profession) measures its lean along that axis.

**WEAT** (the Word Embedding Association Test of Caliskan et al., modelled on the IAT) takes two target sets $X,Y$ (for example career-related and family-related words) and two attribute sets $A,B$ (male and female terms). Define the association of a word with the attributes as $s(w,A,B)=\mathrm{mean}_{a\in A}\cos(\boldsymbol{v}_w,\boldsymbol{v}_a)-\mathrm{mean}_{b\in B}\cos(\boldsymbol{v}_w,\boldsymbol{v}_b)$. The effect size is

$$
d=\frac{\mathrm{mean}_{x\in X}\,s(x,A,B)-\mathrm{mean}_{y\in Y}\,s(y,A,B)}{\mathrm{std}_{w\in X\cup Y}\,s(w,A,B)}.
$$

```python:bias_probes.py
import numpy as np

def unit(v):
    return v / (np.linalg.norm(v) + 1e-9)

def assoc(w, A, B):
    """s(w, A, B): mean cosine to attribute set A minus mean cosine to attribute set B."""
    w = unit(w)
    return np.mean([w @ unit(a) for a in A]) - np.mean([w @ unit(b) for b in B])

def weat_effect_size(X, Y, A, B):
    """Cohen's-d style effect size of the WEAT test; X, Y, A, B are lists of vectors."""
    sx = [assoc(x, A, B) for x in X]
    sy = [assoc(y, A, B) for y in Y]
    return (np.mean(sx) - np.mean(sy)) / (np.std(sx + sy, ddof=1) + 1e-12)

def weat_permutation_p(X, Y, A, B, n_perm=10000, seed=0):
    """One-sided permutation test: how often a random re-split of X u Y gives a statistic at least as large."""
    rng = np.random.default_rng(seed)
    allw = list(X) + list(Y)
    s = np.array([assoc(w, A, B) for w in allw])
    n = len(X)
    observed = s[:n].sum() - s[n:].sum()
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(len(allw))
        count += (s[perm[:n]].sum() - s[perm[n:]].sum()) >= observed
    return count / n_perm

def gender_direction(E, idx, pairs=(("he", "she"), ("man", "woman"), ("his", "her"), ("father", "mother"))):
    """Mean of difference vectors for definitional pairs; unit length."""
    return unit(np.mean([E[idx[a]] - E[idx[b]] for a, b in pairs], axis=0))

def projection_table(E, idx, words, g):
    """Cosine of each word's vector with the gender direction (positive = leans toward the first element of each pair)."""
    return sorted(((w, float(unit(E[idx[w]]) @ g)) for w in words), key=lambda t: t[1])

# Usage with a trained or pretrained embedding matrix E and a word->row dict idx:
#   g = gender_direction(E, idx)
#   projection_table(E, idx, ["nurse", "engineer", "teacher", "programmer", "homemaker"], g)
#   X = [E[idx[w]] for w in career_words]; Y = [E[idx[w]] for w in family_words]
#   A = [E[idx[w]] for w in male_words];   B = [E[idx[w]] for w in female_words]
#   print(weat_effect_size(X, Y, A, B), weat_permutation_p(X, Y, A, B))
```

### Discussion: what to conclude, and what not to {#bias-discussion}

A few points, stated carefully because this topic is easy to oversimplify in both directions.

- **A measurement is not a verdict on the model's use.** Showing that a vector space has a gender direction tells you that the *training text* has statistical regularities along gender. Whether that causes harm depends on the application. A search system that retrieves "nurse" results for "she" and "engineer" results for "he" is more problematic than a topic model that merely reports the association.
- **Bias is multi-dimensional and the word lists matter.** WEAT results depend on the specific target and attribute sets, which are small, culturally specific and in English. A non-significant result on a particular list does not demonstrate absence of bias, and a significant one does not tell you which downstream behaviour will result. Report the lists with the numbers.
- **Debiasing is partial.** The hard-debiasing approach of Bolukbasi et al. projects out the gender direction from gender-neutral words. Gonen and Goldberg (2019) showed that this mostly hides the bias in the geometry: after projection the words that were biased are still clustered together and can still be recovered by a classifier, so the neighbourhood structure still encodes the original association. Removing a direction removes one linear measurement, not the underlying statistics.
- **Representations are only one source.** Data selection, labelling, objective and deployment all contribute. Embedding audits are a useful diagnostic and a low-cost one; they do not replace evaluation of the full system on the population it will serve, with disaggregated metrics.

Practical stance for this course: measure, report what you measured and its limits, avoid claims of "debiased", and think about who is affected before deploying. We will return to this in the discussion of evaluation and responsible deployment in the final weeks.

## Subword tokenization {#subword-tokenization}

Every model so far needed a decision about the *unit* of text. Words are the natural choice but have two problems. The vocabulary is open: new words, names, typos, numbers and inflections keep arriving, so any fixed word list leaves some inputs as `<unk>`, and an `<unk>` token throws away all information. And the softmax over a 500,000-word vocabulary is expensive and statistically thin, since most words are rare. Characters have the opposite profile. The vocabulary is tiny and nothing is ever unknown, but sequences become 4 to 6 times longer, the model must spend capacity re-learning spelling, and long-range dependencies stretch over more steps (see [Week 7](week-07.html) on why that hurts recurrent models; attention costs grow quadratically with length too).

**Subword tokenization** sits between the two. Frequent words stay whole ("the", "language"); rarer words are split into reusable pieces ("un", "friend", "li", "est"); and in the worst case everything falls back to single characters or bytes, so nothing is unknown. The vocabulary size (typically 8k to 100k pieces) is a hyperparameter you pick.

### Byte-pair encoding (BPE) {#bpe}

BPE was invented as a data-compression scheme by Philip Gage (1994), which repeatedly replaces the most frequent pair of adjacent bytes by a new symbol. Sennrich, Haddow and Birch (2016) adapted it to word segmentation for neural machine translation and showed that it handles rare and unseen words well. The learning algorithm is almost embarrassingly simple.

1. Pre-tokenize the training text into words (split on whitespace) and count them. Represent each word as a sequence of characters, with a special end-of-word marker `</w>` so that word-final pieces are distinguishable from word-internal ones.
2. Count all adjacent symbol pairs across the corpus, weighted by word frequency.
3. Merge the most frequent pair into a new symbol, in every word. Record the merge.
4. Repeat steps 2 and 3 for $N$ merges. The final vocabulary is the initial characters plus $N$ merged symbols.

To tokenize new text, apply the recorded merges in the order they were learned. The code below is a complete implementation and is the one used for the chart that follows.

{{figure: week08-bpe-merges | One word through BPE. Starting from characters, the most frequent adjacent pair is fused again and again; frequent words end up as a single token while rare ones remain sequences of smaller pieces.}}

```python:bpe.py
import collections

def get_vocab(text):
    """word -> count, each word as a tuple of symbols ending with the end-of-word marker."""
    counts = collections.Counter(text.lower().split())
    return {tuple(w) + ("</w>",): c for w, c in counts.items()}

def pair_counts(vocab):
    pairs = collections.Counter()
    for word, c in vocab.items():
        for a, b in zip(word, word[1:]):
            pairs[(a, b)] += c                       # weight by how many times the word occurs
    return pairs

def merge_pair(pair, vocab):
    out = {}
    for word, c in vocab.items():
        new, i = [], 0
        while i < len(word):
            if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
                new.append(word[i] + word[i + 1]); i += 2
            else:
                new.append(word[i]); i += 1
        out[tuple(new)] = c
    return out

def learn_bpe(text, num_merges):
    vocab, merges = get_vocab(text), []
    for _ in range(num_merges):
        pairs = pair_counts(vocab)
        if not pairs:
            break
        best = max(pairs, key=lambda p: (pairs[p], p))   # most frequent; deterministic tie-break
        vocab = merge_pair(best, vocab)
        merges.append(best)
    return merges

def encode(word, merges):
    """Segment one word by applying merges in the order they were learned."""
    symbols = list(word) + ["</w>"]
    for a, b in merges:
        i = 0
        while i < len(symbols) - 1:
            if symbols[i] == a and symbols[i + 1] == b:
                symbols[i:i + 2] = [a + b]
            else:
                i += 1
    return symbols

text = ("the lower the newer the wider the lowest the newest the widest "
        "low low low low low lower lower newer newer newer newer newer newer "
        "wider wider wider newest newest newest newest newest newest newest newest newest "
        "widest widest widest widest widest lowest lowest ")
merges = learn_bpe(text, 30)
print(merges[:6])
print(encode("lowest", merges), encode("newer", merges), encode("slowest", merges))
words = text.lower().split()
for k in [0, 5, 10, 20, 30]:
    print(k, "merges ->", sum(len(encode(w, merges[:k])) for w in words), "tokens for", len(words), "words")
```

```text
[('w', 'e'), ('t', '</w>'), ('s', 't</w>'), ('n', 'e'), ('ne', 'we'), ('r', '</w>')]
['lowest</w>'] ['newer</w>'] ['s', 'lowest</w>']
0 merges -> 261 tokens for 44 words
5 merges -> 166 tokens for 44 words
10 merges -> 111 tokens for 44 words
20 merges -> 50 tokens for 44 words
30 merges -> 44 tokens for 44 words
```

Look at the third example: "slowest" never occurred in training, yet it is segmented as `s` + `lowest</w>`, two familiar pieces, instead of becoming `<unk>`. That is the entire point. (The merges are not intuitive linguistic units in this tiny corpus: the first merge fuses `w` and `e`. Real BPE vocabularies trained on large corpora produce more morpheme-like pieces, but they are discovered by frequency, not by linguistics.)

```chart
{
  "type": "line",
  "caption": "Corpus size in tokens (44 words) as BPE merges accumulate, computed with the code above on a tiny corpus. Each merge shortens the sequence; the curve flattens when every word is a single token. The trade-off at real scale: more merges give shorter sequences but a larger embedding table and softmax.",
  "height": 300,
  "data": {
    "labels": [0,2,4,6,8,10,12,15,20,25,30],
    "datasets": [{"label": "tokens needed to encode the corpus", "data": [261,219,183,152,131,111,94,76,50,44,44], "borderColor": "var(--c-purple)", "backgroundColor": "var(--c-purple)", "pointRadius": 3, "tension": 0.15}]
  },
  "options": {"scales": {"x": {"title": {"display": true, "text": "number of merges learned"}}, "y": {"title": {"display": true, "text": "total tokens"}, "min": 0}}}
}
```

Step through the algorithm yourself, with a corpus of your choice, in the next demo.

{{demo: week08-bpe-stepper | BPE merge step-through | 560}}

:::note The implementation is simple on purpose
This code is quadratic-ish and fine for a few thousand words. Production tokenizers (Hugging Face `tokenizers`, SentencePiece, `tiktoken`) keep an incrementally updated pair-count table and a priority queue, so that each merge costs time proportional to the number of affected words, not the whole corpus. The algorithm is the same.
:::

### Byte-level BPE {#byte-bpe}

Plain character-level BPE still has an unbounded base alphabet: Unicode has over 140,000 characters, and a rare one may be unseen at training time. **Byte-level BPE**, popularized by GPT-2 (Radford et al., 2019), runs BPE on the UTF-8 *bytes* of the text. The base alphabet is exactly 256 symbols, so any string in any language, emoji or binary garbage included, is representable with zero unknowns. The cost: a non-Latin character takes 2 to 4 base symbols, so text in under-represented scripts can be split into many more tokens unless the merges cover it well. This is one concrete way in which tokenizers create unequal costs across languages (longer sequences mean higher latency and price and less usable context).

### WordPiece {#wordpiece}

WordPiece (Schuster and Nakajima, 2012, for Japanese and Korean voice search; used for BERT by Devlin et al., 2018) has the same overall shape as BPE (start from characters, merge) but with a different criterion. Instead of the raw pair frequency, it picks the merge that most increases the likelihood of the training data under a unigram language model over pieces. Under simplifying assumptions this reduces to choosing the pair $(a,b)$ that maximizes

$$
\mathrm{score}(a,b)=\frac{\mathrm{count}(ab)}{\mathrm{count}(a)\,\mathrm{count}(b)},
$$

a PMI-like quantity: it favours pairs that are strongly associated, not just frequent. Frequent letters such as `e` and `s` form frequent pairs, but a pair is only merged first when its parts rarely appear apart. At inference WordPiece segments a word with a greedy longest-match-first rule, and marks non-initial pieces with a `##` prefix (`play ##ing`). The details of the original Google implementation were never fully published, so open-source "WordPiece" trainers differ in their specifics.

### Unigram language model and SentencePiece {#sentencepiece}

BPE and WordPiece build a vocabulary bottom-up. Kudo (2018) proposed the opposite, **Unigram LM** tokenization: start with a large candidate vocabulary (all frequent substrings), assign each piece $x$ a probability $p(x)$, and define the probability of a segmentation $\boldsymbol{x}=(x_1,\dots,x_M)$ of a sentence as $\prod_m p(x_m)$. The EM algorithm re-estimates the piece probabilities, and pieces whose removal least reduces the corpus likelihood are pruned until the target vocabulary size is reached. Because the model is probabilistic, a word has several possible segmentations with different probabilities, which enables **subword regularization**: sampling a segmentation at training time acts as data augmentation that makes translation models more robust to tokenization noise.

**SentencePiece** (Kudo and Richardson, 2018) is a library, not an algorithm. It implements BPE and Unigram on *raw text*, treating whitespace as just another character (written as the visible symbol `▁`) rather than relying on a language-specific pre-tokenizer. That makes tokenization fully reversible (`decode(encode(text)) == text`) and language independent, which is essential for Chinese or Japanese, where there are no spaces to split on. T5, LLaMA-family and many multilingual models use SentencePiece.

| Method | Builds vocabulary | Merge / pruning rule | Segmentation at inference | Notable users |
| --- | --- | --- | --- | --- |
| BPE | bottom-up | most frequent adjacent pair | apply merges in learned order | original NMT (Sennrich 2016) |
| Byte-level BPE | bottom-up on bytes | same, base alphabet of 256 bytes | same | GPT-2 and successors |
| WordPiece | bottom-up | highest likelihood gain (about count(ab) / (count(a) count(b))) | greedy longest match, `##` continuation | BERT |
| Unigram LM | top-down pruning | remove pieces with the smallest likelihood loss | Viterbi (or sampled, for regularization) | SentencePiece, T5 |

```python:sentencepiece_usage.py
# pip install sentencepiece
import sentencepiece as spm

# Train a small Unigram tokenizer on a plain-text file with one sentence per line.
spm.SentencePieceTrainer.train(
    input="corpus.txt", model_prefix="toy", vocab_size=2000,
    model_type="unigram",            # or "bpe"
    character_coverage=0.9995,       # keep nearly all characters; rare ones map to <unk> or byte fallback
)

sp = spm.SentencePieceProcessor(model_file="toy.model")
ids = sp.encode("Unfriendliest tokenizers are fun.", out_type=int)
pieces = sp.encode("Unfriendliest tokenizers are fun.", out_type=str)
print(pieces)                        # e.g. ['▁Un', 'friend', 'li', 'est', ...]; depends on the corpus
print(sp.decode(ids))                # reversible: recovers the original text

# Subword regularization: sample a different segmentation each call (Unigram only)
for _ in range(3):
    print(sp.encode("tokenizers", out_type=str, enable_sampling=True, alpha=0.1, nbest_size=-1))
```

### Practical consequences of tokenization {#tokenization-practice}

- **Vocabulary size is a trade-off.** Larger vocabularies mean shorter sequences (cheaper attention, better context use) but a bigger embedding table and softmax, and more rarely seen tokens whose vectors are badly trained. Typical values are 30k to 50k for English models and 100k or more for multilingual ones.
- **Comparisons across tokenizers need care.** As noted in [Week 7](week-07.html#language-modeling), perplexity per token is not comparable between models with different tokenizers. Use bits per byte or per character.
- **Tokenizers fail in visible ways.** Numbers are split inconsistently ("1234" may be one piece, "1235" three), which hurts arithmetic. Leading spaces change tokens (`" the"` is not `"the"`). Capitalization, typos and code get fragmented. When a model is bad at spelling or counting letters, tokenization is a prime suspect, since the model never sees the characters.
- **Always use the tokenizer the model was trained with.** A mismatched tokenizer silently produces garbage: the integer ids index into the wrong embedding rows.

## Character-level and byte-level models {#character-models}

The extreme end of the spectrum drops the vocabulary entirely. You already built a **character-level language model** in the [Week 7 lab](week-07.html#lab-char-lm): the input is a sequence of characters (an embedding table of a few dozen rows), and the model predicts the next character. The strengths are the ones listed at the start of this section: no out-of-vocabulary tokens, robustness to typos and morphology, a tiny softmax, and the ability to model any string. The weaknesses are as concrete: sequences are about 5 times longer than word sequences, so the compute per sentence rises and dependencies span more steps, and the model has to rediscover spelling and word boundaries before it can use semantics.

Several designs recover the benefits without the cost.

- **Character-aware word models.** Kim, Jernite, Sontag and Rush (2016) build a word representation from its characters with a small 1D CNN and a max-over-time pooling layer (the 1D convolutions of [Week 7](week-07.html#tcn)), feed the result to a word-level LSTM, and obtain good language-model perplexity with far fewer parameters than a word-embedding table. This is a learned counterpart to fastText's $n$-gram sum.
- **Hierarchical and downsampling models.** The character sequence is compressed by strided convolutions or pooling (as in CANINE, Clark et al., 2022) so that the expensive layers run on a shorter sequence.
- **Byte-level pretrained models.** ByT5 (Xue et al., 2022) trains an encoder-decoder directly on bytes and is more robust to noise and spelling variation, at the price of longer sequences and more compute per example.

For most text applications at scale, subword tokenization remains the pragmatic default because the cost of sequence length dominates. Character- or byte-level modelling is the right choice for small-vocabulary or noisy domains (DNA, chemical strings, social-media text with creative spelling), for morphologically rich languages with little data, and as a robustness baseline.

## Embeddings beyond text {#beyond-text}

Nothing in word2vec is specific to words. The recipe is: *treat the objects as symbols, define what it means for two to be in the same context, and learn vectors that predict context*. Whatever the domain, the resulting table of vectors can feed a downstream model, be used for nearest-neighbour retrieval, or be visualized.

- **Items in sessions** (item2vec, Barkan and Koenigstein, 2016): a shopping basket or listening session plays the role of a sentence, and the items in it are the words. Products bought together get nearby vectors.
- **Nodes in graphs** (DeepWalk, Perozzi et al., 2014; node2vec, Grover and Leskovec, 2016): random walks on the graph produce "sentences" of node ids, and skip-gram is run on them. The two methods differ in how the walks are biased (breadth-first versus depth-first exploration).
- **Entities and relations in knowledge graphs** (TransE, Bordes et al., 2013): a fact (head $h$, relation $r$, tail $t$) is modelled as a translation in embedding space, $\boldsymbol{h}+\boldsymbol{r}\approx\boldsymbol{t}$, trained with a margin-based ranking loss against corrupted triples. The word2vec offset structure becomes a modelling assumption rather than a discovery.
- **Categorical features** in tabular data: user ids, zip codes, product categories. An `nn.Embedding` per categorical feature, trained jointly with the rest of the model, replaces one-hot encoding (a standard trick, popularized for tabular deep learning and for ranking systems such as the YouTube recommender of Covington et al., 2016).

### Matrix factorization as an embedding model {#matrix-factorization}

The cleanest bridge to non-text data is collaborative filtering. You have a sparse matrix $\mathbf{R}\in\R^{U\times I}$ of ratings by $U$ users on $I$ items, with only a small fraction of entries observed. **Matrix factorization**, the core of the winning approaches in the Netflix Prize (Koren, Bell and Volinsky, 2009), models each rating as an inner product of a user embedding $\boldsymbol{p}_u\in\R^d$ and an item embedding $\boldsymbol{q}_i\in\R^d$, plus biases:

$$
\hat r_{ui}=\mu+b_u+b_i+\boldsymbol{p}_u^\top\boldsymbol{q}_i,\qquad
\min\sum_{(u,i)\in\Omega}\big(r_{ui}-\hat r_{ui}\big)^2+\lambda\big(\lVert\boldsymbol{p}_u\rVert^2+\lVert\boldsymbol{q}_i\rVert^2\big),
$$

where $\Omega$ is the set of observed entries and $\mu$ the global mean. You recognize every ingredient: embeddings, a dot-product score, squared loss, weight decay. It *is* a two-table embedding model, exactly like skip-gram with users as "centres" and items as "contexts". The difference from the SVD at the start of the week is that only observed entries enter the loss, and the regularization stops the model from fitting the sparsity pattern.

{{figure: week08-matrix-factorization | Matrix factorization as embedding learning. The sparse rating matrix is approximated by the product of a user-embedding table and an item-embedding table; only observed entries (coloured) contribute to the loss.}}

```python:mf_torch.py
import torch, torch.nn as nn

class MatrixFactorization(nn.Module):
    """r_hat(u, i) = mu + b_u + b_i + p_u . q_i, trained on observed (user, item, rating) triples."""
    def __init__(self, n_users, n_items, d=32, mu=0.0):
        super().__init__()
        self.P = nn.Embedding(n_users, d)           # user embeddings, (U, d)
        self.Q = nn.Embedding(n_items, d)           # item embeddings, (I, d)
        self.bu = nn.Embedding(n_users, 1)
        self.bi = nn.Embedding(n_items, 1)
        self.mu = mu
        for e in (self.P, self.Q):
            nn.init.normal_(e.weight, std=0.1)
        nn.init.zeros_(self.bu.weight); nn.init.zeros_(self.bi.weight)

    def forward(self, u, i):                        # u, i: (B,) long tensors
        dot = (self.P(u) * self.Q(i)).sum(-1)       # (B,)
        return self.mu + self.bu(u).squeeze(-1) + self.bi(i).squeeze(-1) + dot

# Synthetic ratings from a ground-truth low-rank model, to verify that the code can recover structure
torch.manual_seed(0)
U, I, d_true, n_obs = 200, 100, 4, 8000
Pt, Qt = torch.randn(U, d_true), torch.randn(I, d_true)
u = torch.randint(0, U, (n_obs,)); i = torch.randint(0, I, (n_obs,))
r = (Pt[u] * Qt[i]).sum(-1) + 0.1 * torch.randn(n_obs)
split = int(0.8 * n_obs)
tr, te = slice(0, split), slice(split, n_obs)

model = MatrixFactorization(U, I, d=8, mu=r[tr].mean().item())
opt = torch.optim.Adam(model.parameters(), lr=0.02, weight_decay=1e-4)   # weight decay as the L2 penalty
for epoch in range(200):
    opt.zero_grad()
    loss = ((model(u[tr], i[tr]) - r[tr]) ** 2).mean()
    loss.backward(); opt.step()
    if epoch % 50 == 0:
        with torch.no_grad():
            rmse = ((model(u[te], i[te]) - r[te]) ** 2).mean().sqrt().item()
        print(f"epoch {epoch:3d}  train mse {loss.item():.3f}  test rmse {rmse:.3f}")
```

Modern recommender systems replace the dot product of two lookup tables with **two-tower** neural networks: one tower maps a user's features to an embedding and another maps an item's features to an embedding, so cold-start users and items (no id history) still get vectors. Candidates are retrieved by nearest-neighbour search on the dot product. The tower outputs are trained with the metric-learning losses of the next section.

## Metric learning: contrastive, triplet and InfoNCE {#metric-learning}

So far the objective was predictive (guess the context). **Metric learning** states the goal geometrically instead: learn an embedding $f_\theta$ such that *similar* inputs are close and *dissimilar* inputs are far under a chosen distance (usually Euclidean on unit vectors, or cosine). It applies to any pair of inputs for which you can define "similar": two photos of the same face, a query and a clicked document, an image and its caption, two augmentations of the same image. The classifier-with-$C$-outputs trick fails when classes are open-ended (faces of people never seen in training), whereas an embedding of a new example can be compared with stored ones by distance.

{{figure: week08-metric-learning | Left: the triplet loss pulls a positive within a margin of the anchor relative to a negative. Right: InfoNCE treats the other items of a batch as negatives for each anchor; the diagonal of the similarity matrix holds the positive pairs.}}

### Contrastive loss (pairs) {#contrastive-loss}

Hadsell, Chopra and LeCun (2006) introduced the pairwise contrastive loss. For a pair $(\boldsymbol{x}_1,\boldsymbol{x}_2)$ with label $y=1$ if similar and $0$ if not, and $D=\lVert f(\boldsymbol{x}_1)-f(\boldsymbol{x}_2)\rVert$:

$$
\mathcal{L}=y\,D^2+(1-y)\,\max(0,\,m-D)^2 .
$$

Similar pairs are pulled together; dissimilar pairs are pushed apart only until they are $m$ apart (the margin), after which they contribute nothing. Without the margin the model would push negatives to infinity.

### Triplet loss {#triplet-loss}

The triplet loss of FaceNet (Schroff, Kalenichenko and Philbin, 2015) takes an anchor $a$, a positive $p$ of the same identity, and a negative $n$ of a different identity:

$$
\mathcal{L}=\max\big(0,\ \lVert f(a)-f(p)\rVert^2-\lVert f(a)-f(n)\rVert^2+m\big),
$$

with embeddings L2-normalized onto the unit sphere. It only asks for the *relative* ordering: the positive must be closer than the negative by at least $m$. Triplets already satisfying the constraint give zero loss and zero gradient, so as training progresses most random triplets become useless. The practical difficulty of the method is therefore **triplet mining**: choosing triplets that still teach something. *Hard negatives* (negatives closer to the anchor than the positive) give the strongest signal, but the very hardest ones early in training can be mislabelled or outliers and collapse the embedding, which is why FaceNet used *semi-hard* negatives (farther than the positive but still inside the margin) selected within large mini-batches.

### InfoNCE {#infonce}

The dominant loss in current representation learning treats the problem as classification among many candidates. Given a batch of $N$ positive pairs $(\boldsymbol{x}_i,\boldsymbol{y}_i)$, with embeddings $\boldsymbol{z}_i$ and $\boldsymbol{z}'_j$ (unit-normalized), the **InfoNCE** loss (van den Oord, Li and Vinyals, 2018, in their contrastive predictive coding paper) is, for anchor $i$:

$$
\mathcal{L}_i=-\log\frac{\exp(\boldsymbol{z}_i^\top\boldsymbol{z}'_i/\tau)}{\sum_{j=1}^{N}\exp(\boldsymbol{z}_i^\top\boldsymbol{z}'_j/\tau)} ,
$$

a softmax cross-entropy over the batch in which the correct "class" is the matching item, with $\tau>0$ the **temperature**. The other $N-1$ items of the batch serve as free negatives, so larger batches give more negatives and a tighter bound. Three facts to remember.

1. **A mutual-information bound.** With $K=N-1$ negatives, $I(X;Y)\ge\log N-\mathbb{E}[\mathcal{L}]$. The loss of a random model is $\log N$, and a perfect model reaches 0, so the bound saturates at $\log N$: to certify a high mutual information you need many negatives.
2. **Temperature controls the hardness profile.** The gradient of $\mathcal{L}_i$ with respect to the similarity of negative $j$ is $p_{ij}/\tau$ where $p_{ij}$ is the softmax probability of $j$. Small $\tau$ concentrates $p$ on the hardest negatives (and amplifies the gradient by $1/\tau$); large $\tau$ spreads the weight over all negatives, which behaves like a uniform push. Typical values are 0.05 to 0.5, and $\tau$ is sometimes learned (as in CLIP, Radford et al., 2021).
3. **Relation to skip-gram.** Negative sampling is a cheap, binary approximation of exactly this softmax; InfoNCE (and the noise-contrastive estimation of Gutmann and Hyvarinen, 2010 that inspired both) uses the full softmax over the sampled candidates.

```python:contrastive_losses.py
import torch
import torch.nn.functional as F

def info_nce(z1, z2, tau=0.1):
    """Symmetric InfoNCE for a batch of positive pairs. z1, z2: (N, d) embeddings of matched pairs."""
    z1, z2 = F.normalize(z1, dim=-1), F.normalize(z2, dim=-1)      # cosine similarity via dot product
    logits = z1 @ z2.t() / tau                                      # (N, N); diagonal = positives
    labels = torch.arange(z1.size(0), device=z1.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))

def triplet_loss(a, p, n, margin=0.2):
    """Triplet loss on L2-normalized embeddings; a, p, n: (B, d)."""
    a, p, n = (F.normalize(t, dim=-1) for t in (a, p, n))
    d_ap = (a - p).pow(2).sum(-1)
    d_an = (a - n).pow(2).sum(-1)
    return F.relu(d_ap - d_an + margin).mean()

def batch_hard_triplet(emb, labels, margin=0.2):
    """Batch-hard mining: for each anchor use its farthest positive and nearest negative in the batch."""
    emb = F.normalize(emb, dim=-1)
    dist = torch.cdist(emb, emb)                                    # (B, B) Euclidean distances
    same = labels[:, None] == labels[None, :]
    eye = torch.eye(len(labels), dtype=torch.bool, device=emb.device)
    d_pos = dist.masked_fill(~same | eye, float("-inf")).max(1).values   # hardest positive
    d_neg = dist.masked_fill(same, float("inf")).min(1).values           # hardest negative
    return F.relu(d_pos - d_neg + margin).mean()

torch.manual_seed(0)
z1, z2 = torch.randn(8, 16), torch.randn(8, 16)
print("random pairs: InfoNCE =", round(info_nce(z1, z2, 0.1).item(), 3), " (log 8 =", round(torch.log(torch.tensor(8.)).item(), 3), ")")
print("identical pairs: InfoNCE =", round(info_nce(z1, z1, 0.1).item(), 3))
```

A random model scores near $\log N$ (up to the noise of a finite sample and $\tau$), while a model that places each item close to its partner and away from the rest approaches 0. The chart computes the expected loss for a simple model, and the demo lets you change every quantity.

```chart
{
  "type": "line",
  "caption": "Expected InfoNCE loss (Monte Carlo, 20,000 draws) for a positive cosine similarity of 0.7 and K negatives whose similarities are drawn from N(0, 0.2^2), as a function of the temperature. Dashed lines: the chance level log(K+1) that a random scorer achieves. Small tau separates the positive almost perfectly; large tau drives the loss up toward chance because the softmax cannot discriminate.",
  "height": 330,
  "data": {
    "labels": [0.05,0.1,0.2,0.3,0.5,0.75,1.0],
    "datasets": [
      {"label": "K = 7", "data": [0.008,0.043,0.292,0.607,1.05,1.346,1.513], "borderColor": "var(--c-blue)", "backgroundColor": "var(--c-blue)", "pointRadius": 3},
      {"label": "K = 63", "data": [0.061,0.335,1.413,2.152,2.879,3.283,3.494], "borderColor": "var(--c-accent)", "backgroundColor": "var(--c-accent)", "pointRadius": 3},
      {"label": "chance, K = 7 (log 8)", "data": [2.079,2.079,2.079,2.079,2.079,2.079,2.079], "borderColor": "var(--c-blue)", "borderDash": [6,4], "pointRadius": 0},
      {"label": "chance, K = 63 (log 64)", "data": [4.159,4.159,4.159,4.159,4.159,4.159,4.159], "borderColor": "var(--c-accent)", "borderDash": [6,4], "pointRadius": 0}
    ]
  },
  "options": {"scales": {"x": {"title": {"display": true, "text": "temperature tau"}}, "y": {"title": {"display": true, "text": "InfoNCE loss (nats)"}, "min": 0}}}
}
```

{{demo: week08-infonce-explorer | InfoNCE and triplet loss explorer | 540}}

:::history Contrastive learning grows up
Contrastive losses started in the 1990s and 2000s (Bromley et al.'s Siamese networks for signature verification, 1993; Hadsell et al., 2006). The triplet loss powered face recognition around 2015. The 2018 to 2021 wave (CPC, SimCLR by Chen et al., 2020, MoCo by He et al., 2020, and CLIP by Radford et al., 2021) showed that InfoNCE with strong augmentations or paired text can learn visual features without class labels that rival supervised ones. We come back to self-supervised and multimodal learning in the later weeks.
:::

:::tip Choosing a loss
Few labelled classes, many examples each: use cross-entropy with a classifier head. Open-set identification or retrieval with labelled identities: triplet or a normalized-softmax variant with batch-hard mining. Large batches of naturally paired data (image-caption, query-document, two views): InfoNCE with the largest batch you can afford, or a memory bank / queue of negatives when you cannot.
:::

## Lab: train skip-gram embeddings, probe them, and use them downstream {#lab}

In this lab you will (1) train skip-gram with negative sampling in PyTorch on the first couple of million words of `text8` (a cleaned, lower-cased 100 MB slice of English Wikipedia from 2006, a standard small corpus for word-vector experiments), (2) inspect nearest neighbours and analogies, (3) run bias probes, and (4) use the vectors as features in a news-topic classifier and compare with a bag-of-words baseline. It runs in Colab on a GPU in a few minutes, and on a CPU in somewhat longer.

**Step 1: data, vocabulary, subsampling.**

```python:lab_data.py
import os, zipfile, urllib.request, collections, numpy as np, torch

URL = "http://mattmahoney.net/dc/text8.zip"
if not os.path.exists("text8"):
    urllib.request.urlretrieve(URL, "text8.zip")
    zipfile.ZipFile("text8.zip").extractall(".")
words = open("text8").read().split()[:3_000_000]             # first 3M words keeps the lab fast
print("tokens:", len(words))

counter = collections.Counter(words)
min_count = 5
vocab = ["<unk>"] + [w for w, c in counter.most_common() if c >= min_count]
idx = {w: i for i, w in enumerate(vocab)}
V = len(vocab)
ids = np.array([idx.get(w, 0) for w in words], dtype=np.int64)   # rare words -> <unk> (id 0)
freq = np.bincount(ids, minlength=V).astype(np.float64)
print("vocab size:", V, " most common:", vocab[1:6])

# Frequent-word subsampling: keep token w with prob min(1, sqrt(t/f(w)))   (equivalent to discarding w.p. 1 - sqrt(t/f))
t = 1e-4
f = freq / freq.sum()
keep_prob = np.minimum(1.0, np.sqrt(t / np.maximum(f, 1e-12)))
rng = np.random.default_rng(0)
mask = (rng.random(len(ids)) < keep_prob[ids]) & (ids != 0)
sub = torch.from_numpy(ids[mask])
print("after subsampling:", len(sub), "tokens")

# Noise distribution: unigram^(3/4)
noise = torch.from_numpy(freq ** 0.75); noise[0] = 0; noise /= noise.sum()
```

**Step 2: the model and a batch sampler.** We sample positions at random and, for each, a random signed offset in $1..m$, which approximates word2vec's dynamic window without storing every pair.

```python:lab_model.py
import torch, torch.nn as nn, torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"

class SkipGramNS(nn.Module):
    def __init__(self, V, d=100):
        super().__init__()
        self.inp = nn.Embedding(V, d, sparse=True)            # v_w, (V, d)
        self.out = nn.Embedding(V, d, sparse=True)            # u_w, (V, d)
        nn.init.uniform_(self.inp.weight, -0.5 / d, 0.5 / d)
        nn.init.zeros_(self.out.weight)

    def forward(self, c, o, neg):
        # c: (B,)  o: (B,)  neg: (B, K)
        v = self.inp(c)                                       # (B, d)
        pos_score = (v * self.out(o)).sum(-1)                 # (B,)
        neg_score = torch.bmm(self.out(neg), v.unsqueeze(-1)).squeeze(-1)   # (B, K)
        # J = -log sigma(s_o) - sum_k log sigma(-s_k); logsigmoid is the numerically stable form
        return -(F.logsigmoid(pos_score) + F.logsigmoid(-neg_score).sum(-1)).mean()

def get_batch(sub, B, m, K, noise):
    n = len(sub)
    pos = torch.randint(m, n - m, (B,))
    off = torch.randint(1, m + 1, (B,)) * (torch.randint(0, 2, (B,)) * 2 - 1)    # +-1..m
    c, o = sub[pos], sub[pos + off]
    neg = torch.multinomial(noise, B * K, replacement=True).view(B, K)
    return c.to(device), o.to(device), neg.to(device)

model = SkipGramNS(V, d=100).to(device)
opt = torch.optim.SparseAdam(model.parameters(), lr=2e-3)      # Adam variant for sparse gradients
B, m, K, steps = 4096, 5, 5, 20000
for step in range(1, steps + 1):
    c, o, neg = get_batch(sub, B, m, K, noise)
    loss = model(c, o, neg)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 2000 == 0:
        print(f"step {step:6d}  loss {loss.item():.3f}")
```

For orientation: at initialization the output vectors are zero, so every score is 0 and the loss is $(1+K)\ln 2\approx4.16$ for $K=5$. If your first printed loss is far from that, check your shapes before anything else. The loss will fall to something in the low 2s; do not expect it to approach zero (see the discussion of the noise floor above).

**Step 3: nearest neighbours and analogies.**

```python:lab_neighbours.py
E = model.inp.weight.detach().cpu().numpy()
En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)

def most_similar(positive, negative=(), k=6):
    q = sum(En[idx[w]] for w in positive) - sum(En[idx[w]] for w in negative)
    q /= np.linalg.norm(q) + 1e-9
    sims = En @ q
    banned = {idx[w] for w in list(positive) + list(negative)} | {0}
    return [(vocab[i], round(float(sims[i]), 3)) for i in np.argsort(-sims) if i not in banned][:k]

for w in ["king", "france", "computer", "music", "dog"]:
    if w in idx:
        print(f"{w:10s}", most_similar([w]))
print("king - man + woman ->", most_similar(["king", "woman"], ["man"]))
print("paris - france + italy ->", most_similar(["paris", "italy"], ["france"]))
```

With this little data and training, expect sensible neighbours for common words and unreliable analogies, especially for syntactic relations; the analogy outputs are the place where you should be most skeptical. Try more steps, a larger corpus slice, or $d=200$ and watch the answers improve. Plot a 2D projection (PCA or t-SNE from scikit-learn) of a few hundred words to see topical clusters.

**Step 4: bias probes.** Re-use `gender_direction`, `projection_table` and `weat_effect_size` from the earlier bias section, with the word lists below. Check each list for words missing from the vocabulary, and keep lists short and explicit.

```python:lab_bias.py
# (paste the functions from bias_probes.py above; here E is the learned input-embedding matrix and idx the word->row dict)
male   = ["he", "man", "his", "him", "father", "son", "brother", "male"]
female = ["she", "woman", "her", "hers", "mother", "daughter", "sister", "female"]
career = ["executive", "management", "professional", "corporation", "salary", "office", "business", "career"]
family = ["home", "parents", "children", "family", "cousins", "marriage", "wedding", "relatives"]

def vecs(ws):
    return [E[idx[w]] for w in ws if w in idx]

print("missing words:", [w for w in male + female + career + family if w not in idx])
d_eff = weat_effect_size(vecs(career), vecs(family), vecs(male), vecs(female))
p_val = weat_permutation_p(vecs(career), vecs(family), vecs(male), vecs(female), n_perm=5000)
print(f"WEAT effect size {d_eff:.2f}, one-sided permutation p = {p_val:.3f}")

g = gender_direction(E, idx)
jobs = [w for w in ["nurse", "engineer", "teacher", "doctor", "secretary", "scientist", "librarian", "programmer"] if w in idx]
for w, s in projection_table(E, idx, jobs, g):
    print(f"{w:12s} {s:+.3f}")
```

Interpret the output with the caveats of the bias section: a small corpus gives noisy numbers, so repeat with a different random seed and see how much the effect size moves before reading anything into it. Then try the same probes on pretrained GloVe vectors (for example `gensim.downloader.load("glove-wiki-gigaword-100")`) and compare.

**Step 5: use the embeddings in a classifier.** We classify newsgroup posts into four topics with (a) TF-IDF features, (b) the average of the skip-gram vectors of the words in a post, and (c) the average of *random* vectors of the same dimension (a control that shows what the training bought us).

```python:lab_downstream.py
import re
from sklearn.datasets import fetch_20newsgroups
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

cats = ["rec.sport.hockey", "sci.space", "talk.politics.mideast", "comp.graphics"]
kw = dict(categories=cats, remove=("headers", "footers", "quotes"))
tr = fetch_20newsgroups(subset="train", **kw)
te = fetch_20newsgroups(subset="test", **kw)

def doc_vector(text, table):
    toks = [idx[w] for w in re.findall(r"[a-z]+", text.lower()) if w in idx]
    return table[toks].mean(0) if toks else np.zeros(table.shape[1])

def featurize(docs, table):
    return np.stack([doc_vector(t, table) for t in docs])

rand_table = np.random.default_rng(0).normal(size=En.shape)      # control: same shape, no training
results = {}
for name, table in [("skip-gram (ours)", En), ("random vectors", rand_table)]:
    Xtr, Xte = featurize(tr.data, table), featurize(te.data, table)
    clf = LogisticRegression(max_iter=2000, C=10).fit(Xtr, tr.target)
    results[name] = accuracy_score(te.target, clf.predict(Xte))

tfidf = TfidfVectorizer(min_df=2, sublinear_tf=True)
Xtr, Xte = tfidf.fit_transform(tr.data), tfidf.transform(te.data)
results["TF-IDF + logistic regression"] = accuracy_score(te.target, LogisticRegression(max_iter=2000, C=10).fit(Xtr, tr.target).predict(Xte))

for k, v in results.items():
    print(f"{k:32s} test accuracy {v:.3f}")
```

What to look for (treat these as hypotheses to test, since the numbers depend on your run): the random-vector control should be far below the trained vectors and not much above chance (0.25 for four balanced classes); the trained averaged vectors should do much better; and TF-IDF on this size of training set often matches or beats averaged static embeddings, because the topics are separable by keywords. Embeddings pay off most when labels are scarce (try training on 100 posts only) and when test vocabulary differs from training vocabulary. Also try initializing an `nn.Embedding` of a small classifier from `En` and fine-tuning it, and compare frozen versus fine-tuned.

## Common mistakes and debugging tips {#pitfalls}

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| Nearest neighbours are all very frequent words, or look random | Vectors not normalized before cosine; undertrained; no subsampling of frequent words | Row-normalize; train more; subsample with $t\approx10^{-4}$ to $10^{-5}$ |
| Initial loss is not about $(1+K)\ln2$ | Output table not initialized to zero, wrong score sign for negatives, loss not summed over $K$ | Check `-logsigmoid(-neg_score)` for negatives; print the first-step loss |
| Loss goes to `nan` | `log(sigmoid(x))` computed naively with large $|x|$, or learning rate too high for plain SGD | Use `F.logsigmoid` / `binary_cross_entropy_with_logits`; use a smaller rate |
| Analogies return one of the input words | Inputs not excluded from the candidate list | Mask $a$, $b$, $c$ before the argmax |
| Training is slow with a huge vocabulary | Dense optimizer updating all $V\times d$ rows each step | `sparse=True` embeddings with `SparseAdam` / SGD, or just keep the batch large enough |
| Downstream model does worse than random embeddings | Token to row mapping mismatch (different vocab or tokenizer at train and test), padding index being averaged in | Build features from the same `idx`; mask padding when averaging (`padding_idx` or lengths) |
| Pretrained vectors do not help | Domain mismatch, case mismatch (GloVe is lowercased), tokenization mismatch | Check coverage (fraction of tokens in vocabulary); lowercase; fine-tune |
| Out-of-vocabulary words dominate the data | Word-level vocabulary on noisy text | Switch to subword tokens or fastText-style $n$-gram vectors |
| BPE output differs from the reference tokenizer | Different pre-tokenization (whitespace, punctuation, byte mapping) or merge order | Use the library tokenizer that the model shipped with; unit-test a few sentences |
| Triplet or contrastive loss collapses (all embeddings equal) | Hardest negatives too early, no normalization, margin or temperature mis-set | Semi-hard or batch-hard mining after a warm-up, L2-normalize, tune $\tau$ |
| Contrastive accuracy is great in training and poor in retrieval | Positives leak (augmentations too weak, duplicate items in a batch act as false negatives) | Deduplicate items within a batch, strengthen augmentations, evaluate on held-out identities |

## Exercises {#exercises}

:::exercise Exercise 1: The skip-gram gradient (derivation)
For the loss $J=-\log\sigma(\boldsymbol{u}_o^\top\boldsymbol{v}_c)-\sum_{k=1}^{K}\log\sigma(-\boldsymbol{u}_{n_k}^\top\boldsymbol{v}_c)$ derive $\partial J/\partial\boldsymbol{v}_c$, and check it numerically with finite differences for $d=3$, $K=2$.
:::details Solution
Let $s_o=\boldsymbol{u}_o^\top\boldsymbol{v}_c$, $s_k=\boldsymbol{u}_{n_k}^\top\boldsymbol{v}_c$. Since $\frac{d}{dx}\log\sigma(x)=1-\sigma(x)$ and $\frac{d}{dx}\log\sigma(-x)=-\sigma(x)$, we get $\partial J/\partial s_o=\sigma(s_o)-1$ and $\partial J/\partial s_k=\sigma(s_k)$. Because $\partial s/\partial\boldsymbol{v}_c=\boldsymbol{u}$,
$$\frac{\partial J}{\partial\boldsymbol{v}_c}=(\sigma(s_o)-1)\boldsymbol{u}_o+\sum_k\sigma(s_k)\boldsymbol{u}_{n_k}.$$
Check:
```python
import numpy as np
rng = np.random.default_rng(1)
sig = lambda x: 1 / (1 + np.exp(-x))
vc, uo, un = rng.normal(size=3), rng.normal(size=3), rng.normal(size=(2, 3))
J = lambda v: -np.log(sig(uo @ v)) - np.log(sig(-(un @ v))).sum()
g = (sig(uo @ vc) - 1) * uo + (sig(un @ vc)[:, None] * un).sum(0)
num = np.array([(J(vc + 1e-6 * e) - J(vc - 1e-6 * e)) / 2e-6 for e in np.eye(3)])
print(np.allclose(g, num, atol=1e-6))   # True
```
:::
:::

:::exercise Exercise 2: Cost of the full softmax
A vocabulary has $V=500{,}000$ words and $d=300$. Compare the multiply-adds needed to compute the loss and the gradients for one (centre, context) pair with the full softmax versus negative sampling with $K=10$. How many rows of the output table are updated in each case?
:::details Solution
Full softmax: computing all $V$ scores costs $Vd=1.5\times10^8$ multiply-adds, and the gradient touches the entire $V\times d$ output table (every row receives a nonzero gradient proportional to its softmax probability), $\approx$ another $2Vd$. Roughly $4.5\times10^8$ operations and $500{,}000$ rows. Negative sampling: $K+1=11$ scores, i.e. $11\cdot300=3{,}300$ multiply-adds, plus about as many for the gradients, and only $K+1=11$ output rows plus 1 input row are updated. That is about five orders of magnitude fewer operations, which is the whole reason it made billion-word training feasible.
:::
:::

:::exercise Exercise 3: What the shifted PMI result predicts
Using the SGNS optimum $\boldsymbol{u}_c^\top\boldsymbol{v}_w=\mathrm{PMI}(w,c)-\log K$, what is the optimal dot product for two words that co-occur exactly as often as independence predicts, for $K=5$? What happens to the optimum for a pair that never co-occurs in the data?
:::details Solution
Independence means $P(w,c)=P(w)P(c)$, so $\mathrm{PMI}=0$ and the optimal score is $-\log5\approx-1.61$. Negative scores are normal: most pairs should look "unlikely". For a pair that never co-occurs, $\#(w,c)=0$, the positive term vanishes and the derivation gives $e^{x}=0$, so the optimum is $x\to-\infty$; in practice the score is pushed toward large negative values but only as far as the finite number of negative samples and the regularization allow. This is why the factorized matrix is shifted PMI with unobserved pairs handled implicitly, not an explicit $-\infty$ entry.
:::
:::

:::exercise Exercise 4: Noise distribution (coding)
Write code that samples $10^6$ negatives from $P_n\propto f^{\alpha}$ for a Zipfian vocabulary and verifies that for $\alpha=0.75$ the empirical frequency of the most frequent word is close to the analytic value. Which word types become more likely as negatives when $\alpha$ drops from 1 to 0.75?
:::details Solution
```python
import numpy as np
V = 1000
f = 1 / np.arange(1, V + 1); f /= f.sum()
for alpha in (1.0, 0.75):
    p = f ** alpha; p /= p.sum()
    samples = np.random.default_rng(0).choice(V, size=1_000_000, p=p)
    print(alpha, "analytic", round(p[0], 4), "empirical", round((samples == 0).mean(), 4))
```
The analytic top-word probabilities are 0.1336 for $\alpha=1$ and 0.0525 for $\alpha=0.75$. Reducing $\alpha$ moves probability mass from the head to the tail: rare and mid-frequency words are drawn as negatives more often, so their vectors receive more "push" signal, while the very frequent words, which are weak, uninformative negatives, are drawn less.
:::
:::

:::exercise Exercise 5: Why divide by the temperature? (conceptual)
In InfoNCE with unit-norm embeddings, the logits lie in $[-1/\tau,1/\tau]$. Explain what happens for $\tau=100$ and for $\tau=0.001$.
:::details Solution
For $\tau=100$ all logits are within $\pm0.01$, so the softmax is almost uniform: the loss is close to $\log N$ regardless of how good the embedding is, and gradients with respect to similarities are tiny relative to the $1/\tau$ factor in a nonuniform way, so learning is slow and the embedding barely separates positives. For $\tau=0.001$ the logits are scaled by 1000, so the softmax is nearly one-hot on the largest similarity: the loss is either almost 0 (positive is the top item) or enormous (it is not), gradients vanish for already-correct examples and are dominated by a few hard negatives (and false negatives) otherwise, so training is unstable and noisy. The practical range (0.05 to 0.5) balances the two.
:::
:::

:::exercise Exercise 6: BPE by hand
Corpus (word: count): `hug: 10`, `pug: 5`, `pun: 12`, `bun: 4`, `hugs: 5`. Without an end-of-word marker, start from characters and perform the first three BPE merges. Break ties alphabetically on the pair.
:::details Solution
Pair counts, weighted: `h u`: 10+5=15; `u g`: 10+5+5=20 (hug, pug, hugs); `p u`: 5+12=17; `u n`: 12+4=16; `b u`: 4; `g s`: 5. Merge 1: `u g` (20) gives `ug`: hug = h ug, pug = p ug, hugs = h ug s. Pair counts now: `h ug`: 10+5=15; `p ug`: 5; `p u`: 12 (only pun); `u n`: 16; `b u`: 4; `ug s`: 5. Merge 2: `u n` (16): pun = p un, bun = b un. Now `h ug`: 15, `p un`: 12, `p ug`: 5, `b un`: 4, `ug s`: 5. Merge 3: `h ug` (15): hug = hug, hugs = hug s. Resulting vocabulary additions: `ug`, `un`, `hug`. Note that pairs are recounted after every merge: after merge 1 the pair `p u` survives only in `pun` (count 12), which is why `u n` (16) wins merge 2.
:::
:::

:::exercise Exercise 7: Triplet loss cases
With margin $m=0.2$ and squared distances, compute the loss for $(d_{ap},d_{an})$ equal to $(0.1,0.9)$, $(0.5,0.6)$, $(0.8,0.4)$. Which are "easy", "semi-hard", "hard"? Which have a gradient?
:::details Solution
$\max(0,d_{ap}-d_{an}+m)$: $(0.1,0.9)\to\max(0,-0.6)=0$: easy, no gradient. $(0.5,0.6)\to\max(0,0.1)=0.1$: semi-hard (negative farther than the positive but inside the margin), has a gradient. $(0.8,0.4)\to\max(0,0.6)=0.6$: hard (negative closer than the positive), has the largest gradient. Training on only easy triplets learns nothing; training on only the hardest ones early can collapse the embedding, hence semi-hard or batch-hard mining with warm-up.
:::
:::

:::exercise Exercise 8: Matrix factorization versus SVD (challenging)
A rating matrix has 99% missing entries. (a) Why is a plain truncated SVD of the matrix with missing entries set to 0 a poor recommender? (b) Show that the gradient of the regularized MF loss with respect to $\boldsymbol{p}_u$ only involves the items that user $u$ rated, and write the alternating-least-squares (ALS) closed-form update for $\boldsymbol{p}_u$.
:::details Solution
(a) Setting missing entries to 0 declares "this user rated this item 0", so the SVD spends its capacity reproducing the sparsity pattern and shrinks predictions toward 0; the observed entries are not fitted preferentially. (b) With $\Omega_u$ the items rated by $u$ and $\hat r_{ui}=\boldsymbol{p}_u^\top\boldsymbol{q}_i$ (biases dropped), the terms of the loss containing $\boldsymbol{p}_u$ are $\sum_{i\in\Omega_u}(r_{ui}-\boldsymbol{p}_u^\top\boldsymbol{q}_i)^2+\lambda\lVert\boldsymbol{p}_u\rVert^2$, so the gradient is $-2\sum_{i\in\Omega_u}(r_{ui}-\boldsymbol{p}_u^\top\boldsymbol{q}_i)\boldsymbol{q}_i+2\lambda\boldsymbol{p}_u$. Setting it to zero gives a ridge regression with the item vectors as features:
$$\boldsymbol{p}_u=\Big(\sum_{i\in\Omega_u}\boldsymbol{q}_i\boldsymbol{q}_i^\top+\lambda\mathbf{I}\Big)^{-1}\sum_{i\in\Omega_u}r_{ui}\boldsymbol{q}_i.$$
Alternating this update with the symmetric update for each $\boldsymbol{q}_i$ gives ALS, where each step is an exact convex solve.
:::
:::

:::exercise Exercise 9: Designing a bias audit (open-ended)
You are given a résumé-ranking system that embeds job descriptions and résumés with a pretrained encoder and ranks by cosine similarity. Outline an audit using the tools of this week and say what each result would and would not show.
:::details Solution
A reasonable answer has several layers. (1) Embedding-level: WEAT or a direction projection on occupation words and gendered or name-based attribute sets, with the word lists published; this shows associations in the representation, not a harm. (2) Behavioural: build paired résumés that differ only in a name or pronoun (counterfactual pairs), score them against the same job descriptions, and compare rank distributions and score gaps with a paired test; this is closer to the actual decision. (3) Outcome-level: measure selection rates by group on a labelled evaluation set (with the legal and ethical constraints on collecting group labels) and report disaggregated metrics, not just overall accuracy. (4) Document the limits: lists are English and culturally specific, intersectional groups are undersampled, and debiasing by projection may only hide the effect (Gonen and Goldberg, 2019), so re-run the behavioural test after any mitigation. The key point to state: the embedding audit is necessary but not sufficient, and the decision to deploy should rest on the behavioural and outcome analysis.
:::
:::

:::exercise Exercise 10: Tokenization and cost (conceptual)
Two tokenizers are applied to the same English paragraph (1000 characters). Tokenizer A has a 50k vocabulary and produces 230 tokens; tokenizer B has an 8k vocabulary and produces 340 tokens. A language model has an attention cost proportional to $T^2$ per layer and an embedding-plus-softmax cost proportional to $V$ per token. Discuss which is cheaper under different regimes and how to compare model quality fairly.
:::details Solution
Attention: $230^2=52{,}900$ versus $340^2=115{,}600$, so tokenizer A is about 2.2 times cheaper in the attention part and has 32% fewer positions for everything that scales linearly in $T$. The output softmax per token costs $\propto V$: $230\cdot50{,}000=1.15\times10^7$ versus $340\cdot8{,}000=2.7\times10^6$, so B is cheaper there (about 4.2 times), and A's embedding table is 6 times larger. Which total wins depends on the model width and context: for large models and long contexts attention and MLP costs dominate and the shorter sequence of A wins; for small models with a large softmax, B can win. For quality comparisons, perplexity per token is invalid (different units); compare bits per character (or per byte) on identical text, and also compare downstream task scores.
:::
:::

## Midterm review: Weeks 1 to 7 {#midterm-review}

The midterm takes place in the second session of this week. It covers Weeks 1 to 7 (Week 8 material is not examined). Expect a mix of short conceptual questions, derivations, shape and parameter-count calculations, and "what would happen if" diagnosis questions. The best preparation is to *derive things yourself on paper*, so the review below is built as a map, a list of derivations, and a practice exam.

### Concept map {#concept-map}

{{figure: week08-concept-map | How the first seven weeks fit together. Foundations (blue): tensors and autograd, MLPs and backprop, optimization. Regularization (green) cuts across all of them. Vision architectures (purple) specialise the MLP with convolution and residual connections; sequence models (amber) specialise it with recurrence and lead to attention. Representations (Week 8) feed every model.}}

Three threads connect the weeks, and exam questions often live in the connections.

- **Gradient flow.** Backpropagation (Week 2) is the chain rule. Initialization and normalization (Week 3) keep the per-layer Jacobians near norm 1. Residual connections (Week 6) add an identity path so the Jacobian is $\mathbf{I}+\partial F/\partial x$. LSTM cell states (Week 7) do the same across time. Every "vanishing gradient" question is a question about a product of Jacobians.
- **Inductive bias and parameter sharing.** An MLP shares nothing; a CNN shares weights across space (translation equivariance, local connectivity); an RNN shares weights across time; attention shares weights across positions and makes the connectivity data-dependent. Each choice trades flexibility for sample efficiency.
- **Generalization.** Optimization finds a minimum (Week 3); regularization, data augmentation, early stopping and transfer learning (Weeks 4 and 6) decide whether it generalizes. Bias-variance and the train/validation/test discipline apply to everything.

### Must-know derivations {#must-know}

Be able to reproduce each of these from a blank page, defining every symbol.

1. **Gradient of softmax cross-entropy** with respect to the logits: $\partial\mathcal{L}/\partial z_k=p_k-y_k$ (Weeks 1 and 2).
2. **Backprop through a linear layer and an elementwise nonlinearity:** $\delta^{(l)}=(\mathbf{W}^{(l+1)\top}\delta^{(l+1)})\odot\phi'(a^{(l)})$, and the weight gradient as an outer product (Week 2).
3. **Why a deep linear or saturating network has vanishing or exploding gradients,** via the product of Jacobians; **Xavier and He initialization** variances $2/(n_{in}+n_{out})$ and $2/n_{in}$ (Week 3).
4. **Momentum, Adam with bias correction,** and why AdamW decouples weight decay (Week 3). Convergence rate of gradient descent on a quadratic with condition number $\kappa$: factor $(\kappa-1)/(\kappa+1)$.
5. **BatchNorm forward and backward;** train versus eval statistics; LayerNorm versus BatchNorm axes (Week 3).
6. **Bias-variance decomposition** of the expected squared error; **L2 regularization as a Gaussian prior** (MAP), L1 as a Laplace prior and why it gives sparsity; **inverted dropout** scaling (Week 4).
7. **Convolution output size** $\lfloor(n+2p-d(k-1)-1)/s\rfloor+1$, parameter and FLOP counts, **receptive field** growth, and the cost of depthwise-separable versus standard convolution (Weeks 5 and 6).
8. **Why residual connections help:** $\partial x_L/\partial x_l=\mathbf{I}+\partial(\sum F)/\partial x_l$; and the parameter cost of a bottleneck block (Week 6).
9. **IoU, non-maximum suppression, and average precision** computed from a small example (Week 6).
10. **BPTT:** the gradient as a sum over paths, $\partial h_t/\partial h_k=\prod_{j=k+1}^{t}\mathbf{D}_j\mathbf{W}_{hh}$, the spectral-radius argument for vanishing and exploding, and the LSTM cell-state recursion (Week 7).
11. **Attention:** additive and multiplicative scores, softmax weights, context vector, and its gradient (Week 7).
12. **Perplexity and bits per character** from cross-entropy, and the information-theoretic meaning of cross-entropy and KL divergence (Weeks 1 and 7).

### Formula and fact sheet {#fact-sheet}

| Topic | Fact |
| --- | --- |
| Softmax CE gradient | $\boldsymbol{p}-\boldsymbol{y}$ w.r.t. logits |
| He init | $\mathrm{Var}(w)=2/n_{in}$ for ReLU |
| Adam | $m_t=\beta_1m_{t-1}+(1-\beta_1)g_t$, $v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2$, step $\eta\hat m_t/(\sqrt{\hat v_t}+\epsilon)$, with $\hat m=m/(1-\beta_1^t)$ |
| Conv params | $C_{out}(C_{in}k^2+1)$; depthwise-separable $C_{in}k^2+C_{in}C_{out}$ |
| Receptive field | $r_l=r_{l-1}+(k_l-1)\prod_{i<l}s_i$ |
| LSTM params | $4H(d+H+1)$ per layer (one bias vector); PyTorch uses two bias vectors: $4H(d+H+2)$ |
| GRU params | $3H(d+H+1)$ (one bias vector) |
| Perplexity | $\exp(\text{cross-entropy in nats})=2^{\text{bits}}$ |
| Scaled dot-product | scores divided by $\sqrt{d_k}$ (preview of Week 9) |
| Gradient clipping | $g\leftarrow g\cdot\min(1,c/\lVert g\rVert)$ |

### Practice midterm {#practice-midterm}

Twenty-two questions, in the style of the exam. Attempt each on paper before opening the answer. A realistic pace is about 3 minutes for short questions and 10 for derivations.

:::exercise Q1 (Week 1)
A classifier outputs probabilities $(0.7,0.2,0.1)$ for three classes and the true class is the second. Compute the cross-entropy loss in nats and the gradient with respect to the logits.
:::details Answer
Loss $=-\ln0.2\approx1.609$. Gradient $=\boldsymbol{p}-\boldsymbol{y}=(0.7,\ 0.2-1,\ 0.1)=(0.7,-0.8,0.1)$. It sums to zero, as it must for softmax.
:::
:::

:::exercise Q2 (Week 1)
State the relation between cross-entropy $H(p,q)$, entropy $H(p)$ and KL divergence. Why does minimizing cross-entropy with a one-hot target equal maximum likelihood?
:::details Answer
$H(p,q)=H(p)+\KL(p\Vert q)$. With a one-hot target $p$, $H(p)=0$ and $H(p,q)=-\log q_{y}$, the negative log-likelihood of the correct class, so minimizing the average over a dataset is exactly maximum-likelihood estimation of the parameters of $q$.
:::
:::

:::exercise Q3 (Week 2)
Can a single-hidden-layer network with ReLU units represent XOR on $\{0,1\}^2$? Give weights, or explain why not.
:::details Answer
Yes. With two hidden units $h_1=\relu(x_1+x_2)$ and $h_2=\relu(x_1+x_2-1)$ the output $y=h_1-2h_2$ gives $(0,0)\to0$, $(1,0)\to1$, $(0,1)\to1$, $(1,1)\to2-2=0$. A single linear unit cannot (XOR is not linearly separable), so one hidden layer is the minimum.
:::
:::

:::exercise Q4 (Week 2)
For a two-layer network $\hat y=\boldsymbol{w}_2^\top\relu(\mathbf{W}_1\boldsymbol{x})$ with squared loss $\tfrac12(\hat y-y)^2$, write $\partial\mathcal{L}/\partial\mathbf{W}_1$.
:::details Answer
Let $\boldsymbol{a}=\mathbf{W}_1\boldsymbol{x}$, $\boldsymbol{h}=\relu(\boldsymbol{a})$, $e=\hat y-y$. Then $\partial\mathcal{L}/\partial\boldsymbol{h}=e\,\boldsymbol{w}_2$, $\partial\mathcal{L}/\partial\boldsymbol{a}=e\,\boldsymbol{w}_2\odot\ind[\boldsymbol{a}>0]$, and $\partial\mathcal{L}/\partial\mathbf{W}_1=(e\,\boldsymbol{w}_2\odot\ind[\boldsymbol{a}>0])\,\boldsymbol{x}^\top$, an outer product of shape (hidden, input).
:::
:::

:::exercise Q5 (Week 2)
Reverse-mode versus forward-mode autodiff: which is cheaper for a function from $\R^{10^6}$ to $\R$ and why?
:::details Answer
Reverse mode: one backward pass yields the gradient with respect to all $10^6$ inputs at a cost of a small constant times a forward pass; forward mode would need $10^6$ passes (one per input direction). The price of reverse mode is memory for stored activations.
:::
:::

:::exercise Q6 (Week 3)
A 50-layer plain ReLU network is initialized with $\mathrm{Var}(w)=1/n_{in}$ instead of $2/n_{in}$. What happens to the activation variance across layers? Quantify.
:::details Answer
For a ReLU layer, the pre-activation variance is $n_{in}\,\mathrm{Var}(w)\,\mathbb{E}[h^2]$ and $\mathbb{E}[h^2]=\tfrac12\mathrm{Var}(a_l)$ for a symmetric pre-activation, so with $\mathrm{Var}(w)=1/n_{in}$ each layer multiplies the second moment by $1/2$. After 50 layers it shrinks by $2^{-50}\approx10^{-15}$ (standard deviation by about $2^{-25}\approx3\times10^{-8}$): the signal and the gradient vanish. He initialization ($2/n_{in}$) makes the factor 1.
:::
:::

:::exercise Q7 (Week 3)
Gradient descent on $f(x)=\tfrac12(100x_1^2+x_2^2)$: what is the largest stable learning rate, and how many iterations (order of magnitude) are needed to shrink the $x_2$ coordinate by a factor $10^{-3}$ at that rate?
:::details Answer
Curvatures are 100 and 1; stability needs $\eta<2/\lambda_{\max}=0.02$. At $\eta\approx0.02$ the $x_2$ contraction per step is $1-\eta\lambda_{\min}=0.98$. To reach $10^{-3}$: $n=\ln(10^{-3})/\ln(0.98)\approx342$ steps. The condition number 100 forces the tiny step that makes the shallow direction slow, which is what momentum and preconditioning address.
:::
:::

:::exercise Q8 (Week 3)
Why does Adam need bias correction? At step $t=1$ with $\beta_1=0.9$, what are $m_1$ and $\hat m_1$ for gradient $g_1=2$?
:::details Answer
$m_0=0$ is a biased initialization, so early averages are shrunk toward 0. $m_1=0.1\cdot2=0.2$; the corrected value is $\hat m_1=m_1/(1-0.9)=2$, the true gradient. Without correction the first moment estimate would be ten times too small at step 1, and the second-moment estimate $v_t$ (with $\beta_2=0.999$) a thousand times too small, so the ratio $m_t/\sqrt{v_t}$ would be distorted in the first steps.
:::
:::

:::exercise Q9 (Week 3)
BatchNorm during training versus inference: what statistics are used, and what breaks with batch size 1?
:::details Answer
Training uses the mean and variance of the current mini-batch (per channel) and updates running averages; inference uses the running averages. With batch size 1 (or very small batches) the batch statistics are extremely noisy, and for a fully connected layer the variance over one example is zero (or for conv over a few positions), so training is unstable and the train/inference mismatch grows. Use GroupNorm or LayerNorm in such cases.
:::
:::

:::exercise Q10 (Week 4)
Explain how L2 regularization with coefficient $\lambda$ relates to a Gaussian prior, and what the gradient update looks like (weight decay).
:::details Answer
Minimizing $\mathcal{L}+\tfrac\lambda2\lVert\boldsymbol{w}\rVert^2$ is MAP estimation with a prior $\boldsymbol{w}\sim\mathcal{N}(0,\lambda^{-1}\mathbf{I})$ (up to scaling of the loss by the data size). The SGD update becomes $\boldsymbol{w}\leftarrow(1-\eta\lambda)\boldsymbol{w}-\eta\nabla\mathcal{L}$: the weights decay multiplicatively each step. For Adam this equivalence breaks (the penalty gradient is rescaled by the adaptive denominator), which is the motivation for AdamW's decoupled decay.
:::
:::

:::exercise Q11 (Week 4)
Training loss 0.05, validation loss 0.90, both flat after epoch 20. Diagnose and list three remedies in order of what you would try first.
:::details Answer
Overfitting (high variance): a large generalization gap with a very low training error. Remedies: (1) more data or stronger data augmentation; (2) regularization (weight decay, dropout) and early stopping on validation loss; (3) a smaller or pretrained model. Check first that the validation set is drawn from the same distribution and that no leakage or preprocessing mismatch exists.
:::
:::

:::exercise Q12 (Week 4)
Inverted dropout with keep probability $p=0.8$: how are activations scaled at train time, and why does nothing need to change at test time?
:::details Answer
Surviving activations are divided by $p$ (multiplied by 1.25) during training, so the expected activation $\mathbb{E}[\tilde h]=h$ equals the deterministic one. At test time dropout is switched off and no rescaling is needed because the expectation already matches.
:::
:::

:::exercise Q13 (Week 5)
Input $64\times64\times3$, a conv layer with 32 filters of size $5\times5$, stride 2, padding 2. Give the output shape and parameter count. Then add a second conv layer $3\times3$, stride 1, padding 1, 64 filters: parameters and receptive field of the output unit.
:::details Answer
Output size $\lfloor(64+4-5)/2\rfloor+1=32$, so $32\times32\times32$. Parameters: $32(3\cdot25+1)=2432$. Second layer: $64(32\cdot9+1)=18{,}496$ parameters, output $32\times32\times64$. Receptive field: after layer 1, $r=5$ with jump 2; layer 2 adds $(3-1)\cdot2=4$, so $r=9$ input pixels.
:::
:::

:::exercise Q14 (Week 5)
Equivariance versus invariance: which does a convolution have, which does global average pooling add, and why does that matter for classification versus segmentation?
:::details Answer
Convolution (with weight sharing) is translation-equivariant: shifting the input shifts the feature map. Global average pooling (and, approximately, max pooling plus subsampling) discards position and gives invariance to translation. Classification wants invariance, so pooling to a vector is fine; segmentation and detection need position-aware outputs, so you keep spatial maps (equivariance) and upsample or use skip connections (U-Net).
:::
:::

:::exercise Q15 (Week 6)
A bottleneck residual block (1x1 reduce to 64, 3x3 at 64, 1x1 expand to 256) takes a 256-channel input. Count the weights (ignore biases), and compare with two $3\times3$ convolutions at 256 channels.
:::details Answer
Bottleneck: $256\cdot64+64\cdot64\cdot9+64\cdot256=16{,}384+36{,}864+16{,}384=69{,}632$. Two $3\times3$ convolutions at 256 channels: $2\cdot256\cdot256\cdot9=1{,}179{,}648$. The bottleneck has about 17 times fewer weights while providing three nonlinear layers; this is why ResNet-50 and deeper nets use it.
:::
:::

:::exercise Q16 (Week 6)
Why does the gradient of a residual network not vanish with depth as quickly as that of a plain network? Write the relevant expression.
:::details Answer
With $x_{l+1}=x_l+F(x_l)$, unrolling gives $x_L=x_l+\sum_{i=l}^{L-1}F(x_i)$, so $\partial\mathcal{L}/\partial x_l=\partial\mathcal{L}/\partial x_L\cdot(\mathbf{I}+\partial\sum F/\partial x_l)$. The identity term delivers the loss gradient to every earlier layer directly, regardless of $F$. In a plain network the gradient is a pure product of $L-l$ Jacobians.
:::
:::

:::exercise Q17 (Week 6)
Two boxes, $(0,0,4,4)$ and $(2,2,6,6)$ in $(x_1,y_1,x_2,y_2)$ format. Compute the IoU. Would NMS with threshold 0.5 suppress one of them?
:::details Answer
Intersection: $[2,4]\times[2,4]$, area 4. Areas: 16 and 16. Union $=16+16-4=28$. IoU $=4/28\approx0.143<0.5$, so NMS keeps both.
:::
:::

:::exercise Q18 (Week 6)
You have 2000 labelled images and a ResNet pretrained on ImageNet. Describe the transfer-learning recipe, and say when you would fine-tune the whole network versus only the last layer.
:::details Answer
Replace the classification head, train it first with the backbone frozen (BatchNorm in eval mode), then optionally unfreeze all or the upper blocks with a much smaller learning rate (discriminative rates: lower for earlier layers), with augmentation. Freeze almost everything when data is tiny or very similar to ImageNet; fine-tune more when you have more data or the domain differs (medical, satellite), since early features are generic but late ones are task-specific.
:::
:::

:::exercise Q19 (Week 7)
Derive $\partial h_t/\partial h_k$ for $h_t=\tanh(\mathbf{W}_{hh}h_{t-1}+\mathbf{W}_{xh}x_t)$ and use the spectral norm to bound it. State the condition for guaranteed vanishing.
:::details Answer
$\partial h_t/\partial h_{t-1}=\diag(1-h_t^2)\,\mathbf{W}_{hh}=\mathbf{D}_t\mathbf{W}_{hh}$, so $\partial h_t/\partial h_k=\prod_{j=k+1}^{t}\mathbf{D}_j\mathbf{W}_{hh}$. Since $\lVert\mathbf{D}_j\rVert\le1$, $\lVert\partial h_t/\partial h_k\rVert\le(\gamma\,\lVert\mathbf{W}_{hh}\rVert_2)^{t-k}$ with $\gamma=\max\lVert\mathbf{D}\rVert\le1$. If $\gamma\lVert\mathbf{W}_{hh}\rVert_2<1$ the gradient vanishes exponentially. The converse (explosion when the spectral radius exceeds 1) is a necessary-condition statement, not a guarantee, because the $\mathbf{D}_j$ shrink things when units saturate.
:::
:::

:::exercise Q20 (Week 7)
Write the LSTM equations and explain which term gives it better gradient flow than a vanilla RNN. What forget-gate bias initialization is common and why?
:::details Answer
$f=\sigma(\mathbf{W}_f[h_{t-1},x_t]+b_f)$, $i=\sigma(\cdot)$, $g=\tanh(\cdot)$, $o=\sigma(\cdot)$, $c_t=f\odot c_{t-1}+i\odot g$, $h_t=o\odot\tanh(c_t)$. The additive update in $c_t$ gives the direct path $\partial c_t/\partial c_{t-1}=\diag(f_t)$, so memory is preserved when $f\approx1$ without being multiplied by $\mathbf{W}_{hh}$ at each step. Setting $b_f\approx1$ initially makes $f\approx0.73$, so the cell remembers by default and the network must learn to forget.
:::
:::

:::exercise Q21 (Week 7)
A language model has a validation cross-entropy of 3.2 nats per token. Report its perplexity. If the tokenizer averages 4.2 characters per token, what is its bits per character?
:::details Answer
Perplexity $=e^{3.2}\approx24.5$. Bits per token $=3.2/\ln2\approx4.62$; per character $4.62/4.2\approx1.10$ bpc.
:::
:::

:::exercise Q22 (Week 7)
Explain exposure bias and name two mitigations. Why can a bidirectional RNN not be used as a language model?
:::details Answer
Under teacher forcing the decoder always sees gold previous tokens in training but its own (possibly wrong) predictions at test time, so errors compound along a trajectory it never trained on. Mitigations: scheduled sampling (mix in own predictions), sequence-level training or minimum-risk objectives, beam search with length normalization, and more or better data. A bidirectional RNN's backward states summarize the future tokens, including the one being predicted, so training would leak the answer and the model could not generate left to right.
:::
:::

:::exercise Q23 (Week 7, bonus)
Count the parameters of `nn.GRU(input_size=50, hidden_size=100)` with biases, one layer, unidirectional.
:::details Answer
Three gate blocks: $3\cdot(100\cdot50+100\cdot100+100+100)=3\cdot(5000+10000+200)=45{,}600$ (PyTorch keeps two bias vectors per gate).
:::
:::

:::exercise Q24 (Cross-cutting)
You are training a small LSTM language model and the loss is `nan` after 300 steps. List a diagnosis order and the likely fixes.
:::details Answer
(1) Check the data and the loss at step 0 (should be near $\ln V$). (2) Log the gradient norm: a spike before the nan indicates exploding gradients, so clip the global norm (for example 1 to 5) and lower the learning rate. (3) Check for division or log of zero (use `logsigmoid`, `cross_entropy` on logits, not `log(softmax)`). (4) Check mixed precision overflow (use loss scaling or bf16). (5) Check inputs for infs or nans (corrupt batch). (6) Verify the initialization of the recurrent weights (orthogonal helps) and the forget-gate bias.
:::
:::

### Exam strategy {#exam-strategy}

- Write down what is given and what is asked, with shapes. Half of the mistakes in the course are shape mistakes.
- For derivations, state the chain-rule structure first, then fill in the pieces, and check dimensions at the end.
- For "diagnose" questions, give the symptom-to-cause chain, then the cheapest fix and the check that would confirm it.
- Numbers: leave expressions like $e^{3.2}$ if you do not have a calculator, but show the setup.

## Assignment 2 and the final project proposal {#assignment-proposal}

Two deliverables are due at the end of this week.

### Assignment 2: CNN with transfer learning, and an RNN sequence task {#assignment-2}

The assignment has a code part (auto-checked where possible) and a 2 to 4 page report. It covers [Weeks 5 to 7](week-05.html): a convolutional classifier with transfer learning, and a recurrent model on a sequence task.

A strong submission contains:

- **Part A (CNN and transfer learning).** A from-scratch small CNN baseline, then a pretrained backbone fine-tuned on the target dataset with a justified recipe (frozen head first, then unfreezing, a learning-rate choice, augmentation). Report test accuracy for both, with at least three seeds and the standard deviation, a training-curve plot, and a short analysis of errors (a confusion matrix or example failures). Include an ablation of at least one choice (for example number of unfrozen blocks or augmentation).
- **Part B (sequence task).** An RNN, LSTM or GRU on the assigned task (for example a character-level model or sequence classification), with a correct masking and padding scheme, gradient clipping, and a comparison between at least two cell types or between unidirectional and bidirectional. Include a gradient-norm plot and a discussion of what failed before it worked.
- **A clear report.** Problem, method, experiment table with baselines, ablations, and honest discussion of limits. Figures with labelled axes and units; every claim backed by a number.
- **Reproducibility.** Seeds, versions, a README that runs end to end, and no test-set tuning (hyperparameters chosen on validation data only).

### Final project proposal (1 page) {#project-proposal}

The proposal is the first milestone of the final project (teams of 1 to 3). A good one-page proposal contains four things, each short and concrete:

1. **Problem.** What are you trying to do and why does it matter? State the input, the output, and the success criterion in one or two sentences. A reader should know exactly what "it works" means.
2. **Dataset.** Name the data, its size, its license, how you will split it (train, validation, test, with leakage in mind), and any labelling or cleaning cost. Confirm that you can access it before submitting.
3. **Baseline.** At least one simple baseline you will run first (majority class, logistic regression on features, a pretrained model used off the shelf) with the metric you expect it to get, or how you will measure it. A project without a baseline cannot claim improvement.
4. **Plan.** A week-by-week plan to the progress report (Week 16), the main model and ablations you intend to run, the compute you need, the risks (data, time, compute) and a fallback if the main idea does not work. Say who does what.

Common weaknesses to avoid: a vague problem ("apply deep learning to X"), no evaluation metric, a dataset that does not exist yet, a scope that cannot be finished by the progress report, and a project that only replicates a tutorial. Good projects are narrow, measurable, and have a surprising question inside them (an ablation, a failure analysis, a comparison that is not obvious).

## Key takeaways {#takeaways}

:::key Key takeaways
- A representation maps discrete objects to vectors so that a network can use them. Distributed representations make similarity geometric and let statistical strength be shared between related items.
- The distributional hypothesis (words in similar contexts have similar meanings) drives word2vec, GloVe and fastText. Skip-gram with negative sampling trains two embedding tables with $K+1$ binary logistic losses per pair at cost $O(Kd)$, instead of a $V$-way softmax; its optimum is a PMI matrix shifted by $\log K$.
- GloVe regresses on log co-occurrence counts with a weighting function; fastText sums character $n$-gram vectors so that rare and unseen words get vectors.
- Use cosine similarity with normalized vectors. Analogy tests are a sanity check with known caveats; evaluate embeddings both intrinsically and on a downstream task.
- Embeddings absorb associations in the training text, including social stereotypes. Measure with projections and WEAT, report the word lists and limits, and remember that debiasing by projection is partial.
- Subword tokenization (BPE, WordPiece, Unigram/SentencePiece, byte-level BPE) balances vocabulary size against sequence length and removes unknown words. BPE greedily merges the most frequent adjacent pair; the tokenizer is part of the model and must match.
- Character and byte-level models trade a tiny vocabulary and robustness for longer sequences.
- Users, items, nodes and entities can all be embedded; matrix factorization is an embedding model with a dot-product score trained on observed entries only.
- Metric learning shapes distances directly: contrastive and triplet losses use margins and need mining, while InfoNCE is a softmax over negatives whose temperature controls how hard negatives are emphasized and whose value is bounded by $\log N$.
- The midterm covers Weeks 1 to 7: derive gradients and Jacobian products, count parameters and shapes, and diagnose training failures from symptoms.
:::

## Further reading {#further-reading}

**Essential**

- Mikolov, T., Sutskever, I., Chen, K., Corrado, G. and Dean, J. (2013). *Distributed Representations of Words and Phrases and their Compositionality.* NeurIPS. arXiv:1310.4546. (Negative sampling, subsampling and phrases.)
- Pennington, J., Socher, R. and Manning, C. D. (2014). *GloVe: Global Vectors for Word Representation.* EMNLP.
- Sennrich, R., Haddow, B. and Birch, A. (2016). *Neural Machine Translation of Rare Words with Subword Units.* ACL. arXiv:1508.07909.

**Recommended**

- Mikolov, T., Chen, K., Corrado, G. and Dean, J. (2013). *Efficient Estimation of Word Representations in Vector Space.* arXiv:1301.3781.
- Levy, O. and Goldberg, Y. (2014). *Neural Word Embedding as Implicit Matrix Factorization.* NeurIPS.
- Levy, O., Goldberg, Y. and Dagan, I. (2015). *Improving Distributional Similarity with Lessons Learned from Word Embeddings.* TACL 3.
- Bojanowski, P., Grave, E., Joulin, A. and Mikolov, T. (2017). *Enriching Word Vectors with Subword Information.* TACL 5. arXiv:1607.04606.
- Bolukbasi, T., Chang, K.-W., Zou, J., Saligrama, V. and Kalai, A. (2016). *Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings.* NeurIPS. arXiv:1607.06520.
- Caliskan, A., Bryson, J. J. and Narayanan, A. (2017). *Semantics derived automatically from language corpora contain human-like biases.* Science 356.
- Gonen, H. and Goldberg, Y. (2019). *Lipstick on a Pig: Debiasing Methods Cover up Systematic Gender Biases in Word Embeddings But do not Remove Them.* NAACL. arXiv:1903.03862.
- Kudo, T. and Richardson, J. (2018). *SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing.* EMNLP (demo). arXiv:1808.06226.
- Kudo, T. (2018). *Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates.* ACL. arXiv:1804.10959.
- Kim, Y., Jernite, Y., Sontag, D. and Rush, A. M. (2016). *Character-Aware Neural Language Models.* AAAI. arXiv:1508.06615.
- Koren, Y., Bell, R. and Volinsky, C. (2009). *Matrix Factorization Techniques for Recommender Systems.* IEEE Computer 42(8).
- Perozzi, B., Al-Rfou, R. and Skiena, S. (2014). *DeepWalk: Online Learning of Social Representations.* KDD. arXiv:1403.6652.
- Bordes, A., Usunier, N., Garcia-Duran, A., Weston, J. and Yakhnenko, O. (2013). *Translating Embeddings for Modeling Multi-relational Data.* NeurIPS.
- Schroff, F., Kalenichenko, D. and Philbin, J. (2015). *FaceNet: A Unified Embedding for Face Recognition and Clustering.* CVPR. arXiv:1503.03832.
- van den Oord, A., Li, Y. and Vinyals, O. (2018). *Representation Learning with Contrastive Predictive Coding.* arXiv:1807.03748.
- Chen, T., Kornblith, S., Norouzi, M. and Hinton, G. (2020). *A Simple Framework for Contrastive Learning of Visual Representations.* ICML. arXiv:2002.05709.
- Hadsell, R., Chopra, S. and LeCun, Y. (2006). *Dimensionality Reduction by Learning an Invariant Mapping.* CVPR.
- Goldberg, Y. (2016). *A Primer on Neural Network Models for Natural Language Processing.* JAIR 57. arXiv:1510.00726.
- Jurafsky, D. and Martin, J. H. *Speech and Language Processing* (3rd ed. draft), chapter on vector semantics and embeddings. Free online.
