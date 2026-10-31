---
title: "Graph Neural Networks and Geometric Deep Learning"
phase: Advanced Topics
summary: Learning on graphs and other structured domains. Graph data and tasks, adjacency, degree and Laplacian matrices, the message-passing framework, GCN (with a derivation of normalized propagation and its spectral motivation), GraphSAGE, GAT and GIN, expressive power and the Weisfeiler-Lehman test, over-smoothing and over-squashing with their remedies, pooling and readout, node, link and graph tasks, graph Transformers, and the geometric deep learning blueprint (symmetry, invariance, equivariance) with point clouds, PointNet and equivariant networks. The lab uses PyTorch Geometric on Cora and a molecular dataset.
---

:::key Learning objectives
By the end of this week you should be able to:
- Describe how molecules, social networks, knowledge graphs and recommender systems are represented as graphs, and name the node-, edge- and graph-level tasks defined on them.
- Build the adjacency, degree and Laplacian matrices of a graph, and explain what the Laplacian quadratic form and its eigenvectors measure.
- State the message-passing framework (message, aggregate, update, readout) and prove that it is permutation equivariant.
- Derive the GCN propagation rule $\hat{\mathbf A}=\tilde{\mathbf D}^{-1/2}(\mathbf A+\mathbf I)\tilde{\mathbf D}^{-1/2}$ from a first-order spectral filter and the renormalization trick, and compare it with GraphSAGE, GAT and GIN.
- State precisely what the 1-Weisfeiler-Lehman (1-WL) test is, what message-passing GNNs can and cannot distinguish, and why GIN uses sum aggregation and an MLP.
- Explain over-smoothing (with a Dirichlet-energy argument) and over-squashing (with a receptive-field argument), and list remedies: residual and initial connections, normalization, DropEdge, jumping knowledge, rewiring, and virtual nodes.
- Choose a readout or pooling operator (sum, mean, max, attention, hierarchical) and say what each one can and cannot count.
- Explain how graph Transformers use positional and structural encodings to inject graph structure into full attention.
- State the geometric deep learning blueprint in terms of groups, invariance and equivariance, and identify the symmetry behind CNNs, GNNs, DeepSets and PointNet.
- Implement PointNet and an E(n)-equivariant layer, and verify invariance and equivariance numerically.
- Train GNNs with PyTorch Geometric for node classification on Cora and graph classification on a molecular dataset.
:::

Everything you have built so far assumed a regular layout. A CNN ([Week 5](week-05.html)) needs pixels on a grid, an RNN or Transformer ([Week 7](week-07.html), [Week 9](week-09.html)) needs tokens in a line. Much of the interesting data in science and industry has no such layout. A molecule is a set of atoms joined by bonds; there is no "left" or "up" in it. A social network is a set of people and friendships. A road network, a protein interaction map, the citation record of a research field, the purchase history of a million shoppers: all of these are **graphs**, where the objects matter and the *relations between them* matter equally.

A graph neural network (GNN) is a neural network designed so that its computation respects the structure of a graph. The central design constraint is easy to state and surprisingly deep. There is no canonical order of the nodes, so the network must give the same answer however you number them. That single requirement, **permutation symmetry**, forces the architecture: each node updates itself by pooling over its neighbours with an order-independent function, and this local operation is repeated layer after layer. This is **message passing**.

This week has two halves. The first half builds the graph-learning toolbox: representations, message passing, the four workhorse layers (GCN, GraphSAGE, GAT, GIN), a rigorous look at what such networks can and cannot compute, why deep GNNs misbehave, how to turn node embeddings into graph-level predictions, and where Transformers fit. The second half zooms out. Permutation invariance is one example of a general principle, and CNNs, GNNs, DeepSets and equivariant 3-D networks are all instances of the same **geometric deep learning** blueprint: choose the symmetry group of your domain and build layers that commute with it. The lab uses PyTorch Geometric (PyG) on a citation network and on a molecular dataset.

## Graph-structured data {#data}

A **graph** $G=(V,E)$ consists of a set of nodes (vertices) $V$ with $|V|=n$ and a set of edges $E\subseteq V\times V$. Each node $v$ may carry a feature vector $\boldsymbol{x}_v\in\R^{d}$, each edge $(u,v)$ may carry a feature vector $\boldsymbol{e}_{uv}$, and the whole graph may carry a label or target. Edges can be undirected or directed, weighted or unweighted. A graph can also have several *types* of nodes and edges (a **heterogeneous** graph). Before touching any math, look at how four very different domains map onto this one abstraction.

**Molecules.** Nodes are atoms, edges are covalent bonds. Node features encode the element (one-hot over C, N, O, ...), formal charge, hybridization or aromaticity; edge features encode the bond type (single, double, triple, aromatic). The typical task is **graph-level regression or classification**: predict a property of the whole molecule such as solubility, toxicity, binding affinity or an energy. A typical drug-like molecule has 20 to 50 heavy atoms, so the graphs are small, but there are millions of them. Molecular graphs are also *sparse* (an atom has at most about four bonds) and *planar-ish*, and the identity of a molecule is the graph itself, independent of how the atoms are numbered in a file. This is the setting where the invariance requirement is most obvious: a toxicity predictor that depends on atom numbering is simply wrong.

**Social and information networks.** Nodes are users, papers or web pages; edges are friendships, citations or hyperlinks. These graphs can be enormous (billions of edges), have heavy-tailed degree distributions (a few hubs with millions of neighbours), and usually come with sparse labels. Typical tasks are **node classification** (which research area is this paper in? is this account a bot?) and **link prediction** (who will this user befriend?). Most of this graph is unlabeled, which is why *semi-supervised* learning, where the structure itself spreads information from the few labeled nodes, is the classic setup.

**Knowledge graphs.** Nodes are entities (Marie Curie, Warsaw, physics) and edges are typed relations (born_in, field_of). The data is a set of triples $(h, r, t)$. Typical tasks are **link prediction** (complete the missing triple: is $(\text{Curie}, \text{born\_in}, ?)$ answerable?) and entity classification. The typed edges mean a plain GNN is not enough; Relational GCN (R-GCN, [Schlichtkrull et al., 2018](https://arxiv.org/abs/1703.06103)) gives every relation its own weight matrix.

**Recommender systems.** Users and items form a **bipartite** graph whose edges are interactions (clicks, purchases, ratings). Recommendation is link prediction on this graph: which unobserved user-item edge is most likely? GNN-based recommenders (PinSage at Pinterest, 2018; LightGCN, 2020) learn user and item embeddings by propagating over the interaction graph, so that two items bought by overlapping sets of users end up close together.

Other domains follow the same pattern: proteins (residues as nodes, contacts as edges), road networks (intersections and segments, used for traffic and arrival-time prediction), physical simulation (particles or mesh vertices with interactions), circuits, code (abstract syntax trees and data-flow graphs), and scene graphs of images. What these have in common is that the **relational structure is part of the input**, of variable size, and has no natural ordering.

{{figure: week14-graph-tasks | The three levels of prediction on graphs. Node-level: label every node, given labels on a few (semi-supervised). Edge-level: predict whether or which edges exist. Graph-level: one prediction for the whole graph, as for molecules.}}

### Why not just flatten the graph?

The obvious baseline is to flatten the adjacency matrix and feed it to an MLP. It fails for three independent reasons. First, graphs have **variable size**, while an MLP has a fixed input width. Second, the matrix depends on an arbitrary node ordering: the same graph has up to $n!$ different adjacency matrices, and an MLP would have to learn all of them separately. Third, an $n\times n$ input has $O(n^2)$ entries even when the graph has only $O(n)$ edges. We want a model whose parameters do not depend on $n$, whose output does not depend on the ordering, and whose cost scales with the number of edges. GNNs satisfy all three.

:::history
**A short history.** Neural networks on graphs go back to Gori, Monfardini and Scarselli (2005) and the "graph neural network model" of Scarselli et al. (2009), which iterated a recurrent update to a fixed point. Convolutions on graphs came from the spectral side (Bruna et al., 2014; Defferrard et al., 2016, using Chebyshev polynomials). The pivotal simplification was Kipf and Welling's GCN in 2017, a two-line propagation rule that worked remarkably well on citation graphs. In the same year Hamilton et al. introduced GraphSAGE for inductive learning on large graphs, and Gilmer et al. unified many models as "message passing neural networks" for quantum chemistry. Velickovic et al. added attention in 2018 (GAT), Xu et al. tied GNN power to the Weisfeiler-Lehman test in 2019 (GIN), and the 2021 monograph by Bronstein, Bruna, Cohen and Velickovic placed all of this inside "geometric deep learning".
:::

## Graph basics: adjacency, degree and the Laplacian {#basics}

Number the nodes $1,\dots,n$. The **adjacency matrix** $\mathbf A\in\{0,1\}^{n\times n}$ has $A_{ij}=1$ if there is an edge from $i$ to $j$. For an undirected graph $\mathbf A$ is symmetric. The **degree** of a node is its number of neighbours, $d_i=\sum_j A_{ij}$, and the **degree matrix** is $\mathbf D=\diag(d_1,\dots,d_n)$. Stack node features into $\mathbf X\in\R^{n\times d}$, whose row $i$ is $\boldsymbol{x}_i^\top$.

Multiplying by $\mathbf A$ is the fundamental graph operation. For a feature matrix $\mathbf X$,

$$
(\mathbf A\mathbf X)_{i}=\sum_{j}A_{ij}\,\boldsymbol{x}_j=\sum_{j\in\mathcal N(i)}\boldsymbol{x}_j ,
$$

which is the **sum of the features of the neighbours** of node $i$. One matrix product aggregates every neighbourhood simultaneously. Likewise $\mathbf D^{-1}\mathbf A\mathbf X$ is the neighbour *mean*, and $\mathbf A^k$ counts walks of length $k$: $(\mathbf A^k)_{ij}$ is the number of walks from $i$ to $j$ with exactly $k$ edges. So a $k$-layer network that multiplies by $\mathbf A$ at each layer mixes information within $k$ hops. Keep this in mind; it is the reason depth equals receptive-field radius in a GNN.

### The graph Laplacian

The **combinatorial Laplacian** is $\mathbf L=\mathbf D-\mathbf A$ and the **symmetric normalized Laplacian** is

$$
\mathbf L_{\mathrm{sym}}=\mathbf I-\mathbf D^{-1/2}\mathbf A\,\mathbf D^{-1/2}.
$$

The Laplacian is the graph analogue of the second derivative, and its quadratic form measures how rough a signal is. Let $\boldsymbol{x}\in\R^n$ assign one scalar to each node. Then

$$
\boldsymbol{x}^\top\mathbf L\,\boldsymbol{x}=\sum_i d_i x_i^2-\sum_{i,j}A_{ij}x_ix_j=\frac12\sum_{i,j}A_{ij}(x_i-x_j)^2=\sum_{(i,j)\in E}(x_i-x_j)^2 .
$$

(The middle step expands the square and uses $\sum_j A_{ij}=d_i$; the last sums each undirected edge once.) This **Dirichlet energy** is zero exactly when $\boldsymbol{x}$ is constant on every connected component, and it is large when neighbouring nodes disagree. So the Laplacian measures **smoothness over the graph**. Three facts about its spectrum matter for GNNs:

- $\mathbf L$ is symmetric positive semi-definite, so it has real eigenvalues $0=\lambda_1\le\lambda_2\le\dots\le\lambda_n$ and orthonormal eigenvectors.
- The multiplicity of the eigenvalue $0$ equals the number of connected components. The eigenvector for a connected graph is the constant vector.
- Eigenvectors with small eigenvalues vary slowly over the graph ("low frequency") and those with large eigenvalues oscillate between neighbours ("high frequency"), exactly like sines and cosines of increasing frequency. The eigenvalues of $\mathbf L_{\mathrm{sym}}$ lie in $[0,2]$.

The second eigenvector (the **Fiedler vector**) is the smoothest non-constant signal and is the basis of spectral clustering: its sign pattern splits the graph into two clusters with few edges between them. The code below builds all of these on a toy graph made of two triangles joined by a bridge.

```python:graph_basics.py
import numpy as np

# a small graph: two triangles joined by a bridge (nodes 0-2 and 3-5)
edges = [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5), (2, 3)]
n = 6
A = np.zeros((n, n))
for u, v in edges:
    A[u, v] = A[v, u] = 1.0            # undirected: symmetric adjacency

deg = A.sum(axis=1)                    # degrees d_i = sum_j A_ij
D = np.diag(deg)
L = D - A                              # combinatorial Laplacian
L_sym = np.eye(n) - np.diag(deg**-0.5) @ A @ np.diag(deg**-0.5)   # normalized Laplacian

print("degrees:", deg)
evals, evecs = np.linalg.eigh(L_sym)
print("eigenvalues of L_sym:", np.round(evals, 3))

# Dirichlet energy x^T L x = sum over edges (x_u - x_v)^2
x = np.array([1., 1., 1., -1., -1., -1.])
print("x^T L x =", x @ L @ x, " edge-sum =", sum((x[u]-x[v])**2 for u, v in edges))

# Fiedler vector of L: sign pattern recovers the two clusters (up to a global sign)
w, V = np.linalg.eigh(L)
print("Fiedler vector signs:", np.sign(V[:, 1]).astype(int))
```

```text
degrees: [2. 2. 3. 3. 2. 2.]
eigenvalues of L_sym: [-0.     0.205  1.167  1.5    1.5    1.629]
x^T L x = 4.0  edge-sum = 4.0
Fiedler vector signs: [-1 -1 -1  1  1  1]
```

The signal that is $+1$ on one triangle and $-1$ on the other has energy $4$: only the bridge edge contributes, and it contributes $(1-(-1))^2=4$. The single eigenvalue $0$ confirms one connected component, and the tiny second eigenvalue $0.205$ says the graph is nearly disconnected (there is a bottleneck). Remember this bottleneck picture; it comes back as over-squashing.

### Permutations and the symmetry we need

Relabelling nodes with a permutation $\pi$ is represented by a permutation matrix $\mathbf P$ (exactly one $1$ per row and column). The relabelled graph has adjacency $\mathbf P\mathbf A\mathbf P^\top$ and features $\mathbf P\mathbf X$. Any sensible model $f$ must treat $(\mathbf A,\mathbf X)$ and $(\mathbf P\mathbf A\mathbf P^\top,\mathbf P\mathbf X)$ as the same graph. For *node-level* outputs this means the outputs get permuted along with the nodes (**equivariance**); for *graph-level* outputs the output must not change at all (**invariance**):

$$
f(\mathbf P\mathbf A\mathbf P^\top,\mathbf P\mathbf X)=\mathbf P f(\mathbf A,\mathbf X)\quad\text{(equivariant)},\qquad
g(\mathbf P\mathbf A\mathbf P^\top,\mathbf P\mathbf X)=g(\mathbf A,\mathbf X)\quad\text{(invariant)}.
$$

We will make these precise and general in the [geometric deep learning](#gdl) section. For now, every layer we design must satisfy the first identity, and the final readout must satisfy the second.

## The message-passing framework {#mpnn}

Here is the idea of a GNN in one sentence: **every node repeatedly collects information from its neighbours and updates its own representation**. Let $\boldsymbol{h}_v^{(0)}=\boldsymbol{x}_v$. One layer of **message passing** computes, for every node $v$,

$$
\boldsymbol{a}_v^{(\ell)}=\operatorname{AGGREGATE}^{(\ell)}\Big(\big\{\!\!\big\{\ \operatorname{MESSAGE}^{(\ell)}\big(\boldsymbol{h}_u^{(\ell-1)},\boldsymbol{h}_v^{(\ell-1)},\boldsymbol{e}_{uv}\big)\ :\ u\in\mathcal N(v)\ \big\}\!\!\big\}\Big),
\qquad
\boldsymbol{h}_v^{(\ell)}=\operatorname{UPDATE}^{(\ell)}\big(\boldsymbol{h}_v^{(\ell-1)},\boldsymbol{a}_v^{(\ell)}\big).
$$

The double braces denote a **multiset** (neighbours can repeat a value, and order is meaningless). Three ingredients:

1. **MESSAGE** turns a neighbour's state (optionally with the receiver's state and the edge features) into a message vector. It is often just a linear map $\mathbf W\boldsymbol{h}_u$.
2. **AGGREGATE** combines a variable number of messages into one fixed-size vector. It must be **permutation invariant**: sum, mean, max, or a learned but symmetric function. This is where the symmetry is enforced.
3. **UPDATE** merges the aggregated message with the node's own previous state, typically with an MLP and a nonlinearity.

After $L$ layers, the final node embeddings $\boldsymbol{h}_v^{(L)}$ feed a task head: a linear classifier for node classification, a dot-product or MLP over a pair of embeddings for link prediction, or a **readout** that pools all nodes for graph-level prediction,

$$
\boldsymbol{h}_G=\operatorname{READOUT}\big(\{\!\!\{\boldsymbol{h}_v^{(L)}:v\in V\}\!\!\}\big).
$$

This template was named the **message passing neural network (MPNN)** by Gilmer, Schoenholz, Riley, Vinyals and Dahl (2017, ICML), who used it for predicting quantum-chemical properties of molecules, and it subsumes nearly every layer in the rest of this section.

{{figure: week14-message-passing | One round of message passing at node $v$: neighbour states become messages, a permutation-invariant AGGREGATE reduces them to one vector, and UPDATE combines it with $v$'s own state.}}

### Receptive fields: depth equals hops

After one layer, $\boldsymbol{h}_v^{(1)}$ depends on $v$ and its immediate neighbours. After two layers it depends on the 2-hop neighbourhood, because the neighbours' states already contained *their* neighbours. In general, $\boldsymbol{h}_v^{(L)}$ is a function of exactly the nodes within distance $L$ of $v$ (the **computation tree** of depth $L$ rooted at $v$). Information cannot travel faster than one hop per layer. Two consequences follow immediately. A shallow GNN is blind to anything farther than $L$ hops. And to let two distant nodes interact you need depth at least equal to their distance, which, as we will see, is exactly where trouble starts.

The first demo lets you watch this happen. Put a one-hot signal at node 0 and press Step: after $k$ layers the signal has reached exactly the nodes within $k$ hops, and the numbers show how each aggregator rescales it.

{{demo: week14-message-passing | Message passing step by step on an editable graph | 470}}

### Why this respects permutation symmetry

:::math Proposition: message passing layers are permutation equivariant
Suppose AGGREGATE is permutation invariant (it depends on the multiset of messages only) and MESSAGE and UPDATE are applied identically at every node (shared weights). Let $\pi$ be a relabelling of the nodes. Then the output at node $\pi(v)$ of the relabelled graph equals the output at node $v$ of the original graph.

*Proof sketch.* Induct on the layer. At layer $0$ the states are the (relabelled) features. Assume layer $\ell-1$ is equivariant. The neighbours of $\pi(v)$ in the relabelled graph are exactly $\pi(u)$ for $u\in\mathcal N(v)$, with the same edge features, and by hypothesis their states equal the original states $\boldsymbol{h}_u^{(\ell-1)}$. The multiset of messages arriving at $\pi(v)$ is therefore the same multiset that arrived at $v$, so the (order-independent) aggregate is the same, and so is the update. $\square$
:::

The NumPy check below verifies this on a six-node graph, and shows what goes wrong if you mix node positions with a generic matrix.

```python:mp_equivariance.py
import numpy as np
rng = np.random.default_rng(0)
n, d_in, d_out = 6, 4, 3
A = np.zeros((n, n))
for u, v in [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5), (2, 3)]:
    A[u, v] = A[v, u] = 1.0
X = rng.normal(size=(n, d_in))
W_self, W_nb = rng.normal(size=(d_in, d_out)), rng.normal(size=(d_in, d_out))

def mp_layer(A, X):
    """h_v' = relu(W_self h_v + W_nb * sum_{u in N(v)} h_u), all nodes at once."""
    return np.maximum(0, X @ W_self + (A @ X) @ W_nb)

H = mp_layer(A, X)
print("output shape:", H.shape)

# permutation equivariance: relabel the nodes with permutation matrix P
perm = rng.permutation(n)
P = np.eye(n)[perm]                    # (P X)[i] = X[perm[i]]
H_perm = mp_layer(P @ A @ P.T, P @ X)
print("equivariant:", np.allclose(H_perm, P @ H))

# a readout (sum over nodes) is permutation INVARIANT
print("invariant readout:", np.allclose(H_perm.sum(0), H.sum(0)))

# a layer that mixes node positions is NOT equivariant
W_bad = rng.normal(size=(n, n))
f_bad = lambda X: W_bad @ X
print("position-mixing MLP equivariant:", np.allclose(f_bad(P @ X), P @ f_bad(X)))
```

```text
output shape: (6, 3)
equivariant: True
invariant readout: True
position-mixing MLP equivariant: False
```

:::note Edge-wise implementation
Real libraries never build the dense $n\times n$ matrix. They store the graph as an **edge index** of shape $2\times|E|$ (source and target node of each directed edge), compute one message per edge by *gathering* source features, and *scatter-add* the messages into target nodes. This costs $O(|E|\,d)$ time and memory instead of $O(n^2 d)$. An undirected edge is stored as two directed edges. PyTorch Geometric's `MessagePassing` base class implements exactly this gather, message, scatter-aggregate, update pipeline.
:::

## Graph convolutional networks (GCN) {#gcn}

The **graph convolutional network** of Kipf and Welling (2017, ICLR, "Semi-supervised classification with graph convolutional networks") is the simplest effective message-passing layer. In matrix form, one layer is

$$
\mathbf H^{(\ell)}=\sigma\!\Big(\hat{\mathbf A}\,\mathbf H^{(\ell-1)}\,\mathbf W^{(\ell)}\Big),\qquad
\hat{\mathbf A}=\tilde{\mathbf D}^{-1/2}\tilde{\mathbf A}\,\tilde{\mathbf D}^{-1/2},\quad \tilde{\mathbf A}=\mathbf A+\mathbf I,\quad \tilde D_{ii}=\sum_j\tilde A_{ij}.
$$

Node by node, with $\tilde d_v=d_v+1$,

$$
\boldsymbol{h}_v^{(\ell)}=\sigma\!\Big(\sum_{u\in\mathcal N(v)\cup\{v\}}\frac{1}{\sqrt{\tilde d_u\,\tilde d_v}}\;\mathbf W^{(\ell)\top}\boldsymbol{h}_u^{(\ell-1)}\Big).
$$

So GCN aggregates a weighted sum over the neighbours *and the node itself*, with weights $1/\sqrt{\tilde d_u\tilde d_v}$ that depend only on the degrees, and then applies one shared linear map and a nonlinearity. Three design choices need justification: why the self-loop, why this particular degree normalization, and where the formula comes from.

### Intuition for the normalization

Plain sum aggregation $\mathbf A\mathbf H$ has a scale problem: a node with 1000 neighbours receives a vector 1000 times larger than a node with 1 neighbour, so feature magnitude would track degree and training would be unstable. Mean aggregation $\mathbf D^{-1}\mathbf A$ fixes the scale (every row sums to one) but is asymmetric and treats a hub's message as important as a leaf's. The symmetric form $\mathbf D^{-1/2}\mathbf A\mathbf D^{-1/2}$ down-weights messages *from* high-degree nodes and *to* high-degree nodes: the pair $(u,v)$ gets weight $1/\sqrt{d_ud_v}$. It also has a clean spectral meaning, derived next. The **self-loop** ($+\mathbf I$) lets a node retain its own signal. Without it a node's new state would be built only from its neighbours and its own features would be lost after a single layer.

### Spectral motivation

:::math From spectral graph convolution to the GCN layer
**Step 1: graph Fourier transform.** Take the eigendecomposition $\mathbf L_{\mathrm{sym}}=\mathbf U\boldsymbol\Lambda\mathbf U^\top$. The columns of $\mathbf U$ play the role of Fourier modes. The graph Fourier transform of a signal $\boldsymbol{x}\in\R^n$ is $\mathbf U^\top\boldsymbol{x}$.

**Step 2: spectral filtering.** On a regular grid, convolution is pointwise multiplication in the Fourier domain. By analogy, filter a signal by scaling each frequency component with a learned function $g_\theta$ of the eigenvalue:
$$
\boldsymbol{y}=\mathbf U\,g_\theta(\boldsymbol\Lambda)\,\mathbf U^\top\boldsymbol{x}.
$$
Computing this needs the full eigendecomposition ($O(n^3)$), produces filters that are not localized in space, and has $n$ free parameters.

**Step 3: polynomial filters.** Parametrize $g_\theta$ as a degree-$K$ polynomial in $\boldsymbol\Lambda$ (Defferrard et al. 2016 use Chebyshev polynomials for numerical stability). Because $\mathbf U\boldsymbol\Lambda^k\mathbf U^\top=\mathbf L_{\mathrm{sym}}^k$,
$$
\mathbf U\Big(\sum_{k=0}^K\theta_k\boldsymbol\Lambda^k\Big)\mathbf U^\top\boldsymbol{x}=\sum_{k=0}^K\theta_k\,\mathbf L_{\mathrm{sym}}^k\boldsymbol{x}.
$$
No eigendecomposition is needed, and since $(\mathbf L^k)_{ij}=0$ whenever the distance between $i$ and $j$ exceeds $k$, the filter is **$K$-localized**: it only mixes nodes within $K$ hops.

**Step 4: first order, one parameter.** Kipf and Welling take $K=1$, approximate the largest eigenvalue $\lambda_{\max}\approx2$ and tie the two coefficients, $\theta=\theta_0=-\theta_1$. Using $\mathbf L_{\mathrm{sym}}-\mathbf I=-\mathbf D^{-1/2}\mathbf A\mathbf D^{-1/2}$ this yields
$$
\boldsymbol{y}\approx\theta\big(\mathbf I+\mathbf D^{-1/2}\mathbf A\mathbf D^{-1/2}\big)\boldsymbol{x}.
$$

**Step 5: renormalization trick.** The operator $\mathbf I+\mathbf D^{-1/2}\mathbf A\mathbf D^{-1/2}$ has eigenvalues in $[0,2]$. Stacking many layers repeatedly multiplies by something whose spectrum reaches $2$, which makes activations explode or vanish. Replace it by
$$
\mathbf I+\mathbf D^{-1/2}\mathbf A\mathbf D^{-1/2}\ \longrightarrow\ \tilde{\mathbf D}^{-1/2}(\mathbf A+\mathbf I)\tilde{\mathbf D}^{-1/2}=\hat{\mathbf A},
$$
whose eigenvalues lie in $(-1,1]$, with the largest equal to $1$. Generalizing from a signal $\boldsymbol{x}$ to a feature matrix $\mathbf H$ with $d_{\mathrm{in}}$ channels and $d_{\mathrm{out}}$ filters ($\theta$ becomes a matrix $\mathbf W$) gives the GCN layer $\hat{\mathbf A}\mathbf H\mathbf W$.
:::

The spectral picture explains what GCN *does*. The operator $\hat{\mathbf A}$ acts on a graph Laplacian eigenvector with eigenvalue $\lambda$ roughly as multiplication by $1-\lambda$ (exactly so for the version without self-loops, $\mathbf D^{-1/2}\mathbf A\mathbf D^{-1/2}=\mathbf I-\mathbf L_{\mathrm{sym}}$). A smooth signal ($\lambda\approx0$) passes unchanged, a signal with $\lambda\approx1$ is annihilated, and a very oscillatory signal ($\lambda\approx2$) is passed with its sign *flipped*. GCN is a **low-pass filter**: it smooths node features over the graph. This is exactly what you want when neighbouring nodes tend to share labels (**homophily**), as in citation networks, and exactly wrong when they do not (heterophily), a limitation we return to in the pitfalls.

```chart
{
 "type": "line",
 "caption": "Frequency response g(lambda) = (1 - lambda)^K of K stacked propagation steps in terms of the eigenvalue lambda of the normalized graph Laplacian (lambda in [0, 2]). Small lambda (smooth signals) is kept, lambda near 1 is killed, and for odd K the highest frequencies near 2 are flipped in sign. Computed analytically.",
 "height": 320,
 "data": {
  "labels": [
   0.0,
   0.05,
   0.1,
   0.15,
   0.2,
   0.25,
   0.3,
   0.35,
   0.4,
   0.45,
   0.5,
   0.55,
   0.6,
   0.65,
   0.7,
   0.75,
   0.8,
   0.85,
   0.9,
   0.95,
   1.0,
   1.05,
   1.1,
   1.15,
   1.2,
   1.25,
   1.3,
   1.35,
   1.4,
   1.45,
   1.5,
   1.55,
   1.6,
   1.65,
   1.7,
   1.75,
   1.8,
   1.85,
   1.9,
   1.95,
   2.0
  ],
  "datasets": [
   {
    "label": "K = 1",
    "data": [
     1.0,
     0.95,
     0.9,
     0.85,
     0.8,
     0.75,
     0.7,
     0.65,
     0.6,
     0.55,
     0.5,
     0.45,
     0.4,
     0.35,
     0.3,
     0.25,
     0.2,
     0.15,
     0.1,
     0.05,
     0.0,
     -0.05,
     -0.1,
     -0.15,
     -0.2,
     -0.25,
     -0.3,
     -0.35,
     -0.4,
     -0.45,
     -0.5,
     -0.55,
     -0.6,
     -0.65,
     -0.7,
     -0.75,
     -0.8,
     -0.85,
     -0.9,
     -0.95,
     -1.0
    ],
    "pointRadius": 0,
    "borderColor": "var(--c-blue)"
   },
   {
    "label": "K = 2",
    "data": [
     1.0,
     0.9025,
     0.81,
     0.7225,
     0.64,
     0.5625,
     0.49,
     0.4225,
     0.36,
     0.3025,
     0.25,
     0.2025,
     0.16,
     0.1225,
     0.09,
     0.0625,
     0.04,
     0.0225,
     0.01,
     0.0025,
     0.0,
     0.0025,
     0.01,
     0.0225,
     0.04,
     0.0625,
     0.09,
     0.1225,
     0.16,
     0.2025,
     0.25,
     0.3025,
     0.36,
     0.4225,
     0.49,
     0.5625,
     0.64,
     0.7225,
     0.81,
     0.9025,
     1.0
    ],
    "pointRadius": 0,
    "borderColor": "var(--c-green)"
   },
   {
    "label": "K = 4",
    "data": [
     1.0,
     0.8145,
     0.6561,
     0.522,
     0.4096,
     0.3164,
     0.2401,
     0.1785,
     0.1296,
     0.09151,
     0.0625,
     0.04101,
     0.0256,
     0.01501,
     0.0081,
     0.003906,
     0.0016,
     0.0005062,
     0.0001,
     6.25e-06,
     0.0,
     6.25e-06,
     0.0001,
     0.0005063,
     0.0016,
     0.003906,
     0.0081,
     0.01501,
     0.0256,
     0.04101,
     0.0625,
     0.09151,
     0.1296,
     0.1785,
     0.2401,
     0.3164,
     0.4096,
     0.522,
     0.6561,
     0.8145,
     1.0
    ],
    "pointRadius": 0,
    "borderColor": "var(--c-amber)"
   },
   {
    "label": "K = 8",
    "data": [
     1.0,
     0.6634,
     0.4305,
     0.2725,
     0.1678,
     0.1001,
     0.05765,
     0.03186,
     0.0168,
     0.008373,
     0.003906,
     0.001682,
     0.0006554,
     0.0002252,
     6.561e-05,
     1.526e-05,
     2.56e-06,
     2.563e-07,
     1e-08,
     3.906e-11,
     0.0,
     3.906e-11,
     1e-08,
     2.563e-07,
     2.56e-06,
     1.526e-05,
     6.561e-05,
     0.0002252,
     0.0006554,
     0.001682,
     0.003906,
     0.008373,
     0.0168,
     0.03186,
     0.05765,
     0.1001,
     0.1678,
     0.2725,
     0.4305,
     0.6634,
     1.0
    ],
    "pointRadius": 0,
    "borderColor": "var(--c-red)"
   }
  ]
 },
 "options": {
  "scales": {
   "x": {
    "title": {
     "display": true,
     "text": "Laplacian eigenvalue lambda"
    }
   },
   "y": {
    "title": {
     "display": true,
     "text": "filter response"
    }
   }
  }
 }
}
```

The even-$K$ curves show a subtlety. Without self-loops, $\mathbf D^{-1/2}\mathbf A\mathbf D^{-1/2}$ has eigenvalue $-1$ on bipartite graphs (the alternating signal at $\lambda=2$), and its even powers keep that component at full strength, so the layer is not a pure low-pass filter. Adding the self-loop shifts the spectrum of $\hat{\mathbf A}$ away from $-1$ (its most negative eigenvalues are pulled towards zero), which is one more reason the renormalized operator behaves as a low-pass filter in practice and why it is stable to stack.

### GCN in code

The NumPy version is three lines. Note that the rows of $\hat{\mathbf A}$ do **not** sum to one (it is symmetric-normalized, not a random-walk matrix), but its largest eigenvalue is exactly one.

```python:gcn_numpy.py
import numpy as np

def gcn_norm(A):
    """A_hat = D~^{-1/2} (A + I) D~^{-1/2} (Kipf & Welling renormalization trick)."""
    A_tilde = A + np.eye(len(A))
    d = A_tilde.sum(1)
    return A_tilde / np.sqrt(np.outer(d, d))

def gcn_layer(A_hat, H, W):
    return np.maximum(0, A_hat @ H @ W)          # ReLU(A_hat H W)

rng = np.random.default_rng(1)
A = np.zeros((6, 6))
for u, v in [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5), (2, 3)]:
    A[u, v] = A[v, u] = 1.0
A_hat = gcn_norm(A)
print("row sums of A_hat (NOT 1 in general):", np.round(A_hat.sum(1), 3))
print("largest eigenvalue of A_hat:", np.round(np.linalg.eigvalsh(A_hat).max(), 6))

H0 = rng.normal(size=(6, 4)); W1 = rng.normal(size=(4, 8)) * 0.5; W2 = rng.normal(size=(8, 2)) * 0.5
H1 = gcn_layer(A_hat, H0, W1)
logits = A_hat @ H1 @ W2                          # second layer without ReLU -> logits
print(H1.shape, logits.shape)
```

```text
row sums of A_hat (NOT 1 in general): [0.955 0.955 1.077 1.077 0.955 0.955]
largest eigenvalue of A_hat: 1.0
(6, 8) (6, 2)
```

In PyTorch you would write the same layer on an **edge list** so that memory scales with $|E|$. The following module implements GCN from scratch with `index_add_` as the scatter operation, which is a good way to see what PyG's `GCNConv` does internally.

```python:gcn_torch.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class GCNLayer(nn.Module):
    """H' = A_hat H W on an edge list. edge_index: LongTensor [2, E] with (source, target)."""
    def __init__(self, d_in, d_out):
        super().__init__()
        self.lin = nn.Linear(d_in, d_out, bias=False)
        self.bias = nn.Parameter(torch.zeros(d_out))

    def forward(self, x, edge_index):
        n = x.size(0)
        # add self loops (i -> i for every node)
        loops = torch.arange(n, device=x.device)
        src = torch.cat([edge_index[0], loops])
        dst = torch.cat([edge_index[1], loops])
        # degree of the target node in A + I (counts incoming edges incl. the loop)
        deg = torch.zeros(n, device=x.device).index_add_(0, dst, torch.ones_like(dst, dtype=torch.float))
        norm = deg[src].rsqrt() * deg[dst].rsqrt()          # 1 / sqrt(d_u d_v) per edge
        h = self.lin(x)                                      # [n, d_out], transform first (cheaper if d_out < d_in)
        out = torch.zeros_like(h).index_add_(0, dst, h[src] * norm.unsqueeze(1))   # scatter-add messages
        return out + self.bias

class GCN(nn.Module):
    def __init__(self, d_in, d_hidden, n_classes, p=0.5):
        super().__init__()
        self.l1, self.l2, self.p = GCNLayer(d_in, d_hidden), GCNLayer(d_hidden, n_classes), p

    def forward(self, x, edge_index):
        x = F.dropout(x, self.p, self.training)
        x = F.relu(self.l1(x, edge_index))
        x = F.dropout(x, self.p, self.training)
        return self.l2(x, edge_index)

# shape check on a random graph: 100 nodes, 500 random directed edges, 16 features, 3 classes
x = torch.randn(100, 16)
edge_index = torch.randint(0, 100, (2, 500))
print(GCN(16, 32, 3)(x, edge_index).shape)   # torch.Size([100, 3])
```

Two details worth noticing. The weights are applied *before* propagation (`self.lin(x)` then scatter) because $\hat{\mathbf A}(\mathbf H\mathbf W)=(\hat{\mathbf A}\mathbf H)\mathbf W$ and it is cheaper to propagate the narrower matrix. And because the graph is fixed and the features of **all** nodes (including unlabeled ones) go through the network, the loss on the few labeled nodes backpropagates into the features of their unlabeled neighbours; this is how a labeled node "teaches" its neighbourhood.

:::note Transductive versus inductive
GCN as trained by Kipf and Welling is **transductive**: the whole graph, including the test nodes (but not their labels), is present during training, and $\hat{\mathbf A}$ is computed once over it. A model is **inductive** if it can be applied to nodes or graphs not seen in training. The GCN *layer* has no dependence on $n$, so in principle it is inductive, but the full-batch recipe is not. GraphSAGE was designed around the inductive and large-graph use case.
:::

## GraphSAGE: sampling and separate self weights {#sage}

GraphSAGE (Hamilton, Ying and Leskovec, 2017, NeurIPS, "Inductive representation learning on large graphs") addresses two practical limitations. Full-batch GCN needs the whole adjacency matrix in memory, and a hub node with $10^6$ neighbours makes the cost of a single node unbounded. The remedies are **neighbour sampling** and an explicit separation between a node and its neighbourhood. One layer is

$$
\boldsymbol{h}_v^{(\ell)}=\sigma\Big(\mathbf W_1\,\boldsymbol{h}_v^{(\ell-1)}+\mathbf W_2\,\operatorname{AGG}\big(\{\boldsymbol{h}_u^{(\ell-1)}:u\in\mathcal S(v)\}\big)\Big),
$$

(the original paper writes this as a concatenation $[\boldsymbol{h}_v\,\|\,\operatorname{AGG}]$ followed by one matrix, which is equivalent to two matrices summed), where $\mathcal S(v)\subseteq\mathcal N(v)$ is a **fixed-size random sample** of neighbours and AGG is a mean, a max-pool over per-neighbour MLP outputs, or an LSTM run over a random permutation of the sample (the LSTM is not permutation invariant, which the authors handle by random ordering). The embeddings are typically L2-normalized afterwards.

Sampling bounds the computation. If each layer samples $s$ neighbours, the computation tree of one node has at most $s^L$ leaves regardless of the degrees, so mini-batches of target nodes have a predictable memory cost. At test time one can either sample again or use the full neighbourhood. Because the layer never refers to a global matrix, a trained model embeds nodes of a graph it has never seen (a new user, a new paper), which is the sense of **inductive**. Keeping $\mathbf W_1$ separate from $\mathbf W_2$ is also a real modeling improvement: a node's own features and its neighbourhood summary are different kinds of information, and the model can weight them differently (for example, when neighbours are noisy). GCN forces them through the same weights.

PinSage (Ying et al., 2018, KDD) scaled this recipe to a graph of three billion nodes at Pinterest, replacing uniform sampling by importance sampling from random walks.

## Graph attention networks (GAT) {#gat}

GCN weights every neighbour by a *fixed* function of the degrees. In many graphs some neighbours are more relevant than others, and the relevance depends on the features. **Graph attention networks** (Velickovic, Cucurull, Casanova, Romero, Lio and Bengio, 2018, ICLR) let the model learn the weights. This is the attention of [Week 9](week-09.html) restricted to the graph neighbourhood: instead of attending over all tokens, node $v$ attends over $\mathcal N(v)\cup\{v\}$.

First apply a shared linear map, $\boldsymbol{z}_u=\mathbf W\boldsymbol{h}_u$. For each edge $(u\to v)$ compute an unnormalized score with a small learned vector $\boldsymbol{a}\in\R^{2d'}$ and a LeakyReLU,

$$
e_{vu}=\operatorname{LeakyReLU}\big(\boldsymbol{a}^\top[\boldsymbol{z}_v\,\|\,\boldsymbol{z}_u]\big),\qquad
\alpha_{vu}=\frac{\exp(e_{vu})}{\sum_{k\in\mathcal N(v)\cup\{v\}}\exp(e_{vk})},\qquad
\boldsymbol{h}_v'=\sigma\Big(\sum_{u\in\mathcal N(v)\cup\{v\}}\alpha_{vu}\,\boldsymbol{z}_u\Big).
$$

The softmax runs over the **neighbours of $v$** only, so the weights sum to one. As in Transformers, **multi-head** attention runs $K$ independent copies and concatenates (hidden layers) or averages (final layer). Two remarks that are often misunderstood:

- **Static attention.** Because the score is $\boldsymbol{a}_1^\top\boldsymbol{z}_v+\boldsymbol{a}_2^\top\boldsymbol{z}_u$ (the concatenation splits the dot product), the *ranking* of neighbours $u$ is the same for every query node $v$, since only the $\boldsymbol{a}_2^\top\boldsymbol{z}_u$ term differs between neighbours. This was pointed out by Brody, Alon and Yahav (2022, "How attentive are graph attention networks?"), who proposed **GATv2**, $e_{vu}=\boldsymbol{a}^\top\operatorname{LeakyReLU}(\mathbf W[\boldsymbol{h}_v\|\boldsymbol{h}_u])$, which makes the ranking depend on the query ("dynamic attention").
- **Attention is still a convex combination.** Weights are positive and sum to one, so a GAT neighbourhood summary is a *weighted mean*. It cannot count neighbours, which has consequences for expressive power below.

```python:gat_torch.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseGATLayer(nn.Module):
    """Multi-head GAT on a dense adjacency (clear but O(n^2); fine for n up to a few thousand)."""
    def __init__(self, d_in, d_out, heads=4, concat=True, neg_slope=0.2):
        super().__init__()
        self.h, self.d, self.concat = heads, d_out, concat
        self.W = nn.Linear(d_in, heads * d_out, bias=False)
        self.a_src = nn.Parameter(torch.empty(heads, d_out))   # acts on the neighbour u
        self.a_dst = nn.Parameter(torch.empty(heads, d_out))   # acts on the centre node v
        nn.init.xavier_uniform_(self.a_src); nn.init.xavier_uniform_(self.a_dst)
        self.neg_slope = neg_slope

    def forward(self, x, adj):
        # adj: [n, n] bool, adj[v, u] = True if u is a neighbour of v (include self loops!)
        n = x.size(0)
        z = self.W(x).view(n, self.h, self.d)                   # [n, H, d]
        s_dst = (z * self.a_dst).sum(-1)                        # [n, H]  score part for centre v
        s_src = (z * self.a_src).sum(-1)                        # [n, H]  score part for neighbour u
        e = F.leaky_relu(s_dst.unsqueeze(1) + s_src.unsqueeze(0), self.neg_slope)   # [n(v), n(u), H]
        e = e.masked_fill(~adj.unsqueeze(-1), float('-inf'))    # only real edges compete in the softmax
        alpha = torch.softmax(e, dim=1)                         # normalize over u for each v and head
        out = torch.einsum('vuh,uhd->vhd', alpha, z)            # weighted sum of neighbour features
        return out.reshape(n, -1) if self.concat else out.mean(1)

n = 5
adj = torch.eye(n, dtype=torch.bool); adj[0, 1] = adj[1, 0] = adj[1, 2] = adj[2, 1] = True
layer = DenseGATLayer(8, 16, heads=4)
print(layer(torch.randn(n, 8), adj).shape)                      # torch.Size([5, 64])
```

Forgetting the self-loops in `adj` leaves isolated nodes with a row of all $-\infty$ in the softmax, which produces `nan`s. This is one of the most common ways a hand-written GAT crashes.

## Graph isomorphism networks (GIN) {#gin}

GCN, GraphSAGE-mean and GAT all aggregate with a *mean-like* operator. Xu, Hu, Leskovec and Jegelka (2019, ICLR, "How powerful are graph neural networks?") asked a sharper question: what is the most discriminative message-passing layer one can build? Their answer, the **graph isomorphism network (GIN)**, is

$$
\boldsymbol{h}_v^{(\ell)}=\operatorname{MLP}^{(\ell)}\Big((1+\epsilon^{(\ell)})\,\boldsymbol{h}_v^{(\ell-1)}+\sum_{u\in\mathcal N(v)}\boldsymbol{h}_u^{(\ell-1)}\Big),
$$

with a learnable or fixed scalar $\epsilon$ (often $0$). Two choices distinguish it: the aggregator is a plain **sum** and the transformation is a full **MLP** (at least one hidden layer) instead of one linear map. The reason is developed in the next section: sum over a multiset can be made injective, mean and max cannot, and an MLP is needed to approximate the injective function that realizes it.

{{figure: week14-gnn-variants | The four layers side by side. All of them are permutation-equivariant message-passing layers; they differ in how neighbours are weighted and combined.}}

| Layer | Aggregation weights | Self handled by | Sees degree? | Typical use |
| --- | --- | --- | --- | --- |
| GCN | fixed, $1/\sqrt{\tilde d_u\tilde d_v}$ | self-loop | only through weights | homophilous node classification, small graphs |
| GraphSAGE | mean/max/LSTM over a sample | separate $\mathbf W_1$ | no (mean) | huge graphs, inductive, recommenders |
| GAT | learned softmax $\alpha_{vu}$ | self-loop in the softmax | no (convex combination) | noisy neighbourhoods, edges of varying relevance |
| GIN | sum (weights all $1$) | $(1+\epsilon)$ term | yes (sum counts) | graph classification, chemistry |

Sum aggregation is not free. It is scale-sensitive (large-degree nodes produce large activations), so in practice GIN is used with batch normalization inside the MLP, and on graphs with very heterogeneous degrees a mean can generalize better even though it is theoretically weaker. We now make "theoretically weaker" precise.

## Expressive power and the Weisfeiler-Lehman test {#expressivity}

Which graphs can a GNN tell apart? This matters because if two non-isomorphic graphs (say, two molecules with different properties) always receive the same embedding, no amount of training can separate them. Expressive power is about the limits of the *architecture*, not of optimization.

### The 1-WL colour refinement algorithm

The **1-dimensional Weisfeiler-Lehman test** (Weisfeiler and Leman, 1968), also called colour refinement or naive vertex classification, is a classical heuristic for graph isomorphism. It works as follows.

1. Give every node an initial colour $c_v^{(0)}$ (a constant if the graph has no node features, otherwise a hash of the features).
2. Repeat: $c_v^{(t+1)}=\operatorname{HASH}\big(c_v^{(t)},\ \{\!\!\{c_u^{(t)}:u\in\mathcal N(v)\}\!\!\}\big)$, where HASH is any injective map from (colour, multiset of colours) to new colours.
3. Stop when the partition of the nodes into colour classes no longer changes (at most $n$ rounds).
4. Compare the histograms of colours of the two graphs. If they differ at some round, the graphs are **certainly non-isomorphic**. If they agree forever, the test is inconclusive.

The test is one-sided: "different" is a proof; "same" is not. The message-passing equation above is *exactly* this procedure with neural networks in place of the hash. That is the whole connection, and Xu et al. and, independently, Morris et al. (2019, AAAI, "Weisfeiler and Leman go neural: higher-order graph neural networks") turned it into theorems.

{{figure: week14-wl | The classical failure of 1-WL: a 6-cycle and two disjoint triangles. Every node in both graphs has degree 2, so refinement never splits the single colour class.}}

The next demo runs colour refinement on two graphs simultaneously. Start with the hexagon versus two triangles, then try the pairs that 1-WL separates after one or two rounds, and observe the difference between "histograms differ" and "stable partition with equal histograms".

{{demo: week14-wl | 1-WL colour refinement on pairs of graphs | 480}}

```python:wl.py
from collections import Counter

def wl_colors(adj, rounds=None, colors=None):
    """1-WL colour refinement. adj: list of neighbour lists. Returns the list of colour lists per round.
    Colours are canonical signature tuples (so they are comparable across graphs)."""
    n = len(adj)
    colors = colors or [0] * n
    history = [colors]
    for _ in range(rounds if rounds is not None else n):
        colors = [hash((colors[v], tuple(sorted(colors[u] for u in adj[v])))) for v in range(n)]
        history.append(colors)
    return history

def adj_from_edges(n, edges):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v); adj[v].append(u)
    return adj

hexagon = adj_from_edges(6, [(i, (i + 1) % 6) for i in range(6)])
triangles = adj_from_edges(6, [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)])
path = adj_from_edges(5, [(0, 1), (1, 2), (2, 3), (3, 4)])
star_tail = adj_from_edges(5, [(0, 1), (0, 2), (0, 3), (3, 4)])

def wl_distinguishes(g1, g2, rounds=5):
    h1, h2 = wl_colors(g1, rounds), wl_colors(g2, rounds)
    return any(Counter(a) != Counter(b) for a, b in zip(h1, h2))

print("C6 vs 2xC3 distinguished:", wl_distinguishes(hexagon, triangles))
print("P5 vs tree distinguished:", wl_distinguishes(path, star_tail))
```

```text
C6 vs 2xC3 distinguished: False
P5 vs tree distinguished: True
```

### Precise statements

:::definition Multiset and injectivity
A function $f$ on multisets of node states is **injective** if two different multisets always give different outputs. A message-passing layer is **at least as discriminative as 1-WL** at a node if, whenever its two input states and neighbour multisets differ, its outputs differ.
:::

**Theorem 1 (upper bound; Xu et al. 2019, Morris et al. 2019).** For any message-passing GNN with arbitrary (even learned) MESSAGE, AGGREGATE and UPDATE, if 1-WL assigns the same colour to two nodes after $t$ rounds, then after $t$ layers the GNN assigns them the same embedding. Consequently, if 1-WL does not distinguish two graphs, no such GNN with a permutation-invariant readout does either. *Reason:* by induction, embeddings are a function of the 1-WL colour. If the colours agree, the (state, multiset-of-neighbour-states) pairs agree, so the layer output agrees.

**Theorem 2 (achievability; Xu et al. 2019).** If the layers and the readout are *injective* on multisets of their inputs (for a countable feature space), the GNN distinguishes exactly the pairs of graphs that 1-WL distinguishes. GIN attains this: for a countable input space, Xu et al. show there exist $f$ and $\epsilon$ such that $h(c,X)=(1+\epsilon)f(c)+\sum_{x\in X}f(x)$ is an injective function of the pair $(c,X)$ of a state $c$ and a **finite multiset** $X$, and an MLP can approximate the composition with the next transformation by the universal approximation theorem. Together: **GIN is exactly as powerful as 1-WL, and no message-passing GNN is more powerful.**

### Why sum, and what mean and max lose

On multisets over a finite alphabet, sums distinguish more than the alternatives. Consider neighbour multisets of one-dimensional states:

| Multiset | sum | mean | max |
| --- | --- | --- | --- |
| $\{1,1,2,2\}$ | 6 | 1.5 | 2 |
| $\{1,2\}$ | 3 | 1.5 | 2 |
| $\{1,1,1,2\}$ | 5 | 1.25 | 2 |
| $\{2,2\}$ | 4 | 2 | 2 |

Mean confuses $\{1,1,2,2\}$ with $\{1,2\}$: it keeps the **proportions** (distribution) but loses multiplicities. Max confuses all four: it keeps only the **set** of distinct elements and discards both. Sum keeps the full multiset (given an injective embedding of each element, for instance one-hot encodings, the sum is the vector of counts). The hierarchy is therefore *sum > mean > max* in discriminative power, and the consequence is concrete: with identical node features, a mean-aggregation GNN cannot tell a node with two neighbours from a node with four, and a GAT (a weighted mean) inherits that limitation. A 6-node ring and a 12-node ring are indistinguishable to GCN-mean on constant features, while the **degree** is visible to GIN after a single layer.

:::warning Max and mean are not useless
The comparison is about *distinguishing* arbitrary multisets. In real data you often want the opposite: a statistic that is robust to the number of neighbours (mean), or that detects the presence of a rare feature (max). Corso et al. (2020, "Principal neighbourhood aggregation") combine several aggregators (mean, max, min, std) with degree-based scalers precisely because no single one is best.
:::

### What 1-WL cannot see

The hexagon and the two triangles in the figure are the canonical counterexample: both are 2-regular, so refinement starting from a constant colouring produces identical colourings forever. Any message-passing GNN therefore maps them to the same embedding, yet one is connected and the other is not, and one contains triangles and the other does not. This has practical consequences. Standard message-passing GNNs cannot detect triangles or count cycles in general (the two-triangle graph has them, the hexagon does not, and the embeddings are identical), and they cannot decide whether a graph is connected. For chemistry this matters because rings carry meaning: a well-known example is the pair decalin and bicyclopentyl, whose carbon skeletons are 1-WL-equivalent although one has two fused six-membered rings and the other two five-membered rings joined by a bond.

Ways to go beyond 1-WL, each with a cost:

- **Higher-order WL.** The $k$-WL test colours $k$-tuples of nodes, and $k$-GNNs (Morris et al. 2019) mimic it. Cost grows like $n^k$ in memory, so $k=2$ or $3$ is the practical limit. (There are two conventions in the literature, "folklore" and "oblivious" $k$-WL, which differ by one level of power, so check which one a paper uses.)
- **Structural and positional features.** Add counts of triangles or cycles, random-walk return probabilities, or Laplacian eigenvectors to the node features. This breaks the symmetry between nodes that 1-WL treats alike, and it is cheap, but the features must be chosen well.
- **Random node features** (Sato, Yamada and Kashima, 2021; Abboud et al., 2021): give each node a random label. This makes GNNs more expressive in a probabilistic sense, at the cost of variance and weaker generalization.
- **Subgraph GNNs** that run message passing on many (rooted or node-deleted) subgraphs, and equivariant models with tensor features of higher order.

Expressivity is also not everything. 1-WL equivalence is a worst-case statement about *distinguishability*; real datasets rarely contain such adversarial pairs, and on many benchmarks a "weaker" GCN beats a "stronger" model because of optimization, generalization or over-smoothing. Treat WL power as a map of what is *impossible*, not as a predictor of accuracy.

## Going deep: over-smoothing and over-squashing {#depth}

CNNs got dramatically better with depth (ResNet had 152 layers; [Week 6](week-06.html)). Message-passing GNNs usually do not: the best Cora models have two layers, and accuracy typically *drops* beyond three or four. Two distinct phenomena explain this, and it is worth keeping them apart because they have different causes and different cures.

### Over-smoothing

**Over-smoothing** is the tendency of node representations to become indistinguishable as depth grows. Intuition: every layer replaces a node's feature by an average of its neighbourhood. Averaging repeatedly is diffusion, and diffusion on a connected graph converges to a constant. After enough layers, a node in one community and a node in another hold nearly the same vector, and a classifier on top cannot separate them. Li, Han and Wu (2018, AAAI, "Deeper insights into graph convolutional networks for semi-supervised learning") observed that the GCN layer is a form of **Laplacian smoothing** and that repeated smoothing makes features of different classes converge.

:::math Why repeated propagation converges
Take the linear part of $K$ GCN layers, $\mathbf X^{(K)}=\hat{\mathbf A}^K\mathbf X$. The matrix $\hat{\mathbf A}$ is symmetric with eigenpairs $(\mu_i,\boldsymbol{u}_i)$; for a connected graph with self-loops $1=\mu_1>\mu_2\ge\dots\ge\mu_n>-1$, and the top eigenvector is $\boldsymbol{u}_1\propto\tilde{\mathbf D}^{1/2}\boldsymbol{1}$. Writing $c_i=\boldsymbol{u}_i^\top\mathbf X$ (a row vector of graph-Fourier coefficients),
$$
\hat{\mathbf A}^K\mathbf X=\sum_{i=1}^n\mu_i^K\,\boldsymbol{u}_i\,\boldsymbol{c}_i\ \xrightarrow[K\to\infty]{}\ \boldsymbol{u}_1\boldsymbol{c}_1 .
$$
So every node ends up with the *same direction* $\boldsymbol{c}_1$, scaled by $\sqrt{\tilde d_v}$: only the degree survives. The rate is governed by $\rho=\max_{i\ge2}|\mu_i|<1$ (the spectral gap).

To quantify it, define the degree-normalized **Dirichlet energy** $\mathcal E(\mathbf X)=\tfrac12\sum_{i,j}A_{ij}\big\|\tfrac{\boldsymbol{x}_i}{\sqrt{\tilde d_i}}-\tfrac{\boldsymbol{x}_j}{\sqrt{\tilde d_j}}\big\|^2=\operatorname{tr}\big(\mathbf X^\top(\mathbf I-\hat{\mathbf A})\mathbf X\big)$. Then
$$
\mathcal E\big(\hat{\mathbf A}^K\mathbf X\big)=\sum_{i\ge1}(1-\mu_i)\,\mu_i^{2K}\,\|\boldsymbol{c}_i\|^2\ \le\ \rho^{2K}\,\mathcal E(\mathbf X),
$$
because the $i=1$ term has $1-\mu_1=0$, and for $i\ge2$ we have $\mu_i^{2K}\le\rho^{2K}$. Smoothness measured by energy therefore vanishes **geometrically** in the depth.
:::

Oono and Suzuki (2020, ICLR) extended this to GCNs with ReLU and weight matrices: under conditions on the singular values of the weights, the features approach an invariant subspace exponentially fast, so deep GCNs lose the information needed for node classification. The following NumPy experiment measures the energy of a propagated feature matrix on a two-community random graph, and checks it against the prediction $\rho^{2K}$.

```python:oversmoothing.py
import numpy as np
rng = np.random.default_rng(0)

# two-community random graph (stochastic block model), 200 nodes
n = 200
c = np.repeat([0, 1], n // 2)
P = np.where(c[:, None] == c[None, :], 0.08, 0.01)          # p_in = 0.08, p_out = 0.01
U = rng.random((n, n))
A = np.triu(U < P, 1); A = (A | A.T).astype(float)           # symmetric, no self loops

# node features: 2 weakly informative dims + 6 pure-noise dims
X0 = np.stack([(2 * c - 1) * 0.5, np.zeros(n)], 1) + rng.normal(size=(n, 2))
X0 = np.concatenate([X0, rng.normal(size=(n, 6))], 1)

A_tilde = A + np.eye(n)
d = A_tilde.sum(1)
S = A_tilde / np.sqrt(np.outer(d, d))                         # GCN propagation matrix
dn = 1 / np.sqrt(d)

def dirichlet_energy(X):
    """sum over edges of |x_i/sqrt(d_i) - x_j/sqrt(d_j)|^2 : degree-normalized roughness."""
    i, j = np.nonzero(np.triu(A, 1))
    return (((X[i] * dn[i, None] - X[j] * dn[j, None]) ** 2).sum())

def trajectory(alpha, K=32):
    X, E = X0.copy(), [dirichlet_energy(X0)]
    for _ in range(K):
        X = (1 - alpha) * S @ X + alpha * X0                  # alpha = 0: plain GCN propagation
        E.append(dirichlet_energy(X))
    return np.array(E) / E[0]

for alpha in (0.0, 0.1, 0.3):
    E = trajectory(alpha)
    print(f"alpha={alpha}:", {k: float(f"{E[k]:.3g}") for k in (1, 2, 4, 8, 16, 32)})

# the analytic prediction: energy decays like rho^(2k), rho = second-largest |eigenvalue| of S
mu = np.sort(np.abs(np.linalg.eigvalsh(S)))[::-1]
print("largest |eigenvalues| of S:", np.round(mu[:3], 4))
```

```text
alpha=0.0: {1: 0.0751, 2: 0.0144, 4: 0.00242, 8: 0.000574, 16: 5.52e-05, 32: 5.19e-07}
alpha=0.1: {1: 0.0746, 2: 0.0343, 4: 0.0173, 8: 0.0145, 16: 0.014, 32: 0.014}
alpha=0.3: {1: 0.136, 2: 0.124, 4: 0.108, 8: 0.107, 16: 0.107, 32: 0.107}
largest |eigenvalues| of S: [1. 0.8643 0.6342]
```

With $\rho=0.864$ the late-time decay should be about $\rho^{2K}$: from $K=16$ to $K=32$ that is a factor $0.864^{32}\approx0.009$, and the measured ratio $5.19\times10^{-7}/5.52\times10^{-5}\approx0.0094$ agrees. The residual variants ($\alpha>0$, which mix the original features back in at each step) stop decaying and settle on a positive floor, because the fixed point of $\mathbf X=(1-\alpha)\hat{\mathbf A}\mathbf X+\alpha\mathbf X_0$ is $\alpha(\mathbf I-(1-\alpha)\hat{\mathbf A})^{-1}\mathbf X_0$, which is not constant.

```chart
{"type":"line","caption":"Normalized Dirichlet energy (relative to layer 0) of node features after k steps of GCN propagation on a 200-node, two-community random graph (stochastic block model: p_in = 0.08, p_out = 0.01), computed with the NumPy code in this section. alpha is the weight of a residual link to the input features. Without a residual, energy decays geometrically towards zero; with alpha > 0 it settles on a floor.","height":340,"data":{"labels":[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32],"datasets":[{"label":"alpha = 0 (plain GCN)","data":[1.0,0.07506,0.01442,0.004811,0.002418,0.001529,0.001068,0.0007766,0.0005736,0.0004264,0.0003179,0.0002373,0.0001772,0.0001323,9.885e-05,7.384e-05,5.516e-05,4.121e-05,3.079e-05,2.3e-05,1.718e-05,1.284e-05,9.59e-06,7.165e-06,5.352e-06,3.999e-06,2.987e-06,2.232e-06,1.667e-06,1.246e-06,9.305e-07,6.952e-07,5.193e-07],"borderColor":"var(--c-red)","pointRadius":0},{"label":"alpha = 0.1","data":[1.0,0.07455,0.03434,0.0193,0.01728,0.01558,0.01507,0.01468,0.01449,0.01435,0.01426,0.01419,0.01414,0.01411,0.01408,0.01406,0.01404,0.01403,0.01402,0.01402,0.01401,0.01401,0.014,0.014,0.014,0.014,0.014,0.01399,0.01399,0.01399,0.01399,0.01399,0.01399],"borderColor":"var(--c-blue)","pointRadius":0},{"label":"alpha = 0.3","data":[1.0,0.1355,0.1239,0.1092,0.1084,0.1073,0.1071,0.107,0.107,0.1069,0.1069,0.1069,0.1069,0.1069,0.1069,0.1069,0.1069,0.1069,0.1069,0.1069,0.1069,0.1069,0.1069,0.1069,0.1069,0.1069,0.1069,0.1069,0.1069,0.1069,0.1069,0.1069,0.1069],"borderColor":"var(--c-green)","pointRadius":0}]},"options":{"scales":{"x":{"title":{"display":true,"text":"propagation steps k"}},"y":{"type":"logarithmic","title":{"display":true,"text":"energy(k) / energy(0)"}}}}}
```

The next demo makes the effect visible: a two-community graph with 2-D features, where you choose the number of layers and the strength of the residual link. Watch the two clouds slide together and the energy curve fall.

{{demo: week14-oversmoothing | Over-smoothing explorer: propagation depth and residual strength | 500}}

:::warning Smoothing is not always bad, and depth is not the only cause
Smoothing is the *purpose* of a GCN layer: it averages out noise within a community. The problem is **too much** of it. Also, deep GNNs fail for reasons beyond smoothing: the vanishing-gradient and optimization problems of any deep network (there is no residual path by default), the entanglement of propagation and transformation (each layer both smooths and applies a weight matrix and nonlinearity), and over-squashing. Separating the propagation depth from the transformation depth, as in the APPNP and SGC models below, often recovers accuracy with *many* propagation steps and *two* weight matrices.
:::

### Over-squashing

**Over-squashing** (Alon and Yahav, 2021, ICLR, "On the bottleneck of graph neural networks and its practical implications") is a different failure: it is an information bottleneck, not a convergence. The $r$-hop neighbourhood of a node can contain a number of nodes that grows exponentially with $r$ (on trees and on expander-like graphs), but the embedding $\boldsymbol{h}_v^{(r)}$ has a fixed width $d$. All the information from those exponentially many nodes must be compressed into $d$ numbers *through the narrow edges on the way*. Long-range dependencies therefore get crushed.

{{figure: week14-oversquashing | Over-squashing: the number of nodes that can influence $v$ grows like $2^r$ in a binary tree, while the embedding $\boldsymbol{h}_v$ has fixed size. Information from far leaves must be funnelled through a few paths.}}

```chart
{"type":"line","caption":"Number of nodes within r hops of a node (cumulative receptive field): 2-D grid (4 neighbours), 3-regular tree and 4-regular tree. Exact formulas: 2r(r+1)+1, 3*2^r - 2, and 1 + 2(3^r - 1). Trees grow exponentially, which is the root of over-squashing.","height":320,"data":{"labels":[0,1,2,3,4,5,6,7,8],"datasets":[{"label":"2-D grid","data":[1,5,13,25,41,61,85,113,145],"borderColor":"var(--c-green)"},{"label":"3-regular tree","data":[1,4,10,22,46,94,190,382,766],"borderColor":"var(--c-blue)"},{"label":"4-regular tree","data":[1,5,17,53,161,485,1457,4373,13121],"borderColor":"var(--c-red)"}]},"options":{"scales":{"x":{"title":{"display":true,"text":"hops r (GNN layers)"}},"y":{"type":"logarithmic","title":{"display":true,"text":"nodes in receptive field"}}}}}
```

For a quantitative handle, Topping, Di Giovanni, Chamberlain, Dong and Bronstein (2022, ICLR, "Understanding over-squashing and bottlenecks on graphs via curvature") bound the sensitivity of a node's final embedding to another node's input through a Jacobian. If the message and update functions are Lipschitz with constants bounded by $\alpha$ and $\beta$, then for nodes $u,v$ at distance $r$,

$$
\left\|\frac{\partial\boldsymbol{h}_v^{(r)}}{\partial\boldsymbol{x}_u}\right\|\ \le\ (\alpha\beta)^r\,\big(\hat{\mathbf A}^r\big)_{vu}\quad(\text{up to the exact normalization used}).
$$

The entry $(\hat{\mathbf A}^r)_{vu}$ is the (normalized) number of length-$r$ walks from $v$ to $u$. When $u$ is reachable only through a **bottleneck** (a bridge edge or a narrow cut) the entry is tiny, so the influence of $u$ on $v$ is provably small however the weights are trained. They linked this to negative **Ollivier-Ricci curvature** of the edges on the path: negatively curved edges are the bridges that squash. The two-triangle graph from the [basics](#basics) section is a miniature of this: the single bridge edge carries *all* information between the halves.

The growth of the receptive field is the reason bottlenecks bite on tree-like graphs and not on grids:

```chart
{"type":"line","caption":"Number of nodes within r hops of a node (cumulative receptive field): 2-D grid (4 neighbours), 3-regular tree and 4-regular tree. Exact formulas: 2r(r+1)+1, 3*2^r - 2, and 1 + 2(3^r - 1). Trees grow exponentially, which is the root of over-squashing.","height":320,"data":{"labels":[0,1,2,3,4,5,6,7,8],"datasets":[{"label":"2-D grid","data":[1,5,13,25,41,61,85,113,145],"borderColor":"var(--c-green)"},{"label":"3-regular tree","data":[1,4,10,22,46,94,190,382,766],"borderColor":"var(--c-blue)"},{"label":"4-regular tree","data":[1,5,17,53,161,485,1457,4373,13121],"borderColor":"var(--c-red)"}]},"options":{"scales":{"x":{"title":{"display":true,"text":"hops r (GNN layers)"}},"y":{"type":"logarithmic","title":{"display":true,"text":"nodes in receptive field"}}}}}
```

The tree curves are straight lines on the log axis: $3\cdot2^r-2$ nodes at radius $r$ for a 3-regular tree. A grid grows only quadratically, which is why long-range image tasks do not suffer over-squashing in the same way. Real molecular and social graphs lie in between, and the *bottlenecks* matter more than the raw counts.

### Remedies for deep and long-range GNNs

The fixes fall into three families, matching the diagnoses.

**Against over-smoothing: keep a path to the input and to the shallow features.**

- **Residual connections** (as in ResNet, [Week 6](week-06.html)): $\mathbf H^{(\ell)}=\mathbf H^{(\ell-1)}+\sigma(\hat{\mathbf A}\mathbf H^{(\ell-1)}\mathbf W)$. DeepGCNs (Li et al., 2019, ICCV) combined residual and dense connections with dilated aggregation to train GNNs with over 50 layers.
- **Initial residual and identity mapping**: GCNII (Chen et al., 2020, ICML) mixes the *initial* features $\mathbf H^{(0)}$ into every layer, $\mathbf H^{(\ell)}=\sigma\big(((1-\alpha)\hat{\mathbf A}\mathbf H^{(\ell-1)}+\alpha\mathbf H^{(0)})((1-\beta_\ell)\mathbf I+\beta_\ell\mathbf W^{(\ell)})\big)$, and trains up to 64 layers. The $\alpha$ term is exactly the residual used in the demo and the chart above.
- **Decoupling propagation from transformation**: SGC (Wu et al., 2019) removes nonlinearities and computes $\hat{\mathbf A}^K\mathbf X$ once, then fits logistic regression. APPNP (Klicpera, Bojchevski and Guennemann, 2019, ICLR, "Predict then propagate") first predicts with an MLP and then propagates the *predictions* with personalized PageRank, $\mathbf Z^{(k+1)}=(1-\alpha)\hat{\mathbf A}\mathbf Z^{(k)}+\alpha\mathbf Z^{(0)}$, which is the same fixed-point-with-restart recursion as above.
- **Jumping knowledge** (Xu et al., 2018, ICML): concatenate or max-pool the representations from *all* layers so the classifier can choose the depth for each node.
- **Normalization and regularization**: PairNorm (Zhao and Akoglu, 2020, ICLR) re-centres and rescales features after each layer so that the total pairwise distance stays constant; **DropEdge** (Rong et al., 2020, ICLR) randomly removes edges during training, which slows the rate of smoothing and acts as augmentation; ordinary BatchNorm, LayerNorm and dropout also help optimization.

**Against over-squashing: change the graph, not the weights.**

- **Rewiring**: add edges that shortcut bottlenecks. DIGL (Gasteiger, Weissenberger and Guennemann, 2019) replaces the adjacency by a sparsified personalized-PageRank diffusion; SDRF (Topping et al., 2022) adds edges around the most negatively curved ones and removes edges at the most positively curved; FoSR (Karhadkar, Banerjee and Montufar, 2023) adds edges that increase the spectral gap.
- **Virtual nodes**: add one extra node connected to every node (Gilmer et al., 2017). It gives any two nodes a path of length 2 and carries global context, at the price of a bottleneck of its own.
- **Global attention**: graph Transformers (below) connect every node pair in a single layer, trading structural bias and $O(n^2)$ cost for long-range mixing.
- **Wider hidden dimensions** help in theory (Di Giovanni et al., 2023, ICML, "On over-squashing in message passing neural networks") but do not remove the topological bottleneck.

:::warning Rewiring is a trade-off
Adding edges mixes in information from nodes that were never connected in the data. On homophilous graphs this may hurt; on a molecule it changes the chemistry. Rewiring is a hyperparameter to validate, not a free lunch, and over-squashing and over-smoothing pull in opposite directions: shortcuts reduce squashing but speed up smoothing.
:::

## Pooling and readout {#pooling}

For graph-level prediction we need a single vector $\boldsymbol{h}_G$ from the $n$ node embeddings $\mathbf H\in\R^{n\times d}$, where $n$ varies from graph to graph. The readout must be **permutation invariant** and handle any $n$. This is the same problem as DeepSets (Zaheer et al., 2017, NeurIPS), whose theorem says that a function on a set is invariant if and only if it can be written as $\rho\big(\sum_{v}\phi(\boldsymbol{x}_v)\big)$ for suitable $\phi,\rho$ (for countable inputs, and under suitable conditions). That is exactly "per-node transformation, sum, then an MLP".

{{figure: week14-pooling | From node embeddings to a graph prediction: message-passing layers, then a permutation-invariant readout (flat or hierarchical), then an MLP head.}}

**Flat readouts.**

- **Sum**: $\boldsymbol{h}_G=\sum_v\boldsymbol{h}_v$. It counts: a graph with twice as many identical nodes has twice the output. This is the right choice for *extensive* properties such as molecular energy or number of rings, and it is what GIN uses (with the sum taken at every layer and the results concatenated).
- **Mean**: $\frac1n\sum_v\boldsymbol{h}_v$. Invariant to graph size, so better for *intensive* properties (a ratio, a type), but it cannot tell a graph from two disjoint copies of itself.
- **Max**: element-wise maximum over nodes. It detects whether a pattern occurs anywhere, and ignores how often.
- **Attention / Set2Set / gated readouts**: a learned weighting of nodes. Set2Set (Vinyals, Bengio and Kudlur, 2016) runs an LSTM that repeatedly attends over the set; gated attention readouts (Li et al., 2016, Gated Graph Neural Networks) compute $\sum_v\sigma(g(\boldsymbol{h}_v))\odot f(\boldsymbol{h}_v)$.

**Hierarchical pooling.** Flat readouts discard structure in one step. CNNs instead pool gradually, shrinking resolution while increasing receptive field, and several graph methods mimic that by **coarsening the graph**:

- **DiffPool** (Ying et al., 2018, NeurIPS) learns a soft assignment matrix $\mathbf S^{(\ell)}\in\R^{n_\ell\times n_{\ell+1}}$ of nodes to clusters with a GNN and a softmax, then forms a coarse graph $\mathbf X'=\mathbf S^\top\mathbf H$, $\mathbf A'=\mathbf S^\top\mathbf A\mathbf S$. It is differentiable but needs a dense $\mathbf S$ (memory $O(n^2)$) and auxiliary losses (link-prediction and entropy) to make assignments sensible.
- **Top-k / Graph U-Nets** (Gao and Ji, 2019, ICML) and SAGPool (Lee, Lee and Kang, 2019) score each node with a learned projection and keep the top fraction $k$, together with the induced subgraph. It is sparse and cheap but may disconnect the graph and drops nodes outright.
- **MinCut pooling** (Bianchi et al., 2020) and others learn clusterings with a relaxed normalized-cut objective.

An honest summary of the evidence: on many benchmarks, simple sum or mean readouts of a good GNN match or beat elaborate hierarchical pooling (Mesquita, Souza and Kaski, 2020, "Rethinking pooling in graph neural networks" report that random or local pooling often performs on par with learned pooling). Start simple, and try hierarchical pooling when graphs are large and have clear multiscale structure (proteins, large molecules, meshes).

```python:readout.py
import torch

def readout(h, batch, num_graphs, kind="sum"):
    """h: [N_total, d] node embeddings of a *batch* of graphs concatenated along dim 0.
    batch: LongTensor [N_total] giving the graph index of every node (as in PyG's Batch.batch)."""
    d = h.size(1)
    if kind in ("sum", "mean"):
        out = torch.zeros(num_graphs, d).index_add_(0, batch, h)
        if kind == "mean":
            counts = torch.zeros(num_graphs).index_add_(0, batch, torch.ones(h.size(0)))
            out = out / counts.clamp(min=1).unsqueeze(1)
        return out
    if kind == "max":
        out = torch.full((num_graphs, d), float("-inf"))
        return out.scatter_reduce(0, batch.unsqueeze(1).expand(-1, d), h, reduce="amax", include_self=True)
    raise ValueError(kind)

# two graphs with 3 and 2 nodes -> one concatenated batch of 5 nodes
h = torch.tensor([[1., 0.], [2., 1.], [3., 0.], [10., 5.], [20., 5.]])
batch = torch.tensor([0, 0, 0, 1, 1])
for kind in ("sum", "mean", "max"):
    print(kind, readout(h, batch, 2, kind).tolist())
```

```text
sum [[6.0, 1.0], [30.0, 10.0]]
mean [[2.0, 0.3333333432292938], [15.0, 5.0]]
max [[3.0, 1.0], [20.0, 5.0]]
```

Notice how the batch of graphs is handled: PyG **concatenates** the node features of all graphs in a mini-batch along one axis and the adjacency matrices become one big block-diagonal matrix (no edges between graphs), with a `batch` vector recording which graph each node belongs to. Message passing then needs no changes, and the readout is a *segment* reduction over `batch`. This is more efficient than padding to a common size.

## Tasks: node, link and graph prediction {#tasks}

### Node classification (and regression)

The setup in the GCN paper is **semi-supervised node classification** on a single large graph: a few nodes have labels (for Cora, 20 per class, 140 in total out of 2,708 nodes), and the goal is to label the rest. The loss is cross-entropy computed on the labeled nodes only, $\mathcal L=-\sum_{v\in\mathcal V_{\mathrm{train}}}\log p_\theta(y_v\mid\mathbf A,\mathbf X)_v$, while forward passes use the whole graph. The reason it works is the **homophily** assumption: connected nodes tend to share labels, so the label information of a few nodes can be diffused to their neighbourhoods. Compare with the bar chart below: an MLP that ignores the edges reaches roughly 55% on Cora, the GCN reaches about 81.5%. About 26 points come from the structure alone.

```chart
{"type":"bar","caption":"Test accuracy (%) on the Cora citation graph with the standard Planetoid split (140 labelled nodes). Values as reported in Kipf and Welling (2017) for LP, DeepWalk, Planetoid and GCN, and in Velickovic et al. (2018) for MLP and GAT.","height":330,"data":{"labels":["MLP","Label prop.","DeepWalk","Planetoid","GCN","GAT"],"datasets":[{"label":"Cora test accuracy (%)","data":[55.1,68.0,67.2,75.7,81.5,83.0],"backgroundColor":["var(--c-muted)","var(--c-muted)","var(--c-muted)","var(--c-muted)","var(--c-blue)","var(--c-green)"]}]},"options":{"scales":{"y":{"min":40,"max":90,"title":{"display":true,"text":"accuracy (%)"}}},"plugins":{"legend":{"display":false}}}}
```

Two practical points about node classification. **Splits**: the standard Planetoid split (Yang, Cohen and Salakhutdinov, 2016) uses 20 labeled nodes per class for training, 500 for validation and 1,000 for test; results on it are sensitive to the split, and Shchur et al. (2018, "Pitfalls of graph neural network evaluation") showed that with random splits and tuned baselines the ranking of models changes considerably. **Mini-batching**: for graphs too large for full-batch training, sample the computation tree of each target node (GraphSAGE's neighbour sampling, or cluster-based sampling as in Cluster-GCN), so each step touches a bounded subgraph.

### Link prediction

Link prediction asks whether an edge $(u,v)$ exists (or will exist). The standard recipe is an **encoder-decoder**: a GNN encoder produces node embeddings $\boldsymbol{h}_u$, and a decoder scores pairs, for example a dot product $s_{uv}=\boldsymbol{h}_u^\top\boldsymbol{h}_v$ or an MLP on $[\boldsymbol{h}_u\odot\boldsymbol{h}_v]$. The **graph auto-encoder** (GAE, Kipf and Welling, 2016) trains the encoder so that $\sigma(\boldsymbol{h}_u^\top\boldsymbol{h}_v)$ reconstructs the adjacency matrix. Training needs **negative examples**, because only positive edges are observed. Sample node pairs uniformly at random as negatives (usually one to several per positive) and minimize binary cross-entropy,
$$
\mathcal L=-\sum_{(u,v)\in E^+}\log\sigma(s_{uv})-\sum_{(u,v)\in E^-}\log\big(1-\sigma(s_{uv})\big).
$$
Alternatively use a ranking loss such as BPR (Bayesian personalized ranking, common for recommenders), $-\log\sigma(s_{uv^+}-s_{uv^-})$.

:::warning The two classic leaks in link prediction
(1) **Message-passing leakage**: if the target edge $(u,v)$ is in the graph used for message passing during training, the encoder can read the answer off the structure and the model collapses at test time. Hold edges out: split the edges into *message-passing edges* and *supervision edges*, and never supervise on edges you also propagate over (PyG's `RandomLinkSplit` handles this). (2) **Negative sampling at evaluation**: ranking one positive against randomly drawn easy negatives overstates quality; use harder negatives or full ranking metrics (Hits@K, MRR) as in OGB benchmarks.
:::

A known limit: a GNN encoder cannot distinguish nodes with isomorphic neighbourhoods, so two *different* pairs $(u,v)$ and $(u,w)$ with $v,w$ structurally equivalent receive the same score, even when one pair is far apart in the graph. SEAL (Zhang and Chen, 2018) fixes this by classifying the **enclosing subgraph** around each candidate pair, with labels that mark each node's distance to the two endpoints.

```python:link_prediction.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class LinkPredictor(nn.Module):
    """Encoder = any GNN producing [n, d] embeddings; decoder = dot product or MLP over pairs."""
    def __init__(self, encoder, d, decoder="dot"):
        super().__init__()
        self.encoder, self.kind = encoder, decoder
        if decoder == "mlp":
            self.mlp = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 1))

    def score(self, z, pairs):                    # pairs: [2, P] node ids
        zu, zv = z[pairs[0]], z[pairs[1]]
        return (zu * zv).sum(-1) if self.kind == "dot" else self.mlp(zu * zv).squeeze(-1)

    def loss(self, x, mp_edges, pos, neg):
        z = self.encoder(x, mp_edges)             # message passing ONLY over mp_edges (not over supervision edges)
        s = torch.cat([self.score(z, pos), self.score(z, neg)])
        y = torch.cat([torch.ones(pos.size(1)), torch.zeros(neg.size(1))])
        return F.binary_cross_entropy_with_logits(s, y)

def sample_negatives(num_nodes, num, device=None):
    """Uniform random pairs as negatives (may rarely hit a true edge; fine for large sparse graphs)."""
    return torch.randint(0, num_nodes, (2, num), device=device)

# usage sketch: GCN encoder from earlier in this page (reuse the GCN class with n_classes = embedding size)
# model = LinkPredictor(GCN(d_in, 64, 32), d=32)
# loss = model.loss(x, train_mp_edges, train_pos_edges, sample_negatives(n, train_pos_edges.size(1)))
```

### Graph classification and regression

Here each *graph* is a data point. The pipeline is message passing, then readout, then an MLP head: $\hat y=\operatorname{MLP}(\operatorname{READOUT}(\{\boldsymbol{h}_v^{(L)}\}))$. Datasets are collections of many small graphs: the TU benchmark collection (MUTAG with 188 molecules labeled by mutagenicity, PROTEINS, ENZYMES, social datasets such as IMDB-B), and molecular benchmarks (ZINC, QM9 for quantum properties, the MoleculeNet and OGB-mol datasets). Standard practice, learned the hard way: TU datasets are tiny and noisy, so report 10-fold cross-validation with a fixed protocol (Errica et al., 2020, "A fair comparison of graph neural networks for graph classification" documented that inconsistent protocols had inflated many claims), and use a simple baseline (a tuned degree-and-feature MLP, or a WL kernel) next to any GNN.

### Knowledge graphs and recommendation

For **knowledge graphs** the edges have types. R-GCN (Schlichtkrull et al., 2018) uses one weight matrix per relation,
$$
\boldsymbol{h}_v^{(\ell)}=\sigma\Big(\mathbf W_0\boldsymbol{h}_v^{(\ell-1)}+\sum_{r\in\mathcal R}\sum_{u\in\mathcal N_r(v)}\frac{1}{c_{v,r}}\mathbf W_r\boldsymbol{h}_u^{(\ell-1)}\Big),
$$
with $c_{v,r}$ a normalization such as $|\mathcal N_r(v)|$, and regularizes the $\mathbf W_r$ by basis decomposition (sharing a small set of bases across relations) to avoid parameter explosion with hundreds of relations. The classical non-GNN alternative, **TransE** (Bordes et al., 2013), embeds entities and relations so that $\boldsymbol{h}+\boldsymbol{r}\approx\boldsymbol{t}$ for true triples; GNN encoders add the ability to use node features and neighbourhoods. For **recommendation**, treat users and items as the two sides of a bipartite graph. LightGCN (He et al., 2020, SIGIR) shows that for collaborative filtering without node features one should *remove* the feature transformations and nonlinearities of GCN: the embedding of a user is a weighted sum of embeddings propagated over $K$ layers, $\mathbf E^{(k+1)}=\hat{\mathbf A}\mathbf E^{(k)}$, with the final embedding the average over layers and a BPR loss on user-item pairs. It is a good reminder that more machinery is not always better when the input has no rich features.

## Graph Transformers {#transformers}

Message passing is local. A Transformer ([Week 9](week-09.html)) lets every token attend to every other token in one layer, which removes both over-squashing (any two nodes are one hop apart) and the depth needed for long-range interactions. Applied naively to a graph, however, it treats nodes as a *set* and ignores the edges entirely: self-attention is permutation equivariant but knows nothing about who is connected to whom. A **graph Transformer** therefore has to **inject the structure**. There are three standard ways, usually combined.

1. **Positional and structural encodings on nodes.** Append to each node feature a vector describing where it sits in the graph: the first $k$ eigenvectors of the Laplacian (Dwivedi and Bresson, 2020; Kreuzer et al., 2021, "SAN"), random-walk return probabilities (RWSE), or degree and centrality. Laplacian eigenvectors generalize the sinusoidal position encodings of sequences: on a path graph they are exactly cosines.
2. **Attention bias from structure.** Add a learned bias to the attention logits that depends on the relationship between the two nodes: shortest-path distance, or edge features along the shortest path. Graphormer (Ying et al., 2021, NeurIPS) uses centrality (degree) encodings on nodes, spatial (shortest-path distance) encodings and edge encodings as biases, and won the OGB large-scale quantum-chemistry challenge.
3. **Hybrid local plus global layers.** GPS (Rampasek et al., 2022, NeurIPS, "Recipe for a general, powerful, scalable graph transformer") runs a message-passing layer and a global attention layer in parallel in each block and sums the outputs, with positional encodings fed in at the input. This keeps the local inductive bias that GNNs get cheaply, and adds global mixing.

Laplacian positional encodings come with an annoying **ambiguity**: if $\boldsymbol{v}$ is an eigenvector then so is $-\boldsymbol{v}$, and if an eigenvalue is repeated, any rotation inside the eigenspace is also valid. A network should not depend on these arbitrary choices. Common practice is to flip signs randomly during training (data augmentation), or to use sign-invariant architectures (SignNet, Lim et al., 2023) that process $\boldsymbol{v}$ and $-\boldsymbol{v}$ identically.

```python:laplacian_pe.py
import numpy as np

def laplacian_pe(A, k):
    """First k non-trivial eigenvectors of the symmetric normalized Laplacian as positional encodings."""
    deg = A.sum(1)
    L = np.eye(len(A)) - A / np.sqrt(np.outer(deg, deg))
    w, V = np.linalg.eigh(L)
    return V[:, 1:k + 1]                      # skip the trivial eigenvector (eigenvalue 0)

# path graph with 8 nodes: node 0 and node 7 are the far ends
n = 8
A = np.zeros((n, n))
for i in range(n - 1):
    A[i, i + 1] = A[i + 1, i] = 1
pe = laplacian_pe(A, 2)
print("PE shape:", pe.shape)
print("|pe(0)-pe(1)| =", np.round(np.linalg.norm(pe[0] - pe[1]), 3), "  |pe(0)-pe(7)| =", np.round(np.linalg.norm(pe[0] - pe[7]), 3))

# sign ambiguity: if v is an eigenvector then so is -v, so a model must not rely on the sign of a column
pe_flipped = pe * np.array([-1.0, 1.0])
print("raw PE identical after a sign flip:", np.allclose(pe, pe_flipped))
print("Gram matrix PE PE^T identical:     ", np.allclose(pe @ pe.T, pe_flipped @ pe_flipped.T))

# random-walk structural encoding: probability of returning to the start after k steps (k = 2, 4, 6, 8)
RW = A / A.sum(1, keepdims=True)
Pk, rwse = np.eye(n), []
for _ in range(8):
    Pk = Pk @ RW
    rwse.append(np.diag(Pk))
rwse = np.array(rwse).T                       # [n, 8]
print("RWSE of end node 0:     ", np.round(rwse[0, 1::2], 3))
print("RWSE of middle node 3:  ", np.round(rwse[3, 1::2], 3))
```

```text
PE shape: (8, 2)
|pe(0)-pe(1)| = 0.113   |pe(0)-pe(7)| = 0.756
raw PE identical after a sign flip: False
Gram matrix PE PE^T identical: True
RWSE of end node 0:      [0.5   0.375 0.312 0.273]
RWSE of middle node 3:   [0.5   0.375 0.328 0.309]
```

Note how the RWSE of two nodes that look alike for a few steps starts to differ at $k=6$ (the middle node of a path sees both ends, the end node sees one), which is information a 1-hop aggregator cannot extract cheaply.

Do graph Transformers beat message passing? The evidence is mixed. They help on tasks with strong long-range interactions and on large molecular datasets with good encodings, and several analyses show that GNNs with tuned hyperparameters and virtual nodes are competitive on the long-range benchmarks (Tonshoff et al., 2023, "Where did the gap go? Reassessing the long-range graph benchmark"). The cost is $O(n^2)$ attention (mitigated by sparse or linear attention such as Exphormer), and without good encodings a graph Transformer is *weaker* than a GNN because it has lost the edges. Treat "global attention plus structure encodings" as one more tool, not a replacement.

## Geometric deep learning: symmetry as a design principle {#gdl}

Step back from graphs. A CNN works because of **translation symmetry**: a cat shifted three pixels to the right is still a cat, and the same detector is useful at every position, so convolution shares weights across positions. A GNN works because of **permutation symmetry**: the same molecule with its atoms renumbered is the same molecule, so message passing treats every node identically and aggregates neighbours in an order-independent way. The programme of **geometric deep learning** (Bronstein, Bruna, Cohen and Velickovic, 2021, "Geometric deep learning: grids, groups, graphs, geodesics, and gauges") is to read both as instances of one recipe: *identify the symmetry group of your domain, then restrict the network to functions that respect it*. This shrinks the hypothesis space, improves sample efficiency and builds in generalization that data alone would need huge datasets to learn.

### Groups, actions, invariance and equivariance

A **group** $G$ is a set of transformations that can be composed and inverted (it contains an identity, every element has an inverse, composition is associative). Examples: the integers $\mathbb Z$ under addition (shifts of a sequence), the permutation group $S_n$ (relabelling $n$ nodes), the rotation group $SO(3)$, the group of rigid motions $SE(3)$ (rotations plus translations), the Euclidean group $E(3)$ (adding reflections). A group $G$ **acts** on a space $\mathcal X$ of signals: each $g\in G$ gives a map $\rho_{\mathcal X}(g):\mathcal X\to\mathcal X$ (a shift of an image, a permutation of the rows of a feature matrix, a rotation of a set of 3-D points) with $\rho(gh)=\rho(g)\rho(h)$.

:::definition Invariance and equivariance
Let $G$ act on the input space by $\rho_{\mathrm{in}}$ and on the output space by $\rho_{\mathrm{out}}$. A function $f$ is
- **$G$-equivariant** if $f(\rho_{\mathrm{in}}(g)\,x)=\rho_{\mathrm{out}}(g)\,f(x)$ for all $g\in G$ and all $x$: transform then compute equals compute then transform.
- **$G$-invariant** if $f(\rho_{\mathrm{in}}(g)\,x)=f(x)$ for all $g$. This is the special case where $\rho_{\mathrm{out}}$ is trivial (the identity).
:::

{{figure: week14-symmetry | Equivariance and invariance as commuting diagrams: applying the group action before or after $f$ gives the same result (equivariance), or no change at all in the output (invariance).}}

The right choice depends on the *output*. For a per-node or per-pixel prediction (segmentation, node classification, forces on atoms) you want **equivariance**: move the object and the output moves with it. For a single global answer (a class label, a molecular energy) you want **invariance**. The standard architecture pattern turns the first into the second: a stack of equivariant layers followed by an invariant pooling.

| Domain | Symmetry group | Equivariant layer | Invariant readout |
| --- | --- | --- | --- |
| Images on a grid | translations $\mathbb Z^2$ | convolution | global pooling |
| Sets, graphs (nodes) | permutations $S_n$ | message passing, DeepSets layer | sum/mean/max |
| Sequences with time warps | time re-parametrization (approx.) | gated RNN (heuristic) | final state |
| Spheres, rotated images | $SO(3)$ or $p4m$ | group / spherical convolution | invariant pooling |
| 3-D atoms and molecules | $E(3)$ or $SE(3)$ plus $S_n$ | equivariant message passing | invariant energy |
| Meshes, manifolds | gauge transformations | gauge-equivariant convolution | global pooling |

### The blueprint

Bronstein et al. summarize the recipe as a sequence of building blocks. Given signals on a domain $\Omega$ with symmetry group $G$: (1) **equivariant linear layers** (a general linear map restricted to commute with $G$); (2) a **pointwise nonlinearity**, which commutes with the group action whenever the action only permutes or shifts positions (for actions that also *mix* feature channels, such as rotations of vector features, a plain ReLU on the components breaks equivariance and special nonlinearities are needed); (3) optional **local pooling** (coarsening), which keeps equivariance approximately; and (4) a final **invariant global pooling**. This is *exactly* the CNN pipeline (conv, ReLU, pool, global average pool) and the GNN pipeline (message passing, nonlinearity, hierarchical pooling, readout).

:::math What does an equivariant linear layer look like?
For translations on a 1-D periodic grid, a linear map commuting with all shifts must be a **circular convolution**: this is the standard derivation of the convolution from symmetry rather than from intuition. For permutations the answer is simple enough to derive by hand. Let $\boldsymbol\Theta\in\R^{n\times n}$ act on a vector of node scalars. Equivariance requires $\boldsymbol\Theta\mathbf P=\mathbf P\boldsymbol\Theta$ for **every** permutation matrix $\mathbf P$. Take $\mathbf P$ to be the transposition of coordinates $i$ and $j$. Comparing entries gives $\Theta_{kk}=\Theta_{ll}$ for all $k,l$ (so the diagonal is a constant $a$) and, using a second transposition that fixes the pair, $\Theta_{kl}=\Theta_{k'l'}$ for all $k\ne l$, $k'\ne l'$ (all off-diagonal entries equal a constant $b$). Hence
$$
\boldsymbol\Theta=a\,\mathbf I+b\,\mathbf 1\mathbf 1^\top,\qquad\text{i.e.}\qquad (\boldsymbol\Theta\boldsymbol x)_i=a\,x_i+b\sum_jx_j .
$$
Each node combines its own value with the **sum over all nodes**. With feature channels $a,b$ become matrices: this is the DeepSets layer of Zaheer et al. (2017). Message passing is the same construction when the sum is restricted to graph neighbours.
:::

The NumPy check below confirms that the commutant of the permutation group on $\R^5$ is two-dimensional.

```python:equivariant_linear.py
import numpy as np
rng = np.random.default_rng(0)

# (1) the only linear maps R^n -> R^n that commute with every permutation: Theta = a*I + b*11^T
n = 5
def perm_equivariant_theta(a, b):
    return a * np.eye(n) + b * np.ones((n, n))
P = np.eye(n)[rng.permutation(n)]
Th = perm_equivariant_theta(0.7, -0.2)
print("a I + b 11^T commutes with P:", np.allclose(Th @ P, P @ Th))
Th_rand = rng.normal(size=(n, n))
print("a random matrix commutes with P:", np.allclose(Th_rand @ P, P @ Th_rand))

# (2) numerically recover the commutant: solve Theta P = P Theta for all transpositions -> dimension of solution space
rows = []
for i in range(n):
    for j in range(i + 1, n):
        Pt = np.eye(n); Pt[[i, j]] = Pt[[j, i]]
        # vec(Theta Pt - Pt Theta) = (Pt^T kron I - I kron Pt) vec(Theta)
        rows.append(np.kron(Pt.T, np.eye(n)) - np.kron(np.eye(n), Pt))
M = np.vstack(rows)
print("dimension of the space of permutation-equivariant linear maps:", n * n - np.linalg.matrix_rank(M))
```

```text
a I + b 11^T commutes with P: True
a random matrix commutes with P: False
dimension of the space of permutation-equivariant linear maps: 2
```

A lesson from this: **symmetry drastically reduces the parameter count.** A generic $n\times n$ layer has $n^2$ parameters; the permutation-equivariant one has two, independent of $n$. That independence of $n$ is why GNNs apply to graphs of any size.

:::note Approximate and learned symmetries
Real symmetries are often only approximate (a photograph is not exactly translation-invariant because of borders, a molecule's property is invariant to rigid motion but not to a change of conformation) and sometimes the symmetry is only *partial*, as with gravity breaking rotation invariance about a horizontal axis. Two practical alternatives to hard-wiring are **data augmentation** (train with random group elements and hope the network learns to be approximately invariant) and **canonicalization** (map every input to a canonical pose first, for example by aligning to principal axes). Hard-wired equivariance is exact and sample-efficient but needs custom layers, which can be slower than a Transformer on GPUs; this trade-off drives much current research.
:::

## Point clouds and 3-D data {#pointcloud}

A **point cloud** is a set $\{\boldsymbol p_1,\dots,\boldsymbol p_n\}\subset\R^3$ (optionally with colour, intensity or normals), as produced by LiDAR, depth cameras and photogrammetry. It is the native output of 3-D sensors and the main input of autonomous-driving perception and robotics. Point clouds break every assumption a CNN makes: they have no grid, density varies enormously (points are dense near the sensor and sparse far away), $n$ varies, and the points have **no order**. They are *sets*, so the symmetry group is $S_n$, and for many tasks you also want the answer to be unchanged by a rigid motion of the whole scene.

Three families of approaches existed before and alongside point-based networks: **voxelization** (rasterize into a 3-D occupancy grid and run a 3-D CNN, as in VoxNet and 3-D U-Nets; cubic memory growth limits resolution, though sparse convolutions fix much of it), **multi-view** methods (render the object from several viewpoints and apply a 2-D CNN, effective but hostile to scenes), and **graph/point-based** networks that consume points directly.

### PointNet

**PointNet** (Qi, Su, Mo and Guibas, 2017, CVPR, "PointNet: deep learning on point sets for 3D classification and segmentation") was the first deep network to consume raw point sets. The architecture implements the DeepSets recipe from the previous section:

$$
f(\boldsymbol p_1,\dots,\boldsymbol p_n)=\gamma\Big(\max_{i=1,\dots,n}\ h(\boldsymbol p_i)\Big),
$$

where $h:\R^3\to\R^{1024}$ is a **shared** MLP applied to each point independently, the max is taken **per feature channel** across the points, and $\gamma$ is an MLP on the resulting 1024-dimensional global descriptor. Because $\max$ is symmetric, $f$ is permutation-invariant by construction. Qi et al. prove that such functions can approximate any continuous set function arbitrarily well (continuous with respect to the Hausdorff distance), given enough channels, and give an interpretation: the network selects a sparse set of **critical points** (those that win the max in some channel) that determine the descriptor, and the output is stable to adding points that do not exceed the maxima.

{{figure: week14-pointnet | PointNet: a shared per-point MLP lifts each of $n$ points to 1024 features, a symmetric max-pool collapses the point axis, and an MLP head classifies. For segmentation the global feature is concatenated back to per-point features.}}

Two additions in the original paper deal with *rigid motions*: **T-Nets**, small sub-networks that predict a $3\times3$ (and a $64\times64$ feature-space) alignment matrix that is applied to the points (features) to canonicalize pose, with a regularizer $\|\mathbf I-\mathbf A\mathbf A^\top\|_F^2$ encouraging orthogonality. This is a learned canonicalization, and it is only approximately effective; in practice rotation *augmentation* matters as much. PointNet reported about 89.2% overall accuracy on ModelNet40 object classification with 1,024 points, a landmark for a method with no convolution at all.

The main weakness is that PointNet has **no notion of local structure**: each point is processed alone, and only the final max mixes information. **PointNet++** (Qi, Yi, Su and Guibas, 2017, NeurIPS) restores locality by applying PointNet recursively on local neighbourhoods selected by farthest-point sampling and ball queries, building a hierarchy like a CNN's. **DGCNN** (Wang et al., 2019, ACM TOG) builds a **k-nearest-neighbour graph** in feature space and applies EdgeConv, a message-passing layer whose messages are $\operatorname{MLP}(\boldsymbol h_i\Vert\boldsymbol h_j-\boldsymbol h_i)$; the graph is recomputed after every layer ("dynamic"). Point-cloud networks are thus GNNs on a graph that you construct yourself. Later work uses Transformers on points (Point Transformer, Zhao et al., 2021) and sparse 3-D convolutions (MinkowskiNet, Choy et al., 2019) for large scenes.

The following code builds the permutation-invariance claim in NumPy with random weights and then gives the PyTorch module. The NumPy test also shows the limits of that invariance: it holds for point order and for duplicated points, but **not** for rotations.

```python:pointnet_invariance.py
import numpy as np
rng = np.random.default_rng(0)

# a tiny PointNet-style set function: f(X) = rho( max_i h(x_i) ), random weights
W1, b1 = rng.normal(size=(3, 32)), rng.normal(size=32)
W2, b2 = rng.normal(size=(32, 64)), rng.normal(size=64)
Wr = rng.normal(size=(64, 10))
h = lambda X: np.maximum(0, np.maximum(0, X @ W1 + b1) @ W2 + b2)    # shared per-point MLP, [n,3] -> [n,64]
f = lambda X: np.maximum(0, h(X).max(axis=0)) @ Wr                    # max over points, then a linear head

X = rng.normal(size=(1024, 3))                                        # a point cloud with 1024 points
perm = rng.permutation(len(X))
print("invariant to point order:  ", np.allclose(f(X), f(X[perm])))

# rotation about the z axis by 30 degrees
th = np.pi / 6
R = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
print("invariant to rotation:     ", np.allclose(f(X), f(X @ R.T)))
print("relative change on rotation: %.2f" % (np.linalg.norm(f(X @ R.T) - f(X)) / np.linalg.norm(f(X))))
# duplicating points does not change a max-pooled feature
print("invariant to duplicates:   ", np.allclose(f(X), f(np.vstack([X, X[:100]]))))
```

```text
invariant to point order:   True
invariant to rotation:      False
relative change on rotation: 0.08
invariant to duplicates:    True
```

```python:pointnet_torch.py
import torch
import torch.nn as nn

class PointNetCls(nn.Module):
    """Vanilla PointNet classifier (no T-Nets). Input: [B, N, 3] points. Output: [B, num_classes]."""
    def __init__(self, num_classes=40):
        super().__init__()
        # shared per-point MLP implemented with 1x1 convolutions over the point axis: [B, C, N]
        self.mlp = nn.Sequential(
            nn.Conv1d(3, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, pts):
        x = self.mlp(pts.transpose(1, 2))      # [B, 1024, N]  per-point features
        x = x.max(dim=2).values                # [B, 1024]     symmetric max-pool over points
        return self.head(x)

model = PointNetCls().eval()
pts = torch.randn(8, 1024, 3)
out = model(pts)
perm = torch.randperm(1024)
print(out.shape, torch.allclose(out, model(pts[:, perm]), atol=1e-5))   # invariance to point order (eval mode)
```

In training mode BatchNorm statistics are computed over the batch *and* points but are still permutation-symmetric, so invariance holds; `.eval()` is used above only to make the comparison deterministic with respect to dropout.

## Equivariant networks: an overview {#equivariant}

Many 3-D problems have the symmetry that a rigid motion of the whole system changes nothing physically. The energy of a molecule is **invariant** under rotations and translations of its atoms, and the forces on its atoms are **equivariant** (they rotate with the molecule; indeed forces are minus the gradient of the energy, so an invariant energy gives equivariant forces automatically). Protein structure, material properties, particle simulations, and robotics policies share this structure. There are several ways to respect it, ordered roughly by how much machinery they need.

1. **Augmentation and canonicalization.** Feed randomly rotated copies, or align inputs to a canonical frame. Simple, approximate, and can be remarkably competitive when data is plentiful.
2. **Invariant features only.** Build the network from rotation-invariant quantities: interatomic distances (SchNet, Schutt et al., 2017, continuous-filter convolutions on distances), plus angles between bonds (DimeNet, Gasteiger, Groll and Guennemann, 2020) or dihedral angles. The network is exactly invariant, but purely scalar features can be incomplete: some distinct 3-D arrangements share all distances up to a given order.
3. **Equivariant vector/tensor features.** Keep, in addition to scalars, features that transform as vectors or higher-order tensors under rotation, and combine them only with operations that respect the group: Tensor Field Networks (Thomas et al., 2018), SE(3)-Transformers (Fuchs et al., 2020), and NequIP (Batzner et al., 2022, Nature Communications) for interatomic potentials. They use **irreducible representations** of $SO(3)$ (spherical harmonics), with Clebsch-Gordan tensor products as the equivariant way to multiply features; the e3nn library (Geiger and Smidt, 2022) implements them. These are the most expressive and data-efficient, but costlier and harder to write.
4. **Simple equivariant message passing on coordinates.** EGNN (Satorras, Hoogeboom and Welling, 2021, ICML, "E(n) equivariant graph neural networks") avoids spherical harmonics. It keeps invariant node features $\boldsymbol h_i$ and coordinates $\boldsymbol x_i$ and updates both,

$$
\begin{aligned}
\boldsymbol m_{ij}&=\phi_e\big(\boldsymbol h_i,\boldsymbol h_j,\|\boldsymbol x_i-\boldsymbol x_j\|^2,a_{ij}\big),\\
\boldsymbol x_i'&=\boldsymbol x_i+\frac{1}{|\mathcal N(i)|}\sum_{j\ne i}(\boldsymbol x_i-\boldsymbol x_j)\,\phi_x(\boldsymbol m_{ij}),\\
\boldsymbol h_i'&=\phi_h\Big(\boldsymbol h_i,\sum_{j\ne i}\boldsymbol m_{ij}\Big).
\end{aligned}
$$

Why this is equivariant, in two sentences: the messages $\boldsymbol m_{ij}$ depend on the coordinates only through squared distances, which are unchanged by rotations, reflections and translations, so $\boldsymbol h'$ is invariant; and the coordinate update adds a *linear combination of the relative vectors* $\boldsymbol x_i-\boldsymbol x_j$ with invariant scalar weights, and relative vectors rotate with the system, so $\boldsymbol x'$ rotates and translates with the input. The code below checks both numerically.

```python:egnn_check.py
import numpy as np
rng = np.random.default_rng(0)

d_h, d_m = 4, 8
We = rng.normal(size=(2 * d_h + 1, d_m)) * 0.5      # edge MLP (one layer + tanh for the test)
Wx = rng.normal(size=(d_m, 1)) * 0.5                # scalar weight for coordinate update
Wh = rng.normal(size=(d_h + d_m, d_h)) * 0.5        # node update

def egnn_layer(h, x):
    """E(n)-equivariant graph layer (Satorras et al. 2021) on a fully connected graph.
    h: [n, d_h] invariant features, x: [n, 3] coordinates."""
    n = len(h)
    diff = x[:, None, :] - x[None, :, :]                       # [n, n, 3]  x_i - x_j
    dist2 = (diff ** 2).sum(-1, keepdims=True)                 # [n, n, 1]  rotation/translation invariant
    hi = np.repeat(h[:, None, :], n, 1); hj = np.repeat(h[None, :, :], n, 0)
    m = np.tanh(np.concatenate([hi, hj, dist2], -1) @ We)      # messages m_ij  [n, n, d_m]
    mask = 1 - np.eye(n)[:, :, None]                           # no self messages
    x_new = x + (diff * (m @ Wx) * mask).sum(1) / (n - 1)      # x_i += mean_j (x_i - x_j) * phi_x(m_ij)
    h_new = np.tanh(np.concatenate([h, (m * mask).sum(1)], -1) @ Wh)
    return h_new, x_new

h = rng.normal(size=(6, d_h)); x = rng.normal(size=(6, 3))
# random rotation (via QR) with det +1, and a translation
Q, _ = np.linalg.qr(rng.normal(size=(3, 3))); Q *= np.sign(np.linalg.det(Q))
t = rng.normal(size=3)

h1, x1 = egnn_layer(h, x)
h2, x2 = egnn_layer(h, x @ Q.T + t)
print("h invariant:      ", np.allclose(h1, h2))
print("x equivariant:    ", np.allclose(x1 @ Q.T + t, x2))
# reflections (det = -1) are handled too, since this layer is E(3)- rather than SE(3)-equivariant
Rf = np.diag([1, 1, -1.0])
h3, x3 = egnn_layer(h, x @ Rf.T)
print("reflection too:   ", np.allclose(x1 @ Rf.T, x3) and np.allclose(h1, h3))
```

```text
h invariant:       True
x equivariant:     True
reflection too:    True
```

Beyond molecules, equivariant ideas drive several landmark systems. **Group-equivariant CNNs** (Cohen and Welling, 2016, ICML) extended translation equivariance to rotations and reflections of the image plane by convolving over a larger group, and **spherical CNNs** (Cohen, Geiger, Koehler and Welling, 2018, ICLR) did the same for $SO(3)$ signals on the sphere. AlphaFold 2 (Jumper et al., 2021, Nature) uses an *invariant point attention* module so that attention depends on the geometry of residue frames in a way that is unchanged by global rotation and translation, and equivariant diffusion models for molecule generation (Hoogeboom et al., 2022, ICML) apply the same idea to generative modeling ([Week 13](week-13.html)).

:::note Which symmetry, which output?
Choose the symmetry by looking at what *should not matter*. Predicting a scalar energy: invariant to $E(3)$ and $S_n$. Predicting forces or velocities: equivariant to $E(3)$. Predicting the orientation of an object: you must *not* be invariant to rotations (a rotation-invariant network cannot tell an upright from an upside-down chair). Chirality is another trap: reflections map a molecule to its mirror image, which can have different biological activity, so $SE(3)$ (rotations only) can be the correct group and an E(3)-invariant network will be blind to handedness unless it uses extra features.
:::

## Lab: GNNs with PyTorch Geometric on Cora and MUTAG {#lab}

**PyTorch Geometric (PyG)** is the standard library for graph deep learning. Its `Data` object holds a graph (`x` node features, `edge_index` of shape $[2,|E|]$, `y` labels, optional `edge_attr`), `DataLoader` batches many graphs by concatenation as described above, and the `torch_geometric.nn` module provides `GCNConv`, `SAGEConv`, `GATConv`, `GINConv`, pooling operators and much more. Install it in Colab with `pip install torch_geometric` (recent versions of PyG need no extra compiled wheels for the layers used here). In this lab you will (1) train GCN and GAT on Cora, (2) probe depth and over-smoothing, (3) train a GIN on MUTAG with cross-validation, and (4) verify permutation equivariance and invariance on the trained models.

### Part 1: Cora node classification {#lab-cora}

Cora is a citation network: 2,708 papers (nodes) in seven topics, 5,278 undirected citation links (stored as 10,556 directed edges), and a 1,433-dimensional binary bag-of-words feature vector per paper. Only 140 nodes (20 per class) are labeled for training.

```python:lab_cora_load.py
import torch
from torch_geometric.datasets import Planetoid

dataset = Planetoid(root="data/Planetoid", name="Cora")
data = dataset[0]
print(data)
print("classes:", dataset.num_classes, " features:", dataset.num_features)
print("train/val/test nodes:", int(data.train_mask.sum()), int(data.val_mask.sum()), int(data.test_mask.sum()))
print("undirected:", data.is_undirected(), " isolated nodes:", data.has_isolated_nodes())

# edge homophily: fraction of edges joining two nodes of the same class
src, dst = data.edge_index
print("edge homophily: %.3f" % (data.y[src] == data.y[dst]).float().mean())
```

```text
Data(x=[2708, 1433], edge_index=[2, 10556], y=[2708], train_mask=[2708], val_mask=[2708], test_mask=[2708])
classes: 7  features: 1433
train/val/test nodes: 140 500 1000
undirected: True  isolated nodes: False
edge homophily: 0.810
```

The homophily of about 0.81 is the quantity that makes a low-pass filter (GCN) the right model here: four out of five edges connect papers of the same topic. Now a GCN and a GAT with the hyperparameters of the original papers (hidden size 16 with dropout 0.5 and weight decay $5\times10^{-4}$ for GCN; 8 heads of 8 units for GAT).

```python:lab_cora_train.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, GATConv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = Planetoid(root="data/Planetoid", name="Cora")
data = dataset[0].to(device)

class GCN(nn.Module):
    def __init__(self, d_in, d_hid, n_cls, p=0.5):
        super().__init__()
        self.conv1, self.conv2, self.p = GCNConv(d_in, d_hid), GCNConv(d_hid, n_cls), p
    def forward(self, x, edge_index):
        x = F.dropout(x, self.p, self.training)
        x = F.relu(self.conv1(x, edge_index))            # A_hat X W1  (self loops + sym. norm added inside)
        x = F.dropout(x, self.p, self.training)
        return self.conv2(x, edge_index)                 # logits [N, n_cls]

class GAT(nn.Module):
    def __init__(self, d_in, d_hid, n_cls, heads=8, p=0.6):
        super().__init__()
        self.conv1 = GATConv(d_in, d_hid, heads=heads, dropout=p)             # output [N, heads * d_hid]
        self.conv2 = GATConv(d_hid * heads, n_cls, heads=1, concat=False, dropout=p)
        self.p = p
    def forward(self, x, edge_index):
        x = F.dropout(x, self.p, self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, self.p, self.training)
        return self.conv2(x, edge_index)

class MLP(nn.Module):                                    # structure-blind baseline
    def __init__(self, d_in, d_hid, n_cls, p=0.5):
        super().__init__()
        self.l1, self.l2, self.p = nn.Linear(d_in, d_hid), nn.Linear(d_hid, n_cls), p
    def forward(self, x, edge_index):
        x = F.dropout(x, self.p, self.training)
        return self.l2(F.dropout(F.relu(self.l1(x)), self.p, self.training))

def run(model, epochs=200, lr=0.01, wd=5e-4):
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    best_val, test_at_best = 0.0, 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])   # labelled nodes only
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            pred = model(data.x, data.edge_index).argmax(dim=1)
            accs = [(pred[m] == data.y[m]).float().mean().item() for m in (data.train_mask, data.val_mask, data.test_mask)]
        if accs[1] > best_val:                          # model selection on the validation set, never on test
            best_val, test_at_best = accs[1], accs[2]
        if epoch % 50 == 0:
            print(f"epoch {epoch:3d}  loss {loss.item():.3f}  train {accs[0]:.3f}  val {accs[1]:.3f}  test {accs[2]:.3f}")
    return best_val, test_at_best

d_in, n_cls = dataset.num_features, dataset.num_classes
for name, make in [("MLP", lambda: MLP(d_in, 64, n_cls)), ("GCN", lambda: GCN(d_in, 16, n_cls)), ("GAT", lambda: GAT(d_in, 8, n_cls))]:
    torch.manual_seed(0)
    val, test = run(make())
    print(f"{name}: best val {val:.3f}, test at best val {test:.3f}\n")
```

Expect roughly 55 to 60% test accuracy for the MLP, about 80 to 82% for GCN and 81 to 83% for GAT, with a spread of one to two points across seeds (the published numbers are 81.5% and 83.0%, shown in the chart above). If your GCN lands at 70% or below, look at the first item in the debugging section.

### Part 2: depth, over-smoothing and a remedy {#lab-depth}

Now replace the two-layer GCN by an $L$-layer one, and watch accuracy as $L$ grows. Add DropEdge via `dropout_edge` and a residual connection to see which of the remedies help.

```python:lab_cora_depth.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dropout_edge
# assumes `data`, `device`, `dataset` from the previous script

class DeepGCN(nn.Module):
    def __init__(self, d_in, d_hid, n_cls, L, p=0.5, residual=False, drop_edge=0.0):
        super().__init__()
        self.inp = nn.Linear(d_in, d_hid)
        self.convs = nn.ModuleList([GCNConv(d_hid, d_hid) for _ in range(L)])
        self.out = nn.Linear(d_hid, n_cls)
        self.p, self.residual, self.drop_edge = p, residual, drop_edge

    def forward(self, x, edge_index):
        if self.training and self.drop_edge > 0:
            edge_index, _ = dropout_edge(edge_index, p=self.drop_edge, force_undirected=True)
        h = F.relu(self.inp(F.dropout(x, self.p, self.training)))
        for conv in self.convs:
            z = F.relu(conv(F.dropout(h, self.p, self.training), edge_index))
            h = h + z if self.residual else z                  # residual: x_{l+1} = x_l + f(x_l)
        return self.out(h)

def final_dirichlet_energy(model):
    """Energy of the last hidden state, normalized by the energy of the input projection."""
    model.eval()
    with torch.no_grad():
        h0 = F.relu(model.inp(data.x)); h = h0
        for conv in model.convs:
            z = F.relu(conv(h, data.edge_index)); h = h + z if model.residual else z
        src, dst = data.edge_index
        e = lambda H: ((H[src] - H[dst]) ** 2).sum().item()
        return e(h / h.norm(dim=1, keepdim=True).clamp(min=1e-9)) / e(h0 / h0.norm(dim=1, keepdim=True).clamp(min=1e-9))

for L in (2, 4, 8, 16, 32):
    for tag, kw in [("plain", {}), ("residual", dict(residual=True)), ("residual+DropEdge", dict(residual=True, drop_edge=0.3))]:
        torch.manual_seed(0)
        model = DeepGCN(dataset.num_features, 64, dataset.num_classes, L, **kw)
        val, test = run(model, epochs=200, lr=0.01)          # `run` from the previous script
        print(f"L={L:2d} {tag:18s} val {val:.3f} test {test:.3f}  relative energy {final_dirichlet_energy(model):.3f}")
```

Typical pattern (the exact numbers depend on seeds and hardware; run it and compare): plain GCN accuracy falls from about 80% at $L=2$ to the 30 to 70% range at $L=16$ and beyond (training also gets harder, so part of the drop is optimization rather than smoothing), the relative energy of the final representation shrinks, while residual connections keep accuracy within a few points of the shallow model, and DropEdge gives a further modest gain. The point of the exercise is the *qualitative* effect: depth hurts a plain GCN, skip connections largely fix it, and *more layers do not help on Cora even when they train*, because a 2-hop neighbourhood already contains most of the useful signal.

### Part 3: graph classification on MUTAG with GIN {#lab-mutag}

MUTAG contains 188 small molecular graphs (about 18 atoms each) labeled by whether the compound is mutagenic. Atoms carry a 7-way one-hot atom type. With so little data, a single train/test split is very noisy, so we evaluate with 10-fold cross-validation, selecting the epoch by a fixed rule (final epoch) rather than peeking at the test fold.

```python:lab_mutag.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_add_pool, global_mean_pool

dataset = TUDataset(root="data/TUDataset", name="MUTAG")
print(dataset, "| graphs:", len(dataset), "| node features:", dataset.num_node_features, "| classes:", dataset.num_classes)
g = dataset[0]
print(g, "| nodes:", g.num_nodes, "| edges (directed):", g.num_edges)

class GIN(nn.Module):
    def __init__(self, d_in, d_hid, n_cls, layers=4, p=0.5, pool="sum"):
        super().__init__()
        self.convs, self.bns = nn.ModuleList(), nn.ModuleList()
        for l in range(layers):
            mlp = nn.Sequential(nn.Linear(d_in if l == 0 else d_hid, d_hid), nn.BatchNorm1d(d_hid), nn.ReLU(),
                                nn.Linear(d_hid, d_hid))
            self.convs.append(GINConv(mlp, train_eps=True))        # MLP((1 + eps) h_v + sum_u h_u)
            self.bns.append(nn.BatchNorm1d(d_hid))
        self.pool = global_add_pool if pool == "sum" else global_mean_pool
        self.head = nn.Sequential(nn.Linear(layers * d_hid, d_hid), nn.ReLU(), nn.Dropout(p), nn.Linear(d_hid, n_cls))

    def forward(self, x, edge_index, batch):
        reps = []
        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(bn(conv(x, edge_index)))
            reps.append(self.pool(x, batch))                       # readout after EVERY layer (jumping knowledge)
        return self.head(torch.cat(reps, dim=1))                   # [num_graphs, n_cls]

def train_epoch(model, loader, opt):
    model.train(); total = 0
    for b in loader:
        b = b.to(device)
        opt.zero_grad()
        loss = F.cross_entropy(model(b.x, b.edge_index, b.batch), b.y)
        loss.backward(); opt.step(); total += loss.item() * b.num_graphs
    return total / len(loader.dataset)

@torch.no_grad()
def accuracy(model, loader):
    model.eval(); correct = 0
    for b in loader:
        b = b.to(device)
        correct += (model(b.x, b.edge_index, b.batch).argmax(1) == b.y).sum().item()
    return correct / len(loader.dataset)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def cross_validate(pool="sum", folds=10, epochs=100, seed=0):
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(dataset), generator=gen)
    fold_size = (len(dataset) + folds - 1) // folds
    accs = []
    for k in range(folds):
        test_idx = perm[k * fold_size:(k + 1) * fold_size]
        train_idx = torch.cat([perm[:k * fold_size], perm[(k + 1) * fold_size:]])
        train_loader = DataLoader(dataset[train_idx], batch_size=32, shuffle=True)
        test_loader = DataLoader(dataset[test_idx], batch_size=64)
        torch.manual_seed(seed + k)
        model = GIN(dataset.num_node_features, 32, dataset.num_classes, pool=pool).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=25, gamma=0.5)
        for _ in range(epochs):
            train_epoch(model, train_loader, opt); sched.step()
        accs.append(accuracy(model, test_loader))
    t = torch.tensor(accs)
    return t.mean().item(), t.std().item()

for pool in ("sum", "mean"):
    m, s = cross_validate(pool)
    print(f"GIN with {pool} readout: {100*m:.1f} +- {100*s:.1f} % (10-fold CV)")
```

```text
MUTAG(188) | graphs: 188 | node features: 7 | classes: 2
DataBatch/Data(edge_index=[2, 38], x=[17, 7], edge_attr=[38, 4], y=[1]) | nodes: 17 | edges (directed): 38
```

The first graph has 17 atoms and 19 bonds, stored as 38 directed edges (the exact first-graph line is printed by your installation). With this protocol you should typically see accuracies of roughly 80 to 90% with a standard deviation of 5 to 8 points *across folds*: with 19 test graphs per fold, one graph is more than five points, so differences between models smaller than that are noise. Report the mean and std, and never choose the number of epochs on the test fold.

### Part 4: verify the symmetries on a trained model {#lab-symmetry}

A cheap and powerful test, worth adding to any GNN project: relabel the nodes and check that the outputs transform as promised.

```python:lab_symmetry_test.py
import torch
# assumes a trained node model `model` (Cora) and a trained graph model `gmodel` (MUTAG), `data`, `dataset` loaded

def relabel(x, edge_index, perm):
    """New node i is old node perm[i]. Returns relabelled features and edge_index."""
    inv = torch.empty_like(perm); inv[perm] = torch.arange(len(perm))
    return x[perm], inv[edge_index]

# node-level: the output must be EQUIVARIANT, out'[i] = out[perm[i]]
model.eval()
with torch.no_grad():
    out = model(data.x, data.edge_index)
    perm = torch.randperm(data.num_nodes, device=data.x.device)
    x2, ei2 = relabel(data.x, data.edge_index, perm)
    out2 = model(x2, ei2)
print("node outputs equivariant:", torch.allclose(out2, out[perm], atol=1e-4))

# graph-level: the output must be INVARIANT
gmodel.eval()
g = dataset[0].to(device)
batch = torch.zeros(g.num_nodes, dtype=torch.long, device=device)
with torch.no_grad():
    y1 = gmodel(g.x, g.edge_index, batch)
    perm = torch.randperm(g.num_nodes, device=device)
    x2, ei2 = relabel(g.x, g.edge_index, perm)
    y2 = gmodel(x2, ei2, batch)
print("graph output invariant:", torch.allclose(y1, y2, atol=1e-4))
```

Both lines should print `True`. If they do not, the culprit is almost always something in the pipeline that depends on node order: a feature built from node indices, a non-symmetric aggregation, or a `batch` vector that does not match the nodes.

### Part 5: experiments to try {#lab-experiments}

1. **Structure versus features.** On Cora, train the GCN with the real graph and with a *random* graph that has the same number of edges. Accuracy should fall towards the MLP level, which isolates how much of the gain is due to the true edges.
2. **WL power in practice.** On MUTAG, replace the atom features by a constant vector (so the model sees only the structure) and compare sum with mean readouts. The sum model can still count atoms and degrees; the mean model is weaker.
3. **Rewire.** Add a virtual node to each MUTAG graph (PyG: `torch_geometric.transforms.VirtualNode`) and see whether it helps; try a Laplacian positional encoding with `AddLaplacianEigenvectorPE` and compare.
4. **Link prediction on Cora.** Use `RandomLinkSplit(is_undirected=True, add_negative_train_samples=False)`, a two-layer GCN encoder and the decoder from the link prediction section, and report AUC on the test edges.

## Common mistakes and debugging tips {#mistakes}

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| GCN on Cora gets about 70% or less, training loss is fine | Row-normalized features were not used, dropout missing, or no weight decay; wrong split (training on all nodes) | Use `Planetoid` masks, dropout 0.5, weight decay $5\times10^{-4}$ on the first layer, 200 epochs; optionally `T.NormalizeFeatures()` |
| Test accuracy is much higher than expected | Test labels or test nodes used in training or model selection; leakage via features | Compute the loss only on `train_mask`; select epochs on `val_mask` |
| `nan` loss in a hand-written GAT | Isolated nodes with no self-loop give an all $-\infty$ softmax row | Add self-loops before attention; use `add_self_loops=True` in `GATConv` |
| Accuracy collapses for deeper models | Over-smoothing plus optimization difficulty | Residual or initial-residual connections, JK, DropEdge, or decouple propagation from transformation (APPNP, SGC) |
| Embeddings of high-degree nodes explode with sum aggregation | Unnormalized sum scales with degree | Use mean or symmetric normalization, or normalization layers; clip degrees in PNA-style scalers |
| Link prediction AUC near 1.0 in training, near 0.5 at test | Supervision edges are also used for message passing (leakage) | Separate message-passing edges from supervision edges (`RandomLinkSplit`) |
| Graph classification accuracy has huge variance between runs | Tiny dataset; single split; early stopping on test | 10-fold CV, fixed protocol, report mean and std, compare with a simple baseline |
| Model cannot distinguish graphs with identical degree sequences | Constant node features and 1-WL limits | Add structural features (degree, cycle counts, RWSE), or a more expressive architecture |
| Mini-batch of graphs gives wrong results | `batch` vector inconsistent with the nodes, or readout over the whole batch instead of per graph | Pass `batch` to the pooling op; check `readout(...).shape[0] == num_graphs` |
| Out-of-memory on a big graph | Full-batch training; dense adjacency; very high-degree nodes | Neighbour sampling (`NeighborLoader`), cluster sampling, sparse tensors, smaller hidden size |
| GNN on a heterophilous graph is worse than an MLP | Low-pass smoothing mixes different classes | Separate self and neighbour weights (SAGE), concatenate ego and neighbour embeddings, or use models designed for heterophily (e.g. H2GCN) |
| PointNet score drops on rotated test clouds | Network is invariant to order, not to rotation; no augmentation | Rotation augmentation, canonical alignment, or an equivariant network |
| Coordinates leak into an "invariant" model | Raw $xyz$ fed to an MLP | Use only distances/angles, or an equivariant architecture; test by rotating inputs |

:::tip A short checklist before you trust a GNN result
(1) Run the permutation test from Part 4. (2) Compare against a no-graph MLP and a one-layer model. (3) Check that train, validation and test nodes/edges are properly separated, including for link prediction. (4) Vary the seed at least five times and report the spread. (5) Look at a *shallower* and a *deeper* model; if depth does not matter, you likely have a feature-driven problem rather than a graph-driven one.
:::

## Exercises {#exercises}

:::exercise Exercise 1: Laplacian and smoothness (derivation)
(a) Prove $\boldsymbol x^\top\mathbf L\boldsymbol x=\sum_{(i,j)\in E}(x_i-x_j)^2$ for an undirected graph with $\mathbf L=\mathbf D-\mathbf A$, and conclude that $\mathbf L$ is positive semi-definite. (b) Show that the multiplicity of eigenvalue $0$ equals the number of connected components. (c) For the path graph on three nodes, compute the eigenvalues of $\mathbf L$.

:::details Solution
(a) $\boldsymbol x^\top\mathbf D\boldsymbol x=\sum_id_ix_i^2=\sum_{(i,j)\in E}(x_i^2+x_j^2)$ because node $i$'s square appears once per incident edge. And $\boldsymbol x^\top\mathbf A\boldsymbol x=\sum_{i,j}A_{ij}x_ix_j=2\sum_{(i,j)\in E}x_ix_j$. Subtracting gives $\sum_{(i,j)\in E}(x_i^2+x_j^2-2x_ix_j)=\sum_E(x_i-x_j)^2\ge0$, so every eigenvalue is nonnegative.

(b) $\mathbf L\boldsymbol x=\boldsymbol 0$ implies $\boldsymbol x^\top\mathbf L\boldsymbol x=0$, hence $x_i=x_j$ across every edge, so $\boldsymbol x$ is constant on each connected component; conversely every such vector is in the null space. The null space therefore has one dimension per component.

(c) $\mathbf L=\begin{pmatrix}1&-1&0\\-1&2&-1\\0&-1&1\end{pmatrix}$ has eigenvalues $0,1,3$ with eigenvectors $(1,1,1)$, $(1,0,-1)$, $(1,-2,1)$. The eigenvector for $3$ alternates in sign, the highest frequency.
:::
:::

:::exercise Exercise 2: One GCN layer by hand (coding or pencil)
Take the path graph $0-1-2$ with scalar features $\boldsymbol x=(1,2,3)^\top$ and no learned weights. Compute $\hat{\mathbf A}$ and $\hat{\mathbf A}\boldsymbol x$. Then explain why the output of node 1 is not simply the mean of $(1,2,3)$.

:::details Solution
With self-loops the degrees are $\tilde d=(2,3,2)$. Then $\hat A_{00}=\hat A_{22}=1/2$, $\hat A_{11}=1/3$, and $\hat A_{01}=\hat A_{12}=1/\sqrt{2\cdot3}\approx0.4082$, other entries zero. So
$\hat{\mathbf A}\boldsymbol x=(0.5+2(0.4082),\ 0.4082+2/3+3(0.4082),\ 2(0.4082)+1.5)=(1.3165,\ 2.2997,\ 2.3165)$.
Node 1's output is $2.2997$, not the mean $2$: the weights $(0.408,0.333,0.408)$ sum to $1.15$, not to $1$. Symmetric normalization is not an averaging operator: row sums differ from one (only the *top eigenvalue* equals one). A random-walk normalization $\tilde{\mathbf D}^{-1}\tilde{\mathbf A}$ would give exactly the mean $(1+2+3)/3=2$.
:::
:::

:::exercise Exercise 3: What 1-WL cannot see
(a) The triangular prism (two triangles joined by three parallel edges) and the complete bipartite graph $K_{3,3}$ are both 3-regular on 6 nodes. Show that 1-WL cannot distinguish them. (b) Name two properties on which they differ that a message-passing GNN therefore cannot detect. (c) Using the demo, find a pair of small graphs for which 1-WL succeeds after exactly one round.

:::details Solution
(a) Start with a constant colouring. Every node has the same colour and exactly three neighbours of that colour, so after round 1 every node in both graphs receives the same new colour (determined by (old colour, $\{c,c,c\}$)); by induction the colouring stays constant on both graphs, and the colour histograms agree at every round. (b) $K_{3,3}$ is bipartite and triangle-free; the prism contains triangles. A message-passing GNN cannot detect triangles or bipartiteness in this setting. (c) Any pair with different degree sequences works, for example a path on 5 nodes versus a star with one extra edge: round 1 already produces different colour counts because the degrees differ.
:::
:::

:::exercise Exercise 4: Sum, mean and max (conceptual plus construction)
(a) Give two multisets of scalars that mean and max both fail to distinguish but sum distinguishes. (b) Give two that sum and max fail to distinguish but mean distinguishes. (c) Argue that with a one-hot encoding of node types over a finite alphabet, sum aggregation is injective on multisets.

:::details Solution
(a) $\{1,1,2,2\}$ and $\{1,2\}$ have the same mean $1.5$ and the same max $2$, but sums $6$ and $3$. (b) Sum and max both ignore... need sum equal and max equal: $\{0,3,3\}$ versus $\{1,1,1,3\}$? sums are $6$ and $6$, maxes are $3$ and $3$, means are $2$ and $1.5$. So mean distinguishes them. (c) Encode element $a_k$ of an alphabet of size $K$ as the $k$-th basis vector. The sum of a multiset is the vector whose $k$-th entry is the number of times $a_k$ occurs, which determines the multiset exactly. The Xu et al. theorem generalizes this to countable alphabets with a learned $f$, and the MLP after the sum approximates any function of that count vector.
:::
:::

:::exercise Exercise 5: Spectral analysis of over-smoothing (challenging)
Let $\hat{\mathbf A}$ be the GCN matrix of a connected graph, with eigenvalues $1=\mu_1>\mu_2\ge\dots\ge\mu_n>-1$. (a) Prove $\mathcal E(\hat{\mathbf A}^K\mathbf X)\le\rho^{2K}\mathcal E(\mathbf X)$ with $\rho=\max_{i\ge2}|\mu_i|$ and $\mathcal E(\mathbf X)=\operatorname{tr}(\mathbf X^\top(\mathbf I-\hat{\mathbf A})\mathbf X)$. (b) For the residual iteration $\mathbf X_{k+1}=(1-\alpha)\hat{\mathbf A}\mathbf X_k+\alpha\mathbf X_0$, find the limit and show the energy converges to a positive value (in general). (c) Using the numbers from the chart, estimate how many layers plain propagation needs to reduce the energy by a factor $10^{-6}$ when $\rho=0.864$.

:::details Solution
(a) Expand $\mathbf X=\sum_i\boldsymbol u_i\boldsymbol c_i$ in the orthonormal eigenbasis. Then $\mathcal E(\mathbf X)=\sum_i(1-\mu_i)\|\boldsymbol c_i\|^2$, and $\hat{\mathbf A}^K$ multiplies $\boldsymbol c_i$ by $\mu_i^K$, so $\mathcal E(\hat{\mathbf A}^K\mathbf X)=\sum_i(1-\mu_i)\mu_i^{2K}\|\boldsymbol c_i\|^2$. The $i=1$ term vanishes since $1-\mu_1=0$. For $i\ge2$, $\mu_i^{2K}\le\rho^{2K}$, giving the bound.

(b) The iteration is a contraction (its linear part has spectral radius $(1-\alpha)<1$), with fixed point $\mathbf X_\infty=\alpha(\mathbf I-(1-\alpha)\hat{\mathbf A})^{-1}\mathbf X_0$. In the eigenbasis, component $i$ is $\boldsymbol c_i\cdot\alpha/(1-(1-\alpha)\mu_i)$, which stays nonzero for every $i$ with $\boldsymbol c_i\ne0$, so the energy is $\sum_i(1-\mu_i)\alpha^2\|\boldsymbol c_i\|^2/(1-(1-\alpha)\mu_i)^2>0$. The high-frequency parts are damped by a factor $\alpha/(1-(1-\alpha)\mu_i)\le1$ but not annihilated: for $\mu_i=0$ the factor is exactly $\alpha$.

(c) Need $\rho^{2K}\le10^{-6}$, i.e. $K\ge 6\ln10/(2\ln(1/0.864))=13.8/(2\cdot0.1462)\approx47$. (The measured energy at $K=32$ is about $5\times10^{-7}$, below the bound's $\rho^{64}\approx8\times10^{-5}$, because the bound is worst-case over the spectrum; the true decay for these features is dominated by the components that happen to carry energy.)
:::
:::

:::exercise Exercise 6: DropEdge experiment (coding)
Modify the NumPy over-smoothing code so that at every propagation step a random fraction $p$ of the (undirected) edges is removed before renormalizing, and measure the energy (computed on the full graph) after 4 and 16 layers for $p\in\{0,0.3,0.6\}$. Does DropEdge slow smoothing or speed it up, and why might it still help generalization?

:::details Solution
```python
import numpy as np
rng = np.random.default_rng(0)
n = 200
c = np.repeat([0, 1], n // 2)
P = np.where(c[:, None] == c[None, :], 0.08, 0.01)
U = rng.random((n, n))
A = np.triu(U < P, 1); A = (A | A.T).astype(float)
X0 = np.stack([(2 * c - 1) * 0.5, np.zeros(n)], 1) + rng.normal(size=(n, 2))
X0 = np.concatenate([X0, rng.normal(size=(n, 6))], 1)

def norm_adj(A):
    At = A + np.eye(len(A)); d = At.sum(1)
    return At / np.sqrt(np.outer(d, d))

def energy(X, A):
    d = A.sum(1) + 1
    i, j = np.nonzero(np.triu(A, 1))
    return (((X[i] / np.sqrt(d[i, None]) - X[j] / np.sqrt(d[j, None])) ** 2).sum())

def run(K, p_drop, seed=0):
    r = np.random.default_rng(seed)
    X = X0.copy()
    for _ in range(K):
        keep = np.triu(r.random((n, n)) >= p_drop, 1)
        keep = keep | keep.T
        X = norm_adj(A * keep) @ X          # a fresh random sub-graph at every layer
    return energy(X, A) / energy(X0, A)

for p in (0.0, 0.3, 0.6):
    print(p, ["%.2e" % run(K, p) for K in (4, 16)])
```
Output: $p=0$: $2.42\times10^{-3}$ (4 layers), $5.52\times10^{-5}$ (16); $p=0.3$: $7.67\times10^{-3}$, $1.67\times10^{-4}$; $p=0.6$: $1.86\times10^{-2}$, $2.85\times10^{-4}$. DropEdge slows the *rate* of smoothing (each step mixes along fewer edges, so the effective spectral gap is smaller), by a constant factor in the exponent, but energy still decays geometrically. So it is a mitigation, not a cure. Its benefit for generalization comes largely from acting as a data augmentation and as a regularizer that stops the network from relying on any single edge, as in dropout.
:::
:::

:::exercise Exercise 7: What would happen if... (conceptual)
Predict the consequence of each change to a GCN, and say how you would check it: (a) removing the self-loops from $\tilde{\mathbf A}$; (b) replacing $\hat{\mathbf A}$ by the unnormalized $\mathbf A$ on a graph with one node of degree 1000; (c) applying a GCN to a graph where adjacent nodes mostly have *different* labels (heterophily); (d) using only a sum readout on a molecular dataset where the label is "contains a benzene ring".

:::details Solution
(a) Without self-loops, a node's own features are dropped at every layer, so $\boldsymbol h_v^{(\ell)}$ depends on the node itself only through walks that return to it (even-length paths); on bipartite graphs the even and odd layers see entirely different node sets. Accuracy typically drops, and the filter keeps the highest-frequency component (see the filter chart). Check: ablate `add_self_loops`. (b) The hub receives a sum of 1000 neighbour vectors, so its activations are about 1000 times larger than a leaf's, destabilizing optimization (and a leaf receives the hub's huge message). Symmetric normalization down-weights both directions by $1/\sqrt{d_ud_v}$. Check: print activation norms versus degree. (c) A low-pass filter averages away exactly the signal that distinguishes the classes, so an MLP on the features alone can beat the GCN. Check the edge-homophily number first and compare against the MLP; use separate self and neighbour weights or heterophily-aware models. (d) A sum of node embeddings that came from message passing can encode the ring only if the network has learned features that fire on ring membership; since message passing cannot count 6-cycles in general (1-WL limit), a benzene detector can fail on graphs where the ring is replaced by something 1-WL-equivalent. In real chemistry data atom types and neighbourhood context usually suffice; adding ring-count features is the robust fix.
:::
:::

:::exercise Exercise 8: PointNet design (conceptual plus coding)
(a) What changes if you replace max-pooling in PointNet by sum or mean? Which invariances remain? (b) Show that PointNet is *not* invariant to rotations, and describe three ways to handle rotations. (c) Prove that a function of the form $f(X)=\rho(\sum_i\phi(\boldsymbol p_i))$ is permutation invariant, and explain what extra property max-pooling has that sum-pooling lacks.

:::details Solution
(a) All of sum, mean and max are symmetric, so permutation invariance remains for all three. Sum scales with the number of points (so a denser scan gives a different descriptor), mean is invariant to *duplicating all points* but sensitive to the density distribution, and max is invariant to adding points that do not exceed any channel maximum, which makes it stable to sampling density and to dropout of non-critical points but also discards counting information. (b) The shared MLP $h$ is a function of the raw coordinates, and an arbitrary $h$ is not rotation-symmetric: the NumPy test above changes the output by 8% under a $30^\circ$ rotation about the $z$ axis. Remedies: rotation augmentation, a learned or PCA canonical alignment (T-Net), or an equivariant/invariant architecture using only distances and angles. (c) Permuting the points permutes the terms of the sum, and addition is commutative, so the sum is unchanged. Max-pooling additionally has the *critical point* property: the output depends only on the points that attain some channel's maximum, so the function is robust to any perturbation that adds or removes the other points.
:::
:::

:::exercise Exercise 9: Link prediction leakage (what would happen if...)
You train a GCN encoder with dot-product decoder on a graph. Training positives are all edges in the graph, message passing also uses all edges, and you evaluate on held-out edges that were *deleted* from the graph before training. Training AUC is $0.99$ and test AUC is $0.60$. Explain and fix the setup.

:::details Solution
During training the target edge $(u,v)$ is itself a message-passing edge, so $\boldsymbol h_u$ already contains $\boldsymbol h_v$'s information (and vice versa), and a dot product between embeddings of connected nodes is high *because they are connected in the computation graph*, not because the model learned to predict links. At test time the edge is absent, the shortcut disappears, and performance falls to what has genuinely been learned. Fix: split training edges into a *message-passing* set and a *supervision* set; compute embeddings with message passing over the first set only and compute the loss on pairs from the second (re-sampled every epoch); validate and test with edges held out from both. PyG's `RandomLinkSplit` and the `edge_label_index` attribute implement this.
:::
:::

:::exercise Exercise 10: Equivariance of EGNN (challenging)
Prove that the EGNN coordinate update $\boldsymbol x_i'=\boldsymbol x_i+\frac1{n-1}\sum_{j\ne i}(\boldsymbol x_i-\boldsymbol x_j)\phi_x(\boldsymbol m_{ij})$ is equivariant to translations and orthogonal transformations: if $\boldsymbol x_i\mapsto\mathbf Q\boldsymbol x_i+\boldsymbol t$ with $\mathbf Q^\top\mathbf Q=\mathbf I$, then $\boldsymbol x_i'\mapsto\mathbf Q\boldsymbol x_i'+\boldsymbol t$. State which property of the messages you need, and explain why using $\phi_x$ with output in $\R^3$ (a vector) instead of a scalar would break it.

:::details Solution
The squared distance satisfies $\|(\mathbf Q\boldsymbol x_i+\boldsymbol t)-(\mathbf Q\boldsymbol x_j+\boldsymbol t)\|^2=\|\mathbf Q(\boldsymbol x_i-\boldsymbol x_j)\|^2=\|\boldsymbol x_i-\boldsymbol x_j\|^2$ because $\mathbf Q$ is orthogonal, so the messages $\boldsymbol m_{ij}$ (functions of invariant node features and squared distances) are unchanged, and so is the scalar $\phi_x(\boldsymbol m_{ij})$. The relative vectors transform as $\boldsymbol x_i-\boldsymbol x_j\mapsto\mathbf Q(\boldsymbol x_i-\boldsymbol x_j)$. Therefore the updated coordinate is
$\mathbf Q\boldsymbol x_i+\boldsymbol t+\frac1{n-1}\sum_j\mathbf Q(\boldsymbol x_i-\boldsymbol x_j)\phi_x(\boldsymbol m_{ij})=\mathbf Q\boldsymbol x_i'+\boldsymbol t$. The key property is that the messages are **invariant** and the scalar weight multiplies an **equivariant** vector. If $\phi_x$ returned a vector $\boldsymbol w_{ij}\in\R^3$ produced by an MLP, it would not transform as $\mathbf Q\boldsymbol w_{ij}$ (an arbitrary MLP is not rotation-equivariant), so the update would no longer commute with rotations. Translation is handled because only differences of coordinates enter.
:::
:::

## Key takeaways {#takeaways}

:::key Key takeaways
- A graph model must be **permutation equivariant** (node outputs) or **invariant** (graph outputs). Message passing, with a symmetric aggregation and shared weights, guarantees this by construction.
- A $k$-layer message-passing GNN is a function of each node's $k$-hop computation tree. Depth equals receptive-field radius.
- **GCN** is a first-order spectral low-pass filter, $\hat{\mathbf A}=\tilde{\mathbf D}^{-1/2}(\mathbf A+\mathbf I)\tilde{\mathbf D}^{-1/2}$ (eigenvalues in $(-1,1]$ thanks to the renormalization trick). **GraphSAGE** samples neighbours and keeps separate self weights (inductive, scalable); **GAT** learns neighbour weights by attention but remains a weighted mean; **GIN** uses sum plus an MLP and is as strong as 1-WL.
- Message-passing GNNs are **at most as expressive as 1-WL** and cannot count triangles, tell a 6-cycle from two triangles, or decide connectivity. Sum beats mean beats max on multisets. Beyond 1-WL you pay with higher-order tensors, extra structural features, randomness or subgraph methods.
- **Over-smoothing** (features converge as energy decays like $\rho^{2K}$) and **over-squashing** (an exponentially growing receptive field squeezed through a bottleneck) are different problems with different cures: residual or initial connections, normalization, DropEdge, jumping knowledge for the first; rewiring, virtual nodes, global attention for the second.
- Readouts must be permutation invariant: sum for extensive, mean for intensive, max for presence, attention or hierarchical pooling when structure matters. Simple readouts are strong baselines.
- Node tasks (semi-supervised, homophily), link prediction (negative sampling, avoid leakage), and graph tasks (batching, readout, cross-validation) each have their own evaluation traps.
- **Graph Transformers** need positional and structural encodings (Laplacian eigenvectors, RWSE) or attention biases to know the graph; they trade structural bias and $O(n^2)$ cost for global mixing.
- The **geometric deep learning blueprint**: choose the symmetry group, use equivariant layers, nonlinearities, coarsening, and an invariant readout. CNNs (translation), GNNs and DeepSets (permutation), PointNet (permutation on point sets) and EGNN-type networks ($E(3)$) are all instances. Symmetry reduces parameters (a permutation-equivariant linear map has two, whatever $n$).
:::

## Further reading {#reading}

**Essential**

- Thomas N. Kipf and Max Welling. *Semi-supervised classification with graph convolutional networks.* ICLR 2017. arXiv:1609.02907. The GCN paper; short, clear, and the derivation of the renormalization trick.
- Michael M. Bronstein, Joan Bruna, Taco Cohen and Petar Velickovic. *Geometric deep learning: grids, groups, graphs, geodesics, and gauges.* 2021. arXiv:2104.13478. The blueprint and the unifying view used in the second half of this week.
- Keyulu Xu, Weihua Hu, Jure Leskovec and Stefanie Jegelka. *How powerful are graph neural networks?* ICLR 2019. arXiv:1810.00826. GIN and the WL connection.

**Models and framework**

- Justin Gilmer, Samuel S. Schoenholz, Patrick F. Riley, Oriol Vinyals and George E. Dahl. *Neural message passing for quantum chemistry.* ICML 2017. arXiv:1704.01212.
- William L. Hamilton, Rex Ying and Jure Leskovec. *Inductive representation learning on large graphs.* NeurIPS 2017. arXiv:1706.02216. GraphSAGE.
- Petar Velickovic, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Lio and Yoshua Bengio. *Graph attention networks.* ICLR 2018. arXiv:1710.10903.
- Michael Schlichtkrull, Thomas N. Kipf, Peter Bloem, Rianne van den Berg, Ivan Titov and Max Welling. *Modeling relational data with graph convolutional networks.* ESWC 2018. arXiv:1703.06103. R-GCN.
- Michael Defferrard, Xavier Bresson and Pierre Vandergheynst. *Convolutional neural networks on graphs with fast localized spectral filtering.* NeurIPS 2016. arXiv:1606.09375.

**Expressivity and depth**

- Christopher Morris et al. *Weisfeiler and Leman go neural: higher-order graph neural networks.* AAAI 2019. arXiv:1810.02244.
- Qimai Li, Zhichao Han and Xiao-Ming Wu. *Deeper insights into graph convolutional networks for semi-supervised learning.* AAAI 2018. arXiv:1801.07606.
- Kenta Oono and Taiji Suzuki. *Graph neural networks exponentially lose expressive power for node classification.* ICLR 2020. arXiv:1905.10947.
- Uri Alon and Eran Yahav. *On the bottleneck of graph neural networks and its practical implications.* ICLR 2021. arXiv:2006.05205.
- Jake Topping, Francesco Di Giovanni, Benjamin P. Chamberlain, Xiaowen Dong and Michael M. Bronstein. *Understanding over-squashing and bottlenecks on graphs via curvature.* ICLR 2022. arXiv:2111.14522.
- Ming Chen, Zhewei Wei, Zengfeng Huang, Bolin Ding and Yaliang Li. *Simple and deep graph convolutional networks.* ICML 2020. arXiv:2007.02133. GCNII.
- Johannes Klicpera, Aleksandar Bojchevski and Stephan Guennemann. *Predict then propagate: graph neural networks meet personalized PageRank.* ICLR 2019. arXiv:1810.05997.
- Yu Rong, Wenbing Huang, Tingyang Xu and Junzhou Huang. *DropEdge: towards deep graph convolutional networks on node classification.* ICLR 2020. arXiv:1907.10903.

**Pooling, Transformers and evaluation**

- Manzil Zaheer et al. *Deep Sets.* NeurIPS 2017. arXiv:1703.06114.
- Rex Ying et al. *Hierarchical graph representation learning with differentiable pooling.* NeurIPS 2018. arXiv:1806.08804.
- Chengxuan Ying et al. *Do transformers really perform badly for graph representation?* NeurIPS 2021. arXiv:2106.05234. Graphormer.
- Ladislav Rampasek et al. *Recipe for a general, powerful, scalable graph transformer.* NeurIPS 2022. arXiv:2205.12454. GPS.
- Oleksandr Shchur, Maximilian Mumme, Aleksandar Bojchevski and Stephan Guennemann. *Pitfalls of graph neural network evaluation.* 2018. arXiv:1811.05868.
- Federico Errica, Marco Podda, Davide Bacciu and Alessio Micheli. *A fair comparison of graph neural networks for graph classification.* ICLR 2020. arXiv:1912.09893.

**Geometry, point clouds and equivariance**

- Charles R. Qi, Hao Su, Kaichun Mo and Leonidas J. Guibas. *PointNet: deep learning on point sets for 3D classification and segmentation.* CVPR 2017. arXiv:1612.00593.
- Charles R. Qi, Li Yi, Hao Su and Leonidas J. Guibas. *PointNet++: deep hierarchical feature learning on point sets in a metric space.* NeurIPS 2017. arXiv:1706.02413.
- Yue Wang et al. *Dynamic graph CNN for learning on point clouds.* ACM Transactions on Graphics, 2019. arXiv:1801.07829.
- Taco Cohen and Max Welling. *Group equivariant convolutional networks.* ICML 2016. arXiv:1602.07576.
- Victor Garcia Satorras, Emiel Hoogeboom and Max Welling. *E(n) equivariant graph neural networks.* ICML 2021. arXiv:2102.09844.
- Nathaniel Thomas et al. *Tensor field networks: rotation- and translation-equivariant neural networks for 3D point clouds.* 2018. arXiv:1802.08219.
- Simon Batzner et al. *E(3)-equivariant graph neural networks for data-efficient and accurate interatomic potentials.* Nature Communications, 2022. arXiv:2101.03164.

**Software**

- Matthias Fey and Jan Eric Lenssen. *Fast graph representation learning with PyTorch Geometric.* ICLR 2019 Workshop on Representation Learning on Graphs and Manifolds. arXiv:1903.02428. Documentation at pytorch-geometric.readthedocs.io.
