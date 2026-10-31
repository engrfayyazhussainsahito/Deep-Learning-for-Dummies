---
title: Convolutional Neural Networks
phase: Core Architectures
summary: Why locality, weight sharing and equivariance make convolutions the right prior for images; the arithmetic of convolution layers (stride, padding, dilation, 1x1, depthwise-separable, pooling, receptive fields); the gradients of a conv layer; LeNet, AlexNet and VGG; and how to train an image classifier on CIFAR-10.
---

:::key Learning objectives
By the end of this week you should be able to:
- Explain why locality, weight sharing and translation equivariance make convolution a strong inductive bias for images, and prove that a linear shift-equivariant map is a convolution.
- Distinguish convolution from cross-correlation, and compute output sizes for any combination of kernel, stride, padding and dilation.
- Count parameters and multiply-accumulates (MACs) of standard, 1x1, grouped, and depthwise-separable convolutions.
- Compute receptive fields layer by layer with the recurrence for size and jump.
- Derive the gradients of a convolution layer with respect to its input, kernel and bias, and implement them in NumPy.
- Describe LeNet-5, AlexNet and VGG in detail and say what each contributed.
- Build a data pipeline with augmentation, train a CNN on CIFAR-10, and run controlled experiments on depth, kernel size and batch normalization.
:::

Until about 2012, the state of the art in computer vision was a pipeline: hand-designed features (SIFT, HOG), then a shallow classifier. Convolutional neural networks replaced the first half of that pipeline with features that are *learned*, and in doing so they set off the deep learning boom. But CNNs are not magic. They work because they build two facts about images directly into the architecture: nearby pixels are related, and the same pattern can appear anywhere. Understanding exactly how those facts become a layer, what the layer costs, what it can see, and how its gradients flow is the goal of this week.

This week and the next form a pair. Here we build the toolbox (the convolution layer and its arithmetic, plus three classic networks and a training recipe). In [Week 6](week-06.html) we use the toolbox to study modern architectures, transfer learning, detection and segmentation. You will need the backpropagation machinery from [Week 2](week-02.html), the optimizers and batch normalization from [Week 3](week-03.html), and the regularization ideas (weight decay, data augmentation, early stopping) from [Week 4](week-04.html).

## Motivation: why not just use an MLP on pixels?

Take a modest color image of 224 x 224 pixels. Flattened, it is a vector with $3 \cdot 224 \cdot 224 = 150{,}528$ entries. Feed it to a fully connected hidden layer that has as many units as the image has pixels per channel, say $224 \cdot 224 \cdot 64 = 3{,}211{,}264$ hidden units (the size of a typical first convolutional feature map). The weight matrix has

$$
150{,}528 \times 3{,}211{,}264 \approx 4.8 \times 10^{11}
$$

entries: about 1.9 terabytes in float32 for *one layer*. A $3\times 3$ convolution producing the same 64 feature maps has $3\cdot 3\cdot 3\cdot 64 = 1{,}728$ weights plus 64 biases. That is a factor of roughly $3 \times 10^8$ fewer parameters. The saving is not an implementation trick; it is a statement about what we believe about the data.

{{figure: week05-connectivity | Fully connected versus locally connected versus convolutional connectivity. Locality shrinks each unit's fan-in; weight sharing (same colored arrows) reuses the same few weights at every position.}}

### Three assumptions

A convolutional layer makes three assumptions, each of which reduces the hypothesis space in a different way.

1. **Locality.** The information needed to decide what is at a pixel is mostly contained in a small neighbourhood of that pixel. An edge, a corner or a texture patch is a local property. Instead of connecting every output unit to every input unit, we connect it only to a window of size $k \times k$. This cuts the fan-in from $HWC_{in}$ to $k^2 C_{in}$.
2. **Weight sharing (stationarity).** The statistics of natural images are approximately the same everywhere: a vertical edge detector is useful in the top-left corner as well as in the bottom-right corner. So we use *the same* weights at every spatial position. This cuts the number of parameters by another factor of $HW$ and also means that every training image supplies $HW$ gradient contributions for each weight.
3. **Translation equivariance.** If the input pattern moves, the representation of that pattern should move by the same amount. This is not a third independent assumption: it is a *consequence* of using the same local operator everywhere, and we prove below that it is in fact equivalent to weight sharing for linear maps.

Hierarchy is the fourth ingredient, but it comes from stacking layers rather than from a single layer: local edges combine into local motifs, motifs into parts, parts into objects. Each layer sees the output of the previous one, so the region of the image that influences a unit (its *receptive field*) grows with depth.

:::history Where this came from
The idea of local receptive fields in vision comes from Hubel and Wiesel's recordings of cat visual cortex (1959 to 1962), which found simple cells responding to oriented edges in a small patch and complex cells that pooled over several simple cells. Kunihiko Fukushima's Neocognitron (1980) turned this into a layered artificial model with alternating "S" (feature) and "C" (pooling) layers, but it was trained without backpropagation. Yann LeCun and colleagues added gradient-based training (1989 to 1998), giving the first practical convolutional networks (LeNet) for handwritten digits at AT&T Bell Labs, where they were deployed to read bank cheques.
:::

### Translation equivariance, stated precisely

Let us work with one-dimensional signals on the integers to keep the algebra clean; the 2D case replaces indices by pairs. Define the **shift operator** $T_s$ by $(T_s x)[i] = x[i - s]$: it moves the signal $s$ steps to the right. A map $f$ is **equivariant** to shifts if $f(T_s x) = T_s f(x)$ for all $s$. It is **invariant** if $f(T_s x) = f(x)$.

Consider the layer $y[i] = \sum_j w[j]\, x[i+j]$ (a cross-correlation, to be defined shortly). Then

$$
f(T_s x)[i] = \sum_j w[j]\,(T_s x)[i+j] = \sum_j w[j]\, x[i + j - s] = y[i - s] = (T_s f(x))[i].
$$

So the layer is equivariant. The converse is the interesting statement, because it explains why we do not have a choice.

:::math Theorem (sketch): linear shift-equivariant maps are convolutions
Let $A$ be a linear map on sequences $x : \mathbb{Z} \to \mathbb{R}$ with $A T_s = T_s A$ for every $s$. Write $x = \sum_n x[n]\, e_n$ where $e_n$ is the unit impulse at position $n$, so that $e_n = T_n e_0$. By linearity and equivariance,

$$
A x = \sum_n x[n]\, A T_n e_0 = \sum_n x[n]\, T_n (A e_0) = \sum_n x[n]\, T_n a, \qquad a := A e_0 .
$$

Evaluating at position $i$ gives $(Ax)[i] = \sum_n x[n]\, a[i - n]$, which is the convolution $a * x$ of $x$ with the *impulse response* $a = Ae_0$. Therefore every shift-equivariant linear layer is a convolution; "weight sharing" is not an extra assumption but the only way to be equivariant. Locality is then the further restriction that $a$ has finite (small) support, so only $k$ weights are nonzero.
:::

Two caveats matter in practice. First, equivariance is exact only for the convolution with circular (wrap-around) boundaries; zero padding breaks it near the borders, and strided layers break it for shifts that are not a multiple of the stride. Second, equivariance is not invariance. For classification you want the final decision to be (approximately) *invariant* to the position of the object, and CNNs obtain this by interleaving equivariant layers with pooling and finishing with a global aggregation or a fully connected layer. The demo below lets you see all of this on a 1D signal.

{{demo: week05-equivariance | Equivariance of convolution versus the approximate invariance of pooling | 460}}

:::note CNNs are not rotation or scale equivariant
Convolutions are equivariant to translation only. Rotating an image by 30 degrees or scaling it by 2 changes the feature maps in a complicated way, and the network must learn to cope with these changes from data (this is one reason for data augmentation, covered later). Group-equivariant CNNs (Cohen and Welling, 2016) extend the idea to rotations and reflections.
:::

## Convolution and cross-correlation

### Definitions

The **discrete convolution** of an image $x$ with a kernel $w$ is

$$
(x * w)[i, j] = \sum_{u}\sum_{v} x[i - u,\, j - v]\; w[u, v],
$$

which *flips* the kernel (index $-u, -v$). The **cross-correlation** does not flip:

$$
(x \star w)[i, j] = \sum_{u}\sum_{v} x[i + u,\, j + v]\; w[u, v].
$$

Almost every deep learning library, including PyTorch's `nn.Conv2d`, implements cross-correlation but calls it convolution. The two differ only by a flip of the kernel, $x * w = x \star \tilde w$ with $\tilde w[u,v] = w[-u,-v]$. Because the kernel is *learned*, the distinction does not matter: if the optimal filter is $\tilde w$, the network will learn $\tilde w$ in the cross-correlation parametrization. The flip only matters when you load a hand-designed kernel from signal-processing literature, when you want the associativity and commutativity of true convolution (relevant for theory, not for practice), or when you implement the backward pass (below), where a flipped kernel appears naturally.

### A single layer with channels

For an input tensor $\mathbf{X} \in \mathbb{R}^{C_{in} \times H \times W}$ and a weight tensor $\mathbf{W} \in \mathbb{R}^{C_{out} \times C_{in} \times k_h \times k_w}$ with bias $\boldsymbol{b} \in \mathbb{R}^{C_{out}}$, the output for channel $o$ is

$$
\mathbf{Y}[o, i, j] = b_o + \sum_{c=1}^{C_{in}} \sum_{u=0}^{k_h - 1} \sum_{v=0}^{k_w - 1} \mathbf{W}[o, c, u, v]\; \mathbf{X}[c,\; i s + u d - p,\; j s + v d - p],
$$

where $s$ is the stride, $d$ the dilation and $p$ the padding (out-of-range entries of $\mathbf{X}$ are zero). Read this as: each output channel owns one 3D filter that spans all input channels and slides over space. The parameter count is

$$
\underbrace{C_{out}\, C_{in}\, k_h k_w}_{\text{weights}} + \underbrace{C_{out}}_{\text{biases}},
$$

independent of $H$ and $W$. The cost, counted in multiply-accumulates (MACs), is that count of weights times the number of output positions: $H_{out} W_{out}\, C_{out} C_{in} k_h k_w$.

{{figure: week05-conv-sliding | A 3x3 kernel slides over a 5x5 input; each output entry is a dot product of the kernel with a window. The same nine weights are used at every position.}}

### From-scratch implementation

The following NumPy implementation is deliberately naive (four nested loops) so that every index matches the formula above. It supports stride, zero padding and dilation, and we check it against the equivalent formulation with the flipped kernel.

```python:conv2d_numpy.py
import numpy as np

def conv2d(x, w, b=None, stride=1, pad=0, dilation=1):
    """Cross-correlation as in deep learning libraries.
    x: (C_in, H, W)   w: (C_out, C_in, kh, kw)   b: (C_out,)  ->  (C_out, H_out, W_out)
    """
    C_in, H, W = x.shape
    C_out, C_in2, kh, kw = w.shape
    assert C_in == C_in2, "channel mismatch"
    xp = np.pad(x, ((0, 0), (pad, pad), (pad, pad)))          # zero padding
    keh, kew = dilation * (kh - 1) + 1, dilation * (kw - 1) + 1  # effective kernel size
    H_out = (H + 2 * pad - keh) // stride + 1
    W_out = (W + 2 * pad - kew) // stride + 1
    y = np.zeros((C_out, H_out, W_out))
    for o in range(C_out):
        for i in range(H_out):
            for j in range(W_out):
                acc = 0.0
                for c in range(C_in):
                    for u in range(kh):
                        for v in range(kw):
                            acc += w[o, c, u, v] * xp[c, i * stride + u * dilation, j * stride + v * dilation]
                y[o, i, j] = acc + (0.0 if b is None else b[o])
    return y

def true_convolution(x, w, **kw):
    """Convolution = cross-correlation with the kernel flipped in both spatial axes."""
    return conv2d(x, w[:, :, ::-1, ::-1], **kw)

rng = np.random.default_rng(0)
x = rng.normal(size=(3, 8, 8))
w = rng.normal(size=(4, 3, 3, 3))
y = conv2d(x, w, stride=2, pad=1)
print("output shape:", y.shape)            # (4, 4, 4): floor((8 + 2 - 3)/2) + 1 = 4

# An impulse input reveals the kernel: cross-correlation returns the kernel flipped,
# while true convolution returns the kernel itself (the "impulse response").
imp = np.zeros((1, 5, 5)); imp[0, 2, 2] = 1.0
k = np.arange(9, dtype=float).reshape(1, 1, 3, 3)
print(conv2d(imp, k, pad=1)[0, 1:4, 1:4])
print(true_convolution(imp, k, pad=1)[0, 1:4, 1:4])
```

```text
output shape: (4, 4, 4)
[[8. 7. 6.]
 [5. 4. 3.]
 [2. 1. 0.]]
[[0. 1. 2.]
 [3. 4. 5.]
 [6. 7. 8.]]
```

In real code you never write these loops: libraries lower convolution to a large matrix multiplication (**im2col**, which unrolls every window into a column and multiplies by the flattened filters), to an FFT, or to Winograd transforms, whichever is fastest for the given shapes. Knowing that a convolution is "just" a structured matrix product is useful for the backward pass, as we shall see.

## Output-size arithmetic: stride, padding, dilation

Almost every shape bug in a CNN is an arithmetic mistake, so it pays to be fluent here.

### The formula

For one spatial dimension with input length $n$, kernel size $k$, padding $p$ on each side, stride $s$ and dilation $d$, the number of valid window positions is

$$
n_{out} = \left\lfloor \frac{n + 2p - d(k-1) - 1}{s} \right\rfloor + 1 .
$$

The derivation is one line of counting. The padded input has $n + 2p$ entries. A window with dilation $d$ spans $k_{eff} = d(k-1)+1$ entries. The first window starts at index $0$ and the last must start at or before $n + 2p - k_{eff}$. Starts are $0, s, 2s, \dots$, so the number of starts is $\lfloor (n + 2p - k_{eff})/s \rfloor + 1$.

Special cases worth memorizing:

| Setting | Formula | Example |
| --- | --- | --- |
| "valid" (p = 0, s = 1) | $n - k + 1$ | 32, k=5 gives 28 (LeNet C1) |
| "same" (s = 1, p = (k-1)/2, odd k) | $n$ | 3x3 conv with p = 1 keeps size |
| halve resolution | $s = 2$, $k = 3$, $p = 1$ gives $\lfloor (n-1)/2 \rfloor + 1 = \lceil n/2 \rceil$ | 56 gives 28 |
| AlexNet conv1 | $n=227, k=11, s=4, p=0$ gives $(227-11)/4 + 1 = 55$ | 55 |

### Stride, padding, dilation, intuitively

- **Stride** $s > 1$ evaluates the filter only at every $s$-th position. It downsamples the feature map by about a factor of $s$ and reduces compute by $s^2$, at the cost of skipping information. Strided convolution is a *learned* downsampling, an alternative to pooling.
- **Padding** adds a border of (usually zero) values so that border pixels are covered by as many windows as interior pixels, and so that the output size can be controlled. Without padding, a stack of 3x3 convolutions loses 2 pixels per layer. Zero padding also lets the network detect the border (it sees the artificial zeros), which has been shown to be used to encode absolute position. Alternatives are reflection, replication and circular padding (`padding_mode` in PyTorch).
- **Dilation** $d > 1$ spaces the kernel taps $d$ pixels apart, so a 3x3 kernel with $d=2$ covers a 5x5 area using only 9 weights. It enlarges the receptive field without extra parameters and without downsampling, and is the backbone of DeepLab-style segmentation networks ([Week 6](week-06.html#segmentation)) and of WaveNet-style temporal convolutions.

{{figure: week05-pad-stride-dil | The same 3x3 kernel with different stride, padding and dilation settings and the resulting output sizes.}}

:::warning Floor, not round
The floor in the formula means that for stride 2 an odd and the next even input size can give the same output size (for $k=3$, $p=1$, $s=2$ both $n=7$ and $n=8$ give $n_{out}=4$). Rows or columns at the end of the image are silently ignored if the window does not fit exactly. Frameworks differ: PyTorch floors; some TensorFlow modes use ceil for "same" padding. When porting weights between frameworks, check this.
:::

Playing with the following explorer is the fastest way to build intuition: edit the kernel entries, change stride, padding and dilation, and watch both the output size formula and the output values.

{{demo: week05-conv-explorer | Convolution explorer: paint an image, edit a 3x3 kernel, change stride, padding and dilation | 560}}

### Shape checks in PyTorch

```python:shapes.py
import torch
import torch.nn as nn

def out_size(n, k, s=1, p=0, d=1):
    """Output length of one spatial dimension (same formula PyTorch uses)."""
    return (n + 2 * p - d * (k - 1) - 1) // s + 1

x = torch.randn(8, 3, 224, 224)                      # (batch, C_in, H, W)
cases = [dict(k=3, s=1, p=1), dict(k=3, s=2, p=1), dict(k=7, s=2, p=3),
         dict(k=3, s=1, p=0), dict(k=3, s=1, p=2, d=2), dict(k=11, s=4, p=2)]
for c in cases:
    k, s, p, d = c["k"], c["s"], c["p"], c.get("d", 1)
    conv = nn.Conv2d(3, 16, kernel_size=k, stride=s, padding=p, dilation=d)
    y = conv(x)
    assert y.shape[-1] == out_size(224, k, s, p, d)
    n_par = sum(t.numel() for t in conv.parameters())
    print(f"k={k:2d} s={s} p={p} d={d} -> {tuple(y.shape)}   params={n_par}")

# weight tensor layout is (C_out, C_in, kh, kw); bias is (C_out,)
print(nn.Conv2d(3, 16, 3).weight.shape)               # torch.Size([16, 3, 3, 3])
```

```text
k= 3 s=1 p=1 d=1 -> (8, 16, 224, 224)   params=448
k= 3 s=2 p=1 d=1 -> (8, 16, 112, 112)   params=448
k= 7 s=2 p=3 d=1 -> (8, 16, 112, 112)   params=2368
k= 3 s=1 p=0 d=1 -> (8, 16, 222, 222)   params=448
k= 3 s=1 p=2 d=2 -> (8, 16, 224, 224)   params=448
k=11 s=4 p=2 d=1 -> (8, 16, 55, 55)   params=5824
torch.Size([16, 3, 3, 3])
```

### Transposed convolution (a preview)

Going the other way, from a small map to a larger one, is done by the **transposed convolution** (sometimes misleadingly called deconvolution). It is the linear map whose matrix is the transpose of the matrix of a strided convolution. Its output size is $n_{out} = (n-1)s - 2p + d(k-1) + 1 + \text{output\_padding}$. We need it for segmentation decoders in [Week 6](week-06.html) and for generative models later; the backward pass of a convolution with respect to its input, derived below, *is* a transposed convolution.

## Channels, 1x1 convolutions, and depthwise-separable convolutions

### What "channels" mean

A color image has three channels. After the first layer, an input with $C_{out}$ feature maps has $C_{out}$ channels, each map being the response of one filter. A single filter is *three-dimensional* (it spans all input channels), but it is slid only over the two spatial dimensions, hence "2D convolution". Each filter produces one output channel, and the number of filters is the main knob for model capacity: wider layers can detect more distinct patterns.

### The 1x1 convolution

A $1 \times 1$ convolution has $k_h = k_w = 1$. It looks at one pixel at a time and so it has no spatial extent: for each position it multiplies the channel vector $\mathbf{X}[:, i, j] \in \mathbb{R}^{C_{in}}$ by the same matrix $\mathbf{W} \in \mathbb{R}^{C_{out} \times C_{in}}$. It is a per-pixel fully connected layer. What is it good for?

- **Channel mixing and dimensionality change.** It lets you cheaply increase or decrease the number of channels (the "bottleneck" in ResNet, the "reduction" in Inception).
- **Adding nonlinearity and depth** at almost no cost in receptive field (Network-in-Network, Lin et al. 2014).
- **Cost control.** A 3x3 convolution from 256 to 256 channels costs $9 \cdot 256^2 = 589{,}824$ weights. Squeezing to 64 channels with a 1x1, applying the 3x3 at 64 channels, and expanding back with another 1x1 costs $256\cdot 64 + 9 \cdot 64^2 + 64 \cdot 256 = 69{,}632$ weights, about $8.5\times$ fewer.

### Depthwise-separable convolution

The standard convolution does two jobs in one step: it filters *spatially* and it *mixes channels*. The depthwise-separable convolution (Sifre's thesis 2014; popularized by Xception, Chollet 2017, and MobileNet, Howard et al. 2017) splits them:

1. **Depthwise convolution:** apply one $k \times k$ filter *per input channel* (no mixing across channels). Weight tensor $C_{in} \times 1 \times k \times k$. In PyTorch: `nn.Conv2d(C, C, k, groups=C)`.
2. **Pointwise convolution:** a $1 \times 1$ convolution that mixes the channels, $C_{out} \times C_{in} \times 1 \times 1$.

{{figure: week05-separable | Standard convolution versus depthwise followed by pointwise, with parameter counts for C_in = 64, C_out = 128, k = 3.}}

Counting parameters (ignoring biases):

$$
\underbrace{k^2 C_{in} C_{out}}_{\text{standard}} \quad\text{versus}\quad \underbrace{k^2 C_{in}}_{\text{depthwise}} + \underbrace{C_{in} C_{out}}_{\text{pointwise}} .
$$

The ratio is

$$
\frac{k^2 C_{in} + C_{in}C_{out}}{k^2 C_{in} C_{out}} = \frac{1}{C_{out}} + \frac{1}{k^2}.
$$

The same ratio holds for MACs because both are the parameter count times $H_{out}W_{out}$. With $k=3$ and $C_{out} \ge 64$ the saving is about $8$ to $9\times$. The separable convolution is a *low-rank factorization* of the filter bank: the standard weight tensor is constrained to be a sum of $k \times k$ spatial patterns shared across output channels. It is less expressive per parameter-count-free comparison, but much cheaper, so for a fixed budget you can afford more layers or wider layers, which is a net win on accuracy-per-FLOP.

:::example Counting for one MobileNet-style layer
Take a layer with $C_{in} = 128$, $C_{out} = 256$, $k = 3$ on a $28 \times 28$ map. Standard: $9 \cdot 128 \cdot 256 = 294{,}912$ weights and $294{,}912 \cdot 784 \approx 231$ M MACs. Separable: $9 \cdot 128 + 128 \cdot 256 = 33{,}920$ weights and $33{,}920 \cdot 784 \approx 26.6$ M MACs, so $8.7\times$ fewer. Ratio check: $1/256 + 1/9 = 0.115$, and $33{,}920/294{,}912 = 0.115$.
:::

{{demo: week05-separable-cost | Parameter and FLOP counter: standard, grouped, depthwise-separable, bottleneck, 1x1 | 460}}

**Grouped convolution** generalizes both extremes: split the channels into $g$ groups and convolve within each group only. Parameters are divided by $g$; $g = 1$ is the standard convolution and $g = C_{in} = C_{out}$ is depthwise. AlexNet used $g = 2$ originally to fit on two GPUs; ResNeXt ([Week 6](week-06.html#resnext-densenet)) used it as a design dimension ("cardinality").

:::warning FLOPs are not latency
Depthwise convolutions have low arithmetic intensity (few operations per byte moved), so on GPUs they are often memory-bound and run much slower than their FLOP count suggests. Measure latency on the target hardware; do not rely on MACs alone.
:::

Let us verify the counts in code and build a reusable separable block.

```python:separable.py
import torch
import torch.nn as nn

def count_params(m):
    return sum(p.numel() for p in m.parameters())

class DepthwiseSeparable(nn.Module):
    """3x3 depthwise -> BN -> ReLU -> 1x1 pointwise -> BN -> ReLU (MobileNet v1 style)."""
    def __init__(self, c_in, c_out, stride=1):
        super().__init__()
        self.dw = nn.Conv2d(c_in, c_in, 3, stride=stride, padding=1, groups=c_in, bias=False)
        self.bn1 = nn.BatchNorm2d(c_in)
        self.pw = nn.Conv2d(c_in, c_out, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.act(self.bn1(self.dw(x)))
        return self.act(self.bn2(self.pw(x)))

std = nn.Conv2d(128, 256, 3, padding=1, bias=False)
sep = DepthwiseSeparable(128, 256)
dw_pw = count_params(sep.dw) + count_params(sep.pw)
print("standard 3x3 :", count_params(std))              # 294912
print("dw + pw      :", dw_pw)                          # 1152 + 32768 = 33920
print("ratio        :", dw_pw / count_params(std))      # 0.115
x = torch.randn(2, 128, 28, 28)
print(std(x).shape, sep(x).shape)                       # both (2, 256, 28, 28)

def conv_macs(m, out):
    """Multiply-accumulates of a Conv2d for a given output tensor."""
    k = m.kernel_size[0] * m.kernel_size[1]
    return out.shape[2] * out.shape[3] * m.out_channels * (m.in_channels // m.groups) * k
print("MACs std:", conv_macs(std, std(x)) / 1e6, "M")   # ~231 M
```

## Pooling and other ways to downsample {#pooling}

A convolution with stride 1 keeps the spatial resolution, but a classifier eventually needs to throw spatial detail away: the label "cat" does not depend on exactly which pixel the whiskers are at. Downsampling also makes deeper layers cheaper and enlarges their receptive field. The oldest tool for this is pooling.

### Max, average and global pooling {#pooling-types}

A **pooling layer** slides a window of size $k \times k$ with stride $s$ over each channel independently and replaces every window with one number. It has no learned parameters and does not mix channels, so the number of channels is unchanged and the output size follows the same formula as a convolution.

- **Max pooling** returns $\max$ over the window. It keeps the strongest response of a feature detector in the window and discards its exact location within the window. The typical setting is $k = s = 2$ (non-overlapping, halves the resolution) or $k = 3, s = 2$ (overlapping, used by AlexNet).
- **Average pooling** returns the mean over the window. It is a fixed low-pass filter followed by subsampling. LeNet-5 used a trainable variant (a learned scale and bias per map); modern networks use the plain mean.
- **Global average pooling (GAP)** uses a window equal to the whole feature map, turning a $C \times H \times W$ tensor into a $C$-vector. It was introduced in Network-in-Network (Lin et al., 2014) and used by GoogLeNet and ResNet to replace the huge fully connected layers of AlexNet and VGG. A GAP followed by a linear layer is also what makes class-activation maps possible, since the logit of class $c$ is then a spatial average of a linear function of the last feature map.

Why does max pooling help with invariance? Suppose a feature detector fires at position $i$ with value $a$ and the object shifts by one pixel so that it fires at $i+1$ instead. If both positions are in the same pooling window, the pooled output is unchanged. The invariance is *local* and approximate: shifts across a window boundary do change the output, and for strided pooling that is one reason that small shifts of the input can flip a CNN's prediction (Azulay and Weiss, 2019, show that modern CNNs are less shift-invariant than one would hope). The demo in the motivation section lets you see this effect with a 1D signal.

:::definition Pooling output size
For input length $n$, window $k$, stride $s$ and padding $p$, $n_{out} = \lfloor (n + 2p - k)/s \rfloor + 1$. PyTorch's `ceil_mode=True` rounds up instead of down, which keeps a partial window at the border. Padding for max pooling pads with $-\infty$ (so that padding never wins the max), not with zero.
:::

### Backward pass through pooling {#pooling-backward}

Since pooling has no weights, its backward pass only routes gradients.

- **Max pooling:** the gradient of the window output with respect to the input is 1 for the argmax entry and 0 for all others. So the upstream gradient is delivered to the winning position only. (With ties, frameworks pick one entry; the choice is a subgradient.) The forward pass must therefore remember the argmax indices.
- **Average pooling:** every entry of a $k \times k$ window receives $1/k^2$ of the upstream gradient.

When overlapping windows share an input pixel, the contributions from the different windows simply add.

### Alternatives to pooling {#stride-alternatives}

Springenberg et al. ("Striving for Simplicity", 2015) showed that on CIFAR-10 you can replace each max-pooling layer by a convolution with stride 2 and match or beat the original accuracy. The pooling operator is then *learned*: instead of a fixed max, the network chooses a linear combination followed by the nonlinearity. Most modern architectures (ResNet, EfficientNet, ConvNeXt) use strided convolutions for the downsampling steps and a single global average pool at the end. Max pooling survives in places where a hard "is this feature present anywhere in the window" decision is wanted (for example the first stage of ResNet, which uses a 3x3 stride-2 max pool after the stem convolution).

A related concern is **aliasing**. Subsampling a signal by two without first low-pass filtering it violates the sampling theorem, so high-frequency patterns get folded into low frequencies. Zhang (2019, "Making Convolutional Networks Shift-Invariant Again") proposed *blur pooling*: low-pass filter with a small fixed kernel (for example the binomial filter $[1,2,1]/4$ in each direction) before subsampling. This improves both consistency under small shifts and accuracy.

| Method | Learned? | Parameters | Typical use |
| --- | --- | --- | --- |
| Max pool 2x2, s=2 | no | 0 | VGG, LeNet variants, early layers |
| Average pool | no | 0 | DenseNet transitions, final GAP |
| Strided conv (3x3, s=2) | yes | $9 C_{in} C_{out}$ | ResNet-style stage transitions |
| Blur pool + stride | no (fixed blur) | 0 | anti-aliased variants of any net |
| Global average pool | no | 0 | classifier head instead of big FC layers |

```python:pooling.py
import torch
import torch.nn as nn
import torch.nn.functional as F

x = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4).requires_grad_()
y = F.max_pool2d(x, kernel_size=2, stride=2)
print(y)                      # [[5, 7], [13, 15]]
y.sum().backward()
print(x.grad[0, 0])           # 1 at the argmax of each window, 0 elsewhere

# Average pooling spreads the gradient uniformly
x.grad = None
F.avg_pool2d(x, 2).sum().backward()
print(x.grad[0, 0])           # every entry 0.25

# Global average pooling makes the classifier head independent of input resolution
feat = torch.randn(8, 512, 7, 7)
head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(512, 10))
print(head(feat).shape)       # (8, 10)
print(head(torch.randn(8, 512, 11, 9)).shape)   # also (8, 10): works for any H, W
```

## Receptive fields {#receptive-fields}

The **receptive field** of a unit is the set of input pixels that can influence its value. It answers a question that matters in practice: can this network even *see* enough context to solve the task? A detector for a 200-pixel-wide object cannot work if the final features only see 50 pixels.

{{figure: week05-rf-growth | Tracing one unit of the third layer back through three 3-tap layers: its receptive field grows by (k-1) times the current jump at each layer.}}

### The recurrence {#rf-recurrence}

Track two quantities per layer $\ell$: the receptive field size $r_\ell$ (in input pixels, along one dimension) and the **jump** $j_\ell$, the distance in input pixels between the centers of two adjacent units in layer $\ell$. At the input, $r_0 = 1$ and $j_0 = 1$. A layer with kernel size $k$, stride $s$ and dilation $d$ has effective kernel $k_{eff} = d(k-1)+1$, and

$$
r_\ell = r_{\ell-1} + (k_{eff} - 1)\, j_{\ell-1}, \qquad j_\ell = j_{\ell-1}\, s .
$$

:::math Why the recurrence holds
A unit in layer $\ell$ looks at $k_{eff}$ units of layer $\ell-1$, whose centers are spaced $j_{\ell-1}$ input pixels apart. The outermost two of them are $(k_{eff}-1)\,j_{\ell-1}$ pixels apart, and each of them already has a receptive field of $r_{\ell-1}$ pixels. The union of the $k_{eff}$ fields is therefore an interval of length $r_{\ell-1} + (k_{eff}-1) j_{\ell-1}$ (the fields overlap or touch whenever $r_{\ell-1} \ge j_{\ell-1}$, which is always true). Strides multiply the jump because skipping $s$ units at layer $\ell-1$ skips $s\, j_{\ell-1}$ input pixels.

The center of the field follows $c_\ell = c_{\ell-1} + \left(\frac{k_{eff}-1}{2} - p\right) j_{\ell-1}$, which tells you where in the image the field sits when padding is used.
:::

The key observation is that the growth of $r$ is *linear* in depth for stride-1 layers (each 3x3 layer adds 2) but is multiplied by the accumulated stride after each downsampling: once the jump is $j$, every further 3x3 layer adds $2j$. That is why downsampling is the cheapest way to see more context. Two checks with numbers: a stack of $L$ stride-1 3x3 layers has $r = 2L + 1$; three of them have $r = 7$, exactly the same as a single 7x7 layer, which is the argument VGG made for small kernels (see below).

| Network (conv stack) | Layers | Final jump | Receptive field |
| --- | --- | --- | --- |
| LeNet-5 up to S4 | C1, S2, C3, S4 | 4 | 16 x 16 |
| AlexNet up to pool5 | 5 conv, 3 pool | 32 | 195 x 195 |
| VGG-16 up to pool5 | 13 conv, 5 pool | 32 | 212 x 212 |

These sizes are computed with the recurrence above (kernel sizes and strides from the architectures described below). VGG-16 is trained on 224 x 224 crops, so its last convolutional features see almost the whole image. Use the calculator to reproduce the table and to explore your own stacks.

{{demo: week05-rf-calculator | Receptive field calculator: size, jump and center for any stack of layers | 560}}

```chart
{
  "type": "line",
  "caption": "Receptive field (input pixels, one dimension) after each layer of the VGG-16 convolutional stack, computed with the recurrence r = r_prev + (k-1) * jump. Jumps double at every pooling layer, so growth accelerates with depth.",
  "height": 320,
  "data": {
    "labels": ["c1","c2","p1","c3","c4","p2","c5","c6","c7","p3","c8","c9","c10","p4","c11","c12","c13","p5"],
    "datasets": [
      {"label": "Receptive field", "data": [3,5,6,10,14,16,24,32,40,44,60,76,92,100,132,164,196,212], "borderColor": "var(--c-blue)", "backgroundColor": "var(--c-blue)", "tension": 0.1},
      {"label": "Input size (224)", "data": [224,224,224,224,224,224,224,224,224,224,224,224,224,224,224,224,224,224], "borderColor": "var(--c-muted)", "borderDash": [6,4], "pointRadius": 0}
    ]
  },
  "options": { "scales": { "x": {"title": {"display": true, "text": "layer (c = 3x3 conv, p = 2x2 max pool)"}}, "y": {"title": {"display": true, "text": "receptive field (pixels)"}} } }
}
```

### Theoretical versus effective receptive field {#effective-rf}

The recurrence gives the *theoretical* receptive field: every pixel in it *can* affect the unit. But not equally. Luo et al. (2016, "Understanding the Effective Receptive Field in Deep Convolutional Neural Networks") showed that the influence of input pixels on a unit, measured by the gradient magnitude, is approximately Gaussian: concentrated in the center and much smaller at the edge, and the effective field is a fraction of the theoretical one. The intuition is a sum of random contributions: the center pixel is reached by many more paths through the stack than a pixel at the border, so by a central-limit argument the path-count profile approaches a Gaussian whose standard deviation grows like $\sqrt{L}$ while the theoretical width grows like $L$. Training tends to enlarge the effective field somewhat, and skip connections (Week 6) change it by adding short paths.

:::tip Rules of thumb
- Match the receptive field of the last feature map to the size of the objects or context you need. For 32 x 32 CIFAR images, a field of 30 or more is plenty; for 224 x 224 images you want the final field to be comparable to the image size.
- To increase the field cheaply: add stride (cost: resolution), add dilation (cost: gridding artifacts if used carelessly), or use larger kernels or attention (cost: parameters and compute).
- Dilated stacks with $d = 1, 2, 4, 8, \dots$ grow the field exponentially in depth while keeping resolution; try the "dilated" preset in the calculator.
:::

## Backpropagation through a convolution layer {#backprop-conv}

A convolution layer is a linear map, so its backward pass is two more linear maps. The surprise is that they are themselves convolutions, which is why a framework can reuse the same optimized kernels for forward and backward. We derive the results in 1D with one channel first, then extend them.

{{figure: week05-backprop-conv | The backward pass of a 1D convolution (k = 3). The input gradient is a full convolution of the upstream gradient with the flipped kernel; the kernel gradient is a cross-correlation of the input with the upstream gradient.}}

### The 1D derivation {#backprop-1d}

Let the forward pass be $y[i] = b + \sum_{j=0}^{k-1} w[j]\, x[i+j]$ for $i = 0, \dots, N-k$ (stride 1, no padding), and let $g[i] = \partial \mathcal{L} / \partial y[i]$ be the gradient arriving from above. The chain rule needs $\partial y[i] / \partial w[j] = x[i+j]$ and $\partial y[i] / \partial x[m] = w[m-i]$ when $0 \le m - i \le k-1$ and 0 otherwise.

:::math Gradients of a 1D convolution
**Kernel.** Each weight $w[j]$ is used at every output position, so the contributions add up:

$$
\frac{\partial \mathcal{L}}{\partial w[j]} = \sum_{i} g[i]\, \frac{\partial y[i]}{\partial w[j]} = \sum_{i=0}^{N-k} g[i]\, x[i+j].
$$

This is a cross-correlation of the input $x$ with the upstream gradient $g$ (with $g$ playing the role of the kernel). **Bias.** $\partial \mathcal{L}/\partial b = \sum_i g[i]$.

**Input.** The input entry $x[m]$ feeds the outputs $y[i]$ with $i \in [m-k+1, m]$ (intersected with the valid range), each through the weight $w[m-i]$:

$$
\frac{\partial \mathcal{L}}{\partial x[m]} = \sum_{j=0}^{k-1} g[m-j]\, w[j], \qquad g[i] := 0 \text{ outside } [0, N-k].
$$

Substituting $i = m - j$ shows this is a true convolution $g * w$ of the upstream gradient with $w$. Equivalently it is a cross-correlation of $g$, zero-padded by $k-1$ on each side, with the *flipped* kernel $w[k-1-j]$. This is the "full" convolution in signal-processing terms.
:::

Two consequences are worth stating explicitly. First, the shapes work out: $g$ has length $N-k+1$, padding by $k-1$ on both sides gives $N + k - 1$, and a valid correlation with a length-$k$ kernel gives $N$, the length of $x$. Second, the weight gradient sums over *all* positions, so for a large feature map it is an average over many samples. This is part of the reason convolution kernels can be trained with few examples compared to fully connected layers, and also why the weight gradient is expensive: it costs as many MACs as the forward pass.

### Channels, stride and padding {#backprop-channels}

With channels, $\mathbf{Y}[n,o,i,j] = b_o + \sum_{c,u,v} \mathbf{W}[o,c,u,v]\, \mathbf{X}[n,c,si+u-p,sj+v-p]$ for batch element $n$. Let $\mathbf{G} = \partial \mathcal{L} / \partial \mathbf{Y}$. The same argument gives

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{W}[o,c,u,v]} = \sum_{n,i,j} \mathbf{G}[n,o,i,j]\; \mathbf{X}[n,c,si+u-p,sj+v-p],
$$

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{X}[n,c,a,b]} = \sum_{o,u,v,i,j \,:\, si+u-p = a,\; sj+v-p=b} \mathbf{W}[o,c,u,v]\; \mathbf{G}[n,o,i,j], \qquad \frac{\partial \mathcal{L}}{\partial b_o} = \sum_{n,i,j} \mathbf{G}[n,o,i,j].
$$

The input gradient reads: take every output location, multiply its gradient by the filter, and *scatter-add* the result into the input window that produced it. For stride 1 this is the full convolution with the flipped and channel-transposed kernel (swap the roles of $C_{in}$ and $C_{out}$). For stride $s > 1$ the upstream gradient is first **dilated** by inserting $s-1$ zeros between its entries, because positions that were skipped in the forward pass receive nothing. Padding in the forward pass becomes cropping of the gradient in the backward pass.

### Relation to the transposed convolution {#backprop-transposed}

Flatten the input and output into vectors. The forward convolution is then $\boldsymbol{y} = \mathbf{C}\boldsymbol{x}$ for a sparse matrix $\mathbf{C}$ whose rows contain shifted copies of the kernel. The backward pass to the input is simply $\partial\mathcal{L}/\partial \boldsymbol{x} = \mathbf{C}^\top \boldsymbol{g}$. The map $\boldsymbol{g} \mapsto \mathbf{C}^\top\boldsymbol{g}$ is what we call a **transposed convolution**: it maps a small tensor (the size of $\mathbf{Y}$) to a large one (the size of $\mathbf{X}$) using the same weights. The name "deconvolution" is wrong because $\mathbf{C}^\top$ is not the inverse of $\mathbf{C}$; it only restores the shape. `nn.ConvTranspose2d` is therefore implemented by calling the backward-data kernel of a convolution, and the weight tensor of `ConvTranspose2d(C_in, C_out, k)` has shape $(C_{in}, C_{out}, k, k)$, the transpose of what a `Conv2d` mapping the other way would store. The gradient with respect to the weights is the same expression as before, with the roles of input and upstream gradient exchanged.

### A NumPy implementation with a gradient check {#backprop-code}

The code below implements forward and backward passes with batch, channels, stride and padding, using one loop over the kernel taps. For each tap $(u,v)$ the forward pass is a channel-mixing matrix product on a strided slice of the input, and the backward pass is its transpose, a scatter-add into the same slice. We verify against central finite differences.

```python:conv_backward.py
import numpy as np

def conv_forward(x, w, b, stride=1, pad=0):
    """x: (N, C, H, W), w: (O, C, kh, kw), b: (O,) -> y: (N, O, Ho, Wo)."""
    N, C, H, W = x.shape
    O, _, kh, kw = w.shape
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    Ho = (H + 2 * pad - kh) // stride + 1
    Wo = (W + 2 * pad - kw) // stride + 1
    y = np.zeros((N, O, Ho, Wo))
    for u in range(kh):
        for v in range(kw):
            patch = xp[:, :, u:u + stride * Ho:stride, v:v + stride * Wo:stride]  # (N, C, Ho, Wo)
            y += np.einsum('nchw,oc->nohw', patch, w[:, :, u, v])
    y += b[None, :, None, None]
    return y, (x, w, stride, pad)

def conv_backward(dy, cache):
    x, w, stride, pad = cache
    N, C, H, W = x.shape
    O, _, kh, kw = w.shape
    Ho, Wo = dy.shape[2:]
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    dxp = np.zeros_like(xp)
    dw = np.zeros_like(w)
    db = dy.sum(axis=(0, 2, 3))
    for u in range(kh):
        for v in range(kw):
            sl = (slice(None), slice(None), slice(u, u + stride * Ho, stride), slice(v, v + stride * Wo, stride))
            dw[:, :, u, v] = np.einsum('nohw,nchw->oc', dy, xp[sl])   # correlate x with dy
            dxp[sl] += np.einsum('nohw,oc->nchw', dy, w[:, :, u, v])  # scatter-add = transposed conv
    dx = dxp[:, :, pad:pad + H, pad:pad + W]                           # crop the padding
    return dx, dw, db

def num_grad(f, a, eps=1e-6):
    g = np.zeros_like(a)
    for idx in np.ndindex(*a.shape):
        old = a[idx]
        a[idx] = old + eps; fp = f()
        a[idx] = old - eps; fm = f()
        a[idx] = old
        g[idx] = (fp - fm) / (2 * eps)
    return g

rng = np.random.default_rng(0)
x = rng.normal(size=(2, 3, 7, 7)); w = rng.normal(size=(4, 3, 3, 3)); b = rng.normal(size=4)
y, cache = conv_forward(x, w, b, stride=2, pad=1)
r = rng.normal(size=y.shape)                        # L = sum(r * y), so dL/dy = r
dx, dw, db = conv_backward(r, cache)
L = lambda: (r * conv_forward(x, w, b, 2, 1)[0]).sum()
for name, a, g in [("x", x, dx), ("w", w, dw), ("b", b, db)]:
    print(name, "max abs error:", np.abs(num_grad(L, a) - g).max())
```

```text
x max abs error: 1.7e-08
w max abs error: 1.3e-08
b max abs error: 5.9e-09
```

:::note Finite-difference checks
Always write the gradient check for a new layer before you trust it. Use float64, a random projection of the output as the scalar loss (as above) so that every output entry matters, and a step around $10^{-6}$. Errors near $10^{-8}$ mean the derivation is right; errors of order one mean a transposition or a flip is wrong.
:::

## Classic architectures: LeNet-5, AlexNet, VGG {#classic-architectures}

Three networks, spanning 1998 to 2014, contain most of the design vocabulary of CNNs. Reading their layer tables is worth the effort: you will practice the shape arithmetic, see where the parameters and the compute live, and understand which ideas each design introduced.

{{figure: week05-classic-archs | LeNet-5, AlexNet and VGG-16 drawn on a common scheme: blue = convolution, amber = pooling, purple = fully connected. Resolution halves at each pool while channels grow.}}

### LeNet-5 (LeCun, Bottou, Bengio and Haffner, 1998) {#lenet}

LeNet-5 was designed for 32 x 32 grayscale images of handwritten digits (MNIST digits are 28 x 28 and padded to 32 so that characters are centered in the receptive fields of the top-level features). It is small enough to print in full.

| Layer | Operation | Output shape | Parameters |
| --- | --- | --- | --- |
| Input | 32 x 32 image | 1 x 32 x 32 | 0 |
| C1 | conv 5x5, 6 filters | 6 x 28 x 28 | $6(25+1) = 156$ |
| S2 | pool 2x2, stride 2 | 6 x 14 x 14 | 0 (12 in the original) |
| C3 | conv 5x5, 16 filters | 16 x 10 x 10 | $16(6 \cdot 25 + 1) = 2{,}416$ (1,516 in the original) |
| S4 | pool 2x2, stride 2 | 16 x 5 x 5 | 0 (32 in the original) |
| C5 | conv 5x5, 120 filters | 120 x 1 x 1 | $120(16 \cdot 25 + 1) = 48{,}120$ |
| F6 | fully connected | 84 | $84 \cdot 121 = 10{,}164$ |
| Output | fully connected | 10 | $10 \cdot 85 = 850$ |
| **Total** | | | **61,706** (about 60,850 in the original) |

Details that differ from the textbook "modern" version: the original used a scaled hyperbolic tangent, subsampling layers with a trainable coefficient and bias per map, a *sparse connection table* in C3 (each of the 16 maps looks at only a subset of the 6 input maps, which reduced parameters and broke symmetry), and radial-basis-function output units instead of softmax. The version found in most tutorials (and below) uses full connectivity, ReLU or tanh, average or max pooling and a softmax output. C5 is formally a convolution, but since its input is 5 x 5 and its kernel is 5 x 5, it is equivalent to a fully connected layer from 400 inputs to 120 units, which is why LeNet's paper calls it a convolutional layer only to keep it usable on larger inputs.

:::history LeNet in the world
Backpropagation-trained convolutional networks for zip-code digits appeared in 1989 (LeCun et al.), and the 1998 Proceedings of the IEEE paper "Gradient-based learning applied to document recognition" gave the full system, including a graph-transformer layer for reading whole cheques. For about a decade afterwards, support vector machines and boosting dominated vision benchmarks, partly because they were easier to train and tune, partly because datasets and compute were too small for large CNNs to show their advantage.
:::

### AlexNet (Krizhevsky, Sutskever and Hinton, 2012) {#alexnet}

AlexNet won ILSVRC 2012 (the ImageNet classification challenge) with a top-5 error of 15.3%, against 26.2% for the second-placed entry, a gap so large that it redirected the field. Its contributions are best understood as a collection of *enabling* ideas that made a large CNN trainable at that scale:

- **ReLU activations** instead of tanh or sigmoid, which the authors reported to train several times faster on CIFAR-10 to a fixed training error ([Week 2](week-02.html) and [Week 3](week-03.html) explain why: no saturation for positive inputs).
- **GPU training**: two GTX 580 GPUs with 3 GB each, with the network split across them (the origin of the grouped convolutions in conv2, conv4 and conv5, whose filters see only the maps on their own GPU), taking about five to six days.
- **Dropout** (0.5) in the two big fully connected layers, to fight overfitting of the 60M parameters on 1.2M images.
- **Data augmentation**: random 224 x 224 crops of 256 x 256 images, horizontal flips, a PCA-based color perturbation at training time; averaging 10 crops (4 corners + center, and flips) at test time.
- **Overlapping pooling** (3 x 3, stride 2) and **local response normalization** (LRN) after conv1 and conv2. LRN was later shown to be unnecessary and was replaced by batch normalization.
- SGD with momentum 0.9, weight decay $5 \times 10^{-4}$, batch size 128, learning rate 0.01 divided by 10 when validation error stopped improving.

The table below is the two-GPU network with the grouped layers counted as in the paper (so conv2, conv4 and conv5 have half the input channels per filter). Input crops are effectively 227 x 227 (the paper says 224, but 227 is the size for which the arithmetic works).

| Layer | Operation | Output shape | Parameters |
| --- | --- | --- | --- |
| conv1 | 96 filters 11x11, stride 4 | 96 x 55 x 55 | $96(3 \cdot 121 + 1) = 34{,}944$ |
| pool1 | 3x3, stride 2 | 96 x 27 x 27 | 0 |
| conv2 | 256 filters 5x5, pad 2, groups 2 | 256 x 27 x 27 | $256(48 \cdot 25 + 1) = 307{,}456$ |
| pool2 | 3x3, stride 2 | 256 x 13 x 13 | 0 |
| conv3 | 384 filters 3x3, pad 1 | 384 x 13 x 13 | $384(256 \cdot 9 + 1) = 885{,}120$ |
| conv4 | 384 filters 3x3, pad 1, groups 2 | 384 x 13 x 13 | $384(192 \cdot 9 + 1) = 663{,}936$ |
| conv5 | 256 filters 3x3, pad 1, groups 2 | 256 x 13 x 13 | $256(192 \cdot 9 + 1) = 442{,}624$ |
| pool5 | 3x3, stride 2 | 256 x 6 x 6 | 0 |
| fc6 | $9216 \to 4096$ | 4096 | $9216 \cdot 4096 + 4096 = 37{,}752{,}832$ |
| fc7 | $4096 \to 4096$ | 4096 | $16{,}781{,}312$ |
| fc8 | $4096 \to 1000$ | 1000 | $4{,}097{,}000$ |
| **Total** | | | **60,965,224** |

Look at where the numbers are: the five convolutions hold about 2.3M parameters (under 4% of the total) but perform nearly all of the arithmetic (about 666M of the roughly 725M MACs per image), while the three fully connected layers hold 96% of the parameters but cost under 10% of the compute. This imbalance, convolutions compute-heavy and parameter-light, fully connected layers the reverse, is the single most important structural fact about these networks and is the reason later architectures dropped the big FC layers in favor of global average pooling.

:::note Variants in libraries
`torchvision.models.alexnet` implements the single-stream "one weird trick" variant (Krizhevsky, 2014) with 64, 192, 384, 256, 256 filters and 61.1M parameters. Do not expect its parameter count to match the table above.
:::

### VGG (Simonyan and Zisserman, 2015) {#vgg}

The Oxford Visual Geometry Group asked a controlled question: holding everything else fixed, how does accuracy depend on depth? To make the comparison clean they used one building block throughout: 3 x 3 convolutions with stride 1 and padding 1, 2 x 2 max pooling with stride 2, and a fixed rule that the number of channels doubles after each pooling (64, 128, 256, 512, 512). Configurations A to E ranged from 11 to 19 weight layers; VGG-16 (D) and VGG-19 (E) became standard. In ILSVRC 2014 an ensemble of their models was second in classification with 7.3% top-5 error (behind GoogLeNet's 6.7%) and first in localization.

**Why 3 x 3?** Two stacked 3 x 3 layers have the same receptive field as one 5 x 5 layer ($r = 5$) and three have the field of a 7 x 7 ($r = 7$). With $C$ channels in and out, a single 7 x 7 layer has $49 C^2$ weights and the stack of three 3 x 3 layers has $27 C^2$, 45% fewer. And the stack applies three nonlinearities instead of one, so it represents more functions and (the authors argued) acts as a regularizer on the 7 x 7 filter by forcing it to decompose. Cost per position is smaller too. This argument essentially ended the use of large kernels in CNNs until the recent revival of 7 x 7 depthwise kernels in ConvNeXt.

| Block | Layers | Output shape | Parameters |
| --- | --- | --- | --- |
| 1 | conv3-64, conv3-64, pool | 64 x 112 x 112 | $1{,}792 + 36{,}928 = 38{,}720$ |
| 2 | conv3-128 x 2, pool | 128 x 56 x 56 | $73{,}856 + 147{,}584 = 221{,}440$ |
| 3 | conv3-256 x 3, pool | 256 x 28 x 28 | $295{,}168 + 2 \cdot 590{,}080 = 1{,}475{,}328$ |
| 4 | conv3-512 x 3, pool | 512 x 14 x 14 | $1{,}180{,}160 + 2 \cdot 2{,}359{,}808 = 5{,}899{,}776$ |
| 5 | conv3-512 x 3, pool | 512 x 7 x 7 | $3 \cdot 2{,}359{,}808 = 7{,}079{,}424$ |
| fc6 | $25088 \to 4096$ | 4096 | $102{,}764{,}544$ |
| fc7 | $4096 \to 4096$ | 4096 | $16{,}781{,}312$ |
| fc8 | $4096 \to 1000$ | 1000 | $4{,}097{,}000$ |
| **Total** | | | **138,357,544** |

The 13 convolutional layers have 14.7M parameters but cost 15.3 billion MACs per 224 x 224 image (the whole network is about 15.5 G MACs, roughly 21 times AlexNet), and fc6 alone, at 102.8M, holds 74% of all parameters. Because of the very different widths and resolutions, compute is spread over the five blocks without any one dominating (about 1.9, 2.8, 4.6, 4.6 and 1.4 G MACs); this "area times channels shrinks slowly" balance is visible in the figure above. Practical downsides: the 500+ MB of weights in float32, high memory use for activations at 224 x 224 x 64, and slow inference. VGG features remain popular for perceptual losses and style transfer ([Week 12](week-12.html)) because they are simple and transfer well.

```chart
{
  "type": "bar",
  "caption": "Parameters per weight layer of VGG-16 (millions), computed from the layer shapes in the table above. The first fully connected layer alone holds 74% of the network's parameters.",
  "height": 340,
  "data": {
    "labels": ["c1","c2","c3","c4","c5","c6","c7","c8","c9","c10","c11","c12","c13","fc6","fc7","fc8"],
    "datasets": [{
      "label": "Parameters (millions)",
      "data": [0.0018,0.0369,0.0739,0.1476,0.2952,0.5901,0.5901,1.1802,2.3598,2.3598,2.3598,2.3598,2.3598,102.7645,16.7813,4.097],
      "backgroundColor": ["var(--c-blue)","var(--c-blue)","var(--c-blue)","var(--c-blue)","var(--c-blue)","var(--c-blue)","var(--c-blue)","var(--c-blue)","var(--c-blue)","var(--c-blue)","var(--c-blue)","var(--c-blue)","var(--c-blue)","var(--c-purple)","var(--c-purple)","var(--c-purple)"]
    }]
  },
  "options": { "scales": { "x": {"title": {"display": true, "text": "layer (blue = conv, purple = fully connected)"}}, "y": {"type": "logarithmic", "title": {"display": true, "text": "parameters (millions, log scale)"}} } }
}
```

### The three networks in code {#classic-code}

The following code builds LeNet-5 and AlexNet-style (single stream) and VGG from a configuration list, and verifies the parameter counts in the tables with a dummy forward pass. It also shows a small trick that you will use constantly: register forward hooks to print the output shape of every layer.

```python:classic_nets.py
import torch
import torch.nn as nn

class LeNet5(nn.Module):
    """Modern LeNet-5: full C3 connectivity, tanh, average pooling. Input (N, 1, 32, 32)."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, 5), nn.Tanh(), nn.AvgPool2d(2),        # -> 6 x 14 x 14
            nn.Conv2d(6, 16, 5), nn.Tanh(), nn.AvgPool2d(2),       # -> 16 x 5 x 5
            nn.Conv2d(16, 120, 5), nn.Tanh(),                      # -> 120 x 1 x 1
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(120, 84), nn.Tanh(), nn.Linear(84, num_classes))
    def forward(self, x):
        return self.classifier(self.features(x))

def make_vgg(cfg, num_classes=1000):
    """cfg: list of ints (conv output channels) and 'M' (max-pool)."""
    layers, c_in = [], 3
    for v in cfg:
        if v == "M":
            layers.append(nn.MaxPool2d(2, 2))
        else:
            layers += [nn.Conv2d(c_in, v, 3, padding=1), nn.ReLU(inplace=True)]
            c_in = v
    classifier = nn.Sequential(
        nn.Flatten(), nn.Linear(512 * 7 * 7, 4096), nn.ReLU(True), nn.Dropout(0.5),
        nn.Linear(4096, 4096), nn.ReLU(True), nn.Dropout(0.5), nn.Linear(4096, num_classes))
    return nn.Sequential(nn.Sequential(*layers), classifier)

VGG16 = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"]

def count(m):
    return sum(p.numel() for p in m.parameters())

lenet = LeNet5()
print("LeNet-5 params:", count(lenet))                       # 61706
print(lenet(torch.randn(4, 1, 32, 32)).shape)                # (4, 10)

vgg = make_vgg(VGG16)
print("VGG-16 params:", count(vgg))                          # 138357544
print(vgg(torch.randn(1, 3, 224, 224)).shape)                # (1, 1000)

# Print the output shape of every conv/pool layer with forward hooks
def shape_hook(name):
    def hook(module, inp, out):
        print(f"{name:12s} {tuple(out.shape)}")
    return hook

for i, layer in enumerate(vgg[0]):
    if isinstance(layer, (nn.Conv2d, nn.MaxPool2d)):
        layer.register_forward_hook(shape_hook(f"{i:2d} {layer.__class__.__name__}"))
with torch.no_grad():
    vgg(torch.randn(1, 3, 224, 224))
```

### What each network contributed {#classic-lessons}

| Network | Year | Depth | Parameters | Lasting idea |
| --- | --- | --- | --- | --- |
| LeNet-5 | 1998 | 5 | 61.7K | conv, pool, FC pattern; end-to-end gradient training |
| AlexNet | 2012 | 8 | 61M | ReLU, dropout, GPUs, augmentation: scale makes deep CNNs work |
| VGG-16 | 2014 | 16 | 138M | depth via small 3x3 kernels; uniform, modular design |

The network after these (GoogLeNet, 2014: inception modules, 1x1 bottlenecks, about 7M parameters) and the one after that (ResNet, 2015: skip connections that allow hundreds of layers) are the subject of [Week 6](week-06.html).

## Data pipelines and augmentation {#data-pipelines}

A CNN is only as good as the data it is fed, and for image classification the way you feed it matters as much as the architecture. A good input pipeline does three jobs: it turns files into normalized tensors, it injects *label-preserving randomness* (augmentation) to regularize, and it does both fast enough that the GPU never waits.

{{figure: week05-pipeline | An image-classification input pipeline. Training uses random augmentation; evaluation uses a deterministic resize and center crop. Both use identical normalization.}}

### Preprocessing and normalization {#normalization-input}

Images arrive as `uint8` arrays of shape $H \times W \times 3$ with values in $[0, 255]$. PyTorch expects `float32` tensors of shape $3 \times H \times W$ (channels first). `transforms.ToTensor()` does the layout change and divides by 255. Then you **standardize each channel** with the dataset statistics, $x_c \leftarrow (x_c - \mu_c)/\sigma_c$. The reasons are those of [Week 3](week-03.html): inputs with zero mean and unit scale give well-conditioned first-layer gradients and make a single learning rate sensible for all layers. Typical constants are $\mu = (0.485, 0.456, 0.406)$, $\sigma = (0.229, 0.224, 0.225)$ for ImageNet and $\mu \approx (0.4914, 0.4822, 0.4465)$, $\sigma \approx (0.2470, 0.2435, 0.2616)$ for the CIFAR-10 training set. When you use a pretrained network, you *must* use the statistics it was trained with, whatever your own data looks like.

### Augmentation as a prior {#augmentation}

Augmentation encodes invariances you believe in: a cat remains a cat when flipped, shifted by a few pixels, slightly rotated, brighter or darker. Each augmented copy is a new, plausible training example, so the effective training set is much larger, which is a strong regularizer (see [Week 4](week-04.html)). Think of it as the cheapest way to inject the symmetry that the architecture itself lacks (CNNs are equivariant to translation only).

| Technique | What it does | Notes |
| --- | --- | --- |
| Random crop (with padding 4 on CIFAR; random-resized crop on ImageNet) | translation and scale jitter | The single most useful augmentation for small images |
| Horizontal flip | mirror | Wrong for digits, text, and anything with handedness |
| Color jitter | brightness, contrast, saturation, hue | Helps with illumination changes; use mild values |
| Random rotation, affine, perspective | geometry | Keep small; large rotations alter the label for some classes (6 vs 9) |
| Cutout / random erasing (DeVries and Taylor, 2017; Zhong et al., 2017) | masks a random square | Forces use of context; improves CIFAR error |
| Mixup (Zhang et al., 2018) | convex combination of two images and their labels | Smooths decision boundaries, improves calibration |
| CutMix (Yun et al., 2019) | pastes a patch of one image on another, mixes labels by area | Often better than Cutout and mixup on ImageNet |
| AutoAugment, RandAugment (Cubuk et al., 2019, 2020) | learned or randomly sampled policies of the above | Strong but need tuning of magnitude |

Three rules govern correct augmentation. (1) **Only augment training data.** Validation and test sets use a deterministic transform (resize, center crop, normalize) so that metrics are repeatable. (2) **Augment before normalization** (geometric and color operations on `[0,1]` images or PIL images, normalization last), except for operations that mix pixel values like mixup, which can run on batches of normalized tensors. (3) **Preserve the label**: every transform should be something the test distribution could plausibly contain.

:::warning Augmentation that leaks or lies
Do not compute normalization statistics on the validation or test sets (it leaks information, though mildly). Do not apply random flips to a task whose labels depend on left versus right (for example, reading digits or detecting "left turn" signs). And remember that heavy augmentation *increases* training loss: if you are underfitting, remove augmentation first.
:::

### The DataLoader in practice {#dataloader}

`torch.utils.data.Dataset` defines `__len__` and `__getitem__`; `DataLoader` wraps it with shuffling, batching and parallel workers (`num_workers`). Each worker process applies the transforms, so a different random augmentation is drawn at every epoch. Useful settings: `pin_memory=True` (faster host-to-GPU copies), `persistent_workers=True` (avoids restarting workers every epoch), `drop_last=True` for training when batch norm sees small last batches, and a fixed `generator` plus `worker_init_fn` for reproducibility.

```python:pipeline.py
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)

train_tf = T.Compose([
    T.RandomCrop(32, padding=4),          # translate by up to 4 pixels (zero padding)
    T.RandomHorizontalFlip(),             # mirror with p = 0.5
    T.ToTensor(),                         # uint8 HxWxC [0,255] -> float CxHxW [0,1]
    T.Normalize(MEAN, STD),               # per-channel standardization
])
test_tf = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])   # no randomness

train_set = torchvision.datasets.CIFAR10("data", train=True, download=True, transform=train_tf)
test_set = torchvision.datasets.CIFAR10("data", train=False, download=True, transform=test_tf)
train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2,
                          pin_memory=True, drop_last=True)
test_loader = DataLoader(test_set, batch_size=512, shuffle=False, num_workers=2)

xb, yb = next(iter(train_loader))
print(xb.shape, xb.dtype, yb.shape)                  # (128, 3, 32, 32) float32, (128,)
print("batch mean per channel:", xb.mean(dim=(0, 2, 3)))   # near 0
print("batch std  per channel:", xb.std(dim=(0, 2, 3)))    # near 1

# Mixup on a batch: x~ = lam x_i + (1-lam) x_j,  loss = lam CE(f(x~), y_i) + (1-lam) CE(f(x~), y_j)
def mixup(x, y, alpha=0.2):
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    perm = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[perm], y, y[perm], lam
```

Mixup is a one-line change to the loss: compute `lam * F.cross_entropy(out, y_a) + (1 - lam) * F.cross_entropy(out, y_b)`.

## Benchmarks: ImageNet and CIFAR {#benchmarks}

Two datasets frame most CNN research. Knowing their scale and the numbers on them lets you judge claims.

**ImageNet / ILSVRC.** ImageNet (Deng et al., 2009) is a hierarchy of millions of labeled images organized by WordNet nouns. The ILSVRC classification task uses a subset: 1,000 classes, about 1.28 million training images (between 732 and 1,300 per class), 50,000 validation images and 100,000 test images, held out by the organizers (Russakovsky et al., 2015). The standard metrics are **top-1 error** (the highest-scoring class must be right) and **top-5 error** (the true class must be among the five highest). Images are variable-size photos; the usual training recipe takes 224 x 224 random-resized crops, and evaluation resizes to 256 and center-crops 224. The challenge ran from 2010 to 2017.

**CIFAR-10 and CIFAR-100.** CIFAR-10 (Krizhevsky, 2009) has 60,000 color images of size 32 x 32 in 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck), 6,000 per class, split into 50,000 training and 10,000 test images. CIFAR-100 has the same images size but 100 classes with 600 images each. They are small enough to train a competitive model in minutes on a free GPU, which makes them the standard testbed for ideas, but they are also small enough to be easy to overfit by hyperparameter search against the test set: always hold out part of the training set for tuning.

### The ILSVRC story in numbers {#ilsvrc-history}

```chart
{
  "type": "bar",
  "caption": "Top-5 classification error of the winning ILSVRC entry each year (test set). Values from the ILSVRC reports (Russakovsky et al., 2015, IJCV) and the challenge results for 2015 to 2017: 28.2, 25.8, 15.3 (AlexNet), 11.7 (Clarifai), 6.7 (GoogLeNet), 3.57 (ResNet), 2.99, 2.25 (SENet). The dashed line is the 5.1% error that Russakovsky et al. measured for one trained human annotator.",
  "height": 360,
  "data": {
    "labels": ["2010","2011","2012","2013","2014","2015","2016","2017"],
    "datasets": [
      {"type": "bar", "label": "Winner top-5 error (%)", "data": [28.2, 25.8, 15.3, 11.7, 6.7, 3.57, 2.99, 2.25], "backgroundColor": "var(--c-blue)"},
      {"type": "line", "label": "Human (5.1%)", "data": [5.1,5.1,5.1,5.1,5.1,5.1,5.1,5.1], "borderColor": "var(--c-red)", "borderDash": [6,4], "pointRadius": 0}
    ]
  },
  "options": { "scales": { "x": {"title": {"display": true, "text": "ILSVRC year"}}, "y": {"title": {"display": true, "text": "top-5 error (%)"}, "min": 0} } }
}
```

Three eras are visible. Until 2011 the winners used hand-crafted features (dense SIFT with Fisher vectors or similar) and improved error by a few points per year. In 2012 AlexNet cut the error by about ten points at once. After that, deeper and better-regularized CNNs improved it by a roughly constant factor each year until the challenge ended, passing the human-level estimate in 2015. That estimate has caveats (one annotator, an expert trained on the 1,000 classes, difficult fine-grained dog breeds), so do not read it as "machines understand images better than people".

### Accuracy versus size {#accuracy-vs-size}

Bigger is not always better: architecture matters more than parameter count.

```chart
{
  "type": "scatter",
  "caption": "ImageNet single-crop top-1 accuracy versus parameter count for classic CNNs, using the pretrained torchvision models and the numbers listed in the torchvision documentation (v1 weights where several exist). Parameters in millions.",
  "height": 400,
  "data": {
    "datasets": [
      {"label": "AlexNet", "data": [{"x": 61.1, "y": 56.52}], "backgroundColor": "var(--c-red)", "pointRadius": 7},
      {"label": "VGG-11", "data": [{"x": 132.9, "y": 69.02}], "backgroundColor": "var(--c-purple)", "pointRadius": 7},
      {"label": "VGG-16", "data": [{"x": 138.4, "y": 71.59}], "backgroundColor": "var(--c-purple)", "pointRadius": 7},
      {"label": "GoogLeNet", "data": [{"x": 6.6, "y": 69.78}], "backgroundColor": "var(--c-teal)", "pointRadius": 7},
      {"label": "ResNet-18", "data": [{"x": 11.7, "y": 69.76}], "backgroundColor": "var(--c-blue)", "pointRadius": 7},
      {"label": "ResNet-50", "data": [{"x": 25.6, "y": 76.13}], "backgroundColor": "var(--c-blue)", "pointRadius": 7},
      {"label": "ResNet-152", "data": [{"x": 60.2, "y": 78.31}], "backgroundColor": "var(--c-blue)", "pointRadius": 7},
      {"label": "DenseNet-121", "data": [{"x": 8.0, "y": 74.43}], "backgroundColor": "var(--c-green)", "pointRadius": 7},
      {"label": "MobileNetV2", "data": [{"x": 3.5, "y": 71.88}], "backgroundColor": "var(--c-amber)", "pointRadius": 7}
    ]
  },
  "options": { "scales": { "x": {"title": {"display": true, "text": "parameters (millions)"}, "min": 0}, "y": {"title": {"display": true, "text": "top-1 accuracy (%)"}, "min": 50} } }
}
```

VGG-16 uses 138M parameters to reach 71.6%; MobileNetV2 reaches 71.9% with 3.5M, 40 times fewer; DenseNet-121 gets 74.4% with 8M. Parameters are a poor proxy for quality because, as we saw, VGG's parameters sit in the classifier head, and because architectural ideas (bottlenecks, skip connections, separable convolutions, global pooling) buy accuracy without parameters. Compute (MACs) and latency are the other axes; we return to this trade-off in [Week 16](week-16.html).

### CIFAR-10: depth versus error {#cifar-depth}

On CIFAR-10 the original ResNet paper (He et al., 2016) reports test error for a family of 6n+2-layer networks trained with the same recipe (random crop with 4-pixel padding, horizontal flips). It is a good illustration of how depth helps until optimization or overfitting takes over.

```chart
{
  "type": "line",
  "caption": "CIFAR-10 test error (%) of ResNets of different depth, from Table 6 of He et al. (2016), best of runs. Parameter counts: 0.27M, 0.46M, 0.66M, 0.85M, 1.7M, 19.4M. The 1202-layer network is worse than the 110-layer one, which the authors attribute to overfitting.",
  "height": 340,
  "data": {
    "labels": ["20","32","44","56","110","1202"],
    "datasets": [{"label": "Test error (%)", "data": [8.75, 7.51, 7.17, 6.97, 6.43, 7.93], "borderColor": "var(--c-blue)", "backgroundColor": "var(--c-blue)", "tension": 0.15}]
  },
  "options": { "scales": { "x": {"title": {"display": true, "text": "number of layers"}}, "y": {"title": {"display": true, "text": "CIFAR-10 test error (%)"}, "min": 5} } }
}
```

For orientation, a plain eight-layer CNN with augmentation and batch norm, like the one in the lab below, typically reaches roughly 8 to 10% error in about 30 epochs, while carefully tuned ResNet-style training pipelines reach around 5% or below. Treat these as ballparks and measure your own.

## Lab: a CIFAR-10 classifier with controlled ablations {#lab}

This lab builds a small VGG-style network with batch normalization, trains it on CIFAR-10 with augmentation, evaluates it properly, and then runs three controlled ablations: depth, kernel size and batch normalization. It runs on a free Colab GPU; each training run of 30 epochs takes a few minutes. If you only have a CPU, set `EPOCHS = 3` and `train_subset = 10_000` to check that the code works.

The structure follows the pipeline from [the data section](#data-pipelines): a deterministic train/validation split, augmentation on the training set only, the training loop from [Week 4](week-04.html#lab), and a final one-time look at the test set.

### Step 1: data with a proper validation split {#lab-data}

The 10,000 official test images are reserved for the very end. We carve 5,000 validation images out of the 50,000 training images for choosing hyperparameters and comparing ablations. Because the training subset needs augmentation and the validation subset does not, we create two dataset objects over the same files and select disjoint index sets from each.

```python:lab_data.py
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as T

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0); np.random.seed(0)

# Per-channel statistics of the CIFAR-10 training set (computed once; widely published values)
MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)

train_tf = T.Compose([
    T.RandomCrop(32, padding=4),          # shift by up to 4 px (zero-padded border)
    T.RandomHorizontalFlip(),             # label-preserving for these 10 classes
    T.ToTensor(),                         # uint8 HWC [0,255] -> float CHW [0,1]
    T.Normalize(MEAN, STD),
])
eval_tf = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])   # NO augmentation here

root = "./data"
full_train_aug = torchvision.datasets.CIFAR10(root, train=True, download=True, transform=train_tf)
full_train_clean = torchvision.datasets.CIFAR10(root, train=True, download=True, transform=eval_tf)
test_set = torchvision.datasets.CIFAR10(root, train=False, download=True, transform=eval_tf)

# Deterministic 45k / 5k split
perm = np.random.RandomState(123).permutation(len(full_train_aug))
train_idx, val_idx = perm[:45_000], perm[45_000:]
train_set = Subset(full_train_aug, train_idx)     # augmented view
val_set = Subset(full_train_clean, val_idx)       # clean view of the held-out indices

def make_loaders(batch_size=128, workers=2):
    kw = dict(num_workers=workers, pin_memory=(device == "cuda"), persistent_workers=workers > 0)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True, **kw)
    val_loader = DataLoader(val_set, batch_size=512, shuffle=False, **kw)
    test_loader = DataLoader(test_set, batch_size=512, shuffle=False, **kw)
    return train_loader, val_loader, test_loader

train_loader, val_loader, test_loader = make_loaders()
xb, yb = next(iter(train_loader))
print(xb.shape, xb.dtype, yb.shape)                       # torch.Size([128, 3, 32, 32]) torch.float32 torch.Size([128])
print("batch mean/std per channel:", xb.mean((0, 2, 3)), xb.std((0, 2, 3)))   # roughly 0 and 1
classes = train_set.dataset.classes
print(classes)
```

Check the printed per-channel mean and standard deviation: they should be near 0 and 1. If they are not, your normalization is wrong (a very common silent bug).

### Step 2: a configurable model {#lab-model}

The network is a stack of "stages". Each stage has `n_convs` convolutions (kernel size `k`, optional batch norm, ReLU) followed by 2x2 max pooling. After the last stage, global average pooling and one linear layer produce the 10 logits. With channels 32, 64, 128 and three stages the default model has 288,746 parameters. The arguments let the ablations change depth, kernel size and normalization while keeping everything else identical.

```python:lab_model.py
import torch
import torch.nn as nn

class SmallVGG(nn.Module):
    def __init__(self, widths=(32, 64, 128), n_convs=2, k=3, use_bn=True, num_classes=10):
        super().__init__()
        layers, c_in = [], 3
        for c_out in widths:
            for _ in range(n_convs):
                # With BN directly after the conv, the conv bias is redundant (BN subtracts the mean)
                layers.append(nn.Conv2d(c_in, c_out, k, padding=k // 2, bias=not use_bn))
                if use_bn:
                    layers.append(nn.BatchNorm2d(c_out))
                layers.append(nn.ReLU(inplace=True))
                c_in = c_out
            layers.append(nn.MaxPool2d(2))                   # halve the resolution after each stage
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)                  # global average pooling
        self.fc = nn.Linear(c_in, num_classes)

        # He initialization for ReLU networks (see Week 3); PyTorch's default is a variant of it
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)               # (B, 128, 4, 4) for the default model
        x = self.pool(x).flatten(1)        # (B, 128)
        return self.fc(x)                  # (B, 10)

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

model = SmallVGG()
print(model)
print("parameters:", count_params(model))                  # 288746
x = torch.randn(2, 3, 32, 32)
h = x
for layer in model.features:
    h = layer(h)
    if isinstance(layer, nn.MaxPool2d):
        print(type(layer).__name__, tuple(h.shape))        # (2,32,16,16), (2,64,8,8), (2,128,4,4)
```

Notice that the parameter count is dominated by the last stage: its two 3x3 convolutions with 128 channels hold $9\cdot128\cdot128 = 147{,}456$ weights each in the second one. Doubling channels when halving resolution keeps the compute per stage roughly constant (the same rule as VGG).

### Step 3: training and evaluation loops {#lab-train}

The training function uses SGD with Nesterov momentum, decoupled-style weight decay applied only to weights (not to biases or BN parameters), a one-cycle learning-rate schedule (see [Week 3](week-03.html#lr-schedules)), mixed precision on GPU, and gradient clipping as cheap insurance. It returns a history dictionary so we can plot curves and compare runs.

```python:lab_train.py
import time, copy
import torch
import torch.nn as nn
import torch.nn.functional as F

def param_groups(model, wd):
    """Apply weight decay only to conv/linear weights (dim >= 2), not to biases or BN scale/shift."""
    decay, no_decay = [], []
    for p in model.parameters():
        (decay if p.ndim >= 2 else no_decay).append(p)
    return [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}]

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()                                            # BN uses running statistics, no dropout
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        loss_sum += F.cross_entropy(logits, y, reduction="sum").item()
        correct += (logits.argmax(1) == y).sum().item()
        total += y.numel()
    return loss_sum / total, correct / total

def fit(model, train_loader, val_loader, epochs=30, lr=0.1, wd=5e-4, device="cuda", label_smoothing=0.0, verbose=True):
    model = model.to(device)
    opt = torch.optim.SGD(param_groups(model, wd), lr=lr, momentum=0.9, nesterov=True)
    steps = epochs * len(train_loader)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.2)
    use_amp = device == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    hist = {"train_loss": [], "val_loss": [], "val_acc": [], "lr": []}
    best_acc, best_state = 0.0, None
    for epoch in range(epochs):
        model.train()
        t0, run_loss, n = time.time(), 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                loss = F.cross_entropy(model(x), y, label_smoothing=label_smoothing)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)                            # so clipping sees true gradient magnitudes
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update()
            sched.step()                                    # OneCycle steps per batch, not per epoch
            run_loss += loss.item() * y.numel(); n += y.numel()
        v_loss, v_acc = evaluate(model, val_loader, device)
        hist["train_loss"].append(run_loss / n); hist["val_loss"].append(v_loss)
        hist["val_acc"].append(v_acc); hist["lr"].append(opt.param_groups[0]["lr"])
        if v_acc > best_acc:
            best_acc, best_state = v_acc, copy.deepcopy(model.state_dict())
        if verbose:
            print(f"epoch {epoch+1:3d}/{epochs}  train loss {run_loss/n:.3f}  val loss {v_loss:.3f}  "
                  f"val acc {100*v_acc:.2f}%  lr {hist['lr'][-1]:.4f}  ({time.time()-t0:.1f}s)")
    model.load_state_dict(best_state)                       # return the best-on-validation weights
    hist["best_val_acc"] = best_acc
    return model, hist
```

Before spending minutes on a full run, always do the two sanity checks from [Week 4's debugging section](week-04.html#debugging): the initial loss and overfitting a single batch.

```python:lab_sanity.py
from lab_data import *
from lab_model import *
from lab_train import *
import math

model = SmallVGG().to(device)
x, y = next(iter(train_loader)); x, y = x.to(device), y.to(device)

# 1) Initial loss should be close to ln(10) = 2.303 for a 10-class problem
with torch.no_grad():
    model.train()
    print("initial loss:", F.cross_entropy(model(x), y).item(), " expected ~", math.log(10))

# 2) The model must be able to memorize 128 images (tests the whole forward/backward/update path)
opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
for step in range(100):
    opt.zero_grad(); loss = F.cross_entropy(model(x), y); loss.backward(); opt.step()
    if step % 20 == 0:
        print(step, round(loss.item(), 4))
print("final single-batch loss:", loss.item())              # should be near 0
```

If the initial loss is far from 2.30, suspect the last-layer initialization or a missing normalization. If the single batch cannot be memorized, there is a bug (wrong labels, detached graph, missing `zero_grad`, or a learning rate that is far too small or large).

### Step 4: the baseline run {#lab-baseline}

```python:lab_baseline.py
from lab_data import *
from lab_model import *
from lab_train import *

EPOCHS = 30
model = SmallVGG()
model, hist = fit(model, train_loader, val_loader, epochs=EPOCHS, lr=0.1, wd=5e-4, device=device)

test_loss, test_acc = evaluate(model, test_loader, device)  # touch the test set exactly once
print(f"best val acc {100*hist['best_val_acc']:.2f}%   test acc {100*test_acc:.2f}%")
```

With this recipe a 288k-parameter network typically lands in the high 80s to low 90s percent accuracy on the validation split (roughly 8 to 12% error); exact numbers vary by seed, GPU and library version by a few tenths of a percent. The training loss is above the validation loss for most of training because augmentation makes the training batches harder than clean validation images, and because the training loss is averaged over an epoch while the weights are changing.

### Step 5: error analysis {#lab-errors}

Accuracy hides which classes fail. A confusion matrix tells you what the network confuses, which is often a clue about the data (cats versus dogs) and not the model.

```python:lab_confusion.py
import numpy as np
import matplotlib.pyplot as plt

@torch.no_grad()
def predictions(model, loader, device):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        ps.append(model(x.to(device)).argmax(1).cpu()); ys.append(y)
    return torch.cat(ys).numpy(), torch.cat(ps).numpy()

y_true, y_pred = predictions(model, test_loader, device)
cm = np.zeros((10, 10), dtype=int)
np.add.at(cm, (y_true, y_pred), 1)                          # rows: true class, columns: predicted class
per_class = cm.diagonal() / cm.sum(1)
for name, acc in sorted(zip(classes, per_class), key=lambda t: t[1]):
    print(f"{name:12s} {100*acc:5.1f}%")

off = cm.copy(); np.fill_diagonal(off, 0)
i, j = np.unravel_index(off.argmax(), off.shape)
print(f"most common confusion: true {classes[i]} predicted as {classes[j]} ({off[i, j]} times)")

plt.figure(figsize=(6, 5)); plt.imshow(cm, cmap="Blues")
plt.xticks(range(10), classes, rotation=90); plt.yticks(range(10), classes)
plt.xlabel("predicted"); plt.ylabel("true"); plt.colorbar(); plt.tight_layout(); plt.show()
```

Expect cat, bird and dog to be the weakest classes and cat-to-dog (and dog-to-cat) to be the most frequent single confusion, while automobile, ship and truck are the easiest. A network that looks at 32 x 32 images with a handful of pixels for the animal's face simply has less evidence to separate cat from dog.

### Step 6: ablations on depth, kernel size and batch norm {#lab-ablations}

A good ablation changes **one factor at a time**, keeps everything else (data, schedule, epochs, seed) identical, and reports the spread across seeds. The function below runs a grid of configurations over several seeds and reports mean and standard deviation of the best validation accuracy. We do not touch the test set here.

```python:lab_ablations.py
import json
import numpy as np
from lab_data import *
from lab_model import *
from lab_train import *

def run(config, seeds=(0, 1, 2), epochs=20):
    accs, params = [], None
    for s in seeds:
        torch.manual_seed(s); np.random.seed(s)
        model = SmallVGG(**config)
        params = count_params(model)
        _, hist = fit(model, train_loader, val_loader, epochs=epochs, lr=0.1, device=device, verbose=False)
        accs.append(100 * hist["best_val_acc"])
    return {"config": config, "params": params, "mean": float(np.mean(accs)), "std": float(np.std(accs)), "accs": accs}

experiments = {
    # Ablation A: depth (convs per stage), the 3 stages and widths stay fixed
    "depth": [dict(n_convs=n) for n in (1, 2, 3)],
    # Ablation B: kernel size, same depth and widths (parameters grow with k^2)
    "kernel": [dict(k=k) for k in (1, 3, 5, 7)],
    # Ablation C: batch normalization on/off. Without BN, lower the LR or training diverges.
    "batchnorm": [dict(use_bn=True), dict(use_bn=False)],
}

results = {}
for name, configs in experiments.items():
    results[name] = [run(c) for c in configs]
    for r in results[name]:
        print(f"{name:10s} {str(r['config']):28s} params {r['params']:>9,d}  "
              f"val acc {r['mean']:.2f} +- {r['std']:.2f}")

with open("ablations.json", "w") as f:
    json.dump(results, f, indent=1)
```

Parameter counts for the grid, which you can verify with `count_params` (and with the formulas of [the separable-convolution section](#depthwise-separable-convolution)): the baseline has 288,746; `k=1` has 34,026 (no spatial context at all, only the pooling mixes locations); `k=5` has 798,186; `k=7` has 1,562,346.

:::note Batch norm off needs a smaller learning rate
With `use_bn=False` the default `lr=0.1` often diverges or stalls (the loss becomes `nan` or stays at 2.30). Re-run that configuration with `lr=0.01` and, if needed, a few epochs of warmup. The comparison is then "best learning rate with BN versus best learning rate without BN", which is the fair comparison: BN's main practical benefit is that it makes much larger learning rates usable (see [Week 3](week-03.html#normalization)).
:::

#### What to expect and how to read it {#lab-readout}

These are typical qualitative outcomes; your numbers will differ.

- **Depth.** Going from 1 to 2 convolutions per stage usually gives a clear gain (a couple of percentage points), and 2 to 3 a smaller one, with diminishing returns. A plain stack without skip connections gets harder to optimize as it deepens, which is the motivation for residual networks in [Week 6](week-06.html).
- **Kernel size.** The $1\times1$ network is far worse because it cannot combine neighboring pixels. $3\times3$ is close to the best accuracy per parameter. $5\times5$ and $7\times7$ cost 2.8x and 5.4x the parameters of $3\times3$ for little or no gain, which is why VGG's argument (stack small kernels) won.
- **Batch norm.** BN typically improves both final accuracy and training speed and makes the result less sensitive to the learning rate.
- **Noise.** Differences below roughly two standard deviations across seeds are not evidence of anything. With 3 seeds, report the spread and be modest.

:::tip Make your ablation plot
Plot mean validation accuracy against parameter count for all configurations on one scatter plot (error bars equal to the standard deviation). The cheapest configuration that is within noise of the best is the one you would deploy.
:::

### Step 7: visualize what the first layer learned {#lab-filters}

The first-layer filters of a trained CNN are directly interpretable because they act on pixels. They usually look like oriented edge and color-opponent detectors, as in AlexNet's first layer.

```python:lab_filters.py
import matplotlib.pyplot as plt

w = model.features[0].weight.detach().cpu()                 # (32, 3, 3, 3): 32 filters over RGB, 3x3
w = (w - w.amin((1, 2, 3), keepdim=True)) / (w.amax((1, 2, 3), keepdim=True) - w.amin((1, 2, 3), keepdim=True))
fig, axes = plt.subplots(4, 8, figsize=(8, 4))
for ax, f in zip(axes.ravel(), w):
    ax.imshow(f.permute(1, 2, 0).numpy(), interpolation="nearest"); ax.axis("off")
plt.suptitle("First-layer 3x3 filters (each rescaled to [0,1])"); plt.show()
```

With only 3 x 3 filters the patterns are coarse, but you should still see edge-like and color-blob filters. Try the same with `k=7` for a more AlexNet-like picture.

## Common mistakes and debugging tips {#common-mistakes}

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| Validation accuracy far below training accuracy right away, but good in `train()` mode | Forgot `model.eval()` so BN uses batch statistics, or evaluating with a tiny batch | Call `model.eval()`; evaluate with batch size at least 64; check `model.training` |
| Val accuracy ~10% (chance) while train accuracy is fine | Different normalization at train and test time, or labels shuffled in the val loader | Reuse the same `Normalize`; verify one val batch visually and by statistics |
| Loss `nan` after a few steps | Learning rate too high (especially without BN), a missing normalization of inputs, or fp16 overflow | Lower LR, add warmup and gradient clipping, check `xb.min()/max()` |
| Loss stuck at 2.30 | Dead network: huge LR, wrong init, `ReLU` after the final layer, or labels not matching inputs | Overfit one batch; remove activation on logits; lower LR |
| Training accuracy is 100% but validation is much lower | Overfitting (no augmentation, too large model) | Add crop and flip, weight decay, label smoothing, or more data |
| Augmentation applied to the validation set | One `transform` shared by all splits | Separate `train_tf` and `eval_tf`; apply random ops to train only |
| Shape error `mat1 and mat2 shapes cannot be multiplied` | Hard-coded `Linear(in_features)` after the conv stack when the input size changed | Use global average pooling or print the shape after `features` |
| `RuntimeError: ... expected ... Float but found Byte` or all-black images | `ToTensor()` omitted or applied twice | Check `dtype` and range of one batch |
| Slow training, GPU idle | Data loading bottleneck | `num_workers` 2 to 8, `pin_memory=True`, `persistent_workers=True`; profile with a loop that only iterates the loader |
| Results differ between runs of the same code | Random seeds, non-deterministic CUDA kernels (cuDNN autotuning) | Seed everything, and accept small residual variation; report averages over seeds |
| Worse accuracy after adding `padding=0` everywhere | Feature maps shrink and border information is lost | Use `padding=k//2` to preserve size when you stack many convolutions |
| Flip augmentation hurts a text or digit task | Not label-preserving there (6 versus 9 under rotation, mirrored text) | Choose augmentations that fit the invariances of the task |

Three habits catch most CNN bugs: print the shape of every stage once, overfit a single batch, and view a few augmented training images with their labels before training.

## Exercises {#exercises}

:::exercise Exercise 1: output sizes (computation)
A network begins with `Conv2d(3, 64, kernel_size=7, stride=2, padding=3)`, then `MaxPool2d(kernel_size=3, stride=2, padding=1)`, then `Conv2d(64, 64, 3, padding=1, dilation=2)` on a $224\times224$ image. (a) Give the output spatial size after each layer. (b) What padding keeps the size unchanged for the dilated conv, and what is its effective kernel size? (c) Why does a stride-2, kernel-2 max pool on a $7\times7$ map lose one row and column, and how does `ceil_mode=True` change that?

:::details Solution
Use $\lfloor (n + 2p - d(k-1) - 1)/s\rfloor + 1$.

(a) Conv 7x7: $\lfloor(224+6-6-1)/2\rfloor+1=\lfloor 223/2\rfloor+1=112$. Pool 3x3 stride 2 padding 1: $\lfloor(112+2-2-1)/2\rfloor+1=\lfloor111/2\rfloor+1=56$. Dilated 3x3 with $d=2$, $p=1$: $k_\text{eff}=5$, so $\lfloor(56+2-4-1)/1\rfloor+1=54$. Final: $64\times54\times54$.

(b) For the size to be unchanged with stride 1 you need $2p=k_\text{eff}-1=4$, so $p=2$ (in general $p=d(k-1)/2$). Effective kernel size is 5 with only 9 non-zero taps.

(c) $\lfloor(7-2)/2\rfloor+1=3$: the last row and column are never covered, since windows start at 0, 2, 4 and the one starting at 6 would need index 7. `ceil_mode=True` uses $\lceil(7-2)/2\rceil+1=4$, allowing a partial window at the border.
:::
:::

:::exercise Exercise 2: parameters and MACs (computation)
A feature map is $56\times56\times64$ and you apply a conv with 128 output channels. Count the parameters (ignore the bias) and MACs for (a) a standard $3\times3$ conv, (b) a $1\times1$ conv, (c) a depthwise-separable $3\times3$ (depthwise then pointwise), (d) a grouped $3\times3$ conv with 4 groups. What is the ratio (c)/(a) and how does it compare with the formula $1/C_\text{out}+1/k^2$?

:::details Solution
(a) Params $9\cdot64\cdot128=73{,}728$; MACs $=\text{params}\times56^2=73{,}728\cdot3136=231{,}211{,}008$.

(b) Params $64\cdot128=8{,}192$; MACs $25{,}690{,}112$.

(c) Depthwise $9\cdot64=576$ plus pointwise $8{,}192$ gives $8{,}768$ parameters; MACs $8{,}768\cdot3136=27{,}496{,}448$.

(d) Each group maps $16$ input channels to $32$ output channels: $4\cdot9\cdot16\cdot32=18{,}432$ parameters, a quarter of (a); MACs $57{,}802{,}752$.

Ratio (c)/(a) $=8768/73728=0.1189$. The formula gives $1/128+1/9=0.00781+0.1111=0.1189$. Exactly equal here because both are the same algebra: $(k^2C_\text{in}+C_\text{in}C_\text{out})/(k^2C_\text{in}C_\text{out})$.
:::
:::

:::exercise Exercise 3: receptive field (computation)
Compute the receptive field size and jump after each layer of: conv 3x3 s1, conv 3x3 s2, conv 3x3 s1, conv 3x3 s1. Then compare against four 3x3 stride-1 convs. What is the receptive field of a stack of $n$ stride-1 $3\times3$ convs, and how many such layers replace one $7\times7$ conv with the same receptive field?

:::details Solution
Recurrence $r_l=r_{l-1}+(k-1)j_{l-1}$, $j_l=j_{l-1}s_l$, starting at $r_0=1$, $j_0=1$.

| layer | r | j |
| --- | --- | --- |
| conv 3x3 s1 | 3 | 1 |
| conv 3x3 s2 | 5 | 2 |
| conv 3x3 s1 | 9 | 2 |
| conv 3x3 s1 | 13 | 2 |

Four stride-1 convs give $r=9$. So placing one stride-2 early almost doubles the growth rate of later layers (each later layer adds $2j=4$ instead of 2). A stack of $n$ stride-1 3x3 convs has $r=2n+1$, so three layers equal one 7x7 with $27C^2$ parameters instead of $49C^2$ (for $C$ channels in and out), 45% fewer, plus two extra nonlinearities.
:::
:::

:::exercise Exercise 4: gradient of a 1D cross-correlation (derivation)
Let $y_i=\sum_{m=0}^{k-1}w_m x_{i+m}+b$ for $i=0,\dots,n-k$ and let $\delta_i=\partial\mathcal{L}/\partial y_i$. Derive $\partial\mathcal{L}/\partial w_m$, $\partial\mathcal{L}/\partial b$ and $\partial\mathcal{L}/\partial x_j$. Then say which operation computes the last one.

:::details Solution
By the chain rule, $\partial\mathcal L/\partial w_m=\sum_i\delta_i\,\partial y_i/\partial w_m=\sum_i\delta_i x_{i+m}$: the kernel gradient is the cross-correlation of the input with the upstream gradient. $\partial\mathcal L/\partial b=\sum_i\delta_i$. For the input, $x_j$ appears in $y_i$ whenever $i+m=j$, i.e. $m=j-i\in[0,k-1]$, so $\partial\mathcal L/\partial x_j=\sum_{m}\delta_{j-m}w_m$ over valid indices. This is a full convolution of $\delta$ (zero-padded) with $w$, equivalently a cross-correlation of the zero-padded $\delta$ with the flipped kernel. It is the transposed convolution of [the preview section](#backprop-transposed).
:::
:::

:::exercise Exercise 5: verify your derivation with autograd (coding)
Using `torch.nn.functional.conv2d` on random tensors ($x$: $2\times3\times8\times8$, $w$: $4\times3\times3\times3$, stride 2, padding 1), compute the gradients with autograd, then recompute the input gradient with `torch.nn.functional.conv_transpose2d` and the weight gradient with `torch.nn.grad.conv2d_weight`. Check that they match.

:::details Solution
```python
import torch
import torch.nn.functional as F

x = torch.randn(2, 3, 8, 8, requires_grad=True)
w = torch.randn(4, 3, 3, 3, requires_grad=True)
y = F.conv2d(x, w, stride=2, padding=1)                    # (2, 4, 4, 4)
g = torch.randn_like(y)                                    # pretend upstream gradient dL/dy
y.backward(g)

# Input gradient = transposed convolution of g with the same weight.
# output_padding resolves the ambiguity from the stride: 8 = (4-1)*2 - 2 + 3 + 1
dx = F.conv_transpose2d(g, w, stride=2, padding=1, output_padding=1)
dw = torch.nn.grad.conv2d_weight(x.detach(), w.shape, g, stride=2, padding=1)
print(torch.allclose(x.grad, dx, atol=1e-5), torch.allclose(w.grad, dw, atol=1e-4))   # True True
```
The need for `output_padding=1` shows that several input sizes map to the same output size under stride 2; the transposed convolution must be told which one you want.
:::
:::

:::exercise Exercise 6: why the conv bias is redundant before BN (conceptual)
In `SmallVGG` we set `bias=False` for convs followed by BN. Show that a bias term has no effect on the output. Does the same hold if the BN is placed after the ReLU?

:::details Solution
BN subtracts the per-channel mean over the batch and spatial positions. Adding a constant $b_c$ to every position of channel $c$ shifts the mean by $b_c$ and leaves $x-\mu$ unchanged, so the normalized output is identical (the variance is also unchanged). The gradient with respect to $b_c$ is exactly zero (up to numerics). BN's own shift parameter $\beta$ takes over the role of a bias. If BN comes after the ReLU, the conv output is first passed through a nonlinearity, so the bias changes which inputs are clipped; it is no longer redundant (though it is still often omitted in practice).
:::
:::

:::exercise Exercise 7: flatten versus global average pooling (what would happen if...)
Your final feature map is $128\times4\times4$ and the classifier is `Flatten` + `Linear(2048, 10)`. (a) How many parameters does this head have compared with GAP + `Linear(128, 10)`? (b) You now feed $64\times64$ images. What happens in each design? (c) A cat appears at the left edge in one image and the right in another. Which head treats this more consistently, and why?

:::details Solution
(a) Flatten head: $2048\cdot10+10=20{,}490$. GAP head: $128\cdot10+10=1{,}290$, sixteen times fewer.

(b) The feature map becomes $128\times8\times8$. The flatten head raises a shape error (it expects 2048 inputs) and would need rebuilding; the GAP head works unchanged since it averages over any spatial size, producing 128 features.

(c) GAP discards position, so a translated object produces nearly the same pooled vector (exactly so for a circular, fully equivariant stack). The flatten head learns separate weights per position, so each location must be learned separately and the head is not translation invariant. GAP trades off position information, which is a problem only when "where" matters.
:::
:::

:::exercise Exercise 8: parameter and MAC budget of the lab network (computation, challenging)
For `SmallVGG(widths=(32,64,128), n_convs=2, k=3, use_bn=True)` on $32\times32$ inputs, derive the parameter count 288,746 and count the MACs of the conv layers (ignore BN, ReLU and pooling). Which stage dominates parameters, and which dominates compute? What happens to both if the input resolution is doubled?

:::details Solution
Conv weights without bias: $9(3\cdot32+32\cdot32+32\cdot64+64\cdot64+64\cdot128+128\cdot128)=9\,(96+1024+2048+4096+8192+16384)=9\cdot31{,}840=286{,}560$. BN: $2\cdot(32+32+64+64+128+128)=896$. Linear: $128\cdot10+10=1{,}290$. Total $286{,}560+896+1{,}290=288{,}746$.

MACs: stage 1 at $32^2$: $9(96+1024)\cdot1024=10{,}321{,}920$. Stage 2 at $16^2$: $9(2048+4096)\cdot256=14{,}155{,}776$. Stage 3 at $8^2$: $9(8192+16384)\cdot64=14{,}155{,}776$. Plus the linear layer's 1,280: total $38{,}634{,}752$ MACs (about 39 M).

Parameters are dominated by stage 3 (about 77% of the conv weights: $9\cdot24{,}576=221{,}184$), compute is spread almost evenly, with stages 2 and 3 slightly ahead of stage 1; this is the VGG pattern, where doubling channels balances the halved area. Doubling the input resolution leaves the parameter count unchanged (GAP removes the dependence), but the conv MACs scale 4x, to about 154 M.
:::
:::

:::exercise Exercise 9: the shape of the effective receptive field (derivation, challenging)
Model a stack of $n$ stride-1 convolutions whose kernel is the uniform average over three taps in 1D, $\tfrac13(1,1,1)$, with no nonlinearity. The effective receptive field of the last unit is the distribution of the impulse response of the whole stack. (a) Show that it is the $n$-fold convolution of the kernel with itself. (b) Compute its variance, and show that the standard deviation grows as $\sqrt n$ while the theoretical receptive field grows as $2n+1$. (c) What does this imply about how much of the receptive field is effectively used?

:::details Solution
(a) For a linear stack, the response to an impulse is the successive convolution with each kernel, so the effective weight profile is $K^{*n}$, the kernel convolved with itself $n$ times. (b) Think of the kernel as a probability distribution over offsets $\{-1,0,1\}$ with variance $\sigma_1^2=\tfrac13(1+0+1)=\tfrac23$. Convolving distributions adds independent random variables, so variances add: $\sigma_n^2=2n/3$ and $\sigma_n=\sqrt{2n/3}$. By the central limit theorem the profile approaches a Gaussian. (c) The support grows linearly ($2n+1$) but almost all the mass lies within a few $\sigma_n\propto\sqrt n$ of the center, so the effective receptive field is a Gaussian-like blob that is a shrinking fraction of the theoretical field ($\sigma_n/(2n+1)\to0$). In real networks this is modified by nonlinearities, learned weights and skip connections, but the qualitative conclusion (effective RF smaller than theoretical RF, concentrated at the center) is what Luo et al. (2016) observed.
:::
:::

:::exercise Exercise 10: design an experiment (what would happen if...)
A student reports "my 7x7 network beats the 3x3 one by 0.4% accuracy, so large kernels are better". List at least four problems with this claim, and describe the experiment that would support or refute it.

:::details Solution
Problems: (1) one run each, with no estimate of seed-to-seed variation (typically a few tenths of a percent, so 0.4% may be noise); (2) the 7x7 model has 5.4x more parameters and many more MACs, so the comparison confounds kernel size with capacity and compute; (3) the learning rate and weight decay were probably tuned for one model, favouring it; (4) if the test set was used for choosing epochs or the best model, the numbers are optimistic for both; (5) one dataset and resolution only (32 x 32 images may reward big kernels differently than 224 x 224).

A better experiment: match parameter count or MACs (adjust widths), tune the learning rate for each, use at least 3 to 5 seeds, select on a validation split, report mean and standard deviation, and test on the held-out test set once. If the difference is smaller than about two standard deviations, report no significant difference.
:::
:::

:::exercise Exercise 11: translation equivariance, with and without padding (coding)
Implement a test that checks equivariance of a `Conv2d`: shifting the input by $(s_y,s_x)$ and then convolving should equal convolving and then shifting. Do it with (a) `padding_mode="circular"` and `torch.roll`, and (b) zero padding. Explain what you observe and how it relates to the claim in [the equivariance section](#translation-equivariance-stated-precisely).

:::details Solution
```python
import torch, torch.nn as nn

x = torch.randn(1, 3, 16, 16)
for mode in ["circular", "zeros"]:
    conv = nn.Conv2d(3, 4, 3, padding=1, padding_mode=mode, bias=False)
    shift = lambda t: torch.roll(t, shifts=(2, 3), dims=(2, 3))
    a, b = conv(shift(x)), shift(conv(x))
    err = (a - b).abs()
    print(mode, "max error:", err.max().item(), " interior max error:", err[..., 4:-4, 4:-4].max().item())
```
With circular padding and a circular shift the error is zero (up to float rounding): the layer is exactly equivariant. With zero padding the error is nonzero near the borders (the rolled content wraps around where a real image would have zeros), but the interior is equal to rounding error: equivariance holds away from boundaries. Stride greater than 1 breaks exact equivariance for shifts that are not multiples of the stride, a known aliasing issue.
:::
:::

:::exercise Exercise 12: dilated stacks (derivation and computation)
A stack of 3x3 convolutions with dilations $1,2,4,\dots,2^{L-1}$ and stride 1: (a) derive the receptive field. (b) How many layers of plain 3x3 convs would be needed to match $L=6$? (c) What artifact does dilation introduce when all layers use the same dilation $d=2$, and how do the exponentially increasing dilations avoid it?

:::details Solution
(a) A dilated layer adds $(k-1)d=2d$ to the receptive field at jump 1, so $r=1+2\sum_{i=0}^{L-1}2^i=1+2(2^L-1)=2^{L+1}-1$. For $L=6$, $r=127$. (b) Plain stack: $2n+1=127$ gives $n=63$ layers. (c) With a constant dilation of 2 the layers see only every other pixel (a checkerboard pattern: "gridding"), leaving holes in the receptive field, and in the worst case entire pixels never influence the output. Using dilations $1,2,4,\dots$ (or the hybrid dilated convolution rule of mixing rates with no common factor) fills the holes, because each new layer covers the gaps left by the previous ones; this is the idea Yu and Koltun use in their context module.
:::
:::

## Key takeaways

:::key Key takeaways
- A convolution layer is a linear map with three constraints: **locality**, **weight sharing** and the resulting **translation equivariance**. They cut parameters from $O(n^2)$ to $O(k^2C_\text{in}C_\text{out})$ and encode the right prior for images.
- Deep learning "convolution" is cross-correlation; the distinction matters only for the flip in the backward pass.
- Output size: $\lfloor (n+2p-d(k-1)-1)/s\rfloor+1$. Learn it, and print shapes every time.
- Parameters and MACs scale as $k^2C_\text{in}C_\text{out}$ and $k^2C_\text{in}C_\text{out}HW$ respectively. 1x1 convolutions mix channels cheaply; depthwise-separable convolutions cut cost by roughly $1/C_\text{out}+1/k^2$.
- Receptive field grows by $(k-1)\times$ jump per layer: stride and dilation make it grow faster, and the effective receptive field is a smaller, Gaussian-like region.
- The backward pass of a conv is again a conv: the input gradient is a transposed convolution, the kernel gradient is a correlation of input with output gradient.
- LeNet introduced the recipe, AlexNet scaled it with ReLU, GPUs, dropout and augmentation, VGG showed that stacks of 3x3 convolutions beat large kernels; global average pooling removes the parameter-heavy head.
- Training a CNN well is as much about the pipeline (normalization, augmentation, a clean validation split, a good schedule) as the architecture. Ablate one factor at a time, with several seeds.
:::

## Further reading {#further-reading}

**Essential**

- Yann LeCun, Leon Bottou, Yoshua Bengio, Patrick Haffner. *Gradient-based learning applied to document recognition.* Proceedings of the IEEE, 1998. The LeNet-5 paper, still the best introduction to the ideas.
- Alex Krizhevsky, Ilya Sutskever, Geoffrey Hinton. *ImageNet classification with deep convolutional neural networks.* NeurIPS 2012. The AlexNet paper.
- Karen Simonyan, Andrew Zisserman. *Very deep convolutional networks for large-scale image recognition.* ICLR 2015, arXiv:1409.1556. The VGG paper.

**Foundations and history**

- Kunihiko Fukushima. *Neocognitron: a self-organizing neural network model for a mechanism of pattern recognition unaffected by shift in position.* Biological Cybernetics, 1980. The architectural precursor with S-cells and C-cells.
- David Hubel, Torsten Wiesel. *Receptive fields, binocular interaction and functional architecture in the cat's visual cortex.* Journal of Physiology, 1962. The neuroscience that inspired the Neocognitron.
- Yann LeCun et al. *Backpropagation applied to handwritten zip code recognition.* Neural Computation, 1989.
- Olga Russakovsky et al. *ImageNet large scale visual recognition challenge.* IJCV, 2015, arXiv:1409.0575.
- Alex Krizhevsky. *Learning multiple layers of features from tiny images.* Technical report, University of Toronto, 2009 (CIFAR-10 and CIFAR-100).

**Convolution variants and analysis**

- Fisher Yu, Vladlen Koltun. *Multi-scale context aggregation by dilated convolutions.* ICLR 2016, arXiv:1511.07122.
- Andrew Howard et al. *MobileNets: efficient convolutional neural networks for mobile vision applications.* 2017, arXiv:1704.04861.
- Francois Chollet. *Xception: deep learning with depthwise separable convolutions.* CVPR 2017, arXiv:1610.02357.
- Wenjie Luo, Yujia Li, Raquel Urtasun, Richard Zemel. *Understanding the effective receptive field in deep convolutional neural networks.* NeurIPS 2016, arXiv:1701.04128.
- Min Lin, Qiang Chen, Shuicheng Yan. *Network in network.* ICLR 2014, arXiv:1312.4400 (1x1 convolutions and global average pooling).
- Vincent Dumoulin, Francesco Visin. *A guide to convolution arithmetic for deep learning.* 2016, arXiv:1603.07285. The best reference for stride, padding and transposed convolutions.
- Richard Zhang. *Making convolutional networks shift-invariant again.* ICML 2019, arXiv:1904.11486.
- Sergey Ioffe, Christian Szegedy. *Batch normalization: accelerating deep network training by reducing internal covariate shift.* ICML 2015, arXiv:1502.03167.

**Next week:** residual networks, efficient architectures, transfer learning, detection and segmentation, in [Week 6](week-06.html).
