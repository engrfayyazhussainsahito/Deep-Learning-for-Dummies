---
title: Modern Vision Architectures, Transfer Learning, Detection and Segmentation
phase: Core Architectures
summary: Residual learning and why skip connections make very deep networks trainable; ResNeXt, DenseNet, Inception, MobileNet, EfficientNet and ConvNeXt; transfer learning and fine-tuning; object detection (R-CNN family, SSD, YOLO, anchors, IoU, NMS, mAP); semantic and instance segmentation (FCN, U-Net, DeepLab, Mask R-CNN).
---

:::key Learning objectives
By the end of this week you should be able to:
- Explain the degradation problem and show, with a gradient-flow argument, why residual connections ease optimization of deep networks.
- Draw and count parameters of basic and bottleneck residual blocks, ResNeXt blocks, dense blocks, Inception modules and MobileNetV2 inverted residuals.
- Use EfficientNet-style compound scaling to trade depth, width and resolution against FLOPs, and describe how ConvNeXt modernizes a ResNet.
- Choose between feature extraction, partial and full fine-tuning for a new dataset; handle batch norm, learning rates and domain shift correctly.
- Compute IoU, run non-maximum suppression, and compute precision-recall curves, AP and mAP by hand and in code.
- Describe how R-CNN, Fast/Faster R-CNN, SSD, YOLO, RetinaNet and anchor-free detectors differ, and what the loss and the output tensor of each look like.
- Describe FCN, U-Net, DeepLab (atrous convolution, ASPP) and Mask R-CNN, and compute IoU and Dice for masks.
- Fine-tune a pretrained ResNet, run and evaluate a pretrained detector, and train a small U-Net.
:::

[Week 5](week-05.html) gave you the convolutional toolbox and three classic networks. Those networks stop at about 20 layers, and after VGG people discovered that simply stacking more layers did not work. This week starts with the idea that fixed that, the residual connection, and follows its consequences: networks that are hundreds of layers deep, networks that are cheap enough to run on a phone, and a recipe that makes almost every other vision problem easier, which is to start from a network pretrained on ImageNet. We then leave classification: detection asks *what is where* (a variable-size list of boxes) and segmentation asks *which pixels belong to what*. Both reuse the same backbones, and both bring new ideas about outputs, losses and evaluation.

You will need the convolution arithmetic from [Week 5](week-05.html), batch normalization and initialization from [Week 3](week-03.html), and the fine-tuning intuition about regularization and validation from [Week 4](week-04.html). **Assignment 2 is due at the end of Week 8**, and its image-classification part is a natural place to apply the transfer-learning recipe of this week.

## Residual networks {#resnet}

### The degradation problem {#degradation}

By 2014 the trend was clear: deeper networks (AlexNet's 8 layers, VGG's 16 to 19, GoogLeNet's 22) were more accurate. Batch normalization ([Week 3](week-03.html#normalization)) had removed the worst vanishing and exploding signal problems at initialization, so one might expect that a 56-layer plain network would at least match a 20-layer one. It does not. He et al. (2016) trained plain CNNs of 20 and 56 layers on CIFAR-10 and found that the deeper network had *higher training error* as well as higher test error. This is the **degradation problem**.

Notice what this does and does not say. It is not overfitting, because training error went up. It is not (mainly) vanishing gradients, because the networks use batch norm and their gradients were measured to be healthy. The deeper network is a superset in function space: you could construct a 56-layer network from a trained 20-layer one by adding 36 layers that compute the identity. A solver that is given more capacity should find a solution at least that good, so the failure is one of **optimization**: SGD cannot find the identity-like configuration for the extra layers, or finds it only very slowly. Learning "do nothing" with a stack of nonlinear layers (convolution, BN, ReLU) is surprisingly hard, because the identity is not a point that a randomly initialized stack is anywhere near.

### Residual learning {#residual-learning}

The fix is to change what the layers are asked to learn. Let $\boldsymbol{x}$ be the input of a block and $\mathcal{H}(\boldsymbol{x})$ the mapping we want it to compute. Instead of making the stacked layers fit $\mathcal{H}$ directly, make them fit the **residual** $\mathcal{F}(\boldsymbol{x}) = \mathcal{H}(\boldsymbol{x}) - \boldsymbol{x}$ and add the input back:

$$
\boldsymbol{y} = \mathcal{F}(\boldsymbol{x}; \{\mathbf{W}_i\}) + \boldsymbol{x}.
$$

If the best thing for a block to do is nothing, the optimizer only has to push the weights of $\mathcal{F}$ toward zero, which is easy. If the optimal mapping is close to the identity, $\mathcal{F}$ only needs to model a small perturbation. And since the addition has no parameters, it costs nothing in compute. The path $\boldsymbol{x} \to \boldsymbol{y}$ through the addition is called the **skip connection** or **shortcut**.

{{figure: week06-residual-blocks | Two residual blocks. Left: the basic block of ResNet-18/34 (two 3x3 convolutions). Right: the bottleneck block of ResNet-50/101/152, where 1x1 convolutions shrink the channel count to 64 before the expensive 3x3 convolution and expand it back to 256. The orange path is the identity shortcut; the + node is an element-wise sum, followed by ReLU. Parameter counts exclude BN.}}

The block needs $\mathcal{F}(\boldsymbol{x})$ and $\boldsymbol{x}$ to have the same shape. When a stage halves the resolution and doubles the channels, the shortcut must change shape too. ResNet uses a **projection shortcut**: a $1\times1$ convolution with stride 2 (followed by BN) applied to $\boldsymbol{x}$, so that $\boldsymbol{y} = \mathcal{F}(\boldsymbol{x}) + \mathbf{W}_s\boldsymbol{x}$. Everywhere else the shortcut is the identity.

### The ResNet architecture {#resnet-architecture}

A ResNet for ImageNet has a stem, four stages of residual blocks and a head:

1. **Stem**: $7\times7$ convolution with 64 channels and stride 2 (224 to 112), BN, ReLU, then $3\times3$ max-pool with stride 2 (112 to 56).
2. **Four stages** at resolutions 56, 28, 14 and 7. Each stage begins with a block that halves the resolution (except the first stage) and uses the projection shortcut. Base widths are 64, 128, 256, 512.
3. **Head**: global average pooling, one linear layer with 1000 outputs ([Week 5](week-05.html#pooling-types)).

The depth names (18, 34, 50, 101, 152) count weight layers: convolutions and the final linear layer. They differ in the block type and the number of blocks per stage:

| Model | Block | Blocks per stage | Stage output channels | Params | ImageNet top-1 (torchvision, V1 weights) |
| --- | --- | --- | --- | --- | --- |
| ResNet-18 | basic | 2, 2, 2, 2 | 64, 128, 256, 512 | 11.7M | 69.76% |
| ResNet-34 | basic | 3, 4, 6, 3 | 64, 128, 256, 512 | 21.8M | 73.31% |
| ResNet-50 | bottleneck | 3, 4, 6, 3 | 256, 512, 1024, 2048 | 25.6M | 76.13% |
| ResNet-101 | bottleneck | 3, 4, 23, 3 | 256, 512, 1024, 2048 | 44.5M | 77.37% |
| ResNet-152 | bottleneck | 3, 8, 36, 3 | 256, 512, 1024, 2048 | 60.2M | 78.31% |

Accuracies are those listed in the torchvision documentation for its pretrained weights (single center crop at 224 pixels). They are a little higher than the numbers in the original paper because torchvision's models are the "v1.5" variant and use a somewhat different training pipeline.

Three design points deserve attention.

**Bottlenecks keep deep networks affordable.** In the right-hand block of the figure, a $1\times1$ convolution first reduces 256 channels to 64, the $3\times3$ convolution works on only 64 channels, and a final $1\times1$ restores 256. The cost is $256\cdot64 + 9\cdot64\cdot64 + 64\cdot256 = 69{,}632$ weights, about the same as one basic block at 64 channels (73,728), but the block operates on a feature map four times as wide. This is the same dimension-reduction trick Inception used (below), and the same arithmetic as the $1\times1$ convolutions of [Week 5](week-05.html#the-1x1-convolution).

**Stride placement.** The original paper puts the stride-2 in the first $1\times1$ convolution of a downsampling bottleneck, which discards three quarters of the activations before the $3\times3$ sees them. The "v1.5" variant used by torchvision and most frameworks puts the stride in the $3\times3$ convolution instead, which is slightly more accurate and costs about 4.09 GFLOPs rather than 3.8.

**BN everywhere, and the final ReLU after the addition.** Each convolution is followed by BN. In the original block the ReLU after the sum means that the shortcut is not a perfectly clean identity path, a point the pre-activation variant (below) improves.

### Why skip connections help: gradient flow {#gradient-flow}

Intuition first. A deep plain network is a long product of Jacobians. Each factor needs to have singular values near 1 for the product to be well behaved, which is a knife-edge condition. A residual network replaces each factor $\mathbf{J}_l$ by $\mathbf{I} + \mathbf{J}'_l$, where $\mathbf{J}'_l$ is the Jacobian of the residual branch. Now every factor is "identity plus something", and an identity term is exactly what keeps the product from collapsing or exploding.

:::math Gradient through a stack of residual blocks
Consider blocks $\boldsymbol{x}_{l+1} = \boldsymbol{x}_l + \mathcal{F}_l(\boldsymbol{x}_l)$ with *no* nonlinearity after the addition (the pre-activation design below achieves this). Unrolling from block $l$ to block $L$:
$$
\boldsymbol{x}_L = \boldsymbol{x}_l + \sum_{i=l}^{L-1}\mathcal{F}_i(\boldsymbol{x}_i).
$$
Let $\mathcal{L}$ be the loss. By the chain rule,
$$
\frac{\partial\mathcal{L}}{\partial\boldsymbol{x}_l}
= \frac{\partial\mathcal{L}}{\partial\boldsymbol{x}_L}\,\frac{\partial\boldsymbol{x}_L}{\partial\boldsymbol{x}_l}
= \frac{\partial\mathcal{L}}{\partial\boldsymbol{x}_L}\left(\mathbf{I} + \frac{\partial}{\partial\boldsymbol{x}_l}\sum_{i=l}^{L-1}\mathcal{F}_i(\boldsymbol{x}_i)\right).
$$
The first term, $\partial\mathcal{L}/\partial\boldsymbol{x}_L$, is passed to *every* earlier block unchanged, however deep the network is and whatever the residual branches do. In a plain network the corresponding expression is a product $\prod_{i}\mathbf{J}_i$, which can only vanish or explode exponentially in the depth. Here the gradient is the sum of an undiminished term and a correction, so it is very unlikely to vanish (it would require the correction to cancel $\mathbf{I}$ exactly).
:::

This analysis also exposes the flip side: the sum has $L-l$ terms, so if each residual branch contributes a gradient of similar size, the total *grows* with depth. This is why we do not initialize residual branches with full-strength weights. The next chart shows the effect in a controlled experiment.

```chart
{
  "type": "line",
  "caption": "Gradient norm at each layer (relative to the loss end, log10 scale) in a 50-layer fully connected ReLU network of width 128, computed with NumPy (random Gaussian weights scaled by sqrt(2/128) times the stated gain; no BN). Plain networks are stable only at gain exactly 1 and decay or explode by roughly 5 orders of magnitude otherwise. Residual networks with small-gain branches stay within a factor of a few of 1, because the identity path is always present.",
  "height": 380,
  "data": {
    "labels": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50],
    "datasets": [
      {"label": "plain, gain 0.8", "data": [-4.847, -4.745, -4.631, -4.551, -4.477, -4.376, -4.282, -4.194, -4.098, -4.006, -3.896, -3.806, -3.687, -3.552, -3.445, -3.371, -3.261, -3.162, -3.065, -2.967, -2.854, -2.774, -2.712, -2.642, -2.535, -2.429, -2.338, -2.24, -2.149, -2.075, -1.983, -1.875, -1.771, -1.674, -1.575, -1.481, -1.388, -1.258, -1.19, -1.079, -0.992, -0.902, -0.776, -0.661, -0.55, -0.448, -0.338, -0.277, -0.184, -0.075, 0.0], "borderColor": "var(--c-red)", "pointRadius": 0, "borderWidth": 2.5},
      {"label": "plain, gain 1.0", "data": [-0.001, 0.004, 0.021, 0.004, -0.019, -0.015, -0.018, -0.027, -0.027, -0.033, -0.02, -0.027, -0.005, 0.034, 0.044, 0.021, 0.034, 0.036, 0.036, 0.037, 0.053, 0.036, 0.001, -0.026, -0.015, -0.006, -0.012, -0.011, -0.017, -0.04, -0.045, -0.034, -0.027, -0.026, -0.024, -0.028, -0.031, 0.002, -0.027, -0.013, -0.023, -0.03, -0.001, 0.017, 0.032, 0.036, 0.05, 0.013, 0.01, 0.022, 0.0], "borderColor": "var(--c-muted)", "pointRadius": 0, "borderWidth": 2.5},
      {"label": "plain, gain 1.25", "data": [4.844, 4.753, 4.673, 4.558, 4.439, 4.346, 4.246, 4.14, 4.043, 3.94, 3.857, 3.753, 3.678, 3.62, 3.533, 3.413, 3.329, 3.234, 3.137, 3.042, 2.96, 2.847, 2.715, 2.591, 2.504, 2.417, 2.314, 2.218, 2.115, 1.996, 1.893, 1.807, 1.718, 1.621, 1.526, 1.426, 1.326, 1.262, 1.136, 1.053, 0.946, 0.842, 0.775, 0.695, 0.613, 0.521, 0.438, 0.304, 0.203, 0.119, 0.0], "borderColor": "var(--c-amber)", "pointRadius": 0, "borderWidth": 2.5},
      {"label": "residual, branch gain 0.1", "data": [0.237, 0.23, 0.223, 0.215, 0.206, 0.199, 0.19, 0.182, 0.174, 0.168, 0.162, 0.155, 0.148, 0.142, 0.135, 0.129, 0.123, 0.117, 0.111, 0.106, 0.1, 0.095, 0.089, 0.084, 0.08, 0.076, 0.072, 0.068, 0.064, 0.06, 0.056, 0.051, 0.049, 0.046, 0.043, 0.039, 0.037, 0.034, 0.033, 0.029, 0.026, 0.023, 0.02, 0.018, 0.016, 0.012, 0.01, 0.007, 0.004, 0.003, 0.0], "borderColor": "var(--c-blue)", "pointRadius": 0, "borderWidth": 2.5},
      {"label": "residual, branch gain 0.3", "data": [2.59, 2.537, 2.482, 2.421, 2.352, 2.285, 2.213, 2.138, 2.064, 2.001, 1.947, 1.869, 1.796, 1.731, 1.663, 1.591, 1.517, 1.455, 1.373, 1.297, 1.228, 1.15, 1.081, 1.002, 0.943, 0.882, 0.804, 0.739, 0.68, 0.624, 0.574, 0.518, 0.485, 0.443, 0.402, 0.367, 0.332, 0.3, 0.275, 0.248, 0.223, 0.194, 0.166, 0.144, 0.124, 0.103, 0.083, 0.057, 0.036, 0.021, 0.0], "borderColor": "var(--c-green)", "pointRadius": 0, "borderWidth": 2.5}
    ]
  },
  "options": { "scales": { "x": {"title": {"display": true, "text": "layer index (0 = input side, 50 = loss side)"}}, "y": {"title": {"display": true, "text": "log10 ( gradient norm / gradient norm at the top )"}} } }
}
```

In practice, residual networks with BN are protected against the growth by two habits: BN inside the branch keeps its output at unit scale, and the last BN scale parameter $\gamma$ of every block is initialized to **zero** (Goyal et al., 2017), so that each block starts as the exact identity and the network starts as a shallow one that gradually "turns on" its blocks. In torchvision, `ResNet(..., zero_init_residual=True)` does this.

{{demo: week06-residual-gradient | Gradient flow in plain versus residual networks | 460}}

#### The unraveled view and the loss landscape {#unraveled}

A second way to understand ResNets was given by Veit, Wilber and Belongie (2016). Expand the recursion $\boldsymbol{x}_{l+1} = \boldsymbol{x}_l + \mathcal{F}_l(\boldsymbol{x}_l)$ for a network with $n$ blocks: each block is either "skipped" or "taken", so the network is a sum over $2^n$ paths of lengths ranging from 0 to $n$. The distribution of path lengths is binomial, concentrated around $n/2$, and (for a 110-layer ResNet on CIFAR-10) the paths that carry most of the gradient are much shorter than the full depth. Veit et al. also deleted individual blocks from trained ResNets and found that test error rose only modestly, unlike VGG, where removing a single layer is catastrophic. The picture is of an **ensemble of shallow networks** that share weights, rather than one very deep serial computation. This is a useful, but partial, account: the paths are not independent, and and stochastic depth (Huang et al., 2016), which randomly drops whole residual blocks during training, uses the same property as a regularizer.

A third, geometric view comes from Li et al. (2018), who visualized the loss surface along random directions around the solution. For deep plain networks the surface becomes chaotic with many sharp non-convex regions, and with skip connections it becomes much smoother and more convex-looking. This is consistent with the easier optimization, though visualizations along 2D slices have well-known limits (see the discussion of landscapes in [Week 3](week-03.html)).

### Pre-activation ResNets {#preact}

He et al. (2016b, "Identity mappings") analyzed what happens if the shortcut is a clean identity and the output of the addition is passed on *unchanged*, with no ReLU after the sum. Reordering the layers of the branch to BN-ReLU-conv, BN-ReLU-conv (instead of conv-BN-ReLU) achieves this. Then the derivation above is exactly true for the whole network, and they could train a 1001-layer ResNet on CIFAR with lower error than the original design. The pre-activation form is also what Transformers use later: [pre-norm residual blocks](week-09.html) are the same idea, with LayerNorm and attention in place of BN and convolution.

### ResNet in code {#resnet-code}

The following implements both blocks and a general ResNet from scratch, mirroring torchvision's layout. We check it against the numbers in the table by counting parameters.

```python:resnet_from_scratch.py
import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, c_in, width, stride=1):
        super().__init__()
        c_out = width * self.expansion
        self.conv1 = nn.Conv2d(c_in, width, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, c_out, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.relu = nn.ReLU(inplace=True)
        # Projection shortcut only when the shape changes
        self.shortcut = None
        if stride != 1 or c_in != c_out:
            self.shortcut = nn.Sequential(nn.Conv2d(c_in, c_out, 1, stride, bias=False), nn.BatchNorm2d(c_out))

    def forward(self, x):
        identity = x if self.shortcut is None else self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)          # F(x) + x, then ReLU (original post-activation design)

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, c_in, width, stride=1):
        super().__init__()
        c_out = width * self.expansion
        self.conv1 = nn.Conv2d(c_in, width, 1, bias=False)              # reduce
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, 3, stride, 1, bias=False)  # "v1.5": stride lives here
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, c_out, 1, bias=False)             # expand
        self.bn3 = nn.BatchNorm2d(c_out)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = None
        if stride != 1 or c_in != c_out:
            self.shortcut = nn.Sequential(nn.Conv2d(c_in, c_out, 1, stride, bias=False), nn.BatchNorm2d(c_out))

    def forward(self, x):
        identity = x if self.shortcut is None else self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return self.relu(out + identity)

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=True):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1))
        stages, c_in = [], 64
        for i, (n, width) in enumerate(zip(layers, [64, 128, 256, 512])):
            blocks = []
            for b in range(n):
                stride = 2 if (b == 0 and i > 0) else 1      # downsample at the start of stages 2-4
                blocks.append(block(c_in, width, stride))
                c_in = width * block.expansion
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(c_in, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if zero_init_residual:                                # every block starts as the identity
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.zeros_(m.bn3.weight)
                elif isinstance(m, BasicBlock):
                    nn.init.zeros_(m.bn2.weight)

    def forward(self, x):
        x = self.stages(self.stem(x))
        return self.fc(self.pool(x).flatten(1))

def resnet18(**kw): return ResNet(BasicBlock, [2, 2, 2, 2], **kw)
def resnet50(**kw): return ResNet(Bottleneck, [3, 4, 6, 3], **kw)

for name, net in [("resnet18", resnet18()), ("resnet50", resnet50())]:
    n = sum(p.numel() for p in net.parameters())
    print(name, f"{n:,}")                                     # resnet18 11,689,512   resnet50 25,557,032
x = torch.randn(2, 3, 224, 224)
h = resnet50().stem(x); print("after stem:", tuple(h.shape))   # (2, 64, 56, 56)
print("logits:", tuple(resnet50()(x).shape))                   # (2, 1000)
```

The parameter counts match torchvision's `resnet18` and `resnet50` exactly, which is a good habit for any architecture you reimplement: check your count against a reference.

:::note Training matters as much as architecture
ResNet-50's torchvision accuracy is 76.13% with the original-style recipe and 80.86% with a modern recipe (longer training, label smoothing, mixup/cutmix, EMA, stronger augmentation, learning-rate warmup and cosine decay). Wightman et al. (2021, "ResNet strikes back") make the same point systematically. When you read that architecture A beats architecture B, ask whether they were trained with the same recipe.
:::

```chart
{
  "type": "bar",
  "caption": "ImageNet top-1 accuracy of torchvision ResNets with two sets of pretrained weights: IMAGENET1K_V1 (original-style recipe) and IMAGENET1K_V2 (modern recipe; not released for ResNet-18/34). Values from the torchvision documentation. The recipe moves the same architecture by 4 to 5 points, more than the gain from doubling depth.",
  "height": 340,
  "data": {
    "labels": ["ResNet-18","ResNet-34","ResNet-50","ResNet-101","ResNet-152"],
    "datasets": [
      {"label": "V1 weights", "data": [69.76, 73.31, 76.13, 77.37, 78.31], "backgroundColor": "var(--c-blue)"},
      {"label": "V2 weights", "data": [null, null, 80.86, 81.89, 82.28], "backgroundColor": "var(--c-green)"}
    ]
  },
  "options": { "scales": { "x": {"title": {"display": true, "text": "model"}}, "y": {"title": {"display": true, "text": "top-1 accuracy (%)"}, "min": 60} } }
}
```

:::history Where ResNet came from
Kaiming He, Xiangyu Zhang, Shaoqing Ren and Jian Sun (Microsoft Research Asia) won ILSVRC 2015 classification, detection and localization, and COCO 2015 detection and segmentation, with a 152-layer residual network, reaching 3.57% top-5 error on the test set (see the chart in [Week 5](week-05.html#ilsvrc-history)). The paper (CVPR 2016) is one of the most cited in all of machine learning. Highway networks (Srivastava, Greff and Schmidhuber, 2015) had earlier proposed learned *gates* on skip connections; ResNet's contribution was that the plain, parameter-free identity is enough and works better.
:::

### ResNeXt and DenseNet {#resnext-densenet}

Two natural extensions of the residual idea changed different parts of the block.

**ResNeXt** (Xie et al., 2017) asks how to spend parameters. Instead of making a block wider or the network deeper, split the block's transformation into $C$ parallel paths, each a narrow $1\times1 \to 3\times3 \to 1\times1$ bottleneck, and *sum* them. $C$ is called the **cardinality**. Implemented literally it would be messy, but a sum of $C$ narrow paths is equivalent to a single bottleneck whose middle $3\times3$ convolution is a **grouped convolution** with $C$ groups ([Week 5](week-05.html#depthwise-separable-convolution) treated groups as the family that contains depthwise convolution). The canonical "ResNeXt-50 32x4d" has 32 groups with 4 channels per group in the first stage, and roughly the same parameters (25.0M) and FLOPs as ResNet-50 but higher accuracy (77.62% versus 76.13% in torchvision). The authors showed that at fixed complexity, increasing cardinality beat increasing depth or width.

**DenseNet** (Huang et al., 2017) changes how the shortcut combines. Instead of adding $\boldsymbol{x}$ to the block output, each layer receives the *concatenation* of the outputs of *all* previous layers in the block, and produces only $k$ new feature maps (the **growth rate**, e.g. $k=32$). A block with $L$ layers has $L(L+1)/2$ direct connections. The consequences are: very short paths from every layer to the loss (strong gradient flow), feature reuse (later layers can read early low-level features directly, so each layer can be narrow), and a small parameter count (DenseNet-121 has 8.0M parameters and 74.43% top-1). The price is memory: the concatenations create many large intermediate tensors, so DenseNets are memory-hungry for their FLOPs and slow in practice unless implemented carefully.

{{figure: week06-resnext-densenet | Left: a ResNeXt block with cardinality 32 (path width 4). Only four of the 32 paths are drawn; the dots stand for the rest. The sum of paths equals a grouped 3x3 convolution between two 1x1 convolutions. Right: a DenseNet block in which each layer's input is the concatenation of everything before it; the orange arcs are the long skip connections, and each "H" adds only k feature maps.}}

The distinction is worth remembering because it recurs: **addition** (ResNet) merges information into a fixed-size stream and preserves the interpretation of each channel, **concatenation** (DenseNet, and U-Net below) keeps information separate and grows the channel dimension.

## Efficient and scalable designs {#efficient-designs}

ResNet answered "how do we train deep networks?". The next question was economic: given a budget of parameters, FLOPs, memory or milliseconds on a phone, what is the best architecture? The designs in this section are better read as answers to different budgets than as a linear sequence of improvements.

### Inception: parallel branches and 1x1 bottlenecks {#inception}

**GoogLeNet** (Szegedy et al., 2014, published at CVPR 2015) won ILSVRC 2014 with 22 layers and only about 6.8M parameters, roughly 20 times fewer than VGG-16, by replacing the choice "which kernel size should this layer use?" with "use all of them". An **Inception module** applies, in parallel and to the same input, a $1\times1$ convolution, a $3\times3$ convolution, a $5\times5$ convolution and a $3\times3$ max-pool, then concatenates the resulting feature maps along the channel axis. The network can then choose, per module and per spatial position, how much to rely on small versus large receptive fields.

The naive version is too expensive: every branch sees all input channels. The key trick is to put a $1\times1$ convolution *before* the expensive $3\times3$ and $5\times5$ branches to shrink the channel count, exactly the same device as the ResNet bottleneck. Count it for an input with $C_\text{in}=192$ channels and a $5\times5$ branch with 32 output channels:

| Branch | Parameters (no bias) |
| --- | --- |
| direct $5\times5$, $192\to32$ | $25 \cdot 192 \cdot 32 = 153{,}600$ |
| $1\times1$ reduce $192\to16$, then $5\times5$, $16\to32$ | $192\cdot16 + 25\cdot16\cdot32 = 3{,}072 + 12{,}800 = 15{,}872$ |

A tenfold reduction, at the price of one extra nonlinearity (which is itself useful). The reduction branches turn the module from a "wide" layer into a family of narrow computations, and this idea, a $1\times1$ convolution as a learned channel-mixing projection, is now ubiquitous.

GoogLeNet also attached two **auxiliary classifiers** to intermediate layers during training, injecting gradient into the middle of the network. This was a pre-batch-norm workaround for the same optimization difficulty that ResNet later solved; later work found the auxiliary heads act mostly as regularizers and they are rarely used today.

The Inception family was refined for several years. **Inception-v2/v3** (Szegedy et al., 2016, "Rethinking the Inception architecture") introduced *factorization*: a $5\times5$ convolution can be replaced by two stacked $3\times3$ convolutions (same receptive field, $2\cdot9=18$ versus 25 weights per channel pair, plus an extra nonlinearity), and an $n\times n$ convolution by a $1\times n$ followed by an $n\times1$ convolution, which saves more as $n$ grows. The same paper popularized **label smoothing**, which is still a standard regularizer. **Inception-v4 and Inception-ResNet** (Szegedy et al., 2017) added residual connections to the modules, confirming that the two ideas combine. **Xception** (Chollet, 2017) pushed the Inception logic to its limit: if each branch looks at a separate subset of channels, why not give every channel its own spatial filter? That is a depthwise separable convolution, the building block of the next family.

:::note Why this matters beyond ImageNet
Two ideas from Inception are still everywhere. First, *multi-branch then merge*, which returns in ResNeXt (above), in Feature Pyramid Networks (below) and in the multiple heads of Transformers ([Week 9](week-09.html)). Second, the $1\times1$ convolution as a cheap per-pixel linear layer, which is exactly a position-wise fully connected layer. The "MLP block" of a Transformer is the same operation applied to tokens instead of pixels.
:::

### MobileNets: depthwise separable convolutions {#mobilenet}

**MobileNetV1** (Howard et al., 2017) targets devices where multiplications are scarce. Its only structural idea is the depthwise separable convolution that [Week 5](week-05.html#depthwise-separable-convolution) introduced. Recall the cost of a standard $D_K\times D_K$ convolution mapping $M$ input channels to $N$ output channels on a $D_F\times D_F$ feature map:

$$
\underbrace{D_K^2\, M\, N\, D_F^2}_{\text{standard}}
\quad\text{versus}\quad
\underbrace{D_K^2\, M\, D_F^2}_{\text{depthwise}} + \underbrace{M\, N\, D_F^2}_{\text{pointwise } 1\times1}.
$$

The ratio is

$$
\frac{D_K^2 M D_F^2 + M N D_F^2}{D_K^2 M N D_F^2} = \frac{1}{N} + \frac{1}{D_K^2}.
$$

For $3\times3$ kernels and $N=128$ that is $1/128 + 1/9 \approx 0.119$, an 8 to 9 times saving. Almost all of the remaining cost sits in the $1\times1$ convolutions (about 95% of the multiply-adds and 75% of the parameters in MobileNetV1), which is good news for hardware since $1\times1$ convolutions are plain matrix multiplications.

MobileNetV1 adds two scalar knobs to trade accuracy for cost without redesigning anything: a **width multiplier** $\alpha\in(0,1]$ that multiplies every channel count (cost scales roughly as $\alpha^2$), and a **resolution multiplier** $\rho$ that scales the input size (cost scales as $\rho^2$). The published numbers show how gracefully this degrades.

```chart
{
  "type": "scatter",
  "caption": "MobileNetV1 width multipliers at 224x224 input versus two heavier baselines (ImageNet top-1 against multiply-adds, log x axis). Values from Howard et al. (2017), Tables 4 and 8: MobileNet 1.0/0.75/0.5/0.25 reach 70.6/68.4/63.7/50.6% with 569M/325M/149M/41M multiply-adds; GoogLeNet 69.8% (1550M); VGG-16 71.5% (15,300M).",
  "height": 340,
  "data": {
    "datasets": [
      {"label": "MobileNetV1 (width 1.0, 0.75, 0.5, 0.25)", "data": [{"x":569,"y":70.6},{"x":325,"y":68.4},{"x":149,"y":63.7},{"x":41,"y":50.6}], "borderColor": "var(--c-blue)", "backgroundColor": "var(--c-blue)", "showLine": true, "pointRadius": 6},
      {"label": "GoogLeNet", "data": [{"x":1550,"y":69.8}], "borderColor": "var(--c-amber)", "backgroundColor": "var(--c-amber)", "pointRadius": 8},
      {"label": "VGG-16", "data": [{"x":15300,"y":71.5}], "borderColor": "var(--c-red)", "backgroundColor": "var(--c-red)", "pointRadius": 8}
    ]
  },
  "options": { "scales": { "x": {"type": "logarithmic", "title": {"display": true, "text": "multiply-adds (millions, log scale)"}}, "y": {"title": {"display": true, "text": "ImageNet top-1 accuracy (%)"}, "min": 45, "max": 75} } }
}
```

Read the chart carefully: MobileNet 1.0 matches VGG-16 within one point with 27 times fewer multiply-adds, but accuracy falls off a cliff below width 0.5, because narrow layers have too little capacity.

### MobileNetV2: the inverted residual {#inverted-residual}

**MobileNetV2** (Sandler et al., 2018) kept depthwise separable convolutions and added two ideas that transfer far beyond mobile networks.

**Inverted residual.** A ResNet bottleneck is *wide - narrow - wide*: a $1\times1$ convolution shrinks the representation, the $3\times3$ works on the narrow version, and another $1\times1$ expands again; the shortcut connects the wide ends. MobileNetV2 flips it. The block takes a *narrow* input, **expands** it by a factor $t$ (usually 6) with a $1\times1$ convolution, applies a $3\times3$ **depthwise** convolution in the wide space, and **projects** back to a narrow tensor with a second $1\times1$ convolution; the shortcut joins the narrow ends, and only when stride is 1 and the channel counts match. The depthwise convolution is cheap, so the expensive wide space can be afforded, while the tensors stored between blocks (the ones that cost memory bandwidth) stay small.

{{figure: week06-inverted-residual | ResNet's bottleneck (left) versus MobileNetV2's inverted residual (right). Shortcuts connect the 256-channel tensors in one and the 24-channel tensors in the other. The last 1x1 convolution of the inverted block has no activation (a linear bottleneck).}}

**Linear bottleneck.** The final $1\times1$ projection has no ReLU. The authors' argument is geometric: a ReLU applied to a low-dimensional activation destroys information (every negative coordinate is clipped to zero and cannot be recovered), whereas in a high-dimensional space the same ReLU is much more likely to leave a linear-image of the useful manifold intact. So nonlinearities belong in the wide space, and the narrow space should stay linear. Treat this as a motivated design heuristic rather than a theorem, but the ablation in the paper supports it. MobileNetV2 also uses ReLU6, $\min(\max(x,0),6)$, which is friendlier to low-precision arithmetic.

Parameter count for the block in the figure (24 channels in and out, expansion 6, so 144 in the middle):

$$
\underbrace{24\cdot144}_{\text{expand}} + \underbrace{9\cdot144}_{\text{depthwise}} + \underbrace{144\cdot24}_{\text{project}} = 3456 + 1296 + 3456 = 8{,}208.
$$

**MobileNetV3** (Howard et al., 2019) found a better arrangement of these blocks with neural architecture search, added squeeze-and-excitation (a small per-channel gating module, Hu et al., 2018) and replaced ReLU with the cheap swish approximation "h-swish". In torchvision, `mobilenet_v2` has 3.5M parameters and about 72% top-1, and `mobilenet_v3_large` 5.5M and about 75%.

### EfficientNet and compound scaling {#efficientnet}

Given a baseline network, there are three ways to make it bigger: more **layers** (depth $d$), more **channels** (width $w$), higher **input resolution** ($r$). Before 2019 people usually scaled one at a time: ResNet-18 to ResNet-152 is depth, wide ResNets are width, and Inception variants used larger images. Tan and Le (2019) observed that each dimension saturates quickly when scaled alone (a very deep network on small images, or a very wide shallow one, gains little), and that the dimensions interact: a higher-resolution image has more pixels and needs a larger receptive field (more depth) and more fine-grained patterns (more width).

Their **compound scaling** uses one coefficient $\phi$ to scale all three together:

$$
d = \alpha^{\phi},\qquad w = \beta^{\phi},\qquad r = \gamma^{\phi},\qquad \text{subject to}\quad \alpha\,\beta^2\,\gamma^2 \approx 2,\quad \alpha,\beta,\gamma \ge 1.
$$

The constraint comes from the cost model. FLOPs of a convolutional network are proportional to $d\cdot w^2\cdot r^2$: doubling depth doubles FLOPs, but doubling width quadruples them (both $C_\text{in}$ and $C_\text{out}$ grow) and doubling resolution also quadruples them (height times width). With the constraint, FLOPs grow by $(\alpha\beta^2\gamma^2)^\phi \approx 2^\phi$: each unit of $\phi$ doubles the budget. The base constants are found once, by a small grid search with $\phi=1$ on the small baseline: $\alpha=1.2$, $\beta=1.1$, $\gamma=1.15$ (check: $1.2\cdot1.21\cdot1.3225\approx1.92$). Then $\phi$ is increased to get a family.

The baseline **EfficientNet-B0** was found by neural architecture search over MobileNetV2-like blocks (called MBConv: inverted residual plus squeeze-and-excitation) optimizing accuracy and FLOPs. Scaling it up gives B1 to B7 (the published B-models use hand-picked resolutions and rounding, so they follow the formula only approximately).

{{demo: week06-compound-scaling | Compound scaling: depth, width, resolution and FLOPs | 460}}

```chart
{
  "type": "scatter",
  "caption": "ImageNet top-1 accuracy versus parameter count for the EfficientNet family and some earlier ConvNets. All values from Table 2 of Tan and Le (2019), single-crop, single-model. EfficientNet-B7 reaches 84.3% with 66M parameters.",
  "height": 380,
  "data": {
    "datasets": [
      {"label": "EfficientNet B0 to B7", "data": [{"x":5.3,"y":77.1},{"x":7.8,"y":79.1},{"x":9.2,"y":80.1},{"x":12,"y":81.6},{"x":19,"y":82.9},{"x":30,"y":83.6},{"x":43,"y":84.0},{"x":66,"y":84.3}], "borderColor": "var(--c-green)", "backgroundColor": "var(--c-green)", "showLine": true, "pointRadius": 5},
      {"label": "ResNet-50, ResNet-152", "data": [{"x":26,"y":76.0},{"x":60,"y":77.8}], "borderColor": "var(--c-blue)", "backgroundColor": "var(--c-blue)", "showLine": true, "pointRadius": 5},
      {"label": "ResNeXt-101", "data": [{"x":84,"y":80.9}], "borderColor": "var(--c-purple)", "backgroundColor": "var(--c-purple)", "pointRadius": 7},
      {"label": "Xception", "data": [{"x":23,"y":79.0}], "borderColor": "var(--c-teal)", "backgroundColor": "var(--c-teal)", "pointRadius": 7},
      {"label": "Inception-ResNet-v2", "data": [{"x":56,"y":80.1}], "borderColor": "var(--c-amber)", "backgroundColor": "var(--c-amber)", "pointRadius": 7},
      {"label": "NASNet-A", "data": [{"x":89,"y":82.7}], "borderColor": "var(--c-red)", "backgroundColor": "var(--c-red)", "pointRadius": 7}
    ]
  },
  "options": { "scales": { "x": {"type": "logarithmic", "title": {"display": true, "text": "parameters (millions, log scale)"}}, "y": {"title": {"display": true, "text": "ImageNet top-1 accuracy (%)"}, "min": 74, "max": 86} } }
}
```

EfficientNet-B4 (19M parameters, 4.2 GFLOPs) beats ResNet-152 (60M, 11 GFLOPs) by more than 5 points. Two caveats a careful reader should keep in mind. First, FLOPs and parameters are proxies: EfficientNets use depthwise convolutions and large activations that are memory-bound, so on GPUs their wall-clock speed is much worse than the FLOP count suggests. Second, a good part of the gain came from the training recipe (AutoAugment, stochastic depth, RMSProp with long schedules) that older baselines were not given; the next section is largely about that confound. **EfficientNetV2** (Tan and Le, 2021) addressed the speed issue with "fused" MBConv blocks (a regular $3\times3$ convolution replaces the expand-plus-depthwise pair in early stages, where depthwise convolutions underuse accelerators) and progressive training with growing image sizes.

### ConvNeXt: modernizing the ConvNet {#convnext}

By 2020 Vision Transformers ([Week 10](week-11.html)) were reported to outperform ConvNets. Liu et al. (2022) asked whether this was due to the attention mechanism, or to everything else that had changed around it. Their paper, **"A ConvNet for the 2020s"**, starts with a ResNet-50 and, step by step, applies design choices borrowed from Transformers while keeping it a pure convolutional network, measuring ImageNet accuracy after each change. The resulting model family is **ConvNeXt**. The path, in order, was:

1. **Training recipe.** Train with AdamW, 300 epochs, Mixup, CutMix, RandAugment, random erasing, stochastic depth, label smoothing and EMA of the weights: ResNet-50 goes from 76.1% to about 78.8% with *no architectural change*, the same effect the torchvision V1-to-V2 weights showed in the chart above.
2. **Macro design.** Change the stage ratios from (3, 4, 6, 3) to (3, 3, 9, 3) (Swin-style) and replace the $7\times7$ stride-2 stem plus max-pool by a **patchify stem**: a $4\times4$ convolution with stride 4, equivalent to cutting the image into non-overlapping patches as ViT does.
3. **ResNeXt-ify and invert.** Use depthwise convolutions (the extreme of grouping) and widen the network, then adopt an **inverted bottleneck** where the hidden dimension is four times the width, as in MobileNetV2 and the Transformer MLP.
4. **Large kernels.** Move the depthwise convolution to the start of the block and enlarge it to $7\times7$, approximating the large spatial context of window attention.
5. **Micro design.** Replace ReLU with **GELU**, keep only one activation and one normalization per block (fewer than ResNet), replace BatchNorm with **LayerNorm**, and use separate $2\times2$ stride-2 downsampling layers between stages.

The final block is: depthwise $7\times7$ conv, LayerNorm, $1\times1$ conv expanding by 4, GELU, $1\times1$ conv back, a learnable per-channel **layer scale**, then add the shortcut. In PyTorch:

```python:convnext_block.py
import torch
import torch.nn as nn

class LayerNorm2d(nn.LayerNorm):
    """LayerNorm over the channel dimension of an (N, C, H, W) tensor."""
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)             # N, H, W, C
        x = super().forward(x)
        return x.permute(0, 3, 1, 2)          # back to N, C, H, W

class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, layer_scale_init=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise 7x7
        self.norm = LayerNorm2d(dim, eps=1e-6)
        self.pw1 = nn.Conv2d(dim, 4 * dim, kernel_size=1)    # expand x4 (inverted bottleneck)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(4 * dim, dim, kernel_size=1)    # project back
        self.gamma = nn.Parameter(layer_scale_init * torch.ones(1, dim, 1, 1))  # layer scale

    def forward(self, x):
        h = self.pw2(self.act(self.pw1(self.norm(self.dwconv(x)))))
        return x + self.gamma * h             # residual connection

x = torch.randn(2, 96, 56, 56)
print(ConvNeXtBlock(96)(x).shape)             # torch.Size([2, 96, 56, 56])
n_params = sum(p.numel() for p in ConvNeXtBlock(96).parameters())
print(n_params)                               # 7*7*96 + 96 + 2*96 + (96*384+384) + (384*96+96) + 96
```

ConvNeXt-T (28M parameters, 4.5 GFLOPs) reaches roughly 82% top-1, on par with Swin-T at the same cost. The lesson is methodological: a large share of "architecture progress" is recipe progress, and when you compare architectures, hold the recipe fixed.

:::tip Which backbone should you choose?
For a server GPU and no strict latency budget, a ResNet-50 or ConvNeXt with a modern training recipe is a robust default. For mobile or CPU inference, start with MobileNetV3 or EfficientNet-Lite. When you only care about accuracy and have pretraining data, a pretrained ViT or ConvNeXt will usually win. Always measure latency on the target hardware, not FLOPs.
:::

## Transfer learning and fine-tuning {#transfer-learning}

Almost nobody trains a vision network from scratch for a new problem. A network pretrained on a large dataset (ImageNet-1k with 1.28M images, or something larger) has learned a hierarchy of visual features, and those features are a far better starting point than random weights when your own dataset has hundreds or thousands of images instead of millions. **Transfer learning** is the practice of reusing such a network; **fine-tuning** is the case where some or all of the pretrained weights are updated on the new task.

### Why transfer works: features are general at the bottom {#why-transfer-works}

Yosinski, Clune, Bengio and Lipson (2014) measured how transferable the layers of an AlexNet-style network are by copying the first $n$ layers from a network trained on one half of ImageNet classes to another network, freezing or fine-tuning them, and training the rest on the other half. First-layer features (Gabor-like edge and colour detectors, see [Week 5](week-05.html)) transferred essentially perfectly, while features near the top were specific to the source task. A sensible mental model for a pretrained CNN is therefore:

- **early layers**: edges, textures, colour blobs, nearly universal;
- **middle layers**: parts and motifs, reusable across many natural-image tasks;
- **last layers**: combinations specific to the source label set (the 1000 ImageNet classes), the part you most often replace.

Kornblith, Shlens and Le (2019) later studied 12 image classification datasets and found that, with fine-tuning, better ImageNet accuracy generally predicts better transfer accuracy, so a stronger backbone is usually a better starting point; with a frozen backbone this correlation is weaker and depends on the regularization used during pretraining.

### Feature extraction versus fine-tuning {#feature-extraction-vs-finetuning}

There is a spectrum of recipes, shown in the figure. The right choice depends on two things: how much labelled data you have, and how similar your data are to the pretraining data.

{{figure: week06-transfer-recipes | Three common recipes on a pretrained ResNet. A (linear probe): freeze the whole backbone and train a new classification head. B (partial fine-tuning): also unfreeze the last stage(s) with a small learning rate. C (full fine-tuning): update everything, with learning rates that shrink toward the input. Blue layers are pretrained and trainable, grey are frozen, green is newly initialised.}}

| Recipe | What is trained | Use when | Risks |
| --- | --- | --- | --- |
| **Feature extraction / linear probe** | New head only; the backbone is a fixed feature function | Very little data (tens to hundreds of images per class), target close to ImageNet | Underfits if the domain differs; features may not contain what you need |
| **Partial fine-tuning** | Head plus the top blocks | Moderate data (hundreds to thousands per class), some domain shift | Choosing how many blocks to unfreeze needs validation |
| **Full fine-tuning** | All weights, small learning rates | Plenty of data or large domain gap (medical images, satellite, microscopy) | Overfitting, destroying useful features ("catastrophic forgetting"), expensive |

A useful practical consequence of feature extraction: since the backbone is frozen and deterministic (in eval mode), you can compute the features of every training image *once*, cache them, and train a linear classifier or logistic regression on them in seconds. That makes the linear probe a superb first baseline and a diagnostic: if a linear probe on pretrained features already reaches your target accuracy, there is no point fine-tuning.

### Domain shift {#domain-shift}

Fine-tuning assumes the target data are not too different from the source. **Domain shift** means $p_\text{target}(\boldsymbol{x})$ differs from $p_\text{source}(\boldsymbol{x})$, even when the label function is the same. ImageNet photographs of dogs and Oxford-IIIT pet breeds differ little; ImageNet versus chest X-rays (one grey channel, textures governed by physics rather than natural objects), histopathology, satellite imagery or spectrograms differ a lot. As the gap grows, more layers must adapt: the early features still help optimization (they are better than random initialization in nearly every case studied), but you should unfreeze more of the network and use more augmentation. He, Girshick and Dollar (2019, "Rethinking ImageNet pre-training") showed an important nuance: on COCO detection, training from random initialization can match a pretrained model if you train long enough with normalization such as GroupNorm or SyncBN; pretraining mainly buys **faster convergence** and helps most when the target dataset is small. So pretraining is a prior, not magic: more target data means weaker prior needed.

:::warning Subtle shifts that quietly hurt
Even within natural photos: (1) input statistics: you must normalize with the same mean and standard deviation as pretraining (ImageNet: mean $(0.485, 0.456, 0.406)$, std $(0.229, 0.224, 0.225)$ for torchvision models); (2) resolution: objects that are tiny in your images look different from ImageNet objects occupying a third of the frame; (3) label shift: if your classes are imbalanced differently from the source, calibration of the pretrained head is useless, so replace it; (4) channel layout: RGB versus BGR (OpenCV!) is a classic silent bug.
:::

### Layer freezing, batch norm and discriminative learning rates {#freezing-and-lrs}

Three practical mechanisms implement the recipes.

**Freezing** means excluding parameters from gradient computation: set `p.requires_grad_(False)`, or put them in no optimizer group. Freezing is not only a regularizer; it also saves compute (no backward pass through frozen early layers if nothing before them needs gradient) and memory.

**Batch norm is the trap.** A frozen BatchNorm layer has two kinds of state: learnable $\gamma,\beta$ and *running statistics* (mean, variance). `requires_grad_(False)` freezes the first kind only. If the module is in `train()` mode its running statistics keep updating with your tiny batches (often of size 8 to 32 on a new dataset with a different distribution), and the *batch* statistics used in the forward pass differ from the ones the downstream weights were trained with. For a truly frozen backbone, call `.eval()` on it. In partial fine-tuning with small batches, it is common to keep all BN layers in eval mode, even in the unfrozen blocks, and only train their affine parameters (or not at all). Remember that `model.train()` re-enables train mode for every submodule, so the freeze must be reapplied after each call, a classic source of bugs.

**Discriminative learning rates** (popularized for language models by Howard and Ruder, 2018, in ULMFiT; the same idea is standard in vision) give lower layers a smaller learning rate than upper layers: early features are general and already good, so they should move little; the new head starts random and needs a large step. In PyTorch this is one optimizer with several parameter groups. A typical ratio is 3 to 10 between neighboring stages. **Gradual unfreezing** trains the head first, then unfreezes one stage at a time from the top. Also useful: a short **warmup** (a randomly initialised head sends large, noisy gradients into the pretrained backbone in the first steps), **cosine or step decay**, and a learning rate for the backbone 10 to 100 times smaller than you would use from scratch (for SGD with momentum, around $10^{-3}$ to $10^{-2}$ for the head and $10^{-4}$ for the backbone is a reasonable start; for AdamW, $10^{-5}$ to $10^{-4}$).

```python:finetune_setup.py
import torch
import torch.nn as nn
from torchvision import models

def build_resnet(num_classes, recipe="head"):
    """Pretrained ResNet-18 prepared for one of three transfer recipes."""
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)   # new random head

    stages = [model.layer1, model.layer2, model.layer3, model.layer4]
    stem = [model.conv1, model.bn1]

    if recipe == "head":                      # A: linear probe
        for p in model.parameters():
            p.requires_grad_(False)
        for p in model.fc.parameters():
            p.requires_grad_(True)
        groups = [{"params": model.fc.parameters(), "lr": 1e-3}]
    elif recipe == "partial":                 # B: unfreeze layer3, layer4
        for p in model.parameters():
            p.requires_grad_(False)
        groups = [{"params": model.fc.parameters(), "lr": 1e-3}]
        for m in [model.layer3, model.layer4, model.fc]:
            for p in m.parameters():
                p.requires_grad_(True)
        groups += [{"params": model.layer3.parameters(), "lr": 1e-4},
                   {"params": model.layer4.parameters(), "lr": 1e-4}]
    elif recipe == "full":                    # C: everything, discriminative LRs
        groups = [{"params": [p for m in stem for p in m.parameters()], "lr": 1e-5},
                  {"params": model.layer1.parameters(), "lr": 1e-5},
                  {"params": model.layer2.parameters(), "lr": 3e-5},
                  {"params": model.layer3.parameters(), "lr": 1e-4},
                  {"params": model.layer4.parameters(), "lr": 3e-4},
                  {"params": model.fc.parameters(), "lr": 1e-3}]
    else:
        raise ValueError(recipe)
    return model, weights.transforms(), groups

def set_train_mode(model, recipe):
    """Call after every model.train(): keep frozen BN layers in eval mode."""
    model.train()
    if recipe == "head":
        model.eval(); model.fc.train()        # whole backbone in eval mode
    elif recipe == "partial":
        for m in [model.conv1, model.bn1, model.layer1, model.layer2]:
            m.eval()

model, preprocess, groups = build_resnet(num_classes=10, recipe="partial")
opt = torch.optim.AdamW(groups, weight_decay=1e-2)
n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
n_all = sum(p.numel() for p in model.parameters())
print(f"trainable parameters: {n_train:,} of {n_all:,}")
print(preprocess)    # Resize(256) -> CenterCrop(224) -> normalize with ImageNet statistics
```

### Data augmentation, small data and other practicalities {#transfer-practicalities}

With a few hundred images, augmentation matters more than the choice of optimizer: random resized crops, horizontal flips (when the label is flip-invariant, which is not true of digits or text), colour jitter, and for medical or aerial data rotations and vertical flips. Do not augment the validation set. Use a **stratified** split so that every class appears in validation, and report variance over several seeds, because a 200-image validation set has a standard error of several percent. Label smoothing, weight decay and early stopping on validation loss are cheap safeguards. If you must resize images, resize to the resolution the network was trained at (224 for most torchvision classifiers), though fully convolutional backbones with global average pooling accept other sizes, and a modestly larger test resolution often helps ("FixRes" effect: Touvron et al., 2019).

When the head must output something other than a class (regression, multi-label with sigmoid outputs, embeddings for retrieval), the recipe is identical: drop `fc`, attach a new head. When you have unlabeled data from the target domain, self-supervised or semi-supervised pretraining is the next tool ([Week 11](week-11.html)).

## Object detection {#object-detection}

Classification outputs one label per image. **Object detection** outputs a *variable-size set* of objects, each with a class label, a bounding box $(x_1,y_1,x_2,y_2)$ and a confidence score. That changes almost everything: the output size is not fixed, one image may contain 0 or 80 objects, boxes must be matched to ground truth to define a loss, and duplicates must be removed. Most of the history of detection is a sequence of answers to two questions: where do candidate boxes come from, and how do we turn dense candidate predictions into a clean set?

### Intersection over union and box encoding {#iou}

Everything in detection needs a notion of how well two boxes agree. The standard measure is **intersection over union** (IoU, also called the Jaccard index):

$$
\mathrm{IoU}(A, B) = \frac{|A \cap B|}{|A \cup B|} = \frac{|A\cap B|}{|A| + |B| - |A\cap B|} \in [0, 1].
$$

It is 1 for identical boxes, 0 for disjoint ones, and is invariant to the scale of the boxes (which a plain $L_2$ distance of coordinates is not: a 5-pixel error is huge for a 20-pixel box and negligible for a 500-pixel one). Two boxes of $100\times 100$ shifted by 20 pixels horizontally overlap in $80\times100 = 8000$ pixels, so $\mathrm{IoU} = 8000/(10000 + 10000 - 8000) = 0.67$; shifted by 50 pixels it falls to $5000/15000 = 0.33$.

{{figure: week06-iou-anchors-nms | Left: IoU as the area of the overlap divided by the area of the union. Middle: anchor boxes of several scales and aspect ratios tiled at every location of a feature map. Right: non-maximum suppression keeps the best-scoring box of each cluster of overlapping boxes.}}

```python:iou.py
import numpy as np

def iou_matrix(a, b):
    """Pairwise IoU. a: (N, 4), b: (M, 4), boxes as (x1, y1, x2, y2). Returns (N, M)."""
    a = np.asarray(a, dtype=float)[:, None, :]     # (N, 1, 4)
    b = np.asarray(b, dtype=float)[None, :, :]     # (1, M, 4)
    ix1 = np.maximum(a[..., 0], b[..., 0]); iy1 = np.maximum(a[..., 1], b[..., 1])
    ix2 = np.minimum(a[..., 2], b[..., 2]); iy2 = np.minimum(a[..., 3], b[..., 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    return inter / (area_a + area_b - inter + 1e-12)

A = [[0, 0, 100, 100]]
B = [[20, 0, 120, 100], [50, 0, 150, 100], [200, 200, 300, 300]]
print(iou_matrix(A, B).round(3))                   # [[0.667 0.333 0.   ]]
```

Detectors do not regress raw corner coordinates. Given a reference box (an anchor or proposal) with center $(x_a,y_a)$ and size $(w_a,h_a)$ and a ground-truth box $(x,y,w,h)$, the standard parametrization of Girshick et al. predicts the *offsets*

$$
t_x = \frac{x - x_a}{w_a},\quad t_y = \frac{y - y_a}{h_a},\quad t_w = \log\frac{w}{w_a},\quad t_h = \log\frac{h}{h_a},
$$

so that the targets are roughly scale-free and centered near zero, and a predicted box is recovered by inverting these formulas. The loss on these offsets is a **smooth $L_1$** (Huber-like) loss: quadratic for small errors, linear for large ones, so a few badly localized boxes cannot dominate the gradient. Modern detectors sometimes add an IoU-based loss (GIoU, DIoU, CIoU) that optimizes the metric directly and still gives a gradient when the boxes do not overlap.

### The R-CNN family: region proposals plus classification {#rcnn}

**R-CNN** (Girshick, Donahue, Darrell and Malik, 2014) was the first deep detector to beat hand-crafted pipelines by a large margin, and it is a pipeline in the literal sense. (1) Run an external proposal algorithm, **selective search**, to get about 2000 candidate boxes that are likely to contain *some* object. (2) Warp each box to a fixed size and run it independently through a CNN to get a feature vector. (3) Classify each feature vector with one SVM per class. (4) Refine the box with a class-specific linear regressor. It worked (the paper reports a mAP of 53.3% on PASCAL VOC 2012, more than 30% better in relative terms than the previous best) but was slow: 2000 forward passes per image, around 47 seconds per image on a GPU with VGG-16, and three separately trained stages.

**Fast R-CNN** (Girshick, 2015) observed that the 2000 crops overlap heavily, so most computation is redundant. Run the backbone *once* on the whole image to get a feature map, then for each proposal extract a fixed-size ($7\times7$) feature from the corresponding region of the shared map with **RoI pooling** (divide the region into a $7\times7$ grid and max-pool each cell; the coordinates are quantized to the feature-map grid). Two fully connected heads then output class probabilities and box offsets, trained jointly with a multi-task loss

$$
\mathcal{L} = \mathcal{L}_\text{cls}(p, u) + \lambda\,[u \ge 1]\,\mathcal{L}_\text{loc}(t^u, v),
$$

where $u$ is the true class (0 is background), $p$ the predicted class distribution, $t^u$ the predicted offsets for class $u$, $v$ the target offsets and $[u\ge1]$ switches the localization term off for background proposals. Training and testing became an order of magnitude faster, but proposal generation (selective search, on the CPU, about 2 seconds per image) became the bottleneck.

**Faster R-CNN** (Ren, He, Girshick and Sun, 2015) removed that bottleneck by making the proposal step a part of the network. The **Region Proposal Network (RPN)** is a small convolutional head slid over the backbone feature map. At each location it considers $k$ **anchors**, reference boxes of several scales and aspect ratios (the paper uses 3 scales times 3 ratios, so $k=9$), and predicts for each anchor an *objectness* score (object versus background, 2 outputs) and 4 box offsets: a feature map of size $H\times W$ yields $9HW$ proposals, with $2k$ and $4k$ output channels. Anchors are *labelled* by IoU with the ground truth: positive if their IoU with some ground-truth box is above 0.7 (or if they have the highest IoU with it), negative if below 0.3 for all boxes, and ignored otherwise. The top proposals after NMS are fed to the Fast R-CNN head, which shares the same backbone features. The two parts are trained jointly; the result is a two-stage detector that runs at about 5 fps with VGG-16 and was the foundation for most detection research for years.

Two later improvements are essential in practice. **Feature Pyramid Networks** (Lin et al., 2017) build a multi-scale pyramid with a top-down pathway and lateral connections so that small objects are detected on high-resolution, semantically strong feature maps and large objects on coarse ones; Faster R-CNN with ResNet-50-FPN is the default in torchvision. **RoIAlign** (He et al., 2017, in Mask R-CNN) replaces the quantizing RoI pooling with bilinear sampling at exact fractional positions; the half-pixel misalignments that RoI pooling introduces barely matter for classification but hurt boxes and ruin masks.

{{figure: week06-detection-evolution | From R-CNN to one-stage detectors. Each step shares more computation and learns more of the pipeline: R-CNN runs a CNN per proposal; Fast R-CNN runs the backbone once; Faster R-CNN learns the proposals; one-stage detectors predict class scores and boxes densely and skip the proposal-and-refine step.}}

### One-stage detectors: SSD and YOLO {#one-stage}

Two-stage detectors are accurate but complex. **One-stage detectors** predict classes and boxes directly from the feature map in a single pass.

**YOLO** ("You Only Look Once", Redmon, Divvala, Girshick and Farhadi, 2016) divides the image into an $S\times S$ grid ($S=7$). Each cell predicts $B=2$ boxes, each with $(x,y,w,h)$ and a confidence, and one set of $C$ class probabilities (shared by the cell's boxes), so the output is a single tensor of shape $S\times S\times(5B + C) = 7\times7\times30$ for the 20 PASCAL VOC classes. The loss is a sum-of-squares over all outputs with hand-chosen weights ($\lambda_\text{coord}=5$ for boxes, $\lambda_\text{noobj}=0.5$ for empty cells, and square roots of width and height so that errors on small boxes count more). It reaches 45 fps (155 fps for the small "Fast YOLO"), and because it sees the whole image at once it makes fewer background mistakes than Fast R-CNN, but it localizes less precisely and struggles with small objects in clusters, since each cell can only predict two boxes of one class. Later versions kept the single-pass recipe and fixed its weaknesses: YOLOv2 (2017) added batch norm, anchor boxes and multi-resolution training, YOLOv3 (2018) predicts at three scales with an FPN-like neck, and the line continues through many versions by different groups.

**SSD** (Single Shot MultiBox Detector, Liu et al., 2016) attaches small convolutional prediction heads to *several* feature maps of different resolutions, so large objects are detected on coarse maps and small ones on fine maps. Each location of each map has a set of default boxes (anchors) and the head predicts, per default box, class scores and four offsets using a single $3\times3$ convolution. Training matches default boxes to ground truth (IoU > 0.5) and uses **hard negative mining**: since the overwhelming majority of default boxes are background, keep only the negatives with the highest loss, at a 3:1 ratio to positives.

#### The class imbalance problem and focal loss {#focal-loss}

Dense detectors evaluate on the order of $10^4$ to $10^5$ candidate locations per image, and only a handful contain objects. The sheer number of easy background examples dominates the sum of the cross-entropy losses, even though each one has a tiny loss. That imbalance, rather than any weakness of the one-stage idea, was why one-stage detectors lagged two-stage ones in accuracy. **RetinaNet** (Lin, Goyal, Girshick, He and Dollar, 2017) closed the gap with the **focal loss**, which multiplies cross-entropy by a factor that down-weights well-classified examples. With $p_t$ the model's probability for the true class,

$$
\mathrm{FL}(p_t) = -\alpha_t\,(1 - p_t)^{\gamma}\,\log p_t,\qquad \gamma = 2.
$$

An easy example with $p_t = 0.9$ has its loss multiplied by $(0.1)^2 = 0.01$, and a hard one with $p_t = 0.1$ by $(0.9)^2 = 0.81$: the ratio between the two is now 81 times larger than in plain cross-entropy, so training focuses on the hard cases. With $\gamma=0$ it reduces to (class-weighted) cross-entropy.

```python:focal_loss.py
import torch
import torch.nn.functional as F

def sigmoid_focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    """Binary focal loss, mean over elements. logits, targets: same shape, targets in {0,1}."""
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)           # prob. of the true class
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    return (alpha_t * (1 - p_t) ** gamma * ce).mean()

logits = torch.tensor([4.0, 0.0, -4.0, -4.0])            # easy positive, uncertain positive, easy negative, hard positive
targets = torch.tensor([1.0, 1.0, 0.0, 1.0])             # the last logit (-4) for a positive target is badly wrong: a hard example
print(sigmoid_focal_loss(logits, targets))
# torchvision.ops.sigmoid_focal_loss implements the same formula (reduction="mean")
```

### Anchor-based versus anchor-free detection {#anchor-free}

Anchors are a prior over box shapes. They make training easier (each anchor only has to learn a small correction), but they come with costs: the scales and ratios are hyperparameters that must match the dataset, the matching rules (IoU thresholds) are heuristics, and the number of anchors is huge, most of them negative. **Anchor-free** detectors remove the anchors and treat detection as a dense prediction problem:

- **FCOS** (Tian, Shen, Chen and He, 2019) predicts at every feature-map location inside a ground-truth box the four distances $(l,t,r,b)$ from the location to the box sides, plus a **centerness** score that down-weights locations far from the object center. There are no anchors and no IoU matching, only a rule that assigns each box to the pyramid levels appropriate for its size.
- **CornerNet** (Law and Deng, 2018) predicts the top-left and bottom-right corners as keypoints and groups them; **CenterNet / "Objects as Points"** (Zhou, Wang and Krahenbuhl, 2019) predicts each object's center as a peak in a heatmap and regresses the size there, which removes even NMS in its simplest form (peaks are extracted with a max-pool).
- **DETR** (Carion et al., 2020) is a different kind of anchor-free design: a Transformer decoder outputs a fixed set of object "queries", and a **Hungarian matching** between predictions and ground truth defines the loss, so no NMS is needed. We return to this in [Week 11](week-11.html).

The best current detectors show that anchor-free versus anchor-based matters less than training details: label assignment, loss, augmentation and schedule dominate.

### Non-maximum suppression {#nms}

Dense detectors produce many overlapping boxes for each object: neighbouring anchors or locations all fire. **Non-maximum suppression (NMS)** converts that cluster into a single box. The greedy algorithm, applied separately per class:

:::definition Greedy NMS
1. Discard boxes with score below a threshold (e.g. 0.05).
2. Sort the rest by score. While the list is not empty: take the highest-scoring box $b^*$ and add it to the output; remove from the list every remaining box with $\mathrm{IoU}(b^*, b) > \tau$ (usually $\tau=0.5$).
:::

The threshold $\tau$ is a trade-off. A low $\tau$ removes duplicates aggressively but also removes true neighbours in crowds (pedestrians, dense shelf products). A high $\tau$ leaves duplicate detections, which are counted as false positives in evaluation. **Soft-NMS** (Bodla et al., 2017) decays the scores of overlapping boxes rather than deleting them, for instance multiplying by $\exp(-\mathrm{IoU}^2/\sigma)$, which helps in crowded scenes. Play with the demo below: in IoU mode drag the second box, in NMS mode change the threshold and the scores.

{{demo: week06-iou-nms | IoU and non-maximum suppression, interactively | 520}}

```python:nms.py
import numpy as np
# iou_matrix defined in iou.py above

def nms(boxes, scores, iou_thr=0.5):
    """Greedy NMS for one class. Returns indices of kept boxes, highest score first."""
    order = np.argsort(-scores)
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        ious = iou_matrix(boxes[i:i + 1], boxes[order[1:]])[0]
        order = order[1:][ious <= iou_thr]          # keep only boxes that do not overlap too much
    return keep

boxes = np.array([[10, 10, 110, 110], [15, 12, 112, 108], [20, 20, 120, 125],
                  [200, 200, 260, 280], [205, 195, 258, 275]], dtype=float)
scores = np.array([0.92, 0.88, 0.60, 0.80, 0.75])
print(nms(boxes, scores, 0.5))          # [0, 3]: two objects, three duplicates removed
# torchvision.ops.nms(boxes_t, scores_t, 0.5) and batched_nms (per class) do this on tensors
```

### Evaluating detectors: precision, recall, AP and mAP {#map}

Accuracy is meaningless for detection, since "background" is everywhere. Instead we ask: of the detections the model made (above some confidence), how many were correct (**precision**), and of the real objects, how many were found (**recall**)? Both depend on the confidence threshold, so we sweep it and measure the whole curve.

:::definition Average precision, step by step (one class, one IoU threshold)
1. Sort all detections of the class, over the whole test set, by decreasing confidence.
2. Go down the list. A detection is a **true positive (TP)** if its IoU with a ground-truth box of the same class is at least the IoU threshold and that ground-truth box has not been matched by a higher-scoring detection; otherwise it is a **false positive (FP)**. Duplicates are therefore false positives.
3. After each detection compute cumulative $\mathrm{precision} = \mathrm{TP}/(\mathrm{TP}+\mathrm{FP})$ and $\mathrm{recall} = \mathrm{TP}/N_\text{gt}$. This traces a precision-recall curve.
4. Make precision monotone by replacing it with the maximum precision at any recall $\ge r$ (the "envelope"): $p_\text{interp}(r) = \max_{r'\ge r} p(r')$.
5. **AP** is the area under that envelope. **mAP** is the mean of AP over classes.
:::

A concrete example. A class has $N_\text{gt}=8$ objects. The model produced 14 detections; sorted by score, they are TP, TP, FP, TP, TP, FP, TP, FP, TP, FP, FP, TP, FP, FP (7 TPs; one object is never found). The cumulative precision and recall values (computed with NumPy) are:

| rank | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TP? | 1 | 1 | 0 | 1 | 1 | 0 | 1 | 0 | 1 | 0 | 0 | 1 | 0 | 0 |
| precision | 1.00 | 1.00 | 0.67 | 0.75 | 0.80 | 0.67 | 0.71 | 0.62 | 0.67 | 0.60 | 0.55 | 0.58 | 0.54 | 0.50 |
| recall | .125 | .25 | .25 | .375 | .50 | .50 | .625 | .625 | .75 | .75 | .75 | .875 | .875 | .875 |

The envelope replaces each precision by the largest precision to its right (for example rank 3's 0.67 becomes 0.80, because rank 5 reaches 0.80 at higher recall). The area under the resulting staircase is $\mathrm{AP}=0.6955$. The older PASCAL VOC 2007 metric averaged the envelope at 11 recall points $\{0, 0.1, \dots, 1\}$ and gives $0.669$ here.

```chart
{
  "type": "line",
  "caption": "Precision-recall curve for the 14-detection example in the text (computed with NumPy). The raw curve is jagged; the envelope takes the best precision at any higher recall; AP, the area under the envelope, is 0.6955.",
  "height": 340,
  "data": {
    "labels": [0.125, 0.25, 0.25, 0.375, 0.5, 0.5, 0.625, 0.625, 0.75, 0.75, 0.75, 0.875, 0.875, 0.875],
    "datasets": [
      {"label": "raw precision", "data": [1.0, 1.0, 0.667, 0.75, 0.8, 0.667, 0.714, 0.625, 0.667, 0.6, 0.545, 0.583, 0.538, 0.5], "borderColor": "var(--c-blue)", "pointRadius": 3, "tension": 0},
      {"label": "envelope", "data": [1.0, 1.0, 0.8, 0.8, 0.8, 0.714, 0.714, 0.667, 0.667, 0.6, 0.583, 0.583, 0.538, 0.5], "borderColor": "var(--c-amber)", "pointRadius": 0, "borderDash": [6, 4], "tension": 0}
    ]
  },
  "options": { "scales": { "x": {"title": {"display": true, "text": "recall of the detection at that rank (categorical axis, rank order preserved)"}}, "y": {"title": {"display": true, "text": "precision"}, "min": 0.4, "max": 1.05} } }
}
```

```python:average_precision.py
import numpy as np

def average_precision(is_tp, n_gt):
    """All-point AP from a score-sorted 0/1 list of TP flags (VOC 2010+ style)."""
    is_tp = np.asarray(is_tp, dtype=float)
    tp, fp = np.cumsum(is_tp), np.cumsum(1 - is_tp)
    prec, rec = tp / (tp + fp), tp / n_gt
    mrec = np.concatenate([[0.0], rec, [1.0]])
    mpre = np.concatenate([[0.0], prec, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):          # make precision monotone (the envelope)
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]        # points where recall changes
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))

flags = [1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0]
print(round(average_precision(flags, n_gt=8), 4))   # 0.6955
```

Two refinements define the modern standards. **PASCAL VOC** reports AP at a single IoU threshold of 0.5 ("AP50"). **COCO** reports **AP averaged over ten IoU thresholds from 0.50 to 0.95 in steps of 0.05** (written AP@[.5:.95], usually just "AP" or "mAP"), as well as AP50, AP75 and AP separately for small, medium and large objects, using 101-point interpolation. The averaging rewards precise localization: a detector that finds everything but draws sloppy boxes may have AP50 of 60 and AP of 30. Play with the explorer to see how classifier quality, localization quality and the IoU threshold interact.

{{demo: week06-map-explorer | Precision-recall curves, AP and COCO-style mAP | 520}}

```chart
{
  "type": "bar",
  "caption": "COCO val2017 box AP of detectors available in torchvision (pretrained weights, values from the torchvision model documentation). Backbone is ResNet-50-FPN unless noted. Accuracy depends strongly on the training recipe: the V2 rows differ from the V1 rows mainly by a better recipe and head design.",
  "height": 360,
  "data": {
    "labels": ["SSDlite320 MobileNetV3", "SSD300 VGG-16", "RetinaNet", "Faster R-CNN", "Mask R-CNN (box)", "FCOS", "RetinaNet v2", "Faster R-CNN v2"],
    "datasets": [{"label": "box AP (%)", "data": [21.3, 25.1, 36.4, 37.0, 37.9, 39.2, 41.5, 46.7], "backgroundColor": ["var(--c-teal)", "var(--c-teal)", "var(--c-blue)", "var(--c-blue)", "var(--c-blue)", "var(--c-purple)", "var(--c-green)", "var(--c-green)"]}]
  },
  "options": { "indexAxis": "y", "scales": { "x": {"title": {"display": true, "text": "COCO box AP@[.5:.95] (%)"}, "min": 0}, "y": {"title": {"display": true, "text": "model"}} } }
}
```

```chart
{
  "type": "scatter",
  "caption": "Speed versus accuracy on PASCAL VOC 2007 test, from the YOLO paper (Redmon et al., 2016), Table 1. Fast R-CNN is accurate but runs at about 0.5 fps; YOLO trades 7 points of mAP for a 90x speed-up.",
  "height": 340,
  "data": {
    "datasets": [
      {"label": "DPM (30 Hz)", "data": [{"x":30,"y":26.1}], "backgroundColor": "var(--c-muted)", "borderColor": "var(--c-muted)", "pointRadius": 7},
      {"label": "Fast YOLO", "data": [{"x":155,"y":52.7}], "backgroundColor": "var(--c-green)", "borderColor": "var(--c-green)", "pointRadius": 7},
      {"label": "YOLO", "data": [{"x":45,"y":63.4}], "backgroundColor": "var(--c-teal)", "borderColor": "var(--c-teal)", "pointRadius": 7},
      {"label": "Faster R-CNN (ZF)", "data": [{"x":18,"y":62.1}], "backgroundColor": "var(--c-amber)", "borderColor": "var(--c-amber)", "pointRadius": 7},
      {"label": "Faster R-CNN (VGG-16)", "data": [{"x":7,"y":73.2}], "backgroundColor": "var(--c-blue)", "borderColor": "var(--c-blue)", "pointRadius": 7},
      {"label": "Fast R-CNN", "data": [{"x":0.5,"y":70.0}], "backgroundColor": "var(--c-red)", "borderColor": "var(--c-red)", "pointRadius": 7}
    ]
  },
  "options": { "scales": { "x": {"type": "logarithmic", "title": {"display": true, "text": "speed (frames per second, log scale)"}}, "y": {"title": {"display": true, "text": "mAP on VOC 2007 (%)"}, "min": 20, "max": 80} } }
}
```

## Semantic and instance segmentation {#segmentation}

Segmentation predicts a label for every pixel. **Semantic segmentation** assigns each pixel a class ("road", "person") without distinguishing individual objects; **instance segmentation** separates each object ("person 1", "person 2") and gives a mask per instance; **panoptic segmentation** combines both (countable "things" as instances, uncountable "stuff" such as sky as semantic regions). The output is a tensor of shape $K\times H\times W$ (class logits per pixel), so the architecture needs a way to go from a low-resolution, semantically rich feature map back to full resolution without losing detail.

### Fully convolutional networks {#fcn}

Long, Shelhamer and Darrell (2015) made the key observation: a classification CNN becomes a dense predictor if its fully connected layers are re-interpreted as convolutions. A fully connected layer on a $7\times7\times512$ map is the same as a $7\times7$ convolution with 4096 filters; written that way, the network accepts images of any size and outputs a *grid* of class scores (a coarse heatmap, stride 32) rather than one vector. To get back to full resolution they upsample with **transposed convolutions** (learned, initialised as bilinear interpolation) and, crucially, fuse the coarse prediction with finer predictions from earlier layers: **FCN-32s** upsamples the stride-32 prediction directly (blurry), **FCN-16s** adds the stride-16 pooling layer, **FCN-8s** adds the stride-8 layer too. The principle, "deep and coarse for *what*, shallow and fine for *where*", is the template for everything after it.

:::note Transposed convolution
A transposed convolution (sometimes loosely called deconvolution) is the adjoint of a strided convolution: with stride $s$ it maps an $H\times W$ map to roughly $sH\times sW$ by inserting zeros between the inputs (or, equivalently, scattering each input value times the kernel into the output). With a kernel size not divisible by the stride its overlaps are uneven and produce checkerboard artifacts; a common fix is bilinear (or nearest) upsampling followed by an ordinary convolution, `nn.Upsample` plus `nn.Conv2d`.
:::

### U-Net {#unet}

**U-Net** (Ronneberger, Fischer and Brox, 2015) was designed for biomedical images, where annotated data are scarce (the paper used a few dozen training images and heavy elastic-deformation augmentation) and precise boundaries matter. It is a symmetric **encoder-decoder**. The encoder (contracting path) is a stack of stages, each with two $3\times3$ convolutions followed by a $2\times2$ max-pool, doubling channels as the resolution halves. The decoder (expanding path) mirrors it: a $2\times2$ up-convolution that doubles resolution and halves channels, then **concatenation with the encoder feature map of the same resolution** (the skip connection), then two $3\times3$ convolutions. A final $1\times1$ convolution produces class logits per pixel.

{{figure: week06-unet | U-Net with a 256x256 input (padded convolutions). Left half: encoder with max-pooling. Right half: decoder with up-convolutions. Dashed orange arrows are skip connections that concatenate the encoder feature map to the decoder at the same resolution. Channel counts are written on each block.}}

The skip connections do two jobs. They carry high-resolution detail (edges, thin structures) that max-pooling discarded, so the decoder can localize boundaries precisely, and they give gradients a short path to the early layers, like residual connections do. They *concatenate* rather than add, as in DenseNet, because the encoder and decoder features mean different things. The original paper used unpadded convolutions (so the output was smaller than the input and large images were segmented in overlapping tiles); almost all modern implementations pad, as in the figure. U-Net's descendants are everywhere in medical imaging, and, notably, the denoising network in diffusion models ([Week 13](week-13.html)) is a U-Net.

:::math Counting U-Net parameters (first stage)
With a 3-channel input, the first stage has two $3\times3$ convolutions with 64 filters: $3\cdot3\cdot3\cdot64 + 64 = 1{,}792$ then $3\cdot3\cdot64\cdot64 + 64 = 36{,}928$ parameters. The bottleneck stage (512 to 1024 channels, then 1024 to 1024) costs $9\cdot512\cdot1024 + 9\cdot1024\cdot1024 \approx 14.2$M, which shows where the parameters sit: deep, narrow-resolution stages, even though the *compute* is dominated by the high-resolution ones. The whole network has about 31M parameters (for this 5-level layout, without batch norm).
:::

### DeepLab: atrous convolution and ASPP {#deeplab}

A classification backbone downsamples by 32 to enlarge the receptive field, but that is precisely what hurts dense prediction. The DeepLab family (Chen, Papandreou, Kokkinos, Murphy and Yuille) attacks the problem by increasing the receptive field *without* reducing resolution, using **atrous** (dilated) convolution. A 1D atrous convolution with rate $r$ is

$$
y[i] = \sum_{k=1}^{K} x[i + r\cdot k]\;w[k],
$$

i.e. the kernel taps are spread $r$ samples apart, which inserts $r-1$ holes between the taps. A $3\times3$ kernel with rate $r$ covers an effective window of $3 + 2(r-1)$ pixels per side: $r=1$ gives $3\times3$, $r=2$ gives $5\times5$, $r=4$ gives $9\times9$, all with the same 9 weights and no downsampling. By setting the stride of the last stages of a ResNet to 1 and dilating their convolutions, the output stride drops from 32 to 16 or 8 while the receptive field is preserved.

:::example Receptive field with and without dilation
Stack three $3\times3$ convolutions with rates 1, 2, 4. The receptive field grows by $2r$ each time: $1 + 2(1 + 2 + 4) = 15$. Three plain $3\times3$ convolutions only give $7\times7$. (The sequence 1, 2, 4 also avoids the "gridding" artifact of repeating one rate, where some pixels are never sampled.)
:::

Objects appear at many scales, so **Atrous Spatial Pyramid Pooling (ASPP)** applies several parallel atrous convolutions with different rates (for example 6, 12, 18) plus a global pooling branch to the same feature map and concatenates the results, an Inception-style multi-scale module. **DeepLabv3** (2017) made this the standard head; **DeepLabv3+** (2018) added a light decoder that fuses low-level features, borrowing the U-Net idea, and used depthwise separable atrous convolutions. DeepLab originally also used a **conditional random field** as a post-processing step to sharpen boundaries; later versions dropped it because networks became accurate enough.

### Mask R-CNN: instance segmentation {#mask-rcnn}

**Mask R-CNN** (He, Gkioxari, Dollar and Girshick, 2017) extends Faster R-CNN with a third branch. For each region of interest that the box head accepts, a small fully convolutional network predicts a **binary mask** of size $28\times28$ *per class*, using features pooled with RoIAlign. The mask is then resized and pasted into the box. Two design decisions matter. First, the mask loss is a per-pixel binary cross-entropy applied only to the mask of the ground-truth class, so classes do not compete inside a pixel; classification is left to the box branch (decoupling "what" from "where"). Second, RoIAlign (see above) removes the quantization that otherwise misaligns masks. The total loss is $\mathcal{L} = \mathcal{L}_\text{cls} + \mathcal{L}_\text{box} + \mathcal{L}_\text{mask}$. It is the reference instance segmentation method and, with a keypoint head, also does human pose estimation.

### Segmentation metrics and losses {#seg-metrics}

**Pixel accuracy** is misleading under class imbalance (predicting "background" everywhere can give 95%). The standard metric is **mean IoU** (mIoU): for each class, IoU between the predicted and true pixel sets, averaged over classes. A closely related overlap measure for a binary mask is the **Dice coefficient** (F1 score on pixels):

$$
\mathrm{Dice}(A, B) = \frac{2\,|A\cap B|}{|A| + |B|} = \frac{2\,\mathrm{IoU}}{1 + \mathrm{IoU}}.
$$

(The identity on the right is easy to verify with $|A|+|B| = |A\cup B| + |A\cap B|$.) Training uses per-pixel cross-entropy, often combined with a **soft Dice loss** $1 - \frac{2\sum_i p_i g_i + \epsilon}{\sum_i p_i + \sum_i g_i + \epsilon}$, where $p_i\in[0,1]$ is the predicted foreground probability of pixel $i$ and $g_i\in\{0,1\}$ the label. The Dice term is insensitive to how many background pixels exist, which helps when the object is a tiny fraction of the image (lesions, vessels), at the cost of a less smooth optimization landscape; the combination "cross-entropy + Dice" is a safe default.

```python:seg_metrics.py
import torch

def dice_and_iou(logits, target, eps=1e-6):
    """Binary segmentation metrics. logits: (N,1,H,W), target: (N,1,H,W) in {0,1}. Returns means."""
    pred = (torch.sigmoid(logits) > 0.5).float()
    inter = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - inter
    iou = (inter + eps) / (union + eps)
    dice = (2 * inter + eps) / (pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + eps)
    return dice.mean().item(), iou.mean().item()

def soft_dice_loss(logits, target, eps=1.0):
    p = torch.sigmoid(logits)
    inter = (p * target).sum(dim=(1, 2, 3))
    return 1 - ((2 * inter + eps) / (p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + eps)).mean()
```

For the model's output tensors in the lab you will also meet `torchvision.models.segmentation.deeplabv3_resnet50` and `fcn_resnet50`, which return a dictionary with key `"out"` of shape $(N, K, H, W)$.

## Lab: fine-tune a ResNet, evaluate a detector, train a U-Net {#lab}

This lab has three parts. Part A fine-tunes a pretrained ResNet-18 on a small dataset with the three recipes, Part B runs a pretrained Faster R-CNN and evaluates it with the AP code from above, and Part C trains a small U-Net on synthetic shapes. Everything runs on a free Colab GPU (Parts A and B also work on CPU, slowly). Set the runtime to GPU first. Throughout the lab, print shapes, and compare the *recipes*, not only the final number.

### Part A: fine-tuning ResNet-18 on Flowers102 {#lab-finetune}

`Flowers102` has 102 flower categories; its official "train" split has only 10 images per class (1,020 images), which makes it a realistic small-data problem. Use the "val" split (also 1,020) for validation and a slice of "test" at the end.

```python:lab_a_data.py
import torch, time
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms as T

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

weights = models.ResNet18_Weights.IMAGENET1K_V1
mean, std = weights.transforms().mean, weights.transforms().std     # ImageNet statistics

train_tf = T.Compose([T.RandomResizedCrop(224, scale=(0.5, 1.0)), T.RandomHorizontalFlip(),
                      T.ColorJitter(0.2, 0.2, 0.2), T.ToTensor(), T.Normalize(mean, std)])
eval_tf = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize(mean, std)])

train_ds = datasets.Flowers102("data", split="train", transform=train_tf, download=True)
val_ds = datasets.Flowers102("data", split="val", transform=eval_tf, download=True)
train_dl = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2)
val_dl = DataLoader(val_ds, batch_size=64, num_workers=2)

x, y = next(iter(train_dl))
print(x.shape, y.shape, x.mean().item(), x.std().item())   # (32,3,224,224), (32,), roughly 0 and 1
```

Next, the training function. It re-applies the freeze policy every epoch (see the batch norm warning above) and evaluates on the validation split.

```python:lab_a_train.py
import copy

def build(recipe, num_classes=102):
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(512, num_classes)
    for p in model.parameters():
        p.requires_grad_(recipe == "full")                    # everything frozen unless full fine-tuning
    if recipe == "partial":
        for m in (model.layer4,):
            for p in m.parameters(): p.requires_grad_(True)
    for p in model.fc.parameters():
        p.requires_grad_(True)
    if recipe == "head":
        groups = [{"params": model.fc.parameters(), "lr": 1e-3}]
    elif recipe == "partial":
        groups = [{"params": model.fc.parameters(), "lr": 1e-3},
                  {"params": model.layer4.parameters(), "lr": 1e-4}]
    else:   # full, discriminative learning rates
        groups = [{"params": [*model.conv1.parameters(), *model.bn1.parameters(), *model.layer1.parameters()], "lr": 1e-5},
                  {"params": model.layer2.parameters(), "lr": 3e-5},
                  {"params": model.layer3.parameters(), "lr": 1e-4},
                  {"params": model.layer4.parameters(), "lr": 3e-4},
                  {"params": model.fc.parameters(), "lr": 1e-3}]
    return model.to(device), groups

def set_mode(model, recipe):
    model.train()
    if recipe == "head":
        model.eval(); model.fc.train()                        # frozen backbone must stay in eval mode
    elif recipe == "partial":
        for m in (model.conv1, model.bn1, model.layer1, model.layer2, model.layer3):
            m.eval()

@torch.no_grad()
def evaluate(model, dl):
    model.eval()
    correct = n = 0
    for x, y in dl:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item(); n += len(y)
    return correct / n

def run(recipe, epochs=8):
    model, groups = build(recipe)
    opt = torch.optim.AdamW(groups, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in groups],
                                                total_steps=epochs * len(train_dl), pct_start=0.2)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    history = []
    for ep in range(epochs):
        set_mode(model, recipe)
        t0, tot = time.time(), 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            loss = loss_fn(model(x), y)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            tot += loss.item() * len(y)
        acc = evaluate(model, val_dl)
        history.append(acc)
        print(f"{recipe:8s} epoch {ep+1}  train loss {tot/len(train_ds):.3f}  val acc {acc:.3f}  ({time.time()-t0:.0f}s)")
    return history

results = {r: run(r) for r in ["head", "partial", "full"]}
```

On this problem you should see the ordering "full > partial > head" in validation accuracy only if the learning rates are right; the linear probe is typically already good (flowers are close to ImageNet), and the interesting observations are how many epochs each recipe needs, how large the gap between train and validation is for the full recipe, and what happens when you remove the `set_mode` call or set all learning rates to $10^{-2}$ (the pretrained features are destroyed in the first epoch). Record your numbers; they vary with seed, so run at least two seeds before you conclude anything.

:::tip Cache features for the linear probe
For recipe A you can run the backbone once and train logistic regression on the 512-dimensional features. This is faster by a large factor and gives an accuracy baseline in under a minute: `feats = nn.Sequential(*list(model.children())[:-1])` then `feats(x).flatten(1)` over the dataset with `torch.no_grad()` and eval transforms (no augmentation), followed by `sklearn.linear_model.LogisticRegression`.
:::

### Part B: running and evaluating a pretrained detector {#lab-detector}

We use the Penn-Fudan pedestrian dataset (170 images, 345 annotated pedestrians, the dataset of the official PyTorch detection tutorial). Its COCO-pretrained detector already knows the class "person" (COCO label 1), so no training is needed: we run inference and compute AP@0.5 with our own code. The ground truth comes as instance masks, from which we take tight boxes.

```python:lab_b_data.py
import os, zipfile, urllib.request
import numpy as np
from PIL import Image

url = "https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip"
if not os.path.exists("PennFudanPed"):
    urllib.request.urlretrieve(url, "pf.zip")
    zipfile.ZipFile("pf.zip").extractall(".")

root = "PennFudanPed"
imgs = sorted(os.listdir(f"{root}/PNGImages"))
masks = sorted(os.listdir(f"{root}/PedMasks"))

def load(i):
    img = Image.open(f"{root}/PNGImages/{imgs[i]}").convert("RGB")
    m = np.array(Image.open(f"{root}/PedMasks/{masks[i]}"))     # 0 = background, 1..k = pedestrian ids
    boxes = []
    for obj_id in np.unique(m)[1:]:
        ys, xs = np.where(m == obj_id)
        boxes.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
    return img, np.array(boxes, dtype=float).reshape(-1, 4)

img, gt = load(0)
print(img.size, gt)          # image size (W, H) and one row per pedestrian
```

```python:lab_b_eval.py
import torch
from torchvision import models
from torchvision.transforms.functional import to_tensor

device = "cuda" if torch.cuda.is_available() else "cpu"
det_weights = models.detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1
model = models.detection.fasterrcnn_resnet50_fpn(weights=det_weights, box_score_thresh=0.05).to(device).eval()

all_scores, all_tp, n_gt = [], [], 0
with torch.no_grad():
    for i in range(0, len(imgs), 1):
        img, gt = load(i)
        out = model([to_tensor(img).to(device)])[0]           # dict: boxes (N,4), labels (N,), scores (N,)
        keep = out["labels"] == 1                              # COCO class 1 = person
        pb, ps = out["boxes"][keep].cpu().numpy(), out["scores"][keep].cpu().numpy()
        n_gt += len(gt)
        matched = set()
        for j in np.argsort(-ps):                              # per image, high score first
            ious = iou_matrix(pb[j:j + 1], gt)[0] if len(gt) else np.array([])
            k = int(ious.argmax()) if len(ious) else -1
            if k >= 0 and ious[k] >= 0.5 and k not in matched:
                matched.add(k); all_tp.append(1)               # true positive, consumes this GT box
            else:
                all_tp.append(0)                               # false positive (low IoU or duplicate)
            all_scores.append(float(ps[j]))

order = np.argsort(-np.array(all_scores))                      # then sort globally by confidence
flags = np.array(all_tp)[order]
print("detections:", len(flags), " ground truth:", n_gt)
print("AP@0.5 (person):", round(average_precision(flags, n_gt), 3))
```

(`iou_matrix` and `average_precision` are the functions defined earlier; paste them into the notebook first.) A COCO-pretrained detector scores high here, typically AP@0.5 around 0.9 or above, since the images are easy; the point is to see the full pipeline. Now investigate: (1) re-run with IoU threshold 0.75 and then 0.9, and see how the AP falls (annotation noise in the masks matters); (2) raise `box_score_thresh` to 0.5 and see recall and precision at that operating point; (3) change the NMS threshold through `box_nms_thresh=0.3` and `0.8` and inspect duplicates.

### Part C: a small U-Net on synthetic shapes {#lab-unet}

To keep the training run to a couple of minutes on any machine we generate data on the fly: each image is a noisy $64\times64$ canvas with a few random filled circles and rectangles, and the target is a three-class mask (background, circle, rectangle). This is simple enough that a small U-Net reaches high mIoU quickly, and it lets you ablate the skip connections cleanly.

```python:lab_c_unet.py
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

def make_sample(rng, size=64):
    img = rng.normal(0.0, 0.15, (3, size, size)).astype(np.float32)
    mask = np.zeros((size, size), dtype=np.int64)
    yy, xx = np.mgrid[:size, :size]
    for _ in range(rng.integers(2, 5)):
        cx, cy, r = rng.integers(10, size - 10, 2).tolist() + [int(rng.integers(5, 11))]
        col = rng.uniform(0.4, 1.0, 3).astype(np.float32)
        if rng.random() < 0.5:                                   # circle -> class 1
            m = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r; c = 1
        else:                                                    # square -> class 2
            m = (abs(xx - cx) <= r) & (abs(yy - cy) <= r); c = 2
        img[:, m] += col[:, None]; mask[m] = c
    return img, mask

class ShapesDS(torch.utils.data.Dataset):
    def __init__(self, n, seed): self.rng = np.random.default_rng(seed); self.data = [make_sample(self.rng) for _ in range(n)]
    def __len__(self): return len(self.data)
    def __getitem__(self, i): x, y = self.data[i]; return torch.from_numpy(x), torch.from_numpy(y)

def block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                         nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

class UNet(nn.Module):
    def __init__(self, n_classes=3, base=16, skips=True):
        super().__init__()
        self.skips = skips
        c = [base, base * 2, base * 4, base * 8]
        self.enc1, self.enc2, self.enc3 = block(3, c[0]), block(c[0], c[1]), block(c[1], c[2])
        self.mid = block(c[2], c[3])
        self.up3 = nn.ConvTranspose2d(c[3], c[2], 2, stride=2); self.dec3 = block(c[2] * (2 if skips else 1), c[2])
        self.up2 = nn.ConvTranspose2d(c[2], c[1], 2, stride=2); self.dec2 = block(c[1] * (2 if skips else 1), c[1])
        self.up1 = nn.ConvTranspose2d(c[1], c[0], 2, stride=2); self.dec1 = block(c[0] * (2 if skips else 1), c[0])
        self.head = nn.Conv2d(c[0], n_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)                        # (N, 16, 64, 64)
        e2 = self.enc2(F.max_pool2d(e1, 2))      # (N, 32, 32, 32)
        e3 = self.enc3(F.max_pool2d(e2, 2))      # (N, 64, 16, 16)
        m = self.mid(F.max_pool2d(e3, 2))        # (N, 128, 8, 8)
        d3 = self.up3(m); d3 = self.dec3(torch.cat([d3, e3], 1) if self.skips else d3)   # (N, 64, 16, 16)
        d2 = self.up2(d3); d2 = self.dec2(torch.cat([d2, e2], 1) if self.skips else d2)  # (N, 32, 32, 32)
        d1 = self.up1(d2); d1 = self.dec1(torch.cat([d1, e1], 1) if self.skips else d1)  # (N, 16, 64, 64)
        return self.head(d1)                     # (N, n_classes, 64, 64)

def miou(logits, y, k=3):
    pred = logits.argmax(1); ious = []
    for c in range(k):
        inter = ((pred == c) & (y == c)).sum().item(); union = ((pred == c) | (y == c)).sum().item()
        if union > 0: ious.append(inter / union)
    return float(np.mean(ious))

def train(skips, epochs=6):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tr = torch.utils.data.DataLoader(ShapesDS(1500, 0), batch_size=32, shuffle=True)
    va = torch.utils.data.DataLoader(ShapesDS(300, 1), batch_size=100)
    model = UNet(skips=skips).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    for ep in range(epochs):
        model.train()
        for x, y in tr:
            x, y = x.to(device), y.to(device)
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            scores = [miou(model(x.to(device)), y.to(device)) for x, y in va]
        print(f"skips={skips} epoch {ep+1} loss {loss.item():.3f} val mIoU {np.mean(scores):.3f}")
    return model

print(UNet()(torch.randn(2, 3, 64, 64)).shape)       # torch.Size([2, 3, 64, 64])
m_skip, m_noskip = train(True), train(False)
```

Compare the two models after a few epochs: the version with skips segments sharp corners and thin gaps; the no-skip variant produces blobby masks because the decoder only has the $8\times8$ code to reconstruct from. Then try replacing cross-entropy by cross-entropy plus soft Dice and visualize predictions with `plt.imshow(pred[0].cpu())`. Finally, change the dataset so that shapes have very different sizes and see which model degrades.

## Common mistakes and debugging tips {#common-mistakes}

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| Fine-tuned model is at chance level, or accuracy drops after the first epoch | Learning rate too large for the backbone; or labels shuffled out of sync with images | Backbone LR 10x to 100x smaller than the head; check a batch visually and overfit 20 images first |
| Pretrained model gives much worse results than expected | Wrong normalization (no ImageNet mean/std, or 0-255 range), BGR instead of RGB, wrong resolution | Use `weights.transforms()`; print `x.mean()`, `x.std()` per channel |
| "Frozen" backbone still changes; validation fluctuates wildly | BN layers left in `train()` mode (running stats update, batch stats used) | `model.eval()` for frozen parts, re-apply after each `model.train()` |
| Train accuracy 100%, validation poor, small dataset | Overfitting by unfreezing too much; too little augmentation | Freeze more, add augmentation, weight decay, early stopping, or use more data |
| Training loss does not decrease at all with a frozen backbone | Optimizer built before `requires_grad_` change, or no trainable parameters in the optimizer | Build the optimizer from the final parameter list; print the trainable parameter count |
| Detector outputs hundreds of overlapping boxes | NMS missing or threshold too high; score threshold too low | Check `box_nms_thresh`, class-wise NMS (`batched_nms`), a higher score threshold |
| Detector never detects small objects | Input too small, anchors too large, stride of the finest map too coarse | Increase input size, use FPN, adapt anchor sizes to the dataset statistics |
| mAP is near zero although drawings look right | Box format mix-up: `(x, y, w, h)` versus `(x1, y1, x2, y2)`; image-size scaling mismatch; wrong label indexing (0 is background in torchvision) | Assert `x2 > x1`, plot boxes on images, check label ids |
| mAP much higher than published numbers | Evaluating on training data, or counting duplicate detections as TP | Apply the one-match-per-GT rule; evaluate on held-out data |
| Segmentation predicts only background | Class imbalance with plain cross-entropy; labels mismatched in resize (bilinear on masks!) | Class weights or Dice loss; resize masks with *nearest-neighbour* only |
| Segmentation output is blurry or has checkerboards | Missing skip connections; transposed convolution with uneven overlap | Add skips, or use upsample plus convolution |
| Out-of-memory in a DenseNet or U-Net | Concatenation keeps many large activations | Reduce batch size, use mixed precision (`torch.autocast`), gradient checkpointing |

:::warning Always overfit a tiny batch first
Before any long run, train on 16 to 32 images until the training loss is near zero. If you cannot overfit a batch, there is a bug in the data pipeline, loss, labels or optimizer; if you can, but the full run fails, the problem is generalization or optimization, not plumbing.
:::

## Exercises {#exercises}

:::exercise Exercise 1: Gradient through a residual stack (derivation)
Consider $\boldsymbol{x}_{l+1} = \boldsymbol{x}_l + \mathcal{F}_l(\boldsymbol{x}_l)$ for $l = 0, \dots, L-1$. (a) Write $\boldsymbol{x}_L$ as a sum. (b) Using the chain rule, show that $\partial\mathcal{L}/\partial\boldsymbol{x}_0$ contains the term $\partial\mathcal{L}/\partial\boldsymbol{x}_L$ with coefficient exactly 1. (c) Why does this not guarantee that the gradient is well scaled?
:::details Solution
(a) $\boldsymbol{x}_L = \boldsymbol{x}_0 + \sum_{l=0}^{L-1} \mathcal{F}_l(\boldsymbol{x}_l)$ by telescoping. (b) Differentiating, $\partial\boldsymbol{x}_L/\partial\boldsymbol{x}_0 = \mathbf{I} + \partial/\partial\boldsymbol{x}_0 \sum_l \mathcal{F}_l(\boldsymbol{x}_l)$, so $\partial\mathcal{L}/\partial\boldsymbol{x}_0 = \partial\mathcal{L}/\partial\boldsymbol{x}_L\,(\mathbf{I} + \dots)$. The identity part passes the gradient unchanged to every layer. (c) The other terms can be large or have arbitrary sign, and their number grows with $L$ (the sum has up to $L$ terms and each $\boldsymbol{x}_l$ depends on earlier branches). In the worst case, the gradient norm grows like a product of $(1+\epsilon_l)$, which is why we initialise the last BN scale of each branch at zero or scale branches by a small constant.
:::
:::

:::exercise Exercise 2: Parameter counting
(a) Count the weights (no biases or BN) of a ResNet basic block with 128 input and output channels, and of a bottleneck block with 512 input/output channels and bottleneck width 128. (b) Count the weights of a MobileNetV2 inverted residual with 32 input and output channels and expansion factor 6. (c) Compare each to a plain $3\times3$ convolution from $C$ to $C$ channels with the same $C$.
:::details Solution
(a) Basic: $2\cdot 9\cdot128\cdot128 = 294{,}912$. Bottleneck: $512\cdot128 + 9\cdot128\cdot128 + 128\cdot512 = 65{,}536 + 147{,}456 + 65{,}536 = 278{,}528$. (b) Expand $32\cdot192 = 6144$; depthwise $9\cdot192 = 1728$; project $192\cdot32 = 6144$; total $14{,}016$. (c) A single plain $3\times3$ with $C=128$ has 147,456 weights, with $C=512$ 2.36M, and with $C=32$ 9,216. So the bottleneck block works on a 512-wide stream for about the cost of one $128$-channel $3\times3$ convolution plus the projections, and the inverted residual is only 1.5 times the cost of one dense $32\to32$ convolution but with a 192-channel internal space.
:::
:::

:::exercise Exercise 3: Depthwise separable cost ratio
Derive the ratio $1/N + 1/D_K^2$ for the cost of a depthwise separable convolution relative to a standard one. For $D_K = 3$ and $N = 256$, how many times cheaper is it? What happens when $D_K = 7$ (as in ConvNeXt)?
:::details Solution
Standard: $D_K^2 M N D_F^2$ multiply-adds. Depthwise: each of the $M$ channels is filtered by one $D_K\times D_K$ kernel, $D_K^2 M D_F^2$. Pointwise: $M N D_F^2$. The ratio is $(D_K^2 M D_F^2 + M N D_F^2)/(D_K^2 M N D_F^2) = 1/N + 1/D_K^2$. For $D_K=3$, $N=256$: $0.0039 + 0.1111 = 0.115$, i.e. 8.7 times cheaper. For $D_K = 7$: $1/256 + 1/49 = 0.0243$, about 41 times cheaper than a dense $7\times7$ convolution, which is why large kernels are affordable only in depthwise form (a dense $7\times7$ would be 5.4 times more expensive than a dense $3\times3$).
:::
:::

:::exercise Exercise 4: Compound scaling
With $\alpha=1.2$, $\beta=1.1$, $\gamma=1.15$ and baseline resolution 224, compute depth, width and resolution multipliers and the FLOPs multiplier for $\phi = 3$. Then suppose you instead spend the same FLOPs budget on depth alone: how deep would the network be? Why might this be worse?
:::details Solution
$d = 1.2^3 = 1.728$, $w = 1.1^3 = 1.331$, $r = 1.15^3 = 1.521$, so resolution $\approx 224\cdot1.521 = 341$. FLOPs multiplier $= d\,w^2\,r^2 = 1.728\cdot1.7716\cdot2.3131 \approx 7.08$ (compare with $2^3=8$; the discrepancy comes from $\alpha\beta^2\gamma^2 = 1.92$, not exactly 2). Depth alone would need a multiplier of about 7.08, i.e. a network 7 times deeper at the same resolution and width. The accuracy of very deep narrow networks saturates (the paper's single-dimension scaling curves flatten), the optimization difficulty grows, and latency grows linearly because layers are sequential, whereas width and resolution parallelize well on GPUs.
:::
:::

:::exercise Exercise 5: Choosing a transfer recipe
For each case choose one of linear probe, partial fine-tuning or full fine-tuning and justify it (mention learning rates and BN): (a) 40 images per class of 10 types of mushrooms (photos); (b) 20,000 labelled chest X-rays; (c) 500 images of defects on a metal surface in a single factory, grey-scale; (d) a new classification task with 50,000 natural images.
:::details Solution
(a) Linear probe first (cache features); photos are close to ImageNet and the dataset is tiny, so the risk is overfitting. If accuracy is insufficient, unfreeze the last stage with a small LR. (b) Full fine-tuning with discriminative learning rates: large domain gap (X-rays), plenty of data. Replicate the grey channel to 3 channels or sum the conv1 filters. Keep BN in train mode only if the batch size is large enough (at least 16 to 32). (c) Partial fine-tuning (layer3/layer4 and head) with strong augmentation (flips, rotations, brightness); small data but a clear domain shift. Normalize inputs with the ImageNet statistics after converting to 3 channels. (d) Full fine-tuning: enough data and a natural-image domain; start with a short head-only phase (warmup) and then train everything with LR about $10^{-4}$ (AdamW) for the backbone.
:::
:::

:::exercise Exercise 6: IoU and NMS by hand
Boxes (x1, y1, x2, y2) with scores: A = (0, 0, 10, 10) score 0.9; B = (2, 0, 12, 10) score 0.8; C = (6, 0, 16, 10) score 0.7; D = (30, 30, 40, 40) score 0.6. (a) Compute IoU(A,B), IoU(A,C), IoU(B,C). (b) Run greedy NMS with $\tau = 0.5$ and then $\tau=0.7$. (c) With $\tau = 0.5$, would C survive if B had not existed?
:::details Solution
(a) A and B overlap in $8\times10=80$; union $=100+100-80=120$, IoU $=0.667$. A and C: overlap $4\times10=40$, union $160$, IoU $=0.25$. B and C: overlap $6\times10 = 60$, union $140$, IoU $=0.429$. (b) With $\tau=0.5$: keep A; B has IoU 0.667 > 0.5 with A and is removed; C has 0.25, stays pending; the next is C: keep (D does not overlap). Output: A, C, D. With $\tau=0.7$: A is kept, B (0.667 < 0.7) is not suppressed and C is not either; the next is B: IoU(B,C) $=0.429 < 0.7$; so everything survives: A, B, C, D. (c) Yes: without B, C only overlaps A with IoU 0.25 and survives. Note that C survives even with B present when $\tau=0.5$, because B was suppressed by A before it could suppress C; NMS order matters.
:::
:::

:::exercise Exercise 7: Compute AP (challenging)
A class has 5 ground-truth boxes. Sorted detections have TP flags 1, 0, 1, 1, 0, 0, 1 (the fifth ground-truth object is never found). (a) Compute precision and recall at every rank. (b) Compute the envelope and the all-point AP. (c) Compute the 11-point AP. (d) Suppose the unmatched ground-truth object had instead been found by an extra detection at the very end of the list (rank 8). How much does AP change, and what does that tell you about the metric?
:::details Solution
(a) Ranks 1 to 7: TP cumulative 1, 1, 2, 3, 3, 3, 4; precision $1, 0.5, 0.667, 0.75, 0.6, 0.5, 0.571$; recall $0.2, 0.2, 0.4, 0.6, 0.6, 0.6, 0.8$. (b) Envelope (max precision at recall $\ge r$): at recall 0.2: $1.0$; 0.4: $0.75$ (since rank 4 reached 0.75 at recall 0.6 and 0.667 at 0.4, so the max over ranks $\ge$ 3 is 0.75); 0.6: 0.75; 0.8: 0.571. AP $= 0.2\cdot1.0 + 0.2\cdot0.75 + 0.2\cdot0.75 + 0.2\cdot0.571 = 0.2 + 0.15 + 0.15 + 0.114 = 0.614$. (c) At recall thresholds $0, 0.1, 0.2$: precision 1.0 (3 points, 3.0); $0.3, 0.4$: 0.75 (1.5); $0.5, 0.6$: 0.75 (1.5); $0.7, 0.8$: 0.571 (1.143); $0.9, 1.0$: 0 (0). Sum $= 7.143$, divided by 11 gives $0.649$. (d) The 8th detection (a TP) reaches precision $5/8 = 0.625$ at recall 1.0, which also lifts the envelope at recall 0.8 from 0.571 to 0.625. AP rises from $0.614$ to $0.75$; even a low-confidence true positive at the tail adds a lot. The metric therefore rewards *recall at any precision*; this is why mAP can be increased by lowering the score threshold to output many extra low-confidence boxes.
:::
:::

:::exercise Exercise 8: Atrous convolution (derivation and coding)
(a) Show that a $3\times3$ convolution with dilation $r$ has an effective kernel size of $3 + 2(r-1)$, and give the receptive field of a stack of $n$ such layers with rates $r_1,\dots,r_n$. (b) A ResNet-50 has an output stride of 32. To get output stride 8 you change the strides of layer3 and layer4 to 1; what dilation rates must the $3\times3$ convolutions in those stages use so that the receptive field is preserved? (c) In PyTorch, which argument of `nn.Conv2d` implements it, and what padding keeps the size?
:::details Solution
(a) Taps are at offsets $-r, 0, r$ in each axis, so the extent is $2r + 1 = 3 + 2(r-1)$. Each layer of this kind increases the receptive field by $2r_i$, so the stack has $1 + 2\sum_i r_i$. (b) Removing a stride of 2 shrinks the "step" for all following layers by 2, so their convolutions need twice the dilation: layer3 uses rate 2, and layer4 (after both strides removed, a factor of 4) uses rate 4. (torchvision's `replace_stride_with_dilation=[False, True, True]` does this, and DeepLab uses it.) (c) `nn.Conv2d(c, c, 3, padding=r, dilation=r)`; padding equal to the dilation keeps the spatial size for a $3\times3$ kernel.
:::
:::

:::exercise Exercise 9: What would happen if... (conceptual)
Predict and explain: (a) you fine-tune a pretrained ResNet with batch size 2 and BN in train mode; (b) you freeze a backbone with `requires_grad_(False)` but forget `.eval()`; (c) in Mask R-CNN you replace RoIAlign with RoI pooling; (d) you set the focal loss $\gamma$ to 0; (e) you remove skip connections from a U-Net.
:::details Solution
(a) BN statistics computed from two images are extremely noisy, training is unstable and the running statistics become corrupt; use GroupNorm, freeze BN, or accumulate larger batches (SyncBN across GPUs). (b) The weights stay fixed but the running mean/variance update with every batch, so evaluation statistics drift away from those the head was trained with, and the training forward pass uses batch statistics: noisy features, non-reproducible results. (c) Boxes are slightly worse and masks noticeably worse because quantization shifts features by up to half a stride (up to 16 pixels in the input at stride 32), harmful at high IoU thresholds. (d) Focal loss becomes ordinary (alpha-weighted) cross-entropy, and the many easy negatives dominate the loss again: lower accuracy, with the detector biased toward background unless the prior is initialised carefully. (e) The decoder lacks high-resolution detail, so predictions are coarse and boundaries blur, and the small-object recall drops.
:::
:::

:::exercise Exercise 10: Design a detector head (challenging)
A single-scale one-stage detector uses a $32\times32$ feature map, 5 anchors per location and 20 classes with sigmoid classification. (a) What is the shape of the classification and box-regression output tensors? (b) How many anchors per image, and how many positives if an image has 6 objects and each matches about 4 anchors? What is the positive:negative ratio and what does it imply for the loss? (c) Show that with a prior probability $\pi=0.01$ for foreground, the bias of the last classification layer should be initialised to $-\log\frac{1-\pi}{\pi}$.
:::details Solution
(a) Classification: $(N, 5\cdot20, 32, 32) = (N,100,32,32)$ (or $(N, 5120, 20)$ after reshaping); regression: $(N, 5\cdot4, 32, 32)$. (b) $32\cdot32\cdot5 = 5120$ anchors; about 24 positives, so 5096 negatives: roughly 1:212. The plain loss is dominated by negatives: use focal loss, hard negative mining, or reweighting. (c) The output probability is $\sigma(b)$ at initialization (zero weights contribute nothing in expectation), and we want $\sigma(b)=\pi$, so $b = \log\frac{\pi}{1-\pi} = -\log\frac{1-\pi}{\pi} \approx -4.6$ for $\pi=0.01$. RetinaNet does this to avoid a huge loss and instability in the first iterations when 99%+ of anchors are negative.
:::
:::

## Key takeaways {#key-takeaways}

:::key Key takeaways
- Depth hurts a plain network through optimization (the *degradation* problem), not capacity. A residual block $\boldsymbol{y}=\boldsymbol{x}+\mathcal{F}(\boldsymbol{x})$ gives an identity path for activations and gradients, and its unrolled form is an ensemble of paths of many lengths.
- Bottlenecks (ResNet, Inception) and depthwise separable convolutions (MobileNet) are two ways to cut FLOPs; the inverted residual keeps skip connections between the *narrow* tensors and applies nonlinearities only in the wide space.
- Compound scaling grows depth, width and resolution together ($d=\alpha^\phi$, $w=\beta^\phi$, $r=\gamma^\phi$, with $\alpha\beta^2\gamma^2\approx2$ so FLOPs double per $\phi$). ConvNeXt shows that much of the Transformer's advantage came from the recipe and macro-design, not attention.
- Transfer learning is the default. Choose linear probe, partial or full fine-tuning according to data size and domain gap; use ImageNet normalization, small backbone learning rates, discriminative learning rates, and keep frozen BN layers in eval mode.
- Detection = candidate boxes + classification + box regression + duplicate removal. R-CNN to Faster R-CNN shared and then learned the proposals; SSD, YOLO and RetinaNet predict densely (focal loss fixes class imbalance); FCOS and CenterNet drop anchors; DETR drops NMS.
- IoU measures overlap; NMS keeps the best box per cluster; AP is the area under the precision-recall envelope per class and mAP averages over classes (COCO also over IoU thresholds 0.5 to 0.95).
- Segmentation needs both context and resolution: FCN (convolutionalization and skip fusion), U-Net (encoder-decoder with concatenated skips), DeepLab (atrous convolutions and ASPP), Mask R-CNN (a per-RoI mask branch with RoIAlign). Evaluate with mIoU or Dice, never pixel accuracy alone.
:::

:::note Assignment 2
**Assignment 2** is due at the end of Week 8. The convolutional-network and transfer-learning material of Weeks 5 and 6 (training a CNN, fine-tuning a pretrained model and analysing the result) is the ground for it; plan time for it alongside the Week 7 and Week 8 material.
:::

## Further reading {#further-reading}

Essential (marked with a star):

- (*) Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. "Deep Residual Learning for Image Recognition." CVPR 2016 (arXiv:1512.03385). The original ResNet paper, short and clear.
- (*) Mingxing Tan, Quoc V. Le. "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks." ICML 2019 (arXiv:1905.11946).
- (*) Olaf Ronneberger, Philipp Fischer, Thomas Brox. "U-Net: Convolutional Networks for Biomedical Image Segmentation." MICCAI 2015 (arXiv:1505.04597).
- (*) Joseph Redmon, Santosh Divvala, Ross Girshick, Ali Farhadi. "You Only Look Once: Unified, Real-Time Object Detection." CVPR 2016 (arXiv:1506.02640).

Architectures:

- Kaiming He et al. "Identity Mappings in Deep Residual Networks." ECCV 2016 (arXiv:1603.05027).
- Saining Xie et al. "Aggregated Residual Transformations for Deep Neural Networks" (ResNeXt). CVPR 2017 (arXiv:1611.05431).
- Gao Huang, Zhuang Liu, Laurens van der Maaten, Kilian Weinberger. "Densely Connected Convolutional Networks." CVPR 2017 (arXiv:1608.06993).
- Christian Szegedy et al. "Going Deeper with Convolutions." CVPR 2015 (arXiv:1409.4842); and "Rethinking the Inception Architecture for Computer Vision." CVPR 2016 (arXiv:1512.00567).
- Andrew Howard et al. "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications." 2017 (arXiv:1704.04861).
- Mark Sandler et al. "MobileNetV2: Inverted Residuals and Linear Bottlenecks." CVPR 2018 (arXiv:1801.04381).
- Zhuang Liu et al. "A ConvNet for the 2020s." CVPR 2022 (arXiv:2201.03545).
- Andreas Veit, Michael Wilber, Serge Belongie. "Residual Networks Behave Like Ensembles of Relatively Shallow Networks." NeurIPS 2016 (arXiv:1605.06431).

Transfer learning:

- Jason Yosinski, Jeff Clune, Yoshua Bengio, Hod Lipson. "How transferable are features in deep neural networks?" NeurIPS 2014 (arXiv:1411.1792).
- Simon Kornblith, Jonathon Shlens, Quoc Le. "Do Better ImageNet Models Transfer Better?" CVPR 2019 (arXiv:1805.08974).
- Kaiming He, Ross Girshick, Piotr Dollar. "Rethinking ImageNet Pre-training." ICCV 2019 (arXiv:1811.08883).
- Jeremy Howard, Sebastian Ruder. "Universal Language Model Fine-tuning for Text Classification." ACL 2018 (arXiv:1801.06146), for discriminative learning rates and gradual unfreezing.

Detection and segmentation:

- Ross Girshick et al. "Rich feature hierarchies for accurate object detection and semantic segmentation" (R-CNN). CVPR 2014 (arXiv:1311.2524); Ross Girshick, "Fast R-CNN," ICCV 2015 (arXiv:1504.08083).
- Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun. "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks." NeurIPS 2015 (arXiv:1506.01497).
- Wei Liu et al. "SSD: Single Shot MultiBox Detector." ECCV 2016 (arXiv:1512.02325).
- Tsung-Yi Lin et al. "Feature Pyramid Networks for Object Detection." CVPR 2017 (arXiv:1612.03144); and "Focal Loss for Dense Object Detection." ICCV 2017 (arXiv:1708.02002).
- Zhi Tian, Chunhua Shen, Hao Chen, Tong He. "FCOS: Fully Convolutional One-Stage Object Detection." ICCV 2019 (arXiv:1904.01355).
- Nicolas Carion et al. "End-to-End Object Detection with Transformers" (DETR). ECCV 2020 (arXiv:2005.12872).
- Jonathan Long, Evan Shelhamer, Trevor Darrell. "Fully Convolutional Networks for Semantic Segmentation." CVPR 2015 (arXiv:1411.4038).
- Liang-Chieh Chen et al. "Rethinking Atrous Convolution for Semantic Image Segmentation" (DeepLabv3), 2017 (arXiv:1706.05587).
- Kaiming He, Georgia Gkioxari, Piotr Dollar, Ross Girshick. "Mask R-CNN." ICCV 2017 (arXiv:1703.06870).
- Tsung-Yi Lin et al. "Microsoft COCO: Common Objects in Context." ECCV 2014 (arXiv:1405.0312); and Mark Everingham et al., "The PASCAL Visual Object Classes (VOC) Challenge," IJCV 2010, for the AP definitions.
- The torchvision documentation and the detection reference scripts (pytorch.org/vision) and the CS231n lecture notes (Stanford).
