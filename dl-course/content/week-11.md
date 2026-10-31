---
title: "Vision Transformers, Multimodal Models, and Self-Supervised Learning"
phase: Attention & Pretrained Models
summary: ViT, DeiT and Swin with exact shapes and the CNN-versus-ViT inductive-bias debate; DETR and SAM; self-supervised learning from pretext tasks to SimCLR, MoCo, BYOL, DINO and MAE (with an InfoNCE derivation); CLIP and zero-shot transfer; captioning, VQA and multimodal LLMs; and a lab that fine-tunes a ViT, runs CLIP retrieval and linear-probes self-supervised features.
---

:::key Learning objectives
By the end of this week you should be able to:
- Trace a $224\times224$ image through a ViT-B/16 and state the exact tensor shape after patch embedding, after adding the class token and position embeddings, inside each attention head, and at the classifier.
- Explain what a CNN builds in (locality, weight sharing, translation equivariance, hierarchy), what a plain ViT must learn from data, and summarize the evidence about data efficiency, including how DeiT and Swin recover some of the missing structure.
- Describe DETR as set prediction with Hungarian matching and give the high-level design of the Segment Anything Model.
- Derive the InfoNCE loss as a bound on mutual information, explain the role of temperature, and compare contrastive (SimCLR, MoCo), non-contrastive (BYOL, DINO) and masked-modeling (MAE) objectives, including how each avoids representational collapse.
- Train and use CLIP-style models: write the symmetric contrastive loss, perform zero-shot classification with prompt templates, and perform image-text retrieval.
- Sketch how a vision encoder is connected to a large language model (projector, Q-Former, cross-attention) and how captioning and visual question answering fit that picture.
- Fine-tune a ViT, evaluate CLIP zero-shot, and linear-probe self-supervised features with Hugging Face, torchvision and open_clip.
:::

[Week 9](week-09.html) built the Transformer for text, and [Week 10](week-10.html) showed what happens when you pretrain it on enormous amounts of unlabeled text with nothing but a prediction objective. Two questions follow directly. First: does the architecture need to be changed for images, or can we hand it image patches and let it learn? Second: the success of language models came from *self-supervision*, learning from the structure of the data itself, so can vision do the same, and can we even tie the two modalities together?

The answers shaped the last five years of the field. A nearly unmodified Transformer works for images once there is enough data (ViT), and tricks can reduce the amount of data needed (DeiT, Swin). Labels are not needed to learn good visual features (SimCLR, BYOL, DINO, MAE). And pairing images with the natural-language text that already accompanies them on the web gives a representation that can classify categories nobody labeled (CLIP), which in turn becomes the "eyes" of multimodal chat models. This week follows that line from architecture, to objective, to alignment, to systems.

:::note A map of the week
Three ideas recur. **Tokens**: any signal that can be cut into pieces can become a sequence for a Transformer. **Invariance**: self-supervised learning works by telling the network which changes to the input *should not* change the output (two crops of one photo, an image and its caption). **Collapse**: any objective that only asks for similarity has a trivial solution (output a constant), and every method this week is, at heart, a different way of ruling that solution out.
:::

## Vision Transformers {#vit}

### From convolutions to patches {#from-conv}

For a decade, computer vision meant convolutional networks ([Week 5](week-05.html), [Week 6](week-06.html)). Convolutions are a very good fit for images: a small filter slides over the picture (locality), the same filter is used everywhere (weight sharing), and so a shifted input produces a shifted output (translation equivariance). Stacking them with pooling builds a hierarchy of increasingly large, increasingly abstract features.

Several groups tried to put attention into vision before 2020, but as an *addition* to convolutions (non-local blocks, attention-augmented convolutions) or with restricted attention windows that were awkward to run at scale. The obstacle to using full self-attention on pixels is arithmetic. A $224\times224$ image has $50{,}176$ pixels, and self-attention over that many positions needs a score matrix with $2.5\times10^9$ entries per head per layer. Nobody can afford that.

Dosovitskiy and colleagues at Google (the ViT paper, "An Image is Worth 16x16 Words", ICLR 2021) took the most direct possible route around the cost: **do not treat pixels as tokens; treat non-overlapping patches as tokens**. A $16\times16$ patch is $768$ numbers (three colour channels). A $224\times224$ image becomes $14\times14=196$ patches, and the attention matrix is a modest $197\times197$. After that, the model is the Transformer encoder of [Week 9](week-09.html) with essentially no vision-specific machinery.

:::history Why 2020, not 2017?
Nothing about the idea required new mathematics, and the Transformer had existed since 2017. What changed was data and compute. The ViT paper's central finding is that this simple model, trained on ImageNet-1k alone (1.3 million images), is *worse* than a comparable ResNet, but trained on JFT-300M (a large private Google dataset of about 300 million labeled images) it matches or beats the best ResNets at a fraction of the pretraining compute. The architecture had to wait until pretraining at scale was routine. Image GPT (Chen et al., 2020) had earlier shown that a Transformer on raw pixel sequences learns useful features, but at tiny resolution because of the quadratic cost.
:::

### Patch embedding with exact shapes {#patch-embedding}

Let an image batch be $\boldsymbol{X}\in\mathbb{R}^{B\times C\times H\times W}$ with $C=3$, $H=W=224$, and choose a patch size $P=16$. The number of patches is $N = HW/P^2 = 196$. Each patch, flattened in channel-last or channel-first order, is a vector in $\mathbb{R}^{P^2C}=\mathbb{R}^{768}$.

A learned linear map $\boldsymbol{E}\in\mathbb{R}^{(P^2C)\times D}$ projects every flattened patch to the model width $D$ (also $768$ in ViT-Base, which is a coincidence of the base configuration, not a requirement; ViT-Large has $P^2C=768$ and $D=1024$). The same $\boldsymbol{E}$ is used for every patch, so this is exactly a convolution with kernel size $P$, stride $P$ and $D$ output channels. That is how every real implementation does it:

$$
\underbrace{B\times 3\times224\times224}_{\text{image}}
\xrightarrow{\ \text{Conv2d}(3,D,k{=}16,s{=}16)\ }
\underbrace{B\times D\times14\times14}_{\text{patch grid}}
\xrightarrow{\ \text{flatten, transpose}\ }
\underbrace{B\times196\times D}_{\text{tokens}} .
$$

The patch embedding is the only place where the 2-D structure of the image is touched. After it, tokens are an unordered set until position information is added.

{{figure: week11-patchify | A $4\times4$ patch grid unrolled into a raster-order sequence. Two patches that are vertical neighbours in the image (0 and 4) end up four tokens apart in the sequence, which is why the model needs position embeddings that carry 2-D information.}}

### The class token {#class-token}

A classifier needs a single vector summarizing the whole image. ViT follows BERT ([Week 10](week-10.html)): prepend one extra learnable vector $\boldsymbol{x}_{\text{cls}}\in\mathbb{R}^{D}$ to the sequence. It starts out containing no information about the image, but it attends to all patches at every layer, and the final-layer output at that position, $\boldsymbol{z}_{\text{cls}}^{(L)}$, is fed to the classification head. The sequence length becomes $N+1=197$.

The class token is a convenience, not a necessity. Global average pooling over the final patch outputs works equally well when the learning rate is tuned for it (the ViT paper reports this, and many later models such as Swin use pooling). The token's real attraction is that it keeps the "readout" separate from the patch representations so that, for example, a distillation token (next section) or a register token can be added in the same way.

### Position embeddings {#position-embeddings}

Self-attention is permutation-equivariant ([Week 9](week-09.html#positions)): shuffle the patches and the outputs are shuffled in the same way, so without extra information the model sees a bag of patches. ViT adds a learned embedding $\boldsymbol{E}_{\text{pos}}\in\mathbb{R}^{(N+1)\times D}$ to the tokens, one vector per position *in raster order*, with no 2-D structure imposed. This is a surprising design, since it forces the model to discover that token 17 sits directly below token 3. The ViT authors tried 2-D-aware embeddings (separate row and column components) and relative variants and found little difference once training was long enough, and visualizing the learned embeddings shows they discover the grid on their own: the cosine similarity of one position's embedding with all others is high for nearby patches in the same row and column.

Learned embeddings have a practical cost. If you fine-tune at a higher resolution (say $384\times384$, so a $24\times24$ grid), the number of patches changes and the table has the wrong size. The standard fix is to reshape the pretrained table into a $14\times14\times D$ grid, interpolate it bilinearly or bicubically to $24\times24$, and flatten again. The class-token embedding is kept as is. This "resolution interpolation" is the only place where ViT quietly assumes 2-D structure.

{{demo: week11-vit-patchify | Patch size, sequence length and position-embedding geometry | 520}}

The full forward pass is therefore, for layer index $\ell=1,\dots,L$:

$$
\begin{aligned}
\boldsymbol{z}_0 &= \big[\boldsymbol{x}_{\text{cls}};\ \boldsymbol{x}_1\boldsymbol{E};\ \dots;\ \boldsymbol{x}_N\boldsymbol{E}\big] + \boldsymbol{E}_{\text{pos}} &&\in\mathbb{R}^{(N+1)\times D},\\
\boldsymbol{z}'_\ell &= \boldsymbol{z}_{\ell-1} + \mathrm{MSA}\big(\mathrm{LN}(\boldsymbol{z}_{\ell-1})\big),\\
\boldsymbol{z}_\ell &= \boldsymbol{z}'_\ell + \mathrm{MLP}\big(\mathrm{LN}(\boldsymbol{z}'_\ell)\big),\\
\boldsymbol{y} &= \mathrm{LN}\big(\boldsymbol{z}_L^{(0)}\big)\in\mathbb{R}^{D}, \qquad \text{logits}=\boldsymbol{y}\boldsymbol{W}_{\text{head}}.
\end{aligned}
$$

This is a **pre-norm** Transformer ([Week 9](week-09.html#pre-post-norm)); the MLP uses GELU and expands the width by 4. Inside multi-head self-attention with $h=12$ heads, each head has dimension $D/h=64$: queries, keys and values have shape $B\times12\times197\times64$, the score tensor has shape $B\times12\times197\times197$.

### A ViT in PyTorch {#vit-torch}

The following is a complete, small ViT. It is deliberately close to the paper and to `timm`'s reference implementation, and the shape comments are the point. It uses PyTorch's fused `scaled_dot_product_attention`.

```python:vit.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbed(nn.Module):
    """Image (B, C, H, W) -> tokens (B, N, D) with N = (H/P) * (W/P)."""
    def __init__(self, img_size=224, patch=16, in_ch=3, dim=768):
        super().__init__()
        self.grid = img_size // patch
        self.num_patches = self.grid ** 2
        # a strided convolution is exactly "flatten each patch and apply a shared linear map"
        self.proj = nn.Conv2d(in_ch, dim, kernel_size=patch, stride=patch)

    def forward(self, x):
        x = self.proj(x)                      # (B, D, H/P, W/P)
        return x.flatten(2).transpose(1, 2)   # (B, N, D)


class Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        assert dim % heads == 0
        self.h, self.dh = heads, dim // heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, x):                     # x: (B, T, D)
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]      # each (B, h, T, dh)
        o = F.scaled_dot_product_attention(q, k, v)       # (B, h, T, dh); softmax(QK^T / sqrt(dh)) V
        return self.out(o.transpose(1, 2).reshape(B, T, D))


class Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4.0):
        super().__init__()
        self.n1, self.attn = nn.LayerNorm(dim, eps=1e-6), Attention(dim, heads)
        self.n2 = nn.LayerNorm(dim, eps=1e-6)
        hid = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hid), nn.GELU(), nn.Linear(hid, dim))

    def forward(self, x):
        x = x + self.attn(self.n1(x))         # pre-norm residual
        return x + self.mlp(self.n2(x))


class ViT(nn.Module):
    def __init__(self, img_size=224, patch=16, dim=768, depth=12, heads=12, num_classes=1000):
        super().__init__()
        self.patch = PatchEmbed(img_size, patch, 3, dim)
        n = self.patch.num_patches
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, n + 1, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):                     # x: (B, 3, 224, 224)
        t = self.patch(x)                     # (B, 196, 768)
        t = torch.cat([self.cls.expand(t.shape[0], -1, -1), t], dim=1)   # (B, 197, 768)
        t = t + self.pos                      # broadcast over batch
        for blk in self.blocks:
            t = blk(t)                        # shape unchanged: (B, 197, 768)
        return self.head(self.norm(t)[:, 0])  # classify from the class token: (B, 1000)


model = ViT()
x = torch.randn(2, 3, 224, 224)
print(model(x).shape)                                     # torch.Size([2, 1000])
print(sum(p.numel() for p in model.parameters()) / 1e6)   # about 86.6 (million)
```

A useful check of the claim "the patch embedding is a convolution" is to compare it with an explicit reshape followed by a matrix multiplication, using the same weights:

```python:patch_equivalence.py
import torch
import torch.nn as nn

P, D = 16, 32
conv = nn.Conv2d(3, D, kernel_size=P, stride=P, bias=False)
x = torch.randn(2, 3, 64, 64)

# route 1: conv
a = conv(x).flatten(2).transpose(1, 2)                    # (2, 16, 32)

# route 2: explicit patchify (unfold) followed by one shared matrix multiply
patches = x.unfold(2, P, P).unfold(3, P, P)               # (2, 3, 4, 4, P, P)
patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(2, 16, 3 * P * P)   # (B, N, C*P*P)
W = conv.weight.reshape(D, 3 * P * P)                     # same weights, flattened as (C, P, P)
b = patches @ W.T                                         # (2, 16, 32)

print(torch.allclose(a, b, atol=1e-5))                    # True
```

### Parameters, compute and the ViT family {#vit-family}

The ViT paper defined Base, Large and Huge variants by width, depth and head count; later work added the smaller Tiny and Small sizes (Touvron et al., DeiT). The names are written `ViT-B/16`, meaning the Base model with patch size 16. For a width $D$, depth $L$ and MLP ratio 4, each block holds about $4D^2$ attention weights and $8D^2$ MLP weights, so $\approx12LD^2$ parameters in the blocks, plus the patch embedding, position embeddings and head.

| Model | Layers $L$ | Width $D$ | MLP width | Heads | Params |
| --- | --- | --- | --- | --- | --- |
| ViT-Ti | 12 | 192 | 768 | 3 | 5.7M |
| ViT-S | 12 | 384 | 1536 | 6 | 22M |
| ViT-B | 12 | 768 | 3072 | 12 | 86M |
| ViT-L | 24 | 1024 | 4096 | 16 | 307M |
| ViT-H | 32 | 1280 | 5120 | 16 | 632M |

(ViT-Ti and ViT-S head counts follow DeiT; ViT-B, L, H follow the original paper. Our own count with a 1000-class head and 197 positions gives the numbers in the chart below.)

```chart
{
  "type": "bar",
  "caption": "Parameter counts (millions) of ViT variants computed from the formulas above (16x16 patches, 224x224 input, 1000-class head). The sum of block weights, about 12 L D^2, dominates.",
  "height": 280,
  "data": { "labels": ["ViT-Ti", "ViT-S", "ViT-B", "ViT-L", "ViT-H"], "datasets": [ { "label": "parameters (M)", "data": [5.7, 22.1, 86.6, 304.3, 632.2], "backgroundColor": "var(--c-blue)" } ] },
  "options": { "scales": { "x": {"title": {"display": true, "text": "model"}}, "y": {"title": {"display": true, "text": "parameters (millions)"}} } }
}
```

The compute story is more interesting than the parameter count because it depends on the number of tokens. For a block with $n=N+1$ tokens and width $D$, the multiply-accumulates are about $12nD^2$ for the linear layers (QKV and output projection give $4nD^2$, the MLP gives $8nD^2$) plus $2n^2D$ for the attention scores and the weighted sum. At $224^2$ with $P=16$, $n=197$ and $D=768$ the linear term is $\approx 1.4\times10^9$ per block against $\approx 0.06\times10^9$ for attention: the layers are dominated by the *matrix multiplications over tokens*, not by the quadratic term. At higher resolutions this flips:

```chart
{
  "type": "line",
  "caption": "FLOPs (multiplies and adds counted separately, GFLOPs) for ViT-B/16 as the input resolution grows: linear layers (QKV, projection, MLP) scale linearly in tokens, global attention scales quadratically, windowed attention (7x7 windows, as in Swin) stays linear. Computed from the formulas in the text; ignores LayerNorm, softmax and the head.",
  "height": 340,
  "data": { "labels": [224, 320, 384, 512, 768, 1024], "datasets": [
    { "label": "linear layers (QKV, proj, MLP)", "data": [33.5, 68.1, 98.0, 174.1, 391.5, 696.0], "borderColor": "var(--c-blue)", "pointRadius": 3 },
    { "label": "global attention (QK^T and AV)", "data": [1.4, 5.9, 12.3, 38.7, 195.9, 618.8], "borderColor": "var(--c-accent)", "pointRadius": 3 },
    { "label": "windowed attention (7x7 window)", "data": [0.4, 0.7, 1.0, 1.9, 4.2, 7.4], "borderColor": "var(--c-green)", "pointRadius": 3 } ] },
  "options": { "scales": { "x": {"title": {"display": true, "text": "input side length (pixels), P = 16"}}, "y": {"type": "logarithmic", "title": {"display": true, "text": "GFLOPs per image (log scale)"}} } }
}
```

At 224 pixels attention is about 4% of the total; at 768 pixels it is a third, and at 1024 pixels almost half. This is why dense prediction tasks (detection, segmentation) that use high-resolution inputs pushed the community toward windowed and hierarchical designs such as Swin.

## Data efficiency: DeiT and distillation {#deit}

Recall the ViT headline: trained on ImageNet-1k alone, a ViT underperforms a good CNN, and it needs hundreds of millions of images to win. For most people that makes ViT unusable. Touvron et al. (Facebook AI, ICML 2021, "Training data-efficient image transformers & distillation through attention") showed that most of the gap was a **training recipe** problem. With heavy data augmentation (RandAugment, Mixup, CutMix, random erasing), repeated augmentation, stochastic depth, label smoothing and a long AdamW schedule, a ViT-B trained on ImageNet-1k alone reaches about 82% top-1 on a single machine in a few days, which was competitive with strong CNNs of the same size. The lesson, that Transformers lean on regularization and augmentation to substitute for the missing inductive biases, is important and recurs throughout this week.

### The distillation token {#distill-token}

DeiT's second contribution was a way to import the inductive bias of a CNN without changing the architecture. A **distillation token** $\boldsymbol{x}_{\text{dist}}$ is added next to the class token. Both tokens are learned vectors that participate in attention like any other token, and each has its own linear head:

- the class head is trained against the true label with cross-entropy;
- the distillation head is trained to match a **teacher** (in the paper, a convolutional RegNet) by *hard-label distillation*, i.e. cross-entropy against the teacher's arg-max prediction $y_t=\argmax_c Z_t(c)$.

$$
\mathcal{L} = \tfrac12\,\mathrm{CE}\big(\psi(Z_{\text{cls}}), y\big) + \tfrac12\,\mathrm{CE}\big(\psi(Z_{\text{dist}}), y_t\big),
$$

with $\psi$ the softmax. Interestingly, the hard label worked better than the usual soft-label KL distillation of Hinton, Vinyals and Dean (2015) in their experiments. At test time you average the two heads' softmax outputs. The two tokens end up with different representations (cosine similarity between them is only moderate in the early layers and increases toward the last), and the authors found the distillation token works better than simply using two class tokens trained on both losses, so it is genuinely absorbing something about the teacher.

```python:deit_loss.py
import torch
import torch.nn.functional as F

def deit_hard_distill_loss(cls_logits, dist_logits, labels, teacher_logits):
    """cls_logits, dist_logits, teacher_logits: (B, num_classes); labels: (B,)"""
    teacher_labels = teacher_logits.argmax(dim=1)             # hard teacher decision, (B,)
    loss_cls = F.cross_entropy(cls_logits, labels)
    loss_dist = F.cross_entropy(dist_logits, teacher_labels)
    return 0.5 * loss_cls + 0.5 * loss_dist

B, K = 8, 10
labels = torch.randint(0, K, (B,))
print(deit_hard_distill_loss(torch.randn(B, K), torch.randn(B, K), labels, torch.randn(B, K)))
```

One neat observation of the paper: the CNN teacher passes on its *inductive bias* through its labels, since the labels it produces reflect locality and invariance that the ViT then learns to imitate. The teacher is also often right on augmented crops where the true label is wrong (the crop may contain no object).

## Hierarchical transformers: Swin {#swin}

A plain ViT keeps a single resolution ($14\times14$) at every layer and uses global attention. Convolutional backbones are hierarchical: resolution falls (stride 4, 8, 16, 32) while channel width rises, giving the multi-scale feature maps that detection and segmentation heads (FPN, U-Net) expect. The Swin Transformer (Liu et al., ICCV 2021, "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows") brings both properties to a Transformer.

### Patch merging and the hierarchy {#patch-merging}

Swin starts with small $4\times4$ patches, so a $224\times224$ image gives a $56\times56$ token grid of width $C$ (96 in Swin-T). Each *stage* consists of Transformer blocks at a fixed resolution. Between stages, **patch merging** concatenates each $2\times2$ neighbourhood of tokens (giving $4C$ features) and applies a linear layer to $2C$: the grid goes $56\to28\to14\to7$ and the width $C\to2C\to4C\to8C$. Swin-T has stage depths $(2,2,6,2)$ and heads $(3,6,12,24)$, about 29M parameters and 4.5 GFLOPs.

### Window attention and its cost {#window-attention}

Global attention over $56\times56=3136$ tokens would be far too expensive. Swin partitions the grid into non-overlapping windows of $M\times M$ tokens ($M=7$) and computes attention *within each window only*. With $h\times w$ tokens and width $C$:

$$
\Omega(\text{global MSA}) = 4hwC^2 + 2(hw)^2C,\qquad
\Omega(\text{window MSA}) = 4hwC^2 + 2M^2\,hwC .
$$

The second term is **linear** in the number of tokens for a fixed window size, which is the key to scaling to large images (see the cost chart above). Attention is also computed with a learned **relative position bias** $B\in\mathbb{R}^{M^2\times M^2}$, looked up from a table of $(2M-1)^2$ parameters indexed by the relative offset between two tokens:

$$
\mathrm{Attention}(Q,K,V)=\softmax\!\Big(\frac{QK^\top}{\sqrt d}+B\Big)V .
$$

### Shifted windows {#shifted-windows}

Windowing alone has a flaw: tokens in different windows never communicate. Swin alternates two block types. In the first, windows are the regular grid. In the second, the grid is **shifted by $(\lfloor M/2\rfloor,\lfloor M/2\rfloor)$**, so each new window straddles the borders of four old ones and information crosses window boundaries. Stacking the two gives every token a growing receptive field, as in a CNN.

{{figure: week11-swin-windows | Regular windows in layer $\ell$ (left) and windows shifted by $M/2=2$ in layer $\ell+1$ (right) for an $8\times8$ grid with $M=4$. Colours mark which tokens share attention. On the right, the border windows are the pieces that wrap around by a cyclic shift.}}

The naive shifted grid yields more windows (e.g. $3\times3$ instead of $2\times2$) of unequal size at the borders. Swin avoids the cost with a **cyclic shift**: roll the feature map by $(-s,-s)$, partition into the same regular windows, and apply an attention *mask* so that tokens that were not neighbours in the original image (they were wrapped from the opposite side) cannot attend to each other. The map is rolled back afterwards.

```python:swin_windows.py
import torch

def window_partition(x, M):
    """x: (B, H, W, C) -> windows (B * H/M * W/M, M, M, C)."""
    B, H, W, C = x.shape
    x = x.view(B, H // M, M, W // M, M, C)
    return x.permute(0, 1, 3, 2, 4, 5).reshape(-1, M, M, C)

def window_reverse(w, M, H, W):
    """windows (B * H/M * W/M, M, M, C) -> (B, H, W, C)."""
    C = w.shape[-1]
    B = w.shape[0] // ((H // M) * (W // M))
    x = w.view(B, H // M, W // M, M, M, C).permute(0, 1, 3, 2, 4, 5)
    return x.reshape(B, H, W, C)

x = torch.arange(8 * 8).float().reshape(1, 8, 8, 1)       # a labelled 8x8 grid
M, s = 4, 2
w = window_partition(x, M)
print(w.shape)                                             # torch.Size([4, 4, 4, 1])
assert torch.equal(window_reverse(w, M, 8, 8), x)          # round trip is exact

# cyclic shift used by the shifted-window block
shifted = torch.roll(x, shifts=(-s, -s), dims=(1, 2))
w2 = window_partition(shifted, M)                          # same 4 windows, but different tokens in each
print(w2.shape)                                            # torch.Size([4, 4, 4, 1])
assert torch.equal(torch.roll(window_reverse(w2, M, 8, 8), shifts=(s, s), dims=(1, 2)), x)
```

With Swin-T at 81.3% ImageNet-1k top-1 (29M parameters, 4.5 GFLOPs) and strong detection and segmentation results, Swin became the default Transformer backbone for dense prediction for a couple of years. Its design is an explicit *re-introduction of convolutional inductive biases* into a Transformer: locality (windows), hierarchy (patch merging) and a translation-equivariant position bias (relative bias). That brings us to the question underneath this whole section.

## CNN versus ViT: inductive biases and the data-efficiency evidence {#inductive-bias}

### What a CNN assumes {#cnn-assumptions}

An **inductive bias** is an assumption built into a model that makes some functions easier to learn than others. A convolution layer assumes:

1. **Locality**: output at a position depends only on a small neighbourhood.
2. **Weight sharing / translation equivariance**: $f(\text{shift}(x))=\text{shift}(f(x))$; the same detector is used everywhere.
3. **Hierarchy** (with pooling): features compose from small to large.

These assumptions are almost true for natural images, so they shrink the hypothesis space in the right direction and make small datasets workable. A ViT assumes much less. The patch embedding is local and shared, but after that every token interacts with every token from layer 1, and (with learned absolute positions) translation equivariance has to be learned.

### What the evidence says {#evidence}

- **Dataset scale.** The ViT paper's Figure on pretraining set size shows the crossover: with ImageNet-1k, BiT ResNets beat ViTs; with ImageNet-21k they are comparable; with JFT-300M ViTs win, and the larger the model the larger the gain. This is the cleanest demonstration that the missing biases can be bought back with data.
- **Recipes close much of the gap.** DeiT (above) and Steiner et al. ("How to train your ViT?", 2021) showed that augmentation and regularization recipes are worth as much as more data in the ImageNet-1k regime. Beyer et al. ("Better plain ViT baselines for ImageNet-1k", 2022) reached about 76.5% top-1 for ViT-S/16 in 90 epochs and about 80% in 300 epochs with a simple recipe, undercutting the narrative that ViTs are inherently data-hungry.
- **Early convolutions help.** Xiao et al. ("Early Convolutions Help Transformers See Better", NeurIPS 2021) replaced the large-stride patchify layer with a small stack of stride-2 convolutions and found ViTs optimize more stably and reach higher accuracy. A single $16\times16$ stride-16 convolution is a coarse, hard-to-optimize first layer.
- **Different representations.** Raghu et al. ("Do Vision Transformers See Like Convolutional Neural Networks?", NeurIPS 2021) compared internal representations: ViTs have more uniform representations across layers (strong residual propagation, early attention heads that look both locally and globally), and learn local attention in lower layers only when trained on enough data.
- **Robustness.** Several studies (e.g. Naseer et al., "Intriguing properties of vision transformers", 2021) found ViTs relatively robust to occlusion and patch shuffling, and more shape-biased than CNNs, though later work shows much of this comes from the training recipe rather than the architecture alone.
- **Convolutions strike back.** Liu et al. ("A ConvNet for the 2020s", ConvNeXt, CVPR 2022) modernized a ResNet with the same training recipe, larger kernels, fewer activations and LayerNorm, and matched Swin at similar FLOPs. When the recipe and scale are controlled, the architecture family matters less than the field assumed in 2021.

:::intuition A useful way to hold this in your head
A prior (inductive bias) and data are substitutable. With little data, a strong, correct prior wins. With a lot of data, the prior becomes a constraint and the flexible model wins. The CNN is an excellent default when data are scarce or compute is tight. The ViT is a good default when you can pretrain at scale or reuse someone else's pretrained weights, and its tokens-in, tokens-out interface is what makes the multimodal models later this week possible: images, text, and audio all become sequences for one Transformer.
:::

:::note A caution about benchmarks
Many "ViT beats CNN" or "CNN beats ViT" claims compare models trained with different recipes, datasets, resolutions, augmentations and epochs. If you are doing your own comparison, match the recipe, report the compute (FLOPs and throughput on the same hardware), and use multiple seeds.
:::

## Transformers for detection and segmentation {#detection}

### DETR: detection as set prediction {#detr}

Object detectors before 2020 (Faster R-CNN, YOLO, RetinaNet) are full of hand-designed components: anchor boxes, region proposals, and non-maximum suppression (NMS) to remove duplicate predictions. Carion et al. (Facebook AI, ECCV 2020, "End-to-End Object Detection with Transformers") removed all of it by formulating detection as **set prediction**.

{{figure: week11-detr | DETR. A CNN backbone and a Transformer encoder produce a feature sequence; the decoder takes $N=100$ learned object queries and outputs $N$ (class, box) predictions in parallel. Hungarian matching assigns each ground-truth object to exactly one prediction, and the remaining predictions are trained to say "no object".}}

The architecture: a CNN (ResNet-50) produces a feature map that is flattened to $HW/32^2$ tokens with added 2-D position encodings; a Transformer encoder refines them; a Transformer decoder takes $N$ **object queries** (learned embeddings, $N=100$ in the paper, larger than the number of objects in any image) and cross-attends to the encoder output. Each decoder output is passed through small feed-forward heads that predict a class (including a special $\varnothing$ "no object" class) and a normalized box $(c_x,c_y,w,h)$. All $N$ predictions are produced **in parallel**, not autoregressively.

### Hungarian matching and the set loss {#hungarian}

The network outputs an unordered set $\hat{Y}=\{\hat y_i\}_{i=1}^N$ and the ground truth is a set $Y=\{y_j\}$ padded with $\varnothing$ up to size $N$. There is no natural alignment between them, and the loss must not depend on how predictions are ordered. DETR finds the best **one-to-one assignment** $\hat\sigma$ among permutations $\sigma$ of $N$ elements:

$$
\hat\sigma = \argmin_{\sigma\in\mathfrak{S}_N}\ \sum_{i=1}^{N}\mathcal{L}_{\text{match}}\big(y_i,\hat y_{\sigma(i)}\big),
\qquad
\mathcal{L}_{\text{match}} = -\mathbb{1}_{\{c_i\neq\varnothing\}}\hat p_{\sigma(i)}(c_i) + \mathbb{1}_{\{c_i\neq\varnothing\}}\mathcal{L}_{\text{box}}(b_i,\hat b_{\sigma(i)}).
$$

This is a linear assignment problem solved exactly in $O(N^3)$ by the Hungarian algorithm (Kuhn, 1955), available as `scipy.optimize.linear_sum_assignment`. Given the matching, the training loss is the sum over matched pairs of a classification cross-entropy (the weight of the $\varnothing$ class is reduced by a factor of 10 to counter class imbalance) and a box loss that combines L1 and generalized IoU (Rezatofighi et al., 2019), the latter being scale-invariant.

```python:hungarian_match.py
import torch
from scipy.optimize import linear_sum_assignment

def box_cxcywh_to_xyxy(b):
    cx, cy, w, h = b.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)

@torch.no_grad()
def hungarian_match(pred_logits, pred_boxes, tgt_labels, tgt_boxes, w_cls=1.0, w_l1=5.0):
    """One image. pred_logits (N, K+1), pred_boxes (N, 4), tgt_labels (M,), tgt_boxes (M, 4).
    Returns (pred_idx, tgt_idx): which prediction is assigned to which target."""
    prob = pred_logits.softmax(-1)                          # (N, K+1)
    cost_cls = -prob[:, tgt_labels]                         # (N, M): -p_hat(c_j) for each pair
    cost_l1 = torch.cdist(pred_boxes, tgt_boxes, p=1)       # (N, M)
    cost = w_cls * cost_cls + w_l1 * cost_l1                # (a GIoU term would be added in the full version)
    pi, tj = linear_sum_assignment(cost.cpu().numpy())      # optimal assignment, O(N^3)
    return torch.as_tensor(pi), torch.as_tensor(tj)

N, K, M = 100, 91, 3
logits, boxes = torch.randn(N, K + 1), torch.rand(N, 4)
labels, tboxes = torch.tensor([3, 17, 17]), torch.rand(M, 4)
pi, tj = hungarian_match(logits, boxes, labels, tboxes)
print(pi.tolist(), tj.tolist())                             # three predictions matched; the other 97 become "no object"
```

### Why this works, and where it hurts {#detr-limits}

The one-to-one matching is what makes NMS unnecessary: if two queries both predict the same object, only one is matched, and the other is pushed toward "no object", so the model learns to suppress duplicates through the decoder's self-attention between queries. DETR's accuracy was competitive with a tuned Faster R-CNN (about 42 AP on COCO with a ResNet-50) and strong on large objects, but it trained very slowly (hundreds of epochs) and did worse on small objects, because attention over a single low-resolution feature map has no multi-scale structure and the cross-attention initially looks everywhere. **Deformable DETR** (Zhu et al., ICLR 2021) replaced global attention with attention over a small set of sampled points at multiple scales and cut training by an order of magnitude. A line of follow-ups (conditional DETR, DAB-DETR, DINO-DETR, RT-DETR) kept pushing speed and accuracy.

### Segment Anything (overview) {#sam}

The Segment Anything Model (Kirillov et al., Meta AI, ICCV 2023) is the template for a **promptable foundation model in vision**. The task, *promptable segmentation*, is: given an image and a prompt (a point, a box, a rough mask, or text in exploratory versions), return a valid segmentation mask. The architecture has three parts:

1. an **image encoder**: a large ViT (ViT-H) pretrained with MAE (see below), run once per image at $1024\times1024$ resolution, producing an image embedding;
2. a **prompt encoder**: sparse prompts (points, boxes) are embedded with position encodings plus learned type embeddings; dense prompts (masks) with convolutions;
3. a lightweight **mask decoder**: a small two-layer Transformer decoder with cross-attention between prompt tokens and the image embedding, producing several mask hypotheses (to handle ambiguity: a click on a shirt could mean the shirt, the person, or the shirt's button) and a predicted quality score for each.

The expensive encoder runs once, and the decoder runs in milliseconds, which makes interactive use feasible. The training data came from a **data engine** in three phases (assisted-manual, semi-automatic, fully automatic), yielding SA-1B: 11 million images and over a billion masks. SAM illustrates the recurring pattern of this week: a ViT backbone, self-supervised pretraining, a large dataset built with the model in the loop, and a small task-specific head.

## Self-supervised learning {#ssl}

Everything so far assumed labels. Labels are expensive, they encode only the categories someone thought of, and a single class index is a thin training signal for a network with hundreds of millions of parameters. **Self-supervised learning (SSL)** manufactures a supervisory signal from the data itself: hide or transform part of the input and ask the network to relate it to the rest. The learned representation $f(x)$ is then evaluated by how well it serves *downstream* tasks, usually via a **linear probe** (train only a linear classifier on frozen features), $k$-NN, or fine-tuning.

{{figure: week11-ssl-timeline | A compressed timeline of self-supervised visual learning. The field moved from hand-designed pretext tasks to instance discrimination and contrastive learning, then to methods without negatives, and finally to masked modeling and web-scale language-supervised pretraining.}}

### Pretext tasks: where it started {#pretext}

Before 2018 the standard recipe was a **pretext task**: a puzzle whose answer is known for free, designed so that solving it requires understanding something about images.

- **Exemplar CNN** (Dosovitskiy et al., 2014): treat each image, and its random augmentations, as its own class. This is the ancestor of instance discrimination.
- **Context prediction** (Doersch, Gupta and Efros, ICCV 2015): given two patches from an image, predict their relative position (one of eight).
- **Jigsaw** (Noroozi and Favaro, ECCV 2016): shuffle nine tiles and predict the permutation.
- **Colorization** (Zhang, Isola and Efros, ECCV 2016): predict colour from grayscale.
- **Context Encoders** (Pathak et al., CVPR 2016): inpaint a missing region. This is the direct ancestor of MAE.
- **Rotation prediction** (Gidaris, Singh and Komodakis, ICLR 2018): rotate by 0, 90, 180 or 270 degrees and classify the rotation. Surprisingly strong for its simplicity.

These tasks have a shared weakness: the network can solve them with *shortcuts*. In context prediction, chromatic aberration at patch borders gives away the position, so early work had to add gaps and colour jitter. Features learned for one task are also tuned to that task's specific invariances (rotation prediction is useless for a model that must be rotation-invariant). Contrastive learning offered a more general alternative: rather than a fixed puzzle, define *which transformations should not matter*.

### Contrastive learning and instance discrimination {#contrastive}

The contrastive principle is: two **views** of the same input (the *positive pair*) should have similar embeddings, while views of different inputs (*negatives*) should be dissimilar. Learning is invariant to the augmentations used to create views, and discriminative across instances, which prevents collapse because mapping everything to one point makes negatives indistinguishable from positives.

{{figure: week11-simclr | SimCLR. Two random augmentations of the same image pass through a shared encoder $f$ and a small projection head $g$. The loss is computed in the projection space $z$; the representation used downstream is $h=f(x)$, before the head.}}

**SimCLR** (Chen, Kornblith, Norouzi and Hinton, ICML 2020, "A Simple Framework for Contrastive Learning of Visual Representations") is the cleanest version. A minibatch of $N$ images gives $2N$ augmented views. For a positive pair $(i,j)$, the other $2(N-1)$ views in the batch serve as negatives. Writing $\operatorname{sim}(u,v)=u^\top v/(\norm{u}\norm{v})$ for cosine similarity and $\tau>0$ for the **temperature**, the loss for the pair is the *NT-Xent* (normalized temperature-scaled cross-entropy) loss:

$$
\ell_{i,j} = -\log\frac{\exp\big(\operatorname{sim}(z_i,z_j)/\tau\big)}{\sum_{k=1}^{2N}\mathbb{1}_{[k\neq i]}\exp\big(\operatorname{sim}(z_i,z_k)/\tau\big)} .
$$

The total loss averages $\ell_{i,j}$ and $\ell_{j,i}$ over all positive pairs. It is exactly a $(2N-1)$-way softmax cross-entropy where the "correct class" is the partner view. The design findings of the SimCLR paper are worth memorizing because they generalize:

1. **Augmentation composition is the main ingredient.** Random crop plus resize combined with strong colour jitter is far better than either alone. Cropping makes the task require recognizing the same *object* from different parts ("global to local" prediction), and colour jitter removes the colour-histogram shortcut by which two crops of one image can be matched trivially.
2. **A nonlinear projection head** $g$ (an MLP) before the loss improves the representation $h$ below it by more than 10 points of linear-probe accuracy compared with no head; the head absorbs invariances that the contrastive loss wants but downstream tasks may not.
3. **Big batches and long training matter**: more negatives, better bound (see the derivation below). SimCLR used batches up to 8192 across TPU cores. A ResNet-50 trained this way reached roughly 69% linear-probe top-1 on ImageNet with no labels, and wider ResNets higher.
4. **Normalization and temperature**: L2-normalizing embeddings and tuning $\tau$ matter a lot.

### Deriving InfoNCE {#infonce}

Why does a softmax over a positive and some negatives make for good representations? The answer is that it lower-bounds the **mutual information** between the two views. InfoNCE was introduced by van den Oord, Li and Vinyals ("Representation Learning with Contrastive Predictive Coding", 2018), and the following derivation follows their argument.

:::math Derivation: InfoNCE as a bound on mutual information
**Setup.** Let $x$ be a "target" view and $c$ a "context" (the other view). They are drawn jointly from $p(x,c)$. We want to learn a score $f(x,c)\ge 0$ that identifies the true partner. Draw one positive $x_+\sim p(x\mid c)$ and $N-1$ negatives $x_j\sim p(x)$ independently of $c$, forming a set $X=\{x_1,\dots,x_N\}$ with the positive at a random index. The loss is the cross-entropy of identifying the positive:

$$
\mathcal{L}_N = -\,\mathbb{E}_{X,c}\left[\log\frac{f(x_+,c)}{\sum_{x_j\in X}f(x_j,c)}\right].
$$

**Optimal critic.** The probability that index $i$ is the positive, given the set $X$ and context $c$, is by Bayes' rule

$$
p(\text{pos}=i\mid X,c)=\frac{p(x_i\mid c)\prod_{l\ne i}p(x_l)}{\sum_{j}p(x_j\mid c)\prod_{l\ne j}p(x_l)}=\frac{p(x_i\mid c)/p(x_i)}{\sum_{j}p(x_j\mid c)/p(x_j)} .
$$

The softmax ratio with $f(x,c)\propto p(x\mid c)/p(x)$ reproduces this exactly, so the optimal critic is the **density ratio** (up to a constant that cancels).

**The loss at the optimum.** Substituting,

$$
\mathcal{L}_N^{\text{opt}} = \mathbb{E}\log\left[1+\frac{p(x_+)}{p(x_+\mid c)}\sum_{x_j\in X_{\text{neg}}}\frac{p(x_j\mid c)}{p(x_j)}\right].
$$

Each negative is drawn from $p(x)$ independently of $c$, so $\mathbb{E}_{x_j\sim p(x)}\big[p(x_j\mid c)/p(x_j)\big]=\int p(x\mid c)\,dx=1$. Replacing the sum of $N-1$ terms by its expectation (an approximation that is accurate for large $N$) gives

$$
\mathcal{L}_N^{\text{opt}}\approx \mathbb{E}\log\left[1+(N-1)\frac{p(x_+)}{p(x_+\mid c)}\right]\ \ge\ \mathbb{E}\log\left[(N-1)\frac{p(x_+)}{p(x_+\mid c)}\right]
= \log(N-1) - I(x;c),
$$

since $\mathbb{E}\log\frac{p(x\mid c)}{p(x)}=I(x;c)$. Rearranging (and using $\log N$ in place of $\log(N-1)$ as in the original paper, a small difference for large $N$):

$$
I(x;c)\ \ge\ \log N - \mathcal{L}_N .
$$
:::

Three consequences follow. First, **minimizing the loss maximizes a lower bound on the mutual information** between views, so the representation must retain whatever the two views share. This also explains why augmentations matter: the information shared between views is exactly what survives the augmentation, so we choose augmentations that destroy nuisance information (colour, exact position) and keep semantic information. Second, the bound can never exceed $\log N$ (the loss is $\ge0$), so **a large number of negatives is needed to measure large mutual information**. This is the theoretical reason for SimCLR's big batches and for MoCo's queue. Third, the bound is a *bound*, and is loose: Tschannen et al. ("On Mutual Information Maximization for Representation Learning", ICLR 2020) showed that good downstream performance is not explained by tightness of the MI estimate, and that the choice of encoder architecture and negative sampling matter as much as the MI interpretation.

### The temperature {#temperature}

The temperature $\tau$ multiplies all logits by $1/\tau$. Differentiating the loss for anchor $i$ with respect to the similarity to another view $k$ (with $p_k$ the softmax probability of $k$) gives a very clean result:

$$
\frac{\partial \ell_i}{\partial s_{ik}} = \frac{1}{\tau}\big(p_k-\mathbb{1}[k=\text{pos}]\big),\qquad p_k=\frac{e^{s_{ik}/\tau}}{\sum_m e^{s_{im}/\tau}} .
$$

The push on a negative is proportional to its softmax probability, and that probability is dominated by the *most similar* negatives when $\tau$ is small. A small $\tau$ therefore means **hard-negative mining by softmax**: the loss concentrates on the few negatives that look like the anchor. This yields better separation but is more sensitive to *false negatives* (other images of the same class that are wrongly pushed away) and noisy labels. A large $\tau$ treats negatives more uniformly and gives a smoother, more tolerant objective. Wang and Liu (CVPR 2021, "Understanding the Behaviour of Contrastive Loss") analyze this trade-off, and Wang and Isola (ICML 2020) decompose the loss into an **alignment** term (positives close) and a **uniformity** term (embeddings spread on the hypersphere), which are the two forces visible in the demo.

{{demo: week11-infonce-temperature | The InfoNCE loss, its gradients, and the temperature | 520}}

```chart
{
  "type": "line",
  "caption": "InfoNCE on synthetic embeddings (N = 256, d = 64): positive pairs are a shared Gaussian vector plus independent noise (noise std 0.6), then L2-normalised. The loss is a monotonically increasing function of the temperature and approaches log N = 5.55 as tau grows; the probability of the positive falls toward 1/N. Computed with NumPy.",
  "height": 320,
  "data": { "labels": [0.05, 0.1, 0.2, 0.3, 0.5, 1, 2, 5], "datasets": [
    { "label": "InfoNCE loss", "data": [0.007, 0.365, 2.221, 3.245, 4.135, 4.83, 5.185, 5.401], "borderColor": "var(--c-accent)", "yAxisID": "y", "pointRadius": 3 },
    { "label": "mean P(positive)", "data": [0.994, 0.709, 0.113, 0.04, 0.016, 0.008, 0.006, 0.005], "borderColor": "var(--c-blue)", "yAxisID": "y1", "pointRadius": 3 } ] },
  "options": { "scales": { "x": {"title": {"display": true, "text": "temperature tau"}}, "y": {"position": "left", "title": {"display": true, "text": "loss (nats)"}}, "y1": {"position": "right", "min": 0, "max": 1, "grid": {"drawOnChartArea": false}, "title": {"display": true, "text": "probability of positive"}} } }
}
```

Here is NT-Xent in PyTorch, written carefully so that the diagonal (self-similarity) is masked out of the denominator:

```python:nt_xent.py
import torch
import torch.nn.functional as F

def nt_xent(z1, z2, tau=0.2):
    """SimCLR loss. z1, z2: (N, d) projections of two views of the same N images."""
    N = z1.shape[0]
    z = F.normalize(torch.cat([z1, z2], dim=0), dim=1)    # (2N, d), unit length
    sim = z @ z.T / tau                                   # (2N, 2N) cosine similarities / tau
    sim.fill_diagonal_(float("-inf"))                     # a view is not its own negative or positive
    # the positive of row i is row i+N (and vice versa)
    targets = torch.cat([torch.arange(N, 2 * N), torch.arange(0, N)]).to(z.device)
    return F.cross_entropy(sim, targets)                  # mean over the 2N anchors

z1, z2 = torch.randn(256, 128), torch.randn(256, 128)
print(nt_xent(z1, z2).item())                             # about log(2N - 1) = 6.23 for random vectors
```

### MoCo: a queue of negatives {#moco}

Large batches are expensive. **MoCo** (He, Fan, Wu, Xie and Girshick, CVPR 2020, "Momentum Contrast for Unsupervised Visual Representation Learning") decouples the number of negatives from the batch size with two ideas. First, keep a **queue** of the most recent key embeddings (e.g. 65,536 of them) as negatives: each step, enqueue the newest batch and dequeue the oldest. Second, because the queue contains keys produced by encoders from *many steps ago*, they must be consistent with each other. MoCo therefore computes keys with a **momentum encoder**, an exponential moving average of the query encoder,

$$
\theta_k \leftarrow m\,\theta_k + (1-m)\,\theta_q,\qquad m=0.999 ,
$$

which changes slowly so that queued keys stay comparable. Gradients flow only through the query encoder. MoCo v2 (Chen, Fan, Girshick and He, 2020) borrowed SimCLR's MLP head and stronger augmentations and matched or exceeded SimCLR with batches of 256. MoCo v3 (Chen, Xie and He, ICCV 2021, "An Empirical Study of Training Self-Supervised Vision Transformers") applied the idea to ViTs, found that training is unstable (loss spikes), and showed that freezing the random patch-projection layer fixes much of it, a reminder that the patchify layer is the delicate part of a ViT.

### Collapse, and how to avoid it {#collapse}

Contrastive methods avoid collapse by explicit negatives. Could we drop them and only ask the two views to agree? The naive objective, minimize $\norm{z_1-z_2}^2$, has a global minimum at a constant function, and gradient descent finds it quickly. We can see it in a small real experiment: a linear encoder trained on augmented views of a four-cluster toy dataset, once with an InfoNCE loss and once with only the alignment term on L2-normalized embeddings. The *effective rank* of the batch of embeddings is the exponential of the entropy of its normalized singular values. If features collapse onto a line it approaches 1.

```chart
{
  "type": "line",
  "caption": "Dimensional collapse in a toy SSL run (NumPy, linear encoder 8 -> 4, four Gaussian clusters, noise-based augmentation). With negatives (InfoNCE) the effective rank of the embeddings stays near 3.5; with the alignment term alone it falls to about 1 within 250 steps (plain gradient descent, learning rate 0.5, 128 images per batch). Effective rank = exp(entropy of normalised singular values of the centred batch).",
  "height": 320,
  "data": { "labels": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300], "datasets": [
    { "label": "InfoNCE (with negatives)", "data": [3.597, 3.509, 3.49, 3.468, 3.452, 3.465, 3.453, 3.457, 3.442, 3.446, 3.465, 3.478, 3.458, 3.451, 3.454, 3.466, 3.469, 3.455, 3.454, 3.452, 3.447, 3.468, 3.475, 3.459, 3.463, 3.457, 3.467, 3.473, 3.471, 3.494, 3.476], "borderColor": "var(--c-green)", "pointRadius": 0, "tension": 0.2 },
    { "label": "alignment only (no negatives)", "data": [3.197, 2.841, 2.726, 2.593, 2.521, 2.43, 2.34, 2.262, 2.191, 2.137, 2.065, 2.022, 1.986, 1.94, 1.89, 1.817, 1.747, 1.665, 1.563, 1.457, 1.334, 1.274, 1.141, 1.099, 1.1, 1.107, 1.143, 1.045, 1.071, 1.048, 1.039], "borderColor": "var(--c-red)", "pointRadius": 0, "tension": 0.2 } ] },
  "options": { "scales": { "x": {"title": {"display": true, "text": "training step"}}, "y": {"min": 0, "title": {"display": true, "text": "effective rank of embeddings (max 4)"}} } }
}
```

Two kinds of collapse are worth distinguishing. **Complete collapse** maps every input to the same vector. **Dimensional collapse** uses only a low-dimensional subspace of the embedding space (as in the run above, where the rank drops without reaching exactly 1 on every batch). The families of non-contrastive methods are best understood as different ways to prevent them:

- **Asymmetry and stop-gradient** (BYOL, SimSiam, DINO): the two branches differ, and gradients do not flow through one of them.
- **Redundancy reduction** (Barlow Twins, Zbontar et al., ICML 2021; VICReg, Bardes, Ponce and LeCun, ICLR 2022): add explicit terms to the loss that keep each embedding dimension varying (variance) and decorrelated from the others (covariance), so collapse costs loss.
- **Clustering and centering** (SwAV, Caron et al., NeurIPS 2020; DINO): enforce that prototypes are used evenly across the batch.

### BYOL, SimSiam and DINO {#byol-dino}

{{figure: week11-byol-dino | Self-distillation without negatives. A student is trained to predict a teacher's output on a different view. The teacher is not trained by gradient but is an exponential moving average of the student. BYOL adds a predictor head on the student side; DINO instead uses centering and sharpening of the teacher's output distribution.}}

**BYOL** (Grill et al., DeepMind, NeurIPS 2020, "Bootstrap Your Own Latent") shocked the field by reaching about 74% linear-probe accuracy on ImageNet with ResNet-50 *with no negatives at all*. It has an online network (encoder, projector, **predictor**) and a target network (encoder, projector) that is an EMA of the online one. The online network predicts the target's projection of the other view, and the loss is the mean squared error between L2-normalized vectors (equivalent to $2-2\cos$). The target receives a stop-gradient. Why doesn't it collapse? There is no single airtight proof. Empirically, each of the ingredients is needed: without the predictor, or with the target equal to the online weights (no EMA), training collapses. Tian, Chen and Ganguli (ICML 2021, "Understanding Self-Supervised Learning Dynamics without Contrastive Pairs") analyze a linear model and show that the predictor and the EMA together shape the dynamics so that collapse is an unstable fixed point. A related stir was whether BatchNorm provided implicit negatives, since batch statistics mix information across samples; later work (Richemond et al., 2020, "BYOL works even without batch statistics") showed it need not.

**SimSiam** (Chen and He, CVPR 2021, "Exploring Simple Siamese Representation Learning") removed the EMA, keeping the predictor and the stop-gradient, and still avoided collapse; the stop-gradient is the critical piece. It can be viewed as an alternating optimization (like k-means) over the representation and an implicit target.

**DINO** (Caron et al., Facebook AI, ICCV 2021, "Emerging Properties in Self-Supervised Vision Transformers") applies BYOL-style self-distillation to ViTs but replaces the regression loss by a **cross-entropy between softmax distributions** over $K$ outputs (65,536 "prototypes"), with a teacher that is an EMA of the student. To keep the teacher's distribution from collapsing, DINO uses two opposing operations on the teacher's logits:

- **centering**: subtract a running mean $c$, which prevents one dimension from dominating (a constant collapse in one dimension) but pushes toward a *uniform* distribution;
- **sharpening**: a low teacher temperature $\tau_t$ makes the distribution peaked, which pushes against uniformity.

With student distribution $P_s=\softmax(s/\tau_s)$ and teacher $P_t=\softmax((t-c)/\tau_t)$ the loss on views $x_1,x_2$ is $-\sum_k P_t^{(x_1)}(k)\log P_s^{(x_2)}(k)$, summed over pairs of different views. DINO also introduced **multi-crop**: the teacher sees only two large global crops while the student also sees several small local crops, forcing "local-to-global" correspondence. Remarkably, the self-attention maps of the final-layer class token in a DINO-trained ViT segment the foreground object, with no segmentation labels; k-NN classification on the frozen features works nearly as well as the linear probe. **DINOv2** (Oquab et al., 2023) combined this objective with masked prediction (iBOT) and a curated 142M-image dataset to produce strong general-purpose frozen features.

```python:dino_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class DINOLoss(nn.Module):
    """Cross-entropy between a centred, sharpened teacher and the student."""
    def __init__(self, out_dim, tau_s=0.1, tau_t=0.04, center_momentum=0.9):
        super().__init__()
        self.tau_s, self.tau_t, self.m = tau_s, tau_t, center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(self, student_out, teacher_out):
        """student_out: list of (B, K) logits (one per view); teacher_out: list of (B, K) for the global views."""
        total, n = 0.0, 0
        t_probs = [F.softmax((t - self.center) / self.tau_t, dim=-1).detach() for t in teacher_out]   # stop-grad
        for ti, tp in enumerate(t_probs):
            for si, s in enumerate(student_out):
                if si == ti:
                    continue                                   # skip same-view pairs
                total = total + torch.sum(-tp * F.log_softmax(s / self.tau_s, dim=-1), dim=-1).mean()
                n += 1
        with torch.no_grad():                                  # EMA update of the centre
            batch_center = torch.cat(teacher_out).mean(dim=0, keepdim=True)
            self.center = self.center * self.m + batch_center * (1 - self.m)
        return total / n

@torch.no_grad()
def ema_update(teacher, student, m):
    for pt, ps in zip(teacher.parameters(), student.parameters()):
        pt.mul_(m).add_(ps.detach(), alpha=1 - m)              # theta_t <- m theta_t + (1 - m) theta_s

crit = DINOLoss(out_dim=256)
s = [torch.randn(4, 256, requires_grad=True) for _ in range(4)]    # 2 global + 2 local crops
t = [torch.randn(4, 256) for _ in range(2)]
loss = crit(s, t); loss.backward()
print(loss.item())
```

### Masked image modeling and MAE {#mae}

Language models learn by predicting masked or next tokens (Week 10). The same recipe in vision is **masked image modeling**: hide some patches and train the network to reconstruct them. Early attempts (Context Encoders, iGPT, BEiT) lagged contrastive learning. BEiT (Bao, Dong, Piao and Wei, ICLR 2022) predicted discrete visual tokens from a dVAE for masked patches. The **Masked Autoencoder** (He, Chen, Xie, Li, Dollár and Girshick, CVPR 2022, "Masked Autoencoders Are Scalable Vision Learners") made the idea work with a few crucial design choices:

{{figure: week11-mae | MAE. Randomly mask 75% of the patches. The encoder processes only the visible 25%, so it is cheap. A lightweight decoder receives the encoded visible tokens plus learned mask tokens (with position embeddings), and reconstructs the pixels of the masked patches. After pretraining, the decoder is thrown away.}}

1. **A high masking ratio (75%).** Images are redundant: missing a patch can be filled by interpolation from neighbours. A low masking ratio makes reconstruction trivial and gives weak features. With 75% hidden, the task needs object-level understanding. (Text needs only about 15% masking since each word carries more information.)
2. **An asymmetric encoder-decoder.** The encoder sees only the visible tokens (no mask tokens), which both saves compute (about 3 to 4 times fewer tokens through the expensive encoder) and avoids a train-test mismatch between pretraining with mask tokens and downstream use without them. A narrow, shallow decoder (8 blocks, width 512) puts the mask tokens back, with position embeddings, and predicts pixel values.
3. **The reconstruction target is raw normalized pixels**, with the loss (mean squared error) computed only on masked patches. Normalizing each patch by its own mean and standard deviation improves representation quality.

```chart
{
  "type": "bar",
  "caption": "Encoder cost of a ViT-L/16 (24 layers, width 1024) forward pass at 224x224 as a function of the masking ratio, computed from the per-block formula in the text (GFLOPs, multiplies and adds counted separately). At 75% masking the encoder sees 50 tokens instead of 197 and costs about a quarter of an unmasked pass; the decoder adds a small amount.",
  "height": 300,
  "data": { "labels": ["0%", "25%", "50%", "75%", "85%", "90%"], "datasets": [ { "label": "encoder GFLOPs", "data": [122.8, 91.5, 60.8, 30.4, 18.2, 12.1], "backgroundColor": "var(--c-purple)" } ] },
  "options": { "scales": { "x": {"title": {"display": true, "text": "masking ratio"}}, "y": {"title": {"display": true, "text": "GFLOPs"}} } }
}
```

The implementation trick is **per-sample random shuffling**: generate random noise per patch, argsort it to obtain a random permutation, keep the first $25\%$ of the shuffled indices, and remember the inverse permutation so that the decoder can restore the original order.

```python:mae_masking.py
import torch

def random_masking(x, mask_ratio=0.75):
    """x: (B, L, D) patch tokens. Returns visible tokens, a binary mask (1 = removed), and the restore indices."""
    B, L, D = x.shape
    len_keep = int(L * (1 - mask_ratio))
    noise = torch.rand(B, L, device=x.device)              # independent noise per sample and patch
    ids_shuffle = torch.argsort(noise, dim=1)              # a random permutation per sample
    ids_restore = torch.argsort(ids_shuffle, dim=1)        # its inverse
    ids_keep = ids_shuffle[:, :len_keep]                   # first 25% of the permutation are kept
    x_vis = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D))    # (B, len_keep, D)
    mask = torch.ones(B, L, device=x.device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, 1, ids_restore)              # unshuffle: 0 = visible, 1 = masked (original order)
    return x_vis, mask, ids_restore

x = torch.randn(2, 196, 768)
x_vis, mask, ids_restore = random_masking(x)
print(x_vis.shape, mask.sum(dim=1))                        # torch.Size([2, 49, 768]) tensor([147., 147.])

def mae_loss(pred, target_pixels, mask, norm_pix=True):
    """pred, target_pixels: (B, L, P*P*3); mask: (B, L)."""
    if norm_pix:                                            # normalise each target patch
        mu, var = target_pixels.mean(-1, keepdim=True), target_pixels.var(-1, keepdim=True)
        target_pixels = (target_pixels - mu) / (var + 1e-6).sqrt()
    per_patch = ((pred - target_pixels) ** 2).mean(-1)      # (B, L)
    return (per_patch * mask).sum() / mask.sum()            # average over masked patches only
```

MAE's results: a ViT-Huge pretrained with MAE and fine-tuned on ImageNet-1k with no extra labeled data reaches 87.8% top-1, a record for ImageNet-1k-only methods at the time, and the pretraining is much faster than contrastive learning. There is an important subtlety: MAE features are **excellent for fine-tuning but mediocre for linear probing** (they must be nonlinearly re-arranged), unlike contrastive and DINO features. This difference reflects what each objective encourages: contrastive methods make features *linearly separable by semantic content* (because augmentation invariances align with class structure), whereas reconstruction retains low-level detail that is not linearly aligned with labels. As we saw above, MAE is the pretraining behind the SAM encoder.

### Choosing among the families {#ssl-compare}

| Family | Examples | Avoids collapse by | Strength | Weakness |
| --- | --- | --- | --- | --- |
| Pretext task | Jigsaw, rotation | Task has a non-trivial answer | Simple | Shortcuts, narrow features |
| Contrastive | SimCLR, MoCo | Explicit negatives | Strong linear probe | Needs many negatives, false negatives, augmentation design |
| Self-distillation | BYOL, DINO | Asymmetry, EMA, centering | No negatives, great k-NN | Delicate; hard to analyze |
| Redundancy reduction | Barlow Twins, VICReg | Variance and covariance terms | No negatives, no EMA | Large embedding dimensions |
| Masked modeling | MAE, BEiT | Reconstruction target | Cheap, scalable, best fine-tuning | Weak linear probe |
| Language supervision | CLIP | Paired text | Zero-shot, flexible | Needs large paired data |

Evaluation of SSL is a field of its own. The **linear probe** (freeze the encoder, train a linear classifier) measures linear separability. **$k$-NN** needs no training at all. **Fine-tuning** measures the quality of the initialization and tends to favour masked methods. **Few-label transfer** (1% or 10% of labels) shows the practical value. Always report which one you used.

## Language supervision: CLIP {#clip}

Everything so far removed labels but kept images alone. There is a second, enormous source of free supervision: the **text that already sits next to images** on the web (alt text, captions, surrounding sentences). The text is noisy, but it is *open vocabulary*: it can name anything, not just the 1,000 ImageNet classes. **CLIP** (Radford, Kim, Hallacy, Ramesh, Goh, Agarwal, Sastry, Askell, Mishkin, Clark, Krueger and Sutskever, ICML 2021, "Learning Transferable Visual Models From Natural Language Supervision") trained an image encoder and a text encoder from scratch on 400 million image-text pairs collected from the internet, with a contrastive objective that is InfoNCE, applied across modalities.

{{figure: week11-clip | CLIP. A batch of N image-text pairs yields N image embeddings and N text embeddings. All N x N cosine similarities, scaled by a learnable temperature, form a logit matrix whose diagonal holds the true pairs. A cross-entropy loss is applied along every row (which caption matches this image?) and every column (which image matches this caption?).}}

### The symmetric contrastive loss {#clip-loss}

Let $u_i=f(\text{image}_i)/\norm{f(\text{image}_i)}$ and $v_j=g(\text{text}_j)/\norm{g(\text{text}_j)}$ be unit-norm embeddings in a shared space of dimension $d$ (512 for the common ViT-B models), and let $s=e^{t}$ be a learnable scale equal to the inverse temperature, $1/\tau$. The logits are $\ell_{ij}=s\,u_i^\top v_j$, and the loss is the average of two cross-entropies,

$$
\mathcal{L}=\frac{1}{2N}\sum_{i=1}^N\left[-\log\frac{e^{\ell_{ii}}}{\sum_{j}e^{\ell_{ij}}}-\log\frac{e^{\ell_{ii}}}{\sum_{j}e^{\ell_{ji}}}\right].
$$

This is exactly the InfoNCE of the previous section, with the "two views" being a picture and its caption, and with the other $N-1$ captions (images) in the batch as negatives. Three practical facts matter. First, the temperature is **learned** (as the log of the scale, initialized at $1/0.07$ and clipped so the scale never exceeds 100), rather than tuned. Second, the batch is huge (32,768 in the paper), because more negatives make the task harder and the representation better; the batch is split across many devices, and each device computes only its slice of the similarity matrix after an all-gather of the embeddings. Third, there are no class labels anywhere: a "class" is whatever string appears in the caption.

```python:clip_loss.py
import torch
import torch.nn.functional as F

def clip_loss(img_emb, txt_emb, logit_scale):
    """img_emb, txt_emb: (N, d) raw encoder outputs. logit_scale: scalar parameter holding log(1/tau)."""
    u = F.normalize(img_emb, dim=-1)                    # (N, d) unit norm
    v = F.normalize(txt_emb, dim=-1)                    # (N, d) unit norm
    logits = logit_scale.exp().clamp(max=100) * u @ v.T # (N, N), row i = image i against all captions
    targets = torch.arange(len(u), device=u.device)     # the true match of row i is column i
    loss_i2t = F.cross_entropy(logits, targets)         # which caption goes with each image
    loss_t2i = F.cross_entropy(logits.T, targets)       # which image goes with each caption
    return (loss_i2t + loss_t2i) / 2

img, txt = torch.randn(8, 512), torch.randn(8, 512)
scale = torch.nn.Parameter(torch.tensor(1 / 0.07).log())
print(clip_loss(img, txt, scale).item())               # near log(8) = 2.08 plus a little, for random inputs
```

A later variant, **SigLIP** (Zhai, Mustafa, Kolesnikov and Beyer, ICCV 2023, "Sigmoid Loss for Language Image Pre-Training"), replaces the softmax over the batch by an independent binary (sigmoid) classification for each of the $N^2$ pairs, which removes the global normalization and makes scaling the batch simpler.

{{demo: week11-clip-similarity | Image-text similarity matrix, retrieval and the temperature | 540}}

### Zero-shot classification and prompt engineering {#zero-shot}

A trained CLIP is a classifier for free. To classify an image over $K$ classes you never trained on, write each class as a sentence, embed the $K$ sentences with the text encoder, and pick the class whose text embedding has the highest cosine similarity with the image embedding:

$$
\hat{y}=\argmax_{k}\; u^\top v_k,\qquad v_k=\frac{g(\text{"a photo of a } c_k\text{"})}{\norm{g(\text{"a photo of a } c_k\text{"})}} .
$$

The text encoder acts as a **hypernetwork that writes a linear classifier from a description**: the matrix of $K$ text embeddings *is* the weight matrix of a linear head on the image embedding. Because the training captions are sentences, not bare labels, the wording of the class prompt matters, and this is called **prompt engineering**. The word "crane" alone is ambiguous (the bird or the machine), "a photo of a crane" is a better match to caption statistics, and "a photo of a crane, a type of bird" disambiguates. The CLIP paper reports that the template "a photo of a {label}." alone improves ImageNet zero-shot accuracy by about 1.3 points over using the bare label, and that using task-specific templates (for example "a satellite photo of {label}." for remote sensing) and then **ensembling** the embeddings of many templates (averaging the normalized text embeddings, then renormalizing) together improve accuracy by almost 5 points on ImageNet. Prompt ensembling costs nothing at test time, since the $K$ class embeddings are computed once.

Headline numbers: the best original CLIP (ViT-L/14 at 336 pixels) reaches 76.2% top-1 zero-shot on ImageNet, matching a fully supervised ResNet-50 that was trained on the 1.28M labeled images, and it is far more robust to distribution shift (ImageNet-R, ImageNet-Sketch, ObjectNet) than a supervised model with the same ImageNet accuracy. It is weak at fine-grained or specialized tasks (counting, distances, medical, satellite) and on abstract or systematic tasks, and it is poor at some compositional questions, for instance it is nearly bag-of-words about "a horse eating grass" versus "grass eating a horse".

```python:clip_zero_shot.py
import torch, open_clip
from PIL import Image

model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
tokenizer = open_clip.get_tokenizer("ViT-B-32")
model.eval()

classes = ["cat", "dog", "airplane", "ship"]
templates = ["a photo of a {}.", "a blurry photo of a {}.", "a drawing of a {}.", "a photo of the large {}."]

@torch.no_grad()
def class_embeddings(classes, templates):
    W = []
    for c in classes:
        t = tokenizer([tpl.format(c) for tpl in templates])         # (T, 77) token ids
        e = model.encode_text(t)                                    # (T, 512)
        e = e / e.norm(dim=-1, keepdim=True)
        e = e.mean(0)                                               # prompt ensembling: average, then renormalize
        W.append(e / e.norm())
    return torch.stack(W)                                           # (K, 512) = weights of a linear classifier

W = class_embeddings(classes, templates)

@torch.no_grad()
def classify(path):
    img = preprocess(Image.open(path).convert("RGB")).unsqueeze(0)  # (1, 3, 224, 224)
    u = model.encode_image(img)                                     # (1, 512)
    u = u / u.norm(dim=-1, keepdim=True)
    probs = (100.0 * u @ W.T).softmax(dim=-1)                       # CLIP uses a scale of about 100
    return dict(zip(classes, probs[0].tolist()))
```

:::warning The softmax probabilities are not calibrated
The zero-shot "probabilities" come from a softmax over a fixed list of classes with a learned scale. They always sum to 1 across the candidates you provided, even when none of them applies. CLIP has no "none of the above" unless you add a class for it, so out-of-distribution inputs get confident, wrong predictions.
:::

### Retrieval, and what the embedding space looks like {#retrieval}

Retrieval is the same computation without the argmax. For **text-to-image** retrieval, embed the query, compute its dot product with a precomputed matrix of image embeddings (one per gallery image), and return the top-$k$. For **image-to-text**, swap the roles. With a few million images this is a single matrix multiplication or an approximate nearest-neighbour index. The standard metric is **Recall@K**, the fraction of queries whose correct match appears in the top $K$ results (Flickr30k and MS-COCO are the usual benchmarks).

Real CLIP embeddings show a **modality gap** (Liang, Zhang, Kwon, Yeung and Zou, NeurIPS 2022): image embeddings and text embeddings occupy two separate cones of the sphere, and even matched pairs have cosine similarity well below 1 (typically 0.2 to 0.35 for matching pairs). Only the *relative* order within a row or column matters for the loss and for retrieval, which is why the gap does not hurt accuracy; you can try that in the demo above. A practical consequence is that you should never interpret an absolute cosine similarity threshold across different models.

## Captioning, VQA and the road to multimodal LLMs {#multimodal}

CLIP *matches* images and text, but it does not *generate* text. The classical generative tasks are:

- **Image captioning.** Produce a sentence describing an image. The first neural approaches (Vinyals et al., "Show and Tell", CVPR 2015; Xu et al., "Show, Attend and Tell", ICML 2015) encoded the image with a CNN and decoded text with an LSTM, the latter attending to a grid of CNN features at each word. Today a Transformer decoder with cross-attention (or prefix tokens) over ViT features does the same job, trained with ordinary next-token cross-entropy on caption text.
- **Visual question answering (VQA).** Given an image and a question, produce an answer (Antol et al., "VQA", ICCV 2015). Early models treated it as classification over the most common answers after fusing image and question features. Modern systems generate the answer as text.

**BLIP** (Li, Li, Xiong and Hoi, ICML 2022) trains one model with three objectives at once, image-text contrastive (CLIP-like), image-text matching (a binary classifier on fused features that sees hard negatives) and caption generation, and bootstraps cleaner captions by filtering and re-captioning noisy web data.

### Connecting a vision encoder to a language model {#mllm}

The recipe that emerged in 2022 to 2023 is conceptually simple: **take a strong pretrained image encoder, a strong pretrained language model, and train a small connector between them**. Every part of the pipeline is something you have already seen: the image becomes a sequence of patch embeddings, the connector maps them into the LLM's embedding space, and the LLM treats them as extra "tokens" in its context and is trained with next-token prediction on the text that follows.

{{figure: week11-mllm | A multimodal LLM. A frozen or lightly tuned CLIP ViT turns a 336x336 image into 576 patch embeddings of width 1024. A connector (a small MLP, or a Q-Former with a fixed number of queries) projects them to the LLM width (4096 here). The projected visual tokens are concatenated with the text prompt tokens, and the decoder-only LLM predicts the answer with the usual causal next-token loss.}}

Three connector designs cover most systems:

| Design | How images enter the LLM | Examples |
| --- | --- | --- |
| **Linear / MLP projector** | Each patch embedding is mapped to one LLM token; the number of visual tokens equals the number of patches (576 for $336^2$ at $P=14$) | LLaVA (Liu, Li, Wu and Lee, NeurIPS 2023, "Visual Instruction Tuning"), LLaVA-1.5 |
| **Q-Former / resampler** | A small set of learned query vectors cross-attends to the image features and compresses them to a fixed number of tokens (e.g. 32) | BLIP-2 (Li, Li, Savarese and Hoi, ICML 2023), Flamingo's Perceiver resampler (Alayrac et al., NeurIPS 2022) |
| **Gated cross-attention layers** | New cross-attention layers are inserted between frozen LLM blocks, so visual features are attended to instead of being part of the context | Flamingo |

The trade-offs are simple. A projector keeps all spatial detail but makes the context long (576 tokens per image) and attention cost grows accordingly. A Q-Former compresses to 32 tokens, which is cheap but loses fine detail such as small text. Cross-attention keeps the LLM context free of images and preserves the LLM's text abilities, but adds new parameters throughout the network.

**Training** is typically in two stages. Stage 1 aligns the modalities: freeze the vision encoder and the LLM and train only the connector on image-caption pairs (hundreds of thousands to millions). Stage 2 is **visual instruction tuning**: unfreeze the connector (and often the LLM, or LoRA adapters from [Week 10](week-10.html#lora)) and train on conversations about images, such as questions, detailed descriptions and reasoning prompts, with the loss computed only on the answer tokens. LLaVA generated much of its instruction data by prompting a text-only language model with captions and bounding boxes, a form of the synthetic-data pipelines from [Week 10](week-10.html).

:::note What multimodal LLMs still get wrong
Since the vision encoder was trained for global image-text alignment, spatial and fine-grained abilities are weak: counting, reading small text, precise localization and spatial relations ("left of") are common failures, as is **hallucination**: describing objects that are not in the image because the language prior is stronger than the visual evidence. Higher-resolution inputs (tiling an image into crops), better vision encoders and grounding-style training data are active remedies.
:::

## Lab: fine-tune a ViT, CLIP zero-shot and retrieval, and a linear probe {#lab}

The lab has three parts, each runnable in a Colab notebook with a free GPU. You will need `pip install transformers datasets open_clip_torch scikit-learn`.

### Part A: fine-tune a pretrained ViT on CIFAR-10 {#lab-finetune}

CIFAR-10 images are $32\times32$, but ImageNet-pretrained ViTs expect $224\times224$ (and the position embeddings were learned for 196 patches). The simplest correct approach is to resize the images. We fine-tune `google/vit-base-patch16-224-in21k`, using a small learning rate and the standard warmup and weight decay.

```python:lab_finetune_vit.py
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import ViTForImageClassification, get_cosine_schedule_with_warmup

device = "cuda" if torch.cuda.is_available() else "cpu"
mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)          # this checkpoint was pretrained with mean=std=0.5
tf_train = transforms.Compose([transforms.Resize(224), transforms.RandomHorizontalFlip(),
                               transforms.ToTensor(), transforms.Normalize(mean, std)])
tf_test = transforms.Compose([transforms.Resize(224), transforms.ToTensor(), transforms.Normalize(mean, std)])
train = datasets.CIFAR10("data", train=True, download=True, transform=tf_train)
test = datasets.CIFAR10("data", train=False, download=True, transform=tf_test)
train_dl = DataLoader(train, batch_size=64, shuffle=True, num_workers=2)
test_dl = DataLoader(test, batch_size=256, num_workers=2)

model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224-in21k", num_labels=10).to(device)   # new 768 -> 10 head, randomly initialised

epochs = 1
decay = [p for n, p in model.named_parameters() if p.ndim > 1]        # decay weights, not biases or LayerNorm
no_decay = [p for n, p in model.named_parameters() if p.ndim <= 1]
opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.05},
                         {"params": no_decay, "weight_decay": 0.0}], lr=3e-5)
steps = epochs * len(train_dl)
sched = get_cosine_schedule_with_warmup(opt, int(0.1 * steps), steps)
scaler = torch.cuda.amp.GradScaler(enabled=device == "cuda")

for epoch in range(epochs):
    model.train()
    for i, (x, y) in enumerate(train_dl):
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"):
            logits = model(pixel_values=x).logits                      # (64, 10)
            loss = nn.functional.cross_entropy(logits, y, label_smoothing=0.1)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); sched.step()
        if i % 100 == 0:
            print(f"epoch {epoch} step {i}/{len(train_dl)} loss {loss.item():.3f}")

model.eval(); correct = 0
with torch.no_grad():
    for x, y in test_dl:
        correct += (model(pixel_values=x.to(device)).logits.argmax(-1).cpu() == y).sum().item()
print("test accuracy:", correct / len(test))                           # typically about 0.97 to 0.99 after one epoch
```

Questions to explore afterwards: what happens with learning rate $10^{-3}$ (see the pitfalls section)? What if you freeze everything but the head (a linear probe on supervised features) versus unfreezing the last 4 blocks versus all? How does accuracy change if you do not resize and instead interpolate the position embeddings for $32\times32$ inputs with $P=4$?

### Part B: CLIP zero-shot classification and retrieval {#lab-clip}

We evaluate zero-shot accuracy on the CIFAR-10 test set with the class-embedding code from above, then use the same embeddings for text-to-image retrieval over 1,000 test images.

```python:lab_clip_eval.py
import torch, open_clip
from torchvision import datasets

device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
model = model.to(device).eval()
tok = open_clip.get_tokenizer("ViT-B-32")

test = datasets.CIFAR10("data", train=False, download=True, transform=preprocess)   # preprocess upsamples to 224
classes = test.classes                                  # ['airplane', 'automobile', 'bird', ...]
templates = ["a photo of a {}.", "a blurry photo of a {}.", "a low resolution photo of a {}.", "a photo of the {}."]

@torch.no_grad()
def build_classifier():
    W = []
    for c in classes:
        e = model.encode_text(tok([t.format(c) for t in templates]).to(device))
        e = e / e.norm(dim=-1, keepdim=True)
        W.append(torch.nn.functional.normalize(e.mean(0), dim=0))
    return torch.stack(W)                               # (10, 512)

W = build_classifier()
loader = torch.utils.data.DataLoader(test, batch_size=256, num_workers=2)

feats, labels = [], []
with torch.no_grad():
    for x, y in loader:
        u = model.encode_image(x.to(device))
        feats.append(torch.nn.functional.normalize(u, dim=-1).cpu()); labels.append(y)
feats, labels = torch.cat(feats), torch.cat(labels)     # (10000, 512), (10000,)

pred = (feats @ W.cpu().T).argmax(-1)
print("zero-shot accuracy:", (pred == labels).float().mean().item())      # roughly 0.9 for this model

# text -> image retrieval: precision@10 for a free-form query
def search(query, k=10):
    with torch.no_grad():
        q = torch.nn.functional.normalize(model.encode_text(tok([query]).to(device)), dim=-1).cpu()
    scores = (feats @ q.T).squeeze(1)                   # (10000,) cosine similarity to every image
    top = scores.topk(k).indices
    return top, [classes[labels[i]] for i in top]

top, names = search("a photo of a horse")
print(names)                                            # mostly 'horse'
```

Compare the accuracy with and without prompt ensembling, with the bare class name as the prompt, and with a deliberately bad prompt ("a photo of a {} on the moon"). Then query for something no label covers ("a red vehicle", "an animal with four legs on grass") and inspect the results.

### Part C: linear probe of self-supervised features {#lab-probe}

Extract frozen features from a self-supervised model and train a logistic regression on them. We use DINO ViT-S/16 (`facebook/dino-vits16`), whose class-token features are known to be linearly separable. The probe protocol: freeze the encoder, extract features once, standardize them, and fit a regularized linear classifier.

```python:lab_linear_probe.py
import numpy as np, torch
from torchvision import datasets, transforms
from transformers import AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

device = "cuda" if torch.cuda.is_available() else "cpu"
encoder = AutoModel.from_pretrained("facebook/dino-vits16").to(device).eval()     # ViT-S/16, d = 384
tf = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),
                         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

def extract(train):
    ds = datasets.CIFAR10("data", train=train, download=True, transform=tf)
    dl = torch.utils.data.DataLoader(ds, batch_size=256, num_workers=2)
    X, Y = [], []
    with torch.no_grad():
        for x, y in dl:
            h = encoder(pixel_values=x.to(device)).last_hidden_state          # (B, 197, 384)
            X.append(h[:, 0].cpu().numpy()); Y.append(y.numpy())              # class token: (B, 384)
    return np.concatenate(X), np.concatenate(Y)

Xtr, ytr = extract(True); Xte, yte = extract(False)
print(Xtr.shape, Xte.shape)                                                   # (50000, 384) (10000, 384)

scaler = StandardScaler().fit(Xtr)
for n in [100, 1000, 50000]:                                                  # few-label regime: n labeled examples
    idx = np.random.default_rng(0).permutation(len(Xtr))[:n]
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(scaler.transform(Xtr[idx]), ytr[idx])
    print(n, "labels -> test accuracy", clf.score(scaler.transform(Xte), yte))

# k-NN on L2-normalised features needs no training at all
from sklearn.neighbors import KNeighborsClassifier
norm = lambda a: a / np.linalg.norm(a, axis=1, keepdims=True)
knn = KNeighborsClassifier(n_neighbors=20, metric="cosine").fit(norm(Xtr), ytr)
print("20-NN accuracy:", knn.score(norm(Xte), yte))
```

Extensions: swap the encoder for an MAE checkpoint (`facebook/vit-mae-base`) and compare its linear-probe accuracy to DINO's; the gap illustrates the earlier point about masked features. Probe the **patch tokens** instead of the class token and visualize the first three principal components as colors, which gives the well-known DINO "semantic segmentation for free" picture.

## Common mistakes and debugging tips {#mistakes}

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| Fine-tuned ViT diverges or accuracy collapses to chance in the first epoch | Learning rate too high for fine-tuning (e.g. $10^{-3}$ with Adam) or no warmup | Use $10^{-5}$ to $10^{-4}$ with AdamW, a warmup of a few hundred steps, gradient clipping at 1.0 |
| Pretrained ViT gives poor accuracy with the right code | Wrong normalization constants or image size; mean/std of the checkpoint differ (0.5 vs ImageNet statistics) | Use the checkpoint's own `image_processor`/`preprocess` |
| `size mismatch` for position embeddings when changing resolution | Position embeddings are tied to the number of patches | Resize via 2-D bicubic interpolation of the $14\times14$ grid (not of the flattened sequence), keep the class token separate |
| SimCLR loss stuck at $\log(2N-1)$ | Learning rate too low or augmentations too weak, or no projection head, or the diagonal was not masked | Check the loss at initialization, mask the diagonal, use a LARS/AdamW setup with warmup, use strong crops and color jitter |
| Contrastive loss drops to near 0 immediately | Information leak: both views share low-level cues (same crop, same color histogram) or negatives contain the positive | Use random resized crops with a small minimum scale and color jitter; check indexing of the positive |
| BYOL/DINO outputs become constant, k-NN accuracy at chance | Collapse: missing stop-gradient, EMA momentum too low, or no centering/sharpening in DINO | Detach the target, use momentum 0.99 to 0.996 (increase to 1 over training), center and sharpen teacher outputs |
| CLIP zero-shot much worse than reported | Prompts that are bare class names, image preprocessing different from training, or class names that are ambiguous | Use templates and ensembling, the model's own `preprocess`, disambiguating class names |
| CLIP training: loss does not go below $\log N$ | Batch contains duplicates (same caption twice) or the logit scale is stuck at 1 | Deduplicate within a batch, learn `logit_scale` with a normal learning rate and the clamp at 100 |
| MAE reconstructions look blurry gray | Normal: the loss is MSE on masked patches, which averages over possible fills | Not a bug. Judge by fine-tuning accuracy, and consider `norm_pix_loss` |
| Linear probe accuracy lower than expected | Features not standardized, regularization $C$ untuned, or the wrong layer used | Standardize, sweep $C$ on a validation split, try the average of patch tokens or the last four blocks |
| Out-of-memory when training a ViT at high resolution | Attention memory is quadratic in the number of tokens | Use FlashAttention (`F.scaled_dot_product_attention`), gradient checkpointing, a larger patch size or a windowed model |

## Exercises {#exercises}

:::exercise Exercise 1: shapes through ViT-L/14
A ViT-L/14 takes $224\times224$ RGB inputs. Give the number of patches, the shape of the patch matrix before the linear embedding, the shape of the token sequence entering the first block, the per-head dimension, and the number of parameters in the patch embedding (weights and bias).

:::details Solution
Patches: $(224/14)^2=16^2=256$. Each patch has $14\cdot14\cdot3=588$ values, so the unrolled patch matrix is $(256, 588)$. With width $D=1024$ the embedding matrix is $588\times1024$ and the sequence after adding the class token is $(257, 1024)$ (batch dimension excluded). With 16 heads, the per-head dimension is $1024/16=64$. Patch embedding parameters: $588\cdot1024+1024=603{,}136$. The position embedding has $257\cdot1024$ entries.
:::
:::

:::exercise Exercise 2: attention cost
For a ViT-B/16 block ($D=768$) compute the FLOPs of the attention score and weighted-sum matrix multiplications ($2n^2D$ multiply-accumulates) versus the linear layers ($12nD^2$) at $224$, $512$ and $1024$ pixels, and find the resolution at which the two are equal. Then state what the equivalent crossover would be for Swin with $7\times7$ windows.

:::details Solution
Equality: $2n^2D=12nD^2\Rightarrow n=6D=4608$ tokens, i.e. $\sqrt{4608}\approx68$ patches per side, or about $1086$ pixels at $P=16$. At $224$ ($n=197$): attention $2\cdot197^2\cdot768\approx0.06$ G MACs against $1.4$ G for the linear part. At $512$ ($n=1025$): $1.6$ G against $7.3$ G. At $1024$ ($n=4097$): $25.8$ G against $29.0$ G. In windowed attention each token attends to $49$ keys, so the attention term is $2\cdot n\cdot49\cdot D$, so the ratio to the linear layers is $2\cdot49\,nD/(12nD^2)=49/(6D)\approx1\%$ at every resolution: there is no crossover, the cost is linear in $n$.
:::
:::

:::exercise Exercise 3: InfoNCE limits
Prove that for a batch of $N$ candidates, the InfoNCE loss of any encoder is at least 0 and that the loss of an encoder whose scores are all equal is exactly $\log N$. Then use the bound $I(x;c)\ge\log N-\mathcal{L}_{\text{InfoNCE}}$ to explain why using more negatives can only tighten what you can *certify*, and why a small batch can never certify more than $\log N$ nats.

:::details Solution
The loss is $-\log p_\text{pos}$ with $p_\text{pos}\in(0,1]$, so it is $\ge0$. If all scores are equal, the softmax is uniform and $p_\text{pos}=1/N$, giving $\log N$. Because $\mathcal{L}\ge0$, the bound $\log N-\mathcal{L}\le\log N$: whatever the encoder does, the bound cannot certify more than $\log N$ nats of mutual information. With $N=256$ that is $5.5$ nats, so for representations that share more than that, a larger batch is needed for the estimate to stay informative. This is one reason SimCLR and CLIP use huge batches (and MoCo uses a queue).
:::
:::

:::exercise Exercise 4: gradient of the temperature
For anchor $i$ with similarities $s_{ik}$ and softmax probabilities $p_k$ at temperature $\tau$, show that the gradient with respect to the positive similarity is $-(1-p_\text{pos})/\tau$ and with respect to a negative $k$ is $+p_k/\tau$. For $\tau=0.05$ and $\tau=1$ with similarities (positive 0.8, negatives 0.7, 0.5, 0.0, 0.0), compute which negative gets the largest push in each case.

:::details Solution
Write $\ell=-s_\text{pos}/\tau+\log\sum_m e^{s_m/\tau}$. Differentiating, $\partial\ell/\partial s_\text{pos}=-1/\tau+p_\text{pos}/\tau$ and $\partial\ell/\partial s_k=p_k/\tau$ for a negative. With $\tau=1$, the exponentials are $e^{0.8}=2.23$, $e^{0.7}=2.01$, $e^{0.5}=1.65$, $1$, $1$, sum $7.89$, so $p=(0.28,0.255,0.21,0.127,0.127)$: pushes of $0.255$, $0.21$, $0.127$, $0.127$, all comparable. With $\tau=0.05$ the logits are $16, 14, 10, 0, 0$: $p_\text{pos}=0.88$, $p_{0.7}=0.12$, $p_{0.5}=0.0022$, and the two zeros are about $10^{-7}$. The hardest negative receives almost all the push, scaled by $1/\tau=20$, i.e. $\approx2.4$, while the easy ones are ignored: small $\tau$ is a soft hard-negative miner.
:::
:::

:::exercise Exercise 5: coding the symmetric loss
Implement the CLIP loss with a *manual* log-softmax in NumPy (no library cross-entropy), check it against the PyTorch function on random data, and verify numerically that the loss is $\log N$ when all embeddings are identical.

:::details Solution
```python
import numpy as np

def logsumexp(a, axis):
    m = a.max(axis=axis, keepdims=True)
    return (m + np.log(np.exp(a - m).sum(axis=axis, keepdims=True))).squeeze(axis)

def clip_loss_np(U, V, scale):
    U = U / np.linalg.norm(U, axis=1, keepdims=True)
    V = V / np.linalg.norm(V, axis=1, keepdims=True)
    L = scale * U @ V.T                                  # (N, N)
    diag = np.diag(L)
    row = (logsumexp(L, 1) - diag).mean()                # image -> text
    col = (logsumexp(L, 0) - diag).mean()                # text -> image
    return (row + col) / 2

rng = np.random.default_rng(0)
N = 16
same = np.tile(rng.normal(size=(1, 8)), (N, 1))
print(clip_loss_np(same, same, 50.0), np.log(N))         # both 2.7726
```
Subtracting the max inside `logsumexp` prevents overflow with large scales. With identical embeddings every logit equals the same constant so each softmax is uniform.
:::
:::

:::exercise Exercise 6: why no negatives does not collapse in BYOL (conceptual, challenging)
Explain in a paragraph why BYOL does not collapse even though it has no negatives, mentioning the predictor, the stop-gradient and the EMA. Then predict what happens if (a) you remove the predictor, (b) you remove the stop-gradient and train both branches, (c) you set the EMA momentum to 0 (the target is a copy of the online network at every step).

:::details Solution
The target network gives a *moving target* that is not directly optimized by the loss: the online network is trained to predict it, but the target is only an EMA of the online weights, so it is a slightly older, smoother version of the same function. The predictor makes the two sides asymmetric, so the online side must forecast the target's embedding of another view, not simply copy it. Analyses (e.g. SimSiam, Chen and He, CVPR 2021) show that the stop-gradient together with the predictor behaves like an alternating optimization and, empirically, a constant is an unstable fixed point of that dynamic. (a) Without the predictor, BYOL (and SimSiam) typically collapses or degrades sharply; the predictor is essential. (b) Without the stop-gradient, the minimizer of the plain alignment loss is a constant, and training collapses quickly (SimSiam reports collapse to chance). (c) With momentum 0, the target equals the current online network at each step, a stop-gradient-only variant. SimSiam is precisely this case, and it works with the predictor and stop-gradient, though it is more sensitive to hyperparameters than the EMA version; BYOL reports a degradation when the EMA is removed or made too fast.
:::
:::

:::exercise Exercise 7: MAE masking arithmetic
A ViT-B/16 MAE processes $224\times224$ images at a 75% mask ratio. (a) How many tokens does the encoder process, including the class token? (b) The decoder is 8 blocks at width 512: how many tokens does it process? (c) Estimate the ratio of encoder cost with masking versus without, ignoring the attention term. (d) What would happen to the representation quality if the ratio were reduced to 15%?

:::details Solution
(a) $196\cdot0.25=49$ visible patches plus the class token: $50$. (b) The decoder re-inserts mask tokens, so it sees all $196$ plus the class token: $197$. (c) The encoder cost is linear in the token count when the attention term is ignored: $50/197\approx0.25$, a $4\times$ saving. (d) With only 15% masked, the missing patches can be inferred from their neighbours by local interpolation, so the pretext task is nearly trivial and the network learns low-level features. The MAE paper reports that high ratios (about 75%) give the best fine-tuning and linear-probe results, and that BERT-style 15% is far from optimal for images.
:::
:::

:::exercise Exercise 8: collapse and its two cures (coding, challenging)
In NumPy, generate two Gaussian clusters in 2-D. Train a linear encoder $z=Wx$ with $W\in\R^{2\times2}$ using the alignment loss $\norm{z_a-z_b}^2$ on two noisy views of each point. Then add (i) a VICReg-style variance hinge $\sum_j\max(0,1-\mathrm{std}(z_j))$ and (ii) a covariance penalty on the off-diagonal entry of the embedding covariance. Report the singular values of $W$ and the effective rank of the embeddings in each of the three cases and explain the result.

:::details Solution
```python
import numpy as np
rng = np.random.default_rng(0)
X = np.vstack([rng.normal([-1, 0], 0.5, (500, 2)), rng.normal([1, 0], 0.5, (500, 2))])

def train(var_w=0.0, cov_w=0.0, steps=1000, lr=0.05):
    W = rng.normal(size=(2, 2)) * 0.5
    for _ in range(steps):
        a, b = X + 0.3 * rng.normal(size=X.shape), X + 0.3 * rng.normal(size=X.shape)
        za, zb = a @ W.T, b @ W.T
        g = 2 * (za - zb) / len(X)                       # gradient of mean ||za - zb||^2 w.r.t. za
        dW = g.T @ a - g.T @ b
        for z, x in ((za, a), (zb, b)):
            zc = z - z.mean(0)
            s = zc.std(0) + 1e-6
            gz = -var_w * (s < 1) / s * zc / len(z)      # hinge max(0, 1 - std_j)
            c = (zc[:, 0] * zc[:, 1]).mean()             # covariance between the two dimensions
            gz += cov_w * 2 * c * zc[:, ::-1] / len(z)   # gradient of c^2
            dW += gz.T @ x
        W -= lr * dW
    Z = X @ W.T; Z = Z - Z.mean(0)
    sv = np.linalg.svd(Z, compute_uv=False); p = sv / sv.sum()
    return np.linalg.svd(W, compute_uv=False), np.exp(-(p * np.log(p + 1e-12)).sum())

for name, kw in [("alignment only", {}), ("+ variance", dict(var_w=1.0)), ("+ variance + covariance", dict(var_w=1.0, cov_w=5.0))]:
    sv, er = train(**kw)
    print(f"{name:26s} singular values of W {np.round(sv, 3)}  effective rank {er:.2f}")
```
Output from one run: alignment only gives singular values $(0, 0)$, complete collapse (the "effective rank 1.79" printed there is computed from numerically tiny values and is meaningless). With the variance term the embedding has unit spread, but one singular value of $W$ is 0: both output dimensions are the same feature, so the rank is 1.00 (dimensional collapse). Adding the covariance penalty decorrelates the dimensions: singular values $(1.76, 0.93)$ and effective rank 1.99. This is the logic of VICReg: variance prevents complete collapse, covariance prevents redundancy.
:::
:::

## Key takeaways

:::key Key takeaways
- A ViT is a Transformer applied to a sequence of linearly embedded image patches: $224^2$ with $P=16$ gives 196 tokens plus a class token, width $D=768$ for ViT-B, 12 heads of dimension 64, and a classifier on the class token (or mean-pooled tokens).
- ViT lacks the CNN's locality and translation-equivariance priors, so it needs more data or stronger recipes: DeiT showed that augmentation, regularization and a distillation token let ViTs train on ImageNet-1k, and Swin reintroduces locality and hierarchy with shifted windows and linear cost.
- Quadratic attention cost dominates at high resolution; windowed and hierarchical designs fix it.
- DETR casts detection as set prediction with a Hungarian matching loss and no NMS; SAM is a promptable segmentation model built from an MAE-pretrained image encoder, a prompt encoder and a light mask decoder.
- InfoNCE is a categorical cross-entropy that lower-bounds mutual information by $\log N-\mathcal{L}$; temperature controls how much hard negatives dominate.
- Collapse is the central problem of SSL. Negatives (SimCLR, MoCo), asymmetry plus stop-gradient and EMA (BYOL, DINO), variance and covariance terms (VICReg, Barlow Twins) and masking with reconstruction (MAE) each rule it out differently.
- Contrastive and DINO features are linearly separable; MAE features are better for fine-tuning than for linear probing.
- CLIP applies a symmetric InfoNCE to image-text pairs; zero-shot classification builds a linear classifier from text prompts, and prompt wording and ensembling matter.
- Multimodal LLMs connect a pretrained vision encoder to an LLM through a projector or Q-Former and are tuned in two stages (alignment, then visual instruction tuning). Hallucination and fine-grained perception are their main weaknesses.
- Always report the evaluation protocol for SSL features (linear probe, k-NN, fine-tune, few-label), and match recipes when comparing architectures.
:::

## Further reading {#further-reading}

**Essential**
- Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale", ICLR 2021 (arXiv:2010.11929). **Essential.**
- Radford et al., "Learning Transferable Visual Models From Natural Language Supervision" (CLIP), ICML 2021 (arXiv:2103.00020). **Essential.**
- He et al., "Masked Autoencoders Are Scalable Vision Learners", CVPR 2022 (arXiv:2111.06377). **Essential.**

**Vision Transformers and detection**
- Touvron et al., "Training data-efficient image transformers and distillation through attention" (DeiT), ICML 2021 (arXiv:2012.12877).
- Liu et al., "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows", ICCV 2021 (arXiv:2103.14030).
- Carion et al., "End-to-End Object Detection with Transformers" (DETR), ECCV 2020 (arXiv:2005.12872).
- Kirillov et al., "Segment Anything", ICCV 2023 (arXiv:2304.02643).
- Liu et al., "A ConvNet for the 2020s", CVPR 2022 (arXiv:2201.03545).

**Self-supervised learning**
- Oord, Li and Vinyals, "Representation Learning with Contrastive Predictive Coding" (InfoNCE), 2018 (arXiv:1807.03748).
- Chen, Kornblith, Norouzi and Hinton, "A Simple Framework for Contrastive Learning of Visual Representations" (SimCLR), ICML 2020 (arXiv:2002.05709).
- He et al., "Momentum Contrast for Unsupervised Visual Representation Learning" (MoCo), CVPR 2020 (arXiv:1911.05722).
- Grill et al., "Bootstrap Your Own Latent" (BYOL), NeurIPS 2020 (arXiv:2006.07733).
- Caron et al., "Emerging Properties in Self-Supervised Vision Transformers" (DINO), ICCV 2021 (arXiv:2104.14294).
- Chen and He, "Exploring Simple Siamese Representation Learning" (SimSiam), CVPR 2021 (arXiv:2011.10566).
- Wang and Isola, "Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere", ICML 2020 (arXiv:2005.10242).

**Multimodal models**
- Zhai et al., "Sigmoid Loss for Language Image Pre-Training" (SigLIP), ICCV 2023 (arXiv:2303.15343).
- Li et al., "BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation", ICML 2022 (arXiv:2201.12086).
- Li et al., "BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models", ICML 2023 (arXiv:2301.12597).
- Alayrac et al., "Flamingo: a Visual Language Model for Few-Shot Learning", NeurIPS 2022 (arXiv:2204.14198).
- Liu et al., "Visual Instruction Tuning" (LLaVA), NeurIPS 2023 (arXiv:2304.08485).
- Liang et al., "Mind the Gap: Understanding the Modality Gap in Multi-modal Contrastive Representation Learning", NeurIPS 2022 (arXiv:2203.02053).
