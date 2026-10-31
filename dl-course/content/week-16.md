---
title: "Efficient and Scalable Deep Learning"
phase: Advanced Topics
summary: Making deep learning fit, run fast and ship. GPU memory hierarchy, arithmetic intensity and the roofline model, profiling and bottleneck analysis, mixed precision (FP16 vs BF16, loss scaling), gradient accumulation and checkpointing (the sqrt(n) trade-off derived), data/tensor/pipeline parallelism, ZeRO and FSDP memory accounting, FlashAttention (online softmax derived), KV caching and speculative decoding, pruning, quantization (PTQ, QAT, int8/int4, GPTQ/AWQ), distillation, low-rank factorization, Mixture-of-Experts, NAS, and deployment with ONNX, TorchScript, torch.compile and serving. The lab profiles a training loop and applies AMP, quantization and distillation while reporting accuracy and latency trade-offs.
---

:::key Learning objectives
By the end of this week you should be able to:
- Describe the GPU memory hierarchy and explain why most deep-learning operations are limited by memory traffic rather than arithmetic.
- Compute the arithmetic intensity of an operation, place it on a roofline, and decide whether it is memory-bound or compute-bound.
- Profile a PyTorch training loop with `torch.profiler`, read the result, and separate data-loading, launch-overhead, memory-bound and compute-bound bottlenecks.
- Explain FP16, BF16 and TF32, why FP16 needs loss scaling and BF16 usually does not, and train with `torch.autocast`.
- Use gradient accumulation to emulate large batches, and gradient checkpointing to trade compute for memory, deriving the $\sqrt n$ optimum.
- Account for training memory (parameters, gradients, optimizer states, activations) and predict what ZeRO stages 1 to 3 and FSDP save.
- Explain data, tensor, pipeline and hybrid parallelism, ring all-reduce cost, and the pipeline bubble fraction.
- Derive online softmax and the tiled FlashAttention algorithm, and explain why it is faster although it does not reduce FLOPs.
- Explain KV caching, its memory cost, and the idea behind speculative decoding.
- Apply and analyze pruning, post-training and quantization-aware quantization, distillation with the Hinton loss, and low-rank factorization.
- Describe sparse Mixture-of-Experts routing and the load-balancing loss, and give an overview of neural architecture search.
- Export and accelerate a model with `torch.compile`, TorchScript and ONNX, and reason about serving and edge constraints.
:::

For the past fifteen weeks the question was "what model and what loss". This week the question is "what does it cost, and how do I make it cheaper without making it worse". The question has become unavoidable. A 7-billion-parameter language model needs over 100 GB just to *train* with Adam, which is more than any single GPU holds. Serving the same model to many users is limited less by arithmetic than by how fast bytes can be moved out of memory. A model that scores well in a notebook but needs a data-center GPU is useless on a phone. Efficiency is also the precondition for research: the experiments in your project run only as fast as your training loop does, and a loop that wastes 70% of the GPU makes you three times slower than you need to be.

The unifying idea is that **computation is cheap and data movement is expensive**. A modern GPU can perform hundreds of floating-point operations in the time it takes to fetch one number from main memory. Almost every technique this week is a way of reducing data movement, reducing the number of bytes per number, or splitting the data over more devices: mixed precision and quantization shrink the bytes; FlashAttention and operator fusion avoid round trips to memory; checkpointing and ZeRO trade extra computation or communication for less storage; pruning, distillation and low-rank factorization shrink the model itself; parallelism spreads what remains over many chips. Keep the question "where are the bytes, and how many times do they move?" in mind throughout.

The week is organized as a ladder. We start with the hardware and a simple model of performance (the roofline), then learn to measure (profiling). Next come the training-side tools that fit bigger models and batches onto a device (mixed precision, accumulation, checkpointing), then distributed training for when one device is not enough. FlashAttention and KV caching apply the same logic to the transformer's most expensive operator. We then turn to making finished models smaller and faster (compression, Mixture-of-Experts, architecture search), and finish with deployment and a lab that measures the trade-offs.

## Hardware: how a GPU actually works {#hardware}

### Throughput processors and the memory hierarchy

A CPU is built to run one thread of irregular code quickly: big caches, branch prediction, out-of-order execution. A **GPU** is built to run thousands of nearly identical threads at once and to hide the latency of memory behind sheer parallelism. An NVIDIA A100, a typical 2020-era training GPU, has 108 **streaming multiprocessors (SMs)**. Each SM contains 64 FP32 cores, four **tensor cores** (units that multiply small matrix tiles, for example $16\times 8\times 16$, in a single instruction) and a register file. Threads are grouped into **warps** of 32 that execute the same instruction in lockstep, and warps are grouped into **thread blocks** that are scheduled onto SMs. The tensor cores are where deep learning lives: on the A100 they deliver 312 TFLOP/s of dense BF16 or FP16 arithmetic, against about 19.5 TFLOP/s for ordinary FP32 cores. That 16-fold gap is a major reason mixed precision matters.

Arithmetic is only half the story. Data must reach the cores, and the memory system is a hierarchy:

{{figure: week16-memory-hierarchy | The memory hierarchy of an A100-class GPU. Each level down is roughly an order of magnitude larger and slower. Figures are rounded vendor-spec values (A100 80 GB: about 2.0 TB/s HBM; the 19 TB/s SRAM figure is the aggregate on-chip bandwidth quoted in the FlashAttention paper).}}

- **Registers** are private to each thread and effectively free to access, but scarce (256 KB per SM).
- **Shared memory / L1** is a programmer-managed scratchpad on each SM, up to 164 KB, with bandwidth an order of magnitude above HBM. Fast kernels load a tile of data here once and reuse it many times. We call this level **SRAM** below.
- **L2 cache** (40 MB on A100) is shared across SMs.
- **HBM (high-bandwidth memory)**, also called global or device memory, is the 40 or 80 GB where your tensors live. Its bandwidth is 1.5 to 2.0 TB/s on an A100 and 3.35 TB/s on an H100 SXM. This number sets the speed of every operation that touches each element once.
- **Host memory** is reached over PCIe (about 32 GB/s per direction for PCIe 4.0 x16), roughly 60 times slower than HBM. Moving tensors between CPU and GPU in the inner loop is a classic performance bug. Between GPUs, **NVLink** provides about 600 GB/s per GPU on the A100 and 900 GB/s on the H100, and between servers a network (InfiniBand or Ethernet) provides 10 to 50 GB/s per GPU.

The cost of moving data dominates the cost of computing with it. A useful mental model: the A100's tensor cores can perform about 150 FLOPs in the time HBM delivers a single byte (312 TFLOP/s divided by 2 TB/s), which is about 300 FLOPs per 2-byte FP16 number. Any operation that does only a handful of FLOPs per number it reads is therefore starved for data.

### TPUs and other accelerators

Google's **TPU** takes a different design point. Instead of many programmable SMs, a TPU core is built around a large **systolic array** (a $128\times128$ grid of multiply-accumulate units in the early generations) fed from on-chip memory, with a compiler (XLA) that plans all data movement ahead of time. TPUs are connected in a fixed 2-D or 3-D torus network ("pods") that makes the collective operations used in large-scale training fast. The preferred number format is BF16, which Google introduced for this hardware. The trade-off against GPUs is flexibility: TPUs favour large, regular, statically shaped computations and reward you for expressing your program in JAX or through XLA. The concepts in this chapter (bandwidth versus compute, memory hierarchy, collective communication) apply to both. Other accelerators (AMD Instinct, Intel Gaudi, Apple's GPU and Neural Engine, mobile NPUs) share the same structure: a lot of matrix arithmetic fed by a comparatively narrow memory pipe.

:::history Why BF16 exists
Mixed-precision training with FP16 was popularized by Micicevicius et al. (2018, NVIDIA and Baidu). Google Brain designed **bfloat16** ("brain floating point") for TPUs: take the FP32 format, keep all 8 exponent bits and truncate the mantissa to 7 bits. Conversion from FP32 is just dropping the low 16 bits, and the dynamic range equals FP32's, which removes the need for loss scaling. NVIDIA added BF16 tensor cores in the A100 (2020), and it has since become the default training format for large models.
:::

## Arithmetic intensity and the roofline model {#roofline}

### Two ways to be slow

Any kernel needs to do $F$ floating-point operations and to move $Q$ bytes between main memory and the chip. If the device can perform $P$ FLOP/s and move $B$ bytes/s, the kernel cannot finish faster than the larger of the two times:

$$
t \;\ge\; \max\!\left(\frac{F}{P},\; \frac{Q}{B}\right).
$$

Dividing by $F$ and defining the **arithmetic intensity** $I = F/Q$ (FLOP per byte moved), the best achievable throughput is

$$
\text{attainable FLOP/s} \;=\; \min\bigl(P,\; B\cdot I\bigr).
$$

Plotted against $I$ on log-log axes this is the **roofline model** (Williams, Waterman and Patterson, 2009): a sloped line of slope $B$ (the "memory roof") that meets a flat line at height $P$ (the "compute roof") at the **ridge point** $I^* = P/B$. An operation left of the ridge is **memory-bound**: speeding up the arithmetic does nothing, only moving fewer bytes helps. An operation right of the ridge is **compute-bound**: only faster math (tensor cores, lower precision) helps.

{{figure: week16-roofline | The roofline model. Throughput is limited by memory bandwidth on the sloped part and by peak compute on the flat part. The ridge point $I^*=P/B$ is the intensity an operation needs before the arithmetic units, not memory, become the bottleneck.}}

For the A100 with BF16 tensor cores, $P = 312$ TFLOP/s and $B \approx 2.0$ TB/s, so $I^* \approx 153$ FLOP/byte. For the H100 SXM, $P \approx 989$ TFLOP/s (dense BF16) and $B = 3.35$ TB/s, so $I^* \approx 295$. The ridge has moved right with each generation, because compute grows faster than bandwidth: more and more operations fall on the memory-bound side.

### Intensity of the operations you use

:::math Arithmetic intensity of a matrix multiply
Multiply $A\in\R^{M\times K}$ by $B\in\R^{K\times N}$ into $C\in\R^{M\times N}$, with $b$ bytes per element. The work is $F = 2MNK$ (one multiply and one add per term). The *minimum* traffic reads $A$ and $B$ and writes $C$: $Q = b(MK + KN + MN)$. So

$$
I = \frac{2MNK}{b\,(MK+KN+MN)}.
$$

*Square case.* With $M=N=K=n$: $I = \dfrac{2n^3}{3bn^2} = \dfrac{2n}{3b}$. In FP16 ($b=2$) this is $n/3$. To pass the A100 ridge of 153 you need $n \gtrsim 460$. A $256\times256$ FP16 matmul has $I\approx 85$ and runs at only $2.0\text{ TB/s}\times 85 = 174$ TFLOP/s at best, while a $1024\times1024$ one is compute-bound.

*Matrix-vector case* ($M=1$, the situation when generating one token at a time). The term $KN$ (the weight matrix) dominates the traffic: $I\approx \dfrac{2NK}{bNK}=\dfrac 2b$, which is **1 FLOP/byte in FP16**, 150 times below the ridge. The GPU runs at under 1% of peak. The only way to speed up decoding is to move fewer bytes: quantize the weights, batch more requests together (raising $M$), or use a smaller model.
:::

Elementwise operations are even worse. Adding two FP16 tensors performs 1 FLOP per element but moves 6 bytes (read two, write one): $I\approx 0.17$. The same holds for ReLU, dropout, residual additions, LayerNorm and softmax, which perform a few FLOPs per element. These **memory-bound** operations are individually trivial in FLOPs yet can account for a large share of the run time of a transformer, because each launches its own kernel and makes its own trip through HBM.

```chart
{
  "type": "scatter",
  "caption": "Rooflines for A100 (312 TFLOP/s BF16, 2.039 TB/s) and H100 SXM (989 TFLOP/s, 3.35 TB/s), from vendor specifications, with four example operations on the A100: an FP16 elementwise add (I=0.17), an FP16 matrix-vector product (I=1), and 256x256 and 1024x1024 FP16 matmuls (I=85 and 341, from I = 2n/3b).",
  "height": 380,
  "data": { "datasets": [
    { "label": "A100 roofline", "showLine": true, "pointRadius": 0, "borderColor": "var(--c-blue)", "data": [{"x":0.1,"y":0.2039},{"x":153,"y":312},{"x":1000,"y":312}] },
    { "label": "H100 roofline", "showLine": true, "pointRadius": 0, "borderColor": "var(--c-purple)", "data": [{"x":0.1,"y":0.335},{"x":295.2,"y":989},{"x":1000,"y":989}] },
    { "label": "elementwise add", "pointRadius": 7, "backgroundColor": "var(--c-red)", "data": [{"x":0.17,"y":0.35}] },
    { "label": "matrix-vector (decode)", "pointRadius": 7, "backgroundColor": "var(--c-amber)", "data": [{"x":1,"y":2.04}] },
    { "label": "matmul 256", "pointRadius": 7, "backgroundColor": "var(--c-teal)", "data": [{"x":85.3,"y":174}] },
    { "label": "matmul 1024", "pointRadius": 7, "backgroundColor": "var(--c-green)", "data": [{"x":341,"y":312}] }
  ] },
  "options": { "scales": {
    "x": {"type": "logarithmic", "title": {"display": true, "text": "arithmetic intensity (FLOP/byte)"}},
    "y": {"type": "logarithmic", "title": {"display": true, "text": "attainable TFLOP/s"}} } }
}
```

Use the explorer below to move an operation around the roofline. Start with the decode-like setting ($M=8$) and slowly raise $M$ while watching intensity; then change the bytes per element.

{{demo: week16-roofline-explorer | Roofline explorer: where does this matrix multiply sit? | 480}}

### Fusion and the "memory wall"

If memory traffic is the problem, the cure is to do more per trip. **Operator fusion** combines consecutive memory-bound operations into one kernel: instead of writing the output of `x + bias` to HBM and reading it back for the ReLU, a fused kernel keeps the value in registers and writes only the final result. A chain of $k$ elementwise operations done separately costs about $k$ round trips; fused it costs one. Fusion is what compilers (`torch.compile`, XLA, TVM) do automatically, and it is the same principle that makes FlashAttention fast. Remember the pattern: *matmuls are compute-bound if large enough; everything else is memory-bound, and the job is to avoid trips to HBM.* This is also why the same mathematical operation can differ 10-fold in speed depending on implementation.

:::note Counting FLOPs is not counting time
FLOP counts (for example "6ND FLOPs to train a transformer with $N$ parameters on $D$ tokens", [Week 10](week-10.html)) tell you the *compute-bound* time. The real time also includes memory-bound operators, kernel-launch overhead, communication and data loading. Model FLOPs utilization (**MFU**, the ratio of achieved model FLOP/s to the hardware peak) is the standard summary. Well-tuned large-scale LLM training reaches 40 to 60% MFU. A naive PyTorch loop on a small model often reaches under 10%.
:::

## Profiling and bottleneck analysis {#profiling}

"Premature optimization is the root of all evil" applies doubly to deep learning, where intuition about performance is usually wrong. Measure first. The workflow is: (1) get an end-to-end number (samples per second, step time, GPU utilization); (2) find where the time goes; (3) classify the bottleneck; (4) fix the biggest one; (5) re-measure.

### Measuring correctly

GPU kernels run **asynchronously**: when Python calls `y = model(x)`, PyTorch enqueues kernels and returns immediately. If you time with `time.time()` you measure how long it took to *enqueue*. You must call `torch.cuda.synchronize()` before reading the clock (or use CUDA events). You should also **warm up** (the first iterations include CUDA context creation, cuDNN autotuning and memory allocation), run many iterations, and report a median rather than a single value.

```python:benchmark.py
import time
import torch

def bench(fn, warmup=10, iters=50):
    """Median wall-clock time of fn() in milliseconds, with correct GPU synchronization."""
    use_cuda = torch.cuda.is_available()
    for _ in range(warmup):
        fn()
    if use_cuda:
        torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        if use_cuda:
            torch.cuda.synchronize()          # wait until all queued kernels finish
        times.append((time.perf_counter() - t0) * 1e3)
    times.sort()
    return times[len(times) // 2]

if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for n in (256, 1024, 4096):
        a = torch.randn(n, n, device=dev, dtype=torch.float16 if dev == "cuda" else torch.float32)
        ms = bench(lambda: a @ a)
        tflops = 2 * n ** 3 / (ms * 1e-3) / 1e12
        print(f"n={n:5d}  {ms:8.3f} ms  {tflops:7.1f} TFLOP/s")
```

Run this on a GPU and you will see achieved TFLOP/s rise steeply with $n$ and saturate, exactly the roofline's prediction. (On an A100 the $256$ case typically reaches only a fraction of the compute roof because it is both memory-bound and too small to occupy 108 SMs.)

### The PyTorch profiler

`torch.profiler` records every CPU-side operator and CUDA kernel with timestamps. The usual recipe wraps a few training steps with a schedule that skips start-up noise:

```python:profile_loop.py
import torch, torch.nn as nn
from torch.profiler import profile, record_function, schedule, ProfilerActivity, tensorboard_trace_handler

dev = "cuda" if torch.cuda.is_available() else "cpu"
model = nn.Sequential(nn.Linear(1024, 4096), nn.GELU(), nn.Linear(4096, 1024)).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
x = torch.randn(256, 1024, device=dev)
y = torch.randn(256, 1024, device=dev)

activities = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if dev == "cuda" else [])
sched = schedule(wait=1, warmup=2, active=3, repeat=1)   # skip 1 step, warm up 2, record 3

with profile(activities=activities, schedule=sched, record_shapes=True,
             profile_memory=True, on_trace_ready=tensorboard_trace_handler("./prof")) as prof:
    for step in range(7):
        with record_function("forward"):
            loss = ((model(x) - y) ** 2).mean()
        with record_function("backward"):
            opt.zero_grad(set_to_none=True)
            loss.backward()
        with record_function("optimizer"):
            opt.step()
        prof.step()                                       # tell the profiler a step ended

sort_key = "cuda_time_total" if dev == "cuda" else "cpu_time_total"
print(prof.key_averages().table(sort_by=sort_key, row_limit=8))
```

Open the trace with TensorBoard (`pip install torch_tb_profiler`, `tensorboard --logdir prof`) or load the generated JSON in `chrome://tracing` / Perfetto. In the timeline, look at the GPU stream row. **Gaps** between kernels mean the GPU is idle waiting for the CPU. In the operator table, sort by total CUDA time and check the top five rows. (Recent PyTorch versions are renaming `cuda_time_total` to `device_time_total`; if the key is rejected, pick the nearest column name in the printed table header.)

### Diagnosing the bottleneck

A GPU job is limited by one of five things. The symptoms are distinct:

| Bottleneck | Symptom in `nvidia-smi` / trace | Typical fix |
| --- | --- | --- |
| **Input pipeline** (data loading, augmentation, host-to-device copy) | GPU utilization oscillates and falls to 0% between steps; the `DataLoader` wait appears at the start of each step | more `num_workers`, `pin_memory=True`, `persistent_workers=True`, prefetch, cheaper augmentation, faster storage format, do augmentations on the GPU |
| **CPU / launch overhead** | many tiny kernels of a few microseconds each, with gaps; common for small models, small batches and Python-heavy code | larger batch, fuse with `torch.compile`, CUDA graphs, remove Python loops and `.item()` calls |
| **Memory-bound kernels** | elementwise ops, normalizations, softmax, and attention dominating GPU time; low achieved FLOP/s | fusion (`torch.compile`), FlashAttention, lower precision |
| **Compute-bound** | large GEMMs/convolutions dominate and run near peak | mixed precision/tensor cores, smaller model, better algorithm |
| **Communication** (multi-GPU) | NCCL all-reduce kernels not overlapped with compute; ranks wait for each other | overlap (DDP does this), larger buckets, gradient accumulation, faster interconnect, sharding strategy |

Two cheap diagnostics are worth memorizing. First, *run the model on a single cached batch in a loop*: if throughput jumps, the data pipeline was the bottleneck. Second, *double the batch size*: if time per step barely changes, you were launch-bound (the GPU had spare capacity); if it doubles, you were compute- or bandwidth-bound.

:::tip Always check utilization before optimizing the model
The most common finding by far, in student projects, is not that the model is slow but that the **GPU is idle** because the loop does CPU-side work, synchronizes with `loss.item()` or `print` every step, or loads images with a single worker. Fix this first. Mixed precision on a loop that spends 60% of its time waiting for data gives you almost nothing.
:::

## Mixed-precision training {#mixed-precision}

### Number formats

Floating-point numbers store a sign, an exponent (which sets the *range*) and a mantissa (which sets the *precision*). The formats you will meet:

| Format | Bits | Exponent | Mantissa | Max finite | Smallest normal | Unit roundoff $u$ |
| --- | --- | --- | --- | --- | --- | --- |
| FP32 | 32 | 8 | 23 | $3.4\times10^{38}$ | $1.2\times10^{-38}$ | $6\times10^{-8}$ |
| TF32 (tensor-core math, stored as FP32) | 19 used | 8 | 10 | $3.4\times10^{38}$ | $1.2\times10^{-38}$ | $5\times10^{-4}$ |
| FP16 | 16 | 5 | 10 | 65,504 | $6.1\times10^{-5}$ (subnormals down to $6\times10^{-8}$) | $4.9\times10^{-4}$ |
| BF16 | 16 | 8 | 7 | $3.4\times10^{38}$ | $1.2\times10^{-38}$ | $3.9\times10^{-3}$ |

Unit roundoff $u=2^{-(p+1)}$ is the worst-case relative rounding error of one operation, where $p$ is the number of mantissa bits. FP16 has three more bits of precision than BF16 (about three decimal digits versus two) but only a 5-bit exponent, so its dynamic range is narrow: values above 65,504 overflow to infinity and values below roughly $6\times10^{-8}$ flush to zero. BF16 trades precision for range: it can represent anything FP32 can, in coarser steps. **TF32** is an A100 feature: matmuls take FP32 inputs, round them to 10 mantissa bits internally and accumulate in FP32; you enable it with `torch.backends.cuda.matmul.allow_tf32 = True` (or `torch.set_float32_matmul_precision("high")`) and need no other code change. FP8 formats (E4M3 and E5M2) on H100-class hardware extend the same story one step further.

### The recipe of Micicevicius et al.

Simply casting a network to FP16 fails for two reasons. The **update problem**: weights are updated by small increments $\Delta w=\eta\,g$, and with a 10-bit mantissa an increment smaller than about $w\times 2^{-11}$ is rounded away entirely. A weight of 1.0 cannot change by less than 0.0005. The **range problem**: gradients are often tiny (activation gradients of $10^{-8}$ are common), below FP16's representable range, so they underflow to zero and learning silently stalls.

The recipe in *Mixed Precision Training* (Micicevicius et al., ICLR 2018) has three parts:

1. **FP32 master weights.** Keep an FP32 copy of the parameters. The forward and backward passes use an FP16 copy (so activations and activation gradients are 16-bit and matmuls use tensor cores), but the optimizer updates the FP32 master copy and then re-casts it to FP16. Updates accumulate in a format that can represent them.
2. **Loss scaling.** Multiply the loss by a large constant $S$ (for example $2^{16}$) before backward. By the chain rule every gradient is multiplied by $S$, which shifts the gradient distribution up into FP16's representable range. Before the optimizer step, divide the gradients by $S$ (in FP32) to recover the true values.
3. **FP32 accumulation for sensitive operations.** Matmul and convolution accumulate partial sums in FP32 even when the inputs are FP16; reductions such as softmax, layer norm statistics, losses and batch-norm variances are done in FP32.

**Dynamic loss scaling** automates the choice of $S$. Start high; if any gradient contains `inf` or `NaN` (overflow), skip that optimizer step and halve $S$; if many consecutive steps (say 2000) are clean, double $S$. This is the algorithm in `torch.amp.GradScaler`.

### BF16 needs no loss scaling

With BF16's FP32-sized exponent, gradients that are representable in FP32 are essentially representable in BF16, so underflow is not an issue and the scaler is unnecessary. The price is precision: a 7-bit mantissa has $u\approx 0.4\%$, so you rely even more on FP32 master weights and FP32 accumulation to keep training stable. In practice BF16 mixed precision is simpler, more robust and the standard on hardware that supports it (Ampere and newer, TPUs); FP16 remains relevant on older GPUs (V100, T4) and for inference.

### Mixed precision in PyTorch

`torch.autocast` is a context manager that chooses the precision *per operator*: matmuls, convolutions and linear layers run in 16-bit; operations that are numerically sensitive (softmax, log-softmax, losses, norms, reductions, `exp`, `pow`) run in FP32. Parameters stay in FP32 (they *are* the master weights), and autocast casts copies on the fly.

```python:amp_train.py
import torch, torch.nn as nn, torch.nn.functional as F

dev = "cuda"
model = nn.Sequential(nn.Linear(784, 2048), nn.ReLU(), nn.Linear(2048, 10)).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

# FP16 needs loss scaling; BF16 does not. Pick based on hardware support.
use_bf16 = torch.cuda.is_bf16_supported()
amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
scaler = torch.amp.GradScaler("cuda", enabled=not use_bf16)   # a no-op when disabled

x = torch.randn(256, 784, device=dev)
y = torch.randint(0, 10, (256,), device=dev)

for step in range(100):
    opt.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=amp_dtype):
        logits = model(x)                       # linear layers run in 16-bit tensor-core matmuls
        loss = F.cross_entropy(logits, y)       # cross_entropy is autocast to float32
    scaler.scale(loss).backward()               # backward of (loss * scale)
    scaler.unscale_(opt)                        # gradients back to true scale, in place
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # clip AFTER unscale
    scaler.step(opt)                            # skips the step if any grad is inf/NaN
    scaler.update()                             # adjusts the scale factor
    if step % 20 == 0:
        print(step, f"loss={loss.item():.4f}", f"scale={scaler.get_scale():.0f}",
              "logits dtype:", logits.dtype, "param dtype:", next(model.parameters()).dtype)
```

Note the outputs of the last line: `logits` are 16-bit (the linear layer's output dtype) while the parameters remain `torch.float32`. Two details commonly go wrong. Gradient clipping must happen after `scaler.unscale_`, otherwise you clip scaled gradients against an unscaled threshold. And do not wrap the backward pass in `autocast`; only the forward and the loss.

:::warning Silent numerical failure
FP16 failures are rarely loud. Symptoms: the loss plateaus early while gradients are exactly zero in many layers (underflow); the loss suddenly becomes `nan` (overflow in activations or an unscaled gradient); validation accuracy is worse than FP32 by a few points (a precision-sensitive op ran in 16-bit). First step in debugging: run a few hundred steps in FP32 and compare loss curves. Second: use `torch.autograd.set_detect_anomaly(True)` for a short run to locate the first `inf`. Third: look at `scaler.get_scale()`; if it collapses toward 1 or keeps halving, gradients are overflowing every step.
:::

## Gradient accumulation {#accumulation}

Memory limits the micro-batch size, but optimization often wants a larger batch (large-batch training stabilizes gradient estimates, and some recipes such as LLM pretraining prescribe hundreds of thousands of tokens per step). **Gradient accumulation** simulates a large batch by summing gradients over several micro-batches before one optimizer step.

The justification is the linearity of the gradient. For a loss that is a mean over $B=A\cdot b$ examples, split into $A$ micro-batches of size $b$:

$$
\nabla_\theta \frac1B\sum_{i=1}^{B}\ell_i \;=\; \frac1A\sum_{a=1}^{A}\Bigl(\frac1b\sum_{i\in \text{micro-batch }a}\nabla_\theta\ell_i\Bigr).
$$

PyTorch's `backward()` *adds* into `.grad`, so calling it $A$ times and dividing each loss by $A$ yields exactly the large-batch gradient (up to floating-point reordering). The weights are not changed in between, so the result is mathematically identical to one big batch for models whose forward pass is independent across examples.

```python:accumulate.py
import torch, torch.nn as nn, torch.nn.functional as F

torch.manual_seed(0)
model = nn.Linear(20, 3)
opt = torch.optim.SGD(model.parameters(), lr=0.1)
X, y = torch.randn(64, 20), torch.randint(0, 3, (64,))

# (a) one big batch of 64
opt.zero_grad()
F.cross_entropy(model(X), y).backward()
g_big = model.weight.grad.clone()

# (b) 4 micro-batches of 16, accumulated
opt.zero_grad()
A = 4
for xb, yb in zip(X.chunk(A), y.chunk(A)):
    (F.cross_entropy(model(xb), yb) / A).backward()      # divide so the sum is a mean
g_acc = model.weight.grad.clone()
print("max abs difference:", (g_big - g_acc).abs().max().item())   # about 1e-8
```

Three caveats. **(1) Batch normalization** computes statistics per micro-batch, so accumulation changes the model's behavior (use GroupNorm/LayerNorm, or accept the difference). **(2) Per-example weighting**: if the loss is a *sum over tokens* and micro-batches contain different numbers of non-padding tokens, dividing by $A$ is wrong; normalize by the total token count of the full batch. **(3) Distributed training**: with DDP every `backward()` triggers an all-reduce, which is wasteful for the first $A-1$ micro-batches; wrap them in `with model.no_sync():` and synchronize only on the last. The learning rate should be tuned for the *effective* batch size $A\cdot b\cdot N_{\text{GPUs}}$; the linear scaling rule (Goyal et al., 2017) with a warm-up is the usual starting point ([Week 3](week-03.html)).

## Gradient checkpointing {#checkpointing}

### Where training memory goes

Backpropagation needs the intermediate **activations** of the forward pass, so they are kept until the backward pass reaches them. For a network of $n$ layers with activations of size $a$ each, activation memory is $n\cdot a$, which grows with batch size, sequence length and depth. Parameters, gradients and optimizer states do not depend on batch size; activations do, and for transformers with long sequences they dominate. **Gradient checkpointing** (also called activation recomputation or rematerialization; Chen, Xu, Zhang and Guestrin, 2016) stores only a subset of the activations and recomputes the others during the backward pass.

{{figure: week16-checkpoint | Gradient checkpointing with $n=12$ layers split into $k=3$ segments. Only the segment boundaries (green) are kept after the forward pass. During backward, each segment is re-run once to regenerate its internal activations (faded), which are used and then freed.}}

### Deriving the square-root trade-off

:::math Memory and compute of checkpointing a chain
Consider a chain of $n$ layers, each with the same activation size (one unit) and the same forward cost $c$. Split it into $k$ equal segments of $n/k$ layers. In the forward pass we keep only the $k$ segment inputs (the checkpoints) and discard everything else. During backward, we process segments from last to first. For each segment, we (1) re-run its forward pass from the checkpoint, storing its $n/k$ internal activations, (2) backpropagate through it, (3) free those activations.

*Memory.* At peak we hold all $k$ checkpoints plus the activations of one segment: 
$$
M(k) \;=\; k + \frac{n}{k}.
$$
*Optimum.* Differentiating, $M'(k)=1-n/k^2=0$ gives $k^\star=\sqrt n$ and $M(k^\star)=2\sqrt n$. For $n=64$ layers that is 16 units instead of 64. For $n=1000$, about 63 instead of 1000.

*Compute.* Every layer's forward pass is executed twice (once originally and once during recomputation), except layers in the last segment whose activations could be kept. The extra cost is at most one additional forward pass. Since a backward pass costs about twice a forward pass, one training step costs about $c n(1+2)=3cn$ without checkpointing and $4cn$ with it: roughly **33% more time** for a memory reduction from $O(n)$ to $O(\sqrt n)$.
:::

```chart
{
  "type": "line",
  "caption": "Peak stored activations (in layer units) for a 64-layer chain as a function of the number of checkpoint segments k, M(k)=k+64/k. The minimum 2*sqrt(64)=16 is at k=8. Computed from the formula; the baseline without checkpointing stores 64.",
  "height": 320,
  "data": { "labels": [1,2,3,4,6,8,12,16,24,32,64],
    "datasets": [
      { "label": "checkpointed: k + n/k", "data": [65,34,24.3,20,16.7,16,17.3,20,26.7,34,65], "borderColor": "var(--c-accent)", "tension": 0.2 },
      { "label": "no checkpointing (n)", "data": [64,64,64,64,64,64,64,64,64,64,64], "borderColor": "var(--c-muted)", "borderDash": [6,4], "pointRadius": 0 }
    ] },
  "options": { "scales": { "x": {"title": {"display": true, "text": "number of segments k"}}, "y": {"title": {"display": true, "text": "peak activation memory (layer units)"}} } }
}
```

Chen et al. also gave a recursive variant that reaches $O(\log n)$ memory at $O(n\log n)$ extra compute. In practice the $\sqrt n$ rule is used loosely: you checkpoint every transformer block (n checkpoints of one residual-stream tensor each) and recompute the block's internal activations (attention scores, MLP hidden states), which are by far the largest.

```python:checkpoint_demo.py
import torch, torch.nn as nn
from torch.utils.checkpoint import checkpoint, checkpoint_sequential

class Block(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
    def forward(self, x):
        return x + self.net(x)

class Net(nn.Module):
    def __init__(self, d=1024, n=24, ckpt=True):
        super().__init__()
        self.blocks = nn.ModuleList(Block(d) for _ in range(n))
        self.ckpt = ckpt
    def forward(self, x):
        for blk in self.blocks:
            # use_reentrant=False is the recommended implementation (supports kwargs, nested autograd)
            x = checkpoint(blk, x, use_reentrant=False) if (self.ckpt and self.training) else blk(x)
        return x

if torch.cuda.is_available():
    x = torch.randn(32, 512, 1024, device="cuda", requires_grad=True)
    for ckpt in (False, True):
        torch.cuda.reset_peak_memory_stats()
        net = Net(ckpt=ckpt).cuda().train()
        net(x).sum().backward()
        print(f"checkpointing={ckpt}: peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
        del net
# Alternative for nn.Sequential: checkpoint_sequential(seq, segments=int(len(seq) ** 0.5), input=x, use_reentrant=False)
```

On a transformer, selective recomputation (recompute only the cheap-but-large attention internals, store the expensive matmul outputs) recovers most of the memory saving for a smaller compute penalty (Korthikanti et al., 2022). Two cautions: dropout and other random operations must be replayed with the same random state during recomputation (PyTorch's checkpoint saves and restores RNG state by default), and layers with side effects such as batch-norm running statistics are updated twice.

:::example What does a transformer activation actually cost?
For one transformer layer with sequence length $s$, micro-batch $b$, hidden size $h$ and $a$ heads, and 16-bit activations, Korthikanti et al. (2022) count about $s\,b\,h\,(34 + 5as/h)$ bytes. The second term is the $s\times s$ attention matrices, which grow *quadratically* in sequence length. For $h=4096$, $a=32$, $s=2048$, $b=4$: $34+5\cdot32\cdot2048/4096=114$, so about $2048\cdot4\cdot4096\cdot114\approx3.8$ GB per layer, and 122 GB for 32 layers. That is why long-context training needs checkpointing and FlashAttention (which removes the quadratic term). Try it in the calculator below, where you can also see the model states.
:::

{{demo: week16-memory-calculator | Training memory calculator: parameters, gradients, optimizer states, activations | 520}}

## Distributed training {#distributed}

When a model or the desired throughput exceeds one device, we spread the work. There are four basic axes of parallelism, and large runs combine them.

| Strategy | What is split | What is replicated | Communication | Main use |
| --- | --- | --- | --- | --- |
| Data parallel (DP/DDP) | the batch | the whole model | all-reduce of gradients, once per step | scale throughput when the model fits one GPU |
| Sharded data parallel (ZeRO, FSDP) | the batch, plus optimizer states, gradients, parameters | nothing (at stage 3) | reduce-scatter and all-gather | model too big to replicate, no model code changes |
| Tensor (intra-layer) parallel | each weight matrix | activations (partly) | all-reduce inside every layer | very large layers, needs fast NVLink |
| Pipeline parallel | groups of layers (stages) | nothing | point-to-point activations between stages | very deep models across nodes |

### Data parallelism and all-reduce

In **data parallelism** every one of $N$ workers holds a full copy of the model and processes a different slice of the batch. Each computes a local gradient $\boldsymbol g_r$; the workers then compute the average $\bar{\boldsymbol g}=\frac1N\sum_r\boldsymbol g_r$ (an **all-reduce**) and each applies the same optimizer update. Because all workers start from the same weights, use the same average gradient, and run deterministic optimizers, the replicas stay identical forever. The result equals single-device training on the combined batch, which makes data parallelism the safest way to scale.

{{figure: week16-ring-allreduce | Left: one step of data-parallel training. Right: the ring all-reduce topology. The gradient is split into $N$ chunks; each GPU sends one chunk to its neighbour in $N-1$ reduce-scatter steps, then $N-1$ all-gather steps.}}

:::math Cost of ring all-reduce
Let the gradient have $S$ bytes and let each GPU have a link of bandwidth $\beta$ to its ring neighbours. Split the gradient into $N$ chunks of $S/N$ bytes. *Reduce-scatter:* in each of $N-1$ steps every GPU sends one chunk to its right neighbour and adds the chunk received from the left into its own copy. After $N-1$ steps each GPU holds one chunk that is the complete sum over all GPUs. *All-gather:* in $N-1$ further steps the finished chunks circulate around the ring until every GPU has all of them. Each phase sends $(N-1)\cdot S/N$ bytes per GPU, so

$$
\text{bytes sent per GPU}=\frac{2(N-1)}{N}S,\qquad t_{\text{allreduce}}\approx\frac{2(N-1)}{N}\frac{S}{\beta}\;\xrightarrow{N\to\infty}\;\frac{2S}{\beta}.
$$

The remarkable property is that the per-GPU traffic is **nearly independent of $N$**: adding GPUs does not make the all-reduce more expensive per GPU (ignoring latency terms and link contention). A naive scheme in which every worker sends its gradient to a parameter server would need $O(N)$ traffic through one link. As an example, a 1.3B-parameter model has $S=2.6$ GB in BF16; at 300 GB/s effective NVLink the all-reduce takes about $2\times2.6/300\approx 17$ ms, while across a 25 GB/s network it takes about 0.2 s, comparable with a whole step.
:::

Ring all-reduce for deep learning was popularized by Baidu (2017); it builds on bandwidth-optimal collective algorithms (Patarasuk and Yuan, 2009). Tree-based and hierarchical variants reduce latency across many nodes. NCCL implements all of them.

### DistributedDataParallel (DDP) in PyTorch

`torch.nn.parallel.DistributedDataParallel` implements data parallelism with one process per GPU. It groups parameters into **buckets** (default 25 MB) and launches the all-reduce of a bucket as soon as all its gradients are ready, **overlapping communication with the backward pass**. This is why DDP is far faster than the older single-process `DataParallel`, which should not be used.

```python:ddp_train.py
# Launch with:  torchrun --nproc_per_node=4 ddp_train.py
import os, torch, torch.nn as nn, torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler

def main():
    dist.init_process_group(backend="nccl")                 # torchrun sets RANK, WORLD_SIZE, MASTER_ADDR/PORT
    rank, world = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    torch.manual_seed(0)                                    # identical init on every rank (DDP also broadcasts rank 0)

    model = nn.Sequential(nn.Linear(32, 128), nn.ReLU(), nn.Linear(128, 1)).cuda(local_rank)
    model = DDP(model, device_ids=[local_rank])
    opt = torch.optim.SGD(model.parameters(), lr=0.05)

    X = torch.randn(4096, 32); y = X.sum(1, keepdim=True)
    ds = TensorDataset(X, y)
    sampler = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True)   # disjoint shard per rank
    loader = DataLoader(ds, batch_size=64, sampler=sampler, num_workers=2, pin_memory=True)

    accum = 2
    for epoch in range(3):
        sampler.set_epoch(epoch)                            # reshuffle differently each epoch (easy to forget)
        for i, (xb, yb) in enumerate(loader):
            xb, yb = xb.cuda(local_rank, non_blocking=True), yb.cuda(local_rank, non_blocking=True)
            last = (i + 1) % accum == 0
            ctx = model.no_sync() if not last else torch.enable_grad()   # skip all-reduce on non-final micro-batches
            with ctx:
                loss = nn.functional.mse_loss(model(xb), yb) / accum
                loss.backward()
            if last:
                opt.step(); opt.zero_grad(set_to_none=True)
        if rank == 0:                                       # log / save only on one rank
            print(f"epoch {epoch} loss {loss.item() * accum:.4f}")
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
```

The effective batch size is $64\times N\times\text{accum}$. Common DDP mistakes are forgetting `set_epoch`, saving checkpoints from all ranks, using `model.state_dict()` keys with the `module.` prefix when reloading into an unwrapped model, and logging metrics from one rank as if they were global averages (use `dist.all_reduce` to average them).

:::note Large-batch scaling
Data parallelism with $N$ GPUs multiplies the global batch size. Beyond a "critical batch size" extra samples per step give diminishing returns in steps saved, so scaling the batch without limit wastes compute; Goyal et al. (2017) trained ResNet-50 on ImageNet with batch 8192 using linear LR scaling and warm-up, and LAMB/LARS-style optimizers extend this further. The theory is outside our scope, but the practical rule is to tune the learning rate and warm-up when $N$ changes.
:::

### Tensor (model) parallelism

When a single layer is too big, or when you want to cut per-GPU activation memory, split the matrices. **Megatron-LM** (Shoeybi et al., 2019) shows how to do this for a transformer MLP with a *single* all-reduce per block. The MLP computes $Y=\text{GeLU}(XA)B$ with $A\in\R^{d\times 4d}$, $B\in\R^{4d\times d}$.

- Split $A$ **by columns** into $[A_1, A_2]$ (one per GPU). Each GPU computes $\text{GeLU}(XA_i)$ for its slice of the hidden units independently. Because GeLU is elementwise, no communication is needed: the nonlinearity commutes with a column split (it would not commute with a row split, since GeLU of a sum is not the sum of GeLUs).
- Split $B$ **by rows** into $B_1,B_2$, matching the column slices. GPU $i$ computes $Z_i=\text{GeLU}(XA_i)B_i\in\R^{b\times d}$, a *partial sum* of the output: $Y=Z_1+Z_2$.
- One **all-reduce** sums the partial outputs.

Self-attention is split the same way: each GPU holds a subset of heads (the $Q,K,V$ projections are column-split by head, the output projection row-split), again with one all-reduce at the end. A transformer layer therefore needs two all-reduces in the forward and two in the backward pass, each of size (batch × sequence × $d$). That traffic is on the critical path (it cannot overlap easily with compute), so tensor parallelism is only efficient **within a node** over NVLink, typically with degree 2 to 8.

### Pipeline parallelism and the bubble

**Pipeline parallelism** places consecutive groups of layers on different devices (stages). Only activations (forward) and their gradients (backward) cross device boundaries, which is cheap enough to work across nodes. The problem is that stage 2 cannot start until stage 1 has finished, so with a single batch only one device works at a time. **GPipe** (Huang et al., 2019) splits each batch into $m$ **micro-batches** and pipelines them: while stage 2 processes micro-batch 1, stage 1 already works on micro-batch 2. The first and last devices still idle at the start and end of each step; this idle time is the **bubble**.

{{figure: week16-pipeline | GPipe schedule with $p=4$ stages and $m=4$ micro-batches. The staircase of idle slots at the beginning and end is the pipeline bubble.}}

:::math Bubble fraction
Let each micro-batch take one time unit for the forward pass and (say) the same for backward through one stage, with $p$ stages and $m$ micro-batches. The ideal time (all devices always busy) is $m$ units forward plus $m$ backward. In GPipe the last stage cannot start the first micro-batch until $p-1$ units have elapsed, so the forward phase takes $m+p-1$ units; the backward phase likewise takes $m+p-1$. The idle fraction is

$$
\text{bubble}=\frac{(m+p-1)-m}{m+p-1}=\frac{p-1}{m+p-1}.
$$

For $p=8$ stages, $m=8$ micro-batches wastes $7/15=47\%$; $m=64$ wastes $7/71=10\%$. To make the bubble small you need $m\gg p$, but each micro-batch must also be large enough to keep a GPU busy, so the global batch size must be large. The bubble does not depend on how the model is split as long as stages are balanced.
:::

A second cost is memory: GPipe must hold activations of all $m$ micro-batches until their backward passes start. The **1F1B** schedule (PipeDream-Flush; Narayanan et al., 2019 and 2021) alternates one forward and one backward pass in steady state, so a stage never holds more than $p$ micro-batches of activations, while the bubble stays at $(p-1)/(m+p-1)$. Interleaved schedules assign several small non-contiguous chunks of layers to each device, dividing the bubble by the number of chunks at the price of more communication. Try the simulator to see all of this.

{{demo: week16-pipeline-bubble | Pipeline schedules: bubble and stored activations | 440}}

```chart
{
  "type": "line",
  "caption": "Pipeline bubble fraction (p-1)/(m+p-1) versus the number of micro-batches m for p=4 and p=8 stages (computed from the formula).",
  "height": 320,
  "data": { "labels": [1,2,4,8,16,32,64],
    "datasets": [
      { "label": "p = 4", "data": [0.75,0.6,0.429,0.273,0.158,0.086,0.045], "borderColor": "var(--c-blue)", "tension": 0.2 },
      { "label": "p = 8", "data": [0.875,0.778,0.636,0.467,0.304,0.179,0.099], "borderColor": "var(--c-red)", "tension": 0.2 }
    ] },
  "options": { "scales": { "x": {"title": {"display": true, "text": "micro-batches m"}}, "y": {"title": {"display": true, "text": "idle fraction"}, "min": 0, "max": 1} } }
}
```

### 3D parallelism in practice

Training models with hundreds of billions of parameters (Megatron-Turing NLG, the BLOOM and Llama families) uses all three together: tensor parallelism within a node (fast NVLink), pipeline parallelism across nodes (cheap point-to-point traffic), and data parallelism across replicas of the whole pipeline, usually with sharded optimizer states. Narayanan et al. (2021) describe how to choose the degrees $t, p, d$ with $t\cdot p\cdot d=N_{\text{GPUs}}$ and report about 52% of peak throughput on 3072 A100 GPUs for a one-trillion-parameter model. Sequence/context parallelism (splitting the sequence dimension) and expert parallelism (Mixture-of-Experts, below) are additional axes.

## ZeRO and FSDP: sharding the redundancy {#zero}

### Memory accounting for mixed-precision Adam

In plain data parallelism *every* GPU stores the full model state, which is wasteful. Consider a model with $\Psi$ parameters trained with Adam and mixed precision. Per parameter you store:

- FP16/BF16 parameters: 2 bytes
- FP16/BF16 gradients: 2 bytes
- Optimizer states in FP32: a master copy of the parameters (4 bytes), Adam's first moment $m$ (4) and second moment $v$ (4): 12 bytes, called $K=12$ in the ZeRO paper.

Total $16\Psi$ bytes, before activations. For $\Psi=7.5$ B that is **120 GB**. The optimizer states alone are 75% of it. This is a lot of memory that is identical on every GPU.

**ZeRO** (Zero Redundancy Optimizer; Rajbhandari, Rasley, Ruwase and He, SC 2020) removes the redundancy by *partitioning* model states over the $N$ data-parallel ranks, in three cumulative stages:

- **Stage 1** ($P_{os}$): shard the optimizer states. Each rank owns $1/N$ of them and updates the corresponding $1/N$ of the parameters, then an all-gather distributes the updated parameters. Memory: $4\Psi+12\Psi/N$.
- **Stage 2** ($P_{os+g}$): also shard the gradients. Gradients are *reduce-scattered* so that each rank receives only the (averaged) shard it owns. Memory: $2\Psi+14\Psi/N$.
- **Stage 3** ($P_{os+g+p}$): also shard the parameters. Each rank stores only $1/N$ of the parameters. Before a layer runs, its full parameters are *all-gathered*; after use they are freed. Memory: $16\Psi/N$.

{{figure: week16-zero | Per-GPU model-state memory under DDP and ZeRO stages 1 to 3 for mixed-precision Adam ($K=12$). Each stage removes one more replicated component.}}

```chart
{
  "type": "bar",
  "caption": "Per-GPU model-state memory (GB) for a 7.5B-parameter model with mixed-precision Adam, from the formulas 16P, 4P+12P/N, 2P+14P/N and 16P/N (N=64 reproduces the table in the ZeRO paper: 120, 31.4, 16.6, 1.9 GB). Activations excluded.",
  "height": 340,
  "data": { "labels": ["DDP / baseline", "ZeRO-1", "ZeRO-2", "ZeRO-3 / FSDP"],
    "datasets": [
      { "label": "N = 8 GPUs", "data": [120, 41.3, 28.1, 15.0], "backgroundColor": "var(--c-blue)" },
      { "label": "N = 64 GPUs", "data": [120, 31.4, 16.6, 1.9], "backgroundColor": "var(--c-green)" }
    ] },
  "options": { "scales": { "y": {"title": {"display": true, "text": "GB per GPU"}} } }
}
```

### Communication cost

Baseline DDP all-reduces gradients: $2\Psi$ elements per step in ring terms (reduce-scatter $\Psi$ + all-gather $\Psi$). ZeRO-1 and ZeRO-2 use a reduce-scatter of gradients ($\Psi$) and an all-gather of updated parameters ($\Psi$): **the same $2\Psi$** as DDP, so memory savings are free in volume. ZeRO-3 additionally all-gathers parameters in the forward pass ($\Psi$) and again in the backward pass ($\Psi$, since they were freed), plus the gradient reduce-scatter ($\Psi$): **$3\Psi$, a 1.5x increase** over DDP. In exchange the model-state memory shrinks linearly with $N$. The overhead is hidden by prefetching the next layer's parameters while the current layer computes.

### FSDP in PyTorch

**FullyShardedDataParallel (FSDP)** is PyTorch's implementation of the ZeRO-3 idea (Zhao et al., "PyTorch FSDP", VLDB 2023). Its sharding unit is a wrapped module; you choose the wrapping granularity with an auto-wrap policy, typically one transformer block per unit so that only one block's full parameters are materialized at a time.

```python:fsdp_train.py
# torchrun --nproc_per_node=8 fsdp_train.py
import os, functools, torch, torch.nn as nn, torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import ModuleWrapPolicy

class Block(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
    def forward(self, x):
        return x + self.mlp(self.ln(x))

def main():
    dist.init_process_group("nccl")
    local = int(os.environ["LOCAL_RANK"]); torch.cuda.set_device(local)
    model = nn.Sequential(*[Block(2048) for _ in range(24)])          # about 1.2B parameters

    bf16 = MixedPrecision(param_dtype=torch.bfloat16,                 # compute/all-gather in BF16
                          reduce_dtype=torch.bfloat16,                # gradient reduce-scatter in BF16
                          buffer_dtype=torch.bfloat16)
    model = FSDP(model,
                 auto_wrap_policy=ModuleWrapPolicy({Block}),          # one FSDP unit per Block
                 sharding_strategy=ShardingStrategy.FULL_SHARD,       # ZeRO-3 (SHARD_GRAD_OP is ZeRO-2)
                 mixed_precision=bf16,
                 device_id=torch.cuda.current_device())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)              # create AFTER wrapping: optimizer states are sharded
    x = torch.randn(8, 512, 2048, device="cuda")
    for step in range(5):
        loss = model(x).float().pow(2).mean()
        loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
        if dist.get_rank() == 0:
            print(step, loss.item(), f"peak mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
```

Newer PyTorch releases add a per-parameter-sharding API, `torch.distributed.fsdp.fully_shard` ("FSDP2"), with the same ideas; DeepSpeed implements ZeRO with additional offloading of states to CPU memory or NVMe (ZeRO-Offload, ZeRO-Infinity). **Rule of thumb for choosing:** if the model plus optimizer fit on one GPU, use DDP (fastest). If they do not, use FSDP/ZeRO-3 across the data-parallel group. Add tensor and pipeline parallelism only when you cross many nodes or when even one layer is too large.

Run the calculator again with ZeRO stages and a fixed 8-GPU budget to find which stage first fits a 7B model on 40 GB or 80 GB cards.

## Efficient attention: FlashAttention {#flashattention}

### Why standard attention is slow

Recall single-head attention for a sequence of length $N$ with head dimension $d$ ([Week 9](week-09.html)): $S=QK^\top/\sqrt d\in\R^{N\times N}$, $P=\softmax(S)$ row-wise, $O=PV$. The textbook implementation runs three kernels. It computes $S$ and **writes it to HBM** ($N^2$ numbers), reads it back to compute softmax and **writes $P$** ($N^2$ more), and reads $P$ again for the product with $V$. At $N=8192$, one head's score matrix has 67 million entries, 134 MB in FP16, for a computation whose inputs $Q,K,V$ are only $3Nd=3\times8192\times64\approx1.6$M numbers. The matmuls have high intensity, but softmax and the surrounding data movement are memory-bound and the $N^2$ intermediate dominates both time and memory.

The key observation of **FlashAttention** (Dao, Fu, Ermon, Rudra and Ré, NeurIPS 2022) is that attention is *IO-bound*, not FLOP-bound, and that the exact same result can be computed **tile by tile in SRAM** without ever materializing $S$ or $P$ in HBM. The obstacle is softmax, which needs the whole row to normalize. The solution is the *online softmax*.

### Online softmax

:::math Online (streaming) softmax
For a row of scores $s_1,\dots,s_N$ the numerically stable softmax is $p_j=e^{s_j-m}/\ell$ with $m=\max_j s_j$ and $\ell=\sum_j e^{s_j-m}$. Naively this needs one pass for $m$, one for $\ell$, one to normalize. Milakov and Gimelshein (2018) observed that $m$ and $\ell$ can be maintained in a *single streaming pass*. Suppose after seeing the first $t$ scores we have $m_t=\max_{j\le t}s_j$ and $\ell_t=\sum_{j\le t}e^{s_j-m_t}$. A new score $s_{t+1}$ arrives. Set $m_{t+1}=\max(m_t,s_{t+1})$. The old sum was computed relative to $m_t$, so rescale it to the new reference:

$$
\ell_{t+1}=\ell_t\,e^{m_t-m_{t+1}}+e^{s_{t+1}-m_{t+1}}.
$$

Check: $\ell_t e^{m_t-m_{t+1}}=\sum_{j\le t}e^{s_j-m_t}e^{m_t-m_{t+1}}=\sum_{j\le t}e^{s_j-m_{t+1}}$, so $\ell_{t+1}=\sum_{j\le t+1}e^{s_j-m_{t+1}}$. The same rescaling works when scores arrive in blocks.

**Including the value-weighted sum.** What attention actually needs is $o=\sum_j p_j v_j$. Define the *unnormalized* accumulator $\tilde o_t=\sum_{j\le t}e^{s_j-m_t}v_j$. Exactly as for $\ell$,
$$
\tilde o_{t+1}=\tilde o_t\,e^{m_t-m_{t+1}}+e^{s_{t+1}-m_{t+1}}v_{t+1},
$$
and at the end $o=\tilde o_N/\ell_N$. So a single pass over keys and values, carrying only the running triple $(m,\ell,\tilde o)$ per query row, gives the exact attention output.
:::

### The tiled algorithm

FlashAttention applies this recursion to blocks. Partition $Q$ into row blocks $Q_i$ (of $B_r$ rows) and $K,V$ into blocks $K_j,V_j$ (of $B_c$ rows), with block sizes chosen so that a $Q_i$, a $K_j$, a $V_j$ and the $B_r\times B_c$ score tile all fit in SRAM. Then, for each query block $i$ (outer loop) and each key/value block $j$ (inner loop):

1. Load $Q_i,K_j,V_j$ from HBM into SRAM.
2. Compute the tile $S_{ij}=Q_iK_j^\top/\sqrt d$ in SRAM.
3. Update the running row maxima $m_i^{\text{new}}=\max(m_i,\text{rowmax}(S_{ij}))$.
4. Compute $\tilde P_{ij}=\exp(S_{ij}-m_i^{\text{new}})$ and $\ell_i\leftarrow e^{m_i-m_i^{\text{new}}}\ell_i+\text{rowsum}(\tilde P_{ij})$.
5. Update the accumulator $\tilde O_i\leftarrow e^{m_i-m_i^{\text{new}}}\tilde O_i+\tilde P_{ij}V_j$.

After the last $j$, write $O_i=\tilde O_i/\ell_i$ and the row logsumexp $L_i=m_i+\log\ell_i$ to HBM. The $N\times N$ matrices never leave SRAM.

{{figure: week16-flash-tiling | Standard attention writes and reads the $N\times N$ score and probability matrices in HBM. FlashAttention fuses the three steps and processes tiles in SRAM with the online-softmax recursion, writing only the output and the logsumexp.}}

```python:flash_attention_numpy.py
import numpy as np

def attention_reference(Q, K, V):
    S = Q @ K.T / np.sqrt(Q.shape[1])
    S = S - S.max(axis=1, keepdims=True)
    P = np.exp(S); P /= P.sum(axis=1, keepdims=True)
    return P @ V

def flash_attention(Q, K, V, Br=16, Bc=32):
    """Exact attention computed tile by tile with the online-softmax recursion (forward pass only)."""
    N, d = Q.shape
    O = np.zeros_like(Q)
    L = np.zeros(N)                                    # row logsumexp, saved for the backward pass
    for i in range(0, N, Br):                          # outer loop over query blocks
        Qi = Q[i:i + Br]
        m = np.full(Qi.shape[0], -np.inf)              # running row max
        l = np.zeros(Qi.shape[0])                      # running row sum of exp
        acc = np.zeros_like(Qi)                        # running unnormalized output
        for j in range(0, N, Bc):                      # inner loop over key/value blocks
            Sij = Qi @ K[j:j + Bc].T / np.sqrt(d)      # (Br, Bc) tile: lives in "SRAM"
            m_new = np.maximum(m, Sij.max(axis=1))
            P = np.exp(Sij - m_new[:, None])
            scale = np.exp(m - m_new)                  # rescale old statistics to the new max
            l = scale * l + P.sum(axis=1)
            acc = scale[:, None] * acc + P @ V[j:j + Bc]
            m = m_new
        O[i:i + Br] = acc / l[:, None]
        L[i:i + Br] = m + np.log(l)
    return O, L

rng = np.random.default_rng(0)
N, d = 200, 64                                         # N not divisible by the block sizes on purpose
Q, K, V = (rng.standard_normal((N, d)) for _ in range(3))
O_ref = attention_reference(Q, K, V)
O_fl, L = flash_attention(Q, K, V)
print("max abs error vs reference:", np.abs(O_ref - O_fl).max())     # about 1e-15
```

The result is **exact**, not an approximation. The causal mask is handled by skipping tiles that are entirely above the diagonal (which saves about half the work) and masking within diagonal tiles.

### Why it is faster, and what it costs

Let $M$ be the SRAM size. Standard attention does $\Theta(Nd+N^2)$ HBM accesses. FlashAttention does $\Theta(N^2d^2/M)$ accesses: each of the $N/B_r$ outer iterations streams all of $K,V$ once, and the block size scales as $B_r\sim M/d$, so the total is $\frac{N}{B_r}\cdot Nd=\frac{N^2d^2}{M}$. Since $d^2\ll M$ for typical values ($d=64$ to $128$, $M\approx 100$ KB), this is a large reduction. The FLOP count is unchanged (slightly higher, from the rescaling) but the kernel becomes compute-bound instead of memory-bound. The **memory** footprint drops from $O(N^2)$ to $O(N)$, enabling much longer contexts. Dao et al. reported 2 to 4 times wall-clock speed-ups on attention and 15% end-to-end speed-up on BERT-large training at that time.

For the backward pass, FlashAttention does not store $P$. It stores only $O$ and $L$ and **recomputes** $S$ and $P$ tile by tile from $Q,K,V$ and $L$ ($P_{ij}=\exp(S_{ij}-L_i)$). This is the same compute-for-memory trade as checkpointing, but it is a net speed-up because recomputation in SRAM is cheaper than reading $N^2$ numbers from HBM. FlashAttention-2 (Dao, 2023) improved work partitioning across thread blocks and warps, and FlashAttention-3 (Shah et al., 2024) exploits Hopper-specific asynchrony and FP8. You rarely write this yourself: `torch.nn.functional.scaled_dot_product_attention` dispatches to a fused FlashAttention kernel when the inputs allow it (CUDA, FP16/BF16, supported head sizes).

```python:sdpa.py
import torch, torch.nn.functional as F

B, H, N, d = 2, 8, 4096, 64
q, k, v = (torch.randn(B, H, N, d, device="cuda", dtype=torch.bfloat16) for _ in range(3))

# Fused, memory-efficient attention: never builds the (N x N) matrix.
out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

def naive(q, k, v):
    s = (q @ k.transpose(-2, -1)) / d ** 0.5
    mask = torch.triu(torch.ones(N, N, device=q.device, dtype=torch.bool), 1)
    return (s.masked_fill(mask, float("-inf")).softmax(-1)) @ v

print("max diff:", (out.float() - naive(q, k, v).float()).abs().max().item())
for name, fn in [("fused", lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True)), ("naive", lambda: naive(q, k, v))]:
    torch.cuda.reset_peak_memory_stats(); fn(); torch.cuda.synchronize()
    print(name, f"peak {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB")
```

## KV caching and speculative decoding {#kv-cache}

### Autoregressive decoding and the KV cache

A decoder-only language model generates one token at a time. Computing the attention output for token $t$ requires the keys and values of all tokens $1..t$. Recomputing them at every step would cost $O(t)$ projections per token, $O(T^2)$ overall. Since keys and values of earlier tokens never change (causal masking), we **cache** them. At each step we compute only the query, key and value of the newest token, append $k_t,v_t$ to the cache, and attend over the cache. Per-token compute becomes $O(t)$ for the attention itself and $O(1)$ for everything else.

The cache costs memory. For one token, each layer stores a key and a value for each KV head:

$$
\text{KV bytes per token}=2\cdot n_{\text{layers}}\cdot n_{\text{kv heads}}\cdot d_{\text{head}}\cdot b.
$$

For a Llama-2-7B-like configuration (32 layers, 32 heads of dimension 128, FP16, $b=2$) that is $2\cdot32\cdot32\cdot128\cdot2=524{,}288$ bytes, **0.5 MB per token**. A 4096-token context costs 2 GiB per sequence, and 32 concurrent requests cost 64 GiB, more than the 13 GB of FP16 weights. At long context and large batch the KV cache, not the weights, is the memory bottleneck, and reading it every step makes decoding memory-bound.

```chart
{
  "type": "line",
  "caption": "KV-cache size per sequence (GiB, FP16) versus context length for a 32-layer, 128-dim-head model, computed from 2*L*n_kv*d*2 bytes per token: 32 KV heads (multi-head, 0.5 MB/token) versus 8 KV heads (grouped-query attention, 0.125 MB/token).",
  "height": 320,
  "data": { "labels": [1024, 2048, 4096, 8192, 16384, 32768],
    "datasets": [
      { "label": "32 KV heads (MHA)", "data": [0.5, 1, 2, 4, 8, 16], "borderColor": "var(--c-red)" },
      { "label": "8 KV heads (GQA)", "data": [0.125, 0.25, 0.5, 1, 2, 4], "borderColor": "var(--c-green)" }
    ] },
  "options": { "scales": { "x": {"title": {"display": true, "text": "context length (tokens)"}}, "y": {"title": {"display": true, "text": "GiB per sequence"}} } }
}
```

The main remedies reduce the number of cached values or manage them better. **Multi-query attention** (Shazeer, 2019) shares one key/value head across all query heads; **grouped-query attention** (Ainslie et al., 2023) shares a KV head among a group of query heads, a middle ground adopted by many open models. **PagedAttention** (Kwon et al., SOSP 2023; the basis of the vLLM server) stores the cache in fixed-size blocks like virtual-memory pages, removing the fragmentation that comes from reserving a maximum-length buffer per request. Cache quantization (int8/int4 keys and values) and sliding-window attention reduce it further.

```python:kv_cache.py
import torch, torch.nn as nn, torch.nn.functional as F

class CachedSelfAttention(nn.Module):
    def __init__(self, d=256, h=4):
        super().__init__()
        self.h, self.dk = h, d // h
        self.qkv = nn.Linear(d, 3 * d); self.out = nn.Linear(d, d)

    def forward(self, x, cache=None):
        B, T, D = x.shape                                           # T = number of NEW tokens (1 when decoding)
        q, k, v = self.qkv(x).view(B, T, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)   # each (B, h, T, dk)
        if cache is not None:
            k = torch.cat([cache[0], k], dim=2)                     # (B, h, T_past + T, dk)
            v = torch.cat([cache[1], v], dim=2)
        # With a cache, new queries may attend to all past keys, so no causal mask is needed when T == 1.
        y = F.scaled_dot_product_attention(q, k, v, is_causal=(cache is None and T > 1))
        y = y.transpose(1, 2).reshape(B, T, D)
        return self.out(y), (k, v)

torch.manual_seed(0)
attn = CachedSelfAttention().eval()
x = torch.randn(1, 10, 256)
with torch.no_grad():
    full, _ = attn(x)                                               # process all 10 tokens at once
    cache, outs = None, []
    for t in range(10):                                             # token by token with the cache
        o, cache = attn(x[:, t:t + 1], cache); outs.append(o)
    inc = torch.cat(outs, dim=1)
print("cached == full:", torch.allclose(full, inc, atol=1e-5), "| cache shape:", tuple(cache[0].shape))
```

### Speculative decoding

Decoding is memory-bound: producing one token reads all the weights (13 GB for a 7B model in FP16) and uses almost none of the GPU's arithmetic ($I\approx1$ FLOP/byte, see above). Verifying $\gamma$ tokens at once costs nearly the same time as generating one, because the weights are read once either way. **Speculative decoding** (Leviathan, Kalman and Matias, 2022; Chen et al., 2023) exploits this. A small, fast **draft model** proposes $\gamma$ tokens autoregressively; the large **target model** scores all $\gamma$ proposals in a single forward pass; a rejection-sampling rule accepts a prefix of the proposals and, at the first rejection, resamples from a corrected distribution. The rule guarantees that the output distribution is **exactly** the target model's, so quality is unchanged.

If each proposed token is accepted independently with probability $\alpha$ (a simplification), the expected number of tokens produced per target forward pass is

$$
\mathbb E[\text{tokens}]=\frac{1-\alpha^{\gamma+1}}{1-\alpha},
$$

since the verification pass always yields at least one token (the correction or a bonus token) and each further token requires all earlier ones to be accepted. With $\alpha=0.8,\gamma=4$ this is $(1-0.8^5)/0.2=3.36$ tokens per target pass, a speed-up of up to about 3x if the draft cost is small. The speed-up is greatest when the draft agrees often with the target (similar training data, an easy domain such as code or boilerplate) and when decoding is memory-bound. Variants avoid a separate draft model by adding extra prediction heads to the target (Medusa), or by drafting with early layers of the same network. At large batch sizes the GPU is already compute-bound, and speculation helps less.

## Model compression {#compression}

Compression reduces the cost of a *trained* model: fewer parameters, fewer bits per parameter, or a cheaper architecture with similar behavior. The four classical families are pruning, quantization, distillation and low-rank factorization. They are largely orthogonal and often stacked (Han, Mao and Dally's *Deep Compression*, ICLR 2016, combined pruning, quantization and Huffman coding to shrink AlexNet 35x and VGG-16 49x without accuracy loss on ImageNet). A key practical lesson, which applies to everything below, is that **a reduction in parameter count or theoretical FLOPs is not a reduction in latency** unless the hardware and kernels can exploit it.

### Pruning

**Pruning** removes weights (or larger structures) that contribute little. The earliest work (LeCun, Denker and Solla's Optimal Brain Damage, 1989) estimated the loss increase from deleting each weight with a second-order Taylor expansion. The simplest modern criterion is **magnitude pruning**: remove the weights with the smallest $|w|$, on the assumption that small weights matter least. Han et al. (2015) showed that the loop *train, prune, fine-tune, repeat* ("iterative pruning") can remove 80 to 90% of the weights of many convolutional networks with little loss.

- **Unstructured pruning** zeroes individual weights anywhere. It reaches the highest sparsity for a given accuracy, but the result is an irregular sparse matrix, and dense GPU kernels gain nothing from it: a matrix with 90% zeros takes the same time as a dense one unless special sparse kernels are used (which usually pay off only above ~90% sparsity). NVIDIA's Ampere GPUs support a hardware-accelerated **2:4 structured sparsity** pattern (exactly two non-zeros in every four consecutive weights), which can give up to about 2x matmul speed-up at 50% sparsity.
- **Structured pruning** removes whole units: convolution filters, attention heads, neurons or even entire layers. The remaining network is a smaller *dense* network and runs faster on any hardware. It costs more accuracy per removed parameter, and you must re-wire the dimensions of adjacent layers. Importance scores include the filter's $\ell_1$ or $\ell_2$ norm, the BatchNorm scale $\gamma$ ("network slimming"; Liu et al., 2017) or a first-order saliency $|w\,\partial\mathcal L/\partial w|$.
- **Global versus layerwise pruning.** Comparing magnitudes across layers (global threshold) lets the algorithm decide how much each layer can lose; per-layer fixed ratios are simpler but wasteful. The first and last layers are usually more sensitive and are often left dense.

```python:prune_demo.py
import torch, torch.nn as nn
import torch.nn.utils.prune as prune

torch.manual_seed(0)
model = nn.Sequential(nn.Linear(256, 512), nn.ReLU(), nn.Linear(512, 10))

# Unstructured: zero the 70% smallest-|w| weights in each Linear, using a global threshold.
params = [(m, "weight") for m in model if isinstance(m, nn.Linear)]
prune.global_unstructured(params, pruning_method=prune.L1Unstructured, amount=0.7)

# Structured: remove 25% of the output neurons (rows) of the first layer, ranked by L2 norm.
prune.ln_structured(model[0], name="weight", amount=0.25, n=2, dim=0)

for i in (0, 2):
    w = model[i].weight                         # now = weight_orig * weight_mask (a reparametrization)
    print(f"layer {i}: sparsity {(w == 0).float().mean().item():.2%}")

# During fine-tuning the mask stays fixed. When done, make pruning permanent:
for m, name in params:
    prune.remove(m, name)                       # folds the mask into .weight, deletes weight_orig/weight_mask
print("params remain dense tensors:", model[0].weight.shape)   # zeros are still stored!
```

Note the last line: PyTorch's pruning utilities *mask* weights; they do not make the tensor smaller or faster. To get real savings with structured pruning you must physically slice the weight matrices (or use a toolkit such as torch-pruning), and for unstructured pruning you must convert to a sparse format or 2:4 layout.

:::note The lottery ticket hypothesis
Frankle and Carbin (ICLR 2019) found that dense networks contain small sub-networks ("winning tickets") which, when reset to their *original initialization* (or, for larger models, to an early-training checkpoint) and trained in isolation, can match the dense network's accuracy. This suggested that overparameterization helps *optimization* and that much of the final network is redundant. It remains an empirical observation about trained-from-scratch sparse networks, with subtleties at scale, but it motivated much research on pruning at initialization.
:::

### Quantization

**Quantization** stores weights (and possibly activations) as low-bit integers instead of 16- or 32-bit floats. It shrinks memory 2 to 8 times, reduces the bandwidth cost of every memory-bound operation (recall the decode roofline), and, when the hardware has integer tensor-core instructions, speeds up the arithmetic too.

{{figure: week16-quant-grid | Uniform symmetric 4-bit quantization maps each weight to one of 15 integer levels times a scale. The scale is set by the largest magnitude in the group of weights that shares it, so grouping granularity determines how much one outlier hurts.}}

:::math Uniform affine quantization and its error
Given a real range $[\alpha,\beta]$ to be represented with $b$ bits ($2^b$ levels), the **affine** (asymmetric) scheme uses a scale $s$ and an integer zero-point $z$:
$$
s=\frac{\beta-\alpha}{2^b-1},\quad z=\operatorname{round}\!\Bigl(-\frac{\alpha}{s}\Bigr),\quad
w_q=\operatorname{clip}\bigl(\operatorname{round}(w/s)+z,\;0,\;2^b-1\bigr),\quad \hat w=s\,(w_q-z).
$$
The **symmetric** scheme fixes $z=0$ and uses a range $[-\alpha,\alpha]$ with $\alpha=\max|w|$: $s=\alpha/(2^{b-1}-1)$, $w_q=\operatorname{clip}(\operatorname{round}(w/s),-(2^{b-1}-1),2^{b-1}-1)$. For int8 this gives levels $-127..127$, for int4, $-7..7$.

*Error.* Rounding to the nearest level gives an error $e=w-\hat w$ in $[-s/2,s/2]$. If the weights are spread widely over many levels, $e$ is approximately uniform and $\mathbb E[e^2]=s^2/12$. Taking a range $2\alpha$ split into $2^b$ levels, $s=2\alpha/2^b$ and the noise power is $\frac{\alpha^2}{3}\,2^{-2b}$. If the signal has power $\sigma_w^2$,
$$
\text{SQNR}=\frac{\sigma_w^2}{\alpha^2 2^{-2b}/3}\;\Rightarrow\;\text{SQNR (dB)}=10\log_{10}\frac{3\sigma_w^2}{\alpha^2}+6.02\,b.
$$
**Each extra bit buys about 6 dB (a factor 4 in noise power, 2 in RMS error).** The first term shows the price of range: if an outlier makes $\alpha\gg\sigma_w$, you lose precision for *every* weight that shares that scale.
:::

**Granularity** therefore matters a great deal. *Per-tensor* quantization uses one scale for the whole matrix; *per-channel* (per output row) uses one per row; *group-wise* uses one per group of 32 to 128 consecutive weights within a row (the standard for 4-bit LLM weights), at a cost of 16 bits of scale per group (0.125 to 0.5 extra bits per weight). Run the explorer below with an outlier and compare:

{{demo: week16-quantization-explorer | Quantization explorer: bits, granularity, outliers | 560}}

```python:quant_numpy.py
import numpy as np

rng = np.random.default_rng(0)
# 512 output channels whose scales differ (log-normal), plus one outlier weight: typical of real layers.
W = rng.standard_normal((512, 512)) * np.exp(0.7 * rng.standard_normal((512, 1)))
W[3, 7] = 12.0

def quantize_sym(W, bits, axis=None):
    qmax = 2 ** (bits - 1) - 1
    scale = np.abs(W).max(axis=axis, keepdims=True) / qmax        # per-tensor if axis=None, per-row if axis=1
    Wq = np.clip(np.round(W / scale), -qmax, qmax)                # the stored integers
    return Wq.astype(np.int8), scale

def rel_rmse(W, Wq, scale):
    return np.sqrt(((W - Wq * scale) ** 2).mean() / (W ** 2).mean())

print("bits  per-tensor  per-channel")
for bits in (8, 6, 5, 4, 3):
    r_t = rel_rmse(W, *quantize_sym(W, bits, axis=None))
    r_c = rel_rmse(W, *quantize_sym(W, bits, axis=1))
    print(f"{bits:4d}  {r_t:10.4f}  {r_c:11.4f}")
```

```chart
{
  "type": "bar",
  "caption": "Relative RMS reconstruction error of symmetric weight quantization on a synthetic 512x512 matrix (log-normal row scales, one large outlier), from quant_numpy.py above. Per-channel scales are 3 to 4 times more accurate here; at 4 bits per-tensor quantization destroys half the signal.",
  "height": 340,
  "data": { "labels": ["8 bits", "6 bits", "5 bits", "4 bits", "3 bits"],
    "datasets": [
      { "label": "per-tensor", "data": [0.032, 0.132, 0.265, 0.493, 0.779], "backgroundColor": "var(--c-red)" },
      { "label": "per-channel", "data": [0.0075, 0.031, 0.064, 0.136, 0.317], "backgroundColor": "var(--c-green)" }
    ] },
  "options": { "scales": { "y": {"title": {"display": true, "text": "relative RMS error"}} } }
}
```

### Post-training quantization (PTQ)

**PTQ** quantizes an already-trained model without retraining. Weights are easy: pick scales from their ranges. **Activations** are harder because their ranges depend on the input. *Dynamic* quantization computes the activation scale at run time from each tensor's actual min/max (no calibration, a little overhead). *Static* quantization runs a few hundred **calibration** batches through the model to record activation ranges (min/max, a percentile, or a range minimizing KL divergence or MSE to the float values) and fixes the scales. Integer matmuls then accumulate in int32 and are rescaled: with $y=\sum_k x_k w_k$ and $x\approx s_x x_q$, $w\approx s_w w_q$ we get $y\approx s_xs_w\sum_k x_{q,k}w_{q,k}$, an integer dot product times one floating-point factor (Jacob et al., CVPR 2018, the scheme behind TensorFlow Lite and PyTorch's `fbgemm`/`qnnpack` backends).

```python:ptq_dynamic.py
import io, torch, torch.nn as nn

def size_mb(m):
    buf = io.BytesIO(); torch.save(m.state_dict(), buf); return buf.getbuffer().nbytes / 2**20

torch.manual_seed(0)
fp32 = nn.Sequential(nn.Linear(784, 1024), nn.ReLU(), nn.Linear(1024, 1024), nn.ReLU(), nn.Linear(1024, 10)).eval()

# Dynamic PTQ: int8 weights (per-tensor by default), activations quantized on the fly. CPU inference.
int8 = torch.ao.quantization.quantize_dynamic(fp32, {nn.Linear}, dtype=torch.qint8)

x = torch.randn(64, 784)
with torch.no_grad():
    d = (fp32(x) - int8(x)).abs().max().item()
print(f"size: {size_mb(fp32):.2f} MB -> {size_mb(int8):.2f} MB   max|logit diff| = {d:.4f}")
print(int8[0])      # DynamicQuantizedLinear(in_features=784, out_features=1024, dtype=torch.qint8, ...)
```

### Quantization-aware training (QAT)

PTQ breaks down at very low bit widths or for sensitive architectures. **QAT** simulates quantization *during training* so that the network learns weights that are robust to it. The forward pass applies **fake quantization**, $\hat w=s\,\operatorname{clip}(\operatorname{round}(w/s))$, while the weights themselves remain floats. The difficulty is that $\operatorname{round}$ has zero gradient almost everywhere. The **straight-through estimator (STE)** (Bengio, Léonard and Courville, 2013) pretends $\operatorname{round}$ is the identity in the backward pass, so that $\partial\hat w/\partial w\approx 1$ inside the clipping range and $0$ outside. After training, the real integers are exported.

```python:qat_ste.py
import torch, torch.nn as nn, torch.nn.functional as F

class FakeQuant(torch.autograd.Function):
    """Symmetric per-tensor fake quantization with a straight-through gradient."""
    @staticmethod
    def forward(ctx, w, bits):
        qmax = 2 ** (bits - 1) - 1
        scale = w.detach().abs().max().clamp_min(1e-8) / qmax
        ctx.save_for_backward(w.detach().abs() <= qmax * scale)       # inside the clip range?
        return torch.clamp(torch.round(w / scale), -qmax, qmax) * scale
    @staticmethod
    def backward(ctx, g):
        (inside,) = ctx.saved_tensors
        return g * inside, None                                       # STE: pass gradient through where not clipped

class QLinear(nn.Linear):
    def __init__(self, i, o, bits=4):
        super().__init__(i, o); self.bits = bits
    def forward(self, x):
        return F.linear(x, FakeQuant.apply(self.weight, self.bits), self.bias)

torch.manual_seed(0)
X = torch.randn(2048, 20); y = (X[:, :5].sum(1) > 0).long()           # toy classification
for bits in (None, 4, 2):
    lin = (lambda i, o: QLinear(i, o, bits)) if bits else nn.Linear
    net = nn.Sequential(lin(20, 64), nn.ReLU(), lin(64, 2))
    opt = torch.optim.Adam(net.parameters(), 1e-2)
    for _ in range(300):
        loss = F.cross_entropy(net(X), y); opt.zero_grad(); loss.backward(); opt.step()
    print(f"bits={bits}: train acc {(net(X).argmax(1) == y).float().mean().item():.3f}")
```

PyTorch's `torch.ao.quantization` offers eager-mode and FX-graph workflows (`prepare_qat`, `convert`) that insert observers and fake-quant modules and fuse patterns such as Conv-BN-ReLU; the lab below uses the simplest route. QAT is a short fine-tune (a few epochs at a small learning rate) starting from a trained float model.

### 8-bit and 4-bit quantization of large language models

LLMs revealed a new problem: their activations contain a few **outlier features**, hidden dimensions whose magnitudes are 10 to 100 times larger than the rest, which wreck per-tensor activation quantization. **LLM.int8()** (Dettmers et al., NeurIPS 2022) handles this by decomposing the matmul: the few outlier columns are computed in FP16, the rest in int8 with per-row/per-column scales.

At 4 bits and below, weights are quantized while activations stay in 16 bits (**weight-only quantization**), which is exactly what the memory-bound decode regime needs: bytes read per token drop 4x. Two influential post-training methods:

- **GPTQ** (Frantar, Ashkboos, Hoefler and Alistarh, 2022; ICLR 2023) quantizes a layer's weights one column at a time and, after rounding each, adjusts the *not-yet-quantized* weights to compensate for the error it caused, using second-order information (the Hessian of the layer's squared output error, $H=2XX^\top$, estimated from a few hundred calibration samples). It descends from the Optimal Brain Surgeon line of work and quantizes 175B-parameter models to 3 or 4 bits in a few GPU-hours.
- **AWQ** (Lin et al., 2023; MLSys 2024), *activation-aware weight quantization*, observes that a small fraction (about 1%) of weight channels matter disproportionately because they multiply large activations. Rather than keeping them in high precision, it scales those channels up before quantization (and the corresponding activations down), reducing their relative rounding error. The scale is chosen by a small search on calibration data.

**QLoRA** (Dettmers et al., 2023) combines a 4-bit **NormalFloat (NF4)** base model, whose 16 levels are placed at quantiles of a normal distribution because pretrained weights are roughly Gaussian, with LoRA adapters ([Week 10](week-10.html)), making it possible to fine-tune a 65B model on one 48 GB GPU.

:::warning Evaluate quantized models on what matters
Average perplexity or top-1 accuracy can hide damage on long-tail behaviors: rare classes, reasoning chains, long contexts, multilingual text. Evaluate the compressed model on the task metrics you care about and on a few hard slices, and always compare *latency on the target hardware*, since a 4-bit model on a GPU without a fast dequantization kernel can be slower than FP16.
:::

### Knowledge distillation

**Distillation** (Bucilua, Caruana and Niculescu-Mizil, 2006; Hinton, Vinyals and Dean, 2015) trains a small **student** to imitate a large **teacher**. Why is this better than training the student on the labels alone? Because the teacher's output distribution carries more information than a one-hot label: for an image of a BMW, the teacher might give 0.9 to "BMW", 0.09 to "other car" and $10^{-6}$ to "carrot". Those *relative* small probabilities ("dark knowledge") describe the similarity structure of the classes. But with a confident teacher the small probabilities are so tiny that they barely affect the loss. Hinton et al. fix this with a **temperature** $T$: divide the logits $z_i$ by $T$ before the softmax,

$$
p_i^{(T)}=\frac{\exp(z_i/T)}{\sum_j\exp(z_j/T)}.
$$

$T=1$ is the ordinary softmax; $T>1$ softens the distribution and exposes the ranking of the wrong classes; as $T\to\infty$ it tends to uniform.

```chart
{
  "type": "bar",
  "caption": "Softmax of the logits [8, 5, 2, 1, 0] at four temperatures (computed). At T=1 the top class has 95% and the others are almost invisible; at T=4 the distribution is soft enough that the student can learn the ordering of the wrong classes.",
  "height": 340,
  "data": { "labels": ["class 0", "class 1", "class 2", "class 3", "class 4"],
    "datasets": [
      { "label": "T = 1", "data": [0.949, 0.047, 0.002, 0.001, 0.0], "backgroundColor": "var(--c-blue)" },
      { "label": "T = 2", "data": [0.757, 0.169, 0.038, 0.023, 0.014], "backgroundColor": "var(--c-green)" },
      { "label": "T = 4", "data": [0.499, 0.236, 0.111, 0.087, 0.068], "backgroundColor": "var(--c-amber)" },
      { "label": "T = 8", "data": [0.34, 0.233, 0.16, 0.142, 0.125], "backgroundColor": "var(--c-purple)" }
    ] },
  "options": { "scales": { "y": {"title": {"display": true, "text": "probability"}} } }
}
```

The **Hinton loss** combines a soft-target term with the usual hard-label cross-entropy:

$$
\mathcal L=(1-\lambda)\,\mathrm{CE}\bigl(y,\softmax(z^{s})\bigr)+\lambda\,T^2\,\KL\bigl(\softmax(z^{t}/T)\,\big\|\,\softmax(z^{s}/T)\bigr),
$$

where $z^t,z^s$ are teacher and student logits, $y$ the one-hot label, and $\lambda\in[0,1]$ a mixing weight. The student is evaluated at $T=1$ after training.

:::math Why the factor T squared
Let $q_i=\softmax(z^t/T)_i$ be the teacher's soft targets and $p_i=\softmax(z^s/T)_i$ the student's. The soft cross-entropy $C=-\sum_i q_i\log p_i$ has gradient with respect to a student logit $z_i^s$:
$$
\frac{\partial C}{\partial z^s_i}=\frac1T\,(p_i-q_i).
$$
(The usual softmax-cross-entropy derivative $p_i-q_i$ gets a factor $1/T$ from the chain rule through $z/T$.) If the logits are roughly zero-mean across classes, then for large $T$, $p_i\approx\frac1K\bigl(1+\frac{z_i^s}{T}\bigr)$ and similarly for $q_i$ (first-order expansion of the softmax, $K$ classes), so
$$
\frac{\partial C}{\partial z^s_i}\approx\frac1T\cdot\frac{1}{K}\cdot\frac{z_i^s-z_i^t}{T}=\frac{1}{KT^2}(z_i^s-z_i^t).
$$
The gradient magnitude shrinks like $1/T^2$. Multiplying the soft loss by $T^2$ keeps its gradient on the same scale as that of the hard-label loss, so you can change $T$ without re-tuning $\lambda$ or the learning rate. In the high-temperature limit the soft-target objective reduces to matching the (centered) logits, $\tfrac12\|z^s-z^t\|^2$ up to scaling.
:::

```python:distill_loss.py
import torch, torch.nn.functional as F

def distillation_loss(student_logits, teacher_logits, labels, T=4.0, lam=0.7):
    """Hinton et al. (2015): (1 - lam) * CE(hard) + lam * T^2 * KL(teacher_T || student_T)."""
    hard = F.cross_entropy(student_logits, labels)
    soft = F.kl_div(F.log_softmax(student_logits / T, dim=-1),       # input: log-probs of the student
                    F.softmax(teacher_logits / T, dim=-1),           # target: probs of the teacher
                    reduction="batchmean") * (T * T)
    return (1 - lam) * hard + lam * soft

torch.manual_seed(0)
s = torch.randn(8, 10, requires_grad=True); t = torch.randn(8, 10) * 3; y = torch.randint(0, 10, (8,))
loss = distillation_loss(s, t, y); loss.backward()
print(loss.item(), s.grad.shape)
```

Distillation variants include matching intermediate features (FitNets), attention maps, and sequence-level distillation for generation. DistilBERT (Sanh et al., 2019) retained about 97% of BERT's GLUE performance with 40% fewer parameters and 60% faster inference. Many compact LLMs today are trained on teacher-generated text, which is distillation at the data level. Distillation does not need the teacher's weights, only its outputs, so it also works across architectures; it is the compression method with the least hardware dependency, because the student is an ordinary smaller dense network.

### Low-rank factorization

A dense layer $W\in\R^{m\times n}$ costs $mn$ parameters and multiply-adds per input. If $W$ is approximately low-rank, replace it by $W\approx UV$ with $U\in\R^{m\times r}$, $V\in\R^{r\times n}$: parameters and FLOPs drop to $r(m+n)$, a saving whenever $r<mn/(m+n)$ (for square $n\times n$, whenever $r<n/2$). The best rank-$r$ approximation in Frobenius norm is the truncated SVD (Eckart-Young theorem): $W=\sum_i\sigma_iu_iv_i^\top\approx\sum_{i\le r}\sigma_iu_iv_i^\top$, with error $\|W-W_r\|_F^2=\sum_{i>r}\sigma_i^2$. Convolutions can be factorized analogously (a $k\times k$ convolution into $k\times1$ and $1\times k$, or into a low-rank channel bottleneck), a theme you saw with depthwise-separable convolutions in [Week 6](week-06.html). Low-rank structure is also the idea behind LoRA, but there the factorization is of the *update*, not of the pretrained weight.

```python:lowrank.py
import torch, torch.nn as nn

def factorize_linear(layer: nn.Linear, rank: int) -> nn.Sequential:
    """Replace Linear(n -> m) by Linear(n -> r, no bias) followed by Linear(r -> m) using truncated SVD."""
    W = layer.weight.data                                  # (m, n)
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)    # W = U diag(S) Vh
    A = nn.Linear(W.shape[1], rank, bias=False)
    B = nn.Linear(rank, W.shape[0], bias=layer.bias is not None)
    A.weight.data = (S[:rank, None] * Vh[:rank]).contiguous()     # (r, n)
    B.weight.data = U[:, :rank].contiguous()                       # (m, r)
    if layer.bias is not None:
        B.bias.data = layer.bias.data.clone()
    return nn.Sequential(A, B)

torch.manual_seed(0)
big = nn.Linear(1024, 1024)
small = factorize_linear(big, rank=128)
x = torch.randn(16, 1024)
p = lambda m: sum(q.numel() for q in m.parameters())
print("params:", p(big), "->", p(small), "| relative output error:",
      ((big(x) - small(x)).norm() / big(x).norm()).item())
```

Whether truncation works depends entirely on how fast the singular values decay. A random matrix has a flat spectrum and cannot be compressed; trained weight matrices usually have much more structure, and a short fine-tune after truncation recovers most of the loss. The chart compares magnitude pruning and truncated SVD on the same synthetic matrix as in the quantization experiment.

```chart
{
  "type": "scatter",
  "caption": "Relative Frobenius error versus the fraction of parameters kept, for magnitude pruning and truncated SVD, on the same synthetic 512x512 matrix as above (computed with NumPy). Illustrative only: a random-like matrix is close to worst case. Trained weights, which have decaying spectra and many near-zero entries, compress far better, and error in weights is not error in accuracy.",
  "height": 340,
  "data": { "datasets": [
    { "label": "magnitude pruning", "showLine": true, "borderColor": "var(--c-blue)", "backgroundColor": "var(--c-blue)", "data": [{"x":0.01,"y":0.858},{"x":0.05,"y":0.664},{"x":0.1,"y":0.538},{"x":0.2,"y":0.385},{"x":0.3,"y":0.284},{"x":0.5,"y":0.149},{"x":0.7,"y":0.064},{"x":0.9,"y":0.012},{"x":1.0,"y":0.0}] },
    { "label": "truncated SVD (rank r keeps r(m+n)/(mn) = r/256)", "showLine": true, "borderColor": "var(--c-amber)", "backgroundColor": "var(--c-amber)", "data": [{"x":0.016,"y":0.945},{"x":0.031,"y":0.902},{"x":0.062,"y":0.833},{"x":0.125,"y":0.729},{"x":0.25,"y":0.586},{"x":0.5,"y":0.397},{"x":0.75,"y":0.271},{"x":1.0,"y":0.178}] }
  ] },
  "options": { "scales": { "x": {"title": {"display": true, "text": "fraction of parameters kept"}}, "y": {"title": {"display": true, "text": "relative error ||W - W'||/||W||"}} } }
}
```

## Mixture-of-Experts {#moe}

The scaling laws of [Week 10](week-10.html) say that capacity (parameters) and compute are tied together in a dense model: doubling the parameters doubles the FLOPs per token. **Sparse Mixture-of-Experts (MoE)** breaks this link. Replace the transformer's feed-forward block with $E$ parallel feed-forward networks (the **experts**) and a small **router** that sends each token to only $k$ of them (typically $k=1$ or 2). Parameters grow $E$-fold, but FLOPs per token grow only with $k$. The idea of gating among experts goes back to Jacobs, Jordan, Nowlan and Hinton (1991); Shazeer et al. (2017) made it work at scale with a sparsely-gated layer in LSTMs, and GShard (Lepikhin et al., 2020), the Switch Transformer (Fedus, Zoph and Shazeer, 2021/JMLR 2022) and Mixtral 8x7B (Jiang et al., 2024) established it for transformers.

{{figure: week16-moe | A sparse MoE layer. The router scores each token against $E$ experts, keeps the top $k$, and combines their outputs with the renormalized gate weights. Experts not selected cost no FLOPs for this token.}}

### Routing

For a token representation $x\in\R^d$ the router has a weight matrix $W_r\in\R^{E\times d}$ and computes probabilities $p=\softmax(W_rx)\in\R^E$. The top-$k$ indices $\mathcal T(x)$ are selected, the gates are renormalized over them (in some variants not), and the layer output is

$$
y=\sum_{i\in\mathcal T(x)} g_i(x)\,\mathrm{Expert}_i(x),\qquad g_i=\frac{p_i}{\sum_{j\in\mathcal T(x)}p_j}.
$$

Top-1 routing (Switch) is simplest and cheapest; top-2 gives smoother gradients. Experts are spread over devices (**expert parallelism**): tokens are shuffled to the device holding their expert with an all-to-all communication, processed, and shuffled back. Because the number of tokens per expert varies with the routing decisions, implementations set a **capacity** per expert, $C=\text{capacity factor}\times\frac{\text{tokens}}{E}$; tokens beyond capacity are dropped (they pass through the residual connection unchanged) or rerouted.

### Load balancing

Left alone, routers collapse: a few experts get better, receive more tokens, get better still, and the rest are never trained (a rich-get-richer feedback loop). Unbalanced load also wastes hardware, since all devices wait for the busiest expert. The remedy (Shazeer et al., 2017; Switch Transformer) is an **auxiliary load-balancing loss**. For a batch of $T$ tokens define

- $f_i=\frac1T\sum_{x}\mathbb 1[\text{token }x\text{ is routed to expert }i]$, the *fraction of tokens* dispatched to expert $i$ (not differentiable), and
- $P_i=\frac1T\sum_x p_i(x)$, the *mean router probability* on expert $i$ (differentiable).

$$
\mathcal L_{\text{bal}}=\alpha\,E\sum_{i=1}^{E}f_i\,P_i .
$$

:::math Why this loss is minimized by uniform routing
Both $f$ and $P$ are probability vectors summing to 1 (for top-1 routing). For perfectly balanced routing $f_i=P_i=1/E$, so $\sum_if_iP_i=E\cdot\frac1{E^2}=\frac1E$ and $\mathcal L_{\text{bal}}=\alpha$. For the collapsed case where all tokens go to expert 1 with $P_1=1$, $f_1=1$: $\sum_if_iP_i=1$ and $\mathcal L_{\text{bal}}=\alpha E$, $E$ times larger. In general, if $f$ and $P$ are aligned, $\sum f_iP_i$ is large; the minimum over distributions with the same total mass is at the uniform one (for $f=P$, $\sum_ip_i^2\ge1/E$ by Cauchy-Schwarz, with equality iff $p$ is uniform). The gradient flows through $P_i$: it pushes the router probability *down* on experts that already receive many tokens, since the coefficient of $P_i$ is $\alpha Ef_i$. A typical $\alpha=10^{-2}$ is large enough to balance but small enough not to dominate the language-modelling loss. Some recent models use auxiliary-loss-free balancing via router bias terms or a router z-loss that penalizes large logits for stability.
:::

```python:moe_layer.py
import torch, torch.nn as nn, torch.nn.functional as F

class MoE(nn.Module):
    """Top-k mixture-of-experts feed-forward layer (single device, no capacity limit)."""
    def __init__(self, d, d_ff, n_experts=4, k=2):
        super().__init__()
        self.E, self.k = n_experts, k
        self.router = nn.Linear(d, n_experts, bias=False)
        self.experts = nn.ModuleList(
            nn.Sequential(nn.Linear(d, d_ff), nn.GELU(), nn.Linear(d_ff, d)) for _ in range(n_experts))

    def forward(self, x):                                   # x: (T, d), tokens flattened over batch and sequence
        probs = F.softmax(self.router(x), dim=-1)           # (T, E)
        topv, topi = probs.topk(self.k, dim=-1)             # (T, k)
        gates = topv / topv.sum(-1, keepdim=True)           # renormalize over the chosen experts
        y = torch.zeros_like(x)
        for e in range(self.E):
            tok, slot = (topi == e).nonzero(as_tuple=True)  # which tokens chose expert e, and in which of their k slots
            if tok.numel() == 0:
                continue
            y.index_add_(0, tok, gates[tok, slot, None] * self.experts[e](x[tok]))
        # load-balancing loss (Switch Transformer form; f counts top-1 assignments)
        f = F.one_hot(topi[:, 0], self.E).float().mean(0)   # fraction of tokens whose top choice is expert i
        P = probs.mean(0)                                   # mean router probability per expert
        aux = self.E * (f * P).sum()
        return y, aux

torch.manual_seed(0)
moe = MoE(d=64, d_ff=256, n_experts=4, k=2)
x = torch.randn(1000, 64)
y, aux = moe(x)
print("output:", y.shape, " aux loss (1.0 = perfectly balanced):", round(aux.item(), 3))
# total loss in training:  loss = task_loss + 0.01 * aux
```

Serving MoE models has a catch: although each token touches only $k$ experts, *all* expert weights must be in memory, and at small batch sizes different tokens use different experts, so decoding reads a large fraction of the parameters anyway. MoE helps most when throughput, not memory, is the limit.

## Neural architecture search (overview) {#nas}

**Neural architecture search (NAS)** automates the design of the network. A NAS method has three components: a **search space** (the set of candidate architectures, such as cell structures built from operations like 3x3 convolution, depthwise convolution, skip connection or pooling), a **search strategy** (how to explore it) and a **performance estimation strategy** (how to score a candidate cheaply).

- **Reinforcement learning and evolution.** Zoph and Le (ICLR 2017) used an RNN controller trained with REINFORCE to emit architectures, rewarded by validation accuracy; it needed hundreds of GPUs for days, but found cells that transferred to larger datasets (NASNet). Evolutionary approaches (AmoebaNet; Real et al., 2019) mutate and select architectures.
- **Differentiable NAS.** DARTS (Liu, Simonyan and Yang, ICLR 2019) relaxes the discrete choice of operation into a softmax-weighted mixture, $\bar o(x)=\sum_k\frac{\exp\alpha_k}{\sum_j\exp\alpha_j}o_k(x)$, and trains architecture parameters $\alpha$ by gradient descent alongside the weights, cutting the cost to a few GPU-days. It is known to be unstable (collapse toward parameter-free operations such as skip connections).
- **Weight sharing and one-shot methods.** ENAS (Pham et al., 2018) and **Once-for-All** (Cai et al., ICLR 2020) train one large "supernet" whose sub-networks share weights, then pull out specialized sub-networks for different latency targets without retraining.
- **Hardware-aware NAS** puts measured latency or energy on the target device into the objective (MnasNet, Tan et al., 2019; MobileNetV3; EfficientNet's compound scaling, [Week 6](week-06.html)).

NAS has produced strong, efficient vision models, but its limits are worth knowing. Random search in a well-designed search space is often a surprisingly strong baseline, results are sensitive to training-pipeline details, and for large language models the field has converged on scaling a fixed transformer design rather than searching. As a project tool, NAS is rarely worth its compute; **scaling width, depth and resolution with a simple rule and measuring latency on your device** usually gets you most of the way.

## Deployment: compilers, export formats and serving {#deployment}

### torch.compile: capture, fuse, generate

PyTorch's default **eager** mode runs one operator at a time, which is flexible but launches many small kernels and cannot fuse across operators. `torch.compile` (PyTorch 2.x) traces your Python code into a graph with **TorchDynamo** (a bytecode-level tracer that falls back to eager on code it cannot capture, inserting "graph breaks"), lowers it with **TorchInductor**, which fuses memory-bound operators and generates Triton kernels on GPUs (C++/OpenMP on CPUs), and caches the result. A single line enables it:

```python:compile_demo.py
import torch, torch.nn as nn, time

class Mlp(nn.Module):
    def __init__(self, d=2048):
        super().__init__()
        self.ln = nn.LayerNorm(d); self.fc1 = nn.Linear(d, 4 * d); self.fc2 = nn.Linear(4 * d, d)
    def forward(self, x):
        # LayerNorm + bias + GELU + residual: many memory-bound ops that Inductor can fuse
        return x + self.fc2(nn.functional.gelu(self.fc1(self.ln(x))))

dev = "cuda" if torch.cuda.is_available() else "cpu"
model = Mlp().to(dev).eval()
compiled = torch.compile(model)                      # mode="max-autotune" tries more kernel configs (slower compile)
x = torch.randn(64, 512, 2048, device=dev)

def bench(f, n=30):
    with torch.no_grad():
        for _ in range(5): f(x)                      # first call triggers compilation (seconds to minutes)
        if dev == "cuda": torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(n): f(x)
        if dev == "cuda": torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1e3

print(f"eager {bench(model):.2f} ms   compiled {bench(compiled):.2f} ms")
# Debug graph breaks:  TORCH_LOGS="graph_breaks" python compile_demo.py   or   torch._dynamo.explain(model)(x)
```

Gains typically range from 1.2x to 2x for transformer blocks, more for models dominated by elementwise operations. Pitfalls: the first call is slow; changing input shapes triggers recompilation (mark dynamic dimensions or pad to buckets); data-dependent control flow and `.item()` calls cause graph breaks; and for training, compile the model *before* wrapping it in DDP/FSDP only if your version documents that order (follow the current docs). For small models with tiny kernels, `mode="reduce-overhead"` uses **CUDA graphs** to replay a whole sequence of kernels with one launch, removing the CPU launch overhead that dominates there.

### TorchScript, torch.export and ONNX

To run a model **without a Python interpreter** (in a C++ server, on a phone, in a browser) you need a serialized, framework-neutral representation.

- **TorchScript** (`torch.jit.trace` and `torch.jit.script`) was PyTorch's first answer. *Tracing* runs the model once on an example input and records the operators executed; it is exact for straight-line code but silently bakes in the path taken by Python control flow. *Scripting* compiles the Python source to a restricted IR and handles loops and branches, but supports only a subset of Python. TorchScript is in maintenance mode; new work goes through `torch.export`.
- **`torch.export`** produces a sound, whole-graph capture (an ExportedProgram) with explicit dynamic-shape constraints; it is the basis of ExecuTorch for mobile and of AOT compilation.
- **ONNX** (Open Neural Network Exchange) is an open graph format with a standard set of operators ("opsets"). A model exported to ONNX can be executed by ONNX Runtime on CPU, GPU and many accelerators, or converted by vendor compilers such as TensorRT, OpenVINO and Core ML tools. It is the common interchange format when training and serving use different stacks.

```python:export_onnx.py
import torch, torch.nn as nn, numpy as np

model = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(16, 10)).eval()
example = torch.randn(1, 3, 32, 32)

# TorchScript by tracing
traced = torch.jit.trace(model, example)
traced.save("model_ts.pt")                                   # loadable from C++ with torch::jit::load

# ONNX with a dynamic batch dimension
torch.onnx.export(model, example, "model.onnx",
                  input_names=["image"], output_names=["logits"],
                  dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
                  opset_version=17)

# Verify numerically with ONNX Runtime (pip install onnxruntime)
import onnxruntime as ort
sess = ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])
batch = torch.randn(5, 3, 32, 32)
out_ort = sess.run(["logits"], {"image": batch.numpy()})[0]
with torch.no_grad():
    out_pt = model(batch).numpy()
print("max abs diff PyTorch vs ONNX Runtime:", np.abs(out_pt - out_ort).max())   # about 1e-6
```

Always **verify** an exported model against the original on real inputs (and different batch sizes), because unsupported or approximated operators are the usual source of silent differences. Quantized variants can be exported too (ONNX Runtime has its own int8 quantization tools, and TensorRT/OpenVINO calibrate int8 engines).

### Inference serving

Serving means answering requests under a latency target at the lowest cost. Training maximizes throughput over a long job; serving cares about **latency percentiles** (p50, p95, p99) and requests per second at a given hardware cost. The standard techniques:

- **Batching.** GPUs are inefficient at batch size 1 (low intensity, see the roofline). A server **dynamically batches**: it waits up to a few milliseconds to gather concurrent requests into one batch (this trades a small added latency for much higher throughput). Systems such as NVIDIA Triton Inference Server, TorchServe and Ray Serve implement it.
- **Continuous batching** (Orca, Yu et al., OSDI 2022) for LLMs: since requests produce different numbers of tokens, a static batch would wait for its longest member. Continuous batching schedules at *token* granularity, admitting new requests into the running batch as soon as another finishes. Combined with PagedAttention (vLLM), it raises LLM throughput by large factors over naive batching.
- **Two phases of LLM inference.** *Prefill* processes the whole prompt in parallel (compute-bound, large matmuls) and *decode* generates tokens one by one (memory-bound). They have different bottlenecks, which is why serving systems schedule them separately.
- **Model replication and autoscaling**, caching of repeated prompts or prefixes, request prioritization, and fallback to a smaller model under load.
- **Measure latency correctly**: report percentiles under realistic load rather than the mean of a single-request benchmark, and include pre/post-processing and network time.

### Edge and mobile considerations

On a phone, microcontroller or embedded board, the constraints change: memory of a few hundred MB, **energy** per inference (battery) and thermal limits, no large GPU, and a heterogeneous set of processors (CPU, GPU, DSP, NPU) each with its own supported operators and data types (frequently int8). Rules of thumb:

- **Design for the target.** Choose architectures that map well: depthwise-separable convolutions and inverted residual blocks (MobileNet, EfficientNet-Lite), operator sets supported by the NPU, static shapes.
- **Quantize**, usually to int8 with calibration or QAT; NPUs often support *only* integer math.
- **Measure on the device.** FLOPs and parameters are poor predictors of latency; memory access patterns, operator support and thread scheduling matter. A model with fewer FLOPs can be slower.
- **Runtimes:** ExecuTorch (PyTorch's on-device runtime), TensorFlow Lite / LiteRT, Core ML, ONNX Runtime Mobile, and llama.cpp-style runtimes for quantized LLMs on CPUs and phones. Convert, quantize and validate accuracy again after conversion.
- **Privacy and connectivity** are often the reason to go on-device: data never leaves the phone, and the app works offline. The model must then be small enough to ship in an app bundle and to update over the air.

## Assignment 4 and the project progress report {#deliverables}

Two deliverables fall due at the end of this week: **Assignment 4** (the Week 16 row of the assessment table in the syllabus) and the 2-page **project progress report**. Both reward the same habits, and this week's material is exactly what the efficiency-study option is about.

**What a strong Assignment 4 submission contains**, whichever of the three options you chose (generative modelling, efficiency study, GNN/RL task):

- A clearly stated question and a **baseline** that is trained properly before you try anything fancy (tuned learning rate, fixed seed, reported variance). An improvement over a weak baseline means nothing.
- **Controlled comparisons**: change one thing at a time, keep the rest of the pipeline identical, and report the mean and spread over at least 3 seeds for anything where noise could be mistaken for signal.
- For an **efficiency study** specifically: accuracy *and* a cost axis (latency, throughput, memory, model size, energy) on a named piece of hardware, measured with warm-up, synchronization and medians (see `bench` above); a **Pareto plot** of accuracy against cost with the baseline on it; and ablations that separate the contribution of each technique (AMP, compile, quantization, distillation, pruning). State batch size, precision, input resolution and hardware in every table. Include at least one negative result or limitation ("4-bit lost 3 points on the rare classes", "pruning to 90% gave no speed-up on this GPU"), and explain it using the concepts of this week (roofline, kernel support, granularity).
- A short **error analysis** or sanity check (what does the compressed model get wrong that the baseline got right?) and honest discussion of threats to validity.
- Reproducibility: code that runs end to end, a README with the commands, pinned library versions, and the seeds.
- The 2 to 4 page report: question, method, setup, results with tables and plots, ablations, discussion. Figures must have labeled axes and units, and every claim should point to a number.

**What a strong progress report contains** (2 pages, with baseline results): the problem and why it matters in two paragraphs; the dataset and its basic statistics; **a working baseline with numbers**, at least one simple model and one reasonable reference method, evaluated on a held-out split with the metric you will use in the final report; a table of planned experiments and ablations with status; a candid list of risks (data, compute, time) with fallback plans; and a one-paragraph timeline to Week 18. Include the first profiling numbers for your training loop if compute is a concern: knowing now that an epoch takes 40 minutes lets you plan the remaining experiments. Reports that arrive with only a plan and no baseline are the main cause of late-project trouble.

## Lab: profile a training loop, then apply AMP, quantization and distillation {#lab}

**Goal.** Starting from a deliberately naive CIFAR-10 training loop, (1) profile it and find the bottleneck, (2) speed it up with data-loading fixes, mixed precision and `channels_last`, (3) shrink a model with distillation, (4) quantize its weights, and (5) report an accuracy/latency/size table. Run on a Colab GPU runtime (T4 is enough; with a T4, the BF16 path is unavailable, so the code selects FP16 with a GradScaler automatically). Epoch counts are small; increase them if you want better numbers.

### Step 1: data, models and measurement helpers

```python:lab_setup.py
import time, io, copy, torch, torch.nn as nn, torch.nn.functional as F
import torchvision, torchvision.transforms as T
from torch.utils.data import DataLoader

dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
train_tf = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize(mean, std)])
test_tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
train_ds = torchvision.datasets.CIFAR10("./data", train=True, download=True, transform=train_tf)
test_ds = torchvision.datasets.CIFAR10("./data", train=False, download=True, transform=test_tf)

def loaders(num_workers, pin):
    tr = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=num_workers, pin_memory=pin,
                    persistent_workers=num_workers > 0, drop_last=True)
    te = DataLoader(test_ds, batch_size=1000, num_workers=num_workers, pin_memory=pin)
    return tr, te

def make_cnn(w):
    """VGG-style CNN with BatchNorm; width w controls the size (teacher w=64, student w=16)."""
    def blk(i, o):
        return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))
    return nn.Sequential(
        blk(3, w), blk(w, w), nn.MaxPool2d(2),                 # 16x16
        blk(w, 2 * w), blk(2 * w, 2 * w), nn.MaxPool2d(2),     # 8x8
        blk(2 * w, 4 * w), nn.MaxPool2d(2),                    # 4x4
        nn.Flatten(), nn.Linear(4 * w * 16, 512), nn.ReLU(inplace=True), nn.Dropout(0.3), nn.Linear(512, 10))

def n_params(m): return sum(p.numel() for p in m.parameters())
def size_mb(m):
    b = io.BytesIO(); torch.save(m.state_dict(), b); return b.getbuffer().nbytes / 2**20

@torch.no_grad()
def evaluate(model, loader, amp=False):
    model.eval(); correct = 0
    for x, y in loader:
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        with torch.autocast(dev, dtype=torch.float16, enabled=amp and dev == "cuda"):
            correct += (model(x).argmax(1) == y).sum().item()
    return correct / len(loader.dataset)

@torch.no_grad()
def latency_ms(model, device, batch=1, iters=100, shape=(3, 32, 32)):
    """Median forward latency (ms) for a given batch size, with warm-up and CUDA synchronization."""
    model.eval().to(device); x = torch.randn(batch, *shape, device=device)
    for _ in range(20): model(x)
    ts = []
    for _ in range(iters):
        if device == "cuda": torch.cuda.synchronize()
        t0 = time.perf_counter(); model(x)
        if device == "cuda": torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return sorted(ts)[len(ts) // 2]

teacher, student = make_cnn(64), make_cnn(16)
print(f"teacher {n_params(teacher)/1e6:.2f}M params, student {n_params(student)/1e6:.2f}M params")
```

### Step 2: the naive loop, profiled

```python:lab_baseline.py
from torch.profiler import profile, ProfilerActivity, schedule

def train_epoch(model, loader, opt, amp=False, scaler=None, channels_last=False, teacher=None,
                T_=4.0, lam=0.7, max_steps=None, prof=None):
    """One epoch. Returns images/second. Optionally distills from `teacher` (Hinton loss)."""
    model.train(); n = 0
    if dev == "cuda": torch.cuda.synchronize()
    t0 = time.perf_counter()
    for step, (x, y) in enumerate(loader):
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        if channels_last: x = x.contiguous(memory_format=torch.channels_last)
        with torch.autocast(dev, dtype=torch.float16, enabled=amp and dev == "cuda"):
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            if teacher is not None:
                with torch.no_grad(): t_logits = teacher(x)
                soft = F.kl_div(F.log_softmax(logits.float() / T_, -1), F.softmax(t_logits.float() / T_, -1),
                                reduction="batchmean") * T_ * T_
                loss = (1 - lam) * loss + lam * soft
        opt.zero_grad(set_to_none=True)
        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        else:
            loss.backward(); opt.step()
        n += x.size(0)
        if prof is not None: prof.step()
        if max_steps and step + 1 >= max_steps: break
    if dev == "cuda": torch.cuda.synchronize()
    return n / (time.perf_counter() - t0)

# Naive configuration: 0 workers, no pinned memory, FP32.
tr0, te0 = loaders(num_workers=0, pin=False)
model = make_cnn(64).to(dev)
opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, nesterov=True, weight_decay=5e-4)

acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if dev == "cuda" else [])
with profile(activities=acts, schedule=schedule(wait=1, warmup=2, active=5), record_shapes=True) as prof:
    ips = train_epoch(model, tr0, opt, max_steps=9, prof=prof)
print(f"naive: {ips:.0f} img/s")
print(prof.key_averages().table(sort_by="cuda_time_total" if dev == "cuda" else "cpu_time_total", row_limit=8))
```

**What to look for.** On a typical Colab T4 with a single worker, the profile will show the GPU idle for a large fraction of each step: the `DataLoader` (random crop, flip, `ToTensor`, `Normalize` on PIL images in the main process) is the bottleneck, not the convolutions. In the table, `aten::cudnn_convolution` and `aten::cudnn_batch_norm` dominate GPU time, but the wall-clock step is much longer than their sum. Write down the images per second and the GPU utilization (`nvidia-smi dmon` in another cell).

### Step 3: apply the fixes one at a time

```python:lab_speedups.py
configs = [
    ("naive (0 workers, FP32)",              dict(workers=0, pin=False, amp=False, cl=False)),
    ("+ 4 workers, pinned memory",           dict(workers=4, pin=True,  amp=False, cl=False)),
    ("+ mixed precision (AMP)",              dict(workers=4, pin=True,  amp=True,  cl=False)),
    ("+ channels_last",                      dict(workers=4, pin=True,  amp=True,  cl=True)),
]
rows = []
for name, c in configs:
    tr, te = loaders(c["workers"], c["pin"])
    m = make_cnn(64).to(dev)
    if c["cl"]: m = m.to(memory_format=torch.channels_last)
    o = torch.optim.SGD(m.parameters(), lr=0.05, momentum=0.9, nesterov=True, weight_decay=5e-4)
    sc = torch.amp.GradScaler("cuda", enabled=c["amp"] and dev == "cuda")
    train_epoch(m, tr, o, c["amp"], sc, c["cl"], max_steps=10)                    # warm-up (cuDNN autotune, workers)
    ips = train_epoch(m, tr, o, c["amp"], sc, c["cl"], max_steps=60)
    rows.append((name, ips)); print(f"{name:32s} {ips:8.0f} img/s")
print(f"speed-up over naive: {rows[-1][1] / rows[0][1]:.1f}x")
# Optional: model = torch.compile(model) before the loop; compare again after the first (slow) compile step.
```

Expect each change to move throughput in the direction the theory predicts: workers and pinned memory fix the starving GPU; AMP puts the convolutions on tensor cores and halves the memory traffic of activations; `channels_last` removes layout conversions around tensor-core convolutions. Record the exact numbers for your hardware. If AMP gives nothing, you are still input-bound or launch-bound: re-profile.

### Step 4: train the teacher and distill the student

```python:lab_distill.py
tr, te = loaders(4, True)
EPOCHS = 8

def fit(model, epochs, teacher=None, lr=0.05):
    model.to(dev).to(memory_format=torch.channels_last)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * len(tr))
    scaler = torch.amp.GradScaler("cuda", enabled=dev == "cuda")
    for ep in range(epochs):
        model.train()
        for x, y in tr:
            x, y = x.to(dev, non_blocking=True).contiguous(memory_format=torch.channels_last), y.to(dev, non_blocking=True)
            with torch.autocast(dev, dtype=torch.float16, enabled=dev == "cuda"):
                logits = model(x); loss = F.cross_entropy(logits, y)
                if teacher is not None:
                    with torch.no_grad(): tl = teacher(x)
                    soft = F.kl_div(F.log_softmax(logits.float() / 4, -1), F.softmax(tl.float() / 4, -1), reduction="batchmean") * 16
                    loss = 0.3 * loss + 0.7 * soft
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        print(f"  epoch {ep + 1}/{epochs}  test acc {evaluate(model, te, amp=True):.4f}")
    return model

print("teacher (w=64):");           teacher = fit(make_cnn(64), EPOCHS)
teacher.eval()
print("student from labels only (w=16):");   s_plain = fit(make_cnn(16), EPOCHS)
print("student distilled (w=16):"); s_dist = fit(make_cnn(16), EPOCHS, teacher=teacher)
```

The student has about 1/16 of the teacher's convolution parameters. A distilled student typically gains a few tenths of a point to about two points of accuracy over the label-only student on CIFAR-10, with the gap widening for smaller students and fewer labels; your numbers on 8 epochs will have noise of a similar size, so run two or three seeds before claiming an effect. Try $T\in\{1,2,4,8\}$ and $\lambda\in\{0.3,0.7,1\}$ and note which matter.

### Step 5: quantize and build the trade-off table

Convolution kernels need a full static-quantization toolchain to run as real int8, so the lab separates two questions. *Accuracy:* simulate per-channel weight-only quantization by rounding the weights in place (exactly what a 4-bit weight-only deployment computes, up to kernel details). *Latency:* measure what real int8 kernels give you for the `Linear` layers using `quantize_dynamic` on the CPU, and measure FP32 versus FP16 on the GPU.

```python:lab_quantize.py
@torch.no_grad()
def fake_quantize_weights(model, bits, per_channel=True):
    """Round Conv/Linear weights to `bits` (symmetric), in place; returns the model."""
    qmax = 2 ** (bits - 1) - 1
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            w = m.weight
            amax = w.abs().amax(dim=tuple(range(1, w.dim())), keepdim=True) if per_channel else w.abs().max()
            s = amax.clamp_min(1e-8) / qmax
            m.weight.copy_(torch.clamp(torch.round(w / s), -qmax, qmax) * s)
    return model

# Accuracy of the distilled student under weight-only quantization (BatchNorm stays float).
base_acc = evaluate(s_dist, te, amp=False)
print(f"student FP32: {base_acc:.4f}")
for bits in (8, 6, 4, 3):
    for pc in (False, True):
        q = fake_quantize_weights(copy.deepcopy(s_dist), bits, per_channel=pc)
        print(f"  {bits}-bit {'per-channel' if pc else 'per-tensor '}: {evaluate(q, te):.4f}")

# Real int8 kernels for the Linear layers (dynamic quantization, CPU).
cpu_fp32 = copy.deepcopy(s_dist).cpu().float().eval()
cpu_int8 = torch.ao.quantization.quantize_dynamic(copy.deepcopy(cpu_fp32), {nn.Linear}, dtype=torch.qint8)

# Build the table
rows = []
def add(name, model, device, acc, amp=False):
    m = copy.deepcopy(model).to(device).eval()
    if amp: m = m.half()
    lat1 = latency_ms(m, device, batch=1) if not amp else None
    rows.append((name, n_params(model) / 1e6, size_mb(m), acc, lat1))

acc_t = evaluate(teacher, te); acc_s = evaluate(s_plain, te); acc_d = evaluate(s_dist, te)
for name, mdl, acc in [("teacher w=64", teacher, acc_t), ("student w=16 (labels)", s_plain, acc_s), ("student w=16 (distilled)", s_dist, acc_d)]:
    add(name + " [GPU/CPU fp32]", mdl, dev, acc)
rows.append(("student distilled CPU fp32", n_params(cpu_fp32) / 1e6, size_mb(cpu_fp32), acc_d, latency_ms(cpu_fp32, "cpu", batch=1)))
rows.append(("student distilled CPU int8 Linear", n_params(cpu_fp32) / 1e6, size_mb(cpu_int8),
             evaluate_cpu := None, latency_ms(cpu_int8, "cpu", batch=1)))
print(f"{'model':40s} {'params(M)':>9s} {'size(MB)':>9s} {'acc':>7s} {'lat b=1 (ms)':>13s}")
for r in rows:
    acc_str = f"{r[3]:.4f}" if r[3] is not None else "   n/a"
    print(f"{r[0]:40s} {r[1]:9.2f} {r[2]:9.2f} {acc_str:>7s} {r[4]:13.2f}")
```

**Report.** Hand in the table plus (a) the throughput numbers from Step 3 with the profiler findings that motivated each change, (b) a plot of accuracy versus latency or size for teacher, plain student, distilled student and the quantized variants, and (c) a paragraph answering: which technique gave the best speed-up per accuracy point lost? Did 4-bit weights shrink *latency* in your setup, or only size? (On a GPU without fused dequantization kernels, or for convolutions at large batch, it usually does not: explain using the roofline.) What would change at batch size 1 versus 256?

:::note Honest measurement checklist
Fix seeds and report at least two runs for accuracy; use the same input size and batch for all latency numbers; synchronize the GPU; report median and p95; state the exact hardware and library versions; keep the CPU's thread count fixed (`torch.set_num_threads`); and never compare a latency measured with `torch.no_grad()` against one measured without it.
:::

## Common mistakes and debugging tips {#mistakes}

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| GPU utilization 20 to 40%, throughput barely changes with a faster GPU | input pipeline, CPU-side work, `.item()` or `print` every step, tiny kernels | profile; add workers and `pin_memory`; log every N steps; larger batch; `torch.compile` |
| Timing results are absurdly fast or vary wildly | timed asynchronous CUDA launches without `torch.cuda.synchronize()`; no warm-up; including compile time | synchronize, warm up, take medians |
| Loss becomes `nan` after enabling FP16 | overflow in activations or gradients; unscaled clipping; `GradScaler` missing | use BF16 if available; `scaler.unscale_` before clipping; check the scale value; keep sensitive ops in FP32 |
| AMP is slower than FP32 | tiny matmuls, dimension sizes not multiples of 8, input-bound loop, many casts | pad dimensions to multiples of 8 (or 64), larger batch, profile |
| Gradient accumulation gives a different loss curve than the big batch | loss not divided by the number of accumulation steps; BatchNorm statistics; unequal token counts | scale the loss, use LayerNorm/GroupNorm, normalize by total tokens |
| Out-of-memory at the first backward, not at the forward | activations kept for backward; optimizer states allocated lazily at the first `step()` | checkpointing, smaller micro-batch, ZeRO/FSDP; budget for optimizer states |
| OOM only after many steps | memory fragmentation; accumulating tensors that retain graphs (`total += loss` instead of `loss.item()`) | detach before logging; `torch.cuda.empty_cache()` is rarely the fix; set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |
| DDP hangs at start or at an all-reduce | ranks run different code paths (conditionals on rank, uneven data so one rank finishes early); firewall; wrong `MASTER_ADDR` | identical control flow on all ranks; `drop_last=True`; `find_unused_parameters` only if truly needed (slower) |
| Every DDP epoch sees the same shuffling | missing `sampler.set_epoch(epoch)` | call it each epoch |
| Reloading a DDP/FSDP checkpoint fails or all ranks write the file | `module.` key prefix; sharded state dicts; saving on all ranks | save from rank 0 using the proper state-dict API; strip prefix |
| Pruned model is not faster | zeros are still stored and multiplied; unstructured sparsity without sparse kernels | structured pruning with physical removal, 2:4 sparsity on supported GPUs |
| Int8/int4 model loses several accuracy points | per-tensor scales with outliers; BatchNorm not folded; calibration data unrepresentative; sensitive first/last layers | per-channel or group scales, more calibration data, keep sensitive layers in higher precision, QAT |
| Quantized model is slower than FP16 | no fused int kernels for this operator/hardware; dequantize overhead; batch size 1 on a compute-bound path | check the backend supports the op; benchmark on target hardware |
| Distilled student no better than the plain one | temperature too low (no dark knowledge); teacher not accurate or overconfident; $T^2$ factor missing; student too weak | try $T=2$ to 8, check teacher accuracy, include $T^2$ |
| MoE routing collapses to one expert | no or too small load-balancing loss, router learning rate too high | add the auxiliary loss, router z-loss, noise in routing, check expert utilization histograms |
| ONNX/TorchScript output differs from PyTorch | tracing baked in Python control flow; train-mode layers (dropout, BN); unsupported op approximations | export in `eval()` mode, test with several input shapes, use `script` or `torch.export` |
| `torch.compile` is slow or recompiles constantly | dynamic shapes, graph breaks, Python side effects | `TORCH_LOGS=graph_breaks,recompiles`; fix shapes or mark dynamic |

**General debugging principle:** change one thing, measure, and compare with an unoptimized reference on the *same* inputs. Every optimization in this chapter can be verified numerically (loss curves, outputs within tolerance) and temporally (throughput, latency). If you cannot demonstrate both, you do not yet know that it worked.

## Exercises {#exercises}

:::exercise Exercise 1: Roofline classification
An A100 has $P=312$ TFLOP/s (BF16) and $B=2.0$ TB/s. (a) Compute the ridge point. (b) For a BF16 matrix multiply with $M=16$, $K=N=8192$, compute arithmetic intensity and decide whether it is memory- or compute-bound; give the best-case time. (c) By what factor does int8 weight-only quantization (activations stay BF16) speed up this operation in the memory-bound regime, approximately?

:::details Solution
(a) $I^*=312\times10^{12}/(2.0\times10^{12})=156$ FLOP/byte.

(b) $F=2MNK=2\cdot16\cdot8192\cdot8192=2.15\times10^9$. Traffic with 2 bytes per element: weights $KN=6.7\times10^7$ elements, $MK=1.3\times10^5$, $MN=1.3\times10^5$; total about $6.76\times10^7$ elements, $1.35\times10^8$ bytes. $I=2.15\times10^9/1.35\times10^8\approx15.9$ FLOP/B, about 10x below the ridge: **memory-bound**. Best-case time $\approx Q/B=1.35\times10^8/2\times10^{12}=67\ \mu\text{s}$ (the compute time would be only $2.15\times10^9/3.12\times10^{14}=6.9\ \mu\text{s}$).

(c) The traffic is dominated by the weights; going from 2 to 1 byte per weight roughly halves $Q$ (to about $6.8\times10^7$ B), so about **1.98x faster**, until the operation crosses the ridge. With 4-bit weights: about 3.9x.
:::
:::

:::exercise Exercise 2: Profiling diagnosis
A training loop on an A100 reports 35% GPU utilization. The profiler trace shows short GPU kernels separated by gaps of roughly 5 ms, and a `DataLoader.__next__` call of about 6 ms at the start of each 12 ms step. Give two experiments that distinguish an input-bound loop from a launch-bound one, and list three concrete fixes for each.

:::details Solution
*Experiments:* (1) Train repeatedly on one cached batch already on the GPU; if throughput rises sharply the input pipeline is the cause. (2) Double the batch size; if time per step barely changes, the GPU had slack and the problem is CPU launch overhead or input waiting; if it doubles, you were compute-bound. Also check whether the gaps occur *between* kernels inside the step (launch-bound) or only at the step start (input-bound). Here the 6 ms `DataLoader` wait points to input-bound.

*Fixes, input-bound:* more workers, `pin_memory=True` with `non_blocking=True` copies, `persistent_workers`/`prefetch_factor`, move augmentation to the GPU, store data in a faster format (preprocessed tensors, WebDataset/Parquet shards), overlap by prefetching. *Fixes, launch-bound:* increase batch size, `torch.compile` (fusion) or CUDA graphs (`mode="reduce-overhead"`), remove `.item()`/`print`/`cpu()` synchronizations in the loop, replace Python loops over small ops with batched ops.
:::
:::

:::exercise Exercise 3: Why FP16 needs loss scaling
(a) A gradient entry has true value $3\times10^{-9}$. What happens to it in FP16 without scaling? (b) With scale $S=2^{15}$, what is the stored value, and what is the risk of choosing $S$ too large? (c) Explain why BF16 does not need scaling, and what it gives up.

:::details Solution
(a) The smallest positive FP16 subnormal is $2^{-24}\approx6.0\times10^{-8}$; anything below about half of it ($3\times10^{-8}$) rounds to 0. So $3\times10^{-9}$ **underflows to zero** and the parameter receives no gradient signal.
(b) $3\times10^{-9}\times32768=9.8\times10^{-5}$, representable as a normal FP16 number (the smallest normal is $6.1\times10^{-5}$). Too large an $S$ pushes big gradients above 65,504, giving `inf`/`NaN`; dynamic scaling detects this, skips the step and halves $S$.
(c) BF16 has the same 8-bit exponent as FP32, so values down to $10^{-38}$ are representable: no underflow of gradients. It gives up mantissa precision (7 bits, relative error up to 0.4%), so small updates to weights are lost unless an FP32 master copy is kept, and results have more rounding noise.
:::
:::

:::exercise Exercise 4: Gradient accumulation equivalence
Show that accumulating gradients over $A$ micro-batches of size $b$ (each loss divided by $A$) gives the same gradient as one batch of size $Ab$ for a mean-reduced loss. Then explain precisely what goes wrong if the loss is a *sum over tokens* in a language model and micro-batches contain different numbers of non-padding tokens, and write the correct normalization.

:::details Solution
For a mean loss over $Ab$ examples, $\nabla\frac1{Ab}\sum_{i}\ell_i=\frac1A\sum_{a=1}^A\bigl(\frac1b\sum_{i\in a}\nabla\ell_i\bigr)$, which is exactly the average of the $A$ micro-batch mean-gradients; backward adds into `.grad`, so scaling each by $1/A$ reproduces it.

For token-level losses with $n_a$ valid tokens in micro-batch $a$, using mean-per-micro-batch and then dividing by $A$ gives $\frac1A\sum_a\frac1{n_a}\sum_{t\in a}\nabla\ell_t$, which weights tokens in short micro-batches more. The correct quantity is $\frac1{\sum_an_a}\sum_a\sum_{t\in a}\nabla\ell_t$: compute the *sum* of token losses in each micro-batch, divide by the **total** token count of the whole accumulated batch (known in advance by counting the labels that are not padding).
:::
:::

:::exercise Exercise 5: Checkpointing arithmetic
(a) A 48-layer transformer stores 2 GB of activations per layer. Without checkpointing, how much activation memory is needed? (b) With $k$ equal segments, what is the peak (in layers' worth of memory), and what $k$ minimizes it? Which integer $k$ divides 48 and gets closest to the optimum? (c) By what fraction does training time increase, if backward costs twice a forward pass?

:::details Solution
(a) $48\times2=96$ GB. (b) $M(k)=k+48/k$ layer-units (assuming checkpoints are as large as a layer's stored activations; for transformers the true ratio is much better because a checkpoint is just the block input). $k^\star=\sqrt{48}\approx6.9$; the divisors 6 and 8 give $M=6+8=14$ and $8+6=14$ layers, i.e. 28 GB, versus 48 layers (96 GB). (c) Without checkpointing a step costs $F+2F=3F$; with it $F+F+2F=4F$: about **+33%** time for a 3.4x memory reduction in this model.
:::
:::

:::exercise Exercise 6: ZeRO memory accounting
You want to fine-tune a 13B-parameter model with Adam in mixed precision (FP32 master weights, FP32 $m,v$) on 8 GPUs with 80 GB each. Ignoring activations, (a) compute the per-GPU model-state memory for DDP, ZeRO-1, ZeRO-2 and ZeRO-3. (b) Which stages fit? (c) What is the per-step communication volume per GPU (in elements, ring terms) for DDP and ZeRO-3?

:::details Solution
$\Psi=13\times10^9$. DDP: $16\Psi=208$ GB, does not fit. ZeRO-1: $4\Psi+12\Psi/8=52+19.5=71.5$ GB, fits barely, leaving about 8 GB for activations (too tight in practice). ZeRO-2: $2\Psi+14\Psi/8=26+22.75=48.75$ GB. ZeRO-3: $16\Psi/8=26$ GB. So stage 2 or 3 are comfortable; stage 3 also leaves the most room for activations and a bigger batch.

(c) DDP all-reduce: about $2\Psi$ elements per GPU (reduce-scatter plus all-gather, factor $(N-1)/N$ ignored), ZeRO-3: $3\Psi$ (parameter all-gather in forward, again in backward, plus gradient reduce-scatter), i.e. 1.5x.
:::
:::

:::exercise Exercise 7: Pipeline schedule design
A 24-layer model is split into $p=6$ pipeline stages of 4 layers. The global batch is 192 sequences. (a) For GPipe, compute the bubble fraction when the micro-batch size is 16, 8, 4 and 2. (b) Why can you not simply choose micro-batch size 1? (c) How many micro-batches' activations must stage 0 hold in GPipe versus 1F1B when $m=24$?

:::details Solution
(a) $m=192/\text{mb}$: mb 16 gives $m=12$: $5/17=29\%$; mb 8, $m=24$: $5/29=17\%$; mb 4, $m=48$: $5/53=9.4\%$; mb 2, $m=96$: $5/101=5.0\%$.
(b) Tiny micro-batches reduce the arithmetic intensity of each kernel (weights are re-read per micro-batch), fall to the memory-bound side of the roofline, and increase per-micro-batch communication and launch overhead; the throughput of each stage falls even as the bubble shrinks. The best $m$ balances both.
(c) GPipe: all $m=24$ micro-batches before the first backward arrives. 1F1B: at most $p=6$.
:::
:::

:::exercise Exercise 8: Derive tiled softmax by hand
Two blocks of scores for a single query: block 1 has scores $[1,3]$ with values $v_1=[1,0]$, $v_2=[0,1]$; block 2 has score $[2]$ with value $v_3=[1,1]$. Using the online-softmax recursion with running $(m,\ell,\tilde o)$, compute the attention output after processing both blocks and verify against the direct softmax.

:::details Solution
Block 1: $m_1=3$, $\ell_1=e^{1-3}+e^{0}=e^{-2}+1=1.1353$, $\tilde o_1=e^{-2}[1,0]+1\cdot[0,1]=[0.1353,1]$.
Block 2: score 2, $m_2=\max(3,2)=3$ (no change), $\ell_2=\ell_1+e^{2-3}=1.1353+0.3679=1.5032$, $\tilde o_2=\tilde o_1+e^{-1}[1,1]=[0.1353+0.3679,\ 1+0.3679]=[0.5032,1.3679]$. Output $=\tilde o_2/\ell_2=[0.3348,\ 0.9100]$.
Direct: softmax of $[1,3,2]$ is $[0.0900,0.6652,0.2447]$; $o=0.09[1,0]+0.6652[0,1]+0.2447[1,1]=[0.3347,0.9099]$. ✓. (A case where the second block raises the max, for instance a score of 5, exercises the rescaling factor $e^{m_t-m_{t+1}}$; try it.)
:::
:::

:::exercise Exercise 9: KV-cache budget (challenging)
A model has 80 layers, 64 query heads and head dimension 128, served in FP16 on an 80 GB GPU. The weights use 140 GB, so assume they are split across two GPUs with tensor parallelism and each holds 70 GB, leaving about 8 GB per GPU for the cache. (a) Compute the KV cache per token for multi-head attention. (b) Repeat with grouped-query attention with 8 KV heads. (c) How many 4096-token sequences fit in the 16 GB of total cache space in each case? (d) Explain why decoding throughput increases with batch size until the cache or the compute limit is reached, using the roofline.

:::details Solution
(a) $2\cdot80\cdot64\cdot128\cdot2=2{,}621{,}440$ B $=2.5$ MiB per token. (b) $2\cdot80\cdot8\cdot128\cdot2=327{,}680$ B $=0.31$ MiB per token, 8x smaller. (c) 4096 tokens: MHA $=10$ GiB per sequence, so 1 sequence in about 16 GB (about 1.5); GQA $=1.25$ GiB per sequence, about 12 sequences. (d) At batch 1 the decode step reads all weights for one token ($I\approx1$ FLOP/B). With batch $B$, the same weight read serves $B$ tokens, so the intensity grows roughly linearly in $B$ until it reaches the ridge ($\approx150$ for A100 in BF16); beyond that the step is compute-bound. In parallel, the KV-cache reads grow with $B$ and do not amortize (each sequence has its own cache), so at long contexts the cache traffic dominates, which is why GQA, cache quantization and PagedAttention matter.
:::
:::

:::exercise Exercise 10: Designing a compression pipeline (challenging)
You must deploy a 300M-parameter vision transformer on a phone NPU that supports only int8 and has 2 GB of RAM available, with a latency budget of 40 ms and at most 2 accuracy points of loss. Propose a pipeline using techniques from this week, say in which order you would apply them and why, how you would validate each step, and name two reasons why the final latency could still miss the budget.

:::details Solution
A reasonable plan: (1) *Establish baselines on the device first* (latency, memory, accuracy) and profile to find which layers dominate. (2) *Architecture-level reduction first*: choose or search a smaller or more NPU-friendly backbone (fewer layers/heads/hidden size, windowed or linear attention, lower resolution), because it is the only step that reduces compute for sure. (3) *Distill* from the 300M model into this student (logit plus intermediate-feature distillation) to regain accuracy. (4) *Structured pruning* (heads, MLP channels) if the budget is still missed, followed by a short fine-tune; unstructured sparsity is useless on an NPU. (5) *Quantization last*, because it is hardware-specific: PTQ with representative calibration data; if the loss exceeds the budget use QAT starting from the distilled model, per-channel weight scales, keep sensitive layers (patch embedding, final head, softmax/LayerNorm) in higher precision if the NPU permits. (6) Export through the vendor toolchain (ONNX or ExecuTorch), check numerical agreement on a held-out set, then measure on the real device with warm-up and thermal steady state. Validate after each step on accuracy and on-device latency.
Reasons latency could still be missed: unsupported operators (for example GELU, LayerNorm or attention) fall back to the CPU and incur copies; memory-bound operators and attention at the chosen resolution dominate regardless of int8; thermal throttling under sustained load; memory bandwidth contention with other apps.
:::
:::

## Key takeaways {#takeaways}

:::key Key takeaways
- **Data movement, not arithmetic, is usually the bottleneck.** Roofline: attainable FLOP/s $=\min(P,B\cdot I)$. Large matmuls are compute-bound; elementwise ops, normalization, softmax and token-by-token decoding are memory-bound. Fusion, FlashAttention and quantization all work by moving fewer bytes.
- **Measure before optimizing**: synchronize, warm up, use medians, profile with `torch.profiler`, and classify the bottleneck (input, launch overhead, memory-bound, compute-bound, communication). Check GPU utilization first.
- **Mixed precision** uses 16-bit compute with FP32 master weights and accumulation. FP16 needs dynamic loss scaling because of its narrow range; BF16 has FP32's range and usually needs none.
- **Gradient accumulation** reproduces a large batch exactly (scale the loss; watch BatchNorm and token counts). **Checkpointing** stores $k$ boundaries and recomputes the rest: memory $k+n/k$, minimized at $\sqrt n$, for about 33% more compute.
- **Distributed training**: DDP averages gradients with a ring all-reduce whose per-GPU cost is about $2S/\beta$ regardless of $N$. Tensor parallelism splits matrices (needs NVLink); pipeline parallelism splits layers with bubble $(p-1)/(m+p-1)$. **ZeRO/FSDP** shards optimizer states, gradients and parameters: $16\Psi\to16\Psi/N$ bytes at stage 3 for 1.5x the communication.
- **FlashAttention** computes exact attention tile by tile in SRAM using the online-softmax recursion, never writing the $N\times N$ matrix to HBM: same FLOPs, far less memory traffic and $O(N)$ memory. The **KV cache** makes decoding linear per token but costs $2\,L\,n_{kv}\,d\,b$ bytes per token; GQA, paging and quantization cut it. **Speculative decoding** uses a draft model and exact rejection sampling to produce several tokens per target pass.
- **Compression:** pruning (unstructured gives sparsity but rarely speed; structured gives smaller dense models), quantization (each bit is about 6 dB; scale granularity decides how well outliers are handled; PTQ vs QAT with the straight-through estimator; GPTQ/AWQ for 4-bit LLM weights), **distillation** (Hinton loss with temperature and the $T^2$ factor), low-rank factorization (truncated SVD). Always verify speed on the target hardware.
- **MoE** decouples parameters from per-token FLOPs via top-$k$ routing and needs a load-balancing loss $\alpha E\sum f_iP_i$ to avoid collapse. **NAS** automates architecture design, but a simple scaling rule plus on-device measurement is often enough.
- **Deployment:** `torch.compile` for fused kernels, ONNX/TorchScript/`torch.export` for portability, dynamic and continuous batching for serving, and int8 plus device-specific benchmarks on the edge.
:::

## Further reading {#further-reading}

**Essential**
- Micikevicius, P., et al. *Mixed Precision Training.* ICLR 2018. arXiv:1710.03740. The FP32 master weights and loss-scaling recipe.
- Rajbhandari, S., Rasley, J., Ruwase, O., He, Y. *ZeRO: Memory Optimizations Toward Training Trillion Parameter Models.* SC 2020. arXiv:1910.02054.
- Dao, T., Fu, D. Y., Ermon, S., Rudra, A., Re, C. *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.* NeurIPS 2022. arXiv:2205.14135.

**Also important**
- Hinton, G., Vinyals, O., Dean, J. *Distilling the Knowledge in a Neural Network.* 2015. arXiv:1503.02531.
- Han, S., Mao, H., Dally, W. J. *Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding.* ICLR 2016. arXiv:1510.00149.
- Williams, S., Waterman, A., Patterson, D. *Roofline: An Insightful Visual Performance Model for Multicore Architectures.* Communications of the ACM, 2009.
- Chen, T., Xu, B., Zhang, C., Guestrin, C. *Training Deep Nets with Sublinear Memory Cost.* 2016. arXiv:1604.06174.
- Milakov, M., Gimelshein, N. *Online normalizer calculation for softmax.* 2018. arXiv:1805.02867.
- Shoeybi, M., et al. *Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism.* 2019. arXiv:1909.08053.
- Narayanan, D., et al. *Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM.* SC 2021. arXiv:2104.04473.
- Huang, Y., et al. *GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism.* NeurIPS 2019. arXiv:1811.06965.
- Zhao, Y., et al. *PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel.* VLDB 2023. arXiv:2304.11277.
- Korthikanti, V., et al. *Reducing Activation Recomputation in Large Transformer Models.* 2022. arXiv:2205.05198.
- Jacob, B., et al. *Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference.* CVPR 2018. arXiv:1712.05877.
- Dettmers, T., Lewis, M., Belkada, Y., Zettlemoyer, L. *LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale.* NeurIPS 2022. arXiv:2208.07339.
- Frantar, E., Ashkboos, S., Hoefler, T., Alistarh, D. *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers.* ICLR 2023. arXiv:2210.17323.
- Lin, J., et al. *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration.* MLSys 2024. arXiv:2306.00978.
- Dettmers, T., Pagnoni, A., Holtzman, A., Zettlemoyer, L. *QLoRA: Efficient Finetuning of Quantized LLMs.* NeurIPS 2023. arXiv:2305.14314.
- Frankle, J., Carbin, M. *The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks.* ICLR 2019. arXiv:1803.03635.
- Shazeer, N., et al. *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer.* ICLR 2017. arXiv:1701.06538.
- Fedus, W., Zoph, B., Shazeer, N. *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity.* JMLR 2022. arXiv:2101.03961.
- Leviathan, Y., Kalman, M., Matias, Y. *Fast Inference from Transformers via Speculative Decoding.* ICML 2023. arXiv:2211.17192.
- Kwon, W., et al. *Efficient Memory Management for Large Language Model Serving with PagedAttention.* SOSP 2023. arXiv:2309.06180.
- Dao, T. *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning.* ICLR 2024. arXiv:2307.08691.
- Ainslie, J., et al. *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints.* EMNLP 2023. arXiv:2305.13245.
- Goyal, P., et al. *Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour.* 2017. arXiv:1706.02677.
- Zoph, B., Le, Q. V. *Neural Architecture Search with Reinforcement Learning.* ICLR 2017. arXiv:1611.01578.
- Liu, H., Simonyan, K., Yang, Y. *DARTS: Differentiable Architecture Search.* ICLR 2019. arXiv:1806.09055.
- Sanh, V., Debut, L., Chaumond, J., Wolf, T. *DistilBERT, a distilled version of BERT.* 2019. arXiv:1910.01108.

**Practical guides**
- He, H. *Making Deep Learning Go Brrrr From First Principles* (blog post, 2022).
- PyTorch documentation: *Automatic Mixed Precision*, *torch.profiler* tutorial, *Distributed Data Parallel*, *FSDP*, *torch.compile*, *Quantization*.
- Sze, V., Chen, Y.-H., Yang, T.-J., Emer, J. *Efficient Processing of Deep Neural Networks.* Morgan and Claypool, 2020.
