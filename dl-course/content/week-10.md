---
title: "Large Language Models and Pretrained NLP"
phase: Attention & Pretrained Models
summary: Pretraining objectives (BERT, GPT, T5), scaling laws and Chinchilla compute-optimal training, emergent abilities and their critiques, parameter-efficient fine-tuning (adapters, LoRA, QLoRA, prefix and prompt tuning), instruction tuning, in-context learning and chain-of-thought, alignment with SFT, RLHF and DPO, retrieval-augmented generation, evaluation, hallucination, decoding strategies, a Hugging Face lab, and a guide to reading and presenting papers.
---

:::key Learning objectives
By the end of this week you should be able to:
- Compare masked language modeling, autoregressive modeling and span corruption: what each loss predicts, which architecture it pairs with, and what each is good and bad at.
- Fit and use scaling laws: read a power law, derive the compute-optimal split of a budget between parameters and tokens from the Chinchilla loss model, and say what the laws do not tell you.
- Evaluate claims about "emergent abilities" critically, including the role of the metric.
- Derive the parameter savings of LoRA, size an adapter budget for a given model, and explain QLoRA's three ingredients, adapters, and prefix/prompt tuning.
- Describe the alignment pipeline (SFT, reward modeling, RLHF with a KL penalty) and derive the DPO loss from the KL-regularized reward-maximization objective.
- Explain in-context learning, prompting and chain-of-thought, and design a retrieval-augmented generation (RAG) system with its failure modes.
- Evaluate language models with perplexity and benchmarks, and recognize contamination, hallucination and other failure modes.
- Implement and compare decoding strategies (greedy, beam, temperature, top-k, top-p) and fine-tune BERT and a small causal LM with LoRA using Hugging Face.
- Read, summarize and present a research paper in a way that exposes its claims, evidence and limits.
:::

[Week 9](week-09.html) ended with a model you could train from scratch on a tiny corpus. This week asks what changes when the corpus has a trillion tokens and the model has billions of parameters. The surprising answer is that the architecture hardly changes at all. What changes is the recipe around it: how you turn raw text into a supervised problem without labels, how you decide how big to make the model and how long to train it, how you adapt it cheaply to a task, and how you steer it toward behavior people actually want. Those recipe questions occupy this chapter.

A useful framing is a three-stage pipeline that almost every modern language model follows:

1. **Pretraining** on a huge unlabeled corpus with a self-supervised objective. This is where nearly all the compute goes and where the model acquires knowledge of language, facts and patterns of reasoning.
2. **Adaptation**: supervised fine-tuning on a task or on instruction-response demonstrations, often with parameter-efficient methods.
3. **Alignment**: preference-based training that shapes the model's behavior (helpfulness, honesty, harmlessness) beyond what demonstrations can specify.

At inference time you add a fourth layer of control: prompting, retrieval and decoding. We cover all four.

:::note Terminology: LM, LLM, foundation model
A *language model* assigns probabilities to token sequences. A *large language model* (LLM) is a Transformer language model with billions of parameters, trained on hundreds of billions to trillions of tokens. "Foundation model" (Bommasani et al., 2021) is the broader term for any large model pretrained on broad data and adapted to many downstream uses, including vision and multimodal models ([Week 11](week-11.html)). There is no sharp size threshold; the words describe a regime in which pretraining followed by adaptation beats training task-specific models.
:::

{{figure: week10-timeline | A selected timeline of the ideas in this chapter. Dates are the years of the first public paper or release; the arrows of influence are discussed in the text.}}

## Pretraining objectives {#pretraining}

Supervised learning needs labels, and labels are expensive. Text, however, labels itself: any part of a document can be used to predict any other part. A **pretraining objective** is a rule for hiding part of the text and asking the model to recover it. The choice of rule determines which architecture is natural (recall the encoder/decoder/encoder-decoder families in [Week 9](week-09.html#architectures)), what the model is good at, and how efficiently it uses each training token.

{{figure: week10-objectives | The three canonical pretraining objectives on the same sentence. BERT hides tokens and reads in both directions; GPT predicts every next token from the left context; T5 replaces spans by sentinel tokens and generates the dropped spans.}}

### Masked language modeling (BERT) {#mlm}

Devlin, Chang, Lee and Toutanova (2019) trained a bidirectional Transformer encoder to fill in blanks. Pick a random set $M$ of about 15% of the positions in a sequence $\boldsymbol{x}=(x_1,\dots,x_T)$, corrupt them, and train the model to reconstruct the originals from the corrupted sequence $\tilde{\boldsymbol{x}}$:

$$
\mathcal{L}_{\text{MLM}}(\theta) \;=\; -\,\E_{\boldsymbol{x}}\,\E_{M}\Big[\sum_{i\in M}\log p_\theta\big(x_i \mid \tilde{\boldsymbol{x}}\big)\Big].
$$

The corruption is slightly subtle. Of the selected positions, 80% are replaced by a special `[MASK]` token, 10% by a random token, and 10% are left unchanged. The reason is a train/test mismatch: `[MASK]` never appears when you later fine-tune on a classification task, so if every selected token were masked the network could learn representations that only "switch on" at mask positions. Leaving some tokens unchanged or random forces the model to build a good representation of *every* input token, because any of them might be the one it will be graded on.

The model is the encoder stack from Week 9 (12 layers, hidden size 768, 12 heads, about 110M parameters for BERT-base; 24 layers, 1024, 16 heads, about 340M for BERT-large) with a linear-plus-softmax head over the vocabulary applied to the final hidden state at each masked position. The original recipe used English Wikipedia and BooksCorpus and a 30,000-token WordPiece vocabulary. BERT also included a second objective, next-sentence prediction, which later work (RoBERTa, Liu et al., 2019) showed to be unnecessary; RoBERTa instead trained longer, with larger batches and dynamic masking (a fresh mask each epoch), and improved every benchmark.

Three properties of MLM matter for how you use it:

- **Bidirectional context.** The prediction for position $i$ sees tokens on both sides. This yields excellent contextual representations for understanding tasks: classification, tagging, extractive question answering, sentence similarity.
- **A conditional independence assumption.** If two masked tokens are adjacent, the model predicts each given the unmasked text only, not given the other prediction. The joint $p(x_i, x_j\mid\tilde{\boldsymbol{x}})$ is approximated by a product. So the model is not a proper generative model of text.
- **Low token efficiency.** Only about 15% of positions produce a loss, so each pass over the data teaches less than an autoregressive pass. ELECTRA (Clark et al., 2020) attacks this by training a discriminator to detect which tokens a small generator has replaced, which gives a learning signal at every position.

:::history Why BERT was a turning point
Before 2018, NLP transfer meant reusing static word vectors ([Week 8](week-08.html)), occasionally adding contextual features from a pretrained LSTM (ELMo, Peters et al., 2018). Howard and Ruder (ULMFiT, 2018) and Radford et al. (GPT, 2018) showed that fine-tuning an entire pretrained language model worked better. BERT then set the state of the art on eleven tasks, including GLUE and SQuAD, with one recipe: pretrain once, fine-tune with a tiny task head. After BERT, "pretrain then fine-tune" became the default in NLP.
:::

### Autoregressive modeling (GPT) {#autoregressive}

The autoregressive objective factorizes the probability of a sequence with the chain rule and trains each conditional with cross-entropy:

$$
p_\theta(\boldsymbol{x}) = \prod_{t=1}^{T} p_\theta(x_t \mid x_{<t}), \qquad
\mathcal{L}_{\text{AR}}(\theta) = -\frac{1}{T}\sum_{t=1}^{T}\log p_\theta(x_t\mid x_{<t}).
$$

This is exactly the loss of the decoder-only model you built at the end of [Week 9](week-09.html). Causal masking lets all $T$ conditionals be computed in one parallel forward pass (teacher forcing), so the model receives $T$ training signals per sequence, not $0.15T$. Because the model defines a normalized distribution over sequences, you can sample from it, score text under it, and compute its perplexity (see [evaluation](#evaluation)).

GPT-1 (Radford et al., 2018) had about 117M parameters and was fine-tuned per task. GPT-2 (Radford et al., 2019) scaled to 1.5B parameters trained on WebText, around 40GB of text from outbound links of upvoted Reddit posts, and showed that a single model could attempt many tasks *zero-shot*, simply by being prompted. GPT-3 (Brown et al., 2020) scaled to 175B parameters trained on roughly 300B tokens and showed that few-shot prompting, giving a handful of examples in the context, was competitive with fine-tuned systems on many tasks. That observation, rather than any architectural novelty, reorganized the field.

Why did decoder-only models win for general-purpose systems? There is no single proven reason, but three practical ones are commonly cited. They use every position for learning. They unify pretraining, generation and prompting in one interface (everything is text continuation). And with a KV cache ([Week 9](week-09.html#kv-cache)) they are efficient at inference, since one forward pass per new token reuses all the previous keys and values.

### Span corruption and text-to-text (T5) {#span-corruption}

Raffel et al. (2020) asked a systematic question: given the same data and compute, which objective, architecture and dataset choices matter? Their answer was a framework in which every task, from translation to classification to regression, is cast as *text in, text out*, with a standard encoder-decoder Transformer. Their best pretraining objective was **span corruption**: choose spans of consecutive tokens (mean length 3, corrupting about 15% of tokens in total), replace each span in the input by a unique sentinel token such as `<extra_id_0>`, and train the decoder to emit the sentinels followed by the tokens they replaced.

For the sentence "The cat sat on the mat" with the span "cat sat" and the token "mat" dropped, the encoder sees `The <X> on the <Y>` and the decoder target is `<X> cat sat <Y> mat <Z>`. Compared with BERT's per-token masking, the encoder reads the corrupted input bidirectionally and the decoder is an ordinary causal language model over a short target. Targets are much shorter than the inputs, so the decoder side is cheap, and the sentinels let a single model reconstruct several spans in one pass. T5 was trained on C4 (the Colossal Clean Crawled Corpus, a filtered Common Crawl) in sizes up to 11B parameters.

Because the T5 formulation includes the task description in the input text (for example `translate English to German: That is good.`), a single model can be fine-tuned, and even trained multi-task, without any task-specific heads. That idea, "everything is a prompt", is the direct ancestor of [instruction tuning](#instruction-tuning).

### Comparing the objectives {#compare-objectives}

| | MLM (BERT) | Autoregressive (GPT) | Span corruption (T5) |
| --- | --- | --- | --- |
| Architecture | Encoder | Decoder | Encoder-decoder |
| Context for a prediction | Both sides | Left only | Bidirectional on the input, left on the output |
| Positions with loss | about 15% | 100% | about 15% of tokens, but in short targets |
| Generative? | Not directly | Yes, natively | Yes |
| Typical use today | Embeddings, classification, retrieval, reranking | General-purpose chat and code models | Translation, summarization, structured text-to-text |
| Few-shot prompting | Poor | Strong | Moderate |

Prefix-LM variants (the model reads a prefix bidirectionally and generates the rest causally) and mixtures of objectives (UL2, Tay et al., 2022) sit between these. In practice the field converged on decoder-only autoregressive models for large general-purpose systems, while encoder-only models remain very competitive for small, fast, task-specific classifiers and for embedding-based retrieval.

:::note A rule of thumb for choosing
Need a fast classifier or embeddings for a fixed task, with labeled data in the thousands? Fine-tune an encoder such as BERT or RoBERTa. Need open-ended generation, instruction following or few-shot flexibility? Use a decoder-only model. Need a seq2seq mapping (translation, summarization) at moderate scale with modest data? An encoder-decoder such as T5 remains an excellent choice.
:::

## Tokenization, briefly revisited {#tokenization}

[Week 8](week-08.html) derived byte-pair encoding (BPE). For LLMs the tokenizer is part of the model, fixed before pretraining, and its quirks leak into behavior. Three families dominate:

- **WordPiece** (BERT): greedy longest-match subword vocabulary with `##` continuation markers, vocabulary about 30k.
- **Byte-level BPE** (GPT-2 and successors): merges operate on UTF-8 bytes, so any string is representable and there is no unknown token. Vocabularies range from about 50k to over 100k.
- **SentencePiece** (T5, LLaMA): operates directly on raw text (treating whitespace as a symbol), with BPE or a unigram language model objective, which makes the tokenizer language-agnostic.

Why should you care? Because the tokenizer determines the unit of prediction. Numbers may be split into arbitrary chunks, which makes arithmetic harder; the same word with and without a leading space can be different tokens; non-English languages may need two to five times more tokens per word ("fertility"), making them more expensive and giving them a shorter effective context; and rare strings can become "glitch tokens" that appear in the vocabulary but almost never in training data, producing erratic output. Special tokens (`[CLS]`, `[SEP]`, `<|endoftext|>`, chat-role markers) are part of the interface, and forgetting them or adding them twice is one of the most common silent bugs.

```python:inspect_tokenizers.py
# Inspect how three tokenizer families segment the same strings.
# Requires: pip install transformers sentencepiece
from transformers import AutoTokenizer

texts = ["Tokenization is unbelievably useful.", "3.14159 + 2718", " hello", "hello"]

for name in ["bert-base-uncased", "gpt2", "t5-small"]:
    tok = AutoTokenizer.from_pretrained(name)
    print(f"== {name}  (vocab size {tok.vocab_size}) ==")
    for t in texts:
        ids = tok(t, add_special_tokens=False)["input_ids"]
        print(f"{t!r:45} -> {len(ids):2d} tokens: {tok.convert_ids_to_tokens(ids)}")

# Special tokens added by the tokenizer for a sentence pair (BERT)
bert = AutoTokenizer.from_pretrained("bert-base-uncased")
enc = bert("How are you?", "Fine, thanks.")
print(bert.convert_ids_to_tokens(enc["input_ids"]))
print(enc["token_type_ids"])   # 0 for segment A, 1 for segment B
```

You will see that `" hello"` and `"hello"` differ for the byte-level BPE tokenizer but not for BERT's, and that the number string is split into several pieces by every tokenizer. The BERT pair encoding begins with `[CLS]`, separates the segments with `[SEP]`, and the `token_type_ids` tell the model which segment each token belongs to.

## Scaling laws {#scaling-laws}

Training a frontier model costs millions of dollars, and you only get a handful of attempts. How do you choose the model size, the dataset size and the training length without running every combination? The empirical answer of the last few years is that **language-model loss is a remarkably smooth, predictable function of scale**, so you can run small experiments and extrapolate.

### Power laws: Kaplan et al. {#kaplan}

Kaplan, McCandlish, Henighan, Brown, Chess, Child, Gray, Radford, Wu and Amodei (OpenAI, 2020) trained Transformer language models over many orders of magnitude and measured the held-out cross-entropy loss $L$ (in nats per token). When one factor is the bottleneck and the others are ample, the loss follows a **power law**:

$$
L(N) = \left(\frac{N_c}{N}\right)^{\alpha_N},\qquad
L(D) = \left(\frac{D_c}{D}\right)^{\alpha_D},\qquad
L(C) = \left(\frac{C_c}{C}\right)^{\alpha_C},
$$

where $N$ is the number of non-embedding parameters, $D$ the number of training tokens, $C$ the compute, and the fitted exponents were roughly $\alpha_N\approx 0.076$, $\alpha_D\approx 0.095$ and $\alpha_C\approx 0.050$ (with $N_c\approx 8.8\times10^{13}$ and $D_c\approx 5.4\times 10^{13}$). The exponents are small, so each factor-of-ten in scale lowers the loss by a modest, constant *multiplicative* factor: $10^{-0.076}\approx 0.84$ for parameters. On log-log axes the law is a straight line, and that straightness over seven or more orders of magnitude is what makes extrapolation credible.

```chart
{
  "type": "scatter",
  "caption": "The Kaplan et al. (2020) parameter law L(N) = (N_c / N)^0.076 with N_c = 8.8e13, evaluated at six sizes (published fit constants; the data-unlimited, early-stopped regime). Both axes are logarithmic, so the law is a straight line.",
  "height": 320,
  "data": { "datasets": [ { "label": "L(N), nats per token", "showLine": true, "borderColor": "var(--c-blue)", "backgroundColor": "var(--c-blue)", "pointRadius": 4, "data": [{"x":1e6,"y":4.016},{"x":1e7,"y":3.371},{"x":1e8,"y":2.83},{"x":1e9,"y":2.376},{"x":1e10,"y":1.994},{"x":1e11,"y":1.674}] } ] },
  "options": { "scales": { "x": {"type": "logarithmic", "title": {"display": true, "text": "non-embedding parameters N"}}, "y": {"type": "logarithmic", "title": {"display": true, "text": "test loss (nats/token)"}} } }
}
```

:::intuition What a power law with a small exponent means
A power law has no characteristic scale and no plateau until you hit the irreducible entropy of the data. Going from 1B to 10B parameters buys the same *relative* improvement as 10B to 100B, but the absolute cost grows tenfold each time. Progress is therefore steady but expensive: you pay exponentially more for linear gains on a log scale. It also means that improvements to the "constant" (better data, better architectures) are worth as much as many doublings of compute.
:::

Kaplan et al. also reported that, for a fixed compute budget, loss depends only weakly on model *shape* (depth versus width), that larger models are more **sample-efficient** (they reach a given loss with fewer tokens), and, controversially, that you should spend most of a growing budget on a bigger model and train it for relatively few tokens: roughly $N_{\text{opt}}\propto C^{0.73}$ and $D_{\text{opt}}\propto C^{0.27}$. GPT-3 (175B parameters, about 300B tokens) followed that advice.

### Compute-optimal training: Chinchilla {#chinchilla}

Hoffmann, Borgeaud, Mensch and colleagues (DeepMind, 2022) redid the study with more careful experiments, in particular training each model with a learning-rate schedule matched to its token budget (Kaplan's runs used a fixed schedule, which penalizes long training). They fit three different ways and found a different answer: parameters and tokens should be scaled **in equal proportion**. Their headline result was a 70B-parameter model, Chinchilla, trained on 1.4T tokens with the same compute as the 280B-parameter Gopher trained on 300B tokens, which outperformed Gopher, GPT-3 and the 530B Megatron-Turing NLG across a wide range of evaluations (for instance 67.5% versus 60.0% five-shot accuracy on MMLU for Gopher).

Here is the parametric form they used for the third fitting approach. Treat the loss as a sum of an irreducible term and two penalties, one for having too few parameters and one for having too few tokens:

$$
\hat L(N, D) \;=\; E \;+\; \frac{A}{N^{\alpha}} \;+\; \frac{B}{D^{\beta}}.
$$

$E$ is the entropy of natural text (the best any model could do), the second term is the gap due to a finite-capacity model, and the third is the gap due to finite data. The published fit is $E=1.69$, $A=406.4$, $B=410.7$, $\alpha=0.34$, $\beta=0.28$ (loss in nats per token, for their tokenizer and data). Training compute for a dense Transformer is approximately $C\approx 6ND$ FLOPs (the 6N rule from [Week 9](week-09.html#flops-rule)). The question is: given $C$, what $N$ and $D$ minimize $\hat L$?

:::math Derivation: the compute-optimal allocation
Substitute the constraint $D = C/(6N)$ and minimize over $N$:

$$
\hat L(N) = E + A N^{-\alpha} + B\left(\frac{6N}{C}\right)^{\beta}.
$$

Setting the derivative to zero,

$$
-\alpha A N^{-\alpha-1} + \beta B \left(\frac{6}{C}\right)^{\beta} N^{\beta-1} = 0
\;\;\Longrightarrow\;\;
\alpha A N^{-\alpha} = \beta B \left(\frac{6N}{C}\right)^{\beta} = \beta B D^{-\beta}.
$$

So at the optimum the two reducible penalties are in the fixed ratio $A N^{-\alpha} : B D^{-\beta} = \beta : \alpha$. Solving for $N$,

$$
N^{\alpha+\beta} = \frac{\alpha A}{\beta B}\left(\frac{C}{6}\right)^{\beta}
\;\Longrightarrow\;
N_{\text{opt}}(C) = G\left(\frac{C}{6}\right)^{a},\quad
D_{\text{opt}}(C) = \frac{C}{6N_{\text{opt}}} = G^{-1}\left(\frac{C}{6}\right)^{b},
$$

with $G = \left(\frac{\alpha A}{\beta B}\right)^{1/(\alpha+\beta)}$, $a = \frac{\beta}{\alpha+\beta}$ and $b=\frac{\alpha}{\alpha+\beta}$, so $a+b=1$. If $\alpha\approx\beta$ then $a\approx b\approx 0.5$: **double the compute budget and scale model and data each by about $\sqrt2$**.
:::

The three approaches in the paper all landed near $a\approx b\approx 0.5$, and the practical summary became "about 20 training tokens per parameter". With the quoted parametric constants you get $a\approx 0.45$ and $b\approx 0.55$, and the implied tokens-per-parameter ratio drifts upward with $C$; the paper's own isoFLOP-based estimates are closer to a constant ratio. Later replication attempts reported that the published parametric fit is somewhat off, which is a good reminder that these are *empirical fits to a particular data and tokenizer setup*, not laws of nature. Treat "20 tokens per parameter" as a rule of thumb with real uncertainty, not a theorem.

```chart
{
  "type": "scatter",
  "caption": "IsoFLOP curves from the Chinchilla parametric loss L(N, D) = 1.69 + 406.4/N^0.34 + 410.7/D^0.28 with D = C/(6N) (published fit constants, curves computed by us). At each budget C there is a U-shaped optimum: too-small models are capacity-limited, too-large models see too few tokens.",
  "height": 360,
  "data": { "datasets": [
    { "label": "C = 1e19 FLOPs", "showLine": true, "borderColor": "var(--c-blue)", "backgroundColor": "var(--c-blue)", "data": [{"x":3.16e7,"y":3.244},{"x":1e8,"y":3.029},{"x":3.16e8,"y":2.992},{"x":1e9,"y":3.119},{"x":3.16e9,"y":3.413}] },
    { "label": "C = 1e21 FLOPs", "showLine": true, "borderColor": "var(--c-green)", "backgroundColor": "var(--c-green)", "data": [{"x":3.16e7,"y":2.948},{"x":1e8,"y":2.62},{"x":3.16e8,"y":2.428},{"x":1e9,"y":2.34},{"x":3.16e9,"y":2.338},{"x":1e10,"y":2.416},{"x":3.16e10,"y":2.578}] },
    { "label": "C = 1e23 FLOPs", "showLine": true, "borderColor": "var(--c-purple)", "backgroundColor": "var(--c-purple)", "data": [{"x":3.16e7,"y":2.866},{"x":1e8,"y":2.507},{"x":3.16e8,"y":2.273},{"x":1e9,"y":2.126},{"x":3.16e9,"y":2.042},{"x":1e10,"y":2.007},{"x":3.16e10,"y":2.014},{"x":1e11,"y":2.06},{"x":3.16e11,"y":2.149}] }
  ] },
  "options": { "scales": { "x": {"type": "logarithmic", "title": {"display": true, "text": "parameters N (tokens D = C / 6N)"}}, "y": {"min": 1.9, "max": 3.3, "title": {"display": true, "text": "predicted loss (nats/token)"}} } }
}
```

The next demo lets you move along an isoFLOP curve yourself. Try fixing the budget at the Chinchilla/Gopher level (about $5.8\times10^{23}$ FLOPs) and compare a 280B model trained on 300B tokens with a 70B model trained on 1.4T.

{{demo: week10-scaling-explorer | Scaling-law explorer: the Chinchilla loss model, isoFLOP curves and the compute-optimal point | 600}}

:::example Reading the numbers: Gopher versus Chinchilla
Both models used about $6\times10^{23}$ FLOPs. Plugging into the fitted loss, Gopher ($N=280$B, $D=300$B tokens) gives $\hat L\approx 1.99$ and Chinchilla ($N=70$B, $D=1.4$T) gives $\hat L\approx 1.94$. The 0.05-nat gap looks tiny but corresponds to several points of downstream accuracy, and Chinchilla is also four times cheaper to run at inference. Many deployed models since then have been deliberately **over-trained** relative to the compute-optimal ratio (LLaMA's 7B model saw about 1T tokens, over 100 tokens per parameter), because when a model will answer billions of queries the *inference* cost dominates and it is worth paying extra at training time to get a smaller model of the same quality.
:::

```chart
{
  "type": "bar",
  "caption": "Training tokens per parameter for well-known models, from the figures in their papers (GPT-3 175B on about 300B tokens; Gopher 280B on 300B; MT-NLG 530B on 270B; Chinchilla 70B on 1.4T; LLaMA-65B on 1.4T). The dashed idea is the Chinchilla rule of thumb of about 20.",
  "height": 300,
  "data": { "labels": ["MT-NLG 530B", "Gopher 280B", "GPT-3 175B", "Chinchilla 70B", "LLaMA 65B"], "datasets": [ { "label": "tokens per parameter", "data": [0.51, 1.07, 1.71, 20, 21.5], "backgroundColor": ["var(--c-red)", "var(--c-red)", "var(--c-amber)", "var(--c-green)", "var(--c-green)"] } ] },
  "options": { "scales": { "y": {"title": {"display": true, "text": "tokens / parameter"}} }, "plugins": {"legend": {"display": false}} }
}
```

### What scaling laws do and do not say {#scaling-limits}

Scaling laws predict **average next-token loss on a distribution like the training data**. They do not directly predict downstream task accuracy, robustness, factuality or safety, and the mapping from loss to capability is not smooth for every metric (see the next section). They assume a fixed architecture family and tokenizer, a fixed data distribution and quality, and that you are not data-constrained. When you run out of unique text, repeating data helps for a few epochs and then fades (Muennighoff et al., 2023), so data quality, deduplication, filtering and synthetic data have become first-class concerns. And they are fits, not mechanisms: several theoretical stories (data manifold dimension, quantization of skills) fit the same lines.

:::tip How practitioners use scaling laws
Train a ladder of small models (say 10M to 1B parameters) with the target recipe, fit the loss as a function of $N$ and $D$ on the small runs, then *predict* the loss and the optimal configuration of the big run before launching it. If a run deviates from its extrapolation, something is wrong: a data bug, an instability, or a learning-rate problem. Scaling-law prediction therefore doubles as a debugging tool.
:::

## Emergent abilities and their critiques {#emergent}

Looking at downstream tasks rather than loss, Wei et al. (2022) catalogued "emergent abilities": capabilities that were near chance for small models and then rose sharply above some scale, such as multi-digit arithmetic, some word-unscrambling tasks and, later, chain-of-thought reasoning. They defined an ability as emergent if it is not present in smaller models but present in larger ones, so that it cannot be predicted by extrapolating the small-model curve. This framing gave rise to the widespread idea that new skills appear "suddenly" at certain sizes.

Schaeffer, Miranda and Koyejo (NeurIPS 2023) pointed out that much of the sharpness may come from **the metric rather than the model**. Suppose the per-token accuracy $p(N)$ improves smoothly with scale. A task scored by *exact match* on a $k$-token answer succeeds only if all $k$ tokens are right, so the score is $p^k$. Even for a perfectly smooth $p(N)$, $p^k$ for large $k$ looks like a hockey stick: flat near zero for a long stretch and then a steep rise. Switch to a continuous metric (token-level accuracy, log-likelihood of the right answer) and the curve becomes smooth and predictable. They also showed that sharp transitions can be made to appear or vanish by changing the metric on the same model outputs.

```chart
{
  "type": "line",
  "caption": "Illustrative (computed, not measured): per-token accuracy p(N) = 1 - 0.5 (N / 1e7)^(-0.25) improves smoothly, but exact-match accuracy on a k-token answer, p^k, looks like a sudden emergent jump for large k. The scaling in the model is smooth; the nonlinearity is in the metric.",
  "height": 340,
  "data": { "labels": ["1e7","3e7","1e8","3e8","1e9","3e9","1e10","3e10","1e11","3e11","1e12"], "datasets": [
    { "label": "per-token accuracy (k = 1)", "data": [0.5,0.625,0.719,0.789,0.842,0.881,0.911,0.933,0.95,0.963,0.972], "borderColor": "var(--c-blue)", "tension": 0.2 },
    { "label": "exact match, k = 5", "data": [0.031,0.095,0.192,0.306,0.423,0.532,0.628,0.708,0.774,0.826,0.867], "borderColor": "var(--c-green)", "tension": 0.2 },
    { "label": "exact match, k = 10", "data": [0.001,0.009,0.037,0.094,0.179,0.283,0.394,0.502,0.599,0.682,0.752], "borderColor": "var(--c-amber)", "tension": 0.2 },
    { "label": "exact match, k = 20", "data": [0,0,0.001,0.009,0.032,0.08,0.155,0.252,0.358,0.466,0.565], "borderColor": "var(--c-red)", "tension": 0.2 }
  ] },
  "options": { "scales": { "x": {"title": {"display": true, "text": "parameters N"}}, "y": {"title": {"display": true, "text": "score"}, "min": 0, "max": 1} } }
}
```

The critique is not a complete refutation. Some points that a careful reader should hold together:

- Exact-match behavior *is* what a user experiences on a task where partial credit does not exist, so "the capability becomes usable only at large scale" is a legitimate observation even if the underlying competence grew smoothly.
- Some behaviors, in particular multi-step reasoning, may genuinely require components to be simultaneously correct, so there may be real thresholds (a composition of $k$ steps each with probability $p$ is again $p^k$). The metric argument and the compositional argument are the same arithmetic seen from two sides.
- Benchmarks are finite and noisy, and with few test items a rise from 2% to 20% can be statistical noise or a change of prompt format. Always look at confidence intervals and at how the result changes under a different prompt.
- Predicting *which* capabilities will appear, and when, remains largely unsolved. Loss extrapolates well; specific skills mostly do not.

:::warning Be careful with the word "emergent"
In a paper or a talk, ask three questions of every emergence claim: What is the metric, and is it discontinuous? How many points on the scale axis support the "jump" (often three or four model sizes)? Was the prompt format held fixed? If the answers are "exact match", "few" and "no", treat the result as suggestive at best.
:::

## Fine-tuning strategies {#finetuning}

A pretrained model is a general-purpose starting point. **Fine-tuning** continues training on a smaller, task-specific dataset so that the model specializes. The design question is *which parameters you update*, and the answers range from all of them to a few thousand.

### Full fine-tuning {#full-finetuning}

The classical recipe, used for BERT: add a small head (for classification, a linear layer on the `[CLS]` hidden state), initialize everything else from the pretrained checkpoint, and train all parameters on the labeled data with a small learning rate (typically $10^{-5}$ to $5\times10^{-5}$ for BERT-size encoders), a few epochs, linear warmup and decay. It is the strongest baseline and the simplest to reason about. It also has real costs:

- **Memory.** With mixed-precision AdamW you store, per parameter, a 16-bit weight copy (2 bytes), a 16-bit gradient (2), an FP32 master weight (4) and two FP32 Adam moments (8): about **16 bytes per parameter**, before activations. A 7B-parameter model needs roughly 112 GB just for this state, far beyond one consumer GPU.
- **Storage.** One full copy of the model per task.
- **Forgetting and instability.** A large learning rate or too many epochs on a small dataset can overwrite useful pretrained features (catastrophic forgetting) and results can vary noticeably across random seeds on small datasets. Remedies include small learning rates, layer-wise learning-rate decay, early stopping and mixing in some pretraining-style data.

:::note Fine-tuning a classifier on top of frozen features
The cheapest baseline is to freeze the encoder, extract features once, and train a linear classifier (a "linear probe"). It is fast and a useful diagnostic of representation quality, but usually falls short of full fine-tuning on text tasks. Parameter-efficient methods aim to close that gap while training a tiny fraction of the weights.
:::

### Adapters {#adapters}

Houlsby et al. (2019) inserted small bottleneck modules inside each Transformer layer and froze everything else. An adapter maps a hidden vector $\boldsymbol{h}\in\R^d$ down to a small dimension $m\ll d$, applies a nonlinearity, projects back up, and adds the result to the input:

$$
\boldsymbol{h}' = \boldsymbol{h} + \mathbf{W}_{\text{up}}\,\sigma\!\left(\mathbf{W}_{\text{down}}\boldsymbol{h}\right),\qquad \mathbf{W}_{\text{down}}\in\R^{m\times d},\;\mathbf{W}_{\text{up}}\in\R^{d\times m}.
$$

The up-projection is initialized near zero so the adapter starts as the identity. With one adapter after the attention sublayer and one after the FFN, the extra parameters per layer are about $2\cdot 2dm$. On GLUE, adapters reached within a fraction of a percent of full fine-tuning while adding only a few percent of the parameters per task. The drawback is **inference latency**: the adapters are extra sequential layers that cannot be folded into existing weights.

### LoRA: low-rank adaptation {#lora}

Hu, Shen, Wallis, Allen-Zhu, Li, Wang, Wang and Chen (2022, ICLR) made a bet about the *geometry of fine-tuning*: the change $\Delta\mathbf{W}$ that fine-tuning applies to a pretrained weight matrix has a low "intrinsic rank". Rather than learn a full $d_{\text{out}}\times d_{\text{in}}$ update, learn it as a product of two thin matrices.

{{figure: week10-lora | LoRA on one linear layer. The pretrained weight is frozen. A trainable down-projection A (r x d_in) and up-projection B (d_out x r) form a rank-r update that is added to the output, scaled by alpha / r. B starts at zero, so training starts from the pretrained function.}}

For a frozen weight $\mathbf{W}_0\in\R^{d_{\text{out}}\times d_{\text{in}}}$ the adapted layer computes

$$
\boldsymbol{h} \;=\; \mathbf{W}_0\boldsymbol{x} + \frac{\alpha}{r}\,\mathbf{B}\mathbf{A}\boldsymbol{x},\qquad
\mathbf{B}\in\R^{d_{\text{out}}\times r},\;\mathbf{A}\in\R^{r\times d_{\text{in}}},\;r\ll\min(d_{\text{in}},d_{\text{out}}).
$$

$\mathbf{A}$ is initialized with small random values and $\mathbf{B}$ with zeros, so $\Delta\mathbf{W}=\mathbf{B}\mathbf{A}=\mathbf{0}$ at the start. The scalar $\alpha$ (a constant hyperparameter) divided by $r$ keeps the update magnitude roughly independent of the rank, so that you can change $r$ without retuning the learning rate much.

:::math Derivation: parameter savings and gradients
**Counting.** Full fine-tuning of the matrix trains $d_{\text{out}}d_{\text{in}}$ parameters. LoRA trains $r\,d_{\text{in}} + d_{\text{out}}\,r = r(d_{\text{in}}+d_{\text{out}})$. For a square layer with $d_{\text{in}}=d_{\text{out}}=d$ the ratio is

$$
\frac{r(d_{\text{in}}+d_{\text{out}})}{d_{\text{in}}d_{\text{out}}} = \frac{2r}{d}.
$$

With $d=4096$ and $r=8$ this is $16/4096\approx 0.4\%$ per adapted matrix. For a model with $L$ layers adapting a set $\mathcal{S}$ of matrices, the total is $L\sum_{s\in\mathcal{S}} r\,(d_{\text{in}}^{(s)}+d_{\text{out}}^{(s)})$.

**Gradients.** Let $\boldsymbol{g}=\partial\mathcal{L}/\partial\boldsymbol{h}\in\R^{d_{\text{out}}}$ be the gradient arriving at the layer output and $s=\alpha/r$. By the chain rule

$$
\frac{\partial\mathcal{L}}{\partial\mathbf{B}} = s\,\boldsymbol{g}\,(\mathbf{A}\boldsymbol{x})^\top,\qquad
\frac{\partial\mathcal{L}}{\partial\mathbf{A}} = s\,\mathbf{B}^\top\boldsymbol{g}\,\boldsymbol{x}^\top.
$$

At initialization $\mathbf{B}=\mathbf{0}$, so $\partial\mathcal{L}/\partial\mathbf{A}=\mathbf{0}$: the first update moves only $\mathbf{B}$, and then $\mathbf{A}$ starts to receive gradient. If both were zero, neither would ever move; if both were random, the model would start away from the pretrained function. The asymmetric initialization avoids both problems.

**Rank.** $\operatorname{rank}(\mathbf{B}\mathbf{A})\le r$, so the update lives in a rank-$r$ subspace of weight space. No claim is made that the *optimal* update is exactly low rank, only that a low-rank update is enough to reach good task performance, which is an empirical finding that grows more reliable as the base model grows.
:::

LoRA has several properties that explain its popularity:

- **No inference overhead.** After training, compute $\mathbf{W}=\mathbf{W}_0+\frac{\alpha}{r}\mathbf{B}\mathbf{A}$ once and serve the merged weights; the model is architecturally identical to the original. Alternatively keep the adapter separate and swap it per request, so one base model can serve many tasks or customers with adapters of a few megabytes each.
- **Memory savings in training.** Gradients and optimizer moments exist only for the adapter parameters. The frozen base weights can even be stored in 16-bit or lower precision. (Activation memory is *not* reduced much, because backpropagation still passes through the whole network.)
- **Modularity.** Adapters can be added, removed, or combined.

In the original paper LoRA was applied to the attention projections $\mathbf{W}_q$ and $\mathbf{W}_v$. Current practice often applies it to all linear layers (attention and FFN), which typically gives better quality for the same rank at the cost of more adapter parameters. The rank is usually between 4 and 64; larger ranks help for tasks that are far from the pretraining distribution, such as learning a new domain or language.

```chart
{
  "type": "bar",
  "caption": "Trainable parameters (millions) for LoRA on a LLaMA-7B-shaped model (32 layers, d = 4096, FFN width 11008; total about 6.74B parameters), computed from r(d_in + d_out) summed over adapted matrices. 'q,v only' adapts 2 of 7 matrices per layer; 'all linear' adapts q, k, v, o, gate, up and down.",
  "height": 320,
  "data": { "labels": ["r=1", "r=2", "r=4", "r=8", "r=16", "r=32", "r=64"], "datasets": [
    { "label": "q, v only (M params)", "data": [0.52, 1.05, 2.10, 4.19, 8.39, 16.78, 33.55], "backgroundColor": "var(--c-blue)" },
    { "label": "all linear layers (M params)", "data": [2.50, 5.00, 9.99, 19.99, 39.98, 79.95, 159.9], "backgroundColor": "var(--c-purple)" }
  ] },
  "options": { "scales": { "y": {"title": {"display": true, "text": "trainable parameters (millions)"}}, "x": {"title": {"display": true, "text": "LoRA rank"}} } }
}
```

{{demo: week10-lora-calculator | LoRA calculator: rank, target modules, trainable parameters and training memory | 560}}

Here is LoRA implemented from scratch, so that you see exactly what libraries such as `peft` do.

```python:lora_from_scratch.py
import math
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Wrap a frozen nn.Linear with a trainable low-rank update (alpha/r) * B @ A."""

    def __init__(self, base: nn.Linear, r: int = 8, alpha: float = 16.0, dropout: float = 0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():          # freeze the pretrained weight (and bias)
            p.requires_grad = False
        self.r, self.scale = r, alpha / r
        # A: (r, d_in) random; B: (d_out, r) zeros -> delta W = 0 at initialization
        self.A = nn.Parameter(torch.empty(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):                          # x: (..., d_in)
        low_rank = self.drop(x) @ self.A.T @ self.B.T   # (..., r) -> (..., d_out)
        return self.base(x) + self.scale * low_rank

    @torch.no_grad()
    def merge(self) -> nn.Linear:
        """Fold the update into the base weight: W <- W + (alpha/r) B A."""
        self.base.weight += self.scale * (self.B @ self.A)
        return self.base


torch.manual_seed(0)
base = nn.Linear(64, 32)
lora = LoRALinear(base, r=4, alpha=8)
x = torch.randn(5, 64)

# 1) At init the adapted layer equals the pretrained layer exactly
print("max diff at init:", (lora(x) - base(x)).abs().max().item())

# 2) Only A and B are trainable; compare counts
n_train = sum(p.numel() for p in lora.parameters() if p.requires_grad)
n_total = sum(p.numel() for p in lora.parameters())
print(f"trainable {n_train} of {n_total}  (formula r*(din+dout) = {4 * (64 + 32)})")

# 3) One gradient step: B gets gradient, A's gradient is exactly zero because B = 0
loss = lora(x).pow(2).mean()
loss.backward()
print("|grad B| =", lora.B.grad.norm().item(), "  |grad A| =", lora.A.grad.norm().item())

# 4) After training (here: random B), merging gives the same outputs with no extra layers
with torch.no_grad():
    lora.B.normal_(0, 0.1)
    y_before = lora(x)
    merged = lora.merge()
    print("merge error:", (merged(x) - y_before).abs().max().item())
```

The printed `|grad A|` is exactly zero, reproducing the derivation above, and the merge error is at floating-point precision.

### QLoRA {#qlora}

LoRA shrinks the trainable state but the frozen base model still occupies 2 bytes per parameter in 16-bit precision: 14 GB for a 7B model and 130 GB for a 65B model. Dettmers, Pagnoni, Holtzman and Zettlemoyer (2023) combined LoRA with aggressive quantization of the *frozen* weights, which makes it possible to fine-tune a 65B model on a single 48 GB GPU. Three ingredients:

1. **4-bit NormalFloat (NF4).** Pretrained weights are approximately zero-mean normal. NF4 is a 4-bit data type whose 16 code values are placed at quantiles of a normal distribution, so each code is used about equally often; it is information-theoretically better matched to normal data than uniform 4-bit integers or floats. Weights are quantized in blocks (for example 64 values) with a per-block scale.
2. **Double quantization.** The per-block scales themselves cost memory (about 0.5 bit per parameter at 32-bit scales and block size 64). Quantizing the scales to 8 bits reduces that overhead to about 0.13 bits per parameter.
3. **Paged optimizers.** Optimizer state is allocated in unified memory that can page to CPU RAM when the GPU runs out, which smooths out memory spikes from long sequences.

During the forward and backward pass each 4-bit weight block is dequantized on the fly to 16-bit (BF16) for the matrix multiplication; the LoRA adapters are kept in 16-bit and are the only trained parameters. The gradient flows *through* the quantized weights into the adapters, but the quantized weights are never updated. The result is a model that is not exactly the original (it carries quantization error) but, in the authors' experiments, recovers the quality of 16-bit LoRA and full fine-tuning on the benchmarks they studied.

:::example Memory budget for a 7B model (rough, weights and optimizer state only)
- Full fine-tuning, mixed precision AdamW: about 16 bytes/parameter, or about 112 GB.
- LoRA with BF16 base: 2 bytes/parameter for the frozen weights (14 GB) plus adapters (40M parameters at 16 bytes each, about 0.6 GB): about 15 GB.
- QLoRA with NF4 base: about 0.5 bytes/parameter (3.5 GB, plus quantization constants) plus the same adapter state: about 4 to 5 GB.

Activations, which scale with batch size and sequence length, come on top and can dominate. Gradient checkpointing trades extra compute for lower activation memory.
:::

### Prefix tuning and prompt tuning {#prefix-prompt}

A different family leaves *all* Transformer weights alone and trains a small set of **continuous vectors** that steer the model.

- **Prefix tuning** (Li and Liang, 2021) prepends trainable key and value vectors of length $\ell$ to the attention of *every layer*. The queries of real tokens can attend to the prefix as if it were extra context. Directly optimizing the prefix was unstable, so the paper reparametrized it through a small MLP during training and kept only the resulting vectors afterwards.
- **Prompt tuning** (Lester, Al-Rfou and Constant, 2021) is the simplest version: prepend $\ell$ trainable *input embeddings* ("soft prompt") to the token embeddings and train only those, a few thousand to tens of thousands of parameters in total. Its key finding was that the gap to full fine-tuning **closes as the base model grows**; at around 10B parameters soft prompts matched fine-tuning on SuperGLUE with T5.
- **P-tuning v2** (Liu et al., 2021) uses deep prompts at every layer and is closer to prefix tuning. **(IA)$^3$** (Liu et al., 2022) learns vectors that rescale keys, values and FFN activations. **BitFit** (Ben Zaken et al., 2022) trains only bias terms.

{{figure: week10-peft-placement | Where each parameter-efficient method acts. Adapters add a bottleneck module in series; LoRA adds a parallel low-rank path to weight matrices and can be merged away; prefix tuning prepends learned key/value vectors at every layer; prompt tuning prepends learned embeddings at the input only.}}

| Method | Trainable parameters | Extra inference cost | Works well when |
| --- | --- | --- | --- |
| Full fine-tuning | 100% | none | you have data and memory; strongest baseline |
| Adapters | about 1 to 5% | extra layers (latency) | many tasks, modest-size models |
| LoRA / QLoRA | about 0.1 to 1% | none after merging | default for large decoder-only models |
| Prefix tuning | about 0.1% | longer attention context | generation tasks; low parameter budgets |
| Prompt tuning | under 0.01% | longer input | very large frozen models, many tasks served in a batch |

In practice LoRA and QLoRA became the default because they are simple, mergeable and robust to hyperparameters. Prompt-based methods are attractive when you need thousands of tiny task-specific modules, but they are more sensitive to initialization and learning rate, and underperform at small model sizes.

## Instruction tuning, in-context learning and prompting {#instruction-tuning}

### From pretrained model to instruction follower {#instruction-tuning-what}

A pretrained autoregressive model is a text *continuer*, not an assistant. Given "What is the capital of France?" it may continue with more quiz questions, because that is what such text looks like on the web. **Instruction tuning** fine-tunes the model on many (instruction, response) pairs so that following the instruction becomes the most likely continuation.

The first demonstrations of the idea were multitask: FLAN (Wei et al., 2022) and T0 (Sanh et al., 2022) rewrote dozens of existing NLP datasets as natural-language instructions and fine-tuned on the mixture. The surprising result was **zero-shot generalization to held-out task types**: a model tuned on many instruction-formatted tasks did better on unseen tasks than the same model without that tuning, and scaling the number of tasks and the model size both helped (Chung et al., 2022). A second line collected open-ended prompts and high-quality human-written answers. InstructGPT (Ouyang et al., 2022) used about 13,000 labeler-written demonstrations; Self-Instruct (Wang et al., 2023) had a model generate its own instructions and answers from a small seed set, which inspired cheap imitation datasets such as Alpaca. LIMA (Zhou et al., 2023) showed that about a thousand carefully curated examples were enough to produce a strong chat model from a strong base, supporting the **superficial alignment hypothesis**: most knowledge and ability come from pretraining, and instruction tuning mainly teaches the format and style of the interaction. Quality, diversity and correctness of the demonstrations matter far more than their count.

Mechanically, supervised fine-tuning (SFT) is the same next-token cross-entropy as pretraining with two changes. The input is rendered through a **chat template** that marks roles, and the loss is computed **only on the response tokens**, not on the prompt:

$$
\mathcal{L}_{\text{SFT}}(\theta) = -\sum_{(x,y)\in\mathcal{D}}\ \sum_{t=1}^{|y|}\log \pi_\theta\!\left(y_t \mid x, y_{<t}\right).
$$

:::warning Chat templates are part of the model
Each model family was tuned with specific role markers and special tokens. Formatting a conversation differently from the way the model was trained is a common cause of degraded or strange output. Use `tokenizer.apply_chat_template(messages, ...)` from Hugging Face rather than hand-built strings, and make sure the end-of-turn token is included in the training targets so the model learns to stop.
:::

### In-context learning {#icl}

GPT-3 made an observation with large consequences: without any weight update, a model can perform a new task if you show it the task in its input. With **zero-shot** prompting you only describe the task; with **one-shot** and **few-shot** prompting you include $k$ solved examples before the query:

```text
Translate English to French.
sea otter => loutre de mer
cheese => fromage
plush giraffe => girafe en peluche
mint => 
```

The model completes the pattern. This is **in-context learning** (ICL): the "learning" happens in the forward pass, as a function of the activations, with the weights fixed. A few robust empirical facts:

- Few-shot performance improves with model size much faster than zero-shot (Brown et al., 2020), and more examples help up to the context limit, with diminishing returns.
- Performance depends strongly on the *format* and the *label space* shown in the examples, and often surprisingly little on whether the example labels are correct (Min et al., 2022). The demonstrations tell the model what *kind* of task it is, rather than teaching it a new input-output rule.
- Results are sensitive to the order and selection of the examples (Lu et al., 2022). Choosing demonstrations similar to the query ("retrieved few-shot") usually helps.

Why does ICL work? Several non-exclusive hypotheses are being studied: the model infers a latent task or concept from the examples (an implicit Bayesian view, Xie et al., 2022); certain attention circuits, called **induction heads**, copy and complete repeated patterns and emerge during training (Olsson et al., 2022); and, for linear-attention toy models, a forward pass can implement a step of gradient descent on the in-context examples (von Oswald et al., 2023). None of these is a complete account of large models, so treat them as partial explanations.

### Prompting in practice {#prompting}

Prompt engineering is the practice of writing the input so that the model's most probable continuation is the output you want. The durable principles are mundane but effective:

1. **Be specific about the task, format and constraints.** State the role, the input, the desired output structure (JSON schema, a single word, bullet points) and what to do when the input is invalid.
2. **Separate instructions from data** with delimiters, and be careful with untrusted text (see prompt injection in [Failure modes](#failure-modes)).
3. **Show examples** when the format is unusual; choose diverse and representative ones.
4. **Give the model room to think** when the task needs multi-step reasoning (next section), and ask for the final answer in a fixed, parseable format.
5. **Test the prompt as you would code**: build a small evaluation set, version your prompts, and measure. Prompts that work on three examples often fail on a hundred.

### Chain-of-thought and self-consistency {#chain-of-thought}

Wei et al. (2022) found that for tasks needing several reasoning steps (grade-school math, symbolic manipulation), prompting with examples whose answers include **intermediate reasoning steps** sharply improved large models' accuracy compared with answer-only examples. Kojima et al. (2022) showed that you can often elicit this without examples by adding a trigger phrase such as "Let's think step by step" (zero-shot chain-of-thought). Self-consistency (Wang et al., 2023) goes further: sample several reasoning paths at a nonzero temperature and take the *majority vote* over the final answers, marginalizing over reasoning paths.

Why would writing out steps help? A Transformer has a fixed amount of computation per generated token (a fixed number of layers). Generating intermediate tokens gives the model a **scratchpad** with more sequential computation, and each step is an easier prediction conditional on the previous ones. The tokens themselves carry state across steps, as a recurrent state would.

```text
Q: A cafe sold 23 muffins in the morning and 3 times as many in the afternoon.
   How many muffins in total?
A: Morning: 23. Afternoon: 3 * 23 = 69. Total: 23 + 69 = 92. The answer is 92.
```

:::warning Chain-of-thought is not a faithful explanation
The visible reasoning is generated text, and studies have found that it does not always reflect the factors that actually determined the answer (for example, Turpin et al., 2023, showed that models can be swayed by biasing features in the prompt without mentioning them in their rationale). Use chain-of-thought as a performance technique, not as a guarantee of how the model "really" decided. Also note that CoT helps mainly in large models; on small models it often hurts.
:::

## Alignment: SFT, reward modeling, RLHF and DPO {#alignment}

Instruction tuning on demonstrations has a ceiling: writing a perfect answer is hard, while *comparing* two answers is much easier. Alignment methods exploit that asymmetry. The recipe of InstructGPT (Ouyang et al., 2022), built on earlier work on learning from human preferences (Christiano et al., 2017; Stiennon et al., 2020), has three stages after pretraining.

{{figure: week10-rlhf-pipeline | The InstructGPT-style pipeline. Stage 1 supervised fine-tuning on demonstrations; stage 2 a reward model trained on human rankings of sampled responses; stage 3 policy optimization against the reward with a KL penalty to the SFT model. DPO collapses stages 2 and 3 into one supervised-style loss.}}

### Reward modeling {#reward-model}

Collect prompts $x$, sample several responses from the SFT model, and ask labelers to rank them. Each ranking yields preference pairs $(x, y_w, y_l)$ with a winner $y_w$ and loser $y_l$. The **Bradley-Terry model** says the probability that $y_w$ is preferred is a logistic function of a scalar reward difference:

$$
P(y_w \succ y_l \mid x) = \sigma\!\big(r_\phi(x,y_w) - r_\phi(x,y_l)\big),\qquad
\mathcal{L}_{\text{RM}}(\phi) = -\,\E_{(x,y_w,y_l)}\Big[\log\sigma\big(r_\phi(x,y_w)-r_\phi(x,y_l)\big)\Big].
$$

The reward model $r_\phi$ is usually the SFT Transformer with the vocabulary head replaced by a linear layer that outputs one scalar from the final token's hidden state. Only reward *differences* matter, so the absolute scale is arbitrary.

### RLHF with a KL penalty {#rlhf}

Treat the language model as a policy $\pi_\theta(y\mid x)$ and maximize the learned reward, but keep the policy close to a reference (the SFT model $\pi_{\text{ref}}$) so that it does not drift into text that fools the reward model:

$$
\max_{\pi_\theta}\ \E_{x\sim\mathcal{D},\,y\sim\pi_\theta(\cdot\mid x)}\big[r_\phi(x,y)\big]\;-\;\beta\,\KL\!\big(\pi_\theta(\cdot\mid x)\,\|\,\pi_{\text{ref}}(\cdot\mid x)\big).
$$

This is optimized with a policy-gradient method, typically PPO (Schulman et al., 2017), with the KL term applied per token and a learned value function to reduce variance. The setup is heavy: you keep the policy, a reference model, a reward model and a value model in memory, and training can be unstable. The KL coefficient $\beta$ is the central knob. Too small and the policy exploits flaws in the reward model (**reward hacking**: verbose, sycophantic or formulaic text that scores well but is not better); too large and it barely moves from SFT. Gao, Schulman and Hilton (2023) measured how the *true* reward first rises and then falls as you optimize harder against a proxy reward model, and how the turnaround point scales with reward-model size.

InstructGPT's headline result was that human labelers preferred outputs from the 1.3B-parameter InstructGPT model to those of the 175B GPT-3, showing that alignment can matter more than raw scale for perceived usefulness. The approach also reduced some toxic outputs and hallucinations on their evaluations, though not entirely.

### Direct Preference Optimization {#dpo}

Rafailov, Sharma, Mitchell, Ermon, Manning and Finn (2023) noticed that the KL-regularized objective has a closed-form solution, and that you can substitute it back into the Bradley-Terry loss so that the reward model disappears.

:::math Derivation sketch of the DPO loss
**Step 1: the optimal policy.** For a fixed prompt $x$, the objective $\E_{y\sim\pi}[r(x,y)] - \beta\KL(\pi\,\|\,\pi_{\text{ref}})$ is maximized over distributions $\pi$ by the reward-tilted reference:

$$
\pi^*(y\mid x) = \frac{1}{Z(x)}\,\pi_{\text{ref}}(y\mid x)\exp\!\Big(\tfrac{1}{\beta}r(x,y)\Big),\qquad Z(x)=\sum_y \pi_{\text{ref}}(y\mid x)\exp\!\Big(\tfrac{1}{\beta}r(x,y)\Big).
$$

(Check: write the objective as $-\beta\,\KL(\pi\,\|\,\pi^*) + \beta\log Z(x)$, which is maximized when $\pi=\pi^*$.)

**Step 2: invert for the reward.** Take logs and rearrange:

$$
r(x,y) = \beta\log\frac{\pi^*(y\mid x)}{\pi_{\text{ref}}(y\mid x)} + \beta\log Z(x).
$$

So *any* reward function can be written as a log-ratio between a policy and the reference, up to a prompt-dependent constant.

**Step 3: plug into Bradley-Terry.** Only the difference of rewards for the same prompt matters, and the intractable $\beta\log Z(x)$ cancels:

$$
P(y_w\succ y_l\mid x) = \sigma\!\left(\beta\log\frac{\pi^*(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)} - \beta\log\frac{\pi^*(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}\right).
$$

**Step 4: fit the policy by maximum likelihood.** Replace $\pi^*$ by the trainable $\pi_\theta$ and minimize the negative log-likelihood on the preference data:

$$
\mathcal{L}_{\text{DPO}}(\theta) = -\,\E_{(x,y_w,y_l)}\left[\log\sigma\!\left(\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)} - \beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}\right)\right].
$$
:::

Define the *implicit reward* $\hat r_\theta(x,y)=\beta\log\frac{\pi_\theta(y\mid x)}{\pi_{\text{ref}}(y\mid x)}$. The gradient of the loss increases $\hat r_\theta(x,y_w)$, decreases $\hat r_\theta(x,y_l)$, and weights each pair by $\sigma(\hat r_\theta(x,y_l)-\hat r_\theta(x,y_w))$, so pairs that the implicit reward currently ranks *wrongly* get the largest update. There is no sampling from the policy, no reward model and no value function: it is a classification-style loss over a fixed dataset, computed from four sequence log-probabilities per pair.

```python:dpo_loss.py
import torch
import torch.nn.functional as F


def sequence_logprob(logits, labels, mask):
    """Sum of token log-probs of `labels` under `logits`.
    logits: (B, T, V)  labels: (B, T)  mask: (B, T) 1 on response tokens, 0 on prompt/pad."""
    logp = torch.log_softmax(logits, dim=-1)                       # (B, T, V)
    tok = torch.gather(logp, 2, labels.unsqueeze(-1)).squeeze(-1)  # (B, T)
    return (tok * mask).sum(dim=-1)                                # (B,)


def dpo_loss(pi_w, pi_l, ref_w, ref_l, beta=0.1):
    """Each argument: (B,) summed log-probs of the chosen (w) / rejected (l) response
    under the policy (pi) and the frozen reference (ref)."""
    margin = beta * ((pi_w - ref_w) - (pi_l - ref_l))              # implicit reward difference
    loss = -F.logsigmoid(margin).mean()
    acc = (margin > 0).float().mean()                              # preference accuracy
    return loss, acc


# toy check: if the policy equals the reference the margin is 0 and loss = log 2
B = 4
a = torch.randn(B)
loss, acc = dpo_loss(a, a - 1.0, a, a - 1.0)
print(loss.item(), torch.log(torch.tensor(2.0)).item())            # both 0.6931

# making the chosen response more likely than under the reference lowers the loss
loss2, acc2 = dpo_loss(a + 2.0, a - 1.0, a, a - 1.0)
print(loss2.item(), acc2.item())                                    # loss < 0.693, acc = 1.0

# reward-model loss for comparison: -log sigmoid(r_w - r_l)
r_w, r_l = torch.tensor([1.2, 0.3]), torch.tensor([0.4, 0.9])
print((-F.logsigmoid(r_w - r_l)).mean().item())
```

DPO is not magic. It optimizes the same objective *if* the data come from (or near) the reference policy and the Bradley-Terry assumption holds; it can reduce the likelihood of both chosen and rejected responses; and online RL with fresh samples can still outperform it. A family of variants (IPO, KTO, ORPO, SimPO and others) tweak the loss to fix specific issues, and recent reasoning-oriented models are often trained with RL against *verifiable* rewards (exact answers, unit tests) rather than learned preference models. Other sources of preference signal include AI feedback (RLAIF; Constitutional AI, Bai et al., 2022). We cover the RL machinery itself in [Week 15](week-15.html).

:::note Alignment is not truth-seeking
Preference training optimizes what human or AI raters *prefer*, which includes fluency, confidence and agreement with the user. That is why aligned models can be sycophantic and can assert false things in an authoritative tone. Alignment changes behavior and style; it does not add knowledge the base model lacks.
:::

## Decoding strategies {#decoding}

A trained language model gives you a distribution $p(x_t\mid x_{<t})$ at each step. **Decoding** is the algorithm that turns those distributions into a text. The choice is not a detail: the same model can produce dull, repetitive text or creative, error-prone text depending only on how you sample.

### Greedy and beam search {#greedy-beam}

**Greedy decoding** picks $x_t=\argmax_v p(v\mid x_{<t})$ at each step. It is deterministic and fast, but myopic: the best token now may lead to a worse sequence overall. **Beam search**, which you met for translation in [Week 7](week-07.html), keeps the $B$ highest-scoring partial sequences (by total log-probability, usually length-normalized) and extends all of them. Beam search finds sequences of higher probability than greedy search, which is what you want when there is a *right answer* close to the mode, as in translation, summarization or speech recognition.

For open-ended generation, high probability is the wrong target. Holtzman, Buys, Du, Forbes and Choi (2020) showed that the most likely continuations of a large language model are bland and repetitive ("text degeneration"), while human text is *not* the mode of the model distribution: it contains many individually low-probability words. Greedy and beam outputs often fall into loops, which is a sign that the model's probability mass concentrates on repetition once it has begun.

### Temperature {#temperature}

Sampling draws $x_t\sim p(\cdot\mid x_{<t})$. To control randomness, rescale the logits $z_v$ by a **temperature** $T>0$ before the softmax:

$$
p_T(v) = \frac{\exp(z_v/T)}{\sum_{u}\exp(z_u/T)}.
$$

As $T\to 0$ the distribution collapses onto the argmax (greedy); $T=1$ is the model's own distribution; as $T\to\infty$ it flattens to uniform. Temperature does not change the *ranking* of tokens, only how much mass the top tokens get. It is the same temperature that appeared in attention and in the InfoNCE loss ([Week 8](week-08.html)), and the same trade-off: low temperature means confident and repetitive, high temperature means diverse and error-prone. Two useful facts: the entropy of $p_T$ increases monotonically with $T$, and for a distribution with a long tail of unlikely tokens even a moderately high $T$ makes a "bad" token appear within a few hundred steps, because the tail's *total* mass is large.

```chart
{
  "type": "bar",
  "caption": "One next-token distribution (illustrative logits 4.0, 3.4, 2.9, 2.2, 1.5, 1.0, 0.3, -0.2, -0.8, -1.5) at three temperatures, computed with a softmax. At T = 0.5 the top token has 69% of the mass; at T = 2 it has 27%.",
  "height": 320,
  "data": { "labels": ["1","2","3","4","5","6","7","8","9","10"], "datasets": [
    { "label": "T = 0.5", "data": [0.69,0.208,0.076,0.019,0.005,0.002,0,0,0,0], "backgroundColor": "var(--c-blue)" },
    { "label": "T = 1", "data": [0.448,0.246,0.149,0.074,0.037,0.022,0.011,0.007,0.004,0.002], "backgroundColor": "var(--c-green)" },
    { "label": "T = 2", "data": [0.273,0.202,0.157,0.111,0.078,0.061,0.043,0.033,0.025,0.017], "backgroundColor": "var(--c-amber)" }
  ] },
  "options": { "scales": { "x": {"title": {"display": true, "text": "token rank"}}, "y": {"title": {"display": true, "text": "probability"}} } }
}
```

### Top-k and top-p (nucleus) sampling {#top-k-top-p}

Pure sampling at $T=1$ has a flaw: the tail. A vocabulary of 50,000 tokens means that thousands of nonsense tokens each have tiny probability but jointly carry a few percent of the mass, so over a long generation the model will sample some of them. Truncation methods cut the tail before sampling.

- **Top-$k$** (Fan, Lewis and Dauphin, 2018): keep only the $k$ most probable tokens, renormalize, sample. The weakness is that $k$ is fixed regardless of context: after "The capital of France is" one token deserves nearly all the mass, so $k=50$ admits junk; after "My favorite" many continuations are reasonable, so $k=5$ is too restrictive.
- **Top-$p$ / nucleus sampling** (Holtzman et al., 2020): keep the smallest set $V_p$ of most probable tokens whose cumulative probability reaches $p$ (for example 0.9), renormalize, sample. The set size adapts to the model's uncertainty: small when the model is confident, large when it is not.
- **Min-$p$** keeps tokens whose probability is at least a fraction of the top token's probability; **typical sampling** keeps tokens whose surprise is close to the distribution's entropy. These are refinements in the same spirit.

Common defaults are $T\in[0.2,1]$ with $p\approx 0.9$ to $0.95$. Use $T$ near 0 (or greedy) for tasks with a single right answer such as extraction, code and math, and higher temperature for brainstorming and creative text. For self-consistency ([above](#chain-of-thought)) you *need* a nonzero temperature so that the sampled reasoning paths differ.

{{figure: week10-sampling | Truncation on a concrete distribution. Top-k keeps a fixed number of tokens; top-p keeps the smallest prefix whose cumulative mass (bottom row) reaches p, so its size depends on how peaked the distribution is.}}

{{demo: week10-sampling-playground | Sampling playground: temperature, top-k, top-p on a fixed logit vector | 620}}

```python:sampling.py
import numpy as np

def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()

def sample(logits, temperature=1.0, top_k=0, top_p=1.0, rng=None):
    """Sample one token id from logits with temperature, top-k and top-p applied in that order."""
    rng = rng or np.random.default_rng()
    if temperature <= 1e-8:                       # greedy limit
        return int(np.argmax(logits))
    p = softmax(logits / temperature)
    order = np.argsort(-p)                        # token ids sorted by decreasing probability
    keep = np.ones_like(p, dtype=bool)
    if top_k > 0:
        keep[order[top_k:]] = False               # drop everything after the k-th
    if top_p < 1.0:
        cum = np.cumsum(p[order])
        # smallest prefix with cumulative mass >= top_p: drop tokens whose *preceding* mass already >= top_p
        cutoff = np.searchsorted(cum, top_p, side="left") + 1
        keep[order[cutoff:]] = False
    q = np.where(keep, p, 0.0)
    q /= q.sum()                                  # renormalize over survivors
    return int(rng.choice(len(q), p=q))

logits = np.array([4.0, 3.4, 2.9, 2.2, 1.5, 1.0, 0.3, -0.2, -0.8, -1.5])
rng = np.random.default_rng(0)
for cfg in [dict(temperature=1.0), dict(temperature=1.0, top_k=3),
            dict(temperature=1.0, top_p=0.9), dict(temperature=0.5, top_p=0.9)]:
    draws = [sample(logits, rng=rng, **cfg) for _ in range(20000)]
    freq = np.bincount(draws, minlength=len(logits)) / len(draws)
    print(cfg, np.round(freq, 3))
```

In Hugging Face the same knobs are arguments to `model.generate(...)`: `do_sample=True, temperature=0.7, top_k=50, top_p=0.9`, or `num_beams=4` for beam search, and `max_new_tokens` to bound the length. A very common mistake is to set `temperature` or `top_p` *without* `do_sample=True`, in which case they are ignored and greedy decoding is used.

## Retrieval-augmented generation {#rag}

A language model's knowledge is frozen into its weights at training time. That causes three problems: the knowledge goes stale, the model cannot cite where a claim came from, and rarely seen facts are stored unreliably. **Retrieval-augmented generation** (RAG; Lewis et al., 2020) addresses all three by fetching relevant documents at query time and placing them in the prompt, so that the model can *read* the answer instead of *remembering* it.

{{figure: week10-rag | A typical RAG pipeline. Offline, the corpus is chunked and embedded into a vector index. At query time the question is embedded, the nearest chunks are retrieved, and the generator receives instructions, the retrieved passages and the question.}}

### The components {#rag-components}

1. **Chunking.** Split documents into passages (a few hundred tokens, often with overlap and with titles or section headers attached). Chunks that are too large dilute the relevant sentence; too small and they lose context.
2. **Embedding and indexing.** Encode each chunk with an embedding model, usually a BERT-style encoder trained contrastively (the same InfoNCE idea as in [Week 8](week-08.html), with queries and relevant passages as positive pairs; DPR, Karpukhin et al., 2020). Store vectors in an approximate nearest-neighbor index (HNSW or IVF structures, as in FAISS) so search scales to millions of chunks.
3. **Retrieval.** Embed the query, fetch the top-$k$ chunks by cosine or inner-product similarity. Many systems combine dense retrieval with lexical retrieval (BM25) in a *hybrid* search, because dense models are weak on rare names and exact identifiers while BM25 is weak on paraphrase.
4. **Reranking (optional).** A cross-encoder that reads query and passage *together* rescores the top 50 to 100 candidates much more accurately than the bi-encoder, at higher cost per pair.
5. **Generation.** Insert the passages into the prompt with instructions such as "answer only from the sources; say you don't know otherwise; cite the source ids".

```python:tiny_rag.py
import numpy as np
from collections import Counter

docs = [
    "The Eiffel Tower is 330 metres tall and stands in Paris.",
    "LoRA freezes pretrained weights and trains a low-rank update.",
    "Chinchilla suggests roughly twenty training tokens per parameter.",
    "Beam search keeps the B best partial sequences at each step.",
]

# Build a bag-of-words vocabulary and L2-normalized count vectors (a stand-in for a learned embedding)
tok = lambda s: [w.strip(".,?").lower() for w in s.split()]
vocab = sorted({w for d in docs for w in tok(d)})
index = {w: i for i, w in enumerate(vocab)}

def embed(text):
    v = np.zeros(len(vocab))
    for w, c in Counter(tok(text)).items():
        if w in index:
            v[index[w]] = c
    n = np.linalg.norm(v)
    return v / n if n > 0 else v

E = np.stack([embed(d) for d in docs])             # (n_docs, vocab)

def retrieve(query, k=2):
    sims = E @ embed(query)                        # cosine similarity (rows are unit norm)
    top = np.argsort(-sims)[:k]
    return [(int(i), float(sims[i])) for i in top]

query = "How many tokens per parameter should I train on?"
hits = retrieve(query)
print(hits)

# Assemble the grounded prompt that would be sent to the LLM
context = "\n".join(f"[{i}] {docs[i]}" for i, _ in hits)
prompt = ("Answer using only the sources below and cite them like [0]. "
          "If the sources do not contain the answer, say you do not know.\n\n"
          f"Sources:\n{context}\n\nQuestion: {query}\nAnswer:")
print(prompt)
```

### Why RAG fails {#rag-failures}

RAG moves the failure surface; it does not remove it. Evaluate the retriever and the generator *separately*: recall@$k$ and MRR for retrieval, and, for generation, **faithfulness** (is every claim supported by the retrieved text?) and **answer relevance**. Common failure modes:

- **Retrieval misses.** The relevant chunk is not in the top-$k$ because of vocabulary mismatch, chunking that split the answer in half, or a query that needs multi-hop reasoning. The generator then hallucinates or answers from its parametric memory.
- **Distractors.** Retrieved passages that are topical but wrong can mislead the model more than no passage at all.
- **Position bias.** Models use information in the middle of a long context worse than information at the beginning or end (Liu et al., 2024, "Lost in the Middle"), so stuffing 30 passages is not better than 5 good ones.
- **Ignoring or overriding context.** The model may prefer its memorized answer to the retrieved one, or blend the two.
- **Security.** Retrieved text is untrusted input; documents can contain instructions ("indirect prompt injection").

When to use RAG versus fine-tuning? Use RAG for *knowledge* that changes, that must be attributable, or that is too large to memorize. Use fine-tuning for *behavior*: style, format, domain-specific skills and tool use. They combine well.

## Evaluating language models {#evaluation}

### Perplexity {#perplexity}

The intrinsic metric of a language model is its held-out cross-entropy. **Perplexity** is its exponential:

$$
\text{PPL} = \exp\!\Big(-\frac{1}{T}\sum_{t=1}^{T}\log p_\theta(x_t\mid x_{<t})\Big) = \exp\big(\mathcal{L}_{\text{AR}}\big).
$$

Intuitively, a perplexity of 20 means the model is on average as uncertain as if it were choosing uniformly among 20 equally likely tokens. A uniform model over a vocabulary of size $V$ has perplexity $V$. Three cautions. (i) Perplexity is measured *per token*, so models with different tokenizers are **not comparable**; use bits per byte or bits per character, which normalize by the raw text length. (ii) Perplexity requires a proper left-to-right probability model, so it is not defined for BERT-style masked models without pseudo-likelihood tricks. (iii) Lower perplexity does not guarantee better behavior on tasks, especially after instruction tuning and RLHF, which deliberately move the model away from the data distribution; it is nonetheless the metric that scaling laws are stated in, and it is a good sanity check for pretraining and for continued-pretraining or quantization (a few percent change is a warning sign).

```python:perplexity.py
# Perplexity of GPT-2 on a text with a sliding window (Hugging Face API).
import math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2").to(device).eval()

text = "Language models assign probabilities to sequences of tokens. " * 50
ids = tok(text, return_tensors="pt").input_ids.to(device)        # (1, T)

max_len, stride = 256, 128          # window and hop; the overlap gives later tokens more context
nll_sum, n_tokens, prev_end = 0.0, 0, 0
for begin in range(0, ids.size(1), stride):
    end = min(begin + max_len, ids.size(1))
    target_len = end - prev_end                                  # only score tokens not yet scored
    window = ids[:, begin:end]
    labels = window.clone()
    labels[:, :-target_len] = -100                               # -100 = ignored by the loss
    with torch.no_grad():
        out = model(window, labels=labels)                       # out.loss = mean NLL over scored tokens
    n_scored = (labels[:, 1:] != -100).sum().item()              # labels are shifted inside the model
    nll_sum += out.loss.item() * n_scored
    n_tokens += n_scored
    prev_end = end
    if end == ids.size(1):
        break

print(f"tokens scored: {n_tokens}, perplexity: {math.exp(nll_sum / n_tokens):.2f}")
```

### Benchmarks {#benchmarks}

Task benchmarks test capabilities that perplexity cannot. Representative ones: GLUE and SuperGLUE (language understanding, classic fine-tuning era); MMLU (Hendrycks et al., 2021; 57 subjects of multiple-choice questions from elementary to professional level); BIG-bench (Srivastava et al., 2022; hundreds of diverse tasks); HELM (Liang et al., 2022; a holistic framework that reports accuracy together with calibration, robustness, fairness, toxicity and efficiency); GSM8K (grade-school math word problems); HumanEval (Chen et al., 2021; Python function synthesis checked by unit tests); TruthfulQA (Lin et al., 2022; questions where imitating common human misconceptions yields false answers). Pairwise human preference platforms (Chatbot Arena, Chiang et al., 2024) fit a Bradley-Terry-type model to millions of votes and report a rating per model.

For code, accuracy is reported as **pass@$k$**: the probability that at least one of $k$ sampled programs passes the tests. Chen et al. (2021) give an unbiased estimator: generate $n\ge k$ samples per problem, count the $c$ that pass, and compute

$$
\text{pass@}k = \E_{\text{problems}}\left[1-\frac{\binom{n-c}{k}}{\binom{n}{k}}\right].
$$

```python:pass_at_k.py
import numpy as np

def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k from n samples of which c are correct (numerically stable product form)."""
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

print(pass_at_k(200, 20, 1))    # 0.10  (= c / n)
print(pass_at_k(200, 20, 10))   # about 0.66
print(pass_at_k(200, 20, 100))  # about 1.0
```

**LLM-as-judge.** Because open-ended answers are hard to score automatically, a strong model is often used to grade or compare outputs (Zheng et al., 2023). It correlates well with human preferences on average but has measurable biases: it prefers the response in a particular position, prefers longer answers, and tends to favor its own style. Randomize the order, control for length, and spot-check against humans.

### Contamination and the validity of benchmarks {#contamination}

A benchmark measures generalization only if the model has not seen the test items. Web-scale corpora contain benchmark questions, their answers, discussions of them and near-duplicates. **Contamination** inflates scores and is hard to rule out, especially for closed models whose training data are unknown. GPT-3's paper reported an overlap analysis using 13-gram matches and found that some benchmarks were affected; since then contamination checks (n-gram overlap, canary strings, comparing the model's likelihood on the original and a perturbed or reordered test set) have become standard, but none is perfect. Related problems are **benchmark saturation** (scores approach the ceiling and stop discriminating), **overfitting by repeated evaluation** on the same public test set, **prompt sensitivity** (accuracy differing by many points under a different answer format), and the gap between multiple-choice accuracy and real-world usefulness. Defenses include held-out private test sets, benchmarks refreshed with post-cutoff material, reporting uncertainty over prompt variants, and evaluating on your own task distribution.

:::tip The minimum honest evaluation report
State the exact prompt and decoding settings, the number of shots, the metric and its normalization, the number of test items with a confidence interval, whether contamination was checked, and compare against a strong baseline (including a smaller model or a simple heuristic).
:::

## Hallucination and other failure modes {#failure-modes}

### Hallucination {#hallucination}

A **hallucination** is fluent output that is not supported by the input (*intrinsic*, contradicting the provided source) or not supported by the world (*extrinsic*, an invented fact, citation or API). The term is imperfect, and some authors prefer "confabulation", but it names a real and central failure.

Why does it happen? The training objective rewards *plausible continuation*, not truth: there is no term that penalizes a fluent falsehood more than a fluent truth unless the data happen to. When the model's knowledge is thin (rare entities in the "long tail"), the most plausible continuation of "The author of this obscure paper is" is still a well-formed name. Once a wrong token is generated it becomes part of the context, and later tokens condition on it (**error snowballing**). Preference training can make this worse by rewarding confident, helpful-sounding answers over "I don't know". On the positive side, models carry some usable signal about their own uncertainty: Kadavath et al. (2022) found that large models' stated probabilities of correctness are reasonably calibrated on multiple-choice questions, and that they can be trained to predict whether they know an answer.

Mitigations form a stack: retrieval and grounding with citations ([RAG](#rag)); instructing and training the model to abstain; sampling several answers and checking agreement (self-consistency); verifying claims with tools such as code execution or a calculator or search; constraining output formats; and human review for high-stakes content. None gives a guarantee.

### Other failure modes you should know {#other-failures}

- **Prompt injection and jailbreaks.** Instructions hidden in user input or retrieved documents can override the developer's instructions, because the model has no hard boundary between "code" and "data". Never give a model with access to tools more authority than the least trusted text in its context.
- **Memorization and privacy.** LLMs can reproduce verbatim training sequences, particularly duplicated ones, which can include personal data (Carlini et al., 2021). Deduplication reduces but does not eliminate this.
- **The reversal curse.** A model trained on "A is B" may fail to answer "Who is B?" (Berglund et al., 2023), illustrating that learned facts are directional associations, not an indexed database.
- **Brittleness in counting, arithmetic and character-level tasks**, partly due to tokenization ([above](#tokenization)).
- **Sycophancy**: agreeing with a user's stated opinion or with a false premise in the question.
- **Bias and toxicity** inherited from data and sometimes amplified; evaluate on representative subgroups, not just the average ([Week 17](week-17.html)).
- **Context-window limits and degradation**: nominal context length is not usable context length; accuracy on retrieval of a fact from the middle of a long prompt can drop sharply.
- **Non-determinism**: even at temperature 0, batched GPU inference can yield slightly different outputs across runs; do not build pipelines that assume bit-exact repeatability.

## Lab: fine-tuning BERT, then LoRA on a small causal LM {#lab}

This lab uses Hugging Face `transformers`, `datasets` and `peft`. A free Colab GPU is enough. Install once:

```bash
pip install -U transformers datasets peft accelerate
# optional, for the QLoRA variant (needs a CUDA GPU):
pip install -U bitsandbytes
```

### Part A: fine-tune BERT for sentiment classification {#lab-bert}

We fine-tune `bert-base-uncased` on SST-2 (binary movie-review sentiment, part of GLUE). Expect roughly 92% validation accuracy after two epochs. The pretrained encoder is reused; the classification head (a linear layer on the pooled `[CLS]` representation) is new and randomly initialized, which is why Transformers prints a warning that some weights are newly initialized.

```python:lab_bert_sst2.py
import numpy as np
import torch
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          DataCollatorWithPadding, TrainingArguments, Trainer, set_seed)

set_seed(0)
raw = load_dataset("glue", "sst2")                 # train / validation / test (test labels are hidden)
print(raw["train"][0])                             # {'sentence': ..., 'label': 0 or 1, 'idx': 0}

ckpt = "bert-base-uncased"
tok = AutoTokenizer.from_pretrained(ckpt)

def tokenize(batch):
    # no padding here: the collator pads each batch to its own longest sequence (faster)
    return tok(batch["sentence"], truncation=True, max_length=128)

ds = raw.map(tokenize, batched=True, remove_columns=["sentence", "idx"])
ds = ds.rename_column("label", "labels")           # the Trainer expects the key 'labels'
collator = DataCollatorWithPadding(tokenizer=tok)  # pads input_ids, attention_mask, token_type_ids

model = AutoModelForSequenceClassification.from_pretrained(
    ckpt, num_labels=2, id2label={0: "negative", 1: "positive"}, label2id={"negative": 0, "positive": 1})

def compute_metrics(eval_pred):
    logits, labels = eval_pred                      # logits: (N, 2), labels: (N,)
    return {"accuracy": float((logits.argmax(-1) == labels).mean())}

args = TrainingArguments(
    output_dir="bert-sst2",
    learning_rate=2e-5,                             # small LR: we are adjusting, not relearning
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    num_train_epochs=2,
    weight_decay=0.01,
    warmup_ratio=0.1,                               # linear warmup then linear decay
    eval_strategy="epoch",                          # called evaluation_strategy in older versions
    save_strategy="no",
    fp16=torch.cuda.is_available(),
    report_to="none",
    seed=0,
)
trainer = Trainer(model=model, args=args, train_dataset=ds["train"], eval_dataset=ds["validation"],
                  data_collator=collator, compute_metrics=compute_metrics)
trainer.train()
print(trainer.evaluate())                           # {'eval_loss': ..., 'eval_accuracy': ~0.92, ...}

# Inference on new text
model.eval()
batch = tok(["A moving, beautifully acted film.", "Two hours I will never get back."],
            return_tensors="pt", padding=True).to(model.device)
with torch.no_grad():
    probs = model(**batch).logits.softmax(-1)       # (2, 2)
print(probs.cpu().numpy().round(3))
```

Things to try: freeze the encoder (`for p in model.bert.parameters(): p.requires_grad = False`) and compare accuracy and training time with full fine-tuning; train on only 500 examples and compare with a TF-IDF plus logistic regression baseline; change the learning rate to $10^{-3}$ and watch the model collapse to predicting one class.

### Part B: LoRA on a small causal language model {#lab-lora}

We adapt `facebook/opt-125m` (12 layers, hidden size 768) to the style of children's stories using the TinyStories dataset, training only low-rank adapters on the attention query and value projections. The same code works for larger models by changing the checkpoint and `target_modules` (for LLaMA-style models use `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).

```python:lab_lora_opt.py
import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling,
                          TrainingArguments, Trainer, set_seed)

set_seed(0)
ckpt = "facebook/opt-125m"
tok = AutoTokenizer.from_pretrained(ckpt)
model = AutoModelForCausalLM.from_pretrained(ckpt)

# 1) Wrap the model: base weights are frozen, LoRA matrices A and B are inserted
cfg = LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],            # module names inside OPT's attention
    bias="none", task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, cfg)
model.print_trainable_parameters()
# about 295k trainable parameters out of about 125.5M (0.23%):
# 12 layers x 2 matrices x r (768 + 768) = 12 * 2 * 8 * 1536 = 294,912

# 2) Data: a small slice of TinyStories, tokenized and cut into fixed-length chunks
raw = load_dataset("roneneldan/TinyStories", split="train[:4000]")

def tokenize(batch):
    return tok(batch["text"], truncation=True, max_length=128)

ds = raw.map(tokenize, batched=True, remove_columns=raw.column_names)
# mlm=False -> labels are the input ids (shifted inside the model); padding positions get -100
collator = DataCollatorForLanguageModeling(tok, mlm=False)

# 3) Baseline perplexity before adaptation, then train
def mean_loss(trainer_, dataset):
    return trainer_.evaluate(eval_dataset=dataset)["eval_loss"]

args = TrainingArguments(
    output_dir="opt125m-lora-tinystories",
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,
    num_train_epochs=2,
    learning_rate=2e-4,                              # LoRA tolerates (and needs) a larger LR than full FT
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    logging_steps=25,
    save_strategy="no",
    report_to="none",
    fp16=torch.cuda.is_available(),
)
held_out = ds.select(range(3900, 4000))
trainer = Trainer(model=model, args=args, train_dataset=ds.select(range(3900)),
                  eval_dataset=held_out, data_collator=collator)
before = mean_loss(trainer, held_out)
trainer.train()
after = mean_loss(trainer, held_out)
print(f"held-out loss {before:.3f} -> {after:.3f}  (perplexity {torch.e ** before:.1f} -> {torch.e ** after:.1f})")

# 4) Generate with the adapted model
model.eval()
prompt = tok("Once upon a time, a little robot", return_tensors="pt").to(model.device)
out = model.generate(**prompt, max_new_tokens=60, do_sample=True, temperature=0.8, top_p=0.9,
                     pad_token_id=tok.eos_token_id)
print(tok.decode(out[0], skip_special_tokens=True))

# 5) Save only the adapter (about 1 MB), reload it later, and optionally merge it into the base weights
model.save_pretrained("opt125m-lora-adapter")
from peft import PeftModel
base = AutoModelForCausalLM.from_pretrained(ckpt)
restored = PeftModel.from_pretrained(base, "opt125m-lora-adapter")
merged = restored.merge_and_unload()                 # plain OPT model with W0 + (alpha/r) B A, no adapter overhead
print(type(merged).__name__)
```

You should see the held-out loss drop clearly (perplexity falling by a large factor), the generations adopt the simple vocabulary and sentence structure of the dataset, and the saved adapter file be about a megabyte whereas the full model is about 250 MB.

:::tip Experiments that make the theory concrete
1. Sweep $r\in\{1,2,4,8,16,64\}$ and plot held-out loss against trainable parameters. Where does it saturate?
2. Compare `q_proj, v_proj` against all linear layers (`q_proj, k_proj, v_proj, out_proj, fc1, fc2` in OPT) at a fixed number of trainable parameters (adjust $r$).
3. Set `lora_alpha` to 1, 16 and 64 at a fixed learning rate and observe the effective step size scale as $\alpha/r$.
4. Initialize $\mathbf{B}$ randomly instead of zero and compare the first-step loss with the zero-initialized run.
:::

### Part C (optional): QLoRA {#lab-qlora}

On a CUDA GPU, load the frozen base in 4-bit NF4 and train LoRA adapters on top. The only changes are the quantization config and a preparation call; the training loop is unchanged.

```python:lab_qlora.py
import torch
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",                       # NormalFloat4
    bnb_4bit_use_double_quant=True,                  # quantize the quantization constants too
    bnb_4bit_compute_dtype=torch.bfloat16,           # dequantize to bf16 for the matmuls
)
model = AutoModelForCausalLM.from_pretrained("facebook/opt-1.3b", quantization_config=bnb,
                                             device_map="auto")
model = prepare_model_for_kbit_training(model)       # casts norms, enables gradient checkpointing hooks
cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type=TaskType.CAUSAL_LM,
                 target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"])
model = get_peft_model(model, cfg)
model.print_trainable_parameters()
print(f"GPU memory: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
```

## How to read and present a paper {#reading-papers}

Starting this week, one or two students present a paper each session from the curated list. Reading research papers is a skill, and it is the most transferable one in this course: architectures change, but the habit of extracting claims and checking evidence does not. This section is a practical guide.

{{figure: week10-paper-reading | The three-pass approach to reading a paper (after S. Keshav, "How to Read a Paper", 2007). Spend the time proportional to how much you need from the paper; most papers deserve only the first pass.}}

### Reading: three passes {#three-pass}

**Pass 1 (5 to 10 minutes): survey.** Read the title, abstract and introduction, glance at the section headings, read the conclusion, and look at every figure and table without reading the text. Answer: What category of paper is this (new method, analysis, benchmark, theory, system)? What is the claimed contribution? Is it relevant to what I need? Stop here for most papers.

**Pass 2 (about an hour): grasp the content.** Read carefully but skip derivation details. Study the main figure of the method until you can redraw it. For each experiment table, ask: *what is the comparison, what is held fixed, and what would the numbers look like if the claim were false?* Mark terms and references you do not know.

**Pass 3 (several hours): reproduce.** Re-derive key equations, try to predict the experiments before reading them, and read the appendix, where the hyperparameters and honest details often hide. For a paper you will implement, check the code release against the text.

### A checklist of questions {#paper-questions}

| Question | What to look for |
| --- | --- |
| What problem, and why does it matter? | A concrete task or failure of prior work, stated without buzzwords |
| What is the key idea, in one sentence? | If you cannot say it plainly, you have not understood it yet |
| What exactly is new? | Differences from the closest prior method; ablate the novelty |
| Is the evidence sufficient? | Strong baselines, same compute and tuning budget, multiple seeds, confidence intervals |
| Does the experiment support the claim? | A claim about "reasoning" backed by a benchmark that could be contaminated or solved by shortcuts |
| What are the limitations? | Those stated by the authors, and those you can infer: scale, data, compute, assumptions |
| Is it reproducible? | Code, data, hyperparameters, compute cost, variance across runs |
| What would you do next? | A follow-up experiment, a failure case to test, a different setting |

Be generous and skeptical at the same time. Generous: assume the authors are competent and look for the best version of their argument. Skeptical: look for confounders (differences in data, compute or tuning between method and baseline), cherry-picked qualitative examples, and claims stronger than the experiments. Work through a paper's reading with the guiding example from this chapter: the Chinchilla paper's main claim, that parameters and tokens should scale equally, rests on three different estimation methods. Do they agree? Where do they differ? What assumptions (learning-rate schedule, tokenizer, data distribution) could change the answer?

### Presenting: the 15-minute talk {#presenting}

A good paper presentation teaches the audience the paper's idea, not its every detail. A workable allocation for 15 minutes (about 10 to 12 slides):

1. **Motivation and problem (2 min).** One slide that makes the audience want the answer. Show the failure of the status quo.
2. **Background needed (1 to 2 min).** Only what is required: a definition, a prior method the paper builds on. Link back to earlier weeks.
3. **The method (4 to 5 min).** Use one clear diagram, an annotated equation or a worked example. Walk through a small concrete instance before the general case.
4. **Experiments (3 min).** Pick the two or three results that carry the argument. Say what the baseline is, what the metric is, and what a skeptic would ask. Do not show every table.
5. **Limitations and critique (2 min).** What the paper does not show, what assumptions it makes, what could go wrong.
6. **Discussion questions (1 min).** Two or three genuinely open questions for the class.

Practical advice: put one idea on a slide and a headline sentence at the top that states the takeaway, not just the topic ("LoRA matches fine-tuning with 0.1% of the parameters", not "Results"). Redraw figures large and simple rather than pasting a tiny screenshot; always say what the axes are. Rehearse out loud with a timer. Prepare a backup slide for the questions you expect (the ablation, the hyperparameters, a proof step). Cite where figures come from.

**The reaction note.** Everyone else submits a one-paragraph note before class: summarize the claim in two sentences, name the one result you found most convincing and the one you found least convincing, and write one question for the presenter. The aim is to practice committing to a judgment, not to restate the abstract.

## Common mistakes and debugging tips {#mistakes}

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| Fine-tuned BERT predicts one class for everything, loss stuck near 0.69 | Learning rate too high (e.g. $10^{-3}$), or no warmup | Use $2\text{-}5\times10^{-5}$ with warmup; check labels are 0/1 and not shuffled against inputs |
| Training loss near zero but validation accuracy poor | Label leakage, duplicate train/val examples, or overfitting small data | Deduplicate; use fewer epochs and early stopping; evaluate on a clean split |
| Newly added classification head has terrible accuracy before training | Expected: the head is random. Warning message is normal | Train it; do not evaluate before fine-tuning |
| LoRA run loss does not decrease | Wrong `target_modules` names (nothing adapted), LR too low, or adapters not in train mode | Inspect `model.print_trainable_parameters()` and `model.named_modules()` for the right names; use LR about $10^{-4}$ to $3\times10^{-4}$ |
| LoRA model is worse than the base on other tasks | Adapter learned a narrow distribution; rank or LR too high | Lower $r$ or LR, add regularizing data, or scale the adapter down at inference |
| Model never stops generating, or repeats the prompt | End-of-sequence token missing from the training targets, or wrong chat template | Append the EOS or end-of-turn token to every training example and keep it in the loss |
| Loss computed on prompt tokens in SFT | Labels equal inputs for the whole sequence | Set the label to `-100` for prompt tokens so only responses are trained |
| Padding tokens influence training | Labels at padding positions not masked | Use `-100` on pads; ensure the attention mask is passed |
| `temperature`/`top_p` have no effect in `generate` | `do_sample=False` (the default) | Set `do_sample=True` |
| Out-of-memory when fine-tuning | Full-precision full fine-tune; long sequences; large batch | Use LoRA/QLoRA, gradient checkpointing, gradient accumulation, shorter maximum length, BF16 |
| Right-padded batch gives wrong generations | Decoder-only models must be left-padded for batch generation | Set `tokenizer.padding_side = "left"` before generating |
| Perplexity comparison across models is meaningless | Different tokenizers | Compare bits per byte on the same raw text |
| Benchmark score suspiciously high | Contamination, prompt leakage, or evaluating on the training split | Check overlap; evaluate on fresh or private data |
| RAG answers ignore the retrieved context | Poor prompt, irrelevant chunks, or context too long | Rerank, shorten, put the most relevant chunks first and last, instruct explicit grounding |
| DPO training diverges or both chosen and rejected log-probs fall | $\beta$ and LR mismatched; reference model not frozen or not the SFT model | Start from the SFT checkpoint as reference, small LR ($\sim 5\times10^{-7}$ for full models), check the margin and accuracy logs |

:::tip A debugging order for fine-tuning
(1) Overfit 16 examples to near-zero loss: if you cannot, the pipeline is broken (labels, masking, LR). (2) Verify what is being trained (parameter counts). (3) Print a decoded batch, including special tokens and the label mask. (4) Compare against a trivial baseline. (5) Only then tune hyperparameters.
:::

## Exercises {#exercises}

:::exercise Exercise 1: the BERT masking recipe (conceptual)
BERT selects 15% of tokens; of those, 80% become `[MASK]`, 10% a random token and 10% stay unchanged. (a) What fraction of *all* tokens in a sequence is replaced by `[MASK]`, replaced by a random token, and left unchanged but graded? (b) Why would "always use `[MASK]`" create a train/test mismatch? (c) Is the model told which of the 15% positions are "random" or "unchanged"?

:::details Solution
(a) $0.15\times0.8=12\%$ are `[MASK]`, $0.15\times0.1=1.5\%$ are random tokens, and $1.5\%$ are unchanged but still in the loss; 85% of tokens contribute no loss. (b) At fine-tuning and inference time the input never contains `[MASK]`; a model trained only on masked positions could learn to treat the mask symbol as the only place where careful prediction is required, and to produce weak representations for ordinary tokens. (c) No. For a position holding an ordinary-looking token the model cannot know whether it is genuine, corrupted or is being graded, so it must keep a careful representation of every position.
:::
:::

:::exercise Exercise 2: compute-optimal sizing (calculation)
You have a budget of $C=10^{24}$ FLOPs. (a) Using $C=6ND$ and the rule of thumb $D=20N$, find $N$ and $D$. (b) With the parametric fit of the text, ($A=406.4$, $B=410.7$, $\alpha=0.34$, $\beta=0.28$, $E=1.69$) compute the predicted loss for your answer to (a). (c) A deployment team says the model will serve a huge number of requests; which direction should you move $N$ and $D$ and why?

:::details Solution
(a) $C=6N\cdot20N=120N^2$, so $N=\sqrt{C/120}=\sqrt{8.33\times10^{21}}\approx 9.1\times10^{10}$ (91B parameters) and $D=20N\approx1.8\times10^{12}$ tokens. (b) $A/N^{0.34}$: $N^{0.34}=e^{0.34\ln(9.1\times 10^{10})}=e^{0.34\times25.24}=e^{8.58}\approx5.3\times10^3$, giving $0.077$. $D^{0.28}=e^{0.28\times\ln(1.83\times10^{12})}=e^{0.28\times28.24}=e^{7.91}\approx2.7\times10^3$, giving $0.152$. Sum: $1.69+0.077+0.152\approx1.92$. (c) Move toward a *smaller* model trained on *more* tokens: inference cost per query scales with $N$, so paying extra training compute to reach the same loss with a smaller $N$ pays off over many queries (this is what over-trained models like LLaMA do).
:::
:::

:::exercise Exercise 3: sizing LoRA (calculation)
A model has 32 layers, hidden size $d=4096$ and FFN width 11008 (gated FFN with gate, up and down projections). (a) How many trainable parameters does rank-16 LoRA on $\mathbf{W}_q$ and $\mathbf{W}_v$ in all layers have? (b) And on all seven linear matrices per layer? (c) For the full model of 6.74B parameters, what fraction is that? (d) Roughly how much optimizer-state memory (FP32 Adam moments only) does (b) need, compared with full fine-tuning?

:::details Solution
(a) Each of $q,v$ has $d_{\text{in}}+d_{\text{out}}=8192$, so $32\times2\times16\times8192=8{,}388{,}608\approx8.4$M. (b) Per layer: four attention matrices $4\times8192=32{,}768$; gate and up, each $4096+11008=15{,}104$; down $11008+4096=15{,}104$; total per unit rank $32{,}768+3\times15{,}104=78{,}080$. Times $r=16$ and 32 layers: $78{,}080\times16\times32=39{,}976{,}960\approx40.0$M. (c) $40.0/6740\approx0.59\%$ (and (a) is $0.12\%$). (d) Two FP32 moments are 8 bytes per trainable parameter: $40\text{M}\times8=0.32$ GB, versus $6.74\text{B}\times8=54$ GB for full fine-tuning.
:::
:::

:::exercise Exercise 4: why LoRA initializes B to zero but A randomly (derivation)
Using the gradient formulas in the text, show what happens if (i) both $\mathbf{A}$ and $\mathbf{B}$ are initialized to zero, and (ii) both are initialized randomly with the same scale as $\mathbf{A}$ in the text.

:::details Solution
(i) $\partial\mathcal{L}/\partial\mathbf{B}=s\,\boldsymbol{g}(\mathbf{A}\boldsymbol{x})^\top=\mathbf{0}$ since $\mathbf{A}=\mathbf{0}$, and $\partial\mathcal{L}/\partial\mathbf{A}=s\,\mathbf{B}^\top\boldsymbol{g}\,\boldsymbol{x}^\top=\mathbf{0}$ since $\mathbf{B}=\mathbf{0}$. Both gradients vanish and the parameters never move (a saddle point). (ii) $\Delta\mathbf{W}=\mathbf{B}\mathbf{A}\neq\mathbf{0}$ at step 0, so the network starts from a perturbed version of the pretrained function, adding noise of magnitude set by the initialization scale, which can degrade the starting point. With $\mathbf{B}=\mathbf{0}$ and $\mathbf{A}$ random, the starting function is exactly the pretrained one, $\mathbf{B}$ receives a nonzero gradient immediately, and $\mathbf{A}$ begins to learn once $\mathbf{B}\neq\mathbf{0}$.
:::
:::

:::exercise Exercise 5: temperature and entropy (derivation, challenging)
Let $p_T(v)\propto\exp(z_v/T)$ and let $H(T)$ be its entropy. (a) Show $H(T)=\log Z(T)-\frac{1}{T}\E_{p_T}[z]$ with $Z(T)=\sum_v e^{z_v/T}$. (b) Show $\frac{d}{dT}\E_{p_T}[z]=-\Var_{p_T}(z)/T^2$ and conclude that $\frac{dH}{dT}=\Var_{p_T}(z)/T^3\ge0$.

:::details Solution
(a) $\log p_T(v)=z_v/T-\log Z$, so $H=-\sum_v p_T(v)\log p_T(v)=-\frac1T\E[z]+\log Z$. (b) Differentiate $p_T$: $\frac{\partial}{\partial T}\log p_T(v)=-\frac{z_v}{T^2}+\frac{\E[z]}{T^2}=-\frac{z_v-\E[z]}{T^2}$. Then $\frac{d}{dT}\E[z]=\sum_v z_v\,p_T(v)\,\frac{\partial\log p_T(v)}{\partial T}=-\frac{1}{T^2}\Cov(z,z)=-\frac{\Var(z)}{T^2}$ (negative: raising $T$ moves mass to lower logits, so the mean logit falls). Also $\frac{d\log Z}{dT}=-\frac{\E[z]}{T^2}$. Hence $\frac{d}{dT}\big(-\tfrac1T\E[z]\big)=\frac{\E[z]}{T^2}-\frac1T\frac{d\E[z]}{dT}=\frac{\E[z]}{T^2}+\frac{\Var(z)}{T^3}$ and adding $\frac{d\log Z}{dT}=-\frac{\E[z]}{T^2}$ gives $\frac{dH}{dT}=\frac{\Var_{p_T}(z)}{T^3}\ge0$. So entropy never decreases with temperature, and is strictly increasing unless all logits are equal.
:::
:::

:::exercise Exercise 6: implement top-p in PyTorch (coding)
Write `top_p_filter(logits, p)` that takes a batch of logits of shape `(B, V)` and returns logits with all tokens outside the nucleus set to $-\infty$. Always keep at least the most probable token. Test with $p=0.9$ on the logits of the sampling figure.

:::details Solution
```python
import torch

def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    """logits: (B, V). Keep the smallest set of top tokens with cumulative prob >= p."""
    sorted_logits, sorted_idx = torch.sort(logits, dim=-1, descending=True)    # (B, V)
    probs = torch.softmax(sorted_logits, dim=-1)
    cum = probs.cumsum(dim=-1)
    # remove a token if the mass *before* it already reaches p (this keeps the first token that crosses p)
    remove_sorted = (cum - probs) >= p
    remove_sorted[:, 0] = False                                                # always keep the argmax
    remove = torch.zeros_like(remove_sorted).scatter(1, sorted_idx, remove_sorted)  # undo the sort
    return logits.masked_fill(remove, float("-inf"))

z = torch.tensor([[4.0, 3.4, 2.9, 2.2, 1.5, 1.0, 0.3, -0.2, -0.8, -1.5]])
out = top_p_filter(z, 0.9)
print(torch.isfinite(out).sum().item())            # 4 tokens survive
print(torch.softmax(out, -1))                      # renormalized over the nucleus
```
The subtraction `cum - probs` is the exclusive cumulative sum; thresholding it keeps the token that first pushes the mass past $p$, matching the definition of the smallest set with mass at least $p$.
:::
:::

:::exercise Exercise 7: is it emergence or the metric? (what would happen if...)
A model family has per-token accuracy $p=0.80, 0.90, 0.95, 0.99$ at four sizes on a task whose answers are 10 tokens long and scored by exact match. (a) What exact-match accuracies result? (b) A colleague claims the ability "emerges" between the 3rd and 4th model. What would you check? (c) How would you rescore to test smoothness?

:::details Solution
(a) $0.8^{10}=0.107$, $0.9^{10}=0.349$, $0.95^{10}=0.599$, $0.99^{10}=0.904$. Even with uniform increments in per-token reliability, exact match rises from about 11% to 90%, with the steepest part at the largest sizes. (b) Check whether a continuous metric (per-token accuracy, log-likelihood of the correct answer, edit distance) shows a smooth trend; how many model sizes are in the "jump" (here a single step); whether the prompt format was fixed; and the confidence intervals given the number of test items. (c) Report the continuous metric against model size on a log axis and see whether it follows a power law or a straight line in logit space; if so, the "emergence" is a property of the thresholded metric.
:::
:::

:::exercise Exercise 8: derive the optimal KL-regularized policy (derivation, challenging)
Show that for a fixed prompt, $\pi^*(y)=\frac{1}{Z}\pi_{\text{ref}}(y)e^{r(y)/\beta}$ maximizes $\E_{y\sim\pi}[r(y)]-\beta\KL(\pi\,\|\,\pi_{\text{ref}})$, and derive the DPO loss. Then explain what happens to the DPO objective's implicit regularization as $\beta$ increases.

:::details Solution
Write $\pi^*=\pi_{\text{ref}}e^{r/\beta}/Z$. Then $r(y)=\beta\log\frac{\pi^*(y)}{\pi_{\text{ref}}(y)}+\beta\log Z$. Substituting into the objective,
$$\E_\pi[r]-\beta\KL(\pi\|\pi_{\text{ref}})=\E_\pi\Big[\beta\log\tfrac{\pi^*}{\pi_{\text{ref}}}+\beta\log Z-\beta\log\tfrac{\pi}{\pi_{\text{ref}}}\Big]=\beta\log Z-\beta\KL(\pi\|\pi^*),$$
which is maximized exactly when $\KL(\pi\|\pi^*)=0$, i.e. $\pi=\pi^*$ (KL is nonnegative and zero only at equality). Using $r$ in the Bradley-Terry probability, the $\beta\log Z(x)$ terms cancel in the difference $r(x,y_w)-r(x,y_l)$, leaving $\sigma\big(\beta\log\frac{\pi^*(y_w)}{\pi_{\text{ref}}(y_w)}-\beta\log\frac{\pi^*(y_l)}{\pi_{\text{ref}}(y_l)}\big)$. Replacing $\pi^*$ by $\pi_\theta$ and taking the negative log-likelihood gives $\mathcal{L}_{\text{DPO}}$. A larger $\beta$ corresponds to a stronger KL penalty in the underlying RL problem: reward differences are "bought" with larger log-ratio changes only when $\beta$ is small; with large $\beta$ a small log-ratio gap already saturates the sigmoid, so the optimum stays closer to $\pi_{\text{ref}}$.
:::
:::

:::exercise Exercise 9: design a retrieval system (conceptual)
You are building a question-answering assistant over 50,000 pages of internal technical documentation that is updated weekly. List the design decisions for chunking, embeddings, retrieval, reranking and prompting, how you would evaluate retrieval separately from generation, and three failure modes you would test for.

:::details Solution
A reasonable design: chunk by section headings into 200 to 400 token passages with some overlap, prefixing each chunk with the page title and heading path; embed with a general-purpose or domain-tuned bi-encoder; index with HNSW; run hybrid retrieval (dense plus BM25 for identifiers and error codes), take 50 candidates and rerank with a cross-encoder to 5 passages; prompt with explicit grounding instructions, passage ids and an abstention instruction; re-embed only changed pages weekly. Evaluate retrieval with a labeled set of (question, relevant passage) pairs using recall@$k$ and MRR; evaluate generation with faithfulness (claims supported by the passages) and answer relevance, using human spot checks and an LLM judge with position randomization. Failure modes to test: answer split across two chunks; questions that need two documents; stale or contradictory versions of a page; queries with product codes that dense retrieval misses; a retrieved page containing instructions (prompt injection); questions whose answer is not in the corpus (does it abstain?).
:::
:::

:::exercise Exercise 10: bits per byte and benchmark validity (calculation and design)
Model A (a 32k-token tokenizer) has held-out loss 2.20 nats per token on a text with 1,000,000 tokens and 4,300,000 bytes. Model B (a 100k-token tokenizer) has loss 2.65 nats per token on the same text, which it tokenizes into 800,000 tokens. (a) Compute bits per byte for each. Which model is better? (b) Describe an experiment to test whether a model's high MMLU score is partly due to contamination.

:::details Solution
(a) Total nats = loss $\times$ tokens. Model A: $2.20\times10^6=2.2\times10^6$ nats; bits per byte $=\frac{2.2\times10^6}{\ln2\cdot4.3\times10^6}=\frac{2.2}{0.6931\times4.3}=0.738$. Model B: $2.65\times0.8\times10^6=2.12\times10^6$ nats; bits per byte $=\frac{2.12}{0.6931\times4.3}=0.711$. Model B is slightly better even though its per-token loss (and perplexity) is higher, because its tokens carry more text. (b) Examples: measure n-gram overlap between test items and the training corpus if available; compare accuracy on the original test items versus paraphrased or reordered (answer options shuffled) versions, since a model that memorized the original order loses accuracy under shuffling; test whether the model can complete the test question text verbatim from its first words; and evaluate on a fresh benchmark built after the training cutoff in the same style.
:::
:::

:::exercise Exercise 11: unbiased pass@k (derivation)
Explain why $1-\binom{n-c}{k}/\binom{n}{k}$ is the probability that a random size-$k$ subset of $n$ generated samples contains at least one correct sample, and why the naive estimate $1-(1-c/n)^k$ is biased upward or downward.

:::details Solution
If $c$ of the $n$ samples are correct, the number of size-$k$ subsets with *no* correct sample is $\binom{n-c}{k}$ out of $\binom{n}{k}$ equally likely subsets, so the probability that at least one is correct is $1-\binom{n-c}{k}/\binom{n}{k}$. It is unbiased for $\text{pass@}k=1-(1-\pi)^k$ because, with $c\sim\text{Binomial}(n,\pi)$, the probability of drawing $k$ samples all incorrect (sampling without replacement from the $n$) has expectation $(1-\pi)^k$ for $k\le n$. The plug-in estimate $1-(1-c/n)^k$ replaces $\pi$ by its estimate inside a convex function; by Jensen's inequality $\E[(1-c/n)^k]\ge(1-\pi)^k$, so the plug-in estimate of pass@$k$ is biased *downward*.
:::
:::

## Key takeaways {#takeaways}

:::key Key takeaways
- Pretraining objectives differ in what they hide and what the model sees: MLM (encoder, bidirectional, about 15% of positions graded), autoregressive (decoder, every position graded, generative), span corruption (encoder-decoder, text-to-text). Decoder-only autoregressive models dominate general-purpose LLMs.
- Loss falls as a smooth power law in parameters, data and compute. Chinchilla showed that for a fixed compute budget $C\approx 6ND$ you should scale $N$ and $D$ about equally (roughly 20 tokens per parameter); over-training smaller models is justified by inference cost. The exact constants are empirical.
- "Emergent abilities" are partly an artifact of discontinuous metrics (exact match is $p^k$); loss extrapolates well, specific skills do not.
- Fine-tuning ranges from full (16 bytes per parameter of training state) to LoRA (train $r(d_{\text{in}}+d_{\text{out}})$ parameters per matrix, mergeable, no inference cost), QLoRA (4-bit NF4 base plus LoRA), adapters, and prefix/prompt tuning.
- Instruction tuning teaches format and behavior; in-context learning and chain-of-thought elicit behavior at inference time without weight updates.
- Alignment: SFT, reward model with the Bradley-Terry loss, KL-regularized RL, or DPO, which derives a supervised loss from the same objective by eliminating the reward model.
- RAG supplies fresh, attributable knowledge but adds retrieval failure modes; evaluate retriever and generator separately.
- Evaluate honestly: perplexity (tokenizer-dependent), benchmarks (prompt-sensitive, contamination-prone), pass@k, human and LLM judges (biased). Hallucination is a consequence of the objective, mitigated but not eliminated by grounding and verification.
- Decoding is a modeling choice: greedy and beam for tasks with right answers; temperature, top-$k$ and top-$p$ for open-ended text.
- Reading papers is a skill: three passes, a checklist of questions, and a talk that teaches one idea well.
:::

## Further reading {#further-reading}

**Essential**
- Devlin, Chang, Lee, Toutanova. *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.* NAACL 2019. arXiv:1810.04805.
- Hoffmann et al. *Training Compute-Optimal Large Language Models* (Chinchilla). 2022. arXiv:2203.15556.
- Hu et al. *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR 2022. arXiv:2106.09685.

**Pretraining and scaling**
- Radford et al. *Improving Language Understanding by Generative Pre-Training* (GPT-1), 2018; *Language Models are Unsupervised Multitask Learners* (GPT-2), 2019. OpenAI technical reports.
- Brown et al. *Language Models are Few-Shot Learners* (GPT-3). NeurIPS 2020. arXiv:2005.14165.
- Raffel et al. *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer* (T5). JMLR 2020. arXiv:1910.10683.
- Liu et al. *RoBERTa: A Robustly Optimized BERT Pretraining Approach.* 2019. arXiv:1907.11692.
- Kaplan et al. *Scaling Laws for Neural Language Models.* 2020. arXiv:2001.08361.
- Touvron et al. *LLaMA: Open and Efficient Foundation Language Models.* 2023. arXiv:2302.13971.
- Wei et al. *Emergent Abilities of Large Language Models.* TMLR 2022. arXiv:2206.07682. Schaeffer, Miranda, Koyejo. *Are Emergent Abilities of Large Language Models a Mirage?* NeurIPS 2023. arXiv:2304.15004.

**Fine-tuning**
- Houlsby et al. *Parameter-Efficient Transfer Learning for NLP* (adapters). ICML 2019. arXiv:1902.00751.
- Li, Liang. *Prefix-Tuning: Optimizing Continuous Prompts for Generation.* ACL 2021. arXiv:2101.00190. Lester, Al-Rfou, Constant. *The Power of Scale for Parameter-Efficient Prompt Tuning.* EMNLP 2021. arXiv:2104.08691.
- Dettmers, Pagnoni, Holtzman, Zettlemoyer. *QLoRA: Efficient Finetuning of Quantized LLMs.* NeurIPS 2023. arXiv:2305.14314.

**Instruction tuning, prompting, alignment**
- Wei et al. *Finetuned Language Models Are Zero-Shot Learners* (FLAN). ICLR 2022. arXiv:2109.01652.
- Wei et al. *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models.* NeurIPS 2022. arXiv:2201.11903.
- Ouyang et al. *Training Language Models to Follow Instructions with Human Feedback* (InstructGPT). NeurIPS 2022. arXiv:2203.02155.
- Zhou et al. *LIMA: Less Is More for Alignment.* NeurIPS 2023. arXiv:2305.11206.
- Rafailov et al. *Direct Preference Optimization: Your Language Model Is Secretly a Reward Model.* NeurIPS 2023. arXiv:2305.18290.

**Retrieval, decoding, evaluation**
- Lewis et al. *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* NeurIPS 2020. arXiv:2005.11401.
- Liu et al. *Lost in the Middle: How Language Models Use Long Contexts.* TACL 2024. arXiv:2307.03172.
- Holtzman et al. *The Curious Case of Neural Text Degeneration.* ICLR 2020. arXiv:1904.09751.
- Chen et al. *Evaluating Large Language Models Trained on Code* (HumanEval, pass@k). 2021. arXiv:2107.03374.

**Reading and presenting**
- Keshav. *How to Read a Paper.* ACM SIGCOMM Computer Communication Review, 2007.
- Hugging Face documentation: the `transformers` Trainer, `peft` and `datasets` guides; Karpathy's lecture series "Neural Networks: Zero to Hero" (the GPT-from-scratch lecture).
