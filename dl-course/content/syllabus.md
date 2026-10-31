---
title: Syllabus
summary: The complete 18-week plan, from perceptrons to diffusion models, with grading, projects and policies.
---

**Level:** Master's (graduate) · **Duration:** 18 weeks · **Format:** 2 lectures (75 min each) + 1 lab/tutorial (2 hrs) per week · **Credit hours:** 3 (3+0) with lab component

## Course description

This course takes students from the mathematical and algorithmic foundations of neural networks to the architectures and techniques that define modern deep learning: convolutional networks, sequence models, Transformers, large pretrained models, generative models (VAEs, GANs, diffusion), graph networks, and deep reinforcement learning. It also covers the practical side of training and deploying models at scale, along with robustness, interpretability, and responsible use. Students implement core ideas from scratch, reproduce results from research papers, and complete a capstone research-style project.

## Prerequisites

- Linear algebra (matrices, eigendecomposition, SVD), multivariate calculus, probability and statistics
- Programming in Python; familiarity with NumPy
- An introductory machine learning course (regression, classification, overfitting, cross-validation)

A short math refresher is built into Week 1 for students who need it.

## Learning outcomes

By the end of the course, students will be able to:

1. Derive and implement forward and backward passes for feed-forward networks, and explain how automatic differentiation works.
2. Select and tune optimizers, initializations, normalization layers, and regularizers to train deep models reliably.
3. Design, train, and evaluate CNNs, RNNs, and Transformer models using PyTorch.
4. Apply transfer learning, fine-tuning, and parameter-efficient methods to pretrained models.
5. Explain and implement major generative frameworks (autoencoders, VAEs, GANs, diffusion models).
6. Apply graph neural networks and deep reinforcement learning to appropriate problems.
7. Scale and optimize models using mixed precision, distributed training, quantization, pruning, and distillation.
8. Critically read, reproduce, and present deep learning research papers.
9. Assess robustness, fairness, interpretability, and ethical risks of deep learning systems.

## Tools and software

- Python 3.10+, PyTorch (primary), NumPy, Matplotlib, scikit-learn
- Hugging Face Transformers, Datasets, and PEFT libraries
- PyTorch Geometric (graphs), Gymnasium and Stable-Baselines3 (RL)
- Weights & Biases or TensorBoard for experiment tracking
- Git/GitHub; Google Colab or a university GPU cluster

## Textbooks and references

**Primary:**

- Goodfellow, Bengio, Courville. *Deep Learning* (MIT Press)
- Zhang, Lipton, Li, Smola. *Dive into Deep Learning* (free online, d2l.ai)

**Supplementary:**

- Prince. *Understanding Deep Learning* (free online)
- Bishop & Bishop. *Deep Learning: Foundations and Concepts*
- Murphy. *Probabilistic Machine Learning: Advanced Topics*
- Sutton & Barto. *Reinforcement Learning: An Introduction* (Week 15)
- Stanford CS231n and CS224n lecture notes; selected research papers listed per week

## Assessment and grading

| Component | Weight | Details |
| --- | --- | --- |
| Programming assignments (4) | 25% | Weeks 4, 8, 12, 16; implementation + short report |
| Midterm exam | 15% | Week 8; theory, derivations, and short coding questions |
| Paper presentation | 10% | Present and critique one research paper (Weeks 10 to 17) |
| Final project | 30% | Proposal (5%), progress report (5%), final report and code (15%), presentation (5%) |
| Final exam | 15% | Week 18; cumulative with emphasis on Weeks 9 to 17 |
| Participation and quizzes | 5% | Short weekly quizzes and in-class discussion |

## Course roadmap at a glance

| Phase | Weeks | Theme |
| --- | --- | --- |
| I. Foundations | 1 to 4 | Math, neural networks, backpropagation, optimization, regularization |
| II. Core Architectures | 5 to 8 | CNNs, modern vision models, sequence models, representation learning, midterm |
| III. Attention and Pretrained Models | 9 to 12 | Transformers, LLMs, vision and multimodal models, self-supervised learning, VAEs and GANs |
| IV. Advanced Topics | 13 to 18 | Diffusion, graph networks, deep RL, efficiency and scale, trustworthy DL and theory, final project |

## Weekly Schedule

### Phase I: Foundations

#### Week 1: Introduction, Mathematical Foundations, and Tooling

**Topics**

- What deep learning is; history from the perceptron to modern foundation models; why depth matters
- Machine learning recap: supervised, unsupervised, self-supervised learning; loss, risk, generalization
- Math refresher: linear algebra for tensors, matrix calculus, chain rule, probability, maximum likelihood, information theory (entropy, KL divergence, cross-entropy)
- Tensors, GPUs, and the PyTorch programming model; reproducibility and random seeds

**Readings:** Goodfellow et al., Ch. 1 to 5 (skim); d2l.ai preliminaries

**Lab:** PyTorch tensor operations, autograd basics, linear regression with gradient descent in PyTorch

#### Week 2: Neural Networks and Backpropagation

**Topics**

- The perceptron, multilayer perceptrons, and the universal approximation theorem
- Activation functions: sigmoid, tanh, ReLU, Leaky ReLU, GELU, SiLU/Swish; properties and trade-offs
- Output layers and loss functions: softmax with cross-entropy, MSE, hinge, focal loss
- Computational graphs; forward and backward pass; full derivation of backpropagation
- Automatic differentiation: forward mode vs reverse mode; how PyTorch autograd works

**Readings:** Goodfellow et al., Ch. 6; Rumelhart, Hinton, Williams (1986) on backpropagation

**Lab:** Implement an MLP and backpropagation from scratch in NumPy; verify gradients numerically (gradient checking); train on MNIST

#### Week 3: Optimization for Deep Learning

**Topics**

- Non-convex loss landscapes, saddle points, local minima, and flat vs sharp minima
- Gradient descent variants: batch, mini-batch, stochastic; effect of batch size
- Momentum, Nesterov accelerated gradient; adaptive methods: AdaGrad, RMSProp, Adam, AdamW
- Learning rate schedules: step decay, cosine annealing, warmup, cyclical rates, one-cycle policy
- Weight initialization: Xavier/Glorot and He initialization; vanishing and exploding gradients; gradient clipping
- Normalization: batch normalization, layer normalization, group normalization, and why they help optimization

**Readings:** Goodfellow et al., Ch. 8; Kingma & Ba (Adam, 2015); Ioffe & Szegedy (BatchNorm, 2015); Loshchilov & Hutter (AdamW, 2019)

**Lab:** Compare SGD, momentum, and Adam on the same model; visualize loss curves; study the effect of initialization and learning rate

#### Week 4: Regularization, Generalization, and Training Practice

**Topics**

- Bias-variance trade-off, underfitting and overfitting, double descent
- Regularization: L1/L2 weight decay, dropout, DropConnect, label smoothing, early stopping, data augmentation (including mixup and CutMix)
- Hyperparameter search: grid, random, Bayesian optimization; learning rate range tests
- Debugging neural networks: overfit-a-single-batch test, sanity checks, common bugs, loss curve diagnosis
- Experiment management: logging, checkpointing, reproducibility, train/validation/test discipline

**Readings:** Goodfellow et al., Ch. 7; Srivastava et al. (Dropout, 2014); Karpathy, "A Recipe for Training Neural Networks"

**Lab:** Build a full training pipeline with logging and checkpointing; run a regularization ablation study

**Due:** Assignment 1 (MLP from scratch, optimizers, and regularization experiments)

### Phase II: Core Architectures

#### Week 5: Convolutional Neural Networks

**Topics**

- Motivation: locality, weight sharing, translation equivariance
- Convolution and cross-correlation; stride, padding, dilation, channels; 1x1 convolutions; depthwise-separable convolutions
- Pooling layers; receptive field computation; output-size arithmetic
- Classic architectures: LeNet-5, AlexNet, VGG; what each contributed
- Training image classifiers: data pipelines, augmentation, ImageNet and CIFAR benchmarks

**Readings:** Goodfellow et al., Ch. 9; Krizhevsky et al. (AlexNet, 2012); Simonyan & Zisserman (VGG, 2015); CS231n notes on CNNs

**Lab:** Implement a CNN in PyTorch and train on CIFAR-10; experiment with depth, kernel size, and batch normalization

#### Week 6: Modern Vision Architectures, Transfer Learning, Detection and Segmentation

**Topics**

- Residual learning: ResNet, skip connections, why they ease optimization; ResNeXt, DenseNet
- Efficient and scalable designs: Inception, MobileNet, EfficientNet and compound scaling, ConvNeXt
- Transfer learning and fine-tuning: feature extraction vs full fine-tuning, domain shift, layer freezing
- Object detection: R-CNN family, SSD, YOLO, anchor-based vs anchor-free, IoU, NMS, mAP
- Semantic and instance segmentation: FCN, U-Net, Mask R-CNN, DeepLab

**Readings:** He et al. (ResNet, 2016); Tan & Le (EfficientNet, 2019); Ronneberger et al. (U-Net, 2015); Redmon et al. (YOLO, 2016)

**Lab:** Fine-tune a pretrained ResNet on a custom small dataset; run and evaluate a pretrained detector or train a small U-Net

#### Week 7: Sequence Models: RNNs, LSTMs, and the Road to Attention

**Topics**

- Sequential data and language modeling; one-to-many, many-to-one, many-to-many settings
- Recurrent neural networks; backpropagation through time (BPTT); vanishing/exploding gradients in time
- LSTM and GRU: gating mechanisms in detail; bidirectional and stacked RNNs
- Sequence-to-sequence models: encoder-decoder, teacher forcing, beam search
- Attention mechanism (Bahdanau and Luong): motivation, alignment, and soft attention
- Applications: machine translation, speech recognition, time-series forecasting; 1D CNNs and temporal convolutional networks as alternatives

**Readings:** Goodfellow et al., Ch. 10; Hochreiter & Schmidhuber (LSTM, 1997); Sutskever et al. (Seq2Seq, 2014); Bahdanau et al. (2015); Olah, "Understanding LSTM Networks"

**Lab:** Character-level language model with LSTM/GRU; seq2seq with attention for a toy translation task

#### Week 8: Representation Learning, Embeddings, and Midterm

**Topics**

- Distributed representations; word embeddings: word2vec (skip-gram, CBOW, negative sampling), GloVe, fastText
- Embedding geometry, analogies, and bias in embeddings
- Subword tokenization (BPE, WordPiece, SentencePiece) and character-level models
- Embeddings beyond text: users, items, entities, and metric learning (contrastive and triplet loss)
- Review of Weeks 1 to 8; midterm exam in the second session

**Readings:** Mikolov et al. (word2vec, 2013); Pennington et al. (GloVe, 2014); Sennrich et al. (BPE, 2016)

**Lab:** Train skip-gram embeddings; explore nearest neighbors and bias probes; use embeddings in a downstream classifier

**Due:** Assignment 2 (CNN image classification with transfer learning and an RNN sequence task); final project proposal (1 page: problem, dataset, baseline, plan)

**Exam:** Midterm covering Weeks 1 to 7

### Phase III: Attention, Transformers, and Pretrained Models

#### Week 9: The Transformer Architecture

**Topics**

- Self-attention: queries, keys, values; scaled dot-product attention; why scaling matters
- Multi-head attention, position-wise feed-forward layers, residual connections, layer norm (pre-norm vs post-norm)
- Positional information: sinusoidal, learned, relative positions, rotary embeddings (RoPE), ALiBi
- Encoder-only, decoder-only, and encoder-decoder variants; causal masking
- Computational complexity of attention (quadratic cost) and efficient attention variants: sparse, linear, sliding-window
- Training Transformers: warmup, label smoothing, mixed precision

**Readings:** Vaswani et al., "Attention Is All You Need" (2017); Alammar, "The Illustrated Transformer"; Karpathy, nanoGPT walkthrough

**Lab:** Implement a Transformer block and a small GPT-style decoder from scratch in PyTorch; train on a small text corpus

#### Week 10: Large Language Models and Pretrained NLP

**Topics**

- Pretraining paradigms: masked language modeling (BERT), autoregressive modeling (GPT), span corruption (T5)
- Scaling laws, compute-optimal training, emergent abilities and their limits
- Fine-tuning strategies: full fine-tuning, adapters, LoRA and QLoRA, prefix and prompt tuning
- Instruction tuning, in-context learning, prompting and chain-of-thought
- Alignment: supervised fine-tuning, RLHF, DPO and preference optimization (overview)
- Retrieval-augmented generation; evaluation of language models; hallucination and failure modes

**Readings:** Devlin et al. (BERT, 2019); Brown et al. (GPT-3, 2020); Kaplan et al. (Scaling Laws, 2020); Hoffmann et al. (Chinchilla, 2022); Hu et al. (LoRA, 2022); Ouyang et al. (InstructGPT, 2022)

**Lab:** Fine-tune a pretrained BERT or small LLM with Hugging Face for classification; apply LoRA to a small generative model

**Paper presentations begin**

#### Week 11: Vision Transformers, Multimodal Models, and Self-Supervised Learning

**Topics**

- Vision Transformer (ViT), patch embeddings, DeiT, Swin Transformer; CNN vs ViT inductive biases
- Transformers for detection and segmentation: DETR, SAM (overview)
- Self-supervised learning: contrastive methods (SimCLR, MoCo), non-contrastive methods (BYOL, DINO), masked image modeling (MAE)
- Vision-language models: CLIP, zero-shot transfer, image captioning, visual question answering
- Multimodal large models (overview): connecting vision encoders to language models

**Readings:** Dosovitskiy et al. (ViT, 2021); Chen et al. (SimCLR, 2020); He et al. (MAE, 2022); Radford et al. (CLIP, 2021)

**Lab:** Fine-tune a ViT; zero-shot classification and image-text retrieval with CLIP; linear probe of self-supervised features

#### Week 12: Generative Models I: Autoencoders, VAEs, and GANs

**Topics**

- Taxonomy of generative models; density estimation vs sampling
- Autoencoders: undercomplete, denoising, sparse; latent representations
- Variational autoencoders: ELBO derivation, reparameterization trick, KL regularization, beta-VAE, VQ-VAE
- Generative adversarial networks: minimax objective, training instability, mode collapse
- GAN variants: DCGAN, WGAN and WGAN-GP, conditional GANs, CycleGAN, StyleGAN
- Evaluating generative models: Inception Score, FID, precision and recall

**Readings:** Kingma & Welling (VAE, 2014); Goodfellow et al. (GAN, 2014); Arjovsky et al. (WGAN, 2017); Karras et al. (StyleGAN, 2019)

**Lab:** Implement a VAE and a DCGAN on MNIST or CelebA; visualize latent space interpolation

**Due:** Assignment 3 (Transformer from scratch and fine-tuning a pretrained model)

### Phase IV: Advanced Topics

#### Week 13: Generative Models II: Diffusion, Flows, and Autoregressive Models

**Topics**

- Denoising diffusion probabilistic models (DDPM): forward and reverse processes, noise schedules, training objective
- Score-based models and the SDE view; DDIM and fast sampling; connection to VAEs
- Conditioning and guidance: classifier guidance, classifier-free guidance, text-to-image conditioning
- Latent diffusion (Stable Diffusion-style pipelines); U-Net and DiT (diffusion Transformer) backbones
- Normalizing flows and autoregressive image/audio models (overview); comparison of all generative families
- Applications and risks: image, audio, and video generation; deepfakes, provenance, and watermarking

**Readings:** Ho et al. (DDPM, 2020); Song et al. (Score SDE, 2021); Rombach et al. (Latent Diffusion, 2022); Ho & Salimans (Classifier-Free Guidance, 2022)

**Lab:** Train a small DDPM on MNIST/CIFAR; experiment with sampling steps and guidance

#### Week 14: Graph Neural Networks and Geometric Deep Learning

**Topics**

- Graph-structured data: molecules, social networks, knowledge graphs, recommender systems
- Message passing framework; GCN, GraphSAGE, GAT, GIN; expressive power and the Weisfeiler-Lehman test
- Over-smoothing, over-squashing, and deeper GNNs; graph pooling and readout
- Tasks: node classification, link prediction, graph classification; graph Transformers
- Geometric deep learning: symmetry, equivariance, invariance; point clouds and 3D data; equivariant networks (overview)

**Readings:** Kipf & Welling (GCN, 2017); Hamilton et al. (GraphSAGE, 2017); Velickovic et al. (GAT, 2018); Bronstein et al., "Geometric Deep Learning" (2021)

**Lab:** Node classification on Cora and graph classification on a molecular dataset with PyTorch Geometric

#### Week 15: Deep Reinforcement Learning

**Topics**

- MDPs, returns, value functions, Bellman equations; Monte Carlo and temporal-difference learning
- Value-based deep RL: DQN, experience replay, target networks, Double DQN, Dueling DQN, prioritized replay
- Policy-gradient methods: REINFORCE, actor-critic, A2C/A3C, PPO, TRPO
- Continuous control: DDPG, TD3, SAC; exploration strategies
- Model-based RL and world models (overview); offline RL
- Connection to language models: RLHF, reward models, and policy optimization for LLMs

**Readings:** Sutton & Barto, Ch. 3, 6, 13; Mnih et al. (DQN, 2015); Schulman et al. (PPO, 2017); Haarnoja et al. (SAC, 2018)

**Lab:** Train DQN on CartPole/LunarLander; train a PPO agent with Stable-Baselines3 and compare

#### Week 16: Efficient and Scalable Deep Learning

**Topics**

- Hardware and systems: GPUs/TPUs, memory hierarchy, profiling, bottleneck analysis
- Mixed precision (FP16/BF16), gradient accumulation, gradient checkpointing
- Distributed training: data parallelism, model/tensor parallelism, pipeline parallelism, ZeRO and FSDP
- Efficient attention and kernels: FlashAttention, KV caching, speculative decoding (overview)
- Model compression: pruning, quantization (PTQ and QAT, 8-bit and 4-bit), knowledge distillation, low-rank factorization
- Mixture-of-Experts models; neural architecture search (overview)
- Deployment: ONNX, TorchScript and torch.compile, inference serving, edge and mobile considerations

**Readings:** Micikevicius et al. (Mixed Precision, 2018); Rajbhandari et al. (ZeRO, 2020); Dao et al. (FlashAttention, 2022); Hinton et al. (Distillation, 2015); Han et al. (Deep Compression, 2016)

**Lab:** Profile a training loop; apply mixed precision, quantization, and distillation to a model and report the accuracy/latency trade-off

**Due:** Assignment 4 (generative modeling, efficiency, or graph/RL task chosen from a menu); project progress report (preliminary results and baseline comparison)

#### Week 17: Trustworthy Deep Learning and Theory

**Topics**

- Robustness: adversarial examples (FGSM, PGD), adversarial training, distribution shift, out-of-distribution detection
- Uncertainty estimation: deep ensembles, MC dropout, calibration, conformal prediction (overview)
- Interpretability and explainability: saliency maps, Grad-CAM, integrated gradients, attention analysis, probing, mechanistic interpretability (overview)
- Fairness, bias, privacy (differential privacy, membership inference), safety, and responsible release
- Deep learning theory highlights: generalization in overparameterized models, implicit regularization, neural tangent kernel, lottery ticket hypothesis, grokking
- Open problems and research directions: reasoning, agents, data efficiency, continual learning, world models

**Readings:** Goodfellow et al. (Explaining and Harnessing Adversarial Examples, 2015); Selvaraju et al. (Grad-CAM, 2017); Guo et al. (Calibration, 2017); Zhang et al. (Rethinking Generalization, 2017); Frankle & Carbin (Lottery Ticket, 2019)

**Lab:** Craft adversarial examples and evaluate defenses; produce Grad-CAM explanations; measure model calibration

#### Week 18: Final Project Presentations, Review, and Final Exam

**Topics**

- Session 1: Final project presentations and demos (12 minutes plus Q&A per team)
- Session 2: Cumulative review of the course, emphasizing Weeks 9 to 17; discussion of next steps (reading groups, research directions, career paths)
- Final exam

**Due:** Final project report (6 to 8 pages in conference format) and code repository with a reproducibility README

**Exam:** Final exam (cumulative)

## Programming assignments summary

| # | Due | Topic | Core skills |
| --- | --- | --- | --- |
| 1 | Week 4 | MLPs from scratch, optimizers, regularization | Backpropagation, gradient checking, training diagnostics |
| 2 | Week 8 | CNNs with transfer learning; RNN/LSTM sequence task | PyTorch workflow, fine-tuning, sequence modeling |
| 3 | Week 12 | Transformer from scratch; fine-tuning a pretrained model with LoRA | Attention, Hugging Face, PEFT |
| 4 | Week 16 | Choice of generative modeling (VAE/GAN/diffusion), efficiency study, or GNN/RL task | Advanced architectures, benchmarking, trade-off analysis |

Each assignment includes a code component (auto-checked where possible) and a 2 to 4 page report with experiments, ablations, and discussion.

## Final project guidelines

**Team size:** 1 to 3 students

**Project types (choose one):**

- Applied project: solve a real problem with deep learning on a new or non-trivial dataset
- Reproduction project: reproduce a recent paper's key results and extend it with new experiments
- Research project: propose and test a novel idea, with ablations and analysis

**Milestones**

- Week 8: Proposal (1 page)
- Week 12: Instructor feedback meetings (10 minutes per team)
- Week 16: Progress report (2 pages, with baseline results)
- Week 18: Presentation and final report

**Evaluation criteria:** problem significance, technical correctness, experimental rigor (baselines, ablations, multiple seeds), clarity of writing, reproducibility, and insight in analysis.

## Paper presentation guidelines

Starting Week 10, one or two students present a paper per week from a curated list (aligned with that week's topic). Presenters give a 15-minute talk covering motivation, method, experiments, and limitations, followed by a critical discussion. Presenters submit slides 24 hours before class, and all students submit a one-paragraph reaction note.

## Course policies

- **Attendance and participation:** Regular attendance is expected; weekly quizzes are short and low-stakes.
- **Late submissions:** 10% penalty per day, up to 3 days; no credit afterwards unless arranged in advance.
- **Academic integrity:** Discussion of ideas is encouraged, but all submitted code and writing must be your own. Use of AI coding assistants and external code must be disclosed and cited; unattributed copying is a violation of the academic integrity policy.
- **Compute:** Students receive access to GPU resources (cloud credits or cluster accounts); heavy jobs should be planned early.
- **Accessibility:** Students requiring accommodations should contact the instructor in the first two weeks.

## Suggested additional reading and resources

- Stanford CS231n (vision), CS224n (NLP with deep learning), and CS236 (deep generative models)
- MIT 6.S191 and fast.ai Practical Deep Learning
- Andrej Karpathy, "Neural Networks: Zero to Hero" video series
- Papers With Code and arXiv (cs.LG, cs.CV, cs.CL) for staying current
- Distill.pub articles for intuition and visualization
