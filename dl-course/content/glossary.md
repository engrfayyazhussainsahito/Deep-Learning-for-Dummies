---
title: Glossary
summary: Over 150 deep learning terms with short definitions and links to the week where each is covered.
eyebrow: Reference
---

Short definitions of the terms used throughout the course, in alphabetical order. Each entry links to the week where the idea is developed; use the page search to jump to a term.

## A

**Activation function** ([Week 2](week-02.html)): A nonlinear function applied elementwise after a linear layer (ReLU, GELU, tanh, sigmoid); without it a stack of layers collapses to one linear map.

**Actor-critic** ([Week 15](week-15.html)): A reinforcement-learning method that learns a policy (actor) and a value function (critic) together, using the critic to reduce the variance of the policy gradient.

**Adam** ([Week 3](week-03.html)): An adaptive optimizer that keeps exponential moving averages of the gradient and its square and scales each coordinate's step by them, with bias correction.

**AdamW** ([Week 3](week-03.html)): Adam with weight decay applied directly to the parameters instead of being added to the gradient, so decay is not rescaled by the adaptive denominator.

**Advantage** ([Week 15](week-15.html)): The amount by which an action's value exceeds the average value of the state, $A(s,a)=Q(s,a)-V(s)$; the learning signal in modern policy-gradient methods.

**Adversarial example** ([Week 17](week-17.html)): An input changed by a small, deliberately chosen perturbation that makes a model err while looking unchanged to a human.

**Adversarial training** ([Week 17](week-17.html)): Training on adversarial examples generated on the fly, which solves a min-max problem and buys robustness at a cost in accuracy and compute.

**Aleatoric uncertainty** ([Week 17](week-17.html)): Uncertainty from irreducible noise in the data; it remains no matter how much data you collect.

**Attention** ([Week 7](week-07.html)): A mechanism that forms a weighted average of values, with weights given by a softmax of query-key similarities; first used in seq2seq translation and now the core of Transformers.

**Autoencoder** ([Week 12](week-12.html)): A network that compresses its input to a low-dimensional code and reconstructs it, trained to minimize reconstruction error.

**Automatic differentiation** ([Week 2](week-02.html)): Computing exact derivatives of a program by applying the chain rule to its computational graph; reverse mode is backpropagation.

**Autoregressive model** ([Week 13](week-13.html)): A model that factorizes a joint distribution as a product of conditionals, $p(x)=\prod_t p(x_t\mid x_{<t})$, and samples one element at a time.

## B

**Backpropagation** ([Week 2](week-02.html)): The algorithm that computes the gradient of a loss with respect to all parameters by propagating error signals backward through the computational graph.

**Backpropagation through time** ([Week 7](week-07.html)): Backprop applied to an RNN unrolled over time steps, which multiplies Jacobians across steps and so suffers from vanishing or exploding gradients.

**Batch normalization** ([Week 3](week-03.html)): Normalizes each feature over the minibatch to zero mean and unit variance, then rescales with learned parameters; it uses running statistics at test time.

**Batch size** ([Week 3](week-03.html)): The number of examples used for one gradient estimate; it trades gradient noise against hardware efficiency.

**Beam search** ([Week 7](week-07.html)): A decoding method that keeps the $B$ highest-scoring partial sequences at each step instead of only the best one.

**Bellman equation** ([Week 15](week-15.html)): A recursive relation expressing the value of a state or action in terms of the immediate reward and the discounted value of the next state.

**BERT** ([Week 10](week-10.html)): A bidirectional Transformer encoder pretrained with masked language modeling and fine-tuned for understanding tasks.

**Bias (statistical)** ([Week 4](week-04.html)): The error from a model class that cannot represent the true function; one term in the bias-variance decomposition.

**Bias-variance trade-off** ([Week 4](week-04.html)): The decomposition of expected squared error into squared bias, variance and noise, which explains why capacity must be matched to data.

**Bidirectional RNN** ([Week 7](week-07.html)): A recurrent network that reads the sequence in both directions and concatenates the two hidden states.

**Byte-pair encoding (BPE)** ([Week 8](week-08.html)): A subword tokenizer that starts from characters (or bytes) and repeatedly merges the most frequent adjacent pair.

## C

**Calibration** ([Week 17](week-17.html)): The agreement between a model's stated confidence and its actual accuracy; measured by reliability diagrams and the expected calibration error.

**Catastrophic forgetting** ([Week 6](week-06.html)): The loss of earlier knowledge when a network is fine-tuned on new data without safeguards.

**Causal mask** ([Week 9](week-09.html)): A triangular mask in self-attention that prevents a position from attending to later positions, enabling autoregressive training.

**Chain rule** ([Week 1](week-01.html)): The rule for differentiating a composition of functions; the mathematical basis of backpropagation.

**Checkpointing (gradient)** ([Week 16](week-16.html)): Saving only some activations in the forward pass and recomputing the rest during backward, trading extra compute for memory.

**Chinchilla scaling** ([Week 10](week-10.html)): The finding that, for a fixed compute budget, model size and training tokens should grow together, at roughly 20 tokens per parameter.

**Classifier-free guidance** ([Week 13](week-13.html)): A diffusion sampling technique that extrapolates from the unconditional to the conditional noise prediction to trade diversity for fidelity.

**CLIP** ([Week 11](week-11.html)): A model that trains an image encoder and a text encoder with a contrastive loss over image-caption pairs, enabling zero-shot classification.

**Computational graph** ([Week 2](week-02.html)): A directed acyclic graph of operations and tensors recording how a value is computed, so that gradients can be derived mechanically.

**Conformal prediction** ([Week 17](week-17.html)): A distribution-free method that turns any model's scores into prediction sets with finite-sample marginal coverage guarantees.

**Contrastive learning** ([Week 8](week-08.html)): Learning representations by pulling together embeddings of related pairs and pushing apart unrelated ones.

**Convolution** ([Week 5](week-05.html)): A linear operation that slides a small kernel over the input with shared weights, giving locality and translation equivariance.

**Cosine schedule** ([Week 3](week-03.html)): A learning-rate schedule that decays from the peak value to near zero following a half cosine curve.

**Cross-entropy** ([Week 1](week-01.html)): The expected negative log-probability a model assigns to the true outcomes; minimizing it is maximum likelihood for classification.

**Cross-validation** ([Week 4](week-04.html)): Estimating generalization by repeatedly training on part of the data and validating on the held-out part.

## D

**Data augmentation** ([Week 4](week-04.html)): Creating new training examples by label-preserving transformations such as crops, flips, noise or mixup.

**DDIM** ([Week 13](week-13.html)): A deterministic, non-Markovian sampler for diffusion models that gives good samples in far fewer steps.

**DDPM** ([Week 13](week-13.html)): The denoising diffusion probabilistic model: a fixed noising chain and a network trained to predict the added noise.

**Decoder-only model** ([Week 9](week-09.html)): A Transformer with causal self-attention that predicts the next token; the architecture behind GPT-style language models.

**Deep Q-network (DQN)** ([Week 15](week-15.html)): Q-learning with a neural network, a replay buffer and a target network, introduced for Atari games.

**Dense layer** ([Week 2](week-02.html)): A fully connected layer computing $\mathbf{W}\boldsymbol{x}+\boldsymbol{b}$; also called linear layer.

**Differential privacy** ([Week 17](week-17.html)): A formal guarantee that a mechanism's output distribution changes by a bounded amount when any one individual's data is added or removed.

**Diffusion model** ([Week 13](week-13.html)): A generative model that learns to reverse a gradual noising process, producing samples by iterative denoising.

**Direct preference optimization (DPO)** ([Week 10](week-10.html)): A method that fine-tunes a language model directly on preference pairs using a loss derived from the RLHF objective, with no reward model or RL loop.

**Discount factor** ([Week 15](week-15.html)): The factor $\gamma\in[0,1)$ that weights future rewards less than immediate ones.

**Discriminator** ([Week 12](week-12.html)): The network in a GAN that tries to tell real samples from generated ones.

**Distillation** ([Week 16](week-16.html)): Training a small student model to match the output distribution of a larger teacher.

**Distribution shift** ([Week 17](week-17.html)): A difference between training and deployment data distributions, such as covariate, label or concept shift.

**Double descent** ([Week 4](week-04.html)): The phenomenon in which test error falls, rises near the interpolation threshold, and falls again as model size grows.

**Dropout** ([Week 4](week-04.html)): Randomly zeroing activations during training (with rescaling) to prevent co-adaptation and regularize the model.

## E

**Early stopping** ([Week 4](week-04.html)): Halting training when validation performance stops improving, to limit overfitting.

**Embedding** ([Week 8](week-08.html)): A learned dense vector representing a discrete item such as a word, token, user or node.

**Encoder-decoder model** ([Week 9](week-09.html)): A Transformer in which an encoder reads the input and a decoder generates the output while attending to the encoder's states.

**Ensemble** ([Week 17](week-17.html)): A combination of several independently trained models; averaging their predictions improves accuracy and uncertainty estimates.

**Epistemic uncertainty** ([Week 17](week-17.html)): Uncertainty from limited knowledge of the right model, which shrinks with more data.

**Epoch** ([Week 1](week-01.html)): One full pass through the training dataset.

**Equivariance** ([Week 5](week-05.html)): A property whereby transforming the input transforms the output in a corresponding way; convolutions are translation equivariant.

**Evidence lower bound (ELBO)** ([Week 12](week-12.html)): A tractable lower bound on the log-likelihood used to train VAEs and, in a hierarchical form, diffusion models.

**Exploding gradient** ([Week 3](week-03.html)): Gradients whose norm grows exponentially with depth or time, causing unstable training; mitigated by clipping and good initialization.

**Exploration versus exploitation** ([Week 15](week-15.html)): The trade-off between trying new actions to gain information and using the best known action.

## F

**Fairness (group)** ([Week 17](week-17.html)): Criteria such as demographic parity and equalized odds that compare model behavior across demographic groups.

**Feed-forward network (Transformer)** ([Week 9](week-09.html)): The position-wise two-layer MLP in each Transformer block, usually with hidden width four times the model dimension.

**Fine-tuning** ([Week 6](week-06.html)): Continuing training of a pretrained model on a new task, often with a small learning rate.

**FlashAttention** ([Week 16](week-16.html)): An exact attention algorithm that processes tiles in fast on-chip memory so the full attention matrix is never written to slow memory.

**FLOPs** ([Week 16](week-16.html)): Floating-point operations; a measure of computation. A matrix product of sizes $M\times K$ and $K\times N$ takes $2MKN$ FLOPs.

**Fully sharded data parallel (FSDP)** ([Week 16](week-16.html)): A distributed training method that shards parameters, gradients and optimizer states across workers and gathers them as needed.

## G

**GAN** ([Week 12](week-12.html)): A generative adversarial network: a generator and a discriminator trained against each other.

**GELU** ([Week 2](week-02.html)): A smooth activation, $x\Phi(x)$, widely used in Transformers.

**Generalization** ([Week 4](week-04.html)): The ability of a model to perform well on unseen data drawn from the same distribution as the training data.

**Generator** ([Week 12](week-12.html)): The network in a GAN that maps noise to samples.

**Grad-CAM** ([Week 17](week-17.html)): An explanation method that weights the last convolutional feature maps by the pooled gradient of a class score to highlight relevant regions.

**Gradient** ([Week 1](week-01.html)): The vector of partial derivatives of a scalar function; it points in the direction of steepest increase.

**Gradient accumulation** ([Week 16](week-16.html)): Summing gradients from several micro-batches before one optimizer step, to emulate a large batch with limited memory.

**Gradient clipping** ([Week 7](week-07.html)): Rescaling the gradient when its norm exceeds a threshold, to avoid exploding updates.

**Gradient descent** ([Week 1](week-01.html)): Iteratively updating parameters in the direction opposite to the gradient of the loss.

**Graph convolutional network (GCN)** ([Week 14](week-14.html)): A graph network layer that averages neighbor features using a symmetrically normalized adjacency matrix with self-loops.

**Graph neural network (GNN)** ([Week 14](week-14.html)): A network that updates node representations by passing messages along the edges of a graph.

**Grokking** ([Week 17](week-17.html)): Delayed generalization in which a network that has memorized the training set suddenly begins to generalize after much longer training.

**GRU** ([Week 7](week-07.html)): A gated recurrent unit with update and reset gates; a lighter alternative to the LSTM.

## H

**He initialization** ([Week 3](week-03.html)): Weight initialization with variance $2/n_{in}$, suited to ReLU networks.

**Hyperparameter** ([Week 4](week-04.html)): A setting chosen before training (learning rate, depth, weight decay) rather than learned from data.

## I

**In-context learning** ([Week 10](week-10.html)): The ability of a large language model to perform a new task from examples in the prompt, without weight updates.

**Inductive bias** ([Week 5](week-05.html)): The assumptions built into a model that make some solutions more likely than others, such as locality in CNNs.

**InfoNCE** ([Week 8](week-08.html)): A contrastive loss that treats identifying the positive among many candidates as classification; a lower bound on mutual information.

**Initialization** ([Week 3](week-03.html)): The choice of starting weights, which determines whether signals and gradients keep a healthy scale through depth.

**Instruction tuning** ([Week 10](week-10.html)): Supervised fine-tuning of a language model on instruction-response pairs so it follows user requests.

**Integrated gradients** ([Week 17](week-17.html)): An attribution method that integrates the gradient along a path from a baseline input to the input; attributions sum to the output difference.

**Interpretability** ([Week 17](week-17.html)): Methods for understanding what a model computes and why, from saliency maps to circuit analysis.

## J

**Jacobian** ([Week 2](week-02.html)): The matrix of first partial derivatives of a vector-valued function; products of Jacobians describe gradient flow in deep networks.

## K

**Kernel (convolution)** ([Week 5](week-05.html)): The small tensor of learned weights slid over the input in a convolution; also called a filter.

**Key, query, value** ([Week 9](week-09.html)): The three projections of each token in attention: queries are matched against keys to produce weights over values.

**KL divergence** ([Week 1](week-01.html)): A non-symmetric measure of how one distribution differs from another, $\KL(p\|q)=\sum p\log(p/q)$.

**KV cache** ([Week 16](week-16.html)): Stored keys and values of past tokens during autoregressive decoding, avoiding recomputation at each step.

## L

**Label smoothing** ([Week 4](week-04.html)): Replacing one-hot targets with softened ones (for example 0.9 on the true class), which discourages overconfidence.

**Latent diffusion** ([Week 13](week-13.html)): Running the diffusion process in the compressed latent space of an autoencoder rather than in pixel space.

**Latent variable** ([Week 12](week-12.html)): An unobserved variable that explains the data, such as the code in a VAE.

**Layer normalization** ([Week 3](week-03.html)): Normalizes the features of each example separately; the standard normalization in Transformers.

**Learning rate** ([Week 1](week-01.html)): The step size of gradient-based updates; usually the most important hyperparameter.

**Learning-rate warmup** ([Week 3](week-03.html)): Gradually increasing the learning rate from near zero at the start of training to avoid early instability.

**Lipschitz continuity** ([Week 12](week-12.html)): A bound on how fast a function can change, $|f(x)-f(y)|\le L\|x-y\|$; the critic in a Wasserstein GAN must be 1-Lipschitz.

**Logit** ([Week 2](week-02.html)): The unnormalized score before a softmax or sigmoid.

**LoRA** ([Week 10](week-10.html)): Low-rank adaptation: fine-tuning by learning a low-rank update $\mathbf{B}\mathbf{A}$ to frozen weight matrices.

**Loss function** ([Week 1](week-01.html)): A scalar measuring how badly a model's predictions match the targets; the quantity gradient descent minimizes.

**LSTM** ([Week 7](week-07.html)): Long short-term memory: a recurrent unit with input, forget and output gates and an additive cell state that eases long-range gradient flow.

## M

**Masked language modeling** ([Week 10](week-10.html)): A pretraining objective that hides some tokens and trains the model to recover them from context.

**Maximum likelihood** ([Week 1](week-01.html)): Choosing parameters that maximize the probability of the observed data.

**Mean squared error** ([Week 1](week-01.html)): The average squared difference between predictions and targets; maximum likelihood under Gaussian noise.

**Membership inference** ([Week 17](week-17.html)): An attack that determines whether a given example was in a model's training set.

**Message passing** ([Week 14](week-14.html)): The GNN computation in which each node aggregates messages from its neighbors and updates its state.

**Minibatch** ([Week 1](week-01.html)): A small random subset of the training data used to compute a stochastic gradient.

**Mixed precision** ([Week 16](week-16.html)): Training with lower-precision numbers (fp16 or bf16) for most operations while keeping critical values in fp32.

**Mixture of experts** ([Week 16](week-16.html)): A layer with many expert sub-networks of which a router activates only a few per token, raising capacity without raising per-token compute.

**Mixup** ([Week 4](week-04.html)): An augmentation that trains on convex combinations of pairs of examples and their labels.

**Model card** ([Week 18](week-18.html)): A short document describing a model's intended use, training data, evaluation results and limitations.

**Momentum** ([Week 3](week-03.html)): An optimizer term that accumulates an exponential average of past gradients, damping oscillations and speeding progress along consistent directions.

**Multi-head attention** ([Week 9](week-09.html)): Running several attention operations in parallel on different learned projections and concatenating the results.

## N

**Neural tangent kernel** ([Week 17](week-17.html)): The kernel describing training dynamics of an infinitely wide network in the linearized regime.

**Non-maximum suppression** ([Week 6](week-06.html)): A post-processing step in detection that removes overlapping boxes, keeping the highest-scoring one.

**Normalizing flow** ([Week 13](week-13.html)): A generative model built from invertible transformations, giving exact likelihoods via the change-of-variables formula.

**Nucleus sampling (top-p)** ([Week 10](week-10.html)): Sampling from the smallest set of tokens whose cumulative probability exceeds $p$.

## O

**Offline reinforcement learning** ([Week 15](week-15.html)): Learning a policy from a fixed dataset of interactions, without further environment access.

**One-hot encoding** ([Week 1](week-01.html)): Representing a category as a vector with a single 1 and zeros elsewhere.

**Optimizer** ([Week 3](week-03.html)): The algorithm that updates parameters from gradients, such as SGD, momentum, Adam or AdamW.

**Out-of-distribution (OOD) detection** ([Week 17](week-17.html)): Identifying inputs that differ from the training distribution, for example via max-softmax or energy scores.

**Overfitting** ([Week 4](week-04.html)): Fitting noise in the training data so that training error is low but test error is high.

**Over-smoothing** ([Week 14](week-14.html)): The convergence of node features to similar values as GNN depth grows.

**Over-squashing** ([Week 14](week-14.html)): The compression of information from exponentially many distant nodes into a fixed-size vector, limiting long-range message passing.

## P

**Padding** ([Week 5](week-05.html)): Adding border values (usually zeros) around an input to control the output size of a convolution.

**Parameter** ([Week 1](week-01.html)): A learned quantity of a model, such as a weight or bias.

**Perceptron** ([Week 2](week-02.html)): The earliest trainable neuron model: a thresholded linear function of the inputs.

**Perplexity** ([Week 10](week-10.html)): The exponential of the average per-token negative log-likelihood; a standard language-model metric.

**Policy** ([Week 15](week-15.html)): A mapping from states to a distribution over actions.

**Policy gradient** ([Week 15](week-15.html)): A family of methods that directly ascend the expected return by following $\nabla_\theta\log\pi_\theta$ weighted by returns or advantages.

**Pooling** ([Week 5](week-05.html)): Downsampling that summarizes a local region by its maximum or average.

**Positional encoding** ([Week 9](week-09.html)): Information injected into token embeddings so that an order-agnostic attention layer can use position.

**Pretraining** ([Week 10](week-10.html)): Training a large model on broad data with a generic objective before adapting it to a task.

**Prompting** ([Week 10](week-10.html)): Specifying a task to a language model through its input text.

**Proximal policy optimization (PPO)** ([Week 15](week-15.html)): A policy-gradient algorithm that clips the probability ratio to keep each update close to the previous policy.

**Pruning** ([Week 16](week-16.html)): Removing weights or structures from a trained network to make it smaller or faster.

## Q

**Q-learning** ([Week 15](week-15.html)): An off-policy temporal-difference method that learns action values using the maximum over next actions.

**Quantization** ([Week 16](week-16.html)): Representing weights and activations with fewer bits, such as 8-bit or 4-bit integers.

## R

**Receptive field** ([Week 5](week-05.html)): The region of the input that can influence a given unit's activation.

**Recurrent neural network (RNN)** ([Week 7](week-07.html)): A network that processes a sequence by updating a hidden state at each step with shared weights.

**Regularization** ([Week 4](week-04.html)): Any technique that reduces generalization error, often by constraining or penalizing the model.

**Reinforcement learning** ([Week 15](week-15.html)): Learning to act by maximizing cumulative reward through interaction with an environment.

**ReLU** ([Week 2](week-02.html)): The rectified linear unit $\max(0,x)$; a simple activation that mitigates vanishing gradients.

**Reparameterization trick** ([Week 12](week-12.html)): Expressing a sample as a deterministic function of parameters and independent noise so gradients can flow through sampling.

**Replay buffer** ([Week 15](week-15.html)): A memory of past transitions from which minibatches are sampled to decorrelate updates in off-policy RL.

**Residual connection** ([Week 6](week-06.html)): A skip connection that adds a block's input to its output, giving gradients an identity path.

**Retrieval-augmented generation (RAG)** ([Week 10](week-10.html)): Supplying a language model with retrieved documents at inference time to ground its answers.

**Reward model** ([Week 10](week-10.html)): A model trained on human preference comparisons to score responses in RLHF.

**RLHF** ([Week 10](week-10.html)): Reinforcement learning from human feedback: optimizing a language model against a learned reward with a KL penalty to a reference model.

**RMSProp** ([Week 3](week-03.html)): An adaptive optimizer that divides the gradient by a running average of its recent magnitude.

**Roofline model** ([Week 16](week-16.html)): A performance model bounding throughput by the minimum of peak compute and arithmetic intensity times memory bandwidth.

## S

**Saliency map** ([Week 17](week-17.html)): A map of the sensitivity of a model output to each input feature, usually from the input gradient.

**Scaling law** ([Week 10](week-10.html)): An empirical power-law relation between loss and model size, data or compute.

**Score function** ([Week 13](week-13.html)): The gradient of the log density with respect to the data, $\nabla_x\log p(x)$; diffusion models learn it implicitly.

**Self-attention** ([Week 9](week-09.html)): Attention in which queries, keys and values all come from the same sequence.

**Self-supervised learning** ([Week 11](week-11.html)): Learning from labels derived from the data itself, such as masked patches or augmented views.

**Seq2seq** ([Week 7](week-07.html)): An encoder-decoder architecture mapping an input sequence to an output sequence of a different length.

**SGD** ([Week 1](week-01.html)): Stochastic gradient descent: gradient descent using minibatch gradient estimates.

**Sigmoid** ([Week 2](week-02.html)): The logistic function $1/(1+e^{-x})$, mapping reals to $(0,1)$.

**Skip-gram** ([Week 8](week-08.html)): A word2vec model that predicts surrounding words from a center word.

**Softmax** ([Week 2](week-02.html)): The function that turns a vector of logits into a probability distribution, $e^{z_i}/\sum_j e^{z_j}$.

**Sparse autoencoder** ([Week 17](week-17.html)): An autoencoder with a sparsity penalty used to decompose a model's activations into interpretable features.

**Stochastic depth** ([Week 6](week-06.html)): Randomly skipping residual blocks during training as a regularizer.

**Stride** ([Week 5](week-05.html)): The step by which a convolution kernel moves across the input.

**Subword tokenization** ([Week 8](week-08.html)): Splitting text into units between characters and words, such as BPE or WordPiece pieces.

**Superposition** ([Week 17](week-17.html)): Encoding more features than there are dimensions by using non-orthogonal directions in a representation.

**Swin Transformer** ([Week 11](week-11.html)): A hierarchical vision Transformer with attention restricted to shifted local windows.

## T

**Target network** ([Week 15](week-15.html)): A slowly updated copy of the Q-network used to compute stable bootstrap targets.

**Temperature** ([Week 10](week-10.html)): A scalar dividing the logits before a softmax; low values sharpen and high values flatten the distribution.

**Temperature scaling** ([Week 17](week-17.html)): A post-hoc calibration method that fits one temperature on validation data without changing predictions.

**Temporal-difference learning** ([Week 15](week-15.html)): Updating value estimates toward a bootstrapped target of reward plus the discounted next estimate.

**Tensor** ([Week 1](week-01.html)): A multi-dimensional array, the basic data structure of deep learning frameworks.

**Tokenizer** ([Week 8](week-08.html)): A procedure that converts text into a sequence of integer token ids.

**Transfer learning** ([Week 6](week-06.html)): Reusing knowledge from a source task or dataset to improve performance on a target task.

**Transformer** ([Week 9](week-09.html)): An architecture built from self-attention and feed-forward blocks with residual connections and normalization.

## U

**Underfitting** ([Week 4](week-04.html)): A model too simple or too briefly trained to capture the structure in the data.

**U-Net** ([Week 6](week-06.html)): An encoder-decoder convolutional network with skip connections between matching resolutions, used for segmentation and diffusion denoising.

**Universal approximation theorem** ([Week 2](week-02.html)): The result that a feed-forward network with one hidden layer and enough units can approximate any continuous function on a compact set.

## V

**VAE** ([Week 12](week-12.html)): A variational autoencoder: an encoder and decoder trained to maximize the ELBO, with a prior over the latent code.

**Value function** ([Week 15](week-15.html)): The expected return from a state (or state-action pair) under a policy.

**Vanishing gradient** ([Week 3](week-03.html)): Gradients that shrink exponentially with depth or time so early layers barely learn.

**Variance (statistical)** ([Week 4](week-04.html)): The sensitivity of a model's predictions to the particular training sample; the other term in the bias-variance decomposition.

**Vision Transformer (ViT)** ([Week 11](week-11.html)): A Transformer that treats an image as a sequence of linearly embedded patches.

## W

**Wasserstein distance** ([Week 12](week-12.html)): The earth-mover distance between distributions; it gives useful gradients even when supports do not overlap.

**Weight decay** ([Week 4](week-04.html)): Shrinking weights toward zero at each step; equivalent to L2 regularization for plain SGD.

**Weight sharing** ([Week 5](week-05.html)): Reusing the same parameters at many positions, as in convolutions and recurrent networks.

**Weisfeiler-Lehman test** ([Week 14](week-14.html)): An iterative color-refinement test for graph isomorphism that bounds the power of message-passing GNNs.

**word2vec** ([Week 8](week-08.html)): A family of shallow models (skip-gram, CBOW) that learn word embeddings from co-occurrence.

## X

**Xavier (Glorot) initialization** ([Week 3](week-03.html)): Weight initialization with variance $2/(n_{in}+n_{out})$, designed for tanh or linear layers.

## Z

**Zero-shot learning** ([Week 11](week-11.html)): Performing a task without any task-specific training examples, for instance by prompting CLIP with class names.
