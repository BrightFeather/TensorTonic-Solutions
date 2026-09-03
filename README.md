# TensorTonic Solutions

Welcome to my TensorTonic solutions repository!

Here you'll find my solutions to various machine learning and deep learning problems from [TensorTonic](https://tensortonic.com).

## What is TensorTonic?

TensorTonic is a platform where you can implement core algorithms of Machine Learning from scratch.

This repository contains my personal solutions to these problems, automatically synchronized from the platform.

<!-- tensortonic:start -->
# Weijia Chen's TensorTonic Solutions

Verified machine learning implementations completed on [TensorTonic](https://www.tensortonic.com).

<p align="center">
  <img src="https://www.tensortonic.com/api/badge/chensir1994.svg" alt="TensorTonic Verified Solutions" width="100%" />
</p>

| Problem | Description | Link |
|---|---|---|
| Scaled Dot-Product Attention | Implement scaled dot-product attention in PyTorch using query-key scores, softmax weights, and value aggregation. | https://www.tensortonic.com/research/transformer/transformers-attention |
| Embedding Layer | Create PyTorch token embeddings and scale each lookup by the square root of the Transformer model dimension. | https://www.tensortonic.com/research/transformer/transformers-embedding |
| Encoder Block | Assemble a Transformer encoder block with multi-head attention, residual paths, layer normalization, and a feed-forward network. | https://www.tensortonic.com/research/transformer/transformers-encoder-block |
| Feed-Forward Network | Implement the Transformer's position-wise feed-forward network with two linear projections and a ReLU activation. | https://www.tensortonic.com/research/transformer/transformers-feed-forward |
| Layer Normalization | Implement Transformer layer normalization in NumPy using per-token mean, variance, scale, and bias. | https://www.tensortonic.com/research/transformer/transformers-layer-normalization |
| Multi-Head Attention | Build NumPy multi-head attention with learned projections, per-head scaled attention, concatenation, and output projection. | https://www.tensortonic.com/research/transformer/transformers-multi-head-attention |
| Positional Encoding | Implement sinusoidal Transformer positional encodings in NumPy with alternating sine and cosine dimensions. | https://www.tensortonic.com/research/transformer/transformers-positional-encoding |
| Tokenization | Build a word-level Transformer tokenizer with fixed special-token IDs, sorted vocabulary entries, encoding, and decoding. | https://www.tensortonic.com/research/transformer/transformers-tokenization |
| Document Ranking | Score each document by combining term frequency saturation, inverse document frequency, and document length normalization. | https://www.tensortonic.com/study-plans/cracking-nlp/nlp-document-ranking-bm25 |
| Text Deduplication (MinHash) | Detect near-duplicate documents with token shingles, deterministic MinHash signatures, and a similarity threshold. | https://www.tensortonic.com/study-plans/cracking-nlp/nlp-text-deduplication-minhash |
| TF-IDF | Build a vocabulary V sorted alphabetically and compute the TF-IDF score for every term in every document. | https://www.tensortonic.com/study-plans/cracking-nlp/nlp-tfidf |
| Actor-Critic (A2C) Loss with Entropy | Advantage Actor-Critic (A2C) jointly trains a policy and a value function with a single composite loss. | https://www.tensortonic.com/study-plans/cracking-rl/rl-actor-critic-a2c-loss |
| Bellman Expectation Equation | The Bellman expectation equation is the cornerstone of policy evaluation in reinforcement learning. | https://www.tensortonic.com/study-plans/cracking-rl/rl-bellman-expectation-equation |
| Bellman Optimality Equation | Apply one Bellman optimality backup by maximizing expected immediate reward plus discounted next-state value. | https://www.tensortonic.com/study-plans/cracking-rl/rl-bellman-optimality-equation |
| Bradley-Terry Reward Model Loss | Compute Bradley-Terry pairwise preference loss for chosen and rejected reward-model scores in RLHF. | https://www.tensortonic.com/study-plans/cracking-rl/rl-bradley-terry-reward-loss |
| Discounted Returns | Compute reverse-time discounted returns from a reward sequence using a configurable discount factor and terminal bootstrap. | https://www.tensortonic.com/study-plans/cracking-rl/rl-discounted-returns |
| DPO Closed-Form Loss | Direct Preference Optimization (DPO) replaces the reward-model + PPO pipeline of RLHF with a single supervised loss. | https://www.tensortonic.com/study-plans/cracking-rl/rl-dpo-closed-form-loss |
| Generalized Advantage Estimation (GAE) | Compute generalized advantage estimates by recursively combining rewards, value predictions, discounts, and GAE decay. | https://www.tensortonic.com/study-plans/cracking-rl/rl-generalized-advantage-estimation |
| PPO Clipped Surrogate Objective | Proximal Policy Optimization (PPO) optimizes a clipped surrogate objective that limits how far the new policy may drift from the data-collecting policy. | https://www.tensortonic.com/study-plans/cracking-rl/rl-ppo-clipped-surrogate |
| REINFORCE Gradient | Implement REINFORCE Gradient, and return the scalar loss as a float rounded to 4 decimals. | https://www.tensortonic.com/study-plans/cracking-rl/rl-reinforce-gradient |
| REINFORCE with Baseline | Implement REINFORCE with Baseline, and return the scalar loss as a Python float, rounded to 4 decimals. | https://www.tensortonic.com/study-plans/cracking-rl/rl-reinforce-with-baseline |
| PPO with Reference-Policy KL | Compute a PPO policy objective with reference-policy KL regularization for stable RLHF optimization. | https://www.tensortonic.com/study-plans/cracking-rl/rl-rlhf-ppo-kl-loss |
| Named-Dimension Batched Attention Scores | Compute batched multi-head query-key scores by contracting only the head-width dimension. | https://www.tensortonic.com/study-plans/language-modeling-from-scratch/cs336-l02-einsum-attention-scores |
| Gradient Accumulation Equivalence | Combine mean-loss gradients from unequal microbatches into one full-batch mean gradient, then apply a single SGD update. | https://www.tensortonic.com/study-plans/language-modeling-from-scratch/cs336-l02-gradient-accumulation-step |
| Transformer Training FLOP Estimator | Estimate one training step from forward matrix multiplications and a supplied forward attention cost. | https://www.tensortonic.com/study-plans/language-modeling-from-scratch/cs336-l02-training-flop-estimator |
| Mixed-Precision Training Memory Accountant | Compute exact storage for parameters, gradients, saved activations, and optimizer state from tensor shapes and byte widths. | https://www.tensortonic.com/study-plans/language-modeling-from-scratch/cs336-l02-training-memory-accountant |
| Parameter-Matched SwiGLU Block | Choose a parameter-matched SwiGLU hidden width under an available-width limit, then evaluate the bias-free block. | https://www.tensortonic.com/study-plans/language-modeling-from-scratch/cs336-l03-parameter-matched-swiglu |
| RMSNorm Forward Pass | Normalize each final-dimension vector by its root mean square and apply the learned scale without mean subtraction. | https://www.tensortonic.com/study-plans/language-modeling-from-scratch/cs336-l03-rmsnorm-forward |

View my verified ML profile: [TensorTonic profile](https://www.tensortonic.com/profile/chensir1994)
<!-- tensortonic:end -->
