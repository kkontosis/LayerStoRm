# References

The design is drawn from published work. These are **influences** — the
implementations here are independent unless
[`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md) records that code was
adapted.

## Reflected in the engine

Each of these shaped a subsystem that ships:

- [A Deep-Dive Into the New Flash MLA Kernel](https://github.com/deepseek-ai/FlashMLA) — DeepSeek-AI, 2025 — the SM120 MLA attention kernels derive from FlashMLA (see [`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md))
- [SnapMLA: Efficient Long-Context MLA Decoding via Hardware-Aware FP8 Quantized Pipelining](https://arxiv.org/abs/2602.10718) — Zhang, Su, Hu, Yang et al., 2026 — FP8 compressed-latent decoding with the RoPE slice kept BF16
- [TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate](https://arxiv.org/abs/2504.19874) — Zandieh et al., 2025 — the 4-bit KV codec
- [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) — Kwon et al., 2023 — paged KV with copy-on-write forks
- [HiSparse: Scaling Sparse-Attention Decoding with Hierarchical KV Cache Management](https://arxiv.org/abs/2608.07009) — Xie, Huang, Huang, Xu, Ma, Kozyrakis, 2026 — sparse-guided hot-VRAM / pinned-host KV tiering ([SGLang implementation guide](https://docs.sglang.io/docs/advanced_features/hisparse_guide))
- DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation — Cheng, Yu, Shao, Li, Xiong et al. — the speculative decode arm
- [PreScope: Unleashing the Power of Prefetching for Resource-Constrained MoE Inference](https://arxiv.org/abs/2509.23638) — Yu, Zhang, Dong et al., 2025 — gating lookahead behind the expert prefetch predictor
- [MoE-SpeQ: Speculative Quantized Decoding with Proactive Expert Prefetching and Offloading](https://arxiv.org/abs/2511.14102) — 2025 — the expert-prediction model
- [SP-MoE: Speculative Decoding and Prefetching for Accelerating MoE-based Model Inference](https://arxiv.org/abs/2510.10302) — 2025 — planning verification's expert transfers from the draft's gating weights
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) — DeepSeek-AI, 2024 — the MLA + MoE architecture served here
- [GLM-5: from Vibe Coding to Agentic Engineering](https://arxiv.org/abs/2602.15763) — GLM-5 Team, 2026 — the primary target model

## Background

Read while designing the above; not implemented here.

**Attention, KV cache and long context**

- [FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/abs/2407.08608) — Shah et al., 2024
- [Helix Parallelism: Rethinking Sharding Strategies for Interactive Multi-Million-Token LLM Decoding](https://arxiv.org/abs/2507.07120) — Bhatia et al., 2025
- [IndexCache: Accelerating Sparse Attention via Cross-Layer Index Reuse](https://arxiv.org/abs/2603.12201) — 2026
- [KVShare: An LLM Service System with Efficient and Effective Multi-Tenant KV Cache Reuse](https://arxiv.org/abs/2503.16525) — 2025

**MoE offloading, expert caching and prefetching**

- [MoE-Infinity: Efficient MoE Inference on Personal Machines with Sparsity-Aware Expert Cache](https://arxiv.org/abs/2401.14361) — Xue et al., 2024
- [MoE-Lightning: High-Throughput MoE Inference on Memory-constrained GPUs](https://arxiv.org/abs/2411.11217) — 2024
- [fMoE: Fine-Grained Expert Offloading for Large Mixture-of-Experts Serving](https://arxiv.org/abs/2502.05370) — Yu et al., 2025
- [DALI: A Workload-Aware Offloading Framework for Efficient MoE Inference on Local PCs](https://arxiv.org/abs/2602.03495) — 2026
- [PROBE: Co-Balancing Computation and Communication in MoE Inference via Real-Time Predictive Prefetching](https://arxiv.org/abs/2602.00509) — 2026
- [KTransformers: Unleashing the Full Potential of CPU/GPU Hybrid Inference for MoE Models](https://doi.org/10.1145/3731569.3764843) — Chen, Xie, Zhang et al., 2025

**Speculative decoding, early exit and layer skipping**

- [Draft & Verify: Lossless Large Language Model Acceleration via Self-Speculative Decoding](https://arxiv.org/abs/2309.08168) — Zhang et al., 2023
- [Kangaroo: Lossless Self-Speculative Decoding via Double Early Exiting](https://arxiv.org/abs/2404.18911) — 2024
- [LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding](https://arxiv.org/abs/2404.16710) — Elhoushi, Shrivastava et al., 2024
- [CLaSp: In-Context Layer Skip for Self-Speculative Decoding](https://arxiv.org/abs/2505.24196) — Chen, Shan et al., 2025
- [Confident Adaptive Language Modeling](https://arxiv.org/abs/2207.07061) — Schuster, Fisch et al., 2022
- [Scaling Speculative Decoding with Lookahead Reasoning](https://arxiv.org/abs/2506.19830) — 2025
- [Utility-Driven Speculative Decoding for Mixture-of-Experts](https://arxiv.org/abs/2506.20675) — Saxena, Tsai et al., 2025
- [MoE-Spec: Expert Budgeting for Efficient Speculative Decoding](https://arxiv.org/abs/2602.16052) — McDanel et al., 2026
- Training-Free Loosely Speculative Decoding: Accepting Semantically Correct Drafts Beyond Exact Match

DSpark and Training-Free Loosely Speculative Decoding are listed without
links because the copies consulted here carry no canonical URL.

## Thanks

LayerStoRm stands on the shoulders of the open inference ecosystem — for
reference implementations, design ideas, and (where noted in
[`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md)) adapted code:

- [vLLM](https://github.com/vllm-project/vllm)
- [SGLang](https://github.com/sgl-project/sglang)
- [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)
- [ktransformers](https://github.com/kvcache-ai/ktransformers)
- [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp)

And thank you to the authors of the work referenced above. Nearly every
subsystem here started as someone else's published idea; the measurements
in this project's ledgers exist because that work was shared openly.
Errors in adapting it are mine.
