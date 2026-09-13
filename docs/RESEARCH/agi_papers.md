# AGI Papers — Latest 2023–2026 AI Research Summaries

**Task:** DIFFUSION.txt Phase K, Task 133  
**Date:** 2026-07-16  
**Purpose:** Summaries of key 2023–2026 papers relevant to the Transcender engine, with concrete application notes for each.

---

## 1. FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-Precision

| Field | Value |
|-------|-------|
| **Title** | FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-Precision |
| **Authors** | Jay Shah, Ganesh Bikshandi, Ying Zhang, Shiv Venkataraman, Francisco Rivera, Mert Hidayetoğlu, Brandon Reagen, José L. Abellán, John D. Davis (Tri Dao affiliated lineage) |
| **Year** | 2024 |
| **Venue** | arXiv:2407.08608 / NeurIPS 2024 |
| **Key Idea** | Builds on FlashAttention-2 by exploiting H100 GPU hardware features: (1) **asynchrony** — overlap softmax/exp (Tensor Core MMA) with matmul on the async copy engine (TMA / `cp.async`) using warp-specialization (producer-consumer warps); (2) **low-precision** — FP16 tensor-core matmul with FP32 softmax accumulation in the inner loop, plus an FP8 mode (E4M3) that uses two-stage accumulation (Welford-style residual) to recover FP8 MMA's reduced mantissa precision; (3) **overlap of softmax with matmul** by pipelining the softmax reduction onto a separate warp group while producer warps issue GEMMs. Result: ~1.5–2× over FA-2 on H100, up to 740 TFLOPs/s (74% of H100 FP16 peak), FP8 mode ~1.5× over FP16 mode. |
| **Application to Transcender** | Transcender already implements a CPU tiled online-softmax FlashAttention (`src/flash_attention.cpp`, block 64, row_max/row_sum causal). FA-3's contributions map to two engine improvements: (a) **async softmax overlap** — on CPU this translates to prefetching the next K/V tile while reducing the current tile (software pipelining with `_mm_prefetch` / `__builtin_prefetch`), already partially applicable to the existing block loop; (b) **low-precision FP8 accumulation with residual correction** — Transcender has `FP8Inference` (`src/inference_opt.cpp:665`) and `KVCache::quantize_fp8_block`; applying FA-3's two-stage FP8 accumulation (keep a running FP32 residual `r` to absorb the FP8 rounding error each block) would make the FP8 attention path numerically safe for causal workloads. The warp-specialization producer/consumer pattern is the blueprint for a future CUDA backend in `include/quant/gpu_compute.h`. |

---

## 2. Efficient Memory Management for Large Language Model Serving with PagedAttention (vLLM)

| Field | Value |
|-------|-------|
| **Title** | Efficient Memory Management for Large Language Model Serving with PagedAttention |
| **Authors** | Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, Ion Stoica |
| **Year** | 2023 |
| **Venue** | SOSP 2023 (arXiv:2309.06180) |
| **Key Idea** | KV cache memory is the dominant bottleneck in LLM serving (80% of memory on long contexts) and is managed wastefully with contiguous pre-allocation, causing internal fragmentation and limiting max batch size. PagedAttention borrows the OS **virtual memory / paging** idea: the logical KV sequence is a **block table** mapping logical block indices to non-contiguous physical blocks (fixed-size pages, e.g. 16 tokens). Blocks are allocated on demand from a pool, freed when a sequence finishes, and shared across sequences via **copy-on-write** when prompts share prefixes. This eliminates fragmentation, reduces KV memory waste from 60–80% to <4%, and enables 2–4× higher throughput on the same hardware. |
| **Application to Transcender** | Transcender implements `PagedAttention` (`src/inference_opt.cpp:14`) with a 256-block free-list pool, `alloc_block`/`free_block`, and a block-table-driven `forward` that walks `block_table[blk]` with online softmax per block. The direct improvements are: (a) **copy-on-write prefix sharing** — wire `PrefixCache` (`src/inference_opt.cpp:432`) to the paged pool so two requests sharing a system prompt point to the same physical blocks with refcounting + COW on divergence (currently `PrefixCache::store` makes a fresh `KVCache`, duplicating memory); (b) **block-size tuning** — vLLM shows block size 16 is near-optimal; Transcender defaults to 32, expose it as a config; (c) **1M-context** — the 256-block pool caps at 256×32=8192 tokens; raising to a paged-to-disk / tiered pool (hot blocks in RAM, cold blocks on mmap'd QUANT file) is the path to the 1M-context README claim. |

---

## 3. Fast Inference from Transformers via Speculative Decoding

| Field | Value |
|-------|-------|
| **Title** | Fast Inference from Transformers via Speculative Decoding |
| **Authors** | Yaniv Leviathan, Y. Matias, A. Roy (Google) |
| **Year** | 2023 |
| **Venue** | ICML 2023 (arXiv:2211.17192) |
| **Key Idea** | A small **draft** model proposes γ tokens autoregressively (cheap, fast); the large **target** model verifies all γ in a single parallel forward (amortized, since the target would compute them anyway). Verification uses **modified rejection sampling**: accept draft token `x` with probability `min(1, p_target(x)/p_draft(x))`; on rejection, resample from the normalized residual `norm(max(0, p_target − p_draft))`. Guarantee: output distribution is **exactly** the target distribution (zero quality loss) while latency drops by 2–3× because the target's parallel verify costs the same as one token. |
| **Application to Transcender** | Transcender implements `SpeculativeDecoder` (`src/inference_opt.cpp:98`) with draft+target, rejection sampling, adjusted-distribution resampling, and acceptance-rate tracking (`acceptance_rate_`). Improvements: (a) **adaptive γ** — increase γ when recent acceptance rate is high, decrease on miss (the decoder currently uses a fixed `gamma_`); (b) **draft-model warmup / draft = target shallow layers** — the strongest draft is the target's first N layers + a tiny LM head (Medusa/EAGLE style), which Transcender could wire via the existing `ModelShard` (`src/inference_opt.cpp:700`); (c) the current `generate()` re-runs the draft one token at a time without a KV cache for the draft — caching the draft KV across the γ steps would cut draft cost. |

---

## 4. Switch Transformer / Switch v2: Scaling to Trillion-Parameter Models (MoE)

| Field | Value |
|-------|-------|
| **Title** | Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity (Switch v2 improvements: expert capacity, load balancing) |
| **Authors** | William Fedus, Barret Zoph, Noam Shazeer (Google) |
| **Year** | 2021 (Switch) / 2022–2024 (Switch v2 / Expert Choice follow-ups) |
| **Venue** | JMLR 2022 (arXiv:2101.03961); Expert Choice routing: Zhou et al. ICML 2022 |
| **Key Idea** | Replace the dense FFN with a **Top-1 router**: a linear gate produces a probability per expert, the token is sent to exactly one expert (Top-1, not Top-2) → constant compute regardless of expert count. Key mechanisms: (1) **expert capacity** = `capacity_factor × tokens_per_batch / num_experts` — drop tokens beyond capacity to bound memory and expose load imbalance as a measurable "overflow" signal; (2) **load-balancing loss** = scaled dot-product between expert assignment fractions and expert probability fractions — differentiable, pushes the router toward uniform utilization without auxiliary z-loss; (3) **z-loss** prevents the router logits from drifting large (which would softmax-saturate and stall gradient flow). Expert Choice routing (v2 line) inverts the assignment: experts choose their top-k tokens, guaranteeing exact capacity and balanced load by construction. |
| **Application to Transcender** | Transcender's MoE (`src/moe_variants.cpp`, 24 variants) already has `softmax_with_topk`, `load_balance` loss, GQA, RoPE, SwiGLU, RMSNorm per the protected list. The Switch v2 mechanisms to harden: (a) **expert capacity factor + overflow accounting** — expose `capacity_factor` and report dropped-token rate so the 1T-param claim is measurable; (b) **z-loss** aux term added to the MoE aux loss (currently only load-balance) to stabilize Top-1 routing during long pretraining; (c) **load-balance loss formula** should match Switch's `n·Σ(f_i·P_i)` exactly (fraction assigned × fraction routed) for differentiable balance — verify the existing `load_balance` matches this, not a naive entropy. |

---

## 5. BitNet b1.58: 1-Bit LLMs (Ternary Weights)

| Field | Value |
|-------|-------|
| **Title** | The Era of 1-bit LLMs? All Large Language Models Are in 1.58 Bits |
| **Authors** | Shuming Ma, Hongyu Wang, Lingxiao Ma, Wang Li, Jiawei Jiang, Furu Wei (Microsoft) |
| **Year** | 2024 |
| **Venue** | arXiv:2402.17764 |
| **Key Idea** | Every weight is quantized to **ternary** {-1, 0, +1} (hence "1.58 bits" = log2(3)). Trained **from scratch** (not post-hoc quantization) with a Straight-Through Estimator (STE) so gradients flow through the rounding. Activations stay BF16; only weights are ternary. The matmul becomes pure **addition** (no multiply, since ×{-1,0,+1} is add/sub/skip). Result: BitNet b1.58 matches or beats FP16 baseline at the same size and is dramatically smaller + faster (no FLOP-heavy weight reads). Critical caveat: the quality only holds with **from-scratch** training; post-training ternarization (PTQ) collapses quality. |
| **Application to Transcender** | Transcender has a ternary path (`include/quant/ste_quantizer.h`, `.research/01-bitnet-b1.58-analysis.md`, CompressedKV ternary KV). The concrete applications: (a) the **ternary matmul** should be implemented as an add/sub accumulator (the `.bitnet/` data + `math_avx2.cpp` could gain a `_mm256` ternary GEMM that does `acc += sign*w` with no multiply) — the README's AVX2 GEMM would get a dedicated ternary kernel; (b) `STE quantizer` already exists; ensure training uses it for ternary blocks (train-in-format, which the `.research/06-ste-training-in-format.md` notes is biased-but-practical); (c) QUANT4/ternary MoE variants in `src/moe_variants.cpp` should train their expert weights ternary-from-scratch, not quantize-after. |

---

## 6. DeepSeek MLA (Multi-Latent Attention)

| Field | Value |
|-------|-------|
| **Title** | DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model (Multi-Latent Attention) |
| **Authors** | DeepSeek-AI (Zhihong Shao et al.) |
| **Year** | 2024 |
| **Venue** | arXiv:2405.04434 |
| **Key Idea** | KV cache is the memory bottleneck for long-context MoE. MLA **compresses K and V jointly into a low-rank latent vector**: instead of caching per-head K/V, cache a single down-projected latent `c_KV` (one `LoRA`-style down-projection W_DKV, then up-project W_UK/W_UV on the fly). For RoPE compatibility (RoPE is position-dependent and can't be folded into the up-projection), a small **decoupled RoPE** head stores a low-dim rotated key separately. Net effect: KV cache shrinks ~93% (e.g. 4096→576 latent dim) with negligible quality loss, enabling much longer context per GPU. Combined with MoE (DeepSeekMoE fine-grained experts + shared experts). |
| **Application to Transcender** | Transcender has an `MoE` variant in the 24-variant list (`include/quant/moe_variants.h`) and `CompressedKVCache` ternary compression (`src/inference_opt.cpp:350`). The real improvement: (a) implement the **joint low-rank KV down-projection** so the KV cache stores `c_KV ∈ R^{latent_dim}` instead of full per-head K/V — this is strictly better than the ternary 2-bit compression for quality and is the modern PagedAttention companion (smaller blocks → more concurrent sequences); (b) **decoupled RoPE** — keep a small rotated-key head so RoPE position info survives the low-rank bottleneck; (c) pair with PagedAttention so the paged blocks store the compressed latent, not raw K/V — this is the direct route to the 1M-context + 1T-param co-design. |

---

## 7. Mamba-2: Translatomics / Structured State Space Models with Attention

| Field | Value |
|-------|-------|
| **Title** | Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality (Mamba-2) |
| **Authors** | Tri Dao, Albert Gu |
| **Year** | 2024 |
| **Venue** | ICML 2024 (arXiv:2405.21060) |
| **Key Idea** | Establishes a **duality** between linear attention and State Space Models: the SSM recurrence can be written as a structured (semiseparable) matrix multiply, which equals a form of linear attention. Consequence: Mamba-2 can use the **hardware-efficient parallel scan / chunkwise matmul** of attention (FlashAttention-style tiling) instead of the sequential associative scan of Mamba-1, giving 2–8× throughput on GPUs. Also introduces a **multi-input SSM (SSD)** formulation enabling larger state dimension `d_state` (128→256) cheaply. The structured mask (semiseparable) generalizes the causal mask. |
| **Application to Transcender** | Transcender has a `Mamba MoE` variant and `flash_decoding` tiling (`src/inference_opt.cpp:566`). The Mamba-2 contributions map to: (a) **chunkwise SSM** — implement the SSM recurrence in block-tiled chunks that reuse the FlashAttention online-reduction loop (row_max/row_sum → m/s stats), so the Mamba variant uses the same fast path as attention; (b) **structured mask** — the causal mask builder (`ContinuousBatching::build_attention_mask`, `src/inference_opt.cpp:260`) generalizes to a semiseparable mask for hybrid attention/SSM layers; (c) the **Griffin hybrid** (below) is the direct instantiation — mix Mamba-2 SSM blocks with attention blocks. |

---

## 8. Griffin / RecurrentGemma: Hybrid SSM+Attention

| Field | Value |
|-------|-------|
| **Title** | RecurrentGemma: Moving Past Transformers for Efficient Language Modeling (Griffin RG-LRU blocks) |
| **Authors** | Soham De, Samuel L. Smith, Anushan Fernando, et al. (Google DeepMind) |
| **Year** | 2024 |
| **Venue** | arXiv:2404.07839 |
| **Key Idea** | A **hybrid** architecture alternating **RecurrentGemma Linear Recurrent Unit (RG-LRU)** blocks (an SSM-like gated recurrence with a real-gated small state) with **attention** blocks (most layers recurrent, a few attention for long-range recall). The **RG-LRU** gates the recurrent state with input-dependent forget and input gates (like a GRU but with a diagonal recurrent matrix — no full matrix multiply), giving O(1) memory per token (vs O(N) for attention) and constant-time per-step inference. Result: quality matches pure-attention Gemma at 2B scale while using far less KV memory and faster long-context inference. |
| **Application to Transcender** | The Transcender MoE list includes a `Mamba MoE` variant but no explicit **RG-LRU / recurrent block**. The concrete addition: (a) implement an **RG-LRU block** (diagonal recurrence with real gates, O(1) state) as a non-attention layer type usable in `DenseModel::layers` for hybrid models — pairs with `flash_decoding` for the attention layers; (b) the recurrent state is tiny and never pages, so it directly serves the "Paged KV 1M" goal by replacing most attention layers with recurrence; (c) gating math (sigmoid input gate × tanh candidate + (1-gate)×state) is simple enough to add to the existing `math.cpp`/`math_avx2.cpp` as a fused kernel. |

---

## 9. AGI / RSI: Recursive Self-Improvement (Seed AI)

| Field | Value |
|-------|-------|
| **Title** | Artificial General Intelligence and the Future of Humanity (Seed AI / RSI) — Yudkowsky & Hanson debates; later: "Speculations on Model-Producing Models", "Self-Rewarding Language Models" |
| **Authors** | Eliezer Yudkowsky (seed AI concept, 2007+); Metropolis et al.; Yuan et al. (Self-Rewarding LMs, 2024); Huang et al. (Large Language Models as Self-Improvers, 2024) |
| **Year** | 2007 (seed AI) → 2024 (Self-Rewarding / self-improving LMs) |
| **Venue** | LessWrong / arXiv:2401.10020 (Self-Rewarding); arXiv:2410.13735 (Self-Improvers) |
| **Key Idea** | **RSI (Recursive Self-Improvement)**: an AGI that can rewrite its own source/weights, run the improved version, measure improvement, and iterate — a positive-feedback loop ("seed AI") that could yield an intelligence explosion if each cycle yields a strictly smarter system. Modern grounding: **Self-Rewarding Language Models** (DPO with an LLM judge that the model itself plays) and **Self-Improvers** (generate synthetic instruction data, finetune, filter by self-evaluation, repeat) show measurable compounding gains over iterations on narrow tasks, with the critical open problem being **evaluation integrity** (the self-judge must not reward-hack / self-flatter) and **safety breaks** (halt if capability gain is unbounded or if the model tries to disable its verifier). |
| **Application to Transcender** | Transcender's `src/agi.cpp` implements the RSI loop (`self_modify`, `compile_and_test`, `verify`, `measure`, `simulate_step`, `plan`, `run_episode`, population-based training, `evaluate`) per Phase C tasks 54–65. The research-grounded design: (a) **Self-rewarding loop** — the RSI cycle should be `self_modify → compile_and_test → evaluate(on held-out benchmark, NOT self-judge) → if improved keep else rollback`, with `verify` comparing new vs old on a **fixed external benchmark** (the current task 56 makes `SelfVerifier` check overfit loss decrease + determinism, which is the right anti-reward-hack design); (b) **safety break** (task 65: 10-break) — cap iterations, halt if `measure` diverges or `self_modify` produces code that disables `verify`; (c) **rollback** — keep a snapshot of weights+code before each `self_modify` so a non-improving cycle reverts (the "if improved self_modify else rollback" branch in task 65). The Self-Rewarding DPO judge is a candidate for the `SelfReflector`/`SelfMonitor` stubs (entropy/confidence gating = anti-self-flattery). |

---

## Cross-Cutting Application Summary for Transcender

| Paper | Transcender Target File | Concrete Engine Improvement |
|-------|--------------------|----------------------------|
| FlashAttention-3 | `src/flash_attention.cpp`, FP8 path in `src/inference_opt.cpp` | FP8 two-stage residual accumulation; tile prefetch pipelining |
| PagedAttention / vLLM | `src/inference_opt.cpp` (`PagedAttention`, `PrefixCache`) | COW prefix sharing via paged refcounts; tiered pool for 1M ctx |
| Speculative Decoding | `src/inference_opt.cpp` (`SpeculativeDecoder`) | Adaptive γ; draft KV cache; draft = target shard via `ModelShard` |
| Switch v2 / MoE | `src/moe_variants.cpp` | z-loss aux term; capacity factor + overflow accounting |
| BitNet b1.58 | `include/quant/ste_quantizer.h`, `math_avx2.cpp` | Ternary add-only GEMM kernel; train-in-format for ternary MoE |
| DeepSeek MLA | `src/moe_variants.cpp` (MoE variant), `src/inference_opt.cpp` | Joint low-rank KV latent cache + decoupled RoPE |
| Mamba-2 | `src/moe_variants.cpp` (Mamba variant) | Chunkwise SSM with FlashAttention-style tiling |
| Griffin / RecurrentGemma | `src/transformer.cpp` (new block) | RG-LRU diagonal recurrent block for hybrid models |
| AGI / RSI | `src/agi.cpp` | External-benchmark verify, safety break, rollback (Phase C tasks 54–65) |
