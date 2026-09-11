# Transcender Architecture

> **Understanding the Design Philosophy and System Structure**
>
> **Production-hardening sync (2026-09-11):** version one-truth **1.1.0 / R0001.01**
> (`CMakeLists.txt:3`, `include/quant/version.h:6` — "0.2.0" in older banners is
> STALE); format one-truth **105 formats,
> `FORMAT_COUNT=105`** (`include/quant/types.h:67`, no TWI by design); index one-truth
> **TranscenderIDX** magic (`src/codec/quant_format.cpp:546,585-588`); tests **72 ctest
> cases** (`tests/CMakeLists.txt`; `ctest -N` = 72), **72/72 green 2026-09-11**; Cender is **future/planned** (deferred Phase 19-23,
> proof-frozen in `docs/THEOREM_CENDER.md` — no `Cender/` dir, no `.txn` IR in tree).

---

## 🎯 Overview

Transcender is designed as a **complete, self-contained AI engine** with the following core principles:

1. **Zero Dependencies** - Pure C++20, no external libraries required
2. **Single Format Truth** - The `.quant` format is the single source of truth for models
3. **Research-Driven** - Every design decision is backed by peer-reviewed research
4. **Performance-First** - Hand-optimized kernels, SIMD, and cache-aware implementations
5. **Modular** - Components are cleanly separated for maintainability and extensibility

---

## 🏗️ High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          Transcender                                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐  │
│  │      TOOLS      │    │     ENGINES     │    │      CORE       │  │
│  │                 │    │                 │    │                 │  │
│  │ • quant-convert   │    │ • Inference     │    │ • Tensor         │  │
│  │ • quant-train     │    │ • Trainer       │    │ • Autograd       │  │
│  │ • quant-infer     │    │                 │    │ • Math (AVX2)    │  │
│  │ • quant-finetune  │    │                 │    │ • Memory         │  │
│  │ • quant-info      │    │                 │    │ • Random         │  │
│  │ • quant-bench     │    │                 │    │ • Types          │  │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘  │
│                           │                                              │
│                           ▼                                              │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                        QUANT FORMAT (v3: 105 formats)                  │  │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐         │  │
│  │  │  Q-series  │ │  K-variants│ │  GRP (QG*) │ │ Mix (Q_MX*/ │         │  │
│  │  │  base 10   │ │  L/M/H ×27 │ │  exact +K_G │ │ QG_MX* ×14) │         │  │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘         │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                    SUPPORTED ARCHITECTURES                     │  │
│  │  • Dense Transformers                                        │  │
│  │  • Mixture of Experts (MoE)                                   │  │
│  │  • Multimodal Models (Text, Image, Video, Audio)              │  │
│  │  • Custom Architectures (Extensible)                          │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Source Layout Truth (Phase 24)

> Flat `src/*.cpp` paths elsewhere in this file are **STALE**. Measured layout:
> `src/{adapters,agi,backend,codec,core,gle,inference,kernel,math,model,multimodal,server,tokenizer,trainer}/`.
> Mapping for the files named below: `tensor.cpp`→`src/core/`; `math*.cpp`→`src/core/`+`src/math/`;
> `memory.cpp`,`random.cpp`→`src/core/`; `autograd*.cpp`→`src/model/`;
> `transformer.cpp`,`model.cpp`,`moe_*.cpp`,`kv_cache*.cpp`→`src/model/`;
> `backend.cpp`,`gpu_compute*.cpp`→`src/backend/`; `tokenizer`/`bpe`→`src/tokenizer/`;
> `sampler.cpp`,`generator.cpp`,`inference_*.cpp`→`src/inference/`;
> `trainer*.cpp`,`optimizer.cpp`,`finetune`→`src/trainer/`; `dataloader`→`src/core/dataset.cpp`;
> `block_codec.cpp`,`codebook.cpp`,`format_planner.cpp`,`format_registry.cpp`,`quant_format.cpp`,`ste_quantizer.cpp`,`quant_engines_*.cpp`→`src/codec/`;
> `kernel_*.cpp`,`int8_quant.cpp`,`flash_attention.cpp`→`src/kernel/`.

## 📦 Component Hierarchy

### 1. Core Layer (Foundation)

The foundation upon which everything else is built.

```
┌─────────────────────────────────────────────────────────────────┐
│                        CORE LAYER                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐            │
│  │   types.h    │  │  memory.h    │  │  random.h    │            │
│  │              │  │              │  │              │            │
│  │ • DType      │  │ • Allocator  │  │ • RNG        │            │
│  │ • Format     │  │ • Arena      │  │ • Distributions│         │
│  │ • Shape      │  │ • Pool       │  │              │            │
│  └──────────────┘  └──────────────┘  └──────────────┘            │
│                                                                     │
│  ┌──────────────┐  ┌──────────────┐                               │
│  │   tensor.h   │  │   math.h     │                               │
│  │   tensor.cpp │  │   math.cpp   │                               │
│  │              │  │   math_avx2.cpp│                               │
│  │ • Tensor     │  │ • Operations │                               │
│  │ • View       │  │ • GEMM       │                               │
│  │ • Storage    │  │ • Reductions │                               │
│  └──────────────┘  └──────────────┘                               │
│                                                                     │
└─────────────────────────────────────────────────────────────────┘
```

**Purpose:** Low-level primitives, memory management, and mathematical operations.

**Key Files:**
- `include/quant/types.h` - Data types, formats, shapes
- `include/quant/tensor.h` - Tensor class definition
- `src/tensor.cpp` - Tensor implementation
- `include/quant/math.h` - Math operations interface
- `src/math.cpp` - Math operations (scalar implementation)
- `src/math_avx2.cpp` - Math operations (AVX2 vectorized)
- `include/quant/memory.h` - Memory management
- `src/memory.cpp` - Memory implementation
- `include/quant/random.h` - Random number generation
- `src/random.cpp` - RNG implementation

---

### 2. Autograd Layer (Automatic Differentiation)

The engine that powers training through automatic gradient computation.

```
┌─────────────────────────────────────────────────────────────────┐
│                     AUTOGRAD LAYER                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │                    AutogradEngine                            │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │  │
│  │  │ Forward Pass │  │ Backward Pass│  │  Parameter    │      │  │
│  │  │              │  │              │  │  Management   │      │  │
│  │  │ • Operation  │  │ • Gradient   │  │  • Register   │      │  │
│  │  │   Recording  │  │   Computation│  │  • Track      │      │  │
│  │  │ • Computation│  │ • Chain Rule │  │  • Clear      │      │  │
│  │  │   Graph     │  │   Application│  │              │      │  │
│  │  └──────────────┘  └──────────────┘  └──────────────┘      │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │                    Operations                                  │  │
│  │  • matmul_op    • embedding_op    • cross_entropy_op        │  │
│  │  • add_op       • bias_add_op      • relu_op                │  │
│  │  • mul_op       • layer_norm_op    • softmax_op             │  │
│  │  • ...          • ...              • ...                   │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                                                                     │
└─────────────────────────────────────────────────────────────────┘
```

**Purpose:** Enable training by automatically computing gradients through computational graphs.

**Key Files:**
- `include/quant/autograd.h` - Autograd engine interface
- `src/autograd.cpp` - Autograd implementation

**Key Features:**
- Operation recording during forward pass
- Automatic gradient computation using chain rule
- Parameter management for trainable tensors
- Support for custom operations

---

### 3. Model Layer (Neural Network Components)

Building blocks for neural networks.

```
┌─────────────────────────────────────────────────────────────────┐
│                      MODEL LAYER                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐            │
│  │  transformer │  │   model      │  │  backend     │            │
│  │              │  │              │  │              │            │
│  │ • Embedding  │  │ • DenseModel │  │ • Config     │            │
│  │ • Attention  │  │ • MoEModel   │  │ • Device      │            │
│  │ • LayerNorm  │  │ • Load/Save  │  │ • Precision   │            │
│  │ • FeedForward│  │              │  │              │            │
│  │ • Block      │  │              │  │              │            │
│  └──────────────┘  └──────────────┘  └──────────────┘            │
│                                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐            │
│  │  tokenizer   │  │   sampler    │  │   kv_cache   │            │
│  │              │  │              │  │              │            │
│  │ • BPE        │  │ • Top-K      │  │ • Cache      │            │
│  │ • Encoding   │  │ • Top-P      │  │ • Management │            │
│  │ • Decoding   │  │ • Temperature │  │              │            │
│  └──────────────┘  └──────────────┘  └──────────────┘            │
│                                                                     │
└─────────────────────────────────────────────────────────────────┘
```

**Purpose:** Neural network components and model management.

**Key Files:**
- `include/quant/transformer.h` - Transformer architecture
- `src/transformer.cpp` - Transformer implementation
- `include/quant/model.h` - Model base class and implementations
- `src/model.cpp` - Model implementation
- `include/quant/backend.h` - Backend configuration
- `src/backend.cpp` - Backend implementation
- `include/quant/tokenizer.h` - Tokenizer interface
- `src/bpe_tokenizer.cpp` - BPE tokenizer implementation
- `include/quant/sampler.h` - Sampling strategies
- `src/sampler.cpp` - Sampler implementation
- `include/quant/kv_cache.h` - Key-Value cache
- `src/kv_cache.cpp` - KV cache implementation

---

### 4. Quantization Layer (Format-Specific Kernels)

Specialized kernels for different quantization formats.

```
┌─────────────────────────────────────────────────────────────────┐
│                   QUANTIZATION LAYER                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │                        FORMATS                                │  │
│  │                                                                 │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │  │
│  │  │   QUANT8       │  │   QUANT4       │  │  QG (GRP)    │      │  │
│  │  │              │  │              │  │              │      │  │
│  │  │ • 8-bit INT  │  │ • 4-bit INT  │  │ • 2-bit      │      │  │
│  │  │ • FP32 quality│  │ • FP16 quality│  │ • {-1,0,+1} │      │  │
│  │  │ • Codebook   │  │ • Codebook   │  │ • STE training│     │  │
│  │  └──────────────┘  └──────────────┘  └──────────────┘      │  │
│  │                                                                 │  │
│  │  ┌──────────────┐  ┌──────────────┐                          │  │
│  │  │   QG Mix       │  │   Mixed      │                          │  │
│  │  │              │  │              │                          │  │
│  │  │ • 1-bit      │  │ • Per-block │                          │  │
│  │  │ • {-1,+1}    │  │ • Format     │                          │  │
│  │  │ • XOR+popcnt │  │   allocation │                          │  │
│  │  └──────────────┘  └──────────────┘                          │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │                    KERNELS                                    │  │
│  │  • kernel_quant8.cpp    • kernel_q*.cpp (Q1/Q3/Q6/Q12/Q24 Tier-Lookup) │  │
│  │  • kernel_quant4.cpp    • int8_quant.cpp                       │  │
│  │  • ste_quantizer.cpp  • format_planner.cpp • codebook.cpp    │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                                                                     │
└─────────────────────────────────────────────────────────────────┘
```

**Purpose:** Efficient computation with various quantization formats.

**Key Files (new `src/` layout — flat `src/*.cpp` paths below are STALE):**
- `include/quant/kernel.h` - Kernel interface
- `src/kernel/kernel_quant8.cpp` - Q8 kernel implementation
- `src/kernel/kernel_quant4.cpp` - Q4 kernel implementation
- `src/kernel/kernel_q3.cpp`, `kernel_q6.cpp`, `kernel_q12.cpp`, `kernel_q24.cpp` - Q-series kernels
- `src/kernel/kernel_tl.cpp` - TL (Tier-Lookup) kernel for low-BPW tiers
- `include/quant/int8_quant.h` - INT8 quantization
- `src/kernel/int8_quant.cpp` - INT8 implementation
- `include/quant/ste_quantizer.h` - Straight-Through Estimator
- `src/codec/ste_quantizer.cpp` - STE implementation
- `include/quant/format_planner.h` - Format allocation planner
- `src/codec/format_planner.cpp` - Planner implementation
- `include/quant/codebook.h` - Vector quantization codebooks
- `src/codec/codebook.cpp` - Codebook implementation

---

### 5. MoE Layer (Mixture of Experts)

Implementation of Mixture of Experts architectures.

```
┌─────────────────────────────────────────────────────────────────┐
│                      MoE LAYER                                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │                    MoE Variants                               │  │
│  │                                                                 │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │  │
│  │  │  Dense MoE   │  │  Sparse MoE  │  │  Multimodal   │      │  │
│  │  │              │  │              │  │    MoE       │      │  │
│  │  │ • Top-K      │  │ • Top-1      │  │ • Modality    │      │  │
│  │  │   Routing   │  │   Routing   │  │   Experts    │      │  │
│  │  │ • All-to-All │  │ • Expert     │  │ • Cross-modal │      │  │
│  │  │   Communication││   Parallelism│  │   Attention  │      │  │
│  │  └──────────────┘  └──────────────┘  └──────────────┘      │  │
│  │                                                                 │  │
│  │  ┌─────────────────────────────────────────────────────────┐  │  │
│  │  │                    Routing Functions                        │  │  │
│  │  │  • softmax_with_topk()                                      │  │  │
│  │  │  • hash_token()                                             │  │  │
│  │  │  • compute_load_balance_loss()                              │  │  │
│  │  └─────────────────────────────────────────────────────────┘  │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                                                                     │
└─────────────────────────────────────────────────────────────────┘
```

**Purpose:** Implement various Mixture of Experts routing strategies.

**Key Files:**
- `include/quant/moe_variants.h` - MoE interface and variants
- `src/moe_variants.cpp` - MoE implementation

---

### 6. GPU Layer (Hardware Acceleration)

GPU compute acceleration. REAL-ONLY status (Phase 17 audit 2026-09-07,
evidence: `tests/test_gpu_capability.cpp`): no Q4/Q8 quantized GPU kernels
exist on any backend yet; every unavailable path fails loud (returns false /
nullptr / CPU fallback with `[WARN] ... perf-invalid`, never fake-available).

```
┌─────────────────────────────────────────────────────────────────┐
│                      GPU LAYER                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │                    GPU Compute                               │  │
│  │                                                                 │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │  │
│  │  │  Vulkan     │  │  GLSL/SPIRV  │  │   Buffers    │      │  │
│  │  │              │  │   Shaders    │  │              │      │  │
 │  │  │ • Device    │  │ • ReLU/elem  │  │ • Upload     │      │  │
 │  │  │ • Pipeline  │  │   (GEMM/GEMV │  │ • Readback   │      │  │
 │  │  │ • Command   │  │   = CPU fb)  │  │ • Storage    │      │  │
 │  │  │   Lists     │  │ • GELU       │  │              │      │  │
 │  │  └──────────────┘  └──────────────┘  └──────────────┘      │  │
 │  │                                                                 │  │
 │  │  ┌─────────────────────────────────────────────────────────┐  │  │
 │  │  │              Shaders / Kernels (audited 2026-09-07)            │  │  │
 │  │  │  REAL: SPIRV_RELU/GELU/SILU/ADD/MUL (Vulkan elementwise)       │  │  │
 │  │  │  REAL: FP32 PTX gemm/act/norm/attn (CUDA, needs HW to verify)  │  │  │
 │  │  │  FALLBACK (CPU + WARN, perf-invalid): Vulkan GEMM/GEMV/softmax │  │  │
 │  │  │    norm/MoE/attention; Metal rope/attention/reduce are EMPTY   │  │  │
 │  │  │  ABSENT everywhere: Q4/Q8 quantized GEMV/GEMM (L080 retired)   │  │  │
 │  │  └─────────────────────────────────────────────────────────┘  │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                                                                     │
└─────────────────────────────────────────────────────────────────┘
```

**Purpose:** Accelerate computations using GPU.

**Key Files:**
- `include/quant/gpu_compute.h` - GPU compute interface
- `src/backend/gpu_compute.cpp` - GPU implementation (DirectX 12)
- `src/backend/gpu_compute_vulkan.cpp` - Vulkan GPU backend
- `src/backend/gpu_compute_cuda.cpp` - CUDA backend (dynamic driver load, FP32 PTX)
- `src/backend/gpu_compute_metal.cpp` - Metal backend (Apple-only, FP32 MSL)
- `tests/test_gpu_capability.cpp` - REAL-ONLY capability probe (L080-L085 evidence)

**Capability table (this machine, 2026-09-07 — REAL ONLY, no fake):**

| Backend | Available here | FP32 kernels | Q4/Q8 quant kernels | Fallback honesty |
|---|---|---|---|---|
| CUDA | NO (nvcuda.dll absent, no nvidia-smi/nvcc) | PTX present, unverifiable here | ABSENT (L080 retired) | fail-loud: init=false, alloc=nullptr, pin=false |
| Metal | NO (non-Apple, by construction) | MSL FP32-only; rope/attn/reduce EMPTY | ABSENT (L082 retired) | fail-loud: init=false, alloc=nullptr |
| Vulkan | LOADER present, device unprobed w/o build | SPIR-V relu/gelu/silu/add/mul REAL | ABSENT | CPU fallback + WARN, `is_initialized()==false` (L083 partial) |
| DirectX12 | Probable (Windows) — out of Phase 17 scope | existing path untouched | ABSENT | `is_directx_available()` DLL probe |

> L081 (pinned/prefetch): API REAL (`register_host_memory`, streams,
> `async_upload`, `ExpertPrefetcher`), overlap-timing evidence BLOCKED (no GPU).
> L085 (GPU bench charts): BLOCKED — no device, no measured numbers claimed.

---

### 7. Training Layer

Training infrastructure and optimization.

```
┌─────────────────────────────────────────────────────────────────┐
│                     TRAINING LAYER                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐            │
│  │  trainer     │  │  optimizer   │  │  dataloader  │            │
│  │              │  │              │  │              │            │
│  │ • fit()      │  │ • AdamW      │  │ • next_batch()│            │
│  │ • train_step()│  │ • SGD        │  │ • shuffle()  │            │
│  │ • save/load  │  │ • zero_grad()│  │ • reset()    │            │
│  │   checkpoint │  │ • step()     │  │              │            │
│  └──────────────┘  └──────────────┘  └──────────────┘            │
│                                                                     │
│  ┌──────────────┐  ┌──────────────┐                              │
│  │ finetune     │  │  ste_quant   │                              │
│  │              │  │              │                              │
│  │ • QUANT-Rank   │  │ • STE for    │                              │
│  │ • Low-rank Δ │  │   quantization│                              │
│  │ • Full FT    │  │ • Gradient   │                              │
│  │              │  │   estimation │                              │
│  └──────────────┘  └──────────────┘                              │
│                                                                     │
└─────────────────────────────────────────────────────────────────┘
```

**Purpose:** Training loops, optimization, and fine-tuning.

**Key Files:**
- `include/quant/trainer.h` - Trainer interface
- `src/trainer.cpp` - Trainer implementation
- `include/quant/optimizer.h` - Optimizer interface
- `src/optimizer.cpp` - Optimizer implementation
- `include/quant/finetune.h` - Fine-tuning interface
- `src/finetune.cpp` - Fine-tuning implementation
- `src/dataloader.cpp` - Data loading utilities

---

### 8. QUANT Format Layer

The single binary format for all model data.

```
┌─────────────────────────────────────────────────────────────────┐
│                    QUANT FORMAT LAYER                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │                    QUANT Format Spec                            │  │
│  │                                                                 │  │
│  │  Header (Magic: "TranscenderIDX", Version, Flags)                     │  │
│  │  ┌─────────────────────────────────────────────────────────┐  │  │
│  │  │  Metadata: Model type, dimensions, formats, etc.          │  │  │
│  │  └─────────────────────────────────────────────────────────┘  │  │
│  │                                                                 │  │
│  │  ┌─────────────────────────────────────────────────────────┐  │  │
│  │  │  Weight Blocks (Mixed formats)                            │  │  │
│  │  │  ┌──────────┐ ┌──────────┐ ┌──────────┐                 │  │  │
│  │  │  │ QUANT8     │ │ QUANT4     │ │ Q-series │ ...             │  │  │
│  │  │  │          │ │          │ │          │                 │  │  │
│  │  │  └──────────┘ └──────────┘ └──────────┘                 │  │  │
│  │  └─────────────────────────────────────────────────────────┘  │  │
│  │                                                                 │  │
│  │  ┌─────────────────────────────────────────────────────────┐  │  │
│  │  │  Codebooks (For QUANT8/QUANT4)                                │  │  │
│  │  └─────────────────────────────────────────────────────────┘  │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  Operations: Load, Save, Convert, Validate, Optimize                │
│                                                                     │
└─────────────────────────────────────────────────────────────────┘
```

**Purpose:** Single binary format for model storage and exchange.

**Key Files:**
- `include/quant/quant_format.h` - QUANT format specification (TranscenderIDX magic header)
- `src/codec/quant_format.cpp` - QUANT format implementation

---

### 9. Tools Layer (CLI)

Command-line tools for various operations.

```
┌─────────────────────────────────────────────────────────────────┐
│                      TOOLS LAYER                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Available Tools:                                                   │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │  • quant_infer (tools/infer.cpp)      - Run inference with a model         │  │
│  │  • quant_train (tools/train.cpp)      - Train a model from scratch          │  │
│  │  • quant_finetune (tools/finetune.cpp)- Fine-tune an existing model         │  │
│  │  • quant_convert (tools/convert.cpp)  - Convert models to/from QUANT format │  │
│  │  • quant_info (tools/info.cpp)        - Display model information           │  │
│  │  • quant_bench (tools/bench.cpp)      - Run performance benchmarks          │  │
│  │  • + quant_quantize, quant_serve, quant_server, quant_evaluate,             │  │
│  │    quant_format_list, generate_comparison_visuals (CMakeLists.txt:344-393) │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  Each tool:                                                         │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │  • CLI interface with arguments                                  │  │
│  │  • Configuration file support (JSON)                           │  │
│  │  • Progress reporting                                           │  │
│  │  • Error handling and validation                                │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                                                                     │
└─────────────────────────────────────────────────────────────────┘
```

**Purpose:** User-facing command-line interfaces.

**Key Files:**
- `tools/convert.cpp` - Model conversion tool
- `tools/train.cpp` - Training tool
- `tools/infer.cpp` - Inference tool
- `tools/finetune.cpp` - Fine-tuning tool
- `tools/info.cpp` - Model info tool
- `tools/bench.cpp` - Benchmarking tool

---

## 🔗 Component Dependencies

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DEPENDENCY GRAPH                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  TOOLS (CLI)                                                             │
│       ↓                                                                │
│  ENGINES (Inference, Trainer)                                           │
│       ↓                                                                │
│  CORE (Tensor, Math, Memory) ────┐                                     │
│       ↓                            ↓                                 │
│  AUTOGRAD ────────────────────── MODEL (Transformer, etc.)          │
│       ↓                            ↓                                 │
│  QUANTIZATION (Kernels, Codebooks)       TOKENIZER                    │
│       ↓                                                                │
│  QUANT FORMAT                                                        │
│       ↓                                                                │
│  GPU COMPUTE (Optional, for acceleration)                            │
│                                                                         │
│  TYPES (Foundation - No dependencies)                                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🎯 Design Philosophy

### 1. Zero Dependencies

**Why?** 
- No version conflicts
- No bloated installations
- Complete control over every line of code
- Easier deployment and distribution

**How?**
- Pure C++20 standard library
- Hand-written SIMD (AVX2)
- Custom implementations of everything

### 2. Single Format Truth

**Why?**
- Eliminates format conversion overhead
- Ensures consistency across training and inference
- Simplifies the ecosystem

**How?**
- The `.quant` format handles everything:
  - Model weights (mixed formats)
  - Architecture configuration
  - Tokenizer data
  - Training metadata

### 3. Research-Driven

**Why?**
- Ensures we're using proven techniques
- Avoids reinventing the wheel
- Provides theoretical guarantees

**How?**
- Every major design decision references peer-reviewed papers
- Implementation follows published algorithms
- Performance claims are backed by research

### 4. Performance-First

**Why?**
- AI workloads are computationally intensive
- Every optimization matters at scale
- Users expect fast performance

**How?**
- Hand-optimized kernels
- SIMD vectorization (AVX2)
- Cache-aware data layouts
- GPU acceleration (Vulkan)
- Mixed-precision computation

### 5. Modularity

**Why?**
- Easier to maintain
- Easier to test
- Easier to extend
- Easier for others to contribute

**How?**
- Clear separation of concerns
- Minimal coupling between components
- Well-defined interfaces
- Single responsibility principle

---

## 📊 Performance Characteristics

| Component | Typical Performance | Optimization Techniques |
|-----------|---------------------|-------------------------|
| GEMM (FP32) | ~2-4 GFLOPS/core | AVX2, 6x16 tiling, loop unrolling |
| GEMM (INT8) | ~8-16 GIPS/core | AVX2, quantization |
| Attention | ~1-2x GEMM speed | Fused kernels, memory efficient |
| Token Generation | ~50-100 tok/s | KV cache, efficient sampling |

---

## 🚀 Scalability

Transcender is designed to scale from:

- **Tiny models** (Millions of parameters) - Runs on CPU, great for testing
- **Medium models** (Billions of parameters) - Runs on consumer GPUs
- **Large models** (Trillions of parameters) - Designed for distributed training (future)

### Current Limits:
- **Max tokens**: 2^31-1 (limited by int32)
- **Max parameters**: ~2^63 (limited by int64)
- **Max batch size**: Memory-dependent
- **Max sequence length**: Memory-dependent

### Future Scalability Features:
- Distributed training
- Model parallelism
- Pipeline parallelism
- Tensor parallelism
- Multi-GPU support
- Multi-node support

---

## 🔍 Debugging & Development

### Debug Builds
```bash
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake --build build --parallel
```

### Sanitizers
```bash
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug -DQUANT_SANITIZE=ON
cmake --build build --parallel
```

### Logging
- Use `std::cout` for debug output (temporary)
- Consider adding a proper logging library (future)

### Testing
```bash
ctest --test-dir build --output-on-failure -j$(nproc)
```

---

## 📝 Version History

| Version | Date | Changes |
|---------|------|---------|
| v0.1 | July 2026 | Initial release - Core engine, QUANT format, basic tools |
| v0.1.02 | July 26, 2026 | 82 targets (25 libs + 25 executables + 32 tests), Vulkan GPU backend, 29 QUANT formats, 47 claims |
| v0.2 (Planned) | - | Vision module, improved MoE, more tools |

---

## 🎓 Learning Resources

To understand Transcender better, study these topics:

1. **C++20 Features**
   - Concepts
   - Modules
   - Ranges
   - Coroutines
   - Span
   - Format

2. **SIMD Programming**
   - AVX2 intrinsics
   - Vectorization patterns
   - Cache optimization

3. **Neural Networks**
   - Transformers
   - Attention mechanisms
   - Layer normalization
   - Feed-forward networks

4. **Quantization**
   - Uniform quantization
   - Non-uniform quantization
   - Product quantization
   - Vector quantization

5. **Automatic Differentiation**
   - Forward mode
   - Reverse mode
   - Computational graphs
   - Chain rule

6. **Mixture of Experts**
   - Sparse MoE
   - Dense MoE
   - Routing strategies
   - Load balancing

---

## 🔮 Cender — Future Architecture Core (NOT in tree)

> **Phase 24 truth:** there is **no `Cender/` directory, no `.txn` IR implementation, no
> Cender code** in this repo. Cender (Transformer replacement: constant-memory
> `state_{t+1} = decay·state_t + k_t ⊗ err_t`, solving Context Rot / Lost-In-The-Middle
> by construction) is **deferred to Phase 19-23 behind an owner gate** and its proof is
> frozen in `docs/THEOREM_CENDER.md` (DRAFT, Gauntlet bar: DeepSeek-V4 MLA + DeltaNet +
> Titans). Current engine remains Transformer-based (`src/model/transformer.cpp:300`
> `softmax(Q·K^T/√d)·V`, KV-cache `src/model/kv_cache.cpp`). Any Cender progress claim
> without new code+bench is **UNVERIFIED**.

## 📞 Need More Information?

- See **[MODULES/](MODULES/)** for detailed module documentation
- See **[RESEARCH.md](RESEARCH.md)** for research papers and references
- See **[INTERNAL/](INTERNAL/)** for internal design documents
- See **[wiki/Architecture.md](../wiki/Architecture.md)** for architecture overview in wiki format
- See **[wiki/files/](../wiki/files/_index.md)** for per-file source documentation
- Check the source code - it's well-commented!

---

*Last updated: September 7, 2026 (Phase 24 docs-only sync; no build, no code touched)*
