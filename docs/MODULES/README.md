# Module Documentation

> **Detailed Documentation for Each Transcender Module**

---

## 📚 Available Module Documentation

This directory contains detailed documentation for each major component of Transcender. Each module document covers:

- Overview and purpose
- Key features
- API reference
- Implementation details
- Best practices
- Common pitfalls
- Examples

---

## 🗂️ Module List

### Core Modules

| Module | Description | Files |
|--------|-------------|-------|
| **[Tensor](tensor.md)** | Multi-dimensional array with autograd support | `tensor.h`, `tensor.cpp` |
| **[Autograd](autograd.md)** | Automatic differentiation engine | `autograd.h`, `autograd.cpp` |
| **[Math](math.md)** | Mathematical operations with AVX2 optimization | `math.h`, `math.cpp`, `math_avx2.cpp` |
| **[Types](types.md)** | Core types, formats, and data structures | `types.h` |
| **[Memory](memory.md)** | Memory management and allocation | `memory.h`, `memory.cpp` |

### Model Modules

| Module | Description | Files |
|--------|-------------|-------|
| **[Transformer](transformer.md)** | Transformer architecture implementation | `transformer.h`, `transformer.cpp` |
| **[Model](model.md)** | Base model class and model management | `model.h`, `model.cpp` |
| **[Backend](backend.md)** | Backend configuration and device management | `backend.h`, `backend.cpp` |

### Quantization Modules

| Module | Description | Files |
|--------|-------------|-------|
| **[Quantization](quantization.md)** | Quantization kernels and formats | `kernel_*.cpp`, `int8_quant.cpp`, etc. |
| **[QUANT Format](quant_format.md)** | QUANT binary format specification | `quant_format.h`, `quant_format.cpp` |
| **[Codebook](codebook.md)** | Vector quantization codebooks | `codebook.h`, `codebook.cpp` |
| **[STE Quantizer](ste_quantizer.md)** | Straight-Through Estimator for training | `ste_quantizer.h`, `ste_quantizer.cpp` |
| **[Format Planner](format_planner.md)** | Importance-based format allocation | `format_planner.h`, `format_planner.cpp` |

### Training Modules

| Module | Description | Files |
|--------|-------------|-------|
| **[Trainer](trainer.md)** | Training loop and infrastructure | `trainer.h`, `trainer.cpp` |
| **[Optimizer](optimizer.md)** | Optimization algorithms (SGD, Adam, AdamW) | `optimizer.h`, `optimizer.cpp` |
| **[Finetune](finetune.md)** | Fine-tuning implementation | `finetune.h`, `finetune.cpp` |

### Inference Modules

| Module | Description | Files |
|--------|-------------|-------|
| **[Tokenizer](tokenizer.md)** | BPE tokenizer implementation | `tokenizer.h`, `bpe_tokenizer.cpp` |
| **[Sampler](sampler.md)** | Token sampling strategies | `sampler.h`, `sampler.cpp` |
| **[KV Cache](kv_cache.md)** | Key-Value cache for attention | `kv_cache.h`, `kv_cache.cpp` |

### Advanced Modules

| Module | Description | Files |
|--------|-------------|-------|
| **[MoE Variants](moe.md)** | Mixture of Experts implementation | `moe_variants.h`, `moe_variants.cpp` |
| **[GPU Compute](gpu_compute.md)** | DirectX 12 GPU acceleration | `gpu_compute.h`, `gpu_compute.cpp` |

---

## 🎯 Navigation Guide

### For Beginners

Start with these modules to understand the foundation:

1. **[Tensor](tensor.md)** - The fundamental data structure
2. **[Types](types.md)** - Core types and enums
3. **[Memory](memory.md)** - Memory management
4. **[Math](math.md)** - Mathematical operations

### For Intermediate Users

Understand the model implementation:

1. **[Transformer](transformer.md)** - Transformer architecture
2. **[Model](model.md)** - Model management
3. **[Tokenizer](tokenizer.md)** - Tokenization
4. **[Sampler](sampler.md)** - Sampling strategies

### For Advanced Users

Dive into optimization and specialized features:

1. **[Autograd](autograd.md)** - Automatic differentiation
2. **[Quantization](quantization.md)** - Quantization techniques
3. **[QUANT Format](quant_format.md)** - `.quant` container format (TranscenderIDX index magic; v3 Q-series, 105 formats — `include/quant/types.h:67`)
4. **[Trainer](trainer.md)** - Training infrastructure
5. **[MoE Variants](moe.md)** - Mixture of Experts
6. **[GPU Compute](gpu_compute.md)** - GPU acceleration

---

## 🔗 Module Dependencies

```
┌─────────────────────────────────────────────────────────────────┐
│                        MODULE DEPENDENCIES                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐        │
│  │   Types      │    │   Memory     │    │    Math      │        │
│  │              │    │              │    │              │        │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘        │
│         │                  │                  │                │
│         ▼                  ▼                  ▼                │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                        Tensor                               │    │
│  └───────────────────────────────┬───────────────────────────┘    │
│                                  │                                │
│          ┌───────────────────────┼───────────────────────┐      │
│          │                       │                       │      │
│          ▼                       ▼                       ▼      │
│  ┌──────────────┐       ┌──────────────┐       ┌────────┐   │
│  │  Autograd    │       │  Transformer │       │ Model  │   │
│  └──────┬───────┘       └──────┬───────┘       └────┬───┘   │
│         │                      │                    │         │
│         ▼                      ▼                    ▼         │
│  ┌──────────────┐       ┌──────────────┐   ┌──────────────┐  │
│  │   Trainer    │       │    MoE       │   │   Backend    │  │
│  └──────┬───────┘       └──────────────┘   └──────────────┘  │
│         │                                                        │
│         ▼                                                        │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │  Optimizer   │    │   Tokenizer  │    │     KV      │      │
│  └──────────────┘    │              │    │    Cache    │      │
│                     └──────────────┘    └──────────────┘      │
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │   Quant      │    │  QUANT Format  │    │  GPU Compute │      │
│  │  Kernels     │    │              │    │              │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│                                                                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📖 Documentation Status

| Module | Documentation | Wiki File Docs | Priority |
|--------|--------------|---------------|----------|
| Tensor | ✅ Complete | [tensor.h.md](../../wiki/files/tensor.h.md) | High |
| Types | ✅ Complete | [types.h.md](../../wiki/files/types.h.md) | Medium |
| Memory | ✅ Complete | [memory.h.md](../../wiki/files/memory.h.md) | Medium |
| Math | ✅ Complete | [math.h.md](../../wiki/files/math.h.md) | Medium |
| Autograd | ✅ Complete | [autograd.cpp.md](../../wiki/files/autograd.cpp.md) | Medium |
| Transformer | ✅ Complete | [transformer.h.md](../../wiki/files/transformer.h.md) | Medium |
| Model | ✅ Complete | [model.h.md](../../wiki/files/model.h.md) | Medium |
| Backend | ✅ Complete | [backend.h.md](../../wiki/files/backend.h.md) | Low |
| Quantization | ✅ Complete | [kernel_tl.cpp.md](../../wiki/files/kernel_tl.cpp.md) | Medium |
| QUANT Format | ✅ Complete | [quant_format.h.md](../../wiki/files/quant_format.h.md) | Medium |
| Codebook | ✅ Complete | [codebook.h.md](../../wiki/files/codebook.h.md) | Low |
| STE Quantizer | ✅ Complete | [ste_quantizer.h.md](../../wiki/files/ste_quantizer.h.md) | Low |
| Format Planner | ✅ Complete | [format_planner.h.md](../../wiki/files/format_planner.h.md) | Low |
| Trainer | ✅ Complete | [trainer.h.md](../../wiki/files/trainer.h.md) | Medium |
| Optimizer | ✅ Complete | [optimizer.h.md](../../wiki/files/optimizer.h.md) | Medium |
| Finetune | ✅ Complete | [finetune.h.md](../../wiki/files/finetune.h.md) | Low |
| Tokenizer | ✅ Complete | [tokenizer.h.md](../../wiki/files/tokenizer.h.md) | Medium |
| Sampler | ✅ Complete | [sampler.h.md](../../wiki/files/sampler.h.md) | Medium |
| KV Cache | ✅ Complete | [kv_cache.h.md](../../wiki/files/kv_cache.h.md) | Low |
| MoE Variants | ✅ Complete | [moe_variants.h.md](../../wiki/files/moe_variants.h.md) | Medium |
| GPU Compute | ✅ Complete | [gpu_compute.h.md](../../wiki/files/gpu_compute.h.md) | Medium |

---

---

## 📄 Per-File Wiki Documentation

Every source file in the Transcender codebase has a dedicated documentation page in the **[wiki/files/](../../wiki/files/)** directory. These provide:

- **Line-by-line insights** into each file's purpose and logic
- **Key types & functions** defined in each file
- **Dependencies** and relationships between files
- **Build & usage notes** specific to each component

Browse the full index at **[wiki/files/_index.md](../../wiki/files/_index.md)**.

---

## 🛠️ How to Use This Documentation

### For Understanding a Module

1. Read the module's README (e.g., [tensor.md](tensor.md))
2. Check the API reference in the document
3. Look at the per-file wiki docs in [wiki/files/](../../wiki/files/) for implementation details
4. Run the tests to see usage examples

### For Contributing to a Module

1. Read the existing documentation
2. Follow the patterns established in the code
3. Update documentation when adding new features
4. Add tests for new functionality

### For Debugging

1. Check the "Common Pitfalls" section of the relevant module
2. Use the debug output examples
3. Run with sanitizers enabled

---

## 🎓 Learning Path

### Path 1: Using Transcender for Inference

1. Read **[Tensor](tensor.md)** documentation
2. Read **[Model](model.md)** documentation
3. Read **[Tokenizer](tokenizer.md)** documentation
4. Read **[Usage Guide](../USAGE.md)**
5. Try the CLI tools

### Path 2: Training Models

1. Complete Path 1
2. Read **[Autograd](autograd.md)** documentation
3. Read **[Trainer](trainer.md)** documentation
4. Read **[Optimizer](optimizer.md)** documentation
5. Try training a small model

### Path 3: Understanding Quantization

1. Read **[Types](types.md)** documentation
2. Read **[Quantization](quantization.md)** documentation
3. Read **[QUANT Format](quant_format.md)** documentation
4. Read **[Research Foundation](../RESEARCH.md)**
5. Experiment with different BPW settings

### Path 4: Contributing

1. Complete Path 1 and 2
2. Read **[Contributing Guide](../CONTRIBUTING.md)**
3. Read **[Architecture](../ARCHITECTURE.md)**
4. Read the source code
5. Start with a "good first issue"

---

## 📞 Need More Information?

- Check the **[API Reference](../API_REFERENCE.md)** for complete API documentation
- Check the **[Architecture](../ARCHITECTURE.md)** for system overview
- Check the **[Research Foundation](../RESEARCH.md)** for underlying science
- Check the source code - it's well-commented!
- Ask in GitHub Discussions or open an Issue

---

*Last updated: July 12, 2026*
