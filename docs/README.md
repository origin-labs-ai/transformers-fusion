# Transcender Documentation

> **M**ixed-format **Y**our-own **T**ensor **H**andcrafted **O**ptimized **S**ystem

**Zero-dependency C++20 AI Engine** - Train from scratch, fine-tune, quantize, and run inference, all within a single `.quant` binary format.

---

## 📚 Documentation Overview

Welcome to the comprehensive documentation for Transcender. This documentation is structured to help you understand, use, and contribute to the project.

### 🗂️ Documentation Structure

```
docs/
├── README.md                  # This file - Documentation index
├── ARCHITECTURE.md            # System architecture & design philosophy
├── BUILD.md                   # Build & installation guide
├── USAGE.md                   # Usage guide & examples
├── API_REFERENCE.md           # Complete API documentation
├── MODULES/                   # Per-module deep dives
│   ├── tensor.md
│   ├── autograd.md
│   ├── transformer.md
│   ├── quantization.md
│   ├── moe.md
│   ├── gpu_compute.md
│   └── ...
├── RESEARCH.md                # Research foundation & papers
├── CONTRIBUTING.md           # Contribution guidelines
├── INTERNAL/                  # Internal design documents
│   ├── memory_management.md
│   ├── kernel_optimization.md
│   └── format_design.md
└── EXAMPLES/                  # Code examples & tutorials
    ├── basic_inference.md
    ├── training_guide.md
    └── quantization_guide.md

wiki/                          # Per-file documentation (repo-wiki style)
├── Home.md                    # Wiki home page
├── Architecture.md            # Architecture deep-dive
├── Build-Guide.md             # Build instructions
├── Usage-Guide.md             # Usage with examples
├── Api-Reference.md           # Full API reference
├── QUANT-Format.md              # QUANT binary format spec
├── Training.md                # Training guide
├── Inference.md               # Inference guide
├── Research.md                # Research foundation
├── Contributing.md            # Contribution guide
├── Modules.md                 # Module index
├── _Sidebar.md                # Wiki navigation sidebar
└── files/                     # Per-file detailed documentation (91 files)
    ├── _index.md              # File docs index
    ├── types.h.md, tensor.h.md, model.h.md, ...
    ├── tensor.cpp.md, math.cpp.md, transformer.cpp.md, ...
    ├── engine-inference.cpp.md, engine-quant8-quantize.cpp.md, ...
    └── tool-convert.cpp.md, test-tensor.cpp.md, ...
```

---

## 🚀 Quick Start

If you're new to Transcender, start here:

1. **[BUILD.md](BUILD.md)** - Get the project compiled on your machine
2. **[USAGE.md](USAGE.md)** - Learn how to use the tools and APIs
3. **[ARCHITECTURE.md](ARCHITECTURE.md)** - Understand how everything fits together

---

## 📖 Core Documentation

| Document | Description | Audience |
|----------|-------------|----------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | High-level system design, component relationships | Developers, Architects |
| [API_REFERENCE.md](API_REFERENCE.md) | Complete API documentation for all public interfaces | Developers, Users |
| [BUILD.md](BUILD.md) | Step-by-step build instructions for all platforms | Everyone |
| [USAGE.md](USAGE.md) | How to use Transcender for inference and training | Users |
| [RESEARCH.md](RESEARCH.md) | Research papers and algorithms that inspired the design | Researchers |

---

## 📄 Per-File Documentation

For **detailed per-file documentation** covering every header, source, engine, tool, and test file in the codebase, see the **[wiki/files/](../wiki/files/_index.md)** directory. Each file has its own markdown document explaining:

- **Purpose & responsibilities** — What the file does and why it exists
- **Key types & functions** — Important APIs defined in the file
- **Implementation details** — How the code works internally
- **Dependencies** — What other files/modules it depends on

> Browse the full index at **[wiki/files/_index.md](../wiki/files/_index.md)** or start with the **[wiki/Home.md](../wiki/Home.md)** for a guided tour.

---

## 🏗️ Module Documentation

Detailed documentation for each major component:

| Module | Description | Files |
|--------|-------------|-------|
| **Tensor** | Multi-dimensional array with automatic differentiation support | `tensor.h`, `tensor.cpp` |
| **Autograd** | Automatic differentiation engine for training | `autograd.h`, `autograd.cpp` |
| **Math** | Mathematical operations with AVX2 optimization | `math.h`, `math.cpp`, `math_avx2.cpp` |
| **Transformer** | Transformer architecture implementation | `transformer.h`, `transformer.cpp` |
| **Quantization** | QUANT4, QUANT8, Ternary, Binary formats | `kernel_quant4.cpp`, `kernel_quant8.cpp`, `int8_quant.cpp`, etc. |
| **MoE** | Mixture of Experts implementation | `moe_variants.h`, `moe_variants.cpp` |
| **GPU Compute** | Vulkan GPU acceleration | `gpu_compute.h`, `gpu_compute.cpp`, `gpu_compute_vulkan.cpp` |
| **Model** | Model loading, saving, and management | `model.h`, `model.cpp` |
| **Tokenizer** | BPE tokenizer implementation | `bpe_tokenizer.h`, `bpe_tokenizer.cpp` |
| **Trainer** | Training loop and optimization | `trainer.h`, `trainer.cpp` |
| **QUANT Format** | Binary format specification | `quant_format.h`, `quant_format.cpp` |

See **[MODULES/](MODULES/)** for detailed documentation on each component.

---

## 🔬 Research & Innovation

Transcender is built on a foundation of peer-reviewed research:

- **BitNet b1.58** - Ternary weights that match FP16 quality
- **AWQ** - Activation-aware weight quantization
- **VQ-VAE** - Vector quantization with learned codebooks
- **BitsMoE** - Per-expert bit-width allocation

See **[RESEARCH.md](RESEARCH.md)** for a complete list of research papers and how they're implemented.

---

## 🛠️ Development

Want to contribute or extend Transcender?

- **[CONTRIBUTING.md](CONTRIBUTING.md)** - Guidelines for contributing
- **[INTERNAL/](INTERNAL/)** - Internal design documents
- **[EXAMPLES/](EXAMPLES/)** - Code examples and tutorials

---

## 📞 Need Help?

- Found a bug? Open an issue on GitHub
- Have a question? Check the documentation or open a discussion
- Want to contribute? See [CONTRIBUTING.md](CONTRIBUTING.md)

---

## 🎯 Project Status

| Component | Status | Tests | Documentation |
|-----------|--------|-------|---------------|
| Core Tensor | ✅ Stable | ✅ | ✅ |
| Autograd Engine | ✅ Stable | ✅ | ✅ |
| Transformer | ✅ Stable | ✅ | ✅ |
| QUANT Format | ✅ Stable | ✅ | ✅ |
| Quantization Kernels | ✅ Stable | ✅ | ⏳ |
| MoE Variants | ✅ Stable | ✅ | ⏳ |
| GPU Compute | ✅ Stable (Vulkan) | ✅ | ⏳ |
| Training | ✅ Stable | ✅ | ⏳ |
| Inference | ✅ Stable | ✅ | ⏳ |

**Last Updated:** September 13, 2026
**Version:** v1.1.0 / R0001.01 (was v0.1.02 — stale; one-truth: `CMakeLists.txt:3`, `include/quant/version.h:6`)

---

## 📄 License

Transcender is an open-source project licensed under the Apache License 2.0. See the [LICENSE](../LICENSE) and the main [README](../README.md) for details.


© 2026 Transcender Contributors
