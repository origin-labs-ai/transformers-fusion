# Changelog

All notable changes to InNova will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.02] - 2026-07-26

### Added
- SOPS (Synchronized Optimized Precision Scheduler) with priority queue, cache-line aligned pools, and telemetry
- SOPS bench, format info, and training simulation executables
- Format registry quality heuristics for automatic format selection
- Forced distribution rule enforcing 2-mix/4-mix balance
- ContinualTrainer for incremental learning without full retraining
- DeltaAdapterHost for delta weight application and adapter management
- MTPHeadTrainer for Multi-Token Prediction head training
- ZeRO optimizer (Stage 1/2/3) with offload support
- Pipeline parallelism for distributed training
- HPO/NAS (hyperparameter optimization and neural architecture search)
- RL trainer with reward modeling
- MoE advanced support v1 and v2
- MoMMoE blocks for Mixture-of-Moments
- Native QUANT MoE integration
- Expert parallelism for MoE training
- Multimodal cross-attention and fusion modules
- Production HTTP server with JSON API
- BPE tokenizer advanced Unicode support
- `quant_format_list` CLI tool for listing all registered formats
- `quant_evaluate` CLI tool for model evaluation
- `bench_speed` training speed benchmark
- `bench_multimodal` multimodal benchmark
- `bench_awq_gptq` AWQ/GPTQ comparison benchmark
- `bench_gpt2_inference` GPT-2 real inference quality test
- Code coverage support (`QUANT_COVERAGE` CMake option with lcov/genhtml)
- iGPU zero-copy training via Vulkan unified memory (C-046 PROVEN)
- Out-of-core training via mmap data loader (C-047 PROVEN)
- Linux CI/CD pipeline with Ubuntu + Clang matrix (C-048 PROVEN)
- GitHub Actions build.yml with Windows+Ubuntu+Clang matrix
- Code signing for all 60+ binaries (Authenticode, cert: Satyam Thakur)
- 128-page research whitepaper (PDF, ReportLab-generated)
- constants.h — 80+ named compile-time constants

### Fixed
- `autograd_functions.h`: `_mm256_exp_ps` undeclared — replaced with cross-platform scalar polynomial approximation (clang-cl compatible)
- `gpu_compute_vulkan.cpp`: `VkBool32`, `VK_MAKE_VERSION` undeclared — proper Vulkan type definitions for dynamic loading
- Thread safety: 10 fixes across autograd_engine (mutex for gradient accumulation), MemoryPool and StackAllocator (atomic counters), RingAllReduce and ParameterServer (lock-free operations)
- `CMakeLists.txt`: All `GLOB_RECURSE` removed, every source file explicitly listed
- Duplicate symbols (LNK4006): 7 fixes applied across multiple source files
- LNK4006 duplicate symbols: Removed duplicate factory functions from gpu_compute.cpp, deleted redundant bpe_tokenizer.cpp
- 30 compiler warnings across 14 files → all explicit casts added

### Removed
- LoRA adapter support (adapter_edition/)
- QLoRA adapter support (adapter_edition/)
- DoRA adapter support (adapter_edition/)
- Orphan test files (tests_not_shipped constraint enforced)
- GGUF import adapter (`src/adapters/gguf_import.cpp`)
- Safetensors import adapter (`src/adapters/safetensors.cpp`)

### Changed
- `adapter_edition/` refactored from LoRA/QLoRA/DoRA host to native format converter (industrial formats to QUANT) and native trainer
- Format registry expanded to 12 single formats, 13 twi-mix variants, 4 four-mix variants (29 total)
- Build system: 82 targets (25 libs + 25 executables + 32 tests)
- Claims ledger: 47 total (46 proven + 1 pending)
- README.md, wiki/, docs/ all updated for v0.1.02
- Version bumped to 0.1.02 in `CMakeLists.txt` and `include/quant/version.h`

---

## [0.1.01] - 2026-07-24

### Added
- Initial public release of InNova
- Core QUANT format system: QUANT2, QUANT4, QUANT8, QUANT16, QUANT32
- QUANT_Q0, QUANT_Q1, Binary, Ternary formats
- GRP (Grouped) variants: QUANT2_G, QUANT4_G, QUANT_Q1_G
- Lloyd-Max vector quantization codebook system
- Sub-block grouping for lossless quantization at low BPW
- Vulkan compute backend (dynamic loading, no SDK required)
- DirectX 12 GPU compute backend (Windows)
- AVX2/SIMD kernel library
- Transformer model architecture with flash attention
- KV cache with QUANT4 quantized variant
- Autograd engine with forward/backward pass
- BPE tokenizer with Unicode support
- Dense trainer with checkpoint support
- MoE trainer with vision, audio, embeddings, OCR, video, text modules
- MoE variants (v1, v2) with expert parallelism
- Distributed training: tensor parallelism, FSDP, DDP
- RingAllReduce and ParameterServer
- Inference engine with sampler and generator
- Quantize and convert CLI tools
- Benchmark suite (kernels, inference, quality, QUANT quantized GEMM)
- Hardware probe / auto-select benchmark
- Production inference engine with streaming
- Quant quantize/codec engines
- CLAIMS_LEDGER.md with 47 tracked claims (46 proven + 1 pending)

### Fixed
- QUANT8 per-block k-means now beats Q8_0 by 1.02x (was 1.4x worse with global k-means)
- QUANT4 per-block Lloyd-Max now beats Q4_0 by 1.11x (was 1.6x worse)
- Sub-block grouping: K>=N per block ensures mathematical losslessness

### Changed
- QUANT formats must be better than all industrial formats (Q4_0, Q8_0, INT8, FP16, FP32)
- Pure C++20 enforced — no Python, no CUDA SDK, no BLAS
- All source files explicitly listed in CMakeLists.txt

---

## [0.1.00] - 2026-07-20

### Added
- Internal alpha release
- Initial QUANT format prototypes (QUANT2, QUANT4, QUANT8)
- Basic transformer model scaffolding
- Tensor and memory management foundations
- Math library with scalar and SIMD paths
- Project structure and build system setup
- `.llama/` and `.bitnet/` reference libraries (not linked)

---

*Last Updated: 2026-07-26*
