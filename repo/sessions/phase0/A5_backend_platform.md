# A5 — Backend / GPU / Platform Re-Context Report

**Agent:** A5 (GPU/Backend/Platform)
**Date:** Phase-0 full re-context
**Scope:** `src/backend.cpp`, `src/gpu_compute*.cpp`, `src/igpu_zero_copy.cpp`, `engines/`, `CMakeLists.txt` (root + subdirs), `cmake/*.cmake`, `Dockerfile`, `.github/workflows/*.yml`, `tools/`, `bench/`, `sops/`, `docs/`, `.gitignore`, `build_linux.sh`.
**Mode:** READ-ONLY (only file written is this report).

---

## 1. CODE INVENTORY

Verdict legend: **SOLID** = real, wired, tested · **REFERENCE** = real but not primary path · **SHELL** = stub/fallback · **DEAD** = unreachable · **ORPHAN** = not in build · **STALE** = outdated · **BRANDED** = marketing/claims · **MISSING-TEST** = no test coverage.

### 1.1 Backend abstraction

| File | Verdict | Evidence | Action |
|---|---|---|---|
| `src/backend.cpp` | **SOLID** (with a critical gap) | Factory `ComputeBackend::create` at `backend.cpp:961-991`; hardware probe `probe_hardware` at `1301`; auto-select `select_optimal_backend` at `1369`. | See **GPU_VULKAN factory gap** below. |
| `include/quant/backend.h` | **SOLID** | `BackendType` enum at `backend.h:9-34`; `GPU_VULKAN` at `:16`; `backend_name` at `:36`. | — |

**CONFIRMED — GPU_VULKAN factory gap (CRITICAL):**
- `BackendType::GPU_VULKAN` is declared (`backend.h:16`) and is the **3rd-highest priority** in auto-select (`backend.cpp:1388-1392`).
- `is_vulkan_available()` exists (`backend.cpp:1099-1115`) and `probe_hardware` sets `hw.has_vulkan` (`:1312`).
- **BUT the factory switch `backend.cpp:961-991` has NO `case BackendType::GPU_VULKAN:`.** The switch covers CPU_SCALAR/AVX2/AVX512, DIRECTX, IGPU_SHARED, CUDA, METAL, SYCL, CANN, RPC, RAM_SWAP, DISTRIBUTED, OPENVINO, VIRTGPU, WEBGPU, ZENDNN — and falls through to `default: return new CPUScalarBackend()` (`:989`).
- **Consequence:** On any machine where Vulkan is detected and CUDA/Metal are absent, `select_optimal_backend` returns `GPU_VULKAN`, but `create()` silently returns a **CPUScalarBackend**. The Vulkan backend is never instantiated through the factory. The `VulkanBackend` class exists in `gpu_compute_vulkan.cpp` and is reachable via `gpu::get_vulkan_backend()` (`:1118`), but the factory path is broken.
- **Action:** Add `case BackendType::GPU_VULKAN: return new VulkanBackend();` (guarded by `QUANT_USE_VULKAN` or unconditional) to `backend.cpp:961-991`.

### 1.2 GPU compute backends

| File | Verdict | Evidence | Action |
|---|---|---|---|
| `src/gpu_compute.cpp` | **SOLID** (Windows DX) | HLSL shaders + D3D12 (`gpu_compute.cpp:30-65`); DX compute backend. | — |
| `src/gpu_compute_full.cpp` | **SOLID** (Windows DX) | `#include <d3d12.h>` (`:8`), `D3DCompile` (`:361`), `ID3D12Device` (`:373`). | — |
| `src/gpu_compute_cuda.cpp` | **SOLID** | 1065 lines, real CUDA dynamic-loading backend. | — |
| `src/gpu_compute_vulkan.cpp` | **SOLID** (with CPU-passthrough gap) | Real Vulkan init (`:278`), SPIR-V dispatch (`:1000`). | See **Vulkan gemm CPU passthrough** below. |
| `src/gpu_compute_metal.cpp` | **SOLID** | 491 lines, real Metal backend. | — |
| `src/gpu_compute_hip.cpp` | **SOLID** | 596 lines, real HIP backend. | — |
| `src/gpu_compute_sycl.cpp` | **SOLID** | 468 lines, real SYCL backend. | — |
| `src/gpu_compute_cann.cpp` | **SOLID** | 213 lines, real CANN backend. | — |
| `src/gpu_compute_rpc.cpp` | **SOLID** | 215 lines, real RPC backend. | — |
| `src/gpu_compute_opencl.cpp` | **SOLID** | 167 lines, real OpenCL dynamic-loading (`:75-87`). | — |
| `src/gpu_compute_openvino.cpp` | **SHELL** | 105 lines; `if (!handle_) return false` (`:49`); minimal. | Verify real OpenVINO path. |
| `src/gpu_compute_webgpu.cpp` | **SHELL** | 103 lines; `if (!handle_) return false` (`:49`). | Verify real WebGPU path. |
| `src/gpu_compute_hexagon.cpp` | **SHELL** | 80 lines; `launch_gemm` is a **triple-nested CPU loop** (`:66-75`); `alloc` = `new uint8_t[]` (`:57`). | Stub — CPU fallback only. |
| `src/gpu_compute_musa.cpp` | **SHELL** | 97 lines; `init` only checks `musa_malloc_` (`:42`); `alloc` falls back to `new uint8_t[]` (`:52`). | Stub — partial. |
| `src/gpu_compute_virtgpu.cpp` | **SHELL** | 83 lines; `gemm` is a **triple-nested CPU loop** (`:42-52`); `memory_free/total` return hardcoded `1GB` (`:55-60`); `synchronize` no-op (`:63`). | Stub — CPU fallback only. |
| `src/gpu_compute_zdnn.cpp` | **SHELL** | 83 lines; `alloc` = `new uint8_t[]` (`:50`); `init` calls `zdnn_init_` if present (`:52`). | Stub — partial. |
| `src/gpu_compute_zendnn.cpp` | **SHELL** | 152 lines; ZenDNN backend. | Verify. |
| `src/igpu_zero_copy.cpp` | **SOLID** | Real Vulkan unified-memory allocator (`:266` init, `:458` vulkan_ok, `:671` alloc). | — |

**CONFIRMED — Vulkan gemm CPU passthrough (CRITICAL):**
- `VulkanBackend::gemm` at `gpu_compute_vulkan.cpp:986-989` **unconditionally calls `cpu_gemm`** — it never checks `impl_->vulkan_ok` and never dispatches a GPU shader. The comment at `:993-994` for `gemv` admits "No GEMV SPIR-V shader available; use CPU path."
- `gemv` (`:991-996`) also always uses `cpu_gemv`.
- `softmax` (`:1042-1044`), `rms_norm` (`:1046-1048`), `scale` (`:1038-1040`) are also always CPU.
- Only elementwise ops (relu/gelu/silu/add/mul) dispatch to GPU (`:998-1036`).
- **Consequence:** Even if the factory gap is fixed, the Vulkan backend's GEMM/GEMV are pure CPU — no GPU speedup for the dominant matmul workload. The `cpu_gemm` at `:879-902` is a naive tiled scalar loop (no SIMD).
- **Action:** Implement SPIR-V GEMM/GEMV dispatch or gate the backend as CPU-fallback-only until shaders exist.

**CONFIRMED — math_avx512 has no GEMM:**
- `src/math_avx512.cpp` (437 lines) has `gemv_avx512` (`:407`) but **no `gemm_avx512`**. It has dot/axpy/vec ops and activations only.
- `CMakeLists.txt:62-64` compiles `math_avx512.cpp` into `quant_math` only when `QUANT_AVX512` is set.
- **Action:** Add an AVX512 GEMM kernel or document the gap; the AVX512 backend (`CPUAVX512Backend`) has no fast matmul.

### 1.3 Engines

| Dir | Verdict | Evidence | Action |
|---|---|---|---|
| `engines/inference/` (inference.cpp, stream.cpp) | **SOLID** | Wired into `quant_engine` (`CMakeLists.txt:171-172`). | — |
| `engines/quant/` (quantize.cpp, codec.cpp) | **SOLID** | Wired into `quant_engine` (`CMakeLists.txt:173-174`). | — |
| `engines/trainer/dense/` (trainer, dataloader, checkpoint) | **SOLID** | Wired into `quant_dense` (`CMakeLists.txt:182-186`). | — |
| `engines/trainer/moe/` (moe + vision/audio/embeddings/image/ocr/text/video) | **SOLID** | Wired into `quant_moe` (`CMakeLists.txt:190-210`). | — |
| `engines/trainer/multimodal/` (vision/image/video/text/ocr/embeddings/audio) | **SOLID** | Wired into `quant_multimodal` (`CMakeLists.txt:288-295`). | — |

### 1.4 Build system

| File | Verdict | Evidence | Action |
|---|---|---|---|
| `CMakeLists.txt` (root) | **SOLID** | 524 lines, explicit source lists (no GLOB_RECURSE), 26 libs + 20 exes + 10 benches + 56 tests. | — |
| `cmake/arch.cmake` | **SOLID** | Included at `CMakeLists.txt:11`. | — |
| `cmake/compiler.cmake` | **SOLID** | Included at `CMakeLists.txt:12`. | — |
| `tests/CMakeLists.txt` | **SOLID** | 56 tests via `add_quant_test_full`. | — |
| `src/adapters/CMakeLists.txt` | **ORPHAN** | Separate sub-project; **NOT referenced by root CMakeLists.txt** (verified: no `add_subdirectory(src/adapters)`). | See orphan dirs. |

**CONFIRMED — version drift:**
- `CMakeLists.txt:2` → `project(InNova VERSION 0.1.2 ...)`
- `README.md:5` → `# ⚡ InNova — v0.1.03 Release`
- `CHANGELOG.md:10` → `## [0.1.02] - 2026-07-26`
- Three different versions across the three canonical files. `CHANGELOG.md:69` claims "Version bumped to 0.1.02 in CMakeLists.txt" — but CMakeLists is now 0.1.2 and README is 0.1.03.
- **Action:** Reconcile to a single version string.

**CONFIRMED — README:147 stale format claim:**
- `README.md:147` → "**15 single formats** (8 base + 7 grouped), **8 twi-mix**, **2 four-mix**."
- Actual registry (`src/format_registry.cpp`): **19 singles** (10 base + 9 G variants, `build_singles` `:20-58`), **0 twi-mixes** (`build_two_mixes` at `:73-75` returns an **empty vector**), **14 four-mixes** (`build_four_mixes` `:78-96`).
- `README.md:5061` claims "12 single formats, 13 twi-mix variants, and 4 four-mix variants — 29 total" — also stale.
- **Action:** Update README format counts to match the registry (19 singles / 0 twi / 14 four).

### 1.5 CI / Docker / scripts

| File | Verdict | Evidence | Action |
|---|---|---|---|
| `Dockerfile` | **STALE** | `ctest ... || true` at `:25-26` swallows test failures. | Remove `|| true`. |
| `.github/workflows/release.yml` | **SOLID** | 3-OS matrix, packaging, `create_release` job. | — |
| `.github/workflows/build.yml` | **SOLID** | Windows/MSVC + Ubuntu GCC/Clang. | — |
| `.github/workflows/ci_full.yml` | **SOLID** | 4-OS matrix, ASAN/UBSAN, coverage, clang-tidy, docker. | See sanitizer exclusions. |
| `build_linux.sh` | **STALE** | Uses `-DInNova_USE_CUDA=OFF` (`:19`) — **no such option exists** in CMakeLists (it's `QUANT_CUDA`). Also `-Werror` (`:20`) may break builds. | Fix option name. |
| `.gitignore` | **STALE** | See below. | — |

**CONFIRMED — Dockerfile `ctest || true`:**
- `Dockerfile:25-26`: `RUN ctest ... --exclude-regex "..." || true` — the `|| true` masks all test failures in the Docker build. The image is built and pushed even if every test fails.
- **Action:** Remove `|| true` so CI fails on test regressions.

**CONFIRMED — CI sanitizer exclusions:**
- `ci_full.yml:125` (quick tests), `:138` (ASAN), `:151` (coverage) all exclude `test_protected|test_gpu|test_training|test_native_quant|test_moe_training|paged_kv_1t_test`.
- **`paged_kv_1t_test` does not exist** — the actual test is `test_paged_kv_4m` (`tests/CMakeLists.txt:186`). The exclude regex references a non-existent test name (stale).
- `test_gpu` is only built on WIN32 (`tests/CMakeLists.txt:132`), so excluding it on Linux/macOS is harmless but the regex is inconsistent.
- **Action:** Update exclude regex to `test_paged_kv_4m` (or the intended target) and document why these are excluded.

**CONFIRMED — two HTTP servers:**
1. `src/http_server.cpp` — `HTTPServer` class (`:153`), used by `tools/serve.cpp:93` (`quant::ModelHTTPServer`).
2. `tools/quant_server.cpp` — `QuantHTTPServer` class (`:127`), standalone `quant_server` tool (port 9090, `:37`).
- Both are real, distinct servers. `quant_serve` (tools/serve.cpp) and `quant_server` (tools/quant_server.cpp) are separate executables (`CMakeLists.txt:369,375`).
- **Action:** Document the two servers; consider consolidating.

**CONFIRMED — two trainers:**
1. `quant::Trainer` (`include/quant/trainer.h:240`), used by `tools/train.cpp:109`.
2. `NativeQUANTTrainer` (`include/quant/native_trainer.h:65`), used by `tools/train_64m.cpp:6`.
- Both are real, distinct trainer implementations.
- **Action:** Document the two trainers; clarify which is canonical.

### 1.6 Orphan / dead / stale items

**CONFIRMED — orphan dirs:**
- `src/adapters/` — **33 tracked files** (7 src, 1 include, 1 test, 17 tools, CMakeLists). It is a **separate "Adapter Edition" sub-project** with its own `CMakeLists.txt` that imports parent libs from `../build_verify/Release` (`src/adapters/CMakeLists.txt:31`). **NOT referenced by root CMakeLists.txt** (verified). It is a parallel build tree, not part of the main build.
- `dist/` — **NOT present on disk** and **not tracked** by git. Only referenced in `.gitignore` (lines 56, 72, 113, 116, 123) and `release.yml` (creates it at package time). Not an orphan on disk.
- `preprocessed.cpp` — **NOT found anywhere** in the repo (verified via `rg --files`). Not an orphan; the claim is stale.

**CONFIRMED — stale gitignore entries:**
- `.gitignore:108-110` reference `.research/telemetry/...` — but the repo was restructured to `research/` (`.research` does not exist on disk). Stale.
- `.gitignore:56,72,113,116,123` — `dist/` is listed **4 times** (lines 56, 72, 113) plus `dist/linux/` (116) and `dist/windows/x64/` (123). Redundant duplication.
- `.gitignore:89` ignores `build_linux.sh` — but `build_linux.sh` is **tracked** in git (verified via `git ls-files`). The ignore is ineffective for a tracked file.
- `.gitignore:99-102` ignores `*.ps1` globally, with an exception for `scripts/check_regression.ps1` (`:105`). Other tracked `.ps1` files (e.g. `tools/run_tests.ps1`, `tools/deploy_wdac.ps1`) are tracked despite the ignore.

**CONFIRMED — other:**
- `bench/bench_format_comparison.cpp.bak` — a **tracked** `.bak` file (22495 bytes). Should be removed or gitignored.
- `gpu_compute.cpp` and `gpu_compute_full.cpp` contain **HLSL/D3D12 code** but are compiled into `quant_gpu` on **all OS** (`CMakeLists.txt:214` WIN32, `:221` else). On non-Windows, `quant_gpu` has no `d3d12/dxgi/d3dcompiler` link (`:221`), so these files may fail to compile on Linux/macOS unless guarded. `test_cuda_backend` (`tests/CMakeLists.txt:186`) links `quant_gpu` on all OS — a potential non-Windows build break.

---

## 2. BUILD INVENTORY

Counted from `CMakeLists.txt` (root) + `tests/CMakeLists.txt`:

| Category | Count | Source |
|---|---|---|
| Libraries (`add_library`) | **26** | `CMakeLists.txt` (unique; `quant_gpu` counted once despite WIN32/else branches) |
| Executables (`add_executable`) | **20** | `CMakeLists.txt` (tools + standalone benches + sops tools) |
| Benchmarks (`add_quant_bench`) | **10** | `CMakeLists.txt:439-465` |
| Tests (`add_quant_test_full` + `test_gpu`) | **56** | `tests/CMakeLists.txt` (55 via macro + 1 `test_gpu` on WIN32) |
| **TOTAL build targets** | **112** | 26 + 20 + 10 + 56 |

Note: `README.md:5087` claims "82 targets total (25 libraries, 25 executables, 32 tests)" — **stale**. Actual is 112 targets (26 libs, 20 exes, 10 benches, 56 tests). The README's breakdown (25/25/32) does not match the current CMake.

---

## 3. FROZEN-LIST CANDIDATES

Legend: **(R)** = Repair, **(D)** = Delete/Remove, **(W)** = Watch/Verify. Owner-gate flag = the owner who must approve before change.

| # | Candidate | Type | File:line | Owner-gate |
|---|---|---|---|---|
| 1 | **GPU_VULKAN factory gap** — add `case BackendType::GPU_VULKAN` to `ComputeBackend::create` | (R) | `src/backend.cpp:961-991` (missing case; enum at `backend.h:16`) | Backend owner |
| 2 | **Vulkan gemm/gemv CPU passthrough** — implement SPIR-V GEMM/GEMV or gate as CPU-only | (R) | `src/gpu_compute_vulkan.cpp:986-996` | GPU owner |
| 3 | **math_avx512 missing GEMM** — add `gemm_avx512` or document gap | (R) | `src/math_avx512.cpp` (only `gemv_avx512` at `:407`) | Math/Backend owner |
| 4 | **Version drift** — reconcile 0.1.2 / 0.1.03 / 0.1.02 | (R) | `CMakeLists.txt:2`, `README.md:5`, `CHANGELOG.md:10` | Release owner |
| 5 | **README:147 stale format counts** — update to 19/0/14 | (R) | `README.md:147` (and `:5061`) | Docs owner |
| 6 | **Dockerfile `ctest || true`** — remove `|| true` | (R) | `Dockerfile:25-26` | CI owner |
| 7 | **CI stale exclude `paged_kv_1t_test`** — fix to `test_paged_kv_4m` | (R) | `.github/workflows/ci_full.yml:125,138,151` | CI owner |
| 8 | **build_linux.sh wrong option** — `-DInNova_USE_CUDA=OFF` → `-DQUANT_CUDA=OFF` | (R) | `build_linux.sh:19` | Build owner |
| 9 | **Stub backends (hexagon, virtgpu, musa, zdnn, openvino, webgpu)** — CPU-fallback only | (W) | `src/gpu_compute_hexagon.cpp:66-75`, `src/gpu_compute_virtgpu.cpp:42-52`, `src/gpu_compute_musa.cpp:42`, `src/gpu_compute_zdnn.cpp:50` | GPU owner |
| 10 | **`src/adapters/` orphan sub-project** — not in root build | (W) | `src/adapters/CMakeLists.txt` (33 files) | Build owner |
| 11 | **Stale gitignore** — `.research/` paths, `dist/` duplication, `build_linux.sh` ignore | (D) | `.gitignore:56,72,89,108-110,113,116,123` | Repo owner |
| 12 | **`bench/bench_format_comparison.cpp.bak`** — tracked backup file | (D) | `bench/bench_format_comparison.cpp.bak` | Repo owner |
| 13 | **README build-target count stale** — 82 → 112 | (R) | `README.md:5087` | Docs owner |
| 14 | **HLSL/D3D12 files compiled on all OS** — potential non-Windows build break | (W) | `CMakeLists.txt:214,221`; `src/gpu_compute.cpp`, `src/gpu_compute_full.cpp` | Build owner |

---

## 4. TOP RISKS

1. **GPU_VULKAN factory gap silently degrades to CPU (CRITICAL).** `select_optimal_backend` returns `GPU_VULKAN` as 3rd priority (`backend.cpp:1388-1392`), but `create()` has no Vulkan case (`:961-991`) and falls through to `CPUScalarBackend`. On Vulkan-only machines (no CUDA/Metal), the user believes they're on GPU but runs scalar CPU. This is a silent correctness/performance trap.

2. **Vulkan GEMM/GEMV are pure CPU even when Vulkan initializes (CRITICAL).** `VulkanBackend::gemm` (`gpu_compute_vulkan.cpp:986-989`) and `gemv` (`:991-996`) unconditionally call scalar CPU loops. Even after fixing the factory, the Vulkan backend provides no GPU matmul — the dominant workload. The backend is effectively a CPU fallback with GPU elementwise ops.

3. **CI/Docker can pass with broken tests.** `Dockerfile:25-26` uses `ctest ... || true`, masking all test failures. Combined with the stale `paged_kv_1t_test` exclude regex (`ci_full.yml:125,138,151`), the CI signal for test health is unreliable.

4. **Non-Windows build fragility in `quant_gpu`.** `gpu_compute.cpp` and `gpu_compute_full.cpp` contain D3D12/HLSL code but are compiled into `quant_gpu` on all OS (`CMakeLists.txt:214,221`), and `test_cuda_backend` links `quant_gpu` on all OS (`tests/CMakeLists.txt:186`). On Linux/macOS, `quant_gpu` lacks the `d3d12/dxgi/d3dcompiler` link, risking build breaks that the CI matrix may not catch if these files are guarded by `#ifdef`.

5. **Documentation/build-system drift undermines trust.** Version mismatch (0.1.2/0.1.03/0.1.02), stale format counts (README:147 says 15/8/2, actual 19/0/14), stale target count (README:5087 says 82, actual 112), and a wrong CMake option in `build_linux.sh:19` (`InNova_USE_CUDA` vs `QUANT_CUDA`) mean the documented state does not match the code. This is a release-blocking consistency issue.

---

## Verification Notes
- All line numbers are verbatim from the current working tree.
- `dist/` and `preprocessed.cpp` are **not present** on disk — the "orphan" claims for these are stale.
- `src/adapters/` is a real, tracked, separate sub-project (33 files) but is **not** part of the root build.
- Build target count (112) was derived by enumerating `add_library`/`add_executable`/`add_quant_bench`/`add_quant_test_full` across `CMakeLists.txt` and `tests/CMakeLists.txt`.
