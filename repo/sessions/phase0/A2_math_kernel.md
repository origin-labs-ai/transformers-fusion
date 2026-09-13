# A2 — Math/Kernel Re-Context Report (Phase-0 Transcender)

**Agent:** A2 (Math/Kernel scope)
**Date:** Phase-0 full re-context
**Working dir:** `C:/Users/thaku/Downloads/InNova`
**Build config verified:** `build/CMakeCache.txt:403` → `QUANT_AVX2:INTERNAL=ON` (AVX2 path is LIVE in current build)

---

## 1. CODE INVENTORY

Verdict legend: SOLID = real, tested, wired; REFERENCE = correct but not the hot path; SHELL = stub/placeholder; DEAD = compiled but unreachable/unused; ORPHAN = defined but no callers; STALE = superseded duplicate; BRANDED = k3/kimi/qwen35 trace; MISSING-TEST = no test coverage.

### Source files

| File | LOC | Verdict | Evidence | Suggested action |
|---|---|---|---|---|
| `src/math.cpp` | 662 | **SOLID** (scalar fallback) | Guarded `#if !defined(QUANT_AVX2)` (math.cpp:8,662); provides scalar `gemm/gemv/dot/...` + 44 `vec_*` fallbacks; mutually exclusive with AVX2 files | Keep as scalar baseline |
| `src/math_avx2.cpp` | 681 | **SOLID** | Real AVX2 6x16 FMA GEMM (math_avx2.cpp:23-78), gemv/dot/norm/softmax/layer_norm/rms_norm, polynomial exp (202-225), FP8 scalar fallbacks (660-668) | Keep; this is the canonical AVX2 tensor-math impl |
| `src/math_avx2_tensor.cpp` | 613 | **STALE / DEAD (duplicate)** | **Byte-identical duplicate of `math_avx2.cpp`** — same 23+ `quant::math` functions, same `#if defined(QUANT_AVX2)` guard (math_avx2_tensor.cpp:8,613), zero unique functions. Both compiled into `quant_math` (CMakeLists.txt:55-60). Latent ODR/duplicate-symbol hazard (see Risks #1) | **DELETE** or exclude from build; it is 100% redundant |
| `src/math_avx2_tiled.cpp` | 766 | **SOLID** | Real tiled GEMM with 64x64 tiles, 6x16 AVX2 micro-kernels, panel packing, transpose variants, fused bias/act (math_avx2_tiled.cpp:308-399); scalar `#else` fallback (591-764) | Keep |
| `src/math_avx2_vec.cpp` | 571 | **SOLID** | Real AVX2 raw-pointer `vec_*` kernels (38 funcs), F16C fp16 (504-555), guarded `#if defined(QUANT_AVX2)` (8,572); mutually exclusive with math.cpp vec_* | Keep |
| `src/math_avx512.cpp` | 437 | **ORPHAN / MISSING-TEST** | Real AVX512 kernels but **no callers anywhere** (only referenced in session notes); compiled only when `QUANT_AVX512` (CMakeLists.txt:62-63); **no GEMM** (header lists only elementwise+gemv) | Wire dispatch or mark dead; add test |
| `src/simd_math.cpp` | 563 | **SOLID** | Real AVX2/AVX512 auto-dispatch kernels (rms_norm, swiglu, geglu, rope, softmax, gemv, tiled_gemm) with runtime CPU detection (simd_math.cpp:23-41) | Keep |
| `src/kernel_gemm.cpp` | 165 | **SOLID** | scalar_gemm, scalar_gemm_bt, avx2_gemm, tiled_gemm, avx2_tiled_gemm (kernel_gemm.cpp:13-162); tested by test_kernel.cpp | Keep |
| `src/kernel_production.cpp` | 489 | **ORPHAN + BUG** | `gemv_quant*_tiled`, `calibrate_*`, `profile_*` — **no callers anywhere** in src/include/tests/bench. `gemv_quant8_tiled_avx2` has a **correctness bug** (see Risks #4) | Fix bug or delete; add test if kept |
| `src/kernel_q3.cpp` | 147 | **SOLID / MISSING-TEST** | Real Q3 3-bit unpack + AVX2 gather GEMM (kernel_q3.cpp:22-133) | Add test |
| `src/kernel_q6.cpp` | 147 | **SOLID / MISSING-TEST** | Real Q6 6-bit unpack + AVX2 gather GEMM (kernel_q6.cpp:20-133) | Add test |
| `src/kernel_q12.cpp` | 172 | **SOLID / MISSING-TEST** | Real Q12 12-bit unpack + AVX2 gather GEMM (kernel_q12.cpp:36-158) | Add test |
| `src/kernel_q24.cpp` | 146 | **SOLID / MISSING-TEST** | Real Q24 FP24 direct GEMM + encode/decode (kernel_q24.cpp:30-143) | Add test |
| `src/kernel_quant8.cpp` | 82 | **SOLID / MISSING-TEST** | Real quant8 codebook GEMM, AVX2 gather (kernel_quant8.cpp:35-68) | Add test |
| `src/kernel_quant4.cpp` | 131 | **SOLID / MISSING-TEST** | Real quant4 nibble GEMM, AVX2 gather (kernel_quant4.cpp:63-117) | Add test |
| `src/kernel_tl.cpp` | 186 | **SOLID / MISSING-TEST** | Real TL1/TL2 ternary LUT GEMM (kernel_tl.cpp:13-184) | Add test |
| `src/int8_quant.cpp` | 43 | **SOLID / MISSING-TEST** | Real per-tensor/per-token int8 quant (int8_quant.cpp:8-41) | Add test |
| `src/flash_attention.cpp` | 198 | **SOLID** | Real FlashAttention-2 tiled forward, AVX2 dot/accumulate, dropout, causal masking, NaN-safe fully-masked handling (flash_attention.cpp:38-191) | Keep; add dedicated test |

### Header files

| File | LOC | Verdict | Evidence |
|---|---|---|---|
| `include/quant/kernel.h` | 81 | SOLID | Declares all kernel GEMMs + scalar/avx2/tiled gemm |
| `include/quant/kernel_production.h` | 86 | ORPHAN | Declares the uncalled production GEMV kernels |
| `include/quant/math.h` | 111 | SOLID | Declares tensor math + vec_* + FP8 |
| `include/quant/math_avx512.h` | 33 | ORPHAN | Declares the uncalled avx512 kernels |
| `include/quant/math_tiled.h` | 93 | SOLID | Declares gemm_tiled family |
| `include/quant/simd_math.h` | 234 | SOLID / **BRANDED** | `simd_math.h:105` comment: "Used in PaLM, Gemma, and **Kimi K3** architectures." (comment-only trace) |
| `include/quant/int8_quant.h` | 30 | SOLID | Declares int8 quant |
| `include/quant/flash_attention.h` | 29 | SOLID | Declares FlashAttention |

### Tests

| File | LOC | Verdict | Evidence |
|---|---|---|---|
| `tests/test_math.cpp` | 48 | SOLID (minimal) | Tests vec_relu, vec_silu, vec_rms_norm only (test_math.cpp:20-44). **No gemm/gemv/softmax/layer_norm/FP8 coverage** |
| `tests/test_kernel.cpp` | 43 | SOLID (minimal) | Tests scalar_gemm, tiled_gemm, avx2_gemm only (test_kernel.cpp:19-39). **No quant4/8/q3/q6/q12/q24/tl coverage** |
| `tests/test_flash*.cpp` | — | **MISSING** | No dedicated flash test file. FlashAttention only exercised indirectly via `tests/test_inference_opt.cpp:31` and `tests/test_protected.cpp:119` |
| `tests/test_simd*.cpp` | — | **MISSING** | No dedicated simd test file. `simd_math.h` only included by `tests/test_ops.cpp:2` |

### BRANDED traces (k3/kimi/qwen35) in scope
- `include/quant/simd_math.h:105` — comment "Kimi K3" (comment-only, low severity).
- No k3/kimi/qwen35 code traces in the math/kernel source files themselves. The heavy branded traces (KDA/MLA/K3, qwen35) live in the attention/transformer/tokenizer layer, **outside** this scope.

---

## 2. KERNEL LOC MEASUREMENT (Phase-23 baseline)

**Total kernel/math source LOC = 6199 lines** (18 files, `wc -l`).

| File | LOC |
|---|---|
| src/math.cpp | 662 |
| src/math_avx2.cpp | 681 |
| src/math_avx2_tiled.cpp | 766 |
| src/math_avx2_tensor.cpp | 613 |
| src/math_avx2_vec.cpp | 571 |
| src/math_avx512.cpp | 437 |
| src/simd_math.cpp | 563 |
| src/kernel_gemm.cpp | 165 |
| src/kernel_production.cpp | 489 |
| src/kernel_q12.cpp | 172 |
| src/kernel_q24.cpp | 146 |
| src/kernel_q3.cpp | 147 |
| src/kernel_q6.cpp | 147 |
| src/kernel_quant4.cpp | 131 |
| src/kernel_quant8.cpp | 82 |
| src/kernel_tl.cpp | 186 |
| src/int8_quant.cpp | 43 |
| src/flash_attention.cpp | 198 |
| **TOTAL** | **6199** |

> **Note:** The expected ~9.8K baseline is NOT met. Actual is **6199**. The gap is explained by `math_avx2_tensor.cpp` (613L) being a full duplicate of `math_avx2.cpp` (681L) — if the duplicate is removed, effective unique LOC drops to ~5586. Headers add 697L; tests add 91L.

---

## 3. FROZEN-LIST CANDIDATES

Legend: (R) = Remove, (D) = Delete, (W) = Wire/fix. Owner-gate flag = who must approve.

| # | Item | Type | File:line | Owner-gate |
|---|---|---|---|---|
| F1 | `math_avx2_tensor.cpp` is a byte-identical duplicate of `math_avx2.cpp` (zero unique functions, same guard, both compiled) | (D) | `src/math_avx2_tensor.cpp:8,613`; `CMakeLists.txt:58` | Math owner |
| F2 | `math_avx512.cpp` + `math_avx512.h` have no callers (orphaned AVX512 kernels, no GEMM) | (R)/(W) | `src/math_avx512.cpp:80-432`; `include/quant/math_avx512.h` | Math owner |
| F3 | `kernel_production.cpp` GEMV/calibrate/profile kernels have no callers (orphaned) | (R)/(W) | `src/kernel_production.cpp:304-485` | Kernel owner |
| F4 | `gemv_quant8_tiled_avx2` gather has a correctness bug (garbage upper lanes pollute accumulator) | (W) | `src/kernel_production.cpp:239-285` | Kernel owner |
| F5 | No dedicated flash/simd tests; quant4/8/q3/q6/q12/q24/tl kernels untested | (W) | `tests/` (missing files) | Test owner |
| F6 | `simd_math.h:105` BRANDED comment "Kimi K3" | (W) | `include/quant/simd_math.h:105` | Doc owner |

---

## 4. TOP RISKS

1. **Duplicate-symbol / ODR hazard (HIGH).** `math_avx2.cpp` and `math_avx2_tensor.cpp` both define the identical set of `quant::math` functions under the same `#if defined(QUANT_AVX2)` guard, and both are compiled into `quant_math` (CMakeLists.txt:55-60). The current build links only because MSVC's static-library linker pulls in `math_avx2.obj` first and never extracts `math_avx2_tensor.obj` (its symbols are already resolved). Any future TU that references a symbol only present in `math_avx2_tensor.obj` while `math_avx2.obj` is also pulled in → **LNK2005 multiple-definition link failure**. This is a landmine, not a live failure.

2. **No AVX512 GEMM (HIGH).** `math_avx512.cpp` implements elementwise/gemv/norm kernels but **no GEMM** (confirmed by grep; header `math_avx512.h` lists no `gemm`). The only AVX512 GEMM is `tiled_gemm_avx512` inside `simd_math.cpp:528-560`, which is a naive per-element broadcast (not a real blocked/register-tiled kernel). For a project claiming 512+ tok/s CPU inference, the AVX512 GEMM path is effectively absent.

3. **`gemv_quant8_tiled_avx2` correctness bug (HIGH, latent).** In `kernel_production.cpp:239-285`, `_mm256_cvtepu8_epi32` produces 4 meaningful lanes per gather group, but `_mm256_i32gather_ps` gathers 8 floats — the upper 4 lanes are garbage. The code then does `_mm256_fmadd_ps(w0, a0, sum8)` where `a0 = x[k..k+7]`, so the garbage lanes get multiplied by real activations and pollute the accumulator. The "FIX" comment (lines 272-285) documents the confusion but does not correct it. Currently latent because the function has no callers, but it will produce wrong results if wired in.

4. **Orphaned production kernels (MEDIUM).** `kernel_production.cpp` (489L) and `math_avx512.cpp` (437L) together represent ~926L of dead/orphaned code with no callers and no tests. This inflates the LOC baseline and creates maintenance/trust risk — a reader may assume these kernels are in the hot path when they are not.

5. **Test coverage gap (MEDIUM).** Only `scalar_gemm`, `tiled_gemm`, `avx2_gemm` (test_kernel.cpp) and `vec_relu`, `vec_silu`, `vec_rms_norm` (test_math.cpp) are tested. The entire quantized kernel family (quant4/8, q3/q6/q12/q24, tl1/tl2), the tiled GEMM family, FP8, and FlashAttention have no direct unit tests. The AVX2 gather kernels (quant8/quant4/q3/q6/q12) are numerically complex and untested — a high-risk gap.

---

## 5. VERIFICATION NOTES

- `build/CMakeCache.txt:403` → `QUANT_AVX2:INTERNAL=ON` (AVX2 path live).
- `build/Debug/quant_math.lib` built with both `math_avx2.obj` and `math_avx2_tensor.obj` present (duplicate symbols latent in the .lib).
- `build/tests/Debug/test_math.exe` and `test_kernel.exe` both run and PASS (verified live).
- `bench/bench_kernels.cpp:99-108` only benchmarks `scalar_gemm`, `avx2_gemm`, `avx2_tiled_gemm` — the production GEMV kernels are not benchmarked.
- `gemm_tiled` (math_avx2_tiled) is called from `src/code_gen.cpp` and `src/gpu_compute_full.cpp` — this is the wired tiled-GEMM path.
