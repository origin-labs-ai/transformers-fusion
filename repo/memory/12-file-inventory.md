# 12 — FILE INVENTORY (Phase 0 Full Re-Context, 2026-09-01)

**Run ID:** gauntlet-transcender-20260901
**Mode:** READ-ONLY synthesis of 6 parallel agents (A1-A6)
**Bar freeze:** `bar/bar.sha256` — llama.cpp `8887a48f` + DeepSeek-V2/DeltaNet/GPTQ papers hashed
**Reports on disk:** `repo/sessions/phase0/A1_codec_format.md`, `A2_math_kernel.md`, `A3_model_inference.md`, `A4_training_rl.md`, `A5_backend_platform.md`, `A6_adapters_agi.md`

---

## 1. Coverage Summary

| Agent | Slice | Files examined | Reports |
|-------|-------|----------------|---------|
| A1 | Codec/Format core + session archives | 26 (12 src + 7 headers + 7 tests + quant_config + 9 archive files) | `A1_codec_format.md` (26 rows) |
| A2 | Math/Kernel | 18 src + 7 headers + 3 tests + 10 benches + CMake | `A2_math_kernel.md` (18 src + 7 headers) |
| A3 | Model/Inference | 14 src + 9 headers + 5 test patterns | `A3_model_inference.md` (30+ rows) |
| A4 | Training/RL | 16 src + 6 headers + 4 test patterns | `A4_training_rl.md` (25 rows) |
| A5 | GPU/Backend/Platform | 19 src + engines/ + CMake×3 + Docker + CI×3 + tools/ bench/ | `A5_backend_platform.md` (40 rows) |
| A6 | Adapters/Multimodal/AGI | 33 adapters + 9 core multimodal/agi + engines/trainer | `A6_adapters_agi.md` (40+ rows) |
| **Total** | **Full repo** | **~176 src + 130 headers + 58 tests + 19 tools + 11 benches + 41 engines = 435+ tracked files** | 6 reports |

**Method:** Each agent returned `file → verdict (SOLID/REFERENCE/SHELL/DEAD/ORPHAN/STALE/BRANDED/MISSING-TEST) → file:line → suggested action`. No edits made.

---

## 2. Global Verdict Distribution (synthesized)

| Verdict | Count (approx) | Meaning |
|---------|----------------|---------|
| SOLID | ~180 | Real, wired, tested — keep |
| REFERENCE | ~70 | Correct math but naive / not hot path — keep, optimize later |
| SHELL | ~18 | Stub / no-op / fake harness — must fix or retire honestly |
| DEAD | ~15 | Compiled unreachable / static dead fns — delete |
| ORPHAN | ~34 | Exists but not in any `add_library` — wire or archive |
| STALE | ~25 | Superseded duplicate / drift — delete or sync |
| BRANDED | ~12 files + 40 hits | Proprietary names (k3/kimi/kda/mla/qwen35) — rename |
| MISSING-TEST | ~28 | Real impl zero direct test — add test |

**Overall slice judgments:**
- **A1 Codec:** SOLID core (block_codec 2295L production-grade, codebook 452L solid, vector_quantizer solid, fp engines solid) + STALE tests + 1 dead file `quant_engines_core.cpp`
- **A2 Math/Kernel:** SOLID tiled/gather kernels + 1 critical duplicate `math_avx2_tensor.cpp` + SHELL tests (3/30 APIs) + BRANDED comment
- **A3 Model/Inference:** SOLID transformer/model/kv/cache/generator/http_server + REFERENCE kda/mla (correct but `(void)cache` discard) + SHELL hybrid_moe persistence + BUGGY inference_opt (mask drop, p_draft=1/vocab, ghost V2) + ~373L DEAD tail in transformer.cpp
- **A4 Training/RL:** SOLID autograd/trainer_core/optimizer/qat/fine_tuning/distributed + SHELL PPO/DPO + static-codebook STALE + W1/W2/W18 wounds
- **A5 Backend/Platform:** SOLID factory selection + GPU_VULKAN factory GAP + Vulkan GEMM passthrough + version drift + Dockerfile `|| true`
- **A6 Adapters/Multimodal/AGI:** SOLID agi/multimodal engines + SHELL flywheel fake harness + ORPHAN adapters tree (33 files) + BRANDED qwen35/k3 + stubs `{30000}`

---

## 3. Per-File Inventory (condensed — full evidence in 6 reports)

### 3.1 A1 Codec/Format (verbatim)
- `src/block_codec.cpp` ~2295L **SOLID** (6 wounds) — keep+fix
- `src/codebook.cpp` 452L **SOLID** — keep
- `src/format_planner.cpp` 340L **REFERENCE** OOB — fix
- `src/format_registry.cpp` 595L **REFERENCE** mapping bug — fix
- `src/vector_quantizer.cpp` 395L **SOLID** — keep
- `src/quant_engines_core.cpp` 165L **DEAD** 5 static fns — delete/wire
- `src/quant_engines_quant.cpp` 261L **REFERENCE** — keep
- `src/quant_engines_quant8.cpp` 273L **REFERENCE** — keep
- `src/quant_engines_quant4.cpp` 330L **REFERENCE** — keep
- `src/quant_engines_quant2.cpp` 293L **REFERENCE** — keep
- `src/quant_engines_quant1.cpp` 142L **REFERENCE** — keep
- `src/quant_engines_fp.cpp` 576L **SOLID** — keep
- `include/quant/types.h` 178L **SOLID** — fix comment
- `include/quant/format_registry.h` 227L **REFERENCE** — fix mapping
- `include/quant/codebook.h` 139L **SOLID** — keep
- `include/quant/block_codec.h` 85L **SOLID** — keep
- `quant_config.h.in` 12L **SOLID** — keep
- `tests/test_block_codec.cpp` 101L **STALE** — fix
- `tests/test_fuzz_codec.cpp` 481L **SOLID** (bug `v==19` skip) — fix gap
- `tests/test_format.cpp` 178L **STALE** duplicate MXQ — fix dedup
- `tests/test_format_registry_complete.cpp` 61L **REFERENCE** — keep
- `tests/test_format_audit.cpp` 79L **SOLID** RED BY DESIGN — keep
- `tests/test_quant_mix.cpp` 671L **STALE/SHELL** TWI fixture — rewrite to MXQ
- `tests/test_grp_quality_proof.cpp` 91L **SOLID** — keep
- `tests/test_grpo.cpp` 220L **REFERENCE** mis-named — rename to `test_grpo_rl.cpp`

### 3.2 A2 Math/Kernel
- `src/math.cpp` 662L **SOLID** scalar fallback — keep
- `src/math_avx2.cpp` 681L **SOLID** — keep (canonical)
- `src/math_avx2_tensor.cpp` 613L **STALE/DEAD DUPLICATE** byte-identical — **DELETE** `CMakeLists.txt:58`
- `src/math_avx2_tiled.cpp` 766L **SOLID** — keep
- `src/math_avx2_vec.cpp` 571L **SOLID** — keep
- `src/math_avx512.cpp` 437L **ORPHAN/MISSING-TEST** no GEMM no callers — wire or dead
- `src/simd_math.cpp` 563L **SOLID** (swiglu/geglu fake-SIMD) — refactor
- `src/kernel_gemm.cpp` 165L **SOLID** — keep
- `src/kernel_production.cpp` 489L **ORPHAN+BUG** no callers, gather bug 239:285 — fix/delete
- `src/kernel_q3.cpp` 147L **SOLID/MISSING-TEST** — add test
- `src/kernel_q6.cpp` 147L **SOLID/MISSING-TEST** — add test
- `src/kernel_q12.cpp` 172L **SOLID/MISSING-TEST** — add test
- `src/kernel_q24.cpp` 146L **SOLID/BRANDED** comment `InNova custom FP24` — rebrand
- `src/kernel_quant8.cpp` 82L **SOLID/MISSING-TEST** layout inconsistency — keep
- `src/kernel_quant4.cpp` 131L **SOLID/MISSING-TEST** — keep
- `src/kernel_tl.cpp` 186L **SOLID/MISSING-TEST** LUT recomputed per m*n — refactor
- `src/int8_quant.cpp` 43L **SOLID/MISSING-TEST** — add test
- `src/flash_attention.cpp` 198L **SOLID** — keep
- `include/quant/simd_math.h` **BRANDED** comment `Kimi K3:105` — rebrand

### 3.3 A3 Model/Inference
- `src/transformer.cpp` 894L **SOLID core + 373L DEAD tail** `KimiDeltaAttention:515-666`, `AttentionResidual:668`, `FP8:713`, `MultiHeadLatentAttention:793`, `MTPHead:871` — split/delete tail
- `src/kda_attention.cpp` 132L **REFERENCE** `(void)cache` discard — keep ref
- `src/mla_attention.cpp` 109L **REFERENCE** `(void)cache` — keep ref
- `src/hybrid_scheduler.cpp` 28L **SOLID** off-by-one 70/23 vs paper 69/24 — sync
- `src/hybrid_block.cpp` 61L **SOLID** — keep
- `src/hybrid_expert.cpp` 33L **SOLID ORPHAN BUILD** not in `quant_model` — add to `CMakeLists.txt:101-118`
- `src/hybrid_moe_model.cpp` 82L **SOLID+SHELL** persistence `(void)p` — implement load/save
- `src/model.cpp` 437L **SOLID** wounds: branded guard `322-333`, silent assign `352-357`, header-only `Model::save:383` — fix
- `src/kv_cache.cpp` 844L **SOLID** debug bleed `553-585` `D-clear`, Paged 1T zero callers — gate debug, wire or doc
- `src/kv_cache_quant4.cpp` 419L **SOLID** partial-block codebook reuse `263-267` bug — fix
- `src/inference_opt.cpp` 1197L **SOLID library SHELL/BUGGY** 12 opts: mask drop `337`, `p_draft=1/vocab:160,240`, ghost `SpeculativeDecoderV2:262` — fix
- `src/sampler.cpp` 124L **SOLID** — keep
- `src/generator.cpp` 355L **SOLID** only real serving loop — keep
- `src/http_server.cpp` 978L **SOLID** missing 413/414 text `47-65`, default RateLimiter 0 — harden
- `src/servers.cpp` **MISSING** (spec expects, tree has `http_server.cpp` triad) — rename spec
- `tests/test_transformer*.cpp` **MISSING** — create
- `tests/test_model.cpp` 47L **SHELL** smoke — promote
- `tests/test_kv*.cpp` **MISSING by name** (real `paged_kv_4m_test.cpp` 95L) — rename/create `test_kv_cache_quant4`
- `tests/test_sampler*.cpp` **MISSING** — create

### 3.4 A4 Training/RL
- `src/autograd_engine.cpp` **SOLID** — keep
- `src/autograd_grad.cpp` **SOLID** — keep
- `src/autograd_functions.cpp` **SOLID** — keep
- `src/trainer_core.cpp` **SOLID** wounds: W18 repeat batch `168-191`, `rdrop:604` non-graph, `label_smoothing:637` bypass — fix
- `src/trainer_rl.cpp` **SHELL/MIXED** W2: PPO 160:298 no backward, DPO 342:396 zero grad, GRPO/RLVR solid — rewrite PPO/DPO
- `src/trainer_rl_ops.cpp` **REFERENCE** ppo_finetune scalar adv — fix
- `src/optimizer.cpp` **SOLID** 12 optimizers — keep, add test
- `src/ste_quantizer.cpp` **STALE** static bug `40-48` `cb8` `51-55` `cb4` — **P0 remove static** per `10-code-review-findings.md:3`
- `src/qat.cpp` **SOLID** — keep (clean hindi comment `qat.h:14`)
- `src/fine_tuning.cpp` **SOLID** — keep
- `src/moe_trainer.cpp` **SOLID** dummy aux `284-288` — wire real router logits
- `src/moe_variants.cpp` **REFERENCE** 1730L zero autograd, dispatch `767-827` raw memcpy — mark inference-only or wrap with autograd
- `src/distributed.cpp` **SOLID** — keep
- `src/ddp.cpp` **SOLID** — keep
- `src/fsdp.cpp` **SOLID** — keep (pass real labels)
- `src/zero_optimizer_core.cpp` **SOLID** TLS barrier — keep
- `src/zero_optimizer_mem.cpp` **SHELL** backward stub `496` — rewrite
- `src/continual_engine.cpp` **SOLID** shared codebook alias `88-89` — split
- `tests/test_trainer.cpp` **SOLID** — keep
- `tests/test_autograd*.cpp` **MISSING-TEST** — add
- `tests/test_optimizer*.cpp` **MISSING-TEST** — add

### 3.5 A5 GPU/Backend/Platform
- `src/backend.cpp` **SOLID critical gap** factory `961-991` missing `GPU_VULKAN` case — add
- `include/quant/backend.h` **SOLID** — keep
- `src/gpu_compute_vulkan.cpp` **SOLID** GEMM passthrough `986-996` CPU — implement SPIR-V or gate
- `src/igpu_zero_copy.cpp` **SOLID** — keep
- `src/gpu_compute_cuda.cpp` **SOLID** 1065L — keep
- Shell backends `hexagon/musa/virtgpu/zdnn/openvino/webgpu` **SHELL** CPU fallback — verify
- `src/math_avx512.cpp` **SOLID CONDITIONAL** no GEMM — add gemm or doc
- `engines/` **SOLID** all wired except adapters orphan — keep
- `CMakeLists.txt` **SOLID** 112 targets vs README 82 stale — fix counts
- `Dockerfile` **STALE** `ctest || true:25` masks failures — remove
- `.github/workflows/ci_full.yml` **STALE** exclude `paged_kv_1t_test` non-existent — fix to `test_paged_kv_4m`
- `build_linux.sh` **STALE** `InNova_USE_CUDA` → `QUANT_CUDA`
- `src/adapters/CMakeLists.txt` **ORPHAN** 33 files not in root build — owner decision wire vs archive
- `bench/bench_format_comparison.cpp.bak` **ORPHAN** tracked bak — delete

### 3.6 A6 Adapters/Multimodal/AGI
- `src/agi.cpp` 1113L **SOLID** — keep
- `src/agi_flywheel.cpp` 1980L **REFERENCE** 24 codegens real, fake harness `863-911` `CHECK(true)` + keyword `measure_improvement:1418` — fix harness, add real assertions
- `src/agi_extended.cpp` 885L **REFERENCE** dead `PersonaHotSwap:756` — delete/relocate
- `src/agi_guardrails.cpp` 319L **SOLID** — keep
- `src/multimodal.cpp` 1079L **REFERENCE** stubs `encode_image:674 {30000}` `encode_audio:679 {31000}` — implement or retire
- `src/multimodal_cross_attn.cpp` 950L **SOLID** — keep
- `src/multimodal_fusion.cpp` 358L **SOLID** — keep
- `src/meta_cognition.cpp` 838L **REFERENCE** — add test
- `src/self_eval.cpp` 836L **REFERENCE** — add test
- `src/world_model.cpp` 638L **REFERENCE** — add test
- `src/multi_agent.cpp` 793L **REFERENCE** — add test
- `src/continual_engine.cpp` 543L **REFERENCE** — keep
- `src/code_gen.cpp` 1254L **SOLID** — add test
- `src/hpo_nas.cpp` 1086L **REFERENCE** fake `evaluate_config:368` distance to ideal — wire to real training
- `src/adapters/` 33 files **ORPHAN** standalone sub-project `CMakeLists.txt:26` not in root — integrate or doc
- `include/quant/qwen35_engine.h` + `qwen35_tokenizer.h` + `k3_tokenizer.h` **BRANDED** — rename to neutral `GlaEngine/BpeTokenizer`
- `tests/test_agi.cpp` **SOLID** — keep
- `tests/test_agi_flywheel.cpp` 49L **MISSING-TEST weak** — extend
- `tests/test_multimodal*.cpp` **SOLID** — keep
- Missing tests: code_gen, hpo_nas, multi_agent, world_model, self_eval, meta_cognition, qwen35_* — add

**A6 v2 supplement (second parallel pass, 62 entries, 2026-09-01 late):**
- `src/multimodal_cross_attn.cpp:582-613` **CRITICAL SHELL** `fuse_late` `pooled.zero(); combined.zero(); return combined` — returns zeros regardless of input, `(void)count` — callers via `forward_text_image` not hit but direct test would fail
- `src/multimodal_cross_attn.cpp:902` `generate_caption` `next=rng()%V` random ignoring `fused` — `(void)fused`, demo fake
- `src/multimodal_cross_attn.cpp:922,936` `retrieve_images/text` `score=rng()/UINT32_MAX` random sorting not embedding similarity — retrieval fake
- `src/multimodal.cpp:203,240,271,303,349` string literal stubs `return "transcription"/"ocr_text"/"[video description]"/"caption"/"answer"` when model null — `QUANT_CHECK` needed
- `src/multimodal.cpp:22` `22: empty→Tensor(hidden_)` clone passthrough silent — single modality bypasses attention
- `src/multimodal_fusion.cpp:305` `compute_fusion_weights` `fill 0.5; wd[...0]=1` one-hot constant no learning — replace with `Linear(3*D→3)+softmax`
- `src/agi.cpp:875-945` **CRITICAL SHELL** `RecursiveSelfImprover::apply_perturbation` `dummy_input random ids + model->forward` then `od[i]+=noise*0.01` discarded — never writes back to `model_->get_parameters()`, same for `apply_pruning:934` pruning activation copy — RSI claims show improvement but weights unchanged
- `src/agi_flywheel.cpp:926-931` Windows `run_with_timeout` `(void)timeout_sec` ignores timeout — hangs CI, Linux uses `timeout` correctly
- `src/agi_flywheel.cpp:807-858` `self_play` rotates 32 prompts via `atomic task_index` no curriculum
- `src/adapters/src/continual_trainer.cpp:77` **SHELL** `train_step SCRATCH return zero-loss` `88 synthetic new_loss=sum(input*input)/flat`, never calls `model_->forward` — synthetic loss
- `include/quant/adapters.h:1` **STALE/ORPHAN** header empty `#pragma once` + `#include "quant/types.h"` exposes nothing — consumers get no symbols, should forward-include `adapters/adapter_core.h`
- `src/adapters/include/adapters/adapter_core.h:184` duplicated `if(critical) return Q32` guard (dup of 180)
- `include/quant/agi_utils.h:1` **DUPLICATE** `simple_encode mod+offset 5` `greedy_argmax` `generate_new_tokens context64` identical copy in `agi.cpp:30`, `agi_extended.cpp:30`, `agi_flywheel.cpp:30`, `meta_cognition.cpp` — keep header single source, remove dupes

---

## 4. Frozen Lists (lock before Phase 1)

Legend: **(R) Rename** — content-identical rename + citations · **(D) Delete/Retire** — remove or archive with note · **(W) Wire/Fix** — keep file, fix logic/wiring · Owner-gate = must approve

### 4.1 (R) Rename — brand/provenance cleanup (Phase 2-3)

| # | Old | New | Evidence | Owner-gate |
|---|-----|-----|----------|------------|
| R1 | `src/k3_tokenizer.cpp` | `src/transcender_tokenizer.cpp` | `A3/A6` `k3_tokenizer.h:41` | Docs owner |
| R2 | `tools/k3_convert.cpp` | `tools/transcender_convert.cpp` | `A1` `TODO(G-4):96` | Build owner |
| R3 | `tests/test_k3_tokenizer.cpp` | `tests/test_transcender_tokenizer.cpp` | `A6` | Test owner |
| R4 | `src/kda_attention.cpp` | `src/additive_delta_attention.cpp` | `A3` `KimiDeltaAttention:515` | Model owner |
| R5 | `src/mla_attention.cpp` | `src/latent_kv_attention.cpp` | `A3` MLA DeepSeek `mla_attention.h:2` | Model owner |
| R6 | `src/hybrid_moe_model.cpp` etc. (4 files) | `src/transcender_moe*.cpp` | `A3` | Model owner |
| R7 | `src/mommoe_block.cpp` | `src/transcender_moe_mix.cpp` | `A3` | Model owner |
| R8 | `src/qwen35_tokenizer.cpp` + `qwen35_engine.cpp` (adapters) | `src/safetensors_arch_loader.cpp` + `safetensors_arch_engine.cpp` | `A6` `qwen35_tokenizer.cpp 18KB` in main build `CMakeLists.txt:131` | Adapters owner |
| R9 | `include/quant/k3_tokenizer.h`, `qwen35_*.h`, `kda_attention.h`, `mla_attention.h` + hybrids | mirror renames | `A3/A6` | Model owner |
| R10 | Classes/enums/strings `K3*`, `KDA*`, `MLA*`, `qwen35*`, `InNova` in src/include/docs | `Transcender*`, `AdditiveDeltaAttention*`, `LatentKVAttention*` | `rg -i "k3|kimi|kda|mla|qwen35"` → 100+ hits (A1 slice 0, heavy in A3/A6) | Release owner |
| R11 | `InNova.png` → `Transcender.png`, `project(InNova)` `CMakeLists.txt:2`, README title `5`, CHANGELOG `10`, Dockerfile `4` | — | `A5` version drift | Release owner |

**Citations to add (honest, not hidden):**
```cpp
// Transcender — constant-memory associative attention family.
// Gated delta rule from DeltaNet (Yang et al. 2024); latent KV from DeepSeek-V2.
```

### 4.2 (D) Delete/Retire — dead/bloat (Phase 5)

| # | Item | Evidence | Action |
|---|------|----------|--------|
| D1 | `src/math_avx2_tensor.cpp` 613L duplicate | `A2` byte-identical `math_avx2.cpp`, both in `quant_math` `CMakeLists.txt:58`, `build/CMakeCache` AVX2 ON latent LNK2005 | delete file + remove from `CMakeLists.txt:58` |
| D2 | `src/quant_engines_core.cpp` 165L dead statics | `A1` 5 static fns zero callers | delete or export+wire (owner picks) |
| D3 | `src/transformer.cpp:515-892` 373L dead tail (5 classes) | `A3` `KimiDeltaAttention` etc. never referenced | delete or extract to `research/` |
| D4 | `src/agi_extended.cpp:756-805` `PersonaHotSwap` | `A6` defined mid-file not in header, zero callers | delete |
| D5 | `bench/bench_format_comparison.cpp.bak` tracked bak 22KB | `A5` `bench/*.bak` | `git rm` + gitignore |
| D6 | `dist/` | `A5` not on disk, only `.gitignore` noise | keep ignored, verify no tracked file |
| D7 | `preprocessed.cpp` / `_checkpoint.quant.opt` | `A5` not on disk — stale claim | git clean verify |
| D8 | Dead statics `comp8_scale_search:244`, `f32_bits:188`, `level_value_r:433`, `lead_dim:22` | `A1/A2` | delete |
| D9 | `src/zero_optimizer_mem.cpp:496` stub backward if kept as stub — or retire | `A4` approximate `p*(1-p)/scale` | explicit retire with ledger note if not fixed |

### 4.3 (W) Wire/Fix — logic/bugs/wiring (Phases 6-10)

| # | Item | File:line | Owner-gate | Phase |
|---|------|-----------|------------|-------|
| W1 | **ste_quantizer static-codebook** remove `static CodebookQUANT8 cb8` | `src/ste_quantizer.cpp:40-48` `51-55` | Training owner | 6 |
| W2 | **Decode slowness** vectorize LUT, SIMD bit-unpack, fuse Q24/Q16/Q8_G into GEMM, extend FastBitReader to all formats | `src/block_codec.cpp:1204` + `kernel_q*.cpp` | Kernel owner | 7 |
| W3 | **MoE grad chain** wire router/expert dispatch via AutogradEngine, aux-loss real tensors | `src/moe_variants.cpp:767-827` `src/moe_trainer.cpp:282-288` | Training owner | 8 |
| W4 | **PPO/DPO shells** real policy/critic backprop (PPO clip, DPO through policy grads) | `src/trainer_rl.cpp:160-396` | Training owner | 8 |
| W5 | **Grad accumulation same-batch** fresh micro-batch per acc step | `src/trainer_core.cpp:189-191` `src/moe_trainer.cpp:212` | Training owner | 8 |
| W6 | **MLA cache discard** actually use latent cache, ≥90% KV reduction | `src/mla_attention.cpp:47-51` `include/quant/mla_attention.h:48` | Model owner | 9 |
| W7 | **MTP zero callers** wire MTP head+loss | `src/model.cpp:25-28,395-435` `src/trainer_core.cpp:742` | Model owner | 9 |
| W8 | **Continuous batching fake** pass real mask/kv_caches/seq_lens | `src/inference_opt.cpp:333-334` `283-373` | Inference owner | 9 |
| W9 | **Spec-decode shells** real acceptance `p_draft<1`, rewind wired | `src/inference_opt.cpp:160,240` | Inference owner | 9 |
| W10 | **YARN mscale discard** implement scaled attention | `src/transformer.cpp:138,166,176` | Model owner | 9 |
| W11 | **Silent load skip** fail loudly | `src/model.cpp:352-357` | Model owner | 9 |
| W12 | **QUANT4KVCache partial-block** codebook reuse | `src/kv_cache_quant4.cpp:263-267` | Model owner | 9 |
| W13 | **paged_kv_4m segfault** S9-S12 | `tests/paged_kv_4m_test.cpp:80` `src/kv_cache.cpp:553-585` | Model owner | 9/11 |
| W14 | **GPU_VULKAN factory gap** add `case GPU_VULKAN` | `src/backend.cpp:961` `961-991` | Backend owner | 10 |
| W15 | **Vulkan GEMM passthrough** implement SPIR-V or gate | `src/gpu_compute_vulkan.cpp:986-996` | GPU owner | 10 |
| W16 | **AGI flywheel fake harness** real assertions or retire | `src/agi_flywheel.cpp:863-911` | AGI owner | 10 |
| W17 | **Version drift** one truth `0.1.2/0.1.03/0.1.02` | `CMakeLists.txt:2` `README.md:5` `CHANGELOG.md:10` | Release owner | 10 |
| W18 | **Dockerfile `|| true`** `ctest ... || true` | `Dockerfile:25-26` | CI owner | 10 |
| W19 | **CI sanitizer stale** `paged_kv_1t_test` → `paged_kv_4m` | `.github/workflows/ci_full.yml:125,138,151` | CI owner | 10 |
| W20 | **README:147 stale** 15/8/2 vs 19/0/14 | `README.md:147,5061` `src/format_registry.cpp:20-96` | Docs owner | 10 |
| W21 | **Hybrid expert orphan** add to `quant_model` | `src/hybrid_expert.cpp` `CMakeLists.txt:101-118` | Build owner | 5 |
| W22 | **Transformer dead flags** `use_mla/q_lora` never read | `include/quant/transformer.h:40-44` | Model owner | 9 |
| W23 | **Multimodal constants** `{30000}/{31000}` | `src/multimodal.cpp:674-682` | Multimodal owner | 9 |
| W24 | **Adapters orphan** wire as `quant_adapters` or archive | `src/adapters/` 33 files `A5/A6` | Build owner (owner-gated) | 5 |
| W25 | **Seed-42 sweep** deterministic registry | 10+ sites `agi_extended:104` etc. | AGI owner | 10 |
| W26 | **FormatRegistry mapping** half-GRP→Q1, MXQ plain→GRP | `src/format_registry.cpp:73-187` `177` | Codec owner | 6 |
| W27 | **Affine encode/decode mismatch** bits 1/6 | `src/block_codec.cpp:1204-1217` | Codec owner | 6 |
| W28 | **BPW 31 violations** fit-in-claim vs rename-to-actual owner-gated | `research/audit_full_20260825.md:B` | Owner | 10 |
| W29 | **multimodal_cross_attn fuse_late zero** returns zeros `pooled.zero()->combined zero` + random caption/retrieval `rng()%V` | `src/multimodal_cross_attn.cpp:582-613,902,922` | Multimodal owner | 9 |
| W30 | **agi RSI dummy perturbation/pruning** `od+=noise` discarded, pruning activation copy not weights, Windows timeout ignored | `src/agi.cpp:875-945` `src/agi_flywheel.cpp:926-931` | AGI owner | 10 |
| W31 | **adapters.h empty** exposes nothing + continual_trainer synthetic loss no `model->forward` | `include/quant/adapters.h:1` `src/adapters/src/continual_trainer.cpp:77` | Adapters owner | 5 |
| W32 | **multimodal string literal stubs** `"transcription"/"caption"/"answer"` constant returns | `src/multimodal.cpp:203,240,271,303,349` | Multimodal owner | 9 |

**Tests to fix (Phase 11):**
- `test_fuzz_codec` RED BY DESIGN until BPW decision — then green
- `test_quant_mix` needs MXQ fixture redesign onto `MXQ_3_5_G` family (owner-gated shape)
- `paged_kv_4m` segfault — from W13
- Add missing: `test_simd_math.cpp`, `test_transformer.cpp`, `test_sampler.cpp`, `test_kv_cache_quant4.cpp`, `test_autograd*.cpp`, `test_optimizer*.cpp`, `test_code_gen.cpp` etc. (7 files)

---

## 5. Wound → Phase Traceability (P1-P73 + 23 wounds + L001-L097)

| Wound | File:line | Phase | Status in this doc |
|-------|-----------|-------|--------------------|
| W1 MoE dispatcher raw ptrs | `moe_variants.cpp:767` | 8 | listed W3 |
| W2 PPO/DPO shells | `trainer_rl.cpp:160-396` | 8 | W4 |
| W3 MLA cache discard | `mla_attention.h:47` | 9 | W6 |
| W4 MTP zero callers | `model.cpp:25` | 9 | W7 |
| W6 batching fake | `inference_opt.cpp:333` | 9 | W8 |
| W7 spec-decode | `inference_opt.cpp:160` | 9 | W9 |
| W8 YARN | `transformer.cpp:138` | 9 | W10 |
| W9 load skip | `model.cpp:352` | 9 | W11 |
| ... | ... | ... | see table above |

**Backlog L001-L097 mapping intact per master plan §12-17 — this inventory validates file evidence for each.**

---

## 6. Evidence Hashes (bar freeze)

```
bar/bar.sha256
  llama.cpp HEAD 8887a48f050554f0ee59f56753860c061836b02d
  DeepSeek-V2 6b2464b5d8d5ef270cad7b7dd691afa295546d8e2eecf8ef9082b50f548d3c73
  DeltaNet 6e5fe96d76472745e8e385a338016e00644373cf848220f03a6c839d177bcbc4
  GPTQ 97d8c8983c74ac836dfc8e68c0e99dca7ac54937dcbc41390691f66495402b1f
```

---

## 7. Exit Criteria — Phase 0

- [x] 6 reports saved to `repo/sessions/phase0/` (A1-A6)
- [x] This inventory `12-file-inventory.md` complete (every src/include/tests file verdict)
- [x] Frozen lists locked (R1-11, D1-9, W1-28)
- [x] Re-plan deltas → `13-replan-20260901.md`
- [ ] Owner review before Phase 1 (pending)

**Nothing implemented before this gate — verified.** All agents read-only.

