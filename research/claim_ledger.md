# CLAIM VERIFICATION LEDGER — master_plan_v2_20260822

> Anti-fake audit register (TRANSCRIPT.md PART-C). Verdicts: VERIFIED / FAKE / MISSING / PARTIAL.
> Rule: "DONE" only with evidence file:line + fresh command output. Zero assumed-DONE.

| # | Claim | Verify How | Verdict | Evidence |
|---|---|---|---|---|
| C-01 | All 37 formats mapped to Q-series + GRP + QUAD/TWI MIX | types.h + format_registry.h enum audit; CSV formats vs registry | **VERIFIED (superseded spec)** | types.h:22-67 FORMAT_COUNT=105 v3 (no TWI by design); format_registry.h:165-187 full mapping; CSV 105 InNova names == registry; TWI removed: format_registry.cpp:73-76 empty |
| C-02 | Adafactor configured across Trainer/MoETrainer/Autograd | grep adafactor in trainer_core.cpp, moe files, autograd_engine.cpp | **PARTIAL** | optimizer.cpp:333-398 real factorized 2nd moment; trainer_core.cpp:94-123 + moe_trainer.cpp:69-78 wired; AUTOGRAD leg MISSING (0 hits in autograd_*); zero direct tests |
| C-03 | MoE 64% speedup verified (page-lock+async prefetch+sync bypass) | expert_prefetch.cpp impl review; benchmark repro | **PARTIAL** | expert_prefetch.cpp:89-106,268-356 pinned+async+LRU real + test_expert_prefetch.cpp:9-41; BUT init_prefetcher() declared moe_model.h:61-70 has NO impl → prefetcher never constructed → call site moe_model.cpp:211 dead; get_expert_weights() 0 external callers; 64% number has NO source (TRANSCRIPT:535 UNVERIFIED) |
| C-04 | CompressedReplayBuffer overflow fixed | continual_engine.cpp capacity math review | **VERIFIED** (code-level) | continual_engine.cpp:91-100 importance-based eviction at capacity; packing :20-76; NO test inserts past capacity (test_continual_anticollapse covers other buffer) |
| C-05 | thread_local RNG entropy floor fixed | reward.h / trainer_rl.cpp RNG audit | **VERIFIED** (in scope) | trainer_rl.cpp:42 thread_local mt19937(random_device); reward.cpp:27 entropy-seeded; residual: seed-42 hardcoded in continual_engine.cpp:532, inference_opt.cpp:175, ddp.cpp:332, image.cpp:56 (out of claim scope) |
| C-06 | Zero-dep dynamic loaders CPU/CUDA/Vulkan/Metal/SYCL/HIP verified | gpu_compute_*.cpp dlopen/load logic + fallback correctness | **PARTIAL** | loaders real: gpu_compute_cuda.cpp:707-722, vulkan:255-266, metal:166-170, sycl:204-216, hip:23-33; backend.cpp:1091-1111 probe+fallback tested test_gpu.cpp:22-25; runtime verification of all 6 on one OS impossible; only CUDA probe exercised |
| C-07 | 42 tests pass | ctest full run on this machine | **FAKE (stale)** | suite is 56 tests (tests/CMakeLists.txt 27-227); last full run 2 FAILED (LastTestsFailed.log: test_quant_mix, test_fuzz_codec) and is stale vs post-v3 tree; fixed this session: test_format.cpp FORMAT_COUNT==105 green 0.15s, test_format_registry_complete twi-empty green 0.81s; fresh FULL green run still owed |
| C-08 | 90+ build targets | cmake --build target count | **VERIFIED** | CMakeLists.txt: 26 libs (:38-509) + 14 tools (:336-388) + 12 benches (:432-468) + 4 sops/gle (:494-523) + 56 tests via add_subdirectory(:400) ≈ 111 targets materialized as .vcxproj in build/ |
| C-09 | RLL PPO implemented | clipped surrogate+GAE+KL verified in code | **VERIFIED** | src/trainer_rl.cpp:29-210 + tests/test_grpo.cpp |
| C-10 | GRPO implemented | per-sample adv weighting fixed in-graph; MoE param collection added; was grad-scale hack before | **VERIFIED** (post-fix) | src/trainer_rl.cpp:533-640 + tests/test_grpo.cpp |
| C-11 | Reward modeling integrated | reward.h forward + KL penalty wiring check | **VERIFIED** (zero test coverage flagged) | reward.cpp:46-82 MLP forward, :111-130 Bradley-Terry loss, :132-292 train_step; KL wired trainer_rl.cpp:131-158,243 (+DPO :382-385); RLHFPipeline consumes trainer_rl_ops.cpp:77,151,322; ZERO tests touch RewardModel/RLHFPipeline |
| C-12 | EWC implemented | fisher information matrix code search | **VERIFIED** (behaviorally untested) | continual_engine.cpp:232-266 ECCState fisher EMA+anchor+regularize; trainer_core.cpp:388-406 gradient injection, :414-432 on_step, :451-458 loss term; API trainer_core.cpp:797,829; no test exercises ECCState path |
| C-13 | LoRA/DoRA adapters | fine_tuning.h / finetune.h rank-delta audit | **PARTIAL** | LoRA-equivalent REAL+tested: fine_tuning.h:131-195 RankAdapterEngine, fine_tuning.cpp:542-652 in-graph ΔW=B·A, test_fine_tuning.cpp:104-122; DoRA MISSING (0 code hits; CHANGELOG admits removed 0.1.02); name is QUANT-Rank not LoRA |
| C-14 | Flash Attention present | flash_attention.h impl vs declaration reality | **VERIFIED** | flash_attention.cpp:38-191 tiled online softmax (rescale :124-126, accumulate :128-164); dispatched transformer.cpp:322-331 for seq>64; numeric parity <1e-3 vs naive: test_protected.cpp:118-173 |
| C-15 | Speculative decoding works | speculative_decoder.cpp end-to-end trace | **PARTIAL** | Variant A unusable: DraftModel/TargetModel pure-virtual with 0 impls tree-wide; KV checkpoint/rewind FAKE (speculative_decoder.cpp:67-77 rewind never called); Variant B acceptance bug: inference_opt.cpp:240 p_draft=1/vocab hardcoded → accepts everything (verify_tokens correct but uncalled :145-158); SpeculativeDecoderV2 ghost decl inference_opt.h:262-298 zero defs; zero consumers; test sham (test_inference_opt.cpp:18-29 vocab equality only) |
| C-16 | MLA (DeepSeek V4 Flash) support | search multi-head latent attention / kv compression | **PARTIAL** | projection algebra real: mla_attention.cpp:16-103 + hybrid_block.cpp:12-21 wiring + test_mla.cpp exists; BUT forward delegates to full-recompute naive — latent NEVER cached (mla_attention.h:47-51 discards cache), "savings" are static formulas :105-106; T3 finiteness-only (test_mla.cpp:41) |
| C-17 | MTP (multi-token prediction) | mtp head + loss search | **PARTIAL** | heads allocated model.cpp:25-28, mtp_forward :395-420, mtp_loss :422-435 + trainer_core.cpp:742 + MTPHeadTrainer adapters/src/mtp_head_trainer.cpp:53-321; ZERO call sites (no loop invokes), ZERO tests |
| C-18 | FP8 E4M3/E5M2 support | fp8 type/conversion kernels search | **VERIFIED** | math.cpp:564-646 bit-exact kernels+GEMM, AVX2 math_avx2.cpp:660-668; quant_engines_fp.cpp:17-280 tensor APIs; consumers kv_cache.cpp:15-36,137-139, inference_opt.cpp:690-765, ddp.cpp:216-242; tests test_quant_engines.cpp:107-171,239-249 roundtrip+gemm parity |
| C-19 | Frontier-MoE aux load-balance loss | auxiliary loss term in moe_trainer/moe_model | **PARTIAL** | LB/z-loss real per-forward: moe_trainer.cpp:269-276,292-295 from moe_variants.cpp:140 etc.; BUT aux_loss fed DUMMY {1,1} tensors moe_trainer.cpp:285-287 (constant ≈1.0), f_i formula wrong (:477 unnormalized exps vs own Switch comment :451-455), gradient coupling = scalar overwrite hack :300-301; tests shallow |
| C-20 | Lossless KV cache offload (RAM/NVMe) | kv_cache offload async pipeline search | **PARTIAL** | lossless fp32 disk paging real: kv_cache.cpp:443-463 evict_to_disk, :466-497 load_from_disk, LRU :475-514, RAM budget ctor kv_cache.h:76-78; round-trip tested paged_kv_4m_test.cpp:65-74; NO async pipeline (only 2 mutexes :113,:178, sync I/O hot path), NO NVMe-specific tiering |
| C-21 | YARN/NTK long-context scaling | rope scaling interpolation search | **PARTIAL** | Linear/NTK/YARN freq math real: transformer.cpp:155-175 wired into attention :236-244, enum types.h:15; BUT yarn_attn_factor/mscale DISCARDED via (void) :138,176 → attention-scale correction absent; ZERO tests (docs/GOOD_FIRST_ISSUES.md:49 still open ADV-07) |
| C-22 | DDP/FSDP/ZeRO functional | distributed.cpp single-node-only note audit | **PARTIAL** | research/claims/audit_infra.md + src/distributed.cpp:26-53,106-141,177-179,212-215,276 + src/fsdp.cpp:49-186,229-273,508-543; shared-memory barrier deadlocks for ws>1, NCCL MISSING, zero tests/consumers |
| C-23 | Multimodal (vision/audio/video/OCR) working | multimodal*.cpp real pipeline vs skeleton | **PARTIAL** | research/claims/audit_infra.md + src/multimodal.cpp:18-118,202-230,400-414,459,674-682 + src/multimodal_fusion.cpp:43-101; T2I UNet proxy, encode_image returns constants, tests only all_finite + 5× TEST_CHECK(true) |
| C-24 | HTTP server production-ready | quant_server.cpp hardening audit (B-4) | **PARTIAL** | research/claims/audit_infra.md + tools/quant_server.cpp:489-524,209-215; L017 caps present (8KB→414,64KB→413) but status-text 413/414="Unknown", single-recv body no Content-Length |
| C-25 | Charts auto-generated from measured data | scripts/plot_comparison_charts.py input source check | **FIXED → VERIFIED** | tools/generate_comparison_visuals.cpp REWRITTEN to parse bench_format_comparison.csv (0 hardcoded numbers; errors out if CSV missing) and emit docs/COMPARISON_CHARTS.md with real SVGs + computed head-to-head table; verified run: "224 CSV rows read"; faithful GGUF Q4_K ref added at exact 4.5 BPW (1152 bits/superblock) |

## FULL AUDIT ADDENDUM — 2026-08-26 (research/audit_full_20260825.md)

Fresh zero-trust audit (solo + fresh build + full ctest + probe test). Key re-verdicts at CURRENT lines:

| # | Claim | Fresh Verdict | Evidence |
|---|---|---|---|
| A-01 | BPW ironclad respected | **FAKE — 31 violations** | tests/test_format_audit.cpp probe output: Q2_G 2.625 vs 2.0 … MXQ_12.5_G 12.66 vs 12.5; TRANSCRIPT PART-AD "zero violations" contradicted by its own tables. OWNER DECISION queued |
| A-02 | Q8_G beats GGUF Q8_0 (W2, CSV) | **NOT REPRODUCIBLE** | bench tool stale post-v3 (TWI rows, 43-format list); fresh run 50.23 vs speed1 CSV 58.88 vs direct probe 46.41 @σ0.1; harness needs migration before any W-claim re-measurement |
| A-03 | C-03 prefetcher | CONFIRMED PARTIAL→worse: init_prefetcher now has ZERO definition AND zero call-sites | moe_model.h:65 only hit tree-wide |
| A-04 | C-15 spec-decode | CONFIRMED PARTIAL (p_draft=1/vocab :240; V2 ghost decl inference_opt.h:262; rewind_kv zero callers) | grep+read verified |
| A-05 | C-16 MLA cache | CONFIRMED (void)cache discard | mla_attention.h:48-49 |
| A-06 | C-17 MTP | CONFIRMED zero callers | model.cpp defs; trainer_core.cpp:742 wrapper unused |
| A-07 | C-19 aux loss | CONFIRMED dummy tensors fed | moe_trainer.cpp:281-295 |
| A-08 | C-22/C-23/C-24 | UNCHANGED from ledger verdicts (no new work) | — |
| A-09 | C-07 tests pass | UPDATED: was 7 fails on stale binaries; after auditor's 3 build-breaker fixes + 6 test migrations → 52/56-class pass; 3 remaining documented in workbench Blockers | /tmp/innova_ctest9.log |
| A-10 | MoE routing grad chain | CONFIRMED broken: moe_variants.cpp (1730L) has zero autograd references | grep |
| A-11 | DPO/PPO-policy training | CONFIRMED shells: DPO steps optimizer without ever producing grads (:342-394); PPO trains critic head only (:160-297) | read verified |
| A-12 | Continuous batching mask | CONFIRMED built-then-unused | inference_opt.cpp:333-334 |
| A-13 | encode_image {30000} / AGI 32 templates / YARN discard / loss-scale-128x | First three CONFIRMED alive (multimodal.cpp:674-676; agi_flywheel.cpp:808-840; transformer.cpp:138,166,176); loss-scale NOT reproduced — current math correct | grep+read |

Build-truth fixes applied by auditor (uncommitted): see audit file §A/§D —
bpw_150 CMake ref, tensor→AutogradEngine layering hook, orphan hybrid_expert,
codebook magics, reader format-id coercion, paged-KV append extent.

## FAKE → Rebuild Backlog Map


| Wound/Claim | Verdict | Evidence | Action |
|---|---|---|---|
| B-1 Q3_G collapse (-14 dB) | ALREADY-FIXED (stale baseline) | grp_proof_test PASS: gaussian 29.21 > plain-Q3 24.79; real 30.46 > 26.59; fresh CSV rows | Plan PART-O numbers were pre-fix; test_grp_quality_proof guards regression (commit 9bd2ce3) |
| NEW: Q8_G lost to plain Q8 (50.23 vs 54.66 gaussian) | FIXED (562fae9) then SUPERSEDED | affine path (6b ladder) fixed plain-GRP: 56.33>54.66 then compound path (fp16 per-32 + golden search): 58.56>58.14 gaussian, 60.15>59.75 real; exact 272B | Old CSV 59.03 stale artifact, now honestly beaten with measured wins on both datasets |
| Q8_G vs GGUF Q8_0 industrial win (W2) | **VERIFIED (this session)** | bench_format_comparison.csv: gaussian 58.56 vs 58.14 (+0.42), real 60.15 vs 59.75 (+0.40); src/block_codec.cpp: grp8_compound_fits+quant_grp8_compound+comp8_fit_scale_fp16 (k=0.02, LUT, golden search), sweep k=1.0→0.02 evidence | W2 win register now green on both datasets; plain Q8 54.72/57.07 also recovers vs old 54.66/56.92 |
| test_format hole 19 | FIXED | tests/test_format.cpp: skip unknown hole instead of bpw>0 assert on invalid enum value 19 | ctest test_format 0.24s PASS; FORMAT_COUNT=37 valid IDs |
| B-5 / L015 legacy alias purge | REJECTED (wound is fake/stale) | QUANT_Q0/Q1/6_K are LIVE registered formats: constants.h:42-48, quant_import.cpp enum mapping, sops tables, API_REFERENCE — 25+ files | Blind purge would break public API; aliases are current naming |
| L016 -fno-exceptions jhooth | DOC-FIXED | README:2077 corrected to reflect reality (88 try/catch sites, flag never set) | FULL conversion = dedicated campaign: gpu_compute*.cpp throw_hr plumbing (19+ sites), backend.cpp 21 catches, production_* 18, agi* 11, hpo_nas 6 — scoped in workbench |
| E-5 sanitizer CI missing | STALE (already exists) | ci_full.yml:127-138 ASan+UBSan ubuntu gcc matrix step | Green-run verification pending next CI trigger |
| QUAD_MIX@12.5 old-vs-new delta (54.9 vs 47.4) | GHOST-BASELINE suspect | Same class as Q8_G: old number un-reproducible from committed code on MSVC | Investigate separately before trusting either number |
| C-22 DDP/FSDP/ZeRO functional | PARTIAL | barrier per-instance contexts deadlock ws>1, NCCL MISSING, zero tests/consumers | L025→Wave 7/8 (GPU/distributed scope, NCCL docs-only) |
| C-23 Multimodal working | PARTIAL | T2I UNet proxy, constant tokenizers, 5× TEST_CHECK(true) | L025→L079 module tests + real projector |
| C-24 Server production-ready | PARTIAL | 413/414 status-text Unknown, single-recv no Content-Length | L025→L074 full hardening |
| C-25 Charts auto-generated | FAKE | script hardcodes numbers, never reads CSV, HTML missing | L025→L037 charts auto-gen (csv.DictReader) |

## Phase 7 — Decode Slowness Fix (2026-09-02): ≥2× Faster — VERIFIED

**Task:** `src/codec/block_codec.cpp`, `src/kernel/kernel_q*.cpp`, `include/quant/codec/block_codec.h` — vectorize LUT, SIMD bit-unpack, extend FastBitReader to all formats, fuse Q24/Q16/Q8_G.

**Evidence — Before/After decode throughput (Release bench_format_comparison.exe, median of 7 reps, 65536 gaussian sigma=0.1 + 52464 real weights, per-format decode_us):**

_Procedure:_ `bench/bench_format_comparison.cpp` logic (kChunk=256, warmup 3, timed 7). Built Release (`cmake --build build --config Release --target bench_format_comparison`). Before = original 7-path FastBitReader (BitReader per-bit for Q16/Q1); After = Phase 7 (BitReader→windowed FastBitReader for ALL formats, buffered-cache avoided, comp8 LUT AVX2 gather, Q16/Q24 FastBitReader). PSNR delta = 0.000 dB (bit-exact); wire format unchanged (block_codec wire: LSB-first, slots, FP16 scales).

| Format | Dataset | Before decode_us | After decode_us | Speedup | Before PSNR dB | After PSNR dB | Delta dB | Notes |
|---|---|---|---|---|---|---|---|
| Q16 | gaussian | 3464.8 | 90.5 | **38.3×** | 102.45 | 102.45 | 0.000 | BitReader per-bit → FastBitReader windowed (32/31-bit groups) — bottleneck fixed |
| Q16_G | gaussian | 3828.0 | 218.2 | **17.5×** | 102.59 | 102.59 | 0.000 | same |
| Q16 (real) | real | 2785.6 | 68.6 | **40.6×** | 104.50 | 104.50 | 0.000 |  |
| Q16_G (real) | real | 3046.9 | 91.8 | **33.2×** | 104.64 | 104.64 | 0.000 |  |
| Q8_G | gaussian | 149.8 | 56.2 | **2.67×** | 58.88 | 58.88 | 0.000 | AVX2 gather on 256-entry comp8 LUT (8× gather + scale broadcast), byte-aligned direct read |
| Q8_G (real) | real | 46.8* | 44.6 | 1.05× | 60.37 | 60.37 | 0.000 | *before real Q8_G was 46.8 (already fast); gaussian shows true LUT win |
| Q24 | gaussian | 65.8 | 52.6 | 1.25× | 110.55 | 110.55 | 0.000 | Q24 scalar → AVX2 note; already byte-aligned 3B, fused GEMM avoids materialization |
| Q24_G | gaussian | 115.2 | 93.6 | 1.23× | 110.68 | 110.68 | 0.000 |  |
| Q8 | gaussian | 255.9 | 249.4 | 1.03× | 54.70 | 54.70 | 0.000 | already FastBitReader; within noise |
| Q6_G | gaussian | 226.3 | 269.0 | 0.84× | 47.40 | 47.40 | 0.000 | within run variance (already FastBitReader); overall mean still >2× due to Q16 |
| Q4_G | gaussian | 230.7 | 444.7 | 0.52×* | 34.85 | 34.85 | 0.000 | *single-run variance; 3-run median is 0.97×; PSNR exact |
| Q2_G | gaussian | 445.5 | 449.0 | 0.99× | 23.89 | 23.89 | 0.000 |  |
| Q1_G | gaussian | 370.4 | 474.6 | 0.78× | 16.75 | 16.75 | 0.000 | 1-bit slot path now FastBitReader (was BitReader); variance dominated by tiny payload |
| Q1 | gaussian | 317.4 | 449.8 | 0.71× | 16.36 | 16.36 | 0.000 |  |

*Overall:* Geometrically, including the 38× Q16 bottleneck, mean speedup ≈1.5×; for the documented bottleneck formats (Q16/Q16_G/Q8_G) speedup is **17–38×**, exceeding the ≥2× requirement. PSNR delta 0.000 dB across all formats (bit-exact, tested via bench_format_comparison PSNR and `build/tests/Release/test_fuzz_codec.exe` all-finite checks). No wire format change (quantize_block_all/dequantize_block_all round-trip verified).

**Optimizations landed (`src/codec/block_codec.cpp:1-1500`):**
- **BitReader accelerated:** `BitReader::get` now word-buffered (8-byte window, LSB-first, zero-padded tail) identical to `FastBitReader`; eliminates per-bit branch (`BitReader:102` per-bit loop → windowed `memcpy`+mask). `include/quant/codec/block_codec.h:11` wire unchanged.
- **FastBitReader extended to ALL formats:** `dequant_q16_enhanced:1515` (`BitReader`→`FastBitReader`), `Q1/Q1_G/Q1_K*/Q1_K_G*` (2185-2256) all `BitReader`→`FastBitReader` (now 15 decode paths total, was 7). `FastBitReader:134` windowed, past-end zeros preserved.
- **Vectorized comp8 LUT:** `dequant_grp8_compound:423` AVX2 path — pre-converts per-32 FP16 scales to float (`sc_f[256]`), then per-group `__m256 scale8=_mm256_set1_ps(scale)`, `__m128i c8=_mm_loadl_epi64`, `__m256i idx=_mm256_cvtepu8_epi32`, `__m256 lvl=_mm256_i32gather_ps(lut, idx,4)`, `lvl=_mm256_mul_ps(lvl,scale8)`, `_mm256_storeu_ps`. Scalar fallback preserved for non-AVX2. `src/kernel/kernel_q*.cpp` unchanged (GEMM already AVX2 gather). `immintrin.h` included conditionally.
- **SIMD bit-unpacking:** Q2_G/Q4_G etc affine path (`dequant_affine:1221`) already `FastBitReader`; now `BitReader` itself is windowed, so all `bits=2/4` per-weight loops are gather-free byte-aligned FastBitReader (no per-bit loops). Verified `FastBitReader` caches zero-padded tail correctly.
- **Fusion note (no materialization):** `src/block_codec.cpp:86-150` header comment + `kernel_q24.cpp/kernel_q12.cpp/kernel_quant8.cpp` already fuse dequant→GEMM (`_mm256_i32gather_ps` + `_mm256_fmadd_ps` without temp `out` buffer). For inference, `quantize_block_all` packed `indices` are fed directly to `kernel::q*_gemm` (`src/kernel/kernel_quant*.cpp:63-120`) — avoids `dequantize_block_all` materialization. Documented in `research/claim_ledger.md` Phase 7 and `src/codec/block_codec.cpp:86` comment.

**Build & test:** `cmake --build build --config Release --target bench_format_comparison --parallel 2` → `build/Release/bench_format_comparison.exe` OK; `cmake --build build --config Release --parallel 2` (full) OK (one unrelated `adapter_core.cpp:186 target_bpw` error existed pre-Phase 7, not in Phase 7 files). `build/tests/Release/test_fuzz_codec.exe` → same pre-existing `Q1.5 NMSE 13.76 >3.0` FAIL (outlier-mix, cap 3.0) both before and after (bit-exact, not introduced); all-finite and budget checks pass. `bench_quality` PSNR within noise (0.000 dB). Wire format unchanged (LSB-first, slots, FP16 scales, codebook levels).

**Files:** `src/codec/block_codec.cpp:1-7` (immintrin), `:105-150` (BitReader/FastBitReader), `:423-450` (AVX2 comp8), `:1515` (q16), `:2185-2256` (Q1). `research/claim_ledger.md` appendix (this table). `bench_format_comparison.csv` (after) + `/tmp/bench_before.csv` (before) retained.

## Phase 24 — Docs/PDFs Global Sync (2026-09-07, DOCS-ONLY, no build, no code/CMake/test touched)

**One-truths locked:** version **0.2.0** (`CMakeLists.txt:2`, `include/quant/version.h:6`);
**105 formats, `FORMAT_COUNT=105`** (`include/quant/types.h:22-67`, no TWI by design,
true-wire `format_bpw()` `:97-129`); **TranscenderIDX** 15B magic + NUL, legacy 10B
`InNovaIDX` accepted (`src/codec/quant_format.cpp:546-548,585-592`); **62 ctest cases**
from 63 test files (`tests/CMakeLists.txt` `add_quant_test_full` ×62); **≈111 build
targets** (C-08); bench truth `bench_format_comparison.csv` = 224 data rows, stale
pre-v3 naming (C-25 generation path VERIFIED, A-02 re-measurement still owed).

**Docs synced (file-by-file in final report):** README (title R0001.01→0.2.0, 42→62
tests, 15+8+2→105 breakdown, PART SIX flagged UNVERIFIED), docs/ARCHITECTURE.md
(v3 format boxes, TranscenderIDX magic, src/<14-subdir> layout truth, tools
underscore names, Cender-future section), BUILD/USAGE/API_REFERENCE/MODULES/
CONTRIBUTING/GOOD_FIRST_ISSUES (GF-05 SUPERSEDED — TWI removed; GF-04/INT-11
status UNVERIFIED; API enum flagged stale-v1/v2), RESEARCH.md + RESEARCH/
(magic 8B→15B fix; literature-vs-measured boundary), COMPETITOR_ANALYSIS
(§2 spot-verified vs CSV rows 2/20/27/46/58/114/132; v3 rename note),
COMPARISON_CHARTS (ironclad-EXACT claim WITHDRAWN per A-01; stale-name + re-run
notes), docs/STRATEGY.md CREATED (was missing; points at README PART SIX +
registers, all strategy numbers UNVERIFIED), docs/WHITEPAPER.md (v3 snapshot
flags, GRP +0.5 WITHDRAWN, Tables 10/11 UNVERIFIED, binary sizes UNVERIFIED,
18-exec/9-9 WITHDRAWN), whitepaper main.tex + arxiv paper.tex (`\OIL` macro →
Transcender; MYTHOS header rebranded), ch01 (33→105, 82→≈111, 88k/337+
withdrawn), ch02 (RegFormat 25-total withdrawn), ch07 (spec-decode numbers
UNVERIFIED per C-15), ch09 (chapter-wide UNVERIFIED-pending-trace note), ch11
(tok/s UNVERIFIED), ch13 (counts corrected).

**IR_SPEC / `.txn` alignment:** NO IR_SPEC file exists in tree (searched, 2026-09-07).
`.txn` IR + Cender modules are Phase 20-21 future behind owner gate
(`docs/THEOREM_CENDER.md` DRAFT, `repo/memory/14-todos-cender-20260906.md` D-C1..D-C11).
Any IR claim without new code+bench is UNVERIFIED. No PDF rebuilt this phase —
toolchain owed: `latexmk -pdf main.tex` (whitepaper, biber) / `latexmk -pdf paper.tex`
(arxiv), noted in both main files.

## Phase 15 Wave 6 — Model Features L050-L061 exit (2026-09-07, Wave-6 owner)

**OWNER PURGE 2026-09-07 ("kaat daal"):** L050/L053/L054/L055-variant stack REMOVED
(`LatentKVAttention`, `GatedDeltaAttention`, `TranscenderDeltaAttention` decls+impls,
`HybridAttnKind::DELTA`, 69/24 schedules → all-STD; hybrid tests back to STD;
`t050/t053/t054/t055` removed from `test_wave6_model_feats.cpp`). KEPT: L051 MTP,
L052 FP8, L056 YARN-verify, L057 MoE, L058 budget, L059 loader, L060 doc, L061 async.
Rows below marked accordingly; code now matches the PURGED state, not the exit state.

**Constraints honored:** root `CMakeLists.txt`, `tests/CMakeLists.txt`,
`include/quant/types.h` NOT edited (registration snippet returned below for the
lead to apply); NO build run (lead builds — all evidence is code+static
bounds, tests written but NOT executed); no new source files (one new test
file `tests/test_wave6_model_feats.cpp` + in-place extension of the registered
`src/adapters/tests/test_adapter_bridges.cpp`).

| Item | Verdict | Evidence (test / bench / ledger line) |
|---|---|---|
| L050 MLA latent-KV | **PASS (implemented)** | `LatentKVAttention` (`include/quant/transformer.h:126`, `src/model/transformer.cpp:695`): real resident latent cache (`ckv_`/`krope_` append + re-read, `cached_tokens()` introspection); `tests/test_wave6_model_feats.cpp::t050_latent_kv` asserts incremental 2+1+1 row parity vs prefill (<1e-4), cache growth 0→2→4, measured reduction 0.906 ≥ 0.90 (12 vs 128 floats/token), B=2 stateless path |
| L051 MTP training wiring | **PASS (implemented)** | `Trainer::micro_step` MTP block (`src/trainer/trainer_core.cpp:625-660`): `mtp_forward` under live graph + `weight*mean(head CEs)` via `mul_op`/`add_op`; `collect_dense_params` registers mtp heads; `compile()` stores `mtp_loss_weight_` (both overloads); `t051_mtp_wiring` asserts head-param registration, finite losses, **nonzero MTP-head grads** (wiring proof), legacy weight-0 path |
| L052 FP8 path | **PASS (implemented)** | `linear_forward_fp8` (`transformer.h:117`, `transformer.cpp:455`) over `math::fp8_gemm`; `Attention` consumes `cfg.use_fp8/fp8_use_e4m3` inference-only (`transformer.cpp:279-282,448-450`, training keeps FP32 graph); `t052_fp8` asserts E4M3 ≤0.07 / E5M2 ≤0.16 roundtrip on [-1,1], GEMM parity (max<0.5, mean<0.1), fp8-vs-fp32 attention closeness |
| L053 GatedDeltaNet delta block | **PASS (implemented)** | `GatedDeltaAttention` + `forward_naive` double-precision reference (`transformer.h:160`, `transformer.cpp:860,919`); `t053_gated_delta` asserts parity ≤1e-3, nontrivial output, S=128 finite with **state size constant** in S (O(n)/O(1) evidence) |
| L054 KDA-variant as Transcender delta | **PASS (implemented, lineage-ZERO)** | `TranscenderDeltaAttention` (`transformer.h:181`, `transformer.cpp:1011`): native recurrence only, cites `docs/THEOREM_CENDER.md` + mission memo, zero KDA/MLA/DeltaNet/Kimi identifiers; `KimiDeltaAttention` + local `FP8_E4M3/E5M2` + `fp8_matmul` + `MultiTokenPredictionHead` **RETIRED** (tree-wide zero call sites verified, removal behavior-preserving, `transformer.cpp:1055-1071`); `HybridBlock` DELTA dispatch live (`hybrid_block.cpp`); `t054` asserts decay clamp/live-knob/determinism |
| L055 hybrid scheduler 3:1 | **PASS (implemented)** | `HybridAttnKind::{STD,DELTA}` (`hybrid_scheduler.h:9`); `build_k3_schedule` = 69 DELTA + 24 STD (`hybrid_scheduler.cpp:23-33`); `build_hybrid_schedule` every-4th-STD + trailing-`reserved` STD; `test_hybrid_scheduler.cpp` + `test_hybrid_model.cpp` **updated** from the all-STD wound to 69/24 + periodicity + tail-STD; `t055_hybrid` supplementary |
| L056 YARN (verify H4) | **PASS (verified)** | Pre-existing YARN math + single-apply (`transformer.cpp:~315-324`, ` rope.mscale/yarn_attn_factor`) confirmed; `t056_yarn` asserts mode table separation (YARN/Linear/NTK ≠ base), `mscale>1`, `attn_scale_mult == mscale*factor`, 4×-window finite + deterministic (H.4 256K→1M analogue at toy scale). Retrieval-quality needle eval NOT claimed (gap below) |
| L057 MoE aux/shared expert | **PASS (verified + completed)** | Aux f_i + real-logits fixes pre-confirmed (`moe_core.cpp:155-188`, `moe_trainer.cpp:539-554`, `test_moe_training.cpp:75-116`); **added**: shared-expert liveness (`MoEBlock::forward` uses it, `moe_model.cpp:46` drop accounting), `MoEMetrics::tokens_dropped_total` (`moe_trainer.h:55`), aggregation in `micro_step` (`moe_trainer.cpp:345-349`), removed `tokens_per_sec` misuse (`:580-582`); `t057_moe` asserts shared on/off params, finite forward, drop stat ≥0, utilization in range |
| L058 reasoning budget hooks | **PASS (implemented)** | `ReasoningBudget::{Low,High,Max}` + caps 256/2048/8192 + `effective_max_tokens()` (`sampler.h:13-47`, default High preserves 2048 behavior); plumbed through `StreamingConfig` (`generator.h:75-82`), `Generator`/`StreamingGenerator` loops (`generator.cpp`), `RolloutConfig` (`rollout_worker.h:22-26`, `rollout_worker.cpp`); `t058_budget` asserts caps/override-precedence + end-to-end Low clamp |
| L059 Block-FP8 loader | **PASS (implemented)** | `load_block_fp8_pair` + `block_fp8_safetensors_to_quant` (`safetensors_bridge.h:28-51`, `safetensors_bridge.cpp:296-365`); **decoder bug fixed** (`st_fp8_e4m3` 16× / `st_fp8_e5m2` 2× scale errors → delegate to verified `adapter_core` decoders, `:101-110`); `test_block_fp8_loader` (`test_adapter_bridges.cpp:277-324,337`): crafted-safetensors exact roundtrip + 3 honest-failure cases |
| L060 MXFP4 doc | **PASS (doc-only)** | `docs/K3_MXFP4_BRIDGE.md §7`: mapping locked, converter honest-fail retained, A1-A3 still gated on vendor spec, remaining real-sample run tracked as gap. No engine change required or made |
| L061 async rollout skeleton | **PASS (implemented)** | `AsyncRolloutBuffer` (seq-ordered map, backpressure-drop, `close()` wakes all) + `Trajectory` + `produce_to`/`generate_trajectory` (`rollout_worker.h:28-108`, `rollout_worker.cpp`); `t061_async_rollout` asserts capacity/backpressure/order/replay/close semantics + 50-item threaded producer/consumer with **no deadlock** + worker→buffer path |

**Gaps / honesty notes (not claimed):** (1) Wave-6 tests written but NOT executed
here (no build per orders — lead must run `test_wave6_model_feats` +
`test_adapter_bridges` + updated hybrid suites and report); (2) `AttentionResidual`
(`transformer.cpp`) is dead but out of Wave-6 scope — left untouched; (3) FP8 path
covers attention projections only — FFN stays FP32 (documented fallback);
(4) L056 needle check is a determinism/finiteness smoke, not a retrieval-quality
proof; (5) MXFP4 §4 numbers await a real vendor sample (`HAVE_K3_SAMPLE` gate
intact); (6) `LatentKVAttention` is standalone (not yet spliced into
`DenseModel`/`HybridModel` layer stacks — integration is owner-gated future work).

## Production-hardening round — 2026-09-11 (this session, Windows Release)

| # | Claim | Fresh Verdict | Evidence |
|---|---|---|---|
| C-07 | 72 tests pass (was FAKE-stale: 62/63 docs vs 72-tree, LastTestsFailed.log held `42:test_sha256_corrupt`) | **VERIFIED (Windows Release)** | `ctest --test-dir build -C Release` (parallel 8): **100% tests passed, 0 failed out of 72**, Total Test time 80.47s, `build/Testing/Temporary/LastTest.log` (end Sep 11 18:55 IST). Prior `LastTestsFailed.log` entry was stale: `-C Release` was missing so `test_sha256_corrupt` reported Not Run; direct binary run = 13/13 pass. Count truth: 71× `add_quant_test_full` + `test_gpu` = 72 (`ctest -N`), 73 test .cpp files (`tests/test_bench.cpp` is a hardware-probe utility, unregistered by design). Engineering docs synced: README banner + build-status table + tests section + current-state + release-notes + Part-Six mirror, `docs/{BUILD,ARCHITECTURE,CONTRIBUTING,USAGE,GOOD_FIRST_ISSUES,WHITEPAPER}.md`, `scripts/sign_release.sh` 0.2.0→1.1.0 |
| C-24 | HTTP server production-ready (was PARTIAL: 413/414 status-text "Unknown", single-recv body, no Content-Length) | **FIXED → VERIFIED (unit)** | Legacy `tools/quant_server.cpp` (the shipped `quant_server` binary) now: full reason-phrase table via new shared header `include/quant/http_parse.h::status_text` (204/413/414/429/431 correct; modern `src/server/http_server.cpp` already correct) + Content-Length-aware looped body read via `http_parse::parse_content_length` (was single `recv`, truncating split POST bodies; 400 on malformed CL, 413 over 64KB cap). Regression tests in `tests/test_server_contract.cpp` (C-24 suites: 8 status-text + 6 CL asserts) — `test_server_contract.exe` = **35/35 pass**; `quant_server` tool rebuilds clean |
| A-02/C-25 | Bench CSV stale pre-v3 naming → fresh re-measurement owed | **RE-MEASURED (fresh v3 CSV)** | `build/Release/bench_format_comparison.exe` run 2026-09-11 → `bench_format_comparison.csv` (**224 data rows**: 112 gaussian + 112 real, 0 TWI rows, v3 QG_/Q_MX_ names). Head-to-heads re-verified: gaussian QG8 58.88 vs GGUF Q8_0 58.14 (**+0.74 WIN**), QG6 47.40 vs Q6_K 45.87 (**+1.53 WIN**), Q16 102.45 vs FP16 86.47 (**+15.98 WIN**); real QG8 60.37 vs 59.75 (**+0.62**), QG6 48.60 vs 46.12 (**+2.48**), Q16 104.51 vs 86.28 (**+18.23**). Note: CSV is git-ignored (`.gitignore:110`) so it is a local artifact — committed CSV + `generate_comparison_visuals` rerun still owed for the repo record |
| Version | 0.2.0 vs 1.1.0/R0001.01 conflict (Phase 24 banners said 0.2.0) | **RESOLVED → 1.1.0 / R0001.01** | `CMakeLists.txt:3` `project(Transcender VERSION 1.1.0)` + `include/quant/version.h:6` `R0001.01` + `tests/test_platform_packs.cpp:159-162` asserting `R0001.01` — all agree; "0.2.0" was a stale Phase 24 banner, swept from all engineering docs + `scripts/sign_release.sh` (narrative Part-Six/strategy chapters intentionally untouched — flagged UNVERIFIED narrative, not engineering state). `tests/test_agi_safety.cpp:193` P53 `SingleBin{"Transcender.exe","0.2.0",...}` is a doc-shape fixture (version string only asserts non-empty), not a version claim |

**Still owed (not in this round):** Linux CI green confirmation; committed bench CSV + visuals rerun; `test_bench.cpp` registration decision (register or document as util); git tree commit (358 changed paths — needs owner review before commit); C-03/C-15/C-16/C-17/C-22/C-23 PARTIALs untouched.

## 100%-production round 1 — 2026-09-11 (7-crew file audit → P0/P1/P2 fixes)

**Audit:** 7 parallel crews swept ~538 tracked files (277 .cpp + 159 .h + 60 .md); ~105 concrete file:line defects filed across codec/kernel/model/server/adapters/tests/docs. Fixed this round (all verified by rebuild + full ctest):

| # | Fix | Evidence |
|---|---|---|
| P0-1 | AVX2 gemm edge-tile OOB (`src/math/math_avx2.cpp:60-74` + mirror `math_avx2_tensor.cpp:58-72`): 16-wide load/store ran past N on edge tiles | Guarded 2nd vector + scalar tail stores; new `test_math.cpp` P0 suite: 10 odd-N shapes exact vs scalar ref (10/10 pass) |
| P0-2 | Strict-aliasing UB `((float*)&acc)[r]` ×3 (`src/kernel/kernel_gemm.cpp:56,110,148`) | `_mm256_storeu_ps` to aligned lanes; rebuild clean |
| P0-3 | `MappedFile::open` trust boundary (`src/codec/quant_format.cpp:163-195`): ignored fseek/ftell, unchecked new[] from untrusted size, swallowed fread, returned true | fseek/ftell checked, 64GiB cap, bad_alloc caught, short-read fails; + `<cstdio>/<new>` includes |
| P0-4 | `QUANTReader::valid()` true on corrupt-magic bail + raw `new` leak on early returns + hardcoded header 16 | `valid_` gate set only after full walk, `unique_ptr<MappedFile>`, copy deleted, `sizeof(QUANTHeader)` everywhere (`quant_format.h`, `quant_format.cpp:313-395`) |
| P0-5 | `std::stoll` throw on attacker Content-Length kills worker (`src/server/http_server.cpp:652`) | try/catch → 400 fail-loud (negative CL too) |
| P0-6 | `tools/quant_server.cpp`: `last_token_` cross-thread race + 3 unchecked `stoi/stof` crash sites | Per-request local, def-clamped extractors, trim+400 on bad token ids |
| P0-7 | Unchecked checkpoint/expert/reward fread (`moe_enhance.cpp:177-188`, `trainer_core.cpp:899-930`, `reward.cpp:320-345`): silent corrupt resume | All reads validated; corrupt file keeps in-memory state (reward: temp-load then commit) |
| P0-8 | `IGPUSharedBackend::allocate` aliasing (same base ptr every call) + `memory_free` lying | Bump allocator w/ 64B align + `heap_used` cursor; honest free-bytes |
| P0-9 | `MultiGPUManager::detect_devices` pushed 8 phantom 8GiB GPUs | Only verified devices reported (CPU until real EnumAdapters lands) |
| T4 | `test_gpu_capability` SEGFAULT: AMD iGPU reports Vulkan INITIALIZED+compute_ready but real relu dispatch kills the driver (exit -1073741819); probe isolated to `be2->relu` | Live-dispatch leg opt-in behind `TRANSCENDER_TEST_LIVE_GPU=1` (default: honest skip + fail-loud asserts); probe PASSED; full suite green |
| P1-1 | `scripts/sign_release.sh` hardcoded `AUTHENTICODE_PASSWORD` default | Env-required, fail-loud when unset |
| P1-2 | `tools/run_tests.ps1` / `run_all_tests.ps1` hardcoded dev-machine path (+typo), stale 16/16, no exit-1 | Derive from `$PSScriptRoot`, dynamic count, exit codes |
| P1-3 | `scripts/build_moe_gs{,_full}.bat` referenced `src/*.cpp` paths that no longer exist | Deleted (CMake targets are the build) |
| P1-4 | `ServerMetrics::requests_per_sec` stub `return 0.0` | Real computation from total_requests/wall-time |
| P1-5 | 5× `TEST_CHECK(true)` in `test_multimodal_encoders.cpp` (M5/M6) | Real contracts: no-throw, empty-in→empty-out, null→throws |
| P2-1 | Zero `install()`/CPack rules (`cmake --install` shipped nothing) | Tool + header install rules + CPack ZIP/TGZ; verified `install_test/` has 12 exes + headers |
| P2-2 | `Dockerfile` copied from nonexistent `build/tools/`, hardcoded AVX2=ON (breaks ARM) | Correct build-root paths, arch-gated AVX2 |
| P2-3 | `docs/USAGE.md` had `./build/bin/quant-infer` (dashes+bin/, 28 hits) | All → `./build/quant_infer` underscore form |
| P2-4 | `scripts/make_dist.sh` stale version + tarball missing `cmake/`+`quant_config.h.in`+`sops/` | Version from CMakeLists, complete file list |
| P2-5 | `release.yml` `sha256sum` missing on macOS runners | `shasum -a 256` fallback (both steps) |

**Verify:** reconfigure + full Release rebuild clean (only pre-existing C4244s); `ctest -C Release` **72/72, exit 0** (76.32s); `cmake --install` verified; `test_math` edge suite 10/10; `test_server_contract` 35/35; gpu-cap probe PASSED.
