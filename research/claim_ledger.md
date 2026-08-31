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
| C-19 | Kimi K3 MoE aux load-balance loss | auxiliary loss term in moe_trainer/moe_model | **PARTIAL** | LB/z-loss real per-forward: moe_trainer.cpp:269-276,292-295 from moe_variants.cpp:140 etc.; BUT aux_loss fed DUMMY {1,1} tensors moe_trainer.cpp:285-287 (constant ≈1.0), f_i formula wrong (:477 unnormalized exps vs own Switch comment :451-455), gradient coupling = scalar overwrite hack :300-301; tests shallow |
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
