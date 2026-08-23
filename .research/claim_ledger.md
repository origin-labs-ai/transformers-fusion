# CLAIM VERIFICATION LEDGER — master_plan_v2_20260822

> Anti-fake audit register (TRANSCRIPT.md PART-C). Verdicts: VERIFIED / FAKE / MISSING / PARTIAL.
> Rule: "DONE" only with evidence file:line + fresh command output. Zero assumed-DONE.

| # | Claim | Verify How | Verdict | Evidence |
|---|---|---|---|---|
| C-01 | All 37 formats mapped to Q-series + GRP + QUAD/TWI MIX | types.h + format_registry.h enum audit; CSV formats vs registry | PENDING | - |
| C-02 | Adafactor configured across Trainer/MoETrainer/Autograd | grep adafactor in trainer_core.cpp, moe files, autograd_engine.cpp | PENDING | - |
| C-03 | MoE 64% speedup verified (page-lock+async prefetch+sync bypass) | expert_prefetch.cpp impl review; benchmark repro | PENDING | - |
| C-04 | CompressedReplayBuffer overflow fixed | continual_engine.cpp capacity math review | PENDING | - |
| C-05 | thread_local RNG entropy floor fixed | reward.h / trainer_rl.cpp RNG audit | PENDING | - |
| C-06 | Zero-dep dynamic loaders CPU/CUDA/Vulkan/Metal/SYCL/HIP verified | gpu_compute_*.cpp dlopen/load logic + fallback correctness | PENDING | - |
| C-07 | 42 tests pass | ctest full run on this machine | PENDING | - |
| C-08 | 90+ build targets | cmake --build target count | PENDING | - |
| C-09 | RLL PPO implemented | clipped surrogate+GAE+KL verified in code | **VERIFIED** | src/trainer_rl.cpp:29-210 + tests/test_grpo.cpp |
| C-10 | GRPO implemented | per-sample adv weighting fixed in-graph; MoE param collection added; was grad-scale hack before | **VERIFIED** (post-fix) | src/trainer_rl.cpp:533-640 + tests/test_grpo.cpp |
| C-11 | Reward modeling integrated | reward.h forward + KL penalty wiring check | PENDING | - |
| C-12 | EWC implemented | fisher information matrix code search | PENDING | - |
| C-13 | LoRA/DoRA adapters | fine_tuning.h / finetune.h rank-delta audit | PENDING | - |
| C-14 | Flash Attention present | flash_attention.h impl vs declaration reality | PENDING | - |
| C-15 | Speculative decoding works | speculative_decoder.cpp end-to-end trace | PENDING | - |
| C-16 | MLA (DeepSeek V4 Flash) support | search multi-head latent attention / kv compression | PENDING | - |
| C-17 | MTP (multi-token prediction) | mtp head + loss search | PENDING | - |
| C-18 | FP8 E4M3/E5M2 support | fp8 type/conversion kernels search | PENDING | - |
| C-19 | Kimi K3 MoE aux load-balance loss | auxiliary loss term in moe_trainer/moe_model | PENDING | - |
| C-20 | Lossless KV cache offload (RAM/NVMe) | kv_cache offload async pipeline search | PENDING | - |
| C-21 | YARN/NTK long-context scaling | rope scaling interpolation search | PENDING | - |
| C-22 | DDP/FSDP/ZeRO functional | distributed.cpp single-node-only note audit | **PARTIAL** | .research/claims/audit_infra.md + src/distributed.cpp:26-53,106-141,177-179,212-215,276 + src/fsdp.cpp:49-186,229-273,508-543; shared-memory barrier deadlocks for ws>1, NCCL MISSING, zero tests/consumers |
| C-23 | Multimodal (vision/audio/video/OCR) working | multimodal*.cpp real pipeline vs skeleton | **PARTIAL** | .research/claims/audit_infra.md + src/multimodal.cpp:18-118,202-230,400-414,459,674-682 + src/multimodal_fusion.cpp:43-101; T2I UNet proxy, encode_image returns constants, tests only all_finite + 5× TEST_CHECK(true) |
| C-24 | HTTP server production-ready | quant_server.cpp hardening audit (B-4) | **PARTIAL** | .research/claims/audit_infra.md + tools/quant_server.cpp:489-524,209-215; L017 caps present (8KB→414,64KB→413) but status-text 413/414="Unknown", single-recv body no Content-Length |
| C-25 | Charts auto-generated from measured data | scripts/plot_comparison_charts.py input source check | **FAKE** | .research/claims/audit_infra.md + scripts/plot_comparison_charts.py:7-9,12-43 (never reads CSV, hardcodes Q16=60.71 vs CSV 102.449), docs/comparison_charts.html missing |

## FAKE → Rebuild Backlog Map

| Wound/Claim | Verdict | Evidence | Action |
|---|---|---|---|
| B-1 Q3_GRP collapse (-14 dB) | ALREADY-FIXED (stale baseline) | grp_proof_test PASS: gaussian 29.21 > plain-Q3 24.79; real 30.46 > 26.59; fresh CSV rows | Plan PART-O numbers were pre-fix; test_grp_quality_proof guards regression (commit 9bd2ce3) |
| NEW: Q8_GRP lost to plain Q8 (50.23 vs 54.66 gaussian) | FIXED (562fae9) then SUPERSEDED | affine path (6b ladder) fixed plain-GRP: 56.33>54.66 then compound path (fp16 per-32 + golden search): 58.56>58.14 gaussian, 60.15>59.75 real; exact 272B | Old CSV 59.03 stale artifact, now honestly beaten with measured wins on both datasets |
| Q8_GRP vs GGUF Q8_0 industrial win (W2) | **VERIFIED (this session)** | bench_format_comparison.csv: gaussian 58.56 vs 58.14 (+0.42), real 60.15 vs 59.75 (+0.40); src/block_codec.cpp: grp8_compound_fits+quant_grp8_compound+comp8_fit_scale_fp16 (k=0.02, LUT, golden search), sweep k=1.0→0.02 evidence | W2 win register now green on both datasets; plain Q8 54.72/57.07 also recovers vs old 54.66/56.92 |
| test_format hole 19 | FIXED | tests/test_format.cpp: skip unknown hole instead of bpw>0 assert on invalid enum value 19 | ctest test_format 0.24s PASS; FORMAT_COUNT=37 valid IDs |
| B-5 / L015 legacy alias purge | REJECTED (wound is fake/stale) | QUANT_Q0/Q1/6_K are LIVE registered formats: constants.h:42-48, quant_import.cpp enum mapping, sops tables, API_REFERENCE — 25+ files | Blind purge would break public API; aliases are current naming |
| L016 -fno-exceptions jhooth | DOC-FIXED | README:2077 corrected to reflect reality (88 try/catch sites, flag never set) | FULL conversion = dedicated campaign: gpu_compute*.cpp throw_hr plumbing (19+ sites), backend.cpp 21 catches, production_* 18, agi* 11, hpo_nas 6 — scoped in workbench |
| E-5 sanitizer CI missing | STALE (already exists) | ci_full.yml:127-138 ASan+UBSan ubuntu gcc matrix step | Green-run verification pending next CI trigger |
| QUAD_MIX@12.5 old-vs-new delta (54.9 vs 47.4) | GHOST-BASELINE suspect | Same class as Q8_GRP: old number un-reproducible from committed code on MSVC | Investigate separately before trusting either number |
| C-22 DDP/FSDP/ZeRO functional | PARTIAL | barrier per-instance contexts deadlock ws>1, NCCL MISSING, zero tests/consumers | L025→Wave 7/8 (GPU/distributed scope, NCCL docs-only) |
| C-23 Multimodal working | PARTIAL | T2I UNet proxy, constant tokenizers, 5× TEST_CHECK(true) | L025→L079 module tests + real projector |
| C-24 Server production-ready | PARTIAL | 413/414 status-text Unknown, single-recv no Content-Length | L025→L074 full hardening |
| C-25 Charts auto-generated | FAKE | script hardcodes numbers, never reads CSV, HTML missing | L025→L037 charts auto-gen (csv.DictReader) |
