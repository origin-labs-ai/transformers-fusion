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
| C-22 | DDP/FSDP/ZeRO functional | distributed.cpp single-node-only note audit | PENDING | - |
| C-23 | Multimodal (vision/audio/video/OCR) working | multimodal*.cpp real pipeline vs skeleton | PENDING | - |
| C-24 | HTTP server production-ready | quant_server.cpp hardening audit (B-4) | PENDING | - |
| C-25 | Charts auto-generated from measured data | scripts/plot_comparison_charts.py input source check | PENDING | - |

## FAKE → Rebuild Backlog Map

| Wound/Claim | Verdict | Evidence | Action |
|---|---|---|---|
| B-1 Q3_GRP collapse (-14 dB) | ALREADY-FIXED (stale baseline) | grp_proof_test PASS: gaussian 29.21 > plain-Q3 24.79; real 30.46 > 26.59; fresh CSV rows | Plan PART-O numbers were pre-fix; test_grp_quality_proof guards regression (commit 9bd2ce3) |
| NEW: Q8_GRP lost to plain Q8 (50.23 vs 54.66 gaussian) | FIXED THIS SESSION | affine path (gsz=32, scb=mb=6): gaussian 56.33 > 54.66; real 57.11 > 56.92; budget exact 272B (commit 562fae9) | Old CSV 59.03 proven un-reproducible from d302e46 code on MSVC = stale artifact from another build env |
| B-5 / L015 legacy alias purge | REJECTED (wound is fake/stale) | QUANT_Q0/Q1/6_K are LIVE registered formats: constants.h:42-48, quant_import.cpp enum mapping, sops tables, API_REFERENCE — 25+ files | Blind purge would break public API; aliases are current naming |
| L016 -fno-exceptions jhooth | DOC-FIXED | README:2077 corrected to reflect reality (88 try/catch sites, flag never set) | FULL conversion = dedicated campaign: gpu_compute*.cpp throw_hr plumbing (19+ sites), backend.cpp 21 catches, production_* 18, agi* 11, hpo_nas 6 — scoped in workbench |
| E-5 sanitizer CI missing | STALE (already exists) | ci_full.yml:127-138 ASan+UBSan ubuntu gcc matrix step | Green-run verification pending next CI trigger |
| QUAD_MIX@12.5 old-vs-new delta (54.9 vs 47.4) | GHOST-BASELINE suspect | Same class as Q8_GRP: old number un-reproducible from committed code on MSVC | Investigate separately before trusting either number |
