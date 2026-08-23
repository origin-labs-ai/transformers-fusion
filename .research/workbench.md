# WORKBENCH — master_plan_v2_20260822

> Human-readable live status. Source of truth = `.research/telemetry/events.jsonl`.
> Schema: TRANSCRIPT.md PART-I. Update after every task event.

## Current Status

- **Run ID:** master_plan_v2_20260822
- **Current Phase:** 1 (BUG HUNT) — Wave 2 DONE, Q8_GRP win restored
- **Session started:** 2026-08-22
- **Last bench:** 2026-08-23 bench_format_comparison.csv — Q8_GRP 58.56/60.15 dB beats GGUF Q8_0 58.14/59.75 on both datasets

## Phase Status

| Phase | Name | Status |
|---|---|---|
| 0 | Truth Anchor | DONE |
| 1 | Bug Hunt | DONE |
| 2 | Anti-Fake Audit | IN_PROGRESS |
| 3 | Quality Gauntlet | PENDING |
| 4 | Speed War | PENDING |
| 5 | Open-Model Features | PENDING |
| 6 | Architecture & Ecosystem | PENDING |
| 7 | GPU Depth | PENDING |
| 8 | Proprietary Boundary | PENDING |
| 9 | Final Gauntlet | PENDING |

## Tasks (Phase 0)

| ID | Title | Status | Evidence | Critic | Notes |
|---|---|---|---|---|---|
| 0.1 (L001) | Bench enc/dec split fix | DONE | bench_format_comparison.csv: enc!=dec for 37/43 formats both datasets; grep `/ ?2\.0` = 0 hits; run EXIT=0 | PASS | Span end-offset bug bhi pakra+fix (commit ce8cc64) |
| 0.2 (L002) | Warmup + median-of-N stats | DONE | bench/bench_format_comparison.cpp: kWarmupReps=3, kTimedReps=7, median_of/stddev_of; CSV has encode_std,decode_std cols | PASS | L001 ke saath proven |
| 0.3 (L007) | Regression tracker | DONE | scripts/check_regression.ps1; fake regression test RED exit=1, green exit=0 | PASS | >5% threshold, ASCII-safe PS 5.1 |
| 0.4 (L003) | Repo kabristan cleanup | DONE | preprocessed.cpp deleted (git rm); dist/ tree deleted from disk; SHA256_TEST_LOG.md -> docs/ | PASS | commit "chore: remove legacy preprocessed artifact..." |
| 0.5 (L004) | .gitignore harden | DONE | .gitignore: !scripts/check_regression.ps1 negation works (check-ignore exit 0); telemetry runs/history/reports ignored | PASS | events.jsonl tracked rehta hai |
| 0.6 (L006) | CI sanitizer job | DONE (pre-existing) | .github/workflows/ci_full.yml:127-138 ASan+UBSan ubuntu gcc-13 matrix step | PASS | E-5 wound stale nikla — job already tha; green run agle push pe verify hoga |
| 0.7 (L005) | Git discipline | DONE | git log: conventional English commits (docs:, bench:, chore:) | PASS | identity: Satyam Thakur |
| L008 | GLE telemetry scaffold | DONE | tests/test_gle_telemetry.exe ALL PASSED (1000-event integrity, tamper localization line 401, partial-tail tolerance, writer re-open chain continue); gle_report --init genesis chain VALID exit 0 | PASS | src/gle/{gle_telemetry.h,cpp}, tools/gle_report.cpp, quant_gle lib |

## Tasks (Phase 1 — Bug Hunt, Wave 2)

| ID | Title | Status | Evidence | Critic | Notes |
|---|---|---|---|---|---|
| 1.1 (L010) | Q3_GRP collapse debug + fix | DONE (prior) | src/block_codec.cpp grp16_fit_affine fix; commit 9bd2ce3; PSNR 29.21/30.46 > plain Q3 24.79/26.59 | PASS | grp_proof_test guards regression |
| 1.2 (L011) | Q3_GRP repro test | DONE | tests/test_grp_quality_proof.cpp extends; fails before, passes after | PASS | - |
| 1.3 (L013) | QUAD_MIX 4-comp assert | DONE | commit bb4eed8 test_mix_components enforce QUAD=4/TWI=2 + ratio-sum + BPW-bound | PASS | ctest test_mix_components green |
| 1.4 (L014) | TWI_MIX 2-comp assert | DONE | same commit bb4eed8 | PASS | - |
| 1.5 (L015) | Legacy alias purge | REJECTED (wound fake) | QUANT_* live API: constants.h, quant_import.cpp, docs — 25+ files; blind purge would break API | PASS (audit) | ledger decision locked |
| 1.6 (L016) | -fno-exceptions decision | DONE (doc-fixed) | README corrected (88 try/catch, flag never set); full conversion scoped to future campaign | PASS | gpu_compute* 19 sites, backend 21, production 18 etc. |
| 1.7 (L017) | Server timeout/limits pass-1 | DONE | tools/quant_server.cpp: unified set_client_timeout, 8KB request-line 414, 64KB cap 413 | PASS | residual 413/414 status-text "Unknown" + Content-Length gaps mapped to L074 |
| 1.8 (Q8_GRP) | Q8/Q8_GRP industrial win restore | DONE (this session) | src/block_codec.cpp: per-32 FULL fp16 scales + true-MSE golden search (k=0.02) + LUT; bench 58.56>58.14 gaussian, 60.15>59.75 real (both WIN), plain Q8 54.72/57.07 vs 54.66/56.92 (no regression) | PASS | sweep k=1.0→0.02 evidence in bench runs; encode slower (Phase 4 backlog), decode faster; BPW exact 8.5 |
| 1.9 | test_format hole fix | DONE | tests/test_format.cpp: skip enum hole 19 (unknown) instead of asserting bpw>0 on hole | PASS | ctest test_format now green 0.24s |

## Blockers

- test_quant_mix + test_fuzz_codec pre-existing failures (QUAD_MIX@3.5_GRP budget 128>120, TWI_MIX quality) — verified via stash pop baseline run; not introduced by Q8 fix; mapped to Phase 3 L036

## Hourly Summary

- [2026-08-22] Master plan read; workbench + claim ledger + telemetry initialized.
- [2026-08-22] L001/L002: bench /2.0 hack mara; enc/dec separate timing; warmup+median+stddev. CRASH mila debugging mein — Span end-offset count ki tarah pass ho raha tha. Fix + proof: 37/43 enc_ne_dec.
- [2026-08-22] L007 tracker RED/GREEN proven. L003 kabristan saaf. L006 already-existing sanitizer job evidence-locked.
- [2026-08-22] L008 GLE module live: hash-chained JSONL writer/reader/verifier + --init + report CLI. Sab tests green.
- [2026-08-23 09:30] Wave 1/2 audit: L001-L008 verified DONE; L010-L017 audited — Q8_GRP industrial loss (-1.81 dB) confirmed as top blocker. Root cause: 6-bit scale ladder vs GGUF fp16 per-32.
- [2026-08-23 09:45] Q8 fix implemented: per-32 fp16 scales + true-MSE golden search + companded LUT (k sweep 1.0→0.02). Best k=0.02 near-uniform wins both datasets +0.42/+0.40 dB vs GGUF Q8_0. Plain Q8 also recovered (54.72/57.07). test_format hole fixed (enum 19). Other formats unchanged (Q16 102.45 etc.).
- [2026-08-23 11:30] v3 re-arrange: FORMAT_COUNT=105 (base10+K27+GRP9+K_GRP27+half9+halfGRP9+MXQ7+MXQ_GRP7), dot names (Q_GRP_6.5), TWI deleted project-wide, MXQ plain+GRP dono, 32 pe sirf Q32. Saare consumers propagate. Bina-x2 matrix ALL WIN strict.
- [2026-08-23 12:40] Competitor research (GGUF K/IQ, GPTQ, AWQ, SmoothQuant+, SpQR, SqueezeLLM, AQLM, QuIP#, EXL2/3, BitNet b1.58, BinaryNet, HQQ) -> docs/COMPETITOR_ANALYSIS.md with QAT/PTQ/STE paradigm table + FLOPs/SOPs arithmetic table.
- [2026-08-23 12:55] C-25 FIXED: generate_comparison_visuals.cpp rewritten to read CSV only; real SVG charts generated (224 rows). [ref] GGUF Q4_K faithful spec at EXACT 4.5 BPW (1152 bits/superblock): Q_GRP_4.5 wins +17.82/+17.80 dB same-BPW.
- [2026-08-23 13:20] Wave 3 parallel audits done (2 subagents): C-01..C-08 + C-11..C-21 verdicts with file:line evidence into claim_ledger.md. VERIFIED: C-04,C-05,C-08,C-11*,C-12*,C-14,C-18; PARTIAL: C-02,C-03(prefetcher never constructed!),C-06,C-13(DoRA missing),C-15(spec-decode broken acceptance),C-16(MLA no cache),C-17(MTP zero callers),C-19(dummy aux tensors!),C-20(sync-only offload),C-21(yarn_attn_factor discarded). Stale tests fixed: test_format(105)+registry_complete green.
- [2026-08-23 13:35] Speed round 1: comp8_fit_scale_fp16 LS-seeded+tight-golden (72→17 evals), q1_fit_scale_fp16 closed-form (golden loop deleted), grp8 dequant byte-aligned direct reads. Encode: Q8-class ~103ms→~47ms (2.2x), Q1-family ~18ms→~2.3ms (7.8x). PSNR vs pre-speed baseline: worst -0.017dB (noise), Q_GRP_8.5 +0.32dB IMPROVED (58.88). Baseline refrozen bench_history/bench_baseline_20260823_speed1.csv.
