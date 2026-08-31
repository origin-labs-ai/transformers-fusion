# Claims Audit — Model Architecture Features (C-16..C-21)

Auditor: blind critic-auditor, static analysis only (no builds run).
Date: 2026-08-22. Tree state: working dir `InNova`, ledger rows all PENDING at audit time.
Verdict scale: VERIFIED / FAKE / MISSING / PARTIAL (per `.research/claim_ledger.md` definitions).
Categories inside notes: (a) real impl, (b) declared-but-empty skeleton, (c) comment/doc mention, (d) absent.

Test-registration source of truth: `tests/CMakeLists.txt` (206 lines, read in full). Adapter-side tests live under `src/adapters/tests/` and are built from `src/adapters/CMakeLists.txt`.

---

## Summary Table

| ID | Verdict | Evidence (file:line) | Notes |
|----|---------|----------------------|-------|
| C-16 | FAKE | include/quant/transformer.h:41,44 (dead flags); grep `use_mla\|mla_` tree-wide = 5 hits, all decls/docs; src/moe_variants.cpp:1355-1383 (name-alike) | MLA attention/KV compression ABSENT. Only (b) unused config flags `use_mla=false` + `mla_rope_dim=0` with ZERO consumers anywhere in src/. Trap: `MLAMoE` class exists (moe_variants.cpp:1356) but is a low-rank activation bottleneck (hidden→latent_dim=256→hidden via down_proj/up_proj) feeding an MoE router — no attention, no Q/KV projection, no latent KV cache. Name collision, not DeepSeek-style MLA. No test references MLA attention. Rebuild mapping: **L050**. |
| C-17 | PARTIAL | src/model.cpp:25-27 (head alloc), :93 (params), :395-419 (mtp_forward), :422-434 (mtp_loss); src/trainer_core.cpp:742-764; include/quant/model.h:44-55; include/quant/trainer.h:67-68,269; include/quant/transformer.h:26-28; src/adapters/src/mtp_head_trainer.cpp:10-321 (built: src/adapters/CMakeLists.txt:46) | (a) Real code: DenseModel owns `mtp_heads` (Linear hidden→vocab per future position), `mtp_forward` returns per-head logits, `mtp_loss` averages shifted cross-entropy over heads; standalone `MTPHeadTrainer` adapter has own fwd/bwd/update + speculative propose_tokens. BUT: grep for call sites of `mtp_forward\|mtp_loss(` across src/ returns ONLY the 3 definition sites — no training loop ever invokes them (`mtp_loss_weight` config never read in a loop either). Tests: ZERO matches for `(?i)mtp` under tests/; MTPHeadTrainer also untested (0 hits in src/adapters/tests/). Core exists, edges (integration + tests) broken. Rebuild mapping: **L051** (wire into Trainer::train when `mtp_loss_weight>0` + add test_training_features case). |
| C-18 | VERIFIED | Conversion kernels: src/math.cpp:564-641 (scalar e4m3/e5m2 bits↔f32 + vec + fp8_gemm); AVX2: src/math_avx2.cpp:660-668; tensor APIs: src/quant_engines_fp.cpp:17-280 (quantize/dequant/per-channel/error/SNR/quant_gemm both formats); headers: include/quant/math.h:87-100, include/quant/quant_engines.h:11-31; KV-FP8: src/kv_cache.cpp:15-36,:137-139 + include/quant/kv_cache.h:38-44; inference: src/inference_opt.cpp:690-765 (FP8Inference, two-stage residual accum); grad-compress: src/ddp.cpp:141,:216-242; adapters decode: src/adapters/src/adapter_core.cpp:49-75 | (a) Fully real, multiple layers of stack. Repro: `ctest -R test_quant_engines` — test_quant_engines.cpp:107-171 exercises e4m3+e5m2 scalar/tensor/per-channel round-trip + error/SNR bounds, :239-246 quant GEMM vs fp32 reference; registered tests/CMakeLists.txt:75-76 (label "training"). Also test_inference_opt registered :102-103 covers FP8Inference path; adapter round-trip test_adapter_bridges.cpp:49-53 asserts 1.0f decodes. Caveat (non-blocking): no native `DType::FP8` storage enum — types.h:223-231 has U8/F16/F32 only; FP8 lives in uint8_t buffers. That is a scope nuance, not a mismatch. |
| C-19 | PARTIAL | LB loss real: src/moe_variants.cpp:140 (compute_load_balance_loss), set per-forward e.g. :219,:670; aggregated src/moe_trainer.cpp:269-275; weighted total :292-295; stamped pre-backward :300-301; expert-parallel variant src/expert_parallel.cpp:368,:372; 24 MoE variants implement `load_balance_loss` (src/moe_advanced_support.cpp:20-1260). AUX loss broken: src/moe_trainer.cpp:281-290 + compute_aux_loss :449-494 | Split verdict. Load-balance term = (a) real: computed from actual router logits every forward, folded into reported total_loss. Aux term = (b)/(broken wiring): micro_step calls `compute_aux_loss(dummy_logits{1,1}, dummy_indices{1,1})` (moe_trainer.cpp:285-287) — with n_tokens=1,n_experts=1 this always returns ≈1.0 constant regardless of data; additionally f_i at :489 accumulates unnormalized softmax exps, not token fractions (Switch/GShard formula misimplemented). Gradient coupling is only a scalar overwrite hack (`*ld = total_loss_val` at :300-301 before backward) — LB/aux terms are not in the autograd graph, so their gradient influence is unproven. Tests exist but shallow: test_moe_training.cpp:43 (aux_loss_coef=0.01), :69 (lb>=0 non-negative only) registered tests/CMakeLists.txt:87-88; NaN guard test_protected.cpp:298-315 registered :146-147; test_expert_parallel.cpp:44 registered :69-70. Rebuild mapping: **L057** (feed real router logits into compute_aux_loss, fix f_i, add gradient-flow test). |
| C-20 | PARTIAL | Lossless offload real: src/kv_cache.cpp:443-451 (evict_to_disk writes raw fp32 K/V via ofstream), :462-463 on_disk flag, :466-497 load_from_disk w/ memory-pressure LRU (:475-479), evict_lru :499-514, flush/load_all :534-552; RAM budget ctor param include/quant/kv_cache.h:76-78,:163; eviction trigger src/kv_cache.cpp:537-541 | Offload itself is (a) real and LOSSLESS (fp32 byte round-trip to file). But claimed "async pipeline" is (d) absent: grep `thread\|async\|std::future` in kv_cache.cpp = 0 hits — all disk I/O is synchronous ofstream/ifstream inside append/get_range hot path. NVMe-specific logic (d): none; default path is generic `$TEMP//tmp/InNova_*` files (:516-531). Test: tests/paged_kv_4m_test.cpp:65-74 does flush_to_disk→clear→load_from_disk with <1e-6 verification, registered as test_paged_kv_4m tests/CMakeLists.txt:181-182 (label "inference"). Edge gap: test's 256MB RAM limit vs tiny fixture never triggers mid-flight LRU eviction-under-pressure path. Rebuild mapping: **L054** (add background writer thread + prefetch queue; extend test to force LRU eviction under small memory limit). |
| C-21 | PARTIAL | src/transformer.cpp:132-184 scaling RotaryEmbedding ctor — Linear pos-interp :155-156, NTK-aware base rescale `theta*alpha^(hd/(hd-2))` :157-162, YARN ramp interpolation with beta_fast/beta_slow :163-175 (cosine smooth :172-173); wired: Attention ctor transformer.cpp:236-244; enum include/quant/types.h:19 `{None,Linear,NTK,YARN}`; config knobs include/quant/transformer.h:30-38 (yarn_attn_factor, yarn_beta_fast=32, yarn_beta_slow=1) | (a) Real implementation of all three modes, correctly wired into every Attention instance. BUT zero test coverage: grep `(?i)yarn|ntk|rope_scaling` under tests/ = 0 hits; nothing in tests/CMakeLists.txt exercises scaling modes (test_model/test_training use defaults). Untested edges: yarn_attn_factor accepted but ignored `(void)yarn_attn_factor` (transformer.cpp:138,:176), mscale/inter_len discarded same line — long-context output-quality parity therefore unproven. Rebuild mapping: **L056** (add RoPE-scaling unit test: freq-ratio assertions per mode + needle-in-haystack mini-bench). |

---

## Per-category classification recap

| ID | (a) real impl | (b) skeleton/stub | (c) comment/doc only | (d) absent |
|----|----|----|----|----|
| C-16 | MLAMoE low-rank block (wrong semantics) | use_mla/mla_rope_dim flags, never consumed | TRANSCRIPT.md:752,1620; docs/GOOD_FIRST_ISSUES.md:45 | DeepSeek-style latent-KV attention |
| C-17 | model/trainer/adapter code | — (wiring missing, not stubbed) | README.md:5039; CHANGELOG.md:19 | call sites in training loop; tests |
| C-18 | full stack (kernels→KV→inference→DDP→IO) | — | — | native DType::FP8 storage dtype |
| C-19 | load_balance_loss family | aux_loss path (dummy inputs ⇒ constant) | Switch/GShard formula comment moe_trainer.cpp:451-455 | autograd-graph coupling; strict f_i formula |
| C-20 | sync lossless fp32 disk offload + LRU | — | — | async pipeline; NVMe tiering; pressure-path test |
| C-21 | Linear/NTK/YARN freq math + wiring | yarn_attn_factor/mscale accepted-then-discarded | — | any test |

## Reproducible commands

```
rg -n "use_mla|mla_" --glob '!*.md'          # C-16: only header decls
rg -n "mtp_forward|mtp_loss\(" src/           # C-17: definitions only, no callers
rg -n "fp8_e4m3_quantize" src/math.cpp        # C-18 kernels
rg -n "compute_aux_loss" src/moe_trainer.cpp  # C-19: see :285-287 dummies
rg -n "thread|async|future" src/kv_cache.cpp  # C-20: 0 hits => sync only
rg -ni "yarn|ntk|rope_scaling" tests/         # C-21: 0 hits => untested
grep -n "test_paged_kv_4m\|test_quant_engines\|test_moe_training" tests/CMakeLists.txt
```

(No builds executed per task rules; ctest names above are registration-level evidence only.)

## FAKE→backlog traceability (task 2.6)

| Claim | Verdict | Wave-6 backlog item |
|-------|---------|---------------------|
| C-16 MLA | FAKE | **L050** |
| C-17 MTP | PARTIAL | **L051** |
| C-18 FP8 | VERIFIED | none needed |
| C-19 aux-loss | PARTIAL | **L057** |
| C-20 KV-offload async | PARTIAL | **L054** |
| C-21 YARN/NTK | PARTIAL | **L056** |

Nothing tracked by git was modified by this audit; only this new file under `.research/claims/` was created.
