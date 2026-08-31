# 03 — PROJECT MAP (InNova Engine, Full Technical Reference)

## 1. What InNova is

A **pure C++20, zero-dependency AI engine** with its own `.quant` mixed-precision weight format. No Python, no PyTorch, no GGUF, no BLAS — the tensor library, autograd engine, training stack, fine-tuning, inference, HTTP serving, and all SIMD kernels are handwritten. A model is *born*, *trained*, *fine-tuned*, and *served* entirely inside the `.quant` format. The 2026-08-31 Verdent session independently verified the code quality spread: the codec core is production-grade, the autograd/math layers are solid engineering, the attention modules are reference-grade, and the training-path quantization is shortcut-grade (see `10-code-review-findings.md`).

## 2. Repository statistics (verified 2026-08-25/31)

| Area | Count / Size |
|---|---|
| `src/` .cpp files | 176 total (~82K lines of .cpp), including `src/adapters/` (33 files, NOT built) and `src/gle/` (2) |
| `include/quant/` headers | 130 |
| `tests/` | 58 files, 56 registered test targets |
| `tools/` | 19 CLI tools |
| `bench/` | 11 benchmark executables |
| `engines/` | 41 files (inference/, quant/, trainer/dense, trainer/moe/*, trainer/multimodal/*) |
| `docs/` | 33 files incl. RESEARCH/ subfolder |
| Build targets | 26 libs + 14 tools + 12 benches + 4 sops/gle + 56 tests ≈ **111 targets**, 166 built exes in `build/` |
| Kernel LOC | ~9,862 lines across kernel_* + math_* + simd_math + flash_attention (25K+ mission pending) |

## 3. The .quant format system (v3, current truth)

### 3.1 Format inventory (FORMAT_COUNT = 105, `include/quant/types.h:22-67`)

| Family | Count | Naming | Notes |
|---|---|---|---|
| Base | 10 | Q1..Q32 (only Q32 at 32-bit) | affine/comp8 paths |
| K variants | 27 | Q1_K_L/M/H … Q24_K_L/M/H (9 widths × 3 tiers) | encoders identical to plain twins |
| GRP→G | 9 | Q1_G … Q24_G (renamed from _GRP) | compound: dominant tier + 16-bit fp16 per-32 scales |
| K_G | 27 | Q1_K_L_G … Q24_K_H_G | K × G compound |
| Half-BPW | 9 | Q_GRP_1_5 … (dot display "Q_G_1.5") | |
| Half_G | 9 | half × GRP | |
| MXQ | 7 | MXQ_3_5 etc. (display MXQ_3.5) | mixed-tier quad |
| MXQ_G | 7 | MXQ_3_5_GRP → MXQ_3_5_G | mixed-tier + GRP |

- Enum slot 19 = `Q4_K_L` (valid; the fuzz test's old "slot 19 is a gap" skip was a stale relic — fixed).
- TWI/QUAD families deleted project-wide (v3 re-arrangement, 2026-08-23; commits 390079a, a8b4a26).
- Naming law details in `02-iron-rules.md` Rule 12.

### 3.2 Codec internals (`src/block_codec.cpp`, 2295 lines — the heart)

- **comp8 compander**: tanh-companded 256-entry decode LUT, k=0.02 selected by measured PSNR sweep (k=1.0→0.02 evidence recorded in bench runs); comment explicitly says "k was selected by measured PSNR on the production bench; see commit evidence before changing it."
- **FP16 converter**: hand-written round-to-nearest-even with subnormal handling.
- **Scale fitting**: `comp8_fit_scale_fp16` (block_codec.cpp:400-406) does log-spaced coarse scan + golden-section refinement over **true MSE including clipping**, avoiding the LS-fit overload failure mode. Speed round 1 (commit 10735a6) made it least-squares-seeded + tight golden search (72→17 evals), added closed-form `q1_fit_scale_fp16`, and byte-aligned grp8 dequant reads. Result: encode 2-8× faster, PSNR within noise (worst −0.017 dB; Q_G_8.5 +0.32 dB improved).
- **is_slot mechanism**: 16 slots carry (bits−1)-bit indices to fund a 16-bit FP16 scale — the exact-BPW funding trick. Audit note: slot weights are NOT zeroed on decode in the lattice path despite the comment at :120-121 claiming "zeroed on decode and skipped on encode" — flagged risk.
- **BitWriter/BitReader**: LSB-first; BitWriter silently truncates writes beyond the buffer and BitReader returns zeros past the end — silent-corruption risk if any budget computation is off (defense is only the budget math itself).
- **Known dead statics** (audit): `comp8_scale_search` (:215), `f32_bits`/`f32_from_bits` (:158,:164), `level_value_r`/`nearest_level_r` (:407,:412).
- **Q24_K_L/M/H + Q24_K_L_GRP paths**: unreachable memcpy/assign code after `return true` (block_codec.cpp:1986, :2010, :2184, :2210) — leftover from an earlier wire layout.
- **Affine grid mismatch (unfixed)**: `level_value_affine` special-cases bits 2/3/8 and defaults to idx/15; `nearest_level_affine` special-cases 4/3/8 and defaults to round(v*3) — encode/decode mismatch for bits==1 and bits==6 confirmed at head.
- **MXQ_G per-32-vs-per-64 mismatch**: encoder fits scales on 32-weight windows (`nsc=(c+31)/32`) but the guard/read logic used /64 — 4 guard/read sites fixed 2026-08-26; residual: at n=256, MXQ_3.5_G still emits 121 B = 3.78 BPW vs the 3.5 claim (tier-shift cannot absorb the dominant-tier scale overhead).

### 3.3 Supporting format modules

- `format_registry.cpp` (594 L): real; `build_singles` = 10 base + 9 G only (K/half families live in types.h enum, not the singles planner list); `format_to_regformat` maps ALL 9 half-GRP formats → RegFormat::Q1 (bug) and collapses MXQ plain → GRP ids (:177-181).
- `format_planner.cpp` (340 L): importance indexing can OOB-read if activation array is shorter than weights (no length check); tier assignment sorts descending importance.
- `codebook.cpp` (452 L): serialize/deserialize magic mismatch FIXED (2026-08-25; roundtrip used to always throw).
- `quant_format.cpp` (624 L): QUANTReader format-id≥38→Q32 coercion (v3 relic) FIXED.
- `vector_quantizer.cpp` (395 L): EMA + residual VQ for the audio stack — real.
- `ste_quantizer.cpp` (282 L): static-codebook bug, see `10-code-review-findings.md`.
- `qat.cpp` (529 L): LSQ-style QAT — real.

## 4. Library layering + build system

- Explicit source lists (GLOB banned). Dependency chain: `quant_core` (tensor, bitio) → `quant_math` → `quant_kernel` → `quant_format` → `quant_model` (autograd, transformer, moe, kv_cache) → higher libs; tools/benches/tests on top.
- **Circular-dependency fix (B-2, 2026-08-25)**: `tensor.cpp` (quant_core) called `AutogradEngine` (quant_model) in its destructor → LNK2019 in 5+ targets. Fix: layering-safe hook — `namespace detail { extern void (*autograd_unregister_hook)(Tensor*); }` declared in `include/quant/tensor.h`, defined in `src/tensor.cpp`, installed by the engine on first use and cleared in the engine destructor; `~Tensor` only calls the hook when registered.
- **Orphan fix (F-7)**: `src/hybrid_expert.cpp` was compiled by NO target → added to quant_model.
- **CMake breaker fix (B-1)**: dead `test_bpw_150_proof` reference removed from `tests/CMakeLists.txt` (:179 registration + :222 dependency list).
- Compiler config: MSVC, `/O2 /DNDEBUG` Release, `/Od /Zi` Debug; `QUANT_AVX2` → `/arch:AVX2` + `-D__AVX2__` globally (`cmake/compiler.cmake:30-43`); AVX512 conditional (math_avx512 has elementwise + gemv only — NO GEMM).
- Version drift (unfixed): CMakeLists 0.1.2 / README v0.1.03 / CHANGELOG 0.1.02.
- CI: `.github/workflows/ci_full.yml` has ASan+UBSan ubuntu gcc-13 matrix (:127-138) but EXCLUDES heavy tests (test_protected, test_gpu, test_training, test_native_quant, test_moe_training, paged_kv_1t) from sanitizer runs; `Dockerfile:26` runs ctest with `|| true` (failures never fail the image build).

## 5. Model + inference stack

- **Transformer** (`src/transformer.cpp`, 894 L): pre-norm LLaMA-style block (RMSNorm → Attention → residual → RMSNorm → SwiGLU FFN), optional GPT-NeoX parallel residual (:493-512); activations SiLU/SwiGLU/GeGLU/GELU/ReLU (:428-481). GQA = materialized memcpy K/V expansion (:300-320) — correct, not grouped-GEMM. RoPE half-dim rotate-neox, AVX2 apply (:186-228); Linear/NTK/YARN modes all present; **YARN mscale/attention-factor discarded** via unnamed param :138 and `(void)mscale;(void)inter_len;` :176. FlashAttention dispatched for seq>64 (:322-331); B>1 bypasses cache with full-window attention (limitation).
- **flash_attention.cpp** (tiled online softmax, rescale :124-126, accumulate :128-164): parity <1e-3 vs naive (test_protected.cpp:118-173) — VERIFIED.
- **kda_attention.cpp** (132 L) + **mla_attention.cpp** (109 L): correct math, reference-grade; KDA per-token per-head std::vector allocs (delta_step), no chunked form; MLA `(void)cache` discard (mla_attention.h:47-51) → latent NEVER cached, "savings" are static formulas :105-106. Both headers self-document as "reference path recomputes full sequence".
- **hybrid stack**: hybrid_block/hybrid_moe_model/hybrid_scheduler + hybrid_expert.cpp (orphan-fixed into quant_model). HybridExpertShard ctor originally unlinked (test_hybrid_expert couldn't link).
- **kv_cache.cpp** (828 L): PagedKVCache 4M/1T (L1/L2 tables), Q4 quantized KV variant, disk offload LRU (evict_to_disk :443-463, load_from_disk :466-497). Append-extent bug FIXED (multi-token appends were clamped to 1 token on read). **Latent segfault open**: crash after clear() → load_from_disk() → final get_range (S9-S12 markers instrumented in tests/paged_kv_4m_test.cpp; session died mid-debug 2026-08-26).
- **Serving**: two separate HTTP implementations — ModelHTTPServer in production_core.cpp (/v1/completions + /completions, :619+) and http_server.cpp (978 L; /health, /v1/models, /v1/completions, /v1/chat/completions, /v1/embeddings; parses Content-Length :539-541). tools/quant_server.cpp (617 L) = hardened CLI server (8KB request-line→414, 64KB→413; status-text "Unknown" + single-recv body gaps = L074).
- **inference_opt.cpp** (1197 L): continuous batching is FAKE (mask built then never passed — :331-336, kv_caches_ created never used, greedy argmax, ignores seq_lens); spec-decode: p_draft=1/vocab hardcoded :240 (accepts everything), rewind_kv zero callers, DraftModel/TargetModel pure-virtual with 0 impls, SpeculativeDecoderV2 ghost declaration inference_opt.h:262-298; FP8 paths real (:690-765).
- **Tokenizers**: BPETokenizer real (train/save/load, bpe_tokenizer_bpe.cpp:64 naive pair-count BPE); Qwen35 loader reads HF tokenizer.json (load-only); k3_tokenizer branded — rename candidate in Transcender Phase 0. encode_image returns `{30000}` / encode_audio `{31000}` constants (multimodal.cpp:674-676).
- **model.cpp**: DenseModel + MTP heads allocated (:25-28) and registered (:93) with mtp_forward/mtp_loss (:395-435) — but ZERO call sites and no persistence in named tensors (dead wiring + save gap). Silent load skip: `assign` lambda no-ops on missing name OR numel mismatch (model.cpp:352-357) — truncated/corrupt .quant loads without error.

## 6. GPU / accelerator backends

- `include/quant/backend.h:7-31`: **24 BackendType values** (CPU_SCALAR … CPU_ZENDNN). Adapters in `src/backend.cpp` (1568 L) — one class per backend.
- Dynamic loading without SDKs (dlopen/LoadLibrary): CUDA runtime API (gpu_compute_cuda.cpp:707-722), Vulkan (:255-266), Metal (:166-170), SYCL (:204-216), HIP (:23-33). Probe/fallback: `probe_hardware()` backend.cpp:1301-1367, `select_optimal_backend()` priority chain :1369-1471 (CUDA > Metal > Vulkan > DX12 > SYCL > CANN > RPC > OpenVINO > VirtGPU > WebGPU > ZenDNN > CPU_AVX2 > IGPU_SHARED > RAM_SWAP > CPU_SCALAR).
- **GPU_VULKAN gap (unfixed)**: selection returns GPU_VULKAN at :1389 but the factory (:961-991) has NO case for it → silent fallback to CPUScalarBackend.
- gpu_compute_vulkan.cpp (1349 L): only 5 SPIR-V modules (5× magic 0x07230203); its gemm is a pure CPU passthrough (cpu_gemm always, :986-988). igpu_zero_copy.cpp = Vulkan unified-memory allocator (HOST_VISIBLE|HOST_COHERENT|DEVICE_LOCAL) for iGPU zero-copy training.
- Stub-class backends (small files, return-false probes): musa (97 L), hexagon (80), zdnn (83), virtgpu (83), webgpu (103), opencl (167), openvino (105), rpc (215).

## 7. Training stack

- **Autograd**: singleton `AutogradEngine` (autograd.h:57); ops recorded as AutogradNode{fn,inputs,outputs} keyed by raw data() pointer (weak_ptr); `backward()` = consumer-count/topological-stack walk (engine :352); seeds loss with ones; 12 hand-written backward kernels (matmul, add, silu, mul, RMSNorm, CE, RoPE, SDPA+GQA, bias, embedding, flatten, transpose) + gradient checkpointing (CheckpointFn, recompute in backward). Teardown is race-safe (destructor nulls hook before member teardown — independently praised 2026-08-31).
- **Trainer**: `trainer_core.cpp` fit loop with micro-steps; loss-scale math is CORRECT (the "128× shrink" rumor was investigated and NOT reproduced — audit A-13). Real bugs nearby: gradient accumulation re-uses the SAME batch across acc_steps (no fresh micro-batch); R-Drop loss added AFTER backward (report-only, no gradient).
- **RL**: GRPO real (graph-based per-sample advantage-weighted CE, trainer_rl.cpp:533-640 + test_grpo.cpp); PPO = shell (computes losses as floats, manually sets critic grads with wrong indexing, policy never updated — :160-297); DPO = shell (loss float, optimizer.step() with zero grads — :342-394); rollout_worker + reward.cpp (MLP + Bradley-Terry :111-130) real but zero test coverage; RLHFPipeline consumed only by trainer_rl_ops.
- **MoE**: `moe_variants.cpp` (1730 L) has ZERO autograd references — dispatch assembles outputs via raw pointers, router/expert gradient chain dead (A-10 CONFIRMED). Aux loss fed DUMMY {1,1} tensors (moe_trainer.cpp:281-295, constant ≈1.0); f_i formula wrong (:477, unnormalized exps vs own Switch comment :451-455); gradient coupling = scalar overwrite hack (:300-301). init_prefetcher declared moe_model.h:61-70 with NO impl and zero call sites (C-03 worse than ledger said). MoMMoE has a second router impl (mommoe_block).
- **Distributed**: DDP/FSDP/ZeRO single-node only (distributed.cpp:26-53,106-141,...); NCCL MISSING; shared-memory barrier deadlocks for world_size>1; zero tests/consumers (C-22 PARTIAL).
- **Data**: dataset/dataloader/mmap_dataloader/data_gen real; continual_engine has real EWC (fisher EMA+anchor :232-266) + compressed replay buffer with importance eviction (:91-100) — but seed-42 residuals scattered (continual_engine:532, ddp:332, inference_opt:175, image:56, agi_extended×3, moe_enhance:91, multimodal×3, multi_agent:736).
- **Fine-tuning**: RankAdapterEngine = LoRA-equivalent, in-graph ΔW=B·A, tested (fine_tuning.h:131-195, fine_tuning.cpp:542-652, test_fine_tuning.cpp:104-122); DoRA MISSING (removed 0.1.02).

## 8. Tests (56 registered via tests/CMakeLists.txt)

- **Current state: 52/56-class.** The 3 standing failures, all documented:
  1. `test_fuzz_codec` — enforces the BPW contract; red BY DESIGN until the owner BPW decision (currently also asserts supported-format count 104 and v3 caps; duplicates in kCaps were deduped).
  2. `test_quant_mix` — silent crash (exit 3, no stdout flush), built on deleted TWI fixtures; needs full MXQ redesign.
  3. `paged_kv_4m_test` — latent segfault now REACHABLE after the append-extent fix; instrumented S9-S12; crash between S12 (post clear+load_from_disk) and the final print; empty-tensor semantics suspected (numel=1 observed where 0 expected).
- Migrated to v3 (2026-08-25): test_all (37→105 + skip hole 19), test_mix_components (TWI-removed asserts), test_grp_quality_proof (GRP ids 37-45), test_mixed_precision_proof (direct GRP lookup), paged_kv_4m (structural invariants), test_fuzz_codec (caps ×104).
- **test_format_audit.cpp** = the BPW probe (created 2026-08-25) — exits 1 by design (31 violations) until the owner decision.
- Known-stale: `bench/bench_format_comparison.cpp` — still the OLD 43-format matrix with TWI rows; Debug-run numbers differ 10-50× from Release; the "Q8_G 58.88" speed1-era number is NOT reproducible at head (A-02 NOT REPRODUCIBLE). Direct probe showed codec correctness (Q8_G ≈ 46.41 > plain Q8 ≈ 41.39 @σ0.1). Until this bench migrates to the 105-format matrix and re-freezes baselines, every CSV-derived W-claim is frozen.

## 9. AGI / experimental modules (verdicts)

- agi_flywheel (1980 L): real harness mechanics (sandbox compile→test→benchmark→diff→rollback, cyclomatic complexity, timeout runner) BUT generation is template-based (32 hardcoded templates, :808-840), test harness partially fake (tests 2,3 pass by construction), "improvement" = append to src/asi_generated.cpp (aspirational). agi_guardrails.cpp = real SafetyGuardrails.
- code_gen.cpp (1254 L): real string-template kernel codegen (GEMM/attention/norm/activation × AVX2/AVX512/NEON).
- hpo_nas.cpp (1086 L): real algorithm zoo (random, Bayesian, successive halving, genetic, NAS mutate ops).
- meta_cognition/self_eval/world_model/multi_agent: real code, toy semantics (keyword matching).
- multimodal: real-ish vision transformer + cross-attn; T2I UNet proxy; constant tokenizers.
- **Persona violation**: agi_extended.cpp:756 (Rule 11).
