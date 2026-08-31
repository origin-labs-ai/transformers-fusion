# 06 — ZCODE SESSIONS HISTORY (29 sessions, Aug 24-29 2026, 1013 messages)

Raw material: `repo/sessions/_zcode_sessions.txt` (709,297 chars, full dump) and `repo/sessions/_zcode_condensed.txt` (287,014 chars, condensed). The ZCode persistent memory originals are copied at `repo/sessions/zcode-memory-original/`. This file is the structured, session-by-session chronology with every finding, fix, decision, and quote — the pre-history that the current repo state rests on.

---

## AUG 24 — Exploration Day (the first full-contact session)

**What was ordered:** "Hinglish bol be saale!" (language lock) → "topoora project ka har ek file khaa jaa!" (absorb the entire project) → read TRANSCRIPT.md fully and finish it → run honest sub-agent audits → check whether TWI/QUAD formats really exist → **"5 alag alag baar khud se baar baar audit"** (five full-project audits).

**How it ran:** Six sub-agent scopes were defined — (1) quant core, (2) training stack, (3) GPU backends, (4) AGI modules, (5) tools/tests/docs, (6) model+inference. Two agents (GPU, AGI) finished with no usable output and were re-run; the rest returned real reports. All findings below come from those reports + the solo passes.

**GPU backend findings (report):**
- `include/quant/backend.h:7-31`: 24 BackendType values (CPU_SCALAR … CPU_ZENDNN).
- **GPU_VULKAN selection gap (never fixed):** `select_optimal_backend()` can return GPU_VULKAN (backend.cpp:1389) but the factory (:961-991) has no case for it → silent CPUScalarBackend fallback.
- gpu_compute_vulkan.cpp (1349 L): only 5 SPIR-V modules; its `gemm` is a pure CPU passthrough (cpu_gemm always, :986-988).
- math_avx512: elementwise + gemv only — no GEMM implementation.
- Stub backends (small files, return-false probes): musa 97 L, hexagon 80, zdnn 83, virtgpu 83, webgpu 103, opencl 167, openvino 105, rpc 215.
- Real dynamic loaders without SDKs: CUDA runtime API (gpu_compute_cuda.cpp:707-722), Vulkan (:255-266), Metal (:166-170), SYCL (:204-216), HIP (:23-33). Probe chain at :1301-1471.

**Codec findings (report):**
- Unreachable memcpy/assign leftovers after `return true` in Q24 paths: block_codec.cpp:1986, :2010, :2184, :2210.
- Dead statics: comp8_scale_search :215, f32_bits/f32_from_bits :158/:164, level_value_r/nearest_level_r :407/:412.
- Fuzz test only covered formats 0-37, skipping slot 19 entirely (stale "gap" belief; slot 19 is actually Q4_K_L).
- kCaps duplicate rows.

**AGI module findings (report):**
- agi_flywheel (1980 L): sandbox compile→test→benchmark→diff→rollback mechanics are real, cyclomatic complexity real, timeout runner real. BUT generation is 32 hardcoded templates (:808-840); test harness partially fake (tests 2,3 pass by construction); "self-improvement" = append to src/asi_generated.cpp (aspirational).
- code_gen.cpp (1254 L): REAL kernel codegen (GEMM/attention/norm/activation × AVX2/AVX512/NEON string templates).
- hpo_nas.cpp (1086 L): REAL algorithm zoo (random/Bayesian/successive-halving/genetic/NAS-mutate).
- multimodal: encode_image returns `{30000}` (multimodal.cpp:674-676), encode_audio `{31000}`; T2I is a UNet-shaped proxy; vision transformer + cross-attn real-ish.
- agi_guardrails.cpp: real SafetyGuardrails.

**Training findings (report):**
- PPO = metric shell: losses computed as floats, critic grads set with wrong indexing, policy NEVER updated (trainer_rl.cpp:160-297).
- DPO = shell: loss float, `optimizer.step()` with zero grads (:342-394).
- GRPO = REAL (graph-based per-sample advantage CE, :533-640, test_grpo.cpp).
- MoE dispatcher autograd break: raw pointer assembly, no gradient node (later confirmed A-10).
- init_prefetcher: header-only declaration (moe_model.h:61-70), zero impls, zero call sites.
- Aux loss fed DUMMY {1,1} tensors (moe_trainer.cpp:281-295); f_i formula wrong (:477); gradient coupling via scalar overwrite (:300-301).
- Gradient accumulation re-uses the SAME batch across acc_steps.
- R-Drop loss added AFTER backward (report-only).
- NCCL absent; MoETrainer world_size>1 deadlock hazard; ZeRO/FSDP/DDP single-node only.

**Inference findings (report):**
- Continuous batching FAKE: mask built then never passed (inference_opt.cpp:331-336), kv_caches_ created never used, greedy argmax, seq_lens ignored.
- Speculative decoding: Variant A unusable (DraftModel/TargetModel pure-virtual, zero impls; rewind_kv zero callers); Variant B p_draft=1/vocab hardcoded (:240) so acceptance is meaningless; SpeculativeDecoderV2 ghost declaration (inference_opt.h:262-298).
- YARN mscale/attention-factor discarded (transformer.cpp:138 unnamed param, :166, :176 `(void)mscale;(void)inter_len;`).
- GQA via memcpy K/V expansion (transformer.cpp:300-320) — correct but not grouped-GEMM.
- MTP heads allocated and registered with zero callers (model.cpp:25-28, :93, :395-435; trainer_core.cpp:742).
- Two separate HTTP servers (production_core.cpp ModelHTTPServer vs http_server.cpp 978 L).
- BPE tokenizer real (train/save/load); Qwen35 loader load-only.
- FP8 inference paths real (inference_opt.cpp:690-765).

**Persona violation found:** `class PersonaHotSwap` at src/agi_extended.cpp:756 (load_persona/swap_to/current) violates the system-prompt-only personality rule. Owner verdict: "abe persona to system prompt se banega naa! Wo code me kya karne gayaa?!" and the process decision "Ya to hataa ya fir TRANSCRIPT me daal de kyuki fir usi se fix start hoga!" — documented (not silently deleted).

**CSV wins verified (fresh bench, all rows re-measured):**
- W1: Q16 vs FP16 +15.98 / +18.22 dB (σ=0.05 / σ=0.10).
- W2: Q8_G vs GGUF Q8_0 +0.62 / +0.74 dB (Q8_G row 58.88 — later flagged NOT REPRODUCIBLE at head; see Aug 25).
- W3: Q6_G vs GGUF Q6_K +0.19 dB.
- CSV shape: 105 InNova + 7 [ref] rows (GGUF Q4_K/Q6_K/Q8_0/FP16/INT8/BitNet b1.58/Binary), no TWI.

**Memory discovery moment:** the agent referenced its persistent memory folder; owner was genuinely startled — "Ye kaise pata tumhe?!" — and then ordered the memory content preserved. (This inspired the repo/memory/ system built on Aug 31.)

**test_bpw_150_proof.cpp deletion:** owner confirmed intentional and PERMANENT ("Jis file me 150 kahi pe laga hai wo deleted rahenge!").

**Goal-status audit:** old goal_status.json claimed "256 tasks DONE"; the live ledger contradicted it with 2 failures; the probe found the truth was worse (7 failures). Lesson recorded: status files must be regenerated from telemetry, not narrated.

## AUG 25 — The Big Audit Session (solo, deepest single session)

**Trigger:** "Abe saale, tujhe 5 baar poore project ka har ek file ka honest audit karne bheja tha! Aur tune sirf 2-3 reports di! Ab jaake poora honest audit kar! ... Abhi bhi asli mean nahi pakra hai tu!" — the 5x full audit demand.

**Constraint hit:** sub-agents blocked by the environment ("model concurrency limit exceeded") → **solo file-by-file audit** became the proven fallback (owner later capped parallel agents at 2).

**Build breakers found and fixed (B-1/2/3):**
1. **B-1:** dead `test_bpw_150_proof` references — removed from tests/CMakeLists.txt registration (:179) and dependency list (:222).
2. **B-2:** tensor→AutogradEngine circular link — `~Tensor` (quant_core) called AutogradEngine methods (quant_model) → LNK2019 across 5+ targets. Fix: layering-safe hook — `namespace detail { extern void (*autograd_unregister_hook)(Tensor*); }` in tensor.h, defined in tensor.cpp, installed by the engine on first use, cleared in the engine destructor; `~Tensor` calls the hook only when registered.
3. **B-3:** orphan `src/hybrid_expert.cpp` compiled by no target → added to quant_model sources.

**First fresh build: 0 errors.** First fresh ctest: **48/55 → 7 failures** (the ledger had claimed 2 — itself recorded as an audit finding: "ledger understates reality").

**Stale tests migrated to v3 (6):** test_all (37→105 formats, hole-19 skip removed), test_mix_components (TWI-removed asserts), test_grp_quality_proof (GRP ids 37-45), test_mixed_precision_proof (direct GRP lookup), paged_kv_4m (structural invariants), test_fuzz_codec (caps ×104).

**The BPW probe (tests/test_format_audit.cpp) — the session's centerpiece:** wrote a per-format payload-overflow audit; **31 violations proven**: integer GRP family +0.4375..+0.625 (Q2_G 2.625 vs 2.0; Q3_G 3.5; Q4_G 4.5; Q6_G 6.5625; Q8_G 8.5; Q12_G 12.5; Q16_G 16.5; Q24_G 24.5), K_G variants (2.4375/3.4375/4.4375/6.5625/8.5/12.5), and 5 of 7 MXQ_G (3.78 vs 3.5; 4.63 vs 4.5; 6.69 vs 6.5; 8.69 vs 8.5; 12.66 vs 12.5 — root cause: encoder scales per-32 weights while guards/readers assumed per-64). Fuzz's live catch: "Q2_G payload 84 B exceeds claimed budget 64 B (+1)".

**Real bug fixes (codec):**
- codebook serialize/deserialize magic mismatch (roundtrip always threw).
- QUANTReader format-id ≥ 38 → Q32 coercion (v3 relic).
- PagedKVCache append extent: multi-token appends were clamped to 1 token.
- Q6_K_G payload truncation: 210 B → 192 B cut.

**Speed work:** FastBitReader word-buffered decode (bit-exact) wired into 7 decode paths (later shipped in commit 10735a6 as "speed round 1": encode 2-8× faster, PSNR within noise, worst −0.017 dB, Q_G_8.5 +0.32 dB).

**Bench staleness discovered:** bench_format_comparison.cpp is still the old 43-format matrix with TWI rows; Debug-run numbers differ 10-50× from Release; the "Q8_G 58.88" speed1-era number is NOT reproducible at head. Direct codec probe proved correctness (Q8_G ≈ 46.41 > plain Q8 ≈ 41.39 @ σ0.1) — the HARNESS is unreliable, not the codec. Consequence: all CSV-derived W-claims frozen until bench migration.

**The rename begins + hard stop:** owner ordered `_GRP` → `_G` across all files; ~28/43 done when owner called it: "Bas yaar, bas ab ruk jaa! Ab ho gaya! STOP EVERYTHING SUDDENLY!" — session paused at the rename's midpoint.

**The full audit report was finalized** at `.research/audit_full_20260825.md` (now `research/audit_full_20260825.md`) and shown to the owner in chat ("Abe tu Audit report mujhe yahaa pe dikhaa!") followed by "Ab mere demands ke hisaab se fix karne pe lag jao!"

## AUG 25-26 — Strict auditor sub-sessions + rename completion

- Three read-only strict-auditor slices re-verified verdicts: codec core (36 files) and training stack (41 files) confirmed the earlier findings with line numbers; no verdict reversals.
- **Rename COMPLETED** (43+ files, all consumers, CMake, tests, docs) + **BPW truth-fix**: `format_bpw()` now returns actual wire values (Q2_G→2.625, Q6_G→6.5625, Q8_G→8.5, K_G family matched), format_registry singles table synced, quad_mix guards corrected per-64→per-32 (4 sites in block_codec.cpp). "Rebuild with BPW truth fix" → exit 0.
- Honest-reporting debate left open: MXQ_3.5_G still emits 3.78 actual BPW at n=256 (dominant-tier scale overhead cannot fit 3.5 under any tier shift) — the fit-vs-rename owner decision was born here.

## AUG 26 — TRANSCRIPT completion + paged_kv segfault hunt (final substantive ZCode session)

**Order:** "TRANSCRIPT.md padh and complete kar! Har kaam complete kar!"

**State reconciliation:** detected the uncommitted fix set; fresh build + ctest = **52/55** (the standing baseline).

**paged_kv_4m segfault debugging (UNRESOLVED — session died mid-hunt):**
- After the append-extent fix, the previously masked latent crash became reachable.
- Instrumented markers S9-S12: flush_to_disk → clear() → load_from_disk() → final get_range.
- clear() completes; the crash lands AFTER S12, between the final get_range and the print.
- Mystery data point: an expected-empty tensor reports numel=1 (empty-tensor semantics suspect).
- Session terminated mid-debug; the test remains one of the 3 standing failures.

**test_fuzz_codec cleanup:** format-count assertions updated (37→104 supported), v3 caps table aligned, kCaps deduped.

**BPW decision documented as pending:** F-1 is enforced by fuzz (red until decided).

## AUG 26 (parallel tool) — OpenCode SESSION-00

Format-naming session (61 messages) — full details in `07-opencode-session00.md`. Produced the 105-format naming architecture and the `_GRP`→`_G` design that landed as commits 390079a + a8b4a26.

## AUG 27-29 — Idle ("Hello!" sessions only)

No work sessions. The tree sat untouched for ~5 days with ~60 uncommitted files until the Verdent session of Aug 31.

---

## MASTER WOUND LIST (23 items, file:line, as of 2026-08-31)

1. **MoE routing gradient chain dead** — moe_variants.cpp (1730 L) has zero autograd references; dispatch via raw pointers.
2. **MoE aux loss dummy** — moe_trainer.cpp:281-295 {1,1}; f_i wrong :477; scalar-overwrite coupling :300-301.
3. **Continuous batching fake** — inference_opt.cpp:333-334 mask never passed.
4. **Spec-decode shells** — p_draft=1/vocab :240; rewind zero callers; V2 ghost inference_opt.h:262-298.
5. **GPU_VULKAN factory gap** — backend.cpp ~:961, :1388-1390 silent scalar fallback.
6. **Multimodal constant tokenizers** — multimodal.cpp:674-676 (`{30000}` / `{31000}`).
7. **AGI flywheel templates + fake harness** — agi_flywheel.cpp:808-840; tests 2,3 by-construction.
8. **PPO/DPO no backprop** — trainer_rl.cpp:160-297 (critic-only, wrong indexing), :342-394 (zero-grad step).
9. **YARN mscale discarded** — transformer.cpp:138, :166, :176.
10. **MTP zero callers** — model.cpp:25-28, :93, :395-435; trainer_core.cpp:742.
11. **MLA cache discard** — mla_attention.h:47-51 (`(void)cache`).
12. **Persona violation** — agi_extended.cpp:756 (Rule 11).
13. **adapters/ orphan tree** — 33 files, own RegFormat universe (TWI/QUAD era), not in CMake; fate = owner decision.
14. **quant_engines_core.cpp dead file** — all-static, no callers, no target.
15. **init_prefetcher header-only** — moe_model.h:61-70.
16. **seed-42 residuals** — continual_engine:532, ddp:332, inference_opt:175, image:56, agi_extended ×3, moe_enhance:91, multimodal ×3, multi_agent:736.
17. **Version drift** — CMake 0.1.2 / README 0.1.03 / CHANGELOG 0.1.02; README:147 stale "15 formats, 8 twi-mix".
18. **Gradient accumulation same-batch** — no fresh micro-batch per acc step; loss-scale math itself CORRECT (A-13 rumor busted).
19. **Silent load skip** — model.cpp:352-357 assign lambda no-ops on missing name or numel mismatch (corrupt .quant loads silently).
20. **Bench stale** — bench_format_comparison.cpp 43-format matrix + TWI rows; Debug/Release mixing; W2 number not reproducible (A-02).
21. **CI masking** — Dockerfile:26 `ctest || true`; sanitizer runs exclude heavy tests.
22. **paged_kv latent segfault** — post-S12, unresolved (see Aug 26).
23. **BPW 31 violations** — partially truth-fixed; MXQ_G @n=256 residual (3.78 vs 3.5).

## MASTER FIX LEDGER (what was actually repaired, with mechanism)

- B-1 bpw_150 refs removed (tests/CMakeLists.txt:179, :222) · B-2 layering hook (tensor.h/tensor.cpp/autograd_engine.cpp) · B-3 hybrid_expert.cpp into quant_model
- Codebook magic match · QUANTReader ≥38→Q32 coercion removed · PagedKV append extent · Q6_K_G truncation (210→192 B)
- 6 stale tests migrated to v3 · test_format_audit BPW probe created (31 violations) · fuzz ×104 + caps + dedupe
- FastBitReader word-buffered decode (7 paths, bit-exact) → speed round 1 (commit 10735a6)
- `_GRP`→`_G` full rename + BPW truth fix (format_bpw actual values, registry singles, 4 guard/read sites per-32)
- Full audit report research/audit_full_20260825.md; ledger C-01..C-21 (+ later C-22..C-25)

## OPEN OWNER DECISIONS (unchanged queue)

1. BPW: fit-in-claim vs rename-to-actual (Rule 1 gate)
2. `src/adapters/` fate: wire in vs archive
3. Bench migration timing (43→105 matrix) + baseline re-freeze
4. test_quant_mix fixture design (MXQ)
5. Transcender Phase 0 start
6. Committing the ~60-file uncommitted working tree
