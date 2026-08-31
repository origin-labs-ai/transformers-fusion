# FULL HONEST AUDIT — InNova repo — 2026-08-25/26 — FINAL (v1)

> Auditor mode: most strict, zero trust. Solo deep-audit (sub-agents blocked by
> model concurrency limit all session) + fresh builds + full ctest + probe tests.
> All evidence below is reproducible from current working tree.

## EXECUTIVE VERDICT

Engineering core (block codec math, registry, GLE, autograd graph, FP8, GEMM
kernels) is REAL and often excellent. But the repo ships with: (1) a broken
build tree nobody could have ctest'd recently, (2) **31 formats storing more
bits than their names claim** — the exact crime BPW ironclad forbids,
(3) multiple headline features that are shells (DPO/PPO-policy, spec-decode,
MTP, MLA-cache, continuous batching mask), (4) a stale benchmark tool whose
numbers (incl. the W2 "beats GGUF" win) are not reproducible at HEAD, and
(5) test-suite rot: only 52/55 runnable after migrations, 2 needing redesign.

## A. BUILD WAS BROKEN AT HEAD — fixed by auditor

| ID | Finding | Evidence | Status |
|---|---|---|---|
| B-1 | tests/CMakeLists referenced deleted test_bpw_150_proof.cpp → CMake generate FAILED; tree unbuildable | tests/CMakeLists.txt:179-180,222 (old); git status ` D tests/test_bpw_150_proof.cpp` | FIXED (entry removed; deletion intentional — TWI_MIX API gone in v3) |
| B-2 | tensor.cpp (quant_core) called AutogradEngine (quant_model impl) → LNK2019×3 for every core-only target since commit 5d038b9 | src/tensor.cpp:97-98 old; /tmp/innova_build2.log; CMakeLists.txt:42 vs :111 | FIXED via layering hook `quant::detail::autograd_unregister_hook` (tensor.h/tensor.cpp/autograd_engine.cpp); semantics preserved (hook null before member teardown) |
| B-3 | src/hybrid_expert.cpp orphan — in NO CMake target; test_hybrid_expert could never link | root CMakeLists.txt had no hybrid_expert.cpp; LNK2019 build5.log | FIXED (added to quant_model) |
| B-4 | Entire src/adapters/ tree (33 files incl. quant_import, safetensors_bridge, gguf_bridge, probe tools) NOT built by root CMake | grep add_subdirectory adapters = none | OPEN — decide: wire in or document as dead |
| B-5 | Windows parallel-link resource exhaustion (rc.exe LNK1327, link 0xC0000142) at -j8 | /tmp/innova_build4.log | Not a repo bug; use -j3 |

## B. BPW IRONCLAD — 31 VIOLATIONS (CRITICAL, owner decision required)

New permanent probe test `tests/test_format_audit.cpp` prints claimed vs
actual bytes for all 104 defined formats (fresh output, exit 1 while violated):

- Q2_G 2.625 actual vs 2.0 claim (+0.625!) — block_codec.cpp:1944 affine claimed=2.625
- Q3_G/Q4_G +0.5 (affine 3.5/4.5) — :1941/:1938
- Q6_G & Q6_K_{L,M,H}_G 6.5625 vs 6.0 (+0.5625) — quant_6k budget :971
- Q8_G & Q8_K_*_G 8.5 vs 8.0 (compound) — grp8_compound_fits :357 enforces exactly 8.5n bits
- Q12_G & Q12_K_*_G 12.5 vs 12.0 — :879-881
- Q16_G 16.5 vs 16.0; Q24_G 24.5 vs 24.0 (plain+mean-residual tail)
- Q2/Q3/Q4_K_*_G +0.4375 (grp16 fallback @X.5 budgets)
- MXQ_{3.5,4.5,6.5,8.5,12.5}_G +0.125..+0.28 (budget guard computes scale
  bytes per-64 but writer emits per-32 — guard/writer mismatch, quad_mix ~:1663 vs :1703)

Consumer harm: FormatRegistry::select_best_format gates eligibility on CLAIMED
bpw (format_registry.cpp:329) while storing ACTUAL bytes → user asking ≤8.0
silently receives 8.5. TRANSCRIPT PART-AD note claims "zero violations found"
while its own tables list Q8_G=8.50 — the document contradicts types.h.
Honest counterweights that DO exist: FormatRegistry::actual_bpw() reports true
bytes; allocate_mix_blocks spends under hard budget using actual bytes;
half-BPW variants (Q_G_X_Y, MXQ plain) are honest (probe: all ok);
Q16_K_G/Q24_K_G are honest plain-ties.

DECISION REQUIRED (owner): (a) make encoders fit claims (quality will drop on
those rows), or (b) rename/re-map claims to actual (Q8_G ≡ Q_G_8.5 etc.),
or (c) keep dual naming but make format_bpw/select_best_format report truth.
Recommendation: (c) short-term + (a) long-term per iron rule.

Also CRITICAL and now FIXED: Q6_K_*_G truncated its 210B payload to 192B —
scales discarded, decode produced raw level ints (garbage). Fixed by mirroring
Q6_G path (no resize).

## C. FRESH BUILD + FULL CTEST (current tree, after auditor fixes)

- Build: 0 errors, all 26 libs + tools + benches + 56 test targets.
- ctest -C Debug: **52/56 PASS** across runs; final state:
  - PASSING incl. previously never-runnable test_tensor, test_math,
    test_sha256_corrupt, test_hybrid_expert.
  - MIGRATED TO v3 BY AUDITOR (were failing on stale assumptions):
    test_all (37→105, skip hole 19), test_mix_components (TWI removed by
    design), test_grp_quality_proof (enum ids 10..18→37..45; explicit
    Q4_G lookup), test_mixed_precision_proof (explicit GRP descriptor),
    paged_kv_4m capacity assert (structural invariant),
    test_fuzz_codec caps table extended to all 104 formats + count 104.
  - REMAINING FAILURES (3):
    1. test_fuzz_codec — NOW CORRECTLY ENFORCES the BPW contract:
       "Q2_G payload 84B exceeds claimed 64B" — this is finding B above,
       failing by design until owner decision. (Fuzz is doing its job.)
    2. test_quant_mix — silent abort (exit 3, TWI-era fixtures; get_twi_mix()
       returns empty post-v3). Needs fixture redesign onto MXQ descriptors.
       Mapped to Phase-3 scope.
    3. test_paged_kv_4m — SEGFAULT in previously-unreachable code after
       auditor fixed the append/current_pos bug (see D-2). Latent crash in
       far-page/disk-offload path needs dedicated debugging.

## D. REAL BUGS FOUND & FIXED THIS SESSION

1. Codebook serialize/deserialize magic mismatch — round-trip always threw.
   serialize wrote 0x51554138/"QUA8", deserialize demanded 0x4F494C38 (and
   0x51554134 vs 0x4F494C34 for QUANT4). codebook.cpp:204,:353. FIXED to match.
2. PagedKVCacheBase::append advanced current_pos by exactly 1 token even when
   appending a whole block → get_range clamped reads to 1 token. kv_cache.cpp:384.
   FIXED: advance by tokens actually written (k_num/(heads*dim)); single-token
   behavior unchanged. NOTE: fixing this unblocked the test into a deeper
   latent segfault (see C-3 above).
3. QUANTReader::read_block coerced every format id ≥38 to Q32 (pre-v3 relic,
   quant_format.cpp:410,:521) → silent garbage decode of GRP/half/MXQ blocks
   from .quant files. FIXED: valid range check against FORMAT_COUNT (hole 19
   still rejected → Q32 fallback for corrupt tables).

## E. SHELLS & FAKE/DEAD CONFIRMED AT CURRENT LINES (not fixed — feature work)

| Wound | Current evidence |
|---|---|
| DPO train_step never backprops; calls optimizer_->step() on stale grads | trainer_rl.cpp:342-394 (loss value computed; no grad writes; step at :389) |
| PPO policy leg trains nothing — only manual critic-head grads; policy loss produces no gradients | trainer_rl.cpp:160-297 |
| Continuous batching drops attention mask (built then unused) | inference_opt.cpp:333-334 (`mask` constructed, forward called without it) |
| Spec-decode acceptance hardcoded p_draft=1/vocab → accepts everything | inference_opt.cpp:240 |
| SpeculativeDecoderV2 ghost declaration, zero definitions | include/quant/inference_opt.h:262 |
| KV rewind/checkpoint never called by anyone | only decl speculative_decoder.h:102; zero call-sites |
| MoE aux-loss fed dummy tensors; LB loss real but coupling is scalar overwrite | moe_trainer.cpp:281-295 |
| moe_variants.cpp (1730L) has ZERO grad/backward references — routing breaks autograd chain | grep verified |
| init_prefetcher declared, never defined, never called (prefetcher never constructed) | moe_model.h:65; zero src hits |
| MTP heads+loss implemented, ZERO callers | model.cpp defs; trainer_core.cpp:742 wrapper; no invocation |
| MLA latent cache discarded — forward takes KVCache& then (void)s it | mla_attention.h:48-49 |
| YARN attn-factor/mscale discarded | transformer.cpp:138 (unnamed param), :166 mscale=1.0, :176 (void)mscale |
| encode_image returns constant {30000} | multimodal.cpp:674-676 |
| AGI flywheel self_play = round-robin over 32 hardcoded coding prompts | agi_flywheel.cpp:808-840 |
| GPU_VULKAN selectable (:1389-1390) but factory has no case → silent fallback | backend.cpp factory cases :963-988 (no GPU_VULKAN/GPU_HIP/etc.) |
| quant_engines_core.cpp = dead file (5 static AVX2 fns, zero callers) | grep verified |
| seed-42 hardcoded RNGs at ~25 sites (agi, mommoe, multimodal×6, quant_engines×3, ddp, inference_opt, continual, multi_agent, self_eval…) | grep list in session log |
| Loss-scale "~128x shrink" wound | NOT REPRODUCED — unscale math correct at trainer_core.cpp:465-479, training_utils.cpp:333-341; likely fixed in interim |

## F. BENCHMARK TOOL IS STALE POST-V3 (evidence-integrity issue)

Fresh run of bench/bench_format_comparison.cpp (Debug, build/ cwd):
- Only 43 rows/dataset — never extended to the v3 105-format matrix.
- Still benchmarks "Q_TWI_MIX@1.5/@2.5(_G)" — formats DELETED in v3;
  get_twi_mix() now returns empty descriptors → those rows measure who-knows-what.
- Its Q8_G gaussian row (50.23 dB) matches NEITHER speed1-era CSV (58.88)
  NOR direct codec probe at same σ (46.41 @σ0.1 single-block; compound taken,
  272B, beats plain Q8 by +5 dB). Conclusion: bench outputs are unreliable
  until migrated; ALL CSV-derived claims (W2 "beats GGUF Q8_0", W3, speed
  tables) require re-measurement. Ledger C-01's "CSV 105 names == registry"
  is therefore STALE too.
- Probe-vs-bench absolute PSNR gap (plain Q8 41.4 one-block vs 54.66 whole-
  vector) also unresolved — flag for the bench-migration task, do not trust
  either number until the harness is regenerated and reviewed.

## G. SPEED WORK DONE (decode-first, bit-exactness preserved)

Added FastBitReader (word-buffered LSB-first, identical semantics incl.
past-end zeros) and swapped it into 7 decode hot paths: dequant_lattice,
dequant_fixed_codebook, dequant_grp16, dequant_6k, dequant_affine,
dequant_grp12_compound, dequant_quad_mix. Encode side unchanged. All codec
tests + fuzz NMSE caps pass → reconstruction quality unchanged.
HONEST CAVEAT: absolute decode-speed delta NOT yet measured because the bench
harness itself is stale (F above) and Debug-build timings are meaningless vs
the Release-bench baselines. Measuring properly = build Release +
bench-migration first. This is the top item of the follow-up queue.

## H. POSITIVE VERIFICATIONS (Slice-1 deep read + spot checks)

REAL and well-engineered: fp32↔fp16 RNE conversions; true-MSE golden-search
scale fitting incl. clipping; closed-form Q1 fit; Lloyd-Max Q2/Q4 tables;
N(0,1)-quantile fixed codebooks (zero transported bytes); signed-min affine
grouping (genuine improvement vs industry unsigned-min); companded-Q8 LUT;
MXQ budget-guard concept; startup MSE measurement on 3×16384 samples
(deterministic); allocate_mix_blocks greedy benefit-per-byte under hard
actual-bytes budget; SHA-256 (FIPS constants correct) + hardened QUANTReader
bounds checks; GLE hash-chained telemetry; AVX2 gather/LUT kernels in
quant_engines_fp.cpp (E4M3/E5M2 conversions correct); kernel_gemm scalar/
AVX2/tiled trio with proper fallbacks; VectorQuantizer EMA/RVQ audio stack.

## I. DOC CONTRADICTIONS (TRANSCRIPT.md read fully, 2048 lines)

1. PART-AD note "(4) BPW values verified against ironclad rule — zero
   violations found" vs same table listing Q8_G=8.50/Q6_G=6.56 — false
   statement; probe shows 31 violations.
2. PART-AD board says Q3_G "BROKEN -14 dB FIX top bug" — stale; fixed long
   ago (grp_proof_test: 29.2 dB ✓) and "NEXT MILESTONE L010" note likewise stale.
3. PART-C ledger table shows all PENDING — real verdicts live in
   research/claim_ledger.md (acceptable pointer, but header text misleads).
4. PART-O baseline numbers frozen from a pre-recovery tree era; several not
   reproducible at HEAD (ties to F).
5. Workbench Blockers section ("only test_quant_mix + test_fuzz_codec fail")
   understated reality (was 7, now 3 with different causes).

## J. FILES TOUCHED BY AUDITOR (all working-tree, uncommitted)

Fixes: tests/CMakeLists.txt, include/quant/tensor.h, src/tensor.cpp,
src/autograd_engine.cpp, CMakeLists.txt (+hybrid_expert.cpp), src/codebook.cpp,
src/quant_format.cpp, src/block_codec.cpp (Q6_K_G + FastBitReader),
src/kv_cache.cpp (append extent).
Test migrations: test_all.cpp, test_mix_components.cpp,
test_grp_quality_proof.cpp, test_mixed_precision_proof.cpp,
paged_kv_4m_test.cpp, test_fuzz_codec.cpp (caps ×104 + counts).
New: tests/test_format_audit.cpp (BPW probe — currently exits 1 BY DESIGN
until the BPW decision is made; flip verdict logic or fix encoders then).
Docs: this file.

## K. RECOMMENDED NEXT QUEUE (priority order)

1. Owner decision on BPW contract (B) → either fit-in-claim encoders or truth
   in format_bpw()/select_best_format; then test_fuzz_codec + test_format_audit go green.
2. Bench migration to v3 matrix (fix TWI rows, extend to 105, regenerate
   baseline; THEN re-run Release decode benchmark to quantify FastBitReader).
3. Debug paged_kv segfault (far-page/disk path) — now reachable thanks to D-2.
4. test_quant_mix fixture redesign onto MXQ.
5. Decide adapters/ fate (B-4); wire in or archive.
6. Feature rebuilds per ledger backlog (DPO/PPO backprop, mask, p_draft,
   MLA cache, MTP callers, prefetcher) — already mapped to Waves 5-7.
