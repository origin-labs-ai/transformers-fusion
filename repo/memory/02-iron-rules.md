# 02 — IRON RULES (TRANSCRIPT.md PART-B, BINDING ON EVERY AGENT)

These rules were laid down by the owner in TRANSCRIPT.md (Master Plan v2, Gauntlet Edition, ~2048 lines, read fully on 2026-08-24/25/26) and re-confirmed through the audit sessions. They bind every agent that touches this repository. Rule violations found in the wild are recorded in `research/claim_ledger.md` and `06-zcode-sessions-history.md`.

---

## Rule 1 — BPW Ironclad (the first rule)

**Statement:** A quantization format must never store more bits per weight than its name claims. Improving quality by raising the bit budget is cheating. Zero tolerance.

**Origin:** The GRP family historically stored +0.5 BPW over its claim while the bench compared at the ACTUAL budget (which made the comparison fair but the public claim a lie). TRANSCRIPT itself admitted the actual budgets in the W2/W3 win register ("same 8.5 BPW") while `types.h` claimed 8.0 — an internal contradiction documented by the audit.

**Current status (2026-08-31):**
- `tests/test_format_audit.cpp` (the BPW probe, created 2026-08-25) proved **31 violations**: integer GRP family at +0.4375..+0.625 (Q2_G 2.625 vs 2.0, Q3_G 3.5, Q4_G 4.5, Q6_G 6.5625, Q8_G 8.5, Q12_G 12.5, Q16_G 16.5, Q24_G 24.5), K_G variants (2.4375/3.4375/4.4375/6.5625/8.5/12.5), and 5 of 7 MXQ_G (3.78 vs 3.5, 4.63 vs 4.5, 6.69 vs 6.5, 8.69 vs 8.5, 12.66 vs 12.5 — caused by the per-32-vs-per-64 guard/writer mismatch at n=256).
- The probe exits 1 BY DESIGN until the owner decision is made.
- A partial "truth in reporting" fix landed: `format_bpw()` now returns actual wire values; `format_registry.cpp` singles table matches; the quad_mix guards were corrected per-64 → per-32 (4 sites in `block_codec.cpp`).
- Remaining known gap: MXQ_3.5_G still produces 3.78 actual BPW at n=256 because the per-32 scale overhead of the dominant tier cannot fit the 3.5 budget under any tier shift.
- `test_fuzz_codec` enforces this contract live ("Q2_G payload 84 B exceeds claimed budget 64 B (+1)") and stays red until the decision.
- **OWNER DECISION PENDING:** (a) make encoders fit the claim, or (b) rename formats to actual budgets. Do not pick unilaterally.

## Rule 2 — Adafactor only, AdamW banned

The training optimizer is Adafactor (real implementation at `optimizer.cpp:333-398`, factorized second moment). AdamW is banned anywhere in the training path. Wired into `trainer_core.cpp:94-123` and `moe_trainer.cpp:69-78`. Known gap (ledger C-02): the autograd leg has zero Adafactor references and zero direct tests.

## Rule 3 — STE mandatory for quant-aware paths

Straight-Through Estimator is required wherever quantization meets training (train-in-format). Related findings: `src/ste_quantizer.cpp` implements STE for Q1/Q8/Q4 but carries the static-codebook bug (see `10-code-review-findings.md`) and passes other formats through unquantized. QAT lives in `qat.cpp` (529 lines, LSQ-style). The claim ledger treats "native QUANT training" as PARTIAL until the static-codebook bug and format coverage are fixed.

## Rule 4 — Component counts exact (QUAD=4, TWI=2)

Mixed/compound formats must use exactly 4 components (QUAD) or 2 (TWI), with ratio-sum and BPW-bound checks enforced in `test_mix_components.cpp` (commit bb4eed8). Historical note: TWI_MIX was deleted entirely in the v3 format re-arrangement (2026-08-23); the rule survives as the registry-must-be-empty assertion (`get_all_twi_mixes()` returns `{}`, and `test_mix_components` now asserts that emptiness by design).

## Rule 5 — No stubs, no TODOs, no "coming soon"

Shell code is FAKE and gets a ledger entry with file:line evidence. Verified examples that must never be presented as features: PPO/DPO training shells (no policy backprop), speculative decoding variants (A unusable, B broken acceptance, V2 ghost declaration), AGI flywheel template generation, multimodal constant tokenizers, init_prefetcher header-only declaration. Global grep (2026-08-25) found almost no TODO/stub markers in comments — the danger is behavioral shells, not markers.

## Rule 6 — Every number measured, never hand-copied

Any PSNR/BPW/latency number in docs, benches, or chat claims must originate from a fresh command run whose output is visible. Hand-copied numbers = fabrications. Direct consequence: the entire `bench_format_comparison.csv`-derived win register is FROZEN because the bench tool itself is stale post-v3 (43-format matrix, TWI rows, Debug/Release mixing discovered 2026-08-25). Re-measure after bench migration; until then no W-claim may be re-asserted.

## Rule 7 — Real-weights testing

Synthetic-only or by-construction passing tests are fake. Known offenders: the AGI flywheel harness (tests 2 and 3 always increment pass counters), multimodal tests with 5× `TEST_CHECK(true)`-style assertions. The fuzz codec, BPW probe, and roundtrip tests are the positive examples.

## Rule 8 — Chat in Hinglish, commits in English

Conventional-commit style, English, e.g. `perf(codec): speed round 1 — encode 2-8x faster, quality within noise` (10735a6), `docs+audit: competitor analysis...` (557c32f), `chore: propagate MXQ/K/GRP rename to all consumers...` (a8b4a26). Git identity: Satyam Thakur.

## Rule 9 — llama.cpp is a black-box measurement bar

Compare quality/speed by measuring llama.cpp outputs as a reference; never read or copy its code. The `[ref]` baselines in the CSV (GGUF Q4_K, Q6_K, Q8_0, FP16, INT8, BitNet b1.58, Binary) exist for exactly this. The faithful GGUF Q4_K reference was re-specced at EXACT 4.5 BPW (1152 bits/superblock) for honest same-budget comparison (C-25, 2026-08-23).

## Rule 10 — The BPW150 test deletion is PERMANENT

`tests/test_bpw_150_proof.cpp` was deleted on purpose. Never restore the file; never re-add its CMakeLists entry (the dead reference was itself build-breaker B-1 and was removed from `tests/CMakeLists.txt:179` and the dependency list at :222). This decision is locked by the owner ("Jis file me 150 kahi pe laga hai wo deleted rahenge!").

## Rule 11 — Persona code is banned; personality comes from the system prompt

Owner quote: "abe persona to system prompt se banega naa! Wo code me kya karne gayaa?!" The violation: `class PersonaHotSwap` at `src/agi_extended.cpp:756` (load_persona/swap_to/current). Decision (2026-08-25): do NOT silently delete — document in TRANSCRIPT ("Ya to hataa ya fir TRANSCRIPT me daal de kyuki fir usi se fix start hoga!") so the fix starts from the plan. The DoD checklist requires ZERO persona/personality symbols in the codebase (grep-provable) before any release claim.

## Rule 12 — Format naming law (v3)

- FORMAT_COUNT = 105: base Q1-Q32 (10) + K_L/M/H × 9 widths (27) + GRP→G (9) + K_G (27) + half (9) + half_G (9) + MXQ (7) + MXQ_G (7).
- Dot-BPW display names: `Q8.5`, `Q_G_6.5`, `MXQ_3.5_G`.
- Enum identifiers use underscores: `Q8_G`, `Q_GRP_8_5`, `MXQ_3_5_G`.
- `_GRP` → `_G` rename is COMPLETE across all 105 formats and every file (uppercase token only; lowercase internals like `grp16_fit_scale`, `format_is_grp`, `kGrp16Size` untouched).
- TWI/QUAD names are deleted project-wide. Enum slot 19 = `Q4_K_L` (valid format — the fuzz test's old "slot 19 is a gap" skip was a stale relic and was fixed).
- `src/adapters/` still contains a legacy parallel universe (own RegFormat with QUANT_TWI_MIX_Q0 etc.) — it is NOT built by the root CMake (orphan tree); its fate is an owner decision.

## Rule 13 — Honest audit culture

Full-project audits are run on demand ("5 baar poora project audit"), each pass from a different lens, findings accumulated, and the final report delivered in chat with file:line evidence (`research/audit_full_20260825.md` is the standing example). Any claim found fake is re-verdicted immediately in `research/claim_ledger.md`. The ledger understating reality (said 2 failures, truth was 7) is itself recorded as a finding — audit reports must be corrected when the ground truth moves.

## Rule 14 — Owner-decision gate

These items may not be closed unilaterally by any agent; present options + recommendation, then wait: BPW fit-vs-rename (Rule 1), `src/adapters/` fate (wire in vs archive), bench migration timing, `test_quant_mix` fixture design, Transcender Phase 0 start, any commit of the ~60-file uncommitted working tree.
