# 05 — TRANSCENDER MISSION (Replacing the Transformer)

Status: **Name locked 2026-08-31. Phases 0-4 defined. Phase 0 NOT started** (owner has not yet said "shuru"). This file is the mission's single source of truth.

---

## 1. The name (LOCKED 2026-08-31)

**Transcender** — Transformer → Transcender, a one-letter-class shift: from the thing that transforms sequences to the thing that *transcends* them.

- Tagline (ready-made): **"Attention was all you needed. Transcendence is all you need."**
- Brand pairing: **"InNova engine, running the Transcender architecture."** Engine and architecture have different-sounding but complementary names — a two-product brand from one repo.
- Owner's constraint that shaped the choice: an English name that visually/phonetically matches "Transformers" ("jaise maine is project ka naam InNova socha hai waisa English naam soch jaisa Transformers se matching bhi lage thora sa!"). The same-meaning clause ("Ye mean same ho ye bhi kar sakte hai!") was also satisfied.

### Candidate history (why Transcender won)

| Candidate | Meaning angle | Verdict |
|---|---|---|
| **Transcender** | transform → transcend; morphology directly echoes Transformer | **WINNER** — strongest pitch + matching |
| Everformer | "Ever" = never forgets; "-former" = literal lineage | Strong #2 — most literal matching |
| Transmuter | alchemy: slop → genius, restraint → power | Strong #3 |
| Accretor | physics: grows only by adding, nothing leaves | Deepest meaning, weakest matching |
| Fluxion | Newton's original word for "delta" | Perfect math heritage, no Transformer echo |
| AKSHAY / SANCHAY / DHARA | Sanskrit meanings (indestructible / accumulation / stream) | Rejected — owner wanted English with Transformer resonance |

### Blacklist — names ALREADY TAKEN by famous work (never use)

`Reformer` (Google), `Informer`, `Performer`, `Synthesizer`, `Mamba`, `RetNet`, `Titans` (Google's additive-memory paper — conceptually close to the additive-pages doctrine; MUST be cited), `Deltaformer` (already exists on arXiv).

## 2. Citation & provenance policy (the anti-"chor" doctrine)

The owner's fear: *"Nahi to koi aira gaira aake bolega saala chor hai!"* (someone will call us thieves because K3-branded files exist). The resolved position:

1. **Ideas are not copyrightable; code is.** The gated delta rule originates from the DeltaNet papers (Yang et al. 2024); MLA originates from DeepSeek-V2. Kimi K3 itself took these ideas and built its hybrid. Implementing the same math from the papers, with citations, is science — not theft.
2. **Own naming + honest citation is the shield.** Every renamed file carries a header like:

```cpp
// AdditiveDeltaAttention — InNova's constant-memory associative attention.
// Built from the gated delta rule (DeltaNet, Yang et al. 2024); InNova
// variant: decay gating + orthogonal page writes (see docs/ARCHITECTURE.md).
```

3. **The two traps, both forbidden:** (a) renaming K3-branded files while hiding the lineage = worse than theft (a "jhootha"/liar attack); (b) omitting citations = the community assumes concealment.
4. **Scratch implementation from the math + tests + citations** is the clean path for Phase 2.

## 3. Phase 0 — Provenance cleanup (first executable work)

**Status: NOT STARTED.** Contents (mechanical, completable in one session):

- `src/kda_attention.cpp` → `additive_delta_attention.cpp` (final name owner-confirmable)
- `src/mla_attention.cpp` → `latent_kv_attention.cpp`
- `k3_tokenizer` → InNova-owned tokenizer name
- `hybrid_moe_model.cpp` / hybrid_* family → Transcender-branded model names
- All consumers, CMake source lists, tests, and docs updated atomically
- Citation headers on every renamed file (template above)
- `research/claim_ledger.md` entry recording the rename with evidence

## 4. Phase 1 — Design document

`docs/ARCHITECTURE.md` (or TRANSCRIPT-linked doc): the math of the gated delta rule + latent attention + hybrid routing; why each choice; how it differs from softmax attention (constant-memory state vs KV cache; additive pages vs fine-tune overwrite); the Titan/DeltaNet/Mamba/RetNet prior-art map with the InNova deltas.

## 5. Phase 2 — Scratch implementation (parity-gated)

- New core written from math with its own tests; **old renamed modules stay alive until the new implementation proves parity** — a big-bang rewrite would break the 42-test suite and the audit culture forbids that.
- Parity criterion: numerical equivalence on fixture models + all existing tests green.
- Design-time avoidance of the 12 shell-findings (see `06-zcode-sessions-history.md` master wound list): no shells, no discarded params (YARN mscale lesson), no unwired heads (MTP lesson), no fake batching.

## 6. Phase 3 — Kernels (the 25K+ LOC mission, mission-aligned)

**Baseline: ~9,862 kernel LOC** (kernel_quant8/4/q3/q6/q12/q24 + math_avx2/math_avx2_tiled/math_tensor + simd_math + flash_attention). Target: 25K+ LOC of REAL kernels — never padding (the owner rejected generic new-kernel sprawl on 2026-08-31: "Abe saale, tu naye kernels kyu bana raha hai?!").

Locked order (mission-critical first):

1. **`kernel_kda.cpp` — chunked parallel delta-rule kernel.** The replacement's FlashAttention moment. Current KDA is naive (per-token per-head std::vector allocation in `delta_step`). Requirements: chunked state updates, no per-step allocations, AVX2 + scalar paths, decay/k-orthogonality handled in-register.
2. **`kernel_mla.cpp`** — latent attention kernels (the cache-discard wound must die here).
3. **Tiled + threaded GEMM** (+ gemm_bt) — P35; the RLL loop's biggest bottleneck; multi-row micro-kernels, packing, std::thread.
4. **Norm/elementwise fused kernels** — layernorm/RMSNorm/softmax/activations in SIMD (every layer of every rollout touches these).
5. **Block-codec compute kernels** (dot/gemv/gemm over canonical block payloads) + in-place optimization of the existing small format kernels (the owner's original demand, preserved).

Conventions: one kernel family = one file; CMake explicit listing; tests green after every batch; measured numbers only.

## 7. Phase 4 — Orphan wiring + bloat cleanup

The full wound list is ready-made in `06-zcode-sessions-history.md` (23 items with file:line). Policy: wire or explicitly retire — nothing stays hidden (the audit culture forbids a "chhupa rahej"). Highlights: MTP zero callers, MLA cache discard, spec-decode shells, GPU_VULKAN factory gap, `src/adapters/` 33-file orphan tree (owner decision pending), quant_engines_core.cpp dead file, AGI flywheel templates, encode_image `{30000}`, persona violation (agi_extended.cpp:756), MoE gradient chain, PPO/DPO backprop, continuous-batching mask drop, YARN mscale discard, bench migration.

## 8. The `.txn` graph IR (accepted direction)

Where `.quant` stores weights, `.txn` will describe the architecture: delta-rule ops, page writes, routing graph, decay gates — a declarative Transcender definition that the engine compiles, validates (AST-gate style checks), and runs. Rationale + precedents in `04-philosophy-mean.md` §6. This satisfies the owner's "even if I must build a language, I will not bend" instinct without the multi-year compiler trap.

## 9. Kernel-work tracker (history)

- 2026-08-31: Verdent session created a 14-item kernel todo (block-codec kernels → gemm_tiled → attention → norm → elementwise → dispatch → conv → kv_quant → AVX512 expansion → CMake wiring → tests → build verify → 25K verify). Survey stage only — **zero lines written**. Owner paused it ("Abe saale, tu naye kernels kyu bana raha hai?!"), mission was clarified, and the plan was folded into Transcender Phase 3 with the revised order above.
