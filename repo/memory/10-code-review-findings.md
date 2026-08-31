# 10 — CODE REVIEW FINDINGS (Verdent Session, 2026-08-31)

An independent code-level review performed by the Verdent session on 2026-08-31, reading: `src/block_codec.cpp` (2295 L), `src/autograd_engine.cpp` (487 L), `src/math_avx2.cpp` (681 L), `src/kernel_quant8.cpp` (82 L), `src/kernel_quant4.cpp`, `src/kda_attention.cpp` (132 L), `src/ste_quantizer.cpp` (282 L), `include/quant/kernel.h`, `include/quant/quant_format.h`, plus CMake/arch config. This complements the ZCode audit history (`06-zcode-sessions-history.md`) with fresh first-hand verdicts.

---

## 1. What is SOLID (genuinely strong engineering)

### 1.1 `src/block_codec.cpp` — the heart, and the heart is honest

- **Hand-written FP16 converter** with round-to-nearest-even AND subnormal handling — the kind of detail most "from scratch" projects fake with bit shifts and lose half the dynamic range.
- **Tanh compander with a 256-entry decode LUT**: the constant k=0.02 is not arbitrary — the comment explicitly says it was selected by measured PSNR on the production bench and warns "see commit evidence before changing it." Evidence-driven constants are the project's culture made literal.
- **Scale fitting done right** (`comp8_fit_scale_fp16`, block_codec.cpp:400-406): log-spaced coarse scan + golden-section refinement over **true MSE including clipping**. This avoids the classic LS-fit failure mode (scales that minimize squared error but overload the compander's peak). Verified snippet:

```400:406:src/block_codec.cpp
static float comp8_fit_scale_fp16(const float* w, int cnt) {
    double maxa = 0.0, energy = 0.0;
    for (int i = 0; i < cnt; ++i) {
        maxa = std::max(maxa, (double)std::fabs(w[i]));
```

- Speed round 1 (commit 10735a6) kept this structure but seeded golden search with a least-squares estimate (72→17 evals) — encode 2-8× faster with PSNR within noise. This is how you optimize a codec without lying about quality.

### 1.2 `src/autograd_engine.cpp` — subtle lifetime correctness

The static teardown race is handled properly: the destructor hook is nulled BEFORE member teardown, registry cleanup is noexcept-safe, and `~Tensor` only touches the hook when registered (the layering hook added 2026-08-25 as build-breaker B-2's fix). Many production codebases get this wrong (use-after-free during static destruction); here it is deliberate and commented.

### 1.3 `src/math_avx2.cpp` — textbook-quality SIMD GEMM

Clean 6×16 tiled FMA structure, proper horizontal reduction, scalar tails handled without correctness traps. Nothing fancy (no packing/prefetch yet — that is Transcender Phase 3 item 3), but nothing wrong either.

### 1.4 `src/kda_attention.cpp` — the math is right

The gated delta rule is correctly implemented: `state ← decay·state + k⊗err`, with the decay=0 hard-overwrite boundary explicitly documented and asserted by the unit test. The contract comments (math obligations) are exactly what a replacement architecture needs for auditability.

## 2. What is REFERENCE-GRADE (correct, not fast)

- **KDA forward is naive**: per-token per-head `std::vector` allocation inside `delta_step`; no chunked parallel form; no SIMD. Transformers have FlashAttention; the replacement has a reference loop. This is the single biggest gap between the thesis and its survival (see `05-transcender-mission.md` §6 item 1 — `kernel_kda.cpp` is the "FlashAttention moment").
- MLA (`mla_attention.cpp`, 109 L): correct latent math, `(void)cache` discard (the ZCode wound #11).

## 3. What is KACHCHA (the real smell) — `src/ste_quantizer.cpp` static-codebook bug

This file serves the project's biggest claim ("train-in-format", STE mandatory) and carries a genuine defect:

```40:48:src/ste_quantizer.cpp
        case Format::Q8: {
            static CodebookQUANT8 cb8;
            static bool cb8_trained = false;
            if (!cb8_trained) {
                cb8.train(src, (size_t)n);
                cb8_trained = true;
```

**The bug:** the codebook is a function-local `static`, trained ONCE on whichever tensor arrives first, then silently reused for every subsequent tensor/layer. Consequences:

1. **Cross-layer contamination** — layers after the first are quantized against a codebook fitted to a different weight distribution (this is the same "global k-means" defect the changelog fixed on the inference side; the training side still has it).
2. **Order-dependent training results** — the same model trains differently if the layer iteration order changes. Untestable, unrepeatable, unhonest.
3. **First-tensor bias** — layer 0's distribution dominates the entire model's quantization alphabet.

**Secondary issues in the same file:**
- Only Q1/Q8/Q4 are handled; every other format falls into `default:` and gets **passthrough (memcpy, no quantization)** — so "native QUANT training" is PARTIAL at best (the ledger already suspects this; now with the exact mechanism).
- The file is 282 lines total versus the weight of the claim it backs.

**Fix options (owner-decision-gated, per Rule 14):**
- (a) **Per-tensor codebook** — train inside the call for each tensor (no static). Simple, correct, slower during training setup; matches the inference-side fix philosophy.
- (b) **Per-block fit** — fit per 32/64-weight block like the codec's own scale fitting (consistent with comp8's golden-section philosophy; best quality, most code).
- (c) **Shared-but-seeded** — keep one codebook but seed it deterministically (e.g. k-means++ on a fixed pseudo-sample + EMA updates during training), making it order-independent. Middle ground.

**Recommendation:** (a) now, (b) later inside Transcender Phase 2's rewrite. Whichever is chosen, add a regression test: two tensors with different distributions must produce different codebooks (order-independence assert).

## 4. Kernel-landscape observations (for Phase 3 planning)

- Current kernel LOC ~9,862 across: kernel_quant8 (82), kernel_quant4, kernel_q3, kernel_q6, kernel_q12, kernel_q24, kernel_gemm, kernel_tensor_loops, kernel_production, math_avx2 (681), math_avx2_tiled, math_tensor, simd_math, flash_attention.
- `kernel_quant8/4` operate on the **codebook-index abstraction**, not the canonical block-codec payload — meaning inference over a real `.quant` file's canonical payloads (comp8 grids, bit-packed scales, slots) has NO fast compute layer yet. That is the block-codec compute kernel family (Phase 3 item 5).
- `include/quant/kernel.h` API is clean and dispatch-friendly; adding families does not require breaking it.
- Build conventions that new kernels must respect: one family = one file; explicit CMake source lists (GLOB banned); QUANT_AVX2 defines `__AVX2__` globally so `#ifdef __AVX2__` guards work; every family ships scalar + AVX2 paths and a test.

## 5. Relation to other memory

- The static-codebook bug is queued as open item 8 in `08-current-state-20260831.md` §4 and is NOT in the ZCode wound list (new finding).
- The kernel plan consuming these observations lives in `05-transcender-mission.md` §6.
- The session that produced these findings is logged in `11-verdent-session-log.md`.
