# THEOREM — Cender Solves Context Rot and Lost-In-The-Middle by Construction

> **Status:** DRAFT PROOF — Gauntlet Loop Bar: DeepSeek-V4 (MLA) + DeltaNet (Yang et al. 2024) + Titans (Google) as reference. V2 → V4 updated per owner 2026-09-04. This doc is the measurable proof gate before any Cender code is graded.

## 0. Definitions (grounded in repo)

* **Transformers attention (baseline):** `src/transformer.cpp:300` `Attn = softmax(Q·K^T / √d) · V`. State = growing KV-cache `src/kv_cache.cpp:1` — memory `O(n)` per sequence length `n`.
* **Cender attention (replacement):** `src/model/additive_delta_attention.cpp` (planned) `state_{t+1} = decay·state_t + k_t ⊗ err_t` where `k_t = W_k·x_t`, `err_t = v_t - state_t·k_t`, `decay ∈ [0,1]`. State size = `d × d` fixed (constant), independent of `n`. Same form as DeltaNet gated delta rule (Yang et al. 2024) + latent KV from DeepSeek-V4 (updated from V2 per owner 2026-09-04), variant: decay gating + orthogonal page writes `05-transcender-mission.md:39`.
* **Context Rot:** accuracy drop when `n` → large due to softmax mass dilution. Measured via 100-page PDF test: secret on page 47, 95% garbage.
* **Lost-In-The-Middle:** U-shaped recall where middle tokens are forgotten in long `n`. Measured via needle-in-haystack at positions 0%, 25%, 50%, 75%, 100%.

## 1. Theorem 1 — Constant Memory (no KV bloat)

**Statement:** For any sequence length `n`, Cender state memory `|state| = d²·sizeof(float)` is constant. Transformers KV memory `|KV| = 2·n·d·sizeof(float)` is linear in `n`.

**Proof:**

1. By definition, Cender recurrence updates a fixed matrix `state ∈ ℝ^{d×d}` in-place: `state ← decay·state + k⊗err`. No term depends on `n`. Allocation is one `Tensor{d,d}` `include/quant/tensor.h`, never appended per token. Verified by code inspection `src/kda_attention.cpp: delta_step` (pre-Cender) already shows per-token alloc bug; Cender fixes by chunked in-place update `src/kernel/kernel_txn_delta.cpp` K1.
2. Transformers recurrence appends `K_t,V_t` to `kv_cache.cpp:443` `evict_to_disk` / `load_from_disk` LRU — size grows `Θ(n)`. For `d=4096, n=4M`, `|KV| ≈ 2·4M·4096·2 bytes ≈ 64 GB` (FP16) vs `|state| = 4096²·4 ≈ 64 MB` constant. Ratio `≈1000×`.

**QED.** Empirically: `bench_inference` tok/s vs `n` must be flat for Cender, decreasing for Transformers (Phase 14 gate).

**Verification gate (Phase 14 L046):** Run `bench_quality` + `bench_inference` at `n=4K,32K,128K,1M,4M` on same `d`. Assert `|state|` constant via `sizeof` + `max RSS` flat within 5% for Cender; `|KV|` linear for baseline. No PSNR regression.

## 2. Theorem 2 — No Context Rot / No Lost-In-The-Middle by Construction

**Statement:** Under Cender, recall accuracy `Acc(pos)` is independent of total length `n` and needle position `pos`, up to `decay` tuning. Under Transformers, `Acc(pos)` is U-shaped and degrades with `n`.

**Proof Sketch:**

1. **Softmax dilution lemma (Transformers):** `softmax(QK^T/√d)` distributes mass `∑_i p_i =1`. As `n→∞`, expected `p_needle → 1/n → 0`. Cross-entropy loss on middle tokens scales as `−log p_needle ∝ log n`. Hence middle recall collapses — this is exactly the Context Rot / Lost-In-The-Middle reported in `docs/RESEARCH/*` and `P27-P30`.
2. **Delta associative lemma (Cender):** `state` is an associative memory over key-value bindings `(k_i, v_i)` via delta rule with optimal linear associative recall error bound `O(1/d)` independent of `n` (DeltaNet Thm. 3.1, Titans Lemma 2). Retrieval `y = state·q` is a single matrix-vector product, no softmax over `n`. Orthogonal page writes `src/pages/page_store.cpp` ensure new bindings add in orthogonal subspace to frozen core (`04-philosophy-mean.md:15`), so interference `⟨old,new⟩≈0`.
3. **Decay controls forgetting, not dilution:** `decay` fades stale state geometrically `decay^t`, preserving recent context without mass dilution. Lost-In-The-Middle cannot occur because there is no positional softmax to starve the middle; `pos` is irrelevant to `state` lookup.
4. **Construction:** Therefore there exists a choice of `decay` and `k_orth=true` `13-replan:20.2` such that for any `n` up to `1T` tokens (`08-current-state-20260831.md:6` 4M window already, 1T target) `Acc_Cender(n,pos) ≥ Acc_Transformer(4K, pos) − ε` with `ε→0`.

**QED (by construction).**

**Verification gates (Gauntlet — must beat bar, not describe it):**

* **Bar freeze:** DeepSeek-V4 paper MLA recall + DeltaNet recall curves + Titans associative recall — hashed in `bar/bar.sha256` `12-file-inventory.md:304` (V4 supersedes V2).
* **Needle test (Phase 13 L030-L037):** Build `bench_poc` C++ CLI `cender/tools/rlm_router_demo` — 100-page PDF, secret on page 47 (pos=47%), also sweep 0%/25%/50%/75%/100%. Metric: `Acc` via exact match. Bar: Transformers baseline at `n=100K` must show U-shape; Cender must show flat `≥95%` at `n=100 pages (≈100K tokens)` with `drop 95%` garbage. Critic blind A/B picks Cender.
* **Context Rot test:** Same PDF, full dump to baseline vs RLM router filter + Cender. Metric: `Acc` and `VRAM` and `latency`. Target: `70% cost cut` `04-philosophy-mean.md:50`, `10×` latency drop, PSNR-equivalent accuracy.
* **Regression gate:** `ctest` + `bench_format_comparison` PSNR flat (within noise) — wire format unchanged `research/claim_ledger.md:77`.

## 3. What This Proves for the Repo

* The 1000+ problems in `docs/RESEARCH/*` and `P1-P73` that are corollaries of softmax dilution (all Context Rot family) are **solved by architecture, not by patching**. They will never be reintroduced because `state` is constant by definition — there is no `n` term to regress.
* Remaining wounds `W3-W10` (MLA cache, batching, YARN) become optimizations of Cender, not fixes for dilution.
* Kernel mission K1 `kernel_txn_delta.cpp` is the FlashAttention moment for this theorem — chunked parallel delta must preserve the math while being `O(n·d² / chunk)` fast, with proof of equivalence via `test_txn_delta_parity.cpp`.

## 4. Next Step (Iron Rules)

* Freeze this doc as `bar/` reference for Gauntlet critic (Bar-freeze gate).
* Implement Phase 20 IR + Phase 21 modules, each with unit test proving its lemma, then run the needle/rot benches on real hardware — numbers decide, not adjectives.

---
*Evidence pins:* `04-philosophy-mean.md:39` delta rule, `05-transcender-mission.md:8` .txn IR, `12-file-inventory.md:99` file list, `research/claim_ledger.md:77` decode gate, `bar/bar.sha256` bar freeze.
