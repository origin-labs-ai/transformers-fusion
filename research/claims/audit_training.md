# AUDIT: Training / RL Stack Claims (C-09 … C-15)

Auditor: blind critic-audit, read-only static analysis (no builds run — build owned by another agent).
Repo state audited: git HEAD `e9257cf` **plus** uncommitted working tree (`src/trainer_rl.cpp`, `tests/CMakeLists.txt` modified; `tests/test_grpo.cpp` new) — diff reviewed explicitly per C-10 mandate.
Method: full reads of `src/trainer_rl.cpp`, `src/trainer_rl_ops.cpp`, `src/reward.cpp`, `src/fine_tuning.cpp`, `src/flash_attention.cpp`, `src/speculative_decoder.cpp`, `src/inference_opt.cpp`, `src/trainer_core.cpp`, `src/continual_engine.cpp`, all matching headers, plus call-site/test greps across `src/ include/ tests/ tools/ bench/ engines/`.

Verdict scale: VERIFIED / FAKE / MISSING / PARTIAL (definitions per claim_ledger).

---

## Summary Table

| ID | Verdict | Evidence (file:line) | Notes |
|---|---|---|---|
| C-09 | PARTIAL | ratio `src/trainer_rl.cpp:206`; clip `:207-211`; adv-norm `:197-203,:210`; GAE `:108-129`; NO policy backward `:243-298`; no optimizer member `include/quant/trainer.h:343-383` | Clip math sahi hai, par trainer policy ko update hi nahi karta — pure logging theater |
| C-10 | PARTIAL | group baseline `src/trainer_rl.cpp:581-589`; HEAD loss broken (see diff); WT fix `:608-628`; MoE params `:534-575`; test `tests/test_grpo.cpp:99-115` | HEAD me GRPO = sign-hacked SFT (provably). Working-tree fix correct; test solid but uncommitted |
| C-11 | PARTIAL | RM fwd `src/reward.cpp:46-96`; RM train+step `:132-292`; wired `src/trainer_rl_ops.cpp:77,:151,:322`; KL dead `src/trainer_rl.cpp:237-243` | Reward model REAL; "KL-penalty wired into training loop" claim FAILS — kl_scale & total_loss are dead code |
| C-12 | VERIFIED | Fisher `src/fine_tuning.cpp:349-373`; `src/continual_engine.cpp:241-265`; penalty grad `src/trainer_core.cpp:388-406` | 3 independent Fisher impls + EWC gradient genuinely injected into Trainer::micro_step; anchor-drift quirk noted |
| C-13 | PARTIAL | `include/quant/fine_tuning.h:141-195`; impl `src/fine_tuning.cpp:542-708`; test `tests/test_fine_tuning.cpp:104-122` | LoRA-equivalent ("QUANT-Rank") real & tested; identifiers LoRA/DoRA exist NOWHERE in source; DoRA = MISSING |
| C-14 | VERIFIED | kernel `src/flash_attention.cpp:38-191`; call-site `src/transformer.cpp:322-331`; numeric test `tests/test_protected.cpp:119-174` | Real tiled online-softmax w/ AVX2 FMA; max_err<1e-3 vs naive reference. Genuine article |
| C-15 | FAKE | zero call sites (grep); sham test `tests/test_inference_opt.cpp:18-30`; dead rewind `src/speculative_decoder.cpp:67-77`; ghost V2 `include/quant/inference_opt.h:262` | "Works end-to-end" PROVEN false: koi bhi generation path use nahi karta; only test never touches the decoder |

---

## C-09 — RLL PPO implemented → **PARTIAL**

What EXISTS (all three requested components present verbatim):
- `ratio = exp(logp_new − logp_old)`: `src/trainer_rl.cpp:206` (`float ratio = std::exp(lp[i] - olp[i]);`)
- Clipped surrogate: `src/trainer_rl.cpp:207-211` — `clipped = min(max(ratio, 1−ε), 1+ε)` then `policy_loss -= min(ratio·adv_norm, clipped·adv_norm)`
- Advantage normalization: mean/var/std computed `src/trainer_rl.cpp:197-203`, applied per-token `:210`
- Supporting machinery: GAE (`compute_gae` `:108-129`), full KL divergence helper (`:131-158`), value head with hand-coded backprop (`:247-297`)

Why NOT VERIFIED — the trainer is a no-op on the policy:
1. `PPOTrainer` has **no optimizer member at all** (`include/quant/trainer.h:343-383`). Nothing ever steps any parameter.
2. `total_loss` (`src/trainer_rl.cpp:243`) and `kl_scale` (`:237-240`) are **computed and never read** — dead locals. The clipped policy loss produces a float for the log callback (`:245`) and dies there.
3. There is **no backward pass through the policy network** anywhere in `train_step`. Only the 4 critic-head tensors get manual gradients (`:247-297`) — and no consumer applies them.
4. Sole caller `RLHFPipeline::ppo_finetune` (`src/trainer_rl_ops.cpp:86-197`) passes raw reward as advantage shape{1} (`:185-188`), calls `ppo.train_step(...)` (`:190`), and never updates a single weight of `model_`. `compute_gae` is never invoked in the loop despite existing.
5. Zero test coverage: grep for `PPOTrainer` across `tests/*.cpp` → no hits.

Bottom line: objective formula likha hai, training loop nahi hai. As a *trainer*, PPO does not exist; as a loss calculator, it does. PARTIAL, generously.

---

## C-10 — GRPO implemented → **PARTIAL** (HEAD provably broken; working tree fixes it)

Committed HEAD version (from `git diff src/trainer_rl.cpp`, removed lines):
- Group-relative advantage WAS computed correctly: mean baseline + std normalization over the candidate group — `src/trainer_rl.cpp:581-589` (unchanged by diff).
- But the loss was: one `cross_entropy_op(logits, labels)` over the whole batch → `backward()` → multiply **all** parameter grads by `adv_scale = mean(adv)`. Since GRPO advantages are normalized to zero mean by construction, `mean(adv) ≈ 0` always → all gradients ≈ 0. The old code even contained the confession-hack: `if (std::abs(adv_scale) < 1e-6f) adv_scale = (adv[0] > 0 ? 1.0f : -1.0f)` — i.e. when the math said "no signal", it invented an arbitrary ±1 global sign. Committed GRPO = random-sign SFT, not GRPO.

Working-tree changes (uncommitted, from parallel session) — reviewed, not touched:
1. Correct per-sample in-graph loss: `loss = (1/G) Σ_i adv_i · CE(logits_i, labels_i)`, each CE inside the autograd graph with its own advantage multiplier — `src/trainer_rl.cpp:608-628`. This is genuine REINFORCE-with-group-baseline (matches original DeepSeek GRPO formulation minus the ratio/clip variant).
2. `collect_grpo_params` now walks `MoEModel` (router + experts + shared expert) — `:534-575`.
3. `beta_` honestly re-documented as update damping, explicitly NOT a KL term — `:632-641`.
4. NaN guards: vocab clamping `:663-665`, `max_seq_len` generation cap `:674-681`.

New test `tests/test_grpo.cpp` (untracked): **correct and substantive**.
- Test 1 (`:74-116`): behavioral — fixed rewards (+,+,-,-), 30 steps, asserts positive-advantage samples' CE drops relative to negative ones. This actually discriminates the broken-vs-fixed implementations.
- Test 2 (`:121-140`): zero-variance rewards → finite loss. WEAKNESS: asserts finiteness only, not that parameters stay frozen (the true GRPO no-op property under zero variance). Missed assertion.
- Test 3 (`:145-216`): MoE smoke through stages A–E, finite-loss checks.
- Registered: YES, via uncommitted hunk in `tests/CMakeLists.txt` (`add_quant_test_full(test_grpo ... LABELS "training;rl")`, after line 88). Cannot confirm it passes — builds forbidden today.

Verdict PARTIAL: mechanism now real in working tree, but the claim was FALSE at HEAD, and the fix+test are uncommitted.

---

## C-11 — Reward modeling integrated → **PARTIAL**

Reward model itself: REAL.
- `RewardModel::forward_mlp` 3-layer MLP forward — `src/reward.cpp:46-82`; `score` `:94-96`; `score_pair` `:98-109`; Bradley–Terry pairwise logistic loss `:111-130` / inline `:187-197`.
- `train_step` does full manual backprop and actually applies updates: grads written to params `:267-286`, then `opt->step(); opt->zero_grad();` `:288-289`. Save/load `:298-345`.

Integration: REAL within RLHFPipeline.
- RM training loop calls `reward_model_->train_step(chosen_feat, rejected_feat, rm_opt_)` — `src/trainer_rl_ops.cpp:77` and `:322` (with val split + accuracy tracking `:278-374`).
- Policy scoring consumes `reward_model_->score(seq_tensor)` in `ppo_finetune` — `:151`.

KL-penalty-wired-into-training-loop claim: FAILS.
- `kl_div` is computed against the ref model (`src/trainer_rl.cpp:189-195`) but flows ONLY into `total_loss` at `:243`, which is never used. The adaptive controller `kl_scale` (`:237-240`) is likewise write-only. No gradient anywhere contains a KL term. GRPO's `beta_` is explicitly documented as NOT-KL damping (`:632-634` working tree). To be fair, PPOTrainer ke paas optimizer hai hi nahi, toh KL apply hota kahan se?

Additional integrity problem found while auditing: `generate_comparisons` fabricates preference pairs — `chosen_ids == rejected_ids == all_ids` (identical sequences, `src/trainer_rl.cpp:485-487`), `reward_rejected = reward_chosen − 0.01f` (`:491`). RM accuracy metrics computed over such data (`compute_reward_accuracy`, trainer_rl_ops.cpp:262-276) are meaningless-by-construction theater.

Also: NOTHING outside `trainer_rl*.cpp` / `reward.cpp` constructs `RLHFPipeline` or `RewardModel` (grep over all of src/include/tests → only self-references). "Integrated" means integrated into its own unused pipeline. Zero tests reference RewardModel/PPOTrainer/RLHFPipeline.

---

## C-12 — EWC implemented → **VERIFIED**

Fisher Information Matrix computation — three independent, real implementations:
1. `SelectiveFineTuner::accumulate_fisher`: autograd backward on anchor batch, diagonal Fisher as EMA of squared grads — `src/fine_tuning.cpp:349-373`. TESTED: `tests/test_fine_tuning.cpp:73-98` (fisher populated + drives block selection), registered `tests/CMakeLists.txt:95`.
2. `ECCState::update_fisher` (grad² EMA) — `src/continual_engine.cpp:241-246`; anchor snapshot `:248-253`; wired live via `ContinualEngine::on_step` `:361-378`, which `Trainer::micro_step` calls every step with flattened grads/weights — `src/trainer_core.cpp:414-431`.
3. `ContinualTrainer::accumulate_fisher` + `ecc_regularizer` (Σ F·Δ²) — `src/adapters/src/continual_trainer.cpp:187-215`.

Penalty term — genuinely wired:
- Gradient injection INSIDE the real Trainer training loop: `gd[i] += 2.0f * ewc_lambda_ * fisher_diagonal[idx] * (w[i] − anchor[idx])` — `src/trainer_core.cpp:388-406` (correct ∂/∂θ[λF(θ−θ*)²]).
- Reported loss adds `λ Σ F·Δ²` via `ECCState::regularize` — `src/continual_engine.cpp:255-265`, consumed at `trainer_core.cpp:451-457`.
- Public API: `enable_continual()`, `set_ewc_lambda()` — `include/quant/trainer.h:290-299`; defaults `:332-335`.

Honest caveats (do not change verdict, but note them):
- Anchor drift: `on_step` re-snapshots the anchor every `forgetting_check_interval` steps (`continual_engine.cpp:371-374`) — θ* chases θ over long runs, diluting protection. Design quirk, not absence.
- No unit test exercises the EWC-penalized step itself (no test calls `enable_continual`/`set_ewc_lambda`; `test_continual_anticollapse.cpp` covers orthogonal projection/replay/entropy-floor instead). Implementation verified statically; behavior untested.

---

## C-13 — LoRA/DoRA adapters → **PARTIAL**

Identifiers: strings "LoRA"/"DoRA" appear NOWHERE in `src/`, `include/`, `tests/` (only marketing prose in `README.md:2793,3592`). The header explicitly brags "no external adapter code of any kind" — `include/quant/fine_tuning.h:15-16`.

What actually exists — `RankAdapterEngine` ("QUANT-Rank"), a faithful LoRA equivalent:
- Declaration: `include/quant/fine_tuning.h:131-195` — rank/alpha config `:131-139`, A {rank,in} pre-scaled α/rank, B {out,rank} `:143-151`.
- Init: A ~ N(0, 1/√in)·(α/rank), **B = 0 ⇒ zero initial delta**, base frozen, adapters registered with Adafactor — `src/fine_tuning.cpp:542-576`. Textbook LoRA init.
- Forward: in-graph low-rank delta `ΔW(x) = ((x·A)·B)` added to FFN down-proj output via `AutogradEngine::matmul_op/add_op` — `ffn_forward` `src/fine_tuning.cpp:578-603`; full block rewiring `:605-623`.
- Train: real autograd backward + Adafactor + warmup LR — `train_step` `:661-685`.
- Merge: exact fold `W += B·A` with a documented memory-layout warning preventing the naive-textbook-layout corruption — `merge_into_base` `:687-708`.
- Quantized `.nrad` adapter store save/load — `:716+`.
- TESTED: `tests/test_fine_tuning.cpp:104-122` (params>0, adapter-per-layer, forward shape, train-step finiteness, merge runs), registered `tests/CMakeLists.txt:95`.

MISSING: DoRA. Zero magnitude/direction weight-decomposition code anywhere in the tree.
Scope limits: adapters attach ONLY to the FFN down-projection of each layer (`fine_tuning.h:157-159`), not attention q/k/v/o.

Verdict logic: "rank-delta low-rank adapters" = VERIFIED-equivalent functionality; the named claim "LoRA/**DoRA** adapters" overstates by half → PARTIAL.

---

## C-14 — Flash Attention present → **VERIFIED**

Declaration vs reality: both symbols declared in `include/quant/flash_attention.h:10-27` are defined and non-trivial in `src/flash_attention.cpp`.

It is a real tiled online-softmax kernel (FlashAttention-2 style), not a stub:
- Online softmax state (running row_max / row_sum) — `:60-61`.
- K/V block tiling sized for L1 (~32KB comment, block 32/16) — `:57-79`.
- Rescale of accumulated output by `exp(old_max − new_max)` incl. AVX2 path — `:124-141`.
- Accumulation `O += e·V_block` with AVX2 FMA — `:143-163`; final 1/row_sum normalization — `:170-187`.
- Per-block causal mask `:95` + skip of fully-masked key blocks with explicit NaN-poisoning rationale — `:114-122`. AVX2 dot product with scalar tail — `:16-36`.
- Dropout as log-space score masking preserving expected weights — `:96-108`.

Wired into the production attention path: `src/transformer.cpp:322-331` dispatches `flash_attention_forward(q_t, k_expanded, v_expanded, causal_mask, ...)` whenever `S_full > 64`; short seqs take the dense path. Mask indexing `m[i*N + (jb+j)]` is consistent with the caller's `{S,S_full}` causal mask.

Tested FOR REAL: `tests/test_flash_attention_reference` builds random Q/K/V + causal mask, computes naive softmax attention reference, asserts `max_err < 1e-3` — `tests/test_protected.cpp:119-174`; registered `tests/CMakeLists.txt:146` (`test_protected`, labels core;training). Contrast: the flash "test" inside `test_inference_opt.cpp:32-44` is decorative (zero tensors, dim asserts).

Minor notes: single-threaded; per-block K/V staging memcpys cost bandwidth; O(n²) compute remains (as in all FA kernels) — the win here is the online-softmax structure and cache blocking.

---

## C-15 — Speculative decoding works → **FAKE** (claim "works/wired"; salvageable core exists)

Proven mismatch between claim and reality:

1. **Zero wiring**: exhaustive grep for `SpeculativeDecoder|SmallBatchVerifier|speculative_decoder.h` across `src/ tests/ tools/ bench/ engines/` returns only the two implementation files themselves. Neither `generator.cpp`, nor `production_gen.cpp`, nor `http_server.cpp`, nor anything else constructs either decoder or calls `generate()`. Koi generation path isko use nahi karta — library-only dead code.

2. **Impl A** (`src/speculative_decoder.cpp`, interface-based) — defective even in isolation:
   - Accept criterion is sampled-token equality: target token sampled from TARGET logits (`draft_logits_vec` — misnamed, holds target logits, `:97-99`), accepted iff equals draft token (`:103`). The draft model's probabilities NEVER enter the accept decision → this is NOT distribution-preserving speculative sampling.
   - KV rollback advertised in header ("Draft-then-verify loop with KV cache checkpoint/rewind", `include/quant/speculative_decoder.h:6`) is fake: `checkpoint_kv` records only `context_len` — `saved_k/saved_v` never populated (`:67-71`) — and `rewind_kv` is **never called** anywhere (`:73-77` dead code). After mid-draft rejection the cache state is simply abandoned.
   - `DraftModel`/`TargetModel` are pure abstract interfaces (`speculative_decoder.h:26-42`) with **zero concrete implementations** in the entire tree.

3. **Impl B** (`SpeculativeDecoder` in `src/inference_opt.cpp:114-278`, Model*-based) — the salvageable core, still flawed:
   - Real math present: accept prob `min(1, p_target/p_draft)` (`:162`, `:242`), rejection-resampling from normalized residual `max(0, p_target − p_draft)` (`:248-264`), adaptive γ via EWMA (`:124-129`).
   - BUG: target verification forwards ONLY the γ draft tokens (`target_input` shape {1,num_draft}, `:208-220`) — never the prefix/prompt context — and `DenseModel::forward` is stateless, so p_target is conditioned on the wrong context every verification.
   - BUG/dead code: `verify_tokens()` (`:131-172`), the variant that queries the actual draft model for p_draft, is never called; `generate()`'s inline loop instead hardcodes `p_draft = 1/vocab` uniform (`:240`), ignoring the draft model it holds. In verify_tokens itself the draft forward is context-free single-token (`:146-150`).

4. **Ghost class**: `SpeculativeDecoderV2` (tree attention, KV caches, RejectionStats) is declared `include/quant/inference_opt.h:262-294` but has **zero definitions** — `grep "SpeculativeDecoderV2::"` empty. Instantiating it would fail to link. Vaporware declaration.

5. **Sham test**: `test_speculative_decoding` (`tests/test_inference_opt.cpp:18-30`) creates two DenseModels and asserts their `vocab_size` fields are equal. It never constructs any decoder, never calls generate/decode_step/verify. Registered at `tests/CMakeLists.txt:102`, lending false confidence to "42 tests pass"-style claims.

Verdict: "Speculative decoding works [and is wired into generation]" is disproven. What exists is (a) an orphaned interface-based skeleton with fake KV rollback, and (b) an uncalled, wrongly-conditioned-but-mathematically-half-right kernel. If someone claims spec-dec speedups from this tree: demand the trace.

---

## Cross-cutting observations (for the ledger)

- The whole RL stack (`PPOTrainer`, `DPOTrainer`, `GRPOTrainer`, `RLVRTrainer`, `RLHFPipeline`, `RewardModel`) is reachable ONLY from within `trainer_rl*.cpp`/`reward.cpp` and `tests/test_grpo.cpp`. No AGI/server/production entry point touches it.
- Pattern recurring in this audit: correct-looking formulas computed into floats that are never differentiated/applied (PPO total_loss, kl_scale; DPO steps optimizer without any backward — `src/trainer_rl.cpp:390-393` receives logits whose graph was already discarded by caller `dpo_finetune`, and is constructed without optimizer anyway at trainer_rl_ops.cpp:379).
- Uncommitted working-tree state (GRPO fix + test) is the highest-quality RL artifact in the repo and should be committed after `ctest -R test_grpo` passes.
