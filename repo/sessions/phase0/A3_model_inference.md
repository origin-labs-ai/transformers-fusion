# A3 — Model / Inference Re-context Report (Phase-0)

Scope: transformer, KDA/MLA attention, hybrid stack, model, KV cache, inference_opt,
sampler/generator, HTTP servers, quant_server, and model/inference tests.
Read-only audit. All line numbers verbatim from current `main`.

---

## 1. CODE INVENTORY

### Core model / attention

| File | Verdict | Evidence | Suggested action |
|---|---|---|---|
| `src/transformer.cpp` (894L) | **SOLID** (core path) / **DEAD** (embedded classes) | `Attention::forward` (L251-417) is the real inference path with GQA expansion + FlashAttention; `TransformerBlock` (L484-513) correct. But `KimiDeltaAttention` (L515-666), `AttentionResidual` (L668-710), `MultiHeadLatentAttention` (L793-869), `MultiTokenPredictionHead` (L871-892), `fp8_matmul` (L765-791) are **private classes with zero callers** — dead code inside the same TU. | Keep `Attention`/`TransformerBlock`/`FFN`/`RotaryEmbedding`; delete or move the 5 dead classes to a reference/experimental TU. |
| `src/kda_attention.cpp` (132L) | **REFERENCE** | Header comment (kda_attention.h L15-17) explicitly marks it "REFERENCE implementation: sequential recurrence, plain loops." `forward_naive` (L80-125) is O(S) recurrent; `forward` (header L46-50) discards cache/positions and calls `forward_naive`. | Keep as reference; wire a chunked/SIMD path later. No action now. |
| `src/mla_attention.cpp` (109L) | **REFERENCE** | Header (mla_attention.h L14-16) marks "REFERENCE implementation: sequential full-recompute forward." `forward` (header L47-51) discards cache/positions → `forward_naive`. | Keep as reference. |
| `src/hybrid_block.cpp` (61L) | **SOLID** | `HybridBlock::forward` (L16-29) dispatches KDA/MLA and wires norm/FFN residual correctly. | None. |
| `src/hybrid_moe_model.cpp` (82L) | **SOLID** | `HybridMoeBlock::forward` (L18-39) + `HybridMoeModel::forward` (L62-74) correct wiring. | None. |
| `src/hybrid_scheduler.cpp` (28L) | **SOLID** | `build_hybrid_schedule` (L5-15) 3:1 cycle; `build_k3_schedule` (L17-26) 93 layers. | None. |
| `src/hybrid_expert.cpp` (33L) | **SOLID** | `assign_layer`/`assign_schedule` (L8-31) round-robin. | None. |
| `src/model.cpp` (437L) | **SOLID** (core) / **DEAD** (MTP) | `DenseModel::forward` (L96-126), save/load (L157-375) real. MTP (`mtp_forward` L395-420, `mtp_loss` L422-435) has **zero callers** (see wound W2). | See W2. |

### KV cache

| File | Verdict | Evidence | Suggested action |
|---|---|---|---|
| `src/kv_cache.cpp` (844L) | **SOLID** (KVCache) / **STALE** (paged) | `KVCache::append` (L112-175), `get_range` (L177-223) correct. `PagedKVCacheBase` (L287-585) has **debug `std::cerr` spam** in `load_from_disk()` (L553-561) and `clear()` (L565-584) — `D-lfd-*`, `D-clear-*` markers. `PagedKVCache::init` facade (L833-838) only wraps 4M, ignores `max_seq_len`. | Remove debug cerr spam; see W6 for append. |
| `src/kv_cache_quant4.cpp` (419L) | **SOLID** | `QUANT4KVCache` block-quantized cache, self-contained. | None. |
| `include/quant/kv_cache.h` | **SOLID** | Matches impl. | None. |

### Inference optimizations

| File | Verdict | Evidence | Suggested action |
|---|---|---|---|
| `src/inference_opt.cpp` (1197L) | **MIXED** — many SHELL/DEAD | `PagedAttention` (L16-109), `SpeculativeDecoder` (L114-278), `ContinuousBatching` (L283-369), `CompressedKVCache` (L378-455), `PrefixCache` (L460-494), `TreeDecoder` (L499-586), `flash_decoding` (L591-655), `INT8/FP8Inference` (L660-769), `ModelShard` (L774-813), `DynamicBatcher` (L818-886), `RequestScheduler` (L891-905), `InferenceMemoryPool` (L910-934), `EmbeddingEndpoint` (L968-1052), `Reranker` (L1057-1094), `GrammarDecoder` (L1099-1195). **No production caller** for most; `ContinuousBatching` mask is dropped (W3). | See W3; most are reference/shell demos. |
| `include/quant/inference_opt.h` | **SOLID** | Declares all above. | None. |

### Sampling / generation

| File | Verdict | Evidence | Suggested action |
|---|---|---|---|
| `src/sampler.cpp` (124L) | **SOLID** | `greedy`/`sample_top_k`/`sample_top_p`/`sample` (L12-122) correct. | None. |
| `src/generator.cpp` (355L) | **SOLID** | `Generator` (L14-144), `StreamingGenerator` (L149-353) correct, real streaming. | None. |

### HTTP servers — **DUPLICATION WOUND (W5)**

| File | Verdict | Evidence | Suggested action |
|---|---|---|---|
| `src/http_server.cpp` (978L) | **SOLID** (standalone) | `HTTPServer` (L153-978): full HTTP/1.1, thread pool, rate limiter, SSE, `/v1/*` endpoints. | See W5. |
| `src/production_core.cpp` (server part L224-763) | **STALE / DUPLICATE** | `ModelHTTPServer` (L227-763) is a **second, independent HTTP server** with same socket/select/worker architecture, `/v1/completions` (L614-724), `/v1/chat/completions` (L725-744, **echo stub**), `/health`, `/v1/models`. | See W5. |
| `tools/quant_server.cpp` (617L) | **SOLID** (standalone tool) | `QuantHTTPServer` (L127-573) — third server, `/api/generate`, `/api/tokenize`, `/api/detokenize`, `/health`. | Keep as the CLI tool; distinct from the two library servers. |

### Tests

| File | Verdict | Evidence | Suggested action |
|---|---|---|---|
| `tests/test_kda.cpp` | **SOLID** | P1-P5 delta-rule correctness (L39-156). | None. |
| `tests/test_mla.cpp` | **SOLID** (weak) | T1 shape/finite, T2 compression ratio, T3 "incremental cache correctness" (L30-43) is **vacuous** — only checks `isfinite`, does not actually verify incremental vs full. | Strengthen T3 or remove. |
| `tests/test_model.cpp` | **SOLID** | forward shape + save/load roundtrip. | None. |
| `tests/test_inference_opt.cpp` | **SHELL** | `test_speculative_decoding` (L18-29) and `test_flash_attention` (L31-44) only assert config/shape, never run the algorithm. | Add real assertions or mark as smoke. |
| `tests/test_inference_batch.cpp` | **SOLID** | MoE batch forward finite check. | None. |
| `tests/test_hybrid_*.cpp` (4 files) | **SOLID** | wiring + shape/finite. | None. |
| `tests/paged_kv_4m_test.cpp` | **SOLID** | roundtrip + hierarchical paging + disk flush/reload. | None. |
| `tests/test_transformer*.cpp`, `tests/test_kv*.cpp`, `tests/test_http*.cpp` | **MISSING-TEST** | **Do not exist** (verified via `ls` and `rg --files`). No direct unit test for `Attention`/`TransformerBlock`/`KVCache`/`HTTPServer`. | Add coverage for the core attention path and KV cache. |

---

## 2. KNOWN WOUNDS — VERIFICATION

### W1. MLA cache discard — `include/quant/mla_attention.h:47-51`
**CONFIRMED.** `MLAAttention::forward` discards `positions/mask/cache/layer_idx` and calls `forward_naive(x)`. The header itself (L14-16) admits "Incremental latent caching is exposed but verified separately" — but no incremental cache path exists in `mla_attention.cpp`. The latent-cache compression claim (L4-12) is **not implemented**; `forward_naive` re-materializes full K/V per call. `test_mla.cpp` T3 does not actually test incremental caching (vacuous). **Action:** either implement latent caching or downgrade the header's compression claim.

### W2. MTP zero callers — `src/model.cpp:395-435`
**CONFIRMED.** `mtp_forward` and `mtp_loss` are defined (L395-435) and declared in `model.h` (L45-47), but `rg` shows **no callers** anywhere in `src/`, `include/`, `tests/`, `tools/`. `mtp_heads` are built (L26-28) and counted in params (L93) but never used in `forward`. `trainer_core.cpp:742` has a *separate* `Trainer::mtp_loss` (different class). **Action:** either wire MTP into training/inference or mark the DenseModel MTP path DEAD.

### W3. Continuous batching mask drop — `src/inference_opt.cpp:336-337`
**CONFIRMED.** `build_attention_mask(B, 1, seq_lens)` is computed at L336, then **discarded** — `model_->forward(batch_input, batch_pos)` at L337 takes no mask (signature `Model::forward(input_ids, positions, cache)`). The mask is never used. Also `kv_caches_` (L311-315) are created but **never passed** to `forward` and never appended to. **Action:** either thread the mask/cache through the model or delete the dead mask construction.

### W4. Spec-decode shells — `src/inference_opt.cpp:114-278, 262-298`
**CONFIRMED.** `SpeculativeDecoder` (L114-278) and `SpeculativeDecoderV2` (header L262-298) exist with no production caller. `test_inference_opt.cpp` `test_speculative_decoding` (L18-29) only asserts config equality — never runs the decoder. **Action:** mark as reference/shell; add a real test or remove.

### W5. Two HTTP servers duplication — `src/http_server.cpp` vs `src/production_core.cpp:227-763`
**CONFIRMED.** Two independent, near-identical HTTP servers in the library:
- `HTTPServer` (`http_server.cpp` L153-978) — full-featured, JSON parser, rate limiter, SSE, `/v1/completions`, `/v1/chat/completions`, `/v1/embeddings`, `/v1/models`.
- `ModelHTTPServer` (`production_core.cpp` L227-763) — same socket/select/worker pattern, but `/v1/chat/completions` is an **echo stub** (L743), no embeddings endpoint, no rate limiter.
Plus a third standalone `QuantHTTPServer` in `tools/quant_server.cpp` (L127-573).
**Action:** consolidate to one library server; delete or delegate `ModelHTTPServer`.

### W6. paged_kv append behavior — `src/kv_cache.cpp:350-392`
**CONFIRMED (fixed but fragile).** `PagedKVCacheBase::append` (L350-392) accepts a whole block `{heads, tokens, dim}` and advances `current_pos` by `tokens_written` (L388-391) rather than 1 — the comment (L385-387) documents a prior bug where multi-token appends clamped to one token. The `PagedKVCache` facade (`init` L833-838) ignores `max_seq_len` and only wraps 4M. **Action:** keep the token-count advance; add a test for multi-token append (current `paged_kv_4m_test.cpp` only appends one block at pos 0).

### W7. YARN mscale discard — `src/transformer.cpp:138, 166, 176`
**CONFIRMED.** `yarn_attn_factor` is accepted as unnamed `/*yarn_attn_factor*/` (L138) and **discarded**. `mscale` is set to `1.0f` (L166) and `(void)mscale; (void)inter_len;` (L176) — the YARN attention scaling factor is never applied. `inter_len` is also unused. **Action:** apply `mscale`/`yarn_attn_factor` to the attention scale, or remove the dead params.

### W8. Silent load skip — `src/model.cpp:352-357`
**CONFIRMED.** `DenseModel::load`'s `assign` lambda (L352-357) silently skips any weight whose name is missing **or whose numel mismatches** the built model. A mismatched/missing tensor loads with **no error or warning** — the model runs with random init weights silently. **Action:** log/throw on missing or size-mismatched tensors.

---

## 3. FROZEN-LIST CANDIDATES

| Candidate | (R)/(D)/(W) | File:line | Owner-gate flag |
|---|---|---|---|
| `KimiDeltaAttention`, `AttentionResidual`, `MultiHeadLatentAttention`, `MultiTokenPredictionHead`, `fp8_matmul` (dead private classes in transformer.cpp) | (D) | `src/transformer.cpp:515-892` | A3 |
| `DenseModel::mtp_forward` / `mtp_loss` (zero callers) | (D) or (W) | `src/model.cpp:395-435` | A3 |
| `ModelHTTPServer` (duplicate server, echo chat stub) | (D) | `src/production_core.cpp:227-763` | A3 |
| `SpeculativeDecoder` / `SpeculativeDecoderV2` (no callers) | (R) | `src/inference_opt.cpp:114-278`; `include/quant/inference_opt.h:262-298` | A3 |
| `ContinuousBatching` (mask/cache dropped) | (W) | `src/inference_opt.cpp:283-369` | A3 |
| `PagedKVCache` facade (ignores max_seq_len, 4M-only) | (W) | `src/kv_cache.cpp:833-838` | A3 |
| Debug `std::cerr` spam in paged cache | (W) | `src/kv_cache.cpp:553-561, 565-584` | A3 |
| `test_mla.cpp` T3 (vacuous incremental test) | (W) | `tests/test_mla.cpp:30-43` | A3 |
| `test_inference_opt.cpp` spec/flash smoke tests | (W) | `tests/test_inference_opt.cpp:18-44` | A3 |

---

## 4. TOP RISKS

1. **Silent weight-load failure (W8)** — `model.cpp:352-357` silently skips missing/mismatched tensors, so a corrupt or version-mismatched `.quant` file loads a model with random weights and no error. Highest-severity correctness risk for any served model.

2. **YARN long-context is non-functional (W7)** — `transformer.cpp:138/166/176` discards `yarn_attn_factor`/`mscale`, so YARN scaling does nothing. Any long-context claim built on YARN is false.

3. **Continuous batching is broken by design (W3)** — `inference_opt.cpp:336-337` builds a mask then drops it, and never uses the per-request KV caches. The feature cannot produce correct batched output as written.

4. **Three competing HTTP servers (W5)** — `HTTPServer`, `ModelHTTPServer`, and `QuantHTTPServer` duplicate socket/parse/SSE logic with divergent behavior (the `ModelHTTPServer` chat endpoint is an echo stub). Maintenance and correctness drift risk.

5. **MLA latent-cache claim unimplemented (W1)** — `mla_attention.h:4-16` advertises latent KV compression, but `forward` discards the cache and re-materializes full K/V. The advertised memory savings do not exist in the reference path, and the "incremental cache" test is vacuous.

---

## 5. VERIFICATION NOTES

- Read in full: `transformer.cpp`, `kda_attention.cpp`, `mla_attention.cpp`, `kv_cache.cpp`, `inference_opt.cpp`, `model.cpp`, `sampler.cpp`, `generator.cpp`, `http_server.cpp`, `production_core.cpp` (server part), `tools/quant_server.cpp`, all hybrid `*.cpp`, all in-scope headers, and all in-scope tests.
- Scanned: `kv_cache_quant4.cpp`, `moe_model.cpp`, `world_model.cpp` (out of primary scope but checked for cross-references).
- Confirmed absent: `tests/test_transformer*.cpp`, `tests/test_kv*.cpp`, `tests/test_http*.cpp` (MISSING-TEST).
- No source files were modified. Only this report was written.
