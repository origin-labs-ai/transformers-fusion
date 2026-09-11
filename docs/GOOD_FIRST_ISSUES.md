# 🛠️ Contributor Board — Find Your First Issue

> Live list of **open work**, tiered by difficulty. Pick one, comment on the
> GitHub issue (or open one referencing the ID), and go.
>
> **Production-hardening sync (2026-09-11):** v1.1.0 / R0001.01 / 105 formats / TranscenderIDX /
> 72 ctest cases (72/72 green 2026-09-11). Path truth: `src/codec/` (not `src/` flat) for codec files.
>
> Already done / in progress (do NOT pick these): bench encode/decode timing
> split, warmup+median stats, repo cleanup, `.gitignore` hardening, CSV
> regression tracker, GLE telemetry module, Q3_G 3-bit affine fix,
> legacy alias purge.

---

## 🟢 Starter (good first issue)

| ID | Task | Files | Why it matters |
|---|---|---|---|
| GF-01 | Finalize the exceptions policy: either set `-fno-exceptions` and remove/replace the remaining `try/catch` blocks, or update the README coding standard to match reality | README.md, CMakeLists.txt, ~66 try blocks | Docs-vs-reality consistency; great way to learn the codebase |
| GF-02 | HTTP server pass-1 hardening: unify timeout handling (single style), add request-line size cap (~8KB) and header cap (~64KB) with `413` responses | tools/quant_server.cpp | First step toward production server |
| GF-03 | Add module smoke tests: world_model, multi_agent, ocr/video/audio currently have ZERO tests | tests/ (new files) | Coverage gap; simple asserts welcome |
| GF-04 | Investigate `test_quant_mix` (was failing in a stale run per ledger C-07; **72/72 green on 2026-09-11** — pick only if a fresh `ctest -R test_quant_mix` fails) | tests/test_quant_mix.cpp, src/codec/block_codec.cpp (mix paths) | Pre-existing failure; needs a detective |
| GF-05 | SUPERSEDED (Phase 24): TWI_MIX was removed by design (ledger C-01; `src/codec/format_registry.cpp:73-76` stub-empty) — QUAD_MIX no longer exists either (v3 has only 4-variant `Q_MX_*`/`QG_MX_*`). Replacement task: add runtime asserts that `format_bpw()` true-wire values match the names' budgets per `include/quant/types.h:97-129` | include/quant/format_registry.h, src/codec/format_planner.cpp | Guards a core design rule |

## 🟡 Intermediate

| ID | Task | Files | Notes |
|---|---|---|---|
| INT-01 | SentencePiece tokenizer support (unigram model load + round-trip vs HF reference) | src/tokenizer* | Unblocks many HF models |
| INT-02 | OpenAI-compatible endpoints: POST `/v1/chat/completions` (SSE streaming), `/v1/completions`, `/v1/embeddings`, GET `/v1/models` | include/quant/http_server.h, tools/quant_server.cpp | Endpoints are declared but not implemented |
| INT-03 | Server contract tests: curl/openai-client based integration suite for all endpoints | tests/test_http_server.cpp (new) | Pairs well with INT-02 |
| INT-04 | macOS CI runner + green build | .github/workflows/ | Third platform badge |
| INT-05 | Perplexity harness: wikitext-2 + tinyshakespeare PPL per format/BPW | eval.h/cpp, tools/evaluate.cpp | The most-requested proof in LLM land |
| INT-06 | Real-model eval harness: HF safetensors → convert → per-tensor MSE/PSNR | tools/eval_real.cpp (new) | Start with TinyLlama-1.1B / Qwen2.5-0.5B |
| INT-07 | Codec fuzz property tests: random tensors → round-trip → error-bound asserts (10k cases, ASan-clean) | tests/test_fuzz_codec.cpp (new) | Quality-lock infrastructure |
| INT-08 | Auto-generate benchmark tables/plots from CSV only (no hand-copied numbers anywhere) | scripts/plot_comparison_charts.py | Generated file should embed source-CSV hash |
| INT-09 | Reasoning-budget hooks: low/high/max sampling-depth parameter plumbed through sampler/generator | sampler.h, generator.h | GLM-5.3-style serving flexibility |
| INT-10 | MXFP4 weight-format study → feasibility doc for a `.quant` bridge | docs/ (new doc) | Research + writing; no kernel work needed |
| INT-11 | Split god-files: `src/codec/block_codec.cpp` (1900+ lines — UNVERIFIED line count, recount before scoping) → codec modules under 800 lines each | src/codec/block_codec.cpp | Refactor with zero behavior change; tests must stay green |

## 🔴 Advanced (high impact)

| ID | Task | Files | Reference architecture |
|---|---|---|---|
| ADV-01 | Multi-arch safetensors loader: LLaMA-family dense first (llama/mistral/qwen-dense) → convert → generate | src/adapters/, converters | Llama-3.2-1B end-to-end demo is the exit test |
| ADV-02 | MoE arch loader (Mixtral-style, DeepSeek-style routing configs) | same | Builds on ADV-01 |
| ADV-03 | compressed (multi-head latent attention) KV compression — target ≥90% cache reduction vs MHA at parity quality | NEW compressed_attention.*, kv_cache.h | DeepSeek V4 Flash papers |
| ADV-04 | FP8 E4M3/E5M2 dtype + conversion kernels + block-FP8 checkpoint loader | types.h, adapters/ | Qwen3.8 block-FP8 expert checkpoints |
| ADV-05 | GatedDeltaNet linear attention layer (gated delta-rule, O(n) sequence cost) | transformer stack | Qwen3.8 hybrid attention |
| ADV-06 | linear-style delta-rule linear attention variant + hybrid attention scheduler (configurable full:linear ratio, e.g. 3:1) | transformer stack | Shares math with ADV-05 |
| ADV-07 | YARN/NTK context scaling: 256K native → 1M expandable mode; needle-in-haystack mini-benchmark | rope utils | Qwen3.8/GLM-5.3 long-context |
| ADV-08 | MTP (multi-token prediction) head + training loss (+ optional speculative-decode tie-in) | trainer, generator | Qwen3.8 training method |
| ADV-09 | CUDA quantized kernels: Q4/Q8 GEMV/GEMM decode path; exit test = ≥5x CPU tok/s on a consumer GPU | gpu_compute_cuda | Parity vs CPU reference mandatory |
| ADV-10 | Metal quantized kernels for M-series | gpu_compute_metal | Same bar as ADV-09 |
| ADV-11 | Async RL rollout worker skeleton (single-node producer/consumer, deterministic replay flag) | trainer_rl_ops | vLLM/slime-inspired pattern |
| ADV-12 | GRPO verification + completion in the native RLL loop | src/trainer/trainer_rl.cpp (was `src/trainer_rl.cpp` — STALE flat path) | Claim-ledger item C-10 |

---

## 📦 How to claim a task

1. **Comment** on the issue: "taking <ID>" (open one if it doesn't exist).
2. **Branch** from `main`: `<ID>-short-description`.
3. **Build + test**: `cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build --parallel && ctest --test-dir build -C Release`.
4. **Evidence in PR**: what changed, measured numbers (if perf), test output.
5. Commit messages: conventional English (`feat(cuda): ...`, `fix(codec): ...`, `docs: ...`).

## ⚠️ House rules (non-negotiable)

- **No stubs, no TODOs, no fake numbers.** Every claim ships with a measurement or a test.
- **Zero external dependencies** beyond the C++20 standard library.
- **BPW budgets are ironclad**: never inflate bits to win a comparison.
- Benchmarks come from the production codec path only — no hardcoded tables.
