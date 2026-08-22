# 🚀 InNova Launch Kit — Community & Contributor Outreach

> Everything needed to announce InNova and attract contributors.
> Rule for ALL posts: **claim only measured facts.** Current verified claims:
> Q16 > FP16 (+16 dB gaussian / +18 dB real), Q6_GRP > GGUF Q6_K (+1.5/+2.5 dB),
> 120K LOC pure C++20, zero dependencies, 44-test suite, Windows+Linux CI.
> Do NOT post perf numbers still under investigation.

---

## 1. X / Twitter

### Bio options (≤160 chars)

**A (recommended):**
```
Solo building InNova — zero-dependency C++20 AI engine. Train→Quantize→Infer in one .quant file. No PyTorch. No Eigen. Just SIMD. 🚀
```

**B:**
```
Writing 120K lines of pure C++20 so AI can run without Python. Quantization that beats FP16 at its own game. ORIGIN Labs. Open everything.
```

**C:**
```
If it needs Python, it's not fast enough. Building the InNova Engine — full AI stack, one binary, zero dependencies.
```

### Launch thread (4 posts)

**Post 1 — Hook:**
```
I got tired of needing 12 Python environments to run a model.

So I'm building InNova: a complete AI engine in pure C++20.
Zero dependencies. Zero Python. One .quant file format.

🧵 Here's what 120,000 lines of hand-written C++ looks like: 👇
```

**Post 2 — Receipts:**
```
No PyTorch. No Eigen. No BLAS. Every kernel hand-written — AVX2/AVX-512, CUDA, Metal, Vulkan backends.

Train from scratch → fine-tune → quantize → inference. All native. All one format.

The numbers? Q16 quantization beats IEEE FP16 by +16 dB PSNR at the SAME bit rate. Measured, not marketed.
```

**Post 3 — Honest hook (attracts serious engineers):**
```
What's NOT done yet (documented honestly):
• Multi-arch model loading (LLaMA family next)
• GPU depth: CUDA/Metal kernels need specialists
• Perplexity eval harness
• SentencePiece tokenizer

Solo project hitting its limits. That's why the repo is open.
If you know C++/CUDA/SIMD — DMs open.
```

**Post 4 — CTA:**
```
⭐ Repo: github.com/origin-labs-ai/InNova
📋 Contributor board with tiered tasks: docs/GOOD_FIRST_ISSUES.md
📖 Whitepaper in /docs

RTs appreciated. Let's build the engine the big labs said couldn't be built. 🔥

#OpenSource #CPP #MachineLearning #LLM #Quantization
```

---

## 2. Reddit

### r/programming or r/cpp (title + body)

**Title:**
```
I'm building a complete AI engine in pure C++20 — training, quantization and inference, zero external dependencies (120K LOC, open source)
```

**Body:**
```
Hi r/programming — long-time lurker, first-time show-er.

Frustrated that running a local LLM usually means dragging in Python,
PyTorch, and a pile of dependencies, I started building InNova: an
engine where the model is born, trained, fine-tuned, quantized, and
served entirely inside ONE binary format (.quant) — no conversion
pipelines, no Python glue.

What exists today (all measured through the production codec):
- Hand-written SIMD kernels: AVX2/AVX-512 dispatch via CPUID
- 37 quantization formats (Q1–Q32 + grouped super-blocks + importance-routed mixes)
- Autograd + trainer + MoE + speculative decoding + BPE tokenizer
- HTTP server, CLI tools, 44-test suite, Windows/Linux CI

One result I'm proud of: our Q16 format beats IEEE FP16 by ~16 dB PSNR
at identical bits-per-weight (same test harness as our GGUF baselines).

Honest flags: multi-arch model loading, deep CUDA/Metal kernels, and a
perplexity harness are still open — all tracked on a public contributor
board with exact files and exit criteria.

Repo: https://github.com/origin-labs-ai/InNova
Contributor board: docs/GOOD_FIRST_ISSUES.md

Roast away — I'd genuinely love feedback from C++ folks, especially on
the codec design and build system.
```

### Show HN (title)

```
Show HN: InNova – Train, quantize and run LLMs in pure C++20, zero dependencies
```

---

## 3. LinkedIn (professional tone)

```
🚀 Open-sourcing my life's project: InNova Engine.

Most AI inference today stands on a mountain of Python dependencies.
InNova is my answer: a complete AI engine — training, quantization,
inference — written entirely in C++20 with ZERO external libraries.

120,000 lines. Every SIMD kernel hand-written. One portable .quant
model format that carries models from birth (training) to deployment
(inference) without a single conversion step.

Recent measured milestone: our Q16 quantized format outperforms IEEE
FP16 by 16+ dB PSNR at the same memory footprint.

I'm opening this up to contributors. Whether you're a CUDA wizard, a
SIMD nerd, a tokenizer tamer, or a docs person — there's a task board
with your name on it:

🔗 github.com/origin-labs-ai/InNova
📋 Contributor board: docs/GOOD_FIRST_ISSUES.md

If this resonates, a share helps more than you know. 🙏

#OpenSource #Cpp20 #AIEngineering #PerformanceEngineering #LLM
```

---

## 4. Step-by-step rollout (do in THIS order)

| Step | Action | Why |
|---|---|---|
| 1 | Update X bio (option A) + pin repo link | Profile ready before traffic arrives |
| 2 | Verify repo front-door: README top shows badges, contributor-board link, quickstart | First impression = conversion |
| 3 | Add GitHub topics: `cpp`, `cpp20`, `llm`, `quantization`, `inference-engine`, `moe`, `simd`, `avx2`, `no-dependencies` | Searchability |
| 4 | Post X thread (posts 1-4), attach benchmark chart image to post 2 | Chart = 10x engagement |
| 5 | Pin the thread | Every profile visitor sees it |
| 6 | Next day morning: Reddit r/cpp post | X momentum warms the audience |
| 7 | Same evening: Show HN (weekday, 7-9 AM US Eastern = peak) | HN punishes off-peak |
| 8 | LinkedIn post next weekday morning | Professional crowd, business hours |
| 9 | Answer EVERY comment for 48h; drop `GOOD_FIRST_ISSUES.md` link wherever someone asks "how do I help?" | Engagement = algorithm fuel |
| 10 | Weekly follow-up post: one measured win per week ("this week: X% faster decode") | Sustained growth loop |

## 5. Comment templates

**When someone asks "how is this different from llama.cpp?":**
```
Great question — llama.cpp is inference-first with GGUF; excellent project.
InNova differs in three ways: (1) native training + fine-tuning inside the
same .quant format (llama.cpp doesn't train), (2) importance-routed mix
formats (QUAD/TWI_MIX) instead of fixed-width only, (3) zero deps even for
build tooling beyond CMake. We also benchmark against GGUF formats directly
— same harness, honest tables in the repo.
```

**When someone says "120K LOC solo is impossible/AI-generated":**
```
Fair skepticism! That's why every number in the README comes from a runnable
bench target (no hardcoded tables), and there's a 44-test suite. Clone it,
run `ctest`, try to break it. The contributor board lists exactly where help
is needed — including auditing the code you're skeptical about. 😄
```
