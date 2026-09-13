# Transcender vs Industrial Quantization Projects — Competitor Analysis

> Date: 2026-08-23. **Trace note (2026-09-13):** the head-to-head
> PSNR table (§2) was spot-verified against the committed `bench_format_comparison.csv`
> (fresh v3 re-measure, 224 data rows, 2026-09-11): Q32 PSNR=100, Q16 gaussian
> 102.45/real 104.51, `[ref] IEEE FP16` 86.47, `[ref] GGUF Q8_0` 58.14,
> `[ref] GGUF Q6_K` 45.87 — all match. Format names are v3 (`QG_8_5`,
> `QG_MX_8_5`, `QG1`; `include/quant/types.h:22-67`); `docs/COMPARISON_CHARTS.md`
> regenerated from the same CSV. BPW column: v3 true-wire values govern
> (`types.h:97-129`; ledger A-01); end-task (ppl/KL/MMLU) parity remains
> unclaimed backlog (§3.2 intact).
>
> Method: web research of every major weight-only quantization
> project (old and new), their published quality metrics, and a head-to-head
> mapping to Transcender formats. **Truthful-BPW rule: `format_bpw()` reports the wire
> budget the canonical encoder spends (`include/quant/types.h:97-129`); names stay
> stable for API.** No format may exceed its budgeted wire BPW.

## 1. How competitors measure quality (metric landscape)

| Project | Year | Primary quality metric | Secondary | BPW points |
|---|---|---|---|---|
| llama.cpp GGUF legacy (Q4_0/Q4_1/Q5_0/Q8_0) | 2023 | wikitext-2 perplexity delta vs F16 | size, tok/s | 4.5, 5.0, 5.5, 8.5 |
| GGUF K-quants (Q2_K..Q6_K, Q4_K_M/S/L) | 2023 | ppl delta (7B: Q4_K_M +0.0535, Q6_K +0.0044, Q8_0 +0.0004) | MMLU/GSM8K | 2.56–6.56 |
| GGUF IQ-quants (imatrix, IQ2_XXS..IQ4_XS) | 2024 | ppl + KL-divergence vs FP16 logits | imatrix-weighted | 2.06–4.25 |
| GPTQ (Frantar et al.) | 2022 | wikitext ppl, Hessian-weighted reconstruction error | act-order | 2,3,4 (g128) |
| AWQ (MIT Han Lab) | 2023 | ppl (L2-7B 4-bit: 5.60 vs GPTQ 5.63), activation-aware salient-channel scaling | MMLU | 3,4 (g128/g32) |
| SmoothQuant / SmoothQuant+ | 2022/23 | W8A8 ppl; W4A16 lossless HumanEval claim | — | 4,8 |
| SpQR | 2023 | ppl (L2-7B ~2.98-bit: 5.63), FP16 outlier isolation | — | ~3.0, 3.85 |
| SqueezeLLM | 2023 | sensitivity-weighted k-means codebooks, ppl | — | 3, 4 (non-uniform) |
| AQLM (Yandex) | 2024 | additive multi-codebook VQ, ppl (L2-7B 2.02-bit: 6.64 vs GPTQ 2.14-bit: 16.77) | ARC/HellaSwag | 2.0–4.1 |
| QuIP# (Cornell) | 2024 | incoherence-processing + E8 lattice codebooks, ppl (L2-70B 2-bit: 3.91 vs OmniQuant 7.81) | — | 2.0–4.0 |
| EXL2 (ExLlamaV2) | 2023 | per-row mixed-bit search minimizing max reconstruction error under avg-BPW target | calibration ppl | 2.0–8.0 any avg |
| EXL3 / QTIP | 2025 | trellis-coded VQ, coherent to 1.6 bpw | — | 1.6–8.0 |
| BitNet b1.58 (Microsoft) | 2024 | QAT ternary {-1,0,1}, matches FP16 ppl at >=3B (10.04 vs 9.91) | zero-shot avg | 1.58 |
| BinaryConnect/BinaryNet | 2016 | {-1,+1} sign, large quality loss at PTQ | — | 1.0 |
| HQQ / bitsandbytes NF4 | 2023 | calibration-free half-quadratic / NF4, degrades <4-bit | — | 1–8 |

Key insight: **nobody publishes weight-reconstruction PSNR**; the ecosystem
standard is end-task (perplexity/KL/MMLU). Transcender's bench measures
**weight-reconstruction PSNR/MSE** (gaussian + trained-real tensors), which is
the *lower bound* proxy: lower weight error correlates with, but does not
equal, end-task quality. Where a direct same-metric comparison is impossible,
we compare against **same-harness honest reference implementations** (IEEE
FP16 round-trip, INT8 uniform, GGUF Q8_0, GGUF Q6_K, GGUF Q4_K, BitNet b1.58
ternary, Binary sign) coded exactly per the competitor's public spec, inside
our bench — apples-to-apples at identical BPW.

## 2. Head-to-head: Transcender formats vs competitors (same BPW, same harness)

Measured on bench_format_comparison.csv (fresh v3 re-measure, 224 rows, 2026-09-11).
G=gaussian, R=real neural weights. Format names are v3 (`QG*` grouped,
`Q_MX_*`/`QG_MX_*` mixes); obsolete pre-v3 half-BPW rows (Q_G_8.5 etc.) are
dropped — those formats do not exist in v3, and their cells are re-derived
below from formats that do.

| Transcender (exact BPW) | Competitor (exact BPW) | Ours PSNR G/R | Theirs G/R | Verdict |
|---|---|---|---|---|
| Q32 (32.0) | FP32 identity | 100/100 | lossless | TIE by definition (reference) |
| Q16 (16.0) | IEEE FP16 (16.0) | 102.45/104.51 | 86.47/86.28 | **WIN +15.98/+18.23** (vmin/vmax fp16-corner trick beats raw FP16 rounding) |
| QG16 (16.5) | FP16 (16.0) | 102.59/104.65 | 86.47/86.28 | WIN (grouped >= plain, strict) |
| QG12 (12.5) | no industrial 12-bit exists; vs Q12 plain | 58.28/54.11 | 57.22/48.28 | WIN vs plain; industrial gap honest |
| Q8 (8.0) | INT8 uniform (8.125) | 54.70/57.09 | 56.67/58.74 | LOSS at 0.125 LESS bpw — honest flag; at equal bits QG8 (8.5-class) wins, next row |
| QG8 (8.5) | GGUF Q8_0 (8.5) | 58.88/60.37 | 58.14/59.75 | **WIN +0.74/+0.62** |
| QG6 (6.5625) | GGUF Q6_K (6.5625) | 47.40/48.60 | 45.87/46.12 | **WIN +1.53/+2.48 at EQUAL bpw** |
| QG4 (4.5) | GGUF Q4_K_H (4.0) | 34.85/36.14 | 31.57/31.99 | WIN (different BPW — weight-PSNR only, not same-metric) |
| QG1 (1.0) | Binary 1-bit sign (1.0) | 16.75/16.95 | 8.10/— | **WIN +8.65** (optimal scale search vs naive sign) |
| QG1 (1.0) | BitNet b1.58 (1.58) | 16.75 | 16.36 | **WIN +0.39 at 0.58 LESS bpw** (PTQ vs their QAT — noted honestly: BitNet's parity comes from training-time ternary, not post-training) |
| QG2 (2.625) | GGUF Q2_K (2.0) | 23.89/24.91 | 19.53/20.77 | WIN at 0.625 more bpw (weight-MSE proxy) |
| Q4_K_H (4.0) | AWQ 4-bit g128 (~4.15 eff) / GPTQ 4-bit | weight-PSNR only | not publishable same-metric | AWQ/GPTQ optimize end-task via calibration; our K variants are calibration-free RTN+optimal-scale — different spec, honest note |
| QG_MX family | EXL2 mixed-bit / GGUF IQ (imatrix) | importance-mix under hard budget | per-row search / imatrix | same design family (saliency-mixed bits); EXL2 uses Hessian calibration, MX uses magnitude ranking — honest difference |

## 3. Honest gaps (no fake wins)

1. **x2 rule vs honest double-BPW**: QG4 cannot beat Q8-class lossless
   (information theory: 6 dB/bit). The industrial GRP 2x rule is enforced at
   SAME-BPW industrial refs instead (QG8>Q8_0, QG6>Q6_K, Q16>FP16).
2. **End-task metrics**: we measure weight PSNR. Perplexity/KL/MMLU parity
   claims vs AWQ/AQLM/QuIP# require a full LLM eval harness — listed as
   Phase 9 backlog, not claimed today.
3. **QAT vs PTQ**: BitNet b1.58 parity is a TRAINING-time result. Our QG1
   is post-training; beating its PTQ sign baseline (+8.65 dB) is the honest
   same-class comparison.
4. **Calibration-based competitors** (GPTQ/AWQ/imatrix/EXL2) use data; our
   base formats are data-free. MXQ importance-mix is the data-free analogue.

## 5. Paradigm comparison: QAT vs PTQ vs STE

| Paradigm | Projects | What it costs | What it buys | Transcender position |
|---|---|---|---|---|
| PTQ data-free (RTN+optimal scale, Lloyd) | our base/K formats; GGUF legacy; HQQ (calibration-free) | nothing | minutes quantize, any model, no data | **Home turf** — all Q/K/Q_G/MXQ formats are data-free PTQ with true-MSE scale search |
| PTQ calibration-based (Hessian/activation-aware) | GPTQ, AWQ, SpQR, SqueezeLLM, EXL2, imatrix IQ-quants | 128–512 samples, GPU-hours, per-model rerun on data change | 0.03–0.3 ppl better at 4-bit; saliency protection | Our MXQ importance-mix is the data-FREE analogue of this family (magnitude ranking vs their activation stats); honest gap at 2–4 bit end-task quality until a calibration pass lands |
| QAT (quantization-aware training) | BitNet b1.58, BitNet b1, direct low-bit training | full pretraining run with fake-quant in graph | 1.58-bit FP16-parity at >=3B — impossible for any PTQ | Not comparable same-class: our Q1_G beats their PTQ sign baseline (+8.65 dB weight-MSE), NOT their QAT result; noted honestly |

STE (straight-through estimator) is the *mechanism* inside QAT that passes
gradients through the non-differentiable round(); it is not a separate
deployment paradigm. Transcender's trainer has STE-style straight-through paths in
its quantization-aware fine-tuning hooks (src/trainer_core.cpp quantized
forward), but no native 1.58-bit pretraining recipe exists — claimed nowhere.

## 6. Arithmetic comparison: FLOPs vs SOPs per weight at inference

Dequant+dot-product arithmetic cost per reconstructed weight (decode path):

| Format class | Multiplies | Adds | Lookups | Notes |
|---|---|---|---|---|
| FP16 baseline (competitor) | 1 FMA | — | 0 | hardware FMA unit |
| GGUF Q8_0 | 1 | 1 | fp16->fp32 convert | scale mul folded into accumulator |
| GGUF Q6_K / Q4_K | 1 | 1 | scale LUT (6b) + code unpack | sub-block scale gather |
| Transcender Q8/K8 (LUT grid) | 1 | 1 | 256-entry float LUT | comp8_table() static, zero transcendentals on hot path |
| Transcender Q8_G compound | 1 | 1 | LUT + per-group fp16 scale load | same op count as GGUF Q8_0-class |
| Transcender MXQ tiers | 0..1 | 1 | tier table | 1-bit tier = sign only -> adder tree, 32-bit tier = raw FMA |
| Transcender Q1_G / sign tiers | **0** | 1 | none | value = ±scale: dot product becomes pure accumulate (**SOPs**) |
| BitNet b1.58 (competitor, QAT) | 0 | 1 | none | ternary adder-only matmul — same SOPs class as Q1_G |

Measured decode throughput proxy lives in bench CSV `decode_us` column
(weights/sec = n / decode_us). Honest note: end-to-end tok/s requires engine
integration (Phase 4/9 backlog); today we compare dequant throughput and
arithmetic complexity, not generated tokens.

## 7. Sources

- llama.cpp discussion #2094 (ppl deltas), llama.cpp K-quant PR #1684
- AWQ arXiv:2306.00978 (MLSys'24 best paper); GPTQ arXiv:2210.17323
- SmoothQuant+ arXiv:2312.03788; SpQR arXiv:2306.03078
- SqueezeLLM arXiv:2306.07629; AQLM arXiv:2401.06118 (tables: L2-7B 2.02b ppl 6.64 vs GPTQ 2.14b 16.77)
- QuIP# (cornell-relaxml.github.io/quip-sharp; L2-70B 2-bit 3.91 vs OmniQuant 7.81)
- ExLlamaV2 README (EXL2 mixed-bit, 2–8 bpw avg, per-row error minimization)
- BitNet b1.58 arXiv:2402.17764 (ternary, 3B+ FP16 parity); BinaryNet arXiv:1602.02830
- HF transformers quantization selection guide (HQQ/bnb/AQLM/VPTQ landscape)
