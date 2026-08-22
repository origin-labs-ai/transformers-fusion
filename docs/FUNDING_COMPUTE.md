# 💰 Compute & Funding Strategy — InNova Engine

> How to get serious GPU time without money. Ordered by speed-to-access.
> Reality check first: **GB300 NVL72 is frontier-training scale** (multi-crore
> racks). Nothing in the current master plan needs it yet — the plan's own
> funding order (README Ch. 46/55) is: demo locally → win trust layer → THEN
> raise specifically for Blackwell-tier hardware expansion.

## Tier 0 — FREE, today (no application)

| Resource | What you get | Use for |
|---|---|---|
| Kaggle Notebooks | 2x T4 16GB, ~30 hrs/week FREE | Perplexity eval runs, small-model inference benchmarks |
| Google Colab free | T4 sessions | Quick sanity runs |
| Hugging Face ZeroGPU | Free shared A100 for Spaces | Host live demos of quant-infer |
| Local CPU (AVX-512) | Your machine | Wave 4 codec/SIMD work needs NO GPU at all |

## Tier 1 — Government-subsidised (India-specific, apply this week)

**IndiaAI Compute Portal** — https://compute.indiaai.gov.in
- 38,000+ GPUs empanelled (H100/H200/A100/L40S; 25,000 B200-class being added)
- Rate ≈ **₹65 per GPU-hour** (up to 40% subsidy) vs dollar pricing abroad
- Eligibility: **DPIIT-recognised startup**, MSME, or researcher (h-index ≥5)
- Requests under **5,000 GPU-hours get monthly auto-approval**
- Providers: E2E Networks, Yotta, NxtGen (Sarvam AI trained sovereign models here)

**Steps:** (1) Get DPIIT recognition (free, online, days) → (2) Register on
portal with project proposal → (3) Request <5,000 GPU-hrs for auto-approve.

**Also (India):** Startup India Seed Fund (up to ₹50L), NIDHI PRAYAS
(prototype grant), T-Hub/iHub accelerator compute perks.

## Tier 2 — Free research & startup programs (global)

| Program | What | Notes |
|---|---|---|
| Google TPU Research Cloud (TRC) | FREE TPU pods for approved research | Perfect for RLL/RL-loop experiments; write proposal around open-source engine |
| NVIDIA Inception | Free membership → cloud-compute credits + HW discounts | Open-source AI engine = strong application |
| Microsoft for Startups Founders Hub | Up to ~$150K Azure credits (incl. GPU) | Needs company entity |
| AWS Activate | Up to $100K credits | Same |
| Google for Startups Cloud Program | Up to $200K GCP credits over 2 yrs | Same |

## Tier 3 — Dirt-cheap burst GPU (hourly, no approvals)

| Provider | Hardware | Price ballpark |
|---|---|---|
| Vast.ai | RTX 3090/4090 spot | ~$0.20–0.45/hr |
| RunPod | RTX 4090 / A5000 | ~$0.35–0.70/hr |
| Lambda / Nebius | H100 on-demand | ~$2–3/hr |
| IndiaAI providers | H100 SXM (subsidised) | ~₹65–450/hr |

Use-case mapping: CUDA kernel dev (ADV-09) = one 4090 spot instance is plenty;
Phase-3 PPL sweeps = Kaggle; RLL rollouts = TRC TPUs; anything bigger =
IndiaAI allocation.

## Mapping to master-plan phases

| Phase | Compute actually needed |
|---|---|
| 4 Speed War (SIMD codec) | CPU only — free |
| 3 Quality Gauntlet (PPL, real-model evals) | Kaggle T4s / one A100 burst |
| 5 Feature completion (FP8/MLA/MTP tests) | Single H100 hourly bursts |
| 7 CUDA kernels | 4090-class spot instances |
| RLL self-improvement loops | TRC TPUs (applied-for) |

## The GB300 answer

Per README Chapter 46/55 (funding order already written): complete the local
demo + proof suite first, open verification audits, then raise strategically
for the three Blackwell tiers. Investors fund measured engines, not wishes.
Every subsidised hour spent now on REAL_EVAL.md numbers IS the fundraising
deck.
