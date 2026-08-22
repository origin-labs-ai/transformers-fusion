# IndiaAI PMEC Proposal DRAFT — InNova RLL Research Program
> STATUS: WORKING DRAFT v0.1 (needs founder review before submission)
> Target: Large allocation track (PMEC review) — 512–1024 GPU class
> Companion docs: TRANSCRIPT.md (master plan), docs/FUNDING_COMPUTE.md

---

## 1. Project Summary (one paragraph)

ORIGIN Labs requests subsidised GPU allocation under the IndiaAI Compute
Capacity framework to conduct Reinforcement Learning Loop (RLL) research on
the openly released Kimi K3 foundation model (2.8T-parameter MoE), using a
custom zero-dependency C++20 training/inference engine (InNova). Deliverables:
(1) a native-C++ RLL training stack validated at multi-hundred-GPU scale,
(2) measured quality improvements over the K3 base on a fixed public eval
suite, (3) an expert-budgeting technique capping active parameters via router
capacity control, (4) open publication of methods, benchmarks, and tooling.

## 2. Applicant profile (TO FILL)

- Entity: ORIGIN Labs [Pvt Ltd registration — IN PROGRESS / DPIIT ID: ___]
- Founder: Satyam Thakur
- Prior art: InNova Engine (open source, ~120K LOC C++20), 128-page
  whitepaper, published measured benchmarks (Q16 > FP16 by +16 dB PSNR at
  equal BPW; grouped formats beating GGUF Q6_K/Q8_0 baselines), 44-test CI
  suite, Windows/Linux build matrix.
- Links: github.com/origin-labs-ai/InNova

## 3. Technical plan

### 3.1 Why Kimi K3 as base
- Fully open weights (MXFP4/MXFP8, QAT-trained) — full reproducibility
- 2.8T total / 16 experts active per token / 1M-token context
- Hybrid attention (69 KDA linear + 24 gated-MLA layers) — matches our
  engine's long-context roadmap
- License permits research use and derivative publication (verify exact
  terms at application time)

### 3.2 Work packages

| WP | Description | Output |
|---|---|---|
| WP1 | K3 → .quant pipeline validation; lossless round-trip proofs at MXFP4 parity | Converter + fidelity report |
| WP2 | Expert-budget routing: hard cap active params (e.g., 2.0T-equivalent) via router capacity factors; measure quality/compute Pareto | Technique paper + engine flags |
| WP3 | RLL loop at scale: native C++ PPO/GRPO, async rollout workers, reward modelling, KL-penalty integration | Training stack + logs |
| WP4 | Eval suite: identical public benchmarks across (a) K3 base, (b) RLL-tuned, (c) routing-capped variant | docs/REAL_EVAL.md extension |
| WP5 | Safety & alignment: safeguard suite design + red-team harness; capped variant ships WITH safeguards; research artifacts shared only under signed agreements | Safety report |
| WP6 | Open publication: methods, benchmarks, tooling; sovereign-stack contribution | Papers/reports |

### 3.3 Compute request (PMEC format)

| Phase | Duration | GPUs | Type | GPU-hours | Purpose |
|---|---|---|---|---|---|
| P0 Pipeline bring-up | Month 1 | 64 | H100/H200 | ~46K | WP1 converter validation at scale |
| P1 Routing research | Months 2–3 | 256 | H100/H200 | ~370K | WP2 Pareto sweeps |
| P2 RLL training | Months 3–6 | 1024 | H100/H200/B200 | ~2.2M | WP3 training + rollouts |
| P3 Eval + ablations | Month 6–7 | 256 | H100 | ~185K | WP4/WP5 |

Total ask: ~2.8M GPU-hours across 7 months (peak 1024 concurrent).
At subsidised rates this maps to the Sarvam-class allocation tier
(prior art: 4096×H100 approved for sovereign-model development).

Rollout-inference note: RLL is inference-dominated (>70% of cycles are
rollout generation). Our engine's quantized decode path directly reduces
cost-per-rollout versus FP16 reference stacks — this is the core efficiency
claim of the proposal.

## 4. Measurable milestones (quarterly)

- M1 (Month 2): .quant↔K3 round-trip fidelity report published
- M2 (Month 3): routing-cap Pareto curve (quality vs active-param budget)
- M3 (Month 5): RLL checkpoint beating K3 base on fixed eval suite by ≥X%
  [X to be committed after M1 baseline lock]
- M4 (Month 7): full report + open tooling release + safety evaluation

## 5. Budget & subsidies sought

[TO FILL: storage, networking, supporting services line items per empanelled
provider quotes — E2E Networks / Yotta / NxtGen]

## 6. Risk register (honest)

| Risk | Mitigation |
|---|---|
| K3 license terms change | Pin weights hash at proposal time; fallback to Qwen3.8-A95B base |
| Rollout throughput below plan | Quantized decode path (our core IP) + speculative decoding |
| Single-node coordination bugs at 1024 GPUs | GLE telemetry + staged scaling gates (64→256→1024) |
| Safety incidents | Capped variant ships guarded; research artifacts under agreement only |

## 7. Compliance commitments

- Outputs (methods/benchmarks/tooling) published openly per IndiaAI norms
- Model derivatives released only with safeguard suite enabled
- No redistribution of safeguard-disabled artifacts outside signed research
  agreements with vetted institutions
- Usage logs retained per End User Policy

---
(NOTE TO FOUNDER: fill sections 2/5, verify K3 license text, lock eval-suite
list before submission. Do NOT submit with placeholder values.)
