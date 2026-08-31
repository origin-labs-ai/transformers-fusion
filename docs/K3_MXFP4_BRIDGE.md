# K3 MXFP4 → .quant Bridge — Feasibility Study (G-4)

**Status:** Feasibility DOC (engine code change not required for G-4; converter skeleton gated on real checkpoint sample). Assumptions explicitly listed for later invalidation when Moonshot publishes exact block spec.

## 1. K3 Weight Layout (Reconstructed)

- Base: 2.8T MoE, 93 layers (69 KDA + 24 MLA), 896 experts / 16 active.
- MoE expert FFNs: `gate_proj, up_proj, down_proj` stored as **MXFP4**; attention & router remain BF16.
- MXFP4 per spec (OCP MX): 32 values share one 8-bit exponent **E8M0**, each value 4-bit mantissa (including sign). Effective block: **32×4b + 8b = 136b → 4.25 BPW raw** + QAT scale.
- MoonViT-V2 vision encoder: separate safetensors, not covered here.
- Checkpoint on HF: `model.safetensors.index.json` sharded, BF16 tensors + `*.mxfp4.safetensors` shards for experts. No public dequant reference yet → assumptions below.

## 2. Assumptions (Explicit)

| # | Assumption | If Wrong, Fix |
|---|---|---|
| A1 | MXFP4 block = 32, E8M0 shared exp, 4b per value S1E2M1 (OCP). | Swap E2M2 variant, re-run probe on 1 block vs Python `mx` lib |
| A2 | Expert weight tensors named `layers.*.moe.experts.*.*.weight` | Adapt glob in `src/adapters/k3_converter.cpp` |
| A3 | QAT scale folded into MXFP4 mantissa, no extra per-block BF16 | If extra scale present, dequant adds `* block_scale` factor |

## 3. Bridge Pipeline (.quant)

```
HF shard (MXFP4 block) --E8M0 dequant--> FP32 block (32 values)
        --Lloyd-Max 4-bit codebook (per 32-block, EM 3 iters)--> Q4 indices
        --quant_format writer--> .quant block (Q4, 4.5 BPW with per-group scale)
```

- Dequant: `fp32 = ldexp((mantissa / 8.0) , shared_exp - 2)` for E2M1 variant; probe both S1E2M1 vs S1E1M2 on first block vs Python reference.
- Re-quant to Q4: reuse existing `codebook` + `block_codec` Lloyd-Max paths; Q4_G not needed for experts (already grouped by MoE sharding).
- Converter skeleton: `tools/k3_convert.cpp` (not yet landed) will stream shards, dequant on the fly, and emit `.quant` with preserved `num_experts/top_k` in model header.

## 4. Validation Plan (When Weights Drop)

1. Dequant 1 MXFP4 block with both mantissa variants, compare to `mlx`/`mx` Python dequant if Moonshot publishes snippet — pick exact variant.
2. Round-trip: MXFP4 → FP32 → Q4 → FP32. Assert per-expert MSE < 2× Q4-native baseline at same BPW.
3. Tiny MoE smoke: load converted `.quant` into `MoEModel` (quant_model path), forward 2×32 tokens, assert finite logits.

## 5. Decision

Feasible with **zero engine core changes**; only converter tool + test data needed. G-4 exits as DOC + blocked code stub (converter gated on `HAVE_K3_SAMPLE`).

## 6. References

- OCP MX Spec v1.0 (E8M0 shared exponent)
- InNova docs/Hybrid 3:1 scheduler (G-3) — layer map needed to place converted expert weights
- `src/block_codec.cpp` Q4 paths (existing Lloyd-Max)
