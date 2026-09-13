# Distributed Training Research — ZeRO 1/2/3 and Collective Primitives

**Task:** DIFFUSION.txt Phase K, Task 135  
**Date:** 2026-07-16  
**Scope:** ZeRO stage 1/2/3 sharding, the AllReduce/AllGather/ReduceScatter collectives they reduce to, and how to apply them to Transcender `src/distributed.cpp`.

---

## 1. The Problem ZeRO Solves

Plain data-parallel (DP) training **replicates the full model + optimizer state + gradients on every GPU**. Memory per GPU = `2Ψ + 2Ψ + KΨ` (params FP16 copy + grads + Adam optimizer state, K≈12 for FP32 m+v+master ≈ 12 bytes/param). For a 7B model that's ~112 GB — fits on no single commodity GPU; for 1T it is absurd. The redundant replication of optimizer state + gradients + params across N GPUs is pure waste: only the *data* needs to differ per GPU; the *state* could be sharded.

**ZeRO** (Zero Redundancy Optimizer, Rajbhandari et al., Microsoft, NeurIPS 2020, arXiv:1910.02054) eliminates this redundancy in three stages, each sharding one more piece across the N data-parallel ranks.

---

## 2. ZeRO Stage 1 — Shard Optimizer State (Pos)

**Shards:** optimizer state only (Adam `m`, `v`, and FP32 master copy).  
**Kept replicated:** params + gradients on every GPU.

| Tensor | Replicated? | Memory / GPU (N ranks) |
|--------|-------------|------------------------|
| FP16 params | yes | 2Ψ |
| FP16 grads | yes | 2Ψ |
| FP32 master + Adam m,v (12 bytes) | **sharded** | 12Ψ / N |
| **Total** | | 4Ψ + 12Ψ/N |

**How it works:** each rank holds a 1/N slice of the optimizer state for a non-overlapping subset of parameters. After AllReduce of gradients (standard DP), only the rank owning param-group `i` performs the Adam update on its shard (using its sharded optimizer state), then **AllGathers** the updated FP16 params so all ranks get the fresh copy. The optimizer-step communication adds one AllGather of params; the AllReduce of grads is unchanged.

**Memory win:** for 7.5B, K=12: plain = 122.6 GB; ZeRO-1 = 4Ψ + 12Ψ/N. With N=64, ~4Ψ + ~1.4 GB ≈ 31.4 GB — fits on one 40 GB A100. Roughly 4× effective memory vs plain DP for large N.

**When to use:** N is large, model fits in param+grad memory on one GPU but optimizer state doesn't. Minimal code change from plain DP (shard the optimizer, add a param AllGather).

---

## 3. ZeRO Stage 2 — Shard Gradients + Optimizer State (Pos+g)

**Shards:** optimizer state **and** gradients.  
**Kept replicated:** params only.

| Tensor | Replicated? | Memory / GPU |
|--------|-------------|--------------|
| FP16 params | yes | 2Ψ |
| FP16 grads | **sharded** | 2Ψ / N |
| FP32 optimizer state | **sharded** | 12Ψ / N |
| **Total** | | 2Ψ + 14Ψ/N |

**How it works:** after the backward pass produces gradients, instead of AllReduce (which sums grads across all ranks), use **ReduceScatter**: each rank ends up with the *reduced* (summed) gradient **only for its owned shard** of parameters. So rank `r` has `grad[r]` reduced across all ranks, and nothing else. It then runs Adam on its param+grad+optimizer-state shard and AllGathers the updated params.

**Communication:** backward's AllReduce becomes a ReduceScatter (same byte volume as AllReduce, but no final AllGather-of-grad step); plus a param AllGather after the step. Net comm volume ≈ plain DP (AllReduce = 2Ψ) replaced by ReduceScatter (Ψ) + AllGather (Ψ) = 2Ψ — same total, but memory drops to `2Ψ + 14Ψ/N`.

**Memory win vs Stage 1:** removes the 2Ψ replicated grad. For 7.5B, N=64: ≈ 2Ψ + ~1.6 GB ≈ 16.6 GB.

**When to use:** the most common "sweet spot" — fits the whole model in params on one GPU, but grads+optimizer state don't. This is the default for most 7B–70B training on multi-GPU. **Note:** Stage 2 requires that the *reduction* of grads is fused with the *scatter*, so the backward hooks must call ReduceScatter per parameter group, not a single post-backward AllReduce.

---

## 4. ZeRO Stage 3 — Shard Parameters + Gradients + Optimizer State (Pos+g+p) = FSDP

**Shards:** everything — params, grads, optimizer state.  
**Replicated:** nothing (each rank holds 1/N of the model).

| Tensor | Replicated? | Memory / GPU |
|--------|-------------|--------------|
| FP16 params | **sharded** | 2Ψ / N |
| FP16 grads | **sharded** | 2Ψ / N |
| FP32 optimizer state | **sharded** | 12Ψ / N |
| **Total** | | 16Ψ / N |

**How it works (FSDP — Fully Sharded Data Parallel):**
1. **Before a layer's forward:** AllGather that layer's params from all ranks → materialize the full layer weights → compute forward → **discard** the gathered params (free the buffer).
2. **Before a layer's backward:** AllGather that layer's params again → compute input grads → ReduceScatter the param grads → discard gathered params.
3. After backward, each rank has the reduced grad for its owned shard → Adam step on its shard → params stay sharded until the next layer's AllGather.

This is **layer-by-layer materialization**: only the *currently computing* layer's params are ever resident in full; the rest stay sharded across ranks. Memory = `16Ψ/N` plus a small "one layer" working buffer. For 1T params, N=1024: ≈ 16 TB / 1024 ≈ 16 GB/rank — fits on one 40 GB GPU. This is what makes 1T-param training feasible.

**Communication volume is higher than Stage 2:** every layer does an AllGather before fwd **and** before bwd (≈ 2 × 2Ψ/N × num_layers total = 2Ψ per pass, same order as plain DP but spread across the pass). Overlap comm with compute (start the next layer's AllGather while the current layer computes) to hide it — the core FSDP optimization.

**When to use:** model does NOT fit in params on one GPU (e.g. 30B+ on 40 GB GPUs). This is the path for the Transcender "MoE 1T capable" claim — 1T-param MoE is only trainable via ZeRO-3 / FSDP (or 3D parallelism combining FSDP + tensor parallel + pipeline parallel).

---

## 5. Collective Primitives

All three stages are built from three MPI/NCCL collectives:

### AllReduce
`out[i] = Σ_{rank r} in_r[i]` for all `i`, delivered to **every** rank.
- Implementation: Ring AllReduce (bandwidth-optimal) — each rank sends/receives `Ψ/N` per step in `2(N-1)` steps, total comm per rank = `2Ψ(N-1)/N ≈ 2Ψ`. Bandwidth-optimal (matches the lower bound for a ring).
- Used by: **plain DP** (sum grads), **ZeRO-1** (sum grads).
- Transcender: `distributed.cpp` already has a single-node AllReduce-sum (`condition 113`).

### ReduceScatter
`out_r[i] = Σ_{rank s} in_s[i]` **but** rank `r` only receives the slice for its shard `r`; the full reduced vector is distributed across ranks (reduced *and* scattered). Each rank ends with `Ψ/N` bytes.
- Implementation: ring ReduceScatter, `N-1` steps of `Ψ/N` each, total per rank = `Ψ(N-1)/N ≈ Ψ`.
- Used by: **ZeRO-2** and **ZeRO-3** backward (fuse grad reduction with scatter).
- This is the primitive that replaces the post-backward AllReduce in Stage 2/3.

### AllGather
`out_r[i] = in_s[i]` for **all** `s` — every rank ends up with the concatenation of every rank's shard → the full vector (`Ψ` bytes on every rank).
- Implementation: ring AllGather, `N-1` steps of `Ψ/N`, total per rank = `Ψ(N-1)/N ≈ Ψ`.
- Used by: **ZeRO-1/2/3** to re-materialize full params after the sharded optimizer step; **ZeRO-3** to materialize a layer's params before fwd/bwd.

### Relationship
- `AllReduce = ReduceScatter + AllGather` (reduce-then-broadcast). So plain-DP AllReduce (2Ψ) = ReduceScatter (Ψ) + AllGather (Ψ). ZeRO-2 keeps the ReduceScatter, drops the grad AllGather (grads aren't needed replicated). ZeRO-3 adds back param AllGathers per layer.
- All three are **bandwidth-optimal** in the ring formulation; latency dominates only for tiny messages (overlap with compute to hide it).

### Hierarchy / failure modes
- **Single-node** (shared-memory or NVLink): very high bandwidth; ring is fine.
- **Multi-node** (NCCL over IB/RoCE): ring across nodes; bandwidth = inter-node link. Hierarchical collectives (reduce within node, then across nodes) cut inter-node traffic.
- **No-NCCL fallback**: Transcender condition 113 requires `distributed.cpp` to compile *without* NCCL and provide a single-node sum. The collectives should have a pure-C++ fallback (memcpy + add) so the engine builds everywhere; NCCL is an optional fast path.

---

## 6. Application to Transcender `src/distributed.cpp`

Transcender's `distributed.cpp` (condition 113: "AllReduce single node sum multi NCCL placeholder compiles without NCCL") currently has a single-node AllReduce-sum. To reach the ZeRO/FSDP design:

### Target architecture
```
enum class ZeroStage { NONE, OS, OS_G, OS_G_P };   // = ZeRO-1/2/3
class DistributedEngine {
  int rank_, world_size_;
  ZeroStage stage_;
  // collectives (NCCL if available, else C++ fallback):
  void all_reduce_sum(float* buf, size_t n);       // plain DP / ZeRO-1 grad
  void reduce_scatter(float* grad, size_t full_n); // ZeRO-2/3 backward
  void all_gather(float* param, size_t full_n);     // ZeRO-1/2/3 param broadcast
};
```

### Concrete implementation steps
1. **Abstract the collectives** behind `all_reduce_sum` / `reduce_scatter` / `all_gather` with a compile-time `#if USE_NCCL` switch and a pure-C++ ring fallback (rank `r` sends chunk `r` to `r+1`, accumulates, etc.). This satisfies "compiles without NCCL."
2. **ZeRO-1 in the Trainer** (`trainer.cpp`): shard the `AdamW` optimizer state (`state.m`, `state.v`, master copy) across ranks; after `clip_gradients` + `all_reduce_sum(grads)`, only the owning rank steps its shard; then `all_gather(params)`. Minimal change: `AdamW::step` becomes rank-aware.
3. **ZeRO-2**: replace the grad `all_reduce_sum` with `reduce_scatter(grads)` so each rank only holds its grad shard; the Adam step is local; `all_gather(params)` at the end. Backward hooks call `reduce_scatter` per param group instead of one post-pass AllReduce.
4. **ZeRO-3 / FSDP**: shard params too; before each `DenseModel::layers[l]->forward`, `all_gather(layer[l].params)` into a temporary full-weight buffer, compute, free; same for backward. This needs the layer to expose a `forward_with_flat_params(buf)` path or to gather *in place* into the sharded storage. The layer-by-layer gather/scatter is the FSDP memory trick.
5. **Overlap**: start the next layer's AllGather while the current layer computes (double-buffer the gather buffer). This hides comm — the headline FSDP throughput result.
6. **MoE-aware**: for the 1T-param MoE claim, ZeRO-3 must shard experts too; the AllGather before a layer gathers only the *active* experts for that batch (expert-parallel + data-parallel hybrid = the DeepSeek/Megatron expert-parallel scheme). AllReduce for the non-MoE layers, All-to-All for expert dispatch.

### Memory math for Transcender targets
- 0.1B smoke (condition 112): fits on one GPU; `ZeroStage::NONE` is fine.
- 1T-param MoE (README claim): only feasible at `ZeroStage::OS_G_P` (ZeRO-3) + expert parallel, N large. Memory ≈ 16Ψ/N + one-layer working set. This is the architecture the README "MoE 1T capable" line requires.

### Safety / determinism
- ReduceScatter and AllGather are deterministic in fixed order, but **floating-point reduction order across ranks is not associative** — summed in a different ring order, grads differ at FP roundoff. For determinism (condition 145) either fix the ring order + reduction tree, or accumulate in FP32 (Transcender already keeps master FP32) and round once. Document that distributed determinism requires a fixed world-size + ring order.

---

## Summary Table

| Stage | Shards | Memory / GPU | Extra comm vs DP | Use when |
|-------|--------|-------------|-------------------|----------|
| Plain DP | nothing | 16Ψ | (baseline: 2Ψ AllReduce) | model + optimizer fits on 1 GPU |
| ZeRO-1 (Pos) | optimizer state | 4Ψ + 12Ψ/N | + param AllGather | optimizer state doesn't fit |
| ZeRO-2 (Pos+g) | + gradients | 2Ψ + 14Ψ/N | ReduceScatter + param AllGather (≈ same volume) | grads + optimizer don't fit; **default** |
| ZeRO-3 / FSDP (Pos+g+p) | + params | 16Ψ/N | + per-layer AllGather (×2, overlap) | params don't fit on 1 GPU; **1T-MoE path** |
