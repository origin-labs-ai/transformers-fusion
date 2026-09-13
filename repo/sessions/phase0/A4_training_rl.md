# A4 — Training/RL — Phase 0 Re-Context — READ-ONLY Audit

**Repo:** `C:\Users\thaku\Downloads\Transcender`
**Slice:** `src/autograd*.cpp, src/trainer_core.cpp, src/trainer_rl*.cpp, src/optimizer.cpp, src/ste_quantizer.cpp, src/qat.cpp, src/fine_tuning.cpp, src/moe_trainer.cpp, src/moe_variants.cpp, src/distributed*.cpp, src/ddp.cpp, src/fsdp.cpp, src/zero*.cpp, src/continual_engine.cpp, include/quant/autograd*.h, include/quant/trainer*.h, include/quant/optimizer.h, include/quant/ste_quantizer.h, tests/test_autograd*.cpp, tests/test_trainer*.cpp, tests/test_optimizer*.cpp`
**Mode:** READ-ONLY

## 1. Verdict Matrix

| File | Verdict | Key Evidence |
|------|---------|--------------|
| `src/autograd_engine.cpp` | SOLID | Real DAG, registry, checkpoint, backward 353:450 |
| `src/autograd_grad.cpp` | SOLID | Real kernels matmul_grad 12:66, rms_norm_grad 177:208 |
| `src/autograd_functions.cpp` | SOLID | 14 AutogradFunction impls, real SDPA 17:221, MatMul 227:282 |
| `include/quant/autograd.h` | SOLID | Engine decl 57:119, helpers 83:97 |
| `src/trainer_core.cpp` | SOLID (wounds) | micro_step + QAT/EWC/replay 244:460, but label_smoothing 637:670 / rdrop 604:635 non-graph |
| `src/trainer_rl.cpp` | SHELL | PPOTrainer 160:298 manual grad no backward; DPOTrainer 342:396 no grad; GRPOTrainer/RLVR solid |
| `src/trainer_rl_ops.cpp` | REFERENCE | RewardModel train_step real, ppo_finetune wired to shell PPO |
| `src/optimizer.cpp` | SOLID | 12 optimizers AdamW 91:119, Adafactor 338:398; MISSING-TEST |
| `src/ste_quantizer.cpp` | STALE | Static-codebook bug 40:58, real STE 19:65 but forward_mixed fresh per-block 145:202 |
| `src/qat.cpp` | SOLID | FakeQuantize 14:37 STE identity, LSQ 42:110, observers 305:499 |
| `src/fine_tuning.cpp` | SOLID | 3 engines Selective 244:522, RankAdapter 528:778 |
| `src/moe_trainer.cpp` | SOLID (dummy aux) | micro_step 259:307 real CE but dummy_logits 284:288 aux 0 |
| `src/moe_variants.cpp` | REFERENCE | ExpertFFN bypass autograd 77:91, dispatch no nodes 767:827 |
| `src/distributed.cpp` | SOLID | all_reduce 37:53, RingAllReduce 261:326 |
| `src/ddp.cpp` | SOLID | bucketed 59:81, compress 174:214 |
| `src/fsdp.cpp` | SOLID | backward 215:273 real |
| `src/zero_optimizer_core.cpp` | SOLID | reduce_scatter 169:209, step 211:246 |
| `src/zero_optimizer_mem.cpp` | SHELL | backward 496:574 stub dQ/dK/dV zeros |
| `src/continual_engine.cpp` | SOLID (codebook alias) | insert 78:104 shared codebook bug 89 |

## 2. Wounds

- **STE static bug `ste_quantizer.cpp:40-48`** `static CodebookQUANT8 cb8; static bool cb8_trained` — first call trains forever, cross-contamination, thread-unsafe. Fix: per-call local.
- **W1 MoE grad chain** `moe_variants.cpp:767-827` `moe_dispatch_batched` raw memcpy no AutogradNode → experts zero grad; `moe_trainer.cpp:282-288` dummy aux.
- **W2 PPO/DPO shells** `trainer_rl.cpp:160-396` PPO never backward policy, DPO zero grads, fabricates logprobs.
- **W18 Grad accumulation** `trainer_core.cpp:189-191` repeats same batch acc_steps times; needs per-micro-batch `next_batch`.
- **Silent passthrough** `trainer_core.cpp:604-670` rdrop/label_smoothing return non-graph Tensor → zero grad.
- **Continual alias** `continual_engine.cpp:88-89` shared codebook for input+target overwrites.

## 3. Suggested Actions (P0)
- Remove static codebooks `ste_quantizer.cpp:40,51`
- Wire `label_smoothing`/`rdrop` through AutogradEngine ops or document no-grad
- Reimplement PPO/DPO via autograd graph
- Split `CompressedReplayEntry` codebooks
- Replace dummy aux with real router logits
- Mark `moe_variants` inference-only or wrap with autograd ops
