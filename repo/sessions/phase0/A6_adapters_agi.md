# A6 — Adapters / Multimodal / AGI — Phase-0 Full Re-Context

**Agent:** A6 · **Scope:** `src/adapters/*`, `src/multimodal*.cpp`, `src/agi*.cpp`, `src/meta_cognition*.cpp`, `src/self_eval*.cpp`, `src/world_model*.cpp`, `src/multi_agent*.cpp`, `src/code_gen.cpp`, `src/hpo_nas.cpp`, `src/continual_engine.cpp`, `engines/trainer/multimodal/*`, `engines/trainer/moe/*`, `include/quant/*.h`, related tests.
**Mode:** READ-ONLY (no source modified). Only this report written.
**Date:** Phase-0 re-context.

---

## 1. CODE INVENTORY

Verdict legend: **SOLID** = real, integrated, tested · **REFERENCE** = real, integrated, no direct test · **SHELL** = stub/placeholder · **DEAD** = unreferenced/unreachable · **ORPHAN** = standalone tree not wired into main build · **STALE** = superseded/duplicated · **BRANDED** = proprietary/third-party model names embedded · **MISSING-TEST** = real but no test.

### 1.1 AGI core (`src/agi*.cpp`)

| File | Verdict | Evidence | Suggested action |
|---|---|---|---|
| `src/agi.cpp` (1113L) | **SOLID** | SelfMonitor/SelfReflector/MetaCognition/RecursiveSelfImprover all real; tested by `tests/test_agi.cpp`. | Keep. |
| `src/agi_flywheel.cpp` (1980L) | **REFERENCE** (with SHELL sub-parts) | Real run loop + 24 template codegens + 32 task templates, but **fake test harness** (see 1.1a). Tested only trivially by `test_agi_flywheel.cpp`. | Fix fake harness; add real correctness tests. |
| `src/agi_extended.cpp` (885L) | **REFERENCE** (with BRANDED/DEAD sub-parts) | WorldModel/Curiosity/Evaluation real; **PersonaHotSwap class defined mid-file at :756** (not in header, dead code). | Remove/relocate PersonaHotSwap; see 1.1b. |
| `src/agi_guardrails.cpp` (319L) | **SOLID** | SafetyGuardrails/HITL/Alignment/CodeGenSelfImprover/SelfVerifier/CapabilityAmplifier all real; tested. | Keep. |
| `src/agi_enhanced.cpp` | **MISSING** | File does not exist anywhere in repo. | Scope description references a non-existent file. |
| `src/agi_memory.cpp` | **MISSING** | File does not exist. `src/memory.cpp` is a memory-pool util, unrelated. | Scope description references a non-existent file. |

#### 1.1a — AGI flywheel: 32 templates + fake harness (VERIFIED)
- **32 task templates** confirmed: `src/agi_flywheel.cpp:808` — `static const std::array<const char*, 32> task_templates`.
- **24 template code generators** confirmed: `case ProblemCategory::` appears **24** times (`src/agi_flywheel.cpp:160,173,192,224,253,288,313,335,351,372,392,423,457,476,496,516,545,573,599,615,634,658,681,700`). Enum has 24 categories (`:75-85`). So "32 templates" = 32 *tasks*, 24 *code templates* — the two numbers are distinct.
- **FAKE HARNESS (critical):** `generate_test_program` (`src/agi_flywheel.cpp:863-911`) wraps the solution but the harness body **never calls any solution function**. It hardcodes `CHECK(true, ...)` (`:889`), `g_passed++` (`:893,897,903`), and a loop that just prints `INFO` (`:900-902`). Every solution that compiles "passes" 4/4 regardless of correctness. `sandbox_compile_and_test` (`:1322`) then reports `result.passed = (exit_code==0) || (passed>0)` (`:1387`) and `score = passed/5` (`:1391`) — so **any compiling program scores 0.8**. `measure_improvement` (`:1418`) uses keyword-count heuristics (`:1444-1454`), not real capability. The flywheel's "self-improvement" is therefore **cosmetic** — it cannot actually verify or improve code.

#### 1.1b — PersonaHotSwap (VERIFIED)
- `class PersonaHotSwap` defined **inside the .cpp** at `src/agi_extended.cpp:756-805`, after `evaluate_all()` (`:747`). It is **not declared in `include/quant/agi.h`**, has no callers, and is dead code. It also uses `#include <atomic>/<mutex>/<map>` mid-file (`:751-754`). This is the "persona violation" — a stray class that does not belong in the AGI extended translation unit.

#### 1.1c — Seed-42 residuals (VERIFIED)
Hardcoded `42` seeds scattered across AGI/multimodal-adjacent code (deterministic but undocumented):
- `src/agi_extended.cpp:104` (`RNG rng(42)`), `:252`, `:267`, `:295` (`std::mt19937 rng(42)`)
- `src/agi.cpp:924` (`std::mt19937 rng(42)`)
- `src/self_eval.cpp:504`, `:586` (`std::mt19937 rng(42)`)
- `src/world_model.cpp:62` (`rng_(42)`)
- `src/multi_agent.cpp:736` (`std::mt19937 rng((unsigned)(42 + step))`)
- `src/adapters/tests/test_adapter_bridges.cpp:83` (`std::mt19937 rng(42)`)
These are not bugs per se but are magic constants that should be centralized/configurable.

### 1.2 Multimodal (`src/multimodal*.cpp`)

| File | Verdict | Evidence | Suggested action |
|---|---|---|---|
| `src/multimodal.cpp` (1079L) | **REFERENCE** (with SHELL sub-parts) | H1-H15 real (cross-attn, joint model, ImageNet, ASR, OCR, video, caption, VQA, T2I DDIM, audio synth, mel, tokenizer, modality encoder, contrastive). **encode_image/encode_audio are stubs** (see 1.2a). Tested by `test_multimodal.cpp`. | Fix tokenizer stubs. |
| `src/multimodal_cross_attn.cpp` (950L) | **SOLID** | PatchEmbedding, VisionTransformer, AudioTransformer, CrossAttentionBlock, ModalityFusionEncoder, ContrastiveAlignmentHead all real. Empty `Impl {}` structs (`:277,365,465`) are pimpl placeholders, not shells. Tested by `test_multimodal_encoders.cpp`. | Keep. |
| `src/multimodal_fusion.cpp` (358L) | **SOLID** | ModalityProjection, CrossAttentionFusion, MultimodalFusion (fuse_vision_audio/text, fuse_all, multistream, weighted) all real. Tested. | Keep. |

#### 1.2a — encode_image {30000}/{31000} constants (VERIFIED)
- `src/multimodal.cpp:674-677` — `encode_image` returns `{30000}` regardless of image content.
- `src/multimodal.cpp:679-682` — `encode_audio` returns `{31000}` regardless of audio content.
- `include/quant/multimodal.h:140-141` — `image_token_id()==30000`, `audio_token_id()==31000`.
- `tests/test_multimodal.cpp:65-66` only asserts the constant IDs, not real encoding. These are **placeholder sentinel tokens**, not real multimodal encoding.

### 1.3 Meta-cognition / self-eval / world-model / multi-agent / continual

| File | Verdict | Evidence | Suggested action |
|---|---|---|---|
| `src/meta_cognition.cpp` (838L) | **REFERENCE** | MetaCognitionEngine: entropy, reasoning chain, calibration, goal decomposition, strategy selection, feedback — all real. **No test file.** | Add test. |
| `src/self_eval.cpp` (836L) | **REFERENCE** | SelfEvalSuite: accuracy (MMLU/ARC/HellaSwag/GSM8K), calibration, consistency, robustness, speed, memory, pass@k — all real. **No test file.** | Add test. |
| `src/world_model.cpp` (638L) | **REFERENCE** | WorldState, WorldModel (predict, ensemble, trajectory, transition training) — real. **No test file.** | Add test. |
| `src/multi_agent.cpp` (793L) | **REFERENCE** | MultiAgentSystem: roles, routing, task decomposition, consensus voting, episodes — real. **No test file.** | Add test. |
| `src/continual_engine.cpp` (543L) | **REFERENCE** | CompressedReplayBuffer, ECCState, ForgettingBenchmark, ContinualEngine, ExperienceReplayBuffer, DistributionWatchdog — real. Tested by `test_continual_anticollapse.cpp`. | Keep. |

### 1.4 Code-gen / HPO-NAS (real-ness VERIFIED)

| File | Verdict | Evidence | Suggested action |
|---|---|---|---|
| `src/code_gen.cpp` (1254L) | **SOLID** | **REAL.** `CodeGenerator` emits actual C++ kernels: tiled GEMM (`:162-205`), flash attention (`:219`), RMS/LayerNorm, softmax, activations, SIMD (AVX2/AVX512) load/store/FMA. Header `include/quant/code_gen.h:96-131` declares full API. **No test file.** | Add test. |
| `src/hpo_nas.cpp` (1086L) | **REFERENCE** | **REAL algorithms, FAKE evaluation.** Bayesian optimization (`:203`), PBT (`:282`), successive halving (`:329`), crossover/mutation (`:402-449`) all implemented. BUT `evaluate_config` (`:368-400`) scores configs by **distance from ideal hyperparameters** (lr≈3e-4, depth≈12, width≈768) — it never trains a model. **No test file.** | Wire to real training; add test. |

### 1.5 Adapters tree (`src/adapters/`, 33 files) — ORPHAN

**Key finding:** `src/adapters/CMakeLists.txt` is a **standalone sub-project** that links against pre-built parent libs (`:26-35`). The **main `CMakeLists.txt` never calls `add_subdirectory(adapters)`** (only `add_subdirectory(tests)` at `:401`). Therefore the entire adapters tree is an **ORPHAN** — it must be built separately and is not part of the main build.

| File | Verdict | Evidence | Suggested action |
|---|---|---|---|
| `src/adapters/CMakeLists.txt` | **ORPHAN** | Standalone project; not referenced by main CMakeLists. | Wire into main build or document as separate edition. |
| `src/adapters/include/adapters/adapter_core.h` (145L) | **ORPHAN** | Real header (mixed-precision funnel, foreign dtype dequant). | Integrate. |
| `src/adapters/include/adapters/continual_trainer.h` (119L) | **ORPHAN** | Real header. | Integrate. |
| `src/adapters/include/adapters/delta_adapter.h` (74L) | **ORPHAN** | Real header. | Integrate. |
| `src/adapters/include/adapters/gguf_bridge.h` (20L) | **ORPHAN** | Real header. | Integrate. |
| `src/adapters/include/adapters/mtp_head_trainer.h` (107L) | **ORPHAN** | Real header. | Integrate. |
| `src/adapters/include/adapters/ptq_bridge.h` (31L) | **ORPHAN** | Real header. | Integrate. |
| `src/adapters/include/adapters/safetensors_bridge.h` (20L) | **ORPHAN** | Real header. | Integrate. |
| `src/adapters/src/adapter_core.cpp` (423L) | **ORPHAN** | Real: fp16/bf16/fp8 dequant, mixed-precision write. | Integrate. |
| `src/adapters/src/continual_trainer.cpp` (268L) | **ORPHAN** | Real. | Integrate. |
| `src/adapters/src/delta_adapter.cpp` (157L) | **ORPHAN** | Real. | Integrate. |
| `src/adapters/src/gguf_bridge.cpp` (421L) | **ORPHAN** | Real GGUF bridge. | Integrate. |
| `src/adapters/src/mtp_head_trainer.cpp` (333L) | **ORPHAN** | Real MTP head trainer. | Integrate. |
| `src/adapters/src/ptq_bridge.cpp` (111L) | **ORPHAN** | Real PTQ bridge. | Integrate. |
| `src/adapters/src/safetensors_bridge.cpp` (301L) | **ORPHAN** | Real safetensors bridge. | Integrate. |
| `src/adapters/tests/test_adapter_bridges.cpp` (183L) | **ORPHAN** | Real smoke tests (FP conversions, format detection, mixed write, QUAD budget, raw load). | Integrate into main test suite. |
| `src/adapters/tools/bench_bitnet_comparison.cpp` (260L) | **ORPHAN** | Real bench tool. | Integrate/keep. |
| `src/adapters/tools/bench_compare.cpp` (529L) | **ORPHAN** | Real bench tool. | Integrate/keep. |
| `src/adapters/tools/bench_full.cpp` (867L) | **ORPHAN** | Real bench tool. | Integrate/keep. |
| `src/adapters/tools/bench_industry_real.cpp` (1038L) | **ORPHAN** | Real bench tool. | Integrate/keep. |
| `src/adapters/tools/bench_ollama_comparison.cpp` (270L) | **ORPHAN** | Real bench tool. | Integrate/keep. |
| `src/adapters/tools/bench_quant_loss.cpp` (147L) | **ORPHAN** | Real bench tool. | Integrate/keep. |
| `src/adapters/tools/dump_quant.cpp` (58L) | **ORPHAN** | Real tool. | Integrate/keep. |
| `src/adapters/tools/probe_grp.cpp` (198L) | **ORPHAN** | Real tool. | Integrate/keep. |
| `src/adapters/tools/probe_search.cpp` (405L) | **ORPHAN** | Real tool. | Integrate/keep. |
| `src/adapters/tools/probe_t4.cpp` (60L) | **ORPHAN** | Real tool. | Integrate/keep. |
| `src/adapters/tools/quant_adapt.cpp` (107L) | **ORPHAN** | Real tool. | Integrate/keep. |
| `src/adapters/tools/quant_chat.cpp` (163L) | **ORPHAN** | Real tool (chat). | Integrate/keep. |
| `src/adapters/tools/quant_import.cpp` (445L) | **ORPHAN** | Real tool. | Integrate/keep. |
| `src/adapters/tools/quant_ptq.cpp` (106L) | **ORPHAN** | Real tool. | Integrate/keep. |
| `src/adapters/tools/quant_refcheck.cpp` (323L) | **ORPHAN** | Real tool. | Integrate/keep. |
| `src/adapters/tools/quant_verify_roundtrip.cpp` (248L) | **ORPHAN** | Real tool. | Integrate/keep. |
| `src/adapters/tools/qwen35_engine.cpp` (837L) | **ORPHAN + BRANDED** | Real GLA/attention engine, but branded "Qwen35". | Rename to neutral name. |

### 1.6 BRANDED qwen35 / k3 names (VERIFIED)

- `include/quant/qwen35_engine.h` (92L) — `class Qwen35Engine` (GLA/attention inference engine).
- `include/quant/qwen35_tokenizer.h` (56L) — `class Qwen35Tokenizer` (BPE tokenizer, eos=248046).
- `include/quant/k3_tokenizer.h` (41L) — `class K3Tokenizer` (Kimi-style, vocab 152064, im_start 151644).
- `src/qwen35_tokenizer.cpp` (18KB) — **integrated into main build** (`CMakeLists.txt:131`).
- `src/k3_tokenizer.cpp` (1772B) — **integrated into main build** (`CMakeLists.txt:132`); thin wrapper over `BPETokenizer` (`:19,25,37`).
- `src/adapters/tools/qwen35_engine.cpp` — in adapters orphan tree.
- Tests: `tests/test_k3_tokenizer.cpp` exists (wired at `tests/CMakeLists.txt:53`). **No test for qwen35_tokenizer or qwen35_engine.**

These embed proprietary model names ("Qwen3.5", "K3"/Kimi) into a generic engine. **BRANDED** — should be renamed to neutral identifiers (e.g., `GlaEngine`, `BpeTokenizer`) to avoid IP/branding contamination.

### 1.7 Trainer multimodal / moe

| File | Verdict | Evidence | Suggested action |
|---|---|---|---|
| `engines/trainer/multimodal/audio/audio.cpp` (733L) | **SOLID** | Real audio encoder. In main build (`CMakeLists.txt:294`). | Keep. |
| `engines/trainer/multimodal/embeddings/embeddings.cpp` (321L) | **SOLID** | Real. In build (`:293`). | Keep. |
| `engines/trainer/multimodal/image/image.cpp` (871L) | **SOLID** | Real ImageEncoder/Decoder. In build (`:289`). | Keep. |
| `engines/trainer/multimodal/ocr/ocr.cpp` (664L) | **SOLID** | Real. In build (`:292`). | Keep. |
| `engines/trainer/multimodal/text/text.cpp` (403L) | **SOLID** | Real. In build (`:291`). | Keep. |
| `engines/trainer/multimodal/video/video.cpp` (601L) | **SOLID** | Real. In build (`:290`). | Keep. |
| `engines/trainer/multimodal/vision/vision.cpp` (776L) | **SOLID** | Real VisionEncoder/ViT. In build (`:288`). | Keep. |
| `engines/trainer/moe/moe.cpp` | **SOLID** | Real ModalityClassifier/MoERouter/MoEFFN/CrossModalAttention/MoMBlock. In build (`:191`). | Keep. |
| `engines/trainer/moe/audio/audio.cpp` (63L) | **SOLID** | Real (small) audio encoder. In build (`:193`). | Keep. |
| `engines/trainer/moe/embeddings/embeddings.cpp` (40L) | **SOLID** | Real sentence embedder. In build (`:194`). | Keep. |
| `engines/trainer/moe/image/vision.cpp` (94L) | **SOLID** | Real. In build (`:195`). | Keep. |
| `engines/trainer/moe/ocr/ocr.cpp` (86L) | **SOLID** | Real. In build (`:196`). | Keep. |
| `engines/trainer/moe/text/multimodal_text.cpp` (57L) | **SOLID** | Real. In build (`:197`). | Keep. |
| `engines/trainer/moe/video/video.cpp` (111L) | **SOLID** | Real. In build (`:198`). | Keep. |
| `engines/trainer/moe/vision/vision.cpp` (393L) | **SOLID** | Real ObjectDetector/SceneGraph/VisionEncoder. In build (`:192`). | Keep. |

### 1.8 Quant headers (AGI/multimodal side)

| Header | Verdict | Notes |
|---|---|---|
| `include/quant/agi.h` (535L) | **SOLID** | Full AGI API (all subsystems + Flywheel). |
| `include/quant/agi_utils.h` (75L) | **REFERENCE** | Small util header. |
| `include/quant/multimodal.h` (297L) | **SOLID** | Full multimodal API. |
| `include/quant/multimodal_cross_attn.h` | **SOLID** | Cross-attn API. |
| `include/quant/meta_cognition.h` (158L) | **REFERENCE** | MetaCognitionEngine API. |
| `include/quant/self_eval.h` (144L) | **REFERENCE** | SelfEvalSuite API. |
| `include/quant/world_model.h` (172L) | **REFERENCE** | WorldModel API. |
| `include/quant/multi_agent.h` (201L) | **REFERENCE** | MultiAgentSystem API. |
| `include/quant/continual_engine.h` (315L) | **REFERENCE** | ContinualEngine API. |
| `include/quant/code_gen.h` (188L) | **SOLID** | CodeGenerator API. |
| `include/quant/hpo_nas.h` (241L) | **REFERENCE** | HPOEngine/NAS API. |
| `include/quant/qwen35_engine.h` (92L) | **BRANDED** | Rename. |
| `include/quant/qwen35_tokenizer.h` (56L) | **BRANDED** | Rename. |
| `include/quant/k3_tokenizer.h` (41L) | **BRANDED** | Rename. |

### 1.9 Tests

| Test | Verdict | Notes |
|---|---|---|
| `tests/test_agi.cpp` (165L) | **SOLID** | Covers monitor/reflector/metacognition/safety/HITL/alignment/multi-agent/planning/memory/tools. |
| `tests/test_agi_flywheel.cpp` (49L) | **MISSING-TEST (weak)** | Only checks history empty, no-improvement count, log path, and that a trivial program compiles. **Does not exercise the run loop or verify correctness.** |
| `tests/test_multimodal.cpp` (173L) | **SOLID** | Covers H1-H16 + fusion. |
| `tests/test_multimodal_encoders.cpp` (152L) | **SOLID** | Covers cross-attn encoders. |
| `tests/test_continual_anticollapse.cpp` | **SOLID** | Covers continual engine. |
| `tests/test_k3_tokenizer.cpp` | **SOLID** | Covers K3 tokenizer. |
| **No test for** code_gen, hpo_nas, multi_agent, world_model, self_eval, meta_cognition, qwen35_tokenizer, qwen35_engine | **MISSING-TEST** | All real implementations with zero coverage. |

---

## 2. FROZEN-LIST CANDIDATES

Legend: **(R)** = Refactor · **(D)** = Delete · **(W)** = Wire-in/Integrate. Owner-gate flag = the subsystem owner must approve before change.

| # | Candidate | Type | file:line | Owner-gate |
|---|---|---|---|---|
| 1 | **Flywheel fake test harness** — `generate_test_program` never tests solution code; every compiling program scores 0.8. | (R) | `src/agi_flywheel.cpp:863-911`, `:1387-1391` | AGI owner |
| 2 | **Flywheel heuristic "improvement"** — `measure_improvement` uses keyword-count, not real capability. | (R) | `src/agi_flywheel.cpp:1418-1476` | AGI owner |
| 3 | **PersonaHotSwap dead class** defined mid-file, not in header, no callers. | (D) | `src/agi_extended.cpp:756-805` | AGI owner |
| 4 | **encode_image/encode_audio sentinel stubs** returning `{30000}`/`{31000}`. | (R) | `src/multimodal.cpp:674-682`; `include/quant/multimodal.h:140-141` | Multimodal owner |
| 5 | **BRANDED qwen35/k3 names** — proprietary model names in generic engine/tokenizer. | (R) | `include/quant/qwen35_engine.h`, `qwen35_tokenizer.h`, `k3_tokenizer.h`; `src/qwen35_tokenizer.cpp`, `src/k3_tokenizer.cpp`, `src/adapters/tools/qwen35_engine.cpp` | Adapters owner |
| 6 | **Adapters orphan tree** — standalone sub-project not wired into main build. | (W) | `src/adapters/CMakeLists.txt`; main `CMakeLists.txt:401` (no add_subdirectory) | Build owner |
| 7 | **Seed-42 magic constants** scattered across AGI files. | (R) | `src/agi_extended.cpp:104,252,267,295`; `src/agi.cpp:924`; `src/self_eval.cpp:504,586`; `src/world_model.cpp:62`; `src/multi_agent.cpp:736` | AGI owner |
| 8 | **HPO `evaluate_config` fake scoring** — never trains a model, scores by distance to ideal hyperparams. | (R) | `src/hpo_nas.cpp:368-400` | HPO owner |
| 9 | **Missing tests** for code_gen, hpo_nas, multi_agent, world_model, self_eval, meta_cognition, qwen35_tokenizer, qwen35_engine. | (W) | `tests/` (absent) | Test owner |
| 10 | **Weak flywheel test** — does not exercise run loop. | (R) | `tests/test_agi_flywheel.cpp` | AGI owner |

---

## 3. TOP RISKS

1. **AGI flywheel is cosmetic, not self-improving (HIGH).** The fake test harness (`agi_flywheel.cpp:863-911`) never validates solution correctness, and `measure_improvement` (`:1418`) uses keyword heuristics. The flywheel's "improvements" are not real — it cannot detect or apply genuine code improvements. Any claim of self-improvement is unsupported. This is the single most important integrity issue in the AGI scope.

2. **Adapters tree is an orphan (HIGH).** The entire 33-file `src/adapters/` tree (bridges, trainers, 20+ tools, tests) is a standalone sub-project not referenced by the main `CMakeLists.txt`. It is effectively dead from the main build's perspective — real code that is not compiled or tested as part of the product. Risk of bit-rot and silent divergence.

3. **BRANDED proprietary model names (MEDIUM-HIGH).** `Qwen35Engine`, `Qwen35Tokenizer`, `K3Tokenizer` embed third-party model names ("Qwen3.5", "Kimi") into generic engine/tokenizer code. This is an IP/branding contamination risk and misrepresents the code as tied to those models. Should be renamed to neutral identifiers.

4. **Multimodal tokenizer is a stub (MEDIUM).** `encode_image`/`encode_audio` return fixed sentinel tokens (`{30000}`/`{31000}`) regardless of input. The multimodal pipeline cannot actually encode real images/audio — only the constant IDs are tested. Any downstream multimodal feature relying on real encoding is non-functional.

5. **HPO/NAS evaluation is fake (MEDIUM).** `hpo_nas.cpp:368-400` scores hyperparameter configs by distance to hand-picked "ideal" values rather than training/evaluating a model. The Bayesian/PBT/successive-halving machinery is real, but the objective it optimizes is a heuristic, so results are not grounded in actual model performance.

---

*Report generated by A6 (Adapters/Multimodal/AGI) — Phase-0 re-context. All line numbers verified against current tree. No source files modified.*
