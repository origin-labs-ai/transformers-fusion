# A1 — Codec/Format Core — Phase 0 Re-Context Report (READ-ONLY)

**Scope:** `src/block_codec.cpp`, `src/codebook.cpp`, `src/format_*.cpp`, `src/vector_quantizer.cpp`, `src/quant_engines_*.cpp`, `include/quant/types.h`, `include/quant/format_registry.h`, `include/quant/codebook.h`, `include/quant/block_codec.h`, `tests/test_*codec*.cpp`, `tests/test_*format*.cpp`, `tests/test_quant_mix.cpp`, `tests/test_grp*.cpp`, `quant_config.h.in` + session-archive pass
**Mode:** READ-ONLY, no edits
**Date:** 2026-09-01 Phase-0

## 1. Slice Inventory — File → Verdict → Evidence → Action

| File | Verdict | Evidence | Action |
|------|---------|----------|--------|
| `src/block_codec.cpp` (~2295L) | **SOLID** (6 wounds) | `f32_to_f16:26`, `Comp8Table:216`, `comp8_fit_scale_fp16:328`, `FastBitReader:123`, `grp16_fit_scale:704`, `quant_6k:997` wired `CMakeLists.txt:73` | keep + fix 6 wounds |
| `src/codebook.cpp` (452L) | **SOLID** | `CodebookQUANT8::train:70`, `ema_update:131`, magic `0x51554138` | keep |
| `src/format_planner.cpp` (340L) | **REFERENCE** (OOB) | `score_importance:12`, `compute_format_mix_extended:71`, OOB `ad[i]` weight `30` no guard | fix OOB + keep |
| `src/format_registry.cpp` (595L) | **REFERENCE** (mapping bug) | `build_singles:19` 22 desc, `quantize:152` slices `WIRE_BLOCK=256`, `format_to_regformat:165` maps `Q_G_1_5..Q_G_24_5→Q1` bug | fix mapping + keep |
| `src/vector_quantizer.cpp` (395L) | **SOLID** | `initialize_codebook:21`, `QUANT8VectorQuantizer:166`, `QUANT4VectorQuantizer:229` | keep |
| `src/quant_engines_core.cpp` (165L) | **DEAD** | 5 `static` AVX2 helpers `dequant_tensor_quant8_avx2:15` zero callers (static TU) | delete or wire |
| `src/quant_engines_quant.cpp` (261L) | **REFERENCE** | ternary `{-1,0,+1}` `quantize:16`, NOT canonical wire encoding | keep (rename clarifier) |
| `src/quant_engines_quant8.cpp` (273L) | **REFERENCE** | 256-entry LUT `train_codebook:21`, in-memory only | keep |
| `src/quant_engines_quant4.cpp` (330L) | **REFERENCE** | 16-entry `train_codebook:21`, in-memory only | keep |
| `src/quant_engines_quant2.cpp` (293L) | **REFERENCE** | 4-entry `train_codebook:21` | keep |
| `src/quant_engines_quant1.cpp` (142L) | **REFERENCE** | block-mean `quantize:17`, in-memory only | keep |
| `src/quant_engines_fp.cpp` (576L) | **SOLID** | FP8 `fp8_e4m3_dequantize:17`, `QUANT16Engine:412` | keep |
| `include/quant/types.h` (178L) | **SOLID** (2 nits) | `Format:22` `FORMAT_COUNT=105:67`, `format_bpw:97` truthful `Q2_G 2.625` | fix comment + keep |
| `include/quant/format_registry.h` (227L) | **REFERENCE** | `RegFormat:9` 33 ids, mapping bugs `165-187` | fix mapping + keep |
| `include/quant/codebook.h` (139L) | **SOLID** | `CodebookQUANT8:28` SIZE 256, `CodebookQUANT4:57` SIZE 16 F16 | keep |
| `include/quant/block_codec.h` (85L) | **SOLID** | BPW table `11-54`, `quantize_block_all:69` | keep |
| `quant_config.h.in` (12L) | **SOLID** | `QUANT_VERSION_* 0.1.2`, `#cmakedefine QUANT_AVX2` | keep |
| `tests/test_block_codec.cpp` (101L) | **STALE** | smoke only `test_q24_codec:10`, not testing `quantize_block_all` | fix + keep |
| `tests/test_fuzz_codec.cpp` (481L) | **SOLID** | L036 fuzz `kCaps:118` 104 caps, `roundtrip:260` budget `claimed+1:272`, bug `v==19` skip `249` claims gap but `Q4_K_L` valid | fix gap skip + keep (RED BY DESIGN until BPW decision) |
| `tests/test_format.cpp` (178L) | **STALE** | enum checks `FORMAT_COUNT==105:16`, duplicate MXQ `157-163` | dedup + expand to 105 |
| `tests/test_format_registry_complete.cpp` (61L) | **REFERENCE** | checks `get_all_singles>=19:16`, minimal | keep |
| `tests/test_format_audit.cpp` (79L) | **SOLID** | BPW ironclad probe `55-76`, exits 1 BY DESIGN proving 31 violations | keep (gate until owner BPW decision) |
| `tests/test_quant_mix.cpp` (671L) | **STALE/SHELL** | `q0=get_twi_mix(1.5):99` expects 2-tier TWI which empty → fail | rewrite fixtures to `get_four_mix` / `MXQ_3_5_G` |
| `tests/test_grp_quality_proof.cpp` (91L) | **SOLID** | GRP superiority `grp>=plain:83` | keep |
| `tests/test_grpo.cpp` (220L) | **REFERENCE** (mis-named) | GRPO RL correctness, not GRP codec — naming collision | rename to `test_grpo_rl.cpp` |

## 2. Wounds Detail (codec)

1. Affine encode/decode mismatch `block_codec.cpp:1204-1217` bits 1/6 grids diverge — fix unify
2. Slot comment vs behavior `120-121` vs `452-531` — fix comment
3. Dead code `comp8_scale_search:244`, `f32_bits:188`, `level_value_r:433` — delete
4. Unreachable wire payload `1986,2010,2184,2210` return before memcpy — delete
5. MXQ_G guard/writer 3.78 vs 3.5 at n=256 — owner decision fit-vs-rename
6. Silent truncation `BitWriter::put:92` silent return — add QUANT_CHECK

## 3. BRANDED Inventory — Slice CLEAN
- `k3/K3/kimi/kda/mla/qwen35/innova` = 0 hits in slice code (header guards only). Branded files are out-of-slice: `k3_tokenizer.h:41`, `qwen35_*`, `kda_attention`, `mla_attention` per 100-hit grep.

## 4. STUB/TODO Inventory — Slice CLEAN
- `TODO(G-4)` only in `tools/k3_convert.cpp:96` out-of-slice
- `FIXME/XXX/HACK/STUB` 0 hits in slice
- Silent passthrough risks: `BitWriter::put:92` silent, `BitReader::get:109` breaks, `FastBitReader::get:130` returns 0 past end, `format_is_twi_mix:133` always false (dead alias), `build_two_mixes:73` empty by design
- Slice has no STUB markers; behavioral shells out-of-slice

## 5. Session-Archive Pass (A1 extra)

| File | Verdict | Evidence |
|------|---------|----------|
| `repo/TRANSCRIPT.md` (2048L) | STALE/REFERENCE | PART A win register vs `block_codec.cpp:356` 8.5 BPW consistent, PART D wound table legacy aliases without rename, C-01..C-25 PENDING vs live `research/claim_ledger.md` 14 VERIFIED. Bench CSV stale post-v3 per `audit_full_20260825.md:F` |
| `repo/sessions/_chat_extract.txt` (821L) | REFERENCE | Hinglish P1-P73, InNova frozen-core doctrine |
| `repo/sessions/_zcode_sessions.txt` (~8.5K msgs) | REFERENCE | 5-audit sweep with file:line, verified `_GRP→_G` rename |
| `repo/sessions/_opencode_session00.txt` (453L) | REFERENCE | 2026-08-26 26 files 452 replacements, letters-before-numbers rule |
| `repo/memory/01-11.md` | SOLID | 11 files, BPW ironclad Rule1, block_codec wounds 03:48-51, citation policy 05:32, static bug 10:54 |
| `research/workbench.md` (75L) | SOLID | Phase 0 DONE etc., blockers `test_fuzz_codec` RED BY DESIGN |
| `research/claim_ledger.md` (74L+addendum) | SOLID | 25 verdicts + A-01..A-13 |
| `research/audit_full_20260825.md` (200L) | SOLID | B-01..B-05 fixed, 31 BPW violations, bench stale |
| `research/claims/*` (3 files) | SOLID | granular ledger |

Session archive coherent: TRANSCRIPT frozen 2026-08-22 + live workbench + ledger + audit form consistent chain.

## 6. Build Reality
- Wiring `CMakeLists.txt:68-84` quant_format explicit source list — not orphan
- Consumers: `format_registry.cpp:152` → `block_codec.h` canonical; `vector_quantizer` independent; `quant_engines_*` in-memory engines with own codebook_ vectors (`quant_engines.h:56,86`) — naming collision not bug
- BPW contract `block_codec.h:65` vs `types.h:112` now truthful after fix, but stores > claim for `Q2_G..Q24_G` (31 violations)
- Tests vs code drift: `test_fuzz_codec:249` skips valid id 19 (`Q4_K_L`) — fix one-line

## 7. Suggested Actions Ranked
- P0 Owner BPW decision (fit vs rename) — until then `test_fuzz_codec`+`test_format_audit` stay red by design
- P0 Delete or wire `quant_engines_core.cpp` 165L dead
- P1 Fix affine mismatch `1204` + slot comment + dead code sweep
- P1 Fix `test_fuzz_codec:249` gap + `test_format.cpp:157` duplicate + `test_quant_mix:99` TWI fixture
- P1 Fix `format_registry.h:177` half-GRP→Q1 map + MXQ plain→GRP collapse
- P1 Add QUANT_CHECK to `BitWriter` overflow `92`
- P2 Rename `test_grpo.cpp` out of codec glob
- P2 Wire `src/adapters/` or archive (owner decision)
- P3 Sync `TRANSCRIPT.md` PART-A to `audit_full_20260825.md:I`

**Overall:** SOLID core, STALE tests, ONE dead file. Codec math production-grade; registry/planner reference-grade with small mapping/OOB bugs; build correct; zero branded/STUB contamination in slice; 3 tests need post-v3 migration and 1 dead file should be deleted or wired.
