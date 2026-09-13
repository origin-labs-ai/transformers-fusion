# INNOVA PERSISTENT MEMORY — MASTER INDEX

> Persistent memory system for the InNova project, modeled after the ZCode CLI memory folder.
> Every agent session MUST read this index first, then load the files relevant to the task.
> Every agent session MUST update the relevant memory files before ending (especially `08-current-state-*.md`).
> Raw session sources live in `repo/sessions/`. Operational state lives in `repo/state/`.
> Created: 2026-08-31 by the Verdent session. Language: English (owner's explicit order — "saari memory English me convert kar and ultra detailed banaa").

## File Index

| File | Content | When to read |
|---|---|---|
| `01-user-profile.md` | Who the owner is, how he communicates, what he values/hates, all locked decisions, working agreement | ALWAYS (every session) |
| `02-iron-rules.md` | The 14 binding rules from TRANSCRIPT.md PART-B with rationale, past violations, enforcement | ALWAYS (every session) |
| `03-project-map.md` | Full technical map: stats, 105-format system, codec internals, library layering, model stack, GPU backends, training stack, tests | Any code/format/test work |
| `04-philosophy-mean.md` | The project's single core meaning ("never overwrite"), the RLL loop, the Transformers-blame thesis, kernel mission justification | Any architecture/mission work |
| `05-transcender-mission.md` | The Transformer-replacement mission: locked name, citation policy, Phases 0-4, kernel LOC plan, .txn IR idea | Any Transcender work |
| `06-zcode-sessions-history.md` | Full chronology of all 29 ZCode sessions (Aug 24-29) with every finding, fix, decision, quote; master wound list | Before any repair/audit work |
| `07-opencode-session00.md` | OpenCode SESSION-00 (format naming session) + reusable DB-extraction methodology | Format naming work |
| `08-current-state-20260831.md` | Ground truth as of 2026-08-31: git, tests, today's session log, open queue, environment notes | ALWAYS (every session) |
| `09-repo-layout.md` | The research/ + repo/ restructure: what moved where, why, code-path updates, rules for new files | Any file placement work |
| `10-code-review-findings.md` | Verdent session's independent code review (2026-08-31): codec praise, autograd teardown, KDA perf, ste_quantizer static-codebook bug with fix options | ste_quantizer work, Transcender Phase 2 |
| `11-verdent-session-log.md` | Turn-by-turn log of the 2026-08-31 Verdent session (the full conversation arc, all owner interventions) | Context recovery |
| `14-todos-cender-20260906.md` | 23 todos exact + D-C1..D-C11 Cender decisions + 33 user inputs verbatim index (2026-09-06 owner order: memory+telemetry dump, nothing missed) | Cender/todo work |

## Critical Quick-Facts (memorize these)

1. **Owner**: 15-year-old solo developer. Hinglish casual chat ("saale"/"bhai" are friendly). Mission: beat llama.cpp in every measurable dimension — his exact words: "llama.cpp waalo ki band bajani hai" and "Baap to banna hi hai!"
2. **Architecture name: Transcender** (locked 2026-08-31). The engine is InNova; the architecture running on it is Transcender. Tagline: "Attention was all you needed. Transcendence is all you need."
3. **The single core meaning of the entire project**: "Never overwrite." Frozen core + additive correction pages; decay + delta-write; 0% catastrophic forgetting. Everything else derives from this one rule.
4. **The Transformers-blame thesis**: today's model diseases (Context Rot, KV-cache bloat, brute-force compute) are genetic defects of the 2017 softmax-attention architecture, NOT tooling problems. If they were PyTorch's fault, building InNova would have been enough. Therefore the real mission is to REPLACE the Transformer.
5. **Iron rule #1 — BPW ironclad**: a format must never store more bits per weight than its name claims. 31 violations were proven by the audit probe (2026-08-25/26); the owner decision (fit-in-claim vs rename-to-actual) is still pending.
6. **Test state**: 52/56-class passing. The 3 documented failures: `test_fuzz_codec` (BPW contract guard — failing BY DESIGN until owner decides), `test_quant_mix` (needs MXQ fixture redesign), `paged_kv_4m` (latent segfault, mid-debug at S9-S12 markers).
7. **All committed (2026-08-31 21:0x, commits 96295e9..1a376f1)** — the audit repair set, restructure, and memory system are in git. Working tree clean. `repo/.mimosa/` gitignored (41 MB hook cache, on disk only). TRANSCRIPT.md lives at `repo/TRANSCRIPT.md` (owner moved it; 2048 lines intact).
8. **`tests/test_bpw_150_proof.cpp` deletion is PERMANENT** — never restore it, never re-add its CMakeLists reference. This is a locked owner decision.
9. **Persona code is banned from the codebase** — personality comes from the system prompt, not engine code. Known violation: `src/agi_extended.cpp:756` (PersonaHotSwap class), documented for TRANSCRIPT follow-up.
10. **Chat in Hinglish, commits in English** (conventional commit style, e.g. `perf(codec): ...`).
11. **Sub-agent policy**: owner allows max 2 parallel; on 2026-08-25 the environment blocked even 1 ("model concurrency limit exceeded") — solo fallback is the proven path.
12. **Format system v3**: FORMAT_COUNT=105, dot-BPW names (Q8.5, Q_G_6.5, MXQ_3.5_G), `_GRP` → `_G` rename complete, TWI/QUAD deleted project-wide, enum slot 19 = Q4_K_L (not a gap).
13. **Bench tool is stale post-v3** (43-format matrix, TWI rows) — every CSV-derived claim is frozen/unreliable until the bench migrates to the 105-format matrix.
14. **Version drift**: CMakeLists says 0.1.2, README says v0.1.03, CHANGELOG says 0.1.02 — three different version stories, needs a one-truth fix.
15. **Shell environment**: only Git Bash works (`shellExecutable: C:\Program Files\Git\bin\bash.exe`); plain cmd/powershell tools are missing from PATH. Windows directory locks are real — use `robocopy /MOVE` (full path `C:/Windows/System32/Robocopy.exe` with `//`-prefixed flags in Git Bash) or cp+rm retry.
16. **Node v24 is available** (with built-in `node:sqlite`) — this is how OpenCode/ZCode databases were read. Python is NOT installed (Windows Store alias only).
17. **ZCode memory inheritance**: this memory system replaces/absorbs the ZCode memory at `~/.zcode/cli/memories/projects/innova-b81804904f2e081b/memory/` (copy preserved at `repo/sessions/zcode-memory-original/`).
18. **The kernel LOC mission**: grow kernel code from ~9.8K to 25K+ lines as part of Transcender Phase 3 — but ONLY with mission-aligned kernels (chunked delta-rule first), never padding.
19. **ste_quantizer static-codebook bug** (found by Verdent 2026-08-31, NOT in the ZCode wound list): function-local static codebook trains on the first tensor seen and is then reused for every subsequent layer. Fix options documented in `10-code-review-findings.md`.
20. **Restructure done 2026-08-31**: `.research/` → `research/` (root, reports only); `repo/` created with `memory/`, `sessions/`, `state/`, `.mimosa/` (true cut-paste, verified). Code paths updated in gle_report.cpp, test_gle_telemetry.cpp, release.yml.
