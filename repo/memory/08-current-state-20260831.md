# 08 — CURRENT STATE (2026-08-31)

Snapshot of ground truth at the end of the 2026-08-31 Verdent session. Update this file (and re-date the filename) whenever a session ends.

## 1. Git state

- Branch: `main`.
- Recent commits: `10735a6` perf(codec) speed round 1 · `557c32f` docs+audit competitor analysis + Wave 3 ledger C-01..C-21 + faithful GGUF Q4_K ref · `a8b4a26` MXQ/K/GRP rename propagation (remove TWI, dot BPW) · `390079a` format variants re-arranged per BPW · `9da085f` GRP≥plain tie elimination.
- **~60-65 entries uncommitted** (git status count: 65 at restructure time). Contents: the whole Aug 25-26 repair set (B-1/2/3 build breakers, codec bug fixes, 6 test migrations, `_GRP`→`_G` rename completion, BPW truth fix, BPW probe test, FastBitReader speed paths) PLUS the Aug 31 restructure (research/ rename, repo/ tree, path updates in tools/tests/CI) PLUS the memory system itself.
- **First priority on owner's green light: commit this tree** (conventional message, English). Suggested split: (1) audit fixes, (2) rename+truthfix, (3) restructure+memory.
- Tracked `.research/*` files appear as deletions; renames auto-detect on commit (moves were done with plain mv/robocopy because `git mv` hit Windows locks).

## 2. Test state (last full fresh run: Aug 26; smoke re-run Aug 31)

- **52/56-class passing** (Aug 31 ctest smoke: 53/56 visible, 95%).
- Standing failures, all documented and owned:
  1. `test_fuzz_codec` — BPW contract enforcement; red BY DESIGN until the owner BPW decision (31 violations live-caught).
  2. `test_quant_mix` — silent crash (exit 3); built on deleted TWI-era fixtures; needs MXQ redesign (owner-gated).
  3. `paged_kv_4m_test` — latent segfault after S12 (post clear()→load_from_disk(), before final get_range print); numel=1 empty-tensor mystery; session died mid-debug Aug 26.
- `test_format_audit` (BPW probe) exits 1 by design — it is the evidence engine for Rule 1.

## 3. What the 2026-08-31 Verdent session did (chronological)

1. **Project scan** — stats (176 src files, 130 headers, 58 tests, 19 tools, 166 exes), live ctest smoke (53/56), honest-flags pass.
2. **Deep code review** — read block_codec.cpp (2295 L), autograd_engine.cpp, math_avx2.cpp, kda_attention.cpp, ste_quantizer.cpp. Verdict spread: codec production-grade; autograd/math solid; KDA/MLA reference-grade; ste_quantizer carries the static-codebook bug (new finding — see `10-code-review-findings.md`).
3. **Kernel 25K+ LOC mission** — surveyed kernels (~9,862 LOC), built a 14-item todo, read format layouts (Q24/Q16 canonical payloads, dispatch tables). **Cancelled by owner at survey stage** ("Abe saale, tu naye kernels kyu bana raha hai?!") — zero lines written. Mission later re-anchored as Transcender Phase 3.
4. **Chat export read** (owner's order) — 37 messages → distilled the single core meaning: **"never overwrite"** (`04-philosophy-mean.md`).
5. **Transformers-blame thesis received** — the mission redefined: replace the Transformer, not just optimize tooling.
6. **Age reveal** — owner is 15; respect doubled, behavior unchanged.
7. **K3-provenance debate** — "chor" fear resolved via own-names + honest citations; **architecture name locked: Transcender**; mission phases 0-4 defined (`05-transcender-mission.md`).
8. **Session archaeology (owner's order)** — OpenCode SESSION-00 (61 msgs) read via opencode.db; all 29 ZCode InNova sessions (1013 msgs) extracted from ~/.zcode/cli/db/db.sqlite and read; memory inheritance from `~/.zcode/cli/memories/.../innova-*/memory/`.
9. **Restructure (owner's orders, with two corrections)** — `.research` → `research/` AT ROOT (owner correction #1: "research folder root me hi rahega!"); `repo/` created with memory/sessions/state/.mimosa; `.mimosa` true cut-paste verified (root gone, 41 MB intact at repo/.mimosa); reports-only rule enforced in research/ (telemetry/bar/goal_status moved OUT to repo/state); session extracts moved to repo/sessions; ZCode memory originals copied.
10. **Code-path sync** — tools/gle_report.cpp (telemetry path ×3), tests/test_gle_telemetry.cpp (evidence strings ×2), .github/workflows/release.yml:82 (tar excludes now `research` + `repo`), research/workbench.md + claim_ledger.md + audit_full_20260825.md self-references. gle_report rebuilt clean (Debug).
11. **Memory system build** — repo/memory/ with this index + 01-09, then owner's refinement order: ALL memory in ENGLISH, ULTRA DETAILED ("koi bhi file chhoti nahi banani hai") → full rewrite pass (this file set), plus new 10 (code review) and 11 (session log).
12. **Chat export note**: `chat-export-1788185212392.json` vanished from root during the session (owner removed it); its extract is preserved at repo/sessions/_chat_extract.txt.

## 4. Open queue (priority order)

1. **Commit the uncommitted tree** (owner green light) — 3-commit split suggested above.
2. **Owner decisions pending**: BPW fit-vs-rename · adapters/ fate · bench migration timing · test_quant_mix fixture design · Transcender Phase 0 start.
3. **Transcender Phase 0** (renames + citation headers) — awaiting "shuru".
4. **paged_kv_4m segfault** — resume at S12→final-get_range region; instrument empty-tensor numel semantics in clear/load_from_disk.
5. **test_quant_mix** — MXQ fixture redesign (after owner picks shape).
6. **Bench migration** — 43→105 matrix, Release-only runs, re-freeze baselines, then re-derive W-claims (A-02 not-reproducible resolution).
7. **Transcender Phase 3 kernels** — locked order in `05-transcender-mission.md` §6 (kernel_kda first).
8. **ste_quantizer static-codebook fix** — per-tensor codebook or per-block fit (see `10-code-review-findings.md` §3).
9. Backlog (from wounds): MTP wiring-or-removal, MLA cache use, spec-decode real acceptance, GPU_VULKAN factory case, persona removal (TRANSCRIPT-documented path), MoE grad chain, PPO/DPO real backprop, YARN mscale, silent-load-skip, version drift, README:147 refresh, CI `|| true`, seed-42 sweep.

## 5. Environment notes (hard-won, reuse these)

- **Shell**: only Git Bash works — pass `shellExecutable: C:\Program Files\Git\bin\bash.exe`. Plain `cmd`/PowerShell tool lookups fail (`dir`, `Get-ChildItem`, `echo`, `pwd` all "command not found" — the shell wrapper needs the explicit executable).
- **Windows locks are real**: `git mv` and `mv` on directories can fail with Permission denied (some process holds the dir). Use `robocopy`: `"C:/Windows/System32/Robocopy.exe" src dst //E //MOVE //R:2 //W:1 //NFL //NDL //NJH //NJS` (double-slash flags in Git Bash). For stubborn subdirs: `cp -r` then `rm -rf` and ignore the rm error if the copy verified.
- **Node v24.17.0** available (`node:sqlite` built-in) — the DB extraction workhorse. **Python is NOT installed** (Windows Store alias errors out).
- **Verdent file tools**: >256 KB files need offset/limit reads; the 6,144-line README and the 709 KB ZCode dump were read chunked; a condensed skeleton (session headers + user msgs + capped AI msgs) made the 29-session corpus tractable (287 KB → readable in 4-6 passes).
- **ctest**: `ctest --test-dir build -C Release` works; full suite ≈ 2 min.
- **CMake targets**: `cmake --build build --target <name> --config Debug` (multi-config; MSVC 18.9.1).
- ZCode/OpenCode DBs: extraction recipe in `07-opencode-session00.md`.

## 6. Standing verify commands

```
ctest --test-dir build -C Release            # full suite (~2 min)
cmake --build build --target gle_report --config Debug   # path-ref sanity
node -e "..." (see 07)                        # session archaeology when needed
```
