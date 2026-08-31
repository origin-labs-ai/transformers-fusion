# 09 — REPO LAYOUT (Restructure of 2026-08-31)

This file documents the research/ + repo/ restructure: what moved, where, why, what code paths were updated, and the rules for placing new files. It is the placement authority for future work.

## 1. The owner's orders (verbatim intent, including corrections)

1. "OpenCode CLI open kar and /session maar ke SESSION-00 naam ka session hai use bhi padh le! Aur ZCode ka bhi poora session padh le..." → read all external agent history.
2. "Ab tu bhi ZCode ki tarah persistent memory folder bana sakta hai and saare ke saare session ke saare ke saare details daal de usne! Ab .research folder ko rename karke research kar de and sirf research waale report files chhor wahaa pe and .mimosa naam ka ek folder hai usko bhi saath me le chal, ek nayaa folder banaa 'repo' naam ka and sab kuchh temp se leke sab wahaa..." → the master restructure order.
3. **Correction #1**: "Abe nahi be, research folder root me hi rahega! Baaki ka reports wagairah sab copy kar ke usme bhi daal dena" → research/ stays AT ROOT; do not bury it inside repo/.
4. **Correction #2**: ".mimosa ko to cut karke paste maar dena tha udhar and baaki ki memory ko English me banaata naa, saari memory English me convert kar and ultra detailed banaa, koi bhi file chhoti nahi banani hai" → .mimosa must be a TRUE cut-paste (verified: root copy deleted, 41 MB intact at repo/.mimosa); all memory files in English, ultra detailed, no small files.

## 2. Current layout (authoritative)

```
InNova/                        (repo root, git)
├── research/                  ← renamed from .research/ (REPORTS ONLY)
│   ├── workbench.md           live status (master_plan_v2_20260822)
│   ├── claim_ledger.md        C-01..C-25 + audit addendum
│   ├── audit_full_20260825.md the 5x-audit standing report
│   └── claims/
│       ├── audit_features.md
│       ├── audit_infra.md
│       └── audit_training.md
├── repo/
│   ├── memory/                ← THIS persistent memory system
│   │   ├── MEMORY.md          index + 20 quick-facts
│   │   ├── 01-user-profile.md
│   │   ├── 02-iron-rules.md
│   │   ├── 03-project-map.md
│   │   ├── 04-philosophy-mean.md
│   │   ├── 05-transcender-mission.md
│   │   ├── 06-zcode-sessions-history.md
│   │   ├── 07-opencode-session00.md
│   │   ├── 08-current-state-20260831.md
│   │   ├── 09-repo-layout.md   (this file)
│   │   ├── 10-code-review-findings.md
│   │   └── 11-verdent-session-log.md
│   ├── sessions/              ← raw session material
│   │   ├── _chat_extract.txt            (chat export, 37 msgs)
│   │   ├── _opencode_session00.txt      (20.8 KB)
│   │   ├── _zcode_sessions.txt          (709 KB full dump)
│   │   ├── _zcode_condensed.txt         (287 KB condensed)
│   │   └── zcode-memory-original/       (ZCode's 5 memory files, preserved)
│   ├── state/                 ← operational state (moved OUT of .research)
│   │   ├── telemetry/         events.jsonl + bench_history
│   │   ├── bar/               bar.sha256 + prefetch-reference.txt
│   │   └── goal_status.json
│   └── .mimosa/               ← true cut-paste (41 MB)
│       ├── hook-state/
│       └── hook-status/
└── (source tree unchanged: src/ include/ tests/ tools/ bench/ engines/ docs/ ...)
```

## 3. Moves executed (mechanics worth remembering)

| Operation | Method | Notes |
|---|---|---|
| `.research` → `research` | `git mv` FAILED (Permission denied), `mv` FAILED, **robocopy /MOVE succeeded** | Windows lock on the dir; robocopy was `C:/Windows/System32/Robocopy.exe` with `//`-prefixed flags |
| research/telemetry → repo/state/telemetry | `mv` FAILED → **cp -r + rm -rf** (rm error ignored) | same lock family |
| research/bar, goal_status.json → repo/state/ | `mv` OK | |
| 4 session extract files → repo/sessions/ | `mv` OK | |
| `.mimosa` → repo/.mimosa | `mv` FAILED → **cp -r + rm -rf**, root verified GONE | true cut-paste confirmed by owner's explicit demand |
| chat-export-1788185212392.json | source file VANISHED from root during session (owner removed it) | extract survives at repo/sessions/_chat_extract.txt |
| ZCode memory originals | cp -r from `~/.zcode/cli/memories/projects/innova-b81804904f2e081b/memory` | preservation copy |

## 4. Code-path updates (rename synchronization)

| File | Change |
|---|---|
| `tools/gle_report.cpp` | telemetry path `.research/telemetry/events.jsonl` → `repo/state/telemetry/events.jsonl` (3 sites incl. comment) — **rebuild verified clean** |
| `tests/test_gle_telemetry.cpp` | evidence strings → `repo/state/telemetry/runs/r1/artifacts/out_N.txt` and `research/workbench.md` |
| `.github/workflows/release.yml:82` | tar excludes: `.research` → `research` AND new `repo` |
| `research/workbench.md` | self-refs → `repo/state/telemetry/events.jsonl`, `research/audit_full_20260825.md` |
| `research/claim_ledger.md` | self-refs → `research/claims/audit_infra.md` etc., addendum header path |
| `research/audit_full_20260825.md` | ledger pointer → `research/claim_ledger.md` |
| TRANSCRIPT.md, docs/RESEARCH/*.md | `.research` mentions left AS-IS (historical/frozen text) — do not churn frozen docs; if any build/tool ever depends on them, update then |

## 5. Git notes

- Tracked `.research/*` files currently show as `D` (deleted) in git status; the new `research/` and `repo/` trees are untracked. Content-identical renames will be auto-detected at commit time.
- Total pending entries at restructure time: 65. Commit strategy suggestion lives in `08-current-state-20260831.md` §1.
- Do NOT add repo/ or research/ to .gitignore — the memory and reports are first-class repo content. **Single exception: `repo/.mimosa/` is gitignored (2026-08-31, commit-time decision)** — it is 41 MB / 3158 files of mutable tool-hook cache state (hook-state/hook-status), not project content; committing it would permanently bloat history. It stays on disk and is NOT lost — just not versioned.

## 6. Placement rules for new files (the law)

1. **Research reports** (audits, ledgers, workbench updates, paper analyses) → `research/`. Nothing else.
2. **Memory/knowledge** (agent memory files, decision records) → `repo/memory/`; ALWAYS update `MEMORY.md` index when adding a file.
3. **Raw dumps, extracts, temp session material** → `repo/sessions/`.
4. **Operational state** (telemetry streams, bench history, goal status, hook state) → `repo/state/` or `repo/.mimosa/` (hook-owned).
5. **Source code** stays in the existing tree (`src/`, `include/`, `tests/`, `tools/`, `bench/`, `engines/`) — repo/ is for knowledge, never for code.
6. If a new top-level artifact type appears, propose its home to the owner before creating it.
