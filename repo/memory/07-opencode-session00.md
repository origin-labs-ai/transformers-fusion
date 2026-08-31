# 07 — OPENCODE SESSION-00 (Format Naming Session)

## Session metadata

| Field | Value |
|---|---|
| Tool | OpenCode CLI (`~/.local/share/opencode/opencode.db`, SQLite: session/message/part tables with JSON `data` columns) |
| Session ID | `ses_fc14bbcd5ffe4TEzERTQhBTl6Z` |
| Title | SESSION-00 |
| Messages | 61 (text-bearing ~35; tool steps filtered) |
| Date | 2026-08-26, starting 15:33 local |
| Model | muse-spark-1.2-contributor-free (opencode provider), agent=build, variant=xhigh |
| Raw extract | `repo/sessions/_opencode_session00.txt` (20,794 chars) |
| Read by Verdent | 2026-08-31 (owner's order: "OpenCode CLI open kar and /session maar ke SESSION-00 naam ka session hai use bhi padh le!") |

## What the session decided

This session designed the **105-format naming architecture** that the whole repo now speaks:

1. **Dot-BPW display convention**: formats are named by their exact claimed bits-per-weight with a dot — `Q8.5`, `Q_G_6.5`, `MXQ_3.5_G`. This makes BPW claims visible in the name itself, which is the cultural prerequisite of Iron Rule 1 (BPW ironclad): if the name says 8.5, the wire must carry 8.5.
2. **`_GRP` → `_G` rename design**: uppercase token `_GRP` becomes `_G` everywhere; lowercase internals (`grp16_fit_scale`, `format_is_grp`, `kGrp16Size`) are preserved. Enum identifiers use underscores (`Q8_G`, `Q_GRP_8_5`, `MXQ_3_5_G`); string/display names use the dot form.
3. **Half-GRP edge cases**: display forms like `Q_GRP_8.5` → `Q_G_8.5` (the session counted 45 `_GRP` occurrences in the file it was editing and enumerated each).
4. **Scope discipline**: enum + string tables + all consumers (codec, registry, planner, tests, tools, benches) in one atomic sweep — which is exactly how it later landed.

## Where the work landed

- Commit `390079a` — "refactor(formats): re-arrange all variants per BPW — exact BPW, K_L/M/H, GRP exact, MXQ_GRP only"
- Commit `a8b4a26` — "chore: propagate MXQ/K/GRP rename to all consumers — remove TWI, use dot BPW"

## Verbatim key exchanges (from the extract)

- User: "SESSION-00" (the session was explicitly named to be the numbering anchor for future OpenCode sessions)
- The session worked through: which files carry _GRP, how half-BPW families display, whether MXQ keeps its plain tier (verdict: MXQ_GRP only per the re-arrangement commit), and the slot-19 identity (Q4_K_L — "not a gap").

## Reusable extraction methodology (proven twice)

OpenCode and ZCode both store conversations in SQLite with JSON columns. The working recipe (Node v24 built-in `node:sqlite`):

```js
const { DatabaseSync } = require('node:sqlite');
const db = new DatabaseSync('<db path>', { readOnly: true });
// 1) discover: SELECT name FROM sqlite_master WHERE type='table'
// 2) messages: SELECT id, time_created, data FROM message WHERE session_id=? ORDER BY time_created
//    data JSON has role (+ model/agent metadata for OpenCode)
// 3) parts: OpenCode: SELECT data FROM part WHERE message_id=? ORDER BY time_created
//           ZCode:    SELECT data, sequence FROM part WHERE message_id=? ORDER BY sequence, time_created
//    part JSON types: text (use d.text), reasoning (cap length), tool/step-start (collect names only)
// 4) join text + [tools: ...] lines into a per-message block; skip empty
```

- OpenCode DB: `~/.local/share/opencode/opencode.db` — sessions table has id/title/directory/time_updated; find InNova sessions by `directory` = `c:\Users\thaku\Downloads\InNova`.
- ZCode DB: `~/.zcode/cli/db/db.sqlite` — sessions carry `project_id='proj_c-users-thaku-downloads-innova'`; 29 sessions, 1013 messages total.
- ZCode UI sessions (Electron) additionally live under `~/AppData/Roaming/ZCode/` (browser-profile shape) — not needed while the CLI DB holds the history.

## Relation to other memory files

- Naming law details now live in `02-iron-rules.md` Rule 12.
- The audit trail that later verified BPW honesty is in `06-zcode-sessions-history.md` (Aug 25-26).
- Current naming ground truth in `03-project-map.md` §3.1.
