---
name: innova-workflow-constraints
description: Environment constraints when working the InNova repo — sub-agents
  blocked, Mimosa hook forbids bash source-writes, flaky parallel MSVC links
metadata:
  node_type: memory
  type: project
  originSessionId: sess_fca3fa84-5457-46a9-abae-ead1c93e5dcb
---

Discovered during the 2026-08-26 InNova audit session:

1. **Sub-agents cannot be used** in this setup — even a single Agent launch fails with `model concurrency limit exceeded` (three independent attempts, one also got cancelled). Do audits/work solo with batched tool calls instead of burning turns retrying agents.
2. **Mimosa security hook blocks Bash-based writes to source files** (`*.cpp`, `*.h`, shell scripts): sed/in-place edits via Bash get rejected ("Bash 直接写源码… 会绕过安全扫描"). All source modifications MUST go through the Edit/Write tools. Bulk renames = per-file Edit with replace_all (Read each file first).
3. **Windows parallel MSVC builds flaky above -j3**: rc.exe `LNK1327` and link.exe `0xC0000142` (STATUS_DLL_INIT_FAILED resource exhaustion) appeared at `-j8`. Use `--parallel 3`.
4. Edit-tool read-state expires after context summarization — a file you read long ago may throw "File has not been read yet"; just re-Read (a tiny limit=2 read suffices) before the Edit.
5. Build/test artifacts land in `build/Debug` / `build/Release` / `build/tests/<cfg>`; bench writes `bench_format_comparison.csv` to its CWD — copy a snapshot before runs you want to diff against.

**Why:** each of these burned session time when rediscovered. **How to apply:** apply them proactively in any future InNova session. Related: [[innova-project-map]].
