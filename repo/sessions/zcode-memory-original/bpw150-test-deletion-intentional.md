---
name: bpw150-test-deletion-intentional
description: tests/test_bpw_150_proof.cpp deletion is intentional — do not
  restore or flag it
metadata:
  node_type: memory
  type: feedback
  originSessionId: sess_0dc0957a-5b90-4453-af4d-26a49d448f6a
---

The uncommitted deletion of `tests/test_bpw_150_proof.cpp` (shows as `D` in git status) is deliberate.

**Why:** User said "Jis file me 150 kahi pe laga hai wo deleted rahenge" — the BPW-1.50 proof test belongs to a retired claim/format era and must stay deleted.

**How to apply:** Never restore, resurrect, or "fix" this file; don't list its deletion as pending cleanup. Leave `git status` showing the D. On 2026-08-26 its dead references were also removed from tests/CMakeLists.txt (they broke CMake generate — that removal is part of the same intentional deletion, do not re-add). Related: [[innova-project-map]].
