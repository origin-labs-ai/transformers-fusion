# TransCender Production Scorecard — ledger-derived composite (2026-09-14)

> Single source of truth for the "how production-ready?" number.
> Rule: the % below is recomputable from the listed evidence. Any other file
> that quotes readiness MUST point here, never copy the number.
> Scope: engine build/test/hardening only. Strategy numbers (P59–P73) stay
> UNVERIFIED per `docs/STRATEGY.md`.

## Headline

**Engine production readiness: ~95.0–96.7% (ledger-derived composite, HEAD `8f06002`).**

- Strict (claims counted VERIFIED/FIXED only): **95.0%**
- Engine-scope (SCOPED rows accepted with their stated scope): **96.7%**
- Status of this number: **LEDGER-DERIVED, this-host re-run owed**
  (`build-final/` + `build-head/` absent in this workspace; working tree clean).

## Methodology (equal weights, 5 legs)

```
composite = (tests + claims + CI + ASan + BPW) / 5
```

| Leg | Score | Evidence |
|---|---|---|
| Tests | 100% (72/72) | `tests/CMakeLists.txt`: 71× `add_quant_test_full` + WIN32-only `test_gpu` = 72; `research/claim_ledger.md` rounds 13–19 (HEAD `8f06002`): Windows MSVC Release 72/72, Linux GCC 71/71 |
| Claims (C-01..C-25) | strict 75.0% (18/24) · engine-scope 83.3% (20/24) | C-16 RETIRED (owner purge 2026-09-07, out of count) → 24 active. VERIFIED/FIXED 18: C-01, 02, 04, 05, 07, 08, 09, 10, 11, 12, 13, 14, 17, 18, 19, 21, 24, 25. SCOPED 2: C-22 (single-host threading tested; multi-process/NCCL out of scope), C-23 (pipelines real, generative quality placeholder). PARTIAL/narrowed 4: C-03 (64% number unsourced), C-06 (non-host backends un-runnable here), C-15 (realistic-draft numbers), C-20 (NVMe tiering) |
| CI | 100% (3/3 workflows, 7/7 Full legs) | `research/claim_ledger.md` round 19: `CI Full` SUCCESS (all 7 legs incl. GCC-13 ASAN), `CI Build` 3/3, `CI macOS` FULL SUCCESS |
| Sanitizers | 100% (62/62 Quick + 7/7 heavies, clean) | `research/claim_ledger.md` rounds 10–11, 13: `-fsanitize=address,undefined -fno-sanitize-recover=all`, RoPE clamp + link-dep fixes landed |
| BPW ironclad | 100% (105/105 ok, 0 violations) | `include/quant/types.h:67` `FORMAT_COUNT=105`; `research/claim_ledger.md` round 4 (A-01 re-probe 2026-09-13): `test_format_audit` 0 violations |

Composite: `(100 + 100 + 100 + 100 + 83.3)/5 = 96.7%`;
strict: `(100 + 100 + 100 + 100 + 75)/5 = 95.0%`.

## Size truth (measured 2026-09-14, this host)

- Code tree (`src/` + `include/` + `engines/` + `tests/` + `bench/` + `tools/`,
  `.cpp/.h`): **452 files, 139,304 lines**
  (code 109,254 + blank 15,195 + comment/preproc 14,855).
- Plus `sops/` + `scripts/` + `cmake/` + CMakeLists: 17 files, 3,239 lines.
- **Engine code ≈ 140K lines — the "180K+ LOC" claim is NOT supported.**
  180K needs docs/research/memory counted in; that is repo size, not code LOC.
- Biggest files (split candidates, L078): `src/backend/backend.cpp` 3,049 ·
  `src/codec/block_codec.cpp` 2,818 · `src/agi/agi_flywheel.cpp` 2,553.
- Version truth: `CMakeLists.txt:3` `1.1.0` · `include/quant/types.h:67` 105 formats.

## Openly owed (not hidden, not in the %)

1. CUDA/Metal/SYCL/HIP runtime (no device on dev hosts).
2. Multi-process/NCCL (out of scope; single-host threading tested).
3. End-task quality benches (synthetic MSE only).
4. C-15 realistic-draft acceptance numbers (no trained draft + corpus in-tree).
5. P59–P73 strategy numbers (UNVERIFIED-direction, `docs/STRATEGY.md`).
6. This-host fresh full-suite re-run (build dirs absent here; HEAD proof cited).

## How to re-verify (anyone, any machine)

1. `cmake -B build-verify` + full Release build (0 errors expected).
2. `ctest --test-dir build-verify -C Release --timeout 600` → expect 72/72.
3. `test_format_audit` → expect `BPW violations: 0` (105 formats).
4. Linux: same minus WIN32-only `test_gpu` → expect 71/71.
5. Recompute §Methodology from `research/claim_ledger.md` last round + this file.
6. If any leg disagrees, THIS FILE gets corrected — never a bare number elsewhere.

## History

- 2026-09-14: created from ledger rounds 1–19 audit (HEAD `8f06002`).
  Prior deleted-thread claims (94%, 149/256 rounds, 1000+ verifications) carry
  NO in-repo evidence and are NOT used here (owner statement recorded as
  UNVERIFIED in chat only).
- Recompute on every production-round close; supersede, never overwrite.

*Last verified against: git HEAD `8f06002`, working tree clean (2026-09-14).*
