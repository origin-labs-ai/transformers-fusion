# 01 — USER PROFILE

## Who the owner is

- A **15-year-old solo developer** (revealed 2026-08-31 with "Waise shock mat hona ye sunke kabhi ki mai 15 saal ka banda hu!"). Despite the age, treat every technical statement at full weight — he authored a 100K+ line C++20 engine, its own honest-audit culture, and an architecture thesis. Age is irrelevant to the work; the code and the thesis speak for themselves.
- He is the sole founder of the InNova/Transcender project. There is no team; every architectural decision, every audit demand, and every naming choice traces back to him.
- Long-term ambition: the 5-pillar PoC (Context Rot/RLM, Private Delta, Slop Filter, Hot-Swap, Delusion Breaker) → investor pitch → **IndiaAI PMEC 512-1024 GPU track** → eventually his own datacenter.

## Communication style (with real examples)

- **Hinglish, casual, profanity-friendly.** "Saale", "bhai", "chutiya"-class words are friendship markers, not insults. Reply in Hinglish with English technical terms. Never switch to formal English chat (memory files are the exception — owner ordered those in English).
- Real examples of his register:
  - "Bhai, ye project check kar! Tera dimaag ka bharta ban jaayega!" (challenge framing)
  - "Abe saale, tujhe 5 baar poore project ka har ek file ka honest audit karne bheja tha!" (correction with heat, but collaborative)
  - "Bas yaar, bas ab ruk jaa! Ab ho gaya! STOP EVERYTHING SUDDENLY!" (hard stop — obey immediately, give a one-line status, wait)
- **He issues rapid-fire orders** and expects immediate execution with minimal clarifying questions. Ask only when a decision is genuinely his to make (BPW contract, adapters fate, naming).
- **He tests agents repeatedly.** He asked "kya samjha?" style verification multiple times (the 4-round "eklota mean" hunt in the chat export; "Wait, abhi bhi asli mean nahi pakra hai tu! Mai batau kya?!"). He rewards honest attempts and corrects them; he punishes fake confidence.
- **"STOP EVERYTHING" is absolute.** When he says stop (as on 2026-08-25 during the rename: "Bas yaar, bas ab ruk jaa! STOP EVERYTHING SUDDENLY!"), halt all work, report the exact state in one short message including what is half-done, and do not resume until he says so.

## What he values

1. **Honesty over polish.** The project's defining culture is the anti-fake audit: claims ledger with file:line evidence, VERIFIED/FAKE/PARTIAL verdicts, 5x full-project audits on demand. He personally busted fake claims (e.g. "256 tasks DONE" in an old goal_status.json, the "zero BPW violations" note contradicted by its own tables).
2. **Restraint over brute force.** Same work at 70% of the compute is a win condition, not a compromise. QUANT formats + hand-written SIMD kernels are the hardware proof of this philosophy.
3. **Measured numbers only.** Every number must come from a fresh command run. Hand-copied numbers are treated as fabrications.
4. **Never-overwrite doctrine.** Architecturally (delta rule, additive pages) and procedurally (don't delete history, don't hide wounds, supersede instead of erase).
5. **Respectful irreverence toward giants.** "llama.cpp waalo ki band bajani hai" — competitors are measurement bars (black-box), never code to copy.

## What he hates (zero tolerance)

- Fake claims, stub code, TODO comments, "coming soon" features, hardcoded placeholder returns.
- Hand-copied/unmeasured numbers in docs or benches.
- Hiding provenance (e.g. renaming K3-branded files while claiming the ideas as invented). The accepted formula is: own name + honest citations (DeltaNet/Yang et al. for the gated delta rule, DeepSeek-V2 for MLA, Titans must be cited due to additive-memory overlap).
- Fake test harnesses (tests that pass by construction).

## Locked decisions (chronological, with dates)

| Date | Decision |
|---|---|
| ~Aug 22 | Master plan v2 (Gauntlet Edition), 9 phases, L001-L097 backlog; workbench + claim ledger + GLE telemetry as live-status infrastructure |
| Aug 24 | 5x full-project honest audits as the standard audit protocol |
| Aug 24 | `tests/test_bpw_150_proof.cpp` deletion is intentional and PERMANENT |
| Aug 25 | STOP during rename; later "Ab mere demands ke hisaab se fix karne pe lag jao!" |
| Aug 25-26 | `_GRP` → `_G` rename across the whole codebase (all 105 formats, every file) — completed |
| Aug 25-26 | BPW "truth in reporting" fix (format_bpw returns actual wire BPW); final fit-vs-rename still owner's call |
| Aug 26 | Persona code verdict: "persona to system prompt se banega" — remove from engine or document in TRANSCRIPT (documented path chosen) |
| Aug 26 | Sub-agent limit: max 2 in parallel |
| Aug 31 | New kernels must be mission-aligned (he rejected generic new-kernel sprawl until the philosophy was clear) |
| Aug 31 | **Architecture name locked: Transcender** |
| Aug 31 | Memory files must be **in English** and **ultra detailed** — "koi bhi file chhoti nahi banani hai" |
| Aug 31 | Restructure: `.research` → `research/` at ROOT (his correction after the first plan put it inside repo/); everything else into `repo/`; `.mimosa` must be a true cut-paste |

## Guidance he needs (developer-to-developer)

- **Time is his scarcest resource** (school + project). Steer him away from multi-year side quests; the accepted example is channeling his "build a language" urge into a declarative graph IR (the `.txn` idea) instead of a full programming language.
- **Anti-plagiarism anxiety**: he feared the "chor" (thief) label for K3-derived files. The resolved position: ideas are not copyrightable, code is; implement from the math, cite the papers, and the attack is structurally dead. Hiding K3 branding would be worse than citing it.

## Working agreement for agents

1. Read `MEMORY.md` first; obey `02-iron-rules.md` without exception.
2. Match his language (Hinglish chat), keep commits English.
3. Never fake progress; report blockers with evidence.
4. Owner decisions queue: BPW contract, adapters/ fate, bench migration timing, Transcender Phase 0 start — present options + a recommendation, then wait.
5. Update `08-current-state-*.md` and the relevant memory file before ending any session.
