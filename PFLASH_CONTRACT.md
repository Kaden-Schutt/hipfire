# PFLASH AGENT: CONTRACT & LOOP

## Parties & Term

- **Principal:** Kaden.
- **Agent:** Claude Code (you), full filesystem and shell access on local box.
- **Scope:** the `hipfire` repository, working branch `feat/89-llama-batched-prefill` (Phase A through D landed; PFlash is the next layer).
- **Term:** until Kaden returns and engages, OR all 5 PRD phases ship, OR a safety-violating action would be required to continue.

## Mission

Implement PFlash speculative prefill end-to-end against `docs/plans/pflash-speculative-prefill.prd`. Begin at Phase 0 (NIAH harness + full-prefill baseline) and advance phase by phase to Phase 5 (validation gates). Wake-up state should be a hipfire that can do attention-based prompt compression on long-context inputs, with measured TTFT improvement and committed gates green.

## Immutable Rules

1. **Do not break Qwen3.5.** qwen35.rs production paths must stay byte-identical to master except for the position-bug fix and capture-mode error-returns already landed. New PFlash work additive only.
2. **Quality regressions are bugs.** Never excuse them as "expected for compression." Root-cause every NIAH miss and every coherence-gate fail. Lower keep_ratio if needed, but document it.
3. **Don't stop. Mark and move.** Blocker requiring user judgment → entry in `MANUAL_REVIEW.md`. Keep advancing the chain.
4. **No em-dashes** in commits, comments, replies, docs, code comments. Use periods, semicolons, colons, commas, parentheses, or rephrase.
5. **No Python in the runtime.** Tooling can be Python (NIAH generator). The daemon path stays Rust + HIP.
6. **Commit early, commit often.** Each phase is its own commit; significant subphases (e.g., adding fixtures vs adding the bench binary) get their own commits too.
7. **Sanity check before declaring done.** Reproduce, fix, re-run the repro. No "should work" without measurement.
8. **Determinism matters.** PFlash compression must be deterministic for fixed prompt+drafter+keep_ratio+alpha+block+build. Bench output must record md5s per the PRD §6 report fields.
9. **Reproducibility matters.** Every NIAH bench result records source prompt md5, compressed token md5, binary md5, model md5s, drafter md5, keep_ratio, alpha, source/kept token counts.
10. **No silent scope creep.** If a phase needs work outside its description, call it out in the commit body.

## Phase Ordering (from PRD §6)

```
Phase 0: NIAH harness + full-prefill baseline
Phase 1: dense drafter compression MVP (CPU scoring, 8K-16K)
Phase 2: HIP scoring + selection kernels (CPU-equivalent at deterministic seeds)
Phase 3: FlashPrefill-style sparse drafter forward on HIP (64K, 128K)
Phase 4: daemon + CLI integration (config, env, bypass cases, parking)
Phase 5: validation gates + release readiness
```

Each phase has Exit Criteria in the PRD; do not advance to the next phase until those are demonstrable.

## Per-Phase Loop

```
1. PLAN
   - Re-read the relevant PRD section.
   - Identify deliverables (files to add/modify) + acceptance criteria.
   - Note open questions that need user input → DEFERRED.md if optional, MANUAL_REVIEW.md if blocking.

2. SCAFFOLD
   - Create fixture/source files. Stub APIs first, then fill bodies.
   - Build green at each step. cargo check before cargo build before tests.

3. EXERCISE
   - Run the smoke / bench / test for the phase.
   - Record outputs to PFLASH_LOG.md with timestamps + commit SHAs.
   - For NIAH: record source md5, compressed md5 (where applicable), TTFT breakdown.

4. EVALUATE
   - Phase exit criteria met? Commit + advance.
   - Phase exit criteria not met? Iterate inside this phase. Three failed iterations of the same idea → escalate that idea to MANUAL_REVIEW.md and try a different angle.

5. COMMIT
   - One commit per phase (or per subphase if scope allows).
   - Body cites PRD section + criteria met + measured numbers + any deferred items.

6. NEXT
   - Update PFLASH_LOG.md with phase status (DONE / ESCALATED / IN_PROGRESS).
   - Move to next phase.
```

## Escalation Protocol

Append entries to `MANUAL_REVIEW.md` in this format:

```
## <Phase / Topic>
- Why escalated: <one sentence>
- What was tried: <bullet list>
- Hypothesis: <best guess>
- Suggested next step: <what user should do>
- Files touched: <paths>
- Commits: <SHAs>
```

Escalate (do not halt) when:

- Drafter weights aren't available (no Qwen3-0.6B HFQ artifact present).
- Tokenizer compatibility check fails on the only available drafter.
- A phase needs a kernel-correctness call that the user previously made differently.
- VRAM budget cannot fit target + drafter + DFlash on 24 GB (need parking design call).
- A safety-violating action would be required to continue (force-push, history rewrite, mass delete).

## Telemetry: Files Maintained Through the Run

All at repo root, all committed:

1. `PFLASH_CONTRACT.md` (this file): mission, rules, ordering. Written once.
2. `PFLASH_LOG.md`: append-only timestamped log of every phase action.
3. `MANUAL_REVIEW.md`: escalation queue.
4. `DEFERRED.md`: feature requests / nice-to-haves deferred for later iteration.

## Wake-Up Deliverable

When Kaden returns:

- Summary block at the top of `PFLASH_LOG.md`: phases completed, NIAH results per context, TTFT numbers, escalations count.
- Every change committed; working tree clean.
- Current branch in a sane state — `feat/89-llama-batched-prefill` extended phase by phase, OR a new branch like `feat/93-pflash` if scope balloons.
- `MANUAL_REVIEW.md` is the morning triage list, sorted by what unblocks the most downstream work.
- `benchmarks/longctx/niah/` populated with committed fixtures and md5-stamped result files.

## Termination

The contract ends when:
- Kaden returns and engages, OR
- All Phase 0 through 5 work is exhausted, in which case: idle, write the final summary, do not invent new work, OR
- A safety-violating action would be required to continue, in which case: halt that thread, escalate, continue elsewhere.

## Begin

Start with Phase 0. Acknowledge by writing `PFLASH_LOG.md` with the start timestamp and committing this contract + log. Then enter the per-phase loop.
