# autoresearch — arch-gated RDNA kernel autoresearch

The **fixed-eval loop** that turns the bill-of-decode (which kernel is bound how,
per arch) into certified kernel wins. Karpathy single-experiment + fixed-eval
discipline: mutate one kernel, fight the champion under a fixed A/B eval, keep the
winner, log every round (win, loss, or noise). **The ledger IS the research.**

As of the 2026-07-09 migration, the loop is **one Python package**
(`autoresearch/ar/`) driven through the role-scoped `ar` CLI — the 38-script bash
harness is gone. Reach for the docs:

- **`docs/autoresearch/architecture.md`** — module map + data flow + the certify
  gate (parity → conjunctive perf → coherence) + guards.
- **`docs/autoresearch/operations.md`** — the `ar` runbook (ingest / bod / why /
  status / certify / start / stop / fold / rollover), the watcher daemon, and the
  first-run Sol/Terra/Luna eval.
- **`docs/autoresearch/config-reference.md`** — every per-arch TOML key + default.

## The loop (one line each)

1. **BOD census** (`ar.census`) names the lever per arch/kernel from a
   `profile_standard` rocprof pass → `state/bod_<arch>.json`.
2. **Candidate select** (`ar.candidates`) keeps `wall% >= cand_wall` kernels,
   marks `OPEN` / `EXHAUSTED` (5 consecutive deads; a WIN resets).
3. **Author** a variant `.hip` (an agent round via `ar.agent_exec`).
4. **Certify** (`ar.certify`) — the fixed eval: **parity** (token-id exact) →
   **conjunctive perf** (`kernel_decode_tok_s` UP *and* rocprof duration DOWN,
   each Mann-Whitney U; either alone ⇒ DEAD) → **coherence** (real serve,
   McNemar). Guards: symbol→file resolve + cross-arch preprocessor-invariance.
5. **Ledger** every A/B → `ledger/*.jsonl`, self-describing rows keyed by
   `measurement_hash`. A WIN folds into `loop/<arch>` via `update-ref` CAS.

## Layout

```
ar/         the package (config, db, gitpilot, census, candidates, driver,
            swarm, watcher, agent_exec, cli, certify/)
config/     loop_<arch>.toml + prompt_<arch>.md
db/         schema.sql (+ ar.db, gitignored — rebuilt by `ar ingest`)
ledger/     *.jsonl history (git-tracked, the source of truth)
levers/     gfx1100.md gfx1151.md gfx1201.md
probes/     *.hip probe sources
variants/   winning kernel sources
harness/    KEPT bash oracles only (ab_certify_v2p.sh, v2/rollover_v2.sh — see
            operations.md §7: their Python parity is UNVERIFIED/REFUTED)
```

## Quick start

```bash
ar ingest                       # rebuild the index from the ledger + BOD
ar bod --arch gfx1201           # ranked candidates + coverage
ar why <kernel> --arch gfx1201  # what was already tried (read before authoring)
pytest autoresearch/ar/tests/ -q   # the no-GPU decision suite
```
