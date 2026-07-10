# autoresearch loop — operations runbook

How to drive the loop through the `ar` CLI and the watcher daemon. Architecture
is in `architecture.md`; every config key is in `config-reference.md`.

> **The contract: agents touch `ar`, never a raw script.** ssh in → `ar bod` /
> `ar why` to read state → author a lever → `ar certify` to submit. The bounds
> are mechanical (exit 3), not prompt-discipline.

## 0. Prerequisites

- **Python 3.11+ stdlib only** (`tomllib`, `sqlite3`, `fcntl`, `subprocess`) —
  no third-party deps. `pytest` only for the test suite.
- A per-arch config: `autoresearch/config/loop_<arch>.toml` (ships:
  `loop_gfx1201.toml`) and its round prompt `autoresearch/config/prompt_<arch>.md`.
- **Fleet card→dev mapping verified first.** `rocminfo` lies about device order;
  confirm each worker's `{card, dev}` against the daemon-reported arch before
  launching (a worker on the wrong dev grades the wrong GPU).
- **All target GPUs at AUTO clock** (`power_dpm_force_performance_level=auto`).
  `high` underclocks the R9700/gfx1201 ~13% and corrupts perf deltas.

Path resolution (all overridable; nothing hardcoded):

| Thing | Flag | Env | Default |
|---|---|---|---|
| repo root | — | `$AR_REPO` | three levels up from `cli.py` |
| store | `--db` | `$AR_DB` | `<repo>/autoresearch/db/ar.db` (gitignored) |
| ledger dir | `--ledger` (ingest) | `$AR_LEDGER` | `<repo>/autoresearch/ledger` |
| BOD glob | `--bod` (ingest) | `$AR_BOD_GLOB` | `<repo>/autoresearch/state/bod_*.json` |

## 1. Roles + exit codes

`ar --role {operator,agent} <verb> …`. Default role is `operator`.

- **operator** (Claude): `start · stop · status · why · bod · ingest · fold ·
  rollover · config · certify`
- **agent** (codex/grok): `why · status · bod · certify`

Exit codes: `0` OK · `2` usage/empty (e.g. `ar bod` with no census) · `3`
**mechanical refusal** (role-forbidden verb, or a `certify` on an
exhausted/off-target/over-budget kernel).

## 2. Read verbs (agent + operator)

```bash
# Rebuild the index from the git-tracked ledger + BOD snapshots (idempotent).
ar ingest

# Ranked candidate kernels for an arch: wall%, L2%, roofline lens, tried/wins,
# dead-streak, and OPEN / EXHAUSTED / below-threshold status.
ar bod --arch gfx1201
ar bod --arch gfx1201 --json          # machine-readable

# What was already tried on a kernel (levers, verdicts, tok/dur deltas) — read
# this BEFORE authoring a lever so you don't re-derive a dead one.
ar why fused_qkvza_hfq4g256 --arch gfx1201

# Run health: banked attempts/wins + running loops (calls/budget, ttl_left).
ar status
```

## 3. Submitting a candidate (`ar certify`) — the leash

```bash
ar --role agent certify --arch gfx1201 --kernel fused_qkvza_hfq4g256 \
   --lever "row-tile NUM_ROWS=2 X-reuse" --variant /path/to/variant.hip
```

`certify` is a **bounds gate**, not the grader. It refuses (exit 3) if the kernel
is `EXHAUSTED` (`k` consecutive deads), off-target (`wall% < cand_wall`), or the
arch's running loop is over `call_budget` / past `wall_ttl_s`; otherwise it
accepts (exit 0) and bumps the run's call counter. On accept, the driver loop's
`certify()` orchestrator runs the real A/B (parity → conjunctive perf →
coherence) over the `LiveServeRunner`, writes the self-describing row to the
ledger, and `ar ingest` re-indexes it.

## 4. Operator verbs

```bash
# Launch the config's workers as detached loops; records each as a bounded run.
# A worker without an advancing .aw/sw_card<N> worktree is SKIPPED (bash parity).
ar start --config autoresearch/config/loop_gfx1201.toml

# Stop running loops (all, or one --run <id>) — flips the ar.db run flag.
ar stop
ar stop --run gfx1201-w0-1720000000

# Fold a certified WIN into the shared baseline via update-ref CAS (records the
# prior SHA → reversible). Preview first with --dry-run.
ar fold --ref loop/gfx1201 --sha <win_sha> --dry-run
ar fold --ref loop/gfx1201 --sha <win_sha>

# Re-census + re-ingest after a baseline advance / exhaustion.
ar rollover --reason advance --dry-run
ar rollover --reason advance

# Print the resolved loop config (TOML → LoopConfig).
ar config --arch gfx1201 --json
```

## 5. The watcher daemon (`ar.watcher.Watcher`)

One persistent process per box owns run lifecycle and **auto-enforces**
fold/rollover under guardrails:

- **`track` / `reap`** — records runs in `ar.db.runs`; reaps dead pids.
- **`enforce`** — auto-stops any run over its `call_budget` or past its
  `wall_ttl_s` (`_leash_reason`), and folds certified WINs into the shared
  baseline.
- **Guardrails (the whole point):** every enforced mutation is (1) **dry-run
  previewable** first, (2) **git-reversible** — the prior SHA is recorded before
  a fold/rollover, (3) leashed by budget/TTL/exhaustion.
- **Human gates preserved:** master-push and default-flips are **never**
  automated. `Watcher` has no `push_master` method by construction — the watcher
  stages and notifies; a human lands the master push / default flip.

## 6. First run — Sol/Terra/Luna lever-finding eval (Phase-5 acceptance)

The headline first run: three heterogeneous agents as unbiased lever-finders on
the shared MoE-GEMV kernels, `mq4r` / gfx1201, on hiptrx (4× R9700).

**Runbook (on hiptrx):**

```bash
# 1. Confirm the 4 R9700 are at AUTO clock and card→dev mapping is real
#    (rocminfo lies — verify via the daemon-reported arch).
# 2. Rebuild the index + confirm the fd6deaa9 BOD is present.
ar ingest
ar bod --arch gfx1201

# 3. Launch the config's 3 workers (Luna@max card1, Terra@xhigh card2,
#    Sol@medium card3; model=mq4r; k_exhaust=5). Either via the CLI:
ar start --config autoresearch/config/loop_gfx1201.toml
#    …or the swarm module directly:
python3 -c "from autoresearch.ar.config import load_config; \
            from autoresearch.ar.swarm import launch; \
            launch(load_config('autoresearch/config/loop_gfx1201.toml'), '.')"
```

**Acceptance checks:** all 3 workers produce ledger rows that are self-describing
(`measurement_hash` present, `gpu_arch=gfx1201`, `model=…mq4r`) carrying **both**
`tok_delta_pct` and `dur_delta_pct`; any WIN folds into the shared baseline via
CAS; the cross-arch guard is clean (no worker's gfx1201 edit perturbs another
arch's device TU).

**Results:** _pending the live hiptrx run._ This branch lands the harness
(Phases 1–8, no-GPU-tested); the live Sol/Terra/Luna eval is a GPU-box run whose
ledger rows + tok/s·τ numbers are appended here once measured (byte-identical
prompt, md5 `d97ec9d3…` recorded per row — never fabricate a perf number).

## 7. KEPT bash scripts — DO NOT treat as migrated

Two bash predecessors were **deliberately retained** because their Codex
cross-model parity verifications came back **REFUTED** — the Python ports are not
yet proven byte/semantics-equivalent, so the bash stays as the ground-truth
oracle until a follow-up verify passes.

> **// KEPT: `autoresearch/harness/ab_certify_v2p.sh` — python certify parity
> UNVERIFIED, see Phase-2 verify.**
> The certify-verdict parity check (`test_certify_parity.py` vs the captured
> `ab_certify_v2p.sh` rows) was **REFUTED** by the Codex seam. Until it passes,
> `ab_certify_v2p.sh` is the authoritative certify oracle. Do **not** delete it,
> and treat `ar certify`'s verdict as advisory-only for a bankable WIN until the
> parity seam is re-verified against this script.

> **// KEPT: `autoresearch/harness/v2/rollover_v2.sh` — git-CAS/fold parity
> UNVERIFIED, see Phase-2 verify.**
> The git-CAS/fold semantics verification was **REFUTED** by the Codex seam.
> Until it passes, `rollover_v2.sh` is the authoritative fold/rollover oracle.
> Do **not** delete it. Prefer it (or a `--dry-run` `ar fold` cross-checked
> against it) for any real baseline advance until `gitpilot.update_ref_cas` +
> `Watcher.enforce_fold` are proven equivalent.

Every *other* loop `.sh` (36 of the 38) is fully ported and deleted; the probe
`.hip` sources moved to `autoresearch/probes/`.

## 8. No-GPU verification

```bash
# The decision suite — pure stdlib + pytest, GPU seams mocked. Run from repo root.
pytest autoresearch/ar/tests/ -q
```

All modules except `census.run_census`, `serve_runner.LiveServeRunner`'s live
methods, and the `Watcher`'s real git mutations are exercised here; the GPU
seams are dependency-injected and mocked.

## 9. Housekeeping follow-ups (tracked)

- **`probes.py` runner** over `autoresearch/probes/*.hip` (`cu_scale_probe`,
  `gemv_occ_probe`, `gemv_roofline_probe`, `mall_probe`) is not yet built — the
  probe sources are preserved but currently have no thin Python runner.
- **Stray hiptrx worktrees** — reconcile the ~12 detached worktrees
  (`git worktree prune` + a documented per-card `.aw/sw_card<N>` policy).
- **Wire `autoresearch/ar/tests/` into `scripts/no-gpu-ci.sh`** so the suite runs
  in CI alongside `pytest tests scripts/test_astrea.py`.
