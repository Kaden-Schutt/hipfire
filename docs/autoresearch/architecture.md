# autoresearch loop — architecture

The autoresearch loop turns the **bill-of-decode** (which kernel is bound how, per
arch) into **certified kernel wins**. It is the Karpathy single-experiment +
fixed-eval discipline applied to HIP kernels: census the baseline, pick a
candidate kernel, let an agent author a variant, grade it under a fixed A/B eval,
keep only real wins, and log *every* round (win, loss, or noise) to an
append-only ledger. **The ledger IS the research.**

This document is the module map + data flow for the Python package
(`autoresearch/ar/`) that replaced the 38-script bash harness. For the *why* of
the migration see `2026-07-09-python-migration-design.md`; for how to run it see
`operations.md`; for config keys see `config-reference.md`.

## Design principle

One package, thin single-responsibility modules, **dependency-injected GPU
seams** so every decision (verdict, candidate selection, exhaustion, config,
git-CAS) is no-GPU unit-testable. Git is piloted via `subprocess` (not
GitPython/pygit2) to preserve the death-safe `flock` + `update-ref` compare-and-swap
semantics the bash harness relied on. Nothing hardcodes arch/model/card — it all
comes from a per-arch TOML.

## Data flow

```
                       ┌── cli.py (`ar`) ───────────────┐   operator + agent roles
                       │                                │
   config.toml ─► config.py ─► watcher.py (daemon) ─────┤
                       │            │                   │
                       │            ├─► swarm.py ─► driver.py (×N workers)
                       │            │                        │
                       │            │        ┌───────────────┤
                       │            │   census.py       candidates.py   agent_exec.py
                       │            │   (→ bod json)    (OPEN/EXHAUSTED)  (codex/grok round)
                       │            │                        │
                       │            │                  certify/orchestrator.py
                       │            │                   ├ resolve.py    (symbol→file, DeadFile/NoOp)
                       │            │                   ├ serve_runner.py (parity/coherence/perf gens)
                       │            │                   ├ verdict.py / perf.py / coherence.py
                       │            │                   └ cross_arch.py (preprocessor-invariance guard)
                       │            │                        │  WIN
                       │            └─► gitpilot.py ◄─────────┘  (update-ref CAS advance, fold, rollover)
                       │                     │
                       └──────────► db.py  ◄─┘   (ar.db: attempts, bod, runs)  + ledger/*.jsonl
```

## Module map

| Module | Responsibility | Ports / absorbs |
|---|---|---|
| `config.py` | Per-arch TOML → `LoopConfig`/`WorkerCfg`/`Bounds` dataclasses; spec-§6 defaults | (new, stdlib `tomllib`) |
| `db.py` | SQLite index (`attempts`/`bod`/`runs`); idempotent ledger `ingest`; `wins`/`best`/`history`/`kernel_stats` | `oracle_db.py` + `hipfire_ar.py` schema |
| `gitpilot.py` | `subprocess` git: `gpu_lock` (flock-on-fd), `update_ref_cas`, `current_sha`, `show_file`, `worktree_*` | git bits of `ab_certify_v2p.sh` + `scripts/gpu-lock.sh` |
| `census.py` | BOD census: `run_census` (GPU seam) + `parse_rocprof` (pure CSV→rows) + `write_bod` | `oracle_profile.sh` |
| `candidates.py` | `Candidate`; `select`; `is_exhausted`; `gen_digest`; `update_exhaustion` + per-round primitives | `exhaustion.py` + `v2/{check_exhausted,gen_digest,update_exhaustion}.py` |
| `agent_exec.py` | One autonomous coding round: `build_argv` + `run_round` (per-worker model+effort, codex/grok) | `agent_exec.sh` |
| `driver.py` | Self-exhausting per-worker loop `run_loop` with injectable hooks | `v2/driver_v3.sh` |
| `swarm.py` | Config-driven parallel launcher: `plan_workers` (pure) + `launch` (spawns detached) | `swarm_explore.sh` (kills the `sed`-munge) |
| `watcher.py` | `RunStore` + `Watcher` (track/reap/enforce; guardrailed auto-fold/rollover; leashes) | `hipfire_ar.py` supervisor + `v2/rollover_v2.sh` |
| `cli.py` | Role-scoped `ar` entrypoint (operator/agent verbs; mechanical bounds) | `hipfire_ar.py` CLI surface |
| `certify/verdict.py` | The three decision arms + verdict combiner; **conjunctive perf gate** | `certify_verdict.py` + `ab_certify_serve.py` |
| `certify/perf.py` | Perf statistics: `mwu` (one-sided Mann-Whitney U), `median_delta_pct`, `clock_void`, dominance | `perf.py` |
| `certify/coherence.py` | `detect_attractor`, `run_validators`, `mcnemar_worse` | `coherence_arm.py` |
| `certify/orchestrator.py` | `ServeRunner` (abc) + `certify(...)` → the self-describing ledger row | `ab_certify_serve.py` |
| `certify/serve_runner.py` | `LiveServeRunner` GPU adapter (serve_harness tok/s + rocprof duration) | `harness/serve_runner.py` |
| `certify/resolve.py` | `resolve_kernel_file` (symbol→`.hip`); `DeadFile`/`NoOp` guards | symbol→file guard of `ab_certify_v2p.sh` |
| `certify/cross_arch.py` | `check_cross_arch` (a gfx1201 edit must not change any other arch's device TU) | `cross_arch_guard.sh` |

Probe `.hip` sources live in `autoresearch/probes/` (moved out of `harness/`); a
thin `probes.py` runner over them is a tracked follow-up (see operations.md).

## The certify gate — three arms, cheapest-first

`certify/orchestrator.py::certify(runner, *, arch, kernel, lever, base_daemon,
var_daemon, base_ref, model, kv, maxtok, prompt_md5, seeds=None, expects=None)`
grades a variant daemon against the baseline (`B_a`) daemon and returns one
self-describing ledger row. The arms run in precedence order (short-circuit on
the first failure):

1. **PARITY** — `verdict.parity_result(base_gens, var_gens)`. The one sanctioned
   raw-daemon voodoo path: a plain-prompt, no-thinking, short greedy run is
   reproducible, so the variant's committed **token-ids must equal** the
   baseline's on every parity prompt. A value-changing kernel flips a token →
   `PARITY_FAIL` (short-circuits; perf/coherence never run).
2. **PERF (conjunctive)** — `verdict.perf_result(base_tok, var_tok, base_dur,
   var_dur, base_clk, var_clk)`. **A WIN requires `kernel_decode_tok_s` UP *and*
   rocprof pinned-clock kernel-`duration` DOWN**, each by an *independent*
   one-sided Mann-Whitney U (`perf.mwu`). A gain in only one statistic ⇒ `DEAD`
   — this closes the failure mode where a thermal artifact inflates tok/s while
   duration is flat (or a duration win hides a tok/s regression). A clock-skew
   (`perf.clock_void`, >4% median sclk delta) or missing samples ⇒
   `INCONCLUSIVE` (never faked into a DEAD). **Both `tok_delta_pct` and
   `dur_delta_pct` are written to every row.**
3. **COHERENCE** — `verdict.coherence_result(base_gens, var_gens, expects)`. The
   real `hipfire serve` path (thinking on, sampled), paired seed-set →
   `mcnemar_worse`, with the token-id attractor detector + semantic validators.
   The variant must not introduce *more* failures than the baseline.

**Guards** run before/around the arms:
- `resolve.resolve_kernel_file(kernel, repo)` maps a kernel symbol to its embedded
  `.hip` source; a symbol with no `include_str!`'d file, or an ambiguous match,
  raises `DeadFile` (nothing to compile) / `NoOp` (byte-identical variant).
- `cross_arch.check_cross_arch(kernel_file, arch, other_archs, repo)` preprocesses
  the kernel for every *other* arch (`hipcc --cuda-device-only -E`) and returns
  the archs whose device TU changed — a gfx1201 edit that perturbs a gfx1100
  build is rejected. `.gfxNNNN.hip` arch-suffixed files are skipped.

`Verdict ∈ {WIN, DEAD, PARITY_FAIL, COHERENCE_FAIL, INCONCLUSIVE, VOID,
BUILD_FAIL, …}`. Only `WIN` is bankable (`verdict.is_bankable`) — it advances the
agent's baseline and commits to `loop/card<N>`.

### GPU seam injection

`certify()` takes a `ServeRunner` (abstract). All decision logic is pure; the
GPU lives entirely behind the runner's four methods:
`parity_gens(daemon)`, `coherence_gens(daemon, seeds)`,
`perf_measure(daemon) -> (tok_list, dur_list)`, `clocks(daemon)`. Tests inject a
`MockRunner` that replays fixed measurements; on-device, `LiveServeRunner` reads
through `scripts/serve_harness.py` (tok/s) and `rocprofv3 --pmc`
(`profile_standard`, duration). This is why the whole gate is green in no-GPU CI.

## Candidate selection + exhaustion

`census.parse_rocprof` attributes per-dispatch wall%, L2-hit ratio, mem-busy, and
occupancy into a BOD (`bod_<arch>.json`). `candidates.select(bod, exhaustion,
cand_wall, k)` keeps kernels with `wall_pct >= cand_wall`, joins tried/win counts
from `ar.db`, and marks each `Candidate.state ∈ {OPEN, EXHAUSTED}`.

**Exhaustion is the loop's termination contract.** A kernel is `EXHAUSTED` after
`k_exhaust` (default 5) *consecutive* DEAD/INCONCLUSIVE rounds; **a WIN resets the
streak**. `candidates.is_exhausted(...)` is the global predicate — the driver
loop stops when *every* live candidate (unfolded, above `cand_wall`) is
exhausted. `gen_digest(...)` renders the coverage text (candidate kernels, wall%,
N/K exhaustion, levers already tried) that is injected into each agent round so
the agent avoids re-deriving a dead lever.

## Storage

- **`autoresearch/ledger/*.jsonl`** — the git-tracked source of truth. Every A/B
  is one append-only, **self-describing** JSON line carrying `gpu_arch`, `model`,
  `base_sha`, `variant_sha`, `prompt_md5`, `kv`, `maxtok`, and
  `measurement_hash = sha256("|".join([gpu_arch, model, base_sha, var_sha,
  prompt_md5, kv, maxtok]))[:16]`.
- **`autoresearch/db/ar.db`** — a SQLite *index* rebuilt from the ledger
  (gitignored). `db.ingest` is **idempotent on `measurement_hash`** (`INSERT OR
  IGNORE`), so `ar ingest` can re-run any time without double-counting. Schema
  (`autoresearch/db/schema.sql`): `attempts(arch, kernel, lever, verdict,
  tok_delta, dur_delta, profile, base_sha, var_sha, measurement_hash UNIQUE, ts)`,
  `bod(arch, kernel, wall_pct, l2_hit, mem_busy, occ, vgpr, snap_ts)`,
  `runs(id, arch, model, card, status, budget, calls, ttl, pid, ts)`.
- **`autoresearch/state/bod_<arch>.json`** — the current census snapshot per arch.

## Git-pilot (death-safe by construction)

`gitpilot.py` shells out to `git` so the harness keeps two correctness
guarantees:
- **GPU lock** — `gpu_lock(lockfile)` holds `fcntl.flock` on an *open* fd; the
  kernel releases it when the holder dies for any reason. The lockfile is **never
  unlinked** (unlinking an flock'd file lets a second acquirer lock a fresh inode
  → two holders).
- **Baseline advance** — `update_ref_cas(ref, new_sha, expected, repo)` is
  `git update-ref <ref> <new> <expected>`, an atomic compare-and-swap: it fails
  (returns `False`, no change) if the ref moved since it was read. This is the
  shared-advancing-baseline that lets parallel workers compound wins without the
  stale-baseline double-count.

## Roles

`cli.py` is role-scoped (mechanically, not by prompt-discipline):
- **operator** (Claude): `start · stop · status · why · bod · ingest · fold ·
  rollover · config · certify`
- **agent** (codex/grok): `why · status · bod · certify` — read state + submit
  candidates only.

An agent calling an operator-only verb is refused with **exit 3**. `ar certify`
is itself the mechanical leash: it refuses (exit 3) a submission on an
`EXHAUSTED` kernel, an off-target kernel (`wall% < cand_wall`), or when the arch's
running loop is over its call-budget / past its wall-TTL. The bounds live in the
tool, so an agent literally cannot re-burn a dead kernel.
