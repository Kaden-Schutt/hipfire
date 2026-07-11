# Autoresearch Loop → Python Migration — Design

**Date:** 2026-07-09
**Status:** Approved design (pending spec review)
**Branch:** `feat/rdna-kernel-oracle` (harness code lands here; operates on `loop/gfx1201 @ fd6deaa9` as the kernel baseline)
**Supersedes/absorbs:** `autoresearch/AR_SUPERVISOR_DESIGN.md` (the `hipfire-ar` supervisor is folded in as the watcher)

## 1. Problem

The autoresearch kernel-optimization loop *works* — the science is proven (BOD-census candidate selection, self-exhausting per-kernel A/B certify, tried-lever digest to the agent, fold/rollover). What rots is the **harness**: it is a bash sprawl with a half-built Python track bolted alongside, and no clean entrypoint.

Concrete debt (measured on this branch):

- **38 bash scripts, 2937 lines**, with rampant generational duplication: **6** `ab_certify*` variants (`ab_certify.sh`, `ab_certify_v2.sh`, `harness/ab_certify_v2b.sh`, `harness/ab_certify_v2p.sh`, `v2/ab_certify_v2.sh`, `v3-queue/ab_certify_v2.sh`), **5** `rollover*` variants, and an entire dead `v3-queue/` parallel generation.
- **Embedded Python-in-bash heredocs** (`ab_certify_v2p.sh` `measure()`, the ledger writer, the coherence check): unlintable, untestable, quoting-fragile. This is the real wart — Python smuggled through bash strings.
- **`swarm_explore.sh` configures workers by `sed`-rewriting a prompt file** — it cannot cleanly express per-worker `{card, dev, model, effort}`, which is exactly what the Sol/Terra/Luna eval needs.
- **Ephemeral state**: the supervisor's `ar.db` (the 490-attempt store the supervisor design promised) has evaporated — it lived in a since-removed checkout. State scatters across `/tmp` and stray worktrees.
- **~12+ detached worktrees on hiptrx** and a `.md`/`.txt`/`.json` jumble under `autoresearch/`.
- **No agent-facing entrypoint**: an agent that ssh-es into the box enters through scripts written for other purposes, guesses env vars, and re-derives dead levers.

The loop is also **half-migrated already**: `ab_certify_serve.py` + `serve_runner.py` (the serve-`tok/s` + coherence arms), `coherence_arm.py`, `perf.py`, `certify_verdict.py`, `exhaustion.py`, `oracle_db.py`, the exhaustion trio, and `hipfire_ar.py` (the watcher seed) exist **with unit tests**. The task is to *finish the convergence*, not start over.

## 2. Goals / Non-goals

**Goals**
- One Python package (`autoresearch/ar/`) — no more `harness/` vs `v2/` vs `v3-queue/`.
- **Configurable**: per-arch and per-worker (`{card, dev, model, effort}`), thresholds, baseline ref — nothing hardcodes a3b/mq4/gfx1201.
- Reuse the **daemon for coherence** and **`serve_harness` for tok/s** (already the design of the serve arms) — no raw-daemon voodoo except the one sanctioned parity path.
- **Python pilots git** (worktree, fold, rollover, CAS) while preserving bash's death-safe `flock` semantics.
- A **watcher daemon** that tracks runs and **auto-enforces fold/rollover** under guardrails.
- An **agent-facing CLI + skill** so ssh-in / on-device interaction goes through `ar`, never a repurposed shell script.
- **Clean folder layout** (db, ledger, levers/`gfxNNNN.md`, config) and **rigorous docs**.
- **All loop `.sh` migrated to Python**; dead generations deleted.

**Non-goals**
- Rewriting kernels or changing the certify *science* (parity / perf / coherence arms stay as specified).
- MTP/DFlash work (this loop is AR-only).
- Any inference-hot-path Python (this is *tooling* — Python is blessed per CLAUDE.md).
- Any ggml/Q4_K import (mqN only).
- Master-push / default-flip automation (those stay human gates per the master-push-scope rule).

## 3. Principle: converge, don't rewrite

Every new module has a named predecessor to port and validate against. Reuse map:

| New module (`autoresearch/ar/`) | Ports / absorbs | LOC seed |
|---|---|---|
| `config.py` | (new) TOML loader | — |
| `db.py` | `oracle_db.py` | 126 |
| `gitpilot.py` | git bits of `ab_certify_v2p.sh` + `rollover_v2.sh` (CAS, worktree, flock, fold) | — |
| `census.py` | `oracle_profile.sh` | 106 |
| `candidates.py` | `exhaustion.py` + `v2/{check_exhausted,gen_digest,update_exhaustion}.py` | 78+71 |
| `certify/orchestrator.py` | `ab_certify_serve.py` | 218 |
| `certify/serve_runner.py` | `serve_runner.py` | 205 |
| `certify/coherence.py` | `coherence_arm.py` | 158 |
| `certify/perf.py` | `perf.py` | 80 |
| `certify/verdict.py` | `certify_verdict.py` | 54 |
| `certify/cross_arch.py` | `cross_arch_guard.sh` | 44 |
| `certify/resolve.py` | symbol→file resolver + DEAD_FILE/NO_OP guards from `ab_certify_v2p.sh` | — |
| `driver.py` | `v2/driver_v3.sh` | 104 |
| `swarm.py` | `swarm_explore.sh` (kills the `sed`-munge) | 57 |
| `agent_exec.py` | `agent_exec.sh` (codex/grok dispatch) | 51 |
| `watcher.py` | `hipfire_ar.py` supervisor + `rollover_v2.sh` enforcement | 340 |
| `cli.py` | `hipfire_ar.py` CLI surface | (part of 340) |
| `probes.py` | `cu_*.sh`, `gemv_*.sh`, `mall_probe.sh` (thin runners over the `.hip` probe sources) | ~45 |

**Deleted outright** (dead generations, no port): `v3-queue/*`, `ab_certify.sh`, `ab_certify_v2.sh`, `harness/ab_certify_v2b.sh`, `harness/ab_certify_v2p.sh` (after `certify/` reaches parity), `loop_driver_v2.sh`, `rollover.sh`, `rollover_serve.sh`, `rollover_stack.sh`, `stop_and_rollover.sh`, `launch24h.sh`, `launch_v2.sh`, `catchup_census.sh`, `fix_baseline.sh`, `swarm_certify.sh`, `certify_v3.sh`, `ab_certify_swarm.sh`, `ab_certify_serve.sh`.

## 4. Architecture

A single package, thin modules with one job each, dependency-injected GPU seams so the decision logic stays no-GPU-testable (the existing pattern — keep it).

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
                       │            │                   ├ resolve.py  (symbol→file, DEAD_FILE/NO_OP)
                       │            │                   ├ serve_runner.py  (parity/coherence/perf gens)
                       │            │                   ├ coherence.py / perf.py / verdict.py
                       │            │                   └ cross_arch.py  (preprocessor-invariance guard)
                       │            │                        │  WIN
                       │            └─► gitpilot.py ◄─────────┘  (update-ref CAS advance, fold, rollover)
                       │                     │
                       └──────────► db.py  ◄─┘   (ar.db: attempts, bod, runs)  + ledger/*.jsonl
```

**Interfaces (the contracts that matter):**
- `certify(runner, *, arch, kernel, lever, base_daemon, var_daemon, base_ref, ...) -> Verdict` — already the `ab_certify_serve` signature. Verdict ∈ {WIN, DEAD, PARITY_FAIL, CROSS_ARCH, DEAD_FILE, NO_OP, INCONCLUSIVE}.
- `candidates.select(bod, exhaustion, cand_wall, k) -> list[Candidate]` with `state ∈ {OPEN, EXHAUSTED}` + roofline + tried/win counts.
- `gitpilot.advance_baseline(shared_ref, new_sha, expected_sha) -> bool` (CAS under flock) and `.fold(...)`, `.rollover(...)`.
- Every certify writes a **self-describing ledger row**: `gpu_arch`, `model`, `base_sha`, `variant_sha`, `prompt_md5`, `kv`, `maxtok`, and `measurement_hash = sha256(gpu_arch|model|base_sha|variant_sha|prompt_md5|kv|maxtok)[:16]` (already implemented; keep the format).

## 5. Folder layout (target)

```
autoresearch/
  ar/                     # THE package
    __init__.py  config.py  db.py  gitpilot.py  census.py  candidates.py
    driver.py  swarm.py  watcher.py  agent_exec.py  probes.py  cli.py
    certify/  __init__.py orchestrator.py serve_runner.py coherence.py
              perf.py verdict.py cross_arch.py resolve.py
    tests/    test_*.py   # the existing no-GPU tests move here
  config/                 # loop_gfx1100.toml  loop_gfx1151.toml  loop_gfx1201.toml
                          # round prompts: prompt_<arch>.md
  db/                     # ar.db (durable) + schema.sql          [gitignored: ar.db]
  ledger/                 # *.jsonl history (unchanged, git-tracked)
  levers/                 # gfx1100.md  gfx1151.md  gfx1201.md      (already here)
  probes/                 # *.hip probe sources                    (moved from harness/)
  variants/               # winner kernel sources                  (unchanged)
docs/
  autoresearch/           # rigorous docs: architecture.md, operations.md, config-reference.md
  skills/autoresearch-loop.md   # the agent-facing skill (ssh-in → `ar`, never raw scripts)
```

`autoresearch/harness/` and `autoresearch/*.sh` are removed at the end of Phase 8. Loose design `.md`s under `autoresearch/` (`BANKED.md`, `DECODE_AR_2STAGE_DESIGN.md`, `EXPANDED_SURFACE.md`, `AR_SUPERVISOR_DESIGN.md`) move to `docs/autoresearch/` (superseded ones marked historical).

## 6. Config schema (TOML)

`config` is a param object → env for `driver`/`certify` (which already read `ARCH`/`BOD`/`MODEL`). One file per arch:

```toml
# autoresearch/config/loop_gfx1201.toml
arch          = "gfx1201"
baseline_ref  = "loop/gfx1201"          # the kernel source of truth this loop advances
model         = "qwen3.6-35b-a3b.mq4r"  # measurement SKU (mq4r for the shared-kernel loop)
kv_mode       = "q8"
max_tokens    = 128
prompt_md5    = "d97ec9d3f761ec68093631be27d32441"
cand_wall     = 3.0                     # min BOD wall% to be a candidate
k_exhaust     = 5                       # the "5 tests/kernel" budget: consecutive DEAD/INCONCLUSIVE per kernel ⇒ EXHAUSTED (a WIN resets the counter)
agent_harness = "codex"

[[workers]]                             # per-worker heterogeneity — the sed-munge replacement
card = 1; dev = 1; model = "gpt-5.6-luna";  effort = "max"
[[workers]]
card = 2; dev = 2; model = "gpt-5.6-terra"; effort = "xhigh"
[[workers]]
card = 3; dev = 3; model = "gpt-5.6-sol";   effort = "medium"

[bounds]                                # watcher leashes
call_budget = 400
wall_ttl_s  = 43200
```

## 7. Git-pilot (`gitpilot.py`)

Python drives git via `subprocess` (not GitPython/pygit2) — we still shell out to `git`/`cargo`/daemon for everything, and this preserves the **death-safe** guarantee that makes the current harness correct:

- **GPU lock**: hold `flock` on an open fd (via a context manager over `flock(1)` or `fcntl.flock`) so the kernel releases on holder death. Never unlink the lockfile.
- **Baseline advance**: `git update-ref <shared_ref> <new> <expected>` (compare-and-swap) under a per-SHA build lock — the shared-advancing-baseline that fixes the stale-PREBUILT_BASE double-count.
- **Worktree lifecycle**: `git worktree add/list/remove`; per-worker checkout anchor branch.
- **Fold / rollover**: port `rollover_v2.sh` (fold certified wins into the shared baseline, re-census, re-ingest). Every mutation is dry-run-previewable and reversible (records the prior SHA).

## 8. Certify (three arms — unchanged science)

Order **parity → perf → coherence** (cheapest-first), reusing the built arms:
- **PARITY**: raw-daemon short greedy, token-id-exact value preservation (the one sanctioned voodoo path). A value change ⇒ `PARITY_FAIL`.
- **PERF**: measured through **`serve_harness`** (the real CLI spawn — no raw-daemon voodoo). **Two statistics, BOTH gating (conjunctive): a WIN requires `kernel_decode_tok_s` UP *and* rocprof pinned-clock kernel-`duration` DOWN** (`profile_standard`), each by an independent Mann-Whitney U. A gain in one but not the other ⇒ `DEAD`. Requiring both closes the failure mode where a thermal artifact inflates tok/s while duration is flat (or a duration win hides a tok/s regression). **Both statistics are written to every ledger row**; tok/s is also the human-facing number.
- **COHERENCE**: the real `hipfire serve` path (thinking on, sampled), paired seed-set → McNemar, with the token-id attractor detector + semantic validators.
- **Guards**: `resolve.py` (symbol→file, DEAD_FILE/NO_OP) and `cross_arch.py` (preprocessor-invariance: a gfx1201 edit must not change any other arch's device TU) run before/around the arms.

## 9. Watcher daemon (`watcher.py`) — auto-enforce, guardrailed

A persistent process (one per box) that owns run lifecycle and **auto-enforces fold/rollover**:

- **Tracks runs** in `ar.db` (`runs`: id, arch, model, card, status, calls, budget, ttl, pid). Reaps dead pids; single-owner lock per `(box, card)`.
- **Auto-fold**: when a worker certifies a WIN and advances its anchor, the watcher folds it into the shared baseline via `gitpilot`. **Auto-rollover**: when the baseline advances past a threshold (or a worker exhausts its candidate set), it re-censuses the BOD and re-ingests.
- **Guardrails (the whole point):** every enforced action is (1) **dry-run-logged** first, (2) **git-reversible** (prior SHA recorded), (3) gated by **budget/TTL/exhaustion leashes** — a run past its call-budget or wall-TTL is auto-stopped; a certify on an EXHAUSTED/off-target kernel is refused (exit 3, mechanical, not prompt-discipline).
- **Human gates preserved:** master-push and default-flips are never automated — the watcher stages them and notifies.

## 10. Agent CLI + skill

`ar` (`cli.py`), role-scoped (from the supervisor design):
- **operator / Claude:** `start · stop · status · why · bod · ingest · fold · rollover · config`
- **agent / codex:** `why · status · bod · certify` — read state + submit candidates; bounds mechanically enforced.

`docs/skills/autoresearch-loop.md` — the agent-facing skill: how to ssh in, read `ar bod`/`ar why` before authoring a lever, submit via `ar certify`, and read run health via `ar status`. **The contract: agents touch `ar`, never a raw script.** Indexed in CLAUDE.md's skills list.

## 11. Migration strategy — strangler-lite

For each module: build Python → **validate against the bash it replaces** (fixture parity in no-GPU CI + one live A/B that agrees within noise) → **then delete the bash**. Rationale: certify + git-CAS correctness is too subtle to cut over blind, but nothing is running live right now (ar.db evaporated), so validation is cheap and there's no traffic to protect. No big-bang.

## 12. Build phases (each independently testable)

1. **Foundation** — `config.py` (TOML), `db.py` (port `oracle_db` + `ar.db` schema), `gitpilot.py` (flock/CAS/worktree). *Accept:* no-GPU tests green; `ar ingest` rebuilds `ar.db` from the ledger idempotently.
2. **Certify** — wire `certify/` end-to-end against a live daemon; port `resolve.py` + `cross_arch.py`. *Accept:* a known WIN and a known DEAD from the ledger reproduce the same verdict as `ab_certify_v2p.sh`.
3. **Census + candidates** — `census.py`, `candidates.py`. *Accept:* `ar bod --arch gfx1201` ranks candidates + marks EXHAUSTED matching the current trio on the same ledger.
4. **Driver** — `driver.py` self-exhausting loop + `agent_exec.py`. *Accept:* one worker runs a full round on gfx1201 (codex authors → `ar certify` gates → ledger row → exhaustion updates).
5. **Swarm** — `swarm.py` from config. ***Accept (headline): the Sol/Terra/Luna eval launches on `mq4r`/gfx1201*** — 3 workers (Luna@max card1, Terra@xhigh card2, Sol@medium card3), budget 5 tests/kernel, wins fold into the shared baseline.
6. **Watcher** — `watcher.py` daemon: run tracking + auto-fold/rollover (guardrailed) + leashes. *Accept:* a simulated WIN auto-folds (dry-run logged, reversible); an over-budget run auto-stops.
7. **CLI + skill** — finalize `ar` roles; write `docs/skills/autoresearch-loop.md`; index in CLAUDE.md. *Accept:* operator + agent role scoping enforced; a fresh agent can drive a round using only the skill.
8. **Cleanup + docs** — delete the 38 `.sh`, move folders to the §5 layout, write `docs/autoresearch/{architecture,operations,config-reference}.md`. *Accept:* `rg '\.sh' autoresearch/` returns only probe sources; no-GPU CI green; docs complete.

## 13. Execution via multi-agent workflow

The migration is built by a **Workflow**: per phase, parallel Claude agents build the modules (worktree-isolated where they mutate files concurrently), and **Codex nodes adversarially verify** the two subtle correctness seams — certify verdict parity and git-CAS/fold semantics — before each phase's bash is deleted. Phases are sequential (later depend on earlier); modules within a phase fan out. The workflow returns per-phase acceptance results; the operator reviews between phases.

## 14. Testing

- **No-GPU (CI):** unit tests for every decision module (the existing `test_*.py` culture — port and extend). Fixtures: a captured ledger, a captured `bod_<arch>.json`, mock `ServeRunner`. Cover verdict logic, candidate/exhaustion, config parse, gitpilot CAS (against a temp repo), watcher leash refusals.
- **Live (gfx1201 on hiptrx, gfx1100 on k9lin blocking; gfx1151 on hipx async):** the per-phase acceptance criteria above; Phase 5 is the real Sol/Terra/Luna run.
- **Parity-vs-bash:** each ported module reproduces the bash predecessor's output on the same input before the bash is deleted.

## 15. Decisions log

- **Source of truth:** `loop/gfx1201 @ fd6deaa9` ("loop/gfx1200" was a typo). gfx1201 naming throughout.
- **Sequencing:** migrate first; the Sol/Terra/Luna eval is Phase 5's acceptance test, run on **mq4r** (the loop tunes the shared MoE-GEMV kernels; mq4r is the primary speed SKU at 160 tok/s vs ZINC 166). mq4p's Q8-linear-attention gap (108 tok/s) is a separate later effort.
- **Perf gate:** conjunctive — a WIN needs tok/s UP *and* duration DOWN (both recorded on every ledger row); either alone ⇒ DEAD.
- **Watcher authority:** auto-enforce fold/rollover, guardrailed (dry-run + reversible + leashes); master-push/default-flip stay human gates.
- **Workflow:** the migration is executed by a multi-agent workflow; the loop runtime stays the `ar/` package.
- **Language:** Python (tooling, not hot path). Not TS (that owns the user CLI). Git piloted via `subprocess` to keep flock-on-death semantics.

## Open follow-ups (tracked, not in this spec)

- Reconcile the ~12 stray hiptrx worktrees (`git worktree prune` + a documented worktree policy) — operational cleanup, done during Phase 8.
- Merge/rebase the `feat/rdna-kernel-oracle` harness lineage with the `loop/gfx1201` kernel lineage so the umbrella branch holds both ("all of this folds into kernel-oracle").
