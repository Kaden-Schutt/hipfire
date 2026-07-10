# Skill: autoresearch-loop (drive the kernel-optimization loop via `ar`)

The autoresearch loop finds per-arch kernel wins: a BOD census ranks the hot
kernels, per-worker agents author levers, and each candidate is A/B-certified
(parity → perf → coherence) into the git-tracked ledger. `ar`
(`autoresearch/ar/cli.py`) is the **one surface** you touch to interact with a
running loop — read state, then submit a candidate. There is no reason to open a
raw `.sh` under `autoresearch/`; every one has been converged into the `ar`
package.

**Reach for this when:** you ssh into a GPU box (hiptrx/gfx1201, k9lin/gfx1100,
hipx/gfx1151) to interact with a loop; you're an agent worker asked to propose a
kernel lever and want it graded; or you're the operator starting/stopping a
swarm, folding a win, or checking run health.

## The contract (do this in order, every time)

1. **Read the census before authoring anything.** `ar bod --arch <arch>` ranks
   the candidate kernels (wall%, roofline bound, tried/win counts, OPEN vs
   EXHAUSTED). Pick an **OPEN** candidate — never an `EXHAUSTED` one, never a
   `below-threshold` one.
2. **Read what already failed.** `ar why <kernel> --arch <arch>` lists every
   lever already tried on that kernel and its verdict. **Never re-propose a lever
   already marked DEAD/INCONCLUSIVE** — the history is there so the search space
   narrows, not re-treads.
3. **Author the lever, then submit via `ar certify`.** Bounds are enforced
   mechanically in the tool: a `certify` on an EXHAUSTED / off-target
   (below-cand-wall / non-census) / over-budget kernel is **refused with exit
   code 3** — not discouraged, refused. When bounds pass, `certify` accepts and
   the driver grades the A/B; the row lands in the ledger.
4. **Never touch a raw script.** Agents interact through `ar` only. If you think
   you need a shell script under `autoresearch/`, you want an `ar` verb instead.

## Roles

The CLI is role-scoped. Pass `--role`:

- **`--role operator`** (Claude / you at the wheel): `start · stop · status ·
  why · bod · ingest · fold · rollover · config`.
- **`--role agent`** (codex/grok worker): `why · status · bod · certify` only.
  Any operator-only verb is refused with exit 3. Default role is `operator`.

## Verbs

Invoke as `python -m autoresearch.ar <verb> …` (from the repo root), or `ar`
where the entrypoint is installed.

| Verb | Role | What it does |
|---|---|---|
| `bod --arch <a> [--json]` | both | Ranked candidate kernels + coverage (OPEN/EXHAUSTED). |
| `why <kernel> --arch <a> [--json]` | both | Levers tried on a kernel + verdicts (the why-dead surface). |
| `status [--json]` | both | Banked attempts/wins + running loops (budget/TTL left). |
| `certify --arch <a> --kernel <k> --lever <L> --variant <path>` | both | Submit a candidate; **exit 3** if EXHAUSTED / off-target / over-budget. |
| `ingest [--ledger <dir>] [--bod <glob>]` | operator | Re-index the ledger + BOD into `ar.db` (idempotent). |
| `start --config <toml>` | operator | Launch the config's workers (swarm). |
| `stop [--run <id>]` | operator | Stop running loops. |
| `fold --ref <ref> --sha <win> [--dry-run]` | operator | Fold a WIN into the shared baseline (`update-ref` CAS, reversible). |
| `rollover [--reason <r>] [--dry-run]` | operator | Re-census + re-ingest after an advance/exhaustion. |
| `config (--config <toml>\|--arch <a>) [--json]` | operator | Print the resolved per-arch loop config. |

## Mechanical bounds (why `certify` may refuse you)

`ar certify` is the leash. It returns **exit 3** with a JSON `reason` when:

- `KERNEL_EXHAUSTED` — the kernel hit `k` consecutive DEADs with no WIN (a WIN
  resets the streak). Move to another OPEN kernel from `ar bod`.
- `OFF_TARGET` — the kernel is not a BOD census candidate (`wall% < cand_wall`);
  optimizing it can't move the wall.
- `BUDGET_SPENT` / `TTL_EXPIRED` — the arch's running loop has spent its
  call-budget or wall-TTL. The operator restarts it; you don't override it.

On accept (exit 0) the driver runs the real A/B — a conjunctive perf gate
(`kernel_decode_tok_s` UP **and** rocprof kernel-duration DOWN) plus parity and
serve-path coherence — and writes a self-describing ledger row. After a batch of
certifies, the operator runs `ar ingest` to re-index.

## Store + config

- **`ar.db`** — the queryable index, rebuilt idempotently from the ledger by
  `ar ingest`. Path resolves from `--db` › `$AR_DB` ›
  `<repo>/autoresearch/db/ar.db`. The ledger (`autoresearch/ledger/*.jsonl`,
  git-tracked) is the source of truth; `ar.db` is disposable.
- **Config** — one TOML per arch (`autoresearch/config/loop_<arch>.toml`): the
  baseline ref, measurement SKU/kv/maxtok, `cand_wall`, `k_exhaust`, the
  per-worker `{card, dev, model, effort}` list, and the watcher leashes. Nothing
  is hardcoded elsewhere — `ar config --arch <a>` prints the resolved values.

## Guardrails you inherit (don't fight them)

- **master-push and default-flips are never automated** — the watcher stages +
  notifies; a human lands them.
- **Every fold is a `update-ref` CAS with the prior SHA recorded** — reversible,
  and it fails cleanly if another worker advanced the baseline first (no
  double-count).
- **Byte-identical prompts only** for any perf claim (`prompt_md5` on every row);
  measure at AUTO clock. See `CLAUDE.md` § "Perf benchmarking".
