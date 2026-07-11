# autoresearch loop — config reference

Every loop is fully described by one per-arch TOML,
`autoresearch/config/loop_<arch>.toml`, parsed by `ar.config.load_config(path)`
into a `LoopConfig` (stdlib `tomllib`, Python 3.11+). **Nothing hardcodes
arch/model/card** — it all lives here. This page documents every key, its type,
and its default.

Load / inspect a config:

```bash
ar config --arch gfx1201 --json        # resolve loop_gfx1201.toml → LoopConfig
```

## Top-level keys (`LoopConfig`)

| Key | Type | Required | Default | Meaning |
|---|---|---|---|---|
| `arch` | str | **yes** | — | GPU arch id the loop targets (e.g. `gfx1201`). Selects the BOD snapshot, the `loop/<arch>` baseline, and the per-arch levers. |
| `baseline_ref` | str | **yes** | — | The kernel source-of-truth git ref this loop advances (e.g. `loop/gfx1201`). Workers branch anchors off it; WINs fold back into it via `update-ref` CAS. |
| `model` | str | **yes** | — | Measurement SKU used to grade kernels (e.g. `qwen3.6-35b-a3b.mq4r`). `mq4r` is the shared-kernel speed SKU for the gfx1201 loop. **mqN quant only** — never a ggml/Q4_K model. |
| `kv_mode` | str | no | `"q8"` | KV-cache mode for the perf/coherence measurement. `q8` for the standard loop; prefer an FWHT mode for long-context. |
| `max_tokens` | int | no | `128` | Decode length per A/B measurement. Kept short (128) so the perf arm samples the decode kernels cheaply. |
| `prompt_md5` | str | no | `""` | md5 of the **byte-identical** measurement prompt. Recorded on every ledger row; a mismatch across sessions/agents invalidates a cross-run perf comparison (one newline can swing τ ~17%). |
| `cand_wall` | float | no | `3.0` | Minimum BOD wall% for a kernel to be a candidate. Kernels below this are `below-threshold` (never selected, `certify` refuses them as off-target). |
| `k_exhaust` | int | no | `5` | The "5 tests/kernel" budget: **consecutive** DEAD/INCONCLUSIVE rounds on one kernel ⇒ `EXHAUSTED`. **A WIN resets the counter.** The loop stops when every live candidate is exhausted. |
| `agent_harness` | str | no | `"codex"` | Default coding-agent harness for a round (`codex` or `grok`). A per-worker `model` still selects the concrete model; this picks the CLI dispatch shape. |

## `[[workers]]` — per-worker heterogeneity (`WorkerCfg`)

An array of tables, one per GPU worker. This is the **`sed`-munge replacement**:
the old `swarm_explore.sh` rewrote a shared prompt file to retarget each worker;
now each worker's identity comes straight from the TOML. All four keys are
required within a worker entry.

| Key | Type | Meaning |
|---|---|---|
| `card` | int | DRM card index → the worker's `.aw/sw_card<card>` worktree and per-dev lockfile. |
| `dev` | int | HIP device index the worker's daemon binds. **Verify against the daemon-reported arch** — `rocminfo` lies about device order. |
| `model` | str | The agent model this worker runs (e.g. `gpt-5.6-luna`). Heterogeneous by design — different models generate orthogonal kernel levers over the same certify substrate. |
| `effort` | str | Reasoning effort for that model (e.g. `max` / `xhigh` / `medium`). |

If the `[[workers]]` list is empty, `workers` is `[]` (no workers planned).

## `[bounds]` — watcher leashes (`Bounds`)

| Key | Type | Default | Meaning |
|---|---|---|---|
| `call_budget` | int | `400` | Max agent calls per run. The watcher auto-stops a run at/over this; `ar certify` refuses (`BUDGET_SPENT`, exit 3) once spent. |
| `wall_ttl_s` | int | `43200` | Wall-clock TTL (seconds, default 12 h). The watcher auto-stops a run past it; `ar certify` refuses (`TTL_EXPIRED`, exit 3). |

If `[bounds]` is omitted entirely, it defaults to `Bounds(call_budget=400,
wall_ttl_s=43200)`.

## Round prompt

Each arch also has a round prompt at `autoresearch/config/prompt_<arch>.md` (the
`driver.py` primary lookup; falls back to the generic harness prompt if absent).
It is the user prompt handed to the coding agent each round; the per-kernel
tried-levers digest (`candidates.gen_digest`) is prepended so the agent avoids
re-deriving dead levers. Ships: `prompt_gfx1201.md`, `prompt_gfx1151.md`.

## Environment overrides (paths only — not loop params)

Loop parameters come **only** from the TOML. These env vars just relocate the
store/ledger for a checkout and are read by `ar.cli`:

| Env | Default | Meaning |
|---|---|---|
| `AR_REPO` | three levels up from `cli.py` | Repo root. |
| `AR_DB` | `<repo>/autoresearch/db/ar.db` | The SQLite index (gitignored; rebuilt by `ar ingest`). Also settable per-invocation with `--db`. |
| `AR_LEDGER` | `<repo>/autoresearch/ledger` | Ledger dir (`*.jsonl`, git-tracked source of truth). |
| `AR_BOD_GLOB` | `<repo>/autoresearch/state/bod_*.json` | BOD census snapshots ingested into the `bod` table. |

## Annotated example — `loop_gfx1201.toml`

```toml
# Copyright (c) Kaden Schutt
arch          = "gfx1201"               # R9700; selects BOD, levers, loop/<arch>
baseline_ref  = "loop/gfx1201"          # the kernel source of truth this loop advances
model         = "qwen3.6-35b-a3b.mq4r"  # measurement SKU (mq4r = shared-kernel speed SKU)
kv_mode       = "q8"
max_tokens    = 128
prompt_md5    = "d97ec9d3f761ec68093631be27d32441"   # byte-identical prompt hash
cand_wall     = 3.0                     # min BOD wall% to be a candidate
k_exhaust     = 5                       # 5 consecutive deads ⇒ EXHAUSTED (a WIN resets)
agent_harness = "codex"

# Per-worker heterogeneity (the sed-munge replacement):
# Luna@max card1, Terra@xhigh card2, Sol@medium card3.
[[workers]]
card = 1
dev = 1
model = "gpt-5.6-luna"
effort = "max"

[[workers]]
card = 2
dev = 2
model = "gpt-5.6-terra"
effort = "xhigh"

[[workers]]
card = 3
dev = 3
model = "gpt-5.6-sol"
effort = "medium"

[bounds]                                # watcher leashes
call_budget = 400
wall_ttl_s  = 43200
```

## Adding a new arch

1. Copy `loop_gfx1201.toml` → `loop_<arch>.toml`; set `arch`, `baseline_ref`
   (`loop/<arch>`), `model`, and the `[[workers]]` `{card, dev}` for that box's
   fleet (verify card→dev against the daemon-reported arch).
2. Add `autoresearch/config/prompt_<arch>.md` (the round prompt).
3. Produce a `bod_<arch>.json` census (`census.run_census` on the box) and
   `ar ingest`.
4. `ar bod --arch <arch>` to confirm candidates rank + exhaustion marks are sane
   before launching.
