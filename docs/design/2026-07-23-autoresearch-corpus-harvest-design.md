<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Kaden Schutt
hipfire — see LICENSE and NOTICE in the project root.
-->

# Autoresearch corpus harvest — design

Consolidate the autoresearch experiment ledgers scattered across the fleet
into one git-tracked corpus with a regenerable SQLite index, and join the
outcome rows to the static ISA rows that already exist alongside them.

| Field | Value |
|---|---|
| Date | 2026-07-23 |
| Status | **Implemented** — `scripts/harvest_ledgers.py`. See "As built" below. |
| Scope | `autoresearch/` ledgers only. Redline is explicitly excluded. |
| Boxes surveyed | k9lin (local), hipx, hiptrx |

---

## Problem

The autoresearch loop has been writing structured experiment results since
roughly 2026-07-03. Those results are real, labelled, and statistically
tested — and they are scattered across ~30 worktrees per box on three
machines, two of which are explicitly disposable. Nothing has ever read them
in aggregate.

Measured 2026-07-23:

| | k9lin | hipx | hiptrx | total |
|---|---:|---:|---:|---:|
| ledger rows | 1,593 | 765 | 3,905 | **6,263** |
| unique ledger files (name+size) | 23 | 63 | 42 | 128 |
| `kernel-ledger` rows (static/ISA) | 24 | 18 | 36 | **78** |
| atlas JSONL rows | 0 | 0 | 243 | **243** |
| `bod_*.json` | 6 | 4 | 20 | 30 |
| `*cert*.json` | 10 | 15 | 42 | 67 |
| `results.json` | 374 | 581 | 461 | 1,416 |

Pooled verdict distribution over all 6,263 outcome rows:

```
DEAD  3,826   NOISE  938   WIN  879   LOSS  122   INCONCLUSIVE  81
VOID     25   DEAD_FILE 4   CROSS_ARCH 2   NO_OP 2   COHERENCE_FAIL 1
```

Archs covered: `gfx1010`, `gfx1030`, `gfx1100`, `gfx1151`, `gfx1201` — the
full tracked line. Distinct kernels: 26 (hipx), 17 (k9lin / hiptrx).

Three findings drive the design:

1. **The falsification corpus is ~4x the win corpus** (3,826 DEAD + 122 LOSS
   vs 879 WIN). This is the half that says where *not* to spend GPU time, and
   it is the part that cannot be reconstructed without re-burning the hours
   that produced it.

2. **A second, static corpus already exists and has never been joined.**
   `tests/kernel-ledger/*.jsonl` is already `hipfire.kernel_atlas.v0` and
   carries a pre-run feature vector — `{isa_fingerprint, vgpr, sgpr, lds,
   scratch}` plus `bound_class`, `shape_bucket`, and a pinned `.hsaco`
   fixture path. Joining it to outcomes on `(arch, kernel)` is what turns the
   corpus into a pre-flight filter instead of a historical record.

3. **`measurement_hash` is present on only ~204 of 6,263 rows (~3%).** k9lin
   has zero. The `INSERT OR IGNORE` idempotency contract in
   `autoresearch/db/schema.sql` therefore does not work on historical rows,
   and dedup needs a composite fallback. This was the single most consequential
   discovery of the survey.

Two supporting facts: `ar.db` has **never** been populated with real data (the
51 `ar.db` files on k9lin are all pytest fixtures under `/tmp/pytest-of-kaden/`,
totalling 8 unique hashes across 3 kernels), and the `sqlite3` CLI is absent on
all three boxes — Python's built-in `sqlite3` module is unaffected, so this
blocks inspection but not ingest.

## Scope

**In scope.** `autoresearch/ledger/*.jsonl`, `tests/kernel-ledger/*.jsonl`,
`atlas*.jsonl`, and `autoresearch/state/bod_*.json`, harvested from k9lin,
hipx, and hiptrx.

**Explicitly out of scope.** Redline artifacts — `.redline-work/`,
`redline-results/`, `redline-cert-*/`, `redline-runs/`. Redline measurement has
a different identity model and a different question (retained-PM4 replay
parity, not lever-vs-baseline perf). Mixing the two corpora would force a
lowest-common-denominator schema that serves neither. If a redline corpus is
wanted later it gets its own spec and its own tables.

**Deferred to v2.** The ~35,400 `.hsaco` binaries and ~3.3 GB of rocprof CSV
are referenced *by path* from static rows but not ingested. Whether to pull
them in depends on whether the `(arch, kernel)` join shows signal, which we
cannot know until v1 exists.

## Data model

Two fact types, deliberately not merged into one table.

**`attempts` — outcome facts.** "This lever was tried on this kernel and here
is what happened." Sourced from `autoresearch/ledger/*.jsonl`.

Observed row shape (real example, hiptrx):

```json
{"arch":"gfx1100","kernel":"gemm_hfq4g256_moe_grouped_wmma_gfx12",
 "label":"R18c0_header_lane_broadcast","variant":"exp_R18c0.hip",
 "verdict":"DEAD","WIN":false,"delta_pct":-0.2,
 "mwu_dominance":0.281,"rounds":4,
 "base_decode":174.45,"var_decode":174.1,
 "base_coh":"OK","var_coh":"OK","win_commit":null,
 "roofline":{"target_base":{"wall_pct":8.4,"occ":47.3,"l2_hit_pct":74.2,
                            "mem_busy":81.6,"vgpr":128,"lds":0},
             "target_var":{...}},
 "profile_feedback":"no clear lever signal",
 "base_sha":"62200a00...","prompt_md5":"d97ec9d3...",
 "base_runs":[174.1,174.8,...]}
```

**`kernel_static` — static facts.** "This kernel, compiled for this arch, has
these ISA characteristics." Sourced from `kernel-ledger` / atlas rows:

```json
{"schema":"hipfire.kernel_atlas.v0","phase":"decode",
 "workload_kind":"moe_gate_up","quant":"hfq4g256","shape_bucket":"decode_gemv",
 "metrics":{"isa_fingerprint":10254263892362831541,"lds":0,"scratch":0,
            "sgpr":8,"vgpr":80},
 "arch":"gfx1201","bound_class":"valu_issue",
 "kernel":"gemv_hfq4g256_moe_gate_up_indexed_batched",
 "reproducer":{"cmd":"...","fixture_path":"tests/kernel-fixtures/gfx1201/....hsaco"}}
```

**`bod`** keeps the existing shape from `autoresearch/db/schema.sql`
(`arch, kernel, wall_pct, l2_hit, mem_busy, occ, vgpr, snap_ts`).

**`provenance`** is new and mandatory: `(row_id, source_box, source_path,
source_mtime, harvest_ts, key_confidence)`. Non-negotiable because the same
logical file exists in up to 6 worktrees at *different* row counts —
`hipfire-loop` carried 130 rows where its copies carried 121. Without
provenance, "which copy is authoritative" is unanswerable.

The join is `attempts.(arch, kernel)` → `kernel_static.(arch, kernel)`.

## Identity and dedup

`autoresearch/db/schema.sql` documents the canonical key as:

```
measurement_hash = sha256(gpu_arch|model|base_sha|var_sha|prompt_md5|kv|maxtok)[:16]
```

Harvest recomputes this verbatim wherever all seven components are present, so
back-filled rows agree bit-for-bit with rows the loop emits going forward.

They frequently are not present. The survey found `model` and `var_sha` absent
from most historical swarm rows, while `kernel`, `label`, and `variant` are
near-universal (`kernel` 3,220/3,374 on hiptrx). So harvest defines a **second,
explicitly distinct** fallback key:

```
fallback_key = sha256(arch|kernel|label|base_sha|variant|prompt_md5)[:16]
```

The two are stored in separate columns and never conflated — `measurement_hash`
stays null when it cannot be computed canonically. Dedup uses
`COALESCE(measurement_hash, fallback_key)`. This keeps the canonical key
authoritative and honest about which rows have one, rather than minting
look-alike hashes under the same column name that would silently fail to match
future loop output.

Rows missing components are **kept**, not dropped, with `key_confidence:
"weak"` and full `source_path`. Rationale: a DEAD result with a weak key still
carries its "do not retry" value, and silently dropping historical rows would
overstate coverage while biasing the corpus toward recent runs — precisely the
distortion this corpus exists to prevent.

Dedup precedence when two rows collide on key: prefer the row with (1) more
populated fields, then (2) later `source_mtime`. Ties keep both, flagged
`key_confidence: "collision"`.

## Verdict taxonomy

The ten observed verdicts are recorded **verbatim**, normalized for case only:

```
WIN · DEAD · NOISE · LOSS · INCONCLUSIVE · VOID
COHERENCE_FAIL · NO_OP · DEAD_FILE · CROSS_ARCH
```

They are deliberately *not* collapsed to win/lose. `NOISE` (measurement did not
separate), `DEAD` (real, no effect), `INCONCLUSIVE` (insufficient rounds), and
`LOSS` (real regression) are four different instructions to a future reader.
Collapsing them destroys the property that makes this corpus more useful than a
recipes-that-worked KB.

## Harvest mechanism

`scripts/harvest_ledgers.py`, run from k9lin. Pull-based over ssh, read-only at
the source, idempotent, re-runnable as boxes accumulate rows.

```
for box in (k9lin, hipx, hiptrx):
    stream ledger / kernel-ledger / atlas / bod files   # read-only
    → normalize to attempts / kernel_static / bod rows
    → attach provenance; back-fill measurement_hash where canonical
      components exist, else compute fallback_key
merge + dedup
→ autoresearch/corpus/attempts.jsonl       (git-tracked, sorted, stable order)
→ autoresearch/corpus/kernels.jsonl        (git-tracked)
→ ar.db.ingest()  →  autoresearch/db/ar.db (gitignored, regenerable)
```

Git-tracked JSONL is the source of truth; `ar.db` is a regenerable index. This
preserves the contract already written into `autoresearch/db/schema.sql` and
keeps the corpus diffable and reviewable in PRs.

Rows are emitted in a stable sort order —
`arch, kernel, ts, COALESCE(measurement_hash, fallback_key)` — so re-harvesting
produces a minimal diff rather than a reshuffled file. The `COALESCE` matters:
`measurement_hash` is null on ~97% of historical rows, and sorting on a
nullable column would make ordering non-deterministic and defeat the point.

## Error handling

- **Box unreachable.** Harvest continues with the remaining boxes, exits
  non-zero, and reports which box was skipped. A partial corpus is committed
  only with an explicit `--allow-partial`; otherwise nothing is written, so a
  transient ssh failure cannot silently shrink the corpus.
- **Malformed JSONL line.** Counted, logged with `source_path:line`, skipped.
  A nonzero malformed count is reported in the summary. The survey found 0 bad
  lines across 3,374 hiptrx rows, so this should stay at zero.
- **Schema drift.** Unknown fields are preserved in an `extra` object rather
  than dropped, matching how `AtlasRow` already handles this.
- **Never writes to the source boxes.** Harvest is strictly read-only remotely.

## Testing

- Unit: normalization of each of the three observed row shapes → expected
  canonical row; `measurement_hash` back-fill correctness; composite-key dedup
  including the collision path.
- Idempotency: harvest twice against a fixed fixture tree → byte-identical
  output and zero new `ar.db` rows on the second ingest. This is the property
  the existing `test_ingest_idempotent` fixture already gestures at.
- Provenance: the 6-copy duplication case (121 vs 130 rows) resolves to the
  130-row source with the other five recorded as superseded.
- No-GPU: the whole suite runs without a GPU and joins `./scripts/no-gpu-ci.sh`.

## What this does not do

This spec delivers a consolidated, queryable corpus and nothing more. It does
**not** build embeddings, a similarity index, a verdict classifier, or a
pre-flight filter. Those depend on questions the corpus itself has to answer
first — chiefly whether the `(arch, kernel)` join has enough cell density to
support inference at 26 kernels x 5 archs, where spurious correlation is a real
risk and `mwu_dominance` is the natural filter for statistically-real rows.

The immediate payoff is smaller and certain: the falsification record stops
living only on two disposable boxes.

---

## As built (2026-07-23)

Implemented in `scripts/harvest_ledgers.py`. Five things diverged from the
design above; all five were forced by the real data.

**1. `/tmp` is harvested, not just `$HOME`.** hiptrx keeps 23 ledger files /
531 rows under `/tmp`. Excluding it lost exactly those rows (3,374 + 531 =
3,905, the independently-measured hiptrx total). Tmp data is the most
loss-prone on the fleet, which makes it the most important to capture, not the
least.

**2. Redline paths are walked, not pruned.** The design said exclude redline.
Blanket-pruning `.redline-work/` dropped ~1.3k *autoresearch* rows, because
redline worktrees contain full hipfire checkouts whose loop wrote genuine
ledgers (`~/.redline-work/hipfire-redline-kernel-oracle/autoresearch/ledger/`).
Only redline's own artifact roots (`redline-results|runs|cert`) are excluded.
The two corpora stay separate; the path is not the discriminator.

**3. Tables are `corpus_attempts` / `kernel_static` / `provenance`,** not a
shared `attempts`. The live loop's `attempts` table keys on canonical
`measurement_hash` with a UNIQUE constraint. 97% of harvested rows have no
canonical hash, so sharing the table would either drop them or mint look-alike
hashes. `corpus_key = COALESCE(measurement_hash, fallback_key)` lives in its
own table instead. `ar.db.ingest()` is deliberately not reused — it globs
`*.jsonl` in the directory given, which would have slurped `kernels.jsonl` and
`bod.jsonl` in as bogus attempt rows.

**4. `harvest_ts` is in `manifest.json`, not per row.** Stamping it on every
row made the corpus non-idempotent: every line diffed on every harvest, which
defeats git-tracking. Provenance also picks the lexicographically smallest
`(box, path)` rather than "latest mtime wins" — mtime ties were resolved by
`os.walk` order, which is not stable across runs. Both were caught by the
idempotency test, which now passes byte-identically.

**5. Timestamps come in two formats.** k9lin/hipx emit epoch ints; hiptrx emits
ISO-8601 (`2026-07-09T23:47:48Z`). Both are normalized.

### Result

```
attempts :   6,263 raw ->  932 unique (5,331 identical collapsed, 30 collisions kept)
kernels  :     321 raw ->  157 unique
bod      :      25 snapshots
archs    : gfx1010, gfx1030, gfx1100, gfx1151, gfx1201   (28 distinct kernels)
verdicts : DEAD 632 · WIN 107 · NOISE 87 · VOID 25 · INCONCLUSIVE 20 · LOSS 11 · +4
key conf : weak 698 · canonical 204 · collision 30
```

The 6,263 raw figure matches the independent pre-implementation count exactly.
Worst-case duplication was one row present **12 times** across the fleet.

### The finding that matters

**The `(arch, kernel)` join currently has 2 cells.** 57 outcome pairs, 2 static
pairs, 2 joinable — both gfx1201 `gemv_hfq4g256_moe_*`. The static corpus is
gfx1201-only and tiny (78 kernel-ledger rows), so the outcome/static join that
would make this a pre-flight filter is not yet viable.

This is a collection gap, not a modelling gap, and it is cheap to close:
`kernel_perf_instrument` already emits `kernel_atlas.v0` rows, and there are
~35,400 `.hsaco` on disk fleet-wide. Running it per arch would populate the
static side without any new GPU experiments. That is the next step — not a
classifier, which at 2 joinable cells would be fitting noise.

### Known asymmetry

`attempts.jsonl` keeps all 932 rows including the 30 collision-flagged ones;
`corpus_attempts` holds 902, since collisions share a `corpus_key` and are
`INSERT OR IGNORE`d. The JSONL is the source of truth and retains everything;
the DB is a keyed index. Query the JSONL when collisions matter.
