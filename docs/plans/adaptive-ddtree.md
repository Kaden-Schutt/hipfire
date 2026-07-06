# Draft-Confidence-Adaptive DDTree Spec-Decode

Status: design note (POC-ready). No retraining, no kernel work. Gated behind
`HIPFIRE_DDTREE_ADAPTIVE` (default off → byte-identical to today).

## 1. Problem

DDTree branches **uniformly** up to a fixed budget: every popped node forks the
next-rank sibling whenever `rank + 1 < topk`
(`crates/hipfire-runtime/src/ddtree.rs:269`), regardless of how confident the
draft is at that position. On confident content the draft's top-1 is already
right, so those siblings are dead weight — verify cost paid for nodes the target
will never take. That uniform overhead (~1.4× linear-DFlash per cycle) is what
makes the tree throughput-bound. Genre sweep (27B-3.6, ddtree-batched vs
linear-DFlash, this session):

| genre  | linear τ / tok/s | tree τ / tok/s | verdict |
|--------|------------------|----------------|---------|
| code   | **8.77**         | 7.15           | linear wins — branches wasted (confident draft) |
| reason | 4.69 / 106.8     | **5.71** / 91.5 | tree out-accepts but **loses tok/s** to overhead |
| prose  | 1.15             | **1.70**       | tree out-accepts; both ≪ AR (~44.5 tok/s) |

The tree's acceptance edge is real on uncertain content; the uniform branching
cost eats it everywhere else. Goal: **chain-speed on confident content,
tree-acceptance on uncertain content, AR floor where neither helps.**

## 2. Policy

### 2a. Per-position branch gate (sparse branching)

The draft-confidence signal is **already computed and in-scope** at the branch
site. `topk_from_logits` (`ddtree.rs:626`) returns `top_log_probs[r*topk + j]`
= per-position normalized log-prob of each top-K token; the batched path gets the
same array on-device from `run_dflash_draft_for_topk_gpu`
(`speculative.rs:4150`). At the sibling push (`ddtree.rs:269`) the parent depth
`d` is in scope and both `top_log_probs[(d-1)*topk + rank]` and
`...+ rank+1` are **already read on lines 271-272** to compute `sibling_logw`. So:

- **top-1 prob** `p1 = exp(top_log_probs[(d-1)*topk + 0])`
- **top1/top2 margin** `m = top_log_probs[(d-1)*topk+0] - top_log_probs[(d-1)*topk+1]`

are free (zero extra GPU work, zero extra D2H).

**Rule** — replace the constant width bound at `ddtree.rs:269` with a per-depth
effective width `k_eff(d)`:

```
push sibling (rank+1) ONLY IF  rank + 1 < topk            // existing cap, kept
                          AND  p1 < CONF_FLOOR            // confident ⇒ chain
                          AND  top_log_probs[next] > SIB_FLOOR  // sibling worth it
```

- **Confident** (`p1 ≥ CONF_FLOOR`, e.g. 0.9): never push a sibling ⇒ `k_eff=1`
  ⇒ the node chains exactly like linear DFlash. This is the code-path win.
- **Flat/uncertain** (`p1 < CONF_FLOOR`): allow siblings up to `topk`, throttled
  by the marginal floor `SIB_FLOOR` so we widen only toward candidates with real
  mass. This preserves the prose/reason acceptance edge.

`CONF_FLOOR` and `SIB_FLOOR` are thresholds on the **per-position** distribution,
which is exactly the axis the existing `HIPFIRE_DDTREE_LOGW_CUTOFF` lever
(`speculative.rs:424`) does **not** cover — that one prunes on *cumulative*
root-to-node logw, cycle-globally. The two **compose** (both still fire); the new
gate is per-position fan-out, the cutoff is whole-tree depth.

This maps onto budget/topk as an *emergent* shape, not a per-cycle scalar: width
collapses to 1 on confident positions, so the global heap (ordered by cumulative
logw) spends the freed budget deepening/widening the uncertain positions —
chain on code, bushy on prose, automatically, under the same `budget` cap
(`ddtree.rs:227`).

### 2b. AR-floor gate (rolling-τ collapse → drop the tree)

When even branching can't out-accept a plain chain (prose: tree τ1.70 vs AR
~44.5 tok/s), spec-decode overhead is pure loss. Gate on **rolling τ**: the demo
already maintains `accepts_window` (VecDeque, window=8,
`dflash_spec_demo.rs:1232`) and computes `win_tau` per cycle
(`dflash_spec_demo.rs:1796`). Rule, modeled on the adaptive-B controller
(`dflash_spec_demo.rs:1573-1599`):

```
if win_tau < TAU_FLOOR  (after ≥4-cycle warmup, with hysteresis/cooldown)
    → route this cycle to linear DFlash (spec_step_dflash), or plain AR
else
    → route to spec_step_ddtree_batched
```

`TAU_FLOOR ≈ 1.3` (just above the prose τ ceiling). Warmup + hysteresis are
mandatory or it flaps at the boundary, exactly like adaptive-B's
`accepts_window.len() >= 4` + `ADAPTIVE_B_COOLDOWN`.

## 3. Hooks (cite key_locations)

| piece | location | change |
|-------|----------|--------|
| **branch-decision site** | `ddtree.rs:269` (`if rank+1 < topk`) | AND-in the `p1 < CONF_FLOOR && top_log_probs[next] > SIB_FLOOR` gate. Confidence values already in registers (`ddtree.rs:271-272`). |
| **builder signature** | `ddtree.rs:176-183` (`build_ddtree_tree_with_cutoff`) | add two `f32` params (`conf_floor`, `sib_floor`) next to `logw_cutoff`; `NEG_INFINITY`/`-INF` sentinel = disabled (mirror `ddtree_logw_cutoff()` precedent). |
| **draft-entropy source** | `ddtree.rs:626` (`topk_from_logits`, CPU per-path) + `speculative.rs:4150` (`run_dflash_draft_for_topk_gpu`, GPU top-K) | none — both already produce `top_log_probs`. **Note `k==1` fast path** (`ddtree.rs:643`) returns no rank-2, so margin gating needs `topk ≥ 2`. |
| **env parser** | `speculative.rs:424` (`ddtree_logw_cutoff`) | add sibling `ddtree_branch_conf()` reading `HIPFIRE_DDTREE_ADAPTIVE` / `HIPFIRE_DDTREE_CONF_FLOOR` / `HIPFIRE_DDTREE_SIB_FLOOR`. |
| **three build call-sites (byte-identical)** | `speculative.rs:4476` (ddtree), **`4797`** (batched — the one benched), `5423` (path_c) | thread the two new args through all three or the build breaks. |
| **AR-floor / mode dispatch** | demo `dflash_spec_demo.rs:1635` (`let step = if ddtree_enabled {…}`), consuming `win_tau` from `:1796` / `accepts_window` `:1232`; production analog `daemon.rs:4607`. | turn the static `ddtree_enabled` selector into a per-cycle `cycle_mode` chosen from rolling τ, branching to `spec_step_dflash`/AR. |
| **`_adaptive_b` flag** | `dflash_spec_demo.rs:1573-1599`, consumed **only** by `spec_step_dflash` via `block_override` (`:1723`) | leave alone. It adapts chain **length** off acceptance for the *linear* path; orthogonal axis to per-position fan-out. The tree arms ignore `block_override`. |

Telemetry surface to extend: `record_ddtree_meta_nodes` / `read_ddtree_meta_stats`
(`speculative.rs:442/458`) already report per-cycle tree size — sparser trees show
smaller `num_nodes()` (expected); add a rolling-τ accumulator here for the
AR-floor gate's serve-path analog.

## 4. POC scope (behind `HIPFIRE_DDTREE_ADAPTIVE`, default off)

Minimal, all reuse, no kernels, no retraining:

1. `ddtree.rs`: add `conf_floor: f32, sib_floor: f32` params to
   `build_ddtree_tree_with_cutoff` (`:176`); AND the gate into the sibling push
   (`:269`). Sentinel `-INF`/`+INF` ⇒ unchanged behavior.
2. `speculative.rs`: add `ddtree_branch_conf()` env parser next to `:424`; pass
   the two values through call-sites `:4476`, `:4797`, `:5423`. When
   `HIPFIRE_DDTREE_ADAPTIVE` unset → pass sentinels (no-op).
3. **(stretch, separate commit)** `dflash_spec_demo.rs`: per-cycle AR-floor at
   `:1635`, reusing `win_tau`/`accepts_window`, behind the same env flag.

Step 1+2 is the core experiment and is the smallest landable unit. Default-off
guarantees `cargo build --release` and existing gates are byte-identical.

**Invariant guards (must hold):**
- Only ever **narrow** below `topk` — never exceed it; `top_log_probs` has only
  `topk` columns and the GPU path asserts `topk ≤ 8` (`speculative.rs:4168`).
- **Suppress siblings by not pushing**, never by reordering. Heap-pop descending
  cumulative-logw order is load-bearing for `select_main_path` (`ddtree.rs:393`)
  and Path C `enumerate_branches` (`ddtree.rs:456`) — "smallest node index under
  a parent = highest-logw child" (`ddtree.rs:370-392`).
- Per-position width composes with `budget` (`:227`) and `logw_cutoff` (`:235`),
  doesn't replace them — validate the **emergent** tree shape, not the local gate.

## 5. Validation plan

Reuse `scripts/sweep_dflash_full.sh` harness; per CLAUDE.md **q8 KV, `max=256`,
`--no-chatml`, byte-identical committed prompts with recorded md5,
`prompt_normalize=true`**. Three arms per genre: **adaptive** (`HIPFIRE_DDTREE_ADAPTIVE=1`)
vs **fixed-tree** (today's uniform branching) vs **linear-DFlash**.

| genre  | prompt | metrics | expected outcome |
|--------|--------|---------|------------------|
| code   | `benchmarks/prompts/lru_cache_pep8_strict.txt` (PEP-8 strict) | decode tok/s, τ | adaptive **≈ linear** (confident ⇒ chains; closes the 8.77 vs 7.15 gap) |
| reason | committed reasoning prompt | tok/s, τ | adaptive keeps tree's τ edge (5.71) **but recovers tok/s** toward 106.8 |
| prose  | committed prose prompt | tok/s, τ | AR-floor trips ⇒ adaptive **≈ AR/linear** (no tree overhead paid) |

**Gates (blocking, per CLAUDE.md):** sparser branching changes acceptance, so
re-run `scripts/coherence-gate-dflash.sh` — the per-position gate lands in the
**shared** builder so it affects all three variants (path_c, batched, per-path);
check coherence across all three. Tier-1/Tier-2 attractor checks are mandatory
(tree-mode RoPE phase skew has historically produced single-token attractors that
pass naive benches; tight stddev is *suspicious*, not reassuring). The
acceptance edge on prose/reason is the thing being preserved — confirm it
**survives** the width gate before claiming the code-path speedup. Also run
`scripts/serve-multiturn-gate.sh` if the AR-floor mode-flip lands in serving
(per-cycle DDTree↔linear↔AR switches must leave both slots at
`pos+committed.len()` with consistent DN/KV state — #462 class).

**Threshold sweep:** `CONF_FLOOR ∈ {0.85, 0.9, 0.95}`, `SIB_FLOOR` swept against
`record_ddtree_meta_nodes` mean tree size; pick the point where code tree-size →
≈chain while reason/prose tree-size and τ hold.
