# 2026-08-15 — DFlash pays a full-context tax on every warm serve turn

**Status: diagnosis banked, repair reverted.** No fix has landed. The root
cause below is supported by measurement and by the code; the two repair
attempts described at the end were both reverted, and the second one failed in
a way that is itself evidence.

Measurements: [`docs/perf-checkpoints/2026-08-15-dflash-3.6-27b-awq-four-arm-matrix.md`](../perf-checkpoints/2026-08-15-dflash-3.6-27b-awq-four-arm-matrix.md)
and [`…-matched-vs-mismatched-draft-27b.md`](../perf-checkpoints/2026-08-15-dflash-matched-vs-mismatched-draft-27b.md).
All on `hiptrx` (4× R9700, gfx1201) against the digest-verified 3.6-27B AWQ
trunk and the committed `benchmarks/prompts/session_coding.json` chain.

## Symptom

On a multi-turn serve session, DFlash decode falls ~60% from ctx 2.5k to 15.6k
while AR stays flat, and **acceptance does not move**.

| | with prefix cache | cache disabled |
|---|---:|---:|
| mean τ | 2.44 | 2.47 |
| mean decode tok/s | 31.1 | **34.7** |
| turns beating AR | 3/8 | **5/8** |
| t6 (shortest generation) | **10.5** | **40.3** |

τ is unchanged to +1.2% while decode moves 11.4% — and 284% on t6. Whatever
this is, it is not a speculation-quality effect.

The same signature appears in prefill. Fitting warm-turn prefill against new
tokens alone leaves large residuals; adding a context term collapses them:

```
A   ms = 71.7 + 2.170*new                  RSS = 4195
B   ms = -30.5 + 3.202*new + 0.00669*ctx   RSS =   22     (-99%)
```

At ctx 14k that context term is **94 ms on a turn with 33 new tokens**. A
prefix-cache hit is supposed to cost `O(new)`.

*(Eight points, and `ctx` correlates with turn index, so "grows with context"
and "grows with turn count" are not fully separated. A fixture with
non-monotonic context would settle it.)*

## Root cause

**The prefix cache restores the target's KV. It does not restore the drafter's
projections, and nothing else does either on a warm turn.**

Three facts, each verifiable in the tree:

1. `TargetHiddenLog` (`hipfire-runtime/src/dflash.rs:~556`) tracks the
   drafter's coverage **separately** from the target's KV, via
   `proj_cached_rows` and `full_cached_rows`.

2. The `full_cached_rows` field comment states the dependency outright:

   > *"Separate K/V-fill watermark for the windowed mode's last
   > (full-attention) layer. Rows older than `l − swa_w` are not resident in
   > the proj ring, so the last layer's fill for them runs in the post-seed
   > backfill (host shadow), tracked here."*

3. That post-seed backfill — `draft_seed_backfill` — was called under
   `if !cache_hit` in `hipfire-arch-qwen35/src/dflash_spec.rs`. **Cold prefill
   only.**

So on a warm turn the draft ring has wrapped past `swa_w`, the backfill that
would refill it is skipped, and `target_hidden_host` was cleared at the top of
`prefill` — leaving *no source at all* for rows older than the window. The
draft re-derives the shortfall, and the shortfall is the context length.

Every observation follows from this:

| observation | explanation |
|---|---|
| τ unchanged | the rows still arrive, just expensively — correctness never depended on this |
| AR unaffected | AR has no drafter, so no projection ring |
| cost grows with `L` | the gap between `full_cached_rows` and `L` *is* the context |
| disabling the cache "fixes" it | every turn becomes a cold prefill, so the backfill runs |
| t6 worst at 4× | shortest generation, so a per-turn cost amortises over fewest tokens |
| `O(ctx)` prefill term | same re-derivation, measured at the prefill boundary |

This is why DFlash looks unusable on serve unless the whole conversation is
re-prefilled: re-prefilling is currently the only path that repopulates the
drafter.

## Repair attempt 1 — gap backfill from a cumulative host shadow (reverted)

Run the backfill every turn from `thlog.full_cached_rows()` instead of only on
a miss, sourcing rows from a `target_hidden_host` retained across turns rather
than cleared. Attractive because it allocates nothing: `k_full_cached` /
`v_full_cached` are already `[w_full × kvd]`, and the host `Vec` is already
built at `ctx_capacity * dim` — `clear()` only dropped the length.

**It failed, and the failure is the useful part.** A desync guard added with it
(`target_hidden_host.len() == start_pos * ne * h` on entry to the suffix
seeder) fired on turn 3, killing two turns loudly rather than projecting the
draft from stale rows.

The guard was right. "Clear on miss, append the suffix on hit" does **not**
keep the shadow mirroring `[0, pos)` — decode-committed tokens advance the
target's position without passing through the prefill seeder. After a turn
generates ~2,000 tokens the target sits far ahead of the shadow, and the next
turn's `start_pos` no longer matches.

Making that approach correct means appending accepted-token hidden rows from
the **spec accept loop**, not just from prefill. That is a hot-path change with
a per-cycle D2H unless it can reuse hidden already resident on GPU
(`speculative.rs` already distinguishes a fast GPU-scatter path from a
`ctx_slice` CPU-shadow path for exactly this reason). Materially larger than it
first appears.

Turns that did run were unchanged from baseline (47.8/44.4/28.5/28.5 vs
48.0/44.5/28.5/27.7), consistent with the backfill never receiving a valid
shadow.

## Repair attempt 2 — carry the watermark instead of re-deriving the rows (NOT attempted)

**This is the recommended next probe, and it is cheap.**

Attempt 1 assumed the drafter's rows must be *rebuilt* on a warm turn. That may
be false. `reset_upload_tracking()` is already skipped on a cache hit
specifically so the draft "reuses the cached `[0..start_pos]` projections", and
`k_full_cached` is a persistent GPU allocation that is not freed between turns.
If the ring's contents are in fact still resident and valid, the only thing
actually lost across the turn boundary is the **watermark** — and the fix is to
persist `full_cached_rows` alongside the KV rather than re-derive the rows it
describes.

That would be a bookkeeping change, not a data-movement one: no host shadow, no
D2H, no hot-path edit.

The probe that decides it, in order:

1. Instrument `full_cached_rows` at entry and exit of `prefill` across the
   8-turn chain. If it is non-zero and tracks `LCP` on warm turns, the rows are
   resident and attempt 2 is the whole fix.
2. If it resets to 0, find what resets it — that is then the actual bug, and it
   is likely a single line.
3. The load-bearing question either way: **has the ring wrapped?** `w_full` is
   sized to the full supported context, so for sessions under `w_full` no wrap
   can have occurred and the rows must still be there. If they are, attempt 1
   was solving a problem that does not exist below `w_full`, and only sessions
   exceeding it need row rebuilding at all.

Point 3 is the crux and can be answered by reading `w_full` at load against the
session's context — no GPU run required.

## Related, and worth separating

`w_full` being sized to the entire context means the drafter's last layer
already carries **2.58 GB of the 2.66 GB windowed draft KV at ctx 128k**. That
is a real cost and is independent of this bug; it was the target of a separate
PFlash-shim experiment, also reverted. The shim's premise — that reducing draft
context would pay for itself — was never tested, because acceptance under a
*reduced* draft context was never measured. That remains the open question if
anyone revisits it.

## What is on the branch

Nothing from either repair. `arch/saddle` is at the pre-attempt state; both
were reverted and the workspace builds clean. What survives is this document,
the perf-checkpoints, the committed `session_coding.json` fixture, and the
`serve_harness.py` default-path fix that made the fixture reproducible.
