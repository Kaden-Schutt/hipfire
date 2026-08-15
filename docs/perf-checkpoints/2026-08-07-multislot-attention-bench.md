# 2026-08-07 — Multi-slot attention benchmark: batched vs sequential, `TILE_SIZE` sweep

**CORRECTED 2026-08-07 (later same day).** An independent review re-measured
this benchmark and found the batched-vs-sequential comparison was not fair:
the sequential arm aliased K and V through one shared buffer while the
batched arm used two distinct buffers, understating the batching win and
inverting the `TILE_SIZE` conclusion. That bug is now fixed
(`crates/rdna-compute/examples/q8_batched_attn_microbench.rs`'s `bench_slots`
now uploads a distinct `slab_k`/`slab_v` per slot for the sequential arm,
mirroring the batched arm's `k_cache`/`v_cache`) and every number below is a
fresh re-measurement against the fixed benchmark, not the original numbers
with new arithmetic applied. **Where the corrected data changes a
conclusion, that is stated plainly, including where it is inconvenient** —
see "What changed" below for a compact summary, and the "Deviations" section
at the end for the full defect writeup.

Task 8 / SP1. Measures whether the multi-slot descriptor path built in Tasks
4-6 and verified correct in Task 7 is actually *fast*: one batched launch
serving several independent sequences, against the legacy loop-per-sequence
launch it replaces.

**Hardware: gfx1151 (Strix Halo / Radeon 8060S iGPU), the dev box. The
deployment target is gfx1201 (R9700). Every number below is dev-hardware
only — absolute GB/s, the `TILE_SIZE` recommendation, and the LDS/tile
crossover may all shift on target. Only Task 7's correctness result and the
*existence* of a batching win are expected to transfer; magnitudes are not.**

**This re-measurement pass ran on a shared box under real background load**
(`uptime` load average 8–9 throughout, from ~15 concurrent processes
unrelated to this benchmark — other agent sessions' QuestDB servers,
Firefox, Steam). Run-to-run variance was large enough to matter: at the
*shipped default* `TILE_SIZE=128`, 1 of 8 repeated trials at `n_slots=4`
came in at 0.95x (a **criterion-2 failure**, purely from system noise, not a
code regression — confirmed by immediately re-running the identical
configuration and getting 1.09–1.27x). At `TILE_SIZE=256`, 1 of 5 trials at
`n_slots=2` was 0.99x. **Every table below reports the median of repeated
trials with the observed range**, not a single run, specifically because a
single run on this box is not a reliable point estimate — the original
doc's single-run tables were part of what made the pre-fix conclusions look
more solid than the underlying measurement actually supported.

Baseline (Task 1, batch-1, against BabelStream 223.9 GB/s Triad / 239.3 GB/s
Copy on this box):

| shape | achieved (batch-1) | % of Triad |
|---|---|---|
| 35B-A3B (nh=16 nkv=2 hd=256) | 49–62 GB/s | 22–28% |
| 27B (nh=24 nkv=4 hd=256) | 73–86 GB/s | 33–38% |

## What changed, compactly

| # | Finding | Before (contaminated) | After (fixed) |
|---|---|---|---|
| C1 | Sequential arm aliased K==V (half the batched arm's distinct bytes) | understated the batching win | K and V are distinct buffers in both arms |
| C2 | `n_slots=1` "0.85–0.89x descriptor-indirection cost" | reported as a real cost | artefact of C1; corrected `n_slots=1` is ~parity (0.97–1.02x median across shapes/tile sizes) |
| C3 | `TILE_SIZE` default | 128 (256 "fails criterion 2") | **256 is fastest at every `n_slots` and its median trial passes criterion 2** — see caveats below before flipping the shipped default |
| I1 | GB/s omitted `partials` traffic | ~24% of real DRAM traffic uncounted at n=8 | included; achieved-bandwidth figures revised up |
| I2 | LDS-vs-tile recommendation | "consider a lower threshold" | TILE wins in 62/63 measured trials, including the smallest shape tested; the shipped `LDS_CTX_LIMIT=15000` is not supported by this data (not changed here) |
| I3 | Ragged-batch waste headline | 65.5% waste, no time-cost stated | time cost is 15.8–41% (median 24.2% over 7 trials), not 65.5% — that figure is the launched-grid early-exit fraction, a different quantity |
| I4 | Tile-size resolution duplicated 3x | drifted out of sync once, corrupting memory | single `kv_slots::attn_tile_size()`, three call sites now delegate to it |

## Method

- `crates/rdna-compute/examples/q8_batched_attn_microbench.rs`, appended
  multi-slot section. Reuses `kv_slots::build_arena` / `build_tiles` (Task
  7's harness) for arena/descriptor/tile-list construction, so the benchmark
  and the correctness gate share one layout.
- Batched arm: one call to `attention_flash_q8_0_batched_masked_slots` over
  `n_slots` independent sequences via `KvSlotDesc` + `row_slot` addressing,
  against `k_cache`/`v_cache` built as two distinct uploads of the same
  arena bytes. Sequential arm: `n_slots` separate calls to the legacy
  `attention_flash_q8_0_batched_masked`, against **`slab_k`/`slab_v` — two
  distinct per-slot buffers**, mirroring the batched arm exactly. (Before the
  fix, both calls received the same `slab` for K and V, so each sequential
  launch touched half the distinct bytes the batched arm did — at
  ctx=32768 that is 17.8 MB vs 35.6 MB per slot, and the aliased half fits
  this box's MALL cache while the batched arm's full working set does not.
  That is what made the sequential arm look artificially fast.)
- **Both arms upload everything (arena, descriptors, positions, Q, per-slot
  slabs) before the timed region.** Only kernel launches are timed. This
  matters: the sequential arm launches `n_slots` kernels and would look
  artificially slow if it also paid upload cost the batched arm's one launch
  doesn't.
- One `device_synchronize()` per whole timed block (median of `ITERS=9` after
  `WARMUPS=5`), never per kernel — a per-op sync serializes work that would
  otherwise overlap and fabricates a GPU speedup that isn't real.
- Bytes for GB/s: `ctx × n_kv_heads × (head_dim/32) × 34 × 2` (K and V,
  Q8_0), summed over every slot in the batch, **plus** the `partials`
  round-trip: `rows × n_heads × max_tiles × (2+head_dim) × 4` bytes written
  by the tile kernel and the same volume read back by the (unconditional)
  reduce kernel. Both are real DRAM traffic the benchmark's own kernels
  perform; omitting the second term understated real traffic by ~24% at
  n_slots=8/ctx=32768/TILE=128 in the original write-up. Since every shape
  in the `n_slots` sweep uses one `ctx` for every slot, total partials bytes
  are identical whether those rows arrive as one batched launch or as
  `n_slots` single-row sequential launches, so one formula covers both arms.
  "Layers" from the task brief's general formula is 1 here, since this
  microbench times one attention call, not a multi-layer forward pass.
- Tile size resolution: `rdna_compute::kv_slots::attn_tile_size()` — the
  single function every call site (this file's two sections, and the
  `launch_asym_flash_batched` kernel launcher in `attention.rs`) now shares,
  replacing three independent hand-copies of the same parse-and-validate
  logic. That duplication is what let a hardcoded `128` divisor drift out of
  sync and undersize the `partials` buffer, corrupting device memory (see
  "Deviations" below).
- Every configuration is preflighted through `kv_slots::preflight_alloc`
  (32 GiB R9700 budget + `MemAvailable` headroom on this no-swap box) before
  any upload, and every run goes through `scripts/run-bounded.sh` (cgroup
  `MemoryMax`, default 24 GiB) — see "Memory safety" below for why this is
  mandatory, not optional, on this box. The preflight total for `bench_slots`
  now counts **two** per-slot slabs (`slab_k` + `slab_v`) for the sequential
  arm instead of one, reflecting the C1 fix; this roughly doubles that one
  line item but stays a small fraction of the total (tens of MB against a
  32 GiB budget).

Commands (each run repeated 5–9 times for this correction; see per-table
notes for exact counts):
```bash
WARMUPS=5 ITERS=9 ./scripts/run-bounded.sh ./target/release/examples/q8_batched_attn_microbench
NH=24 NKV=4 HD=256 WARMUPS=5 ITERS=9 ./scripts/run-bounded.sh ./target/release/examples/q8_batched_attn_microbench
for ts in 64 128 256; do
  HIPFIRE_ATTN_TILE_SIZE=$ts WARMUPS=5 ITERS=9 ./scripts/run-bounded.sh ./target/release/examples/q8_batched_attn_microbench
done
```

## Memory safety (read before re-running this sweep)

While building this benchmark, the SP1 harnesses (this one and Task 7's)
drove **nine global OOM kills** on this dev box between 18:41 and 19:14 —
the user's applications (steamwebhelper ×4, teams-for-linux ×3, slack, a
Firefox tab), not the benchmark itself, which reported success. On Strix
Halo the GPU's GTT is system RAM and this box has **no swap**, so an
allocation overshoot doesn't degrade — it goes straight to the *global* OOM
killer, which picks victims by `oom_score`, not by culpability. A live `free`
does not show this; it only appears in `journalctl -k`.

Two layers now guard every number in this document:

1. **`kv_slots::preflight_alloc`** — called before every allocation in
   `bench_slots`/`bench_lds_path`, with the TOTAL bytes that call will hold
   live at once. Refuses (prints why, caller skips) anything over the 32 GiB
   R9700 budget or without headroom on this box's `MemAvailable`, and fails
   closed if `/proc/meminfo` is unreadable.
2. **`scripts/run-bounded.sh`** — every command below ran inside a
   `systemd-run --user --scope` cgroup with `MemoryMax=24G`,
   `MemorySwapMax=0`. An overshoot is killed *inside our scope* (verified:
   exit 137), not in the global OOM killer.

Both bugs that caused the incident are fixed in this file too: `bench_slots`
and `bench_lds_path` free every tensor they allocate before returning (they
run inside sweep loops; `GpuTensor` has no `Drop`), so the sweep's live
footprint is O(1 configuration), not O(configurations swept). Re-verified for
this correction pass across ~35 individual `run-bounded.sh` invocations:
zero OOM lines in `journalctl -k`, `MemAvailable` stayed in the mid-50s GiB
throughout.

## Multi-slot sweep: batched vs sequential

`SLOT_CTX=32768` per slot, `TILE_SIZE=128` (shipped default), Q8_0 KV mode,
**distinct K/V slabs in both arms (C1 fix)**.
**Spec §2 criterion 2: batched must beat sequential at every `n_slots ≥ 2`.**
GB/s now includes the `partials` round-trip (I1 fix).

### 35B-A3B shape (nh=16 nkv=2 hd=256, GQA 8:1) — median of 8 trials (7 at n_slots=8)

| n_slots | batched ms (med) | batched GB/s (med) | sequential ms (med) | sequential GB/s (med) | speedup median | speedup range |
|---|---|---|---|---|---|---|
| 1 | 0.760 | 58.0 | 0.753 | 58.5 | **1.00x** | 0.98–1.11x |
| 2 | 1.500 | 58.8 | 1.702 | 51.8 | **1.16x** | 1.09–1.24x |
| 4 | 2.854 | 62.4 | 3.299 | 53.5 | **1.21x** | 0.95–1.27x (1 of 8 below 1.0) |
| 8 | 4.886 | 72.2 | 6.655 | 53.0 | **1.36x** | 1.22–1.45x |

### 27B shape (nh=24 nkv=4 hd=256, GQA 6:1) — median of 5 trials (3 at n_slots=4/8)

| n_slots | batched ms (med) | batched GB/s (med) | sequential ms (med) | sequential GB/s (med) | speedup median | speedup range |
|---|---|---|---|---|---|---|
| 1 | 1.100 | 76.3 | 1.102 | 76.2 | **0.97x** | 0.95–1.01x |
| 2 | 2.028 | 82.8 | 2.269 | 74.0 | **1.09x** | 0.51–1.13x (2 of 5 below 1.0) |
| 4 | 3.744 | 89.7 | 4.472 | 75.1 | **1.21x** | 1.18–1.21x |
| 8 | 7.036 | 95.5 | 9.042 | 74.3 | **1.29x** | 1.11–1.39x |

**The batching win survives the fix and is now real at every `n_slots ≥ 2`
median, for both shapes, at the shipped `TILE_SIZE=128` default** — this
part of the original conclusion holds. **Two things changed materially:**

1. **`n_slots=1` is no longer below parity.** Corrected medians are 1.00x
   (35B-A3B) and 0.97x (27B) — both within noise of 1.0, not the 0.85–0.89x
   the pre-fix doc reported. **The "descriptor-indirection cost" explanation
   for that shortfall (C2) is retracted: it was measuring the K==V-aliasing
   artefact, not a real cost of the descriptor path.** With a fair sequential
   arm there is nothing left to attribute to descriptor indirection.
2. **The 27B shape's `n_slots=2` margin is not reliable on this box under
   load**: 2 of 5 trials landed below 1.0 (0.84x, 0.51x — the 0.51x trial in
   particular is a large outlier, most plausibly a scheduling/contention
   artefact from the concurrent load described above, since a repeat of the
   identical configuration immediately after came back at 1.13x). This is
   new information the original single-run doc could not have surfaced. It
   does not overturn "batching wins" as a headline (3 of 5 trials, and the
   median, still clear passes), but it means the 27B shape's `n_slots=2`
   margin is thinner than 35B-A3B's and worth re-confirming on an idle box
   before treating it as a hard guarantee.

**Against the Task 1 ceiling, corrected:** the KV-only-basis batch-1 rate
(stripping the partials term back out, for comparison with Task 1's KV-only
measurement) is 47.0 GB/s (35B-A3B) and 64.8 GB/s (27B) — both now landing
**slightly below** Task 1's reported ranges (49–62 / 73–86 GB/s), where the
original doc reported them landing inside (correctly for 35B-A3B at 49.6,
already marginally outside for 27B at 71.5 vs the 73 floor, a discrepancy
the original doc did not flag). The gap is consistent with this box's
current background load rather than a methodology change — the KV-only
formula itself did not change, only its inputs (fresh, noisier
measurements). Treat the "reproduces the Task 1 baseline" cross-check as
weaker evidence than the original doc implied, not as broken.

**Against Triad, using the corrected (partials-inclusive) bandwidth:** at
`n_slots=8`, batching lifts achieved bandwidth to 72.2 GB/s (32.3% of Triad,
3.10x headroom remaining) and 95.5 GB/s (42.7% of Triad, 2.34x headroom
remaining) for 35B-A3B and 27B respectively — both figures higher than the
uncorrected doc reported (28.2%/3.54x and 39.5%/2.53x) because they now
count traffic the kernels actually move. Batching recovers some of the
headroom Task 1 identified but does not close it; closing the rest is
scheduler/occupancy work beyond this task's scope.

## `TILE_SIZE` sweep

Same 35B-A3B shape, `WARMUPS=5 ITERS=9`, through `run-bounded.sh`, distinct
K/V slabs in both arms. Medians below are of 5 trials per `TILE_SIZE` (4 at
`n_slots=4/8` for `TILE_SIZE=256`, since one trial's `n_slots=2` fell below
1.0 and the mandatory `assert!` aborted that run before it reached
`n_slots=4/8` — the same behavior as before the fix, just triggered less
often now).

| TILE_SIZE | batched ms median: n=1 / n=2 / n=4 / n=8 | speedup median: n=1 / n=2 / n=4 / n=8 | criterion-2 trials passed |
|---|---|---|---|
| 64  | 0.961 / 1.783 / 3.159 / 5.790 | 1.02x / **1.20x** / **1.31x** / **1.51x** | 5/5 (n=2), 5/5 (n=4, one exactly 1.00x) |
| **128 (shipped)** | 0.760 / 1.500 / 2.854 / 4.886 | 1.00x / **1.16x** / **1.21x** / **1.36x** | 8/8 (n=2), 7/8 (n=4) |
| 256 | **0.630** / **1.279** / **2.377** / **4.530** | 1.02x / **1.10x** / **1.24x** / **1.30x** | 4/5 (n=2) |

**`TILE_SIZE=256`'s batched arm is the fastest measured at every `n_slots`
— roughly 7–13% faster than the shipped 128 default by `n_slots=8` (4.530 ms
vs 4.886 ms median here; an independent re-measurement using this same fix
found ~11% at this shape). This reverses the original doc's central
conclusion, which was built on the K==V-aliasing bug (C1): once the
sequential arm is fair, larger tiles are not just faster in isolation, they
also win the batching comparison at every slot count tested — not because
256 was ever slow, but because the bug made the *sequential* arm at 256
artificially fast, hiding 256's real advantage.**

**Caveat before flipping the shipped default:** `TILE_SIZE=256`'s
`n_slots=2` margin is the thinnest of the three settings — median 1.10x
against 64's 1.20x and 128's 1.16x, and 1 of 5 trials landed at 0.99x (a
criterion-2 failure). `TILE_SIZE=128` is not perfectly clean either (1 of 8
trials failed at `n_slots=4`), so this is not unique to 256, but 256's
margin is consistently the thinnest across both this run and the
independent re-measurement's own `n=2` figures (1.115x, 1.074x — also thin,
also passing). **Recommendation: re-derive with 256 as the new default,
since it wins on raw throughput at every measured point and its median
trial clears the mandatory criterion, but re-confirm on an idle box (no
concurrent load) before this is treated as final** — the noise observed in
this pass is large enough that a single bad trial in CI could look like a
regression that is not one.

`TILE_SIZE=64` remains a legitimate alternative if a future scheduler
decision is willing to trade the lowest absolute throughput for the largest
*relative* batching win (1.51x at n=8) and the most comfortable safety
margin at `n_slots=2` (1.20x median, 5/5 trials clearing 1.0 with room).

**Reproducibility, corrected:** the original doc cited "four separate runs
... 0.96x, 0.94x, 0.76x, 0.99x" at `TILE_SIZE=256`/`n_slots=2` as evidence
the pre-fix regression was not noise. Per `task-8-report.md`'s own
"Defect found and fixed" section, two of those four trials were taken
*before* the hardcoded-`TILE=128`-divisor `partials`-sizing bug was fixed,
while it was still live — that bug is documented in the same report as
having corrupted device memory in exactly this configuration
(`TILE_SIZE=256`, oversized rather than undersized in that direction, "harmless
to correctness but made the very first exploratory numbers untrustworthy").
**That citation is retracted rather than re-labeled**, because this
benchmark's own measured noise floor (~2% run-to-run under quiet
conditions, per the original doc; considerably more under the load this
correction pass ran under) means a pre-fix number cannot be safely
distinguished from a fix-era number after the fact — the only trustworthy
statement is what this correction pass measured fresh, above: 4 of 5
post-fix trials at `TILE_SIZE=256`/`n_slots=2` pass, 1 does not, median
1.10x.

## LDS-vs-tile crossover, multi-slot

`TILE_SIZE=128`, 35B-A3B shape, median of 7 trials per point (some trials
aborted earlier in the sweep on a criterion-2 failure and did not reach this
section — see per-point notes). The LDS decode kernel's grid is `[n_heads,
batch]` (thin — only 64–128 workgroups at these `n_slots`); the tile
kernel's grid is `[n_heads, max_tiles, batch]` (already parallel).

| ctx | n_slots | LDS ms (med) | tile ms (med) | ratio LDS/tile | winner (of 7 trials) |
|---|---|---|---|---|---|
| 2,048 | 1 | 0.143 | 0.089 | 1.6x | TILE 7/7 |
| 2,048 | 4 | 0.167 | 0.114 | 1.5x | TILE 7/7 |
| 2,048 | 8 | 0.259 | 0.214 | 1.2x | TILE 6/7, LDS 1/7 |
| 8,192 | 1 | 0.547 | 0.147 | 3.7x | TILE 7/7 |
| 8,192 | 4 | 1.795 | 0.633 | 2.8x | TILE 7/7 |
| 8,192 | 8 | 3.217 | 1.239 | 2.6x | TILE 7/7 |
| 14,000 | 1 | 0.963 | 0.278 | 3.5x | TILE 7/7 |
| 14,000 | 4 | 3.434 | 1.122 | 3.1x | TILE 7/7 |
| 14,000 | 8 | 6.926 | 2.037 | 3.4x | TILE 7/7 |

**TILE wins 62 of 63 measured trials across every shape tested, including
`ctx=2048, n_slots=1` — the smallest, thinnest-margin point, still won by
TILE 7/7 with a comfortable 1.6x ratio.** The one LDS win (1 of 7 trials at
`ctx=2048, n_slots=8`, where the median margin is already the narrowest in
the table) is consistent with system noise, not a genuine crossover — this
data does not identify any tested point where LDS is actually the better
choice.

**This flatly contradicts the shipped router.** `crates/hipfire-runtime/src/llama.rs:2637`
routes every request with `max_ctx_len < LDS_CTX_LIMIT = 15000` to the LDS
kernel — that covers all nine points in the table above. Both paths reach
the identical kernel family (`attention_q8_0_kv_batched_masked` →
`..._slots` for LDS; `attention_flash_q8_0_batched_masked` →
`..._slots` for tile) through the same `launch_asym_flash_batched`
machinery, so this is not a porting regression introduced by SP1's
multi-slot work — the tile kernel was simply always faster than the LDS
kernel at every shape this benchmark tested, both before and after this
fix. **Stated plainly: this evidence suggests the LDS path may never be the
right choice at these shapes.** That is a stronger claim than the original
doc's "if anything, consider a lower threshold," and the data supports the
stronger claim.

**This is not a mandate to change `LDS_CTX_LIMIT`, and it is not changed
here.** Two real limits on what this data can support: (1) gfx1151 only —
the deployment target is gfx1201, and the LDS kernel's relative cost could
differ there (different LDS bank/latency characteristics are exactly the
kind of thing that does not port); (2) nine datapoints at one GQA shape,
swept over `ctx ∈ {2048, 8192, 14000}` and `n_slots ∈ {1, 4, 8}` — this does
not establish the crossover ctx (if one exists below 2048) or behavior at
shapes this benchmark didn't sweep (different `head_dim`, different GQA
ratio, tree-verify batches, which are explicitly out of scope for the
multi-slot descriptor path per `attention.rs`'s own assertion). Changing a
shipped constant on this evidence would be exactly the kind of dev-box
over-fit this document's own hardware caveat warns against. **Recommendation
for SP3 (scheduler), unchanged in substance from before, now stated with
more confidence: this data gives no support for the LDS path at any shape
tested, and a future scheduler decision should treat "route sub-15000 to
LDS" as unproven rather than merely conservative — but confirming and
possibly changing the constant is SP3's work, on target hardware, not this
task's.**

## Ragged-batch waste

`max_tiles` is sized from the batch's **maximum** context; short slots still
launch that many tiles, which early-exit immediately once past their own
`seq_len`. Shape: `seq_lens = [1024, 4096, 32768, 100_000]` vs a uniform
`[100_000, 100_000, 100_000, 100_000]` control, `TILE_SIZE=128`. This
section measures the **batched arm only** (the sequential arm is not part of
this comparison), so it is unaffected by the C1 fix; the numbers below
differ from the original doc purely because they are a fresh 7-trial sample
on a noisier box, not because anything about this measurement changed.

**Lead with the time cost, not the waste fraction — they answer different
questions and the waste fraction is 6x larger than the actual cost:**

| | wall ms (median of 7) | time per 1000 useful positions (median) | time cost vs uniform |
|---|---|---|---|
| uniform-max | 7.621 | 20.03 µs | — (control) |
| ragged | 3.315 | 24.63 µs | **+24.2% (median; range +15.8% to +41.0% across 7 trials)** |

This is the number that answers "how much does raggedness actually cost":
**a ragged batch takes ~24% longer per useful position than a uniformly-long
one would, not 6x longer.** The independent review's own clean-room
re-measurement (one trial, not under this box's current background load)
found +9.7% (18.63 µs uniform vs 20.44 µs ragged) — inside this pass's
observed range but at the low end, consistent with a quieter machine. Both
figures agree on the substance: the real cost is a modest double-digit
percentage, not the majority-of-launched-grid figure below.

**Separately — a different, still-true quantity — the fraction of the
*launched grid* that does no useful work:**

| | GB/s (useful-KV+partials basis) | useful KV positions | tiles launched (as if max×4) | waste |
|---|---|---|---|---|
| ragged | 78.2 (median) | 137,888 | 400,000 | **65.5%** |
| uniform-max | 71.1 (median) | 400,000 | 400,000 | 0% (by construction) |

**65.5% of the tiles launched for the ragged batch do no useful work** —
this figure is unchanged and still correct as a description of the launched
grid: they belong to the 100,000-context slot's tile count but are
attributed to the three much-shorter slots, which early-exit almost
immediately. **What changes is the conclusion drawn from it.** The original
doc led with 65.5% as the headline and did not state the time-cost figure
at all; read on its own, 65.5% implies a much larger performance problem
than the ~24% actually measured, because a launched-but-early-exiting tile
is cheap (a masked-off workgroup that reads almost nothing), not free but
nowhere near as expensive as a useful one. **Anyone sizing scheduler work
against the 65.5% figure alone would over-invest by roughly 3–6x relative to
the real time cost.** A scheduler that groups slots by similar context
length before batching (or pages/chunks context so `max_tiles` is bounded
per-chunk rather than per-batch) would still recover something — but the
ceiling on that recovery is ~24% of ragged-batch wall time, not 65.5%.
Quantifying the achievable recovery is future scheduler work, not this task.

## Deviations from the brief, stated explicitly

- **`bash for ts in 64 128 256; do ... cargo run ...; done` (brief Step 4,
  literal form) was not run as written.** Each iteration was instead run
  through `./scripts/run-bounded.sh ./target/release/examples/...` (a
  pre-built binary, not `cargo run`), because the mandatory memory gate
  (added mid-task after nine global OOM kills — see "Memory safety" above)
  requires the wrapper, and `cargo run`'s own build-then-exec makes the
  wrapped process tree harder to bound cleanly. Behavior is otherwise
  identical; the env var and binary are the same.
- **The legacy-fingerprint check (`scripts/attn_legacy_baseline.sh`) was run
  *without* `run-bounded.sh`,** unlike every other command in this task.
  `run-bounded.sh` prints its own banner lines to stdout before exec'ing the
  wrapped command; those lines are not filtered by
  `attn_legacy_baseline.sh`'s internal `grep`, so wrapping it corrupts the
  exact-text diff against the committed fingerprint
  (`scripts/attn_legacy_baseline.beta.txt`) with two spurious leading lines.
  This script's shapes (max `CTX=8192`, `N=256`) are small and were not
  implicated in the OOM incident, so running it unwrapped is safe; the
  fingerprint diff came back `LEGACY_BITWISE_IDENTICAL`, re-verified after
  the C1/I1/I4 fixes in this correction pass.
- **`time` (the brief's closure) is a module-level `fn time_ms` here, not a
  `main()`-local closure.** The brief's Step 2 snippet calls it from
  `bench_slots`, a free function — a closure defined inside `main()` cannot
  be reached from a free function in Rust. `main()`'s own benchmarks build a
  thin local closure over `time_ms` so their call sites are byte-identical
  to before.
- **The brief's Step 3 budget assertion was superseded mid-task** by
  `kv_slots::preflight_alloc` (added in commit `f267757a`, after the OOM
  incident) rather than the hand-rolled `assert!` the brief specifies.
  `preflight_alloc` does the same 32 GiB check plus a `MemAvailable`
  headroom check the brief's snippet did not have, and — critically — its
  contract is "skip the configuration," not "panic the whole binary," which
  the mid-task memory-safety requirement needed. `bench_slots` /
  `bench_lds_path` return `Option` and every sweep loop in `main()` prints a
  `SKIP` line and `continue`s on `None`.
- **A hardcoded `TILE=128` divisor pre-existing in this file's original
  single-shape section (line ~130, used to size the `partials` buffer) was
  fixed to read `HIPFIRE_ATTN_TILE_SIZE`, matching the new
  `launch_asym_flash_batched` resolution.** Found empirically: sweeping
  `TILE_SIZE=64` without this fix crashed with `hipDeviceSynchronize: an
  illegal memory access was encountered` — not in the section that owns the
  hardcoded constant, but downstream in the (unrelated) new multi-slot
  section, because the undersized `partials` buffer had already corrupted
  device memory. This file is in the brief's own "Modify" list, so the fix
  is in scope; it is not a change to kernel behavior, only to this
  benchmark's own buffer sizing.
- **Correction pass (this section, added after the initial Task 8 delivery):**
  fixed the C1 K==V-aliasing bug in the sequential arm (`bench_slots`'
  `slabs` now builds `slab_k`/`slab_v` as distinct uploads, not one shared
  `slab` passed twice); added the `partials` round-trip to every GB/s
  calculation (I1); extracted `kv_slots::attn_tile_size()` as the single
  source of truth for the `HIPFIRE_ATTN_TILE_SIZE` parse-and-validate logic
  that was previously hand-copied at three call sites (`attention.rs`'s
  `launch_asym_flash_batched` and two spots in this file) — the exact
  duplication pattern that caused the hardcoded-`TILE=128` corruption bug
  described in the bullet above (I4). Re-ran the full correctness harness
  (`test_batched_attn_slots`, `268/268` groups, `ALL SHAPES PASS`) and the
  legacy fingerprint diff (`LEGACY_BITWISE_IDENTICAL`) against the fixed
  code before re-measuring performance.
