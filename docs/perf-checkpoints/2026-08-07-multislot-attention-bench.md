# 2026-08-07 — Multi-slot attention benchmark: batched vs sequential, `TILE_SIZE` sweep

Task 8 / SP1. Measures whether the multi-slot descriptor path built in Tasks
4-6 and verified correct in Task 7 is actually *fast*: one batched launch
serving several independent sequences, against the legacy loop-per-sequence
launch it replaces.

**Hardware: gfx1151 (Strix Halo / Radeon 8060S iGPU), the dev box. The
deployment target is gfx1201 (R9700). Every number below is dev-hardware
only — absolute GB/s, the TILE_SIZE=256 regression, and the LDS/tile
crossover may all shift on target. Only Task 7's correctness result and the
*existence* of a batching win are expected to transfer; magnitudes are not.**

Baseline (Task 1, batch-1, against BabelStream 223.9 GB/s Triad / 239.3 GB/s
Copy on this box):

| shape | achieved (batch-1) | % of Triad |
|---|---|---|
| 35B-A3B (nh=16 nkv=2 hd=256) | 49–62 GB/s | 22–28% |
| 27B (nh=24 nkv=4 hd=256) | 73–86 GB/s | 33–38% |

## Method

- `crates/rdna-compute/examples/q8_batched_attn_microbench.rs`, appended
  multi-slot section. Reuses `kv_slots::build_arena` / `build_tiles` (Task
  7's harness) for arena/descriptor/tile-list construction, so the benchmark
  and the correctness gate share one layout.
- Batched arm: one call to `attention_flash_q8_0_batched_masked_slots` over
  `n_slots` independent sequences via `KvSlotDesc` + `row_slot` addressing.
  Sequential arm: `n_slots` separate calls to the legacy
  `attention_flash_q8_0_batched_masked`, one slab per slot.
- **Both arms upload everything (arena, descriptors, positions, Q, per-slot
  slabs) before the timed region.** Only kernel launches are timed. This
  matters: the sequential arm launches `n_slots` kernels and would look
  artificially slow if it also paid upload cost the batched arm's one launch
  doesn't.
- One `device_synchronize()` per whole timed block (median of `ITERS=9` after
  `WARMUPS=5`), never per kernel — a per-op sync serializes work that would
  otherwise overlap and fabricates a GPU speedup that isn't real.
- Bytes for GB/s: `ctx × n_kv_heads × (head_dim/32) × 34 × 2` (K and V,
  Q8_0), summed over every slot in the batch. This is a *single kernel
  launch's* KV traffic — "layers" from the task brief's general formula is 1
  here, since this microbench times one attention call, not a multi-layer
  forward pass.
- Every configuration is preflighted through `kv_slots::preflight_alloc`
  (32 GiB R9700 budget + `MemAvailable` headroom on this no-swap box) before
  any upload, and every run goes through `scripts/run-bounded.sh` (cgroup
  `MemoryMax`, default 24 GiB) — see "Memory safety" below for why this is
  mandatory, not optional, on this box.

Commands:
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
footprint is O(1 configuration), not O(configurations swept).

**Observed footprint for the full default sweep** (`WARMUPS=5 ITERS=9`,
through `run-bounded.sh`, `HIPFIRE_MEM_CAP=24G` default): `/usr/bin/time -v`
reported **464 MB peak RSS** for the wrapped process tree; `MemAvailable`
moved from ~61 GiB to ~55–58 GiB during the run and recovered afterward; zero
OOM lines in `journalctl -k` across every run in this document. The
per-configuration analytical estimate (`preflight_alloc`'s own accounting)
tops out around 700 MB for the largest single configuration (`ragged`/
`uniform-max` at ctx=100,000 × 4 slots) — comfortably inside both the 24 GiB
cgroup cap and the 32 GiB R9700 target. **Caveat, stated in
`run-bounded.sh`'s own header:** amdgpu GTT pages are charged to the process
memcg on most paths but not provably all, so RSS is a strong signal, not a
proof — the cgroup backstop is what actually protects the desktop.

## Multi-slot sweep: batched vs sequential

`SLOT_CTX=32768` per slot, `TILE_SIZE=128` (shipped default), Q8_0 KV mode.
**Spec §2 criterion 2: batched must beat sequential at every `n_slots ≥ 2`.**

### 35B-A3B shape (nh=16 nkv=2 hd=256, GQA 8:1)

| n_slots | batched ms | batched GB/s | sequential ms | sequential GB/s | speedup |
|---|---|---|---|---|---|
| 1 | 0.719 | 49.6 | 0.611 | 58.3 | 0.85x |
| 2 | 1.251 | 57.0 | 1.321 | 54.0 | **1.06x** |
| 4 | 2.445 | 58.3 | 2.806 | 50.8 | **1.15x** |
| 8 | 4.513 | 63.2 | 5.676 | 50.3 | **1.26x** |

### 27B shape (nh=24 nkv=4 hd=256, GQA 6:1)

| n_slots | batched ms | batched GB/s | sequential ms | sequential GB/s | speedup |
|---|---|---|---|---|---|
| 1 | 0.998 | 71.5 | 0.885 | 80.6 | 0.89x |
| 2 | 1.756 | 81.2 | 1.822 | 78.3 | **1.04x** |
| 4 | 3.414 | 83.6 | 3.778 | 75.5 | **1.11x** |
| 8 | 6.452 | 88.4 | 7.855 | 72.6 | **1.22x** |

**Both shapes pass the criterion at every `n_slots ≥ 2` at the shipped
default (`TILE_SIZE=128`).** `n_slots=1` is expected to be at or slightly
below parity (0.85–0.89x): one batched launch and one sequential launch do
the same single-sequence work, and the batched path carries the extra
descriptor-indirection read the legacy path doesn't — the criterion is
explicitly `n_slots ≥ 2` for exactly this reason.

The win grows with `n_slots` (1.06x → 1.26x for 35B-A3B; 1.04x → 1.22x for
27B) and is larger for the narrower-head shape (35B-A3B) than the wider one
(27B) at the same `n_slots` — consistent with batching amortizing a
roughly-fixed per-launch overhead that is a larger fraction of a smaller
per-launch cost.

**Against the Task 1 ceiling:** batch-1 numbers here (49.6 / 71.5 GB/s) land
inside Task 1's reported batch-1 ranges (49–62 / 73–86 GB/s), which is a
useful methodology cross-check — this benchmark's single-sequence arm
reproduces the earlier baseline. At `n_slots=8`, batching lifts achieved
bandwidth to 63.2 GB/s (28.2% of Triad, 3.54x headroom remaining) and 88.4
GB/s (39.5% of Triad, 2.53x headroom remaining) respectively — batching
recovers some of the headroom Task 1 identified, but does not close it.
Closing the rest is scheduler/occupancy work beyond this task's scope.

## `TILE_SIZE` sweep

Same 35B-A3B shape, `WARMUPS=5 ITERS=9`, through `run-bounded.sh`.

| TILE_SIZE | n_slots=1 | n_slots=2 | n_slots=4 | n_slots=8 | ragged GB/s (useful) |
|---|---|---|---|---|---|
| 64  | 0.86x (39.1 GB/s) | **1.08x** | **1.36x** | **1.39x** | 39.9 |
| **128 (default)** | 0.85x (49.6 GB/s) | **1.06x** | **1.15x** | **1.26x** | 53.2 |
| 256 | 0.81x (59.5 GB/s) | **0.99x — FAILS** | *(run aborts)* | *(run aborts)* | — |

**`TILE_SIZE=256` fails spec §2 criterion 2 at `n_slots=2`, reproducibly.**
Four separate runs (two exploratory at default 3/5 warmup/iters, two at the
final 5/9 protocol) all land batched ≥ sequential at `n_slots=2` under
`TILE_SIZE=256`: 0.96x, 0.94x, 0.76x, 0.99x. Never once above parity. This is
not noise at the margin — every trial failed, by a spread that includes
comfortably-below-1.0 outcomes, not just a hair under 1.0. **Per the task
brief, this is reported as a finding, not tuned or dropped.** Because the
committed benchmark's `assert!` is the mandatory hard gate from spec §2
(unmodified — see `crates/rdna-compute/examples/q8_batched_attn_microbench.rs`),
the process aborts at `n_slots=2` under `TILE_SIZE=256`, so `n_slots=4/8` and
the ragged/crossover sections were not collected in that configuration.

**Diagnosis:** larger tiles (256 vs 128 vs 64) mean fewer, coarser-grained
tile launches — measurably higher raw batch-1 bandwidth (59.5 vs 49.6 vs 39.1
GB/s) because there is less per-tile overhead to amortize *within* a single
launch. But that is exactly the overhead that *batching* itself amortizes:
with fewer tiles, a single sequential launch is already closer to its own
floor, so stacking several sequences into one launch has less waste left to
remove. `TILE_SIZE=64` shows the mirror image — the lowest raw bandwidth at
`n_slots=1` (39.1 GB/s) but the *largest* batching win (1.39x at `n_slots=8`)
because small tiles leave the most per-launch overhead on the table for
batching to amortize.

**Chosen default: `TILE_SIZE=128` (unchanged from the pre-existing
value).** It is the only setting tested that (a) passes the mandatory
criterion at every `n_slots ≥ 2` for both GQA shapes and (b) sits between
64's larger-relative-win-but-lower-absolute-bandwidth and 256's
higher-absolute-bandwidth-but-broken-criterion. 64 is a legitimate
alternative if a future scheduler decision is willing to trade ~20% lower
absolute GB/s for a larger batching win — that trade is not evaluated here.
256 should not ship as the default while criterion 2 fails under it.

## LDS-vs-tile crossover, multi-slot

`TILE_SIZE=128`, 35B-A3B shape. The LDS decode kernel's grid is
`[n_heads, batch]` (thin — only 64–128 workgroups at these `n_slots`); the
tile kernel's grid is `[n_heads, max_tiles, batch]` (already parallel).

| ctx | n_slots | LDS ms | tile ms | winner |
|---|---|---|---|---|
| 2,048 | 1 | 0.139 | 0.084 | TILE |
| 2,048 | 4 | 0.167 | 0.110 | TILE |
| 2,048 | 8 | 0.248 | 0.211 | TILE |
| 8,192 | 1 | 0.534 | 0.150 | TILE |
| 8,192 | 4 | 1.639 | 0.574 | TILE |
| 8,192 | 8 | 3.069 | 1.054 | TILE |
| 14,000 | 1 | 0.876 | 0.267 | TILE |
| 14,000 | 4 | 3.335 | 0.936 | TILE |
| 14,000 | 8 | 6.644 | 1.822 | TILE |

**TILE wins everywhere tested, including `n_slots=1`, and the margin widens
sharply with both `ctx` and `n_slots`** (1.7x at ctx=2048/n_slots=1 → 3.6x at
ctx=14000/n_slots=8). The LDS kernel's cost scales roughly linearly with
`ctx × n_slots` (one thin `[n_heads, batch]` grid doing `O(ctx)` work per
row, serially per row in `n_slots`), while the tile kernel's `[n_heads,
max_tiles, batch]` grid keeps the device saturated as both dimensions grow.

**Recommendation for SP3 (scheduler):** the existing single-sequence
`LDS_CTX_LIMIT = 15000` router threshold (not changed by this task) is
already conservative for multi-slot batches — nothing here suggests raising
it. If anything, a multi-slot-aware router should consider routing to TILE
at a *lower* ctx threshold than the single-sequence router does, since the
tile kernel's advantage over LDS only grows with slot count and every real
serving batch has `n_slots > 1`. This benchmark did not sweep `ctx` below
2048 or `n_slots` at the boundary transition itself, so it cannot pin an
exact lowered threshold — that measurement is left to SP3.

## Ragged-batch waste

`max_tiles` is sized from the batch's **maximum** context; short slots still
launch that many tiles, which early-exit immediately once past their own
`seq_len`. Shape: `seq_lens = [1024, 4096, 32768, 100_000]` vs a uniform
`[100_000, 100_000, 100_000, 100_000]` control, `TILE_SIZE=128`.

| | wall ms | GB/s (useful-KV basis) | useful KV positions | tiles launched (as if max×4) | waste |
|---|---|---|---|---|---|
| ragged | 2.818 | 53.2 | 137,888 | 400,000 | **65.5%** |
| uniform-max | 7.450 | 58.4 | 400,000 | 400,000 | 0% (by construction) |

**65.5% of the tiles launched for the ragged batch do no useful work** —
they belong to the 100,000-context slot's tile count but are attributed to
the three much-shorter slots, which early-exit almost immediately. This is
not negligible: it is the majority of the launched grid. The ragged run
(2.818 ms) is far faster than the uniform-max run (7.450 ms) only because
the *wasted* tiles for the short slots do less real memory traffic even
though they are launched — the grid is oversized, not the bytes moved. A
scheduler that groups slots by similar context length before batching (or
pages/chunks context so `max_tiles` is bounded per-chunk rather than
per-batch) would recover a meaningful fraction of this — quantifying that
recovery is future scheduler work, not this task.

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
  fingerprint diff came back `LEGACY_BITWISE_IDENTICAL`.
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
  `SKIP` line and `continue`s on `None` rather than proceeding.
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
