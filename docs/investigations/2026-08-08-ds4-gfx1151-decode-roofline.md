# DS4 gfx1151 DSpark decode roofline: tiled LDS gather and the bandwidth ceiling

Date: 2026-08-08
Branch: `ds4-beta-staging`
Host/device: `hipx`, Radeon 8060S, `gfx1151`, ROCm 7.14

This doc records two things on a single `gfx1151` (Strix Halo, 96 GiB):
(1) the measured end-to-end acceptance of the tiled LDS top-K gather, and
(2) the post-fix GPU-time roofline that bounds further kernel work at k6.
All throughput figures in §1 are from the acceptance oracle
(`scripts/serve_harness.py`). All kernel-time shares in §3 are from a
profiling diagnostic (`rocprofv3` over `examples/dspark_bench`) and are
explicitly **not** acceptance numbers — see §3 caveat.

## 1. The shipped win: tiled LDS top-K gather

**Oracle:** `scripts/serve_harness.py` on the golden k6 fixture. This is the
acceptance oracle; `dspark_bench` is not used for throughput in this section.

**Fixture:**

- Model: `/home/kaden/ds4-gfx1151-evidence/2026-08-03-ds4-dspark-pm4-canary/model-e8/deepseek-v4-flash-0731.mq2r`
- KV: q8, kv-backend contiguous
- Speculation: dspark, mtp off, dflash off
- Thinking: off, thinking-effort none
- Sampling: greedy, max-tokens 128, mode battery
- Prompts file: `benchmarks/prompts/ds4_dspark_genre_code.json`
- Repetition: 3 fresh processes per arm
- Framing: ctx=25 gen=128

| Arm | Gate | Samples (tok/s) | Median | Range spread |
|---|---|---|---|---:|
| A | off | 37.18118152786592 / 37.20601468747717 / 37.21941582544455 | 37.20601 | 0.10% |
| B | `HIPFIRE_DS4_GATHER_TILED=1` | 38.76923560233788 / 38.74626710516697 / 38.78793767502053 | 38.76924 | 0.11% |

Delta: +1.5632 tok/s = +4.20% (B median minus A median).

**Validity signature:** tau = 2.0238095238095237 identical across ALL SIX runs;
decoded answer text identical across all six (md5 of the result line
`53c8ce5ed7b1`). The change is byte-identical by construction: the kernel
reorders only memory access (32x33 LDS tile transpose), performing no arithmetic
reordering.

**Historical golden cross-check:** 37.31652 tok/s median (k6-matched trio),
recorded in `2026-08-06-ds4-dspark-localmaxxing-k4-k6.md`. Arm A reproduced it
to -0.30%.

## 2. Why that kernel was slow

The incumbent `deepseek4_topk_kv_gather_batched_f32` scatters its store: thread
`d` writes `out[(b*head_dim + d)*out_stride + col_offset + k]`, so adjacent
threads are `out_stride` floats apart and every store occupies its own cache
line. The read was already coalesced; only the write was transposed. Flipping
the thread mapping does not help — putting a thread on `k` makes the READ
scatter instead, because each `k` has its own `topk_idx` row.

The fix is an LDS tile transpose, coalesced in both directions. That kernel already existed in-tree as `deepseek4_topk_kv_gather_batched_tiled.gfx1201.hip`, carried no gfx12 ISA dependency (no WMMA, no gfx12 builtins — only a `debug_assert` on arch), and compiles clean for gfx1151 at VGPR 14, occupancy 16, LDS 4352 bytes, zero spills.

The gather is `32x33` LDS-tiled (33 to avoid bank conflicts on the transpose),
so each tile is written coalesced to LDS and read coalesced to global with
neither side scattered.

## 3. Post-fix GPU attribution (diagnostic, NOT an acceptance number)

**Source:** `rocprofv3 --kernel-trace` over `examples/dspark_bench`, k6,
`gfx1151` backend, 64 generated tokens, tau 2.065, tiled gather enabled.

**Caveat — read before quoting any tok/s from this source:**
`dspark_bench` absolute tok/s is NOT a valid `gfx1151` baseline (it frames the
prompt as 24 tokens vs `serve`'s 25). Only the relative attribution (shares of
summed kernel time) is being used here. All acceptance tok/s in this doc come
from §1 (`serve_harness.py`).

Summed kernel time: 2.492 s across 67 distinct kernels.

**Category shares (share of summed kernel time):**

| category | share |
|---|---:|
| E8 dense GEMV | 40.74% |
| MoE GEMV | 36.94% |
| small-kernel tail | 11.96% |
| WMMA GEMM | 7.72% |
| attention | 2.64% |

The tiled gather now measures 0.23% of GPU time (756 calls, 5.85 ms), down from
the 5.0% the untiled kernel occupied before the change — a 21x reduction in
share, independently corroborating the +4.20% end-to-end result in §1.

**Small-kernel tail detail (diagnostic — `rocprofv3` over `dspark_bench`):**

| kernel | calls | ms | share |
|---|---:|---:|---:|
| hc_compute_control_batched | 3282 | 69.26 ms | 2.78% |
| fused_rmsnorm_mq_rotate_plain | 3282 | 29.41 ms | 1.18% |
| rope_tail_yarn_interleaved_batched_f32 | 3552 | 29.19 ms | 1.17% |
| __amd_rocclr_copyBuffer | 15093 | 23.86 ms | 0.96% |
| rmsnorm_f32 | 6329 | 19.99 ms | 0.80% |
| mq_rotate_x | 8488 | 15.12 ms | 0.61% |
| hc_sinkhorn_4x4_batched | 3282 | 11.64 ms | 0.47% |
| sqrt_softplus_f32 | 1641 | 9.21 ms | 0.37% |
| argmax_f32 | 64 | 8.85 ms | 0.36% |
| hc_mix_4stream_batched | 3282 | 8.63 ms | 0.35% |

Unlisted tail kernels make up the remainder of the 11.96% category.

## 4. The roofline conclusion (this is the point of the doc)

77.68% of decode GPU time is weight-bandwidth-bound GEMV (E8 dense + MoE)
running at 83-108% of measured peak bandwidth. Only the 11.96% small-kernel
tail is addressable by kernel optimization.

**Arithmetic — why 45 tok/s is out of reach for kernel work alone at k6 on one
gfx1151:**

Reaching 45 tok/s from 38.77 requires:

```
45/38.77 = 1.1606
```

i.e. removing a fraction `f` of GPU time where `1/(1-f) = 1.1606`, so
`f = 13.8%`.

The entire addressable tail is 11.96%, which is less than 13.8%. Therefore even
perfect elimination of every small kernel yields:

```
38.77/(1-0.1196) = 44.04 tok/s
```

and still falls short of 45 tok/s.

**Conclusion:** kernel-level optimization alone cannot reach 45 tok/s at k6 on
one `gfx1151`.

A realistic tail campaign (halving the top five tail items, 6.89% combined) is
worth roughly +3.4%, landing near 40.1 tok/s. That estimate follows the same
`1/(1-f)` scaling with `f = 0.0689/2` and is stated as an approximation.

## 5. What remains

Two levers, both outside kernel work:

**(a) Fewer weight bytes per token** — i.e. a reduced routed-expert count or
lower weight precision. Both are quality trades and are explicitly NOT taken
here.

**(b) More tokens per weight load** — raise tau. Weights load once per verify
cycle regardless of how many tokens that cycle yields, so throughput scales
nearly linearly with tau. Reaching 45 needs tau 2.024 -> 2.349 (+16%), i.e.
accept rate roughly 67% -> 73%. That is drafter quality, not kernel work.

**Closed finding — the `admit an existing kernel that was gated to another
architecture` seam is now exhausted for this route.**

This was checked two ways. First, the four architecture-specific gates in
`crates/hipfire-arch-deepseek4/src/forward.rs`:

- `e8_wo_grouped` — `gfx1151` has its own grouped O-LoRA path, selected first
  at `forward.rs:8352`.
- `rmsnorm_rotate_nox` — `gfx1151` already admitted via
  `weights.mq2r_backend.is_gfx1151()` at the two call sites, and via
  `norm.rs:4244`.
- `indexer_rope_heads` — candidate-only, default off, previously measured at
  +0.045% and not promoted.
- `indexer_topk_two_stage`.

A gate census is the weaker check, because a kernel can be arch-restricted
without owning a named gate. The stronger check is the kernel inventory: all
21 `kernels/src/*.gfx1201.hip` files, each classified by why it is or is not a
`gfx1151` decode lever.

- `deepseek4_topk_kv_gather_batched_tiled` — **portable; shipped** (§1).
- `hc_compute_control_batched_fused24` — admission requires
  `batch_size == 1024` (`forward.rs:13422`), a prefill-chunk shape. It assigns
  one workgroup per token to share the X load and RMS reduction across all 24
  control rows, which needs a large batch to fill the machine. At decode
  batch (B is approximately 2) it never fires and would not help if it did.
- `hc_compute_control_wmma`, `hc_inv_rms_batched` — use
  `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12`, a gfx12-only intrinsic,
  so this is a port rather than a re-gate. It also lowers each decoded X value
  to F16 at the WMMA boundary, so unlike the gather it is **not** bit-identical.
  That makes it a precision trade, which this route does not take.
- `hc_mix_4stream_peer4`, `tp4_graph_signal` — three/four-rank tensor-parallel
  reductions relying on HIP peer access across ranks. Not applicable to a
  single card.
- `rope_tail_interleaved_h64d128r64` — reached only through
  `gfx1201_indexer_rope_heads_on`, the candidate above that measured +0.045%.
- The eleven `gemv_*` files — `gfx1151` already has its own specialised
  `_gfx1151` GEMV family (visible in the §3 attribution). More importantly the
  GEMV block is bandwidth-bound at 83-108% of measured peak, so a different
  code shape cannot beat the physics; only fewer bytes can.
- `conv1d_silu_split_qknorm` — not on the DS4 decode path.

The tiled gather (`HIPFIRE_DS4_GATHER_TILED`) was the last portable item on
that seam.

## 6. The verify cost model: MoE does not amortise weights across the block

§5 asserted that raising tau is the lever, on the reasoning that "weights load
once per verify cycle regardless of how many tokens that cycle yields, so
throughput scales nearly linearly with tau." **That reasoning is wrong for this
model.** It holds for a dense target. It does not hold for DS4, because each
additional verify position routes to its own `k6` experts, so the union of
experts touched grows with the block.

### Units: tau is accepted drafts, a window emits tau + 1

The harness `tau` counts *accepted drafts*. Each window also emits one bonus
token from the target's own prediction, so a window yields `tau + 1` tokens and

```
t_window = (tau + 1) / decode_tok_per_s
```

The ledger confirms it: the adaptive run accepted 85 drafts over 42 windows
(85/42 = 2.0238095238095237 exactly) and emitted 85 + 42 + 1 seed = 128 tokens.
Dividing `tau` rather than `tau + 1` understates the window by a third; an
earlier revision of this section did exactly that and its cost model was wrong.

### Measured cost curve

Two serve_harness runs on the golden fixture, identical except for
`HIPFIRE_DSPARK_ADAPTIVE_BLOCK`:

| | adaptive (B settles ~2) | `ADAPTIVE_BLOCK=0` (B=5) |
| --- | ---: | ---: |
| tau (accepted drafts) | 2.0238095238095237 | 3.2333333333333334 |
| tokens per window | 3.0238 | 4.2333 |
| decode tok/s | 39.034 | 32.535 |
| windows | 42 | 30 |
| t_window | 77.47 ms | 130.1 ms |

tau rose 59.8% while throughput *fell* 16.6%. Solving the two points:

```
t_window(B) ~= 58.7 + 17.84 * (B - 1)   ms
```

An extra verify position costs 17.84 ms against a 58.7 ms fixed cost. On a
dense target that marginal term would be near zero. Here it is large enough
that the controller settling at B=2 is correct behaviour, not a missed
opportunity, and it explains the previously measured -5.7% for pinned B=5.

### Authoritative phase split

`HIPFIRE_DSPARK_PROFILE=1` over `dspark_bench` (k6, 35 windows). Note this
instrument was unreachable until the `Drop` fix in the same series -- its emit
hung off an explicit model unload that neither a signal-terminated daemon nor a
returning example ever performed.

| phase | mean/window | share |
| --- | ---: | ---: |
| bootstrap (initial-ctx seed capture) | 2.29 ms | 3.0% |
| draft_block | 4.30 ms | 5.6% |
| run_heads | 3.73 ms | 4.9% |
| verify_block | 66.33 ms | 86.5% |
| rest (accept + commit) | 0.01 ms | 0.0% |
| **total** | **76.66 ms** | |

Draft-side cost is `draft_block + run_heads` = 8.03 ms/window, **10.5%**.

An earlier revision of this section put drafting at 1.11%, obtained by grouping
the §3 rocprof capture on the drafter's `q8_0` quantisation. That number is
wrong by roughly 9x and should not be reused. Two reasons: rocprof measures GPU
kernel time while the draft phase is launch-latency-bound, so its wall time far
exceeds its kernel time; and `run_heads` does not run `q8_0` kernels at all, so
format-grouping missed it entirely.

### Per-depth acceptance

From `HIPFIRE_DSPARK_DEBUG=1`, comparing `drafts` against `target_pick`
prefix-wise. Conditioning matters: a window that proposed fewer than k drafts
can never accept at depth k, and under the adaptive controller the decision to
propose deeper is itself made on predicted confidence, which selects easy
windows. Only the forced-depth run gives an unbiased estimate; 28 of its 30
windows proposed the full 5.

| depth k | p(k) = P(hit at k \| proposed k, survived k-1) | R(k) = P(accept >= k \| proposed >= k) |
| ---: | ---: | ---: |
| 1 | 96.7% | 96.7% |
| 2 | 79.3% | 76.7% |
| 3 | 90.9% | 71.4% |
| 4 | 70.0% | 50.0% |
| 5 | 78.6% | 39.3% |

There is no acceptance cliff. Per-depth accuracy stays in the 70-97% band all
the way to depth 5; deep blocks die of verify cost, not drafter collapse.

### The break-even acceptance bar

Adding verify position k costs 17.84 ms and yields R(k) tokens, so it is worth
taking only while it beats the target rate. For 45 tok/s (0.045 tok/ms):

```
R(k) > 0.045 * 17.84 = 0.803
```

**Every depth carried must accept at better than ~80%.** Measured R(1) = 96.7%
clears it; R(2) = 76.7% sits just under; R(3) = 71.4% and below are well under.

This is the sharpest statement of what is missing. Not "tau must reach some
value" -- tau is bounded by B, and at B=2 the window already emits 3.02 of a
possible 3.0 tokens, so B=2 is saturated. Reaching 45 requires carrying *more*
depth, which requires clearing the 80.3% bar at each depth carried. Depth 2 is
roughly three points short, and depth 3 about nine.

With drafting at 10.5% of the window, a stronger drafter is affordable but not
free: doubling draft-side cost adds 8.03 ms (+10.5% window) and must buy back
more than that in acceptance. That trade -- not deeper blocks at current
accuracy, and not further kernel work -- is the remaining path to 45 tok/s.

