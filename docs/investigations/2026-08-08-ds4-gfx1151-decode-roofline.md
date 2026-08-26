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

## 0. Current k6 golden (benchmark against this)

This supersedes the 37.3165 tok/s k6 golden in
[`2026-08-06-ds4-dspark-localmaxxing-k4-k6.md`](2026-08-06-ds4-dspark-localmaxxing-k4-k6.md).
That figure remains correct for the commit it was taken at; it is simply no
longer the shipping number.

| | |
| --- | --- |
| **Median** | **38.97192 tok/s** |
| Runs (3 fresh processes) | 38.94236 / 38.97192 / 38.97297 |
| Range spread | 0.079% |
| tau (accepted drafts) | 2.0238095238095237 |
| Prompt / generated | ctx 25 / gen 128 |
| Decoded-answer md5 | `53c8ce5ed7b1` |
| Commit | `b071cff8a` |
| `examples/daemon` sha256 | `ae11516e338558d119d1d7e44403619536359d37a164837306fc3f2936f7b749` |

Fixture — trunk 82,191,359,851 B and sidecar 5,788,397,278 B, sizes verified at
run time; sha256 as recorded with the previous golden
(`cbf2bbcf…` trunk, `bc695a00…` sidecar):

```bash
ROCR_VISIBLE_DEVICES=1 python3 scripts/serve_harness.py \
  --model /home/kaden/ds4-gfx1151-evidence/2026-08-03-ds4-dspark-pm4-canary/model-e8/deepseek-v4-flash-0731.mq2r \
  --kv f32 --kv-backend contiguous \
  --speculation dspark --mtp off --dflash off \
  --thinking off --thinking-effort none \
  --sampling greedy --max-tokens 128 --mode battery \
  --prompts-file benchmarks/prompts/ds4_dspark_genre_code.json
```

**The `--kv` value changed, the configuration did not.** Every measurement in
this document was originally taken with `--kv q8`. DS4 never implemented a q8
compressor cache: it silently ran F32 under a q8 label. The
`ds4-gfx1201-opt` merge makes that fail closed —
`kv_cache=q8 is not implemented; use f32` — so the command above now says
`f32`, which is what was always executing and is also the registry default for
`deepseek-v4-flash:mq2r`. Re-verified at the merge commit: 38.96833 and
38.96865 tok/s, tau and decoded text unchanged, within 0.01% of the recorded
median. Numbers taken before the merge under `--kv q8` remain comparable.

**Validity signature.** A run of this fixture that does not report
`tau = 2.0238095238095237` with `ctx=25 gen=128` is not the golden
configuration, whatever tok/s it prints. Two failure modes hit this exact
fixture during the work recorded here: DSpark silently falling back to AR when
no `-dspark` sibling sits beside the model path (`tau` comes back null), and a
neighbouring evidence trio that used `--max-tokens 256` and therefore reports
a legitimately different tau of 1.8022. Check tau before trusting the number.

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

## 7. Output determinism at k6, and why k4 stays a diagnostic

### k6 is bit-deterministic

Every k6 run recorded here — six fresh processes across three commits
(`8bcba53ea`, `b071cff8a`, and the golden trio) — produced byte-identical
generated text:

```
assistant_content md5 e49b9893a207d8a6, length 534, on all six
tau                   2.0238095238095237, identical to 13 digits
```

Greedy DSpark decode on this fixture is reproducible, not merely
reproducible-in-throughput. That is a stronger property than a tok/s figure and
is worth stating explicitly when the number is quoted.

### Output depends on the block trajectory

k4 (`--deepseek4-experts-per-token 4`) is a diagnostic here; it is not adopted,
because reducing routed experts is a quality trade. Running it did surface a
property of the engine worth recording. Three k4 arms, same fixture, same
prompt, greedy:

| arm | text md5 | len | tau | decode |
| --- | --- | ---: | ---: | ---: |
| gather off, adaptive B | `0f6363c0da139637` | 538 | 1.6458333333333333 | 39.1633 |
| gather on, adaptive B | `beb683948784e749` / `0f6363c0da139637` | 534 / 538 | 1.886 / 1.54 / 1.396 | 41.79 / 40.26 / 39.40 |
| gather on, `ADAPTIVE_BLOCK=0` | `59c689f95b8e383a` | 530 | 2.5277777777777777 | 31.1730 / 31.1728 / 31.1692 |

Each *fixed* block trajectory is perfectly deterministic — the pinned arm
repeats to 0.012% with identical text and identical tau. What varies is the
trajectory itself: at k4 the verify cost curve is flat enough that the block
controller's argmax wanders between runs on ordinary timing noise, and a
different block selects different `b1..b6` GEMV variants, whose differing
reduction orders round differently and flip the argmax at a near-tie.

So output is a deterministic function of the block trajectory, and the
trajectory is timing-sensitive wherever the cost curve is flat. This is not a
race: pinning the block removes the variation completely. It is also not
specific to the tiled gather — the gather only perturbed timing enough to
expose it. k6 never trips it because its cost curve is steep enough (17.84
ms/position, §6) that the argmax lands in the same place every run.

The pinned k4 arm independently reproduces the §6 result: tau rose to 2.5278,
the highest of any k4 arm, while throughput fell to 31.17 tok/s. Deeper blocks
lose on this architecture whatever the expert count.

**Consequence for benchmarking.** A configuration whose block controller
wanders cannot be quoted to four significant figures, and its spread will be
dominated by trajectory selection rather than by measurement noise: the k4
adaptive arm spread 6.1% against k6's 0.079%. Report tau alongside tok/s; a
varying tau is the signal that the trajectory moved.


## 8. Routed-expert duplication across verify positions

§6 established a 17.84 ms marginal cost per verify position and read it as "MoE
does not amortise weights across the block." That is the symptom. The cause is
an implementation choice, not physics.

The routed gate/up kernel actually used in decode is
`gemv_mq2g256_lloyd_moe_gate_up_k8_indexed_batched_k4096_lds` -- rocprof
attributes all 1641 dispatches to it and none to the `_k4` variant. It launches
on grid `(M, K_TOP, N)` and resolves the expert per position:

```
const int expert_id = topk_indices[bid * K_TOP + krank];
```

so an expert's weights stream once per position that routes to it, even when
two verify positions in the same call route to the same expert. Its
`__shared__ float cb_lds[64]` caches the quantisation codebook, not weights.

`HIPFIRE_DS4_EXPERT_OVERLAP=1` measures the duplicate fraction. On the golden
fixture (tau 2.0238095238095237 confirmed, so this is the shipping routing
behaviour):

| B | distinct / total expert refs | ratio |
| ---: | ---: | ---: |
| 6 | 12168 / 21954 | 0.5542 |
| 3 | 15441 / 26544 | 0.5817 |
| 2 | 18884 / 31056 | 0.6081 |
| 3 | 21941 / 35562 | 0.6170 |
| 3 | 25080 / 40062 | 0.6260 |

Ratios are cumulative across calls; the `B` column is the current call's block.
A ratio of 1.0 would mean zero reuse. At 0.626, **37.4% of routed-expert weight
references are duplicates.**

The ratio *falls* as B rises (0.554 at B=6 against ~0.61 at B=2): deeper blocks
carry more redundancy, so grouping is worth most exactly where the §6 cost
model is most punishing. Grouping verify columns by routed expert would
therefore flatten the marginal-position cost that currently pins the controller
at B=2, which is the mechanism capping tau.

Upper bound, assuming `moe_down_residual_scaled_k8all` shares the structure
(measured on gate/up only):

```
(24.75% + 11.40%) * 0.374  ~= 13.5% of GPU
1 / (1 - 0.135)            ~= +15.6%
38.97 tok/s                -> ~45.0 tok/s
```

This is the first identified lever whose ceiling reaches 45, and unlike the
small-kernel tail (capped at 44.04 by §4 even if every small kernel were
deleted) it removes bytes from the 77.68% GEMV block rather than trying to
outrun bandwidth.

Three caveats before anyone treats 45 as forecast rather than ceiling:

1. **L2 already absorbs some duplicates.** §3 measured the GEMV block at
   83-108% of peak bandwidth, and above 100% implies cache reuse is already
   occurring. Realised savings will be below 37.4%, possibly well below.
2. **Grouping changes MoE reduction order**, which is the same near-tie
   territory that produced the k4 output flicker in §7. It requires the
   golden's byte-identity check, not a throughput delta.
3. **The down-projection is assumed, not measured.** Only gate/up was
   instrumented.

### Result: grouping was implemented, measured, and reverted

The upper bound above was tested, not left as a projection. An expert-grouped
variant kept the grid, buffers and launch shape identical and deduplicated
inside each block: only the first routing index referencing an expert did work,
looping over later indices sharing it, with the packed codes staged in LDS so
each `(row, expert)` was read once.

It was **correct and slower**, on both fixtures:

| fixture | incumbent | grouped | delta | decoded text |
| --- | ---: | ---: | ---: | --- |
| golden, ctx 25 / gen 128 | 38.96293 | 38.25604 | **-1.81%** | `e49b9893a207d8a6` both |
| pp526, ctx 505 / gen 256 | 29.07771 | 28.52744 | **-1.87%** | `72516fb2e7172ec4` both |

Byte-identity held on both arms and tau was exact on both, which confirms the
construction argument: grouping changes which block loads a weight, never how
an output is summed. The kernel did what it was designed to do.

The premise was wrong. The 37.4% duplicate fraction counts expert
**references**, not **HBM traffic**. Those loads were already served from
cache — §3 had already measured the GEMV block at 83-108% of peak bandwidth,
and exceeding 100% is only possible with cache reuse. So grouping saved no
bandwidth while adding real load imbalance: a multi-member block serialises its
members while duplicate blocks exit immediately, which lengthens the critical
path.

The long-context arm is what makes this conclusive rather than suggestive. If
cache capture were an artifact of the golden's small working set, ctx=505
should have favoured grouping. It regressed by the same margin, so the
duplicates are absorbed at both working-set sizes.

**Conclusion: routed-expert duplication is not a lever, and the 17.84 ms
marginal verify position is not explained by redundant weight traffic.** It is
genuine additional traffic — distinct experts, distinct rows — which returns
the remaining path to §6's conclusion: per-position acceptance, i.e. drafter
quality. `HIPFIRE_DS4_EXPERT_OVERLAP` is retained; the measurement was sound,
the inference drawn from it was not.

## 9. Open problem: DSpark tau collapses at longer context

Not solved here. Recorded so the next attempt starts from evidence rather than
rediscovering it.

### The observation

tau degrades monotonically as the request grows, on the same engine and model:

| fixture | tau |
| --- | ---: |
| ctx 25, gen 128 (golden) | 2.0238095238095237 |
| ctx 25, gen 256 | 1.8021978021978025 |
| ctx 505, gen 256 (`ds4_dspark_pp526_code.json`) | 0.795774647887324 |

At tau 0.796 speculation is barely paying for itself. Decode at ctx 505 is
29.07 tok/s against 38.97 on the golden, and most of that gap is tau, not
kernel time.

### The drafter does depend on context

`HIPFIRE_DSPARK_ZERO_CTX=1` zeroes the context `main_hidden` handed to the
drafter. On the pp526 fixture:

| | tau | decode |
| --- | ---: | ---: |
| context on | 0.795774647887324 | 29.060 |
| context zeroed | 0.032388663967611336 | 17.598 |

tau falls to 0.032, so the head is critically dependent on what it is fed. The
long-context weakness is not the drafter ignoring the prompt.

### It is NOT attention dilution, and DFlash's fix does not port

The obvious hypothesis — the drafter drowning in 505 rows of context — is
wrong. DSpark's drafter never sees the prompt. Its context is a slot list
bounded by the verify window, not by sequence length:

- `dspark_core.rs`: `max_context_floats = (self.block + 1) * layers.len() * hidden`
- bootstrap sets `ctx_positions = vec![position]` (one slot)
- steady state sets `ctx_positions = (start_slot..start_slot + new_ctx_len)`

so the head consumes roughly `block + 1` hidden-state vectors — about 3 to 6 —
regardless of context length. That is already a tighter window than DFlash's.

This matters because DFlash has a mature answer to drafter-degradation at long
context (`HIPFIRE_DFLASH_WINDOW`: SWA over the last W rows on draft layers
`0..n-2`, full attention on the last layer, W defaulting to the draft
artifact's declared `sliding_window`). **That fix does not transfer.** The two
drafters are architecturally different: DFlash's is a standalone small model
with its own KV cache attending over real context, so a sliding window is
correct for it; DSpark's is an MTP-style head consuming target hidden states,
already windowed by construction, with no long attention to mask.

What degrades is therefore the *statistics of the hidden states* the head
consumes — a hidden state at position 505 inside a dense multi-part request is
a different distribution from one at position 25 after a one-line request — not
the quantity of them. That is drafter quality, consistent with §6.

### The transferable idea from DFlash

Not the SWA mask: the **regime declaration**. DFlash reads the width the draft
was trained at out of the artifact and enforces it, commenting that this is
"the only width correct by construction," and deliberately degrades tau with a
warning when a request exceeds it. DSpark declares nothing and enforces
nothing — it ran at tau 0.796, and at 0.032 under `ZERO_CTX`, while still
reporting a healthy tok/s. Nothing in the system can currently answer "is this
drafter in-regime for this request?"

### Next two steps, both cheap

1. **Disambiguate length from difficulty.** `ds4_dspark_pp526_code.json` is
   both longer *and* a harder multi-part deliverable, so it confounds them. A
   short prompt demanding complex output against a long prompt demanding
   trivial output separates the two in two runs.
2. **Read the sidecar's metadata.** If the DS4 DSpark artifact carries no
   trained-regime declaration at all, that gap is the first thing to close —
   detection before correction.

## 10. F16 compressor cache: gfx1151 ported with generation-correct WMMA

`ds4-gfx1201-opt` was merged (29 commits, principally 2830c5cd8 with
certification 1019a0e56). It adds a selectable F16 compressor cache confining
F16 storage to `main_kv_cache` and `indexer_kv_cache` — the two ctx-scaled
allocations traced in §9 — halving the replicated compressor/indexer VMM
footprint from 14,428,405,760 to 7,214,202,880 bytes per rank and roughly
doubling admissible context (475,136 tokens against 229,376 on the matched F32
bracket, three 34.2 GB ranks).

Before the port, `gfx1151` was refused at load:

```
deepseek4: kv_cache=f16 currently requires gfx1201 MQ2R TP3/TP4.
GPU: gfx1151 (13532 MB free / 98304 MB total)
```

This was the fifth encounter with the "kernel gated to another architecture"
seam, and the second where the answer was *port*, not *re-gate*:

| kernel | gfx12-only intrinsics | verdict |
| --- | --- | --- |
| `compressor_commit_staged_f16.gfx1201.hip` | none | portable, re-gate |
| `deepseek4_compressor_cache_f16.gfx1201.hip` | `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12` (one call site, line 125) | port required |

A macro swapping the builtin spelling is **not** sufficient. The two
generations take different fragment layouts:

```
gfx11  typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;   // attention_dflash_wmma.hip
gfx12  typedef _Float16 __attribute__((ext_vector_type(8)))  half8_t;    // deepseek4_compressor_cache_f16
```

so the staging loop (`q_frag[i] = qh[k + k_half * 8 + i]`) must be rewritten for
16-wide fragments and gfx11 lane duplication. That failure mode is silent —
wrong attention scores, not a compile error — so it needs the coherence gate,
not a throughput delta. The gfx11 spelling is already proven in-tree
(`attention_dflash_wmma.hip`, and `attention_dflash_wmma_m128_n32_f16kv_v7_f32.hip`
is f16-KV WMMA attention), so the target form exists to model against.

Worth being clear about the payoff before anyone spends the day: the
certification calls F16 "a capacity route, not yet a speed promotion." It buys
context length, not tok/s. Its own stated next step — "keep gathered compressed
K/V in native F16 through the WMMA consumers" rather than widening halves back
to F32 — is where the speed would come from, and that is downstream of the port.

### Ported: F16 compressor cache runs on gfx1151, free

The port was done and measured. Four changes were needed, not one:

1. **Kernel.** The indexer score WMMA is now generation-selected in-source. The
   two generations take different fragment layouts, so both the staging and the
   D row mapping differ — a macro swapping the builtin spelling is not enough.

   ```
   gfx12  8-wide fragments, K chunk split across lane halves, D row = 8*k_half + j
   gfx11  16-wide fragments, whole K chunk per lane with 16..31 duplicating,
          D row = 2*j + k_half
   ```

2. **Eleven wrapper guards.** The loader gates admitted gfx1151 but every
   per-kernel wrapper still asserted `is_gfx1201()`, so the first run panicked
   at `attention.rs:8859` with `gen=0`. Widened exactly the eleven wrappers
   compiling from the two verified-portable sources; the sharded/TP wrappers in
   the same files keep their gfx1201 assert.

3. **File names.** Both kernels lost their `.gfx1201` suffix. This is not
   cosmetic: `scripts/compile-kernels.sh` treats `\.gfx[0-9]+\.hip$` as a
   variant tag, so an arch-suffixed file is AOT-compiled only for that arch and
   everything else silently falls back to JIT.

4. **Capability and symbol cleanup.** The admission gate is the union of the
   actual intrinsic capabilities: `has_wmma_w32` for gfx11 and
   `has_wmma_w32_gfx12` for gfx12. It must not require the gfx11 bit while also
   claiming to admit RDNA4. The last dual-arch wrapper also lost its stale
   `_gfx1201` suffix.

Measured on the golden fixture, warm kernels:

| arm | decode tok/s | tau | decoded text |
| --- | ---: | ---: | --- |
| f16 | 38.92349 | 2.0238095238095237 | `e49b9893a207d8a6` |
| f32 | 38.97186 | 2.0238095238095237 | `e49b9893a207d8a6` |
| f16 | 38.90918 | 2.0238095238095237 | `e49b9893a207d8a6` |

**-0.14%, inside run-to-run noise, with byte-identical decoded output.** F16
storage costs nothing here and halves the two ctx-scaled allocations, so it is
purely a capacity gain on this route.

One measurement trap worth recording: the first f16 run after the rename
reported 30.62997, a 21% apparent regression. That was JIT compilation of the
newly-named kernels inside the measured window — the AOT cache had no entry for
the new file names. The second run, with `.hipfire_kernels/gfx1151` warm,
matched F32. Any kernel rename invalidates that cache, so the run immediately
following one is not a valid measurement.

The committed 8K NIAH fixture (`ctx=5427`) also showed no warm throughput
penalty: F16 decoded at 20.6074 tok/s and F32 at 20.6186 tok/s. That fixture is
not a valid exact-retrieval gate for this checkpoint: F16 returned
`mauve-velocirapto-7741`, while the F32 baseline returned
`mauve-velocirapt-7741`; both miss the expected `mauve-velociraptor-7741`.
Consequently the byte-identical claim above is scoped to the short golden
fixture, while the long-context evidence establishes coherent execution and
throughput parity rather than exact output parity.

### Sub-F16 kv selectors now redirect instead of failing

DS4 stores its compressor cache as F32 or F16 only. `q8`, `asym{2,3,4}`,
`fwht{2,3,4}` and `turbo*` now resolve to F16 — the nearest implemented storage
below F32, and a widening in precision terms relative to what was requested —
with a one-line notice at resolve time rather than a hard load failure. Only
unrecognised strings still fail.

This means the historic `--kv q8` invocation runs again, but it now selects F16
storage rather than the F32 golden. Section 0's fixture says `--kv f32`
explicitly for that reason.
