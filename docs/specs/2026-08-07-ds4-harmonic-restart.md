<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: 2026 Kaden Schutt <kaden@hipfire.dev> -->

# DS4 harmonic restart — gfx1100 + gfx1151

Date: 2026-08-07
Branch: `ds4-beta-staging` (now fast-forwarded to `ds4-gfx1201-opt` @ `eb55cda9b`)
Scope: the `hipx` pair **gfx1100 (RX 7900 XTX, 24 GiB, ~960 GB/s) + gfx1151
(Strix Halo 8060S, 96 GiB, ~256 GB/s)** only. gfx1010 / gfx1030 on the same
host are explicitly out of scope.

Supersedes the forward-work portions of
[`2026-08-06-deepseek4-harmonic-gfx1100-gfx1151.md`](2026-08-06-deepseek4-harmonic-gfx1100-gfx1151.md).
That document and the H0–H8 investigations remain the historical record.

## 1. Decision

Restart, importing from `ds4-gfx1201-opt` its **method and its decode kernel
levers** — and explicitly **not** its topology.

1. **Adopt the gfx1201 admission discipline verbatim** (§4). This is the
   highest-value import and it costs nothing.
2. **Reallocate the effort budget from transport to the gfx1100 serial tier**
   (§3). This is where the headroom is, and the failed campaign never spent
   there.
3. **Keep the asymmetric role split.** It is correct for this hardware. Do not
   port TP3/TP4, peer-HC, or attention-TP (§5).

The prior campaign is not being restarted. Its accepted mechanisms (ring
dataplane, residency plan, worker supervision) carry forward; its composition
program does not.

## 2. What actually happened

Measured product line, canonical 2,048/512 fixture:

| Route | tok/s | Note |
|---|---:|---|
| Single-gfx1151 retained-PM4 | 28.8678 | waterline; model fits in 96 GiB |
| Hetero + attention overlap | 30.0439 | G5 accepted |
| Hetero + grouped O-LoRA | **32.0029** | G5 accepted; still the high-water mark |
| DS4HARM2 fault-contained | 27.7100 | safety regression *below* single device |
| DS4HARM3 hotset 1400 | 31.5721 | recovers HARM2; does not beat 32.0029 |
| TG128 per-layer checkpointed AQL | 12.5142 | −58.74% vs 30.3318 control |

179 commits and ~27,400 net lines produced **+10.9% over a single gfx1151**,
and the fault-contained path is still 1.3% *below* the unsafe waterline it
replaced. T1 (50 tok/s) never came into range.

### 2.1 The arithmetic that condemned the chosen lever

The branch's own H1 bill (`docs/investigations/2026-08-06-ds4-harmonic-h1-critical-path.md:19-40`)
prices a token as:

```text
gfx1100 useful interval union                 17.014 ms
gfx1151 useful expert interval union           9.846 ms
measured cross-device useful overlap          -1.648 ms
                                              ---------
global useful interval union                  25.212 ms
canonical product wall                        31.247 ms
host/launch/queue/protocol residual            6.036 ms
```

Transport and composition work can only attack the **6.036 ms residual**.
Zeroing it *entirely* yields 25.212 ms → **39.66 tok/s**. The campaign's own
T1 gate is 50 tok/s.

**The lever the campaign spent 179 commits on could not reach its own minimum
target even at perfection, and H1 said so on 2026-08-06 at line 38.** Every
subsequent transport result is consistent with that: the mechanisms got
cheaper (ring 74.593 → 4.626 µs/chain; host-gated AQL 6.181 µs/gate) while
product throughput stayed pinned near 32.

## 3. Where the headroom actually is

At expert-branch balance the **gfx1100 serial tier is 77.7% of the useful
union**. Marginal return, computed from the residency model:

| Remove 1 ms from | Union improves by | Relative return |
|---|---:|---:|
| gfx1100 serial tier | 1.000 ms | **3.18×** |
| routed-expert work | 0.314 ms | 1.00× |

That tier is 15.3657 ms/token and its largest line item is a kernel H1 labels
*"exact-compiled generic fallback; not gfx1100-tuned"*: 511 calls, 7.586 ms,
2.858 GB, **376.7 GB/s on a 960 GB/s card** — 39% of peak. The one
exact-gfx1100 E8 kernel in the tree (grouped O-LoRA) reaches **544.2 GB/s** on
the same card and the same format.

### 3.1 Answering "what changes when you add a 2.2× CU / 3.8× BW tier"

Less than intuition suggests, and this is the load-bearing result. Sweeping
`r`, the unmeasured gfx1100/gfx1151 MQ2-Lloyd expert speed ratio, through the
whole plausible range:

| r | balanced hot fraction | useful union | useful ceiling |
|---:|---:|---:|---:|
| 1.00 | 46.35% | 21.422 ms | 46.68 tok/s |
| 2.184 (Qwen, borrowed) | 63.58% | 19.783 ms | 50.55 tok/s |
| 3.75 (pure BW ratio) | 73.18% | 18.870 ms | 52.99 tok/s |
| 5.00 | 77.25% | 18.483 ms | 54.10 tok/s |

A 5× swing in `r` moves the ceiling by 7.4 tok/s, because `T_serial` dominates
the union. **Consequences:**

- The residency plan is robust to `r`. Measuring it precisely is a tuning
  step, not a gate. Do not build a campaign around it.
- gfx1151's expert kernels are already at 213.7 GB/s ≈ 83% of a ~256 GB/s
  part. Agreed — that tier is well-tuned and is **not** a target.
- Adding gfx1100 as an *expert co-owner* is worth little. Its value is as the
  **dense/serial owner**, and that is where it is currently squandered.

### 3.2 Sizing the gfx1100 kernel campaign

H1's own H3 note proposed 600 GB/s. **That target is too weak** — it is 62.5%
of peak, and this codebase already does better than that on harder memory.

#### The efficiency proof point

| Achieved | Peak | % | Kernel |
|---:|---:|---:|---|
| 213.7 | 256 | **83.5%** | gfx1151 MQ2-Lloyd expert gate/up + down (production) |
| 544.2 | 960 | 56.7% | gfx1100 grouped O-LoRA E8 (accepted G5) |
| 403.1 | 960 | 42.0% | gfx1100 dense E8, weighted (the problem) |
| 376.7 | 960 | 39.2% | gfx1100 generic tier (H1 baseline symbol) |

Our own gfx1151 expert kernel sustains **83.5% of peak on unified LPDDR5X** —
an APU memory system that is *harder* to saturate than discrete GDDR6. The
same efficiency on gfx1100 is **802 GB/s**. That, not 600, is the target.

#### Why the dense kernel sits at 39%

From source. The limiter is memory-level parallelism, not decode arithmetic:

- `__launch_bounds__(32)` with no min-waves hint. The grouped and gfx1151
  twins both set `(32, 7)`; the generic and buffer variants omit it.
- One wave per workgroup, one row per workgroup.
- Codeword loads are 32 × u32 = 128 B per wave — **4 B per lane, a plain
  `dword`**. A `dwordx4` gives 4× the bytes in flight per instruction.
- Zero LDS, register decode. ALU is ~10% utilized at these rates; it is not
  the constraint.
- The grouped variant reached 544.2 GB/s from **CU fill alone** (one
  8,192-wave grid replacing eight 1,024-wave grids), with an identical decode
  body. That is the direct evidence that occupancy and request concurrency are
  what is missing.

Little's Law confirms it: 800 GB/s at ~400 ns VRAM latency needs ~313 KB in
flight, i.e. ~3.3 KB per CU across 96 CUs. At 128 B per wave-load that is 26
concurrent loads per CU; at `dwordx4` it is 7. The second is comfortably
reachable, the first is not.

#### The lift

| Weighted dense-E8 BW | % peak | serial tier | shared branch |
|---:|---:|---:|---:|
| 403.1 (today) | 42.0% | 14.62 ms | 1.65 ms |
| 600 | 62.5% | 12.14 ms | 1.18 ms |
| 700 | 72.9% | 11.41 ms | 1.04 ms |
| **802** | **83.5%** | **10.86 ms** | **0.93 ms** |

Work items, in the order the evidence supports: widen loads to `dwordx4`;
multiple rows per workgroup (256 threads × 8 rows) for memory parallelism;
the `(32, 7)` min-waves hint; deeper unroll for independent loads in flight.

**Blocking caveat:** `ds4_dense_e8` already dispatches
`gemv_mfp4g32_e8_soa_buffer_gfx1100`
(`crates/rdna-compute/src/rdna3/gfx1100.rs:70-74`), while H1's 376.7 GB/s is
the **pre-buffer** profiled symbol. The 7.586 ms line item may already be
stale. **Re-billing is mandatory before this sizing is trusted** — gate R1.

## 4. The imported method (non-negotiable)

Verbatim from the gfx1201 campaign:

- **2% product admission threshold.** A candidate projecting under 2% gets no
  product bench. A candidate measuring under 2% gets one sample, then stop —
  no second or third process.
- **Micro projections are admission filters, never ceilings.** The attention-TP
  micro projected 36.6 tok/s and product delivered 41.2059.
- **Three fresh-process samples** for any decode promotion; report median and
  range spread. Accepted gfx1201 spreads ran 0.073%–0.65%.
- **Mandatory byte-identical golden.** Full decoded output SHA-256 must match.
  A coherent-but-different output is a rejection, not a judgement call — the
  gfx1201 prefill HC WMMA candidate measured 479.3291 tok/s and was rejected
  on SHA alone.
- **Screen, then `Revert`.** A rejected experiment is reverted in the same
  ladder, not left in the tree. This is the single biggest process difference
  between the two branches.
- **Durable evidence tree per checkpoint**, with binary and prompt digests.

## 5. What does not port, and why

| gfx1201 lever | Status here | Reason |
|---|---|---|
| Attention TP over RCCL (+16.50%) | **Unavailable** | Mixed gfx1100/gfx1151 RCCL communicator fails `invalid device function` |
| Peer barrier + `hc_mix_4stream_peer4` (+4.32%) | **Prohibited** | Device-side reciprocal peer wait — the exact pattern quarantined after two incidents that stranded both GPUs |
| Shared-expert dense TP4 (+4.35%) | **Not applicable** | Requires equal shards on identical devices |
| Owned-expert skip (+14.42%) | **Not applicable** | Requires symmetric EP ranks |
| Prefill native DSA WMMA / wide-E8 | **ISA-locked** | `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12`; RDNA4 fragment shapes ≠ RDNA3 |

**Do not attempt symmetric TP/EP on this pair.** Replicated work runs at the
speed of the slowest rank; gfx1151 is 3.75× slower on bandwidth. Symmetric
sharding would be *worse* than the current role split. The asymmetric split is
correct — it was never the problem.

## 6. What does port — all four decode levers are structural

Source audit found **no WMMA intrinsics and no gfx12 guards** in any of the
four accepted gfx1201 *decode* levers. All are launch fusion, LDS pattern, job
packing, or workgroup width. The merge in §7 already brought the kernels into
the tree, so most of this is **re-gating, not porting**.

| Lever | gfx1201 Δ | Mechanism | Port cost | Numerics |
|---|---:|---|---|---|
| `hc-fusions` | +7.3041% | 4 launches → 1 control path | **trivial re-gate** — harmonic gates it gfx1151-only at `forward.rs:888-896` | order-preserving |
| grouped O-LoRA half | — | 8→1 launch | **already banked** on gfx1100 at 544.2 GB/s — do not re-port | raw-bit |
| `nox` low-LDS RMSNorm/FWHT | +2.1826% | LDS (K+256)·4 → 32 B, wave-first reduce | **trivial re-gate** — `norm.rs:4193` excludes gfx1100 | **changes FP32 reduce order — needs golden** |
| `mixed-e8-projections` | +2.7045% | ≤7-job mixed-M packing | **moderate** — `shared_jobs.gfx1100.hip` exists but is same-M, 2–3 jobs, and unwired | low |
| `T1024 HC control` | +2.8615% | workgroup 256 → 1024 | **trivial** after `hc-fusions` | **LDS tree 8→32 — needs golden** |

## 7. Merge completed

`ds4-beta-staging` was a strict ancestor of `ds4-gfx1201-opt`, so the merge was
a pure fast-forward: `b4e944370..eb55cda9b`, 229 commits, pushed to `origin`.

This is behaviour-neutral off gfx1201: every lever is gated on exact gfx1201 +
MQ2R + TP3/TP4, with generic binding defaults false for Qwen, MiniMax,
gfx1100, gfx1151, other formats, and other rank counts.

It also changes the restart's shape — `hc_finalize_control.hip`,
`hc_compute_control.hip` (vec4_finalize + T1024), the `nox` RMSNorm variant,
and `gemv_mfp4g32_e8_soa_shared_jobs.gfx1100.hip` are now all in the base tree.

## 8. Target and feasibility envelope

**AR target: 50 tok/s** on the canonical 2,048/512 fixture, gfx1100 + gfx1151.
Locked 2026-08-07. `20.000 ms/token` total wall.

The single variable that decides the campaign is the **achieved dense-E8
bandwidth on gfx1100** (§3.2). Sweeping it against the residual:

| Dense-E8 BW | % peak | useful union | resid 5.15 | resid 4.0 | resid 3.0 | resid 1.0 |
|---:|---:|---:|---:|---:|---:|---:|
| 403 (today) | 42.0% | 19.04 ms | 41.3 | 43.4 | 45.4 | 49.9 |
| 500 | 52.1% | 17.38 ms | 44.4 | 46.8 | 49.1 | **54.4** |
| 600 | 62.5% | 16.23 ms | 46.8 | 49.4 | **52.0** | **58.0** |
| 700 | 72.9% | 15.41 ms | 48.6 | **51.5** | **54.3** | **60.9** |
| **802** | **83.5%** | **14.79 ms** | **50.2** | **53.2** | **56.2** | **63.3** |
| 850 | 88.5% | 14.54 ms | **50.8** | **53.9** | **57.0** | **64.3** |

`resid 5.15` is Levers A + B with launch fusion but **no Lever C**.
`resid 1.0` is Lever C landing fully.

Two conclusions:

1. **At 802 GB/s the target is met without Lever C** (50.2 tok/s). The kernel
   campaign carries the campaign. This is the primary path.
2. **T2 = 60 tok/s is back on the table** at 802 GB/s with Lever C (63.3). An
   earlier revision of this document called T2 unsupported; that was an
   artefact of the weak 600 GB/s target, and is withdrawn.

### 8.1 Why the residual only falls to ~5.15 ms from kernels alone

The residual is dominated by host launch cost, so it tracks dispatch count.
H1 measured 3,165 dispatches/token against a 6.036 ms residual. Launch fusion
attacks both terms: `hc-fusions` removes 3 × 86 = 258 dispatches, mixed-E8
packing ~205. That is 3,165 → 2,702, a 14.6% cut, and 6.036 → **5.153 ms**.
Note the bandwidth work does *not* reduce launch count — wider loads and
better occupancy leave the dispatch count unchanged — so these two halves of
Lever A are independent and both are needed.

### 8.2 Lever C — coarse whole-token retained owner body

**Status: margin, not requirement.** It is the difference between hitting 50
and hitting 60, and it is the insurance policy if the E8 tier lands at 700
rather than 802.

Already measured on this exact hardware
(`docs/investigations/2026-08-07-ds4-gfx1100-owner-throughput-gate.md:13-20`):

| gfx1100 owner body | ms/token | tok/s |
|---|---:|---:|
| direct HIP | 21.318 | 46.9086 |
| retained as **one** PM4 packet | 16.440 | 60.8265 |

**4.878 ms/token of host/launch cost removed, bit-identical logits, 12/12
samples.** Discounted for the dispatches Lever A already removes, retention is
worth ~4.16 ms — taking the residual to ~1.0 ms.

**This is the lever that killed the last campaign, and the distinction is the
whole plan.** What was rejected on TG128 (−58.74%) was *43 separately prepared,
per-layer checkpointed queues* costing ~1.20 ms/layer of submit/wakeup/wait.
What is proposed is *one persistent owner tape per token* with owner-local
finite gates, measured at **6.181 µs/gate = 0.266 ms/token** across 43 gates.
That is a 194× difference in synchronization tax, and it is the next cut the
TG128 doc itself prescribes at `:118-132`.

**Hard pre-gate, no exceptions:** before any model run, screen the
continuation protocol with a small multi-checkpoint oracle and demonstrate a
projected ≥2% end-to-end win. If the oracle does not clear, Lever C is dead
and we ship on the kernel campaign — not pursued on faith. This is exactly
the gate the previous campaign lacked. Because C is now margin rather than
requirement, killing it is cheap.

## 9. Restart ladder

Measurement on `hipx` is a **serial resource** — fresh-process benchmarking
with ±10–15% DPM/thermal drift means no two candidates may be timed
concurrently. Implementation parallelizes; screening does not.

**Wave 1 — implement concurrently, each behind its own admission flag,
default off.** No benchmarking, no shared-file coordination needed.

| Slice | Work | Owns |
|---|---|---|
| **R4-BW** | **The campaign.** Rewrite the gfx1100 dense E8 GEMV for memory-level parallelism: `dwordx4` loads, multiple rows per workgroup, `(32,7)` min-waves, deeper unroll. Target 802 GB/s. | `gemv_mfp4g32_e8_soa_buffer.gfx1100.hip` |
| R2 | `hc-fusions` re-gate onto exact-gfx1100 MQ2R | `forward.rs:888-896` |
| R3 | `nox` re-gate + T1024, as two independent gates | `norm.rs:4193`, `attention.rs` |
| R4-PACK | `shared_jobs` wiring for `w1`/`w3` + compressor pairs (~1.45 ms) | `gemv.rs`, `forward.rs` call sites |
| C-oracle | Multi-checkpoint continuation oracle for Lever C (§8.2 pre-gate) | new bench |

R4-BW is the critical path and should be staffed accordingly: §8 shows the
whole target turns on it, and every other slice is worth 2–7% against its
~9 tok/s.

**Wave 2 — screen serially on hipx, in this order.** Each under §4: 3 fresh
processes, median + range spread, mandatory byte-identical golden, 2% gate,
revert on reject.

- **R0** Re-establish the waterline. Every figure in §2 predates the merge.
- **R1** Re-bill the dense E8 tier **per projection shape**, not flat, and
  confirm whether `buffer_gfx1100` or the generic symbol is live. This scopes
  R4-BW and may show the 7.586 ms line item is already partly stale.
- **R4-BW** first, because it is the campaign. Screen at the shape level
  before the product run: a per-shape bandwidth micro is cheap and its
  projection gates the product sample.
- **R2** → **R3** (two separate screens, both need fresh goldens) →
  **R4-PACK**.
- **C** only if its oracle cleared ≥2%. Skip without regret if R4-BW landed
  at or above 802 GB/s.
- **R5** Rate-matched residency last. Measure `r` once; §3.1 shows it is worth
  ~4 tok/s of ceiling, not a campaign. Do not max-fill VRAM.

### 9.1 Stop conditions

- R4-BW below **700 GB/s** after the shape work: Lever C moves from margin to
  required, and its oracle becomes a blocking gate rather than optional.
- R4-BW below **600 GB/s**: stop and re-bill. The memory-parallelism diagnosis
  in §3.2 is then wrong and the campaign needs a new root cause before more
  effort is spent.

## 10. Do not retry

- **Per-layer checkpointed / host-gated AQL composition.** −58.74% on TG128.
  The gfx1201 branch independently closed the same family after one screen
  (graph-resident barrier, −17.442%). Two branches, two mechanisms, same
  verdict: fine-grained in-queue synchronization loses. Lever C (§8.2) is the
  *coarse* one-tape-per-token shape, not this; keep the distinction sharp.
- **E8 four-group prefetch.** Micro won 1.034 ms; product lost 0.534% because
  the gfx1151 wait branch lengthened. Conditional revisit *after* R5 balance,
  never as a cold retry.
- **ROCr IPC signal as a GPU dependency.** Cycle-0 KFD page-not-present.
- **Ragged wkv+compressor collapse.** HIP 700, stuck in
  `drm_sched_entity_flush`.
- **Any device-side reciprocal peer wait.** Quarantined; strands both GPUs.

## 11. Follow-up

`docs/investigations/2026-08-07-gfx1201-ds4-dense-tp.md:11-17` states the
gfx1201 work "is isolated on `ds4-gfx1201-opt`" and that the heterogeneous line
"remains on `ds4-beta-staging`". Both clauses are stale after §7. The file is a
dated investigation record, so it is left as-written; this section is the
correction.
