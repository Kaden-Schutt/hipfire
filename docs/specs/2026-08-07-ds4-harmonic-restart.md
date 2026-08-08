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

### 3.2 Honest sizing of the gfx1100 kernel campaign

Two corrections to naive sizing, both from source inspection:

1. **The grouped O-LoRA 544.2 GB/s win came from CU fill** — one 8,192-wave
   grid replacing eight 1,024-wave grids — *not* from a better decode, LDS
   codebook, or coalescing change. Both kernels use zero LDS and identical
   register-decode. That win does **not** transfer to already-fat single-row
   GEMVs.
2. **64.5% of the generic tier's bytes are `wq_b` (27.6%), `wo_b` (27.0%) and
   `lm_head` (9.9%)** — single-row serial GEMVs that grouping cannot touch.
   Only ~1.45 ms of the 7.586 ms is addressable by the existing `shared_jobs`
   packing (shared `w1`/`w3`, compressor pairs, indexer compressor).

So the lift is a real per-shape kernel job, not one trick:

| Weighted generic-tier BW | ms/token | Saving |
|---:|---:|---:|
| 376.7 (H1 baseline) | 7.587 | — |
| 450 | 6.351 | 1.236 ms |
| 500 | 5.716 | 1.871 ms |
| 600 (H1's own H3 target) | 4.763 | 2.824 ms |

**Blocking caveat:** `ds4_dense_e8` already dispatches
`gemv_mfp4g32_e8_soa_buffer_gfx1100`
(`crates/rdna-compute/src/rdna3/gfx1100.rs:70-74`), while H1's 376.7 GB/s is
the **pre-buffer** profiled symbol. The 7.586 ms line item may already be
stale. **Re-billing is mandatory before any of this sizing is trusted** — that
is gate R1.

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

## 8. Restart ladder

Gates run in order. Each is a single lever under §4 rules.

- **R0 — Re-establish the waterline.** Rebuild HEAD, re-measure the canonical
  2,048/512 direct-HIP hetero route, 3 fresh processes. Record binary and
  prompt digests and the golden output SHA. Nothing proceeds without a current
  number; every figure in §2 predates the merge.
- **R1 — Re-bill the gfx1100 serial tier, per shape.** rocprof the decode tail
  and attribute the dense E8 tier **by projection shape**, not flat. Confirm
  whether `buffer_gfx1100` or the generic symbol is live. Deliverable: the
  §3.2 table with measured per-shape bandwidth. This gate decides whether
  Lever A is worth 1.2 ms or 2.8 ms.
- **R2 — `hc-fusions` re-gate.** Admit `hc_finalize_control` +
  `hc_compute_control_vec4_finalize` on exact-gfx1100 MQ2R. Largest gfx1201
  delta, trivial cost, order-preserving. Golden required.
- **R3 — `nox` re-gate + T1024.** Both change FP32 reduction order; each needs
  its own gfx1100 golden re-certification. Run as two separate screens, not a
  bundle.
- **R4 — E8 shape work.** Driven by R1. Cheap trial first: clone
  `__launch_bounds__(32,7)` onto the dense buffer kernel (grouped and gfx1151
  set the min-waves hint; generic and buffer omit it). Then wire `shared_jobs`
  for the `w1`/`w3` and compressor pairs (~1.45 ms addressable). Then fat-M
  single-row bandwidth for `wq_b`/`wo_b`/`lm_head` (~4.90 ms, 64.5% of bytes,
  not reachable by packing).
- **R5 — Rate-matched hot-expert residency.** Only after R2–R4. Measure `r`
  once, pick the balanced hot fraction, do not max-fill VRAM. §3.1 shows this
  is worth ~4 tok/s of ceiling, not a campaign.

### Projected outcome

With Lever A at the mid case and residency at `r = 2.184`:

| Residual | Product |
|---:|---:|
| 6.0 ms (today's unsafe path) | 44.0 tok/s |
| 3.0 ms | **49.6 tok/s** |
| 1.5 ms | 54.9 tok/s |

T1 = 50 tok/s is reachable, and only reachable, with **all three** of the
kernel campaign, the residency revision, and the residual at ~3 ms. T2 = 60 is
not supported by this bill. Revise the targets accordingly rather than
carrying an unreachable number.

## 9. Do not retry

- **Per-layer checkpointed / host-gated AQL composition.** −58.74% on TG128.
  The gfx1201 branch independently closed the same family after one screen
  (graph-resident barrier, −17.442%). Two branches, two mechanisms, same
  verdict: fine-grained in-queue synchronization loses.
- **E8 four-group prefetch.** Micro won 1.034 ms; product lost 0.534% because
  the gfx1151 wait branch lengthened. Conditional revisit *after* R5 balance,
  never as a cold retry.
- **ROCr IPC signal as a GPU dependency.** Cycle-0 KFD page-not-present.
- **Ragged wkv+compressor collapse.** HIP 700, stuck in
  `drm_sched_entity_flush`.
- **Any device-side reciprocal peer wait.** Quarantined; strands both GPUs.

## 10. Follow-up

`docs/investigations/2026-08-07-gfx1201-ds4-dense-tp.md:11-17` states the
gfx1201 work "is isolated on `ds4-gfx1201-opt`" and that the heterogeneous line
"remains on `ds4-beta-staging`". Both clauses are stale after §7. The file is a
dated investigation record, so it is left as-written; this section is the
correction.
