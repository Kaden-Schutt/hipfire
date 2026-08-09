---
title: "G4: batched E8 GEMVs collapse the ds4 spec-verify window; bytes/token corrected to 6.04 GB"
date: 2026-07-26
tags: [deepseek4, gfx1151, spec-decode, wmma, gemv, roofline, mq2r]
---

## The pathology: WMMA token tiles at speculative-verify widths

The ds4 batched forward sends dense projections through WMMA GEMMs that tile the
TOKEN axis at 16. `e8_prefill_batch_tiles` returns 1 for any `batch_size <= 16`,
so the `_b2`/`_b4` tile multipliers NEVER engage at spec-decode widths. Below
B=16 the tile is mostly padding and launches only M/16 waves. Isolated, the
grouped WMMA is FLAT at 345-357 us from B=1 to B=16 — the tile signature.

Fixed in TWO places (commits b8efa47a7, 7f954ad98), both default-off behind
`HIPFIRE_DEEPSEEK4_E8_BATCHED_GEMV=<max B>`:

- `gemv_mfp4g32_e8_soa_batched` — plain dense (`gemv_auto_batched_wmma`)
- `gemv_mfp4g32_e8_soa_grouped_batched` — O-LoRA A (`wo_per_group_batched_e8_fallback`)

Both keep the decode GEMV's M-wave occupancy and single weight-row read.

**Bit-exactness is against whichever decode kernel the path replaces, and the
accumulation orders DIFFER**: the plain decode GEMV strides FOUR accumulators
across four groups (tail into acc0, `((acc0+acc1)+acc2)+acc3`); the grouped one
strides TWO across a pair (`acc0+acc1`). Mirroring the wrong one costs
bit-exactness — first attempt diverged 2037/4096, max_rel 4.6e-5. Both now pass
at 0 ULP. This matters because verify logits decide token acceptance.

Also: the WMMA path calls `ensure_fp16_x`, so before this, verify ran **f16**
activations while AR decode ran **f32** — the two models disagreed by
construction. The batched GEMVs take f32 directly.

## Measured (MQ2R P3, pos 2048, AR reference in the same process)

```
 B     WMMA      plain     +grouped
 1   120.82 ms   57.97 ms   47.61 ms
 2   131.57      69.78      59.47
 4   154.94     101.22      91.65
 6   180.43     139.50     135.47
 8   203.87     171.24     169.12
 16  crossover: WMMA wins (tile finally full) — hence the explicit batch ceiling

window(B): 107.7 + 12.0*B -> 36.0 + 16.9*B -> 22.9 + 18.3*B   (constant -79%)
AR decode reference: 35.60 ms/token (reproduced 35.57 on a prior run, 0.08%)
```

Spec-decode break-even (`window(B)/tau < 35.60`): was IMPOSSIBLE at B<=4 (required
tau exceeded B itself). Now B=2 needs tau 1.671 / p>=0.671 vs DSpark's measured
p=0.660 — **99.4% of break-even on the verify side**. Remaining B=1 overhead vs
AR decode: 12.01 ms (was 22.37); batched attention
(`deepseek4_attn_swa_topk_batched_wmma` + `topk_kv_gather_batched`, +4.82 ms) is
now the largest single term — the third instance of the same tile pathology.

## Method that worked: differenced rocprof arms

Profile arm A (AR decode xN) and arm B (batched B=1 xN) with an **identical**
`--prefix` prefill, so the prefill's kernel time is the same in both and cancels
in the per-kernel diff. Predicted -10.38 ms for the grouped substitution;
measured -10.36 (0.2% error). `deepseek4_prefill_bench` gained `--tokens 0`
(window mode), `--prefix P`, `--ar-ref N`, `--e8-batched`, and now frees
`PrefillBatchScratch` between batch sizes (the leak OOM'd the sweep at B=6).
The flag is cached in an AtomicUsize, not a OnceLock, so one process A/Bs both
arms against one loaded 80 GB trunk.

## Bytes/token CORRECTED: 6.04 GB (was quoted 4.68 — understated 29%)

From the HFQ tensor table (`dump_hfq_dtypes`), accounting closes to 0.09% of the
82.191 GB file:

```
qt=19 MQ2G256Lloyd 33,024 tensors (256 exp x 43 L x 3) 277.0e9 elems @2.25bpw = 77.90 GB
qt=35 MFP4G32E8SOA    554           6.742e9 @4.25bpw                          =  3.58 GB
qt=3  Q8_0              1  129280x4096 — embed and lm_head are TIED           =  0.56 GB
qt=1  F16             641          35.8e6                                     =  0.07 GB

per token = all-but-experts + 6/256 of experts = 0.07+0.56+1.826+3.58 = 6.04 GB
```

**Dense tier is 59% of bytes/token; routed experts only 30%.** Top-k tuning
touches the smaller half. Corrected, top-6 -> top-4 removes 10.1% of bytes ->
predicted +11.2%; measured +12% (see [[ds4-topk4-bandwidth-scaling]] if written).

At 27.70 tok/s that is 167 GB/s vs the 207 GB/s measured on the dense E8 GEMV at
high wave count = **81% of practical roofline**. BUT during GPU-busy time
(~30 ms of the 35.57 ms token) we move 6.04 GB = **201 GB/s — at the ceiling**.
Per family: dense 192, experts 214, rest 227 GB/s. The ~19% headroom is the
**~5.5 ms/token that is not running kernels at all**, not kernel inefficiency —
and that gap is exactly what nine lever classes failed to convert earlier in this
campaign. Real, measurable, has resisted everything tried.

## Drafter cost is now the dominant term (verify is nearly solved)

The break-even figures above are VERIFY-ONLY. Draft cost:

```
DSpark sidecar  5,996,338,910 B = 6.0 GB  -> ~31.6 ms/pass  (89% of an AR token!)
MTP head        1,998,047,355 B = 2.0 GB  -> ~10.5 ms/pass
```

That is almost certainly the real content of the "-96% perf" DSpark observation —
never a tuning problem, the drafter is nearly as expensive as the token it skips.
Open question worth checking: the trunk's `qt=3` is ONE tensor (embed/lm_head
tied). If the MTP sidecar carries its own copy, ~1.06 GB of its 2.0 GB is
redundant with resident weights; sharing them drops the drafter to ~0.94 GB /
~4.9 ms, at which point B=2 needs p~0.651 and DSpark's 0.660 clears outright.

**NPU (aie2p) does NOT help this.** Strix Halo's NPU shares the same unified
LPDDR5X and the same ~256 GB/s. A drafter costs what it costs because it streams
GB of weights; the NPU adds compute, not bandwidth. Note the objection CHANGED
from the earlier G8 finding — there the blocker was >300 us round-trip against a
1.428 ms kernel; a 10-30 ms drafter amortizes that fine. Bandwidth is the
blocker now, not latency.

## Traps

- **`k8` in MoE kernel symbol names is LEGACY, not a bound.** Source file is
  `gemv_mq2g256_lloyd_moe_gate_up_indexed.hip`; exported symbol is
  `..._gate_up_k8_indexed`. `k_top` is `blockIdx.y`, a runtime grid dimension,
  passed `cfg.num_experts_per_tok` (=6). Nothing is pinned at 8. Confirmed
  empirically: 6->4 moved throughput by the byte-predicted amount, which a
  hardcoded 8 could not do. Rename these symbols.
- rocBLAS/hipBLAS are **structurally unusable** here: they consume dense
  fp16/int8, our weights are MQ2G256Lloyd (2.25 bpw codebook) and MFP4G32E8SOA
  (E8 lattice). Using them requires dequant to fp16 = ~1.8x MORE bytes on the
  biggest tier, on a bandwidth-bound path. Fusing dequant into the GEMV is
  incompatible with the BLAS interface, not an optimization on top of it. See
  also the gfx12 rocBLAS 5.6x-slower finding. rocWMMA = header wrapper over
  intrinsics already called directly (maintenance only). rocPRIM/hipCUB is a
  real but small lever for MoE routing (~2 ms addressable, <1 ms recoverable).
- `/sys/firmware/acpi/platform_profile` does NOT exist on hipx; the amdgpu
  equivalent is `power_dpm_force_performance_level` (currently `auto`), which is
  what `rocm-smi --setperflevel` writes. Untested on gfx1151; note the measured
  gfx1201 finding that `high` UNDER-clocks. Aimed at sclk, but the binding
  constraint here is memory/fabric clock.
- rsync + `setsid nohup script.sh` silently no-ops if the script is not
  `chmod +x` — the log lands 0 bytes and nothing runs. Cost two launches.
- `pkill -f <name>` over ssh can self-kill the ssh command; `pkill -x` truncates
  at 15 chars. Check binary mtime vs source mtime instead of trusting a fast
  cargo "Finished" — mold relinks a big example in <1 s, which looks like a
  no-op build but is not.
