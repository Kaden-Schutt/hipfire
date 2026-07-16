# gfx1201 RMSNorm/FWHT split: exact, but 80 more dispatches lose

## Verdict

Reverted. Splitting `fused_rmsnorm_mq_rotate` into an exact RMS reduction plus
eight independently scheduled 256-element FWHT workgroups is bit-exact, but it
expands the retained tape from 833 to 913 dispatches. Matched tg128 falls 1.43%
and the sampled eight-turn session falls 1.36%. The existing single-block
kernel remains the product path.

## Candidate

The decode-only, gfx1201-gated experiment used:

```text
rmsnorm_reduce_gfx1201 (1 WG x 256 threads)
    -> same-agent acquire
rotate_with_rms_gfx1201 (K/256 WGs x 32 threads)
```

For A3B's `K=2048`, phase B exposes eight independent workgroups instead of
eight wave32s inside one workgroup. Phase A preserves the current kernel's
exact arithmetic order: 256 strided accumulators, wave shuffles at
16/8/4/2/1, then the eight warp sums at 4/2/1. Phase B preserves every
multiply and FWHT butterfly order.

The scalar RMS buffer was allocated once so its address remained stable across
capture and replay. The first PM4 shadow failed while HIP and blob execution
matched each other. That isolated a missing visibility operation between the
new producer and consumer. Adding one gfx12 same-agent acquire after the
dependency wait made logits, KV, and recurrent state exact for all 15 shadow
positions.

Capture after the fix:

- 913 launches (833 control + 80, one extra per 40 layers per token)
- 27 unique kernels
- sequence hash `8de666d12bd0b3d2`
- zero changes to any gfx1100 kernel source or selection path

## 30-row tg128 result

Automatic clocks, ten warmups, 30 rows, 100 measured tokens:

| Arm | Ordinary HIP | Retained PM4 |
|---|---:|---:|
| fused control | **163.137 tok/s** | **193.683 tok/s** |
| gfx1201 split | 153.282 tok/s | 190.908 tok/s |

The split is 1.43% slower in retained PM4 and 6.04% slower through ordinary
HIP. It also misses the 195 tok/s product bar.

## Eight-turn serve

Both arms used the same sampled battery, seed, prompt-cache shape, Q8 KV, and
medium thinking budget. Both produced the same token counts and coherent
outputs with zero runaways, empties, or attractors.

| Arm | Average decode | Final decode |
|---|---:|---:|
| fused control | **166.013 tok/s** | **148.0 @ 20,069** |
| gfx1201 split | 163.750 tok/s | 146.3 @ 20,069 |

The split loses 1.36% average decode and 1.15% at the final long-context turn.

Do not interpret this run's displayed `prefill_ms` as a full-prompt prefill
benchmark. Session mode reuses the prompt cache, so the daemon timed only the
25--54 uncached suffix tokens at each turn's existing context. Averaging those
heterogeneous suffix latencies produced the previously reported `100 ms`
number, which was invalid and has been removed. This experiment changed only
the unbatched decode method; the batched prefill path was source-identical.
Future fusion gates need a separate cold, fixed-token prefill arm.

## Mechanism

The control is already one 256-thread block containing eight wave32 FWHT
groups. Moving those groups into eight blocks improves spatial schedulability,
but requires a second launch, a global scalar RMS round-trip, a true dependency
wait, and a cache acquire on every layer. Retained PM4 amortizes much of the HIP
launch cost, but cannot erase those 80 serialized tape boundaries. The
structural occupancy prize is smaller than the added synchronization and
dispatch cost.

Artifacts on `hiptrx`:

```text
/home/kaden/.redline-work/hipfire-pm4-lean/.redline-work/fusion/fwht-split/
```
