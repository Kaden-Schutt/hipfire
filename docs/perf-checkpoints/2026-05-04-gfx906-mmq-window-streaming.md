# gfx906 MMQ — window streaming + LDS bank-conflict diagnostic

Date: 2026-05-04
Hardware: gfx906 (MI50). HIP 6.4.3.
Workload: Qwen 3.5 9B mq4, `bench_qwen35_mq4`.

## TL;DR

Refactored the gfx906 MMQ kernel body from **8 syncs/HFQ4-group**
(Option B) to **4 syncs/HFQ4-group** (Option C window streaming),
then fixed a +175% LDS bank-conflict rate via a single int of row
padding. Net: **pp128 462 → 512 tok/s (+10.8%)**, pp512 554 → 584
(+5.4%), 78% of stock llama.cpp pp512.

The diagnostic itself is the more valuable artifact: PMC counters
(`LDSBankConflict`, `ALUStalledByLDS`) directly identified the bank-
conflict pattern caused by `X_STRIDE % 32 == 0`. Adding 1 pad int
made `X_STRIDE % 32 == 1`, rotating the bank index per row and
collapsing the conflict rate from 47% → 0%.

## Context

After the default-on commit (`52eb6bb`) the gfx906 MMQ kernel
delivered 461 tok/s pp128. rocprof attribution showed the two MMQ
kernels (`_full_set_x64` 1.94 ms/call, `_full_add_x64` 2.36 ms/call)
were running ~2× slower than stock llama.cpp's `mul_mat_q<Q4_K, 64>`
extrapolated for the same shape. Plan §P2 hypothesized that
sync-frequency reduction (8/group → 2/group) would close the gap.

## Probe (Option C 4-syncs/group, no padding)

Initial implementation: load 128-K of x_qs per window (X_STRIDE=32
ints/row), run 4 sub-blocks back-to-back without intermediate
syncs, 4 syncs/HFQ4-group total.

Probe ELF (kernels/src/gfx906_mmq_probe_option_c.hip):
- vgpr_count: 68 (lower than Option B's 112 — more headroom)
- vgpr_spill_count: 0
- group_segment_fixed_size: 25,600 B (fits 2 WGs/CU at 64 KiB cap)

All-clear in ELF. Built and benched: **regression**.

| pp | Option B | Option C unpadded | Δ |
|---|---|---|---|
| 32  | 277 | 160 | -42% |
| 64  | 365 | 236 | -35% |
| 128 | 462 | 293 | -37% |
| 256 | 561 | 343 | -39% |
| 512 | 554 | 341 | -39% |

Both MMQ kernels showed **per-call regressions**: `_full_set_x64`
1.23 → 1.86 ms (+51%), `_full_add_x16` 0.41 → 0.77 ms (+87%).

## Diagnostic: PMC counter sweep

Ran `rocprof -i pmc.txt` for VALUBusy, MemUnitStalled, FetchSize,
WriteSize, LDSBankConflict, ALUStalledByLDS on both Option B and
Option C unpadded.

| Counter | Kernel | Opt B (stride 8) | Opt C (stride 32) |
|---|---|---|---|
| **LDSBankConflict** | `_full_set_x64` | 13.5% | **37.2%** |
| | `_full_add_x16` | 20.6% | **47.0%** |
| ALUStalledByLDS | `_full_set_x64` | 6.0% | 4.0% |
| | `_full_add_x16` | 15.7% | 21.7% |
| VALUBusy | `_full_set_x64` | 41.1% | **19.8%** ⛔ |
| | `_full_add_x16` | 18.8% | **7.9%** ⛔ |
| MemUnitStalled | `_full_set_x64` | 0.249 | 0.088 |
| | `_full_add_x16` | 0.079 | 0.036 |
| FetchSize KB/call | `_full_set_x64` | 51.9 | 30.0 |
| | `_full_add_x16` | 24.3 | 20.4 |

**Key insight**: Option C *reads less HBM, fewer memory stalls* — but
**triple the LDS bank-conflict rate**, with VALUBusy halved.

The kernel is "stuck waiting on LDS but not memory" — a telltale
sign of bank conflicts.

## Root cause: stride-32 LDS layout is bank-pathological

AMD GCN/Vega has **32 LDS banks at 4 bytes each** (128 bytes total
per cycle). Bank index for an int access: `(byte_addr / 4) % 32`.

For `x_qs[i * X_STRIDE + v]` where `i` = lane:

| X_STRIDE | Bank index for lane i | Conflict pattern |
|---|---|---|
| 8 (Option B) | `(i*8 + v) % 32` | banks {0,8,16,24}, 4-way conflict per warp half |
| **32 (Option C unpadded)** | `(i*32 + v) % 32 = v` | **all 64 lanes hit the same bank — 64-way conflict** |
| **33 (Option C+pad)** | `(i*33 + v) % 32 = (i + v) % 32` | **32 distinct banks per warp — 0 conflicts** |

A power-of-2 stride that's a multiple of 32 ints aligns *every lane*
to the *same bank column*. The LDS arbiter has to serialize 32-way
or 64-way reads, costing tens of cycles per access.

## Fix: +1 padding (X_STRIDE 32 → 33)

```c
#define X_STRIDE 33  // 32 data ints + 1 pad
```

LDS budget per WG: `128 rows × 33 ints × 4 B = 16,896 B` (was
16,384 unpadded). Adds 512 B/WG, 1 KiB across 2 WGs/CU — well
within the 64 KiB cap.

Re-ran PMC after the change:

| Counter | Kernel | Opt B | Opt C unpadded | **Opt C+1 padding** |
|---|---|---|---|---|
| LDSBankConflict | `_full_set_x64` | 13.5% | 37.2% | **0.0%** ✅ |
| | `_full_add_x16` | 20.6% | 47.0% | **0.0%** ✅ |
| ALUStalledByLDS | `_full_set_x64` | 6.0% | 4.0% | **0.2%** ✅ |
| | `_full_add_x16` | 15.7% | 21.7% | **1.1%** ✅ |
| VALUBusy | `_full_set_x64` | 41.1% | 19.8% | **37.2%** |
| | `_full_add_x16` | 18.8% | 7.9% | **26.6%** ✅ |

Bank-conflict rate **collapsed to 0**. ALU-stalled-by-LDS dropped
to 0.2-1.1%. VALUBusy on the small kernel actually *improved* over
Option B (18.8% → 26.6%).

## End-to-end perf

Full prefill sweep, no env vars (default-on), Qwen 9B mq4 on MI50,
5 runs/config, last-run measurement:

| Prefill | Pre-redesign baseline | Option B (committed) | **Option C+pad (this)** | Speedup vs B | Speedup vs baseline |
|---|---|---|---|---|---|
| pp32  | 136 | 277 | **312** | +12.6% | 2.29× |
| pp64  | 139 | 365 | **383** | +5.0% | 2.75× |
| **pp128** | **141** | **462** | **512** | **+10.8%** | **3.66×** |
| pp256 | 143 | 561 | **592** | +5.6% | 4.21× |
| **pp512** | **142** | **554** | **584** | **+5.4%** | **4.16×** |

vs stock llama.cpp pp512 (750 tok/s): 74% → **78%**.

## Correctness

| Gate | Result |
|---|---|
| Synthetic NRMSE (6 shapes from full sweep) | 0.04–0.18%, all PASS — bit-identical to Option B |
| Real-data NRMSE (`/tmp/mmq_dump_0`, M=4096 K=4096 N=128) | **0.2881%** identical to Option B |
| Coherence gate (6 rows) | all PASS |

Math is preserved. The change is purely a sync-frequency reduction
+ LDS layout adjustment.

## Code change

`kernels/src/gemm_hfq4g256_residual_mmq_gfx906_body.cuh`:
- `X_STRIDE` 8 → 33 (was 32 data, now 32 data + 1 pad).
- `load_hfq4_tile_streaming` parameter renamed `sub_iter` → `window`,
  loop count 2 → 8 (loads 64 bytes/row instead of 16).
- `vec_dot_dp4a_streaming` reads `x_qs[i * X_STRIDE + sub_block * 8 + v]`
  (was `x_qs[i * X_STRIDE + v]`).
- Outer loop in `mmq_body_templated` restructured: 8 sub_iters/group
  → 2 windows × 4 sub-blocks (back-to-back).

`crates/rdna-compute/src/dispatch.rs`: `X_STRIDE` constant 8 → 33
in both residual and gate_up dispatchers, paired with updated LDS
budget comment.

## Lesson learned

The probe ELF (vgpr=68, lds=25 KiB) said "this should work" — but
ELF doesn't measure dynamic execution patterns. **PMC counters are
the right tool for diagnosing performance regressions on the
post-build kernel**, not just the LDS budget gates that Phase 2a
defined.

The next time we make a kernel change that affects LDS layout,
check `LDSBankConflict` *before* benching wallclock. A 47% bank-
conflict rate would have flagged the issue immediately, before
the perf regression.

## Reproducing this report

```sh
# PMC counter sweep (one counter per pass on gfx906)
for ctr in VALUBusy MemUnitStalled FetchSize LDSBankConflict ALUStalledByLDS; do
  printf 'pmc: %s\ngpu: 0\n' "$ctr" > pmc.txt
  rocprof -i pmc.txt -o "run_${ctr}.csv" \
    $HIPFIRE/target/release/examples/bench_qwen35_mq4 \
    $HIPFIRE_MODELS/qwen3.5-9b.mq4 \
    --prefill 128 --prefill-runs 1 --gen 0 --warmup 0
done

# Aggregate per-kernel averages
awk -F, 'NR>1 { gsub(/"/,"",$2); gsub(/\.kd$/,"",$2);
  if ($2 ~ /full_(set_x64|add_x16)/) { n[$2]++; sum[$2]+=$NF }
} END { for (k in n) printf "  %-60s %4d × %12.3f\n", k, n[k], sum[k]/n[k] }' run_*.csv
```
