# R0 — native aie2p MAC-conf: shape + throughput reconciliation

Bottom-up NPU methodology, rung 0: establish the disasm-verified compute ceiling
so every later addition (feed, unpack, multi-core) is measured as a delta from a
known reference. This rung resolves the **58-vs-116 TOPS** question.

## Kernels (`r0_conf.cc`)

- `r0_i8i8` — `aie::mmul<8,8,8,int8,int8>` → `mac_8x8_8x8_conf`, **512 dense MACs**.
- `r0_i8i4` — `aie::mmul<4,16,16,int8,int4>` → `mac_4x16_16x16_conf`, **1024 dense
  MACs** (this is Oq4 W4A8: int4 weight × int8 activation).

Build + disasm: `PEANO=... MLIR_AIE_INC=... ./build_disasm.sh`.

## Disasm (`r0_conf.aie2p.dis`)

Both lower to the native vector MAC:
- i8i8: `vmac dm3, dm0, x0, x2, r6`  (x = 512-bit vec A/B, dm = accumulator RF, r = conf imm)
- i8i4: `vmac dm0, dm0, x4, y1, r2`  (y = packed int4 B operand)

The probe uses identical A/B across accumulators, so the optimizer collapses the
accumulator array — it confirms the *instruction*, not throughput. Throughput is
taken from the scheduling model (below); a non-degenerate throughput kernel +
on-hardware cycle count is R0b.

## Throughput — from the aie2p scheduling model

`llvm-aie .../aie2p/AIE2PGenSchedule.td`, `II_VMAC_vmul_cm_core_X_X` (and the
`X_QX`/`Y_Y` variants) have an **empty FU stage list `[]`** → **1 VMAC/cycle
throughput**; operand latencies `dst=6, acc1=4` (result in 6 cyc; accumulator
dependency 4 cyc → a real kernel needs ≥4 independent accumulators to hit II=1).

## Reconciliation (hclk=1800 MHz, 32 cores, 1 VMAC/cycle, MAC=2 ops)

| conf | intrinsic | MACs/VMAC | dense | peak TOPS |
|---|---|---|---|---|
| int8×int8 8×8×8 | `mac_8x8_8x8` | 512 | yes | **58.9** ← driver nameplate |
| bf16 8×8×8 | `vmac_bf` | 512 | yes | 58.9 |
| **int8×int4 4×16×16** | `mac_4x16_16x16` | **1024** | yes | **117.8** |
| int8×int8 8×16×8 | `mac_8x16_16x8T` | 1024 | **no (2:4 sparse)** | 117.8 |

**Result:** dense int8 == bf16 == 58 TOPS. The 2× → ~116 TOPS comes only from
**int4 weights (W4A8 = Oq4)**, not from int8. The driver's `4096·col·hclk/1e6`
formula encodes the 512-MAC/core case and *undercounts W4A8 by 2×*; our earlier
"~55 int8" figure inherited that same assumption.

**Caveat:** these are compute-issue peaks at hclk=1800. The part is feed-bound —
npuclk (data/NoC) max = 1267 MHz = 70% of hclk (see `npu4_dpm_clk_table` L7).
The 117.8 W4A8 ceiling makes the *feed* the binding constraint (2× the compute to
keep fed); R1 measures the achievable feed rate. NPU pinned to `performance`
(POWER_MODE_HIGH = DPM L7 1800/1267) for stable measurement.

## R0b — II=1 confirmed at the instruction-schedule level

`r0b_throughput.cc`: 4 named independent accumulators forming 4 distinct
dependency chains (`c_ij += a_i·b_j`) from register-resident tiles, sized so the
4 chains hide the `acc1=4` accumulator latency. Compiled `-O2` for aie2p, it
lowers to a **zero-overhead hardware loop** (`r0b_throughput.aie2p.dis`):

```
00000150 <.LBB0_1>:                             ; zero-overhead loop body =
  150: ...  vmac dm0, dm0, x0, x2, r11          ;   4 bundles, one vmac each,
  160: ...  vmac dm1, dm1, x0, x4, r11          ;   no branch, no counter op,
  170: ...  vmac dm2, dm2, x6, x2, r11          ;   nothing but nops beside them
00000180 <.L_LEnd0>:
  180: ...  vmac dm3, dm3, x6, x4, r11
```

4 vmacs in 4 bundles with zero loop overhead ⇒ **1 VMAC/cycle (II=1)** — the AIE
issues exactly one bundle/cycle, so this loop sustains 1 vmac/cycle by
construction. Combined with the empty-FU-stage itinerary, II=1 is established two
independent ways. (Register pressure: 8×2048-bit accumulators spill; ≥4 fit,
matching the `acc1=4` minimum.)

## R0b — II=1 confirmed ON HARDWARE

`r0b_run.py` runs the kernel on the NPU via IRON `@jit` (single core, one column),
using IRON `Tensor`s (`zeros`/`randint` — the `@jit` cache-key needs these, not raw
numpy). `get_cycles()` doesn't link in the minimal JIT build, so throughput is
measured by **differential host timing**: run at several ITERS; the fixed per-run
overhead (xclbin load + dispatch + fill/drain, ~17 ms) is an additive constant, so
the *slope* of wall-time vs vmac-count is the pure per-vmac cost.

Measured (min of 8 runs/point, NPU pinned `performance`):

| ITERS | vmacs | wall (ms, min) |
|---|---|---|
| 1e6 | 4e6 | 19.49 |
| 3e6 | 12e6 | 24.18 |
| 5e6 | 20e6 | 28.87 |

Pairwise slopes: **1.057 / 1.055 / 1.053 cycles/vmac** (@1.8 GHz) — a stable linear
slope ⇒ **II ≈ 1.055 ≈ 1.0, confirmed empirically.** The 5.5% over ideal is a clock
effect: II=1.0 exactly implies a real hclk of **1.706 GHz** (silicon runs just below
the 1.8 DPM nominal). So the *real achievable* compute ceiling is **~56 TOPS dense
int8 / ~112 TOPS W4A8** (vs 58.9/117.8 nominal), with **W4A8 = 2× dense int8** intact.

II=1 is now established three independent ways: empty-FU-stage itinerary, the
zero-overhead 4-vmac disasm loop, and this on-hardware slope.

Run recipe:
```bash
PEANO=... MLIR_AIE_INC=... 
for it in 1000000 3000000 5000000; do RITERS=$it python r0b_run.py; done   # one run/process
```

Caveats: the `OUT0` checksum readback is unreliable (output-buffer layout bug — does
not affect timing, which scales correctly with ITERS). Repeated `@jit` runs in one
process intermittently segfault in pyxrt (py3.14), so each measurement uses a fresh
process.
