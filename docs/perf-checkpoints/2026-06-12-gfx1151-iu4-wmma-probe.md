# gfx1151 IU4 WMMA probe checkpoint

Date: 2026-06-12
Arch: `gfx1151` / Radeon 8060S
Goal: Qwen3.5/Qwen3.6 uniform MQ/HFQ batched-kernel upgrade path

## Result

The isolated gfx1151 probe validates the first step of
`docs/plans/gfx1151-iu4-mq-packed-ops.md`: native IU4 WMMA is available,
the ROCm builtin accepts two packed 32-bit A/B registers, and the
accumulator layout matches the RDNA3 calculator model.

Command:

```bash
cargo run --release -p rdna-compute --example probe_gfx1151_iu4_wmma
```

Output:

```text
blocks=4096 iters=512 trials=7
expected accumulator value per cell=8192
IU4 median: 0.1728 ms  99397.5 GOPS
IU8 median: 0.3254 ms  52795.7 GOPS
IU4/IU8 throughput ratio: 1.883x
```

The probe writes all eight accumulator registers for every lane and checks
that each cell equals `16 * iters`, so the benchmark is not timing a dead
instruction chain.

## ISA Evidence

Calculator facts:

```bash
python3 third_party/amd_matrix_instruction_calculator/matrix_calculator.py \
  --architecture gfx1151 --instruction v_wmma_i32_16x16x16_iu4 \
  --detail-instruction
```

Key details:

- `v_wmma_i32_16x16x16_iu4`: 8192 ops, 16 modeled cycles, 2 A GPRs,
  2 B GPRs, 8 C/D GPRs on wave32.
- `v_wmma_i32_16x16x16_iu8`: 8192 ops, 32 modeled cycles, 4 A GPRs,
  4 B GPRs, 8 C/D GPRs on wave32.

One-off assembly check:

```bash
hipcc --offload-arch=gfx1151 -O3 -S \
  kernels/src/gfx1151/bench_iu4_wmma.gfx1151.hip \
  -o /tmp/bench_iu4_wmma_gfx1151.s
rg -n "v_wmma_i32_16x16x16_(iu4|iu8)" /tmp/bench_iu4_wmma_gfx1151.s
```

Observed:

```text
v_wmma_i32_16x16x16_iu4 ... neg_lo:[1,1,0] clamp
v_wmma_i32_16x16x16_iu8 ... neg_lo:[1,1,0] clamp
```

## Implication

IU4 is a real gfx1151 lever for uniform MQ4/HFQ4 batched paths, but it is
still not a drop-in replacement for the current Q8_1 MMQ path. The next
engineering step remains a Q4 activation scratch format plus an MQ4/HFQ4
uniform residual-MMQ prototype that measures quality drift against the
current Q8_1 + IU8 control path.
