# kernels — per-question, per-arch kernel microbenches

## Why this exists

`tools/quant-design/` answers *format*-design questions with sweep programs.
This tree answers *kernel*-design questions — one candidate mechanism at a
time, against the verbatim production construct, on the exact architecture
the question belongs to — **before** anything touches `kernels/src/` or the
engine.

The failure mode this spec exists to prevent: packing a whole policy study
(6 arms × 3 bit-widths × 5 batch sizes × 200 samples) into one binary. That
is a macrobench; you wait ten minutes for a number you needed in thirty
seconds, and a mid-run GPU wedge costs you everything.

## Layout

```
tools/kernels/
  gfx1151/   # Strix Halo (LPDDR5X, 32 MiB MALL, latency-sensitive)
  gfx1100/   # Navi 31 dGPU (GDDR6, high BW, deep CU count)
  gfx1201/   # Navi 48 (RDNA4, gfx12 WMMA fragment map)
```

gfx1100 ≠ gfx1151 geometrically and computationally even though they share
the gfx11 ISA — a question answered on one is **not** answered on the other.
Each directory owns its arch's files, defaults, and verdicts. Sharing kernel
*text* across sibling files is fine; sharing a verdict is not.

## The spec

- **One file = one question about one production kernel.** The header
  comment names the production kernel(s) interrogated (authority lineage),
  states the question, and gives exact build + run lines.
- **Two or three arms, max.** Arm 0 is the verbatim production construct.
- **Bit-equality is mandatory.** Every candidate arm's full output buffer
  must be `f32 to_bits`-equal to arm 0 (design arms so accumulation order is
  preserved; then equality is exact, not approximate). Arm 0 additionally
  checks against a sampled host reference to catch fixture bugs.
- **Seconds, not minutes.** One bit-width, one shape, ~50 timed samples by
  default; target < 30 s wall per run including fixture generation. Sweeps
  happen by invoking the binary several times, not inside it.
- **Zero-arg runnable except `--arch`.** Device is selected by
  `gcnArchName` prefix match, never by `HIP_VISIBLE_DEVICES` ordering.
- **Timing conventions** follow `tools/quant-design/bench_mqv2_affine_dot.hip`:
  warmup, HIP events, 8 independent weight slabs rotated per sample so the
  working set stays DRAM-resident, per-launch µs + weight GiB/s.
- Sources are committed here; binaries and logs go to
  `.codeinsight+research/kernels/<arch>/`.

## Build / run

```bash
# gfx1151 (fat binary required: ROCm 7.15 rejects single-arch gfx1151 images)
/opt/rocm/bin/hipcc --offload-arch=gfx1100 --offload-arch=gfx1151 -O3 \
    tools/kernels/gfx1151/mb_qkv_mw_vs_bt.hip -o mb_qkv_mw_vs_bt
./mb_qkv_mw_vs_bt --arch gfx1151

# gfx1201 (local)
/opt/rocm/core/bin/hipcc --offload-arch=gfx1201 -O3 \
    tools/kernels/gfx1201/<file>.hip -o <bin>
```

Hardware-health discipline on shared hosts: check
`sudo dmesg | grep -cE "TransferTableSmu2Dram|MES ring|GPU reset|page fault"`
before and after each binary; any increase is a stop condition (see the
2026-08-29 gfx1100 SMU wedge).

## Current questions

|File|Question|Production kernel(s)|
|---|---|---|
|`gfx1151/mb_qkv_mw_vs_bt.hip`|Does MW-LDS same-row staging beat BT4 for QKV on the latency-bound LPDDR part?|`gemm_qkv_mq{3,5,6}g256v2_wmma_gfx11_bt4` vs prototype MW|
|`gfx1151/mb_lds_pad18.hip`|Does `[16][16][18]` LDS padding fix the 4-way bank conflict on MW compute reads — and does a mis-aligned b128 cast even function?|MW-LDS family LDS layout|
|`gfx1100/…`, `gfx1201/…`|Per-arch twins of the still-open decode/NT/MW questions|see file headers|

Superseded: the sweep monoliths `tools/quant-design/bench_mqv2_decode_screen.hip`
and `bench_mqv2_mw_qkv_screen.hip`. Their banked results
(`.codeinsight+research/quant-design/decode-screen/report.md`: gfx1201
verdicts — dword-window loss on production shapes, u16-convert neutral,
non-temporal loss) remain valid evidence; the open questions they carried
factor into the files above.
