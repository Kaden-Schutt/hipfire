# XDNA2 native-Q8 persistent-microtile record

Date: 2026-07-23
Host: `hipx`
GPU: gfx1151, HIP device 1
NPU node: real `/dev/accel/accel1`

## Contract

Hipfire projection weights are row-major `W[M,K]`. Each Q8_0 K=32 block is:

```text
byte 0..1   little-endian IEEE FP16 scale
byte 2..33  32 signed int8 weights
```

The projection is `Y[N,M] = X[N,K] @ W[M,K]^T`. The XDNA2 API installed on
hipx provides BF16 x BF16 MMUL but no BF16 x int8 MMUL, so the selected path
decodes only the active B microtile to BF16 in AIE-local memory.

The diagnostic arithmetic identities are intentionally distinct from
production:

- `q8_decode_bf16_diagnostic`
- `q8_w8a16_microtile_diagnostic`
- production remains `q8_w8a16_f32`

The production loader rejects both diagnostics.

## Sources

- `crates/redline-xdna/kernels/q8_decode_xdna2/`
- `crates/redline-xdna/kernels/q8_persistent_microtile_xdna2/`
- `crates/redline-xdna/examples/hip_q8_decode.rs`
- `crates/redline-xdna/examples/hip_q8_persistent_microtile.rs`

The source was copied into an isolated `/tmp/hipfire-xdna-q8-decode.*` build
root on hipx. No source, artifact, or ledger in the existing dirty
`mlir-aie-xdna-gemm-autoresearch` campaign was changed.

Toolchain:

- mlir-aie `0.0.1.2026052005+886d932`
- llvm-aie/Peano `21.0.0.2026072201+db5014ed`
- no XRT library in the Rust dispatch/runtime path

## Decoder proof

Shape: K=2048, 64 native Q8_0 blocks.

The fixture cycles exactly representable positive and negative FP16 scales and
signed weights. All 2,048 decoded BF16 values matched bit-for-bit through:

1. the imported dma-buf mapping after XDNA completion; and
2. a subsequent HIP device-to-host copy of the GPU-owned output allocation.

Ten retained submissions:

| Metric | Result |
|---|---:|
| p50 | 269.845 us |
| p99 | 392.254 us |
| mismatches | 0 / 2048 |

## Persistent W8A16 microtile

Shape per command:

- native Q8_0 `W[16,64]`
- BF16 activation tiles with 8 rows per logical chunk
- 2, 8, or 16 logical chunks
- AIE2P 4x8x8 BF16 MMUL
- F32 tile-major output
- one B import and one B decode per command

Thirty retained submissions per shape:

| Chunks | Rows | p50 | p99 | p50 / chunk | Cosine | NRMSE |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 16 | 173.685 us | 326.080 us | 86.843 us | 1.0 | 0 |
| 8 | 64 | 179.817 us | 332.072 us | 22.477 us | 1.0 | 0 |
| 16 | 128 | 187.902 us | 366.276 us | 11.744 us | 1.0 | 0 |

The total p50 rises only 14.217 us from 2 to 16 chunks while useful MMUL work
grows 8x. Amortized p50 falls 7.4x. This validates the scheduling hypothesis:
decode/upload cost must be paid at the B-panel boundary, then reused across
prompt chunks inside one command.

## Safety finding

The first 16-chunk build used default double-buffered whole-A/C FIFO objects
and hit the external 20-second guard. The process exited and left no
`/dev/accel/accel1` or `/dev/kfd` holder and no new kernel fault.

A depth-1 rebuild made the memory issue observable at 2 chunks: an implicit
2 KiB C++ decoded array overlapped an allocator-placed activation object and
left part of chunk 2 untouched. The final design exposes the decoded panel as
an IRON `Buffer`; generated placement assigns it its own bank. All three shapes
then passed.

No device recovery action was required. In particular, this run did not use
gpukill, driver unbind/rebind, FLR, warmboot, or reboot.

## Next scale point

This proof deliberately does not claim the complete-panel gate. The next
artifact must:

1. retain a decoded 16x64 B tile;
2. stream 8x64 A tiles from every 256-token chunk;
3. preserve F32 partial sums across all 32 K tiles for K=2048;
4. use multiple shims/cores for output panels; and
5. time the complete GPU-ready -> NPU -> GPU-visible panel against 340 us p50
   and 400 us p99.

Allocating a whole 256-token chunk on one compute tile is rejected by this
experiment; A/C must be streamed and independently ping-ponged.
