# XDNA2 native-Q8 full-array gate

Date: 2026-07-23
Host: `hipx`
GPU: gfx1151, HIP device 1
NPU node: real `/dev/accel/accel1`

## Gate shape

- M=256, K=N=2048
- GPU-owned BF16 activation and F32 output allocations
- GPU-owned native Hipfire Q8_0 `W[N,K]`
- 32 AIE2P cores and eight shims
- one retained command and one weight import per projection
- mapped dma-buf and subsequent HIP-visible output verification

The artifact has the diagnostic arithmetic identity
`q8_w8a16_full_array_diagnostic`. Production loading still requires
`q8_w8a16_f32`, so this result cannot route a Hipfire projection.

## Implementation

The full-array source is under
`crates/redline-xdna/kernels/q8_full_array_xdna2/`; the verifier is
`crates/redline-xdna/examples/hip_q8_full_array.rs`.

The final diagnostic:

1. gathers native 32-byte Q8 values plus their FP16 scale without a model-sized
   repack;
2. uses an AIE2P four-way lookup to convert all signed Q8 byte values to BF16;
3. applies the scale and transposes 8x8 blocks into MMUL order;
4. uses 2x2-expanded 8x8x8 accumulators;
5. double-buffers bank-local K=512 activation and packed-weight panels in
   memory-tile SRAM; and
6. drains F32 output directly into the imported GPU allocation.

The AIE2P BF16 lookup layout differs from the public 128-bit-bank example:
XDNA2 duplicates 16 BF16 entries per 256-bit bank line. A guarded probe of all
256 raw signed byte values established this layout before it was used by the
projection.

## Results

The initial scalar decoder was exact but took 53.066 ms for one projection.
Vector conversion and 8x8 transposed stores reduced that to 5.644 ms. A
four-way lookup, 2x2 MMUL expansion, and retained K=512 panels produced the
following 30-submission results:

| Build | p50 | p99 | Effective throughput | Correctness |
|---|---:|---:|---:|---:|
| Native BF16 MMUL | 2,285.008 us | 4,556.432 us | 0.940 TOPS | exact |
| BFP16-emulated 8x8x8 | 1,674.234 us | 3,936.571 us | 1.283 TOPS | exact fixture |
| Retained-panel DMA-only | 258.594 us | 2,468.983 us | n/a | zero fixture |

The exact projection fixture had no NaN/Inf, `max_abs=0`, cosine 1.0, and
NRMSE 0 through both readback routes. BFP16 emulation still requires real
40-layer shadow parity; this synthetic exact result is not a numerics
certification.

The 340 us p50 / 400 us p99 complete-panel gate fails. The best p50 is 4.9x
over the limit and the cold-tail p99 is 9.8x over. Retained K=512 panels
reduce the steady transport floor into range, but do not make the whole
projection competitive:

- the four-row schedule decodes the same B data on each row;
- Q8 lookup, scaling, and transpose dominate steady execution;
- native BF16 MMUL alone is already above the total gate; and
- the first submission adds an approximately 2.2 ms cold tail.

## Verdict

Do not wire this artifact into the Qwen scheduler, including `shadow` or
`force`. `off` remains the only executable production state and the existing
HIP path is unchanged.

The next technically distinct experiment would need to remove decode
replication across AIE rows or use a separately certified scaled-int16 x int8
pipeline. Both change the kernel/dataflow contract; neither is a tuning pass
on this artifact. Persistent multi-chunk reuse can amortize decode on much
longer prompts or larger models, but it does not satisfy the required
single-panel adoption gate for Qwen.

Every live run used the external process timeout, a one-second internal ticket
wait, and `/tmp/hipfire-xdna.lock`. The node and `/dev/kfd` were holder-free
before each run. No gpukill, unbind/rebind, FLR, warmboot, or reboot was
required.
