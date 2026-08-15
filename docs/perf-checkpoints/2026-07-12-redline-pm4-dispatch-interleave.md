# Retained PM4 dispatch interleave: bit-exact, not reproducible

## Verdict

Reverted. Programming `COMPUTE_DISPATCH_INTERLEAVE` once at the IB head is
bit-exact for both 64 and 256 threads, but neither setting produces a stable
performance win over the single-stream retained-PM4 default.

## Method

- Host: `hiptrx`, gfx1201, automatic clocks.
- Model: Qwen 3.6 35B A3B MQ4R, Q8 KV.
- Stable capture: 833 launches, sequence hash `8ba4c8d66f32d116`.
- Candidate was one `SET_SH_REG` at the tape head; all dispatches, waits,
  acquires, kernels, and kernargs were unchanged.
- Both 64- and 256-thread arms passed the fifteen-position logits, KV, and
  recurrent-state shadow exactly.
- Context 128, 100 measured tokens, ten warmups, 30 rows; 64 was repeated in
  reverse order.
- Fixed-seed sampled eight-turn serve session through context 20,069.

## Results

| Run order | Control median | 64 median | 256 median |
|---|---:|---:|---:|
| control then candidates | 192.135 tok/s | 193.578 tok/s | 193.016 tok/s |
| 64 then control | **194.797 tok/s** | 193.110 tok/s | - |

The apparent first-pass gain reversed with execution order. The product gate
confirmed the negative:

| Eight-turn sampled session | Average decode |
|---|---:|
| control | **167.013 tok/s** |
| 64-thread interleave | 165.263 tok/s |

The 64-thread arm is 1.05% slower across the eight-turn session. It is therefore
not part of the retained-PM4 default. The experiment changed only host-side PM4
encoding, preserving gfx1100 kernel `.text` byte-for-byte.

Artifacts on `hiptrx`:

```text
/home/kaden/.redline-work/hipfire-pm4-lean/.redline-work/pm4-lean/interleave/
```
