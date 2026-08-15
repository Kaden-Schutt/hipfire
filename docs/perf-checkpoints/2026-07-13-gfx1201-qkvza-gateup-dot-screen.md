# gfx1201 QKVZA and routed gate/up dot screen: replay-exact negatives

## Verdict

Do not ship the gfx1201 Q8_1 activation sidecar, mixed-sign dot, or single-token
QKVZA WMMA experiments. The valid candidates are replay-exact, but none beats
the 200.834 tok/s no-environment bookend. All implementation changes were
removed; the 201 tok/s conv+QK-normalization champion remains unchanged.

gfx1201 does not expose the legacy `dot1-insts` used by the gfx906
`__builtin_amdgcn_sdot4` kernels. The native scalar-dot probe therefore used
`__builtin_amdgcn_sudot4`, which lowers unsigned HFQ4 nibbles multiplied by
signed Q8_1 activations to `v_dot4_i32_iu8`.

## Replay-correct fused sidecar

The first 205.684 tok/s retained-PM4 result was invalid. The original probe
called the internal standalone Q8_1 converter, which Redline did not record;
the 733-dispatch tape therefore consumed stale Q8_1 data. Its 15-position
shadow correctly failed logits, KV, and recurrent-state parity.

The corrected implementation emitted `block_q8_1_mmq` directly from the
preceding `fused_rmsnorm_mq_rotate` values already held in registers. The RMS
node wrote both F32 FWHT output and the Q8_1 sidecar; QKVZA or gate/up read the
sidecar through a prequantized ABI. This retained 733 dispatches and passed
PM4-versus-HIP shadow parity (15 positions for QKVZA R1, four-position prefix
gate for routed gate/up R1).

## Five-shape QKVZA screen

R9700/gfx1201, automatic clocks, Qwen 3.6 35B-A3B MQ4R, Q8 KV, retained PM4,
ten warmups, 100 measured positions, five rows per shape:

| Rows/workgroup | HIP | Retained PM4 |
|---:|---:|---:|
| 1 | 169.080 | 197.779 tok/s |
| 2 | 172.570 | 197.949 tok/s |
| 4 | 172.553 | 197.551 tok/s |
| 8 | 172.609 | **198.253 tok/s** |
| 16 | 172.719 | 195.826 tok/s |

The row-tiled FP16 WMMA control was also replay-exact but measured only
187.980 tok/s: a 16x16 WMMA tile computes fifteen duplicate batch columns at
single-token decode. A four-launch reuse of the existing i8 MMQ set kernel
expanded the tape from 733 to 943 dispatches and was immediately
noncompetitive; it also exposed dormant module/symbol ownership debt in that
prefill helper and was not pursued.

## Five-shape routed gate/up screen

The same fused Q8_1 producer fed a new indexed gate/up dot kernel while
preserving device-side expert-pointer and top-k routing:

| Rows/workgroup | HIP | Retained PM4 |
|---:|---:|---:|
| 1 | 163.599 | 194.669 tok/s |
| 2 | 162.619 | **197.492 tok/s** |
| 4 | 162.137 | 197.393 tok/s |
| 8 | 162.923 | 195.456 tok/s |
| 16 | 165.562 | 196.334 tok/s |

The no-environment bookend immediately afterward was 169.865 tok/s through
HIP and **200.834 tok/s** through retained PM4. The loss is therefore not
automatic-clock drift. On gfx1201, the scalar mixed-sign dot issue shape and
Q8_1 scale/correction work do not beat the established FP32 four-chain kernels.

## Next target

The next structural candidate is recurrence-plus-normalization fusion:

```text
gated_delta_net_q8_compact2_b2 -> gated_norm_f32 -> residual GEMV
```

Folding the per-head gated normalization into the GDN producer removes one
launch and the materialized attention-output write/read in each of 30
DeltaNet layers. It matches the producer/consumer traffic-removal pattern that
made conv+QK normalization the current +1.93% winner, without changing the
matrix GEMV arithmetic already shown to be near its gfx1201 ceiling.

Artifacts:

```text
hiptrx:/home/kaden/.redline-work/hipfire-w64-raw/.redline-work/dp4a-gfx1201/
```
