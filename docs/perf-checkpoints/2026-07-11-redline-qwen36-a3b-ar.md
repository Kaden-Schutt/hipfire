# Redline Qwen3.6 A3B ordinary-AR checkpoint

**Date:** 2026-07-11

**Status:** certified on the kernel-oracle composition; transplanted onto `redline`

**Workload:** `qwen3.6-35b-a3b.mq4r`, ordinary autoregressive decode

**Host:** `hiptrx`, Radeon AI PRO R9700 (`gfx1201`), GPU 0

**Clock policy:** automatic; no clock pinning or performance-level override

## Result

The retained gfx12 PM4 replay path improves the already-tuned HipGraph product
path from **165.839 to 178.320 tok/s**, a **1.07526x** speedup. This is a
matched resident-daemon comparison using 10 measured runs of 100 decode
positions after three 100-position warmups.

| Route | Minimum | Median | Maximum |
| --- | ---: | ---: | ---: |
| Tuned HipGraph | 165.592 tok/s | 165.839 tok/s | 165.920 tok/s |
| Redline retained PM4 | 176.900 tok/s | 178.320 tok/s | 178.812 tok/s |

The earlier conservative Redline policy measured 164.220 -> 174.087 tok/s
(1.06009x). Removing one additional proven-independent boundary between the
shared-expert down GEMV and routed-expert gate/up GEMV raised the Redline median
to 178.320 tok/s. Because clocks remain automatic and the two checkpoints used
separate daemon processes, the matched HipGraph-to-Redline ratio is the primary
reported result.

## What changed

Redline does not substitute or rewrite the winning kernels. The tested source
base was `origin/feat/rdna-kernel-oracle` at `35502d550`, which contains the
`loop/gfx1201` winners through `53aab4775`. The clean publication branch is
instead based on master at `db492c9cf`, where PR #522 landed those gfx1201
winners without the kernel-oracle experimental history.

One ordinary A3B decode position records 833 launches across 26 unique kernels
(sequence hash `8d5620ca2ca8a536`). Redline lowers that fixed tape to one
retained PM4 indirect buffer. The final buffer contains 34,563 dwords and is
submitted through one public ROCr vendor AQL packet.

The extra overlap is safe because the adjacent launches have disjoint live
resources:

- `gemv_hfq4g256_residual_sigmoid_scaled_gpu` reads the shared-expert rotated
  activation and writes the shared residual;
- `gemv_hfq4g256_moe_gate_up_k8_indexed` reads the routed activation and writes
  separate gate/up buffers;
- their first dependency join is the later routed/shared MoE combine.

The boundary occurs once in each of the model's 40 layers. Redline omits only
that intermediate compute-idle wait; the next dependent boundary performs the
fan-in. The artifact resolver also binds the runtime launch names to the actual
loaded shared-residual and indexed-MoE code objects, failing closed if either
object is unavailable.

## Correctness gate

The PM4 route was compared against both ordinary HIP execution and an exact
HIP-kernarg-blob oracle for 15 consecutive token positions:

| Surface | Bytes | Hash | Result |
| --- | ---: | --- | --- |
| Logits | 993,280 | `9874244965e2c7d6` | identical |
| KV cache | 22,297,600 | `fa5f3bb2b32fffcd` | identical |
| Recurrent state | 50,626,560 | `609db41ffad8ceb6` | identical |

The aggregate gates all passed: `bit_exact`, `blob_bit_exact`,
`logits_equal`, `kv_equal`, and `recurrent_equal`. The 15-position shadow pass
took 86,396.945 us through retained PM4 versus 105,077.758 us through ordinary
HIP. This shadow timing is diagnostic; the 10-by-100 product benchmark above is
the throughput result.

## Reproduction

The product harness keeps each model resident within an arm, resets and primes
the full Qwen decode path for every row, sets Q8 KV, disables CASK and DFlash,
and never changes clocks:

```bash
HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 \
python3 -m tools.redline bench \
  --model /home/kaden/.hipfire/models/qwen3.6-35b-a3b.mq4r \
  --daemon target/release/examples/daemon \
  --context 128 \
  --iterations 100 \
  --warmups 3 \
  --runs 10 \
  --transport pm4 \
  --max-seq 2048 \
  --timeout 1200 \
  --work-dir .redline-work/a3b-r3 \
  --out .redline-work/a3b-r3/product-pm4-mq4r-overlap.json
```

The parity run used `scripts/redline_daemon_harness.py` with `--skip-prefill`,
`--pm4`, and `--shadow-iterations 15`. Its raw report is
`.redline-work/a3b-r3/shadow15-overlap.json` in the isolated hiptrx test
checkout.

## Scope and publication note

This checkpoint covers ordinary AR only. DFlash and MTP were disabled, and the
gain is dispatch replay plus selective fencing on top of the existing tuned
gfx1201 kernels. The GPU measurements were collected on the kernel-oracle
composition before the five Redline commits were transplanted onto current
master. The master-based `redline` tree passed the replay unit suites and daemon
compilation, but has not been GPU-rerun; this document deliberately keeps that
distinction explicit.
