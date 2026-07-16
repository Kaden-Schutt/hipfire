# Cooperative MoE down+combine fusion: exact, scheduler-bound negative

## Verdict

Do not enable the cooperative down+combine fusion for A3B AR. Binding routed
experts into one workgroup removes the expanded-output round-trip and one
dispatch, but loses the scheduler freedom of the established one-wave-per-block
expanded kernel. The loss persists after eliminating register pressure as a
confound.

The experiment was reverted. The single-stream retained-PM4 champion remains
the expanded down GEMV followed by `moe_down_combine_k8_batched`.

## Design

Each block owns four output rows. A wave computes one routed expert using the
same K=512 two-group accumulator and shuffle order as
`gemv_hfq4g256_moe_down_k8_indexed_batched_expanded`. Expert results cross a
small LDS tile; wave zero applies routing weights in exact k=0..7 order and adds
the result to the residual. There are no atomics and no global expanded buffer.

Two scheduling shapes were tested:

- Eight-wave block: all eight experts execute concurrently, then one exact fold.
- Four-wave block: experts 0..3 and 4..7 execute in two batches while the same
  per-row accumulator is carried across the batches.

## Gates

- Host: `hiptrx`, gfx1201, automatic clocks.
- Model: Qwen 3.6 35B A3B MQ4R, Q8 KV, MTP off.
- Fifteen consecutive retained-PM4 positions against the HIP oracle.
- Context 128, 100 measured tokens, ten warmups, 30 rows.
- Opt-in was gfx1201-only; all candidate changes were reverted.

Every shape passed exact logits, KV, recurrent-state, and blob shadow parity.
The retained tape fell from 833 to 793 launches: exactly one combine launch was
removed in each of 40 MoE layers.

## Results

| Shape | Workgroup | VGPR | LDS | Candidate | Matched control | Delta |
|---|---:|---:|---:|---:|---:|---:|
| Generic eight-wave | 256 | 148 | 128 B | 185.909 tok/s | 194.114 tok/s | -4.23% |
| K=512 specialized eight-wave | 256 | **46** | 128 B | 186.458 tok/s | 191.793 tok/s | -2.78% |
| K=512 specialized four-wave | 128 | 54 | 64 B | 182.388 tok/s | 191.793 tok/s nearby control | -4.90% |

All kernels used zero scratch and zero spills. The specialized eight-wave
variant had half the live kernel's 92-VGPR footprint, so occupancy/register
pressure cannot explain the remaining regression.

## Mechanism

The established expanded kernel launches one wave per workgroup and lets the
hardware scheduler distribute routed-expert waves independently. The fused
kernel makes four or eight expert waves a single scheduling unit with mandatory
LDS barriers. The eight-wave form loses placement/MLP freedom; the four-wave
form adds a second compute-and-barrier phase without recovering it.

The earlier one-wave fused prototype failed in the opposite direction by
serializing all eight experts. Together the experiments bound this topology:
one wave has insufficient expert concurrency, while multi-wave cooperative
blocks constrain scheduling enough to erase the saved traffic and dispatch.

Artifacts on `hiptrx`:

```text
/home/kaden/.redline-work/hipfire-pm4-lean/.redline-work/
  moe-down-coop-shadow15.json
  moe-down-coop-control-30.json
  moe-down-coop-candidate-30.json
  moe-down-coop-v2-shadow15.json
  moe-down-coop-v2-candidate-30.json
  moe-down-coop-v2-control-30.json
  moe-down-coop-v3-shadow15.json
  moe-down-coop-v3-candidate-30.json
```
