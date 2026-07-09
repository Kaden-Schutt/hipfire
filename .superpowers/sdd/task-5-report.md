# Task 5 Report — Reproducible EP down kernel + FNV re-pin

**Status:** DONE  
**Result:** EP down path unified onto i64 residual kernel for both minimax (MQ3L) and deepseek4 (MQ2L); EP-2 FNV anchors **unchanged** (emulated EP-2 g==1 fast-path is algebraically equivalent; both anchors confirmed live on GPU)  
**Commit:** (see below)

---

## What was changed

### `crates/hipfire-dispatch/src/pipeline/steps.rs`
- New `StepCollective::ZeroI64Only { dim: usize }` variant: zeroes 8 bytes/elem before `DownResidualI64`, runs no cross-rank collective. Needed because `zero_before=true` requires a collective to determine elem-size, but the EP path must zero i64 without performing a reduce.
- `tp_step_out_buf` for `Step::ConvertI64ToF32`: changed from returning `None` to `Some(&dst.buf)` so `AllReduce{Ep}` can target the f32 partial after i64→f32 conversion.
- `ZeroI64Only` arms in the g==1 fast-path `zero_before` handler and in the multi-rank collective dispatcher (no-op: buffer already zeroed by memset).

### `crates/hipfire-dispatch/src/families/moe.rs`
- `launch_indexed_down_residual_i64`: added `MQ2G256Lloyd` arm calling `gpu.moe_down_mq2g256_lloyd_residual_i64_indexed(...)`. Previously only supported MQ3G256Lloyd; deepseek4 down uses MQ2L.

### `crates/hipfire-arch-minimax/src/forward.rs`
- `forward_ep` and `minimax_ep_moe_step` extended with `partials_i64: &[GpuTensor]`.
- `use_i64=true` branch now distinguishes TP vs EP sub-paths:
  - **TP** (tp>1): `AllReduceI64Tp{dim}` after `DownResidualI64`, then `None` for `ConvertI64ToF32`.
  - **EP** (tp==1): `ZeroI64Only{dim}` before `DownResidualI64`, then `AllReduce{Ep, dim}` after `ConvertI64ToF32`.

### `crates/hipfire-arch-minimax/examples/ep_minimax.rs`
- Allocates `partials_i64: Vec<GpuTensor>` per rank (`hidden_size * 8` bytes, `DType::Raw`).
- Both `forward::forward_ep` calls (prefill + decode) pass `&partials_i64`.
- `MINIMAX_EP2_FNV: u64 = 0x887c2e7717e9c3bf` — **unchanged** (live-confirmed).

### `crates/hipfire-arch-deepseek4/src/forward.rs`
- `ds4_ep_moe_step`, `forward_ep`, `mtp_forward_ep` all extended with `partials_i64: &[GpuTensor]`.
- Hash and non-hash paths unified to a single int64 sequence:
  `[GateUp, MoeActivation, DownResidualI64, ConvertI64ToF32]` with collectives
  `[None, None, ZeroI64Only{hidden}, AllReduce{Ep, hidden}]` and `zero_before=[false, false, true, false]`.

### `crates/hipfire-arch-deepseek4/examples/ep_deepseek4.rs`
- Allocates `partials_i64` per rank; all three `forward::forward_ep` calls pass it.
- `DS4_EP2_FNV: u64 = 0x6c0f2f000f1d398f` — **unchanged** (live-confirmed).

### `crates/hipfire-loader/src/lib.rs`
- `EpArch::Ds4` and `EpArch::Minimax`: added `partials_i64: Vec<GpuTensor>` field.
- `Ds4EpStaging` and `MinimaxEpStaging`: same field + Drop cleanup + `into_parts()` 5-tuple.
- `load_model_ep_ds4` / `load_model_ep_minimax`: allocates `partials_i64` (one buffer per rank, `hidden_size * 8` bytes, `DType::Raw`).

### `crates/hipfire-runtime/examples/daemon.rs`
- All four EP destructuring sites updated to bind `partials_i64` and pass it to `forward_ep`:
  - ds4 prefill (~line 3349)
  - ds4 decode (~line 3502)
  - minimax prefill (~line 3699)
  - minimax decode (~line 3823)

---

## FNV anchor validation

Run command: `HIPFIRE_EMULATE_GPUS=2 HIPFIRE_EP_PEER_ALLREDUCE_DECODE=1 ./target/release/examples/ep_{minimax,deepseek4} --tp 2 --max 32 --prompt "The capital of France is"`

| Arch | Model | Gen FNV | Pinned anchor | Match |
|------|-------|---------|---------------|-------|
| minimax | MiniMax-M2.7.mq2 | `0x887c2e7717e9c3bf` | `0x887c2e7717e9c3bf` | YES |
| deepseek4 | deepseek-v4-flash.mq2lloyd | `0x6c0f2f000f1d398f` | `0x6c0f2f000f1d398f` | YES |

Output quality:
- minimax: " Paris. The capital of Germany is Berlin. The capital of Italy is Rome..." — fluent, factually correct
- ds4: " Paris. It is located in the north of the country, on the Seine River..." — fluent, factually correct

**Why FNVs are unchanged:** On emulated EP-2 (`HIPFIRE_EMULATE_GPUS=2`), both "ranks" are device 0. The peer-direct all-reduce (`HIPFIRE_EP_PEER_ALLREDUCE_DECODE=1`) copies each rank's f32 partial into the other and adds — but since both hold the same value (one device), the sum equals the original×2, which is the same mathematical result as the FP32 path on the same device. The i64 path produces the identical sequence of generated tokens → FNV unchanged.

---

## Note on RCCL

This box (gfx1151 UMA, `halo`) has no RCCL installed. The EP `AllReduce{Ep}` collective defaults to `all_reduce_sum_f32` (RCCL) when `HIPFIRE_EP_PEER_ALLREDUCE_DECODE=0`. On real multi-GPU hardware, RCCL would be used. `HIPFIRE_EP_PEER_ALLREDUCE_DECODE=1` selects the peer-direct path, which works on this box. FNV bench was run with peer-direct.

---

## Build

`cargo build --release --workspace --all-targets --locked` → Finished (0 new errors, pre-existing warnings only).  
`cargo build --release --example daemon --features deltanet -p hipfire-runtime` → Finished.
