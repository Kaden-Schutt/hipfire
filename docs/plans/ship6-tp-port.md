# Ship 6 — Tensor-Parallel (TP) Port Plan

**Goal:** Port the complete TP implementation from `origin/tp-mtp-prototype`
(tip `cceb07c0`) onto today's base (`d9dbe834`, `capstone/forward-as-pipeline`
= integration/dispatch-unification). **Working-first**: get it compiling +
the TP=2≡TP=1 parity gate green. NO super-op lowering yet — that is a later
phase, and we explicitly do **not** over-invest in structure that the later
forward-as-pipeline lowering will refactor away.

- **Merge-base:** `a7a8d89b` (~1497 commits behind).
- **Parity gold:** TP=2 output ≡ TP=1 output (prototype reports 7.5e-7 max
  abs diff / 32-of-32 token match).
- **TP needs ≥2 visible GPUs.** Validate on **hiptrx** (4× gfx1201/RDNA4,
  use devices 0+1). **hipx CANNOT validate TP=2** (only 1 usable big-VRAM
  device — gfx1151 on HIP dev 1; gfx1010 5700XT on dev 0 is a different
  arch and too small, so it fails the `preflight_vram_with_opts` arch-match
  + VRAM-delta gate). k9lin is single-GPU → also cannot run TP=2; it remains
  the perf/coherence box for the TP=1 (degenerate) path only.

## Base-reality findings (what actually drifted)

Verified against current source, not folklore:

| Component | Status on current base | Implication |
|---|---|---|
| `hip-bridge` FFI primitives (`malloc`/`memset`/`stream_*`/`memcpy_*`/`set_device`/`device_count`/`DeviceBuffer::as_ptr`) | **All present** (`ffi.rs`) | `rccl.rs` + smoke compile against them unchanged |
| `Stream::raw_ptr()` | **MISSING** — current only has `Stream::as_raw()` (`ffi.rs:1321`). Prototype `ffi.rs:1334` ADDED `raw_ptr()`. | All TP callers use `.raw_ptr()`. Fix: re-add the 4-line `raw_ptr()` alias to `ffi.rs` (cheapest, keeps prototype callers byte-identical) **or** sed callers to `as_raw()`. Recommend re-add the alias. |
| `DeviceBuffer` pub at `hip-bridge` top level | **Yes** (`lib.rs:61`, `unsafe impl Send`) | smoke `use hip_bridge::DeviceBuffer` works |
| `hip-bridge` deps (`libloading`, `thiserror`) | identical to prototype | `rccl.rs` needs only `libloading` + `std` → no Cargo change |
| `Gpus` struct (`multi_gpu.rs:58`) | identical to prototype **except missing `rccl_comms` field** + the 3 TP fns | clean additive port (field + `init_tp` + `ensure_rccl` + `all_reduce_sum_f32`); helpers `resolve_device_ids`/`construct_devices`/`preflight_vram_with_opts`/`from_parts`/`single` all unchanged → just thread `rccl_comms: None` into each constructor |
| `Gpu` fields/methods (`dispatch.rs`): `device_id` (335), `active_stream` (341), `bind_thread` (395) | **present** | `ensure_rccl`/`all_reduce_sum_f32` compile |
| `Gpu::add_f32` (norm.rs:115), `add_inplace_f32` (172), `GpuTensor::sub_offset` (dispatch.rs:125) | **present** | `tp_allreduce_add` / `_batched` compile |
| `config::get().tp_use_rccl` | **MISSING** field | add `tp_use_rccl: Option<bool>` to `RuntimeConfig` + parse `HIPFIRE_TP_USE_RCCL` |
| `tp_shard.rs` (568 lines, pure-CPU) | **NEW**, zero external deps (only `std::ops::Range`) | drops in verbatim; its 16 unit tests run on CPU/CI |
| `WeightTensor` (llama.rs:521) | **byte-identical** (`buf`/`gpu_dtype`/`m`/`k`/`row_stride`/`paro`/`awq_scale`) | `load_weights_tp` slicing path (`awq_scale`, `paro`, `free_tensor`) ports cleanly |
| `LayerWeights` enum (qwen35.rs:649) | **byte-identical** (DeltaNet/FullAttn/DeltaNetMoe/FullAttnMoe) | pattern-matches port unchanged |
| `DType` enum (dispatch.rs:137) | **DRIFT** — prototype has `PARO4G128` + `PARO4G128T`; current does **NOT**. Current has a `fused_fa3_hfq6` branch the prototype lacks. | independent feature divergence. Prototype `run_fa_layer_body` references `DType::PARO4G128T` → **won't compile**. The FaPhase seam must be re-cut on the CURRENT body, not pasted. |
| `run_fa_layer_body` (qwen35.rs:11030) | **present but PRIVATE, NO `FaPhase` param** | `FaPhase` seam (proto enum at 12901) must be **re-introduced** on the current (drifted) body; make it `pub` |
| `FaPhase` enum | **MISSING** | new |
| `run_dn_layer_body` | **MISSING as standalone fn** (DeltaNet is inline in current forward) | prototype extracted it (`pub`, 13975). Must extract it on current base from the inline DeltaNet path. |
| `run_fa_ffn_body` / `run_fa_ffn_body_sharded` | **MISSING** | new (proto 13643 / 13667) |
| `run_moe_ffn_ep` | **MISSING** | new (proto 13514, EP-MoE) |
| `embedding_lookup_{q8,hfq4g256,hfq4g128,F32}` (embedding.rs) | **all present** | `forward_scratch_tp` embed dispatch ports |
| `hipfire-dispatch` crate + `DispatchCtx`/`GemmFamily::run_key`/families/pipeline/superop | **NEW on current** (`qwen35.rs:20-29`); **prototype has ZERO** | the dispatch-unification refactor is the single biggest structural delta. The current `run_fa_layer_body` body has partially migrated to dispatch families. **The TP seam wraps a body that itself differs.** |
| `mtp_spec.rs` | present (122 KB); current `spec_step_mtp` at 1253; proto `spec_step_mtp_tp` at 1698 | TP+MTP is a tail-stage rebuild on the current `spec_step_mtp` |
| TP wiring into engine/CLI/daemon | **NONE on prototype** — TP is **example-driven only** (`tp_*` examples). No daemon dispatch, no CLI flag. | scope is bounded: we port library fns + examples + parity, not a serving path |

---

## Stage-by-stage plan

### Stage 1 — ShardConfig + `Gpus::init_tp` scaffold
*(prototype 2b6a7144)*

**Files**
- ADD `crates/hipfire-runtime/src/tp_shard.rs` (verbatim from proto; 568 lines incl. 16 tests).
- MODIFY `crates/hipfire-runtime/src/lib.rs` — `pub mod tp_shard;`.
- MODIFY `crates/hipfire-runtime/src/multi_gpu.rs` — add `rccl_comms: Option<RcclComms>` field; add `init_tp`; thread `rccl_comms: None` into `init_uniform`/`init_layers`/`single`/`from_parts`. (Import `RcclComms` — but that doesn't exist until Stage 2; gate the field/import behind Stage 2 or land Stage 1+2 together. **Recommend folding Stage 1 field-add into Stage 2** and keeping Stage 1 = pure `tp_shard` + `init_tp` that sets nothing rccl-related yet. `init_tp` only needs the existing constructors.)

**API drift**
- `init_tp` uses `resolve_device_ids`/`construct_devices`/`preflight_vram_with_opts`/`HipError::new` — all present unchanged on current `multi_gpu.rs`.
- `Self { rccl_comms: None, … }` literal — needs the field, which is Stage 2. Sequence accordingly.

**Validation:** `cargo test -p hipfire-runtime tp_shard` (CPU, no GPU). All 16 ShardConfig tests must pass. CI-runnable.

**Effort:** S. **Risk:** trivial — pure CPU, no kernels.

---

### Stage 2 — RCCL FFI wrapper + `all_reduce_sum_f32` collective + smoke
*(prototype 6185d094 + 796af3fd)*

**Files**
- ADD `crates/hip-bridge/src/rccl.rs` (verbatim, 356 lines).
- MODIFY `crates/hip-bridge/src/lib.rs` — `mod rccl;` + `pub use rccl::{RcclComms, RcclDataType, RcclError, RcclRedOp, RcclResult, NCCL_SUCCESS};`.
- MODIFY `crates/hip-bridge/src/ffi.rs` — re-add `Stream::raw_ptr()` (4 lines) [see drift table].
- MODIFY `crates/hipfire-runtime/src/multi_gpu.rs` — add `rccl_comms` field (if not in Stage 1), `ensure_rccl`, `all_reduce_sum_f32`; import `RcclComms`.
- MODIFY `crates/hipfire-runtime/src/config.rs` — add `tp_use_rccl: Option<bool>` to `RuntimeConfig` + parse `HIPFIRE_TP_USE_RCCL` (mirror existing `Option<bool>` env pattern, e.g. `draft_f16`).
- ADD `crates/hip-bridge/examples/rccl_smoke.rs` (verbatim).
- ADD `crates/hipfire-runtime/examples/tp_allreduce_smoke.rs` (verbatim).

**API drift**
- `rccl.rs` is fully self-contained (`libloading` + `std` only) → compiles as-is.
- Smoke examples call `streams[r].raw_ptr()` and `stream.raw_ptr()` → fixed by the `ffi.rs` re-add.
- `tp_allreduce_smoke.rs` calls `Gpus::init_uniform`, `enable_peer_all`, `dev.active_stream`, `all_reduce_sum_f32`, `hip.stream_create/synchronize/destroy`, `memcpy_htod/dtoh` — all present.
- `ensure_rccl` reads `crate::config::get().tp_use_rccl` → the config add above.

**Validation:**
1. `cargo build -p hip-bridge --example rccl_smoke` + `cargo build -p hipfire-runtime --example tp_allreduce_smoke` (compile only — CI/k9lin).
2. **hiptrx devices 0+1:** `HIP_VISIBLE_DEVICES=0,1 HIPFIRE_TP_BENCH_N=2 cargo run -p hipfire-runtime --release --example tp_allreduce_smoke`. Gold check: rank r filled with (r+1).0 → after all-reduce-sum each rank reads `N*(N+1)/2` (= 3.0 for N=2). Latency ~110 µs floor at 4 KB.
3. Requires `librccl.so` present (`/opt/rocm/lib/librccl.so.1` or `apt install rccl`) on hiptrx.

**Effort:** M. **Risk:** RCCL not installed / wrong soname on hiptrx → `init_all` fails (recoverable error, clear message already wired). Single-process-multi-device comm init is the classic gotcha; the prototype already uses `ncclCommInitAll` (one-shot) which sidesteps `ncclCommInitRank` ordering.

---

### Stage 3a — FullAttn TP machinery: `FaPhase` seam on a real layer + wo-partial all-reduce smoke
*(prototype fb03ebd1, 5e2712ee, 338da6be)*

**THE crux stage.** The prototype split `run_fa_layer_body` into phases via a
`FaPhase<'a> { Full, TpAttn{mask}, TpFfn, TpFfnShard }` enum and extracted
`run_fa_ffn_body` / `run_fa_ffn_body_sharded`. On current base the body has
**drifted** (dispatch-families migration + `fused_fa3_hfq6` branch + no PARO
variants). **Do NOT paste the prototype body.** Re-cut the seam on the
current body.

**Files**
- MODIFY `crates/hipfire-arch-qwen35/src/qwen35.rs`:
  - Add `pub enum FaPhase<'a>` (verbatim — it's just a 4-arm enum).
  - Make `run_fa_layer_body` `pub` + add `phase: FaPhase` param. At top: `FaPhase::TpFfn → run_fa_ffn_body(...)`, `FaPhase::TpFfnShard → run_fa_ffn_body_sharded(...)`. In the attention output stage, branch on `FaPhase::TpAttn { mask }` to mask non-local Q-heads + run partial `wo` (port the proto attn-mask logic from 13339, **adapted** to the current attn output codepath).
  - Extract `run_fa_ffn_body` and `run_fa_ffn_body_sharded` from the current inline FFN code (the dense FFN block that follows attention in the current body).
  - Update the **two** existing private callers (forward_scratch :9784, prefill :7929) to pass `FaPhase::Full` — must remain byte-identical (this is the regression guard).
- ADD `crates/hipfire-arch-qwen35/examples/tp_wo_allreduce_smoke.rs` (verbatim — uses `LayerType`, `LayerWeights`, `ShardConfig`, `Gpus::init_tp`, `wo_col_range`).
- ADD `crates/hipfire-arch-qwen35/examples/tp_fa_layer_parity.rs` + `tp_attn_parity.rs` (the FaPhase parity harnesses).

**API drift**
- The body references current dispatch-family calls (e.g. `DispatchCtx`, `GemmFamily::run_key`) where it has migrated — the seam must thread these through, not the prototype's direct `gpu.fused_qkv_*` calls. **Read the current 11030 body end-to-end before cutting.**
- `DType::PARO4G128T` branch from proto body → omit (current lacks it; current has `fused_fa3_hfq6` instead — keep current's branches).
- attn-mask GpuTensor: proto `TpAttn{mask: Option<&GpuTensor>}` zeroes attn output outside local Q-heads — verify the current attn-out tensor name/layout matches.

**Validation**
1. **TP=1 byte-identical guard (mandatory):** with `FaPhase::Full` wired into the existing callers, run `./scripts/coherence-gate.sh` (k9lin + hiptrx) → must be unchanged from pre-port. This proves the seam is non-invasive.
2. **hiptrx 0+1:** `tp_wo_allreduce_smoke` — each rank zeroes attn_out outside its `wo_col_range`, runs partial wo, all-reduces; sum ≡ full-`wo` single-GPU output (max abs diff < ~1e-6).
3. **hiptrx 0+1:** `tp_fa_layer_parity` — one full FA layer via TpAttn+TpFfn phases ≡ Full phase.

**Effort:** L. **Risk:** HIGHEST. The drifted body + dispatch-families mean the seam is a re-implementation, not a transplant. Easy to break TP=1 byte-identity. Mask-vs-shard correctness is subtle (gated wq Q+gate interleave per head).

---

### Stage 3 — `forward_scratch_tp` single-token attn-shard forward
*(prototype 9c42c928)*

**Files**
- MODIFY `qwen35.rs`: add `pub fn forward_scratch_tp` (proto 15872), `fn tp_allreduce_add` (15795), `local_attn_config` (15754). Extract `pub fn run_dn_layer_body` from the current inline DeltaNet forward path (DeltaNet runs **replicated** in TP, no all-reduce — deterministic state stays in sync).

**API drift**
- `forward_scratch_tp` loops layers calling `run_fa_layer_body(…, FaPhase::TpAttn/TpFfn)`, `run_dn_layer_body(…)`, then `tp_allreduce_add` → all from Stage 3a + the DN extraction.
- `tp_allreduce_add` uses `gpus.all_reduce_sum_f32` + `add_f32` + scratch `.x`/`.o` (all present).
- `local_attn_config` clones `Qwen35Config` with local head counts — verify current `Qwen35Config` fields (n_heads/n_kv_heads/head_dim/dim/hidden_dim) match what proto mutates (config struct at qwen35.rs:137).
- embedding dispatch (`EmbeddingFormat::*`) — all methods present.
- MoE arms call `run_moe_ffn_ep` (Stage 3e) — stub/`unimplemented!()` until then; dense FullAttn + DeltaNet are the Stage 3 scope.

**Validation:** **hiptrx 0+1, the GOLD gate:** `tp_attn_parity` driving a real
single-token decode through `forward_scratch_tp` (TP=2, **replicated** weights)
≡ single-GPU `forward_scratch` (TP=1). Target: 7.5e-7 max abs diff, 32/32
token argmax match. Use a dense + DeltaNet model (0.8B-class) that fits 2 cards.

**Effort:** L. **Risk:** High — first end-to-end TP forward; DeltaNet replicated-state drift across ranks (must be bit-deterministic) is the classic failure.

---

### Stage 3b — per-rank FullAttn weight slicing + AWQ sidecar slicing
*(prototype 841d2cf8, 2a3857ae)*

**Files**
- MODIFY `qwen35.rs`: add `pub fn load_weights_tp` (proto 3517). Delegates to `load_weights` then row-slices `wq` (`wq_row_range`), col-gathers `wo` (`wo_col_range`), splits FFN gate/up (col) + down (row), frees the full buffers (+ AWQ sidecar via `free_full`).

**API drift**
- `load_weights` signature unchanged `(hfq, config, gpu)`. `WeightTensor`/`awq_scale`/`paro`/`free_tensor` byte-identical → clean.
- `wo` col-gather is a per-row gather (non-contiguous) for row-major quant blobs — verify the current upload helpers expose a column-gather (or port the proto's slicing util). **Check whether proto added an upload-cols helper to rdna-compute** (the row-slice is contiguous and cheap; the col-gather is the load-bearing one).
- AWQ sidecar: scale vector is length-K → slices with the **input** dim. For `wq` (row/output shard) the awq_scale (over K=input) is **replicated**, not sliced; for `wo`/`w_down` (input shard) the awq_scale **is** sliced. Confirm the proto handles this asymmetry correctly when porting.

**Validation:** Re-run the Stage 3 gold gate but with **sharded** weights
(`load_weights_tp` + `local_attn_config(global, shard)`, `fa_masks=None`)
instead of replicated. TP=2 sharded ≡ TP=1. This is the real
compute/memory-savings path.

**Effort:** M. **Risk:** Medium — wo col-gather + AWQ-scale replicate-vs-slice asymmetry are the bug magnets.

---

### Stage 3c — DeltaNet sharding + lean-sync decode
*(prototype 590ef74e)*

**Files**
- MODIFY `qwen35.rs`: extend `load_weights_tp` to shard DeltaNet wqkv by VALUE head (KEY/QUERY follow GQA repeat-interleave ratio); `forward_scratch_tp` DN arm runs local heads + all-reduce. Uses `ShardConfig::validate_deltanet` / `dn_value_head_range` / `dn_key_head_range` (already in `tp_shard.rs`).
- `DeltaNetState` (qwen35.rs:861) sized for local value heads per rank.

**API drift**
- `DeltaNetState` field/method drift — verify the current state struct (861) constructor signature matches what proto's sharded path expects (head-count-parameterized).
- lean-sync: producing GEMM + RCCL all-reduce + add all on each rank's `active_stream`, NO host `device_synchronize` — verify current DN kernels honor the active_stream (per CLAUDE.md memset-gating note).

**Validation:** Gold gate on a DeltaNet-heavy model (27B-3.6-class — but that
needs more VRAM than 2× gfx1201 32 GB at full precision; use an MQ4 27B that
fits 2 cards, or a small DN model). TP=2 sharded-DN ≡ TP=1.

**Effort:** M. **Risk:** Medium-high — DeltaNet recurrent state sharding + GQA-group preservation per rank.

---

### Stage 3d — batched-TP prefill `forward_prefill_chunk_tp`
*(prototype b1b850c4)*

**Files**
- MODIFY `qwen35.rs`: add `pub fn forward_prefill_chunk_tp` (proto 16116) + `fn tp_allreduce_add_batched` (15826). Uses `PrefillBatchScratch` (current struct at qwen35.rs:5370), `partials[r]` `[N×dim]`, `x_batch.sub_offset`, `add_inplace_f32`.

**API drift**
- Current `forward_prefill_chunk` (7716) has migrated to dispatch families / batched GEMM. The TP batched path must mirror the **current** prefill structure, not the proto's. **This is a second high-drift seam** (same class as 3a).
- `PrefillBatchScratch` field drift — verify `x_batch` exists/sized as proto expects (5370).

**Validation:** **hiptrx 0+1:** batched TP prefill (N tokens) TP=2 ≡ TP=1 prefill; then decode continuation argmax-matches. Gold 7.5e-7 / token match over the prefill+first-N-decode window.

**Effort:** L. **Risk:** High — second dispatch-drift seam; batched all-reduce on the live `n·dim` prefix only (off-by-N tail bugs).

---

### Stage 3e — EP-MoE: expert helpers + `load_weights_tp` MoE sharding + decode forward
*(prototype 93f63b63, 06de8122)*

**Files**
- `tp_shard.rs` expert helpers (`experts_per_rank`/`owns_expert`/`experts_on_rank`/`expert_to_rank`/`ExpertAssign`) — **already in Stage 1's verbatim port**.
- MODIFY `qwen35.rs`: add `fn run_moe_ffn_ep` (proto 13514) — each rank computes ONLY its owned experts into a `[N×dim]` partial, all-reduced. Extend `load_weights_tp` to load only owned experts per rank. Wire the MoE arms in `forward_scratch_tp` (replace the Stage-3 stub).

**API drift**
- Current MoE forward routes through `hipfire_dispatch::families::moe` (`MoeDtypes`/`MoeParams`, qwen35.rs:4581+). `run_moe_ffn_ep` must build the EP partial on top of the **current** MoE family call, not the proto's direct expert loop. **Third dispatch-drift seam.**
- `FullAttnMoe` / `DeltaNetMoe` LayerWeights arms — present on current.

**Validation:** **hiptrx 0+1:** A3B-class MoE model (qwen3.5-A3B MQ-quant fitting 2 cards), TP=2↔1 decode parity. ExpertAssign Stride vs Contiguous both validated.

**Effort:** L. **Risk:** High — MoE dispatch-family drift + expert→rank routing + all-reduce of expert partials.

---

### Stage 3f — shard MoE-layer attention + Q8F16 group size
*(prototype 89a1eabe, 53a43937)*

**Files**
- MODIFY `qwen35.rs`: enable attention sharding **within** MoE layers (combine 3b attn-shard with 3e EP-MoE in the same layer); handle Q8F16 group-size for the MoE-attn shard.

**API drift:** combinatorial — both seams (attn-shard + EP-MoE) active per layer. Verify Q8F16 group-size handling matches current quant dispatch.

**Validation:** Full A3B model TP=2↔1 end-to-end decode parity (the 3f snapshot gate).

**Effort:** M (given 3b+3e done). **Risk:** Medium — integration of two validated seams.

---

### Stage 4 (tail) — TP + MTP: `spec_step_mtp_tp`
*(prototype 49a35745, 95e4e958)*

**Files**
- MODIFY `crates/hipfire-arch-qwen35/src/mtp_spec.rs`: add `pub fn spec_step_mtp_tp` (proto mtp_spec.rs:1698), built on the current `spec_step_mtp` (1253).
- ADD examples `tp_mtp_demo.rs`, `tp_mtp_cost.rs`.

**API drift**
- Current `mtp_spec.rs` (122 KB) has its own drift since merge-base. `spec_step_mtp_tp` must wrap the **current** `spec_step_mtp` draft/verify loop, threading `forward_scratch_tp` / `forward_prefill_chunk_tp` for target verification.
- DFlash/MTP coherence gate applies (`scripts/coherence-gate-dflash.sh`).

**Validation:** **hiptrx 0+1:** `tp_mtp_demo` τ + token stream; TP=2 MTP committed-token stream ≡ TP=1 AR-greedy (byte-identical commits). Pass DFlash coherence gate (Tier 1/2/3).

**Effort:** L. **Risk:** High — spec-decode + TP interaction; coherence (attractor) risk per the DFlash gate rules.

---

## Dependency order

```
Stage 1 (tp_shard + init_tp)           [CPU, no blockers]
        │
Stage 2 (rccl.rs + all_reduce + smoke) [needs S1 field threading]
        │
Stage 3a (FaPhase seam + wo smoke)     [needs S2; THE crux]
        │
        ├── Stage 3  (forward_scratch_tp, replicated)   [needs 3a + run_dn extract]
        │       │
        │   Stage 3b (FA weight slicing + AWQ)          [needs 3]
        │       │
        │   Stage 3c (DeltaNet sharding)                [needs 3b]
        │       │
        │   Stage 3d (batched prefill TP)               [needs 3; parallel to 3c]
        │       │
        │   Stage 3e (EP-MoE)                           [needs 3b + 3d]
        │       │
        │   Stage 3f (MoE-attn shard + Q8F16)           [needs 3c + 3e]
        │
Stage 4  (spec_step_mtp_tp)            [needs 3d (+3f for MoE MTP)]
```

Stages 1–2 land together (rccl_comms field couples them). 3a gates everything.
3d (prefill) and 3c (DN) can proceed in parallel after 3. 3e needs 3b+3d.

---

## Highest-drift-risk shortlist (the 3–5 painful things)

1. **`run_fa_layer_body` re-seaming on the dispatch-migrated body (Stage 3a).**
   The current body routes through `hipfire_dispatch` (DispatchCtx, GemmFamily,
   families/pipeline) that the prototype has none of; the prototype body also
   carries a `PARO4G128T` branch current lacks, while current carries
   `fused_fa3_hfq6` proto lacks. The seam is a re-implementation. Breaking
   TP=1 byte-identity here silently corrupts the whole feature. **Guard:
   FaPhase::Full byte-identical via coherence-gate before touching TP.**

2. **Three independent dispatch-unification seams.** Same class as #1, repeated
   in `forward_prefill_chunk_tp` (Stage 3d, current prefill is dispatch+batched-GEMM)
   and `run_moe_ffn_ep` (Stage 3e, current MoE is `families::moe`). Each is a
   "rebuild on current, don't transplant from proto" hazard.

3. **`Stream::raw_ptr()` missing + `tp_use_rccl` config field missing.** Small
   but blocking — every TP collective call and `ensure_rccl` depends on them.
   Cheap fixes, but forgotten = compile wall on day one.

4. **RCCL runtime availability + single-process-multi-device init on hiptrx.**
   `librccl.so` soname/version coupling; `ncclCommInitAll` across 2 of 4
   gfx1201 devices. If RCCL is absent or peer-access is incomplete, Stage 2
   fails before any forward work. Validate `librccl.so.1` on hiptrx FIRST.

5. **DeltaNet replicated-state bit-determinism (Stage 3c) + AWQ-scale
   replicate-vs-slice asymmetry (Stage 3b).** Recurrent DN state must stay
   bit-identical across ranks (no all-reduce on it); a 1-ULP per-step drift
   compounds into divergence. AWQ scale slices on the **input** dim only
   (replicated for output-sharded `wq`, sliced for input-sharded `wo`/`w_down`)
   — getting this backwards passes Stage 1 tests but fails the gold gate.

---

## Port-vs-rebuild recommendation

**Split by drift class — this is the load-bearing decision given the later
super-op lowering:**

**PORT VERBATIM (zero structural investment lost; pure leaf code):**
- `tp_shard.rs` — pure CPU, no GPU/dispatch coupling. Survives any lowering.
- `rccl.rs` — self-contained FFI leaf. Survives.
- `Gpus::{init_tp, ensure_rccl, all_reduce_sum_f32}` + `rccl_comms` field — thin
  orchestration over stable `Gpu`/RCCL primitives. Survives.
- `ShardConfig` expert/head helpers, `local_attn_config`, `tp_allreduce_add{,_batched}` — small pure helpers. Survive.
- All `tp_*` examples + parity harnesses — these ARE the validation; port verbatim.

**REBUILD ON CURRENT (do NOT transplant the prototype body):**
- `run_fa_layer_body` FaPhase seam, `forward_prefill_chunk_tp`,
  `run_moe_ffn_ep`, `spec_step_mtp_tp`. These wrap forward bodies that the
  dispatch-unification refactor already rewrote, and that the later super-op
  lowering will rewrite **again**. Transplanting the prototype's pre-dispatch
  bodies would (a) not compile against `DispatchCtx`/families, and (b) bake in
  structure the lowering deletes.

**Pragmatic path for the rebuild half: minimal-seam, not pretty-seam.** Cut the
FaPhase / EP / batched-TP seams as **thin branches inside the current bodies**
(early-return phases + an all-reduce hook), keeping the per-rank loop in
`forward_scratch_tp` / `forward_prefill_chunk_tp` as standalone functions for
now. Do **not** refactor the current forward into a clean phase-pipeline to host
TP — that is exactly the super-op lowering's job. The standalone TP forward fns
are acceptable throwaway scaffolding: they get the parity gate green now, and
the later lowering subsumes them. Optimize for "smallest diff that passes
TP=2≡TP=1", accepting some duplication between `forward_scratch` and
`forward_scratch_tp`, rather than a premature unification that the capstone will
redo.

**Net:** ~40% of the prototype LOC ports verbatim (the leaf/orchestration
layer); ~60% (the four forward seams) is a guided rebuild against the current
dispatch-routed bodies, kept deliberately thin and standalone.
