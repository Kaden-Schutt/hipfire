# Multi-GPU Pipeline-Parallel (PP) Roadmap

**Status:** Stages 1–9 implemented on `feat/multi-gpu-pp`
**Author:** alpineQ (external contributor)
**Date:** 2026-05-05
**Target Release:** v0.2.0 (at maintainer discretion)
**Related:** issue [#58](https://github.com/Kaden-Schutt/hipfire/issues/58),
`docs/multi-gpu.md` (user-facing reference), `AGENTS.md` §threading,
`CLAUDE.md` "Coherence Gate"

---

## 1. Objective

Add Pipeline-Parallel (PP) inference over `n_devices` homogeneous AMD GPUs
to the Qwen3.5 forward path. Layers are sharded into contiguous bands;
the residual stream `s.x` flows through bands sequentially with a
single cross-device copy per band boundary. `output_norm + lm_head`
run on the last device (Variant 2 placement; matches Megatron / DeepSpeed
/ vLLM convention) so the final residual never copies back to dev_0.

**What v1 unlocks on 2× 24 GB:**

- Run 27B / 35B-A3B with extended context that doesn't fit on a single
  card (single-GPU 27B OOMs at `--max-ctx ≥ 16K` with `kv_mode=asym3`;
  see `AGENTS.md:356`).
- ~50-70 % of single-GPU throughput on already-fitting models. PP is
  sequential — it does NOT speed up models that already fit on one card.
  Multi-user serving needs TP, which is a separate roadmap.

**Reference hardware for the contributor branch:** 2× 7900 XTX
(gfx1100), Ryzen 9 9950X, NixOS, ROCm 6.4.3, peer access bidirectional
via PCIe Gen4 (validated in Stage 1 — see `findings/` and
`peer_smoke` example).

---

## 2. Why PP first

This roadmap intentionally ships PP before TP:

1. **Realistic first contribution.** PP is ~6 weeks median; TP from
   scratch is multi-month with deeper kernel-side risk.
2. **PP infrastructure is fundament for TP.** Multi-device FFI,
   `Vec<Gpu>`, `bind_thread` audit, `output_device` plumbing — all of
   it gets reused. TP layers on top.
3. **PP is the right shape for memory-bound deployments.** Users
   wanting 27B / 35B at extended context on consumer 2× 24 GB are PP-
   shaped, not TP-shaped.

---

## 3. Architectural decisions

### 3.1 Layer-band assignment (uniform split, escape hatch)

Default — `Gpus::init_uniform(n_devices, n_layers)` — distributes
layers contiguously with `base = n_layers / n_devices; rem = n_layers % n_devices;
per_device[i] = base + (i < rem ? 1 : 0)` (max-min ≤ 1). Pre-flight VRAM
check enforces homogeneous arch + bounded VRAM delta (default 2 GiB,
override via `HIPFIRE_UNIFORM_VRAM_TOLERANCE_GB`).

Asymmetric / mixed-VRAM systems use the explicit
`Gpus::init_layers(per_device: &[usize])` escape hatch.
`Gpus::init_vram_weighted(...)` is reserved for v1.1.

### 3.2 `output_norm + lm_head` on dev_last (Variant 2)

Convention adopted from Megatron-LM / DeepSpeed / vLLM. Removes the
final cross-device copy of `s.x` back to dev_0 after the layer loop,
and naturally balances the per-card weight footprint (dev_0 carries
`token_embd`, dev_last carries `lm_head` — sizes match for symmetric
vocabs).

### 3.3 Boundary-copy primitive

`Gpus::boundary_copy(src_dev, dst_dev, src, dst, n_bytes)` returns a
`BoundaryEvent` (post-copy event recorded on the dst-device stream).
Caller awaits via `Gpus::wait_boundary(evt)` before the next layer's
dispatch on dst_dev. Modes:

- **Peer mode** (default when peer access enabled): single
  `hipMemcpyPeerAsync` on src-stream + post-copy event.
- **Host-stage mode** (fallback): pinned host buffer + D2H async + event
  sync + H2D async on dst-stream + post-copy event.

Selection at `Gpus::enable_peer_all` time: bidirectional probe via
`hipDeviceCanAccessPeer`, bidirectional enable via
`hipDeviceEnablePeerAccess`. Any leg failing flips the orchestrator
into host-stage mode globally (no mixed peer/host within a forward).

### 3.4 Threading invariant

hipfire engine is single-threaded for HIP work; `Gpu::bind_thread()` is
a thread-local-cached `hipSetDevice` that runs at the entry of every
non-trivial `Gpu::*` method. Stage 2b audits all 278 `pub fn` in
`crates/rdna-compute/src/dispatch.rs` against a verify script
(`scripts/verify-bind-thread.sh`) wired into `.githooks/pre-commit`.

A module-level invariant doc in `crates/hipfire-runtime/src/multi_gpu.rs`
enumerates what is **not** supported (rayon/tokio worker threads
calling `Gpu::*`, HIP stream callbacks touching `Gpu`).

### 3.5 Replicated globals

Per-device replication (small, scratch-grade):

- `givens_cos_per_dev`, `givens_sin_per_dev` — asym{2,3,4} KV rotation
  tables (~2 KB × n_devices).

KV cache and DeltaNet state are **partitioned** by `layer_to_device`,
not replicated.

---

## 4. Scope (v1)

### 4.1 In scope

- 2-N device PP for the **Qwen3.5** decode + prefill paths
  (`crates/hipfire-arch-qwen35/src/qwen35.rs`)
- Homogeneous arch only (hard-fail on `gfx*` mismatch in `init_uniform`)
- Peer access + bidirectional enable + host-stage fallback
- Pre-flight VRAM check
- `HIPFIRE_DEVICES` env var (logical IDs **post**-`HIP_VISIBLE_DEVICES`
  filter — uphold ROCm convention; do not override visibility)
- 14 KvCache constructors updated (`new_gpu`, `new_gpu_q4`, `new_gpu_q8`,
  `new_gpu_q8_capped`, `new_gpu_int8`, `new_gpu_int8c`, `new_gpu_hfq4kv`,
  `new_gpu_hfq8`, `new_gpu_asym{2,3,4}`, `new_gpu_asym{2,3,4}_capped`)
- DFlash + `pp > 1` **refused at load time** (`"DFlash requires single-GPU
  in v1; see issue #58 v1.1 roadmap"`)
- CASK / TriAttention sidecar + `pp > 1` **refused at load time**
- Spec decode + `pp > 1` **refused at load time** (deferred to v1.1)
- `pp_parity` test: PP=1 ↔ PP=2 token sequence identical at temp=0
  ≥ 100 tokens
- `scripts/coherence-gate.sh --pp N` + new `scripts/pp-gate.sh` +
  `scripts/verify-bind-thread.sh`

### 4.2 Out of scope (v1)

- Mixed-arch (separate sub-feature)
- Llama-path multi-GPU (`crates/hipfire-runtime/src/llama.rs` decode loop)
- VL path multi-GPU
- DFlash + PP integration (v1.1+)
- Spec decode + PP (v1.1)
- Pipelined prefill (chunk N+1 on dev_0 while chunk N processes on dev_1) — v1.1
- VRAM-weighted automatic split — v1.1
- TP (tensor parallel) — separate roadmap
- Stream rebind for rocBLAS handle — pre-existing tech debt; out as
  separate PR after PP merge

---

## 5. Stages

The branch `feat/multi-gpu-pp` is the umbrella; each Stage is a logical
PR. Stage names below are the proposed PR branch / commit-prefix labels.

### Stage 1 — `multi-gpu-pp-1-probing-ffi` (~2d, **shipped**)

`crates/hip-bridge/src/{ffi.rs, lib.rs, error.rs}`,
`crates/hip-bridge/examples/peer_smoke.rs`.

- Add fn-pointer + `HipRuntime` methods: `hipGetDevice`,
  `hipDeviceCanAccessPeer`, `hipDeviceEnablePeerAccess` (silently maps
  `hipErrorPeerAccessAlreadyEnabled = 704` to `Ok`), `hipMemcpyPeer{,Async}`,
  `hipPointerGetAttributes`.
- New types: `MemoryType` enum (raw `u32` accessor for forward-compat),
  `HipPointerAttribute` `repr(C)` mirroring ROCm 6.4.3 layout (6 fields,
  including `allocationFlags`).
- New error constants: `HIP_ERROR_PEER_ACCESS_{UNSUPPORTED,
  ALREADY_ENABLED, NOT_ENABLED}`.
- `peer_smoke` example: alloc on dev 0/1, `memcpy_peer` byte-equality
  round-trip, `pointer_get_attributes` device verification,
  bidirectional `enable_peer_access` idempotency check.

**Validated on 2× 7900 XTX:** PASS, including 704 → Ok translation.

### Stage 2 — split into 2a + 2b (~6-8d total)

#### Stage 2a — `multi-gpu-pp-2a-device-binding` (~2d)

`crates/rdna-compute/src/dispatch.rs:211,399`.

- Add `pub device_id: i32` to `Gpu` struct.
- Refactor `Gpu::init` → `Gpu::init_with_device(id: i32)` (no-arg
  preserved as `Self::init_with_device(0)`); call `set_device(id)`
  **before** `try_init_rocblas`.
- Add `bind_thread()` helper: `thread_local! Cell<i32>` with sentinel
  `-1`, `debug_assert_eq!` invariant against `current_device()`.
- `impl Drop for Gpu` calls `bind_thread()` before resource free.
- New example: `crates/rdna-compute/examples/dual_gpu_smoke.rs`.

#### Stage 2b — `multi-gpu-pp-2b-bind-audit` (~4-6d)

`crates/rdna-compute/src/dispatch.rs:355+`.

- Audit all 278 `pub fn` in `impl Gpu`. Non-getter methods get
  `self.bind_thread()?;` as the first statement; pure getters carry
  an explicit `// bind_thread: skip — pure getter` whitelist comment.
- New: `scripts/verify-bind-thread.sh` (greps `impl Gpu` and validates
  invariant).
- Wire `verify-bind-thread.sh` into `.githooks/pre-commit`.

**Risk:** Medium-high. Silent mis-bind = malloc-on-wrong-device →
corrupted activations. Mitigations: `debug_assert_eq!` in
`bind_thread`, verify-script gate, single-GPU regression byte-equal.

### Stage 3 — `multi-gpu-pp-3-gpus-type` (~2d)

New: `crates/hipfire-runtime/src/multi_gpu.rs`.

- `Gpus { devices, layer_to_device, band_starts, peer_access_enabled,
  output_device, givens_cos_per_dev, givens_sin_per_dev }`.
- `Gpus::init_uniform`, `Gpus::init_layers`, `Gpus::single`,
  `Gpus::enable_peer_all`, `Gpus::device_for_layer`,
  `Gpus::is_band_boundary`, `Gpus::output_device`,
  `Gpus::boundary_copy`, `Gpus::wait_boundary`.
- Pre-flight VRAM check (arch match, bounded delta, headroom > required
  shard + scratch).
- `HIPFIRE_VRAM_PROBE=1` debug print.

### Stage 4 — `multi-gpu-pp-4-weight-load` (~3-4d)

`crates/hipfire-arch-qwen35/src/qwen35.rs`.

- Additive `load_weights_multi(hfq, config, gpus: &mut Gpus)`. Master's
  `load_weights` (single-GPU) stays untouched — the helpers
  (`load_token_embd_into`, `load_output_into`, `load_layer_into`) are
  extracted but the master body inlines them verbatim, so pp=1
  behavior is byte-equivalent.
- `token_embd` → `gpus.devices[0]`; `output_norm` + `output / lm_head` →
  `gpus.devices[gpus.output_device]`; per-layer weights →
  `gpus.devices[gpus.device_for_layer(i)]`.
- `Qwen35Weights::free_gpu_multi` mirrors per-device ownership.
- Update `docs/multi-gpu.md` memory budget table from real measurements.

### Stage 5 — `multi-gpu-pp-5-state-placement` (~4-5d)

`crates/hipfire-arch-qwen35/src/qwen35.rs`,
`crates/hipfire-runtime/src/llama.rs` (KvCache constructors),
`crates/hipfire-arch-qwen35/src/speculative.rs` (signatures).

- `Qwen35Scratch` → `Qwen35ScratchSet { per_device: Vec<Qwen35Scratch> }`.
  `s.logits` lives on `dev_last` (Variant 2).
- `DeltaNetState::new_with_quant_multi` accepts `gpus`; per-LA-layer state
  placed by `layer_to_device`. Returns `(state, la_to_device)` for
  daemon reset routing.
- All 14 `KvCache::new_gpu*_multi` constructors updated to per-layer
  device placement; asym{2,3,4} ctors use
  `gpus.givens_*_per_dev[layer_to_device[i]]` instead of a single
  global tensor.
- Single-GPU constructors unchanged (`new_gpu*` without `_multi` suffix
  retained for pp=1 path back-compat).

### Stage 6 — `multi-gpu-pp-6-forward-decode` (~5-6d)

`crates/hipfire-arch-qwen35/src/qwen35.rs` (`forward_scratch_multi`,
`forward_scratch_layers_multi`).

- Layer loop: `device_for_layer` lookup, boundary_copy + wait at band
  boundary, per-device scratch dispatch.
- Final `rmsnorm + lm_head` runs on `output_device`; no copy back.
- Disable hipGraph capture when `gpus.devices.len() > 1`.
- All 4 layer-type branches reproduced (DeltaNet, FullAttn,
  DeltaNetMoe, FullAttnMoe) — no MoE-pp deferral.

**Boundary-copy size in decode:** `dim × 4` bytes = 32 KB at dim=8192,
~6-10 µs/copy on PCIe peer. Forward-shape audit (`findings/phase0-forward-shape.md`)
confirms `s.x` is the only cross-band cross-layer buffer; per-layer
scratch (`s.tmp`, `s.x_rot`, `s.dn_*`, `s.fa_*`, `s.flash_partials`,
`s.gate_ffn`, `s.up`, `s.ffn_hidden`, `s.moe_x_rot`) lives entirely
within one layer's body and does not cross boundaries.

**Acceptance:**

- PP=1 numeric byte-identical to current state.
- PP=2 logits within 1e-4 of PP=1 on qwen3.5:0.8b.
- `pp_parity` test: PP=1 and PP=2 with identical seed/prompt/temp=0
  produce identical token sequence ≥ 100 tokens.

### Stage 7 — `multi-gpu-pp-7-daemon` (~1.5d)

`crates/hipfire-runtime/examples/daemon.rs`.

- `Gpu::init()` → `Gpus::init_uniform(...) + enable_peer_all()`.
- Read `pp` from load command JSON (default = `gpus.devices.len()`).
- `HIPFIRE_DEVICES` env (post-VISIBLE filter).
- Refuse `dflash + pp > 1`, `cask + pp > 1`, `spec_decode + pp > 1`.
- `LoadedModel.pp_*` extension fields (`pp_gpus`, `pp_scratch_set`,
  `la_to_device`, `pp_kv_cache`).
- `generate_multi(...)` ports the full master `generate` feature set:
  EosFilter, LoopGuard, PromptFrame, repeat_penalty, top_p sampler,
  attractor block, max_think_tokens.
- Sample/argmax reads `s.logits` from `gpus.devices[gpus.output_device]`.
- Reset handler routes DeltaNetState memset through `la_to_device`.

### Stage 8 — `multi-gpu-pp-8-prefill` (~2-3d)

`crates/hipfire-arch-qwen35/src/qwen35.rs` (`PrefillBandCtx`,
`forward_prefill_chunk` modification, `forward_prefill_batch_multi`).

- Same band/boundary shape as Stage 6, applied to the prefill loop.
- `forward_prefill_chunk` gains `band: Option<&PrefillBandCtx<'_>>`
  parameter — `None` preserves byte-exact pp=1 behavior. All
  master callers (`forward_prefill_batch`, `speculative.rs`,
  `pflash.rs`) pass `None`.
- Boundary-copy size: `batch_size × dim × 4` = 4 MB at batch=128
  dim=8192, ~70-80 µs through peer.
- Pipelined prefill (overlap chunk N+1 on dev_0 with chunk N on dev_1)
  is **out of scope** for v1; v1.1 PR.

### Stage 9 — `multi-gpu-pp-9-validation` (~1.5d)

`scripts/coherence-gate.sh`, `scripts/pp-gate.sh` (new),
`.githooks/pre-commit`,
`crates/hipfire-arch-qwen35/tests/pp_parity.rs` (new),
`crates/hipfire-runtime/examples/{pp_parity,pp_parity_chatml,pp2_vram_probe}.rs`
(new),
`docs/multi-gpu.md`,
`tests/speed-baselines/gfx1100x2_pp.txt` (new).

- `coherence-gate.sh --pp N`; for changes touching `multi_gpu.rs` or
  forward, run with both `--pp 1` and `--pp 2`.
- `pp-gate.sh` env-gates PP-specific tests on `HIPFIRE_HAVE_2_GPU=1`,
  exits 0 with skip message when only 1 GPU is visible.
- `.githooks/pre-commit` extends the HOTSPOT regex with
  `pipeline|stages|multi_gpu|pp_|peer_access|forward_prefill_batch_multi|forward_scratch_multi|Gpus|init_uniform|init_layers|boundary_copy`.

---

## 6. Validation matrix

| # | Test | Hardware | Command |
|---|------|----------|---------|
| 1 | Single-GPU regression | 1× any | `cargo test --workspace && ./scripts/coherence-gate.sh` |
| 2 | PP=1 byte-identical | 1× any | `./scripts/coherence-gate.sh --pp 1` |
| 3 | PP=2 numeric parity (0.8B) | 2× 7900 XTX | `./scripts/coherence-gate.sh --pp 2 --model qwen3.5:0.8b` |
| 4 | PP=2 token-stream parity (9B) | 2× 7900 XTX | `HIPFIRE_HAVE_2_GPU=1 cargo test pp_parity --release` |
| 5 | 27B + extended ctx unlock | 2× 7900 XTX | `HIPFIRE_DEVICES=0,1 hipfire run qwen3.5:27b --max-ctx 16384 "Hi"` |
| 6 | 35B-A3B smoke | 2× 7900 XTX | `HIPFIRE_DEVICES=0,1 hipfire run qwen3.5:35b-a3b "Hi"` |
| 7 | Perf baseline | 2× 7900 XTX | `HIPFIRE_DEVICES=0,1 hipfire bench qwen3.5:9b --runs 5` |
| 8 | DFlash+PP refusal | any | `hipfire run --pp 2 --dflash on …` → expect clear error |
| 9 | CASK+PP refusal | any | `hipfire run --pp 2 --cask on …` → expect clear error |
| 10 | Pre-flight VRAM check | 2× any | `HIPFIRE_UNIFORM_VRAM_TOLERANCE_GB=0.1 hipfire run --pp 2 …` → expect clear refusal |

---

## 7. Critical files (cheat sheet)

- `crates/hip-bridge/src/ffi.rs` — peer FFI (Stage 1, shipped)
- `crates/rdna-compute/src/dispatch.rs:211,355,399` — `Gpu` struct,
  init, 278 `pub fn` audit
- `crates/hipfire-runtime/src/multi_gpu.rs` — `Gpus` orchestrator
- `crates/hipfire-runtime/src/llama.rs` — 14 KvCache `*_multi` constructors
- `crates/hipfire-arch-qwen35/src/qwen35.rs` — weights, scratch,
  decode + prefill multi loops
- `crates/hipfire-runtime/examples/daemon.rs` — daemon integration

---

## 8. Open questions for maintainer

These are the four points raised in [issue #58](https://github.com/Kaden-Schutt/hipfire/issues/58)
and re-stated here for permanence:

1. **DFlash + PP scope.** v1 refuses the combination. Is that the right
   call long-term, or should DFlash + PP land together in a v1.1 patch?
2. **Mixed-arch policy.** v1 hard-fails `gfx*` mismatch. Soft-warn vs
   hard-fail — preference?
3. **API surface for `HIPFIRE_DEVICES`.** Current direction: logical IDs
   post-`HIP_VISIBLE_DEVICES`. Alternative: re-use `HIP_VISIBLE_DEVICES`
   semantics directly. Maintainer call.
4. **Test gates.** Repo has no CI; `pp-gate.sh` runs locally on
   `HIPFIRE_HAVE_2_GPU=1`. Sufficient, or should there be additional
   instrumentation?

---

## 9. References

- Issue [#58](https://github.com/Kaden-Schutt/hipfire/issues/58) —
  multi-GPU roadmap
- `docs/multi-gpu.md` — user-facing reference (memory budget table,
  deployment recipes, limitations)
- `docs/methodology/perf-benchmarking.md` — perf claim discipline
- `CLAUDE.md` "Coherence Gate" — mandatory gate for forward-pass changes
- `AGENTS.md` §threading — single-thread-for-HIP-work invariant
- Megatron-LM, DeepSpeed, vLLM PP literature for Variant 2 placement
  precedent
