# Consolidated Code Review: Multi-GPU PP + MTP

**Date:** 2026-06-01
**Branch:** `merge/master-pr352`
**Sources:**
- Gemini CLI review (`multi-gpu-pp-code-rev-gemini.md`)
- Claude Opus 4.8 review (`multi-gpu-pp-code-rev-claude.md`)
- This document validates/rejects each finding against the codebase.

---

## Finding 1: DeltaNetSnapshot cross-device copy  **[Gemini §1.1 + Claude §1]**

### Gemini's claim: CRITICAL — "will crash or produce incorrect results"

**REJECTED as critical/correctness bug.**

The snapshot and restore use `hipMemcpy(DeviceToDevice)`. With bidirectional
peer access enabled (`gpus.enable_peer_all()` at daemon.rs:2881), ROCm
routes cross-device copies over the peer link correctly. Empirical evidence
confirms: thousands of PP+MTP cycles with rollback (partial-accept
`restore_to`) produce byte-identical output, τ tracks single-GPU MTP across
all cross-device splits (48,16 / 52,12 / 56,08), and the coherence gate
passes cleanly.

### Claude's claim: Major portability/robustness hazard

**ACCEPTED.** The correctness depends on an undocumented invariant:
peer access must be enabled across all device edges holding DeltaNetState
buffers. If a future topology has a degraded peer edge
(`enable_peer_all` returns `Ok(false)` for that edge, which the code
tolerates at multi_gpu.rs:199-206), the same `hipMemcpy(D2D)` could
fail silently or fall back to host staging with different semantics
depending on ROCm version.

Additionally, `DeltaNetSnapshot::new_for` allocates all backup buffers on
one device, so on a partial-peer topology the failure is not loud — it's
a slow host-stage or wrong-device write.

**Action (low cost):** Either:
- (a) Add a `debug_assert!` in `save_from`/`restore_to` checking that
  source buffers are reachable from the snapshot device, and document the
  "requires `enable_peer_all` success" contract on the type; or
- (b) Make `DeltaNetSnapshot` device-aware: allocate each backup buffer on
  the same device as its source tensor (mirror `new_with_quant_multi`'s
  per-layer device loop). This also reclaims the perf lost to cross-peer
  copies on rollback.

(a) is sufficient to ship v1; (b) is the right fix if PpMTP graduates
past v1 or onto 3+ GPU boxes.

---

## Finding 2: Missing `bind_thread()` before 2b GEMM + snapshot blocks  **[Claude §2, unique]**

**ACCEPTED.**

Three blocks in `_multi` enter `output_device` without explicit
`bind_thread()`:

1. L3427: `state.trunk_snap.save_from(target_dn, target_gpu)?`
2. L3460: lm_head GEMM + argmax + GPU accept block
3. L3579: `state.trunk_snap.restore_to(target_dn, target_gpu)?`

The thread IS currently bound to `output_device` by accident of iteration
order: `forward_prefill_batch_multi_with_caps` ends with a cleanup loop
that calls `bind_thread()` for `b in 0..n_bands` ascending (qwen35.rs:11216),
leaving the thread bound to the last band = `output_device`. The
`gdn_tape_shards` assembly that could rebind afterward is a no-op in v1.

So it works today, but the correctness silently depends on:
- The cleanup loop iterating bands ascending
- The last band being output_device
- Tape assembly staying disabled

**Action:** Add `target_gpu.bind_thread()?;` as the first line of all three
blocks. One line each; removes fragile cross-function coupling. The
drafter block (§1 chain) already binds explicitly — the verify blocks
should match.

---

## Finding 3: BoundaryEvent HIP event handle leak  **[Gemini §2.1 + Claude §3]**

**ACCEPTED as low-priority.**

`multi_gpu.rs:46-56`: `Drop` detects an un-waited `BoundaryEvent` and
prints a warning but cannot free the event (no runtime handle stored).
Any `?` between `boundary_copy` and `wait_boundary` leaks one HIP event.

In practice the pair is always tight (copy then immediate wait), so the
leak is not hit on the happy path. Real under fault injection / OOM
mid-forward.

**Action (optional for v1):** Store the device id (or cloned runtime
handle) in `BoundaryEvent` so `Drop` can bind + `event_destroy`.

---

## Finding 4: GdnTapeShards::assemble_into synchronous peer copy  **[Gemini §2.3 + Claude §4]**

**REJECTED as actionable for v1.**

Both reviews flag `memcpy_peer` in `assemble_into` as synchronous on the
MTP critical path. But PpMTP v1 forces `tape_captured = false`
(mtp_spec.rs:3433) and the tape replay branch is `unreachable!()` at
line 3584. The `assemble_into` call is never reached in the shipping
multi MTP stepper. **Cold path — no action for v1.**

If tape replay is revived (the 5d-iii path), this should move to
`memcpy_peer_async` + single sync point.

---

## Finding 5: Llama multi-GPU support is KV-alloc-only  **[Gemini §2.2 + Claude §5]**

**REJECTED — the guard already exists.**

Both reviews claim there's no user-facing guard rejecting PP>1 on Llama.
This is wrong. `load_model_pp` (daemon.rs:2755) explicitly checks
`hfq.arch_id != 5 && hfq.arch_id != 6` and returns a clear error:

> "pp>1 supports Qwen3.5 dense (arch_id=5) and Qwen3.5-MoE /
> Qwen3.6-A3B (arch_id=6) only; got arch_id={}. LLaMA / Qwen3
> dense (arch_id<5) is pp=1 only."

No action needed.

---

## What is correct (cross-validated from Claude's §positive + Gemini's §3)

Both reviews independently confirm these are well-executed:

- **Variant-2 output layout:** `output_norm` + `lm_head` on output_device,
  `per_token_hidden_out` gated to `b == last_band`. Standard Megatron/vLLM
  optimization, implemented correctly.
- **Band-boundary copy sizing:** Decode copies `dim*4`; prefill copies
  `chunk_n*dim*4`. Copy-then-`wait_boundary` ordering correct on both paths.
- **Per-layer KV placement:** Each layer's K/V allocated on
  `device_for_layer(i)`; freed on the owning device.
- **Rotation-table replication:** Givens/FWHT cos/sin replicated to every
  device; forward reads from the correct per-device copy.
- **`split_pair_mut` aliasing discipline:** Clean split-borrow with
  strict-distinct assert replaces prior `unsafe` workaround.
- **`enable_peer_all` ordering:** Enable-after-alloc ordering obeyed;
  partial-topology failures tolerated and flagged.
- **Honest perf accounting:** PpMtp documented as below pp2-ar and
  single-gpu AR; value is long-ctx VRAM, not decode speed.
- **Greedy-only / compressed-sidecar v1 scope:** Asserted loudly at
  stepper entry rather than silently mishandled.

---

## Summary table

| # | Source | Severity | Finding | Disposition |
|---|--------|----------|---------|-------------|
| 1 | Gemini+Claude | Major (portability) | DeltaNetSnapshot cross-device copy depends on peer access | **ACCEPTED** — document contract or make device-aware |
| 2 | Claude | Minor (robustness) | Missing `bind_thread()` in 3 blocks of `_multi` | **ACCEPTED** — add 3 one-line binds |
| 3 | Gemini+Claude | Minor | BoundaryEvent leaks handle on early-exit | **ACCEPTED** — low priority, fix in cleanup pass |
| 4 | Gemini+Claude | Minor (cold path) | GdnTapeShards sync peer copy | **REJECTED for v1** — tape replay is disabled, path is cold |
| 5 | Gemini+Claude | Cosmetic | Llama PP guard missing | **REJECTED** — guard exists at daemon.rs:2755 |

**No correctness blockers.** The branch is empirically coherent on the
tested gfx906+gfx1031 pair. Recommend: land the 3 `bind_thread` calls
(Finding 2), document the peer-access contract on DeltaNetSnapshot
(Finding 1 option a), and address the rest in follow-up.

---

*Last updated: 2026-06-01*
