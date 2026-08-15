<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Kaden Schutt
hipfire — see LICENSE and NOTICE in the project root.
-->
# Speculative decode for a new model arch

**Step 7 of the arch-port playbook** — only after the AR forward pass is
correct under the routes in `validation.md` / `docs/VALIDATION.md`. Wrong AR
plus speculation is faster garbage.

This file is the **how-to**. Mutable per-arch status snapshots live in
[`docs/speculation-support-inventory.md`](../../../docs/speculation-support-inventory.md)
(**historical** inventory — re-check source before product claims). Arch ids:
[`docs/architecture-ids.md`](../../../docs/architecture-ids.md).

> **TL;DR:** implement `SpecTarget` on your bundle, register the `arch_id` in
> `build_speculator`’s n-gram gate, and return a `Carrier::spec_target_guard`.
> That earns the model-free **n-gram** drafter (`HIPFIRE_NGRAM_DRAFT=1`).
> Under **greedy** verify (argmax fallback on miss), committed tokens match the
> greedy AR path for a correct `SpecTarget`. That is **not** a blanket claim for
> temp>0, sampled verify, or emitter/rendering differences.

## Two-trait split

The daemon decode loop drives `&mut dyn Speculator` and does not hardcode most
text arches. Policy vs mechanics:

| Trait | Owns | Defined in | Implementer |
|---|---|---|---|
| **`Speculator`** | Draft policy, accept rule, window | `crates/hipfire-runtime/src/spec.rs` | Drafter (shared n-gram; DFlash/MTP per family) |
| **`SpecTarget`** | Verify forward, snapshot/rewind, EOS/capacity, optional hidden hooks | same | **Your arch bundle** |

```
daemon  ──▶  &mut dyn Speculator     (draft + accept)
                  │
                  ▼
             &mut dyn SpecTarget     (your arch verify mechanics)
```

## What you get vs build

| Path | Cost | Notes |
|---|---|---|
| **n-gram** `ChainSpeculator<NgramDrafter>` | `SpecTarget` + registries | `crates/hipfire-runtime/src/spec_ngram.rs`. Opt-in: env `HIPFIRE_NGRAM_DRAFT=1` (or CLI/load `SpecLoadCfg` when env unset). Defaults: `HIPFIRE_NGRAM_DRAFT_K` / `ngram_k` → **12** (min 2), `HIPFIRE_NGRAM_MIN_COUNT` → **2**. |
| **Learned** DFlash / MTP / EAGLE | Weights + kernels + often a full `Speculator` | DFlash generic chain: `dflash_generic.rs`. qwen35 DFlash/MTP and deepseek4 MTP are family-specific. MTP helper traits: `MtpDrafter` / `MtpSpeculator` in `spec.rs`. |

Start with n-gram on a new arch.

## Step 1 — `impl SpecTarget`

Put `crates/hipfire-arch-<yours>/src/spec_impl.rs` and `mod spec_impl;` in
`lib.rs`.

**Templates (read source, do not copy line numbers as API):**

| Kind | Template crate |
|---|---|
| Pure attention | `hipfire-arch-qwen2` (`spec_impl.rs`) |
| Recurrent (DeltaNet / conv) | `hipfire-arch-qwen35`, `hipfire-arch-lfm2moe` |
| VL decode-phase only | `hipfire-arch-dots-ocr` + daemon VL n-gram loop |
| MTP target, **no** n-gram allowlist | `hipfire-arch-deepseek4` (DSpark `SpecTarget` implemented; arch 9 excluded from n-gram gate — MTP/DSpark path) |

### Required / defaulted methods (`spec.rs` — `trait SpecTarget`)

| Method | Role |
|---|---|
| `as_any_mut` | Downcast |
| `reset_recurrent` | Zero recurrent + KV eviction offset as applicable |
| `new_spec_scratch` | Verify scratch sized to `block_size` |
| `spec_advance` | Advance over tokens from `start_pos`; last-position greedy (or `SpecAdvance`); optional hidden capture |
| `verify_block` | Per-position **argmax** over `block`; **snapshot into `scratch` before forward** |
| `commit_prefix` | Fix state to accepted prefix after over-advance |
| `eos_token` / `ctx_capacity` | Loop bounds |
| `kv_cache_mut` | Default `None`. Override only for shared `llama::KvCache` (FlashCASK) |
| `verify_block_sampled` | Default `Err`. Only arches that implement it may take temp>0 n-gram |
| `verify_block_logits` / tree / DFlash hooks | Default `Err`/`None`; required for DFlash/DDTree/DSpark-style paths |

### Contracts that bite

1. **`verify_block` snapshots before advance.** Partial accept restores via
   `commit_prefix`. Recurrent arches (DeltaNet S, LFM conv rings, Q8 error
   feedback) must capture into `scratch` first. Pure attention usually no-ops
   `commit_prefix`.
2. **Per-slot picks:** `argmax[i]` after consuming `block[0..=i]`. Accept rule
   (`accept_greedy_prefix`) needs the full window including bonus on full accept.
3. **Position follows `emit.len()`**, not a private accept counter — leave
   KV/recurrent consistent with the daemon’s advance.
4. **Do not double-count `n_tokens`** if `decode_step` already sets
   `position + 1` (cohere2moe latent bug class).

### Deterministic parity (exact scope)

For **greedy** n-gram (drafter miss → target argmax; `samples=false` or temp=0
path), a correct `SpecTarget` yields **token text identical to greedy AR** on
the same prompt, weights, and sampler seed policy.

| In scope | Out of scope |
|---|---|
| Greedy AR vs `HIPFIRE_NGRAM_DRAFT=1` greedy verify | temp>0 unless `verify_block_sampled` is implemented and the n-gram arm was built with `samples=true` |
| Token ids / decoded text under the same emitter rules | Cosmetic `SpecEmit` differences (e.g. whether a think delimiter is consumed) |
| In-crate parity helpers when present (e.g. `crates/hipfire-arch-qwen2/examples/verify_block_parity.rs`) | Retired batteries are not current evidence — see `docs/VALIDATION.md` |

`build_speculator` sets `samples` from a live `arch_id` match (today:
qwen35 only) — read `crates/hipfire-loader/src/spec_build.rs`. Other n-gram
arches stay **greedy-only** (`requires_greedy`); temp>0 routes to AR unless
that arm and `verify_block_sampled` say otherwise.

## Step 2 — n-gram gate in `build_speculator`

File: `crates/hipfire-loader/src/spec_build.rs`.

Cascade (confirm in source): DFlash state → qwen35 MTP head → n-gram if
enabled and `arch_id` matches the current gate arm.

**Do not copy the allowlist into this skill.** Before adding or claiming an
id:

1. Open the `ngram_enabled && matches!(arch_id, …)` arm in current
   `crates/hipfire-loader/src/spec_build.rs` and read the live match set.
2. Resolve each selected id through
   [`docs/architecture-ids.md`](../../../docs/architecture-ids.md) (family /
   crate owner).
3. Confirm whether the arch intentionally uses another drafter path instead
   of n-gram (e.g. MTP/DSpark-only targets, toy `0xFF`) — inventory snapshot:
   [`docs/speculation-support-inventory.md`](../../../docs/speculation-support-inventory.md)
   (**historical**; verify in source).

Add your id only after `SpecTarget` exists and the carrier can borrow it.

## Step 3 — `Carrier::spec_target_guard`

File: `crates/hipfire-loader/src/carriers.rs` (+ `Carrier` trait in loader
`lib.rs`).

- **In-place:** `InPlaceGuard { bundle }` when the bundle **is** the
  `SpecTarget` (qwen2, llama, minimax, lfm2moe, cohere2moe, deepseek4, …).
- **Move-out + reopen:** qwen35 `Qwen35SlotGuard` in `spec_build.rs` (lazy
  `HfqFile`, restore on `Drop` — cross-request bleed class #462).

Call `build_speculator(arch_id, …, SpecLoadCfg, …)` from `load` so
`LoadedModel.speculator` is populated when enabled.

## Daemon routing

Generic `generate_spec` / `generate_dflash` already drive `&mut dyn Speculator`
when the carrier yields a guard — **most text arches need no new daemon arm**
beyond ensuring the load-time speculator is set and any **bespoke**
`generate_<arch>` short-circuit does not win first.

**Exception — bespoke decode:** dots-ocr (8) keeps vision prefill, then
decode-phase n-gram (`decode_vl_dots_ocr_ngram` / `run_dots_ocr_ngram_loop` in
`daemon.rs`): move flat fields into a bundle, `prefill(cache_hit=true, empty
suffix)` to seed without re-running vision, plain UTF-8 stream (no `SpecEmit`).

## Emitter (`SpecEmit`)

`Carrier::make_spec_emitter` affects **wire rendering** only (ChatML, think
markers). Default ChatML-family: shared `Qwen35Emit`. Cohere2 uses
`Cohere2MoeEmit`. OCR: no emitter. Rendering deltas are not generation
divergences.

## Validation checklist (speculation)

Route through `docs/VALIDATION.md`. Retired batteries are not current evidence.

1. **Greedy token parity** — same prompt, AR vs `HIPFIRE_NGRAM_DRAFT=1`, greedy;
   diff token text. Divergence ⇒ `SpecTarget` / state bug.
2. **Partial-accept / multi-turn** — recurrent arches: long windows + multi-request
   serve; prefer path-specific oracles and `serve_harness.py` / LFM harness for
   semantics.
3. **τ before speed claims** — τ = accepted/window. Protocol:
   `docs/methodology/perf-benchmarking.md`. Historical τ notes in
   `speculation-support-inventory.md` (verify before citing). Batched verify is
   not automatically a win (compute-bound small models vs BW-bound MoE).
4. **Learned drafter** — family-specific harnesses and `REDLINE.md` only when the
   claim is Redline-related; MTP/DFlash have their own load gates
   (`HIPFIRE_QWEN35_MTP`, draft path / arch_id=20 HFQ, deepseek4 MTP auto at
   temp=0 when weights present, etc.). Read loader source for the arm you claim.

## Reference map

| Thing | Path |
|---|---|
| `Speculator` / `SpecTarget` / accept | `crates/hipfire-runtime/src/spec.rs` |
| n-gram | `crates/hipfire-runtime/src/spec_ngram.rs` |
| Generic DFlash | `crates/hipfire-runtime/src/dflash_generic.rs` |
| `build_speculator` / `Qwen35SlotGuard` | `crates/hipfire-loader/src/spec_build.rs` |
| Carriers / guards | `crates/hipfire-loader/src/carriers.rs` |
| Load knobs | `hipfire_runtime::loader_api::SpecLoadCfg` |
| Arch ids | `docs/architecture-ids.md` |
| Status inventory (historical) | `docs/speculation-support-inventory.md` |
| Validation routes | `docs/VALIDATION.md` |
