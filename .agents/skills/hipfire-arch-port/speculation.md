<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Kaden Schutt
hipfire — see LICENSE and NOTICE in the project root.
-->
# Adding speculative decode to a new arch

How to give a freshly-ported architecture speculative decoding. This is
**step 7 of the arch-port playbook** — do it only after the AR forward pass is
correct (steps 1–6 + the channel/coherence gates in `validation.md`). A wrong
forward pass with spec-decode bolted on top just produces faster garbage.

> TL;DR: implement **one trait** (`SpecTarget`) on your model bundle and add
> your `arch_id` to **two registries** (`build_speculator`'s gate +
> `Carrier::spec_target_guard`). That alone earns the model-free **n-gram**
> drafter — zero kernels, zero draft model, output **byte-identical to AR**.
> The daemon never learns your arch exists. A *learned* drafter (DFlash / MTP /
> EAGLE) is a separate, much larger effort layered on the same seam.

## The two-trait split (why this is bounded)

The decode loop in `examples/daemon.rs` drives a `&mut dyn Speculator` and
**never names an arch**. Adding a drafter is a bounded-context change because
the seam splits cleanly into *policy* and *mechanics*:

| Trait | Owns | Lives in | Who implements it |
|---|---|---|---|
| **`Speculator`** | *Policy*: what tokens to draft, the accept rule, the window | `hipfire-runtime/src/spec.rs:362` | the **drafter** (n-gram is shared; DFlash/MTP are per-family) |
| **`SpecTarget`** | *Mechanics*: how to run THIS arch's verify forward, snapshot/rewind its state, report EOS/capacity | impl'd on your **model bundle** in your arch crate | **you** (the arch porter) |

A model-free speculator (n-gram / PLD) drives **any** arch's target through
`SpecTarget` without knowing its internals: the target owns all verify
mechanics (the batched forward, the per-position lm_head/argmax, the recurrent
snapshot/rewind, the arch-specific scratch); the speculator owns only drafting
+ acceptance. That is why `impl SpecTarget` is the *whole* arch-side cost of
getting n-gram spec-decode.

```
daemon decode loop  ──drives──▶  &mut dyn Speculator        (policy: draft + accept)
                                      │ borrows per window
                                      ▼
                                 &mut dyn SpecTarget         (YOUR arch: verify mechanics)
```

## What you get for free vs. what you build

- **Free (just `impl SpecTarget` + 2 registry lines):** the model-free n-gram
  drafter `ChainSpeculator<NgramDrafter>` (`hipfire-runtime/src/spec_ngram.rs`).
  It proposes tokens from the prompt+output suffix and **always falls back to
  the target's own greedy argmax** on a verify miss — so the spec output is
  **byte-identical to AR by construction**. Only τ (accepted/window) and speed
  differ. Opt-in via `HIPFIRE_NGRAM_DRAFT=1`. Best on high-repetition workloads
  (verbatim copy, long-context retrieval, structured output); a loss at τ≪1 on
  free-form prose — see `docs/speculation-support-inventory.md` for measured τ
  per arch and when it actually wins.
- **Build (your own `Speculator` impl):** a *learned* drafter — DFlash (small
  same-family draft model), MTP (multi-token-prediction head), or EAGLE. These
  need trained weights and per-arch kernels. If your arch is an MTP family,
  implement the smaller `MtpDrafter` (`spec.rs:492`) and let `MtpSpeculator`
  adapt it to `Speculator` for you — you never write a whole `Speculator`.

For a brand-new arch, **start with n-gram**: it's the cheap correctness-preserving
win and validates your `SpecTarget` impl end-to-end.

## Step 1 — `impl SpecTarget for YourBundle`

Put it in `crates/hipfire-arch-<yours>/src/spec_impl.rs` (and `mod spec_impl;`
in `lib.rs`). The **canonical template is the qwen2 impl**
(`crates/hipfire-arch-qwen2/src/spec_impl.rs`) — a pure-attention arch with no
recurrent state, ~120 lines. Copy it and adapt.

The trait (`hipfire-runtime/src/spec.rs:155`), method by method:

| Method | What to do | Pure-attention (qwen2/llama/minimax) | Recurrent (qwen35 DeltaNet, lfm2moe conv-state) |
|---|---|---|---|
| `as_any_mut` | `self` | trivial | trivial |
| `reset_recurrent` | zero recurrent state + rewind KV cursor | `self.state.reset()` (cursor rewind only) | zero S/conv state **and** KV offset |
| `new_spec_scratch` | allocate verify scratch sized to `block_size` | return a ZST scratch (nothing to carry) | allocate the recurrent **snapshot** buffers (S/conv + any Q8 error-feedback residual) |
| `spec_advance` | advance over `tokens` from `start_pos`, return argmax at the LAST position; `reset=true` ⇒ cache-miss prefill | loop `forward_step`, then `argmax_f32` | same, plus reset recurrent state when `reset` |
| `verify_block` | run `block` at `position`, return argmax at **each** of `block.len()` slots; leaves state advanced by `block.len()` | one batched layer loop (e.g. `forward_verify_block_batched`) | **first snapshot** recurrent state into `scratch`, *then* the batched forward |
| `commit_prefix` | fix state to exactly the committed prefix after verify over-advanced | **no-op** (accepted-prefix KV already correct; rejected tail overwritten next verify) | restore the `scratch` snapshot and **replay** `block[..accept_len+1]` with the *same* batched forward (numerics must match the accepted argmax) |
| `eos_token` | `self.config.eos_token_id` | trivial | trivial |
| `ctx_capacity` | usable context length | `self.state.max_seq` | same |
| `kv_cache_mut` | only override if your KV is the shared `llama::KvCache` | leave `None` default (own KV repr ⇒ no FlashCASK eviction) | override only for qwen35/llama-style shared KV |

### The three contracts that bite

1. **`verify_block` snapshots BEFORE it advances.** `commit_prefix` rewinds on a
   partial accept, so any recurrent state it needs to restore must be captured
   *into `scratch`* at the top of `verify_block`, before the forward mutates it.
   Stateless arches snapshot nothing — that's why pure-attention `commit_prefix`
   is a no-op. Get this wrong on a recurrent arch and partial-accept windows
   silently bleed drifted state → attractor loops (the #462 class).
2. **`verify_block` returns the verifier's pick at EVERY slot**, not just the
   last: `argmax[i]` is the target's next-token prediction after consuming
   `block[0..=i]`. The shared accept rule (`accept_greedy_prefix`, `spec.rs:112`)
   needs `block.len()+1` picks (one extra for the bonus at full acceptance).
3. **Position math is driven by `emit.len()`**, not by `accepted`. The daemon
   does `position += step.emit.len()` and reseeds from `next_seed`, blind to
   which drafter ran. Your `verify_block` must leave KV/recurrent state
   consistent with that advance.

### Byte-identical property (your correctness oracle)

Because the n-gram verify falls back to the **same** greedy argmax your AR path
uses, the n-gram output **must be byte-identical to greedy AR**. That is the
validation test: run the same prompt with and without `HIPFIRE_NGRAM_DRAFT=1`
and diff the concatenated token text. Any divergence is a bug in your
`SpecTarget` impl (usually a `verify_block`/`commit_prefix` state mismatch), not
a sampling difference. (Drivers for the wired arches live in
`/home/bjoern/hipfire-ngram-validate/`; the qwen2 in-crate parity check is
`crates/hipfire-arch-qwen2/examples/verify_block_parity.rs`.)

## Step 2 — register the arch in `build_speculator`

`crates/hipfire-loader/src/spec_build.rs`. This is the single load-time registry
the daemon routes through; add your `arch_id` to the n-gram gate:

```rust
// spec_build.rs — the model-free n-gram arm
if ngram_enabled && matches!(arch_id, 0 | 1 | 5 | 6 | 7 | 8 | 10 | 11 | 12 /* + yours */) {
```

That's all that's needed for the n-gram arm — it's arch-typeless and builds its
verify scratch lazily via `SpecTarget::new_spec_scratch` on first `prefill`.

## Step 3 — give the daemon a `SpecTarget` borrow via the carrier

`crates/hipfire-loader/src/carriers.rs`. The daemon borrows the target through
`Carrier::spec_target_guard`. For a pure-attention arch whose bundle **is** a
`SpecTarget`, use the generic in-place guard — one line:

```rust
fn spec_target_guard<'m>(&self, state: &'m mut Option<ModelState>, _model_path: &str)
    -> Result<Box<dyn SpecTargetGuard + 'm>, String> {
    match state.as_mut() {
        Some(ModelState::YourArch(bundle)) => Ok(Box::new(InPlaceGuard { bundle })),
        _ => Err("not a loaded <yourarch> bundle".into()),
    }
}
```

Only a recurrent arch that must *move its bundle out and reopen an mmap on every
window* needs a bespoke guard (see `Qwen35SlotGuard` in `spec_build.rs` — it
rebuilds the bundle on every exit path to structurally kill the #462
cross-request state-bleed class). Most new arches use `InPlaceGuard`.

## The daemon arm (usually already generic)

The generic spec loop (`generate_spec` / `generate_dflash` in `daemon.rs`)
already drives `&mut dyn Speculator` for any arch whose carrier yields a
`SpecTargetGuard` — so for a standard text arch there is **no daemon edit**.

**Exception — bespoke decode paths.** If your arch decodes through its own
non-generic loop (e.g. an image-conditioned VL path), route only the *decode*
phase: keep the bespoke prefill, then if `speculator.is_some()` hand off to a
small spec loop. dots-ocr (arch 8) is the worked example —
`decode_vl_dots_ocr_ngram` / `run_dots_ocr_ngram_loop` in `daemon.rs`: it moves
the flat model fields into a bundle, primes the drafter + first token *without
re-running vision prefill* (`prefill(cache_hit=true, empty suffix)` just argmaxes
the live post-prefill logits), then runs the standard `prefill`→`step` contract.

## Cosmetic: the emitter (`SpecEmit`)

`Carrier::make_spec_emitter` controls how committed tokens render to the wire
(ChatML framing, `<think>` handling, special-marker state machines). Most arches
reuse `Qwen35Emit` (ChatML-clean). Only override it if your arch has a bespoke
output protocol — e.g. cohere2moe's `Cohere2MoeEmit` ports North's marker state
machine. This does **not** affect generated tokens, only their rendering.

## Validation checklist (before you claim it works)

1. **Byte-identical to AR** — diff `HIPFIRE_NGRAM_DRAFT=1` vs unset on the same
   greedy prompt. MUST match. This is the hard gate for n-gram.
2. **Detectors clean** — no attractor / special-token leak (run the relevant
   coherence gate; spec-decode changes also trigger
   `scripts/coherence-gate-dflash.sh`). A recurrent-state `commit_prefix` bug
   shows up here even when short prompts pass byte-parity — also run
   `scripts/serve-multiturn-gate.sh` (catches cross-request state bleed).
3. **Measure τ before claiming a speedup.** τ = accepted/window. n-gram only
   beats AR at τ≳3; on a BW-bound MoE a batched verify can pay off at lower τ,
   on a small compute-bound decoder it cannot. Record τ + the prompt md5 (per
   the perf-benchmarking rules in `CLAUDE.md`). See
   `docs/speculation-support-inventory.md` for the measured τ table and the
   "batched verify falsified except for BW-bound MoE" analysis — read it before
   you spend GPU-days on a verify kernel.

## Reference map

| Thing | Where |
|---|---|
| `Speculator` / `SpecTarget` / `SpecStep` / accept rule | `crates/hipfire-runtime/src/spec.rs` |
| Model-free n-gram drafter | `crates/hipfire-runtime/src/spec_ngram.rs` |
| `build_speculator` registry | `crates/hipfire-loader/src/spec_build.rs` |
| Carrier guard wiring | `crates/hipfire-loader/src/carriers.rs` |
| Canonical `SpecTarget` template (pure attention) | `crates/hipfire-arch-qwen2/src/spec_impl.rs` |
| Recurrent-state template (snapshot/rewind) | `crates/hipfire-arch-qwen35/src/spec_impl.rs`, `crates/hipfire-arch-lfm2moe/src/spec_impl.rs` |
| VL bespoke-decode routing example | `decode_vl_dots_ocr_ngram` in `crates/hipfire-runtime/examples/daemon.rs` |
| MTP drafter core (learned, multi-token) | `MtpDrafter` (`spec.rs:492`) / `MtpSpeculator` (`spec.rs:543`) |
| Per-arch support status + measured τ | `docs/speculation-support-inventory.md` |
