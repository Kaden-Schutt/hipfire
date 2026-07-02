# Abliteration → rank-1 LoRA + a daemon adapter stack

Status: PROPOSED. Materialize the steer/abliteration output (`{v_L}` directions)
as a reusable rank-1 LoRA artifact, and generalize the in-forward steer session
into an adapter **stack** with per-adapter **scale** (intensity) so adapters can
be loaded, stacked, and dialed at runtime.
Date: 2026-06-30
Builds on `docs/plans/2026-06-30-steer-daemon-pivot.md` (the daemon steer ops and
`DaemonHarness` it landed).

## Why

The abliteration artifact is the per-block unit directions `{v_L}` from
`derive_directions`. The apply is linear, so it factors into a low-rank weight
delta — i.e. it can live as a LoRA instead of (only) a live steer session. A LoRA
form is portable (any stack can serve it), stackable (compose several adapters),
and dial-able (a per-adapter scale = the steer `strength`, adjustable without
rebuilding). Base weights stay quantized; the delta is a bf16/fp16 low-rank add —
so still **no re-quantization**, the same benefit the runtime hook gives.

## The factorization (two conventions)

For block `L`, unit direction `v` (`vᵀv = 1`), strength `λ`, the residual
transform is `x' = (I − λ v vᵀ) x`.

1. **Residual-adapter form** — exact to the current block-boundary hook,
   directions-only:
   - `target = Residual@L`, `A = vᵀ` (`[1,hidden]`), `B = −v` (`[hidden,1]`),
     `scale = λ`. Apply: `Δx = scale · B (A x) = −λ (vᵀx) v` — bit-identical to
     `apply_direction(ablate)` / `apply_on_gpu`. No base-weight dependence.
2. **Fused-projection form** — portable (PEFT-style), approximate:
   - Fold into each residual-writing projection (attn `o_proj`, MLP `down_proj`):
     `ΔW = −λ v (vᵀW)`, `A = vᵀW` (`[1,d_in]`, needs the base weight), `B = −v`,
     `scale = λ`. Only *approximates* the boundary hook (RMSNorm sits between the
     residual and the next block; the boundary form also lumps in the
     embedding/carried-forward component).

Default to **(1)** for hipfire (exact, trivial). **(2)** is an explicit
`--portable` export for other stacks.

Note: **steer** mode (`x += s·v`) is an additive *bias*, not a `B(Ax)` delta, so
it is out of scope for the rank-1 form — it needs a bias adapter (future). Only
**ablate** is the clean rank-1 case.

## Artifact

`.lora.hfq` sidecar, dotted naming, e.g. `MedGemma-4B-it.abliterate.lora.hfq`:

```
LoraMeta { base_model_sha256, arch_id, rank, default_scale, mode,
           targets, derived_from (good/bad set hashes) }
deltas:  [ { layer, target, rank, A:[r,d_in] bf16, B:[d_out,r] bf16 } ]
```

bf16/fp16 deltas over the untouched quantized base — no re-quant. Base-specific
(form 2 carries `vᵀW`), so loads compat-gate `base_model_sha256`.

## Daemon adapter stack (stacking + intensity)

The apply primitive already exists: `apply_on_gpu` = `gemv_f32` (`A x`) +
`scaled_add_inplace_gpu_scalar_f32` (`+= scale·B·…`). A rank-`r` delta is
`y += scale · B(A x)` = two gemvs; rank-1 = dot + axpy. So this generalizes the
steer session — not new kernel work.

```
LoraStack { adapters: Vec<ResidentAdapter> }          // process-global, like SESSION
ResidentAdapter { id, scale: f32, deltas: Vec<ResidentDelta> }
ResidentDelta   { layer, target, a: GpuTensor, b: GpuTensor, rank }
TargetProj      { Q,K,V,OProj,Gate,Up,DownProj, Residual }
```

- Gate with `ACTIVE: AtomicBool` + `EPOCH` exactly like the steer session
  (per-thread upload cache keyed on epoch; `GpuTensor` is `!Sync`).
- **Hook points:** residual targets reuse `maybe_steer_block`; projection targets
  add `maybe_lora(gpu, &y, layer, target)` after each `weight_gemv`.
- **Stacking** = additive sum at each target: `y += Σ_k scale_k · B_k(A_k x)`,
  order-independent. *Caveat:* projective ablations don't compose linearly
  (`P₁P₂ ≠ I − λ(v₁v₁ᵀ+v₂v₂ᵀ)` unless `v₁⊥v₂`); as summed deltas they're
  first-order. To stack abliteration directions exactly, orthogonalize the set at
  export; warn on `lora_load` when stacking two ablate/Residual adapters.
- **Intensity** = per-adapter `scale` (the steer `strength`, now per-adapter and
  live). `0`=off, `1`=nominal, `>1` amplify, `<0` invert. `lora_set_scale` writes
  one scalar + bumps EPOCH (no A/B re-upload).

### Protocol ops (mirror the steer/collect pattern)

```
lora_load      { path | inline, id, scale } → upload A/B, push to stack → ok
lora_set_scale { id, scale }                → live intensity            → ok
lora_unload    { id } / lora_clear          → drop                      → ok
lora_list                                   → [{id,scale,n_deltas,targets}]
lora_export    { directions|from_capture, mode, strength, layer_range,
                 targets, output, portable } → { n_deltas, rank, output }
```

`clear()` on model load/unload (adapters can't leak across models). `lora_load`
compat-gates `base_model_sha256` vs the resident model.

## Build order

1. **Host LoRA core** (`hipfire-steer::lora`, no GPU): `LoraTarget`/`LoraDelta`/
   `LoraAdapter` types, `abliteration_adapter` (directions+strength → rank-1
   residual deltas, ablate-only), `apply_residual_stack` (host apply with
   per-adapter scale), serde round-trip. Unit tests: equivalence to
   `apply_direction(ablate)`, orthogonal stacking == sequential, scale=0 no-op,
   serialize round-trip. ← THIS INCREMENT.
2. **GPU stack apply + session generalization**: turn the steer `Session::Applying`
   into a resident `LoraStack`; the existing `begin_apply(SteerSpec)` becomes
   "load one ablate adapter". Reuse `apply_on_gpu`; add `maybe_lora` at projection
   sites for non-residual targets. Validate coherence on halo/nix2.
3. **Daemon ops + `.lora.hfq` container**: the 5 protocol ops + the hfq sidecar
   read/write + `DaemonEngine` clients. `lora_export` is daemon-resident (form 2
   needs `vᵀW` from the dequantized resident weights).
4. **Portable fused-projection export** (`o_proj`/`down_proj`) + optional
   `hipfire-quantize --merge-lora` (bake + re-quant for zero runtime cost when the
   config is frozen).

## Reused unchanged

`derive_directions`, the block-boundary hook, `apply_on_gpu` (= the rank-1 apply
primitive), the steer-session state-machine pattern (ACTIVE/EPOCH + per-thread
cache), the daemon op pattern (protocol struct → string-keyed arm → DaemonEngine
client), hfq sidecar naming.
