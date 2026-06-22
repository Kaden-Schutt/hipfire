# Reference kernel layer — boring, complete, portable

Status: **decision + active** — 2026-06-22. Owner: chaingun (with a parallel
session building the bf16/fp32 batched GEMM — see Coordination). Builds on
[quantization-targets](2026-06-22-quantization-targets.md).

## The idea

A **reference kernel layer**: for every forward op × {single-token decode, batched
prefill} × the committed precision targets, one boring, portable, parity-gated HIP
kernel that is *always correct and always present*. It is three things at once:

- the **correctness floor**,
- the **dispatch fallback** (so there is never a missing/silent-bypass path), and
- the **parity oracle** every optimized variant is checked against.

**Optimized / fused / arch-specific kernels become opt-in overlays** the dispatcher
selects only when present, arch-matched, and parity-clean against the reference —
never the only path. The reference must NOT replace a working fast path (perf
regression); it is the floor and the oracle.

### Why now / why this fixes a class of bugs

The format sprawl is now a small enumerable matrix (quant-targets decision), so
"complete coverage" is finally tractable. And the bugs that dominated the
hierarchical-KV + quant work — kvarn garbage in the daemon, W8A8/W4A8 unwired in the
forward, oq4 batched gated off, paths silently bypassing the per-token dispatch — are
all the *same disease*: **incomplete coverage with no guaranteed-correct fallback**.
A complete reference layer + a dispatcher that always has a correct floor structurally
removes that class. The discipline already exists in parts (`gemm_iu8_i32_wmma` is
literally "generic … unperfed"; the `*_serial_reference` forward paths) — this names
it, completes the matrix, and makes the dispatcher honor it.

## Warn-on-generic-fallback

When the dispatcher runs a reference kernel because no optimized overlay matched, it
emits a **rate-limited warning, once per `(op, precision, mode, arch)` tuple**:

```
hipfire: generic fallback — gate_up W4A8 batched on gfx1103 (no optimized overlay)
```

This makes the reference layer a **runtime coverage map**: logs show exactly where
optimization is missing, and eval/CI can assert "no generic fallback on the hot path
for config X". `HIPFIRE_WARN_GENERIC=0` silences; default on (warn-once). For
completeness-only tiers (see W4A4) the warning also flags known-low-quality selection.

## The matrix

**Ops** (qwen3.5 forward): rmsnorm, RoPE, dense GEMMs {qkv, o, gate/up, down}, MoE
grouped GEMM, attention {flash decode, batched prefill}, gated-delta-net / LA path,
lm_head, embed.

**Modes**: decode (n=1, GEMV-shaped) and prefill (batched, GEMM-shaped) — distinct
kernels.

**Precision** (GEMMs): bf16/fp32 (oracle), f16 (A16), iu8 (A8), iu4 (A4).
Attention: the KV tiers (f32 / fp16 / q8 / KVarN / cold).

### GEMM precision cells — status / ownership

| Cell | Role | Status |
|---|---|---|
| **W(bf16)A(fp32) batched** | prefill **correctness oracle** | **in progress — parallel session, do not touch** |
| W4A16 / W8A16 (f16 WMMA) | quality default | exists (mq4/q8 families) |
| W8A8 (iu8) | integer baseline | core exists (`gemm_iu8_i32_wmma`, unperfed); needs scale-dequant + wiring |
| W4A8 (iu8 + nibble-expand) | production target | numerics parity done (`parity_w4a8`); per-group + forward wiring remain |
| W4A4 (iu4) | **completeness-only, known-low-quality** | oq4 exists; keep in stack, numerics-gated NOT quality-gated |

## W4A4 policy: in the stack, flagged

W4A4 (oq4) stays as a first-class member of the reference matrix **for completeness**,
even though it is not expected to produce good output (activation-precision cliff —
established: W4A4 is fragile because of the int4 activations, not the weights).
Coverage ≠ endorsement. It is **numerics-parity-gated** (the GEMM is mathematically
exact) but **not quality-gated**; the registry marks it `completeness-only` and the
warn-on-generic path notes the known-low-quality selection. Having it in the stack
means the dispatcher can always express it and we can measure it; it is never the
default and never claimed as production.

## Principles

1. **Portable shared intrinsics only** — the WMMA variants every fleet arch has
   (`f16`/`bf16`/`iu8`/`iu4`). No gfx-specific asm in the reference. One set runs
   gfx1100→1201; RDNA4 fp8/sparse are overlays. (Satisfies the RDNA1→4 portability
   rule by construction.)
2. **Zero-LDS register-tiled by default** — gfx1103 LDS-wedge-safe.
3. **One parity test per kernel** — the real deliverable and the real cost. Quantized
   batched GEMMs parity vs the bf16/fp32 oracle (or an f32 CPU ref); decode GEMVs vs
   an f32 ref. (`parity_*` examples are the template.)
4. **Additive, dispatcher-floored.** Don't touch working overlays; wire the dispatcher
   to pick optimized-when-clean / reference-otherwise, warning on fallback.

## Coordination (parallel sessions)

A parallel session owns the **W(bf16)A(fp32) batched GEMM**. To avoid collisions:
- I do NOT build or register that cell; I rebase chaingun against their commits.
- I work non-colliding slices first: the **warn-on-generic dispatcher mechanism** and
  the **iu8/iu4 reference-GEMM fallback wiring** (the int cells), leaving the
  bf16/fp32 cell and its `kernels.rs`/`dispatch.rs` registration to them.
- Their bf16/fp32 path, once landed, becomes the batched parity oracle the int
  reference GEMMs validate against.

## Starting slice (GEMM-first, non-colliding)

1. **Warn-on-generic mechanism** in the dispatcher (rate-limited per-tuple registry +
   `HIPFIRE_WARN_GENERIC`). Small, shared infra, low collision risk.
2. **iu8 W8A8 reference**: scale-dequant wrapper over `gemm_iu8_i32_wmma` + forward
   wiring for the four dense GEMMs, parity vs f32, warn-on-fallback registered.
3. **W4A8**: nibble-expand prologue + per-group grouped-iu8 accumulation (the quality
   fix), parity vs f32 / the bf16-fp32 oracle once available.
4. Extend the floor/overlay/parity/warn pattern to the remaining ops (rmsnorm, RoPE,
   attention tiers, MoE, lm_head) — each: reference kernel + parity + dispatcher floor.
