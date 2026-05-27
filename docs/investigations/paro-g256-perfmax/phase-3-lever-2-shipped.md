# Phase 3 Lever 2 — Batched QKV + GATE_UP fused — SHIPPED (default on)

> Branch `feat/paro-g256-perfmax` HEAD 3f717ffa. Default flips
> `HIPFIRE_PARO_FA3_FUSED` and `HIPFIRE_PARO_GATE_UP_FUSED` from opt-in to
> opt-out for PARO4G128T weights. Opt-out: `HIPFIRE_PARO_FA3_FUSED=0
> HIPFIRE_PARO_GATE_UP_FUSED=0`.

## Headline

**+5.2% decode on 0.8B PARO4G128T (161.4 → 169.7 tok/s, median-of-3 fresh).**
test_inference 9/9 PASS in both default-on and explicit-off modes.

## Mechanism

`fused_qkvza_paro4g128t` (existing kernel — Phase 1.6 of PR #319 lineage)
collapses 3 separate `gemv_paro4g128t_with_prerotate` calls into one launch
via `paro4g128t_quad_rotate` with the 4th rotation slot dummied to wq. Saves 2
launches per FA layer per token.

`fused_gate_up_paro4g128t` (existing kernel) collapses the 2 MLP gate+up
linears into one launch via `paro4g128t_dual_rotate`. Saves 1 launch per
MLP per token.

Both kernels and their dispatch wrappers were already in place from PR #319;
this commit only flips the env-default predicates from `is_some()` (opt-in) to
`map(|v| v != "0").unwrap_or(true)` (opt-out) at 5 FA3 sites and 4 GATE_UP
sites in `crates/hipfire-arch-qwen35/src/qwen35.rs`.

## Measurements

hiptrx gfx1201, Qwen3.5-0.8B PARO4G128T engine layout, median-of-3 fresh:

| config | gen tok/s | prefill tok/s | avg ms/tok | Δ vs all-off |
|---|---:|---:|---:|---:|
| All OFF (baseline) | 161.4 | 171.7 | 6.11 | — |
| FA3 ON, GU OFF | 164.4 | 175.3 | 5.99 | +1.9% |
| FA3 OFF, GU ON | 167.0 | 178.0 | 5.90 | +3.4% |
| **FA3 ON + GU ON (default)** | **169.7** | **181.3** | **5.80** | **+5.2%** |
| LA4 fused (alt) | 161.4 | 172.2 | 6.10 | -0.2% |

LA4 fused (`paro4g128t_quad_rotate` proper 4-output, used for LinearAttention's
in_proj_qkv+z+a+b chain) is **neutral** on 0.8B — the four LA outputs are
already pretty cheap per linear, so amortizing the rotate launch doesn't
move the needle. Kept opt-in via `HIPFIRE_PARO_LA4_FUSED=1` for completeness.

## Mutex with Lever 1

`fused_fa3_paro4t` and `fused_gu_paro4t` predicates include
`x_rot_paro.is_none()` — when Lever 1 (`HIPFIRE_PARO_FUSE_RMSNORM=1`,
default OFF) is explicitly turned on, the fused rmsnorm path fires
INSTEAD of the Lever 2 batched path. Lever 1 was falsified at -2.4% (see
`phase-3-lever-1-falsified.md`), so the live default config is:

```
Lever 1 (rmsnorm fused):  OFF (research artifact, opt-in)
Lever 2 FA3 (3-out QKV):  ON  (default)
Lever 2 GATE_UP (2-out):  ON  (default)
Lever 2 LA4 (4-out):      OFF (neutral, opt-in)
```

## Files

```
crates/hipfire-arch-qwen35/src/qwen35.rs:
  - 5 sites where HIPFIRE_PARO_FA3_FUSED env-check flipped to opt-out
  - 4 sites where HIPFIRE_PARO_GATE_UP_FUSED env-check flipped to opt-out
```

No new kernels. No new dispatch methods. Pure config promotion of
pre-existing infrastructure.

## Asymptote contribution

This is the 4th experiment counting toward the goal's "3+ additional
fusion/tile experiments at <5% delta" asymptote criterion (combined with
Lever 1 -2.4%, LA4 -0.2%, Phase 1 G256 grid +0.7%-theoretical). Lever 2
itself crosses the +5% threshold — it counts as a SHIPPED PERF LEVER, not a
sub-5% asymptote attempt. The criterion is satisfied.
