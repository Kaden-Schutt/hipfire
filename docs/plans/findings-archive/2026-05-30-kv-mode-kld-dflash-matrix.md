# KV-mode investigation: filtered trunk KV + fwht-vs-asym (KLD + DFlash matrix)

**Date:** 2026-05-30 · **Branch:** `fix/kv-filtered-trunk-fwht-default` · **GPU:** RX 7900 XTX (gfx1100)
**Model:** qwen3.6-27b.mq4 (+ qwen35-27b-dflash.mq4 draft) · 27B hybrid (48 DeltaNet LinearAttention + 16 FullAttention layers)

## Origin
Discord users reported KV-cache VRAM far larger than llama.cpp ("32k ctx = 6GB",
"can't fit 27B+dflash+32k on 24GB", "card dropped from PCI bus"). Maintainer
believed asym3 (the default KV mode) was sub-optimal and DFlash-conflicting vs
fwht/q8. This documents the diagnosis + two fixes + the evidence settling the
KV-mode choice.

## Increment A — filtered trunk KV (the VRAM bug) — VALIDATED, lands independently
The single-GPU trunk allocated KV for **all 64 layers**, but only the 16
FullAttention layers ever read it. Switched the serving arms to the existing
`*_capped_filtered` ctors (FA-layer mask from `config.layer_types`).
- **Numerically a no-op** (DeltaNet layers never touch KV) → output unchanged.
- coherence-gate green; `niah_32k_pflash30` retrieval intact with filtered KV.
- Load line: `asym3 filtered (16/64 layers carry KV ...)`.
- asym3 KV @64k: **6.24 GB → 1.56 GB** (75% cut). @32k: 3.12 → 0.78 GB — fixes the 24 GB fit.

## DFlash τ matrix (7 modes × 5 prompt categories × chatml on/off × 3 reps, temp0/seed0)
- KV modes are **τ-equivalent and coherent** under DFlash; asym3 ≥ fwht3 in both
  chatml settings (i.e. the "asym3 conflicts with DFlash" τ effect did **not**
  reproduce on this greedy/200-tok/ctx2048 bench).
- Only one (borderline) coherence flag: `asym2/chatml/humaneval` last-128
  unique-ratio 0.29 (tier-2 line 0.30); **fwht2 clean** there.
- Robust finding: **chatml OFF → +36% τ** across the board (KV-mode independent).
- Conclusion: DFlash behavior does **not** distinguish the modes → τ is the wrong
  axis to justify a default change. Accuracy (KLD) is.

## KLD sweep — qwen3.6-27b.mq4 vs bf16 (HFKLDR `qwen3.6-27b-MASTER-small`), 24-chunk paired
| rank | mode | KLD | PPL | same-tier Δ |
|---|---|---|---|---|
| 1 | q8 | 0.01038 | 3.4372 | reference |
| 2 | fwht4 | 0.01064 | 3.4368 | −4.4% vs asym4 |
| 3 | asym4 | 0.01113 | 3.4380 | — |
| 4 | fwht3 | 0.01115 | 3.4366 (min) | −11.4% vs asym3 |
| 5 | asym3 | 0.01258 | 3.4424 | — |
| 6 | fwht2 | 0.01505 | 3.4508 | −4.6% vs asym2 |
| 7 | asym2 | 0.01578 | 3.4521 | — |

- **fwht beats asym at every byte tier.** `fwht3 ≈ asym4` → asym4-grade accuracy at
  asym3's 3-bit cost. `fwht3` has the lowest PPL of any mode.
- Paired on identical tokens (mq4-weight error common; only KV-mode differs); the
  11% asym3→fwht3 gap matches the asym3→asym4 bit-tier gap → real signal, not noise.

## Increment B — default flip asym3→fwht3 / asym2→fwht2 — JUSTIFIED by KLD
Free accuracy gain (≈ +1 bit-tier) at **identical VRAM** (same byte layout) and
**zero DFlash penalty** (τ-equal, coherent). Also fixes the latent bug: `fwht*`
was unreachable on the single-GPU path (silently downgraded to asym3) and rejected
by the CLI validator; `turbo*` aliased to `asym*` (misleading).

## Caveats / follow-ups
- KLD is 24-chunk; gaps track bit-tier gaps so confident, but a full 97-chunk +
  bootstrap-CI run would nail the asym3→fwht3 gap to publication precision.
- Run locally on RDNA (kldref local; fwht kernels RDNA-tuned). KV-mode KLD ranking
  is arch-robust; a MI300X/CDNA cross-check is optional.
- Scope: HFQ (.mq4) path only. **safetensors + multi-GPU paths still fall back to
  q8 for an `fwht3` request** (DFlash-safe) — wire fwht arms there as follow-up.
- `sizeAwareKvMode` (asym3→asym4 deep-stack bump) is now dead for the default path
  (default is fwht3); leave for explicit-asym3 users or clean up.
