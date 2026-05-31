# Design: V-cache quantization — FWHT-V → Lloyd-V pipeline

**Date:** 2026-05-31 · **Status:** design / approved-to-write, pending spec review · **Priority #1.**
**Supersedes** Section 1 of `2026-05-31-kv-vquant-and-setup-wizard.md` with the implementable detail; the wizard/autodetect/Hermes-autoconfigure work in that doc stays deferred.

## 1. Goal

Quantize the **V-cache below its current always-Q8** layout. Across every KV mode today
(`q8`, `asym2/3/4`, `fwht2/3/4`) only **K** varies; **V is always Q8_0 = 272 B/head** (head_dim
256: 8 blocks × [2 B fp16 scale + 32 int8]). V is the *bigger half* (K is 68–132 B) and the
dominant untouched term.

We add a **FWHT-rotated, centroid-LUT-quantized V** ("Lloyd-V") at 2/3/4-bit, and **decouple
the K-bit and V-bit choice** so the cache can carry, e.g., `fwht3`-K with `lloyd4`-V. The
deliverable is the new modes + a KLD matrix that picks the best byte split; **no default flip
without KLD parity to Q8-V and a green coherence gate.**

## 2. Why this works (the V-vs-K asymmetry)

The existing K rotation is free because **K is dotted**: both Q and K are forward-rotated and
the orthogonal transform cancels inside the score, `(Hq)·(Hk) = q·k`. No inverse is ever
materialized for K.

V is different — **V is summed, not dotted**: the attention output is `o = Σₜ pₜ·vₜ` (softmax
weights `pₜ` over the tile). Rotation is linear and the softmax weights are scalars, so

```
Σₜ pₜ·(H·vₜ) = H·(Σₜ pₜ·vₜ) = H·o
```

and the online-softmax rescaling (scalar `exp`-max corrections) also commutes with `H`.
Therefore we can **store H·vₜ, accumulate entirely in rotated space, and apply a single inverse
FWHT to the output accumulator once per (query, head)** — at the very end of the flash-attention
loop, after final renormalization. One `fwht_shfl_inverse_256` over the wave-resident output
registers. Cost is ~1 transform per head per query token, independent of context length.

Two consequences:
- **FWHT-V is nearly free at decode** (the dominant inner loop is unchanged; only the tail adds
  one inverse), unlike a naive per-element V rotation.
- **V rotation does not touch the softmax probabilities at all** (those come from K/Q). It only
  reshapes the value distribution, so it is numerically transparent up to V-quant error. The
  Hadamard mixing Gaussianizes V (CLT over the butterfly), which is exactly the precondition the
  fixed `TURBO_C*` centroid LUTs assume (unit-Gaussian after a per-head `cnorm` scale).

## 3. Byte layout

Lloyd-V mirrors the K layout exactly: **one f32 `cnorm` per head + packed b-bit indices over all
256 dims** (no per-32-block scales — the rotation makes one scale per head sufficient). This is
why Lloyd-V is both smaller *and* better-conditioned than a naive per-block Q4.

| V scheme | bits | bytes/head (hd=256) | quantizer |
|---|---|---|---|
| Q8 (today) | 8 | 272 | uniform int8, per-32-block fp16 scale, **no rotation** |
| **lloyd4-V** | 4 | **132** (4 + 256·4/8) | FWHT-V + `TURBO_C4_256` (16 centroids), 1 cnorm |
| **lloyd3-V** | 3 | **100** (4 + 256·3/8) | FWHT-V + `TURBO_C3_256` (8 centroids), 1 cnorm |
| **lloyd2-V** | 2 | **68** (4 + 256·2/8) | FWHT-V + `TURBO_C2_256` (4 centroids), 1 cnorm |

**Total B/head = K + V** (per-token on the 27B = ×64: 16 FA layers × 4 KV heads):

| K ↓ \ V → | q8 (272) | lloyd4 (132) | lloyd3 (100) | lloyd2 (68) |
|---|---|---|---|---|
| **fwht2 (68)** | 340 | 200 | 168 | 136 |
| **fwht3 (100)** | **372** ← today | 232 | 200 | 168 |
| **fwht4 (132)** | 404 | 264 | 232 | 200 |

e.g. today's `fwht3`/Q8-V = 23,808 B/tok → `fwht3`-K/`lloyd3`-V (200 B/head) = 12,800 B/tok, **−46%**.

## 4. Components to build

The K FWHT path is the template throughout. New code is V-side rotation + the output inverse + V
byte-sizing.

### 4a. V write kernels (rotate → quant → pack)
Mirror `kv_cache_write_asym_k_fwht{2,3,4}.hip` + `_batched`, but for V: apply
`fwht_shfl_forward_256` (sign tables from `gen_fwht_signs`, seeds 42/1042 — **reuse K's tables**),
derive a per-head `cnorm`, quantize to `TURBO_C{2,3,4}_256` via the existing branchless threshold
match, pack indices, write `[cnorm | packed]`. Need batched (prefill) + single-token (decode)
variants. Today V writes go through `kv_cache_write_q8_0[_batched]` for *all* modes
(`llama.rs:2148-2177, 2938`); those call sites become V-mode-dispatched.

### 4b. Attention kernel V path (dequant in rotated space + single inverse)
In `attention_flash_fwht{2,3,4}_tile[_batched].hip` (and the q8 tile if V-mode is independent of
K-mode), the V phase changes from "load fp16 scale + int8, accumulate" to: **unpack b-bit index →
`cnorm · TURBO_C*[idx]` → accumulate into the rotated output accumulator**; then, after the tile
loop and final softmax renorm, call **`fwht_shfl_inverse_256` once** on the output registers
before writeback. The online-softmax max/sum bookkeeping is unchanged (commutes with `H`).

### 4c. Cache allocation / sizing
Every ctor currently hardcodes `v_bpp = n_kv_heads · (head_dim/32) · 34`
(`llama.rs:3416-3418` and ~10 sibling sites). Parameterize V bytes-per-head as
`v_bpp = n_kv_heads · (4 + head_dim·vbits/8)` when V-mode is Lloyd, keeping `·34` for Q8-V. The
row-stride / block-offset `·34` constants in the attention + write kernels become V-mode constants.

### 4d. `KvCache` V-mode tracking
Today V-ness is implicit (`quant_q8: bool` + the `quant_asym*` flags, `llama.rs:3319`). Add an
explicit **V-mode** (enum: `Q8 | Lloyd2 | Lloyd3 | Lloyd4`) carried alongside the K-mode so
stride math and kernel dispatch can branch. K-mode and V-mode are independent.

### 4e. `eval_hipfire` — independent K/V selection (for the matrix)
Keep `--kv-mode` selecting **K** (already accepts `fwht2/3/4`); add **`--kv-v {q8,lloyd2,lloyd3,lloyd4}`**
(default `q8` → byte-identical to today). This is the minimal surface needed to run all 12 cells.
User-facing CLI naming (composite mode strings vs two flags) is a later cosmetic decision (§7).

## 5. Validation — the KLD sweep (RESULTS)

**Design change during implementation (build-time finding).** The original plan assumed the
existing K fwht kernels could be reused for V to give a clean *independent* 3 K × 4 V matrix.
They can't: the existing kernels **bundle rotation width with bit-count** — `fwht3` is a
**256-wide** FWHT (8 dims/thread, `TURBO_C3_256`) while `fwht2`/`fwht4` are **128-wide** (per-half,
`TURBO_C2`/`TURBO_C4`). The two layouts use different thread→dim mappings and do not interoperate,
so an attention kernel can only un-rotate V written in *its own* layout. Rather than build a full
set of dedicated cross-layout V kernels, we **fixed K = fwht3 (256-wide, the KLD-validated best K)
and swept V at 2/3/4 bits all in the 256-wide layout** — two new 256-wide V writers
(`kv_cache_write_fwht256_{2bit,4bit}`) plus the existing fwht3 writer for lloyd3. This directly
answers Priority #1 ("how low can V go below Q8") and drops only the secondary K-axis (fwht3
already won the K comparison). The full independent matrix is a deferred follow-up (§7).

**Harness:** `eval_hipfire --kv-mode fwht3 --kv-v <V> --scoring-mode prefill --max-chunks 24`
vs the bf16 ref `~/.hipfire/kldref/qwen3.6-27b-MASTER-small.kldref.bin` (qwen3.6-27b), run on
local gfx1100 (7900 XTX) for internal consistency (all cells same machine/session).

**Results (2026-05-31, 24-chunk, fwht3-K, qwen3.6-27b):**

| V mode | bits | V B/head | total B/head | per-tok (27B) | **KLD** | PPL | vs Q8-V |
|---|---|---|---|---|---|---|---|
| q8 (today) | 8 | 272 | 372 | 23,808 B | **0.011148** | 3.4366 | baseline |
| **lloyd4** | 4 | 132 | **232 (−38%)** | 14,848 B | **0.011487** | 3.4374 | **+3.0%** |
| **lloyd3** | 3 | 100 | **200 (−46%)** | 12,800 B | **0.012120** | 3.4413 | +8.7% |
| lloyd2 | 2 | 68 | **168 (−55%)** | 10,752 B | **0.014568** | 3.4525 | +30.7% |

(The Q8-V cell at 0.01115 reproduces the historical fwht3 baseline ≈0.0112 — harness validated.
4-chunk smoke earlier showed the same monotonic ordering, with lloyd4 ≈ q8 to 4 digits.)

**Read-out:** monotonic and physics-consistent (more V bits → lower KLD; a layout/inverse bug
would have exploded KLD). **lloyd4-V is Q8-V-grade** (+3.0% KLD, PPL essentially unchanged) at
**−38% total KV** — the FWHT rotation Gaussianizes V so 4-bit centroids are near-lossless.
**lloyd3-V** trades +8.7% KLD for **−46% KV**. **lloyd2-V** (+30.7%) confirms V *is* bit-sensitive
below 3 bits and is not default material (available for extreme-context users who accept the cost).

## 6. Decision rule & guardrails

- Ship a V-quant **default flip** only for cell(s) with **KLD ≈ the Q8-V baseline** at the fewest
  bytes — same bar that justified the asym3→fwht3 flip.
- **Mandatory before any claim:** `./scripts/coherence-gate.sh` green. V lands directly in the
  residual stream (not just softmax scores), so this is prime "synthetic win hides a real
  regression" territory — exactly the failure mode CLAUDE.md's perf rule and the memory
  falsification log warn about. KLD parity is necessary, coherence is the gate.
- New/changed kernels + dispatch ⇒ the pre-commit coherence hook applies.

**Coherence result (2026-05-31, lloyd4-V):** `coherence-gate.sh` (HIPFIRE_KV_V=lloyd4) → exit 0,
all 11 short-battery cases ran with no hard errors. Direct 27B check (qwen3.5-27b.mq3,
fwht3-K + lloyd4-V) confirmed the override engaged (`[daemon] V-cache mode override → lloyd4`)
and produced fluent, on-topic output with no attractor/loop/special-token leak. lloyd4-V is
coherence-clean. (lloyd3-V / lloyd2-V coherence not yet gated — gate before defaulting either.)

## 7. Open decisions

1. **User-facing mode surface:** composite names (`fwht3_lloyd4`) vs two independent flags
   (`--kv-k` / `--kv-v`). Internal decoupling is required either way; the CLI face can be decided
   after the matrix shows which pairs are worth shipping.
2. **lloyd2-V fate:** include in the sweep, but expected to fail KLD for V — keep only if it
   surprises us.
3. **CASK / V eviction interaction:** the `_capped` (CASK) path quantizes/evicts V; confirm the
   capped V ctors get the same Lloyd-V treatment or are explicitly Q8-only for now.
4. **Decode single-token V kernel:** needs its own Lloyd-V write variant (not just the batched
   prefill one).
5. **Multi-GPU parity:** the `_multi_filtered` ctor family added in PR #366 must learn V-mode too.

## 8. Out of scope (deferred, unchanged)

- **Plain Q4-V (unrotated uniform 4-bit):** dropped for now — suspected dead end; revisit only if
  Lloyd-V underperforms and we want to attribute rotation-vs-codebook.
- Adaptive per-block MQ-Lloyd refit and ParoQuant learned rotation for KV — weight-static
  techniques that don't transfer to a dynamic cache without offline calibration.
- Hardware-aware setup wizard / VRAM autodetect / Hermes-autoconfigure (companion doc).
- DFlash `hidden_rb` full-ctx over-allocation (separate ~24–32k DFlash ceiling track).
