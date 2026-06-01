# Design: KV V-cache quantization + hardware-aware setup wizard

**Date:** 2026-05-31 · **Status:** design / not started · **Priority #1: V-cache quantization (Section 1). Everything else follows.**

## Context

PR #366 (`fix/kv-filtered-trunk-fwht-default`) landed AR KV-occupancy correctness:
filter trunk KV to the FullAttention layers of the hybrid Qwen3.5/3.6 stack across
all paths (single-GPU / safetensors / multi-GPU), default flipped asym3→fwht3 /
asym2→fwht2 (KLD-validated: fwht beats asym at every byte tier; fwht3 ≈ asym4 at
asym3 cost), DFlash to default-off. AR context on a 24 GB card went ~80–100k → ≥327k.

This doc is the **next phase**. The prior turn's full read-back was confirmed by the
user; the remaining open decisions are preserved in Section 4.

---

## 1. PRIORITY #1 — V-cache quantization (the big remaining KV saving)

**Observation.** Across *every* current KV mode (q8 / asym2/3/4 / fwht2/3/4) the **K**
side is what varies (2/3/4-bit, Givens- or FWHT-rotated) — but **V is *always* Q8_0**.
For head_dim=256: V = `blocks_per_head(8) × 34 B = 272 B/head`, while K is only
**100 B (3-bit) … 132 B (4-bit)**. So **V is the bigger half** and is the dominant
term we've never touched.

**Idea.** Quantize V below Q8 — **Lloyd-V** (Lloyd-Max optimal scalar codebook, like
the MQ4-Lloyd weight path) or **Q4-V** (4-bit) — as new KV-mode variants.

**Byte math (head_dim=256, per head):**
| V scheme | bytes/head | K (fwht3=100) → total | vs fwht3-today (372) |
|---|---|---|---|
| Q8 (today) | 272 | 372 | — |
| Q4-V (Q4_0: 18 B/block × 8) | ~144 | ~244 | **−34%** |
| Lloyd-V 4b | ~144 | ~244 | −34%, better fidelity than Q4 |
| Lloyd-V 3b | ~112 | ~212 | −43% (riskier) |

On the 27B (16 FA layers, 4 KV heads): fwht3-today = 23,808 B/tok; fwht3+Q4-V ≈
15,616 B/tok → another ~34% on top of the layer filter already shipped.

**The risk (why this needs validation, not just shipping).** V quant is *more*
accuracy-sensitive than K quant: K only perturbs the softmax *scores*, but **V is
summed directly into the attention output** — error there propagates straight into
the residual stream. Q4-V may hurt materially more than Q4-K. **Must KLD-validate.**

**Validation plan (reuse the fwht harness):**
- Extend `eval_hipfire --kv-mode` (already extended to fwht2/3/4 in PR #366) to the
  new V-quant variants.
- Paired KLD sweep vs the bf16 reference `~/.hipfire/kldref/qwen3.6-27b-MASTER-small.kldref.bin`
  (`eval_hipfire` → `kld_reduce.py`), same method that validated fwht (24-chunk paired,
  fwht3=0.0112 etc.). Compare V-quant variants against the Q8-V baselines.
- DFlash-coherence check if DFlash is in play (though DFlash is default-off now).
- **Ship a V-quant default only if KLD ≈ Q8-V (or within an acceptable, measured band).**

**Deliverable:** new KV-mode variants with sub-Q8 V (e.g. `fwht3v4`, `lloydv`),
KLD-swept + coherence-gated, exposed in the wizard's cost table (Section 2). The K-side
ctor/byte-layout work in `llama.rs` (the `new_gpu_*` family) is the template; V layout
+ the V quant/dequant kernels (`rdna-compute`) are the new code.

---

## 2. Hardware-aware setup wizard (follows V-quant)

**Core shift:** stop hardcoding assumptions (the gfx→VRAM table is *wrong* — 7900 XT=16
(is 20), gfx1201=16-or-32). Detect, compute, let the user choose.

### 2a. Hardware autodetect
- Enumerate amdgpu devices (`/sys/class/drm/card*/device` or `hipGetDeviceCount`); read
  each one's VRAM **as amdgpu/HIP reports it** (`hipMemGetInfo` / `get_vram_info()` exists).
- Default device = **highest-VRAM GPU**.

### 2b. Setup wizard TUI
- Interactive (`hipfire setup` and/or extend `hipfire config` TUI): pick default model,
  KV mode, context length, device.
- **Live cost table:** for the selected model, show **every KV mode and its VRAM at the
  selected ctx** — model-aware rate (FA-layer count varies: 0.8b=6, 4b/9b=8, 27b=16,
  35b-a3b=10 of total) AND total fit (weights + KV + scratch), not just KV.
- Offer ctx = a fixed number **or** "auto = max that fits" (computed at load from
  free-VRAM-after-weights ÷ kv_rate − headroom). Auto-size is the principled default and
  folds in the VRAM-preflight idea.

### 2c. Autoconfigure-for-Hermes preset (one-click)
VRAM → model (user's bands; **flagged conflicts in Section 4**):
- `<16 GB → 9b` · `16–24 GB → 27b` · `32–64 GB → 35b-a3b` · `96 GB+ → DeepSeek / MiniMax (optional)`
- Plus: DFlash off, fwht3/fwht2 (or a validated V-quant mode) KV, agentic serve settings
  (idle_timeout handling for sporadic long sessions).

---

## 3. Reusable building blocks
`get_vram_info()` (hipMemGetInfo) · existing `hipfire config` TUI + validator · the
per-(model, KV-mode) byte formulas · `eval_hipfire` KLD harness + `kld_reduce.py` +
the bf16 kldref · amdgpu device enum.

## 4. Open decisions (resolve before building)
1. **`16 GB → 27b` doesn't fit** (27b wts ~14 GB → ~16–20k ctx or OOM). Shift to
   `<20 GB → 9b`, `20–24 GB → 27b`? Or cap+warn?
2. **Band gaps:** 24–32 GB and 64–96 GB unspecified.
3. **96 GB+** = single card (MI300X 192 GB) or aggregate multi-GPU? Wizard handles
   multi-GPU/pp config or single-device only for now?
4. **Wizard surface:** new `hipfire setup` vs extend `hipfire config`; trigger first-run /
   explicit / both.
5. **Ctx fixed vs "auto"** (offer both).
6. **Cost table:** model-aware + total-fit (confirmed yes).
7. **a3b vs 27b at 32–64 GB** (user picked a3b — MoE/throughput).
8. **V-quant scope:** part of this effort (add modes + KLD) vs separate research spike.

## 5. Out of scope / parallel
- DFlash `hidden_rb` full-ctx over-allocation (the ~24–32k DFlash-on ceiling) — separate PR.
- CASK-aware `*_capped_filtered` asym4/asym2/fwht4 single-GPU variants — small follow-up.
- The Hermes-cluster serve bugs (config-key mismatch, no prompt-caching → 20× hello,
  thinking-only loop) — separate UX PR slate (see prior session).
