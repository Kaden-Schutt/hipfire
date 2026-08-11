# gemma4-12B AR-decode efficiency: why 48% roofline (vs qwen3.5-9B 81%)

**Date:** 2026-06-07
**Box:** hiptrx, GPU3 = AMD Radeon AI PRO R9700 (gfx1201 / RDNA4), 640 GB/s GDDR6, ROCm 7.2.2
**Worktree/branch:** `/home/kaden/hipfire-gemma4`, `gemma4-rz` @ 3bf2a267
**Model:** `~/gemma4-12b-mq4attn.hfq` (6.9 GB; dim 3840, 48 layers = 40 sliding hd256 / 8 full hd512, FFN intermediate 15360, vocab 262144, MQ4 attention projections + MQ4 FFN + tied Q8 lm_head)
**Bench prompt:** `benchmarks/prompts/lru_cache_pep8_strict.txt` (md5 `df5dedc8`, 271 tokens incl BOS), greedy, rep-pen 1.3 (generation halts at 120 tokens on EOS — deterministic)
**Method:** release build, warmed per cell, fresh process per measure, median of 3 (variance was nil). rocprofv3 1.1.0 `--kernel-trace --stats` (no segfault on AR decode — low dispatch rate).

> **Diagnosis only.** No optimizations landed. This doc ranks the fixes; the fix is directed separately.

---

## TL;DR

The 2× gap (44.8 vs an 88 tok/s roofline ceiling) is **not** one hog. It is:

1. **gemma4 simply reads more bytes/token.** 7.20 GB of weights/token vs qwen3.5-9B's effective ~5.3 GB. gemma4 is 48 **dense full-Transformer** layers; qwen3.5-9B is a DeltaNet hybrid whose linear-attention layers read far less and launch far fewer kernels per token.
2. **gemma4 runs those bytes at lower efficiency** — ~51% of the all-at-roofline ideal — because of **1535 GPU kernel dispatches/token**, of which **289 are tiny RMSNorm launches** (11% of GPU-busy time) plus 232 separate FWHT-rotate launches, glue ops, and the inter-kernel dispatch gap.

The big GEMMs are **already at roofline** (lm_head 633 GB/s = 99%, fused gate+up 639 GB/s = 100%). The "lm_head is ~4× off roofline" hypothesis is **refuted** — it is essentially perfect. **hipGraph is NOT the lever** (it removes host-launch overhead only — measured +2.9% — and does not reduce the 1535 GPU dispatches, confirmed identical dispatch count graph-on vs -off).

**The only path past ~55 tok/s is reading fewer weight bytes/token (lower-bit FFN). Fusion + graph stack to ~+13% (≈51 tok/s) and no further.**

---

## Phase 1 — warm AR baseline (graph on/off)

Warmed, fresh process, median of 3 (all three runs were byte-identical in tok/s — the determinism is total):

| Config | tok/s | ms/token | effective BW (7.20 GB/tok) | % of 640 GB/s |
|---|---|---|---|---|
| `HIPFIRE_GEMMA4_GRAPH=0` (eager) | **44.8** | 22.32 | 322 GB/s | **50.3%** |
| `HIPFIRE_GEMMA4_GRAPH=1` (hipGraph) | **46.1** | 21.69 | 332 GB/s | **51.8%** |
| Δ (graph on) | **+2.9%** | −0.65 ms | | |

The earlier-cited 44.8 figure was the **eager** path. hipGraph buys +2.9% (matches the module-doc's +2.6% claim). This is small because — as on the MiniMax/gfx1151 attempt — the recoverable portion is only host-launch API time; the GPU command processor still issues all 1535 dispatches per token. On this dGPU the host already runs ahead, so graph recovers little.

(Note: effective BW computed against **7.20 GB/token** of actual weight reads, not the 6.9 GB file size — the tied lm_head is re-read each token and the per-token read basis includes group-scale overhead. See Phase 2.)

---

## Phase 2 — per-token cost attribution (the core deliverable)

### Kernel launches per token: **1535** (measured)

465,204 total GPU dispatches over 303 forward passes (271 prefill + 32 decode; eager prefill and eager decode launch the identical kernel set). 465204 / 303 = **1535.3 dispatches/token**. Exact per-token composition (÷303), which sums to 1535:

| kernel | launches/tok | what it is |
|---|---|---|
| `rmsnorm_f32` | **289** | input + q/k/v + post_attn + post_ffn norms (6×48) + final |
| `mq_rotate_x` | 232 | FWHT rotate before each MQ4 attention/down GEMV |
| `gemv_hfq4g256_multirow_r2` | 232 | MQ4 GEMV: q/k/v/o (attn) + down_proj |
| `__amd_rocclr_copyBuffer` | 202 | residual saves / x resets (memcpy_dtod) |
| `scale_f32` | 97 | q-scale (√hd) + layer_scalar |
| `kv_cache_write_q8_0` | 96 | K + V writes (2×48) |
| `add_inplace_f32` | 96 | attn + ffn residual adds (2×48) |
| `attention_q8_0_kv_swa` | 48 | windowed/full attention |
| `fused_gate_up_hfq4g256` | 48 | FFN gate+up (fused) |
| `fused_rmsnorm_mq_rotate` | 48 | pre-FFN norm+rotate (fused) |
| `mul_f32` / `gelu_tanh_f32` | 48 / 48 | SwiGLU |
| `rope_f32` / `rope_partial_halved_f32` | 40 / 8 | sliding / full RoPE |
| `embedding_q8` / `gemv_q8_0` (lm_head) / `logit_softcap_f32` | 1 / 1 / 1 | head |

### Busy vs inter-kernel gap

From the kernel timeline over the decode region (under rocprof, which inflates the gap via per-launch instrumentation):

| | decode/token |
|---|---|
| wall (profiled) | 24.86 ms |
| GPU-busy (sum of kernel durations) | 19.43 ms (78.2%) |
| inter-kernel gap | 5.43 ms (21.8%) |

The **real** (non-profiled) gap is smaller: at 44.8 tok/s = 22.32 ms/tok wall, with on-device busy ≈ 19.4 ms (kernel durations are measured on-device and accurate), the real gap is ≈ 2.9 ms/token ≈ **13%**. hipGraph recovers only 0.65 ms of that (+2.9%), so **~10% of wall is GPU-CP dispatch latency the host cannot hide** across 1535 tiny launches — fixable only by issuing fewer dispatches (fusion), not by hipGraph.

### Per-token busy-time attribution (grouped)

Total GPU-busy = 17.56 ms/token (profiled). Grouped:

| group | ms/tok | % busy | note |
|---|---|---|---|
| **weight GEMMs** (lm_head + fused gate_up + MQ4 q/k/v/o/down) | 11.98 | **68.2%** | the irreducible weight reads |
| **attention** (`attention_q8_0_kv_swa`, 48/tok) | 2.19 | 12.5% | |
| **RMSNorms** (`rmsnorm_f32` 289 + `fused_rmsnorm_mq_rotate` 48) | 1.94 | **11.1%** | 289 tiny launches |
| **FWHT rotates** (`mq_rotate_x`, 232) | 0.36 | 2.0% | separate launch per MQ4 GEMV |
| RoPE | 0.27 | 1.5% | |
| glue (copy/add/scale/mul/gelu/kvwrite) | 0.80 | 4.6% | |
| **= GEMM+attn (irreducible)** | **14.17** | **80.7%** | runs at ~roofline |
| **= non-GEMM overhead** | **3.38** | **19.3%** | the efficiency tax |

### Effective bandwidth of the big kernels — the lm_head and FFN findings

| kernel | bytes/call | measured | **eff. BW** | % roofline |
|---|---|---|---|---|
| **lm_head** (`gemv_q8_0`, tied Q8, 262144×3840) | 1.070 GB | 1.690 ms | **633 GB/s** | **99%** |
| **fused gate+up** (`fused_gate_up_hfq4g256`, 2×15360×3840 MQ4) | 66.4 MB | 103.8 µs | **639 GB/s** | **100%** |
| all MQ4 GEMVs combined (q/k/v/o/down) | 2.946 GB | 5.307 ms | 555 GB/s | 87% |

- **lm_head: 633 GB/s = 99% of roofline. The "~4× off roofline" claim is REFUTED** — the single-token Q8 lm_head GEMV is essentially perfect. (The spec-verify ~125 GB/s figure was a *batched per-row* GEMV at small B — a different code path that re-streams the weight B times. The B=1 AR `gemv_q8_0` path is optimal.)
- **fused gate+up FFN: 100% of roofline.** Already optimal; nothing to win on the FFN gate/up *kernel*.
- The MQ4 attention/down GEMVs run at **555 GB/s = 87%** — the only sub-roofline weight reads. Cause: small projections (k_proj_full m=512, sliding q/k m=2048/4096) where launch/tail overhead dominates, and they go through the **gfx1010-baseline `multirow_r2`** path (`gemv_rows_default()` does not classify gfx1201 as `is_rdna3_dgpu()` → falls to the R=2 default), not an RDNA4-tuned kernel.

### Full-attention (hd512, 8 layers) vs sliding (hd256, 40 layers)

`attention_q8_0_kv_swa` averages 45.7 µs/call with a wide spread (min 4.8 / p50 44.0 / max 130.9 µs), but the spread is **driven by KV length, not head_dim**: at the bench seq length (~290), full layers read ~290 positions and sliding layers (window 1024) also read ~290, so per-call cost is comparable — **the 8 full layers are NOT disproportionately expensive yet** (no bimodal cluster at 3× the median). They *would* diverge at long context (full grows unbounded; sliding caps at 1024). At this prompt, attention is 12.5% of busy and not the bottleneck. Optimizing the hd512 kernel is a **long-context** lever, not a short-decode one.

### The 4 norms + layer_scalar

RMSNorms are **11.1% of GPU-busy (1.94 ms/token) across 289 launches** — the single largest *non-weight-read* cost and the clearest dispatch-count waste. 288 = 6 norms/layer × 48 (input, q_norm, k_norm, v_norm, post_attn, post_ffn) + 1 final. Each is a trivial 3840-float (or per-head) op paying full launch + memory-roundtrip overhead. `layer_scalar` folds into `scale_f32` (97/tok) — negligible on its own.

### GeGLU FFN

`fused_gate_up` 639 GB/s (100% roofline); `down_proj` is the largest single MQ4 GEMV (33 MB/call, part of the 555 GB/s aggregate); `gelu_tanh_f32` + `mul_f32` are 1.3 µs each (negligible). **FFN is at roofline** — but it is **66% of all weight bytes/token (4.78 GB)**, so it is where the *bytes* are, even though the *kernels* are optimal.

### Weight bytes/token — the dominant fact

| component | GB/tok | % weight traffic |
|---|---|---|
| **FFN** (gate+up+down, 48 layers, MQ4) | **4.778** | **66%** |
| attention proj (q/k/v/o, MQ4) | 1.353 | 19% |
| lm_head (tied Q8) | 1.070 | 15% |
| **total weights/token** | **7.201** | |
| + KV reads (Q8, seq~290) | 0.053 | |
| **total memory traffic/token** | **7.254** | |

**Ideal @ 640 GB/s = 88.2 tok/s. Measured 44.8 = 51% of ideal.** The 2× gap = the 19% non-GEMM overhead × the ~13% dispatch gap × the 87% MQ4-GEMV efficiency, compounded.

### qwen3.5-9B contrast (architectural — the 9B model is not on this box)

The full qwen3.5-9B weights are not present on hiptrx (only the 557 MB dflash draft head), so this is reasoned from architecture + the reference number, **not** a same-box re-profile (stated honestly):

- qwen3.5-9B @ 97 tok/s, 80.7% roofline ⇒ 516 GB/s effective ⇒ **~5.32 GB/token** read.
- qwen3.5-9B is a **DeltaNet hybrid**: ~3/4 of layers are linear-attention (gated delta rule) with **no growing-KV softmax attention** and **no 6-norm sandwich** — each such layer collapses the q/k/v/o + softmax + 6-norm stack (≈30 dispatches in gemma4) into a handful of fused recurrence kernels. So qwen35 launches **far fewer kernels/token** than gemma4's 1535 and reads ~26% fewer bytes/token.
- **Net:** gemma4 is penalised on BOTH axes — more bytes/token (7.2 vs 5.3 GB, +36%) AND lower efficiency (51% vs 81%, because of 1535 dispatches with 289 norms vs qwen35's lean linear layers). That is the entire 44.8-vs-97 gap.

---

## Phase 3 — ranked fix plan

Baseline to beat: **44.8 tok/s eager / 46.1 tok/s graph-on**. Ceiling at this byte budget: **88 tok/s**. Two distinct regimes: (A) **byte-reduction** levers move the ceiling; (B) **dispatch/fusion** levers move efficiency toward the ceiling but cap at ~+13%.

### Tier 1 — byte reduction (moves the ceiling; the only path past ~55 tok/s)

**1. Lower-bit FFN (MQ3 gate/up/down).** *Targets: FFN = 66% of weight bytes (4.78 GB/tok).*
- **Expected gain:** FFN MQ4→MQ3 cuts ~1.06 GB/tok → 6.14 GB/tok total. At the current 51% efficiency that is **~53 tok/s (+18%)**; if paired with the Tier-2 fusion (push efficiency up) the ideal moves to 104 tok/s and a realistic 60-65 tok/s is in reach.
- **Impl sketch:** requantize gate/up/down to MQ3G256 in the converter; the `fused_gate_up` + down kernels already have MQ3 arms (MQ3G256 is in the rotation/GEMV dtype set). No new kernel.
- **Risk:** **HIGH on coherence.** This is the real lever but FFN is the bulk of model capacity. Must pass `coherence-gate.sh` + tiny-oracle cosine vs HF (the gemma4 memory shows MQ3 attn corrupts; FFN MQ3 is less explored). Validate cosine before any tok/s claim. Down-proj is the most sensitive (largest, outlier-heavy) — consider MQ3 gate/up but keep down at MQ4 (mixed precision), which still cuts ~0.7 GB/tok.

**2. lm_head: leave as Q8 (already 99% roofline). NOT a lever.** Listed only to close it out — do not touch. It is 15% of bytes and already optimal; MQ4 lm_head would risk logit quality for a ~0.6 GB/tok save that the tied-embed sharing complicates.

### Tier 2 — dispatch/fusion (moves efficiency toward ceiling; safe, byte-identical; stacks to ~+13%)

These are the qwen35-mirror pattern the FFN path already proves works. All byte-identical (no coherence risk) if implemented as fused equivalents.

**3. Fuse q_norm / k_norm (+ √hd scale + RoPE) into one launch.** *Targets: ~144 of the 289 norm launches + the q-scale + rope.*
- **Expected gain:** ~0.7 ms/tok → **~46.3 tok/s (+3.2%)** standalone; bigger combined (also removes ~96 dispatches → cuts dispatch gap).
- **Impl sketch:** `kernels/src/fused_qk_l2_norm_scale.hip` and `fused_qk_l2_norm_scale_interleave_f32_batched.hip` already exist — wire them onto the AR path to replace the 3 separate `rmsnorm_batched` + `scale_f32` + feed `rope`. The v_norm (weight-less) can fold similarly.
- **Risk:** LOW. Byte-equivalent; gate behind an env like the existing fused flags; verify cosine unchanged.

**4. Fuse input_layernorm into the q/k/v projection (rmsnorm+rotate), mirroring the FFN path.** *Targets: 48 input-norm launches + their FWHT rotates.*
- **Expected gain:** ~0.5 ms/tok → **+2.3%**.
- **Impl sketch:** `fused_rmsnorm_mq_rotate` already folds pre-FFN norm+rotate; apply the same to the pre-attention norm so the MQ4 q/k/v GEMVs consume a pre-normed, pre-rotated input (removes 48 `rmsnorm_f32` + folds into the 232 `mq_rotate_x`).
- **Risk:** LOW (proven pattern). gemma4 attention is MQ4 here, so `mq_rotate_x` is already in the path — folding the norm in is purely launch reduction.

**5. Fuse post_attn / post_ffn norms into the residual-add.** *Targets: 96 norm launches + 96 add launches + 96 memcpy resets.*
- **Expected gain:** ~0.4 ms/tok → **+1.8%** (mostly dispatch-count reduction).
- **Impl sketch:** a `rmsnorm_then_residual_add` fused kernel (norm tmp + add to residual in one pass), replacing `rmsnorm_f32` + `memcpy_dtod` (reset x) + `add_inplace_f32`. Eliminates the 202 `copyBuffer` residual-reset launches too.
- **Risk:** LOW-MED (new small kernel); byte-identical math.

**6. RDNA4-tune the MQ4 multirow GEMV.** *Targets: the 555-vs-639 GB/s (87%) gap on small attention projections.*
- **Expected gain:** ~0.5 ms/tok → **+2.3%** if it closes to ~roofline.
- **Impl sketch:** `gemv_rows_default()` does not classify gfx1201 as `is_rdna3_dgpu()`, so it falls to the **gfx1010-baseline R=2** path. Add an RDNA4 arm (try R=1 RDNA3-style or an R=4 sweep) — empirically determine on the fleet per the "don't disable on uncertainty" rule. Sweep `HIPFIRE_GEMV_ROWS ∈ {1,2,4,8}` first (no code change) to size the win before committing a default.
- **Risk:** LOW (it is a tuning knob with an env override); the RDNA4 coherence gate catches a wrong predicate.

**7. hipGraph ON by default.** *Already measured +2.9%.*
- **Expected gain:** +2.9% (recovers host-launch overhead only; does **not** reduce the 1535 GPU dispatches — confirmed identical dispatch count graph-on vs -off).
- **Impl sketch:** flip `HIPFIRE_GEMMA4_GRAPH` default to ON (it is byte-identical and already validated).
- **Risk:** LOW. Capture-safety invariants already in place. Caveat: capture is incompatible with the per-layer-download oracle path (already gated).

### Stacked estimate

Tier-2 fusion + graph (items 3–7) stack to **~50-51 tok/s (+12-13%)** — matching the module-doc's own "46.8→50.7" finding. That is the safe ceiling. **Tier-1 byte reduction (item 1) is the only way to 60+ tok/s**, and it must clear coherence first.

### What is NOT worth doing

- **lm_head GEMV tuning** — already 99% roofline.
- **fused gate+up tuning** — already 100% roofline.
- **fused-Q8-QKV for attention** — N/A: this model's attention projections are MQ4, not Q8, so `fused_gate_up_q8_0` / a Q8 fused-QKV never fires. (The `fused_qk_enabled()` path in `forward.rs` is dead for `mq4attn`.)
- **hd512 full-attention kernel work** — not a short-decode bottleneck (only matters at long context).

---

## Reproduction

```
ssh hiptrx; cd /home/kaden/hipfire-gemma4
cargo build --release -p hipfire-arch-gemma4 --features deltanet --example infer_gemma4
source scripts/gpu-lock.sh && gpu_acquire gemma4-ar-profile
PROMPT=$(cat benchmarks/prompts/lru_cache_pep8_strict.txt)
# baseline (warm once at --max 16, then measure):
HIP_VISIBLE_DEVICES=3 HIPFIRE_GEMMA4_GRAPH=0 ./target/release/examples/infer_gemma4 \
  --model ~/gemma4-12b-mq4attn.hfq --prompt "$PROMPT" --max 200 --rep-pen 1.3
# attribution:
HIP_VISIBLE_DEVICES=3 HIPFIRE_GEMMA4_GRAPH=0 rocprofv3 --kernel-trace --stats \
  --output-format csv -d /tmp/rp -- ./target/release/examples/infer_gemma4 \
  --model ~/gemma4-12b-mq4attn.hfq --prompt "$PROMPT" --max 32 --rep-pen 1.3
gpu_release
```
