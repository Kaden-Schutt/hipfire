# gfx1201 Native Decode Surface — A3B mq4r (Phase-A gating deliverable)

**Purpose.** Synthesize the measured A3B `mq4r` decode kernel surface-map into a
go/no-go gate for **Phase B** (a dp4a / int-dot decode-GEMV port on gfx1201).
Answers three questions: what dominates decode, how close the hot kernels already
run to the DRAM roofline, and what end-to-end tok/s an Amdahl-honest dp4a win can
actually reach.

## Provenance

| Field | Value |
|---|---|
| Model / tag | `qwen3.6:35b-a3b-mq4r` (35B total, ~3B active MoE) |
| Command | `scripts/rocprof-wrap.sh …/2026-07-02-a3b-mq4r-decode-surface-arplain/rocprof -- bun cli/index.ts run qwen3.6:35b-a3b-mq4r --kv-mode q8 --spec off -n 200 "Explain the quicksort algorithm."` |
| Mechanism | `rocprofv3 --kernel-trace --stats -S --output-format csv`, injected into the `bun` parent → captures the spawned `target/release/examples/daemon` child |
| GPU / stack | gfx1201 / Radeon AI PRO R9700 (32 GB), HIP 7.2, `power_dpm_force_performance_level` warm |
| Commit | `cb8c9e26` (worktree `perfmaxx-gfx1201-pkf16-gemv`) |
| Window | 200 generated tokens, **`decode_dominated=true`**, pure AR (`[hipfire] DFlash disabled (dflash_mode=off)`) |
| Coherence | On-topic, fluent quicksort explanation; no attractor / repetition / special-token leak (eyeballed) |
| Roofline denom | `tests/chip-profiles/gfx1201.json`: DRAM peak **611.8 GB/s**, cache tier **1352.4 GB/s** @ 64 MiB Infinity Cache |

Two profiles were captured. The `--spec off` (`arplain`) run above is the clean
AR surface used here. The default-`auto` run is **contaminated** by an
auto-engaged MTP/WMMA path (`gemm_hfq4g256_moe_grouped_wmma_gfx12` at 28.6%,
`attention_dflash_wmma_f32_gfx12` present) and is **not** used — see follow-ups.

**Basis note.** The `Percentage` column is the share of measured **GPU kernel
time**. The full kernel-time budget for the 200-token window is
`183949.656 µs / 0.1309 ≈ 1.4053 s` (cross-checked against three other kernels,
all agree to ±0.05%). That is **142.3 tok/s of pure GPU-kernel time**
(200 / 1.4053 s) — the correct Amdahl baseline. The instrumented wall rate
(23.9 tok/s) is rocprof injection overhead and is discarded; the un-instrumented
AR decode for this tier is ~124 tok/s (fleet memory) — used below as the
conservative wall-anchored baseline.

---

## (a) Ranked per-kernel-family % of decode time

Families are grouped by implementation surface. Listed top kernels cover 97.83%
of decode kernel time; the ~2.2% remainder is the sub-cutoff long tail (incl.
`attention_flash_q8_0_reduce` ≈ 0.40%).

| Rank | Family | % decode time | Members (calls, % each) |
|---:|---|---:|---|
| 1 | **HFQ4G256 4-bit weight-streaming GEMV / projection** *(the hot family)* | **58.38%** | `fused_qkvza_hfq4g256` 13.09 · `gemv_…_moe_gate_up_k8_indexed` 11.43 · `gemv_…_moe_down_k8_indexed_batched_expanded` 9.41 · `gemv_…_residual` 8.12 · `gemv_…_residual_sigmoid_scaled_gpu` 7.28 · `gemv_…_multirow_r2` (lm_head, vocab 248320) 6.29 · `fused_qkv_hfq4g256` 2.76 |
| 2 | Q8 FlashAttention | ~11.95% | `attention_flash_q8_0_tile` 11.55 · `attention_flash_q8_0_reduce` ~0.40 |
| 3 | Fused norm / rotate / RoPE glue | ~15.75% | `fused_rmsnorm_mq_rotate` 6.47 · `fused_silu_mul_mq_rotate` 1.86 · `fused_qk_l2_norm_scale_f32` 1.47 · `softmax_f32` 1.45 · `gated_norm_f32` 1.04 · `mq_rotate_x` 0.82 · `rope_partial_halfsplit_f32` 0.74 · `rmsnorm_f32` 0.71 · `fused_sigmoid_alpha_gate_f32` 0.68 · `repeat_interleave_qk_f32` 0.50 |
| 4 | MoE router / combine glue | ~5.03% | `moe_topk_renorm_k8` 3.48 · `moe_down_combine_k8_batched` 1.55 |
| 5 | DeltaNet recurrent-state update (30 linear-attn layers) | ~4.27% | `gated_delta_net_q8_fast` 3.56 · `conv1d_silu_split_f32` 0.71 |
| 6 | Greedy top-k sampling | ~2.02% | `sample_topk_partial` 1.23 · `sample_topk_finalize` 0.79 |
| 7 | WMMA prefill residue (not decode) | 0.84% | `gemm_hfq4g256_moe_grouped_wmma_gfx12` (80 calls, prefill-scale) |

**Read:** classic bandwidth-bound single-token decode. Many small (10–30 µs)
4-bit weight GEMVs dominate; the only WMMA kernel present (0.84%) is prefill
residue. There is **essentially zero decode tensor-core presence** to accelerate
— the compute surface that matters is the 4-bit GEMV family.

---

## (b) Hot 4-bit decode GEMV — % decode time and roofline utilization

**Hot family true decode share: 58.38%** (52.09% excluding the inferred
`multirow_r2` lm_head attribution). The single largest kernel is only **13.09%**
(`fused_qkvza_hfq4g256`) — the mass is spread across ≥7 GEMV variants, so this is
a **family**, not one kernel.

Roofline denominator = committed measured **611.8 GB/s DRAM** (gfx1201.json,
~96% of the R9700's 256-bit GDDR6@20 Gbps theoretical). The 1352.4 GB/s cache
tier (64 MiB Infinity Cache) is **not exploitable for decode weight streaming**:
at batch=1 each 4-bit weight is read exactly once with no reuse, so the working
set is DRAM-resident and the cache tier only benefits the (small) activation/KV
traffic — the weight GEMVs are bound by the 611.8 GB/s DRAM plateau.

Per-token budget: 1.4053 s / 200 = **7.026 ms/token**; hot family =
0.5838 × 7.026 = **4.102 ms/token**.

Achieved bandwidth (estimated; assumes ~3B active params × ~4.25 bits/param at
`hfq4g256`, lm_head = 248320 × 2048 × 4.25 bits):

| Surface | µs/token | Est. weight bytes/token | Achieved BW | % of 611.8 DRAM peak |
|---|---:|---:|---:|---:|
| **lm_head** (`multirow_r2`, ~442 µs, 1 call/tok) | 442 | ~0.27 GB | **~611 GB/s** | **~100%** (DRAM-saturated) |
| Non-lm_head hot GEMVs | 3660 | ~1.59 GB | ~434 GB/s | ~71% |
| **Hot family (blended)** | 4102 | ~1.86 GB | **~453 GB/s** | **~74%** |

**Interpretation (load-bearing for the verdict):** the hot family already runs
at **~74% of DRAM peak**, and lm_head is **at the roofline (~100%)**. dp4a
accelerates the *int8 dot-product compute*, not DRAM reads. On a kernel already
near the DRAM plateau, dp4a delivers ~nothing (lm_head: **0 headroom** —
**exclude from scope**). The non-lm_head GEMVs have ~1.41× headroom *to* the DRAM
roofline (611.8 / 434) — and dp4a can only capture that gap **if those kernels
are currently ALU/dequant-bound below the plateau**, not already DRAM-bound.
Bounding the whole hot family to the DRAM roofline gives a **max hot-family
speedup of ≈1.35×** (non-lm_head → roofline, lm_head unchanged). That is the
physical ceiling on any dp4a decode-GEMV win here.

> Byte counts are estimated from the A3B active-param headline + 4.25 bits/param,
> not measured per-kernel. The utilization band is therefore ±; treat ~71–74% as
> "clearly bandwidth-bound with modest sub-roofline headroom," not an exact
> figure. Confirming the ALU-vs-mem split per kernel (rocprofv3 `--pmc`
> VALU-busy vs mem-busy + `gfx-kernel-metadata` occupancy) is the cheap gate
> before committing Phase B engineering.

---

## (c) Amdahl table — end-to-end decode tok/s vs hot-GEMV speedup

Amdahl on the hot-family fraction **p = 0.5838**. Total speedup
`f(s) = 1 / ((1−p) + p/s)`. Two baselines shown: **kernel-time 142.3 tok/s** (the
% decompose against this — primary) and **wall-anchored 124 tok/s** (real
un-instrumented AR — conservative).

| Hot-GEMV speedup `s` | Total speedup `f(s)` | tok/s @ 142.3 (kernel) | tok/s @ 124 (wall) |
|---:|---:|---:|---:|
| 1.0× (baseline) | 1.00× | 142.3 | 124.0 |
| 1.35× *(roofline ceiling)* | 1.18× | **167.7** | 146.1 |
| 1.5× | 1.242× | 176.7 | 154.0 |
| 2.0× | 1.412× | 201.0 | 175.1 |
| 3.0× | 1.637× | 233.0 | 203.0 |
| ∞ (hot family → 0 time) | 2.403× | **341.9** | 297.9 |

**Minimum hot-GEMV speedup to hit a target:**

| Target | On kernel basis (142.3) | On wall basis (124) |
|---|---|---|
| ~200 tok/s | s ≈ **1.98×** | s ≈ **2.87×** |
| ~300 tok/s | s ≈ **10.0×** | **unreachable** (∞-asymptote = 297.9 < 300) |

The **hard Amdahl ceiling** (infinite hot speedup) is 2.40× → **341.9 tok/s**
(kernel) / **297.9 tok/s** (wall). Both 200-tok/s targets and (especially) 300
demand hot-family speedups **well above the ~1.35× DRAM-roofline ceiling** from
(b). The physically reachable end-to-end ceiling is the s=1.35× row:
**~168 tok/s (kernel) / ~146 tok/s (wall), i.e. +15–18% end-to-end.**

---

## (d) VERDICT — Phase B (dp4a decode GEMV): **QUALIFIED GO, small win only**

**Go/no-go: conditional GO for family-wide, NO-GO for one-kernel or as a
"solve A3B decode" claim.**

The decode surface is genuinely concentrated in one implementation family — the
HFQ4G256 4-bit weight-streaming GEMVs are **58.38%** of decode kernel time — so a
**family-wide** dp4a port (targeting `moe_gate_up`, `moe_down`, `residual`,
`residual_sigmoid_scaled`, `fused_qkvza`, `fused_qkv`) is the right unit of work.
This matches the Codex verdict (`refuted:false`, conf 0.82): GO **only** if
scoped family-wide; NO-GO for a one-kernel experiment marketed as solving decode
(the largest single kernel is 13.09% → a 2× on it alone is only +7% end-to-end).

**But the realistic ceiling is small, and this must be said plainly.** Two
independent caps agree:

1. **Amdahl:** 58.38% hot mass → even *infinite* hot speedup tops out at
   2.40× (341.9 tok/s kernel / 297.9 wall). 200 tok/s needs ~1.98× (kernel) or
   ~2.87× (wall) on the hot family; 300 tok/s needs ~10× or is outright
   unreachable on the wall basis.
2. **Roofline:** the hot family already runs at **~74% of the 611.8 GB/s DRAM
   peak**, lm_head at **~100%**. dp4a cannot exceed DRAM bandwidth by speeding up
   compute. The roofline-bounded max hot-family speedup is **≈1.35×**, which
   lands *below* even the 1.5× row — so the physically reachable end-to-end gain
   is **~+15–18% (≈168 tok/s kernel / ≈146 wall)**, not 200 and never 300.

**Honest bottom line.** Phase B is **not Amdahl-dead** (58% is a real,
worth-attacking surface) but it **is roofline-limited**: this is a
bandwidth-bound decode, and dp4a's lever is compute. The expected payoff is a
**modest ~15–18% end-to-end** win, contingent on the target GEMVs actually being
ALU/dequant-bound below the DRAM plateau. If a cheap pre-check (rocprofv3
`--pmc` VALU-busy vs mem-busy on 2–3 representative hot GEMVs +
`gfx-kernel-metadata` occupancy) shows them already DRAM-saturated like lm_head,
**Phase B should be rejected** — there is no compute headroom to capture.

**Recommendation:** GO to the **one-day dp4a spike on `fused_qkvza_hfq4g256` +
`moe_gate_up` only**, gated behind the ALU-vs-mem pre-check, targeting the
~1.35× roofline ceiling and validated on `./scripts/coherence-gate.sh`. Do
**not** greenlight full family engineering, and do **not** attach any 200/300
tok/s target — the roofline forbids it. Exclude lm_head (`multirow_r2`) from dp4a
scope entirely (already at DRAM peak).

---

## Follow-ups (out of scope here)

1. **MTP auto-engages on A3B `mq4r` despite `--spec` help text.** The default
   (`auto`) run showed a speculative WMMA/MTP path even though the CLI help says
   "A3B → off" (registry `cli/registry.json:171-179` ships an `.mtp` sidecar for
   this tag). File as a CLI-help / spec-auto-default discrepancy.
2. **lm_head attribution is inferred** (`multirow_r2`, 6.29%, ~442 µs, 201 calls,
   vocab 248320) from call count + shape, not a name-labeled log line —
   high-confidence but not ground-truth.
3. **DPM read caveat:** `power_dpm_force_performance_level` returned
   "Device or resource busy" on read (pre-existing box quirk); `auto` was written
   back (exit 0) but unverifiable by read.
