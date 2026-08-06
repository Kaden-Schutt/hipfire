# mfp4-E8 as a graded MoE tier — gfx11 + gfx12 findings

Branch `feat/moe-awq-experts` (local). Qwen3.6-35B-A3B (`qwen3_5_moe`, arch_id 5/6).
Validated on **gfx1201** (hiptrx, RDNA4) and **gfx1100** (k9lin, RDNA3, RX 7900 XTX).

## What this is

`T3-3L-E8` is a per-expert graded recipe: within each of the 40 layers the 256
routed experts are ranked by routing contribution and assigned
**MQ6 (hot, top 20%) / mfp4-E8 (mid, next 30%) / MQ3-Lloyd (cold, bottom 50%)**
(`scripts/gen_tier_map.py`). It is the E8-mid variant of `T3-3L` (MQ4-mid). The
shared expert is **MQ6**; router + scalar gate are Q8.

The question it answers: **is mfp4-E8 (4.25 bpw) a better mid-tier than MQ4
(4.0 bpw) in a graded recipe?** Answer: **yes, by a wide margin on the decode
path.**

## KLD (quality)

`eval_hipfire`, wt2 f32 kldref (`q36a3b-wt2-f32.kldref.bin`), full 32-chunk,
8160 scored tokens. Lower is better.

| recipe (iso-RTN tier-map) | gfx1201 per-token | gfx1201 batched | gfx1100 per-token | gfx1100 batched |
|---|---|---|---|---|
| `T3-3L` **MQ4**-mid | 0.057418 | 0.041435 | — | — |
| `T3-3L-E8` **E8**-mid | **0.039051** | 0.039680 | 0.039051 | **0.038964** |

- **E8-mid vs MQ4-mid: −32.0% KLD per-token, −4.2% batched.** Decode runs the
  per-token path, so **−32%** is the output-quality-relevant number.
- **Cross-arch parity:** gfx1100 per-token == gfx1201 per-token *byte-identical*
  (0.039051); gfx1100 batched (0.038964) ≈ gfx1201 batched (0.039680) — the
  ~1.8% gap is f16-WMMA accumulation ordering between the half16 (gfx11 `_k2`)
  and half8 (gfx12) layouts, not a decode bug.
- **Per-token vs batched** for E8 is path-stable (≈0.039 both). MQ4 is NOT
  (0.0574 per-token vs 0.0414 batched) — a pre-existing mixed-gemv-vs-grouped
  MQ4 precision quirk, unrelated to E8.
- Context: uniform-E8 (GPTQ-v2) = 0.0397; uniform-MQ4 = ~0.10. RTN-E8-mid
  already ≈ uniform-E8-GPTQ → a **GPTQ-E8-mid** would likely beat the GPTQ
  `mq4p` champion (0.0372). [open lever]

Coherence: every run is non-NaN, PPL ~5.43–6.05; teacher-forced KLD over 8160
tokens is the fidelity evidence.

## Perf (speed)

Daemon `bench_prefill` (q8 KV, DPM-warmed, median-of-3). tok/s.

| arch | model | pp512 | pp1024 | pp2048 | decode |
|---|---|---|---|---|---|
| gfx1201 | `T3-3L-MQ4` | 2234 | 2178 | — | 96.8 |
| gfx1201 | `T3-3L-E8` | 2008 | 1952 | — | 96.5 |
| gfx1100 | `T3-3L-E8` (admit_mq6 on) | 1012 | 1002 | 949 | 109.8 |
| gfx1100 | `T3-3L-E8` (admit_mq6 off → per-token fallback) | 105.6 | — | — | 110.0 |

- **E8-mid costs ~10% prefill vs MQ4-mid** (gfx12 2008 vs 2234) for the −32%
  decode-KLD win; **decode is at parity** (96.5 vs 96.8).
- **admit_mq6 is prefill-only.** Decode is **identical** on/off (109.8 vs 110.0)
  — every `admit_mq6` reference is inside `prefill_batch_pbs_eligible`; none in
  `run_moe_decode`. So "unbatched MQ6 dragging decode" = **no**.
- The gfx12-only `admit_mq6` default was silently forcing **~9.6× slower
  prefill** on gfx11 for graded-MQ6 models (1012 → 105.6). Fixed by widening the
  default to gfx11 (see below).
- gfx1100 prefill (1012) is ~half gfx1201 (2008): the `_k2` (half16, K2 unroll)
  kernel is less optimized than `.gfx12` (half8, K4 unroll + k_grp split). See
  Prefill levers.

## The gate / admit-site maze

The batched-prefill decision is a chain of gates. To debug a per-token fallback,
set `HIPFIRE_DEBUG_BATCH=1` and read `[hipfire::batch_eligible] ... all_dtypes_ok=`.
All line numbers in `crates/hipfire-arch-qwen35/src/qwen35.rs` unless noted.

1. **`prefill_batch_pbs_eligible`** (:7791) — TOP gate. Components: `has_dn`,
   `moe_topk_ok`, `moe_router_logits_present`, `all_dtypes_ok`, `n>=MIN_BATCH`.
   Dump via `HIPFIRE_DEBUG_BATCH=1`. Force-off via `HIPFIRE_PREFILL_BATCHED=0`.
2. **`all_dtypes_ok`** (~:7833) — per-layer AND: `is_batchable_la` for every LA
   (DeltaNet) weight + `moe_ffn_batched_admissible` for the MoE FFN.
3. **`is_batchable_la(dt, arch)`** (:7406) — LA-weight batchability. E8 clause
   `e8_with_wmma` (:7519): `MFP4G32E8 && arch∈{gfx1100..1201} && HIPFIRE_E8_GFX12=1`.
4. **`moe_ffn_batched_admissible(ffn, admit_mq6, arch)`** (:7901) — computes
   `admit_e8`, `admit_paro`, then delegates to `_for_dtypes`.
   - **`admit_e8`** (:7911): `arch∈{gfx1100..1201} && HIPFIRE_E8_GFX12=1`.
   - **`admit_mq6`** (:7823) = `mq6_batched_admit_enabled_from_env(`
     `HIPFIRE_MOE_MQ6_ADMIT, arch)` (:7893). **Default: `gfx11*||gfx12*`**
     (was `gfx12*` only — the silent gfx11 prefill regression).
5. **`moe_ffn_batched_admissible_for_dtypes(dtypes, admit_mq6, admit_paro,
   admit_e8)`** (:7700) — the arms (first match wins):
   - `routed_ok = routed_mixed_merged || (expert_gate_up_uniform &&
     expert_down_uniform)` (:7706). `routed_mixed_merged = expert_dtype_tags.
     is_some()` (:7682).
   - **routed_mixed_merged arm** (:7712) — GRADED path (incl. `T3-3L-E8`):
     checks ONLY the **shared** expert (MQ4 always; **MQ6 needs `admit_mq6`**;
     MQ6 is why gfx11 needed the widen). Returns early — never reaches the E8
     arms below.
   - uniform-E8 / Q8-shared arm (:7732); uniform-E8 / {Q8,E8}-shared arm (:7749);
     PARO (:7766); uniform MQ6 (:7776); uniform MQ4 (:7784).
6. **Tag table** (:4940, in `load_moe_ffn`) — per-expert `gpu_dtype → u8` tag:
   `0=MQ6, 1=MQ2L, 2=MQ4, 3=MQ3L, 4=MFP4G32E8`. (E8 was the missing tag → fell
   through `_=>2` (MQ4) → read as MQ4 → NaN. This was the whole bug.)
7. **Per-token dispatch** (`crates/hipfire-dispatch/src/pipeline/mod.rs`):
   - mixed gate_up :446 → `gemv_mixed_moe_gate_up_k8_indexed_batched` (tag 4 →
     `e8_row_partial`).
   - mixed down :518 → `gemv_mixed_moe_down_k8_indexed_batched_expanded` (tag 4).
   - uniform-E8 gate_up :481 / down :559 (bypassed for graded; tags=None only).
8. **Batched grouped dispatch** (`pipeline/mod.rs`:1188) →
   `gemm_mixed_moe_grouped_wmma_{k2 (gfx11), _gfx12}` (tag 4 E8 branch). Uniform-E8
   grouped at :1209.
9. **Path-2 (grouped) vs Path-1 (indexed)** for *uniform* E8
   (`crates/hipfire-dispatch/src/families/moe.rs`:551, `e8_no_grouped`).

### Env flags

| flag | effect |
|---|---|
| `HIPFIRE_E8_GFX12=1` | enable E8 batchability (`is_batchable_la` + `admit_e8`) on gfx11/gfx12 |
| `HIPFIRE_MOE_MQ6_ADMIT=0\|1` | override `admit_mq6` (default `gfx11*\|\|gfx12*`) |
| `HIPFIRE_PREFILL_BATCHED=0` | force per-token (skip batched prefill) |
| `HIPFIRE_DEBUG_BATCH=1` | print the eligibility dump per forward |

### Kernels (`kernels/src/`)

| role | gfx11 | gfx12 |
|---|---|---|
| per-token mixed gate_up (tag 4 = E8) | `gemv_mixed_moe_gate_up_k8_indexed_batched.hip` (generic) | same |
| per-token mixed down (tag 4 = E8) | `gemv_mixed_moe_down_k8_indexed_batched_expanded.hip` (generic) | same |
| batched grouped mixed (tag 4 = E8) | `gemm_mixed_moe_grouped_wmma_k2.hip` (half16) | `gemm_mixed_moe_grouped_wmma.gfx12.hip` (half8) |
| uniform-E8 decode source | `gemv_mfp4g32_e8_moe_*_{dgpu,gfx1151}.hip` | (gfx11 twins) |
| uniform-E8 grouped source | `gemm_mfp4g32_e8_moe_grouped_wmma.gfx1151.hip` | `.gfx12.hip` |

E8 decode (byte-identical everywhere): `value = row_scale × cvt_e4m3(block_scale)
× 0.88 × e8_decode(codeword)[i]`. Row layout: 16-byte header (fp16 row_scale) +
`(K/32)×17`-byte blocks; one 256-group = 8 blocks = 136 B. Rotation is the same
FwhtG256 (`signs1/signs2`) as MQ — E8's group_size 32 is only the codeword block,
NOT the rotation, so a graded model rotates once and both MQ and E8 tiers decode
correctly.

## Prefill levers (load-bound; roofline-grounded)

The grouped-WMMA MoE prefill kernels are **load-bound, not dequant-bound** (see
`memory/project_qwen35_prefill_decode_roofline_2026_06_08` and
`project_fp8_gfx12_e8_grouped_prefill_roofline_2026_06_14`). They run `LDS: 0`,
so each block re-reads its X tile from L2. Levers, biggest first:

1. **LDS-tiling (≈10×, precision-free, #1 lever).** Stage X (and/or weight) tiles
   into LDS so the inner WMMA loop reuses them instead of re-reading L2. Raises
   arithmetic intensity past the roofline ridge. Applies to all the grouped
   kernels (mixed + uniform) on both arches. This is the dominant headroom.
   **gfx11 UPDATE (2026-06-15): empirically WASHES** — a 4-warp LDS-X variant
   (`gemm_mixed_moe_grouped_wmma_4w_k2`, commit 5a95df0a, env
   `HIPFIRE_MOE_GROUPED_4W`) is byte-parity-correct but gives no prefill gain on
   gfx1100 (pp512 1019 vs 1023). gfx11's 96 MB Infinity Cache already absorbs
   the X re-reads the lever targets, so the gfx12 load-bound regime does NOT
   transfer down. See Open levers. gfx12 (where the roofline measured the
   bottleneck) is the real test and remains untested.
2. **fp8 WMMA (gfx12/RDNA4 only, conditional 2nd ~2× ON TOP of LDS-tiling).**
   RDNA4 fp8 WMMA measured 2.01× fp16 peak (393 vs 196 TFLOPS). Only pays off
   once the kernel is compute-bound (i.e. after LDS-tiling); costs fp8-X
   (activation) precision.
3. **gfx11 `_k2` retune (close the ~2× gfx11-vs-gfx12 gap).** gfx1100 pp512 1012
   vs gfx1201 2008. The `_k2` kernel uses half16 + K2 unroll; the `.gfx12`
   kernel uses half8 + K4 unroll + a k_grp K-split. Port the K4/occupancy
   structure (and the unwired bigger-row-tile direction) to `_k2`.
4. **Bigger row-tiles per block (cut X re-reads).** Process more output rows per
   block so each X tile load amortizes over more MACs (the dense-model lever,
   `_2tile`/`_ldscoop` directions).

## Open levers

- **GPTQ-E8-mid** — the tier-map E8 arm is RTN (`quantize_mfp4g32_e8_2d`); a
  GPTQ pass (`quantize_mfp4g32_e8_gptq_2d`, needs Hessians) on the mid tier
  likely takes T3-3L-E8 below the GPTQ `mq4p` champion (0.0372).
- **LDS-tiling** the grouped prefill kernels (above). gfx11 = **WASH** (done,
  commit 5a95df0a; Infinity Cache absorbs the X re-reads). gfx12 = the real
  test, **untested** — roofline measured load-bound there (gfx1201 ~16% peak),
  but RDNA4's ~64 MB IC could mask it too.
- **gfx11 `_k2` retune** (above) — close the ~2× gfx11-vs-gfx12 prefill gap via
  the K4/occupancy structure + k_grp split. With LDS-tiling washing on gfx11,
  this is now the leading gfx11 prefill lever.

## Reproduce

KLD (per-token = decode-path quality; prefill = batched grouped path):
```
HIPFIRE_E8_GFX12=1 [HIPFIRE_MOE_MQ6_ADMIT=1 on gfx11 if pre-widen] \
  eval_hipfire --model q36a3b.rq-T3-3L-E8.hfq \
  --ref q36a3b-wt2-f32.kldref.bin --output /tmp/x.kldseq \
  --scoring-mode {per-token|prefill}
```
Perf (daemon): `bench_prefill.py <model> 4096 512,1024,2048 64` with
`DAEMON_BIN=…/examples/daemon HIPFIRE_E8_GFX12=1`. (No `bun` on hiptrx → use the
Python driver, not the production `scripts/serve_harness.py` route.) Pitfall: `pkill -f examples/daemon`
self-kills any shell whose cmdline contains `DAEMON_BIN=…examples/daemon`.

Build pitfall: editing a `.hip` needs `touch crates/rdna-compute/src/kernels.rs`
to force the `include_str!` rebuild; verify embedding with
`grep -a e8_row_partial target/release/examples/eval_hipfire` (plain `grep` hits
binary-mode — use `-a`).
