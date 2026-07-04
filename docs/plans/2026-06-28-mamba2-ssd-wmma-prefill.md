# Mamba-2 SSD chunked prefill on WMMA (bf16 inputs / f32 accumulate)

Status: **Phase A done** (f32 chunked GPU floor). **Phase B kernel done + validated
standalone** (bf16-WMMA, all 4 stages, single + multi-chunk). Remaining: serving
wiring + per-group Gram reuse. Branch: `chaingun`. Target: nemotron_h Mamba-2 mixer.

## Why

`mamba2_ssd_seq.hip` is the sequential decode looped over the prompt: one thread
per `(head, head_dim)` channel walking the sequence with an O(N) dot product per
position. It is correct and memory-lean, but it leaves the matrix cores 100%
idle and is O(seq) serial within each thread. The selective-state-duality (SSD)
chunked decomposition turns the intra-chunk recurrence into dense matmuls, which
is exactly the shape WMMA accelerates.

Instruction-calculator throughput (`third_party/amd_matrix_instruction_calculator`):

| arch | instruction | FLOPs/WGP/cycle | notes |
|------|-------------|-----------------|-------|
| gfx1151 (RDNA3.5) | `v_wmma_f32_16x16x16_bf16` | 1024 | f32 accum, cannot co-exec with VALU |
| gfx1201 (RDNA4)   | `v_wmma_f32_16x16x16_bf16` | 2048 | 2× of RDNA3 |

Use **bf16 inputs, f32 accumulate**: the decay weights `exp(S_t − S_s)` span a
large dynamic range, so bf16's f32-exponent is needed (f16 would clip it).
f32-accumulate is free vs f16-accumulate on gfx1103/1151
([[reference_rdna3_wmma_accumulate]]) and preserves cross-K precision.

**Precision bar.** bf16 inputs carry ~0.4% relative error per element, so a
WMMA-bf16 matmul lands around ~1% relative — the f32 paths' 1e-4 absolute bar
does NOT apply. This is consistent with the reference: the nemotron checkpoint is
bf16 and mamba-ssm's chunk scan itself uses bf16 inputs / f32 accumulate. So the
WMMA kernel is validated by **relative / cosine similarity (~1%, cos > 0.999)**
against the f32 chunked floor, with end-to-end nemotron coherence as the real
gate — not the 1e-4 micro-bar.

## Shapes (Nano-4B)

`H=96` heads, `P=80` head_dim, `N=128` state, `G=8` groups (12 heads/group),
model `chunk=256`. GPU chunk tile `L` chosen independently (the decomposition
matches the sequential scan for any `L`).

## Algorithm → matmul mapping

Per chunk of `L` positions, with `S_t = Σ_{r≤t} dt_r·A` (per head, `A=−exp(A_log)`):

```
y_t[p] = exp(S_t)·(C_t·h_in[p])                       # inter-chunk (state read)
       + Σ_{s≤t} exp(S_t−S_s)·dt_s·(C_t·B_s)·x_s[p]   # intra-chunk
       + D·x_t[p]
h_out[p][n] = exp(S_{L-1})·h_in[p][n]
            + Σ_s exp(S_{L-1}−S_s)·dt_s·B_s[n]·x_s[p] # carry
```

Four matmul stages (per group/head per chunk):

1. **CB Gram** `CB[L×L] = C[L×N] · B[L×N]ᵀ` — per **group** (B/C shared across the
   12 heads of a group), reused by all heads. Maps natively to the intrinsic,
   which computes `A@Bᵀ`. K=N=128 → 8 K-tiles of 16.
2. **Intra-chunk** `Y_intra[L×P] = M[L×L] · X[L×P]` where
   `M[t][s] = (s≤t) ? exp(S_t−S_s)·dt_s·CB[t][s] : 0` — per **head** (M is
   per-head: S depends on head). M built in LDS by VALU (the exp/mask), then a
   WMMA matmul against the head's `X[L×P]`.
3. **Inter-chunk** `Y_state[L×P] = Cˢ[L×N] · H_in[N×P]` with
   `Cˢ[t][n] = exp(S_t)·C_t[n]` — per head. K=N=128.
4. **State carry** `H_out[P×N] = coef[P×L] · B[L×N]` with
   `coef[s][p] = exp(S_{L-1}−S_s)·dt_s·x_s[p]`, plus `exp(S_{L-1})·H_in` — per head.

`Y = Y_state + Y_intra + D⊙X`. Chunks are serial through the `H_in→H_out` state
carry; everything inside a chunk is parallel across heads and tiles.

## Tiling / kernel structure

- **L = 16** (one WMMA M/N tile) as the first cut: keeps every stage a single
  16-tile in the L dimension, no intra-L blocking. Revisit L=32/64 later (more
  arithmetic intensity, bigger LDS, the `M` triangular mask spans tiles).
- Block per `(group, chunk)` cooperatively builds the CB Gram + stages B/C tiles
  in LDS; the group's 12 heads then each run stages 2–4. Or block-per-head with
  the Gram recomputed/loaded — decide by LDS budget (CB `L×L` f32 = 1 KB at
  L=16; B/C tiles `L×N` bf16 = 4 KB each).
- Serial chunk loop carrying `H_in` in LDS/global between chunks.
- Lane/register layout per `gemm_f16_wmma.hip`: lane `t&15` owns row `t`; acc
  `float8_t`, `acc[j]` → `D[2j+(t>>4)][t&15]`; intrinsic
  `__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a,b,c)`.

## Portability (AGENTS invariant)

WMMA is RDNA3+ only. RDNA2 (gfx10xx) has no WMMA → keep `mamba2_ssd_seq.hip` as
the generic floor and gate the WMMA kernel behind `arch_caps` (the conv1d path
already does this with gfx1151 overlays). `mamba2_ssd_chunk_f32.hip` (Phase A) is
the f32 chunked reference and the gpu-vs-cpu oracle for the WMMA output.

## Validation

- Phase A floor: `test_ssd_chunk_gpu` — GPU f32 chunked vs `ssd_chunked` (3.7e-9)
  and `ssd_sequence` (≤1.2e-8) across L∈{64,16,8,7}. **DONE.**
- Phase B: same harness, compare WMMA-bf16 output vs `ssd_sequence` at the 1e-4
  bar, then vs `mamba2_ssd_chunk_f32` to isolate WMMA/bf16 error from algorithm
  error. Then wire into `block_gpu.rs` prefill behind the arch gate and run the
  full nemotron prefill parity (`test_model_prefill_gpu`) + coherence.

## Status / files

- **Phase A (done):** `kernels/src/mamba2_ssd_chunk_f32.hip`,
  `Gpu::mamba2_ssd_chunk_f32` in `crates/hipfire-rdna/src/dispatch/mamba2.rs`,
  `kernels::MAMBA2_SSD_CHUNK_SRC`, test `test_ssd_chunk_gpu.rs`. The GPU oracle.
- **Phase B kernel (done):** `kernels/src/mamba2_ssd_chunk_wmma.hip` — all 4
  stages, one wave per head, serial chunk loop, state carry in global. bf16
  inputs / f32 accumulate; LDS-staged transposes (x^T, B^T, coef^T); a
  `__syncthreads()` between stage 3 (reads all of h_in) and stage 4 (overwrites
  it). `Gpu::mamba2_ssd_chunk_wmma`, `kernels::MAMBA2_SSD_CHUNK_WMMA_SRC`, test
  `test_ssd_chunk_wmma_gpu.rs`. Validated vs `ssd_sequence`: cos 1.00000, relL2
  1–3e-3 across tiny/nano shapes, single + multi-chunk (state carry).
- **Serving wiring (done):** `block_gpu.rs` prefill gates the `Fp32` arm onto
  `mamba2_ssd_chunk_wmma` when `arch_caps.has_wmma_w32()` (RDNA3/3.5) and
  head_dim/state_size ≤ 128, else the `ssd_seq_f32` floor (also the RDNA2/RDNA4
  path). Overridable via `HIPFIRE_MAMBA2_WMMA_PREFILL` (FeatureFlags
  `mamba2_wmma_prefill`). Q8-state prefill stays on `ssd_seq_q8`. Validated:
  - test_block_prefill_gpu / test_model_prefill_gpu (synthetic): pass both modes
    (made dual-mode: tight 1e-4/1e-3 floor OR cosine, argmax preserved).
  - **test_model_prefill_hfq_gpu (real Nano-4B mq4, 42 layers):** WMMA on →
    argmax 6993 (' Paris') == decode, cos 0.999999, max|Δlogit| 4.4e-2; floor →
    6.2e-6. Greedy token preserved → coherent.
- **Occupancy (done — 2.3×):** multi-wave per head (`WAVES` wave32/block) + LDS-
  staged `h_in`. stage2/3 p-tiles and stage4 (p,n)-tiles split across waves by
  `warp_id`; `h_in` staged to LDS once/chunk (coalesced) so stage 3 stops doing
  scattered per-element global loads. WAVES sweep @ seq512 (baseline single-wave
  4825µs): W4 4199 / W8 2737 / **W16 2096 (2.3×)** / W32 2543 → WAVES=16 (512
  thr). ~43× over the f32 floor. Correctness cos 1.0; real Nano-4B argmax 6993
  preserved. NB: kernel `#define WAVES` and dispatch blockDim (=32·WAVES) must
  stay in sync (desync = silent missing-warp wrong results).
- **Per-group Gram reuse: SKIPPED (data-driven).** Benched: the kernel is
  memory-bound, Gram is ~8% of *compute*, and block-per-group would drop the grid
  96→8 (worse occupancy). Not worth it.
- **State-I/O amortization (done — 2.5× total).** Instead of literal L>16
  (which grows L² intra-chunk compute + blows LDS), the head's SSM state is held
  **resident in dynamic LDS across the whole sequence** (`extern __shared__ float
  St[P*ND]`, sized by the launch): read global state once at kernel start, carry
  it in LDS through every chunk (stage 3 reads St→bf16 lane-side; stage 4 updates
  St in place, f32), write global state once at the end. Same f32-carry
  arithmetic as the global path → identical numerics (real Nano-4B argmax 6993,
  cos 0.999999). seq512: 2096→1922µs (+9% on multi-wave); **2.5× over the
  original single-wave, ~47× over the f32 floor.** Now bound by the unavoidable
  B/C/x input staging, not state I/O. (Limit: `P*ND*4 + ~21KB static ≤ 64KB`;
  nano P=80,N=128 → 40KB+static fits. Larger dims would need a fallback.)
- **Phase B remaining:**
  1. **RDNA4 (gfx12):** validate/enable the `_w32` path (gated to is_rdna3 only).
  2. q8-state prefill equivalent (later).
  3. (lower value) reduce B/C input re-staging across a group's 12 heads.
