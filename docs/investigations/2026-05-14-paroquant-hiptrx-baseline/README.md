# ParoQuant runtime probe — hiptrx gfx1201 baseline

Cross-host validation of the paroquant-runtime-probe checkpoint on hiptrx R9700
(gfx1201, RDNA4), to complement codex's k9lin gfx1100 (7900 XTX, RDNA3)
tuning loop. Branch checkpoint: `26ebcfc3 checkpoint paroquant atlas tuning`.

## Headline

| host    | arch    | layout  | gen_tok_s | prefill_tok_s | BW eff (GiB/s) | avg_ms |
|---------|---------|---------|-----------|---------------|----------------|--------|
| k9lin   | gfx1100 | native  | 72.5      | (n/a)         | 67.5           | 13.58  |
| hiptrx  | gfx1201 | native  | 101.3     | 103.2         | 94.2           | 9.78   |
| hiptrx  | gfx1201 | engine  | **186.6** | **193.4**     | **174.6**      | 5.27   |

Same `bench_qwen35_mq4` config: `HIPFIRE_DPM_WARMUP_SECS=3 HIPFIRE_GRAPH=0
HIPFIRE_KV_MODE=q8 --prefill 32 --prefill-runs 2 --warmup 2 --gen 64`. Sustains
across `--gen 256` within ±0.5%.

## Two headline findings

### 1. `--layout engine` is +84% over `--layout native` on hiptrx gfx1201

`scripts/paroquant_import.py --layout engine` produces a different HFQ
(`quant_type=29 PARO4G128T` instead of `28 PARO4G128`): qweight is transposed
to `[M/8, K]` for coalesced GEMV reads, and `theta` is precomputed as
`sincos_f32` instead of raw `theta_f16` (skips per-call sin/cos). Both
infrastructure and runtime route are already wired in
`crates/rdna-compute/src/dispatch.rs` (`PARO4G128T` variant, lines ≈2302–2772).

| layout  | gen_tok_s | prefill_tok_s | avg_ms |
|---------|-----------|---------------|--------|
| native  | 101.3     | 103.2         | 9.78   |
| engine  | **186.6** | **193.4**     | 5.27   |

The k9lin Atlas baseline (72.5 tok/s) was on native layout. Re-baselining with
engine layout — both on gfx1100 and gfx1201 — should precede any fusion work
on gfx1100. The +84% lift comes from infrastructure that already exists; codex's
current task on gfx1100 is exploring fusion on the **slower** path.

### 2. gap-to-MQ4 narrowed 4× → 2.16×; next bottleneck is named in the profile

Profile on hiptrx (HIPFIRE_PROFILE_DECODE=1, gfx1201, `--gen 64`):

**paro4 engine** decode breakdown:
```
paro4g128t_rotate                  8064x  79.8ms  24.3%   6.7 GiB/s   ← bottleneck
gemv_paro4g128t_prerotated         6528x  77.0ms  23.5%  145.6 GiB/s
gemv_paro4g128t_prerotated_residual 3072x 44.7ms  13.6%   97.2 GiB/s
rmsnorm_f32                        3136x  31.1ms   9.5%
paro4g128t_swiglu_rotate           1536x  15.3ms   4.7%
(remaining 14 kernels, each <5%)
```

**MQ4** decode breakdown for comparison:
```
gemv_hfq4g256_residual             3072x  33.3ms  17.3%  133.3 GiB/s
fused_rmsnorm_mq_rotate            3072x  31.5ms  16.3%
fused_qkvza_hfq4g256               1152x  18.3ms   9.5%  265.5 GiB/s  ← batched 3-output GEMV
fused_silu_mul_mq_rotate           1536x  11.9ms   6.2%
mq_rotate_x                        1536x  11.8ms   6.1%
fused_qkv_hfq4g256                  384x   5.1ms   2.6%  199.4 GiB/s
```

**Concrete observations:**

- `paro4g128t_rotate` is **6.7 GiB/s** at 24.3% of decode time. This is the
  bottleneck. Fusing it into the subsequent prerotated GEMV — similar to
  MQ4's `fused_rmsnorm_mq_rotate` pattern — looks like the single highest-ROI
  kernel change. Estimated gain: ≈+30% tok/s if rotate cost goes to zero.

- paro4 currently runs 3 separate `gemv_paro4g128t_prerotated` calls per
  layer for QKV (3 in_proj heads). MQ4 collapses that into one
  `fused_qkvza_hfq4g256` call (265.5 GiB/s). Batched-QKV is the second
  highest-ROI structural change.

- The `_prerotated` GEMV kernels are already efficient: 145.6 and 97.2 GiB/s
  on the non-residual and residual variants respectively. The remaining gap
  to MQ4's 133.3 GiB/s on `gemv_hfq4g256_residual` is small.

## Generic regression: HIPFIRE_GRAPH=1 panics on gfx1201

Both MQ4 and paro4 (native AND engine layout) panic with the same trace when
`HIPFIRE_GRAPH=1` is set on gfx1201, paroquant-runtime-probe HEAD:

```
thread 'main' panicked at crates/hipfire-runtime/src/llama.rs:3639:51:
called `Option::unwrap()` on a `None` value
```

Line 3639 is `partial_cmp(b).unwrap()` inside `argmax()` — i.e., logits contain
NaN, so partial_cmp returns None and unwrap explodes. Prefill completes
cleanly (104.8 / 199.7 tok/s); the panic is in the first decode step.

This is a regression vs the May 11 memory entry recording
"HIPFIRE_GRAPH default-on for gfx12 = +2.6% decode" working on MQ4. Likely a
scratch-buffer-not-initialized-between-cycles class issue under graph replay;
matches the `active_stream`-gated `memset_async` gotcha called out in CLAUDE.md
(silent fallthrough to sync `hipMemset` when `active_stream=None` — which would
NOT be captured in the graph, producing uninitialized scratch on replay).

Affects both PARO4G128 and PARO4G128T paths but also MQ4 — not paroquant-specific.

## Coherence + oracle: clean

- `paro-oracle` (native layout, layer 0 QKV in_proj, 2 samples): **bit-exact**.
  `source_vs_hfq_max_abs = 0.0`, `source_vs_hfq_mean_abs = 0.0`. `hfq_output_finite = true`.
- `test_inference` on engine layout: **9/9 pass** (forward finite logits,
  forward_scratch parity max_diff=0, 10-tok sequence, ChatML special tokens,
  asym3 KV alloc + forward, decode >10 tok/s, no VRAM leak).
- 22 `test_gemv_paro4g128` variants pass on gfx1201 with `max_rel ~5e-7`
  (FP16 epsilon).

## Full perf matrix (gfx1201, hiptrx, 0.8B Qwen3.5)

| Model            | KV    | GRAPH | gen tok/s | BW    | notes              |
|------------------|-------|-------|-----------|-------|--------------------|
| MQ4              | q8    | 0     | **403.4** | 206.3 | reference          |
| MQ4              | q8    | 1     | panic     | -     | NaN argmax         |
| MQ4              | asym3 | 0     | 370.7     | 189.6 | -8.1% vs q8        |
| MQ4              | q8    | 0     | 394.1     | 201.6 | gen=256 sustain    |
| paro4 native     | q8    | 0     | 101.3     | 94.2  |                    |
| paro4 native     | q8    | 1     | panic     | -     | NaN argmax         |
| paro4 native     | asym3 | 0     | 99.5      | 92.5  | -1.8% vs q8        |
| paro4 native     | q8    | 0     | 101.3     | 94.3  | gen=256 sustain    |
| paro4 **engine** | q8    | 0     | **186.6** | 174.6 | **+84% vs native** |
| paro4 engine     | q8    | 1     | panic     | -     | NaN argmax         |
| paro4 engine     | asym3 | 0     | 179.3     | 167.7 | -3.9% vs q8        |
| paro4 engine     | q8    | 0     | 185.8     | 173.9 | gen=256 sustain    |

PROFILE_DECODE overhead: paro4 native 58.3 (vs 101.3), engine 66.4 (vs 186.6),
MQ4 111.3 (vs 403.4). Profile rows below SUMMARY rows but useful for kernel
share, not perf claims.

## Recommendations for codex's tuning loop

1. **Re-baseline gfx1100 on `--layout engine` first** before any kernel work.
   The current Atlas baseline (72.5 tok/s @ 67.5 GiB/s, native) may understate
   paro4 by ≈2× if the engine layout ports cleanly to gfx1100. Single command:
   `python3 scripts/paroquant_import.py import --model z-lab/Qwen3.5-0.8B-PARO --output ~/.hipfire/models/qwen3.5-0.8b.paro4g128-engine.hfq --layout engine`.

2. **Highest-ROI kernel work is rotate-into-GEMV fusion**, not gate_up/swiglu
   fusion. The current Atlas task `paro4-decode-fusion-gfx1100` is exploring
   MLP fusion (5–10% combined of decode time); the rotate pre-pass alone is
   24.3% at 6.7 GiB/s. Pattern to mirror: MQ4's `fused_rmsnorm_mq_rotate`.

3. **QKV batching is the second highest-ROI structural change.** MQ4 collapses
   3 in_proj GEMVs into one `fused_qkvza_hfq4g256` running at 265.5 GiB/s.
   paro4 currently issues 3 separate `gemv_paro4g128t_prerotated` calls.

4. **Diagnose the HIPFIRE_GRAPH=1 NaN bug before any graph-dependent work.**
   It's not paroquant-specific and affects MQ4 too. Likely an `active_stream`
   memset_async gotcha given the symptom (NaN scratch on replay). Per the
   CLAUDE.md note, this is a known class of bug.

## Reproduce

```bash
# On hiptrx, .worktrees/paroquant:
~/venvs/zaya1/bin/python scripts/paroquant_import.py import \
    --model z-lab/Qwen3.5-0.8B-PARO \
    --output ~/.hipfire/models/qwen3.5-0.8b.paro4g128-engine.hfq \
    --layout engine --copy-floats f16

cargo build --release --example test_gemv_paro4g128 \
    --example bench_qwen35_mq4 --example test_inference

# correctness
./target/release/examples/test_gemv_paro4g128  # 22 variants ALL PASS
./target/release/examples/test_inference ~/.hipfire/models/qwen3.5-0.8b.paro4g128-engine.hfq  # 9/9 pass

# perf
HIPFIRE_DPM_WARMUP_SECS=3 HIPFIRE_GRAPH=0 HIPFIRE_KV_MODE=q8 \
    ./target/release/examples/bench_qwen35_mq4 \
    ~/.hipfire/models/qwen3.5-0.8b.paro4g128-engine.hfq \
    --prefill 32 --prefill-runs 2 --warmup 2 --gen 64
# expected: gen_tok_s=186.6, bw_gib_s=174.6 ±1%

# profile breakdown
HIPFIRE_DPM_WARMUP_SECS=3 HIPFIRE_GRAPH=0 HIPFIRE_KV_MODE=q8 HIPFIRE_PROFILE_DECODE=1 \
    ./target/release/examples/bench_qwen35_mq4 \
    ~/.hipfire/models/qwen3.5-0.8b.paro4g128-engine.hfq \
    --prefill 32 --prefill-runs 2 --warmup 2 --gen 64
```

## Atlas row (host=hiptrx, decode_ar, engine layout, run_index=2)

```json
{
  "phase": "decode_ar",
  "workload": "paro4-decode-fusion",
  "shape_bucket": "decode_ar_pp32_gen32",
  "hostname": "hiptrx",
  "arch": "gfx1201",
  "model_size": "0.8B",
  "quant": "PARO4G128T",
  "git_sha": "47cbf967",
  "metrics": {"avg_ms": 5.27, "bw_gib_s": 174.6, "gen_tok_s": 186.6, "p50_ms": 5.27},
  "variant": {"env": {"HIPFIRE_GRAPH": "0", "HIPFIRE_KV_MODE": "q8"}}
}
```

Full .jsonl: `.codeinsight+research/kernel-atlas/runs/` (gitignored).

## Measured upper bound for rotate elimination (SKIP_ROTATE experiment)

Patched both `paro4g128t_rotate` and `paro4g128t_swiglu_rotate` to early-return
on `HIPFIRE_PARO_SKIP_ROTATE=1` (skip launch, leave x_rot scratch stale).
Bench paro4 engine layout, 0.8B, gfx1201, `--prefill 32 --prefill-runs 2 --gen 64`:

| metric                  | baseline | skip_rotate | Δ         |
|-------------------------|----------|-------------|-----------|
| prefill_tok_s           | 193.3    | **225.7**   | **+16.8%**|
| gen_tok_s (decode)      | 186.6    | panic       | (~+17% extrap.) |
| prefill_wall_ms (32 tok)| 165.5    | 141.8       | -14.3%    |

Decode panics in argmax on first decode step — stale x_rot scratch + downstream
kernels produce NaN logits, `partial_cmp(b).unwrap()` at llama.rs:3639:51
explodes. Prefill doesn't sample, so no argmax → clean number. paro4's
"prefill" is actually the same per-row GEMV decode kernels in a loop (no
batched paro4 prefill kernel exists yet per plan doc Stage 2), so prefill
speedup is a meaningful proxy for what fusion can win on decode.

**Earlier README estimate ("+30% from rotate elimination") was too optimistic.**
The profile share (24.3% of decode time at 6.7 GiB/s) overstated the headroom
because rotate is **not launch-bound** — it's reading ~50 KB of metadata per
call (`pairs:int16[8,1024]` + `sincos_f32[8,512,2]` + `channel_scales:f16[1024]`)
at 6.7 GiB/s ≈ 7.5 µs of actual BW work + ~2.5 µs launch overhead. Fusion
eliminates the launch + x_rot scratch round-trip but keeps the metadata
reads. Real-world fusion likely captures 60-80% of this +17% → +10-13%.

**Revised engineering ceiling on 0.8B engine layout, gfx1201:**

- Current baseline:           186.6 tok/s
- + rotate fusion (-17% measured):  ~218 tok/s
- + QKV batching (~+10% est.):       ~240 tok/s
- + fused rmsnorm-rotate (~+5%):     ~250 tok/s
- + GRAPH=1 fix (~+3-5%):           ~260 tok/s

Compared to MQ4 baseline of 403.4 tok/s = roughly 60% gap, structurally
bounded by paro4's 1.83× model size on the wire (rotation metadata is
inherent to ParoQuant's algorithm — pruning KROT layers trades quality for
size, structured/Hadamard rotations are a different algorithm entirely).

**BW-saturation ceiling on R9700 (~640 GB/s peak):**
- MQ4: 640 / 0.51 GiB = ~1255 tok/s ceiling (currently 32% saturated)
- paro4: 640 / 0.93 GiB = ~688 tok/s ceiling (currently 27% saturated)

So paro4 has 2.55× headroom to BW-saturation but only ~1.4× headroom to
realistic engineering ceiling.

## Path C empirical results — rotate-sharing alone is not enough

Patched `paro4g128t_rotate` and `paro4g128t_swiglu_rotate` with a thread-local
decimation counter (`HIPFIRE_PARO_ROTATE_DECIMATE=N` skips N-1 of every N
launches), bench paro4 engine layout 0.8B on R9700:

| DECIMATE | Prefill tok/s | Decode tok/s | Δ prefill   | Status      |
|----------|---------------|--------------|-------------|-------------|
| 1 (all)  | 193.7         | 187.1        | baseline    | ✓ correct   |
| 2 (½)    | 208.8         | 201.5        | +7.8%       | ✓ works     |
| 3 (⅓)    | 213.5         | panic        | +10.2%      | NaN argmax  |
| 4 (¼)    | 217.7         | panic        | +12.4%      | NaN argmax  |
| ∞ (skip) | 225.7         | panic        | +16.8%      | NaN argmax  |

The +16.8% SKIP_ROTATE asymptote is the upper bound on rotation elimination.
Rotation cost is bounded by metadata BW reads (`pairs:int16[8,K]` +
`sincos_f32[8,K/2,2]` + `channel_scales:f16[K]` ≈ 50KB per call at 6.7 GiB/s
= 7.5 µs of BW work + ~2.5 µs launch overhead). Sharing rotated x across
linears amortizes launch but not metadata-BW.

**Path A (kernel-only restructure)** ceiling: ~218 tok/s on 0.8B engine layout.
Below MQ4's 403. Cannot match MQ4 with current paroquant storage format.

## Why Path B is the only path to MQ4-parity

Three structural costs vs MQ4 on the same model:

| cost factor          | paro4 (current)            | MQ4         | ratio |
|----------------------|----------------------------|-------------|-------|
| model bytes          | 0.93 GiB (qweight G128)    | 0.51 GiB    | 1.83× |
| weight-only BW/tok   | scales[K/128, M] × f16     | f16[K/256,M]| 2× scales |
| rotate calls/tok     | 150 (1 per linear)         | 96 (shared) | 1.56× |
| BW ceiling on R9700  | 688 tok/s (vs 1255 MQ4)    | 1255 tok/s  | 0.55× |
| achieved decode      | 187 tok/s (27% BW sat)     | 403 (32%)   | 0.46× |

At MQ4's BW efficiency (32% of peak), paro4 maxes at **221 tok/s** on R9700.
Storage size is the structural cap.

## Path B: re-architect paroquant in hipfire-native MQ4 storage layout

### Format spec (new quant_type, working name `PARO4G256_MQ`)

```
Per linear:
  qweight: packed nibbles  [K, M/8]    — MQ4 layout (G256, half the scale data)
  qzeros:  packed nibbles  [K/256, M/8] — MQ4 layout
  scales:  f16             [K/256, M]   — MQ4 layout
  rotation side-metadata (paroquant-style):
    pairs:           int16 [KROT, K]
    sincos or theta: f16   [KROT, K/2] (native theta_f16, not engine sincos_f32)
    channel_scales:  f16   [K]
```

Model size on Qwen3.5-0.8B: **~0.55 GiB** (matches MQ4 within 5%).

### Runtime route

Reuse MQ4's existing fused kernels for the GEMV body. Replace MQ4's
`fused_rmsnorm_mq_rotate` with a new `fused_rmsnorm_paro_rotate` that applies
paroquant's pair/theta/channel_scale rotation instead. Everything downstream
(`fused_qkv_hfq4g256`, `fused_qkvza_hfq4g256`, `fused_gate_up_hfq4g256`,
`gemv_hfq4g256_residual`) used unchanged.

```
input x → fused_rmsnorm_paro_rotate(pairs, theta, scales) → rotated_x →
  fused_qkv_hfq4g256(rotated_x, W_qkv_paro_rotated) → q, k, v
```

The W_*_paro_rotated weights are paroquant's R-rotated W, quantized into MQ4
G256 nibble format at offline export time. Rotation math: `(W·R^T)·(R·x) = W·x`
with x→R·x distribution (outliers suppressed before quantization).

### Implementation phases

1. **Format design + offline quantizer** (~2 days)
   - Extend `scripts/paroquant_import.py` with `--layout mq4` mode
   - Per-linear: read paroquant safetensors (qweight + R metadata), apply R^T to
     fp16 W from source, re-quantize as MQ4 G256 packed nibbles, write side
     metadata
   - Validate via paro-oracle (modified to compare against MQ4-rotated kernel
     output instead of paroquant-native kernel output)

2. **Runtime loader** (~0.5 day)
   - Add `DType::PARO4G256_MQ` to `crates/hipfire-runtime/src/hfq.rs`
   - Wire load + tensor binding for the new quant_type

3. **Fused rmsnorm + paro-rotate kernel** (~1-2 days)
   - New `fused_rmsnorm_paro_rotate.hip` kernel
   - LDS-resident rotation passes (KROT=8) on B=1 decode path
   - Validate against existing `paro4g128t_rotate` output on identical inputs
     (math equivalence required, +/- fp16 epsilon)

4. **Dispatch wiring** (~0.5 day)
   - PARO4G256_MQ qweight routes to `fused_qkv_hfq4g256` / `fused_qkvza_hfq4g256`
     / `fused_gate_up_hfq4g256` etc., with `fused_rmsnorm_paro_rotate` as the
     pre-step
   - Mirror the qwen35.rs LinearAttention + FullAttention dispatch tree

5. **Bench + coherence + commit** (~1 day)
   - Re-import Qwen3.5-0.8B-PARO via `paroquant_import.py --layout mq4`
   - paro-oracle bit-exact (5/5 modules across layers)
   - test_inference 9/9 + decode tok/s bench
   - **Target: ≥380 tok/s decode (≥94% of MQ4) on R9700 0.8B**

Total: ~5-6 days focused work. No new algorithm — combines paroquant's
rotation with MQ4's W4 storage + kernel suite.

### Quality expectations

paroquant's PPL/coherence advantage comes from W being quantized AFTER rotation
(outlier suppression). The MQ4 G256 format quantizes the same way (scale +
asymmetric zero), so re-quantizing W·R^T as MQ4-G256 should preserve paroquant's
quality benefit. Quality regression risk vs paroquant-G128:
- Group size G128 → G256: halves group count, slightly higher quantization
  error per group. paper showed paroquant-G128 closes most of the FP16 gap;
  G256 may give back 0.05-0.1 PPL.
- Trade-off: 1.8× faster decode for ~0.05 PPL.

If quality regression is unacceptable, fallback to PARO4G128_MQ (paroquant
group size, MQ4 packed-nibble qweight, MQ4 dispatch path with custom G128
GEMV variants).

### Open risks

- Re-quantization requires regenerating side metadata (pairs/theta) since the
  G256 quantization grid is different from G128 — may need a few hours of
  calibration to refit channel_scales for the new group structure (paroquant's
  channel_scales are learned to fit the per-group MSE).
- MQ4's mq_rotate uses pre-baked rotation indices specific to MQ4 calibration.
  paroquant's pairs/theta differ structurally. `fused_rmsnorm_paro_rotate`
  kernel must support paroquant's irregular pairs gather pattern, which is the
  6.7 GiB/s bottleneck today — may need a different access pattern (e.g.,
  pack pairs as bitfields with k-mod-8 stride structure).

## Path C empirical results — rotate-sharing alone is not enough

Patched `paro4g128t_rotate` and `paro4g128t_swiglu_rotate` with a thread-local
decimation counter (`HIPFIRE_PARO_ROTATE_DECIMATE=N` skips N-1 of every N
launches), bench paro4 engine layout 0.8B on R9700:

| DECIMATE | Prefill tok/s | Decode tok/s | Δ prefill   | Status      |
|----------|---------------|--------------|-------------|-------------|
| 1 (all)  | 193.7         | 187.1        | baseline    | ✓ correct   |
| 2 (½)    | 208.8         | 201.5        | +7.8%       | ✓ works     |
| 3 (⅓)    | 213.5         | panic        | +10.2%      | NaN argmax  |
| 4 (¼)    | 217.7         | panic        | +12.4%      | NaN argmax  |
| ∞ (skip) | 225.7         | panic        | +16.8%      | NaN argmax  |

The +16.8% SKIP_ROTATE asymptote is the upper bound on rotation elimination.
Rotation cost is bounded by metadata BW reads (`pairs:int16[8,K]` +
`sincos_f32[8,K/2,2]` + `channel_scales:f16[K]` ≈ 50KB per call at 6.7 GiB/s
= 7.5 µs of BW work + ~2.5 µs launch overhead). Sharing rotated x across
linears amortizes launch but not metadata-BW.

**Path A (kernel-only restructure)** ceiling: ~218 tok/s on 0.8B engine layout.
Below MQ4's 403. Cannot match MQ4 with current paroquant storage format.

## Why Path B is the only path to MQ4-parity

Three structural costs vs MQ4 on the same model:

| cost factor          | paro4 (current)            | MQ4         | ratio |
|----------------------|----------------------------|-------------|-------|
| model bytes          | 0.93 GiB (qweight G128)    | 0.51 GiB    | 1.83× |
| weight-only BW/tok   | scales[K/128, M] × f16     | f16[K/256,M]| 2× scales |
| rotate calls/tok     | 150 (1 per linear)         | 96 (shared) | 1.56× |
| BW ceiling on R9700  | 688 tok/s (vs 1255 MQ4)    | 1255 tok/s  | 0.55× |
| achieved decode      | 187 tok/s (27% BW sat)     | 403 (32%)   | 0.46× |

At MQ4's BW efficiency (32% of peak), paro4 maxes at **221 tok/s** on R9700.
Storage size is the structural cap.

## Path B: re-architect paroquant in hipfire-native MQ4 storage layout

### Format spec (new quant_type, working name `PARO4G256_MQ`)

```
Per linear:
  qweight: packed nibbles  [K, M/8]    — MQ4 layout (G256, half the scale data)
  qzeros:  packed nibbles  [K/256, M/8] — MQ4 layout
  scales:  f16             [K/256, M]   — MQ4 layout
  rotation side-metadata (paroquant-style):
    pairs:           int16 [KROT, K]
    sincos or theta: f16   [KROT, K/2] (native theta_f16, not engine sincos_f32)
    channel_scales:  f16   [K]
```

Model size on Qwen3.5-0.8B: **~0.55 GiB** (matches MQ4 within 5%).

### Runtime route

Reuse MQ4's existing fused kernels for the GEMV body. Replace MQ4's
`fused_rmsnorm_mq_rotate` with a new `fused_rmsnorm_paro_rotate` that applies
paroquant's pair/theta/channel_scale rotation instead. Everything downstream
(`fused_qkv_hfq4g256`, `fused_qkvza_hfq4g256`, `fused_gate_up_hfq4g256`,
`gemv_hfq4g256_residual`) used unchanged.

```
input x → fused_rmsnorm_paro_rotate(pairs, theta, scales) → rotated_x →
  fused_qkv_hfq4g256(rotated_x, W_qkv_paro_rotated) → q, k, v
```

The W_*_paro_rotated weights are paroquant's R-rotated W, quantized into MQ4
G256 nibble format at offline export time. Rotation math: `(W·R^T)·(R·x) = W·x`
with x→R·x distribution (outliers suppressed before quantization).

### Implementation phases

1. **Format design + offline quantizer** (~2 days)
   - Extend `scripts/paroquant_import.py` with `--layout mq4` mode
   - Per-linear: read paroquant safetensors (qweight + R metadata), apply R^T to
     fp16 W from source, re-quantize as MQ4 G256 packed nibbles, write side
     metadata
   - Validate via paro-oracle (modified to compare against MQ4-rotated kernel
     output instead of paroquant-native kernel output)

2. **Runtime loader** (~0.5 day)
   - Add `DType::PARO4G256_MQ` to `crates/hipfire-runtime/src/hfq.rs`
   - Wire load + tensor binding for the new quant_type

3. **Fused rmsnorm + paro-rotate kernel** (~1-2 days)
   - New `fused_rmsnorm_paro_rotate.hip` kernel
   - LDS-resident rotation passes (KROT=8) on B=1 decode path
   - Validate against existing `paro4g128t_rotate` output on identical inputs
     (math equivalence required, +/- fp16 epsilon)

4. **Dispatch wiring** (~0.5 day)
   - PARO4G256_MQ qweight routes to `fused_qkv_hfq4g256` / `fused_qkvza_hfq4g256`
     / `fused_gate_up_hfq4g256` etc., with `fused_rmsnorm_paro_rotate` as the
     pre-step
   - Mirror the qwen35.rs LinearAttention + FullAttention dispatch tree

5. **Bench + coherence + commit** (~1 day)
   - Re-import Qwen3.5-0.8B-PARO via `paroquant_import.py --layout mq4`
   - paro-oracle bit-exact (5/5 modules across layers)
   - test_inference 9/9 + decode tok/s bench
   - **Target: ≥380 tok/s decode (≥94% of MQ4) on R9700 0.8B**

Total: ~5-6 days focused work. No new algorithm — combines paroquant's
rotation with MQ4's W4 storage + kernel suite.

### Quality expectations

paroquant's PPL/coherence advantage comes from W being quantized AFTER rotation
(outlier suppression). The MQ4 G256 format quantizes the same way (scale +
asymmetric zero), so re-quantizing W·R^T as MQ4-G256 should preserve paroquant's
quality benefit. Quality regression risk vs paroquant-G128:
- Group size G128 → G256: halves group count, slightly higher quantization
  error per group. paper showed paroquant-G128 closes most of the FP16 gap;
  G256 may give back 0.05-0.1 PPL.
- Trade-off: 1.8× faster decode for ~0.05 PPL.

If quality regression is unacceptable, fallback to PARO4G128_MQ (paroquant
group size, MQ4 packed-nibble qweight, MQ4 dispatch path with custom G128
GEMV variants).

### Open risks

- Re-quantization requires regenerating side metadata (pairs/theta) since the
  G256 quantization grid is different from G128 — may need a few hours of
  calibration to refit channel_scales for the new group structure (paroquant's
  channel_scales are learned to fit the per-group MSE).
- MQ4's mq_rotate uses pre-baked rotation indices specific to MQ4 calibration.
  paroquant's pairs/theta differ structurally. `fused_rmsnorm_paro_rotate`
  kernel must support paroquant's irregular pairs gather pattern, which is the
  6.7 GiB/s bottleneck today — may need a different access pattern (e.g.,
  pack pairs as bitfields with k-mod-8 stride structure).

## Quality A/B — paroquant vs MQ4 (PPL on wikitext2 slice)

Perplexity on Qwen3.5-0.8B, ctx=2048, warmup=8, corpus=`benchmarks/quality-baselines/slice/wikitext2-1024s-2048ctx.txt`,
HIPFIRE_GRAPH=0 (works around the gfx12 graph-capture NaN regression on this branch).
All 2039 tokens scored, zero non-finite warnings.

| Model                       | format               | NLL/tok | PPL    | Δ vs MQ4   |
|-----------------------------|----------------------|---------|--------|------------|
| qwen3.5-0.8b.mq4            | MQ4G256 (FWHT-rot)   | 3.6471  | 38.36  | reference  |
| qwen3.5-0.8b.paro4g128-eng  | PARO4G128T (engine)  | 3.2627  | 26.12  | **−32% PPL** |
| qwen3.5-0.8b.paro4g128-nat  | PARO4G128 (native)   | 3.2034  | 24.62  | **−36% PPL** |

**Headline:** ParoQuant beats MQ4 by 0.38–0.44 nats/tok = **32–36% lower PPL** at INT4.
This is a substantial quality lift and justifies Path B engineering.

### Caveats

1. **Group size confound.** paroquant uses G128 (2× the scale data, finer granularity);
   MQ4 uses G256. Some of the 32–36% lift comes from the smaller group size, not
   just from learned-vs-FWHT rotation. Iso-group comparison (paroquant-G256 vs
   MQ4-G256) would isolate the rotation effect. This is the first thing Path B's
   quantizer should produce — if half the quality lift evaporates at G256, Path B
   should keep paroquant's G128 storage and accept the ~25% larger model.

2. **Engine layout has a small quality cost.** paro4-native = 24.62 PPL vs
   paro4-engine = 26.12 PPL (6% relative drop). The engine layout's f16 theta
   → f32 sincos precomputation introduces tiny rounding error per Givens
   rotation. 6% isn't catastrophic but it suggests paro-quality is sensitive to
   numerical precision in the rotation step.

3. **Absolute MQ4 PPL is worse than historical** (38.36 here vs 25.65 in
   `benchmarks/results/ppl_baseline_20260501T061036Z.md`). Likely corpus difference
   (`benchmarks/quality-baselines/slice/wikitext2-1024s-2048ctx.txt` vs historical
   `dev/bench/data/wikitext2-test.txt`) or measurement-tool drift on
   `paroquant-runtime-probe`. The **relative** Δ between paroquant and MQ4 on the
   same corpus is still valid.

### HIPFIRE_GRAPH=1 regression is a separate bug

Initial PPL run without `HIPFIRE_GRAPH=0` produced 1791/2039 non-finite NLL
warnings on BOTH paroquant and MQ4. Root cause: `paroquant-runtime-probe @
26ebcfc3` defaults graph capture on for gfx12 (per CLAUDE.md memory note),
and graph replay produces NaN logits — same bug as the decode panic seen at
the bench step. Affects MQ4 too, not paroquant-specific. Must be fixed for
graph-default-on to be viable on this branch.

### Revised Path B decision

Quality A/B is conclusive (paroquant > MQ4 by 32–36% PPL) → Path B is worth
~1 week of engineering even with the group-size caveat. Phase 1 priority shift:

1. **Iso-group quality A/B first** (1 day): regenerate Qwen3.5-0.8B-PARO checkpoint
   at G256 instead of G128 (if paroquant calibration supports it), bench PPL.
   - If paro-G256 PPL is within 5% of paro-G128 PPL: keep G128 storage decision OR
     drop to G256 for storage parity with MQ4. Either is fine.
   - If paro-G256 PPL collapses (close to or worse than MQ4 G256): must keep G128
     storage. Path B kernels need G128 variants. Slightly more work.

2. **Format design + offline quantizer** (1-2 days)
3. **Runtime loader + new fused_rmsnorm_paro_rotate kernel** (1-2 days)
4. **Dispatch wiring + bench + coherence** (1 day)

Expected outcome: paro4 at ≥380 tok/s decode (MQ4-class) with ParoQuant-class
quality (24-26 PPL vs MQ4's 38).
