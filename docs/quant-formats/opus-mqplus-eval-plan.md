# Eval program: MQ4 vs MQ+ vs Opus Quant vs Opus-A8 (KLD / PPL / tok-s)

Goal: benchmark all four W4 quant schemes on a real model across **KLD**,
**perplexity**, and **tok/s**. **MQ+ is kept as a distinct format** (its own
quant-type id and artifact), not a flag on MQ4.

Anchor model (local): `~/.hipfire/models/qwen3.5-0.8b-bf16.hfq` (reference) and
`-mq4.hfq` (baseline already runs today).

## The four contenders (from the SQNR study, E6)

| name | quant-type id | weight | activation | compute kernel | new work |
|------|--------------:|--------|------------|----------------|----------|
| MQ4 (baseline) | 13 (exists) | affine u4, FWHT g256 | int8 (Q8_1) | iu8 mmq | none |
| **MQ+** | **32 (reserve)** | affine u4 + clip + SmoothQuant | int8 | **iu8 mmq (unchanged)** | offline only |
| **Opus-A8** | **33 (reserve)** | symmetric s4 + clip + SmoothQuant | int8 | sym-int4→iu8 | loader+kernel |
| **Opus Quant** | **34 (reserve)** | symmetric s4 + clip + SmoothQuant | int4 | grouped iu4 | loader+kernel+act-quant |

Reserve ids 32/33/34 (31 = Qtip3G256 is the current max; 22–27 are
documented-reserved interop slots — do not reuse). Defining MQ+ as id 32 with
its own `HfqInputFormat` keeps it separate from MQ4 per the directive.

## Pipeline (per format), reusing existing infra

1. **Produce artifact:** `hipfire-quantize --input qwen3.5-0.8b-bf16.hfq
   --output qwen3.5-0.8b-<fmt>.hfq --format <fmt>`
   (`HfqInputFormat` in `crates/hipfire-quantize/src/main.rs:4561`;
   `QuantType` enum at `:3614`).
2. **KLD reference:** build once from the bf16 model (`build_kld_ref*`,
   `crates/hipfire-eval/src/quality.rs:161`).
3. **Eval:** `hipfire-eval --models qwen3.5-0.8b-<fmt>` → KLD (`quality.rs`),
   PPL (`quality.rs`), tok/s (`performance.rs`, metric keys `tok_s`/`pp*_tok_s`).

## Per-format integration work

### MQ+ (id 32) — offline-only, lowest risk, do FIRST
- **Quantizer:** new `--format mq+`/`mq4plus`. Same affine-u4/FWHT/g256 payload
  as MQ4 (so it loads via the existing qt=13 kernel path), but:
  - clip-search the per-group range (MSE-optimal) — already prototyped in
    `examples/quant_opus_mqplus.rs::quant_affine(search=true)`.
  - SmoothQuant per-channel scale `s` from activation calibration, **folded into
    the preceding RMSNorm weight offline** (`W·s` into the proj weight, `1/s`
    into the norm) → zero runtime change, same iu8 kernel.
- **Loader/forward:** none (it's MQ4-format weights with rescaled norms).
- **Separateness:** distinct qt id 32 + `-mq+` artifact name; loader maps 32 →
  `DType::MQ4G256` (same kernel) but the file is its own thing.
- ⇒ KLD/PPL/tok-s all available immediately; tok/s ≈ MQ4 (same kernel).

### Opus-A8 (id 33) — symmetric W4, int8 activations
- **Quantizer:** symmetric s4 (no zero-point) + clip + SmoothQuant
  (`quant_sym` from the prototype). New `--format opus-a8`.
- **Loader:** new `DType::OpusS4G128` (hfq.rs match arm + rdna-compute DType).
- **Forward:** sym-int4 weight → upcast int8 → existing iu8 mmq path, but
  **drop the Q8_1 zero-point correction** (symmetric has none) — simpler than
  MQ4's affine mmq. int8 dynamic per-token activation quant already exists for
  MQ4.

### Opus Quant (id 34) — fused iu4 (W4A4), biggest lift
- **Quantizer:** symmetric s4 weights + clip + SmoothQuant (shares Opus-A8
  front-end). New `--format opus`.
- **Loader:** `DType::OpusS4G128` (shared) tagged for the iu4 path.
- **Forward (new):**
  - dynamic **per-token int4** activation quantizer (new kernel; MQ4 only does
    int8 acts);
  - **grouped iu4 GEMM** with per-group scale rescale in the epilogue — extend
    the validated `gemm_iu4_i32_wmma` (E5 did this via K-tiling host-side; the
    production kernel folds the rescale in).
- Validated numerically end-to-end already (E5: 20–22.6 dB on gfx1103).

## Quality-only fast path (optional, parallel)

For KLD/PPL **without** native kernels, the repo's **sim-quant** path
(`mq4_simquant_masked` at `main.rs:1143`, plus `qtip*sim` precedent and the
`*-qtip3sim.hfq` artifacts) bakes quant error into bf16 weights and runs the
normal forward. Extending it to also simulate per-token activation int4/int8
quant gives KLD/PPL for all four quickly; tok/s still needs the native path.

## Open decision

- **SmoothQuant activation calibration corpus.** MQ+/Opus-A8/Opus need
  per-channel activation max stats. The repo has Hessian/imatrix infra
  (`crates/hipfire-quantize/src/gptq.rs`, `hessian_io.rs`). Decision: reuse an
  existing imatrix/calibration corpus or pick one (e.g. a wikitext/code slice).

## Suggested sequencing

1. Anchor: run KLD/PPL/tok-s on existing `mq4` vs `bf16` → baseline + harness check.
2. **MQ+** (offline-only) → first new data point, validates the SmoothQuant+clip
   front-end against real KLD/PPL with zero kernel risk.
3. **Opus-A8** (symmetric W4, int8 acts) → isolates symmetric-vs-affine and the
   sym-int4→iu8 path.
4. **Opus Quant** (fused iu4 + int4 acts) → the full W4A4, reusing 2–3's
   front-end + the validated iu4 kernel.
