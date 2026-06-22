# Eval Tooling Refactor — Daemon-Resident Unified Eval Core

Status: in progress (2026-06-22)

## Why

The model-quality eval tooling grew organically: `build_kld_ref_hipfire` and
`eval_hipfire` are two **standalone example binaries** that each reload the
model, roll their own forward driver, lm-head dispatch, top-k/KLD math, and env
setup. They drift. A multi-day bisection on gfx1103 traced a 2.85-nat
*self-inconsistency* (the same fp16 model scored against a reference built from
itself gave KLD 2.85 instead of ~0) to exactly this drift:

- `build_kld_ref` sets `HIPFIRE_KLD_FP32_GQA4_ATTN` / `HIPFIRE_KLD_DIRECT_F16KV_ATTN`;
  `eval_hipfire` never reads them.
- The two use different prefill drivers (`forward_prefill_batch_single_chunk_captured_opts`
  vs `forward_prefill_batch`) and different F32/BF16 lm-head paths.

Core inference is correct (generation is fluent); the *eval tooling* has two heads.

## Principle

**The model-resident daemon is the eval host.** A loaded model serves many
passes/queries/tests/diagnostics without reloading. Reference build and
candidate scoring run through the **same resident forward with the same config**,
so self-consistency holds *by construction*. Diagnostics are **gated daemon
instruments** on a warm model (seconds, not 7-minute reload loops).

## Leverage that already exists

- Daemon holds `LoadedModel` resident across requests (weights/KV/DeltaNetState/
  scratch/tokenizer): `crates/hipfire-serving-core/src/model.rs:175`.
- The **`collect`** op already calibrates a resident model in-place and emits
  KLDREF top-K logits via `CalibCollector` + `gpu.active_capture`
  (`crates/hipfire-daemon/src/main.rs:4288`, `crates/hipfire-runtime/src/calibration.rs:82`).
  Reference build is ~80% a resident op already.
- Daemon uses the **same** `forward_prefill_batch` / `forward_scratch` as
  generation (`crates/hipfire-serving-core/src/generate.rs`) — the proven-coherent path.
- Protocol streams multi-frame results (like `generate`'s token stream), so
  per-chunk KLD and per-layer dumps fit naturally.

## Architecture

### 1. `hipfire-kld` crate (new, GPU-independent, CPU-testable)

Single source of truth for the pure scoring core:

- `math`: `log_z` (fp64 logsumexp), `top_k_log_softmax` (ref reduction →
  indices/log_probs/residual), `kld_position` (candidate scoring → kld, nll).
  Faithful extraction of `eval_hipfire::score_position` +
  `build_kld_ref::log_softmax_top_k_row`.
- `config`: `KldConfig` owns **every** env flag (`KLD_GRAPH`,
  `FP32_GQA4_ATTN`, `DIRECT_F16KV_ATTN`, `PREFILL_MAX_BATCH`,
  `NORMALIZE_PROMPT`, …). Ref-build and score read the SAME config; they cannot
  diverge.
- `refblock`: `RefBlock` view + canonical block (de)serialize (pure bytes;
  HFQM blob slices passed in by the caller to stay model-independent).
- `hfkseq`: per-chunk result (mean_kld/p99_kld/mean_nll) read/write
  (`HFKSEQ\0\0`, v2).
- `meta`: `RefMeta` self-describing header + `compat()` guard (see below).
- `codec`: per-blob payload codecs (bit-packed ids now; fp16/zstd reserved).

Precision policy: **wide accumulate, narrow store.** f64 is an accumulator type
only (the `log_z` partition sum over the full vocab, the K-term cross-entropy);
per-position `kld`/`nll` are stored `f32` (a KLD of ~1e-3 needs ~4 sig figs, f32
gives 7), so the high-volume arrays / per-layer divergence matrices stay in the
cheap SIMD+bandwidth lane. f64 is retained only for the ≤~1175 per-chunk HFKSEQ
aggregates (on-disk format compatibility).

### Reference artifact format (`.kldref`)

A full 0.8B/1175-chunk ref is ~2.47 GB, dominated by two equal arrays:
`top_indices` (u32, 1.23 GB) and `top_log_probs` (f32, 1.23 GB).

**Compression** (per-blob codec tag in the header):
- `top_indices` → **bit-pack to `ceil(log2(n_vocab))` bits** (vocab 248k → 18
  bits): deterministic ~44%, lossless. Shipped.
- `top_log_probs` → **fp16** (~2×) is the natural win but **lossy on the
  reference baseline**; reserved behind a measured KLD-shift tolerance, not a
  default. (Block-fp is more complex for marginal extra gain.)
- `zstd` wrap (lossless, ~1.5–2× on the structured blobs) reserved pending the
  dependency decision.
- Net: ~2.47 GB → ~1.2 GB lossless (bit-pack + zstd), ~0.7 GB with fp16 logprobs.

**Self-containment**: the artifact embeds `kldref.tokens` (the *tokenized* slice
the eval scores against) so it is functionally portable without the slice file;
provenance is `slice_md5` + `source_model_sha256` + `producer_cmd`. Raw slice
*text* is referenced, not embedded (optional ~3 MB zstd add for full archival).

**Self-describing metadata + `compat()` guard** — the operative lesson from the
2.85 bisection. The ref records the COMPLETE `(code, config, arch, tokenizer)` it
was built under: `git_commit`/`git_describe`/`git_dirty` (was **null** — the gap
that hid the cross-version comparison), the full `KldConfig`, `scoring_mode`,
`tokenizer_sha256`, `arch_id`/`n_vocab`, and a payload `content_sha256` (integrity
+ daemon resident-cache key). On score, the consumer's `RunEnv` is diffed against
the ref: **Error** (refuse) on arch/vocab/tokenizer mismatch; **Warn** on differing
`git_commit`, GPU arch, or any `KldConfig` flag. This makes silently scoring a
ref with diverged code/config unrepresentable.

### 2. Daemon `kld_eval` op

Resident op with modes: `build_ref`, `score` (vs a ref), `self_score`
(build + score same loaded model → must be ≈0). Reuses `collect`'s forward +
`active_capture`; uses `hipfire-kld` for all math/config. Streams per-chunk.

### 3. Gated instrument suite (warm-model debugger)

- `capture_hidden_layers` — per-layer hidden-state dump (the per-layer diff that
  was too expensive standalone).
- `dump_logits` — per-position logit capture.
- `scoring_mode` — prefill | per-token | single-shot toggle.
- `self_consistency` — build ref + score same model; assert ≈0. **CI guard** that
  would have caught the 2.85 instantly.

### 4. Deprecate standalone bins

Remove `build_kld_ref_hipfire` / `eval_hipfire` once the daemon op covers them.
`hipfire-eval` KLD batteries route through `executor_daemon` (resident), not
`executor_examples` (reload).

## Migration order (low-risk first)

1. ✅ DONE — `hipfire-kld` crate: math + config + refblock/hfkseq + codec
   (bit-pack) + meta/`compat()` + HFKREF `archive`. 28 CPU tests, no-GPU-CI.
2. ✅ DONE — daemon `kld_eval` op: `self_score`, `build_ref`, `score`, streaming
   per-chunk, `compat()` guard on score. **Validated on local gfx1103** (the arch
   that produced the 2.85): self_score and build_ref→persist→score both return
   mean_kld ≈ 8.43e-10. `qwen35::{kld_eval_self_score,kld_build_ref,kld_score}` +
   the daemon handler; client via `hipfire-daemon-adapter::kld_eval`.
3. TODO — gated instrument suite (per-layer hidden capture, logit dump,
   scoring-mode toggle). `self_consistency` already shipped as `self_score`.
4. TODO (STAGED — do not big-bang) — repoint `hipfire-eval` KLD batteries at the
   daemon op via `executor_daemon`, migrate the ~25 scripts/consumers off the
   standalone bins, THEN delete `build_kld_ref_hipfire`/`eval_hipfire`. Both bins
   now carry a DEPRECATED notice pointing here. Note: the new HFKREF reference is
   a *different format* from the legacy HFQM `.kldref.hfq`, so consumers migrate
   to "build ref resident → score" (or self_score) rather than loading old refs;
   the old refs lack the provenance (`git_commit` was null) the `compat()` guard
   needs, so regenerating them via the resident `build_ref` is the intended path.
5. (WS3) per-arch eval becomes a cheap warm daemon call → arch-coverage matrix
   (gfx1103 first-class: dispatch gaps, eval arch-map, perf baseline).

## Out of scope (separate workstreams)

- WS2 format registry/codec trait in `hipfire-quantize` (makes the 8-bit W8A8
  format land cleanly; ~18-edit friction today).
- WS4 wire `hipfire-detect` into the bash gates (remove ~1200 lines inline Python).
