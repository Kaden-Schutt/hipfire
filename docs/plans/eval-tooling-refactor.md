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
- `refblock`: `KldRefBlock` struct + legacy binary (de)serialize (pure bytes;
  HFQM blob slices passed in by the caller to stay model-independent).
- `hfkseq`: per-chunk result (mean_kld/p99_kld/mean_nll) read/write
  (`HFKSEQ\0\0`, v2).

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

1. `hipfire-kld` crate: pure math + config + formats + CPU unit tests (no GPU).
2. Daemon `kld_eval` op (build_ref/score/self_score, streaming) on the resident model.
3. Instrument suite + `self_consistency` guard.
4. Repoint `hipfire-eval`; delete standalone bins.
5. (WS3) per-arch eval becomes a cheap warm daemon call → arch-coverage matrix
   (gfx1103 first-class: dispatch gaps, eval arch-map, perf baseline).

## Out of scope (separate workstreams)

- WS2 format registry/codec trait in `hipfire-quantize` (makes the 8-bit W8A8
  format land cleanly; ~18-edit friction today).
- WS4 wire `hipfire-detect` into the bash gates (remove ~1200 lines inline Python).
