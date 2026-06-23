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

## Step 4 — finalized scope (2026-06-23)

Decisions locked: scripts **drive the daemon op directly** (no wrapper CLI);
**delete the 2 qwen35 bins only** (`eval_hipfire`, `build_kld_ref_hipfire`).
`eval_hipfire_llama` stays — the daemon op is qwen3.5-only
(`hipfire-daemon/src/main.rs:4437` arch-gate); llama migration is a separate task.
`build_kld_ref` (cross-engine) and `eval_gguf` (GGUF anchor) are different-purpose,
out of scope.

### STATUS (2026-06-23): Phases A/B/C DONE; old llama format removed

Decision evolved mid-execution: rather than preserve cross-engine validation, we
**removed all support for the old llama (cross-engine HFKLDR `.kldref.bin`) KLD
format** and consolidated everything onto the hipfire-self daemon path.

- **Phase A (c0eb9942):** hipfire-eval Quality battery → daemon `kld_eval` (validated
  green: OQ+C vs resident wikitext ref, mean_kld=0.0507).
- **Removal (cf730a87):** deleted `build_kld_ref` (HFKLDR producer), `eval_hipfire` +
  `eval_hipfire_llama` (HFKLDR/HFQM scorers), `eval_gguf` (llama-perplexity + HFKLDR),
  `convert_kldref`, `build_kld_ref_hipfire` (HFQM self-builder, replaced by daemon) +
  their Cargo entries; stripped the eval_hipfire-spawning path from hipfire-eval;
  Quality now routes through the daemon for every non-Mock executor.
- **Phase B/C (09878ce4):** `scripts/lib/kld_daemon.sh` (kld_build_ref/kld_score/
  kld_field) + converted all 8 scripts (awq_*, quant_cohort, mi300x_*) off
  eval_hipfire/`.kldref.bin` onto the daemon op. gfx942 bring-up scripts converted
  mechanically (untested on-box, flagged in headers).

Residual (optional follow-up): peripheral Python/shell tools still *mention* the old
format in comments/paths — `scripts/cross_engine_check.py`, `scripts/fetch-eval-refs.sh`,
`scripts/astrea.py` (a source-file list). Not on the KLD path; clean up if revisited.

---

### Phase A — repoint `hipfire-eval` Quality battery (Rust)
- `crates/hipfire-eval/Cargo.toml`: add deps `hipfire-kld`, `hipfire-daemon-adapter`,
  `hipfire-daemon-protocol` (none present today).
- `crates/hipfire-eval/src/executor_daemon.rs:21-36` (`daemon_battery_rows`): add a
  `BatteryId::Quality` arm. For each candidate: ensure the daemon has the candidate
  model loaded, then `adapter.kld_eval(KldEvalRequest{ mode: Score, ref_path, output,
  .. })`; map `KldEvalResponse{ mean_kld, p99_kld, mean_nll, ppl }` → eval rows.
  Reuse the existing daemon harness the other batteries use (spawn/connect + lock).
- `crates/hipfire-eval/src/quality.rs:161-263` (`run_kld_reference_row`): delete the
  `std::process::Command::new(eval_hipfire)` spawn (+ `resolve_eval_hipfire_bin`,
  the arg builder, and the HFKSEQ-file reparse). Reference production moves to
  daemon `build_ref` (resident) — drop the `build_kld_ref_hipfire` dependency.
- `crates/hipfire-eval/src/driver.rs:625-634`: Quality now routes via the daemon
  executor, not `kld_reference_rows`→examples.
- Ref-format note: HFKREF ≠ legacy HFQM `.kldref.hfq`. The battery regenerates refs
  via resident `build_ref` (or `self_score`); old refs are not reused (no provenance
  for `compat()`).

### Phase B — rewrite the 8 scripts to drive the daemon
Pattern (one resident daemon per sweep, sequential `load`+`kld_eval` per variant
over the JSON-lines stdin protocol — no per-variant process spawn, no `eval_hipfire`):
```
{ echo '{"type":"load","model":"<ref-model>", ...}'
  echo '{"type":"kld_eval","mode":"build_ref","ref_path":"<ref>", ...}'
  for v in variants; do
    echo '{"type":"load","model":"'$v'", ...}'
    echo '{"type":"kld_eval","mode":"score","ref_path":"<ref>","output":"'$v.kldseq'"}'
  done
} | hipfire-daemon | jq -c 'select(.type=="kld_evaled")'
```
Scripts to convert: `scripts/quant_cohort.sh` (158/255/373), `scripts/awq_alpha_sweep.sh`
(30/47), `scripts/awq_f1_vs_f2.sh` (24), `scripts/awq_f2_alpha_sweep_wait.sh` (8),
`scripts/mi300x_bootstrap.sh` (187-192), `scripts/mi300x_sub_0_10_attempt.sh` (25/28),
`scripts/mi300x_v3_matrix.sh` (41/69), `tests/mi300x_smoke_gfx942.sh` (14/20/22).
Drop the `cargo build --example eval_hipfire` lines; build `hipfire-daemon` instead.
The mi300x scripts also need the daemon's resource-lock env (`HIPFIRE_RESOURCE_LOCK_WAIT_MS`).

### Phase C — delete the 2 bins (only after A+B land + green)
- Delete `crates/hipfire-runtime/examples/eval_hipfire.rs`,
  `crates/hipfire-runtime/examples/build_kld_ref_hipfire.rs`.
- Remove their `[[example]]` entries from `crates/hipfire-runtime/Cargo.toml`
  (`eval_hipfire` 334-336; the `build_kld_ref_hipfire` entry).
- Update `benchmarks/quality-baselines/README.md` (drops the two entries; keep
  `build_kld_ref`, `eval_gguf`).
- Verify: `./tests/no-gpu-ci.sh`, then a warm Quality battery run on gfx1103, and
  re-run one migrated sweep (e.g. `awq_alpha_sweep.sh`) to confirm parity.

### Risks / sequencing
- A before B before C (scripts must stop referencing the bins before deletion).
- Daemon op is qwen35-only: any script pointed at a non-qwen35 model must keep
  `eval_hipfire_llama`/`eval_gguf` (don't force those onto the daemon op).
- Parity check: one variant scored old-path vs daemon-path should match within
  the established 8.43e-10 self-consistency band before mass-converting scripts.

## Out of scope (separate workstreams)

- WS2 format registry/codec trait in `hipfire-quantize` (makes the 8-bit W8A8
  format land cleanly; ~18-edit friction today). **Partly addressed** 2026-06-23:
  W8A8 (Oq8G256), OQ+/OQ+T/OQ+C (Opus Plus W4A8 tiers) landed without the trait,
  but each new format still touches QuantType/HfqInputFormat/from_flag/loader-arm
  by hand — the registry would collapse that.
- Llama (and other non-qwen35 arch) support in the daemon `kld_eval` op — would let
  `eval_hipfire_llama` migrate + delete; needs arch-generalizing the handler gate.
- WS4 wire `hipfire-detect` into the bash gates (remove ~1200 lines inline Python).
