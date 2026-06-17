# hipfire-train Phase 3 (scope): export tuned model → servable .hfq

Status: SCOPE (not yet implemented)
Goal: make a model trained/recovered in `hipfire-train` loadable by
`hipfire-daemon` for inference, i.e. emit a `Qtip3G256` `.hfq`.

## The target format already exists end-to-end

- **Quantizer:** `hipfire-quantize --format qtip3` (main.rs:5637) packs each
  linear to `Qtip3G256` — `pack_qtip3_group(symbols, scale)`, 100 B/group
  (0.391 B/w), `quant_type = QuantType::Qtip3G256 (=31)`.
- **Daemon/runtime:** serves it via `gemv_qtip3g256` / `_residual` (dispatch
  gemv family, `kernels/src/gemv_qtip3g256.hip`).
- **Container:** `.hfq` = `u64 LE header_len` → safetensors-style JSON header
  (per-tensor dtype/shape/data_offsets + `__metadata__` with arch_id, config,
  tokenizer) → packed tensor blob.

So the daemon can already load+serve qtip3 produced by the quantizer. The bridge
only has to get `hipfire-train`'s tuned weights into that pipeline.

## Decisive finding: reuse the quantizer, don't re-pack

The quantizer packs qtip3 **per-row** (`wf.par_chunks(k)`; 256-groups *within
each row*; needs `k % 256 == 0`). `hipfire-train`'s `qtip_quantize_dequant`
groups over the **flat** buffer (fine for our own fp32 forward, wrong for the
daemon kernel's per-row layout). Re-implementing packing in `hipfire-train`
would risk a silent layout mismatch with `gemv_qtip3g256`.

→ **The bridge emits tuned fp32 weights and runs them through
`hipfire-quantize --format qtip3`.** That reuses the canonical, tested packing,
header writer, arch detection, and tokenizer/config embedding — and guarantees
the served codes match the kernel. `hipfire-train`'s only new code is a tuned
fp32 **safetensors writer** (it currently only reads).

Edge case to verify: Supra `down_proj` has `k = 1408` (not ÷256). The per-row
qtip3 path can't 256-group it — confirm what the quantizer does (fp/other-format
fallback for `k % 256 != 0` tensors) and mirror that expectation.

## The real design fork: what happens to the LoRA delta

Recovery FT tunes **LoRA + layernorms with the trellis codes frozen**. Codes and
tuned layernorms export cleanly; the LoRA *weight delta* is the problem.

- **Path A — layernorm-only recovery (faithful QTIP, lossless export).** Run
  recovery with LoRA disabled (train only RMSNorm weights — `rmsnorm_backward`
  already yields `dw`; we already thread it). Then: quantize the *original* model
  with `--format qtip3` (servable base codes) and **patch the layernorm tensors**
  in that `.hfq` with the tuned ones (norms are stored fp; same shape/dtype →
  in-place tensor-record overwrite). Zero re-quantization loss. Smallest bridge.
  Matches what real QTIP actually tunes.
- **Path B — LoRA recovery, merge + re-quantize (lossy).** Merge `W ← hatW +
  scale·(B·A)` per linear, write tuned fp32 safetensors (with tuned norms), run
  `--format qtip3`. Supports the higher-capacity LoRA recovery, but re-quantizing
  the merged weights re-introduces qtip3 error on top of the recovery — partially
  undoing it. Must **measure** the post-re-quant KL vs the in-train KL to see how
  much survives.
- **Path C — LoRA fp sidecar (lossless, most work).** Export base codes + tuned
  norms + LoRA A/B as an fp16 sidecar; daemon applies LoRA at inference. No
  re-quant loss but needs new daemon LoRA-apply support (doesn't exist).

**Recommendation:** ship **Path A first** (faithful, lossless, tiny) as the v1
servable export; it's also the cleanest demonstration that recovery → servable
works. Add **Path B** if LoRA capacity proves necessary, gated on a measured
re-quant-loss number. Defer Path C unless a no-loss LoRA serve is required.

## Work breakdown (Path A v1)

1. **Layernorm-only recovery mode** in the recovery example/driver (disable LoRA
   B updates, or a flag that trains only `recovery_params` minus the LoRA
   tensors). Small.
2. **`.hfq` layernorm-patch utility:** read an existing qtip3 `.hfq`, replace the
   `*.input_layernorm.weight` / `*.post_attention_layernorm.weight` /
   `model.norm.weight` tensor data with the tuned fp tensors (download from GPU,
   match the .hfq's norm dtype), rewrite the file. ~one focused module + bin.
3. **Acceptance test (the gate):** quantize Supra-50M `--format qtip3` → patch
   with tuned norms → load in the daemon (or a runtime test) → generate the same
   prompt → confirm output matches `hipfire-train`'s post-recovery generation
   (exactly for Path A, since codes+norms are identical). Also re-measure KL vs
   teacher through the daemon path.

## Work breakdown (Path B, if pursued)

1. **fp32 safetensors writer** in `hipfire-train` (header JSON + blob; merge LoRA
   into base linears; write tuned norms; write embed).
2. Run existing `hipfire-quantize --format qtip3` on it.
3. Acceptance: load in daemon, generate, and **report the re-quant KL delta**
   (in-train post-recovery KL vs daemon-served KL) so the loss is explicit.

## Risks / open items

- `k % 256 != 0` tensors (down_proj 1408): confirm quantizer behavior; the daemon
  may serve them in a non-qtip3 format — the bridge must not assume all-qtip3.
- Header/metadata parity: the patched/emitted `.hfq` must carry arch_id +
  tokenizer + config the daemon expects; reusing the quantizer's writer (Path B)
  gives this for free, patching (Path A) preserves the original's metadata.
- The vendored `hipfire-train` qtip encoder vs the quantizer's: only the
  *quantizer's* output is ever served, so vendored-vs-canonical divergence is
  moot for serving (it only affected our internal fp32 student, which we're not
  shipping). Still worth the eventual `hipfire-quantize` lib extraction so both
  share one codec.

## Open decisions for the user

1. Path A (faithful, lossless, layernorm-only recovery) for v1, or go straight to
   Path B (LoRA capacity, with measured re-quant loss)?
2. Is daemon-served inference of the recovered model actually needed now, or is
   in-`hipfire-train` train+eval sufficient (in which case this stays scoped,
   unbuilt)?
