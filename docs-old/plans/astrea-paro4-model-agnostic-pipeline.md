# Astrea PARO4 Model-Agnostic Pipeline Plan

Status: stretch goal, after Qwen3.5 PARO4 runtime correctness and first Atlas
perf loop are stable.

## Goal

Build a model-agnostic ParoQuant path:

```text
input safetensors
  -> Astrea automated calibration
  -> tensor shape/metadata reproduction
  -> Atlas kernel task/eval loop when the shape needs new kernels
  -> hipfire PARO4 HFQ output
```

This is not the current ship path. Current scope remains Qwen/Qwen3.5 runtime
support and imported z-lab Paro checkpoints.

## Principles

- Astrea owns calibration and quality evidence. It can use PyTorch, calibration
  corpora, imatrix/AWQ/GPTQ-style signals, and learned Paro metadata.
- The runtime owns exact producer-consumer contracts: HFQ qtype, tensor payload
  layout, loader, dispatch, and HIP kernels.
- Atlas owns kernel evidence and bounded optimization tasks. It should not
  mutate kernels blindly; it emits task/eval/ledger artifacts with correctness
  gates.
- Model ingress should be shape-driven, not architecture-name-driven, wherever
  possible. Qwen is allowed to be the first supported family.

## Target Artifact

Astrea should produce an HFQ-compatible PARO4 artifact with:

- source model fingerprint and tokenizer/config metadata
- per-linear tensor records using `quant_type=28` / `PARO4G128`
- native Paro/AWQ buffers:
  - `qweight:int32[K, M/8]`
  - `qzeros:int32[K/128, M/8]`
  - `scales:f16[K/128, M]`
  - `pairs:int16[8, K]`
  - `theta:f16[8, K/2]`
  - `channel_scales:f16[K]`
- calibration/eval ledger with KLD/PPL/MSE and runtime fingerprint
- Atlas rows for AR, and DFlash when a paired draft exists

## Stages

1. Native import/runtime probe
   - Import existing Paro safetensors.
   - Verify source-vs-HFQ tensor exactness with `astrea paro-oracle`.
   - Run short AR smoke and finite-logit checks.

2. Qwen3.5 PARO4 performance loop
   - Use Atlas to collect AR rows for the current `gemv_paro4g128` route.
   - Generate bounded Atlas tasks for Paro decode tiling.
   - Keep `test_gemv_paro4g128`, `test_inference`, and `paro-oracle` as the
     minimum correctness gate for each kernel variant.

3. Astrea calibration prototype
   - Implement PyTorch Paro calibration for one Qwen3.5 dense module class.
   - Learn or select `pairs`, `theta`, and `channel_scales`.
   - Quantize rotated weights into AWQ buffers.
   - Compare MSE against MQ4/MFP4/HFQ4 and run model-level KLD/PPL.

4. Quantizer packaging
   - Add `hipfire-quantize --format paro4` only after the Astrea calibration
     recipe is concrete.
   - Rust quantizer may pack a calibrated Astrea side artifact into HFQ.
   - A no-calibration identity-rotation `paro4` mode is allowed only for kernel
     perf testing, not for quality claims.

5. Shape-general ingress
   - Parse safetensors shapes and module topology without assuming Qwen names.
   - Route unsupported shapes to `atlas task-pytorch` so kernel needs are
     explicit.
   - Add architecture-specific loaders only after shape contracts and quality
     gates exist.

## Open Risks

- Paro calibration cost may be high; start with representative layer classes
  before full-model calibration.
- PARO4 decode can become rotation-bound unless kernels tile or precompute the
  rotated activation effectively.
- Non-Qwen models may need loader/runtime architecture work even if tensor
  shapes are understood.
- A performant prefill path will likely need separate GEMM kernels; decode GEMV
  tiling is only the first runtime target.

## Current Recommendation

Do not add a user-facing `--format paro4` quantizer path yet. Add Atlas evidence
for the imported Qwen3.5 PARO4 route first, then prototype Astrea calibration on
one Qwen3.5 module class. Once quality and runtime perf both move, promote the
packaging path into `hipfire-quantize`.
