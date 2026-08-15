# OvisOCR2 Qwen3.5-VL corruption: source-precision diagnosis

Status: measured on gfx1201 in `codex/beta-batched-sampled-serve`; this is an
investigation record, not a registry admission.

## Verdict

The malformed OvisOCR2 output was not one Qwen3.5 architecture bug.
Three independent problems were stacked:

1. Qwen3.5-VL requires 3D MRoPE positions around the image span.
2. The VL daemon path ignored the request's closed-think assistant framing and
   unconditionally banned repeated 3- through 6-grams. That ban corrupts valid
   repeated HTML such as table rows and cells.
3. After fixing those semantics, the local MQ4, MQ6, and Q8 decoder artifacts
   still failed because this 0.8B OCR fine-tune is unusually
   quantization-sensitive. The source checkpoint's decoder is BF16. Even a
   whole-decoder BF16-to-F16 round trip changes its greedy trajectory; Q8
   round-trip corruption reproduces in Hugging Face without Hipfire.

The practical correctness tier is therefore source BF16 decoder weights with
the established F16 vision tower, not `ovisocr2.q8`, `.mq6`, or `.mq4`.

## Falsification path

`benchmarks/vision/dump_hf_decoder_reference.py` bypasses the Hugging Face
vision tower and injects Hipfire's exact post-merger rows. With source
precision, Hipfire and Hugging Face agreed through all 24 decoder layers at
the final prompt token and a later teacher-forced decode token, with relative
L2 error around `1e-6`; their full-vocabulary logits selected the same token.
This exonerated the visual splice, DeltaNet/full-attention alternation, MRoPE,
final RMSNorm, and tied output head.

`benchmarks/vision/hf_ovis_generate.py --fake-q8` applies Hipfire's exact
32-value symmetric Q8/F16-scale round trip inside Hugging Face. It reproduced
the same wrong title/table trajectory. A fake-F16 round trip also failed.
Selective Q8 preservation of attention, MLP, or tied embedding/head classes
did not recover the model, so there is no small "keep one layer family BF16"
recipe supported by this checkpoint.

## Working artifact

```bash
./target/release/hipfire-quantize \
  --input ~/.cache/huggingface/hub/models--ATH-MaaS--OvisOCR2/snapshots/65c619d374b55d4152e85150fc1b003700bc1f0c/ \
  --output ~/.hipfire/models/ovisocr2.bf16-source.hfq \
  --format bf16 \
  --include-vision
```

Measured artifact:

- Size: `1,720,324,224` bytes
- SHA-256:
  `a6faafd1da385d8a8c9c61765e667be07ace4312f0e6c6196fde240505588691`
- Decoder: exact source BF16 bytes, native BF16 GEMV with F32 accumulation
- Vision tower: F16, matching the established Qwen3.5-VL vision path

## End-to-end gates

On `benchmarks/images/dots_ocr_smoke_001.jpg`:

- Q8 KV + Q8 DeltaNet state, 500 generated tokens: byte-exact against the
  Hugging Face source-precision reference. Both texts have SHA-256
  `007f0f4d1339299bf002edeb586588910b56c37ac96297240225f57273ba7501`.
- Q8 KV + FP32 DeltaNet state, `max_tokens=4096`: natural stop after 3,935
  tokens; 11/11 content checks passed; three `<table>` / `</table>` pairs,
  46 `<tr>` / `</tr>` pairs, and 368 `<td>` / `</td>` pairs were balanced.
  Prefill was 147.6 token/s and decode was 135.5 token/s on the test gfx1201.
- Dots OCR regression: 4,633 tokens, valid JSON, byte-exact against the
  recorded Hipfire golden with SHA-256
  `89c74516b7c023c3a0c49145417ec27516feb402903d2fd7c162db533b2884a8`.
  The older vLLM reference differs by one pre-existing bbox pixel.

The 448-by-448 debug image is not a correctness oracle. The source-precision
model itself hallucinates repeated rows at that destructive resolution; the
full smart-resized page is the one-to-one reference.

## Redline follow-up

Correctness currently takes the dedicated MRoPE path, which intentionally
bypasses a text-only retained replay because it reads a three-axis position
buffer and launches a different RoPE kernel. Redline admission must capture
and certify that BF16 + MRoPE dispatch sequence as its own variant. Replaying
the existing text/Q8 tape would be a silent semantic mismatch.

## Batched-serve follow-up

The source-BF16 artifact is correct for the single-token path but is
deliberately not admitted by Qwen3.5's batched prefill/independent-decode
allowlist. The current gfx11/gfx12 batch matchers have no BF16-weight ×
F32-activation GEMM family; admitting BF16 by allowlist alone would fall into
an HFQ4 matcher and interpret two-byte BF16 rows with the wrong layout.

Supporting this model in batched serve therefore requires a real gfx11/gfx12
BF16 batched GEMM dispatch family and explicit BF16 arms for dense QKVZA/QKV,
gate/up/down, residual, and output projections. Until that exists, BF16
correctly falls back rather than silently corrupting batched requests.
