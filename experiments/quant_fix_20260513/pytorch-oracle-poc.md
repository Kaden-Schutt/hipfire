# PyTorch Oracle POC - 2026-05-13

Astrea oracle integration is deferred. The current loop uses standalone PyTorch/HF scripts directly on `hiptrx` with ROCm PyTorch as the oracle/debugger.

## Engine-vs-HFQ Replay

Baseline artifact: `/home/kaden/.hipfire/models/qwen3.5-9b.mq4`.
Prompt tokens: `785,6725,3460,14125,279,16026,311,1430,264,4275,311,1265,279,1515,11625,13`.
KV mode for hidden dump: `f32`.

Artifacts:
- Hidden dump: `experiments/quant_fix_20260513/oracle_runtime/baseline-mq4-seq16-f32kv.meta.json`
- Layer replay summary: `experiments/quant_fix_20260513/oracle_runtime/replay_baseline_mq4_seq16/summary.json`
- L0 projection probe: `experiments/quant_fix_20260513/oracle_runtime/baseline-mq4-l0-projection-probe.json`

Findings:
- Layer 1-31 PyTorch replay using the same dequantized HFQ weights matches hipfire at `~5e-7` to `1.34e-6` last-token relative RMSE.
- L0 rotation matches Python exactly; fused and split projection paths match exactly; projection-vs-PyTorch matvec rel RMSE is `~8e-8` to `9e-8`.
- This rules out a RoPE-like runtime/kernel mismatch on the tested MQ4 f32-KV path. The remaining quality loss is not explained by operation ordering or fused-kernel math here.

## BF16 Oracle Drift

BF16/HF oracle output: `experiments/quant_fix_20260513/oracle_runtime/baseline-mq4-seq16-bf16-oracle.json`.

Short-seq baseline MQ4 vs BF16:
- logits last rel RMSE: `0.161703`
- final norm last rel RMSE: `0.225021`
- largest last-token hidden rel RMSE in this 16-token probe: layer 18 at `0.376013`

The drift accumulates smoothly rather than as a single hard cliff.

## KV Policy Isolation

All rows below are 5 chunks unless noted.

| Artifact | KV mode | KLD | PPL | tok/s | Notes |
|---|---:|---:|---:|---:|---|
| full MQ6 control | q8 | `0.030427` | `11.1210` | `175` | best quality control |
| full MQ6 control | asym4 | `0.047938` | `11.3109` | `130` | run concurrently with asym2; speed not final |
| full MQ6 control | asym3 | `0.080580` | `11.6358` | `178` | same slice before any rejected patch |
| full MQ6 control | asym2 | `0.106807` | `10.7816` | `130` | run concurrently with asym4; speed not final |
| cand72 | q8 | `0.183720` | `10.7151` | `215` | q8 helps MQ4 modestly |
| cand72 | asym3 | `0.197441` | `11.0955` | `213` | same slice before rejected patch |
| baseline MQ4 | q8 | `0.244429` | `11.1417` | `218` | q8 helps MQ4 modestly |
| baseline MQ4 | asym3 | `0.265792` | `11.5447` | `216` | same slice |

20-chunk full-MQ6 controls:
- asym3: `KLD 0.134671`, `PPL 9.8683`, `192 tok/s`.
- q8: `KLD 0.067687`, `PPL 9.3978`, `233 tok/s`.

Interpretation:
- Once weights are accurate enough, asym3 KV is a major quality tax versus q8/asym4.
- On current MQ4/cand72, weight quantization remains the dominant error, but KV still adds measurable KLD.

## Rejected Kernel Patch

Tried changing asym3 K cache scaling from norm-preserving scale to least-squares scale for the selected 3-bit code vector, including batched write and fold kernels.

Result:
- full MQ6 control improved on 5 chunks: `0.080580 -> 0.073317`.
- cand72 regressed on 5 chunks: `0.197441 -> 0.206949`.

Decision: rejected and reverted. The binary was rebuilt after revert.

## Next POC Direction

Use PyTorch directly, not Astrea, for the next iteration:
1. Capture real pre-RoPE/post-RoPE K distributions and attention-score deltas under q8/asym3/asym4.
2. Optimize asym3 codebook/scale against attention score error or KLD, not raw vector MSE alone.
3. Test whether asym4 can be made the high-quality KV policy while preserving enough AR/DFlash throughput, or whether asym3 needs a learned/calibrated codebook per model/head.
4. Keep Astrea as the future wrapper once this standalone loop produces a stable artifact contract.
