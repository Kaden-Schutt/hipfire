# Quant Fix 2026-05-13 Current Summary

Best accepted artifact: `/home/kaden/.hipfire/models/qwen3.5-9b.mq4.cand148-c146-l5-conv1d-mq6`

- candidate: `cand148-c146-l5-conv1d-mq6`
- md5: `3f82428f18f621c740a679c405d14f80`
- size: `5.178 GiB` (`5559695706` bytes)
- 20-chunk KLD: `0.206809`
- 20-chunk mean NLL: `2.228706`
- 20-chunk PPL: `9.2878`
- eval throughput: `328 tok/s`
- BF16 ref sha256: `06948cd36bab71fce2df5d9af1be03c9cfb4090637d881056a6937a29caa65a7`
- fixed eval env: `HIPFIRE_NORMALIZE_PROMPT=0 HIPFIRE_GRAPH=0 HIPFIRE_KV_MODE=asym3`

Astrea oracle wiring remains future work. The current POC uses ROCm PyTorch directly on `hiptrx` for projection-error and hidden/logits attribution against BF16 safetensors, then validates candidates with hipfire's 20-chunk KLD/PPL/speed gate.

Baseline and current controls:

| Artifact | KLD | PPL | Throughput | Logits rel RMSE | Worst hidden rel RMSE | Status |
|---|---:|---:|---:|---:|---:|---|
| baseline MQ4 `/home/kaden/.hipfire/models/qwen3.5-9b.mq4` | `0.330882` | `9.3404` | `346` | `0.234345` | `0.380971` | baseline |
| previous best `cand111` | `0.249053` | `9.2863` | `328` | `0.1480363` | `0.2616444` | superseded |
| previous best `cand118` | `0.244046` | `9.3011` | `328` | `0.1398290` | `0.2555000` | superseded |
| previous best `cand121` | `0.232697` | `9.2930` | `328` | `0.1289699` | `0.2404495` | superseded |
| previous best `cand133` | `0.210654` | `9.2676` | `328` | `0.1230890` | `0.2330394` | superseded |
| previous best `cand143` | `0.210238` | `9.2806` | `328` | `0.1236203` | `0.2330394` | superseded; last full hidden/logits oracle |
| current best `cand148` | `0.206809` | `9.2878` | `328` | pending | pending | accepted by KLD/PPL/speed; full oracle/decode validation pending |
| full-MQ6 control + asym3 KV | `0.134671` | `9.8683` | `192` | n/a | n/a | control only; fails PPL/speed |
| full-MQ6 control + q8 KV | `0.067687` | `9.3978` | `233` | n/a | n/a | control only; KLD passes but PPL/speed fail |

Current best improves KLD by `37.50%` versus baseline and preserves PPL/throughput gates. It still does not meet the final KLD target `<=0.09`.

Current accepted lineage tail:

- `cand118-c117-l25-conv1d-mq6`: KLD `0.244046`, PPL `9.3011`, throughput `328`.
- `cand120-c118-l0-conv1d-mq6`: L0 conv1d MQ6, KLD `0.237663`, PPL `9.1617`, throughput `328`.
- `cand121-c120-l14-conv1d-mq6`: L14 conv1d MQ6 after L0 restored PPL budget, KLD `0.232697`, PPL `9.2930`, throughput `328`.
- `cand127-c123-l2-qkv-mq4ls-pplrepair`: salvaged rejected L8 conv1d MQ6 with L2 qkv MQ4-LS, KLD `0.226117`, PPL `9.3007`, throughput `328`.
- `cand129-c127-l12conv-l5qkvls`: L12 conv1d MQ6 plus L5 qkv MQ4-LS repair, KLD `0.219196`, PPL `9.2484`, throughput `328`.
- `cand132-c131-l4-qkv-mq4ls-pplrepair`: salvaged L9 conv1d MQ6 with L4 qkv MQ4-LS, KLD `0.217498`, PPL `9.3009`, throughput `328`.
- `cand133-c132-l6-conv1d-mq6`: L6 conv1d MQ6, KLD `0.210654`, PPL `9.2676`, throughput `328`.
- `cand137-c133-l29-conv1d-mq6`: L29 conv1d MQ6, KLD `0.210378`, PPL `9.2987`, throughput `328`.
- `cand143-c137-l29-qkv-mq4ls`: L29 qkv MQ4-LS paired with L29 conv1d MQ6, KLD `0.210238`, PPL `9.2806`, throughput `328`.
- `cand148-c146-l5-conv1d-mq6`: L10 conv1d MQ6 plus L10 qkv rotated-weighted LS, repaired by L5 conv1d MQ6, KLD `0.206809`, PPL `9.2878`, throughput `328`.

Key rejected candidates after `cand143`:

- `cand146-c133-l10conv-l10qkv-rotwls-c8s512`: KLD `0.205683`, PPL `9.4331`, throughput `326`; best KLD seen, rejected on PPL and speed.
- `cand147-c146-l13-conv1d-mq6`: KLD `0.206135`, PPL `9.2875`, throughput `327`; KLD/PPL passed, rejected on speed.
- `cand144-c137-l29-qkv-rotwls-c8s512`: KLD `0.210347`, PPL `9.2872`, throughput `327`; rotated activation weighting failed to beat plain LS for L29 qkv.

PyTorch/oracle findings:

- Same-HFQ PyTorch replay of baseline MQ4 matched hipfire at `~5e-7` to `1.34e-6` last-token rel RMSE across layers 1-31; layer-0 projection probe was `~8e-8` to `9e-8`. This ruled out a RoPE-style runtime/kernel mismatch on the tested f32-KV path.
- Direct attribution found KLD-positive conv1d fixes, but many are PPL-toxic when applied alone. The effective pattern is PPL-aware pairing: add a KLD-positive conv1d MQ6 promotion, then add a targeted qkv MQ4-LS or rotated-weighted LS repair when PPL crosses the gate.
- Local tensor ranking is not sufficient. L13 ranked high locally and helped PPL, but regressed KLD in one branch and missed speed in another.
- The full-MQ6/q8-KV control proves the BF16 reference/eval harness can see KLD below `0.09`; the current MQ4/MQ6 selective policy is quality-limited, not harness-limited.

Next hypotheses:

1. KLD is not theoretically asymptotic at `0.20`, but the current cheap tensor-transplant/MQ4-refit loop is flattening around `0.205-0.21`.
2. The L10 branch is the strongest quality lever found so far, but it needs a stronger PPL/speed repair than diagonal rotated activation weighting.
3. The next real lever should be GPTQ-style or Hessian-aware block refit against calibration activations, not more local rel-RMSE ranking.
4. `cand148` needs the full accepted-candidate hidden/logits oracle and decode smoke before it can replace `cand143` as a deliverable model artifact.
