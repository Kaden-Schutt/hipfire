# LFM2 DFlash Chat32 Sidecar Fit

Date: 2026-06-26
Host: gfx1103
Prompt file: `benchmarks/prompts/lfm2_dflash_chat32.txt`
Prompt md5: `7ac0ca056c2005ab709d5584ab175f8e`

## Artifacts

- Teacher dump: `/tmp/lfm2-dflash-teacher-dump-chat32`
- FC fit sidecar: `/tmp/LFM2.5-350M.dflash.fcfit-chat32.oq4+.hfq`
- Selected sidecar: `/tmp/LFM2.5-350M.dflash.fitnorm-logitgrid-chat32-train24-score8.oq4+.hfq`
- Down-fit sidecar: `/tmp/LFM2.5-350M.dflash.fcdownfit-chat32-train24.oq4+.hfq`
- Down-fit selected sidecar: `/tmp/LFM2.5-350M.dflash.fcdownfit-normlogit-chat32-train24-score8.oq4+.hfq`
- Down-fit demote sidecar: `/tmp/LFM2.5-350M.dflash.fcdownfit-normlogit-demote-chat32-train24-score8.oq4+.hfq`
- Target model: `/srv/huggingface/_Hipfire/lfm2.5-350m-oq4plus-smoke.hfq`
- Base draft: `/tmp/lfm2-dflash-block-ce-sidecar-smoke.hfq`

## Procedure

1. Built a 32-prompt generation-boundary teacher dump with block size 4 and top-k 8.
2. Fit `fc.weight` from all 578 prompt rows with ridge `1e-2`.
3. Fit final `norm.weight` and scanned logit-bias candidates using blocks 0-23 for training and blocks 24-31 for held-out scoring.
4. Fit final-layer `mlp.down_proj.weight` on blocks 0-23, then repeated the norm/logit-bias scan.
5. Replayed the written sidecars on train and held-out blocks.
6. Compared end-to-end DFlash acceptance against the base draft on the same 32 prompts.

## Results

Teacher dump:

- prompts: 32
- rows: 578
- blocks: 32
- block size: 4
- target layers: `[2, 5, 8, 10, 13]`

FC fit:

- `fc.weight` quant type: F32
- train MSE: `2.711137e-6`

Norm/logit-bias scan:

- selected norm max scale: `4`
- selected logit bias: epochs `4`, lr `0.25`, max `4`, demote `false`
- held-out before bias: argmax `4/24`, top-k `8/24`, weighted CE `2.056471`
- held-out after bias: argmax `7/24`, top-k `15/24`, weighted CE `2.255824`

Independent block replay:

- train blocks 0-23: argmax `26/72`, top-k `61/72`, weighted CE `1.912998`, hidden cosine `0.389403`
- held-out blocks 24-31: argmax `7/24`, top-k `15/24`, weighted CE `2.255824`, hidden cosine `0.305660`

Down-fit branch:

- final `down_proj` fit: `delta_mse=1.785100e-1`, `prefinal_mse=1.785100e-1`
- down-fit selected norm max scale: `2`
- down-fit selected logit bias: epochs `4`, lr `0.25`, max `4`, demote `false`
- train replay: argmax `25/72`, top-k `59/72`, weighted CE `1.949490`, hidden cosine `0.373708`
- held-out replay: argmax `7/24`, top-k `15/24`, weighted CE `2.238022`, hidden cosine `0.297392`

Down-fit demoting-bias check:

- fixed logit bias: epochs `4`, lr `0.5`, max `2`, demote `true`
- train replay: argmax `23/72`, top-k `49/72`, weighted CE `1.645337`, hidden cosine `0.373708`
- held-out replay: argmax `7/24`, top-k `14/24`, weighted CE `1.749163`, hidden cosine `0.297392`

End-to-end acceptance, 32 prompts, max tokens 16, block 4:

- base draft: accepted `0`, drafted `900`, accept rate `0.0`
- selected sidecar: accepted `84`, drafted `657`, accept rate `0.127854`
- down-fit sidecar: accepted `85`, drafted `669`, accept rate `0.127055`
- down-fit demote sidecar: accepted `57`, drafted `726`, accept rate `0.078512`

## Conclusion

This is a real improvement over the base draft on the same corpus, but it is not
admission quality. Held-out argmax remains low and decoded previews are still
mostly repetitive punctuation or short fragments. The final `down_proj` fit is
neutral on acceptance, and the demoting-bias variant improves held-out CE while
hurting top-k and end-to-end acceptance. Keep iterating on the sidecar training
objective before promoting or packaging this DFlash artifact.
