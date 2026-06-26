# LFM2 OQ Bring-up Status - 2026-06-25

Host: `gfx1103`, HIP 7.14, 45.1 GB VRAM.

## Artifacts

Generated from:

`/srv/huggingface/models--LiquidAI--LFM2.5-350M/snapshots/7728373d9f752dc3669ee3bf70786aef397874bb`

Artifacts:

| Artifact | Size | Notes |
|---|---:|---|
| `/srv/huggingface/_Hipfire/lfm2.5-350m-oq4.hfq` | 222,740,208 B | dense LFM2 linears as OQ4, routers/embed/norm/conv-filter as safer formats |
| `/srv/huggingface/_Hipfire/lfm2.5-350m-oq8.hfq` | 366,395,120 B | dense LFM2 linears as OQ8 |
| `/srv/huggingface/_Hipfire/lfm2.5-350m-oqplus.hfq` | 222,740,208 B | legacy OQ+ W4A8 tag; distinct from calibrated public `oq4+` |
| `/srv/huggingface/_Hipfire/lfm2.5-350m-conv0-in-proj-smoke.hessian.bin` | 4,194,371 B | HFHS v1 smoke Hessian for `model.layers.0.conv.in_proj` only; 1 sequence x 16 tokens |
| `/srv/huggingface/_Hipfire/lfm2.5-350m-oq4plus-smoke.hfq` | 222,742,256 B | OQ4 storage with `--format oq4+ --awq --ldlq --hessian`; only `model.layers.0.conv.in_proj.weight` has real LDLQ+AWQ calibration |

No full-model LFM2 `*.hessian.bin`, `*.calib.hfq`, or imatrix sidecar was found
under `/srv/huggingface` during the initial pass. The smoke HFHS sidecar above
now proves the `oq4+` producer-consumer path for one tensor, but it is not broad
enough for a quality-gated public `oq4+` artifact. The generated `oq4`,
`oqplus`, and `oq4plus-smoke` artifacts are runtime bring-up artifacts, not
calibrated admission artifacts.

## Runtime Checks

Short prompt: `The capital of France is a city with`

| Path | Command shape | Result |
|---|---|---|
| OQ4 act4 | `HIPFIRE_OQ4_PREFILL_ACT_BITS=4 infer_lfm2moe --max 2` | smoke passed, IDs `[523,523]` |
| OQ4 act4 parity | `HIPFIRE_OQ4_PREFILL_ACT_BITS=4 prefill_parity_lfm2moe` | failed argmax parity versus decode replay; expected risk because decode is W4A16 while this forces W4A4 |
| OQ4 act8 | `HIPFIRE_OQ4_PREFILL_ACT_BITS=8 prefill_parity_lfm2moe` | passed; prompt cosine 0.99979605, continuation cosine 0.99977983 |
| OQ8 act8 | `prefill_parity_lfm2moe` | passed; prompt cosine 0.99945274, continuation cosine 0.99948593 |
| legacy OQ+ W4A8 | `prefill_parity_lfm2moe` | passed; prompt cosine 0.99925757, continuation cosine 0.99959501 |
| OQ4+ smoke act8 | `HIPFIRE_OQ4_PREFILL_ACT_BITS=8 prefill_parity_lfm2moe` | passed; loader attached `model.layers.0.conv.in_proj.awq_scale.weight`; prompt cosine 0.99970197, continuation cosine 0.99974224 |
| OQ4+ smoke act4 | `HIPFIRE_OQ4_PREFILL_ACT_BITS=4 infer_lfm2moe --max 2` | smoke passed with AWQ sidecar attached, IDs `[523,523]` |

The vendored AMD Matrix Instruction Calculator reports both required gfx11
integer WMMA instructions:

- `v_wmma_i32_16x16x16_iu4`
- `v_wmma_i32_16x16x16_iu8`

## Sidecar Discovery

The Qwen3.5 DFlash sidecar naming/discovery template has been extended to LFM2
artifact names in `hipfire-model`. Examples now discovered next to the target
model include:

- `LFM2.5-350M-oq4.dflash.hfq`
- `LFM2.5-350M-op4.dflash.hfq`
- `LFM2.5-350M-mq4.dflash.hfq`
- `LFM2.5-1.2B-Thinking.op4+.dflash.hfq`

This is only the role-sidecar admission/discovery bridge. The generated support
matrix still marks arch 11 (`lfm2-moe`) DFlash as `none`, so attaching one of
these drafts is refused until the LFM2 spec-decode implementation and trained
draft are present.

## CASK / TriAttention Bridge

Arch 11 now has a runtime bridge for CASK/TriAttention sidecars:

- LFM2 state allocates the shared Q8 `KvCache` with a separate `physical_cap`,
  so `--cask-sidecar` can bound the physical KV allocation the same way Qwen3.5
  does.
- The LFM2 decode path uses physical positions for KV writes and logical
  positions (`physical + compact_offset`) for RoPE after compaction.
- LFM2 generation falls back to serial prefill while eviction is active, then
  calls `maybe_evict` after each prompt/decode token so the physical cursor does
  not overrun the capped KV buffer.
- LFM2 pre-RoPE Q capture now feeds the generic TriAttention calibration tap.
  Because LFM2 stores KV slots only for attention layers, its sidecars use
  attention-ordinal layer indices (`0..num_attention_layers`), not full model
  layer ids.

This is not a trained-sidecar claim. A usable LFM2 CASK/TriAttention artifact
still needs calibration over an LFM2 corpus and a recall/long-context quality
gate before it should be treated as an admitted sidecar.

## 800-token Local Bench

Prompt: repeated local sentence; embedded tokenizer produced 800 tokens.
Single warm-ish release run per artifact, `--max 4`; timings exclude model load.

| Artifact/path | Prefill | Approx prefill tok/s | Decode |
|---|---:|---:|---:|
| OQ4 act4 | 800 tok in 0.35 s | 2286 tok/s | 4 tok in 0.03 s, 124.6 tok/s |
| OQ4 act8 | 800 tok in 0.26 s | 3077 tok/s | 4 tok in 0.03 s, 124.8 tok/s |
| OQ8 act8 | 800 tok in 0.71 s | 1127 tok/s | 4 tok in 0.04 s, 89.9 tok/s |
| legacy OQ+ W4A8 | 800 tok in 0.71 s | 1127 tok/s | 4 tok in 0.04 s, 89.4 tok/s |
| MQ4 baseline | 800 tok in 1.03 s | 777 tok/s | 4 tok in 0.14 s, 28.7 tok/s |

## Follow-ups

- Scale the HFHS collection beyond the one-tensor smoke sidecar so
  `--format oq4+ --awq --ldlq --hessian` can produce a real full-model
  calibrated `oq4+` artifact.
- Run quality evidence against a BF16 or accepted high-precision reference
  before promoting `oq4+`.
- Calibrate and gate real LFM2 CASK/TriAttention sidecars using the
  attention-ordinal sidecar convention.
- Implement and train the LFM2 DFlash drafter/runtime path; only discovery is
  wired today.
- Repeat benches with a dedicated multi-run bench harness and record variance.
