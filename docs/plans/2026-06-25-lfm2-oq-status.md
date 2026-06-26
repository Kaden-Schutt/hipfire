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
| `/srv/huggingface/_Hipfire/lfm2.5-350m-oq4plus-smoke.hfq` | 222,742,256 B | OQ4 storage from the old `oq4+` LDLQ spelling; under the current parser this is `oq4++` semantics (`--format oq4++ --hessian`), and only `model.layers.0.conv.in_proj.weight` has real LDLQ+AWQ calibration |
| `/srv/huggingface/_Hipfire/LFM2.5-350M-awq-smoke.oq4+.hfq` | 222,740,208 B | canonical public `oq4+` spelling; AWQ/activation-aware smoke artifact derived from the one-tensor Hessian diagonal, without LDLQ error feedback |
| `/srv/huggingface/_Hipfire/LFM2.5-350M-awq-smoke.oq8+.hfq` | 366,395,120 B | canonical public `oq8+` spelling; AWQ/activation-aware smoke artifact derived from the one-tensor Hessian diagonal, without LDLQ error feedback |

No full-model LFM2 `*.hessian.bin`, `*.calib.hfq`, or imatrix sidecar was found
under `/srv/huggingface` during the initial pass. The smoke HFHS sidecar above
now proves the LDLQ producer-consumer path for one tensor, but it is not broad
enough for a quality-gated public `oq4++` artifact. The generated `oq4`,
`oq8`, `oqplus`, canonical `oq4+`/`oq8+` AWQ smoke, and old-spelling
`oq4plus-smoke` artifacts are runtime bring-up artifacts, not calibrated
admission artifacts.

On 2026-06-26, `collect_artifacts` gained arch 11 (`lfm2`) support for text
decoder calibration. A four-token plumbing smoke on
`/srv/huggingface/_Hipfire/LFM2.5-350M-awq-smoke.oq8+.hfq` wrote
`/tmp/lfm2-350m-collector-smoke.calib.hfq` (402 MiB) with 92 Hessian tensors
and 92 imatrix tensors; max `diag(H)` versus `sum(x^2)` relative error was
`0.000e0`. This proves the collector path and residual-GEMV capture hook for
LFM2 dense projections, not a quality/admission calibration run.

Also on 2026-06-26, LFM2-MoE indexed expert capture was added as a
calibration-only top-k tap. A two-token 8B-A1B smoke on
`/srv/huggingface/_Hipfire/lfm2.5-8b-a1b-mq4.hfq` wrote
`/tmp/lfm2-8b-a1b-collector-expert-smoke.calib.hfq` (447 MiB) with 88 Hessian
tensors and 538 imatrix tensors; max `diag(H)` versus `sum(x^2)` relative
error was `0.000e0`. The package contains checkpoint-style expert keys such as
`model.layers.10.feed_forward.experts.17.w1.imatrix`, `.w2.imatrix`, and
`.w3.imatrix`, and metadata marks routed experts as
`imatrix-only-selected-experts`.

The quantizer now imports HFQM `.imatrix` vectors from `--hessian` packages in
addition to Hessian diagonals. A one-tensor LFM2 quantizer smoke with
`--include-prefix model.layers.2.feed_forward.experts.26.w1.weight` and the
8B-A1B smoke calibration package reported 1076 imatrix keys, aggregated 14
gate/up plus 7 down imatrix vectors for layer 2, emitted
`model.layers.2.feed_forward.awq_scale_gate_up.weight` and
`model.layers.2.feed_forward.awq_scale_down.weight`, and wrote the selected
expert tensor. This proves routed expert OQ+/AWQ plumbing, but it is still not
a quality/admission calibration run.

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
| canonical OQ4+ AWQ smoke act8 | `HIPFIRE_OQ4_PREFILL_ACT_BITS=8 prefill_parity_lfm2moe` on `LFM2.5-350M-awq-smoke.oq4+.hfq` | passed; loader attached `model.layers.0.conv.in_proj.awq_scale.weight`; prompt cosine 0.99949416, continuation cosine 0.99976073 |
| canonical OQ4+ AWQ smoke act4 | `HIPFIRE_OQ4_PREFILL_ACT_BITS=4 infer_lfm2moe --max 2` on `LFM2.5-350M-awq-smoke.oq4+.hfq` | smoke passed with AWQ sidecar attached, IDs `[574,574]` |
| canonical OQ8+ AWQ smoke act8 | `prefill_parity_lfm2moe` on `LFM2.5-350M-awq-smoke.oq8+.hfq` | passed; loader attached `model.layers.0.conv.in_proj.awq_scale.weight`; prompt cosine 0.99933023, continuation cosine 0.99911341 |

The vendored AMD Matrix Instruction Calculator reports both required gfx11
integer WMMA instructions:

- `v_wmma_i32_16x16x16_iu4`
- `v_wmma_i32_16x16x16_iu8`

## Canonical OQ+ Smoke Bench

Measured 2026-06-26 on gfx1103 with repo-built `infer_lfm2moe` dev example,
`HIPFIRE_PREFILL_BATCHED=1`, and GPU lock held through the repo-built
`hipfire lock` command. Artifacts:
`/srv/huggingface/_Hipfire/LFM2.5-350M-awq-smoke.oq4+.hfq` and
`/srv/huggingface/_Hipfire/LFM2.5-350M-awq-smoke.oq8+.hfq`.

Prompt: `The capital of France is a city with` (8 embedded-tokenizer tokens).
Timings include the example's printed model-load, prefill, and decode windows;
generation was repetitive, so this is runtime smoke evidence, not quality
evidence.

| Activation path | Load | Prefill | Decode |
|---|---:|---:|---:|
| OQ4+ act8 | 2.6 s | 8 tok in 0.05 s | 32 tok in 0.26 s, 124.7 tok/s |
| OQ4+ act4 | 2.5 s | 8 tok in 0.02 s | 32 tok in 0.26 s, 121.2 tok/s |
| OQ8+ act8 | 4.4 s | 8 tok in 0.06 s | 32 tok in 0.36 s, 88.2 tok/s |

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
- LFM2 daemon `generate_batch_prefill` now follows the same eviction-safe serial
  path when CASK/TriAttention is loaded: prompt/suffix tokens are applied with
  `decode_step`, `maybe_evict` runs after each token, and returned resident
  session handles carry the compacted logical position.
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

## 8B-A1B MoE Host Baseline

Measured 2026-06-26 on gfx1103 with repo-built `infer_lfm2moe` release example,
`HIPFIRE_PREFILL_BATCHED=1`, and GPU lock held through the repo-built
`hipfire lock` command. Artifact:
`/srv/huggingface/_Hipfire/lfm2.5-8b-a1b-mq4.hfq`.

| Run | Prompt tokens | Load | Prefill | Decode |
|---|---:|---:|---:|---:|
| Long-context MQ4 | 2641 | 53.1 s | 40.70 s, 64.9 tok/s | 64 tok in 2.11 s, 30.4 tok/s |
| Short-context MQ4 | 9 | 41.4 s | 0.16 s, 56.3 tok/s | 128 tok in 3.50 s, 36.5 tok/s |

Both generations were repetitive, so these are runtime baseline measurements, not
quality/admission evidence. No 8B-A1B `oq4+`/`oq8+` artifact exists locally yet;
generating one still requires a full calibration/quantization pass rather than
the 350M smoke Hessian.

## Daemon Batch Prefill

Arch 11 now has a daemon `/generate_batch_prefill` path for LFM2-MoE single-GPU
models. The first slice is serial-exact and resident-only: each request session
owns an isolated `Lfm2MoeState`, materializes its own prompt or suffix, and calls
the arch-local `prefill_batch` before returning a resident `lfm2_session` state
handle. This validates the protocol surface and session lifecycle before adding
a fused cross-session worker.

Runtime smoke passed on 2026-06-26 with:

```text
CARGO_TARGET_DIR=/tmp/hipfire-target-lfm2-daemon \
HIPFIRE_PREFILL_BATCHED=1 HIPFIRE_OQ4_PREFILL_ACT_BITS=8 \
HIPFIRE_RESOURCE_LOCK_WAIT_MS=60000 \
cargo run -q -p hipfire-daemon --features arch-lfm2moe
```

Model:
`/srv/huggingface/_Hipfire/LFM2.5-350M-awq-smoke.oq4+.hfq`.

The probe returned `generate_batch_prefill_ready` with
`mode=lfm2_serial_prefill_batch`. A two-session prompt batch completed with
`backend=lfm2_arch_prefill_batch`, `plan=serial_exact`, resident session handles
for `lfm2-a` and `lfm2-b`, and `prefill_tokens=31` total.

A follow-up lifecycle smoke also passed on the same artifact: the returned
`lfm2_session` handle round-tripped through `describe_state` with three owned
pages, suffix continuation advanced a resident session from logical position 15
to 16, `release_state` released the returned handle with `loaded_released=1`,
`release_sessions` accepted the returned handle object, and daemon `reset`
cleared a resident LFM2 session so a later `describe_state` reported it unknown.

The resident state handle path now supports attach/fork by
`runtime_state_handle` with prefix-hash validation. A branch request can attach
to an existing LFM2 resident session, deep-copy the KV/conv/scratch state into a
new LFM2 session, and continue from the requested suffix. `prefix_hash_preflight`
also routes arch 11 through the LFM2 hash domain (`lfm2_q8_kv`) so clients can
compute candidate prefix hashes before requesting an attach.

Semantic-boundary checkpoint smoke also passed on the same artifact. A fresh
Jinja-chat prompt with `semantic_boundary_checkpoints=true` emitted three
attachable `prefix_checkpoints`; an attached branch reused
`lfm2-checkpoint:lfm2-boundary-first:lfm2-boundary-src:boundary:0:13`,
prefilled the remaining 18 tokens, and reached the same final logical position
31 as the full prompt.

The direct daemon handoff from `generate_batch_prefill` to `generate` now also
works for LFM2 AR sessions. A host smoke loaded
`/srv/huggingface/_Hipfire/LFM2.5-350M-awq-smoke.oq4+.hfq`, prefetched a session
to logical position 18, then decoded through `session_id` +
`prefill_already_done=true` with `prefill_ms=0` and one generated token (`sky`).

The same handoff now also runs with an LFM2 DFlash draft loaded. The daemon
loaded the target plus `/srv/huggingface/_Hipfire/LFM2.5-350M.oq4+.dflash.hfq`,
captured per-session target hidden history during `generate_batch_prefill`, and
resumed DFlash `generate` from logical position 18 with `prefill_ms=0`,
`dflash=true`, one DFlash cycle, and one generated token (`sky`). Acceptance was
0 in that smoke, so this is session-state plumbing evidence, not a speedup or
quality claim.

Current refusals are deliberate: pipeline parallelism, and DFlash combined with
CASK/TriAttention eviction. Those need their own state-sharing contracts instead
of being implied by the resident-only smoke path.

## DFlash Training Evidence

The LFM2 DFlash path now has direct runtime probes:

- `lfm2_dflash_seed_smoke` loads an LFM2 target and LFM2 DFlash sidecar, then
  runs the draft forward path for finite-logit smoke coverage.
- `lfm2_dflash_acceptance_eval` runs the greedy speculative step used by the
  daemon over one or more prompts and reports accepted versus drafted tokens.
- `lfm2_dflash_teacher_dump` produces `hipfire-lfm2-dflash-teacher-v1`
  block-teacher windows from the actual LFM2 target runtime, including captured
  target hidden features, pre-final target hidden rows, seed tokens, target
  argmax labels, and target top-k logits for each DFlash slot. The dump now
  writes `target_hidden.f32` and records `target_hidden_shape` so trainer probes
  can fit against the target residual stream before final embedding norm.
  It also writes per-block hidden labels:
  `dflash_block_target_hidden.f32` and
  `dflash_block_target_norm_hidden.f32`. The normalized rows align with the
  exact `block, slot-1` draft rows that are fed through the target LM head.
- `lfm2_dflash_fit_fc` is a runtime-aligned smoke fitter for the DFlash
  `fc.weight` projection. It solves a ridge least-squares projection from saved
  DFlash extract features to `target_hidden.f32`, then rewrites only
  `fc.weight` in the sidecar as an `F32` tensor while copying all other tensors
  byte-for-byte.
- `lfm2_dflash_fit_norm` is the next deployed-path trainer slice. It replays
  saved block windows through the actual DFlash runtime, downloads the draft
  rows after final DFlash norm, fits a diagonal correction against
  `dflash_block_target_norm_hidden.f32`, and rewrites only `norm.weight` as F32.
- `lfm2_dflash_fit_down` captures the final DFlash FFN `gate_up` activation and
  `residual_ffn` stream, then fits the final-layer `mlp.down_proj.weight`
  against `dflash_block_target_hidden.f32 - residual_ffn`. It rewrites only that
  `down_proj` tensor as F32.
- `lfm2_dflash_block_teacher_eval` replays saved
  `hipfire-lfm2-dflash-teacher-v1` block windows through the actual
  `run_dflash_draft_for_logits` runtime path and compares draft logits with the
  saved target argmax/top-k labels. When block normalized hidden labels are
  present, it also reports draft-vs-target hidden MSE and cosine.

Current local artifacts:

| Artifact | Notes |
|---|---|
| `/tmp/lfm2-dflash-block-teacher-smoke` | teacher dump with `dflash_blocks`, block size 4, max context 2 |
| `/tmp/lfm2-dflash-teacher-dump-current` | current `lfm2_dflash_teacher_dump` output, block size 4, two blocks, target layers `[2, 5, 8, 10, 13]` |
| `/tmp/lfm2-dflash-teacher-dump-fcfit8` | teacher dump with `target_hidden.f32`, block size 4, eight blocks, target layers `[2, 5, 8, 10, 13]` |
| `/tmp/lfm2-dflash-teacher-dump-hidden-smoke` | teacher dump with block-level pre-final and final-normalized hidden labels, block size 4, two blocks |
| `/tmp/lfm2-dflash-teacher-dump-hidden8` | teacher dump with block-level hidden labels, block size 4, eight blocks |
| `/tmp/lfm2-dflash-block-ce-sidecar-smoke.hfq` | block-CE-trained smoke sidecar, 14 tensors, about 7.5 MiB |
| `/tmp/lfm2-dflash-fcfit8-sidecar-smoke.hfq` | FC-only ridge-fitted sidecar derived from `/tmp/lfm2-dflash-teacher-dump-fcfit8`; `fc.weight` stored as F32, all other tensors copied from the block-CE sidecar |
| `/tmp/lfm2-dflash-fcfit8-normfit8-sidecar-smoke.hfq` | FC-fit sidecar with `norm.weight` additionally fitted from `/tmp/lfm2-dflash-teacher-dump-hidden8` |
| `/tmp/lfm2-dflash-fcfit8-normfit8-ms1.0-sidecar-smoke.hfq` | conservative `norm.weight` fit (`max_scale=1.0`); first local LFM2 DFlash smoke with non-zero acceptance |
| `/tmp/lfm2-dflash-fcfit8-normscan8-sidecar-smoke.hfq` | automated `lfm2_dflash_fit_norm --scan-max-scale 1,2,4,8` output; selected `max_scale=1.0` by weighted top-k/CE scoring |
| `/tmp/lfm2-dflash-fcfit8-downfit8-sidecar-smoke.hfq` | FC-fit sidecar with final `layers.0.mlp.down_proj.weight` fitted from `/tmp/lfm2-dflash-teacher-dump-hidden8` |
| `/tmp/lfm2-dflash-fcfit8-downnormfit8-sidecar-smoke.hfq` | down-fit sidecar with follow-up `norm.weight` fit |

The proxy block-CE trainer reduced sampled block CE from `1.386195e0` to
`1.318314e0` on the smoke replay set, but the acceptance evaluator still
reported `accepted=0`, `drafted=45`, `accept_rate=0.0`.

On 2026-06-26, the first current-teacher dump run completed:

```text
HIP_VISIBLE_DEVICES=0 HIPFIRE_OQ4_PREFILL_ACT_BITS=8 \
  CARGO_TARGET_DIR=/tmp/hipfire-target-lfm2-dflash-smoke \
  cargo run -p hipfire-arch-lfm2moe --features deltanet \
  --example lfm2_dflash_teacher_dump -- \
  --model /srv/huggingface/_Hipfire/lfm2.5-350m-oq4plus-smoke.hfq \
  --draft /tmp/lfm2-dflash-block-ce-sidecar-smoke.hfq \
  --out /tmp/lfm2-dflash-teacher-dump-current \
  --prompt "Write a tiny Rust add function. Then explain why it works in one sentence." \
  --block-size 4 --max-blocks 2 --topk 8
```

The run wrote 16 prompt tokens, 2 DFlash blocks, and 327,680 feature floats.
It initially exposed a small-B full-attention dispatch bug: `AttnFullF32` could
select the WMMA full-attention rung for DFlash `B=4`. The table now gates that
rung to large batches so LFM2 DFlash blocks fall through to the scalar
`attention_dflash_f32` path; the fixed `dflash_smoke --ctx 8 --block 4` run
completed with finite output in `4.85 ms`.

The block-teacher evaluator now runs on the current dump without segfaulting:

```text
HIP_VISIBLE_DEVICES=0 HIPFIRE_OQ4_PREFILL_ACT_BITS=8 \
  CARGO_TARGET_DIR=/tmp/hipfire-target-lfm2-dflash-smoke \
  cargo run -p hipfire-arch-lfm2moe --features deltanet \
  --example lfm2_dflash_block_teacher_eval -- \
  --model /srv/huggingface/_Hipfire/lfm2.5-350m-oq4plus-smoke.hfq \
  --draft /tmp/lfm2-dflash-block-ce-sidecar-smoke.hfq \
  --teacher-dump /tmp/lfm2-dflash-teacher-dump-current \
  --max-blocks 2
```

Result: `slots=6`, `argmax_hits=0`, `token_hits=0`, `topk_hits=0`,
`weighted_ce=2.9073215160743837`, `forward_ms=41.425646`.

The same prompt through `lfm2_dflash_seed_smoke --block 4` now passes and the
multi-token acceptance evaluator remains stable but unaccepted:
`tokens=8`, `cycles=7`, `accepted=0`, `drafted=21`, `accept_rate=0.0`.

The FC-only ridge fitter is stable and reconstructs the saved target hidden
rows on its training dump (`train_mse=9.934724e-7`). On the eight-block replay
set, it improved the block-teacher weighted CE to `2.31385312877755`, but
agreement remained zero: `slots=24`, `argmax_hits=0`, `token_hits=0`,
`topk_hits=0`. The acceptance evaluator also remained at `accepted=0`,
`drafted=21`, `accept_rate=0.0`, with the generated preview still degenerate.

The hidden-label smoke confirms the failure is not only a sampled-logit
calibration issue. Replaying `/tmp/lfm2-dflash-fcfit8-sidecar-smoke.hfq` on the
new two-block hidden dump produced the same top-k/argmax miss pattern and
reported draft-vs-target normalized hidden agreement near zero:
`hidden_rows=6`, `hidden_mse=12.859830075984002`,
`hidden_cosine=0.0011303966325458444`, `weighted_ce=2.5193043849059533`.

Fitting only the deployed final `norm.weight` is enough to move the hidden
metric, which proves the new label path is actionable, but it is not enough for
admission. On the two-block smoke, the norm fit improved training-set hidden
MSE from `1.288231e1` to `7.012918e0`, hidden cosine from `-2.152683e-3` to
`6.507086e-1`, replay weighted CE from `2.5193043849059533` to
`1.8626029608166237`, and top-k hits from `0/6` to `5/6`.

On the eight-block hidden dump, the same norm-fit path improved training-set
hidden MSE from `1.317971e1` to `9.294223e0` and hidden cosine from
`-3.362994e-2` to `4.717041e-1`. Replay improved weighted CE to
`2.021177649962546`, but agreement was still weak: `argmax_hits=0/24`,
`topk_hits=1/24`. End-to-end speculative acceptance remained zero on the same
prompt: `accepted=0`, `drafted=21`, `accept_rate=0.0`.

The final `down_proj` fit is a useful negative result. It solved the fitted
delta objective on the eight-block dump (`delta_mse=3.600996e-1`), but replayed
with the original final norm it did not improve hidden/logit agreement:
`weighted_ce=2.5264846900092746`, `hidden_mse=13.179692737705372`,
`hidden_cosine=-0.033627470964076527`, `topk_hits=0/24`. Applying the same
`norm.weight` fit after the down fit reproduced the norm-only behavior rather
than improving it: `weighted_ce=2.0230095988201726`, `hidden_mse=9.294611454195316`,
`hidden_cosine=0.4716675585209639`, `topk_hits=1/24`, and acceptance still
`accepted=0`, `drafted=21`.

A small `norm.weight` scale sweep found the first non-zero LFM2 DFlash
acceptance on this host. The conservative clamp `--max-scale 1.0` produced
weaker hidden reconstruction than `4.0` or `8.0`, but better speculative
behavior:

| Norm fit | Hidden cosine | Weighted CE | Top-k hits | Acceptance |
|---|---:|---:|---:|---:|
| `max_scale=1.0` | `0.4092986461817538` | `1.9778001620204941` | `3/24` | `accepted=3`, `drafted=18`, `accept_rate=0.16666666666666666` |
| `max_scale=2.0` | `0.4450917031375716` | `1.9986575214446125` | `2/24` | `accepted=0`, `drafted=21`, `accept_rate=0.0` |
| `max_scale=8.0` | `0.479634392795824` | `2.0721855831259943` | `2/24` | `accepted=0`, `drafted=21`, `accept_rate=0.0` |

This narrows the failure further: reconstructing hidden space monotonically is
not the same as improving speculative acceptance. The next DFlash training step
should optimize the deployed logit/acceptance objective directly, or train the
attention/MLP stack with an explicit held-out acceptance probe, then re-run
block-teacher and acceptance gates before treating any sidecar as admitted.

`lfm2_dflash_fit_norm` now automates the clamp sweep with
`--scan-max-scale 1,2,4,8`. The fitter scores each candidate through the real
target LM head against saved teacher top-k labels, then selects by max weighted
top-k rate, min weighted CE, and max hidden cosine. On the eight-block hidden
dump, the automated scan selected the same conservative `max_scale=1.0`
candidate and wrote
`/tmp/lfm2-dflash-fcfit8-normscan8-sidecar-smoke.hfq`.

Post-pull replay of that selected sidecar reproduced the block-teacher numbers:
`slots=24`, `argmax_hits=0`, `topk_hits=3`, `weighted_ce=1.9778001620204941`,
`hidden_mse=10.386262598544578`, `hidden_cosine=0.4092986461817538`. However,
the current acceptance evaluator did not reproduce the earlier non-zero
acceptance on the default prompt after the pull. With normal EOS handling, the
target stopped immediately on the first greedy token (`tokens=0`, `cycles=0`).
With `--ignore-eos --block 4`, the same sidecar exercised the DFlash path but
reported `first_token=7`, `first_token_is_eos=true`, `tokens=8`, `cycles=7`,
`accepted=0`, `drafted=21`, `accept_rate=0.0`. The acceptance evaluator now
emits the first greedy seed token, EOS classification, block size, context
slice, and `ignore_eos` state in each prompt result and summary so EOS-only
probes are not mistaken for DFlash acceptance evidence. A three-prompt
plain-text probe likewise remained at `accepted=0`, `drafted=24`. Treat the
earlier `accepted=3/18` result as a useful lead, not admission evidence, until
the acceptance probe is reproducible on a held-out prompt set.

The teacher dump now supports `--position-mode prefix|spread` and explicit
`--positions`, plus `--prompt-mode separate` for independent multi-prompt
generation-boundary dumps. Multi-prompt dumps concatenate the raw tensor rows
but record `prompt_offsets`, `prompt_lengths`, and per-block `prompt_indices`,
so block replay and fit tools slice the correct prompt-local hidden-prefix table
before calling the deployed DFlash runtime. Valid block starts now match the
runtime prefix contract: any `position` in `1..=prompt_rows` is legal because
the draft only needs prefix hidden rows. This matters for acceptance evidence
because the first speculative cycle starts at the post-prefill boundary, not
inside the prompt.

Two 2026-06-26 probes make that distinction clear:

| Probe | Artifact | Result |
|---|---|---|
| Spread prompt windows | `/tmp/LFM2.5-350M.dflash.fcfit-normscan-spread8.oq4+.hfq` from `/tmp/lfm2-dflash-teacher-dump-spread8` | block replay improved to `argmax_hits=14/24`, `topk_hits=19/24`, `weighted_ce=1.519566820625048`; same-prompt EOS loop can accept under `--ignore-eos`, but two non-EOS prompts still had `accepted=0`, `drafted=42` |
| Generation-boundary window | `/tmp/LFM2.5-350M.dflash.fcfit-normscan-genstart-testcode.oq4+.hfq` from `/tmp/lfm2-dflash-teacher-dump-genstart-testcode` with `--positions 11` | block replay at `position=11`, `seed_token=523` hit `argmax_hits=3/3`, `topk_hits=3/3`, `weighted_ce=1.2733865757292784`; real acceptance on the same non-EOS prompt reached `accepted=3`, `drafted=12`, `accept_rate=0.25` |
| Two-prompt generation-boundary windows | `/tmp/LFM2.5-350M.dflash.fcfit-normscan-genstart-2prompt.oq4+.hfq` from `/tmp/lfm2-dflash-teacher-dump-genstart-2prompt` with `--prompt-mode separate --position-mode generation` | metadata recorded `prompt_offsets=[0,11]`, `prompt_lengths=[11,5]`, `prompt_indices=[0,1]`, `positions=[11,5]`; block replay hit `argmax_hits=4/6`, `topk_hits=6/6`, `weighted_ce=1.47687549959154`; real acceptance on the two non-EOS prompts reached `accepted=12`, `drafted=15`, `accept_rate=0.8` |

The fit/eval tools now have explicit split controls so overfit progress can be
separated from held-out evidence:

- `lfm2_dflash_fit_fc --skip-rows N --max-rows M`
- `lfm2_dflash_fit_norm --skip-blocks N --max-blocks M`
- `lfm2_dflash_fit_norm --score-skip-blocks N --score-max-blocks M`
- `lfm2_dflash_fit_down --skip-blocks N --max-blocks M`
- `lfm2_dflash_block_teacher_eval --skip-blocks N --max-blocks M`

A train-one/hold-one split on the two-prompt dump is the first useful negative
held-out result. Training only prompt 0 rows and block 0 wrote
`/tmp/LFM2.5-350M.dflash.fcfit-normscan-genstart-train1.oq4+.hfq`. Replay on
the training block remained perfect (`argmax_hits=3/3`, `topk_hits=3/3`,
`weighted_ce=1.2733865757292784`), but replay on held-out block 1 failed
(`argmax_hits=0/3`, `topk_hits=0/3`, `weighted_ce=4.883694050832167`,
`hidden_cosine=0.10694146165558616`). End-to-end acceptance on both prompts
dropped to `accepted=3`, `drafted=33`, `accept_rate=0.09090909090909091`, with
the held-out prompt at `accepted=0`, `drafted=21`.

This confirms the previous two-prompt sidecar is still overfit smoke, not
admission evidence. The actionable result is that LFM2 DFlash training needs
more generation-boundary teacher windows across multiple prompts plus held-out
block-teacher and acceptance gates; prompt-internal block replay alone is not a
sufficient proxy for speculative acceptance.

`lfm2_dflash_fit_norm` now separates the fit and candidate-selection ranges.
`--skip-blocks/--max-blocks` still select the blocks used to solve the diagonal
norm scale, while `--score-skip-blocks/--score-max-blocks` optionally select a
different replay range for candidate scoring and metadata. When score flags are
omitted, scoring defaults to the train range to preserve the earlier smoke
workflow. When they are present, the selected clamp is chosen by held-out score
weighted top-k rate, weighted CE, and hidden cosine rather than by the fit rows.

Held-out score smoke on the two-prompt generation-boundary dump trained the
diagonal norm scale on block 0 and selected the clamp on held-out block 1:

```text
lfm2_dflash_fit_norm \
  --skip-blocks 0 --max-blocks 1 \
  --score-skip-blocks 1 --score-max-blocks 1 \
  --scan-max-scale 1,2,4,8
```

The scorer selected `max_scale=1.0`, wrote
`/tmp/LFM2.5-350M.dflash.fcfit-normscore-genstart-train1-heldout.oq4+.hfq`,
and recorded `train_topk_hits=3/3`, `train_weighted_ce=1.6965071768019788`,
`score_topk_hits=0/3`, `score_weighted_ce=2.0648041856145696`,
`score_hidden_cosine=0.1259602825514884`. Replaying held-out block 1 with
`lfm2_dflash_block_teacher_eval --skip-blocks 1 --max-blocks 1` reproduced the
score result: `argmax_hits=0/3`, `topk_hits=0/3`,
`weighted_ce=2.0648042145897767`, `hidden_cosine=0.12596028222690106`.

An eight-prompt generation-boundary split gives a clearer held-out signal. The
raw plain-text prompt dump
`/tmp/lfm2-dflash-teacher-dump-genstart-8prompt` recorded
`prompt_lengths=[13,11,11,12,10,9,18,11]`; the first six prompts were used for
FC/norm fitting and the last two blocks for scoring. The held-out scorer
selected `max_scale=2.0` and wrote
`/tmp/LFM2.5-350M.dflash.fcfit-normscore-genstart-8prompt-train6-score2.oq4+.hfq`.
Replay showed strong training fit but weak held-out agreement:
`train_argmax_hits=14/18`, `train_topk_hits=15/18`,
`train_weighted_ce=1.117955096017171`; held-out was `argmax_hits=0/6`,
`topk_hits=1/6`, `weighted_ce=2.100889188306698`,
`hidden_cosine=0.22785523453463097`. The acceptance probe reported
`accepted=24`, `drafted=108`, `accept_rate=0.2222222222222222`, but this is not
good admission evidence because 7 of 8 raw prompts had target first token
`<|im_end|>` and the run required `--ignore-eos`.

Repeating the same split with hand-rendered ChatML prompts produced a
non-degenerate acceptance surface. The dump
`/tmp/lfm2-dflash-teacher-dump-genstart-chat8` recorded
`prompt_lengths=[21,19,19,20,18,17,26,19]`. The FC fit on the first six prompts
wrote `/tmp/LFM2.5-350M.dflash.fcfit-genstart-chat8-train6.oq4+.hfq`
(`train_mse=1.766190e-6`). Held-out norm scoring selected `max_scale=8.0` and
wrote
`/tmp/LFM2.5-350M.dflash.fcfit-normscore-genstart-chat8-train6-score2.oq4+.hfq`.
The final candidate improved held-out hidden alignment and top-k agreement over
both the seed and FC-only sidecars, but still had zero held-out argmax hits:

| ChatML held-out sidecar | Hidden cosine | Weighted CE | Top-k hits | Acceptance |
|---|---:|---:|---:|---:|
| Seed block-CE | `-0.012851896592319516` | `2.4695703137829508` | `0/6` | `accepted=0`, `drafted=168`, `accept_rate=0.0` |
| FC-only | `-0.012219819159913823` | `2.811207198352154` | `0/6` | `accepted=0`, `drafted=168`, `accept_rate=0.0` |
| FC + held-out norm score | `0.3323275588935704` | `2.5780603002137776` | `3/6` | `accepted=7`, `drafted=147`, `accept_rate=0.047619047619047616` |
| FC + norm + logit-bias probe (`epochs=8`, `lr=1.0`, `max=8`, demote) | `0.3323275588935704` | `2.5938663075538444` | `1/6` | `accepted=10`, `drafted=138`, `accept_rate=0.07246376811594203` |
| FC + norm + logit-bias probe (`epochs=8`, `lr=0.5`, `max=4`, no demote) | `0.3323275588935704` | `3.7993563908232058` | `4/6` | `accepted=14`, `drafted=126`, `accept_rate=0.1111111111111111` |
| FC + held-out norm/logit grid (`epochs=8`, `lr=0.5`, `max=2`, no demote) | `0.3323275588935704` | `3.0941116705143004` | `4/6` | `accepted=13`, `drafted=195`, `accept_rate=0.06666666666666667` |

`DflashWeights` now accepts an optional `logit_bias.weight` tensor for this
training probe, and the LFM2 DFlash bridge adds it after the target `lm_head`.
This is not part of the intended upstream draft architecture; it exists to test
whether a cheap deployed-logit correction can move acceptance before investing
in deeper draft training. The first aggressive demoting bias overfit train
agreement (`train_topk` moved `9/18 -> 13/18`) and worsened held-out top-k
(`3/6 -> 1/6`), but still improved acceptance to `10/138`. A small sweep found
the best acceptance point at `epochs=8`, `lr=0.5`, `max=4`, `--no-logit-bias-demote`:
held-out top-k improved to `4/6`, but weighted CE degraded to `3.7993563908232058`.
Gentler demoting variants preserved held-out top-k at `3/6` and reached
`accepted=12/135` (`epochs=4`, `lr=0.25`, `max=2`) or `12/132` (`epochs=4`,
`lr=0.5`, `max=4`).

The ChatML result is progress over the seed/FC-only baselines on this host, not
an admitted DFlash sidecar. It still fails the practical gate because held-out
argmax is `0/6`; the best acceptance probe is only `14/126` and has poor
held-out CE. The next training step should expand generation-boundary coverage
and optimize the deployed logit/acceptance objective with a real held-out set;
the current diagonal norm and logit-bias probes mostly prove which labels and
runtime hooks are worth keeping for the next trainer.

The held-out scan is now integrated into `lfm2_dflash_fit_norm` rather than
requiring manual sidecar sweeps. A 2026-06-26 rerun on
`/tmp/lfm2-dflash-teacher-dump-genstart-chat8` trained on blocks `0..6`, scored
on blocks `6..8`, scanned `--scan-max-scale 1,2,4,8`, and swept
`--scan-logit-bias-epochs 4,8 --scan-logit-bias-lr 0.25,0.5
--scan-logit-bias-max 2,4 --scan-logit-bias-demote true,false`. It selected
`max_scale=8` and the gentler no-demote bias candidate
(`epochs=8`, `lr=0.5`, `max=2`) because it preserved the best held-out top-k
while lowering held-out CE versus the more aggressive no-demote candidates.
The output
`/tmp/LFM2.5-350M.dflash.fitnorm-logitgrid-genstart-chat8-train6-score2.oq4+.hfq`
replayed as `train_argmax_hits=6/18`, `train_topk_hits=15/18`,
`train_weighted_ce=2.0112734864721213`; held-out replay was
`argmax_hits=0/6`, `topk_hits=4/6`, `weighted_ce=3.0941116705143004`,
`hidden_cosine=0.3323275588935704`. A comparable ChatML acceptance run with
`--block 4` and the default `--max-tokens 16` reported `accepted=13`,
`drafted=195`, `accept_rate=0.06666666666666667`. This confirms the integrated
selector is useful tooling, but it also confirms held-out argmax is still zero
and the sidecar is not admitted.

A larger 32-prompt ChatML-style corpus now lives at
`benchmarks/prompts/lfm2_dflash_chat32.txt`
(`md5=7ac0ca056c2005ab709d5584ab175f8e`) and the first result bundle is
`benchmarks/results/lfm2-dflash-chat32-20260626T143023Z/`. The teacher dump
`/tmp/lfm2-dflash-teacher-dump-chat32` has 578 prompt rows and 32
generation-boundary DFlash blocks. `lfm2_dflash_fit_fc` fit all prompt rows
from the seed block-CE sidecar and wrote
`/tmp/LFM2.5-350M.dflash.fcfit-chat32.oq4+.hfq`
(`train_mse=2.711137e-6`). The integrated norm/logit-bias grid then trained on
blocks `0..24`, scored on blocks `24..32`, selected `max_scale=4`, and selected
a no-demote bias candidate (`epochs=4`, `lr=0.25`, `max=4`). Held-out score
moved from `argmax=4/24`, `topk=8/24`, `weighted_ce=2.056471` before bias to
`argmax=7/24`, `topk=15/24`, `weighted_ce=2.255824` after bias. Independent
block replay of the written sidecar matched that held-out result:
`argmax_hits=7/24`, `topk_hits=15/24`, `weighted_ce=2.255824135858777`,
`hidden_cosine=0.3056602010578631`. End-to-end acceptance on the same 32 prompts
improved from the base draft's `accepted=0`, `drafted=900`, `accept_rate=0.0`
to `accepted=84`, `drafted=657`, `accept_rate=0.1278538812785388`. This is the
best current LFM2 DFlash evidence because it uses a larger held-out split and a
zero-acceptance baseline, but it is still not admission quality: held-out argmax
is low and decoded previews remain mostly repetitive punctuation/fragments.

The same 32-prompt bundle also tested a final-layer `down_proj` fit and a
demoting logit-bias variant. `lfm2_dflash_fit_down` solved
`down_proj(gate_up) ~= dflash_block_target_hidden - residual_ffn` on the
training split and wrote
`/tmp/LFM2.5-350M.dflash.fcdownfit-chat32-train24.oq4+.hfq`
(`delta_mse=1.785100e-1`). Re-fitting norm/logit bias after that selected
`max_scale=2` and a no-demote bias candidate (`epochs=4`, `lr=0.25`, `max=4`).
Independent replay was slightly better on held-out CE but not acceptance:
`argmax_hits=7/24`, `topk_hits=15/24`, `weighted_ce=2.2380222506556633`,
`hidden_cosine=0.29739164466146817`, with end-to-end
`accepted=85`, `drafted=669`, `accept_rate=0.12705530642750373`. A fixed
demoting-bias pass (`epochs=4`, `lr=0.5`, `max=2`) improved held-out CE to
`1.7491628097505187`, but reduced held-out top-k to `14/24` and acceptance to
`accepted=57`, `drafted=726`, `accept_rate=0.07851239669421488`. Treat both
branches as negative/neutral evidence; neither should be promoted.

## Follow-ups

- Scale the HFHS collection beyond the one-tensor smoke sidecar so
  `--format oq4++ --hessian` can produce a real full-model Hessian-calibrated
  `oq4++` artifact. For first-plus `oq4+`, collect imatrix/Hessian-diagonal
  coverage for the AWQ pass without requiring LDLQ.
- Run quality evidence against a BF16 or accepted high-precision reference
  before promoting `oq4+`.
- Calibrate and gate real LFM2 CASK/TriAttention sidecars using the
  attention-ordinal sidecar convention.
- Train the LFM2 DFlash sidecar against generation-boundary teacher windows
  from multiple prompts, then require held-out non-zero block-teacher agreement
  and reproducible non-zero acceptance before admission gating.
- Repeat benches with a dedicated multi-run bench harness and record variance.
