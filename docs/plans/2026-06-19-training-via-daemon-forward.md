# Plan: training that reuses the daemon's forward path (teacher/student split)

Status: EXPLORATION (2026-06-19). Prompted by: "this is why I wanted to build the
training system into the daemon — can you explore that as an option?"

## The tension and the resolution

The naive read of "merge training into the daemon" hits the wall the user already
named: the daemon's forward uses **fused, quantized, inference kernels with no
backward**, so you cannot backpropagate a trained model through them. But the goal
doesn't require that. Split the problem:

- **Teacher (forward-only):** run ANY model — including the qwen3.5 hybrid
  (DeltaNet/SSM + full-attn) — via the daemon's existing inference kernels and
  **capture** activations / K / logits / scores. No gradients needed; the teacher
  is frozen.
- **Student (differentiable):** train only the SMALL, tractable model (the PFlash
  drafter; recovery LoRA/norms) in `hipfire-train`'s un-fused fp32 forward+backward.
  It consumes the teacher signals the daemon captured.

This is exactly the right division: it kills the need to re-implement qwen3.5's
SSM forward in `hipfire-train` (a multi-session port), while keeping a clean
differentiable path for the thing we actually optimize.

## What already exists in the daemon (reuse, don't rebuild)

- **`collect` JSONL op** (`hipfire-daemon/src/main.rs:9530`) — forwards the
  resident qwen3.5 model (`q35_weights`/`q35_config`/`tokenizer`) over a corpus.
  Today it drives calibration; it is a working "forward qwen3.5 over text" path.
- **`CalibCollector`** (`hipfire-runtime/src/calibration.rs`) — forward-hook
  activation capture armed via `gpu.active_capture`; stages per-tensor activation
  rows on GPU. General mechanism, currently specialized to Hessian/imatrix.
- **PFlash K-capture** (`hipfire-arch-qwen35/src/pflash.rs`) — already captures K
  at `score_layer_idx` (=3, the first full-attention layer for qwen3.5) and runs a
  **partial forward that skips ~80% of the stack**. `compute_scores_batched_gpu`
  even produces the per-block cosine scores directly.

So ~80% of the qwen3.5 "teacher" is built: it can forward to a chosen layer and
emit K / block scores.

## Concrete design — PFlash drafter P5 (real qwen3.5 target)

1. **Daemon: a training-label capture op.** Generalize the PFlash K-capture into a
   JSONL op that, for a corpus + a layer index, emits the qwen3.5 target's K (or
   the block-cosine scores) per chunk. Load `qwen3.5-0.8b-bf16.hfq` (exists) for a
   high-precision teacher. Reuses the partial-forward + `active_capture` plumbing.
2. **Bridge: persist labels.** Write the captured K/scores to the same v2 label
   cache the drafter trainer already reads (`checkpoint.rs::save_labels`), keyed by
   target+corpus+geometry.
3. **Student: drafter trains unchanged.** `pflash_drafter_train` consumes the
   cached labels exactly as today — but now they come from the REAL qwen3.5 target
   instead of the Llama-3.2-3B stand-in. The drafter still shares the target's
   embedding (read-only) and trains its own small attention layers.

Net: replaces the stand-in with the real target by adding ONE daemon capture op,
not by porting the SSM forward. Removes the need for the qwen3.5-training-support
plan's Q1–Q4 entirely *for the drafter use case*.

## Where the split does NOT help (be honest)

- **Recovery-FT / QAT of a quantized model** (the QTIP work) needs gradients
  through the model being recovered — the daemon's inference forward can't provide
  that. That stays in `hipfire-train`'s differentiable path (which is why it
  exists). The teacher/student split only removes the need to *re-implement a
  frozen teacher's forward*, not the need for a differentiable student.
- If we ever want to fine-tune qwen3.5 itself (not just a drafter), we still need
  a differentiable qwen3.5 forward — but that's a separate, later goal.

## STATUS (2026-06-19) — teacher VALIDATED on the real qwen3.5-0.8B

**Teacher half done.** Built + validated end-to-end:
- `qwen35::capture_pflash_block_scores` (+ `full_attention_layers`) — forward the
  resident qwen3.5 target via the per-token `forward_scratch` path (FP32 KV + FP32
  DeltaNet state), then run the fp32 `pflash_score_f32` kernel on `k_gpu` at each
  FullAttention layer. Only FullAttention layers carry K (linear_attn = SSM).
- Daemon `pflash_labels` JSONL op — forward a corpus (n_chunks × seq), emit
  per-chunk per-block cosine-K scores at shallow + mid FullAttention layers to
  JSONL.
- **Validated on `qwen3.5-0.8b-bf16.hfq`:** loads (24 layers, 6/24 = FullAttention
  carry KV), shallow=L3 mid=L15, 8 valid cosine scores/block in [0.35,0.77], no
  NaN, Spearman(shallow,mid)=+0.69/+0.48 — the same imperfect shallow→mid tracking
  (= drafter headroom) seen on the Llama stand-in, now on the REAL target.

**Student half remaining (well-scoped):**
1. hipfire-train: a daemon-label loader (read the JSONL → chunks + mid/shallow
   scores) that fills the same structures `pflash_drafter_train` already trains on
   (and/or writes them into the v2 label cache via `save_labels`).
2. The qwen3.5 fp32 **embedding** into hipfire-train (the drafter shares it,
   read-only). Cleanest: extend `pflash_labels` to also dump `embed_tokens` as an
   fp32 sidecar binary (daemon already has the weights); trainer mmaps it. Avoids
   teaching hipfire-train to parse the qwen3.5 `.hfq`.
3. Point `pflash_drafter_train` at the daemon labels + qwen3.5 embed instead of the
   Llama-3.2-3B stand-in capture. Train; confirm it beats the shallow bar on the
   REAL target.

## Recommendation

Adopt the teacher/student split as the standing architecture: **daemon = frozen
forward/capture engine for any arch; `hipfire-train` = differentiable trainer for
small students.** First concrete step: the daemon training-label capture op for
the drafter P5 (above), which is mostly wiring over existing `collect` /
`CalibCollector` / pflash K-capture. Supersedes
`2026-06-18-qwen35-training-support.md` for the drafter; that plan's full qwen3.5
forward port is only needed if/when we want to *train qwen3.5 itself*.

## P5 RESULT + the SSM-drafter signal (2026-06-19)

Ran the full pipeline on the REAL qwen3.5-0.8B target (daemon captured 40 chunks +
1GB embed sidecar; drafter trained, zero SSM forward in hipfire-train). Pipeline
works. But the tiny-ATTENTION drafter underperforms badly:

| target | nature | shallow bar | attn drafter best |
|--------|--------|-------------|-------------------|
| Llama-3.2-3B | dense attention | +0.595 | **+0.676 ✓** |
| qwen3.5-0.8B | gated-delta-net (SSM) hybrid | +0.702 | +0.47 ✗ |

Tuning sweep on the cached real-target labels (wd/lr/epochs/tau) all plateaued at
+0.36–0.47 — and MORE weight decay made it WORSE. That rules out overfit: the
drafter is UNDERFITTING the task. Real architectural ceiling (~+0.47), tuning-
resistant, far below the +0.702 bar.

**Conclusion:** the attention drafter matches an attention target but hits a wall
on an SSM-driven target — empirical support for a **gated-delta-net drafter**
(shares the target's inductive bias). Next: build a tiny GDN drafter (student-only
change; daemon labels unchanged) and test head-to-head vs the +0.47 attn ceiling.
Confounds noted but weak: higher bar (less headroom) and the attention-shaped
cosine-K metric / L15 choice.
