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

## Recommendation

Adopt the teacher/student split as the standing architecture: **daemon = frozen
forward/capture engine for any arch; `hipfire-train` = differentiable trainer for
small students.** First concrete step: the daemon training-label capture op for
the drafter P5 (above), which is mostly wiring over existing `collect` /
`CalibCollector` / pflash K-capture. Supersedes
`2026-06-18-qwen35-training-support.md` for the drafter; that plan's full qwen3.5
forward port is only needed if/when we want to *train qwen3.5 itself*.
