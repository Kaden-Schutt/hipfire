# Merge the trainer into the daemon: `train_drafter` as a daemon op

**Status:** design map (2026-06-19). Not yet implemented.

## Why

The PFlash/SSM-drafter trainer and the daemon are two separate processes that
each open a HIP context and serialize on the GPU by hand. That two-process split
is the *only* reason the GPU-locking problem exists (see
`2026-06-19`-era locking discussion: four un-unified lock mechanisms). Collapse
trainer + daemon into **one process, one HIP context** and:

- **The locking problem dissolves for training.** One process → the daemon's
  existing `acquire_daemon_lock` flock (main.rs) is the only lock. No leaf crate,
  no `Gpu::init` lease, no `scripts/gpu-lock.sh` for the train path.
- **No 1 GB sidecar round-trip.** Today: daemon captures → `dump_embed_fp32`
  writes a **1 GB** `QEMB` embed sidecar → trainer process boots → re-reads 1 GB →
  uploads. In-process, the drafter reads the target's embedding straight from
  VRAM. Capture → train in one resident pass.
- **No target reload between data-scaling rounds.** The qwen3.5 target stays
  resident across capture and training.

This is the project's stated direction (teacher/student split:
"daemon = frozen forward/capture engine; hipfire-train = differentiable trainer
for small students"). Half of it already shipped — `pflash_labels` is a daemon
op. This finishes it.

## Why it's architecturally clean (grounded findings)

- **Same GPU type both sides.** Daemon: `let mut gpu = rdna_compute::Gpu::init()`
  (main.rs:2332/2365). hipfire-train ops: `gpu: &mut rdna_compute::Gpu`
  (e.g. `ssm_drafter.rs`). Identical type — the daemon passes its `&mut gpu`
  straight into `ssm_drafter_forward_train` / `_backward`. **No bridging.**
- **No dependency cycle.** hipfire-train → {rdna-compute, hipfire-runtime,
  hipfire-model}. Daemon → {rdna-compute, hipfire-runtime, arch crates}.
  Adding `hipfire-train` to the daemon's `Cargo.toml` is acyclic (hipfire-train
  does not depend on hipfire-daemon).
- **Working precedent.** `pflash_labels` op (main.rs:4452) already does in-daemon
  GPU work against the resident model: `model.as_ref()` →
  `m.q35_weights / m.q35_config / m.tokenizer`, then
  `qwen35::capture_pflash_block_scores(&mut gpu, weights, config, &toks, block,
  &[shallow, mid])`. The training op is the same shape, one stage further.
- **Multi-context drafter machinery already exists.** The daemon has
  `pflash_drafter_gpu: Option<rdna_compute::Gpu>` (main.rs:2391) for hetero PFlash
  (drafter on a sibling device). Training can reuse that pattern to optionally
  train on a sibling GPU while the target serves on the primary.

## What moves, what stays

**Stays in the `hipfire-train` lib (unchanged, still gradcheckable standalone):**
all differentiable logic — `SsmDrafter`/`Drafter`, the fwd/bwd ops (`gated_scan`,
`sigmoid`, `ssm_block`, `linear`, `rmsnorm`, `swiglu`, `pflash_score`), `AdamW`,
`checkpoint`, the ListNet loss helpers. The `gradcheck_*` examples stay
standalone (fast iterate, no daemon, no resident target needed).

**Extracted to a lib fn** (so there is ONE loop, called by both the op and the
standalone example):
```rust
// hipfire-train/src/train_loop.rs
pub struct DrafterTrainReport { pub best_eval: f32, pub best_epoch: usize, pub bar: f32, pub final_eval: f32 }
pub fn train_drafter_loop(
    gpu: &mut rdna_compute::Gpu,
    drafter: &mut SsmDrafter,            // or an enum over {Ssm, Attention}
    chunks: &[Vec<u32>], label_mid: &[Vec<f32>], base_shallow: &[Vec<f32>],
    cfg: &TrainCfg,
    mut on_epoch: impl FnMut(usize, f32, f32),   // (epoch, train_loss, eval) — progress hook
) -> HipResult<DrafterTrainReport>
```
The current `ssm_drafter_train.rs` main() collapses to: load labels → build
drafter → `train_drafter_loop(.., |ep,l,e| println!(...))`. The daemon op calls
the same fn with `on_epoch` emitting JSONL progress.

**Moves into the daemon:** only the *orchestration* — label sourcing (capture vs
file), the embed GpuTensor wiring, progress streaming, checkpoint I/O — as a new
`train_drafter` op handler next to `pflash_labels`. New daemon dep:
`hipfire-train = { path = "../hipfire-train" }`.

## Op surface (JSONL protocol)

Request:
```json
{"type":"train_drafter",
 "arch":"ssm",
 "config":{"h_draft":512,"n_layers":3,"inter":1024,"n_kv":4,"head_dim":64},
 "labels":{"source":"capture","corpus":"/path/corpus.txt","n_chunks":100,
           "seq":512,"block":64,"n_eval":20,"shuffle_seed":24077},
 "train":{"epochs":300,"lr":1e-3,"wd":0.0,"tau":0.1,"eval_every":15},
 "output":"~/.hipfire/drafters/qwen3.5-0.8b-ssm.drafter.hfq",
 "resume":true}
```
`labels.source`:
- `"capture"` — op calls `capture_pflash_block_scores` per chunk **in-process**
  against the resident target; builds the embed GpuTensor directly from resident
  weights (skips `dump_embed_fp32` + the 1 GB disk round-trip).
- `"file"` — load a pre-captured labels JSONL + `.embed.bin` (back-compat with the
  current sidecar; for offline / reproducible retrains).

Streamed responses:
```json
{"type":"train_progress","epoch":30,"train_loss":1.674,"eval":0.504,"best":0.554,"best_epoch":15}
{"type":"train_done","best_eval":0.554,"best_epoch":240,"bar":0.702,"beat_bar":false,
 "checkpoint":"~/.hipfire/drafters/qwen3.5-0.8b-ssm.drafter.hfq"}
```

## Embedding access (the 1 GB win)

The resident qwen3.5 model already holds the **F32** `token_embd` in VRAM
(`embd_fmt = EmbeddingFormat::F32`; it's what `capture_pflash_block_scores` reads).
The drafter uses it as a **frozen shared** embedding. In-process that's either a
zero-copy borrow of the resident `GpuTensor` or a single device-to-device copy —
versus today's write-1 GB-to-disk → read-1 GB → re-upload. `dump_embed_fp32` and
the `QEMB` sidecar stay only for the `source:"file"` offline path.

## Concurrency / lifecycle

- **v1:** the op runs inline in the daemon's stdin loop, blocking it for the run's
  duration — exactly like `pflash_labels` (already blocks ~80 min during a 100-chunk
  capture). Acceptable for a single-tenant dev daemon.
- **v2:** run on a worker thread with a cancel channel; honor the existing `abort`
  op so the daemon stays responsive and the run is interruptible. Checkpoint on
  abort.
- **Lock:** one process → `acquire_daemon_lock` flock is the whole story.
- **VRAM budget:** target (~1.5 GB bf16) + drafter (8–22 M params f32) + AdamW
  state (2× params) + activations (seq 512 × h_draft) co-resident — trivial on the
  45 GB dev box. Small cards: optional sibling-device drafter via the existing
  `pflash_drafter_gpu` pattern.

## Checkpoint format

Reuse `hipfire-train/src/checkpoint.rs` (`save_drafter`), but:
- add `save_ssm_drafter` / generalize over the drafter enum (current `save_drafter`
  is typed to the attention `Drafter`),
- emit under the artifact naming convention:
  `qwen3.5-0.8b-ssm.drafter.hfq` (`<family>-<ver>-<size>-<variant>.drafter.hfq`),
  so PFlash P6 can load it as a drafter sidecar later.

## Migration (incremental — each step ships + is verifiable)

1. **Plumbing.** ✅ DONE (commit beba0214)
    Add `hipfire-train` dep; add a `train_drafter` op that validates
   args and returns `"not implemented yet"`. Daemon still builds + passes the
   no-GPU CI subset.
2. **Lib loop extraction.** ✅ DONE (with step 1)
    Move the epoch loop from `ssm_drafter_train.rs` into
   `hipfire-train::train_drafter_loop`; the example calls it. No behavior change —
   re-run the shuffled sweep, confirm identical curve.
3. **Op: file-source training.** ✅ DONE — verified: op streams train_start/
   progress/done + checkpoints; bar matches the example exactly, ep0/15 eval within
   GPU-atomic noise. ONE shared loader (hipfire-train::labels) + train_loop.
    `train_drafter` with `labels.source:"file"` loads
   the existing JSONL+sidecar and runs `train_drafter_loop`, streaming progress +
   saving a checkpoint. Verify the result matches the standalone example
   (byte-identical drafter checkpoint for a fixed seed).
4. **Op: capture-source.** Add `labels.source:"capture"` — capture in-process,
   build the embed GpuTensor from resident weights, drop the 1 GB round-trip.
   Verify labels byte-match the `pflash_labels` sidecar.
5. **Responsiveness.** Worker thread + `abort` + checkpoint-on-abort (v2).

## Risks / honest costs

- **Daemon relinks on train-loop edits.** Mitigated by keeping the loop in the lib
  (`train_drafter_loop`); lib edits are cheaper than editing daemon internals, but
  the daemon binary still relinks. Acceptable.
- **Blocking the daemon loop** during a long train until step 5.
- **A daemon crash kills the run.** Mitigated by checkpoint/resume (already in
  `checkpoint.rs`); `resume:true` reattaches.
- **Binary size.** +hipfire-train adds the optim/drafter/checkpoint code; the
  kernels are already in shared `rdna-compute`. Modest.

## Net

Training becomes `{"type":"train_drafter", ...}` against a resident target:
capture + train in one HIP context, one flock, zero sidecar round-trip. The
flock leaf-crate idea shrinks to a dev-tool GPU guard for the standalone
gradcheck/microbench examples only.
