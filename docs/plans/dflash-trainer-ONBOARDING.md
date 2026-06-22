<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Kaden Schutt
hipfire — see LICENSE and NOTICE in the project root.
-->
# MTP + DFlash training — contributor onboarding

**Branch:** `feat/mtp-dflash-training` (this branch). Built 2026-06-22 off current
`master` (`725e886d`). It bundles two things onto master:

1. **The from-scratch DFlash drafter trainer** (merge of `feat/dflash-lfm2-drafter`)
   — Rust/HIP-native, no PyTorch: forward + hand-coded reverse-mode backward +
   block-diffusion CE loss + AdamW + MFMA kernels + data-gen + `.dfnet`→`.hf4` bridge.
2. **The MTP-head autoregressive DDTree drafter scaffold** (cherry-pick of `e49eb4f2`
   from `feat/mtp-tree-drafter-q8ef`) — the inference-side path a trained drafter
   plugs into. Gated behind `HIPFIRE_MTP_AR_TREE=1` (default off).

> The full 148-commit `feat/mtp-tree-drafter-q8ef` branch was **not** merged — it
> collides with master's dispatch-unification refactor across 19 files / 69
> coherence-critical hunks (a re-port, not a merge). Only the self-contained MTP-AR
> scaffold was taken. See "Companion branch" below.

---

## The goal (and the design correction)

Train a DFlash speculative-decode drafter for **Qwen3.6-27B** that beats the
published z-lab baseline (online greedy τ ≈ 3.98).

**The drafter is a 5-layer, target-width (d=5120), target-initialized decoder — NOT
an LFM2-350M warm-start.** The LFM2.5-350M path (`Cfg::lfm2_350m`,
`load_lfm2_warmstart`, d=1024 conv body) was the cheap downscaled *spike* that proved
the trainer wiring; the capacity verdict found it underfit (25× `fc` compression).
Finetuning either LFM2-350M or the published z-lab drafter on the in-repo greedy
datagen is the **falsified** path — it does not beat baseline (more steps don't help,
naive more-data hurts). **Do not re-run that finetune.**

The right architecture is already in code:

```rust
// crates/hipfire-arch-lfm2moe/src/dflash_train.rs
Cfg::zlab_27b(vocab) // d=5120 (== d_tgt → identity in/out, fc [5120, 5×5120], NO
                     // compression), is_attn: vec![true;5] (5 dense-attn layers),
                     // GQA nh32/nkv8, hd128 — Qwen3.6-27B-native dims.
```

Loss / backward / Adam / kernels are dim-generic — rung-0 already pushed a published
z-lab d=5120 drafter through the trainer's eval and scored τ5.45, so the machinery is
proven at this width.

**The one missing piece** (grep-confirmed not built): a **target-init loader** —
`load_target_init(gpu, &Cfg::zlab_27b(vocab), qwen27b.hfq)` that seeds the 5 drafter
layers (q/k/v/o, gate/up/down, norms) from a strided slice of the **Qwen3.6-27B
target's own weights**, instead of `load_zlab` (pretrained z-lab ckpt) /
`load_lfm2_warmstart`. That loader + the bigger (~1.6B, z-lab-scale) training run are
the net-new work.

---

## File map

| Part | Path |
|---|---|
| Trainer lib (fwd+bwd+loss+Adam+loaders) | `crates/hipfire-arch-lfm2moe/src/dflash_train.rs` |
| Training kernels | `kernels/src/dflash_train.hip` (+ `conv1d_gated_batched.hip`) |
| Gradcheck (toy config) | `crates/hipfire-arch-lfm2moe/examples/dflash_body_gradcheck.rs` |
| Synthetic overfit | `…/examples/dflash_train_overfit.rs` |
| Real-data smoke | `…/examples/dflash_train_smoke.rs` |
| **Full training run** | `…/examples/dflash_train_run.rs` |
| Kernel unit tests | `crates/rdna-compute/examples/test_dflash_train_{kernels,bwd}.rs` |
| Data-gen: embed / lm_head | `crates/hipfire-arch-qwen35/examples/dflash_extract_{head,lmhead}.rs` |
| Data-gen: hiddens + tokens | `crates/hipfire-runtime/examples/{gen_qwen35_regen,dump_qwen35_hidden_states,gen_qwen35_bulk}.rs` |
| Convert `.dfnet`→deployable | `scripts/dfnet_to_hf.py` → `crates/hipfire-quantize/src/bin/dflash_convert.rs` |
| MTP-AR tree-drafter scaffold | `crates/hipfire-arch-qwen35/src/mtp_compose.rs::build_mtp_tree_ar` |
| Design / context | `docs/plans/dflash-trainer.md`, `docs/plans/dflash-lfm2-350m-spike.md`, this file |

---

## Build status (read before you `cargo run`)

The trainer is **work-in-progress code carried faithfully from the source branch**.
Two entry-point examples have **pre-existing breaks** (they do not compile on the
source branch either — this merge did not introduce them):

- **`dflash_train_run.rs`** calls `gpu.kl_topk_loss_bwd_f32(...)` (line ~437), but
  that **KL-loss backward kernel was never committed** — its definition exists
  nowhere in the repo. It was almost certainly an uncommitted local on the mi300 box,
  which has since been **torn down**, so the implementation is **lost**. Since the KL
  objective was disproven for greedy spec-decode (CE is the win), the pragmatic fix is
  to gate/remove the KL path in this example, or re-implement the kernel only if you
  intend to test sampled (rejection) spec-decode.
- **`dflash_train_smoke.rs`** initializes `dt::Net { … }` without the `hidden_norm`
  field (added to `Net` after smoke.rs was last touched). One-line fix:
  `hidden_norm: None,` in each `Net { … }` literal.

`dflash_body_gradcheck` and `dflash_train_overfit` use their own toy structs and
should build. The trainer **lib** and the MTP-AR scaffold compile past the merge
(see the merge commits for conflict resolutions; both were trivial).

---

## Run it (escalating validation ladder)

```bash
# 1. kernels correct
cargo run --release -p rdna-compute --example test_dflash_train_kernels
cargo run --release -p rdna-compute --example test_dflash_train_bwd
# 2. full-body backward vs finite-diff — run FP32 (see gotcha)
cargo run --release -p hipfire-arch-lfm2moe --example dflash_body_gradcheck
# 3. synthetic overfit → loss must collapse to ~0
cargo run --release -p hipfire-arch-lfm2moe --example dflash_train_overfit
# 4. real-data overfit of ONE block  (after the hidden_norm fix above)
cargo run --release -p hipfire-arch-lfm2moe --example dflash_train_smoke -- <head.f32> <hfhs> <kldref> [st_lfm2]
# 5. the real streaming run  (after the kl-kernel fix above)
cargo run --release -p hipfire-arch-lfm2moe --example dflash_train_run -- \
    <embed.f32> <lmhead.f32> <data_dir> <kldref> <out.dfnet> [steps] [lr] [n_train_chunks]
```
`data_dir` holds `chunk{c}.hfhs` (5 target layers, `SEL = [2,16,31,46,61]`); `kldref`
supplies the tokens per chunk.

---

## Data-gen (mi300's `/workspace` is gone — regenerate)

The trainer **code** is in-repo; the generated **data**, the trained checkpoints
(`ws-g-*.dfnet`), and the z-lab reference drafter (`/workspace/zlab-dflash`) lived on
mi300 and are gone. Regenerating needs a **Qwen3.6-27B `.hfq`** target:

```bash
# embed (tied == lm_head, for input gather) + faithful lm_head (output GEMM)
cargo run --release -p hipfire-arch-qwen35 --example dflash_extract_head    -- <qwen27b.hfq> embed.f32
cargo run --release -p hipfire-arch-qwen35 --features deltanet --example dflash_extract_lmhead -- <qwen27b.hfq> lmhead.f32
# AR target-regen: drafter trains on the target's OWN argmax outputs (hiddens + tokens)
cargo run --release -p hipfire-runtime --example gen_qwen35_regen -- \
    --model <qwen27b.hfq> --ref <kldref> --chunk C --n N --seed-len S \
    --out chunkC.hfhs --toks chunkC.toks [--kv-mode q8]
```
Convert `.dfnet`→deploy: `python scripts/dfnet_to_hf.py <out.dfnet> <hfdir>`
(⚠️ it hardcodes `ZLAB="/workspace/zlab-dflash"` for the frozen `hidden_norm` —
repoint to a real z-lab DFlash safetensors dir), then
`dflash_convert --input <hfdir> --output drafter.hf4 --mq4`; online-eval with
`dflash_spec_demo` under `HIPFIRE_VERIFY_GRAPH=0`.

---

## MTP-AR tree-drafter scaffold (the inference target)

`HIPFIRE_MTP_AR_TREE=1` routes `spec_step_ddtree_batched` through `build_mtp_tree_ar`
(best-first MTP-head heap expansion → DdTree, per-node KV-tape restore) instead of the
DFlash drafter. v1 is intentionally low-perf (τ≈1.91 vs DFlash-tree 5.74 — the current
MTP head is 1 layer vs a 5-layer draft); it's the *scaffold* a capacity-adequate
trained drafter plugs into. **Opt-in only**; production decode (flag off) is unchanged.
**Not GPU-validated on this branch** — run `./scripts/coherence-gate.sh` and
`./scripts/coherence-gate-dflash.sh` before any τ / tok-s claim.

---

## Gotchas

- **`HIPFIRE_DFLASH_MFMA=1`** ≈ 7.5× faster/step, but **corrupts the d=5120 forward**
  (eval→0, loss rises). For the target-width drafter, train **non-MFMA** (~1.8 s/step).
  `HIPFIRE_DFLASH_DX_MFMA=0` disables just the dX path.
- **Gradcheck must run FP32** — finite-diff ε is below the bf16 quantum.
- **`HIPFIRE_CKPT_STEPS`** writes milestone checkpoints mid-run.
- **`HIPFIRE_VERIFY_GRAPH=0`** required for `dflash_spec_demo` online eval (capture panics).

---

## Companion branch & further reading

- **`feat/mtp-tree-drafter-q8ef`** — the full tree-verify / DDTree kernel work
  (q8_ef tree-GDN kernel, path-c phase2, VP RoPE fix). Integrating it onto current
  master is a separate 19-file / 69-hunk re-port over the dispatch refactor, to be
  done with GPU coherence-gate validation in the loop. Tracked in the handoff issue.
- Design: `docs/plans/dflash-trainer.md` (note its "PLAN — not started" header is
  stale — the trainer is built), `docs/plans/dflash-lfm2-350m-spike.md`.
- Empirical verdicts live in the project memory (`project_gdn_drafter_treeverify_goal`,
  `project_dflash_lfm2_capacity_verdict_final`, `project_dflash_rung0_protocol_flaw`).
