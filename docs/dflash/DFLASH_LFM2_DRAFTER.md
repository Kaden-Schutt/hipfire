# DFlash LFM2 Drafter — Native Rust Trainer (paused 2026-06-11)

A greenfield, hipfire-native (no PyTorch / SpecForge / vLLM) trainer for a
**DFlash block-diffusion speculative-decode drafter**, targeting Qwen3.6-27B on
MI300X (gfx942). Built end-to-end this session; paused at a clean **capacity
verdict** with the next step scoped.

Branch: `feat/dflash-lfm2-drafter` (mi300:/root/dflash, local — **not pushed**,
push auth revoked). All GPU work on `ssh mi300`.

---

## 1. What DFlash is

A lightweight drafter predicts a **block of ~16 tokens in one parallel forward**
(block diffusion), conditioned on the target's hidden states at a set of layers
("KV-injection" of target context at every drafter layer). The target then
verifies the block in one batched forward; accepted-prefix length τ drives the
speedup `S = τ / (1 + C_draft/C_target)`. Paper: arXiv:2602.06036.

Our target: Qwen3.6-27B (d=5120, 64 layers, vocab 248320). Context layers
(SEL) = `[1, 16, 31, 46, 61]`. block_size 16, mask_token_id 248070.

---

## 2. The trainer (built this session)

Greenfield Rust trainer in `crates/hipfire-arch-lfm2moe`:

- **`src/dflash_train.rs`** — the module. `Cfg`, `LW` (per-layer weights+grads),
  `Net`, `body_forward`/`body_backward` (hand-coded backward, gradcheck-validated
  to ≤4.7e-5 via f64 host-analytic — fp32 GPU finite-diff has a ~1e-3 noise
  floor), `StFile` (bf16/f16/f32 safetensors reader), warm-start + checkpoint
  loaders, Adam tensor iterators.
- **`examples/dflash_train_run.rs`** — streaming driver: corpus glob + filter,
  seed-anchor masking (pos0 = revealed seed weight 0, predict 1..B with
  `w_k = exp(-(k-1)/γ)`), pool-scoped per-step alloc/free, warmup + cosine lr,
  periodic checkpoint, per_pos / proxy_tau eval.
- **`kernels/src/dflash_train.hip`** — 14 training kernels (linear fwd/dx/dw,
  rmsnorm bwd, silu_mul bwd, rope bwd, online-softmax block cross-attn fwd/bwd,
  ce_loss bwd, adam, cgate_add, colsum) + the MFMA kernels below.

Data generation: `crates/hipfire-runtime/examples/gen_qwen35_bulk.rs` —
target-regenerates sequences (one model load, many seqs), dumps the 5 SEL-layer
hidden states (HFHS, 52 MB/seq) + tokens. Flags: `--sel`, `--rep-penalty`,
`--n-seqs`, `--start`, `--n`.

---

## 3. MFMA kernel optimization — 7.5× GPU / ~10× wall

`rocprofv3 --kernel-trace` on the fp32 step found the **dW backward was 56%**:
`linear_bwd_dw_f32` launched N×K warps each doing only M=batch=16 MACs
(4.7M warps for one layer) — pathologically occupancy-starved. Every body GEMM
has M=16, which maps exactly onto the 16-wide MFMA k-tile. Gated by
`HIPFIRE_DFLASH_MFMA=1`. Inputs f32, cast to bf16 inline, **fp32 accumulate**
(F16 tensor reused as a 2-byte bf16 container — no `DType::BF16` needed).

| kernel | before | after | how |
|---|---|---|---|
| dW backward | 4011 ms | **27 ms** (150×) | one `v_mfma_f32_16x16x16bf16` per 16×16 tile, chunked over M |
| lm_head fwd/bwd | 1855 ms | **316 ms** | frozen GEMM ran at ~6% HBM BW (lane-strided A); LDS-coalesced A+B stage + split-K |
| dX backward | 660 ms | **97 ms** | split-K over the N-contraction (256-slabs over grid.y) + fp32 atomicAdd partials |
| body forwards | 277 ms | ~75 ms | bf16-MFMA inline (norms/attn/conv stay f32) |
| **step (GPU)** | **7115 ms** | **951 ms** | **7.5×** |

Wall: **287 → 27.8 ms/step (~10×)**. A 30k-step run dropped from ~2.4 hr to
~14 min. 25-step loss matches fp32 exactly (12.35 vs 12.44 at 50 steps).
Validation: `test_dw_mfma` / `test_dx_mfma` / `test_lmhead_bf16` /
`test_lin_parity` (rel-L2 layout checks, ~6e-5 to ~2e-3). Gradcheck is
**structurally invalid under bf16 forward** (fd ε=1e-3 < bf16 quantum) so it
skips under the flag; parity tests are the layout oracle.

Key negative result: the **first** dX-MFMA (grid=K/64, ~16 WGs) *lost* to the
naive kernel (839 > 660 ms) — small-M, reduction-dominated GEMMs need split-K,
not just MFMA tiling.

Commits: `af54b4a4` (round 1, 3.86×), `8b5e612b` (round 2, 7.5×).

---

## 4. Rung-0 — harness validated against published weights (`4f5c3c4c`)

To prove the trainer eval/data/injection conventions are correct, we loaded the
**published z-lab Qwen3.6-27B-DFlash drafter** (3.46 GB bf16 safetensors,
`/workspace/zlab-dflash`) into the trainer `Net` and ran our eval:

- **proxy_tau 5.45 / per_pos 0.463 / pos1 0.78** (ctx384, greedy seqs) — the
  τ≈6 ballpark. → trainer conventions (KV-injection, RoPE, positions, GQA, fc,
  **hidden_norm after fc**, mask 248070) are all correct.
- Cross-checked against z-lab's official `dflash.py` (trust_remote_code; absent
  in the 27B repo — fetched from the 9B repo) on identical inputs: it collapsed
  *identically* to our harness on bad data → harness vindicated, protocol
  indicted.

Code: `examples/dflash_eval_zlab.rs`, `dt::load_zlab`, `dt::Cfg::zlab_27b`,
`Net.hidden_norm`. The published drafter config: d=5120, 5 GQA layers
(32h/8kv/hd128), inter 17408, ~1.6B params.

---

## 5. Capacity verdict — rigorous (final A/B)

The earlier "30k capacity verdict" was **confounded** by two data-protocol flaws
that crippled both training and eval:
1. **rep-penalty 1.3** in corpus regen distorted target tokens off raw argmax
   (the drafter mimics the raw target distribution) — roughly halves pos1.
2. **n_ctx 32** far too short — the drafter needs hundreds of committed rows
   (pos6 accuracy went 0 → 0.31 just by ctx 32 → 128).

After fixing both (greedy targets, ctx 256–384, mask 248070), the clean A/B on
**12 fresh unseen sequences** (identical blocks for both drafters):

| drafter | width | per_pos | proxy_tau | pos1 |
|---|---|---|---|---|
| z-lab (published) | d=5120, 1.6B | **0.398** | **3.80** | 0.68 |
| LFM2-350M (ours) | d=1024, 350M | **0.021** | ~0 | 0.13 |

train ≈ heldout ≈ low → **UNDERFIT**. The 350M's 25× context compression at
d=1024 (fc squeezes 5×5120 → 1024) cannot model coherent target continuations.
Capacity, not data or steps.

### Loop economy (the subtle part)
~92% of pure-greedy sequences degenerate into loops (uniq < 0.35) — this **is**
the production greedy-serving distribution. Loops are trivially predictable, so
a drafter trained+evaled on loop-heavy data hits τ≈7.5 in 3000 steps. The
7.5→0→7.6 eval bounce across milestones was **2-seq heldout variance, not
divergence** (loss was steady throughout). On coherent-only data (116 surviving
seqs, 30k steps) per_pos plateaued at 0.03–0.06. Even the published drafter
measures τ **3.80**, not 6.3, on fresh coherent-mixed greedy data — production
realized τ rides loops.

---

## 6. Data-quality finding (why the corpus is the next lever)

Our corpus is 512-token **greedy rollouts from short seeds** → most sequences
drift into loops within tens of tokens, so the drafter spends its gradient
learning the target's failure mode. The deployment-relevant distribution is
different: in production every block starts from a **coherent accepted prefix**
(chat sampled at temp>0), and the drafter predicts the target's short greedy
continuation from there.

**Data-gen v2 (designed, not built):** take real coherent text as prefixes
(agentic/code/chat corpora we already have), teacher-force through the target
(batched prefill), dump the 5 SEL layers, and generate only **16 greedy tokens**
as block labels per sampled position (not a 512-token rollout). That is the
z-lab recipe shape; loops then appear only at their natural rate.

---

## 7. Next steps (the ladder)

- **Rung 1 — data-gen v2**: teacher-forced coherent-prefix corpus (above).
  Strictly better data, cheaper per example. Feeds everything below.
- **Rung 2 — target-init d=5120 drafter**: 5 attention layers initialized from
  Qwen3.6-27B's own full-attn layers (`target_layer_ids` ≈ SEL), frozen
  embed/lm_head, fresh fc. This is the z-lab construction re-derived natively;
  the trainer + eval pipeline already works at d=5120 (rung-0 ran it). Weight
  extraction needs **no new kernels** (fused MFMA batched-identity GEMM). Gate:
  approach z-lab per_pos → trainer can build SOTA drafters natively → unlock the
  goal (retrain on agentic mix to beat published τ).
- **Rung 3 — LFM2 conv hybrid**: from rung-2's checkpoint, swap 2–3 attention
  layers for conv layers at d=5120 (the LFM thesis: cheaper drafts). Same eval
  decides it.

Cost: a full 30k retrain at d=5120 ≈ 1–1.7 hr post-MFMA (the d=1024 350M was
~14 min; d=5120 GEMMs are ~25× the FLOPs).

---

## 8. Env knobs & artifacts

**Trainer flags:** `HIPFIRE_DFLASH_MFMA=1` (MFMA kernels),
`HIPFIRE_DFLASH_DX_MFMA=0` (opt out of split-K dX),
`HIPFIRE_TRAIN_NCTX` (context length, default 256),
`HIPFIRE_UNIQ_MIN` (degenerate-seq filter, default 0.05; 0.35 = coherent-only),
mask_token_id 248070.

**On-disk (mi300):**
- `/workspace/zlab-dflash/` — published z-lab drafter (safetensors + dflash.py)
- `/workspace/dflash-greedy/` — 1514 greedy seqs, SEL=[1,16,31,46,61], rep-penalty 1.0
- `/workspace/ab-fresh/` — 12 fresh A/B seqs (seed start 1600)
- `/workspace/lfm2-coherent-30k.dfnet`, `/workspace/lfm2-greedy-30k.dfnet` — checkpoints
- `/workspace/qwen3.6-27b.head.f32`, `.lmhead.f32` — extracted target embed/lm_head
- `/workspace/qwen3.6-27b.mq4-awq` — target model (data-gen)

**Tests:** `cargo build --release --example {test_dw_mfma,test_dx_mfma,test_lmhead_bf16,test_lin_parity} -p hipfire-arch-lfm2moe`

---

## 9. License note

LFM2.5 is a custom Liquid license — fine for research/spike; gates *shipping* a
derived drafter. Rung-2 (target-init from Qwen) inherits Qwen's license instead.
