# hipfire-train — Phase 0 plan: backprop-correctness proof (LoRA SFT)

Status: READY — design finalized, all open questions resolved. First primitive
(`gemm_f32_train`) landed; `hipfire-train` crate + ops are the next build step.
Date: 2026-06-17
Target model: `SupraLabs/Supra-50M-Instruct` (local: `/srv/huggingface/models--SupraLabs--Supra-50M-Instruct`)

## Goal

Stand up the **first backward pass + optimizer** in hipfire and prove it is
numerically correct, using the smallest real LLaMA-family model available. This
is the prerequisite for everything else (LoRA-over-quantized fine-tuning, full
QAT, etc.). Phase 0 deliberately ignores the quantized formats — base weights
run in **fp32** — so a wrong loss curve points at our gradient math, not at
quant error.

### Success criterion (the one number that matters)

A LoRA `A·B` adapter on `q_proj`/`v_proj` of Supra-50M trains via a hand-written
backward + AdamW loop and **overfits ~50 alpaca examples to near-zero loss on
gfx1100**, with:

1. **Finite-difference gradient check passes** (primary gate): for a tiny
   input, every trainable-param analytic gradient matches a central-difference
   numeric gradient to rel-err < 1e-2 in fp32. This is independent of PyTorch
   and is the gold-standard correctness test.
2. **Loss-curve sanity vs `sft.py`** (secondary): same data/loss/hparams on a
   small overfit subset trends the same way. (Not byte-exact — see Tokenizer
   caveat below.)

## Non-goals (Phase 0)

- No quantized base (that's Phase 1).
- No general autograd tape — hand-written backward for exactly the llama-dense
  op set. A tape can come later.
- No multi-GPU, no checkpointing-to-resume, no fused training kernels.
- Not differentiating the inference hot path (see Architecture decision).

## Why Supra-50M

- `model_type: llama`, `LlamaForCausalLM`, `attention_bias: false`,
  `mlp_bias: false` → **arch_id=0**, the clean dense pre-norm transformer
  (RMSNorm → GQA+RoPE → SwiGLU). Minimal backward surface.
- Tiny: 12 layers, hidden 512, 8 heads (head_dim 64) / 4 KV heads, intermediate
  1408, vocab 32000, ctx 1024. ~50M params → fp32 master copy ~200MB, overfits
  in seconds, hundreds of debug iterations/min.
- Ships `sft.py` (alpaca-cleaned, prompt-masked CE) and `training_args.bin` —
  a reference recipe + hparams to mirror.

**Tokenizer caveat:** `sft.py` builds its tokenizer from
`custom_llama_tokenizer-vocab.json` / `-merges.txt`, which are **not** in the
snapshot. The snapshot's `tokenizer.json` is present and is what hipfire will
load. Token IDs may not match the original training tokenizer exactly, so do
NOT expect byte-identical loss vs a PyTorch rerun — use finite-difference
gradient checking as the hard correctness gate, and the PyTorch loss curve only
as a directional sanity check.

## Architecture decision: separate fp32 training forward

The inference forward in `crates/hipfire-runtime/src/llama.rs` is heavily fused
and quant-specialized (`fused_rmsnorm_rotate_mq`, `fused_silu_mul_rotate_mq`,
prerotated GEMV, …). Differentiating those is intractable and pointless.

Instead the training path gets its **own un-fused fp32 forward**, one clean op
per node, each with a matching backward. Matmuls go through the dedicated
**`gemm_f32_train`** primitive (general transpose flags, shipped 2026-06-17 —
see Resolved decisions §1); backward of a matmul is just two more calls into the
same kernel with different transpose/ld args, so that single primitive covers
most of the gradient work. The frozen base's forward needs no gradient; only the
LoRA branch + activations carry grad.

## New crate: `hipfire-train`

```
crates/hipfire-train/
├── Cargo.toml            # deps: rdna-compute, hipfire-runtime (loaders/types),
│                         #       hipfire-model, hipfire-dispatch
├── src/
│   ├── lib.rs
│   ├── tensor.rs         # TrainTensor: fp32 GpuTensor + optional .grad buffer
│   ├── ops/              # forward+backward pairs (each op: fn fwd, fn bwd)
│   │   ├── linear.rs     # y = x·Wᵀ ; dX = dY·W ; dW = dYᵀ·x   (gemm_f32)
│   │   ├── lora.rs       # y += (x·Aᵀ)·Bᵀ·(α/r) ; grads to A,B only
│   │   ├── rmsnorm.rs
│   │   ├── rope.rs       # RoPE is parameter-free; bwd = inverse rotation
│   │   ├── attention.rs  # GQA: scores=QKᵀ, softmax, ·V, o_proj
│   │   ├── softmax.rs
│   │   ├── swiglu.rs     # silu(gate)*up  (silu' = σ(x)(1+x(1-σ(x))))
│   │   └── cross_entropy.rs  # fused logsoftmax+NLL, ignore_index=-100
│   ├── model_llama.rs    # un-fused fp32 forward + backward for the dense stack
│   ├── optim.rs          # AdamW (fp32 m,v state), grad-clip by global norm
│   ├── lora.rs           # adapter config: which linears, rank r, alpha, init
│   ├── data/
│   │   ├── alpaca.rs     # alpaca-cleaned loader, build_prompt, prompt mask
│   │   └── batch.rs      # right-pad to longest, label pad = -100
│   ├── loader.rs         # Supra-50M safetensors → fp32 GPU weights
│   │                     #   (reuse hipfire-runtime::safetensors_source)
│   └── gradcheck.rs      # central finite-difference gradient checker
└── examples/
    └── sft_supra50m.rs   # the Phase 0 driver (overfit + full run)
```

## Forward graph (un-fused, per layer)

Pre-norm LLaMA block, batch B, seq T, hidden H=512:

```
h_in
 ├─ rmsnorm(h_in, w_attn_norm)              → xn
 ├─ q = xn·Wqᵀ  (+ LoRA_q)                  → reshape [B,T,nH,d]
 ├─ k = xn·Wkᵀ                              → [B,T,nKV,d]
 ├─ v = xn·Wvᵀ  (+ LoRA_v)                  → [B,T,nKV,d]
 ├─ rope(q), rope(k)
 ├─ scores = q·kᵀ / √d   (GQA head broadcast: 8 q-heads share 4 kv-heads)
 ├─ causal mask + softmax                   → p
 ├─ ctx = p·v
 ├─ attn_out = ctx_merged·Woᵀ
 ├─ h_mid = h_in + attn_out                 (residual)
 ├─ rmsnorm(h_mid, w_mlp_norm)              → yn
 ├─ gate = yn·Wgateᵀ ; up = yn·Wupᵀ
 ├─ act  = silu(gate) * up
 ├─ mlp_out = act·Wdownᵀ
 └─ h_out = h_mid + mlp_out                 (residual)
```

Head: `final_rmsnorm`, then `logits = h·E_outᵀ` (E_out tied to embedding,
`tie_word_embeddings: true` — frozen in Phase 0, so no grad to embedding).

## Backward kernel list (forward op → required grad)

| Forward op | Backward needs | Reuses |
|---|---|---|
| `linear` y=x·Wᵀ | dX=dY·W ; dW=dYᵀ·x | `gemm_f32` (×2, transposes) |
| `lora` | dA, dB only (base W frozen) | `gemm_f32` |
| `rmsnorm` | dX given dY, norm, eps (per-row reduction) | new small kernel |
| `rope` | inverse rotation of dQ,dK (param-free) | new (or reuse rope fwd w/ −θ) |
| `matmul QKᵀ` / `p·V` | two matmuls each | `gemm_f32` (batched) |
| `softmax` (causal) | dS = p∘(dP − Σ(dP∘p)) | new kernel |
| `silu*up` (swiglu) | d_gate, d_up via silu′ | new elementwise kernel |
| `cross_entropy` (logsoftmax+NLL, ignore -100) | dLogits = softmax − onehot, masked | new kernel (fused fwd+bwd) |
| residual add | gradient splits/sums (free) | — |

New kernels required: rmsnorm-bwd, softmax-bwd, swiglu-bwd, cross-entropy
(fwd+bwd), and a transpose helper if `gemm_f32` can't take a transposed
operand. Everything else is `gemm_f32` + elementwise.

## Optimizer

AdamW in fp32, state `m,v` per trainable param (LoRA only in Phase 0, so state
is tiny). Match `sft.py`: β1=0.9, β2=0.999, eps=1e-8, weight_decay=0.0, LR=3e-4,
cosine schedule + 10% warmup, **global-norm grad clip at 1.0**.

## Data pipeline (mirror sft.py)

- Dataset: `yahma/alpaca-cleaned`. For the overfit gate, take the first ~50
  samples; for the directional run, a few thousand.
- Prompt templates: the two Alpaca templates from `sft.py` verbatim.
- Labels: `[-100]*prompt_len + response_ids`, `eos` appended to response.
- Collate: right-pad to longest in batch, label pad = -100, build attention
  mask (Phase 0 can also just process one sequence at a time to start).

## Milestones

- **M0 — crate + loader.** `hipfire-train` builds; load Supra-50M safetensors to
  fp32 GPU tensors. ✅ DONE (commit b9c3f69): config parse + bf16→f32 + all
  weights verified on gfx1103.
  - *Approach change (2026-06-17):* the original M0 also asked for an fp32
    forward matching the inference logits as a sanity check. But the inference
    fp32 ops carry conventions (RoPE interleave, batched-row layout) we'd have to
    match exactly and then still write backward for — throwaway work. Since the
    crate owns its own un-fused ops anyway (plan architecture), we instead
    validate each op directly by **finite-difference gradcheck** (M1) and the
    eventual full forward against a **PyTorch logit dump on fixed input_ids**
    (tokenizer-independent). The inference-logit match is dropped.
- **M1 — single-op gradient checks.** linear, rmsnorm, softmax, swiglu, rope,
  cross_entropy each pass finite-difference check in isolation.
  - **linear** ✅ DONE: `ops/linear.rs` (forward + dX + dW via `gemm_f32_train`),
    `examples/gradcheck_linear.rs` passes on gfx1103 (max|analytic−numeric|
    ≈1.5e-5, tol 1e-2). No new kernel needed.
  - **rmsnorm** ✅ DONE: `kernels/src/rmsnorm_train.hip` (fwd saves 1/r per row;
    bwd computes dx + atomic-accumulates dw), gradcheck dX 3.2e-5, dW 2.0e-5.
  - **softmax** ✅ DONE: `softmax_train.hip` (fwd writes p; bwd Jacobian),
    gradcheck dS 3.7e-5.
  - **swiglu** ✅ DONE: `swiglu_train.hip` (elementwise; silu'), gradcheck
    d_gate 8.0e-5, d_up 2.3e-5.
  - **cross_entropy** ✅ DONE: `cross_entropy_train.hip` (fused fwd+bwd,
    ignore_index masking, sum-reduction grad), gradcheck 4.3e-4 + masking
    verified.
  - **rope** ✅ DONE: `rope_train.hip` (HF half-split; bwd = rotation by −angle),
    gradcheck dX 1.4e-4, norm-preservation 2.4e-7.

  **M1 COMPLETE** — all six ops pass finite-difference gradchecks on gfx1103.
- **M2 — full-graph gradient check.** End-to-end fp32 model, finite-difference
  check on LoRA params (and one base param via a temporarily-unfrozen linear)
  for a 1–2 token toy input.
  - **LoRA op** ✅ DONE: `ops/lora.rs`, gradcheck dA/dB/dX ≈1e-5.
  - **single-head causal SDPA** ✅ DONE: `ops/attention.rs` sdpa_*,
    `causal_mask_train.hip`, gradcheck dQ/dK/dV ≈6–8e-5.
  - **GQA multi-head** ✅ DONE: gqa_forward/gqa_backward via gather/scatter
    (`strided_copy_2d.hip`, scatter-add for shared kv heads),
    gradcheck dQ 6.2e-5 / dK 9.4e-5 / dV 7.8e-5 with n_heads=4,n_kv=2.
  - TODO: full transformer block (norm→attn proj+LoRA→residual→norm→swiglu
    MLP→residual) → full model fwd+bwd → end-to-end LoRA gradcheck.
- **M3 — AdamW overfit.** LoRA on q_proj/v_proj, ~50 examples → loss → ~0.
  THE success criterion.
- **M4 — directional vs PyTorch.** Few-thousand-example run trends like
  `sft.py`. Bank the loss curve.

## Phase 1 preview (the actual goal: fine-tune quantized)

Once M3 holds: keep the LoRA branch + optimizer + backward identical; swap the
frozen base linear's forward from fp32 `gemm_f32` to the **dequant/MQ GEMV**
(forward-only, no grad on the frozen quantized weights). Validate the same
overfit converges with base at MQ4. Optional later: offline merge `W+BA` and
re-run the QTIP/MQ encoder to fold the adapter into the base (the only place
quanta are rewritten — batch, not per-step). Full QAT-on-quanta (STE + periodic
Viterbi re-projection) is a separate, later increment built on this same loop.

## Resolved decisions (M0 open questions — settled 2026-06-17)

### 1. GEMM / transpose — RESOLVED: dedicated training GEMM shipped

The inference `gemm_f32_batched` is **NT-only**: `Y[n,m] = Σ_k A[m,k]·B[n,k]`
(both operands row-major, contract inner K, output stored `[N,M]`). It matches
the forward linear but neither backward matmul. Per user go-ahead, the training
crate uses **one general fp32 GEMM with transpose flags**, now landed:

- `kernels/src/gemm_f32_train.hip` — `gemm_f32_train` and `gemm_f32_train_accum`
  (the `_accum` form does `C = beta*C + op(A)·op(B)` for gradients that land on
  a buffer holding a partial — e.g. residual-merged or LoRA-added `dX`).
- `Gpu::gemm_f32_train(a,b,c, m,n,k, lda,ldb, trans_a,trans_b)` in
  `rdna-compute/src/dispatch.rs` (and `…_accum`). Builds clean.

Computes `C[M,N] = op(A)·op(B)`, C row-major, `op(A)`=`[M,K]`, `op(B)`=`[K,N]`:
```
op(A)[m,k] = trans_a ? A[k*lda + m] : A[m*lda + k]
op(B)[k,n] = trans_b ? B[n*ldb + k] : B[k*ldb + n]
```

**The three linear-layer matmuls** (X = `[M,K]` tokens×in, W = `[Nout,K]`):

| product | call |
|---|---|
| forward `Y[M,Nout] = X·Wᵀ` | `gemm(X, W, Y,  M, Nout, K,  lda=K, ldb=K, trans_a=0, trans_b=1)` |
| `dX[M,K] = dY·W` | `gemm(dY, W, dX, M, K, Nout, lda=Nout, ldb=K, trans_a=0, trans_b=0)` |
| `dW[Nout,K] = dYᵀ·X` | `gemm(dY, X, dW, Nout, K, M, lda=Nout, ldb=K, trans_a=1, trans_b=0)` |

This is naive (one wave32 block per output element) — correctness-first;
tiled/WMMA is a later optimization.

**Correctness VERIFIED 2026-06-17** on gfx1103 via
`crates/rdna-compute/examples/test_gemm_f32_train_gpu_vs_cpu.rs`: forward, both
backward matmuls, and the `_accum` variant all match a CPU reference to fp32
epsilon (max_abs_err ≤ 6e-8). Run it with `/opt/rocm/llvm/bin` on PATH (see
env note below).

> **JIT env note:** kernel JIT needs `clang-offload-bundler`. Running an example
> binary directly (not via the `~/.hipfire/bin` wrapper) won't find it and dies
> with `failed to run clang-offload-bundler … (os error 2)`. Canonical fix
> (matches the daemon wrapper, see memory `project-rocm-env`): set
> `ROCM_PATH=/opt/rocm`. Equivalently prepend the llvm bin to PATH —
> `export PATH="/opt/rocm/llvm/bin:$PATH"`.

### 2. fp32 load path — RESOLVED: `SafetensorsSource::tensor_data`

`SafetensorsSource` (`hipfire-runtime/src/safetensors_source.rs`) serves raw
tensor bytes by name via `tensor_data(name) -> (&TensorInfo, &[u8])`, with
`TensorInfo.dtype` ("BF16"/"F16"/"F32") and `.shape`, mmap-backed, **no
quantizer involvement**. The training loader opens the Supra-50M dir, reads each
tensor's bytes + dtype, and converts to fp32 on upload. Supra-50M is bf16 →
bf16→f32 is a 16-bit left shift into the mantissa. No new loader infra needed.

### 3. Batching — RESOLVED: single-sequence for M0–M3, batch at M4

Process one sequence at a time through M3 to cut the padding/attention-mask
surface while proving the backward. Add right-padded batching + masked loss at
M4 (matching `sft.py`'s collator) for the directional run.
