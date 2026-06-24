# nemotron_h (Mamba-2 hybrid) support — starting with Nemotron-3-Nano-4B

Status: **active** (autonomous /loop, started 2026-06-24). Owner: chaingun.
Target: `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16` (smallest nemotron, `/srv/huggingface`).
Builds on the P2c seam foundation (`SequenceState` / `MixerKind::Mamba2`) and the
Mamba-2 line of `docs/plans/2026-06-23-seam-finish-and-mamba2.md` (P7).

## Why Nano-4B first

It is the cleanest nemotron: **Mamba2 + GQA-attention + dense ReLU² MLP**, with
**no MoE** (`hybrid_override_pattern` has only `M`/`*`/`-`, no `E`). So it isolates
the one genuinely-new kernel (Mamba-2 SSD) without MoE/attention confounds. The
30B/120B add MoE on top of the same blocks.

## Nano-4B config (verified)

- `model_type: nemotron_h`, `num_hidden_layers: 42`, `hidden_size: 3136`,
  `vocab_size: 131072`, `tie_word_embeddings: false`, `rms_norm_eps: 1e-5`.
- **`hybrid_override_pattern`** (42 chars, one per block):
  `M-M-M-MM-M-M*-M-M*-M-M-M*-M-M-MM*-MMM-M-M-`
  → `M`=Mamba2 mixer (21), `*`=attention mixer (4), `-`=MLP/FFN (17). **Flat block
  sequence** (each char is its own residual block, NOT a mixer+FFN pair).
- **Mamba2:** `mamba_num_heads: 96`, `mamba_head_dim: 80`
  (⇒ d_inner = 96×80 = **7680**, NOT expand×hidden), `ssm_state_size: 128`,
  `n_groups: 8`, `conv_kernel: 4`, `chunk_size: 256`, `time_step_*` bounds,
  `use_conv_bias: true`, `mamba_proj_bias: false`, `mamba_hidden_act: silu`.
  - conv operates on `xBC = [x(7680) | B(n_groups·ssm=1024) | C(1024)]` = 9728.
  - in_proj → `[z(7680) | xBC(9728) | dt(heads=96)]`. **CONFIRMED** from
    `modeling_nemotron_h.py`: `projection_size = intermediate_size + conv_dim +
    num_heads`; the HF split is `[d_mlp, d_mlp, gate(z), hidden_states_B_C, dt]`
    with **d_mlp=0** for nemotron_h (pure Mamba, no inline gated MLP), so the
    effective split is `[z | xBC | dt]`. `conv_dim = intermediate_size +
    2·n_groups·ssm_state_size = 9728`.
- **Attention (`*`):** `head_dim: 128`, `num_attention_heads: 40`,
  `num_key_value_heads: 8` (GQA), `attention_bias: false`, no sliding window,
  `max_position_embeddings: 262144`. **NoPE (to confirm at impl):** the HF
  `NemotronHAttention` class carries a `position_embeddings #TODO` and applies no
  rotary in forward — nemotron_h attention appears positional-information-free
  (the Mamba layers supply position). Verify against the checkpoint (absence of
  `rotary_emb` / `inv_freq` tensors) before wiring RoPE; default to NoPE.
- **MLP (`-`):** `mlp_hidden_act: relu2` (ReLU-squared, **not** SwiGLU),
  `intermediate_size: 12544`, `mlp_bias: false`. Up→ReLU²→down.

## What's reusable (the leverage)

- **P2c `SequenceState` / `MixerKind::Mamba2` / `RecurrentMixerState`** — the
  heterogeneous per-layer state container is exactly right: Mamba2 blocks carry
  SSM+conv state, `*` blocks carry KV, `-` blocks carry none. `MixerProfile`
  derived from the mixer blocks; `needs_kv_cache()` true (has `*` attention).
- **conv1d kernels** (`conv1d_silu_split*`) — basis for the xBC short-conv
  (conv_kernel=4, rolling K-1 decode state). Needs a Mamba2 xBC-split variant.
- ~~`gated_norm.hip`~~ — **NOT reusable** (confirmed from HF source). qwen35's
  `gated_norm.hip` is norm-then-gate per-head over head_dim; nemotron_h's
  `MambaRMSNormGated` is **gate-then-group-RMSNorm** (`norm_before_gate=False`)
  with `group_size = intermediate_size / n_groups = 960`. New kernel
  `mamba2_gated_norm.hip` written + validated gpu-vs-cpu (max|Δ|=4.8e-6).
- **GQA attention** — existing flash/batched attention for the `*` blocks.
- **The seam** (`SimpleAr`/`ServingBackend`/`decode_loop` with P3.3 sampling).

## The genuinely-new work

1. **Mamba-2 SSD / selective-scan kernel** (the hard part; no existing analog —
   `gated_delta_net` is the delta-rule outer-product recurrence, `gated_scan` is
   the GLA-lite twin; Mamba-2 is scalar-per-head decay):
   `h_t = exp(dt·A) ⊙ h_{t-1} + dt·B·x_t ; y_t = C·h_t + D·x_t`,
   A = per-head scalar (`A_log`), state `[heads × head_dim × ssm_state]`.
   - **f32 decode** (single-token recurrence) first — correctness vehicle.
   - **chunked-SSD prefill** (matmul form, chunk_size=256) second — throughput.
   - q8 state later (mirror the GDN q8 work).
2. **conv1d xBC variant** — depthwise causal conv (K=4) over `[x,B,C]` with the
   Mamba2 split + SiLU, decode rolling-state ring.
3. **ReLU² MLP** — `down(relu(up(x))²)`; add a `relu2`/`relu_sq` elementwise (or
   fuse into the up-GEMV epilogue). No relu2 kernel today.
4. **`hipfire-arch-nemotron` crate (arch_id 14)** — config (parse
   `hybrid_override_pattern` + mamba/attn/mlp dims), weights loader (per-block
   tensors), the **per-block hybrid forward** (dispatch M/*/- by the pattern),
   `SimpleAr`/`ServingBackend` impls, `MixerProfile` from the mixer blocks.
5. **Loader/registration** — `model_type: "nemotron_h" → arch_id 14` in
   `safetensors_source.rs` + `hfq.rs` + quantize `main.rs`; ingest the BF16
   checkpoint (or quantize to mq4).

## Roadmap (loop-iteration sized; commit green each)

- **N0 ✅ (this) — crate scaffold + config.** `hipfire-arch-nemotron` with the
  config struct, `BlockKind {Mamba2, Attention, Mlp}`, `hybrid_override_pattern`
  parser → `Vec<BlockKind>`, derived `MixerProfile`. Pure, no-GPU, unit-tested.
- **N1 ✅ — Mamba-2 SSD f32 decode kernel** (single-token) + CPU reference +
  gpu-vs-cpu test (max|Δy|=1.2e-7). `mamba2_ssd_decode_f32`.
- **N2 — conv1d xBC variant** + ReLU² MLP kernel + gated-norm:
  - ✅ ReLU² (`relu2_f32`, max|Δ|=0) — nemotron_h MLP act.
  - ✅ `mamba2_gated_norm_f32` (gate-then-group-RMSNorm, max|Δ|=4.8e-6) — the
    `RMSNormGated` epilogue; qwen35's `gated_norm` was confirmed not reusable.
  - ⏳ conv1d xBC depthwise causal (K=4) decode-rolling-state variant.
- **N3 — Mamba2 block forward** (in_proj GEMV → conv+silu → split x/B/C → SSD →
  `mamba2_gated_norm` → out_proj GEMV) wired on `SequenceState` recurrent slot,
  f32. (gated-norm + SSD kernels now in hand; needs conv1d-xBC + in_proj wiring.)
- **N4 — nemotron_h forward** (per-block M/*/- dispatch over the pattern;
  attention reuses GQA, MLP reuses dense GEMV + ReLU²) + loader for the BF16
  checkpoint.
- **N5 — serve + validate** on Nano-4B: `SimpleAr`/`ServingBackend`, route arch
  14, daemon load+generate, coherence eyeball + (where it exists) an HF-reference
  numeric bisect (`dump_hf_reference.py` is multi-family).
- **N6 — chunked-SSD prefill** (throughput) + q8 SSD state.
- **Then:** Nano-30B (adds MoE — reuse qwen35/lfm2moe MoE) and the broader P4–P6
  seam finish.

## Validation

- No-GPU: `./tests/no-gpu-ci.sh` for config/parser changes.
- GPU kernels: gpu-vs-cpu reference tests, gpu-tcas/`hipfire lock`-coordinated.
- End-to-end: daemon load+generate on Nano-4B; coherence (no attractors) +
  numeric bisect vs HF reference where available.
- Perf gates do NOT apply until the forward is coherent (correctness first).
