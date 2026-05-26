#!/usr/bin/env python3
"""
ParoQuant DeltaNet Layer 0 debug script.

Loads Qwen3.5-0.8B-PARO from safetensors, runs the full DeltaNet forward
pass for layer 0 on a single token (248045 = <|im_start|>) at pos=0,
and dumps every intermediate value for comparison with hipfire.

Usage:
  nix-shell -p python3Packages.torch python3Packages.safetensors \
    --run "python3 scripts/paro_layer0_debug.py"
"""

import math
import struct
import sys
import os
import json
import numpy as np

# Try torch first, fall back to numpy-only
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("WARNING: torch not available, using numpy only (slower)")

from safetensors import safe_open

MODEL_DIR = os.path.expanduser("~/.hipfire/models/Qwen3.5-0.8B-PARO")
TOKEN_ID = 248045  # <|im_start|>

# Config from config.json
DIM = 1024
NUM_LAYERS = 24
NUM_ATTN_HEADS = 8
HEAD_DIM = 256
N_KV_HEADS = 2
LINEAR_NUM_KEY_HEADS = 16
LINEAR_NUM_VALUE_HEADS = 16
LINEAR_KEY_HEAD_DIM = 128
LINEAR_VALUE_HEAD_DIM = 128
INTERMEDIATE_SIZE = 3584
NORM_EPS = 1e-6
KROT = 8
GROUP_SIZE = 128
CONV_KERNEL_DIM = 4
PARTIAL_ROTARY_FACTOR = 0.25

K_DIM = LINEAR_NUM_KEY_HEADS * LINEAR_KEY_HEAD_DIM   # 2048
V_DIM = LINEAR_NUM_VALUE_HEADS * LINEAR_VALUE_HEAD_DIM  # 2048
QKV_DIM = K_DIM * 2 + V_DIM  # 6144


def fmt_arr(arr, n=8):
    """Format first n elements of an array."""
    if HAS_TORCH and isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().float().numpy()
    if isinstance(arr, np.ndarray):
        return "[" + ", ".join(f"{x:.6f}" for x in arr.flat[:n]) + "]"
    return str(arr[:n])


def load_tensor(model, name):
    """Load a tensor from the safetensors file."""
    return model.get_tensor(name)


def f16_to_f32_tensor(t):
    """Convert an F16 tensor to F32."""
    if HAS_TORCH:
        if t.dtype == torch.float16:
            return t.float()
        return t.float()
    return t.astype(np.float32)


def rmsnorm(x, weight, eps=1e-6):
    """RMSNorm: out = x * weight / rms(x)"""
    if HAS_TORCH:
        x = x.float()
        weight = weight.float()
        rms = torch.rsqrt(torch.mean(x * x) + eps)
        return x * weight * rms
    else:
        x = x.astype(np.float64)
        weight = weight.astype(np.float64)
        rms = 1.0 / np.sqrt(np.mean(x * x) + eps)
        return (x * weight * rms).astype(np.float32)


def givens_rotate(x_in, pairs, theta, channel_scales, hidden_dim, krot, group_size=128):
    """
    Apply ParoQuant Givens rotation to activation vector.

    Matches the HIP kernel: givens_rotate_f32.
    1. Apply channel_scales (multiply)
    2. For each rotation round: apply 2x2 Givens rotation per pair
    """
    if HAS_TORCH:
        x = x_in.float().clone()
        channel_scales = channel_scales.float()
    else:
        x = x_in.astype(np.float32).copy()
        channel_scales = channel_scales.astype(np.float32)

    # Step 1: Apply channel scales
    cs = channel_scales.flatten()[:hidden_dim]
    x[:hidden_dim] *= cs

    # Step 2: Apply krot rounds of Givens rotations
    n_groups = hidden_dim // group_size
    half_group = group_size // 2

    for rot in range(krot):
        if HAS_TORCH:
            xn = x.clone()
        else:
            xn = x.copy()

        for g in range(n_groups):
            ch_base = g * group_size
            for tid in range(half_group):
                # pairs layout: [krot, hidden_dim]
                # Thread tid reads pairs[rot, ch_base + 2*tid] and pairs[rot, ch_base + 2*tid+1]
                pair_offset = rot * hidden_dim + ch_base + 2 * tid
                i_local = int(pairs.flatten()[pair_offset].item())
                j_local = int(pairs.flatten()[pair_offset + 1].item())

                # theta layout: [krot, hidden_dim/2]
                theta_offset = rot * (hidden_dim // 2) + g * half_group + tid
                if HAS_TORCH:
                    angle = theta.flatten()[theta_offset].float().item()
                else:
                    angle = float(theta.flatten()[theta_offset])

                c = math.cos(angle)
                s = math.sin(angle)

                # Indices are within the group (0..127)
                gi = ch_base + i_local
                gj = ch_base + j_local

                if HAS_TORCH:
                    xi = x[gi].item()
                    xj = x[gj].item()
                else:
                    xi = float(x[gi])
                    xj = float(x[gj])

                # Forward Givens: x[i] = xi*c + xj*s, x[j] = xj*c - xi*s
                xn[gi] = xi * c + xj * s
                xn[gj] = xj * c - xi * s

        x = xn

    return x


def awq_dequant_gemv(qweight, qzeros, scales, x, in_dim, out_dim, group_size=128):
    """
    Dequantize AWQ INT4 weights and compute GEMV using torch matrix ops.

    AWQ layout:
      qweight: I32 [in_dim, out_dim/8] - 8 nibbles per I32
      qzeros:  I32 [in_dim/group_size, out_dim/8] - 8 zero-point nibbles per I32
      scales:  F16 [in_dim/group_size, out_dim] - per-group scales

    Dequant: w[k, m] = scales[k//gs, m] * (qw_nibble[k, m] - qz_nibble[k//gs, m])
    GEMV:    y[m] = sum_k w[k, m] * x[k]
    """
    groups_per_row = in_dim // group_size

    if HAS_TORCH:
        # Parse as int32
        qw = qweight.view(torch.int32).reshape(in_dim, out_dim // 8)
        qz = qzeros.view(torch.int32).reshape(groups_per_row, out_dim // 8)
        sc = scales.float().reshape(groups_per_row, out_dim)

        # Build the full dequantized weight matrix [in_dim, out_dim]
        W = torch.zeros(in_dim, out_dim, dtype=torch.float32)

        for m in range(out_dim):
            col_word = m // 8
            nibble_shift = (m % 8) * 4

            for g in range(groups_per_row):
                scale_val = sc[g, m].item()
                zero_nibble = ((qz[g, col_word].item() >> nibble_shift) & 0xF)

                for i in range(group_size):
                    k = g * group_size + i
                    q_nibble = ((qw[k, col_word].item() >> nibble_shift) & 0xF)
                    W[k, m] = scale_val * (q_nibble - zero_nibble)

        y = x.float() @ W
        return y
    else:
        raise NotImplementedError("numpy AWQ dequant not implemented")


def awq_dequant_gemv_fast(qweight_bytes, qzeros_bytes, scales_bytes, x_np, in_dim, out_dim, group_size=128):
    """
    Fast AWQ dequant + GEMV using vectorized numpy/torch ops.
    Builds the full weight matrix first, then does a single matmul.
    """
    groups_per_row = in_dim // group_size

    # Parse raw bytes
    qw = np.frombuffer(qweight_bytes, dtype=np.int32).reshape(in_dim, out_dim // 8)
    qz = np.frombuffer(qzeros_bytes, dtype=np.int32).reshape(groups_per_row, out_dim // 8)

    # F16 scales -> F32
    sc_u16 = np.frombuffer(scales_bytes, dtype=np.float16).reshape(groups_per_row, out_dim)
    sc = sc_u16.astype(np.float32)

    # Build weight matrix [in_dim, out_dim]
    W = np.zeros((in_dim, out_dim), dtype=np.float32)

    for m in range(out_dim):
        col_word = m // 8
        nibble_shift = (m % 8) * 4

        for g in range(groups_per_row):
            scale_val = sc[g, m]
            zero_nibble = (qz[g, col_word] >> nibble_shift) & 0xF

            for i in range(group_size):
                k = g * group_size + i
                q_nibble = (qw[k, col_word] >> nibble_shift) & 0xF
                W[k, m] = scale_val * (q_nibble - zero_nibble)

    # GEMV
    if HAS_TORCH:
        W_t = torch.from_numpy(W)
        x_t = torch.from_numpy(x_np).float() if not isinstance(x_np, torch.Tensor) else x_np.float()
        return (x_t @ W_t).numpy()
    else:
        return x_np.astype(np.float32) @ W


def fp16_gemv(weight_bytes, x, out_dim, in_dim):
    """
    FP16 weight GEMV.
    weight: [out_dim, in_dim] stored as F16 row-major.
    y[m] = sum_k W[m, k] * x[k]
    """
    W_f16 = np.frombuffer(weight_bytes, dtype=np.float16).reshape(out_dim, in_dim)
    W_f32 = W_f16.astype(np.float32)

    if HAS_TORCH:
        W_t = torch.from_numpy(W_f32)
        if isinstance(x, torch.Tensor):
            x_t = x.float()
        else:
            x_t = torch.from_numpy(x).float()
        return W_t @ x_t
    else:
        if isinstance(x, np.ndarray):
            return W_f32 @ x.astype(np.float32)
        return W_f32 @ np.array(x, dtype=np.float32)


def paro_rotate_gemv(qw_bytes, qz_bytes, sc_bytes, pairs_bytes, theta_bytes, cs_bytes,
                     x, in_dim, out_dim, group_size=128, krot=8):
    """
    ParoQuant: rotate activations + AWQ dequant GEMV.
    1. Apply Givens rotation to x
    2. AWQ dequant GEMV on rotated x
    """
    # Parse rotation metadata
    pairs = np.frombuffer(pairs_bytes, dtype=np.int16).reshape(krot, in_dim)
    theta = np.frombuffer(theta_bytes, dtype=np.float16).reshape(krot, in_dim // 2)
    cs = np.frombuffer(cs_bytes, dtype=np.float16).reshape(-1)

    if HAS_TORCH:
        pairs_t = torch.from_numpy(pairs.copy())
        theta_t = torch.from_numpy(theta.copy()).float()
        cs_t = torch.from_numpy(cs.copy()).float()
        x_t = x.float() if isinstance(x, torch.Tensor) else torch.from_numpy(x).float()

        # Apply rotation
        x_rot = givens_rotate(x_t, pairs_t, theta_t, cs_t, in_dim, krot, group_size)

        # AWQ dequant GEMV on rotated x
        return awq_dequant_gemv_fast(qw_bytes, qz_bytes, sc_bytes, x_rot.numpy(), in_dim, out_dim, group_size)
    else:
        x_np = x if isinstance(x, np.ndarray) else np.array(x, dtype=np.float32)
        pairs_f = pairs
        theta_f = theta.astype(np.float32)
        cs_f = cs.astype(np.float32)

        x_rot = givens_rotate(x_np, pairs_f, theta_f, cs_f, in_dim, krot, group_size)
        return awq_dequant_gemv_fast(qw_bytes, qz_bytes, sc_bytes, x_rot, in_dim, out_dim, group_size)


def conv1d_single_token(qkv, conv_weight_bytes, qkv_dim, conv_kernel_dim=4):
    """
    Conv1d on a single token at pos=0 with empty conv state.

    At pos=0, conv state is all zeros, so it's just:
      out[ch] = sum_{k=0}^{kernel_dim-1} conv_weight[ch, 0, k] * state[k]
    But state is [0, 0, 0, qkv[ch]] (the current token is at the end).

    conv_weight shape: [qkv_dim, 1, kernel_dim]
    At pos=0: only the last kernel tap (k=kernel_dim-1) is multiplied with qkv[ch].
    """
    # Parse conv_weight: stored as F16 [qkv_dim, 1, conv_kernel_dim]
    cw = np.frombuffer(conv_weight_bytes, dtype=np.float16).reshape(qkv_dim, 1, conv_kernel_dim)
    cw_f32 = cw.astype(np.float32)

    if HAS_TORCH:
        qkv_t = qkv if isinstance(qkv, torch.Tensor) else torch.from_numpy(qkv).float()
        # At pos=0: only the last tap matters (state = [0, 0, 0, current])
        out = qkv_t.float() * torch.from_numpy(cw_f32[:, 0, conv_kernel_dim - 1])
        return out
    else:
        qkv_np = qkv if isinstance(qkv, np.ndarray) else np.array(qkv, dtype=np.float32)
        return qkv_np * cw_f32[:, 0, conv_kernel_dim - 1]


def silu(x):
    """SiLU activation: x * sigmoid(x)"""
    if HAS_TORCH and isinstance(x, torch.Tensor):
        return x * torch.sigmoid(x)
    sig = 1.0 / (1.0 + np.exp(-x.astype(np.float64)))
    return (x * sig).astype(np.float32)


def l2_norm_per_head(x, n_heads, head_dim, eps=1e-6):
    """L2 normalize per head."""
    if HAS_TORCH and isinstance(x, torch.Tensor):
        x = x.float().reshape(n_heads, head_dim)
        norms = torch.sqrt(torch.sum(x * x, dim=1, keepdim=True) + eps)
        return (x / norms).reshape(-1)
    else:
        x = x.astype(np.float32).reshape(n_heads, head_dim)
        norms = np.sqrt(np.sum(x * x, axis=1, keepdims=True) + eps)
        return (x / norms).reshape(-1)


def main():
    print("=" * 80)
    print("ParoQuant DeltaNet Layer 0 Debug — Qwen3.5-0.8B-PARO")
    print("=" * 80)
    print(f"Model: {MODEL_DIR}")
    print(f"Token: {TOKEN_ID}")
    print(f"Config: dim={DIM}, k_dim={K_DIM}, v_dim={V_DIM}, qkv_dim={QKV_DIM}")
    print(f"ParoQuant: krot={KROT}, group_size={GROUP_SIZE}")
    print()

    # Load model
    print("Loading model...")
    model = safe_open(os.path.join(MODEL_DIR, "model.safetensors"), framework="numpy")

    # Also open as raw bytes for rotation metadata
    with open(os.path.join(MODEL_DIR, "model.safetensors"), "rb") as f:
        header_len = struct.unpack('<Q', f.read(8))[0]
        header_json = json.loads(f.read(header_len))
        data_offset_base = 8 + header_len

    def load_raw_bytes(name):
        info = header_json[name]
        start, end = info["data_offsets"]
        with open(os.path.join(MODEL_DIR, "model.safetensors"), "rb") as f:
            f.seek(data_offset_base + start)
            return f.read(end - start)

    # =========================================================================
    # Step 1: Embedding lookup
    # =========================================================================
    print("--- Step 1: Embedding ---")
    embed_weight = model.get_tensor("model.language_model.embed_tokens.weight")  # [248320, 1024] F16
    x = embed_weight[TOKEN_ID].astype(np.float32)  # [1024]
    print(f"  embed[{TOKEN_ID}][0:8] = {fmt_arr(x)}")

    if HAS_TORCH:
        x = torch.from_numpy(x)

    # =========================================================================
    # Step 2: RMSNorm
    # =========================================================================
    print("\n--- Step 2: RMSNorm (input_layernorm) ---")
    norm_w = model.get_tensor("model.language_model.layers.0.input_layernorm.weight")
    norm_w = norm_w.astype(np.float32)  # F16 -> F32

    if HAS_TORCH:
        norm_w_t = torch.from_numpy(norm_w)
        x_normed = rmsnorm(x, norm_w_t, NORM_EPS)
    else:
        x_normed = rmsnorm(x, norm_w, NORM_EPS)

    print(f"  norm_weight[0:8] = {fmt_arr(norm_w)}")
    print(f"  x_normed[0:8]    = {fmt_arr(x_normed)}")

    # =========================================================================
    # Step 3: wqkv GEMV (ParoQuant: rotate + AWQ dequant)
    # =========================================================================
    print("\n--- Step 3: wqkv GEMV (ParoQuant rotated, {DIM} -> {QKV_DIM}) ---")

    qw_bytes = load_raw_bytes("model.language_model.layers.0.linear_attn.in_proj_qkv.qweight")
    qz_bytes = load_raw_bytes("model.language_model.layers.0.linear_attn.in_proj_qkv.qzeros")
    sc_bytes = load_raw_bytes("model.language_model.layers.0.linear_attn.in_proj_qkv.scales")
    pairs_bytes = load_raw_bytes("model.language_model.layers.0.linear_attn.in_proj_qkv.pairs")
    theta_bytes = load_raw_bytes("model.language_model.layers.0.linear_attn.in_proj_qkv.theta")
    cs_bytes = load_raw_bytes("model.language_model.layers.0.linear_attn.in_proj_qkv.channel_scales")

    print(f"  qweight size = {len(qw_bytes)} bytes, shape = {header_json['model.language_model.layers.0.linear_attn.in_proj_qkv.qweight']['shape']}")
    print(f"  scales  size = {len(sc_bytes)} bytes, shape = {header_json['model.language_model.layers.0.linear_attn.in_proj_qkv.scales']['shape']}")
    print(f"  pairs   size = {len(pairs_bytes)} bytes, shape = {header_json['model.language_model.layers.0.linear_attn.in_proj_qkv.pairs']['shape']}")
    print(f"  theta   size = {len(theta_bytes)} bytes, shape = {header_json['model.language_model.layers.0.linear_attn.in_proj_qkv.theta']['shape']}")
    print(f"  cs      size = {len(cs_bytes)} bytes, shape = {header_json['model.language_model.layers.0.linear_attn.in_proj_qkv.channel_scales']['shape']}")

    # Show rotation metadata
    pairs_arr = np.frombuffer(pairs_bytes, dtype=np.int16).reshape(KROT, DIM)
    theta_arr = np.frombuffer(theta_bytes, dtype=np.float16).reshape(KROT, DIM // 2)
    cs_arr = np.frombuffer(cs_bytes, dtype=np.float16).flatten()

    print(f"  pairs[0, 0:8]  = {pairs_arr[0, :8]}")
    print(f"  theta[0, 0:8]  = {theta_arr[0, :8].astype(np.float32)}")
    print(f"  ch_scales[0:8] = {cs_arr[:8].astype(np.float32)}")

    # Show x_normed pre-rotation
    if HAS_TORCH:
        x_normed_np = x_normed.numpy()
    else:
        x_normed_np = x_normed

    # Apply rotation to see the rotated x
    cs_f32 = cs_arr.astype(np.float32)
    if HAS_TORCH:
        pairs_t = torch.from_numpy(pairs_arr.copy())
        theta_t = torch.from_numpy(theta_arr.copy().astype(np.float32))
        cs_t = torch.from_numpy(cs_f32.copy())
        x_rot = givens_rotate(x_normed.clone(), pairs_t, theta_t, cs_t, DIM, KROT, GROUP_SIZE)
        print(f"  x_rotated[0:8] = {fmt_arr(x_rot)}")
    else:
        x_rot = givens_rotate(x_normed_np.copy(), pairs_arr, theta_arr.astype(np.float32), cs_f32, DIM, KROT, GROUP_SIZE)
        print(f"  x_rotated[0:8] = {fmt_arr(x_rot)}")

    # Full GEMV with rotation
    print("  Computing rotated GEMV (this takes a while for full matrix)...")
    qkv_result = paro_rotate_gemv(
        qw_bytes, qz_bytes, sc_bytes, pairs_bytes, theta_bytes, cs_bytes,
        x_normed if HAS_TORCH else x_normed_np,
        DIM, QKV_DIM, GROUP_SIZE, KROT
    )
    if HAS_TORCH:
        qkv_result = torch.from_numpy(qkv_result) if isinstance(qkv_result, np.ndarray) else qkv_result
    print(f"  qkv[0:8] = {fmt_arr(qkv_result)}")

    # =========================================================================
    # Step 4: wz GEMV (ParoQuant rotated, dim -> v_dim=2048)
    # =========================================================================
    print(f"\n--- Step 4: wz GEMV (ParoQuant rotated, {DIM} -> {V_DIM}) ---")

    z_qw = load_raw_bytes("model.language_model.layers.0.linear_attn.in_proj_z.qweight")
    z_qz = load_raw_bytes("model.language_model.layers.0.linear_attn.in_proj_z.qzeros")
    z_sc = load_raw_bytes("model.language_model.layers.0.linear_attn.in_proj_z.scales")
    z_pairs = load_raw_bytes("model.language_model.layers.0.linear_attn.in_proj_z.pairs")
    z_theta = load_raw_bytes("model.language_model.layers.0.linear_attn.in_proj_z.theta")
    z_cs = load_raw_bytes("model.language_model.layers.0.linear_attn.in_proj_z.channel_scales")

    print("  Computing rotated GEMV...")
    z_result = paro_rotate_gemv(
        z_qw, z_qz, z_sc, z_pairs, z_theta, z_cs,
        x_normed if HAS_TORCH else x_normed_np,
        DIM, V_DIM, GROUP_SIZE, KROT
    )
    if HAS_TORCH:
        z_result = torch.from_numpy(z_result) if isinstance(z_result, np.ndarray) else z_result
    print(f"  z[0:8] = {fmt_arr(z_result)}")

    # =========================================================================
    # Step 5: w_alpha GEMV (FP16, NO rotation)
    # =========================================================================
    print(f"\n--- Step 5: w_alpha GEMV (FP16, NO rotation, {DIM} -> {LINEAR_NUM_VALUE_HEADS}) ---")

    alpha_w_bytes = load_raw_bytes("model.language_model.layers.0.linear_attn.in_proj_a.weight")
    alpha_raw = fp16_gemv(alpha_w_bytes, x_normed if HAS_TORCH else x_normed_np, LINEAR_NUM_VALUE_HEADS, DIM)
    print(f"  alpha_raw[0:8] = {fmt_arr(alpha_raw)}")

    # =========================================================================
    # Step 6: w_beta GEMV (FP16, NO rotation)
    # =========================================================================
    print(f"\n--- Step 6: w_beta GEMV (FP16, NO rotation, {DIM} -> {LINEAR_NUM_VALUE_HEADS}) ---")

    beta_w_bytes = load_raw_bytes("model.language_model.layers.0.linear_attn.in_proj_b.weight")
    beta_raw = fp16_gemv(beta_w_bytes, x_normed if HAS_TORCH else x_normed_np, LINEAR_NUM_VALUE_HEADS, DIM)
    print(f"  beta_raw[0:8] = {fmt_arr(beta_raw)}")

    # =========================================================================
    # Step 7: Sigmoid/alpha_gate
    # =========================================================================
    print(f"\n--- Step 7: Sigmoid(beta + dt_bias), alpha_gate(alpha) ---")

    dt_bias = model.get_tensor("model.language_model.layers.0.linear_attn.dt_bias").astype(np.float32)
    a_log = model.get_tensor("model.language_model.layers.0.linear_attn.A_log").astype(np.float32)

    print(f"  dt_bias[0:8]  = {fmt_arr(dt_bias)}")
    print(f"  A_log[0:8]    = {fmt_arr(a_log)}")

    # hipfire kernel: fused_sigmoid_alpha_gate_f32
    # beta = sigmoid(beta_raw)  -- NO dt_bias for beta
    # alpha = softplus(alpha_raw + dt_bias) * (-exp(a_log))
    #   where softplus(x) = log(1 + exp(x)) with guards for large/small x
    if HAS_TORCH:
        dt_bias_t = torch.from_numpy(dt_bias)
        a_log_t = torch.from_numpy(a_log)
        beta_raw_t = beta_raw if isinstance(beta_raw, torch.Tensor) else torch.from_numpy(beta_raw)
        alpha_raw_t = alpha_raw if isinstance(alpha_raw, torch.Tensor) else torch.from_numpy(alpha_raw)

        # Beta: plain sigmoid (no dt_bias)
        beta_final = torch.sigmoid(beta_raw_t)

        # Alpha: softplus(alpha_raw + dt_bias) * (-exp(a_log))
        biased = alpha_raw_t + dt_bias_t
        sp = torch.nn.functional.softplus(biased)
        alpha_final = sp * (-torch.exp(a_log_t))
    else:
        beta_final = 1.0 / (1.0 + np.exp(-beta_raw.astype(np.float64)))
        biased = (alpha_raw + dt_bias).astype(np.float64)
        sp = np.log(1.0 + np.exp(np.clip(biased, -20, 20)))
        sp = np.where(biased > 20.0, biased, np.where(biased < -20.0, np.exp(biased), sp))
        alpha_final = (sp * (-np.exp(a_log.astype(np.float64)))).astype(np.float32)
        beta_final = beta_final.astype(np.float32)

    print(f"  beta_gated[0:8]  = {fmt_arr(beta_final)}")
    print(f"  alpha_gated[0:8] = {fmt_arr(alpha_final)}")

    # =========================================================================
    # Step 8: Conv1d + SiLU + split
    # =========================================================================
    print(f"\n--- Step 8: Conv1d + SiLU + split ---")

    conv_w_bytes = load_raw_bytes("model.language_model.layers.0.linear_attn.conv1d.weight")
    conv_w = np.frombuffer(conv_w_bytes, dtype=np.float16).reshape(QKV_DIM, 1, CONV_KERNEL_DIM)
    print(f"  conv_weight shape: {conv_w.shape}")
    print(f"  conv_weight[0, 0, :] = {conv_w[0, 0, :].astype(np.float32)}")

    # At pos=0: conv state is empty. Conv1d with [0,0,0,x] state.
    # Output = conv_weight[:, 0, 3] * qkv (only last tap is nonzero)
    if HAS_TORCH:
        qkv_np = qkv_result.numpy() if isinstance(qkv_result, torch.Tensor) else qkv_result
    else:
        qkv_np = qkv_result

    conv_out = qkv_np.astype(np.float32) * conv_w[:, 0, CONV_KERNEL_DIM - 1].astype(np.float32)
    print(f"  conv1d_out[0:8] = {fmt_arr(conv_out)}")

    # SiLU
    if HAS_TORCH:
        conv_out_t = torch.from_numpy(conv_out)
        silu_out = silu(conv_out_t)
        silu_np = silu_out.numpy()
    else:
        silu_out = silu(conv_out)
        silu_np = silu_out

    print(f"  silu_out[0:8]   = {fmt_arr(silu_out)}")

    # Split: q_raw[k_dim], k_raw[k_dim], v[v_dim]
    q_raw = silu_np[:K_DIM].copy()
    k_raw = silu_np[K_DIM:K_DIM * 2].copy()
    v = silu_np[K_DIM * 2:].copy()

    print(f"  q_raw[0:8] = {fmt_arr(q_raw)}")
    print(f"  k_raw[0:8] = {fmt_arr(k_raw)}")
    print(f"  v[0:8]     = {fmt_arr(v)}")

    # =========================================================================
    # Step 9: L2 norm q/k per head, scale q
    # =========================================================================
    print(f"\n--- Step 9: L2 norm + scale ---")

    if HAS_TORCH:
        q_normed = l2_norm_per_head(torch.from_numpy(q_raw), LINEAR_NUM_KEY_HEADS, LINEAR_KEY_HEAD_DIM, NORM_EPS)
        k_normed = l2_norm_per_head(torch.from_numpy(k_raw), LINEAR_NUM_KEY_HEADS, LINEAR_KEY_HEAD_DIM, NORM_EPS)
        # Scale q by 1/sqrt(head_dim)
        q_scaled = q_normed * (1.0 / math.sqrt(LINEAR_KEY_HEAD_DIM))
    else:
        q_normed = l2_norm_per_head(q_raw, LINEAR_NUM_KEY_HEADS, LINEAR_KEY_HEAD_DIM, NORM_EPS)
        k_normed = l2_norm_per_head(k_raw, LINEAR_NUM_KEY_HEADS, LINEAR_KEY_HEAD_DIM, NORM_EPS)
        q_scaled = q_normed * (1.0 / math.sqrt(LINEAR_KEY_HEAD_DIM))

    print(f"  q_normed_scaled[0:8] = {fmt_arr(q_scaled)}")
    print(f"  k_normed[0:8]        = {fmt_arr(k_normed)}")

    # =========================================================================
    # Step 10: Gated DeltaNet (pos=0)
    # =========================================================================
    print(f"\n--- Step 10: Gated DeltaNet (pos=0, S=0) ---")

    if HAS_TORCH:
        q_t = q_scaled if isinstance(q_scaled, torch.Tensor) else torch.from_numpy(q_scaled)
        k_t = k_normed if isinstance(k_normed, torch.Tensor) else torch.from_numpy(k_normed)
        v_t = torch.from_numpy(v) if isinstance(v, np.ndarray) else v
        beta_t = beta_final if isinstance(beta_final, torch.Tensor) else torch.from_numpy(beta_final)
        alpha_t = alpha_final if isinstance(alpha_final, torch.Tensor) else torch.from_numpy(alpha_final)

        # n_v_heads == n_k_heads == 16, head_dim == 128
        # No repeat-interleave needed (n_k_heads == n_v_heads)
        q_h = q_t.reshape(LINEAR_NUM_VALUE_HEADS, LINEAR_VALUE_HEAD_DIM)
        k_h = k_t.reshape(LINEAR_NUM_VALUE_HEADS, LINEAR_VALUE_HEAD_DIM)
        v_h = v_t.reshape(LINEAR_NUM_VALUE_HEADS, LINEAR_VALUE_HEAD_DIM)

        # At pos=0: S = zeros, so output = beta * (k^T * v) * q
        # Gated DeltaNet: S_new = alpha * S + beta * v outer k
        #                 out = S_new @ q
        # At pos=0: S=0, so S_new = beta * v outer k
        # out[h] = S_new[h] @ q[h] = (beta[h] * v[h] outer k[h]) @ q[h]
        #        = beta[h] * v[h] * (k[h] dot q[h])

        attn_out = torch.zeros(LINEAR_NUM_VALUE_HEADS, LINEAR_VALUE_HEAD_DIM)
        for h in range(LINEAR_NUM_VALUE_HEADS):
            kq_dot = torch.dot(k_h[h], q_h[h])
            attn_out[h] = beta_t[h] * v_h[h] * kq_dot

        attn_out_flat = attn_out.reshape(-1)
    else:
        q_h = q_scaled.reshape(LINEAR_NUM_VALUE_HEADS, LINEAR_VALUE_HEAD_DIM)
        k_h = k_normed.reshape(LINEAR_NUM_VALUE_HEADS, LINEAR_VALUE_HEAD_DIM)
        v_h = v.reshape(LINEAR_NUM_VALUE_HEADS, LINEAR_VALUE_HEAD_DIM)

        attn_out = np.zeros((LINEAR_NUM_VALUE_HEADS, LINEAR_VALUE_HEAD_DIM), dtype=np.float32)
        for h in range(LINEAR_NUM_VALUE_HEADS):
            kq_dot = np.dot(k_h[h], q_h[h])
            attn_out[h] = beta_final[h] * v_h[h] * kq_dot

        attn_out_flat = attn_out.reshape(-1)

    print(f"  attn_out[0:8] = {fmt_arr(attn_out_flat)}")

    # =========================================================================
    # Step 11: Gated norm (norm(attn_out) * sigmoid(z))
    # =========================================================================
    print(f"\n--- Step 11: Gated norm ---")

    norm_w_attn = model.get_tensor("model.language_model.layers.0.linear_attn.norm.weight").astype(np.float32)
    print(f"  norm.weight[0:8] = {fmt_arr(norm_w_attn)}")

    if HAS_TORCH:
        # Per-head group norm then gate by sigmoid(z)
        z_t = z_result if isinstance(z_result, torch.Tensor) else torch.from_numpy(z_result)
        attn_h = attn_out_flat.reshape(LINEAR_NUM_VALUE_HEADS, LINEAR_VALUE_HEAD_DIM)
        norm_w_t = torch.from_numpy(norm_w_attn)

        normed = torch.zeros_like(attn_h)
        for h in range(LINEAR_NUM_VALUE_HEADS):
            head_vec = attn_h[h].float()
            rms = torch.rsqrt(torch.mean(head_vec * head_vec) + NORM_EPS)
            normed[h] = head_vec * norm_w_t * rms

        # Gate with sigmoid(z)
        z_gate = torch.sigmoid(z_t.float())
        dn_normed = normed.reshape(-1) * z_gate
    else:
        z_np = z_result if isinstance(z_result, np.ndarray) else np.array(z_result, dtype=np.float32)
        attn_h = attn_out_flat.reshape(LINEAR_NUM_VALUE_HEADS, LINEAR_VALUE_HEAD_DIM)

        normed = np.zeros_like(attn_h)
        for h in range(LINEAR_NUM_VALUE_HEADS):
            head_vec = attn_h[h].astype(np.float64)
            rms = 1.0 / np.sqrt(np.mean(head_vec * head_vec) + NORM_EPS)
            normed[h] = (head_vec * norm_w_attn * rms).astype(np.float32)

        z_gate = 1.0 / (1.0 + np.exp(-z_np.astype(np.float64)))
        dn_normed = (normed.reshape(-1) * z_gate).astype(np.float32)

    print(f"  dn_normed[0:8] = {fmt_arr(dn_normed)}")

    print("\n" + "=" * 80)
    print("SUMMARY of key intermediates for hipfire comparison:")
    print("=" * 80)
    print(f"  x_normed[0:8]        = {fmt_arr(x_normed)}")
    print(f"  x_rotated_qkv[0:8]   = {fmt_arr(x_rot)}")
    print(f"  qkv[0:8]             = {fmt_arr(qkv_result)}")
    print(f"  z[0:8]               = {fmt_arr(z_result)}")
    print(f"  alpha_raw[0:8]       = {fmt_arr(alpha_raw)}")
    print(f"  beta_raw[0:8]        = {fmt_arr(beta_raw)}")
    print(f"  beta_gated[0:4]      = {fmt_arr(beta_final, 4)}")
    print(f"  alpha_gated[0:4]     = {fmt_arr(alpha_final, 4)}")
    print(f"  q_raw[0:8]           = {fmt_arr(q_raw)}")
    print(f"  k_raw[0:8]           = {fmt_arr(k_raw)}")
    print(f"  v[0:8]               = {fmt_arr(v)}")
    print(f"  q_normed_scaled[0:8] = {fmt_arr(q_scaled)}")
    print(f"  k_normed[0:8]        = {fmt_arr(k_normed)}")
    print(f"  attn_out[0:8]        = {fmt_arr(attn_out_flat)}")
    print(f"  dn_normed[0:8]       = {fmt_arr(dn_normed)}")


if __name__ == "__main__":
    main()
