"""Pure-Python ParoQuant inference for Qwen3-0.6B-PARO — no CUDA kernel needed."""
import torch
import math
import json
from safetensors.torch import load_file

model_path = '/home/bjoern/.hipfire/models/Qwen3-0.6B-PARO'

print('Loading config...')
with open(f'{model_path}/config.json') as f:
    config = json.load(f)

dim = config['hidden_size']       # 1024
n_layers = config['num_hidden_layers']  # 28
n_heads = config['num_attention_heads']  # 16
n_kv_heads = config['num_key_value_heads']  # 8
head_dim = config['head_dim']     # 128
hidden_dim = config['intermediate_size']  # 3072
vocab_size = config['vocab_size'] # 151936
eps = config.get('rms_norm_eps', 1e-6)
rope_theta = config.get('rope_theta', 1000000.0)
gs = config['quantization_config']['group_size']  # 128
krot = config['quantization_config']['krot']  # 8

print(f'dim={dim}, layers={n_layers}, heads={n_heads}, kv_heads={n_kv_heads}, head_dim={head_dim}')

print('Loading tensors...')
tensors = load_file(f'{model_path}/model.safetensors')

def rmsnorm(x, w, eps=1e-6):
    rms = (x.pow(2).mean() + eps).sqrt()
    return (x / rms) * w

def rotate_gemv(x, prefix, out_dim):
    """ParoQuant: channel_scale → Givens rotate → AWQ dequant GEMV."""
    qw = tensors[f'{prefix}.qweight']
    qz = tensors[f'{prefix}.qzeros']
    sc = tensors[f'{prefix}.scales'].float()
    pairs = tensors[f'{prefix}.pairs']
    theta = tensors[f'{prefix}.theta']
    cs = tensors[f'{prefix}.channel_scales'].squeeze(0).float()
    in_dim = qw.shape[0]

    # Channel scale
    x_s = x * cs

    # Givens rotation
    for rot in range(krot):
        xn = x_s.clone()
        for g in range(in_dim // gs):
            ch = g * gs
            for tid in range(gs // 2):
                i = pairs[rot, ch + 2*tid].item()
                j = pairs[rot, ch + 2*tid + 1].item()
                a = theta[rot, g*(gs//2)+tid].float().item()
                c, s = math.cos(a), math.sin(a)
                xi, xj = x_s[ch+i].item(), x_s[ch+j].item()
                xn[ch+i] = xi*c + xj*s
                xn[ch+j] = xj*c - xi*s
        x_s = xn

    # AWQ dequant GEMV
    y = torch.zeros(out_dim)
    qw_cols = out_dim // 8
    for m in range(out_dim):
        acc = 0.0
        for k in range(in_dim):
            g = k // gs
            nib = ((qw[k, m//8].item() >> ((m%8)*4)) & 0xF)
            zero = ((qz[g, m//8].item() >> ((m%8)*4)) & 0xF)
            scale = sc[g, m].item()
            acc += scale * (nib - zero) * x_s[k].item()
        y[m] = acc
    return y

def fp16_gemv(x, name, out_dim):
    """Plain FP16 weight GEMV (for unquantized layers)."""
    w = tensors[name].float()  # [out_dim, in_dim]
    return w @ x

def rope_halfsplit(q, k, pos, n_heads, n_kv_heads, head_dim, theta_base):
    """Half-split RoPE (HF rotate_half convention)."""
    half = head_dim // 2
    for h in range(n_heads):
        base = h * head_dim
        for i in range(half):
            freq = 1.0 / (theta_base ** (2.0 * i / head_dim))
            angle = pos * freq
            cos_a, sin_a = math.cos(angle), math.sin(angle)
            v0 = q[base + i].item()
            v1 = q[base + i + half].item()
            q[base + i] = v0 * cos_a - v1 * sin_a
            q[base + i + half] = v0 * sin_a + v1 * cos_a
    for h in range(n_kv_heads):
        base = h * head_dim
        for i in range(half):
            freq = 1.0 / (theta_base ** (2.0 * i / head_dim))
            angle = pos * freq
            cos_a, sin_a = math.cos(angle), math.sin(angle)
            v0 = k[base + i].item()
            v1 = k[base + i + half].item()
            k[base + i] = v0 * cos_a - v1 * sin_a
            k[base + i + half] = v0 * sin_a + v1 * cos_a

# Load tokenizer
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
prompt = '2+2='
tokens = tokenizer.encode(prompt)
print(f'Tokens: {tokens}')

# Only process first token for speed (full layer 0 forward)
tok_id = tokens[0]
print(f'\n=== Processing token {tok_id} at pos=0 ===')

# Embedding
emb = tensors['model.embed_tokens.weight']
x = emb[tok_id].float().clone()
print(f'embed rms={x.pow(2).mean().sqrt().item():.6f}')

# Layer 0 forward
print('\n--- Layer 0 ---')
# RMSNorm
norm_w = tensors['model.layers.0.input_layernorm.weight'].float()
x_n = rmsnorm(x, norm_w, eps)
print(f'x_normed[0:4] = {x_n[:4].tolist()}')

# Q, K, V projections (only first 8 elements for speed)
q_dim = n_heads * head_dim      # 2048
kv_dim = n_kv_heads * head_dim  # 1024

print('Computing Q projection (rotated GEMV, this is slow)...')
q = rotate_gemv(x_n, 'model.layers.0.self_attn.q_proj', q_dim)
print(f'q[0:8] = {q[:8].tolist()}')

print('Computing K projection...')
k = rotate_gemv(x_n, 'model.layers.0.self_attn.k_proj', kv_dim)
print(f'k[0:8] = {k[:8].tolist()}')

print('Computing V projection...')
v = rotate_gemv(x_n, 'model.layers.0.self_attn.v_proj', kv_dim)
print(f'v[0:8] = {v[:8].tolist()}')

# QK norm
q_norm_w = tensors['model.layers.0.self_attn.q_norm.weight'].float()
k_norm_w = tensors['model.layers.0.self_attn.k_norm.weight'].float()
# Per-head RMSNorm
for h in range(n_heads):
    q_h = q[h*head_dim:(h+1)*head_dim]
    q[h*head_dim:(h+1)*head_dim] = rmsnorm(q_h, q_norm_w, eps)
for h in range(n_kv_heads):
    k_h = k[h*head_dim:(h+1)*head_dim]
    k[h*head_dim:(h+1)*head_dim] = rmsnorm(k_h, k_norm_w, eps)
print(f'q_normed[0:8] = {q[:8].tolist()}')
print(f'k_normed[0:8] = {k[:8].tolist()}')

# RoPE
rope_halfsplit(q, k, 0, n_heads, n_kv_heads, head_dim, rope_theta)
print(f'q_rope[0:8] = {q[:8].tolist()}')
print(f'k_rope[0:8] = {k[:8].tolist()}')

# Attention (pos=0, single token → trivial: attn_out = v with GQA repeat)
# With 16 Q heads and 8 KV heads, each KV head covers 2 Q heads
# At pos=0: softmax([q·k/sqrt(d)]) = [1.0], so attn_out = v (repeated for GQA)
attn_out = torch.zeros(q_dim)
heads_per_kv = n_heads // n_kv_heads
for h in range(n_heads):
    kv_h = h // heads_per_kv
    attn_out[h*head_dim:(h+1)*head_dim] = v[kv_h*head_dim:(kv_h+1)*head_dim]
print(f'attn_out[0:8] = {attn_out[:8].tolist()}')

# O projection (rotated)
print('Computing O projection...')
o = rotate_gemv(attn_out, 'model.layers.0.self_attn.o_proj', dim)
print(f'o[0:8] = {o[:8].tolist()}')

# Residual
x = x + o
print(f'x_after_attn[0:4] = {x[:4].tolist()}')
print(f'x_after_attn rms = {x.pow(2).mean().sqrt().item():.6f}')

# FFN
ffn_norm_w = tensors['model.layers.0.post_attention_layernorm.weight'].float()
x_n2 = rmsnorm(x, ffn_norm_w, eps)

print('Computing gate projection...')
gate = rotate_gemv(x_n2, 'model.layers.0.mlp.gate_proj', hidden_dim)
print(f'gate[0:8] = {gate[:8].tolist()}')

print('Computing up projection...')
up = rotate_gemv(x_n2, 'model.layers.0.mlp.up_proj', hidden_dim)

# SiLU(gate) * up
ffn_hidden = torch.nn.functional.silu(gate) * up

print('Computing down projection...')
down = rotate_gemv(ffn_hidden, 'model.layers.0.mlp.down_proj', dim)
print(f'down[0:8] = {down[:8].tolist()}')

# Residual
x = x + down
print(f'\n=== After layer 0 ===')
print(f'x rms = {x.pow(2).mean().sqrt().item():.6f}')
print(f'x[0:8] = {x[:8].tolist()}')
