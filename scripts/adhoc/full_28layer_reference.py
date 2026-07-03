"""Full 28-layer Qwen3-0.6B-PARO forward pass with ParoQuant rotation.
Uses torch matmul for speed (not element-wise GEMV)."""
import torch
import math
import json
import time
from safetensors.torch import load_file

model_path = '/home/bjoern/.hipfire/models/Qwen3-0.6B-PARO'

with open(f'{model_path}/config.json') as f:
    config = json.load(f)

dim = config['hidden_size']
n_layers = config['num_hidden_layers']
n_heads = config['num_attention_heads']
n_kv_heads = config['num_key_value_heads']
head_dim = config['head_dim']
hidden_dim = config['intermediate_size']
vocab_size = config['vocab_size']
eps = config.get('rms_norm_eps', 1e-6)
rope_theta = config.get('rope_theta', 1000000.0)
gs = config['quantization_config']['group_size']
krot = config['quantization_config']['krot']

print(f'Config: dim={dim}, layers={n_layers}, heads={n_heads}, kv={n_kv_heads}, hd={head_dim}')
print(f'FFN: {hidden_dim}, vocab: {vocab_size}, gs={gs}, krot={krot}')

print('Loading tensors...')
t0 = time.time()
tensors = load_file(f'{model_path}/model.safetensors')
print(f'Loaded in {time.time()-t0:.1f}s')


def rmsnorm(x, w):
    return (x / (x.pow(2).mean(-1, keepdim=True) + eps).sqrt()) * w


def dequant_weight(prefix):
    """Dequant AWQ INT4 weight to dense FP32 [out_dim, in_dim]."""
    qw = tensors[f'{prefix}.qweight']  # [in_dim, out_dim/8] I32
    qz = tensors[f'{prefix}.qzeros']   # [groups, out_dim/8] I32
    sc = tensors[f'{prefix}.scales'].float()  # [groups, out_dim] F16->F32
    in_dim = qw.shape[0]
    out_dim = sc.shape[1]

    W = torch.zeros(out_dim, in_dim, dtype=torch.float32)
    qw_np = qw.numpy().astype('uint32')
    qz_np = qz.numpy().astype('uint32')

    for m in range(out_dim):
        for g in range(in_dim // gs):
            scale = sc[g, m].item()
            zero = int((qz_np[g, m // 8] >> ((m % 8) * 4)) & 0xF)
            for k_in_g in range(gs):
                k = g * gs + k_in_g
                nib = int((qw_np[k, m // 8] >> ((m % 8) * 4)) & 0xF)
                W[m, k] = scale * (nib - zero)
    return W


def dequant_weight_fast(prefix):
    """Faster dequant using vectorized operations."""
    qw = tensors[f'{prefix}.qweight'].long()  # [in_dim, out_dim/8]
    qz = tensors[f'{prefix}.qzeros'].long()   # [groups, out_dim/8]
    sc = tensors[f'{prefix}.scales'].float()   # [groups, out_dim]
    in_dim = qw.shape[0]
    out_dim = sc.shape[1]
    groups = in_dim // gs

    # Unpack all nibbles at once
    W = torch.zeros(out_dim, in_dim, dtype=torch.float32)
    for m in range(out_dim):
        col = m // 8
        shift = (m % 8) * 4
        # Extract nibbles for all input positions
        nibs = ((qw[:, col] >> shift) & 0xF).float()  # [in_dim]
        # Extract zeros per group
        zeros = torch.zeros(in_dim)
        scales = torch.zeros(in_dim)
        for g in range(groups):
            z = ((qz[g, col].item() >> shift) & 0xF)
            zeros[g*gs:(g+1)*gs] = z
            scales[g*gs:(g+1)*gs] = sc[g, m].item()
        W[m, :] = scales * (nibs - zeros)
    return W


def givens_rotate(x, prefix):
    """Apply ParoQuant Givens rotation to activation vector."""
    pairs = tensors[f'{prefix}.pairs']     # [krot, in_dim]
    theta = tensors[f'{prefix}.theta']     # [krot, in_dim/2]
    cs = tensors[f'{prefix}.channel_scales'].squeeze(0).float()  # [in_dim]
    in_dim = x.shape[-1]

    x_s = x * cs
    for rot in range(krot):
        xn = x_s.clone()
        for g in range(in_dim // gs):
            ch = g * gs
            for tid in range(gs // 2):
                i = pairs[rot, ch + 2*tid].item()
                j = pairs[rot, ch + 2*tid + 1].item()
                a = theta[rot, g*(gs//2)+tid].float().item()
                c, s = math.cos(a), math.sin(a)
                xi, xj = x_s[..., ch+i].clone(), x_s[..., ch+j].clone()
                xn[..., ch+i] = xi*c + xj*s
                xn[..., ch+j] = xj*c - xi*s
        x_s = xn
    return x_s


def paro_linear(x, prefix, out_dim):
    """ParoQuant linear: rotate(x) @ dequant(W)^T"""
    has_qw = f'{prefix}.qweight' in tensors
    if has_qw:
        x_rot = givens_rotate(x, prefix)
        W = dequant_weight_fast(prefix)
        return x_rot @ W.T
    else:
        W = tensors[f'{prefix}.weight'].float()
        return x @ W.T


def rope_halfsplit(q, k, pos):
    """Half-split RoPE for single token."""
    half = head_dim // 2
    for h in range(n_heads):
        b = h * head_dim
        for i in range(half):
            freq = 1.0 / (rope_theta ** (2.0*i/head_dim))
            angle = pos * freq
            ca, sa = math.cos(angle), math.sin(angle)
            v0, v1 = q[b+i].item(), q[b+i+half].item()
            q[b+i] = v0*ca - v1*sa
            q[b+i+half] = v0*sa + v1*ca
    for h in range(n_kv_heads):
        b = h * head_dim
        for i in range(half):
            freq = 1.0 / (rope_theta ** (2.0*i/head_dim))
            angle = pos * freq
            ca, sa = math.cos(angle), math.sin(angle)
            v0, v1 = k[b+i].item(), k[b+i+half].item()
            k[b+i] = v0*ca - v1*sa
            k[b+i+half] = v0*sa + v1*ca


# Tokenize: raw "2+2="
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
tokens = tokenizer.encode('2+2=')
print(f'Tokens: {tokens}')

# Pre-dequant all weights (slow but only once)
print('Dequanting all weights (this takes a while)...')
t0 = time.time()
layers_w = []
for i in range(n_layers):
    p = f'model.layers.{i}'
    print(f'  layer {i}/{n_layers}...', end=' ', flush=True)
    lt = time.time()
    lw = {
        'attn_norm': tensors[f'{p}.input_layernorm.weight'].float(),
        'ffn_norm': tensors[f'{p}.post_attention_layernorm.weight'].float(),
        'wq': dequant_weight_fast(f'{p}.self_attn.q_proj'),
        'wk': dequant_weight_fast(f'{p}.self_attn.k_proj'),
        'wv': dequant_weight_fast(f'{p}.self_attn.v_proj'),
        'wo': dequant_weight_fast(f'{p}.self_attn.o_proj'),
        'wg': dequant_weight_fast(f'{p}.mlp.gate_proj'),
        'wu': dequant_weight_fast(f'{p}.mlp.up_proj'),
        'wd': dequant_weight_fast(f'{p}.mlp.down_proj'),
        'q_norm': tensors[f'{p}.self_attn.q_norm.weight'].float(),
        'k_norm': tensors[f'{p}.self_attn.k_norm.weight'].float(),
        # Rotation params
        'q_rot': f'{p}.self_attn.q_proj',
        'k_rot': f'{p}.self_attn.k_proj',
        'v_rot': f'{p}.self_attn.v_proj',
        'o_rot': f'{p}.self_attn.o_proj',
        'g_rot': f'{p}.mlp.gate_proj',
        'u_rot': f'{p}.mlp.up_proj',
        'd_rot': f'{p}.mlp.down_proj',
    }
    layers_w.append(lw)
    print(f'{time.time()-lt:.1f}s')

output_norm = tensors['model.norm.weight'].float()
emb = tensors['model.embed_tokens.weight'].float()

# Check if lm_head is tied
if 'lm_head.weight' in tensors:
    lm_head = tensors['lm_head.weight'].float()
else:
    lm_head = emb  # tied
print(f'Dequant done in {time.time()-t0:.0f}s')

# KV cache
kv_cache_k = []  # [n_layers][max_pos, kv_dim]
kv_cache_v = []
for _ in range(n_layers):
    kv_cache_k.append([])
    kv_cache_v.append([])

heads_per_kv = n_heads // n_kv_heads
kv_dim = n_kv_heads * head_dim
q_dim = n_heads * head_dim

def forward_token(tok_id, pos):
    """Forward one token through all layers."""
    x = emb[tok_id].clone()

    for i in range(n_layers):
        lw = layers_w[i]

        # Attention
        x_n = rmsnorm(x.unsqueeze(0), lw['attn_norm'].unsqueeze(0)).squeeze(0)

        # Q/K/V with rotation
        q = givens_rotate(x_n, lw['q_rot']) @ lw['wq'].T
        k = givens_rotate(x_n, lw['k_rot']) @ lw['wk'].T
        v = givens_rotate(x_n, lw['v_rot']) @ lw['wv'].T

        # QK norm (per-head)
        q_heads = q.view(n_heads, head_dim)
        k_heads = k.view(n_kv_heads, head_dim)
        q = rmsnorm(q_heads, lw['q_norm'].unsqueeze(0)).view(-1)
        k = rmsnorm(k_heads, lw['k_norm'].unsqueeze(0)).view(-1)

        # RoPE
        rope_halfsplit(q, k, pos)

        # KV cache
        kv_cache_k[i].append(k.clone())
        kv_cache_v[i].append(v.clone())

        # Attention
        attn_out = torch.zeros(q_dim)
        for h in range(n_heads):
            kv_h = h // heads_per_kv
            q_h = q[h*head_dim:(h+1)*head_dim]

            # Score against all cached K
            scores = []
            for p2 in range(pos + 1):
                k_p = kv_cache_k[i][p2][kv_h*head_dim:(kv_h+1)*head_dim]
                score = (q_h * k_p).sum().item() / math.sqrt(head_dim)
                scores.append(score)

            # Softmax
            max_s = max(scores)
            exp_s = [math.exp(s - max_s) for s in scores]
            sum_e = sum(exp_s)
            probs = [e / sum_e for e in exp_s]

            # Weighted sum of V
            v_out = torch.zeros(head_dim)
            for p2 in range(pos + 1):
                v_p = kv_cache_v[i][p2][kv_h*head_dim:(kv_h+1)*head_dim]
                v_out += probs[p2] * v_p

            attn_out[h*head_dim:(h+1)*head_dim] = v_out

        # O projection with rotation
        o = givens_rotate(attn_out, lw['o_rot']) @ lw['wo'].T
        x = x + o

        # FFN
        x_n2 = rmsnorm(x.unsqueeze(0), lw['ffn_norm'].unsqueeze(0)).squeeze(0)
        gate = givens_rotate(x_n2, lw['g_rot']) @ lw['wg'].T
        up = givens_rotate(x_n2, lw['u_rot']) @ lw['wu'].T
        ffn_h = torch.nn.functional.silu(gate) * up
        down = givens_rotate(ffn_h, lw['d_rot']) @ lw['wd'].T
        x = x + down

        if i == 0 and pos == 0:
            print(f'  [L0 pos0] x[0:4]={x[:4].tolist()}, rms={x.pow(2).mean().sqrt().item():.6f}')

    return x

# Process all tokens
print(f'\nProcessing {len(tokens)} tokens...')
for pos, tok_id in enumerate(tokens):
    t0 = time.time()
    x = forward_token(tok_id, pos)
    elapsed = time.time() - t0
    print(f'  pos={pos} tok={tok_id}: x rms={x.pow(2).mean().sqrt().item():.4f} ({elapsed:.1f}s)')

# Final logits
x_norm = rmsnorm(x.unsqueeze(0), output_norm.unsqueeze(0)).squeeze(0)
logits = x_norm @ lm_head.T

top5 = torch.topk(logits, 10)
print(f'\nTop-10 logits:')
for idx, val in zip(top5.indices.tolist(), top5.values.tolist()):
    tok = tokenizer.decode([idx])
    print(f'  [{idx}] = {val:.4f}  {repr(tok)}')

# What token does hipfire produce?
print(f'\nhipfire top: [31868] = "thood"')
print(f'Expected: [19] = "4"')
# Check where 19 ranks
rank_19 = (logits > logits[19]).sum().item()
print(f'Token 19 ("4") rank: {rank_19} (logit={logits[19].item():.4f})')
