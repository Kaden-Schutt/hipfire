"""Full 28-layer Qwen3-0.6B-PARO with CORRECTED AWQ nibble order."""
import torch, math, json, time
from safetensors.torch import load_file

model_path = '/home/bjoern/.hipfire/models/Qwen3-0.6B-PARO'
with open(f'{model_path}/config.json') as f:
    config = json.load(f)

dim=config['hidden_size']; n_layers=config['num_hidden_layers']
n_heads=config['num_attention_heads']; n_kv_heads=config['num_key_value_heads']
head_dim=config['head_dim']; hidden_dim=config['intermediate_size']
vocab_size=config['vocab_size']; eps=config.get('rms_norm_eps',1e-6)
rope_theta=config.get('rope_theta',1000000.0)
gs=config['quantization_config']['group_size']; krot=config['quantization_config']['krot']

# AWQ nibble reordering: element m is at position reverse_order[m%8] in the I32
AWQ_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]
AWQ_REVERSE = [0, 2, 4, 6, 1, 3, 5, 7]  # reverse_order[m] = position of element m

print(f'Config: dim={dim}, layers={n_layers}, heads={n_heads}, kv={n_kv_heads}, hd={head_dim}')
print('Loading tensors...')
tensors = load_file(f'{model_path}/model.safetensors')

def rmsnorm(x, w):
    return (x / (x.pow(2).mean(-1, keepdim=True) + eps).sqrt()) * w

def dequant_weight_awq(prefix):
    """Dequant with CORRECTED AWQ nibble order."""
    qw = tensors[f'{prefix}.qweight'].long()
    qz = tensors[f'{prefix}.qzeros'].long()
    sc = tensors[f'{prefix}.scales'].float()
    in_dim = qw.shape[0]; out_dim = sc.shape[1]
    W = torch.zeros(out_dim, in_dim, dtype=torch.float32)
    for m in range(out_dim):
        col = m // 8
        shift = AWQ_REVERSE[m % 8] * 4  # CORRECTED: use reverse order
        nibs = ((qw[:, col] >> shift) & 0xF).float()
        zeros = torch.zeros(in_dim); scales = torch.zeros(in_dim)
        zero_shift = AWQ_REVERSE[m % 8] * 4  # zeros also use AWQ order
        for g in range(in_dim // gs):
            z = ((qz[g, col].item() >> zero_shift) & 0xF)
            zeros[g*gs:(g+1)*gs] = z
            scales[g*gs:(g+1)*gs] = sc[g, m].item()
        W[m, :] = scales * (nibs - zeros)
    return W

def givens_rotate(x, prefix):
    pairs = tensors[f'{prefix}.pairs']; theta = tensors[f'{prefix}.theta']
    cs = tensors[f'{prefix}.channel_scales'].squeeze(0).float()
    in_dim = x.shape[-1]; x_s = x * cs
    for rot in range(krot):
        xn = x_s.clone()
        for g in range(in_dim // gs):
            ch = g * gs
            for tid in range(gs // 2):
                i=pairs[rot,ch+2*tid].item(); j=pairs[rot,ch+2*tid+1].item()
                a=theta[rot,g*(gs//2)+tid].float().item(); c,s=math.cos(a),math.sin(a)
                xi,xj=x_s[...,ch+i].clone(),x_s[...,ch+j].clone()
                xn[...,ch+i]=xi*c+xj*s; xn[...,ch+j]=xj*c-xi*s
        x_s = xn
    return x_s

def rope_halfsplit(q, k, pos):
    half = head_dim // 2
    for h in range(n_heads):
        b=h*head_dim
        for i in range(half):
            freq=1.0/(rope_theta**(2.0*i/head_dim)); angle=pos*freq
            ca,sa=math.cos(angle),math.sin(angle)
            v0,v1=q[b+i].item(),q[b+i+half].item()
            q[b+i]=v0*ca-v1*sa; q[b+i+half]=v0*sa+v1*ca
    for h in range(n_kv_heads):
        b=h*head_dim
        for i in range(half):
            freq=1.0/(rope_theta**(2.0*i/head_dim)); angle=pos*freq
            ca,sa=math.cos(angle),math.sin(angle)
            v0,v1=k[b+i].item(),k[b+i+half].item()
            k[b+i]=v0*ca-v1*sa; k[b+i+half]=v0*sa+v1*ca

print('Dequanting weights (AWQ-corrected)...')
t0 = time.time()
layers_w = []
for i in range(n_layers):
    p = f'model.layers.{i}'
    print(f'  layer {i}/{n_layers}...', end=' ', flush=True)
    lt = time.time()
    lw = {
        'an': tensors[f'{p}.input_layernorm.weight'].float(),
        'fn': tensors[f'{p}.post_attention_layernorm.weight'].float(),
        'wq': dequant_weight_awq(f'{p}.self_attn.q_proj'),
        'wk': dequant_weight_awq(f'{p}.self_attn.k_proj'),
        'wv': dequant_weight_awq(f'{p}.self_attn.v_proj'),
        'wo': dequant_weight_awq(f'{p}.self_attn.o_proj'),
        'wg': dequant_weight_awq(f'{p}.mlp.gate_proj'),
        'wu': dequant_weight_awq(f'{p}.mlp.up_proj'),
        'wd': dequant_weight_awq(f'{p}.mlp.down_proj'),
        'qn': tensors[f'{p}.self_attn.q_norm.weight'].float(),
        'kn': tensors[f'{p}.self_attn.k_norm.weight'].float(),
        'qr': f'{p}.self_attn.q_proj', 'kr': f'{p}.self_attn.k_proj',
        'vr': f'{p}.self_attn.v_proj', 'or': f'{p}.self_attn.o_proj',
        'gr': f'{p}.mlp.gate_proj', 'ur': f'{p}.mlp.up_proj', 'dr': f'{p}.mlp.down_proj',
    }
    layers_w.append(lw)
    print(f'{time.time()-lt:.1f}s')

output_norm = tensors['model.norm.weight'].float()
emb = tensors['model.embed_tokens.weight'].float()
lm_head = tensors.get('lm_head.weight', emb).float()
print(f'Dequant done in {time.time()-t0:.0f}s')

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
tokens = tokenizer.encode('2+2=')
print(f'Tokens: {tokens}')

heads_per_kv = n_heads // n_kv_heads
kv_dim = n_kv_heads * head_dim; q_dim = n_heads * head_dim
kv_k = [[] for _ in range(n_layers)]; kv_v = [[] for _ in range(n_layers)]

def forward_token(tok_id, pos):
    x = emb[tok_id].clone()
    for i in range(n_layers):
        lw = layers_w[i]
        x_n = rmsnorm(x.unsqueeze(0), lw['an'].unsqueeze(0)).squeeze(0)
        q = givens_rotate(x_n, lw['qr']) @ lw['wq'].T
        k = givens_rotate(x_n, lw['kr']) @ lw['wk'].T
        v = givens_rotate(x_n, lw['vr']) @ lw['wv'].T
        q = rmsnorm(q.view(n_heads, head_dim), lw['qn'].unsqueeze(0)).view(-1)
        k = rmsnorm(k.view(n_kv_heads, head_dim), lw['kn'].unsqueeze(0)).view(-1)
        rope_halfsplit(q, k, pos)
        kv_k[i].append(k.clone()); kv_v[i].append(v.clone())
        attn_out = torch.zeros(q_dim)
        for h in range(n_heads):
            kv_h = h // heads_per_kv; q_h = q[h*head_dim:(h+1)*head_dim]
            scores = []
            for p2 in range(pos + 1):
                k_p = kv_k[i][p2][kv_h*head_dim:(kv_h+1)*head_dim]
                scores.append((q_h * k_p).sum().item() / math.sqrt(head_dim))
            max_s = max(scores); exp_s = [math.exp(s-max_s) for s in scores]
            sum_e = sum(exp_s); probs = [e/sum_e for e in exp_s]
            v_out = torch.zeros(head_dim)
            for p2 in range(pos + 1):
                v_out += probs[p2] * kv_v[i][p2][kv_h*head_dim:(kv_h+1)*head_dim]
            attn_out[h*head_dim:(h+1)*head_dim] = v_out
        o = givens_rotate(attn_out, lw['or']) @ lw['wo'].T; x = x + o
        x_n2 = rmsnorm(x.unsqueeze(0), lw['fn'].unsqueeze(0)).squeeze(0)
        gate = givens_rotate(x_n2, lw['gr']) @ lw['wg'].T
        up = givens_rotate(x_n2, lw['ur']) @ lw['wu'].T
        down = givens_rotate(torch.nn.functional.silu(gate)*up, lw['dr']) @ lw['wd'].T
        x = x + down
    return x

print(f'\nProcessing {len(tokens)} tokens...')
for pos, tok_id in enumerate(tokens):
    t0 = time.time()
    x = forward_token(tok_id, pos)
    print(f'  pos={pos} tok={tok_id}: x rms={x.pow(2).mean().sqrt().item():.4f} ({time.time()-t0:.1f}s)')

x_norm = rmsnorm(x.unsqueeze(0), output_norm.unsqueeze(0)).squeeze(0)
logits = x_norm @ lm_head.T
top10 = torch.topk(logits, 10)
print(f'\nTop-10 logits (AWQ-corrected):')
for idx, val in zip(top10.indices.tolist(), top10.values.tolist()):
    print(f'  [{idx}] = {val:.4f}  {repr(tokenizer.decode([idx]))}')
rank_19 = (logits > logits[19]).sum().item()
print(f'\nToken 19 ("4") rank: {rank_19} (logit={logits[19].item():.4f})')
