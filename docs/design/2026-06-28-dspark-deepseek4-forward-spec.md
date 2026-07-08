# DSpark (DeepSeek-V4) draft-module forward — implementation blueprint

Line-accurate spec extracted from `inference/model.py` of
`deepseek-ai/DeepSeek-V4-Flash-DSpark`. This is the blueprint for the hipfire
`hipfire-arch-deepseek4` DSpark drafter. Branch `feature/dspark-deepseek4`.

## Dims (V4-Flash)
dim=4096, hc_mult=4, block_size=5, noise_token_id=128799,
target_layer_ids=[40,41,42] (len 3), markov_rank=256, vocab=129280,
n_mtp_layers=3 (stages mtp.0/1/2), n_layers=43 (target depth), head_dim=512,
rope_head_dim=64, n_heads=64, o_groups=8, o_lora_rank=1024, window_size=128.
**DSpark stages are sliding-window-only (compress_ratio==0)** — the plain SWA MLA
path, NOT the compressed/indexer MLA used by mid target layers.

## Orchestration
```
forward_spec(input_ids, main_hidden, start_pos):
    h, main_x = mtp[0].forward_embed(main_hidden, input_ids)   # built ONCE
    for s in mtp: h = s.forward(h, start_pos, input_ids, main_x)  # 3 stages chain h, share main_x
    if start_pos==0: return None                               # prefill = warm KV only
    return mtp[-1].forward_head(h, input_ids)                  # -> output_ids[b,6], logits[b,5,V], confidence[b,5]
```

## main_hidden (captured in the TARGET forward, Phase 2)
At each target layer i in [40,41,42], after the block runs, take the HC residual
`[b,s,hc_mult,4096]`, **mean-pool over hc_mult** → `[b,s,4096]`; concat the 3 →
`main_hidden [b,s,12288]`. One per decode step (s=1 decode / s=S prefill).

## forward_embed (stage 0 only)
- `main_x = main_norm(main_proj(main_hidden))`  (main_proj: Linear 12288→4096, no bias; main_norm RMS) → `[b,*,4096]`.
- noise block ids `[b,5]` = `[real_token, NOISE×4]` (slot0 = committed input token).
- `x = embed(noise_ids)` → `[b,5,4096]`, then HC-expand `unsqueeze(2).repeat(1,1,4,1)` → `x [b,5,4,4096]`.

## DSparkBlock.forward (decode, start_pos>0) = parent Block.forward
Per stage, on `x[b,5,4,4096]` with `main_x[b,1,4096]`:
1. residual=x; (x,post,comb)=hc_pre(x, hc_attn_fn, hc_attn_scale, hc_attn_base); x=attn_norm(x);
   x=DSparkAttention(x, start_pos, main_x); x=hc_post(x,residual,post,comb)
2. residual=x; (x,post,comb)=hc_pre(x, hc_ffn_*); x=ffn_norm(x); x=MoE(x); x=hc_post(...)
**main_x enters ONLY attention (as KV). Not added/concat to hidden, not in HC/MoE.**

### HC mixing
- `hc_pre`: flatten hc→`[b,s,16384]`; rms; `mixes=Linear(hc_fn[24,16384])·rsqrt`;
  `hc_split_sinkhorn(mixes, scale[3], base[24], hc_mult=4, iters=20)` → pre[b,s,4], post[b,s,4], comb[b,s,4,4];
  reduce `y=Σ_hc pre·x → [b,s,4096]`. (mix_hc=(2+hc_mult)*hc_mult=24.)
- `hc_post`: `y[hc]=post[hc]·attn_out + Σ_j comb[hc,j]·residual[j]` → `[b,s,4,4096]`.
- `hc_head` (last stage, head-side): **sigmoid gate, NOT sinkhorn**:
  `mixes=Linear(hc_head_fn[4,16384])·rsqrt`; `pre=sigmoid(mixes·hc_head_scale[1]+hc_head_base[4])+eps`;
  `y=Σ pre·x → [b,5,4096]`.

### DSparkAttention(x, start_pos, main_x) — the crux
- KV side from main_x: `main_kv = kv_norm(wkv(main_x))[b,seqlen,512]`; RoPE last 64 dims.
- prefill (start_pos==0): write main_kv into sliding-window ring `kv_cache[:, start_pos%win]`, return x.
- decode: query from x: `q=q_norm(wq_a(x))`; `q=wq_b(q)→[b,5,64,512]`; per-head qk-norm; RoPE last 64.
  block kv: `kv=kv_norm(wkv(x))[b,5,512]`; RoPE last 64.
  positions: block slots get `freqs[start_pos+seqlen : +block_size]` = position+1..+5; main_kv gets `start_pos`.
  commit `kv_cache[:, start_pos%win]=main_kv`; `kv=cat([kv_cache(window), block_kv(5)])[b,win+5,512]`.
  `o=sparse_attn(q, kv, attn_sink[h], topk_idxs, softmax_scale)`; inverse-RoPE o; grouped low-rank
  `o=einsum(o, wo_a[groups,o_lora,*]); x=wo_b(o.flatten)→[b,5,4096]`.
  `topk_idxs` (same row broadcast to all 5 slots, bidirectional): `[0..min(win,start_pos+1)-1] ++ [win..win+4]`.

## forward_head (last stage mtp.2)
```
x = hc_head(x)                         # [b,5,4096]; also fed to confidence
logits = head(norm(x), full=True)      # shared lm_head -> [b,5,vocab]
out[b,6]; out[:,0]=input_ids
for i in 0..5:
    bias, emb = markov_head(out[:,i])  # bias=markov_w2(markov_w1(tok)) [b,vocab]; emb [b,256]
    logits[:,i] += bias
    out[:,i+1] = sample(logits[:,i], temp)   # greedy if temp==0
confidence = confidence_head(cat[x, stack(embs)[b,5,256]])  # Linear(4352→1) -> [b,5]
return out[b,6], logits[b,5,vocab], confidence[b,5]
```
markov_w1=Emb(vocab→256), markov_w2=Linear(256→vocab). confidence_head.proj=Linear(4352→1, fp32).

## Stage tensor ownership
- mtp.0: full block + `main_proj`,`main_norm` (embed/main_x owner).
- mtp.1: plain block (HC + SWA attn + MoE) only.
- mtp.2: full block + `hc_head_*`,`norm`,`markov_head.{w1,w2}`,`confidence_head.proj` (head owner).
All stages: attn_norm, ffn_norm, attn.{q_norm,kv_norm,attn_sink,wq_a,wq_b,wkv,wo_a,wo_b},
ffn.{gate,shared_experts,experts.0..255}, hc_attn_{fn,base,scale}, hc_ffn_{fn,base,scale}.

## Port notes / reuse
- Reuse hipfire deepseek4 HC (`mhc_pre`/`hc_*_mix`), MoE (`ffn_stub`/`ffn_routed`), SWA MLA kernels.
- NEW vs existing `mtp_forward`: input from main_proj+noise-block (not e_proj/h_proj); block-batched
  (5 query slots) SWA attention with main_kv ring; 3-stage chain; markov GPU kernel + sequential
  sampling; confidence. The shared lm_head + greedy accept reuse the existing verify path.
