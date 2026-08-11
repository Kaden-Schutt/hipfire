# Muse Glimmer assistant (DFlash drafter) — authoritative forward contract

Source: `transformers/models/muse_glimmer_assistant/modeling_muse_glimmer_assistant.py`
(fetched to `/tmp/ga_modeling.py`; modular at `/tmp/ga_modular.py`).
This is the reference implementation — it is not inference or reconstruction.

## Shape

- 5 layers, hidden 6656, FFN 19968, 32 Q heads / 8 KV heads, head_dim 128.
- `layer_types` = `sliding_attention` x5, `sliding_window` 2048, rope theta 500000.
- `block_size` 16, `mask_token_id` 201818, `rms_norm_eps` 1e-5.
- `target_layer_ids` = `[1,13,25,37,49]`.
- Tensors (58 = 11x5 + 3): per layer `input_layernorm`, `post_attention_layernorm`,
  `self_attn.{q,k,v,o}_proj`, `self_attn.{q,k}_norm`, `mlp.{gate,up,down}_proj`;
  plus `encoder.fc`, `encoder.output_norm_enc`, `norm`.
  There is NO pre/post feedforward norm — it is a standard two-norm block.
  `q_norm`/`k_norm` are REAL WEIGHTED norms, not the target's scale-less variant.

## Context projection — `MuseGlimmerAssistantContextProjection`

```python
context_hidden_states = self.fc(context_hidden_states)      # [*, 5*6656] -> [*, 6656]
context_hidden_states = self.output_norm_enc(context_hidden_states)
```

Input is `[batch, n_prev_accepted_tokens, 5*hidden]` — **every** previous accepted
token, not one row. Computed ONCE and reused by every layer.

## Model forward

```python
context_hidden_states = self.encoder(context_hidden_states)
position_ids = arange(noise_embeds.shape[1] + context_hidden_states.shape[1]) + past_seen
hidden_states = noise_embeds                     # context is NOT added into this
for layer in self.layers:
    hidden_states = layer(hidden_states, context_hidden_states=context_hidden_states, ...)
```

## Attention — the part the current implementation gets backwards

```python
kv_hidden_states = torch.cat([context_hidden_states, hidden_states], dim=1)
query_states = self.q_proj(hidden_states)        # BLOCK only            -> B rows
key_states   = self.k_proj(kv_hidden_states)     # CONTEXT ++ BLOCK      -> ctx+B rows
value_states = self.v_proj(kv_hidden_states)     # CONTEXT ++ BLOCK      -> ctx+B rows
query_states = self.q_norm(query_states)
key_states   = self.k_norm(key_states)
```

So Q length != K/V length. Upstream comment: *"The total k/v states in Dflash are
the concatenation of the previous `context_hidden_states` (same for every layer)
and the actual projections on the diffusion window."*

`position_ids` span `ctx + block`, so RoPE is applied over the concatenated extent.

## Masking

Bidirectional, everywhere, for the queries. Upstream comment: *"The queries
corresponding to `noise_embeds` must attend bi-directionally to each other, and
causally to previous k/v states derived from the main model's
`context_hidden_states`. However, since they strictly correspond to positions
larger than the cache, this corresponds to bi-directional attention everywhere."*
Built as `create_bidirectional_sliding_window_mask` (these layers are all
`sliding_attention`, window 2048).

## How the current hipfire implementation diverges

| aspect | reference | current code |
|---|---|---|
| context use | concatenated into K/V | broadcast-ADDED into `x` |
| context rows | all previously accepted tokens | `ctx_len = 1` (last row only) |
| K/V extent | `ctx + block` | `block` only |
| positions | span `ctx + block` | `block` only (B=16) |
| Q extent | `block` | `block` (correct) |

The first four rows are why tau is 1.0: the drafter is being run with its
conditioning delivered through the wrong pathway entirely. It still produces
well-formed hidden states of plausible magnitude, which is why nothing errors and
only the acceptance rate reveals it.

Note the earlier scratch sizing `(max_ctx + block) * kv_dim` was RIGHT and was
later resized to block-only. Restore the `ctx + block` extent.
