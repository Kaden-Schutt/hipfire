# 00 - CCA Disambiguation

**Date:** 2026-05-07
**Branch:** `feat/zaya1-port-intake`
**Source:** `Zyphra/transformers@zaya1` (cloned at `/tmp/zaya-port/source-reads/zyphra-transformers`)
**Files read:** `src/transformers/models/zaya/{configuration_zaya.py, modeling_zaya.py}`

## Verdict

**RECURRENT.**

CCA carries two per-layer per-sequence state buffers across decode steps. This is unambiguous from the source: the cache class explicitly allocates them, the forward explicitly reads and writes them in the generation branch, and the operator is structurally a stateful causal Conv1d-along-time fused with a 1-step hidden-state delay for the value path.

The earlier reading (HF model card, blog post) that suggested CCA was attention-only was incomplete; the recurrence is not described in the marketing material but is plainly visible in the modeling code.

`mamba_cache_dtype: float32` in `config.json` is **dead config** - zero references across the entire `zaya/` module - but the recurrence it suggests is real, just stored at fp16 (Python default) by the actual cache class.

## Evidence

### Two state buffers, allocated per-layer

`modeling_zaya.py:187-229` - `ZayaDynamicCache(DynamicCache)`:

```python
class ZayaDynamicCache(DynamicCache):
    """Cache that includes both the KV cache and the CCA cache."""

    def __init__(self, config: ZayaConfig, batch_size: int, dtype: torch.dtype = torch.float16, ...):
        ...
        self.conv_kernel_size = 2
        self.num_layers = config.num_hidden_layers
        self.latent_k_dim = num_k_heads * head_dim                # 2 * 128 = 256
        self.latent_q_dim = num_q_heads * head_dim                # 8 * 128 = 1024  (note: cca_num_q_heads, not num_attention_heads)
        self.in_out_ch = self.latent_k_dim + self.latent_q_dim    # 1280
        self.has_previous_state = False

        self.conv_states = torch.zeros(
            self.num_layers, batch_size, self.in_out_ch, self.conv_kernel_size,
            device=device, dtype=dtype,
        )                                                         # [80, B, 1280, 2]

        self.prev_hs = torch.zeros(
            self.num_layers, batch_size, config.hidden_size,
            device=device, dtype=dtype,
        )                                                         # [80, B, 2048]
```

The cache docstring is the smoking gun: "Cache that includes **both** the KV cache and the CCA cache." Standard KV is inherited from `DynamicCache`; the CCA cache is `conv_states + prev_hs` added on top.

### Cache update is per-step, time-rolling

`modeling_zaya.py:231-237`:

```python
def update_conv_state(self, layer_idx: int, new_conv_state: torch.Tensor) -> torch.Tensor:
    if not self.has_previous_state:
        # Prefill: install the conv-warm window from the tail of the prompt
        self.conv_states[layer_idx] = new_conv_state.to(self.conv_states.device)
    else:
        # Decode: shift left by 1 (drop oldest), write current step at [-1]
        self.conv_states[layer_idx] = self.conv_states[layer_idx].roll(shifts=-1, dims=-1)
        self.conv_states[layer_idx][:, :, -1] = new_conv_state[:, 0, :].to(self.conv_states.device)
    return self.conv_states[layer_idx]
```

This is a textbook circular-buffer update for a causal 1D conv with persistent state. Identical in pattern to Mamba's conv state caching, just smaller (kernel=2 → 1 cached step instead of kernel=4 → 3).

### CCA forward reads + writes both buffers in the decode branch

`modeling_zaya.py:359-368` (decode = `has_previous_state` branch):

```python
if past_key_values.has_previous_state:
    # Generation
    qk_packed0 = qk_packed0.transpose(0, 1)                                  # [B, 1, H]
    qk_packed0_cached = past_key_values.conv_states[self.layer_number]       # [B, H, 2]
    qk_packed0_cat = torch.cat([qk_packed0_cached, qk_packed0.transpose(1, 2)], dim=-1)  # [B, H, 3]
    qk_packed3 = self.conv_qk(qk_packed0_cat).permute(2, 0, 1)               # [S, B, E]
    qk_packed0_cache = past_key_values.update_conv_state(
        layer_idx=self.layer_number, new_conv_state=qk_packed0
    )                                                                        # rolls cache
```

`modeling_zaya.py:409-415` (the v2 / `prev_hs` half):

```python
if past_key_values is not None:
    if past_key_values.has_previous_state:
        # Generation
        hs_d = past_key_values.prev_hs[self.layer_number].clone()  # [B, H]
        hs_d = hs_d.unsqueeze(0)                                   # [1, B, H]
    past_key_values.prev_hs[self.layer_number].copy_(hs[-1, :, :])  # writeback (always at decode)
```

Two distinct recurrences in one operator:

1. **Conv state** - last `conv_kernel_size - 1 = 1` step of post-projection QK (concatenated 1280-dim), used to compose the conv1d input for the next step.
2. **`prev_hs`** - last step's input hidden state, used as the "delayed" stream for `val_proj2(hs_d)` (the v2 half of the value projection).

### Convs are along the SEQUENCE axis (not channel)

`modeling_zaya.py:289-308` - the conv stack:

```python
in_out_ch = self.latent_k_dim + self.latent_q_dim   # 1280
self.conv_qk = nn.Sequential(
    nn.Conv1d(in_channels=in_out_ch, out_channels=in_out_ch,
              kernel_size=self.cca_time0, groups=in_out_ch, padding=0, stride=1),     # depthwise, k=2
    nn.Conv1d(in_channels=in_out_ch, out_channels=in_out_ch,
              kernel_size=self.cca_time1, groups=(num_kv_heads + num_q_heads),         # grouped (10 groups), k=2
              padding=0, stride=1),
)
```

The prefill path makes this explicit (`modeling_zaya.py:385-394`):

```python
qk_packed1 = qk_packed0.permute(1, 2, 0)                  # [S, B, E] -> [B, E, S]   (channels=E, length=S)
qk_packed2 = F.pad(qk_packed1, (self.total_padding, 0))   # left-pad ONLY (causal)
qk_packed3 = self.conv_qk(qk_packed2).permute(2, 0, 1)    # back to [S, B, E]
```

`F.pad(..., (total_padding, 0))` left-pads on the LAST dim (sequence). Two stacked k=2 convs → effective kernel = 3 over time. So each output position is a function of the current and the previous 2 positions of QK projections.

### Cross-reference: why earlier grep missed it

```
grep -nE "mamba|ssm|recurrent|scan|conv1d|state\.copy_|state_in|past_state|\.cache_params"
   modeling_zaya.py modular_zaya.py configuration_zaya.py
   → ZERO matches
```

The keywords for THIS recurrence are different. The right query was:

```
grep -nE "Conv1d|conv_states|prev_hs|has_previous_state|update_conv_state"
   → modeling_zaya.py: 4 + N hits (CCA constructor, cache class, forward decode branch)
```

`mamba_cache_dtype` (singular field in config) returns zero hits across the module - confirmed dead. The Zaya cache class allocates fp16 by default and never consults that config field.

## Implications for hipfire-runtime

### What's structurally new

1. **First per-layer recurrent state in hipfire.** Existing infra (Qwen 3.5 DeltaNet linear-attention path is the closest analogue, but DeltaNet's recurrence is folded into the LA forward signature, not externalized as a stateful cache the way CCA does).
2. **State allocation:** 80 layers × (1280 × 2 + 2048) = 80 × 4608 = ~370k elems per sequence at fp16 = ~740 KB per sequence. Trivial vs KV cache. But it needs a home.
3. **State carry:** every decode step reads `conv_states[layer]` and `prev_hs[layer]`, computes new outputs, writes both back. The `roll(-1) + write [-1]` semantic for `conv_states` is a circular buffer; on RDNA we'd implement as a small persistent HBM tensor or a single-line LDS-resident buffer per layer.
4. **State reset** on session end / context reset (`reset()` zeroes both buffers).

### What this breaks for hipfire

1. **Speculative decoding (DFlash, MTP):** drafter and target both run CCA forward, both must have a coherent recurrent state. Parallel speculation across N candidate tokens means N parallel possible state advances; on rollback, state must be restored. This is the same problem Mamba speculative decoding solves; non-trivial.
2. **KV pager:** the existing pager only knows about KV. If a sequence is paged out, conv_states + prev_hs must travel with it.
3. **Multi-GPU sharding:** if layers are pipeline-sharded across devices (PP), each device owns its layers' recurrent state - already the natural shard axis. If layers are tensor-sharded (TP within a layer), conv_states is sharded along its channel dim (1280) which decomposes cleanly per-head.
4. **Multi-batch / continuous batching:** state is per-sequence per-layer, must be allocated and reset alongside KV pages.

### What's contained in the arch crate

- The CCA forward op itself (the conv kernels, the L2-norm-and-scale, the v1/v2 mixing).
- Math is straightforward; the "novelty" is structural (state plumbing), not arithmetic. The conv is k=2 depthwise + k=2 grouped over a 1280-channel × short-time window - small enough to run as a single fused HIP kernel per decode step.

### What requires hipfire-runtime changes (per contract: REQUIRES-KADEN-DECISION)

- A new state container in `hipfire-runtime` parallel to KV. Either:
  - **Option A** - extend the existing `State` per-arch type (already on the `Architecture` trait at `crates/hipfire-runtime/src/arch.rs:44`) to carry recurrent buffers as opaque per-arch state. Per-arch crate owns shape/dtype; runtime only allocates and reset-zeros via trait methods. **Lower runtime churn.**
  - **Option B** - first-class "recurrent cache" abstraction in `hipfire-runtime` parallel to KV cache, with paging/sharding/spec-decode hooks. **Higher payoff, more design surface.**
- Spec-decode policy: either **gate ZAYA1 to AR-only** (simplest, ships) or design state-fork/merge primitives.
- Wire `state.reset()` into the session-end / chat-template-reset path.

### Sequencing recommendation for the rest of tonight

Per the contract's RECURRENT branch:

- **Phase 1** (foundations) - proceed: arch crate scaffolding, intake harness, PyTorch reference dumps. The recurrent state shapes are now known and can be modeled in `Self::State`.
- **Phase 2** (free components) - proceed: config parse, RMSNorm/SwiGLU/GQA reuse, partial-RoPE 0.5, scale_residual_merge, MLP router, top-1 routing. None of these touch CCA.
- **Phase 3 (CCA kernel)** - **DO NOT start tonight.** Per contract.
- **Phase 4 (MoD design doc)** - proceed.
- **Phase 5 (EDA identification)** - proceed.
- **Phase 6 (recurrent-state design doc)** - **headline deliverable.** Author with Option A vs Option B comparison and a recommended path. Flag REQUIRES-KADEN-DECISION.

### Side-finding to flag for Phase 1

CCA's `cca_num_q_heads = 8` is **distinct from** the standard attention's `num_attention_heads = 16`. CCA's forward returns `query: [B, S, 8*128] = [B, S, 1024]`, `key: [B, S, 2*128] = [B, S, 256]`, `value: [B, S, 2*128] = [B, S, 256]`. The downstream `ZayaAttention` (line 483) then does 16-head attention. How does the 8-head CCA Q project up to 16-head attention input? Either repeat-interleave (treating 8 CCA heads as 8 "groups" doubled) or a learned projection inside `ZayaAttention`. Read in Phase 1 before writing the arch crate's forward signature.

## Confidence

**High.** The cache class explicitly says "CCA cache," the forward explicitly reads + writes per-layer state in the generation branch, and the update semantics match a textbook circular conv buffer. There is no plausible reading where CCA is stateless.
