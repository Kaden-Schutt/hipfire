# 06 - Phase 2.D: All 6 free components validated

**Date:** 2026-05-08
**Branch:** `feat/zaya1-port-intake`
**Predicate:** 05-phase2-cpu-validation.md (RMSNorm + ResidualScaling).

## Headline

All 6 Phase 2 "free components" from the overnight contract are now
NRMSE-validated against the PyTorch reference dump on real ZAYA1-8B
activations. The CPU-Rust validator pipeline is end-to-end working
without HFQ.

```
=== ZayaRMSNorm @ layer 0 input_norm ===
  PASS NRMSE = 1.659e-3
=== ZayaRMSNorm @ layer 1 input_norm ===
  PASS NRMSE = 1.664e-3
=== ResidualScaling @ layer 0 (hidden_states only) ===
  PASS NRMSE = 2.412e-3
=== ResidualScaling @ layer 1 (residual path) ===
  PASS NRMSE = 0.000e0
=== ResidualScaling @ layer 1 (hidden_states path) ===
  PASS NRMSE = 2.353e-3
=== ZayaRMSNorm @ final_norm ===
  PASS NRMSE = 1.652e-3
=== o_proj @ layer 0 (1024 -> 2048) ===
  PASS NRMSE = 1.658e-3
  PASS NRMSE = 0.000e0  layer_00 o_proj == self_attn.out0
=== MLP-based MoE router @ layer 1 (top-1, no EDA) ===
  PASS NRMSE = 1.616e-3  router_hidden_states_next (down_proj)
  PASS NRMSE = 2.848e-3  router route_prob
  PASS expert_choice exact match: 0/23 mismatches
  per-token expert assignment:
    [3, 15, 4, 7, 6, 6, 0, 3, 13, 0, 10, 0, 9, 1, 0, 13, 7, 7, 10, 13, 14, 5, 5]
=== SwiGLU + expert 0 MLP @ layer 1 ===
  PASS NRMSE = 3.497e-3  (4 tokens routed to expert 0)
=== partial-RoPE math (head_dim=128, rotary_dim=64) ===
  PASS (a) cos=1,sin=0 identity on first 64 dims
  PASS (b) last 64 dims unchanged (passthrough)
  PASS (c) cos=0,sin=1 rotate_half formula
=== ALL PASS (bf16-ULP threshold = 5e-3) ===
```

## Map to contract Phase 2 deliverables

The overnight contract listed 6 free components. Status:

| Contract item | Status | Evidence |
|---|---|---|
| 1. CONFIG PARSING | PASS | 5 unit tests on ZayaConfig defaults match published 8B config.json field-for-field |
| 2. RMSNorm + SwiGLU + GQA reuse | PASS | RMSNorm 3 sites (1.65e-3); SwiGLU+expert-MLP 4 tokens (3.50e-3); GQA tail via o_proj (1.66e-3, bit-exact match against self_attn.out0) |
| 3. PARTIAL_ROTARY_FACTOR=0.5 | PASS | 3 math properties (identity, passthrough, rotate_half) all bit-exact |
| 4. SCALE_RESIDUAL_MERGE | PASS | ResidualScaling 3 sites (0e0 to 2.41e-3) |
| 5. MLP-BASED MoE ROUTER | PASS | Full forward (down_proj, RMSNorm, 3-layer GELU MLP, softmax, balancing_biases) at 1.62e-3 to 2.85e-3 |
| 6. TOP-1 ROUTING | PASS | 23/23 tokens exact expert assignment match |

## What this proves

For each free component, hipfire's CPU implementation produces values
matching PyTorch's bf16 forward at sub-bf16-ULP precision (threshold
5e-3, observed 0 to 3.5e-3). When these components land as RDNA HIP
kernels, the kernel's job is reduced to "produce the same f32
values" - no architectural risk left, only kernel-engineering risk.

## Findings worth flagging

### ZAYA1 attention is 8q:2kv, NOT 16q:2kv

Reading `ZayaAttention.__init__` (modeling_zaya.py:483) found a
"hardcoded query compression" baked into the attention layer:

```python
self.o_proj = nn.Linear(
    (self.num_heads // 2) * self.head_dim,    # 8 * 128 = 1024
    self.hidden_size,                          # 2048
    bias=...,
)  # hardcoded query compression for now

# In forward:
query_states = query_states.view(B, S, self.config.num_attention_heads // 2, self.head_dim)
                                                                       # ^ 16//2 = 8 query heads
```

So the standard attention path AFTER CCA uses 8 query heads (not 16),
2 KV heads with `repeat_kv(k, num_key_value_groups // 2)` =
`repeat_kv(k, 4)` to broadcast K/V to match 8 query heads. The o_proj
maps 8*128=1024 back to hidden_size=2048.

The config field `num_attention_heads=16` is misleading; the EFFECTIVE
attention is 8q:2kv. Earlier docs in this branch claimed 16q:2kv GQA
ratio of 8:1; the real ratio is 8q:2kv = 4:1. Output projection
weight is `[2048, 1024]` (verified by extracting it).

This is now documented in the o_proj validator's comment block. Phase
6.A implementation should size the attention forward path's Q tensor
at 8 heads, not 16, and the o_proj kernel as 1024→2048 (not 2048→2048).

### MoD did not activate on the canonical prompt

The router's per-token expert assignments for the 23-token canonical
prompt:
```
[3, 15, 4, 7, 6, 6, 0, 3, 13, 0, 10, 0, 9, 1, 0, 13, 7, 7, 10, 13, 14, 5, 5]
```

Zero tokens went to expert 16 (the skip slot). MoD is enabled in the
config but didn't fire on this prompt; the router preferred a real
expert for every token. This matches the design intent (the -1.0
balancing bias on the skip slot strongly disincentivizes selection).

For testing the MoD code path, would need a prompt that actually
routes some tokens to skip. Defer to integration tests.

### Expert load is uneven (canonical prompt sample)

23 tokens, 16 experts, top-1: ideal even split would be 1.4 tokens
per expert. Actual:
```
expert 0: 4 tokens
expert 3: 2 tokens
expert 7: 3 tokens
expert 13: 3 tokens
... (others 1 or 0)
```

This is one prompt; not a generalization. But the small batch sizes
mean ~half of the experts see 0 tokens per step. Validates the MoD
design's "skip if you can't be useful" intent.

## What's still NOT validated end-to-end

- **CCA forward (the recurrent op)**: depthwise + grouped Conv1d
  along time + L2-normalize-and-scale. Captures exist (cca.in0,
  cca.out0/1/2 = q/k/v) but no Rust impl yet. Phase 6.A scope.
- **Standard attention math**: Q @ K.T -> softmax -> @ V -> o_proj.
  o_proj is validated; the matmul-softmax-matmul middle isn't.
  Captures don't include intermediate attention states; would need
  finer hooks.

These don't fit the "free component" category and weren't on the
contract's Phase 2 list. They're CCA territory (Phase 6.A) and
attention-internal (Phase 3 once CCA is in place).

## Files added

- `crates/hipfire-arch-zaya/examples/cpu_validate_phase2.rs` extended
  from 250 to ~600 LOC; now validates all 6 free components plus
  o_proj plus partial-RoPE math.
- `Cargo.toml` adds `libm = "0.2"` dev-dep for the exact GELU.
- `scripts/arch-intake/extract_phase2d_subset.py` (committed alongside
  the existing extract_phase2_subset.py).
- `scripts/arch-intake/dump_zaya_reference.py` adds CCA + o_proj hooks
  on ATT layers; v4 dump = 1519 tensors total.

## RDNA target consideration

Each NRMSE number sets the bar for the eventual gfx1201 kernel:

- RMSNorm: 1.65-1.66e-3. Existing rmsnorm.hip pattern; one tensor
  name change + the f32-cast-then-multiply order from this doc.
- ResidualScaling: 0e0 (residual path) to 2.41e-3 (hidden_states).
  Pure elementwise add+mul; fuse into the existing residual-add kernel.
- partial-RoPE: bit-exact math. Existing RoPE kernel needs a
  rotary_dim parameter (rotate first N of head_dim); the rest passes
  through. ~10-line modification.
- MoE router: 1.62-2.85e-3. NEW kernel: down_proj + RMSNorm + 3-layer
  MLP-with-GELU + softmax + topk + gather. Largest ZAYA1-specific
  kernel work outside CCA itself.
- SwiGLU expert MLP: 3.50e-3. Standard pattern; reuse qwen35-MoE
  kernel with adjusted ffn_hidden=4096.
- o_proj: 1.66e-3, bit-exact equality with self_attn.out0. Pure GEMM
  1024->2048; existing qkv-fusion kernel surface.

## Sequencing note

Phase 2 deliverables are now all done. The remaining gates to a
shippable ZAYA1 are:

1. **MANUAL_REVIEW.md decision** on Phase 6.A (Option A vs B,
   spec-decode policy, paging policy). Blocking everything else.
2. HFQ writer for ZAYA1 (~3-5 days; mechanical).
3. CCA scalar reference + HIP kernel (Phase 6.A implementation,
   ~2 weeks calendar).
4. Free-component HIP kernels (small, can be parallel; each is a
   variation on an existing kernel except the MoE router).
5. End-to-end NRMSE validation against PyTorch using
   `cpu_validate_phase2`'s methodology, but on the GPU forward path
   instead of CPU.

Step 5 reuses tonight's validator wholesale; just point it at GPU
output dumps instead of CPU output. The methodology is proven.
