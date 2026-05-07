# 02 - Mixture-of-Depths (MoD) Design

**Date:** 2026-05-07
**Branch:** `feat/zaya1-port-intake`
**Source:** `Zyphra/transformers@zaya1` `modeling_zaya.py:1197-1280` (ZayaBlock),
`917-1048` (ZayaRouter)

## TL;DR

The contract treated MoD as a structural concern (per-token layer-skip,
breaks DFlash/spec-decode, conditionalized KV writes). After reading
the implementation: **Zyphra's MoD is much friendlier than the spec
suggested.** It is implemented as `num_experts + 1` (the +1 = skip slot)
plus a sort-and-skip-last-bin pattern in the MoE dispatcher. Skip
tokens get `y = x * route_prob_skip`, then continue through the
residual add. **MoD touches zero KV state and zero attention compute.**

This is the lightweight implementation option from the contract's
Phase 4 menu, not the heavy compaction option, and it is what Zyphra
shipped. Recommendation: implement straightforward; no spec-decode
incompatibility, no gen-loop changes, no KV conditionalization.

## How Zyphra implements MoD

### Router side (ZayaRouter)

```python
self.use_mod = bool(getattr(config, "zaya_use_mod", False))
self.num_experts = (num_moe_experts + 1) if self.use_mod else num_moe_experts  # +1 skip
...
self.balancing_biases[-1] = -1.0   # bias the router AWAY from skip
```

The router has one extra "expert" slot reserved for skip. With
`moe_router_topk=1` (ZAYA1-8B), each token picks exactly one expert.
That choice can be the skip slot (last index), which means the token
elects to bypass MoE compute for this layer.

The `-1.0` balancing bias on the skip slot prevents it from dominating;
without it the router would learn to route everything to skip (free
loss reduction since the residual stream survives).

### MoE side (ZayaBlock.forward)

```python
sorted_indices, sort_order = torch.sort(indices_flat)
tokens_per_expert = torch.bincount(sorted_indices, minlength=self.router.num_experts)
sorted_hidden_states = hidden_states_flat[sort_order]
original_order = torch.argsort(sort_order)

if self.config.zaya_use_mod:
    # Run experts on all tokens EXCEPT those in the last bin (skip slot)
    expert_output, mlp_bias = self.experts(
        sorted_hidden_states[:sum(tokens_per_expert[:-1])],
        tokens_per_expert[:-1],
    )
    # Skip-slot tokens passthrough unchanged
    expert_output = torch.cat(
        [expert_output, sorted_hidden_states[sum(tokens_per_expert[:-1]):]],
        dim=0,
    )
    ...
else:
    expert_output, mlp_bias = self.experts(sorted_hidden_states, tokens_per_expert)

expert_output = expert_output[original_order]
expert_output = expert_output.view(B, S, E)
expert_output = expert_output * probs.unsqueeze(-1)   # scale by route_prob
```

The pattern:
1. Sort tokens by expert choice.
2. Run the expert MLP on the prefix (everything before the skip-slot
   bin). The skip-slot bin is the last bin because skip is the highest
   index.
3. Concatenate the un-processed skip-slot tokens unchanged.
4. Reorder back to original token order.
5. Scale by `route_prob` (for skip tokens this is the softmax weight
   on the skip slot, somewhere in [0, 1]).

The output is added to the residual stream upstream. Skip tokens
contribute `route_prob_skip * x` to the residual instead of
`route_prob_expert * MLP(x)`.

### What MoD does NOT touch

- **No KV conditionalization.** Attention runs first (in a separate
  ZayaDecoderATTLayer block; modeling_zaya.py:809). MoD lives in the
  MLP block; KV is already written by the time the router decides.
- **No layer skip in the strict sense.** Every token still flows
  through the entire block; only the expert MLP's compute is gated.
  Norm + residual still happen.
- **No gen-loop changes.** From the daemon's perspective, the forward
  function returns one logit per token regardless of MoD decisions.
- **No spec-decode incompatibility.** Drafter and target both run the
  same forward; if they disagree about which experts skip, that's a
  draft/target divergence the existing acceptance loop already handles.

## Hipfire implementation plan

Lightweight; lives entirely inside the per-arch crate, no runtime
changes needed.

### Forward pass

1. Router produces `expert_choice` of shape `[B*S]` with values in
   `0..(num_experts + 1)` when `zaya_use_mod=true`.
2. Sort by `expert_choice` (already done in qwen35-MoE; reuse the same
   permutation kernel).
3. Compute `tokens_per_expert[num_experts + 1]` via bincount.
4. Dispatch experts to indices `0..num_experts - 1` only (skip the
   last bin). Existing qwen35 MoE dispatch takes a `tokens_per_expert`
   slice, so this is a single `&tpe[..num_experts]` call vs
   `&tpe[..num_experts + 1]`.
5. Skip-bin tokens stay un-processed; a single device-side memcpy
   (or a slice-clone) materializes them into the output buffer at
   the right offsets.
6. Re-permute back to original token order.
7. Scale by `route_prob` (existing kernel; route_prob_skip values are
   normal softmax outputs and need no special handling).

### Kernel work

- **Bincount with extra bin**: existing kernel takes `num_experts`
  parameter; pass `num_experts + 1` when MoD is on. Trivial.
- **Skip-bin memcpy**: a device-side copy from
  `sorted_hidden[skip_start..]` to `expert_output[skip_start..]`. One
  HIP `hipMemcpyDtoD` call or a fused copy in the permutation kernel.
- **Permute back**: existing inverse-permutation kernel works as-is
  (the skip tokens have valid permutation indices like any other).

### Tests

- Per-layer NRMSE: with MoD on, MoE block output must match the
  reference's MoD-on output. Per-token: tokens that the reference
  routed to skip should be unchanged-modulo-`route_prob_scale` in
  hipfire's output too.
- Bincount sanity: `sum(tokens_per_expert) == B*S` always.
- Skip-bias: load the model, dump `balancing_biases`, assert
  `balancing_biases[-1] == -1.0` and others are 0.

## Why the contract was right to escalate but not block

The contract said "Do NOT attempt to integrate MoD overnight" because
it expected:

> Touch points in hipfire's gen loop (specifically: where do KV writes
> get conditionalized?). Compatibility with DFlash / speculative
> decoding (likely incompatible without explicit handling; spec it out).

Both were valid concerns based on prior MoD literature (DeepMind's
original MoD paper has a per-token layer skip that DOES touch KV).
Zyphra opted for the friendlier impl that is structurally an MoE
expert with a free pass-through. This caps the impl to per-arch crate
work; no runtime changes needed.

That said, the design doc still recommends:

1. **Land MoD with feature flag** `HIPFIRE_ZAYA_MOD=1`, default off
   for the first ship, on once we confirm parity at decode in
   long-context. Coherence at decode is the only meaningful risk;
   the math is straightforward.
2. **Coherence-gate test** specifically for MoD: run a 1k-token
   prompt with MoD on vs MoD off (force `balancing_biases[-1] = -inf`
   to disable the skip slot), confirm both produce sane output. The
   skip-bias literature suggests a small (~5-15%) decode quality
   delta from MoD, but it should be COHERENT, not garbled.
3. **DFlash compatibility audit**: drafter and target must agree on
   `num_experts + 1` to share the bincount math. If the drafter is a
   different model (a small dense), it has no MoD; the existing
   spec-decode acceptance machinery doesn't care, since it accepts
   on token agreement, not on intermediate-state agreement. No work.

## Open questions for Kaden

None. The MoD design is contained in the arch crate; the contract's
"REQUIRES-KADEN-DECISION" flag is for the recurrent-state design
(Phase 6), not MoD.

## RDNA mapping

The skip-bin memcpy is a pure BW operation; on gfx1201 the existing
DtoD copy path is fine. No new kernels.

The per-token branchless skip means the wave32 SIMD pattern in the
MoE dispatcher needs no per-lane gating; the lanes that map to skip
tokens just don't reach the expert MLP. Wave occupancy is unchanged.
