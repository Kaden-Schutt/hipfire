# 03 - EDA Identification

**Date:** 2026-05-07
**Branch:** `feat/zaya1-port-intake`
**Source:** `Zyphra/transformers@zaya1` `modeling_zaya.py:917-1048`, `modular_zaya.py:1054-1206`

## What it is

EDA = "depth-wise averaging" inside `ZayaRouter`. Per its own docstring
(modeling_zaya.py:921): "Optional EDA (depth-wise averaging) via
`router_states_scale` and prior `router_states`."

It is a **cross-layer pipe**, not a cross-decode-step recurrence: layer
N+1's router takes layer N's pre-norm router state and adds it
(elementwise scaled) to layer N+1's own pre-norm router state before
the router's RMSNorm + MLP + softmax. Within one forward pass, the
state flows up the layer stack like an additional residual stream
that lives in the router subspace (`mlp_expansion=256`-dim, not
`hidden_size=2048`-dim).

## Wiring

`ZayaBlock.forward` (modeling_zaya.py:1229-1280):

```python
def forward(self, hidden_states, prev_router_hidden_states=None, ...):
    route_prob, expert_choice, prev_router_hidden_states = self.router(
        hidden_states, router_states=prev_router_hidden_states
    )
    ...
    return expert_output, mlp_bias, prev_router_hidden_states
```

The block takes `prev_router_hidden_states` from the previous layer
and returns the (post-down_proj, pre-norm) router state for the next
layer to consume. ZayaModel (line 1490) threads this through.

`ZayaRouter.forward` (modeling_zaya.py:992-1048):

```python
hs = self.down_proj(hidden_states)              # (B, S, mlp_expansion=256)
if self.use_eda and (router_states is not None):
    hs = hs + router_states * self.router_states_scale   # EDA add
router_hidden_states_next = hs[:, -S:].clone()  # what next layer receives
hs_norm = self.rmsnorm_eda(hs)
logits = self.router_mlp(hs_norm)
expert_prob = torch.softmax(logits, dim=-1)
```

## Per-layer cost

Adds one new parameter per ZayaBlock that has EDA enabled:

- `router_states_scale`: `[mlp_expansion=256]` fp16 = 512 bytes per layer
- For ZAYA1-8B (80 layers, EDA off on layer 1, on for the other 79):
  79 * 512 = ~40 KB total. Negligible.

Per-step compute: one elementwise scale-and-add over `[B, S, 256]`,
then proceeds with the existing router math. Effectively free.

## Gating

`self.use_eda = use_eda_cfg AND (zaya_first_layer is not None) AND
(self.layer_number != zaya_first_layer)` where
`zaya_first_layer = 1` (hardcoded line 965). Translation: enabled on
all layers EXCEPT layer index 1. (Layer 0 is also enabled, it just
receives `router_states=None` since there is no prior layer, and the
add is skipped.)

This asymmetry is awkward and undocumented. The hipfire impl mirrors
it bit-for-bit; deviating risks NRMSE drift at every layer >= 2.

## Implications for hipfire

EDA is the cheapest item on the porting list. It does NOT require any
new infrastructure:

1. **No recurrent state.** The cross-layer threading is entirely
   within one forward pass, same regime as the standard residual
   stream. One live `[B, S, 256]` tensor at any moment.
2. **No per-step carry.** Decode steps reset `router_states` to
   `None` at the layer-0 entry (next call to `prefill` / `decode_step`
   re-enters with `prev_router_hidden_states=None`).
3. **Trivial kernel.** The scale-and-add is identical in shape to a
   residual-add and can borrow the existing fused-residual kernel
   with one extra elementwise multiply.

## Implementation notes

- Add `router_states_scale: WeightTensor` per layer to ZayaWeights.
- ZayaState carries one live `router_states: Option<GpuTensor>` slot
  reused across layers.
- Layer-0 forward writes the slot (no read), layer-1 reads-but-skips-add
  (per the hardcoded `layer_number != 1` gate), layers 2..N read+add.
- Reset slot to None at every prefill / decode-step entry; do not
  carry across decode steps.

## RDNA mapping

`mlp_expansion=256` is wave32-friendly (8 lanes per row group on
gfx1201). The scale-and-add is BW-bound; reuse existing residual-add
kernel with a third input pointer.

## No surprises

EDA is an undocumented marketing term for a published technique. There
is no novel math here, no recurrent state, no incompatibility with
spec-decode or paging or sharding. It can be deferred or implemented
in any order; it does not gate any other component.
