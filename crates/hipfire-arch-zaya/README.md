# hipfire-arch-zaya

Zyphra ZAYA1 architecture for hipfire. **Phase 1 intake scaffold (2026-05-07).**

Status: every method on the `Architecture` trait either returns a typed default
(arch_id=7, name="zaya", overrides) or returns `Err` with a pointer to the
intake docs. Forward pass is not implemented. No HFQ representation exists yet.

See `docs/investigations/2026-05-07-zaya1-port-intake/` for the port plan,
Phase 0 disambiguation verdict (CCA is RECURRENT), and the design docs that
gate Phase 3+ work.

## Architectural elements

(From `Zyphra/transformers@zaya1` reading; cross-referenced against
`Zyphra/ZAYA1-8B/config.json`.)

### Free (map onto existing hipfire infrastructure)

- **RMSNorm**, **SwiGLU**, **GQA** (16Q : 2KV, head_dim 128) — Qwen 3.5 path.
- **Gemma tokenizer** (vocab 262272, eos 106 = `<end_of_turn>`) — Gemma 4 port.
- **Top-1 MoE routing** — degenerate case of qwen35's top-k.
- **Residual fp32 accumulator** — existing rmsnorm pattern.

### Small kernel additions (Phase 2)

- **partial_rotary_factor=0.5** — rotates first 64 of 128 head dims; either
  parameterize the existing RoPE kernel or add a half-RoPE entry point.
- **scale_residual_merge** — load per-block learnable scalar, multiply
  during residual add.
- **MLP-based MoE router** (2-layer, `zaya_mlp_expansion=256`) — replace
  the existing linear router with a 2-layer MLP for routing logits.
- **Top-1 routing path** — verify existing top-k handles k=1 cleanly,
  skip combine.

### Structurally new (Phase 6 design + escalation)

- **CCA (Compressed Convolutional Attention)** — Zyphra-novel. Two
  per-layer per-sequence recurrent buffers carry across decode steps:
  - `conv_states[layer]` — `[B, 1280, 2]` fp16, circular buffer for the
    causal Conv1d-along-time stack (depthwise k=2 + grouped k=2).
  - `prev_hs[layer]` — `[B, 2048]` fp16, 1-step lag of input hidden
    state for the v2 value-projection stream.
  - Per ZAYA1-8B: ~720 KB recurrent state per sequence across all
    layers. Trivial vs KV cache, but new infrastructure.
  - Decode update: `roll(-1) + write[-1]` for `conv_states`,
    `prev_hs[layer].copy_(hs[-1, :, :])` for `prev_hs`.

- **Mixture-of-Depths (MoD)** — per-token layer-skip routing.
  Conditionalizes KV writes; breaks DFlash/spec-decode assumptions.
  Phase 4 design doc.

- **EDA component** — `zaya_use_eda=true` is undocumented in the model
  card and blog. Phase 5 source-read identifies it.

## Files

| File | Purpose |
|---|---|
| `Cargo.toml` | Workspace member; deps mirror toy + qwen35. |
| `src/lib.rs` | Module declarations + re-exports. |
| `src/arch.rs` | `Architecture for Zaya` trait impl. arch_id=7, eos_filter overrides for Gemma turn markers. |
| `src/config.rs` | `ZayaConfig` — every field from `Zyphra/ZAYA1-8B/config.json` with port-status notes. |
| `src/state.rs` | `ZayaState` — typed slots for KV + CCA recurrent buffers; `cca_state_bytes_per_seq` helper. |
| `src/weights.rs` | `ZayaWeights` — placeholder; per-layer weight inventory in doc-block. |
| `src/forward.rs` | `prefill` / `decode_step` stubs that return Err. |
| `examples/verify_against_torch.rs` | Skeleton for the per-layer NRMSE harness. |

## Phase plan

1. **Phase 0 (done):** CCA disambiguation. VERDICT: RECURRENT.
2. **Phase 1 (in progress):** Crate scaffold + harness scaffold + reference dumps.
3. **Phase 2:** Free components, each per-layer NRMSE-validated.
4. **Phase 3:** Deferred. CCA kernel waits for Phase 6.
5. **Phase 4:** MoD design doc; no autonomous integration.
6. **Phase 5:** EDA identification.
7. **Phase 6:** Recurrent-state design doc. REQUIRES-KADEN-DECISION.

## RDNA target

R9700 (gfx1201, 64 CUs, 32 GB VRAM). The CCA conv kernel is small and
amenable to packed-fp16 (`v_pk_fma_f16`) on wave32; `in_out_ch=1280` is
divisible by 64, kernel size 2 is a perfect fit for the 2-element
packed FMA. The recurrent state's `roll(-1) + write[-1]` becomes a
single uint32 swap if conv_states are laid out as consecutive fp16
pairs in HBM. These are sketches, not commitments; the Phase 6 design
doc owns the kernel-shape decision.
