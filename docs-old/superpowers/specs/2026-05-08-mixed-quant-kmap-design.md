# Mixed Precision K-Map Quantization

**Issue:** [#196](https://github.com/Kaden-Schutt/hipfire/issues/196)
**Branch:** `feat/mixed-quant-kmap`
**Date:** 2026-05-08

## Problem

Uniform 4-bit quantization (MQ4/HFQ4) causes structural attractors on MoE models
and suboptimal quality on dense models. MoE expert FFN weights accumulate
non-uniform quantization errors across expert paths, drifting the hidden state
off-manifold after ~50-200 tokens. Dense models lose quality unnecessarily in
edge layers (first/last) that disproportionately influence output.

llama.cpp's Q4_K_M avoids this by silently promoting sensitive tensors to Q6_K
or Q8_0 based on tensor role, layer position, and MoE status. hipfire currently
requires manual re-quantization to HFQ6 for the entire model.

## Design

### Data Model

```rust
#[derive(Clone, Copy, Debug, PartialEq)]
enum QuantLevel {
    F16,       // norms, biases, 1D tensors
    Q8,        // embeddings, lm_head, routers
    Promote6,  // bump to 6-bit variant of base format
    Base,      // default for the chosen --format
}
```

### Classification Layering

The quantizer has two classification stages. Understanding the layering is
critical for correct implementation:

1. **`should_quantize(name)` (primary gate)** — returns `false` for norms,
   biases, and vision encoder tensors. These are stored as F16 and never
   reach the K-map. This gate is NOT replaced by K-map.

2. **`kmap_resolve(name, ...)` (format selection)** — determines the quant
   level for tensors that pass the primary gate. Only governs tensors that
   `should_quantize` returns `true` for.

Similarly, `is_q8_tensor` remains independent — it serves `--format q8-mixed`
and `--format q8-fast` modes. K-map does NOT subsume it. When K-map is active,
`is_q8_tensor` is not consulted (K-map's own rules apply). When `--no-kmap`
is passed, the existing `is_q8_tensor` paths work as before.

### K-Map Resolution

```rust
fn kmap_resolve(
    name: &str,
    n_layers: usize,
    is_moe: bool,
) -> QuantLevel
```

Evaluated in order (first match wins). Note: rules 1-2 overlap with the
`should_quantize` gate — they exist for completeness (GGUF path) and as
documentation. In the safetensors path, tensors matching rules 1-2 never
reach `kmap_resolve`.

1. Norm / bias / 1D → `F16`
2. Embedding / lm_head / output.weight → `Q8`
3. MoE router (`mlp.gate.weight`, `shared_expert_gate`) → `Q8`
4. MoE expert FFN (`mlp.experts.*.{gate_up_proj,down_proj}`) → `Promote6`
5. Layer index in first 2 or last 2 (both attn and FFN) → `Promote6`
6. Everything else → `Base`

**Layer index extraction:** parse from tensor name using unanchored search
for `layers.{N}.` (safetensors) or `blk.{N}.` (GGUF). Must handle both
`model.layers.{N}.` (dense Qwen) and `model.language_model.layers.{N}.`
(MoE Qwen) — do NOT anchor to a specific prefix.

**Edge layer count:** fixed at 2 (first 2 + last 2). On small models
(e.g. 0.8B with 24 layers) this promotes 4/24 = 17% of layers. This is
intentional: small models are more sensitive to edge-layer quantization
error, and the VRAM cost is negligible (~50 MB on 0.8B).

`n_layers` is available from model config in both paths. If unavailable
(missing config key), disable edge-layer promotion (rules 4 still applies
for MoE experts).

### Pre-Pass HashMap

Before quantization, iterate all tensor names, call `kmap_resolve` for each,
store in `HashMap<String, QuantLevel>`. Print a summary:

```
K-map plan (mq4 base, 64 layers, MoE):
  F16:       130 tensors (norms, biases)
  Q8:         5 tensors (embed, lm_head, routers)
  Promote6:  520 tensors (expert FFN, edge layers 0-1/62-63)
  Base:      360 tensors (remaining)
```

This HashMap is the extension point for step 2 of #196 (cosim adaptive
override — merge per-tensor cosim results into the same map).

### Format Promotion Mapping

Each base format has a defined Promote6 target:

| Base format | Promote6 target | Notes |
|---|---|---|
| MQ4G256 | MQ6G256 | FWHT rotation preserved |
| MQ3G256 | MQ6G256 | No MQ5 exists |
| MQ2G256 | MQ6G256 | Same |
| HFQ4G256 | HFQ6G256 | No rotation |
| HFQ3G128 | HFQ6G256 | Upgrade group size |
| HFQ2G128 | HFQ6G256 | Same |
| MQ6G256 | MQ6G256 | No-op (already 6-bit) |
| HFQ6G256 | HFQ6G256 | No-op |
| MQ8G256 | MQ8G256 | No-op (already higher) |
| Lloyd variants | MQ6G256 | Loses Lloyd codebook (fine — MQ6 doesn't need it) |

### Integration

#### Safetensors path (~line 2460)

1. Build K-map HashMap before the tensor loop
2. Look up `kmap[&name]` per tensor
3. If `Promote6`: call the 6-bit quantize function for the base format
4. If `Q8`: call `quantize_q8f16`
5. If `F16`: emit f16 bytes
6. If `Base`: existing behavior unchanged

#### GGUF path (~line 1920)

Same pattern. K-map lookup happens after `gguf_to_safetensors_name`
translation so the same tensor-name patterns work. For tensors where
translation returns `None` (untranslated, raw GGUF name kept), the
K-map also checks GGUF-style patterns (`blk.{N}.`, `_norm`, `token_embd`).

**GGUF + MoE is out of scope.** The GGUF path has no MoE expert tensor
splitting. K-map rule 4 (expert FFN → Promote6) only fires in the
safetensors path. MoE models are quantized from safetensors source;
GGUF MoE input is not a supported workflow. If a GGUF MoE tensor name
happens to match rule 4's pattern after translation, it will be promoted,
but this is best-effort.

#### MoE expert split path (~line 2323)

The expert split loop iterates per-expert, producing child tensors like
`layers.{N}.mlp.experts.{X}.gate_up_proj.weight`. K-map entries are
keyed by these child names (not the parent 3D tensor name). Rule 4
matches on `mlp.experts.` substring → `Promote6`.

### UX

- **Default behavior changes:** `--format mq4` (and all other formats) now
  applies K-map automatically. This is the safe default.
- **Opt-out:** `--no-kmap` or `--uniform` disables the pre-pass, giving
  the current uniform behavior.
- **`--q8-router` stays independent:** `--no-kmap` does not disable Q8
  routers on MoE models. The router fix is a separate safety mechanism.
  When K-map IS active, rule 3 (router → Q8) and the existing `q8_router`
  logic are redundant but idempotent — no conflict.
- **`--format mq4-mq6exp` deprecated:** prints a warning ("deprecated: use
  --format mq4, K-map promotes experts automatically") and behaves
  identically to `--format mq4` with K-map. K-map is a superset
  (experts + edge layers). Remove in a future release.

### Edge Cases

1. **Non-256-aligned k_dim:** Promoted tensors with `k_dim % 256 != 0` fall
   back to Q8F16 (existing fallback logic handles this).

2. **Sub-4-bit base formats:** K-map still applies. Edge layers and expert
   FFNs bump to MQ6. The larger gap (2→6) is correct — sub-4-bit formats
   need the safety net even more.

3. **6-bit or 8-bit base:** K-map is a no-op for Promote6 (already at or
   above 6-bit). F16/Q8 rules still apply for norms/embeds.

4. **VRAM cost:** For 27B dense (64 layers): ~28 tensors promoted, ~0.5 GB
   extra. For 122B MoE: expert promotion adds ~8 GB (matches manual HFQ6
   re-quant that users already do).

5. **Vision encoder tensors:** excluded by `should_quantize` (returns false
   for `model.visual.*`). Never reach K-map. If `--include-vision` is used,
   vision tensors fall to rule 6 (Base) — K-map does not promote vision
   weights.

6. **DeltaNet `linear_attn` tensors:** fall to rule 6 (Base). K-map does not
   give them special treatment. They are handled by `is_q8_tensor` only in
   q8-mixed/q8-fast modes.

## Validation Matrix

Empirical comparison across formats to measure the actual quality/size/speed
tradeoff. All runs use byte-identical committed prompts with recorded md5.

### KPIs

1. **Perplexity vs FP16 baseline** — cross-entropy loss on a fixed calibration
   set, measured against FP16 (BF16) reference logits. Primary quality metric.
2. **Coherence** — coherence gate pass/fail, unique token ratio, 3gram density
3. **Token agreement** — argmax agreement rate against FP16 baseline on fixed
   prompt set (% of positions where quantized model picks the same top token)
4. **Model size** — bytes on disk, peak VRAM at runtime
5. **Throughput** — tok/s decode, tok/s prefill (MQ6 GEMV is slightly slower
   than MQ4; quantify the cost of promotion)

### Test matrix

| Model | MQ4 uniform | MQ4+kmap | MQ6 uniform | HFQ6 uniform | FP16 baseline |
|---|---|---|---|---|---|
| 0.8B dense | measure | measure | measure | — | reference |
| 27B dense | measure | measure | measure | — | reference |
| 3.5-A3B MoE | measure | measure | — | measure | reference |
| 3.6-35B-A3B | measure | measure | — | measure | reference |

Each cell records: perplexity, token agreement %, coherence verdict, model GB,
decode tok/s, prefill tok/s.

**Expected outcome:** MQ4+kmap sits between uniform MQ4 and full MQ6/HFQ6 on
perplexity — close to 6-bit quality at close to 4-bit size. MoE models
specifically should show coherence parity with HFQ6 (no attractors).

## Out of Scope

- **Adaptive cosim override** (step 2 of #196): designed for but not
  implemented here. The HashMap is the extension point.
- **Per-layer importance profiling**: would require calibration data. Not
  needed — the static table covers the known failure modes.
- **New quant types**: no new QuantType variants. Reuses existing MQ6/HFQ6.
- **GGUF + MoE K-map**: GGUF path lacks MoE expert splitting. MoE models
  use safetensors input.

## Success Criteria

- [ ] `--format mq4` on MoE models produces clean output without manual HFQ6 re-quant
- [ ] `--format mq4` on dense models promotes first/last 2 layers to MQ6
- [ ] K-map summary printed during quantization
- [ ] `--no-kmap` preserves current uniform behavior exactly
- [ ] Both safetensors and GGUF paths use the same `kmap_resolve` function
- [ ] `--format mq4-mq6exp` prints deprecation warning, redirects to mq4+kmap
- [ ] Coherence gate passes on 0.8B, 27B dense and A3B MoE models
- [ ] Validation matrix populated with perplexity, token agreement, throughput
