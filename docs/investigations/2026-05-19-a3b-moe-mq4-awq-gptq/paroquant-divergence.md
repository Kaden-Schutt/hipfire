# ParoQuant A3B divergence investigation — Bug #3: lm_head tied to embed_tokens.weight

## TL;DR

**Root cause: hipfire's `load_weights_paroquant` unconditionally uses
`model.language_model.embed_tokens.weight` as the lm_head, but the
shisa-ai Qwen3.6-35B-A3B-PARO model has `tie_word_embeddings: false`
and ships a SEPARATE `lm_head.weight` tensor with different values.**
Every logit prediction is therefore computed against the wrong
unembedding matrix, which manifests at decode time as: (1) the model
echoes the chat-template structure back instead of answering, and
(2) it drifts into a Chinese-character token attractor (118401 =
`åĩºéĶĻ` / "出错" / "error").

## Evidence

### Config

`/home/bjoern/.hipfire/models/shisa-Qwen3.6-35B-A3B-PARO-unpacked/config.json`:

```
"tie_word_embeddings": false,        // both at top-level and inside text_config
```

### Safetensors index

The PARO checkpoint has BOTH tensors as distinct entries:

```
[248320, 2048] F16  lm_head.weight                                  <-- ignored by hipfire
[248320, 2048] F16  model.language_model.embed_tokens.weight        <-- used by hipfire as lm_head
```

### Loader (the bug)

`crates/hipfire-arch-qwen35/src/qwen35.rs:2081-2088` (inside
`load_weights_paroquant`):

```rust
eprintln!("  loading output (tied embeddings)...");
let output = {
    let (_, td) = source.tensor_data(embd_name).expect("embed_tokens for lm_head");
    let f: Vec<f32> = td.chunks_exact(2).map(|c| f16_to_f32(...)).collect();
    ...
    WeightTensor { buf, gpu_dtype: DType::F32, m: config.vocab_size, k: config.dim, ... }
};
```

`embd_name` is `"model.language_model.embed_tokens.weight"` (line 2067).
The comment "tied embeddings" is hard-coded — there is no check of
`config.tie_word_embeddings` (or the equivalent field on the safetensors
config). The HFQ path at `qwen35.rs:1843+` has the same issue, but HFQ
files are produced by hipfire's own quantizer which writes the lm_head
into the embed_tokens slot for tied models. Safetensors checkpoints
ship the two tensors separately.

### Reference (hipEngine)

`/tmp/hipengine-survey/hipengine/runtime/qwen35_paro_runner.py:509`:

```python
head_key = "lm_head.weight" if "lm_head.weight" in self.normalized_infos else "language_model.embed_tokens.weight"
```

hipEngine prefers `lm_head.weight` when present, falls back to embeddings
only when missing. Mirrored at qwen35_paro_runner.py:2747 in the resident
session path.

### Forward trace (added under `HIPFIRE_PARO_DEBUG=1`)

Decode of `What is 2 plus 2? Reply with just the number.` after chat-
template wrap produces tokens:

```
pos=0 -> 248045 (<|im_start|>)   <-- first generated token is a chat-template tag, NOT an answer
pos=1 -> 846    ("user")
pos=2 -> 198    ("\n")
pos=3 -> 3710   ("What")
pos=4 -> 369    (" is")
pos=5 -> 220    (" ")
pos=6 -> 17     ("2")
pos=7 -> 5346   (" plus")
pos=8 -> 220    (" ")
pos=9 -> 17     ("2")
... then drifts into 118401 (åĩºéĶĻ / 出错 / "error") repeated 8× in 26 tokens
```

The model is replaying the user prompt verbatim and then falling into a
Chinese-character attractor. This is the classic signature of "every
logit projection is computed against a wrong matrix" — argmax is
deterministic but semantically off-axis from the trained outputs.

Intermediate-layer numerics at pos=0 (decode of `<|im_start|>` start)
were checked at L0 of the DeltaNetMoe branch and look plausible:

- `x_normed[0:8]` post-RMSNorm has magnitudes ~0.5-2.2 (expected band for
  Gemma-style RMSNorm with `(1 + weight)` scale)
- `dn_qkv[0:8]` magnitudes 0.4-1.6
- `dn_v[0:8]` magnitudes ~1e-3 — small because conv1d is contracting
- `dn_attn_out[0:8]` magnitudes ~1e-6 — algebraically consistent with
  `attn[i] = beta * v[i] * (k . q)` at pos=0 where the recurrent S is
  empty (decay × 0 + beta × v ⊗ k^T)
- `x_post_attn_residual[0:8]` magnitudes 0.005-0.05 (dominated by
  embedding residual)
- `x_post_moe[0:8]` magnitudes 0.005-0.06

These all look numerically reasonable for a working forward — meaning
the bug must be after the layer stack, which is consistent with the
lm_head being wrong.

## Fix

`crates/hipfire-arch-qwen35/src/qwen35.rs:2081-2088` should:

1. Check whether `config.tie_word_embeddings` is true (it's exposed via
   `source.config` or the JSON metadata).
2. If `tie_word_embeddings: true` OR `lm_head.weight` is missing, keep
   the current behavior (reuse embed_tokens).
3. Otherwise, load `lm_head.weight` directly via
   `load_fp16_weight_from_source(source, gpu, "lm_head.weight",
   config.vocab_size, config.dim)` — note the absence of the
   `model.language_model.` prefix; `lm_head.weight` lives at the top
   level of the safetensors index (unlike `model.language_model.norm.weight`).

Sketch (drop-in for lines 2081-2088):

```rust
eprintln!("  loading output (lm_head)...");
let output = if source.tensor_info("lm_head.weight").is_some() {
    // Separate lm_head — load it directly (tie_word_embeddings=false case).
    load_fp16_weight_from_source(source, gpu, "lm_head.weight",
                                  config.vocab_size, config.dim)?
} else {
    // Tied embeddings — reuse embed_tokens.
    let (_, td) = source.tensor_data(embd_name).expect("embed_tokens for lm_head");
    let f: Vec<f32> = td.chunks_exact(2).map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect();
    let bytes: &[u8] = unsafe { std::slice::from_raw_parts(f.as_ptr() as *const u8, f.len() * 4) };
    let buf = gpu.upload_raw(bytes, &[config.vocab_size, config.dim])?;
    WeightTensor { buf, gpu_dtype: DType::F32, m: config.vocab_size, k: config.dim,
                   row_stride: 0, paro: None, awq_scale: None }
};
```

Same fix is needed in the HFQ load_weights path at `qwen35.rs:1843+`
if/when a HFQ artifact ships with `tie_word_embeddings=false` separately
(today's HFQ quantizer produces only tied output, so no test coverage
exists — leave alone for now and audit when adding HFQ untied support).

## Things ruled out

- **AWQ INT4 nibble repack** (`repack_awq_to_hfq4g128`): the
  `AWQ_DEQUANT = [0, 4, 1, 5, 2, 6, 3, 7]` permutation matches the
  canonical `awq_shift_for_pack_lane` in hipEngine's `paro_awq_gemv.hip`
  and the z-lab nano-vllm reference.
- **Givens rotation kernel** (`kernels/src/givens_rotate.hip`): pair-
  index layout, theta indexing (`[krot, hidden/2]`), channel_scales
  multiplication ordering, and Givens-2x2 sign convention
  (`x[i] = xi*c + xj*s, x[j] = xj*c - xi*s`) all match the z-lab
  reference at `paroquant/kernels/cuda/rotation.cuh:41-58`.
- **MoE shared sidecar aliasing** (`alias_paro_rotation` at
  `qwen35.rs:1248`): `MoeFfnWeights.paro_shared` retains ownership of
  the per-layer sidecar tensors and each expert's `ParoRotation` is a
  non-owning alias — no use-after-free risk; the sidecars outlive the
  forward pass.
- **GemmaRMSNorm `(1 + weight)` baking** (`paro_load_norm` at
  `qwen35.rs:2038-2048`): correctly baked into the host-side F32 vector
  before upload; verified by the `x_normed` traces having reasonable
  ~σ scale instead of the ~0.05 scale you'd see with the +1 missing.
- **Per-expert dispatch**: `weight_gemv` correctly dispatches on
  `expert.gate_up.gpu_dtype = ParoQ4G128` to the Givens-rotate +
  HFQ4G128 GEMV path (`llama.rs:719-742`).
- **The `in_proj_a` / `in_proj_b` dense fallback**: shapes
  `[32, 2048]` F16, no `.qweight` siblings — `paro_load_wt` correctly
  falls back to `load_fp16_weight_from_source` per the existing
  comparison doc.

## Reproducer

```
source ./scripts/gpu-lock.sh && gpu_acquire "paro-divergence"
cargo build --release --example daemon --example coherence_probe
HIPFIRE_PARO_DEBUG=1 HIPFIRE_GRAPH=0 \
LD_LIBRARY_PATH=/nix/store/q9mb3b1wcns128mgvcn0cf6dq2zd1pgh-system-path/lib \
./target/release/examples/coherence_probe \
    --model /home/bjoern/.hipfire/models/shisa-Qwen3.6-35B-A3B-PARO-unpacked \
    --prompt-file .scratch/smoke-prompt.txt \
    --max-tokens 26 --temperature 0.0
gpu_release
```

Expected (current): `attractor_last_128 FAIL: max_freq 0.42 (tok 118401),
unique_ratio 0.23 over 26 tokens`.

Expected (after fix): coherent output answering "4", no special-token
echo of the prompt structure.

## Trace instrumentation

Added `if layer_idx == 0 { paro_debug_dump(...) }` checkpoints in the
`forward_scratch_layers` DeltaNetMoe branch
(`qwen35.rs:8082-8200`). Gated behind the existing `HIPFIRE_PARO_DEBUG=1`
env var (helper at `qwen35.rs:7459`). Also added `eprintln!` entry
markers in `forward_scratch` (line 3728) and `forward_scratch_layers`
(line 7562). **Leave in place** for the orchestrator to verify the fix
takes; they're zero-cost when the env var is unset.

Note: the daemon binary must be rebuilt for these to fire — the probe
spawns `target/release/examples/daemon`, not `coherence_probe` itself.
