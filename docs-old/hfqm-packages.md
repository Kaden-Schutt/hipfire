# HFQM Artifact Packages

Hipfire `.hfq` files use the `HFQM` container layout for model weights and
for non-weight artifacts that should travel with structured metadata.

## Container Layout

```text
[0..4]    magic "HFQM"
[4..8]    version u32 little-endian
[8..12]   arch_id u32 little-endian
[12..16]  entry_count u32 little-endian
[16..24]  metadata_offset u64 little-endian
[24..32]  data_offset u64 little-endian

metadata JSON at metadata_offset

entry index immediately after JSON:
  u32 entry_count
  repeated:
    u16 name_len
    bytes name
    u8 quant_type
    u8 n_dims
    u32[n_dims] shape
    u32 group_size
    u64 data_size

payload data at data_offset, packed in index order
```

The weight loader interprets entries as tensors. Generic artifact tools should
interpret entries as named byte payloads and use metadata to define roles,
dtypes, and shapes.

## Reserved Arch ID

`arch_id = 0` is reserved for non-weight HFQM packages. Examples include KLD
references, imatrix captures, CASK/TriAttention centers, DFlash sidecars, and
eval evidence bundles.

Model-weight containers should continue to use their architecture-specific
`arch_id`. Non-weight packages must set `metadata.artifact_kind` so readers can
validate semantics before reading payload bytes.

## KLD Reference Package

KLD references use filenames like:

```text
qwen3.5-0.8b-bf16.kldref.hfq
```

Required metadata:

```json
{
  "artifact_kind": "hipfire.kldref",
  "package_schema": "hipfire.kldref.v1",
  "base_model_id": "qwen3.5-0.8b",
  "reference_precision": "bf16",
  "source_model_sha256": "...",
  "slice_md5": "...",
  "n_ctx": 2048,
  "n_vocab": 248320,
  "n_chunk": 1175,
  "top_k": 256,
  "kv_mode": "Fp32",
  "deltanet_state_precision": "fp32",
  "producer_cmd": "...",
  "scored_per_chunk": 1023
}
```

Required entries:

```text
kldref.tokens         u32 [n_chunk, n_ctx]
kldref.top_indices    u32 [n_chunk, scored_per_chunk, top_k]
kldref.top_log_probs  f32 [n_chunk, scored_per_chunk, top_k]
kldref.residual_mass  f32 [n_chunk, scored_per_chunk]
```

The filename is descriptive. Compatibility should be checked through metadata
and hashes, not by filename alone.
