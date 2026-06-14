# HFHS — Hipfire Hessian Sidecar binary format (v1)

**Status:** v1 spec. Companion to `docs/plans/gptq.md` Stage B Phase 1.
**Producer:** `scripts/collect_hessian.py` (Python, HF transformers + ROCm PyTorch).
**Consumer:** `crates/hipfire-runtime/src/hessian_io.rs` (Rust, reader); `crates/hipfire-quantize/src/gptq.rs` (Rust, quantize-time consumer).

## 1. Purpose + layout overview

A per-tensor Hessian sidecar stores `H_t = (1/N) · Σ_t x_t · x_tᵀ` for each linear layer's input activations, accumulated over the calibration corpus. The Rust quantizer mmaps this file and, per tensor, transforms the Hessian into the AWQ+FWHT basis, runs FP64 Cholesky, then executes GPTQ's column-sequential update.

File layout (little-endian throughout, native float byte order):

```
[ 24-byte header ]
[ tensor record 1 ]
[ tensor record 2 ]
...
[ tensor record N ]
```

No padding between records, no index — the consumer parses linearly and builds an in-memory name→offset map. For a 9B model with 184 tensors averaging K=4096, the file is ~6 GB; a one-time linear scan to index is negligible vs the per-tensor Cholesky cost.

## 2. Header (24 bytes)

| Offset | Size | Field | Notes |
|---:|---:|---|---|
| 0 | 4 | `magic` | ASCII `b"HFHS"` (Hipfire Hessian Sidecar) |
| 4 | 4 | `version` | `u32_le = 1` for this spec |
| 8 | 8 | `n_tensors` | `u64_le` total number of tensor records |
| 16 | 8 | `reserved` | `u64_le = 0` (for future format flags) |

The consumer MUST verify magic + version. Any deviation aborts loading with a clear error — silent compatibility breaks would corrupt downstream GPTQ output.

## 3. Per-tensor record (variable length)

| Offset | Size | Field | Notes |
|---:|---:|---|---|
| 0 | 4 | `name_len` | `u32_le` length of the tensor name in bytes |
| 4 | `name_len` | `name` | UTF-8 bytes; matches `.hfq` tensor naming (e.g. `model.language_model.layers.0.self_attn.q_proj`) — i.e. WITHOUT the trailing `.weight` |
| 4+`name_len` | 4 | `expert_idx` | `u32_le`; default `0`. Reserved for future MoE expert-conditional Hessians (Stage B.1). Stage B reads only records with `expert_idx == 0`. |
| 8+`name_len` | 4 | `K` | `u32_le`; the K dimension of the linear layer's input (number of input features) |
| 12+`name_len` | 4 | `dtype_flag` | `u32_le`; `1` = FP32, `2` = FP64. v1 producer always emits `1`; consumer must support `1` and may support `2`. |
| 16+`name_len` | `K * K * dtype_size` | `payload` | `K × K` Hessian matrix, row-major, little-endian floats |

`dtype_size` = 4 for FP32, 8 for FP64.

### 3.1 Why the `.weight` suffix is stripped from `name`

Storage in `.hfq` uses the safetensors convention (`model.language_model.layers.0.q_proj.weight` — with `.weight`). But the Hessian is a property of the *layer's input*, not the weight tensor itself. Stripping `.weight` from the Hessian key:
- Removes the redundant suffix (every linear layer has a `.weight`).
- Makes the key shorter (~7 bytes per tensor saved across 200+ tensors).
- Aligns with `nn.Linear` module naming (which doesn't have `.weight` at the end).

The Rust quantizer's lookup must therefore strip `.weight` from the .hfq tensor name when querying the Hessian sidecar.

### 3.2 Why `expert_idx` is reserved in v1

Stage B is dense + linear_attn only — no per-expert Hessians. Including the field now means Stage B.1 can populate it without bumping the format version. Stage B's reader skips records where `expert_idx != 0`.

For MoE in Stage B.1: each expert `e` gets its own Hessian record with `expert_idx = e`, keyed by the same tensor name (e.g. all 64 experts of `mlp.experts.gate_up_proj` share the name but differ in `expert_idx`). The Python collector's hook gates accumulation on the router's expert selection per token.

### 3.3 Why FP32 default, FP64 supported

Hessians are *expected outer products* — by the central limit theorem, they're smooth functions with limited dynamic range across entries. FP32 stores them at < 1 ULP rounding per entry per token, well below the damping `λ` GPTQ adds at Cholesky time anyway. For 9B's largest K=4096 tensor, FP32 saves 32 MB per Hessian vs FP64 — 6 GB total instead of 12 GB.

For high-precision experiments (e.g. comparing GPTQ output to a Python reference), FP64 producers may emit `dtype_flag = 2`. The quantizer transparently promotes FP32 → FP64 for Cholesky regardless.

## 4. Validation requirements (consumer side)

The Rust reader MUST:

1. **Verify magic + version.** Mismatch → return `Err(InvalidFormat)`.
2. **Bounds-check every record.** A truncated file (e.g. partial download from NFS) must fail loading, not silently feed garbage to GPTQ.
3. **Verify diagonal positivity** for each Hessian: every `H[i,i] >= 0` (Hessian is PSD by construction; sum of rank-1 PSD matrices). Negative diagonals indicate FP corruption — log per-tensor and either fall back to plain MQ4 quant for that tensor or abort.
4. **Verify symmetry** to a tolerance (e.g. `|H[i,j] - H[j,i]| < 1e-3 * max(|H[i,i]|, |H[j,j]|)`). Allow a sampled subset (say 32 random off-diagonal pairs) to keep validation cheap.

The Rust reader SHOULD (best-effort):

5. **Sanity-check K against the corresponding `.hfq` tensor's K dimension.** Mismatch → log warning + skip GPTQ for that tensor (likely indicates the Hessian was collected on a different model variant).
6. **mmap the file** with `POSIX_FADV_SEQUENTIAL` for the per-tensor Cholesky walk (sequential access pattern). The pager evicts pages we've finished with as Cholesky progresses tensor-by-tensor.

## 5. Reference: in-tree producers + consumers

**Producer (Python):** `scripts/collect_hessian.py:write_hessian_file()` — writes the v1 format exactly as specified above.

**Consumer (Rust, to be implemented in Phase 1.3):**

```rust
// crates/hipfire-runtime/src/hessian_io.rs

use memmap2::Mmap;

pub struct HessianSidecar {
    mmap: Mmap,
    index: HashMap<(String, u32), TensorOffset>,  // (name, expert_idx) → offset
}

pub struct HessianRef<'a> {
    pub name: &'a str,
    pub expert_idx: u32,
    pub k: usize,
    pub dtype: HessianDtype,
    pub bytes: &'a [u8],          // K*K*dtype_size
}

pub enum HessianDtype { F32, F64 }

impl HessianSidecar {
    pub fn open(path: &Path) -> Result<Self, HessianError> { ... }
    pub fn get(&self, name: &str, expert_idx: u32) -> Option<HessianRef<'_>> { ... }
    pub fn tensors(&self) -> impl Iterator<Item = HessianRef<'_>> { ... }
}
```

The quantizer queries `sidecar.get(&tensor_name.trim_end_matches(".weight"), 0)` per MQ4G256 weight; missing tensors fall through to plain MQ4 quantization (with a warning).

## 6. Sample artifact

`benchmarks/quality-baselines/refs/qwen3.5-9b-bf16.hessian.bin` (~6 GB, ~184 records).

First few bytes (hex dump for verification, 2026-05-12 0.8B smoke test):

```
HFHS\x01\x00\x00\x00  \xba\x00\x00\x00\x00\x00\x00\x00  \x00\x00\x00\x00\x00\x00\x00\x00
^magic                ^n_tensors=186 (0xba)              ^reserved=0
^version=1
```

Followed by record 1 (varies by collection order — typically the first-seen module in `model.named_modules()`).

## 7. Versioning policy

- **v1 (this spec):** baseline. FP32/FP64 dtype flag, `expert_idx` reserved, no compression.
- **v2 (potential):** add per-tensor `metadata_bytes` slot for arbitrary key-value (e.g. AWQ scale vector co-located with the Hessian — saves a separate sidecar). Bump magic to `b"HFHS"` + version=2. Reader rejects unknown versions to fail safely.
- **v3+ (speculative):** compression (zstd per Hessian — Hessians have low entropy on natural language activations; could halve disk cost). Requires consumer changes to decompress lazily.

Bumping the format requires updating the Python producer (`scripts/collect_hessian.py`), the Rust reader (`hessian_io.rs`), and this spec doc together. The consumer MUST reject older / newer versions unless explicitly accepting compatibility.
