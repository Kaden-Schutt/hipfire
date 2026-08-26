# Quantization

How hipfire stores weights and KV cache (design / math). For the operator
workflow (`hipfire quantize` / `hipfire-quantize`), see [QUANTIZE.md](QUANTIZE.md).
Format-specific specs: [quant-formats/mq4-v2.md](quant-formats/mq4-v2.md),
[quant-formats/mq-v2-family.md](quant-formats/mq-v2-family.md),
[quant-formats/ladder.md](quant-formats/ladder.md),
[quant-formats/qt-register.txt](quant-formats/qt-register.txt),
[quant-formats/hfp4.md](quant-formats/hfp4.md).

Truth labels follow [INDEX.md](INDEX.md). Mutable inventories (CLI flags, registry
KV recommendations, arch admissions) are owned by their source pages — this file
describes the math and wire contracts.

## Weight families at a glance

| Family | Rotation | Element | Group | Role |
|---|---|---|---|---|
| **HFQ** | none | INT asymmetric | G256 (G128 variants exist) | Dense Llama / Mistral / Gemma / plain Qwen |
| **MQ (MagnumQuant)** | offline FWHT-256 | INT asymmetric (or Lloyd codebook) | G256 | Qwen 3.5+ hybrid path; FWHT baked into weights |
| **HFP / MFP** | optional offline FWHT | OCP E2M1 FP4 | G32 rows | RDNA-oriented FP4; MFP = HFP + FWHT |
| **PARO** | pairwise Givens | AWQ W4 | G128 | Imported ParoQuant/AWQ probe path |
| **Q8F16** | none | INT8 + f16 scale | G32 | Embeddings, routers, precision-sensitive 2D |

Production defaults for day-to-day use:

| CLI / extension | Wire type | Bits | Bytes / 256 el | When |
|---|---|---|---|---|
| `mq4` / `.mq4` | **`MQ4G256V2` (qt 44)** | 4 | 136 (8 hdr dual-fp16 per-128 + 128 data) | Qwen 3.5+ safetensors default; **`--format mq4` → V2** |
| `mq4v1` / `mq4g256` / `magnum` | `MQ4G256` (qt 13) | 4 | 136 (8 hdr f32 scale+min + 128 data) | Legacy Magnum v1 only — not the `mq4` alias |
| `mq6` / `.mq6` | `MQ6G256` (qt 15) | 6 | 200 | Qwen 3.5+ higher quality (v1 body) |
| `mq6v2` … `mq2v2` | V2 family qt 47/48/49/50 | 6…2 | 200/168/104/72 | Dense Magnum V2 ladder bodies — see below |
| `hf4` / `.hf4` | `HFQ4G256` (qt 6) | 4 | 136 | Dense GGUF default; Llama-style dense |
| `hf6` / `.hf6` | `HFQ6G256` (qt 8) | 6 | 200 | Dense, higher quality |
| `mq3` / `.mq3` | `MQ3G256` (qt 17) | 3 | 104 (8 hdr + 96 data) | Sub-4-bit bandwidth; quality-sensitive |
| `q8` | `Q8F16` (qt 3) | 8 | **34 B / 32 el** (2 B f16 scale + 32×i8; 272 B / 256 el) | Reference / debug / protected tensors |

Under standard MQ/HF/HFP/MFP recipes, embeddings are forced to `Q8F16`
(Q4-grade embedding error compounds). 1D norms / scales stay F16. Direct
recipes such as `q4k-all` intentionally bypass that embedding rule.

### Magnum V2 family (qt 44 / 47–50) and MQ4C (qt 45)

Magnum V2 uses a neutral-size **8 B dual half `fp16 scale + fp16 zero` header
per 128 weights**, with payload packing unchanged from the matching v1 bit
width. Group strides: bits **2/3/4/5/6 → 72/104/136/168/200** B per 256
elements. MQ4C is separate: one per-256 fp16 scale+zero pair at `[0..4)`,
zero padding at `[4..8)`, and the 4-bit payload at `+8`.

| `--format` | qt | `DType` | B/group | Product note |
|---|---:|---|---:|---|
| `mq4` *(alias)*, `mq4v2` | **44** | `MQ4G256V2` | 136 | Default MQ4 body; dense Qwen3.8 ladder evidence |
| `mq4c` | **45** | `MQ4CG256` | 136 | v1.5 pad (fp16 header @+0, zero pad +4..8, nibbles @+8) |
| `mq6v2` | **47** | `MQ6G256V2` | 200 | Dense product candidate (ladder) |
| `mq5v2` | **48** | `MQ5G256V2` | 168 | Dense product candidate (ladder) |
| `mq3v2` | **49** | `MQ3G256V2` | 104 | Dense product candidate (ladder) |
| `mq2v2` | **50** | `MQ2G256V2` | 72 | **Wire + runtime supported; product quality-rejected** |

**Wire implementation ≠ product admission.** MQ2V2 loads, decodes, and passes
parity, but measured Qwen3.8 ladder KLD is catastrophic (~12–14 nats WT2/v6sel).
**Do not call MQ2V2 a shipping product body.** MQ3–MQ6 V2 cells are measured
candidates on the product ladder, not automatic registry cards. Full contracts:
[mq4-v2.md](quant-formats/mq4-v2.md), [mq-v2-family.md](quant-formats/mq-v2-family.md),
[ladder.md](quant-formats/ladder.md).

Product names (`mq4`, `mq4-xt`, `mq4 pro`, …) are **bpw-class + role-lift**
labels owned by the ladder, separate from codec generation (`V2`, `C`) and
from wire ids. A product row must satisfy measured model bpw ∈ [N, N+1).

### Bonsai (ternary / 1-bit)

| Type | qt | Notes |
|---|---:|---|
| `TQ2G128` | 40 | PrismML Bonsai ternary — passthrough |
| `BQ1G128` | 41 | PrismML Bonsai 1-bit — passthrough; won qt=41 adjudication 2026-08-18 |

### Sub-4-bit and research formats (v1 / Lloyd / FP4)

| Type | qt | Bytes/group | Truth state (INDEX) |
|---|---|---|---|
| `MQ3G256` | 17 | 104 | **shipped / ref-pinned** encoder + decode/prefill paths; quality degrades below ~9B (historical local battery — see below) |
| `MQ2G256` | 18 | 72 | **shipped / ref-pinned** encoder; reserved product use — uniform 4-level codebook collapses; needs `--allow-mq2` / `HIPFIRE_ALLOW_MQ2=1` (research opt-in qualifier) |
| `MQ2G256Lloyd` | 19 | 72 | **shipped / ref-pinned** implementation; research opt-in (`--allow-mq2-lloyd`) — not a product default |
| `MQ3G256Lloyd` | 20 | 112 | **shipped / ref-pinned** implementation; research opt-in (`--allow-mq3-lloyd`) — not a product default |
| `MQ4G256Lloyd` | 30 | 160 | **shipped / ref-pinned** implementation; research opt-in (`--allow-mq4-lloyd`) — qt renumbered 21→30; pre-renumber artifacts must be re-quantized |
| `MQ5G256` | 31 | 168 | **shipped / ref-pinned** encoder present (5-bit FWHT, 5.25 bpw) |
| `MQ2G256GL` / `MQ3G256GL` | 38 / 39 | GL SoA | **shipped / ref-pinned** Magnum GL passthrough |
| `HFP4G32` | 21 | row layout | **shipped / ref-pinned** FP4 family (see hfp4.md) |
| `MFP4G32` | 24 | same as HFP4 + FWHT flag | **shipped / ref-pinned** drop-in MQ4-class FP4+FWHT |
| `MFP4G32Lloyd` / `MFP4G32P` / `MFP4G32E8` / SoA / `MFP3G32E8` / `MFP2G32E8` | 32–37 | row layouts | **shipped / ref-pinned** encoders + selective kernels; experimental opt-in qualifier; MoE cold-tier recipes — not CLI defaults |
| `PARO4G128` / `PARO4G128T` | 28 / 29 | Paro buffers | **shipped / ref-pinned** load/probe path for ParoQuant imports |
| `MQ2G256V2` | 50 | 72 | **wire/runtime supported; product quality-rejected** — see V2 family section |

**Historical quality note (not a live admission):** uniform MQ3/MQ2 local
battery on **gfx1100** (7900 XTX), **Qwen 3.5 / 3.6**, master ref **`c448d5e`**
(2026-04-30). MQ3 was fluent on-topic at 9B and 27B across a 4-prompt coherence
set; 4B was partial collapse; 0.8B gibberish. Uniform MQ2 collapsed at every
size tested. Treat as scoped **historical** provenance, not a product floor.
MQ3-Lloyd / MQ2-Lloyd remain research-gated until product validation. Separately,
the **2026-08-20** dense Qwen3.8 MQ V2 ladder rejects **MQ2V2** on KLD while
keeping wire support.

### Prefill / WMMA runtime batch eligibility (weights)

`is_batchable_la(dtype, arch)` is **runtime batch eligibility only** — it is
**not** product admission and is **not** an empty/full admission registry.
llama and qwen35 each own their own predicate and are **not** currently in
lockstep (`crates/hipfire-runtime/src/llama.rs`,
`crates/hipfire-arch-qwen35/src/qwen35.rs`).

**Always batch-eligible (both owners):** `MQ4G256`, `HFQ4G256`, `MQ6G256`,
`HFQ6G256`, `Q8_0`. (qwen35 also treats `ParoQ4G128` / `F32` as always-ok for
the PARO probe path.)

| DType | `llama::is_batchable_la` | `qwen35::is_batchable_la` |
|---|---|---|
| `MQ3G256` | WMMA on gfx1100/1101/1102/1150/1151/1200/1201; scalar batched on gfx101x/103x; else not | Same plus **gfx1103** and **gfx1152** on the WMMA set; same gfx10 scalar set |
| `HFP4G32`, `MFP4G32` | WMMA arches only (llama set above) | WMMA arches only (qwen35 set, includes gfx1103/1152) |
| `MQ3G256Lloyd` | **Not** batch-eligible | WMMA on gfx1100/1101/1102/1150/1151; gfx1200/1201 only with `HIPFIRE_LLOYD_GFX12=1` |
| `MQ4G256Lloyd` | **Not** batch-eligible | WMMA on gfx1100/1101/1102/1151 (no gfx1150); gfx1200/1201 only with `HIPFIRE_LLOYD_GFX12=1` |
| Other Lloyd / E8 / research dtypes | Not batch-eligible here | Additional arms (e.g. E8) may exist behind their own env gates — see source |

Uniform MQ3 ships WMMA residual / fused GEMM kernels on RDNA3+ and is
batch-eligible on the arches above. Do **not** claim “no MQ3 WMMA.” Lloyd
batch eligibility is owner-specific (qwen35 yes on listed arches; llama no).

## FWHT rotation (the M in MQ)

Each 256-wide group is transformed by **two** fixed `±1` sign diagonals and a
Walsh–Hadamard butterfly, then scaled by `1/√256 = 1/16`. Encoder and runtime
share the same contract (`cpu_fwht_256` / `fwht_forward_256` /
`rotate_x_mq`):

```
R = D2 · H · D1 / 16
  signs1 → Hadamard butterfly → ×(1/16) → signs2

quant:       w′ = R w   →  quantize w′
inference:   x′ = R x   (same R, not an inverse)
             y  = (w′)ᵀ x′  = wᵀ x     in exact arithmetic
```

Sign tables use fixed PRNG seeds **42** (`D1` / signs1) and **1042** (`D2` /
signs2), shared by quantizer and runtime. Cancellation holds in exact
arithmetic; **fp32** mixed-precision paths accumulate rounding drift (typically
below per-block quantization error, but not bit-exact identity).

**Why it matters:** outliers spread across the group, narrowing dynamic range
before 4-bit (or lower) quantization. The rotation was calibrated against Qwen
3.5 / 3.6 (DeltaNet hybrid) weight distributions — MQ4 targeting Q8-class quality
at Q4 bandwidth on that family is the central product claim.

On Llama-style dense models the math still cancels, but there is typically **no
quality lift** over plain HFQ4 — only the extra `rotate_x_mq` cost. Practical
rule: **MQ for Qwen 3.5+; HFQ for everything else.** CLI defaults match
(`safetensors` → `mq4`, GGUF → `hf4`).

## Header layout (HFQ/MQ G256 affine blocks)

### v1 (qt 6/13/15/17/18/…)

Per 256 elements, 8-byte header + packed payload. HFQ4 and MQ4 v1 both store an
**f32 affine scale** and an **f32 minimum** (not FP16-in-f32):

- scale: f32
- zero / min: f32
- data: `bitwidth × 256 / 8` bytes (128 for 4-bit, 96 for 3-bit, 192 for 6-bit, …)

### V2 (qt 44 / 47–50) and MQ4C (qt 45)

Same 8-byte header budget and payload offset **+8**, different header encoding:

- **MQ\*G256V2:** fp16 scale+zero for half 0 (weights 0–127) at `[0..4)`, half 1
  at `[4..8)`; payload packing identical to the matching v1 bit width.
- **MQ4CG256:** one packed fp16 scale+zero at `[0..4)`, mandatory zero pad
  `[4..8)`, 128 B nibbles at `[8..136)` (v1-compatible geometry).

Lloyd variants replace the uniform grid with a per-block (or per-tensor) fp16
codebook plus packed indices; byte sizes differ (e.g. MQ3-Lloyd 112 B/group,
MQ4-Lloyd 160 B/group).

## KV cache

K and V are quantized differently: K enters softmax (errors exponentiate); V is
already attention-weighted. Live modes store **V as Q8** and compress **K**:

| Mode | K | V | Notes |
|---|---|---|---|
| `q8` | Q8_0 (G32) | Q8_0 | Default fall-through; DFlash-safe |
| `fwht4` / `fwht3` / `fwht2` | FWHT-rotated K at 4/3/2-bit | Q8_0 | Same byte layout family as legacy asym; FWHT basis matches MQ drafts → better DFlash acceptance |
| `asym4` / `asym3` / `asym2` | Givens + Lloyd-Max K | Q8_0 | **Legacy**; degrades DFlash acceptance vs fwht* |
| `turbo` / `turbo2` / `turbo3` / `turbo4` | aliases | | Map to asym3 / asym2 / asym3 / asym4 |

**Default resolution** (`crates/hipfire-cli/src/main.rs`):

1. `HIPFIRE_KV_MODE` env
2. Per-model config `kv_cache`
3. Registry `default_kv_mode` when config is `auto`
4. Else **`q8`**

There is **no** live per-arch `archDefaults` table (removed). Compressed modes
are opt-in via registry cards or explicit config — not guessed from GPU VRAM.
Override: `hipfire config set kv_cache <mode>` or `HIPFIRE_KV_MODE=<mode>`.

Optional **adaptive** downshift (`kv_adaptive`: `off` | `conservative` |
`balanced` | `aggressive` | `advanced:k=…,v=…`) can lower V and K tiers as
context grows; see [CONFIG.md](CONFIG.md). Works best when K starts at a higher
fwht tier.

## HFQM container (`.hfq` / `.mq4` / `.hf4` / …)

Extensions are naming convenience; the on-disk magic is always **`HFQM`**:

```
0x00  "HFQM"              (4 bytes)
0x04  version             (u32 LE = 1)
0x08  arch_id             (u32 LE — see architecture-ids.md)
0x0C  n_tensors           (u32 LE)
0x10  metadata_offset     (u64 LE)
0x18  data_offset         (u64 LE — 4096-aligned)

[metadata_offset .. data_offset):
  metadata_json (UTF-8: config, tokenizer, gguf_meta, source, …)
  tensor_index:
    n_tensors: u32 LE
    for each tensor:
      name_len: u16 LE
      name: utf-8
      quant_type: u8      (see QuantType table below)
      n_dims: u8
      shape: u32 LE × n_dims
      group_size: u32 LE
      data_size: u64 LE

[data_offset .. EOF]: tensor bytes in index order
```

`HfqFile::open` mmaps the file; tensor slices feed HIP upload without an extra
host copy. Bundled `.mq4-mtp` files append an MTP HFQM section + `HFBNDMTP`
trailer; trunk loaders ignore the trailer and see only the trunk.

### QuantType IDs (encoder / loader contract)

Source of truth: [`quant-formats/qt-register.txt`](quant-formats/qt-register.txt).

| ID | Name | Notes |
|---:|---|---|
| 0 | Q4F16G64 | Legacy |
| 1 | F16 | |
| 2 | F32 | |
| 3 | Q8F16 | Embeddings / protected |
| 4 | Q4K | GGUF-family |
| 5 | Q8HFQ | |
| 6 | HFQ4G256 | Production dense 4-bit (unrotated) |
| 7 | HFQ4G128 | |
| 8 | HFQ6G256 | |
| 9–12 | HFQ2/3 G256/G128 | |
| 13 | MQ4G256 | Legacy Magnum 4-bit v1 (`mq4v1` / `mq4g256` / `magnum`) |
| 14 | MQ8G256 | |
| 15 | MQ6G256 | Magnum 6-bit v1 |
| 16 | BF16 | Vision / lossless |
| 17 | MQ3G256 | Uniform 3-bit MQ v1 — **shipped / ref-pinned** |
| 18 | MQ2G256 | Uniform 2-bit v1 — **shipped / ref-pinned** encoder; reserved product / research opt-in |
| 19 | MQ2G256Lloyd | **shipped / ref-pinned** impl; research opt-in |
| 20 | MQ3G256Lloyd | **shipped / ref-pinned** impl; research opt-in |
| 21 | HFP4G32 | Canonical HFP4 — **shipped / ref-pinned** |
| 22 | TidI32 | DeepSeek tid2eid tables (**reassigned** from former HFP4G16 reservation) |
| 23 | *(reserved)* | Former HFP4G64 ablation slot — still reserved, not built |
| 24 | MFP4G32 | HFP4 + offline FWHT — **shipped / ref-pinned** |
| 25–27 | *(reserved)* | Former MX/NV/HFP8 interop slots — still reserved, not built |
| 28 | PARO4G128 | ParoQuant native — **shipped / ref-pinned** load path |
| 29 | PARO4G128T | Engine-tiled Paro — **reassigned** from former `MFP4G32R` online-R reservation |
| 30 | MQ4G256Lloyd | Research MQ Lloyd — **reassigned** (was 21 pre-renumber off HFP4) |
| 31 | MQ5G256 | 5-bit MQ v1 — **shipped / ref-pinned** encoder |
| 32–37 | MFP4Lloyd / MFP4P / E8 / SoA / MFP3E8 / MFP2E8 | **shipped / ref-pinned** encoders + selective kernels; experimental opt-in |
| 38 | MQ2G256GL | Magnum GL 2-bit — passthrough |
| 39 | MQ3G256GL | Magnum GL 3-bit — passthrough |
| 40 | TQ2G128 | Bonsai ternary — passthrough |
| 41 | BQ1G128 | Bonsai 1-bit — passthrough |
| 44 | MQ4G256V2 | **Default `mq4` body** — dual-fp16 per-128 header, 136 B |
| 45 | MQ4CG256 | MQ4 v1.5 pad — 136 B |
| 47 | MQ6G256V2 | Magnum 6-bit V2 — 200 B |
| 48 | MQ5G256V2 | Magnum 5-bit V2 — 168 B |
| 49 | MQ3G256V2 | Magnum 3-bit V2 — 104 B |
| 50 | MQ2G256V2 | Magnum 2-bit V2 — 72 B; **wire OK, product rejected** |

**Current reserved wire IDs:** 23 and 25–27 only. IDs **22, 29, and 30 were
reassigned** (TidI32, PARO4G128T, MQ4G256Lloyd). qt **42–43 and 46** are not
live product rows in the register (43 burned by retired SEL; 46 is hierarchical
`mq4_k` design-only). Do not invent slots — edit `qt-register.txt` to claim a
number.

### PARO4G128 probe payload

`quant_type=28` is **not** row-major HFQ4/MQ4. Each tensor stores ParoQuant
buffers in order:

```
qweight        int32 [K, M/8]
qzeros         int32 [K/128, M/8]
scales         f16   [K/128, M]
pairs          int16 [8, K]
theta          f16   [8, K/2]
channel_scales f16   [K]
```

Runtime contract:

```
x_rot = rotate(x, pairs, theta, channel_scales)
y     = awq_w4_gemv(x_rot, qweight, qzeros, scales)
```

Verify imports with `python3 scripts/astrea.py paro-oracle --source PARO_SAFE_DIR
--hfq MODEL.hfq --module MODULE --pretty` before quality or perf claims.

## Why a custom format

llama.cpp GGUF Q4_K_M is similar on disk size and in-place GEMV cost. hipfire’s
gains are engineering, not a new algorithm class:

1. **Fused projections** — QKV / QKVZA / gate+up fused launches for HFQ4/MQ4
   (and siblings) cut launch overhead on decode.
2. **WMMA prefill** — batched GEMM on RDNA3+ WMMA for MQ/HFQ (and MQ3/HFP on
   eligible arches) amortizes dequant across prompt tokens.

Published speedups belong in measured owners ([BENCHMARKS.md](BENCHMARKS.md),
perf-checkpoints); do not promote undated multiples here.

## Related

- Operator quantize: [QUANTIZE.md](QUANTIZE.md)
- Magnum V2 / family / ladder / register:
  [quant-formats/mq4-v2.md](quant-formats/mq4-v2.md),
  [quant-formats/mq-v2-family.md](quant-formats/mq-v2-family.md),
  [quant-formats/ladder.md](quant-formats/ladder.md),
  [quant-formats/qt-register.txt](quant-formats/qt-register.txt)
- HFP4 wire spec: [quant-formats/hfp4.md](quant-formats/hfp4.md)
- Config / env KV knobs: [CONFIG.md](CONFIG.md), [env-vars.md](env-vars.md)
- Models / registry: [MODELS.md](MODELS.md)
- Arch IDs: [architecture-ids.md](architecture-ids.md)
- Sources: `crates/hipfire-quantize/src/` (`QuantType`, encoders, pipeline),
  `crates/hipfire-runtime/src/{hfq,llama,weight_backend}.rs`,
  `crates/hipfire-cli/src/main.rs`, `crates/hipfire-config/src/lib.rs`
