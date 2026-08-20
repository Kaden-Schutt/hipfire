# MQ V2 family — wire spec (qt 47 / 48 / 49 / 50)

**Status:** qt=47 `MQ6G256V2`, qt=48 `MQ5G256V2`, qt=49 `MQ3G256V2`, and
qt=50 `MQ2G256V2` are **wired for dense gfx12** (gfx1200/gfx1201) beside
qt=44 `MQ4G256V2` (see [`mq4-v2.md`](mq4-v2.md)). Register disposition is
`passthrough` for all four (`qt-register.txt`).

**One-line summary:** same **neutral-size Magnum V2** header as MQ4V2 —
8 B of dual half `fp16 scale + fp16 zero` per 128 weights — on top of the
**legacy bit-width payloads** from MQ6/5/3/2 v1. Group byte counts and
nominal codec bpw are unchanged from v1 at each bit width.

**Wire implementation ≠ product admission.** A format that loads, decodes,
and passes parity can still be rejected as a product body. In particular
**MQ2V2 is measured and rejected for product quality** (catastrophic KLD;
decoded text coherent but degraded/repetitive). Do not call MQ2V2 shipping.

Product tier names, rung recipes, and the full 15-cell Qwen3.8 ladder live in
[`ladder.md`](ladder.md) and
`docs/perf-checkpoints/2026-08-20-qwen38-mq-v2-product-ladder.md`. This file
is the **byte contract** a loader/kernel implementer needs.

---

## 1 · Family table

| `--format` | qt | `DType` | B/group | nominal bpw | payload | levels | step divisor |
|---|---:|---|---:|---:|---|---:|---:|
| `mq6v2` | **47** | `MQ6G256V2` | **200** | **6.25** | `[8..200)` 192 B 6-bit | 0..63 | 63 |
| `mq5v2` | **48** | `MQ5G256V2` | **168** | **5.25** | `[8..168)` 160 B 5-bit | 0..31 | 31 |
| `mq4v2` | **44** | `MQ4G256V2` | **136** | **4.25** | `[8..136)` 128 B 4-bit nibbles | 0..15 | 15 |
| `mq3v2` | **49** | `MQ3G256V2` | **104** | **3.25** | `[8..104)` 96 B 3-bit | 0..7 | 7 |
| `mq2v2` | **50** | `MQ2G256V2` | **72** | **2.25** | `[8..72)` 64 B 2-bit | 0..3 | 3 |

Constants (source of truth):

- `crates/rdna-compute/src/dispatch.rs` —
  `MQ6G256V2_GROUP_BYTES=200`, `MQ5G256V2_GROUP_BYTES=168`,
  `MQ3G256V2_GROUP_BYTES=104`, `MQ2G256V2_GROUP_BYTES=72`
  (and `MQ4V2_GROUP_BYTES=136` for qt=44).
- Encoders: `quantize_mq6g256v2` / `quantize_mq5g256v2` in
  `crates/hipfire-quantize/src/quant_fwht.rs`;
  `quantize_mq3g256v2` / `quantize_mq2g256v2` in
  `crates/hipfire-quantize/src/quant_mq.rs`.
- RAW_CODECS: qt 47→`MQ6G256V2`, 48→`MQ5G256V2`, 49→`MQ3G256V2`, 50→`MQ2G256V2`
  in `crates/hipfire-runtime/src/weight_backend.rs`.
- qwen35 `dtype_from_quant_type`: 47/48/49/50 map to the same four `DType`s
  and **must not** collapse to legacy MQ6/5/3/2.

qt=44 is specified fully in [`mq4-v2.md`](mq4-v2.md). It is listed here only
so the family header contract is visible in one place.

---

## 2 · Shared header (all five V2 formats)

Little-endian throughout. Per 256-weight group:

| offset | size | field |
|---|---|---|
| `[0..2)` | fp16 | `scale` half 0 (weights 0–127) |
| `[2..4)` | fp16 | `zero` half 0 |
| `[4..6)` | fp16 | `scale` half 1 (weights 128–255) |
| `[6..8)` | fp16 | `zero` half 1 |
| `[8..B)` | `B-8` | bit-width payload (format-specific; **byte-identical layout to the matching v1**) |

Reconstruction:

```
h = (weight_index >= 128) ? 1 : 0
w = q * f32(scale[h]) + f32(zero[h])
```

Half 0 owns `q[0..128)`; half 1 owns `q[128..256)`.

### Geometry invariants

- `K % 256 == 0` (loader and encoder both assert).
- Tensor blob length = `M * (K/256) * GROUP_BYTES` for that format.
- Group alignment is 8-byte (header is four fp16s).
- Payload offset is always **+8**, same as v1 MQ*/HFQ* G256 affine formats.
- FWHT-256 rotation is baked into the weights at encode time; runtime rotates
  `x` on the Magnum path (`RotationPlan::FwhtG256`).

### Why the halves land on lane boundaries

Same argument as MQ4V2: decode kernels index contiguous packs per lane so that
weights 0–127 and 128–255 do not straddle a half-wave. Header dwords stay
lane-invariant scalar loads. Derive the half select from the kernel's own
payload addressing; do **not** hard-code a lane id without re-deriving
(see `mq4-v2.md` §4).

---

## 3 · Payload packing (legacy, bit-width specific)

Payload packing is **unchanged from the matching v1** format. Only the header
is V2.

### 3.1 · MQ6G256V2 — 6-bit, 4 values / 3 bytes

For `i = 0,4,8,…,252` with byte offset `bo = 8 + (i/4)*3`:

```
q0..q3 in 0..63
out[bo+0] =  q0        | (q1 << 6)
out[bo+1] = (q1 >> 2)  | (q2 << 4)
out[bo+2] = (q2 >> 4)  | (q3 << 2)
```

192 payload bytes / group.

### 3.2 · MQ5G256V2 — 5-bit, 8 values / 5 bytes

For `i = 0,8,16,…,248` with `bo = 8 + (i/8)*5`:

```
q0..q7 in 0..31
out[bo+0] =  q0        | (q1 << 5)
out[bo+1] = (q1 >> 3)  | (q2 << 2) | (q3 << 7)
out[bo+2] = (q3 >> 1)  | (q4 << 4)
out[bo+3] = (q4 >> 4)  | (q5 << 1) | (q6 << 6)
out[bo+4] = (q6 >> 2)  | (q7 << 3)
```

160 payload bytes / group.

### 3.3 · MQ3G256V2 — 3-bit LE bitstream, 8 values / 3 bytes

For `chunk = 0..31`, `ci = chunk*8`, `bo = 8 + chunk*3`, each `qq[j] = q[ci+j] & 7`:

```
b0 = (qq[0] & 7) | ((qq[1] & 7) << 3) | ((qq[2] & 3) << 6)
b1 = ((qq[2] >> 2) & 1) | ((qq[3] & 7) << 1) | ((qq[4] & 7) << 4) | ((qq[5] & 1) << 7)
b2 = ((qq[5] >> 1) & 3) | ((qq[6] & 7) << 2) | ((qq[7] & 7) << 5)
```

96 payload bytes / group.

### 3.4 · MQ2G256V2 — 2-bit, 4 values / byte

For `i = 0..63`:

```
byte = (q[4i] & 3) | ((q[4i+1] & 3) << 2) | ((q[4i+2] & 3) << 4) | ((q[4i+3] & 3) << 6)
out[8 + i] = byte
```

64 payload bytes / group.

### 3.5 · MQ4G256V2 — 4-bit nibbles

See [`mq4-v2.md`](mq4-v2.md) §2: 128 B nibble payload at `[8..136)`,
byte-identical to qt=1/6/13.

---

## 4 · Encoder contract

For each 256-weight group, after FWHT-256 (`cpu_fwht_256` with the tensor's
sign seeds):

```
for h in {0, 1}:
    lo = min(w[128h .. 128h+127])
    hi = max(w[128h .. 128h+127])
    levels_max = 2^bits - 1          # 63 / 31 / 15 / 7 / 3
    step_f32 = (hi > lo) ? (hi - lo) / levels_max : 0
    scale[h] = f16(step_f32)         # force 0 bits if hi == lo
    zero[h]  = f16(lo)
    # REQUIRED: quantize against ROUND-TRIPPED fp16 values
    st = f32(scale[h]); z = f32(zero[h])
    degenerate = (hi == lo) || (step_f32 == 0) || (st == 0)
    if degenerate:
        q[half] = 0
    else:
        for i in half h:
            q[i] = clamp(floor((w[i] - z) / st + 0.5), 0, levels_max)
pack header LE: s0, z0, s1, z1
pack payload with the bit-width rules in §3
```

**Degenerate half:** `scale=0`, `zero=f16(lo)`, `q=0` → decoder reproduces `lo`
exactly.

**Round-tripping s/z through fp16 before selecting `q` is mandatory.** Skipping
it makes encoder reconstruction disagree with the kernel.

v1 MQ6/5/3/2 used a single f32 scale+zero per 256 and the same payload
packing. V2 only changes the header geometry and the per-half fit.

---

## 5 · Decoder contract

```
hA = load_u32(group + 0)   // s0 | z0
hB = load_u32(group + 4)   // s1 | z1
hs = half0 ? hA : hB       // half from payload address, not a raw lane id
sc = f16_to_f32(hs & 0xFFFF)
zp = f16_to_f32(hs >> 16)
// unpack q from payload (§3)
w  = q * sc + zp
```

Mis-routing a V2 tensor to a v1 kernel (f32 scale/zero per 256 at the same
payload offset) decodes every weight near `~1e-14`. Dispatch fail-closes that
class of error in `crates/hipfire-dispatch/src/families/gemm.rs` (explicit
qt47–50 vs v1 / cross-V2 key guards). Equal group stride across bit widths is
**not** interchangeable either (e.g. MQ3V2 104 B vs MQ4V2 136 B).

---

## 6 · Support matrix (runtime)

| surface | qt47–50 |
|---|---|
| Dense GEMV prerotated / residual / SwiGLU residual | **yes** (wave32 predicate) |
| Dense fused QKV / QKVZA / gate_up | **yes** (dense gfx12 path) |
| Dense residual GEMM / plain GEMM keys | **yes, gfx12 WMMA only** (`gemm_table`: explicit V2 GEMM keys are gfx12-only) |
| Batched lm_head / DFlash verify rotate+GEMM | **yes** on dense qwen35 gfx12 (scratch size asserts per V2 dtype) |
| Prefill batchable LA | gfx1200/gfx1201 only (`prefill.rs` arch match includes all four V2 dtypes) |
| AWQ sidecar | **yes** — `DType::supports_awq_sidecar` lists MQ6/5/3/2G256V2 with the same pre-scale contract as qt=44 |
| MoE mixed-tier / routed expert path | **no** — `MIXED_SUPPORTED_TIERS` is only `{MQ4G256, MQ6G256, ParoQ4G128}`; V2 dtypes are absent and fail closed up front |
| Non-gfx12 batched prefill WMMA | **no** — fall back or refuse; do not invent a cross-arch WMMA route |
| Host RAW_CODECS length check | **yes** — blob must be `M*(K/256)*GROUP_BYTES` |

Parity / kernel gate: `crates/hipfire-runtime/examples/mqv2_family_parity.rs`
exercises all four formats for byte count, header divergence, sentinel, plain
GEMV B=1, and B=16 batched lm-head vs CPU dequant.

---

## 7 · Codec MSE (tool-level)

Measured on 278,528 real post-FWHT G256 blocks from Qwen3.8-27B layer-20
down-proj (`M=5120`, `K=17408`) with
`tools/quant-design/optimize_mq_lowbit_ladder.hip`. Each comparison keeps the
payload, header budget, group stride, and codec bpw fixed; only the header
changes from one G256 f32 affine pair to two G128 fp16 affine pairs.

| bits | bytes/G256 | v1 overall MSE | V2 overall MSE | Δ | v1 tail-1% MSE | V2 tail-1% MSE | Δ |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 72 | 3.6316e-05 | 3.0413e-05 | **−16.3%** | 1.5532e-05 | 9.3995e-06 | **−39.5%** |
| 3 | 104 | 6.6234e-06 | 5.5528e-06 | **−16.2%** | 4.3259e-06 | 2.5291e-06 | **−41.5%** |
| 4 | 136 | 1.4415e-06 | 1.2080e-06 | **−16.2%** | 9.4567e-07 | 5.6442e-07 | **−40.3%** |
| 5 | 168 | 3.3748e-07 | 2.8282e-07 | **−16.2%** | 2.2038e-07 | 1.3224e-07 | **−40.0%** |
| 6 | 200 | 8.1702e-08 | 6.8476e-08 | **−16.2%** | 5.3568e-08 | 3.2502e-08 | **−39.3%** |

The matched HIP microbench (`tools/quant-design/bench_mq_lowbit_kernels.hip`,
100 warmups, 1,000 interleaved samples, gfx1201) passed host/GPU sentinels for
all formats. MQ5V2 was throughput-neutral versus MQ5 v1; MQ6V2 was
throughput-neutral at B=1 and 2.4% faster in the B=16 shared-weight proxy.

Codec MSE is **not** a KLD proxy. Product quality is §8 / the ladder checkpoint.

---

## 8 · Product admission boundary (measured, not claimed shipping)

Evidence host: hiptrx, 4×gfx1201, ROCm/HIP 7.14. Fixture: Qwen3.8-27B,
26,895,998,464 params. Scoring: WT2 + v6sel, 24 chunks, prompt md5
`253c7ac50857fe6d0e10fb0d2c5e35c0`. Ladder summary generated
`2026-08-20T14:13:16Z`, commit `bfe3cfead5856c5ee85e972b609712b0badbd1b1`,
source `ssh://hiptrx/home/kaden/qcal/ladder-v2/summary.json`.

Acceptance rate from native generate is **null**
(`acceptance_reason: unavailable_in_native_generate_v1`).

### 8.1 · Measured KLD / AR (body format = cell `format`)

| cell | format | model bpw | bytes | md5 | WT2 KLD | v6sel KLD | AR dec/pre tok/s |
|---|---|---:|---:|---|---:|---:|---|
| mq2-xt | mq2v2 | 2.551 | 8574872576 | `12d3cafef619b552f78e9ae557b899ed` | **13.239513** | **14.202591** | 48.4 / 538.3 |
| mq2-base | mq2v2 | 2.848 | 9574976512 | `8693a1305eec5f77eed8db19bd49a94b` | **12.441964** | **13.428047** | 43.5 / 521.4 |
| mq2-pro | mq2v2 | 2.966 | 9971776512 | `4823a452391cb3cdc58616b327a74ed2` | **12.466792** | **13.447430** | 42.7 / 516.2 |
| mq3-xt | mq3v2 | 3.503 | 11777616896 | `80bb9198e6a565fc006b2ae1b7c89eca` | 0.248348 | 1.411523 | 40.7 / 468.9 |
| mq3-base | mq3v2 | 3.753 | 12618796032 | `ad20b3d3a9a7254e7a9c596fc97b411a` | 0.153658 | 1.032401 | 37.6 / 458.8 |
| mq3-pro | mq3v2 | 3.922 | 13184433152 | `b171dd618eecf1fe6aed2c1dc5eef4dc` | 0.130314 | 0.924635 | 36.7 / 453.9 |
| mq4-xt | mq4v2 | 4.456 | 14980361216 | `e45d15bfe0c9a87132697101d17cbed6` | 0.057449 | 0.771410 | 35.3 / 490.6 |
| mq4-base | mq4v2 | 4.659 | 15662615552 | `d1292b4d5bd6046693604201a6ca8074` | 0.039033 | 0.544517 | 33.2 / 480.7 |
| mq4-pro | mq4v2 | 4.897 | 16464182272 | `279c563786499c6651de6a3a57b42b02` | 0.032495 | 0.484145 | 31.9 / 474.7 |
| mq5-xt | mq5v2 | 5.408 | 18183105536 | `e7c17382812f33df90f5138eb7cf9973` | 0.015028 | 0.440984 | 29.4 / 345.5 |
| mq5-base | mq5v2 | 5.564 | 18706435072 | `cff5d051ce8a2a44a5eba6b9b43d595d` | 0.010255 | 0.278077 | 28.2 / 341.5 |
| mq5-pro | mq5v2 | 5.746 | 19319258112 | `50119af7b6974320e574774b663d2f6d` | 0.009006 | 0.237993 | 27.5 / 345.8 |
| mq6-xt | mq6v2 | 6.361 | 21385849856 | `de6eb059a10577b80f0a162e5c89249e` | 0.004389 | 0.220915 | 26.0 / 333.9 |
| mq6-base | mq6v2 | 6.469 | 21750254592 | `ae18a5d9e7926474064586387126aa5c` | 0.002771 | 0.152813 | 25.2 / 330.1 |
| mq6-pro | mq6v2 | 6.596 | 22174333952 | `aac76dcc771b424d605e813b0b4aa1c1` | 0.002208 | 0.136504 | 24.8 / 331.5 |

Artifact sha256 (full):

| cell | sha256 |
|---|---|
| mq2-xt | `b86de92dd07e3ac53e239459ef4f4881b798ba1296ce15a0974f59007a8e894e` |
| mq2-base | `547bc0ae3dfefaef583c4f8756511f362389e93765f2136fab64691b778ac186` |
| mq2-pro | `6cc00eddd39bbb06824905276bfa0fd34edeed40b3dd3020b7b7a0412b0fad1b` |
| mq3-xt | `3e04fc8db80bda557b965ec60ac876cf2500fced7f340624f3fcbeae134af5c5` |
| mq3-base | `09c3544690aceca29e1822d79adab6ffcc8fd9e4b58359fe8dfb185ef49811c9` |
| mq3-pro | `394c50966bf4f68172df8eb34cd7ded8f9d0576c9ef24ea6ba639a88c184f795` |
| mq4-xt | `9f91556f7e0431a077d03756a7102d0154108757289e6e5fe9a2d204c0c9eeb7` |
| mq4-base | `5bb556a6cc84035234995c017c9791aa3951ad1eae4cf8c8172b0eaef399e507` |
| mq4-pro | `e6f2ac87042b9e314c323f00bc499a6304cd5624021ae501ce90b12a3a7ea3fa` |
| mq5-xt | `f4760c159f80d5d3ca237593a191dd553790cba0f89de59827f430b23109b39b` |
| mq5-base | `c018a0d7510bffbb3788844d5d8f72694464244e16f128b7e34804045324cf25` |
| mq5-pro | `7a46204f5ce16b260ebb028359a945c8c02f5abe87c040df063db782a99ee7cd` |
| mq6-xt | `9d472ddc5b4e11a1986bfc83c54c0dac979a8e1d7186613dd7c7436f69ec8b2f` |
| mq6-base | `b798ea1166fc03a568f6daf8090b20b7d6314af7429951c03d86540385568db8` |
| mq6-pro | `58ac3ee645ede2bad2f6f833db1d4abbfdc1850047fdadef036d35a023bb9401` |

Fixed-tier overlays (product recipe, not wire):

- mq2-pro: `lm_head:mq6v2,ssm_out:mq6v2`
- mq3-pro: `ssm_out:mq6v2`
- all other cells: none

### 8.2 · MQ2V2 product rejection

MQ2V2 **codec math is valid** (encoder round-trips, kernels decode, parity
gate covers qt=50). Product quality is not:

- WT2/v6sel KLD ≈ 12–14 on every mq2 rung (three orders of magnitude above
  mq3+).
- Serve battery (`serve-mq2.json`): no attractor/empty flags, but outputs are
  coherent-yet-degraded / repetitive (`runaway` on the 5-prompt battery).
- Do **not** advertise MQ2V2 as a shipping body format. Wire support remains
  for research, drafts, and mechanical parity.

MQ3–MQ6 V2 serve batteries (`serve-mq3.json` … `serve-mq6.json`): coherent,
no attractor/empty. Product rung selection still belongs in `ladder.md`, not
here.

### 8.3 · DFlash same-bit vs mq4v2 control (decode tok/s medians)

| cell | same-bit draft | same median | mq4v2 control | chosen draft |
|---|---|---:|---:|---|
| mq2-xt | mq2v2 | **59.9** | 39.6 | **mq2v2** |
| mq2-base | mq2v2 | 45.9 | **63.3** | mq4v2 |
| mq2-pro | mq2v2 | 62.6 | **88.4** | mq4v2 |
| mq3-xt | mq3v2 | 236.7 | **264.9** | mq4v2 |
| mq3-base | mq3v2 | 225.4 | **251.1** | mq4v2 |
| mq3-pro | mq3v2 | 221.3 | **248.1** | mq4v2 |
| mq4-xt | mq4v2 | 251.9 | — (controller) | mq4v2 |
| mq4-base | mq4v2 | 263.3 | — | mq4v2 |
| mq4-pro | mq4v2 | 258.0 | — | mq4v2 |
| mq5-xt | mq5v2 | 221.3 | **224.7** | mq4v2 |
| mq5-base | mq5v2 | 214.4 | **217.8** | mq4v2 |
| mq5-pro | mq5v2 | 213.4 | **217.1** | mq4v2 |
| mq6-xt | mq6v2 | 207.4 | **211.5** | mq4v2 |
| mq6-base | mq6v2 | 202.6 | **206.7** | mq4v2 |
| mq6-pro | mq6v2 | 202.1 | **205.9** | mq4v2 |

Draft digests (from summary):

| draft_id | sha256 | bytes |
|---|---|---:|
| mq2v2 | `0545d189d9486b84b2e65070cf4920ce762a590b4ed8a17092662b69e004f649` | 760353792 |
| mq3v2 | `dd335fb2634c371d120077c2fe754643c91b27f32d0c32668278fe4b8f1ca319` | 984978432 |
| mq4v2 | `d0a74a232a0e2166d889f823e91e0fbf778d21dd9668d7de055cdecb065401bc` | 1209603072 |
| mq5v2 | `8a8d3daeaa3788743ef9aedfa1a6cd9961395ecbd88e9b5e2eb981e0506a861f` | 1434227712 |
| mq6v2 | `d190ef2faa953252ac40e9706cf2c6763d095f7817c2c7a0466792fd6dfa9015` | 1658852352 |

Median τ is present in the remote summary per cell; acceptance remains null.

---

## 9 · Name and coexistence with v1

| name | qt | notes |
|---|---:|---|
| `mq6v2` | 47 | v2 header; v1 payload; v1 remains qt=15 `MQ6G256` |
| `mq5v2` | 48 | v2 header; v1 payload; v1 remains qt=31 `MQ5G256` |
| `mq4v2` | 44 | see `mq4-v2.md`; v1 remains qt=13 |
| `mq3v2` | 49 | v2 header; v1 payload; v1 remains qt=17 `MQ3G256` |
| `mq2v2` | 50 | v2 header; v1 payload; v1 remains qt=18 `MQ2G256` |

Artifacts self-describe by qt. V2 never redefines a v1 number. Dedicated
kernel symbols / TUs (not a compile-time header macro on v1 sources) keep
byte consumption explicit per dtype.

---

## 10 · Implementer checklist

1. Header: four LE fp16s at +0/+2/+4/+6; payload at +8.
2. `GROUP_BYTES` and payload pack match §1–§3 exactly.
3. Encode with fp16 round-trip + degenerate half rule (§4).
4. Decode `w = q*sc + zp` with half from payload geometry (§5).
5. `K % 256 == 0`; blob length = `M*(K/256)*GROUP_BYTES`.
6. Register RAW_CODECS + `supports_awq_sidecar` for the dtype.
7. Route only to matching V2 kernel keys; refuse v1/cross-width keys.
8. Dense gfx12 GEMV/GEMM/DFlash only; MoE mixed path stays fail-closed.
9. Pass `mqv2_family_parity` before claiming wire readiness.
10. Never promote MQ2V2 as product-quality from wire readiness alone.

---

## 11 · References

- [`qt-register.txt`](qt-register.txt) — qt 47–50 passthrough rows
- [`mq4-v2.md`](mq4-v2.md) — qt=44/45 detailed sibling
- [`ladder.md`](ladder.md) — product names / rungs / honesty invariant
- `docs/perf-checkpoints/2026-08-20-qwen38-mq-v2-product-ladder.md` — immutable
  ladder checkpoint (peer-owned)
- `crates/rdna-compute/src/dispatch.rs` — group-byte constants, AWQ allow-list
- `crates/hipfire-quantize/src/quant_fwht.rs` — MQ6/5 V2 encoders
- `crates/hipfire-quantize/src/quant_mq.rs` — MQ3/2 V2 encoders
- `crates/hipfire-runtime/examples/mqv2_family_parity.rs` — parity gate
- `ssh://hiptrx/home/kaden/qcal/ladder-v2/summary.{json,md,csv}` — measured ladder
