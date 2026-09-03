# Escha-W2 Phase 1 (35B-A3B) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Serve `EschaLabs/Qwen3.6-35B-A3B-Escha-W2` in hipfire on gfx1151 by repacking its trellis codes verbatim into `.hfq` and decoding them to `Q8_0` resident weights at load, with escha's two H128 activation transforms running at runtime.

**Architecture:** Three layers, each gated against the layer above it. A pure-Rust CPU reference (`escha_ref`) ported from Escha's Apache-2.0 `ref.py` is the numerical oracle. A converter turns safetensors into `.hfq` with two new quant types whose code streams are byte-identical to the source. At load, one GPU kernel expands those codes to `Q8_0`; two more apply the input and output Hadamards at runtime, and every GEMV downstream is existing hipfire code.

**Tech Stack:** Rust (workspace crates `hipfire-quantize`, `hipfire-runtime`, `hipfire-dispatch`, `rdna-compute`, `hipfire-arch-qwen35`), HIP/ROCm 7.2.2 targeting gfx1151, `cargo test`.

**Design spec:** `docs/plans/escha-w2-port-design.md`. Read §1.1–§1.4 before Task 1.

## Global Constraints

- Branch `nw_escha_w2`, worktree `~/repos/hipfire-escha`, off `origin/master` @ `8cd15a62b`. Upstream is deliberately unset; push with `git push origin nw_escha_w2`.
- Quant type ids are **`ESCHA2T16 = 42`** and **`ESCHA3T16 = 43`**. The authoritative registry is `crates/hipfire-quantize/src/hfq.rs` — the `#[repr(u8)] enum QuantType` **and** its `from_u8`, which its own doc comment requires be kept in sync. Do not consult the stale partial enum in the `loop/gfx1151` checkout.
- `RS = 0.088388347648` exactly (`1/sqrt(128)`). `kernels/src/gemv_mq4g128.hip:116` already pins `0.0883883476f`.
- Codebook hash constants, exact: multiplier `0xCBAC1FED`, mask `0x8FFF8FFF`, xor `0x3B603B60`.
- **Every `f16(...)` in the escha contract is round-to-nearest-even.** Do NOT use `crate::float16::f32_to_f16` for it — that helper **truncates**, deliberately, to keep existing HFQ bytes stable (see its module doc). Truncating breaks the codebook: it misses published constants at states 3, 6 and 7. Use `escha_ref::f16_rne` (Task 2), which routes through `half::f16::from_f32`. Decoding with `crate::float16::f16_to_f32` is fine — only the encode direction differs. Found by TDD in Task 1; upstream `ref.py` states the contract outright: "numpy f16+f16 rounds RNE, like the GPU".
- **No codebook LUT in any kernel.** 65536 × f16 = 128 KB; gfx1151 has 64 KB LDS (`crates/rdna-compute/src/profiler.rs`, `lds_per_cu: 65536`). Decode inline.
- Required leaves are `escha_code`, `escha_rin`, `escha_rout`. Optional: `escha_s_in`, `escha_s_out`, `escha_config`, `bias`. Unknown `escha_*` leaves are a hard error. See spec §1.4.
- `K` comes from `code.shape[-1] / 16`. Never from `escha_config` (optional) and never from `layer_meta.bits` (self-inconsistent across releases).
- Build: `cargo build --release --workspace --all-targets --locked`. Never run bare `cargo fmt` — it rewrites the workspace and buries the change (`CLAUDE.md:109`).
- hipcc must be on PATH and the kernel cache must be single-toolchain, or you will chase attractor garbage that mimics a codec bug. See `hipfire_kernel_rebuild_gfx1151`.

## Scope

**This plan is Phase 1, 35B-A3B only.** Out of scope, each needing its own plan: Phase 2 fused decode+GEMV; the Qwen3.8-27B dense model (larger kernel surface, needs arch-5 bias slots, and cannot ship on the decode-at-load tier).

## File Structure

| File | Responsibility |
|---|---|
| `crates/hipfire-quantize/src/escha_ref.rs` (new) | CPU oracle: codebook, tile decode, H128, transforms, MoE block. No GPU, no hipfire deps. |
| `crates/hipfire-quantize/src/lib.rs` (modify) | Add `pub mod escha_ref;` |
| `crates/hipfire-quantize/tests/data/escha/` (new) | Vendored packed golden inputs + digest constants |
| `crates/hipfire-quantize/src/hfq.rs` (modify) | `ESCHA2T16 = 42`, `ESCHA3T16 = 43` in enum + `from_u8` |
| `crates/hipfire-quantize/src/pipeline_escha.rs` (new) | safetensors → `.hfq` converter |
| `crates/hipfire-quantize/src/main.rs` (modify) | `mod pipeline_escha;` + CLI arm |
| `crates/rdna-compute/src/dispatch.rs` (modify) | `DType::Escha2T16`, `DType::Escha3T16` |
| `crates/hipfire-dispatch/src/types.rs` (modify) | `RotationPlan::EschaH128` + `dtype_rotation_plan` arms |
| `kernels/src/escha_decode_tiles.hip` (new) | One-shot tile → `Q8_0` expansion |
| `kernels/src/escha_h128.hip` (new) | Input and output H128 transforms |
| `crates/rdna-compute/src/kernels.rs` (modify) | `include_str!` consts for both kernels |
| `crates/hipfire-arch-qwen35/src/` (modify) | Load escha experts, call the transforms |
| `registry/v1.json` (modify) | `qwen3.6:35b-a3b-escha` |

---

### Task 1: `escha_ref` codebook and tile decode

The trellis decode is the heart of the port. Everything else is gated against it, so it is built first and proven against Escha's own golden vectors.

**Files:**
- Create: `crates/hipfire-quantize/src/escha_ref.rs`
- Create: `crates/hipfire-quantize/tests/data/escha/fetch-goldens.sh`
- Modify: `crates/hipfire-quantize/src/lib.rs:3` (add module)
- Test: inline `#[cfg(test)]` in `escha_ref.rs` (this crate has no `tests/` dir; all tests are inline)

**Interfaces:**
- Consumes: `crate::float16::f16_to_f32`, `half::f16::from_f32` (RNE encode — see Global Constraints)
- Produces: `pub const RS: f32`, `pub fn cba_decode(state: u16) -> u16`, `pub fn decode8_k2(words: &[u32; 16], lane: usize) -> [u16; 8]`, `pub fn decode8_k3(words: &[u32; 24], lane: usize) -> [u16; 8]`, `pub fn lane_positions(lane: usize) -> [(usize, usize); 8]`, `pub fn reconstruct(code: &[i16], in_features: usize, out_features: usize, k: usize) -> Vec<u16>` (returns f16 **bits**, row-major `[in_features, out_features]`)

- [ ] **Step 1: Vendor the packed golden inputs**

Only the packed inputs are committed (0.9 MB). The expected outputs are 6.3 MB and are asserted by SHA-256 digest instead, so the repo stays light and the gate stays exact.

```bash
mkdir -p crates/hipfire-quantize/tests/data/escha
cd crates/hipfire-quantize/tests/data/escha
B=https://raw.githubusercontent.com/EschaLabs/escha-mlx/HEAD/tests/data/codec
curl -sL "$B/packed_gu_e0_k2.i16"   -o packed_gu_e0_k2.i16
curl -sL "$B/packed_down_e0_k3.i16" -o packed_down_e0_k3.i16
sha256sum packed_gu_e0_k2.i16 packed_down_e0_k3.i16
```

Expected output, exactly:

```
c164583731eed50d52ae3bfcc6a58a72b50329f8d4c40e5d62f940d67991ec1b  packed_gu_e0_k2.i16
aa2f2dd1f165d03ae868ce14913c05313876fa4d6c0e52d2ee16c60ff22eb062  packed_down_e0_k3.i16
```

If either digest differs, stop — upstream changed the fixtures and the constants in Step 3 are stale.

- [ ] **Step 2: Write the fetch script for the full goldens**

Create `crates/hipfire-quantize/tests/data/escha/fetch-goldens.sh`:

```bash
#!/usr/bin/env bash
# Fetch the full escha-mlx golden vectors (6.3 MB of expected outputs, not
# committed). Only needed to regenerate the digests in escha_ref.rs; the
# committed packed inputs plus those digests are a complete gate.
set -euo pipefail
cd "$(dirname "$0")"
B=https://raw.githubusercontent.com/EschaLabs/escha-mlx/HEAD/tests/data
for f in codec/packed_gu_e0_k2.i16 codec/expected_gu_e0_k2.f16 \
         codec/packed_down_e0_k3.i16 codec/expected_down_e0_k3.f16 \
         qwen3_5_moe/moeblk_x.f16 qwen3_5_moe/moeblk_out.f16 \
         qwen3_5_moe/moeblk_ids.i64 qwen3_5_moe/moeblk_scores.f32; do
  curl -sL --fail "$B/$f" -o "$(basename "$f")"
done
sha256sum ./*.f16 ./*.i16 ./*.i64 ./*.f32
```

```bash
chmod +x crates/hipfire-quantize/tests/data/escha/fetch-goldens.sh
```

- [ ] **Step 3: Write the failing tests**

Create `crates/hipfire-quantize/src/escha_ref.rs` containing **only** this test module for now:

```rust
//! Portable CPU reference for the escha codec — the numerical oracle for
//! every GPU kernel in this port.
//!
//! Ported from `escha_mlx/ref.py` (EschaLabs/escha-mlx, Apache-2.0), which
//! declares itself "the semantic contract for every Metal kernel in this
//! package". Rounding points are deliberate; do not "simplify" them.

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn data(name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/escha").join(name)
    }

    fn sha256_hex(bytes: &[u8]) -> String {
        use sha2::{Digest, Sha256};
        let mut h = Sha256::new();
        h.update(bytes);
        h.finalize().iter().map(|b| format!("{b:02x}")).collect()
    }

    /// The codebook is a pure function of the 16-bit state. These eight values
    /// were computed from the published constants and pin the hash, the
    /// masking, and the fp16 RNE add all at once.
    #[test]
    fn cba_decode_matches_published_constants() {
        let want: [u16; 8] =
            [0x3f60, 0x304e, 0xba13, 0x3ab8, 0x3952, 0xb75f, 0xbea4, 0xbc71];
        for (state, &bits) in want.iter().enumerate() {
            assert_eq!(cba_decode(state as u16), bits, "state {state}");
        }
    }

    #[test]
    fn reconstruct_k2_matches_golden() {
        let raw = std::fs::read(data("packed_gu_e0_k2.i16")).unwrap();
        let code: Vec<i16> =
            raw.chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]])).collect();
        assert_eq!(code.len(), 262144);
        let out = reconstruct(&code, 2048, 1024, 2);
        assert_eq!(out.len(), 2048 * 1024);
        let bytes: Vec<u8> = out.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(
            sha256_hex(&bytes),
            "51ddde9a07613aafcc9f5db79702349d19e18357d9fb910f48b38eeea028dcab",
            "decoded K=2 tensor does not match expected_gu_e0_k2.f16"
        );
    }

    #[test]
    fn reconstruct_k3_matches_golden() {
        let raw = std::fs::read(data("packed_down_e0_k3.i16")).unwrap();
        let code: Vec<i16> =
            raw.chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]])).collect();
        assert_eq!(code.len(), 196608);
        let out = reconstruct(&code, 512, 2048, 3);
        assert_eq!(out.len(), 512 * 2048);
        let bytes: Vec<u8> = out.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(
            sha256_hex(&bytes),
            "51c99817d00282f4aa9d618140eb4503083e1238cb59dd71a670aa4c320f7438",
            "decoded K=3 tensor does not match expected_down_e0_k3.f16"
        );
    }

    /// Every one of the 256 tile slots must be written exactly once. A
    /// permutation bug that drops and duplicates slots still produces a
    /// full-rank, plausible-looking matrix, so check the permutation directly.
    #[test]
    fn lane_positions_is_a_permutation_of_the_tile() {
        let mut seen = [0u8; 256];
        for lane in 0..32 {
            for (r, c) in lane_positions(lane) {
                assert!(r < 16 && c < 16, "lane {lane} -> ({r},{c})");
                seen[r * 16 + c] += 1;
            }
        }
        assert!(seen.iter().all(|&n| n == 1), "lane_positions is not a bijection");
    }
}
```

Add to `crates/hipfire-quantize/src/lib.rs` after line 3 (`pub mod float16;`):

```rust
pub mod escha_ref;
```

Add the test-only digest dependency to `crates/hipfire-quantize/Cargo.toml` under `[dev-dependencies]` (create the section if absent):

```toml
[dev-dependencies]
sha2 = "0.10"
```

- [ ] **Step 4: Run the tests to verify they fail**

```bash
cargo test -p hipfire-quantize --lib escha_ref 2>&1 | tail -20
```

Expected: compile error, `cannot find function 'cba_decode' in this scope` (and the same for `reconstruct`, `lane_positions`). A compile failure is the correct "red" here — the functions do not exist yet.

- [ ] **Step 5: Implement the codec**

Insert above the `#[cfg(test)] mod tests` block in `escha_ref.rs`:

```rust
use crate::float16::f16_to_f32;

/// 1/sqrt(128) — the exact f32 constant the format pins.
pub const RS: f32 = 0.088388347648;

/// Round f32 to fp16 bits, round-to-nearest-even.
///
/// Every `f16(...)` in the escha contract is RNE. `crate::float16::f32_to_f16`
/// TRUNCATES — deliberately, to keep existing HFQ bytes stable — and using it
/// here silently corrupts the codec: it misses the published cbA constants at
/// states 3, 6 and 7. Do not "simplify" this back to the crate helper.
#[inline]
pub fn f16_rne(v: f32) -> u16 {
    half::f16::from_f32(v).to_bits()
}

/// Decode one 16-bit trellis state to fp16 **bits** via the cbA codebook.
///
/// `decode(x) = f16_lo(r) + f16_hi(r)` with fp16 RNE addition, where
/// `r = ((x * 0xCBAC1FED) & 0x8FFF8FFF) ^ 0x3B603B60` in 32-bit arithmetic.
///
/// Adding in f32 and rounding once is exactly an fp16 RNE add: the exact sum
/// of two fp16 values is always representable in f32, so the single rounding
/// here is the correctly-rounded fp16 result.
///
/// The round MUST be `half::f16::from_f32`, not `crate::float16::f32_to_f16`
/// — the latter truncates by design and misses states 3, 6 and 7.
///
/// There are 65536 reachable values, so a lookup table would be 128 KB and
/// will not fit gfx1151's 64 KB LDS. This is five integer/FP ops and no
/// memory traffic — keep it that way in the kernels.
#[inline]
pub fn cba_decode(state: u16) -> u16 {
    let r = ((state as u32).wrapping_mul(0xCBAC_1FED) & 0x8FFF_8FFF) ^ 0x3B60_3B60;
    let lo = f16_to_f32((r & 0xFFFF) as u16);
    let hi = f16_to_f32((r >> 16) as u16);
    half::f16::from_f32(lo + hi).to_bits()
}

/// The 8 states lane `lane` owns, K=2. `words` is the tile's 16 u32.
///
/// DELIBERATE DUPLICATION: `kernels/src/escha_decode_tiles.hip` implements
/// this same lane maths independently. That is the G2 gate — the GPU decode
/// is asserted bit-exact against this one. Generating either from the other,
/// or sharing a source, would make G2 circular: both paths could be wrong in
/// exactly the same way and still agree. Two independent implementations of
/// a published spec is the point. Do not deduplicate.
pub fn decode8_k2(words: &[u32; 16], lane: usize) -> [u16; 8] {
    let t_off = lane * 8;
    let i1 = t_off >> 4;
    let i0 = (i1 + 15) & 15;
    let merged = ((words[i0] as u64) << 32) | words[i1] as u64;
    let shift = ((!t_off) & 8) << 1; // 16 for even lanes, 0 for odd
    let w = ((merged >> shift) & 0xFFFF_FFFF) as u32;
    let mut out = [0u16; 8];
    for (j, o) in out.iter_mut().enumerate() {
        *o = (w >> (2 * (7 - j))) as u16;
    }
    out
}

/// The 8 states lane `lane` owns, K=3. `words` is the tile's 24 u32.
///
/// Structurally different from K=2 — 24 words, a computed bit offset, and a
/// modular wrap. Do not attempt to unify the two.
pub fn decode8_k3(words: &[u32; 24], lane: usize) -> [u16; 8] {
    const BITS: usize = 3;
    let t_off = lane * 8;
    let b1 = (t_off + 257) * BITS;
    let b0 = b1 - 16;
    let b2 = b1 + BITS * 7;
    let i0 = b0 >> 5;
    let i2 = (b2 - 1) >> 5;
    let s2 = ((i2 + 1) << 5) - b2;
    let merged = ((words[i0 % 24] as u64) << 32) | words[i2 % 24] as u64;
    let w7 = (merged >> s2) & 0xFFFF_FFFF;
    let w3 = (merged >> (s2 + BITS * 4)) & 0xFFFF_FFFF;
    [
        (w3 >> 9) as u16,
        (w3 >> 6) as u16,
        (w3 >> 3) as u16,
        w3 as u16,
        (w7 >> 9) as u16,
        (w7 >> 6) as u16,
        (w7 >> 3) as u16,
        w7 as u16,
    ]
}

/// `(row, col)` inside the 16x16 tile for each of the lane's 8 values.
///
/// This permutation is the single easiest thing to get subtly wrong: a wrong
/// shuffle still yields a full-rank, plausible weight matrix. It is gated
/// directly on golden vectors, never on end-to-end coherence.
pub fn lane_positions(lane: usize) -> [(usize, usize); 8] {
    let l0 = lane & !4;
    let c_off = (lane >> 2) & 1;
    let mut out = [(0usize, 0usize); 8];
    for (j, o) in out.iter_mut().enumerate() {
        let fi = j >> 1;
        let row = (lane & 3) * 2 + (j & 1) + (fi & 1) * 8;
        let col = 2 * ((l0 >> 3) + if j >= 4 { 4 } else { 0 }) + c_off;
        *o = (row, col);
    }
    out
}

/// Decode one packed tile (`16*K` i16) to a 16x16 fp16-bit tile, row-major.
pub fn decode_tile(tile: &[i16], k: usize) -> [u16; 256] {
    debug_assert_eq!(tile.len(), 16 * k);
    let mut words = [0u32; 24];
    for (i, w) in words.iter_mut().enumerate().take(8 * k) {
        *w = (tile[2 * i] as u16 as u32) | ((tile[2 * i + 1] as u16 as u32) << 16);
    }
    let mut out = [0u16; 256];
    for lane in 0..32 {
        let states = match k {
            2 => {
                let mut w16 = [0u32; 16];
                w16.copy_from_slice(&words[..16]);
                decode8_k2(&w16, lane)
            }
            3 => decode8_k3(&words, lane),
            _ => panic!("unsupported escha K={k}"),
        };
        for (j, (r, c)) in lane_positions(lane).into_iter().enumerate() {
            out[r * 16 + c] = cba_decode(states[j]);
        }
    }
    out
}

/// Packed `(in/16, out/16, 16K)` i16 -> `(in, out)` fp16 bits, row-major.
pub fn reconstruct(code: &[i16], in_features: usize, out_features: usize, k: usize) -> Vec<u16> {
    let (tk, tn) = (in_features / 16, out_features / 16);
    assert_eq!(code.len(), tk * tn * 16 * k, "escha code length mismatch");
    let mut out = vec![0u16; in_features * out_features];
    for kt in 0..tk {
        for nt in 0..tn {
            let base = (kt * tn + nt) * 16 * k;
            let tile = decode_tile(&code[base..base + 16 * k], k);
            for r in 0..16 {
                let dst = (kt * 16 + r) * out_features + nt * 16;
                out[dst..dst + 16].copy_from_slice(&tile[r * 16..r * 16 + 16]);
            }
        }
    }
    out
}
```

- [ ] **Step 6: Run the tests to verify they pass**

```bash
cargo test -p hipfire-quantize --lib escha_ref 2>&1 | tail -15
```

Expected: `test result: ok. 4 passed; 0 failed`.

If `reconstruct_k2_matches_golden` fails while `cba_decode_matches_published_constants` passes, the bug is in `lane_positions` or the word packing, not the codebook — check `lane_positions_is_a_permutation_of_the_tile` first.

- [ ] **Step 7: Commit**

```bash
git add crates/hipfire-quantize/src/escha_ref.rs \
        crates/hipfire-quantize/src/lib.rs \
        crates/hipfire-quantize/Cargo.toml \
        crates/hipfire-quantize/tests/data/escha/
git commit -m "feat(escha): trellis codec CPU reference, gated on golden vectors"
```

---

### Task 2: `escha_ref` H128 and the transform pair

**Files:**
- Modify: `crates/hipfire-quantize/src/escha_ref.rs`

**Interfaces:**
- Consumes: `RS` from Task 1
- Produces: `pub fn f16_rne(v: f32) -> u16`, `pub fn h128_inplace(x: &mut [f32])`, `pub fn input_transform(x: &[f32], rin: &[f32]) -> Vec<u16>`, `pub fn output_transform(mid: &[f32], rout: &[f32]) -> Vec<u16>`, `pub fn fold_scales(rin: &[u16], rout: &[u16], s_in: Option<&[f32]>, s_out: Option<&[f32]>) -> (Vec<f32>, Vec<f32>)`
- Note: Task 1 inlined `half::f16::from_f32(..).to_bits()` in `cba_decode`. Extract that into `f16_rne` here and have `cba_decode` call it, so there is exactly one RNE encode site in the module.

- [ ] **Step 1: Write the failing tests**

Add inside the existing `mod tests` block in `escha_ref.rs`:

```rust
    /// H128 is its own inverse up to a factor of 128. That is necessary but
    /// NOT sufficient: a wrong butterfly order is also self-inverse and would
    /// pass this alone. The Hadamard-of-a-basis-vector check below pins the
    /// actual transform.
    #[test]
    fn h128_roundtrip_scales_by_128() {
        let mut x: Vec<f32> = (0..256).map(|i| (i as f32 * 0.37).sin()).collect();
        let orig = x.clone();
        h128_inplace(&mut x);
        h128_inplace(&mut x);
        for (a, b) in x.iter().zip(orig.iter()) {
            assert!((a - b * 128.0).abs() < 1e-2, "{a} vs {}", b * 128.0);
        }
    }

    /// H128 applied to e_0 must give all ones (Sylvester, unnormalised).
    /// Applied to e_1 it must give the alternating +1/-1 pattern of row 1.
    #[test]
    fn h128_matches_sylvester_order() {
        let mut e0 = vec![0.0f32; 128];
        e0[0] = 1.0;
        h128_inplace(&mut e0);
        assert!(e0.iter().all(|&v| v == 1.0), "row 0 must be all ones");

        let mut e1 = vec![0.0f32; 128];
        e1[1] = 1.0;
        h128_inplace(&mut e1);
        for (i, &v) in e1.iter().enumerate() {
            let want = if i % 2 == 0 { 1.0 } else { -1.0 };
            assert_eq!(v, want, "index {i}");
        }
    }

    /// Blocks are independent: H128 must never mix across a 128 boundary.
    #[test]
    fn h128_does_not_mix_across_blocks() {
        let mut x = vec![0.0f32; 256];
        x[0] = 1.0;
        h128_inplace(&mut x);
        assert!(x[..128].iter().all(|&v| v == 1.0));
        assert!(x[128..].iter().all(|&v| v == 0.0), "second block was contaminated");
    }

    /// MoE exports ship all-ones s_in/s_out; dense exports ship real values;
    /// and an export without the end-to-end stage ships neither. All three
    /// must go through one code path.
    #[test]
    fn fold_scales_handles_absent_scales() {
        let rin = [f16_rne(2.0), f16_rne(-3.0)];
        let rout = [f16_rne(0.5)];
        let (a, b) = fold_scales(&rin, &rout, None, None);
        assert_eq!(a, vec![2.0, -3.0]);
        assert_eq!(b, vec![0.5]);
        let (c, d) = fold_scales(&rin, &rout, Some(&[3.0, 2.0]), Some(&[4.0]));
        assert_eq!(c, vec![6.0, -6.0]);
        assert_eq!(d, vec![2.0]);
    }
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
cargo test -p hipfire-quantize --lib escha_ref 2>&1 | tail -15
```

Expected: compile error, `cannot find function 'h128_inplace' in this scope`.

- [ ] **Step 3: Implement the transforms**

Append to the non-test portion of `escha_ref.rs`:

```rust
/// Unnormalised 128-point Walsh-Hadamard (Sylvester / natural order), applied
/// independently to each contiguous 128-element block of `x`.
///
/// `x.len()` must be a multiple of 128. Every dimension in both checkpoints
/// satisfies this (512, 1024, 2048, 5120, 6144, 10240, 17408 are all
/// multiples of 128), including the gate|up split point at 512 — so a block
/// never straddles the gate/up boundary.
pub fn h128_inplace(x: &mut [f32]) {
    assert_eq!(x.len() % 128, 0, "H128 needs a multiple of 128, got {}", x.len());
    for block in x.chunks_exact_mut(128) {
        let mut h = 1;
        while h < 128 {
            let mut i = 0;
            while i < 128 {
                for j in i..i + h {
                    let (a, b) = (block[j], block[j + h]);
                    block[j] = a + b;
                    block[j + h] = a - b;
                }
                i += 2 * h;
            }
            h *= 2;
        }
    }
}

/// `xh = f16( H128(x * rin) * RS )`. Returns fp16 bits.
pub fn input_transform(x: &[f32], rin: &[f32]) -> Vec<u16> {
    let ic = rin.len();
    assert_eq!(x.len() % ic, 0);
    let mut buf: Vec<f32> = x.iter().zip(rin.iter().cycle()).map(|(a, b)| a * b).collect();
    for row in buf.chunks_exact_mut(ic) {
        h128_inplace(row);
    }
    buf.iter().map(|v| f16_rne(v * RS)).collect()
}

/// `y = f16( H128(mid) * RS * rout )`. Returns fp16 bits.
pub fn output_transform(mid: &[f32], rout: &[f32]) -> Vec<u16> {
    let oc = rout.len();
    assert_eq!(mid.len() % oc, 0);
    let mut buf = mid.to_vec();
    for row in buf.chunks_exact_mut(oc) {
        h128_inplace(row);
    }
    buf.iter()
        .zip(rout.iter().cycle())
        .map(|(v, s)| f16_rne(v * RS * s))
        .collect()
}

/// Fold the optional end-to-end scales into the transform vectors.
///
/// `s_in` multiplies the activation at exactly the point `rin` does, and
/// `s_out` at exactly the point `rout` does, so the pair collapses with no new
/// kernel and no new tensor. Folding keeps both products in f32 and rounds
/// once — one rounding point FEWER than applying the scales separately.
/// `None` returns that vector unchanged (as f32), which is the path MoE
/// exports and end-to-end-free exports both take.
pub fn fold_scales(
    rin: &[u16],
    rout: &[u16],
    s_in: Option<&[f32]>,
    s_out: Option<&[f32]>,
) -> (Vec<f32>, Vec<f32>) {
    let mut ri: Vec<f32> = rin.iter().map(|&b| f16_to_f32(b)).collect();
    let mut ro: Vec<f32> = rout.iter().map(|&b| f16_to_f32(b)).collect();
    if let Some(s) = s_in {
        assert_eq!(s.len(), ri.len());
        for (a, b) in ri.iter_mut().zip(s) {
            *a *= b;
        }
    }
    if let Some(s) = s_out {
        assert_eq!(s.len(), ro.len());
        for (a, b) in ro.iter_mut().zip(s) {
            *a *= b;
        }
    }
    (ri, ro)
}
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
cargo test -p hipfire-quantize --lib escha_ref 2>&1 | tail -15
```

Expected: `test result: ok. 8 passed; 0 failed`.

- [ ] **Step 5: Commit**

```bash
git add crates/hipfire-quantize/src/escha_ref.rs
git commit -m "feat(escha): H128 transforms and scale folding in the CPU reference"
```

---

### Task 3: `escha_ref` expert linear, SwiGLU and w8a16

Completes the oracle. `expert_linear` is what Task 7's GPU decode is gated against, and `swiglu`/`w8a16` pin the two rounding points the Task 10 wiring must reproduce.

No `moe_block` port is needed: Task 10's G4 gate compares hipfire directly against Escha's shipped `moeblk_out.f16` golden, so a Rust reimplementation of the block would add a second thing to keep in sync without gating anything.

**Files:**
- Modify: `crates/hipfire-quantize/src/escha_ref.rs`

**Interfaces:**
- Consumes: `reconstruct`, `input_transform`, `output_transform` from Tasks 1–2
- Produces: `pub fn expert_linear(x: &[f32], w_bits: &[u16], rin: &[f32], rout: &[f32]) -> Vec<u16>`, `pub fn swiglu(gate_up_bits: &[u16], inter: usize) -> Vec<u16>`, `pub fn w8a16(x: &[f32], w8: &[i8], scale: &[u16], oc: usize, ic: usize) -> Vec<u16>`

- [ ] **Step 1: Write the failing test**

Add inside `mod tests`:

```rust
    /// A pruned output channel must be EXACTLY zero, not approximately.
    /// gate_up.rout carries a per-expert prune mask (design §1.2): on the
    /// shipped layer-0 expert 0, 560 of 1024 channels are hard zeros. A kernel
    /// that "optimises away" the zero multiply must preserve exact zero.
    #[test]
    fn zero_rout_gives_exactly_zero_output() {
        let ic = 128;
        let oc = 128;
        let w: Vec<u16> = (0..ic * oc).map(|i| f16_rne((i % 7) as f32 - 3.0)).collect();
        let rin = vec![1.0f32; ic];
        let mut rout = vec![1.0f32; oc];
        rout[3] = 0.0;
        rout[57] = 0.0;
        let x: Vec<f32> = (0..ic).map(|i| (i as f32 * 0.11).cos()).collect();
        let y = expert_linear(&x, &w, &rin, &rout);
        assert_eq!(y.len(), oc);
        assert_eq!(f16_to_f32(y[3]), 0.0, "pruned channel 3 must be exactly zero");
        assert_eq!(f16_to_f32(y[57]), 0.0, "pruned channel 57 must be exactly zero");
        assert!(y.iter().enumerate().any(|(i, &v)| i != 3 && i != 57 && f16_to_f32(v) != 0.0));
    }

    /// SwiGLU splits the f16-ROUNDED merged output at the halfway point, gate
    /// first. Rounding before the split is part of the contract.
    #[test]
    fn swiglu_uses_gate_first_half() {
        let inter = 2;
        // gate = [0, 0], up = [5, 7]; silu(0) == 0 so both outputs are zero.
        let gu: Vec<u16> = [0.0, 0.0, 5.0, 7.0].iter().map(|&v| f16_rne(v)).collect();
        let h = swiglu(&gu, inter);
        assert_eq!(h.len(), inter);
        assert_eq!(f16_to_f32(h[0]), 0.0);
        assert_eq!(f16_to_f32(h[1]), 0.0);
        // gate = [large, large] -> silu(x) ~ x, so out ~ gate*up.
        let gu2: Vec<u16> = [10.0, 10.0, 2.0, 3.0].iter().map(|&v| f16_rne(v)).collect();
        let h2 = swiglu(&gu2, inter);
        assert!((f16_to_f32(h2[0]) - 20.0).abs() < 0.1, "{}", f16_to_f32(h2[0]));
        assert!((f16_to_f32(h2[1]) - 30.0).abs() < 0.2, "{}", f16_to_f32(h2[1]));
    }

    /// Escha's int8 is per-output-ROW: y = f16(x @ f16(w8*scale)^T).
    #[test]
    fn w8a16_applies_per_row_scale() {
        let (ic, oc) = (4, 2);
        let w8: Vec<i8> = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let scale: Vec<u16> = vec![f16_rne(0.5), f16_rne(2.0)];
        let x = vec![1.0f32, 1.0, 1.0, 1.0];
        let y = w8a16(&x, &w8, &scale, oc, ic);
        assert_eq!(f16_to_f32(y[0]), 5.0); // (1+2+3+4)*0.5
        assert_eq!(f16_to_f32(y[1]), 52.0); // (5+6+7+8)*2.0
    }
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
cargo test -p hipfire-quantize --lib escha_ref 2>&1 | tail -15
```

Expected: compile error, `cannot find function 'expert_linear' in this scope`.

- [ ] **Step 3: Implement**

Append to the non-test portion of `escha_ref.rs`:

```rust
/// Full single-expert linear for one token: `x [ic] -> [oc]` fp16 bits.
///
/// `w_bits` is the decoded bare weight, row-major `[ic, oc]` — decode it once
/// with `reconstruct` and reuse it. `ref.moe_block` in the Python original
/// re-decodes per (token, slot), which is 128 full tile decodes for an
/// 8-token fixture; do not reproduce that.
pub fn expert_linear(x: &[f32], w_bits: &[u16], rin: &[f32], rout: &[f32]) -> Vec<u16> {
    let (ic, oc) = (rin.len(), rout.len());
    assert_eq!(x.len(), ic);
    assert_eq!(w_bits.len(), ic * oc);
    let xh = input_transform(x, rin);
    let mut mid = vec![0.0f32; oc];
    for i in 0..ic {
        // Unconditional MAC — no zero-activation skip. This is the oracle, so
        // it must be a faithful matmul: 0.0 * NaN is NaN, not 0, and skipping
        // would MASK a corrupted decode instead of surfacing it.
        let a = f16_to_f32(xh[i]);
        let row = &w_bits[i * oc..(i + 1) * oc];
        for (m, &wb) in mid.iter_mut().zip(row) {
            *m += a * f16_to_f32(wb);
        }
    }
    output_transform(&mid, rout)
}

/// `silu(g) * u` on the fp16-rounded merged output; gate is the first half.
pub fn swiglu(gate_up_bits: &[u16], inter: usize) -> Vec<u16> {
    assert_eq!(gate_up_bits.len(), 2 * inter);
    let mut out = Vec::with_capacity(inter);
    for i in 0..inter {
        let g = f16_to_f32(gate_up_bits[i]);
        let s = f16_to_f32(f16_rne(g / (1.0 + (-g).exp())));
        out.push(f16_rne(s * f16_to_f32(gate_up_bits[inter + i])));
    }
    out
}

/// `y = f16( x @ f16(w8 * scale)^T )`. `w8` is `[oc, ic]`, `scale` is `[oc]`
/// fp16 bits — Escha's int8 is per-output-row, not per-block.
pub fn w8a16(x: &[f32], w8: &[i8], scale: &[u16], oc: usize, ic: usize) -> Vec<u16> {
    assert_eq!(w8.len(), oc * ic);
    assert_eq!(scale.len(), oc);
    assert_eq!(x.len(), ic);
    let mut out = Vec::with_capacity(oc);
    for o in 0..oc {
        let s = f16_to_f32(scale[o]);
        let mut acc = 0.0f32;
        for i in 0..ic {
            acc += x[i] * f16_to_f32(f16_rne(w8[o * ic + i] as f32 * s));
        }
        out.push(f16_rne(acc));
    }
    out
}
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
cargo test -p hipfire-quantize --lib escha_ref 2>&1 | tail -15
```

Expected: `test result: ok. 11 passed; 0 failed`.

- [ ] **Step 5: Commit**

```bash
git add crates/hipfire-quantize/src/escha_ref.rs
git commit -m "feat(escha): expert linear, SwiGLU and w8a16 in the CPU reference"
```

---

### Task 4: Register the quant types and the rotation plan

Two ids, one rotation plan, and the guard that stops them silently falling through to an unrotated kernel. That fallthrough is the highest-severity failure mode in the port: it produces coherent-looking text rather than a crash.

**Files:**
- Modify: `crates/hipfire-quantize/src/hfq.rs` (enum + `from_u8`)
- Modify: `crates/rdna-compute/src/dispatch.rs:285-310` (`DType`)
- Modify: `crates/hipfire-dispatch/src/types.rs:107-130` (`RotationPlan`, `dtype_rotation_plan`)
- Test: `crates/hipfire-dispatch-tests/src/dtype.rs`

**Interfaces:**
- Produces: `hfq::QuantType::{ESCHA2T16, ESCHA3T16}` (bytes 42, 43), `DType::{Escha2T16, Escha3T16}`, `RotationPlan::EschaH128`

- [ ] **Step 1: Write the failing tests**

Add to `crates/hipfire-dispatch-tests/src/dtype.rs`:

```rust
#[test]
fn escha_types_use_the_escha_rotation_plan() {
    assert_eq!(dtype_rotation_plan(DType::Escha2T16), RotationPlan::EschaH128);
    assert_eq!(dtype_rotation_plan(DType::Escha3T16), RotationPlan::EschaH128);
}

/// Escha weights are stored in the rotated domain. Reaching a Plain GEMV
/// without the H128 pair does not crash — it produces coherent-looking
/// garbage. Both types must therefore refuse to resolve to Plain, exactly as
/// MQ4G128 does (see coverage_tests.rs).
#[test]
fn escha_types_never_resolve_to_plain() {
    for dt in [DType::Escha2T16, DType::Escha3T16] {
        assert!(
            KernelKey::for_gemv(dt, GemvVariant::Plain, false).is_err(),
            "{dt:?} must not have a Plain GEMV arm — that would skip the H128 pair"
        );
    }
}
```

Add to `crates/hipfire-quantize/src/hfq.rs`, inside its existing `#[cfg(test)] mod tests`:

```rust
    /// from_u8 and the enum discriminants must agree — the doc comment on
    /// from_u8 makes this a contract, and a drifted pair silently mislabels
    /// every tensor written after it.
    #[test]
    fn escha_quant_types_round_trip() {
        assert_eq!(QuantType::from_u8(42), Some(QuantType::ESCHA2T16));
        assert_eq!(QuantType::from_u8(43), Some(QuantType::ESCHA3T16));
        assert_eq!(QuantType::ESCHA2T16 as u8, 42);
        assert_eq!(QuantType::ESCHA3T16 as u8, 43);
    }
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
cargo test -p hipfire-quantize --lib hfq::tests::escha 2>&1 | tail -10
cargo test -p hipfire-dispatch-tests escha 2>&1 | tail -10
```

Expected: compile errors — `no variant named 'ESCHA2T16'`, `no variant or associated item named 'Escha2T16'`.

- [ ] **Step 3: Add the quant types**

In `crates/hipfire-quantize/src/hfq.rs`, add to the `#[repr(u8)] enum QuantType` after `MQ2G256LloydU`:

```rust
    /// Escha-W2 trellis, K=2, 16x16 tile, cbA hash codebook (2.00 bpw).
    /// Codes are stored verbatim from the source safetensors.
    ESCHA2T16 = 42,
    /// Escha-W2 trellis, K=3, 16x16 tile, cbA hash codebook (3.00 bpw).
    ESCHA3T16 = 43,
```

And in `from_u8`, before the `_ => None` arm:

```rust
            42 => Some(Self::ESCHA2T16),
            43 => Some(Self::ESCHA3T16),
```

In `crates/rdna-compute/src/dispatch.rs`, add to `enum DType` after `MQ4G256V2`:

```rust
    Escha2T16,
    Escha3T16,
```

In `crates/hipfire-dispatch/src/types.rs`, add to `enum RotationPlan` after `Givens`:

```rust
    /// Escha-W2: unnormalised 128-point Walsh-Hadamard on BOTH sides,
    /// RS = 1/sqrt(128), signs folded into rin/rout rather than seeded.
    EschaH128,
```

And add an arm to `dtype_rotation_plan`, before the `_ => RotationPlan::None` catch-all:

```rust
        Escha2T16 | Escha3T16 => RotationPlan::EschaH128,
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
cargo test -p hipfire-quantize --lib hfq::tests::escha 2>&1 | tail -10
cargo test -p hipfire-dispatch-tests escha 2>&1 | tail -10
```

Expected: both `test result: ok`.

If `escha_types_never_resolve_to_plain` fails, a `_ =>` catch-all in `KernelKey::for_gemv` is swallowing the new types. Find it and make the escha arms explicit errors — do not leave the catch-all to handle them.

- [ ] **Step 5: Verify the whole workspace still builds**

```bash
cargo build --release --workspace --all-targets --locked 2>&1 | tail -20
```

Expected: no errors. Non-exhaustive-match errors elsewhere are the point — every site that matches on `DType` or `RotationPlan` now has to say what it does with escha. Add explicit `Escha2T16 | Escha3T16 => Err(...)` arms rather than folding them into existing catch-alls.

- [ ] **Step 6: Commit**

```bash
git add crates/hipfire-quantize/src/hfq.rs crates/rdna-compute/src/dispatch.rs \
        crates/hipfire-dispatch/src/types.rs crates/hipfire-dispatch-tests/src/dtype.rs
git commit -m "feat(escha): register ESCHA2T16/ESCHA3T16 and the EschaH128 rotation plan"
```

---

### Task 5: Converter — safetensors to `.hfq`

**Files:**
- Create: `crates/hipfire-quantize/src/pipeline_escha.rs`
- Modify: `crates/hipfire-quantize/src/main.rs` (add `mod pipeline_escha;` beside the other `pipeline_*` modules, and a CLI arm)

**Interfaces:**
- Consumes: `crate::hfq::{HfqTensor, QuantType, write_hfq}`, `crate::safetensors_file::SafetensorsFile`, `crate::escha_ref::fold_scales`
- Produces: `pub(crate) fn convert_escha(src_dir: &Path, out: &Path) -> Result<(), String>`, `pub(crate) fn classify_leaf(name: &str) -> Leaf`, `pub(crate) enum Leaf { Code, Rin, Rout, SIn, SOut, Config, Bias, Int8, Int8Scale, Passthrough, UnknownEscha }`

- [ ] **Step 1: Write the failing tests**

Create `crates/hipfire-quantize/src/pipeline_escha.rs` with only this test module:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::escha_ref::f16_rne;

    #[test]
    fn k_comes_from_the_code_shape_not_metadata() {
        // gate_up: [E, in/16, out/16, 16K] with K=2 -> last dim 32
        assert_eq!(k_from_code_shape(&[256, 128, 64, 32]), Ok(2));
        // down: K=3 -> last dim 48
        assert_eq!(k_from_code_shape(&[256, 32, 128, 48]), Ok(3));
        // dense exports have no E axis
        assert_eq!(k_from_code_shape(&[320, 1088, 32]), Ok(2));
        assert!(k_from_code_shape(&[256, 128, 64, 33]).is_err());
    }

    #[test]
    fn quant_type_follows_k() {
        assert_eq!(quant_type_for_k(2), Ok(QuantType::ESCHA2T16));
        assert_eq!(quant_type_for_k(3), Ok(QuantType::ESCHA3T16));
        assert!(quant_type_for_k(4).is_err());
    }

    /// `ignore` means "not escha-coded", NOT "not quantized" — both models
    /// list embed_tokens and lm_head there and ship them as weight_int8.
    /// Classification must key off the tensor suffix actually present.
    #[test]
    fn classify_leaf_keys_off_the_suffix() {
        assert_eq!(classify_leaf("l.0.mlp.experts.gate_up_proj.escha_code"), Leaf::Code);
        assert_eq!(classify_leaf("l.0.mlp.experts.gate_up_proj.escha_rin"), Leaf::Rin);
        assert_eq!(classify_leaf("l.0.mlp.experts.gate_up_proj.escha_s_out"), Leaf::SOut);
        assert_eq!(classify_leaf("lm_head.weight_int8"), Leaf::Int8);
        assert_eq!(classify_leaf("lm_head.weight_scale"), Leaf::Int8Scale);
        assert_eq!(classify_leaf("l.0.input_layernorm.weight"), Leaf::Passthrough);
    }

    /// A future export carrying a rotation variant this version does not
    /// implement must stop conversion, not decode under the wrong rotation.
    #[test]
    fn unknown_escha_leaf_is_rejected() {
        assert_eq!(
            classify_leaf("l.0.mlp.gate_proj.escha_rotation_theta"),
            Leaf::UnknownEscha
        );
    }

    /// Required: code, rin, rout. Missing any is "incomplete escha linear" —
    /// fail loudly at load, never a partial decode.
    #[test]
    fn incomplete_linear_is_rejected() {
        let mut present = vec![Leaf::Code, Leaf::Rin, Leaf::Rout];
        assert!(check_linear_complete("proj", &present).is_ok());
        present.pop();
        let err = check_linear_complete("proj", &present).unwrap_err();
        assert!(err.contains("incomplete escha linear"), "{err}");
    }

    /// Optional: s_in, s_out, config, bias. An export without the end-to-end
    /// stage ships none of them and must still convert.
    #[test]
    fn optional_leaves_may_all_be_absent() {
        assert!(check_linear_complete("proj", &[Leaf::Code, Leaf::Rin, Leaf::Rout]).is_ok());
    }

    /// The row scale must be replicated into every block with the int8 bytes
    /// untouched — that is what makes the repack bit-exact. Recomputing block
    /// scales would be a second quantisation.
    #[test]
    fn int8_repack_replicates_the_row_scale() {
        let oc = 2;
        let ic = 64; // two Q8_0 blocks per row
        let w8: Vec<i8> = (0..(oc * ic)).map(|i| (i % 127) as i8).collect();
        let scale = vec![f16_rne(0.5), f16_rne(2.0)];
        let q8 = int8_rows_to_q8_0(&w8, &scale, oc, ic).unwrap();
        assert_eq!(q8.len(), oc * (ic / 32) * 34);
        // Both blocks of row 0 carry row 0's scale, unchanged.
        assert_eq!(&q8[0..2], &scale[0].to_le_bytes());
        assert_eq!(&q8[34..36], &scale[0].to_le_bytes());
        // Row 1's blocks carry row 1's scale.
        assert_eq!(&q8[68..70], &scale[1].to_le_bytes());
        // Payload bytes are passed through verbatim.
        assert_eq!(q8[2] as i8, w8[0]);
        assert_eq!(q8[36] as i8, w8[32]);
    }

    #[test]
    fn int8_repack_rejects_a_ragged_row() {
        assert!(int8_rows_to_q8_0(&[0i8; 20], &[0u16], 1, 20).is_err());
    }
}
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
cargo test -p hipfire-quantize --bin hipfire-quantize pipeline_escha 2>&1 | tail -15
```

Expected: compile error, `cannot find function 'k_from_code_shape'`.

- [ ] **Step 3: Implement the classification core**

Insert above the test module in `pipeline_escha.rs`:

```rust
//! Converter for EschaLabs Escha-W2 checkpoints (`quant_method` = `escha` /
//! `eschamoe`) into `.hfq`.
//!
//! Code streams are copied byte-for-byte; `memcmp` on the round-trip is a
//! post-condition. See docs/plans/escha-w2-port-design.md.

use crate::hfq::QuantType;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Leaf {
    Code,
    Rin,
    Rout,
    SIn,
    SOut,
    Config,
    Bias,
    Int8,
    Int8Scale,
    Passthrough,
    UnknownEscha,
}

/// The complete escha leaf namespace. Anything else beginning `escha_` is a
/// format mismatch from a newer exporter and must stop conversion.
pub(crate) fn classify_leaf(name: &str) -> Leaf {
    let suffix = name.rsplit('.').next().unwrap_or("");
    match suffix {
        "escha_code" => Leaf::Code,
        "escha_rin" => Leaf::Rin,
        "escha_rout" => Leaf::Rout,
        "escha_s_in" => Leaf::SIn,
        "escha_s_out" => Leaf::SOut,
        "escha_config" => Leaf::Config,
        "bias" => Leaf::Bias,
        "weight_int8" => Leaf::Int8,
        "weight_scale" => Leaf::Int8Scale,
        s if s.starts_with("escha_") => Leaf::UnknownEscha,
        _ => Leaf::Passthrough,
    }
}

/// `K` from the code tensor's own shape: the last dimension is `16 * K`.
///
/// This is the ONLY source of truth. `escha_config` is optional (spec §1.4)
/// and `layer_meta.bits` is self-inconsistent across releases — the 35B
/// records down_proj as bits 3.0 / K 3, the 27B as bits 2.0 / K 3 (spec §1.3).
pub(crate) fn k_from_code_shape(shape: &[u64]) -> Result<usize, String> {
    let last = *shape.last().ok_or("escha_code has no dimensions")? as usize;
    if last % 16 != 0 {
        return Err(format!("escha_code last dim {last} is not a multiple of 16"));
    }
    let k = last / 16;
    if k != 2 && k != 3 {
        return Err(format!("unsupported escha code rate K={k} (expected 2 or 3)"));
    }
    Ok(k)
}

pub(crate) fn quant_type_for_k(k: usize) -> Result<QuantType, String> {
    match k {
        2 => Ok(QuantType::ESCHA2T16),
        3 => Ok(QuantType::ESCHA3T16),
        _ => Err(format!("unsupported escha code rate K={k}")),
    }
}

/// Required: code, rin, rout. Optional: s_in, s_out, config, bias.
pub(crate) fn check_linear_complete(proj: &str, present: &[Leaf]) -> Result<(), String> {
    for req in [Leaf::Code, Leaf::Rin, Leaf::Rout] {
        if !present.contains(&req) {
            return Err(format!(
                "incomplete escha linear '{proj}': missing {req:?}; \
                 refusing to decode into noise"
            ));
        }
    }
    Ok(())
}
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
cargo test -p hipfire-quantize --bin hipfire-quantize pipeline_escha 2>&1 | tail -15
```

Expected: `test result: ok. 8 passed; 0 failed`.

- [ ] **Step 5: Implement `convert_escha`**

Append to `pipeline_escha.rs`:

```rust
use crate::escha_ref::fold_scales;
use crate::hfq::{write_hfq, HfqTensor};
use crate::safetensors_file::SafetensorsFile;
use std::collections::BTreeMap;

/// Convert an Escha-W2 checkpoint directory into a single `.hfq`.
///
/// `arch` is 6 for `eschamoe` (MoE) and 5 for `escha` (dense).
pub(crate) fn convert_escha(src_dir: &Path, out: &Path) -> Result<(), String> {
    let cfg: serde_json::Value = serde_json::from_slice(
        &std::fs::read(src_dir.join("config.json")).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    let qc = &cfg["quantization_config"];
    let method = qc["quant_method"].as_str().unwrap_or_default();
    let version = qc["format_version"].as_str().unwrap_or_default();
    if version != "2.0" {
        return Err(format!("unsupported escha format_version {version:?}; expected \"2.0\""));
    }
    let arch: u32 = match method {
        "eschamoe" => 6,
        "escha" => 5,
        other => return Err(format!("not an escha checkpoint: quant_method {other:?}")),
    };

    // Tensors can straddle shards (the 27B's mlp.up_proj has its escha_code in
    // shard 2 while its metadata sits in shard 1), so resolve through every
    // shard rather than per-file.
    let mut shards = Vec::new();
    let mut paths: Vec<_> = std::fs::read_dir(src_dir)
        .map_err(|e| e.to_string())?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
        .collect();
    paths.sort();
    for p in &paths {
        shards.push(SafetensorsFile::open(p).map_err(|e| e.to_string())?);
    }
    let find = |name: &str| shards.iter().find_map(|s| s.tensor_data(name));

    // Group leaves by projection prefix so completeness can be checked.
    let mut by_proj: BTreeMap<String, Vec<(String, Leaf)>> = BTreeMap::new();
    let mut passthrough: Vec<String> = Vec::new();
    for s in &shards {
        for name in s.tensor_names() {
            let leaf = classify_leaf(name);
            match leaf {
                Leaf::UnknownEscha => {
                    return Err(format!(
                        "unknown escha tensor '{name}': this build implements \
                         escha_code/rin/rout/s_in/s_out/config only. A newer \
                         exporter shipped a leaf we do not decode; refusing."
                    ))
                }
                Leaf::Passthrough | Leaf::Int8 | Leaf::Int8Scale => {
                    passthrough.push(name.to_string())
                }
                _ => {
                    let prefix = name.rsplit_once('.').unwrap().0.to_string();
                    by_proj.entry(prefix).or_default().push((name.to_string(), leaf));
                }
            }
        }
    }

    let mut tensors: Vec<HfqTensor> = Vec::new();
    for (proj, leaves) in &by_proj {
        let kinds: Vec<Leaf> = leaves.iter().map(|(_, l)| *l).collect();
        check_linear_complete(proj, &kinds)?;

        let (meta, data) = find(&format!("{proj}.escha_code"))
            .ok_or_else(|| format!("{proj}: escha_code vanished between passes"))?;
        let k = k_from_code_shape(&meta.shape)?;
        let qt = quant_type_for_k(k)?;

        // Verbatim: the code stream is copied byte-for-byte. memcmp on the
        // round-trip is the post-condition (G1).
        tensors.push(HfqTensor {
            name: format!("{proj}.escha_code"),
            quant_type: qt,
            shape: meta.shape.iter().map(|&d| d as u32).collect(),
            group_size: 16,
            data: data.to_vec(),
            spilled_len: 0,
        });

        // Fold the optional end-to-end scales into rin/rout — one f32 pair per
        // projection, per row when the tensor is E-stacked.
        let (rin_m, rin_d) = find(&format!("{proj}.escha_rin")).unwrap();
        let (rout_m, rout_d) = find(&format!("{proj}.escha_rout")).unwrap();
        let s_in = find(&format!("{proj}.escha_s_in")).map(|(_, d)| as_f32(d));
        let s_out = find(&format!("{proj}.escha_s_out")).map(|(_, d)| as_f32(d));
        let (ri, ro) = fold_scales(
            &as_u16(rin_d),
            &as_u16(rout_d),
            s_in.as_deref(),
            s_out.as_deref(),
        );
        tensors.push(f32_tensor(&format!("{proj}.escha_rin_eff"), &rin_m.shape, ri));
        tensors.push(f32_tensor(&format!("{proj}.escha_rout_eff"), &rout_m.shape, ro));

        if let Some((bm, bd)) = find(&format!("{proj}.bias")) {
            tensors.push(HfqTensor {
                name: format!("{proj}.bias"),
                quant_type: QuantType::F16,
                shape: bm.shape.iter().map(|&d| d as u32).collect(),
                group_size: 0,
                data: bd.to_vec(),
                spilled_len: 0,
            });
        }
    }

    for name in &passthrough {
        match classify_leaf(name) {
            // Consumed alongside its weight_int8 sibling.
            Leaf::Int8Scale => continue,
            Leaf::Int8 => {
                let prefix = name.rsplit_once('.').unwrap().0;
                let (m, d) = find(name).unwrap();
                let (_, sd) = find(&format!("{prefix}.weight_scale")).ok_or_else(|| {
                    format!("{name}: weight_int8 without a matching weight_scale")
                })?;
                let oc = m.shape[0] as usize;
                let ic = m.shape[1] as usize;
                let w8: Vec<i8> = d.iter().map(|&b| b as i8).collect();
                let q8 = int8_rows_to_q8_0(&w8, &as_u16(sd), oc, ic)?;
                tensors.push(HfqTensor {
                    name: format!("{prefix}.weight"),
                    quant_type: QuantType::Q8F16,
                    shape: vec![oc as u32, ic as u32],
                    group_size: 32,
                    data: q8,
                    spilled_len: 0,
                });
            }
            _ => {
                let (m, d) = find(name).unwrap();
                tensors.push(HfqTensor {
                    name: name.clone(),
                    quant_type: match m.dtype.as_str() {
                        "F16" => QuantType::F16,
                        "F32" => QuantType::F32,
                        "BF16" => QuantType::BF16,
                        other => return Err(format!("{name}: unhandled dtype {other}")),
                    },
                    shape: m.shape.iter().map(|&d| d as u32).collect(),
                    group_size: 0,
                    data: d.to_vec(),
                    spilled_len: 0,
                });
            }
        }
    }

    // The top-level "config" key is REQUIRED: `hipfire-arch-qwen35`'s
    // `config_from_hfq` -> `config_from_metadata_json` errors "qwen35: missing
    // config" before touching a single tensor, so an .hfq without it is
    // unloadable no matter how correct the codec is. Every sibling converter
    // (pipeline_gguf.rs, pipeline_deepseek.rs, pipeline_maple.rs) embeds the
    // parsed config.json the same way. `from_config_value` self-detects the
    // nested text_config/vision_config these VL-shaped checkpoints carry.
    let metadata = serde_json::json!({
        "config": cfg,
        "escha": { "format_version": version, "quant_method": method },
    })
    .to_string();
    write_hfq(out, arch, &metadata, &tensors, None).map_err(|e| e.to_string())
}

/// Escha's int8 is per-output-ROW; hipfire's `Q8_0` is per-32-element block
/// (34 bytes: f16 scale then 32 int8, per `llama.rs:148`). Replicating the row
/// scale into every block of that row passes the int8 bytes through unchanged,
/// so the reconstruction is bit-identical to Escha's `w8a16`. Cost is 2 bytes
/// per 32 elements — 6.25% — for scales that are all equal within a row.
///
/// Do NOT recompute per-block scales from the dequantised values. That is a
/// second quantisation and adds avoidable error (design §4.2.1).
pub(crate) fn int8_rows_to_q8_0(
    w8: &[i8],
    scale_f16: &[u16],
    oc: usize,
    ic: usize,
) -> Result<Vec<u8>, String> {
    if w8.len() != oc * ic {
        return Err(format!("int8 tensor is {} bytes, expected {oc}x{ic}", w8.len()));
    }
    if scale_f16.len() != oc {
        return Err(format!("expected {oc} row scales, got {}", scale_f16.len()));
    }
    if ic % 32 != 0 {
        return Err(format!("Q8_0 needs a multiple of 32 per row, got ic={ic}"));
    }
    let mut out = Vec::with_capacity(oc * (ic / 32) * 34);
    for o in 0..oc {
        let s = scale_f16[o].to_le_bytes();
        for blk in 0..ic / 32 {
            out.extend_from_slice(&s);
            let base = o * ic + blk * 32;
            out.extend(w8[base..base + 32].iter().map(|&v| v as u8));
        }
    }
    Ok(out)
}

fn as_u16(d: &[u8]) -> Vec<u16> {
    d.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect()
}

fn as_f32(d: &[u8]) -> Vec<f32> {
    d.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn f32_tensor(name: &str, shape: &[u64], v: Vec<f32>) -> HfqTensor {
    HfqTensor {
        name: name.to_string(),
        quant_type: QuantType::F32,
        shape: shape.iter().map(|&d| d as u32).collect(),
        group_size: 0,
        data: v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        spilled_len: 0,
    }
}
```

Add to `crates/hipfire-quantize/src/main.rs`, beside the existing `mod pipeline_deepseek;` / `mod pipeline_gguf;` declarations:

```rust
mod pipeline_escha;
```

- [ ] **Step 6: Build and run the full crate test suite**

```bash
cargo build --release -p hipfire-quantize 2>&1 | tail -20
cargo test -p hipfire-quantize 2>&1 | tail -15
```

Expected: build succeeds; all tests pass.

- [ ] **Step 7: Commit**

```bash
git add crates/hipfire-quantize/src/pipeline_escha.rs crates/hipfire-quantize/src/main.rs
git commit -m "feat(escha): safetensors -> hfq converter with the leaf contract enforced"
```

---

### Task 6: Convert the real 35B and prove the round-trip (G1)

The first gate that touches the actual model. It catches shape and classification errors that no synthetic test reaches.

**Files:**
- Modify: `crates/hipfire-quantize/src/main.rs` (CLI arm)
- Create: `scripts/escha-verify-roundtrip.py`

**Interfaces:**
- Consumes: `pipeline_escha::convert_escha`
- Produces: CLI `hipfire-quantize escha --src <dir> --out <file.hfq>`

- [ ] **Step 1: Wire the CLI arm**

In `crates/hipfire-quantize/src/main.rs`, add to the subcommand dispatch (follow the existing `deepseek4` / `gguf` arms verbatim for style):

```rust
        Some("escha") => {
            let src = arg_value(&args, "--src").expect("--src <checkpoint dir> required");
            let out = arg_value(&args, "--out").expect("--out <file.hfq> required");
            pipeline_escha::convert_escha(Path::new(&src), Path::new(&out))
                .unwrap_or_else(|e| panic!("escha conversion failed: {e}"));
            println!("wrote {out}");
        }
```

- [ ] **Step 2: Write the round-trip verifier**

Create `scripts/escha-verify-roundtrip.py`:

```python
#!/usr/bin/env python3
"""G1: every escha_code tensor in the .hfq must be byte-identical to source.

Verbatim repack is the whole basis for claiming no codec loss, so this is a
memcmp against the tensor at its indexed offset — not a substring search,
which would be quadratic over a 12 GB file.

HFQ layout (see hipfire-quantize/src/hfq.rs::write_hfq):
  header 32B : magic[4] "HFQM", version u32, arch u32, n_tensors u32,
               metadata_offset u64, data_offset u64
  metadata   : JSON at metadata_offset
  index      : n_tensors u32, then per tensor
               name_len u16, name, quant_type u8, ndim u8,
               dims u32*ndim, group_size u32, data_len u64
  data       : at data_offset (4096-aligned), tensors concatenated in order
"""
import json, mmap, struct, sys
from pathlib import Path

ESCHA_QT = {42: "ESCHA2T16", 43: "ESCHA3T16"}


def hfq_tensors(mm):
    assert mm[:4] == b"HFQM", "not an HFQ file"
    version, arch, n_tensors = struct.unpack_from("<III", mm, 4)
    metadata_offset, data_offset = struct.unpack_from("<QQ", mm, 16)
    del version, arch
    # The index begins immediately after the metadata JSON blob, which the
    # writer emits with no length prefix — so walk the JSON to find its end.
    meta_end = metadata_offset
    depth, in_str, esc = 0, False, False
    while meta_end < data_offset:
        c = mm[meta_end]
        meta_end += 1
        if esc:
            esc = False
        elif in_str:
            if c == 0x5C:
                esc = True
            elif c == 0x22:
                in_str = False
        elif c == 0x22:
            in_str = True
        elif c == 0x7B:
            depth += 1
        elif c == 0x7D:
            depth -= 1
            if depth == 0:
                break
    pos = meta_end
    (count,) = struct.unpack_from("<I", mm, pos)
    pos += 4
    assert count == n_tensors, f"index count {count} != header {n_tensors}"
    out, running = {}, 0
    for _ in range(n_tensors):
        (name_len,) = struct.unpack_from("<H", mm, pos)
        pos += 2
        name = bytes(mm[pos:pos + name_len]).decode()
        pos += name_len
        qt, ndim = struct.unpack_from("<BB", mm, pos)
        pos += 2
        pos += 4 * ndim
        pos += 4  # group_size
        (data_len,) = struct.unpack_from("<Q", mm, pos)
        pos += 8
        out[name] = (qt, data_offset + running, data_len)
        running += data_len
    return out


def safetensors_tensors(d):
    out = {}
    for shard in sorted(Path(d).glob("*.safetensors")):
        raw = shard.read_bytes()
        (n,) = struct.unpack_from("<Q", raw, 0)
        hdr = json.loads(raw[8:8 + n])
        for name, meta in hdr.items():
            if name == "__metadata__":
                continue
            s, e = meta["data_offsets"]
            out[name] = raw[8 + n + s:8 + n + e]
    return out


def main(src, hfq_path):
    st = safetensors_tensors(src)
    codes = {k: v for k, v in st.items() if k.endswith(".escha_code")}
    with open(hfq_path, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        idx = hfq_tensors(mm)
        missing = [n for n in codes if n not in idx]
        wrong_qt, mismatch = [], []
        for name, src_bytes in codes.items():
            if name in missing:
                continue
            qt, off, ln = idx[name]
            if qt not in ESCHA_QT:
                wrong_qt.append(f"{name}: quant_type {qt}")
            elif ln != len(src_bytes) or mm[off:off + ln] != src_bytes:
                mismatch.append(name)
        print(f"escha_code tensors in source : {len(codes)}")
        print(f"  absent from the .hfq index : {len(missing)}")
        print(f"  wrong quant_type           : {len(wrong_qt)}")
        print(f"  present but not byte-equal : {len(mismatch)}")
        for n in (missing + wrong_qt + mismatch)[:10]:
            print("   ", n)
        if missing or wrong_qt or mismatch:
            return 1
    print("G1 PASS: every code stream is byte-identical")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
```

```bash
chmod +x scripts/escha-verify-roundtrip.py
```

- [ ] **Step 3: Download the model**

```bash
mkdir -p /data/hipfire-models/escha-35b
huggingface-cli download EschaLabs/Qwen3.6-35B-A3B-Escha-W2 \
  --local-dir /data/hipfire-models/escha-35b
du -sh /data/hipfire-models/escha-35b
```

Expected: about 12.3 GB. Do not run concurrent resumable downloads over the same files — a partial-range clobber corrupts the shard silently.

- [ ] **Step 4: Convert**

```bash
cargo build --release -p hipfire-quantize
./target/release/hipfire-quantize escha \
  --src /data/hipfire-models/escha-35b \
  --out /data/hipfire-models/escha-35b.hfq
ls -la /data/hipfire-models/escha-35b.hfq
```

Expected: writes without error. Any `unknown escha tensor`, `incomplete escha linear`, or `unhandled dtype` failure is a real finding — fix the classifier, do not add a skip.

- [ ] **Step 5: Verify the round-trip (G1)**

```bash
python3 scripts/escha-verify-roundtrip.py \
  /data/hipfire-models/escha-35b /data/hipfire-models/escha-35b.hfq
```

Expected: `escha_code tensors in source: 80` and `G1 PASS: every code stream is byte-identical`.

(80 = 40 layers × 2 projections. If the count is not 80, the classifier is dropping tensors.)

- [ ] **Step 6: Commit**

```bash
git add crates/hipfire-quantize/src/main.rs scripts/escha-verify-roundtrip.py
git commit -m "feat(escha): quantiser CLI arm and G1 verbatim round-trip verifier"
```

---

### Task 7: GPU tile decode to `Q8_0` (G2)

**Files:**
- Create: `kernels/src/escha_decode_tiles.hip`
- Modify: `crates/rdna-compute/src/kernels.rs` (add `include_str!` const beside `GEMV_Q8_0_SRC` at line 4810)
- Modify: `crates/rdna-compute/src/dispatch.rs:4251` (register the kernel spec)
- Test: `crates/rdna-compute/examples/test_escha_decode_gpu_vs_cpu.rs`

**Interfaces:**
- Consumes: `escha_ref::reconstruct` as the oracle
- Produces: kernel `escha_decode_tiles`, const `ESCHA_DECODE_TILES_SRC`, `Gpu::escha_decode_tiles(&mut self, code: &GpuTensor, out_q8: &GpuTensor, in_features: u32, out_features: u32, k: u32) -> HipResult<()>`

- [ ] **Step 1: Write the kernel**

Create `kernels/src/escha_decode_tiles.hip`:

```cpp
// Escha-W2 one-shot tile decode -> Q8_0 resident weights (Phase 1).
//
// Reads the verbatim int16 code stream, decodes 16x16 trellis tiles, and
// writes hipfire Q8_0 (34 bytes per 32 elements: f16 scale + 32 int8).
//
// The codebook is computed inline. A 65536-entry fp16 LUT would be 128 KB and
// gfx1151 has 64 KB LDS, so there is no table anywhere in this file.
//
// Escha's tile grid is in-major [in/16, out/16]; hipfire stores weights
// out-major [out, in]. This kernel transposes on the way out.
//
// DELIBERATE DUPLICATION: hipfire-quantize/src/escha_ref.rs implements this
// same lane maths in Rust. That is the G2 gate — this kernel is asserted
// bit-exact against it. Generating either from the other would make G2
// circular. Do not deduplicate.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

__device__ __forceinline__ __half escha_cba(unsigned short state) {
    unsigned int r = ((unsigned int)state * 0xCBAC1FEDu) & 0x8FFF8FFFu;
    r ^= 0x3B603B60u;
    __half lo = __ushort_as_half((unsigned short)(r & 0xFFFFu));
    __half hi = __ushort_as_half((unsigned short)(r >> 16));
    return __hadd(lo, hi);  // fp16 RNE add — matches the reference exactly
}

__device__ __forceinline__ void escha_decode8_k2(
    const unsigned int* w, int lane, unsigned short* out) {
    int t_off = lane * 8;
    int i1 = t_off >> 4;
    int i0 = (i1 + 15) & 15;
    unsigned long long merged = ((unsigned long long)w[i0] << 32) | w[i1];
    int shift = ((~t_off) & 8) << 1;   // 16 for even lanes, 0 for odd
    unsigned int v = (unsigned int)((merged >> shift) & 0xFFFFFFFFull);
    #pragma unroll
    for (int j = 0; j < 8; ++j) out[j] = (unsigned short)(v >> (2 * (7 - j)));
}

__device__ __forceinline__ void escha_decode8_k3(
    const unsigned int* w, int lane, unsigned short* out) {
    const int BITS = 3;
    int t_off = lane * 8;
    int b1 = (t_off + 257) * BITS;
    int b0 = b1 - 16;
    int b2 = b1 + BITS * 7;
    int i0 = b0 >> 5;
    int i2 = (b2 - 1) >> 5;
    int s2 = ((i2 + 1) << 5) - b2;
    unsigned long long merged =
        ((unsigned long long)w[i0 % 24] << 32) | w[i2 % 24];
    unsigned int w7 = (unsigned int)((merged >> s2) & 0xFFFFFFFFull);
    unsigned int w3 = (unsigned int)((merged >> (s2 + BITS * 4)) & 0xFFFFFFFFull);
    out[0] = (unsigned short)(w3 >> 9); out[1] = (unsigned short)(w3 >> 6);
    out[2] = (unsigned short)(w3 >> 3); out[3] = (unsigned short)(w3);
    out[4] = (unsigned short)(w7 >> 9); out[5] = (unsigned short)(w7 >> 6);
    out[6] = (unsigned short)(w7 >> 3); out[7] = (unsigned short)(w7);
}

// One block per tile. 32 lanes, 8 values each = the 256 tile slots.
extern "C" __global__ void escha_decode_tiles(
    const short* __restrict__ code,   // [in/16, out/16, 16K]
    __half* __restrict__ bare,        // [in, out] fp16 scratch
    int in_features, int out_features, int K) {
    int tile = blockIdx.x;
    int tn = out_features / 16;
    int kt = tile / tn, nt = tile % tn;
    int lane = threadIdx.x;
    if (lane >= 32) return;

    unsigned int words[24];
    const short* src = code + (size_t)tile * 16 * K;
    for (int i = 0; i < 8 * K; ++i)
        words[i] = ((unsigned int)(unsigned short)src[2 * i]) |
                   (((unsigned int)(unsigned short)src[2 * i + 1]) << 16);

    unsigned short st[8];
    if (K == 2) escha_decode8_k2(words, lane, st);
    else        escha_decode8_k3(words, lane, st);

    int l0 = lane & ~4;
    int c_off = (lane >> 2) & 1;
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        int fi = j >> 1;
        int row = (lane & 3) * 2 + (j & 1) + (fi & 1) * 8;
        int col = 2 * ((l0 >> 3) + (j >= 4 ? 4 : 0)) + c_off;
        bare[(size_t)(kt * 16 + row) * out_features + (nt * 16 + col)] = escha_cba(st[j]);
    }
}
```

- [ ] **Step 2: Register the kernel source**

In `crates/rdna-compute/src/kernels.rs`, beside line 4810:

```rust
pub const ESCHA_DECODE_TILES_SRC: &str =
    include_str!("../../../kernels/src/escha_decode_tiles.hip");
```

In `crates/rdna-compute/src/dispatch.rs` near line 4251:

```rust
                specs.push(("escha_decode_tiles", kernels::ESCHA_DECODE_TILES_SRC.to_string()));
```

- [ ] **Step 3: Write the GPU-vs-CPU parity test**

Create `crates/rdna-compute/examples/test_escha_decode_gpu_vs_cpu.rs`:

```rust
//! G2: GPU tile decode must match escha_ref::reconstruct EXACTLY in fp16,
//! for both K. Run:
//!   cargo run --release -p rdna-compute --example test_escha_decode_gpu_vs_cpu
use hipfire_quantize::escha_ref;

fn main() {
    for (name, ic, oc, k) in [
        ("packed_gu_e0_k2.i16", 2048usize, 1024usize, 2usize),
        ("packed_down_e0_k3.i16", 512, 2048, 3),
    ] {
        let path = format!(
            "{}/../hipfire-quantize/tests/data/escha/{name}",
            env!("CARGO_MANIFEST_DIR")
        );
        let raw = std::fs::read(&path).expect("run fetch-goldens.sh first");
        let code: Vec<i16> =
            raw.chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]])).collect();
        let want = escha_ref::reconstruct(&code, ic, oc, k);

        let mut gpu = rdna_compute::Gpu::new().expect("gpu");
        let got = gpu.escha_decode_tiles_host(&code, ic as u32, oc as u32, k as u32)
            .expect("decode");

        let bad = want.iter().zip(&got).filter(|(a, b)| a != b).count();
        println!("{name}: {bad} mismatched of {} elements", want.len());
        assert_eq!(bad, 0, "{name}: GPU decode diverges from the CPU reference");
    }
    println!("G2 PASS");
}
```

Add `hipfire-quantize` to `crates/rdna-compute/Cargo.toml` under `[dev-dependencies]`:

```toml
hipfire-quantize = { path = "../hipfire-quantize" }
```

- [ ] **Step 4: Run it and confirm it fails**

```bash
cargo run --release -p rdna-compute --example test_escha_decode_gpu_vs_cpu 2>&1 | tail -10
```

Expected: compile error, `no method named 'escha_decode_tiles_host'`.

- [ ] **Step 5: Implement the host wrapper**

Add to `crates/rdna-compute/src/gemv.rs` beside `gemv_q8_0` (line 13563), following its `ensure_kernel` + launch style:

```rust
    /// Decode an escha code stream to a bare fp16 weight matrix `[ic, oc]`.
    /// Host-side helper used by the G2 parity gate; the load path uses the
    /// device-resident form.
    pub fn escha_decode_tiles_host(
        &mut self,
        code: &[i16],
        in_features: u32,
        out_features: u32,
        k: u32,
    ) -> HipResult<Vec<u16>> {
        self.bind_thread()?;
        let n_elems = (in_features as usize) * (out_features as usize);
        let n_tiles = (in_features / 16) * (out_features / 16);
        let code_bytes: Vec<u8> = code.iter().flat_map(|v| v.to_le_bytes()).collect();
        let d_code = self.upload_raw(&code_bytes, &[code.len()])?;
        let d_bare = self.alloc_tensor(&[n_elems], DType::F16)?;
        self.ensure_kernel(
            "escha_decode_tiles",
            kernels::ESCHA_DECODE_TILES_SRC,
            "escha_decode_tiles",
        )?;

        let mut code_ptr = d_code.buf.as_ptr();
        let mut bare_ptr = d_bare.buf.as_ptr();
        let mut ic = in_features as i32;
        let mut oc = out_features as i32;
        let mut kk = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut code_ptr as *mut _ as *mut c_void,
            &mut bare_ptr as *mut _ as *mut c_void,
            &mut ic as *mut _ as *mut c_void,
            &mut oc as *mut _ as *mut c_void,
            &mut kk as *mut _ as *mut c_void,
        ];
        let func = &self.functions["escha_decode_tiles"];
        unsafe {
            self.hip
                .launch_kernel(func, [n_tiles, 1, 1], [32, 1, 1], 0, None, &mut params)?;
        }

        let mut out = vec![0u8; n_elems * 2];
        self.hip.memcpy_dtoh(&mut out, &d_bare.buf)?;
        Ok(out
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect())
    }
```

This follows the launch convention used throughout `gemv.rs` (see `gemv_q8_0`
at line 13563): `bind_thread` → `upload_raw`/`alloc_tensor` → `ensure_kernel` →
build a `Vec<*mut c_void>` of `&mut` locals → look the function up in
`self.functions` → `unsafe { self.hip.launch_kernel(...) }`. There is no
`Gpu::launch_kernel(name, ...)` helper taking buffers directly; do not invent
one. `c_void` is already imported in `gemv.rs`.

- [ ] **Step 6: Run the parity gate (G2)**

```bash
which hipcc || echo "FIX: hipcc must be on PATH or the daemon cannot JIT"
cargo run --release -p rdna-compute --example test_escha_decode_gpu_vs_cpu 2>&1 | tail -10
```

Expected:

```
packed_gu_e0_k2.i16: 0 mismatched of 2097152 elements
packed_down_e0_k3.i16: 0 mismatched of 1048576 elements
G2 PASS
```

Any nonzero mismatch count is a decode bug, not a rounding artifact — `__hadd` is RNE and the CPU reference rounds identically. If K=2 passes and K=3 fails, the bug is in `escha_decode8_k3`'s modular wrap.

- [ ] **Step 7: Commit**

```bash
git add kernels/src/escha_decode_tiles.hip crates/rdna-compute/src/kernels.rs \
        crates/rdna-compute/src/dispatch.rs crates/rdna-compute/src/gemv.rs \
        crates/rdna-compute/examples/test_escha_decode_gpu_vs_cpu.rs \
        crates/rdna-compute/Cargo.toml
git commit -m "feat(escha): GPU tile decode kernel, bit-exact against the CPU reference"
```

---

### Task 8: H128 input and output transforms (G3)

**Files:**
- Create: `kernels/src/escha_h128.hip`
- Modify: `crates/rdna-compute/src/kernels.rs`, `crates/rdna-compute/src/dispatch.rs`
- Test: `crates/rdna-compute/examples/test_escha_h128_gpu_vs_cpu.rs`

**Interfaces:**
- Consumes: `escha_ref::{h128_inplace, input_transform, output_transform, RS}`
- Produces: kernels `escha_h128_in`, `escha_h128_out`; `Gpu::escha_h128_in(...)`, `Gpu::escha_h128_out(...)`

- [ ] **Step 1: Write the kernel**

Create `kernels/src/escha_h128.hip`:

```cpp
// Escha-W2 activation transforms.
//   in : xh = f16( H128(x * rin) * RS )
//   out: y  = f16( H128(mid) * RS * rout )
//
// H128 is the UNNORMALISED 128-point Walsh-Hadamard in Sylvester (natural)
// order. Escha folds its sign flips into rin/rout, so unlike hipfire's
// gemv_mq4g128 there is no sign-seed stage here.
//
// A pruned output channel (rout == 0) must come out EXACTLY zero — do not
// reorder the final multiply in a way that could produce -0.0 or a denormal.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define ESCHA_RS 0.0883883476f   // 1/sqrt(128)

__device__ __forceinline__ void h128_block(float* v) {
    for (int h = 1; h < 128; h <<= 1) {
        for (int i = 0; i < 128; i += (h << 1)) {
            for (int j = i; j < i + h; ++j) {
                float a = v[j], b = v[j + h];
                v[j] = a + b;
                v[j + h] = a - b;
            }
        }
    }
}

// One block per 128-channel group. 128 threads cooperate via LDS.
extern "C" __global__ void escha_h128_in(
    const float* __restrict__ x, const float* __restrict__ rin,
    __half* __restrict__ xh, int n) {
    __shared__ float s[128];
    int g = blockIdx.x, t = threadIdx.x;
    int idx = g * 128 + t;
    if (idx >= n) return;
    s[t] = x[idx] * rin[idx];
    __syncthreads();
    if (t == 0) h128_block(s);
    __syncthreads();
    xh[idx] = __float2half(s[t] * ESCHA_RS);
}

extern "C" __global__ void escha_h128_out(
    const float* __restrict__ mid, const float* __restrict__ rout,
    __half* __restrict__ y, int n) {
    __shared__ float s[128];
    int g = blockIdx.x, t = threadIdx.x;
    int idx = g * 128 + t;
    if (idx >= n) return;
    s[t] = mid[idx];
    __syncthreads();
    if (t == 0) h128_block(s);
    __syncthreads();
    y[idx] = __float2half(s[t] * ESCHA_RS * rout[idx]);
}
```

- [ ] **Step 2: Register both kernels**

In `crates/rdna-compute/src/kernels.rs`:

```rust
pub const ESCHA_H128_SRC: &str = include_str!("../../../kernels/src/escha_h128.hip");
```

In `crates/rdna-compute/src/dispatch.rs` near line 4251:

```rust
                specs.push(("escha_h128", kernels::ESCHA_H128_SRC.to_string()));
```

- [ ] **Step 3: Write the parity test**

Create `crates/rdna-compute/examples/test_escha_h128_gpu_vs_cpu.rs`:

```rust
//! G3: the H128 kernels must match escha_ref directly.
//!
//! A round-trip check (H128 . H128 == 128 I) is NOT sufficient — a wrong
//! butterfly order is also self-inverse and would pass it while being wrong.
use hipfire_quantize::escha_ref;
use hipfire_quantize::float16::f16_to_f32;

fn main() {
    let n = 2048usize;
    let x: Vec<f32> = (0..n).map(|i| ((i * 37) as f32 * 0.017).sin()).collect();
    let rin: Vec<f32> = (0..n).map(|i| if i % 3 == 0 { -0.0023 } else { 0.0023 }).collect();
    let mut rout: Vec<f32> = (0..n).map(|i| 1.0 + (i % 5) as f32 * 0.1).collect();
    rout[7] = 0.0;
    rout[1000] = 0.0; // pruned channels must stay exactly zero

    let want_in = escha_ref::input_transform(&x, &rin);
    let want_out = escha_ref::output_transform(&x, &rout);

    let mut gpu = rdna_compute::Gpu::new().expect("gpu");
    let got_in = gpu.escha_h128_in_host(&x, &rin).expect("h128 in");
    let got_out = gpu.escha_h128_out_host(&x, &rout).expect("h128 out");

    let bad_in = want_in.iter().zip(&got_in).filter(|(a, b)| a != b).count();
    let bad_out = want_out.iter().zip(&got_out).filter(|(a, b)| a != b).count();
    println!("h128_in : {bad_in} mismatched of {n}");
    println!("h128_out: {bad_out} mismatched of {n}");
    assert_eq!(f16_to_f32(got_out[7]), 0.0, "pruned channel 7 must be exactly zero");
    assert_eq!(f16_to_f32(got_out[1000]), 0.0, "pruned channel 1000 must be exactly zero");
    assert_eq!(bad_in, 0);
    assert_eq!(bad_out, 0);
    println!("G3 PASS");
}
```

- [ ] **Step 4: Run it and confirm it fails**

```bash
cargo run --release -p rdna-compute --example test_escha_h128_gpu_vs_cpu 2>&1 | tail -10
```

Expected: compile error, `no method named 'escha_h128_in_host'`.

- [ ] **Step 5: Implement the host wrappers**

Add to `crates/rdna-compute/src/gemv.rs`:

```rust
    /// `xh = f16( H128(x * rin) * RS )` on device. Host-side helper for the
    /// G3 parity gate; the forward path uses the device-resident form.
    pub fn escha_h128_in_host(&mut self, x: &[f32], rin: &[f32]) -> HipResult<Vec<u16>> {
        self.escha_h128_host_impl("escha_h128_in", x, rin)
    }

    /// `y = f16( H128(mid) * RS * rout )` on device.
    pub fn escha_h128_out_host(&mut self, mid: &[f32], rout: &[f32]) -> HipResult<Vec<u16>> {
        self.escha_h128_host_impl("escha_h128_out", mid, rout)
    }

    fn escha_h128_host_impl(
        &mut self,
        entry: &str,
        a: &[f32],
        vec_in: &[f32],
    ) -> HipResult<Vec<u16>> {
        assert_eq!(a.len(), vec_in.len());
        assert_eq!(a.len() % 128, 0, "H128 needs a multiple of 128");
        self.bind_thread()?;
        let n = a.len();
        let a_bytes: Vec<u8> = a.iter().flat_map(|v| v.to_le_bytes()).collect();
        let v_bytes: Vec<u8> = vec_in.iter().flat_map(|v| v.to_le_bytes()).collect();
        let d_a = self.upload_raw(&a_bytes, &[n])?;
        let d_v = self.upload_raw(&v_bytes, &[n])?;
        let d_out = self.alloc_tensor(&[n], DType::F16)?;
        self.ensure_kernel("escha_h128", kernels::ESCHA_H128_SRC, entry)?;

        let mut a_ptr = d_a.buf.as_ptr();
        let mut v_ptr = d_v.buf.as_ptr();
        let mut o_ptr = d_out.buf.as_ptr();
        let mut n_val = n as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut v_ptr as *mut _ as *mut c_void,
            &mut o_ptr as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];
        let func = &self.functions[entry];
        unsafe {
            self.hip
                .launch_kernel(func, [(n / 128) as u32, 1, 1], [128, 1, 1], 0, None, &mut params)?;
        }

        let mut raw = vec![0u8; n * 2];
        self.hip.memcpy_dtoh(&mut raw, &d_out.buf)?;
        Ok(raw
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect())
    }
```

Note `ensure_kernel(module, SRC, entry)` is called once per entry point — the
two entries live in the same `.hip` source but are separate device functions,
so each needs its own `ensure_kernel` call and its own `self.functions` lookup
key.

- [ ] **Step 6: Run the parity gate (G3)**

```bash
cargo run --release -p rdna-compute --example test_escha_h128_gpu_vs_cpu 2>&1 | tail -10
```

Expected:

```
h128_in : 0 mismatched of 2048
h128_out: 0 mismatched of 2048
G3 PASS
```

- [ ] **Step 7: Commit**

```bash
git add kernels/src/escha_h128.hip crates/rdna-compute/src/kernels.rs \
        crates/rdna-compute/src/dispatch.rs crates/rdna-compute/src/gemv.rs \
        crates/rdna-compute/examples/test_escha_h128_gpu_vs_cpu.rs
git commit -m "feat(escha): H128 input/output transform kernels, gated against the reference"
```

---

### Task 9: Verify the arch-6 router contract (G4b)

Done before wiring, because if the router disagrees the wiring is built on
sand. The spec assumed arch-6's router is reusable; this exercises the real
router and proves or disproves it.

**Files:**
- Create: `crates/hipfire-arch-qwen35/examples/escha_router_contract.rs`
- Possibly modify: the arch-6 router path (only if Step 4 shows a mismatch)

**Interfaces:**
- Consumes: the `.hfq` from Task 6 (for `layers.0.mlp.gate.weight`), the golden
  fixture from `crates/hipfire-quantize/tests/data/escha/`
- Produces: a passing G4b assertion, or a fix to the router

- [ ] **Step 1: Fetch the fixture**

```bash
crates/hipfire-quantize/tests/data/escha/fetch-goldens.sh
```

Expected: eight files. `moeblk_ids.i64` must have digest
`0781eecdacd5fbfe30887f6d1f6af5d5ca001d32253da8bb3e1294230c2ed649`.

- [ ] **Step 2: Write the failing test**

This calls hipfire's **actual** arch-6 router — not a reimplementation of it —
and asserts against Escha's shipped selection.

Create `crates/hipfire-arch-qwen35/examples/escha_router_contract.rs`:

```rust
//! G4b: hipfire's arch-6 router must select the same experts Escha does.
//!
//! Escha rounds router logits to f16 BEFORE top-k (`ref.py`: the logits are
//! computed as f16 then widened to f32 to select). Selecting on unrounded f32
//! logits is a different function, and the rounding manufactures exact ties
//! that f32 never produces.
//!
//! Asserts the SET, not the order: the combine is a sum over slots, so intra-k
//! order cannot change the output. On the fixture, token 3 has two experts on
//! identical f16 logits (1.80078), and which one lands in which slot is
//! implementation-defined.
//!
//! Run:
//!   cargo run --release -p hipfire-arch-qwen35 \
//!     --example escha_router_contract -- /data/hipfire-models/escha-35b.hfq
use std::collections::HashSet;
use std::path::PathBuf;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../hipfire-quantize/tests/data/escha")
        .join(name)
}

fn read_f16_as_f32(name: &str) -> Vec<f32> {
    let raw = std::fs::read(fixture(name)).expect("run fetch-goldens.sh first");
    raw.chunks_exact(2)
        .map(|c| hipfire_quantize::float16::f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect()
}

fn main() {
    let hfq = std::env::args().nth(1).expect("usage: <model.hfq>");
    let x = read_f16_as_f32("moeblk_x.f16"); // [8, 2048]
    let raw_ids = std::fs::read(fixture("moeblk_ids.i64")).unwrap();
    let want_ids: Vec<i64> = raw_ids
        .chunks_exact(8)
        .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
        .collect(); // [8, 8]

    // Call hipfire's real router for layer 0 on each token.
    let got = hipfire_arch_qwen35::escha_router_topk_for_test(&hfq, 0, &x, 8, 2048, 8)
        .expect("router");

    let mut bad = 0usize;
    for t in 0..8 {
        let want: HashSet<i64> = want_ids[t * 8..(t + 1) * 8].iter().copied().collect();
        let mine: HashSet<i64> = got[t * 8..(t + 1) * 8].iter().map(|&v| v as i64).collect();
        if want != mine {
            bad += 1;
            println!("token {t}: escha={:?}", &want_ids[t * 8..(t + 1) * 8]);
            println!("         hipfire={:?}", &got[t * 8..(t + 1) * 8]);
        }
    }
    println!("tokens with a differing top-8 SET: {bad}/8");
    assert_eq!(
        bad, 0,
        "arch-6 router selects different experts than escha. Most likely cause: \
         it is not rounding logits to f16 before top-k."
    );
    println!("G4b PASS");
}
```

- [ ] **Step 3: Run it and confirm it fails**

```bash
cargo run --release -p hipfire-arch-qwen35 --example escha_router_contract \
  -- /data/hipfire-models/escha-35b.hfq 2>&1 | tail -15
```

Expected: compile error — `escha_router_topk_for_test` does not exist yet.

- [ ] **Step 4: Expose the router and run the gate**

Read the arch-6 router path first:

```bash
grep -rn "moe_topk\|router\|norm_topk_prob" crates/hipfire-arch-qwen35/src/ | head -20
grep -rn "moe_topk" kernels/src/*.hip | head
```

Add a thin `pub fn escha_router_topk_for_test(hfq_path: &str, layer: usize, x: &[f32], n_tokens: usize, hidden: usize, top_k: usize) -> Result<Vec<u32>, String>` that loads `layers.{layer}.mlp.gate.weight` from the `.hfq` and runs **the production router path** — not a reimplementation. If that path is not callable in isolation, extract the selection step into a function both it and this helper call, and leave the production behaviour unchanged.

Then run the gate:

```bash
cargo run --release -p hipfire-arch-qwen35 --example escha_router_contract \
  -- /data/hipfire-models/escha-35b.hfq 2>&1 | tail -15
```

- **If it passes:** the router already matches. Record that in the commit message and change nothing.
- **If it fails:** check whether the router rounds logits to f16 before top-k. If it does not, add the rounding **on the escha path only**, keyed on the model carrying `ESCHA2T16`/`ESCHA3T16` experts, so existing `qwen3.6:35b-a3b-*` SKUs keep their current selection bit-for-bit. Re-run until `G4b PASS`.

- [ ] **Step 5: Confirm no existing SKU changed**

Only required if Step 4 modified the router.

```bash
cargo test -p hipfire-arch-qwen35 2>&1 | tail -15
cargo test -p hipfire-dispatch-tests qwen35 2>&1 | tail -10
```

Expected: all pass, unchanged.

- [ ] **Step 6: Commit**

```bash
git add crates/hipfire-arch-qwen35
git commit -m "test(escha): G4b — arch-6 router selects the same experts as escha"
```

---

### Task 10: Wire arch-6 to load and run escha experts (G4)

**Files:**
- Modify: `crates/hipfire-arch-qwen35/src/qwen35/weights.rs:69-80` (expert loading)
- Modify: `crates/hipfire-arch-qwen35/src/qwen35.rs` (MoE forward: call the transforms)
- Modify: `crates/hipfire-loader/src/lib.rs` (accept the new quant types)
- Test: `crates/hipfire-arch-qwen35/examples/escha_moe_block_gate.rs` (new)

**Interfaces:**
- Consumes: `Gpu::escha_decode_tiles_host` (Task 7), `Gpu::escha_h128_in_host` / `Gpu::escha_h128_out_host` (Task 8), `escha_ref::{expert_linear, swiglu, w8a16}`. The load path needs device-resident variants of all three that take an existing `GpuTensor` instead of host slices; add them beside the `_host` helpers rather than round-tripping through the CPU per expert.
- Produces: arch-6 loads `ESCHA2T16`/`ESCHA3T16` experts as `Q8_0` resident

- [ ] **Step 1: Fetch the MoE golden fixture**

```bash
crates/hipfire-quantize/tests/data/escha/fetch-goldens.sh
```

Expected: eight files, with `moeblk_out.f16` at digest `484abdf7257631105d90e5e8f7794974d02fab6c7dd587e30955f660a688c4d4`.

- [ ] **Step 2: Write the G4 gate**

Create `crates/hipfire-arch-qwen35/examples/escha_moe_block_gate.rs`. It loads `/data/hipfire-models/escha-35b.hfq`, runs the layer-0 MoE block on `moeblk_x.f16` with `moeblk_ids.i64` / `moeblk_scores.f32` injected, and compares against `moeblk_out.f16` with these **measured** tolerances:

```rust
    let max_abs = diffs.iter().cloned().fold(0.0f32, f32::max);
    let mean_abs = diffs.iter().sum::<f32>() / diffs.len() as f32;
    println!("MoE block: max|diff|={max_abs:.3e} mean|diff|={mean_abs:.3e}");
    // Measured against the shipped layer-0 weights: max 1.22e-4, mean 2.1e-6,
    // on outputs of mean magnitude 0.0185. The golden came from the Metal
    // path, not ref.py, so this is a TOLERANCE gate. The codec goldens in G0
    // are bit-exact; do not generalise these bounds to them.
    assert!(max_abs <= 2e-4, "max|diff| {max_abs:.3e} exceeds 2e-4");
    assert!(mean_abs <= 1e-5, "mean|diff| {mean_abs:.3e} exceeds 1e-5");
    println!("G4 PASS");
```

Inject the routing rather than computing it — the fixture ships ids/scores precisely because it does not gate the router (that is Task 9's job).

- [ ] **Step 3: Run it and confirm it fails**

```bash
cargo run --release -p hipfire-arch-qwen35 --example escha_moe_block_gate 2>&1 | tail -15
```

Expected: failure — the loader does not yet accept `ESCHA2T16`.

- [ ] **Step 4: Implement expert loading**

In `crates/hipfire-arch-qwen35/src/qwen35/weights.rs`, extend the expert loader documented at line 69 (`experts[X].gate_up: [2*moe_intermediate, hidden]`). For each expert:

1. Read the `ESCHA2T16` `gate_up` and `ESCHA3T16` `down` code streams.
2. Call the device-resident `escha_decode_tiles` to produce bare fp16 `[ic, oc]`.
3. Quantise to `Q8_0` per output row and store in the existing `experts[X].gate_up` / `.down` slots — hipfire is out-major, escha's grid is in-major, so this is where the transpose lands.
4. Keep `escha_rin_eff` / `escha_rout_eff` as f32 device tensors alongside.

In the MoE forward path, wrap each expert projection: the device-resident `escha_h128_in` before the `Q8_0` GEMV, `escha_h128_out` after. SwiGLU consumes the **f16-rounded** merged `gate_up` output; the combine multiplies by `f16(score)`.

- [ ] **Step 5: Run the G4 gate**

```bash
cargo run --release -p hipfire-arch-qwen35 --example escha_moe_block_gate 2>&1 | tail -15
```

Expected: `max|diff|` around `1.2e-4`, `mean|diff|` around `2e-6`, then `G4 PASS`.

If `max|diff|` is around 1e-1 rather than 1e-4, the H128 pair is not being applied — check that the escha types did not fall through to a Plain GEMV (Task 4's guard should have made that impossible; if it did happen, the guard has a hole).

- [ ] **Step 6: Commit**

```bash
git add crates/hipfire-arch-qwen35/src crates/hipfire-loader/src/lib.rs \
        crates/hipfire-arch-qwen35/examples/escha_moe_block_gate.rs
git commit -m "feat(escha): load escha experts as Q8_0 and run the H128 pair in arch-6"
```

---

### Task 11: Registry entry, coherence and KLD (G5)

**Files:**
- Modify: `registry/v1.json`
- Create: `scripts/escha-kld.sh`

- [ ] **Step 1: Add the registry entry**

In `registry/v1.json`, beside the existing `qwen3.6:35b-a3b` entries:

```json
{ "name": "qwen3.6:35b-a3b-escha", "arch_id": 6, "quant": "escha" }
```

Match the surrounding entries' exact field set — copy a neighbouring `qwen3.6:35b-a3b-mq2` object and change only `name` and `quant`.

- [ ] **Step 2: Serve and check coherence**

```bash
which hipcc || echo "FIX: hipcc must be on PATH"
HIPFIRE_MODEL=/data/hipfire-models/escha-35b.hfq ~/.hipfire/serve-stable.sh &
sleep 60
curl -s localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.6:35b-a3b-escha","messages":[{"role":"user","content":"What is the capital of France? Answer in one word."}],"max_tokens":16}' \
  | python3 -c "import sys,json;print(json.load(sys.stdin)['choices'][0]['message']['content'])"
```

Expected: `Paris`. Garbled output on a first run is usually the kernel cache, not the codec — confirm a single-toolchain cache before investigating the decode.

- [ ] **Step 3: Measure VRAM against the prediction**

```bash
rocm-smi --showmeminfo vram | head -20
```

Expected: about **36.7 GB** resident. A number far below that means experts are not all resident; far above means something is being held at fp16 rather than `Q8_0`.

- [ ] **Step 4: Run the KLD gate (G5)**

Create `scripts/escha-kld.sh`:

```bash
#!/usr/bin/env bash
# G5: two KLD comparisons on a FIXED corpus slice, teacher-forced.
#
# The reference is escha_ref on CPU, NOT any Escha runtime: escha-mlx is
# Metal, the escha wheel is CUDA, and ZML needs an NVIDIA driver, so none of
# them execute on gfx1151. ref.py declares itself the semantic contract for
# their kernels and is gated on the goldens, so this is exact rather than
# cross-machine.
#
# Score on the fixed corpus, never on the model's own greedy output: for ds4
# that scored 8x better on the median and was optimistic.
set -euo pipefail
HFQ=${1:-/data/hipfire-models/escha-35b.hfq}
SLICE=benchmarks/quality-baselines/slice/wikitext2-1024s-2048ctx.txt
POS=${POS:-192}

echo "== 1/2  hipfire vs escha_ref (CPU oracle) =="
cargo run --release -p hipfire-runtime --example eval_hipfire -- \
  --model "$HFQ" --corpus "$SLICE" --positions "$POS" \
  --reference escha-ref --teacher-forced --report kld

echo "== 2/2  hipfire vs bf16 parent Qwen/Qwen3.6-35B-A3B =="
echo "   (compare this number against the existing qwen3.6:35b-a3b-mq2 result)"
cargo run --release -p hipfire-runtime --example eval_hipfire -- \
  --model "$HFQ" --corpus "$SLICE" --positions "$POS" \
  --reference /data/hipfire-models/qwen3.6-35b-a3b-bf16 \
  --teacher-forced --report kld
```

```bash
chmod +x scripts/escha-kld.sh
./scripts/escha-kld.sh 2>&1 | tail -30
```

`eval_hipfire` currently has no `--reference escha-ref` arm — add one that calls `escha_ref` per position. Teacher forcing is load-bearing and easy to get subtly wrong: two quants diverge at roughly token 0, so an unforced position-wise KL compares unrelated continuations. Force at **both** sites (the pre-loop sampled token and the decode loop) and key on a call counter, not `pos` — the two share a `pos`. Assert the committed token streams are identical; without that assertion a wrong number reads as fact.

- [ ] **Step 5: Record the results**

Expected: comparison 1 near zero, dominated by the `Q8_0` intermediate rather than the codec. A handful of divergent positions traceable to boundary ties in the router is expected and is not a codec bug (§5) — attribute before investigating.

Write both numbers into `docs/plans/escha-w2-port-design.md` under a new "Phase 1 results" section, alongside the measured tok/s. State plainly that Phase 1 loses to `qwen3.6:35b-a3b-mq2` on speed; that is the expected outcome and Phase 2's job.

- [ ] **Step 6: Commit**

```bash
git add registry/v1.json scripts/escha-kld.sh docs/plans/escha-w2-port-design.md
git commit -m "feat(escha): registry entry, coherence and KLD results for Phase 1"
```

---

## Definition of done

- G0–G5 all pass, with G2 and G3 bit-exact and G4 inside the stated tolerances.
- `cargo build --release --workspace --all-targets --locked` is clean.
- `qwen3.6:35b-a3b-escha` serves coherent text at roughly 36.7 GB resident.
- Both KLD numbers are recorded in the design doc.

## Deferred to later plans

- **Phase 2**: fused decode+GEMV; the prune-mask optimisation (skip `down_proj` input rows for the channels `gate_up.rout` zeroes — spec §4.5).
- **Qwen3.8-27B**: 400 escha tensors over 10 projections, the 3-way `FusedQkvQ8_0` path, and arch-5 bias slots.
