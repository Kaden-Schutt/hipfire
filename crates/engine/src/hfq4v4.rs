//! HFQ4v4 (and MQ4v4): gfx12-native 4-bit weight format designed for the
//! iu4 K=32 wmma instruction.
//!
//! Solves the v1 iu4 quality-fail (PR #140) by redesigning the weight format
//! around the constraint that native gfx12 wmma is type-symmetric: iu4 weights
//! force iu4 *activations* with ~14% per-element precision, which on raw FP16
//! activation distribution clips fat tails and cascades into garbage output.
//!
//! ## SmoothQuant transfer
//!
//! Q4_1 activations live in [-7, 7] (signed 4-bit). To make this range usable
//! we redistribute the activation distribution into something Q4_1 can capture:
//!
//!   1. Hoist the per-output-channel mean of the *equivalent FP16* product
//!      `sum_k W[r,k] * X[k]` onto a per-row scalar `mu`. After subtraction,
//!      the residual is mean-centered.
//!   2. Re-quantize the residual into K=32 groups with a single FP16 scale
//!      `d` (no per-group bias — that's gone now, absorbed into per-row mu).
//!   3. At dispatch time, activations are quantized as Q4_1 (existing kernel,
//!      unchanged); the GEMM applies the post-correction:
//!        out[r,c] = d_w[g] * d_a[c] * wmma_acc(W_iu4, X_iu4)
//!                 + mu_w[r] * sum_q_a[c]
//!      where sum_q_a[c] is the per-K=32 sum of activation Q4 values that the
//!      Q4_1 quantizer already produces (we reuse `ds.y / d_a`).
//!
//! ## Layout
//!
//! Per-row, per-K=32 group:
//!     [16 B nibbles] [2 B FP16 d] = 18 B
//!
//! For K=k, groups_per_row = k / 32, so a row is groups_per_row * 18 bytes.
//!
//! Per-row sidecar (separate buffer):
//!     mu : FP16, M values total. M * 2 B.
//!
//! Total bits/weight: 4 + 16/32 = 4.5 b/w (vs 4.25 b/w for HFQ4-G256, +6%).
//!
//! ## Compatibility with FWHT (mq4v4 variant)
//!
//! When `--rotate` is requested, the converter applies an FWHT-32 (5
//! butterfly levels) on each K=32 group of the weight row BEFORE the
//! SmoothQuant analysis. The FWHT is its own inverse up to scaling and is
//! self-cancelling when paired with an FWHT-32 applied to the *activation*
//! K=32 group at runtime — the kernel cycles the activation through an
//! identical FWHT-32 before the Q4_1 quantize pass.
//!
//! On v4 specifically (Q4 acts), FWHT is load-bearing for quality: it
//! diffuses outliers across the K=32 group, producing a near-Gaussian
//! distribution that the SmoothQuant per-row mu absorption makes nearly
//! mean-zero, which Q4_1 [-7, 7] then captures with minimal clipping.

use std::io::{Read, Write};

/// File magic for HFQ4v4 weight blob (per-tensor / standalone use).
pub const HFQ4V4_MAGIC: [u8; 4] = *b"HQ4V";
/// File magic for MQ4v4 (HFQ4v4 + FWHT-32 pre-rotation).
pub const MQ4V4_MAGIC: [u8; 4] = *b"MQ4V";

pub const HFQ4V4_VERSION: u32 = 1;

/// On-disk header for a single HFQ4v4 / MQ4v4 weight tensor blob (used by the
/// CLI converter when emitting one tensor at a time, e.g. for the
/// correctness test harness). Within a `.hfq` archive the per-tensor
/// metadata is stored in the archive index instead.
///
/// Total: 32 bytes.
#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
pub struct Hfq4v4Header {
    pub magic: [u8; 4], // HFQ4V4_MAGIC or MQ4V4_MAGIC
    pub version: u32,
    pub m: u32,         // output dim (rows)
    pub k: u32,         // input dim (cols), MUST be a multiple of 32
    pub group_k: u32,   // always 32 (K=32 per group)
    pub flags: u32,     // bit 0: rotate (FWHT-32) applied
    pub _pad: [u8; 8],
}

const _: () = assert!(std::mem::size_of::<Hfq4v4Header>() == 32);

pub const FLAG_ROTATE_FWHT32: u32 = 1 << 0;

pub const GROUP_K: usize = 32;
pub const BYTES_PER_GROUP: usize = 18; // 16 nibbles + 2 FP16 d

/// Bytes for one row's nibbles+scales blob.
#[inline]
pub fn row_bytes(k: usize) -> usize {
    debug_assert!(k % GROUP_K == 0);
    (k / GROUP_K) * BYTES_PER_GROUP
}

/// Total bytes for the dense weight blob.
#[inline]
pub fn weight_bytes(m: usize, k: usize) -> usize {
    m * row_bytes(k)
}

/// Total bytes for the per-row mu sidecar (FP16 per row).
#[inline]
pub fn mu_bytes(m: usize) -> usize {
    m * 2
}

// ─── FP16 helpers ───────────────────────────────────────────────────────────

/// Plain f32 → f16 (round-to-nearest-even, no subnormal handling beyond
/// what the bit twiddle gives). Mirrors the converter elsewhere in tree.
#[inline]
pub fn f32_to_f16(f: f32) -> u16 {
    let bits = f.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let mut mant = (bits & 0x007F_FFFF) as i32;
    let mut exp = ((bits >> 23) & 0xFF) as i32 - 127 + 15;

    if exp >= 31 {
        // overflow → inf or NaN
        if (bits & 0x7FFF_FFFF) > 0x7F80_0000 {
            return sign | 0x7E00; // qNaN
        }
        return sign | 0x7C00; // inf
    }
    if exp <= 0 {
        // subnormal / underflow → flush to zero (good enough for our use)
        if exp < -10 {
            return sign;
        }
        mant |= 0x0080_0000;
        let shift = 14 - exp;
        let result = (mant >> shift) as u16;
        let round_bit = (mant >> (shift - 1)) & 1;
        return sign | result + (round_bit as u16);
    }
    let result = (((exp as u32) << 10) | ((mant as u32) >> 13)) as u16;
    // round-to-nearest-even
    let round_bit = (mant >> 12) & 1;
    let sticky = (mant & 0x0FFF) != 0;
    let lsb = (result & 1) as i32;
    if round_bit == 1 && (sticky || lsb == 1) {
        return sign | (result + 1);
    }
    sign | result
}

#[inline]
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1F) as u32;
    let mant = (h & 0x3FF) as u32;
    let bits = if exp == 0 {
        if mant == 0 {
            sign << 31
        } else {
            // subnormal → normalize
            let mut m = mant << 1;
            let mut e: u32 = 0;
            while (m & 0x400) == 0 {
                m <<= 1;
                e += 1;
            }
            (sign << 31) | ((127 - 15 - e) << 23) | ((m & 0x3FF) << 13)
        }
    } else if exp == 0x1F {
        (sign << 31) | 0x7F80_0000 | (mant << 13)
    } else {
        (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

// ─── FWHT-32 (5 levels in registers, no signs1/signs2) ──────────────────────

/// In-place FWHT on a 32-element slice. 5 butterfly levels.
///
/// We do NOT use the random sign flips that mq4-G256 uses, because at K=32 the
/// orthogonal Hadamard transform alone is a sufficient outlier diffuser, and
/// the smaller block already gives less room for outliers to dominate.
/// Critically, this means the kernel-side activation FWHT-32 is just the same
/// orthogonal transform with no PRG seed dependency.
///
/// Scale 1/sqrt(32) = 1/(4*sqrt(2)) is applied at the end for orthogonality.
pub fn fwht_32(x: &mut [f32; 32]) {
    let mut stride = 1usize;
    while stride < 32 {
        let mut i = 0usize;
        while i < 32 {
            for j in 0..stride {
                let a = x[i + j];
                let b = x[i + j + stride];
                x[i + j] = a + b;
                x[i + j + stride] = a - b;
            }
            i += stride * 2;
        }
        stride <<= 1;
    }
    let scale = 1.0f32 / 32f32.sqrt(); // ~0.17677669
    for v in x.iter_mut() {
        *v *= scale;
    }
}

// ─── Quantization ───────────────────────────────────────────────────────────

/// Symmetric Q4 (signed) quantization for a single K=32 group.
///
/// Returns:
///   - `q`: 32 nibbles in i8 range [-7, 7]
///   - `d`: scale (positive)
///
/// `q[i] = clamp(round(x[i] / d), -7, 7)` with `d = max(|x|) / 7`.
fn quantize_group_sym4(x: &[f32; 32]) -> ([i8; 32], f32) {
    let mut amax = 0.0f32;
    for &v in x {
        let a = v.abs();
        if a > amax {
            amax = a;
        }
    }
    let d = if amax > 0.0 { amax / 7.0 } else { 1.0e-8 };
    let inv_d = 1.0 / d;
    let mut q = [0i8; 32];
    for i in 0..32 {
        let qi = (x[i] * inv_d).round();
        q[i] = qi.clamp(-7.0, 7.0) as i8;
    }
    (q, d)
}

/// Pack 32 signed nibbles into 16 bytes. Within a byte: low nibble = element
/// 0 of the pair, high nibble = element 1. Signed nibble stored as the low 4
/// bits of the 8-bit two's-complement int (the kernel sign-extends via
/// `(int8_t)(byte << 4) >> 4` and `(int8_t)(byte) >> 4`).
fn pack_nibbles_signed(q: &[i8; 32]) -> [u8; 16] {
    let mut out = [0u8; 16];
    for i in 0..16 {
        let lo = (q[2 * i] as u8) & 0x0F;
        let hi = (q[2 * i + 1] as u8) & 0x0F;
        out[i] = lo | (hi << 4);
    }
    out
}

/// Convert HFQ4-G256 (FP32 d + FP32 m + 128 B nibbles per K=256) into HFQ4v4
/// (per-K=32 FP16 d + 16 B nibbles, plus per-row FP16 mu sidecar).
///
/// `rotate`: when true, FWHT-32 is applied to each K=32 group of the
/// dequantized weight row before SmoothQuant analysis (the mq4v4 variant).
///
/// `mu_strategy`:
///   - `MuStrategy::WeightMean` — mu is the per-row mean of the weight row
///     (after FWHT, if rotated). Cheap; works empirically when activations are
///     near-zero-mean (post-RMSNorm). This is the v4 default.
///   - `MuStrategy::Calibration { x_mean }` — mu is computed as
///     `sum_k W[r, k] * x_mean[k]`, the SmoothQuant-canonical formulation.
///     The x_mean vector is supplied externally (calibration pass).
///
/// On output:
///   - `weight_blob` length = m * row_bytes(k) bytes.
///   - `mu_blob` length = mu_bytes(m) bytes (FP16 per row).
pub enum MuStrategy<'a> {
    WeightMean,
    Calibration { x_mean: &'a [f32] },
}

pub fn convert_hfq4g256_to_hfq4v4(
    hfq4g256: &[u8],
    m: usize,
    k: usize,
    rotate: bool,
    mu_strategy: &MuStrategy,
) -> (Vec<u8>, Vec<u8>) {
    assert_eq!(k % 256, 0, "HFQ4-G256 source needs k % 256 == 0, got k={k}");
    assert_eq!(k % 32, 0);
    let groups_per_row_g256 = k / 256;
    let row_bytes_in = groups_per_row_g256 * 136;
    assert_eq!(
        hfq4g256.len(),
        m * row_bytes_in,
        "expected {} bytes (m={m} k={k} g256), got {}",
        m * row_bytes_in,
        hfq4g256.len()
    );

    if let MuStrategy::Calibration { x_mean } = mu_strategy {
        assert_eq!(x_mean.len(), k, "calibration x_mean must be length k={k}");
    }

    let mut weight_blob = vec![0u8; weight_bytes(m, k)];
    let mut mu_blob = vec![0u8; mu_bytes(m)];

    let groups_per_row_v4 = k / GROUP_K;

    for r in 0..m {
        // 1. Dequantize one full row from HFQ4-G256 to f32.
        let mut row = vec![0f32; k];
        for g in 0..groups_per_row_g256 {
            let off = (r * groups_per_row_g256 + g) * 136;
            let scale =
                f32::from_le_bytes(hfq4g256[off..off + 4].try_into().unwrap());
            let zero =
                f32::from_le_bytes(hfq4g256[off + 4..off + 8].try_into().unwrap());
            let nib_off = off + 8;
            for i in 0..128 {
                let byte = hfq4g256[nib_off + i];
                let lo = (byte & 0x0F) as f32;
                let hi = ((byte >> 4) & 0x0F) as f32;
                row[g * 256 + 2 * i] = scale * lo + zero;
                row[g * 256 + 2 * i + 1] = scale * hi + zero;
            }
        }

        // 2. Optional FWHT-32 on each K=32 group.
        if rotate {
            for g in 0..groups_per_row_v4 {
                let mut buf = [0f32; 32];
                buf.copy_from_slice(&row[g * 32..g * 32 + 32]);
                fwht_32(&mut buf);
                row[g * 32..g * 32 + 32].copy_from_slice(&buf);
            }
        }

        // 3. Compute per-row mu and subtract.
        let mu_f32: f32 = match mu_strategy {
            MuStrategy::WeightMean => {
                let s: f64 = row.iter().map(|&v| v as f64).sum();
                (s / k as f64) as f32
            }
            MuStrategy::Calibration { x_mean } => {
                // mu_r = sum_k w[r,k] * x_mean[k] (the SmoothQuant residual).
                // The runtime correction will multiply by sum_q_a[c] (which
                // already encodes the scaled activation magnitude per
                // K=32). We store mu *normalized to* per-K=32-element-mean
                // — i.e. we divide by k so that mu * sum_q_a[c] is the
                // contribution of "average activation row × W row" per
                // K=32 sub-block. This matches the kernel correction
                // formula `mu_w[r] * sum_q_a[c]` where sum_q_a is summed
                // over a K=32 sub-block, not over the full K.
                //
                // Residual is then `w[r,k] - (mu_r / k) * 1` so per-K=32
                // sum_k w_residual = sum_k w - mu_r/k * 32 = sum_k w
                // - 32/k * <w·x_mean>. Approximation: subtract mu uniformly
                // (channel-uniform).
                let s: f64 = row
                    .iter()
                    .zip(x_mean.iter())
                    .map(|(&w, &x)| (w as f64) * (x as f64))
                    .sum();
                (s / k as f64) as f32
            }
        };
        for v in row.iter_mut() {
            *v -= mu_f32;
        }
        let mu_h = f32_to_f16(mu_f32);
        mu_blob[2 * r..2 * r + 2].copy_from_slice(&mu_h.to_le_bytes());

        // 4. Re-quantize each K=32 group (symmetric INT4) and pack into the
        //    output row.
        let row_off = r * row_bytes(k);
        for g in 0..groups_per_row_v4 {
            let mut buf = [0f32; 32];
            buf.copy_from_slice(&row[g * 32..g * 32 + 32]);
            let (q, d) = quantize_group_sym4(&buf);
            let nibs = pack_nibbles_signed(&q);
            let go = row_off + g * BYTES_PER_GROUP;
            weight_blob[go..go + 16].copy_from_slice(&nibs);
            let d_h = f32_to_f16(d);
            weight_blob[go + 16..go + 18].copy_from_slice(&d_h.to_le_bytes());
        }
    }

    (weight_blob, mu_blob)
}

// ─── Standalone-blob serialization (for the correctness test harness) ───────

pub fn write_blob(
    out: &mut impl Write,
    m: usize,
    k: usize,
    rotated: bool,
    weight_blob: &[u8],
    mu_blob: &[u8],
) -> std::io::Result<()> {
    let magic = if rotated { MQ4V4_MAGIC } else { HFQ4V4_MAGIC };
    let mut flags: u32 = 0;
    if rotated {
        flags |= FLAG_ROTATE_FWHT32;
    }
    let hdr = Hfq4v4Header {
        magic,
        version: HFQ4V4_VERSION,
        m: m as u32,
        k: k as u32,
        group_k: GROUP_K as u32,
        flags,
        _pad: [0; 8],
    };
    let hdr_bytes: [u8; 32] = unsafe { std::mem::transmute(hdr) };
    out.write_all(&hdr_bytes)?;
    out.write_all(weight_blob)?;
    out.write_all(mu_blob)?;
    Ok(())
}

pub struct LoadedBlob {
    pub m: usize,
    pub k: usize,
    pub rotated: bool,
    pub weights: Vec<u8>,
    pub mu: Vec<u8>,
}

pub fn read_blob(rdr: &mut impl Read) -> std::io::Result<LoadedBlob> {
    let mut hdr_bytes = [0u8; 32];
    rdr.read_exact(&mut hdr_bytes)?;
    let hdr: Hfq4v4Header = unsafe { std::mem::transmute(hdr_bytes) };
    let magic = hdr.magic;
    let version = hdr.version;
    let m = hdr.m as usize;
    let k = hdr.k as usize;
    let group_k = hdr.group_k as usize;
    let flags = hdr.flags;
    if magic != HFQ4V4_MAGIC && magic != MQ4V4_MAGIC {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("bad magic: {:?}", magic),
        ));
    }
    if version != HFQ4V4_VERSION {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("bad version: {version}"),
        ));
    }
    if group_k != GROUP_K {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("group_k must be {GROUP_K}, got {group_k}"),
        ));
    }
    let mut weights = vec![0u8; weight_bytes(m, k)];
    rdr.read_exact(&mut weights)?;
    let mut mu = vec![0u8; mu_bytes(m)];
    rdr.read_exact(&mut mu)?;
    Ok(LoadedBlob {
        m,
        k,
        rotated: (flags & FLAG_ROTATE_FWHT32) != 0,
        weights,
        mu,
    })
}

// ─── Small sanity tests ─────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fwht_32_self_inverse() {
        // Up to scaling: applying FWHT twice and dividing by 1 (since we
        // already scale by 1/sqrt(32) per pass, applying twice scales by
        // 1/32) should give back the input.
        let mut x = [0f32; 32];
        for i in 0..32 {
            x[i] = ((i as i32 * 37) % 13 - 6) as f32 * 0.1 + 0.05;
        }
        let orig = x;
        fwht_32(&mut x);
        fwht_32(&mut x);
        // Two passes scaled by (1/sqrt(32))^2 = 1/32 each, plus the Hadamard
        // matrix squared = 32 I. Net: identity.
        for i in 0..32 {
            assert!((x[i] - orig[i]).abs() < 1e-4, "fwht round-trip failed at {i}");
        }
    }

    #[test]
    fn quantize_roundtrip_constant() {
        let x = [0.5f32; 32];
        let (q, d) = quantize_group_sym4(&x);
        // 0.5 / (0.5/7) = 7 → all values clamp to 7.
        for &qi in &q {
            assert_eq!(qi, 7);
        }
        assert!((d - 0.5 / 7.0).abs() < 1e-6);
    }

    #[test]
    fn weight_bytes_layout() {
        assert_eq!(row_bytes(32), 18);
        assert_eq!(row_bytes(256), 8 * 18);
        assert_eq!(weight_bytes(2, 64), 2 * 2 * 18);
        assert_eq!(mu_bytes(128), 256);
    }

    #[test]
    fn header_size_locked() {
        assert_eq!(std::mem::size_of::<Hfq4v4Header>(), 32);
    }

    #[test]
    fn convert_smoke() {
        // Make a deterministic 4-row × K=512 HFQ4-G256 weight blob.
        let m = 4;
        let k = 512;
        let groups = k / 256;
        let mut blob = vec![0u8; m * groups * 136];
        for r in 0..m {
            for g in 0..groups {
                let o = (r * groups + g) * 136;
                let scale = 0.01_f32 + r as f32 * 0.001;
                let zero = -0.05_f32;
                blob[o..o + 4].copy_from_slice(&scale.to_le_bytes());
                blob[o + 4..o + 8].copy_from_slice(&zero.to_le_bytes());
                for i in 0..128 {
                    blob[o + 8 + i] = ((r * 31 + g * 7 + i) & 0xFF) as u8;
                }
            }
        }
        let (w_blob, mu_blob) = convert_hfq4g256_to_hfq4v4(
            &blob, m, k, false, &MuStrategy::WeightMean,
        );
        assert_eq!(w_blob.len(), weight_bytes(m, k));
        assert_eq!(mu_blob.len(), mu_bytes(m));
        // Round-trip via blob serializer.
        let mut buf = Vec::new();
        write_blob(&mut buf, m, k, false, &w_blob, &mu_blob).unwrap();
        let loaded = read_blob(&mut std::io::Cursor::new(buf)).unwrap();
        assert_eq!(loaded.m, m);
        assert_eq!(loaded.k, k);
        assert_eq!(loaded.rotated, false);
        assert_eq!(loaded.weights, w_blob);
        assert_eq!(loaded.mu, mu_blob);
    }

    #[test]
    fn convert_with_rotation() {
        let m = 2;
        let k = 256;
        let groups = 1;
        let mut blob = vec![0u8; m * groups * 136];
        for r in 0..m {
            let o = r * 136;
            let scale = 0.02f32;
            let zero = 0.0f32;
            blob[o..o + 4].copy_from_slice(&scale.to_le_bytes());
            blob[o + 4..o + 8].copy_from_slice(&zero.to_le_bytes());
            for i in 0..128 {
                blob[o + 8 + i] = ((r * 11 + i * 3) & 0xFF) as u8;
            }
        }
        let (w_blob, _) = convert_hfq4g256_to_hfq4v4(
            &blob, m, k, true, &MuStrategy::WeightMean,
        );
        assert_eq!(w_blob.len(), weight_bytes(m, k));
    }
}
