//! SmoothQuant-style activation calibration sidecar for the gfx12 iu4 K=32
//! GEMM path.
//!
//! # Mathematical recipe
//!
//! Given an iu4 GEMM call site `y = W·x` (W shape [m × k], x shape [n × k]):
//!
//! ```text
//!   y = W·x = (W·diag(s_a)) · (diag(1/s_a)·(x - mu_a)) + W·mu_a
//! ```
//!
//! - `mu_a[col]`: per-input-channel arithmetic mean of the activation `x`
//!   (averaged across calibration tokens). Shape: [k].
//! - `s_a[col]`: per-input-channel scale (e.g. 99-percentile-abs after
//!   centering). Shape: [k].
//! - `W·diag(s_a)`: bake `s_a` into the per-row weight scale at load time.
//!   Per HFQ4-G256, each weight row has `groups_per_row` (= k/256) FP32
//!   scales; we multiply each scale by the geometric mean of `s_a` over
//!   that K=256 group. (Per-channel scale baking would require per-element
//!   weight rescale which is incompatible with the group-shared HFQ4
//!   representation; group-mean baking is the practical SmoothQuant
//!   approximation used in production stacks.)
//! - `(x - mu_a) / s_a`: runtime preshift kernel BEFORE the Q4_1 quantizer.
//!   Now activations are mean-zero and roughly unit-variance per channel,
//!   so symmetric Q4_1 captures the bulk of the distribution.
//! - `W·mu_a`: precomputed at calibration time. One MV per call site against
//!   the `mu_a` vector → `bias[m]`. Stored in the sidecar; added to the
//!   GEMM output once.
//!
//! # Sidecar wire format
//!
//! Little-endian throughout.
//!
//! ```text
//!   magic:    u32 = 0x49553443  ("IU4C")
//!   version:  u32 = 1
//!   n_sites:  u32   number of GEMM call sites
//!   _pad:     u32   = 0 (alignment / future use)
//!
//!   per site (in dispatch-counter order):
//!     layer_idx:    u32   logical layer index
//!     proj_id:      u32   0=wo, 1=w_down (extend if more sites added)
//!     n_channels:   u32   = k (input channel dim at this site)
//!     n_output_rows:u32   = m (output dim at this site)
//!     mu_a:         [f16; n_channels]
//!     s_a:          [f16; n_channels]
//!     w_mu_bias:    [f16; n_output_rows]
//! ```
//!
//! ## Why FP16
//!
//! Calibration vectors are O(L * K) which on 27B (L=64, K=4096) is small
//! (~32 MB at FP16). FP16 is enough precision for centering / scaling
//! statistics — calibration noise dominates round-off. The runtime preshift
//! kernel reads FP16 mu/inv_s and emits FP32 (which goes into the existing
//! Q4_1 quantizer that already takes FP32 in).
//!
//! # Site identity
//!
//! Site `i` corresponds to the i-th call to `Gpu::gemm_hfq4g256_residual`
//! through the dense forward path. For Qwen3.5 dense, that's
//! `(layer_idx, proj_id) = (i / 2, i % 2)` with `proj_id ∈ {0=wo, 1=w_down}`.
//! The dispatcher's `dispatch_call_idx` counter (per-process, increments on
//! every call regardless of arch / iu4 gating) is the canonical site key.

use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::Path;

use hip_bridge::HipResult;
use rdna_compute::{Gpu, GpuIu4CalSite, GpuIu4Calibration};

use crate::llama::{f16_to_f32, f32_to_f16};

pub const IU4CAL_MAGIC: u32 = 0x49553443; // "IU4C"
pub const IU4CAL_VERSION: u32 = 1;

/// One calibration site: per-channel mu, s, plus the precomputed W·mu_a bias.
#[derive(Clone, Debug)]
pub struct Iu4CalSite {
    /// Logical layer index (informational; the runtime keys by call order).
    pub layer_idx: u32,
    /// Projection identifier within the layer. Convention:
    ///   0 = wo  (attention output projection)
    ///   1 = w_down  (FFN down projection)
    pub proj_id: u32,
    /// Input channel dim (= k).
    pub n_channels: u32,
    /// Output dim (= m).
    pub n_output_rows: u32,
    /// Per-input-channel activation mean. FP16 storage. Length = n_channels.
    pub mu_a: Vec<u16>,
    /// Per-input-channel activation scale (e.g. 99-pctile-abs). FP16 storage.
    /// Length = n_channels.
    pub s_a: Vec<u16>,
    /// Precomputed W·mu_a bias. FP16 storage. Length = n_output_rows.
    pub w_mu_bias: Vec<u16>,
}

impl Iu4CalSite {
    pub fn new(layer_idx: u32, proj_id: u32, n_channels: usize, n_output_rows: usize) -> Self {
        Self {
            layer_idx,
            proj_id,
            n_channels: n_channels as u32,
            n_output_rows: n_output_rows as u32,
            mu_a: vec![0u16; n_channels],
            s_a: vec![0u16; n_channels],
            w_mu_bias: vec![0u16; n_output_rows],
        }
    }

    /// Decode `mu_a` to FP32. Allocates a new Vec.
    pub fn mu_a_f32(&self) -> Vec<f32> {
        self.mu_a.iter().map(|&b| f16_to_f32(b)).collect()
    }

    /// Decode `s_a` to FP32 with a small floor to avoid divide-by-zero in
    /// inv_s computation. Returns `(s_a, inv_s_a)` so callers don't repeat
    /// the floor logic.
    pub fn s_a_f32_with_inv(&self, floor: f32) -> (Vec<f32>, Vec<f32>) {
        let s: Vec<f32> = self
            .s_a
            .iter()
            .map(|&b| {
                let v = f16_to_f32(b);
                if v.is_finite() && v > floor {
                    v
                } else {
                    floor
                }
            })
            .collect();
        let inv: Vec<f32> = s.iter().map(|&v| 1.0 / v).collect();
        (s, inv)
    }

    /// Decode `w_mu_bias` to FP32.
    pub fn w_mu_bias_f32(&self) -> Vec<f32> {
        self.w_mu_bias.iter().map(|&b| f16_to_f32(b)).collect()
    }
}

#[derive(Clone, Debug, Default)]
pub struct Iu4Calibration {
    pub sites: Vec<Iu4CalSite>,
}

impl Iu4Calibration {
    pub fn new() -> Self {
        Self { sites: Vec::new() }
    }

    /// Look up the site for a given dispatch call index. Returns `None` if
    /// out of range (caller must fall back to non-calibrated path).
    pub fn site_at(&self, call_idx: usize) -> Option<&Iu4CalSite> {
        self.sites.get(call_idx)
    }

    /// Return the number of recorded sites.
    pub fn n_sites(&self) -> usize {
        self.sites.len()
    }

    /// Serialize to disk in the wire format documented at the top of the file.
    pub fn write_to<W: Write>(&self, mut w: W) -> io::Result<()> {
        w.write_all(&IU4CAL_MAGIC.to_le_bytes())?;
        w.write_all(&IU4CAL_VERSION.to_le_bytes())?;
        w.write_all(&(self.sites.len() as u32).to_le_bytes())?;
        w.write_all(&0u32.to_le_bytes())?;
        for site in &self.sites {
            w.write_all(&site.layer_idx.to_le_bytes())?;
            w.write_all(&site.proj_id.to_le_bytes())?;
            w.write_all(&site.n_channels.to_le_bytes())?;
            w.write_all(&site.n_output_rows.to_le_bytes())?;
            for &h in &site.mu_a {
                w.write_all(&h.to_le_bytes())?;
            }
            for &h in &site.s_a {
                w.write_all(&h.to_le_bytes())?;
            }
            for &h in &site.w_mu_bias {
                w.write_all(&h.to_le_bytes())?;
            }
        }
        Ok(())
    }

    /// Convenience: open path and write.
    pub fn write_path(&self, path: &Path) -> io::Result<()> {
        let f = std::fs::File::create(path)?;
        let mut w = std::io::BufWriter::new(f);
        self.write_to(&mut w)?;
        w.flush()?;
        Ok(())
    }

    /// Deserialize from a reader. Validates magic + version + size. Each
    /// site's mu/s/bias arrays are read straight into FP16 storage.
    pub fn read_from<R: Read + Seek>(mut r: R) -> io::Result<Self> {
        let mut buf4 = [0u8; 4];
        r.read_exact(&mut buf4)?;
        let magic = u32::from_le_bytes(buf4);
        if magic != IU4CAL_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("iu4cal: bad magic 0x{magic:08x}, want 0x{IU4CAL_MAGIC:08x}"),
            ));
        }
        r.read_exact(&mut buf4)?;
        let version = u32::from_le_bytes(buf4);
        if version != IU4CAL_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("iu4cal: unsupported version {version}, want {IU4CAL_VERSION}"),
            ));
        }
        r.read_exact(&mut buf4)?;
        let n_sites = u32::from_le_bytes(buf4) as usize;
        r.read_exact(&mut buf4)?; // pad

        let mut sites = Vec::with_capacity(n_sites);
        for _ in 0..n_sites {
            r.read_exact(&mut buf4)?;
            let layer_idx = u32::from_le_bytes(buf4);
            r.read_exact(&mut buf4)?;
            let proj_id = u32::from_le_bytes(buf4);
            r.read_exact(&mut buf4)?;
            let n_channels = u32::from_le_bytes(buf4);
            r.read_exact(&mut buf4)?;
            let n_output_rows = u32::from_le_bytes(buf4);

            let nc = n_channels as usize;
            let nm = n_output_rows as usize;
            let mut mu_a = vec![0u16; nc];
            let mut s_a = vec![0u16; nc];
            let mut w_mu_bias = vec![0u16; nm];
            // Bulk-read raw little-endian bytes; on LE hosts (x86_64,
            // aarch64) we could borrow + transmute, but the explicit
            // byteswap is portable and not in the inference hot path.
            let mut bytes = vec![0u8; nc * 2];
            r.read_exact(&mut bytes)?;
            for (i, h) in mu_a.iter_mut().enumerate() {
                *h = u16::from_le_bytes([bytes[2 * i], bytes[2 * i + 1]]);
            }
            r.read_exact(&mut bytes)?;
            for (i, h) in s_a.iter_mut().enumerate() {
                *h = u16::from_le_bytes([bytes[2 * i], bytes[2 * i + 1]]);
            }
            let mut bias_bytes = vec![0u8; nm * 2];
            r.read_exact(&mut bias_bytes)?;
            for (i, h) in w_mu_bias.iter_mut().enumerate() {
                *h = u16::from_le_bytes([bias_bytes[2 * i], bias_bytes[2 * i + 1]]);
            }

            sites.push(Iu4CalSite {
                layer_idx,
                proj_id,
                n_channels,
                n_output_rows,
                mu_a,
                s_a,
                w_mu_bias,
            });
        }

        // Sanity: don't blow up on trailing bytes; the format is fixed-width
        // per site and we don't need to seek-to-end-check.
        let _ = r.seek(SeekFrom::Current(0));
        Ok(Self { sites })
    }

    /// Convenience: open path and read.
    pub fn read_path(path: &Path) -> io::Result<Self> {
        let f = std::fs::File::open(path)?;
        let mut r = std::io::BufReader::new(f);
        Self::read_from(&mut r)
    }
}

// ---------------------------------------------------------------------------
// Streaming statistics accumulator. One per call site. Used by the
// calibration binary to fold per-batch mu/s contributions in.
//
// Tracks:
//   - count: total tokens processed
//   - sum:   sum_t x[t][col]                  → mu = sum / count
//   - centered_abs_sum: sum_t |x[t][col] - mu_running[col]|  (running)
//   - reservoir for percentile estimation: bucket histogram of |x - mu|
//
// We use a two-pass strategy in the calibration binary: first pass collects
// mu, second pass collects centered-abs distribution. That's simpler than
// online-streaming percentile estimation and the calibration corpus is
// small enough (~64K tokens) that two FP16-forward passes is cheap.

/// Per-channel running sums for first-pass mean estimation.
pub struct MeanAccumulator {
    /// Sum of x[t][col] across calibration tokens.
    pub sum: Vec<f64>,
    /// Number of tokens accumulated.
    pub count: u64,
}

impl MeanAccumulator {
    pub fn new(n_channels: usize) -> Self {
        Self {
            sum: vec![0.0; n_channels],
            count: 0,
        }
    }

    pub fn add_batch(&mut self, x: &[f32], n_tokens: usize) {
        let nc = self.sum.len();
        debug_assert_eq!(x.len(), n_tokens * nc);
        for t in 0..n_tokens {
            let row = &x[t * nc..(t + 1) * nc];
            for (col, &v) in row.iter().enumerate() {
                self.sum[col] += v as f64;
            }
        }
        self.count += n_tokens as u64;
    }

    pub fn mean(&self) -> Vec<f32> {
        let n = self.count.max(1) as f64;
        self.sum.iter().map(|&s| (s / n) as f32).collect()
    }
}

/// Per-channel histogram for percentile-abs estimation against a fixed mu.
/// Uses log-spaced bins from `EPS_LO` to `MAX_HI`. Order-of-magnitude bin
/// resolution is fine for 99-percentile selection.
pub struct CenteredHistogram {
    pub mu: Vec<f32>,
    /// Per-channel histogram. Shape [n_channels × NBUCKETS].
    /// Bucket i covers |z| in [BIN_EDGES[i], BIN_EDGES[i+1]).
    pub hist: Vec<u32>,
    /// Per-channel running max-abs (for fallback / floor).
    pub max_abs: Vec<f32>,
    /// Per-channel running sum-abs (for mean-abs s_a strategy).
    pub sum_abs: Vec<f64>,
    pub count: u64,
}

pub const NBUCKETS: usize = 64;

/// Log-spaced bin edges from 1e-4 to 1e4 (8 decades, 8 buckets per decade).
/// Bucket index for value v: `((v.log10() + 4.0) * 8.0).floor()` clamped to
/// [0, NBUCKETS). Out-of-range high goes to the last bucket; ≤0 is undefined
/// (we accumulate |z| > 0 only).
pub fn bin_edges() -> [f32; NBUCKETS + 1] {
    let mut edges = [0.0f32; NBUCKETS + 1];
    for i in 0..=NBUCKETS {
        let log10 = -4.0 + (i as f32) / 8.0;
        edges[i] = 10f32.powf(log10);
    }
    edges
}

pub fn bucket_for(abs_v: f32) -> usize {
    if !(abs_v > 0.0) {
        return 0;
    }
    let log10 = abs_v.log10();
    let raw = ((log10 + 4.0) * 8.0).floor() as i32;
    raw.clamp(0, NBUCKETS as i32 - 1) as usize
}

impl CenteredHistogram {
    pub fn new(mu: Vec<f32>) -> Self {
        let nc = mu.len();
        Self {
            mu,
            hist: vec![0u32; nc * NBUCKETS],
            max_abs: vec![0.0f32; nc],
            sum_abs: vec![0.0f64; nc],
            count: 0,
        }
    }

    pub fn n_channels(&self) -> usize {
        self.mu.len()
    }

    pub fn add_batch(&mut self, x: &[f32], n_tokens: usize) {
        let nc = self.mu.len();
        debug_assert_eq!(x.len(), n_tokens * nc);
        for t in 0..n_tokens {
            let row = &x[t * nc..(t + 1) * nc];
            for (col, &v) in row.iter().enumerate() {
                let z = (v - self.mu[col]).abs();
                self.sum_abs[col] += z as f64;
                if z > self.max_abs[col] {
                    self.max_abs[col] = z;
                }
                let b = bucket_for(z);
                self.hist[col * NBUCKETS + b] = self.hist[col * NBUCKETS + b].saturating_add(1);
            }
        }
        self.count += n_tokens as u64;
    }

    /// Estimate the per-channel `pctile`-th percentile of |x - mu|. Returns
    /// the upper-edge of the bin where the cumulative fraction crosses the
    /// percentile (a slight over-estimate, which is the conservative side for
    /// Q4 fitting — we'd rather widen the bucket than clip too aggressively).
    pub fn percentile_abs(&self, pctile: f32) -> Vec<f32> {
        let edges = bin_edges();
        let nc = self.n_channels();
        let mut out = Vec::with_capacity(nc);
        for col in 0..nc {
            let total: u64 = self.hist[col * NBUCKETS..(col + 1) * NBUCKETS]
                .iter()
                .map(|&v| v as u64)
                .sum();
            if total == 0 {
                out.push(0.0);
                continue;
            }
            let target = (total as f32 * pctile / 100.0) as u64;
            let mut cum: u64 = 0;
            let mut sel: usize = NBUCKETS - 1;
            for b in 0..NBUCKETS {
                cum += self.hist[col * NBUCKETS + b] as u64;
                if cum >= target {
                    sel = b;
                    break;
                }
            }
            // Use upper edge of bin for percentile estimate, capped by
            // observed max_abs (small bin counts can over-shoot).
            let est = edges[sel + 1].min(self.max_abs[col].max(edges[sel]));
            out.push(est);
        }
        out
    }

    pub fn mean_abs(&self) -> Vec<f32> {
        let n = self.count.max(1) as f64;
        self.sum_abs.iter().map(|&s| (s / n) as f32).collect()
    }
}

// ---------------------------------------------------------------------------
// Encoding helpers — convert FP32 calibration vectors into FP16 storage in
// place.

pub fn f32_vec_to_f16(src: &[f32]) -> Vec<u16> {
    src.iter().map(|&v| f32_to_f16(v)).collect()
}

// ---------------------------------------------------------------------------
// GPU upload — convert sidecar to GPU-resident form.

/// Compute per-K=256-group activation scales by averaging per-channel s_a
/// values within each group. Used for the SmoothQuant weight-side bake
/// (the iu4 GEMM's HFQ4-G256 layout has a single FP32 scale per group, so
/// the s_a migration must operate at group resolution).
///
/// Returns `s_group` with length `n_channels / 256` (must be exact). Uses
/// the geometric mean by default — matches how s values are commonly
/// composed in SmoothQuant. Falls back to arithmetic mean if any per-channel
/// s_a is non-positive (avoids log(0) blowing up).
pub fn s_group_from_per_channel(s_a: &[f32]) -> Vec<f32> {
    const GROUP: usize = 256;
    assert!(s_a.len() % GROUP == 0, "s_a length must be divisible by 256");
    let n_groups = s_a.len() / GROUP;
    let mut out = Vec::with_capacity(n_groups);
    for g in 0..n_groups {
        let slice = &s_a[g * GROUP..(g + 1) * GROUP];
        let any_nonpos = slice.iter().any(|&v| !(v > 0.0) || !v.is_finite());
        if any_nonpos {
            // Arithmetic mean as fallback; clamp to a small positive floor
            // so the inv_s preshift doesn't blow up.
            let sum: f64 = slice.iter().map(|&v| v as f64).sum();
            let mean = (sum / GROUP as f64) as f32;
            out.push(mean.max(1e-6));
        } else {
            // Geometric mean: exp(mean(log(s))).
            let log_sum: f64 = slice.iter().map(|&v| (v as f64).ln()).sum();
            let g_mean = ((log_sum / GROUP as f64).exp()) as f32;
            out.push(g_mean.max(1e-6));
        }
    }
    out
}

/// Replace per-channel s_a with the group-broadcast version: for each
/// channel c, `s_a_grouped[c] = s_group[group(c)]`. Used so the activation
/// preshift's per-channel `inv_s_a` exactly inverts the weight-side bake's
/// per-group `s_group` — the math closes at group granularity.
pub fn broadcast_s_group_to_channels(s_group: &[f32]) -> Vec<f32> {
    const GROUP: usize = 256;
    let mut out = Vec::with_capacity(s_group.len() * GROUP);
    for &v in s_group {
        for _ in 0..GROUP {
            out.push(v);
        }
    }
    out
}

const INV_S_FLOOR: f32 = 1e-6;

/// Upload an in-memory `Iu4Calibration` sidecar to GPU memory. Allocates
/// per-site FP16 mu/inv_s/s_group buffers and an FP32 W·mu_a bias buffer.
///
/// At upload time we apply the group-shared s_a transform: per-channel
/// `s_a` is reduced to per-K=256-group `s_group` (geometric mean), then
/// broadcast back to per-channel for the preshift kernel. This guarantees
/// the math closes exactly at group resolution against the weight-side
/// bake.
///
/// The W·mu_a bias is uploaded as-is (FP16 source on disk → FP16-decoded
/// FP32 → uploaded as FP32 bytes for the bias_add_f32 kernel).
pub fn upload_to_gpu(cal: &Iu4Calibration, gpu: &mut Gpu) -> HipResult<GpuIu4Calibration> {
    let mut sites_gpu = Vec::with_capacity(cal.sites.len());
    for site in &cal.sites {
        let nc = site.n_channels as usize;
        let nm = site.n_output_rows as usize;
        if nc % 256 != 0 {
            return Err(hip_bridge::HipError::new(0, &format!(
                "iu4 calibration site n_channels={} not divisible by 256 \
                 — HFQ4-G256 group bake requires K alignment.", nc
            )));
        }
        let groups_per_row = nc / 256;

        // Decode per-channel s_a and reduce to per-group, then broadcast
        // back so preshift uses group-constant per-channel scales.
        let s_a_raw_f32: Vec<f32> = site.s_a.iter().map(|&b| f16_to_f32(b)).collect();
        let s_group_f32 = s_group_from_per_channel(&s_a_raw_f32);
        debug_assert_eq!(s_group_f32.len(), groups_per_row);
        let s_a_grouped_f32 = broadcast_s_group_to_channels(&s_group_f32);
        debug_assert_eq!(s_a_grouped_f32.len(), nc);

        // Compute per-channel inv_s with a small floor (avoids divide-by-
        // zero in degenerate channels with no calibration signal).
        let inv_s_grouped_f32: Vec<f32> = s_a_grouped_f32
            .iter()
            .map(|&v| if v > INV_S_FLOOR { 1.0 / v } else { 1.0 / INV_S_FLOOR })
            .collect();

        // Encode for upload.
        let mu_a_fp16: Vec<u16> = site.mu_a.clone();
        let inv_s_a_fp16: Vec<u16> = inv_s_grouped_f32.iter().map(|&v| f32_to_f16(v)).collect();
        let s_group_fp16: Vec<u16> = s_group_f32.iter().map(|&v| f32_to_f16(v)).collect();
        let w_mu_bias_f32: Vec<f32> = site.w_mu_bias.iter().map(|&b| f16_to_f32(b)).collect();

        // Upload to GPU.
        let mu_a_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(mu_a_fp16.as_ptr() as *const u8, mu_a_fp16.len() * 2)
        };
        let inv_s_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(inv_s_a_fp16.as_ptr() as *const u8, inv_s_a_fp16.len() * 2)
        };
        let s_group_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(s_group_fp16.as_ptr() as *const u8, s_group_fp16.len() * 2)
        };
        let bias_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(w_mu_bias_f32.as_ptr() as *const u8, w_mu_bias_f32.len() * 4)
        };

        let mu_a_buf = gpu.hip.malloc(mu_a_bytes.len())?;
        gpu.hip.memcpy_htod(&mu_a_buf, mu_a_bytes)?;
        let inv_s_buf = gpu.hip.malloc(inv_s_bytes.len())?;
        gpu.hip.memcpy_htod(&inv_s_buf, inv_s_bytes)?;
        let s_group_buf = gpu.hip.malloc(s_group_bytes.len())?;
        gpu.hip.memcpy_htod(&s_group_buf, s_group_bytes)?;
        let bias_buf = gpu.hip.malloc(bias_bytes.len())?;
        gpu.hip.memcpy_htod(&bias_buf, bias_bytes)?;

        sites_gpu.push(GpuIu4CalSite {
            layer_idx: site.layer_idx,
            proj_id: site.proj_id,
            n_channels: site.n_channels,
            n_output_rows: site.n_output_rows,
            groups_per_row: groups_per_row as u32,
            mu_a: mu_a_buf,
            inv_s_a: inv_s_buf,
            s_group: s_group_buf,
            w_mu_bias_f32: bias_buf,
        });
    }
    Ok(GpuIu4Calibration { sites: sites_gpu })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_empty() {
        let cal = Iu4Calibration::new();
        let mut buf = Vec::new();
        cal.write_to(&mut buf).unwrap();
        let mut cur = std::io::Cursor::new(buf);
        let back = Iu4Calibration::read_from(&mut cur).unwrap();
        assert_eq!(back.n_sites(), 0);
    }

    #[test]
    fn roundtrip_single_site() {
        let mut cal = Iu4Calibration::new();
        let mut s = Iu4CalSite::new(0, 1, 16, 8);
        for i in 0..16 {
            s.mu_a[i] = f32_to_f16((i as f32) * 0.1);
            s.s_a[i] = f32_to_f16(0.5 + (i as f32) * 0.01);
        }
        for i in 0..8 {
            s.w_mu_bias[i] = f32_to_f16((i as f32) * 0.25);
        }
        cal.sites.push(s);

        let mut buf = Vec::new();
        cal.write_to(&mut buf).unwrap();

        let mut cur = std::io::Cursor::new(buf);
        let back = Iu4Calibration::read_from(&mut cur).unwrap();
        assert_eq!(back.n_sites(), 1);
        let s2 = &back.sites[0];
        assert_eq!(s2.layer_idx, 0);
        assert_eq!(s2.proj_id, 1);
        assert_eq!(s2.n_channels, 16);
        assert_eq!(s2.n_output_rows, 8);
        assert_eq!(s2.mu_a.len(), 16);
        assert_eq!(s2.s_a.len(), 16);
        assert_eq!(s2.w_mu_bias.len(), 8);
        // bit-exact roundtrip on FP16 fields
        assert_eq!(s2.mu_a[5], f32_to_f16(0.5));
    }

    #[test]
    fn bucket_monotone() {
        assert_eq!(bucket_for(0.0), 0);
        let bsmall = bucket_for(1e-3);
        let bmid = bucket_for(1.0);
        let bbig = bucket_for(100.0);
        assert!(bsmall < bmid);
        assert!(bmid < bbig);
    }

    #[test]
    fn percentile_basic() {
        // Single channel, 10000 samples uniform in [-1, 1] → 99% of |x|
        // should be ≤ ~0.99.
        let mu = vec![0.0f32];
        let mut h = CenteredHistogram::new(mu);
        let mut state = 0xC0FFEEu64;
        let n = 10000usize;
        let mut x = Vec::with_capacity(n);
        for _ in 0..n {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let t = ((state >> 33) as f32) / (u32::MAX as f32);
            x.push(t * 2.0 - 1.0);
        }
        h.add_batch(&x, n);
        let p99 = h.percentile_abs(99.0)[0];
        // log-spaced bucket: edge near 1.0 should land us within [0.56, 1.78]
        // (8 buckets per decade, log10(1.0)=0 → edges 1.0 and 1.33).
        assert!(p99 >= 0.5 && p99 <= 2.0, "p99 = {p99}");
    }
}
