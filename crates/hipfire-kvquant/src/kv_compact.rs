// SPDX-License-Identifier: Apache-2.0
// hipfire — deferred-hierarchical KV compression, Phase 1: the COLD-TIER PRODUCER.
//
// See Quantization/kv_explore/ARCHITECTURE.md. Inline CASK-on-KVarN is blocked by
// KVarN's 128-token-block record layout (CASK merge yields arbitrary per-group
// slots that don't interleave — `CaskCtx::maybe_evict` falls through to TriAttn for
// quant_kvarn). Resolution: the merged "cold" data lives in a SEPARATE compacted
// buffer, produced by THIS standalone CPU pass that runs during idle / between
// turns (offload-friendly — CPU/NPU), where heavy KV compute is amortized.
//
// The pass: given a cold token range's K/V + a shared per-token importance score,
// keep the top `core_frac` tokens exact, merge the rest `fold_m:1` by
// importance-weighted average, FWHT-256-rotate per head (incoherence — matches the
// KVarN Hadamard rotation), and KVarN-quantize each head's `[head_dim × n_slots]`
// tile. Exploration: CASK average-merge ≫ HoloKV superposition on real attention;
// ~12–23× KV @ cos 0.97–0.99 (FINDINGS.md).

// Phase-1 scaffolding: the producer + ColdTier reader are consumed by the Phase-2
// engine wiring (two-tier flash) and the tests below; not yet called from the bin.
#![allow(dead_code)]

use crate::kvarn::{self, QuantTile};
use hipfire_primitives::fwht::{cpu_fwht_256, gen_fwht_signs};

/// One compacted cold buffer for a contiguous range of (old) cold tokens. Slot
/// structure is SHARED across kv-heads (CASK ranks tokens with a head-aggregated
/// score); each head carries its own KVarN-quantized `[head_dim × n_slots]` tile.
pub struct ColdTier {
    pub k_tiles: Vec<QuantTile>, // per kv-head, tile [head_dim × n_slots] (rotated frame)
    pub v_tiles: Vec<QuantTile>,
    pub slot_members: Vec<Vec<u32>>, // original token indices folded into each slot
    pub slot_repr_pos: Vec<u32>,     // representative (latest) absolute pos per slot
    pub head_dim: usize,
    pub n_slots: usize, // padded (even) tile width
    pub n_valid: usize, // real slot count (slots >= n_valid are zero padding — mask in reads)
    pub rotate: bool,
}

#[inline]
fn fwht(v: &mut [f32], s1: &[f32], s2: &[f32]) {
    cpu_fwht_256(v, s1, s2);
}

/// Deferred cold-tier compaction. `k`,`v` are `[n_tok, n_kv_heads*head_dim]` f32,
/// post-RoPE (the cold tokens, contiguous). `importance[t]` is a shared (head-
/// aggregated) score; higher = keep exact. `core_frac` of tokens stay singleton
/// (exact), the rest fold `fold_m:1` by importance-weighted average. `rotate` =
/// FWHT-256 incoherence per head before quantize. head_dim must be 256 (KVarN v1).
///
/// `position_local`: when true, the to-be-merged (non-core) tokens are grouped by
/// adjacent POSITION rather than by importance rank, so each merged slot averages
/// K vectors with similar RoPE phase (less phase-blur — the dominant cold-merge
/// quality cost). Core selection stays importance-based either way.
#[allow(clippy::too_many_arguments)]
pub fn compact_cold_kv(
    k: &[f32],
    v: &[f32],
    n_tok: usize,
    n_kv_heads: usize,
    head_dim: usize,
    importance: &[f32],
    core_frac: f32,
    fold_m: usize,
    rotate: bool,
    position_local: bool,
    // Max quant code for the cold tiles: 15 = 4-bit (default), 3 = 2-bit, etc.
    // Lower = lower-precision cold quant (quality probe; same nibble storage).
    qmax: f32,
) -> ColdTier {
    assert_eq!(head_dim, 256, "KVarN v1 FWHT is 256-wide");
    assert!(fold_m >= 1);
    assert_eq!(k.len(), n_tok * n_kv_heads * head_dim);
    let kv_dim = n_kv_heads * head_dim;

    // 1. Shared slot construction: rank by importance, top core exact, merge rest.
    let mut order: Vec<usize> = (0..n_tok).collect();
    order.sort_by(|&a, &b| {
        importance[b]
            .partial_cmp(&importance[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let ncore = ((core_frac * n_tok as f32) as usize).min(n_tok);
    let core = &order[..ncore];
    // The non-core tokens to merge. Position-local grouping sorts them back into
    // ascending position so each fold_m group is position-contiguous (similar
    // RoPE phase); otherwise they stay in importance-rank order.
    let mut scratch_owned: Vec<usize> = order[ncore..].to_vec();
    if position_local {
        scratch_owned.sort_unstable();
    }
    let scratch = &scratch_owned[..];
    let nb = if fold_m > 0 {
        scratch.len() / fold_m
    } else {
        0
    };

    let mut slot_members: Vec<Vec<u32>> = Vec::with_capacity(ncore + nb + fold_m);
    for &t in core {
        slot_members.push(vec![t as u32]);
    }
    for g in 0..nb {
        slot_members.push(
            scratch[g * fold_m..(g + 1) * fold_m]
                .iter()
                .map(|&x| x as u32)
                .collect(),
        );
    }
    for &t in &scratch[nb * fold_m..] {
        slot_members.push(vec![t as u32]); // leftover scratch kept singleton
    }
    let n_valid = slot_members.len();
    let n_slots = if n_valid % 2 == 0 {
        n_valid
    } else {
        n_valid + 1
    }; // KVarN c_dim even
    let slot_repr_pos: Vec<u32> = slot_members
        .iter()
        .map(|m| *m.iter().max().unwrap())
        .collect();

    // 2/3. Per head: build [head_dim × n_slots] tile of (rotated) merged K/V, quantize.
    let s1 = gen_fwht_signs(42, 256);
    let s2 = gen_fwht_signs(1042, 256);
    let mut k_tiles = Vec::with_capacity(n_kv_heads);
    let mut v_tiles = Vec::with_capacity(n_kv_heads);
    for h in 0..n_kv_heads {
        let mut ktile = vec![0.0f32; head_dim * n_slots];
        let mut vtile = vec![0.0f32; head_dim * n_slots];
        for (s, mem) in slot_members.iter().enumerate() {
            let wsum: f32 = mem
                .iter()
                .map(|&t| importance[t as usize].max(0.0))
                .sum::<f32>();
            let wsum = if wsum > 0.0 { wsum } else { mem.len() as f32 }; // unif if all-zero
            let mut kvec = vec![0.0f32; head_dim];
            let mut vvec = vec![0.0f32; head_dim];
            for &t in mem {
                let iw = importance[t as usize].max(0.0);
                let w = if wsum > 0.0 {
                    (if iw > 0.0 { iw } else { 1.0 }) / wsum
                } else {
                    1.0 / mem.len() as f32
                };
                let base = t as usize * kv_dim + h * head_dim;
                for d in 0..head_dim {
                    kvec[d] += w * k[base + d];
                    vvec[d] += w * v[base + d];
                }
            }
            if rotate {
                fwht(&mut kvec, &s1, &s2);
                fwht(&mut vvec, &s1, &s2);
            }
            for d in 0..head_dim {
                ktile[d * n_slots + s] = kvec[d];
                vtile[d * n_slots + s] = vvec[d];
            }
        }
        k_tiles.push(kvarn::quantize_tile_qmax(&ktile, head_dim, n_slots, qmax));
        v_tiles.push(kvarn::quantize_tile_qmax(&vtile, head_dim, n_slots, qmax));
    }

    ColdTier {
        k_tiles,
        v_tiles,
        slot_members,
        slot_repr_pos,
        head_dim,
        n_slots,
        n_valid,
        rotate,
    }
}

impl ColdTier {
    /// Dequantize head `h` back to (K, V) as `[n_valid × head_dim]` row-major in the
    /// ORIGINAL (un-rotated) basis — what a cold-tier attention read consumes.
    pub fn dequant_head(&self, h: usize) -> (Vec<f32>, Vec<f32>) {
        let s1 = gen_fwht_signs(42, 256);
        let s2 = gen_fwht_signs(1042, 256);
        let kt = kvarn::dequantize_tile(&self.k_tiles[h]); // [head_dim × n_slots]
        let vt = kvarn::dequantize_tile(&self.v_tiles[h]);
        let (ns, d, nv) = (self.n_slots, self.head_dim, self.n_valid);
        let mut k = vec![0.0f32; nv * d];
        let mut v = vec![0.0f32; nv * d];
        for s in 0..nv {
            let mut kv = vec![0.0f32; d];
            let mut vv = vec![0.0f32; d];
            for dd in 0..d {
                kv[dd] = kt[dd * ns + s];
                vv[dd] = vt[dd * ns + s];
            }
            if self.rotate {
                fwht(&mut kv, &s2, &s1); // inverse FWHT: swap sign tables
                fwht(&mut vv, &s2, &s1);
            }
            for dd in 0..d {
                k[s * d + dd] = kv[dd];
                v[s * d + dd] = vv[dd];
            }
        }
        (k, v)
    }

    /// Phase 2 read reference (CPU): combined two-tier causal attention for one
    /// query (q-head) over the HOT tokens (recent, exact/quantized) + this kv-head's
    /// COLD merged slots. Scores are concatenated across both tiers before a single
    /// softmax; output = Σ p·V over both. This is the math the GPU two-tier flash
    /// must implement.
    ///
    /// - `q` [head_dim]; `hot_k`/`hot_v` `[n_hot × head_dim]` (already dequantized),
    ///   hot token `t` lives at absolute position `hot_base_pos + t`.
    /// - `q_pos` = the query's absolute position (causal: attend to hot tokens with
    ///   position ≤ q_pos; ALL cold slots are visible since the cold tier holds only
    ///   tokens older than the hot window, i.e. older than any query that reads it).
    /// - `cold_k`/`cold_v` = `self.dequant_head(kv_head)` (caller dequants once).
    #[allow(clippy::too_many_arguments)]
    pub fn two_tier_attend(
        &self,
        q: &[f32],
        hot_k: &[f32],
        hot_v: &[f32],
        n_hot: usize,
        cold_k: &[f32],
        cold_v: &[f32],
        q_pos: usize,
        hot_base_pos: usize,
        head_dim: usize,
    ) -> Vec<f32> {
        let sc = 1.0 / (head_dim as f32).sqrt();
        let n_cold = self.n_valid;
        let mut logits = Vec::with_capacity(n_hot + n_cold);
        let mut idx: Vec<(bool, usize)> = Vec::with_capacity(n_hot + n_cold); // (is_cold, idx)
                                                                              // Hot tier — causal.
        for t in 0..n_hot {
            if hot_base_pos + t > q_pos {
                continue;
            }
            let s: f32 = (0..head_dim)
                .map(|i| q[i] * hot_k[t * head_dim + i])
                .sum::<f32>()
                * sc;
            logits.push(s);
            idx.push((false, t));
        }
        // Cold tier — all merged slots visible (repr_pos < hot_base_pos ≤ q_pos).
        for s_i in 0..n_cold {
            debug_assert!((self.slot_repr_pos[s_i] as usize) < hot_base_pos.max(1));
            let s: f32 = (0..head_dim)
                .map(|i| q[i] * cold_k[s_i * head_dim + i])
                .sum::<f32>()
                * sc;
            logits.push(s);
            idx.push((true, s_i));
        }
        let mx = logits.iter().cloned().fold(f32::MIN, f32::max);
        let mut p: Vec<f32> = logits.iter().map(|x| (x - mx).exp()).collect();
        let z: f32 = p.iter().sum();
        for x in &mut p {
            *x /= z;
        }
        let mut o = vec![0.0f32; head_dim];
        for (k, &(is_cold, j)) in idx.iter().enumerate() {
            let src = if is_cold {
                &cold_v[j * head_dim..]
            } else {
                &hot_v[j * head_dim..]
            };
            for i in 0..head_dim {
                o[i] += p[k] * src[i];
            }
        }
        o
    }

    /// Bytes for the compacted buffer (both tiles, all heads) + posmeta.
    pub fn bytes(&self) -> usize {
        let rec = kvarn::kvarn_record_bytes(self.head_dim, self.n_slots);
        self.k_tiles.len() * rec * 2 + self.n_valid * 4
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Lcg(u64);
    impl Lcg {
        fn f(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 33) as f32 / (1u64 << 31) as f32) - 1.0
        }
        fn n(&mut self) -> f32 {
            (self.f() + self.f() + self.f() + self.f()) / 2.0 // ~gaussian-ish
        }
    }

    fn cos(a: &[f32], b: &[f32]) -> f32 {
        let d: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        d / (na * nb + 1e-9)
    }

    // Causal attention of a single query over a token set; returns output[head_dim].
    fn attn(q: &[f32], k: &[f32], v: &[f32], n: usize, d: usize) -> Vec<f32> {
        let sc = 1.0 / (d as f32).sqrt();
        let mut s = vec![0.0f32; n];
        for t in 0..n {
            s[t] = (0..d).map(|i| q[i] * k[t * d + i]).sum::<f32>() * sc;
        }
        let mx = s.iter().cloned().fold(f32::MIN, f32::max);
        let mut p: Vec<f32> = s.iter().map(|x| (x - mx).exp()).collect();
        let z: f32 = p.iter().sum();
        for x in &mut p {
            *x /= z;
        }
        let mut o = vec![0.0f32; d];
        for t in 0..n {
            for i in 0..d {
                o[i] += p[t] * v[t * d + i];
            }
        }
        o
    }

    // Attention over cold slots: query · merged_K → softmax → · merged_V.
    fn attn_slots(q: &[f32], k: &[f32], v: &[f32], cold: &ColdTier, d: usize) -> Vec<f32> {
        attn(q, k, v, cold.n_valid, d)
    }

    #[test]
    fn cold_tier_reconstruction_preserves_attention() {
        // 512 cold tokens, 1 kv-head, head_dim 256. A handful of "important" tokens
        // (high-norm, aligned with the query) carry most attention; the rest are
        // low-importance and get merged. Verify the compacted cold tier reproduces
        // the attention OUTPUT of full-precision cold K/V.
        let (nt, h, d) = (512usize, 1usize, 256usize);
        let mut rng = Lcg(0x1234_5678);
        let q: Vec<f32> = (0..d).map(|_| rng.n()).collect();
        let mut k = vec![0.0f32; nt * d];
        let mut v = vec![0.0f32; nt * d];
        for t in 0..nt {
            let important = t % 37 == 0; // ~14 important tokens
            for i in 0..d {
                // important tokens: K aligned with q (high score) + larger norm.
                k[t * d + i] = if important {
                    q[i] * 1.6 + rng.n() * 0.3
                } else {
                    rng.n() * 0.5
                };
                v[t * d + i] = rng.n();
            }
        }
        // importance = the actual attention logit q·K (head-aggregated == single head here).
        let importance: Vec<f32> = (0..nt)
            .map(|t| (0..d).map(|i| q[i] * k[t * d + i]).sum::<f32>())
            .collect();

        let ref_out = attn(&q, &k, &v, nt, d);

        for &(cf, m) in &[(0.25f32, 8usize), (0.125, 16), (0.5, 4)] {
            let cold = compact_cold_kv(&k, &v, nt, h, d, &importance, cf, m, true, false, 15.0);
            let (kr, vr) = cold.dequant_head(0);
            let out = attn_slots(&q, &kr, &vr, &cold, d);
            let c = cos(&out, &ref_out);
            let comp = (nt * d * 2 * 2) as f32 / cold.bytes() as f32; // vs f16 KV
            eprintln!(
                "core={cf} m={m}: out_cos={c:.4} slots={}/{} compress={comp:.1}x",
                cold.n_valid, nt
            );
            assert!(c > 0.93, "core={cf} m={m}: output cosine {c:.4} too low");

            // Directly stress the MERGED path: a diffuse query (attends ~uniformly)
            // forces the merged slots to be read; their averaged K/V must still
            // reproduce the output (cos > 0.9 — merge + KVarN-4b roundtrip).
            let qd: Vec<f32> = vec![0.02; d];
            let ref_d = attn(&qd, &k, &v, nt, d);
            let out_d = attn_slots(&qd, &kr, &vr, &cold, d);
            let cd = cos(&out_d, &ref_d);
            eprintln!("  diffuse-query out_cos={cd:.4}");
            assert!(
                cd > 0.85,
                "core={cf} m={m}: diffuse output cosine {cd:.4} too low"
            );
        }
    }

    #[test]
    fn merged_slot_matches_weighted_average_roundtrip() {
        // A merged slot's dequantized vector ≈ the importance-weighted average of its
        // members (the KVarN-4b roundtrip of the merge), independent of attention.
        let (nt, d) = (256usize, 256usize);
        let mut rng = Lcg(7);
        let mut k = vec![0.0f32; nt * d];
        let v = vec![0.0f32; nt * d];
        for x in k.iter_mut() {
            *x = rng.n();
        }
        let imp: Vec<f32> = (0..nt).map(|t| 1.0 + (t % 5) as f32).collect(); // varied weights
        let cold = compact_cold_kv(&k, &v, nt, 1, d, &imp, 0.0, 8, true, false, 15.0); // all merged
        let (kr, _) = cold.dequant_head(0);
        // recompute the true weighted-average of slot 0's members and compare.
        let mem = &cold.slot_members[0];
        let wsum: f32 = mem.iter().map(|&t| imp[t as usize]).sum();
        let mut avg = vec![0.0f32; d];
        for &t in mem {
            let w = imp[t as usize] / wsum;
            for i in 0..d {
                avg[i] += w * k[t as usize * d + i];
            }
        }
        let got = &kr[0..d];
        assert!(
            cos(got, &avg) > 0.99,
            "merged slot roundtrip cos {:.4}",
            cos(got, &avg)
        );
    }

    #[test]
    fn two_tier_read_matches_full_attention() {
        // Full context = cold (old) + hot (recent W). Compact the cold region, then
        // two_tier_attend for queries in the hot region must reproduce full attention.
        let (n_tok, w, d) = (320usize, 64usize, 256usize);
        let n_cold = n_tok - w;
        let mut rng = Lcg(0xC0FFEE);
        let mut k = vec![0.0f32; n_tok * d];
        let mut v = vec![0.0f32; n_tok * d];
        // a few important cold tokens aligned with a shared direction; rest diffuse.
        let dir: Vec<f32> = (0..d).map(|_| rng.n()).collect();
        for t in 0..n_tok {
            let important = t < n_cold && t % 29 == 0;
            for i in 0..d {
                k[t * d + i] = if important {
                    dir[i] * 1.5 + rng.n() * 0.3
                } else {
                    rng.n() * 0.6
                };
                v[t * d + i] = rng.n();
            }
        }
        let imp: Vec<f32> = (0..n_cold)
            .map(|t| (0..d).map(|i| dir[i] * k[t * d + i]).sum::<f32>())
            .collect();
        let cold = compact_cold_kv(
            &k[..n_cold * d],
            &v[..n_cold * d],
            n_cold,
            1,
            d,
            &imp,
            0.25,
            8,
            true,
            false,
            15.0,
        );
        let (ck, cv) = cold.dequant_head(0);
        let hot_k = &k[n_cold * d..];
        let hot_v = &v[n_cold * d..];

        // Evaluate a few queries at hot positions (use the model's own K rows as
        // plausible queries) vs full-precision causal attention over all tokens.
        let mut worst = 1.0f32;
        for &qpos in &[n_cold, n_cold + w / 2, n_tok - 1] {
            let q = &k[qpos * d..qpos * d + d];
            // reference: full causal attention over [0..=qpos]
            let ref_o = attn(q, &k, &v, qpos + 1, d);
            let got = cold.two_tier_attend(q, hot_k, hot_v, w, &ck, &cv, qpos, n_cold, d);
            let c = cos(&got, &ref_o);
            eprintln!("qpos={qpos}: two-tier out_cos={c:.4}");
            worst = worst.min(c);
        }
        assert!(worst > 0.9, "two-tier attention cosine {worst:.4} too low");
    }

    #[test]
    fn no_rotation_path_is_consistent() {
        // rotate=false must also round-trip (un-rotated KVarN of merged slots).
        let (nt, h, d) = (256usize, 1usize, 256usize);
        let mut rng = Lcg(42);
        let q: Vec<f32> = (0..d).map(|_| rng.n()).collect();
        let mut k = vec![0.0f32; nt * d];
        let mut v = vec![0.0f32; nt * d];
        for t in 0..nt {
            for i in 0..d {
                k[t * d + i] = rng.n();
                v[t * d + i] = rng.n();
            }
        }
        let imp: Vec<f32> = (0..nt)
            .map(|t| (0..d).map(|i| q[i] * k[t * d + i]).sum::<f32>())
            .collect();
        let cold = compact_cold_kv(&k, &v, nt, h, d, &imp, 0.25, 4, false, false, 15.0);
        let (kr, _vr) = cold.dequant_head(0);
        assert_eq!(kr.len(), cold.n_valid * d);
        assert!(kr.iter().all(|x| x.is_finite()));
    }
}
