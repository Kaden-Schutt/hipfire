// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kevin Read
// hipfire — see LICENSE and NOTICE in the project root.

//! Reference-distribution block: the top-K representation of `P_ref` at one
//! scored position.
//!
//! The canonical on-wire layout for a block (used by the legacy `.kldref`
//! binary and as the natural row layout inside an HFQM `.kldref.hfq` package's
//! `top_indices` / `top_log_probs` / `residual_mass` payloads) is:
//!
//! ```text
//! [ indices  : u32 × top_k ]   // top-K token ids, descending logit
//! [ log_probs: f32 × top_k ]   // matching log-probabilities
//! [ residual : f32          ]   // 1 − Σ p_topk (clamped ≥ 0)
//! ```
//!
//! `RefBlock` is a borrowed *view* used for zero-copy scoring; the caller owns
//! the backing slices (downloaded from a file or an HFQM blob).

/// Borrowed view of one reference block, consumed by
/// [`crate::math::score_position`].
#[derive(Debug, Clone, Copy)]
pub struct RefBlock<'a> {
    pub top_indices: &'a [u32],
    pub top_log_probs: &'a [f32],
    pub residual_mass: f32,
}

/// Bytes for one canonical block given a top-K reduction.
pub fn block_to_bytes(indices: &[u32], log_probs: &[f32], residual_mass: f32) -> Vec<u8> {
    assert_eq!(
        indices.len(),
        log_probs.len(),
        "indices/log_probs length mismatch"
    );
    let mut out = Vec::with_capacity(indices.len() * 8 + 4);
    for &i in indices {
        out.extend_from_slice(&i.to_le_bytes());
    }
    for &lp in log_probs {
        out.extend_from_slice(&lp.to_le_bytes());
    }
    out.extend_from_slice(&residual_mass.to_le_bytes());
    out
}

/// Size in bytes of one canonical block for a given `top_k`.
pub const fn block_len(top_k: usize) -> usize {
    top_k * 8 + 4
}

/// Parse one canonical block of `top_k` entries from `buf` (must be at least
/// [`block_len`] bytes). Returns `(indices, log_probs, residual_mass)`.
pub fn block_from_bytes(buf: &[u8], top_k: usize) -> Result<(Vec<u32>, Vec<f32>, f32), String> {
    let need = block_len(top_k);
    if buf.len() < need {
        return Err(format!("ref block too short: {} < {need}", buf.len()));
    }
    let mut indices = Vec::with_capacity(top_k);
    for j in 0..top_k {
        let o = j * 4;
        indices.push(u32::from_le_bytes(buf[o..o + 4].try_into().unwrap()));
    }
    let lp_off = top_k * 4;
    let mut log_probs = Vec::with_capacity(top_k);
    for j in 0..top_k {
        let o = lp_off + j * 4;
        log_probs.push(f32::from_le_bytes(buf[o..o + 4].try_into().unwrap()));
    }
    let resid_off = top_k * 8;
    let residual = f32::from_le_bytes(buf[resid_off..resid_off + 4].try_into().unwrap());
    Ok((indices, log_probs, residual))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_round_trips() {
        let indices = vec![3u32, 7, 1, 9];
        let log_probs = vec![-0.1f32, -0.5, -1.2, -3.0];
        let residual = 0.0125f32;
        let bytes = block_to_bytes(&indices, &log_probs, residual);
        assert_eq!(bytes.len(), block_len(4));
        let (i2, lp2, r2) = block_from_bytes(&bytes, 4).unwrap();
        assert_eq!(i2, indices);
        assert_eq!(lp2, log_probs);
        assert_eq!(r2, residual);
    }

    #[test]
    fn short_buffer_errors() {
        assert!(block_from_bytes(&[0u8; 3], 4).is_err());
    }
}
