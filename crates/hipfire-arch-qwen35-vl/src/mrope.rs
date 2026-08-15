// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! 3D mrope position construction for Qwen3.5-VL.
//!
//! Mirrors `get_rope_index` / `get_vision_position_ids` in
//! `transformers/models/qwen3_5/modeling_qwen3_5.py`. Text tokens take the
//! same value on all three axes (which makes 3D mrope identical to 1D RoPE
//! for pure text); image tokens take their (t, h, w) grid coordinate.
//!
//! Pure CPU, no GPU, no I/O — this is the unit under test in
//! `tests/mrope_positions.rs`.

/// Default `mrope_section` for the Qwen3.5 family: 11 T + 11 H + 10 W = 32,
/// exactly the rotary-pair count (head_dim 256 * partial_rotary_factor 0.25
/// = 64 rotary dims = 32 pairs).
pub const DEFAULT_MROPE_SECTION: [usize; 3] = [11, 11, 10];

/// One contiguous run of visual tokens in the prompt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageSpan {
    /// Index of the first visual token.
    pub start: usize,
    /// Visual token count, POST-merge (== (grid_h/merge) * (grid_w/merge)).
    pub len: usize,
    /// PRE-merge grid height, as reported by the vision tower.
    pub grid_h: usize,
    /// PRE-merge grid width.
    pub grid_w: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MropePositions {
    /// Per-token (t, h, w).
    pub positions: Vec<[i32; 3]>,
    /// `positions.max() + 1 - n_tokens`. Added to the running sequence
    /// length to get decode-step positions.
    pub rope_delta: i32,
}

/// Which position axis frequency index `d` reads: 0 = T, 1 = H, 2 = W.
///
/// HF `apply_interleaved_mrope` starts every frequency on T, then overwrites
/// `slice(1, 3*section[1], 3)` with H and `slice(2, 3*section[2], 3)` with W,
/// producing `[THWTHW...TT]`. A chunked `[TTT...HHH...WWW]` reading is wrong.
pub fn mrope_axis_for_freq(d: usize, section: [usize; 3]) -> usize {
    match d % 3 {
        1 if d < 3 * section[1] => 1,
        2 if d < 3 * section[2] => 2,
        _ => 0,
    }
}

/// Build per-token (t, h, w) positions for a prompt containing `spans`
/// image runs. `spans` must be sorted by `start` and non-overlapping.
pub fn build_mrope_positions(
    n_tokens: usize,
    spans: &[ImageSpan],
    spatial_merge_size: usize,
) -> MropePositions {
    assert!(spatial_merge_size > 0, "spatial_merge_size must be positive");
    let mut positions = Vec::with_capacity(n_tokens);
    let mut cursor: i32 = 0;
    let mut tok = 0usize;

    for span in spans {
        debug_assert!(span.start >= tok, "image spans must be sorted and disjoint");
        // Text run before this image.
        while tok < span.start && tok < n_tokens {
            positions.push([cursor; 3]);
            cursor += 1;
            tok += 1;
        }
        // Image run: t constant, h slowest, w fastest, over the merged grid.
        let lh = span.grid_h / spatial_merge_size;
        let lw = span.grid_w / spatial_merge_size;
        for hh in 0..lh {
            for ww in 0..lw {
                positions.push([cursor, cursor + hh as i32, cursor + ww as i32]);
            }
        }
        tok += lh * lw;
        // Advance by ONE GRID DIMENSION, not by the token count.
        cursor += (span.grid_h.max(span.grid_w) / spatial_merge_size) as i32;
    }

    // Trailing text run.
    while tok < n_tokens {
        positions.push([cursor; 3]);
        cursor += 1;
        tok += 1;
    }

    let max_pos = positions
        .iter()
        .flat_map(|p| p.iter().copied())
        .max()
        .unwrap_or(0);
    let rope_delta = max_pos + 1 - n_tokens as i32;

    MropePositions { positions, rope_delta }
}
