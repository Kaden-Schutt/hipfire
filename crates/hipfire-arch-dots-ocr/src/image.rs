//! Image preprocessing for dots.ocr — phase 2b stub.
//!
//! Lands in the next commit (task 2b). Provides:
//!
//! - **Smart-resize** (`smart_resize`) — 28-divisible H/W with
//!   `[min_pixels=3136, max_pixels=11_289_600]` clamp, beta scaling,
//!   and AR > 200:1 guard. Ported from `dots_ocr/utils/image_utils.py`.
//! - **CLIP normalisation** — mean `[0.48145466, 0.4578275, 0.40821073]`,
//!   std `[0.26862954, 0.26130258, 0.27577711]`, RGB; RGBA composited
//!   onto white background first.
//! - **Patch extraction** — `extract_patches` reshape +
//!   `transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)` from
//!   `image_processing_qwen2_vl.py:281-295`. This is the
//!   silent-failure trap of §2.7: emitting patches in raw raster order
//!   without the transpose makes the merger group horizontally-
//!   adjacent patches instead of 2×2 spatial blocks, producing
//!   plausible JSON with subtly-wrong bbox coordinates.
//!
//! Unit tests committed alongside:
//! - `smart_resize` clamps a 100×3000 input within bounds and lands
//!   both dims on 28-multiples; AR-guard rejects 1×500.
//! - `extract_patches` on a synthetic 56×28 RGB input (1×4 patch grid
//!   with `merge_size=2`) produces a `flatten_patches` byte sequence
//!   byte-identical to HF `Qwen2VLImageProcessor` on the same input.
//!   No GPU required.
//!
//! See `docs/plans/qwen_2.0_vlm_plus_dots_ocr.md` §2.7 for the
//! authoritative algorithm spec; `crates/hipfire-arch-qwen35-vl/src/image.rs`
//! for the qwen35-vl port as a starting skeleton (note: it uses a
//! different resize policy — do not copy verbatim).
