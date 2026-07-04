// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! Canonical on-disk HFQ `quant_type` byte-contract.
//!
//! The single byte stored per tensor in an `.hfq` index identifies the weight
//! encoding. It is the contract between the **writer** (the offline
//! `hipfire-quantize` binary, which emits `QuantType as u8`) and every
//! **reader** (the engine loaders in `hipfire-runtime` and the per-arch
//! crates, which map the byte back to a GPU dispatch dtype).
//!
//! This enum used to live privately inside the `hipfire-quantize` binary, so
//! every reader re-hardcoded the integers (`6`, `13`, `31`, …) — they drifted
//! (e.g. only qwen35 knew `31 == Qtip3G256`). Homing it in this leaf crate,
//! depended on by both sides, makes the numbering authoritative in one place.
//!
//! This crate is intentionally GPU-agnostic: it owns the *byte identity* only.
//! The byte → GPU `DType` mapping (which needs `hipfire-rdna`) lives in
//! `hipfire_runtime::quant::dtype_for_quant_type`, which matches on the
//! variants here.

/// On-disk HFQ weight encoding id (`#[repr(u8)]`; the stored byte is the
/// discriminant). Reserved/retired ids are documented inline — DO NOT REUSE.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuantType {
    Q4F16G64 = 0,
    F16 = 1,
    F32 = 2,
    Q8F16 = 3,
    Q4K = 4,
    Q8HFQ = 5,
    HFQ4G256 = 6,
    HFQ4G128 = 7,
    HFQ6G256 = 8,
    HFQ2G256 = 9,
    HFQ2G128 = 10,
    HFQ3G256 = 11,
    HFQ3G128 = 12,
    MQ4G256 = 13,      // MagnumQuant: FWHT-rotated HFQ4-G256
    MQ8G256 = 14,      // MagnumQuant: FWHT-rotated symmetric INT8, dp4a target
    MQ6G256 = 15,      // MagnumQuant: FWHT-rotated HFQ6-G256 (6-bit, 200 B/group)
    BF16 = 16,         // Original BF16 weights (zero precision loss for vision)
    MQ3G256 = 17,      // MagnumQuant: FWHT-rotated HFQ3-G256 (3-bit, 104 B/group)
    MQ2G256 = 18,      // MagnumQuant: FWHT-rotated HFQ2-G256 (2-bit, 72 B/group)
    MQ2G256Lloyd = 19, // MagnumQuant 2-bit + per-block Lloyd-Max 4-entry fp16 codebook (72 B/group)
    MQ3G256Lloyd = 20, // MagnumQuant 3-bit + per-block Lloyd-Max 8-entry fp16 codebook (112 B/group)
    // HFP4 family — RDNA-optimal FP4 (E2M1 elements + UE8M0 block scale + FP16 row scale).
    // See docs/quant-formats/hfp4.md for byte layout, dequant, rotation modes.
    // Per-row header is 16 B; per-block payload is (1 + g/2) bytes (UE8M0 + nibbles).
    HFP4G32 = 21, // E2M1 + UE8M0 g32 + FP16 row scale — canonical (FP8-WMMA-K aligned)
    /// I64→U32 downcast of DeepSeek V4 hash-routing `tid2eid` lookup tables.
    /// Shape `[vocab, num_experts_per_tok]`. Stored as raw u32 LE; the
    /// loader reads `bytes.chunks_exact(4)`. ID 22 was reserved for the
    /// HFP4G16 NV-aligned ablation (never built) — we re-use the slot
    /// for tid2eid storage to stay byte-compatible with antirezQ8.hfq.
    TidI32 = 22,
    // Reserved IDs — DO NOT REUSE for unrelated formats. Documented in docs/quant-formats/hfp4.md.
    // HFP4G16     = 22, // v1.5 — NV-aligned FP16-WMMA-K alignment ablation (re-used by TidI32)
    // HFP4G64     = 23, // v1.5 — RDNA1/2 sweet-spot ablation
    // HFP4G32MX   = 25, // v2  — strict OCP MXFP4 interop alias (no row scale, UE8M0 only)
    // HFP4G16NV   = 26, // v2  — strict NVFP4 interop alias (E4M3 scale + FP32 tensor)
    // HFP8E4M3G32 = 27, // v2  — HFP8 E4M3 family
    // MFP4G32 = HFP4G32 + offline FWHT rotation (256-element FWHT applied to weights at quant time;
    // runtime applies the same FWHT to x via mq_rotate_x). format_flags bit 0 + bits 2-3 = 0b0101
    // signals "rotation present, offline FWHT" for future interop/detection.
    MFP4G32 = 24,    // v1.5 — HFP4G32 + offline FWHT (drop-in MQ4 replacement)
    PARO4G128 = 28,  // ParoQuant native AWQ W4 + pairwise activation rotation metadata
    PARO4G128T = 29, // ParoQuant engine-tiled qweight [M/8, K] for coalesced GEMV reads
    // MFP4G32R    = 29, // v3  — HFP4G32 + online block-diag-128 rotation (AMD recipe)
    // HFP8E5M2G32 = 30, // v2  — HFP8 E5M2 family
    MQ4G256Lloyd = 30, // MagnumQuant 4-bit + per-block Lloyd-Max 16-entry fp16 codebook (160 B/group)
    // Renumbered from 21 → 30 in mq4-lloyd merge to avoid HFP4G32=21 collision.
    // Models quantized pre-renumber MUST be re-quantized.
    /// QTIP-3: trellis-coded 3-bit, FWHT-rotated. Block = [f32 scale][96 B
    /// packed 3-bit trellis symbols] = 100 B/group (0.391 B/weight). Decoded by
    /// the fused `gemv_qtip3g256` kernel (computed 1MAD codebook, zero LDS); the
    /// runtime FWHT-rotates x (shared mq_rotate_x path). See qtip.rs / Phase C2.
    Qtip3G256 = 31,
    /// OQ+ / Opus Plus (W4A8) — the symmetric-int4 analog of MQ4+: the SAME
    /// on-disk bytes as [`QuantType::Oq4G256`] (symmetric signed-INT4, FWHT,
    /// per-group f32 scale, codec `quantize_oq4g256`, including its LDLQ/AWQ
    /// calibration), but the loader (qt=33) nibble-EXPANDS the int4 weights to
    /// int8 and dispatches the iu8 W8A8 grouped-WMMA path with int8
    /// ACTIVATIONS. Weight values stay 4-bit (16 levels); activations gain int8
    /// precision. Id 33 = the eval-plan's reserved Opus-A8 slot.
    OqPlusG256 = 33,
    /// Opus Quant W4A4 — symmetric signed-INT4, FWHT-rotated, per-group f32
    /// scale. On-disk block = [f16 scale][128 nibbles] = 130 B/256-group
    /// (codec `quantize_oq4g256`). Loader (qt=34) repacks to the kernel layout;
    /// forward int4-quantizes activations and runs the iu4·iu4 GEMM. Id 34 =
    /// the eval-plan's reserved "Opus Quant (W4A4)" slot.
    Oq4G256 = 34,
    /// Opus Quant W8A8 — symmetric signed-INT8, FWHT-rotated, per-group f32
    /// scale. On-disk block = [f16 scale][256 int8] = 258 B/256-group (codec
    /// `quantize_oq8g256`). Loader (qt=35) repacks to the kernel layout;
    /// forward int8-quantizes activations and runs the iu8 GEMM. Near-lossless,
    /// matrix-core-fast.
    Oq8G256 = 35,
    /// OQ+ compact magnitude-tiered (Opus Plus W4A8, ~4 b/w). On-disk block =
    /// `[f16 scale][128 int4 nibbles][N_out × (u8 idx, i8 val)]` = 130 + 2·N_out
    /// B/256-group (codec `quantize_oqplus_compact`; N_out = round(w8_frac·256)).
    /// Loader (qt=36) derives N_out from the byte length, expands the int4
    /// bulk to int8 and overlays the sparse int8 outliers → the iu8 W8A8 buffer.
    OqPlusCompact = 36,
    /// Opus Quant W4A4, already in the arch-combined **kernel** layout on disk
    /// (the [`QuantType::Oq4G256`] payload after the per-arch repack the loader
    /// would otherwise do at load time). The loader uploads it raw and tags
    /// `Oq4G256` — no repack. Distinct id so the byte length is validated
    /// against `oq4_arch_combined_len`, not the 130 B/group on-disk form.
    Oq4G256ArchPacked = 37,
    /// Opus Quant W3A4 — symmetric signed-INT3, FWHT-rotated, per-group f32 scale.
    /// On-disk block = `[f16 scale][8 × (3 u32 bit-planes)]` = 98 B/256-group
    /// (codec `quantize_oq3g256`), 3.0625 b/w — the memory-ceiling lever (25% less
    /// weight traffic than Oq4). Bit-plane storage IS the W3A4 kernel layout (tuned
    /// iu4 GEMM + W3 decode GEMV, cheap Morton spread-to-int4 unpack). Forward
    /// int4-quantizes activations and runs the W3A4 iu4·iu4 GEMM. Id 38 = the next
    /// free Opus-family slot. 3-bit is only viable atop the SpinQuant learned
    /// rotation (see the W3A4 / SpinQuant memory notes).
    Oq3G256 = 38,
    /// Opus Quant W2 — symmetric signed-INT2, FWHT-rotated, per-group f32 scale.
    /// Family completion; codec/loader/kernel pending. 2-bit is quality-marginal
    /// (see project_lowbit_quant_findings) — mixed-precision / heavy-treatment only.
    Oq2G256 = 39,
    /// Opus Quant W6 — symmetric signed-INT6, FWHT-rotated. Family completion; the
    /// near-lossless mid-tier between Oq4 and Oq8. Codec/loader/kernel pending.
    Oq6G256 = 40,
    /// QTIP W2 — trellis-coded 2-bit, FWHT-rotated (bit-parametric sibling of
    /// [`QuantType::Qtip3G256`]). Codec/kernel pending.
    Qtip2G256 = 41,
    /// QTIP W4 — trellis-coded 4-bit, FWHT-rotated (bit-parametric sibling of
    /// [`QuantType::Qtip3G256`]). Codec/kernel pending.
    Qtip4G256 = 42,
}

impl QuantType {
    /// The stored on-disk byte (the `#[repr(u8)]` discriminant).
    #[inline]
    pub const fn code(self) -> u8 {
        self as u8
    }

    /// Map a stored byte back to its [`QuantType`], or `None` for an
    /// unknown/reserved id. The inverse of [`QuantType::code`].
    pub fn from_code(code: u8) -> Option<Self> {
        use QuantType::*;
        Some(match code {
            0 => Q4F16G64,
            1 => F16,
            2 => F32,
            3 => Q8F16,
            4 => Q4K,
            5 => Q8HFQ,
            6 => HFQ4G256,
            7 => HFQ4G128,
            8 => HFQ6G256,
            9 => HFQ2G256,
            10 => HFQ2G128,
            11 => HFQ3G256,
            12 => HFQ3G128,
            13 => MQ4G256,
            14 => MQ8G256,
            15 => MQ6G256,
            16 => BF16,
            17 => MQ3G256,
            18 => MQ2G256,
            19 => MQ2G256Lloyd,
            20 => MQ3G256Lloyd,
            21 => HFP4G32,
            22 => TidI32,
            24 => MFP4G32,
            28 => PARO4G128,
            29 => PARO4G128T,
            30 => MQ4G256Lloyd,
            31 => Qtip3G256,
            33 => OqPlusG256,
            34 => Oq4G256,
            35 => Oq8G256,
            36 => OqPlusCompact,
            37 => Oq4G256ArchPacked,
            38 => Oq3G256,
            39 => Oq2G256,
            40 => Oq6G256,
            41 => Qtip2G256,
            42 => Qtip4G256,
            _ => return None,
        })
    }

    /// Elements per quantization group/block for this format.
    ///
    /// This is the divisor for `n_groups = ceil(n_elems / group_size())`.
    /// Per-element formats (F16/F32/BF16) report 1. Variable-layout formats
    /// still report their nominal group width; callers that need the byte
    /// geometry should consult [`QuantType::block_bytes`], which is `None` for
    /// variable formats.
    pub const fn group_size(self) -> usize {
        use QuantType::*;
        match self {
            F16 | F32 | BF16 => 1,
            Q8F16 => 32,
            Q4F16G64 => 64,
            HFP4G32 | MFP4G32 => 32,
            HFQ4G128 | HFQ2G128 | HFQ3G128 | PARO4G128 | PARO4G128T => 128,
            // Everything else is a 256-wide group: the G256 families, the Q4K
            // superblock, Qtip/Oq/Mq G256, the Lloyd codebook variants, OqPlus.
            _ => 256,
        }
    }

    /// Bytes per packed block for fixed-geometry formats, or `None` for formats
    /// whose block byte length varies with tensor shape / per-block metadata
    /// (Q8HFQ, HFP4/MFP4 row-scaled, OqPlus tiered/compact, Paro, arch-packed,
    /// and any not-yet-single-sourced trellis variant).
    ///
    /// Single source of truth for the block geometry that was previously
    /// re-hardcoded across codecs, loaders, and arch crates (review 2026-07-03
    /// §3.9). `tensor_bytes(n) = ceil(n / group_size()) * block_bytes()`.
    pub const fn block_bytes(self) -> Option<usize> {
        use QuantType::*;
        match self {
            // Dense / GGUF-style
            F32 => Some(4),
            F16 | BF16 => Some(2),
            Q8F16 => Some(34),    // 2 (f16 scale) + 32 int8
            Q4F16G64 => Some(36), // 4 (f16 scale+min) + 32 nibbles
            Q4K => Some(144),     // llama.cpp Q4_K superblock
            // HFQ (rotation-free) + MQ (FWHT-rotated) share byte geometry
            HFQ4G256 | MQ4G256 => Some(136), // 8 meta + 128 nibbles
            HFQ4G128 => Some(72),
            HFQ6G256 | MQ6G256 => Some(200), // 8 meta + 192 (6-bit×256)
            HFQ3G256 | MQ3G256 => Some(104), // 8 meta + 96 packed 3-bit
            HFQ3G128 => Some(56),
            HFQ2G256 | MQ2G256 | MQ2G256Lloyd => Some(72), // 8 meta + 64 packed
            HFQ2G128 => Some(40),
            MQ8G256 => Some(258), // 2 (f16 scale) + 256 int8
            MQ3G256Lloyd => Some(112),
            MQ4G256Lloyd => Some(160),
            // Opus Quant (symmetric)
            Oq4G256 => Some(130), // 2 (f16 scale) + 128 nibbles
            Oq3G256 => Some(98),  // 2 (f16 scale) + 8×3 u32 bit-planes
            Oq6G256 => Some(194), // 2 (f16 scale) + 192 (6-bit×256)
            Oq8G256 => Some(258), // 2 (f16 scale) + 256 int8
            // QTIP trellis (f32 scale + packed symbols)
            Qtip3G256 => Some(100), // 4 + 96 (256×3-bit)
            Qtip4G256 => Some(132), // 4 + 128 (256×4-bit)
            // Variable-length or not-yet-single-sourced here:
            //  - Q8HFQ: row-dependent
            //  - HFP4G32 / MFP4G32: per-row FP scale
            //  - OqPlusG256 / OqPlusCompact: tiered / 130 + 2·N_out
            //  - Oq2G256 / Oq4G256ArchPacked / Qtip2G256: geometry unconfirmed
            //  - Paro / TidI32: engine-tiled, arch-specific
            Q8HFQ | HFP4G32 | MFP4G32 | OqPlusG256 | OqPlusCompact | Oq2G256
            | Oq4G256ArchPacked | Qtip2G256 | PARO4G128 | PARO4G128T | TidI32 => None,
        }
    }

    /// Total packed byte length for `n_elems` of a fixed-geometry format, or
    /// `None` for variable-layout formats. `ceil(n / group) * block_bytes`.
    pub fn tensor_bytes(self, n_elems: usize) -> Option<usize> {
        let block = self.block_bytes()?;
        let groups = n_elems.div_ceil(self.group_size());
        Some(groups * block)
    }
}

#[cfg(test)]
mod tests {
    use super::QuantType;

    #[test]
    fn code_roundtrips_through_from_code() {
        // Every variant must survive code() → from_code().
        for c in 0u8..=255 {
            if let Some(qt) = QuantType::from_code(c) {
                assert_eq!(
                    qt.code(),
                    c,
                    "from_code({c}) gave {qt:?} whose code() != {c}"
                );
            }
        }
    }

    #[test]
    fn block_bytes_match_the_on_disk_geometry() {
        // On-disk contract: these byte counts are the format's block size and
        // must never move. Mirrors the codecs.rs encoders + arch-crate loaders.
        let cases = [
            (QuantType::F32, 4usize),
            (QuantType::F16, 2),
            (QuantType::BF16, 2),
            (QuantType::Q8F16, 34),
            (QuantType::Q4F16G64, 36),
            (QuantType::Q4K, 144),
            (QuantType::HFQ4G256, 136),
            (QuantType::MQ4G256, 136),
            (QuantType::HFQ4G128, 72),
            (QuantType::HFQ6G256, 200),
            (QuantType::MQ6G256, 200),
            (QuantType::HFQ3G256, 104),
            (QuantType::MQ3G256, 104),
            (QuantType::HFQ3G128, 56),
            (QuantType::HFQ2G256, 72),
            (QuantType::MQ2G256, 72),
            (QuantType::MQ2G256Lloyd, 72),
            (QuantType::HFQ2G128, 40),
            (QuantType::MQ8G256, 258),
            (QuantType::MQ3G256Lloyd, 112),
            (QuantType::MQ4G256Lloyd, 160),
            (QuantType::Oq4G256, 130),
            (QuantType::Oq3G256, 98),
            (QuantType::Oq6G256, 194),
            (QuantType::Oq8G256, 258),
            (QuantType::Qtip3G256, 100),
            (QuantType::Qtip4G256, 132),
        ];
        for (qt, bytes) in cases {
            assert_eq!(qt.block_bytes(), Some(bytes), "{qt:?} block_bytes");
        }
    }

    #[test]
    fn variable_layout_formats_have_no_fixed_block_bytes() {
        for qt in [
            QuantType::Q8HFQ,
            QuantType::HFP4G32,
            QuantType::MFP4G32,
            QuantType::OqPlusG256,
            QuantType::OqPlusCompact,
            QuantType::Oq2G256,
            QuantType::Oq4G256ArchPacked,
            QuantType::Qtip2G256,
            QuantType::PARO4G128,
            QuantType::PARO4G128T,
            QuantType::TidI32,
        ] {
            assert_eq!(qt.block_bytes(), None, "{qt:?} must be variable-layout");
            assert_eq!(qt.tensor_bytes(1024), None, "{qt:?} tensor_bytes");
        }
    }

    #[test]
    fn group_size_and_tensor_bytes_compose() {
        assert_eq!(QuantType::MQ4G256.group_size(), 256);
        assert_eq!(QuantType::HFQ4G128.group_size(), 128);
        assert_eq!(QuantType::Q4F16G64.group_size(), 64);
        assert_eq!(QuantType::Q8F16.group_size(), 32);
        assert_eq!(QuantType::F16.group_size(), 1);
        // 300 elems of MQ4G256 = 2 groups of 256 × 136 B.
        assert_eq!(QuantType::MQ4G256.tensor_bytes(300), Some(2 * 136));
        // Exact multiple: 512 elems = 2 groups.
        assert_eq!(QuantType::Oq4G256.tensor_bytes(512), Some(2 * 130));
        // Empty tensor = 0 groups.
        assert_eq!(QuantType::MQ4G256.tensor_bytes(0), Some(0));
    }

    #[test]
    fn key_discriminants_are_stable() {
        // On-disk contract: these bytes must never move.
        assert_eq!(QuantType::F16.code(), 1);
        assert_eq!(QuantType::Qtip3G256.code(), 31);
        assert_eq!(QuantType::OqPlusG256.code(), 33);
        assert_eq!(QuantType::Oq4G256.code(), 34);
        assert_eq!(QuantType::Oq8G256.code(), 35);
        assert_eq!(QuantType::OqPlusCompact.code(), 36);
        assert_eq!(QuantType::Oq4G256ArchPacked.code(), 37);
        assert_eq!(QuantType::Oq3G256.code(), 38);
        assert_eq!(QuantType::Oq2G256.code(), 39);
        assert_eq!(QuantType::Oq6G256.code(), 40);
        assert_eq!(QuantType::Qtip2G256.code(), 41);
        assert_eq!(QuantType::Qtip4G256.code(), 42);
    }
}
