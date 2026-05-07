//! ZayaWeights: GPU-resident model weights for ZAYA1.
//!
//! Phase 1 status: TYPE STUB. `load` returns Err; per-layer weight
//! tensors are unspecified pending HFQ-side quantizer support and the
//! Phase 6 recurrent-state plumbing decision.
//!
//! Per-layer weight inventory (from Zyphra/transformers@zaya1
//! modeling_zaya.py:1197 ZayaBlock):
//!   - CCA: linear_q, linear_k, val_proj1, val_proj2, conv_qk[0], conv_qk[1],
//!          temp (per-kv-head scalar)
//!   - Standard attention: q_proj, k_proj, v_proj, o_proj (16q/2kv layout)
//!   - Norms: 4× RMSNorm per block (pre-CCA, pre-attn, pre-mlp, pre-mod-router?)
//!     Exact set TBD by Phase 1 read of ZayaBlock.
//!   - MoE: 16× expert FFN (gate/up/down per expert, SwiGLU), MLP router
//!     (2-layer with `zaya_mlp_expansion=256` hidden), shared expert(?)
//!   - MoD: per-block top-k token-router weights (Phase 4)
//!   - Residual scaling: per-block learnable scalar (`scale_residual_merge`)
//!   - EDA: TBD pending Phase 5
//! Plus model-global: token_embed (262272 × 2048), final_norm, lm_head
//! (tied to embed per `tie_word_embeddings=true`).

use crate::config::ZayaConfig;
use hipfire_runtime::hfq::HfqFile;

/// Per-decode model weights for ZAYA1.
///
/// Phase 1: opaque placeholder. Real fields populate as forward-pass
/// kernels come online (Phase 2 free components, then Phase 3/6 CCA).
pub struct ZayaWeights {
    pub config: ZayaConfig,
    /// Total bytes uploaded (for accounting; populated by `load`).
    pub uploaded_bytes: usize,
}

impl ZayaWeights {
    /// Load ZAYA1 weights from an HFQ file.
    ///
    /// Phase 1 status: returns Err. The HFQ representation for ZAYA1
    /// requires (a) the quantizer learning the Zaya tensor naming
    /// convention from the safetensors checkpoint, and (b) any new
    /// quant-format slots the recurrent state's fp16 conv_states /
    /// prev_hs need. Both pending Phase 6 decisions.
    pub fn load(_hfq: &mut HfqFile, _cfg: &ZayaConfig) -> Result<Self, String> {
        Err(
            "ZayaWeights::load not implemented (Phase 1 scaffold). HFQ \
             representation for ZAYA1 pending quantizer support; see \
             docs/investigations/2026-05-07-zaya1-port-intake/."
                .to_string(),
        )
    }
}
