//! hipfire-arch-zaya: Zyphra ZAYA1 architecture for hipfire (Phase 1 scaffold).
//!
//! Status: intake scaffold landed 2026-05-07 on `feat/zaya1-port-intake`.
//! Implements the [`hipfire_runtime::arch::Architecture`] trait surface
//! with stubs that return `Err` from the load and state-alloc paths.
//! Real loading + forward come after the recurrent-state design lands
//! (Phase 6 of the intake contract).
//!
//! See docs/investigations/2026-05-07-zaya1-port-intake/ for the port plan,
//! Phase 0 disambiguation verdict (CCA is RECURRENT), and the implication
//! that hipfire-runtime grows a per-layer recurrent-state primitive before
//! ZAYA1 can run end-to-end.
//!
//! Architectural elements (from Zyphra/transformers@zaya1):
//!   - CCA (Compressed Convolutional Attention) with two per-layer
//!     state buffers per sequence: `conv_states` and `prev_hs`.
//!   - Standard GQA attention downstream of CCA's Q/K/V outputs.
//!   - 16-expert MoE, top-1 routing, MLP-based router.
//!   - MoD (Mixture-of-Depths) per-token layer-skip routing.
//!   - partial_rotary_factor=0.5 (NeoX/GPT-J-style half-RoPE).
//!   - scale_residual_merge (learnable per-block residual scalar).
//!   - EDA component (zaya_use_eda=true; identification pending Phase 5).

pub mod arch;
pub mod config;
pub mod forward;
pub mod state;
pub mod weights;

pub use arch::Zaya;
pub use config::ZayaConfig;
pub use state::ZayaState;
pub use weights::ZayaWeights;
