//! Quantization sidecars and runtime calibration data.
//!
//! Currently provides:
//!   - `iu4_calibration` — runtime SmoothQuant-style activation calibration
//!     for the gfx12 iu4 K=32 GEMM path. Sidecar produced offline by
//!     `examples/calibrate_iu4_activations.rs`, loaded at model-load time
//!     when `<model>.iu4cal` exists alongside the model file.

pub mod iu4_calibration;
