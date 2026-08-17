//! Reference construction and candidate scoring.
//!
//! # Device is the target, not the fallback
//!
//! The intended execution target for reference building and calibration is a
//! datacentre GPU (gfx942, 205 GB HBM at ~5.3 TB/s). The CPU path in
//! [`estimator`] exists so someone without a device can still produce a
//! reference overnight — it is the degraded mode, not the design centre.
//!
//! What made the previous pipeline slow was not the model. Per scored
//! position `build_kld_ref_native` ran a **single-token** `forward_scratch`
//! (re-reading all 53.8 GB of BF16 weights for one token), downloaded the
//! **entire** 248,320-wide logit row — 993 KB, ~97.5 GB of D2H across a
//! 96-chunk run — and then, on one CPU thread, took 248,320 `f64::exp()`
//! calls, materialised a 248,320-element `Vec<(u32, f32)>`, and ran
//! `select_nth_unstable_by` over all of it. Measured: **16 tok/s**, against a
//! ~98 tok/s ceiling implied by weight bandwidth alone, on a device that
//! reported 100% busy because it was saturated with tiny kernels.
//!
//! The device-side shape is: batched prefill over the whole chunk (weights
//! read once, not once per position), one batched lm_head GEMM, then top-k +
//! log-sum-exp **on device**, returning only the `k` pairs the reference
//! format actually stores. At `k = 256` that is ~2 KB per position instead of
//! 993 KB — a **484x** cut in D2H — and the transcendentals never touch the
//! host.
//!
//! # Kernel status
//!
//! `kernels/src/topk_logsumexp_batched.hip` already implements exactly this
//! operation, batched, and its own header says it exists "to avoid a 20 ms CPU
//! sort" — but it is hard-capped at `MAX_K = 8`. Its algorithm does not scale
//! to `k = 256`: per-thread `loc_val[K]`/`loc_idx[K]` would need ~2 KB of
//! registers per thread, its LDS candidate buffer would need 512 KB against a
//! 64 KB budget, and its final merge is serial on thread 0 over `nth * K`
//! candidates. A `k >= 256` variant is required, and the proven in-tree
//! precedent is the sampler's two-stage `sample_topk_partial` ->
//! `sample_topk_finalize` split, which already spans blocks at TOP_K up to 65.
//!
//! [`estimator::TopKBlock`] is the contract that variant must satisfy. The CPU
//! implementation here is the oracle it is validated against.

pub mod estimator;
pub mod teacher;
