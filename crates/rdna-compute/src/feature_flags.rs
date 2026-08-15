// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Typed, immutable startup-policy resolution for rdna-compute.
//!
//! The native daemon installs a schema-validated process configuration before
//! `Gpu::init()`. Dispatching hot paths access `self.flags.*` instead of
//! parsing TOML or consulting ambient process state on every call.

use hipfire_config::ProcessConfig;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mb4Mode {
    Pack1,
    Pack2,
    Pack4,
}

#[derive(Debug, Clone)]
pub struct FeatureFlags {
    // ── Arch identity ──────────────────────────────────────────────
    pub arch: String,

    // ── GEMV tuning ────────────────────────────────────────────────
    pub gemv_rows: Option<u32>,
    pub gemv_dp4a_default_on: bool,
    pub gemv_prefetch: Option<bool>,
    pub gemv_prefetch_default_on: bool,
    pub gfx942_lds_gemv: Option<bool>,
    pub gfx942_lds_gemv_default_on: bool,
    pub gemv_rows_default: u32,
    pub gemv_dp4a: Option<bool>,
    /// Override the gfx1151 temporal raw-buffer E8 U4 route. `None` selects
    /// the accepted DeepSeek4 MQ2R route default while retaining portable
    /// behavior for other models and architectures.
    pub gfx1151_e8_buffer: Option<bool>,

    // ── Quant / format toggles ────────────────────────────────────
    pub hfq3_dp4a: Option<bool>,
    pub hfq3_mmq: Option<bool>,
    pub hfq4_mmq_rdna2: Option<bool>,
    pub fp8_wmma: bool,
    pub dot2_gemv: bool,
    pub gcn5_wave64_hybrid: Option<bool>,
    /// Radiowave experiment: pack two independent HFQ4 QKV rows into one
    /// explicitly compiled wave64 on RDNA3. Default off until model-level
    /// correctness and throughput gates promote it.
    pub rdna3_hfq4_qkv_wave64: bool,
    /// Radiowave experiment: split each gfx1100 QKVZA output row across two
    /// lighter wave32s and join their partials through LDS.
    pub rdna3_hfq4_qkvza_2wave: bool,
    /// Radiowave experiment: package four independent gfx1100 QKVZA row
    /// waves in one workgroup without changing per-row arithmetic.
    pub rdna3_hfq4_qkvza_wavepack4: bool,
    /// Stage the K=2048 QKVZA activation once per eight exact row waves in a
    /// transposed LDS tile on gfx1100.
    pub rdna3_hfq4_qkvza_ldsx8: bool,
    /// Radiowave recovery of the exact gfx1100 autoresearch schedule: keep an
    /// explicit row stride and spell out the five shuffle-reduction stages so
    /// LLVM cannot re-form the reduction as a loop.
    pub rdna3_hfq4_qkvza_reduce_chain: bool,
    /// Radiowave experiment: port the gfx12 QKVZA float4 activation-hoist
    /// schedule to gfx1100 while retaining one independent wave per row.
    pub rdna3_hfq4_qkvza_hoist_x32: bool,
    /// Compile the exact A3B decode shape with eight HFQ4 groups fixed at
    /// compile time, deleting the general QKVZA loop and tail machinery.
    /// Certified default on gfx1100; set
    /// `HIPFIRE_RDNA3_HFQ4_QKVZA_K2048=0` to restore the general kernel.
    pub rdna3_hfq4_qkvza_k2048: bool,
    /// Stage one residual GEMV row's 32 activation values ahead of the
    /// independent packed-weight loads. Certified default on exact gfx1100;
    /// other RDNA3 targets remain explicit opt-in until separately measured.
    pub rdna3_hfq4_residual_stage_x32: bool,
    /// Combine the certified activation staging with a fixed eight-group
    /// K=2048 loop on gfx1100. Opt-in until exact and tg128 gates promote it.
    pub rdna3_hfq4_residual_k2048: bool,
    /// Launch only the active half of the two-row shared-expert down grid.
    /// The kernel maps row0 = blockIdx.x * 2; the legacy launcher submits M
    /// workgroups, leaving the upper half to return immediately.
    pub rdna3_hfq4_sigmoid_tight_grid: bool,
    /// Compile the shared-expert sigmoid-scaled down kernel with gfx1100
    /// temporal buffer-resource weight loads instead of global addressing.
    /// Default-on for gfx1100; set `HIPFIRE_RDNA3_HFQ4_SIGMOID_BUFFER=0`
    /// to restore global addressing for comparison.
    pub rdna3_hfq4_sigmoid_buffer: bool,
    /// Schedule four shared-expert down rows per wave on gfx1100, reusing
    /// each activation load across four independent weight streams.
    pub rdna3_hfq4_sigmoid_rows4: bool,
    /// Compile the exact Qwen3.5-35B LM-head shape (M=248320, K=2048) with
    /// its eight HFQ4 groups fixed at compile time. gfx1100 experiment only.
    pub rdna3_hfq4_lm_head_k2048: bool,
    /// Compile the gfx1100 A3B MoE gate/up GEMV with its eight K groups fixed
    /// at compile time. Opt-in until exact shadow and tg128 gates promote it.
    pub rdna3_hfq4_moe_gate_up_k2048: bool,
    pub mmq_override: Option<bool>,
    pub mmq_min_batch: Option<usize>,
    pub fp16_disabled: bool,
    pub fp16_layer_min: Option<usize>,
    pub fp16_layer_max: Option<usize>,
    pub wo_mmq: bool,
    pub lm_head_wmma_disabled: bool,
    pub lm_head_overwrite: bool,

    // ── MMQ screening ─────────────────────────────────────────────
    pub mmq_screen: bool,
    pub mmq_screen_threshold: f32,
    pub mmq_diag_quantize_only: bool,

    // ── Kernel variant overrides ─────────────────────────────────
    pub lloyd_mb4: Option<Mb4Mode>,
    pub mq3_mb4: Option<Mb4Mode>,
    pub hfq4g128_mmq: bool,
    pub hfq3_mmq_layer_min: Option<usize>,
    pub hfq3_mmq_layer_max: Option<usize>,
    pub hfq4_mmq_gfx906_y64: bool,
    pub gate_up_variant: Option<String>,
    pub gate_up_nosync: bool,
    /// gfx12 LDS-staged HFQ4-G256 residual/gate_up WMMA path. Reorders FP32
    /// K accumulation across 8 waves (not bit-exact). Default OFF; opt in with
    /// `HIPFIRE_HFQ4G256_LDSSTAGE=1`. Requires K % 512 == 0 at the launch site.
    pub hfq4g256_ldsstage_wmma: bool,
    /// RDNA3 QKVZA prefill route that keeps the large QKV/Z projections on
    /// MMQ while sending the narrow beta/alpha tails through dot2.
    /// Opt in with HIPFIRE_QKVZA_SPLIT_TAIL=1.
    pub qkvza_split_tail: bool,
    pub gfx942_gemv_v2: Option<bool>,
    pub gfx942_gemv_v3: bool,
    pub gfx942_rmsnorm_split: bool,
    /// Reserve only the reduction scratch still used by the current fused
    /// RMSNorm+MQ kernel. The historical K-float LDS cache was removed when
    /// the prefetch schedule landed, but its launch reservation remained.
    pub rmsnorm_mq_tight_lds: bool,
    /// Radiowave experiment: split the K=2048 fused RMSNorm+MQ rotation over
    /// eight resident wave32 workgroups on gfx1100. The workgroups retain the
    /// baseline reduction order and rendezvous through persistent GPU scratch.
    pub rdna3_rmsnorm_wavegrid: bool,
    /// Radiowave experiment: replace the cross-workgroup rendezvous with one
    /// exact reduction dispatch followed by eight independent rotate waves.
    pub rdna3_rmsnorm_split: bool,
    /// Compute the K=2048 RMS sum from each FWHT wave's prefetched float4
    /// values, eliminating the baseline's second x load. Certified default on
    /// gfx1100; set `HIPFIRE_RDNA3_RMSNORM_VECSUM=0` to restore the baseline.
    pub rdna3_rmsnorm_vecsum: bool,
    /// Radiowave experiment: stage the two shared MQ sign tables once in LDS
    /// for the gfx1100 vecsum kernel instead of reloading them in eight waves.
    pub rdna3_rmsnorm_sign_lds: bool,
    /// Radiowave experiment: compile the deterministic MQ sign streams as
    /// packed constants and apply them with FP32 sign-bit XORs.
    pub rdna3_rmsnorm_sign_const: bool,
    pub gfx942_mfma_prefill: Option<String>,
    pub moe_grouped_i8: Option<bool>,
    pub moe_grouped_i8_k8: bool,
    pub moe_grouped_i8_k4: bool,
    pub moe_grouped_i8_k4_gfx12: bool,
    pub moe_grouped_m2: bool,
    pub moe_grouped_4w: bool,
    /// Radiowave experiment: fold four adjacent K=8 MoE down outputs per
    /// thread with vector loads and independent exact accumulators.
    pub moe_down_combine_vec4: bool,
    pub moe_hfq6_i8: bool,
    pub moe_hfq6_v2: bool,
    // ── MoE prefill (Ship 4.2) ────────────────────────────────────
    /// Grouped-GEMM MoE prefill gate (HIPFIRE_MOE_GROUPED_GEMM). Default ON.
    pub moe_grouped_gemm: bool,
    /// gfx1151 i8 MMQ opt-in for Paro grouped GEMM (HIPFIRE_MOE_PARO_I8).
    /// None = arch-default (gfx1151 → true, else false).
    pub moe_paro_i8: Option<bool>,
    /// gfx1151 i8 MMQ k8 opt-in for Paro grouped GEMM (HIPFIRE_MOE_PARO_I8_K8).
    /// None = arch-default (gfx1151 → true, else false).
    pub moe_paro_i8_k8: Option<bool>,

    // ── Graph / capture / deterministic ─────────────────────────────
    pub graph_forward: Option<bool>,
    pub graph_ar: bool,
    pub graph_moe: bool,
    pub force_blob_path: bool,
    pub gemm_dump: bool,
    pub deterministic: bool,
    pub mw16: bool,
    pub q8_batched_legacy: bool,
    /// Fuse Gemma 4 Q8 prefill projections on exact gfx1100. This remains an
    /// opt-in while the E-series path is validated across QKV-sharing shapes.
    pub gemma4_q8_fused_prefill: bool,
    /// Batch Gemma 4 main and PLE embedding lookup on validated gfx1100 and
    /// gfx1201 paths. The `auto` policy remains independently disableable.
    pub gemma4_batched_embedding_prefill: bool,
    /// Batch the E-series per-layer-input model projection on exact gfx1100
    /// instead of re-streaming the same matrix through one GEMV per row.
    pub gemma4_ple_batched_prefill: bool,
    /// Experimental batched E-series PLE branch projections. Explicit opt-in:
    /// the WMMA route can change long greedy trajectories.
    pub gemma4_ple_branch_batched_prefill: bool,
    /// Fuse the E-series PLE GELU and strided per-layer multiply on validated
    /// gfx1100 and gfx1201 paths.
    pub gemma4_ple_activation_fused_prefill: bool,
    /// `HIPFIRE_DEEPSEEK4_Q8_WMMA=0` disables the Q8_0 dense WMMA prefill path
    /// (forces the scalar chunked fallback). Consumed by
    /// `gemm_q8_0_wmma_prefill_auto`.
    pub deepseek4_q8_wmma_off: bool,
    /// `HIPFIRE_DEEPSEEK4_Q8_4W=0` forces the single-warp 16×16 Q8_0 WMMA
    /// kernel instead of the 4-warp 64×64 tile. Consumed by
    /// `gemm_q8_0_wmma_prefill_auto`.
    pub deepseek4_q8_4w_off: bool,
    pub rope_interleaved_legacy: bool,
    pub wo_wmma_variant: Option<String>,

    // ── rocBLAS ────────────────────────────────────────────────────
    pub rocblas_all_archs: bool,
    pub rocblas_off: bool,
    pub rocblas_min_batch: Option<usize>,
    /// Batched-attention tile size (`HIPFIRE_ATTN_TILE_SIZE`). `None` = use the
    /// 128 default. gfx1151 is the dev box, gfx1201 the deployment target, so
    /// this must never be a baked-in `const` (spec §11).
    pub attn_tile_size: Option<usize>,
    /// Multi-slot attention flash-vs-scalar crossover in tokens
    /// (`HIPFIRE_SLOTS_ATTN_CROSSOVER`). `None` = the per-arch default. Same
    /// reasoning as `attn_tile_size`: gfx1151 is the dev box and gfx1201 the
    /// deployment target, so this must be overridable rather than baked in.
    pub slots_attn_crossover: Option<usize>,
    /// Capture a pure-decode multi-slot step into a hipGraph and replay it
    /// (`HIPFIRE_SLOTS_DECODE_GRAPH`). Off by default.
    pub slots_decode_graph: bool,
    /// Trace multi-slot session continuation matching (`HIPFIRE_SLOT_TRACE`).
    /// Diagnostic only.
    pub slot_trace: bool,

    // ── Kernels.rs env reads ───────────────────────────────────────
    pub lloyd_force_baseline: bool,
    pub rdna2_variant: Option<u32>,

    // ── Compiler.rs env reads ──────────────────────────────────────
    pub hipcc_extra_flags: String,

    // ── Interpreter Phase 2a ───────────────────────────────────────
    /// Force the discrete (un-fused) projection path where supported, for
    /// fused-vs-unfused validation. Env: HIPFIRE_FORCE_UNFUSED=1. Single-GPU
    /// decode projection fusions only (see Phase-2a spec §4b honest-scope).
    pub force_unfused: bool,

    // ── Speculative decode (DFlash/DDTree) ────────────────────────
    /// DDTree tree-SWOR verify arm. **Default OFF** — linear chain wins on
    /// every drafter measured (the DFlash drafter's independent per-position
    /// marginals give a tree branch no joint to exploit). Opt in with the CLI
    /// `--ddtree` flag (sets `HIPFIRE_DFLASH_TREE=1`). Tree-SWOR is still
    /// distribution-exact at any temperature and lossless (== AR) at temp 0 —
    /// it's just slower than chain — so the opt-in path stays correct.
    pub dflash_tree: bool,
    /// Override DDTree node budget (`HIPFIRE_DDTREE_BUDGET`). `None` → use the
    /// per-call-site default (`DEFAULT_TREE_BUDGET = 8`).
    pub ddtree_budget: Option<usize>,
    /// Override DDTree per-position top-K breadth (`HIPFIRE_DDTREE_TOPK`).
    /// `None` → use the per-call-site default (`DEFAULT_TREE_TOPK = 2`).
    pub ddtree_topk: Option<usize>,
    // D8: ddtree_verify_naive removed — SWOR is the only temp>0 verify path.
    // The naive fallback required a ~37 MB/cycle full-logits D2H and is
    // superseded by SWOR (distribution-exact, on-GPU). HIPFIRE_DDTREE_VERIFY
    // env var is no longer parsed; setting it has no effect.
    /// qwen35 ddtree tree-LA (linearized-ancestor) fast-tape path. **Default ON**;
    /// opt out with `HIPFIRE_DDTREE_TREE_LA=0`.
    pub ddtree_tree_la: bool,
    /// qwen35 DFlash GPU softmax/nucleus fast path on the temp>0 sampled verify.
    /// **Default ON**; opt out with `HIPFIRE_DFLASH_FAST_SAMPLE=0`.
    pub dflash_fast_sample: bool,
    /// qwen35 ddtree meta-verifier expansion cutoff (`HIPFIRE_DDTREE_LOGW_CUTOFF`).
    /// Stores the positive X the user set (stop tree expansion when the next
    /// candidate's cumulative logw < −X); `None`/0/unparseable → no cutoff.
    /// Use [`FeatureFlags::ddtree_logw_cutoff_value`] for the resolved threshold.
    pub ddtree_logw_cutoff: Option<f32>,
    /// qwen35 DFlash Q8 WMMA lm_head in verify. **Default ON**; opt out with
    /// `HIPFIRE_DFLASH_Q8_LMHEAD_WMMA` ∈ {0,false,off,no}.
    pub dflash_q8_lmhead_wmma: bool,

    /// Fold the 3 separate QKV `BiasAdd` launches (qwen2-family `attention_bias`)
    /// into the per-row fused-QKV decode kernel's lane-0 store. Eliminates 3 tiny
    /// overhead-dominated launches/layer. Env: HIPFIRE_FUSE_QKV_BIAS — default ON;
    /// set `=0` to opt out (for bisection). NULL bias pointers are a kernel no-op,
    /// so non-bias callers stay byte-identical regardless. Only fires for fused-QKV
    /// keys whose dispatch arm reads the bias (see `qkv_bias_fold_supported`).
    pub fuse_qkv_bias: bool,

    /// Diagnostic: log each QKV-bias fold to stderr. Env:
    /// HIPFIRE_FUSE_QKV_BIAS_DEBUG=1. Default off. Resolved once at init so the
    /// default-on fold hot path takes no per-launch `env::var` lock.
    pub fuse_qkv_bias_debug: bool,
}

impl FeatureFlags {
    pub fn from_process_config(arch: &str, config: &ProcessConfig) -> Self {
        Self::from_lookup(arch, |name| config.legacy_value(name).ok_or(()))
    }

    pub fn from_active_config(arch: &str) -> Self {
        Self::from_process_config(arch, hipfire_config::active_or_local_process_config())
    }

    fn from_lookup(arch: &str, value: impl Fn(&str) -> std::result::Result<String, ()>) -> Self {
        let parse_bool = |name: &str| -> Option<bool> {
            match value(name).ok().as_deref() {
                Some("1") | Some("true") | Some("TRUE") | Some("on") | Some("ON") => Some(true),
                Some("0") | Some("false") | Some("FALSE") | Some("off") | Some("OFF") => {
                    Some(false)
                }
                _ => None,
            }
        };

        let parse_usize =
            |name: &str| -> Option<usize> { value(name).ok().and_then(|s| s.parse().ok()) };

        let parse_mb4 = |name: &str| -> Option<Mb4Mode> {
            match value(name).ok().as_deref() {
                Some("1") => Some(Mb4Mode::Pack1),
                Some("2") => Some(Mb4Mode::Pack2),
                Some("4") => Some(Mb4Mode::Pack4),
                _ => None,
            }
        };

        let is_gfx906 = arch == "gfx906";

        let mmq_screen_default: bool = false;
        let mmq_screen_threshold_default: f32 = if is_gfx906 { 0.50 } else { 0.10 };

        let gemv_rows_default: u32 = match arch {
            "gfx1100" | "gfx1101" | "gfx1102" => 1,
            "gfx1030" | "gfx1031" => 1,
            "gfx906" | "gfx908" | "gfx940" | "gfx941" | "gfx942" => 1,
            _ => 2,
        };

        let mut hipcc_extra_flags = value("HIPFIRE_HIPCC_EXTRA_FLAGS").unwrap_or_default();
        let mut append_hipcc_flag = |flag: &str| {
            if !hipcc_extra_flags.is_empty() {
                hipcc_extra_flags.push(' ');
            }
            hipcc_extra_flags.push_str(flag);
        };
        if arch == "gfx1100" {
            match value("HIPFIRE_GFX11_WEIGHT_LOAD_POLICY").ok().as_deref() {
                None | Some("") | Some("buffer") => {}
                Some("global") => append_hipcc_flag("-DHIPFIRE_GFX11_WEIGHT_GLOBAL_LOADS=1"),
                Some("flat-buffer") => {
                    append_hipcc_flag("-DHIPFIRE_WEIGHT_BUFFER_LOADS_FLAT_GEMV_OPT_IN=1")
                }
                Some(other) => {
                    eprintln!("unknown HIPFIRE_GFX11_WEIGHT_LOAD_POLICY={other:?}; using buffer")
                }
            }
        }
        if arch == "gfx1201" {
            let policy_flag = match value("HIPFIRE_GFX12_WEIGHT_LOAD_POLICY").ok().as_deref() {
                None | Some("") | Some("rt") => None,
                Some("global") => Some("-DHIPFIRE_GFX12_WEIGHT_GLOBAL_LOADS=1"),
                Some("ht") => Some("-DHIPFIRE_GFX12_WEIGHT_CPOL_AUX=18"),
                Some("nt-rt") => Some("-DHIPFIRE_GFX12_WEIGHT_CPOL_AUX=20"),
                Some("nt-ht") => Some("-DHIPFIRE_GFX12_WEIGHT_CPOL_AUX=22"),
                Some(other) => {
                    eprintln!("unknown HIPFIRE_GFX12_WEIGHT_LOAD_POLICY={other:?}; using rt");
                    None
                }
            };
            if let Some(flag) = policy_flag {
                append_hipcc_flag(flag);
            }
        }

        Self {
            arch: arch.to_string(),

            // GEMV tuning
            gemv_rows: value("HIPFIRE_GEMV_ROWS")
                .ok()
                .and_then(|v| v.parse::<u32>().ok())
                .map(|r| match r {
                    1 | 2 | 4 | 8 => r,
                    _ => 1,
                }),
            gemv_dp4a_default_on: is_gfx906,
            gemv_dp4a: parse_bool("HIPFIRE_GEMV_DP4A"),
            gfx1151_e8_buffer: parse_bool("HIPFIRE_GFX1151_E8_BUFFER"),
            gemv_prefetch: parse_bool("HIPFIRE_GEMV_PREFETCH"),
            gemv_prefetch_default_on: is_gfx906,
            gfx942_lds_gemv: parse_bool("HIPFIRE_GFX942_LDS_GEMV"),
            gfx942_lds_gemv_default_on: false,
            gemv_rows_default,

            // Quant/format toggles
            hfq3_dp4a: parse_bool("HIPFIRE_HFQ3_DP4A"),
            hfq3_mmq: parse_bool("HIPFIRE_HFQ3_MMQ"),
            hfq4_mmq_rdna2: parse_bool("HIPFIRE_HFQ4_MMQ_RDNA2"),
            fp8_wmma: value("HIPFIRE_FP8_WMMA").map_or(false, |v| v == "1"),
            dot2_gemv: value("HIPFIRE_DOT2_GEMV").map_or(false, |v| v == "1"),
            gcn5_wave64_hybrid: parse_bool("HIPFIRE_GCN5_WAVE64_HYBRID"),
            rdna3_hfq4_qkv_wave64: value("HIPFIRE_RDNA3_HFQ4_QKV_WAVE64").as_deref() == Ok("1"),
            rdna3_hfq4_qkvza_2wave: value("HIPFIRE_RDNA3_HFQ4_QKVZA_2WAVE").as_deref() == Ok("1"),
            rdna3_hfq4_qkvza_wavepack4: value("HIPFIRE_RDNA3_HFQ4_QKVZA_WAVEPACK4").as_deref()
                == Ok("1"),
            rdna3_hfq4_qkvza_ldsx8: value("HIPFIRE_RDNA3_HFQ4_QKVZA_LDSX8").as_deref() == Ok("1"),
            rdna3_hfq4_qkvza_reduce_chain: value("HIPFIRE_RDNA3_HFQ4_QKVZA_REDUCE_CHAIN")
                .as_deref()
                == Ok("1"),
            rdna3_hfq4_qkvza_hoist_x32: value("HIPFIRE_RDNA3_HFQ4_QKVZA_HOIST_X32").as_deref()
                == Ok("1"),
            rdna3_hfq4_qkvza_k2048: parse_bool("HIPFIRE_RDNA3_HFQ4_QKVZA_K2048")
                .unwrap_or(arch == "gfx1100"),
            rdna3_hfq4_residual_stage_x32: parse_bool("HIPFIRE_RDNA3_HFQ4_RESIDUAL_STAGE_X32")
                .unwrap_or(arch == "gfx1100"),
            rdna3_hfq4_residual_k2048: value("HIPFIRE_RDNA3_HFQ4_RESIDUAL_K2048").as_deref()
                == Ok("1"),
            rdna3_hfq4_sigmoid_tight_grid: value("HIPFIRE_RDNA3_HFQ4_SIGMOID_TIGHT_GRID")
                .as_deref()
                == Ok("1"),
            rdna3_hfq4_sigmoid_buffer: parse_bool("HIPFIRE_RDNA3_HFQ4_SIGMOID_BUFFER")
                .unwrap_or(arch == "gfx1100"),
            rdna3_hfq4_sigmoid_rows4: value("HIPFIRE_RDNA3_HFQ4_SIGMOID_ROWS4").as_deref()
                == Ok("1"),
            rdna3_hfq4_lm_head_k2048: value("HIPFIRE_RDNA3_HFQ4_LM_HEAD_K2048").as_deref()
                == Ok("1"),
            rdna3_hfq4_moe_gate_up_k2048: value("HIPFIRE_RDNA3_HFQ4_MOE_GATE_UP_K2048").as_deref()
                == Ok("1"),
            mmq_override: match value("HIPFIRE_MMQ").ok().as_deref() {
                Some("0") | Some("off") => Some(false),
                Some("1") | Some("on") => Some(true),
                _ => None,
            },
            mmq_min_batch: parse_usize("HIPFIRE_MMQ_MIN_BATCH"),
            fp16_disabled: value("HIPFIRE_FP16").map_or(false, |v| v == "0"),
            fp16_layer_min: parse_usize("HIPFIRE_FP16_LAYER_MIN"),
            fp16_layer_max: parse_usize("HIPFIRE_FP16_LAYER_MAX"),
            wo_mmq: value("HIPFIRE_WO_MMQ").ok().as_deref() == Some("1"),
            lm_head_wmma_disabled: value("HIPFIRE_LM_HEAD_WMMA").map_or(false, |v| v == "0"),
            lm_head_overwrite: value("HIPFIRE_LM_HEAD_OVERWRITE").as_deref() == Ok("1"),

            // MMQ screening
            mmq_screen: value("HIPFIRE_MMQ_SCREEN")
                .ok()
                .map(|v| v == "1")
                .unwrap_or(mmq_screen_default),
            mmq_screen_threshold: value("HIPFIRE_MMQ_SCREEN_THRESHOLD")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(mmq_screen_threshold_default),
            mmq_diag_quantize_only: value("HIPFIRE_MMQ_DIAG_QUANTIZE_ONLY").ok().as_deref()
                == Some("1"),

            // Kernel variant overrides
            lloyd_mb4: parse_mb4("HIPFIRE_LLOYD_MB4"),
            mq3_mb4: parse_mb4("HIPFIRE_MQ3_MB4"),
            hfq4g128_mmq: value("HIPFIRE_HFQ4G128_MMQ").as_deref() != Ok("0"),
            hfq3_mmq_layer_min: parse_usize("HIPFIRE_HFQ3_MMQ_LAYER_MIN"),
            hfq3_mmq_layer_max: parse_usize("HIPFIRE_HFQ3_MMQ_LAYER_MAX"),
            hfq4_mmq_gfx906_y64: value("HIPFIRE_HFQ4_MMQ_GFX906_Y64").map_or(false, |v| v == "1"),
            gate_up_variant: value("HIPFIRE_GATE_UP_VARIANT").ok(),
            gate_up_nosync: value("HIPFIRE_GATE_UP_NOSYNC").as_deref() == Ok("1"),
            hfq4g256_ldsstage_wmma: value("HIPFIRE_HFQ4G256_LDSSTAGE").as_deref() == Ok("1"),
            qkvza_split_tail: parse_bool("HIPFIRE_QKVZA_SPLIT_TAIL").unwrap_or(false),
            gfx942_gemv_v2: parse_bool("HIPFIRE_GFX942_GEMV_V2"),
            gfx942_gemv_v3: value("HIPFIRE_GFX942_GEMV_V3").map_or(false, |v| v == "1"),
            gfx942_rmsnorm_split: matches!(arch, "gfx940" | "gfx941" | "gfx942")
                && value("HIPFIRE_GFX942_RMSNORM_SPLIT").as_deref() != Ok("0"),
            rmsnorm_mq_tight_lds: value("HIPFIRE_RMSNORM_MQ_TIGHT_LDS").as_deref() == Ok("1"),
            rdna3_rmsnorm_wavegrid: value("HIPFIRE_RDNA3_RMSNORM_WAVEGRID").as_deref() == Ok("1"),
            rdna3_rmsnorm_split: value("HIPFIRE_RDNA3_RMSNORM_SPLIT").as_deref() == Ok("1"),
            rdna3_rmsnorm_vecsum: parse_bool("HIPFIRE_RDNA3_RMSNORM_VECSUM")
                .unwrap_or(arch == "gfx1100"),
            rdna3_rmsnorm_sign_lds: value("HIPFIRE_RDNA3_RMSNORM_SIGN_LDS").as_deref() == Ok("1"),
            rdna3_rmsnorm_sign_const: value("HIPFIRE_RDNA3_RMSNORM_SIGN_CONST").as_deref()
                == Ok("1"),
            gfx942_mfma_prefill: value("HIPFIRE_GFX942_MFMA_PREFILL").ok(),
            moe_grouped_i8: match value("HIPFIRE_MOE_GROUPED_I8").ok().as_deref() {
                Some("1") => Some(true),
                Some("0") => Some(false),
                _ => None,
            },
            moe_grouped_i8_k8: value("HIPFIRE_MOE_GROUPED_I8_K8").as_deref() == Ok("1"),
            moe_grouped_i8_k4: value("HIPFIRE_MOE_GROUPED_I8_K4").as_deref() == Ok("1"),
            moe_grouped_i8_k4_gfx12: value("HIPFIRE_MOE_GROUPED_I8_K4_GFX12").as_deref() == Ok("1"),
            moe_grouped_m2: value("HIPFIRE_MOE_GROUPED_M2").as_deref() == Ok("1"),
            moe_grouped_4w: value("HIPFIRE_MOE_GROUPED_4W").as_deref() == Ok("1"),
            moe_down_combine_vec4: value("HIPFIRE_MOE_DOWN_COMBINE_VEC4").as_deref() == Ok("1"),
            moe_hfq6_i8: value("HIPFIRE_MOE_HFQ6_I8").as_deref() == Ok("1"),
            moe_hfq6_v2: value("HIPFIRE_MOE_HFQ6_V2").as_deref() == Ok("1"),
            // MoE prefill (Ship 4.2)
            moe_grouped_gemm: match value("HIPFIRE_MOE_GROUPED_GEMM").ok().as_deref() {
                Some("0") | Some("off") => false,
                _ => true,
            },
            moe_paro_i8: parse_bool("HIPFIRE_MOE_PARO_I8"),
            moe_paro_i8_k8: parse_bool("HIPFIRE_MOE_PARO_I8_K8"),

            // Graph / capture / deterministic
            graph_forward: parse_bool("HIPFIRE_GRAPH"),
            graph_ar: value("HIPFIRE_AR_GRAPH").ok().as_deref() != Some("0"),
            graph_moe: value("HIPFIRE_GRAPH_MOE").ok().as_deref() != Some("0"),
            force_blob_path: value("HIPFIRE_BLOB_FORCE").ok().as_deref() == Some("1"),
            gemm_dump: value("HIPFIRE_GEMM_DUMP").ok().as_deref() == Some("1"),
            deterministic: value("HIPFIRE_DETERMINISTIC").ok().as_deref() == Some("1"),
            mw16: value("HIPFIRE_MW16").map_or(false, |v| v == "1"),
            q8_batched_legacy: value("HIPFIRE_Q8_BATCHED_LEGACY").as_deref() == Ok("1"),
            gemma4_q8_fused_prefill: parse_bool("HIPFIRE_GEMMA4_Q8_FUSED_PREFILL").unwrap_or(false),
            gemma4_batched_embedding_prefill: parse_bool(
                "HIPFIRE_GEMMA4_BATCHED_EMBEDDING_PREFILL",
            )
            .unwrap_or(matches!(arch, "gfx1100" | "gfx1201")),
            gemma4_ple_batched_prefill: parse_bool("HIPFIRE_GEMMA4_PLE_BATCHED_PREFILL")
                .unwrap_or(false),
            gemma4_ple_branch_batched_prefill: parse_bool(
                "HIPFIRE_GEMMA4_PLE_BRANCH_BATCHED_PREFILL",
            )
            .unwrap_or(false),
            gemma4_ple_activation_fused_prefill: parse_bool(
                "HIPFIRE_GEMMA4_PLE_ACTIVATION_FUSED_PREFILL",
            )
            .unwrap_or(matches!(arch, "gfx1100" | "gfx1201")),
            deepseek4_q8_wmma_off: value("HIPFIRE_DEEPSEEK4_Q8_WMMA").as_deref() == Ok("0"),
            deepseek4_q8_4w_off: value("HIPFIRE_DEEPSEEK4_Q8_4W").as_deref() == Ok("0"),
            rope_interleaved_legacy: value("HIPFIRE_ROPE_INTERLEAVED_LEGACY").ok().as_deref()
                == Some("1"),
            wo_wmma_variant: value("HIPFIRE_WO_WMMA_VARIANT").ok(),

            // rocBLAS
            rocblas_all_archs: value("HIPFIRE_ROCBLAS_ALL_ARCHS").ok().as_deref() == Some("1"),
            rocblas_off: value("HIPFIRE_ROCBLAS_OFF").ok().as_deref() == Some("1"),
            rocblas_min_batch: parse_usize("HIPFIRE_ROCBLAS_MIN_BATCH"),
            attn_tile_size: parse_usize("HIPFIRE_ATTN_TILE_SIZE"),
            slots_attn_crossover: parse_usize("HIPFIRE_SLOTS_ATTN_CROSSOVER"),
            slots_decode_graph: value("HIPFIRE_SLOTS_DECODE_GRAPH").ok().as_deref() == Some("1"),
            slot_trace: value("HIPFIRE_SLOT_TRACE").ok().as_deref() == Some("1"),

            // Kernels.rs
            lloyd_force_baseline: value("HIPFIRE_LLOYD_FORCE_BASELINE").ok().as_deref()
                == Some("1"),
            rdna2_variant: value("HIPFIRE_RDNA2_VARIANT")
                .ok()
                .and_then(|s| s.parse::<u32>().ok()),

            // Compiler.rs
            hipcc_extra_flags,

            // Interpreter Phase 2a
            force_unfused: value("HIPFIRE_FORCE_UNFUSED")
                .map(|v| v == "1")
                .unwrap_or(false),

            // Speculative decode (DFlash/DDTree). DEFAULT OFF (chain): ddtree
            // tree-verify loses to linear chain on every drafter measured
            // (DeltaNet + non-DeltaNet qwen3-8b/Bielik) because the DFlash
            // drafter emits independent per-position marginals — a tree branch
            // has no joint to exploit. Opt in with the CLI `--ddtree` flag (sets
            // HIPFIRE_DFLASH_TREE=1).
            dflash_tree: value("HIPFIRE_DFLASH_TREE").as_deref() == Ok("1"),
            ddtree_budget: parse_usize("HIPFIRE_DDTREE_BUDGET").filter(|&b| b > 0),
            ddtree_topk: parse_usize("HIPFIRE_DDTREE_TOPK").filter(|&k| k >= 1),
            // D8: ddtree_verify_naive removed; HIPFIRE_DDTREE_VERIFY env is no-op.
            ddtree_tree_la: value("HIPFIRE_DDTREE_TREE_LA").as_deref() != Ok("0"),
            dflash_fast_sample: value("HIPFIRE_DFLASH_FAST_SAMPLE").as_deref() != Ok("0"),
            ddtree_logw_cutoff: value("HIPFIRE_DDTREE_LOGW_CUTOFF")
                .ok()
                .and_then(|s| s.parse::<f32>().ok())
                .filter(|&x| x > 0.0),
            dflash_q8_lmhead_wmma: match value("HIPFIRE_DFLASH_Q8_LMHEAD_WMMA") {
                Ok(v) => {
                    let v = v.trim().to_ascii_lowercase();
                    !(v == "0" || v == "false" || v == "off" || v == "no")
                }
                Err(_) => true,
            },

            // QKV bias fold — default ON, opt out with HIPFIRE_FUSE_QKV_BIAS=0.
            fuse_qkv_bias: parse_bool("HIPFIRE_FUSE_QKV_BIAS").unwrap_or(true),
            fuse_qkv_bias_debug: value("HIPFIRE_FUSE_QKV_BIAS_DEBUG").as_deref() == Ok("1"),
        }
    }

    // ── Methods replacing free functions ─────────────────────────────

    pub fn gemv_dp4a_enabled(&self) -> bool {
        self.gemv_dp4a.unwrap_or(self.gemv_dp4a_default_on)
    }

    /// Resolved ddtree meta-verifier expansion threshold: `−X` for a user-set
    /// positive `X` (stop expanding when cumulative logw drops below it), else
    /// `f32::NEG_INFINITY` (expand to the full budget).
    pub fn ddtree_logw_cutoff_value(&self) -> f32 {
        match self.ddtree_logw_cutoff {
            Some(x) => -x,
            None => f32::NEG_INFINITY,
        }
    }

    pub fn gemv_prefetch_enabled(&self) -> bool {
        self.gemv_prefetch.unwrap_or(self.gemv_prefetch_default_on)
    }

    pub fn gfx942_lds_gemv_enabled(&self) -> bool {
        self.gfx942_lds_gemv
            .unwrap_or(self.gfx942_lds_gemv_default_on)
    }

    pub fn hfq3_mmq_layer_gate_pass(&self) -> bool {
        let lo = self.hfq3_mmq_layer_min;
        let hi = self.hfq3_mmq_layer_max;
        if lo.is_none() && hi.is_none() {
            return true;
        }
        let layer = super::dispatch::MMQ_CURRENT_LAYER.load(std::sync::atomic::Ordering::Relaxed);
        if let Some(lo) = lo {
            if layer < lo {
                return false;
            }
        }
        if let Some(hi) = hi {
            if layer > hi {
                return false;
            }
        }
        true
    }

    pub fn fp16_disabled_for_current_layer(&self) -> bool {
        if self.fp16_disabled {
            return true;
        }
        let lo = self.fp16_layer_min;
        let hi = self.fp16_layer_max;
        if lo.is_none() && hi.is_none() {
            return false;
        }
        let layer = super::dispatch::MMQ_CURRENT_LAYER.load(std::sync::atomic::Ordering::Relaxed);
        let above_min = lo.map(|m| layer >= m).unwrap_or(true);
        let below_max = hi.map(|m| layer <= m).unwrap_or(true);
        above_min && below_max
    }

    pub fn hfq4_mmq_gfx906_y64_enabled(&self) -> bool {
        self.hfq4_mmq_gfx906_y64
    }

    /// Test-only constructor: uses defaults for the given arch.
    /// Provides deterministic FeatureFlags for unit tests regardless of the
    /// developer's env-var configuration.
    #[doc(hidden)]
    pub fn for_test(arch: &str) -> Self {
        let is_gfx906 = arch == "gfx906";

        let gemv_rows_default: u32 = match arch {
            "gfx1100" | "gfx1101" | "gfx1102" => 1,
            "gfx1030" | "gfx1031" => 1,
            "gfx906" | "gfx908" | "gfx940" | "gfx941" | "gfx942" => 1,
            _ => 2,
        };

        Self {
            arch: arch.to_string(),
            gemv_rows: None,
            gemv_dp4a_default_on: is_gfx906,
            gemv_dp4a: None,
            gfx1151_e8_buffer: None,
            gemv_prefetch: None,
            gemv_prefetch_default_on: is_gfx906,
            gfx942_lds_gemv: None,
            gfx942_lds_gemv_default_on: false,
            gemv_rows_default,
            hfq3_dp4a: None,
            hfq3_mmq: None,
            hfq4_mmq_rdna2: None,
            fp8_wmma: false,
            dot2_gemv: false,
            gcn5_wave64_hybrid: None,
            rdna3_hfq4_qkv_wave64: false,
            rdna3_hfq4_qkvza_2wave: false,
            rdna3_hfq4_qkvza_wavepack4: false,
            rdna3_hfq4_qkvza_ldsx8: false,
            rdna3_hfq4_qkvza_reduce_chain: false,
            rdna3_hfq4_qkvza_hoist_x32: false,
            rdna3_hfq4_qkvza_k2048: false,
            rdna3_hfq4_residual_stage_x32: false,
            rdna3_hfq4_residual_k2048: false,
            rdna3_hfq4_sigmoid_tight_grid: false,
            rdna3_hfq4_sigmoid_buffer: false,
            rdna3_hfq4_sigmoid_rows4: false,
            rdna3_hfq4_lm_head_k2048: false,
            rdna3_hfq4_moe_gate_up_k2048: false,
            mmq_override: None,
            mmq_min_batch: None,
            fp16_disabled: false,
            fp16_layer_min: None,
            fp16_layer_max: None,
            wo_mmq: false,
            lm_head_wmma_disabled: false,
            lm_head_overwrite: false,
            mmq_screen: false,
            mmq_screen_threshold: if is_gfx906 { 0.50 } else { 0.10 },
            mmq_diag_quantize_only: false,
            lloyd_mb4: None,
            mq3_mb4: None,
            hfq4g128_mmq: true,
            hfq3_mmq_layer_min: None,
            hfq3_mmq_layer_max: None,
            hfq4_mmq_gfx906_y64: false,
            gate_up_variant: None,
            gate_up_nosync: false,
            hfq4g256_ldsstage_wmma: false,
            qkvza_split_tail: false,
            gfx942_gemv_v2: None,
            gfx942_gemv_v3: false,
            gfx942_rmsnorm_split: matches!(arch, "gfx940" | "gfx941" | "gfx942"),
            rmsnorm_mq_tight_lds: false,
            rdna3_rmsnorm_wavegrid: false,
            rdna3_rmsnorm_split: false,
            rdna3_rmsnorm_vecsum: false,
            rdna3_rmsnorm_sign_lds: false,
            rdna3_rmsnorm_sign_const: false,
            gfx942_mfma_prefill: None,
            moe_grouped_i8: None,
            moe_grouped_i8_k8: false,
            moe_grouped_i8_k4: false,
            moe_grouped_i8_k4_gfx12: false,
            moe_grouped_m2: false,
            moe_grouped_4w: false,
            moe_down_combine_vec4: false,
            moe_hfq6_i8: false,
            moe_hfq6_v2: false,
            moe_grouped_gemm: true,
            moe_paro_i8: None,
            moe_paro_i8_k8: None,
            graph_forward: None,
            graph_ar: true,
            graph_moe: true,
            force_blob_path: false,
            gemm_dump: false,
            deterministic: false,
            mw16: false,
            q8_batched_legacy: false,
            gemma4_q8_fused_prefill: false,
            gemma4_batched_embedding_prefill: false,
            gemma4_ple_batched_prefill: false,
            gemma4_ple_branch_batched_prefill: false,
            gemma4_ple_activation_fused_prefill: false,
            deepseek4_q8_wmma_off: false,
            deepseek4_q8_4w_off: false,
            rope_interleaved_legacy: false,
            wo_wmma_variant: None,
            rocblas_all_archs: false,
            rocblas_off: false,
            rocblas_min_batch: None,
            attn_tile_size: None,
            slots_attn_crossover: None,
            slots_decode_graph: false,
            slot_trace: false,
            lloyd_force_baseline: false,
            rdna2_variant: None,
            hipcc_extra_flags: String::new(),
            force_unfused: false,
            dflash_tree: true,
            ddtree_budget: None,
            ddtree_topk: None,
            // D8: ddtree_verify_naive removed.
            ddtree_tree_la: true,
            dflash_fast_sample: true,
            ddtree_logw_cutoff: None,
            dflash_q8_lmhead_wmma: true,
            fuse_qkv_bias: true,
            fuse_qkv_bias_debug: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hipfire_config::{resolve, ConfigLayer, ConfigSource, NamedLayer, ProcessConfig};

    #[test]
    fn force_unfused_defaults_false_in_test_ctor() {
        let f = FeatureFlags::for_test("gfx1151");
        assert!(!f.force_unfused);
    }

    #[test]
    fn fuse_qkv_bias_defaults_on_in_test_ctor() {
        let f = FeatureFlags::for_test("gfx1151");
        assert!(f.fuse_qkv_bias);
    }

    #[test]
    fn qkvza_split_tail_defaults_false_in_test_ctor() {
        let f = FeatureFlags::for_test("gfx1100");
        assert!(!f.qkvza_split_tail);
        assert!(!f.gemma4_q8_fused_prefill);
        assert!(!f.gemma4_batched_embedding_prefill);
        assert!(!f.gemma4_ple_batched_prefill);
        assert!(!f.gemma4_ple_branch_batched_prefill);
        assert!(!f.gemma4_ple_activation_fused_prefill);
    }

    #[test]
    fn gemma4_prefill_auto_defaults_on_validated_arches() {
        let resolved = resolve([]).unwrap();
        let process = ProcessConfig::from_resolved(&resolved).unwrap();

        let gfx1100 = FeatureFlags::from_process_config("gfx1100", &process);
        assert!(gfx1100.gemma4_batched_embedding_prefill);
        assert!(!gfx1100.gemma4_ple_branch_batched_prefill);
        assert!(gfx1100.gemma4_ple_activation_fused_prefill);
        assert!(!gfx1100.gemma4_q8_fused_prefill);
        assert!(!gfx1100.gemma4_ple_batched_prefill);

        let gfx1201 = FeatureFlags::from_process_config("gfx1201", &process);
        assert!(gfx1201.gemma4_batched_embedding_prefill);
        assert!(!gfx1201.gemma4_ple_branch_batched_prefill);
        assert!(gfx1201.gemma4_ple_activation_fused_prefill);
        assert!(!gfx1201.gemma4_q8_fused_prefill);
        assert!(!gfx1201.gemma4_ple_batched_prefill);

        for arch in ["gfx1101", "gfx1102", "gfx1151", "gfx1200"] {
            let flags = FeatureFlags::from_process_config(arch, &process);
            assert!(!flags.gemma4_batched_embedding_prefill, "arch={arch}");
            assert!(!flags.gemma4_ple_branch_batched_prefill, "arch={arch}");
            assert!(!flags.gemma4_ple_activation_fused_prefill, "arch={arch}");
        }
    }

    #[test]
    fn gemma4_prefill_auto_accepts_explicit_opt_out_on_validated_arches() {
        let mut layer = ConfigLayer::default();
        layer
            .set_cli("kernel.gemma4_batched_embedding_prefill", "false")
            .unwrap();
        layer
            .set_cli("kernel.gemma4_ple_branch_batched_prefill", "false")
            .unwrap();
        layer
            .set_cli("kernel.gemma4_ple_activation_fused_prefill", "false")
            .unwrap();
        let resolved = resolve([NamedLayer {
            source: ConfigSource::GlobalUser {
                path: "config.toml".into(),
            },
            layer,
        }])
        .unwrap();
        let process = ProcessConfig::from_resolved(&resolved).unwrap();
        for arch in ["gfx1100", "gfx1201"] {
            let flags = FeatureFlags::from_process_config(arch, &process);
            assert!(!flags.gemma4_batched_embedding_prefill, "arch={arch}");
            assert!(!flags.gemma4_ple_branch_batched_prefill, "arch={arch}");
            assert!(!flags.gemma4_ple_activation_fused_prefill, "arch={arch}");
        }
    }

    #[test]
    fn process_config_preserves_explicit_and_arch_default_flags() {
        let mut layer = ConfigLayer::default();
        layer.set_cli("kernel.qkvza_split_tail", "true").unwrap();
        layer
            .set_cli("kernel.gemma4_q8_fused_prefill", "true")
            .unwrap();
        layer
            .set_cli("kernel.gemma4_batched_embedding_prefill", "true")
            .unwrap();
        layer
            .set_cli("kernel.gemma4_ple_batched_prefill", "true")
            .unwrap();
        layer
            .set_cli("kernel.gemma4_ple_branch_batched_prefill", "true")
            .unwrap();
        layer
            .set_cli("kernel.gemma4_ple_activation_fused_prefill", "true")
            .unwrap();
        layer.set_cli("diagnostic.kernel.gemv_rows", "4").unwrap();
        let resolved = resolve([NamedLayer {
            source: ConfigSource::GlobalUser {
                path: "config.toml".into(),
            },
            layer,
        }])
        .unwrap();
        let process = ProcessConfig::from_resolved(&resolved).unwrap();
        let flags = FeatureFlags::from_process_config("gfx1100", &process);

        assert!(flags.qkvza_split_tail);
        assert!(flags.gemma4_q8_fused_prefill);
        assert!(flags.gemma4_batched_embedding_prefill);
        assert!(flags.gemma4_ple_batched_prefill);
        assert!(flags.gemma4_ple_branch_batched_prefill);
        assert!(flags.gemma4_ple_activation_fused_prefill);
        assert_eq!(flags.gemv_rows, Some(4));
        assert!(flags.rdna3_hfq4_qkvza_k2048);
        assert!(flags.rdna3_hfq4_residual_stage_x32);
    }
}
