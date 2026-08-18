// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

use std::sync::atomic;
use std::sync::{LazyLock, OnceLock};

fn flag_one(name: &'static str) -> bool {
    hipfire_config::developer_var(name).ok().as_deref() == Some("1")
}

/// `HIPFIRE_DEEPSEEK4_MOE` — default ON. Opt out with "0" for diagnostic
/// shared-only-FFN runs. Without routed-expert dispatch the model is
/// architecturally broken (DeepSeek V4 is MoE), so leaving this off was only
/// useful during initial bring-up before the forward path consumed the
/// expert blobs.
pub(crate) fn moe_on() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| {
        hipfire_config::developer_var("HIPFIRE_DEEPSEEK4_MOE")
            .ok()
            .as_deref()
            != Some("0")
    })
}
/// `HIPFIRE_DEEPSEEK4_SKIP_FFN` — diagnostic: zero ffn_out to isolate attn growth.
pub(crate) fn skip_ffn() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| flag_one("HIPFIRE_DEEPSEEK4_SKIP_FFN"))
}
/// `HIPFIRE_DEEPSEEK4_ATTN` — when "pos0", attn_stub uses the diagnostic
/// pos-0 attention path instead of SWA. Default false (i.e. use SWA).
pub(crate) fn attn_pos0() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| {
        hipfire_config::developer_var("HIPFIRE_DEEPSEEK4_ATTN")
            .ok()
            .as_deref()
            == Some("pos0")
    })
}
/// `HIPFIRE_DEEPSEEK4_MTP_HEAD_HC` — default ON since 2026-05-21: route
/// the MTP output (step 8 of mtp_forward) through head-HC mix using
/// `mtp.0.hc_head_fn / hc_head_base / hc_head_scale`. Mirrors the
/// main model's final_norm_and_head head-HC reduction. Without
/// this, MTP step 8 reads only stream 0 — discarding 75% of HC
/// signal at the head boundary (same architectural pattern as the
/// input-side HC fix shipped in 82224ad). Opt out with =0 for
/// debugging or pre-fix-compat builds.
pub(crate) fn mtp_head_hc_on() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| {
        hipfire_config::developer_var("HIPFIRE_DEEPSEEK4_MTP_HEAD_HC")
            .ok()
            .as_deref()
            != Some("0")
    })
}
/// `HIPFIRE_DEEPSEEK4_E8_WO_GROUPED` — collapse the eight gfx1151
/// E8-SoA O-LoRA GEMVs into one block-diagonal launch.
pub(crate) fn e8_wo_grouped_on(arch: &str, mq2r: bool) -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    arch == "gfx1151"
        && (mq2r
            || *V.get_or_init(|| {
                hipfire_config::developer_var("HIPFIRE_DEEPSEEK4_E8_WO_GROUPED")
                    .ok()
                    .as_deref()
                    != Some("0")
            }))
}
/// Exact-gfx1201 MQ2R route for collapsing the eight O-LoRA E8 GEMVs.
/// This is a shipped architecture default, not an environment experiment.
pub(crate) fn gfx1201_e8_wo_grouped_on(arch: &str, mq2r: bool) -> bool {
    arch == "gfx1201" && mq2r
}
/// Exact-gfx1201 MQ2R admission for the DeepSeek-only low-LDS fused
/// RMSNorm + FWHT path. Adjacent models still call the generic method.
pub(crate) fn gfx1201_rmsnorm_rotate_nox_on(arch: &str, mq2r: bool) -> bool {
    arch == "gfx1201" && mq2r
}
/// Candidate-only exact-gfx1201 head-strided indexer-Q RoPE. Default OFF
/// until the composed product route clears its performance and decoded-byte
/// gates; the promoted form becomes an architecture default.
pub(crate) fn gfx1201_indexer_rope_heads_on(arch: &str, mq2r: bool) -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    arch == "gfx1201"
        && mq2r
        && *V.get_or_init(|| flag_one("HIPFIRE_DEEPSEEK4_GFX1201_INDEXER_ROPE_HEADS"))
}
/// `HIPFIRE_DEEPSEEK4_GFX942_COMPRESSOR_GATE` — host-gate non-commit
/// compressor commit-stage launches on exact `gfx942` MQ2R ordinary HIP.
/// Default OFF; set `=1` to enable. Forced off during hipGraph capture
/// so the captured graph retains full sentinel-driven commit nodes
/// (see `compressor_forward_impl` decode path). Unvalidated for
/// state/output exactness — must stay opt-in until measured.
pub(crate) fn gfx942_compressor_gate_on(arch: &str, mq2r: bool) -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    arch == "gfx942"
        && mq2r
        && *V.get_or_init(|| flag_one("HIPFIRE_DEEPSEEK4_GFX942_COMPRESSOR_GATE"))
}
/// `HIPFIRE_DEEPSEEK4_GFX942_E8_WO_GROUPED` — wire the gfx942 grouped E8
/// O-LoRA kernel (`gemv_mfp4g32_e8_soa_grouped_gfx942`) for `wo_a`
/// instead of the 8-way `gemv_auto` loop. Default ON for gfx942 route v1
/// after the production-shape raw-bit channel passed; set `=0` for the
/// serial emergency fallback.
/// Separate from `HIPFIRE_DEEPSEEK4_E8_WO_GROUPED` (gfx1151).
pub(crate) fn gfx942_e8_wo_grouped_on(arch: &str, gfx942_route_v1: bool) -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    arch == "gfx942"
        && gfx942_route_v1
        && *V.get_or_init(|| {
            hipfire_config::developer_var("HIPFIRE_DEEPSEEK4_GFX942_E8_WO_GROUPED")
                .ok()
                .as_deref()
                != Some("0")
        })
}
/// `HIPFIRE_DEEPSEEK4_GFX942_HC_FINALIZE_FUSED` — replace the three
/// scalar mHC finalization launches (alpha, sigmoid/scale, Sinkhorn) with
/// their source-order-equivalent single-wave finalizer. Kept opt-in until
/// the gfx942 raw-state channel and product screen pass.
pub(crate) fn gfx942_hc_finalize_fused_on(arch: &str, gfx942_route_v1: bool) -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    arch == "gfx942"
        && gfx942_route_v1
        && *V.get_or_init(|| flag_one("HIPFIRE_DEEPSEEK4_GFX942_HC_FINALIZE_FUSED"))
}
/// `HIPFIRE_DEEPSEEK4_GFX942_INDEXER_TOPK_PARALLEL` — F1: admit the
/// filtered block-parallel indexer top-K on exact `gfx942` MQ2R.
/// Default ON for the gfx942-v1 route after its raw-i32 channel passed;
/// set `=0` for the serial emergency fallback. Independent of L1/L3.
/// The actual dispatch lives behind the model-owned `Mq2rBackend` and the
/// exact-device `Gfx942Device`; this helper is only the emergency kill
/// switch and operator-log surface.
pub(crate) fn gfx942_indexer_topk_parallel_on(arch: &str, mq2r: bool) -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    arch == "gfx942"
        && mq2r
        && *V.get_or_init(|| {
            hipfire_config::developer_var("HIPFIRE_DEEPSEEK4_GFX942_INDEXER_TOPK_PARALLEL")
                .ok()
                .as_deref()
                != Some("0")
        })
}
/// `HIPFIRE_DEEPSEEK4_GFX942_INDEXER_TOPK_BOUNDED` — F2: select the
/// bounded tile-merge bitonic indexer top-K on exact `gfx942` MQ2R
/// instead of F1's rank-count kernel. F1's kernel is O(N^2) on a single
/// workgroup (`n_idx_heads == 1`), which is what walled long context.
/// Requires F1 to be on — it selects which kernel that path dispatches.
/// Default ON for the gfx942-v1 route after both promotion conditions
/// passed; set `=0` for the emergency fallback to the rank-count kernel.
///
/// Exactness (raw-i32 channel, `test_indexer_top_k_buf`, MI300X): 10/10
/// PASS, every arm byte-identical in ORDER, finite cases also identical
/// to the host oracle. Covers the `n_compressed <= max_k` identity path,
/// the N=513 boundary, all three non-finite eligibility cases
/// (-inf pool / NaN / -FLT_MAX), and N = 8192 / 32768 / 262144.
///
/// Perf (same binary, only this env differing, 3 fresh processes/arm,
/// interleaved A,B,B,A,A,B, 18/18 pass): cap 2048 25.046 -> 34.533 tok/s
/// (1.38x), cap 8192 3.910 -> 25.957 (6.64x), cap 32768 0.274 -> 13.051
/// (47.60x). Isolated kernel: 22.3x at N=8192, 88.3x at N=32768, 715x at
/// N=262144. Output is bit-identical at every N and the identity path
/// makes it a literal no-op below N = max_k, so short-context product
/// shapes are unaffected by construction.
///
/// Remaining limit: the grid is still one workgroup, so the tile loop is
/// serial on one CU of 304. At N=262144 the bounded kernel is still ~75%
/// of projected decode (~324 ms of ~430 ms, ~2.3 tok/s). A multi-block
/// two-stage top-K is the next lever; the O(N) indexer scoring under it
/// is only ~0.31 ms per 1000 rows.
pub(crate) fn gfx942_indexer_topk_bounded_on(arch: &str, mq2r: bool) -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    arch == "gfx942"
        && mq2r
        && *V.get_or_init(|| {
            hipfire_config::developer_var("HIPFIRE_DEEPSEEK4_GFX942_INDEXER_TOPK_BOUNDED")
                .ok()
                .as_deref()
                != Some("0")
        })
}
/// `HIPFIRE_DEEPSEEK4_GFX942_INDEXER_TOPK_TWOSTAGE` — F3: select the
/// multi-block two-stage merge-tree indexer top-K on exact `gfx942` MQ2R
/// instead of F2's single-workgroup bounded kernel. F2 collapsed the
/// O(N^2) rank count but its grid is still `[n_idx_heads,1,1]` with
/// `n_idx_heads == 1`, so the tile loop is serial on one CU; F3 replaces
/// it with a chunk-sort + parallel merge tree whose launch count and
/// grids derive only from `max_compressed` (graph-safe). Default ON for
/// the gfx942-v1 route after both promotion conditions passed; set `=0`
/// to fall back to F2. Selected only when
/// the active compressed-cache bucket is at least
/// `indexer_topk_two_stage_min()`.
///
/// Exactness (raw-i32 channel, `test_indexer_top_k_buf`, MI300X): 13/13
/// PASS, all four arms (SERIAL / PARALLEL / BOUNDED / TWOSTAGE)
/// byte-identical in ORDER, finite cases also identical to the host
/// oracle. Covers the identity path, the N=513 boundary, all three
/// non-finite eligibility cases, and N = 8192 / 32768 / 262144. The same
/// harness passes on gfx1151.
///
/// Perf vs F2 (same binary, only this env differing, 3 fresh processes
/// per arm, interleaved A,B,B,A,A,B, 18/18 pass, spread <=0.44% per arm):
///   cap  2048  34.481 -> 36.327 tok/s  (1.054x)
///   cap  8192  25.941 -> 33.367 tok/s  (1.286x)
///   cap 32768  13.043 -> 25.961 tok/s  (1.990x)
/// Cumulative against the pre-F2 default on the same box and model:
/// 25.046 -> 36.327 (1.5x), 3.910 -> 33.367 (8.5x), 0.274 -> 25.961
/// (94.7x). Isolated kernel: 128x vs F2 at N=262144 on gfx942, 44.7x on
/// gfx1151.
///
/// Caveat on the numbers: every one was taken with graph capture OFF and
/// no Redline retained replay. The sequence is launch-bound (7.3-13.5 us
/// per launch, near-flat in N) and F3 adds 231 launches per token across
/// 21 ratio-4 layers, so a retained-replay path that amortises dispatch
/// cost should make F3 look BETTER than these figures, not worse. Do not
/// invert that reasoning from a graphs-off benchmark.
pub(crate) fn gfx942_indexer_topk_two_stage_on(arch: &str, mq2r: bool) -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    arch == "gfx942"
        && mq2r
        && *V.get_or_init(|| {
            hipfire_config::developer_var("HIPFIRE_DEEPSEEK4_GFX942_INDEXER_TOPK_TWOSTAGE")
                .ok()
                .as_deref()
                != Some("0")
        })
}
/// `HIPFIRE_DEEPSEEK4_GFX1151_INDEXER_TOPK_TWOSTAGE` — G1: the gfx1151
/// twin of F3. gfx1151's legacy indexer top-K is the SAME O(N^2)
/// rank-count on one workgroup that F2 removed from gfx942. The two-stage
/// route is now the DeepSeek4 gfx1151 product default after exactness
/// through the 1M scale and the long-context model campaign; set `=0`
/// only as an emergency diagnostic fallback. Indexer scores and ranks are
/// F32 state, so selection depends on the model architecture and device,
/// not whether dense weights are MQ2-Lloyd or the MQ2R E8 tier. This is a
/// SEPARATE lever
/// from F3, not one shared `*_TWOSTAGE` flag: strict arch gating (no
/// gfx942 lever ever bleeding into gfx1151 or vice versa) is enforced by
/// the arch-gating unit tests, and the two arches promote independently.
pub(crate) fn gfx1151_indexer_topk_two_stage_on(arch: &str) -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    arch == "gfx1151"
        && *V.get_or_init(|| {
            hipfire_config::developer_var("HIPFIRE_DEEPSEEK4_GFX1151_INDEXER_TOPK_TWOSTAGE")
                .ok()
                .as_deref()
                != Some("0")
        })
}
/// gfx1100 heterogeneous twin of the portable F3/G1 two-stage indexer.
///
/// The kernel body is already shared across wave32 gfx1151 and wave64
/// gfx942, while `Gpu::ensure_kernel` still emits an exact-device code
/// object for this `Gpu`.  Keep admission separate from G1 so enabling the
/// dense half of the heterogeneous route cannot change gfx1151's certified
/// tape or any Qwen-owned path.  Default ON; `=0` is an emergency
/// diagnostic fallback to the legacy O(N^2) rank-count kernel.
pub(crate) fn gfx1100_indexer_topk_two_stage_on(arch: &str) -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    arch == "gfx1100"
        && *V.get_or_init(|| {
            hipfire_config::developer_var("HIPFIRE_DEEPSEEK4_GFX1100_INDEXER_TOPK_TWOSTAGE")
                .ok()
                .as_deref()
                != Some("0")
        })
}
/// gfx1201 expert-parallel admission for the portable two-stage indexer.
///
/// The initial 2,048-row capacity bucket uses the faster one-launch bounded
/// network; after capacity growth this admits the portable merge tree. Both
/// replace the legacy single-workgroup O(N^2) rank count and compile into
/// exact-gfx1201 code objects. Keep admission separate from every other
/// architecture.
pub(crate) fn gfx1201_indexer_topk_two_stage_on(arch: &str) -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    arch == "gfx1201"
        && *V.get_or_init(|| {
            hipfire_config::developer_var("HIPFIRE_DEEPSEEK4_GFX1201_INDEXER_TOPK_TWOSTAGE")
                .ok()
                .as_deref()
                != Some("0")
        })
}
/// `HIPFIRE_DEEPSEEK4_INDEXER_TOPK_TWOSTAGE_MIN` — minimum
/// active compressed-cache row count at which the two-stage path may be selected.
/// Default 1024, from the measured standalone-probe crossover against the
/// F2 bounded kernel (ms per full sequence): the ONLY losing regime is
/// cap 512, where the bounded kernel takes the `n_compressed <= max_k`
/// identity fast path and is essentially free (gfx1151: 0.0024 bounded vs
/// 0.0146 two-stage, a 6x loss; gfx942: 0.0030 vs 0.0271, 9x). At 1024
/// two-stage already wins (gfx1151 0.0255 -> 0.0198, 1.29x), and the win
/// grows monotonically: gfx1151 2.64x/5.28x/16.9x/51.9x and gfx942
/// 2.70x/8.15x/25.7x/144.9x at N = 2048/8192/32768/262144.
pub(crate) fn indexer_topk_two_stage_min() -> usize {
    static V: OnceLock<usize> = OnceLock::new();
    *V.get_or_init(|| {
        hipfire_config::developer_var("HIPFIRE_DEEPSEEK4_INDEXER_TOPK_TWOSTAGE_MIN")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1024)
    })
}
/// `HIPFIRE_DEEPSEEK4_GFX942_FFN_OVERLAP` — evaluate the dense shared
/// expert and routed MQ2 experts concurrently after their common FFN
/// normalization. Eligibility comes from the model-owned exact-gfx942
/// backend proof. Default OFF pending the product and boundary gates.
pub(crate) fn gfx942_ffn_overlap_on(exact_gfx942_backend: bool) -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    exact_gfx942_backend && *V.get_or_init(|| flag_one("HIPFIRE_DEEPSEEK4_GFX942_FFN_OVERLAP"))
}
/// One-shot daemon-log line for A2 gfx942 levers so ABBA runs cannot
/// misattribute measurements.
pub(crate) fn log_gfx942_a2_levers(arch: &str, gfx942_route_v1: bool) {
    if arch != "gfx942" {
        return;
    }
    static LOGGED: OnceLock<()> = OnceLock::new();
    let _ = LOGGED.get_or_init(|| {
        let l1 = gfx942_compressor_gate_on(arch, gfx942_route_v1);
        let l3 = gfx942_e8_wo_grouped_on(arch, gfx942_route_v1);
        let l2 = gfx942_hc_finalize_fused_on(arch, gfx942_route_v1);
        let f1 = gfx942_indexer_topk_parallel_on(arch, gfx942_route_v1);
        let f2 = gfx942_indexer_topk_bounded_on(arch, gfx942_route_v1);
        let f3 = gfx942_indexer_topk_two_stage_on(arch, gfx942_route_v1);
        let ffn_overlap = gfx942_ffn_overlap_on(gfx942_route_v1 && arch == "gfx942");
        eprintln!(
            "deepseek4 gfx942 A2 levers: \
             L1 compressor_gate={} (HIPFIRE_DEEPSEEK4_GFX942_COMPRESSOR_GATE default OFF, =1 enables) ; \
             L3 e8_wo_grouped={} (gfx942-v1 default ON, HIPFIRE_DEEPSEEK4_GFX942_E8_WO_GROUPED=0 disables) ; \
             L2 hc_finalize_fused={} (default OFF, HIPFIRE_DEEPSEEK4_GFX942_HC_FINALIZE_FUSED=1 enables) ; \
             F1 indexer_topk_parallel={} (gfx942-v1 default ON, HIPFIRE_DEEPSEEK4_GFX942_INDEXER_TOPK_PARALLEL=0 disables) ; \
             F2 indexer_topk_bounded={} (gfx942-v1 default ON, HIPFIRE_DEEPSEEK4_GFX942_INDEXER_TOPK_BOUNDED=0 disables) ; \
             F3 indexer_topk_two_stage={} (gfx942-v1 default ON, HIPFIRE_DEEPSEEK4_GFX942_INDEXER_TOPK_TWOSTAGE=0 disables) ; \
             C1 ffn_overlap={} (default OFF, HIPFIRE_DEEPSEEK4_GFX942_FFN_OVERLAP=1 enables)",
            if l1 { "ON" } else { "OFF" },
            if l3 { "ON" } else { "OFF" },
            if l2 { "ON" } else { "OFF" },
            if f1 { "ON" } else { "OFF" },
            if f2 { "ON" } else { "OFF" },
            if f3 { "ON" } else { "OFF" },
            if ffn_overlap { "ON" } else { "OFF" },
        );
    });
}

/// Resolve the routed-MoE scale for one call.
///
/// `HIPFIRE_DEEPSEEK4_ROUTE_SCALE` is an explicit process-level override and
/// is the only value cached. The silent default is always
/// `cfg.routed_scaling_factor` (checkpoint value, typically 1.5) — never a
/// hardcoded constant. Caching the combined scale would be wrong the
/// moment two models with different checkpoint scales are served in one
/// process.
pub(crate) fn route_scale(cfg_routed_scaling_factor: f32, mq2r: bool) -> f32 {
    static ENV_OVERRIDE: LazyLock<Option<f32>> = LazyLock::new(|| {
        hipfire_config::developer_var("HIPFIRE_DEEPSEEK4_ROUTE_SCALE")
            .ok()
            .and_then(|s| s.parse().ok())
    });
    let env_override = *ENV_OVERRIDE;
    let effective = resolve_route_scale(cfg_routed_scaling_factor, env_override, mq2r);
    if (effective - cfg_routed_scaling_factor).abs() > f32::EPSILON {
        static LOGGED: OnceLock<()> = OnceLock::new();
        let _ = LOGGED.get_or_init(|| {
            // Never silent. A route scale that differs from the header is
            // a compensation, and the last one to go unannounced survived
            // two months.
            let src = if env_override.is_some() {
                "HIPFIRE_DEEPSEEK4_ROUTE_SCALE"
            } else {
                ".mq2r artifact default (measured optimum)"
            };
            eprintln!(
                "deepseek4: route_scale={effective} from {src} overrides \
                 cfg.routed_scaling_factor={cfg_routed_scaling_factor}"
            );
        });
    }
    effective
}

/// Pure combiner used by [`route_scale`] and unit tests.
/// `env_override = None` → checkpoint scale; `Some(v)` → explicit override.
///
/// # Do not "fix" this back to a hardcoded 2.2
///
/// Until 2026-08-02 the default here was a hardcoded `2.2`, introduced in
/// the initial DS4 bring-up (`b263fb3cc`, 2026-05-23) and never derived
/// from any checkpoint. Every DeepSeek V4 artifact declares
/// `routed_scaling_factor: 1.5` — the 0731 safetensors config, the
/// 2026-05-27 `deepseek-v4-flash.mq2lloyd` HFQ header, and the 2026-07-24
/// `deepseek-v4-flash.mq2r` HFQ header alike — so `2.2` never matched a
/// model. `cfg.routed_scaling_factor` was parsed correctly the whole time
/// and simply never read here.
///
/// Measured on `mi300x`/gfx942 against
/// `deepseek-v4-flash.mq2r`, wikitext2 slice md5
/// `83b0205a304bf4e52172ecdb05f2e895`, fresh process per run:
///
/// | scale | PPL @ctx256 | PPL @ctx512 |
/// |---|---:|---:|
/// | 2.2 (old silent default) | 6.0804 | 6.5453 |
/// | 1.5 (checkpoint value)   | 7.0131 | 8.1091 |
///
/// So the *wrong* value measurably wins on that quantized artifact. The
/// reading is that MQ2 expert quantization loses routed-expert magnitude
/// and an inflated route scale has been silently compensating: a
/// ~1.47x factor applied to the routed branch only. That is an artifact
/// property, not a model property, and conflating the two in one constant
/// is what made this invisible for two months.
///
/// The contract value stays the silent default for every artifact whose
/// header we trust. The one exception is keyed to the **artifact**, never
/// to the architecture:
///
/// # `.mq2r` ships at 2.0
///
/// A route_scale sweep on the 0731 MQ2R (wikitext2 slice md5
/// `83b0205a304bf4e52172ecdb05f2e895`, ctx256, fresh process per run) puts
/// the minimum at **2.0**, sharply:
///
/// | scale | 1.2 | 1.5 | 1.8 | 2.0 | 2.2 | 2.4 | 2.6 |
/// |---|---:|---:|---:|---:|---:|---:|---:|
/// | PPL | 15.59 | 9.13 | 6.63 | **6.03** | 6.84 | 7.30 | 7.90 |
///
/// so serving MQ2R on its header's 1.5 costs ~51% PPL. The optimum is
/// **per build**, not per architecture: MQ2R and MQ2-Lloyd differ in tier
/// layout (shared expert and `ffn.gate.weight` are qt=3 Q8_0 in Lloyd
/// against qt=35 MFP4G32E8SOA in R), so the routed:shared gain each build
/// wants is different. The table in the git history recording 2.2 as the
/// winner is a Lloyd-era measurement and must not be applied to MQ2R.
///
/// This is gated on `cfg.mq2r`, which comes from the exact MQ2R-family
/// artifact identity and is never inferred from tensor dtype. MQ2RXT keeps
/// the same routed MQ2-Lloyd payload and initially inherits this measured
/// compensation; its quality certification must re-check the optimum.
/// MQ2-Lloyd and every other DeepSeek V4 artifact keep their header value.
///
/// It remains a compensation, not a model property: MQ2 expert
/// quantization loses routed-expert magnitude and an inflated route scale
/// buys it back on the routed branch only. The real fix is at
/// quantization time, where recovering that magnitude should make the
/// header value both correct *and* better. Re-measure after any re-quant
/// and delete this branch when it stops winning.
///
/// # The checkpoint value is NOT usable on this path
///
/// `672373ce1` (2026-08-02) changed the default from the long-standing
/// hipfire value to `cfg.routed_scaling_factor` on the reasoning that
/// `2.2` "never matched a model". That reasoning was right about
/// provenance and wrong about behaviour: it silently moved every non-`mq2r`
/// DeepSeek V4 artifact from 2.2 to 1.5, and 1.5 is measurably bad here —
/// 16.31 PPL at ctx2048 against 10.81 at 2.0, a 51% penalty. This restores
/// the calibrated value and keeps the checkpoint out of the decision.
///
/// **Both** builds want well above 1.5: MQ2-Lloyd was calibrated at 2.2 in
/// the initial bring-up (`b263fb3cc`, 2026-05-23) and ran there for two
/// months, and the 0731 MQ2R sweep lands at 2.0. That is the signature of a
/// **systematic** shortfall in this crate's MoE routed branch, not a
/// per-artifact quantization property — the reference applies 1.5 and
/// scores PPL 4.693, so 1.5 is correct for the *model* and wrong for *us*.
/// The per-build numbers are fine tuning on top of that gap.
///
/// So this is a **known-defect compensation with a measured value**, and it
/// stays until the routed-branch shortfall is found. Do not "restore" the
/// checkpoint value again on provenance grounds alone; that experiment has
/// now been run and it costs ~51%.
///
/// Precedence: `HIPFIRE_DEEPSEEK4_ROUTE_SCALE` (explicit, logged) beats the
/// per-build default, which beats the checkpoint value. Every deviation
/// from the header is logged — a silent one survived two months.
///
/// The parent-calibration path (parent) must always use the
/// checkpoint value — inheriting a serving compensation into a reference
/// forward is exactly the failure this whole effort exists to prevent.
#[inline]
pub(crate) fn resolve_route_scale(
    cfg_routed_scaling_factor: f32,
    env_override: Option<f32>,
    mq2r: bool,
) -> f32 {
    let _ = cfg_routed_scaling_factor;
    /// Measured optimum for the 0731 `.mq2r` build at ctx2048 (table
    /// above), the most recent and best-controlled sweep: fresh process per
    /// run, single methodology, `effective_route_scale` logged per run.
    ///
    /// An older ctx256 sweep put the minimum at 2.0, but a second ctx256
    /// table taken in the same period disagrees with it, so that data is not
    /// trustworthy enough to override a clean long-context measurement.
    /// Serving context is also closer to 2048 than to 256.
    ///
    /// Caveat worth knowing before re-tuning: 1.8 beats 2.0 by only 1.1% at
    /// ctx2048 (10.688 against 10.810), which is a narrow margin. Both are
    /// far from 1.5 (16.306). If you need to defend this number, run
    /// repeats rather than a single fresh-process pair.
    const MQ2R_ROUTE_SCALE: f32 = 1.8;
    /// Calibrated in the initial DS4 bring-up and served for two months.
    /// Never swept, restored because it is known-good and the alternative
    /// (1.5) is known-bad. Sweep it and replace with a measured value.
    const DS4_ROUTE_SCALE: f32 = 2.2;
    env_override.unwrap_or(if mq2r {
        MQ2R_ROUTE_SCALE
    } else {
        DS4_ROUTE_SCALE
    })
}

/// `HIPFIRE_DEEPSEEK4_E8_U4` — use the four-group-unrolled E8-SoA
/// decode schedule for DeepSeek V4 dense projections on gfx1151. This is
/// deliberately architecture-owned: the same schedule regresses smaller
/// A3B projection shapes. MQ2R pins it ON; other DeepSeek artifacts may set
/// `=0` for the original one-row schedule.
pub(crate) fn e8_u4_on(arch: &str, mq2r: bool) -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    arch == "gfx1151"
        && (mq2r
            || *V.get_or_init(|| {
                hipfire_config::developer_var("HIPFIRE_DEEPSEEK4_E8_U4")
                    .ok()
                    .as_deref()
                    != Some("0")
            }))
}
/// `HIPFIRE_DEEPSEEK4_FUSED_ROUTE_ACT` — fold the MoE routing activation
/// `sqrt(softplus(.))` into the store of the gate GEMV instead of running
/// the standalone `sqrt_softplus_f32` launch.
///
/// In the retained gfx1151 ds4 route the gate GEMV (M = 256 experts) has a
/// NEGATIVE marginal — already fully hidden — and so does the `topk` that
/// consumes it, while the activation between them costs ~1.43 ms/token on a
/// 1x1x1 grid that is almost entirely launch and drain. Fusing into the
/// producer moves exposed work into a kernel that already runs for free.
/// Default OFF pending its own shadow and product evidence.
pub(crate) fn moe_route_fused_activation() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| {
        hipfire_config::developer_var("HIPFIRE_DEEPSEEK4_FUSED_ROUTE_ACT")
            .ok()
            .as_deref()
            == Some("1")
    })
}
/// `HIPFIRE_DEEPSEEK4_E8_PREFILL_B2` — reuse decoded E8 fragments across
/// two 16-token WMMA tiles on gfx1151. Set `=0` for the B1 admission
/// baseline.
pub(crate) fn e8_prefill_b2_on(arch: &str, mq2r: bool) -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    arch == "gfx1151"
        && (mq2r
            || *V.get_or_init(|| {
                hipfire_config::developer_var("HIPFIRE_DEEPSEEK4_E8_PREFILL_B2")
                    .ok()
                    .as_deref()
                    != Some("0")
            }))
}
/// `HIPFIRE_DEEPSEEK4_E8_PREFILL_B4` — candidate four-tile reuse schedule.
/// Set `=0` to retain the admitted B2 path for same-binary A/B.
pub(crate) fn e8_prefill_b4_on(arch: &str, mq2r: bool) -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    arch == "gfx1151"
        && (mq2r
            || *V.get_or_init(|| {
                hipfire_config::developer_var("HIPFIRE_DEEPSEEK4_E8_PREFILL_B4")
                    .ok()
                    .as_deref()
                    != Some("0")
            }))
}
/// `HIPFIRE_DEEPSEEK4_E8_BATCHED_GEMV` — largest batch size routed through
/// the batched E8 decode GEMV instead of the WMMA token tile. `0` (the
/// default) keeps every batch size on WMMA.
///
/// The WMMA prefill GEMM tiles the token axis at 16 and launches only M/16
/// waves, so a B-token speculative verify pays a full 16-token tile at
/// 1/16th the occupancy. The batched GEMV keeps the decode GEMV's M waves
/// and single weight-row read, and — unlike WMMA, which converts X to f16
/// via `ensure_fp16_x` — consumes f32 activations, so B=1 is bit-identical
/// to the AR decode GEMV.
///
/// Cached in an atomic rather than a `OnceLock` so a bench can A/B both
/// arms in one process via [`super::set_e8_batched_gemv_max_batch`]. The
/// trunk is ~80 GB on gfx1151; a per-arm reload would dominate the
/// measurement and put the two arms in different thermal states.
pub(crate) fn e8_batched_gemv_max_batch() -> usize {
    let cur = E8_BATCHED_GEMV_MAX.load(atomic::Ordering::Relaxed);
    if cur != E8_BATCHED_GEMV_UNSET {
        return cur;
    }
    // Default 8 (was 0 = disabled). `e8_batched_gemv_applies` already
    // requires `arch == "gfx1151"`, so this only changes gfx1151.
    //
    // At speculative-verify batch the dense E8 WMMA tile (`tiles=1`, since
    // `e8_prefill_batch_tiles` only tiles above batch 16/32) wastes 15/16 of
    // each 16-wide tile and runs at ~22-53 GiB/s, while the batched GEMV
    // reads the weights once and hits 56-193 GiB/s. Measured across the real
    // DS4 dense shape mix, DRAM-resident, gfx1151
    // (`bench_e8_verify_tiles`): the GEMV wins at EVERY shape for every
    // B in 1..=8 — 1.07x to 6.92x. The b2/b4 tiles are never better than b1.
    //
    // End-to-end on the production daemon (0731 MQ2R, `hipfire run --spec
    // dspark`), reproducible across a 8/0/8 ordering:
    //     =0   13.2 tok/s   verify_block 144.72 ms/window
    //     =8   25.8 tok/s   verify_block  74.90 ms/window
    // i.e. 1.93x on verify and 1.95x end-to-end. AR is unaffected (27.9
    // both ways) because AR decode is B=1 through a different arm.
    //
    // 8 is the cap because `E8_BATCHED_GEMV_BATCHES` is [1..=8, 16] and the
    // GEMV's margin decays with B (it re-reads x per row): at B=8 it is only
    // 1.07-1.74x, and at B=16 the WMMA tile wins. Verify runs B<=6.
    let parsed = hipfire_config::developer_var("HIPFIRE_DEEPSEEK4_E8_BATCHED_GEMV")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(8);
    E8_BATCHED_GEMV_MAX.store(parsed, atomic::Ordering::Relaxed);
    parsed
}
pub(crate) const E8_BATCHED_GEMV_UNSET: usize = usize::MAX;
pub(crate) static E8_BATCHED_GEMV_MAX: atomic::AtomicUsize =
    atomic::AtomicUsize::new(E8_BATCHED_GEMV_UNSET);
/// `HIPFIRE_DEEPSEEK4_FFN_OVERLAP` — evaluate the shared E8 expert
/// concurrently with the routed MQ2 experts on gfx1151. The fresh-process
/// A/B gate retained this route; set `=0` for the serial fallback.
pub(crate) fn ffn_overlap_on(arch: &str, mq2r: bool) -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    !mq2r
        && arch == "gfx1151"
        && *V.get_or_init(|| {
            hipfire_config::developer_var("HIPFIRE_DEEPSEEK4_FFN_OVERLAP")
                .ok()
                .as_deref()
                != Some("0")
        })
}
/// `HIPFIRE_DEEPSEEK4_HC_PINGPONG` — write each HC mix into a dedicated
/// alternate residual buffer and swap handles instead of issuing 86 D2D
/// copies per token. Opt-in until exact-model parity/perf admission.
pub(crate) fn hc_pingpong_on(arch: &str, mq2r: bool) -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    arch == "gfx1151" && (mq2r || *V.get_or_init(|| flag_one("HIPFIRE_DEEPSEEK4_HC_PINGPONG")))
}
pub(crate) fn hc_finalize_fused_on(arch: &str, mq2r: bool) -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    (arch == "gfx1151"
        && (mq2r || *V.get_or_init(|| flag_one("HIPFIRE_DEEPSEEK4_HC_FINALIZE_FUSED"))))
        || (arch == "gfx1201" && mq2r)
}
pub(crate) fn hc_control_finalize_fused_on(arch: &str, mq2r: bool) -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    (arch == "gfx1151"
        && (mq2r || *V.get_or_init(|| flag_one("HIPFIRE_DEEPSEEK4_HC_CONTROL_FINALIZE_FUSED"))))
        || (arch == "gfx1201" && mq2r)
}
pub(crate) fn retained_embedding_on(arch: &str, mq2r: bool) -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    !mq2r
        && arch == "gfx1151"
        && *V.get_or_init(|| flag_one("HIPFIRE_DEEPSEEK4_RETAINED_EMBEDDING"))
}
pub(crate) fn hc_control_rsqrt_once_on(arch: &str, mq2r: bool) -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    !mq2r
        && arch == "gfx1151"
        && *V.get_or_init(|| flag_one("HIPFIRE_DEEPSEEK4_HC_CONTROL_RSQRT_ONCE"))
}
pub(crate) fn hc_finalize_input_map_on(arch: &str, mq2r: bool) -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    !mq2r
        && arch == "gfx1151"
        && *V.get_or_init(|| flag_one("HIPFIRE_DEEPSEEK4_HC_FINALIZE_INPUT_MAP"))
}
pub(crate) fn qnorm_rotate_fused_on(arch: &str, mq2r: bool) -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    !mq2r
        && arch == "gfx1151"
        && *V.get_or_init(|| flag_one("HIPFIRE_DEEPSEEK4_QNORM_ROTATE_FUSED"))
}
pub(crate) fn redline_ffn_split_on(arch: &str, mq2r: bool) -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    !mq2r
        && arch == "gfx1151"
        && *V.get_or_init(|| flag_one("HIPFIRE_DEEPSEEK4_REDLINE_FFN_SPLIT"))
}
