#![allow(dead_code)]

// SPDX-License-Identifier: Apache-2.0
//
// Generated automatically from source env usage by scripts/gen-env-docs.py.
// Do not hand-edit. Re-run `./scripts/regen-env-vars-doc.sh`.

/// Canonical environment-variable documentation registry.
///
/// Each entry is sourced from inline comments or generated defaults.
pub struct EnvVarDoc {
    pub name: &'static str,
    pub description: &'static str,
    pub source: &'static str,
}

impl EnvVarDoc {
    pub const fn name(&self) -> &'static str {
        self.name
    }
}

/// `BENCH_BATCH` — Runtime variable controlling bench batch in hipfire
pub const ENV_BENCH_BATCH: EnvVarDoc = EnvVarDoc {
    name: "BENCH_BATCH",
    description: "Runtime variable controlling bench batch in hipfire",
    source: "/home/sadara/.hipfire/src/crates/rdna-compute/examples/bench_stream_overlap.rs:56",
};

/// `BENCH_DRAFT_K` — Runtime variable controlling bench draft k in hipfire
pub const ENV_BENCH_DRAFT_K: EnvVarDoc = EnvVarDoc {
    name: "BENCH_DRAFT_K",
    description: "Runtime variable controlling bench draft k in hipfire",
    source: "/home/sadara/.hipfire/src/crates/rdna-compute/examples/bench_stream_overlap.rs:188",
};

/// `BENCH_DRAFT_LAYERS` — Runtime variable controlling bench draft layers in hipfire
pub const ENV_BENCH_DRAFT_LAYERS: EnvVarDoc = EnvVarDoc {
    name: "BENCH_DRAFT_LAYERS",
    description: "Runtime variable controlling bench draft layers in hipfire",
    source: "/home/sadara/.hipfire/src/crates/rdna-compute/examples/bench_stream_overlap.rs:215",
};

/// `BENCH_DRAFT_M` — Runtime variable controlling bench draft m in hipfire
pub const ENV_BENCH_DRAFT_M: EnvVarDoc = EnvVarDoc {
    name: "BENCH_DRAFT_M",
    description: "Runtime variable controlling bench draft m in hipfire",
    source: "/home/sadara/.hipfire/src/crates/rdna-compute/examples/bench_stream_overlap.rs:184",
};

/// `BENCH_DRAFT_N` — Runtime variable controlling bench draft n in hipfire
pub const ENV_BENCH_DRAFT_N: EnvVarDoc = EnvVarDoc {
    name: "BENCH_DRAFT_N",
    description: "Runtime variable controlling bench draft n in hipfire",
    source: "/home/sadara/.hipfire/src/crates/rdna-compute/examples/bench_stream_overlap.rs:192",
};

/// `BENCH_K` — Runtime variable controlling bench k in hipfire
pub const ENV_BENCH_K: EnvVarDoc = EnvVarDoc {
    name: "BENCH_K",
    description: "Runtime variable controlling bench k in hipfire",
    source: "/home/sadara/.hipfire/src/crates/rdna-compute/examples/bench_stream_overlap.rs:52",
};

/// `BENCH_M` — Runtime variable controlling bench m in hipfire
pub const ENV_BENCH_M: EnvVarDoc = EnvVarDoc {
    name: "BENCH_M",
    description: "Runtime variable controlling bench m in hipfire",
    source: "/home/sadara/.hipfire/src/crates/rdna-compute/examples/bench_stream_overlap.rs:48",
};

/// `BENCH_VERIFY_LAYERS` — Runtime variable controlling bench verify layers in hipfire
pub const ENV_BENCH_VERIFY_LAYERS: EnvVarDoc = EnvVarDoc {
    name: "BENCH_VERIFY_LAYERS",
    description: "Runtime variable controlling bench verify layers in hipfire",
    source: "/home/sadara/.hipfire/src/crates/rdna-compute/examples/bench_stream_overlap.rs:219",
};

/// `CLICOLOR` — disables when stdout/stderr aren't TTYs (already gated above for stdin
pub const ENV_CLICOLOR: EnvVarDoc = EnvVarDoc {
    name: "CLICOLOR",
    description: "disables when stdout/stderr aren't TTYs (already gated above for stdin",
    source: "/home/sadara/.hipfire/src/cli/chat.ts:101",
};

/// `DDTREE_TIMING` — pre_verify / verify. The draft and top-K are fused into one GPU-
pub const ENV_DDTREE_TIMING: EnvVarDoc = EnvVarDoc {
    name: "DDTREE_TIMING",
    description: "pre_verify / verify. The draft and top-K are fused into one GPU-",
    source: "/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/speculative.rs:4838",
};

/// `DEBUG_LAYERS` — Runtime variable controlling debug layers in hipfire
pub const ENV_DEBUG_LAYERS: EnvVarDoc = EnvVarDoc {
    name: "DEBUG_LAYERS",
    description: "Runtime variable controlling debug layers in hipfire",
    source: "/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/qwen35.rs:6793",
};

/// `DFLASH_LIVE_TAU` — Runtime variable controlling dflash live tau in hipfire
pub const ENV_DFLASH_LIVE_TAU: EnvVarDoc = EnvVarDoc {
    name: "DFLASH_LIVE_TAU",
    description: "Runtime variable controlling dflash live tau in hipfire",
    source: "/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/dflash_spec_demo.rs:1493",
};

/// `DP4A` — Force the candidate into out_tp: W64=1 → dp4a wave64, DP4A=1 →
pub const ENV_DP4A: EnvVarDoc = EnvVarDoc {
    name: "DP4A",
    description: "Force the candidate into out_tp: W64=1 → dp4a wave64, DP4A=1 →",
    source:
        "/home/sadara/.hipfire/src/crates/rdna-compute/examples/q8_batched_attn_microbench.rs:104",
};

/// `FP32_STATE` — Runtime variable controlling fp32 state in hipfire
pub const ENV_FP32_STATE: EnvVarDoc = EnvVarDoc {
    name: "FP32_STATE",
    description: "Runtime variable controlling fp32 state in hipfire",
    source: "/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/infer_qwen35.rs:191",
};

/// `HFQ_TEST_N_ITER` — Parses "HFQ_TEST_N_ITER" with fallback defaults
pub const ENV_HFQ_TEST_N_ITER: EnvVarDoc = EnvVarDoc {
    name: "HFQ_TEST_N_ITER",
    description: "Parses \"HFQ_TEST_N_ITER\" with fallback defaults",
    source: "/home/sadara/.hipfire/src/crates/rdna-compute/examples/test_hfq4_residual_dp4a.rs:74",
};

/// `HFQ_TEST_SCALE_LOG10` — Parses "HFQ_TEST_SCALE_LOG10" with fallback defaults
pub const ENV_HFQ_TEST_SCALE_LOG10: EnvVarDoc = EnvVarDoc {
    name: "HFQ_TEST_SCALE_LOG10",
    description: "Parses \"HFQ_TEST_SCALE_LOG10\" with fallback defaults",
    source: "/home/sadara/.hipfire/src/crates/rdna-compute/examples/test_hfq4_residual_dp4a.rs:196",
};

/// `HFQ_TEST_ZP_MAX` — Parses "HFQ_TEST_ZP_MAX" with fallback defaults
pub const ENV_HFQ_TEST_ZP_MAX: EnvVarDoc = EnvVarDoc {
    name: "HFQ_TEST_ZP_MAX",
    description: "Parses \"HFQ_TEST_ZP_MAX\" with fallback defaults",
    source: "/home/sadara/.hipfire/src/crates/rdna-compute/examples/test_hfq4_residual_dp4a.rs:200",
};

/// `HF_TOKEN` — Runtime variable controlling hf token in hipfire
pub const ENV_HF_TOKEN: EnvVarDoc = EnvVarDoc {
    name: "HF_TOKEN",
    description: "Runtime variable controlling hf token in hipfire",
    source: "/home/sadara/.hipfire/src/cli/index.ts:989",
};

/// `HIPFIRE_ADAPTIVE_B_DOWN` — Runtime variable controlling adaptive b down in hipfire
pub const ENV_HIPFIRE_ADAPTIVE_B_DOWN: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_ADAPTIVE_B_DOWN",
    description: "Runtime variable controlling adaptive b down in hipfire",
    source: "/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/dflash_spec_demo.rs:1750",
};

/// `HIPFIRE_ADAPTIVE_B_UNSAFE` — the user explicitly widens. Opt out via HIPFIRE_ADAPTIVE_B_UNSAFE=1
pub const ENV_HIPFIRE_ADAPTIVE_B_UNSAFE: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_ADAPTIVE_B_UNSAFE",
    description: "the user explicitly widens. Opt out via HIPFIRE_ADAPTIVE_B_UNSAFE=1",
    source: "/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/dflash_spec_demo.rs:902",
};

/// `HIPFIRE_ADAPTIVE_B_UP` — HIPFIRE_ADAPTIVE_B_UP=0.XX / HIPFIRE_ADAPTIVE_B_DOWN=0.XX
pub const ENV_HIPFIRE_ADAPTIVE_B_UP: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_ADAPTIVE_B_UP",
    description: "HIPFIRE_ADAPTIVE_B_UP=0.XX / HIPFIRE_ADAPTIVE_B_DOWN=0.XX",
    source: "/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/dflash_spec_demo.rs:1746",
};

/// `HIPFIRE_ALLOW_MIXED_ARCH` — Runtime variable controlling allow mixed arch in hipfire
pub const ENV_HIPFIRE_ALLOW_MIXED_ARCH: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_ALLOW_MIXED_ARCH",
    description: "Runtime variable controlling allow mixed arch in hipfire",
    source: "/home/sadara/.hipfire/src/crates/hipfire-runtime/src/config.rs:95",
};

/// `HIPFIRE_ALLOW_MQ2` — Enabled when set to 1
pub const ENV_HIPFIRE_ALLOW_MQ2: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_ALLOW_MQ2",
    description: "Enabled when set to 1",
    source: "/home/sadara/.hipfire/src/crates/hipfire-quantize/src/main.rs:5917",
};

/// `HIPFIRE_ALLOW_MQ2_LLOYD` — Enabled when set to 1
pub const ENV_HIPFIRE_ALLOW_MQ2_LLOYD: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_ALLOW_MQ2_LLOYD",
    description: "Enabled when set to 1",
    source: "/home/sadara/.hipfire/src/crates/hipfire-quantize/src/main.rs:5952",
};

/// `HIPFIRE_ALLOW_MQ3_LLOYD` — Enabled when set to 1
pub const ENV_HIPFIRE_ALLOW_MQ3_LLOYD: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_ALLOW_MQ3_LLOYD",
    description: "Enabled when set to 1",
    source: "/home/sadara/.hipfire/src/crates/hipfire-quantize/src/main.rs:5937",
};

/// `HIPFIRE_ALLOW_MQ4_LLOYD` — Enabled when set to 1
pub const ENV_HIPFIRE_ALLOW_MQ4_LLOYD: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_ALLOW_MQ4_LLOYD",
    description: "Enabled when set to 1",
    source: "/home/sadara/.hipfire/src/crates/hipfire-quantize/src/main.rs:5987",
};

/// `HIPFIRE_ALLOW_UNIT_IMATRIX` — Environment toggle value controls runtime behavior
pub const ENV_HIPFIRE_ALLOW_UNIT_IMATRIX: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_ALLOW_UNIT_IMATRIX",
    description: "Environment toggle value controls runtime behavior",
    source: "/home/sadara/.hipfire/src/crates/hipfire-quantize/src/main.rs:5695",
};

/// `HIPFIRE_ATTN_FLASH` — Honors HIPFIRE_ATTN_FLASH=never\\\\\\\\\\\\|0\\\\\\\\\\\\|off as an explicit override
pub const ENV_HIPFIRE_ATTN_FLASH: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_ATTN_FLASH",
    description: "Honors HIPFIRE_ATTN_FLASH=never\\\\\\\\\\\\\\\\\\\\\\\\|0\\\\\\\\\\\\\\\\\\\\\\\\|off as an explicit override",
    source: "/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/qwen35.rs:7635",
};

/// `HIPFIRE_AWQ_F1_ONLY` — bench against the same binary's F2 quant. Default (env unset):
pub const ENV_HIPFIRE_AWQ_F1_ONLY: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_AWQ_F1_ONLY",
    description: "bench against the same binary's F2 quant. Default (env unset):",
    source: "/home/sadara/.hipfire/src/crates/hipfire-quantize/src/main.rs:4770",
};

/// `HIPFIRE_BASELINE_ARCH` — Runtime variable controlling baseline arch in hipfire
pub const ENV_HIPFIRE_BASELINE_ARCH: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_BASELINE_ARCH",
    description: "Runtime variable controlling baseline arch in hipfire",
    source: "/home/sadara/.hipfire/src/crates/hipfire-runtime/src/coherence_runtime.rs:440",
};

/// `HIPFIRE_BF16_DENSE_M128` — Enabled by default; set to 0 to disable
pub const ENV_HIPFIRE_BF16_DENSE_M128: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_BF16_DENSE_M128",
    description: "Enabled by default; set to 0 to disable",
    source: "/home/sadara/.hipfire/src/crates/rdna-compute/src/dispatch.rs:42726",
};

/// `HIPFIRE_BF16_MOE_M256` — Enabled when set to 1
pub const ENV_HIPFIRE_BF16_MOE_M256: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_BF16_MOE_M256",
    description: "Enabled when set to 1",
    source: "/home/sadara/.hipfire/src/crates/rdna-compute/src/dispatch.rs:21853",
};

/// `HIPFIRE_BF16_WEIGHTS` — Qwen35 BF16 weight load policy: unset/"auto" keeps native BF16 on BF16-capable RDNA arches and downgrades to F16 elsewhere; "native" forces BF16, "f16" forces same-size downgrade, "f32"/"legacy" expands for debug
pub const ENV_HIPFIRE_BF16_WEIGHTS: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_BF16_WEIGHTS",
    description: "Qwen35 BF16 weight load policy: unset/\"auto\" keeps native BF16 on BF16-capable RDNA arches and downgrades to F16 elsewhere; \"native\" forces BF16, \"f16\" forces same-size downgrade, \"f32\"/\"legacy\" expands for debug",
    source: "/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/qwen35.rs:207",
};

/// `HIPFIRE_BLOB_FORCE` — Graph / capture / deterministic
pub const ENV_HIPFIRE_BLOB_FORCE: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_BLOB_FORCE",
    description: "Graph / capture / deterministic",
    source: "/home/sadara/.hipfire/src/crates/rdna-compute/src/feature_flags.rs:234",
};

/// `HIPFIRE_CALIB_PROFILE` — Enable with HIPFIRE_CALIB_PROFILE=1; emits to stderr
pub const ENV_HIPFIRE_CALIB_PROFILE: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_CALIB_PROFILE",
    description: "Enable with HIPFIRE_CALIB_PROFILE=1; emits to stderr",
    source: "/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/triattn_validate.rs:40",
};

/// `HIPFIRE_CASK_OFF` — HIPFIRE_CASK_OFF=1 is an ops escape hatch: forces no auto-attach
pub const ENV_HIPFIRE_CASK_OFF: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_CASK_OFF",
    description: "HIPFIRE_CASK_OFF=1 is an ops escape hatch: forces no auto-attach",
    source: "/home/sadara/.hipfire/src/cli/index.ts:878",
};

/// `HIPFIRE_CHATML` — Enabled when set to 1
pub const ENV_HIPFIRE_CHATML: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_CHATML",
    description: "Enabled when set to 1",
    source:
        "/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/probe_argmax_agreement.rs:58",
};

/// `HIPFIRE_CHAT_TEMPLATE_FILE` — 1. Env-var override
pub const ENV_HIPFIRE_CHAT_TEMPLATE_FILE: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_CHAT_TEMPLATE_FILE",
    description: "1. Env-var override",
    source: "/home/sadara/.hipfire/src/crates/hipfire-daemon/src/main.rs:8963",
};

/// `HIPFIRE_CONV1D_TREE_GFX1151` — Environment toggle value controls runtime behavior
pub const ENV_HIPFIRE_CONV1D_TREE_GFX1151: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_CONV1D_TREE_GFX1151",
    description: "Environment toggle value controls runtime behavior",
    source:
        "/home/sadara/.hipfire/src/crates/rdna-compute/examples/bench_conv1d_tree_gfx1151.rs:22",
};

/// `HIPFIRE_DAEMON_BIN` — Runtime variable controlling daemon bin in hipfire
pub const ENV_HIPFIRE_DAEMON_BIN: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DAEMON_BIN",
    description: "Runtime variable controlling daemon bin in hipfire",
    source: "/home/sadara/.hipfire/src/crates/hipfire-server/src/daemon/engine.rs:166",
};

/// `HIPFIRE_DAEMON_RESIDENT_STATE_BUDGET_MB` — Runtime variable controlling daemon resident state budget mb in hipfire
pub const ENV_HIPFIRE_DAEMON_RESIDENT_STATE_BUDGET_MB: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DAEMON_RESIDENT_STATE_BUDGET_MB",
    description: "Runtime variable controlling daemon resident state budget mb in hipfire",
    source: "/home/sadara/.hipfire/src/crates/hipfire-daemon/src/main.rs:3998",
};

/// `HIPFIRE_DDTREE_BUDGET` — Runtime variable controlling DDTree budget in hipfire
pub const ENV_HIPFIRE_DDTREE_BUDGET: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DDTREE_BUDGET",
    description: "Runtime variable controlling DDTree budget in hipfire",
    source: "/home/sadara/.hipfire/src/crates/hipfire-runtime/src/config.rs:74",
};

/// `HIPFIRE_DDTREE_FORCE_SLOW` — HIPFIRE_DDTREE_FORCE_SLOW=1: force the slow (re-verify) path even when
pub const ENV_HIPFIRE_DDTREE_FORCE_SLOW: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DDTREE_FORCE_SLOW",
    description: "HIPFIRE_DDTREE_FORCE_SLOW=1: force the slow (re-verify) path even when",
    source: "/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/speculative.rs:5063",
};

/// `HIPFIRE_DDTREE_LOGW_CUTOFF` — Adaptive-B usage report — only meaningful when --adaptive-b is on
pub const ENV_HIPFIRE_DDTREE_LOGW_CUTOFF: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DDTREE_LOGW_CUTOFF",
    description: "Adaptive-B usage report — only meaningful when --adaptive-b is on",
    source: "/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/dflash_spec_demo.rs:2382",
};

/// `HIPFIRE_DDTREE_PATH_B_CAPTURE` — Path B slow-path-kill (opt-in, WIP). Replaces the ~40-50 ms full
pub const ENV_HIPFIRE_DDTREE_PATH_B_CAPTURE: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DDTREE_PATH_B_CAPTURE",
    description: "Path B slow-path-kill (opt-in, WIP). Replaces the ~40-50 ms full",
    source: "/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/speculative.rs:5102",
};

/// `HIPFIRE_DDTREE_PATH_C` — Resolve "HIPFIRE_DDTREE_PATH_C" ONCE before the decode loop. The
pub const ENV_HIPFIRE_DDTREE_PATH_C: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DDTREE_PATH_C",
    description: "Resolve \"HIPFIRE_DDTREE_PATH_C\" ONCE before the decode loop. The",
    source: "/home/sadara/.hipfire/src/crates/hipfire-daemon/src/main.rs:11532",
};

/// `HIPFIRE_DDTREE_PATH_C_VERBOSE` — Runtime variable controlling DDTree path c verbose in hipfire
pub const ENV_HIPFIRE_DDTREE_PATH_C_VERBOSE: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DDTREE_PATH_C_VERBOSE",
    description: "Runtime variable controlling DDTree path c verbose in hipfire",
    source: "/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/speculative.rs:5617",
};

/// `HIPFIRE_DDTREE_TAPE_DUMP` — Enabled when set to 1
pub const ENV_HIPFIRE_DDTREE_TAPE_DUMP: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DDTREE_TAPE_DUMP",
    description: "Enabled when set to 1",
    source: "/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/speculative.rs:5082",
};

/// `HIPFIRE_DDTREE_TOPK` — Runtime variable controlling DDTree topk in hipfire
pub const ENV_HIPFIRE_DDTREE_TOPK: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DDTREE_TOPK",
    description: "Runtime variable controlling DDTree topk in hipfire",
    source: "/home/sadara/.hipfire/src/crates/hipfire-runtime/src/config.rs:78",
};

/// `HIPFIRE_DDTREE_TREE_LA` — Opt out with HIPFIRE_DDTREE_TREE_LA=0 if a regression is suspected
pub const ENV_HIPFIRE_DDTREE_TREE_LA: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DDTREE_TREE_LA",
    description: "Opt out with HIPFIRE_DDTREE_TREE_LA=0 if a regression is suspected",
    source: "/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/speculative.rs:4944",
};

/// `HIPFIRE_DEBUG_CORRUPT_PREFIX_HASH_ONCE` — Runtime variable controlling debug corrupt prefix hash once in hipfire
pub const ENV_HIPFIRE_DEBUG_CORRUPT_PREFIX_HASH_ONCE: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DEBUG_CORRUPT_PREFIX_HASH_ONCE",
    description: "Runtime variable controlling debug corrupt prefix hash once in hipfire",
    source: "/home/sadara/.hipfire/src/cli/index.ts:2182",
};

/// `HIPFIRE_DEBUG_PREFIX_BOUNDARIES` — Runtime variable controlling debug prefix boundaries in hipfire
pub const ENV_HIPFIRE_DEBUG_PREFIX_BOUNDARIES: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DEBUG_PREFIX_BOUNDARIES",
    description: "Runtime variable controlling debug prefix boundaries in hipfire",
    source: "/home/sadara/.hipfire/src/crates/hipfire-daemon/src/main.rs:4837",
};

/// `HIPFIRE_DEBUG_SHARED_PREFIX_FANOUT` — Runtime variable controlling debug shared prefix fanout in hipfire
pub const ENV_HIPFIRE_DEBUG_SHARED_PREFIX_FANOUT: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DEBUG_SHARED_PREFIX_FANOUT",
    description: "Runtime variable controlling debug shared prefix fanout in hipfire",
    source: "/home/sadara/.hipfire/src/cli/index.ts:5386",
};

/// `HIPFIRE_DEEPSEEK4_ATTN` — main model's final_norm_and_head head-HC reduction. Without
pub const ENV_HIPFIRE_DEEPSEEK4_ATTN: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DEEPSEEK4_ATTN",
    description: "main model's final_norm_and_head head-HC reduction. Without",
    source: "/home/sadara/.hipfire/src/crates/hipfire-arch-deepseek4/src/forward.rs:68",
};

/// `HIPFIRE_DEEPSEEK4_ATTN_DEBUG_BISECT` — DEBUG: in-kernel bisect (HIPFIRE_DEEPSEEK4_ATTN_DEBUG_BISECT=1)
pub const ENV_HIPFIRE_DEEPSEEK4_ATTN_DEBUG_BISECT: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DEEPSEEK4_ATTN_DEBUG_BISECT",
    description: "DEBUG: in-kernel bisect (HIPFIRE_DEEPSEEK4_ATTN_DEBUG_BISECT=1)",
    source: "/home/sadara/.hipfire/src/crates/hipfire-arch-deepseek4/src/forward.rs:5368",
};

/// `HIPFIRE_DEEPSEEK4_ATTN_PER_POS` — DEBUG: HIPFIRE_DEEPSEEK4_ATTN_PER_POS=1 substitutes a per-position loop
pub const ENV_HIPFIRE_DEEPSEEK4_ATTN_PER_POS: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DEEPSEEK4_ATTN_PER_POS",
    description: "DEBUG: HIPFIRE_DEEPSEEK4_ATTN_PER_POS=1 substitutes a per-position loop",
    source: "/home/sadara/.hipfire/src/crates/hipfire-arch-deepseek4/src/forward.rs:5301",
};

/// `HIPFIRE_DEEPSEEK4_ATTN_TOPK_DIRECT` — Runtime variable controlling deepseek4 attn topk direct in hipfire
pub const ENV_HIPFIRE_DEEPSEEK4_ATTN_TOPK_DIRECT: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DEEPSEEK4_ATTN_TOPK_DIRECT",
    description: "Runtime variable controlling deepseek4 attn topk direct in hipfire",
    source: "/home/sadara/.hipfire/src/crates/hipfire-arch-deepseek4/src/forward.rs:5650",
};

/// `HIPFIRE_DEEPSEEK4_ATTN_TWIN` — DEBUG: same-process twin-call test (HIPFIRE_DEEPSEEK4_ATTN_TWIN=1)
pub const ENV_HIPFIRE_DEEPSEEK4_ATTN_TWIN: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DEEPSEEK4_ATTN_TWIN",
    description: "DEBUG: same-process twin-call test (HIPFIRE_DEEPSEEK4_ATTN_TWIN=1)",
    source: "/home/sadara/.hipfire/src/crates/hipfire-arch-deepseek4/src/forward.rs:5346",
};

/// `HIPFIRE_DEEPSEEK4_BATCH_HEAD` — Opt-out: HIPFIRE_DEEPSEEK4_BATCH_HEAD=0 forces the legacy per-position
pub const ENV_HIPFIRE_DEEPSEEK4_BATCH_HEAD: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DEEPSEEK4_BATCH_HEAD",
    description: "Opt-out: HIPFIRE_DEEPSEEK4_BATCH_HEAD=0 forces the legacy per-position",
    source: "/home/sadara/.hipfire/src/crates/hipfire-arch-deepseek4/src/forward.rs:7229",
};

/// `HIPFIRE_DEEPSEEK4_CACHE_TRACE` — Runtime variable controlling deepseek4 cache trace in hipfire
pub const ENV_HIPFIRE_DEEPSEEK4_CACHE_TRACE: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DEEPSEEK4_CACHE_TRACE",
    description: "Runtime variable controlling deepseek4 cache trace in hipfire",
    source: "/home/sadara/.hipfire/src/crates/hipfire-daemon/src/main.rs:15061",
};

/// `HIPFIRE_DEEPSEEK4_CHAT_RAW` — Enabled when set to 1
pub const ENV_HIPFIRE_DEEPSEEK4_CHAT_RAW: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DEEPSEEK4_CHAT_RAW",
    description: "Enabled when set to 1",
    source:
        "/home/sadara/.hipfire/src/crates/hipfire-arch-deepseek4/examples/deepseek4_chat.rs:160",
};

/// `HIPFIRE_DEEPSEEK4_COMP_F16_WMMA` — (gemv_f32 path) while default MQ4G256 quants land as Raw
pub const ENV_HIPFIRE_DEEPSEEK4_COMP_F16_WMMA: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DEEPSEEK4_COMP_F16_WMMA",
    description: "(gemv_f32 path) while default MQ4G256 quants land as Raw",
    source: "/home/sadara/.hipfire/src/crates/hipfire-arch-deepseek4/src/forward.rs:5721",
};

/// `HIPFIRE_DEEPSEEK4_COMP_ROPE_POS` — Runtime variable controlling deepseek4 comp rope pos in hipfire
pub const ENV_HIPFIRE_DEEPSEEK4_COMP_ROPE_POS: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DEEPSEEK4_COMP_ROPE_POS",
    description: "Runtime variable controlling deepseek4 comp rope pos in hipfire",
    source: "/home/sadara/.hipfire/src/crates/hipfire-arch-deepseek4/src/forward.rs:4390",
};

/// `HIPFIRE_DEEPSEEK4_DUMP_PROMPT` — Runtime variable controlling deepseek4 dump prompt in hipfire
pub const ENV_HIPFIRE_DEEPSEEK4_DUMP_PROMPT: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DEEPSEEK4_DUMP_PROMPT",
    description: "Runtime variable controlling deepseek4 dump prompt in hipfire",
    source: "/home/sadara/.hipfire/src/crates/hipfire-daemon/src/main.rs:14411",
};

/// `HIPFIRE_DEEPSEEK4_DUMP_STATE` — Determinism-bisection helper: when "HIPFIRE_DEEPSEEK4_DUMP_STATE=<dir>" is
pub const ENV_HIPFIRE_DEEPSEEK4_DUMP_STATE: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DEEPSEEK4_DUMP_STATE",
    description: "Determinism-bisection helper: when \"HIPFIRE_DEEPSEEK4_DUMP_STATE=<dir>\" is",
    source: "/home/sadara/.hipfire/src/crates/hipfire-arch-deepseek4/src/forward.rs:208",
};

/// `HIPFIRE_DEEPSEEK4_DUMP_TOPK` — [B, K_TOP] block of i32 per layer per chunk. Off by default
pub const ENV_HIPFIRE_DEEPSEEK4_DUMP_TOPK: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DEEPSEEK4_DUMP_TOPK",
    description: "[B, K_TOP] block of i32 per layer per chunk. Off by default",
    source: "/home/sadara/.hipfire/src/crates/hipfire-arch-deepseek4/src/forward.rs:6630",
};

/// `HIPFIRE_DEEPSEEK4_EXPERT_LAYER_END` — Runtime variable controlling deepseek4 expert layer end in hipfire
pub const ENV_HIPFIRE_DEEPSEEK4_EXPERT_LAYER_END: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DEEPSEEK4_EXPERT_LAYER_END",
    description: "Runtime variable controlling deepseek4 expert layer end in hipfire",
    source: "/home/sadara/.hipfire/src/crates/hipfire-arch-deepseek4/src/arch.rs:490",
};

/// `HIPFIRE_DEEPSEEK4_F32_TRACE` — Runtime variable controlling deepseek4 f32 trace in hipfire
pub const ENV_HIPFIRE_DEEPSEEK4_F32_TRACE: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DEEPSEEK4_F32_TRACE",
    description: "Runtime variable controlling deepseek4 f32 trace in hipfire",
    source: "/home/sadara/.hipfire/src/crates/hipfire-arch-deepseek4/src/forward.rs:236",
};

/// `HIPFIRE_DEEPSEEK4_FUSED_UNSCATTER_SILU` — feedback_kernel_fusion. Default OFF; opt in via
pub const ENV_HIPFIRE_DEEPSEEK4_FUSED_UNSCATTER_SILU: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DEEPSEEK4_FUSED_UNSCATTER_SILU",
    description: "feedback_kernel_fusion. Default OFF; opt in via",
    source: "/home/sadara/.hipfire/src/crates/hipfire-arch-deepseek4/src/forward.rs:6860",
};

/// `HIPFIRE_DEEPSEEK4_GEN_TOKENS` — Runtime variable controlling deepseek4 gen tokens in hipfire
pub const ENV_HIPFIRE_DEEPSEEK4_GEN_TOKENS: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DEEPSEEK4_GEN_TOKENS",
    description: "Runtime variable controlling deepseek4 gen tokens in hipfire",
    source:
        "/home/sadara/.hipfire/src/crates/hipfire-arch-deepseek4/examples/deepseek4_chat.rs:156",
};

/// `HIPFIRE_DEEPSEEK4_GRAPH` — 200+ steps on gfx1151 (graph_drift_check). Default ON for RDNA3+
pub const ENV_HIPFIRE_DEEPSEEK4_GRAPH: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DEEPSEEK4_GRAPH",
    description: "200+ steps on gfx1151 (graph_drift_check). Default ON for RDNA3+",
    source: "/home/sadara/.hipfire/src/crates/hipfire-arch-deepseek4/src/forward.rs:1488",
};

/// `HIPFIRE_DEEPSEEK4_HFQ4_WMMA` — HFQ4G256/Raw. WMMA route requires F16 input staging
pub const ENV_HIPFIRE_DEEPSEEK4_HFQ4_WMMA: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DEEPSEEK4_HFQ4_WMMA",
    description: "HFQ4G256/Raw. WMMA route requires F16 input staging",
    source: "/home/sadara/.hipfire/src/crates/hipfire-arch-deepseek4/src/forward.rs:310",
};

/// `HIPFIRE_DEEPSEEK4_INDEXER_WMMA` — catch-up plan. Opt out via HIPFIRE_DEEPSEEK4_INDEXER_WMMA=0
pub const ENV_HIPFIRE_DEEPSEEK4_INDEXER_WMMA: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DEEPSEEK4_INDEXER_WMMA",
    description: "catch-up plan. Opt out via HIPFIRE_DEEPSEEK4_INDEXER_WMMA=0",
    source: "/home/sadara/.hipfire/src/crates/hipfire-arch-deepseek4/src/forward.rs:6076",
};

/// `HIPFIRE_DEEPSEEK4_LOAD_MTP` — Runtime variable controlling deepseek4 load MTP in hipfire
pub const ENV_HIPFIRE_DEEPSEEK4_LOAD_MTP: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DEEPSEEK4_LOAD_MTP",
    description: "Runtime variable controlling deepseek4 load MTP in hipfire",
    source: "/home/sadara/.hipfire/src/crates/hipfire-arch-deepseek4/src/arch.rs:889",
};

/// `HIPFIRE_DEEPSEEK4_MAX_COMPRESS_POS` — ratio == 128: identity gather, no indexer. Per-batch n_compressed
pub const ENV_HIPFIRE_DEEPSEEK4_MAX_COMPRESS_POS: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DEEPSEEK4_MAX_COMPRESS_POS",
    description: "ratio == 128: identity gather, no indexer. Per-batch n_compressed",
    source: "/home/sadara/.hipfire/src/crates/hipfire-arch-deepseek4/src/forward.rs:6158",
};

/// `HIPFIRE_DEEPSEEK4_MODEL` — Runtime variable controlling deepseek4 model in hipfire
pub const ENV_HIPFIRE_DEEPSEEK4_MODEL: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DEEPSEEK4_MODEL",
    description: "Runtime variable controlling deepseek4 model in hipfire",
    source:
        "/home/sadara/.hipfire/src/crates/hipfire-arch-deepseek4/examples/deepseek4_chat.rs:154",
};

/// `HIPFIRE_DEEPSEEK4_MOE` — default ON. Opt out with "0" for diagnostic
pub const ENV_HIPFIRE_DEEPSEEK4_MOE: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DEEPSEEK4_MOE",
    description: "default ON. Opt out with \"0\" for diagnostic",
    source: "/home/sadara/.hipfire/src/crates/hipfire-arch-deepseek4/src/forward.rs:6534",
};

/// `HIPFIRE_DEEPSEEK4_MOE_8W` — Lever 3 (HIPFIRE_DEEPSEEK4_MOE_8W=1): 8-warp variant (shares staged X
pub const ENV_HIPFIRE_DEEPSEEK4_MOE_8W: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DEEPSEEK4_MOE_8W",
    description: "Lever 3 (HIPFIRE_DEEPSEEK4_MOE_8W=1): 8-warp variant (shares staged X",
    source: "/home/sadara/.hipfire/src/crates/hipfire-arch-deepseek4/src/forward.rs:6740",
};

/// `HIPFIRE_DEEPSEEK4_MOE_CND` — Lever 2 (HIPFIRE_DEEPSEEK4_MOE_CND=1): cndmask dequant on the n16 4w
pub const ENV_HIPFIRE_DEEPSEEK4_MOE_CND: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DEEPSEEK4_MOE_CND",
    description: "Lever 2 (HIPFIRE_DEEPSEEK4_MOE_CND=1): cndmask dequant on the n16 4w",
    source: "/home/sadara/.hipfire/src/crates/hipfire-arch-deepseek4/src/forward.rs:6737",
};

/// `HIPFIRE_DEEPSEEK4_MOE_DETERMINISTIC` — - HIPFIRE_DEEPSEEK4_MOE_DETERMINISTIC=0 (faster ~5–10 % prefill, non-
pub const ENV_HIPFIRE_DEEPSEEK4_MOE_DETERMINISTIC: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DEEPSEEK4_MOE_DETERMINISTIC",
    description: "- HIPFIRE_DEEPSEEK4_MOE_DETERMINISTIC=0 (faster ~5–10 % prefill, non-",
    source: "/home/sadara/.hipfire/src/crates/hipfire-arch-deepseek4/src/forward.rs:7096",
};

/// `HIPFIRE_DEEPSEEK4_MOE_GROUPED` — future sweeps. HIPFIRE_DEEPSEEK4_MOE_GROUPED=0 forces the scalar
pub const ENV_HIPFIRE_DEEPSEEK4_MOE_GROUPED: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DEEPSEEK4_MOE_GROUPED",
    description: "future sweeps. HIPFIRE_DEEPSEEK4_MOE_GROUPED=0 forces the scalar",
    source: "/home/sadara/.hipfire/src/crates/hipfire-arch-deepseek4/src/forward.rs:6679",
};

/// `HIPFIRE_DEEPSEEK4_MOE_GROUPED_GATE` — 46.02 vs 39.28 = +17.2%). Setting threshold at 128 captures the
pub const ENV_HIPFIRE_DEEPSEEK4_MOE_GROUPED_GATE: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DEEPSEEK4_MOE_GROUPED_GATE",
    description: "46.02 vs 39.28 = +17.2%). Setting threshold at 128 captures the",
    source: "/home/sadara/.hipfire/src/crates/hipfire-arch-deepseek4/src/forward.rs:6674",
};

/// `HIPFIRE_DEEPSEEK4_MOE_LLOYD_4W` — the default ON for gfx11+ — #356 broadened the gate from gfx1151-only
pub const ENV_HIPFIRE_DEEPSEEK4_MOE_LLOYD_4W: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DEEPSEEK4_MOE_LLOYD_4W",
    description: "the default ON for gfx11+ — #356 broadened the gate from gfx1151-only",
    source: "/home/sadara/.hipfire/src/crates/hipfire-arch-deepseek4/src/forward.rs:6913",
};

/// `HIPFIRE_DEEPSEEK4_MOE_MMQLOAD` — gfx1151; the corrected kernel measured flat, so no reason to default it)
pub const ENV_HIPFIRE_DEEPSEEK4_MOE_MMQLOAD: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DEEPSEEK4_MOE_MMQLOAD",
    description: "gfx1151; the corrected kernel measured flat, so no reason to default it)",
    source: "/home/sadara/.hipfire/src/crates/hipfire-arch-deepseek4/src/forward.rs:6921",
};

/// `HIPFIRE_DEEPSEEK4_MOE_N32` — Lever 1 (HIPFIRE_DEEPSEEK4_MOE_N32=1): N_TILE=32 tile-pairing on the
pub const ENV_HIPFIRE_DEEPSEEK4_MOE_N32: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DEEPSEEK4_MOE_N32",
    description: "Lever 1 (HIPFIRE_DEEPSEEK4_MOE_N32=1): N_TILE=32 tile-pairing on the",
    source: "/home/sadara/.hipfire/src/crates/hipfire-arch-deepseek4/src/forward.rs:6734",
};

/// `HIPFIRE_DEEPSEEK4_MOE_NOSYNC` — gfx1151; the corrected kernel measured flat, so no reason to default it)
pub const ENV_HIPFIRE_DEEPSEEK4_MOE_NOSYNC: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DEEPSEEK4_MOE_NOSYNC",
    description: "gfx1151; the corrected kernel measured flat, so no reason to default it)",
    source: "/home/sadara/.hipfire/src/crates/hipfire-arch-deepseek4/src/forward.rs:6923",
};

/// `HIPFIRE_DEEPSEEK4_MTP_ADDON` — Convention 1: append ".mtp-addon.hfq" (legacy)
pub const ENV_HIPFIRE_DEEPSEEK4_MTP_ADDON: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DEEPSEEK4_MTP_ADDON",
    description: "Convention 1: append \".mtp-addon.hfq\" (legacy)",
    source: "/home/sadara/.hipfire/src/crates/hipfire-arch-deepseek4/src/arch.rs:510",
};

/// `HIPFIRE_DEEPSEEK4_MTP_HEAD_HC` — input-side HC fix shipped in 82224ad). Opt out with =0 for
pub const ENV_HIPFIRE_DEEPSEEK4_MTP_HEAD_HC: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DEEPSEEK4_MTP_HEAD_HC",
    description: "input-side HC fix shipped in 82224ad). Opt out with =0 for",
    source: "/home/sadara/.hipfire/src/crates/hipfire-arch-deepseek4/src/forward.rs:81",
};

/// `HIPFIRE_DEEPSEEK4_MTP_SKIP_HEAD` — HIPFIRE_DEEPSEEK4_MTP_SKIP_HEAD=1 short-circuits steps 8+9 (return empty
pub const ENV_HIPFIRE_DEEPSEEK4_MTP_SKIP_HEAD: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DEEPSEEK4_MTP_SKIP_HEAD",
    description: "HIPFIRE_DEEPSEEK4_MTP_SKIP_HEAD=1 short-circuits steps 8+9 (return empty",
    source: "/home/sadara/.hipfire/src/crates/hipfire-arch-deepseek4/src/forward.rs:8078",
};

/// `HIPFIRE_DEEPSEEK4_POST_SCALE` — Default post_scale = 1.5: empirical optimum under mixed attention
pub const ENV_HIPFIRE_DEEPSEEK4_POST_SCALE: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DEEPSEEK4_POST_SCALE",
    description: "Default post_scale = 1.5: empirical optimum under mixed attention",
    source: "/home/sadara/.hipfire/src/crates/hipfire-arch-deepseek4/src/forward.rs:7424",
};

/// `HIPFIRE_DEEPSEEK4_PP_BATCH` — B=1024 default (bumped from 16 on 2026-05-26). Sweep at 2.1k tokens:
pub const ENV_HIPFIRE_DEEPSEEK4_PP_BATCH: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DEEPSEEK4_PP_BATCH",
    description: "B=1024 default (bumped from 16 on 2026-05-26). Sweep at 2.1k tokens:",
    source: "/home/sadara/.hipfire/src/crates/hipfire-daemon/src/main.rs:9474",
};

/// `HIPFIRE_DEEPSEEK4_Q8_4W` — Opt out via HIPFIRE_DEEPSEEK4_Q8_4W=0 for diagnosis
pub const ENV_HIPFIRE_DEEPSEEK4_Q8_4W: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DEEPSEEK4_Q8_4W",
    description: "Opt out via HIPFIRE_DEEPSEEK4_Q8_4W=0 for diagnosis",
    source: "/home/sadara/.hipfire/src/crates/hipfire-arch-deepseek4/src/forward.rs:275",
};

/// `HIPFIRE_DEEPSEEK4_Q8_WMMA` — bench_q8_wmma_variants. Opt-out via HIPFIRE_DEEPSEEK4_Q8_WMMA=0
pub const ENV_HIPFIRE_DEEPSEEK4_Q8_WMMA: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DEEPSEEK4_Q8_WMMA",
    description: "bench_q8_wmma_variants. Opt-out via HIPFIRE_DEEPSEEK4_Q8_WMMA=0",
    source: "/home/sadara/.hipfire/src/crates/hipfire-arch-deepseek4/src/forward.rs:256",
};

/// `HIPFIRE_DEEPSEEK4_ROUTE_SCALE` — Runtime variable controlling deepseek4 route scale in hipfire
pub const ENV_HIPFIRE_DEEPSEEK4_ROUTE_SCALE: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DEEPSEEK4_ROUTE_SCALE",
    description: "Runtime variable controlling deepseek4 route scale in hipfire",
    source: "/home/sadara/.hipfire/src/crates/hipfire-arch-deepseek4/src/forward.rs:6554",
};

/// `HIPFIRE_DEEPSEEK4_SEED` — Runtime variable controlling deepseek4 seed in hipfire
pub const ENV_HIPFIRE_DEEPSEEK4_SEED: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DEEPSEEK4_SEED",
    description: "Runtime variable controlling deepseek4 seed in hipfire",
    source: "/home/sadara/.hipfire/src/crates/hipfire-daemon/src/main.rs:14635",
};

/// `HIPFIRE_DEEPSEEK4_SPEC_DECODE` — Speculative decode opt-in. Default off because at current DeepSeek V4 MQ2-Lloyd
pub const ENV_HIPFIRE_DEEPSEEK4_SPEC_DECODE: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DEEPSEEK4_SPEC_DECODE",
    description: "Speculative decode opt-in. Default off because at current DeepSeek V4 MQ2-Lloyd",
    source: "/home/sadara/.hipfire/src/crates/hipfire-daemon/src/main.rs:14433",
};

/// `HIPFIRE_DEEPSEEK4_SPEC_K` — Runtime variable controlling deepseek4 spec k in hipfire
pub const ENV_HIPFIRE_DEEPSEEK4_SPEC_K: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DEEPSEEK4_SPEC_K",
    description: "Runtime variable controlling deepseek4 spec k in hipfire",
    source: "/home/sadara/.hipfire/src/crates/hipfire-daemon/src/main.rs:14441",
};

/// `HIPFIRE_DEEPSEEK4_TEMP` — Runtime variable controlling deepseek4 temp in hipfire
pub const ENV_HIPFIRE_DEEPSEEK4_TEMP: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DEEPSEEK4_TEMP",
    description: "Runtime variable controlling deepseek4 temp in hipfire",
    source:
        "/home/sadara/.hipfire/src/crates/hipfire-arch-deepseek4/examples/deepseek4_chat.rs:161",
};

/// `HIPFIRE_DEEPSEEK4_TOP_K` — for local deployment; we honor that as the default. Pure greedy
pub const ENV_HIPFIRE_DEEPSEEK4_TOP_K: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DEEPSEEK4_TOP_K",
    description: "for local deployment; we honor that as the default. Pure greedy",
    source: "/home/sadara/.hipfire/src/crates/hipfire-daemon/src/main.rs:14631",
};

/// `HIPFIRE_DEEPSEEK4_UPLOAD_EXPERTS` — For VRAM-constrained partial-MoE testing, set
pub const ENV_HIPFIRE_DEEPSEEK4_UPLOAD_EXPERTS: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DEEPSEEK4_UPLOAD_EXPERTS",
    description: "For VRAM-constrained partial-MoE testing, set",
    source: "/home/sadara/.hipfire/src/crates/hipfire-arch-deepseek4/src/arch.rs:486",
};

/// `HIPFIRE_DEEPSEEK4_WO_MULTIROW` — Multi-row variant if HIPFIRE_DEEPSEEK4_WO_MULTIROW=2 or 4
pub const ENV_HIPFIRE_DEEPSEEK4_WO_MULTIROW: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DEEPSEEK4_WO_MULTIROW",
    description: "Multi-row variant if HIPFIRE_DEEPSEEK4_WO_MULTIROW=2 or 4",
    source: "/home/sadara/.hipfire/src/crates/hipfire-arch-deepseek4/src/forward.rs:6295",
};

/// `HIPFIRE_DEEPSEEK4_WO_Q8_WMMA` — Runtime variable controlling deepseek4 wo Q8 wmma in hipfire
pub const ENV_HIPFIRE_DEEPSEEK4_WO_Q8_WMMA: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DEEPSEEK4_WO_Q8_WMMA",
    description: "Runtime variable controlling deepseek4 wo Q8 wmma in hipfire",
    source: "/home/sadara/.hipfire/src/crates/rdna-compute/src/dispatch.rs:47790",
};

/// `HIPFIRE_DELTANET_STATE` — Interprets "HIPFIRE_DELTANET_STATE" from environment to select behavior
pub const ENV_HIPFIRE_DELTANET_STATE: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DELTANET_STATE",
    description: "Interprets \"HIPFIRE_DELTANET_STATE\" from environment to select behavior",
    source: "/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/greedy_dump_top5.rs:245",
};

/// `HIPFIRE_DETERMINISTIC` — Enabled when set to 1
pub const ENV_HIPFIRE_DETERMINISTIC: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DETERMINISTIC",
    description: "Enabled when set to 1",
    source: "/home/sadara/.hipfire/src/crates/rdna-compute/src/feature_flags.rs:236",
};

/// `HIPFIRE_DEVICES` — Runtime variable controlling devices in hipfire
pub const ENV_HIPFIRE_DEVICES: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DEVICES",
    description: "Runtime variable controlling devices in hipfire",
    source: "/home/sadara/.hipfire/src/crates/hipfire-runtime/src/config.rs:94",
};

/// `HIPFIRE_DFLASH_DRAFT` — Runtime variable controlling dflash draft in hipfire
pub const ENV_HIPFIRE_DFLASH_DRAFT: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DFLASH_DRAFT",
    description: "Runtime variable controlling dflash draft in hipfire",
    source: "/home/sadara/.hipfire/src/crates/hipfire-runtime/src/config.rs:69",
};

/// `HIPFIRE_DFLASH_LOOP_BREAK` — Memory: HashSet<u64>, ≤ max_tokens / 12 entries (~125 for max=1500)
pub const ENV_HIPFIRE_DFLASH_LOOP_BREAK: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DFLASH_LOOP_BREAK",
    description: "Memory: HashSet<u64>, ≤ max_tokens / 12 entries (~125 for max=1500)",
    source: "/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/dflash_spec_demo.rs:1414",
};

/// `HIPFIRE_DFLASH_LOOP_BREAK_MAX_ESCALATIONS` — Runtime variable controlling dflash loop break max escalations in hipfire
pub const ENV_HIPFIRE_DFLASH_LOOP_BREAK_MAX_ESCALATIONS: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DFLASH_LOOP_BREAK_MAX_ESCALATIONS",
    description: "Runtime variable controlling dflash loop break max escalations in hipfire",
    source: "/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/dflash_spec_demo.rs:1443",
};

/// `HIPFIRE_DFLASH_LOOP_BREAK_RECOVERY` — Runtime variable controlling dflash loop break recovery in hipfire
pub const ENV_HIPFIRE_DFLASH_LOOP_BREAK_RECOVERY: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DFLASH_LOOP_BREAK_RECOVERY",
    description: "Runtime variable controlling dflash loop break recovery in hipfire",
    source: "/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/dflash_spec_demo.rs:1438",
};

/// `HIPFIRE_DFLASH_LOOP_BREAK_RP_MAX` — Runtime variable controlling dflash loop break rp max in hipfire
pub const ENV_HIPFIRE_DFLASH_LOOP_BREAK_RP_MAX: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DFLASH_LOOP_BREAK_RP_MAX",
    description: "Runtime variable controlling dflash loop break rp max in hipfire",
    source: "/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/dflash_spec_demo.rs:1434",
};

/// `HIPFIRE_DFLASH_LOOP_BREAK_RP_STEP` — Runtime variable controlling dflash loop break rp step in hipfire
pub const ENV_HIPFIRE_DFLASH_LOOP_BREAK_RP_STEP: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DFLASH_LOOP_BREAK_RP_STEP",
    description: "Runtime variable controlling dflash loop break rp step in hipfire",
    source: "/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/dflash_spec_demo.rs:1430",
};

/// `HIPFIRE_DFLASH_LOOP_BREAK_STOP_AFTER` — Runtime variable controlling dflash loop break stop after in hipfire
pub const ENV_HIPFIRE_DFLASH_LOOP_BREAK_STOP_AFTER: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DFLASH_LOOP_BREAK_STOP_AFTER",
    description: "Runtime variable controlling dflash loop break stop after in hipfire",
    source: "/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/dflash_spec_demo.rs:1426",
};

/// `HIPFIRE_DFLASH_LOOP_BREAK_TEMP` — Runtime variable controlling dflash loop break temp in hipfire
pub const ENV_HIPFIRE_DFLASH_LOOP_BREAK_TEMP: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DFLASH_LOOP_BREAK_TEMP",
    description: "Runtime variable controlling dflash loop break temp in hipfire",
    source: "/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/dflash_spec_demo.rs:1422",
};

/// `HIPFIRE_DFLASH_MODE` — Defaults to off when unset
pub const ENV_HIPFIRE_DFLASH_MODE: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DFLASH_MODE",
    description: "Defaults to off when unset",
    source: "/home/sadara/.hipfire/src/crates/hipfire-runtime/src/config.rs:70",
};

/// `HIPFIRE_DFLASH_MOE_DRAFT_FFN_GRAPH` — cycle); off-by-default for that reason
pub const ENV_HIPFIRE_DFLASH_MOE_DRAFT_FFN_GRAPH: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DFLASH_MOE_DRAFT_FFN_GRAPH",
    description: "cycle); off-by-default for that reason",
    source: "/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/speculative.rs:3284",
};

/// `HIPFIRE_DFLASH_MOE_VERIFY_GRAPH_LMHEAD` — shapes. sub_offset returns a non-owning view; do NOT free these
pub const ENV_HIPFIRE_DFLASH_MOE_VERIFY_GRAPH_LMHEAD: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DFLASH_MOE_VERIFY_GRAPH_LMHEAD",
    description: "shapes. sub_offset returns a non-owning view; do NOT free these",
    source: "/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/speculative.rs:2660",
};

/// `HIPFIRE_DFLASH_NGRAM_BLOCK` — Enabled when set to 1
pub const ENV_HIPFIRE_DFLASH_NGRAM_BLOCK: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DFLASH_NGRAM_BLOCK",
    description: "Enabled when set to 1",
    source: "/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/speculative.rs:3281",
};

/// `HIPFIRE_DFLASH_Q8_LMHEAD_WMMA` — Selects behavior from recognized values
pub const ENV_HIPFIRE_DFLASH_Q8_LMHEAD_WMMA: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DFLASH_Q8_LMHEAD_WMMA",
    description: "Selects behavior from recognized values",
    source: "/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/speculative.rs:33",
};

/// `HIPFIRE_DFLASH_ROLLBACK_COMPARE` — Diagnostic-only fast GDN-tape rollback replay comparison against serial replay
pub const ENV_HIPFIRE_DFLASH_ROLLBACK_COMPARE: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DFLASH_ROLLBACK_COMPARE",
    description: "Diagnostic-only fast GDN-tape rollback replay comparison against serial replay",
    source: "/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/speculative.rs:829",
};

/// `HIPFIRE_DFLASH_SEED_ORACLE` — Enabled when set to 1
pub const ENV_HIPFIRE_DFLASH_SEED_ORACLE: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DFLASH_SEED_ORACLE",
    description: "Enabled when set to 1",
    source: "/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/speculative.rs:3872",
};

/// `HIPFIRE_DFLASH_SPEC_DEMO_BIN` — Runtime variable controlling dflash spec demo bin in hipfire
pub const ENV_HIPFIRE_DFLASH_SPEC_DEMO_BIN: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DFLASH_SPEC_DEMO_BIN",
    description: "Runtime variable controlling dflash spec demo bin in hipfire",
    source: "/home/sadara/.hipfire/src/crates/hipfire-runtime/src/eval_harness/mod.rs:7924",
};

/// `HIPFIRE_DOT2_GEMV` — Interprets "HIPFIRE_DOT2_GEMV" from environment to select behavior
pub const ENV_HIPFIRE_DOT2_GEMV: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DOT2_GEMV",
    description: "Interprets \"HIPFIRE_DOT2_GEMV\" from environment to select behavior",
    source: "/home/sadara/.hipfire/src/crates/rdna-compute/src/feature_flags.rs:158",
};

/// `HIPFIRE_DOTS_OCR_BF16_RESIDUAL` — HF cast x to bf16 at vision forward entry (modeling_dots_vision.py
pub const ENV_HIPFIRE_DOTS_OCR_BF16_RESIDUAL: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DOTS_OCR_BF16_RESIDUAL",
    description: "HF cast x to bf16 at vision forward entry (modeling_dots_vision.py",
    source: "/home/sadara/.hipfire/src/crates/hipfire-arch-dots-ocr/src/dots_ocr.rs:1118",
};

/// `HIPFIRE_DOTS_OCR_DUMP_DIR` — HIPFIRE_DOTS_OCR_DUMP_DIR=<path>: dump full per-stage tensor
pub const ENV_HIPFIRE_DOTS_OCR_DUMP_DIR: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DOTS_OCR_DUMP_DIR",
    description: "HIPFIRE_DOTS_OCR_DUMP_DIR=<path>: dump full per-stage tensor",
    source: "/home/sadara/.hipfire/src/crates/hipfire-arch-dots-ocr/src/dots_ocr.rs:1020",
};

/// `HIPFIRE_DOTS_OCR_TRACE` — HIPFIRE_DOTS_OCR_TRACE=1: sync after every step + print probe so
pub const ENV_HIPFIRE_DOTS_OCR_TRACE: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DOTS_OCR_TRACE",
    description: "HIPFIRE_DOTS_OCR_TRACE=1: sync after every step + print probe so",
    source: "/home/sadara/.hipfire/src/crates/hipfire-arch-dots-ocr/src/dots_ocr.rs:1148",
};

/// `HIPFIRE_DPM_WARMUP_SECS` — Runtime variable controlling dpm warmup secs in hipfire
pub const ENV_HIPFIRE_DPM_WARMUP_SECS: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DPM_WARMUP_SECS",
    description: "Runtime variable controlling dpm warmup secs in hipfire",
    source: "/home/sadara/.hipfire/src/crates/rdna-compute/examples/bench_stream_overlap.rs:60",
};

/// `HIPFIRE_DRAFT_F16` — Enabled by default; set to 0 to disable
pub const ENV_HIPFIRE_DRAFT_F16: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DRAFT_F16",
    description: "Enabled by default; set to 0 to disable",
    source: "/home/sadara/.hipfire/src/crates/hipfire-runtime/src/config.rs:71",
};

/// `HIPFIRE_DRAFT_GEMM_DUMP` — Enabled when set to 1
pub const ENV_HIPFIRE_DRAFT_GEMM_DUMP: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DRAFT_GEMM_DUMP",
    description: "Enabled when set to 1",
    source: "/home/sadara/.hipfire/src/crates/hipfire-runtime/src/config.rs:72",
};

/// `HIPFIRE_DRAFT_SUBPHASE` — Enabled when set to 1
pub const ENV_HIPFIRE_DRAFT_SUBPHASE: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DRAFT_SUBPHASE",
    description: "Enabled when set to 1",
    source: "/home/sadara/.hipfire/src/crates/hipfire-runtime/src/config.rs:73",
};

/// `HIPFIRE_DTOH_DUMP` — Enabled when set to 1
pub const ENV_HIPFIRE_DTOH_DUMP: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DTOH_DUMP",
    description: "Enabled when set to 1",
    source: "/home/sadara/.hipfire/src/crates/hip-bridge/src/ffi.rs:938",
};

/// `HIPFIRE_DUMMY_GENERATE_DELAY_MS` — Parses "HIPFIRE_DUMMY_GENERATE_DELAY_MS" with fallback defaults
pub const ENV_HIPFIRE_DUMMY_GENERATE_DELAY_MS: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DUMMY_GENERATE_DELAY_MS",
    description: "Parses \"HIPFIRE_DUMMY_GENERATE_DELAY_MS\" with fallback defaults",
    source: "/home/sadara/.hipfire/src/crates/hipfire-daemon/src/main.rs:417",
};

/// `HIPFIRE_DUMMY_PREFILL_DELAY_MS` — Parses "HIPFIRE_DUMMY_PREFILL_DELAY_MS" with fallback defaults
pub const ENV_HIPFIRE_DUMMY_PREFILL_DELAY_MS: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DUMMY_PREFILL_DELAY_MS",
    description: "Parses \"HIPFIRE_DUMMY_PREFILL_DELAY_MS\" with fallback defaults",
    source: "/home/sadara/.hipfire/src/crates/hipfire-daemon/src/main.rs:409",
};

/// `HIPFIRE_DUMP_REQUEST` — a strace. Off by default — gigantic for typical agent prompts
pub const ENV_HIPFIRE_DUMP_REQUEST: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DUMP_REQUEST",
    description: "a strace. Off by default — gigantic for typical agent prompts",
    source: "/home/sadara/.hipfire/src/cli/index.ts:3867",
};

/// `HIPFIRE_EMIT_TOKEN_IDS` — or "\" in it would corrupt the line, breaking the client's JSONL
pub const ENV_HIPFIRE_EMIT_TOKEN_IDS: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_EMIT_TOKEN_IDS",
    description: "or \"\\\" in it would corrupt the line, breaking the client's JSONL",
    source: "/home/sadara/.hipfire/src/crates/hipfire-daemon/src/main.rs:602",
};

/// `HIPFIRE_EVAL_DATASET_MIRROR` — Runtime variable controlling eval dataset mirror in hipfire
pub const ENV_HIPFIRE_EVAL_DATASET_MIRROR: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_EVAL_DATASET_MIRROR",
    description: "Runtime variable controlling eval dataset mirror in hipfire",
    source: "/home/sadara/.hipfire/src/crates/hipfire-runtime/src/eval_harness/mod.rs:2538",
};

/// `HIPFIRE_EVAL_EVIDENCE_DIR` — --no-tape: disable GdnTape capture so spec_step_dflash replays via
pub const ENV_HIPFIRE_EVAL_EVIDENCE_DIR: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_EVAL_EVIDENCE_DIR",
    description: "--no-tape: disable GdnTape capture so spec_step_dflash replays via",
    source: "/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/run.rs:413",
};

/// `HIPFIRE_EVAL_HIPFIRE_BIN` — Runtime variable controlling eval hipfire bin in hipfire
pub const ENV_HIPFIRE_EVAL_HIPFIRE_BIN: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_EVAL_HIPFIRE_BIN",
    description: "Runtime variable controlling eval hipfire bin in hipfire",
    source: "/home/sadara/.hipfire/src/crates/hipfire-runtime/src/eval_harness/mod.rs:7954",
};

/// `HIPFIRE_EXPERIMENTAL_BUDGET_ALERT` — cache so the model "sees" them as part of its own trajectory,
pub const ENV_HIPFIRE_EXPERIMENTAL_BUDGET_ALERT: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_EXPERIMENTAL_BUDGET_ALERT",
    description: "cache so the model \"sees\" them as part of its own trajectory,",
    source: "/home/sadara/.hipfire/src/crates/hipfire-daemon/src/main.rs:7674",
};

/// `HIPFIRE_FLASH_PARTIALS_BATCH` — Parses "HIPFIRE_FLASH_PARTIALS_BATCH" with fallback defaults
pub const ENV_HIPFIRE_FLASH_PARTIALS_BATCH: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_FLASH_PARTIALS_BATCH",
    description: "Parses \"HIPFIRE_FLASH_PARTIALS_BATCH\" with fallback defaults",
    source: "/home/sadara/.hipfire/src/crates/hipfire-runtime/src/config.rs:83",
};

/// `HIPFIRE_FORCE_A3B_EVICTION` — Runtime variable controlling force a3b eviction in hipfire
pub const ENV_HIPFIRE_FORCE_A3B_EVICTION: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_FORCE_A3B_EVICTION",
    description: "Runtime variable controlling force a3b eviction in hipfire",
    source: "/home/sadara/.hipfire/src/cli/index.ts:10200",
};

/// `HIPFIRE_FP16` — Runtime variable controlling fp16 in hipfire
pub const ENV_HIPFIRE_FP16: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_FP16",
    description: "Runtime variable controlling fp16 in hipfire",
    source: "/home/sadara/.hipfire/src/crates/rdna-compute/src/feature_flags.rs:166",
};

/// `HIPFIRE_FP8_WMMA` — Interprets "HIPFIRE_FP8_WMMA" from environment to select behavior
pub const ENV_HIPFIRE_FP8_WMMA: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_FP8_WMMA",
    description: "Interprets \"HIPFIRE_FP8_WMMA\" from environment to select behavior",
    source: "/home/sadara/.hipfire/src/crates/rdna-compute/src/feature_flags.rs:157",
};

/// `HIPFIRE_FUSED_HFQ4_2ROW_GFX1151` — Opt-in ("1") gfx1151 two-row fused HFQ4 decode probe for QKV/QKVZA; default off after flat A3B/9B measurements
pub const ENV_HIPFIRE_FUSED_HFQ4_2ROW_GFX1151: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_FUSED_HFQ4_2ROW_GFX1151",
    description: "Opt-in (\"1\") gfx1151 two-row fused HFQ4 decode probe for QKV/QKVZA; default off after flat A3B/9B measurements",
    source: "/home/sadara/.hipfire/src/crates/rdna-compute/src/dispatch.rs:24066",
};

/// `HIPFIRE_GATE_UP_VARIANT` — Runtime variable controlling gate up variant in hipfire
pub const ENV_HIPFIRE_GATE_UP_VARIANT: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_GATE_UP_VARIANT",
    description: "Runtime variable controlling gate up variant in hipfire",
    source: "/home/sadara/.hipfire/src/crates/rdna-compute/src/feature_flags.rs:196",
};

/// `HIPFIRE_GDN_Q8_REG_GFX1151` — Opt-in ("1") gfx1151 register-state GDN Q8 probe; default off after A3B pp256 prefill regressed 11.8 ms to 168.5 ms total GDN time
pub const ENV_HIPFIRE_GDN_Q8_REG_GFX1151: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_GDN_Q8_REG_GFX1151",
    description: "Opt-in (\"1\") gfx1151 register-state GDN Q8 probe; default off after A3B pp256 prefill regressed 11.8 ms to 168.5 ms total GDN time",
    source: "/home/sadara/.hipfire/src/crates/rdna-compute/src/dispatch.rs:38439",
};

/// `HIPFIRE_GEMM_DUMP` — Enabled when set to 1
pub const ENV_HIPFIRE_GEMM_DUMP: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_GEMM_DUMP",
    description: "Enabled when set to 1",
    source: "/home/sadara/.hipfire/src/crates/rdna-compute/src/feature_flags.rs:235",
};

/// `HIPFIRE_GEMV_ROWS` — Runtime variable controlling gemv rows in hipfire
pub const ENV_HIPFIRE_GEMV_ROWS: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_GEMV_ROWS",
    description: "Runtime variable controlling gemv rows in hipfire",
    source: "/home/sadara/.hipfire/src/crates/rdna-compute/src/feature_flags.rs:138",
};

/// `HIPFIRE_GEN` — Runtime variable controlling gen in hipfire
pub const ENV_HIPFIRE_GEN: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_GEN",
    description: "Runtime variable controlling gen in hipfire",
    source: "/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/a3b_multiturn_oneshot.rs:25",
};

/// `HIPFIRE_GENERATE_BATCH_PREFILL_DEBUG_SAMPLE` — Runtime variable controlling generate batch prefill debug sample in hipfire
pub const ENV_HIPFIRE_GENERATE_BATCH_PREFILL_DEBUG_SAMPLE: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_GENERATE_BATCH_PREFILL_DEBUG_SAMPLE",
    description: "Runtime variable controlling generate batch prefill debug sample in hipfire",
    source: "/home/sadara/.hipfire/src/crates/hipfire-daemon/src/main.rs:5607",
};

/// `HIPFIRE_GFX942_GEMV_V3` — Interprets "HIPFIRE_GFX942_GEMV_V3" from environment to select behavior
pub const ENV_HIPFIRE_GFX942_GEMV_V3: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_GFX942_GEMV_V3",
    description: "Interprets \"HIPFIRE_GFX942_GEMV_V3\" from environment to select behavior",
    source: "/home/sadara/.hipfire/src/crates/rdna-compute/src/feature_flags.rs:198",
};

/// `HIPFIRE_GFX942_MFMA_PREFILL` — Interprets "HIPFIRE_GFX942_MFMA_PREFILL" from environment to select behavior
pub const ENV_HIPFIRE_GFX942_MFMA_PREFILL: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_GFX942_MFMA_PREFILL",
    description: "Interprets \"HIPFIRE_GFX942_MFMA_PREFILL\" from environment to select behavior",
    source: "/home/sadara/.hipfire/src/crates/rdna-compute/src/feature_flags.rs:201",
};

/// `HIPFIRE_GFX942_RMSNORM_SPLIT` — Environment toggle value controls runtime behavior
pub const ENV_HIPFIRE_GFX942_RMSNORM_SPLIT: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_GFX942_RMSNORM_SPLIT",
    description: "Environment toggle value controls runtime behavior",
    source: "/home/sadara/.hipfire/src/crates/rdna-compute/src/feature_flags.rs:200",
};

/// `HIPFIRE_GPTQ_DAMPING` — Inject env override since the quantizer reads it at fn entry
pub const ENV_HIPFIRE_GPTQ_DAMPING: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_GPTQ_DAMPING",
    description: "Inject env override since the quantizer reads it at fn entry",
    source: "/home/sadara/.hipfire/src/crates/hipfire-quantize/src/main.rs:8672",
};

/// `HIPFIRE_GPU_SLAB_LOAD` — Runtime variable controlling gpu slab load in hipfire
pub const ENV_HIPFIRE_GPU_SLAB_LOAD: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_GPU_SLAB_LOAD",
    description: "Runtime variable controlling gpu slab load in hipfire",
    source: "/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/qwen35.rs:3511",
};

/// `HIPFIRE_GPU_SLAB_MIB` — Parses "HIPFIRE_GPU_SLAB_MIB" with fallback defaults
pub const ENV_HIPFIRE_GPU_SLAB_MIB: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_GPU_SLAB_MIB",
    description: "Parses \"HIPFIRE_GPU_SLAB_MIB\" with fallback defaults",
    source: "/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/qwen35.rs:3192",
};

/// `HIPFIRE_GPU_TOPK` — HIPFIRE_GPU_TOPK=1 enables the GPU topk_logits_f32 kernel + CPU
pub const ENV_HIPFIRE_GPU_TOPK: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_GPU_TOPK",
    description: "HIPFIRE_GPU_TOPK=1 enables the GPU topk_logits_f32 kernel + CPU",
    source: "/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/infer_qwen35.rs:221",
};

/// `HIPFIRE_GQA_CHUNK` — Runtime variable controlling gqa chunk in hipfire
pub const ENV_HIPFIRE_GQA_CHUNK: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_GQA_CHUNK",
    description: "Runtime variable controlling gqa chunk in hipfire",
    source: "/home/sadara/.hipfire/src/crates/rdna-compute/src/dispatch.rs:31017",
};

/// `HIPFIRE_GQA_FUSED` — Fused variant (opt-in via HIPFIRE_GQA_FUSED=1): single launch
pub const ENV_HIPFIRE_GQA_FUSED: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_GQA_FUSED",
    description: "Fused variant (opt-in via HIPFIRE_GQA_FUSED=1): single launch",
    source: "/home/sadara/.hipfire/src/crates/hipfire-arch-qwen2/src/qwen2.rs:1061",
};

/// `HIPFIRE_GRAPH` — Used to configure runtime execution by explicitly setting "HIPFIRE_GRAPH"
pub const ENV_HIPFIRE_GRAPH: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_GRAPH",
    description: "Used to configure runtime execution by explicitly setting \"HIPFIRE_GRAPH\"",
    source: "/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/prefill_microbench.rs:119",
};

/// `HIPFIRE_GRAPH_MOE` — - gfx11 (RDNA3 / 3.5): default-ON. +0.6-0.7% decode on 9B and
pub const ENV_HIPFIRE_GRAPH_MOE: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_GRAPH_MOE",
    description: "- gfx11 (RDNA3 / 3.5): default-ON. +0.6-0.7% decode on 9B and",
    source: "/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/qwen35.rs:7884",
};

/// `HIPFIRE_GRAPH_PREFILL` — HIPFIRE_GRAPH_PREFILL=1: route the timed prefill loop through
pub const ENV_HIPFIRE_GRAPH_PREFILL: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_GRAPH_PREFILL",
    description: "HIPFIRE_GRAPH_PREFILL=1: route the timed prefill loop through",
    source: "/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/bench_qwen35_speed.rs:275",
};

/// `HIPFIRE_HAVE_2_GPU` — Environment toggle value controls runtime behavior
pub const ENV_HIPFIRE_HAVE_2_GPU: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_HAVE_2_GPU",
    description: "Environment toggle value controls runtime behavior",
    source: "/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/tests/pp_parity.rs:193",
};

/// `HIPFIRE_HETERO_DIFF` — above (#352's GPU greedy-accept path doesn't materialize it),
pub const ENV_HIPFIRE_HETERO_DIFF: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_HETERO_DIFF",
    description: "above (#352's GPU greedy-accept path doesn't materialize it),",
    source: "/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/mtp_spec.rs:2953",
};

/// `HIPFIRE_HFQ4G128_MMQ` — Environment toggle value controls runtime behavior
pub const ENV_HIPFIRE_HFQ4G128_MMQ: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_HFQ4G128_MMQ",
    description: "Environment toggle value controls runtime behavior",
    source: "/home/sadara/.hipfire/src/crates/rdna-compute/src/feature_flags.rs:191",
};

/// `HIPFIRE_HFQ4G256_MMQ_GFX1151` — gfx1151 one-wave i8-WMMA HFQ4-G256 MMQ control: unset auto-routes aligned K=2048 A3B/shared shapes, "0" forces fallback, "1" forces the probe on aligned shapes
pub const ENV_HIPFIRE_HFQ4G256_MMQ_GFX1151: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_HFQ4G256_MMQ_GFX1151",
    description: "gfx1151 one-wave i8-WMMA HFQ4-G256 MMQ control: unset auto-routes aligned K=2048 A3B/shared shapes, \"0\" forces fallback, \"1\" forces the probe on aligned shapes",
    source: "/home/sadara/.hipfire/src/crates/rdna-compute/src/dispatch.rs:24049",
};

/// `HIPFIRE_HFQ4_MMQ_GFX906_Y64` — Runtime variable controlling hfQ4 mmq gfx906 y64 in hipfire
pub const ENV_HIPFIRE_HFQ4_MMQ_GFX906_Y64: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_HFQ4_MMQ_GFX906_Y64",
    description: "Runtime variable controlling hfQ4 mmq gfx906 y64 in hipfire",
    source: "/home/sadara/.hipfire/src/crates/rdna-compute/src/feature_flags.rs:194",
};

/// `HIPFIRE_HFQ6_QKVZA_4W` — Interprets "HIPFIRE_HFQ6_QKVZA_4W" from environment to select behavior
pub const ENV_HIPFIRE_HFQ6_QKVZA_4W: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_HFQ6_QKVZA_4W",
    description: "Interprets \"HIPFIRE_HFQ6_QKVZA_4W\" from environment to select behavior",
    source: "/home/sadara/.hipfire/src/crates/rdna-compute/src/dispatch.rs:26153",
};

/// `HIPFIRE_HFQ6_QKV_4W` — Interprets "HIPFIRE_HFQ6_QKV_4W" from environment to select behavior
pub const ENV_HIPFIRE_HFQ6_QKV_4W: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_HFQ6_QKV_4W",
    description: "Interprets \"HIPFIRE_HFQ6_QKV_4W\" from environment to select behavior",
    source: "/home/sadara/.hipfire/src/crates/rdna-compute/src/dispatch.rs:26906",
};

/// `HIPFIRE_HFQ6_RESIDUAL_4W` — Interprets "HIPFIRE_HFQ6_RESIDUAL_4W" from environment to select behavior
pub const ENV_HIPFIRE_HFQ6_RESIDUAL_4W: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_HFQ6_RESIDUAL_4W",
    description: "Interprets \"HIPFIRE_HFQ6_RESIDUAL_4W\" from environment to select behavior",
    source: "/home/sadara/.hipfire/src/crates/rdna-compute/src/dispatch.rs:25760",
};

/// `HIPFIRE_HIPCC_EXTRA_FLAGS` — Parses "HIPFIRE_HIPCC_EXTRA_FLAGS" with fallback defaults
pub const ENV_HIPFIRE_HIPCC_EXTRA_FLAGS: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_HIPCC_EXTRA_FLAGS",
    description: "Parses \"HIPFIRE_HIPCC_EXTRA_FLAGS\" with fallback defaults",
    source: "/home/sadara/.hipfire/src/crates/rdna-compute/src/feature_flags.rs:261",
};

/// `HIPFIRE_HIP_WAIT` — Runtime variable controlling hip wait in hipfire
pub const ENV_HIPFIRE_HIP_WAIT: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_HIP_WAIT",
    description: "Runtime variable controlling hip wait in hipfire",
    source: "/home/sadara/.hipfire/src/crates/rdna-compute/src/dispatch.rs:698",
};

/// `HIPFIRE_HOST_PROFILE_BIN` — Runtime variable controlling host profile bin in hipfire
pub const ENV_HIPFIRE_HOST_PROFILE_BIN: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_HOST_PROFILE_BIN",
    description: "Runtime variable controlling host profile bin in hipfire",
    source: "/home/sadara/.hipfire/src/crates/hipfire-runtime/src/eval_harness/mod.rs:7969",
};

/// `HIPFIRE_HOST_TIMING` — HIPFIRE_HOST_TIMING=1: dump per-cycle host-side wall-clock breakdown
pub const ENV_HIPFIRE_HOST_TIMING: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_HOST_TIMING",
    description: "HIPFIRE_HOST_TIMING=1: dump per-cycle host-side wall-clock breakdown",
    source: "/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/dflash_spec_demo.rs:1703",
};

/// `HIPFIRE_JINJA_CHAT` — Enabled when set to 1
pub const ENV_HIPFIRE_JINJA_CHAT: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_JINJA_CHAT",
    description: "Enabled when set to 1",
    source: "/home/sadara/.hipfire/src/crates/hipfire-daemon/src/main.rs:15662",
};

/// `HIPFIRE_KERNEL_CACHE` — HIPFIRE_KERNEL_CACHE=/tmp/hipfire_kernels if tmpfs speed matters
pub const ENV_HIPFIRE_KERNEL_CACHE: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_KERNEL_CACHE",
    description: "HIPFIRE_KERNEL_CACHE=/tmp/hipfire_kernels if tmpfs speed matters",
    source: "/home/sadara/.hipfire/src/crates/rdna-compute/src/compiler.rs:106",
};

/// `HIPFIRE_KLD_DIRECT_F16KV_ATTN` — Used to configure runtime execution by explicitly setting "HIPFIRE_KLD_DIRECT_F16KV_ATTN"
pub const ENV_HIPFIRE_KLD_DIRECT_F16KV_ATTN: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_KLD_DIRECT_F16KV_ATTN",
    description: "Used to configure runtime execution by explicitly setting \"HIPFIRE_KLD_DIRECT_F16KV_ATTN\"",
    source: "/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/build_kld_ref_hipfire.rs:698",
};

/// `HIPFIRE_KLD_DIRECT_WMMA_ATTN` — Used to configure runtime execution by explicitly setting "HIPFIRE_KLD_DIRECT_WMMA_ATTN"
pub const ENV_HIPFIRE_KLD_DIRECT_WMMA_ATTN: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_KLD_DIRECT_WMMA_ATTN",
    description:
        "Used to configure runtime execution by explicitly setting \"HIPFIRE_KLD_DIRECT_WMMA_ATTN\"",
    source:
        "/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/build_kld_ref_hipfire.rs:697",
};

/// `HIPFIRE_KLD_FP32_GQA4_ATTN` — Used to configure runtime execution by explicitly setting "HIPFIRE_KLD_FP32_GQA4_ATTN"
pub const ENV_HIPFIRE_KLD_FP32_GQA4_ATTN: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_KLD_FP32_GQA4_ATTN",
    description:
        "Used to configure runtime execution by explicitly setting \"HIPFIRE_KLD_FP32_GQA4_ATTN\"",
    source:
        "/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/build_kld_ref_hipfire.rs:701",
};

/// `HIPFIRE_KLD_GPU_TOPK` — Enabled by default; set to 0 to disable
pub const ENV_HIPFIRE_KLD_GPU_TOPK: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_KLD_GPU_TOPK",
    description: "Enabled by default; set to 0 to disable",
    source:
        "/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/build_kld_ref_hipfire.rs:955",
};

/// `HIPFIRE_KLD_GRAPH` — Enabled by default; set to 0 to disable
pub const ENV_HIPFIRE_KLD_GRAPH: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_KLD_GRAPH",
    description: "Enabled by default; set to 0 to disable",
    source:
        "/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/build_kld_ref_hipfire.rs:676",
};

/// `HIPFIRE_KLD_NO_ACTIVE_STREAM` — Runtime variable controlling kld no active stream in hipfire
pub const ENV_HIPFIRE_KLD_NO_ACTIVE_STREAM: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_KLD_NO_ACTIVE_STREAM",
    description: "Runtime variable controlling kld no active stream in hipfire",
    source:
        "/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/build_kld_ref_hipfire.rs:290",
};

/// `HIPFIRE_KLD_PREFILL_ONLY` — Enabled when set to 1
pub const ENV_HIPFIRE_KLD_PREFILL_ONLY: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_KLD_PREFILL_ONLY",
    description: "Enabled when set to 1",
    source:
        "/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/build_kld_ref_hipfire.rs:843",
};

/// `HIPFIRE_KLD_SOURCE_SHA256` — Enabled when set to 1
pub const ENV_HIPFIRE_KLD_SOURCE_SHA256: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_KLD_SOURCE_SHA256",
    description: "Enabled when set to 1",
    source:
        "/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/build_kld_ref_hipfire.rs:839",
};

/// `HIPFIRE_KV_MODE` — Defaults to asym3 when unset
pub const ENV_HIPFIRE_KV_MODE: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_KV_MODE",
    description: "Defaults to asym3 when unset",
    source:
        "/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/probe_argmax_agreement.rs:116",
};

/// `HIPFIRE_KV_PHYSICAL_CAP` — Parses "HIPFIRE_KV_PHYSICAL_CAP" with fallback defaults
pub const ENV_HIPFIRE_KV_PHYSICAL_CAP: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_KV_PHYSICAL_CAP",
    description: "Parses \"HIPFIRE_KV_PHYSICAL_CAP\" with fallback defaults",
    source: "/home/sadara/.hipfire/src/crates/hipfire-daemon/src/main.rs:9251",
};

/// `HIPFIRE_LFM2_CAPTURE_POSTMIXER` — Runtime variable controlling lfm2 capture postmixer in hipfire
pub const ENV_HIPFIRE_LFM2_CAPTURE_POSTMIXER: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_LFM2_CAPTURE_POSTMIXER",
    description: "Runtime variable controlling lfm2 capture postmixer in hipfire",
    source: "/home/sadara/.hipfire/src/crates/hipfire-arch-lfm2moe/src/forward.rs:132",
};

/// `HIPFIRE_LFM2_EXPERT_MQ6` — opt-in via HIPFIRE_LFM2_EXPERT_MQ6 for higher quality), else mq4
pub const ENV_HIPFIRE_LFM2_EXPERT_MQ6: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_LFM2_EXPERT_MQ6",
    description: "opt-in via HIPFIRE_LFM2_EXPERT_MQ6 for higher quality), else mq4",
    source: "/home/sadara/.hipfire/src/crates/hipfire-quantize/src/main.rs:6734",
};

/// `HIPFIRE_LFM2_GRAPH` — opt-in switch. Default OFF (unset / "0") →
pub const ENV_HIPFIRE_LFM2_GRAPH: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_LFM2_GRAPH",
    description: "opt-in switch. Default OFF (unset / \"0\") →",
    source: "/home/sadara/.hipfire/src/crates/hipfire-arch-lfm2moe/src/forward.rs:57",
};

/// `HIPFIRE_LFM2_PROJ_MQ4` — flags are set, MQ6 wins (the safer of the two). 200 B/group vs
pub const ENV_HIPFIRE_LFM2_PROJ_MQ4: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_LFM2_PROJ_MQ4",
    description: "flags are set, MQ6 wins (the safer of the two). 200 B/group vs",
    source: "/home/sadara/.hipfire/src/crates/hipfire-quantize/src/main.rs:6815",
};

/// `HIPFIRE_LFM2_PROJ_MQ6` — MQ6 variant (HIPFIRE_LFM2_PROJ_MQ6=1) is the lower-quality-loss
pub const ENV_HIPFIRE_LFM2_PROJ_MQ6: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_LFM2_PROJ_MQ6",
    description: "MQ6 variant (HIPFIRE_LFM2_PROJ_MQ6=1) is the lower-quality-loss",
    source: "/home/sadara/.hipfire/src/crates/hipfire-quantize/src/main.rs:6814",
};

/// `HIPFIRE_LLOYD_FORCE_BASELINE` — Runtime variable controlling lloyd force baseline in hipfire
pub const ENV_HIPFIRE_LLOYD_FORCE_BASELINE: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_LLOYD_FORCE_BASELINE",
    description: "Runtime variable controlling lloyd force baseline in hipfire",
    source: "/home/sadara/.hipfire/src/crates/rdna-compute/src/feature_flags.rs:252",
};

/// `HIPFIRE_LLOYD_GFX12` — Used to configure runtime execution by explicitly setting "HIPFIRE_LLOYD_GFX12"
pub const ENV_HIPFIRE_LLOYD_GFX12: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_LLOYD_GFX12",
    description:
        "Used to configure runtime execution by explicitly setting \"HIPFIRE_LLOYD_GFX12\"",
    source: "/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/prefill_microbench.rs:140",
};

/// `HIPFIRE_LLOYD_K3` — HIPFIRE_LLOYD_K3=1 → ternary "MQ1.58" (3-level codebook, reuses kernel)
pub const ENV_HIPFIRE_LLOYD_K3: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_LLOYD_K3",
    description: "HIPFIRE_LLOYD_K3=1 → ternary \"MQ1.58\" (3-level codebook, reuses kernel)",
    source: "/home/sadara/.hipfire/src/crates/hipfire-quantize/src/main.rs:8102",
};

/// `HIPFIRE_LLOYD_MB4` — Force MB4=0 to skip the size-gated routing
pub const ENV_HIPFIRE_LLOYD_MB4: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_LLOYD_MB4",
    description: "Force MB4=0 to skip the size-gated routing",
    source: "/home/sadara/.hipfire/src/crates/rdna-compute/examples/test_gemm_mq4g256_lloyd_residual_wmma.rs:274",
};

/// `HIPFIRE_LM_HEAD_F16` — Runtime variable controlling lm head f16 in hipfire
pub const ENV_HIPFIRE_LM_HEAD_F16: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_LM_HEAD_F16",
    description: "Runtime variable controlling lm head f16 in hipfire",
    source: "/home/sadara/.hipfire/src/crates/hipfire-runtime/src/config.rs:100",
};

/// `HIPFIRE_LM_HEAD_OVERWRITE` — Environment toggle value controls runtime behavior
pub const ENV_HIPFIRE_LM_HEAD_OVERWRITE: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_LM_HEAD_OVERWRITE",
    description: "Environment toggle value controls runtime behavior",
    source: "/home/sadara/.hipfire/src/crates/rdna-compute/src/feature_flags.rs:172",
};

/// `HIPFIRE_LM_HEAD_WMMA` — Runtime variable controlling lm head wmma in hipfire
pub const ENV_HIPFIRE_LM_HEAD_WMMA: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_LM_HEAD_WMMA",
    description: "Runtime variable controlling lm head wmma in hipfire",
    source: "/home/sadara/.hipfire/src/crates/rdna-compute/src/feature_flags.rs:170",
};

/// `HIPFIRE_LOAD_TRANSPORT` — Selects behavior from recognized values
pub const ENV_HIPFIRE_LOAD_TRANSPORT: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_LOAD_TRANSPORT",
    description: "Selects behavior from recognized values",
    source: "/home/sadara/.hipfire/src/crates/hipfire-runtime/src/weight_pager.rs:764",
};

/// `HIPFIRE_LOCAL` — API — saves the 2-5s cold-start cost of loading the model every invocation
pub const ENV_HIPFIRE_LOCAL: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_LOCAL",
    description: "API — saves the 2-5s cold-start cost of loading the model every invocation",
    source: "/home/sadara/.hipfire/src/cli/index.ts:1738",
};

/// `HIPFIRE_MAX_RESIDENT_WORKERS` — Runtime variable controlling max resident workers in hipfire
pub const ENV_HIPFIRE_MAX_RESIDENT_WORKERS: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_MAX_RESIDENT_WORKERS",
    description: "Runtime variable controlling max resident workers in hipfire",
    source: "/home/sadara/.hipfire/src/cli/index.ts:2021",
};

/// `HIPFIRE_MAX_SEQ` — Runtime variable controlling max seq in hipfire
pub const ENV_HIPFIRE_MAX_SEQ: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_MAX_SEQ",
    description: "Runtime variable controlling max seq in hipfire",
    source: "/home/sadara/.hipfire/src/cli/index.ts:718",
};

/// `HIPFIRE_MEMSET_DUMP` — Enabled when set to 1
pub const ENV_HIPFIRE_MEMSET_DUMP: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_MEMSET_DUMP",
    description: "Enabled when set to 1",
    source: "/home/sadara/.hipfire/src/crates/hip-bridge/src/ffi.rs:1012",
};

/// `HIPFIRE_MINIMAX_CAPTURE_POSTATTN` — Runtime variable controlling minimax capture postattn in hipfire
pub const ENV_HIPFIRE_MINIMAX_CAPTURE_POSTATTN: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_MINIMAX_CAPTURE_POSTATTN",
    description: "Runtime variable controlling minimax capture postattn in hipfire",
    source: "/home/sadara/.hipfire/src/crates/hipfire-arch-minimax/src/forward.rs:200",
};

/// `HIPFIRE_MINIMAX_DOWN_FORMAT` — 2-bit error dominates the block output. HIPFIRE_MINIMAX_DOWN_FORMAT=
pub const ENV_HIPFIRE_MINIMAX_DOWN_FORMAT: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_MINIMAX_DOWN_FORMAT",
    description: "2-bit error dominates the block output. HIPFIRE_MINIMAX_DOWN_FORMAT=",
    source: "/home/sadara/.hipfire/src/crates/hipfire-quantize/src/main.rs:7049",
};

/// `HIPFIRE_MINIMAX_ENABLE_DOWN_AWQ` — down-AWQ harmful (shared s_down bad approx); opt-in
pub const ENV_HIPFIRE_MINIMAX_ENABLE_DOWN_AWQ: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_MINIMAX_ENABLE_DOWN_AWQ",
    description: "down-AWQ harmful (shared s_down bad approx); opt-in",
    source: "/home/sadara/.hipfire/src/crates/hipfire-arch-minimax/src/minimax.rs:410",
};

/// `HIPFIRE_MINIMAX_EXPERT_MQ2L` — dispatches expert dtype per-layer (experts[0].gpu_dtype), so the model
pub const ENV_HIPFIRE_MINIMAX_EXPERT_MQ2L: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_MINIMAX_EXPERT_MQ2L",
    description: "dispatches expert dtype per-layer (experts[0].gpu_dtype), so the model",
    source: "/home/sadara/.hipfire/src/crates/hipfire-quantize/src/main.rs:7028",
};

/// `HIPFIRE_MINIMAX_EXPERT_MQ3L` — dispatches expert dtype per-layer (experts[0].gpu_dtype), so the model
pub const ENV_HIPFIRE_MINIMAX_EXPERT_MQ3L: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_MINIMAX_EXPERT_MQ3L",
    description: "dispatches expert dtype per-layer (experts[0].gpu_dtype), so the model",
    source: "/home/sadara/.hipfire/src/crates/hipfire-quantize/src/main.rs:7030",
};

/// `HIPFIRE_MINIMAX_EXPERT_MQ6` — HIPFIRE_MINIMAX_EXPERT_MQ6), else mq4 (MQ4G256, default + validated)
pub const ENV_HIPFIRE_MINIMAX_EXPERT_MQ6: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_MINIMAX_EXPERT_MQ6",
    description: "HIPFIRE_MINIMAX_EXPERT_MQ6), else mq4 (MQ4G256, default + validated)",
    source: "/home/sadara/.hipfire/src/crates/hipfire-quantize/src/main.rs:7026",
};

/// `HIPFIRE_MINIMAX_GRAPH` — Default OFF — measured only +1.0% on gfx1151 (the sole arch MiniMax fits);
pub const ENV_HIPFIRE_MINIMAX_GRAPH: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_MINIMAX_GRAPH",
    description: "Default OFF — measured only +1.0% on gfx1151 (the sole arch MiniMax fits);",
    source: "/home/sadara/.hipfire/src/crates/hipfire-arch-minimax/src/forward.rs:107",
};

/// `HIPFIRE_MMQ` — Selects behavior from recognized values
pub const ENV_HIPFIRE_MMQ: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_MMQ",
    description: "Selects behavior from recognized values",
    source: "/home/sadara/.hipfire/src/crates/rdna-compute/src/feature_flags.rs:160",
};

/// `HIPFIRE_MMQ_DIAG_QUANTIZE_ONLY` — Runtime variable controlling mmq diag quantize only in hipfire
pub const ENV_HIPFIRE_MMQ_DIAG_QUANTIZE_ONLY: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_MMQ_DIAG_QUANTIZE_ONLY",
    description: "Runtime variable controlling mmq diag quantize only in hipfire",
    source: "/home/sadara/.hipfire/src/crates/rdna-compute/src/feature_flags.rs:183",
};

/// `HIPFIRE_MMQ_SCREEN` — Runtime variable controlling mmq screen in hipfire
pub const ENV_HIPFIRE_MMQ_SCREEN: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_MMQ_SCREEN",
    description: "Runtime variable controlling mmq screen in hipfire",
    source: "/home/sadara/.hipfire/src/crates/rdna-compute/src/feature_flags.rs:175",
};

/// `HIPFIRE_MMQ_SCREEN_THRESHOLD` — Runtime variable controlling mmq screen threshold in hipfire
pub const ENV_HIPFIRE_MMQ_SCREEN_THRESHOLD: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_MMQ_SCREEN_THRESHOLD",
    description: "Runtime variable controlling mmq screen threshold in hipfire",
    source: "/home/sadara/.hipfire/src/crates/rdna-compute/src/feature_flags.rs:179",
};

/// `HIPFIRE_MODEL` — Pre-warm: load default model and compile kernels before accepting requests
pub const ENV_HIPFIRE_MODEL: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_MODEL",
    description: "Pre-warm: load default model and compile kernels before accepting requests",
    source: "/home/sadara/.hipfire/src/cli/index.ts:3320",
};

/// `HIPFIRE_MOE_GROUPED_GEMM` — GEMV path. Opt out with "HIPFIRE_MOE_GROUPED_GEMM=0"
pub const ENV_HIPFIRE_MOE_GROUPED_GEMM: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_MOE_GROUPED_GEMM",
    description: "GEMV path. Opt out with \"HIPFIRE_MOE_GROUPED_GEMM=0\"",
    source: "/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/qwen35.rs:13742",
};

/// `HIPFIRE_MOE_GROUPED_I8` — Run FP16 reference (i8 disabled)
pub const ENV_HIPFIRE_MOE_GROUPED_I8: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_MOE_GROUPED_I8",
    description: "Run FP16 reference (i8 disabled)",
    source: "/home/sadara/.hipfire/src/crates/rdna-compute/src/feature_flags.rs:203",
};

/// `HIPFIRE_MOE_GROUPED_I8_4W` — Experimental gfx1151 HFQ4 grouped MoE k8 4w variant; default off after A3B/122B regression vs k8
pub const ENV_HIPFIRE_MOE_GROUPED_I8_4W: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_MOE_GROUPED_I8_4W",
    description: "Experimental gfx1151 HFQ4 grouped MoE k8 4w variant; default off after A3B/122B regression vs k8",
    source: "/home/sadara/.hipfire/src/crates/rdna-compute/src/feature_flags.rs:208",
};

/// `HIPFIRE_MOE_GROUPED_I8_K4` — Opt-in gfx1151 HFQ4 grouped MoE k4 variant; use with "HIPFIRE_MOE_GROUPED_I8_K8=0". Default off after 122B pp64 measured slower than k8
pub const ENV_HIPFIRE_MOE_GROUPED_I8_K4: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_MOE_GROUPED_I8_K4",
    description: "Opt-in gfx1151 HFQ4 grouped MoE k4 variant; use with \"HIPFIRE_MOE_GROUPED_I8_K8=0\". Default off after 122B pp64 measured slower than k8",
    source: "/home/sadara/.hipfire/src/crates/rdna-compute/src/feature_flags.rs:219",
};

/// `HIPFIRE_MOE_GROUPED_I8_K4_GFX12` — Runtime variable controlling moe grouped i8 k4 gfx12 in hipfire
pub const ENV_HIPFIRE_MOE_GROUPED_I8_K4_GFX12: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_MOE_GROUPED_I8_K4_GFX12",
    description: "Runtime variable controlling moe grouped i8 k4 gfx12 in hipfire",
    source: "/home/sadara/.hipfire/src/crates/rdna-compute/src/feature_flags.rs:220",
};

/// `HIPFIRE_MOE_GROUPED_I8_K8` — gfx1151 HFQ4 grouped MoE k8 control: default on for gfx1151, set to 0 to test k4 or base k2 variants
pub const ENV_HIPFIRE_MOE_GROUPED_I8_K8: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_MOE_GROUPED_I8_K8",
    description: "gfx1151 HFQ4 grouped MoE k8 control: default on for gfx1151, set to 0 to test k4 or base k2 variants",
    source: "/home/sadara/.hipfire/src/crates/rdna-compute/src/feature_flags.rs:214",
};

/// `HIPFIRE_MOE_GROUPED_M2` — Run m2 kernel (env set)
pub const ENV_HIPFIRE_MOE_GROUPED_M2: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_MOE_GROUPED_M2",
    description: "Run m2 kernel (env set)",
    source: "/home/sadara/.hipfire/src/crates/rdna-compute/src/feature_flags.rs:222",
};

/// `HIPFIRE_MOE_HFQ6_4W` — Default ON for gfx1151 HFQ6 grouped MoE; set to 0 to use the older v1/v2 path
pub const ENV_HIPFIRE_MOE_HFQ6_4W: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_MOE_HFQ6_4W",
    description: "Default ON for gfx1151 HFQ6 grouped MoE; set to 0 to use the older v1/v2 path",
    source: "/home/sadara/.hipfire/src/crates/rdna-compute/src/feature_flags.rs:230",
};

/// `HIPFIRE_MOE_HFQ6_V2` — Opt-in HFQ6 grouped MoE v2 path; on gfx1151 it only routes when "HIPFIRE_MOE_HFQ6_4W=0" and remains slower than the default 4w path on 122B pp64
pub const ENV_HIPFIRE_MOE_HFQ6_V2: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_MOE_HFQ6_V2",
    description: "Opt-in HFQ6 grouped MoE v2 path; on gfx1151 it only routes when \"HIPFIRE_MOE_HFQ6_4W=0\" and remains slower than the default 4w path on 122B pp64",
    source: "/home/sadara/.hipfire/src/crates/rdna-compute/src/feature_flags.rs:226",
};

/// `HIPFIRE_MOE_INDEXED_2ROW_GFX1151` — Opt-in ("1") gfx1151 two-row indexed MoE HFQ4 decode probe for gate/up and expanded down; default off after flat A3B measurements
pub const ENV_HIPFIRE_MOE_INDEXED_2ROW_GFX1151: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_MOE_INDEXED_2ROW_GFX1151",
    description: "Opt-in (\"1\") gfx1151 two-row indexed MoE HFQ4 decode probe for gate/up and expanded down; default off after flat A3B measurements",
    source: "/home/sadara/.hipfire/src/crates/rdna-compute/src/dispatch.rs:24084",
};

/// `HIPFIRE_MOE_PARO_I8` — Default-on for gfx1151 since 2026-05-21: i8 MMQ +6.3% over
pub const ENV_HIPFIRE_MOE_PARO_I8: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_MOE_PARO_I8",
    description: "Default-on for gfx1151 since 2026-05-21: i8 MMQ +6.3% over",
    source: "/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/qwen35.rs:14321",
};

/// `HIPFIRE_MOE_PARO_I8_K8` — Runtime variable controlling moe paro i8 k8 in hipfire
pub const ENV_HIPFIRE_MOE_PARO_I8_K8: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_MOE_PARO_I8_K8",
    description: "Runtime variable controlling moe paro i8 k8 in hipfire",
    source: "/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/qwen35.rs:14325",
};

/// `HIPFIRE_MQ3_MB4` — Helper: run kernel "f" with HIPFIRE_MQ3_MB4 set to "mode", return Y
pub const ENV_HIPFIRE_MQ3_MB4: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_MQ3_MB4",
    description: "Helper: run kernel \"f\" with HIPFIRE_MQ3_MB4 set to \"mode\", return Y",
    source: "/home/sadara/.hipfire/src/crates/rdna-compute/examples/test_gemm_hfq3g256_wmma.rs:92",
};

/// `HIPFIRE_MTP_DEVICE_TOKEN_CHAIN` — Default on: this path is token-identical in greedy mode and removes the
pub const ENV_HIPFIRE_MTP_DEVICE_TOKEN_CHAIN: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_MTP_DEVICE_TOKEN_CHAIN",
    description: "Default on: this path is token-identical in greedy mode and removes the",
    source: "/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/mtp_spec.rs:82",
};

/// `HIPFIRE_MTP_GPU_ACCEPT` — Default on for greedy device-token-chain MTP: candidates and verify
pub const ENV_HIPFIRE_MTP_GPU_ACCEPT: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_MTP_GPU_ACCEPT",
    description: "Default on for greedy device-token-chain MTP: candidates and verify",
    source: "/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/mtp_spec.rs:107",
};

/// `HIPFIRE_MTP_K` — dflash_spec_demo treat unset as OFF (zero overhead)
pub const ENV_HIPFIRE_MTP_K: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_MTP_K",
    description: "dflash_spec_demo treat unset as OFF (zero overhead)",
    source: "/home/sadara/.hipfire/src/crates/hipfire-runtime/src/config.rs:103",
};

/// `HIPFIRE_MTP_MODE` — explicit boolean. Only set the env var when we want it ON; daemon /
pub const ENV_HIPFIRE_MTP_MODE: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_MTP_MODE",
    description: "explicit boolean. Only set the env var when we want it ON; daemon /",
    source: "/home/sadara/.hipfire/src/crates/hipfire-runtime/src/config.rs:102",
};

/// `HIPFIRE_MTP_PROPOSAL_GRAPH` — Runtime variable controlling MTP proposal graph in hipfire
pub const ENV_HIPFIRE_MTP_PROPOSAL_GRAPH: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_MTP_PROPOSAL_GRAPH",
    description: "Runtime variable controlling MTP proposal graph in hipfire",
    source: "/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/mtp_spec.rs:160",
};

/// `HIPFIRE_MTP_Q8_VERIFY_WMMA` — Prompt-sweep parity is clean; opt out with HIPFIRE_MTP_Q8_VERIFY_WMMA=0
pub const ENV_HIPFIRE_MTP_Q8_VERIFY_WMMA: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_MTP_Q8_VERIFY_WMMA",
    description: "Prompt-sweep parity is clean; opt out with HIPFIRE_MTP_Q8_VERIFY_WMMA=0",
    source: "/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/mtp_spec.rs:131",
};

/// `HIPFIRE_MTP_SMOKE_HEAD` — Runtime variable controlling MTP smoke head in hipfire
pub const ENV_HIPFIRE_MTP_SMOKE_HEAD: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_MTP_SMOKE_HEAD",
    description: "Runtime variable controlling MTP smoke head in hipfire",
    source: "/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/examples/mtp_head_smoke.rs:28",
};

/// `HIPFIRE_MTP_SMOKE_TRUNK` — Runtime variable controlling MTP smoke trunk in hipfire
pub const ENV_HIPFIRE_MTP_SMOKE_TRUNK: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_MTP_SMOKE_TRUNK",
    description: "Runtime variable controlling MTP smoke trunk in hipfire",
    source: "/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/examples/mtp_head_smoke.rs:24",
};

/// `HIPFIRE_MTP_SNAPSHOT_OVERLAP` — Default off: current gfx1201 benches show this stream split regresses,
pub const ENV_HIPFIRE_MTP_SNAPSHOT_OVERLAP: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_MTP_SNAPSHOT_OVERLAP",
    description: "Default off: current gfx1201 benches show this stream split regresses,",
    source: "/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/mtp_spec.rs:94",
};

/// `HIPFIRE_MW16` — Runtime variable controlling mw16 in hipfire
pub const ENV_HIPFIRE_MW16: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_MW16",
    description: "Runtime variable controlling mw16 in hipfire",
    source: "/home/sadara/.hipfire/src/crates/rdna-compute/src/feature_flags.rs:237",
};

/// `HIPFIRE_NGRAM_LOOP_THRESHOLD` — Parses "HIPFIRE_NGRAM_LOOP_THRESHOLD" with fallback defaults
pub const ENV_HIPFIRE_NGRAM_LOOP_THRESHOLD: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_NGRAM_LOOP_THRESHOLD",
    description: "Parses \"HIPFIRE_NGRAM_LOOP_THRESHOLD\" with fallback defaults",
    source: "/home/sadara/.hipfire/src/crates/hipfire-runtime/src/config.rs:86",
};

/// `HIPFIRE_NGRAM_WINDOW` — Runtime variable controlling ngram window in hipfire
pub const ENV_HIPFIRE_NGRAM_WINDOW: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_NGRAM_WINDOW",
    description: "Runtime variable controlling ngram window in hipfire",
    source: "/home/sadara/.hipfire/src/crates/hipfire-runtime/src/config.rs:90",
};

/// `HIPFIRE_NORMALIZE_PROMPT` — Opt-out must skip CRLF/NBSP/trailing-ws too, not just newline collapse
pub const ENV_HIPFIRE_NORMALIZE_PROMPT: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_NORMALIZE_PROMPT",
    description: "Opt-out must skip CRLF/NBSP/trailing-ws too, not just newline collapse",
    source: "/home/sadara/.hipfire/src/crates/hipfire-runtime/src/tokenizer.rs:2206",
};

/// `HIPFIRE_NO_PID_FILE` — HIPFIRE_NO_PID_FILE=1 suppresses the write — used by "hipfire chat" when it
pub const ENV_HIPFIRE_NO_PID_FILE: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_NO_PID_FILE",
    description: "HIPFIRE_NO_PID_FILE=1 suppresses the write — used by \"hipfire chat\" when it",
    source: "/home/sadara/.hipfire/src/cli/index.ts:1914",
};

/// `HIPFIRE_PAGED_MOE_DEBUG` — Enabled when set to 1
pub const ENV_HIPFIRE_PAGED_MOE_DEBUG: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_PAGED_MOE_DEBUG",
    description: "Enabled when set to 1",
    source: "/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/qwen35.rs:6008",
};

/// `HIPFIRE_PARO_BATCHED` — Runtime variable controlling paro batched in hipfire
pub const ENV_HIPFIRE_PARO_BATCHED: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_PARO_BATCHED",
    description: "Runtime variable controlling paro batched in hipfire",
    source: "/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/qwen35.rs:12982",
};

/// `HIPFIRE_PARO_FA3_FUSED` — Lever 1 disables the env-gated fused_fa3_paro4t path so the fused rmsnorm
pub const ENV_HIPFIRE_PARO_FA3_FUSED: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_PARO_FA3_FUSED",
    description: "Lever 1 disables the env-gated fused_fa3_paro4t path so the fused rmsnorm",
    source: "/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/qwen35.rs:22950",
};

/// `HIPFIRE_PARO_FUSED_PACK2` — Runtime variable controlling paro fused pack2 in hipfire
pub const ENV_HIPFIRE_PARO_FUSED_PACK2: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_PARO_FUSED_PACK2",
    description: "Runtime variable controlling paro fused pack2 in hipfire",
    source: "/home/sadara/.hipfire/src/crates/rdna-compute/src/dispatch.rs:4881",
};

/// `HIPFIRE_PARO_FUSE_RMSNORM` — time per call. Net loss on every site. Default OFF; explicit opt-in for
pub const ENV_HIPFIRE_PARO_FUSE_RMSNORM: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_PARO_FUSE_RMSNORM",
    description: "time per call. Net loss on every site. Default OFF; explicit opt-in for",
    source: "/home/sadara/.hipfire/src/crates/hipfire-runtime/src/llama.rs:976",
};

/// `HIPFIRE_PARO_GATE_UP_FUSED` — Runtime variable controlling paro gate up fused in hipfire
pub const ENV_HIPFIRE_PARO_GATE_UP_FUSED: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_PARO_GATE_UP_FUSED",
    description: "Runtime variable controlling paro gate up fused in hipfire",
    source: "/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/qwen35.rs:22553",
};

/// `HIPFIRE_PARO_LA2_FUSED` — Runtime variable controlling paro la2 fused in hipfire
pub const ENV_HIPFIRE_PARO_LA2_FUSED: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_PARO_LA2_FUSED",
    description: "Runtime variable controlling paro la2 fused in hipfire",
    source: "/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/qwen35.rs:22664",
};

/// `HIPFIRE_PARO_LA4_FUSED` — Runtime variable controlling paro la4 fused in hipfire
pub const ENV_HIPFIRE_PARO_LA4_FUSED: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_PARO_LA4_FUSED",
    description: "Runtime variable controlling paro la4 fused in hipfire",
    source: "/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/qwen35.rs:22660",
};

/// `HIPFIRE_PARO_LA_GATES_MQ4G128` — Used to configure runtime execution by explicitly setting "HIPFIRE_PARO_LA_GATES_MQ4G128"
pub const ENV_HIPFIRE_PARO_LA_GATES_MQ4G128: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_PARO_LA_GATES_MQ4G128",
    description: "Used to configure runtime execution by explicitly setting \"HIPFIRE_PARO_LA_GATES_MQ4G128\"",
    source: "/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/paro_la_gates_codec.rs:325",
};

/// `HIPFIRE_PARO_PACK1` — Runtime variable controlling paro pack1 in hipfire
pub const ENV_HIPFIRE_PARO_PACK1: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_PARO_PACK1",
    description: "Runtime variable controlling paro pack1 in hipfire",
    source: "/home/sadara/.hipfire/src/crates/rdna-compute/src/dispatch.rs:4670",
};

/// `HIPFIRE_PARO_PACK2` — Runtime variable controlling paro pack2 in hipfire
pub const ENV_HIPFIRE_PARO_PACK2: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_PARO_PACK2",
    description: "Runtime variable controlling paro pack2 in hipfire",
    source: "/home/sadara/.hipfire/src/crates/rdna-compute/src/dispatch.rs:4673",
};

/// `HIPFIRE_PARO_PACK4` — Runtime variable controlling paro pack4 in hipfire
pub const ENV_HIPFIRE_PARO_PACK4: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_PARO_PACK4",
    description: "Runtime variable controlling paro pack4 in hipfire",
    source: "/home/sadara/.hipfire/src/crates/rdna-compute/src/dispatch.rs:4676",
};

/// `HIPFIRE_PARO_PREROTATE` — Runtime variable controlling paro prerotate in hipfire
pub const ENV_HIPFIRE_PARO_PREROTATE: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_PARO_PREROTATE",
    description: "Runtime variable controlling paro prerotate in hipfire",
    source: "/home/sadara/.hipfire/src/crates/hipfire-runtime/src/llama.rs:1544",
};

/// `HIPFIRE_PARO_SHARED_PAIRS` — Runtime variable controlling paro shared pairs in hipfire
pub const ENV_HIPFIRE_PARO_SHARED_PAIRS: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_PARO_SHARED_PAIRS",
    description: "Runtime variable controlling paro shared pairs in hipfire",
    source: "/home/sadara/.hipfire/src/crates/rdna-compute/src/dispatch.rs:4875",
};

/// `HIPFIRE_PARO_SMALL_DIRECT` — Runtime variable controlling paro small direct in hipfire
pub const ENV_HIPFIRE_PARO_SMALL_DIRECT: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_PARO_SMALL_DIRECT",
    description: "Runtime variable controlling paro small direct in hipfire",
    source: "/home/sadara/.hipfire/src/crates/hipfire-runtime/src/llama.rs:624",
};

/// `HIPFIRE_PARO_SWIGLU_FUSED` — Runtime variable controlling paro swiglu fused in hipfire
pub const ENV_HIPFIRE_PARO_SWIGLU_FUSED: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_PARO_SWIGLU_FUSED",
    description: "Runtime variable controlling paro swiglu fused in hipfire",
    source: "/home/sadara/.hipfire/src/crates/hipfire-runtime/src/llama.rs:1561",
};

/// `HIPFIRE_PFLASH_DEBUG` — Runtime variable controlling pflash debug in hipfire
pub const ENV_HIPFIRE_PFLASH_DEBUG: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_PFLASH_DEBUG",
    description: "Runtime variable controlling pflash debug in hipfire",
    source: "/home/sadara/.hipfire/src/crates/hipfire-daemon/src/main.rs:12947",
};

/// `HIPFIRE_PFLASH_DRAFTER_KV` — Read "HIPFIRE_PFLASH_DRAFTER_KV" env var; defaults to Q8 (current
pub const ENV_HIPFIRE_PFLASH_DRAFTER_KV: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_PFLASH_DRAFTER_KV",
    description: "Read \"HIPFIRE_PFLASH_DRAFTER_KV\" env var; defaults to Q8 (current",
    source: "/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/pflash.rs:533",
};

/// `HIPFIRE_PFLASH_DRAFTER_STATE` — Hybrid drafter only stores K (and V for chat-path) at
pub const ENV_HIPFIRE_PFLASH_DRAFTER_STATE: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_PFLASH_DRAFTER_STATE",
    description: "Hybrid drafter only stores K (and V for chat-path) at",
    source: "/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/pflash.rs:620",
};

/// `HIPFIRE_PFLASH_SCORE_LAYER` — Parses "HIPFIRE_PFLASH_SCORE_LAYER" with fallback defaults
pub const ENV_HIPFIRE_PFLASH_SCORE_LAYER: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_PFLASH_SCORE_LAYER",
    description: "Parses \"HIPFIRE_PFLASH_SCORE_LAYER\" with fallback defaults",
    source: "/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/pflash.rs:1081",
};

/// `HIPFIRE_PP_DFLASH` — Environment toggle value controls runtime behavior
pub const ENV_HIPFIRE_PP_DFLASH: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_PP_DFLASH",
    description: "Environment toggle value controls runtime behavior",
    source: "/home/sadara/.hipfire/src/crates/hipfire-daemon/src/main.rs:7139",
};

/// `HIPFIRE_PP_LAYERS` — Runtime variable controlling pp layers in hipfire
pub const ENV_HIPFIRE_PP_LAYERS: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_PP_LAYERS",
    description: "Runtime variable controlling pp layers in hipfire",
    source:
        "/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/pp_overlap_microbench.rs:121",
};

/// `HIPFIRE_PP_PARITY_MODEL` — Runtime variable controlling pp parity model in hipfire
pub const ENV_HIPFIRE_PP_PARITY_MODEL: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_PP_PARITY_MODEL",
    description: "Runtime variable controlling pp parity model in hipfire",
    source: "/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/tests/pp_parity.rs:197",
};

/// `HIPFIRE_PP_PFLASH` — Environment toggle value controls runtime behavior
pub const ENV_HIPFIRE_PP_PFLASH: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_PP_PFLASH",
    description: "Environment toggle value controls runtime behavior",
    source: "/home/sadara/.hipfire/src/crates/hipfire-daemon/src/main.rs:7157",
};

/// `HIPFIRE_PREFILL_ALPHA` — Runtime variable controlling prefill alpha in hipfire
pub const ENV_HIPFIRE_PREFILL_ALPHA: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_PREFILL_ALPHA",
    description: "Runtime variable controlling prefill alpha in hipfire",
    source: "/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/pflash.rs:124",
};

/// `HIPFIRE_PREFILL_BATCHED` — Enabled by default; set to 0 to disable
pub const ENV_HIPFIRE_PREFILL_BATCHED: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_PREFILL_BATCHED",
    description: "Enabled by default; set to 0 to disable",
    source: "/home/sadara/.hipfire/src/crates/hipfire-runtime/src/config.rs:82",
};

/// `HIPFIRE_PREFILL_BLOCK` — Runtime variable controlling prefill block in hipfire
pub const ENV_HIPFIRE_PREFILL_BLOCK: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_PREFILL_BLOCK",
    description: "Runtime variable controlling prefill block in hipfire",
    source: "/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/pflash.rs:144",
};

/// `HIPFIRE_PREFILL_COMPRESSION` — Runtime variable controlling prefill compression in hipfire
pub const ENV_HIPFIRE_PREFILL_COMPRESSION: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_PREFILL_COMPRESSION",
    description: "Runtime variable controlling prefill compression in hipfire",
    source: "/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/pflash.rs:104",
};

/// `HIPFIRE_PREFILL_DRAFTER` — Runtime variable controlling prefill drafter in hipfire
pub const ENV_HIPFIRE_PREFILL_DRAFTER: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_PREFILL_DRAFTER",
    description: "Runtime variable controlling prefill drafter in hipfire",
    source: "/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/pflash.rs:157",
};

/// `HIPFIRE_PREFILL_KEEP_RATIO` — Runtime variable controlling prefill keep ratio in hipfire
pub const ENV_HIPFIRE_PREFILL_KEEP_RATIO: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_PREFILL_KEEP_RATIO",
    description: "Runtime variable controlling prefill keep ratio in hipfire",
    source: "/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/pflash.rs:114",
};

/// `HIPFIRE_PREFILL_MAX_BATCH` — Runtime variable controlling prefill max batch in hipfire
pub const ENV_HIPFIRE_PREFILL_MAX_BATCH: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_PREFILL_MAX_BATCH",
    description: "Runtime variable controlling prefill max batch in hipfire",
    source: "/home/sadara/.hipfire/src/crates/hipfire-daemon/src/main.rs:6351",
};

/// `HIPFIRE_PREFILL_MAX_LAYER` — Runtime variable controlling prefill max layer in hipfire
pub const ENV_HIPFIRE_PREFILL_MAX_LAYER: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_PREFILL_MAX_LAYER",
    description: "Runtime variable controlling prefill max layer in hipfire",
    source: "/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/qwen35.rs:11996",
};

/// `HIPFIRE_PREFILL_MIN_KEEP` — Runtime variable controlling prefill min keep in hipfire
pub const ENV_HIPFIRE_PREFILL_MIN_KEEP: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_PREFILL_MIN_KEEP",
    description: "Runtime variable controlling prefill min keep in hipfire",
    source: "/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/pflash.rs:129",
};

/// `HIPFIRE_PREFILL_PROFILE` — Enabled when set to 1
pub const ENV_HIPFIRE_PREFILL_PROFILE: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_PREFILL_PROFILE",
    description: "Enabled when set to 1",
    source: "/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/pflash.rs:154",
};

/// `HIPFIRE_PREFILL_RECENT` — Runtime variable controlling prefill recent in hipfire
pub const ENV_HIPFIRE_PREFILL_RECENT: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_PREFILL_RECENT",
    description: "Runtime variable controlling prefill recent in hipfire",
    source: "/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/pflash.rs:139",
};

/// `HIPFIRE_PREFILL_REUSE_PBS` — Used to configure runtime execution by explicitly setting "HIPFIRE_PREFILL_REUSE_PBS"
pub const ENV_HIPFIRE_PREFILL_REUSE_PBS: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_PREFILL_REUSE_PBS",
    description:
        "Used to configure runtime execution by explicitly setting \"HIPFIRE_PREFILL_REUSE_PBS\"",
    source: "/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/prefill_microbench.rs:121",
};

/// `HIPFIRE_PREFILL_SINK` — Runtime variable controlling prefill sink in hipfire
pub const ENV_HIPFIRE_PREFILL_SINK: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_PREFILL_SINK",
    description: "Runtime variable controlling prefill sink in hipfire",
    source: "/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/pflash.rs:134",
};

/// `HIPFIRE_PREFILL_SPARSE_THRESHOLD` — Runtime variable controlling prefill sparse threshold in hipfire
pub const ENV_HIPFIRE_PREFILL_SPARSE_THRESHOLD: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_PREFILL_SPARSE_THRESHOLD",
    description: "Runtime variable controlling prefill sparse threshold in hipfire",
    source: "/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/pflash.rs:149",
};

/// `HIPFIRE_PREFILL_STOP_AFTER_LA_LAYER` — Parses "HIPFIRE_PREFILL_STOP_AFTER_LA_LAYER" with fallback defaults
pub const ENV_HIPFIRE_PREFILL_STOP_AFTER_LA_LAYER: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_PREFILL_STOP_AFTER_LA_LAYER",
    description: "Parses \"HIPFIRE_PREFILL_STOP_AFTER_LA_LAYER\" with fallback defaults",
    source: "/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/qwen35.rs:14580",
};

/// `HIPFIRE_PREFILL_STOP_STAGE` — Runtime variable controlling prefill stop stage in hipfire
pub const ENV_HIPFIRE_PREFILL_STOP_STAGE: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_PREFILL_STOP_STAGE",
    description: "Runtime variable controlling prefill stop stage in hipfire",
    source: "/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/qwen35.rs:14586",
};

/// `HIPFIRE_PREFILL_STOP_STAGE_LAYER` — Parses "HIPFIRE_PREFILL_STOP_STAGE_LAYER" with fallback defaults
pub const ENV_HIPFIRE_PREFILL_STOP_STAGE_LAYER: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_PREFILL_STOP_STAGE_LAYER",
    description: "Parses \"HIPFIRE_PREFILL_STOP_STAGE_LAYER\" with fallback defaults",
    source: "/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/qwen35.rs:14583",
};

/// `HIPFIRE_PREFILL_THRESHOLD` — Runtime variable controlling prefill threshold in hipfire
pub const ENV_HIPFIRE_PREFILL_THRESHOLD: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_PREFILL_THRESHOLD",
    description: "Runtime variable controlling prefill threshold in hipfire",
    source: "/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/pflash.rs:109",
};

/// `HIPFIRE_PREFIX_BOUNDARY_CHECKPOINTS` — Interprets "HIPFIRE_PREFIX_BOUNDARY_CHECKPOINTS" from environment to select behavior
pub const ENV_HIPFIRE_PREFIX_BOUNDARY_CHECKPOINTS: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_PREFIX_BOUNDARY_CHECKPOINTS",
    description:
        "Interprets \"HIPFIRE_PREFIX_BOUNDARY_CHECKPOINTS\" from environment to select behavior",
    source: "/home/sadara/.hipfire/src/crates/hipfire-daemon/src/main.rs:4829",
};

/// `HIPFIRE_PROFILE` — HIPFIRE_PROFILE=1 + HIPFIRE_PROFILE_CYCLES=N: per-kernel profiling
pub const ENV_HIPFIRE_PROFILE: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_PROFILE",
    description: "HIPFIRE_PROFILE=1 + HIPFIRE_PROFILE_CYCLES=N: per-kernel profiling",
    source: "/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/mtp_only_demo.rs:495",
};

/// `HIPFIRE_PROFILE_CYCLES` — Runtime variable controlling profile cycles in hipfire
pub const ENV_HIPFIRE_PROFILE_CYCLES: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_PROFILE_CYCLES",
    description: "Runtime variable controlling profile cycles in hipfire",
    source: "/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/mtp_only_demo.rs:496",
};

/// `HIPFIRE_PROFILE_DECODE` — HIPFIRE_PROFILE_DECODE=1 wraps the timed gen loop in the per-kernel
pub const ENV_HIPFIRE_PROFILE_DECODE: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_PROFILE_DECODE",
    description: "HIPFIRE_PROFILE_DECODE=1 wraps the timed gen loop in the per-kernel",
    source: "/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/bench_qwen35_speed.rs:683",
};

/// `HIPFIRE_PROMPT_HEAT_JSON` — Enabled when set to 1
pub const ENV_HIPFIRE_PROMPT_HEAT_JSON: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_PROMPT_HEAT_JSON",
    description: "Enabled when set to 1",
    source: "/home/sadara/.hipfire/src/crates/hipfire-runtime/src/config.rs:58",
};

/// `HIPFIRE_PROMPT_HEAT_LIMIT` — Runtime variable controlling prompt heat limit in hipfire
pub const ENV_HIPFIRE_PROMPT_HEAT_LIMIT: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_PROMPT_HEAT_LIMIT",
    description: "Runtime variable controlling prompt heat limit in hipfire",
    source: "/home/sadara/.hipfire/src/crates/hipfire-runtime/src/config.rs:59",
};

/// `HIPFIRE_PROMPT_TOKEN_HEAT` — Enabled when set to 1
pub const ENV_HIPFIRE_PROMPT_TOKEN_HEAT: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_PROMPT_TOKEN_HEAT",
    description: "Enabled when set to 1",
    source: "/home/sadara/.hipfire/src/crates/hipfire-runtime/src/config.rs:56",
};

/// `HIPFIRE_Q8_BATCHED_LEGACY` — Environment toggle value controls runtime behavior
pub const ENV_HIPFIRE_Q8_BATCHED_LEGACY: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_Q8_BATCHED_LEGACY",
    description: "Environment toggle value controls runtime behavior",
    source: "/home/sadara/.hipfire/src/crates/rdna-compute/src/feature_flags.rs:238",
};

/// `HIPFIRE_Q8_DP4A` — (A2) dp4a wave32 (gfx906 v_dot4_i32_i8 Q·K)
pub const ENV_HIPFIRE_Q8_DP4A: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_Q8_DP4A",
    description: "(A2) dp4a wave32 (gfx906 v_dot4_i32_i8 Q·K)",
    source:
        "/home/sadara/.hipfire/src/crates/rdna-compute/examples/q8_batched_attn_microbench.rs:185",
};

/// `HIPFIRE_Q8_DP4A_W64` — (A3) dp4a WAVE64 (full 64-lane utilization)
pub const ENV_HIPFIRE_Q8_DP4A_W64: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_Q8_DP4A_W64",
    description: "(A3) dp4a WAVE64 (full 64-lane utilization)",
    source:
        "/home/sadara/.hipfire/src/crates/rdna-compute/examples/q8_batched_attn_microbench.rs:196",
};

/// `HIPFIRE_Q8_GATE_UP_4W` — gfx1151 Q8 gate+up 4w control: unset auto-routes aligned N>=128, "0" forces single-wave WMMA, "1" forces 4w for aligned N%64==0 experiments
pub const ENV_HIPFIRE_Q8_GATE_UP_4W: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_Q8_GATE_UP_4W",
    description: "gfx1151 Q8 gate+up 4w control: unset auto-routes aligned N>=128, \"0\" forces single-wave WMMA, \"1\" forces 4w for aligned N%64==0 experiments",
    source: "/home/sadara/.hipfire/src/crates/rdna-compute/examples/test_gemm_q8_gate_up_wmma.rs:130",
};

/// `HIPFIRE_Q8_GATE_UP_BENCH` — Enabled when set to 1
pub const ENV_HIPFIRE_Q8_GATE_UP_BENCH: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_Q8_GATE_UP_BENCH",
    description: "Enabled when set to 1",
    source:
        "/home/sadara/.hipfire/src/crates/rdna-compute/examples/test_gemm_q8_gate_up_wmma.rs:129",
};

/// `HIPFIRE_Q8_TOKPAR` — (A) NEW token-parallel
pub const ENV_HIPFIRE_Q8_TOKPAR: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_Q8_TOKPAR",
    description: "(A) NEW token-parallel",
    source:
        "/home/sadara/.hipfire/src/crates/rdna-compute/examples/q8_batched_attn_microbench.rs:174",
};

/// `HIPFIRE_Q8_WMMA_4W` — originally shipped default-ON on gfx11/gfx12, but the auto-enable
pub const ENV_HIPFIRE_Q8_WMMA_4W: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_Q8_WMMA_4W",
    description: "originally shipped default-ON on gfx11/gfx12, but the auto-enable",
    source: "/home/sadara/.hipfire/src/crates/rdna-compute/src/dispatch.rs:28519",
};

/// `HIPFIRE_Q8_WMMA_X64` — Environment toggle value controls runtime behavior
pub const ENV_HIPFIRE_Q8_WMMA_X64: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_Q8_WMMA_X64",
    description: "Environment toggle value controls runtime behavior",
    source: "/home/sadara/.hipfire/src/crates/rdna-compute/src/dispatch.rs:28330",
};

/// `HIPFIRE_QA_KV_MODES` — Defaults to q8,asym4,asym3,asym2 when unset
pub const ENV_HIPFIRE_QA_KV_MODES: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_QA_KV_MODES",
    description: "Defaults to q8,asym4,asym3,asym2 when unset",
    source: "/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/test_inferenceQA.rs:761",
};

/// `HIPFIRE_QUANT_DIAG_PATH` — Runtime variable controlling quant diag path in hipfire
pub const ENV_HIPFIRE_QUANT_DIAG_PATH: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_QUANT_DIAG_PATH",
    description: "Runtime variable controlling quant diag path in hipfire",
    source: "/home/sadara/.hipfire/src/crates/hipfire-quantize/src/main.rs:10397",
};

/// `HIPFIRE_QUANT_THREADS` — Runtime variable controlling quant threads in hipfire
pub const ENV_HIPFIRE_QUANT_THREADS: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_QUANT_THREADS",
    description: "Runtime variable controlling quant threads in hipfire",
    source: "/home/sadara/.hipfire/src/crates/hipfire-quantize/src/main.rs:5463",
};

/// `HIPFIRE_QWEN35_DECODE_BATCH` — Defaults to auto when unset
pub const ENV_HIPFIRE_QWEN35_DECODE_BATCH: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_QWEN35_DECODE_BATCH",
    description: "Defaults to auto when unset",
    source: "/home/sadara/.hipfire/src/crates/hipfire-daemon/src/main.rs:5964",
};

/// `HIPFIRE_QWEN35_DECODE_BATCH_MAX` — Runtime variable controlling qwen35 decode batch max in hipfire
pub const ENV_HIPFIRE_QWEN35_DECODE_BATCH_MAX: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_QWEN35_DECODE_BATCH_MAX",
    description: "Runtime variable controlling qwen35 decode batch max in hipfire",
    source: "/home/sadara/.hipfire/src/crates/hipfire-daemon/src/main.rs:6061",
};

/// `HIPFIRE_DECODE_BATCH_GROUPED_PARITY_CHUNK_SIZE` — Smoke-only override for grouped-MoE decode parity matrix chunk size
pub const ENV_HIPFIRE_DECODE_BATCH_GROUPED_PARITY_CHUNK_SIZE: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DECODE_BATCH_GROUPED_PARITY_CHUNK_SIZE",
    description: "Smoke-only override for grouped-MoE decode parity matrix chunk size",
    source: "/home/sadara/.hipfire/src/tests/smoke-server-decode-batch.sh:17",
};

/// `HIPFIRE_QWEN35_DECODE_INTERNAL_PARITY` — Interprets "HIPFIRE_QWEN35_DECODE_INTERNAL_PARITY" from environment to select behavior
pub const ENV_HIPFIRE_QWEN35_DECODE_INTERNAL_PARITY: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_QWEN35_DECODE_INTERNAL_PARITY",
    description:
        "Interprets \"HIPFIRE_QWEN35_DECODE_INTERNAL_PARITY\" from environment to select behavior",
    source: "/home/sadara/.hipfire/src/crates/hipfire-daemon/src/main.rs:6080",
};

/// `HIPFIRE_QWEN35_DECODE_NATIVE_MULTIROW` — Interprets "HIPFIRE_QWEN35_DECODE_NATIVE_MULTIROW" from environment to select behavior
pub const ENV_HIPFIRE_QWEN35_DECODE_NATIVE_MULTIROW: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_QWEN35_DECODE_NATIVE_MULTIROW",
    description:
        "Interprets \"HIPFIRE_QWEN35_DECODE_NATIVE_MULTIROW\" from environment to select behavior",
    source: "/home/sadara/.hipfire/src/crates/hipfire-daemon/src/main.rs:6071",
};

/// `HIPFIRE_QWEN35_EXPERT_CACHE_MB` — Parses "HIPFIRE_QWEN35_EXPERT_CACHE_MB" with fallback defaults
pub const ENV_HIPFIRE_QWEN35_EXPERT_CACHE_MB: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_QWEN35_EXPERT_CACHE_MB",
    description: "Parses \"HIPFIRE_QWEN35_EXPERT_CACHE_MB\" with fallback defaults",
    source: "/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/qwen35.rs:686",
};

/// `HIPFIRE_QWEN35_EXPERT_CACHE_TRACE` — Interprets "HIPFIRE_QWEN35_EXPERT_CACHE_TRACE" from environment to select behavior
pub const ENV_HIPFIRE_QWEN35_EXPERT_CACHE_TRACE: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_QWEN35_EXPERT_CACHE_TRACE",
    description:
        "Interprets \"HIPFIRE_QWEN35_EXPERT_CACHE_TRACE\" from environment to select behavior",
    source: "/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/qwen35.rs:3782",
};

/// `HIPFIRE_QWEN35_FINITE_TRACE` — Runtime variable controlling qwen35 finite trace in hipfire
pub const ENV_HIPFIRE_QWEN35_FINITE_TRACE: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_QWEN35_FINITE_TRACE",
    description: "Runtime variable controlling qwen35 finite trace in hipfire",
    source: "/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/qwen35.rs:12558",
};

/// `HIPFIRE_QWEN35_PAGED_EXPERTS` — Interprets "HIPFIRE_QWEN35_PAGED_EXPERTS" from environment to select behavior
pub const ENV_HIPFIRE_QWEN35_PAGED_EXPERTS: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_QWEN35_PAGED_EXPERTS",
    description: "Interprets \"HIPFIRE_QWEN35_PAGED_EXPERTS\" from environment to select behavior",
    source: "/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/qwen35.rs:3226",
};

/// `HIPFIRE_QWEN35_PREFILL_SESSION_BATCH` — Runtime variable controlling qwen35 prefill session batch in hipfire
pub const ENV_HIPFIRE_QWEN35_PREFILL_SESSION_BATCH: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_QWEN35_PREFILL_SESSION_BATCH",
    description: "Runtime variable controlling qwen35 prefill session batch in hipfire",
    source: "/home/sadara/.hipfire/src/crates/hipfire-daemon/src/main.rs:5709",
};

/// `HIPFIRE_QWEN35_ROUTED_ONLY_MOE_FORWARD` — Runtime variable controlling qwen35 routed only moe forward in hipfire
pub const ENV_HIPFIRE_QWEN35_ROUTED_ONLY_MOE_FORWARD: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_QWEN35_ROUTED_ONLY_MOE_FORWARD",
    description: "Runtime variable controlling qwen35 routed only moe forward in hipfire",
    source: "/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/qwen35.rs:6018",
};

/// `HIPFIRE_QWEN35_STAGE_SYNC` — Runtime variable controlling qwen35 stage sync in hipfire
pub const ENV_HIPFIRE_QWEN35_STAGE_SYNC: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_QWEN35_STAGE_SYNC",
    description: "Runtime variable controlling qwen35 stage sync in hipfire",
    source: "/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/qwen35.rs:12592",
};

/// `HIPFIRE_QWEN35_STAGE_TRACE` — Runtime variable controlling qwen35 stage trace in hipfire
pub const ENV_HIPFIRE_QWEN35_STAGE_TRACE: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_QWEN35_STAGE_TRACE",
    description: "Runtime variable controlling qwen35 stage trace in hipfire",
    source: "/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/qwen35.rs:12586",
};

/// `HIPFIRE_QWEN35_STATE_QUANT` — Runtime variable controlling qwen35 state quant in hipfire
pub const ENV_HIPFIRE_QWEN35_STATE_QUANT: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_QWEN35_STATE_QUANT",
    description: "Runtime variable controlling qwen35 state quant in hipfire",
    source: "/home/sadara/.hipfire/src/cli/index.ts:746",
};

/// `HIPFIRE_RDNA2_VARIANT` — Runtime variable controlling rdna2 variant in hipfire
pub const ENV_HIPFIRE_RDNA2_VARIANT: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_RDNA2_VARIANT",
    description: "Runtime variable controlling rdna2 variant in hipfire",
    source: "/home/sadara/.hipfire/src/crates/rdna-compute/src/feature_flags.rs:256",
};

/// `HIPFIRE_REPLAY_GRAPH` — Enabled when set to 1
pub const ENV_HIPFIRE_REPLAY_GRAPH: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_REPLAY_GRAPH",
    description: "Enabled when set to 1",
    source: "/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/speculative.rs:1335",
};

/// `HIPFIRE_RESPONSES_STATE_MAX` — Runtime variable controlling responses state max in hipfire
pub const ENV_HIPFIRE_RESPONSES_STATE_MAX: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_RESPONSES_STATE_MAX",
    description: "Runtime variable controlling responses state max in hipfire",
    source: "/home/sadara/.hipfire/src/cli/index.ts:2845",
};

/// `HIPFIRE_ROCBLAS_ALL_ARCHS` — Runtime variable controlling rocblas all archs in hipfire
pub const ENV_HIPFIRE_ROCBLAS_ALL_ARCHS: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_ROCBLAS_ALL_ARCHS",
    description: "Runtime variable controlling rocblas all archs in hipfire",
    source: "/home/sadara/.hipfire/src/crates/rdna-compute/src/feature_flags.rs:246",
};

/// `HIPFIRE_ROCBLAS_OFF` — Enabled when set to 1
pub const ENV_HIPFIRE_ROCBLAS_OFF: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_ROCBLAS_OFF",
    description: "Enabled when set to 1",
    source: "/home/sadara/.hipfire/src/crates/rdna-compute/src/feature_flags.rs:248",
};

/// `HIPFIRE_ROCPROF_BIN` — Runtime variable controlling rocprof bin in hipfire
pub const ENV_HIPFIRE_ROCPROF_BIN: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_ROCPROF_BIN",
    description: "Runtime variable controlling rocprof bin in hipfire",
    source: "/home/sadara/.hipfire/src/crates/hipfire-runtime/src/eval_harness/mod.rs:1599",
};

/// `HIPFIRE_ROCPROF_CSV` — Runtime variable controlling rocprof csv in hipfire
pub const ENV_HIPFIRE_ROCPROF_CSV: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_ROCPROF_CSV",
    description: "Runtime variable controlling rocprof csv in hipfire",
    source: "/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/bench_qwen35_speed.rs:447",
};

/// `HIPFIRE_ROPE_INTERLEAVED_LEGACY` — Runtime variable controlling rope interleaved legacy in hipfire
pub const ENV_HIPFIRE_ROPE_INTERLEAVED_LEGACY: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_ROPE_INTERLEAVED_LEGACY",
    description: "Runtime variable controlling rope interleaved legacy in hipfire",
    source: "/home/sadara/.hipfire/src/crates/rdna-compute/src/feature_flags.rs:239",
};

/// `HIPFIRE_RUN_EXAMPLE_BIN` — Runtime variable controlling run example bin in hipfire
pub const ENV_HIPFIRE_RUN_EXAMPLE_BIN: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_RUN_EXAMPLE_BIN",
    description: "Runtime variable controlling run example bin in hipfire",
    source: "/home/sadara/.hipfire/src/crates/hipfire-runtime/src/eval_harness/mod.rs:7939",
};

/// `HIPFIRE_SAMPLE_COMPARE` — Enabled when set to 1
pub const ENV_HIPFIRE_SAMPLE_COMPARE: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_SAMPLE_COMPARE",
    description: "Enabled when set to 1",
    source: "/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/infer_qwen35.rs:222",
};

/// `HIPFIRE_SERVER_PREFILL_SHARED_PREFIX_FANOUT` — Runtime variable controlling server prefill shared prefix fanout in hipfire
pub const ENV_HIPFIRE_SERVER_PREFILL_SHARED_PREFIX_FANOUT: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_SERVER_PREFILL_SHARED_PREFIX_FANOUT",
    description: "Runtime variable controlling server prefill shared prefix fanout in hipfire",
    source: "/home/sadara/.hipfire/src/cli/index.ts:5121",
};

/// `HIPFIRE_SERVER_RESIDENT_STATE_BUDGET_MB` — Runtime variable controlling server resident state budget mb in hipfire
pub const ENV_HIPFIRE_SERVER_RESIDENT_STATE_BUDGET_MB: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_SERVER_RESIDENT_STATE_BUDGET_MB",
    description: "Runtime variable controlling server resident state budget mb in hipfire",
    source: "/home/sadara/.hipfire/src/crates/hipfire-daemon/src/main.rs:3999",
};

/// `HIPFIRE_SMOKE_KV` — Select KV cache quant via HIPFIRE_SMOKE_KV (default q8, matches the
pub const ENV_HIPFIRE_SMOKE_KV: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_SMOKE_KV",
    description: "Select KV cache quant via HIPFIRE_SMOKE_KV (default q8, matches the",
    source: "/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/a3b_smoke_forward.rs:109",
};

/// `HIPFIRE_SMOKE_KV_SEQ` — production CLI default). asym3/asym4 engage the Givens-rotated 3/4-bit
pub const ENV_HIPFIRE_SMOKE_KV_SEQ: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_SMOKE_KV_SEQ",
    description: "production CLI default). asym3/asym4 engage the Givens-rotated 3/4-bit",
    source: "/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/a3b_smoke_forward.rs:102",
};

/// `HIPFIRE_SMOKE_MODE` — raw (default for back-compat): tokenize "Hello" and decode from pos=0
pub const ENV_HIPFIRE_SMOKE_MODE: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_SMOKE_MODE",
    description: "raw (default for back-compat): tokenize \"Hello\" and decode from pos=0",
    source: "/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/a3b_smoke_forward.rs:157",
};

/// `HIPFIRE_SMOKE_PROMPT` — Defaults to Hello when unset
pub const ENV_HIPFIRE_SMOKE_PROMPT: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_SMOKE_PROMPT",
    description: "Defaults to Hello when unset",
    source: "/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/a3b_smoke_forward.rs:158",
};

/// `HIPFIRE_SMOKE_STEPS` — Runtime variable controlling smoke steps in hipfire
pub const ENV_HIPFIRE_SMOKE_STEPS: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_SMOKE_STEPS",
    description: "Runtime variable controlling smoke steps in hipfire",
    source: "/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/a3b_smoke_forward.rs:38",
};

/// `HIPFIRE_SPEC_PHASES` — Enabled when set to 1
pub const ENV_HIPFIRE_SPEC_PHASES: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_SPEC_PHASES",
    description: "Enabled when set to 1",
    source: "/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/speculative.rs:3251",
};

/// `HIPFIRE_STATE` — Interprets "HIPFIRE_STATE" from environment to select behavior
pub const ENV_HIPFIRE_STATE: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_STATE",
    description: "Interprets \"HIPFIRE_STATE\" from environment to select behavior",
    source: "/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/greedy_dump_top5.rs:245",
};

/// `HIPFIRE_STATE_QUANT` — Runtime variable controlling state quant in hipfire
pub const ENV_HIPFIRE_STATE_QUANT: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_STATE_QUANT",
    description: "Runtime variable controlling state quant in hipfire",
    source: "/home/sadara/.hipfire/src/cli/index.ts:746",
};

/// `HIPFIRE_TARGET_ARCH` — HIPFIRE_TARGET_ARCH overrides the detected GPU arch for kernel
pub const ENV_HIPFIRE_TARGET_ARCH: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_TARGET_ARCH",
    description: "HIPFIRE_TARGET_ARCH overrides the detected GPU arch for kernel",
    source: "/home/sadara/.hipfire/src/crates/rdna-compute/src/dispatch.rs:727",
};

/// `HIPFIRE_TIER_RATIO` — Runtime variable controlling tier ratio in hipfire
pub const ENV_HIPFIRE_TIER_RATIO: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_TIER_RATIO",
    description: "Runtime variable controlling tier ratio in hipfire",
    source: "/home/sadara/.hipfire/src/crates/hipfire-quantize/src/main.rs:5738",
};

/// `HIPFIRE_UNIFORM_VRAM_TOLERANCE_GB` — Runtime variable controlling uniform vram tolerance gb in hipfire
pub const ENV_HIPFIRE_UNIFORM_VRAM_TOLERANCE_GB: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_UNIFORM_VRAM_TOLERANCE_GB",
    description: "Runtime variable controlling uniform vram tolerance gb in hipfire",
    source: "/home/sadara/.hipfire/src/crates/hipfire-runtime/src/config.rs:97",
};

/// `HIPFIRE_VERIFY_GRAPH` — Enabled by default; set to 0 to disable
pub const ENV_HIPFIRE_VERIFY_GRAPH: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_VERIFY_GRAPH",
    description: "Enabled by default; set to 0 to disable",
    source: "/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/speculative.rs:2717",
};

/// `HIPFIRE_VERIFY_GRAPH_TIMING` — (HIPFIRE_VERIFY_GRAPH_TIMING=1). Two device-sync points bracket the
pub const ENV_HIPFIRE_VERIFY_GRAPH_TIMING: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_VERIFY_GRAPH_TIMING",
    description: "(HIPFIRE_VERIFY_GRAPH_TIMING=1). Two device-sync points bracket the",
    source: "/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/speculative.rs:2730",
};

/// `HIPFIRE_VERIFY_GRAPH_TREE` — Enabled when set to 1
pub const ENV_HIPFIRE_VERIFY_GRAPH_TREE: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_VERIFY_GRAPH_TREE",
    description: "Enabled when set to 1",
    source: "/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/speculative.rs:2707",
};

/// `HIPFIRE_VL_DUMP_DIR` — little-endian f32 blobs + JSON sidecars to $HIPFIRE_VL_DUMP_DIR
pub const ENV_HIPFIRE_VL_DUMP_DIR: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_VL_DUMP_DIR",
    description: "little-endian f32 blobs + JSON sidecars to $HIPFIRE_VL_DUMP_DIR",
    source: "/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/infer.rs:148",
};

/// `HIPFIRE_WO_MMQ` — Enabled when set to 1
pub const ENV_HIPFIRE_WO_MMQ: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_WO_MMQ",
    description: "Enabled when set to 1",
    source: "/home/sadara/.hipfire/src/crates/rdna-compute/src/feature_flags.rs:169",
};

/// `HIPFIRE_WO_WMMA_VARIANT` — Runtime variable controlling wo wmma variant in hipfire
pub const ENV_HIPFIRE_WO_WMMA_VARIANT: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_WO_WMMA_VARIANT",
    description: "Runtime variable controlling wo wmma variant in hipfire",
    source: "/home/sadara/.hipfire/src/crates/rdna-compute/src/feature_flags.rs:243",
};

/// `HIP_PATH` — fails with "file not found". Add well-known candidates as -I flags;
pub const ENV_HIP_PATH: EnvVarDoc = EnvVarDoc {
    name: "HIP_PATH",
    description: "fails with \"file not found\". Add well-known candidates as -I flags;",
    source: "/home/sadara/.hipfire/src/crates/rdna-compute/src/compiler.rs:367",
};

/// `HOME` — Runtime variable controlling home in hipfire
pub const ENV_HOME: EnvVarDoc = EnvVarDoc {
    name: "HOME",
    description: "Runtime variable controlling home in hipfire",
    source: "/home/sadara/.hipfire/src/crates/rdna-compute/src/compiler.rs:129",
};

/// `HOSTNAME` — Runtime variable controlling hostname in hipfire
pub const ENV_HOSTNAME: EnvVarDoc = EnvVarDoc {
    name: "HOSTNAME",
    description: "Runtime variable controlling hostname in hipfire",
    source: "/home/sadara/.hipfire/src/crates/hipfire-runtime/src/coherence_runtime.rs:452",
};

/// `MAX_TOKENS` — Parses "MAX_TOKENS" with fallback defaults
pub const ENV_MAX_TOKENS: EnvVarDoc = EnvVarDoc {
    name: "MAX_TOKENS",
    description: "Parses \"MAX_TOKENS\" with fallback defaults",
    source: "/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/greedy_dump.rs:112",
};

/// `MMQ_TEST_MODE` — Defaults to residual when unset
pub const ENV_MMQ_TEST_MODE: EnvVarDoc = EnvVarDoc {
    name: "MMQ_TEST_MODE",
    description: "Defaults to residual when unset",
    source:
        "/home/sadara/.hipfire/src/crates/rdna-compute/examples/test_gfx906_mmq_correctness.rs:65",
};

/// `NO_COLOR` — disables when stdout/stderr aren't TTYs (already gated above for stdin
pub const ENV_NO_COLOR: EnvVarDoc = EnvVarDoc {
    name: "NO_COLOR",
    description: "disables when stdout/stderr aren't TTYs (already gated above for stdin",
    source: "/home/sadara/.hipfire/src/cli/chat.ts:100",
};

/// `NO_NGRAM` — Disabled for perf measurement — re-enable after implementing GPU n-gram kernel
pub const ENV_NO_NGRAM: EnvVarDoc = EnvVarDoc {
    name: "NO_NGRAM",
    description: "Disabled for perf measurement — re-enable after implementing GPU n-gram kernel",
    source: "/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/infer_vl.rs:350",
};

/// `PATH` — well-known locations, augment process.env.PATH with any found dirs,
pub const ENV_PATH: EnvVarDoc = EnvVarDoc {
    name: "PATH",
    description: "well-known locations, augment process.env.PATH with any found dirs,",
    source: "/home/sadara/.hipfire/src/crates/hipfire-runtime/src/eval_harness/mod.rs:1609",
};

/// `PROMPT_MODE` — Defaults to thinking when unset
pub const ENV_PROMPT_MODE: EnvVarDoc = EnvVarDoc {
    name: "PROMPT_MODE",
    description: "Defaults to thinking when unset",
    source: "/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/greedy_dump_top5.rs:91",
};

/// `QWEN35_TEST_MODEL` — Runtime variable controlling qwen35 test model in hipfire
pub const ENV_QWEN35_TEST_MODEL: EnvVarDoc = EnvVarDoc {
    name: "QWEN35_TEST_MODEL",
    description: "Runtime variable controlling qwen35 test model in hipfire",
    source: "/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/test_qwen35_loadQA.rs:22",
};

/// `TINYLLAMA_GGUF` — Runtime variable controlling tinyllama gguf in hipfire
pub const ENV_TINYLLAMA_GGUF: EnvVarDoc = EnvVarDoc {
    name: "TINYLLAMA_GGUF",
    description: "Runtime variable controlling tinyllama gguf in hipfire",
    source: "/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/test_q4f16QA.rs:38",
};

/// `TRIALS` — Runtime variable controlling trials in hipfire
pub const ENV_TRIALS: EnvVarDoc = EnvVarDoc {
    name: "TRIALS",
    description: "Runtime variable controlling trials in hipfire",
    source:
        "/home/sadara/.hipfire/src/crates/rdna-compute/examples/bench_gfx1151_hfq4_s4_mmq.rs:151",
};

/// `USERPROFILE` — Runtime variable controlling userprofile in hipfire
pub const ENV_USERPROFILE: EnvVarDoc = EnvVarDoc {
    name: "USERPROFILE",
    description: "Runtime variable controlling userprofile in hipfire",
    source: "/home/sadara/.hipfire/src/crates/hipfire-daemon/src/main.rs:648",
};

/// `USE_SAMPLE` — Enabled when set to 1
pub const ENV_USE_SAMPLE: EnvVarDoc = EnvVarDoc {
    name: "USE_SAMPLE",
    description: "Enabled when set to 1",
    source:
        "/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/a3b_multiturn_oneshot.rs:117",
};

/// `VERIFY` — VERIFY=1: run the token-parallel path vs the tile_batched fallback on
pub const ENV_VERIFY: EnvVarDoc = EnvVarDoc {
    name: "VERIFY",
    description: "VERIFY=1: run the token-parallel path vs the tile_batched fallback on",
    source:
        "/home/sadara/.hipfire/src/crates/rdna-compute/examples/q8_batched_attn_microbench.rs:92",
};

/// `W64` — Force the candidate into out_tp: W64=1 → dp4a wave64, DP4A=1 →
pub const ENV_W64: EnvVarDoc = EnvVarDoc {
    name: "W64",
    description: "Force the candidate into out_tp: W64=1 → dp4a wave64, DP4A=1 →",
    source:
        "/home/sadara/.hipfire/src/crates/rdna-compute/examples/q8_batched_attn_microbench.rs:105",
};

/// All documented environment variables in deterministic order.
pub const ALL_ENV_VARS: &[EnvVarDoc] = &[
    ENV_BENCH_BATCH,
    ENV_BENCH_DRAFT_K,
    ENV_BENCH_DRAFT_LAYERS,
    ENV_BENCH_DRAFT_M,
    ENV_BENCH_DRAFT_N,
    ENV_BENCH_K,
    ENV_BENCH_M,
    ENV_BENCH_VERIFY_LAYERS,
    ENV_CLICOLOR,
    ENV_DDTREE_TIMING,
    ENV_DEBUG_LAYERS,
    ENV_DFLASH_LIVE_TAU,
    ENV_DP4A,
    ENV_FP32_STATE,
    ENV_HFQ_TEST_N_ITER,
    ENV_HFQ_TEST_SCALE_LOG10,
    ENV_HFQ_TEST_ZP_MAX,
    ENV_HF_TOKEN,
    ENV_HIPFIRE_ADAPTIVE_B_DOWN,
    ENV_HIPFIRE_ADAPTIVE_B_UNSAFE,
    ENV_HIPFIRE_ADAPTIVE_B_UP,
    ENV_HIPFIRE_ALLOW_MIXED_ARCH,
    ENV_HIPFIRE_ALLOW_MQ2,
    ENV_HIPFIRE_ALLOW_MQ2_LLOYD,
    ENV_HIPFIRE_ALLOW_MQ3_LLOYD,
    ENV_HIPFIRE_ALLOW_MQ4_LLOYD,
    ENV_HIPFIRE_ALLOW_UNIT_IMATRIX,
    ENV_HIPFIRE_ATTN_FLASH,
    ENV_HIPFIRE_AWQ_F1_ONLY,
    ENV_HIPFIRE_BASELINE_ARCH,
    ENV_HIPFIRE_BF16_DENSE_M128,
    ENV_HIPFIRE_BF16_MOE_M256,
    ENV_HIPFIRE_BF16_WEIGHTS,
    ENV_HIPFIRE_BLOB_FORCE,
    ENV_HIPFIRE_CALIB_PROFILE,
    ENV_HIPFIRE_CASK_OFF,
    ENV_HIPFIRE_CHATML,
    ENV_HIPFIRE_CHAT_TEMPLATE_FILE,
    ENV_HIPFIRE_CONV1D_TREE_GFX1151,
    ENV_HIPFIRE_DAEMON_BIN,
    ENV_HIPFIRE_DAEMON_RESIDENT_STATE_BUDGET_MB,
    ENV_HIPFIRE_DDTREE_BUDGET,
    ENV_HIPFIRE_DDTREE_FORCE_SLOW,
    ENV_HIPFIRE_DDTREE_LOGW_CUTOFF,
    ENV_HIPFIRE_DDTREE_PATH_B_CAPTURE,
    ENV_HIPFIRE_DDTREE_PATH_C,
    ENV_HIPFIRE_DDTREE_PATH_C_VERBOSE,
    ENV_HIPFIRE_DDTREE_TAPE_DUMP,
    ENV_HIPFIRE_DDTREE_TOPK,
    ENV_HIPFIRE_DDTREE_TREE_LA,
    ENV_HIPFIRE_DEBUG_CORRUPT_PREFIX_HASH_ONCE,
    ENV_HIPFIRE_DEBUG_PREFIX_BOUNDARIES,
    ENV_HIPFIRE_DEBUG_SHARED_PREFIX_FANOUT,
    ENV_HIPFIRE_DEEPSEEK4_ATTN,
    ENV_HIPFIRE_DEEPSEEK4_ATTN_DEBUG_BISECT,
    ENV_HIPFIRE_DEEPSEEK4_ATTN_PER_POS,
    ENV_HIPFIRE_DEEPSEEK4_ATTN_TOPK_DIRECT,
    ENV_HIPFIRE_DEEPSEEK4_ATTN_TWIN,
    ENV_HIPFIRE_DEEPSEEK4_BATCH_HEAD,
    ENV_HIPFIRE_DEEPSEEK4_CACHE_TRACE,
    ENV_HIPFIRE_DEEPSEEK4_CHAT_RAW,
    ENV_HIPFIRE_DEEPSEEK4_COMP_F16_WMMA,
    ENV_HIPFIRE_DEEPSEEK4_COMP_ROPE_POS,
    ENV_HIPFIRE_DEEPSEEK4_DUMP_PROMPT,
    ENV_HIPFIRE_DEEPSEEK4_DUMP_STATE,
    ENV_HIPFIRE_DEEPSEEK4_DUMP_TOPK,
    ENV_HIPFIRE_DEEPSEEK4_EXPERT_LAYER_END,
    ENV_HIPFIRE_DEEPSEEK4_F32_TRACE,
    ENV_HIPFIRE_DEEPSEEK4_FUSED_UNSCATTER_SILU,
    ENV_HIPFIRE_DEEPSEEK4_GEN_TOKENS,
    ENV_HIPFIRE_DEEPSEEK4_GRAPH,
    ENV_HIPFIRE_DEEPSEEK4_HFQ4_WMMA,
    ENV_HIPFIRE_DEEPSEEK4_INDEXER_WMMA,
    ENV_HIPFIRE_DEEPSEEK4_LOAD_MTP,
    ENV_HIPFIRE_DEEPSEEK4_MAX_COMPRESS_POS,
    ENV_HIPFIRE_DEEPSEEK4_MODEL,
    ENV_HIPFIRE_DEEPSEEK4_MOE,
    ENV_HIPFIRE_DEEPSEEK4_MOE_8W,
    ENV_HIPFIRE_DEEPSEEK4_MOE_CND,
    ENV_HIPFIRE_DEEPSEEK4_MOE_DETERMINISTIC,
    ENV_HIPFIRE_DEEPSEEK4_MOE_GROUPED,
    ENV_HIPFIRE_DEEPSEEK4_MOE_GROUPED_GATE,
    ENV_HIPFIRE_DEEPSEEK4_MOE_LLOYD_4W,
    ENV_HIPFIRE_DEEPSEEK4_MOE_MMQLOAD,
    ENV_HIPFIRE_DEEPSEEK4_MOE_N32,
    ENV_HIPFIRE_DEEPSEEK4_MOE_NOSYNC,
    ENV_HIPFIRE_DEEPSEEK4_MTP_ADDON,
    ENV_HIPFIRE_DEEPSEEK4_MTP_HEAD_HC,
    ENV_HIPFIRE_DEEPSEEK4_MTP_SKIP_HEAD,
    ENV_HIPFIRE_DEEPSEEK4_POST_SCALE,
    ENV_HIPFIRE_DEEPSEEK4_PP_BATCH,
    ENV_HIPFIRE_DEEPSEEK4_Q8_4W,
    ENV_HIPFIRE_DEEPSEEK4_Q8_WMMA,
    ENV_HIPFIRE_DEEPSEEK4_ROUTE_SCALE,
    ENV_HIPFIRE_DEEPSEEK4_SEED,
    ENV_HIPFIRE_DEEPSEEK4_SPEC_DECODE,
    ENV_HIPFIRE_DEEPSEEK4_SPEC_K,
    ENV_HIPFIRE_DEEPSEEK4_TEMP,
    ENV_HIPFIRE_DEEPSEEK4_TOP_K,
    ENV_HIPFIRE_DEEPSEEK4_UPLOAD_EXPERTS,
    ENV_HIPFIRE_DEEPSEEK4_WO_MULTIROW,
    ENV_HIPFIRE_DEEPSEEK4_WO_Q8_WMMA,
    ENV_HIPFIRE_DELTANET_STATE,
    ENV_HIPFIRE_DETERMINISTIC,
    ENV_HIPFIRE_DEVICES,
    ENV_HIPFIRE_DFLASH_DRAFT,
    ENV_HIPFIRE_DFLASH_LOOP_BREAK,
    ENV_HIPFIRE_DFLASH_LOOP_BREAK_MAX_ESCALATIONS,
    ENV_HIPFIRE_DFLASH_LOOP_BREAK_RECOVERY,
    ENV_HIPFIRE_DFLASH_LOOP_BREAK_RP_MAX,
    ENV_HIPFIRE_DFLASH_LOOP_BREAK_RP_STEP,
    ENV_HIPFIRE_DFLASH_LOOP_BREAK_STOP_AFTER,
    ENV_HIPFIRE_DFLASH_LOOP_BREAK_TEMP,
    ENV_HIPFIRE_DFLASH_MODE,
    ENV_HIPFIRE_DFLASH_MOE_DRAFT_FFN_GRAPH,
    ENV_HIPFIRE_DFLASH_MOE_VERIFY_GRAPH_LMHEAD,
    ENV_HIPFIRE_DFLASH_NGRAM_BLOCK,
    ENV_HIPFIRE_DFLASH_Q8_LMHEAD_WMMA,
    ENV_HIPFIRE_DFLASH_ROLLBACK_COMPARE,
    ENV_HIPFIRE_DFLASH_SEED_ORACLE,
    ENV_HIPFIRE_DFLASH_SPEC_DEMO_BIN,
    ENV_HIPFIRE_DOT2_GEMV,
    ENV_HIPFIRE_DOTS_OCR_BF16_RESIDUAL,
    ENV_HIPFIRE_DOTS_OCR_DUMP_DIR,
    ENV_HIPFIRE_DOTS_OCR_TRACE,
    ENV_HIPFIRE_DPM_WARMUP_SECS,
    ENV_HIPFIRE_DRAFT_F16,
    ENV_HIPFIRE_DRAFT_GEMM_DUMP,
    ENV_HIPFIRE_DRAFT_SUBPHASE,
    ENV_HIPFIRE_DTOH_DUMP,
    ENV_HIPFIRE_DUMMY_GENERATE_DELAY_MS,
    ENV_HIPFIRE_DUMMY_PREFILL_DELAY_MS,
    ENV_HIPFIRE_DUMP_REQUEST,
    ENV_HIPFIRE_EMIT_TOKEN_IDS,
    ENV_HIPFIRE_EVAL_DATASET_MIRROR,
    ENV_HIPFIRE_EVAL_EVIDENCE_DIR,
    ENV_HIPFIRE_EVAL_HIPFIRE_BIN,
    ENV_HIPFIRE_EXPERIMENTAL_BUDGET_ALERT,
    ENV_HIPFIRE_FLASH_PARTIALS_BATCH,
    ENV_HIPFIRE_FORCE_A3B_EVICTION,
    ENV_HIPFIRE_FP16,
    ENV_HIPFIRE_FP8_WMMA,
    ENV_HIPFIRE_FUSED_HFQ4_2ROW_GFX1151,
    ENV_HIPFIRE_GATE_UP_VARIANT,
    ENV_HIPFIRE_GDN_Q8_REG_GFX1151,
    ENV_HIPFIRE_GEMM_DUMP,
    ENV_HIPFIRE_GEMV_ROWS,
    ENV_HIPFIRE_GEN,
    ENV_HIPFIRE_GENERATE_BATCH_PREFILL_DEBUG_SAMPLE,
    ENV_HIPFIRE_GFX942_GEMV_V3,
    ENV_HIPFIRE_GFX942_MFMA_PREFILL,
    ENV_HIPFIRE_GFX942_RMSNORM_SPLIT,
    ENV_HIPFIRE_GPTQ_DAMPING,
    ENV_HIPFIRE_GPU_SLAB_LOAD,
    ENV_HIPFIRE_GPU_SLAB_MIB,
    ENV_HIPFIRE_GPU_TOPK,
    ENV_HIPFIRE_GQA_CHUNK,
    ENV_HIPFIRE_GQA_FUSED,
    ENV_HIPFIRE_GRAPH,
    ENV_HIPFIRE_GRAPH_MOE,
    ENV_HIPFIRE_GRAPH_PREFILL,
    ENV_HIPFIRE_HAVE_2_GPU,
    ENV_HIPFIRE_HETERO_DIFF,
    ENV_HIPFIRE_HFQ4G128_MMQ,
    ENV_HIPFIRE_HFQ4G256_MMQ_GFX1151,
    ENV_HIPFIRE_HFQ4_MMQ_GFX906_Y64,
    ENV_HIPFIRE_HFQ6_QKVZA_4W,
    ENV_HIPFIRE_HFQ6_QKV_4W,
    ENV_HIPFIRE_HFQ6_RESIDUAL_4W,
    ENV_HIPFIRE_HIPCC_EXTRA_FLAGS,
    ENV_HIPFIRE_HIP_WAIT,
    ENV_HIPFIRE_HOST_PROFILE_BIN,
    ENV_HIPFIRE_HOST_TIMING,
    ENV_HIPFIRE_JINJA_CHAT,
    ENV_HIPFIRE_KERNEL_CACHE,
    ENV_HIPFIRE_KLD_DIRECT_F16KV_ATTN,
    ENV_HIPFIRE_KLD_DIRECT_WMMA_ATTN,
    ENV_HIPFIRE_KLD_FP32_GQA4_ATTN,
    ENV_HIPFIRE_KLD_GPU_TOPK,
    ENV_HIPFIRE_KLD_GRAPH,
    ENV_HIPFIRE_KLD_NO_ACTIVE_STREAM,
    ENV_HIPFIRE_KLD_PREFILL_ONLY,
    ENV_HIPFIRE_KLD_SOURCE_SHA256,
    ENV_HIPFIRE_KV_MODE,
    ENV_HIPFIRE_KV_PHYSICAL_CAP,
    ENV_HIPFIRE_LFM2_CAPTURE_POSTMIXER,
    ENV_HIPFIRE_LFM2_EXPERT_MQ6,
    ENV_HIPFIRE_LFM2_GRAPH,
    ENV_HIPFIRE_LFM2_PROJ_MQ4,
    ENV_HIPFIRE_LFM2_PROJ_MQ6,
    ENV_HIPFIRE_LLOYD_FORCE_BASELINE,
    ENV_HIPFIRE_LLOYD_GFX12,
    ENV_HIPFIRE_LLOYD_K3,
    ENV_HIPFIRE_LLOYD_MB4,
    ENV_HIPFIRE_LM_HEAD_F16,
    ENV_HIPFIRE_LM_HEAD_OVERWRITE,
    ENV_HIPFIRE_LM_HEAD_WMMA,
    ENV_HIPFIRE_LOAD_TRANSPORT,
    ENV_HIPFIRE_LOCAL,
    ENV_HIPFIRE_MAX_RESIDENT_WORKERS,
    ENV_HIPFIRE_MAX_SEQ,
    ENV_HIPFIRE_MEMSET_DUMP,
    ENV_HIPFIRE_MINIMAX_CAPTURE_POSTATTN,
    ENV_HIPFIRE_MINIMAX_DOWN_FORMAT,
    ENV_HIPFIRE_MINIMAX_ENABLE_DOWN_AWQ,
    ENV_HIPFIRE_MINIMAX_EXPERT_MQ2L,
    ENV_HIPFIRE_MINIMAX_EXPERT_MQ3L,
    ENV_HIPFIRE_MINIMAX_EXPERT_MQ6,
    ENV_HIPFIRE_MINIMAX_GRAPH,
    ENV_HIPFIRE_MMQ,
    ENV_HIPFIRE_MMQ_DIAG_QUANTIZE_ONLY,
    ENV_HIPFIRE_MMQ_SCREEN,
    ENV_HIPFIRE_MMQ_SCREEN_THRESHOLD,
    ENV_HIPFIRE_MODEL,
    ENV_HIPFIRE_MOE_GROUPED_GEMM,
    ENV_HIPFIRE_MOE_GROUPED_I8,
    ENV_HIPFIRE_MOE_GROUPED_I8_4W,
    ENV_HIPFIRE_MOE_GROUPED_I8_K4,
    ENV_HIPFIRE_MOE_GROUPED_I8_K4_GFX12,
    ENV_HIPFIRE_MOE_GROUPED_I8_K8,
    ENV_HIPFIRE_MOE_GROUPED_M2,
    ENV_HIPFIRE_MOE_HFQ6_4W,
    ENV_HIPFIRE_MOE_HFQ6_V2,
    ENV_HIPFIRE_MOE_INDEXED_2ROW_GFX1151,
    ENV_HIPFIRE_MOE_PARO_I8,
    ENV_HIPFIRE_MOE_PARO_I8_K8,
    ENV_HIPFIRE_MQ3_MB4,
    ENV_HIPFIRE_MTP_DEVICE_TOKEN_CHAIN,
    ENV_HIPFIRE_MTP_GPU_ACCEPT,
    ENV_HIPFIRE_MTP_K,
    ENV_HIPFIRE_MTP_MODE,
    ENV_HIPFIRE_MTP_PROPOSAL_GRAPH,
    ENV_HIPFIRE_MTP_Q8_VERIFY_WMMA,
    ENV_HIPFIRE_MTP_SMOKE_HEAD,
    ENV_HIPFIRE_MTP_SMOKE_TRUNK,
    ENV_HIPFIRE_MTP_SNAPSHOT_OVERLAP,
    ENV_HIPFIRE_MW16,
    ENV_HIPFIRE_NGRAM_LOOP_THRESHOLD,
    ENV_HIPFIRE_NGRAM_WINDOW,
    ENV_HIPFIRE_NORMALIZE_PROMPT,
    ENV_HIPFIRE_NO_PID_FILE,
    ENV_HIPFIRE_PAGED_MOE_DEBUG,
    ENV_HIPFIRE_PARO_BATCHED,
    ENV_HIPFIRE_PARO_FA3_FUSED,
    ENV_HIPFIRE_PARO_FUSED_PACK2,
    ENV_HIPFIRE_PARO_FUSE_RMSNORM,
    ENV_HIPFIRE_PARO_GATE_UP_FUSED,
    ENV_HIPFIRE_PARO_LA2_FUSED,
    ENV_HIPFIRE_PARO_LA4_FUSED,
    ENV_HIPFIRE_PARO_LA_GATES_MQ4G128,
    ENV_HIPFIRE_PARO_PACK1,
    ENV_HIPFIRE_PARO_PACK2,
    ENV_HIPFIRE_PARO_PACK4,
    ENV_HIPFIRE_PARO_PREROTATE,
    ENV_HIPFIRE_PARO_SHARED_PAIRS,
    ENV_HIPFIRE_PARO_SMALL_DIRECT,
    ENV_HIPFIRE_PARO_SWIGLU_FUSED,
    ENV_HIPFIRE_PFLASH_DEBUG,
    ENV_HIPFIRE_PFLASH_DRAFTER_KV,
    ENV_HIPFIRE_PFLASH_DRAFTER_STATE,
    ENV_HIPFIRE_PFLASH_SCORE_LAYER,
    ENV_HIPFIRE_PP_DFLASH,
    ENV_HIPFIRE_PP_LAYERS,
    ENV_HIPFIRE_PP_PARITY_MODEL,
    ENV_HIPFIRE_PP_PFLASH,
    ENV_HIPFIRE_PREFILL_ALPHA,
    ENV_HIPFIRE_PREFILL_BATCHED,
    ENV_HIPFIRE_PREFILL_BLOCK,
    ENV_HIPFIRE_PREFILL_COMPRESSION,
    ENV_HIPFIRE_PREFILL_DRAFTER,
    ENV_HIPFIRE_PREFILL_KEEP_RATIO,
    ENV_HIPFIRE_PREFILL_MAX_BATCH,
    ENV_HIPFIRE_PREFILL_MAX_LAYER,
    ENV_HIPFIRE_PREFILL_MIN_KEEP,
    ENV_HIPFIRE_PREFILL_PROFILE,
    ENV_HIPFIRE_PREFILL_RECENT,
    ENV_HIPFIRE_PREFILL_REUSE_PBS,
    ENV_HIPFIRE_PREFILL_SINK,
    ENV_HIPFIRE_PREFILL_SPARSE_THRESHOLD,
    ENV_HIPFIRE_PREFILL_STOP_AFTER_LA_LAYER,
    ENV_HIPFIRE_PREFILL_STOP_STAGE,
    ENV_HIPFIRE_PREFILL_STOP_STAGE_LAYER,
    ENV_HIPFIRE_PREFILL_THRESHOLD,
    ENV_HIPFIRE_PREFIX_BOUNDARY_CHECKPOINTS,
    ENV_HIPFIRE_PROFILE,
    ENV_HIPFIRE_PROFILE_CYCLES,
    ENV_HIPFIRE_PROFILE_DECODE,
    ENV_HIPFIRE_PROMPT_HEAT_JSON,
    ENV_HIPFIRE_PROMPT_HEAT_LIMIT,
    ENV_HIPFIRE_PROMPT_TOKEN_HEAT,
    ENV_HIPFIRE_Q8_BATCHED_LEGACY,
    ENV_HIPFIRE_Q8_DP4A,
    ENV_HIPFIRE_Q8_DP4A_W64,
    ENV_HIPFIRE_Q8_GATE_UP_4W,
    ENV_HIPFIRE_Q8_GATE_UP_BENCH,
    ENV_HIPFIRE_Q8_TOKPAR,
    ENV_HIPFIRE_Q8_WMMA_4W,
    ENV_HIPFIRE_Q8_WMMA_X64,
    ENV_HIPFIRE_QA_KV_MODES,
    ENV_HIPFIRE_QUANT_DIAG_PATH,
    ENV_HIPFIRE_QUANT_THREADS,
    ENV_HIPFIRE_DECODE_BATCH_GROUPED_PARITY_CHUNK_SIZE,
    ENV_HIPFIRE_QWEN35_DECODE_BATCH,
    ENV_HIPFIRE_QWEN35_DECODE_BATCH_MAX,
    ENV_HIPFIRE_QWEN35_DECODE_INTERNAL_PARITY,
    ENV_HIPFIRE_QWEN35_DECODE_NATIVE_MULTIROW,
    ENV_HIPFIRE_QWEN35_EXPERT_CACHE_MB,
    ENV_HIPFIRE_QWEN35_EXPERT_CACHE_TRACE,
    ENV_HIPFIRE_QWEN35_FINITE_TRACE,
    ENV_HIPFIRE_QWEN35_PAGED_EXPERTS,
    ENV_HIPFIRE_QWEN35_PREFILL_SESSION_BATCH,
    ENV_HIPFIRE_QWEN35_ROUTED_ONLY_MOE_FORWARD,
    ENV_HIPFIRE_QWEN35_STAGE_SYNC,
    ENV_HIPFIRE_QWEN35_STAGE_TRACE,
    ENV_HIPFIRE_QWEN35_STATE_QUANT,
    ENV_HIPFIRE_RDNA2_VARIANT,
    ENV_HIPFIRE_REPLAY_GRAPH,
    ENV_HIPFIRE_RESPONSES_STATE_MAX,
    ENV_HIPFIRE_ROCBLAS_ALL_ARCHS,
    ENV_HIPFIRE_ROCBLAS_OFF,
    ENV_HIPFIRE_ROCPROF_BIN,
    ENV_HIPFIRE_ROCPROF_CSV,
    ENV_HIPFIRE_ROPE_INTERLEAVED_LEGACY,
    ENV_HIPFIRE_RUN_EXAMPLE_BIN,
    ENV_HIPFIRE_SAMPLE_COMPARE,
    ENV_HIPFIRE_SERVER_PREFILL_SHARED_PREFIX_FANOUT,
    ENV_HIPFIRE_SERVER_RESIDENT_STATE_BUDGET_MB,
    ENV_HIPFIRE_SMOKE_KV,
    ENV_HIPFIRE_SMOKE_KV_SEQ,
    ENV_HIPFIRE_SMOKE_MODE,
    ENV_HIPFIRE_SMOKE_PROMPT,
    ENV_HIPFIRE_SMOKE_STEPS,
    ENV_HIPFIRE_SPEC_PHASES,
    ENV_HIPFIRE_STATE,
    ENV_HIPFIRE_STATE_QUANT,
    ENV_HIPFIRE_TARGET_ARCH,
    ENV_HIPFIRE_TIER_RATIO,
    ENV_HIPFIRE_UNIFORM_VRAM_TOLERANCE_GB,
    ENV_HIPFIRE_VERIFY_GRAPH,
    ENV_HIPFIRE_VERIFY_GRAPH_TIMING,
    ENV_HIPFIRE_VERIFY_GRAPH_TREE,
    ENV_HIPFIRE_VL_DUMP_DIR,
    ENV_HIPFIRE_WO_MMQ,
    ENV_HIPFIRE_WO_WMMA_VARIANT,
    ENV_HIP_PATH,
    ENV_HOME,
    ENV_HOSTNAME,
    ENV_MAX_TOKENS,
    ENV_MMQ_TEST_MODE,
    ENV_NO_COLOR,
    ENV_NO_NGRAM,
    ENV_PATH,
    ENV_PROMPT_MODE,
    ENV_QWEN35_TEST_MODEL,
    ENV_TINYLLAMA_GGUF,
    ENV_TRIALS,
    ENV_USERPROFILE,
    ENV_USE_SAMPLE,
    ENV_VERIFY,
    ENV_W64,
];
