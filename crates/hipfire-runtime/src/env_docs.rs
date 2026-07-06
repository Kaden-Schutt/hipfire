#![allow(dead_code)]

// SPDX-License-Identifier: Apache-2.0
//
// Generated automatically from source env usage by `hipfire gen-env-docs`.
// Do not hand-edit. Re-run `cargo run -p hipfire-cli -- gen-env-docs`.

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
    source: "crates/hipfire-rdna/examples/bench_stream_overlap.rs:93",
};

/// `BENCH_DRAFT_K` — Runtime variable controlling bench draft k in hipfire
pub const ENV_BENCH_DRAFT_K: EnvVarDoc = EnvVarDoc {
    name: "BENCH_DRAFT_K",
    description: "Runtime variable controlling bench draft k in hipfire",
    source: "crates/hipfire-rdna/examples/bench_stream_overlap.rs:225",
};

/// `BENCH_DRAFT_LAYERS` — Runtime variable controlling bench draft layers in hipfire
pub const ENV_BENCH_DRAFT_LAYERS: EnvVarDoc = EnvVarDoc {
    name: "BENCH_DRAFT_LAYERS",
    description: "Runtime variable controlling bench draft layers in hipfire",
    source: "crates/hipfire-rdna/examples/bench_stream_overlap.rs:252",
};

/// `BENCH_DRAFT_M` — Runtime variable controlling bench draft m in hipfire
pub const ENV_BENCH_DRAFT_M: EnvVarDoc = EnvVarDoc {
    name: "BENCH_DRAFT_M",
    description: "Runtime variable controlling bench draft m in hipfire",
    source: "crates/hipfire-rdna/examples/bench_stream_overlap.rs:221",
};

/// `BENCH_DRAFT_N` — Runtime variable controlling bench draft n in hipfire
pub const ENV_BENCH_DRAFT_N: EnvVarDoc = EnvVarDoc {
    name: "BENCH_DRAFT_N",
    description: "Runtime variable controlling bench draft n in hipfire",
    source: "crates/hipfire-rdna/examples/bench_stream_overlap.rs:229",
};

/// `BENCH_K` — Runtime variable controlling bench k in hipfire
pub const ENV_BENCH_K: EnvVarDoc = EnvVarDoc {
    name: "BENCH_K",
    description: "Runtime variable controlling bench k in hipfire",
    source: "crates/hipfire-rdna/examples/bench_stream_overlap.rs:89",
};

/// `BENCH_M` — Runtime variable controlling bench m in hipfire
pub const ENV_BENCH_M: EnvVarDoc = EnvVarDoc {
    name: "BENCH_M",
    description: "Runtime variable controlling bench m in hipfire",
    source: "crates/hipfire-rdna/examples/bench_stream_overlap.rs:85",
};

/// `BENCH_VERIFY_LAYERS` — Runtime variable controlling bench verify layers in hipfire
pub const ENV_BENCH_VERIFY_LAYERS: EnvVarDoc = EnvVarDoc {
    name: "BENCH_VERIFY_LAYERS",
    description: "Runtime variable controlling bench verify layers in hipfire",
    source: "crates/hipfire-rdna/examples/bench_stream_overlap.rs:256",
};

/// `CAP_POS` — Environment toggle value controls runtime behavior
pub const ENV_CAP_POS: EnvVarDoc = EnvVarDoc {
    name: "CAP_POS",
    description: "Environment toggle value controls runtime behavior",
    source: "crates/hipfire-arch-nemotron/examples/bisect_nano4b.rs:84",
};

/// `CARGO_FEATURE_NPU_KERNELS` — Runtime variable controlling cargo feature npu kernels in hipfire
pub const ENV_CARGO_FEATURE_NPU_KERNELS: EnvVarDoc = EnvVarDoc {
    name: "CARGO_FEATURE_NPU_KERNELS",
    description: "Runtime variable controlling cargo feature npu kernels in hipfire",
    source: "crates/hipfire-arch-qwen35/build.rs:29",
};

/// `CARGO_MANIFEST_DIR` — CARGO_MANIFEST_DIR = <workspace>/crates/hipfire-arch-qwen35
pub const ENV_CARGO_MANIFEST_DIR: EnvVarDoc = EnvVarDoc {
    name: "CARGO_MANIFEST_DIR",
    description: "CARGO_MANIFEST_DIR = <workspace>/crates/hipfire-arch-qwen35",
    source: "crates/hipfire-arch-qwen35/build.rs:592",
};

/// `CARGO_PKG_VERSION` — Defaults to unknown when unset
pub const ENV_CARGO_PKG_VERSION: EnvVarDoc = EnvVarDoc {
    name: "CARGO_PKG_VERSION",
    description: "Defaults to unknown when unset",
    source: "crates/hipfire-build-info/build.rs:24",
};

/// `COLORTERM` — Runtime variable controlling colorterm in hipfire
pub const ENV_COLORTERM: EnvVarDoc = EnvVarDoc {
    name: "COLORTERM",
    description: "Runtime variable controlling colorterm in hipfire",
    source: "crates/hipfire-system-monitor/src/lib.rs:316",
};

/// `DDTREE_TIMING` — pre_verify / verify. The draft and top-K are fused into one GPU-
pub const ENV_DDTREE_TIMING: EnvVarDoc = EnvVarDoc {
    name: "DDTREE_TIMING",
    description: "pre_verify / verify. The draft and top-K are fused into one GPU-",
    source: "crates/hipfire-arch-qwen35/src/speculative.rs:10798",
};

/// `DEBUG_LAYERS` — Runtime variable controlling debug layers in hipfire
pub const ENV_DEBUG_LAYERS: EnvVarDoc = EnvVarDoc {
    name: "DEBUG_LAYERS",
    description: "Runtime variable controlling debug layers in hipfire",
    source: "crates/hipfire-arch-qwen35/src/qwen35.rs:8757",
};

/// `DFLASH_LIVE_TAU` — Runtime variable controlling dflash live tau in hipfire
pub const ENV_DFLASH_LIVE_TAU: EnvVarDoc = EnvVarDoc {
    name: "DFLASH_LIVE_TAU",
    description: "Runtime variable controlling dflash live tau in hipfire",
    source: "crates/hipfire-runtime/examples/dflash_spec_demo.rs:1540",
};

/// `FP32_STATE` — Runtime variable controlling fp32 state in hipfire
pub const ENV_FP32_STATE: EnvVarDoc = EnvVarDoc {
    name: "FP32_STATE",
    description: "Runtime variable controlling fp32 state in hipfire",
    source: "crates/hipfire-runtime/examples/infer_qwen35.rs:223",
};

/// `GPU_LOCK_TIMEOUT` — Runtime variable controlling gpu lock timeout in hipfire
pub const ENV_GPU_LOCK_TIMEOUT: EnvVarDoc = EnvVarDoc {
    name: "GPU_LOCK_TIMEOUT",
    description: "Runtime variable controlling gpu lock timeout in hipfire",
    source: "crates/hipfire-cli/src/commands/lock.rs:111",
};

/// `GPU_POLL_INTERVAL` — Runtime variable controlling gpu poll interval in hipfire
pub const ENV_GPU_POLL_INTERVAL: EnvVarDoc = EnvVarDoc {
    name: "GPU_POLL_INTERVAL",
    description: "Runtime variable controlling gpu poll interval in hipfire",
    source: "crates/hipfire-cli/src/commands/lock.rs:104",
};

/// `HFHS_REAL` — Runtime variable controlling hfhs real in hipfire
pub const ENV_HFHS_REAL: EnvVarDoc = EnvVarDoc {
    name: "HFHS_REAL",
    description: "Runtime variable controlling hfhs real in hipfire",
    source: "crates/hipfire-quantize/src/hfhs_diag.rs:223",
};

/// `HFQ_TEST_N_ITER` — Parses "HFQ_TEST_N_ITER" with fallback defaults
pub const ENV_HFQ_TEST_N_ITER: EnvVarDoc = EnvVarDoc {
    name: "HFQ_TEST_N_ITER",
    description: "Parses \"HFQ_TEST_N_ITER\" with fallback defaults",
    source: "crates/hipfire-rdna/examples/test_hfq4_residual_dp4a.rs:111",
};

/// `HFQ_TEST_SCALE_LOG10` — Parses "HFQ_TEST_SCALE_LOG10" with fallback defaults
pub const ENV_HFQ_TEST_SCALE_LOG10: EnvVarDoc = EnvVarDoc {
    name: "HFQ_TEST_SCALE_LOG10",
    description: "Parses \"HFQ_TEST_SCALE_LOG10\" with fallback defaults",
    source: "crates/hipfire-rdna/examples/test_hfq4_residual_dp4a.rs:233",
};

/// `HFQ_TEST_ZP_MAX` — Parses "HFQ_TEST_ZP_MAX" with fallback defaults
pub const ENV_HFQ_TEST_ZP_MAX: EnvVarDoc = EnvVarDoc {
    name: "HFQ_TEST_ZP_MAX",
    description: "Parses \"HFQ_TEST_ZP_MAX\" with fallback defaults",
    source: "crates/hipfire-rdna/examples/test_hfq4_residual_dp4a.rs:237",
};

/// `HF_HOME` — Runtime variable controlling hf home in hipfire
pub const ENV_HF_HOME: EnvVarDoc = EnvVarDoc {
    name: "HF_HOME",
    description: "Runtime variable controlling hf home in hipfire",
    source: "crates/hipfire-server/src/routes/sdapi.rs:4303",
};

/// `HIPFIRE_ADAPTIVE_B_DOWN` — Runtime variable controlling adaptive b down in hipfire
pub const ENV_HIPFIRE_ADAPTIVE_B_DOWN: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_ADAPTIVE_B_DOWN",
    description: "Runtime variable controlling adaptive b down in hipfire",
    source: "crates/hipfire-runtime/examples/dflash_spec_demo.rs:1807",
};

/// `HIPFIRE_ADAPTIVE_B_UNSAFE` — the user explicitly widens. Opt out via HIPFIRE_ADAPTIVE_B_UNSAFE=1
pub const ENV_HIPFIRE_ADAPTIVE_B_UNSAFE: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_ADAPTIVE_B_UNSAFE",
    description: "the user explicitly widens. Opt out via HIPFIRE_ADAPTIVE_B_UNSAFE=1",
    source: "crates/hipfire-runtime/examples/dflash_spec_demo.rs:949",
};

/// `HIPFIRE_ADAPTIVE_B_UP` — HIPFIRE_ADAPTIVE_B_UP=0.XX / HIPFIRE_ADAPTIVE_B_DOWN=0.XX
pub const ENV_HIPFIRE_ADAPTIVE_B_UP: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_ADAPTIVE_B_UP",
    description: "HIPFIRE_ADAPTIVE_B_UP=0.XX / HIPFIRE_ADAPTIVE_B_DOWN=0.XX",
    source: "crates/hipfire-runtime/examples/dflash_spec_demo.rs:1803",
};

/// `HIPFIRE_ALLOW_MIXED_ARCH` — Runtime variable controlling allow mixed arch in hipfire
pub const ENV_HIPFIRE_ALLOW_MIXED_ARCH: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_ALLOW_MIXED_ARCH",
    description: "Runtime variable controlling allow mixed arch in hipfire",
    source: "crates/hipfire-runtime/src/config.rs:103",
};

/// `HIPFIRE_ALLOW_MQ2` — Enabled when set to 1
pub const ENV_HIPFIRE_ALLOW_MQ2: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_ALLOW_MQ2",
    description: "Enabled when set to 1",
    source: "crates/hipfire-quantize/src/main.rs:5552",
};

/// `HIPFIRE_ALLOW_MQ2_LLOYD` — Enabled when set to 1
pub const ENV_HIPFIRE_ALLOW_MQ2_LLOYD: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_ALLOW_MQ2_LLOYD",
    description: "Enabled when set to 1",
    source: "crates/hipfire-quantize/src/main.rs:5587",
};

/// `HIPFIRE_ALLOW_MQ3_LLOYD` — Enabled when set to 1
pub const ENV_HIPFIRE_ALLOW_MQ3_LLOYD: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_ALLOW_MQ3_LLOYD",
    description: "Enabled when set to 1",
    source: "crates/hipfire-quantize/src/main.rs:5572",
};

/// `HIPFIRE_ALLOW_MQ4_LLOYD` — Enabled when set to 1
pub const ENV_HIPFIRE_ALLOW_MQ4_LLOYD: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_ALLOW_MQ4_LLOYD",
    description: "Enabled when set to 1",
    source: "crates/hipfire-quantize/src/main.rs:5622",
};

/// `HIPFIRE_ALLOW_UNIT_IMATRIX` — Environment toggle value controls runtime behavior
pub const ENV_HIPFIRE_ALLOW_UNIT_IMATRIX: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_ALLOW_UNIT_IMATRIX",
    description: "Environment toggle value controls runtime behavior",
    source: "crates/hipfire-quantize/src/main.rs:5174",
};

/// `HIPFIRE_ATTN_FLASH` — Honors HIPFIRE_ATTN_FLASH=never|0|off as an explicit override
pub const ENV_HIPFIRE_ATTN_FLASH: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_ATTN_FLASH",
    description: "Honors HIPFIRE_ATTN_FLASH=never|0|off as an explicit override",
    source: "crates/hipfire-arch-qwen35/src/qwen35.rs:8994",
};

/// `HIPFIRE_AWQ_F1_ONLY` — F1-vs-F2 A/B gate. When "HIPFIRE_AWQ_F1_ONLY=1" is set, the F2
pub const ENV_HIPFIRE_AWQ_F1_ONLY: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_AWQ_F1_ONLY",
    description: "F1-vs-F2 A/B gate. When \"HIPFIRE_AWQ_F1_ONLY=1\" is set, the F2",
    source: "crates/hipfire-quantize/src/main.rs:4311",
};

/// `HIPFIRE_BASELINE_ARCH` — Runtime variable controlling baseline arch in hipfire
pub const ENV_HIPFIRE_BASELINE_ARCH: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_BASELINE_ARCH",
    description: "Runtime variable controlling baseline arch in hipfire",
    source: "crates/hipfire-runtime/examples/coherence_probe.rs:423",
};

/// `HIPFIRE_BATCHES_STATE_MAX` — Parses "HIPFIRE_BATCHES_STATE_MAX" with fallback defaults
pub const ENV_HIPFIRE_BATCHES_STATE_MAX: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_BATCHES_STATE_MAX",
    description: "Parses \"HIPFIRE_BATCHES_STATE_MAX\" with fallback defaults",
    source: "crates/hipfire-server/src/routes/batches.rs:687",
};

/// `HIPFIRE_BENCH_QWEN35_SPEED_BIN` — Runtime variable controlling bench qwen35 speed bin in hipfire
pub const ENV_HIPFIRE_BENCH_QWEN35_SPEED_BIN: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_BENCH_QWEN35_SPEED_BIN",
    description: "Runtime variable controlling bench qwen35 speed bin in hipfire",
    source: "crates/hipfire-eval/src/lib.rs:1162",
};

/// `HIPFIRE_BF16_DENSE_M128` — Enabled by default; set to 0 to disable
pub const ENV_HIPFIRE_BF16_DENSE_M128: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_BF16_DENSE_M128",
    description: "Enabled by default; set to 0 to disable",
    source: "crates/hipfire-rdna/src/dispatch/gemm_base.rs:662",
};

/// `HIPFIRE_BF16_MOE_M256` — Enabled when set to 1
pub const ENV_HIPFIRE_BF16_MOE_M256: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_BF16_MOE_M256",
    description: "Enabled when set to 1",
    source: "crates/hipfire-rdna/src/dispatch/overlays/gfx1151.rs:161",
};

/// `HIPFIRE_BF16_WEIGHTS` — Runtime variable controlling bf16 weights in hipfire
pub const ENV_HIPFIRE_BF16_WEIGHTS: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_BF16_WEIGHTS",
    description: "Runtime variable controlling bf16 weights in hipfire",
    source: "crates/hipfire-arch-qwen35/src/qwen35.rs:222",
};

/// `HIPFIRE_BLOB_FORCE` — Graph / capture / deterministic
pub const ENV_HIPFIRE_BLOB_FORCE: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_BLOB_FORCE",
    description: "Graph / capture / deterministic",
    source: "crates/hipfire-rdna/src/feature_flags.rs:291",
};

/// `HIPFIRE_CALIB_F64_AUDIT` — Enabled when set to 1
pub const ENV_HIPFIRE_CALIB_F64_AUDIT: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_CALIB_F64_AUDIT",
    description: "Enabled when set to 1",
    source: "crates/hipfire-runtime/src/calibration.rs:126",
};

/// `HIPFIRE_CALIB_HESSIAN_STORAGE` — Selects behavior from recognized values
pub const ENV_HIPFIRE_CALIB_HESSIAN_STORAGE: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_CALIB_HESSIAN_STORAGE",
    description: "Selects behavior from recognized values",
    source: "crates/hipfire-runtime/src/calibration.rs:38",
};

/// `HIPFIRE_CALIB_PROFILE` — Enable with HIPFIRE_CALIB_PROFILE=1; emits to stderr
pub const ENV_HIPFIRE_CALIB_PROFILE: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_CALIB_PROFILE",
    description: "Enable with HIPFIRE_CALIB_PROFILE=1; emits to stderr",
    source: "crates/hipfire-runtime/examples/triattn_validate.rs:60",
};

/// `HIPFIRE_CASK_OFF` — HIPFIRE_CASK_OFF=1 is an ops escape hatch: forces no auto-attach
pub const ENV_HIPFIRE_CASK_OFF: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_CASK_OFF",
    description: "HIPFIRE_CASK_OFF=1 is an ops escape hatch: forces no auto-attach",
    source: "crates/hipfire-server/src/routes/chat.rs:258",
};

/// `HIPFIRE_CHATML` — Enabled when set to 1
pub const ENV_HIPFIRE_CHATML: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_CHATML",
    description: "Enabled when set to 1",
    source: "crates/hipfire-runtime/examples/probe_argmax_agreement.rs:78",
};

/// `HIPFIRE_CHAT_TEMPLATE_FILE` — 1. Env-var override
pub const ENV_HIPFIRE_CHAT_TEMPLATE_FILE: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_CHAT_TEMPLATE_FILE",
    description: "1. Env-var override",
    source: "crates/hipfire-prompt/src/lib.rs:2034",
};

/// `HIPFIRE_COLLECT_ARTIFACTS_BIN` — Runtime variable controlling collect artifacts bin in hipfire
pub const ENV_HIPFIRE_COLLECT_ARTIFACTS_BIN: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_COLLECT_ARTIFACTS_BIN",
    description: "Runtime variable controlling collect artifacts bin in hipfire",
    source: "crates/hipfire-eval/src/lib.rs:1209",
};

/// `HIPFIRE_COMP_DUMP` — Stage-bisect dump: HIPFIRE_COMP_DUMP="<pos>,<layer>" prints each
pub const ENV_HIPFIRE_COMP_DUMP: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_COMP_DUMP",
    description: "Stage-bisect dump: HIPFIRE_COMP_DUMP=\"<pos>,<layer>\" prints each",
    source: "crates/hipfire-arch-deepseek4/src/forward.rs:647",
};

/// `HIPFIRE_CONV1D_TREE_GFX1151` — Environment toggle value controls runtime behavior
pub const ENV_HIPFIRE_CONV1D_TREE_GFX1151: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_CONV1D_TREE_GFX1151",
    description: "Environment toggle value controls runtime behavior",
    source: "crates/hipfire-rdna/examples/bench_conv1d_tree_gfx1151.rs:59",
};

/// `HIPFIRE_DAEMON_BIN` — Runtime variable controlling daemon bin in hipfire
pub const ENV_HIPFIRE_DAEMON_BIN: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DAEMON_BIN",
    description: "Runtime variable controlling daemon bin in hipfire",
    source: "crates/hipfire-daemon-adapter/src/lib.rs:1229",
};

/// `HIPFIRE_DAEMON_RESIDENT_STATE_BUDGET_MB` — Runtime variable controlling daemon resident state budget mb in hipfire
pub const ENV_HIPFIRE_DAEMON_RESIDENT_STATE_BUDGET_MB: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DAEMON_RESIDENT_STATE_BUDGET_MB",
    description: "Runtime variable controlling daemon resident state budget mb in hipfire",
    source: "crates/hipfire-serving-core/src/session.rs:1445",
};

/// `HIPFIRE_DDTREE_BUDGET` — Runtime variable controlling DDTree budget in hipfire
pub const ENV_HIPFIRE_DDTREE_BUDGET: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DDTREE_BUDGET",
    description: "Runtime variable controlling DDTree budget in hipfire",
    source: "crates/hipfire-serving-core/src/load.rs:3432",
};

/// `HIPFIRE_DDTREE_FORCE_SLOW` — HIPFIRE_DDTREE_FORCE_SLOW=1: force the slow (re-verify) path even when
pub const ENV_HIPFIRE_DDTREE_FORCE_SLOW: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DDTREE_FORCE_SLOW",
    description: "HIPFIRE_DDTREE_FORCE_SLOW=1: force the slow (re-verify) path even when",
    source: "crates/hipfire-arch-qwen35/src/speculative.rs:11025",
};

/// `HIPFIRE_DDTREE_LOGW_CUTOFF` — Adaptive-B usage report — only meaningful when --adaptive-b is on
pub const ENV_HIPFIRE_DDTREE_LOGW_CUTOFF: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DDTREE_LOGW_CUTOFF",
    description: "Adaptive-B usage report — only meaningful when --adaptive-b is on",
    source: "crates/hipfire-runtime/examples/dflash_spec_demo.rs:2545",
};

/// `HIPFIRE_DDTREE_PATH_B_CAPTURE` — Path B slow-path-kill (opt-in, WIP). Replaces the ~40-50 ms full
pub const ENV_HIPFIRE_DDTREE_PATH_B_CAPTURE: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DDTREE_PATH_B_CAPTURE",
    description: "Path B slow-path-kill (opt-in, WIP). Replaces the ~40-50 ms full",
    source: "crates/hipfire-arch-qwen35/src/speculative.rs:11064",
};

/// `HIPFIRE_DDTREE_PATH_C` — Resolve "HIPFIRE_DDTREE_PATH_C" ONCE before the decode loop. The
pub const ENV_HIPFIRE_DDTREE_PATH_C: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DDTREE_PATH_C",
    description: "Resolve \"HIPFIRE_DDTREE_PATH_C\" ONCE before the decode loop. The",
    source: "crates/hipfire-serving-core/src/generate.rs:864",
};

/// `HIPFIRE_DDTREE_PATH_C_VERBOSE` — Runtime variable controlling DDTree path c verbose in hipfire
pub const ENV_HIPFIRE_DDTREE_PATH_C_VERBOSE: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DDTREE_PATH_C_VERBOSE",
    description: "Runtime variable controlling DDTree path c verbose in hipfire",
    source: "crates/hipfire-arch-qwen35/src/speculative.rs:11589",
};

/// `HIPFIRE_DDTREE_PATH_C_VERIFY_GRAPH` — Runtime variable controlling DDTree path c verify graph in hipfire
pub const ENV_HIPFIRE_DDTREE_PATH_C_VERIFY_GRAPH: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DDTREE_PATH_C_VERIFY_GRAPH",
    description: "Runtime variable controlling DDTree path c verify graph in hipfire",
    source: "crates/hipfire-arch-qwen35/src/speculative.rs:219",
};

/// `HIPFIRE_DDTREE_TAPE_DUMP` — Enabled when set to 1
pub const ENV_HIPFIRE_DDTREE_TAPE_DUMP: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DDTREE_TAPE_DUMP",
    description: "Enabled when set to 1",
    source: "crates/hipfire-arch-qwen35/src/speculative.rs:11044",
};

/// `HIPFIRE_DDTREE_TOPK` — Runtime variable controlling DDTree topk in hipfire
pub const ENV_HIPFIRE_DDTREE_TOPK: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DDTREE_TOPK",
    description: "Runtime variable controlling DDTree topk in hipfire",
    source: "crates/hipfire-serving-core/src/load.rs:3504",
};

/// `HIPFIRE_DDTREE_TREE_LA` — Opt out with HIPFIRE_DDTREE_TREE_LA=0 if a regression is suspected
pub const ENV_HIPFIRE_DDTREE_TREE_LA: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DDTREE_TREE_LA",
    description: "Opt out with HIPFIRE_DDTREE_TREE_LA=0 if a regression is suspected",
    source: "crates/hipfire-arch-qwen35/src/speculative.rs:10906",
};

/// `HIPFIRE_DEBUG_CHAT` — Enabled when set to 1
pub const ENV_HIPFIRE_DEBUG_CHAT: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DEBUG_CHAT",
    description: "Enabled when set to 1",
    source: "crates/hipfire-server/src/routes/chat.rs:112",
};

/// `HIPFIRE_DEBUG_PREFILL_ELIGIBLE` — Environment toggle value controls runtime behavior
pub const ENV_HIPFIRE_DEBUG_PREFILL_ELIGIBLE: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DEBUG_PREFILL_ELIGIBLE",
    description: "Environment toggle value controls runtime behavior",
    source: "crates/hipfire-arch-qwen35/src/qwen35.rs:13904",
};

/// `HIPFIRE_DEBUG_PREFIX_BOUNDARIES` — Runtime variable controlling debug prefix boundaries in hipfire
pub const ENV_HIPFIRE_DEBUG_PREFIX_BOUNDARIES: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DEBUG_PREFIX_BOUNDARIES",
    description: "Runtime variable controlling debug prefix boundaries in hipfire",
    source: "crates/hipfire-serving-core/src/qwen35_prefill.rs:639",
};

/// `HIPFIRE_DEEPSEEK4_ATTN` — main model's final_norm_and_head head-HC reduction. Without
pub const ENV_HIPFIRE_DEEPSEEK4_ATTN: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DEEPSEEK4_ATTN",
    description: "main model's final_norm_and_head head-HC reduction. Without",
    source: "crates/hipfire-arch-deepseek4/src/forward.rs:73",
};

/// `HIPFIRE_DEEPSEEK4_ATTN_DEBUG_BISECT` — DEBUG: in-kernel bisect (HIPFIRE_DEEPSEEK4_ATTN_DEBUG_BISECT=1)
pub const ENV_HIPFIRE_DEEPSEEK4_ATTN_DEBUG_BISECT: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DEEPSEEK4_ATTN_DEBUG_BISECT",
    description: "DEBUG: in-kernel bisect (HIPFIRE_DEEPSEEK4_ATTN_DEBUG_BISECT=1)",
    source: "crates/hipfire-arch-deepseek4/src/forward.rs:6224",
};

/// `HIPFIRE_DEEPSEEK4_ATTN_PER_POS` — DEBUG: HIPFIRE_DEEPSEEK4_ATTN_PER_POS=1 substitutes a per-position loop
pub const ENV_HIPFIRE_DEEPSEEK4_ATTN_PER_POS: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DEEPSEEK4_ATTN_PER_POS",
    description: "DEBUG: HIPFIRE_DEEPSEEK4_ATTN_PER_POS=1 substitutes a per-position loop",
    source: "crates/hipfire-arch-deepseek4/src/forward.rs:6157",
};

/// `HIPFIRE_DEEPSEEK4_ATTN_TOPK_DIRECT` — Runtime variable controlling deepseek4 attn topk direct in hipfire
pub const ENV_HIPFIRE_DEEPSEEK4_ATTN_TOPK_DIRECT: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DEEPSEEK4_ATTN_TOPK_DIRECT",
    description: "Runtime variable controlling deepseek4 attn topk direct in hipfire",
    source: "crates/hipfire-arch-deepseek4/src/forward.rs:6506",
};

/// `HIPFIRE_DEEPSEEK4_ATTN_TWIN` — DEBUG: same-process twin-call test (HIPFIRE_DEEPSEEK4_ATTN_TWIN=1)
pub const ENV_HIPFIRE_DEEPSEEK4_ATTN_TWIN: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DEEPSEEK4_ATTN_TWIN",
    description: "DEBUG: same-process twin-call test (HIPFIRE_DEEPSEEK4_ATTN_TWIN=1)",
    source: "crates/hipfire-arch-deepseek4/src/forward.rs:6202",
};

/// `HIPFIRE_DEEPSEEK4_BATCH_HEAD` — Opt-out: HIPFIRE_DEEPSEEK4_BATCH_HEAD=0 forces the legacy per-position
pub const ENV_HIPFIRE_DEEPSEEK4_BATCH_HEAD: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DEEPSEEK4_BATCH_HEAD",
    description: "Opt-out: HIPFIRE_DEEPSEEK4_BATCH_HEAD=0 forces the legacy per-position",
    source: "crates/hipfire-arch-deepseek4/src/forward.rs:8164",
};

/// `HIPFIRE_DEEPSEEK4_CACHE_TRACE` — Runtime variable controlling deepseek4 cache trace in hipfire
pub const ENV_HIPFIRE_DEEPSEEK4_CACHE_TRACE: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DEEPSEEK4_CACHE_TRACE",
    description: "Runtime variable controlling deepseek4 cache trace in hipfire",
    source: "crates/hipfire-serving-core/src/generate_arch.rs:979",
};

/// `HIPFIRE_DEEPSEEK4_CHAT_RAW` — Enabled when set to 1
pub const ENV_HIPFIRE_DEEPSEEK4_CHAT_RAW: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DEEPSEEK4_CHAT_RAW",
    description: "Enabled when set to 1",
    source: "crates/hipfire-arch-deepseek4/examples/deepseek4_chat.rs:183",
};

/// `HIPFIRE_DEEPSEEK4_COMP_F16_WMMA` — Opt out via HIPFIRE_DEEPSEEK4_COMP_F16_WMMA=0
pub const ENV_HIPFIRE_DEEPSEEK4_COMP_F16_WMMA: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DEEPSEEK4_COMP_F16_WMMA",
    description: "Opt out via HIPFIRE_DEEPSEEK4_COMP_F16_WMMA=0",
    source: "crates/hipfire-arch-deepseek4/src/forward.rs:6577",
};

/// `HIPFIRE_DEEPSEEK4_COMP_ROPE_POS` — Runtime variable controlling deepseek4 comp rope pos in hipfire
pub const ENV_HIPFIRE_DEEPSEEK4_COMP_ROPE_POS: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DEEPSEEK4_COMP_ROPE_POS",
    description: "Runtime variable controlling deepseek4 comp rope pos in hipfire",
    source: "crates/hipfire-arch-deepseek4/src/forward.rs:5238",
};

/// `HIPFIRE_DEEPSEEK4_DSA_WMMA` — Head-batched f16-WMMA gathered DSA attention; f32 fallback on
pub const ENV_HIPFIRE_DEEPSEEK4_DSA_WMMA: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DEEPSEEK4_DSA_WMMA",
    description: "Head-batched f16-WMMA gathered DSA attention; f32 fallback on",
    source: "crates/hipfire-arch-deepseek4/src/forward.rs:7114",
};

/// `HIPFIRE_DEEPSEEK4_DUMP_PROMPT` — Runtime variable controlling deepseek4 dump prompt in hipfire
pub const ENV_HIPFIRE_DEEPSEEK4_DUMP_PROMPT: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DEEPSEEK4_DUMP_PROMPT",
    description: "Runtime variable controlling deepseek4 dump prompt in hipfire",
    source: "crates/hipfire-serving-core/src/generate_arch.rs:326",
};

/// `HIPFIRE_DEEPSEEK4_DUMP_STATE` — Selects behavior from recognized values
pub const ENV_HIPFIRE_DEEPSEEK4_DUMP_STATE: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DEEPSEEK4_DUMP_STATE",
    description: "Selects behavior from recognized values",
    source: "crates/hipfire-arch-deepseek4/src/forward.rs:203",
};

/// `HIPFIRE_DEEPSEEK4_DUMP_TOPK` — Gate 1 validation (2026-05-22): dump per-layer topk_indices to a
pub const ENV_HIPFIRE_DEEPSEEK4_DUMP_TOPK: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DEEPSEEK4_DUMP_TOPK",
    description: "Gate 1 validation (2026-05-22): dump per-layer topk_indices to a",
    source: "crates/hipfire-dispatch/src/pipeline/mod.rs:1072",
};

/// `HIPFIRE_DEEPSEEK4_EXPERT_LAYER_END` — Runtime variable controlling deepseek4 expert layer end in hipfire
pub const ENV_HIPFIRE_DEEPSEEK4_EXPERT_LAYER_END: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DEEPSEEK4_EXPERT_LAYER_END",
    description: "Runtime variable controlling deepseek4 expert layer end in hipfire",
    source: "crates/hipfire-arch-deepseek4/src/arch.rs:599",
};

/// `HIPFIRE_DEEPSEEK4_F32_TRACE` — Runtime variable controlling deepseek4 f32 trace in hipfire
pub const ENV_HIPFIRE_DEEPSEEK4_F32_TRACE: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DEEPSEEK4_F32_TRACE",
    description: "Runtime variable controlling deepseek4 f32 trace in hipfire",
    source: "crates/hipfire-arch-deepseek4/src/forward.rs:231",
};

/// `HIPFIRE_DEEPSEEK4_FUSED_UNSCATTER_SILU` — Measured perf at PP_BATCH=512 / 2.1k-tok prompt: -0.4% prefill —
pub const ENV_HIPFIRE_DEEPSEEK4_FUSED_UNSCATTER_SILU: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DEEPSEEK4_FUSED_UNSCATTER_SILU",
    description: "Measured perf at PP_BATCH=512 / 2.1k-tok prompt: -0.4% prefill —",
    source: "crates/hipfire-dispatch/src/pipeline/mod.rs:1163",
};

/// `HIPFIRE_DEEPSEEK4_GEN_TOKENS` — Runtime variable controlling deepseek4 gen tokens in hipfire
pub const ENV_HIPFIRE_DEEPSEEK4_GEN_TOKENS: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DEEPSEEK4_GEN_TOKENS",
    description: "Runtime variable controlling deepseek4 gen tokens in hipfire",
    source: "crates/hipfire-arch-deepseek4/examples/deepseek4_chat.rs:179",
};

/// `HIPFIRE_DEEPSEEK4_GRAPH` — Selects behavior from recognized values
pub const ENV_HIPFIRE_DEEPSEEK4_GRAPH: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DEEPSEEK4_GRAPH",
    description: "Selects behavior from recognized values",
    source: "crates/hipfire-arch-deepseek4/src/forward.rs:1568",
};

/// `HIPFIRE_DEEPSEEK4_HFQ4_WMMA` — HFQ4G256/Raw. WMMA route requires F16 input staging
pub const ENV_HIPFIRE_DEEPSEEK4_HFQ4_WMMA: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DEEPSEEK4_HFQ4_WMMA",
    description: "HFQ4G256/Raw. WMMA route requires F16 input staging",
    source: "crates/hipfire-arch-deepseek4/src/forward.rs:332",
};

/// `HIPFIRE_DEEPSEEK4_INDEXER_WMMA` — Runtime variable controlling deepseek4 indexer wmma in hipfire
pub const ENV_HIPFIRE_DEEPSEEK4_INDEXER_WMMA: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DEEPSEEK4_INDEXER_WMMA",
    description: "Runtime variable controlling deepseek4 indexer wmma in hipfire",
    source: "crates/hipfire-arch-deepseek4/src/forward.rs:6932",
};

/// `HIPFIRE_DEEPSEEK4_LOAD_MTP` — Runtime variable controlling deepseek4 load MTP in hipfire
pub const ENV_HIPFIRE_DEEPSEEK4_LOAD_MTP: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DEEPSEEK4_LOAD_MTP",
    description: "Runtime variable controlling deepseek4 load MTP in hipfire",
    source: "crates/hipfire-arch-deepseek4/src/arch.rs:1002",
};

/// `HIPFIRE_DEEPSEEK4_MAX_COMPRESS_POS` — ratio == 128: identity gather, no indexer. Per-batch n_compressed
pub const ENV_HIPFIRE_DEEPSEEK4_MAX_COMPRESS_POS: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DEEPSEEK4_MAX_COMPRESS_POS",
    description: "ratio == 128: identity gather, no indexer. Per-batch n_compressed",
    source: "crates/hipfire-arch-deepseek4/src/forward.rs:7014",
};

/// `HIPFIRE_DEEPSEEK4_MODEL` — Runtime variable controlling deepseek4 model in hipfire
pub const ENV_HIPFIRE_DEEPSEEK4_MODEL: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DEEPSEEK4_MODEL",
    description: "Runtime variable controlling deepseek4 model in hipfire",
    source: "crates/hipfire-arch-deepseek4/examples/deepseek4_chat.rs:176",
};

/// `HIPFIRE_DEEPSEEK4_MOE` — Layers 0..num_hash_layers use STATIC tid2eid routing per upstream DeepSeek V4
pub const ENV_HIPFIRE_DEEPSEEK4_MOE: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DEEPSEEK4_MOE",
    description: "Layers 0..num_hash_layers use STATIC tid2eid routing per upstream DeepSeek V4",
    source: "crates/hipfire-arch-deepseek4/src/forward.rs:7456",
};

/// `HIPFIRE_DEEPSEEK4_MOE_8W` — Lever 3 (HIPFIRE_DEEPSEEK4_MOE_8W=1): 8-warp variant (shares staged X
pub const ENV_HIPFIRE_DEEPSEEK4_MOE_8W: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DEEPSEEK4_MOE_8W",
    description: "Lever 3 (HIPFIRE_DEEPSEEK4_MOE_8W=1): 8-warp variant (shares staged X",
    source: "crates/hipfire-dispatch/src/pipeline/mod.rs:1112",
};

/// `HIPFIRE_DEEPSEEK4_MOE_CND` — Lever 2 (HIPFIRE_DEEPSEEK4_MOE_CND=1): cndmask dequant on the n16 4w
pub const ENV_HIPFIRE_DEEPSEEK4_MOE_CND: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DEEPSEEK4_MOE_CND",
    description: "Lever 2 (HIPFIRE_DEEPSEEK4_MOE_CND=1): cndmask dequant on the n16 4w",
    source: "crates/hipfire-dispatch/src/pipeline/mod.rs:1111",
};

/// `HIPFIRE_DEEPSEEK4_MOE_DETERMINISTIC` — Environment toggle value controls runtime behavior
pub const ENV_HIPFIRE_DEEPSEEK4_MOE_DETERMINISTIC: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DEEPSEEK4_MOE_DETERMINISTIC",
    description: "Environment toggle value controls runtime behavior",
    source: "crates/hipfire-dispatch/src/pipeline/mod.rs:1264",
};

/// `HIPFIRE_DEEPSEEK4_MOE_GROUPED` — Environment toggle value controls runtime behavior
pub const ENV_HIPFIRE_DEEPSEEK4_MOE_GROUPED: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DEEPSEEK4_MOE_GROUPED",
    description: "Environment toggle value controls runtime behavior",
    source: "crates/hipfire-dispatch/src/pipeline/mod.rs:1101",
};

/// `HIPFIRE_DEEPSEEK4_MOE_GROUPED_GATE` — HIPFIRE_DEEPSEEK4_MOE_GROUPED_GATE overrides the threshold for
pub const ENV_HIPFIRE_DEEPSEEK4_MOE_GROUPED_GATE: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DEEPSEEK4_MOE_GROUPED_GATE",
    description: "HIPFIRE_DEEPSEEK4_MOE_GROUPED_GATE overrides the threshold for",
    source: "crates/hipfire-dispatch/src/pipeline/mod.rs:1096",
};

/// `HIPFIRE_DEEPSEEK4_MOE_LLOYD_4W` — RECONCILED (#355 + #356): same 4w default + levers as gate_up above
pub const ENV_HIPFIRE_DEEPSEEK4_MOE_LLOYD_4W: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DEEPSEEK4_MOE_LLOYD_4W",
    description: "RECONCILED (#355 + #356): same 4w default + levers as gate_up above",
    source: "crates/hipfire-dispatch/src/pipeline/mod.rs:1104",
};

/// `HIPFIRE_DEEPSEEK4_MOE_MMQLOAD` — n32_env / cnd_env / eightw_env reused from the gate_up block above
pub const ENV_HIPFIRE_DEEPSEEK4_MOE_MMQLOAD: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DEEPSEEK4_MOE_MMQLOAD",
    description: "n32_env / cnd_env / eightw_env reused from the gate_up block above",
    source: "crates/hipfire-dispatch/src/pipeline/mod.rs:1113",
};

/// `HIPFIRE_DEEPSEEK4_MOE_N32` — Lever 1 (HIPFIRE_DEEPSEEK4_MOE_N32=1): N_TILE=32 tile-pairing on the
pub const ENV_HIPFIRE_DEEPSEEK4_MOE_N32: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DEEPSEEK4_MOE_N32",
    description: "Lever 1 (HIPFIRE_DEEPSEEK4_MOE_N32=1): N_TILE=32 tile-pairing on the",
    source: "crates/hipfire-dispatch/src/pipeline/mod.rs:1110",
};

/// `HIPFIRE_DEEPSEEK4_MOE_NOSYNC` — n32_env / cnd_env / eightw_env reused from the gate_up block above
pub const ENV_HIPFIRE_DEEPSEEK4_MOE_NOSYNC: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DEEPSEEK4_MOE_NOSYNC",
    description: "n32_env / cnd_env / eightw_env reused from the gate_up block above",
    source: "crates/hipfire-dispatch/src/pipeline/mod.rs:1114",
};

/// `HIPFIRE_DEEPSEEK4_MTP_ADDON` — Convention 1: append ".mtp-addon.hfq" (legacy)
pub const ENV_HIPFIRE_DEEPSEEK4_MTP_ADDON: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DEEPSEEK4_MTP_ADDON",
    description: "Convention 1: append \".mtp-addon.hfq\" (legacy)",
    source: "crates/hipfire-arch-deepseek4/src/arch.rs:619",
};

/// `HIPFIRE_DEEPSEEK4_MTP_HEAD_HC` — Runtime variable controlling deepseek4 MTP head hc in hipfire
pub const ENV_HIPFIRE_DEEPSEEK4_MTP_HEAD_HC: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DEEPSEEK4_MTP_HEAD_HC",
    description: "Runtime variable controlling deepseek4 MTP head hc in hipfire",
    source: "crates/hipfire-arch-deepseek4/src/forward.rs:86",
};

/// `HIPFIRE_DEEPSEEK4_MTP_SKIP_HEAD` — 3. Batched MTP fill — single pass through the MTP layer for all
pub const ENV_HIPFIRE_DEEPSEEK4_MTP_SKIP_HEAD: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DEEPSEEK4_MTP_SKIP_HEAD",
    description: "3. Batched MTP fill — single pass through the MTP layer for all",
    source: "crates/hipfire-arch-deepseek4/src/forward.rs:9012",
};

/// `HIPFIRE_DEEPSEEK4_POST_SCALE` — Runtime variable controlling deepseek4 post scale in hipfire
pub const ENV_HIPFIRE_DEEPSEEK4_POST_SCALE: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DEEPSEEK4_POST_SCALE",
    description: "Runtime variable controlling deepseek4 post scale in hipfire",
    source: "crates/hipfire-arch-deepseek4/src/forward.rs:8359",
};

/// `HIPFIRE_DEEPSEEK4_PP_BATCH` — on 2026-05-26). PP_BATCH sweep on the 2.1k-tok bench (3 trials/cell):
pub const ENV_HIPFIRE_DEEPSEEK4_PP_BATCH: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DEEPSEEK4_PP_BATCH",
    description: "on 2026-05-26). PP_BATCH sweep on the 2.1k-tok bench (3 trials/cell):",
    source: "crates/hipfire-serving-core/src/load.rs:1280",
};

/// `HIPFIRE_DEEPSEEK4_Q8_4W` — Opt out via HIPFIRE_DEEPSEEK4_Q8_4W=0 for diagnosis
pub const ENV_HIPFIRE_DEEPSEEK4_Q8_4W: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DEEPSEEK4_Q8_4W",
    description: "Opt out via HIPFIRE_DEEPSEEK4_Q8_4W=0 for diagnosis",
    source: "crates/hipfire-arch-deepseek4/src/forward.rs:296",
};

/// `HIPFIRE_DEEPSEEK4_Q8_WMMA` — bench_q8_wmma_variants. Opt-out via HIPFIRE_DEEPSEEK4_Q8_WMMA=0
pub const ENV_HIPFIRE_DEEPSEEK4_Q8_WMMA: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DEEPSEEK4_Q8_WMMA",
    description: "bench_q8_wmma_variants. Opt-out via HIPFIRE_DEEPSEEK4_Q8_WMMA=0",
    source: "crates/hipfire-arch-deepseek4/src/forward.rs:246",
};

/// `HIPFIRE_DEEPSEEK4_ROUTE_SCALE` — Runtime variable controlling deepseek4 route scale in hipfire
pub const ENV_HIPFIRE_DEEPSEEK4_ROUTE_SCALE: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DEEPSEEK4_ROUTE_SCALE",
    description: "Runtime variable controlling deepseek4 route scale in hipfire",
    source: "crates/hipfire-arch-deepseek4/src/forward.rs:7476",
};

/// `HIPFIRE_DEEPSEEK4_SEED` — Runtime variable controlling deepseek4 seed in hipfire
pub const ENV_HIPFIRE_DEEPSEEK4_SEED: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DEEPSEEK4_SEED",
    description: "Runtime variable controlling deepseek4 seed in hipfire",
    source: "crates/hipfire-serving-core/src/generate_arch.rs:554",
};

/// `HIPFIRE_DEEPSEEK4_SPEC_DECODE` — Priority: 1. legacy env var → 2. generic env var → 3. stored config → default
pub const ENV_HIPFIRE_DEEPSEEK4_SPEC_DECODE: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DEEPSEEK4_SPEC_DECODE",
    description: "Priority: 1. legacy env var → 2. generic env var → 3. stored config → default",
    source: "crates/hipfire-serving-core/src/generate_arch.rs:348",
};

/// `HIPFIRE_DEEPSEEK4_SPEC_K` — Runtime variable controlling deepseek4 spec k in hipfire
pub const ENV_HIPFIRE_DEEPSEEK4_SPEC_K: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DEEPSEEK4_SPEC_K",
    description: "Runtime variable controlling deepseek4 spec k in hipfire",
    source: "crates/hipfire-serving-core/src/generate_arch.rs:356",
};

/// `HIPFIRE_DEEPSEEK4_TEMP` — Runtime variable controlling deepseek4 temp in hipfire
pub const ENV_HIPFIRE_DEEPSEEK4_TEMP: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DEEPSEEK4_TEMP",
    description: "Runtime variable controlling deepseek4 temp in hipfire",
    source: "crates/hipfire-arch-deepseek4/examples/deepseek4_chat.rs:184",
};

/// `HIPFIRE_DEEPSEEK4_TOP_K` — for local deployment; we honor that as the default. Pure greedy
pub const ENV_HIPFIRE_DEEPSEEK4_TOP_K: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DEEPSEEK4_TOP_K",
    description: "for local deployment; we honor that as the default. Pure greedy",
    source: "crates/hipfire-serving-core/src/generate_arch.rs:550",
};

/// `HIPFIRE_DEEPSEEK4_UPLOAD_EXPERTS` — without them). Opt out with "HIPFIRE_DEEPSEEK4_UPLOAD_EXPERTS=0"
pub const ENV_HIPFIRE_DEEPSEEK4_UPLOAD_EXPERTS: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DEEPSEEK4_UPLOAD_EXPERTS",
    description: "without them). Opt out with \"HIPFIRE_DEEPSEEK4_UPLOAD_EXPERTS=0\"",
    source: "crates/hipfire-arch-deepseek4/src/arch.rs:595",
};

/// `HIPFIRE_DEEPSEEK4_WO_MULTIROW` — Q8_0 contract: plain (non-FWHT) input. Same layout
pub const ENV_HIPFIRE_DEEPSEEK4_WO_MULTIROW: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DEEPSEEK4_WO_MULTIROW",
    description: "Q8_0 contract: plain (non-FWHT) input. Same layout",
    source: "crates/hipfire-arch-deepseek4/src/forward.rs:7217",
};

/// `HIPFIRE_DEEPSEEK4_WO_Q8_WMMA` — Runtime variable controlling deepseek4 wo Q8 wmma in hipfire
pub const ENV_HIPFIRE_DEEPSEEK4_WO_Q8_WMMA: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DEEPSEEK4_WO_Q8_WMMA",
    description: "Runtime variable controlling deepseek4 wo Q8 wmma in hipfire",
    source: "crates/hipfire-rdna/src/dispatch/deepseek4.rs:2864",
};

/// `HIPFIRE_DELTANET_STATE` — Interprets "HIPFIRE_DELTANET_STATE" from environment to select behavior
pub const ENV_HIPFIRE_DELTANET_STATE: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DELTANET_STATE",
    description: "Interprets \"HIPFIRE_DELTANET_STATE\" from environment to select behavior",
    source: "crates/hipfire-runtime/examples/greedy_dump_top5.rs:266",
};

/// `HIPFIRE_DETERMINISTIC` — Enabled when set to 1
pub const ENV_HIPFIRE_DETERMINISTIC: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DETERMINISTIC",
    description: "Enabled when set to 1",
    source: "crates/hipfire-runtime/examples/pp_parity_chatml.rs:225",
};

/// `HIPFIRE_DEVICES` — Runtime variable controlling devices in hipfire
pub const ENV_HIPFIRE_DEVICES: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DEVICES",
    description: "Runtime variable controlling devices in hipfire",
    source: "crates/hipfire-runtime/src/config.rs:102",
};

/// `HIPFIRE_DFLASH_DRAFT` — Runtime variable controlling dflash draft in hipfire
pub const ENV_HIPFIRE_DFLASH_DRAFT: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DFLASH_DRAFT",
    description: "Runtime variable controlling dflash draft in hipfire",
    source: "crates/hipfire-server/src/routes/chat.rs:242",
};

/// `HIPFIRE_DFLASH_LOOP_BREAK` — Memory: HashSet<u64>, ≤ max_tokens / 12 entries (~125 for max=1500)
pub const ENV_HIPFIRE_DFLASH_LOOP_BREAK: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DFLASH_LOOP_BREAK",
    description: "Memory: HashSet<u64>, ≤ max_tokens / 12 entries (~125 for max=1500)",
    source: "crates/hipfire-runtime/examples/dflash_spec_demo.rs:1461",
};

/// `HIPFIRE_DFLASH_LOOP_BREAK_MAX_ESCALATIONS` — Runtime variable controlling dflash loop break max escalations in hipfire
pub const ENV_HIPFIRE_DFLASH_LOOP_BREAK_MAX_ESCALATIONS: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DFLASH_LOOP_BREAK_MAX_ESCALATIONS",
    description: "Runtime variable controlling dflash loop break max escalations in hipfire",
    source: "crates/hipfire-runtime/examples/dflash_spec_demo.rs:1490",
};

/// `HIPFIRE_DFLASH_LOOP_BREAK_RECOVERY` — Runtime variable controlling dflash loop break recovery in hipfire
pub const ENV_HIPFIRE_DFLASH_LOOP_BREAK_RECOVERY: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DFLASH_LOOP_BREAK_RECOVERY",
    description: "Runtime variable controlling dflash loop break recovery in hipfire",
    source: "crates/hipfire-runtime/examples/dflash_spec_demo.rs:1485",
};

/// `HIPFIRE_DFLASH_LOOP_BREAK_RP_MAX` — Runtime variable controlling dflash loop break rp max in hipfire
pub const ENV_HIPFIRE_DFLASH_LOOP_BREAK_RP_MAX: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DFLASH_LOOP_BREAK_RP_MAX",
    description: "Runtime variable controlling dflash loop break rp max in hipfire",
    source: "crates/hipfire-runtime/examples/dflash_spec_demo.rs:1481",
};

/// `HIPFIRE_DFLASH_LOOP_BREAK_RP_STEP` — Runtime variable controlling dflash loop break rp step in hipfire
pub const ENV_HIPFIRE_DFLASH_LOOP_BREAK_RP_STEP: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DFLASH_LOOP_BREAK_RP_STEP",
    description: "Runtime variable controlling dflash loop break rp step in hipfire",
    source: "crates/hipfire-runtime/examples/dflash_spec_demo.rs:1477",
};

/// `HIPFIRE_DFLASH_LOOP_BREAK_STOP_AFTER` — Runtime variable controlling dflash loop break stop after in hipfire
pub const ENV_HIPFIRE_DFLASH_LOOP_BREAK_STOP_AFTER: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DFLASH_LOOP_BREAK_STOP_AFTER",
    description: "Runtime variable controlling dflash loop break stop after in hipfire",
    source: "crates/hipfire-runtime/examples/dflash_spec_demo.rs:1473",
};

/// `HIPFIRE_DFLASH_LOOP_BREAK_TEMP` — Runtime variable controlling dflash loop break temp in hipfire
pub const ENV_HIPFIRE_DFLASH_LOOP_BREAK_TEMP: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DFLASH_LOOP_BREAK_TEMP",
    description: "Runtime variable controlling dflash loop break temp in hipfire",
    source: "crates/hipfire-runtime/examples/dflash_spec_demo.rs:1469",
};

/// `HIPFIRE_DFLASH_MODE` — Defaults to off when unset
pub const ENV_HIPFIRE_DFLASH_MODE: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DFLASH_MODE",
    description: "Defaults to off when unset",
    source: "crates/hipfire-runtime/src/config.rs:74",
};

/// `HIPFIRE_DFLASH_MOE_DRAFT_FFN_GRAPH` — Runtime variable controlling dflash moe draft ffn graph in hipfire
pub const ENV_HIPFIRE_DFLASH_MOE_DRAFT_FFN_GRAPH: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DFLASH_MOE_DRAFT_FFN_GRAPH",
    description: "Runtime variable controlling dflash moe draft ffn graph in hipfire",
    source: "crates/hipfire-arch-qwen35/src/speculative.rs:6771",
};

/// `HIPFIRE_DFLASH_MOE_VERIFY_GRAPH_LMHEAD` — Runtime variable controlling dflash moe verify graph lmhead in hipfire
pub const ENV_HIPFIRE_DFLASH_MOE_VERIFY_GRAPH_LMHEAD: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DFLASH_MOE_VERIFY_GRAPH_LMHEAD",
    description: "Runtime variable controlling dflash moe verify graph lmhead in hipfire",
    source: "crates/hipfire-arch-qwen35/src/speculative.rs:6142",
};

/// `HIPFIRE_DFLASH_NGRAM_BLOCK` — Enabled when set to 1
pub const ENV_HIPFIRE_DFLASH_NGRAM_BLOCK: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DFLASH_NGRAM_BLOCK",
    description: "Enabled when set to 1",
    source: "crates/hipfire-server/src/routes/chat.rs:181",
};

/// `HIPFIRE_DFLASH_Q8_LMHEAD_WMMA` — Selects behavior from recognized values
pub const ENV_HIPFIRE_DFLASH_Q8_LMHEAD_WMMA: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DFLASH_Q8_LMHEAD_WMMA",
    description: "Selects behavior from recognized values",
    source: "crates/hipfire-arch-qwen35/src/speculative.rs:130",
};

/// `HIPFIRE_DFLASH_ROLLBACK_COMPARE` — Used to configure runtime execution by explicitly setting "HIPFIRE_DFLASH_ROLLBACK_COMPARE"
pub const ENV_HIPFIRE_DFLASH_ROLLBACK_COMPARE: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DFLASH_ROLLBACK_COMPARE",
    description: "Used to configure runtime execution by explicitly setting \"HIPFIRE_DFLASH_ROLLBACK_COMPARE\"",
    source: "crates/hipfire-arch-qwen35/src/speculative.rs:12119",
};

/// `HIPFIRE_DFLASH_ROLLBACK_FA_RAW_ATOL` — Runtime variable controlling dflash rollback fa raw atol in hipfire
pub const ENV_HIPFIRE_DFLASH_ROLLBACK_FA_RAW_ATOL: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DFLASH_ROLLBACK_FA_RAW_ATOL",
    description: "Runtime variable controlling dflash rollback fa raw atol in hipfire",
    source: "crates/hipfire-arch-qwen35/src/speculative.rs:969",
};

/// `HIPFIRE_DFLASH_ROLLBACK_LOGIT_COMPARE_STEPS` — Used to configure runtime execution by explicitly setting "HIPFIRE_DFLASH_ROLLBACK_LOGIT_COMPARE_STEPS"
pub const ENV_HIPFIRE_DFLASH_ROLLBACK_LOGIT_COMPARE_STEPS: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DFLASH_ROLLBACK_LOGIT_COMPARE_STEPS",
    description: "Used to configure runtime execution by explicitly setting \"HIPFIRE_DFLASH_ROLLBACK_LOGIT_COMPARE_STEPS\"",
    source: "crates/hipfire-arch-qwen35/src/speculative.rs:12143",
};

/// `HIPFIRE_DFLASH_ROLLBACK_PREFIX_VERIFY` — Used to configure runtime execution by explicitly setting "HIPFIRE_DFLASH_ROLLBACK_PREFIX_VERIFY"
pub const ENV_HIPFIRE_DFLASH_ROLLBACK_PREFIX_VERIFY: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DFLASH_ROLLBACK_PREFIX_VERIFY",
    description: "Used to configure runtime execution by explicitly setting \"HIPFIRE_DFLASH_ROLLBACK_PREFIX_VERIFY\"",
    source: "crates/hipfire-arch-qwen35/src/speculative.rs:12047",
};

/// `HIPFIRE_DFLASH_ROLLBACK_SERIAL_REPLAY` — Used to configure runtime execution by explicitly setting "HIPFIRE_DFLASH_ROLLBACK_SERIAL_REPLAY"
pub const ENV_HIPFIRE_DFLASH_ROLLBACK_SERIAL_REPLAY: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DFLASH_ROLLBACK_SERIAL_REPLAY",
    description: "Used to configure runtime execution by explicitly setting \"HIPFIRE_DFLASH_ROLLBACK_SERIAL_REPLAY\"",
    source: "crates/hipfire-arch-qwen35/src/speculative.rs:12015",
};

/// `HIPFIRE_DFLASH_ROLLBACK_SERIAL_TAPE` — Used to configure runtime execution by explicitly setting "HIPFIRE_DFLASH_ROLLBACK_SERIAL_TAPE"
pub const ENV_HIPFIRE_DFLASH_ROLLBACK_SERIAL_TAPE: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DFLASH_ROLLBACK_SERIAL_TAPE",
    description: "Used to configure runtime execution by explicitly setting \"HIPFIRE_DFLASH_ROLLBACK_SERIAL_TAPE\"",
    source: "crates/hipfire-arch-qwen35/src/speculative.rs:12095",
};

/// `HIPFIRE_DFLASH_ROLLBACK_VERIFY_FRAMES` — Used to configure runtime execution by explicitly setting "HIPFIRE_DFLASH_ROLLBACK_VERIFY_FRAMES"
pub const ENV_HIPFIRE_DFLASH_ROLLBACK_VERIFY_FRAMES: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DFLASH_ROLLBACK_VERIFY_FRAMES",
    description: "Used to configure runtime execution by explicitly setting \"HIPFIRE_DFLASH_ROLLBACK_VERIFY_FRAMES\"",
    source: "crates/hipfire-arch-qwen35/src/speculative.rs:12071",
};

/// `HIPFIRE_DFLASH_ROLLBACK_X_IN_ATOL` — Parses "HIPFIRE_DFLASH_ROLLBACK_X_IN_ATOL" with fallback defaults
pub const ENV_HIPFIRE_DFLASH_ROLLBACK_X_IN_ATOL: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DFLASH_ROLLBACK_X_IN_ATOL",
    description: "Parses \"HIPFIRE_DFLASH_ROLLBACK_X_IN_ATOL\" with fallback defaults",
    source: "crates/hipfire-arch-qwen35/src/speculative.rs:977",
};

/// `HIPFIRE_DFLASH_SERIAL_QKVZA_SELF_COMPARE` — Runtime variable controlling dflash serial qKVza self compare in hipfire
pub const ENV_HIPFIRE_DFLASH_SERIAL_QKVZA_SELF_COMPARE: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DFLASH_SERIAL_QKVZA_SELF_COMPARE",
    description: "Runtime variable controlling dflash serial qKVza self compare in hipfire",
    source: "crates/hipfire-arch-qwen35/src/qwen35.rs:14342",
};

/// `HIPFIRE_DFLASH_SERIAL_TAPE_X_IN_COMPARE` — Runtime variable controlling dflash serial tape x in compare in hipfire
pub const ENV_HIPFIRE_DFLASH_SERIAL_TAPE_X_IN_COMPARE: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DFLASH_SERIAL_TAPE_X_IN_COMPARE",
    description: "Runtime variable controlling dflash serial tape x in compare in hipfire",
    source: "crates/hipfire-arch-qwen35/src/qwen35.rs:14347",
};

/// `HIPFIRE_DFLASH_SPEC_DEMO_BIN` — Runtime variable controlling dflash spec demo bin in hipfire
pub const ENV_HIPFIRE_DFLASH_SPEC_DEMO_BIN: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DFLASH_SPEC_DEMO_BIN",
    description: "Runtime variable controlling dflash spec demo bin in hipfire",
    source: "crates/hipfire-eval/src/lib.rs:1147",
};

/// `HIPFIRE_DFLASH_TRACE_EXPECTED_TOKEN` — Runtime variable controlling dflash trace expected token in hipfire
pub const ENV_HIPFIRE_DFLASH_TRACE_EXPECTED_TOKEN: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DFLASH_TRACE_EXPECTED_TOKEN",
    description: "Runtime variable controlling dflash trace expected token in hipfire",
    source: "crates/hipfire-runtime/examples/dflash_spec_demo.rs:165",
};

/// `HIPFIRE_DFLASH_TRACE_POSITION` — Runtime variable controlling dflash trace position in hipfire
pub const ENV_HIPFIRE_DFLASH_TRACE_POSITION: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DFLASH_TRACE_POSITION",
    description: "Runtime variable controlling dflash trace position in hipfire",
    source: "crates/hipfire-runtime/examples/dflash_spec_demo.rs:159",
};

/// `HIPFIRE_DFLASH_TRACE_TOKEN_INDEX` — Runtime variable controlling dflash trace token index in hipfire
pub const ENV_HIPFIRE_DFLASH_TRACE_TOKEN_INDEX: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DFLASH_TRACE_TOKEN_INDEX",
    description: "Runtime variable controlling dflash trace token index in hipfire",
    source: "crates/hipfire-runtime/examples/dflash_spec_demo.rs:1832",
};

/// `HIPFIRE_DIAG` — Diagnostic: quant-vs-bf16 fidelity per layer (target is the bf16 FFN)
pub const ENV_HIPFIRE_DIAG: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DIAG",
    description: "Diagnostic: quant-vs-bf16 fidelity per layer (target is the bf16 FFN)",
    source: "crates/hipfire-train/examples/qwen35_mlp_norm_recovery.rs:284",
};

/// `HIPFIRE_DIFFUSION_CPU_REFERENCE` — Runtime variable controlling diffusion cpu reference in hipfire
pub const ENV_HIPFIRE_DIFFUSION_CPU_REFERENCE: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DIFFUSION_CPU_REFERENCE",
    description: "Runtime variable controlling diffusion cpu reference in hipfire",
    source: "crates/hipfire-diffusion/src/lib.rs:127",
};

/// `HIPFIRE_DIFFUSION_DUMP_DIR` — Runtime variable controlling diffusion dump dir in hipfire
pub const ENV_HIPFIRE_DIFFUSION_DUMP_DIR: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DIFFUSION_DUMP_DIR",
    description: "Runtime variable controlling diffusion dump dir in hipfire",
    source: "crates/hipfire-diffusion/src/lib.rs:656",
};

/// `HIPFIRE_DIFFUSION_LAYER_RUNG` — Selects behavior from recognized values
pub const ENV_HIPFIRE_DIFFUSION_LAYER_RUNG: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DIFFUSION_LAYER_RUNG",
    description: "Selects behavior from recognized values",
    source: "crates/hipfire-diffusion/src/denoise.rs:289",
};

/// `HIPFIRE_DIR` — Runtime variable controlling dir in hipfire
pub const ENV_HIPFIRE_DIR: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DIR",
    description: "Runtime variable controlling dir in hipfire",
    source: "crates/hipfire-serving-core/src/generate_vl.rs:1199",
};

/// `HIPFIRE_DN_STATE_EF` — Runtime variable controlling dn state ef in hipfire
pub const ENV_HIPFIRE_DN_STATE_EF: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DN_STATE_EF",
    description: "Runtime variable controlling dn state ef in hipfire",
    source: "crates/hipfire-arch-qwen35/src/qwen35.rs:1432",
};

/// `HIPFIRE_DN_STATE_FP32_BELOW` — Used to configure runtime execution by explicitly setting "HIPFIRE_DN_STATE_FP32_BELOW"
pub const ENV_HIPFIRE_DN_STATE_FP32_BELOW: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DN_STATE_FP32_BELOW",
    description:
        "Used to configure runtime execution by explicitly setting \"HIPFIRE_DN_STATE_FP32_BELOW\"",
    source: "crates/hipfire-arch-qwen35/src/qwen35.rs:30430",
};

/// `HIPFIRE_DOT2_GEMV` — Interprets "HIPFIRE_DOT2_GEMV" from environment to select behavior
pub const ENV_HIPFIRE_DOT2_GEMV: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DOT2_GEMV",
    description: "Interprets \"HIPFIRE_DOT2_GEMV\" from environment to select behavior",
    source: "crates/hipfire-rdna/src/feature_flags.rs:192",
};

/// `HIPFIRE_DOTS_OCR_BF16_RESIDUAL` — HF cast x to bf16 at vision forward entry (modeling_dots_vision.py
pub const ENV_HIPFIRE_DOTS_OCR_BF16_RESIDUAL: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DOTS_OCR_BF16_RESIDUAL",
    description: "HF cast x to bf16 at vision forward entry (modeling_dots_vision.py",
    source: "crates/hipfire-arch-dots-ocr/src/dots_ocr.rs:1127",
};

/// `HIPFIRE_DOTS_OCR_DUMP_DIR` — HIPFIRE_DOTS_OCR_DUMP_DIR=<path>: dump full per-stage tensor
pub const ENV_HIPFIRE_DOTS_OCR_DUMP_DIR: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DOTS_OCR_DUMP_DIR",
    description: "HIPFIRE_DOTS_OCR_DUMP_DIR=<path>: dump full per-stage tensor",
    source: "crates/hipfire-arch-dots-ocr/src/dots_ocr.rs:1029",
};

/// `HIPFIRE_DOTS_OCR_TRACE` — HIPFIRE_DOTS_OCR_TRACE=1: sync after every step + print probe so
pub const ENV_HIPFIRE_DOTS_OCR_TRACE: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DOTS_OCR_TRACE",
    description: "HIPFIRE_DOTS_OCR_TRACE=1: sync after every step + print probe so",
    source: "crates/hipfire-arch-dots-ocr/src/dots_ocr.rs:1157",
};

/// `HIPFIRE_DPM_WARMUP_SECS` — Runtime variable controlling dpm warmup secs in hipfire
pub const ENV_HIPFIRE_DPM_WARMUP_SECS: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DPM_WARMUP_SECS",
    description: "Runtime variable controlling dpm warmup secs in hipfire",
    source: "crates/hipfire-runtime/src/speed_bench.rs:148",
};

/// `HIPFIRE_DRAFT_F16` — Enabled by default; set to 0 to disable
pub const ENV_HIPFIRE_DRAFT_F16: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DRAFT_F16",
    description: "Enabled by default; set to 0 to disable",
    source: "crates/hipfire-runtime/src/config.rs:75",
};

/// `HIPFIRE_DRAFT_GEMM_DUMP` — Enabled when set to 1
pub const ENV_HIPFIRE_DRAFT_GEMM_DUMP: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DRAFT_GEMM_DUMP",
    description: "Enabled when set to 1",
    source: "crates/hipfire-runtime/src/config.rs:76",
};

/// `HIPFIRE_DRAFT_SUBPHASE` — Enabled when set to 1
pub const ENV_HIPFIRE_DRAFT_SUBPHASE: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DRAFT_SUBPHASE",
    description: "Enabled when set to 1",
    source: "crates/hipfire-runtime/src/config.rs:77",
};

/// `HIPFIRE_DSPARK` — Disabled when set to 0
pub const ENV_HIPFIRE_DSPARK: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DSPARK",
    description: "Disabled when set to 0",
    source: "crates/hipfire-serving-core/src/load.rs:3271",
};

/// `HIPFIRE_DSPARK_ADAPTIVE_BLOCK` — Default-on; HIPFIRE_DSPARK_ADAPTIVE_BLOCK=0 opts out (fixed block == today)
pub const ENV_HIPFIRE_DSPARK_ADAPTIVE_BLOCK: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DSPARK_ADAPTIVE_BLOCK",
    description: "Default-on; HIPFIRE_DSPARK_ADAPTIVE_BLOCK=0 opts out (fixed block == today)",
    source: "crates/hipfire-specdecode-dspark/src/dspark_core.rs:1426",
};

/// `HIPFIRE_DSPARK_HFQ4_WMMA` — Runtime variable controlling dspark hfQ4 wmma in hipfire
pub const ENV_HIPFIRE_DSPARK_HFQ4_WMMA: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DSPARK_HFQ4_WMMA",
    description: "Runtime variable controlling dspark hfQ4 wmma in hipfire",
    source: "crates/hipfire-specdecode-dspark/src/dspark_core.rs:463",
};

/// `HIPFIRE_DSPARK_PROFILE` — Runtime variable controlling dspark profile in hipfire
pub const ENV_HIPFIRE_DSPARK_PROFILE: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DSPARK_PROFILE",
    description: "Runtime variable controlling dspark profile in hipfire",
    source: "crates/hipfire-specdecode-dspark/src/dspark_core.rs:25",
};

/// `HIPFIRE_DSPARK_Q8_4W` — Environment toggle value controls runtime behavior
pub const ENV_HIPFIRE_DSPARK_Q8_4W: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DSPARK_Q8_4W",
    description: "Environment toggle value controls runtime behavior",
    source: "crates/hipfire-specdecode-dspark/src/dspark_core.rs:420",
};

/// `HIPFIRE_DSPARK_Q8_WMMA` — Runtime variable controlling dspark Q8 wmma in hipfire
pub const ENV_HIPFIRE_DSPARK_Q8_WMMA: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DSPARK_Q8_WMMA",
    description: "Runtime variable controlling dspark Q8 wmma in hipfire",
    source: "crates/hipfire-specdecode-dspark/src/dspark_core.rs:391",
};

/// `HIPFIRE_DTOH_DUMP` — Enabled when set to 1
pub const ENV_HIPFIRE_DTOH_DUMP: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DTOH_DUMP",
    description: "Enabled when set to 1",
    source: "crates/hip-bridge/src/ffi.rs:1060",
};

/// `HIPFIRE_DUMMY_GENERATE_DELAY_MS` — Parses "HIPFIRE_DUMMY_GENERATE_DELAY_MS" with fallback defaults
pub const ENV_HIPFIRE_DUMMY_GENERATE_DELAY_MS: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DUMMY_GENERATE_DELAY_MS",
    description: "Parses \"HIPFIRE_DUMMY_GENERATE_DELAY_MS\" with fallback defaults",
    source: "crates/hipfire-serving-core/src/dummy.rs:137",
};

/// `HIPFIRE_DUMMY_PREFILL_DELAY_MS` — Parses "HIPFIRE_DUMMY_PREFILL_DELAY_MS" with fallback defaults
pub const ENV_HIPFIRE_DUMMY_PREFILL_DELAY_MS: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DUMMY_PREFILL_DELAY_MS",
    description: "Parses \"HIPFIRE_DUMMY_PREFILL_DELAY_MS\" with fallback defaults",
    source: "crates/hipfire-serving-core/src/dummy.rs:126",
};

/// `HIPFIRE_DUMP_COND` — Debug hook: HIPFIRE_DUMP_COND prints stats for the text conditioning
pub const ENV_HIPFIRE_DUMP_COND: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DUMP_COND",
    description: "Debug hook: HIPFIRE_DUMP_COND prints stats for the text conditioning",
    source: "crates/hipfire-diffusion/src/pipeline_generate.rs:290",
};

/// `HIPFIRE_DUMP_HIDDEN` — DIAG: dump router logits before softmax (mirrors qwen35 HIPFIRE_DUMP_HIDDEN)
pub const ENV_HIPFIRE_DUMP_HIDDEN: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DUMP_HIDDEN",
    description: "DIAG: dump router logits before softmax (mirrors qwen35 HIPFIRE_DUMP_HIDDEN)",
    source: "crates/hipfire-dispatch/src/pipeline/mod.rs:386",
};

/// `HIPFIRE_DUMP_HIDDEN_ALL` — Activation-capture mode (HIPFIRE_DUMP_HIDDEN_ALL=1): dump EVERY row for a
pub const ENV_HIPFIRE_DUMP_HIDDEN_ALL: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DUMP_HIDDEN_ALL",
    description: "Activation-capture mode (HIPFIRE_DUMP_HIDDEN_ALL=1): dump EVERY row for a",
    source: "crates/hipfire-arch-qwen35/src/qwen35.rs:16839",
};

/// `HIPFIRE_DUMP_HIDDEN_ALLLAYERS` — - HIPFIRE_DUMP_HIDDEN_ALLLAYERS=1: capture EVERY layer to per-layer
pub const ENV_HIPFIRE_DUMP_HIDDEN_ALLLAYERS: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DUMP_HIDDEN_ALLLAYERS",
    description: "- HIPFIRE_DUMP_HIDDEN_ALLLAYERS=1: capture EVERY layer to per-layer",
    source: "crates/hipfire-arch-qwen35/src/qwen35.rs:16846",
};

/// `HIPFIRE_DUMP_HIDDEN_LAYER` — Runtime variable controlling dump hidden layer in hipfire
pub const ENV_HIPFIRE_DUMP_HIDDEN_LAYER: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DUMP_HIDDEN_LAYER",
    description: "Runtime variable controlling dump hidden layer in hipfire",
    source: "crates/hipfire-arch-qwen35/src/qwen35.rs:16850",
};

/// `HIPFIRE_DUMP_HIDDEN_POS` — Runtime variable controlling dump hidden pos in hipfire
pub const ENV_HIPFIRE_DUMP_HIDDEN_POS: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DUMP_HIDDEN_POS",
    description: "Runtime variable controlling dump hidden pos in hipfire",
    source: "crates/hipfire-arch-qwen35/src/qwen35.rs:16878",
};

/// `HIPFIRE_DUMP_LATENT` — Debug hook: when HIPFIRE_DUMP_LATENT names a path, write the final latent
pub const ENV_HIPFIRE_DUMP_LATENT: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DUMP_LATENT",
    description: "Debug hook: when HIPFIRE_DUMP_LATENT names a path, write the final latent",
    source: "crates/hipfire-diffusion/src/lib.rs:1139",
};

/// `HIPFIRE_DUMP_REQUEST` — a strace. Off by default — gigantic for typical agent prompts
pub const ENV_HIPFIRE_DUMP_REQUEST: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_DUMP_REQUEST",
    description: "a strace. Off by default — gigantic for typical agent prompts",
    source: "crates/hipfire-server/src/routes/chat.rs:85",
};

/// `HIPFIRE_EMIT_TOKEN_IDS` — or "\" in it would corrupt the line, breaking the client's JSONL
pub const ENV_HIPFIRE_EMIT_TOKEN_IDS: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_EMIT_TOKEN_IDS",
    description: "or \"\\\" in it would corrupt the line, breaking the client's JSONL",
    source: "crates/hipfire-serving-core/src/events.rs:111",
};

/// `HIPFIRE_EP_DECODE_TIMING` — 2. Per-layer EP program (Attend replicated; Moe all-reduce-EP'd)
pub const ENV_HIPFIRE_EP_DECODE_TIMING: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_EP_DECODE_TIMING",
    description: "2. Per-layer EP program (Attend replicated; Moe all-reduce-EP'd)",
    source: "crates/hipfire-arch-minimax/src/forward.rs:1697",
};

/// `HIPFIRE_EP_DUMP_IDX` — top-k chaos. HIPFIRE_EP_DUMP_IDX=1 to enable
pub const ENV_HIPFIRE_EP_DUMP_IDX: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_EP_DUMP_IDX",
    description: "top-k chaos. HIPFIRE_EP_DUMP_IDX=1 to enable",
    source: "crates/hipfire-arch-deepseek4/src/forward.rs:2432",
};

/// `HIPFIRE_EP_DUMP_POS` — Divergence-localization dump: HIPFIRE_EP_DUMP_POS="0,64,...,302" prints a
pub const ENV_HIPFIRE_EP_DUMP_POS: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_EP_DUMP_POS",
    description: "Divergence-localization dump: HIPFIRE_EP_DUMP_POS=\"0,64,...,302\" prints a",
    source: "crates/hipfire-arch-deepseek4/src/forward.rs:2367",
};

/// `HIPFIRE_EP_KV_SEQ` — Runtime variable controlling ep KV seq in hipfire
pub const ENV_HIPFIRE_EP_KV_SEQ: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_EP_KV_SEQ",
    description: "Runtime variable controlling ep KV seq in hipfire",
    source: "crates/hipfire-runtime/examples/ep_decode_parity.rs:92",
};

/// `HIPFIRE_EP_PEER_ALLREDUCE` — RCCL with HIPFIRE_EP_PEER_ALLREDUCE=0. The peer temps live in Gpus (shared
pub const ENV_HIPFIRE_EP_PEER_ALLREDUCE: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_EP_PEER_ALLREDUCE",
    description: "RCCL with HIPFIRE_EP_PEER_ALLREDUCE=0. The peer temps live in Gpus (shared",
    source: "crates/hipfire-arch-qwen35/src/qwen35.rs:28070",
};

/// `HIPFIRE_EP_PEER_ALLREDUCE_DECODE` — Environment toggle value controls runtime behavior
pub const ENV_HIPFIRE_EP_PEER_ALLREDUCE_DECODE: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_EP_PEER_ALLREDUCE_DECODE",
    description: "Environment toggle value controls runtime behavior",
    source: "crates/hipfire-runtime/src/ep.rs:135",
};

/// `HIPFIRE_EP_PREFILL` — Prefill mode: HIPFIRE_EP_PREFILL=batched → WMMA batched prefill EP (E6b)
pub const ENV_HIPFIRE_EP_PREFILL: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_EP_PREFILL",
    description: "Prefill mode: HIPFIRE_EP_PREFILL=batched → WMMA batched prefill EP (E6b)",
    source: "crates/hipfire-runtime/examples/ep_decode_parity.rs:119",
};

/// `HIPFIRE_EP_PREFILL_TIMING` — Gpus::all_reduce_sum_f32_peer (direct P2P copy + local add), which is ~1 ms
pub const ENV_HIPFIRE_EP_PREFILL_TIMING: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_EP_PREFILL_TIMING",
    description: "Gpus::all_reduce_sum_f32_peer (direct P2P copy + local add), which is ~1 ms",
    source: "crates/hipfire-arch-qwen35/src/qwen35.rs:28063",
};

/// `HIPFIRE_EP_SKIP_ALLREDUCE` — Gpus::all_reduce_sum_f32_peer (direct P2P copy + local add), which is ~1 ms
pub const ENV_HIPFIRE_EP_SKIP_ALLREDUCE: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_EP_SKIP_ALLREDUCE",
    description: "Gpus::all_reduce_sum_f32_peer (direct P2P copy + local add), which is ~1 ms",
    source: "crates/hipfire-arch-qwen35/src/qwen35.rs:28064",
};

/// `HIPFIRE_EVAL_DATASET_MIRROR` — Runtime variable controlling eval dataset mirror in hipfire
pub const ENV_HIPFIRE_EVAL_DATASET_MIRROR: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_EVAL_DATASET_MIRROR",
    description: "Runtime variable controlling eval dataset mirror in hipfire",
    source: "crates/hipfire-eval/src/datasets.rs:203",
};

/// `HIPFIRE_EVAL_EVIDENCE_DIR` — Runtime variable controlling eval evidence dir in hipfire
pub const ENV_HIPFIRE_EVAL_EVIDENCE_DIR: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_EVAL_EVIDENCE_DIR",
    description: "Runtime variable controlling eval evidence dir in hipfire",
    source: "crates/hipfire-runtime/examples/run.rs:253",
};

/// `HIPFIRE_EVAL_KLDREF` — Runtime variable controlling eval kldref in hipfire
pub const ENV_HIPFIRE_EVAL_KLDREF: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_EVAL_KLDREF",
    description: "Runtime variable controlling eval kldref in hipfire",
    source: "crates/hipfire-eval/src/executor_examples.rs:678",
};

/// `HIPFIRE_EVAL_PERPLEXITY_CORPUS` — Runtime variable controlling eval perplexity corpus in hipfire
pub const ENV_HIPFIRE_EVAL_PERPLEXITY_CORPUS: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_EVAL_PERPLEXITY_CORPUS",
    description: "Runtime variable controlling eval perplexity corpus in hipfire",
    source: "crates/hipfire-eval/src/run.rs:175",
};

/// `HIPFIRE_EVAL_PERPLEXITY_CTX` — Parses "HIPFIRE_EVAL_PERPLEXITY_CTX" with fallback defaults
pub const ENV_HIPFIRE_EVAL_PERPLEXITY_CTX: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_EVAL_PERPLEXITY_CTX",
    description: "Parses \"HIPFIRE_EVAL_PERPLEXITY_CTX\" with fallback defaults",
    source: "crates/hipfire-eval/src/executor_examples.rs:764",
};

/// `HIPFIRE_EXPERIMENTAL_BUDGET_ALERT` — cache so the model "sees" them as part of its own trajectory,
pub const ENV_HIPFIRE_EXPERIMENTAL_BUDGET_ALERT: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_EXPERIMENTAL_BUDGET_ALERT",
    description: "cache so the model \"sees\" them as part of its own trajectory,",
    source: "crates/hipfire-daemon/src/main.rs:3778",
};

/// `HIPFIRE_FILES_STATE_MAX` — Parses "HIPFIRE_FILES_STATE_MAX" with fallback defaults
pub const ENV_HIPFIRE_FILES_STATE_MAX: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_FILES_STATE_MAX",
    description: "Parses \"HIPFIRE_FILES_STATE_MAX\" with fallback defaults",
    source: "crates/hipfire-server/src/routes/files.rs:174",
};

/// `HIPFIRE_FLASH_PARTIALS_BATCH` — Parses "HIPFIRE_FLASH_PARTIALS_BATCH" with fallback defaults
pub const ENV_HIPFIRE_FLASH_PARTIALS_BATCH: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_FLASH_PARTIALS_BATCH",
    description: "Parses \"HIPFIRE_FLASH_PARTIALS_BATCH\" with fallback defaults",
    source: "crates/hipfire-runtime/src/config.rs:87",
};

/// `HIPFIRE_FORCE_GENERIC` — Runtime variable controlling force generic in hipfire
pub const ENV_HIPFIRE_FORCE_GENERIC: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_FORCE_GENERIC",
    description: "Runtime variable controlling force generic in hipfire",
    source: "crates/hipfire-rdna/src/feature_flags.rs:325",
};

/// `HIPFIRE_FORCE_UNFUSED` — Interpreter Phase 2a
pub const ENV_HIPFIRE_FORCE_UNFUSED: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_FORCE_UNFUSED",
    description: "Interpreter Phase 2a",
    source: "crates/hipfire-rdna/src/feature_flags.rs:321",
};

/// `HIPFIRE_FORWARD_LOWERED` — Enabled by default; set to 0 to disable
pub const ENV_HIPFIRE_FORWARD_LOWERED: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_FORWARD_LOWERED",
    description: "Enabled by default; set to 0 to disable",
    source: "crates/hipfire-arch-qwen35/src/qwen35.rs:27728",
};

/// `HIPFIRE_FP16` — escape hatch to the LA qkvza projection while debugging DFlash
pub const ENV_HIPFIRE_FP16: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_FP16",
    description: "escape hatch to the LA qkvza projection while debugging DFlash",
    source: "crates/hipfire-runtime/examples/test_wmma_correctness.rs:115",
};

/// `HIPFIRE_FP8_WMMA` — Interprets "HIPFIRE_FP8_WMMA" from environment to select behavior
pub const ENV_HIPFIRE_FP8_WMMA: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_FP8_WMMA",
    description: "Interprets \"HIPFIRE_FP8_WMMA\" from environment to select behavior",
    source: "crates/hipfire-rdna/src/feature_flags.rs:191",
};

/// `HIPFIRE_FUSED_HFQ4_2ROW_GFX1151` — Selects behavior from recognized values
pub const ENV_HIPFIRE_FUSED_HFQ4_2ROW_GFX1151: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_FUSED_HFQ4_2ROW_GFX1151",
    description: "Selects behavior from recognized values",
    source: "crates/hipfire-rdna/src/dispatch/mod.rs:3049",
};

/// `HIPFIRE_GATE_UP_VARIANT` — Runtime variable controlling gate up variant in hipfire
pub const ENV_HIPFIRE_GATE_UP_VARIANT: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_GATE_UP_VARIANT",
    description: "Runtime variable controlling gate up variant in hipfire",
    source: "crates/hipfire-rdna/src/feature_flags.rs:246",
};

/// `HIPFIRE_GDN_Q8_REG_GFX1151` — HIPFIRE_GDN_Q8_REG_GFX1151=1 enables the gfx1151 register-state
pub const ENV_HIPFIRE_GDN_Q8_REG_GFX1151: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_GDN_Q8_REG_GFX1151",
    description: "HIPFIRE_GDN_Q8_REG_GFX1151=1 enables the gfx1151 register-state",
    source: "crates/hipfire-rdna/src/dispatch/mod.rs:3604",
};

/// `HIPFIRE_GEMMA3_CALIB_IMATRIX_ONLY` — AWQ-style calibration mode: with no K×K outer-product or multi-GB Hessian
pub const ENV_HIPFIRE_GEMMA3_CALIB_IMATRIX_ONLY: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_GEMMA3_CALIB_IMATRIX_ONLY",
    description: "AWQ-style calibration mode: with no K×K outer-product or multi-GB Hessian",
    source: "crates/hipfire-arch-gemma3/src/calibration.rs:258",
};

/// `HIPFIRE_GEMMA3_CALIB_LAYERS_PER_PASS` — Runtime variable controlling gemma3 calib layers per pass in hipfire
pub const ENV_HIPFIRE_GEMMA3_CALIB_LAYERS_PER_PASS: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_GEMMA3_CALIB_LAYERS_PER_PASS",
    description: "Runtime variable controlling gemma3 calib layers per pass in hipfire",
    source: "crates/hipfire-arch-gemma3/src/calibration.rs:88",
};

/// `HIPFIRE_GEMMA3_CALIB_MICROBATCH` — Runtime variable controlling gemma3 calib microbatch in hipfire
pub const ENV_HIPFIRE_GEMMA3_CALIB_MICROBATCH: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_GEMMA3_CALIB_MICROBATCH",
    description: "Runtime variable controlling gemma3 calib microbatch in hipfire",
    source: "crates/hipfire-arch-gemma3/src/calibration.rs:120",
};

/// `HIPFIRE_GEMMA3_CALIB_NO_BATCH` — Enabled when set to 1
pub const ENV_HIPFIRE_GEMMA3_CALIB_NO_BATCH: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_GEMMA3_CALIB_NO_BATCH",
    description: "Enabled when set to 1",
    source: "crates/hipfire-arch-gemma3/src/calibration.rs:111",
};

/// `HIPFIRE_GEMMA3_NO_BATCHED_PREFILL` — Runtime variable controlling gemma3 no batched prefill in hipfire
pub const ENV_HIPFIRE_GEMMA3_NO_BATCHED_PREFILL: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_GEMMA3_NO_BATCHED_PREFILL",
    description: "Runtime variable controlling gemma3 no batched prefill in hipfire",
    source: "crates/hipfire-arch-gemma3/src/arch.rs:100",
};

/// `HIPFIRE_GEMMA3_PREFILL_MICROBATCH` — Runtime variable controlling gemma3 prefill microbatch in hipfire
pub const ENV_HIPFIRE_GEMMA3_PREFILL_MICROBATCH: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_GEMMA3_PREFILL_MICROBATCH",
    description: "Runtime variable controlling gemma3 prefill microbatch in hipfire",
    source: "crates/hipfire-arch-gemma3/src/arch.rs:106",
};

/// `HIPFIRE_GEMMA3_VISION_NOWMMA` — Runtime variable controlling gemma3 vision nowmma in hipfire
pub const ENV_HIPFIRE_GEMMA3_VISION_NOWMMA: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_GEMMA3_VISION_NOWMMA",
    description: "Runtime variable controlling gemma3 vision nowmma in hipfire",
    source: "crates/hipfire-arch-gemma3-vl/src/forward.rs:204",
};

/// `HIPFIRE_GEMM_DUMP` — Enabled when set to 1
pub const ENV_HIPFIRE_GEMM_DUMP: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_GEMM_DUMP",
    description: "Enabled when set to 1",
    source: "crates/hipfire-rdna/src/feature_flags.rs:292",
};

/// `HIPFIRE_GEMV_ROWS` — Runtime variable controlling gemv rows in hipfire
pub const ENV_HIPFIRE_GEMV_ROWS: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_GEMV_ROWS",
    description: "Runtime variable controlling gemv rows in hipfire",
    source: "crates/hipfire-rdna/src/feature_flags.rs:172",
};

/// `HIPFIRE_GEN` — Runtime variable controlling gen in hipfire
pub const ENV_HIPFIRE_GEN: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_GEN",
    description: "Runtime variable controlling gen in hipfire",
    source: "crates/hipfire-runtime/examples/a3b_multiturn_oneshot.rs:46",
};

/// `HIPFIRE_GENERATE_BATCH_PREFILL_DEBUG_SAMPLE` — Runtime variable controlling generate batch prefill debug sample in hipfire
pub const ENV_HIPFIRE_GENERATE_BATCH_PREFILL_DEBUG_SAMPLE: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_GENERATE_BATCH_PREFILL_DEBUG_SAMPLE",
    description: "Runtime variable controlling generate batch prefill debug sample in hipfire",
    source: "crates/hipfire-serving-core/src/qwen35_prefill.rs:1645",
};

/// `HIPFIRE_GFX942_GEMV_V3` — Interprets "HIPFIRE_GFX942_GEMV_V3" from environment to select behavior
pub const ENV_HIPFIRE_GFX942_GEMV_V3: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_GFX942_GEMV_V3",
    description: "Interprets \"HIPFIRE_GFX942_GEMV_V3\" from environment to select behavior",
    source: "crates/hipfire-rdna/src/feature_flags.rs:248",
};

/// `HIPFIRE_GFX942_MFMA_PREFILL` — Interprets "HIPFIRE_GFX942_MFMA_PREFILL" from environment to select behavior
pub const ENV_HIPFIRE_GFX942_MFMA_PREFILL: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_GFX942_MFMA_PREFILL",
    description: "Interprets \"HIPFIRE_GFX942_MFMA_PREFILL\" from environment to select behavior",
    source: "crates/hipfire-rdna/src/feature_flags.rs:251",
};

/// `HIPFIRE_GFX942_RMSNORM_SPLIT` — Environment toggle value controls runtime behavior
pub const ENV_HIPFIRE_GFX942_RMSNORM_SPLIT: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_GFX942_RMSNORM_SPLIT",
    description: "Environment toggle value controls runtime behavior",
    source: "crates/hipfire-rdna/src/feature_flags.rs:250",
};

/// `HIPFIRE_GPTQ_DAMPING` — Inject env override since the quantizer reads it at fn entry
pub const ENV_HIPFIRE_GPTQ_DAMPING: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_GPTQ_DAMPING",
    description: "Inject env override since the quantizer reads it at fn entry",
    source: "crates/hipfire-quantize/src/main.rs:10305",
};

/// `HIPFIRE_GPU_SLAB_LOAD` — Runtime variable controlling gpu slab load in hipfire
pub const ENV_HIPFIRE_GPU_SLAB_LOAD: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_GPU_SLAB_LOAD",
    description: "Runtime variable controlling gpu slab load in hipfire",
    source: "crates/hipfire-arch-qwen35/src/qwen35.rs:5284",
};

/// `HIPFIRE_GPU_SLAB_MIB` — Parses "HIPFIRE_GPU_SLAB_MIB" with fallback defaults
pub const ENV_HIPFIRE_GPU_SLAB_MIB: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_GPU_SLAB_MIB",
    description: "Parses \"HIPFIRE_GPU_SLAB_MIB\" with fallback defaults",
    source: "crates/hipfire-arch-qwen35/src/qwen35.rs:4252",
};

/// `HIPFIRE_GPU_TOPK` — HIPFIRE_GPU_TOPK=1 enables the GPU topk_logits_f32 kernel + CPU
pub const ENV_HIPFIRE_GPU_TOPK: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_GPU_TOPK",
    description: "HIPFIRE_GPU_TOPK=1 enables the GPU topk_logits_f32 kernel + CPU",
    source: "crates/hipfire-runtime/examples/infer_qwen35.rs:253",
};

/// `HIPFIRE_GQA_CHUNK` — Runtime variable controlling gqa chunk in hipfire
pub const ENV_HIPFIRE_GQA_CHUNK: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_GQA_CHUNK",
    description: "Runtime variable controlling gqa chunk in hipfire",
    source: "crates/hipfire-rdna/src/dispatch/attention.rs:249",
};

/// `HIPFIRE_GQA_FUSED` — Fused variant (opt-in via HIPFIRE_GQA_FUSED=1): single launch
pub const ENV_HIPFIRE_GQA_FUSED: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_GQA_FUSED",
    description: "Fused variant (opt-in via HIPFIRE_GQA_FUSED=1): single launch",
    source: "crates/hipfire-arch-qwen2/src/qwen2.rs:1700",
};

/// `HIPFIRE_GRAPH` — Used to configure runtime execution by explicitly setting "HIPFIRE_GRAPH"
pub const ENV_HIPFIRE_GRAPH: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_GRAPH",
    description: "Used to configure runtime execution by explicitly setting \"HIPFIRE_GRAPH\"",
    source: "crates/hipfire-runtime/examples/prefill_microbench.rs:146",
};

/// `HIPFIRE_GRAPH_MOE` — - gfx11 (RDNA3 / 3.5): default-ON. +0.6-0.7% decode on 9B and
pub const ENV_HIPFIRE_GRAPH_MOE: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_GRAPH_MOE",
    description: "- gfx11 (RDNA3 / 3.5): default-ON. +0.6-0.7% decode on 9B and",
    source: "crates/hipfire-arch-qwen35/src/qwen35.rs:9244",
};

/// `HIPFIRE_GRAPH_PREFILL` — HIPFIRE_GRAPH_PREFILL=1: route the timed prefill loop through
pub const ENV_HIPFIRE_GRAPH_PREFILL: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_GRAPH_PREFILL",
    description: "HIPFIRE_GRAPH_PREFILL=1: route the timed prefill loop through",
    source: "crates/hipfire-runtime/examples/bench_qwen35_speed.rs:206",
};

/// `HIPFIRE_HAVE_2_GPU` — Environment toggle value controls runtime behavior
pub const ENV_HIPFIRE_HAVE_2_GPU: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_HAVE_2_GPU",
    description: "Environment toggle value controls runtime behavior",
    source: "crates/hipfire-arch-qwen35/tests/pp_parity.rs:193",
};

/// `HIPFIRE_HETERO_DIFF` — above (#352's GPU greedy-accept path doesn't materialize it),
pub const ENV_HIPFIRE_HETERO_DIFF: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_HETERO_DIFF",
    description: "above (#352's GPU greedy-accept path doesn't materialize it),",
    source: "crates/hipfire-arch-qwen35/src/mtp_spec.rs:3007",
};

/// `HIPFIRE_HFQ4G128_MMQ` — Environment toggle value controls runtime behavior
pub const ENV_HIPFIRE_HFQ4G128_MMQ: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_HFQ4G128_MMQ",
    description: "Environment toggle value controls runtime behavior",
    source: "crates/hipfire-rdna/src/feature_flags.rs:241",
};

/// `HIPFIRE_HFQ4G256_MMQ_GFX1151` — Selects behavior from recognized values
pub const ENV_HIPFIRE_HFQ4G256_MMQ_GFX1151: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_HFQ4G256_MMQ_GFX1151",
    description: "Selects behavior from recognized values",
    source: "crates/hipfire-rdna/src/dispatch/mod.rs:3032",
};

/// `HIPFIRE_HFQ4_GATE_UP_FAST` — HIPFIRE_HFQ4_GATE_UP_FAST=0 narrows the escape hatch to the
pub const ENV_HIPFIRE_HFQ4_GATE_UP_FAST: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_HFQ4_GATE_UP_FAST",
    description: "HIPFIRE_HFQ4_GATE_UP_FAST=0 narrows the escape hatch to the",
    source: "crates/hipfire-rdna/src/feature_flags.rs:214",
};

/// `HIPFIRE_HFQ4_MMQ_GFX906_Y64` — Runtime variable controlling hfQ4 mmq gfx906 y64 in hipfire
pub const ENV_HIPFIRE_HFQ4_MMQ_GFX906_Y64: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_HFQ4_MMQ_GFX906_Y64",
    description: "Runtime variable controlling hfQ4 mmq gfx906 y64 in hipfire",
    source: "crates/hipfire-rdna/src/feature_flags.rs:244",
};

/// `HIPFIRE_HFQ4_QKVZA_FAST` — HIPFIRE_HFQ4_QKVZA_FAST=0 narrows the FP16/WMMA/dot2 prefill
pub const ENV_HIPFIRE_HFQ4_QKVZA_FAST: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_HFQ4_QKVZA_FAST",
    description: "HIPFIRE_HFQ4_QKVZA_FAST=0 narrows the FP16/WMMA/dot2 prefill",
    source: "crates/hipfire-rdna/src/feature_flags.rs:206",
};

/// `HIPFIRE_HFQ4_QKV_FAST` — HIPFIRE_HFQ4_QKV_FAST=0 narrows the escape hatch to the
pub const ENV_HIPFIRE_HFQ4_QKV_FAST: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_HFQ4_QKV_FAST",
    description: "HIPFIRE_HFQ4_QKV_FAST=0 narrows the escape hatch to the",
    source: "crates/hipfire-rdna/src/feature_flags.rs:210",
};

/// `HIPFIRE_HFQ4_RESIDUAL_FAST` — HIPFIRE_HFQ4_RESIDUAL_FAST=0 narrows the residual-projection
pub const ENV_HIPFIRE_HFQ4_RESIDUAL_FAST: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_HFQ4_RESIDUAL_FAST",
    description: "HIPFIRE_HFQ4_RESIDUAL_FAST=0 narrows the residual-projection",
    source: "crates/hipfire-rdna/src/feature_flags.rs:218",
};

/// `HIPFIRE_HFQ6_QKVZA_4W` — Interprets "HIPFIRE_HFQ6_QKVZA_4W" from environment to select behavior
pub const ENV_HIPFIRE_HFQ6_QKVZA_4W: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_HFQ6_QKVZA_4W",
    description: "Interprets \"HIPFIRE_HFQ6_QKVZA_4W\" from environment to select behavior",
    source: "crates/hipfire-rdna/src/dispatch/gemm_qkv.rs:3994",
};

/// `HIPFIRE_HFQ6_QKV_4W` — Interprets "HIPFIRE_HFQ6_QKV_4W" from environment to select behavior
pub const ENV_HIPFIRE_HFQ6_QKV_4W: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_HFQ6_QKV_4W",
    description: "Interprets \"HIPFIRE_HFQ6_QKV_4W\" from environment to select behavior",
    source: "crates/hipfire-rdna/src/dispatch/gemm_qkv.rs:4436",
};

/// `HIPFIRE_HFQ6_RESIDUAL_4W` — Interprets "HIPFIRE_HFQ6_RESIDUAL_4W" from environment to select behavior
pub const ENV_HIPFIRE_HFQ6_RESIDUAL_4W: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_HFQ6_RESIDUAL_4W",
    description: "Interprets \"HIPFIRE_HFQ6_RESIDUAL_4W\" from environment to select behavior",
    source: "crates/hipfire-rdna/src/dispatch/gemm_hfq.rs:2404",
};

/// `HIPFIRE_HIPCC_EXTRA_FLAGS` — Parses "HIPFIRE_HIPCC_EXTRA_FLAGS" with fallback defaults
pub const ENV_HIPFIRE_HIPCC_EXTRA_FLAGS: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_HIPCC_EXTRA_FLAGS",
    description: "Parses \"HIPFIRE_HIPCC_EXTRA_FLAGS\" with fallback defaults",
    source: "crates/hipfire-rdna/src/feature_flags.rs:318",
};

/// `HIPFIRE_HIP_WAIT` — Runtime variable controlling hip wait in hipfire
pub const ENV_HIPFIRE_HIP_WAIT: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_HIP_WAIT",
    description: "Runtime variable controlling hip wait in hipfire",
    source: "crates/hipfire-rdna/src/dispatch/mod.rs:693",
};

/// `HIPFIRE_HOST_PROFILE_BIN` — Runtime variable controlling host profile bin in hipfire
pub const ENV_HIPFIRE_HOST_PROFILE_BIN: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_HOST_PROFILE_BIN",
    description: "Runtime variable controlling host profile bin in hipfire",
    source: "crates/hipfire-eval/src/lib.rs:1239",
};

/// `HIPFIRE_HOST_TIMING` — HIPFIRE_HOST_TIMING=1: dump per-cycle host-side wall-clock breakdown
pub const ENV_HIPFIRE_HOST_TIMING: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_HOST_TIMING",
    description: "HIPFIRE_HOST_TIMING=1: dump per-cycle host-side wall-clock breakdown",
    source: "crates/hipfire-runtime/examples/dflash_spec_demo.rs:1760",
};

/// `HIPFIRE_JINJA_CHAT` — Enabled when set to 1
pub const ENV_HIPFIRE_JINJA_CHAT: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_JINJA_CHAT",
    description: "Enabled when set to 1",
    source: "crates/hipfire-serving-core/src/qwen35_prefill.rs:254",
};

/// `HIPFIRE_KERNEL_CACHE` — Overrides the default kernel cache root ("~/.hipfire/kernels")
pub const ENV_HIPFIRE_KERNEL_CACHE: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_KERNEL_CACHE",
    description: "Overrides the default kernel cache root (\"~/.hipfire/kernels\")",
    source: "crates/hipfire-rdna/src/compiler.rs:242",
};

/// `HIPFIRE_KLD_DIRECT_F16KV_ATTN` — Used to configure runtime execution by explicitly setting "HIPFIRE_KLD_DIRECT_F16KV_ATTN"
pub const ENV_HIPFIRE_KLD_DIRECT_F16KV_ATTN: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_KLD_DIRECT_F16KV_ATTN",
    description: "Used to configure runtime execution by explicitly setting \"HIPFIRE_KLD_DIRECT_F16KV_ATTN\"",
    source: "crates/hipfire-arch-qwen35/src/qwen35.rs:12842",
};

/// `HIPFIRE_KLD_DIRECT_WMMA_ATTN` — Used to configure runtime execution by explicitly setting "HIPFIRE_KLD_DIRECT_WMMA_ATTN"
pub const ENV_HIPFIRE_KLD_DIRECT_WMMA_ATTN: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_KLD_DIRECT_WMMA_ATTN",
    description:
        "Used to configure runtime execution by explicitly setting \"HIPFIRE_KLD_DIRECT_WMMA_ATTN\"",
    source: "crates/hipfire-arch-qwen35/src/qwen35.rs:12841",
};

/// `HIPFIRE_KLD_FP32_GQA4_ATTN` — Used to configure runtime execution by explicitly setting "HIPFIRE_KLD_FP32_GQA4_ATTN"
pub const ENV_HIPFIRE_KLD_FP32_GQA4_ATTN: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_KLD_FP32_GQA4_ATTN",
    description:
        "Used to configure runtime execution by explicitly setting \"HIPFIRE_KLD_FP32_GQA4_ATTN\"",
    source: "crates/hipfire-arch-qwen35/src/qwen35.rs:12895",
};

/// `HIPFIRE_KLD_SCORING_MODE` — Runtime variable controlling kld scoring mode in hipfire
pub const ENV_HIPFIRE_KLD_SCORING_MODE: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_KLD_SCORING_MODE",
    description: "Runtime variable controlling kld scoring mode in hipfire",
    source: "crates/hipfire-kld/src/config.rs:102",
};

/// `HIPFIRE_KLD_TOP_K` — Runtime variable controlling kld top k in hipfire
pub const ENV_HIPFIRE_KLD_TOP_K: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_KLD_TOP_K",
    description: "Runtime variable controlling kld top k in hipfire",
    source: "crates/hipfire-kld/src/config.rs:106",
};

/// `HIPFIRE_KVARN_ROTATE` — In-place: mq_rotate_x loads each 256-group into registers (ds_swizzle
pub const ENV_HIPFIRE_KVARN_ROTATE: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_KVARN_ROTATE",
    description: "In-place: mq_rotate_x loads each 256-group into registers (ds_swizzle",
    source: "crates/hipfire-arch-qwen35/src/qwen35.rs:27007",
};

/// `HIPFIRE_KVARN_SIM` — Environment toggle value controls runtime behavior
pub const ENV_HIPFIRE_KVARN_SIM: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_KVARN_SIM",
    description: "Environment toggle value controls runtime behavior",
    source: "crates/hipfire-runtime/examples/perplexity.rs:242",
};

/// `HIPFIRE_KVNOISE` — Environment toggle value controls runtime behavior
pub const ENV_HIPFIRE_KVNOISE: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_KVNOISE",
    description: "Environment toggle value controls runtime behavior",
    source: "crates/hipfire-train/src/kv_noise.rs:36",
};

/// `HIPFIRE_KVNOISE_BITS` — Defaults to 4 when unset
pub const ENV_HIPFIRE_KVNOISE_BITS: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_KVNOISE_BITS",
    description: "Defaults to 4 when unset",
    source: "crates/hipfire-train/examples/qat_w3_kvarn.rs:138",
};

/// `HIPFIRE_KVNOISE_FOLD` — Defaults to 4 when unset
pub const ENV_HIPFIRE_KVNOISE_FOLD: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_KVNOISE_FOLD",
    description: "Defaults to 4 when unset",
    source: "crates/hipfire-train/examples/qat_w3_kvarn.rs:140",
};

/// `HIPFIRE_KVNOISE_HOT` — Defaults to 4 when unset
pub const ENV_HIPFIRE_KVNOISE_HOT: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_KVNOISE_HOT",
    description: "Defaults to 4 when unset",
    source: "crates/hipfire-train/examples/qat_w3_kvarn.rs:139",
};

/// `HIPFIRE_KVNOISE_LR` — Runtime variable controlling KVnoise lr in hipfire
pub const ENV_HIPFIRE_KVNOISE_LR: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_KVNOISE_LR",
    description: "Runtime variable controlling KVnoise lr in hipfire",
    source: "crates/hipfire-train/examples/kvnoise_recovery_supra50m.rs:61",
};

/// `HIPFIRE_KV_COLD_BITS` — Cold-tile quant precision probe: code max = 2^bits - 1 (4-bit=15 default,
pub const ENV_HIPFIRE_KV_COLD_BITS: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_KV_COLD_BITS",
    description: "Cold-tile quant precision probe: code max = 2^bits - 1 (4-bit=15 default,",
    source: "crates/hipfire-runtime/src/kv_hier.rs:184",
};

/// `HIPFIRE_KV_COLD_V_BITS` — match it. Defaults to "cold_bits" (symmetric) when unset
pub const ENV_HIPFIRE_KV_COLD_V_BITS: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_KV_COLD_V_BITS",
    description: "match it. Defaults to \"cold_bits\" (symmetric) when unset",
    source: "crates/hipfire-runtime/src/kv_hier.rs:194",
};

/// `HIPFIRE_KV_COLD_V_PERSLOT` — Enabled when set to 1
pub const ENV_HIPFIRE_KV_COLD_V_PERSLOT: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_KV_COLD_V_PERSLOT",
    description: "Enabled when set to 1",
    source: "crates/hipfire-runtime/src/kv_hier.rs:201",
};

/// `HIPFIRE_KV_CORE_FRAC` — Runtime variable controlling KV core frac in hipfire
pub const ENV_HIPFIRE_KV_CORE_FRAC: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_KV_CORE_FRAC",
    description: "Runtime variable controlling KV core frac in hipfire",
    source: "crates/hipfire-runtime/src/kv_hier.rs:174",
};

/// `HIPFIRE_KV_FOLD_M` — Cold-tier compaction knobs. fold_m=1 disables the m:1 merge (cold = pure
pub const ENV_HIPFIRE_KV_FOLD_M: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_KV_FOLD_M",
    description: "Cold-tier compaction knobs. fold_m=1 disables the m:1 merge (cold = pure",
    source: "crates/hipfire-runtime/src/kv_hier.rs:169",
};

/// `HIPFIRE_KV_HIERARCHICAL` — (→ qwen35_prefill_active_session → per-token forward_scratch), which honours
pub const ENV_HIPFIRE_KV_HIERARCHICAL: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_KV_HIERARCHICAL",
    description: "(→ qwen35_prefill_active_session → per-token forward_scratch), which honours",
    source: "crates/hipfire-serving-core/src/qwen35_prefill.rs:545",
};

/// `HIPFIRE_KV_HIER_KEEP` — Parsed as value configuration from environment value
pub const ENV_HIPFIRE_KV_HIER_KEEP: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_KV_HIER_KEEP",
    description: "Parsed as value configuration from environment value",
    source: "crates/hipfire-runtime/examples/parity_kv_hier.rs:78",
};

/// `HIPFIRE_KV_HIER_TEST_IDLE` — Optional: exercise the deferred between-turns drain. idle_compact moves hot
pub const ENV_HIPFIRE_KV_HIER_TEST_IDLE: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_KV_HIER_TEST_IDLE",
    description: "Optional: exercise the deferred between-turns drain. idle_compact moves hot",
    source: "crates/hipfire-runtime/examples/parity_kv_hier.rs:77",
};

/// `HIPFIRE_KV_HOT_BUDGET` — Runtime variable controlling KV hot budget in hipfire
pub const ENV_HIPFIRE_KV_HOT_BUDGET: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_KV_HOT_BUDGET",
    description: "Runtime variable controlling KV hot budget in hipfire",
    source: "crates/hipfire-runtime/src/kv_hier.rs:156",
};

/// `HIPFIRE_KV_IDLE_KEEP` — Runtime variable controlling KV idle keep in hipfire
pub const ENV_HIPFIRE_KV_IDLE_KEEP: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_KV_IDLE_KEEP",
    description: "Runtime variable controlling KV idle keep in hipfire",
    source: "crates/hipfire-serving-core/src/qwen35_prefill.rs:139",
};

/// `HIPFIRE_KV_IMPORTANCE` — Cold-tile quant precision probe: code max = 2^bits - 1 (4-bit=15 default,
pub const ENV_HIPFIRE_KV_IMPORTANCE: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_KV_IMPORTANCE",
    description: "Cold-tile quant precision probe: code max = 2^bits - 1 (4-bit=15 default,",
    source: "crates/hipfire-runtime/src/kv_hier.rs:179",
};

/// `HIPFIRE_KV_MIGRATE_BATCH` — Cold-tier compaction knobs. fold_m=1 disables the m:1 merge (cold = pure
pub const ENV_HIPFIRE_KV_MIGRATE_BATCH: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_KV_MIGRATE_BATCH",
    description: "Cold-tier compaction knobs. fold_m=1 disables the m:1 merge (cold = pure",
    source: "crates/hipfire-runtime/src/kv_hier.rs:160",
};

/// `HIPFIRE_KV_MODE` — Runtime variable controlling KV mode in hipfire
pub const ENV_HIPFIRE_KV_MODE: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_KV_MODE",
    description: "Runtime variable controlling KV mode in hipfire",
    source: "crates/hipfire-serving-core/src/load.rs:2667",
};

/// `HIPFIRE_KV_PHYSICAL_CAP` — Parses "HIPFIRE_KV_PHYSICAL_CAP" with fallback defaults
pub const ENV_HIPFIRE_KV_PHYSICAL_CAP: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_KV_PHYSICAL_CAP",
    description: "Parses \"HIPFIRE_KV_PHYSICAL_CAP\" with fallback defaults",
    source: "crates/hipfire-serving-core/src/load.rs:625",
};

/// `HIPFIRE_KV_POS_LOCAL` — Cold-tile quant precision probe: code max = 2^bits - 1 (4-bit=15 default,
pub const ENV_HIPFIRE_KV_POS_LOCAL: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_KV_POS_LOCAL",
    description: "Cold-tile quant precision probe: code max = 2^bits - 1 (4-bit=15 default,",
    source: "crates/hipfire-runtime/src/kv_hier.rs:181",
};

/// `HIPFIRE_LFM2_CAPTURE_POSTMIXER` — Runtime variable controlling lfm2 capture postmixer in hipfire
pub const ENV_HIPFIRE_LFM2_CAPTURE_POSTMIXER: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_LFM2_CAPTURE_POSTMIXER",
    description: "Runtime variable controlling lfm2 capture postmixer in hipfire",
    source: "crates/hipfire-arch-lfm2moe/src/forward.rs:1154",
};

/// `HIPFIRE_LFM2_DFLASH` — Environment toggle value controls runtime behavior
pub const ENV_HIPFIRE_LFM2_DFLASH: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_LFM2_DFLASH",
    description: "Environment toggle value controls runtime behavior",
    source: "crates/hipfire-serving-core/src/load.rs:492",
};

/// `HIPFIRE_LFM2_DFLASH_F16` — Enabled when set to 1
pub const ENV_HIPFIRE_LFM2_DFLASH_F16: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_LFM2_DFLASH_F16",
    description: "Enabled when set to 1",
    source: "crates/hipfire-arch-lfm2moe/src/dflash.rs:25",
};

/// `HIPFIRE_LFM2_DFLASH_SYNC_GEMM` — Runtime variable controlling lfm2 dflash sync gemm in hipfire
pub const ENV_HIPFIRE_LFM2_DFLASH_SYNC_GEMM: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_LFM2_DFLASH_SYNC_GEMM",
    description: "Runtime variable controlling lfm2 dflash sync gemm in hipfire",
    source: "crates/hipfire-arch-lfm2moe/src/dflash.rs:31",
};

/// `HIPFIRE_LFM2_GRAPH` — Interprets "HIPFIRE_LFM2_GRAPH" from environment to select behavior
pub const ENV_HIPFIRE_LFM2_GRAPH: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_LFM2_GRAPH",
    description: "Interprets \"HIPFIRE_LFM2_GRAPH\" from environment to select behavior",
    source: "crates/hipfire-arch-lfm2moe/src/forward.rs:67",
};

/// `HIPFIRE_LLOYD_FORCE_BASELINE` — Runtime variable controlling lloyd force baseline in hipfire
pub const ENV_HIPFIRE_LLOYD_FORCE_BASELINE: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_LLOYD_FORCE_BASELINE",
    description: "Runtime variable controlling lloyd force baseline in hipfire",
    source: "crates/hipfire-rdna/src/feature_flags.rs:309",
};

/// `HIPFIRE_LLOYD_GFX12` — Used to configure runtime execution by explicitly setting "HIPFIRE_LLOYD_GFX12"
pub const ENV_HIPFIRE_LLOYD_GFX12: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_LLOYD_GFX12",
    description:
        "Used to configure runtime execution by explicitly setting \"HIPFIRE_LLOYD_GFX12\"",
    source: "crates/hipfire-runtime/examples/prefill_microbench.rs:167",
};

/// `HIPFIRE_LLOYD_K3` — Fallback to HFQ2-G128 for non-256-aligned (no rotation)
pub const ENV_HIPFIRE_LLOYD_K3: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_LLOYD_K3",
    description: "Fallback to HFQ2-G128 for non-256-aligned (no rotation)",
    source: "crates/hipfire-quantize/src/main.rs:8153",
};

/// `HIPFIRE_LLOYD_MB4` — Force MB4=0 to skip the size-gated routing
pub const ENV_HIPFIRE_LLOYD_MB4: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_LLOYD_MB4",
    description: "Force MB4=0 to skip the size-gated routing",
    source: "crates/hipfire-rdna/examples/test_gemm_mq4g256_lloyd_residual_wmma.rs:274",
};

/// `HIPFIRE_LM_DUMP` — Selects behavior from recognized values
pub const ENV_HIPFIRE_LM_DUMP: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_LM_DUMP",
    description: "Selects behavior from recognized values",
    source: "crates/hipfire-arch-gemma3/src/forward.rs:343",
};

/// `HIPFIRE_LM_HEAD_F16` — Runtime variable controlling lm head f16 in hipfire
pub const ENV_HIPFIRE_LM_HEAD_F16: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_LM_HEAD_F16",
    description: "Runtime variable controlling lm head f16 in hipfire",
    source: "crates/hipfire-runtime/src/config.rs:108",
};

/// `HIPFIRE_LM_HEAD_OVERWRITE` — Environment toggle value controls runtime behavior
pub const ENV_HIPFIRE_LM_HEAD_OVERWRITE: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_LM_HEAD_OVERWRITE",
    description: "Environment toggle value controls runtime behavior",
    source: "crates/hipfire-rdna/src/feature_flags.rs:222",
};

/// `HIPFIRE_LM_HEAD_WMMA` — Runtime variable controlling lm head wmma in hipfire
pub const ENV_HIPFIRE_LM_HEAD_WMMA: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_LM_HEAD_WMMA",
    description: "Runtime variable controlling lm head wmma in hipfire",
    source: "crates/hipfire-rdna/src/feature_flags.rs:220",
};

/// `HIPFIRE_LOAD_TRANSPORT` — Selects behavior from recognized values
pub const ENV_HIPFIRE_LOAD_TRANSPORT: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_LOAD_TRANSPORT",
    description: "Selects behavior from recognized values",
    source: "crates/hipfire-runtime/src/weight_pager.rs:768",
};

/// `HIPFIRE_LOCAL` — Force local-spawn behavior and skip serve HTTP in documented workflows
pub const ENV_HIPFIRE_LOCAL: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_LOCAL",
    description: "Force local-spawn behavior and skip serve HTTP in documented workflows",
    source: "README.md:962",
};

/// `HIPFIRE_LOWRANK_R` — HIPFIRE_LOWRANK_R=r adds a rank-r correction of the quant error back
pub const ENV_HIPFIRE_LOWRANK_R: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_LOWRANK_R",
    description: "HIPFIRE_LOWRANK_R=r adds a rank-r correction of the quant error back",
    source: "crates/hipfire-quantize/src/main.rs:10060",
};

/// `HIPFIRE_MAX_GEN` — Runtime variable controlling max gen in hipfire
pub const ENV_HIPFIRE_MAX_GEN: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_MAX_GEN",
    description: "Runtime variable controlling max gen in hipfire",
    source: "crates/hipfire-runtime/examples/infer_qwen35.rs:313",
};

/// `HIPFIRE_MAX_SEQ_ALLOW_OVERRIDE` — Interprets "HIPFIRE_MAX_SEQ_ALLOW_OVERRIDE" from environment to select behavior
pub const ENV_HIPFIRE_MAX_SEQ_ALLOW_OVERRIDE: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_MAX_SEQ_ALLOW_OVERRIDE",
    description:
        "Interprets \"HIPFIRE_MAX_SEQ_ALLOW_OVERRIDE\" from environment to select behavior",
    source: "crates/hipfire-serving-core/src/load.rs:312",
};

/// `HIPFIRE_MEMCPY_DUMP` — Enabled when set to 1
pub const ENV_HIPFIRE_MEMCPY_DUMP: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_MEMCPY_DUMP",
    description: "Enabled when set to 1",
    source: "crates/hip-bridge/src/ffi.rs:1093",
};

/// `HIPFIRE_MEMSET_DUMP` — Enabled when set to 1
pub const ENV_HIPFIRE_MEMSET_DUMP: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_MEMSET_DUMP",
    description: "Enabled when set to 1",
    source: "crates/hip-bridge/src/ffi.rs:1145",
};

/// `HIPFIRE_MINIMAX_CAPTURE_POSTATTN` — Runtime variable controlling minimax capture postattn in hipfire
pub const ENV_HIPFIRE_MINIMAX_CAPTURE_POSTATTN: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_MINIMAX_CAPTURE_POSTATTN",
    description: "Runtime variable controlling minimax capture postattn in hipfire",
    source: "crates/hipfire-arch-minimax/src/forward.rs:206",
};

/// `HIPFIRE_MINIMAX_DOWN_FORMAT` — Runtime variable controlling minimax down format in hipfire
pub const ENV_HIPFIRE_MINIMAX_DOWN_FORMAT: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_MINIMAX_DOWN_FORMAT",
    description: "Runtime variable controlling minimax down format in hipfire",
    source: "crates/hipfire-quantize/src/main.rs:6977",
};

/// `HIPFIRE_MINIMAX_ENABLE_DOWN_AWQ` — down-AWQ harmful (shared s_down bad approx); opt-in
pub const ENV_HIPFIRE_MINIMAX_ENABLE_DOWN_AWQ: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_MINIMAX_ENABLE_DOWN_AWQ",
    description: "down-AWQ harmful (shared s_down bad approx); opt-in",
    source: "crates/hipfire-arch-minimax/src/minimax.rs:868",
};

/// `HIPFIRE_MINIMAX_EXPERT_MQ2L` — dispatches expert dtype per-layer (experts[0].gpu_dtype), so the model
pub const ENV_HIPFIRE_MINIMAX_EXPERT_MQ2L: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_MINIMAX_EXPERT_MQ2L",
    description: "dispatches expert dtype per-layer (experts[0].gpu_dtype), so the model",
    source: "crates/hipfire-quantize/src/main.rs:6956",
};

/// `HIPFIRE_MINIMAX_EXPERT_MQ3L` — dispatches expert dtype per-layer (experts[0].gpu_dtype), so the model
pub const ENV_HIPFIRE_MINIMAX_EXPERT_MQ3L: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_MINIMAX_EXPERT_MQ3L",
    description: "dispatches expert dtype per-layer (experts[0].gpu_dtype), so the model",
    source: "crates/hipfire-quantize/src/main.rs:6958",
};

/// `HIPFIRE_MINIMAX_EXPERT_MQ6` — _MQ6 hold comma-separated layer ranges ("12-45,50") whose experts are
pub const ENV_HIPFIRE_MINIMAX_EXPERT_MQ6: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_MINIMAX_EXPERT_MQ6",
    description: "_MQ6 hold comma-separated layer ranges (\"12-45,50\") whose experts are",
    source: "crates/hipfire-quantize/src/main.rs:6954",
};

/// `HIPFIRE_MMQ` — Selects behavior from recognized values
pub const ENV_HIPFIRE_MMQ: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_MMQ",
    description: "Selects behavior from recognized values",
    source: "crates/hipfire-rdna/src/feature_flags.rs:194",
};

/// `HIPFIRE_MMQ_DIAG_QUANTIZE_ONLY` — Runtime variable controlling mmq diag quantize only in hipfire
pub const ENV_HIPFIRE_MMQ_DIAG_QUANTIZE_ONLY: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_MMQ_DIAG_QUANTIZE_ONLY",
    description: "Runtime variable controlling mmq diag quantize only in hipfire",
    source: "crates/hipfire-rdna/src/feature_flags.rs:233",
};

/// `HIPFIRE_MMQ_SCREEN` — Runtime variable controlling mmq screen in hipfire
pub const ENV_HIPFIRE_MMQ_SCREEN: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_MMQ_SCREEN",
    description: "Runtime variable controlling mmq screen in hipfire",
    source: "crates/hipfire-rdna/src/feature_flags.rs:225",
};

/// `HIPFIRE_MMQ_SCREEN_THRESHOLD` — Runtime variable controlling mmq screen threshold in hipfire
pub const ENV_HIPFIRE_MMQ_SCREEN_THRESHOLD: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_MMQ_SCREEN_THRESHOLD",
    description: "Runtime variable controlling mmq screen threshold in hipfire",
    source: "crates/hipfire-rdna/src/feature_flags.rs:229",
};

/// `HIPFIRE_MODELS_DIR` — Runtime variable controlling models dir in hipfire
pub const ENV_HIPFIRE_MODELS_DIR: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_MODELS_DIR",
    description: "Runtime variable controlling models dir in hipfire",
    source: "crates/hipfire-eval/src/run.rs:416",
};

/// `HIPFIRE_MOE_GROUPED_GEMM` — Runtime variable controlling moe grouped gemm in hipfire
pub const ENV_HIPFIRE_MOE_GROUPED_GEMM: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_MOE_GROUPED_GEMM",
    description: "Runtime variable controlling moe grouped gemm in hipfire",
    source: "crates/hipfire-rdna/src/feature_flags.rs:283",
};

/// `HIPFIRE_MOE_GROUPED_I8` — HIPFIRE_MOE_GROUPED_I8_K8: gfx1151 HFQ4 grouped MoE k8 control;
pub const ENV_HIPFIRE_MOE_GROUPED_I8: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_MOE_GROUPED_I8",
    description: "HIPFIRE_MOE_GROUPED_I8_K8: gfx1151 HFQ4 grouped MoE k8 control;",
    source: "crates/hipfire-rdna/src/feature_flags.rs:253",
};

/// `HIPFIRE_MOE_GROUPED_I8_4W` — default on for gfx1151, set to 0 to test k4 or base k2 variants
pub const ENV_HIPFIRE_MOE_GROUPED_I8_4W: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_MOE_GROUPED_I8_4W",
    description: "default on for gfx1151, set to 0 to test k4 or base k2 variants",
    source: "crates/hipfire-rdna/src/feature_flags.rs:258",
};

/// `HIPFIRE_MOE_GROUPED_I8_K4` — HIPFIRE_MOE_GROUPED_I8_K4: opt-in gfx1151 HFQ4 grouped MoE k4
pub const ENV_HIPFIRE_MOE_GROUPED_I8_K4: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_MOE_GROUPED_I8_K4",
    description: "HIPFIRE_MOE_GROUPED_I8_K4: opt-in gfx1151 HFQ4 grouped MoE k4",
    source: "crates/hipfire-rdna/src/feature_flags.rs:269",
};

/// `HIPFIRE_MOE_GROUPED_I8_K4_GFX12` — HIPFIRE_MOE_HFQ6_V2: opt-in HFQ6 grouped MoE v2 path; on
pub const ENV_HIPFIRE_MOE_GROUPED_I8_K4_GFX12: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_MOE_GROUPED_I8_K4_GFX12",
    description: "HIPFIRE_MOE_HFQ6_V2: opt-in HFQ6 grouped MoE v2 path; on",
    source: "crates/hipfire-rdna/src/feature_flags.rs:270",
};

/// `HIPFIRE_MOE_GROUPED_I8_K8` — variant; use with HIPFIRE_MOE_GROUPED_I8_K8=0. Default off
pub const ENV_HIPFIRE_MOE_GROUPED_I8_K8: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_MOE_GROUPED_I8_K8",
    description: "variant; use with HIPFIRE_MOE_GROUPED_I8_K8=0. Default off",
    source: "crates/hipfire-rdna/src/feature_flags.rs:264",
};

/// `HIPFIRE_MOE_GROUPED_M2` — HIPFIRE_MOE_HFQ6_V2: opt-in HFQ6 grouped MoE v2 path; on
pub const ENV_HIPFIRE_MOE_GROUPED_M2: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_MOE_GROUPED_M2",
    description: "HIPFIRE_MOE_HFQ6_V2: opt-in HFQ6 grouped MoE v2 path; on",
    source: "crates/hipfire-rdna/src/feature_flags.rs:272",
};

/// `HIPFIRE_MOE_HFQ6_4W` — Environment toggle value controls runtime behavior
pub const ENV_HIPFIRE_MOE_HFQ6_4W: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_MOE_HFQ6_4W",
    description: "Environment toggle value controls runtime behavior",
    source: "crates/hipfire-rdna/src/feature_flags.rs:280",
};

/// `HIPFIRE_MOE_HFQ6_V2` — HIPFIRE_MOE_HFQ6_V2: opt-in HFQ6 grouped MoE v2 path; on
pub const ENV_HIPFIRE_MOE_HFQ6_V2: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_MOE_HFQ6_V2",
    description: "HIPFIRE_MOE_HFQ6_V2: opt-in HFQ6 grouped MoE v2 path; on",
    source: "crates/hipfire-rdna/src/feature_flags.rs:276",
};

/// `HIPFIRE_MOE_INDEXED_2ROW_GFX1151` — Opt-in ("1") gfx1151 two-row indexed MoE HFQ4 decode probe for gate/up and expanded down; default off after flat A3B measurements
pub const ENV_HIPFIRE_MOE_INDEXED_2ROW_GFX1151: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_MOE_INDEXED_2ROW_GFX1151",
    description: "Opt-in (\"1\") gfx1151 two-row indexed MoE HFQ4 decode probe for gate/up and expanded down; default off after flat A3B measurements",
    source: "crates/hipfire-rdna/src/dispatch/mod.rs:3067",
};

/// `HIPFIRE_MOE_MQ2L_N32_GFX1151` — Runtime variable controlling moe mq2l n32 gfx1151 in hipfire
pub const ENV_HIPFIRE_MOE_MQ2L_N32_GFX1151: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_MOE_MQ2L_N32_GFX1151",
    description: "Runtime variable controlling moe mq2l n32 gfx1151 in hipfire",
    source: "crates/hipfire-arch-qwen35/src/qwen35.rs:14907",
};

/// `HIPFIRE_MOE_PARO_I8` — Runtime variable controlling moe paro i8 in hipfire
pub const ENV_HIPFIRE_MOE_PARO_I8: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_MOE_PARO_I8",
    description: "Runtime variable controlling moe paro i8 in hipfire",
    source: "crates/hipfire-arch-qwen35/src/qwen35.rs:16523",
};

/// `HIPFIRE_MOE_PARO_I8_K8` — Runtime variable controlling moe paro i8 k8 in hipfire
pub const ENV_HIPFIRE_MOE_PARO_I8_K8: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_MOE_PARO_I8_K8",
    description: "Runtime variable controlling moe paro i8 k8 in hipfire",
    source: "crates/hipfire-arch-qwen35/src/qwen35.rs:16527",
};

/// `HIPFIRE_MQ3_MB4` — Used to configure runtime execution by explicitly setting "HIPFIRE_MQ3_MB4"
pub const ENV_HIPFIRE_MQ3_MB4: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_MQ3_MB4",
    description: "Used to configure runtime execution by explicitly setting \"HIPFIRE_MQ3_MB4\"",
    source: "crates/hipfire-rdna/examples/test_gemm_hfq3g256_wmma.rs:129",
};

/// `HIPFIRE_MTP_DEVICE_TOKEN_CHAIN` — Default on: this path is token-identical in greedy mode and removes the
pub const ENV_HIPFIRE_MTP_DEVICE_TOKEN_CHAIN: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_MTP_DEVICE_TOKEN_CHAIN",
    description: "Default on: this path is token-identical in greedy mode and removes the",
    source: "crates/hipfire-arch-qwen35/src/mtp_spec.rs:82",
};

/// `HIPFIRE_MTP_GPU_ACCEPT` — Default on for greedy device-token-chain MTP: candidates and verify
pub const ENV_HIPFIRE_MTP_GPU_ACCEPT: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_MTP_GPU_ACCEPT",
    description: "Default on for greedy device-token-chain MTP: candidates and verify",
    source: "crates/hipfire-arch-qwen35/src/mtp_spec.rs:107",
};

/// `HIPFIRE_MTP_HEAD_LMHEAD_WMMA` — HIPFIRE_MTP_HEAD_LMHEAD_WMMA=0. (Ported from origin/master 5ac96a8f.)
pub const ENV_HIPFIRE_MTP_HEAD_LMHEAD_WMMA: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_MTP_HEAD_LMHEAD_WMMA",
    description: "HIPFIRE_MTP_HEAD_LMHEAD_WMMA=0. (Ported from origin/master 5ac96a8f.)",
    source: "crates/hipfire-arch-qwen35/src/mtp_head.rs:2063",
};

/// `HIPFIRE_MTP_K` — Runtime variable controlling MTP k in hipfire
pub const ENV_HIPFIRE_MTP_K: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_MTP_K",
    description: "Runtime variable controlling MTP k in hipfire",
    source: "crates/hipfire-serving-core/src/generate_arch.rs:360",
};

/// `HIPFIRE_MTP_MODE` — Defaults to auto when unset
pub const ENV_HIPFIRE_MTP_MODE: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_MTP_MODE",
    description: "Defaults to auto when unset",
    source: "crates/hipfire-serving-core/src/generate_arch.rs:351",
};

/// `HIPFIRE_MTP_PHASE_TIMERS` — Env-gated phase timers (HIPFIRE_MTP_PHASE_TIMERS=1). The existing
pub const ENV_HIPFIRE_MTP_PHASE_TIMERS: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_MTP_PHASE_TIMERS",
    description: "Env-gated phase timers (HIPFIRE_MTP_PHASE_TIMERS=1). The existing",
    source: "crates/hipfire-arch-qwen35/src/mtp_spec.rs:1348",
};

/// `HIPFIRE_MTP_PROPOSAL_GRAPH` — Runtime variable controlling MTP proposal graph in hipfire
pub const ENV_HIPFIRE_MTP_PROPOSAL_GRAPH: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_MTP_PROPOSAL_GRAPH",
    description: "Runtime variable controlling MTP proposal graph in hipfire",
    source: "crates/hipfire-arch-qwen35/src/mtp_spec.rs:187",
};

/// `HIPFIRE_MTP_P_MIN` — Runtime variable controlling MTP p min in hipfire
pub const ENV_HIPFIRE_MTP_P_MIN: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_MTP_P_MIN",
    description: "Runtime variable controlling MTP p min in hipfire",
    source: "crates/hipfire-arch-qwen35/src/mtp_spec.rs:147",
};

/// `HIPFIRE_MTP_Q8_VERIFY_WMMA` — Runtime variable controlling MTP Q8 verify wmma in hipfire
pub const ENV_HIPFIRE_MTP_Q8_VERIFY_WMMA: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_MTP_Q8_VERIFY_WMMA",
    description: "Runtime variable controlling MTP Q8 verify wmma in hipfire",
    source: "crates/hipfire-arch-qwen35/src/mtp_spec.rs:131",
};

/// `HIPFIRE_MTP_SMOKE_HEAD` — Runtime variable controlling MTP smoke head in hipfire
pub const ENV_HIPFIRE_MTP_SMOKE_HEAD: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_MTP_SMOKE_HEAD",
    description: "Runtime variable controlling MTP smoke head in hipfire",
    source: "crates/hipfire-arch-qwen35/examples/mtp_head_smoke.rs:48",
};

/// `HIPFIRE_MTP_SMOKE_TRUNK` — Runtime variable controlling MTP smoke trunk in hipfire
pub const ENV_HIPFIRE_MTP_SMOKE_TRUNK: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_MTP_SMOKE_TRUNK",
    description: "Runtime variable controlling MTP smoke trunk in hipfire",
    source: "crates/hipfire-arch-qwen35/examples/mtp_head_smoke.rs:44",
};

/// `HIPFIRE_MTP_SNAPSHOT_OVERLAP` — Default off: current gfx1201 benches show this stream split regresses,
pub const ENV_HIPFIRE_MTP_SNAPSHOT_OVERLAP: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_MTP_SNAPSHOT_OVERLAP",
    description: "Default off: current gfx1201 benches show this stream split regresses,",
    source: "crates/hipfire-arch-qwen35/src/mtp_spec.rs:94",
};

/// `HIPFIRE_MTP_VERIFY_DECOUPLE` — mq4). Opt-out HIPFIRE_MTP_VERIFY_DECOUPLE=0. Other archs are opt-in (=1)
pub const ENV_HIPFIRE_MTP_VERIFY_DECOUPLE: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_MTP_VERIFY_DECOUPLE",
    description: "mq4). Opt-out HIPFIRE_MTP_VERIFY_DECOUPLE=0. Other archs are opt-in (=1)",
    source: "crates/hipfire-arch-qwen35/src/qwen35.rs:14765",
};

/// `HIPFIRE_MW16` — Runtime variable controlling mw16 in hipfire
pub const ENV_HIPFIRE_MW16: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_MW16",
    description: "Runtime variable controlling mw16 in hipfire",
    source: "crates/hipfire-rdna/src/feature_flags.rs:294",
};

/// `HIPFIRE_NEMOTRON_SSM_STATE` — Selects behavior from recognized values
pub const ENV_HIPFIRE_NEMOTRON_SSM_STATE: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_NEMOTRON_SSM_STATE",
    description: "Selects behavior from recognized values",
    source: "crates/hipfire-arch-nemotron/src/block_gpu.rs:39",
};

/// `HIPFIRE_NGRAM_LOOP_THRESHOLD` — Parses "HIPFIRE_NGRAM_LOOP_THRESHOLD" with fallback defaults
pub const ENV_HIPFIRE_NGRAM_LOOP_THRESHOLD: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_NGRAM_LOOP_THRESHOLD",
    description: "Parses \"HIPFIRE_NGRAM_LOOP_THRESHOLD\" with fallback defaults",
    source: "crates/hipfire-runtime/src/config.rs:94",
};

/// `HIPFIRE_NGRAM_WINDOW` — Runtime variable controlling ngram window in hipfire
pub const ENV_HIPFIRE_NGRAM_WINDOW: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_NGRAM_WINDOW",
    description: "Runtime variable controlling ngram window in hipfire",
    source: "crates/hipfire-runtime/src/config.rs:98",
};

/// `HIPFIRE_NORMALIZE_PROMPT` — Opt-out must skip CRLF/NBSP/trailing-ws too, not just newline collapse
pub const ENV_HIPFIRE_NORMALIZE_PROMPT: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_NORMALIZE_PROMPT",
    description: "Opt-out must skip CRLF/NBSP/trailing-ws too, not just newline collapse",
    source: "crates/hipfire-serving-core/src/output_filter.rs:30",
};

/// `HIPFIRE_NORM_PLUS1` — qwen3.5 may store RMSNorm weight as (1+γ) (Gemma-style). HIPFIRE_NORM_PLUS1=1
pub const ENV_HIPFIRE_NORM_PLUS1: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_NORM_PLUS1",
    description: "qwen3.5 may store RMSNorm weight as (1+γ) (Gemma-style). HIPFIRE_NORM_PLUS1=1",
    source: "crates/hipfire-train/examples/qwen35_mlp_norm_recovery.rs:212",
};

/// `HIPFIRE_NO_EXPERT_AWQ` — experts fall back to plain MQ4/MQ8) — an A/B knob for measuring the
pub const ENV_HIPFIRE_NO_EXPERT_AWQ: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_NO_EXPERT_AWQ",
    description: "experts fall back to plain MQ4/MQ8) — an A/B knob for measuring the",
    source: "crates/hipfire-quantize/src/main.rs:7257",
};

/// `HIPFIRE_NO_QUANT` — Student weights = OQ+ sim-quant (HIPFIRE_NO_QUANT=1 → identity sanity check)
pub const ENV_HIPFIRE_NO_QUANT: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_NO_QUANT",
    description: "Student weights = OQ+ sim-quant (HIPFIRE_NO_QUANT=1 → identity sanity check)",
    source: "crates/hipfire-train/examples/qwen35_mlp_norm_recovery.rs:269",
};

/// `HIPFIRE_NPU_ATTN_GATE_CONFIGS` — Runtime variable controlling npu attn gate configs in hipfire
pub const ENV_HIPFIRE_NPU_ATTN_GATE_CONFIGS: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_NPU_ATTN_GATE_CONFIGS",
    description: "Runtime variable controlling npu attn gate configs in hipfire",
    source: "crates/hipfire-arch-qwen35/build.rs:193",
};

/// `HIPFIRE_NPU_DIR` — Defaults to target/npu when unset
pub const ENV_HIPFIRE_NPU_DIR: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_NPU_DIR",
    description: "Defaults to target/npu when unset",
    source: "crates/hipfire-arch-qwen35/src/qwen35.rs:2083",
};

/// `HIPFIRE_NPU_HEADNORM_CONFIGS` — Parse "n_heads:n_kv_heads:head_dim" tuples (n_rot field ignored if present)
pub const ENV_HIPFIRE_NPU_HEADNORM_CONFIGS: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_NPU_HEADNORM_CONFIGS",
    description: "Parse \"n_heads:n_kv_heads:head_dim\" tuples (n_rot field ignored if present)",
    source: "crates/hipfire-arch-qwen35/build.rs:156",
};

/// `HIPFIRE_NPU_HEADNORM_ROPE_CONFIGS` — Runtime variable controlling npu headnorm rope configs in hipfire
pub const ENV_HIPFIRE_NPU_HEADNORM_ROPE_CONFIGS: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_NPU_HEADNORM_ROPE_CONFIGS",
    description: "Runtime variable controlling npu headnorm rope configs in hipfire",
    source: "crates/hipfire-arch-qwen35/build.rs:277",
};

/// `HIPFIRE_NPU_HIDDEN_SIZES` — Runtime variable controlling npu hidden sizes in hipfire
pub const ENV_HIPFIRE_NPU_HIDDEN_SIZES: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_NPU_HIDDEN_SIZES",
    description: "Runtime variable controlling npu hidden sizes in hipfire",
    source: "crates/hipfire-arch-qwen35/build.rs:54",
};

/// `HIPFIRE_NPU_PYTHON` — Runtime variable controlling npu python in hipfire
pub const ENV_HIPFIRE_NPU_PYTHON: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_NPU_PYTHON",
    description: "Runtime variable controlling npu python in hipfire",
    source: "crates/hipfire-arch-qwen35/build.rs:601",
};

/// `HIPFIRE_NPU_RMSNORM_SIZES` — Runtime variable controlling npu rmsnorm sizes in hipfire
pub const ENV_HIPFIRE_NPU_RMSNORM_SIZES: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_NPU_RMSNORM_SIZES",
    description: "Runtime variable controlling npu rmsnorm sizes in hipfire",
    source: "crates/hipfire-arch-qwen35/build.rs:89",
};

/// `HIPFIRE_NPU_ROPE_CONFIGS` — Parse "n_heads:n_kv_heads:head_dim:n_rot" tuples
pub const ENV_HIPFIRE_NPU_ROPE_CONFIGS: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_NPU_ROPE_CONFIGS",
    description: "Parse \"n_heads:n_kv_heads:head_dim:n_rot\" tuples",
    source: "crates/hipfire-arch-qwen35/build.rs:118",
};

/// `HIPFIRE_NPU_SOFTMAX_CONFIGS` — Parse "n_heads:ctx_len1+ctx_len2+..." entries
pub const ENV_HIPFIRE_NPU_SOFTMAX_CONFIGS: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_NPU_SOFTMAX_CONFIGS",
    description: "Parse \"n_heads:ctx_len1+ctx_len2+...\" entries",
    source: "crates/hipfire-arch-qwen35/build.rs:232",
};

/// `HIPFIRE_NPU_TARGETS` — Runtime variable controlling npu targets in hipfire
pub const ENV_HIPFIRE_NPU_TARGETS: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_NPU_TARGETS",
    description: "Runtime variable controlling npu targets in hipfire",
    source: "crates/hipfire-arch-qwen35/build.rs:60",
};

/// `HIPFIRE_OQ4_BATCHED_PREFILL` — Environment toggle value controls runtime behavior
pub const ENV_HIPFIRE_OQ4_BATCHED_PREFILL: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_OQ4_BATCHED_PREFILL",
    description: "Environment toggle value controls runtime behavior",
    source: "crates/hipfire-arch-qwen35/src/qwen35.rs:14243",
};

/// `HIPFIRE_OQ4_PREFILL_ACT_BITS` — Selects behavior from recognized values
pub const ENV_HIPFIRE_OQ4_PREFILL_ACT_BITS: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_OQ4_PREFILL_ACT_BITS",
    description: "Selects behavior from recognized values",
    source: "crates/hipfire-runtime/src/weights.rs:1420",
};

/// `HIPFIRE_OQ4_TRACE` — Runtime variable controlling oQ4 trace in hipfire
pub const ENV_HIPFIRE_OQ4_TRACE: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_OQ4_TRACE",
    description: "Runtime variable controlling oQ4 trace in hipfire",
    source: "crates/hipfire-runtime/src/weights.rs:655",
};

/// `HIPFIRE_PAGED_MOE_DEBUG` — Enabled when set to 1
pub const ENV_HIPFIRE_PAGED_MOE_DEBUG: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_PAGED_MOE_DEBUG",
    description: "Enabled when set to 1",
    source: "crates/hipfire-arch-qwen35/src/qwen35.rs:7818",
};

/// `HIPFIRE_PARO_BATCHED` — Runtime variable controlling paro batched in hipfire
pub const ENV_HIPFIRE_PARO_BATCHED: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_PARO_BATCHED",
    description: "Runtime variable controlling paro batched in hipfire",
    source: "crates/hipfire-arch-qwen35/src/qwen35.rs:14835",
};

/// `HIPFIRE_PARO_FA3_FUSED` — Runtime variable controlling paro fa3 fused in hipfire
pub const ENV_HIPFIRE_PARO_FA3_FUSED: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_PARO_FA3_FUSED",
    description: "Runtime variable controlling paro fa3 fused in hipfire",
    source: "crates/hipfire-arch-qwen35/src/qwen35.rs:29485",
};

/// `HIPFIRE_PARO_FUSED_PACK2` — Runtime variable controlling paro fused pack2 in hipfire
pub const ENV_HIPFIRE_PARO_FUSED_PACK2: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_PARO_FUSED_PACK2",
    description: "Runtime variable controlling paro fused pack2 in hipfire",
    source: "crates/hipfire-rdna/src/dispatch/fused.rs:332",
};

/// `HIPFIRE_PARO_FUSE_RMSNORM` — time per call. Net loss on every site. Default OFF; explicit opt-in for
pub const ENV_HIPFIRE_PARO_FUSE_RMSNORM: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_PARO_FUSE_RMSNORM",
    description: "time per call. Net loss on every site. Default OFF; explicit opt-in for",
    source: "crates/hipfire-runtime/src/weights.rs:768",
};

/// `HIPFIRE_PARO_GATE_UP_FUSED` — Runtime variable controlling paro gate up fused in hipfire
pub const ENV_HIPFIRE_PARO_GATE_UP_FUSED: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_PARO_GATE_UP_FUSED",
    description: "Runtime variable controlling paro gate up fused in hipfire",
    source: "crates/hipfire-arch-qwen35/src/qwen35.rs:29083",
};

/// `HIPFIRE_PARO_LA2_FUSED` — Runtime variable controlling paro la2 fused in hipfire
pub const ENV_HIPFIRE_PARO_LA2_FUSED: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_PARO_LA2_FUSED",
    description: "Runtime variable controlling paro la2 fused in hipfire",
    source: "crates/hipfire-arch-qwen35/src/qwen35.rs:29185",
};

/// `HIPFIRE_PARO_LA4_FUSED` — Runtime variable controlling paro la4 fused in hipfire
pub const ENV_HIPFIRE_PARO_LA4_FUSED: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_PARO_LA4_FUSED",
    description: "Runtime variable controlling paro la4 fused in hipfire",
    source: "crates/hipfire-arch-qwen35/src/qwen35.rs:29181",
};

/// `HIPFIRE_PARO_LA_GATES_MQ4G128` — Used to configure runtime execution by explicitly setting "HIPFIRE_PARO_LA_GATES_MQ4G128"
pub const ENV_HIPFIRE_PARO_LA_GATES_MQ4G128: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_PARO_LA_GATES_MQ4G128",
    description: "Used to configure runtime execution by explicitly setting \"HIPFIRE_PARO_LA_GATES_MQ4G128\"",
    source: "crates/hipfire-arch-qwen35/src/paro_la_gates_codec.rs:295",
};

/// `HIPFIRE_PARO_PACK1` — Runtime variable controlling paro pack1 in hipfire
pub const ENV_HIPFIRE_PARO_PACK1: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_PARO_PACK1",
    description: "Runtime variable controlling paro pack1 in hipfire",
    source: "crates/hipfire-rdna/src/dispatch/gemv.rs:1097",
};

/// `HIPFIRE_PARO_PACK2` — Runtime variable controlling paro pack2 in hipfire
pub const ENV_HIPFIRE_PARO_PACK2: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_PARO_PACK2",
    description: "Runtime variable controlling paro pack2 in hipfire",
    source: "crates/hipfire-rdna/src/dispatch/gemv.rs:1100",
};

/// `HIPFIRE_PARO_PACK4` — Runtime variable controlling paro pack4 in hipfire
pub const ENV_HIPFIRE_PARO_PACK4: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_PARO_PACK4",
    description: "Runtime variable controlling paro pack4 in hipfire",
    source: "crates/hipfire-rdna/src/dispatch/gemv.rs:1103",
};

/// `HIPFIRE_PARO_PREROTATE` — Runtime variable controlling paro prerotate in hipfire
pub const ENV_HIPFIRE_PARO_PREROTATE: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_PARO_PREROTATE",
    description: "Runtime variable controlling paro prerotate in hipfire",
    source: "crates/hipfire-runtime/src/weights.rs:1343",
};

/// `HIPFIRE_PARO_SHARED_PAIRS` — Runtime variable controlling paro shared pairs in hipfire
pub const ENV_HIPFIRE_PARO_SHARED_PAIRS: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_PARO_SHARED_PAIRS",
    description: "Runtime variable controlling paro shared pairs in hipfire",
    source: "crates/hipfire-rdna/src/dispatch/fused.rs:326",
};

/// `HIPFIRE_PARO_SWIGLU_FUSED` — Runtime variable controlling paro swiglu fused in hipfire
pub const ENV_HIPFIRE_PARO_SWIGLU_FUSED: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_PARO_SWIGLU_FUSED",
    description: "Runtime variable controlling paro swiglu fused in hipfire",
    source: "crates/hipfire-runtime/src/weights.rs:1360",
};

/// `HIPFIRE_PERF_BASELINE` — Runtime variable controlling perf baseline in hipfire
pub const ENV_HIPFIRE_PERF_BASELINE: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_PERF_BASELINE",
    description: "Runtime variable controlling perf baseline in hipfire",
    source: "crates/hipfire-eval/src/executor_examples.rs:1325",
};

/// `HIPFIRE_PERF_BASELINE_DIR` — Runtime variable controlling perf baseline dir in hipfire
pub const ENV_HIPFIRE_PERF_BASELINE_DIR: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_PERF_BASELINE_DIR",
    description: "Runtime variable controlling perf baseline dir in hipfire",
    source: "crates/hipfire-eval/src/executor_examples.rs:1338",
};

/// `HIPFIRE_PERPLEXITY_BIN` — Runtime variable controlling perplexity bin in hipfire
pub const ENV_HIPFIRE_PERPLEXITY_BIN: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_PERPLEXITY_BIN",
    description: "Runtime variable controlling perplexity bin in hipfire",
    source: "crates/hipfire-eval/src/lib.rs:1224",
};

/// `HIPFIRE_PFLASH_DAEMON_LABELS` — Runtime variable controlling pflash daemon labels in hipfire
pub const ENV_HIPFIRE_PFLASH_DAEMON_LABELS: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_PFLASH_DAEMON_LABELS",
    description: "Runtime variable controlling pflash daemon labels in hipfire",
    source: "crates/hipfire-train/examples/ssm_drafter_train.rs:66",
};

/// `HIPFIRE_PFLASH_DEBUG` — Runtime variable controlling pflash debug in hipfire
pub const ENV_HIPFIRE_PFLASH_DEBUG: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_PFLASH_DEBUG",
    description: "Runtime variable controlling pflash debug in hipfire",
    source: "crates/hipfire-serving-core/src/generate.rs:2498",
};

/// `HIPFIRE_PFLASH_DRAFTER_KV` — Selects behavior from recognized values
pub const ENV_HIPFIRE_PFLASH_DRAFTER_KV: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_PFLASH_DRAFTER_KV",
    description: "Selects behavior from recognized values",
    source: "crates/hipfire-arch-qwen35/src/pflash.rs:534",
};

/// `HIPFIRE_PFLASH_DRAFTER_STATE` — Hybrid drafter only stores K (and V for chat-path) at
pub const ENV_HIPFIRE_PFLASH_DRAFTER_STATE: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_PFLASH_DRAFTER_STATE",
    description: "Hybrid drafter only stores K (and V for chat-path) at",
    source: "crates/hipfire-arch-qwen35/src/pflash.rs:621",
};

/// `HIPFIRE_PFLASH_FRESH` — resume: reload weights + AdamW state from the checkpoint unless FRESH=1
pub const ENV_HIPFIRE_PFLASH_FRESH: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_PFLASH_FRESH",
    description: "resume: reload weights + AdamW state from the checkpoint unless FRESH=1",
    source: "crates/hipfire-train/examples/pflash_drafter_train.rs:366",
};

/// `HIPFIRE_PFLASH_NIAH_BENCH_BIN` — Runtime variable controlling pflash niah bench bin in hipfire
pub const ENV_HIPFIRE_PFLASH_NIAH_BENCH_BIN: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_PFLASH_NIAH_BENCH_BIN",
    description: "Runtime variable controlling pflash niah bench bin in hipfire",
    source: "crates/hipfire-eval/src/lib.rs:1179",
};

/// `HIPFIRE_PFLASH_REPORT_TRAIN` — Runtime variable controlling pflash report train in hipfire
pub const ENV_HIPFIRE_PFLASH_REPORT_TRAIN: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_PFLASH_REPORT_TRAIN",
    description: "Runtime variable controlling pflash report train in hipfire",
    source: "crates/hipfire-train/examples/ssm_drafter_train.rs:115",
};

/// `HIPFIRE_PFLASH_SCORE_LAYER` — Parses "HIPFIRE_PFLASH_SCORE_LAYER" with fallback defaults
pub const ENV_HIPFIRE_PFLASH_SCORE_LAYER: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_PFLASH_SCORE_LAYER",
    description: "Parses \"HIPFIRE_PFLASH_SCORE_LAYER\" with fallback defaults",
    source: "crates/hipfire-arch-qwen35/src/pflash.rs:1082",
};

/// `HIPFIRE_PP_DFLASH` — Environment toggle value controls runtime behavior
pub const ENV_HIPFIRE_PP_DFLASH: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_PP_DFLASH",
    description: "Environment toggle value controls runtime behavior",
    source: "crates/hipfire-daemon/src/main.rs:3069",
};

/// `HIPFIRE_PP_LAYERS` — Runtime variable controlling pp layers in hipfire
pub const ENV_HIPFIRE_PP_LAYERS: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_PP_LAYERS",
    description: "Runtime variable controlling pp layers in hipfire",
    source: "crates/hipfire-serving-core/src/load.rs:2718",
};

/// `HIPFIRE_PP_PARITY_MODEL` — Runtime variable controlling pp parity model in hipfire
pub const ENV_HIPFIRE_PP_PARITY_MODEL: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_PP_PARITY_MODEL",
    description: "Runtime variable controlling pp parity model in hipfire",
    source: "crates/hipfire-arch-qwen35/tests/pp_parity.rs:197",
};

/// `HIPFIRE_PP_PFLASH` — Environment toggle value controls runtime behavior
pub const ENV_HIPFIRE_PP_PFLASH: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_PP_PFLASH",
    description: "Environment toggle value controls runtime behavior",
    source: "crates/hipfire-daemon/src/main.rs:3087",
};

/// `HIPFIRE_PREFILL_ALPHA` — Runtime variable controlling prefill alpha in hipfire
pub const ENV_HIPFIRE_PREFILL_ALPHA: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_PREFILL_ALPHA",
    description: "Runtime variable controlling prefill alpha in hipfire",
    source: "crates/hipfire-arch-qwen35/src/pflash.rs:125",
};

/// `HIPFIRE_PREFILL_BATCHED` — Enabled by default; set to 0 to disable
pub const ENV_HIPFIRE_PREFILL_BATCHED: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_PREFILL_BATCHED",
    description: "Enabled by default; set to 0 to disable",
    source: "crates/hipfire-runtime/src/config.rs:86",
};

/// `HIPFIRE_PREFILL_BLOCK` — Runtime variable controlling prefill block in hipfire
pub const ENV_HIPFIRE_PREFILL_BLOCK: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_PREFILL_BLOCK",
    description: "Runtime variable controlling prefill block in hipfire",
    source: "crates/hipfire-arch-qwen35/src/pflash.rs:145",
};

/// `HIPFIRE_PREFILL_COMPRESSION` — Runtime variable controlling prefill compression in hipfire
pub const ENV_HIPFIRE_PREFILL_COMPRESSION: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_PREFILL_COMPRESSION",
    description: "Runtime variable controlling prefill compression in hipfire",
    source: "crates/hipfire-arch-qwen35/src/pflash.rs:105",
};

/// `HIPFIRE_PREFILL_DRAFTER` — Runtime variable controlling prefill drafter in hipfire
pub const ENV_HIPFIRE_PREFILL_DRAFTER: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_PREFILL_DRAFTER",
    description: "Runtime variable controlling prefill drafter in hipfire",
    source: "crates/hipfire-arch-qwen35/src/pflash.rs:158",
};

/// `HIPFIRE_PREFILL_KEEP_RATIO` — Runtime variable controlling prefill keep ratio in hipfire
pub const ENV_HIPFIRE_PREFILL_KEEP_RATIO: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_PREFILL_KEEP_RATIO",
    description: "Runtime variable controlling prefill keep ratio in hipfire",
    source: "crates/hipfire-arch-qwen35/src/pflash.rs:115",
};

/// `HIPFIRE_PREFILL_MAX_BATCH` — Runtime variable controlling prefill max batch in hipfire
pub const ENV_HIPFIRE_PREFILL_MAX_BATCH: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_PREFILL_MAX_BATCH",
    description: "Runtime variable controlling prefill max batch in hipfire",
    source: "crates/hipfire-serving-core/src/qwen35_prefill.rs:1431",
};

/// `HIPFIRE_PREFILL_MAX_LAYER` — Runtime variable controlling prefill max layer in hipfire
pub const ENV_HIPFIRE_PREFILL_MAX_LAYER: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_PREFILL_MAX_LAYER",
    description: "Runtime variable controlling prefill max layer in hipfire",
    source: "crates/hipfire-arch-qwen35/src/qwen35.rs:13605",
};

/// `HIPFIRE_PREFILL_MIN_KEEP` — Runtime variable controlling prefill min keep in hipfire
pub const ENV_HIPFIRE_PREFILL_MIN_KEEP: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_PREFILL_MIN_KEEP",
    description: "Runtime variable controlling prefill min keep in hipfire",
    source: "crates/hipfire-arch-qwen35/src/pflash.rs:130",
};

/// `HIPFIRE_PREFILL_PROFILE` — Enabled when set to 1
pub const ENV_HIPFIRE_PREFILL_PROFILE: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_PREFILL_PROFILE",
    description: "Enabled when set to 1",
    source: "crates/hipfire-arch-qwen35/src/pflash.rs:155",
};

/// `HIPFIRE_PREFILL_RECENT` — Runtime variable controlling prefill recent in hipfire
pub const ENV_HIPFIRE_PREFILL_RECENT: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_PREFILL_RECENT",
    description: "Runtime variable controlling prefill recent in hipfire",
    source: "crates/hipfire-arch-qwen35/src/pflash.rs:140",
};

/// `HIPFIRE_PREFILL_REUSE_PBS` — Used to configure runtime execution by explicitly setting "HIPFIRE_PREFILL_REUSE_PBS"
pub const ENV_HIPFIRE_PREFILL_REUSE_PBS: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_PREFILL_REUSE_PBS",
    description:
        "Used to configure runtime execution by explicitly setting \"HIPFIRE_PREFILL_REUSE_PBS\"",
    source: "crates/hipfire-runtime/examples/prefill_microbench.rs:148",
};

/// `HIPFIRE_PREFILL_SINK` — Runtime variable controlling prefill sink in hipfire
pub const ENV_HIPFIRE_PREFILL_SINK: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_PREFILL_SINK",
    description: "Runtime variable controlling prefill sink in hipfire",
    source: "crates/hipfire-arch-qwen35/src/pflash.rs:135",
};

/// `HIPFIRE_PREFILL_SPARSE_THRESHOLD` — Runtime variable controlling prefill sparse threshold in hipfire
pub const ENV_HIPFIRE_PREFILL_SPARSE_THRESHOLD: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_PREFILL_SPARSE_THRESHOLD",
    description: "Runtime variable controlling prefill sparse threshold in hipfire",
    source: "crates/hipfire-arch-qwen35/src/pflash.rs:150",
};

/// `HIPFIRE_PREFILL_STOP_AFTER_LA_LAYER` — Parses "HIPFIRE_PREFILL_STOP_AFTER_LA_LAYER" with fallback defaults
pub const ENV_HIPFIRE_PREFILL_STOP_AFTER_LA_LAYER: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_PREFILL_STOP_AFTER_LA_LAYER",
    description: "Parses \"HIPFIRE_PREFILL_STOP_AFTER_LA_LAYER\" with fallback defaults",
    source: "crates/hipfire-arch-qwen35/src/qwen35.rs:16978",
};

/// `HIPFIRE_PREFILL_STOP_STAGE` — Runtime variable controlling prefill stop stage in hipfire
pub const ENV_HIPFIRE_PREFILL_STOP_STAGE: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_PREFILL_STOP_STAGE",
    description: "Runtime variable controlling prefill stop stage in hipfire",
    source: "crates/hipfire-arch-qwen35/src/qwen35.rs:16984",
};

/// `HIPFIRE_PREFILL_STOP_STAGE_LAYER` — Parses "HIPFIRE_PREFILL_STOP_STAGE_LAYER" with fallback defaults
pub const ENV_HIPFIRE_PREFILL_STOP_STAGE_LAYER: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_PREFILL_STOP_STAGE_LAYER",
    description: "Parses \"HIPFIRE_PREFILL_STOP_STAGE_LAYER\" with fallback defaults",
    source: "crates/hipfire-arch-qwen35/src/qwen35.rs:16981",
};

/// `HIPFIRE_PREFILL_THRESHOLD` — Runtime variable controlling prefill threshold in hipfire
pub const ENV_HIPFIRE_PREFILL_THRESHOLD: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_PREFILL_THRESHOLD",
    description: "Runtime variable controlling prefill threshold in hipfire",
    source: "crates/hipfire-arch-qwen35/src/pflash.rs:110",
};

/// `HIPFIRE_PREFIX_BOUNDARY_CHECKPOINTS` — Interprets "HIPFIRE_PREFIX_BOUNDARY_CHECKPOINTS" from environment to select behavior
pub const ENV_HIPFIRE_PREFIX_BOUNDARY_CHECKPOINTS: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_PREFIX_BOUNDARY_CHECKPOINTS",
    description:
        "Interprets \"HIPFIRE_PREFIX_BOUNDARY_CHECKPOINTS\" from environment to select behavior",
    source: "crates/hipfire-serving-core/src/qwen35_prefill.rs:459",
};

/// `HIPFIRE_PROFILE` — HIPFIRE_PROFILE=1 + HIPFIRE_PROFILE_CYCLES=N: per-kernel profiling
pub const ENV_HIPFIRE_PROFILE: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_PROFILE",
    description: "HIPFIRE_PROFILE=1 + HIPFIRE_PROFILE_CYCLES=N: per-kernel profiling",
    source: "crates/hipfire-runtime/examples/mtp_only_demo.rs:515",
};

/// `HIPFIRE_PROFILE_CYCLES` — Runtime variable controlling profile cycles in hipfire
pub const ENV_HIPFIRE_PROFILE_CYCLES: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_PROFILE_CYCLES",
    description: "Runtime variable controlling profile cycles in hipfire",
    source: "crates/hipfire-runtime/examples/mtp_only_demo.rs:516",
};

/// `HIPFIRE_PROFILE_DECODE` — Enabled when set to 1
pub const ENV_HIPFIRE_PROFILE_DECODE: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_PROFILE_DECODE",
    description: "Enabled when set to 1",
    source: "crates/hipfire-runtime/examples/bench_qwen35_speed.rs:579",
};

/// `HIPFIRE_PROMPT_HEAT_JSON` — Enabled when set to 1
pub const ENV_HIPFIRE_PROMPT_HEAT_JSON: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_PROMPT_HEAT_JSON",
    description: "Enabled when set to 1",
    source: "crates/hipfire-runtime/src/config.rs:62",
};

/// `HIPFIRE_PROMPT_HEAT_LIMIT` — Runtime variable controlling prompt heat limit in hipfire
pub const ENV_HIPFIRE_PROMPT_HEAT_LIMIT: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_PROMPT_HEAT_LIMIT",
    description: "Runtime variable controlling prompt heat limit in hipfire",
    source: "crates/hipfire-runtime/src/config.rs:63",
};

/// `HIPFIRE_PROMPT_TOKEN_HEAT` — Enabled when set to 1
pub const ENV_HIPFIRE_PROMPT_TOKEN_HEAT: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_PROMPT_TOKEN_HEAT",
    description: "Enabled when set to 1",
    source: "crates/hipfire-runtime/src/config.rs:60",
};

/// `HIPFIRE_PYTHON` — Python interpreter used by the no-GPU CI shell gate for Python tooling and tests
pub const ENV_HIPFIRE_PYTHON: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_PYTHON",
    description: "Python interpreter used by the no-GPU CI shell gate for Python tooling and tests",
    source: ".github/CONTRIBUTING.md:86",
};

/// `HIPFIRE_Q8_BATCHED_LEGACY` — Environment toggle value controls runtime behavior
pub const ENV_HIPFIRE_Q8_BATCHED_LEGACY: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_Q8_BATCHED_LEGACY",
    description: "Environment toggle value controls runtime behavior",
    source: "crates/hipfire-rdna/src/feature_flags.rs:295",
};

/// `HIPFIRE_Q8_FA_ATTENTION_IGNORE_TREE_BIAS` — Interprets "HIPFIRE_Q8_FA_ATTENTION_IGNORE_TREE_BIAS" from environment to select behavior
pub const ENV_HIPFIRE_Q8_FA_ATTENTION_IGNORE_TREE_BIAS: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_Q8_FA_ATTENTION_IGNORE_TREE_BIAS",
    description: "Interprets \"HIPFIRE_Q8_FA_ATTENTION_IGNORE_TREE_BIAS\" from environment to select behavior",
    source: "crates/hipfire-arch-qwen35/src/qwen35.rs:12943",
};

/// `HIPFIRE_Q8_FA_ATTENTION_ROW_LOOP` — Interprets "HIPFIRE_Q8_FA_ATTENTION_ROW_LOOP" from environment to select behavior
pub const ENV_HIPFIRE_Q8_FA_ATTENTION_ROW_LOOP: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_Q8_FA_ATTENTION_ROW_LOOP",
    description:
        "Interprets \"HIPFIRE_Q8_FA_ATTENTION_ROW_LOOP\" from environment to select behavior",
    source: "crates/hipfire-arch-qwen35/src/qwen35.rs:12907",
};

/// `HIPFIRE_Q8_FA_ATTENTION_SCALAR_LOOP` — Interprets "HIPFIRE_Q8_FA_ATTENTION_SCALAR_LOOP" from environment to select behavior
pub const ENV_HIPFIRE_Q8_FA_ATTENTION_SCALAR_LOOP: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_Q8_FA_ATTENTION_SCALAR_LOOP",
    description:
        "Interprets \"HIPFIRE_Q8_FA_ATTENTION_SCALAR_LOOP\" from environment to select behavior",
    source: "crates/hipfire-arch-qwen35/src/qwen35.rs:12919",
};

/// `HIPFIRE_Q8_FA_ATTENTION_SERIAL_KV_LOOP` — Interprets "HIPFIRE_Q8_FA_ATTENTION_SERIAL_KV_LOOP" from environment to select behavior
pub const ENV_HIPFIRE_Q8_FA_ATTENTION_SERIAL_KV_LOOP: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_Q8_FA_ATTENTION_SERIAL_KV_LOOP",
    description:
        "Interprets \"HIPFIRE_Q8_FA_ATTENTION_SERIAL_KV_LOOP\" from environment to select behavior",
    source: "crates/hipfire-arch-qwen35/src/qwen35.rs:12931",
};

/// `HIPFIRE_Q8_GATE_UP_4W` — Disabled when set to 0
pub const ENV_HIPFIRE_Q8_GATE_UP_4W: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_Q8_GATE_UP_4W",
    description: "Disabled when set to 0",
    source: "crates/hipfire-rdna/examples/test_gemm_q8_gate_up_wmma.rs:167",
};

/// `HIPFIRE_Q8_GATE_UP_BENCH` — Enabled when set to 1
pub const ENV_HIPFIRE_Q8_GATE_UP_BENCH: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_Q8_GATE_UP_BENCH",
    description: "Enabled when set to 1",
    source: "crates/hipfire-rdna/examples/test_gemm_q8_gate_up_wmma.rs:166",
};

/// `HIPFIRE_Q8_GDN_VERIFY_PER_TOKEN` — Interprets "HIPFIRE_Q8_GDN_VERIFY_PER_TOKEN" from environment to select behavior
pub const ENV_HIPFIRE_Q8_GDN_VERIFY_PER_TOKEN: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_Q8_GDN_VERIFY_PER_TOKEN",
    description:
        "Interprets \"HIPFIRE_Q8_GDN_VERIFY_PER_TOKEN\" from environment to select behavior",
    source: "crates/hipfire-arch-qwen35/src/qwen35.rs:12955",
};

/// `HIPFIRE_Q8_GDN_VERIFY_SERIAL_FRAMES` — Interprets "HIPFIRE_Q8_GDN_VERIFY_SERIAL_FRAMES" from environment to select behavior
pub const ENV_HIPFIRE_Q8_GDN_VERIFY_SERIAL_FRAMES: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_Q8_GDN_VERIFY_SERIAL_FRAMES",
    description:
        "Interprets \"HIPFIRE_Q8_GDN_VERIFY_SERIAL_FRAMES\" from environment to select behavior",
    source: "crates/hipfire-arch-qwen35/src/qwen35.rs:12967",
};

/// `HIPFIRE_Q8_WMMA_4W` — Environment toggle value controls runtime behavior
pub const ENV_HIPFIRE_Q8_WMMA_4W: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_Q8_WMMA_4W",
    description: "Environment toggle value controls runtime behavior",
    source: "crates/hipfire-rdna/src/dispatch/gemm_misc.rs:788",
};

/// `HIPFIRE_Q8_WMMA_X64` — Environment toggle value controls runtime behavior
pub const ENV_HIPFIRE_Q8_WMMA_X64: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_Q8_WMMA_X64",
    description: "Environment toggle value controls runtime behavior",
    source: "crates/hipfire-rdna/src/dispatch/gemm_misc.rs:620",
};

/// `HIPFIRE_QAT_KVNOISE` — Teacher must run CLEAN — force KV-noise off during its precompute
pub const ENV_HIPFIRE_QAT_KVNOISE: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_QAT_KVNOISE",
    description: "Teacher must run CLEAN — force KV-noise off during its precompute",
    source: "crates/hipfire-train/examples/qat_w3_kvarn.rs:89",
};

/// `HIPFIRE_QA_KV_MODES` — Defaults to q8,asym4,asym3,asym2 when unset
pub const ENV_HIPFIRE_QA_KV_MODES: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_QA_KV_MODES",
    description: "Defaults to q8,asym4,asym3,asym2 when unset",
    source: "crates/hipfire-runtime/examples/test_inferenceQA.rs:770",
};

/// `HIPFIRE_QTIP_BBT_ALPHA` — BBT-spectral influence scaling (SpectralLLM): HIPFIRE_QTIP_BBT_ALPHA=α
pub const ENV_HIPFIRE_QTIP_BBT_ALPHA: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_QTIP_BBT_ALPHA",
    description: "BBT-spectral influence scaling (SpectralLLM): HIPFIRE_QTIP_BBT_ALPHA=α",
    source: "crates/hipfire-quantize/src/main.rs:8837",
};

/// `HIPFIRE_QTIP_BEAM` — K-map gate: applies to MoE models by default. Dense models opt in
pub const ENV_HIPFIRE_QTIP_BEAM: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_QTIP_BEAM",
    description: "K-map gate: applies to MoE models by default. Dense models opt in",
    source: "crates/hipfire-quantize/src/main.rs:5516",
};

/// `HIPFIRE_QTIP_CODEBOOK` — Environment toggle value controls runtime behavior
pub const ENV_HIPFIRE_QTIP_CODEBOOK: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_QTIP_CODEBOOK",
    description: "Environment toggle value controls runtime behavior",
    source: "crates/hipfire-quantize/src/main.rs:4934",
};

/// `HIPFIRE_QTIP_EVAL_ST` — Selects behavior from recognized values
pub const ENV_HIPFIRE_QTIP_EVAL_ST: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_QTIP_EVAL_ST",
    description: "Selects behavior from recognized values",
    source: "crates/hipfire-quantize/src/qtip.rs:894",
};

/// `HIPFIRE_QTIP_HESSIAN` — Runtime variable controlling qtip hessian in hipfire
pub const ENV_HIPFIRE_QTIP_HESSIAN: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_QTIP_HESSIAN",
    description: "Runtime variable controlling qtip hessian in hipfire",
    source: "crates/hipfire-quantize/src/main.rs:4956",
};

/// `HIPFIRE_QUANTIZE_BIN` — Runtime variable controlling quantize bin in hipfire
pub const ENV_HIPFIRE_QUANTIZE_BIN: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_QUANTIZE_BIN",
    description: "Runtime variable controlling quantize bin in hipfire",
    source: "crates/hipfire-eval/src/executor_tinyquant.rs:126",
};

/// `HIPFIRE_QUANT_CALIB` — Runtime variable controlling quant calib in hipfire
pub const ENV_HIPFIRE_QUANT_CALIB: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_QUANT_CALIB",
    description: "Runtime variable controlling quant calib in hipfire",
    source: "crates/hipfire-diffusion/src/tests/misc.rs:267",
};

/// `HIPFIRE_QUANT_CANDS` — Runtime variable controlling quant cands in hipfire
pub const ENV_HIPFIRE_QUANT_CANDS: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_QUANT_CANDS",
    description: "Runtime variable controlling quant cands in hipfire",
    source: "crates/hipfire-diffusion/src/tests/quant.rs:396",
};

/// `HIPFIRE_QUANT_DIAG_PATH` — Runtime variable controlling quant diag path in hipfire
pub const ENV_HIPFIRE_QUANT_DIAG_PATH: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_QUANT_DIAG_PATH",
    description: "Runtime variable controlling quant diag path in hipfire",
    source: "crates/hipfire-quantize/src/main.rs:12030",
};

/// `HIPFIRE_QUANT_SRC` — Runtime variable controlling quant src in hipfire
pub const ENV_HIPFIRE_QUANT_SRC: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_QUANT_SRC",
    description: "Runtime variable controlling quant src in hipfire",
    source: "crates/hipfire-diffusion/src/tests/quant.rs:393",
};

/// `HIPFIRE_QUANT_THREADS` — Runtime variable controlling quant threads in hipfire
pub const ENV_HIPFIRE_QUANT_THREADS: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_QUANT_THREADS",
    description: "Runtime variable controlling quant threads in hipfire",
    source: "crates/hipfire-quantize/src/main.rs:4705",
};

/// `HIPFIRE_QWEN35_DECODE_BATCH` — Defaults to auto when unset
pub const ENV_HIPFIRE_QWEN35_DECODE_BATCH: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_QWEN35_DECODE_BATCH",
    description: "Defaults to auto when unset",
    source: "crates/hipfire-serving-core/src/qwen35_decode.rs:360",
};

/// `HIPFIRE_QWEN35_DECODE_BATCH_MAX` — Runtime variable controlling qwen35 decode batch max in hipfire
pub const ENV_HIPFIRE_QWEN35_DECODE_BATCH_MAX: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_QWEN35_DECODE_BATCH_MAX",
    description: "Runtime variable controlling qwen35 decode batch max in hipfire",
    source: "crates/hipfire-serving-core/src/qwen35_decode.rs:484",
};

/// `HIPFIRE_QWEN35_DECODE_INTERNAL_PARITY` — Interprets "HIPFIRE_QWEN35_DECODE_INTERNAL_PARITY" from environment to select behavior
pub const ENV_HIPFIRE_QWEN35_DECODE_INTERNAL_PARITY: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_QWEN35_DECODE_INTERNAL_PARITY",
    description:
        "Interprets \"HIPFIRE_QWEN35_DECODE_INTERNAL_PARITY\" from environment to select behavior",
    source: "crates/hipfire-serving-core/src/qwen35_decode.rs:506",
};

/// `HIPFIRE_QWEN35_DECODE_NATIVE_MULTIROW` — Interprets "HIPFIRE_QWEN35_DECODE_NATIVE_MULTIROW" from environment to select behavior
pub const ENV_HIPFIRE_QWEN35_DECODE_NATIVE_MULTIROW: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_QWEN35_DECODE_NATIVE_MULTIROW",
    description:
        "Interprets \"HIPFIRE_QWEN35_DECODE_NATIVE_MULTIROW\" from environment to select behavior",
    source: "crates/hipfire-serving-core/src/qwen35_decode.rs:495",
};

/// `HIPFIRE_QWEN35_EXPERT_CACHE_MB` — Parses "HIPFIRE_QWEN35_EXPERT_CACHE_MB" with fallback defaults
pub const ENV_HIPFIRE_QWEN35_EXPERT_CACHE_MB: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_QWEN35_EXPERT_CACHE_MB",
    description: "Parses \"HIPFIRE_QWEN35_EXPERT_CACHE_MB\" with fallback defaults",
    source: "crates/hipfire-arch-qwen35/src/qwen35.rs:701",
};

/// `HIPFIRE_QWEN35_EXPERT_CACHE_TRACE` — Interprets "HIPFIRE_QWEN35_EXPERT_CACHE_TRACE" from environment to select behavior
pub const ENV_HIPFIRE_QWEN35_EXPERT_CACHE_TRACE: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_QWEN35_EXPERT_CACHE_TRACE",
    description:
        "Interprets \"HIPFIRE_QWEN35_EXPERT_CACHE_TRACE\" from environment to select behavior",
    source: "crates/hipfire-arch-qwen35/src/qwen35.rs:5555",
};

/// `HIPFIRE_QWEN35_FFN_BF16` — Selects behavior from recognized values
pub const ENV_HIPFIRE_QWEN35_FFN_BF16: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_QWEN35_FFN_BF16",
    description: "Selects behavior from recognized values",
    source: "crates/hipfire-arch-qwen35/src/ffn_bf16.rs:57",
};

/// `HIPFIRE_QWEN35_FFN_BF16_LAYER` — Selects behavior from recognized values
pub const ENV_HIPFIRE_QWEN35_FFN_BF16_LAYER: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_QWEN35_FFN_BF16_LAYER",
    description: "Selects behavior from recognized values",
    source: "crates/hipfire-arch-qwen35/src/ffn_bf16.rs:71",
};

/// `HIPFIRE_QWEN35_FFN_BF16_TRACE` — Parses "HIPFIRE_QWEN35_FFN_BF16_TRACE" with fallback defaults
pub const ENV_HIPFIRE_QWEN35_FFN_BF16_TRACE: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_QWEN35_FFN_BF16_TRACE",
    description: "Parses \"HIPFIRE_QWEN35_FFN_BF16_TRACE\" with fallback defaults",
    source: "crates/hipfire-arch-qwen35/src/ffn_bf16.rs:83",
};

/// `HIPFIRE_QWEN35_FINITE_TRACE` — Runtime variable controlling qwen35 finite trace in hipfire
pub const ENV_HIPFIRE_QWEN35_FINITE_TRACE: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_QWEN35_FINITE_TRACE",
    description: "Runtime variable controlling qwen35 finite trace in hipfire",
    source: "crates/hipfire-arch-qwen35/src/qwen35.rs:14297",
};

/// `HIPFIRE_QWEN35_PAGED_EXPERTS` — Interprets "HIPFIRE_QWEN35_PAGED_EXPERTS" from environment to select behavior
pub const ENV_HIPFIRE_QWEN35_PAGED_EXPERTS: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_QWEN35_PAGED_EXPERTS",
    description: "Interprets \"HIPFIRE_QWEN35_PAGED_EXPERTS\" from environment to select behavior",
    source: "crates/hipfire-arch-qwen35/src/qwen35.rs:4271",
};

/// `HIPFIRE_QWEN35_PREFILL_SESSION_BATCH` — Runtime variable controlling qwen35 prefill session batch in hipfire
pub const ENV_HIPFIRE_QWEN35_PREFILL_SESSION_BATCH: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_QWEN35_PREFILL_SESSION_BATCH",
    description: "Runtime variable controlling qwen35 prefill session batch in hipfire",
    source: "crates/hipfire-serving-core/src/qwen35_prefill.rs:1756",
};

/// `HIPFIRE_QWEN35_ROUTED_ONLY_MOE_FORWARD` — Runtime variable controlling qwen35 routed only moe forward in hipfire
pub const ENV_HIPFIRE_QWEN35_ROUTED_ONLY_MOE_FORWARD: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_QWEN35_ROUTED_ONLY_MOE_FORWARD",
    description: "Runtime variable controlling qwen35 routed only moe forward in hipfire",
    source: "crates/hipfire-arch-qwen35/src/qwen35.rs:7828",
};

/// `HIPFIRE_QWEN35_STAGE_SYNC` — Runtime variable controlling qwen35 stage sync in hipfire
pub const ENV_HIPFIRE_QWEN35_STAGE_SYNC: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_QWEN35_STAGE_SYNC",
    description: "Runtime variable controlling qwen35 stage sync in hipfire",
    source: "crates/hipfire-arch-qwen35/src/qwen35.rs:14331",
};

/// `HIPFIRE_QWEN35_STAGE_TRACE` — Runtime variable controlling qwen35 stage trace in hipfire
pub const ENV_HIPFIRE_QWEN35_STAGE_TRACE: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_QWEN35_STAGE_TRACE",
    description: "Runtime variable controlling qwen35 stage trace in hipfire",
    source: "crates/hipfire-arch-qwen35/src/qwen35.rs:14325",
};

/// `HIPFIRE_QWEN35_XDNA1_INSTR` — Runtime variable controlling qwen35 xdna1 instr in hipfire
pub const ENV_HIPFIRE_QWEN35_XDNA1_INSTR: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_QWEN35_XDNA1_INSTR",
    description: "Runtime variable controlling qwen35 xdna1 instr in hipfire",
    source: "crates/hipfire-npu/src/lib.rs:76",
};

/// `HIPFIRE_QWEN35_XDNA1_XCLBIN` — Runtime variable controlling qwen35 xdna1 xclbin in hipfire
pub const ENV_HIPFIRE_QWEN35_XDNA1_XCLBIN: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_QWEN35_XDNA1_XCLBIN",
    description: "Runtime variable controlling qwen35 xdna1 xclbin in hipfire",
    source: "crates/hipfire-npu/src/lib.rs:73",
};

/// `HIPFIRE_QWEN3_DSPARK_CONF_THRESHOLD` — conf_threshold ladder: env > 0.1 default (source's sweep-tuned default)
pub const ENV_HIPFIRE_QWEN3_DSPARK_CONF_THRESHOLD: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_QWEN3_DSPARK_CONF_THRESHOLD",
    description: "conf_threshold ladder: env > 0.1 default (source's sweep-tuned default)",
    source: "crates/hipfire-serving-core/src/load.rs:3334",
};

/// `HIPFIRE_RDNA2_VARIANT` — Runtime variable controlling rdna2 variant in hipfire
pub const ENV_HIPFIRE_RDNA2_VARIANT: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_RDNA2_VARIANT",
    description: "Runtime variable controlling rdna2 variant in hipfire",
    source: "crates/hipfire-rdna/src/feature_flags.rs:313",
};

/// `HIPFIRE_RECOVER_MODE` — HIPFIRE_RECOVER_MODE=lora+norms → LoRA + layernorms (default, more capacity)
pub const ENV_HIPFIRE_RECOVER_MODE: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_RECOVER_MODE",
    description: "HIPFIRE_RECOVER_MODE=lora+norms → LoRA + layernorms (default, more capacity)",
    source: "crates/hipfire-train/examples/recovery_generalization_supra50m.rs:158",
};

/// `HIPFIRE_RECOVER_NOISE` — KVarN+CASK: tail queries read merged cold keys strictly in their past). 0 =
pub const ENV_HIPFIRE_RECOVER_NOISE: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_RECOVER_NOISE",
    description: "KVarN+CASK: tail queries read merged cold keys strictly in their past). 0 =",
    source: "crates/hipfire-train/examples/recovery_generalization_supra50m.rs:157",
};

/// `HIPFIRE_REPLAY_GRAPH` — Enabled when set to 1
pub const ENV_HIPFIRE_REPLAY_GRAPH: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_REPLAY_GRAPH",
    description: "Enabled when set to 1",
    source: "crates/hipfire-arch-qwen35/src/speculative.rs:4865",
};

/// `HIPFIRE_RESOURCE_LOCK` — HIPFIRE_RESOURCE_LOCK=0 disables daemon startup resource leases
pub const ENV_HIPFIRE_RESOURCE_LOCK: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_RESOURCE_LOCK",
    description: "HIPFIRE_RESOURCE_LOCK=0 disables daemon startup resource leases",
    source: "crates/hipfire-daemon-adapter/src/lib.rs:1082",
};

/// `HIPFIRE_RESOURCE_LOCK_CPU_CORES` — HIPFIRE_RESOURCE_LOCK_CPU_CORES=0,2-4 adds daemon startup leases for CPU cores
pub const ENV_HIPFIRE_RESOURCE_LOCK_CPU_CORES: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_RESOURCE_LOCK_CPU_CORES",
    description: "HIPFIRE_RESOURCE_LOCK_CPU_CORES=0,2-4 adds daemon startup leases for CPU cores",
    source: "crates/hipfire-daemon-adapter/src/lib.rs:967",
};

/// `HIPFIRE_RESOURCE_LOCK_DIR` — HIPFIRE_RESOURCE_LOCK_DIR overrides the daemon resource-lock root directory
pub const ENV_HIPFIRE_RESOURCE_LOCK_DIR: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_RESOURCE_LOCK_DIR",
    description: "HIPFIRE_RESOURCE_LOCK_DIR overrides the daemon resource-lock root directory",
    source: "crates/hipfire-lock/src/lib.rs:213",
};

/// `HIPFIRE_RESOURCE_LOCK_NPUS` — HIPFIRE_RESOURCE_LOCK_NPUS=1 leases every detected NPU; comma lists lease explicit NPU IDs
pub const ENV_HIPFIRE_RESOURCE_LOCK_NPUS: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_RESOURCE_LOCK_NPUS",
    description:
        "HIPFIRE_RESOURCE_LOCK_NPUS=1 leases every detected NPU; comma lists lease explicit NPU IDs",
    source: "crates/hipfire-daemon-adapter/src/lib.rs:932",
};

/// `HIPFIRE_RESOURCE_LOCK_WAIT_MS` — HIPFIRE_RESOURCE_LOCK_WAIT_MS waits for busy daemon resource leases before failing startup
pub const ENV_HIPFIRE_RESOURCE_LOCK_WAIT_MS: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_RESOURCE_LOCK_WAIT_MS",
    description:
        "HIPFIRE_RESOURCE_LOCK_WAIT_MS waits for busy daemon resource leases before failing startup",
    source: "crates/hipfire-daemon-adapter/src/lib.rs:1100",
};

/// `HIPFIRE_RESPONSES_STATE_MAX` — Runtime variable controlling responses state max in hipfire
pub const ENV_HIPFIRE_RESPONSES_STATE_MAX: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_RESPONSES_STATE_MAX",
    description: "Runtime variable controlling responses state max in hipfire",
    source: "crates/hipfire-server/src/routes/responses.rs:713",
};

/// `HIPFIRE_ROCBLAS_ALL_ARCHS` — Runtime variable controlling rocblas all archs in hipfire
pub const ENV_HIPFIRE_ROCBLAS_ALL_ARCHS: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_ROCBLAS_ALL_ARCHS",
    description: "Runtime variable controlling rocblas all archs in hipfire",
    source: "crates/hipfire-rdna/src/feature_flags.rs:303",
};

/// `HIPFIRE_ROCBLAS_OFF` — Enabled when set to 1
pub const ENV_HIPFIRE_ROCBLAS_OFF: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_ROCBLAS_OFF",
    description: "Enabled when set to 1",
    source: "crates/hipfire-rdna/src/feature_flags.rs:305",
};

/// `HIPFIRE_ROCPROF_BIN` — Runtime variable controlling rocprof bin in hipfire
pub const ENV_HIPFIRE_ROCPROF_BIN: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_ROCPROF_BIN",
    description: "Runtime variable controlling rocprof bin in hipfire",
    source: "crates/hipfire-eval/src/rocprof.rs:275",
};

/// `HIPFIRE_ROCPROF_CSV` — Runtime variable controlling rocprof csv in hipfire
pub const ENV_HIPFIRE_ROCPROF_CSV: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_ROCPROF_CSV",
    description: "Runtime variable controlling rocprof csv in hipfire",
    source: "crates/hipfire-runtime/examples/bench_qwen35_speed.rs:345",
};

/// `HIPFIRE_ROPE_INTERLEAVED_LEGACY` — Runtime variable controlling rope interleaved legacy in hipfire
pub const ENV_HIPFIRE_ROPE_INTERLEAVED_LEGACY: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_ROPE_INTERLEAVED_LEGACY",
    description: "Runtime variable controlling rope interleaved legacy in hipfire",
    source: "crates/hipfire-rdna/src/feature_flags.rs:296",
};

/// `HIPFIRE_RQ2_BULK_BITS` — Runtime variable controlling rq2 bulk bits in hipfire
pub const ENV_HIPFIRE_RQ2_BULK_BITS: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_RQ2_BULK_BITS",
    description: "Runtime variable controlling rq2 bulk bits in hipfire",
    source: "crates/hipfire-quantize/src/main.rs:9043",
};

/// `HIPFIRE_RQ2_DAMP` — De-risk B: single shared, foldable residual-stream rotation. With
pub const ENV_HIPFIRE_RQ2_DAMP: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_RQ2_DAMP",
    description: "De-risk B: single shared, foldable residual-stream rotation. With",
    source: "crates/hipfire-quantize/src/main.rs:9047",
};

/// `HIPFIRE_RQ2_PROTECT_FRAC` — Runtime variable controlling rq2 protect frac in hipfire
pub const ENV_HIPFIRE_RQ2_PROTECT_FRAC: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_RQ2_PROTECT_FRAC",
    description: "Runtime variable controlling rq2 protect frac in hipfire",
    source: "crates/hipfire-quantize/src/main.rs:9039",
};

/// `HIPFIRE_RQ2_Q8_EMBED` — Q8 (~20% of params on a tied-embedding 0.8B). With HIPFIRE_RQ2_Q8_EMBED=1,
pub const ENV_HIPFIRE_RQ2_Q8_EMBED: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_RQ2_Q8_EMBED",
    description: "Q8 (~20% of params on a tied-embedding 0.8B). With HIPFIRE_RQ2_Q8_EMBED=1,",
    source: "crates/hipfire-quantize/src/main.rs:9165",
};

/// `HIPFIRE_RQ2_SHARE_RESID` — HIPFIRE_RQ2_SHARE_RESID=1, every k==1024 weight (the d_model residual
pub const ENV_HIPFIRE_RQ2_SHARE_RESID: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_RQ2_SHARE_RESID",
    description: "HIPFIRE_RQ2_SHARE_RESID=1, every k==1024 weight (the d_model residual",
    source: "crates/hipfire-quantize/src/main.rs:9061",
};

/// `HIPFIRE_RQ3_BULK_BITS` — Runtime variable controlling rq3 bulk bits in hipfire
pub const ENV_HIPFIRE_RQ3_BULK_BITS: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_RQ3_BULK_BITS",
    description: "Runtime variable controlling rq3 bulk bits in hipfire",
    source: "crates/hipfire-quantize/src/main.rs:9202",
};

/// `HIPFIRE_RQ3_PROTECT_FRAC` — Runtime variable controlling rq3 protect frac in hipfire
pub const ENV_HIPFIRE_RQ3_PROTECT_FRAC: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_RQ3_PROTECT_FRAC",
    description: "Runtime variable controlling rq3 protect frac in hipfire",
    source: "crates/hipfire-quantize/src/main.rs:9198",
};

/// `HIPFIRE_RQ3_Q8_EMBED` — Iso-bit embed for an honest mq4 comparison (same as roughquant2 de-risk A)
pub const ENV_HIPFIRE_RQ3_Q8_EMBED: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_RQ3_Q8_EMBED",
    description: "Iso-bit embed for an honest mq4 comparison (same as roughquant2 de-risk A)",
    source: "crates/hipfire-quantize/src/main.rs:9264",
};

/// `HIPFIRE_RQ4_BULK` — Bulk codec: "mq4" → real mq4 format (fair mq4+protect-vs-mq4 test, set
pub const ENV_HIPFIRE_RQ4_BULK: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_RQ4_BULK",
    description: "Bulk codec: \"mq4\" → real mq4 format (fair mq4+protect-vs-mq4 test, set",
    source: "crates/hipfire-quantize/src/main.rs:9305",
};

/// `HIPFIRE_RQ4_BULK_BITS` — Bulk codec: "mq4" → real mq4 format (fair mq4+protect-vs-mq4 test, set
pub const ENV_HIPFIRE_RQ4_BULK_BITS: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_RQ4_BULK_BITS",
    description: "Bulk codec: \"mq4\" → real mq4 format (fair mq4+protect-vs-mq4 test, set",
    source: "crates/hipfire-quantize/src/main.rs:9299",
};

/// `HIPFIRE_RQ4_DUMP_RANK` — HIPFIRE_RQ4_DUMP_RANK=1: print the residual-channel saliency ranking
pub const ENV_HIPFIRE_RQ4_DUMP_RANK: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_RQ4_DUMP_RANK",
    description: "HIPFIRE_RQ4_DUMP_RANK=1: print the residual-channel saliency ranking",
    source: "crates/hipfire-quantize/src/main.rs:9411",
};

/// `HIPFIRE_RQ4_INVERT` — HIPFIRE_RQ4_INVERT=1: protect the LOWEST-saliency channels instead of the
pub const ENV_HIPFIRE_RQ4_INVERT: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_RQ4_INVERT",
    description: "HIPFIRE_RQ4_INVERT=1: protect the LOWEST-saliency channels instead of the",
    source: "crates/hipfire-quantize/src/main.rs:9429",
};

/// `HIPFIRE_RQ4_MQ_BITS` — Uniform bulk bit-width for the mq bulk (4=mq4, 5, 6=mq6). protect_frac=0
pub const ENV_HIPFIRE_RQ4_MQ_BITS: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_RQ4_MQ_BITS",
    description: "Uniform bulk bit-width for the mq bulk (4=mq4, 5, 6=mq6). protect_frac=0",
    source: "crates/hipfire-quantize/src/main.rs:9315",
};

/// `HIPFIRE_RQ4_OBS_DAMP` — Uniform bulk bit-width for the mq bulk (4=mq4, 5, 6=mq6). protect_frac=0
pub const ENV_HIPFIRE_RQ4_OBS_DAMP: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_RQ4_OBS_DAMP",
    description: "Uniform bulk bit-width for the mq bulk (4=mq4, 5, 6=mq6). protect_frac=0",
    source: "crates/hipfire-quantize/src/main.rs:9309",
};

/// `HIPFIRE_RQ4_PROTECT_FRAC` — diag(H) residual-channel energy from true residual readers
pub const ENV_HIPFIRE_RQ4_PROTECT_FRAC: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_RQ4_PROTECT_FRAC",
    description: "diag(H) residual-channel energy from true residual readers",
    source: "crates/hipfire-quantize/src/main.rs:9885",
};

/// `HIPFIRE_RQ4_PROTECT_Q8` — on diag(H) alone). diag = E[x²] (activation energy); wnorm = ‖W[:,c]‖²
pub const ENV_HIPFIRE_RQ4_PROTECT_Q8: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_RQ4_PROTECT_Q8",
    description: "on diag(H) alone). diag = E[x²] (activation energy); wnorm = ‖W[:,c]‖²",
    source: "crates/hipfire-quantize/src/main.rs:9320",
};

/// `HIPFIRE_RQ4_Q8_EMBED` — Enabled when set to 1
pub const ENV_HIPFIRE_RQ4_Q8_EMBED: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_RQ4_Q8_EMBED",
    description: "Enabled when set to 1",
    source: "crates/hipfire-quantize/src/main.rs:9671",
};

/// `HIPFIRE_RQ4_RANDOM_SEED` — Runtime variable controlling rQ4 random seed in hipfire
pub const ENV_HIPFIRE_RQ4_RANDOM_SEED: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_RQ4_RANDOM_SEED",
    description: "Runtime variable controlling rQ4 random seed in hipfire",
    source: "crates/hipfire-quantize/src/main.rs:9534",
};

/// `HIPFIRE_RQ4_SALIENCY` — (weight energy); product = ‖W[:,c]‖²·E[x²] (output-error contribution)
pub const ENV_HIPFIRE_RQ4_SALIENCY: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_RQ4_SALIENCY",
    description: "(weight energy); product = ‖W[:,c]‖²·E[x²] (output-error contribution)",
    source: "crates/hipfire-quantize/src/main.rs:9324",
};

/// `HIPFIRE_RQ4_VOID_ONLY` — Runtime variable controlling rQ4 void only in hipfire
pub const ENV_HIPFIRE_RQ4_VOID_ONLY: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_RQ4_VOID_ONLY",
    description: "Runtime variable controlling rQ4 void only in hipfire",
    source: "crates/hipfire-quantize/src/main.rs:9453",
};

/// `HIPFIRE_RQ_BULK_BITS` — Runtime variable controlling rq bulk bits in hipfire
pub const ENV_HIPFIRE_RQ_BULK_BITS: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_RQ_BULK_BITS",
    description: "Runtime variable controlling rq bulk bits in hipfire",
    source: "crates/hipfire-quantize/src/main.rs:8972",
};

/// `HIPFIRE_RQ_GROUP` — Runtime variable controlling rq group in hipfire
pub const ENV_HIPFIRE_RQ_GROUP: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_RQ_GROUP",
    description: "Runtime variable controlling rq group in hipfire",
    source: "crates/hipfire-quantize/src/main.rs:8976",
};

/// `HIPFIRE_RQ_HAND` — Environment toggle value controls runtime behavior
pub const ENV_HIPFIRE_RQ_HAND: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_RQ_HAND",
    description: "Environment toggle value controls runtime behavior",
    source: "crates/hipfire-arch-qwen35/src/qwen35.rs:23328",
};

/// `HIPFIRE_RQ_PROTECT_FRAC` — Runtime variable controlling rq protect frac in hipfire
pub const ENV_HIPFIRE_RQ_PROTECT_FRAC: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_RQ_PROTECT_FRAC",
    description: "Runtime variable controlling rq protect frac in hipfire",
    source: "crates/hipfire-quantize/src/main.rs:8968",
};

/// `HIPFIRE_RUN_EXAMPLE_BIN` — Runtime variable controlling run example bin in hipfire
pub const ENV_HIPFIRE_RUN_EXAMPLE_BIN: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_RUN_EXAMPLE_BIN",
    description: "Runtime variable controlling run example bin in hipfire",
    source: "crates/hipfire-eval/src/lib.rs:1194",
};

/// `HIPFIRE_SAMPLE_COMPARE` — Enabled when set to 1
pub const ENV_HIPFIRE_SAMPLE_COMPARE: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_SAMPLE_COMPARE",
    description: "Enabled when set to 1",
    source: "crates/hipfire-runtime/examples/infer_qwen35.rs:254",
};

/// `HIPFIRE_SCORE_TAIL` — HIPFIRE_SCORE_TAIL=N → score only the last N query positions (leak-free for
pub const ENV_HIPFIRE_SCORE_TAIL: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_SCORE_TAIL",
    description: "HIPFIRE_SCORE_TAIL=N → score only the last N query positions (leak-free for",
    source: "crates/hipfire-train/examples/recovery_generalization_supra50m.rs:163",
};

/// `HIPFIRE_SERVER_RESIDENT_STATE_BUDGET_MB` — Runtime variable controlling server resident state budget mb in hipfire
pub const ENV_HIPFIRE_SERVER_RESIDENT_STATE_BUDGET_MB: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_SERVER_RESIDENT_STATE_BUDGET_MB",
    description: "Runtime variable controlling server resident state budget mb in hipfire",
    source: "crates/hipfire-serving-core/src/session.rs:1446",
};

/// `HIPFIRE_SMOKE_KV` — Select KV cache quant via HIPFIRE_SMOKE_KV (default q8, matches the
pub const ENV_HIPFIRE_SMOKE_KV: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_SMOKE_KV",
    description: "Select KV cache quant via HIPFIRE_SMOKE_KV (default q8, matches the",
    source: "crates/hipfire-runtime/examples/a3b_smoke_forward.rs:130",
};

/// `HIPFIRE_SMOKE_KV_SEQ` — production CLI default). asym3/asym4 engage the Givens-rotated 3/4-bit
pub const ENV_HIPFIRE_SMOKE_KV_SEQ: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_SMOKE_KV_SEQ",
    description: "production CLI default). asym3/asym4 engage the Givens-rotated 3/4-bit",
    source: "crates/hipfire-runtime/examples/a3b_smoke_forward.rs:123",
};

/// `HIPFIRE_SMOKE_MODE` — raw (default for back-compat): tokenize "Hello" and decode from pos=0
pub const ENV_HIPFIRE_SMOKE_MODE: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_SMOKE_MODE",
    description: "raw (default for back-compat): tokenize \"Hello\" and decode from pos=0",
    source: "crates/hipfire-runtime/examples/a3b_smoke_forward.rs:178",
};

/// `HIPFIRE_SMOKE_PROMPT` — Defaults to Hello when unset
pub const ENV_HIPFIRE_SMOKE_PROMPT: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_SMOKE_PROMPT",
    description: "Defaults to Hello when unset",
    source: "crates/hipfire-runtime/examples/a3b_smoke_forward.rs:179",
};

/// `HIPFIRE_SMOKE_STEPS` — Runtime variable controlling smoke steps in hipfire
pub const ENV_HIPFIRE_SMOKE_STEPS: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_SMOKE_STEPS",
    description: "Runtime variable controlling smoke steps in hipfire",
    source: "crates/hipfire-runtime/examples/a3b_smoke_forward.rs:59",
};

/// `HIPFIRE_SPEC_PHASES` — Enabled when set to 1
pub const ENV_HIPFIRE_SPEC_PHASES: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_SPEC_PHASES",
    description: "Enabled when set to 1",
    source: "crates/hipfire-arch-qwen35/src/speculative.rs:6738",
};

/// `HIPFIRE_STATE` — Interprets "HIPFIRE_STATE" from environment to select behavior
pub const ENV_HIPFIRE_STATE: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_STATE",
    description: "Interprets \"HIPFIRE_STATE\" from environment to select behavior",
    source: "crates/hipfire-runtime/examples/greedy_dump_top5.rs:266",
};

/// `HIPFIRE_STEER_DUMP` — refusal scorer actually sees — the qualitative companion to the numeric report
pub const ENV_HIPFIRE_STEER_DUMP: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_STEER_DUMP",
    description: "refusal scorer actually sees — the qualitative companion to the numeric report",
    source: "crates/hipfire-steer/src/driver.rs:227",
};

/// `HIPFIRE_TARGET_ARCH` — Runtime variable controlling target arch in hipfire
pub const ENV_HIPFIRE_TARGET_ARCH: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_TARGET_ARCH",
    description: "Runtime variable controlling target arch in hipfire",
    source: "crates/hipfire-rdna/src/dispatch/mod.rs:722",
};

/// `HIPFIRE_TIER_RATIO` — Runtime variable controlling tier ratio in hipfire
pub const ENV_HIPFIRE_TIER_RATIO: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_TIER_RATIO",
    description: "Runtime variable controlling tier ratio in hipfire",
    source: "crates/hipfire-quantize/src/main.rs:5217",
};

/// `HIPFIRE_TINYQUANT_FAMILIES` — Optional comma-separated family allowlist ("HIPFIRE_TINYQUANT_FAMILIES")
pub const ENV_HIPFIRE_TINYQUANT_FAMILIES: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_TINYQUANT_FAMILIES",
    description: "Optional comma-separated family allowlist (\"HIPFIRE_TINYQUANT_FAMILIES\")",
    source: "crates/hipfire-eval/src/executor_tinyquant.rs:370",
};

/// `HIPFIRE_TINYQUANT_RECORD` — Optional comma-separated family allowlist ("HIPFIRE_TINYQUANT_FAMILIES")
pub const ENV_HIPFIRE_TINYQUANT_RECORD: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_TINYQUANT_RECORD",
    description: "Optional comma-separated family allowlist (\"HIPFIRE_TINYQUANT_FAMILIES\")",
    source: "crates/hipfire-eval/src/executor_tinyquant.rs:364",
};

/// `HIPFIRE_TINY_QUANT_PROBE_BIN` — Runtime variable controlling tiny quant probe bin in hipfire
pub const ENV_HIPFIRE_TINY_QUANT_PROBE_BIN: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_TINY_QUANT_PROBE_BIN",
    description: "Runtime variable controlling tiny quant probe bin in hipfire",
    source: "crates/hipfire-eval/src/executor_tinyquant.rs:142",
};

/// `HIPFIRE_TINY_SD_HFQ` — Runtime variable controlling tiny sd hfq in hipfire
pub const ENV_HIPFIRE_TINY_SD_HFQ: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_TINY_SD_HFQ",
    description: "Runtime variable controlling tiny sd hfq in hipfire",
    source: "crates/hipfire-server/src/routes/sdapi.rs:4478",
};

/// `HIPFIRE_TP_BENCH_ITERS` — Runtime variable controlling tp bench iters in hipfire
pub const ENV_HIPFIRE_TP_BENCH_ITERS: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_TP_BENCH_ITERS",
    description: "Runtime variable controlling tp bench iters in hipfire",
    source: "crates/hip-bridge/examples/rccl_smoke.rs:40",
};

/// `HIPFIRE_TP_BENCH_N` — Runtime variable controlling tp bench n in hipfire
pub const ENV_HIPFIRE_TP_BENCH_N: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_TP_BENCH_N",
    description: "Runtime variable controlling tp bench n in hipfire",
    source: "crates/hip-bridge/examples/rccl_smoke.rs:36",
};

/// `HIPFIRE_TP_EXPERT_ASSIGN` — Selects behavior from recognized values
pub const ENV_HIPFIRE_TP_EXPERT_ASSIGN: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_TP_EXPERT_ASSIGN",
    description: "Selects behavior from recognized values",
    source: "crates/hipfire-runtime/src/tp_shard.rs:57",
};

/// `HIPFIRE_TP_USE_RCCL` — Runtime variable controlling tp use rccl in hipfire
pub const ENV_HIPFIRE_TP_USE_RCCL: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_TP_USE_RCCL",
    description: "Runtime variable controlling tp use rccl in hipfire",
    source: "crates/hipfire-runtime/src/config.rs:90",
};

/// `HIPFIRE_UNIFORM_VRAM_TOLERANCE_GB` — Runtime variable controlling uniform vram tolerance gb in hipfire
pub const ENV_HIPFIRE_UNIFORM_VRAM_TOLERANCE_GB: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_UNIFORM_VRAM_TOLERANCE_GB",
    description: "Runtime variable controlling uniform vram tolerance gb in hipfire",
    source: "crates/hipfire-runtime/src/config.rs:105",
};

/// `HIPFIRE_VERIFY_GRAPH` — Runtime variable controlling verify graph in hipfire
pub const ENV_HIPFIRE_VERIFY_GRAPH: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_VERIFY_GRAPH",
    description: "Runtime variable controlling verify graph in hipfire",
    source: "crates/hipfire-arch-qwen35/src/speculative.rs:6200",
};

/// `HIPFIRE_VERIFY_GRAPH_TIMING` — (HIPFIRE_VERIFY_GRAPH_TIMING=1). Two device-sync points bracket the
pub const ENV_HIPFIRE_VERIFY_GRAPH_TIMING: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_VERIFY_GRAPH_TIMING",
    description: "(HIPFIRE_VERIFY_GRAPH_TIMING=1). Two device-sync points bracket the",
    source: "crates/hipfire-arch-qwen35/src/speculative.rs:6215",
};

/// `HIPFIRE_VERIFY_GRAPH_TREE` — Enabled when set to 1
pub const ENV_HIPFIRE_VERIFY_GRAPH_TREE: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_VERIFY_GRAPH_TREE",
    description: "Enabled when set to 1",
    source: "crates/hipfire-arch-qwen35/src/speculative.rs:6190",
};

/// `HIPFIRE_VISION_CACHE` — Disabled when set to 0
pub const ENV_HIPFIRE_VISION_CACHE: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_VISION_CACHE",
    description: "Disabled when set to 0",
    source: "crates/hipfire-serving-core/src/generate_vl.rs:1195",
};

/// `HIPFIRE_VISION_CACHE_DIR` — Runtime variable controlling vision cache dir in hipfire
pub const ENV_HIPFIRE_VISION_CACHE_DIR: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_VISION_CACHE_DIR",
    description: "Runtime variable controlling vision cache dir in hipfire",
    source: "crates/hipfire-serving-core/src/generate_vl.rs:1198",
};

/// `HIPFIRE_VISION_CACHE_MAX_BYTES` — Parses "HIPFIRE_VISION_CACHE_MAX_BYTES" with fallback defaults
pub const ENV_HIPFIRE_VISION_CACHE_MAX_BYTES: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_VISION_CACHE_MAX_BYTES",
    description: "Parses \"HIPFIRE_VISION_CACHE_MAX_BYTES\" with fallback defaults",
    source: "crates/hipfire-serving-core/src/generate_vl.rs:1207",
};

/// `HIPFIRE_VISION_DUMP` — Selects behavior from recognized values
pub const ENV_HIPFIRE_VISION_DUMP: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_VISION_DUMP",
    description: "Selects behavior from recognized values",
    source: "crates/hipfire-arch-gemma3-vl/src/forward.rs:44",
};

/// `HIPFIRE_VISION_PROFILE` — Optional per-category timing (HIPFIRE_VISION_PROFILE=1): device-sync around
pub const ENV_HIPFIRE_VISION_PROFILE: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_VISION_PROFILE",
    description: "Optional per-category timing (HIPFIRE_VISION_PROFILE=1): device-sync around",
    source: "crates/hipfire-arch-gemma3-vl/src/forward.rs:140",
};

/// `HIPFIRE_VL_DUMP_DIR` — little-endian f32 blobs + JSON sidecars to $HIPFIRE_VL_DUMP_DIR
pub const ENV_HIPFIRE_VL_DUMP_DIR: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_VL_DUMP_DIR",
    description: "little-endian f32 blobs + JSON sidecars to $HIPFIRE_VL_DUMP_DIR",
    source: "crates/hipfire-runtime/examples/infer.rs:171",
};

/// `HIPFIRE_WARN_GENERIC` — Enabled by default; set to 0 to disable
pub const ENV_HIPFIRE_WARN_GENERIC: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_WARN_GENERIC",
    description: "Enabled by default; set to 0 to disable",
    source: "crates/hipfire-rdna/src/generic_warn.rs:53",
};

/// `HIPFIRE_WA_CODEC` — HIPFIRE_WA_CODEC=qtip3 swaps the symmetric-int weight quant for the QTIP3 trellis;
pub const ENV_HIPFIRE_WA_CODEC: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_WA_CODEC",
    description:
        "HIPFIRE_WA_CODEC=qtip3 swaps the symmetric-int weight quant for the QTIP3 trellis;",
    source: "crates/hipfire-train/examples/wa_matrix_sweep.rs:634",
};

/// `HIPFIRE_WA_LEARNED` — Learn dense R1 (h×h) on CALIB residual activations
pub const ENV_HIPFIRE_WA_LEARNED: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_WA_LEARNED",
    description: "Learn dense R1 (h×h) on CALIB residual activations",
    source: "crates/hipfire-train/examples/wa_matrix_sweep.rs:541",
};

/// `HIPFIRE_WA_R1_ITERS` — heavier M is affordable at h=2048. HIPFIRE_WA_R1_ITERS tunes it (default 100)
pub const ENV_HIPFIRE_WA_R1_ITERS: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_WA_R1_ITERS",
    description: "heavier M is affordable at h=2048. HIPFIRE_WA_R1_ITERS tunes it (default 100)",
    source: "crates/hipfire-train/examples/wa_matrix_sweep.rs:560",
};

/// `HIPFIRE_WO_MMQ` — Enabled when set to 1
pub const ENV_HIPFIRE_WO_MMQ: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_WO_MMQ",
    description: "Enabled when set to 1",
    source: "crates/hipfire-rdna/src/feature_flags.rs:219",
};

/// `HIPFIRE_WO_WMMA_VARIANT` — Runtime variable controlling wo wmma variant in hipfire
pub const ENV_HIPFIRE_WO_WMMA_VARIANT: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_WO_WMMA_VARIANT",
    description: "Runtime variable controlling wo wmma variant in hipfire",
    source: "crates/hipfire-rdna/src/feature_flags.rs:300",
};

/// `HIPFIRE_XDNA1_LIB` — Runtime variable controlling xdna1 lib in hipfire
pub const ENV_HIPFIRE_XDNA1_LIB: EnvVarDoc = EnvVarDoc {
    name: "HIPFIRE_XDNA1_LIB",
    description: "Runtime variable controlling xdna1 lib in hipfire",
    source: "crates/hipfire-npu/src/lib.rs:69",
};

/// `HIP_PATH` — fails with "file not found". Add well-known candidates as -I flags;
pub const ENV_HIP_PATH: EnvVarDoc = EnvVarDoc {
    name: "HIP_PATH",
    description: "fails with \"file not found\". Add well-known candidates as -I flags;",
    source: "crates/hipfire-rdna/src/compiler.rs:518",
};

/// `HIP_VISIBLE_DEVICES` — Used to configure runtime execution by explicitly setting "HIP_VISIBLE_DEVICES"
pub const ENV_HIP_VISIBLE_DEVICES: EnvVarDoc = EnvVarDoc {
    name: "HIP_VISIBLE_DEVICES",
    description:
        "Used to configure runtime execution by explicitly setting \"HIP_VISIBLE_DEVICES\"",
    source: "crates/hipfire-daemon-adapter/src/lib.rs:1511",
};

/// `HOME` — Runtime variable controlling home in hipfire
pub const ENV_HOME: EnvVarDoc = EnvVarDoc {
    name: "HOME",
    description: "Runtime variable controlling home in hipfire",
    source: "crates/hipfire-tui/src/hipfire/mod.rs:40",
};

/// `HOSTNAME` — Runtime variable controlling hostname in hipfire
pub const ENV_HOSTNAME: EnvVarDoc = EnvVarDoc {
    name: "HOSTNAME",
    description: "Runtime variable controlling hostname in hipfire",
    source: "crates/hipfire-tui/src/hipfire/status.rs:130",
};

/// `HUGGINGFACE_HUB_CACHE` — Runtime variable controlling huggingface hub cache in hipfire
pub const ENV_HUGGINGFACE_HUB_CACHE: EnvVarDoc = EnvVarDoc {
    name: "HUGGINGFACE_HUB_CACHE",
    description: "Runtime variable controlling huggingface hub cache in hipfire",
    source: "crates/hipfire-server/src/routes/sdapi.rs:4306",
};

/// `MAX_TOKENS` — Parses "MAX_TOKENS" with fallback defaults
pub const ENV_MAX_TOKENS: EnvVarDoc = EnvVarDoc {
    name: "MAX_TOKENS",
    description: "Parses \"MAX_TOKENS\" with fallback defaults",
    source: "crates/hipfire-runtime/examples/greedy_dump.rs:133",
};

/// `MMQ_TEST_MODE` — Defaults to residual when unset
pub const ENV_MMQ_TEST_MODE: EnvVarDoc = EnvVarDoc {
    name: "MMQ_TEST_MODE",
    description: "Defaults to residual when unset",
    source: "crates/hipfire-rdna/examples/test_gfx906_mmq_correctness.rs:102",
};

/// `NANO30B_DIR` — Runtime variable controlling nano30b dir in hipfire
pub const ENV_NANO30B_DIR: EnvVarDoc = EnvVarDoc {
    name: "NANO30B_DIR",
    description: "Runtime variable controlling nano30b dir in hipfire",
    source: "crates/hipfire-arch-nemotron/examples/test_load_nano30b_hfq.rs:69",
};

/// `NANO4B_DIR` — Runtime variable controlling nano4b dir in hipfire
pub const ENV_NANO4B_DIR: EnvVarDoc = EnvVarDoc {
    name: "NANO4B_DIR",
    description: "Runtime variable controlling nano4b dir in hipfire",
    source: "crates/hipfire-arch-nemotron/examples/test_model_prefill_hfq_gpu.rs:62",
};

/// `NEMO_TOKENS` — Selects behavior from recognized values
pub const ENV_NEMO_TOKENS: EnvVarDoc = EnvVarDoc {
    name: "NEMO_TOKENS",
    description: "Selects behavior from recognized values",
    source: "crates/hipfire-arch-nemotron/examples/test_load_nano30b_hfq.rs:45",
};

/// `NO_COLOR` — Runtime variable controlling no color in hipfire
pub const ENV_NO_COLOR: EnvVarDoc = EnvVarDoc {
    name: "NO_COLOR",
    description: "Runtime variable controlling no color in hipfire",
    source: "crates/hipfire-system-monitor/src/lib.rs:315",
};

/// `NO_NGRAM` — Disabled for perf measurement — re-enable after implementing GPU n-gram kernel
pub const ENV_NO_NGRAM: EnvVarDoc = EnvVarDoc {
    name: "NO_NGRAM",
    description: "Disabled for perf measurement — re-enable after implementing GPU n-gram kernel",
    source: "crates/hipfire-runtime/examples/infer_vl.rs:373",
};

/// `PATH` — Runtime variable controlling path in hipfire
pub const ENV_PATH: EnvVarDoc = EnvVarDoc {
    name: "PATH",
    description: "Runtime variable controlling path in hipfire",
    source: "crates/hipfire-rdna/src/compiler.rs:204",
};

/// `PROMPT_MODE` — Defaults to thinking when unset
pub const ENV_PROMPT_MODE: EnvVarDoc = EnvVarDoc {
    name: "PROMPT_MODE",
    description: "Defaults to thinking when unset",
    source: "crates/hipfire-runtime/examples/greedy_dump_top5.rs:112",
};

/// `QWEN35_TEST_MODEL` — Runtime variable controlling qwen35 test model in hipfire
pub const ENV_QWEN35_TEST_MODEL: EnvVarDoc = EnvVarDoc {
    name: "QWEN35_TEST_MODEL",
    description: "Runtime variable controlling qwen35 test model in hipfire",
    source: "crates/hipfire-runtime/examples/test_qwen35_loadQA.rs:40",
};

/// `ROCM_DEVICE_LIB_PATH` — Runtime variable controlling rocm device lib path in hipfire
pub const ENV_ROCM_DEVICE_LIB_PATH: EnvVarDoc = EnvVarDoc {
    name: "ROCM_DEVICE_LIB_PATH",
    description: "Runtime variable controlling rocm device lib path in hipfire",
    source: "crates/hipfire-rdna/src/compiler.rs:506",
};

/// `ROCM_PATH` — Runtime variable controlling rocm path in hipfire
pub const ENV_ROCM_PATH: EnvVarDoc = EnvVarDoc {
    name: "ROCM_PATH",
    description: "Runtime variable controlling rocm path in hipfire",
    source: "crates/hipfire-rdna/src/compiler.rs:503",
};

/// `ROCR_VISIBLE_DEVICES` — Runtime variable controlling rocr visible devices in hipfire
pub const ENV_ROCR_VISIBLE_DEVICES: EnvVarDoc = EnvVarDoc {
    name: "ROCR_VISIBLE_DEVICES",
    description: "Runtime variable controlling rocr visible devices in hipfire",
    source: "crates/hipfire-daemon-adapter/src/lib.rs:891",
};

/// `TERM` — Runtime variable controlling term in hipfire
pub const ENV_TERM: EnvVarDoc = EnvVarDoc {
    name: "TERM",
    description: "Runtime variable controlling term in hipfire",
    source: "crates/hipfire-system-monitor/src/lib.rs:317",
};

/// `TRIALS` — Runtime variable controlling trials in hipfire
pub const ENV_TRIALS: EnvVarDoc = EnvVarDoc {
    name: "TRIALS",
    description: "Runtime variable controlling trials in hipfire",
    source: "crates/hipfire-rdna/examples/bench_gfx1151_hfq4_s4_mmq.rs:188",
};

/// `USERPROFILE` — Runtime variable controlling userprofile in hipfire
pub const ENV_USERPROFILE: EnvVarDoc = EnvVarDoc {
    name: "USERPROFILE",
    description: "Runtime variable controlling userprofile in hipfire",
    source: "crates/hipfire-rdna/src/compiler.rs:228",
};

/// `USE_SAMPLE` — Enabled when set to 1
pub const ENV_USE_SAMPLE: EnvVarDoc = EnvVarDoc {
    name: "USE_SAMPLE",
    description: "Enabled when set to 1",
    source: "crates/hipfire-runtime/examples/a3b_multiturn_oneshot.rs:138",
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
    ENV_CAP_POS,
    ENV_CARGO_FEATURE_NPU_KERNELS,
    ENV_CARGO_MANIFEST_DIR,
    ENV_CARGO_PKG_VERSION,
    ENV_COLORTERM,
    ENV_DDTREE_TIMING,
    ENV_DEBUG_LAYERS,
    ENV_DFLASH_LIVE_TAU,
    ENV_FP32_STATE,
    ENV_GPU_LOCK_TIMEOUT,
    ENV_GPU_POLL_INTERVAL,
    ENV_HFHS_REAL,
    ENV_HFQ_TEST_N_ITER,
    ENV_HFQ_TEST_SCALE_LOG10,
    ENV_HFQ_TEST_ZP_MAX,
    ENV_HF_HOME,
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
    ENV_HIPFIRE_BATCHES_STATE_MAX,
    ENV_HIPFIRE_BENCH_QWEN35_SPEED_BIN,
    ENV_HIPFIRE_BF16_DENSE_M128,
    ENV_HIPFIRE_BF16_MOE_M256,
    ENV_HIPFIRE_BF16_WEIGHTS,
    ENV_HIPFIRE_BLOB_FORCE,
    ENV_HIPFIRE_CALIB_F64_AUDIT,
    ENV_HIPFIRE_CALIB_HESSIAN_STORAGE,
    ENV_HIPFIRE_CALIB_PROFILE,
    ENV_HIPFIRE_CASK_OFF,
    ENV_HIPFIRE_CHATML,
    ENV_HIPFIRE_CHAT_TEMPLATE_FILE,
    ENV_HIPFIRE_COLLECT_ARTIFACTS_BIN,
    ENV_HIPFIRE_COMP_DUMP,
    ENV_HIPFIRE_CONV1D_TREE_GFX1151,
    ENV_HIPFIRE_DAEMON_BIN,
    ENV_HIPFIRE_DAEMON_RESIDENT_STATE_BUDGET_MB,
    ENV_HIPFIRE_DDTREE_BUDGET,
    ENV_HIPFIRE_DDTREE_FORCE_SLOW,
    ENV_HIPFIRE_DDTREE_LOGW_CUTOFF,
    ENV_HIPFIRE_DDTREE_PATH_B_CAPTURE,
    ENV_HIPFIRE_DDTREE_PATH_C,
    ENV_HIPFIRE_DDTREE_PATH_C_VERBOSE,
    ENV_HIPFIRE_DDTREE_PATH_C_VERIFY_GRAPH,
    ENV_HIPFIRE_DDTREE_TAPE_DUMP,
    ENV_HIPFIRE_DDTREE_TOPK,
    ENV_HIPFIRE_DDTREE_TREE_LA,
    ENV_HIPFIRE_DEBUG_CHAT,
    ENV_HIPFIRE_DEBUG_PREFILL_ELIGIBLE,
    ENV_HIPFIRE_DEBUG_PREFIX_BOUNDARIES,
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
    ENV_HIPFIRE_DEEPSEEK4_DSA_WMMA,
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
    ENV_HIPFIRE_DFLASH_ROLLBACK_FA_RAW_ATOL,
    ENV_HIPFIRE_DFLASH_ROLLBACK_LOGIT_COMPARE_STEPS,
    ENV_HIPFIRE_DFLASH_ROLLBACK_PREFIX_VERIFY,
    ENV_HIPFIRE_DFLASH_ROLLBACK_SERIAL_REPLAY,
    ENV_HIPFIRE_DFLASH_ROLLBACK_SERIAL_TAPE,
    ENV_HIPFIRE_DFLASH_ROLLBACK_VERIFY_FRAMES,
    ENV_HIPFIRE_DFLASH_ROLLBACK_X_IN_ATOL,
    ENV_HIPFIRE_DFLASH_SERIAL_QKVZA_SELF_COMPARE,
    ENV_HIPFIRE_DFLASH_SERIAL_TAPE_X_IN_COMPARE,
    ENV_HIPFIRE_DFLASH_SPEC_DEMO_BIN,
    ENV_HIPFIRE_DFLASH_TRACE_EXPECTED_TOKEN,
    ENV_HIPFIRE_DFLASH_TRACE_POSITION,
    ENV_HIPFIRE_DFLASH_TRACE_TOKEN_INDEX,
    ENV_HIPFIRE_DIAG,
    ENV_HIPFIRE_DIFFUSION_CPU_REFERENCE,
    ENV_HIPFIRE_DIFFUSION_DUMP_DIR,
    ENV_HIPFIRE_DIFFUSION_LAYER_RUNG,
    ENV_HIPFIRE_DIR,
    ENV_HIPFIRE_DN_STATE_EF,
    ENV_HIPFIRE_DN_STATE_FP32_BELOW,
    ENV_HIPFIRE_DOT2_GEMV,
    ENV_HIPFIRE_DOTS_OCR_BF16_RESIDUAL,
    ENV_HIPFIRE_DOTS_OCR_DUMP_DIR,
    ENV_HIPFIRE_DOTS_OCR_TRACE,
    ENV_HIPFIRE_DPM_WARMUP_SECS,
    ENV_HIPFIRE_DRAFT_F16,
    ENV_HIPFIRE_DRAFT_GEMM_DUMP,
    ENV_HIPFIRE_DRAFT_SUBPHASE,
    ENV_HIPFIRE_DSPARK,
    ENV_HIPFIRE_DSPARK_ADAPTIVE_BLOCK,
    ENV_HIPFIRE_DSPARK_HFQ4_WMMA,
    ENV_HIPFIRE_DSPARK_PROFILE,
    ENV_HIPFIRE_DSPARK_Q8_4W,
    ENV_HIPFIRE_DSPARK_Q8_WMMA,
    ENV_HIPFIRE_DTOH_DUMP,
    ENV_HIPFIRE_DUMMY_GENERATE_DELAY_MS,
    ENV_HIPFIRE_DUMMY_PREFILL_DELAY_MS,
    ENV_HIPFIRE_DUMP_COND,
    ENV_HIPFIRE_DUMP_HIDDEN,
    ENV_HIPFIRE_DUMP_HIDDEN_ALL,
    ENV_HIPFIRE_DUMP_HIDDEN_ALLLAYERS,
    ENV_HIPFIRE_DUMP_HIDDEN_LAYER,
    ENV_HIPFIRE_DUMP_HIDDEN_POS,
    ENV_HIPFIRE_DUMP_LATENT,
    ENV_HIPFIRE_DUMP_REQUEST,
    ENV_HIPFIRE_EMIT_TOKEN_IDS,
    ENV_HIPFIRE_EP_DECODE_TIMING,
    ENV_HIPFIRE_EP_DUMP_IDX,
    ENV_HIPFIRE_EP_DUMP_POS,
    ENV_HIPFIRE_EP_KV_SEQ,
    ENV_HIPFIRE_EP_PEER_ALLREDUCE,
    ENV_HIPFIRE_EP_PEER_ALLREDUCE_DECODE,
    ENV_HIPFIRE_EP_PREFILL,
    ENV_HIPFIRE_EP_PREFILL_TIMING,
    ENV_HIPFIRE_EP_SKIP_ALLREDUCE,
    ENV_HIPFIRE_EVAL_DATASET_MIRROR,
    ENV_HIPFIRE_EVAL_EVIDENCE_DIR,
    ENV_HIPFIRE_EVAL_KLDREF,
    ENV_HIPFIRE_EVAL_PERPLEXITY_CORPUS,
    ENV_HIPFIRE_EVAL_PERPLEXITY_CTX,
    ENV_HIPFIRE_EXPERIMENTAL_BUDGET_ALERT,
    ENV_HIPFIRE_FILES_STATE_MAX,
    ENV_HIPFIRE_FLASH_PARTIALS_BATCH,
    ENV_HIPFIRE_FORCE_GENERIC,
    ENV_HIPFIRE_FORCE_UNFUSED,
    ENV_HIPFIRE_FORWARD_LOWERED,
    ENV_HIPFIRE_FP16,
    ENV_HIPFIRE_FP8_WMMA,
    ENV_HIPFIRE_FUSED_HFQ4_2ROW_GFX1151,
    ENV_HIPFIRE_GATE_UP_VARIANT,
    ENV_HIPFIRE_GDN_Q8_REG_GFX1151,
    ENV_HIPFIRE_GEMMA3_CALIB_IMATRIX_ONLY,
    ENV_HIPFIRE_GEMMA3_CALIB_LAYERS_PER_PASS,
    ENV_HIPFIRE_GEMMA3_CALIB_MICROBATCH,
    ENV_HIPFIRE_GEMMA3_CALIB_NO_BATCH,
    ENV_HIPFIRE_GEMMA3_NO_BATCHED_PREFILL,
    ENV_HIPFIRE_GEMMA3_PREFILL_MICROBATCH,
    ENV_HIPFIRE_GEMMA3_VISION_NOWMMA,
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
    ENV_HIPFIRE_HFQ4_GATE_UP_FAST,
    ENV_HIPFIRE_HFQ4_MMQ_GFX906_Y64,
    ENV_HIPFIRE_HFQ4_QKVZA_FAST,
    ENV_HIPFIRE_HFQ4_QKV_FAST,
    ENV_HIPFIRE_HFQ4_RESIDUAL_FAST,
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
    ENV_HIPFIRE_KLD_SCORING_MODE,
    ENV_HIPFIRE_KLD_TOP_K,
    ENV_HIPFIRE_KVARN_ROTATE,
    ENV_HIPFIRE_KVARN_SIM,
    ENV_HIPFIRE_KVNOISE,
    ENV_HIPFIRE_KVNOISE_BITS,
    ENV_HIPFIRE_KVNOISE_FOLD,
    ENV_HIPFIRE_KVNOISE_HOT,
    ENV_HIPFIRE_KVNOISE_LR,
    ENV_HIPFIRE_KV_COLD_BITS,
    ENV_HIPFIRE_KV_COLD_V_BITS,
    ENV_HIPFIRE_KV_COLD_V_PERSLOT,
    ENV_HIPFIRE_KV_CORE_FRAC,
    ENV_HIPFIRE_KV_FOLD_M,
    ENV_HIPFIRE_KV_HIERARCHICAL,
    ENV_HIPFIRE_KV_HIER_KEEP,
    ENV_HIPFIRE_KV_HIER_TEST_IDLE,
    ENV_HIPFIRE_KV_HOT_BUDGET,
    ENV_HIPFIRE_KV_IDLE_KEEP,
    ENV_HIPFIRE_KV_IMPORTANCE,
    ENV_HIPFIRE_KV_MIGRATE_BATCH,
    ENV_HIPFIRE_KV_MODE,
    ENV_HIPFIRE_KV_PHYSICAL_CAP,
    ENV_HIPFIRE_KV_POS_LOCAL,
    ENV_HIPFIRE_LFM2_CAPTURE_POSTMIXER,
    ENV_HIPFIRE_LFM2_DFLASH,
    ENV_HIPFIRE_LFM2_DFLASH_F16,
    ENV_HIPFIRE_LFM2_DFLASH_SYNC_GEMM,
    ENV_HIPFIRE_LFM2_GRAPH,
    ENV_HIPFIRE_LLOYD_FORCE_BASELINE,
    ENV_HIPFIRE_LLOYD_GFX12,
    ENV_HIPFIRE_LLOYD_K3,
    ENV_HIPFIRE_LLOYD_MB4,
    ENV_HIPFIRE_LM_DUMP,
    ENV_HIPFIRE_LM_HEAD_F16,
    ENV_HIPFIRE_LM_HEAD_OVERWRITE,
    ENV_HIPFIRE_LM_HEAD_WMMA,
    ENV_HIPFIRE_LOAD_TRANSPORT,
    ENV_HIPFIRE_LOCAL,
    ENV_HIPFIRE_LOWRANK_R,
    ENV_HIPFIRE_MAX_GEN,
    ENV_HIPFIRE_MAX_SEQ_ALLOW_OVERRIDE,
    ENV_HIPFIRE_MEMCPY_DUMP,
    ENV_HIPFIRE_MEMSET_DUMP,
    ENV_HIPFIRE_MINIMAX_CAPTURE_POSTATTN,
    ENV_HIPFIRE_MINIMAX_DOWN_FORMAT,
    ENV_HIPFIRE_MINIMAX_ENABLE_DOWN_AWQ,
    ENV_HIPFIRE_MINIMAX_EXPERT_MQ2L,
    ENV_HIPFIRE_MINIMAX_EXPERT_MQ3L,
    ENV_HIPFIRE_MINIMAX_EXPERT_MQ6,
    ENV_HIPFIRE_MMQ,
    ENV_HIPFIRE_MMQ_DIAG_QUANTIZE_ONLY,
    ENV_HIPFIRE_MMQ_SCREEN,
    ENV_HIPFIRE_MMQ_SCREEN_THRESHOLD,
    ENV_HIPFIRE_MODELS_DIR,
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
    ENV_HIPFIRE_MOE_MQ2L_N32_GFX1151,
    ENV_HIPFIRE_MOE_PARO_I8,
    ENV_HIPFIRE_MOE_PARO_I8_K8,
    ENV_HIPFIRE_MQ3_MB4,
    ENV_HIPFIRE_MTP_DEVICE_TOKEN_CHAIN,
    ENV_HIPFIRE_MTP_GPU_ACCEPT,
    ENV_HIPFIRE_MTP_HEAD_LMHEAD_WMMA,
    ENV_HIPFIRE_MTP_K,
    ENV_HIPFIRE_MTP_MODE,
    ENV_HIPFIRE_MTP_PHASE_TIMERS,
    ENV_HIPFIRE_MTP_PROPOSAL_GRAPH,
    ENV_HIPFIRE_MTP_P_MIN,
    ENV_HIPFIRE_MTP_Q8_VERIFY_WMMA,
    ENV_HIPFIRE_MTP_SMOKE_HEAD,
    ENV_HIPFIRE_MTP_SMOKE_TRUNK,
    ENV_HIPFIRE_MTP_SNAPSHOT_OVERLAP,
    ENV_HIPFIRE_MTP_VERIFY_DECOUPLE,
    ENV_HIPFIRE_MW16,
    ENV_HIPFIRE_NEMOTRON_SSM_STATE,
    ENV_HIPFIRE_NGRAM_LOOP_THRESHOLD,
    ENV_HIPFIRE_NGRAM_WINDOW,
    ENV_HIPFIRE_NORMALIZE_PROMPT,
    ENV_HIPFIRE_NORM_PLUS1,
    ENV_HIPFIRE_NO_EXPERT_AWQ,
    ENV_HIPFIRE_NO_QUANT,
    ENV_HIPFIRE_NPU_ATTN_GATE_CONFIGS,
    ENV_HIPFIRE_NPU_DIR,
    ENV_HIPFIRE_NPU_HEADNORM_CONFIGS,
    ENV_HIPFIRE_NPU_HEADNORM_ROPE_CONFIGS,
    ENV_HIPFIRE_NPU_HIDDEN_SIZES,
    ENV_HIPFIRE_NPU_PYTHON,
    ENV_HIPFIRE_NPU_RMSNORM_SIZES,
    ENV_HIPFIRE_NPU_ROPE_CONFIGS,
    ENV_HIPFIRE_NPU_SOFTMAX_CONFIGS,
    ENV_HIPFIRE_NPU_TARGETS,
    ENV_HIPFIRE_OQ4_BATCHED_PREFILL,
    ENV_HIPFIRE_OQ4_PREFILL_ACT_BITS,
    ENV_HIPFIRE_OQ4_TRACE,
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
    ENV_HIPFIRE_PARO_SWIGLU_FUSED,
    ENV_HIPFIRE_PERF_BASELINE,
    ENV_HIPFIRE_PERF_BASELINE_DIR,
    ENV_HIPFIRE_PERPLEXITY_BIN,
    ENV_HIPFIRE_PFLASH_DAEMON_LABELS,
    ENV_HIPFIRE_PFLASH_DEBUG,
    ENV_HIPFIRE_PFLASH_DRAFTER_KV,
    ENV_HIPFIRE_PFLASH_DRAFTER_STATE,
    ENV_HIPFIRE_PFLASH_FRESH,
    ENV_HIPFIRE_PFLASH_NIAH_BENCH_BIN,
    ENV_HIPFIRE_PFLASH_REPORT_TRAIN,
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
    ENV_HIPFIRE_PYTHON,
    ENV_HIPFIRE_Q8_BATCHED_LEGACY,
    ENV_HIPFIRE_Q8_FA_ATTENTION_IGNORE_TREE_BIAS,
    ENV_HIPFIRE_Q8_FA_ATTENTION_ROW_LOOP,
    ENV_HIPFIRE_Q8_FA_ATTENTION_SCALAR_LOOP,
    ENV_HIPFIRE_Q8_FA_ATTENTION_SERIAL_KV_LOOP,
    ENV_HIPFIRE_Q8_GATE_UP_4W,
    ENV_HIPFIRE_Q8_GATE_UP_BENCH,
    ENV_HIPFIRE_Q8_GDN_VERIFY_PER_TOKEN,
    ENV_HIPFIRE_Q8_GDN_VERIFY_SERIAL_FRAMES,
    ENV_HIPFIRE_Q8_WMMA_4W,
    ENV_HIPFIRE_Q8_WMMA_X64,
    ENV_HIPFIRE_QAT_KVNOISE,
    ENV_HIPFIRE_QA_KV_MODES,
    ENV_HIPFIRE_QTIP_BBT_ALPHA,
    ENV_HIPFIRE_QTIP_BEAM,
    ENV_HIPFIRE_QTIP_CODEBOOK,
    ENV_HIPFIRE_QTIP_EVAL_ST,
    ENV_HIPFIRE_QTIP_HESSIAN,
    ENV_HIPFIRE_QUANTIZE_BIN,
    ENV_HIPFIRE_QUANT_CALIB,
    ENV_HIPFIRE_QUANT_CANDS,
    ENV_HIPFIRE_QUANT_DIAG_PATH,
    ENV_HIPFIRE_QUANT_SRC,
    ENV_HIPFIRE_QUANT_THREADS,
    ENV_HIPFIRE_QWEN35_DECODE_BATCH,
    ENV_HIPFIRE_QWEN35_DECODE_BATCH_MAX,
    ENV_HIPFIRE_QWEN35_DECODE_INTERNAL_PARITY,
    ENV_HIPFIRE_QWEN35_DECODE_NATIVE_MULTIROW,
    ENV_HIPFIRE_QWEN35_EXPERT_CACHE_MB,
    ENV_HIPFIRE_QWEN35_EXPERT_CACHE_TRACE,
    ENV_HIPFIRE_QWEN35_FFN_BF16,
    ENV_HIPFIRE_QWEN35_FFN_BF16_LAYER,
    ENV_HIPFIRE_QWEN35_FFN_BF16_TRACE,
    ENV_HIPFIRE_QWEN35_FINITE_TRACE,
    ENV_HIPFIRE_QWEN35_PAGED_EXPERTS,
    ENV_HIPFIRE_QWEN35_PREFILL_SESSION_BATCH,
    ENV_HIPFIRE_QWEN35_ROUTED_ONLY_MOE_FORWARD,
    ENV_HIPFIRE_QWEN35_STAGE_SYNC,
    ENV_HIPFIRE_QWEN35_STAGE_TRACE,
    ENV_HIPFIRE_QWEN35_XDNA1_INSTR,
    ENV_HIPFIRE_QWEN35_XDNA1_XCLBIN,
    ENV_HIPFIRE_QWEN3_DSPARK_CONF_THRESHOLD,
    ENV_HIPFIRE_RDNA2_VARIANT,
    ENV_HIPFIRE_RECOVER_MODE,
    ENV_HIPFIRE_RECOVER_NOISE,
    ENV_HIPFIRE_REPLAY_GRAPH,
    ENV_HIPFIRE_RESOURCE_LOCK,
    ENV_HIPFIRE_RESOURCE_LOCK_CPU_CORES,
    ENV_HIPFIRE_RESOURCE_LOCK_DIR,
    ENV_HIPFIRE_RESOURCE_LOCK_NPUS,
    ENV_HIPFIRE_RESOURCE_LOCK_WAIT_MS,
    ENV_HIPFIRE_RESPONSES_STATE_MAX,
    ENV_HIPFIRE_ROCBLAS_ALL_ARCHS,
    ENV_HIPFIRE_ROCBLAS_OFF,
    ENV_HIPFIRE_ROCPROF_BIN,
    ENV_HIPFIRE_ROCPROF_CSV,
    ENV_HIPFIRE_ROPE_INTERLEAVED_LEGACY,
    ENV_HIPFIRE_RQ2_BULK_BITS,
    ENV_HIPFIRE_RQ2_DAMP,
    ENV_HIPFIRE_RQ2_PROTECT_FRAC,
    ENV_HIPFIRE_RQ2_Q8_EMBED,
    ENV_HIPFIRE_RQ2_SHARE_RESID,
    ENV_HIPFIRE_RQ3_BULK_BITS,
    ENV_HIPFIRE_RQ3_PROTECT_FRAC,
    ENV_HIPFIRE_RQ3_Q8_EMBED,
    ENV_HIPFIRE_RQ4_BULK,
    ENV_HIPFIRE_RQ4_BULK_BITS,
    ENV_HIPFIRE_RQ4_DUMP_RANK,
    ENV_HIPFIRE_RQ4_INVERT,
    ENV_HIPFIRE_RQ4_MQ_BITS,
    ENV_HIPFIRE_RQ4_OBS_DAMP,
    ENV_HIPFIRE_RQ4_PROTECT_FRAC,
    ENV_HIPFIRE_RQ4_PROTECT_Q8,
    ENV_HIPFIRE_RQ4_Q8_EMBED,
    ENV_HIPFIRE_RQ4_RANDOM_SEED,
    ENV_HIPFIRE_RQ4_SALIENCY,
    ENV_HIPFIRE_RQ4_VOID_ONLY,
    ENV_HIPFIRE_RQ_BULK_BITS,
    ENV_HIPFIRE_RQ_GROUP,
    ENV_HIPFIRE_RQ_HAND,
    ENV_HIPFIRE_RQ_PROTECT_FRAC,
    ENV_HIPFIRE_RUN_EXAMPLE_BIN,
    ENV_HIPFIRE_SAMPLE_COMPARE,
    ENV_HIPFIRE_SCORE_TAIL,
    ENV_HIPFIRE_SERVER_RESIDENT_STATE_BUDGET_MB,
    ENV_HIPFIRE_SMOKE_KV,
    ENV_HIPFIRE_SMOKE_KV_SEQ,
    ENV_HIPFIRE_SMOKE_MODE,
    ENV_HIPFIRE_SMOKE_PROMPT,
    ENV_HIPFIRE_SMOKE_STEPS,
    ENV_HIPFIRE_SPEC_PHASES,
    ENV_HIPFIRE_STATE,
    ENV_HIPFIRE_STEER_DUMP,
    ENV_HIPFIRE_TARGET_ARCH,
    ENV_HIPFIRE_TIER_RATIO,
    ENV_HIPFIRE_TINYQUANT_FAMILIES,
    ENV_HIPFIRE_TINYQUANT_RECORD,
    ENV_HIPFIRE_TINY_QUANT_PROBE_BIN,
    ENV_HIPFIRE_TINY_SD_HFQ,
    ENV_HIPFIRE_TP_BENCH_ITERS,
    ENV_HIPFIRE_TP_BENCH_N,
    ENV_HIPFIRE_TP_EXPERT_ASSIGN,
    ENV_HIPFIRE_TP_USE_RCCL,
    ENV_HIPFIRE_UNIFORM_VRAM_TOLERANCE_GB,
    ENV_HIPFIRE_VERIFY_GRAPH,
    ENV_HIPFIRE_VERIFY_GRAPH_TIMING,
    ENV_HIPFIRE_VERIFY_GRAPH_TREE,
    ENV_HIPFIRE_VISION_CACHE,
    ENV_HIPFIRE_VISION_CACHE_DIR,
    ENV_HIPFIRE_VISION_CACHE_MAX_BYTES,
    ENV_HIPFIRE_VISION_DUMP,
    ENV_HIPFIRE_VISION_PROFILE,
    ENV_HIPFIRE_VL_DUMP_DIR,
    ENV_HIPFIRE_WARN_GENERIC,
    ENV_HIPFIRE_WA_CODEC,
    ENV_HIPFIRE_WA_LEARNED,
    ENV_HIPFIRE_WA_R1_ITERS,
    ENV_HIPFIRE_WO_MMQ,
    ENV_HIPFIRE_WO_WMMA_VARIANT,
    ENV_HIPFIRE_XDNA1_LIB,
    ENV_HIP_PATH,
    ENV_HIP_VISIBLE_DEVICES,
    ENV_HOME,
    ENV_HOSTNAME,
    ENV_HUGGINGFACE_HUB_CACHE,
    ENV_MAX_TOKENS,
    ENV_MMQ_TEST_MODE,
    ENV_NANO30B_DIR,
    ENV_NANO4B_DIR,
    ENV_NEMO_TOKENS,
    ENV_NO_COLOR,
    ENV_NO_NGRAM,
    ENV_PATH,
    ENV_PROMPT_MODE,
    ENV_QWEN35_TEST_MODEL,
    ENV_ROCM_DEVICE_LIB_PATH,
    ENV_ROCM_PATH,
    ENV_ROCR_VISIBLE_DEVICES,
    ENV_TERM,
    ENV_TRIALS,
    ENV_USERPROFILE,
    ENV_USE_SAMPLE,
];
