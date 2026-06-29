# hipfire environment variables — canonical reference

Generated automatically from source and inline comments by `hipfire gen-env-docs`.

| Variable | Description | Defined at |
|---|---|---|
| `BENCH_BATCH` | Runtime variable controlling bench batch in hipfire | `crates/rdna-compute/examples/bench_stream_overlap.rs:56` |
| `BENCH_DRAFT_K` | Runtime variable controlling bench draft k in hipfire | `crates/rdna-compute/examples/bench_stream_overlap.rs:188` |
| `BENCH_DRAFT_LAYERS` | Runtime variable controlling bench draft layers in hipfire | `crates/rdna-compute/examples/bench_stream_overlap.rs:215` |
| `BENCH_DRAFT_M` | Runtime variable controlling bench draft m in hipfire | `crates/rdna-compute/examples/bench_stream_overlap.rs:184` |
| `BENCH_DRAFT_N` | Runtime variable controlling bench draft n in hipfire | `crates/rdna-compute/examples/bench_stream_overlap.rs:192` |
| `BENCH_K` | Runtime variable controlling bench k in hipfire | `crates/rdna-compute/examples/bench_stream_overlap.rs:52` |
| `BENCH_M` | Runtime variable controlling bench m in hipfire | `crates/rdna-compute/examples/bench_stream_overlap.rs:48` |
| `BENCH_VERIFY_LAYERS` | Runtime variable controlling bench verify layers in hipfire | `crates/rdna-compute/examples/bench_stream_overlap.rs:219` |
| `CAP_POS` | Environment toggle value controls runtime behavior | `crates/hipfire-arch-nemotron/examples/bisect_nano4b.rs:64` |
| `CARGO_FEATURE_NPU_KERNELS` | Runtime variable controlling cargo feature npu kernels in hipfire | `crates/hipfire-arch-qwen35/build.rs:29` |
| `CARGO_MANIFEST_DIR` | CARGO_MANIFEST_DIR = <workspace>/crates/hipfire-arch-qwen35 | `crates/hipfire-arch-qwen35/build.rs:591` |
| `CARGO_PKG_VERSION` | Defaults to unknown when unset | `crates/hipfire-build-info/build.rs:24` |
| `DDTREE_TIMING` | pre_verify / verify. The draft and top-K are fused into one GPU- | `crates/hipfire-arch-qwen35/src/speculative.rs:11133` |
| `DEBUG_LAYERS` | Runtime variable controlling debug layers in hipfire | `crates/hipfire-arch-qwen35/src/qwen35.rs:8800` |
| `DFLASH_LIVE_TAU` | Runtime variable controlling dflash live tau in hipfire | `crates/hipfire-runtime/examples/dflash_spec_demo.rs:1514` |
| `FP32_STATE` | Runtime variable controlling fp32 state in hipfire | `crates/hipfire-runtime/examples/infer_qwen35.rs:205` |
| `GPU_LOCK_TIMEOUT` | Runtime variable controlling gpu lock timeout in hipfire | `crates/hipfire-cli/src/commands/lock.rs:86` |
| `GPU_POLL_INTERVAL` | Runtime variable controlling gpu poll interval in hipfire | `crates/hipfire-cli/src/commands/lock.rs:79` |
| `HFHS_REAL` | Runtime variable controlling hfhs real in hipfire | `crates/hipfire-quantize/src/hfhs_diag.rs:223` |
| `HFQ_TEST_N_ITER` | Parses "HFQ_TEST_N_ITER" with fallback defaults | `crates/rdna-compute/examples/test_hfq4_residual_dp4a.rs:74` |
| `HFQ_TEST_SCALE_LOG10` | Parses "HFQ_TEST_SCALE_LOG10" with fallback defaults | `crates/rdna-compute/examples/test_hfq4_residual_dp4a.rs:196` |
| `HFQ_TEST_ZP_MAX` | Parses "HFQ_TEST_ZP_MAX" with fallback defaults | `crates/rdna-compute/examples/test_hfq4_residual_dp4a.rs:200` |
| `HF_HOME` | Runtime variable controlling hf home in hipfire | `crates/hipfire-server/src/routes/sdapi.rs:4115` |
| `HIPFIRE_ADAPTIVE_B_DOWN` | Runtime variable controlling adaptive b down in hipfire | `crates/hipfire-runtime/examples/dflash_spec_demo.rs:1781` |
| `HIPFIRE_ADAPTIVE_B_UNSAFE` | the user explicitly widens. Opt out via HIPFIRE_ADAPTIVE_B_UNSAFE=1 | `crates/hipfire-runtime/examples/dflash_spec_demo.rs:923` |
| `HIPFIRE_ADAPTIVE_B_UP` | HIPFIRE_ADAPTIVE_B_UP=0.XX / HIPFIRE_ADAPTIVE_B_DOWN=0.XX | `crates/hipfire-runtime/examples/dflash_spec_demo.rs:1777` |
| `HIPFIRE_ALLOW_MIXED_ARCH` | Runtime variable controlling allow mixed arch in hipfire | `crates/hipfire-runtime/src/config.rs:103` |
| `HIPFIRE_ALLOW_MQ2` | Enabled when set to 1 | `crates/hipfire-quantize/src/main.rs:6040` |
| `HIPFIRE_ALLOW_MQ2_LLOYD` | Enabled when set to 1 | `crates/hipfire-quantize/src/main.rs:6075` |
| `HIPFIRE_ALLOW_MQ3_LLOYD` | Enabled when set to 1 | `crates/hipfire-quantize/src/main.rs:6060` |
| `HIPFIRE_ALLOW_MQ4_LLOYD` | Enabled when set to 1 | `crates/hipfire-quantize/src/main.rs:6110` |
| `HIPFIRE_ALLOW_UNIT_IMATRIX` | Environment toggle value controls runtime behavior | `crates/hipfire-quantize/src/main.rs:5679` |
| `HIPFIRE_ATTN_FLASH` | Honors HIPFIRE_ATTN_FLASH=never\|0\|off as an explicit override | `crates/hipfire-arch-qwen35/src/qwen35.rs:9037` |
| `HIPFIRE_AWQ_F1_ONLY` | F1-vs-F2 A/B gate. When "HIPFIRE_AWQ_F1_ONLY=1" is set, the F2 | `crates/hipfire-quantize/src/main.rs:4417` |
| `HIPFIRE_BASELINE_ARCH` | Runtime variable controlling baseline arch in hipfire | `crates/hipfire-runtime/examples/coherence_probe.rs:403` |
| `HIPFIRE_BATCHES_STATE_MAX` | Parses "HIPFIRE_BATCHES_STATE_MAX" with fallback defaults | `crates/hipfire-server/src/routes/batches.rs:687` |
| `HIPFIRE_BENCH_QWEN35_SPEED_BIN` | Runtime variable controlling bench qwen35 speed bin in hipfire | `crates/hipfire-eval/src/lib.rs:1164` |
| `HIPFIRE_BF16_DENSE_M128` | Enabled by default; set to 0 to disable | `crates/rdna-compute/src/dispatch/gemm_base.rs:716` |
| `HIPFIRE_BF16_MOE_M256` | Enabled when set to 1 | `crates/rdna-compute/src/dispatch/overlays/gfx1151.rs:216` |
| `HIPFIRE_BF16_WEIGHTS` | Runtime variable controlling bf16 weights in hipfire | `crates/hipfire-arch-qwen35/src/qwen35.rs:222` |
| `HIPFIRE_BLOB_FORCE` | Graph / capture / deterministic | `crates/rdna-compute/src/feature_flags.rs:291` |
| `HIPFIRE_CALIB_F64_AUDIT` | Enabled when set to 1 | `crates/hipfire-runtime/src/calibration.rs:135` |
| `HIPFIRE_CALIB_HESSIAN_STORAGE` | Selects behavior from recognized values | `crates/hipfire-runtime/src/calibration.rs:38` |
| `HIPFIRE_CALIB_PROFILE` | Enable with HIPFIRE_CALIB_PROFILE=1; emits to stderr | `crates/hipfire-runtime/examples/triattn_validate.rs:40` |
| `HIPFIRE_CASK_OFF` | HIPFIRE_CASK_OFF=1 is an ops escape hatch: forces no auto-attach | `crates/hipfire-server/src/routes/chat.rs:258` |
| `HIPFIRE_CHATML` | Enabled when set to 1 | `crates/hipfire-runtime/examples/probe_argmax_agreement.rs:58` |
| `HIPFIRE_CHAT_TEMPLATE_FILE` | 1. Env-var override | `crates/hipfire-prompt/src/lib.rs:2034` |
| `HIPFIRE_COLLECT_ARTIFACTS_BIN` | Runtime variable controlling collect artifacts bin in hipfire | `crates/hipfire-eval/src/lib.rs:1211` |
| `HIPFIRE_COMP_DUMP` | Stage-bisect dump: HIPFIRE_COMP_DUMP="<pos>,<layer>" prints each | `crates/hipfire-arch-deepseek4/src/forward.rs:643` |
| `HIPFIRE_CONV1D_TREE_GFX1151` | Environment toggle value controls runtime behavior | `crates/rdna-compute/examples/bench_conv1d_tree_gfx1151.rs:22` |
| `HIPFIRE_DAEMON_BIN` | Runtime variable controlling daemon bin in hipfire | `crates/hipfire-daemon-adapter/src/lib.rs:927` |
| `HIPFIRE_DAEMON_RESIDENT_STATE_BUDGET_MB` | Runtime variable controlling daemon resident state budget mb in hipfire | `crates/hipfire-serving-core/src/session.rs:1410` |
| `HIPFIRE_DDTREE_BUDGET` | Runtime variable controlling DDTree budget in hipfire | `crates/hipfire-serving-core/src/load.rs:3269` |
| `HIPFIRE_DDTREE_FORCE_SLOW` | HIPFIRE_DDTREE_FORCE_SLOW=1: force the slow (re-verify) path even when | `crates/hipfire-arch-qwen35/src/speculative.rs:11360` |
| `HIPFIRE_DDTREE_LOGW_CUTOFF` | Adaptive-B usage report — only meaningful when --adaptive-b is on | `crates/hipfire-runtime/examples/dflash_spec_demo.rs:2519` |
| `HIPFIRE_DDTREE_PATH_B_CAPTURE` | Path B slow-path-kill (opt-in, WIP). Replaces the ~40-50 ms full | `crates/hipfire-arch-qwen35/src/speculative.rs:11399` |
| `HIPFIRE_DDTREE_PATH_C` | Resolve "HIPFIRE_DDTREE_PATH_C" ONCE before the decode loop. The | `crates/hipfire-serving-core/src/generate.rs:843` |
| `HIPFIRE_DDTREE_PATH_C_VERBOSE` | Runtime variable controlling DDTree path c verbose in hipfire | `crates/hipfire-arch-qwen35/src/speculative.rs:11924` |
| `HIPFIRE_DDTREE_PATH_C_VERIFY_GRAPH` | Runtime variable controlling DDTree path c verify graph in hipfire | `crates/hipfire-arch-qwen35/src/speculative.rs:203` |
| `HIPFIRE_DDTREE_TAPE_DUMP` | Enabled when set to 1 | `crates/hipfire-arch-qwen35/src/speculative.rs:11379` |
| `HIPFIRE_DDTREE_TOPK` | Runtime variable controlling DDTree topk in hipfire | `crates/hipfire-serving-core/src/load.rs:3341` |
| `HIPFIRE_DDTREE_TREE_LA` | Opt out with HIPFIRE_DDTREE_TREE_LA=0 if a regression is suspected | `crates/hipfire-arch-qwen35/src/speculative.rs:11241` |
| `HIPFIRE_DEBUG_CHAT` | Enabled when set to 1 | `crates/hipfire-server/src/routes/chat.rs:112` |
| `HIPFIRE_DEBUG_PREFILL_ELIGIBLE` | Environment toggle value controls runtime behavior | `crates/hipfire-arch-qwen35/src/qwen35.rs:13842` |
| `HIPFIRE_DEBUG_PREFIX_BOUNDARIES` | Runtime variable controlling debug prefix boundaries in hipfire | `crates/hipfire-serving-core/src/qwen35_prefill.rs:626` |
| `HIPFIRE_DEEPSEEK4_ATTN` | main model's final_norm_and_head head-HC reduction. Without | `crates/hipfire-arch-deepseek4/src/forward.rs:73` |
| `HIPFIRE_DEEPSEEK4_ATTN_DEBUG_BISECT` | DEBUG: in-kernel bisect (HIPFIRE_DEEPSEEK4_ATTN_DEBUG_BISECT=1) | `crates/hipfire-arch-deepseek4/src/forward.rs:6220` |
| `HIPFIRE_DEEPSEEK4_ATTN_PER_POS` | DEBUG: HIPFIRE_DEEPSEEK4_ATTN_PER_POS=1 substitutes a per-position loop | `crates/hipfire-arch-deepseek4/src/forward.rs:6153` |
| `HIPFIRE_DEEPSEEK4_ATTN_TOPK_DIRECT` | Runtime variable controlling deepseek4 attn topk direct in hipfire | `crates/hipfire-arch-deepseek4/src/forward.rs:6502` |
| `HIPFIRE_DEEPSEEK4_ATTN_TWIN` | DEBUG: same-process twin-call test (HIPFIRE_DEEPSEEK4_ATTN_TWIN=1) | `crates/hipfire-arch-deepseek4/src/forward.rs:6198` |
| `HIPFIRE_DEEPSEEK4_BATCH_HEAD` | Opt-out: HIPFIRE_DEEPSEEK4_BATCH_HEAD=0 forces the legacy per-position | `crates/hipfire-arch-deepseek4/src/forward.rs:8162` |
| `HIPFIRE_DEEPSEEK4_CACHE_TRACE` | Runtime variable controlling deepseek4 cache trace in hipfire | `crates/hipfire-serving-core/src/generate_arch.rs:970` |
| `HIPFIRE_DEEPSEEK4_CHAT_RAW` | Enabled when set to 1 | `crates/hipfire-arch-deepseek4/examples/deepseek4_chat.rs:161` |
| `HIPFIRE_DEEPSEEK4_COMP_F16_WMMA` | Opt out via HIPFIRE_DEEPSEEK4_COMP_F16_WMMA=0 | `crates/hipfire-arch-deepseek4/src/forward.rs:6573` |
| `HIPFIRE_DEEPSEEK4_COMP_ROPE_POS` | Runtime variable controlling deepseek4 comp rope pos in hipfire | `crates/hipfire-arch-deepseek4/src/forward.rs:5234` |
| `HIPFIRE_DEEPSEEK4_DSA_WMMA` | Head-batched f16-WMMA gathered DSA attention; f32 fallback on | `crates/hipfire-arch-deepseek4/src/forward.rs:7111` |
| `HIPFIRE_DEEPSEEK4_DUMP_PROMPT` | Runtime variable controlling deepseek4 dump prompt in hipfire | `crates/hipfire-serving-core/src/generate_arch.rs:320` |
| `HIPFIRE_DEEPSEEK4_DUMP_STATE` | Selects behavior from recognized values | `crates/hipfire-arch-deepseek4/src/forward.rs:203` |
| `HIPFIRE_DEEPSEEK4_DUMP_TOPK` | Gate 1 validation (2026-05-22): dump per-layer topk_indices to a | `crates/hipfire-dispatch/src/pipeline/mod.rs:1072` |
| `HIPFIRE_DEEPSEEK4_EXPERT_LAYER_END` | Runtime variable controlling deepseek4 expert layer end in hipfire | `crates/hipfire-arch-deepseek4/src/arch.rs:599` |
| `HIPFIRE_DEEPSEEK4_F32_TRACE` | Runtime variable controlling deepseek4 f32 trace in hipfire | `crates/hipfire-arch-deepseek4/src/forward.rs:231` |
| `HIPFIRE_DEEPSEEK4_FUSED_UNSCATTER_SILU` | Measured perf at PP_BATCH=512 / 2.1k-tok prompt: -0.4% prefill — | `crates/hipfire-dispatch/src/pipeline/mod.rs:1163` |
| `HIPFIRE_DEEPSEEK4_GEN_TOKENS` | Runtime variable controlling deepseek4 gen tokens in hipfire | `crates/hipfire-arch-deepseek4/examples/deepseek4_chat.rs:157` |
| `HIPFIRE_DEEPSEEK4_GRAPH` | Selects behavior from recognized values | `crates/hipfire-arch-deepseek4/src/forward.rs:1564` |
| `HIPFIRE_DEEPSEEK4_HFQ4_WMMA` | HFQ4G256/Raw. WMMA route requires F16 input staging | `crates/hipfire-arch-deepseek4/src/forward.rs:328` |
| `HIPFIRE_DEEPSEEK4_INDEXER_WMMA` | Runtime variable controlling deepseek4 indexer wmma in hipfire | `crates/hipfire-arch-deepseek4/src/forward.rs:6928` |
| `HIPFIRE_DEEPSEEK4_LOAD_MTP` | Runtime variable controlling deepseek4 load MTP in hipfire | `crates/hipfire-arch-deepseek4/src/arch.rs:998` |
| `HIPFIRE_DEEPSEEK4_MAX_COMPRESS_POS` | ratio == 128: identity gather, no indexer. Per-batch n_compressed | `crates/hipfire-arch-deepseek4/src/forward.rs:7010` |
| `HIPFIRE_DEEPSEEK4_MODEL` | Runtime variable controlling deepseek4 model in hipfire | `crates/hipfire-arch-deepseek4/examples/deepseek4_chat.rs:154` |
| `HIPFIRE_DEEPSEEK4_MOE` | Layers 0..num_hash_layers use STATIC tid2eid routing per upstream DeepSeek V4 | `crates/hipfire-arch-deepseek4/src/forward.rs:7454` |
| `HIPFIRE_DEEPSEEK4_MOE_8W` | Lever 3 (HIPFIRE_DEEPSEEK4_MOE_8W=1): 8-warp variant (shares staged X | `crates/hipfire-dispatch/src/pipeline/mod.rs:1112` |
| `HIPFIRE_DEEPSEEK4_MOE_CND` | Lever 2 (HIPFIRE_DEEPSEEK4_MOE_CND=1): cndmask dequant on the n16 4w | `crates/hipfire-dispatch/src/pipeline/mod.rs:1111` |
| `HIPFIRE_DEEPSEEK4_MOE_DETERMINISTIC` | Environment toggle value controls runtime behavior | `crates/hipfire-dispatch/src/pipeline/mod.rs:1264` |
| `HIPFIRE_DEEPSEEK4_MOE_GROUPED` | Environment toggle value controls runtime behavior | `crates/hipfire-dispatch/src/pipeline/mod.rs:1101` |
| `HIPFIRE_DEEPSEEK4_MOE_GROUPED_GATE` | HIPFIRE_DEEPSEEK4_MOE_GROUPED_GATE overrides the threshold for | `crates/hipfire-dispatch/src/pipeline/mod.rs:1096` |
| `HIPFIRE_DEEPSEEK4_MOE_LLOYD_4W` | RECONCILED (#355 + #356): same 4w default + levers as gate_up above | `crates/hipfire-dispatch/src/pipeline/mod.rs:1104` |
| `HIPFIRE_DEEPSEEK4_MOE_MMQLOAD` | n32_env / cnd_env / eightw_env reused from the gate_up block above | `crates/hipfire-dispatch/src/pipeline/mod.rs:1113` |
| `HIPFIRE_DEEPSEEK4_MOE_N32` | Lever 1 (HIPFIRE_DEEPSEEK4_MOE_N32=1): N_TILE=32 tile-pairing on the | `crates/hipfire-dispatch/src/pipeline/mod.rs:1110` |
| `HIPFIRE_DEEPSEEK4_MOE_NOSYNC` | n32_env / cnd_env / eightw_env reused from the gate_up block above | `crates/hipfire-dispatch/src/pipeline/mod.rs:1114` |
| `HIPFIRE_DEEPSEEK4_MTP_ADDON` | Convention 1: append ".mtp-addon.hfq" (legacy) | `crates/hipfire-arch-deepseek4/src/arch.rs:619` |
| `HIPFIRE_DEEPSEEK4_MTP_HEAD_HC` | Runtime variable controlling deepseek4 MTP head hc in hipfire | `crates/hipfire-arch-deepseek4/src/forward.rs:86` |
| `HIPFIRE_DEEPSEEK4_MTP_SKIP_HEAD` | 3. Batched MTP fill — single pass through the MTP layer for all | `crates/hipfire-arch-deepseek4/src/forward.rs:9010` |
| `HIPFIRE_DEEPSEEK4_POST_SCALE` | Runtime variable controlling deepseek4 post scale in hipfire | `crates/hipfire-arch-deepseek4/src/forward.rs:8357` |
| `HIPFIRE_DEEPSEEK4_PP_BATCH` | on 2026-05-26). PP_BATCH sweep on the 2.1k-tok bench (3 trials/cell): | `crates/hipfire-serving-core/src/load.rs:1231` |
| `HIPFIRE_DEEPSEEK4_Q8_4W` | Opt out via HIPFIRE_DEEPSEEK4_Q8_4W=0 for diagnosis | `crates/hipfire-arch-deepseek4/src/forward.rs:292` |
| `HIPFIRE_DEEPSEEK4_Q8_WMMA` | bench_q8_wmma_variants. Opt-out via HIPFIRE_DEEPSEEK4_Q8_WMMA=0 | `crates/hipfire-arch-deepseek4/src/forward.rs:246` |
| `HIPFIRE_DEEPSEEK4_ROUTE_SCALE` | Runtime variable controlling deepseek4 route scale in hipfire | `crates/hipfire-arch-deepseek4/src/forward.rs:7474` |
| `HIPFIRE_DEEPSEEK4_SEED` | Runtime variable controlling deepseek4 seed in hipfire | `crates/hipfire-serving-core/src/generate_arch.rs:544` |
| `HIPFIRE_DEEPSEEK4_SPEC_DECODE` | Priority: 1. legacy env var → 2. generic env var → 3. stored config → default | `crates/hipfire-serving-core/src/generate_arch.rs:342` |
| `HIPFIRE_DEEPSEEK4_SPEC_K` | Runtime variable controlling deepseek4 spec k in hipfire | `crates/hipfire-serving-core/src/generate_arch.rs:350` |
| `HIPFIRE_DEEPSEEK4_TEMP` | Runtime variable controlling deepseek4 temp in hipfire | `crates/hipfire-arch-deepseek4/examples/deepseek4_chat.rs:162` |
| `HIPFIRE_DEEPSEEK4_TOP_K` | for local deployment; we honor that as the default. Pure greedy | `crates/hipfire-serving-core/src/generate_arch.rs:540` |
| `HIPFIRE_DEEPSEEK4_UPLOAD_EXPERTS` | without them). Opt out with "HIPFIRE_DEEPSEEK4_UPLOAD_EXPERTS=0" | `crates/hipfire-arch-deepseek4/src/arch.rs:595` |
| `HIPFIRE_DEEPSEEK4_WO_MULTIROW` | Q8_0 contract: plain (non-FWHT) input. Same layout | `crates/hipfire-arch-deepseek4/src/forward.rs:7215` |
| `HIPFIRE_DEEPSEEK4_WO_Q8_WMMA` | Runtime variable controlling deepseek4 wo Q8 wmma in hipfire | `crates/rdna-compute/src/dispatch/deepseek4.rs:3287` |
| `HIPFIRE_DELTANET_STATE` | Interprets "HIPFIRE_DELTANET_STATE" from environment to select behavior | `crates/hipfire-runtime/examples/greedy_dump_top5.rs:246` |
| `HIPFIRE_DETERMINISTIC` | Enabled when set to 1 | `crates/rdna-compute/src/feature_flags.rs:293` |
| `HIPFIRE_DEVICES` | Runtime variable controlling devices in hipfire | `crates/hipfire-runtime/src/config.rs:102` |
| `HIPFIRE_DFLASH_DRAFT` | Runtime variable controlling dflash draft in hipfire | `crates/hipfire-server/src/routes/chat.rs:242` |
| `HIPFIRE_DFLASH_LOOP_BREAK` | Memory: HashSet<u64>, ≤ max_tokens / 12 entries (~125 for max=1500) | `crates/hipfire-runtime/examples/dflash_spec_demo.rs:1435` |
| `HIPFIRE_DFLASH_LOOP_BREAK_MAX_ESCALATIONS` | Runtime variable controlling dflash loop break max escalations in hipfire | `crates/hipfire-runtime/examples/dflash_spec_demo.rs:1464` |
| `HIPFIRE_DFLASH_LOOP_BREAK_RECOVERY` | Runtime variable controlling dflash loop break recovery in hipfire | `crates/hipfire-runtime/examples/dflash_spec_demo.rs:1459` |
| `HIPFIRE_DFLASH_LOOP_BREAK_RP_MAX` | Runtime variable controlling dflash loop break rp max in hipfire | `crates/hipfire-runtime/examples/dflash_spec_demo.rs:1455` |
| `HIPFIRE_DFLASH_LOOP_BREAK_RP_STEP` | Runtime variable controlling dflash loop break rp step in hipfire | `crates/hipfire-runtime/examples/dflash_spec_demo.rs:1451` |
| `HIPFIRE_DFLASH_LOOP_BREAK_STOP_AFTER` | Runtime variable controlling dflash loop break stop after in hipfire | `crates/hipfire-runtime/examples/dflash_spec_demo.rs:1447` |
| `HIPFIRE_DFLASH_LOOP_BREAK_TEMP` | Runtime variable controlling dflash loop break temp in hipfire | `crates/hipfire-runtime/examples/dflash_spec_demo.rs:1443` |
| `HIPFIRE_DFLASH_MODE` | Defaults to off when unset | `crates/hipfire-runtime/src/config.rs:74` |
| `HIPFIRE_DFLASH_MOE_DRAFT_FFN_GRAPH` | Runtime variable controlling dflash moe draft ffn graph in hipfire | `crates/hipfire-arch-qwen35/src/speculative.rs:7123` |
| `HIPFIRE_DFLASH_MOE_VERIFY_GRAPH_LMHEAD` | Runtime variable controlling dflash moe verify graph lmhead in hipfire | `crates/hipfire-arch-qwen35/src/speculative.rs:6494` |
| `HIPFIRE_DFLASH_NGRAM_BLOCK` | Enabled when set to 1 | `crates/hipfire-server/src/routes/chat.rs:181` |
| `HIPFIRE_DFLASH_Q8_LMHEAD_WMMA` | Selects behavior from recognized values | `crates/hipfire-arch-qwen35/src/speculative.rs:114` |
| `HIPFIRE_DFLASH_ROLLBACK_COMPARE` | Used to configure runtime execution by explicitly setting "HIPFIRE_DFLASH_ROLLBACK_COMPARE" | `crates/hipfire-arch-qwen35/src/speculative.rs:12454` |
| `HIPFIRE_DFLASH_ROLLBACK_FA_RAW_ATOL` | Runtime variable controlling dflash rollback fa raw atol in hipfire | `crates/hipfire-arch-qwen35/src/speculative.rs:1090` |
| `HIPFIRE_DFLASH_ROLLBACK_LOGIT_COMPARE_STEPS` | Used to configure runtime execution by explicitly setting "HIPFIRE_DFLASH_ROLLBACK_LOGIT_COMPARE_STEPS" | `crates/hipfire-arch-qwen35/src/speculative.rs:12478` |
| `HIPFIRE_DFLASH_ROLLBACK_PREFIX_VERIFY` | Used to configure runtime execution by explicitly setting "HIPFIRE_DFLASH_ROLLBACK_PREFIX_VERIFY" | `crates/hipfire-arch-qwen35/src/speculative.rs:12382` |
| `HIPFIRE_DFLASH_ROLLBACK_SERIAL_REPLAY` | Used to configure runtime execution by explicitly setting "HIPFIRE_DFLASH_ROLLBACK_SERIAL_REPLAY" | `crates/hipfire-arch-qwen35/src/speculative.rs:12350` |
| `HIPFIRE_DFLASH_ROLLBACK_SERIAL_TAPE` | Used to configure runtime execution by explicitly setting "HIPFIRE_DFLASH_ROLLBACK_SERIAL_TAPE" | `crates/hipfire-arch-qwen35/src/speculative.rs:12430` |
| `HIPFIRE_DFLASH_ROLLBACK_VERIFY_FRAMES` | Used to configure runtime execution by explicitly setting "HIPFIRE_DFLASH_ROLLBACK_VERIFY_FRAMES" | `crates/hipfire-arch-qwen35/src/speculative.rs:12406` |
| `HIPFIRE_DFLASH_ROLLBACK_X_IN_ATOL` | Parses "HIPFIRE_DFLASH_ROLLBACK_X_IN_ATOL" with fallback defaults | `crates/hipfire-arch-qwen35/src/speculative.rs:1098` |
| `HIPFIRE_DFLASH_SEED_ORACLE` | Enabled when set to 1 | `crates/hipfire-arch-qwen35/src/speculative.rs:7763` |
| `HIPFIRE_DFLASH_SERIAL_QKVZA_SELF_COMPARE` | Runtime variable controlling dflash serial qKVza self compare in hipfire | `crates/hipfire-arch-qwen35/src/qwen35.rs:14282` |
| `HIPFIRE_DFLASH_SERIAL_TAPE_X_IN_COMPARE` | Runtime variable controlling dflash serial tape x in compare in hipfire | `crates/hipfire-arch-qwen35/src/qwen35.rs:14287` |
| `HIPFIRE_DFLASH_SPEC_DEMO_BIN` | Runtime variable controlling dflash spec demo bin in hipfire | `crates/hipfire-eval/src/lib.rs:1149` |
| `HIPFIRE_DFLASH_TRACE_EXPECTED_TOKEN` | Runtime variable controlling dflash trace expected token in hipfire | `crates/hipfire-runtime/examples/dflash_spec_demo.rs:139` |
| `HIPFIRE_DFLASH_TRACE_POSITION` | Runtime variable controlling dflash trace position in hipfire | `crates/hipfire-runtime/examples/dflash_spec_demo.rs:133` |
| `HIPFIRE_DFLASH_TRACE_TOKEN_INDEX` | Runtime variable controlling dflash trace token index in hipfire | `crates/hipfire-runtime/examples/dflash_spec_demo.rs:1806` |
| `HIPFIRE_DIAG` | Diagnostic: quant-vs-bf16 fidelity per layer (target is the bf16 FFN) | `crates/hipfire-train/examples/qwen35_mlp_norm_recovery.rs:264` |
| `HIPFIRE_DIFFUSION_CPU_REFERENCE` | Runtime variable controlling diffusion cpu reference in hipfire | `crates/hipfire-diffusion/src/lib.rs:221` |
| `HIPFIRE_DIR` | Runtime variable controlling dir in hipfire | `crates/hipfire-serving-core/src/generate_vl.rs:1123` |
| `HIPFIRE_DN_STATE_EF` | Runtime variable controlling dn state ef in hipfire | `crates/hipfire-arch-qwen35/src/qwen35.rs:1432` |
| `HIPFIRE_DN_STATE_FP32_BELOW` | Used to configure runtime execution by explicitly setting "HIPFIRE_DN_STATE_FP32_BELOW" | `crates/hipfire-arch-qwen35/src/qwen35.rs:30447` |
| `HIPFIRE_DOT2_GEMV` | Interprets "HIPFIRE_DOT2_GEMV" from environment to select behavior | `crates/rdna-compute/src/feature_flags.rs:192` |
| `HIPFIRE_DOTS_OCR_BF16_RESIDUAL` | HF cast x to bf16 at vision forward entry (modeling_dots_vision.py | `crates/hipfire-arch-dots-ocr/src/dots_ocr.rs:1119` |
| `HIPFIRE_DOTS_OCR_DUMP_DIR` | HIPFIRE_DOTS_OCR_DUMP_DIR=<path>: dump full per-stage tensor | `crates/hipfire-arch-dots-ocr/src/dots_ocr.rs:1021` |
| `HIPFIRE_DOTS_OCR_TRACE` | HIPFIRE_DOTS_OCR_TRACE=1: sync after every step + print probe so | `crates/hipfire-arch-dots-ocr/src/dots_ocr.rs:1149` |
| `HIPFIRE_DPM_WARMUP_SECS` | Runtime variable controlling dpm warmup secs in hipfire | `crates/rdna-compute/examples/bench_stream_overlap.rs:60` |
| `HIPFIRE_DRAFT_F16` | Enabled by default; set to 0 to disable | `crates/hipfire-runtime/src/config.rs:75` |
| `HIPFIRE_DRAFT_GEMM_DUMP` | Enabled when set to 1 | `crates/hipfire-runtime/src/config.rs:76` |
| `HIPFIRE_DRAFT_SUBPHASE` | Enabled when set to 1 | `crates/hipfire-runtime/src/config.rs:77` |
| `HIPFIRE_DTOH_DUMP` | Enabled when set to 1 | `crates/hip-bridge/src/ffi.rs:955` |
| `HIPFIRE_DUMMY_GENERATE_DELAY_MS` | Parses "HIPFIRE_DUMMY_GENERATE_DELAY_MS" with fallback defaults | `crates/hipfire-serving-core/src/dummy.rs:140` |
| `HIPFIRE_DUMMY_PREFILL_DELAY_MS` | Parses "HIPFIRE_DUMMY_PREFILL_DELAY_MS" with fallback defaults | `crates/hipfire-serving-core/src/dummy.rs:129` |
| `HIPFIRE_DUMP_HIDDEN` | DIAG: dump router logits before softmax (mirrors qwen35 HIPFIRE_DUMP_HIDDEN) | `crates/hipfire-dispatch/src/pipeline/mod.rs:386` |
| `HIPFIRE_DUMP_HIDDEN_ALL` | Activation-capture mode (HIPFIRE_DUMP_HIDDEN_ALL=1): dump EVERY row for a | `crates/hipfire-arch-qwen35/src/qwen35.rs:16779` |
| `HIPFIRE_DUMP_HIDDEN_ALLLAYERS` | - HIPFIRE_DUMP_HIDDEN_ALLLAYERS=1: capture EVERY layer to per-layer | `crates/hipfire-arch-qwen35/src/qwen35.rs:16786` |
| `HIPFIRE_DUMP_HIDDEN_LAYER` | Runtime variable controlling dump hidden layer in hipfire | `crates/hipfire-arch-qwen35/src/qwen35.rs:16790` |
| `HIPFIRE_DUMP_HIDDEN_POS` | Runtime variable controlling dump hidden pos in hipfire | `crates/hipfire-arch-qwen35/src/qwen35.rs:16818` |
| `HIPFIRE_DUMP_REQUEST` | a strace. Off by default — gigantic for typical agent prompts | `crates/hipfire-server/src/routes/chat.rs:85` |
| `HIPFIRE_EMIT_TOKEN_IDS` | or "\" in it would corrupt the line, breaking the client's JSONL | `crates/hipfire-serving-core/src/events.rs:111` |
| `HIPFIRE_EP_DECODE_TIMING` | 2. Per-layer EP program (Attend replicated; Moe all-reduce-EP'd) | `crates/hipfire-arch-minimax/src/forward.rs:1292` |
| `HIPFIRE_EP_DUMP_IDX` | top-k chaos. HIPFIRE_EP_DUMP_IDX=1 to enable | `crates/hipfire-arch-deepseek4/src/forward.rs:2428` |
| `HIPFIRE_EP_DUMP_POS` | Divergence-localization dump: HIPFIRE_EP_DUMP_POS="0,64,...,302" prints a | `crates/hipfire-arch-deepseek4/src/forward.rs:2363` |
| `HIPFIRE_EP_KV_SEQ` | Runtime variable controlling ep KV seq in hipfire | `crates/hipfire-runtime/examples/ep_decode_parity.rs:72` |
| `HIPFIRE_EP_PEER_ALLREDUCE` | RCCL with HIPFIRE_EP_PEER_ALLREDUCE=0. The peer temps live in Gpus (shared | `crates/hipfire-arch-qwen35/src/qwen35.rs:28087` |
| `HIPFIRE_EP_PEER_ALLREDUCE_DECODE` | Environment toggle value controls runtime behavior | `crates/hipfire-runtime/src/ep.rs:135` |
| `HIPFIRE_EP_PREFILL` | Prefill mode: HIPFIRE_EP_PREFILL=batched → WMMA batched prefill EP (E6b) | `crates/hipfire-runtime/examples/ep_decode_parity.rs:99` |
| `HIPFIRE_EP_PREFILL_TIMING` | Gpus::all_reduce_sum_f32_peer (direct P2P copy + local add), which is ~1 ms | `crates/hipfire-arch-qwen35/src/qwen35.rs:28080` |
| `HIPFIRE_EP_SKIP_ALLREDUCE` | Gpus::all_reduce_sum_f32_peer (direct P2P copy + local add), which is ~1 ms | `crates/hipfire-arch-qwen35/src/qwen35.rs:28081` |
| `HIPFIRE_EVAL_DATASET_MIRROR` | Runtime variable controlling eval dataset mirror in hipfire | `crates/hipfire-eval/src/datasets.rs:203` |
| `HIPFIRE_EVAL_EVIDENCE_DIR` | Runtime variable controlling eval evidence dir in hipfire | `crates/hipfire-runtime/examples/run.rs:231` |
| `HIPFIRE_EVAL_KLDREF` | Runtime variable controlling eval kldref in hipfire | `crates/hipfire-eval/src/executor_examples.rs:678` |
| `HIPFIRE_EVAL_PERPLEXITY_CORPUS` | Runtime variable controlling eval perplexity corpus in hipfire | `crates/hipfire-eval/src/run.rs:175` |
| `HIPFIRE_EVAL_PERPLEXITY_CTX` | Parses "HIPFIRE_EVAL_PERPLEXITY_CTX" with fallback defaults | `crates/hipfire-eval/src/executor_examples.rs:764` |
| `HIPFIRE_EXPERIMENTAL_BUDGET_ALERT` | cache so the model "sees" them as part of its own trajectory, | `crates/hipfire-daemon/src/main.rs:3692` |
| `HIPFIRE_FILES_STATE_MAX` | Parses "HIPFIRE_FILES_STATE_MAX" with fallback defaults | `crates/hipfire-server/src/routes/files.rs:174` |
| `HIPFIRE_FLASH_PARTIALS_BATCH` | Parses "HIPFIRE_FLASH_PARTIALS_BATCH" with fallback defaults | `crates/hipfire-runtime/src/config.rs:87` |
| `HIPFIRE_FORCE_GENERIC` | Runtime variable controlling force generic in hipfire | `crates/rdna-compute/src/feature_flags.rs:325` |
| `HIPFIRE_FORCE_UNFUSED` | Interpreter Phase 2a | `crates/rdna-compute/src/feature_flags.rs:321` |
| `HIPFIRE_FORWARD_LOWERED` | Enabled by default; set to 0 to disable | `crates/hipfire-arch-qwen35/src/qwen35.rs:27745` |
| `HIPFIRE_FP16` | escape hatch to the LA qkvza projection while debugging DFlash | `crates/rdna-compute/src/feature_flags.rs:200` |
| `HIPFIRE_FP8_WMMA` | Interprets "HIPFIRE_FP8_WMMA" from environment to select behavior | `crates/rdna-compute/src/feature_flags.rs:191` |
| `HIPFIRE_FUSED_HFQ4_2ROW_GFX1151` | Selects behavior from recognized values | `crates/rdna-compute/src/dispatch/mod.rs:3331` |
| `HIPFIRE_GATE_UP_VARIANT` | Runtime variable controlling gate up variant in hipfire | `crates/rdna-compute/src/feature_flags.rs:246` |
| `HIPFIRE_GDN_Q8_REG_GFX1151` | HIPFIRE_GDN_Q8_REG_GFX1151=1 enables the gfx1151 register-state | `crates/rdna-compute/src/dispatch/mod.rs:3886` |
| `HIPFIRE_GEMMA3_CALIB_IMATRIX_ONLY` | AWQ-style calibration mode: with no K×K outer-product or multi-GB Hessian | `crates/hipfire-arch-gemma3/src/calibration.rs:256` |
| `HIPFIRE_GEMMA3_CALIB_LAYERS_PER_PASS` | Runtime variable controlling gemma3 calib layers per pass in hipfire | `crates/hipfire-arch-gemma3/src/calibration.rs:88` |
| `HIPFIRE_GEMMA3_CALIB_MICROBATCH` | Runtime variable controlling gemma3 calib microbatch in hipfire | `crates/hipfire-arch-gemma3/src/calibration.rs:118` |
| `HIPFIRE_GEMMA3_CALIB_NO_BATCH` | Enabled when set to 1 | `crates/hipfire-arch-gemma3/src/calibration.rs:112` |
| `HIPFIRE_GEMMA3_NO_BATCHED_PREFILL` | Runtime variable controlling gemma3 no batched prefill in hipfire | `crates/hipfire-arch-gemma3/src/arch.rs:100` |
| `HIPFIRE_GEMMA3_PREFILL_MICROBATCH` | Runtime variable controlling gemma3 prefill microbatch in hipfire | `crates/hipfire-arch-gemma3/src/arch.rs:106` |
| `HIPFIRE_GEMMA3_VISION_NOWMMA` | Runtime variable controlling gemma3 vision nowmma in hipfire | `crates/hipfire-arch-gemma3-vl/src/forward.rs:204` |
| `HIPFIRE_GEMM_DUMP` | Enabled when set to 1 | `crates/rdna-compute/src/feature_flags.rs:292` |
| `HIPFIRE_GEMV_ROWS` | Runtime variable controlling gemv rows in hipfire | `crates/rdna-compute/src/feature_flags.rs:172` |
| `HIPFIRE_GEN` | Runtime variable controlling gen in hipfire | `crates/hipfire-runtime/examples/a3b_multiturn_oneshot.rs:26` |
| `HIPFIRE_GENERATE_BATCH_PREFILL_DEBUG_SAMPLE` | Runtime variable controlling generate batch prefill debug sample in hipfire | `crates/hipfire-serving-core/src/qwen35_prefill.rs:1624` |
| `HIPFIRE_GFX942_GEMV_V3` | Interprets "HIPFIRE_GFX942_GEMV_V3" from environment to select behavior | `crates/rdna-compute/src/feature_flags.rs:248` |
| `HIPFIRE_GFX942_MFMA_PREFILL` | Interprets "HIPFIRE_GFX942_MFMA_PREFILL" from environment to select behavior | `crates/rdna-compute/src/feature_flags.rs:251` |
| `HIPFIRE_GFX942_RMSNORM_SPLIT` | Environment toggle value controls runtime behavior | `crates/rdna-compute/src/feature_flags.rs:250` |
| `HIPFIRE_GPTQ_DAMPING` | Inject env override since the quantizer reads it at fn entry | `crates/hipfire-quantize/src/main.rs:10472` |
| `HIPFIRE_GPU_SLAB_LOAD` | Runtime variable controlling gpu slab load in hipfire | `crates/hipfire-arch-qwen35/src/qwen35.rs:5330` |
| `HIPFIRE_GPU_SLAB_MIB` | Parses "HIPFIRE_GPU_SLAB_MIB" with fallback defaults | `crates/hipfire-arch-qwen35/src/qwen35.rs:4288` |
| `HIPFIRE_GPU_TOPK` | HIPFIRE_GPU_TOPK=1 enables the GPU topk_logits_f32 kernel + CPU | `crates/hipfire-runtime/examples/infer_qwen35.rs:235` |
| `HIPFIRE_GQA_CHUNK` | Runtime variable controlling gqa chunk in hipfire | `crates/rdna-compute/src/dispatch/attention.rs:297` |
| `HIPFIRE_GQA_FUSED` | Fused variant (opt-in via HIPFIRE_GQA_FUSED=1): single launch | `crates/hipfire-arch-qwen2/src/qwen2.rs:1720` |
| `HIPFIRE_GRAPH` | Used to configure runtime execution by explicitly setting "HIPFIRE_GRAPH" | `crates/hipfire-runtime/examples/prefill_microbench.rs:124` |
| `HIPFIRE_GRAPH_MOE` | - gfx11 (RDNA3 / 3.5): default-ON. +0.6-0.7% decode on 9B and | `crates/hipfire-arch-qwen35/src/qwen35.rs:9286` |
| `HIPFIRE_GRAPH_PREFILL` | HIPFIRE_GRAPH_PREFILL=1: route the timed prefill loop through | `crates/hipfire-runtime/examples/bench_qwen35_speed.rs:186` |
| `HIPFIRE_HAVE_2_GPU` | Environment toggle value controls runtime behavior | `crates/hipfire-arch-qwen35/tests/pp_parity.rs:193` |
| `HIPFIRE_HETERO_DIFF` | above (#352's GPU greedy-accept path doesn't materialize it), | `crates/hipfire-arch-qwen35/src/mtp_spec.rs:3008` |
| `HIPFIRE_HFQ4G128_MMQ` | Environment toggle value controls runtime behavior | `crates/rdna-compute/src/feature_flags.rs:241` |
| `HIPFIRE_HFQ4G256_MMQ_GFX1151` | Selects behavior from recognized values | `crates/rdna-compute/src/dispatch/mod.rs:3314` |
| `HIPFIRE_HFQ4_GATE_UP_FAST` | HIPFIRE_HFQ4_GATE_UP_FAST=0 narrows the escape hatch to the | `crates/rdna-compute/src/feature_flags.rs:214` |
| `HIPFIRE_HFQ4_MMQ_GFX906_Y64` | Runtime variable controlling hfQ4 mmq gfx906 y64 in hipfire | `crates/rdna-compute/src/feature_flags.rs:244` |
| `HIPFIRE_HFQ4_QKVZA_FAST` | HIPFIRE_HFQ4_QKVZA_FAST=0 narrows the FP16/WMMA/dot2 prefill | `crates/rdna-compute/src/feature_flags.rs:206` |
| `HIPFIRE_HFQ4_QKV_FAST` | HIPFIRE_HFQ4_QKV_FAST=0 narrows the escape hatch to the | `crates/rdna-compute/src/feature_flags.rs:210` |
| `HIPFIRE_HFQ4_RESIDUAL_FAST` | HIPFIRE_HFQ4_RESIDUAL_FAST=0 narrows the residual-projection | `crates/rdna-compute/src/feature_flags.rs:218` |
| `HIPFIRE_HFQ6_QKVZA_4W` | Interprets "HIPFIRE_HFQ6_QKVZA_4W" from environment to select behavior | `crates/rdna-compute/src/dispatch/gemm_qkv.rs:4700` |
| `HIPFIRE_HFQ6_QKV_4W` | Interprets "HIPFIRE_HFQ6_QKV_4W" from environment to select behavior | `crates/rdna-compute/src/dispatch/gemm_qkv.rs:5216` |
| `HIPFIRE_HFQ6_RESIDUAL_4W` | Interprets "HIPFIRE_HFQ6_RESIDUAL_4W" from environment to select behavior | `crates/rdna-compute/src/dispatch/gemm_hfq.rs:2672` |
| `HIPFIRE_HIPCC_EXTRA_FLAGS` | Parses "HIPFIRE_HIPCC_EXTRA_FLAGS" with fallback defaults | `crates/rdna-compute/src/feature_flags.rs:318` |
| `HIPFIRE_HIP_WAIT` | Runtime variable controlling hip wait in hipfire | `crates/rdna-compute/src/dispatch/mod.rs:902` |
| `HIPFIRE_HOST_PROFILE_BIN` | Runtime variable controlling host profile bin in hipfire | `crates/hipfire-eval/src/lib.rs:1241` |
| `HIPFIRE_HOST_TIMING` | HIPFIRE_HOST_TIMING=1: dump per-cycle host-side wall-clock breakdown | `crates/hipfire-runtime/examples/dflash_spec_demo.rs:1734` |
| `HIPFIRE_JINJA_CHAT` | Enabled when set to 1 | `crates/hipfire-serving-core/src/qwen35_prefill.rs:241` |
| `HIPFIRE_KERNEL_CACHE` | Overrides the default kernel cache root ("~/.hipfire/kernels") | `crates/rdna-compute/src/compiler.rs:242` |
| `HIPFIRE_KLD_DIRECT_F16KV_ATTN` | Used to configure runtime execution by explicitly setting "HIPFIRE_KLD_DIRECT_F16KV_ATTN" | `crates/hipfire-arch-qwen35/src/qwen35.rs:12809` |
| `HIPFIRE_KLD_DIRECT_WMMA_ATTN` | Used to configure runtime execution by explicitly setting "HIPFIRE_KLD_DIRECT_WMMA_ATTN" | `crates/hipfire-arch-qwen35/src/qwen35.rs:12808` |
| `HIPFIRE_KLD_FP32_GQA4_ATTN` | Used to configure runtime execution by explicitly setting "HIPFIRE_KLD_FP32_GQA4_ATTN" | `crates/hipfire-arch-qwen35/src/qwen35.rs:12862` |
| `HIPFIRE_KLD_SCORING_MODE` | Runtime variable controlling kld scoring mode in hipfire | `crates/hipfire-kld/src/config.rs:102` |
| `HIPFIRE_KLD_TOP_K` | Runtime variable controlling kld top k in hipfire | `crates/hipfire-kld/src/config.rs:106` |
| `HIPFIRE_KVARN_ROTATE` | In-place: mq_rotate_x loads each 256-group into registers (ds_swizzle | `crates/hipfire-arch-qwen35/src/qwen35.rs:27025` |
| `HIPFIRE_KVARN_SIM` | Environment toggle value controls runtime behavior | `crates/hipfire-runtime/examples/perplexity.rs:220` |
| `HIPFIRE_KVNOISE` | Environment toggle value controls runtime behavior | `crates/hipfire-train/src/kv_noise.rs:36` |
| `HIPFIRE_KVNOISE_BITS` | Defaults to 4 when unset | `crates/hipfire-train/examples/kvnoise_recovery_supra50m.rs:88` |
| `HIPFIRE_KVNOISE_FOLD` | Defaults to 4 when unset | `crates/hipfire-train/examples/kvnoise_recovery_supra50m.rs:87` |
| `HIPFIRE_KVNOISE_HOT` | Defaults to 4 when unset | `crates/hipfire-train/examples/kvnoise_recovery_supra50m.rs:86` |
| `HIPFIRE_KVNOISE_LR` | Runtime variable controlling KVnoise lr in hipfire | `crates/hipfire-train/examples/kvnoise_recovery_supra50m.rs:41` |
| `HIPFIRE_KV_COLD_BITS` | Cold-tile quant precision probe: code max = 2^bits - 1 (4-bit=15 default, | `crates/hipfire-runtime/src/kv_hier.rs:174` |
| `HIPFIRE_KV_CORE_FRAC` | Runtime variable controlling KV core frac in hipfire | `crates/hipfire-runtime/src/kv_hier.rs:164` |
| `HIPFIRE_KV_FOLD_M` | Cold-tier compaction knobs. fold_m=1 disables the m:1 merge (cold = pure | `crates/hipfire-runtime/src/kv_hier.rs:159` |
| `HIPFIRE_KV_HIERARCHICAL` | (→ qwen35_prefill_active_session → per-token forward_scratch), which honours | `crates/hipfire-serving-core/src/qwen35_prefill.rs:532` |
| `HIPFIRE_KV_HIER_KEEP` | Parsed as value configuration from environment value | `crates/hipfire-runtime/examples/parity_kv_hier.rs:58` |
| `HIPFIRE_KV_HIER_TEST_IDLE` | Optional: exercise the deferred between-turns drain. idle_compact moves hot | `crates/hipfire-runtime/examples/parity_kv_hier.rs:57` |
| `HIPFIRE_KV_HOT_BUDGET` | Runtime variable controlling KV hot budget in hipfire | `crates/hipfire-runtime/src/kv_hier.rs:146` |
| `HIPFIRE_KV_IDLE_KEEP` | Runtime variable controlling KV idle keep in hipfire | `crates/hipfire-serving-core/src/qwen35_prefill.rs:138` |
| `HIPFIRE_KV_IMPORTANCE` | Cold-tile quant precision probe: code max = 2^bits - 1 (4-bit=15 default, | `crates/hipfire-runtime/src/kv_hier.rs:169` |
| `HIPFIRE_KV_MIGRATE_BATCH` | Cold-tier compaction knobs. fold_m=1 disables the m:1 merge (cold = pure | `crates/hipfire-runtime/src/kv_hier.rs:150` |
| `HIPFIRE_KV_MODE` | Runtime variable controlling KV mode in hipfire | `crates/hipfire-serving-core/src/load.rs:2631` |
| `HIPFIRE_KV_PHYSICAL_CAP` | Parses "HIPFIRE_KV_PHYSICAL_CAP" with fallback defaults | `crates/hipfire-serving-core/src/load.rs:552` |
| `HIPFIRE_KV_POS_LOCAL` | Cold-tile quant precision probe: code max = 2^bits - 1 (4-bit=15 default, | `crates/hipfire-runtime/src/kv_hier.rs:171` |
| `HIPFIRE_LFM2_CAPTURE_POSTMIXER` | Runtime variable controlling lfm2 capture postmixer in hipfire | `crates/hipfire-arch-lfm2moe/src/forward.rs:1154` |
| `HIPFIRE_LFM2_DFLASH` | Environment toggle value controls runtime behavior | `crates/hipfire-serving-core/src/load.rs:419` |
| `HIPFIRE_LFM2_DFLASH_F16` | Enabled when set to 1 | `crates/hipfire-arch-lfm2moe/src/dflash.rs:25` |
| `HIPFIRE_LFM2_DFLASH_SYNC_GEMM` | Runtime variable controlling lfm2 dflash sync gemm in hipfire | `crates/hipfire-arch-lfm2moe/src/dflash.rs:31` |
| `HIPFIRE_LFM2_GRAPH` | Interprets "HIPFIRE_LFM2_GRAPH" from environment to select behavior | `crates/hipfire-arch-lfm2moe/src/forward.rs:67` |
| `HIPFIRE_LLOYD_FORCE_BASELINE` | Runtime variable controlling lloyd force baseline in hipfire | `crates/rdna-compute/src/feature_flags.rs:309` |
| `HIPFIRE_LLOYD_GFX12` | Used to configure runtime execution by explicitly setting "HIPFIRE_LLOYD_GFX12" | `crates/hipfire-runtime/examples/prefill_microbench.rs:145` |
| `HIPFIRE_LLOYD_K3` | Fallback to HFQ2-G128 for non-256-aligned (no rotation) | `crates/hipfire-quantize/src/main.rs:8463` |
| `HIPFIRE_LLOYD_MB4` | Force MB4=0 to skip the size-gated routing | `crates/rdna-compute/examples/test_gemm_mq4g256_lloyd_residual_wmma.rs:274` |
| `HIPFIRE_LM_DUMP` | Selects behavior from recognized values | `crates/hipfire-arch-gemma3/src/forward.rs:228` |
| `HIPFIRE_LM_HEAD_F16` | Runtime variable controlling lm head f16 in hipfire | `crates/hipfire-runtime/src/config.rs:108` |
| `HIPFIRE_LM_HEAD_OVERWRITE` | Environment toggle value controls runtime behavior | `crates/rdna-compute/src/feature_flags.rs:222` |
| `HIPFIRE_LM_HEAD_WMMA` | Runtime variable controlling lm head wmma in hipfire | `crates/rdna-compute/src/feature_flags.rs:220` |
| `HIPFIRE_LOAD_TRANSPORT` | Selects behavior from recognized values | `crates/hipfire-runtime/src/weight_pager.rs:768` |
| `HIPFIRE_LOCAL` | Force local-spawn behavior and skip serve HTTP in documented workflows | `README.md:962` |
| `HIPFIRE_MAX_GEN` | Runtime variable controlling max gen in hipfire | `crates/hipfire-runtime/examples/infer_qwen35.rs:295` |
| `HIPFIRE_MAX_SEQ_ALLOW_OVERRIDE` | Interprets "HIPFIRE_MAX_SEQ_ALLOW_OVERRIDE" from environment to select behavior | `crates/hipfire-serving-core/src/load.rs:238` |
| `HIPFIRE_MEMCPY_DUMP` | Enabled when set to 1 | `crates/hip-bridge/src/ffi.rs:988` |
| `HIPFIRE_MEMSET_DUMP` | Enabled when set to 1 | `crates/hip-bridge/src/ffi.rs:1040` |
| `HIPFIRE_MINIMAX_CAPTURE_POSTATTN` | Runtime variable controlling minimax capture postattn in hipfire | `crates/hipfire-arch-minimax/src/forward.rs:131` |
| `HIPFIRE_MINIMAX_DOWN_FORMAT` | Runtime variable controlling minimax down format in hipfire | `crates/hipfire-quantize/src/main.rs:7355` |
| `HIPFIRE_MINIMAX_ENABLE_DOWN_AWQ` | down-AWQ harmful (shared s_down bad approx); opt-in | `crates/hipfire-arch-minimax/src/minimax.rs:447` |
| `HIPFIRE_MINIMAX_EXPERT_MQ2L` | dispatches expert dtype per-layer (experts[0].gpu_dtype), so the model | `crates/hipfire-quantize/src/main.rs:7334` |
| `HIPFIRE_MINIMAX_EXPERT_MQ3L` | dispatches expert dtype per-layer (experts[0].gpu_dtype), so the model | `crates/hipfire-quantize/src/main.rs:7336` |
| `HIPFIRE_MINIMAX_EXPERT_MQ6` | _MQ6 hold comma-separated layer ranges ("12-45,50") whose experts are | `crates/hipfire-quantize/src/main.rs:7332` |
| `HIPFIRE_MMQ` | Selects behavior from recognized values | `crates/rdna-compute/src/feature_flags.rs:194` |
| `HIPFIRE_MMQ_DIAG_QUANTIZE_ONLY` | Runtime variable controlling mmq diag quantize only in hipfire | `crates/rdna-compute/src/feature_flags.rs:233` |
| `HIPFIRE_MMQ_SCREEN` | Runtime variable controlling mmq screen in hipfire | `crates/rdna-compute/src/feature_flags.rs:225` |
| `HIPFIRE_MMQ_SCREEN_THRESHOLD` | Runtime variable controlling mmq screen threshold in hipfire | `crates/rdna-compute/src/feature_flags.rs:229` |
| `HIPFIRE_MODELS_DIR` | Runtime variable controlling models dir in hipfire | `crates/hipfire-eval/src/run.rs:416` |
| `HIPFIRE_MOE_GROUPED_GEMM` | Runtime variable controlling moe grouped gemm in hipfire | `crates/rdna-compute/src/feature_flags.rs:283` |
| `HIPFIRE_MOE_GROUPED_I8` | HIPFIRE_MOE_GROUPED_I8_K8: gfx1151 HFQ4 grouped MoE k8 control; | `crates/rdna-compute/src/feature_flags.rs:253` |
| `HIPFIRE_MOE_GROUPED_I8_4W` | default on for gfx1151, set to 0 to test k4 or base k2 variants | `crates/rdna-compute/src/feature_flags.rs:258` |
| `HIPFIRE_MOE_GROUPED_I8_K4` | HIPFIRE_MOE_GROUPED_I8_K4: opt-in gfx1151 HFQ4 grouped MoE k4 | `crates/rdna-compute/src/feature_flags.rs:269` |
| `HIPFIRE_MOE_GROUPED_I8_K4_GFX12` | HIPFIRE_MOE_HFQ6_V2: opt-in HFQ6 grouped MoE v2 path; on | `crates/rdna-compute/src/feature_flags.rs:270` |
| `HIPFIRE_MOE_GROUPED_I8_K8` | variant; use with HIPFIRE_MOE_GROUPED_I8_K8=0. Default off | `crates/rdna-compute/src/feature_flags.rs:264` |
| `HIPFIRE_MOE_GROUPED_M2` | HIPFIRE_MOE_HFQ6_V2: opt-in HFQ6 grouped MoE v2 path; on | `crates/rdna-compute/src/feature_flags.rs:272` |
| `HIPFIRE_MOE_HFQ6_4W` | Environment toggle value controls runtime behavior | `crates/rdna-compute/src/feature_flags.rs:280` |
| `HIPFIRE_MOE_HFQ6_V2` | HIPFIRE_MOE_HFQ6_V2: opt-in HFQ6 grouped MoE v2 path; on | `crates/rdna-compute/src/feature_flags.rs:276` |
| `HIPFIRE_MOE_INDEXED_2ROW_GFX1151` | Opt-in ("1") gfx1151 two-row indexed MoE HFQ4 decode probe for gate/up and expanded down; default off after flat A3B measurements | `crates/rdna-compute/src/dispatch/mod.rs:3349` |
| `HIPFIRE_MOE_MQ2L_N32_GFX1151` | Runtime variable controlling moe mq2l n32 gfx1151 in hipfire | `crates/hipfire-arch-qwen35/src/qwen35.rs:14847` |
| `HIPFIRE_MOE_PARO_I8` | Runtime variable controlling moe paro i8 in hipfire | `crates/hipfire-arch-qwen35/src/qwen35.rs:16463` |
| `HIPFIRE_MOE_PARO_I8_K8` | Runtime variable controlling moe paro i8 k8 in hipfire | `crates/hipfire-arch-qwen35/src/qwen35.rs:16467` |
| `HIPFIRE_MQ3_MB4` | Used to configure runtime execution by explicitly setting "HIPFIRE_MQ3_MB4" | `crates/rdna-compute/examples/test_gemm_hfq3g256_wmma.rs:92` |
| `HIPFIRE_MTP_DEVICE_TOKEN_CHAIN` | Default on: this path is token-identical in greedy mode and removes the | `crates/hipfire-arch-qwen35/src/mtp_spec.rs:82` |
| `HIPFIRE_MTP_GPU_ACCEPT` | Default on for greedy device-token-chain MTP: candidates and verify | `crates/hipfire-arch-qwen35/src/mtp_spec.rs:107` |
| `HIPFIRE_MTP_HEAD_LMHEAD_WMMA` | HIPFIRE_MTP_HEAD_LMHEAD_WMMA=0. (Ported from origin/master 5ac96a8f.) | `crates/hipfire-arch-qwen35/src/mtp_head.rs:2063` |
| `HIPFIRE_MTP_K` | Runtime variable controlling MTP k in hipfire | `crates/hipfire-serving-core/src/generate_arch.rs:354` |
| `HIPFIRE_MTP_MODE` | Defaults to auto when unset | `crates/hipfire-serving-core/src/generate_arch.rs:345` |
| `HIPFIRE_MTP_PHASE_TIMERS` | Env-gated phase timers (HIPFIRE_MTP_PHASE_TIMERS=1). The existing | `crates/hipfire-arch-qwen35/src/mtp_spec.rs:1348` |
| `HIPFIRE_MTP_PROPOSAL_GRAPH` | Runtime variable controlling MTP proposal graph in hipfire | `crates/hipfire-arch-qwen35/src/mtp_spec.rs:187` |
| `HIPFIRE_MTP_P_MIN` | Runtime variable controlling MTP p min in hipfire | `crates/hipfire-arch-qwen35/src/mtp_spec.rs:147` |
| `HIPFIRE_MTP_Q8_VERIFY_WMMA` | Runtime variable controlling MTP Q8 verify wmma in hipfire | `crates/hipfire-arch-qwen35/src/mtp_spec.rs:131` |
| `HIPFIRE_MTP_SMOKE_HEAD` | Runtime variable controlling MTP smoke head in hipfire | `crates/hipfire-arch-qwen35/examples/mtp_head_smoke.rs:28` |
| `HIPFIRE_MTP_SMOKE_TRUNK` | Runtime variable controlling MTP smoke trunk in hipfire | `crates/hipfire-arch-qwen35/examples/mtp_head_smoke.rs:24` |
| `HIPFIRE_MTP_SNAPSHOT_OVERLAP` | Default off: current gfx1201 benches show this stream split regresses, | `crates/hipfire-arch-qwen35/src/mtp_spec.rs:94` |
| `HIPFIRE_MTP_VERIFY_DECOUPLE` | mq4). Opt-out HIPFIRE_MTP_VERIFY_DECOUPLE=0. Other archs are opt-in (=1) | `crates/hipfire-arch-qwen35/src/qwen35.rs:14705` |
| `HIPFIRE_MW16` | Runtime variable controlling mw16 in hipfire | `crates/rdna-compute/src/feature_flags.rs:294` |
| `HIPFIRE_NEMOTRON_SSM_STATE` | Selects behavior from recognized values | `crates/hipfire-arch-nemotron/src/block_gpu.rs:39` |
| `HIPFIRE_NGRAM_LOOP_THRESHOLD` | Parses "HIPFIRE_NGRAM_LOOP_THRESHOLD" with fallback defaults | `crates/hipfire-runtime/src/config.rs:94` |
| `HIPFIRE_NGRAM_WINDOW` | Runtime variable controlling ngram window in hipfire | `crates/hipfire-runtime/src/config.rs:98` |
| `HIPFIRE_NORMALIZE_PROMPT` | Opt-out must skip CRLF/NBSP/trailing-ws too, not just newline collapse | `crates/hipfire-serving-core/src/output_filter.rs:30` |
| `HIPFIRE_NORM_PLUS1` | qwen3.5 may store RMSNorm weight as (1+γ) (Gemma-style). HIPFIRE_NORM_PLUS1=1 | `crates/hipfire-train/examples/qwen35_mlp_norm_recovery.rs:192` |
| `HIPFIRE_NO_EXPERT_AWQ` | experts fall back to plain MQ4/MQ8) — an A/B knob for measuring the | `crates/hipfire-quantize/src/main.rs:7608` |
| `HIPFIRE_NO_QUANT` | Student weights = OQ+ sim-quant (HIPFIRE_NO_QUANT=1 → identity sanity check) | `crates/hipfire-train/examples/qwen35_mlp_norm_recovery.rs:249` |
| `HIPFIRE_NPU_ATTN_GATE_CONFIGS` | Runtime variable controlling npu attn gate configs in hipfire | `crates/hipfire-arch-qwen35/build.rs:193` |
| `HIPFIRE_NPU_DIR` | Defaults to target/npu when unset | `crates/hipfire-arch-qwen35/src/qwen35.rs:2083` |
| `HIPFIRE_NPU_HEADNORM_CONFIGS` | Parse "n_heads:n_kv_heads:head_dim" tuples (n_rot field ignored if present) | `crates/hipfire-arch-qwen35/build.rs:156` |
| `HIPFIRE_NPU_HEADNORM_ROPE_CONFIGS` | Runtime variable controlling npu headnorm rope configs in hipfire | `crates/hipfire-arch-qwen35/build.rs:277` |
| `HIPFIRE_NPU_HIDDEN_SIZES` | Runtime variable controlling npu hidden sizes in hipfire | `crates/hipfire-arch-qwen35/build.rs:54` |
| `HIPFIRE_NPU_PYTHON` | Runtime variable controlling npu python in hipfire | `crates/hipfire-arch-qwen35/build.rs:600` |
| `HIPFIRE_NPU_RMSNORM_SIZES` | Runtime variable controlling npu rmsnorm sizes in hipfire | `crates/hipfire-arch-qwen35/build.rs:89` |
| `HIPFIRE_NPU_ROPE_CONFIGS` | Parse "n_heads:n_kv_heads:head_dim:n_rot" tuples | `crates/hipfire-arch-qwen35/build.rs:118` |
| `HIPFIRE_NPU_SOFTMAX_CONFIGS` | Parse "n_heads:ctx_len1+ctx_len2+..." entries | `crates/hipfire-arch-qwen35/build.rs:232` |
| `HIPFIRE_NPU_TARGETS` | Runtime variable controlling npu targets in hipfire | `crates/hipfire-arch-qwen35/build.rs:60` |
| `HIPFIRE_OQ4_BATCHED_PREFILL` | Environment toggle value controls runtime behavior | `crates/hipfire-arch-qwen35/src/qwen35.rs:14183` |
| `HIPFIRE_OQ4_PREFILL_ACT_BITS` | Selects behavior from recognized values | `crates/hipfire-runtime/src/weights.rs:1408` |
| `HIPFIRE_OQ4_TRACE` | Runtime variable controlling oQ4 trace in hipfire | `crates/hipfire-runtime/src/weights.rs:646` |
| `HIPFIRE_PAGED_MOE_DEBUG` | Enabled when set to 1 | `crates/hipfire-arch-qwen35/src/qwen35.rs:7861` |
| `HIPFIRE_PARO_BATCHED` | Runtime variable controlling paro batched in hipfire | `crates/hipfire-arch-qwen35/src/qwen35.rs:14775` |
| `HIPFIRE_PARO_FA3_FUSED` | Runtime variable controlling paro fa3 fused in hipfire | `crates/hipfire-arch-qwen35/src/qwen35.rs:29502` |
| `HIPFIRE_PARO_FUSED_PACK2` | Runtime variable controlling paro fused pack2 in hipfire | `crates/rdna-compute/src/dispatch/fused.rs:422` |
| `HIPFIRE_PARO_FUSE_RMSNORM` | time per call. Net loss on every site. Default OFF; explicit opt-in for | `crates/hipfire-runtime/src/weights.rs:759` |
| `HIPFIRE_PARO_GATE_UP_FUSED` | Runtime variable controlling paro gate up fused in hipfire | `crates/hipfire-arch-qwen35/src/qwen35.rs:29100` |
| `HIPFIRE_PARO_LA2_FUSED` | Runtime variable controlling paro la2 fused in hipfire | `crates/hipfire-arch-qwen35/src/qwen35.rs:29202` |
| `HIPFIRE_PARO_LA4_FUSED` | Runtime variable controlling paro la4 fused in hipfire | `crates/hipfire-arch-qwen35/src/qwen35.rs:29198` |
| `HIPFIRE_PARO_LA_GATES_MQ4G128` | Used to configure runtime execution by explicitly setting "HIPFIRE_PARO_LA_GATES_MQ4G128" | `crates/hipfire-arch-qwen35/src/paro_la_gates_codec.rs:325` |
| `HIPFIRE_PARO_PACK1` | Runtime variable controlling paro pack1 in hipfire | `crates/rdna-compute/src/dispatch/gemv.rs:1371` |
| `HIPFIRE_PARO_PACK2` | Runtime variable controlling paro pack2 in hipfire | `crates/rdna-compute/src/dispatch/gemv.rs:1374` |
| `HIPFIRE_PARO_PACK4` | Runtime variable controlling paro pack4 in hipfire | `crates/rdna-compute/src/dispatch/gemv.rs:1377` |
| `HIPFIRE_PARO_PREROTATE` | Runtime variable controlling paro prerotate in hipfire | `crates/hipfire-runtime/src/weights.rs:1331` |
| `HIPFIRE_PARO_SHARED_PAIRS` | Runtime variable controlling paro shared pairs in hipfire | `crates/rdna-compute/src/dispatch/fused.rs:416` |
| `HIPFIRE_PARO_SWIGLU_FUSED` | Runtime variable controlling paro swiglu fused in hipfire | `crates/hipfire-runtime/src/weights.rs:1348` |
| `HIPFIRE_PERF_BASELINE` | Runtime variable controlling perf baseline in hipfire | `crates/hipfire-eval/src/executor_examples.rs:1325` |
| `HIPFIRE_PERF_BASELINE_DIR` | Runtime variable controlling perf baseline dir in hipfire | `crates/hipfire-eval/src/executor_examples.rs:1338` |
| `HIPFIRE_PERPLEXITY_BIN` | Runtime variable controlling perplexity bin in hipfire | `crates/hipfire-eval/src/lib.rs:1226` |
| `HIPFIRE_PFLASH_DAEMON_LABELS` | Runtime variable controlling pflash daemon labels in hipfire | `crates/hipfire-train/examples/ssm_drafter_train.rs:46` |
| `HIPFIRE_PFLASH_DEBUG` | Runtime variable controlling pflash debug in hipfire | `crates/hipfire-serving-core/src/generate.rs:2453` |
| `HIPFIRE_PFLASH_DRAFTER_KV` | Selects behavior from recognized values | `crates/hipfire-arch-qwen35/src/pflash.rs:534` |
| `HIPFIRE_PFLASH_DRAFTER_STATE` | Hybrid drafter only stores K (and V for chat-path) at | `crates/hipfire-arch-qwen35/src/pflash.rs:621` |
| `HIPFIRE_PFLASH_FRESH` | resume: reload weights + AdamW state from the checkpoint unless FRESH=1 | `crates/hipfire-train/examples/pflash_drafter_train.rs:346` |
| `HIPFIRE_PFLASH_NIAH_BENCH_BIN` | Runtime variable controlling pflash niah bench bin in hipfire | `crates/hipfire-eval/src/lib.rs:1181` |
| `HIPFIRE_PFLASH_REPORT_TRAIN` | Runtime variable controlling pflash report train in hipfire | `crates/hipfire-train/examples/ssm_drafter_train.rs:95` |
| `HIPFIRE_PFLASH_SCORE_LAYER` | Parses "HIPFIRE_PFLASH_SCORE_LAYER" with fallback defaults | `crates/hipfire-arch-qwen35/src/pflash.rs:1082` |
| `HIPFIRE_PP_DFLASH` | Environment toggle value controls runtime behavior | `crates/hipfire-daemon/src/main.rs:3023` |
| `HIPFIRE_PP_LAYERS` | Runtime variable controlling pp layers in hipfire | `crates/hipfire-serving-core/src/load.rs:2682` |
| `HIPFIRE_PP_PARITY_MODEL` | Runtime variable controlling pp parity model in hipfire | `crates/hipfire-arch-qwen35/tests/pp_parity.rs:197` |
| `HIPFIRE_PP_PFLASH` | Environment toggle value controls runtime behavior | `crates/hipfire-daemon/src/main.rs:3041` |
| `HIPFIRE_PREFILL_ALPHA` | Runtime variable controlling prefill alpha in hipfire | `crates/hipfire-arch-qwen35/src/pflash.rs:125` |
| `HIPFIRE_PREFILL_BATCHED` | Enabled by default; set to 0 to disable | `crates/hipfire-runtime/src/config.rs:86` |
| `HIPFIRE_PREFILL_BLOCK` | Runtime variable controlling prefill block in hipfire | `crates/hipfire-arch-qwen35/src/pflash.rs:145` |
| `HIPFIRE_PREFILL_COMPRESSION` | Runtime variable controlling prefill compression in hipfire | `crates/hipfire-arch-qwen35/src/pflash.rs:105` |
| `HIPFIRE_PREFILL_DRAFTER` | Runtime variable controlling prefill drafter in hipfire | `crates/hipfire-arch-qwen35/src/pflash.rs:158` |
| `HIPFIRE_PREFILL_KEEP_RATIO` | Runtime variable controlling prefill keep ratio in hipfire | `crates/hipfire-arch-qwen35/src/pflash.rs:115` |
| `HIPFIRE_PREFILL_MAX_BATCH` | Runtime variable controlling prefill max batch in hipfire | `crates/hipfire-serving-core/src/qwen35_prefill.rs:1413` |
| `HIPFIRE_PREFILL_MAX_LAYER` | Runtime variable controlling prefill max layer in hipfire | `crates/hipfire-arch-qwen35/src/qwen35.rs:13543` |
| `HIPFIRE_PREFILL_MIN_KEEP` | Runtime variable controlling prefill min keep in hipfire | `crates/hipfire-arch-qwen35/src/pflash.rs:130` |
| `HIPFIRE_PREFILL_PROFILE` | Enabled when set to 1 | `crates/hipfire-arch-qwen35/src/pflash.rs:155` |
| `HIPFIRE_PREFILL_RECENT` | Runtime variable controlling prefill recent in hipfire | `crates/hipfire-arch-qwen35/src/pflash.rs:140` |
| `HIPFIRE_PREFILL_REUSE_PBS` | Used to configure runtime execution by explicitly setting "HIPFIRE_PREFILL_REUSE_PBS" | `crates/hipfire-runtime/examples/prefill_microbench.rs:126` |
| `HIPFIRE_PREFILL_SINK` | Runtime variable controlling prefill sink in hipfire | `crates/hipfire-arch-qwen35/src/pflash.rs:135` |
| `HIPFIRE_PREFILL_SPARSE_THRESHOLD` | Runtime variable controlling prefill sparse threshold in hipfire | `crates/hipfire-arch-qwen35/src/pflash.rs:150` |
| `HIPFIRE_PREFILL_STOP_AFTER_LA_LAYER` | Parses "HIPFIRE_PREFILL_STOP_AFTER_LA_LAYER" with fallback defaults | `crates/hipfire-arch-qwen35/src/qwen35.rs:16918` |
| `HIPFIRE_PREFILL_STOP_STAGE` | Runtime variable controlling prefill stop stage in hipfire | `crates/hipfire-arch-qwen35/src/qwen35.rs:16924` |
| `HIPFIRE_PREFILL_STOP_STAGE_LAYER` | Parses "HIPFIRE_PREFILL_STOP_STAGE_LAYER" with fallback defaults | `crates/hipfire-arch-qwen35/src/qwen35.rs:16921` |
| `HIPFIRE_PREFILL_THRESHOLD` | Runtime variable controlling prefill threshold in hipfire | `crates/hipfire-arch-qwen35/src/pflash.rs:110` |
| `HIPFIRE_PREFIX_BOUNDARY_CHECKPOINTS` | Interprets "HIPFIRE_PREFIX_BOUNDARY_CHECKPOINTS" from environment to select behavior | `crates/hipfire-serving-core/src/qwen35_prefill.rs:446` |
| `HIPFIRE_PROFILE` | HIPFIRE_PROFILE=1 + HIPFIRE_PROFILE_CYCLES=N: per-kernel profiling | `crates/hipfire-runtime/examples/mtp_only_demo.rs:495` |
| `HIPFIRE_PROFILE_CYCLES` | Runtime variable controlling profile cycles in hipfire | `crates/hipfire-runtime/examples/mtp_only_demo.rs:496` |
| `HIPFIRE_PROFILE_DECODE` | Enabled when set to 1 | `crates/hipfire-runtime/examples/bench_qwen35_speed.rs:559` |
| `HIPFIRE_PROMPT_HEAT_JSON` | Enabled when set to 1 | `crates/hipfire-runtime/src/config.rs:62` |
| `HIPFIRE_PROMPT_HEAT_LIMIT` | Runtime variable controlling prompt heat limit in hipfire | `crates/hipfire-runtime/src/config.rs:63` |
| `HIPFIRE_PROMPT_TOKEN_HEAT` | Enabled when set to 1 | `crates/hipfire-runtime/src/config.rs:60` |
| `HIPFIRE_PYTHON` | Python interpreter used by the no-GPU CI shell gate for Python tooling and tests | `.github/CONTRIBUTING.md:86` |
| `HIPFIRE_Q8_BATCHED_LEGACY` | Environment toggle value controls runtime behavior | `crates/rdna-compute/src/feature_flags.rs:295` |
| `HIPFIRE_Q8_FA_ATTENTION_IGNORE_TREE_BIAS` | Interprets "HIPFIRE_Q8_FA_ATTENTION_IGNORE_TREE_BIAS" from environment to select behavior | `crates/hipfire-arch-qwen35/src/qwen35.rs:12910` |
| `HIPFIRE_Q8_FA_ATTENTION_ROW_LOOP` | Interprets "HIPFIRE_Q8_FA_ATTENTION_ROW_LOOP" from environment to select behavior | `crates/hipfire-arch-qwen35/src/qwen35.rs:12874` |
| `HIPFIRE_Q8_FA_ATTENTION_SCALAR_LOOP` | Interprets "HIPFIRE_Q8_FA_ATTENTION_SCALAR_LOOP" from environment to select behavior | `crates/hipfire-arch-qwen35/src/qwen35.rs:12886` |
| `HIPFIRE_Q8_FA_ATTENTION_SERIAL_KV_LOOP` | Interprets "HIPFIRE_Q8_FA_ATTENTION_SERIAL_KV_LOOP" from environment to select behavior | `crates/hipfire-arch-qwen35/src/qwen35.rs:12898` |
| `HIPFIRE_Q8_GATE_UP_4W` | Disabled when set to 0 | `crates/rdna-compute/examples/test_gemm_q8_gate_up_wmma.rs:130` |
| `HIPFIRE_Q8_GATE_UP_BENCH` | Enabled when set to 1 | `crates/rdna-compute/examples/test_gemm_q8_gate_up_wmma.rs:129` |
| `HIPFIRE_Q8_GDN_VERIFY_PER_TOKEN` | Interprets "HIPFIRE_Q8_GDN_VERIFY_PER_TOKEN" from environment to select behavior | `crates/hipfire-arch-qwen35/src/qwen35.rs:12922` |
| `HIPFIRE_Q8_GDN_VERIFY_SERIAL_FRAMES` | Interprets "HIPFIRE_Q8_GDN_VERIFY_SERIAL_FRAMES" from environment to select behavior | `crates/hipfire-arch-qwen35/src/qwen35.rs:12934` |
| `HIPFIRE_Q8_WMMA_4W` | Environment toggle value controls runtime behavior | `crates/rdna-compute/src/dispatch/gemm_misc.rs:964` |
| `HIPFIRE_Q8_WMMA_X64` | Environment toggle value controls runtime behavior | `crates/rdna-compute/src/dispatch/gemm_misc.rs:778` |
| `HIPFIRE_QA_KV_MODES` | Defaults to q8,asym4,asym3,asym2 when unset | `crates/hipfire-runtime/examples/test_inferenceQA.rs:762` |
| `HIPFIRE_QTIP_EVAL_ST` | Selects behavior from recognized values | `crates/hipfire-quantize/src/qtip.rs:749` |
| `HIPFIRE_QTIP_HESSIAN` | Runtime variable controlling qtip hessian in hipfire | `crates/hipfire-quantize/src/main.rs:5461` |
| `HIPFIRE_QUANTIZE_BIN` | Runtime variable controlling quantize bin in hipfire | `crates/hipfire-eval/src/executor_tinyquant.rs:126` |
| `HIPFIRE_QUANT_DIAG_PATH` | Runtime variable controlling quant diag path in hipfire | `crates/hipfire-quantize/src/main.rs:12197` |
| `HIPFIRE_QUANT_THREADS` | Runtime variable controlling quant threads in hipfire | `crates/hipfire-quantize/src/main.rs:5233` |
| `HIPFIRE_QWEN35_DECODE_BATCH` | Defaults to auto when unset | `crates/hipfire-serving-core/src/qwen35_decode.rs:356` |
| `HIPFIRE_QWEN35_DECODE_BATCH_MAX` | Runtime variable controlling qwen35 decode batch max in hipfire | `crates/hipfire-serving-core/src/qwen35_decode.rs:480` |
| `HIPFIRE_QWEN35_DECODE_INTERNAL_PARITY` | Interprets "HIPFIRE_QWEN35_DECODE_INTERNAL_PARITY" from environment to select behavior | `crates/hipfire-serving-core/src/qwen35_decode.rs:502` |
| `HIPFIRE_QWEN35_DECODE_NATIVE_MULTIROW` | Interprets "HIPFIRE_QWEN35_DECODE_NATIVE_MULTIROW" from environment to select behavior | `crates/hipfire-serving-core/src/qwen35_decode.rs:491` |
| `HIPFIRE_QWEN35_EXPERT_CACHE_MB` | Parses "HIPFIRE_QWEN35_EXPERT_CACHE_MB" with fallback defaults | `crates/hipfire-arch-qwen35/src/qwen35.rs:701` |
| `HIPFIRE_QWEN35_EXPERT_CACHE_TRACE` | Interprets "HIPFIRE_QWEN35_EXPERT_CACHE_TRACE" from environment to select behavior | `crates/hipfire-arch-qwen35/src/qwen35.rs:5601` |
| `HIPFIRE_QWEN35_FFN_BF16` | Selects behavior from recognized values | `crates/hipfire-arch-qwen35/src/ffn_bf16.rs:57` |
| `HIPFIRE_QWEN35_FFN_BF16_LAYER` | Selects behavior from recognized values | `crates/hipfire-arch-qwen35/src/ffn_bf16.rs:71` |
| `HIPFIRE_QWEN35_FFN_BF16_TRACE` | Parses "HIPFIRE_QWEN35_FFN_BF16_TRACE" with fallback defaults | `crates/hipfire-arch-qwen35/src/ffn_bf16.rs:83` |
| `HIPFIRE_QWEN35_FINITE_TRACE` | Runtime variable controlling qwen35 finite trace in hipfire | `crates/hipfire-arch-qwen35/src/qwen35.rs:14237` |
| `HIPFIRE_QWEN35_PAGED_EXPERTS` | Interprets "HIPFIRE_QWEN35_PAGED_EXPERTS" from environment to select behavior | `crates/hipfire-arch-qwen35/src/qwen35.rs:4323` |
| `HIPFIRE_QWEN35_PREFILL_SESSION_BATCH` | Runtime variable controlling qwen35 prefill session batch in hipfire | `crates/hipfire-serving-core/src/qwen35_prefill.rs:1735` |
| `HIPFIRE_QWEN35_ROUTED_ONLY_MOE_FORWARD` | Runtime variable controlling qwen35 routed only moe forward in hipfire | `crates/hipfire-arch-qwen35/src/qwen35.rs:7871` |
| `HIPFIRE_QWEN35_STAGE_SYNC` | Runtime variable controlling qwen35 stage sync in hipfire | `crates/hipfire-arch-qwen35/src/qwen35.rs:14271` |
| `HIPFIRE_QWEN35_STAGE_TRACE` | Runtime variable controlling qwen35 stage trace in hipfire | `crates/hipfire-arch-qwen35/src/qwen35.rs:14265` |
| `HIPFIRE_QWEN35_XDNA1_INSTR` | Runtime variable controlling qwen35 xdna1 instr in hipfire | `crates/hipfire-npu/src/lib.rs:72` |
| `HIPFIRE_QWEN35_XDNA1_XCLBIN` | Runtime variable controlling qwen35 xdna1 xclbin in hipfire | `crates/hipfire-npu/src/lib.rs:69` |
| `HIPFIRE_RDNA2_VARIANT` | Runtime variable controlling rdna2 variant in hipfire | `crates/rdna-compute/src/feature_flags.rs:313` |
| `HIPFIRE_RECOVER_MODE` | HIPFIRE_RECOVER_MODE=lora+norms → LoRA + layernorms (default, more capacity) | `crates/hipfire-train/examples/recovery_generalization_supra50m.rs:138` |
| `HIPFIRE_RECOVER_NOISE` | KVarN+CASK: tail queries read merged cold keys strictly in their past). 0 = | `crates/hipfire-train/examples/recovery_generalization_supra50m.rs:137` |
| `HIPFIRE_REPLAY_GRAPH` | Enabled when set to 1 | `crates/hipfire-arch-qwen35/src/speculative.rs:5089` |
| `HIPFIRE_RESOURCE_LOCK` | HIPFIRE_RESOURCE_LOCK=0 disables daemon startup resource leases | `crates/hipfire-daemon-adapter/src/lib.rs:780` |
| `HIPFIRE_RESOURCE_LOCK_CPU_CORES` | HIPFIRE_RESOURCE_LOCK_CPU_CORES=0,2-4 adds daemon startup leases for CPU cores | `crates/hipfire-daemon-adapter/src/lib.rs:665` |
| `HIPFIRE_RESOURCE_LOCK_DIR` | HIPFIRE_RESOURCE_LOCK_DIR overrides the daemon resource-lock root directory | `crates/hipfire-lock/src/lib.rs:213` |
| `HIPFIRE_RESOURCE_LOCK_NPUS` | HIPFIRE_RESOURCE_LOCK_NPUS=1 leases every detected NPU; comma lists lease explicit NPU IDs | `crates/hipfire-daemon-adapter/src/lib.rs:630` |
| `HIPFIRE_RESOURCE_LOCK_WAIT_MS` | HIPFIRE_RESOURCE_LOCK_WAIT_MS waits for busy daemon resource leases before failing startup | `crates/hipfire-daemon-adapter/src/lib.rs:798` |
| `HIPFIRE_RESPONSES_STATE_MAX` | Runtime variable controlling responses state max in hipfire | `crates/hipfire-server/src/routes/responses.rs:660` |
| `HIPFIRE_ROCBLAS_ALL_ARCHS` | Runtime variable controlling rocblas all archs in hipfire | `crates/rdna-compute/src/feature_flags.rs:303` |
| `HIPFIRE_ROCBLAS_OFF` | Enabled when set to 1 | `crates/rdna-compute/src/feature_flags.rs:305` |
| `HIPFIRE_ROCPROF_BIN` | Runtime variable controlling rocprof bin in hipfire | `crates/hipfire-eval/src/rocprof.rs:275` |
| `HIPFIRE_ROCPROF_CSV` | Runtime variable controlling rocprof csv in hipfire | `crates/hipfire-runtime/examples/bench_qwen35_speed.rs:325` |
| `HIPFIRE_ROPE_INTERLEAVED_LEGACY` | Runtime variable controlling rope interleaved legacy in hipfire | `crates/rdna-compute/src/feature_flags.rs:296` |
| `HIPFIRE_RQ2_BULK_BITS` | Runtime variable controlling rq2 bulk bits in hipfire | `crates/hipfire-quantize/src/main.rs:9221` |
| `HIPFIRE_RQ2_DAMP` | De-risk B: single shared, foldable residual-stream rotation. With | `crates/hipfire-quantize/src/main.rs:9225` |
| `HIPFIRE_RQ2_PROTECT_FRAC` | Runtime variable controlling rq2 protect frac in hipfire | `crates/hipfire-quantize/src/main.rs:9217` |
| `HIPFIRE_RQ2_Q8_EMBED` | Q8 (~20% of params on a tied-embedding 0.8B). With HIPFIRE_RQ2_Q8_EMBED=1, | `crates/hipfire-quantize/src/main.rs:9343` |
| `HIPFIRE_RQ2_SHARE_RESID` | HIPFIRE_RQ2_SHARE_RESID=1, every k==1024 weight (the d_model residual | `crates/hipfire-quantize/src/main.rs:9239` |
| `HIPFIRE_RQ3_BULK_BITS` | Runtime variable controlling rq3 bulk bits in hipfire | `crates/hipfire-quantize/src/main.rs:9380` |
| `HIPFIRE_RQ3_PROTECT_FRAC` | Runtime variable controlling rq3 protect frac in hipfire | `crates/hipfire-quantize/src/main.rs:9376` |
| `HIPFIRE_RQ3_Q8_EMBED` | Iso-bit embed for an honest mq4 comparison (same as roughquant2 de-risk A) | `crates/hipfire-quantize/src/main.rs:9442` |
| `HIPFIRE_RQ4_BULK` | Bulk codec: "mq4" → real mq4 format (fair mq4+protect-vs-mq4 test, set | `crates/hipfire-quantize/src/main.rs:9483` |
| `HIPFIRE_RQ4_BULK_BITS` | Bulk codec: "mq4" → real mq4 format (fair mq4+protect-vs-mq4 test, set | `crates/hipfire-quantize/src/main.rs:9477` |
| `HIPFIRE_RQ4_DUMP_RANK` | HIPFIRE_RQ4_DUMP_RANK=1: print the residual-channel saliency ranking | `crates/hipfire-quantize/src/main.rs:9589` |
| `HIPFIRE_RQ4_INVERT` | HIPFIRE_RQ4_INVERT=1: protect the LOWEST-saliency channels instead of the | `crates/hipfire-quantize/src/main.rs:9607` |
| `HIPFIRE_RQ4_MQ_BITS` | Uniform bulk bit-width for the mq bulk (4=mq4, 5, 6=mq6). protect_frac=0 | `crates/hipfire-quantize/src/main.rs:9493` |
| `HIPFIRE_RQ4_OBS_DAMP` | Uniform bulk bit-width for the mq bulk (4=mq4, 5, 6=mq6). protect_frac=0 | `crates/hipfire-quantize/src/main.rs:9487` |
| `HIPFIRE_RQ4_PROTECT_FRAC` | diag(H) residual-channel energy from true residual readers | `crates/hipfire-quantize/src/main.rs:10063` |
| `HIPFIRE_RQ4_PROTECT_Q8` | on diag(H) alone). diag = E[x²] (activation energy); wnorm = ‖W[:,c]‖² | `crates/hipfire-quantize/src/main.rs:9498` |
| `HIPFIRE_RQ4_Q8_EMBED` | Enabled when set to 1 | `crates/hipfire-quantize/src/main.rs:9849` |
| `HIPFIRE_RQ4_RANDOM_SEED` | Runtime variable controlling rQ4 random seed in hipfire | `crates/hipfire-quantize/src/main.rs:9712` |
| `HIPFIRE_RQ4_SALIENCY` | (weight energy); product = ‖W[:,c]‖²·E[x²] (output-error contribution) | `crates/hipfire-quantize/src/main.rs:9502` |
| `HIPFIRE_RQ4_VOID_ONLY` | Runtime variable controlling rQ4 void only in hipfire | `crates/hipfire-quantize/src/main.rs:9631` |
| `HIPFIRE_RQ_BULK_BITS` | Runtime variable controlling rq bulk bits in hipfire | `crates/hipfire-quantize/src/main.rs:9150` |
| `HIPFIRE_RQ_GROUP` | Runtime variable controlling rq group in hipfire | `crates/hipfire-quantize/src/main.rs:9154` |
| `HIPFIRE_RQ_HAND` | Environment toggle value controls runtime behavior | `crates/hipfire-arch-qwen35/src/qwen35.rs:23346` |
| `HIPFIRE_RQ_PROTECT_FRAC` | Runtime variable controlling rq protect frac in hipfire | `crates/hipfire-quantize/src/main.rs:9146` |
| `HIPFIRE_RUN_EXAMPLE_BIN` | Runtime variable controlling run example bin in hipfire | `crates/hipfire-eval/src/lib.rs:1196` |
| `HIPFIRE_SAMPLE_COMPARE` | Enabled when set to 1 | `crates/hipfire-runtime/examples/infer_qwen35.rs:236` |
| `HIPFIRE_SCORE_TAIL` | HIPFIRE_SCORE_TAIL=N → score only the last N query positions (leak-free for | `crates/hipfire-train/examples/recovery_generalization_supra50m.rs:143` |
| `HIPFIRE_SERVER_RESIDENT_STATE_BUDGET_MB` | Runtime variable controlling server resident state budget mb in hipfire | `crates/hipfire-serving-core/src/session.rs:1411` |
| `HIPFIRE_SMOKE_KV` | Select KV cache quant via HIPFIRE_SMOKE_KV (default q8, matches the | `crates/hipfire-runtime/examples/a3b_smoke_forward.rs:110` |
| `HIPFIRE_SMOKE_KV_SEQ` | production CLI default). asym3/asym4 engage the Givens-rotated 3/4-bit | `crates/hipfire-runtime/examples/a3b_smoke_forward.rs:103` |
| `HIPFIRE_SMOKE_MODE` | raw (default for back-compat): tokenize "Hello" and decode from pos=0 | `crates/hipfire-runtime/examples/a3b_smoke_forward.rs:158` |
| `HIPFIRE_SMOKE_PROMPT` | Defaults to Hello when unset | `crates/hipfire-runtime/examples/a3b_smoke_forward.rs:159` |
| `HIPFIRE_SMOKE_STEPS` | Runtime variable controlling smoke steps in hipfire | `crates/hipfire-runtime/examples/a3b_smoke_forward.rs:39` |
| `HIPFIRE_SPEC_PHASES` | Enabled when set to 1 | `crates/hipfire-arch-qwen35/src/speculative.rs:7090` |
| `HIPFIRE_STATE` | Interprets "HIPFIRE_STATE" from environment to select behavior | `crates/hipfire-runtime/examples/greedy_dump_top5.rs:246` |
| `HIPFIRE_TARGET_ARCH` | Runtime variable controlling target arch in hipfire | `crates/rdna-compute/src/dispatch/mod.rs:931` |
| `HIPFIRE_TIER_RATIO` | Runtime variable controlling tier ratio in hipfire | `crates/hipfire-quantize/src/main.rs:5722` |
| `HIPFIRE_TINYQUANT_FAMILIES` | Optional comma-separated family allowlist ("HIPFIRE_TINYQUANT_FAMILIES") | `crates/hipfire-eval/src/executor_tinyquant.rs:370` |
| `HIPFIRE_TINYQUANT_RECORD` | Optional comma-separated family allowlist ("HIPFIRE_TINYQUANT_FAMILIES") | `crates/hipfire-eval/src/executor_tinyquant.rs:364` |
| `HIPFIRE_TINY_QUANT_PROBE_BIN` | Runtime variable controlling tiny quant probe bin in hipfire | `crates/hipfire-eval/src/executor_tinyquant.rs:142` |
| `HIPFIRE_TINY_SD_HFQ` | Runtime variable controlling tiny sd hfq in hipfire | `crates/hipfire-server/src/routes/sdapi.rs:4290` |
| `HIPFIRE_TP_BENCH_ITERS` | Runtime variable controlling tp bench iters in hipfire | `crates/hip-bridge/examples/rccl_smoke.rs:19` |
| `HIPFIRE_TP_BENCH_N` | Runtime variable controlling tp bench n in hipfire | `crates/hip-bridge/examples/rccl_smoke.rs:15` |
| `HIPFIRE_TP_EXPERT_ASSIGN` | Selects behavior from recognized values | `crates/hipfire-runtime/src/tp_shard.rs:57` |
| `HIPFIRE_TP_USE_RCCL` | Runtime variable controlling tp use rccl in hipfire | `crates/hipfire-runtime/src/config.rs:90` |
| `HIPFIRE_UNIFORM_VRAM_TOLERANCE_GB` | Runtime variable controlling uniform vram tolerance gb in hipfire | `crates/hipfire-runtime/src/config.rs:105` |
| `HIPFIRE_VERIFY_GRAPH` | Runtime variable controlling verify graph in hipfire | `crates/hipfire-arch-qwen35/src/speculative.rs:6552` |
| `HIPFIRE_VERIFY_GRAPH_TIMING` | (HIPFIRE_VERIFY_GRAPH_TIMING=1). Two device-sync points bracket the | `crates/hipfire-arch-qwen35/src/speculative.rs:6567` |
| `HIPFIRE_VERIFY_GRAPH_TREE` | Enabled when set to 1 | `crates/hipfire-arch-qwen35/src/speculative.rs:6542` |
| `HIPFIRE_VISION_CACHE` | Disabled when set to 0 | `crates/hipfire-serving-core/src/generate_vl.rs:1119` |
| `HIPFIRE_VISION_CACHE_DIR` | Runtime variable controlling vision cache dir in hipfire | `crates/hipfire-serving-core/src/generate_vl.rs:1122` |
| `HIPFIRE_VISION_CACHE_MAX_BYTES` | Parses "HIPFIRE_VISION_CACHE_MAX_BYTES" with fallback defaults | `crates/hipfire-serving-core/src/generate_vl.rs:1131` |
| `HIPFIRE_VISION_DUMP` | Selects behavior from recognized values | `crates/hipfire-arch-gemma3-vl/src/forward.rs:44` |
| `HIPFIRE_VISION_PROFILE` | Optional per-category timing (HIPFIRE_VISION_PROFILE=1): device-sync around | `crates/hipfire-arch-gemma3-vl/src/forward.rs:140` |
| `HIPFIRE_VL_DUMP_DIR` | little-endian f32 blobs + JSON sidecars to $HIPFIRE_VL_DUMP_DIR | `crates/hipfire-runtime/examples/infer.rs:149` |
| `HIPFIRE_WARN_GENERIC` | Enabled by default; set to 0 to disable | `crates/rdna-compute/src/generic_warn.rs:53` |
| `HIPFIRE_WO_MMQ` | Enabled when set to 1 | `crates/rdna-compute/src/feature_flags.rs:219` |
| `HIPFIRE_WO_WMMA_VARIANT` | Runtime variable controlling wo wmma variant in hipfire | `crates/rdna-compute/src/feature_flags.rs:300` |
| `HIPFIRE_XDNA1_LIB` | Runtime variable controlling xdna1 lib in hipfire | `crates/hipfire-npu/src/lib.rs:65` |
| `HIP_PATH` | fails with "file not found". Add well-known candidates as -I flags; | `crates/rdna-compute/src/compiler.rs:509` |
| `HIP_VISIBLE_DEVICES` | Used to configure runtime execution by explicitly setting "HIP_VISIBLE_DEVICES" | `crates/hipfire-daemon-adapter/src/lib.rs:1209` |
| `HOME` | Runtime variable controlling home in hipfire | `crates/rdna-compute/src/compiler.rs:227` |
| `HOSTNAME` | Runtime variable controlling hostname in hipfire | `crates/hipfire-tui/src/hipfire/status.rs:130` |
| `HUGGINGFACE_HUB_CACHE` | Runtime variable controlling huggingface hub cache in hipfire | `crates/hipfire-server/src/routes/sdapi.rs:4118` |
| `MAX_TOKENS` | Parses "MAX_TOKENS" with fallback defaults | `crates/hipfire-runtime/examples/greedy_dump.rs:113` |
| `MMQ_TEST_MODE` | Defaults to residual when unset | `crates/rdna-compute/examples/test_gfx906_mmq_correctness.rs:65` |
| `NANO30B_DIR` | Runtime variable controlling nano30b dir in hipfire | `crates/hipfire-arch-nemotron/examples/test_load_nano30b_hfq.rs:49` |
| `NANO4B_DIR` | Runtime variable controlling nano4b dir in hipfire | `crates/hipfire-arch-nemotron/examples/test_model_prefill_hfq_gpu.rs:42` |
| `NEMO_TOKENS` | Selects behavior from recognized values | `crates/hipfire-arch-nemotron/examples/test_load_nano30b_hfq.rs:25` |
| `NO_NGRAM` | Disabled for perf measurement — re-enable after implementing GPU n-gram kernel | `crates/hipfire-runtime/examples/infer_vl.rs:351` |
| `PATH` | Runtime variable controlling path in hipfire | `crates/rdna-compute/src/compiler.rs:204` |
| `PROMPT_MODE` | Defaults to thinking when unset | `crates/hipfire-runtime/examples/greedy_dump_top5.rs:92` |
| `QWEN35_TEST_MODEL` | Runtime variable controlling qwen35 test model in hipfire | `crates/hipfire-runtime/examples/test_qwen35_loadQA.rs:22` |
| `ROCM_DEVICE_LIB_PATH` | Runtime variable controlling rocm device lib path in hipfire | `crates/rdna-compute/src/compiler.rs:497` |
| `ROCM_PATH` | Runtime variable controlling rocm path in hipfire | `crates/rdna-compute/src/compiler.rs:494` |
| `ROCR_VISIBLE_DEVICES` | Runtime variable controlling rocr visible devices in hipfire | `crates/hipfire-daemon-adapter/src/lib.rs:589` |
| `TINYLLAMA_GGUF` | Runtime variable controlling tinyllama gguf in hipfire | `crates/hipfire-runtime/examples/test_q4f16QA.rs:38` |
| `TRIALS` | Runtime variable controlling trials in hipfire | `crates/rdna-compute/examples/bench_gfx1151_hfq4_s4_mmq.rs:151` |
| `USERPROFILE` | Runtime variable controlling userprofile in hipfire | `crates/rdna-compute/src/compiler.rs:228` |
| `USE_SAMPLE` | Enabled when set to 1 | `crates/hipfire-runtime/examples/a3b_multiturn_oneshot.rs:118` |

- Total env vars: **523**
- `HIPFIRE_*` vars: **479**
- non-`HIPFIRE_*` vars: **44**
