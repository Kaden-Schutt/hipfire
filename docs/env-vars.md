# hipfire environment variables — canonical reference

Generated automatically from source and inline comments by `scripts/gen-env-docs.py`.

| Variable | Description | Defined at |
|---|---|---|
| `BENCH_BATCH` | Runtime variable controlling bench batch in hipfire | `/home/sadara/.hipfire/src/crates/rdna-compute/examples/bench_stream_overlap.rs:56` |
| `BENCH_DRAFT_K` | Runtime variable controlling bench draft k in hipfire | `/home/sadara/.hipfire/src/crates/rdna-compute/examples/bench_stream_overlap.rs:188` |
| `BENCH_DRAFT_LAYERS` | Runtime variable controlling bench draft layers in hipfire | `/home/sadara/.hipfire/src/crates/rdna-compute/examples/bench_stream_overlap.rs:215` |
| `BENCH_DRAFT_M` | Runtime variable controlling bench draft m in hipfire | `/home/sadara/.hipfire/src/crates/rdna-compute/examples/bench_stream_overlap.rs:184` |
| `BENCH_DRAFT_N` | Runtime variable controlling bench draft n in hipfire | `/home/sadara/.hipfire/src/crates/rdna-compute/examples/bench_stream_overlap.rs:192` |
| `BENCH_K` | Runtime variable controlling bench k in hipfire | `/home/sadara/.hipfire/src/crates/rdna-compute/examples/bench_stream_overlap.rs:52` |
| `BENCH_M` | Runtime variable controlling bench m in hipfire | `/home/sadara/.hipfire/src/crates/rdna-compute/examples/bench_stream_overlap.rs:48` |
| `BENCH_VERIFY_LAYERS` | Runtime variable controlling bench verify layers in hipfire | `/home/sadara/.hipfire/src/crates/rdna-compute/examples/bench_stream_overlap.rs:219` |
| `CARGO_FEATURE_NPU_KERNELS` | Runtime variable controlling cargo feature npu kernels in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/build.rs:29` |
| `CARGO_MANIFEST_DIR` | CARGO_MANIFEST_DIR = <workspace>/crates/hipfire-arch-qwen35 | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/build.rs:591` |
| `DDTREE_TIMING` | pre_verify / verify. The draft and top-K are fused into one GPU- | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/speculative.rs:11125` |
| `DEBUG_LAYERS` | Runtime variable controlling debug layers in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/qwen35.rs:8352` |
| `DFLASH_LIVE_TAU` | Runtime variable controlling dflash live tau in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/dflash_spec_demo.rs:1512` |
| `FP32_STATE` | Runtime variable controlling fp32 state in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/infer_qwen35.rs:191` |
| `GPU_LOCK_TIMEOUT` | Runtime variable controlling gpu lock timeout in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-cli/src/commands/gpu_lock.rs:83` |
| `GPU_POLL_INTERVAL` | Runtime variable controlling gpu poll interval in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-cli/src/commands/gpu_lock.rs:76` |
| `HFQ_TEST_N_ITER` | Parses "HFQ_TEST_N_ITER" with fallback defaults | `/home/sadara/.hipfire/src/crates/rdna-compute/examples/test_hfq4_residual_dp4a.rs:74` |
| `HFQ_TEST_SCALE_LOG10` | Parses "HFQ_TEST_SCALE_LOG10" with fallback defaults | `/home/sadara/.hipfire/src/crates/rdna-compute/examples/test_hfq4_residual_dp4a.rs:196` |
| `HFQ_TEST_ZP_MAX` | Parses "HFQ_TEST_ZP_MAX" with fallback defaults | `/home/sadara/.hipfire/src/crates/rdna-compute/examples/test_hfq4_residual_dp4a.rs:200` |
| `HIPFIRE_ADAPTIVE_B_DOWN` | Runtime variable controlling adaptive b down in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/dflash_spec_demo.rs:1779` |
| `HIPFIRE_ADAPTIVE_B_UNSAFE` | the user explicitly widens. Opt out via HIPFIRE_ADAPTIVE_B_UNSAFE=1 | `/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/dflash_spec_demo.rs:921` |
| `HIPFIRE_ADAPTIVE_B_UP` | HIPFIRE_ADAPTIVE_B_UP=0.XX / HIPFIRE_ADAPTIVE_B_DOWN=0.XX | `/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/dflash_spec_demo.rs:1775` |
| `HIPFIRE_ALLOW_MIXED_ARCH` | Runtime variable controlling allow mixed arch in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-runtime/src/config.rs:103` |
| `HIPFIRE_ALLOW_MQ2` | Enabled when set to 1 | `/home/sadara/.hipfire/src/crates/hipfire-quantize/src/main.rs:6783` |
| `HIPFIRE_ALLOW_MQ2_LLOYD` | Enabled when set to 1 | `/home/sadara/.hipfire/src/crates/hipfire-quantize/src/main.rs:6818` |
| `HIPFIRE_ALLOW_MQ3_LLOYD` | Enabled when set to 1 | `/home/sadara/.hipfire/src/crates/hipfire-quantize/src/main.rs:6803` |
| `HIPFIRE_ALLOW_MQ4_LLOYD` | Enabled when set to 1 | `/home/sadara/.hipfire/src/crates/hipfire-quantize/src/main.rs:6853` |
| `HIPFIRE_ALLOW_UNIT_IMATRIX` | Environment toggle value controls runtime behavior | `/home/sadara/.hipfire/src/crates/hipfire-quantize/src/main.rs:6564` |
| `HIPFIRE_ATTN_FLASH` | Honors HIPFIRE_ATTN_FLASH=never\\\\|0\\\\|off as an explicit override | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/qwen35.rs:8589` |
| `HIPFIRE_AWQ_F1_ONLY` | F1-vs-F2 A/B gate. When "HIPFIRE_AWQ_F1_ONLY=1" is set, the F2 | `/home/sadara/.hipfire/src/crates/hipfire-quantize/src/main.rs:5394` |
| `HIPFIRE_BASELINE_ARCH` | Runtime variable controlling baseline arch in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/coherence_probe.rs:403` |
| `HIPFIRE_BATCHES_STATE_MAX` | Parses "HIPFIRE_BATCHES_STATE_MAX" with fallback defaults | `/home/sadara/.hipfire/src/crates/hipfire-server/src/routes/batches.rs:687` |
| `HIPFIRE_BENCH_QWEN35_SPEED_BIN` | Runtime variable controlling bench qwen35 speed bin in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-eval/src/lib.rs:909` |
| `HIPFIRE_BF16_DENSE_M128` | Enabled by default; set to 0 to disable | `/home/sadara/.hipfire/src/crates/rdna-compute/src/dispatch.rs:44716` |
| `HIPFIRE_BF16_MOE_M256` | Enabled when set to 1 | `/home/sadara/.hipfire/src/crates/rdna-compute/src/dispatch.rs:22408` |
| `HIPFIRE_BF16_WEIGHTS` | Runtime variable controlling bf16 weights in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/qwen35.rs:220` |
| `HIPFIRE_BLOB_FORCE` | Graph / capture / deterministic | `/home/sadara/.hipfire/src/crates/rdna-compute/src/feature_flags.rs:276` |
| `HIPFIRE_CALIB_F64_AUDIT` | Enabled when set to 1 | `/home/sadara/.hipfire/src/crates/hipfire-runtime/src/calibration.rs:100` |
| `HIPFIRE_CALIB_PROFILE` | Enable with HIPFIRE_CALIB_PROFILE=1; emits to stderr | `/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/triattn_validate.rs:40` |
| `HIPFIRE_CASK_OFF` | HIPFIRE_CASK_OFF=1 is an ops escape hatch: forces no auto-attach | `/home/sadara/.hipfire/src/crates/hipfire-server/src/routes/chat.rs:212` |
| `HIPFIRE_CHATML` | Enabled when set to 1 | `/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/probe_argmax_agreement.rs:58` |
| `HIPFIRE_CHAT_TEMPLATE_FILE` | 1. Env-var override | `/home/sadara/.hipfire/src/crates/hipfire-prompt/src/lib.rs:1933` |
| `HIPFIRE_COLLECT_ARTIFACTS_BIN` | Runtime variable controlling collect artifacts bin in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-eval/src/lib.rs:956` |
| `HIPFIRE_COMP_DUMP` | Stage-bisect dump: HIPFIRE_COMP_DUMP="<pos>,<layer>" prints each | `/home/sadara/.hipfire/src/crates/hipfire-arch-deepseek4/src/forward.rs:643` |
| `HIPFIRE_CONV1D_TREE_GFX1151` | Environment toggle value controls runtime behavior | `/home/sadara/.hipfire/src/crates/rdna-compute/examples/bench_conv1d_tree_gfx1151.rs:22` |
| `HIPFIRE_DAEMON_BIN` | Runtime variable controlling daemon bin in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-daemon-adapter/src/lib.rs:845` |
| `HIPFIRE_DAEMON_RESIDENT_STATE_BUDGET_MB` | Runtime variable controlling daemon resident state budget mb in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-serving-core/src/session.rs:706` |
| `HIPFIRE_DDTREE_BUDGET` | Runtime variable controlling DDTree budget in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-serving-core/src/load.rs:2165` |
| `HIPFIRE_DDTREE_FORCE_SLOW` | HIPFIRE_DDTREE_FORCE_SLOW=1: force the slow (re-verify) path even when | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/speculative.rs:11352` |
| `HIPFIRE_DDTREE_LOGW_CUTOFF` | Adaptive-B usage report — only meaningful when --adaptive-b is on | `/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/dflash_spec_demo.rs:2517` |
| `HIPFIRE_DDTREE_PATH_B_CAPTURE` | Path B slow-path-kill (opt-in, WIP). Replaces the ~40-50 ms full | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/speculative.rs:11391` |
| `HIPFIRE_DDTREE_PATH_C` | Resolve "HIPFIRE_DDTREE_PATH_C" ONCE before the decode loop. The | `/home/sadara/.hipfire/src/crates/hipfire-serving-core/src/generate.rs:846` |
| `HIPFIRE_DDTREE_PATH_C_VERBOSE` | Runtime variable controlling DDTree path c verbose in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/speculative.rs:11916` |
| `HIPFIRE_DDTREE_PATH_C_VERIFY_GRAPH` | Runtime variable controlling DDTree path c verify graph in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/speculative.rs:202` |
| `HIPFIRE_DDTREE_TAPE_DUMP` | Enabled when set to 1 | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/speculative.rs:11371` |
| `HIPFIRE_DDTREE_TOPK` | Runtime variable controlling DDTree topk in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-serving-core/src/load.rs:2237` |
| `HIPFIRE_DDTREE_TREE_LA` | Opt out with HIPFIRE_DDTREE_TREE_LA=0 if a regression is suspected | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/speculative.rs:11233` |
| `HIPFIRE_DEBUG_PREFIX_BOUNDARIES` | Runtime variable controlling debug prefix boundaries in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-serving-core/src/qwen35_prefill.rs:581` |
| `HIPFIRE_DEEPSEEK4_ATTN` | main model's final_norm_and_head head-HC reduction. Without | `/home/sadara/.hipfire/src/crates/hipfire-arch-deepseek4/src/forward.rs:73` |
| `HIPFIRE_DEEPSEEK4_ATTN_DEBUG_BISECT` | DEBUG: in-kernel bisect (HIPFIRE_DEEPSEEK4_ATTN_DEBUG_BISECT=1) | `/home/sadara/.hipfire/src/crates/hipfire-arch-deepseek4/src/forward.rs:6220` |
| `HIPFIRE_DEEPSEEK4_ATTN_PER_POS` | DEBUG: HIPFIRE_DEEPSEEK4_ATTN_PER_POS=1 substitutes a per-position loop | `/home/sadara/.hipfire/src/crates/hipfire-arch-deepseek4/src/forward.rs:6153` |
| `HIPFIRE_DEEPSEEK4_ATTN_TOPK_DIRECT` | Runtime variable controlling deepseek4 attn topk direct in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-arch-deepseek4/src/forward.rs:6502` |
| `HIPFIRE_DEEPSEEK4_ATTN_TWIN` | DEBUG: same-process twin-call test (HIPFIRE_DEEPSEEK4_ATTN_TWIN=1) | `/home/sadara/.hipfire/src/crates/hipfire-arch-deepseek4/src/forward.rs:6198` |
| `HIPFIRE_DEEPSEEK4_BATCH_HEAD` | Opt-out: HIPFIRE_DEEPSEEK4_BATCH_HEAD=0 forces the legacy per-position | `/home/sadara/.hipfire/src/crates/hipfire-arch-deepseek4/src/forward.rs:8162` |
| `HIPFIRE_DEEPSEEK4_CACHE_TRACE` | Runtime variable controlling deepseek4 cache trace in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-serving-core/src/generate_arch.rs:967` |
| `HIPFIRE_DEEPSEEK4_CHAT_RAW` | Enabled when set to 1 | `/home/sadara/.hipfire/src/crates/hipfire-arch-deepseek4/examples/deepseek4_chat.rs:161` |
| `HIPFIRE_DEEPSEEK4_COMP_F16_WMMA` | Opt out via HIPFIRE_DEEPSEEK4_COMP_F16_WMMA=0 | `/home/sadara/.hipfire/src/crates/hipfire-arch-deepseek4/src/forward.rs:6573` |
| `HIPFIRE_DEEPSEEK4_COMP_ROPE_POS` | Runtime variable controlling deepseek4 comp rope pos in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-arch-deepseek4/src/forward.rs:5234` |
| `HIPFIRE_DEEPSEEK4_DSA_WMMA` | Head-batched f16-WMMA gathered DSA attention; f32 fallback on | `/home/sadara/.hipfire/src/crates/hipfire-arch-deepseek4/src/forward.rs:7111` |
| `HIPFIRE_DEEPSEEK4_DUMP_PROMPT` | Runtime variable controlling deepseek4 dump prompt in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-serving-core/src/generate_arch.rs:317` |
| `HIPFIRE_DEEPSEEK4_DUMP_STATE` | Selects behavior from recognized values | `/home/sadara/.hipfire/src/crates/hipfire-arch-deepseek4/src/forward.rs:203` |
| `HIPFIRE_DEEPSEEK4_DUMP_TOPK` | Gate 1 validation (2026-05-22): dump per-layer topk_indices to a | `/home/sadara/.hipfire/src/crates/hipfire-dispatch/src/pipeline/mod.rs:1072` |
| `HIPFIRE_DEEPSEEK4_EXPERT_LAYER_END` | Runtime variable controlling deepseek4 expert layer end in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-arch-deepseek4/src/arch.rs:599` |
| `HIPFIRE_DEEPSEEK4_F32_TRACE` | Runtime variable controlling deepseek4 f32 trace in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-arch-deepseek4/src/forward.rs:231` |
| `HIPFIRE_DEEPSEEK4_FUSED_UNSCATTER_SILU` | Measured perf at PP_BATCH=512 / 2.1k-tok prompt: -0.4% prefill — | `/home/sadara/.hipfire/src/crates/hipfire-dispatch/src/pipeline/mod.rs:1163` |
| `HIPFIRE_DEEPSEEK4_GEN_TOKENS` | Runtime variable controlling deepseek4 gen tokens in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-arch-deepseek4/examples/deepseek4_chat.rs:157` |
| `HIPFIRE_DEEPSEEK4_GRAPH` | Selects behavior from recognized values | `/home/sadara/.hipfire/src/crates/hipfire-arch-deepseek4/src/forward.rs:1564` |
| `HIPFIRE_DEEPSEEK4_HFQ4_WMMA` | HFQ4G256/Raw. WMMA route requires F16 input staging | `/home/sadara/.hipfire/src/crates/hipfire-arch-deepseek4/src/forward.rs:328` |
| `HIPFIRE_DEEPSEEK4_INDEXER_WMMA` | Runtime variable controlling deepseek4 indexer wmma in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-arch-deepseek4/src/forward.rs:6928` |
| `HIPFIRE_DEEPSEEK4_LOAD_MTP` | Runtime variable controlling deepseek4 load MTP in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-arch-deepseek4/src/arch.rs:998` |
| `HIPFIRE_DEEPSEEK4_MAX_COMPRESS_POS` | ratio == 128: identity gather, no indexer. Per-batch n_compressed | `/home/sadara/.hipfire/src/crates/hipfire-arch-deepseek4/src/forward.rs:7010` |
| `HIPFIRE_DEEPSEEK4_MODEL` | Runtime variable controlling deepseek4 model in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-arch-deepseek4/examples/deepseek4_chat.rs:154` |
| `HIPFIRE_DEEPSEEK4_MOE` | Layers 0..num_hash_layers use STATIC tid2eid routing per upstream DeepSeek V4 | `/home/sadara/.hipfire/src/crates/hipfire-arch-deepseek4/src/forward.rs:7454` |
| `HIPFIRE_DEEPSEEK4_MOE_8W` | Lever 3 (HIPFIRE_DEEPSEEK4_MOE_8W=1): 8-warp variant (shares staged X | `/home/sadara/.hipfire/src/crates/hipfire-dispatch/src/pipeline/mod.rs:1112` |
| `HIPFIRE_DEEPSEEK4_MOE_CND` | Lever 2 (HIPFIRE_DEEPSEEK4_MOE_CND=1): cndmask dequant on the n16 4w | `/home/sadara/.hipfire/src/crates/hipfire-dispatch/src/pipeline/mod.rs:1111` |
| `HIPFIRE_DEEPSEEK4_MOE_DETERMINISTIC` | Environment toggle value controls runtime behavior | `/home/sadara/.hipfire/src/crates/hipfire-dispatch/src/pipeline/mod.rs:1264` |
| `HIPFIRE_DEEPSEEK4_MOE_GROUPED` | Environment toggle value controls runtime behavior | `/home/sadara/.hipfire/src/crates/hipfire-dispatch/src/pipeline/mod.rs:1101` |
| `HIPFIRE_DEEPSEEK4_MOE_GROUPED_GATE` | HIPFIRE_DEEPSEEK4_MOE_GROUPED_GATE overrides the threshold for | `/home/sadara/.hipfire/src/crates/hipfire-dispatch/src/pipeline/mod.rs:1096` |
| `HIPFIRE_DEEPSEEK4_MOE_LLOYD_4W` | RECONCILED (#355 + #356): same 4w default + levers as gate_up above | `/home/sadara/.hipfire/src/crates/hipfire-dispatch/src/pipeline/mod.rs:1104` |
| `HIPFIRE_DEEPSEEK4_MOE_MMQLOAD` | n32_env / cnd_env / eightw_env reused from the gate_up block above | `/home/sadara/.hipfire/src/crates/hipfire-dispatch/src/pipeline/mod.rs:1113` |
| `HIPFIRE_DEEPSEEK4_MOE_N32` | Lever 1 (HIPFIRE_DEEPSEEK4_MOE_N32=1): N_TILE=32 tile-pairing on the | `/home/sadara/.hipfire/src/crates/hipfire-dispatch/src/pipeline/mod.rs:1110` |
| `HIPFIRE_DEEPSEEK4_MOE_NOSYNC` | n32_env / cnd_env / eightw_env reused from the gate_up block above | `/home/sadara/.hipfire/src/crates/hipfire-dispatch/src/pipeline/mod.rs:1114` |
| `HIPFIRE_DEEPSEEK4_MTP_ADDON` | Convention 1: append ".mtp-addon.hfq" (legacy) | `/home/sadara/.hipfire/src/crates/hipfire-arch-deepseek4/src/arch.rs:619` |
| `HIPFIRE_DEEPSEEK4_MTP_HEAD_HC` | Runtime variable controlling deepseek4 MTP head hc in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-arch-deepseek4/src/forward.rs:86` |
| `HIPFIRE_DEEPSEEK4_MTP_SKIP_HEAD` | 3. Batched MTP fill — single pass through the MTP layer for all | `/home/sadara/.hipfire/src/crates/hipfire-arch-deepseek4/src/forward.rs:9011` |
| `HIPFIRE_DEEPSEEK4_POST_SCALE` | Runtime variable controlling deepseek4 post scale in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-arch-deepseek4/src/forward.rs:8357` |
| `HIPFIRE_DEEPSEEK4_PP_BATCH` | on 2026-05-26). PP_BATCH sweep on the 2.1k-tok bench (3 trials/cell): | `/home/sadara/.hipfire/src/crates/hipfire-serving-core/src/load.rs:601` |
| `HIPFIRE_DEEPSEEK4_Q8_4W` | Opt out via HIPFIRE_DEEPSEEK4_Q8_4W=0 for diagnosis | `/home/sadara/.hipfire/src/crates/hipfire-arch-deepseek4/src/forward.rs:292` |
| `HIPFIRE_DEEPSEEK4_Q8_WMMA` | bench_q8_wmma_variants. Opt-out via HIPFIRE_DEEPSEEK4_Q8_WMMA=0 | `/home/sadara/.hipfire/src/crates/hipfire-arch-deepseek4/src/forward.rs:246` |
| `HIPFIRE_DEEPSEEK4_ROUTE_SCALE` | Runtime variable controlling deepseek4 route scale in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-arch-deepseek4/src/forward.rs:7474` |
| `HIPFIRE_DEEPSEEK4_SEED` | Runtime variable controlling deepseek4 seed in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-serving-core/src/generate_arch.rs:541` |
| `HIPFIRE_DEEPSEEK4_SPEC_DECODE` | Priority: 1. legacy env var → 2. generic env var → 3. stored config → default | `/home/sadara/.hipfire/src/crates/hipfire-serving-core/src/generate_arch.rs:339` |
| `HIPFIRE_DEEPSEEK4_SPEC_K` | Runtime variable controlling deepseek4 spec k in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-serving-core/src/generate_arch.rs:347` |
| `HIPFIRE_DEEPSEEK4_TEMP` | Runtime variable controlling deepseek4 temp in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-arch-deepseek4/examples/deepseek4_chat.rs:162` |
| `HIPFIRE_DEEPSEEK4_TOP_K` | for local deployment; we honor that as the default. Pure greedy | `/home/sadara/.hipfire/src/crates/hipfire-serving-core/src/generate_arch.rs:537` |
| `HIPFIRE_DEEPSEEK4_UPLOAD_EXPERTS` | without them). Opt out with "HIPFIRE_DEEPSEEK4_UPLOAD_EXPERTS=0" | `/home/sadara/.hipfire/src/crates/hipfire-arch-deepseek4/src/arch.rs:595` |
| `HIPFIRE_DEEPSEEK4_WO_MULTIROW` | Q8_0 contract: plain (non-FWHT) input. Same layout | `/home/sadara/.hipfire/src/crates/hipfire-arch-deepseek4/src/forward.rs:7215` |
| `HIPFIRE_DEEPSEEK4_WO_Q8_WMMA` | Runtime variable controlling deepseek4 wo Q8 wmma in hipfire | `/home/sadara/.hipfire/src/crates/rdna-compute/src/dispatch.rs:49792` |
| `HIPFIRE_DELTANET_STATE` | Interprets "HIPFIRE_DELTANET_STATE" from environment to select behavior | `/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/greedy_dump_top5.rs:245` |
| `HIPFIRE_DETERMINISTIC` | Enabled when set to 1 | `/home/sadara/.hipfire/src/crates/rdna-compute/src/feature_flags.rs:278` |
| `HIPFIRE_DEVICES` | Runtime variable controlling devices in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-runtime/src/config.rs:102` |
| `HIPFIRE_DFLASH_DRAFT` | Runtime variable controlling dflash draft in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-server/src/routes/chat.rs:196` |
| `HIPFIRE_DFLASH_LOOP_BREAK` | Memory: HashSet<u64>, ≤ max_tokens / 12 entries (~125 for max=1500) | `/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/dflash_spec_demo.rs:1433` |
| `HIPFIRE_DFLASH_LOOP_BREAK_MAX_ESCALATIONS` | Runtime variable controlling dflash loop break max escalations in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/dflash_spec_demo.rs:1462` |
| `HIPFIRE_DFLASH_LOOP_BREAK_RECOVERY` | Runtime variable controlling dflash loop break recovery in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/dflash_spec_demo.rs:1457` |
| `HIPFIRE_DFLASH_LOOP_BREAK_RP_MAX` | Runtime variable controlling dflash loop break rp max in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/dflash_spec_demo.rs:1453` |
| `HIPFIRE_DFLASH_LOOP_BREAK_RP_STEP` | Runtime variable controlling dflash loop break rp step in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/dflash_spec_demo.rs:1449` |
| `HIPFIRE_DFLASH_LOOP_BREAK_STOP_AFTER` | Runtime variable controlling dflash loop break stop after in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/dflash_spec_demo.rs:1445` |
| `HIPFIRE_DFLASH_LOOP_BREAK_TEMP` | Runtime variable controlling dflash loop break temp in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/dflash_spec_demo.rs:1441` |
| `HIPFIRE_DFLASH_MODE` | Defaults to off when unset | `/home/sadara/.hipfire/src/crates/hipfire-runtime/src/config.rs:74` |
| `HIPFIRE_DFLASH_MOE_DRAFT_FFN_GRAPH` | Runtime variable controlling dflash moe draft ffn graph in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/speculative.rs:7102` |
| `HIPFIRE_DFLASH_MOE_VERIFY_GRAPH_LMHEAD` | Runtime variable controlling dflash moe verify graph lmhead in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/speculative.rs:6473` |
| `HIPFIRE_DFLASH_NGRAM_BLOCK` | Enabled when set to 1 | `/home/sadara/.hipfire/src/crates/hipfire-server/src/routes/chat.rs:135` |
| `HIPFIRE_DFLASH_Q8_LMHEAD_WMMA` | Selects behavior from recognized values | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/speculative.rs:113` |
| `HIPFIRE_DFLASH_ROLLBACK_COMPARE` | Used to configure runtime execution by explicitly setting "HIPFIRE_DFLASH_ROLLBACK_COMPARE" | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/speculative.rs:12446` |
| `HIPFIRE_DFLASH_ROLLBACK_FA_RAW_ATOL` | Runtime variable controlling dflash rollback fa raw atol in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/speculative.rs:1089` |
| `HIPFIRE_DFLASH_ROLLBACK_LOGIT_COMPARE_STEPS` | Used to configure runtime execution by explicitly setting "HIPFIRE_DFLASH_ROLLBACK_LOGIT_COMPARE_STEPS" | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/speculative.rs:12470` |
| `HIPFIRE_DFLASH_ROLLBACK_PREFIX_VERIFY` | Used to configure runtime execution by explicitly setting "HIPFIRE_DFLASH_ROLLBACK_PREFIX_VERIFY" | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/speculative.rs:12374` |
| `HIPFIRE_DFLASH_ROLLBACK_SERIAL_REPLAY` | Used to configure runtime execution by explicitly setting "HIPFIRE_DFLASH_ROLLBACK_SERIAL_REPLAY" | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/speculative.rs:12342` |
| `HIPFIRE_DFLASH_ROLLBACK_SERIAL_TAPE` | Used to configure runtime execution by explicitly setting "HIPFIRE_DFLASH_ROLLBACK_SERIAL_TAPE" | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/speculative.rs:12422` |
| `HIPFIRE_DFLASH_ROLLBACK_VERIFY_FRAMES` | Used to configure runtime execution by explicitly setting "HIPFIRE_DFLASH_ROLLBACK_VERIFY_FRAMES" | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/speculative.rs:12398` |
| `HIPFIRE_DFLASH_ROLLBACK_X_IN_ATOL` | Parses "HIPFIRE_DFLASH_ROLLBACK_X_IN_ATOL" with fallback defaults | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/speculative.rs:1097` |
| `HIPFIRE_DFLASH_SEED_ORACLE` | Enabled when set to 1 | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/speculative.rs:7742` |
| `HIPFIRE_DFLASH_SERIAL_QKVZA_SELF_COMPARE` | Runtime variable controlling dflash serial qKVza self compare in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/qwen35.rs:13750` |
| `HIPFIRE_DFLASH_SERIAL_TAPE_X_IN_COMPARE` | Runtime variable controlling dflash serial tape x in compare in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/qwen35.rs:13755` |
| `HIPFIRE_DFLASH_SPEC_DEMO_BIN` | Runtime variable controlling dflash spec demo bin in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-eval/src/lib.rs:894` |
| `HIPFIRE_DFLASH_TRACE_EXPECTED_TOKEN` | Runtime variable controlling dflash trace expected token in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/dflash_spec_demo.rs:137` |
| `HIPFIRE_DFLASH_TRACE_POSITION` | Runtime variable controlling dflash trace position in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/dflash_spec_demo.rs:131` |
| `HIPFIRE_DFLASH_TRACE_TOKEN_INDEX` | Runtime variable controlling dflash trace token index in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/dflash_spec_demo.rs:1804` |
| `HIPFIRE_DN_STATE_EF` | Runtime variable controlling dn state ef in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/qwen35.rs:1430` |
| `HIPFIRE_DN_STATE_FP32_BELOW` | Used to configure runtime execution by explicitly setting "HIPFIRE_DN_STATE_FP32_BELOW" | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/qwen35.rs:29509` |
| `HIPFIRE_DOT2_GEMV` | Interprets "HIPFIRE_DOT2_GEMV" from environment to select behavior | `/home/sadara/.hipfire/src/crates/rdna-compute/src/feature_flags.rs:177` |
| `HIPFIRE_DOTS_OCR_BF16_RESIDUAL` | HF cast x to bf16 at vision forward entry (modeling_dots_vision.py | `/home/sadara/.hipfire/src/crates/hipfire-arch-dots-ocr/src/dots_ocr.rs:1119` |
| `HIPFIRE_DOTS_OCR_DUMP_DIR` | HIPFIRE_DOTS_OCR_DUMP_DIR=<path>: dump full per-stage tensor | `/home/sadara/.hipfire/src/crates/hipfire-arch-dots-ocr/src/dots_ocr.rs:1021` |
| `HIPFIRE_DOTS_OCR_TRACE` | HIPFIRE_DOTS_OCR_TRACE=1: sync after every step + print probe so | `/home/sadara/.hipfire/src/crates/hipfire-arch-dots-ocr/src/dots_ocr.rs:1149` |
| `HIPFIRE_DPM_WARMUP_SECS` | Runtime variable controlling dpm warmup secs in hipfire | `/home/sadara/.hipfire/src/crates/rdna-compute/examples/bench_stream_overlap.rs:60` |
| `HIPFIRE_DRAFT_F16` | Enabled by default; set to 0 to disable | `/home/sadara/.hipfire/src/crates/hipfire-runtime/src/config.rs:75` |
| `HIPFIRE_DRAFT_GEMM_DUMP` | Enabled when set to 1 | `/home/sadara/.hipfire/src/crates/hipfire-runtime/src/config.rs:76` |
| `HIPFIRE_DRAFT_SUBPHASE` | Enabled when set to 1 | `/home/sadara/.hipfire/src/crates/hipfire-runtime/src/config.rs:77` |
| `HIPFIRE_DTOH_DUMP` | Enabled when set to 1 | `/home/sadara/.hipfire/src/crates/hip-bridge/src/ffi.rs:938` |
| `HIPFIRE_DUMMY_GENERATE_DELAY_MS` | Parses "HIPFIRE_DUMMY_GENERATE_DELAY_MS" with fallback defaults | `/home/sadara/.hipfire/src/crates/hipfire-serving-core/src/dummy.rs:140` |
| `HIPFIRE_DUMMY_PREFILL_DELAY_MS` | Parses "HIPFIRE_DUMMY_PREFILL_DELAY_MS" with fallback defaults | `/home/sadara/.hipfire/src/crates/hipfire-serving-core/src/dummy.rs:129` |
| `HIPFIRE_DUMP_HIDDEN` | DIAG: dump router logits before softmax (mirrors qwen35 HIPFIRE_DUMP_HIDDEN) | `/home/sadara/.hipfire/src/crates/hipfire-dispatch/src/pipeline/mod.rs:386` |
| `HIPFIRE_DUMP_HIDDEN_POS` | Runtime variable controlling dump hidden pos in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/qwen35.rs:16222` |
| `HIPFIRE_DUMP_REQUEST` | a strace. Off by default — gigantic for typical agent prompts | `/home/sadara/.hipfire/src/crates/hipfire-server/src/routes/chat.rs:85` |
| `HIPFIRE_EMIT_TOKEN_IDS` | or "\" in it would corrupt the line, breaking the client's JSONL | `/home/sadara/.hipfire/src/crates/hipfire-serving-core/src/events.rs:111` |
| `HIPFIRE_EP_DECODE_TIMING` | 2. Per-layer EP program (Attend replicated; Moe all-reduce-EP'd) | `/home/sadara/.hipfire/src/crates/hipfire-arch-minimax/src/forward.rs:1292` |
| `HIPFIRE_EP_DUMP_IDX` | top-k chaos. HIPFIRE_EP_DUMP_IDX=1 to enable | `/home/sadara/.hipfire/src/crates/hipfire-arch-deepseek4/src/forward.rs:2428` |
| `HIPFIRE_EP_DUMP_POS` | Divergence-localization dump: HIPFIRE_EP_DUMP_POS="0,64,...,302" prints a | `/home/sadara/.hipfire/src/crates/hipfire-arch-deepseek4/src/forward.rs:2363` |
| `HIPFIRE_EP_KV_SEQ` | Runtime variable controlling ep KV seq in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/ep_decode_parity.rs:70` |
| `HIPFIRE_EP_PEER_ALLREDUCE` | RCCL with HIPFIRE_EP_PEER_ALLREDUCE=0. The peer temps live in Gpus (shared | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/qwen35.rs:27149` |
| `HIPFIRE_EP_PEER_ALLREDUCE_DECODE` | Environment toggle value controls runtime behavior | `/home/sadara/.hipfire/src/crates/hipfire-runtime/src/ep.rs:135` |
| `HIPFIRE_EP_PREFILL` | Prefill mode: HIPFIRE_EP_PREFILL=batched → WMMA batched prefill EP (E6b) | `/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/ep_decode_parity.rs:97` |
| `HIPFIRE_EP_PREFILL_TIMING` | Gpus::all_reduce_sum_f32_peer (direct P2P copy + local add), which is ~1 ms | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/qwen35.rs:27142` |
| `HIPFIRE_EP_SKIP_ALLREDUCE` | Gpus::all_reduce_sum_f32_peer (direct P2P copy + local add), which is ~1 ms | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/qwen35.rs:27143` |
| `HIPFIRE_EVAL_DATASET_MIRROR` | Runtime variable controlling eval dataset mirror in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-eval/src/datasets.rs:203` |
| `HIPFIRE_EVAL_EVIDENCE_DIR` | Runtime variable controlling eval evidence dir in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/run.rs:231` |
| `HIPFIRE_EVAL_HIPFIRE_BIN` | Runtime variable controlling eval hipfire bin in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-eval/src/lib.rs:986` |
| `HIPFIRE_EVAL_KLDREF` | Runtime variable controlling eval kldref in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-eval/src/executor_examples.rs:512` |
| `HIPFIRE_EVAL_PERPLEXITY_CORPUS` | Runtime variable controlling eval perplexity corpus in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-eval/src/run.rs:175` |
| `HIPFIRE_EVAL_PERPLEXITY_CTX` | Parses "HIPFIRE_EVAL_PERPLEXITY_CTX" with fallback defaults | `/home/sadara/.hipfire/src/crates/hipfire-eval/src/executor_examples.rs:598` |
| `HIPFIRE_EXPERIMENTAL_BUDGET_ALERT` | cache so the model "sees" them as part of its own trajectory, | `/home/sadara/.hipfire/src/crates/hipfire-daemon/src/main.rs:3251` |
| `HIPFIRE_FILES_STATE_MAX` | Parses "HIPFIRE_FILES_STATE_MAX" with fallback defaults | `/home/sadara/.hipfire/src/crates/hipfire-server/src/routes/files.rs:174` |
| `HIPFIRE_FLASH_PARTIALS_BATCH` | Parses "HIPFIRE_FLASH_PARTIALS_BATCH" with fallback defaults | `/home/sadara/.hipfire/src/crates/hipfire-runtime/src/config.rs:87` |
| `HIPFIRE_FORCE_UNFUSED` | Interpreter Phase 2a | `/home/sadara/.hipfire/src/crates/rdna-compute/src/feature_flags.rs:306` |
| `HIPFIRE_FORWARD_LOWERED` | Enabled by default; set to 0 to disable | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/qwen35.rs:26812` |
| `HIPFIRE_FP16` | escape hatch to the LA qkvza projection while debugging DFlash | `/home/sadara/.hipfire/src/crates/rdna-compute/src/feature_flags.rs:185` |
| `HIPFIRE_FP8_WMMA` | Interprets "HIPFIRE_FP8_WMMA" from environment to select behavior | `/home/sadara/.hipfire/src/crates/rdna-compute/src/feature_flags.rs:176` |
| `HIPFIRE_FUSED_HFQ4_2ROW_GFX1151` | Selects behavior from recognized values | `/home/sadara/.hipfire/src/crates/rdna-compute/src/dispatch.rs:24691` |
| `HIPFIRE_GATE_UP_VARIANT` | Runtime variable controlling gate up variant in hipfire | `/home/sadara/.hipfire/src/crates/rdna-compute/src/feature_flags.rs:231` |
| `HIPFIRE_GDN_Q8_REG_GFX1151` | HIPFIRE_GDN_Q8_REG_GFX1151=1 enables the gfx1151 register-state | `/home/sadara/.hipfire/src/crates/rdna-compute/src/dispatch.rs:39349` |
| `HIPFIRE_GEMM_DUMP` | Enabled when set to 1 | `/home/sadara/.hipfire/src/crates/rdna-compute/src/feature_flags.rs:277` |
| `HIPFIRE_GEMV_ROWS` | Runtime variable controlling gemv rows in hipfire | `/home/sadara/.hipfire/src/crates/rdna-compute/src/feature_flags.rs:157` |
| `HIPFIRE_GEN` | Runtime variable controlling gen in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/a3b_multiturn_oneshot.rs:25` |
| `HIPFIRE_GENERATE_BATCH_PREFILL_DEBUG_SAMPLE` | Runtime variable controlling generate batch prefill debug sample in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-serving-core/src/qwen35_prefill.rs:1551` |
| `HIPFIRE_GFX942_GEMV_V3` | Interprets "HIPFIRE_GFX942_GEMV_V3" from environment to select behavior | `/home/sadara/.hipfire/src/crates/rdna-compute/src/feature_flags.rs:233` |
| `HIPFIRE_GFX942_MFMA_PREFILL` | Interprets "HIPFIRE_GFX942_MFMA_PREFILL" from environment to select behavior | `/home/sadara/.hipfire/src/crates/rdna-compute/src/feature_flags.rs:236` |
| `HIPFIRE_GFX942_RMSNORM_SPLIT` | Environment toggle value controls runtime behavior | `/home/sadara/.hipfire/src/crates/rdna-compute/src/feature_flags.rs:235` |
| `HIPFIRE_GPTQ_DAMPING` | Inject env override since the quantizer reads it at fn entry | `/home/sadara/.hipfire/src/crates/hipfire-quantize/src/main.rs:10933` |
| `HIPFIRE_GPU_LOCKFILE` | Runtime variable controlling gpu lockfile in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-lock/src/lib.rs:200` |
| `HIPFIRE_GPU_SLAB_LOAD` | Runtime variable controlling gpu slab load in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/qwen35.rs:4904` |
| `HIPFIRE_GPU_SLAB_MIB` | Parses "HIPFIRE_GPU_SLAB_MIB" with fallback defaults | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/qwen35.rs:3953` |
| `HIPFIRE_GPU_TOPK` | HIPFIRE_GPU_TOPK=1 enables the GPU topk_logits_f32 kernel + CPU | `/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/infer_qwen35.rs:221` |
| `HIPFIRE_GQA_CHUNK` | Runtime variable controlling gqa chunk in hipfire | `/home/sadara/.hipfire/src/crates/rdna-compute/src/dispatch.rs:31901` |
| `HIPFIRE_GQA_FUSED` | Fused variant (opt-in via HIPFIRE_GQA_FUSED=1): single launch | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen2/src/qwen2.rs:1720` |
| `HIPFIRE_GRAPH` | Used to configure runtime execution by explicitly setting "HIPFIRE_GRAPH" | `/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/prefill_microbench.rs:124` |
| `HIPFIRE_GRAPH_MOE` | - gfx11 (RDNA3 / 3.5): default-ON. +0.6-0.7% decode on 9B and | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/qwen35.rs:8838` |
| `HIPFIRE_GRAPH_PREFILL` | HIPFIRE_GRAPH_PREFILL=1: route the timed prefill loop through | `/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/bench_qwen35_speed.rs:186` |
| `HIPFIRE_HAVE_2_GPU` | Environment toggle value controls runtime behavior | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/tests/pp_parity.rs:193` |
| `HIPFIRE_HETERO_DIFF` | above (#352's GPU greedy-accept path doesn't materialize it), | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/mtp_spec.rs:2981` |
| `HIPFIRE_HFQ4G128_MMQ` | Environment toggle value controls runtime behavior | `/home/sadara/.hipfire/src/crates/rdna-compute/src/feature_flags.rs:226` |
| `HIPFIRE_HFQ4G256_MMQ_GFX1151` | Selects behavior from recognized values | `/home/sadara/.hipfire/src/crates/rdna-compute/src/dispatch.rs:24674` |
| `HIPFIRE_HFQ4_GATE_UP_FAST` | HIPFIRE_HFQ4_GATE_UP_FAST=0 narrows the escape hatch to the | `/home/sadara/.hipfire/src/crates/rdna-compute/src/feature_flags.rs:199` |
| `HIPFIRE_HFQ4_MMQ_GFX906_Y64` | Runtime variable controlling hfQ4 mmq gfx906 y64 in hipfire | `/home/sadara/.hipfire/src/crates/rdna-compute/src/feature_flags.rs:229` |
| `HIPFIRE_HFQ4_QKVZA_FAST` | HIPFIRE_HFQ4_QKVZA_FAST=0 narrows the FP16/WMMA/dot2 prefill | `/home/sadara/.hipfire/src/crates/rdna-compute/src/feature_flags.rs:191` |
| `HIPFIRE_HFQ4_QKV_FAST` | HIPFIRE_HFQ4_QKV_FAST=0 narrows the escape hatch to the | `/home/sadara/.hipfire/src/crates/rdna-compute/src/feature_flags.rs:195` |
| `HIPFIRE_HFQ4_RESIDUAL_FAST` | HIPFIRE_HFQ4_RESIDUAL_FAST=0 narrows the residual-projection | `/home/sadara/.hipfire/src/crates/rdna-compute/src/feature_flags.rs:203` |
| `HIPFIRE_HFQ6_QKVZA_4W` | Interprets "HIPFIRE_HFQ6_QKVZA_4W" from environment to select behavior | `/home/sadara/.hipfire/src/crates/rdna-compute/src/dispatch.rs:26782` |
| `HIPFIRE_HFQ6_QKV_4W` | Interprets "HIPFIRE_HFQ6_QKV_4W" from environment to select behavior | `/home/sadara/.hipfire/src/crates/rdna-compute/src/dispatch.rs:27535` |
| `HIPFIRE_HFQ6_RESIDUAL_4W` | Interprets "HIPFIRE_HFQ6_RESIDUAL_4W" from environment to select behavior | `/home/sadara/.hipfire/src/crates/rdna-compute/src/dispatch.rs:26389` |
| `HIPFIRE_HIPCC_EXTRA_FLAGS` | Parses "HIPFIRE_HIPCC_EXTRA_FLAGS" with fallback defaults | `/home/sadara/.hipfire/src/crates/rdna-compute/src/feature_flags.rs:303` |
| `HIPFIRE_HIP_WAIT` | Runtime variable controlling hip wait in hipfire | `/home/sadara/.hipfire/src/crates/rdna-compute/src/dispatch.rs:754` |
| `HIPFIRE_HOST_PROFILE_BIN` | Runtime variable controlling host profile bin in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-eval/src/lib.rs:1001` |
| `HIPFIRE_HOST_TIMING` | HIPFIRE_HOST_TIMING=1: dump per-cycle host-side wall-clock breakdown | `/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/dflash_spec_demo.rs:1732` |
| `HIPFIRE_JINJA_CHAT` | Enabled when set to 1 | `/home/sadara/.hipfire/src/crates/hipfire-serving-core/src/qwen35_prefill.rs:208` |
| `HIPFIRE_KERNEL_CACHE` | HIPFIRE_KERNEL_CACHE=/tmp/hipfire_kernels if tmpfs speed matters | `/home/sadara/.hipfire/src/crates/rdna-compute/src/compiler.rs:213` |
| `HIPFIRE_KLD_DIRECT_F16KV_ATTN` | Used to configure runtime execution by explicitly setting "HIPFIRE_KLD_DIRECT_F16KV_ATTN" | `/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/build_kld_ref_hipfire.rs:699` |
| `HIPFIRE_KLD_DIRECT_WMMA_ATTN` | Used to configure runtime execution by explicitly setting "HIPFIRE_KLD_DIRECT_WMMA_ATTN" | `/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/build_kld_ref_hipfire.rs:698` |
| `HIPFIRE_KLD_FP32_GQA4_ATTN` | Used to configure runtime execution by explicitly setting "HIPFIRE_KLD_FP32_GQA4_ATTN" | `/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/build_kld_ref_hipfire.rs:702` |
| `HIPFIRE_KLD_GPU_TOPK` | Enabled by default; set to 0 to disable | `/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/build_kld_ref_hipfire.rs:956` |
| `HIPFIRE_KLD_GRAPH` | Enabled by default; set to 0 to disable | `/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/build_kld_ref_hipfire.rs:677` |
| `HIPFIRE_KLD_NO_ACTIVE_STREAM` | Runtime variable controlling kld no active stream in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/build_kld_ref_hipfire.rs:291` |
| `HIPFIRE_KLD_PREFILL_ONLY` | Enabled when set to 1 | `/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/build_kld_ref_hipfire.rs:844` |
| `HIPFIRE_KLD_SOURCE_SHA256` | Enabled when set to 1 | `/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/build_kld_ref_hipfire.rs:840` |
| `HIPFIRE_KVARN_SIM` | Environment toggle value controls runtime behavior | `/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/perplexity.rs:210` |
| `HIPFIRE_KV_MODE` | Runtime variable controlling KV mode in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-serving-core/src/load.rs:1671` |
| `HIPFIRE_KV_PHYSICAL_CAP` | Parses "HIPFIRE_KV_PHYSICAL_CAP" with fallback defaults | `/home/sadara/.hipfire/src/crates/hipfire-serving-core/src/load.rs:370` |
| `HIPFIRE_LFM2_CAPTURE_POSTMIXER` | Runtime variable controlling lfm2 capture postmixer in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-arch-lfm2moe/src/forward.rs:140` |
| `HIPFIRE_LFM2_EXPERT_MQ6` | opt-in via HIPFIRE_LFM2_EXPERT_MQ6 for higher quality), else mq4 | `/home/sadara/.hipfire/src/crates/hipfire-quantize/src/main.rs:7828` |
| `HIPFIRE_LFM2_GRAPH` | Interprets "HIPFIRE_LFM2_GRAPH" from environment to select behavior | `/home/sadara/.hipfire/src/crates/hipfire-arch-lfm2moe/src/forward.rs:65` |
| `HIPFIRE_LFM2_PROJ_MQ4` | Runtime variable controlling lfm2 proj mQ4 in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-quantize/src/main.rs:7909` |
| `HIPFIRE_LFM2_PROJ_MQ6` | MQ6 variant (HIPFIRE_LFM2_PROJ_MQ6=1) is the lower-quality-loss | `/home/sadara/.hipfire/src/crates/hipfire-quantize/src/main.rs:7908` |
| `HIPFIRE_LLOYD_FORCE_BASELINE` | Runtime variable controlling lloyd force baseline in hipfire | `/home/sadara/.hipfire/src/crates/rdna-compute/src/feature_flags.rs:294` |
| `HIPFIRE_LLOYD_GFX12` | Used to configure runtime execution by explicitly setting "HIPFIRE_LLOYD_GFX12" | `/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/prefill_microbench.rs:145` |
| `HIPFIRE_LLOYD_K3` | Fallback to HFQ2-G128 for non-256-aligned (no rotation) | `/home/sadara/.hipfire/src/crates/hipfire-quantize/src/main.rs:9206` |
| `HIPFIRE_LLOYD_MB4` | Force MB4=0 to skip the size-gated routing | `/home/sadara/.hipfire/src/crates/rdna-compute/examples/test_gemm_mq4g256_lloyd_residual_wmma.rs:274` |
| `HIPFIRE_LM_HEAD_F16` | Runtime variable controlling lm head f16 in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-runtime/src/config.rs:108` |
| `HIPFIRE_LM_HEAD_OVERWRITE` | Environment toggle value controls runtime behavior | `/home/sadara/.hipfire/src/crates/rdna-compute/src/feature_flags.rs:207` |
| `HIPFIRE_LM_HEAD_WMMA` | Runtime variable controlling lm head wmma in hipfire | `/home/sadara/.hipfire/src/crates/rdna-compute/src/feature_flags.rs:205` |
| `HIPFIRE_LOAD_TRANSPORT` | Selects behavior from recognized values | `/home/sadara/.hipfire/src/crates/hipfire-runtime/src/weight_pager.rs:768` |
| `HIPFIRE_MEMSET_DUMP` | Enabled when set to 1 | `/home/sadara/.hipfire/src/crates/hip-bridge/src/ffi.rs:1012` |
| `HIPFIRE_MINIMAX_CAPTURE_POSTATTN` | Runtime variable controlling minimax capture postattn in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-arch-minimax/src/forward.rs:131` |
| `HIPFIRE_MINIMAX_DOWN_FORMAT` | Runtime variable controlling minimax down format in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-quantize/src/main.rs:8147` |
| `HIPFIRE_MINIMAX_ENABLE_DOWN_AWQ` | down-AWQ harmful (shared s_down bad approx); opt-in | `/home/sadara/.hipfire/src/crates/hipfire-arch-minimax/src/minimax.rs:445` |
| `HIPFIRE_MINIMAX_EXPERT_MQ2L` | dispatches expert dtype per-layer (experts[0].gpu_dtype), so the model | `/home/sadara/.hipfire/src/crates/hipfire-quantize/src/main.rs:8126` |
| `HIPFIRE_MINIMAX_EXPERT_MQ3L` | dispatches expert dtype per-layer (experts[0].gpu_dtype), so the model | `/home/sadara/.hipfire/src/crates/hipfire-quantize/src/main.rs:8128` |
| `HIPFIRE_MINIMAX_EXPERT_MQ6` | _MQ6 hold comma-separated layer ranges ("12-45,50") whose experts are | `/home/sadara/.hipfire/src/crates/hipfire-quantize/src/main.rs:8124` |
| `HIPFIRE_MMQ` | Selects behavior from recognized values | `/home/sadara/.hipfire/src/crates/rdna-compute/src/feature_flags.rs:179` |
| `HIPFIRE_MMQ_DIAG_QUANTIZE_ONLY` | Runtime variable controlling mmq diag quantize only in hipfire | `/home/sadara/.hipfire/src/crates/rdna-compute/src/feature_flags.rs:218` |
| `HIPFIRE_MMQ_SCREEN` | Runtime variable controlling mmq screen in hipfire | `/home/sadara/.hipfire/src/crates/rdna-compute/src/feature_flags.rs:210` |
| `HIPFIRE_MMQ_SCREEN_THRESHOLD` | Runtime variable controlling mmq screen threshold in hipfire | `/home/sadara/.hipfire/src/crates/rdna-compute/src/feature_flags.rs:214` |
| `HIPFIRE_MODELS_DIR` | Runtime variable controlling models dir in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-eval/src/run.rs:416` |
| `HIPFIRE_MOE_GROUPED_GEMM` | Runtime variable controlling moe grouped gemm in hipfire | `/home/sadara/.hipfire/src/crates/rdna-compute/src/feature_flags.rs:268` |
| `HIPFIRE_MOE_GROUPED_I8` | HIPFIRE_MOE_GROUPED_I8_K8: gfx1151 HFQ4 grouped MoE k8 control; | `/home/sadara/.hipfire/src/crates/rdna-compute/src/feature_flags.rs:238` |
| `HIPFIRE_MOE_GROUPED_I8_4W` | default on for gfx1151, set to 0 to test k4 or base k2 variants | `/home/sadara/.hipfire/src/crates/rdna-compute/src/feature_flags.rs:243` |
| `HIPFIRE_MOE_GROUPED_I8_K4` | HIPFIRE_MOE_GROUPED_I8_K4: opt-in gfx1151 HFQ4 grouped MoE k4 | `/home/sadara/.hipfire/src/crates/rdna-compute/src/feature_flags.rs:254` |
| `HIPFIRE_MOE_GROUPED_I8_K4_GFX12` | HIPFIRE_MOE_HFQ6_V2: opt-in HFQ6 grouped MoE v2 path; on | `/home/sadara/.hipfire/src/crates/rdna-compute/src/feature_flags.rs:255` |
| `HIPFIRE_MOE_GROUPED_I8_K8` | variant; use with HIPFIRE_MOE_GROUPED_I8_K8=0. Default off | `/home/sadara/.hipfire/src/crates/rdna-compute/src/feature_flags.rs:249` |
| `HIPFIRE_MOE_GROUPED_M2` | HIPFIRE_MOE_HFQ6_V2: opt-in HFQ6 grouped MoE v2 path; on | `/home/sadara/.hipfire/src/crates/rdna-compute/src/feature_flags.rs:257` |
| `HIPFIRE_MOE_HFQ6_4W` | Environment toggle value controls runtime behavior | `/home/sadara/.hipfire/src/crates/rdna-compute/src/feature_flags.rs:265` |
| `HIPFIRE_MOE_HFQ6_V2` | HIPFIRE_MOE_HFQ6_V2: opt-in HFQ6 grouped MoE v2 path; on | `/home/sadara/.hipfire/src/crates/rdna-compute/src/feature_flags.rs:261` |
| `HIPFIRE_MOE_INDEXED_2ROW_GFX1151` | Opt-in ("1") gfx1151 two-row indexed MoE HFQ4 decode probe for gate/up and expanded down; default off after flat A3B measurements | `/home/sadara/.hipfire/src/crates/rdna-compute/src/dispatch.rs:24709` |
| `HIPFIRE_MOE_MQ2L_N32_GFX1151` | Runtime variable controlling moe mq2l n32 gfx1151 in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/qwen35.rs:14298` |
| `HIPFIRE_MOE_PARO_I8` | Runtime variable controlling moe paro i8 in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/qwen35.rs:15914` |
| `HIPFIRE_MOE_PARO_I8_K8` | Runtime variable controlling moe paro i8 k8 in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/qwen35.rs:15918` |
| `HIPFIRE_MQ3_MB4` | Used to configure runtime execution by explicitly setting "HIPFIRE_MQ3_MB4" | `/home/sadara/.hipfire/src/crates/rdna-compute/examples/test_gemm_hfq3g256_wmma.rs:92` |
| `HIPFIRE_MTP_DEVICE_TOKEN_CHAIN` | Default on: this path is token-identical in greedy mode and removes the | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/mtp_spec.rs:82` |
| `HIPFIRE_MTP_GPU_ACCEPT` | Default on for greedy device-token-chain MTP: candidates and verify | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/mtp_spec.rs:107` |
| `HIPFIRE_MTP_K` | Runtime variable controlling MTP k in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-serving-core/src/generate_arch.rs:351` |
| `HIPFIRE_MTP_MODE` | Defaults to auto when unset | `/home/sadara/.hipfire/src/crates/hipfire-serving-core/src/generate_arch.rs:342` |
| `HIPFIRE_MTP_PHASE_TIMERS` | Env-gated phase timers (HIPFIRE_MTP_PHASE_TIMERS=1). The existing | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/mtp_spec.rs:1321` |
| `HIPFIRE_MTP_PROPOSAL_GRAPH` | Runtime variable controlling MTP proposal graph in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/mtp_spec.rs:160` |
| `HIPFIRE_MTP_Q8_VERIFY_WMMA` | Runtime variable controlling MTP Q8 verify wmma in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/mtp_spec.rs:131` |
| `HIPFIRE_MTP_SMOKE_HEAD` | Runtime variable controlling MTP smoke head in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/examples/mtp_head_smoke.rs:28` |
| `HIPFIRE_MTP_SMOKE_TRUNK` | Runtime variable controlling MTP smoke trunk in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/examples/mtp_head_smoke.rs:24` |
| `HIPFIRE_MTP_SNAPSHOT_OVERLAP` | Default off: current gfx1201 benches show this stream split regresses, | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/mtp_spec.rs:94` |
| `HIPFIRE_MW16` | Runtime variable controlling mw16 in hipfire | `/home/sadara/.hipfire/src/crates/rdna-compute/src/feature_flags.rs:279` |
| `HIPFIRE_NGRAM_LOOP_THRESHOLD` | Parses "HIPFIRE_NGRAM_LOOP_THRESHOLD" with fallback defaults | `/home/sadara/.hipfire/src/crates/hipfire-runtime/src/config.rs:94` |
| `HIPFIRE_NGRAM_WINDOW` | Runtime variable controlling ngram window in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-runtime/src/config.rs:98` |
| `HIPFIRE_NORMALIZE_PROMPT` | Opt-out must skip CRLF/NBSP/trailing-ws too, not just newline collapse | `/home/sadara/.hipfire/src/crates/hipfire-serving-core/src/output_filter.rs:30` |
| `HIPFIRE_NPU_ATTN_GATE_CONFIGS` | Runtime variable controlling npu attn gate configs in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/build.rs:193` |
| `HIPFIRE_NPU_DIR` | Defaults to target/npu when unset | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/qwen35.rs:2053` |
| `HIPFIRE_NPU_HEADNORM_CONFIGS` | Parse "n_heads:n_kv_heads:head_dim" tuples (n_rot field ignored if present) | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/build.rs:156` |
| `HIPFIRE_NPU_HEADNORM_ROPE_CONFIGS` | Runtime variable controlling npu headnorm rope configs in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/build.rs:277` |
| `HIPFIRE_NPU_HIDDEN_SIZES` | Runtime variable controlling npu hidden sizes in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/build.rs:54` |
| `HIPFIRE_NPU_PYTHON` | Runtime variable controlling npu python in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/build.rs:600` |
| `HIPFIRE_NPU_RMSNORM_SIZES` | Runtime variable controlling npu rmsnorm sizes in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/build.rs:89` |
| `HIPFIRE_NPU_ROPE_CONFIGS` | Parse "n_heads:n_kv_heads:head_dim:n_rot" tuples | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/build.rs:118` |
| `HIPFIRE_NPU_SOFTMAX_CONFIGS` | Parse "n_heads:ctx_len1+ctx_len2+..." entries | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/build.rs:232` |
| `HIPFIRE_NPU_TARGETS` | Runtime variable controlling npu targets in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/build.rs:60` |
| `HIPFIRE_PAGED_MOE_DEBUG` | Enabled when set to 1 | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/qwen35.rs:7435` |
| `HIPFIRE_PARO_BATCHED` | Runtime variable controlling paro batched in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/qwen35.rs:14226` |
| `HIPFIRE_PARO_FA3_FUSED` | Runtime variable controlling paro fa3 fused in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/qwen35.rs:28564` |
| `HIPFIRE_PARO_FUSED_PACK2` | Runtime variable controlling paro fused pack2 in hipfire | `/home/sadara/.hipfire/src/crates/rdna-compute/src/dispatch.rs:4992` |
| `HIPFIRE_PARO_FUSE_RMSNORM` | time per call. Net loss on every site. Default OFF; explicit opt-in for | `/home/sadara/.hipfire/src/crates/hipfire-runtime/src/llama.rs:1030` |
| `HIPFIRE_PARO_GATE_UP_FUSED` | Runtime variable controlling paro gate up fused in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/qwen35.rs:28162` |
| `HIPFIRE_PARO_LA2_FUSED` | Runtime variable controlling paro la2 fused in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/qwen35.rs:28264` |
| `HIPFIRE_PARO_LA4_FUSED` | Runtime variable controlling paro la4 fused in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/qwen35.rs:28260` |
| `HIPFIRE_PARO_LA_GATES_MQ4G128` | Used to configure runtime execution by explicitly setting "HIPFIRE_PARO_LA_GATES_MQ4G128" | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/paro_la_gates_codec.rs:325` |
| `HIPFIRE_PARO_PACK1` | Runtime variable controlling paro pack1 in hipfire | `/home/sadara/.hipfire/src/crates/rdna-compute/src/dispatch.rs:4781` |
| `HIPFIRE_PARO_PACK2` | Runtime variable controlling paro pack2 in hipfire | `/home/sadara/.hipfire/src/crates/rdna-compute/src/dispatch.rs:4784` |
| `HIPFIRE_PARO_PACK4` | Runtime variable controlling paro pack4 in hipfire | `/home/sadara/.hipfire/src/crates/rdna-compute/src/dispatch.rs:4787` |
| `HIPFIRE_PARO_PREROTATE` | Runtime variable controlling paro prerotate in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-runtime/src/llama.rs:1583` |
| `HIPFIRE_PARO_SHARED_PAIRS` | Runtime variable controlling paro shared pairs in hipfire | `/home/sadara/.hipfire/src/crates/rdna-compute/src/dispatch.rs:4986` |
| `HIPFIRE_PARO_SWIGLU_FUSED` | Runtime variable controlling paro swiglu fused in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-runtime/src/llama.rs:1600` |
| `HIPFIRE_PERF_BASELINE` | Runtime variable controlling perf baseline in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-eval/src/executor_examples.rs:1158` |
| `HIPFIRE_PERF_BASELINE_DIR` | Runtime variable controlling perf baseline dir in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-eval/src/executor_examples.rs:1171` |
| `HIPFIRE_PERPLEXITY_BIN` | Runtime variable controlling perplexity bin in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-eval/src/lib.rs:971` |
| `HIPFIRE_PFLASH_DAEMON_LABELS` | Runtime variable controlling pflash daemon labels in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-train/examples/ssm_drafter_train.rs:77` |
| `HIPFIRE_PFLASH_DEBUG` | Runtime variable controlling pflash debug in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-serving-core/src/generate.rs:2284` |
| `HIPFIRE_PFLASH_DRAFTER_KV` | Selects behavior from recognized values | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/pflash.rs:533` |
| `HIPFIRE_PFLASH_DRAFTER_STATE` | Hybrid drafter only stores K (and V for chat-path) at | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/pflash.rs:620` |
| `HIPFIRE_PFLASH_FRESH` | resume: reload weights + AdamW state from the checkpoint unless FRESH=1 | `/home/sadara/.hipfire/src/crates/hipfire-train/examples/pflash_drafter_train.rs:259` |
| `HIPFIRE_PFLASH_NIAH_BENCH_BIN` | Runtime variable controlling pflash niah bench bin in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-eval/src/lib.rs:926` |
| `HIPFIRE_PFLASH_REPORT_TRAIN` | Runtime variable controlling pflash report train in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-train/examples/ssm_drafter_train.rs:127` |
| `HIPFIRE_PFLASH_SCORE_LAYER` | Parses "HIPFIRE_PFLASH_SCORE_LAYER" with fallback defaults | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/pflash.rs:1081` |
| `HIPFIRE_PP_DFLASH` | Environment toggle value controls runtime behavior | `/home/sadara/.hipfire/src/crates/hipfire-daemon/src/main.rs:2627` |
| `HIPFIRE_PP_LAYERS` | Runtime variable controlling pp layers in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-serving-core/src/load.rs:1707` |
| `HIPFIRE_PP_PARITY_MODEL` | Runtime variable controlling pp parity model in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/tests/pp_parity.rs:197` |
| `HIPFIRE_PP_PFLASH` | Environment toggle value controls runtime behavior | `/home/sadara/.hipfire/src/crates/hipfire-daemon/src/main.rs:2645` |
| `HIPFIRE_PREFILL_ALPHA` | Runtime variable controlling prefill alpha in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/pflash.rs:124` |
| `HIPFIRE_PREFILL_BATCHED` | Enabled by default; set to 0 to disable | `/home/sadara/.hipfire/src/crates/hipfire-runtime/src/config.rs:86` |
| `HIPFIRE_PREFILL_BLOCK` | Runtime variable controlling prefill block in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/pflash.rs:144` |
| `HIPFIRE_PREFILL_COMPRESSION` | Runtime variable controlling prefill compression in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/pflash.rs:104` |
| `HIPFIRE_PREFILL_DRAFTER` | Runtime variable controlling prefill drafter in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/pflash.rs:157` |
| `HIPFIRE_PREFILL_KEEP_RATIO` | Runtime variable controlling prefill keep ratio in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/pflash.rs:114` |
| `HIPFIRE_PREFILL_MAX_BATCH` | Runtime variable controlling prefill max batch in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-serving-core/src/qwen35_prefill.rs:1347` |
| `HIPFIRE_PREFILL_MAX_LAYER` | Runtime variable controlling prefill max layer in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/qwen35.rs:13078` |
| `HIPFIRE_PREFILL_MIN_KEEP` | Runtime variable controlling prefill min keep in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/pflash.rs:129` |
| `HIPFIRE_PREFILL_PROFILE` | Enabled when set to 1 | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/pflash.rs:154` |
| `HIPFIRE_PREFILL_RECENT` | Runtime variable controlling prefill recent in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/pflash.rs:139` |
| `HIPFIRE_PREFILL_REUSE_PBS` | Used to configure runtime execution by explicitly setting "HIPFIRE_PREFILL_REUSE_PBS" | `/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/prefill_microbench.rs:126` |
| `HIPFIRE_PREFILL_SINK` | Runtime variable controlling prefill sink in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/pflash.rs:134` |
| `HIPFIRE_PREFILL_SPARSE_THRESHOLD` | Runtime variable controlling prefill sparse threshold in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/pflash.rs:149` |
| `HIPFIRE_PREFILL_STOP_AFTER_LA_LAYER` | Parses "HIPFIRE_PREFILL_STOP_AFTER_LA_LAYER" with fallback defaults | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/qwen35.rs:16324` |
| `HIPFIRE_PREFILL_STOP_STAGE` | Runtime variable controlling prefill stop stage in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/qwen35.rs:16330` |
| `HIPFIRE_PREFILL_STOP_STAGE_LAYER` | Parses "HIPFIRE_PREFILL_STOP_STAGE_LAYER" with fallback defaults | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/qwen35.rs:16327` |
| `HIPFIRE_PREFILL_THRESHOLD` | Runtime variable controlling prefill threshold in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/pflash.rs:109` |
| `HIPFIRE_PREFIX_BOUNDARY_CHECKPOINTS` | Interprets "HIPFIRE_PREFIX_BOUNDARY_CHECKPOINTS" from environment to select behavior | `/home/sadara/.hipfire/src/crates/hipfire-serving-core/src/qwen35_prefill.rs:413` |
| `HIPFIRE_PROFILE` | HIPFIRE_PROFILE=1 + HIPFIRE_PROFILE_CYCLES=N: per-kernel profiling | `/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/mtp_only_demo.rs:495` |
| `HIPFIRE_PROFILE_CYCLES` | Runtime variable controlling profile cycles in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/mtp_only_demo.rs:496` |
| `HIPFIRE_PROFILE_DECODE` | Enabled when set to 1 | `/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/bench_qwen35_speed.rs:559` |
| `HIPFIRE_PROMPT_HEAT_JSON` | Enabled when set to 1 | `/home/sadara/.hipfire/src/crates/hipfire-runtime/src/config.rs:62` |
| `HIPFIRE_PROMPT_HEAT_LIMIT` | Runtime variable controlling prompt heat limit in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-runtime/src/config.rs:63` |
| `HIPFIRE_PROMPT_TOKEN_HEAT` | Enabled when set to 1 | `/home/sadara/.hipfire/src/crates/hipfire-runtime/src/config.rs:60` |
| `HIPFIRE_Q8_BATCHED_LEGACY` | Environment toggle value controls runtime behavior | `/home/sadara/.hipfire/src/crates/rdna-compute/src/feature_flags.rs:280` |
| `HIPFIRE_Q8_FA_ATTENTION_IGNORE_TREE_BIAS` | Interprets "HIPFIRE_Q8_FA_ATTENTION_IGNORE_TREE_BIAS" from environment to select behavior | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/qwen35.rs:12445` |
| `HIPFIRE_Q8_FA_ATTENTION_ROW_LOOP` | Interprets "HIPFIRE_Q8_FA_ATTENTION_ROW_LOOP" from environment to select behavior | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/qwen35.rs:12409` |
| `HIPFIRE_Q8_FA_ATTENTION_SCALAR_LOOP` | Interprets "HIPFIRE_Q8_FA_ATTENTION_SCALAR_LOOP" from environment to select behavior | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/qwen35.rs:12421` |
| `HIPFIRE_Q8_FA_ATTENTION_SERIAL_KV_LOOP` | Interprets "HIPFIRE_Q8_FA_ATTENTION_SERIAL_KV_LOOP" from environment to select behavior | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/qwen35.rs:12433` |
| `HIPFIRE_Q8_GATE_UP_4W` | Disabled when set to 0 | `/home/sadara/.hipfire/src/crates/rdna-compute/examples/test_gemm_q8_gate_up_wmma.rs:130` |
| `HIPFIRE_Q8_GATE_UP_BENCH` | Enabled when set to 1 | `/home/sadara/.hipfire/src/crates/rdna-compute/examples/test_gemm_q8_gate_up_wmma.rs:129` |
| `HIPFIRE_Q8_GDN_VERIFY_PER_TOKEN` | Interprets "HIPFIRE_Q8_GDN_VERIFY_PER_TOKEN" from environment to select behavior | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/qwen35.rs:12457` |
| `HIPFIRE_Q8_GDN_VERIFY_SERIAL_FRAMES` | Interprets "HIPFIRE_Q8_GDN_VERIFY_SERIAL_FRAMES" from environment to select behavior | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/qwen35.rs:12469` |
| `HIPFIRE_Q8_WMMA_4W` | Environment toggle value controls runtime behavior | `/home/sadara/.hipfire/src/crates/rdna-compute/src/dispatch.rs:29148` |
| `HIPFIRE_Q8_WMMA_X64` | Environment toggle value controls runtime behavior | `/home/sadara/.hipfire/src/crates/rdna-compute/src/dispatch.rs:28959` |
| `HIPFIRE_QA_KV_MODES` | Defaults to q8,asym4,asym3,asym2 when unset | `/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/test_inferenceQA.rs:761` |
| `HIPFIRE_QTIP_EVAL_ST` | Selects behavior from recognized values | `/home/sadara/.hipfire/src/crates/hipfire-quantize/src/qtip.rs:749` |
| `HIPFIRE_QTIP_HESSIAN` | Runtime variable controlling qtip hessian in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-quantize/src/main.rs:6375` |
| `HIPFIRE_QUANT_DIAG_PATH` | Runtime variable controlling quant diag path in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-quantize/src/main.rs:12658` |
| `HIPFIRE_QUANT_THREADS` | Runtime variable controlling quant threads in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-quantize/src/main.rs:6185` |
| `HIPFIRE_QWEN35_DECODE_BATCH` | Defaults to auto when unset | `/home/sadara/.hipfire/src/crates/hipfire-serving-core/src/qwen35_decode.rs:348` |
| `HIPFIRE_QWEN35_DECODE_BATCH_MAX` | Runtime variable controlling qwen35 decode batch max in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-serving-core/src/qwen35_decode.rs:462` |
| `HIPFIRE_QWEN35_DECODE_INTERNAL_PARITY` | Interprets "HIPFIRE_QWEN35_DECODE_INTERNAL_PARITY" from environment to select behavior | `/home/sadara/.hipfire/src/crates/hipfire-serving-core/src/qwen35_decode.rs:484` |
| `HIPFIRE_QWEN35_DECODE_NATIVE_MULTIROW` | Interprets "HIPFIRE_QWEN35_DECODE_NATIVE_MULTIROW" from environment to select behavior | `/home/sadara/.hipfire/src/crates/hipfire-serving-core/src/qwen35_decode.rs:473` |
| `HIPFIRE_QWEN35_EXPERT_CACHE_MB` | Parses "HIPFIRE_QWEN35_EXPERT_CACHE_MB" with fallback defaults | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/qwen35.rs:699` |
| `HIPFIRE_QWEN35_EXPERT_CACHE_TRACE` | Interprets "HIPFIRE_QWEN35_EXPERT_CACHE_TRACE" from environment to select behavior | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/qwen35.rs:5175` |
| `HIPFIRE_QWEN35_FFN_BF16` | Selects behavior from recognized values | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/ffn_bf16.rs:57` |
| `HIPFIRE_QWEN35_FFN_BF16_LAYER` | Selects behavior from recognized values | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/ffn_bf16.rs:71` |
| `HIPFIRE_QWEN35_FFN_BF16_TRACE` | Parses "HIPFIRE_QWEN35_FFN_BF16_TRACE" with fallback defaults | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/ffn_bf16.rs:83` |
| `HIPFIRE_QWEN35_FINITE_TRACE` | Runtime variable controlling qwen35 finite trace in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/qwen35.rs:13705` |
| `HIPFIRE_QWEN35_PAGED_EXPERTS` | Interprets "HIPFIRE_QWEN35_PAGED_EXPERTS" from environment to select behavior | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/qwen35.rs:3988` |
| `HIPFIRE_QWEN35_PREFILL_SESSION_BATCH` | Runtime variable controlling qwen35 prefill session batch in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-serving-core/src/qwen35_prefill.rs:1656` |
| `HIPFIRE_QWEN35_ROUTED_ONLY_MOE_FORWARD` | Runtime variable controlling qwen35 routed only moe forward in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/qwen35.rs:7445` |
| `HIPFIRE_QWEN35_STAGE_SYNC` | Runtime variable controlling qwen35 stage sync in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/qwen35.rs:13739` |
| `HIPFIRE_QWEN35_STAGE_TRACE` | Runtime variable controlling qwen35 stage trace in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/qwen35.rs:13733` |
| `HIPFIRE_QWEN35_XDNA1_INSTR` | Runtime variable controlling qwen35 xdna1 instr in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-npu/src/lib.rs:72` |
| `HIPFIRE_QWEN35_XDNA1_XCLBIN` | Runtime variable controlling qwen35 xdna1 xclbin in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-npu/src/lib.rs:69` |
| `HIPFIRE_RDNA2_VARIANT` | Runtime variable controlling rdna2 variant in hipfire | `/home/sadara/.hipfire/src/crates/rdna-compute/src/feature_flags.rs:298` |
| `HIPFIRE_RECOVER_MODE` | HIPFIRE_RECOVER_MODE=lora+norms → LoRA + layernorms (default, more capacity) | `/home/sadara/.hipfire/src/crates/hipfire-train/examples/coherence_recovery_supra50m.rs:152` |
| `HIPFIRE_REPLAY_GRAPH` | Enabled when set to 1 | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/speculative.rs:5080` |
| `HIPFIRE_RESOURCE_LOCK` | HIPFIRE_RESOURCE_LOCK=0 disables daemon startup resource leases | `/home/sadara/.hipfire/src/crates/hipfire-daemon-adapter/src/lib.rs:721` |
| `HIPFIRE_RESOURCE_LOCK_CPU_CORES` | HIPFIRE_RESOURCE_LOCK_CPU_CORES=0,2-4 adds daemon startup leases for CPU cores | `/home/sadara/.hipfire/src/crates/hipfire-daemon-adapter/src/lib.rs:625` |
| `HIPFIRE_RESOURCE_LOCK_DIR` | HIPFIRE_RESOURCE_LOCK_DIR overrides the daemon resource-lock root directory | `/home/sadara/.hipfire/src/crates/hipfire-daemon-adapter/src/lib.rs:737` |
| `HIPFIRE_RESOURCE_LOCK_NPUS` | HIPFIRE_RESOURCE_LOCK_NPUS=1 leases every detected NPU; comma lists lease explicit NPU IDs | `/home/sadara/.hipfire/src/crates/hipfire-daemon-adapter/src/lib.rs:590` |
| `HIPFIRE_RESOURCE_LOCK_WAIT_MS` | HIPFIRE_RESOURCE_LOCK_WAIT_MS waits for busy daemon resource leases before failing startup | `/home/sadara/.hipfire/src/crates/hipfire-daemon-adapter/src/lib.rs:741` |
| `HIPFIRE_RESPONSES_STATE_MAX` | Runtime variable controlling responses state max in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-server/src/routes/responses.rs:660` |
| `HIPFIRE_ROCBLAS_ALL_ARCHS` | Runtime variable controlling rocblas all archs in hipfire | `/home/sadara/.hipfire/src/crates/rdna-compute/src/feature_flags.rs:288` |
| `HIPFIRE_ROCBLAS_OFF` | Enabled when set to 1 | `/home/sadara/.hipfire/src/crates/rdna-compute/src/feature_flags.rs:290` |
| `HIPFIRE_ROCPROF_BIN` | Runtime variable controlling rocprof bin in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-eval/src/rocprof.rs:275` |
| `HIPFIRE_ROCPROF_CSV` | Runtime variable controlling rocprof csv in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/bench_qwen35_speed.rs:325` |
| `HIPFIRE_ROPE_INTERLEAVED_LEGACY` | Runtime variable controlling rope interleaved legacy in hipfire | `/home/sadara/.hipfire/src/crates/rdna-compute/src/feature_flags.rs:281` |
| `HIPFIRE_RQ2_BULK_BITS` | Runtime variable controlling rq2 bulk bits in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-quantize/src/main.rs:9788` |
| `HIPFIRE_RQ2_DAMP` | De-risk B: single shared, foldable residual-stream rotation. With | `/home/sadara/.hipfire/src/crates/hipfire-quantize/src/main.rs:9792` |
| `HIPFIRE_RQ2_PROTECT_FRAC` | Runtime variable controlling rq2 protect frac in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-quantize/src/main.rs:9784` |
| `HIPFIRE_RQ2_Q8_EMBED` | Q8 (~20% of params on a tied-embedding 0.8B). With HIPFIRE_RQ2_Q8_EMBED=1, | `/home/sadara/.hipfire/src/crates/hipfire-quantize/src/main.rs:9910` |
| `HIPFIRE_RQ2_SHARE_RESID` | HIPFIRE_RQ2_SHARE_RESID=1, every k==1024 weight (the d_model residual | `/home/sadara/.hipfire/src/crates/hipfire-quantize/src/main.rs:9806` |
| `HIPFIRE_RQ3_BULK_BITS` | Runtime variable controlling rq3 bulk bits in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-quantize/src/main.rs:9947` |
| `HIPFIRE_RQ3_PROTECT_FRAC` | Runtime variable controlling rq3 protect frac in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-quantize/src/main.rs:9943` |
| `HIPFIRE_RQ3_Q8_EMBED` | Iso-bit embed for an honest mq4 comparison (same as roughquant2 de-risk A) | `/home/sadara/.hipfire/src/crates/hipfire-quantize/src/main.rs:10009` |
| `HIPFIRE_RQ4_BULK` | Bulk codec: "mq4" → real mq4 format (fair mq4+protect-vs-mq4 test, set | `/home/sadara/.hipfire/src/crates/hipfire-quantize/src/main.rs:10050` |
| `HIPFIRE_RQ4_BULK_BITS` | Bulk codec: "mq4" → real mq4 format (fair mq4+protect-vs-mq4 test, set | `/home/sadara/.hipfire/src/crates/hipfire-quantize/src/main.rs:10044` |
| `HIPFIRE_RQ4_DUMP_RANK` | HIPFIRE_RQ4_DUMP_RANK=1: print the residual-channel saliency ranking | `/home/sadara/.hipfire/src/crates/hipfire-quantize/src/main.rs:10156` |
| `HIPFIRE_RQ4_INVERT` | HIPFIRE_RQ4_INVERT=1: protect the LOWEST-saliency channels instead of the | `/home/sadara/.hipfire/src/crates/hipfire-quantize/src/main.rs:10174` |
| `HIPFIRE_RQ4_MQ_BITS` | Uniform bulk bit-width for the mq bulk (4=mq4, 5, 6=mq6). protect_frac=0 | `/home/sadara/.hipfire/src/crates/hipfire-quantize/src/main.rs:10060` |
| `HIPFIRE_RQ4_OBS_DAMP` | Uniform bulk bit-width for the mq bulk (4=mq4, 5, 6=mq6). protect_frac=0 | `/home/sadara/.hipfire/src/crates/hipfire-quantize/src/main.rs:10054` |
| `HIPFIRE_RQ4_PROTECT_FRAC` | diag(H) residual-channel energy from true residual readers | `/home/sadara/.hipfire/src/crates/hipfire-quantize/src/main.rs:10630` |
| `HIPFIRE_RQ4_PROTECT_Q8` | on diag(H) alone). diag = E[x²] (activation energy); wnorm = ‖W[:,c]‖² | `/home/sadara/.hipfire/src/crates/hipfire-quantize/src/main.rs:10065` |
| `HIPFIRE_RQ4_Q8_EMBED` | Enabled when set to 1 | `/home/sadara/.hipfire/src/crates/hipfire-quantize/src/main.rs:10416` |
| `HIPFIRE_RQ4_RANDOM_SEED` | Runtime variable controlling rQ4 random seed in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-quantize/src/main.rs:10279` |
| `HIPFIRE_RQ4_SALIENCY` | (weight energy); product = ‖W[:,c]‖²·E[x²] (output-error contribution) | `/home/sadara/.hipfire/src/crates/hipfire-quantize/src/main.rs:10069` |
| `HIPFIRE_RQ4_VOID_ONLY` | Runtime variable controlling rQ4 void only in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-quantize/src/main.rs:10198` |
| `HIPFIRE_RQ_BULK_BITS` | Runtime variable controlling rq bulk bits in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-quantize/src/main.rs:9717` |
| `HIPFIRE_RQ_GROUP` | Runtime variable controlling rq group in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-quantize/src/main.rs:9721` |
| `HIPFIRE_RQ_HAND` | Environment toggle value controls runtime behavior | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/qwen35.rs:22538` |
| `HIPFIRE_RQ_PROTECT_FRAC` | Runtime variable controlling rq protect frac in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-quantize/src/main.rs:9713` |
| `HIPFIRE_RUN_EXAMPLE_BIN` | Runtime variable controlling run example bin in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-eval/src/lib.rs:941` |
| `HIPFIRE_SAMPLE_COMPARE` | Enabled when set to 1 | `/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/infer_qwen35.rs:222` |
| `HIPFIRE_SERVER_RESIDENT_STATE_BUDGET_MB` | Runtime variable controlling server resident state budget mb in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-serving-core/src/session.rs:707` |
| `HIPFIRE_SMOKE_KV` | Select KV cache quant via HIPFIRE_SMOKE_KV (default q8, matches the | `/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/a3b_smoke_forward.rs:109` |
| `HIPFIRE_SMOKE_KV_SEQ` | production CLI default). asym3/asym4 engage the Givens-rotated 3/4-bit | `/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/a3b_smoke_forward.rs:102` |
| `HIPFIRE_SMOKE_MODE` | raw (default for back-compat): tokenize "Hello" and decode from pos=0 | `/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/a3b_smoke_forward.rs:157` |
| `HIPFIRE_SMOKE_PROMPT` | Defaults to Hello when unset | `/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/a3b_smoke_forward.rs:158` |
| `HIPFIRE_SMOKE_STEPS` | Runtime variable controlling smoke steps in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/a3b_smoke_forward.rs:38` |
| `HIPFIRE_SPEC_PHASES` | Enabled when set to 1 | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/speculative.rs:7069` |
| `HIPFIRE_STATE` | Interprets "HIPFIRE_STATE" from environment to select behavior | `/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/greedy_dump_top5.rs:245` |
| `HIPFIRE_TARGET_ARCH` | Runtime variable controlling target arch in hipfire | `/home/sadara/.hipfire/src/crates/rdna-compute/src/dispatch.rs:783` |
| `HIPFIRE_TIER_RATIO` | Runtime variable controlling tier ratio in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-quantize/src/main.rs:6607` |
| `HIPFIRE_TP_BENCH_ITERS` | Runtime variable controlling tp bench iters in hipfire | `/home/sadara/.hipfire/src/crates/hip-bridge/examples/rccl_smoke.rs:19` |
| `HIPFIRE_TP_BENCH_N` | Runtime variable controlling tp bench n in hipfire | `/home/sadara/.hipfire/src/crates/hip-bridge/examples/rccl_smoke.rs:15` |
| `HIPFIRE_TP_EXPERT_ASSIGN` | Selects behavior from recognized values | `/home/sadara/.hipfire/src/crates/hipfire-runtime/src/tp_shard.rs:57` |
| `HIPFIRE_TP_USE_RCCL` | Runtime variable controlling tp use rccl in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-runtime/src/config.rs:90` |
| `HIPFIRE_UNIFORM_VRAM_TOLERANCE_GB` | Runtime variable controlling uniform vram tolerance gb in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-runtime/src/config.rs:105` |
| `HIPFIRE_VERIFY_GRAPH` | Runtime variable controlling verify graph in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/speculative.rs:6531` |
| `HIPFIRE_VERIFY_GRAPH_TIMING` | (HIPFIRE_VERIFY_GRAPH_TIMING=1). Two device-sync points bracket the | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/speculative.rs:6546` |
| `HIPFIRE_VERIFY_GRAPH_TREE` | Enabled when set to 1 | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/speculative.rs:6521` |
| `HIPFIRE_VL_DUMP_DIR` | little-endian f32 blobs + JSON sidecars to $HIPFIRE_VL_DUMP_DIR | `/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/infer.rs:148` |
| `HIPFIRE_WO_MMQ` | Enabled when set to 1 | `/home/sadara/.hipfire/src/crates/rdna-compute/src/feature_flags.rs:204` |
| `HIPFIRE_WO_WMMA_VARIANT` | Runtime variable controlling wo wmma variant in hipfire | `/home/sadara/.hipfire/src/crates/rdna-compute/src/feature_flags.rs:285` |
| `HIPFIRE_XDNA1_LIB` | Runtime variable controlling xdna1 lib in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-npu/src/lib.rs:65` |
| `HIP_PATH` | fails with "file not found". Add well-known candidates as -I flags; | `/home/sadara/.hipfire/src/crates/rdna-compute/src/compiler.rs:476` |
| `HIP_VISIBLE_DEVICES` | Used to configure runtime execution by explicitly setting "HIP_VISIBLE_DEVICES" | `/home/sadara/.hipfire/src/crates/hipfire-daemon-adapter/src/lib.rs:1127` |
| `HOME` | Runtime variable controlling home in hipfire | `/home/sadara/.hipfire/src/crates/rdna-compute/src/compiler.rs:236` |
| `HOSTNAME` | Runtime variable controlling hostname in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/coherence_probe.rs:416` |
| `MAX_TOKENS` | Parses "MAX_TOKENS" with fallback defaults | `/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/greedy_dump.rs:112` |
| `MMQ_TEST_MODE` | Defaults to residual when unset | `/home/sadara/.hipfire/src/crates/rdna-compute/examples/test_gfx906_mmq_correctness.rs:65` |
| `NO_NGRAM` | Disabled for perf measurement — re-enable after implementing GPU n-gram kernel | `/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/infer_vl.rs:350` |
| `PATH` | Runtime variable controlling path in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-eval/src/rocprof.rs:285` |
| `PROMPT_MODE` | Defaults to thinking when unset | `/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/greedy_dump_top5.rs:91` |
| `QWEN35_TEST_MODEL` | Runtime variable controlling qwen35 test model in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/test_qwen35_loadQA.rs:22` |
| `ROCM_PATH` | Runtime variable controlling rocm path in hipfire | `/home/sadara/.hipfire/src/crates/rdna-compute/src/compiler.rs:133` |
| `ROCR_VISIBLE_DEVICES` | Runtime variable controlling rocr visible devices in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-daemon-adapter/src/lib.rs:549` |
| `TINYLLAMA_GGUF` | Runtime variable controlling tinyllama gguf in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/test_q4f16QA.rs:38` |
| `TRIALS` | Runtime variable controlling trials in hipfire | `/home/sadara/.hipfire/src/crates/rdna-compute/examples/bench_gfx1151_hfq4_s4_mmq.rs:151` |
| `USERPROFILE` | Runtime variable controlling userprofile in hipfire | `/home/sadara/.hipfire/src/crates/hipfire-daemon/src/main.rs:99` |
| `USE_SAMPLE` | Enabled when set to 1 | `/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/a3b_multiturn_oneshot.rs:117` |

- Total env vars: **456**
- `HIPFIRE_*` vars: **421**
- non-`HIPFIRE_*` vars: **35**
