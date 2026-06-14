# hipfire environment variables — canonical reference

Generated automatically from source and inline comments by `scripts/gen-env-docs.py`.

| Variable | Description | Defined at |
|---|---|---|
| `BENCH_BATCH` | Runtime variable controlling bench batch in hipfire. | `/home/sadara/.hipfire/src/crates/rdna-compute/examples/bench_stream_overlap.rs:56` |
| `BENCH_DRAFT_K` | Runtime variable controlling bench draft k in hipfire. | `/home/sadara/.hipfire/src/crates/rdna-compute/examples/bench_stream_overlap.rs:188` |
| `BENCH_DRAFT_LAYERS` | Runtime variable controlling bench draft layers in hipfire. | `/home/sadara/.hipfire/src/crates/rdna-compute/examples/bench_stream_overlap.rs:215` |
| `BENCH_DRAFT_M` | Runtime variable controlling bench draft m in hipfire. | `/home/sadara/.hipfire/src/crates/rdna-compute/examples/bench_stream_overlap.rs:184` |
| `BENCH_DRAFT_N` | Runtime variable controlling bench draft n in hipfire. | `/home/sadara/.hipfire/src/crates/rdna-compute/examples/bench_stream_overlap.rs:192` |
| `BENCH_K` | Runtime variable controlling bench k in hipfire. | `/home/sadara/.hipfire/src/crates/rdna-compute/examples/bench_stream_overlap.rs:52` |
| `BENCH_M` | Runtime variable controlling bench m in hipfire. | `/home/sadara/.hipfire/src/crates/rdna-compute/examples/bench_stream_overlap.rs:48` |
| `BENCH_VERIFY_LAYERS` | Runtime variable controlling bench verify layers in hipfire. | `/home/sadara/.hipfire/src/crates/rdna-compute/examples/bench_stream_overlap.rs:219` |
| `CARGO_FEATURE_NPU_KERNELS` | Runtime variable controlling cargo feature npu kernels in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/build.rs:29` |
| `CARGO_MANIFEST_DIR` | CARGO_MANIFEST_DIR = <workspace>/crates/hipfire-arch-qwen35 | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/build.rs:596` |
| `CLICOLOR` | Runtime variable controlling clicolor in hipfire. | `/home/sadara/.hipfire/src/cli/chat.ts:101` |
| `DDTREE_TIMING` | pre_verify / verify. The draft and top-K are fused into one GPU- | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/speculative.rs:10881` |
| `DEBUG_LAYERS` | Runtime variable controlling debug layers in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/qwen35.rs:7195` |
| `DFLASH_LIVE_TAU` | Runtime variable controlling dflash live tau in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/dflash_spec_demo.rs:1512` |
| `DP4A` | Force the candidate into out_tp: W64=1 → dp4a wave64, DP4A=1 → | `/home/sadara/.hipfire/src/crates/rdna-compute/examples/q8_batched_attn_microbench.rs:104` |
| `FP32_STATE` | Runtime variable controlling fp32 state in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/infer_qwen35.rs:191` |
| `HFQ_TEST_N_ITER` | Parses "HFQ_TEST_N_ITER" with fallback defaults | `/home/sadara/.hipfire/src/crates/rdna-compute/examples/test_hfq4_residual_dp4a.rs:74` |
| `HFQ_TEST_SCALE_LOG10` | Parses "HFQ_TEST_SCALE_LOG10" with fallback defaults | `/home/sadara/.hipfire/src/crates/rdna-compute/examples/test_hfq4_residual_dp4a.rs:196` |
| `HFQ_TEST_ZP_MAX` | Parses "HFQ_TEST_ZP_MAX" with fallback defaults | `/home/sadara/.hipfire/src/crates/rdna-compute/examples/test_hfq4_residual_dp4a.rs:200` |
| `HF_TOKEN` | Runtime variable controlling hf token in hipfire. | `/home/sadara/.hipfire/src/cli/index.ts:989` |
| `HIPFIRE_ADAPTIVE_B_DOWN` | Runtime variable controlling adaptive b down in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/dflash_spec_demo.rs:1779` |
| `HIPFIRE_ADAPTIVE_B_UNSAFE` | the user explicitly widens. Opt out via HIPFIRE_ADAPTIVE_B_UNSAFE=1 | `/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/dflash_spec_demo.rs:921` |
| `HIPFIRE_ADAPTIVE_B_UP` | HIPFIRE_ADAPTIVE_B_UP=0.XX / HIPFIRE_ADAPTIVE_B_DOWN=0.XX | `/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/dflash_spec_demo.rs:1775` |
| `HIPFIRE_ALLOW_MIXED_ARCH` | Runtime variable controlling allow mixed arch in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-runtime/src/config.rs:95` |
| `HIPFIRE_ALLOW_MQ2` | Enabled when set to 1 | `/home/sadara/.hipfire/src/crates/hipfire-quantize/src/main.rs:5914` |
| `HIPFIRE_ALLOW_MQ2_LLOYD` | Enabled when set to 1 | `/home/sadara/.hipfire/src/crates/hipfire-quantize/src/main.rs:5949` |
| `HIPFIRE_ALLOW_MQ3_LLOYD` | Enabled when set to 1 | `/home/sadara/.hipfire/src/crates/hipfire-quantize/src/main.rs:5934` |
| `HIPFIRE_ALLOW_MQ4_LLOYD` | Enabled when set to 1 | `/home/sadara/.hipfire/src/crates/hipfire-quantize/src/main.rs:5984` |
| `HIPFIRE_ALLOW_UNIT_IMATRIX` | Environment toggle value controls runtime behavior | `/home/sadara/.hipfire/src/crates/hipfire-quantize/src/main.rs:5695` |
| `HIPFIRE_ATTN_FLASH` | Honors HIPFIRE_ATTN_FLASH=never\|0\|off as an explicit override | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/qwen35.rs:8037` |
| `HIPFIRE_AWQ_F1_ONLY` | F1-vs-F2 A/B gate. When "HIPFIRE_AWQ_F1_ONLY=1" is set, the F2 | `/home/sadara/.hipfire/src/crates/hipfire-quantize/src/main.rs:4770` |
| `HIPFIRE_BASELINE_ARCH` | Runtime variable controlling baseline arch in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-runtime/src/coherence_runtime.rs:462` |
| `HIPFIRE_BENCH_QWEN35_SPEED_BIN` | Runtime variable controlling bench qwen35 speed bin in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-runtime/src/eval_harness/mod.rs:9742` |
| `HIPFIRE_BF16_DENSE_M128` | Enabled by default; set to 0 to disable | `/home/sadara/.hipfire/src/crates/rdna-compute/src/dispatch.rs:43165` |
| `HIPFIRE_BF16_MOE_M256` | Enabled when set to 1 | `/home/sadara/.hipfire/src/crates/rdna-compute/src/dispatch.rs:22222` |
| `HIPFIRE_BF16_WEIGHTS` | Runtime variable controlling bf16 weights in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/qwen35.rs:209` |
| `HIPFIRE_BLOB_FORCE` | Graph / capture / deterministic | `/home/sadara/.hipfire/src/crates/rdna-compute/src/feature_flags.rs:254` |
| `HIPFIRE_CALIB_PROFILE` | Enable with HIPFIRE_CALIB_PROFILE=1; emits to stderr | `/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/triattn_validate.rs:40` |
| `HIPFIRE_CASK_OFF` | HIPFIRE_CASK_OFF=1 is an ops escape hatch: forces no auto-attach | `/home/sadara/.hipfire/src/cli/index.ts:878` |
| `HIPFIRE_CHATML` | Enabled when set to 1 | `/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/probe_argmax_agreement.rs:58` |
| `HIPFIRE_CHAT_TEMPLATE_FILE` | 1. Env-var override | `/home/sadara/.hipfire/src/crates/hipfire-daemon/src/main.rs:11763` |
| `HIPFIRE_CONV1D_TREE_GFX1151` | Environment toggle value controls runtime behavior | `/home/sadara/.hipfire/src/crates/rdna-compute/examples/bench_conv1d_tree_gfx1151.rs:22` |
| `HIPFIRE_DAEMON_BIN` | Runtime variable controlling daemon bin in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-server/src/daemon/engine.rs:234` |
| `HIPFIRE_DAEMON_RESIDENT_STATE_BUDGET_MB` | Runtime variable controlling daemon resident state budget mb in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-daemon/src/main.rs:5500` |
| `HIPFIRE_DDTREE_BUDGET` | Runtime variable controlling DDTree budget in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-runtime/src/config.rs:74` |
| `HIPFIRE_DDTREE_FORCE_SLOW` | HIPFIRE_DDTREE_FORCE_SLOW=1: force the slow (re-verify) path even when | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/speculative.rs:11108` |
| `HIPFIRE_DDTREE_LOGW_CUTOFF` | Adaptive-B usage report — only meaningful when --adaptive-b is on | `/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/dflash_spec_demo.rs:2517` |
| `HIPFIRE_DDTREE_PATH_B_CAPTURE` | Path B slow-path-kill (opt-in, WIP). Replaces the ~40-50 ms full | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/speculative.rs:11147` |
| `HIPFIRE_DDTREE_PATH_C` | Resolve "HIPFIRE_DDTREE_PATH_C" ONCE before the decode loop. The | `/home/sadara/.hipfire/src/crates/hipfire-daemon/src/main.rs:14342` |
| `HIPFIRE_DDTREE_PATH_C_VERBOSE` | Runtime variable controlling DDTree path c verbose in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/speculative.rs:11672` |
| `HIPFIRE_DDTREE_PATH_C_VERIFY_GRAPH` | Runtime variable controlling DDTree path c verify graph in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/speculative.rs:98` |
| `HIPFIRE_DDTREE_TAPE_DUMP` | Enabled when set to 1 | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/speculative.rs:11127` |
| `HIPFIRE_DDTREE_TOPK` | Runtime variable controlling DDTree topk in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-runtime/src/config.rs:78` |
| `HIPFIRE_DDTREE_TREE_LA` | Opt out with HIPFIRE_DDTREE_TREE_LA=0 if a regression is suspected | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/speculative.rs:10989` |
| `HIPFIRE_DEBUG_CORRUPT_PREFIX_HASH_ONCE` | Runtime variable controlling debug corrupt prefix hash once in hipfire. | `/home/sadara/.hipfire/src/cli/index.ts:2180` |
| `HIPFIRE_DEBUG_PREFIX_BOUNDARIES` | Runtime variable controlling debug prefix boundaries in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-daemon/src/main.rs:6978` |
| `HIPFIRE_DEBUG_SHARED_PREFIX_FANOUT` | Runtime variable controlling debug shared prefix fanout in hipfire. | `/home/sadara/.hipfire/src/cli/index.ts:5382` |
| `HIPFIRE_DEEPSEEK4_ATTN` | main model's final_norm_and_head head-HC reduction. Without | `/home/sadara/.hipfire/src/crates/hipfire-arch-deepseek4/src/forward.rs:68` |
| `HIPFIRE_DEEPSEEK4_ATTN_DEBUG_BISECT` | DEBUG: in-kernel bisect (HIPFIRE_DEEPSEEK4_ATTN_DEBUG_BISECT=1) | `/home/sadara/.hipfire/src/crates/hipfire-arch-deepseek4/src/forward.rs:5368` |
| `HIPFIRE_DEEPSEEK4_ATTN_PER_POS` | DEBUG: HIPFIRE_DEEPSEEK4_ATTN_PER_POS=1 substitutes a per-position loop | `/home/sadara/.hipfire/src/crates/hipfire-arch-deepseek4/src/forward.rs:5301` |
| `HIPFIRE_DEEPSEEK4_ATTN_TOPK_DIRECT` | Runtime variable controlling deepseek4 attn topk direct in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-arch-deepseek4/src/forward.rs:5650` |
| `HIPFIRE_DEEPSEEK4_ATTN_TWIN` | DEBUG: same-process twin-call test (HIPFIRE_DEEPSEEK4_ATTN_TWIN=1) | `/home/sadara/.hipfire/src/crates/hipfire-arch-deepseek4/src/forward.rs:5346` |
| `HIPFIRE_DEEPSEEK4_BATCH_HEAD` | Opt-out: HIPFIRE_DEEPSEEK4_BATCH_HEAD=0 forces the legacy per-position | `/home/sadara/.hipfire/src/crates/hipfire-arch-deepseek4/src/forward.rs:7229` |
| `HIPFIRE_DEEPSEEK4_CACHE_TRACE` | Runtime variable controlling deepseek4 cache trace in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-daemon/src/main.rs:17882` |
| `HIPFIRE_DEEPSEEK4_CHAT_RAW` | Enabled when set to 1 | `/home/sadara/.hipfire/src/crates/hipfire-arch-deepseek4/examples/deepseek4_chat.rs:161` |
| `HIPFIRE_DEEPSEEK4_COMP_F16_WMMA` | Opt out via HIPFIRE_DEEPSEEK4_COMP_F16_WMMA=0 | `/home/sadara/.hipfire/src/crates/hipfire-arch-deepseek4/src/forward.rs:5721` |
| `HIPFIRE_DEEPSEEK4_COMP_ROPE_POS` | Runtime variable controlling deepseek4 comp rope pos in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-arch-deepseek4/src/forward.rs:4390` |
| `HIPFIRE_DEEPSEEK4_DUMP_PROMPT` | Runtime variable controlling deepseek4 dump prompt in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-daemon/src/main.rs:17232` |
| `HIPFIRE_DEEPSEEK4_DUMP_STATE` | Selects behavior from recognized values | `/home/sadara/.hipfire/src/crates/hipfire-arch-deepseek4/src/forward.rs:208` |
| `HIPFIRE_DEEPSEEK4_DUMP_TOPK` | Gate 1 validation (2026-05-22): dump per-layer topk_indices to a | `/home/sadara/.hipfire/src/crates/hipfire-arch-deepseek4/src/forward.rs:6630` |
| `HIPFIRE_DEEPSEEK4_EXPERT_LAYER_END` | Runtime variable controlling deepseek4 expert layer end in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-arch-deepseek4/src/arch.rs:490` |
| `HIPFIRE_DEEPSEEK4_F32_TRACE` | Runtime variable controlling deepseek4 f32 trace in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-arch-deepseek4/src/forward.rs:236` |
| `HIPFIRE_DEEPSEEK4_FUSED_UNSCATTER_SILU` | Measured perf at PP_BATCH=512 / 2.1k-tok prompt: -0.4% prefill — | `/home/sadara/.hipfire/src/crates/hipfire-arch-deepseek4/src/forward.rs:6860` |
| `HIPFIRE_DEEPSEEK4_GEN_TOKENS` | Runtime variable controlling deepseek4 gen tokens in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-arch-deepseek4/examples/deepseek4_chat.rs:157` |
| `HIPFIRE_DEEPSEEK4_GRAPH` | Selects behavior from recognized values | `/home/sadara/.hipfire/src/crates/hipfire-arch-deepseek4/src/forward.rs:1488` |
| `HIPFIRE_DEEPSEEK4_HFQ4_WMMA` | HFQ4G256/Raw. WMMA route requires F16 input staging | `/home/sadara/.hipfire/src/crates/hipfire-arch-deepseek4/src/forward.rs:310` |
| `HIPFIRE_DEEPSEEK4_INDEXER_WMMA` | Runtime variable controlling deepseek4 indexer wmma in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-arch-deepseek4/src/forward.rs:6076` |
| `HIPFIRE_DEEPSEEK4_LOAD_MTP` | Runtime variable controlling deepseek4 load MTP in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-arch-deepseek4/src/arch.rs:889` |
| `HIPFIRE_DEEPSEEK4_MAX_COMPRESS_POS` | ratio == 128: identity gather, no indexer. Per-batch n_compressed | `/home/sadara/.hipfire/src/crates/hipfire-arch-deepseek4/src/forward.rs:6158` |
| `HIPFIRE_DEEPSEEK4_MODEL` | Runtime variable controlling deepseek4 model in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-arch-deepseek4/examples/deepseek4_chat.rs:154` |
| `HIPFIRE_DEEPSEEK4_MOE` | Layers 0..num_hash_layers use STATIC tid2eid routing per upstream DeepSeek V4 | `/home/sadara/.hipfire/src/crates/hipfire-arch-deepseek4/src/forward.rs:6534` |
| `HIPFIRE_DEEPSEEK4_MOE_8W` | Lever 3 (HIPFIRE_DEEPSEEK4_MOE_8W=1): 8-warp variant (shares staged X | `/home/sadara/.hipfire/src/crates/hipfire-arch-deepseek4/src/forward.rs:6740` |
| `HIPFIRE_DEEPSEEK4_MOE_CND` | Lever 2 (HIPFIRE_DEEPSEEK4_MOE_CND=1): cndmask dequant on the n16 4w | `/home/sadara/.hipfire/src/crates/hipfire-arch-deepseek4/src/forward.rs:6737` |
| `HIPFIRE_DEEPSEEK4_MOE_DETERMINISTIC` | Environment toggle value controls runtime behavior | `/home/sadara/.hipfire/src/crates/hipfire-arch-deepseek4/src/forward.rs:7096` |
| `HIPFIRE_DEEPSEEK4_MOE_GROUPED` | Environment toggle value controls runtime behavior | `/home/sadara/.hipfire/src/crates/hipfire-arch-deepseek4/src/forward.rs:6679` |
| `HIPFIRE_DEEPSEEK4_MOE_GROUPED_GATE` | HIPFIRE_DEEPSEEK4_MOE_GROUPED_GATE overrides the threshold for | `/home/sadara/.hipfire/src/crates/hipfire-arch-deepseek4/src/forward.rs:6674` |
| `HIPFIRE_DEEPSEEK4_MOE_LLOYD_4W` | RECONCILED (#355 + #356): same 4w default + levers as gate_up above | `/home/sadara/.hipfire/src/crates/hipfire-arch-deepseek4/src/forward.rs:6913` |
| `HIPFIRE_DEEPSEEK4_MOE_MMQLOAD` | n32_env / cnd_env / eightw_env reused from the gate_up block above | `/home/sadara/.hipfire/src/crates/hipfire-arch-deepseek4/src/forward.rs:6921` |
| `HIPFIRE_DEEPSEEK4_MOE_N32` | Lever 1 (HIPFIRE_DEEPSEEK4_MOE_N32=1): N_TILE=32 tile-pairing on the | `/home/sadara/.hipfire/src/crates/hipfire-arch-deepseek4/src/forward.rs:6734` |
| `HIPFIRE_DEEPSEEK4_MOE_NOSYNC` | n32_env / cnd_env / eightw_env reused from the gate_up block above | `/home/sadara/.hipfire/src/crates/hipfire-arch-deepseek4/src/forward.rs:6923` |
| `HIPFIRE_DEEPSEEK4_MTP_ADDON` | Convention 1: append ".mtp-addon.hfq" (legacy) | `/home/sadara/.hipfire/src/crates/hipfire-arch-deepseek4/src/arch.rs:510` |
| `HIPFIRE_DEEPSEEK4_MTP_HEAD_HC` | Runtime variable controlling deepseek4 MTP head hc in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-arch-deepseek4/src/forward.rs:81` |
| `HIPFIRE_DEEPSEEK4_MTP_SKIP_HEAD` | 3. Batched MTP fill — single pass through the MTP layer for all | `/home/sadara/.hipfire/src/crates/hipfire-arch-deepseek4/src/forward.rs:8078` |
| `HIPFIRE_DEEPSEEK4_POST_SCALE` | Runtime variable controlling deepseek4 post scale in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-arch-deepseek4/src/forward.rs:7424` |
| `HIPFIRE_DEEPSEEK4_PP_BATCH` | on 2026-05-26). PP_BATCH sweep on the 2.1k-tok bench (3 trials/cell): | `/home/sadara/.hipfire/src/crates/hipfire-daemon/src/main.rs:12276` |
| `HIPFIRE_DEEPSEEK4_Q8_4W` | Opt out via HIPFIRE_DEEPSEEK4_Q8_4W=0 for diagnosis | `/home/sadara/.hipfire/src/crates/hipfire-arch-deepseek4/src/forward.rs:275` |
| `HIPFIRE_DEEPSEEK4_Q8_WMMA` | bench_q8_wmma_variants. Opt-out via HIPFIRE_DEEPSEEK4_Q8_WMMA=0 | `/home/sadara/.hipfire/src/crates/hipfire-arch-deepseek4/src/forward.rs:256` |
| `HIPFIRE_DEEPSEEK4_ROUTE_SCALE` | Runtime variable controlling deepseek4 route scale in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-arch-deepseek4/src/forward.rs:6554` |
| `HIPFIRE_DEEPSEEK4_SEED` | Runtime variable controlling deepseek4 seed in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-daemon/src/main.rs:17456` |
| `HIPFIRE_DEEPSEEK4_SPEC_DECODE` | Priority: 1. legacy env var → 2. generic env var → 3. stored config → default | `/home/sadara/.hipfire/src/crates/hipfire-daemon/src/main.rs:17254` |
| `HIPFIRE_DEEPSEEK4_SPEC_K` | Runtime variable controlling deepseek4 spec k in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-daemon/src/main.rs:17262` |
| `HIPFIRE_DEEPSEEK4_TEMP` | Runtime variable controlling deepseek4 temp in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-arch-deepseek4/examples/deepseek4_chat.rs:162` |
| `HIPFIRE_DEEPSEEK4_TOP_K` | for local deployment; we honor that as the default. Pure greedy | `/home/sadara/.hipfire/src/crates/hipfire-daemon/src/main.rs:17452` |
| `HIPFIRE_DEEPSEEK4_UPLOAD_EXPERTS` | without them). Opt out with "HIPFIRE_DEEPSEEK4_UPLOAD_EXPERTS=0" | `/home/sadara/.hipfire/src/crates/hipfire-arch-deepseek4/src/arch.rs:486` |
| `HIPFIRE_DEEPSEEK4_WO_MULTIROW` | Q8_0 contract: plain (non-FWHT) input. Same layout | `/home/sadara/.hipfire/src/crates/hipfire-arch-deepseek4/src/forward.rs:6295` |
| `HIPFIRE_DEEPSEEK4_WO_Q8_WMMA` | Runtime variable controlling deepseek4 wo Q8 wmma in hipfire. | `/home/sadara/.hipfire/src/crates/rdna-compute/src/dispatch.rs:48229` |
| `HIPFIRE_DELTANET_STATE` | Interprets "HIPFIRE_DELTANET_STATE" from environment to select behavior | `/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/greedy_dump_top5.rs:245` |
| `HIPFIRE_DETERMINISTIC` | Enabled when set to 1 | `/home/sadara/.hipfire/src/crates/rdna-compute/src/feature_flags.rs:256` |
| `HIPFIRE_DEVICES` | Runtime variable controlling devices in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-runtime/src/config.rs:94` |
| `HIPFIRE_DFLASH_DRAFT` | Runtime variable controlling dflash draft in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-runtime/src/config.rs:69` |
| `HIPFIRE_DFLASH_LOOP_BREAK` | Memory: HashSet<u64>, ≤ max_tokens / 12 entries (~125 for max=1500) | `/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/dflash_spec_demo.rs:1433` |
| `HIPFIRE_DFLASH_LOOP_BREAK_MAX_ESCALATIONS` | Runtime variable controlling dflash loop break max escalations in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/dflash_spec_demo.rs:1462` |
| `HIPFIRE_DFLASH_LOOP_BREAK_RECOVERY` | Runtime variable controlling dflash loop break recovery in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/dflash_spec_demo.rs:1457` |
| `HIPFIRE_DFLASH_LOOP_BREAK_RP_MAX` | Runtime variable controlling dflash loop break rp max in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/dflash_spec_demo.rs:1453` |
| `HIPFIRE_DFLASH_LOOP_BREAK_RP_STEP` | Runtime variable controlling dflash loop break rp step in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/dflash_spec_demo.rs:1449` |
| `HIPFIRE_DFLASH_LOOP_BREAK_STOP_AFTER` | Runtime variable controlling dflash loop break stop after in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/dflash_spec_demo.rs:1445` |
| `HIPFIRE_DFLASH_LOOP_BREAK_TEMP` | Runtime variable controlling dflash loop break temp in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/dflash_spec_demo.rs:1441` |
| `HIPFIRE_DFLASH_MODE` | Defaults to off when unset | `/home/sadara/.hipfire/src/crates/hipfire-runtime/src/config.rs:70` |
| `HIPFIRE_DFLASH_MOE_DRAFT_FFN_GRAPH` | Runtime variable controlling dflash moe draft ffn graph in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/speculative.rs:6962` |
| `HIPFIRE_DFLASH_MOE_VERIFY_GRAPH_LMHEAD` | Runtime variable controlling dflash moe verify graph lmhead in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/speculative.rs:6333` |
| `HIPFIRE_DFLASH_NGRAM_BLOCK` | Enabled when set to 1 | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/speculative.rs:6959` |
| `HIPFIRE_DFLASH_Q8_LMHEAD_WMMA` | Selects behavior from recognized values | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/speculative.rs:33` |
| `HIPFIRE_DFLASH_ROLLBACK_COMPARE` | Used to configure runtime execution by explicitly setting "HIPFIRE_DFLASH_ROLLBACK_COMPARE" | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/speculative.rs:12202` |
| `HIPFIRE_DFLASH_ROLLBACK_FA_RAW_ATOL` | Runtime variable controlling dflash rollback fa raw atol in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/speculative.rs:949` |
| `HIPFIRE_DFLASH_ROLLBACK_LOGIT_COMPARE_STEPS` | Used to configure runtime execution by explicitly setting "HIPFIRE_DFLASH_ROLLBACK_LOGIT_COMPARE_STEPS" | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/speculative.rs:12226` |
| `HIPFIRE_DFLASH_ROLLBACK_PREFIX_VERIFY` | Used to configure runtime execution by explicitly setting "HIPFIRE_DFLASH_ROLLBACK_PREFIX_VERIFY" | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/speculative.rs:12130` |
| `HIPFIRE_DFLASH_ROLLBACK_SERIAL_REPLAY` | Used to configure runtime execution by explicitly setting "HIPFIRE_DFLASH_ROLLBACK_SERIAL_REPLAY" | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/speculative.rs:12098` |
| `HIPFIRE_DFLASH_ROLLBACK_SERIAL_TAPE` | Used to configure runtime execution by explicitly setting "HIPFIRE_DFLASH_ROLLBACK_SERIAL_TAPE" | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/speculative.rs:12178` |
| `HIPFIRE_DFLASH_ROLLBACK_VERIFY_FRAMES` | Used to configure runtime execution by explicitly setting "HIPFIRE_DFLASH_ROLLBACK_VERIFY_FRAMES" | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/speculative.rs:12154` |
| `HIPFIRE_DFLASH_ROLLBACK_X_IN_ATOL` | Parses "HIPFIRE_DFLASH_ROLLBACK_X_IN_ATOL" with fallback defaults | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/speculative.rs:957` |
| `HIPFIRE_DFLASH_SEED_ORACLE` | Enabled when set to 1 | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/speculative.rs:7587` |
| `HIPFIRE_DFLASH_SERIAL_QKVZA_SELF_COMPARE` | Runtime variable controlling dflash serial qKVza self compare in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/qwen35.rs:13166` |
| `HIPFIRE_DFLASH_SERIAL_TAPE_X_IN_COMPARE` | Runtime variable controlling dflash serial tape x in compare in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/qwen35.rs:13171` |
| `HIPFIRE_DFLASH_SPEC_DEMO_BIN` | Runtime variable controlling dflash spec demo bin in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-runtime/src/eval_harness/mod.rs:9727` |
| `HIPFIRE_DFLASH_TRACE_EXPECTED_TOKEN` | Runtime variable controlling dflash trace expected token in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/dflash_spec_demo.rs:137` |
| `HIPFIRE_DFLASH_TRACE_POSITION` | Runtime variable controlling dflash trace position in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/dflash_spec_demo.rs:131` |
| `HIPFIRE_DFLASH_TRACE_TOKEN_INDEX` | Runtime variable controlling dflash trace token index in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/dflash_spec_demo.rs:1804` |
| `HIPFIRE_DOT2_GEMV` | Interprets "HIPFIRE_DOT2_GEMV" from environment to select behavior | `/home/sadara/.hipfire/src/crates/rdna-compute/src/feature_flags.rs:162` |
| `HIPFIRE_DOTS_OCR_BF16_RESIDUAL` | HF cast x to bf16 at vision forward entry (modeling_dots_vision.py | `/home/sadara/.hipfire/src/crates/hipfire-arch-dots-ocr/src/dots_ocr.rs:1118` |
| `HIPFIRE_DOTS_OCR_DUMP_DIR` | HIPFIRE_DOTS_OCR_DUMP_DIR=<path>: dump full per-stage tensor | `/home/sadara/.hipfire/src/crates/hipfire-arch-dots-ocr/src/dots_ocr.rs:1020` |
| `HIPFIRE_DOTS_OCR_TRACE` | HIPFIRE_DOTS_OCR_TRACE=1: sync after every step + print probe so | `/home/sadara/.hipfire/src/crates/hipfire-arch-dots-ocr/src/dots_ocr.rs:1148` |
| `HIPFIRE_DPM_WARMUP_SECS` | Runtime variable controlling dpm warmup secs in hipfire. | `/home/sadara/.hipfire/src/crates/rdna-compute/examples/bench_stream_overlap.rs:60` |
| `HIPFIRE_DRAFT_F16` | Enabled by default; set to 0 to disable | `/home/sadara/.hipfire/src/crates/hipfire-runtime/src/config.rs:71` |
| `HIPFIRE_DRAFT_GEMM_DUMP` | Enabled when set to 1 | `/home/sadara/.hipfire/src/crates/hipfire-runtime/src/config.rs:72` |
| `HIPFIRE_DRAFT_SUBPHASE` | Enabled when set to 1 | `/home/sadara/.hipfire/src/crates/hipfire-runtime/src/config.rs:73` |
| `HIPFIRE_DTOH_DUMP` | Enabled when set to 1 | `/home/sadara/.hipfire/src/crates/hip-bridge/src/ffi.rs:938` |
| `HIPFIRE_DUMMY_GENERATE_DELAY_MS` | Parses "HIPFIRE_DUMMY_GENERATE_DELAY_MS" with fallback defaults | `/home/sadara/.hipfire/src/crates/hipfire-daemon/src/main.rs:420` |
| `HIPFIRE_DUMMY_PREFILL_DELAY_MS` | Parses "HIPFIRE_DUMMY_PREFILL_DELAY_MS" with fallback defaults | `/home/sadara/.hipfire/src/crates/hipfire-daemon/src/main.rs:412` |
| `HIPFIRE_DUMP_REQUEST` | a strace. Off by default — gigantic for typical agent prompts | `/home/sadara/.hipfire/src/cli/index.ts:3863` |
| `HIPFIRE_EMIT_TOKEN_IDS` | or "\" in it would corrupt the line, breaking the client's JSONL | `/home/sadara/.hipfire/src/crates/hipfire-daemon/src/main.rs:605` |
| `HIPFIRE_EVAL_DATASET_MIRROR` | Runtime variable controlling eval dataset mirror in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-runtime/src/eval_harness/mod.rs:2550` |
| `HIPFIRE_EVAL_EVIDENCE_DIR` | Runtime variable controlling eval evidence dir in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/run.rs:413` |
| `HIPFIRE_EVAL_HIPFIRE_BIN` | Runtime variable controlling eval hipfire bin in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-runtime/src/eval_harness/mod.rs:9789` |
| `HIPFIRE_EXPERIMENTAL_BUDGET_ALERT` | cache so the model "sees" them as part of its own trajectory, | `/home/sadara/.hipfire/src/crates/hipfire-daemon/src/main.rs:10375` |
| `HIPFIRE_FLASH_PARTIALS_BATCH` | Parses "HIPFIRE_FLASH_PARTIALS_BATCH" with fallback defaults | `/home/sadara/.hipfire/src/crates/hipfire-runtime/src/config.rs:83` |
| `HIPFIRE_FORCE_A3B_EVICTION` | Runtime variable controlling force a3b eviction in hipfire. | `/home/sadara/.hipfire/src/cli/index.ts:10164` |
| `HIPFIRE_FP16` | escape hatch to the LA qkvza projection while debugging DFlash | `/home/sadara/.hipfire/src/crates/rdna-compute/src/feature_flags.rs:170` |
| `HIPFIRE_FP8_WMMA` | Interprets "HIPFIRE_FP8_WMMA" from environment to select behavior | `/home/sadara/.hipfire/src/crates/rdna-compute/src/feature_flags.rs:161` |
| `HIPFIRE_FUSED_HFQ4_2ROW_GFX1151` | Selects behavior from recognized values | `/home/sadara/.hipfire/src/crates/rdna-compute/src/dispatch.rs:24505` |
| `HIPFIRE_GATE_UP_VARIANT` | Runtime variable controlling gate up variant in hipfire. | `/home/sadara/.hipfire/src/crates/rdna-compute/src/feature_flags.rs:216` |
| `HIPFIRE_GDN_Q8_REG_GFX1151` | HIPFIRE_GDN_Q8_REG_GFX1151=1 enables the gfx1151 register-state | `/home/sadara/.hipfire/src/crates/rdna-compute/src/dispatch.rs:38878` |
| `HIPFIRE_GEMM_DUMP` | Enabled when set to 1 | `/home/sadara/.hipfire/src/crates/rdna-compute/src/feature_flags.rs:255` |
| `HIPFIRE_GEMV_ROWS` | Runtime variable controlling gemv rows in hipfire. | `/home/sadara/.hipfire/src/crates/rdna-compute/src/feature_flags.rs:142` |
| `HIPFIRE_GEN` | Runtime variable controlling gen in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/a3b_multiturn_oneshot.rs:25` |
| `HIPFIRE_GENERATE_BATCH_PREFILL_DEBUG_SAMPLE` | Runtime variable controlling generate batch prefill debug sample in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-daemon/src/main.rs:8083` |
| `HIPFIRE_GFX942_GEMV_V3` | Interprets "HIPFIRE_GFX942_GEMV_V3" from environment to select behavior | `/home/sadara/.hipfire/src/crates/rdna-compute/src/feature_flags.rs:218` |
| `HIPFIRE_GFX942_MFMA_PREFILL` | Interprets "HIPFIRE_GFX942_MFMA_PREFILL" from environment to select behavior | `/home/sadara/.hipfire/src/crates/rdna-compute/src/feature_flags.rs:221` |
| `HIPFIRE_GFX942_RMSNORM_SPLIT` | Environment toggle value controls runtime behavior | `/home/sadara/.hipfire/src/crates/rdna-compute/src/feature_flags.rs:220` |
| `HIPFIRE_GPTQ_DAMPING` | Inject env override since the quantizer reads it at fn entry | `/home/sadara/.hipfire/src/crates/hipfire-quantize/src/main.rs:8676` |
| `HIPFIRE_GPU_SLAB_LOAD` | Runtime variable controlling gpu slab load in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/qwen35.rs:3881` |
| `HIPFIRE_GPU_SLAB_MIB` | Parses "HIPFIRE_GPU_SLAB_MIB" with fallback defaults | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/qwen35.rs:3561` |
| `HIPFIRE_GPU_TOPK` | HIPFIRE_GPU_TOPK=1 enables the GPU topk_logits_f32 kernel + CPU | `/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/infer_qwen35.rs:221` |
| `HIPFIRE_GQA_CHUNK` | Runtime variable controlling gqa chunk in hipfire. | `/home/sadara/.hipfire/src/crates/rdna-compute/src/dispatch.rs:31456` |
| `HIPFIRE_GQA_FUSED` | Fused variant (opt-in via HIPFIRE_GQA_FUSED=1): single launch | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen2/src/qwen2.rs:1061` |
| `HIPFIRE_GRAPH` | Used to configure runtime execution by explicitly setting "HIPFIRE_GRAPH" | `/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/prefill_microbench.rs:119` |
| `HIPFIRE_GRAPH_MOE` | - gfx11 (RDNA3 / 3.5): default-ON. +0.6-0.7% decode on 9B and | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/qwen35.rs:8286` |
| `HIPFIRE_GRAPH_PREFILL` | HIPFIRE_GRAPH_PREFILL=1: route the timed prefill loop through | `/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/bench_qwen35_speed.rs:186` |
| `HIPFIRE_HAVE_2_GPU` | Environment toggle value controls runtime behavior | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/tests/pp_parity.rs:193` |
| `HIPFIRE_HETERO_DIFF` | above (#352's GPU greedy-accept path doesn't materialize it), | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/mtp_spec.rs:2956` |
| `HIPFIRE_HFQ4G128_MMQ` | Environment toggle value controls runtime behavior | `/home/sadara/.hipfire/src/crates/rdna-compute/src/feature_flags.rs:211` |
| `HIPFIRE_HFQ4G256_MMQ_GFX1151` | Selects behavior from recognized values | `/home/sadara/.hipfire/src/crates/rdna-compute/src/dispatch.rs:24488` |
| `HIPFIRE_HFQ4_GATE_UP_FAST` | HIPFIRE_HFQ4_GATE_UP_FAST=0 narrows the escape hatch to the | `/home/sadara/.hipfire/src/crates/rdna-compute/src/feature_flags.rs:184` |
| `HIPFIRE_HFQ4_MMQ_GFX906_Y64` | Runtime variable controlling hfQ4 mmq gfx906 y64 in hipfire. | `/home/sadara/.hipfire/src/crates/rdna-compute/src/feature_flags.rs:214` |
| `HIPFIRE_HFQ4_QKVZA_FAST` | HIPFIRE_HFQ4_QKVZA_FAST=0 narrows the FP16/WMMA/dot2 prefill | `/home/sadara/.hipfire/src/crates/rdna-compute/src/feature_flags.rs:176` |
| `HIPFIRE_HFQ4_QKV_FAST` | HIPFIRE_HFQ4_QKV_FAST=0 narrows the escape hatch to the | `/home/sadara/.hipfire/src/crates/rdna-compute/src/feature_flags.rs:180` |
| `HIPFIRE_HFQ4_RESIDUAL_FAST` | HIPFIRE_HFQ4_RESIDUAL_FAST=0 narrows the residual-projection | `/home/sadara/.hipfire/src/crates/rdna-compute/src/feature_flags.rs:188` |
| `HIPFIRE_HFQ6_QKVZA_4W` | Interprets "HIPFIRE_HFQ6_QKVZA_4W" from environment to select behavior | `/home/sadara/.hipfire/src/crates/rdna-compute/src/dispatch.rs:26592` |
| `HIPFIRE_HFQ6_QKV_4W` | Interprets "HIPFIRE_HFQ6_QKV_4W" from environment to select behavior | `/home/sadara/.hipfire/src/crates/rdna-compute/src/dispatch.rs:27345` |
| `HIPFIRE_HFQ6_RESIDUAL_4W` | Interprets "HIPFIRE_HFQ6_RESIDUAL_4W" from environment to select behavior | `/home/sadara/.hipfire/src/crates/rdna-compute/src/dispatch.rs:26199` |
| `HIPFIRE_HIPCC_EXTRA_FLAGS` | Parses "HIPFIRE_HIPCC_EXTRA_FLAGS" with fallback defaults | `/home/sadara/.hipfire/src/crates/rdna-compute/src/feature_flags.rs:281` |
| `HIPFIRE_HIP_WAIT` | Runtime variable controlling hip wait in hipfire. | `/home/sadara/.hipfire/src/crates/rdna-compute/src/dispatch.rs:716` |
| `HIPFIRE_HOST_PROFILE_BIN` | Runtime variable controlling host profile bin in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-runtime/src/eval_harness/mod.rs:9804` |
| `HIPFIRE_HOST_TIMING` | HIPFIRE_HOST_TIMING=1: dump per-cycle host-side wall-clock breakdown | `/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/dflash_spec_demo.rs:1732` |
| `HIPFIRE_JINJA_CHAT` | Enabled when set to 1 | `/home/sadara/.hipfire/src/crates/hipfire-daemon/src/main.rs:18483` |
| `HIPFIRE_KERNEL_CACHE` | HIPFIRE_KERNEL_CACHE=/tmp/hipfire_kernels if tmpfs speed matters | `/home/sadara/.hipfire/src/crates/rdna-compute/src/compiler.rs:213` |
| `HIPFIRE_KLD_DIRECT_F16KV_ATTN` | Used to configure runtime execution by explicitly setting "HIPFIRE_KLD_DIRECT_F16KV_ATTN" | `/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/build_kld_ref_hipfire.rs:699` |
| `HIPFIRE_KLD_DIRECT_WMMA_ATTN` | Used to configure runtime execution by explicitly setting "HIPFIRE_KLD_DIRECT_WMMA_ATTN" | `/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/build_kld_ref_hipfire.rs:698` |
| `HIPFIRE_KLD_FP32_GQA4_ATTN` | Used to configure runtime execution by explicitly setting "HIPFIRE_KLD_FP32_GQA4_ATTN" | `/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/build_kld_ref_hipfire.rs:702` |
| `HIPFIRE_KLD_GPU_TOPK` | Enabled by default; set to 0 to disable | `/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/build_kld_ref_hipfire.rs:956` |
| `HIPFIRE_KLD_GRAPH` | Enabled by default; set to 0 to disable | `/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/build_kld_ref_hipfire.rs:677` |
| `HIPFIRE_KLD_NO_ACTIVE_STREAM` | Runtime variable controlling kld no active stream in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/build_kld_ref_hipfire.rs:291` |
| `HIPFIRE_KLD_PREFILL_ONLY` | Enabled when set to 1 | `/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/build_kld_ref_hipfire.rs:844` |
| `HIPFIRE_KLD_SOURCE_SHA256` | Enabled when set to 1 | `/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/build_kld_ref_hipfire.rs:840` |
| `HIPFIRE_KV_MODE` | Runtime variable controlling KV mode in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-runtime/src/speed_bench.rs:119` |
| `HIPFIRE_KV_PHYSICAL_CAP` | Parses "HIPFIRE_KV_PHYSICAL_CAP" with fallback defaults | `/home/sadara/.hipfire/src/crates/hipfire-daemon/src/main.rs:12051` |
| `HIPFIRE_LFM2_CAPTURE_POSTMIXER` | Runtime variable controlling lfm2 capture postmixer in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-arch-lfm2moe/src/forward.rs:132` |
| `HIPFIRE_LFM2_EXPERT_MQ6` | opt-in via HIPFIRE_LFM2_EXPERT_MQ6 for higher quality), else mq4 | `/home/sadara/.hipfire/src/crates/hipfire-quantize/src/main.rs:6732` |
| `HIPFIRE_LFM2_GRAPH` | Interprets "HIPFIRE_LFM2_GRAPH" from environment to select behavior | `/home/sadara/.hipfire/src/crates/hipfire-arch-lfm2moe/src/forward.rs:57` |
| `HIPFIRE_LFM2_PROJ_MQ4` | Runtime variable controlling lfm2 proj mQ4 in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-quantize/src/main.rs:6813` |
| `HIPFIRE_LFM2_PROJ_MQ6` | MQ6 variant (HIPFIRE_LFM2_PROJ_MQ6=1) is the lower-quality-loss | `/home/sadara/.hipfire/src/crates/hipfire-quantize/src/main.rs:6812` |
| `HIPFIRE_LLOYD_FORCE_BASELINE` | Runtime variable controlling lloyd force baseline in hipfire. | `/home/sadara/.hipfire/src/crates/rdna-compute/src/feature_flags.rs:272` |
| `HIPFIRE_LLOYD_GFX12` | Used to configure runtime execution by explicitly setting "HIPFIRE_LLOYD_GFX12" | `/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/prefill_microbench.rs:140` |
| `HIPFIRE_LLOYD_K3` | Fallback to HFQ2-G128 for non-256-aligned (no rotation) | `/home/sadara/.hipfire/src/crates/hipfire-quantize/src/main.rs:8106` |
| `HIPFIRE_LLOYD_MB4` | Force MB4=0 to skip the size-gated routing | `/home/sadara/.hipfire/src/crates/rdna-compute/examples/test_gemm_mq4g256_lloyd_residual_wmma.rs:274` |
| `HIPFIRE_LM_HEAD_F16` | Runtime variable controlling lm head f16 in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-runtime/src/config.rs:100` |
| `HIPFIRE_LM_HEAD_OVERWRITE` | Environment toggle value controls runtime behavior | `/home/sadara/.hipfire/src/crates/rdna-compute/src/feature_flags.rs:192` |
| `HIPFIRE_LM_HEAD_WMMA` | Runtime variable controlling lm head wmma in hipfire. | `/home/sadara/.hipfire/src/crates/rdna-compute/src/feature_flags.rs:190` |
| `HIPFIRE_LOAD_TRANSPORT` | Selects behavior from recognized values | `/home/sadara/.hipfire/src/crates/hipfire-runtime/src/weight_pager.rs:768` |
| `HIPFIRE_LOCAL` | API — saves the 2-5s cold-start cost of loading the model every invocation | `/home/sadara/.hipfire/src/cli/index.ts:1735` |
| `HIPFIRE_MAX_RESIDENT_WORKERS` | Runtime variable controlling max resident workers in hipfire. | `/home/sadara/.hipfire/src/cli/index.ts:2019` |
| `HIPFIRE_MAX_SEQ` | Runtime variable controlling max seq in hipfire. | `/home/sadara/.hipfire/src/cli/index.ts:719` |
| `HIPFIRE_MEMSET_DUMP` | Enabled when set to 1 | `/home/sadara/.hipfire/src/crates/hip-bridge/src/ffi.rs:1012` |
| `HIPFIRE_MINIMAX_CAPTURE_POSTATTN` | Runtime variable controlling minimax capture postattn in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-arch-minimax/src/forward.rs:200` |
| `HIPFIRE_MINIMAX_DOWN_FORMAT` | Runtime variable controlling minimax down format in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-quantize/src/main.rs:7047` |
| `HIPFIRE_MINIMAX_ENABLE_DOWN_AWQ` | down-AWQ harmful (shared s_down bad approx); opt-in | `/home/sadara/.hipfire/src/crates/hipfire-arch-minimax/src/minimax.rs:410` |
| `HIPFIRE_MINIMAX_EXPERT_MQ2L` | dispatches expert dtype per-layer (experts[0].gpu_dtype), so the model | `/home/sadara/.hipfire/src/crates/hipfire-quantize/src/main.rs:7026` |
| `HIPFIRE_MINIMAX_EXPERT_MQ3L` | dispatches expert dtype per-layer (experts[0].gpu_dtype), so the model | `/home/sadara/.hipfire/src/crates/hipfire-quantize/src/main.rs:7028` |
| `HIPFIRE_MINIMAX_EXPERT_MQ6` | _MQ6 hold comma-separated layer ranges ("12-45,50") whose experts are | `/home/sadara/.hipfire/src/crates/hipfire-quantize/src/main.rs:7024` |
| `HIPFIRE_MINIMAX_GRAPH` | Default OFF — measured only +1.0% on gfx1151 (the sole arch MiniMax fits); | `/home/sadara/.hipfire/src/crates/hipfire-arch-minimax/src/forward.rs:107` |
| `HIPFIRE_MMQ` | Selects behavior from recognized values | `/home/sadara/.hipfire/src/crates/rdna-compute/src/feature_flags.rs:164` |
| `HIPFIRE_MMQ_DIAG_QUANTIZE_ONLY` | Runtime variable controlling mmq diag quantize only in hipfire. | `/home/sadara/.hipfire/src/crates/rdna-compute/src/feature_flags.rs:203` |
| `HIPFIRE_MMQ_SCREEN` | Runtime variable controlling mmq screen in hipfire. | `/home/sadara/.hipfire/src/crates/rdna-compute/src/feature_flags.rs:195` |
| `HIPFIRE_MMQ_SCREEN_THRESHOLD` | Runtime variable controlling mmq screen threshold in hipfire. | `/home/sadara/.hipfire/src/crates/rdna-compute/src/feature_flags.rs:199` |
| `HIPFIRE_MODEL` | Pre-warm: load default model and compile kernels before accepting requests | `/home/sadara/.hipfire/src/cli/index.ts:3313` |
| `HIPFIRE_MOE_GROUPED_GEMM` | Runtime variable controlling moe grouped gemm in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/qwen35.rs:14422` |
| `HIPFIRE_MOE_GROUPED_I8` | HIPFIRE_MOE_GROUPED_I8_K8: gfx1151 HFQ4 grouped MoE k8 control; | `/home/sadara/.hipfire/src/crates/rdna-compute/src/feature_flags.rs:223` |
| `HIPFIRE_MOE_GROUPED_I8_4W` | default on for gfx1151, set to 0 to test k4 or base k2 variants | `/home/sadara/.hipfire/src/crates/rdna-compute/src/feature_flags.rs:228` |
| `HIPFIRE_MOE_GROUPED_I8_K4` | HIPFIRE_MOE_GROUPED_I8_K4: opt-in gfx1151 HFQ4 grouped MoE k4 | `/home/sadara/.hipfire/src/crates/rdna-compute/src/feature_flags.rs:239` |
| `HIPFIRE_MOE_GROUPED_I8_K4_GFX12` | HIPFIRE_MOE_HFQ6_V2: opt-in HFQ6 grouped MoE v2 path; on | `/home/sadara/.hipfire/src/crates/rdna-compute/src/feature_flags.rs:240` |
| `HIPFIRE_MOE_GROUPED_I8_K8` | variant; use with HIPFIRE_MOE_GROUPED_I8_K8=0. Default off | `/home/sadara/.hipfire/src/crates/rdna-compute/src/feature_flags.rs:234` |
| `HIPFIRE_MOE_GROUPED_M2` | HIPFIRE_MOE_HFQ6_V2: opt-in HFQ6 grouped MoE v2 path; on | `/home/sadara/.hipfire/src/crates/rdna-compute/src/feature_flags.rs:242` |
| `HIPFIRE_MOE_HFQ6_4W` | Environment toggle value controls runtime behavior | `/home/sadara/.hipfire/src/crates/rdna-compute/src/feature_flags.rs:250` |
| `HIPFIRE_MOE_HFQ6_V2` | HIPFIRE_MOE_HFQ6_V2: opt-in HFQ6 grouped MoE v2 path; on | `/home/sadara/.hipfire/src/crates/rdna-compute/src/feature_flags.rs:246` |
| `HIPFIRE_MOE_INDEXED_2ROW_GFX1151` | Opt-in ("1") gfx1151 two-row indexed MoE HFQ4 decode probe for gate/up and expanded down; default off after flat A3B measurements | `/home/sadara/.hipfire/src/crates/rdna-compute/src/dispatch.rs:24523` |
| `HIPFIRE_MOE_MQ2L_N32_GFX1151` | Runtime variable controlling moe mq2l n32 gfx1151 in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/qwen35.rs:13709` |
| `HIPFIRE_MOE_PARO_I8` | Runtime variable controlling moe paro i8 in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/qwen35.rs:15035` |
| `HIPFIRE_MOE_PARO_I8_K8` | Runtime variable controlling moe paro i8 k8 in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/qwen35.rs:15039` |
| `HIPFIRE_MQ3_MB4` | Used to configure runtime execution by explicitly setting "HIPFIRE_MQ3_MB4" | `/home/sadara/.hipfire/src/crates/rdna-compute/examples/test_gemm_hfq3g256_wmma.rs:92` |
| `HIPFIRE_MTP_DEVICE_TOKEN_CHAIN` | Default on: this path is token-identical in greedy mode and removes the | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/mtp_spec.rs:82` |
| `HIPFIRE_MTP_GPU_ACCEPT` | Default on for greedy device-token-chain MTP: candidates and verify | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/mtp_spec.rs:107` |
| `HIPFIRE_MTP_K` | Runtime variable controlling MTP k in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-runtime/src/config.rs:103` |
| `HIPFIRE_MTP_MODE` | Defaults to auto when unset | `/home/sadara/.hipfire/src/crates/hipfire-runtime/src/config.rs:102` |
| `HIPFIRE_MTP_PROPOSAL_GRAPH` | Runtime variable controlling MTP proposal graph in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/mtp_spec.rs:160` |
| `HIPFIRE_MTP_Q8_VERIFY_WMMA` | Runtime variable controlling MTP Q8 verify wmma in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/mtp_spec.rs:131` |
| `HIPFIRE_MTP_SMOKE_HEAD` | Runtime variable controlling MTP smoke head in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/examples/mtp_head_smoke.rs:28` |
| `HIPFIRE_MTP_SMOKE_TRUNK` | Runtime variable controlling MTP smoke trunk in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/examples/mtp_head_smoke.rs:24` |
| `HIPFIRE_MTP_SNAPSHOT_OVERLAP` | Default off: current gfx1201 benches show this stream split regresses, | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/mtp_spec.rs:94` |
| `HIPFIRE_MW16` | Runtime variable controlling mw16 in hipfire. | `/home/sadara/.hipfire/src/crates/rdna-compute/src/feature_flags.rs:257` |
| `HIPFIRE_NGRAM_LOOP_THRESHOLD` | Parses "HIPFIRE_NGRAM_LOOP_THRESHOLD" with fallback defaults | `/home/sadara/.hipfire/src/crates/hipfire-runtime/src/config.rs:86` |
| `HIPFIRE_NGRAM_WINDOW` | Runtime variable controlling ngram window in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-runtime/src/config.rs:90` |
| `HIPFIRE_NORMALIZE_PROMPT` | Opt-out must skip CRLF/NBSP/trailing-ws too, not just newline collapse | `/home/sadara/.hipfire/src/crates/hipfire-runtime/src/tokenizer.rs:2206` |
| `HIPFIRE_NO_PID_FILE` | HIPFIRE_NO_PID_FILE=1 suppresses the write — used by "hipfire chat" when it | `/home/sadara/.hipfire/src/cli/index.ts:1909` |
| `HIPFIRE_NPU_ATTN_GATE_CONFIGS` | Runtime variable controlling npu attn gate configs in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/build.rs:193` |
| `HIPFIRE_NPU_HEADNORM_CONFIGS` | Parse "n_heads:n_kv_heads:head_dim" tuples (n_rot field ignored if present) | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/build.rs:156` |
| `HIPFIRE_NPU_HEADNORM_ROPE_CONFIGS` | Runtime variable controlling npu headnorm rope configs in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/build.rs:273` |
| `HIPFIRE_NPU_HIDDEN_SIZES` | Runtime variable controlling npu hidden sizes in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/build.rs:54` |
| `HIPFIRE_NPU_PYTHON` | Runtime variable controlling npu python in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/build.rs:605` |
| `HIPFIRE_NPU_RMSNORM_SIZES` | Runtime variable controlling npu rmsnorm sizes in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/build.rs:89` |
| `HIPFIRE_NPU_ROPE_CONFIGS` | Parse "n_heads:n_kv_heads:head_dim:n_rot" tuples | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/build.rs:118` |
| `HIPFIRE_NPU_SOFTMAX_CONFIGS` | Parse "n_heads:ctx_len1+ctx_len2+..." entries | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/build.rs:232` |
| `HIPFIRE_NPU_TARGETS` | Runtime variable controlling npu targets in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/build.rs:60` |
| `HIPFIRE_PAGED_MOE_DEBUG` | Enabled when set to 1 | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/qwen35.rs:6410` |
| `HIPFIRE_PARO_BATCHED` | Runtime variable controlling paro batched in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/qwen35.rs:13637` |
| `HIPFIRE_PARO_FA3_FUSED` | Runtime variable controlling paro fa3 fused in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/qwen35.rs:24915` |
| `HIPFIRE_PARO_FUSED_PACK2` | Runtime variable controlling paro fused pack2 in hipfire. | `/home/sadara/.hipfire/src/crates/rdna-compute/src/dispatch.rs:4899` |
| `HIPFIRE_PARO_FUSE_RMSNORM` | time per call. Net loss on every site. Default OFF; explicit opt-in for | `/home/sadara/.hipfire/src/crates/hipfire-runtime/src/llama.rs:980` |
| `HIPFIRE_PARO_GATE_UP_FUSED` | Runtime variable controlling paro gate up fused in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/qwen35.rs:24516` |
| `HIPFIRE_PARO_LA2_FUSED` | Runtime variable controlling paro la2 fused in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/qwen35.rs:24629` |
| `HIPFIRE_PARO_LA4_FUSED` | Runtime variable controlling paro la4 fused in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/qwen35.rs:24625` |
| `HIPFIRE_PARO_LA_GATES_MQ4G128` | Used to configure runtime execution by explicitly setting "HIPFIRE_PARO_LA_GATES_MQ4G128" | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/paro_la_gates_codec.rs:325` |
| `HIPFIRE_PARO_PACK1` | Runtime variable controlling paro pack1 in hipfire. | `/home/sadara/.hipfire/src/crates/rdna-compute/src/dispatch.rs:4688` |
| `HIPFIRE_PARO_PACK2` | Runtime variable controlling paro pack2 in hipfire. | `/home/sadara/.hipfire/src/crates/rdna-compute/src/dispatch.rs:4691` |
| `HIPFIRE_PARO_PACK4` | Runtime variable controlling paro pack4 in hipfire. | `/home/sadara/.hipfire/src/crates/rdna-compute/src/dispatch.rs:4694` |
| `HIPFIRE_PARO_PREROTATE` | Runtime variable controlling paro prerotate in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-runtime/src/llama.rs:1548` |
| `HIPFIRE_PARO_SHARED_PAIRS` | Runtime variable controlling paro shared pairs in hipfire. | `/home/sadara/.hipfire/src/crates/rdna-compute/src/dispatch.rs:4893` |
| `HIPFIRE_PARO_SMALL_DIRECT` | Runtime variable controlling paro small direct in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-runtime/src/llama.rs:624` |
| `HIPFIRE_PARO_SWIGLU_FUSED` | Runtime variable controlling paro swiglu fused in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-runtime/src/llama.rs:1565` |
| `HIPFIRE_PERF_BASELINE` | Runtime variable controlling perf baseline in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-runtime/src/eval_harness/mod.rs:6771` |
| `HIPFIRE_PERF_BASELINE_DIR` | Runtime variable controlling perf baseline dir in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-runtime/src/eval_harness/mod.rs:6784` |
| `HIPFIRE_PFLASH_DEBUG` | Runtime variable controlling pflash debug in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-daemon/src/main.rs:15768` |
| `HIPFIRE_PFLASH_DRAFTER_KV` | Selects behavior from recognized values | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/pflash.rs:533` |
| `HIPFIRE_PFLASH_DRAFTER_STATE` | Hybrid drafter only stores K (and V for chat-path) at | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/pflash.rs:620` |
| `HIPFIRE_PFLASH_NIAH_BENCH_BIN` | Runtime variable controlling pflash niah bench bin in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-runtime/src/eval_harness/mod.rs:9759` |
| `HIPFIRE_PFLASH_SCORE_LAYER` | Parses "HIPFIRE_PFLASH_SCORE_LAYER" with fallback defaults | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/pflash.rs:1081` |
| `HIPFIRE_PP_DFLASH` | Environment toggle value controls runtime behavior | `/home/sadara/.hipfire/src/crates/hipfire-daemon/src/main.rs:9840` |
| `HIPFIRE_PP_LAYERS` | Runtime variable controlling pp layers in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/pp_overlap_microbench.rs:121` |
| `HIPFIRE_PP_PARITY_MODEL` | Runtime variable controlling pp parity model in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/tests/pp_parity.rs:197` |
| `HIPFIRE_PP_PFLASH` | Environment toggle value controls runtime behavior | `/home/sadara/.hipfire/src/crates/hipfire-daemon/src/main.rs:9858` |
| `HIPFIRE_PREFILL_ALPHA` | Runtime variable controlling prefill alpha in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/pflash.rs:124` |
| `HIPFIRE_PREFILL_BATCHED` | Enabled by default; set to 0 to disable | `/home/sadara/.hipfire/src/crates/hipfire-runtime/src/config.rs:82` |
| `HIPFIRE_PREFILL_BLOCK` | Runtime variable controlling prefill block in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/pflash.rs:144` |
| `HIPFIRE_PREFILL_COMPRESSION` | Runtime variable controlling prefill compression in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/pflash.rs:104` |
| `HIPFIRE_PREFILL_DRAFTER` | Runtime variable controlling prefill drafter in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/pflash.rs:157` |
| `HIPFIRE_PREFILL_KEEP_RATIO` | Runtime variable controlling prefill keep ratio in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/pflash.rs:114` |
| `HIPFIRE_PREFILL_MAX_BATCH` | Runtime variable controlling prefill max batch in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/build_kld_ref_hipfire.rs:1264` |
| `HIPFIRE_PREFILL_MAX_LAYER` | Runtime variable controlling prefill max layer in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/qwen35.rs:12519` |
| `HIPFIRE_PREFILL_MIN_KEEP` | Runtime variable controlling prefill min keep in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/pflash.rs:129` |
| `HIPFIRE_PREFILL_PROFILE` | Enabled when set to 1 | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/pflash.rs:154` |
| `HIPFIRE_PREFILL_RECENT` | Runtime variable controlling prefill recent in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/pflash.rs:139` |
| `HIPFIRE_PREFILL_REUSE_PBS` | Used to configure runtime execution by explicitly setting "HIPFIRE_PREFILL_REUSE_PBS" | `/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/prefill_microbench.rs:121` |
| `HIPFIRE_PREFILL_SINK` | Runtime variable controlling prefill sink in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/pflash.rs:134` |
| `HIPFIRE_PREFILL_SPARSE_THRESHOLD` | Runtime variable controlling prefill sparse threshold in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/pflash.rs:149` |
| `HIPFIRE_PREFILL_STOP_AFTER_LA_LAYER` | Parses "HIPFIRE_PREFILL_STOP_AFTER_LA_LAYER" with fallback defaults | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/qwen35.rs:15295` |
| `HIPFIRE_PREFILL_STOP_STAGE` | Runtime variable controlling prefill stop stage in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/qwen35.rs:15301` |
| `HIPFIRE_PREFILL_STOP_STAGE_LAYER` | Parses "HIPFIRE_PREFILL_STOP_STAGE_LAYER" with fallback defaults | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/qwen35.rs:15298` |
| `HIPFIRE_PREFILL_THRESHOLD` | Runtime variable controlling prefill threshold in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/pflash.rs:109` |
| `HIPFIRE_PREFIX_BOUNDARY_CHECKPOINTS` | Interprets "HIPFIRE_PREFIX_BOUNDARY_CHECKPOINTS" from environment to select behavior | `/home/sadara/.hipfire/src/crates/hipfire-daemon/src/main.rs:6791` |
| `HIPFIRE_PROFILE` | HIPFIRE_PROFILE=1 + HIPFIRE_PROFILE_CYCLES=N: per-kernel profiling | `/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/mtp_only_demo.rs:495` |
| `HIPFIRE_PROFILE_CYCLES` | Runtime variable controlling profile cycles in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/mtp_only_demo.rs:496` |
| `HIPFIRE_PROFILE_DECODE` | Enabled when set to 1 | `/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/bench_qwen35_speed.rs:559` |
| `HIPFIRE_PROMPT_HEAT_JSON` | Enabled when set to 1 | `/home/sadara/.hipfire/src/crates/hipfire-runtime/src/config.rs:58` |
| `HIPFIRE_PROMPT_HEAT_LIMIT` | Runtime variable controlling prompt heat limit in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-runtime/src/config.rs:59` |
| `HIPFIRE_PROMPT_TOKEN_HEAT` | Enabled when set to 1 | `/home/sadara/.hipfire/src/crates/hipfire-runtime/src/config.rs:56` |
| `HIPFIRE_Q8_BATCHED_LEGACY` | Environment toggle value controls runtime behavior | `/home/sadara/.hipfire/src/crates/rdna-compute/src/feature_flags.rs:258` |
| `HIPFIRE_Q8_DP4A` | (A2) dp4a wave32 (gfx906 v_dot4_i32_i8 Q·K) | `/home/sadara/.hipfire/src/crates/rdna-compute/examples/q8_batched_attn_microbench.rs:185` |
| `HIPFIRE_Q8_DP4A_W64` | (A3) dp4a WAVE64 (full 64-lane utilization) | `/home/sadara/.hipfire/src/crates/rdna-compute/examples/q8_batched_attn_microbench.rs:196` |
| `HIPFIRE_Q8_FA_ATTENTION_IGNORE_TREE_BIAS` | Interprets "HIPFIRE_Q8_FA_ATTENTION_IGNORE_TREE_BIAS" from environment to select behavior | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/qwen35.rs:11886` |
| `HIPFIRE_Q8_FA_ATTENTION_ROW_LOOP` | Interprets "HIPFIRE_Q8_FA_ATTENTION_ROW_LOOP" from environment to select behavior | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/qwen35.rs:11850` |
| `HIPFIRE_Q8_FA_ATTENTION_SCALAR_LOOP` | Interprets "HIPFIRE_Q8_FA_ATTENTION_SCALAR_LOOP" from environment to select behavior | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/qwen35.rs:11862` |
| `HIPFIRE_Q8_FA_ATTENTION_SERIAL_KV_LOOP` | Interprets "HIPFIRE_Q8_FA_ATTENTION_SERIAL_KV_LOOP" from environment to select behavior | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/qwen35.rs:11874` |
| `HIPFIRE_Q8_GATE_UP_4W` | Disabled when set to 0 | `/home/sadara/.hipfire/src/crates/rdna-compute/examples/test_gemm_q8_gate_up_wmma.rs:130` |
| `HIPFIRE_Q8_GATE_UP_BENCH` | Enabled when set to 1 | `/home/sadara/.hipfire/src/crates/rdna-compute/examples/test_gemm_q8_gate_up_wmma.rs:129` |
| `HIPFIRE_Q8_GDN_VERIFY_PER_TOKEN` | Interprets "HIPFIRE_Q8_GDN_VERIFY_PER_TOKEN" from environment to select behavior | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/qwen35.rs:11898` |
| `HIPFIRE_Q8_GDN_VERIFY_SERIAL_FRAMES` | Interprets "HIPFIRE_Q8_GDN_VERIFY_SERIAL_FRAMES" from environment to select behavior | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/qwen35.rs:11910` |
| `HIPFIRE_Q8_TOKPAR` | (A) NEW token-parallel | `/home/sadara/.hipfire/src/crates/rdna-compute/examples/q8_batched_attn_microbench.rs:174` |
| `HIPFIRE_Q8_WMMA_4W` | Environment toggle value controls runtime behavior | `/home/sadara/.hipfire/src/crates/rdna-compute/src/dispatch.rs:28958` |
| `HIPFIRE_Q8_WMMA_X64` | Environment toggle value controls runtime behavior | `/home/sadara/.hipfire/src/crates/rdna-compute/src/dispatch.rs:28769` |
| `HIPFIRE_QA_KV_MODES` | Defaults to q8,asym4,asym3,asym2 when unset | `/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/test_inferenceQA.rs:761` |
| `HIPFIRE_QUANT_DIAG_PATH` | Runtime variable controlling quant diag path in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-quantize/src/main.rs:10401` |
| `HIPFIRE_QUANT_THREADS` | Runtime variable controlling quant threads in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-quantize/src/main.rs:5463` |
| `HIPFIRE_QWEN35_DECODE_BATCH` | Defaults to auto when unset | `/home/sadara/.hipfire/src/crates/hipfire-daemon/src/main.rs:8426` |
| `HIPFIRE_QWEN35_DECODE_BATCH_MAX` | Runtime variable controlling qwen35 decode batch max in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-daemon/src/main.rs:8558` |
| `HIPFIRE_QWEN35_DECODE_INTERNAL_PARITY` | Interprets "HIPFIRE_QWEN35_DECODE_INTERNAL_PARITY" from environment to select behavior | `/home/sadara/.hipfire/src/crates/hipfire-daemon/src/main.rs:8577` |
| `HIPFIRE_QWEN35_DECODE_NATIVE_MULTIROW` | Interprets "HIPFIRE_QWEN35_DECODE_NATIVE_MULTIROW" from environment to select behavior | `/home/sadara/.hipfire/src/crates/hipfire-daemon/src/main.rs:8568` |
| `HIPFIRE_QWEN35_EXPERT_CACHE_MB` | Parses "HIPFIRE_QWEN35_EXPERT_CACHE_MB" with fallback defaults | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/qwen35.rs:688` |
| `HIPFIRE_QWEN35_EXPERT_CACHE_TRACE` | Interprets "HIPFIRE_QWEN35_EXPERT_CACHE_TRACE" from environment to select behavior | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/qwen35.rs:4152` |
| `HIPFIRE_QWEN35_FFN_BF16` | Selects behavior from recognized values | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/ffn_bf16.rs:186` |
| `HIPFIRE_QWEN35_FFN_BF16_LAYER` | Selects behavior from recognized values | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/ffn_bf16.rs:200` |
| `HIPFIRE_QWEN35_FFN_BF16_TRACE` | Parses "HIPFIRE_QWEN35_FFN_BF16_TRACE" with fallback defaults | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/ffn_bf16.rs:212` |
| `HIPFIRE_QWEN35_FINITE_TRACE` | Runtime variable controlling qwen35 finite trace in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/qwen35.rs:13121` |
| `HIPFIRE_QWEN35_PAGED_EXPERTS` | Interprets "HIPFIRE_QWEN35_PAGED_EXPERTS" from environment to select behavior | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/qwen35.rs:3595` |
| `HIPFIRE_QWEN35_PREFILL_SESSION_BATCH` | Runtime variable controlling qwen35 prefill session batch in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-daemon/src/main.rs:8185` |
| `HIPFIRE_QWEN35_ROUTED_ONLY_MOE_FORWARD` | Runtime variable controlling qwen35 routed only moe forward in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/qwen35.rs:6420` |
| `HIPFIRE_QWEN35_STAGE_SYNC` | Runtime variable controlling qwen35 stage sync in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/qwen35.rs:13155` |
| `HIPFIRE_QWEN35_STAGE_TRACE` | Runtime variable controlling qwen35 stage trace in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/qwen35.rs:13149` |
| `HIPFIRE_QWEN35_STATE_QUANT` | Runtime variable controlling qwen35 state quant in hipfire. | `/home/sadara/.hipfire/src/cli/index.ts:747` |
| `HIPFIRE_QWEN35_XDNA1_INSTR` | Runtime variable controlling qwen35 xdna1 instr in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/ffn_bf16.rs:216` |
| `HIPFIRE_QWEN35_XDNA1_XCLBIN` | Runtime variable controlling qwen35 xdna1 xclbin in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/ffn_bf16.rs:215` |
| `HIPFIRE_RDNA2_VARIANT` | Runtime variable controlling rdna2 variant in hipfire. | `/home/sadara/.hipfire/src/crates/rdna-compute/src/feature_flags.rs:276` |
| `HIPFIRE_REPLAY_GRAPH` | Enabled when set to 1 | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/speculative.rs:4940` |
| `HIPFIRE_RESOURCE_LOCK` | HIPFIRE_RESOURCE_LOCK=0 disables daemon startup resource leases | `/home/sadara/.hipfire/src/crates/hipfire-daemon/src/main.rs:947` |
| `HIPFIRE_RESOURCE_LOCK_CPU_CORES` | HIPFIRE_RESOURCE_LOCK_CPU_CORES=0,2-4 adds daemon startup leases for CPU cores | `/home/sadara/.hipfire/src/crates/hipfire-daemon/src/main.rs:848` |
| `HIPFIRE_RESOURCE_LOCK_DIR` | HIPFIRE_RESOURCE_LOCK_DIR overrides the daemon resource-lock root directory | `/home/sadara/.hipfire/src/crates/hipfire-daemon/src/main.rs:963` |
| `HIPFIRE_RESOURCE_LOCK_NPUS` | HIPFIRE_RESOURCE_LOCK_NPUS=1 leases every detected NPU; comma lists lease explicit NPU IDs | `/home/sadara/.hipfire/src/crates/hipfire-daemon/src/main.rs:813` |
| `HIPFIRE_RESOURCE_LOCK_WAIT_MS` | HIPFIRE_RESOURCE_LOCK_WAIT_MS waits for busy daemon resource leases before failing startup | `/home/sadara/.hipfire/src/crates/hipfire-daemon/src/main.rs:967` |
| `HIPFIRE_RESPONSES_STATE_MAX` | Runtime variable controlling responses state max in hipfire. | `/home/sadara/.hipfire/src/cli/index.ts:2838` |
| `HIPFIRE_ROCBLAS_ALL_ARCHS` | Runtime variable controlling rocblas all archs in hipfire. | `/home/sadara/.hipfire/src/crates/rdna-compute/src/feature_flags.rs:266` |
| `HIPFIRE_ROCBLAS_OFF` | Enabled when set to 1 | `/home/sadara/.hipfire/src/crates/rdna-compute/src/feature_flags.rs:268` |
| `HIPFIRE_ROCPROF_BIN` | Runtime variable controlling rocprof bin in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-runtime/src/eval_harness/mod.rs:1611` |
| `HIPFIRE_ROCPROF_CSV` | Runtime variable controlling rocprof csv in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/bench_qwen35_speed.rs:325` |
| `HIPFIRE_ROPE_INTERLEAVED_LEGACY` | Runtime variable controlling rope interleaved legacy in hipfire. | `/home/sadara/.hipfire/src/crates/rdna-compute/src/feature_flags.rs:259` |
| `HIPFIRE_RUN_EXAMPLE_BIN` | Runtime variable controlling run example bin in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-runtime/src/eval_harness/mod.rs:9774` |
| `HIPFIRE_SAMPLE_COMPARE` | Enabled when set to 1 | `/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/infer_qwen35.rs:222` |
| `HIPFIRE_SERVER_PREFILL_SHARED_PREFIX_FANOUT` | Runtime variable controlling server prefill shared prefix fanout in hipfire. | `/home/sadara/.hipfire/src/cli/index.ts:5117` |
| `HIPFIRE_SERVER_RESIDENT_STATE_BUDGET_MB` | Runtime variable controlling server resident state budget mb in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-daemon/src/main.rs:5501` |
| `HIPFIRE_SMOKE_KV` | Select KV cache quant via HIPFIRE_SMOKE_KV (default q8, matches the | `/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/a3b_smoke_forward.rs:109` |
| `HIPFIRE_SMOKE_KV_SEQ` | production CLI default). asym3/asym4 engage the Givens-rotated 3/4-bit | `/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/a3b_smoke_forward.rs:102` |
| `HIPFIRE_SMOKE_MODE` | raw (default for back-compat): tokenize "Hello" and decode from pos=0 | `/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/a3b_smoke_forward.rs:157` |
| `HIPFIRE_SMOKE_PROMPT` | Defaults to Hello when unset | `/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/a3b_smoke_forward.rs:158` |
| `HIPFIRE_SMOKE_STEPS` | Runtime variable controlling smoke steps in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/a3b_smoke_forward.rs:38` |
| `HIPFIRE_SPEC_PHASES` | Enabled when set to 1 | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/speculative.rs:6929` |
| `HIPFIRE_STATE` | Interprets "HIPFIRE_STATE" from environment to select behavior | `/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/greedy_dump_top5.rs:245` |
| `HIPFIRE_STATE_QUANT` | Runtime variable controlling state quant in hipfire. | `/home/sadara/.hipfire/src/cli/index.ts:747` |
| `HIPFIRE_TARGET_ARCH` | Runtime variable controlling target arch in hipfire. | `/home/sadara/.hipfire/src/crates/rdna-compute/src/dispatch.rs:745` |
| `HIPFIRE_TIER_RATIO` | Runtime variable controlling tier ratio in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-quantize/src/main.rs:5738` |
| `HIPFIRE_UNIFORM_VRAM_TOLERANCE_GB` | Runtime variable controlling uniform vram tolerance gb in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-runtime/src/config.rs:97` |
| `HIPFIRE_VERIFY_GRAPH` | Runtime variable controlling verify graph in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/speculative.rs:6391` |
| `HIPFIRE_VERIFY_GRAPH_TIMING` | (HIPFIRE_VERIFY_GRAPH_TIMING=1). Two device-sync points bracket the | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/speculative.rs:6406` |
| `HIPFIRE_VERIFY_GRAPH_TREE` | Enabled when set to 1 | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/speculative.rs:6381` |
| `HIPFIRE_VL_DUMP_DIR` | little-endian f32 blobs + JSON sidecars to $HIPFIRE_VL_DUMP_DIR | `/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/infer.rs:148` |
| `HIPFIRE_WO_MMQ` | Enabled when set to 1 | `/home/sadara/.hipfire/src/crates/rdna-compute/src/feature_flags.rs:189` |
| `HIPFIRE_WO_WMMA_VARIANT` | Runtime variable controlling wo wmma variant in hipfire. | `/home/sadara/.hipfire/src/crates/rdna-compute/src/feature_flags.rs:263` |
| `HIPFIRE_XDNA1_LIB` | Runtime variable controlling xdna1 lib in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-arch-qwen35/src/xdna1_ffi.rs:254` |
| `HIP_PATH` | fails with "file not found". Add well-known candidates as -I flags; | `/home/sadara/.hipfire/src/crates/rdna-compute/src/compiler.rs:473` |
| `HIP_VISIBLE_DEVICES` | Used to configure runtime execution by explicitly setting "HIP_VISIBLE_DEVICES" | `/home/sadara/.hipfire/src/crates/hipfire-daemon/src/main.rs:2072` |
| `HOME` | Runtime variable controlling home in hipfire. | `/home/sadara/.hipfire/src/crates/rdna-compute/src/compiler.rs:236` |
| `HOSTNAME` | Runtime variable controlling hostname in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-runtime/src/coherence_runtime.rs:474` |
| `MAX_TOKENS` | Parses "MAX_TOKENS" with fallback defaults | `/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/greedy_dump.rs:112` |
| `MMQ_TEST_MODE` | Defaults to residual when unset | `/home/sadara/.hipfire/src/crates/rdna-compute/examples/test_gfx906_mmq_correctness.rs:65` |
| `NO_COLOR` | Runtime variable controlling no color in hipfire. | `/home/sadara/.hipfire/src/cli/chat.ts:100` |
| `NO_NGRAM` | Disabled for perf measurement — re-enable after implementing GPU n-gram kernel | `/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/infer_vl.rs:350` |
| `PATH` | Runtime variable controlling path in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-runtime/src/eval_harness/mod.rs:1621` |
| `PROMPT_MODE` | Defaults to thinking when unset | `/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/greedy_dump_top5.rs:91` |
| `QWEN35_TEST_MODEL` | Runtime variable controlling qwen35 test model in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/test_qwen35_loadQA.rs:22` |
| `ROCM_PATH` | Runtime variable controlling rocm path in hipfire. | `/home/sadara/.hipfire/src/crates/rdna-compute/src/compiler.rs:133` |
| `ROCR_VISIBLE_DEVICES` | Runtime variable controlling rocr visible devices in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-daemon/src/main.rs:772` |
| `TINYLLAMA_GGUF` | Runtime variable controlling tinyllama gguf in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/test_q4f16QA.rs:38` |
| `TRIALS` | Runtime variable controlling trials in hipfire. | `/home/sadara/.hipfire/src/crates/rdna-compute/examples/bench_gfx1151_hfq4_s4_mmq.rs:151` |
| `USERPROFILE` | Runtime variable controlling userprofile in hipfire. | `/home/sadara/.hipfire/src/crates/hipfire-daemon/src/main.rs:651` |
| `USE_SAMPLE` | Enabled when set to 1 | `/home/sadara/.hipfire/src/crates/hipfire-runtime/examples/a3b_multiturn_oneshot.rs:117` |
| `VERIFY` | VERIFY=1: run the token-parallel path vs the tile_batched fallback on | `/home/sadara/.hipfire/src/crates/rdna-compute/examples/q8_batched_attn_microbench.rs:92` |
| `W64` | Environment toggle value controls runtime behavior | `/home/sadara/.hipfire/src/crates/rdna-compute/examples/q8_batched_attn_microbench.rs:105` |

- Total env vars: **412**
- `HIPFIRE_*` vars: **373**
- non-`HIPFIRE_*` vars: **39**
