# hipfire environment variables — canonical reference

**Generated:** 2026-05-07. Auto-extracted from source via `git ls-files | grep -E '\.(rs|ts)$'`. See `Maintenance` at the bottom for re-generation.

This document is the single canonical reference for every environment variable hipfire reads. It supersedes the fragmented mentions across `docs/`, `AGENTS.md`, and inline source comments.

## Governance summary

| Layer | Count | Notes |
|---|---|---|
| `HIPFIRE_*` env vars | 117 | 14 plumbed through TUI, 44 mentioned in some doc, 59 silent |
| Non-`HIPFIRE_*` project env vars | 21 | Test/example/diag scaffolding. Should be renamed `HIPFIRE_*` for consistency. |
| `config.json` schema (`HipfireConfig`) | ~40 keys | Validated by `validateConfigValue()` in `cli/index.ts`. Some keys map 1:1 to env vars set at daemon spawn. |
| `per_model_config.json` overrides | same surface | Sparse overrides on top of the base config, applied per model tag. |
| Cargo features | 4 | `default = ["arch-qwen35", "arch-qwen35-vl", "arch-llama", "deltanet"]` |
| Production-path arch-detection branches | dozens | `self.arch == "gfx906"`, `arch.starts_with("gfx11")`, etc. Behavior changes per GPU silently. |
| File-existence gates | 7 in runtime/arch crates | Mostly cache and model-discovery. |

## Precedence rule

1. **Env var > config.json > built-in default.** When the CLI spawns the daemon, it reads `~/.hipfire/config.json`, then sets corresponding `HIPFIRE_*` env vars in the daemon's environment. The daemon then reads env vars; what the env var says wins, since the CLI may have overridden config-derived values for this run.
2. **`per_model_config.json` overrides apply at config-load time, before the CLI sets env vars.** So per-model settings effectively override the global config but are themselves overridable by env vars set in the operator's shell before launching `hipfire`.
3. **Direct `daemon` invocation (without the CLI) skips step 1.** In that mode, env vars are the only knob; `config.json` is not read.

## How to use this doc

- **Looking up a specific var?** Ctrl-F the table below.
- **Trying to tune a feature area?** Skim the category guide.
- **Stuck on default behavior?** Check the precedence section above first.
- **Adding a new env var?** Read `Adding a new env var` at the bottom.

## Quick reference

Categories are best-effort, derived from naming + source location. See the category guide for details.

| Variable | Category | Default | Defined at |
|---|---|---|---|
| `BENCH_BATCH` | NON-PREFIXED-TEST | 16 | `crates/rdna-compute/examples/bench_stream_overlap.rs:44` |
| `BENCH_DRAFT_K` | NON-PREFIXED-TEST | 5120usize | `crates/rdna-compute/examples/bench_stream_overlap.rs:142` |
| `BENCH_DRAFT_LAYERS` | NON-PREFIXED-TEST | 5usize | `crates/rdna-compute/examples/bench_stream_overlap.rs:155` |
| `BENCH_DRAFT_M` | NON-PREFIXED-TEST | 5120usize | `crates/rdna-compute/examples/bench_stream_overlap.rs:141` |
| `BENCH_DRAFT_N` | NON-PREFIXED-TEST | 16usize | `crates/rdna-compute/examples/bench_stream_overlap.rs:143` |
| `BENCH_K` | NON-PREFIXED-TEST | 5120 | `crates/rdna-compute/examples/bench_stream_overlap.rs:43` |
| `BENCH_M` | NON-PREFIXED-TEST | 5120 | `crates/rdna-compute/examples/bench_stream_overlap.rs:42` |
| `BENCH_VERIFY_LAYERS` | NON-PREFIXED-TEST | 5usize | `crates/rdna-compute/examples/bench_stream_overlap.rs:156` |
| `DDTREE_TIMING` | NON-PREFIXED-DIAG | off (presence-flag) | `crates/hipfire-arch-qwen35/src/speculative.rs:3870` |
| `DEBUG_LAYERS` | NON-PREFIXED-DIAG | off (presence-flag) | `crates/hipfire-arch-qwen35/src/qwen35.rs:2181` |
| `DFLASH_LIVE_TAU` | NON-PREFIXED-DIAG | off (presence-flag) | `crates/hipfire-runtime/examples/dflash_spec_demo.rs:795` |
| `FP32_STATE` | NON-PREFIXED-DIAG | off (presence-flag) | `crates/hipfire-runtime/examples/infer_qwen35.rs:142` |
| `HFQ_TEST_N_ITER` | NON-PREFIXED-TEST | — | `crates/rdna-compute/examples/test_gfx906_mmq_correctness.rs:85` |
| `HFQ_TEST_SCALE_LOG10` | NON-PREFIXED-TEST | — | `crates/rdna-compute/examples/test_fused_gate_up_dp4a.rs:132` |
| `HFQ_TEST_ZP_MAX` | NON-PREFIXED-TEST | — | `crates/rdna-compute/examples/test_fused_gate_up_dp4a.rs:134` |
| `HIPFIRE_ADAPTIVE_B_DOWN` | DFLASH-ADAPT | — | `crates/hipfire-runtime/examples/dflash_spec_demo.rs:981` |
| `HIPFIRE_ADAPTIVE_B_UNSAFE` | DFLASH-ADAPT | "" (set to "1" to enable) | `crates/hipfire-runtime/examples/dflash_spec_demo.rs:444` |
| `HIPFIRE_ADAPTIVE_B_UP` | DFLASH-ADAPT | — | `crates/hipfire-runtime/examples/dflash_spec_demo.rs:979` |
| `HIPFIRE_ALLOW_MQ2` | MISC-USER | "" (set to "1" to enable) | `crates/hipfire-quantize/src/main.rs:2106` |
| `HIPFIRE_ALLOW_MQ2_LLOYD` | MISC-USER | "" (set to "1" to enable) | `crates/hipfire-quantize/src/main.rs:2141` |
| `HIPFIRE_ALLOW_MQ3_LLOYD` | MISC-USER | "" (set to "1" to enable) | `crates/hipfire-quantize/src/main.rs:2126` |
| `HIPFIRE_ATTN_FLASH` | ATTN | — | `cli/index.ts:797` |
| `HIPFIRE_BLOB_FORCE` | GRAPH-DIAG | "" (set to "1" to enable) | `crates/rdna-compute/src/dispatch.rs:652` |
| `HIPFIRE_CALIB_PROFILE` | DIAG-DUMP | — | `crates/hipfire-runtime/examples/triattn_validate.rs:32` |
| `HIPFIRE_CHATML` | PROMPT-FRAME | "" (set to "1" to enable) | `crates/hipfire-runtime/examples/probe_argmax_agreement.rs:43` |
| `HIPFIRE_DDTREE_BUDGET` | DDTREE-RESEARCH | — | `crates/hipfire-runtime/examples/daemon.rs:1702` |
| `HIPFIRE_DDTREE_FORCE_SLOW` | DDTREE-RESEARCH | "" (set to "1" to enable) | `crates/hipfire-arch-qwen35/src/speculative.rs:4073` |
| `HIPFIRE_DDTREE_LOGW_CUTOFF` | DDTREE-RESEARCH | — | `crates/hipfire-arch-qwen35/src/speculative.rs:86` |
| `HIPFIRE_DDTREE_PATH_B_CAPTURE` | DDTREE-RESEARCH | — | `crates/hipfire-arch-qwen35/src/speculative.rs:4011` |
| `HIPFIRE_DDTREE_PATH_C` | DDTREE-RESEARCH | — | `crates/hipfire-runtime/examples/daemon.rs:2095` |
| `HIPFIRE_DDTREE_PATH_C_VERBOSE` | DDTREE-RESEARCH | "" (set to "1" to enable) | `crates/hipfire-arch-qwen35/src/speculative.rs:4566` |
| `HIPFIRE_DDTREE_TAPE_DUMP` | DDTREE-RESEARCH | "" (set to "1" to enable) | `crates/hipfire-arch-qwen35/src/speculative.rs:4090` |
| `HIPFIRE_DDTREE_TOPK` | DDTREE-RESEARCH | — | `crates/hipfire-runtime/examples/daemon.rs:1774` |
| `HIPFIRE_DDTREE_TREE_LA` | DDTREE-RESEARCH | — | `crates/hipfire-arch-qwen35/src/speculative.rs:3968` |
| `HIPFIRE_DETERMINISTIC` | MISC-USER | "" (set to "1" to enable) | `crates/rdna-compute/src/dispatch.rs:7380` |
| `HIPFIRE_DEVICES` | MULTI-GPU | — | `crates/hipfire-runtime/src/multi_gpu.rs:351` |
| `HIPFIRE_DFLASH_DRAFT` | DFLASH-USER | — | `cli/index.ts:479` |
| `HIPFIRE_DFLASH_LOOP_BREAK` | DFLASH-SAFETYNET | — | `crates/hipfire-runtime/examples/dflash_spec_demo.rs:732` |
| `HIPFIRE_DFLASH_LOOP_BREAK_MAX_ESCALATIONS` | DFLASH-SAFETYNET | — | `crates/hipfire-runtime/examples/dflash_spec_demo.rs:750` |
| `HIPFIRE_DFLASH_LOOP_BREAK_RECOVERY` | DFLASH-SAFETYNET | — | `crates/hipfire-runtime/examples/dflash_spec_demo.rs:748` |
| `HIPFIRE_DFLASH_LOOP_BREAK_RP_MAX` | DFLASH-SAFETYNET | — | `crates/hipfire-runtime/examples/dflash_spec_demo.rs:746` |
| `HIPFIRE_DFLASH_LOOP_BREAK_RP_STEP` | DFLASH-SAFETYNET | — | `crates/hipfire-runtime/examples/dflash_spec_demo.rs:744` |
| `HIPFIRE_DFLASH_LOOP_BREAK_STOP_AFTER` | DFLASH-SAFETYNET | — | `crates/hipfire-runtime/examples/dflash_spec_demo.rs:742` |
| `HIPFIRE_DFLASH_LOOP_BREAK_TEMP` | DFLASH-SAFETYNET | — | `crates/hipfire-runtime/examples/dflash_spec_demo.rs:740` |
| `HIPFIRE_DFLASH_NGRAM_BLOCK` | DFLASH-USER | — | `cli/index.ts:825` |
| `HIPFIRE_DFLASH_SEED_ORACLE` | DFLASH-SAFETYNET | "" (set to "1" to enable) | `crates/hipfire-arch-qwen35/src/speculative.rs:2978` |
| `HIPFIRE_DPM_WARMUP_SECS` | PERF-DIAG | — | `crates/hipfire-runtime/examples/bench_qwen35_mq4.rs:95` |
| `HIPFIRE_DRAFT_F16` | DRAFT/SPEC | — | `crates/hipfire-runtime/src/dflash.rs:193` |
| `HIPFIRE_DRAFT_GEMM_DUMP` | DIAG-DUMP | "" (set to "1" to enable) | `crates/hipfire-runtime/src/dflash.rs:571` |
| `HIPFIRE_DRAFT_SUBPHASE` | DRAFT/SPEC | "" (set to "1" to enable) | `crates/hipfire-runtime/src/dflash.rs:805` |
| `HIPFIRE_DTOH_DUMP` | DIAG-DUMP | "" (set to "1" to enable) | `crates/hip-bridge/src/ffi.rs:594` |
| `HIPFIRE_EXPERIMENTAL_BUDGET_ALERT` | MISC-USER | — | `cli/index.ts:808` |
| `HIPFIRE_FLASH_PARTIALS_BATCH` | ATTN | — | `crates/hipfire-arch-qwen35/src/qwen35.rs:2750` |
| `HIPFIRE_FORCE_A3B_EVICTION` | DFLASH-USER | — | `cli/index.ts:4978` |
| `HIPFIRE_FP16` | KERNEL-SELECTOR | — | `crates/rdna-compute/src/dispatch.rs:3520` |
| `HIPFIRE_GATE_UP_VARIANT` | KERNEL-SELECTOR | — | `crates/rdna-compute/src/dispatch.rs:5250` |
| `HIPFIRE_GCN5_WAVE64_HYBRID` | KERNEL-SELECTOR | — | `crates/rdna-compute/src/dispatch.rs:166` |
| `HIPFIRE_GEMM_DUMP` | DIAG-DUMP | "" (set to "1" to enable) | `crates/rdna-compute/src/dispatch.rs:7441` |
| `HIPFIRE_GEMV_DP4A` | KERNEL-SELECTOR | — | `crates/rdna-compute/src/dispatch.rs:51` |
| `HIPFIRE_GEMV_PREFETCH` | KERNEL-SELECTOR | — | `crates/rdna-compute/src/dispatch.rs:80` |
| `HIPFIRE_GEMV_ROWS` | KERNEL-SELECTOR | — | `crates/rdna-compute/src/dispatch.rs:26` |
| `HIPFIRE_GEN` | DAEMON-RUNTIME | — | `crates/hipfire-runtime/examples/a3b_multiturn_oneshot.rs:18` |
| `HIPFIRE_GPU_TOPK` | KERNEL-SELECTOR | "" (set to "1" to enable) | `crates/hipfire-runtime/examples/infer_qwen35.rs:167` |
| `HIPFIRE_GRAPH` | HIPGRAPH | — | `cli/index.ts:3448` |
| `HIPFIRE_GRAPH_MOE` | HIPGRAPH | "" (set to "1" to enable) | `crates/hipfire-arch-qwen35/src/qwen35.rs:2933` |
| `HIPFIRE_HAVE_2_GPU` | MULTI-GPU | "" (set to "1" to enable) | `crates/hipfire-arch-qwen35/tests/pp_parity.rs:159` |
| `HIPFIRE_HIPCC_EXTRA_FLAGS` | MISC-USER | — | `crates/rdna-compute/src/compiler.rs:298` |
| `HIPFIRE_HOST_TIMING` | PERF-DIAG | "" (set to "1" to enable) | `crates/hipfire-runtime/examples/dflash_spec_demo.rs:934` |
| `HIPFIRE_KERNEL_CACHE` | MISC-USER | `.hipfire_kernels` (cwd-relative) | `crates/rdna-compute/src/compiler.rs:89` |
| `HIPFIRE_KV_MODE` | KV-CACHE | — | `cli/index.ts:400` |
| `HIPFIRE_KV_PHYSICAL_CAP` | KV-CACHE | — | `crates/hipfire-runtime/examples/daemon.rs:1211` |
| `HIPFIRE_LLOYD_FORCE_BASELINE` | KERNEL-SELECTOR | "" (set to "1" to enable) | `crates/rdna-compute/src/kernels.rs:76` |
| `HIPFIRE_LLOYD_GFX12` | KERNEL-SELECTOR | "" (set to "1" to enable) | `crates/hipfire-arch-qwen35/src/qwen35.rs:3779` |
| `HIPFIRE_LM_HEAD_WMMA` | KERNEL-SELECTOR | — | `crates/rdna-compute/src/dispatch.rs:7954` |
| `HIPFIRE_LOCAL` | DAEMON-RUNTIME | — | `cli/index.ts:1205` |
| `HIPFIRE_MEMSET_DUMP` | DIAG-DUMP | "" (set to "1" to enable) | `crates/hip-bridge/src/ffi.rs:669` |
| `HIPFIRE_MMQ` | MMQ | — | `crates/rdna-compute/src/dispatch.rs:228` |
| `HIPFIRE_MMQ_DIAG_QUANTIZE_ONLY` | MMQ | "" (set to "1" to enable) | `crates/rdna-compute/src/dispatch.rs:653` |
| `HIPFIRE_MMQ_MIN_BATCH` | MMQ | — | `crates/rdna-compute/src/dispatch.rs:249` |
| `HIPFIRE_MMQ_SCREEN` | MMQ | — | `crates/rdna-compute/src/dispatch.rs:633` |
| `HIPFIRE_MMQ_SCREEN_THRESHOLD` | MMQ | — | `crates/rdna-compute/src/dispatch.rs:648` |
| `HIPFIRE_MODEL` | DAEMON-RUNTIME | — | `cli/index.ts:1350` |
| `HIPFIRE_MW16` | LIB | — | `crates/rdna-compute/src/dispatch.rs:7332` |
| `HIPFIRE_NGRAM_LOOP_THRESHOLD` | NGRAM-DETECTOR | — | `crates/hipfire-runtime/src/loop_guard.rs:47` |
| `HIPFIRE_NGRAM_WINDOW` | NGRAM-DETECTOR | — | `crates/hipfire-runtime/src/loop_guard.rs:49` |
| `HIPFIRE_NO_PID_FILE` | DAEMON-RUNTIME | — | `cli/index.ts:1286` |
| `HIPFIRE_NORMALIZE_PROMPT` | PROMPT-FRAME | — | `cli/index.ts:817` |
| `HIPFIRE_PFLASH_SCORE_LAYER` | LIB | — | `crates/hipfire-arch-qwen35/src/pflash.rs:676` |
| `HIPFIRE_PP_LAYERS` | MULTI-GPU | — | `crates/hipfire-runtime/examples/daemon.rs:1461` |
| `HIPFIRE_PP_PARITY_MODEL` | MULTI-GPU | — | `crates/hipfire-arch-qwen35/tests/pp_parity.rs:163` |
| `HIPFIRE_PP_PFLASH` | MULTI-GPU | "" (set to "1" to enable) | `crates/hipfire-runtime/examples/daemon.rs:604` |
| `HIPFIRE_PREFILL_ALPHA` | PFLASH | — | `crates/hipfire-arch-qwen35/src/pflash.rs:114` |
| `HIPFIRE_PREFILL_BATCHED` | PFLASH | — | `crates/hipfire-arch-qwen35/src/qwen35.rs:3509` |
| `HIPFIRE_PREFILL_BLOCK` | PFLASH | — | `crates/hipfire-arch-qwen35/src/pflash.rs:130` |
| `HIPFIRE_PREFILL_COMPRESSION` | PFLASH | — | `crates/hipfire-arch-qwen35/src/pflash.rs:100` |
| `HIPFIRE_PREFILL_DRAFTER` | PFLASH | — | `crates/hipfire-arch-qwen35/src/pflash.rs:141` |
| `HIPFIRE_PREFILL_KEEP_RATIO` | PFLASH | — | `crates/hipfire-arch-qwen35/src/pflash.rs:108` |
| `HIPFIRE_PREFILL_MAX_BATCH` | PFLASH | — | `crates/hipfire-arch-qwen35/src/qwen35.rs:2817` |
| `HIPFIRE_PREFILL_MIN_KEEP` | PFLASH | — | `crates/hipfire-arch-qwen35/src/pflash.rs:118` |
| `HIPFIRE_PREFILL_PROFILE` | DIAG-DUMP | "" (set to "1" to enable) | `crates/hipfire-arch-qwen35/src/pflash.rs:138` |
| `HIPFIRE_PREFILL_RECENT` | PFLASH | — | `crates/hipfire-arch-qwen35/src/pflash.rs:126` |
| `HIPFIRE_PREFILL_REUSE_PBS` | PFLASH | "" (set to "1" to enable) | `crates/hipfire-arch-qwen35/src/qwen35.rs:2816` |
| `HIPFIRE_PREFILL_SINK` | PFLASH | — | `crates/hipfire-arch-qwen35/src/pflash.rs:122` |
| `HIPFIRE_PREFILL_SPARSE_THRESHOLD` | PFLASH | — | `crates/hipfire-arch-qwen35/src/pflash.rs:134` |
| `HIPFIRE_PREFILL_THRESHOLD` | PFLASH | — | `crates/hipfire-arch-qwen35/src/pflash.rs:104` |
| `HIPFIRE_PROFILE` | TEST | "" (set to "1" to enable) | `crates/hipfire-runtime/examples/bench_qwen35_mq4.rs:108` |
| `HIPFIRE_PROFILE_CYCLES` | EXAMPLE | — | `crates/hipfire-runtime/examples/dflash_spec_demo.rs:870` |
| `HIPFIRE_PROFILE_DECODE` | TEST | "" (set to "1" to enable) | `crates/hipfire-runtime/examples/bench_qwen35_mq4.rs:230` |
| `HIPFIRE_PROMPT_HEAT_JSON` | DIAG-DUMP | "" (set to "1" to enable) | `crates/hipfire-runtime/src/tokenizer.rs:785` |
| `HIPFIRE_PROMPT_HEAT_LIMIT` | DIAG-DUMP | — | `crates/hipfire-runtime/src/tokenizer.rs:807` |
| `HIPFIRE_PROMPT_TOKEN_HEAT` | EXAMPLE | "" (set to "1" to enable) | `crates/hipfire-runtime/examples/daemon.rs:719` |
| `HIPFIRE_QA_KV_MODES` | TEST-HARNESS | — | `crates/hipfire-runtime/examples/test_inferenceQA.rs:606` |
| `HIPFIRE_QUANT_THREADS` | LIB | — | `crates/hipfire-quantize/src/main.rs:2053` |
| `HIPFIRE_RDNA2_VARIANT` | KERNEL-SELECTOR | — | `cli/index.ts:2593` |
| `HIPFIRE_REPLAY_GRAPH` | GRAPH-DIAG | "" (set to "1" to enable) | `crates/hipfire-arch-qwen35/src/speculative.rs:630` |
| `HIPFIRE_ROCBLAS_ALL_ARCHS` | KERNEL-SELECTOR | "" (set to "1" to enable) | `crates/rdna-compute/src/dispatch.rs:690` |
| `HIPFIRE_ROCBLAS_MIN_BATCH` | KERNEL-SELECTOR | — | `crates/rdna-compute/src/dispatch.rs:1413` |
| `HIPFIRE_ROCBLAS_OFF` | KERNEL-SELECTOR | "" (set to "1" to enable) | `crates/rdna-compute/src/dispatch.rs:1410` |
| `HIPFIRE_SAMPLE_COMPARE` | DRAFT/SPEC | "" (set to "1" to enable) | `crates/hipfire-runtime/examples/infer_qwen35.rs:168` |
| `HIPFIRE_SMOKE_KV` | SMOKE-TEST | \|_\| "q8".to_string( | `crates/hipfire-runtime/examples/a3b_smoke_forward.rs:50` |
| `HIPFIRE_SMOKE_KV_SEQ` | SMOKE-TEST | — | `crates/hipfire-runtime/examples/a3b_smoke_forward.rs:45` |
| `HIPFIRE_SMOKE_MODE` | SMOKE-TEST | \|_\| "raw".to_string( | `crates/hipfire-runtime/examples/a3b_smoke_forward.rs:78` |
| `HIPFIRE_SMOKE_PROMPT` | SMOKE-TEST | — | `crates/hipfire-runtime/examples/a3b_smoke_forward.rs:79` |
| `HIPFIRE_SMOKE_STEPS` | SMOKE-TEST | — | `crates/hipfire-runtime/examples/a3b_smoke_forward.rs:28` |
| `HIPFIRE_SPEC_PHASES` | DRAFT/SPEC | "" (set to "1" to enable) | `crates/hipfire-arch-qwen35/src/speculative.rs:2452` |
| `HIPFIRE_UNIFORM_VRAM_TOLERANCE_GB` | MULTI-GPU | — | `crates/hipfire-runtime/src/multi_gpu.rs:405` |
| `HIPFIRE_VERIFY_GRAPH` | GRAPH-DIAG | — | `crates/hipfire-arch-qwen35/src/speculative.rs:1926` |
| `HIPFIRE_VERIFY_GRAPH_TIMING` | GRAPH-DIAG | "" (set to "1" to enable) | `crates/hipfire-arch-qwen35/src/speculative.rs:1938` |
| `HIPFIRE_VERIFY_GRAPH_TREE` | GRAPH-DIAG | "" (set to "1" to enable) | `crates/hipfire-arch-qwen35/src/speculative.rs:1916` |
| `HIPFIRE_WO_MMQ` | KERNEL-SELECTOR | "" (set to "1" to enable) | `crates/rdna-compute/src/dispatch.rs:6666` |
| `HIPFIRE_WO_WMMA_VARIANT` | KERNEL-SELECTOR | — | `crates/rdna-compute/src/dispatch.rs:7393` |
| `MAX_TOKENS` | NON-PREFIXED-TEST | — | `crates/hipfire-runtime/examples/greedy_dump.rs:77` |
| `MMQ_TEST_MODE` | NON-PREFIXED-TEST | \|_\| "residual".to_string( | `crates/rdna-compute/examples/test_gfx906_mmq_correctness.rs:59` |
| `NO_NGRAM` | NON-PREFIXED-DIAG | — | `crates/hipfire-runtime/examples/infer_vl.rs:219` |
| `PROMPT_MODE` | NON-PREFIXED-TEST | \|_\| "thinking".to_string( | `crates/hipfire-runtime/examples/greedy_dump.rs:27` |
| `QWEN35_TEST_MODEL` | NON-PREFIXED-TEST | — | `crates/hipfire-runtime/examples/test_inferenceQA.rs:121` |
| `TINYLLAMA_GGUF` | NON-PREFIXED-TEST | — | `crates/hipfire-runtime/examples/test_gemv_q4kQA.rs:34` |
| `USE_SAMPLE` | NON-PREFIXED-TEST | "" (set to "1" to enable) | `crates/hipfire-runtime/examples/a3b_multiturn_oneshot.rs:75` |

## Category guide

### `KV-CACHE` (2)

KV cache mode and physical capacity. Maps to `cfg.kv_cache` in `config.json`.

- `HIPFIRE_KV_MODE` — `q8` (default), `asym4`, `asym3`, `asym2`. The CLI sets this from `cfg.kv_cache` before spawning daemon. Direct daemon callers can set this themselves.
- `HIPFIRE_KV_PHYSICAL_CAP` — physical cap on KV slots (vs the logical `max_seq`). Used for eviction tuning. Production-path setting.

### `HIPGRAPH` (2)

Decode-loop graph capture, ~5-15% decode speedup on stable kernel sets.

- `HIPFIRE_GRAPH` — set to `1` to enable graph capture. Maps to `cfg.flash_mode == "auto"` in CLI. Default: capture for 4B/9B/27B, off for 0.8B (known hipGraph bug).
- `HIPFIRE_GRAPH_MOE` — opt-in graph capture for MoE forward path. Default off because MoE expert routing changes per-token, breaking graph reuse.

### `MMQ` (5)

Mixed-precision GEMM (Q8_1 activation × 4-bit weight on dp4a, RDNA3+/gfx906). ~+20% prefill on `pp≥256`.

- `HIPFIRE_MMQ` — `0`/`1`/`auto`. Maps to `cfg.mmq_screen != "off"`. Auto-arch-gates to RDNA3/3.5 + gfx906.
- `HIPFIRE_MMQ_MIN_BATCH` — minimum batch size at which MMQ kicks in. Below this, falls through to FP16 path.
- `HIPFIRE_MMQ_SCREEN` — `auto`/`on`/`off`. Per-weight Q8_1 outlier detection. Auto = arch-gate to RDNA3/3.5 only.
- `HIPFIRE_MMQ_SCREEN_THRESHOLD` — float; reject Q8_1 quantize when error exceeds this. Default `0.10`.
- `HIPFIRE_MMQ_DIAG_QUANTIZE_ONLY` — diag flag isolating Q8_1 quantize cost from dp4a kernel cost. Read once at init via the read-once-cache pattern.

### `KERNEL-SELECTOR` (15)

Hot-path kernel choice levers. **All silent today.** Power users who tune for specific arches need to read source.

- `HIPFIRE_FP16` — gate FP16 prefill paths. `0` = disable.
- `HIPFIRE_GEMV_DP4A`, `HIPFIRE_GEMV_PREFETCH`, `HIPFIRE_GEMV_ROWS` — GEMV kernel knobs.
- `HIPFIRE_GATE_UP_VARIANT` — fused gate+up dispatch variant.
- `HIPFIRE_GCN5_WAVE64_HYBRID` — gfx906 (MI50) prefill: hybrid Wave64 FP16. Production-path on gfx906.
- `HIPFIRE_GPU_TOPK` — opt-in GPU-resident topk folding (Gemma4 perf lever). Set `1` to enable.
- `HIPFIRE_LLOYD_FORCE_BASELINE` — disable Lloyd-MQ3 K4+LDS fast variants on gfx11. Used to bisect drift.
- `HIPFIRE_LLOYD_GFX12=1` — opt-in dispatch of Lloyd-MQ3 WMMA kernels on gfx12 (RDNA4) inside `is_batchable_la`. Default off because the gfx12 sibling kernels ship code-complete but runtime-unvalidated locally; gfx12 reviewers set this to exercise parity / coherence-gate on RDNA4. Once external CI confirms gfx12 parity, the gate can be dropped or default-flipped. Ships with PR #195 (MQ3-Lloyd WMMA prefill).
- `HIPFIRE_LM_HEAD_WMMA` — lm_head dispatch lever.
- `HIPFIRE_RDNA2_VARIANT` — RDNA2 (gfx10x0) variant override. Plumbed via TUI + `cfg.rdna2_variant`.
- `HIPFIRE_ROCBLAS_ALL_ARCHS`, `HIPFIRE_ROCBLAS_MIN_BATCH`, `HIPFIRE_ROCBLAS_OFF` — rocBLAS dispatch gates.
- `HIPFIRE_WO_MMQ`, `HIPFIRE_WO_WMMA_VARIANT` — workaround flags for specific arch quirks.

### `MULTI-GPU` (6)

Pipeline-parallel and multi-device orchestration. Tied to `crates/hipfire-runtime/src/multi_gpu.rs` + Stage 7 of issue #58.

- `HIPFIRE_DEVICES` — explicit device selection (alternate to `ROCR_VISIBLE_DEVICES`).
- `HIPFIRE_PP_LAYERS="a,b,..."` — explicit asymmetric layer split (PR #190).
- `HIPFIRE_PP_PFLASH=1` — opt into experimental PFlash + pp>1 compose (PR #190).
- `HIPFIRE_HAVE_2_GPU=1` — pp_parity test gate; required for the 2-GPU parity battery.
- `HIPFIRE_PP_PARITY_MODEL` — model path override for the pp_parity test.
- `HIPFIRE_UNIFORM_VRAM_TOLERANCE_GB` — VRAM-tolerance threshold above which mixed-arch warning fires under `HIPFIRE_ALLOW_MIXED_ARCH=1`.

### `PFLASH` (13)

PFlash long-context prefill compression. `prefill_compression=auto` enables. Most settings have `cfg.prefill_*` mirrors.

- `HIPFIRE_PREFILL_COMPRESSION` — `off`/`auto`/`always`. Maps to `cfg.prefill_compression`.
- `HIPFIRE_PREFILL_DRAFTER` — path to drafter model (e.g. `qwen3.5-0.8b.mq4`). Maps to `cfg.prefill_drafter`.
- `HIPFIRE_PREFILL_THRESHOLD` — token count above which compression activates. Default `32768`.
- `HIPFIRE_PREFILL_KEEP_RATIO` — fraction of tokens kept after compression. Default `0.05` (aggressive).
- `HIPFIRE_PREFILL_ALPHA` — score-mixing coefficient. Default `0.85`.
- `HIPFIRE_PREFILL_MIN_KEEP` — minimum tokens to keep regardless of ratio. Default `2048`.
- `HIPFIRE_PREFILL_SINK` — leading-token sink window. Default `256`.
- `HIPFIRE_PREFILL_RECENT` — trailing-token recent window. Default `1024`.
- `HIPFIRE_PREFILL_BLOCK` — scoring block size. Default `128`.
- `HIPFIRE_PREFILL_SPARSE_THRESHOLD` — sparse-attention threshold. Default `32768`.
- `HIPFIRE_PREFILL_PROFILE=1` — emit prefill scoring timing JSON.
- `HIPFIRE_PREFILL_BATCHED` — batched prefill mode (different from compression). `0` to disable.
- `HIPFIRE_PREFILL_MAX_BATCH` — max prefill batch size when batched mode is on.
- `HIPFIRE_PREFILL_REUSE_PBS=1` — reuse the pre-batched scratch across prefill calls.

### `DFLASH-USER` (3)

User-facing DFlash speculative-decode knobs. All TUI-exposed.

- `HIPFIRE_DFLASH_DRAFT` — path to drafter model. Maps to `cfg.dflash_drafter`-derived auto-discovery.
- `HIPFIRE_DFLASH_NGRAM_BLOCK` — n-gram blocking mode. Maps to `cfg.dflash_ngram_block`.
- `HIPFIRE_FORCE_A3B_EVICTION=1` — force CASK eviction on A3B regardless of load-time heuristic.

### `DFLASH-SAFETYNET` (8)

Spec-decode loop-break recovery system. **All silent.** These knobs control how the daemon recovers from token-attractor loops detected mid-spec-decode (recoverable false-positives).

- `HIPFIRE_DFLASH_LOOP_BREAK` — master enable.
- `HIPFIRE_DFLASH_LOOP_BREAK_MAX_ESCALATIONS` — how many recovery escalation tiers to try before bypassing spec.
- `HIPFIRE_DFLASH_LOOP_BREAK_RECOVERY` — recovery strategy.
- `HIPFIRE_DFLASH_LOOP_BREAK_RP_MAX`, `HIPFIRE_DFLASH_LOOP_BREAK_RP_STEP` — repeat-penalty escalation params.
- `HIPFIRE_DFLASH_LOOP_BREAK_STOP_AFTER` — stop spec entirely after N consecutive loop detections.
- `HIPFIRE_DFLASH_LOOP_BREAK_TEMP` — temperature escalation for recovery.
- `HIPFIRE_DFLASH_SEED_ORACLE=1` — opt-in spec-decode seed oracle (Phase B research artifact; may be retirable).

### `DFLASH-ADAPT` (3)

Adaptive block-size sampler for DFlash spec-decode. Spec accept-rate feedback loop tuning.

- `HIPFIRE_ADAPTIVE_B_DOWN`, `HIPFIRE_ADAPTIVE_B_UP` — accept-rate thresholds for B size adjustment.
- `HIPFIRE_ADAPTIVE_B_UNSAFE=1` — disable safety bounds on B.

### `DRAFT/SPEC` (4)

Per-draft-model behavior knobs.

- `HIPFIRE_DRAFT_F16` — draft model in FP16 instead of MQ4.
- `HIPFIRE_DRAFT_SUBPHASE=1` — diag flag for subphase profiling.
- `HIPFIRE_SPEC_PHASES=1` — emit per-phase spec-decode timing JSON.
- `HIPFIRE_SAMPLE_COMPARE=1` — compare sample tokens against a reference path (development only).

### `DDTREE-RESEARCH` (9)

DDTree (tree-mode spec-decode) research surface. Per `findings/path-d-vs-path-c.md` and CLAUDE.md memory entries, **Path D and tree-mode pipelining are empirically dominated** across all tested model regimes. These vars are research artifacts; most can be retired.

- `HIPFIRE_DDTREE_BUDGET` — tree-search budget.
- `HIPFIRE_DDTREE_TOPK` — top-K branching factor.
- `HIPFIRE_DDTREE_PATH_C` — Path C phase selector (`phase1`/`phase2`).
- `HIPFIRE_DDTREE_PATH_C_VERBOSE=1` — verbose Path C tracing.
- `HIPFIRE_DDTREE_PATH_B_CAPTURE` — Path B capture flag.
- `HIPFIRE_DDTREE_TAPE_DUMP=1` — dump verify tape JSON.
- `HIPFIRE_DDTREE_LOGW_CUTOFF` — log-weight cutoff threshold.
- `HIPFIRE_DDTREE_FORCE_SLOW=1` — force slow-path verify (gather-based; structurally inferior per memory).
- `HIPFIRE_DDTREE_TREE_LA` — linear-attention tree mode flag.

### `NGRAM-DETECTOR` (2)

Loop-detector for output-loop guard (#125, #111).

- `HIPFIRE_NGRAM_LOOP_THRESHOLD` — N-gram repeat threshold.
- `HIPFIRE_NGRAM_WINDOW` — sliding window size.

### `PROMPT-FRAME` (2)

Prompt scaffolding behavior.

- `HIPFIRE_NORMALIZE_PROMPT` — collapse 3+ consecutive newlines to 2. Default `1` since 2026-04-26 (+24% τ on PEP-8 prompts). Maps to `cfg.prompt_normalize`.
- `HIPFIRE_CHATML=1` — opt-in ChatML wrap in `probe_argmax_agreement` example.

### `ATTN` (2)

Flash-attention dispatch.

- `HIPFIRE_ATTN_FLASH` — flash-attention mode override. Maps to `cfg.flash_mode`.
- `HIPFIRE_FLASH_PARTIALS_BATCH` — batch size for partial-flash kernels.

### `MULTI-GPU` daemon-runtime helpers

- `HIPFIRE_LOCAL` — bench-mode local-only flag.
- `HIPFIRE_MODEL` — model path override for one-shot examples.
- `HIPFIRE_GEN` — generation count override for one-shot examples.
- `HIPFIRE_NO_PID_FILE=1` — skip the daemon PID file (used for second-instance bring-up under TUI testing).

### `MISC-USER` (6)

Miscellaneous user-facing flags.

- `HIPFIRE_DETERMINISTIC=1` — byte-exact output mode. Disables non-deterministic optimizations.
- `HIPFIRE_EXPERIMENTAL_BUDGET_ALERT=1` — gate the `budget_alert_at_tok` / `budget_alert_text` daemon params. Maps to `cfg.experimental_budget_alert`.
- `HIPFIRE_HIPCC_EXTRA_FLAGS` — extra flags appended to all hipcc invocations during JIT.
- `HIPFIRE_KERNEL_CACHE` — JIT'd `.hsaco` cache directory. Default `.hipfire_kernels` (cwd-relative). Pin to `/tmp/hipfire_kernels` for tmpfs speed; default isolates parallel worktrees/agents from clobbering each other's blobs.
- `HIPFIRE_ALLOW_MQ2=1`, `HIPFIRE_ALLOW_MQ2_LLOYD=1`, `HIPFIRE_ALLOW_MQ3_LLOYD=1` — opt-in research-grade quant formats during quantizer run.

### `DIAG-DUMP` (8)

Per-event dumps for debugging. All `=1` to enable.

- `HIPFIRE_GEMM_DUMP`, `HIPFIRE_DRAFT_GEMM_DUMP` — write GEMM input/output to /tmp.
- `HIPFIRE_DTOH_DUMP`, `HIPFIRE_MEMSET_DUMP` — track device-to-host copies and memsets.
- `HIPFIRE_PROMPT_HEAT_JSON`, `HIPFIRE_PROMPT_HEAT_LIMIT` — emit per-token tokenizer heat data.
- `HIPFIRE_CALIB_PROFILE` — triattn calibration profiling.
- `HIPFIRE_PREFILL_PROFILE=1` — PFlash scoring profile (also covered under PFLASH).

### `GRAPH-DIAG` (5)

hipGraph capture/replay diagnostics.

- `HIPFIRE_BLOB_FORCE=1` — force kernarg blob accumulation across capture sessions.
- `HIPFIRE_REPLAY_GRAPH=1` — force graph replay even when capture would normally re-fire.
- `HIPFIRE_VERIFY_GRAPH` — verify-side graph capture toggle (default on).
- `HIPFIRE_VERIFY_GRAPH_TIMING=1` — emit per-replay timing.
- `HIPFIRE_VERIFY_GRAPH_TREE=1` — verify-side tree-mode graph variant.

### `SMOKE-TEST` (5)

`a3b_smoke_forward` example knobs. Test harness only.

### `PERF-DIAG` (2)

- `HIPFIRE_DPM_WARMUP_SECS` — DPM (dynamic power management) warmup seconds before timing starts.
- `HIPFIRE_HOST_TIMING=1` — emit host-side timing JSON.

### `TEST` / `EXAMPLE` / `TEST-HARNESS` (5)

Test/bench scaffolding. Not production-path.

## Non-prefixed env vars — rename targets

These 21 vars violate the `HIPFIRE_*` convention. Most are bench/test/example/diag scaffolding; one (`DEBUG_LAYERS`) is in production `qwen35.rs` hot path.

| Current name | Suggested rename | Where | Why |
|---|---|---|---|
| `BENCH_BATCH` | `HIPFIRE_BENCH_BATCH` | `bench_stream_overlap.rs` | bench-only |
| `BENCH_K` | `HIPFIRE_BENCH_K` | same | bench-only |
| `BENCH_M` | `HIPFIRE_BENCH_M` | same | bench-only |
| `BENCH_DRAFT_K` | `HIPFIRE_BENCH_DRAFT_K` | same | bench-only |
| `BENCH_DRAFT_M` | `HIPFIRE_BENCH_DRAFT_M` | same | bench-only |
| `BENCH_DRAFT_N` | `HIPFIRE_BENCH_DRAFT_N` | same | bench-only |
| `BENCH_DRAFT_LAYERS` | `HIPFIRE_BENCH_DRAFT_LAYERS` | same | bench-only |
| `BENCH_VERIFY_LAYERS` | `HIPFIRE_BENCH_VERIFY_LAYERS` | same | bench-only |
| `HFQ_TEST_N_ITER` | `HIPFIRE_HFQ_TEST_N_ITER` | gfx906_mmq_correctness | test-only |
| `HFQ_TEST_SCALE_LOG10` | `HIPFIRE_HFQ_TEST_SCALE_LOG10` | several test_*.rs | test-only |
| `HFQ_TEST_ZP_MAX` | `HIPFIRE_HFQ_TEST_ZP_MAX` | several test_*.rs | test-only |
| `MMQ_TEST_MODE` | `HIPFIRE_MMQ_TEST_MODE` | gfx906_mmq_correctness | test-only |
| `QWEN35_TEST_MODEL` | `HIPFIRE_QWEN35_TEST_MODEL` | test_inferenceQA.rs | test-only |
| `TINYLLAMA_GGUF` | `HIPFIRE_TINYLLAMA_GGUF` | test_gemv_q4kQA.rs | test-only |
| `MAX_TOKENS` | `HIPFIRE_MAX_TOKENS` | greedy_dump.rs | example-only; collides with OS-style env naming |
| `PROMPT_MODE` | `HIPFIRE_PROMPT_MODE` | greedy_dump.rs | example-only |
| `USE_SAMPLE` | `HIPFIRE_USE_SAMPLE` | a3b_multiturn_oneshot.rs | example-only |
| `NO_NGRAM` | `HIPFIRE_NO_NGRAM` | infer_vl.rs | example-only |
| `FP32_STATE` | `HIPFIRE_FP32_STATE` | infer_qwen35.rs | example-only |
| `DDTREE_TIMING` | `HIPFIRE_DDTREE_TIMING` | speculative.rs | diag-only, but in production library |
| `DEBUG_LAYERS` | `HIPFIRE_DEBUG_LAYERS` | qwen35.rs | **diag in production hot path** |
| `DFLASH_LIVE_TAU` | `HIPFIRE_DFLASH_LIVE_TAU` | dflash_spec_demo.rs | example-only |

Rename strategy: backward-compat shim for one release (read both names; emit a deprecation warning if old name set), then drop old names.

## `config.json` schema cross-reference

`HipfireConfig` in `cli/index.ts:157` (`CONFIG_DEFAULTS`). Validated by `validateConfigValue()`. ~40 keys. The CLI translates these into `HIPFIRE_*` env vars before spawning the daemon.

| Config key | Mapped env var | Notes |
|---|---|---|
| `kv_cache` | `HIPFIRE_KV_MODE` | direct |
| `flash_mode` | `HIPFIRE_ATTN_FLASH` | `auto`/`always`/`never` |
| `default_model` | `HIPFIRE_MODEL` | one-shot path |
| `temperature` | (msg field) | per-request |
| `top_p` | (msg field) | per-request |
| `repeat_penalty` | (msg field) | per-request |
| `max_tokens` | (msg field) | per-request |
| `max_seq` | `HIPFIRE_KV_PHYSICAL_CAP`-derived | indirect |
| `thinking` | (msg field) | per-request |
| `max_think_tokens` | (msg field) | per-request |
| `port` | (CLI arg) | not env |
| `idle_timeout` | (CLI arg) | not env |
| `experimental_budget_alert` | `HIPFIRE_EXPERIMENTAL_BUDGET_ALERT` | direct |
| `dflash_adaptive_b` | (msg field) | per-request |
| `dflash_mode` | (msg field) | per-request |
| `dflash_ngram_block` | `HIPFIRE_DFLASH_NGRAM_BLOCK` | direct |
| `cask_sidecar`, `cask`, `cask_*` | (msg fields) | per-request |
| `prompt_normalize` | `HIPFIRE_NORMALIZE_PROMPT` | direct |
| `mmq_screen` | `HIPFIRE_MMQ_SCREEN` | direct |
| `mmq_screen_threshold` | `HIPFIRE_MMQ_SCREEN_THRESHOLD` | direct |
| `prefill_compression` | `HIPFIRE_PREFILL_COMPRESSION` | direct |
| `prefill_threshold` | `HIPFIRE_PREFILL_THRESHOLD` | direct |
| `prefill_keep_ratio` | `HIPFIRE_PREFILL_KEEP_RATIO` | direct |
| `prefill_alpha` | `HIPFIRE_PREFILL_ALPHA` | direct |
| `prefill_min_keep` | `HIPFIRE_PREFILL_MIN_KEEP` | direct |
| `prefill_sink` | `HIPFIRE_PREFILL_SINK` | direct |
| `prefill_recent` | `HIPFIRE_PREFILL_RECENT` | direct |
| `prefill_block` | `HIPFIRE_PREFILL_BLOCK` | direct |
| `prefill_drafter` | `HIPFIRE_PREFILL_DRAFTER` | direct |
| `prefill_profile` | `HIPFIRE_PREFILL_PROFILE` | direct |
| `prefill_sparse_threshold` | `HIPFIRE_PREFILL_SPARSE_THRESHOLD` | direct |

`per_model_config.json` overlays the same key set per model tag (e.g. `"qwen3.5:27b": { "max_seq": 16384, "kv_cache": "q8" }`).

## Cargo features

Declared in `crates/hipfire-runtime/Cargo.toml`:

- `default = ["arch-qwen35", "arch-qwen35-vl", "arch-llama", "deltanet"]`
- `deltanet` — DeltaNet linear-attention support (Qwen3.5-MoE prerequisite).
- `arch-qwen35` — Qwen3.5/3.6 dense + MoE arch crate.
- `arch-qwen35-vl` — Qwen3.5-VL vision arch crate.
- `arch-llama` — Llama / Qwen3 / Mistral / generic dense.

Build with a subset for downstream library use:
```bash
cargo build --release --no-default-features --features "arch-qwen35 deltanet"
```

## Hidden gates beyond env vars

Behavior also branches on these implicit conditions, none of which are documented in env vars:

### Arch-detection branches (~dozens)

`self.arch == "gfx906"`, `arch.starts_with("gfx11")`, `arch.starts_with("gfx12")`, etc.

Fires across:
- `crates/rdna-compute/src/dispatch.rs` — kernel selection per arch
- `crates/rdna-compute/src/kernels.rs` — per-arch kernel matchers (Lloyd-MQ3 fast variants, etc.)
- `crates/hipfire-arch-qwen35/src/qwen35.rs` — forward-pass arch-specific paths

To enumerate which arch a code path takes, grep for `self.arch == "<gfx_id>"` and `starts_with("gfx<n>")`.

### File-existence gates (7 in runtime/arch crates)

Cache and model-discovery checks via `Path::new(...).exists()`. Includes:
- DFlash draft auto-discovery (`~/.hipfire/models/<target>-dflash-mq4.hfq`)
- Triattn sidecar discovery (`<model>.triattn.bin`)
- Model registry lookup
- Etc.

These don't expose env-var knobs; they're discovery-driven. Documented per call site.

## Triage suggestions

Tier categorization for the next pass (grouped by recommended action):

### TUI-promote (~8-12 candidates)

User-facing knobs that should appear in the daemon TUI config flow:

- `HIPFIRE_DETERMINISTIC` — byte-exact mode toggle
- `HIPFIRE_DPM_WARMUP_SECS` — bench-relevant warmup seconds
- `HIPFIRE_PREFILL_*` cluster — already in TUI partially
- `HIPFIRE_PP_LAYERS`, `HIPFIRE_PP_PFLASH` — multi-GPU operator knobs
- `HIPFIRE_FORCE_A3B_EVICTION` — already TUI

### Document-only (~30-40)

Kernel-selectors, MMQ tuning, NGram detector, GraphMoE, KV physical cap. Power-user surface.

### Retire candidates (~15-20)

Likely-dead research artifacts:
- `HIPFIRE_DDTREE_*` cluster (9 vars) — Path D empirically dominated, see CLAUDE.md memory
- `HIPFIRE_DFLASH_SEED_ORACLE` — Phase B oracle research, scrapped
- `HIPFIRE_DDTREE_PATH_B_CAPTURE` — same Path D track
- `HIPFIRE_DRAFT_SUBPHASE`, `HIPFIRE_DRAFT_GEMM_DUMP` — diag dumps tied to deprecated work
- `HIPFIRE_VERIFY_GRAPH_TREE` — tree-mode graph variant, dominated path

Verify each via `git log -S 'HIPFIRE_XXX' --since=3.months.ago` to confirm last-touched is older than the close-out date for the related work.

### Diagnostic-only (~25)

Production-safe but bench/diag-only. Stay env-only with a `DIAGNOSTIC` flag in this doc:
- All `HIPFIRE_*_DUMP`, `HIPFIRE_*_PROFILE`, `HIPFIRE_PROMPT_HEAT_*`
- `HIPFIRE_VERIFY_GRAPH_TIMING`, `HIPFIRE_HOST_TIMING`
- `HIPFIRE_BLOB_FORCE`, `HIPFIRE_REPLAY_GRAPH`
- `HIPFIRE_SAMPLE_COMPARE`, `HIPFIRE_SPEC_PHASES`

## Adding a new env var

When adding a new `HIPFIRE_*` env var, also:

1. Add a row to the quick-reference table above. Run `scripts/regen-env-vars-doc.sh` (forthcoming) to rebuild the table mechanically.
2. Add an entry to the relevant category guide section. One-line description; longer prose only if non-obvious.
3. If it's user-facing, add a `cfg.<key>` to `HipfireConfig` in `cli/index.ts:157` so the TUI can set it.
4. If it's diagnostic-only, prefix with `HIPFIRE_*_DUMP`, `HIPFIRE_*_PROFILE`, or similar so this doc's category guide picks it up automatically.

A pre-commit hook to enforce this is on the roadmap (issue forthcoming).

## Maintenance

To regenerate the quick-reference table:

```bash
# Auto-extract from source. Covers env::var(), env::var_os(), process.env.X
grep -rE 'env::var(_os)?\("HIPFIRE_|process\.env\.HIPFIRE_' \
    $(git ls-files | grep -E '\.(rs|ts)$') \
    | grep -oE '(env::var(_os)?\("[A-Z_0-9]+"\)|process\.env\.[A-Z_0-9]+)' \
    | sed -E 's/env::var(_os)?\("//; s/"\)//; s/process\.env\.//' \
    | sort -u
```

Note the `(_os)?` group — `compiler.rs:89` uses `std::env::var_os("HIPFIRE_KERNEL_CACHE")` rather than the more common `std::env::var(...)`. A regex that only matches `env::var(` will silently miss it; this was caught post-merge by Codex stop-gate review and is the reason this regex now covers both forms.

A future pass should ship `scripts/regen-env-vars-doc.sh` that mechanically rebuilds the quick-reference table while preserving the prose category guide.

Last manual pass: 2026-05-07.
