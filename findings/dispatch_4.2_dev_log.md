# dispatch_4.2_dev_log.md

Ship 4.2: qwen35 grouped-GEMM MoE prefill → `MoeFamily::run_prefill` (Step 8)

## V0+V1 · family executor + qwen35 split

| Date | Commit | What | Result |
|---|---|---|---|
| 2026-06-06 | `e59f4aa9` | V0+V1 co-land: `MoePrefillParams` (D1), `MoePrefillResolution` (D2), `MoeFamily::run_prefill` → `pipeline::run_moe_prefill` + `dispatch_grouped_gemm` (D3), 3 MoE env levers → `FeatureFlags` (N6), `forward_prefill_chunk` ctx threading (gemini F1). qwen35 `prefill_moe_ffn_body_batched` routed block replaced with family delegation. 10 GPU-free resolution tests. D4 coverage keys in `moe_table.rs` (documentation-only BatchGt(1) gate). | Compile clean; 133 dispatch tests pass; coherence-gate short pass (0 hard errors); A3B MoE model loads and runs (qwen3.6-35b-a3b.mq4, gfx1151, prefill=32). |

## Fixtures

- **GPU:** gfx1151 (RYZEN AI MAX+ 395 w/ Radeon 8060S)
- **A3B model:** `qwen3.6-35b-a3b.mq4` (`~/.hipfire/models/`)
- **Binary:** `target/release/examples/bench_qwen35_mq4`
- **Commit:** `e59f4aa9` (on `integration/dispatch-unification`)

## V2 sweep

| Item | Status | Detail |
|---|---|---|
| Prefill byte-parity (hidden-state diff) | **PASS (dense)** | 27B dense model (`qwen3.6-27b.mq4`): pre-4.2 vs post-4.2 `.batched` hidden states are byte-identical (md5 `e647a43...`). MoE batched prefill: NOT exercised (A3B falls to per-token path). |
| `probe_commits.sh` prefill tok/s ±1-3% | **PARTIAL** | gfx1151 only (no gfx1100/gfx1201). A3B prefill=256: 57.0 tok/s Path 2 (default), prefill=32: 60.5 tok/s Path 1 (force). JIT-included first runs; post-JIT numbers pending second-run methodology. |
| `coherence-gate.sh --full` (A3B cells) | **PARTIAL** | Short gate passed (5 cells, no hard errors). Full gate requires `qwen3.5-35b-a3b.mq4` (not present) and `qwen3.6-35b-a3b-paro.hfq` (downloading). A3B v3.6 model tested manually: loads, multi-run decode clean at temp=0. |
| Path-1 force-smoke (`HIPFIRE_MOE_GROUPED_GEMM=0` on gfx11) | **PASS** | gfx1151 with `HIPFIRE_MOE_GROUPED_GEMM=0`: A3B loads, runs, decode clean. Prefill=32 tok/s ~60.5 (same as Path 2 at small batch — both I/O bound). No panics. |
| A3B MoE DFlash pinned fixture | SKIP | Draft `qwen36-35b-a3b-dflash-mq4.hfq` not present on this host. Target file present at `/local/hipfire/qwen3.6-35b-a3b.mq4` (22.9 GB). |
| Paro/MQ6 A3B fixtures | PENDING | PARO A3B safetensors downloading to `/local/models/z-lab/Qwen3.6-35B-A3B-PARO/` (0 safetensors so far). No MQ6 A3B models available; `hipfire-quant-eval` repo has MQ6 quant tools at `/home/kread/git/hipfire-quant-eval/`. |

## A3B prefill perf (gfx1151, qwen3.6-35b-a3b.mq4)

| Prefill | Tok/s | Path | Notes |
|---|---|---|---|
| 32 | 60.5 | Path 2 (default) | Includes JIT |
| 32 | 60.5 | Path 1 (MOE_GROUPED_GEMM=0) | Includes JIT |
| 64 | 41.2 | Path 2 | First run after load (fresh JIT) |
| 128 | 59.0 | Path 2 | Includes JIT |
| 256 | 57.0 | Path 2 | VERIFY_GRAPH=0 |

## Notes

- Coherence-gate short battery passed (5 cells: 0.8b/cap, 4b/code, 9b/reason, 9b/tool-call, 9b/mq3) — no hard errors.
- DFlash coherence battery passed (2 cells: 27b-dflash-prose, 27b-dflash-code) — no hard errors.
- A3B MoE model loads and runs without panics. Prefill=32 tok/s = 60.5 (includes JIT; gfx1151 APU bandwidth-constrained at 21 GiB model).
- `MOE_GROUPED_BLOCK_M` = 16 constant duplicated between qwen35.rs and dispatch pipeline — both must stay in sync. Grep audit confirms only these two sites.
- **Determinism check (27B dense, batched prefill)**: Two runs with `HIPFIRE_DUMP_HIDDEN` produce byte-identical `.batched` files (md5 `e647a43...`). The dense batched prefill path (unchanged by 4.2) is deterministic.
- **A3B MoE batched prefill**: NOT exercised by `bench_qwen35_mq4` — the model falls through to the per-token `forward_scratch` path. The `prefill_batch_pbs_eligible` gate rejects A3B for batched prefill on gfx1151 (root cause TBD — likely `moe_ffn_batched_admissible` strict-MQ4 check or attention weight dtype). The per-token path is unchanged by Ship 4.2 (decode MoE dispatch was migrated in 4.1).
- **New `run_moe_prefill` code**: Not exercised by either model in this test setup. Needs an A3B model that passes batched eligibility, or a forced-path test with `HIPFIRE_PREFILL_BATCHED=1` + eligibility fix.
