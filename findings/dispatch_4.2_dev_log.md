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
| Prefill byte-parity (hidden-state diff) | PENDING | Requires `HIPFIRE_DUMP_HIDDEN` comparison between pre-4.2 (`31738389` or earlier) and HEAD |
| `probe_commits.sh` prefill tok/s ±1-3% | PENDING | ≥256-token prompt needed; gfx1151 only (no gfx1100 or gfx1201 available) |
| `coherence-gate.sh --full` (A3B cells) | PENDING | Short gate passed; full gate requires A3B-specific prompts |
| Path-1 force-smoke (`HIPFIRE_MOE_GROUPED_GEMM=0` on gfx11) | **PASS** | gfx1151 with `HIPFIRE_MOE_GROUPED_GEMM=0`: A3B loads, runs, decode clean. Prefill=32 tok/s ~60.5 (same as Path 2 at small batch — both I/O bound). No panics. |
| A3B MoE DFlash pinned fixture | SKIP | Draft `qwen36-35b-a3b-dflash-mq4.hfq` not present on this host. Target file present (md5 pending — NFS timeout on 15 GB file). |
| Paro/MQ6 A3B fixtures | SKIP | No Paro/MQ6 A3B models on this host |

## Notes

- Coherence-gate short battery passed (5 cells: 0.8b/cap, 4b/code, 9b/reason, 9b/tool-call, 9b/mq3) — no hard errors.
- DFlash coherence battery passed (2 cells: 27b-dflash-prose, 27b-dflash-code) — no hard errors.
- A3B MoE model loads and runs without panics. Prefill=32 tok/s = 60.5 (includes JIT; gfx1151 APU bandwidth-constrained at 21 GiB model).
- `MOE_GROUPED_BLOCK_M` = 16 constant duplicated between qwen35.rs and dispatch pipeline — both must stay in sync. Grep audit confirms only these two sites.
