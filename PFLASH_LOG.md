# PFLASH LOG

Append-only progress log. Entries are timestamped + reference commit SHAs.

## 2026-05-02

### Start

- Branch `feat/89-llama-batched-prefill` at e17684b (Phase A-D landed).
- PFlash PRD: `docs/plans/pflash-speculative-prefill.prd`.
- Contract: `PFLASH_CONTRACT.md`.
- Drafter target: Qwen3-0.6B. Existing artifacts: `~/.hipfire/models/qwen3-0.6b.hf4` (HFQ4G256) and `qwen3-0.6b.mq4` (MQ4G256).
- Tokenizer: per Phase A smoke, qwen3-0.6b.hf4 metadata produces a tokenizer that matches the model. Need to verify cross-tokenizer compatibility with target later.

### Phase 0: NIAH harness + full-prefill baseline (in progress)

Goal: make long-context target measurable before touching runtime behavior.

Deliverables (per PRD §6 Phase 0):
- `benchmarks/longctx/niah/niah_{8k,16k,32k,64k,128k}.jsonl` (committed or generator-deterministic).
- `crates/engine/examples/pflash_niah_bench.rs` (full-prefill baseline, prints TTFT breakdown + md5s).

Acceptance:
- Full-prefill NIAH baseline passes at every supported context size given available VRAM.
- Bench reports TTFT broken into tokenize / prefill / first decode step / total.
- Source md5 + binary md5 logged.
