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

### Phase 0 results (qwen3.5-4b.mq4, gfx1100, asym3 KV)

```
8K   fixture md5 6f24cd79...  ttft 32259 ms  prefill 1748 ms (3139 tok/s)  decode 161.5 tok/s  PASS (mauve-velociraptor-7741 retrieved)
       tokenize 30511 ms (5487 tokens) — see DEFERRED.md, tokenizer perf is the bench TTFT bottleneck, not prefill
```

Phase 0 status: **DONE for 8K**. 16K-128K runs deferred until Phase 0/Phase 1 share the same harness path; Phase 1's compression demo will exercise larger contexts where tokenize-vs-prefill curves matter most.
