# PFLASH Deferred Items

Nice-to-haves and follow-ups that are out of scope for the current PFlash run.

## Tokenizer encode is O(N²)-ish at long context

- Where: `engine::tokenizer::Tokenizer::encode` on the 8K NIAH fixture took
  30511 ms for 5487 tokens (~180 tok/s). Linear extrapolation suggests 65K
  takes 6+ min and 131K takes 12+ min.
- Why deferred: not a PFlash correctness issue; the prefill itself runs at
  3139 tok/s. Bench TTFT becomes dominated by tokenize at long context but
  the bench faithfully reports the breakdown. PFlash compression will run
  on already-tokenized inputs from the daemon path, which doesn't re-encode.
- Workaround for bench at 64K/128K: pretokenize fixtures and embed token IDs
  in the JSONL alongside `filler_text`, add `--pretok` flag to skip encode.
- Track in tokenizer-perf work, not PFlash.

## ~~Pre-tokenized NIAH fixtures~~ DONE (4528013)

- Shipped via `pflash_niah_bench --write-pretok` / `--pretok`.
- Companion file `<fixture>.tok.jsonl` carries `tokens` array,
  `tokens_md5`, `tokenizer_signature` (rejected on signature mismatch),
  `question`, `expected_answer_substring`, `source_fixture_md5`.
- 8K committed at `benchmarks/longctx/niah/niah_8k.tok.jsonl`. 16K, 32K
  to follow once their one-time encode finishes (16K: ~110s, 32K: ~6 min,
  64K: ~24 min, 128K: ~96 min). Each is encoded once and reused
  permanently.
- Pretok readback at 8K cuts a 30+ s bench iteration to 1.3 s.

## Multi-needle NIAH fixtures (PRD §6 Phase 5)

- PRD requires multi-needle sanity at 16K and 64K. Current fixtures are
  single-needle (one secret pass code embedded once). Need a generator
  variant that scatters K needles across the document and asserts at
  least M survive compression.
- Suggested `benchmarks/longctx/niah/multi/niah_multi_<N>k.jsonl` with
  fields: `needles: [{secret, position}]`, `expected_substrings: [...]`,
  `min_recovered: M`.

## ~~3-fresh-process TTFT methodology for PFlash claims~~ DONE (d9af202)

- Shipped via `scripts/pflash-niah-bench.sh`. Drives N fresh processes
  per (target, drafter, fixture, mode) point and emits min / median /
  max + spread per timing bucket. Spread > 5 % flags contamination.
- Use it for any PFlash perf claim that's going into a release note or
  PR description. Single-shot is OK for pass/fail (needle survives) but
  not for tok/s claims.
