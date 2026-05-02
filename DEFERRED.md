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

## Pre-tokenized NIAH fixtures

- Add a `tokens` array to each `niah_<N>k.jsonl` so the bench can skip the
  slow encode path. Generator must record the tokenizer + model_md5 used so
  re-running with a different model rejects mismatched tokens.
