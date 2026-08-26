# DeepSeek V4 Flash 0731 registry and thinking enablement

Date: 2026-08-05  
Branch: `ds4-beta-staging`  
Implementation commits: `7638148bd`, `29ff746cc`

## Result

DeepSeek V4 Flash 0731 now owns the canonical `deepseek-v4-flash` / `deepseek4`
registry identity. The matching MQ2R product is `deepseek-v4-flash:mq2r` /
`deepseek4:mq2r`. The prior preview package remains addressable only as
`deepseek-v4-flash-preview`; there is deliberately no preview MQ2R entry or
alias.

The parent checkpoint's reasoning-effort contract is carried end to end:

- `low`: thinking enabled with no extra effort prefix;
- `high`: the checkpoint's exact high-effort prefix;
- `max`: the checkpoint's exact maximum-effort prefix;
- `none`: closed-think / non-thinking framing.

Effort and token budget are independent. The 0731 general and coding profiles
use `thinking_budget=uncapped`, so `max` does not silently become a hipfire
32K cap. An explicit request cap remains supported up to 393,216 tokens, and
the typed maximum sequence limit is 1,048,576.

## Published artifacts

Repository: `hipfire-models/hipfire-deepseek-v4-flash-0731`  
Repository revision after publication: `584fb6b143e7031e8b8703df7355b408d733f494`

| File | Bytes | LFS SHA-256 |
|---|---:|---|
| `deepseek-v4-flash-0731.mq2r` | 82,191,359,851 | `cbf2bbcfa3f47b1712a071836b2c48232dad7dfb763813a720f7d348a9318cce` |
| `deepseek-v4-flash-0731-dspark.mq2r` | 5,788,397,278 | `bc695a000643801d26e5ae96c9f4ac4c222a36d9db40566f4cc1de0e9d3d5d2e` |

Only these 0731 MQ2R artifacts were uploaded in this change. No preview MQ2R
artifact was uploaded.

## Registry sampling profiles

| Profile | Temperature | Top-p | Effort | Thinking cap |
|---|---:|---:|---|---|
| `general` | 1.0 | 1.0 | `low` | uncapped |
| `coding` | 1.0 | 0.95 | `max` | uncapped |
| `instruct` | 1.0 | 1.0 | `none` | off |

The HTTP adapter keys the parent-specific behavior from the loaded daemon
architecture. The existing Qwen effort-to-budget mapping is retained, and the
focused unit test proves Qwen `high` still maps to 4,096 tokens while DeepSeek
`max` remains uncapped absent an explicit cap.

## Live gfx1151 serve proof

Validation route: user-facing serve semantics via `scripts/serve_harness.py`.
This is a capability/coherence check, not a performance promotion claim.

- GPU: gfx1151, 103.1 GB addressable VRAM, HIP 7.14.
- Model: `deepseek-v4-flash-0731.mq2r`, SHA-256
  `cbf2bbcfa3f47b1712a071836b2c48232dad7dfb763813a720f7d348a9318cce`.
- Commit: `29ff746ccc7a736861c633c61bc5b8e4c5fe0adf`.
- CLI SHA-256: `f59b2f99ae3276edbb50937fa68f4f6c964b7c71ca1fac0c548b237195923891`.
- Daemon SHA-256: `d112a15e370942f7a3aa550b09c142be578687c9cd4a970c8e4ecb0067ad46fc`.
- Fixture: `benchmarks/prompts/ds4_0731_thinking_low.json`, prompt-text MD5
  `37d74e9455aaebc9c490a4925eab8668`.
- Configuration: registry `general`, low effort, uncapped thinking, Q8 request
  mode, contiguous KV, speculation off, seed 1, max sequence 32,768, maximum
  output 1,024.

Observed result: `finish=stop`, 110 generated tokens, 40 reasoning words and
25 answer words, zero empty/runaway/attractor flags, and recall 1/1. The visible
answer was coherent and correct: the final price was `$66`. Decode was measured
at 27.0204 tok/s, but no performance claim is made from this one semantic run.

The serve log proves `GPU dev 0: gfx1151`, verifies the MQ2R P3 tensor recipe,
selects the gfx1151 route v2, and reports automatic compressed-cache VMM growth
to the advertised 1,048,576-token context.

## Evidence

Durable evidence on hipx:

`/home/kaden/ds4-gfx1151-evidence/2026-08-05-ds4-0731-thinking-low/`

- `result.json` SHA-256:
  `eeb84a19ba17d414c27475e97faa713d007e425d1d60fe7aa28f89b7a7659102`
- `serve.log` SHA-256:
  `d8e689ba2cc263808106e8f4e6cdbc257aab3b429570175b4981e8f09ae086be`

## Validation and exclusions

Passed:

- 51 `hipfire-config` tests and 9 `hipfire-registry` tests;
- focused CLI Qwen/DeepSeek reasoning-contract test;
- focused runtime effort parser and exact DeepSeek prefix tests;
- registry generator network check against the published artifacts;
- serve-harness self-tests and four changed-config compatibility tests;
- live low-effort gfx1151 serve semantic check with decoded output inspected.

The full no-GPU run passed its Rust check, the changed CLI test surface, and
372 Python CPU tests. Two unrelated Redline harness tests still expect the old
two-field prompt row even though `serve_harness.py` already returned the
three-field row at the branch's parent commit; they were not changed here.

Skipped: BenchLocal capability certification, multi-turn thinking, live high
and max effort, DSpark thinking, long-context generation, and Qwen GPU testing.
No kernel, weight, quantization, replay route, sampling arithmetic, or Qwen-owned
function body changed.
