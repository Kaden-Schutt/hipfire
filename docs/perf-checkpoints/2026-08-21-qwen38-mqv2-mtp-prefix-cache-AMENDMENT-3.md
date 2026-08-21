# Amendment 3 — compressed MTP full-context session

**Lifecycle:** historical
**Disposition:** long-context correctness/lifecycle evidence; not a cross-arm performance comparison
**Amends:** [`2026-08-21-qwen38-mqv2-mtp-prefix-cache.md`](./2026-08-21-qwen38-mqv2-mtp-prefix-cache.md) and [`AMENDMENT-2`](./2026-08-21-qwen38-mqv2-mtp-prefix-cache-AMENDMENT-2.md)

## Reason for amendment

Amendment 2 established an 11.7% short-fixture throughput gain for the corpus-derived 32K compressed MTP draft head on gfx1201 and checked five independent greedy prompts. This amendment exercises the same artifact across the complete eight-turn coding conversation with sampled thinking and accumulated prefix-cache state.

## Method

- Host: `hiptrx`, HIP device 3, Radeon AI PRO R9700 gfx1201, BDF `0000:13:00.0`, HIP 7.14
- Target: MQ4V2 XT, md5 `e45d15bfe0c9a87132697101d17cbed6`
- Compressed MTP sidecar: md5 `d393fcc8bc4b718ec82ecec81287d83a`
- `hipfire`: md5 `4b8c6f03465eaf0f339b92bdffea3b2d`
- daemon: md5 `bbd3a1defd1fa2c620321e4352c1fb18`
- MTP: K7, `p_min=0`, proposal graph off, verify graph off
- KV: Q8 VMM
- Sampling: registry `temperature=1.0`, `top_p=0.95`, `top_k=20`, `min_p=0`, neutral penalties
- Reasoning: `xhigh`, uncapped thinking
- Limits: no explicit `--max-seq` or `--max-tokens`; registry resolved 262,144 context and 81,920 output tokens
- Seed: 7

The active staged fixture `/home/kaden/qcal/session_coding.json` and the requested `/home/kaden/mv/session_coding.json` were verified byte-identical, both md5 `c0d470288bde3f1e54e4bba04da8f8a2`.

## Results

| turn | request ctx | cached tokens | think words | decode tok/s | tau | finish |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 49 | 0 | 647 | 36.3 | 1.68 | stop |
| 2 | 3,929 | 3,880 | 514 | 28.4 | 2.05 | stop |
| 3 | 7,362 | 7,321 | 1,336 | 26.2 | 1.86 | stop |
| 4 | 12,704 | 12,665 | 791 | 19.5 | 2.11 | stop |
| 5 | 17,138 | 17,097 | 1,166 | 25.4 | 1.70 | stop |
| 6 | 21,286 | 21,260 | 237 | 3.1 | 2.19 | stop |
| 7 | 21,837 | 21,782 | 726 | 8.5 | 2.18 | stop |
| 8 | 23,974 | 23,941 | 149 | 7.3 | 1.92 | stop |

All eight turns reported `mtp=true`. Prefix caching engaged on every extension turn. There was no empty output, runaway, attractor, or retrieval miss. Mean per-turn decode rate was 19.34 tok/s and mean tau was 1.961; these means span materially different contexts and output lengths and are not a throughput A/B.

Visible output was inspected through the final turns. The model produced the requested deduplication haiku, correctly identified the earlier streaming hash function, and returned a coherent full-session summary containing every strict retrieval term.

Evidence:

- `/home/kaden/qcal/mtp-cvs32k-session-gfx1201.json`
- `/home/kaden/qcal/mtp-cvs32k-session-gfx1201.log`
- `/home/kaden/qcal/mtp-cvs32k-gfx1201/qwen3.8-27b.mq4v2.xt-cvs32k.mtp`

The direct gfx1151 throughput proof remains a separate hardware gate; this result establishes compressed-head correctness and prefix-cache durability through approximately 24K request context on gfx1201.
