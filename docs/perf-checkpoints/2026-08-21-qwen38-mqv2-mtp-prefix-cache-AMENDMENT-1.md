# Amendment 1 — gfx1201 full-session completion

**Lifecycle:** historical
**Disposition:** fixture-bound validation evidence; not a product baseline or promotion gate
**Amends:** [`2026-08-21-qwen38-mqv2-mtp-prefix-cache.md`](./2026-08-21-qwen38-mqv2-mtp-prefix-cache.md)

## Reason for amendment

The original checkpoint recorded gfx1201 as unavailable and therefore left all nine gfx1201 full-session arms incomplete. The separate `hiptrx` host was subsequently confirmed reachable with four Radeon AI PRO R9700 devices. This amendment adds the completed gfx1201 evidence without modifying the original immutable record.

## Fixtures and method

The target, sidecar, draft, prompt, source, and binary identities match the original checkpoint:

| item | identity |
|---|---|
| MQ4V2 XT | md5 `e45d15bfe0c9a87132697101d17cbed6` |
| MQ4V2 Base | md5 `d1292b4d5bd6046693604201a6ca8074` |
| MQ4V2 Pro | md5 `279c563786499c6651de6a3a57b42b02` |
| MTP sidecar | md5 `1a78dd0d2c8c8a97abfc2f873193ae58` for all tiers |
| DFlash2 MQ4V2 draft | md5 `013395583cd04206c8aa68f4d061983d` |
| `session_coding.json` | md5 `c0d470288bde3f1e54e4bba04da8f8a2` |
| `hipfire` | md5 `4b8c6f03465eaf0f339b92bdffea3b2d` |
| `daemon` | md5 `bbd3a1defd1fa2c620321e4352c1fb18` |
| source | `21141cce6` plus checkpoint commit `49c30ec1d` |

Hardware was `hiptrx`, HIP 7.14:

- XT: HIP device 0, gfx1201 R9700, BDF `0000:03:00.0`
- Base: HIP device 1, gfx1201 R9700, BDF `0000:c3:00.0`
- Pro: HIP device 2, gfx1201 R9700, BDF `0000:e3:00.0`

The three tier campaigns ran concurrently on distinct GPUs. A per-process guard polled the assigned PCI function and aborted if it disappeared. No endpoint disappeared.

Protocol matched the original full-session section: native `scripts/serve_harness.py --mode session`, registry tag `qwen3.8:27b`, registry sampling (`temperature=1.0`, `top_p=0.95`, `top_k=20`, `min_p=0`, neutral penalties), `--thinking-effort xhigh`, uncapped thinking, Q8 VMM KV, and seed 7. No explicit `--max-seq` or `--max-tokens` was passed; registry policy resolved 262,144 context tokens and an 81,920-token output allowance. Verify and MTP proposal graphs were explicitly off. Every completed turn self-terminated.

## Results

`avg tok/s` is the arithmetic mean of the eight per-turn decode rates. `turn-8 cached` proves the accumulated prefix was reused through the final turn; every arm also had nonzero cache reuse on turns 2–7. Every arm had 8/8 thinking turns, no empty output, no runaway, no attractor, and the requested route flag on all eight turns.

| tier | route | avg tok/s | avg tau | turn-8 cached | strict lexical gate |
|---|---|---:|---:|---:|---|
| XT | AR | 32.35 | — | 28,070 | `dedupe` absent on turns 7/8 |
| XT | DFlash | 26.71 | 2.453 | 33,306 | `dedupe` absent on turns 7/8 |
| XT | MTP | 27.83 | 1.701 | 20,287 | `dedupe` absent on turns 7/8 |
| Base | AR | 30.65 | — | 27,150 | `dedupe` absent on turn 7 |
| Base | DFlash | 27.89 | 2.866 | 33,307 | pass |
| Base | MTP | 18.46 | 1.765 | 28,572 | pass |
| Pro | AR | 29.01 | — | 31,906 | pass |
| Pro | DFlash | 29.00 | 2.633 | 21,537 | `dedupe` absent on turn 8 |
| Pro | MTP | 26.54 | 1.781 | 36,090 | pass |

The flagged visible answers were inspected. They correctly recalled `hash_file`, `hash_stream`, or the session's deduplication tool and summarized content-hash deduplication, but omitted the fixture's exact token `dedupe`. They remain recorded as lexical misses rather than rerun away.

Full decoded output and per-turn metadata:

- `/home/kaden/qcal/full-sessions-gfx1201-xt/`
- `/home/kaden/qcal/full-sessions-gfx1201-base/`
- `/home/kaden/qcal/full-sessions-gfx1201-pro/`

## Updated campaign disposition

All nine gfx1201 XT/Base/Pro × AR/DFlash/MTP arms are complete with thinking, registry/full output allowances, and prefix-cache evidence. The original checkpoint's other blockers remain unchanged: gfx1100 Base/Pro MTP and gfx1151 Pro DFlash/all MTP were interrupted by the gfx1100 link-down, and the gfx1151 40 tok/s target remains unachieved.
