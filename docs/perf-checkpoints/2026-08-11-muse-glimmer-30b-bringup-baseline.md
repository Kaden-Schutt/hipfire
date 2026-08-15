# Muse Glimmer 30B — bring-up decode baseline (gfx1100 / gfx1151 / gfx1201)

**Lifecycle:** `historical`. This is a bring-up baseline captured the day the
arch first decoded coherently. It is evidence under the exact fixture below,
**not** a current default, an automatic baseline, or an admission decision.

**Disposition:** first coherent measurement of arch 14. Recorded so later
Glimmer optimization has a fixed, digest-bound point of comparison. No product
claim is made or authorised from these numbers.

**Workload:** Muse Glimmer 30B (`muse_glimmer`, arch 14), MQ4, ordinary greedy
autoregressive decode. No speculative decoding, no MTP, no prefix cache
(`cache_capable: false` for this arch).

**Branch:** `muse-glimmer` @ `76123215d`, on `beta` = `b65f8159c`.

---

## Fixture

| field | value |
|---|---|
| model | `~/.hipfire/models/muse-glimmer-30b.mq4` (15.5 GB) |
| prompt | `benchmarks/prompts/glimmer_bringup_merge.txt` |
| prompt md5 | `2ef49ee70df1483079b1f73c1f768339` (75 bytes) |
| binary md5 | `e40a86d4f49e6ff9dc9a08ce9f641e26` (identical on both hosts) |
| tokens | 64, greedy |
| process | fresh daemon per run, 3 runs per arm |

## Result

| GPU | arch | tok/s (3 fresh runs) | median | prefill ms |
|---|---|---|---:|---:|
| RX 7900-class, hipx node1 | `gfx1100` | 37.74 · 37.69 · 37.83 | **37.74** | 417 |
| R9700, hiptrx GPU 0 | `gfx1201` | 31.57 · 31.56 · 31.56 | **31.56** | 488 |
| Strix Halo, hipx node2 | `gfx1151` | 13.63 · 13.58 · 13.70 | **13.63** | 1110 |

Spread is ±0.1 tok/s across fresh processes on every arm. Per CLAUDE.md rule 3 a
tight stddev is normally suspicious, but that rule is about *speculative decode*
acceptance noise; this is plain AR decode with no acceptance term, where a tight
spread is the expected shape. The decoded text was eyeballed on every arm.

## Roofline context

Dense decode re-reads the whole 15.5 GB weight set per token, so the ceiling is
bandwidth/size:

| GPU | ~peak BW | roofline tok/s | measured | efficiency |
|---|---:|---:|---:|---:|
| `gfx1100` | ~960 GB/s | ~62 | 37.74 | ~61% |
| `gfx1201` | ~640 GB/s | ~41 | 31.56 | ~77% |
| `gfx1151` | ~256 GB/s (shared LPDDR5X) | ~16.5 | 13.63 | ~83% |

`gfx1100` is the weak arm. Peak-bandwidth figures are vendor nameplate numbers,
not measured on these parts, so treat the efficiency column as indicative.

## Known headroom (not yet applied)

Decode issues **four separate GEMV dispatches per layer** — `q_proj`, `k_proj`,
`v_proj` and the Glimmer-specific `attn_gate_proj` — all reading the *same*
normed input. At 52 layers that is 208 dispatches/token before the FFN, norms,
RoPE, and attention are counted. Weight traffic dominates, so fusing does not
reduce bytes read; it removes launch overhead and redundant activation reads,
which is the plausible explanation for `gfx1100` sitting furthest below its
roofline. Fusion is tracked separately and is NOT reflected in the table above.

## hipGraph

`HIPFIRE_GRAPH=1` was validated on this arch and is **byte-identical** to graph
OFF on all three GPUs, at 1.00× the throughput on each. The flag is currently a
no-op for the Glimmer decode path rather than a working capture, which is the
safe state — the Gemma4 port shipped with capture defaulted OFF precisely because
a captured decode there produced fluent garbage. Do not read this row as
"hipGraph works for Glimmer"; read it as "hipGraph does not engage for Glimmer."

## Reproduce

```bash
ssh <host> 'cd /tmp && HOME=/home/kaden/glimmer-home-<dev> \
  GLIMMER_DAEMON=/home/kaden/wt-glimmer/target/release/examples/daemon \
  GLIMMER_PROMPT_FILE=/tmp/glimmer_prompt.txt \
  HIP_VISIBLE_DEVICES=<dev> \
  python3 /tmp/glimmer-smoke.py x 64'
```
