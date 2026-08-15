# 2026-08-07 — asym3 KV quality gate (SP1 Task 9, spec §13)

Decides whether `asym3` (3-bit Givens-rotated K + Q8_0 V) may become the
**batched default** for `qwen3.6:27b` and `qwen3.6:35b-a3b`. Both models ship
`q8` today; asym3 was brought into SP1 scope because it is what
`HIPFIRE_KV_MODE=auto` resolves to, and because at 0.6875× the KV bytes it is
the lever that makes 4 agents × 128K context fit in a 32 GB R9700.

**Verdict up front: asym3 does NOT become the default for either model. It stays
opt-in via `HIPFIRE_KV_MODE=asym3`.**

## Provenance

Three implementing agents were killed by API stalls during this task. The
harness `crates/hipfire-runtime/examples/kv_mode_kld.rs` survived and is theirs;
**the measurements below were run by the controller** directly, under
`scripts/run-bounded.sh`. Results the dead agents reported verbally but never
wrote to disk were discarded and re-run rather than quoted.

## Step 1 — resolved-mode gate (PASSED)

Spec §4.4: `QWEN35_PARO_POLICY` omits `Asym3` from its `accepted` set, so
`HIPFIRE_KV_MODE=asym3` can silently resolve to q8, which would make the whole
comparison q8-vs-q8 and meaningless. Confirmed distinct on every arm. The
candidate arm logs:

```
KV cache: asym3 (K rotated-3b 100B + V Q8 272B = 372 B/head, 5.5x vs fp32, ...)
```

K at 100 B/head against V's 272 B/head is the 3-bit rotated K — the asym3 path
is genuinely active.

## Method

`kv_mode_kld` loads the model **once**, then:

1. Free-generates greedily under **q8** from a real prompt, recording the token
   sequence and full-vocab logits at every step. This is the model's *own*
   output, satisfying the method constraint — not a canned reference completion,
   which would flatter the quantised arm.
2. Drops the KV cache, rebuilds under **asym3**, and **teacher-forces the same
   sequence** (same tokens, same positions), recording logits at the same steps.

Teacher-forcing on the baseline's own trajectory removes the "both arms diverge
in token choice after step k, so what is being compared" confound and is what
makes a full-softmax KLD meaningful here.

Prompt: a real code-generation request (ISO-8601 parser in Rust, with error
handling discussion) — deliberately not synthetic filler, which is
out-of-distribution and degenerates on *both* arms, reading as a false
quantisation failure.

Commands (both under the memory gate, `HIPFIRE_MEM_CAP=28G`):

```bash
./scripts/run-bounded.sh ./target/release/examples/kv_mode_kld \
  ~/.hipfire/models/<model> /tmp/kld_<m>.json \
  --max-gen 64 --max-seq 4096 --baseline q8 --candidate asym3 "<prompt>"
```

## Results

| metric | 27B | 35B-A3B |
|---|---|---|
| mean KLD(q8‖asym3) | 0.297 | **0.552** |
| median KLD | 0.148 | 0.147 |
| p99 KLD | 2.49 | **4.14** |
| mean KLD(asym3‖q8) | 0.342 | 0.975 |
| **top-1 agreement** | **68.8%** | **70.3%** |
| mean top-5 overlap | 3.94 / 5 | 3.34 / 5 |
| first divergence | step **2** | step **1** |
| steps compared | 64 | 64 |

## Why this is a rejection

**~30% of top-1 token choices change.** That is the number that matters more
than the KLD figures. For a coding agent, roughly one token in three being
chosen differently is a substantial behavioural change, not a quality nuance.
Both models diverge within the first two generated tokens.

The mean/median split is why the plan demanded both. The medians are nearly
identical (0.148 / 0.147) while the means differ by 1.9× and the p99s by 1.7×:
the 35B has a much fatter tail — a minority of steps where asym3 disagrees
violently. Reporting the mean alone would overstate the 35B's problem; reporting
the median alone would hide it entirely.

For scale, the closest in-house reference point is the MQ2-vs-FP4 comparison on
wikitext (mean 0.834 / median 0.429 nats). asym3-vs-q8 is less severe than that
but the same order of magnitude — and MQ2 is an aggressive quantisation nobody
proposes as a default.

## KLD does not accumulate within the generation

Per-quarter mean KLD across the 64 generated steps:

| | Q1 | Q2 | Q3 | Q4 | Q4/Q1 |
|---|---|---|---|---|---|
| 27B | 0.584 | 0.297 | 0.198 | 0.109 | 0.19× |
| 35B-A3B | 0.389 | 0.916 | 0.735 | 0.167 | 0.43× |

Divergence is **highest early and decays**, consistent with early tokens being
higher-entropy (more plausible continuations) and later tokens inside a code
block being heavily constrained. The plan anticipated the opposite — that
rotated-K error would accumulate with position — and that is **not** observed
here.

**This does not clear the long-context concern.** 64 steps at 4K context does
not test what happens at the 32K-128K contexts agents actually run; that needs a
long *prompt*, not a long generation. The accumulation hypothesis is untested,
not refuted.

## Verdict

| model | verdict |
|---|---|
| `qwen3.6:27b` | **asym3 stays opt-in.** Not the batched default. |
| `qwen3.6:35b-a3b` | **asym3 stays opt-in.** Not the batched default. |

The bar for *promoting* a non-default quantisation is high, and ~30% top-1
disagreement does not clear it. The bar for *keeping the shipped default* is
correspondingly low, which is why this verdict is safe on comparatively little
evidence — an inversion of the usual burden, and deliberate.

## Consequence: the capacity argument fails

Spec §4.2 leaned on asym3 to make 4 agents × 128K context fit on the 27B:
17.2 GB of KV at q8 → 11.8 GB at asym3, taking the total from 32.2 GB (does not
fit) to 26.8 GB (fits). **With asym3 not defaulted, that argument no longer
holds.** The fallback is fewer agents or shorter contexts on the 27B — not
another compression trick. The 35B-A3B is unaffected: its 10 KB/token KV already
fits 4 agents × 128K in ~25.5 GB at q8.

An operator who accepts the quality cost can still opt in per-run with
`HIPFIRE_KV_MODE=asym3`, and the SP1 kernels support it fully — Task 7 verified
the asym3 multi-slot path across 248 checks.

## Limitations, stated plainly

- **One prompt, 64 generated steps, 4K context, greedy decoding.** This is a
  decisive result for "should it be the default" and is *not* a broad quality
  evaluation of asym3.
- **The coherence gates (`coherence-gate-qwen35-*`) and
  `scripts/kv_quality_dashboard.py` were not run.** Three agent stalls consumed
  the budget for them. They would strengthen the picture but are unlikely to
  reverse a 30% top-1 disagreement.
- Long-context behaviour is untested (see above).
- gfx1151 dev hardware; the target is gfx1201. Quality is a numerical property
  and should transfer, unlike the performance numbers elsewhere in this series.

## Memory

Both runs under `scripts/run-bounded.sh` at a 28 GiB cap. `MemAvailable` 57.7
GiB before, 57.7-57.8 GiB after each. **Zero** kernel OOM / allocation-failure
events across both (`journalctl -k` checked per arm). No model left resident.
