# Stage A results — query-tiled Q8 flash prefill attention

Date: 2026-07-29
Branch: `perf/flash-prefill-attention` (from `e2169c2d`, PR #534 head)
Spec: [`2026-07-29-attention-prefill-flash-kernel-design.md`](2026-07-29-attention-prefill-flash-kernel-design.md)
Hardware: gfx1151 (Strix Halo), Qwen3.6-35B-A3B MQ4R, Q8 KV, MTP off

## Verdict

**Stage A gate: PASS on correctness and performance. The Redline failure I
originally recorded here is RETRACTED — it did not replicate.**

The kernel is correct and materially faster where it is used. It ships opt-in
(`HIPFIRE_FLASH_PREFILL=1`), default off, pending a decision on whether to make
it default above `MIN_CTX`.

### Retraction (added after replication)

An earlier version of this document reported "Stage A gate FAILS on Redline:
flash ON degenerates 5/6 real-prose requests" and concluded that "Redline does
not fail closed when the prefill attention kernel changes under it." **That
conclusion was based on unreplicated single runs and is not supported.**

Repeating the identical configuration gave 5/6, 5/6, 2/6, 0/6, 0/6, 0/6
degenerate — the failure is intermittent, so a single run could not attribute
it. Two intermediate hypotheses also died on contact with data: PM4 register
elision (`HIPFIRE_REPLAY_PM4_STATEFUL=legacy` was clean, but so was the default
`static` in the same session) and a "request 1 captures, 2-6 replay corrupt"
pattern (did not survive repeats).

With a matched control — identical prompts, flash OFF, Redline, same session
— and a consistent degeneracy criterion (`finish=length` with < 100 answer
words; a long answer merely truncated at the token cap is not degeneration):

| arm | degenerate |
|---|---|
| flash ON, Redline | 6/30 (20.0%) |
| flash OFF, Redline (control) | 1/18 (5.6%) |

**Fisher exact two-tailed p = 0.231 — not significant.** The control degenerates
too, so this is a pre-existing phenomenon, not something the kernel introduces.

Pooling both arms, degeneration concentrates on particular prompts rather than
on the kernel:

| prompt ctx | degenerate (both arms) |
|---:|---|
| 13939 | 0/8 |
| 14072 | 2/8 |
| 14126 | 0/8 |
| 14157 | 3/8 |
| 14216 | 1/8 |
| 14571 | 1/8 |

A confound in the original test inputs explains part of the apparent signal:
`above_prompts.json` slices the docs corpus at `r*9000` across six reps,
reaching table- and code-heavy regions, while the "clean baseline"
`real_prompts.json` used two slices from the prose-heavy start. Those sets are
not equivalent, so "above-threshold degenerates, below-threshold does not" was
partly a content difference, not a context-length or kernel effect. The
degenerate output is also the same multilingual-gibberish signature as the
random-word-filler artifact documented earlier — that is what this model's
failure mode looks like, whatever triggers it.

**What is honestly known:** at n=48 this test is underpowered. It does not
demonstrate an effect, and it does not exclude one either — the point estimate
is 3.6× and the confidence interval is wide. Determining this properly needs a
paired design with many more reps, prompts screened for baseline degeneracy,
and a degeneracy-robust metric. Until someone does that, there is no evidence
of a Redline/flash interaction to fix.

## Correctness

`test_q8_flash_prefill` compares against `attention_q8_0_kv_batched` under
`|ref − new| ≤ ATOL + RTOL·|ref|` (ATOL 1e-5, RTOL 1e-4) plus cosine ≥ 1 − 1e-6.

All shapes PASS, worst element consuming 6.3% of the tolerance budget:

- CTX 32 / 512 / 1024 / 4096 / 8192 / 12288 at N 16..256
- boundaries 33/16, 31/8, 65/17, 97/33, 129/16 (`seq_len % BC != 0`,
  `seq_len < BC`, `batch_size % BR != 0`)
- ragged non-monotonic `positions[]`
- GQA kv_group 1 / 2 / 4 / 8

## Tile-size sweep (CTX=8192, N=256)

| BR | BC | LDS (B) | ms |
|---:|---:|---:|---:|
| 8 | 16 | 17,504 | **23.65** |
| 16 | 16 | 26,304 | 27.32 |
| 8 | 8 | 12,896 | 28.11 |
| 8 | 32 | 26,720 | 30.01 |
| 4 | 8 | 8,624 | 34.70 |
| 32 | 8 | 38,528 | 46.67 |
| 32 | 64 | 76,160 | N/A (>64 KB) |

Not monotonic in LDS — BR=4/BC=8 has half the winner's footprint and is 47%
slower. The optimum balances reuse (BR), tile efficiency (BC) and occupancy
(3 WG/CU). **Chosen: BR=8, BC=16.**

## Kernel-level context sweep (BR=8, BC=16, N=256)

| CTX | tiled fallback | flash | legacy LDS | flash vs tiled | flash vs legacy |
|---:|---:|---:|---:|---:|---:|
| 2048 | 9.44 | 5.59 | 3.76 | 1.69× | 0.67× |
| 4096 | 21.20 | 11.75 | 8.72 | 1.80× | 0.74× |
| 8192 | 43.42 | 23.71 | 22.73 | 1.83× | 0.96× |
| 12288 | 64.06 | 35.98 | 42.78 | 1.78× | 1.19× |
| 16000 | 84.72 | 47.53 | N/A (>64 KB LDS) | 1.78× | — |

Flash beats the tiled fallback by ~1.8× at **every** context. It loses to the
legacy LDS kernel below the break-even, which is measured between CTX 10240
(0.95×) and 11264 (1.17×). The crossing happens because legacy's LDS grows with
context and sheds occupancy while flash's is constant — the mechanism the spec
predicted, now measured.

Consequence: the spec's hoped-for "one kernel for all lengths" does not hold on
this hardware. The right policy is hybrid — legacy below ~10.2K, flash above,
and the tiled fallback retired from this path entirely.

## End-to-end prefill (HIP backend, real prose, `MIN_CTX=10240`)

| ctx | baseline ms | flash ms | speedup | baseline tok/s | flash tok/s |
|---:|---:|---:|---:|---:|---:|
| 7714 | 13498.0 | 12966.6 | 1.04× | 571.5 | 594.9 |
| 7839 | 13133.1 | 12702.4 | 1.03× | 596.9 | 617.1 |
| 11494 | 29609.0 | 27121.8 | 1.09× | 388.2 | 423.8 |
| 11687 | 31147.6 | 27888.5 | 1.12× | 375.2 | 419.1 |
| 13939 | 44749.9 | 36517.9 | **1.23×** | 311.5 | 381.7 |
| 14737 | 49864.2 | 40476.9 | **1.23×** | 295.5 | 364.1 |

Below `MIN_CTX` the numbers are ~1.0× as designed (flash is not used there).
Output quality: 6/6 `finish=stop`, coherent 146–217 word answers.

## The Redline question (unresolved, no evidence of a defect)

See the Retraction above for the full replication data. Summary of what the
first, misleading run looked like and why it misled:

| arm (single runs) | result |
|---|---|
| flash ON, `HIPFIRE_REPLAY_BACKEND=hip` | 6/6 clean |
| flash OFF, Redline | 6/6 clean |
| flash ON, Redline | 1/6 clean |

That table is real but not reproducible: repeats of the last row gave 5/6, 2/6,
0/6, 0/6, 0/6 degenerate. The `real7000r1` anomaly (a below-`MIN_CTX` request,
therefore never touching the flash kernel, yet degenerate) was likewise a
one-off — the below-threshold set is 6/6 clean on repeat. Both were noise in a
process with a ~5-20% baseline degeneracy rate at ~14K context.

The prior investigation's random-word-filler artifact and this one share a root
lesson: this model degenerates intermittently on long, structurally-repetitive
context regardless of backend, and any single-run A/B against that background
will manufacture a false attribution.

## Methodology note

`coherence_probe --temperature 0.0` is **not** reliably deterministic on this
model: across 5 runs of an unchanged binary, 4 matched the baseline exactly and
1 degenerated into repeated token id 0 (caught by the probe's own
`loop_guard_mirror`). Single token-stream diffs are therefore not sufficient
evidence; repeat them. Two matching runs do not establish determinism.

## Stage C (WMMA) — supersedes the Stage A routing decision

Added after Stage A. `attention_q8_0_flash_prefill_wmma` runs QK^T and P·V on
RDNA3 matrix cores. **It beats the legacy LDS kernel at every context**, so the
Stage A break-even and the `MIN_CTX` gate both become unnecessary.

Kernel-level (nh=8 nkv=2 hd=256, N=256), ms:

| CTX | tiled | scalar flash | WMMA | legacy | WMMA/legacy | WMMA/tiled |
|---:|---:|---:|---:|---:|---:|---:|
| 2048 | 10.08 | 5.84 | 3.05 | 3.68 | 1.21× | 3.30× |
| 4096 | 22.13 | 12.65 | 6.24 | 9.27 | 1.49× | 3.54× |
| 8192 | 45.34 | 25.00 | 12.87 | 24.86 | 1.93× | 3.52× |
| 12288 | 68.51 | 39.76 | 19.55 | 48.23 | 2.47× | 3.50× |
| 16000 | 89.88 | 52.59 | 27.68 | N/A | — | 3.25× |

End-to-end prefill (HIP backend, real prose):

| ctx | baseline | WMMA | speedup |
|---:|---:|---:|---:|
| 7714 | 571.5 tok/s | 651.1 | 1.14× |
| 11494 | 388.2 tok/s | 591.3 | 1.52× |
| 13939 | 311.5 tok/s | 526.2 | 1.69× |
| 14737 | 295.5 tok/s | 525.7 | **1.78×** |

Per-position chain curve (ms per new token at prefix depth p):

| p | baseline | WMMA | speedup |
|---:|---:|---:|---:|
| 551 | 1.10 | 1.21 | 0.91× |
| 6573 | 2.15 | 1.91 | 1.13× |
| 8253 | 4.07 | 2.23 | 1.82× |
| 14780 | 6.53 | 3.06 | 2.13× |

**The 8192 step is gone** (baseline 2.15 → 4.07 across it; WMMA 1.91 → 2.23) and
the curve is far flatter: baseline degrades 5.9× from p=551 to p=14780, WMMA
2.5×. Prefill throughput now falls only 1.24× from 7.7K to 14.7K, against 1.93×
before.

Output quality: 6/6 `finish=stop` with coherent 166–227 word answers on **both**
`redline` and `hip`, answer lengths identical across backends. Flag-off remains
inert — committed token stream identical to the pristine baseline in 3/3 runs.

### Quality: perplexity and KLD vs the f32 path

Measured with `crates/hipfire-runtime/examples/flash_prefill_quality.rs`, which
scores **every** position (stride 8) in the second-half window of each chunk
through `forward_prefill_batch` — llama-perplexity chunking semantics, DeltaNet
reset per chunk, KV overwritten from 0 — capturing `per_token_hidden_out` and
recovering logits per row via the lm_head gemv. Both arms score the **identical**
token stream, so NLL is compared **pairwise**.

Three existing tools could not answer this and are documented in the harness:
`perplexity.rs` scores at `batch_size=1` via `forward_scratch` and never reaches
the batched attention kernels; `dump_logits_qwen35` uses a fake prompt (token ids
0,1,2,…) whose output entropy is 9–10 nats on a 248K vocab, so its top-1
comparisons are ties; and `build_kld_ref_native` sets `HIPFIRE_KV_MODE=f32`,
which the config validator's `memory.kv_cache` enum rejects — it panics before
doing any work (pre-existing bug, unrelated to this branch).

**Perplexity, 2048 paired positions, ctx=4096:**

| metric | value |
|---|---|
| ppl f32 | 7.4812 |
| ppl f16-WMMA | 7.5310 (+0.67%) |
| paired ΔNLL | +0.0066 nats (SE 0.0087) |
| 95% CI | [−0.0105, +0.0237] |
| significant at 0.05 | **NO** (t = 0.762) |
| resolvable effect | ±0.017 nats (1.7% ppl) |

No detectable regression; the penalty is **bounded below +2.4% ppl at 95%
confidence**. Note the sign is now physically sensible (f16 worse than f32) — an
earlier 24-position version of this harness reported f16 *better*, which is
arithmetically impossible and was the signal that it was underpowered.

**KLD, 24 windows of real text (full 248K distributions):** mean 6.97e-3,
median 4.36e-3, max 3.70e-2. For scale, this repo accepts a graded-quant change
at KLD 0.038964 — the mean is ~5.6× below it, though the worst window
approaches it.

**Behaviour:** top-1 agreement **95.17%** (99 of 2048 positions differ). Those
disagreements are concentrated where the model is uncertain:

| | top-1 agrees | top-1 differs |
|---|---:|---:|
| mean NLL | 1.92 nats | 3.84 |
| median NLL | 0.30 | 2.81 |
| paired ΔNLL | +0.0033 | +0.0718 |

At confident positions (median NLL 0.30 nats, p≈0.74) the arms agree, and the
f32 top-1 remains inside the f16 top-8 in **99.76%** of cases — a reshuffle
among near-ties, not a lost token. Under greedy decoding this still means
generated text will drift; under sampling it is largely masked.

### Where the f16 error actually lives, and the split-Q mode

Attributed by isolation (round one quantity on the CANDIDATE only, run through
an otherwise-f32 kernel, so that quantity's precision is the only difference):

| term | contribution | note |
|---|---|---|
| **Q** (A fragment) | **5.36e-4 of 5.48e-4 — 98%** | dominant |
| K + V | ~3.2e-4 | `KVEXACT=1` control |
| P (probabilities) | ~2.6e-4 | `SPLIT_P` control |

Three earlier hypotheses were wrong before this was measured properly: the score
round-trip through f16 (fixing it cost 58% perf for zero accuracy), K/V, and P.
Each moved the total only 0–19% because each is a minor term.

`HIPFIRE_FLASH_PREFILL_SPLITQ=1` carries Q as a double-single (hi+lo) pair
through two WMMA passes per d-chunk:

| CTX | base rel_l2 | split-Q | accuracy | time |
|---:|---:|---:|---:|---:|
| 512 | 7.04e-4 | 3.76e-4 | 1.87× | 1.59× |
| 4096 | 5.79e-4 | 1.16e-4 | 4.97× | 1.66× |
| 12288 | 5.28e-4 | 1.16e-4 | 4.54× | 1.59× |

End-to-end quality, 2048 paired positions:

| metric | base | split-Q |
|---|---:|---:|
| ppl vs f32 7.4812 | 7.5310 (+0.67%) | **7.4910 (+0.13%)** |
| paired ΔNLL | +0.0066 nats | **+0.0013** |
| std error | 0.0087 | **0.0049** |
| top-1 agreement | 95.17% | 95.46% |
| top-8 overlap | 7.30/8 | 7.30/8 |

**The distributional gap narrows 5×; behaviour does not change.** Top-1 moves
+0.29pp and top-8 overlap not at all, because top-1 disagreements sit at
genuinely uncertain positions (median NLL 2.81 nats vs 0.30 where the arms
agree) whose leading candidates are separated by less than split-Q's *residual*
error. More precision cannot resolve a near-tie.

**Default off.** 1.6× is not worth paying for a distributional improvement on a
gap that is already statistically undetectable, when it does not change which
tokens are chosen. It is kept as a real high-accuracy operating point: even at
1.6× it beats every alternative (15.10 ms at CTX=8192 vs ~22 ms legacy, ~45 ms
tiled). Turn it on if you have a reason to want maximum fidelity to the f32
path; otherwise leave it alone.

### The trade

WMMA computes in f16. Measured relative L2 against the f32 reference is
5.3e-4…1.6e-3 across 17 configs, versus 3.2e-7 for the scalar kernel — roughly
3500× less accurate, though cosine similarity stays ≥ 0.999999. It is therefore
held to a reduced-precision bar (relative L2 ≤ 5e-3, cosine ≥ 1 − 1e-5) rather
than the fp32-reassociation bar, and it ships opt-in. Note this bar was chosen
*after* seeing the fp32 bar fail, which deserves scrutiny; the justification is
that per-element relative error is not a usable metric for a reduced-precision
kernel — cancellation on near-zero outputs inflates it to 20% while the output
vectors remain aligned to 6 decimal places. Whether f16 attention is acceptable
as a *default* is a product decision that wants perplexity/KLD evidence, not
just coherence.

Also worth recording: Stage C was not a port. Every `attention_dflash_wmma_*`
kernel hard-guards `head_dim == 128`, and the only head_dim-generic variant
needs ~84 KB of LDS at head_dim=256. The new kernel fits in ~18 KB (3 WG/CU) by
storing LDS as f16 and never staging K — each B fragment is dequantised straight
from the Q8_0 cache, and the 16×16×16 fragment shape reuses it across 16 query
rows, amortising the dequant by the layout itself.

## The intermittent long-context degeneracy: four dead hypotheses

Throughout this work, long-context real-prose runs occasionally produce
degenerate output (`finish=length` with a handful of answer words; content is
multilingual gibberish). It is NOT caused by anything on this branch, and it is
not explained. Recording the negative results so nobody repeats them.

Measured rates, degeneracy = `finish=length` AND `ans_words < 100` (a long
answer merely truncated at the token cap does not count):

| arm | rate |
|---|---|
| WMMA (flip replication + fragile-prompt probe) | **0/36** |
| flash OFF, ctx<8192 → legacy LDS kernel | **0/18** |
| flash OFF, ctx>8192 → tiled fallback | **0/18** |
| flash OFF, earlier batch, ctx>8192 | **8/24 (33%)** |
| flash ON (scalar), earlier batch | 6/30 (20%) |
| flash OFF, earliest batch | 1/18 (5.6%) |

**Ruled out:**

1. **Redline retained routes.** Claimed, then retracted: repeats gave 5/6, 5/6,
   2/6, 0/6, 0/6, 0/6, and a matched flash-OFF control degenerated too
   (6/30 vs 1/18, Fisher p=0.231, not significant).
2. **PM4 register elision.** `HIPFIRE_REPLAY_PM4_STATEFUL=legacy` was clean —
   but so was the default `static` in the same session.
3. **The flash/WMMA kernel.** 0/36, and the default-on flip measured 0/24
   against 8/24 for the control (p=0.004) — i.e. if anything the new kernel
   looks *better*, which is itself a reason not to trust the comparison.
4. **The tiled fallback.** 0/18 at ctx>8192 with flash off, in the same band
   and configuration that previously gave 8/24.

**What survives:** the rate is unstable across batches under a *fixed* nominal
configuration (0% → 33%), and the only variable that differs between the 8/24
batch and the 0/18 batch is which corpus slices were used. Per-prompt rates do
vary (one slice degenerated 3/8 while another was 0/8 in the same pooled data) —
but that same 3/8 slice later ran 0/8 with fresh prefill each time. So content
matters and content alone does not determine it.

The most likely remaining explanation is intrinsic: occasional sampling collapse
at long context on documentation-style text (`temperature 0.7`, `top_p 0.8`,
`presence_penalty 1.5`), whose probability depends on the specific text. That is
a model/sampling characteristic, not a kernel defect — but it is not proven, and
it should not be asserted without evidence.

**Why this method cannot settle it, and what would:** end-to-end output
classification has a per-request cost of ~25-45 s at these contexts and a noisy
binary outcome, so distinguishing a 5% from a 20% rate needs hundreds of
requests per arm. The efficient instrument is logit-level: capture the
per-position distribution at the point where a run diverges from a known-good
reference, and compare entropy/top-1 margin at the divergence point rather than
classifying the final text. `flash_prefill_quality.rs` already emits per-position
records and is the natural place to build that.

## Next

1. Stage C (WMMA inner math) is **unblocked** — the tiling foundation is in
   place and there is no demonstrated Redline defect standing in its way.
2. If anyone wants to settle the Redline question properly, it needs a paired
   design: the same prompts across arms, prompts pre-screened for baseline
   degeneracy (drop ctx=14157/14072-style slices or measure them separately),
   many more reps, and a degeneracy metric that does not confuse a long
   truncated answer with a failed one. n=48 was underpowered.
3. Independent of all of the above, the cheap win still stands: raising
   `Q8_BATCHED_LDS_CROSSOVER` from 8192 toward the real ~15.8K LDS bound is
   validated bitwise-identical and worth 1.26–1.72× in the 8.2K–11.5K band,
   because it replaces the tiled fallback with the legacy kernel exactly where
   the legacy kernel is still the faster of the two.
