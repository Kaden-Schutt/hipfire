# RoughQuant — Phase 2h: foldable outlier protection WINS at low bits

**VERDICT (updated after coherence): KLD win is REAL but does NOT survive
generation — NOT a deployable win.** The Q8 cost-trick breaks generation
(hard-loop/empty); bf16-protection is strictly more precise than mq4 yet rambles
on hard prompts (generation is non-monotonic in weight precision — the analog of
the pointwise-PPL wiggle). KLD (teacher-forced) does not guarantee free-generation
coherence. A proper coherence battery is required to settle bf16-protection; the
Q8 form is out. See "Coherence falsification" below.

_Original (teacher-forced KLD) framing — REAL on the metric, but see coherence:_

**Two framings, both wins (on KLD):**

1. **vs the real production baseline (mq4) — the practical headline:** protecting
   the top 5% outlier channels at Q8 nearly **HALVES mq4's KLD (0.162→0.088) for
   +0.19 avg-bits.** There is no `mq5` in hipfire — the only no-protection way to
   that quality is `mq6` (+2 bits). So this is a cheap, foldable new operating
   point strictly between mq4 and mq6.
2. **vs a hypothetical uniform-5-bit Pareto:** still ~25% lower KLD at ~4.4–4.5b.

Foldable protection of the shared outlier channels at **Q8** (honest 8-bit cost)
+ mq4 bulk beats uniform FWHT bit-increase by **~25% KLD at ~4.4–4.5 avg-bits** on
Qwen3.5-0.8B. The win
shrinks toward higher bits (the literature's pattern: protection matters most
when bits are scarce). This is foldable (shared ~75-channel set, no rotation),
literature-consistent (AWQ/super-weight), and measured cleanly (KLD vs bf16).

## How four compounding artifacts hid this (all from user skepticism)

1. **Zeroing bug** (non-monotonic protected quantizer) — fixed to overwrite.
2. **Energy-aggregation error** — summing per-tensor energy flattened the shared
   outlier structure into a fake-linear CDF ("energy is spread"). Per-tensor
   outliers are strong (max/med up to 283×) and SHARED (~75 dims, phase2g).
3. **bf16-protection wasted half its bits** — bf16 has only ~8-bit mantissa, so
   protecting at bf16 paid 16 bits for 8-bit precision. Q8 protection = same KLD,
   half the cost.
4. **PPL pointwise noise** masked everything sub-~1-PPL — KLD is monotonic.

Strip all four → the real result emerges.

## Measurement (KLD vs bf16/fp32-state ref, same machinery, bf16 embed)

Uniform FWHT N-bit anchors (`ln KLD` linear in bits — exponential fit solid):

| uniform | bits | KLD |
|---|---|---|
| 4-bit | 4.25 | 0.161 |
| 5-bit | 5.25 | 0.0335 |
| 6-bit | 6.25 | 0.0069 |

Foldable Q8-outlier-protection + mq4 bulk:

| protect % | bits | KLD | uniform@bits | win |
|---|---|---|---|---|
| 5%  | 4.44 | 0.088 | ~0.119 | −26% |
| 7%  | 4.51 | 0.081 | ~0.106 | −24% |
| 10% | 4.62 | 0.077 | ~0.090 | −14% |
| 15% | 4.81 | 0.062 | ~0.067 | −7%  |

(bf16-protection — phase2g — was tied/losing because it paid 16 bits; Q8 halves
the cost at ~same KLD: protect-7% goes (5.07b,0.076)→(4.51b,0.081).)

## Why it works now (vs the earlier "uniform dominates")

The earlier comparison used bf16 protection (16-bit cost) → the protected channels
ate the bit budget and uniform won. Q8 protection costs 8 bits for the same
~8-bit-precision the outliers need, freeing budget for the mq4 bulk. So at a fixed
~4.5-bit budget: {mq4 bulk + Q8-protected outliers} < {uniform 4.5-bit} in KLD,
because the shared outlier channels carry disproportionate error that mq4's
generic FWHT only partially tames, and Q8 fixes them cheaply.

## Status: promising, NOT yet shipped — remaining work

- **SIM result** (bf16-baked weights, KLD on one 0.8B corpus). Productionizing
  needs: a real packed format (mq4 bulk + Q8 sidecar for ~75 shared channels),
  the offline fold of the shared protected-channel set, and **coherence-gate**
  validation (a KLD win must not ship an attractor).
- **Selection**: used aggregated-energy ranking; **persistence-based** (channels
  that are outliers in the most layers) should be at least as good and is more
  principled — untested.
- **Cross-model (7B/9B)**: confirm the win generalizes (and whether the crossover
  bit-rate shifts). The mechanism (shared outliers + cheap Q8 + FWHT bulk) should
  hold; magnitude may differ.
- **Magnitude**: ~25% KLD at 4.5b is modest but real; whether it justifies the
  format complexity vs just shipping mq5 is a product call once coherence + perf
  are known.

## Tooling / code

- `HIPFIRE_RQ4_PROTECT_Q8=1` (Q8 protection), `HIPFIRE_RQ4_MQ_BITS=N` (uniform
  N-bit FWHT bulk anchor), on `roughquant4-sim`.
- KLD via `perplexity --dump-ref/--kld-ref` (the default quant-quality metric now).
- Supersedes the negative verdicts in phase2e/2f and the "uniform dominates" of
  phase2g: the honest-bit-cost (Q8) result is a low-bit WIN.

## Coherence falsification (the decisive output-axis check)

Generated with `infer_qwen35` (greedy + production guards) on the protect models
vs mq4:

- **mq4**: coherent, on-topic, terminates (sky → structured Rayleigh answer;
  capital → "Paris"/"Tokyo"; code → reasonable).
- **protect-5%-Q8**: CATASTROPHIC — verbatim attractor loop ("Wait, let me check
  my memory on Blue light vs red/green wavelengths one more time...") ×8, or empty
  output. The Q8 per-channel quant of the *sensitive outlier* channels
  destabilizes decoding. Q8-protection is OUT — so protection's real bit-cost is
  ~bf16 (16b) for those channels, not 8b, which removes the iso-bit win (it only
  ties uniform at bf16 cost, phase2g).
- **protect-5%-bf16**: clean on easy prompts (capital→Tokyo) but degrades to
  rambling/self-contradiction (sky) and broken code (Fibonacci). Yet it is
  STRICTLY more precise than mq4 (= mq4 for 95% of channels + bf16 for the top
  5%), and the protect-0% control via the same path generates cleanly (matches
  mq4). So the rambling is **generation non-monotonicity** — small weight changes
  (even improvements) reroute greedy decoding — not a path bug and not necessarily
  systematic degradation. Inconclusive from anecdotes.

## Corrected bottom line

- Outliers are real, shared (~75 dims), foldable; protecting them genuinely
  improves teacher-forced KLD (halves mq4's). Literature-consistent.
- BUT: (a) cheap Q8-protection breaks generation, (b) bf16-protection (the viable
  form) only ties uniform at iso-bits AND has unresolved generation coherence.
- So NO clean DEPLOYABLE win — but for the now-correct reason: protecting the
  sensitive outliers needs ~bf16 precision (Q8 destabilizes), which costs too many
  bits to beat uniform; and KLD does not predict generation coherence.
- **Meta-lesson (3rd time this session): coherence batteries must gate, not
  PPL/KLD.** Generation quality is non-monotonic in weight precision.

## Decisive next step

Run protect-5%-bf16 through a real coherence battery (many prompts + aggregate
detectors: unique-token-ratio, n-gram density, attractor — `hipfire-detect`),
not anecdotes, to settle whether bf16-protection is systematically degraded or
just trajectory-noisy vs mq4. (coherence_probe is currently broken — missing std
imports under deltanet,experimental-examples; would need a fix or a thin harness
over the detectors.) If bf16-protection passes the battery AND a fair iso-bit
coherence comparison vs mq4, it's a candidate; otherwise RoughQuant stays
non-deployable on 0.8B.
