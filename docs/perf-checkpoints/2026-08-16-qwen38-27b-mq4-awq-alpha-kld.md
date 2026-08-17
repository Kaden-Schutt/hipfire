# Qwen3.8 27B — MQ4G256 AWQ alpha sweep against a BF16-teacher calibration (gfx942)

**Lifecycle:** `historical`. Fixture-bound measured evidence from the MI300X
BF16-teacher calibration campaign. It is **not** a current default, an automatic
baseline, a product claim, or an admission decision.

**Disposition:** At a fixed MQ4G256 wire format (136 B/group, unchanged layout),
lowering the AWQ smoothing exponent from the shipped default `alpha=0.55` to
`alpha=0.05` reduces slice-mean KLD by **24.3 %** on the out-of-distribution
WikiText-2 reference and by **15.0 %** on the in-distribution held-out reference.
The gain appears on *both* references, so it is not calibration overfit. The
KLD curve is flat (within 0.003 absolute, 4.5 % relative) across
`alpha in [0.05, 0.25]`, i.e. the alpha lever is asymptotic in that band; the
remaining headroom at this format is not in alpha.

Separately and more seriously, the **uncalibrated MQ4 baseline is degenerate on
gfx942** — it emits a two-token attractor on every prompt tried. See
§ "Uncalibrated MQ4 collapse" below; that observation is *not* explained by this
sweep and is filed here as evidence, not as a diagnosis.

**Source:** worktree `quant/quality` off `origin/master @ 8510ca5f2`, uncommitted
at measurement time. Calibration stack ported from closed PR #441 (`chaingun`).

---

## Fixture identity

| field | value |
|---|---|
| GPU | AMD MI300X, `gfx942`, 205.8 GB VRAM, 20 vCPU, 235 GB RAM |
| ROCm / HIP | HIP 7.0.51831-7c9236b16 |
| parent checkpoint | `/scratch/parents/qwen3.8-27b` (upstream unquantized, 18/18 shards) |
| BF16 teacher | `qwen3.8-27b.teacher.bf16.hfq`, 53.8 GB, 26,895,998,464 params @ 2 B/param |
| calibration | `qwen3.8-27b.calib.hfq`, 31,481,139,200 B, md5 `c94a6560e4b82ac1e7a2236561c2216d` |
| calib corpus | `bartowski_v5.calib.txt`, md5 `fcf30d293b369f71f0eded4e8f62cf74`, 1,495,749 B |
| WT2 slice | `wikitext2-1024s-2048ctx.txt`, md5 `83b0205a304bf4e52172ecdb05f2e895`, 10,506,724 B |
| AG slice | `bartowski_v5.eval.txt`, md5 `9c786be97f505bb8b8e79422b6cb4338`, 223,240 B |
| `hipfire-quantize` md5 | `5fb0c75dae98a7ea83d874e562e25b8d` |
| `calib_sweep` md5 | `b01a0f6d889fb89b153e8e2c1626dd42` |
| `eval_hipfire` md5 | `0a1806f79d928d53004ae344e6327728` |
| `build_kld_ref_native` md5 | `8b5648a882d69f358d4ca5ff93cccd84` |
| `greedy_dump` md5 | `11b11677b912ce549831ee4b34f4ed41` |

### Calibration method

Strict BF16 teacher: `HIPFIRE_CALIB_BF16=1` keeps qt=16 weights as `DType::BF16`
and dispatches them through `KernelKey::GemmBf16Mfma`
(`v_mfma_f32_16x16x16bf16_1k`, gfx942-only). No F32 widening, no q8 teacher.

```
calib_sweep --model qwen3.8-27b.teacher.bf16.hfq \
            --corpus bartowski_v5.calib.txt \
            --output qwen3.8-27b.calib.hfq \
            --max-tokens 262144 --seq-len 2048 --layers-per-pass 64
env: HIPFIRE_NORMALIZE_PROMPT=0 HIPFIRE_GRAPH=0 HIPFIRE_GRAPH_MOE=0 HIPFIRE_CALIB_BF16=1
```

Coverage verified **from artifact bytes**, not from the harness PASS line:
496 tensors, `n_hessian == n_imatrix == len(per_tensor_tokens) == 496`, every
tensor at exactly 262,144 tokens (`scripts/verify_calib_artifacts.py`).
Projection kinds: `q/k/v/o_proj` 16 each (FullAttention layers),
`in_proj_qkv/z/a/b` + `out_proj` 48 each (DeltaNet layers),
`gate/up/down_proj` 64 each. N/K at `down_proj` (K=17408) is 15.1x.

> The harness's own `max_consistency ... [CONSISTENT]` line is **not** evidence.
> It compares `diag(H)` against `sum(x^2)` where both are accumulated from the
> same staged buffer by two kernels (`calibration.rs:342-347`), so it measures
> kernel agreement, not capture completeness, and reads `0.000e0` even when
> tensors are missing entirely.

### Reference construction

```
build_kld_ref_native --model qwen3.8-27b.teacher.bf16.hfq --slice <slice> \
                     --output <ref>.bin --top-k 256 --n-ctx 2048 --max-chunks 24
```

Both references are 24 chunks / 24,552 scored tokens so the two numbers are
directly comparable. Oracle: WT2 NLL 1.830742 / PPL 6.2385; AG NLL 2.052225 /
PPL 7.7852.

### Arm construction and scoring

```
hipfire-quantize --input /scratch/parents/qwen3.8-27b --format mq4 \
                 --output <arm>.mq4 --hessian qwen3.8-27b.calib.hfq --awq-alpha <a>
eval_hipfire --model <arm>.mq4 --ref <ref>.bin --scoring-mode prefill --kv-mode q8
```

Format is constant across every arm: MQ4G256, 136 B/group. Only `--awq-alpha`
varies. AWQ scale is `s[j] = (RMS_act[j])^alpha`, geo-mean normalized to 1
(`main.rs:6303`), with `RMS_act` from the calibration `in_sum2`.

**Repeat count:** `eval_hipfire` is deterministic under this method — a repeated
run of `alpha=0.05` on WT2 returned bit-identical
`KLD 0.063102 / NLL 1.875801 / PPL 6.5260`. Because scoring is greedy prefill
against a fixed reference with no sampling, one run per arm is the full result,
not a sample of one. This is the reason the usual three-fresh-process rule is
not invoked here; it governs throughput measurements, which this record makes
none of.

---

## Result — AWQ alpha vs KLD, fixed MQ4G256

WT2 = WikiText-2, out-of-distribution relative to the calibration corpus, used
as the **selector**. AG = held-out tail of the calibration corpus,
in-distribution, reported for the candidates as the deployment-representative
number. A recipe that gains on AG while flat on WT2 would be calibration
overfit; the selected recipe gains on both.

| alpha | WT2 KLD | WT2 PPL | AG KLD | AG PPL |
|---|---|---|---|---|
| 0.05 | **0.063102** | 6.5260 | **0.224205** | 7.4737 |
| 0.10 | 0.064720 | **6.4835** | 0.239629 | 7.9255 |
| 0.15 | 0.064898 | 6.5147 | — | — |
| 0.20 | 0.064102 | 6.5749 | — | — |
| 0.25 | 0.065860 | 6.5864 | — | — |
| 0.30 | 0.068296 | 6.5820 | — | — |
| 0.35 | 0.073997 | 6.6521 | — | — |
| 0.45 | 0.073824 | 6.5488 | — | — |
| 0.55 (shipped default) | 0.083329 | 6.4632 | 0.263730 | 8.0148 |
| 0.65 | 0.094264 | 6.5085 | — | — |
| 0.75 | 0.099698 | 6.6000 | — | — |
| 0.90 | 0.136004 | 6.9392 | — | — |
| *oracle (BF16 teacher)* | *0* | *6.2385* | *0*  | *7.7852* |

Arm md5s: `alpha=0.05` `aa2f2e50f0d54da18742a109737392a8`;
`alpha=0.10` `d116d81702fed622193c8a9027489994`;
`alpha=0.55` `f1487c9fc7c3323f30d2f11a6f9448f0` (byte-identical to bare `--awq`,
confirming the default is exactly 0.55). All arms 14,987,185,152 B.

### Reading

- **KLD is monotone in alpha above 0.25** and flat below it. The shipped 0.55 is
  on the rising limb, not at the minimum.
- **KLD and PPL disagree.** PPL is minimized at 0.55 (6.4632) while KLD is
  minimized at 0.05. They are different questions: KLD measures divergence from
  the teacher's full top-256 distribution, PPL only the likelihood of the
  realized token. A recipe tuned on PPL alone would have kept 0.55 and missed a
  24 % KLD reduction.
- **Asymptotic in-band.** Across `[0.05, 0.25]` the KLD spread is 0.0028
  absolute. Further alpha refinement is not worth spending measurement on; the
  next lever has to be something other than alpha.
- `alpha=0.05` scores AG PPL 7.4737, *below* the 7.7852 oracle. A quantized
  model scoring better perplexity than its teacher on a 24.5k-token window is
  not a quality claim — it is within the range where quantization noise can
  sharpen a particular realized sequence. The KLD (0.224) is the honest
  divergence figure and it is strictly positive.

---

## Uncalibrated MQ4 collapse on gfx942 (open, not diagnosed here)

The `--format mq4` baseline with no calibration is **degenerate on this host**:

| arm | WT2 KLD | WT2 PPL | AG KLD | AG PPL |
|---|---|---|---|---|
| baseline (no `--hessian`, no `--awq`) | 12.921389 | 2,632,097.9 | 12.872757 | 2,165,329.3 |

`ln(248320) = 12.42`, so an NLL of 12.87–14.78 is at or past uniform over the
vocabulary. Greedy decode confirms it directly (`greedy_dump`, two unrelated
prompts, temperature-free):

| prompt | arm | tokens | distinct | behaviour |
|---|---|---|---|---|
| `def add(x, y):` | baseline | 2025 | 5 | locked to `7227 2629 7227 2629 …`, never terminates |
| `def add(x, y):` | alpha=0.55 | 135 | 72 | terminates on EOS |
| `The capital of France is` | baseline | 2026 | 5 | locked to a different 2-token pair (`323`, `49`) |
| `The capital of France is` | alpha=0.05 | 38 | 27 | terminates on EOS |

**Why this is flagged rather than concluded:** the baseline artifact is md5
`129909ad0fed21dcf72b5b9225e85604`, byte-identical to the `qwen3.8-27b.mq4`
recorded in [`2026-08-14-qwen38-27b-qwen36-parity.md`](2026-08-14-qwen38-27b-qwen36-parity.md),
where the same bytes generated coherently. That record was measured on gfx1100
and gfx1201 (RDNA); this one is gfx942 (CDNA3). The same file therefore behaves
differently across architectures, which points at an arch-specific execution
path rather than at the quantizer — but nothing here isolates it, and the AWQ
arms exercise the same MQ4G256 kernels without collapsing, which argues against
a simple "MQ4 kernels are broken on CDNA3" reading. Left open.

Consequence for this record: the baseline column is reported for completeness
but is **not** usable as the improvement denominator. The alpha comparison is
made against `alpha=0.55`, the shipped default, which is a working artifact.

---

## What this record does not establish

- No throughput, latency, or tok/s measurement of any kind.
- No claim about other models. Glimmer, gemma4, and the two LFM2.5 models were
  calibrated in the same campaign (416/416, 328/328, 92/92, 92/92 tensors, all
  at 262,144 tokens) but are not scored here.
- No GPTQ/LDLQ result. That arm was started and abandoned at ~10 % after
  measuring ~15 tensors/hour (≈33 h for 497 tensors) while holding 18 of 20
  cores; it is deferred to run once against the winning alpha rather than
  against the default.
- No statement that `alpha=0.05` should become a default. It is the KLD minimum
  under this fixture on this host, on one model.

---

## Addendum 2026-08-17 — bf16-Hessian non-PSD-ness grows LINEARLY in K

Measured on the production GPTQ path with the PSD-projection fallback (commit
`0fba538b5`) against `/scratch/work/qwen3.8-27b.calibv6-814d8fd.hfq`. Three
independent K values, each normalized by its own matrix's `mean(diag(H))` so the
comparison is scale-free:

| K | `-lambda_min` | `mean(diag(H))` | ratio | linear-in-K prediction from K=1024 | error |
|---|---|---|---|---|---|
| 1,024 | 0.2088 | 1.0065 | 0.2075 | — | — |
| 5,120 | 1.063657 | 0.9951136 | 1.0688 | 1.0375 | 2.9 % |
| 17,408 | 0.107699 | 0.02967069 | 3.6299 | 3.528 | 2.9 % |

A linear-in-`K` law fits all three points across a **17x span to within 3 %**. An
earlier `sqrt(K)` estimate underpredicted K=17408 by 2x and is withdrawn.

### Why this settles the design question

`cholesky_with_adaptive_damping` caps damping at
`max_damp_multiplier * mean(diag(H))`, default `1.0`. Since the required shift
scales as `K`, that cap is not merely conservative — it is **asymptotically wrong**:

| K | required damp multiplier |
|---|---|
| 5,120 | ~1.07 (right at the default cap, hence the observed 8-pass / 8-fail coin flip) |
| 17,408 | ~3.63 (always over, hence 0/2) |
| ~28,672 (a 70B-class `down_proj`) | ~6 (extrapolated) |

So raising the cap does not scale: every wider model would need a larger constant,
and the damping-based "fix" degrades GPTQ further at every step, because damping
inflates the small eigenvalues that dominate the inverse Hessian GPTQ actually
consumes (measured: damp `1.1x` moves the smallest eigenvalues from ~-0.15 to
~+0.90, crushing the inverse's dominant components ~90x, against PSD projection's
0.0000 % shift of the top 64).

PSD projection is therefore the correct fix rather than the convenient one, and it
becomes **more** necessary as models widen, not less.

### Production evidence

With the fallback active, at layer 1 of Qwen3.8-27B:

```
rescued=6   nonconv=0   hard_failed=0   fired=10
rescued by K: 1x K=17408, 5x K=5120
```

against the pre-fix run's **10 hard failures** at the same point, each of which
silently became round-to-nearest. Example rescue:

```
gptq: PSD projection rescued K=17408 Hessian
      (lambda_min=-1.076990e-1 before projection); Cholesky succeeded at damp=1.000000e-2
```

`damp=1e-2` is the default — i.e. after projection the tensor needs no unusual
damping at all, versus previously exhausting a cap of `2.967069e-2` and failing.
