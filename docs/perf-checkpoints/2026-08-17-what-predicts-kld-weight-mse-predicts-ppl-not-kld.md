# What predicts KLD: weight-space MSE predicts PPL, and KLD lives in a different space

- **Date:** 2026-08-17
- **Lifecycle:** `historical`
- **Disposition:** literature + theory synthesis explaining the qt=40 metric failure. Contains
  no new measurement; the discriminating experiment it specifies is pending.

## The failure being explained

| arm | codec MSE (rotated) | **PPL** | **KLD** |
|---|---|---|---|
| mq4 uniform affine (qt=1, 136 B) | 1.4415e-06 | 6.4088 | **0.043776** |
| mq4gl tensor-global Lloyd (qt=40, 130 B) | **1.1441e-06** | **6.3276** | 0.048713 |

The codebook format wins codec MSE by 20.6% and loses KLD by 11.3% (27.5% on a
conversation-distribution reference). Four format experiments were run against codec
MSE before this was noticed.

## The resolution: our data does not contradict the literature, it confirms it

The HIGGS **linearity theorem** ([arXiv:2411.17525](https://arxiv.org/abs/2411.17525),
Malinovskii, Panferov, Ilin, Guo, Richtárik, Alistarh) states:

$$\mathbb{E}[PPL(\widehat W)] \approx PPL(W^\star) + \sum_{l=1}^{L} \alpha_l t_l^2,
\qquad t_l^2 = \frac{\mathbb{E}\|\widehat W_l - W_l^\star\|_F^2}{\|W_l^\star\|_F^2}$$

with $\alpha_l$ layer-specific and **independent of the quantizer**. HIGGS itself is
Hadamard rotation + Gaussian-MSE-optimal grids, data-free — i.e. **the same
construction as our GL format**, including the scale: HIGGS Algorithm 1 uses
$s_i = \|w_{\{i\}}\|_2$, then $s = [s_1,\dots]/\sqrt g$, which is the RMS scale GL uses.

**So the theorem predicts GL should win PPL, and GL did win PPL.** The theorem worked.
It says nothing about KLD.

The paper is explicit about the boundary, §2: *"the linearity theorem has no direct
bearing on the data-aware layer-wise MSE minimization problems considered in
references such as GPTQ and QuIP, which are of the form
$\min\|W^\star X - \widehat W X\|_F^2$."*

There are two objectives in this literature and they predict different quantities:

| objective | data needed | predicts |
|---|---|---|
| $\sum_l \alpha_l\|\Delta W_l\|_F^2/\|W_l\|_F^2$ | none (data-free) | **PPL** |
| $\|\Delta W X\|_F^2$ | activations | **functional fidelity** — what KLD measures |

**Our error in one line: we used a weight-space metric to predict a function-space
quantity.**

## Four ways our metric violated even the PPL theorem

1. **$t_l^2$ is RELATIVE**, divided by $\|W_l\|_F^2$. Ours was absolute.
2. **$\alpha_l$ is per-layer.** Ours summed uniformly across tensors.
3. **Assumption 1 requires $W^\star$ to be a local minimiser of PPL**, waived only for an
   *unbiased* quantizer ($\mathbb{E}[\widehat W]=W^\star$). Round-to-nearest is biased, and
   we have direct evidence the assumption fails on our eval slice: mq4gl scored
   selector PPL 9.4627 against a 12.8813 teacher — **26.5% BELOW the teacher.** A
   quantized model cannot be more faithful than its teacher, so $W^\star$ is not a local
   PPL minimum there, and the expansion retains a signed first-order term. That is
   precisely how PPL can *improve* under quantization while KLD grows.
4. HIGGS uses **vector** quantization (grid dimension $p$); scalar ($p{=}1$) is its
   weakest configuration. Our formats are all scalar. This retroactively bears on the
   mq3.5gl 2D-VQ experiment, which was rejected on **MSE** grounds — a criterion now
   known not to decide the question.

## Why the two metrics can invert: the 256× cross term

Layer output error is exactly

$$L = \mathrm{Tr}(E A_{\text{rot}} E^\top)
 = \underbrace{\sum_j A_{\text{rot},jj}\|E_{:,j}\|^2}_{\text{diagonal}}
 + \underbrace{\sum_{j\ne k} A_{\text{rot},jk} C_{jk}}_{\text{cross}}$$

with $E = W_{\text{rot}} - \widehat W_{\text{rot}}$, $A_{\text{rot}} = RAR^\top$,
$A = \mathbb{E}[xx^\top]$, and $C_{jk} = \langle E_{:,j}, E_{:,k}\rangle$.

Unweighted Frobenius error is only $\sum_j C_{jj}$. It equals $L$ **iff**
$A_{\text{rot}} \propto I$ *and* the error columns are uncorrelated. Both fail:

- **Hadamard equalisation holds only for diagonal $A$.** For $R = D_2 H D_1$ with
  $R_{ij}^2 = 1/n$, $\mathrm{diag}(RAR^\top)_{ii} = \overline a$ **exactly** when
  $A = \mathrm{diag}(a)$. With off-diagonal mass $O$, $A_{\text{rot},ii} = \overline a + d_i$
  where $d_i = (ROR^\top)_{ii}$; for equicorrelated $A$ with correlation $\rho$,
  $\mathrm{std}(d_i) \approx \rho\,\overline a$, i.e. **5–20% diagonal spread** at
  $\rho = 0.05$–$0.2$. Real activations are correlated, so importance is *not* uniform
  after rotation.
- **Errors are correlated within a block, and the coupling is amplified by block size:**
  $L_{\text{cross}}/L_{\text{diag}} \sim (n_b-1)\rho_e\rho_a$. At $n_b = 256$ with
  $\rho_e = \rho_a = 0.05$ that is **0.64 — the same order as the diagonal term.** A
  $\Delta\rho_e$ of 0.02 between two formats swings $L$ by ~12% at *identical* Frobenius
  error.

**The two formats differ exactly in that error correlation.** Uniform affine fits
per-block min *and* max — two parameters from two extremes — which decorrelates the
residual. The codebook shares a single per-block scale, set by one outlier, with a fixed
level shape, inducing common-mode error across all 256 coefficients.

## Corroborating findings from the wider literature

- **"Accuracy is Not All You Need"** ([arXiv:2407.09141](https://arxiv.org/html/2407.09141v1)):
  KLD correlates with behavioural "flip rate" at Spearman ~0.96–0.97, while accuracy can
  move ≤1–2% as flips reach 5–13%. PPL is robust to roughly symmetric log-prob noise
  that nonetheless changes model behaviour. **This validates ranking on KLD** and
  explains PPL's three inversions in this campaign.
- The HIGGS linearity result is explicitly bounded to **~3–8 bits**; below that it
  breaks, which matters for our mq1/mq2/mq3 arms.

## The discriminating experiment (specified, pending)

Compute three nested metrics per tensor, both arms:

| metric | formula | data needed |
|---|---|---|
| $m_0$ | $\|E\|_F^2$ | weights only |
| $m_1$ | $\sum_j A_{\text{rot},jj}\|E_{:,j}\|^2$ | $\mathrm{diag}(A)$ — the imatrix |
| $m_2$ | $\mathrm{Tr}(E A_{\text{rot}} E^\top)$ | full $A$ — the collected Hessians |
| $c$ | $m_2 - m_1$ | the dropped cross term |

**Prediction if the cross-term explanation is right:** $\Delta M_0 = -20.6\%$ (GL wins),
$\Delta M_1 = -10$ to $-15\%$ (GL still wins), $\Delta M_2 = +8$ to $+15\%$ — **flips, GL
worse** — matching the observed $\Delta\text{KLD} = +11.3\%$.

The full $A$ already exists: the native calibration work stored per-tensor **full**
Hessians in HFQM `Bf16TrilDiagF32` form, ~63 MB each, ~31.5 GB over 496 tensors.
**Caveat:** they were collected on a gfx942 host separately shown to produce degenerate
artifacts (a GPTQ run from them scored KLD 8.37 vs 0.044). For a *relative* comparison
the contamination may cancel, but no $m_2$ number from them is clean in absolute terms.

## Consequences for this campaign

- **Codec MSE is retired as a ranking criterion.** It is a valid PPL proxy and a
  non-predictor of KLD. Every prior MSE-based verdict is scoped to PPL only.
- **Affected earlier decisions**, all decided on MSE and therefore reopened: mq3.5gl
  rejected as "NO-GO" (+61.28% MSE); "polynomial codebook is Pareto-dominated"; the
  GL_CB4 least-squares scale fit (+2.73% MSE); and the 132/136/144 B ladder in the
  sub-block study.
- **The 2×128 sub-block result is now the most interesting open item**, because finer
  scale granularity directly attacks within-block error correlation — the mechanism this
  synthesis implicates — rather than the MSE it was selected on.

---

## Amendment — the cross-term prediction was measured and FALSIFIED

The discriminating experiment ran (`crates/hipfire-quantize/examples/mq_kld_proxy.rs`,
3 tensors: layer-0 `linear_attn.out_proj`, layer-20 `mlp.down_proj`, layer-40
`mlp.gate_proj`; 12,288 blocks / 3,145,728 weights; engine FWHT seeds 42/1042; imatrix
`Qwen3.8-27B-imatrix.gguf`; **full** Hessians from
`/home/kaden/qcal/qwen3.8-27b.calib.hfq`, 496 tensors, gfx942-sourced).

| metric | affine | GL | Δ (gl−aff) | predicted | outcome |
|---|---|---|---|---|---|
| $M_0=\|E\|_F^2$ | 5.8381 | 4.6399 | **−20.52%** | −20.6% | GL wins ✓ |
| $M_1$ diag-weighted | 2.0337 | 1.6111 | **−20.78%** | −10..−15% | GL wins, by *more* |
| $M_2=\mathrm{Tr}(EA_{\text{rot}}E^\top)$ | 2.0036 | 1.6036 | **−19.96%** | **+8..+15% flip** | **no flip** |
| $c/m_1$ cross term | — | — | **+1.24%** | ~64% | **negligible** |
| $\rho$ | −0.015 | −0.005 | — | $\rho_{gl}\gg\rho_{aff}$ | tiny either way |

**The cross term is 1.24% of the diagonal, not the ~64% that $(n_b-1)\rho_e\rho_a$ predicted
at $n_b=256$.** Within-block error correlation is therefore **not** the mechanism. None of
the four decision branches fired.

More importantly: **activation weighting does not explain the inversion at all.** $m_1$
(imatrix diagonal), $m_2$ (full activation covariance), activation-weighted MSE applied
*before* rotation, and the same applied *after* — all four say GL wins by ~20%. The
gfx942 provenance of the Hessians could distort $m_2$'s magnitude but cannot manufacture
a consistent 20% sign agreement across four independent weightings.

### What does reproduce the inversion

| metric | affine | GL | gets affine < GL? |
|---|---|---|---|
| tail-99 MSE | 2.539e-06 | 2.015e-05 | **yes, 7.9×** |
| tail-99.9 MSE | 5.006e-06 | 6.405e-05 | **yes, 12.8×** |
| max-coefficient rel. error | ~0 | 0.1076 | **yes** (structural for affine) |
| everything else tested | — | — | no |

The agent discounted the tail metrics because they invert mq3 vs mq3lloyd. **That pair is
not byte-matched** — 12,618,796,032 B at 3.25 bpw versus 13,379,750,912 B at 3.5 bpw, i.e.
mq3lloyd carries 6.0% more bytes and duly wins KLD by 29.7%. A metric is not refuted by
failing to predict that more bytes lose. **The mq4 affine-vs-GL pair remains the only
byte-comparable comparison in the corpus, and only tail-restricted error gets it right.**

## The predictor the literature names, which we have NOT tested

**SqueezeLLM** ([arXiv:2306.07629](https://arxiv.org/abs/2306.07629)) is the closest
published analogue — a codebook fitted to weights, exactly our construction — and its
ablation is decisive: unweighted k-means gives PPL **28.26**; Fisher-weighted gives
**7.75** at 3-bit LLaMA-7B. Objective:

$$Q^*=\arg\min_Q (W-W_Q)^\top \mathcal F (W-W_Q)
\;\xrightarrow{\ \mathrm{diag}\ }\;
\arg\min\sum_i \mathcal F_{ii}\,(w_i-Q(w_i))^2,
\qquad \mathcal F=\tfrac{1}{|D|}\sum_d g_d g_d^\top$$

**$\mathcal F$ is the Fisher — gradient outer products — not $H=\mathbb{E}[xx^\top]$.**
This checkpoint previously conflated them. GPTQ/QuIP weight by *activation* second
moments (layer-local); SqueezeLLM weights by *loss* sensitivity (global). We measured the
former and it failed; the latter is untested here.

The structural reason they differ: for a linear layer
$\mathcal F_{ij}\approx\mathbb{E}[g_i^2]\cdot\mathbb{E}[x_j^2]$ — a **per-output-row**
factor times a per-input-column factor. Every metric tested above weights *columns* only.
**We have never weighted rows**, and the row factor is where loss sensitivity lives.
Obtaining it needs one backward pass over calibration data; the imatrix cannot supply it.

## Ranked predictors, with our status against each

| rank | predictor | data needed | literature status | our status |
|---|---|---|---|---|
| 1 | **teacher–student KLD** | teacher + student logits | demonstrated (EvoPress fitness, KL-Lens) | **in use as the gate** |
| 2 | layer-output recon $\sum_\ell\|W_\ell X_\ell-\widehat W_\ell X_\ell\|_F^2$ | calib activations | demonstrated as *solver* objective | untested end-to-end |
| 3 | Hessian-weighted $\mathrm{tr}(\Delta W H \Delta W^\top)$ | calib Gram | demonstrated as solver objective | **TESTED — FAILS our inversion** |
| 4 | **Fisher-diagonal weighted codec MSE** | gradients | **demonstrated for codebooks** (SqueezeLLM) | **UNTESTED — the open lead** |
| 5 | activation-RMS weighted (AWQ / imatrix) | $\mathbb{E}[x_j^2]$ | demonstrated for *scale* choice; weak post-FWHT | **TESTED — FAILS** |
| 6 | unweighted rotated codec MSE | weights only | no paper validates it as a within-bpw KLD ranker | **TESTED — FAILS**, predicts PPL |

## Structural findings that outrank any metric choice

- **EvoPress** ([arXiv:2410.14649](https://arxiv.org/abs/2410.14649)): *error monotonicity
  does not hold* — a lower sum of per-layer errors can be **worse** end-to-end, which is
  why they use KL as the search fitness rather than layer error. This says no layer-local
  metric, however weighted, is guaranteed to rank formats.
- **QAM-W**: weight-Frobenius→layer-RMSE amplification varies **0.55–5.5×** across layers,
  so equal weight error can mean 10× different layer harm.
- **KL-Lens**: KL *direction* matters — student→teacher tracks PPL sensitivity while the
  reverse direction is anti-correlated (τ≈−0.14). Worth confirming which direction
  `eval_hipfire` computes before citing any KLD number as comparable to published work.
- **Post-rotation diagonal weighting collapses to unweighted MSE.** Confirmed by
  measurement: our after-rotation imatrix weighting reproduced the plain `mse_rot`
  ordering exactly. If the imatrix is to be used at all, it must be applied in the
  **pre-rotation** basis — and even that failed here.

## Standing conclusion

For format design at fixed bpw, **there is no validated cheap proxy.** The gold metric is
teacher KLD, the one metric that reproduces our only byte-comparable inversion is
tail-restricted error, and the one published predictor with direct evidence for codebook
design — Fisher-diagonal weighting — requires gradients we have not collected. Codec MSE
is confirmed as a PPL proxy and confirmed as a non-predictor of KLD.
