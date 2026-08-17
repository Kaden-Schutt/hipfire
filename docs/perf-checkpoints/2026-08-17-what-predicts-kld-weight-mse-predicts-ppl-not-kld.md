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
