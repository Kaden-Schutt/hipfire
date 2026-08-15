# GPTQ cross-reference harness

Independent PyTorch reference for hipfire's hand-rolled GPTQ / Hessian path.
Built so Gate 9 cannot pass by comparing two copies of the same bug.

| File | Role |
|---|---|
| `gptq_paper_ref.py` | **Oracle.** Written from Frantar et al. 2210.17323 + DASLab `gptq.py`. |
| `hipfire_shadow.py` | Convention shadow of *our* commits (for staged A/B). Not the oracle. |
| `formats.py` | Independent HFHS + E8H1 parsers/writers from the format contracts. |
| `run_harness.py` | Floor → staged equivalence → fault injection → format round-trip. |
| `CONVENTIONS.md` | Hypothesis list: paper vs hipfire answer per convention. |
| `last_run.json` | Machine-readable metrics from the latest run (gitignored if huge; small here). |

## Recipe (CPU)

GPU is contended on the MI300X box — this harness is small-tensor and must
stay on CPU.

```bash
# From anywhere; paths below assume the ds4-mi300x-agentmaxx worktree.
cd crates/hipfire-quantize/reference_gptq

# venv (once)
python3.12 -m venv .venv
source .venv/bin/activate
# CPU wheel is enough; ROCm wheel also fine if already present.
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
# If the machine already has torch (e.g. 2.11+cu130), skip the install.

# Run
python run_harness.py
# wall time: ~1–3 s on a 12-core Zen2 CPU for the default sizes
# exit 0 = required checks + four injections OK
# exit 1 = equivalence/format failure
# exit 2 = required fault injection blind spot
```

Optional: point at a real Gate-7 Hessian directory later without code changes:

```python
from formats import read_hblk, read_hfhs
from pathlib import Path
h = read_hblk(Path("/path/to/hessians") / "layers.0....weight.hblk")
# h.blocks: [n_blocks, 256, 256] float64
```

## Stages

0. **Floor** — paper ref vs algebraically equivalent paths (Hessian two-batch
   running mean, manual inv-Cholesky, GPTQ blocksize 128 vs 8). States
   `floor_max_abs` / `floor_rel_frob` in f64 CPU arithmetic. `err==0` on
   self-repeat is noted; the blocksize comparison supplies the non-trivial
   floor used by later verdicts.
1. **Hessian** — paper `(2/N)XX^T` vs hipfire HFHS `(1/N)XX^T` vs E8H1 raw
   block-diagonal. Reports max abs, rel Frobenius, norm ratio, cosine.
2. **Damp + Cholesky + U** — paper `U` vs hipfire shadow; invariant
   `U^T U = H^{-1}`; wrong `U U^T` probe; absolute vs fractional damp.
3. **Full GPTQ** — quantized weights + activation reconstruction error vs RTN.
4. **Fault injection** — transpose H (broken symmetry), drop damp, actorder
   without invperm, skip error feedback; plus wrong-U-form extra.
5. **Formats** — HFHS f32/f64 and E8H1 header+payload round-trip.

## Interpreting metrics

| Pattern | Meaning |
|---|---|
| cosine ≈ 1, norm_ratio ≈ c ≠ 1 | Scale convention (often benign for OBS if damp scales). |
| cosine ≈ 0, norm_ratio ≈ 1 | Transpose or permutation bug. |
| max_abs ≤ ~50× floor | PASS against stated floor. |
| max_abs = 0 only | INCONCLUSIVE without a non-trivial floor — see Stage 0. |

## Findings policy

If `gptq_paper_ref` and hipfire disagree after accounting for documented
scale invariance, that is a **finding**. Do not edit the paper ref to match
hipfire. Fixing production quant is a separate decision from this slice.

## SPDX

Apache-2.0. New files carry SPDX headers.

## Dual-parser format proof (Rust readers)

Python fixtures under `fixtures/` are written by `make_fixtures.py` (independent
contract writer). Our Rust readers consume them:

```bash
# from repo root
cargo test -p hipfire-quantize --lib hessian_io -- --nocapture
# includes python_fixture_hfhs_roundtrip

cargo test -p hipfire-quantize --bin collect_e8_hessian -- --nocapture
# includes python_fixture_e8h1_roundtrip
```

Do **not** run the full workspace suite from this slice.

## Latest harness numbers (seed 0xc0ffee, torch 2.11 CPU)

Stated floor (f64 CPU): `max_abs <= 1e-14`, `rel_frob <= 1e-14`.

| Stage | Comparison | max_abs | rel_frob | norm_ratio | cosine | verdict |
|---|---|---:|---:|---:|---:|---|
| 0 | Hessian two-path | 1.8e-15 | 2.3e-16 | 1.0 | 1.0 | floor |
| 0 | U helper vs manual | 0 | 0 | 1.0 | 1.0 | floor |
| 0 | GPTQ blocksize 128 vs 8 | 0 | 0 | 1.0 | 1.0 | floor |
| 1 | paper vs 2×HFHS | 0 | 0 | 1.0 | 1.0 | PASS |
| 1 | paper vs HFHS direct | 1.37 | 0.5 | **2.0** | 1.0 | FINDING (scale) |
| 1 | E8H1 block0 vs full raw | 3.6e-15 | 1.7e-17 | 1.0 | 1.0 | PASS |
| 2 | U paper vs hipfire frac | 1.1e-16 | 1.5e-16 | 1.0 | 1.0 | PASS |
| 2 | U^T U vs H^{-1} | 3.3e-16 | 3.0e-16 | 1.0 | 1.0 | PASS |
| 2 | wrong U U^T vs Hinv | 0.22 | 0.27 | 1.0 | 0.96 | distinguished |
| 2 | absolute damp=0.01 | 4.8e-3 | 5.7e-3 | 0.995 | 1.0 | FINDING |
| 3 | Q paper vs hipfire (scale-inv) | 0 | 0 | 1.0 | 1.0 | PASS |
| 3 | recon paper/hip/RTN | 163.8 / 163.8 / 221.2 | | | | GPTQ < RTN |

Fault injection (detected max_abs vs clean):

| Injection | max_abs | cosine | caught? |
|---|---:|---:|---|
| transpose_h (break symmetry) | 1.64e-1 | 0.998 | CAUGHT |
| no_damp | 1.05e-1 | 0.9997 | CAUGHT |
| no_invperm | 2.30 | 0.009 | CAUGHT (perm signature) |
| no_error_feedback | 1.64e-1 | 0.995 | CAUGHT |
| wrong U form L_H^{-T} (extra) | 1.64e-1 | 0.994 | CAUGHT |

**Blind spots:** none of the four required injections were missed on this draw.
Caveats still apply:

- Symmetric H alone makes a pure transpose a no-op; the harness breaks symmetry
  before transposing so that defect stays visible.
- Near-diagonal H can shrink the wrong-U-form gap; Stage 2 still separates
  `U U^T` from `U^T U` on the damped inverse itself (`max_abs≈0.22`).
- Scale-only Hessian mismatches (HFHS factor 2) do **not** fail Stage 3 when
  damp is fractional — by design (OBS ratio invariance). The Stage 1 direct
  compare is what flags them.
