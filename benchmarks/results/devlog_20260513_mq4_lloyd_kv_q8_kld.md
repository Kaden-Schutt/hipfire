# Dev log 2026-05-13 — Q8-KV KLD probes (9B, gfx1151): MQ4-Lloyd + MQ6-q8conv1d

**Branch:** `feat/issue-182-mq4-lloyd` (HEAD `1deeaa5e`, post-master-merge that
brought in halfsplit RoPE + HFP4G32 + MFP4G32).
**Hardware:** gfx1151 (Radeon 8060S, AMD Ryzen AI Max+ 395 / Strix Halo APU,
LPDDR5x ~250 GB/s), ROCm 7.12.
**Goal:** answer "what is MQ4-Lloyd / MQ6 KLD if the KV cache is Q8 instead of
the canonical asym3?" — a one-off editorial probe, not a re-baseline of the §6
matrix in `docs/plans/issue-113-quant-quality-eval.md`.

Three runs (all 9B Qwen 3.5, prefill scoring, gfx1151, Q8 KV cache):
- **MQ4-Lloyd** (conv1d stored as `MQ4G256Lloyd`), 512 chunks. Baseline.
- **MQ6-q8conv1d** (conv1d stored as `Q8_F16`), 20-chunk smoke + 512-chunk
  follow-up. The smoke was within CI of the 512 result; both are kept.
- **MQ4-Lloyd-q8conv1d** (conv1d stored as `Q8_F16`), 20-chunk smoke +
  512-chunk follow-up. Added after cherry-picking `2188e841` ("default
  conv1d weight to Q8 (KLD 0.30 → 0.25)") to isolate the conv1d-precision
  contribution from the MQ4-Lloyd vs MQ6 weight-quant family change. Smoke
  was within CI of the 512 result.

## Headlines

**MQ4-Lloyd 9B with Q8 KV (prefill, 512-chunk slice): mean KLD =
0.3114 (95 % bootstrap CI 0.2999 – 0.3236).** p99 = 18.69, PPL = 9.085.

The 50-chunk smoke that preceded it landed at 0.3005 (CI 0.2609 – 0.3474) —
within CI of the 512-chunk result, confirming the slice has no large-scale
positional drift that would invalidate the smoke.

**MQ6-q8conv1d 9B with Q8 KV (prefill, 512-chunk slice): mean KLD =
0.05095 (95 % bootstrap CI 0.0473 – 0.0549).** p99 = 8.68, PPL = 9.186.
CI half-width 0.004 = 7.8 % relative — tight.

The 20-chunk smoke that preceded it landed at 0.0568 (CI 0.0314 – 0.0950),
within CI of the 512 result. Smoke p99 was 12.13 vs 8.68 at full power — n=20
caught a rare flat-distribution chunk that the bootstrap softened but didn't
fully wash out.

**Note on KLD vs PPL divergence on MQ6-q8conv1d.** MQ6-q8conv1d has 6.1×
lower KLD than MQ4-Lloyd (0.051 vs 0.311) yet **higher** PPL (9.186 vs 9.085).
KLD measures whole-distribution divergence from BF16 over the vocab; PPL
measures the probability assigned to the actual next token. A quant can
preserve tail mass well (low KLD) while slightly hurting argmax confidence
(higher PPL). This is the regime PRD §2 motivates the switch from PPL to
KLD-primary scoring for.

**MQ4-Lloyd-q8conv1d 9B with Q8 KV (prefill, 512-chunk slice): mean KLD =
0.2519 (95 % bootstrap CI 0.2418 – 0.2626).** p99 = 17.78, PPL = 8.8033. CI
half-width 0.010 = 4.0 % relative.

The 20-chunk smoke that preceded it landed at 0.2492 (CI 0.179 – 0.346),
within CI of the 512 result.

**Paired-difference vs MQ4-Lloyd baseline (same 512 chunks):** q8conv1d wins
on **511 / 512 chunks** (one regression of 0.007). Per-chunk KLD delta 0.0595
(95 % paired-bootstrap CI 0.0568 – 0.0624 — CI half-width 0.003, does not
include zero). Both KLD (−19.1 %) and PPL (−3.0 %) improve, unlike the
MQ6→MQ6-q8conv1d direction where PPL slightly worsened. This replicates the
commit message's 0.30 → 0.25 claim almost exactly (0.311 → 0.252) and isolates
~19 % of the MQ4-Lloyd → MQ6 KLD gap to the conv1d-precision change rather
than the bulk-weight change.

## Numbers in context

Existing 9B rows from `result-table.md` (all asym3 KV, gfx1100):

| Variant | Mode | KV | Mean KLD |
|---|---|---|---:|
| MQ3 uniform   | per-token | asym3 | 2.622 |
| MQ3-Lloyd     | per-token | asym3 | 1.691 |
| MQ4 uniform   | per-token | asym3 | 0.876 |
| MQ4 uniform   | prefill   | asym3 | 0.817 |
| MQ6 uniform   | per-token | asym3 | 0.625 |
| **MQ4-Lloyd** | **prefill** | **Q8** | **0.311** (this run, 512/1175 ch, gfx1151) |
| **MQ6-q8conv1d** | **prefill** | **Q8** | **0.051** (this run, 512/1175 ch, gfx1151) |
| **MQ4-Lloyd-q8conv1d** | **prefill** | **Q8** | **0.252** (this run, 512/1175 ch, gfx1151) — 511/512 paired wins vs MQ4-Lloyd on identical chunks |

**Direct comparison is unsafe.** The this-run number differs from every other
row in at least one of: KV mode (Q8 vs asym3), scoring mode (prefill vs
per-token — ~7 % kernel-path numerical effect per PRD §5.3), arch (gfx1151 vs
gfx1100), and quant family (MQ4-Lloyd vs MQ4 vs MQ6). The asym3 / per-token
MQ4-Lloyd row on the same hardware does not exist. Attributing the 2.6×
improvement over uniform MQ4 to any one of these four axes is not supported
by this run.

What this run *does* establish: an MQ4-Lloyd 9B inference with Q8 KV produces
a KLD figure (0.31) that sits closer to MQ6's neighbourhood than to MQ4's,
and the per-sequence variance behaves normally. There is no harness failure
or degenerate output mode.

## Bench config

Shared between both runs:

| Field | Value |
|---|---|
| Eval tool | `target/release/examples/eval_hipfire --features deltanet` |
| Eval binary md5 | `6981886ff2ce12a3a07eabc93972dfc8` |
| Reference | `benchmarks/quality-baselines/refs/qwen3.5-9b-bf16.kldref.bin` (sha256 `06948cd36bab71fce2df5d9af1be03c9cfb4090637d881056a6937a29caa65a7`) |
| Slice md5 | `83b0205a304bf4e52172ecdb05f2e895` |
| `--kv-mode` | `q8` |
| `--scoring-mode` | `prefill` |
| Env (auto-forced) | `HIPFIRE_NORMALIZE_PROMPT=0 HIPFIRE_GRAPH=0 HIPFIRE_KV_MODE=q8` |
| ROCm | `/opt/rocm-7.12` |

Per-run:

### MQ4-Lloyd

| Field | Value |
|---|---|
| Model | `/home/kread/.hipfire/models/qwen3.5-9b.mq4-lloyd` (md5 `cd8626a1701e65055d31986bb5ec840c`, re-quantized on this branch from `Qwen/Qwen3.5-9B` safetensors at snapshot `c2022362...` — see "Quantize provenance" below) |
| `--max-chunks` | `512` (of 1175 in the slice) |
| Wall-clock | 76 min (`116 tok/s` end-to-end, 523 776 scored tokens). First 30 chunks include rmsnorm JIT compile; steady-state throughput is 116 tok/s. |
| Output | `benchmarks/quality-baselines/results/2026-05-13-kv-q8/per-seq/qwen3.5-9b.mq4-lloyd-kvq8-c512__gfx1151__prefill.kldseq` |

### MQ4-Lloyd-q8conv1d (added after 2188e841 cherry-pick)

| Field | Value |
|---|---|
| Model | `/home/kread/.hipfire/models/qwen3.5-9b.mq4-lloyd-q8conv1d` (md5 `d26bcfca41e30caad4b21bddafc20bd1`, 6056.2 MB; re-quantized from the same `Qwen/Qwen3.5-9B` snapshot as the MQ4-Lloyd baseline after cherry-picking `2188e841` "default conv1d weight to Q8") |
| `--max-chunks` | `512` (of 1175) — preceded by a 20-chunk smoke (same flags) |
| Quantize log evidence | every `linear_attn.conv1d.weight` row printed as `Q8_F16: ... 64.0 KB → 34.0 KB` (vs the `MQ4G256Lloyd: ... 64.0 KB → 20.0 KB` rows in the baseline quantize log) |
| Wall-clock | 20-chunk smoke ~3 min (88 tok/s); 512-chunk run ~1h25 (102 tok/s steady — kernels warm from earlier runs) |
| Outputs | `benchmarks/quality-baselines/results/2026-05-13-kv-q8/per-seq/qwen3.5-9b.mq4-lloyd-q8conv1d-kvq8-c20__gfx1151__prefill.kldseq` (smoke) and `qwen3.5-9b.mq4-lloyd-q8conv1d-kvq8-c512__gfx1151__prefill.kldseq` (full) |
| Independent mean KLD | 0.2519 (CI 0.2418 – 0.2626, n=512) |
| Paired vs MQ4-Lloyd (512 ch) | per-chunk delta mean 0.0595 (95 % CI 0.0568 – 0.0624), 511/512 chunks improve, one regression of 0.007 |

### MQ6-q8conv1d

| Field | Value |
|---|---|
| Model | `/data/hipfire/qwen3.5-9b.mq6-q8conv1d` (md5 `7b1b6d822d0f662c78f9c75675c8639b`, 7.30 GB; pre-existing on this branch — not re-quantized in this session) |
| `--max-chunks` | `512` (of 1175 in the slice) — preceded by a 20-chunk smoke (same flags) |
| Wall-clock | 20-chunk smoke ~6 min (steady-state ~60 tok/s); 512-chunk run 1h45 (steady-state 83 tok/s — kernels warmed across both runs, so the second run never paid JIT-compile overhead) |
| Outputs | `benchmarks/quality-baselines/results/2026-05-13-kv-q8/per-seq/qwen3.5-9b.mq6-q8conv1d-kvq8-c20__gfx1151__prefill.kldseq` (smoke) and `qwen3.5-9b.mq6-q8conv1d-kvq8-c512__gfx1151__prefill.kldseq` (full) |

## Eligibility — did batched WMMA prefill actually fire?

PR #197 lands MQ4-Lloyd in `is_batchable_la` for gfx1100/1101/1102/1151
(`crates/hipfire-arch-qwen35/src/qwen35.rs:4072-4075`). gfx1151 is in.
Indirect evidence the batched path fired: end-to-end throughput stabilised
at 116 tok/s. The per-token sibling rate on the same model + arch is 44 tok/s
(`devlog_20260509_mq4_lloyd_gfx1151_bench.md`); 116 tok/s end-to-end is
~2.6× faster, consistent with the batched-LA stack carrying the LinearAttention
layers while the FA layers and the lm_head fan-out go per-position. The
arch-table assertion in the PRD §5.4 ("MQ4-Lloyd: per-token") is stale on
this branch.

The Q8 KV decoder is a separate (asym2/asym3/asym4/q8) gate inside FA — Q8 is
already exercised on this branch (see `kv_cache.quant_q8` in qwen35.rs:3651,
4425, etc.) and did not auto-fallback.

## Quantize provenance

Original `qwen3.5-9b.mq4-lloyd` on disk dated 2026-05-06 was rejected by the
loader on this branch with `unsupported quant_type 21 for layers.0.linear_attn.conv1d.weight` — the qt-renumber from 21 to 30 in the HFP4G32 merge
(`596dd231`, "renumbered to avoid HFP4G32=21 collision") makes any pre-merge
MQ4-Lloyd .hfq unreadable. Re-quantized on this branch:

```
./target/release/hipfire-quantize \
  --input /data/cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a/ \
  --output /home/kread/.hipfire/models/qwen3.5-9b.mq4-lloyd \
  --format mq4-lloyd \
  --allow-mq4-lloyd
```

Wrote 6055.8 MB; md5 `cd8626a1701e65055d31986bb5ec840c`.

## Caveats (mandatory preamble per PRD §8)

- PPL measured with `HIPFIRE_NORMALIZE_PROMPT=0` and `HIPFIRE_GRAPH=0`;
  historical baselines in `lloyd_max_findings_20260501.md` were likely run
  with normalize ON. PPL 9.085 here is not directly comparable to any pre-2026-04-26 PPL number.
- Slice is the frozen `wikitext2-1024s-2048ctx.txt`. PR #115's slice is not committed.
- KV mode is **Q8**, not the canonical asym3. This row does not belong on the
  Pareto matrix in PRD §6 without an adjacent asym3-KV companion run to
  decompose the KV-vs-quant contributions.
- Scoring mode is prefill (PRD §5.3): **do not cross-compare** to per-token
  historical rows. The ~7 % kernel-path bias is in the direction of *lower*
  prefill KLD vs per-token.
- All KLD values are lower bounds on full-vocab KLD (top-K=256 residual
  cross-term assumes both distributions miss similarly in the tail; PRD §2).
  This caveat applies to all rows but is worth re-stating since the headline
  number is the smallest on the table.
- 512 / 1175 chunks. The 95 % bootstrap CI on the per-sequence mean has half-width
  ~0.012 (3.9 % relative) which is tight, but the remaining 663 chunks were
  not scored.

## What this run does NOT answer

- The asym3-KV MQ4-Lloyd 9B prefill number on gfx1151. Without it, the
  Q8-vs-asym3 KV contribution is unisolated.
- Same on gfx1100 (would also disentangle arch).
- Whether the same Q8-KV-prefill improvement is preserved in per-token mode
  (controls for the kernel-path numerical bias).
- Full-slice (1175 chunks) — 512 was the user-set scope for this probe.

## Files

```
benchmarks/quality-baselines/results/2026-05-13-kv-q8/per-seq/
  qwen3.5-9b.mq4-lloyd__gfx1151__kv-q8__prefill__c50.kldseq        (50-chunk smoke;  KLD 0.3005)
  qwen3.5-9b.mq4-lloyd-kvq8-c512__gfx1151__prefill.kldseq          (512-chunk run;   KLD 0.3114)
  qwen3.5-9b.mq4-lloyd-q8conv1d-kvq8-c20__gfx1151__prefill.kldseq  (20-chunk smoke;  KLD 0.2492)
  qwen3.5-9b.mq4-lloyd-q8conv1d-kvq8-c512__gfx1151__prefill.kldseq (512-chunk run;   KLD 0.2519)
  qwen3.5-9b.mq6-q8conv1d-kvq8-c20__gfx1151__prefill.kldseq        (20-chunk smoke;  KLD 0.0568)
  qwen3.5-9b.mq6-q8conv1d-kvq8-c512__gfx1151__prefill.kldseq       (512-chunk run;   KLD 0.0510)
```

The `<variant>-kvq8-c<n>` form makes the 3-segment `kld_reduce.py` parser
work without modification. The MQ4-Lloyd 50-chunk smoke filename retains the
5-segment legacy form (consistent with the existing 2026-05-12
deltanet-discriminator kldseq) and is not parsed by `kld_reduce.py`.
