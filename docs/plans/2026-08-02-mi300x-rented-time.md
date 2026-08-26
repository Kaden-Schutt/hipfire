<!-- SPDX-License-Identifier: Apache-2.0 -->
# What to run on rented MI300X time

Date: 2026-08-02 · Status: living · Context: the MI300X is metered

## The selection rule

The MI300X VF is rented by the minute. Two properties make it different from
the rest of the fleet, and **work that does not need one of them should not run
here**:

1. **192 GiB HBM** — 8x a 24 GB RDNA card. Enables the 150.756 GiB original DS4
   checkpoint, long context, and MoE at scale.
2. **CDNA3 matrix cores** — native FP8 MFMA, wave64. A different architecture
   from hipfire's RDNA production target, and the one datacenter deployments
   run on.

Anything targeting gfx11/gfx12 belongs on a local 9070 XT. That includes the
Q8 prefill specialisation, the DDTree gfx1100 regression, MQ3 work, and most of
Redline. Those are important; they are just not worth rented CDNA minutes.

## Ranked

### 1. gfx1151 long-context proving — IN PROGRESS

**The goal: serve DS4 at 1M context on a Strix Halo.** No discrete consumer
card can hold the ~84 GB MQ2R artifact; a 128 GB unified-memory APU can. DS4 is
already a first-class gfx1151 citizen — 61 `gfx1151` arch gates in DS4's
`forward.rs`, with `e8_u4_on`, `e8_wo_grouped_on`, `e8_prefill_b2/b4`,
`hc_pingpong_on` and `hc_finalize_fused_on` pinned ON for `.mq2r`. Serving a
frontier-class MoE at 1M on a mini-PC would be genuinely unique.

DS4-Flash-0731 is a **1M context** model: `original_seq_len 65536` x
`rope_factor 16` = 1,048,576. YaRN does the extension, so 65536 is the
pre-scaling number, not the limit.

The MI300X's job here is to be a **fast proving and data-collection
instrument**: it has 192 GiB, so it finds the correctness and performance walls
without the memory pressure, and gfx1151 becomes a port against known numbers
instead of an exploration.

**KV is cheap; weights dominate.** KV is an MLA latent (`head_dim 512 +
rope 64` shared across all 64 heads), not per-head:

| ctx | q8 | fwht4 | fwht3 |
|---|---|---|---|
| 65,536 | 1.7 GB | 0.9 GB | 0.6 GB |
| 524,288 | 13.6 GB | 6.8 GB | 5.1 GB |
| 1,048,576 | 27.2 GB | 13.6 GB | 10.2 GB |

With an 84 GB model, 1M at fwht4 totals ~97.6 GB — fits 128 GB, not a 96 GB
allocation. **Use FWHT KV quantization; `asym*` is legacy** and must not be
used for new results (`kv_cache_write_fwht256_2bit`,
`kv_cache_write_asym_k_fwht{2,3,4}`).

**The predicted bottleneck is the indexer, not flash attention.** Decode
attends to `window 128 + topk 512` = 640 keys per step *at any context* — that
is the architecture's whole point. But the indexer must score every compressed
slot to pick that top-512, which is `O(ctx / ratio)` per layer per token:

| ctx | attn keys/step | indexer scores/layer/step |
|---|---|---|
| 1,024 | 640 | 135 |
| 65,536 | 640 | 8,641 |
| 1,048,576 | 640 | **138,264** |

At 1M a ratio-4 layer scores 262,144 slots per token across 43 layers. If
decode is abysmal at 500K, that is the reason — and it is a different fix from
an attention-kernel problem. Confirm with a profile rather than assuming.

**And top-k selection has never been exercised.** `index_topk = 512` is a
strict no-op below ~2048 tokens (a ratio-4 layer has only 256 compressed slots
at 1024), proven against the verbatim PyTorch reference. Every DS4 run in this
project's history has been <= 1024 tokens. Long context is its first real test,
so a selection bug would surface here for the first time.

### 2. gfx942 FP8: close the CDNA gap in Radiowave — DEFERRED

Radiowave landed gfx11 OCP FP8 lowering on 2026-07-30 (`1d6cfd08a`) with a
bench harness and results for two RDNA parts. There is **no CDNA row**:
gfx1100 `ocp-e4m3` at 355.6 logical_wmma_M/s (1.03x over its FP16 control) and
gfx1201 at 1182.2 (**1.96x**, native FP8 WMMA). gfx942's MFMA FP8 should match
or beat that and has never been measured.

The exclusion is deliberate: `recipes_fp8.rs:16` targets **OCP** FP8 while
CDNA3 speaks **FNUZ** (different exponent bias and special-value encoding),
with tests at 630/650/679/682 asserting it. DS4 makes it concrete — its
checkpoint is `torch.float8_e4m3fn`, OCP E4M3FN, on FNUZ hardware.

Deferred because it is an optimization, while item 1 is a capability nobody
else has. Worth picking up when long-context proving is done.

### 3. DS4 gfx942 serving performance

CDNA support is real and active — 562 lines in `rdna-compute/src/cdna/gfx942.rs`,
dedicated `*.gfx942.hip` kernels, and **76 gfx942 arch gates** in DS4's
`forward.rs`. So there is a live serving path here whose performance nobody has
characterised against the RDNA numbers the project quotes.

Natural targets: the A3B MoE DFlash line (`AGENTS.md` pins fixtures and a best
observed 151.00 tok/s at tau 2.711), and MQ2R decode throughput. Needs the
model resident, so it wants the memory.

### 4. The DS4 parent 12.7x gap — deprioritised

`crate::parent` scores PPL 59.507 against the torch teacher's 4.693 and is
marked NOT A CALIBRATION REFERENCE. Residuals match the teacher to sub-1%,
every measured stage sits at its quantization floor, and the head path is clean,
so the defect is somewhere in layers 3-41. Nothing downstream waits on it: the
torch harness is the teacher now. Resume only if a second implementation becomes
worth having on its own merits.

### Not here

- **Lloyd shrinkage** (`docs/plans/2026-08-02-lloyd-shrinkage-gain.md`) —
  CPU-only.
- **Gates 7-9 / parent-calibrated GPTQ** — gated on a value test, and the bar
  moved to beating PPL 9.254 after the `route_scale` fix.

## Operational notes, learned the hard way

- **One agent per remote checkout.** `/root/hipfire-work/ds4-parent-gate` is
  pinned at an old commit and receives files by copy, so it silently lags. Two
  concurrent agents disagreed about a constant for an hour because one rebuilt
  over the other. Author locally, sync over, never edit only on the box.
- **`pgrep -f` matches the polling script's own command line.** Three separate
  pollers hung today, one for 40 minutes against a process that had finished in
  33 seconds. Use `pgrep -af "[d]s4_..."` or poll for output artifacts.
- **Do not wrap gate binaries in `set -e`.** `ds4_parent_forward_gate` exits
  nonzero when its gate fails, which is not the same as the run failing; it
  silently killed a capture chain.
- **sha256 of an 82 GB artifact costs ~5 minutes** of single-threaded CPU and
  looks identical to a hang from outside. `ds4_quant_plog --trust-sha256` skips
  the re-hash when a campaign already verified the same path and digest.
- **Never touch `/root/hipfire-work/ds4-gfx942-port`** — someone else's
  uncommitted work.
