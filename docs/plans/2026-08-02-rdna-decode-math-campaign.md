<!-- SPDX-License-Identifier: Apache-2.0 -->
# Engine-wide math campaign: bytes per token on RDNA

Date: 2026-08-02 · Status: proposed · Scope: engine-wide, RDNA-shaped

## The unifying claim

**RDNA decode is bandwidth-bound, and every lever hipfire pulls is denominated
in the same currency: bytes moved per token.** Quantization format, KV layout,
speculative decode block size, MoE expert residency, prefetch policy — all of
them trade in that one unit.

Today each is tuned separately, by hand, against separate intuitions. There is
no shared cost model, so the engine cannot answer questions like: is saving
2 B/token on KV worth more than raising the spec-decode block size by one? Does
expert prefetch pay for itself at 24 GB? Which tensor class deserves the next
bit?

The campaign is to build that model, validate it, and then use it.

## Why RDNA specifically

The constraint set differs from datacenter parts in ways that change the
answers, not just the constants:

- Consumer bandwidth is roughly 600–960 GB/s against HBM's 5+ TB/s, so the
  compute/bandwidth ratio is inverted. Decisions that are obviously right on
  MI300X can be wrong on gfx1100.
- wave32 with 16x16x16 WMMA rather than wave64 MFMA.
- gfx11 has no native FP8 (software decode to FP16); gfx12 does.
- Unified-memory APUs (gfx1151) change the tradeoff again: no PCIe wall for
  expert weights, but the CPU is competing for the same bandwidth.

A cost model that is not RDNA-shaped will optimize for the wrong machine.

## The keystone

**A validated bytes-per-token roofline for gfx1100 / gfx1151 / gfx1201**,
decomposed by tensor class: routed experts, shared expert, attention
projections, norms, KV read, KV write, activations.

Validation means predicting measured decode tok/s from the model within some
stated error, on real hardware, across at least two architectures and two
models. Until it predicts, it is folklore with arithmetic attached.

Everything below becomes a well-posed optimization only once this exists.

## The problems it unlocks

### 1. Optimal bit allocation across tensor classes

Given a total bytes-per-token budget, how should bits be distributed? This is a
Lagrangian / water-filling problem: allocate to equalize marginal quality loss
per marginal byte.

What was missing until now is the quality term — there was no trustworthy
reference to measure sensitivity against. There is now: a verified teacher at
PPL 4.693 with a KLD floor of 0.0404
(`/mnt/scratch/quantization/deepseek-v4-flash-0731-teacher/`). Per-class
sensitivity is measurable by perturbation.

Prior: uniform bpw across classes is almost certainly not optimal. The 4-bit
shared tier already measures 0.9998 retained — saturated — while the 2-bit
routed tier is where all the mass and all the loss are.

### 2. Speculative-decode economics under bandwidth binding

On RDNA a target forward is approximately "read all the weights", so spec
decode amortizes a bandwidth cost, not a compute cost. The optimal block size
and tree shape therefore depend on the *bandwidth* ratio between draft and
target, which is a different objective from the accept-rate-only framing.

Well-posed: given a draft's per-position accept distribution and the
draft/target byte ratio, maximize expected accepted tokens per unit bandwidth
under a node budget.

`AGENTS.md` records DDTree on gfx1100 as a structural perf regression with
Path C and Path D unresolved, so this is currently stuck and worth the math.

### 3. MoE expert residency and prefetch

24 GB cannot hold 256 experts; a unified-memory APU can hold them but shares
bandwidth with the CPU. The exploitable structure: **routing is computed by the
gate before the expert weights are needed**, so this is prefetching with
lookahead rather than blind caching.

Well-posed: given the routing distribution and a residency budget, choose a
placement and eviction policy minimizing expected bytes fetched per token.
Classical caching theory applies, with the twist that the request sequence is
partially observable one step early.

### 4. KV eviction

Which tokens to drop under a KV budget. Sinks and heavy-hitters have real
structure, and hipfire already has CASK/TriAttention machinery. DS4 makes this
less urgent — its MLA latent KV is small (1.7 GB at 64K, 27 GB at 1M) — but it
binds for conventional MHA models.

## Resource split

**Reasoning / derivation** — the cost model, the allocation Lagrangian, the
spec-decode objective, the prefetch policy. These have right answers and need
thought rather than hardware.

**Local RDNA hardware** — roofline validation. This *must* be RDNA; a CDNA
measurement cannot validate an RDNA roofline.

**MI300X** — ground-truth *data*, not hardware characterisation. It is CDNA
with HBM, so it cannot stand in for gfx1100. What it is good for: collecting
accept distributions, routing statistics, and quality-versus-bpw curves against
the teacher, fast and without memory pressure. Algorithm-shaped data, not
hardware-shaped data.

## What this campaign is not

- Not a quantization-format hunt. A specific negative result: **E8 lattice VQ
  at 2 bpw would be worse than the scalar Lloyd-Max already shipped.** At low
  rate, distribution adaptation beats cell geometry — Lloyd places codewords by
  density while a lattice places them uniformly and relies on a 0.65 dB
  space-filling gain that only materializes when the source is locally uniform
  inside a cell. With ~4 effective levels the Gaussian tail dominates and
  boundary handling costs more than the granular gain returns. Lattice wins at
  higher rates, which is exactly where hipfire already uses E8.
- Not a per-kernel tuning exercise. Kernel scheduling matters but is a separate,
  more local problem.
- Not blocked on the DS4 parent work, which is closed and labelled.

## First step

Build and validate the bytes-per-token model for one architecture and one
model, and show it predicts measured decode tok/s. If it does not predict, the
campaign stops there and that is a cheap answer. If it does, the four
optimization problems above all become tractable at once.
