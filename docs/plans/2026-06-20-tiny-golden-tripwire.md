# Plan: tiny-fixture golden-output tripwire as the gate front-tier

Status: proposed (2026-06-20)
Extends: `TODO.md` §"Tiny random-init fixtures + golden-output tripwire" (§325)
and §"Legible golden for the tiny fixtures" (§293). This plan is the *build-out
+ gate wiring* of that design, not a new design.

## Why

The pre-commit coherence battery (`tests/coherence-gate.sh`) is the only thing
guarding forward-pass output, and it has three structural costs:

1. **GPU-bound + slow.** ~20 real-model loads, 2–4 min, contends for the GPU
   (the serving-core A0 work stalled on exactly this — another box held the
   GPU lock).
2. **Qualitative.** It hard-fails only on panic / zero-tokens / timeout;
   correctness is a *human reads the markdown report* step. That "looked fluent
   to me in 300 tokens" hole is how the long-output attractor-collapse class has
   slipped through.
3. **All-or-nothing.** Any hotspot change runs the whole matrix, including
   rows whose kernels weren't touched.

A deterministic golden over a tiny random-init model is a far more sensitive,
*mechanical* tripwire: it trips on the **first divergent token**, long before a
drift visibly collapses, and it runs in seconds. The two-tier policy (tiny
always → escalate to the real battery on drift) is already specified in §325;
this plan finishes the pieces and wires them in.

**This is a tripwire, not a replacement.** §325's own conclusion holds: random
weights are same-selection-branch coverage, not identical-tile, and lack
trained-weight magnitude structure — so some precision/magnitude-sensitive
cascades won't reproduce. The 35B golden stays as the behavioral backstop.

## Current state (what already exists)

- **`hipfire-quantize::fixture::emit_fixture(arch, out_dir, seed)`** — seeded
  (`SplitMix64`) emitter that writes HF `safetensors` + `config.json`, then the
  caller runs the **normal `--input` quantize path** (so it exercises the
  arch name-mapper + real codec — full-pipeline coverage). Has `preset()`
  (dense) and `moe_preset()`. Currently emits bf16.
- **`crates/hipfire-runtime/examples/fixture_golden.rs`** — golden runner. Now
  has `--mode {tf,ar}` (AR landed in P1/D1, commit `82542fae`); still
  qwen35-family only (handles both arch 5 dense and arch 6 MoE via the qwen35
  module, so the current fixture set needs no separate arch dispatch).
- **`tests/fixture-golden-gate.sh`** — already wires the tripwire: emits both
  fixtures (dense + MoE) → quantizes (`mq4` only) → runs `fixture_golden`
  (`tf`, `len=16`) and does **two checks: (1) determinism** (run twice, hashes
  must match — catches a kernel going nondeterministic), **(2) baseline** drift
  vs committed, **keyed per `gpu_arch × model_arch`**. On drift it prints
  "escalate to the 35B golden (agentic-gate.sh)". `--record` rebaselines. So
  the **determinism check, per-arch baselines, escalation framing, and
  rebaseline command already exist** — earlier drafts of this plan
  over-credited them as "delta"; they are not.
- **`tests/fixture-golden-baselines.txt`** — committed goldens. Currently only
  `gfx1151` (the halo box), `qwen3_5` + `qwen3_5_moe`, `mq4`, `len=16 tf`.
- **`tests/fixture-roundtrip-nogpu.sh`** — emit→quantize round-trip, **already
  in `no-gpu-ci.sh`** (no forward, so GPU-free).

What the existing runner could NOT do: it was **teacher-forced** — fixed inputs,
so it structurally **cannot produce an attractor**. D1 (below) fixed that.

## The delta to build

### D1 — Free-running AR golden (the attractor-relevant upgrade) — ✅ LANDED (82542fae)
`fixture_golden.rs` now has `--mode {tf,ar}`. `ar` seeds a short fixed prompt
(`--prompt-len`), then feeds `argmax` back as the next token for the rest of
`--len`, growing the KV cache — the real decode loop, and the path that can
surface an attractor. `tf` stays the default and is byte-preserved (validated:
reproduces the committed `0xb9929ff22fec2015` gfx1151 baseline exactly; `ar` is
deterministic across two runs). `logit_hash` (sensitive) + the argmax token
sequence (robust) are both emitted.
- No separate arch dispatch was needed: the runner already handles arch 5 dense
  and arch 6 MoE through the qwen35 module (both current fixtures).
- *Remaining for D1's gate side:* `fixture-golden-gate.sh` still calls `tf
  len=16`; switching it to `ar` with a longer len (§293: 256) + rebaselining is
  the P3 wiring step, gated on the MoE determinism pin (D3) since a long AR tail
  is where the atomicAdd near-tie flips bite.

### D2 — The fixture *matrix* (arch × quant-format)
§325 builds two fixtures (tiny dense arch 5, tiny MoE arch 6). Add the
**quant-format axis** the user asked for: one bf16 preset per arch, quantized
to each runtime format that has its own dequant kernel path — `{MQ4, MQ3, Q8,
lloyd-MQ3, lloyd-MQ4, MQ6}` — so each format's dequant kernel gets golden
coverage. The cell set is `{dense, MoE} × {formats}` minus unsupported combos.
- Generation stays **materialize-then-real-load**, not in-memory synthesis. The
  user floated a "loader fills a template in memory" path; §325 deliberately
  chose emit-safetensors→quantize→load-through-the-unmodified-loader because the
  loader / name-mapper / sidecar-attach code is *itself* what we're gating (the
  AWQ-sidecar-drop and MQ3-gating bugs lived there). The seed gives the
  dynamism; a content-hash cache (`(arch, preset, format, quantizer-version)`)
  keeps regeneration cheap without an in-memory branch that could drift from the
  real path.

### D3 — Determinism (so the golden is stable run-to-run)
- **Dense (arch 5):** no `atomicAdd` → byte/token-exact golden is stable; any
  drift is real signal.
- **MoE (arch 6):** MoE-down combine uses `atomicAdd` with documented
  non-deterministic final bits (`kernels.rs:3751`,
  `gemv_hfq4g256_moe_down.hip:19–23`). **Pin the fixture golden to the in-tree
  deterministic combine** (`moe_down_combine_k8_batched`, `kernels.rs:3748`).
  Bonus: the harness then doubles as a **determinism gate** (catches anything
  re-routing MoE-down through the atomicAdd path).
- **Near-tie backstop:** the sensitive (random) tail tokens sit on decision
  boundaries. Where a token-exact diff proves too flaky even with the pinned
  combine, fall back to a **top-token-with-logit-margin** assertion rather than
  raw equality.
- **Goldens are per-arch-family.** Greedy argmax differs across GPU archs
  (gfx1100 vs gfx1201 vs gfx1010) from kernel/precision differences, so commit
  one golden per `(fixture-cell, arch-family)` and have each box check its own.

### D4 — Gate wiring (the two-tier policy from §325)
**Most of this already exists** in `tests/fixture-golden-gate.sh` (determinism +
per-arch baseline + escalation message + `--record`). The real remaining wiring:
- **Hook it into pre-commit.** The gate is currently standalone — `.githooks/
  pre-commit` does NOT call it. Wire it ahead of the `coherence-gate.sh` call so
  it actually runs as the cheap front tier whenever the HOTSPOT set is touched
  (now including renames, post-`ACMR` fix). On front-tier drift, run the real
  battery rather than blocking.
- **Multi-arch baselines.** Only `gfx1151` is recorded; the fleet (gfx1100 /
  gfx1201 / gfx1010) each need `--record` runs committed.
- *(reference)* The front tier already diffs against committed goldens keyed per
  `gpu_arch × model_arch`; the matrix (D2) just adds the format axis to the key.
- **Escalation, not block:** on drift, print which cells drifted and **run the
  real `coherence-gate.sh`** (the 35B *golden*, not the coarse JSON-valid
  check). A drift is never a hard block by itself — that would recreate the
  byte-exact gate that was removed for blocking legitimate numeric fixes.
- **Rebaseline workflow (deliberate, never automatic):**
  - *tiny-only drift:* a tiny-specific change → `tests/tiny-golden.sh --rebaseline`.
  - *35B-also drift:* a real forward-pass change → rebaseline both, deliberately,
    after confirming the new output is still coherent.
- **Golden storage:** committed files under `tests/golden/<arch-family>/
  <cell>.golden`, each carrying its params (prompt seed, len, warmup, format,
  quantizer-version) + an md5 of the input prompt token stream (per CLAUDE.md
  prompt-md5 discipline).

## Phasing

- **P1 — D1 (AR runner):** ✅ landed (82542fae). `--mode ar` free-running greedy
  decode; `tf` preserved; validated on gfx1151.
- **P2 — D2/D3:** extend `fixture-golden-gate.sh` to the format axis (MQ3/Q8/
  lloyd/MQ6, not just mq4); add the MoE deterministic-combine pin so a long AR
  tail is stable; a regenerate-and-cache helper.
- **P3 — D4:** switch the gate to `--mode ar` (longer len) + rebaseline; **wire
  `fixture-golden-gate.sh` into `.githooks/pre-commit`** ahead of the
  `coherence-gate.sh` call (it is standalone today).
- **P4 — capture + commit** the per-arch-family golden set across the fleet
  (gfx1100 / gfx1201 / gfx1010 — gfx1151 already recorded).

## Deferred (not in this plan)

- **Legible memorized preamble** (§293): a trained 1-line human-readable
  preamble so a CI failure is instantly interpretable. Needs the hipfire
  finetune tool (its "first customer"). The core tripwire needs no training —
  seeded random + greedy + deterministic kernels is sufficient.
- **CPU reference tier.** A deterministic CPU forward (via `hipfire-cpu`) would
  be fully GPU-free and could run in `no-gpu-ci.sh` on every commit — but it
  exercises *different* code than the HIP kernels the HOTSPOT gate exists to
  guard, so it covers arch/loader/plumbing regressions only, not kernel
  numerics. Worth a separate always-on no-GPU tripwire later; it does not
  replace the GPU front tier.

## Open questions / risks

- **GPU still required for the front tier.** "Tiny" buys *speed* (seconds vs
  minutes) and *mechanical sensitivity*, not GPU-avoidance — the golden forward
  must run on the real kernels to cover them. So it still takes the GPU lock,
  briefly. (§325's "CPU/no-GPU-friendly" refers to fixture *generation* +
  finetune memorization, not the golden forward.)
- **Quantizer changes rebaseline fixtures.** Because fixtures flow through the
  real quantize path, a quantizer change shifts the golden — correct (the
  quantizer is in the pipeline) but means quant work carries a deliberate
  fixture-rebaseline step.
- **Per-arch-family golden maintenance** across the validation fleet.
