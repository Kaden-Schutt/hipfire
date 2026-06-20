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
- **`crates/hipfire-runtime/examples/fixture_golden.rs`** — golden runner. BUT:
  it is **qwen35-only** (hard-codes `qwen35::{config_from_hfq, load_weights,
  forward_scratch}`) and **teacher-forced** — it feeds a *fixed* random token
  stream and captures per-position argmax + an FNV-1a `logit_hash`. Because the
  inputs are fixed, it **cannot produce an attractor** (an attractor needs the
  model's own output fed back). This is the "single-position prefill golden"
  §293 calls out as insufficient.

## The delta to build

### D1 — Free-running AR golden (the attractor-relevant upgrade)
Generalize `fixture_golden.rs` from teacher-forced to **free-running greedy
decode**: seed a short fixed prompt, then feed `argmax` back as the next token
for a fixed length (§293: 256), growing the KV cache. This is what exercises
the real decode loop + KV growth and what can actually surface an attractor.
- Keep `logit_hash` as the *sensitive* tier and the produced **token sequence**
  as the *robust* tier (both committed).
- Add a `--mode {teacher-forced,ar}` so the existing prefill golden stays
  available.
- **Generalize across archs.** Today it is qwen35-only; dispatch on the
  fixture's `arch_id` so dense (arch 5) and MoE (arch 6) both run. A per-arch
  `forward` shim is fine; the runner just needs argmax + KV growth per arch.

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
- **Front tier:** a script (`tests/tiny-golden.sh`) runs the fixture matrix,
  diffs against committed goldens. Seconds, runs in pre-commit whenever the
  HOTSPOT set is touched (now including renames, post-`ACMR` fix).
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

- **P1 — D1:** free-running AR runner + arch dispatch. Land with the existing
  qwen35 dense fixture; commit its golden. (No gate wiring yet — just the runner
  + one golden + a unit test that it's stable across two runs.)
- **P2 — D2/D3:** dense + MoE presets across the format axis; the MoE
  deterministic-combine pin; regenerate-and-cache script.
- **P3 — D4:** `tests/tiny-golden.sh` + escalation + rebaseline command; wire
  into `.githooks/pre-commit` ahead of the `coherence-gate.sh` call.
- **P4 — capture + commit** the per-arch-family golden set on each dev box in
  the fleet (gfx1100 / gfx1201 / gfx1010 / gfx1151).

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
