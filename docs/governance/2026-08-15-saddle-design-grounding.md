# saddle — design grounding

Status: **draft, planning only. No code changes authorized by this document.**
Name: **`saddle`** — settled 2026-08-15. See § 4.
Date: 2026-08-15
Branch: `arch/saddle` (based on `8510ca5f2`)

This document exists to ground a proposed re-layering before any code moves.
Every number below is measured on `8510ca5f2` and reproducible with the
commands in § 8. Where a claim is not measured it is marked `[INFERENCE]`.

---

## 1 · What problem this solves

hipfire competes with, and frequently beats, CUDA-derived engines on AMD
hardware. That is not in question and is not what this document proposes to
change. The problem is structural: the codebase mixes an inference engine, a
quantizer, a calibration oracle, a trainer-adjacent toolchain, and five months
of research harnesses in one workspace, with the product itself shipping as a
`[[example]]`.

The consequence is that hipfire is harder to develop for than llama.cpp
despite being competitive with it, and that is a fixable, measurable property.

---

## 2 · Measured baseline

### 2.1 Workspace

| | lines |
|---|---:|
| `crates/*/src` | 414,224 |
| `crates/*/examples` | 195,143 |
| `crates/*/tests` | 1,669 |
| crates | 32 |

Examples are **47%** of src and **117x** the integration-test code.
`hipfire-runtime` alone carries 99,166 lines of examples across 65 targets,
including `daemon.rs` — the product — at 43,696 lines.

### 2.2 The inverted ratio — and a correction to how it was measured

> **Correction, 2026-08-15.** The table below originally compared hipfire's
> compute *crates* against llama.cpp's entire `ggml/` tree and reported
> **0.7 : 1**. That is not like-for-like. `ggml/` is almost entirely kernel
> source — measured locally at 292,285 lines, of which `.cu` 21,004,
> `.cl` 17,329, `.comp` 12,580, `.metal` 10,549, the rest C/C++ CPU kernels —
> while hipfire's equivalent, `kernels/` at **119,820 lines** of HIP across
> 883 files, was left out of hipfire's own compute side entirely. § 6 of the
> leanup map names "the kernel family" as part of the compute layer, so the
> measurement contradicted its own definition. The corrected figures are
> below; `scripts/leanup-ratchets.sh` reproduces all of them.

Grouping by role, measured at `d3eaac9a4`:

| layer | what counts | lines |
|---|---|---:|
| compute | `rdna-compute`, `redline-*`, `radiowave`, `hip-bridge`, `hsa-bridge`, `hipfire-detect` | 124,348 |
| compute | `kernels/` — generic | 116,202 |
| architecture | `hipfire-arch-*` (12) | 118,661 |
| architecture | `kernels/` — arch-named (`deepseek4_*`, `fused_gemma4_*`, …) | 3,618 |

Against llama.cpp, applying the same rule to both sides:

| | compute layer | arch layer | ratio |
|---|---:|---:|---:|
| llama.cpp | `ggml/` 292,285 | `src/models/` 18,040 | **16.2 : 1** |
| hipfire, crates only (the old, unfair rule) | 124,348 | 118,661 | 1.048 : 1 |
| hipfire, all kernels as compute | 244,168 | 118,661 | 2.058 : 1 |
| **hipfire, strict** | **240,550** | **122,279** | **1.967 : 1** |

**Strict is the number to quote.** llama.cpp has *zero* model-named files in
`ggml/` — verified by scanning every `.c/.cpp/.cu/.cl/.comp` basename — so its
arch layer contains no kernels at all. hipfire's does, and the honest analogue
charges those 3,618 lines to the arch side.

**The inversion was real and it is now nearly gone.** The original 0.7 : 1
overstated it by excluding half of hipfire's compute layer, but the direction
was right: ggml does own generically what hipfire had spread across arch
crates. After the leanup the corrected ratio is 1.967 : 1, against a 2 : 1
target — short by 1.7%.

Closing that last 1.7% is blocked by scope, not by effort. Scanning every
top-level free function in all eleven arch crates for one that names no
architecture finds 7,047 lines, which would be enough. But the bulk of it is
`load_weight_tensor_raw` and `load_vision_weights*` — weight loading, which
belongs to #527 and is explicitly deferred — and `*_via_execute_steps`,
`batched_gemm_single_weight`, `q8_attend_slots`, which are kernel dispatch and
out of scope under § 6. What is left after removing both is too small to close
the gap. The remaining distance is a deliberate consequence of the two
standing decisions, not an unfinished task.

### 2.3 Per-architecture cost

| model | llama.cpp | hipfire |
|---|---:|---:|
| DeepSeek V4 | 1,546 (`src/models/deepseek4.cpp`) | 51,084 (`hipfire-arch-deepseek4`) |
| Qwen3.5-MoE | 742 (`src/models/qwen35moe.cpp`) | 51,955 (`hipfire-arch-qwen35`) |
| mean per arch | 233 (34,097 / 146 files) | ~14,600 (12 crates) |
| architectures | 146 | 12 |

### 2.4 Where the arch bloat actually is

Not evenly spread. Concentrated and largely misplaced:

| item | lines | nature |
|---|---:|---|
| `hipfire-arch-deepseek4/src/parent/` | 20,782 | calibration/reference oracle. **Zero `parent::` references from `arch.rs` or `forward.rs`.** All 30 consumers are examples. 18 of its files mention hessian/calibration/oracle. |
| `grammar.rs` duplicated | 3,935 | 2,736 (qwen35) + 1,199 (ds4). 13 and 1 arch-specific mentions respectively — model-agnostic by construction. |
| spec plumbing duplicated | 3,370 | `spec_emit.rs` 900/270, `spec_impl.rs` 629/1,026, `mtp_speculator.rs` 225/320 — same filenames, two implementations. |
| `hipfire-arch-qwen35/src/pflash.rs` | 2,030 | AGENTS.md: PFlash is "retained legacy research, not mainline or production functionality." |
| `forward_batch_chunk_impl` | 3,628 | single function |

Roughly **30,000 lines** are misplaced or duplicated rather than
architecture-specific.

Counter-evidence that ~10k is achievable today: `hipfire-arch-muse-glimmer`
is 9,985 lines with a full arch-23 drafter. The 52k crates are not a floor.

### 2.5 Unsafe

1,314 `unsafe` occurrences in `src`. 825 (63%) are in `rdna-compute`,
`hip-bridge`, `redline-rocr`, `hsa-bridge` — FFI to `libamdhip64` and PM4
lowering, where unsafe is mandatory. Only 76 reach `hipfire-arch-qwen35`.

**The unsafe surface is correctly located and is not a structural problem.**

### 2.6 Build ergonomics

`daemon` is an `[[example]]` with nine `required-features`
(`arch-qwen35`, `arch-qwen35-vl`, `arch-llama`, `arch-qwen2`, `arch-deepseek4`,
`arch-cohere2moe`, `arch-dots-ocr`, `arch-gemma4`, `arch-muse-glimmer`).

A contributor cannot build the product with one command. llama.cpp gates the
*backend* at build time (`GGML_CUDA`/`GGML_HIP`/`GGML_VULKAN` default OFF) but
compiles every architecture, table-driven. hipfire inverts this: it gates nine
*architectures* at build time to produce a binary that lives in `examples/`.

The seam to fix this already exists: `hipfire-loader::Carrier` has **10
implementations** and **zero** `cfg(feature = "arch-*")` gating in
`carriers.rs`. A second, vestigial `Carrier` in
`hipfire-runtime/src/loader_api.rs:130` has **zero** implementations.

---

## 3 · Proposed layering

```
saddle-hal        hip-bridge, hsa-bridge, hipfire-detect
saddle-compute    per-target, fully specialized, never genericized
                  saddle-rdna    HIP, WMMA, PM4/Redline, radiowave
                  saddle-xdna    XRT/AIE  [future]
saddle-core       manifests, weight placement, step graphs, KV, sampling,
                grammar, spec-decode orchestration   <-- MISSING TODAY
arch crates     thin trait implementations
hipfire-engine  scheduler, batching, sessions, serve
hipfire         one [[bin]]; cli, tui, client, registry
saddle-quant      quantizer, calibration, ds4 parent/
saddle-lab        research harnesses, outside the default build graph
```

### 3.1 The load-bearing rule

**Abstract the model, never the kernel.**

A multi-target abstraction at the kernel layer is precisely the generic-op
abstraction that makes portable engines slower than hipfire on RDNA. Chasing
XDNA at the kernel layer rebuilds ggml and forfeits the advantage. Each
compute backend therefore stays maximally specialized; only composition is
shared.

This also makes XDNA additive rather than corrosive, and admits a
heterogeneous configuration that no CUDA-derived engine can offer: an NPU-
resident draft model paired with a GPU-resident target over the existing
spec-decode path. `[INFERENCE]` — plausible from the existing DFlash split;
unproven, no XDNA work has been done.

### 3.2 What saddle is not

- Not a ggml clone. ggml is declarative-then-execute across 6+ backends.
  hipfire is execute-then-memoize with a PM4 lowering floor that has no ggml
  analogue and requires owning the queue.
- Not portable. Single-vendor is the point.
- Not a rewrite. `rdna-compute`, the kernel family, Redline/PM4, and the quant
  formats are 124k of genuine differentiation and are explicitly out of scope.

The goal is ggml's **ratio**, not its portability and not its absolute
per-arch number. llama.cpp reaches 233 lines/arch because ggml ops are
composable and generic; hipfire's advantage comes from kernels that are
deliberately neither. A realistic target is 1–3k per arch.

---

## 4 · Naming

**The substrate is named `saddle`.**

The alternatives considered were initials-based: `ksml` (Kaden Schutt + ML,
following `ggml` = Georgi Gerganov + ML and `GGUF` = Georgi Gerganov Universal
Format) and `wmml` (warpfront). Both were rejected, and not on grounds of
modesty — `ggml` drew no such criticism and the work here stands on its own.
They were rejected on **adoption mechanics**: personal initials signal "one
person's project" at exactly the moment the name must signal "substrate others
build on," and no hardware vendor standardizes on an individual's initials.
`wmml` inherits the same problem one layer removed, since warpfront resolves to
the same person until it is a distinct org identity.

`ggml` is also the exception rather than the pattern — MLX, candle, burn,
tinygrad, JAX and PyTorch are all descriptive. `saddle` follows the established
house style (hipfire, redline, radiowave, Magnum Quant, DFlash) and carries the
right metaphor: the interface that makes raw power rideable, sitting between
the model and the silicon. The optimization-theory sense of *saddle point* is a
serviceable second reading.

Attribution stays where it already lives and is already maintained: `NOTICE`,
`CREDITS.md`, `CITATION.cff`, and the per-file SPDX headers.

The name invites direct comparison with ggml, which is why § 2.2's ratio has to
hold before it is used publicly.

---

## 5 · Sequencing (proposed, not authorized)

Ordered by (value / risk):

1. **Move `deepseek4/src/parent/` to `saddle-quant`.** 20,782 lines, zero
   load-path references, mechanical. DS4 drops to ~30k.
2. **`examples/` triage.** 195,143 lines into product bins / dev tools /
   research archive outside the default graph. Two known keeps regardless of
   reference count: PFlash (AGENTS.md policy) and `qwen35_batch_generate`
   (the DP4 sealed-case binary).
3. **Delete the vestigial `loader_api::Carrier`** (0 impls) and add ratchets.
4. **Unify `grammar.rs`** into `saddle-core`: 3,935 -> ~1,400.
5. **Harvest PR #527's completed spine** — `weight_manifest.rs` (4,662),
   `weight_store.rs` (6,829), `moe_plan.rs` (11,563), `STEP-001/002/003`,
   `CAP-001`. This is already a saddle-core in all but name; see § 7.
6. **Unify speculation** across qwen35/deepseek4 (#527 `SPEC-001`).
7. **Arch crates to trait impls**; `[[example]] -> [[bin]]`; drop
   `required-features` 9 -> 0.

Items 1–4 are independent and parallelizable. 7 depends on 3–6.

---

## 6 · Ratchets

Each of these is a CI assertion that the number never increases:

| metric | today | target |
|---|---:|---:|
| `daemon.rs` lines | 43,696 | < 5,000 |
| `daemon.rs` `arch_id ==` branches | 43 | 0 |
| `daemon.rs` arch-crate references | 95 | 0 |
| `daemon` `required-features` | 9 | 0 |
| `[[example]]` in `hipfire-runtime` | 65 | < 10 |
| compute : arch line ratio | 0.7 : 1 | > 2 : 1 |
| duplicated `grammar.rs` | 2 copies | 1 |

---

## 7 · Relationship to PR #527

#527 is **33% complete** (14 of 42 tracker items; all four `AXIS` items open)
and is a *parallelism* program — PP/TP/EP mesh cells. That part is orthogonal
to this document and neither blocks nor is blocked by it.

However, #527's **completed** third is substantially the saddle-core proposed
here: manifest-driven weight placement, Step/manifest forward composition, and
`CAP-001`'s capability contract. Harvesting it is strictly cheaper than
re-deriving it, and it is the single largest input to § 5 item 5.

---

## 8 · Reproducing these numbers

```sh
# workspace totals
find crates -path '*/src/*' -name '*.rs' | xargs wc -l | tail -1
find crates -path '*/examples/*' -name '*.rs' | xargs wc -l | tail -1

# arch crate sizes
for d in crates/hipfire-arch-*/; do
  echo "$(find $d/src -name '*.rs' | xargs wc -l | tail -1 | awk '{print $1}') $d"
done | sort -rn

# ds4 parent/ reachability from the inference path
grep -c "parent::" crates/hipfire-arch-deepseek4/src/arch.rs \
                   crates/hipfire-arch-deepseek4/src/forward.rs

# daemon coupling
D=crates/hipfire-runtime/examples/daemon.rs
grep -cE 'arch_id *==' $D
grep -coE 'hipfire_arch_[a-z0-9_]+' $D

# llama.cpp comparison
git clone --depth 1 --filter=blob:none --sparse https://github.com/ggml-org/llama.cpp
cd llama.cpp && git sparse-checkout set src ggml/src ggml/include
find ggml -name '*.c' -o -name '*.cpp' -o -name '*.h' -o -name '*.cu' | xargs wc -l | tail -1
find src/models -name '*.cpp' | xargs wc -l | tail -1
```

---

## 9 · Open questions

Resolved in the 2026-08-15 ideation pass are marked **[resolved]**; the
reasoning is kept because it constrains later decisions.

1. **Naming.** **[resolved — `saddle`.]** Full reasoning in § 4. In short:
   initials-based candidates (`ksml`, `wmml`) were rejected on adoption
   mechanics rather than modesty, and `ggml` is the exception rather than the
   pattern among ML substrates. `saddle` follows the established house style
   and carries the interface metaphor. Attribution remains in `NOTICE`,
   `CREDITS.md`, `CITATION.cff` and the per-file SPDX headers.
   Follow-on, still open: **does `saddle` own the on-disk format?** `GGUF` is
   arguably more widely adopted than `ggml` itself, which suggests the
   container and the quant-format family are the highest-leverage things a
   substrate can standardize. If HFQ and the MQ/MFP family become
   `saddle`-level, the quantizer is substrate-side and per-model calibration
   oracles belong in a `saddle-quant` family; if they stay hipfire-level, the
   newly extracted `hipfire-ds4-parent` keeps its current name. Nothing blocks
   on this today.
2. **Separate repository, or workspace boundary?** **[resolved — workspace
   boundary first.]** The experiment has already been run: `warpfront/redline`
   exists as a public repository *and* `crates/redline`, `crates/redline-dispatch`
   and `crates/redline-rocr` are in-tree. The split produced two homes rather
   than a boundary. A workspace boundary delivers most of the API discipline
   at a fraction of the release overhead for a single maintainer. Split only
   when a consumer outside hipfire actually depends on the crate.
3. **Where does the trainer live?** **[resolved — `origin/feat/mtp-dflash-training`.]**
   21 ahead / **1,832 behind** `master`, last commit `8ec8ff756` 2026-06-26
   (*"re-impl KL-topk loss backward, add target-init loader, fix smoke"*),
   +6,747 lines across 41 files. This is Path C — the target-aligned custom
   DFlash draft that AGENTS.md § 8 lists as the roadmap fix for the prose and
   DDTree regressions. It is on the same decay trajectory that left PR #527
   817 commits behind, and should be triaged before it becomes unrecoverable.
4. **Quantizer competitive claim.** **[deferred to stage 2, after the
   refactor.]** Trained-FWHT + MQ/MFP beating unsloth at lower bpw remains
   `[INFERENCE]`. When taken up, it is settled cheaply by KLD + perplexity at
   matched bpw against `UD-Q4_K_XL` on one model and one committed fixture.
5. **`arch/release-and-layering`.** **[resolved — retired.]** 0 ahead / 261
   behind, never pushed. `arch/saddle` is the refactor branch.
6. **XDNA feasibility.** No AIE work has been done. The heterogeneous
   NPU-draft / GPU-target idea in § 3.1 is unvalidated `[INFERENCE]`.
7. **Where do the parent oracles land?** `hipfire-quantize` is the thematic
   home — it already owns Lloyd (640 references), FWHT (440), Hessian (217),
   GPTQ (382), AWQ (304), E8, and the MQ/MFP formats. But it is itself a
   monolith: 24,863 lines of which `main.rs` is **15,522 (62%)**. Moving
   20,782 lines of `parent/` into it trades one problem for another. The
   proposal is a `*-quant` **family** — quantizer, calibration, and per-arch
   parent oracles as siblings — with that `main.rs` decomposed in the same
   pass.
