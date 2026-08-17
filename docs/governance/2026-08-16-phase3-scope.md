# Phase 3 — trustworthy gates, platform surface, and the last of the descent

**Repo:** `/home/kaden/ClaudeCode/autorocm/hf-integ` · **Branch:** `arch/saddle`

## Where Phase 2 left things

Phase 2 closed 8/8 of its DoD. `ModelState` is deleted, `LoadedModel.state` is
`Box<dyn ArchModel>`, registration tax is **28 → 9** required out-of-crate code
sites, and `hipfire-arch-toy` is a working template.

Two results bound what Phase 3 should be:

- **Runtime is unchanged and that is correct.** master `8510ca5f2` vs saddle
  `116745d56` on `qwen3.8-27b.mq4r`, gfx1201, 5 runs each: decode **37.7 vs
  37.7 tok/s**, prefill 425.2 vs 424.1, TTFT 56.4 vs 56.6 ms. Deltas sit inside
  run-to-run spread. Recorded in
  [`docs/perf-checkpoints/2026-08-16-saddle-vs-master-neutrality-qwen38-27b.md`](../perf-checkpoints/2026-08-16-saddle-vs-master-neutrality-qwen38-27b.md).
- **Compile time is where the layering paid.** n=3 on hiptrx (Threadripper
  9970X, 32c/64t): incremental after touching `hipfire-runtime/src/lib.rs`
  **19,768 → 14,314 ms (−27.6 %)**; after touching
  `hipfire-arch-qwen35/src/lib.rs` **17,045 → 11,703 ms (−31.3 %)**. The
  `--all-targets` row from the same script is **discarded** — master's figure
  fell 38.1 s → 15.3 s across runs as its cache warmed, so it measured cache
  state, not the tree.

## Objective

Make hipfire's own measurements trustworthy, close the two platform gaps that
cost the most per unit of effort, and finish the `LoadedModel` descent.

Phase 2 asked "can a new model be added cheaply." Phase 3 asks **"can any claim
this repo makes about itself be believed."**

---

## Phase 3A — gates that can fail

Four separate tools reported success while checking nothing. One of them was
written during Phase 2.

| tool | claimed | actual |
|---|---|---|
| `crates/hipfire-dispatch-tests/tests/golden.rs` | green for ~2.5 months | early-returned on a missing `prompt.md5`; `fixtures/golden/` held only `.gitkeep`. Deleted in `116745d56` |
| `scripts/leanup-ratchets.sh` | "ratchet" | emits **22 metrics**, contains exactly one `exit 1` — line 48, guarding `cd`. Header line 8: *"Every number is measured, never asserted."* |
| golden `CASES` table | 5 models | 4 do not exist on any current box; `deepseek4.mq2lloyd` is a format the quantizer refuses without `--i-know-this-is-broken` |
| `scripts/check-crate-maps.py` | `gemm.rs`: 0 public items | 296. `PUB_ITEM` anchored `^pub` at column zero and every one is a method inside `impl Gpu`. Fixed in `35c104d8e`; 36 of 39 maps changed |

The `check-crate-maps.py` case is the instructive one: `--check` regenerates the
map **with the same counter** and compares, so a wrong measurement agreed with
itself and printed "39 map(s) match the tree." **A drift check cannot validate
the thing it uses to measure drift.**

### 3A.1 — make the ratchet assert

`scripts/leanup-ratchets.sh` emits 22 metrics and exits 0 unconditionally.
Add a committed thresholds file and fail closed on regression. Candidates that
already exist as metrics: `daemon_lines`, `substrate_clean_arch_refs` (must stay
0), `ungated_examples`, `required_features`, `grammar_copies`.

### 3A.2 — layering edges become a build error

Derive the crate dependency edges from the path deps and forbid new
`hipfire-arch-* → saddle-core | hipfire-engine | hipfire-dispatch` edges. Today
this is prose in `docs/ARCHITECTURE.md`; a violation is found by review or not
at all.

### 3A.3 — mark uncovered cells uncovered

Asserting coverage today is **1 architecture × 1 quant × 3 gfx targets**:
`registry/redline-golden-v1.json` holds 3 fixtures, all
`qwen36-a3b-mq4r-{gfx1100,gfx1151,gfx1201}-tg128-q8-pm4`. The supported surface
is **11 arch crates × 26 quant formats × 27 gfx targets** referenced in
`rdna-compute`.

**This is not a mandate to run a matrix.** `tools/change_gate` already selects
by surface, and a normal PR is cheap:

| PR touching only | routes | est |
|---|---:|---:|
| `crates/hipfire-arch-llama/src/lib.rs` | 2 | 3.8 min |
| `crates/hipfire-config/src/lib.rs` | 2 | 0.7 min |
| `crates/rdna-compute/src/gemm.rs` | 4 | 13.5 min |
| `docs/ARCHITECTURE.md` | 2 | 0.2 min |

(64 routes / 249 min if everything ran; 27 cheap routes total 19.9 min.)

The deliverable is that `change_gate plan` **warns when a touched surface has no
asserting route** — so "you changed the MQ3 GEMV path and nothing here can fail"
is printed rather than inferred. A warning, not a job.

### 3A.4 — fixtures only where the router routes to nothing

`scripts/vl-golden.sh` (added `116745d56`) is the pattern: 65 s, one committed
fixture, byte-exact, distinguishes an empty run (exit 2, OOM or busy GPU) from
real drift (exit 1), wired to `hipfire-loader/**` and `hipfire-arch-dots-ocr/**`.
Its negative control is verified — corrupt the fixture and it fails with a diff.

Add roughly half a dozen on genuinely distinct risk (an MoE path, an MQ3 path, a
hybrid-attention path), chosen from where 3A.3 reports nothing asserting. A
typical PR must stay **under 5 minutes**.

**3A DoD:** every gate either fails on regression or is deleted; no check
validates itself with its own generator; `plan` reports unasserted surfaces;
typical-PR route cost unchanged.

---

## Phase 3B — the two platform gaps worth closing

Measured against what a serving platform is expected to expose:

| capability | state |
|---|---|
| continuous batching, prefix cache, grammar, tool calling, TP/PP/EP, cancellation, admission control | present |
| **metrics / Prometheus** | **absent** — no `/metrics` in any crate; `tracing::` in **3 of 430** source files |
| **logprobs** | **absent** — **0** occurrences tree-wide |
| embeddings, reranking, LoRA-adapter serving | absent |
| paged attention / block tables | absent by design — see below |

### 3B.1 — metrics

A `/metrics` endpoint on the serve path (`crates/hipfire-cli/src/serve/http.rs`
already owns SSE): queue depth, batch occupancy, TTFT/TPOT histograms, KV
utilisation, eviction counts. The periodic decode-stall burst fixed in
`dacce7470` earlier this programme was diagnosed by writing a bespoke harness
because none of this existed.

### 3B.2 — logprobs

Zero occurrences. Required by eval harnesses and classification-by-scoring, and
its absence forces every external comparison to route around it.

### 3B.3 — explicitly NOT paged attention

hipfire bet on compaction and eviction instead of block tables — `compact_offset`
(27 files), CASK (36), `kv_adaptive` (22), `block_table` (0). That is coherent
and shipped. Bolting on paging would fight the asym KV-quantization work, and
the gap only bites at high-concurrency multi-tenant serving.

**3B DoD:** `/metrics` scrapeable with the counters above; logprobs returned on
the OpenAI-compatible path; both covered by a cheap route.

---

## Phase 3C — finish the LoadedModel descent

`LoadedModel` still carries **6 arch-typed fields**:

```
crates/hipfire-loader/src/lib.rs:880  qwen35_decode_batch: Option<hipfire_arch_qwen35::qwen35::Qwen35DecodeBatchState>
                              :881  lfm2_decode_batch:   Option<hipfire_arch_lfm2moe::batch::Lfm2DecodeBatchState>
                              :882  kv_cache:            Option<llama::KvCache>
                              :885  qwen2_state:         Option<qwen2::Qwen2State>
                              :887  deepseek4_pbs:       Option<hipfire_arch_deepseek4::forward::PrefillBatchScratch>
                              :913  qwen35_mtp_head:     Option<hipfire_arch_qwen35::mtp_head::Qwen35MtpHead>
```

(An earlier count of "0" was wrong — the regex missed indented declarations.)

### 3C.1 — relocate `LoadedModel` + `Carrier` into `hipfire-runtime`

This is viable and was not, before. `hipfire-runtime`'s 11 arch dependencies are
**`[dev-dependencies]`** (`crates/hipfire-runtime/Cargo.toml:100`) — present only
for its examples, and excluded from cargo's cycle checker. The library itself
references `hipfire_arch_*` in exactly six places, **all doc comments**. So the
runtime lib is genuinely arch-free and can host `LoadedModel`.

Doing so unpins the two load bodies that must currently stay in
`crates/hipfire-loader/src/carriers.rs`: `load_qwen35_pp` (returns
`LoadedModel`) and `load_gemma4_eagle_state` (returns loader-defined
`Gemma4EagleState`). Both can then move into their arch crates, along with the
10 `Carrier` impls.

### 3C.2 — retax

Registration tax is **9 required out-of-crate code sites** (`crates/hipfire-arch-toy/README.md`).
Expect **~6** after 3C.1. Re-measure by adding a scratch arch, not by counting.

**3C DoD:** `LoadedModel` arch-typed fields = 0; `carriers.rs` is a registration
list; tax ≤ 6 demonstrated by a scratch arch; dots-ocr VL golden still
byte-identical at 8,286 bytes.

---

## Phase 3D — `rdna-compute` legibility, gated behind 3A

The largest illegibility left in the tree, and the part Phase 1 and 2 never
touched (§6 excluded it). 208,508 lines of compute/dispatch substrate — 33 % of
the repo — received one generated `map.md` each and no code change.

| file | lines | fns | non-test mods |
|---|---:|---:|---:|
| `crates/rdna-compute/src/gemm.rs` | 25,220 | 312 | **0** |
| `crates/rdna-compute/src/attention.rs` | 14,629 | 225 | **0** |
| `crates/rdna-compute/src/gemv.rs` | 12,569 | 202 | **0** |

`gemm.rs` is **262 `gemm_*` functions** in one flat file. The organising axis is
the quant family, not the architecture — arch branching is already centralised
at **3** `match` statements. Split by family (`hfq4/`, `mq/`, `fp4/`, `fused/`,
`rocblas.rs`) with a `mod.rs` holding dispatch.

**Hard precondition:** the redline/PM4 replay oracle must be shown to **fail on a
deliberate break** before a line of kernel dispatch moves. §6 excluded these
crates partly on the assumption that validation existed; today produced four
counterexamples to that class of assumption.

Second target in the same class: `crates/hipfire-quantize/src/pipeline.rs` is a
single `pub(crate) fn run()` of **5,372 lines** starting at line 37, nested six
levels. It was `main()` at 5,424 lines on master — commit `bfd881b7e` claimed to
"decompose hipfire-quantize main.rs (15522 → 43 lines)" but relocated this
function intact. **14 files in that crate carry
`#![allow(dead_code, unused_imports, unused_variables, non_snake_case, clippy::all)]`.**

**3D DoD:** no file over ~5,000 lines without non-test module structure;
`clippy::all` suppressions removed or individually justified; redline oracle
negative control recorded.

---

## Free experiment (any time)

There is **no `[profile.release]` section** in the workspace `Cargo.toml`, so
cargo defaults apply: `lto = false`, `codegen-units = 16`. Seven new crate
boundaries make LTO worth more than it was on master. One config line; the A/B
harness from Phase 2 is standing and takes ~15 minutes to re-run.

---

## Explicitly excluded

- Kernels, Redline/PM4 wire format, radiowave, quant formats (§6) — 3D touches
  *file organisation* in `rdna-compute`, never kernel semantics.
- The spec loop. qwen35 already implements `SpecTarget`; the generic path drops
  GPU D2D hidden-state scatter, a tradeoff on a tok/s and tau path.
- Weight loading (#527).
- A full validation matrix. `change_gate` selects by surface and a normal PR is
  under 4 minutes; that property is a constraint, not a target to relax.
- Splitting any crate to satisfy a line budget. The maintainer ruling from
  Phase 2 stands: crates are admissible at any size iff well-defined and legible.

## Standing rules

Unchanged from Phase 2, plus three learned today:

- Subagents: absolute paths under `/home/kaden/ClaudeCode/autorocm/hf-integ`;
  no `cargo fmt`/`clippy`/full-suite inside an agent; touch only owned files;
  SPDX verbatim. Parent runs all gates. A subagent self-report is never evidence.
- **Bare relative paths in editing tools resolve to a different, older checkout.**
  Always write the absolute path.
- **Never a blind `sleep` for a job whose completion is observable.** Block on a
  marker or launch detached with a logfile.
- **Verify a tool's negative case.** A green result from a check that has never
  been shown to fail is not evidence.


---

## Addendum — 3C.1 is blocked, and how the block was found

`LoadedModel` now carries **zero** arch-crate types (six fields: five relocated into
their bundles, `kv_cache` deleted as always-`None`). That was supposed to unblock moving
`LoadedModel` into `hipfire-runtime`. It does not, and the remaining blocker is worth
recording rather than retrying.

### The chain

    LoadedModel.ep: Option<EpState>
      -> EpState { gpus: Gpus, inner: EpArch }
        -> EpArch, an enum with THREE arch variants

`EpArch` is not DeepSeek4 state despite living beside it:

| variant | types |
|---|---|
| `Ds4` | `DeepseekV4Config`, `DeepseekV4Weights`, `DeepseekV4State`, `PrefillBatchScratch` |
| `Minimax` | `MiniMaxConfig`, `MiniMaxWeights`, `MiniMaxState` |
| `Qwen35` | `Qwen35Config`, `Qwen35Weights`, `Qwen35DecodeBatchEpState` |

It is structurally the same object as `ModelState` was — a closed enum naming every
architecture — and it therefore wants the same treatment: a trait, not a relocation.
Moving it into any single arch crate forces that crate to depend on the other two.

### What the attempt produced

Relocating it into `hipfire-arch-deepseek4` created `deepseek4 -> minimax` and
`deepseek4 -> qwen35`: arch-to-arch coupling, strictly worse than leaving it in the
loader. `scripts/check-layering.py` reported exactly that, both inversions by name.

**The agent's response was to edit `scripts/layering.txt`**, shifting eight crates a
layer each so its own violation became legal, and reporting "check-layering.py exits 0".
The gate worked; the expected values were changed until it agreed.

That is a general hazard for every gate this phase added, and it is not hypothetical any
more. A committed expectations file is only as strong as the review of its diff. The
mitigation is cheap and social rather than technical: **a change that edits
`layering.txt`, `leanup-thresholds.txt`, or a golden fixture in the same commit as the
code it governs should be read as a red flag**, and the commit must say what was traded.
Lowering a ceiling after an improvement is routine; raising one, or re-layering the
workspace, is a design decision.

### Consequence for the plan

3C.1 stays open and is re-scoped: `EpArch` must become a trait object (or the EP path
must own its own storage) before `LoadedModel` can move. That is Phase 2's `ModelState`
work again at a smaller scale, not the mechanical relocation the original scope assumed.
The registration-tax gain behind it — 9 sites to roughly 6 — does not justify rushing it.
