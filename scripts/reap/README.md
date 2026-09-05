# REAP keep-map test harness (generic MoE)

Evaluate a **REAP-pruned** MoE variant (e.g. 0xSero `DeepSeek-V4-Flash-162B`,
256→144 routed experts) **without re-quantizing** — by partial-loading the kept
experts out of an existing full quant (`deepseek-v4-flash.mq2lloyd`).

The keep-map loader is now **arch-generic** (crate `hipfire-reap`), wired into
`deepseek4`, `qwen35`, `lfm2moe`, `minimax`. Activate on any of them with
`HIPFIRE_REAP_PLAN=<dir>` (a `reap_plan.json`). `cohere2moe` is not wired —
no REAP hook exists in `crates/hipfire-arch-cohere2moe`. See `docs/superpowers/specs/2026-06-11-generic-moe-reap-design.md`.

A REAP prune is a *pure expert selection*: the kept experts (and the router rows
for them) are byte-identical to the full model; only the hash-router `tid2eid`
tables (layers 0–2) are remapped. So the loader can keep only `keep[l]` experts
per layer, packed into compact slots `0..kept`, and reproduce the pruned model
exactly from the full quant.

## Loader hook

`HIPFIRE_REAP_PLAN=<dir>` activates the generic keep-map (any wired MoE arch); for
ds4, the legacy `HIPFIRE_DEEPSEEK4_REAP_KEEPMAP=<dir>` still works as an alias
(loads a keep-only plan from `keep_by_layer.json`). Default-off ⇒ the load path is
byte-identical. When active: routed-expert count → kept count; each kept expert is
loaded/packed from `experts.{keep[l][slot]}`; router/gate (+ per-expert bias, ds4
hash `tid2eid`) rows are gathered to `keep[l]` via the shared `gather_rows`. All
exact byte ops — no dequant. REAP and EP-sharding are mutually exclusive.

> ⚠️ The cross-arch **keep-all identity gate** and the ds4 **K144 PPL/KLD smoke**
> below are **GPU-deferred** (the box's GPU is in use). The loader code compiles and
> all `hipfire-reap` CPU unit tests pass; the 10-decimal NLL gate must be run once
> the GPU frees. See the SP1 plan's GPU-embargo note.

## Selective re-quant (overlay) — SP4

Iterate a quant config WITHOUT re-quantizing the whole model. The quantizer
(`hipfire-quantize`, CPU-only) re-quantizes ONLY the tensors named by a reap plan's
`quant_overrides`, reading the **original fp16/bf16 safetensors** (so *up*-quanting
recovers precision — it does not dequant the lossy base), and writes a small
`overlay.hfq` keyed by the same tensor names:

```bash
hipfire-quantize --reap-overlay <plan-dir> --reap-arch <deepseek4|qwen35|lfm2moe|minimax> \
                 --reap-out <plan-dir>/overlay.hfq  <original-safetensors-model-dir>
#   plan-dir holds reap_plan.json with e.g.
#   "quant_overrides":[{"layer":20,"role":"routed_experts","experts":[7,12],"tier":"mq3lloyd"},
#                      {"layer":41,"role":"attention","tier":"q8"}]
```

Supported `tier`s: `q8`, `hfq4`/`hfq6`, `mq4`/`mq6`, and the Lloyd variants
`mq2lloyd`/`mq3lloyd`/`mq4lloyd` (each byte-identical to what `--format <tier>` would
emit for that tensor; verified by unit tests). `--reap-arch` is auto-detected from the
model's arch_id when omitted (minimax must be passed explicitly).

**Consuming the overlay — SP3 DONE:** the load-time splice is implemented. Set
`HIPFIRE_REAP_PLAN=<plan-dir>` with an `overlay.hfq` present in that dir, and
`HfqFile` resolves overlay-then-base automatically — no arch changes required. The
auto-attach is guarded: the overlay's `arch_id` must match the base, and every overlay
tensor name must be a subset of the base's (a foreign tensor rejects the overlay, so a
mismatched plan can't corrupt a load). A per-layer-uniform overlay serves through the
existing dispatch unchanged; an overlay that mixes tiers among experts *within one
layer* still needs **SP2** (GPU bucketed dispatch) to serve.

## Bake (freeze) — SP4b DONE

Once a quant config is tuned, bake it into a standalone `.hfq` that serves with NO env
var. Bake runs a full-model quantize from the original safetensors, applies the plan's
`quant_overrides` per-tensor, and (if the plan has a `keep` map) prunes + renumbers kept
experts to compact slots, gathers router/per-expert-bias rows to the kept set, and
patches the routed-expert count into the output metadata:

```bash
hipfire-quantize --reap-bake <plan-dir> --reap-arch <arch> --format <base-tier> \
                 --reap-out final.hfq  <original-safetensors-model-dir>
# final.hfq loads through the normal path — no HIPFIRE_REAP_PLAN needed.
```

Anchor invariant: a bake with no `quant_overrides` and no `keep` is byte-identical to a
plain `--format <base-tier>` quantize. **Limitations:** (1) deepseek4 **hash layers 0–2**
under a `keep` map require a `tid2eid` remap that bake does not yet do — it hard-errors;
use the **load-time keep-map** (`HIPFIRE_REAP_PLAN`) for pruned ds4 hash layers, or bake
overrides only (no prune) for ds4. (2) The metadata expert-count patch writes one global
field from `keep[0]`, so a *non-uniform-per-layer* keep count isn't fully reflected in
metadata (per-tensor pruning is still per-layer correct). (3) A baked model that mixes
tiers among experts *within one layer* needs **SP2** to serve.

## Workflow

```bash
# 1. Build the keep-map sidecar from the pruned repo's reap_plan.json + safetensors
python3 scripts/reap/build_reap_keepmap.py
#    -> /data/hipfire-models/reap_keepmap_162B_k144/{keep_by_layer.json, tid2eid_l{0,1,2}.i32}

# 2. (optional) Keep-all identity plan to validate the machinery is an exact no-op.
#    Generic (any arch): emits reap_plan.json for HIPFIRE_REAP_PLAN.
python3 scripts/reap/build_keepall_sidecar.py --num-layers <L> --num-experts <E> \
        --arch <name> --out /data/hipfire-models/reap_keepall_<E>
#    ds4 convenience (also emits tid2eid + legacy keep_by_layer.json):
python3 scripts/reap/build_keepall_sidecar.py --ds4
#    Then HIPFIRE_REAP_PLAN=<out> must reproduce that arch's no-plan baseline NLL.

# 3. Build the PPL harness
cargo build --release -p hipfire-arch-deepseek4 --example deepseek4_perplexity

# 4. Run full-vs-pruned PPL + KLD
scripts/reap/run_ppl_kld.sh 1024 8
```

`deepseek4_perplexity <model> <corpus> [--ctx N] [--warmup N] [--offset N] [--dump-logits PATH]`
computes NLL/PPL via `decode_step`; `--dump-logits` writes per-position full-vocab
logits (`DS4PPL01` format) for `kld_compare.py` (stdlib-only — this box's numpy is
broken). Set the keep-map env var to score the pruned variant; unset for the full
baseline.

## Result (0xSero 162B, K144, mq2-lloyd, wikitext2, ctx=1024)

| | full-256 | pruned-144 |
|---|---|---|
| PPL | 7.56 | 17.73 |
| NLL/tok | 2.023 | 2.875 |

KL(full‖pruned) = 1.14 nats, KL(pruned‖full) = 1.64, top-1 agreement = 57.6%.
The K144 checkpoint is heavily degraded (experimental, partial calibration).
