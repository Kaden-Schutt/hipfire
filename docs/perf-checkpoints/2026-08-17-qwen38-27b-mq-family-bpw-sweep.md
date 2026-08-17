# Qwen3.8 27B — full MQ family bpw sweep, and which formats actually have a gfx1201 decode kernel

**Lifecycle:** `historical`. Fixture-bound measured evidence. It is **not** a
current default, an automatic baseline, a product claim, or an admission
decision.

**Disposition:** Thirteen MQ arms of Qwen3.8-27B were built on MI300X and scored
on hiptrx (`gfx1201`) against **both** the WikiText-2 prose tripwire and the v6
conversation selector. Four results matter:

1. **`mq3lloyd` beats `mq3` by 30% on KLD for 6% more bytes** (0.128106 vs
   0.182275 at 13.38 vs 12.62 GB). The Lloyd-Max codebook is a real win at 3 bits.
2. **`mq6` is near-lossless**: WT2 KLD 0.004560 and PPL 6.2649 against an oracle
   PPL of 6.2385 — 0.4% above the teacher.
3. **Only 4 of 13 arms have a production-speed decode kernel on gfx1201.** Three
   more load and run correctly but at 8–16 tok/s against a healthy 163, and six
   do not score at all. Format availability and *deployability* are different
   questions, and this sweep is the first thing in the campaign to separate them.
4. **PPL is not a usable fidelity metric on this fixture** — several arms score
   *below* the oracle PPL. Rank on KLD.

**Source:** branch `quant/quality`, producer binary `aebf90c90` (md5 prefix
`f0932bac806b`), scorer `eval_hipfire` md5 prefix `89083d01a9d4`.

---

## Recipe held constant across every arm

Only `--format` varies:

```
HIPFIRE_Q8_CLASSES=lm_head,embed  \
HIPFIRE_ALLOW_MQ2=1 HIPFIRE_ALLOW_MQ2_LLOYD=1 HIPFIRE_ALLOW_MQ3=1 \
HIPFIRE_ALLOW_MQ3_LLOYD=1 HIPFIRE_ALLOW_MQ4_LLOYD=1 \
hipfire-quantize --input <parent> --output q38.<arm>.mq4 --format <fmt> \
  --q8-router --imatrix Qwen3.8-27B-imatrix.gguf --awq-alpha 0.55
```

The imatrix is the **consumed off-the-shelf GGUF** (md5 prefix `10f0d067a9cf`,
13,642,688 B), not native calibration — an earlier finding established that
off-the-shelf beats native by 1.98x at identical size and identical decode speed.

Scoring: `--ref <kldref> --max-chunks 24 --kv-mode q8 --kv-v q8
--scoring-mode prefill`, with `HIPFIRE_NORMALIZE_PROMPT=0 HIPFIRE_GRAPH=0
HIPFIRE_LLOYD_GFX12=1`.

**Harness validation:** the freshly-built `mq4` arm (md5 `da5b877ea9a8`) scored
`KLD 0.043776 / NLL 1.857666 / PPL 6.4088`, reproducing the previously-recorded
`q8head_a55` arm **to every digit** from a different binary and a different build
run. That cross-validates the recipe, the binary and the scorer at once.

---

## Build results — all 13 arms produced artifacts

| arm | format | bytes | GB | eff. bpw | nominal bpw | build s |
|---|---|---|---|---|---|---|
| `mq1` | mq1 **(new)** | 5,810,937,856 | 5.81 | 1.728 | 1.0156 | 90 |
| `mq2gl` | mq2gl **(new)** | 8,997,436,416 | 9.00 | 2.676 | 2.0625 | 99 |
| `mq2` | mq2 | 9,574,976,512 | 9.57 | 2.848 | 2.0 | 232 |
| `mq2lloyd` | mq2lloyd | 9,574,976,512 | 9.57 | 2.848 | 2.0 | 356 |
| `mq3gl` | mq3gl **(new)** | 12,041,255,936 | 12.04 | 3.582 | 3.0625 | 111 |
| `mq3` | mq3 | 12,618,796,032 | 12.62 | 3.753 | 3.25 | — |
| `mq3lloyd` | mq3lloyd | 13,379,750,912 | 13.38 | 3.980 | 3.5 | 520 |
| `mq35gl` | mq3.5gl **(new)** | 13,563,165,696 | 13.56 | 4.034 | 3.5625 | 347 |
| `mq4` | mq4 | 15,662,615,552 | 15.66 | 4.659 | 4.25 | 228 |
| `mq4lloyd` | mq4lloyd | 17,938,656,256 | 17.94 | 5.336 | 5.0 | 352 |
| `mq5` | mq5 | 18,706,435,072 | 18.71 | 5.564 | 5.25 | 239 |
| `mq6` | mq6 | 21,743,430,656 | 21.74 | 6.467 | 6.25 | 194 |
| `mq8` | mq8 | 27,190,823,936 | 27.19 | 8.088 | 8.0 | 220 |

Effective bpw exceeds nominal because Q8 `lm_head`+`embed` is a **fixed absolute
cost** held constant across arms. Its share therefore falls monotonically, from
**+0.713 bpw at mq1 to −0.162 at mq8**: at the low-bpw end more than a third of a
bit per weight is being spent on the protected head rather than the body. That is
worth knowing before chasing sub-2-bit bodies.

Four formats in this table — `mq1`, `mq2gl`, `mq3gl`, `mq3.5gl` — **did not exist
before this sweep**; see `a4891c2c8` and `aebf90c90`.

---

## Scored arms

| arm | GB | eff bpw | WT2 KLD | WT2 PPL | v6sel KLD | v6sel PPL |
|---|---|---|---|---|---|---|
| `mq3` | 12.62 | 3.753 | 0.182275 | 7.0251 | 1.150402 | 13.4951 |
| `mq3lloyd` | 13.38 | 3.980 | **0.128106** | 7.0101 | **0.966137** | 12.4707 |
| `mq4` | 15.66 | 4.659 | 0.043776 | 6.4088 | 0.587566 | 12.1735 |
| `mq6` | 21.74 | 6.467 | **0.004560** | **6.2649** | **0.232997** | 12.5946 |

Oracle anchors: WT2 PPL 6.2385, v6 selector PPL 12.8813.

KLD is monotone in bpw across all four arms on **both** references, which is the
basic sanity condition this sweep had to satisfy.

### `mq3lloyd` is the interesting rung

| | GB | WT2 KLD | v6sel KLD |
|---|---|---|---|
| `mq3` | 12.62 | 0.182275 | 1.150402 |
| `mq3lloyd` | 13.38 | 0.128106 | 0.966137 |
| delta | **+6.0% bytes** | **−29.7% KLD** | **−16.0% KLD** |

A Lloyd-Max (squared-error-minimising, per-block) codebook buys ~30% of the
divergence on prose and 16% on the deployment distribution for 6% more bytes.
That is a far better exchange rate than any Q8-protection class measured earlier
(the best of those was 0.0124 KLD per GB; this is ~0.036 KLD per GB on WT2).

Note `mq3lloyd` is gated behind `--allow-mq3-lloyd` as research-only. On this
evidence that gate deserves revisiting for the 3-bit case specifically — the
documented quality concern is about MQ2-Lloyd, and MQ2 arms here indeed remain
unusable, but for the wrong reason (no kernel, see below).

### `mq6` is effectively lossless on prose

WT2 KLD 0.004560 with PPL 6.2649 against a 6.2385 oracle. At 21.74 GB it is 39%
larger than `mq4` and buys a 9.6x KLD reduction on prose and 2.5x on the
selector. Whether that is worth 6.1 GB is a product decision, not a measurement.

---

## Kernel availability — the finding that changes what "supported format" means

`--format` accepting a value does **not** mean the artifact is deployable. Three
arms load and produce *correct* output but decode 10–20x too slowly, and six do
not score at all.

| arm | decode on gfx1201 | status |
|---|---|---|
| `mq3`, `mq3lloyd`, `mq4`, `mq6` | ~163 tok/s | **production-speed** |
| `mq2` | **16 tok/s** | correct, ~10x too slow |
| `mq5` | **10 tok/s** | correct, ~16x too slow |
| `mq2lloyd` | **8 tok/s** | correct, ~20x too slow |
| `mq1`, `mq2gl`, `mq3gl`, `mq35gl` | — | encode-only by construction, no kernel exists |
| `mq4lloyd`, `mq8` | — | did not complete scoring; cause not yet established |

The 8–16 tok/s figures are the documented per-token GEMV fallback. They were
initially mistaken here for a missing `HIPFIRE_LLOYD_GFX12=1` flag; setting that
flag fixed `mq3lloyd` (which then scored at full speed) but **not** `mq2`,
`mq2lloyd` or `mq5`, which is what establishes those as genuine kernel gaps
rather than configuration errors.

`mq4lloyd` and `mq8` failing is **unexplained and should not be read as a kernel
gap** — both were cut off by a 420 s scoring guard without emitting a tok/s
figure, so the cause is undetermined. That is an open item, not a result.

---

## PPL is not a fidelity metric on this fixture

Several arms score PPL **below** the oracle:

| arm | v6sel PPL | oracle |
|---|---|---|
| `mq4` | 12.1735 | 12.8813 |
| `mq3lloyd` | 12.4707 | 12.8813 |
| `mq6` | 12.5946 | 12.8813 |

A quantized model cannot be *more faithful* than its teacher, so a
below-oracle PPL means PPL is measuring something other than fidelity here. The
ordering confirms it: `mq6` has a **9.6x better KLD** than `mq4` on WT2 yet a
*worse* selector PPL (12.5946 vs 12.1735).

This is the same KLD/PPL divergence recorded for `attnfull` vs `ssmin`, and it is
now seen a second time on an independent axis. **Rank arms on KLD.** Treat PPL as
a degeneracy tripwire only — it catches collapse (degenerate output reads
PPL ~1.8e6) and nothing finer.

---

## Open items

1. **`mq4lloyd` and `mq8` are unscored** for undetermined reasons. Both built
   cleanly (rc=0, correct byte counts).
2. **The four new formats are unscored end-to-end** and can only be compared by
   encoder-side codec MSE, which is recorded separately in `a4891c2c8` (mq1/mq2gl/
   mq3gl) and `aebf90c90` (mq3.5gl, a measured NO-GO at +61% MSE vs scalar GL_CB4).
   Scoring them requires decode kernels that do not exist.
3. **Decode tok/s was not measured for the scored arms** on the same footing as
   the earlier Q8-protection ladder; the numbers here are scoring throughput, not
   a bench figure.
4. **`--allow-mq3-lloyd` should be reconsidered** given `mq3lloyd`'s result.
