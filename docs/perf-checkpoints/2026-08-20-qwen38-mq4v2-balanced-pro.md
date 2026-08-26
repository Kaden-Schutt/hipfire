# Qwen3.8-27B — MQ4V2 balanced Pro allocation

- **Date:** 2026-08-20
- **Lifecycle:** `historical` — fixture-bound measured evidence under the exact
  method below. **Not** a current default, automatic baseline, product claim, or
  admission decision. Newest file != current baseline.
- **Amends by addition only:** the unchanged 15-cell campaign record
  [`2026-08-20-qwen38-mq-v2-product-ladder.md`](2026-08-20-qwen38-mq-v2-product-ladder.md).
- **Disposition:** balanced Pro reallocates the current MQ4V2 Pro precision
  budget from `ssm_out:q8` to `ssm_out:mq6v2 + attn_full:mq6v2`. It is 5.94 MB
  smaller and improves teacher KLD by **5.19% on WT2** and **4.40% on v6sel**.
  In a matched current-binary A/B, AR decode is neutral (+0.3% at 0.1 tok/s
  reporting precision), AR prefill is 2.8% slower, and DFlash decode is 0.9%
  slower at unchanged tau 13.11. This is a stronger **quality-first Pro recipe
  candidate**, not evidence for another SKU or automatic promotion.
- **Authority:** hiptrx campaign root
  `/home/kaden/qcal/mq4v2-balanced-pro/`; raw logs and outputs remain there.

---

## 1 · Question and allocation

Can MQ4V2 Pro spend nearly the same bytes more effectively by replacing its
48 Q8 DeltaNet output projections with MQ6V2 and spending the recovered bytes
on the 64 full-attention projections at MQ6V2?

The prior Pro recipe:

```text
base MQ4V2 + Q8 embed/lm_head + Q8 ssm_out
```

Balanced Pro:

```text
base MQ4V2 + Q8 embed/lm_head + MQ6V2 ssm_out + MQ6V2 attn_full
```

`attn_full` means only tensors whose names contain `self_attn`; it does not
include DeltaNet `linear_attn` inputs. The branch adds that specific selector
while preserving the existing generic `attn` class and Pro's implicit class
set.

Source commit: `b322fd69c` on `quant/mq4v2-balanced-pro`.

Quantization command:

```text
hipfire-quantize \
  --input /home/kaden/qcal/parents/qwen3.8-27b \
  --output /home/kaden/qcal/mq4v2-balanced-pro/qwen3.8-27b.mq4v2.balanced-pro.hfq \
  --format mq4v2 --tier pro \
  --imatrix /home/kaden/qcal/imatrix/Qwen3.8-27B-imatrix.gguf \
  --awq-alpha 0.55 \
  --fixed-tier ssm_out:mq6v2,attn_full:mq6v2
```

Deterministic recipe census from `logs/quantize.log`:

| dtype | tensors |
|---|---:|
| F16 | 689 |
| MQ4G256V2 | 384 |
| MQ6G256V2 | 112 |
| Q8F16 | 50 |

The 112 MQ6 tensors are exactly 48 `linear_attn.out_proj` plus 64
`self_attn.{q,k,v,o}_proj` tensors. The failed first attempt, retained only in
session history and overwritten before scoring, had 48 MQ6 tensors; its census
caught that an explicit class outside ProductTier lifts was not routed. No
measurement below used that incorrect artifact.

---

## 2 · Fixture and identity

| field | value |
|---|---|
| Host | **hiptrx** |
| GPUs | 4x gfx1201, 34.2 GB VRAM each |
| ROCm / HIP | **7.14** |
| Model | dense Qwen3.8-27B, arch id 5 / qwen3_5, 26,895,998,464 quantized params |
| Parent | `/home/kaden/qcal/parents/qwen3.8-27b` |
| Imatrix | `/home/kaden/qcal/imatrix/Qwen3.8-27B-imatrix.gguf`, md5 `10f0d067a9cf8d989120a04dfc3d0c90` |
| Prompt | `benchmarks/prompts/merge_sort_thinking_off.txt`, 140 B, md5 `253c7ac50857fe6d0e10fb0d2c5e35c0` |
| WT2 ref | `/home/kaden/kldrefs/qwen3.8-27b.ref_wt2.bin`, md5 `8a21364051d844b97c122e2c895f56d8` |
| v6sel ref | `/home/kaden/kldrefs/qwen3.8-27b.ref_v6sel-814d8fd.bin`, md5 `4e5618b6a3485c3afcf326cb26f81fc8` |
| MQ4V2 draft | `/home/kaden/qcal/ladder-v2/drafts/qwen3.8-27b-dflash.mq4v2.hfq`, 1,209,603,072 B, md5 `013395583cd04206c8aa68f4d061983d` |

### Model artifacts

| arm | bytes | decimal GB | model bpw | md5 | sha256 |
|---|---:|---:|---:|---|---|
| current Pro control | 16,464,182,272 | 16.464 | 4.897140 | `279c563786499c6651de6a3a57b42b02` | `e6f2ac87042b9e314c323f00bc499a6304cd5624021ae501ce90b12a3a7ea3fa` |
| balanced Pro | 16,458,247,168 | 16.458 | 4.895374 | `d1d6bc2d4fb9ef61deeed19e74a60789` | `52b0f16c602eafd490f23d796b17054ea5bc37e907bc4cedb8eb0856d1db9c7d` |

Balanced is **5,935,104 B smaller** (-0.036%). `model bpw` is whole-file
bytes x 8 / 26,895,998,464 params, including protected tensors and container
metadata.

### Binaries used for every matched run

| binary | bytes | md5 |
|---|---:|---|
| `hipfire-quantize` | 3,493,608 | `3603911afcca56f57e36115aacf08ac2` |
| `examples/eval_hipfire` | 14,152,216 | `c6da073656fe6f92067e3d664ac95722` |
| `hipfire` | 20,453,024 | `4ad7a26a19e094d82b389fcc921fceb5` |
| `daemon` | 30,682,688 | `51402cd06615559255c4e6aeaa5437da` |

---

## 3 · Full dual-fixture KLD

Both arms were rerun with the same scorer binary:

- 24/24 chunks, 24,552 scored tokens;
- `scoring_mode=prefill`, `kv_mode=q8`, `kv_v=q8`;
- `HIPFIRE_NORMALIZE_PROMPT=0`, `HIPFIRE_GRAPH=0`;
- gfx1201, HIP 7.14.

| arm | WT2 KLD | WT2 NLL | WT2 PPL | v6sel KLD | v6sel NLL | v6sel PPL |
|---|---:|---:|---:|---:|---:|---:|
| current Pro | `0.032495` | `1.845373` | `6.3305` | `0.484145` | `2.374798` | `10.7488` |
| **balanced Pro** | **`0.030809`** | **`1.841807`** | **`6.3079`** | **`0.462852`** | `2.503621` | `12.2267` |

Balanced improves KLD by `0.001686` / **5.19%** on WT2 and `0.021293` /
**4.40%** on v6sel.

The v6 result is another KLD/PPL divergence: balanced has lower teacher KLD but
higher realized-token PPL. The v6 teacher oracle is PPL 12.8813, so the control's
lower 10.7488 is not greater fidelity; KLD is the distribution-matching metric.

KLD output md5s:

| output | md5 |
|---|---|
| `balanced-pro.wt2.kldseq` | `546f457428b4e7a5d63f28da1444bb9b` |
| `balanced-pro.v6sel.kldseq` | `97d133ddcb0b29496dae53b87b773655` |
| `control-pro.wt2.kldseq` | `adf41579f8fc9a5ac51211b2b4f34b50` |
| `control-pro.v6sel.kldseq` | `e988dbd281cb2250a5cd87cc433105ee` |

---

## 4 · Matched AR and DFlash performance

Each row is the median of **three fresh-process medians**. Every process used
3 warmups + 5 measured runs, max tokens 128, Q8 KV, `backend=noslots`,
`workload=stateless`, graph/verify-graph off, and the byte-identical committed
prompt. DFlash used the same pinned MQ4V2 draft.

| arm | AR decode | AR prefill | DFlash decode | DFlash prefill | DFlash tau |
|---|---:|---:|---:|---:|---:|
| current Pro | `31.4` | `484.1` | `260.6` | `329.6` | `13.11` |
| balanced Pro | **`31.5`** | `470.7` | `258.3` | `325.1` | `13.11` |
| balanced delta | +0.3% | -2.8% | -0.9% | -1.4% | 0 |

AR decode is a tie at the output's 0.1 tok/s resolution. Balanced pays a small
prefill and DFlash throughput cost despite being byte-neutral; MQ6V2 placement,
not model size alone, determines those paths.

### Fresh-process samples

AR decode:

```text
control:  [31.4, 31.5, 31.4] process medians
balanced: [31.5, 31.5, 31.5] process medians
```

AR prefill:

```text
control:  [484.1, 485.5, 481.2] process medians
balanced: [470.7, 470.7, 474.4] process medians
```

DFlash decode:

```text
control:  [260.6, 260.4, 261.9] process medians
balanced: [258.6, 258.2, 258.3] process medians
```

DFlash prefill:

```text
control:  [329.0, 329.6, 332.3] process medians
balanced: [325.1, 324.9, 325.6] process medians
```

Raw logs: `logs/bench_{ar,dflash}_rep{1,2,3}.log` and
`logs/bench_control_{ar,dflash}_rep{1,2,3}.log`.

---

## 5 · Decoded-output checks

`serve_harness.py` battery, five genres, greedy, thinking off, max tokens 128,
max sequence 2048, Q8 KV:

| route | turns | empty | attractor | coherent text |
|---|---:|---:|---:|---|
| AR | 5 | 0 | 0 | yes |
| DFlash | 5 | 0 | 0 | yes |

One reasoning response reached the 128-token cap on both routes while writing a
correct step-by-step distance calculation; it was a length cap, not degeneration.
The other four responses stopped normally. AR and DFlash produced byte-identical
assistant content for all five prompts. The code, factual, prose, and instruct
responses were directly inspected.

DFlash was confirmed active in the JSON (`dflash=true`) with per-genre tau:
code 10.40, reason 5.68, factual 3.12, prose 1.68, instruct 2.39. Genre-dependent
DFlash speed is expected; the fixed merge-sort benchmark remains tau 13.11.

Evidence:

- `serve-battery.json`
- `serve-battery-dflash.json`
- `logs/serve-battery.log`
- `logs/serve-battery-dflash.log`

---

## 6 · Verdict

1. **The allocation hypothesis works.** At effectively identical size, spreading
   MQ6V2 across `ssm_out` and full attention improves both independent KLD
   fixtures.
2. **Balanced Pro is not a separate Pro XL/XTX rung.** It occupies the current
   Pro byte budget and should be evaluated as a replacement recipe.
3. **Promotion is a product trade, not strict dominance.** AR decode is neutral,
   while balanced gives up 2.8% AR prefill and 0.9% DFlash decode for 4.4-5.2%
   better KLD.
4. **No admission/default follows from this record.** If the quality-first trade
   is accepted, the next action is to update the Pro recipe and rerun the normal
   admission route; do not add a fourth tier solely from this A/B.
