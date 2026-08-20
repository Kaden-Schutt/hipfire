# Qwen3.8-27B — endpoint-anchored MQ2 Lloyd rescue screen

- **Date:** 2026-08-20
- **Lifecycle:** `historical` — fixture-bound measured evidence under the exact
  method below. **Not** a current default, automatic baseline, product claim,
  or admission decision. Newest file != current baseline.
- **Disposition:** **rejected at the short KLD kill gate.** None of the three
  qt19 arms cleared the predeclared `KLD <= 1.0` threshold on both fixtures.
  No arm was promoted to 24-chunk KLD, AR throughput, or decoded-output
  validation. Endpoint anchoring alone was substantially worse than the plain
  Lloyd control. AWQ alpha 0.55 rescued most of that loss and was the best arm,
  but still failed both short fixtures (`1.114594` WT2, `1.789142` v6sel).
- **Authority:** hiptrx artifacts and `.kldseq` outputs under
  `/home/kaden/qcal/mq2-lloyd-rescue/`.

---

## 1 · Question and codec contract

Can the existing **qt19 `MQ2G256Lloyd` 72 B/G256 layout** be made viable by
keeping the transformed block endpoints exact while Lloyd-updating only the
interior two codepoints?

The experiment did not add a dtype or runtime layout. Every arm uses the same
qt19 payload:

- FWHT-rotated G256 block;
- four fp16 codepoints in the first 8 bytes;
- 256 packed 2-bit indices in the remaining 64 bytes;
- total **72 B/G256 = 2.25 bpw** for tensors encoded as qt19.

Recipes:

| arm | codepoint optimization | AWQ |
|---|---|---|
| `plain` | unconstrained four-centroid Lloyd | none |
| `anchored` | `cp0 = block min`, `cp3 = block max`; Lloyd updates `cp1`, `cp2` only | none |
| `anchored-awq` | same endpoint-anchored Lloyd recipe | imatrix, alpha `0.55` |

The anchored arms round all four codepoints to fp16 before their final
nearest-codepoint assignment and canonicalize degenerate zero blocks to
positive fp16 zero. Plain Lloyd retains the assignments from its f32 Lloyd
loop and rounds its codebook only while writing the header. The anchored
recipe is exposed by `hipfire-quantize --format mq2lloyd-anchored` (aliases
accepted) and,
like plain MQ2 Lloyd, requires `--allow-mq2-lloyd` and rejects MoE models.
Implementation commit: `2b6b4e5b4775c91e8615f4760261d378f03fca21` on
`quant/mq2-lloyd-rescue`.

---

## 2 · Fixture

| field | value |
|---|---|
| Host | **hiptrx** |
| GPUs | **4x gfx1201**, 34.2 GB VRAM each |
| ROCm / HIP | **7.14** (scorer banner) |
| Model | dense **Qwen3.8-27B**, arch id 5 / qwen3_5, 26,895,998,464 quantized params |
| Parent | `/home/kaden/qcal/parents/qwen3.8-27b` |
| Checkout | `/home/kaden/hipfire-quantcal` at `2b6b4e5b4775c91e8615f4760261d378f03fca21` |
| Imatrix | `/home/kaden/qcal/imatrix/Qwen3.8-27B-imatrix.gguf`, 13,642,688 B, md5 `10f0d067a9cf8d989120a04dfc3d0c90` |
| Quantizer | `target/release/hipfire-quantize`, 3,520,552 B, md5 `83d937d9bd9625346a339aeab6c420e8` |
| Scorer | `target/release/examples/eval_hipfire`, 14,152,176 B, md5 `c104ee5893c3e4eb82190c9b075be3e3` |

All three model artifacts use product tier `base`, retaining Q8 embed and
lm-head tensors. AWQ differs only in the third arm.

### Artifacts

| arm | file | bytes | model bpw | md5 |
|---|---|---:|---:|---|
| `plain` | `qwen3.8-27b.mq2lloyd.base.hfq` | 9,568,152,576 | 2.845971 | `66858bfb40cf1a13661cec235e14d131` |
| `anchored` | `qwen3.8-27b.mq2lloyd-anchored.base.hfq` | 9,568,152,576 | 2.845971 | `5b873ece0f6f9638d007ac232b3ed06c` |
| `anchored-awq` | `qwen3.8-27b.mq2lloyd-anchored-awq.base.hfq` | 9,574,976,512 | 2.848000 | `5c30c0142d769dbe71c74b8d763d73f5` |

`model bpw` is whole-file bytes x 8 / 26,895,998,464 params; it includes the
base-tier higher-precision tensors and container metadata, not just qt19
payload bpw.

---

## 3 · Weight-space design screen

The production-faithful design tool evaluated 278,528 post-FWHT G256 blocks:
the first 4096 rows of
`model.language_model.layers.20.mlp.down_proj.weight` (`K=17,408`), widened
from BF16 to f32 and transformed with the production sign seeds 42/1042. Tail
MSE is over coefficients beyond the measured 1% magnitude threshold
(`2.869847e-02`). All three rows below use the same 72 B/G256 ceiling.

| codec arm | overall MSE | tail-1% MSE | mean max-relative error |
|---|---:|---:|---:|
| MQ2V2 affine G128x2 fp16 reference | `3.0432e-05` | `9.4142e-06` | `4.3398e-04` |
| plain per-block Lloyd G256 fp16 | **`1.3688e-05`** | `2.2743e-04` | `4.8941e-01` |
| endpoint-anchored Lloyd G256 fp16 | `2.7966e-05` | `1.9129e-05` | **`1.8949e-04`** |

Anchoring cut plain Lloyd's tail MSE by 11.89x and mean max-relative error by
about 2583x, while giving back most of plain Lloyd's bulk-MSE advantage. It
looked like the tail-safe candidate worth a model-scale kill probe; that
weight-space result did **not** predict the model KLD ordering below.

---

## 4 · Short dual-fixture KLD method

Each arm was scored on the first **3 of 24 chunks** from both references:

- `scoring_mode=prefill`;
- `kv_mode=q8`, `kv_v=q8`;
- `HIPFIRE_GRAPH=0`, `HIPFIRE_NORMALIZE_PROMPT=0`,
  `HIPFIRE_LLOYD_GFX12=1`;
- reference geometry: `n_ctx=2048`, `n_vocab=248320`, `top_k=256`,
  1023 scored tokens/chunk, **3069 scored tokens** per run;
- predeclared promotion rule: an arm must reach **KLD <= 1.0 on both short
  fixtures** before spending a 24-chunk run.

Command shape:

```text
eval_hipfire --model <arm.hfq> --ref <ref.bin> --output <arm-ref.kldseq> \
  --kv-mode q8 --kv-v q8 --scoring-mode prefill --max-chunks 3
```

References:

| fixture | path | bytes | md5 |
|---|---|---:|---|
| WT2 | `/home/kaden/kldrefs/qwen3.8-27b.ref_wt2.bin` | 50,675,552 | `8a21364051d844b97c122e2c895f56d8` |
| v6 selector | `/home/kaden/kldrefs/qwen3.8-27b.ref_v6sel-814d8fd.bin` | 50,675,552 | `4e5618b6a3485c3afcf326cb26f81fc8` |

---

## 5 · Results

| arm | WT2 KLD | WT2 NLL | WT2 PPL | v6sel KLD | v6sel NLL | v6sel PPL | promotion |
|---|---:|---:|---:|---:|---:|---:|---|
| `plain` | `1.295424` | `3.416067` | `30.4494` | `2.293967` | `1.436603` | `4.2064` | **kill** |
| `anchored` | `5.230523` | `7.438779` | `1700.6726` | `6.132794` | `6.447534` | `631.1442` | **kill** |
| `anchored-awq` | **`1.114594`** | **`3.286560`** | **`26.7507`** | **`1.789142`** | `1.770198` | `5.8720` | **kill** |

The anchored+AWQ arm is the best of the three on both KLD fixtures: relative
to plain Lloyd it lowers WT2 KLD by `0.180830` (13.96%) and v6sel KLD by
`0.504825` (22.01%). It nevertheless misses the promotion threshold by
`0.114594` on WT2 and `0.789142` on v6sel.

Endpoint anchoring without AWQ is not a rescue: despite its much safer
weight-space tail metrics, it increases KLD over plain Lloyd by `3.935099`
on WT2 and `3.838827` on v6sel. At this 2-bit operating point, codec MSE and
extrema preservation are insufficient admission proxies.

### Output digests

| output | md5 |
|---|---|
| `plain-wt2-short.kldseq` | `7dc1ea9e134b933845c1ca37a2f555b0` |
| `plain-v6sel-short.kldseq` | `d57cecb1349cb6f7ea5d3692b6b15a77` |
| `anchored-wt2-short.kldseq` | `7e025748caa5e6e62f15e416796c2a2e` |
| `anchored-v6sel-short.kldseq` | `b01ccac42360b6d0a1469ac84fc6ffb1` |
| `anchored-awq-wt2-short.kldseq` | `ab15e1322200c0c9e751ff4216996d46` |
| `anchored-awq-v6sel-short.kldseq` | `fbbb9761589fadd10edaedb64ece94dd` |

---

## 6 · Verdict and stopping boundary

1. **Reject all three recipes for Qwen3.8-27B product admission.** Every arm
   fails the short dual-fixture KLD gate.
2. **Do not spend full KLD or runtime validation on this matrix.** By the
   predeclared kill protocol there are no viable survivors, so no 24-chunk
   KLD, AR benchmark, or serve/decode result exists. This record makes no
   throughput or coherence claim for these artifacts.
3. **AWQ is the only effective rescue lever measured here, but alpha 0.55 is
   insufficient.** Endpoint anchoring by itself is actively harmful at model
   scale even though it repairs plain Lloyd's transformed-tail error.
4. The endpoint-anchored encoder remains a research recipe behind the existing
   explicit MQ2-Lloyd opt-in. It reuses qt19 exactly; no runtime/kernel format
   fork was introduced.
