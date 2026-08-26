# Qwen3.8-27B — MQ V2 15-cell product ladder (hiptrx)

- **Date:** 2026-08-20
- **Lifecycle:** `historical` — fixture-bound measured evidence under the exact
  method below. **Not** a current default, automatic baseline, product claim, or
  admission decision. Newest file ≠ current baseline.
- **Disposition:** Fifteen MQ V2 cells (`mq{2..6}` × `{xt,base,pro}`) were built
  and scored end-to-end on hiptrx. **MQ2V2 is measured and rejected for product
  admission** (catastrophic KLD; decoded text coherent but degraded/repetitive —
  do not call it shipping). MQ3–MQ6 V2 cells produced coherent decoded output
  with no attractor/empty collapse on the serve battery. **Acceptance is
  unavailable** (`null` / `unavailable_in_native_generate_v1`) for every AR and
  DFlash row — native-generate-v1 never reports it; it is never derived.
- **Authority:** hiptrx qcal campaign
  `/home/kaden/qcal/ladder-v2/{summary.json,summary.md,summary.csv,results.json,state.json,manifest.json}`
  regenerated `2026-08-20T15:16:11.069459+00:00`. Every number below is taken
  from that summary (not abbreviated). Phase logs live under the same tree.

---

## 1 · Fixture

| field | value |
|---|---|
| Host | **hiptrx** |
| GPUs | **4× gfx1201**, 34.2 GB VRAM each (ROCm device nodes `card0`–`card3`) |
| ROCm / HIP | **7.14** (`/opt/rocm` → 7.14.0; scorer banner: `HIP 7.14`) |
| CPU | AMD Ryzen Threadripper 9970X 32-Cores |
| Model | dense **Qwen3.8-27B** (arch id 5 / qwen3_5), **26,895,998,464** params |
| Parent | `/home/kaden/qcal/parents/qwen3.8-27b` |
| DFlash draft parent | `/home/kaden/qcal/parents/qwen3.8-27b-dflash2` |
| Imatrix (base/pro) | `/home/kaden/qcal/imatrix/Qwen3.8-27B-imatrix.gguf` — 13,642,688 B; md5 `10f0d067a9cf8d989120a04dfc3d0c90`; sha256 `aaa933d4b9ce23e1f65c548ad34f16956d8af44a51b5c15bf4f393ba59508cd8` |
| Checkout | `/home/kaden/hipfire-quantcal` |
| Manifest commit | `bfe3cfead5856c5ee85e972b609712b0badbd1b1` |
| Manifest diff md5 | `d41d8cd98f00b204e9800998ecf8427e` (clean tree at snapshot) |
| Manifest sha256 | `b73a5f845f4cc6e90501917601ccc823ff50591339c74e8c423ea00a03a8623b` (from `state.json`) |
| Qcal root | `/home/kaden/qcal/ladder-v2` |

### Codec / qt map (from campaign + qt-register)

| codec | qt | `DType` / register name |
|---|---:|---|
| `mq2v2` | 50 | `MQ2G256V2` |
| `mq3v2` | 49 | `MQ3G256V2` |
| `mq4v2` | 44 | `MQ4G256V2` |
| `mq5v2` | 48 | `MQ5G256V2` |
| `mq6v2` | 47 | `MQ6G256V2` |

### Product tiers

| tier | lifted classes (from quantize logs) | imatrix / AWQ | fixed-tier overrides present |
|---|---|---|---|
| `xt` | `embed` | none | none |
| `base` | `embed`, `lm_head` | imatrix + AWQ α=0.55 | none |
| `pro` | `embed`, `lm_head`, `ssm_out` | imatrix + AWQ α=0.55 | **mq2-pro:** `lm_head:mq6v2,ssm_out:mq6v2`; **mq3-pro:** `ssm_out:mq6v2`; mq4–mq6 pro: none |

Representative CLI (logs under `logs/quantize_<cell>.log`):

```
# xt
hipfire-quantize --input <parent> --output <out> --format mqNv2 --tier xt

# base
hipfire-quantize --input <parent> --output <out> --format mqNv2 --tier base \
  --imatrix /home/kaden/qcal/imatrix/Qwen3.8-27B-imatrix.gguf --awq-alpha 0.55

# pro (mq2 example with fixed-tier)
hipfire-quantize --input <parent> --output <out> --format mq2v2 --tier pro \
  --imatrix /home/kaden/qcal/imatrix/Qwen3.8-27B-imatrix.gguf --awq-alpha 0.55 \
  --fixed-tier lm_head:mq6v2,ssm_out:mq6v2
```

---

## 2 · Method

Counts from `summary.json`: **15 cells**, **30 KLD arms**, **15 AR rows**,
**15 DFlash same-bit + 12 mq4-control = 27** DFlash rows. `state.json` reports
all phases complete with empty `failed` arrays for quantize / kld / bench-ar /
bench-dflash / drafts.

### KLD (quality)

- Binary: `eval_hipfire`
- Refs: WT2 tripwire + v6 conversation selector (24 chunks each)
- `scoring_mode=prefill`, `kv_mode=q8`, `kv_v=q8`, `--max-chunks 24`
- Env: `HIPFIRE_GRAPH=0`, `HIPFIRE_NORMALIZE_PROMPT=0`, `HIPFIRE_LLOYD_GFX12=1`
- Ref geometry (from scorer): `n_ctx=2048`, `n_vocab=248320`, `n_chunk=24`,
  `top_k=256`, scored/chunk=1023, **total_scored=24,552**
- Outputs: `/home/kaden/qcal/ladder-v2/kld/<cell>.{wt2,v6sel}.kldseq` +
  `logs/kld_<cell>_{wt2,v6sel}.log`

### Autoregressive throughput (AR)

- Binary: `hipfire bench`
- **3 fresh processes** per cell; each process: **3 warmups + 5 runs**
- `--max-tokens 128 --backend noslots --workload stateless --json --spec off`
- Prompt file: `benchmarks/prompts/merge_sort_thinking_off.txt` (digests below);
  argv embeds the chat-templated merge_sort request with empty think block
- Reported AR decode/prefill = **median of the three fresh-process medians**
- Logs: `logs/bench_ar_<cell>_rep{1,2,3}.log`

### DFlash (same-bit vs mq4 control)

- Same bench shape as AR, but `--spec dflash`
- Drafts converted from `qwen3.8-27b-dflash2` via `dflash_convert --mqNv2`
  (block_size=8, target_layers `[5,19,33,47,61]`, hidden=5120)
- For non-mq4 cells: measure **same-bit draft** and **mq4v2 control draft**;
  choose higher fresh-process median DFlash decode tok/s
  (tie → smaller draft bytes; both preserved). mq4 cells: controller **is**
  same-bit (`reason: controller is same-bit (mq4)`); no separate control row
- **τ (tau)** recorded from the remote summary; **acceptance = null**
  (`unavailable_in_native_generate_v1`)
- Logs: `logs/bench_dflash_<cell>_rep{1,2,3}.log` (both draft arms share the
  three process logs; rows keyed in `results.json` as
  `bench-dflash:<cell>:<draft_id>`)

### Decoded-output serve battery

Post-ladder serve checks wrote `serve-mq{2..6}.{json,log}` and
`serve-home-mq{2..6}/` using each family's **base** artifact + same-bit draft.
Five prompts per family; flags recorded per response. See §6.

### Binaries (final runtime rerun)

The final AR and both DFlash matrices were rerun after all loader, single-row
lm-head, and MQ5 BT fixes. The runner records these digests on every final row.

| binary | bytes | md5 | sha256 |
|---|---:|---|---|
| `hipfire-quantize` | 3,491,160 | `897a12a03aeef9d22e929edbc303bfc8` | `f60762a58070acc86c8c37f33d768af597bb2f94178857bfa8e86b288c853a4c` |
| `dflash_convert` | 944,976 | `f76b992e39d57c708c5e3d3fa3808e95` | `ebbe90cfcdd28365e507d751fa45477c918317b5af0b88a40a97b01ae7f2bc31` |
| `hipfire` (bench client) | 20,269,448 | `c29b2b7569721bf4a48a7e5b1e41659a` | `7a663b429fa045a8057ac1aecb0c59173cb9872f566764402e8f1739072a8f9b` |
| `daemon` | 30,681,632 | `48d4bfa981ff7d7deb7db625766842fe` | `38a57a5c34c84357fe105a34a1a24d67e04cd968ae5474648b7518a134c69364` |
| `examples/eval_hipfire` | 14,152,176 | `c104ee5893c3e4eb82190c9b075be3e3` | `4674a934032e9179686178717b5307ef3c627cce7284f4b900b57a4a82fde349` |

Artifact-generation and early KLD rows retain their own commit and binary
digests in `results.json`; high-bit KLD was rerun serially after the MQ5 BT
symbol fix. The final AR/DFlash rows use the client/daemon pair above. Raw
per-phase rows, not this convenient final snapshot, remain the reproduction
authority.

### Prompt + KLD refs

| asset | path | bytes | md5 | sha256 |
|---|---|---:|---|---|
| AR/DFlash prompt file | `/home/kaden/hipfire-quantcal/benchmarks/prompts/merge_sort_thinking_off.txt` | 140 | `253c7ac50857fe6d0e10fb0d2c5e35c0` | `d671894964cb957643fcb961151f3d1b407cb5c206766eaed60e9c593e6ed9d0` |
| WT2 ref | `/home/kaden/kldrefs/qwen3.8-27b.ref_wt2.bin` | 50,675,552 | `8a21364051d844b97c122e2c895f56d8` | `8c545178fb43647499c4c7bdd33178c24fd402615c641add9ed51a0c9a43234a` |
| v6sel ref | `/home/kaden/kldrefs/qwen3.8-27b.ref_v6sel-814d8fd.bin` | 50,675,552 | `4e5618b6a3485c3afcf326cb26f81fc8` | `b3d1fafc914f5bfa47ba0f0955672b08996f46802a4b0dce3b554c4e879cab04` |

---

## 3 · Core ladder — KLD / AR / prefill (15 cells)

bpw shown to three decimals as in `summary.md` (full float in artifact table).
KLD columns are **WT2 / v6sel**. AR columns are decode / prefill tok/s medians.
**Acceptance: null** on every AR row.

| cell | codec | tier | bpw | KLD wt2 | KLD v6sel | AR dec | AR pre |
|---|---|---|---:|---:|---:|---:|---:|
| mq2-xt | mq2v2 | xt | 2.551 | 13.239513 | 14.202591 | 49.0 | 545.6 |
| mq2-base | mq2v2 | base | 2.848 | 12.441964 | 13.428047 | 43.8 | 523.8 |
| mq2-pro | mq2v2 | pro | 2.966 | 12.466792 | 13.447430 | 42.7 | 517.0 |
| mq3-xt | mq3v2 | xt | 3.503 | 0.248348 | 1.411523 | 40.7 | 467.8 |
| mq3-base | mq3v2 | base | 3.753 | 0.153658 | 1.032401 | 37.6 | 459.4 |
| mq3-pro | mq3v2 | pro | 3.922 | 0.130314 | 0.924635 | 36.7 | 452.5 |
| mq4-xt | mq4v2 | xt | 4.456 | 0.057449 | 0.771410 | 35.3 | 490.7 |
| mq4-base | mq4v2 | base | 4.659 | 0.039033 | 0.544517 | 33.2 | 479.0 |
| mq4-pro | mq4v2 | pro | 4.897 | 0.032495 | 0.484145 | 31.8 | 473.8 |
| mq5-xt | mq5v2 | xt | 5.408 | 0.015028 | 0.440984 | 29.4 | 346.3 |
| mq5-base | mq5v2 | base | 5.564 | 0.010255 | 0.278077 | 28.2 | 341.8 |
| mq5-pro | mq5v2 | pro | 5.746 | 0.009006 | 0.237993 | 27.4 | 346.1 |
| mq6-xt | mq6v2 | xt | 6.361 | 0.004389 | 0.220915 | 26.0 | 333.9 |
| mq6-base | mq6v2 | base | 6.469 | 0.002771 | 0.152813 | 25.2 | 330.5 |
| mq6-pro | mq6v2 | pro | 6.596 | 0.002208 | 0.136504 | 24.8 | 331.2 |

### NLL / PPL companions (same KLD runs)

| cell | NLL wt2 | PPL wt2 | NLL v6sel | PPL v6sel |
|---|---:|---:|---:|---:|
| mq2-xt | 14.958367 | 3135712.9139 | 14.871415 | 2874572.7175 |
| mq2-base | 14.159995 | 1411261.4996 | 14.102656 | 1332617.9732 |
| mq2-pro | 14.178819 | 1438079.0606 | 14.146544 | 1392406.0073 |
| mq3-xt | 2.012922 | 7.4852 | 1.804972 | 6.0798 |
| mq3-base | 1.942141 | 6.9737 | 2.548312 | 12.7855 |
| mq3-pro | 1.913349 | 6.7757 | 2.460758 | 11.7137 |
| mq4-xt | 1.858859 | 6.4164 | 2.325289 | 10.2296 |
| mq4-base | 1.847483 | 6.3438 | 2.465047 | 11.764 |
| mq4-pro | 1.845373 | 6.3305 | 2.374798 | 10.7488 |
| mq5-xt | 1.833796 | 6.2576 | 2.500757 | 12.1917 |
| mq5-base | 1.83146 | 6.243 | 2.532453 | 12.5843 |
| mq5-pro | 1.829878 | 6.2331 | 2.562193 | 12.9642 |
| mq6-xt | 1.832238 | 6.2479 | 2.529821 | 12.5513 |
| mq6-base | 1.83504 | 6.2654 | 2.563041 | 12.9752 |
| mq6-pro | 1.833082 | 6.2531 | 2.570722 | 13.0753 |

### MQ2V2 rejection (explicit)

MQ2V2 is **fully measured** on this fixture and **rejected for product
admission**:

1. **Catastrophic KLD** on both references (WT2 ≈ 12.4–13.2; v6sel ≈ 13.4–14.2)
   with PPL in the 1e6 range — the collapse tripwire, not a near-lossless rung.
2. **Serve battery:** `attractor=false`, `empty=false`, but content is
   **coherent-yet-degraded** (meta-commentary / restating the prompt rather than
   solving it; repetitive structure). Contrast MQ3–MQ6 base, which emit proper
   code/math/prose under the same prompts.
3. **Do not call MQ2V2 shipping.** Presence in this ladder is measurement, not
   admission.

---

## 4 · DFlash — same-bit vs mq4v2 control

Decode tok/s medians; τ medians; chosen draft. **Acceptance: null** on every
DFlash arm (`unavailable_in_native_generate_v1`). mq4 cells have no separate
control column (control ≡ same-bit).

| cell | same dec | same τ | ctrl dec | ctrl τ | chosen | choice reason |
|---|---:|---:|---:|---:|---|---|
| mq2-xt | 60.1 | 1.82 | 39.5 | 0.87 | **mq2v2** | same-bit 60.1 > control 39.5 |
| mq2-base | 46.0 | 1.28 | 63.3 | 2.15 | **mq4v2** | control 63.3 > same-bit 46.0 |
| mq2-pro | 62.4 | 2.17 | 88.2 | 3.54 | **mq4v2** | control 88.2 > same-bit 62.4 |
| mq3-xt | 237.4 | 11.70 | 264.8 | 13.11 | **mq4v2** | control 264.8 > same-bit 237.4 |
| mq3-base | 225.5 | 11.70 | 250.8 | 13.11 | **mq4v2** | control 250.8 > same-bit 225.5 |
| mq3-pro | 221.2 | 11.70 | 248.1 | 13.11 | **mq4v2** | control 248.1 > same-bit 221.2 |
| mq4-xt | 251.6 | 11.70 | — | — | **mq4v2** | controller is same-bit (mq4) |
| mq4-base | 263.3 | 13.11 | — | — | **mq4v2** | controller is same-bit (mq4) |
| mq4-pro | 258.2 | 13.11 | — | — | **mq4v2** | controller is same-bit (mq4) |
| mq5-xt | 221.3 | 13.11 | 225.2 | 13.11 | **mq4v2** | control 225.2 > same-bit 221.3 |
| mq5-base | 214.5 | 13.11 | 217.4 | 13.11 | **mq4v2** | control 217.4 > same-bit 214.5 |
| mq5-pro | 213.2 | 13.11 | 217.0 | 13.11 | **mq4v2** | control 217.0 > same-bit 213.2 |
| mq6-xt | 207.8 | 13.11 | 211.5 | 13.11 | **mq4v2** | control 211.5 > same-bit 207.8 |
| mq6-base | 202.8 | 13.11 | 206.6 | 13.11 | **mq4v2** | control 206.6 > same-bit 202.8 |
| mq6-pro | 202.1 | 13.11 | 205.9 | 13.11 | **mq4v2** | control 205.9 > same-bit 202.1 |

Reading guide (measurement only): only **mq2-xt** prefers the same-bit draft on
decode tok/s. From mq2-base upward the **mq4v2 control draft wins** every
paired comparison, including all mq3/mq5/mq6 cells (mq5/mq6 wins are narrow).

### DFlash drafts (digests)

| draft_id | path | bytes | md5 | sha256 |
|---|---|---:|---|---|
| mq2v2 | `/home/kaden/qcal/ladder-v2/drafts/qwen3.8-27b-dflash.mq2v2.hfq` | 760,353,792 | `ab82ea03654130d1c05086cd173f5b8e` | `0545d189d9486b84b2e65070cf4920ce762a590b4ed8a17092662b69e004f649` |
| mq3v2 | `/home/kaden/qcal/ladder-v2/drafts/qwen3.8-27b-dflash.mq3v2.hfq` | 984,978,432 | `7535c690f9f1204c2158e8b53caed3da` | `dd335fb2634c371d120077c2fe754643c91b27f32d0c32668278fe4b8f1ca319` |
| mq4v2 | `/home/kaden/qcal/ladder-v2/drafts/qwen3.8-27b-dflash.mq4v2.hfq` | 1,209,603,072 | `013395583cd04206c8aa68f4d061983d` | `d0a74a232a0e2166d889f823e91e0fbf778d21dd9668d7de055cdecb065401bc` |
| mq5v2 | `/home/kaden/qcal/ladder-v2/drafts/qwen3.8-27b-dflash.mq5v2.hfq` | 1,434,227,712 | `2aca90f49189f5618cd94daf7b44a043` | `8a8d3daeaa3788743ef9aedfa1a6cd9961395ecbd88e9b5e2eb981e0506a861f` |
| mq6v2 | `/home/kaden/qcal/ladder-v2/drafts/qwen3.8-27b-dflash.mq6v2.hfq` | 1,658,852,352 | `e1de597512d86b4d25b52606ffb16e16` | `d190ef2faa953252ac40e9706cf2c6763d095f7817c2c7a0466792fd6dfa9015` |

Convert recipe (per draft log):

```
dflash_convert --input /home/kaden/qcal/parents/qwen3.8-27b-dflash2 \
  --output /home/kaden/qcal/ladder-v2/drafts/qwen3.8-27b-dflash.mqNv2.hfq --mqNv2
```

---

## 5 · Artifacts (target weights)

All under `/home/kaden/qcal/ladder-v2/artifacts/`. Commit column is the per-cell
`commit` field from `summary.json` products (uniform
`bfe3cfead5856c5ee85e972b609712b0badbd1b1`).

| cell | path | bytes | bpw (full) | md5 | sha256 | fixed_tier |
|---|---|---:|---:|---|---|---|
| mq2-xt | `qwen3.8-27b.mq2v2.xt.hfq` | 8,574,872,576 | 2.550527384206204 | `12d3cafef619b552f78e9ae557b899ed` | `b86de92dd07e3ac53e239459ef4f4881b798ba1296ce15a0974f59007a8e894e` | — |
| mq2-base | `qwen3.8-27b.mq2v2.base.hfq` | 9,574,976,512 | 2.8480003149363653 | `8693a1305eec5f77eed8db19bd49a94b` | `547bc0ae3dfefaef583c4f8756511f362389e93765f2136fab64691b778ac186` | — |
| mq2-pro | `qwen3.8-27b.mq2v2.pro.hfq` | 9,971,776,512 | 2.966025306804539 | `4823a452391cb3cdc58616b327a74ed2` | `6cc00eddd39bbb06824905276bfa0fd34edeed40b3dd3020b7b7a0412b0fad1b` | `lm_head:mq6v2,ssm_out:mq6v2` |
| mq3-xt | `qwen3.8-27b.mq3v2.xt.hfq` | 11,777,616,896 | 3.5031581108287795 | `80bb9198e6a565fc006b2ae1b7c89eca` | `3e04fc8db80bda557b965ec60ac876cf2500fced7f340624f3fcbeae134af5c5` | — |
| mq3-base | `qwen3.8-27b.mq3v2.base.hfq` | 12,618,796,032 | 3.7533601286868365 | `ad20b3d3a9a7254e7a9c596fc97b411a` | `09c3544690aceca29e1822d79adab6ffcc8fd9e4b58359fe8dfb185ef49811c9` | — |
| mq3-pro | `qwen3.8-27b.mq3v2.pro.hfq` | 13,184,433,152 | 3.9216043738691373 | `b171dd618eecf1fe6aed2c1dc5eef4dc` | `394c50966bf4f68172df8eb34cd7ded8f9d0576c9ef24ea6ba639a88c184f795` | `ssm_out:mq6v2` |
| mq4-xt | `qwen3.8-27b.mq4v2.xt.hfq` | 14,980,361,216 | 4.4557888374513555 | `e45d15bfe0c9a87132697101d17cbed6` | `9f91556f7e0431a077d03756a7102d0154108757289e6e5fe9a2d204c0c9eeb7` | — |
| mq4-base | `qwen3.8-27b.mq4v2.base.hfq` | 15,662,615,552 | 4.658719942437308 | `d1292b4d5bd6046693604201a6ca8074` | `5bb556a6cc84035234995c017c9791aa3951ad1eae4cf8c8172b0eaef399e507` | — |
| mq4-pro | `qwen3.8-27b.mq4v2.pro.hfq` | 16,464,182,272 | 4.897139563429743 | `279c563786499c6651de6a3a57b42b02` | `e6f2ac87042b9e314c323f00bc499a6304cd5624021ae501ce90b12a3a7ea3fa` | — |
| mq5-xt | `qwen3.8-27b.mq5v2.xt.hfq` | 18,183,105,536 | 5.408419564073931 | `e7c17382812f33df90f5138eb7cf9973` | `f4760c159f80d5d3ca237593a191dd553790cba0f89de59827f430b23109b39b` | — |
| mq5-base | `qwen3.8-27b.mq5v2.base.hfq` | 18,706,435,072 | 5.564079756187779 | `cff5d051ce8a2a44a5eba6b9b43d595d` | `c018a0d7510bffbb3788844d5d8f72694464244e16f128b7e34804045324cf25` | — |
| mq5-pro | `qwen3.8-27b.mq5v2.pro.hfq` | 19,319,258,112 | 5.746359076532107 | `50119af7b6974320e574774b663d2f6d` | `7a46204f5ce16b260ebb028359a945c8c02f5abe87c040df063db782a99ee7cd` | — |
| mq6-xt | `qwen3.8-27b.mq6v2.xt.hfq` | 21,385,849,856 | 6.3610502906965065 | `de6eb059a10577b80f0a162e5c89249e` | `9d472ddc5b4e11a1986bfc83c54c0dac979a8e1d7186613dd7c7436f69ec8b2f` | — |
| mq6-base | `qwen3.8-27b.mq6v2.base.hfq` | 21,750,254,592 | 6.46943956993825 | `ae18a5d9e7926474064586387126aa5c` | `b798ea1166fc03a568f6daf8090b20b7d6314af7429951c03d86540385568db8` | — |
| mq6-pro | `qwen3.8-27b.mq6v2.pro.hfq` | 22,174,333,952 | 6.595578589634471 | `aac76dcc771b424d605e813b0b4aa1c1` | `58ac3ee645ede2bad2f6f833db1d4abbfdc1850047fdadef036d35a023bb9401` | — |

---

## 6 · Decoded-output serve battery

Files: `/home/kaden/qcal/ladder-v2/serve-mq{2,3,4,5,6}.{json,log}` (and
`serve-home-mqN/`). Each family served its **base** target with the matching
same-bit DFlash draft; **5 prompts × 64 gen tokens**, `finish=length`.

| family | artifact (base) | draft | n prompts | attractor | empty | atem_leak | disposition |
|---|---|---|---:|---:|---:|---:|---|
| mq2 | `…mq2v2.base.hfq` | mq2v2 | 5 | 0 | 0 | 0 | **coherent but degraded / repetitive** (meta-restatement; not collapse, not shipping quality) |
| mq3 | `…mq3v2.base.hfq` | mq3v2 | 5 | 0 | 0 | 0 | **coherent** — code / math / prose on-prompt |
| mq4 | `…mq4v2.base.hfq` | mq4v2 | 5 | 0 | 0 | 0 | **coherent** |
| mq5 | `…mq5v2.base.hfq` | mq5v2 | 5 | 0 | 0 | 0 | **coherent** |
| mq6 | `…mq6v2.base.hfq` | mq6v2 | 5 | 0 | 0 | 0 | **coherent** |

All 25 responses also set `runaway=true` because generation hit the fixed
64-token length cap (`finish=length`); that flag is **not** an attractor signal
here. Standing rule retained: numbers never prove coherence — the battery is the
text check.

---

## 7 · Bring-up / reproducibility footnotes

Recorded only where they affect re-running this campaign:

1. **Retained-Redline is not admitted for these V2 artifacts.** A harness
   attempt loaded mq2v2 but produced no `redline_capture` row (the harness then
   failed closed); parallel mq3/mq5/mq6 attempts collided on the daemon's
   single-instance lock. No PM4/AQL parity claim or JSON report is carried from
   those attempts. Kernel evidence for this checkpoint is the real HIP
   plain-GEMV/B=16 parity oracle plus the user-facing KLD/bench/serve routes.
2. **Kernel cache isolation.** KLD/bench used per-GPU caches under
   `kernel_cache_gpu{0..3}/gfx1201` (scorer banner). Serve logs show occasional
   one-shot recompiles when a pre-compiled blob lacked a hash file.
3. **Acceptance hole.** Every AR and DFlash row carries
   `acceptance: null` / `acceptance_reason: "unavailable_in_native_generate_v1"`.
   Do not back-fill or invent acceptance from τ or tok/s.
4. **Phase binary drift.** See §2 binaries note — pin digests from phase rows in
   `results.json` when replaying a specific phase, not only the final summary
   binary block.

---

## 8 · Raw paths (hiptrx)

| kind | path |
|---|---|
| Summary (authoritative tables) | `/home/kaden/qcal/ladder-v2/summary.json` |
| Summary markdown / CSV | `/home/kaden/qcal/ladder-v2/summary.md`, `summary.csv` |
| Manifest / state / results | `/home/kaden/qcal/ladder-v2/{manifest,state,results}.json` |
| Artifacts | `/home/kaden/qcal/ladder-v2/artifacts/` |
| Drafts | `/home/kaden/qcal/ladder-v2/drafts/` |
| KLD sequences | `/home/kaden/qcal/ladder-v2/kld/` |
| Phase logs | `/home/kaden/qcal/ladder-v2/logs/` |
| Serve battery | `/home/kaden/qcal/ladder-v2/serve-mq{2..6}.{json,log}` |
| Parent / draft parent | `/home/kaden/qcal/parents/qwen3.8-27b`, `…/qwen3.8-27b-dflash2` |
| Checkout | `/home/kaden/hipfire-quantcal` @ manifest `bfe3cfead5856c5ee85e972b609712b0badbd1b1` |

---

## 9 · What this record is not

- Not a SKU admission, registry default, or “ship MQ*V2” decision.
- Not a claim that mq4v2 is the universal DFlash draft (it won most **paired
  decode-tok/s** comparisons on this fixture; that is a measurement, not a
  product rule).
- Not transferable across model, GPU, prompt, ref, tier recipe, or binary
  without a new checkpoint.
- **MQ2V2 is rejected** on this evidence; MQ3–MQ6 V2 remain measured ladder
  cells without an admission claim in this file.
