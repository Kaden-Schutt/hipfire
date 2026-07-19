# LFM2.5 arch-11 eager-prefill baseline and Amdahl report

**Status:** authoritative DESIGN-GATE measurement report; no batched path implemented.  
**Revision:** clean detached `62dedc41aa5d335f0518e70cb9e748da332b6138` (`lfm-redline`, Phase-0 head elision).  
**Daemon:** `/home/kaden/lfm-measure-62ded/target/release/examples/daemon`, md5 `05c622c40483b5512913b21b46704730`.  
**Host/GPU:** `hiptrx`, four AMD Radeon AI PRO R9700, `gfx1201`, 64 CU, 32 GiB, ROCm 7.2.2 (daemon reports HIP 7.2).  
**Scope:** arch 11 prefill only. `tg128` is recorded only as the required reference; no decode analysis or optimization was performed. No source was edited.

## Executive result

1. Eager prefill is still a per-token, launch-heavy **GEMV** pipeline, not GEMM/WMMA. Across all 15 traced prefill points, rocprof saw no eager `gemm_*`, WMMA, or MFMA dispatch. The actual hot projection kernels are `gemv_q8_0[_wide]` and MQ-layout `gemv_hfq4g256_*` preceded by `mq_rotate_x`.
2. At pp512, dense projection/FFN GEMV is **41.86–74.86% of end-to-end host wall** and **60.52–83.29% of device kernel time**. On 8B-A1B, MoE expert/routing is **34.93% wall / 41.58% kernel**, while Q8 projections/dense FFN are **33.26% wall / 39.59% kernel**.
3. Attention grows sharply with context: by pp2048 it is **30.01%** of 350M-Q8 wall, **32.91%** of 350M-MQ4, **15.41%** of 1.2B-Q8, **19.80%** of 1.2B-MQ4, and **14.36%** of 8B-A1B. The observed eager symbol remained `attention_q8_0_kv`; flash objects were compiled but not dispatched at pp128/512/2048.
4. The decode conv core is only **0.61–1.48%** of wall at pp512. A batched scan is still required for correctness/launch count, but isolated conv arithmetic is not the first Amdahl lever.
5. Phase-0 did its job: final-only norm + lm_head is now only **0.01–0.02%** of pp512 wall; the one actual D2H copy-engine event is **8.6–13.0 µs**. This does **not** invalidate Phase-0's measured end-to-end gain: it removed N−1 head GEMVs and full-logit D2Hs, so the post-Phase-0 baseline correctly shows only one final head.
6. Host dispatch/driver/synchronization residual remains **10.12–30.84%** at pp512. This is a reconciled residual, not “pure CPU” and not proof of launch overhead by itself.
7. Internal Hipfire runtime profiling is unavailable for this path: direct `HIPFIRE_PROFILE=1` pp512 witnesses on 1.2B-Q8 and 8B-A1B exited rc=0 but emitted **zero** profile entries/markers. The CLI `profile` JSON is static ISA/resource inventory, not elapsed kernel timing. rocprofv3 is therefore authoritative.

## Critical corrected cohort manifest

The prior blanket `hidden=2048`, `vocab=128000`, `RoPE θ=5e6` statement is wrong for dense cohorts. HF config evidence (`agent://LfmManifest`) and the traced lm_head grid agree:

|cohort|hidden|q/kv heads|head dim|q/kv dims|vocab|RoPE θ|layers / mixer|conv K|dense inter|final logits D2H|
|---|---:|---|---:|---|---:|---:|---|---:|---:|---:|
|350M dense|1024|16 / 8|64|1024 / 512|65,536|1e6|16; C C A C C A C C A C A C A C A C|3|4608|262,144 B|
|1.2B dense (Instruct = Thinking shapes)|2048|32 / 8|64|2048 / 512|65,536|1e6|16; same 10-conv/6-attn sequence|3|8192|262,144 B|
|8B-A1B MoE|2048|32 / 8|64|2048 / 512|128,000|5e6|24; 18 conv / 6 attn; L0-1 dense, L2-23 MoE|3|7168 dense; 1792 expert|512,000 B|

`max_seq` was explicitly 4096 for every direct trace and is the daemon default used by the baseline; no unsupported model-config maximum is inferred. Dense `.mq4` projections are MQ4G256/FWHT and runtime dispatches `mq_rotate_x` plus HFQ-byte-layout GEMVs. Pinned 8B-A1B has Q8 non-expert projections and MQ4G256 experts, empirically confirmed and also visible in the Q8 projection + MQ expert trace mix.

## Reachability, cards, isolation, and identity

- `hiptrx`: reachable (`hostname=hiptrx`).
- `k9lin`: **unreachable**, SSH failed `Permission denied (publickey,password)`.
- `~/.claude/ssh-targets.json`: absent; the OpenSSH alias made `hiptrx` reachable.
- The fifth/local gfx1201 was not used. Baseline maximum was four remote cards. Four-way overlap was actually achieved after daemon/GPU-visibility isolation; all accepted workers exited rc=0.

|worker|SMI ordinal|ROCR ordinal|BDF|GPU UUID|lock|cohort assignment|
|---|---:|---:|---|---|---|---|
|MeasureCard0|0|0|`0000:03:00.0`|`9eb7aeda51c88ffd`|`/tmp/lfm-measure-62ded-card0.lock`|350M Q8 + MQ4|
|MeasureCard1|1|3|`0000:13:00.0`|`e475645fe0200397`|`/tmp/lfm-measure-62ded-card1.lock`|1.2B Q8|
|MeasureCard2|2|1|`0000:C3:00.0`|`5f92432f2312a0e`|`/tmp/lfm-measure-62ded-card2.lock`|1.2B MQ4|
|MeasureCard3|3|2|`0000:E3:00.0`|`085289909a86cc63`|`/tmp/lfm-measure-62ded-card3.lock`|8B-A1B MQ4|

SMI and ROCR enumerate cards differently; BDF is the join key. Read-only HIP probes proved each corrected ROCR mapping exposes exactly one intended R9700/gfx1201. Card1-3 used distinct `HOME=/tmp/measure-home-MeasureCardN`, which isolates `$HOME/.hipfire/daemon.pid`. They used **ROCR only** and explicitly unset HIP visibility. Card0 used the exact recorded `ROCR_VISIBLE_DEVICES=0;HIP_VISIBLE_DEVICES=0`, which resolves to the same sole logical device. Every process held its fixed flock for its full lifetime.

Pinned full model md5s after scp (never rsync):

|artifact|md5|
|---|---|
|`lfm2.5-350m.q8`|`a23f2d1adf62434de8c22000162dc6e6`|
|`lfm2.5-350m.mq4`|`cb5284b8ad5c6f9e4ca859c0aff0bcd0`|
|`lfm2.5-1.2b-instruct.q8`|`241b49c0d8d8c8ed228c7f955aa9e442`|
|`lfm2.5-1.2b.mq4`|`afedbc7086514628646449f1756bd195`|
|`lfm2.5-8b-a1b.mq4`|`34f35422f6b46f5d9f7848015b51a425`|

## Authoritative fresh-process host-wall baseline

Each pp point is three separate `bun cli/index.ts profile MODEL --pp PP --tg 128 --ctx 128 --runs 1 --json` invocations, hence three fresh daemons. Fixed env: `HIPFIRE_DPM_WARMUP_SECS=10`, committed `HIPFIRE_DAEMON_BIN`, `HIPFIRE_FORWARD_LOWERED=1`, `HIPFIRE_LFM2_PREFILL_BATCH=0`, `RUST_LOG=warn`, `HIPFIRE_PROFILE` unset, per-card kernel cache, visibility, HOME, and flock as recorded above. The synthetic ids are exactly `10 + (i % 1000)`; prompt md5 is over canonical little-endian u32 ids.

Tok/s medians:

|cohort|worker/BDF|pp128|pp512|pp2048|pp4096|tg128 @ ctx128|worst PP CV|
|---|---|---:|---:|---:|---:|---:|---:|
|350M Q8|MeasureCard0 / `0000:03:00.0`|746.8|706.6|568.2|452.1|644.1|0.72%|
|350M MQ4|MeasureCard0 / `0000:03:00.0`|830.1|775.7|613.8|475.6|698.6|1.25%|
|1.2B Q8|MeasureCard1 / `0000:13:00.0`|316.6|310.2|279.6|248.4|285.8|0.22%|
|1.2B MQ4|MeasureCard2 / `0000:C3:00.0`|422.2|409.3|358.6|308.3|371.1|5.07%|
|8B-A1B MQ4|MeasureCard3 / `0000:E3:00.0`|291.0|284.9|259.4|231.5|250.1|2.03%|

`tg128` is the median of the 12 repeated ctx128 reference samples per cohort. The 1.2B-MQ4 pp128 triplet retained its first low sample (`378.4, 422.3, 422.2`); two additional fresh-process confirmations were `423.5, 422.2`. The 8B pp128 confirmations were `292.7, 292.4` after the original `278.8, 291.3, 291.0`. No sample was silently discarded.

### Per-point telemetry and samples

Pre/post telemetry is a process-boundary snapshot; low post clocks mean the card re-entered deep sleep immediately after exit, not that DPM warming was omitted. The exact pre/post JSON for every sample is linked by the raw manifest.

|cohort|pp|three tok/s samples|CV|pre gfx MHz / hotspot °C|post gfx MHz / hotspot °C|prompt md5 (LE-u32 ids)|
|---|---:|---|---:|---|---|---|
|350M Q8|128|`736.1, 748.1, 746.8`|0.72%|10–46 / 36–63|2106–2273 / 47–71|`b988825426d8d0777c5c707b012f3815`|
|350M Q8|512|`708.5, 706.1, 706.6`|0.15%|42–54 / 42–61|2342–2362 / 55–70|`d86de7a8e62f65ae54f3ed9007c15588`|
|350M Q8|2048|`570.4, 567.9, 568.2`|0.20%|38–52 / 48–60|2591–2602 / 64–71|`719576fb6ea13fe404e57bddc6d87cc6`|
|350M Q8|4096|`452.5, 452.1, 451.5`|0.09%|43–57 / 56–60|2725–2843 / 72–73|`f67f4594bce6eb907d719e5d409aff92`|
|350M MQ4|128|`809.9, 833.0, 830.1`|1.25%|16–53 / 58–60|3–2295 / 59–67|`b988825426d8d0777c5c707b012f3815`|
|350M MQ4|512|`775.8, 773.5, 775.7`|0.14%|43–51 / 58–59|2288–2340 / 67–67|`d86de7a8e62f65ae54f3ed9007c15588`|
|350M MQ4|2048|`614.1, 612.2, 613.8`|0.14%|39–48 / 58–58|2581–2596 / 67–68|`719576fb6ea13fe404e57bddc6d87cc6`|
|350M MQ4|4096|`476.5, 475.6, 474.7`|0.15%|41–44 / 57–58|5–2842 / 58–69|`f67f4594bce6eb907d719e5d409aff92`|
|1.2B Q8|128|`317.6, 316.6, 315.9`|0.22%|1–57 / 42–65|1–1064 / 56–70|`b988825426d8d0777c5c707b012f3815`|
|1.2B Q8|512|`310.4, 310.2, 309.0`|0.20%|45–57 / 50–62|1–1605 / 58–72|`d86de7a8e62f65ae54f3ed9007c15588`|
|1.2B Q8|2048|`279.3, 280.6, 279.6`|0.20%|44–57 / 58–60|1–2493 / 63–80|`719576fb6ea13fe404e57bddc6d87cc6`|
|1.2B Q8|4096|`248.1, 248.4, 248.4`|0.06%|44–48 / 63–64|2–2001 / 67–79|`f67f4594bce6eb907d719e5d409aff92`|
|1.2B MQ4|128|`378.4, 422.3, 422.2`|5.07%|3–51 / 36–63|2015–2291 / 47–71|`b988825426d8d0777c5c707b012f3815`|
|1.2B MQ4|512|`409.3, 409.0, 409.7`|0.07%|50–72 / 43–61|2033–2393 / 56–69|`d86de7a8e62f65ae54f3ed9007c15588`|
|1.2B MQ4|2048|`358.8, 358.6, 358.1`|0.08%|49–58 / 49–58|2656–2710 / 66–69|`719576fb6ea13fe404e57bddc6d87cc6`|
|1.2B MQ4|4096|`308.3, 308.4, 307.9`|0.07%|46–56 / 57–58|2912–2915 / 71–74|`f67f4594bce6eb907d719e5d409aff92`|
|8B-A1B MQ4|128|`278.8, 291.3, 291.0`|2.03%|3–57 / 43–65|1977–2253 / 55–72|`b988825426d8d0777c5c707b012f3815`|
|8B-A1B MQ4|512|`284.6, 285.2, 284.9`|0.09%|50–55 / 51–62|2381–2395 / 65–71|`d86de7a8e62f65ae54f3ed9007c15588`|
|8B-A1B MQ4|2048|`258.7, 259.7, 259.4`|0.16%|46–63 / 58–60|2125–2692 / 75–76|`719576fb6ea13fe404e57bddc6d87cc6`|
|8B-A1B MQ4|4096|`231.3, 231.5, 231.5`|0.04%|54–56 / 62–63|1249–2901 / 74–79|`f67f4594bce6eb907d719e5d409aff92`|

Raw baseline ledgers: `/home/kaden/lfm-baseline-results/MeasureCard0..3/manifest.tsv`; pp128 confirmations: `/home/kaden/lfm-baseline-confirm128/Confirm128Card2|3/manifest.tsv`.

## Rocprof reconciliation method

Every cohort received a clean direct-daemon load-only control plus separate pp128/512/2048 traces. The clean EOF protocol was:

```json
{"type":"load","model":"ABSOLUTE_PINNED_PATH","params":{"max_seq":4096}}
{"type":"bench_prefill","tokens":N}
{"type":"unload"}
```

Command shape:

```text
flock LOCK env HOME=ISO_HOME ROCR_VISIBLE_DEVICES=MAPPED HIP_VISIBLE_DEVICES=<unset> \
  HIPFIRE_DPM_WARMUP_SECS=10 rocprofv3 \
  --kernel-trace --memory-copy-trace --hip-runtime-trace --stats --summary -f csv \
  -d TRACE_DIR -o trace -- DAEMON < requests.jsonl
```

All 20 direct traces (five controls + 15 pp points) exited rc=0. Start of the prefill window is the first `embedding_q8` dispatch; end is the last non-runtime kernel dispatch. Load/init-only `__amd_rocclr_*` work is outside this window and is independently present in controls. The unprofiled median wall is used for the user-facing denominator because rocprof inflates host dispatch wall; device kernel duration comes from rocprof timestamps. `residual = unprofiled host wall − rocprof kernel duration − copy-engine duration`. It includes dispatch, driver, synchronous API waits, and any cross-run reconciliation error; it is **not pure CPU**.

|cohort|pp|unprofiled host wall ms (median)|traced daemon wall ms|rocprof kernel ms|copy-engine ms|reconciled dispatch/driver/sync residual ms|kernel launches|
|---|---:|---:|---:|---:|---:|---:|---:|
|350M Q8|128|171.40|240.47|129.91|0.00876|41.48|28,034|
|350M Q8|512|724.60|995.76|563.12|0.00884|161.47|112,130|
|350M Q8|2048|3604.36|4544.82|2942.21|0.00908|662.15|448,514|
|350M MQ4|128|154.20|244.27|103.30|0.00848|50.88|35,714|
|350M MQ4|512|660.05|1017.42|456.46|0.00956|203.58|142,850|
|350M MQ4|2048|3336.59|4786.23|2592.74|0.00884|743.85|571,394|
|1.2B Q8|128|404.30|466.24|360.46|0.00828|43.82|28,034|
|1.2B Q8|512|1650.55|1891.28|1483.45|0.00864|167.09|112,130|
|1.2B Q8|2048|7324.75|8298.04|6652.01|0.00828|672.73|448,514|
|1.2B MQ4|128|303.17|377.89|248.60|0.00844|54.56|35,714|
|1.2B MQ4|512|1250.92|1550.30|1035.55|0.00884|215.35|142,850|
|1.2B MQ4|2048|5711.10|6891.24|4847.97|0.00836|863.12|571,394|
|8B-A1B MQ4|128|439.86|545.71|368.26|0.01272|71.59|47,746|
|8B-A1B MQ4|512|1797.12|2192.72|1509.88|0.01300|287.23|190,978|
|8B-A1B MQ4|2048|7895.14|9450.67|6741.75|0.01268|1153.38|763,906|

Trace telemetry:

|cohort|trace worker/BDF|pre gfx MHz / hotspot °C|post gfx MHz / hotspot °C|raw trace ledger|
|---|---|---|---|---|
|350M Q8|Rocprof350Card0 / `0000:03:00.0`|41–51 / 52–67|98–3171 / 64–74|`/home/kaden/lfm-rocprof-direct-results/Rocprof350Card0/manifest.tsv`|
|350M MQ4|Rocprof350Card0 / `0000:03:00.0`|46–53 / 64–66|2993–3089 / 70–72|`/home/kaden/lfm-rocprof-direct-results/Rocprof350Card0/manifest.tsv`|
|1.2B Q8|RocprofDenseCard1 / `0000:13:00.0`|0–58 / 46–64|2954–3177 / 58–77|`/home/kaden/lfm-rocprof-direct-results/RocprofDenseCard1/manifest.tsv`|
|1.2B MQ4|RocprofMqCard2 / `0000:C3:00.0`|0–48 / 39–58|2952–3229 / 52–68|`/home/kaden/lfm-rocprof-direct-results/RocprofMqCard2/manifest.tsv`|
|8B-A1B MQ4|RocprofMoeCard3 / `0000:E3:00.0`|1–60 / 49–64|2763–2862 / 59–74|`/home/kaden/lfm-rocprof-direct-results/RocprofMoeCard3/manifest.tsv`|

## ONE Amdahl-ranked table

`HW rank` ranks additive pp512 host-wall buckets. `kernel rank` and `kernel-only %` rank only rocprof device kernels. Percentages at pp128/pp2048 use the same stage rules and each cohort's unprofiled median wall.

Stage rules: max-grid singleton Q8 GEMV + last RMSNorm = final head; non-head GEMVs and dense MQ rotations = projection/FFN; attention + KV write + RoPE = attention; `conv1d_gated_decode_f32` = conv; MoE expert/top-k/route/rotate/combine kernels and the 32-row router GEMV = MoE; remaining device kernels = norm/activation/residual/embed.

|cohort|HW rank|kernel rank|bucket|pp128 host-wall %|pp512 ms|pp512 host-wall %|pp512 kernel-only %|pp2048 host-wall %|
|---|---:|---:|---|---:|---:|---:|---:|---:|
|350M Q8|1|1|Projection / dense FFN GEMV|52.30%|358.323|49.45%|63.63%|39.67%|
|350M Q8|2|—|Host dispatch / driver / sync residual|24.20%|161.467|22.28%|—|18.37%|
|350M Q8|3|2|Norm / activation / residual / embed|14.37%|98.736|13.63%|17.53%|10.87%|
|350M Q8|4|3|Attention + KV + RoPE|7.64%|96.139|13.27%|17.07%|30.01%|
|350M Q8|5|4|Conv gated decode core|1.41%|9.802|1.35%|1.74%|1.08%|
|350M Q8|6|5|Final-only norm + lm_head|0.07%|0.121|0.02%|0.02%|0.00%|
|350M Q8|7|—|Transfer copy-engine event|0.01%|0.009|0.00%|—|0.00%|
|350M MQ4|1|1|Projection / dense FFN GEMV|44.72%|276.275|41.86%|60.52%|34.51%|
|350M MQ4|2|—|Host dispatch / driver / sync residual|33.00%|203.575|30.84%|—|22.29%|
|350M MQ4|3|2|Attention + KV + RoPE|8.23%|94.956|14.39%|20.80%|32.91%|
|350M MQ4|4|3|Norm / activation / residual / embed|12.38%|75.321|11.41%|16.50%|9.10%|
|350M MQ4|5|4|Conv gated decode core|1.58%|9.778|1.48%|2.14%|1.19%|
|350M MQ4|6|5|Final-only norm + lm_head|0.08%|0.135|0.02%|0.03%|0.00%|
|350M MQ4|7|—|Transfer copy-engine event|0.01%|0.010|0.00%|—|0.00%|
|1.2B Q8|1|1|Projection / dense FFN GEMV|76.48%|1235.562|74.86%|83.29%|67.71%|
|1.2B Q8|2|—|Host dispatch / driver / sync residual|10.84%|167.092|10.12%|—|9.18%|
|1.2B Q8|3|2|Norm / activation / residual / embed|8.02%|128.954|7.81%|8.69%|7.07%|
|1.2B Q8|4|3|Attention + KV + RoPE|3.96%|108.329|6.56%|7.30%|15.41%|
|1.2B Q8|5|4|Conv gated decode core|0.64%|10.370|0.63%|0.70%|0.61%|
|1.2B Q8|6|5|Final-only norm + lm_head|0.06%|0.233|0.01%|0.02%|0.00%|
|1.2B Q8|7|—|Transfer copy-engine event|0.00%|0.009|0.00%|—|0.00%|
|1.2B MQ4|1|1|Projection / dense FFN GEMV|66.59%|804.332|64.30%|77.67%|56.45%|
|1.2B MQ4|2|—|Host dispatch / driver / sync residual|18.00%|215.353|17.22%|—|15.11%|
|1.2B MQ4|3|2|Norm / activation / residual / embed|9.17%|112.816|9.02%|10.89%|7.91%|
|1.2B MQ4|4|3|Attention + KV + RoPE|5.31%|107.780|8.62%|10.41%|19.80%|
|1.2B MQ4|5|4|Conv gated decode core|0.85%|10.393|0.83%|1.00%|0.73%|
|1.2B MQ4|6|5|Final-only norm + lm_head|0.08%|0.234|0.02%|0.02%|0.00%|
|1.2B MQ4|7|—|Transfer copy-engine event|0.00%|0.009|0.00%|—|0.00%|
|8B-A1B MQ4|1|1|MoE experts + routing|35.79%|627.797|34.93%|41.58%|31.80%|
|8B-A1B MQ4|2|2|Projection / dense FFN GEMV|34.05%|597.716|33.26%|39.59%|30.33%|
|8B-A1B MQ4|3|—|Host dispatch / driver / sync residual|16.27%|287.233|15.98%|—|14.61%|
|8B-A1B MQ4|4|3|Norm / activation / residual / embed|9.03%|157.169|8.75%|10.41%|7.95%|
|8B-A1B MQ4|5|4|Attention + KV + RoPE|3.66%|108.296|6.03%|7.17%|14.36%|
|8B-A1B MQ4|6|5|Conv gated decode core|1.09%|18.449|1.03%|1.22%|0.95%|
|8B-A1B MQ4|7|6|Final-only norm + lm_head|0.10%|0.448|0.02%|0.03%|0.01%|
|8B-A1B MQ4|8|—|Transfer copy-engine event|0.00%|0.013|0.00%|—|0.00%|

### Interpretation for contract freeze

- **First vertical slice:** projection/FFN batching is the dominant dense lever, especially 1.2B. This supports a true M=N WMMA path; optimizing eager GEMV does not create batched prefill.
- **MQ cohorts:** the 350M/1.2B MQ projection bucket includes `mq_rotate_x`; the existing batched FWHT rotate must be reused before HFQ-byte-layout WMMA GEMM.
- **8B:** expert/routing plus Q8 projection work accounts for **68.19%** of pp512 wall and **81.17%** of device kernel time. The grouped-WMMA MoE path and Q8 non-expert batched projections are both required.
- **Long context:** attention overtakes some projection benefit by pp2048, especially 350M. Admission must test pp2048/4096; a pp128-only win is insufficient.
- **Conv:** a time-axis scan is semantically necessary and collapses thousands of launches, but its standalone device arithmetic is a low-single-digit Amdahl fraction.
- **Head/transfer:** Phase-0 makes final head/copy negligible at long prompts. Do not spend Phase 1 on further lm_head tuning.

## CPU and transfer decomposition

Synthetic token construction/tokenization is outside the timed loop, so tokenize time is **0 in this benchmark**. Source scout evidence says the eager loop performs per-token position upload and a final logits download. rocprof's copy-engine domain saw exactly one D2H event after first embedding and no H2D event; the tiny synchronous position path appears among HIP `hipMemcpy` API calls but not as standalone copy-engine events. HIP API duration is inclusive of queue synchronization and overlaps device execution, so it is shown but **not added** to Amdahl buckets.

|cohort (pp512)|HIP `hipMemcpy` calls in prefill window|inclusive HIP API ms (not additive)|copy-engine events|copy-engine D2H ms|final logits bytes|
|---|---:|---:|---|---:|---:|
|350M Q8|522|54.726|1 D2H; 0 H2D events|0.00884|262,144|
|350M MQ4|522|42.825|1 D2H; 0 H2D events|0.00956|262,144|
|1.2B Q8|522|161.776|1 D2H; 0 H2D events|0.00864|262,144|
|1.2B MQ4|522|84.208|1 D2H; 0 H2D events|0.00884|262,144|
|8B-A1B MQ4|530|83.561|1 D2H; 0 H2D events|0.01300|512,000|

For dense cohorts the final logits payload is 65,536 f32 = 262,144 B; for 8B it is 128,000 f32 = 512,000 B. At pp512, dense traces have 522 HIP memcpy calls (N+10); 8B has 530 (N+18). The only independent copy-engine event is final D2H. All other host/transfer/API waiting remains conservatively inside the reconciled dispatch/driver/sync residual.

## Hipfire internal profiler versus rocprof

|witness|HIPFIRE_PROFILE runtime entries|rocprof pp512 kernel time|result|
|---|---:|---:|---|
|1.2B-Q8, BDF `13:00.0`, pp512|0|1483.447 ms|internal runtime attribution unavailable|
|8B-A1B, BDF `E3:00.0`, pp512|0|1509.876 ms|internal runtime attribution unavailable|

Both direct `HIPFIRE_PROFILE=1` witnesses returned a valid `prefill_result` and exited rc=0, yet stderr contained only GPU/cache/DPM lines and zero `=== PROFILE` markers. The CLI `profile` JSON supplies compiled-kernel resources only. Source scouts additionally identified no `begin_timer` around the dominant Q8 `gemv_q8_0`, `conv1d_gated_decode_f32`, or `embedding_q8`; even adding a start/stop range would retain major blind spots. This repeats the historical 65%-blindspot class. **Do not call `prefill_result.ms` “kernel prefill”: it is synchronized host wall.**

Raw witnesses: `/home/kaden/lfm-internal-profile-witness/InternalProfileDense|Moe/`; raw direct traces: `/home/kaden/lfm-rocprof-direct-results/Rocprof350Card0|DenseCard1|MqCard2|MoeCard3/`.

## gfx1201 ISA/resource report

Hot code objects were taken from the exact per-card JIT cache, unbundled with `clang-offload-bundler`, disassembled by ROCm 7.2.2 `llvm-objdump --mcpu=gfx1201`, and read with `llvm-readelf --notes`. Metadata counts below are exact code-object fields; “occupancy fit” is a register-only ISA fit using gfx1201's 1536 VGPR/SIMD and 16-wave cap, not measured resident occupancy. Dynamic launch occupancy/counters were not collected. All listed kernels have zero private segment, zero VGPR/SGPR spills, and no scratch.

|kernel|runtime status|VGPR|SGPR|fixed LDS B|scratch B / spills|wave|max WG|occupancy fit|ISA inst|global ld/st|DS|WMMA/MFMA|
|---|---|---:|---:|---:|---|---:|---:|---|---:|---:|---:|---:|
|`gemv_q8_0`|hot: 1.2B + 8B projections/head|29|16|0|0 / 0|32|32|16 wave32/SIMD cap [heuristic]|334|27/1|5|0|
|`gemv_q8_0_wide`|hot: 350M projections/head|25|12|0|0 / 0|32|1024|16 wave32/SIMD cap [heuristic]|342|15/1|5|0|
|`gemv_hfq4g256_multirow_r2`|hot: dense MQ projections|94|19|0|0 / 0|32|32|16 wave32/SIMD cap [heuristic]|805|14/2|10|0|
|`gemv_hfq4g256_residual`|hot: dense MQ residual projections|94|19|0|0 / 0|32|32|16 wave32/SIMD cap [heuristic]|859|15/2|10|0|
|`attention_q8_0_kv`|hot at pp128/512/2048|31|34|0|0 / 0|32|1024|16 wave32/SIMD cap [heuristic]|854|14/1|35|0|
|`attention_flash_q8_0_tile`|disassembled; not observed|71|44|0|0 / 0|32|32|16 wave32/SIMD cap [heuristic]|1461|37/2|59|0|
|`attention_flash_q8_0_reduce`|disassembled; not observed|21|48|8|0 / 0|32|256|16 wave32/SIMD cap [heuristic]|646|11/1|15|0|
|`conv1d_gated_decode_f32`|hot: all cohorts|18|16|0|0 / 0|32|1024|16 wave32/SIMD cap [heuristic]|314|6/3|0|0|

Important correction to the requested “Q8 GEMM” item: eager M=1 runs **GEMV**, so the authoritative disassembly is `gemv_q8_0[_wide]`. Every listed scalar eager/attention/conv kernel has **0 WMMA/MFMA instructions**. Flash tile/reduce objects were disassembled for readiness but were not present in any pp128/512/2048 eager trace; `attention_q8_0_kv` was the actual hot kernel. Raw ISA: `/home/kaden/lfm-isa-results/` (unbundled ELF, disassembly, notes, HSACO md5 ledger).

## Historical context

- There is no prior LFM batched-prefill performance history: `docs/MODELS.md` states LFM prefill is per-token `decode_step`.
- The pre-Phase-0 exploratory local RX 9070 table in the plan had pp≈tg (e.g. 350M-Q8 pp128 521 vs tg 534; 8B pp128 248 vs tg 244). It is not directly comparable to this R9700/ROCm-7.2.2 baseline.
- Phase-0 on local gfx1201 previously measured +9–11% pp for 350M-Q8 and +8–9% for 1.2B-Q8, tg flat, with bit-identical parity. The present report measures that committed state and confirms the remaining final head is negligible.
- Closest transferable history is Qwen35 gfx12 WMMA (`docs/lessons_learned/gfx12_prefill_wmma_2026_05_19.md`): commit `218a88df` moved Q8 batched GEMM onto gfx12 WMMA; pp256 1017.6→2966 tok/s (+192%), pp512 1022→3115. Internal profiling missed the prior `gemm_q8_0_batched` despite 65% GPU share.
- Negative historical controls: M2 grouped GEMM commit `db672aa8` regressed prefill 4%; graph-prefill commits `42ca533d`/`d86fafa3` regressed 1.6%. Do not retry without a new mechanism.
- Methodology applied: DPM 10s, fresh processes, byte-identical prompts, full model/binary identities, card locks, and rocprof cross-check. The repo's ±1–3% warm noise guidance is why the retained pp128 first-sample anomalies were confirmed rather than discarded.

## Infra incidents and exclusions

1. Initial Card1-3 starts collided on shared `$HOME/.hipfire/daemon.pid`; no result was accepted. Fixed with supported per-worker HOME isolation.
2. Initial isolated retries exported both ROCR and HIP ordinal and double-filtered to no device; no result was accepted. Read-only HIP probes established the non-identical SMI/ROCR map; accepted runs use ROCR-only on Card1-3.
3. One CLI-under-rocprof attempt finalized files but hung after CLI SIGTERM. It is excluded. The exact stale bash/bun PIDs were identified by lock/fuser; only those verified children were terminated, then flock was proven free. All accepted traces use direct daemon JSONL + unload + EOF and exit rc=0.
4. No broad `pkill`, no rsync, no coherence gate, no local GPU, no overlapping jobs on one lock, no decode optimization, and no production/source edits.

## Raw artifact index

- Baseline: `/home/kaden/lfm-baseline-results/MeasureCard0..3/`
- pp128 confirmations: `/home/kaden/lfm-baseline-confirm128/`
- Direct rocprof: `/home/kaden/lfm-rocprof-direct-results/`
- Internal profile witnesses: `/home/kaden/lfm-internal-profile-witness/`
- ISA/disassembly: `/home/kaden/lfm-isa-results/`
- Exact revision staging: `/home/kaden/lfm-measure-62ded`

All accepted manifests carry host, worker, SMI/ROCR ordinal, BDF, lock, full HEAD, binary md5, model md5, workload kind/pp, prompt md5, start/end UTC, rc, full env, raw output, and pre/post telemetry paths.
