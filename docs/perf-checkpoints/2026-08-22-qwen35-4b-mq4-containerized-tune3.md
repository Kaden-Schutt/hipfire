# Qwen3.5-4B MQ4 AR decode — containerized verification of tune/iter3-gate-up-bt2

- **Date:** 2026-08-22
- **Lifecycle:** historical
- **Fixture:** `~/.local/models` → `/root/.hipfire/models/Qwen/Qwen3.5-4B-MQ4/qwen3.5-4b.mq4`,
  md5 `712b69f8cf1016081cfa507c4d50e33d`, 2,588,006,400 bytes
- **Commit:** `98824dd37204996097bf35230154a86b0482c8df` (branch `tune/iter3-gate-up-bt2`)
- **Image:** `hipfire-rocmfp4:git-98824dd3-rocm7.14.0-tune3`
  (`REPO_URL=https://github.com/ghazni101/hipfire`, `BASE_IMAGE=rocm-dev:7.14.0`,
  `ROCM_VERSION=7.14.0`; daemon sha256 `3d333ea86da709e8…`)
- **Method:** one-shot compose service `bench-4b-tune3`
  (`docker-compose.bench-4b-tune3.yml`), `hipfire bench --matrix --json
  --kv-mode q8 --spec off --pp 64,2048 --ctx 64,2048 --tg 128 --runs 5
  --warmups 3`, gfx1100 RX 7900 XTX 24 GiB, exclusive GPU,
  `HIPFIRE_KERNEL_CACHE=/var/cache/hipfire` (external volume
  `hipfire-rocmfp4-kcache`). Plain AR — no speculative decoding.
- **Result JSON:** fork tree `~/projects/docker-containers/hipfire/out/bench-4b/tune3-4b-mq4-q8.json`

## Decode (tok/s, 5 fresh runs each)

| Context | Samples | Median |
|---|---|---|
| 64 | 209.82 / 209.86 / 210.15 / 209.95 / 209.92 | **209.92** |
| 2048 | 205.33 / 205.18 / 205.21 / 205.18 / 204.99 | **205.18** |

Prefill: 64 tok → 3746.0 tok/s median; 2048 tok → 4394.2 tok/s median.

All ten decode samples exceed 200 tok/s; spread < 0.2% within each context point.
Session-start baseline on the same fixture was 192.5 tok/s @ ctx64
(2026-08-05 `out/bench-4b/mq4.json`); the four admitted fusions
(`57b189fd`, `1a254267`, `fb244077`→`fb24c077`, `2b1d5dcb`) carry the delta.

## Coherence spot-check

Greedy `--temp 0` generation inside the same image
(`HIPFIRE_VERIFY_GRAPH=1`): prompt "What is the capital of France?" →
"The capital of France is Paris." Exit 0. Token-exact certification of the
four fused kernels was done during the campaign (see
`docs/kernel-tune-decode-campaign-2026-08.md`).
