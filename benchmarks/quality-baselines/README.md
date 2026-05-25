# Quant quality eval — KLD-primary harness

Tracking issues #113 (uniform MQ-family quality) and #116 (Lloyd
quality). Canonical reference: `docs/plans/issue-113-quant-quality-eval.md`
(PRD). The prefill-mode scoring methodology is §5 of that PRD.

This directory holds the eval harness: slice + scripts +
canary fixture + reference manifest + result tables.

## Quick-start (download BF16 references, then run an eval)

The two BF16 reference dumps (Qwen3.5-9B and Qwen3.6-27B, ~2.5 GB
each) live at HF Hub **dataset** repo
[`hipfire-models/qwen-kldref`](https://huggingface.co/datasets/hipfire-models/qwen-kldref).
The `manifest.json` in `harness/` is the SHA-pinned index. To pull
them locally and verify SHA256 in one step:

```bash
# 1. From the repo root, set up the project venv (one-time).
python3 -m venv .venv
.venv/bin/pip install huggingface_hub

# 2. Pull both refs into benchmarks/quality-baselines/refs/.
./scripts/fetch-eval-refs.sh
```

What the script does (read `scripts/fetch-eval-refs.sh` for the
authoritative recipe):

1. Parses `benchmarks/quality-baselines/harness/manifest.json`.
2. For each entry under `.references`, checks
   `benchmarks/quality-baselines/refs/<name>` — if present, verifies
   sha256 against the manifest's expected value; if missing, pulls
   it via `huggingface_hub.hf_hub_download(repo_id=hf_repo,
   repo_type=hf_repo_type, filename=name)` and then verifies sha256.
3. Returns non-zero on any SHA256 mismatch or download failure.

The refs are gitignored. After `fetch-eval-refs.sh` succeeds the
runtime examples (`eval_hipfire`, `eval_gguf`) find them at the
paths `eval_hipfire --ref benchmarks/quality-baselines/refs/<name>`.
The examples' internal `verify_ref_sha256` re-checks the SHA on
each invocation, so a corrupted local ref is caught at run start.

### Alternative download paths

If you only want one ref (e.g. just the 9B), inline Python:

```bash
.venv/bin/python3 - <<'EOF'
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id="hipfire-models/qwen-kldref",
    repo_type="dataset",
    filename="qwen3.5-9b-bf16.kldref.bin",
    local_dir="benchmarks/quality-baselines/refs/",
)
EOF
```

Or via the `hf` CLI (requires `pip install huggingface_hub[cli]`):

```bash
hf download --repo-type dataset hipfire-models/qwen-kldref \
  qwen3.5-9b-bf16.kldref.bin qwen3.6-27b-bf16.kldref.bin \
  --local-dir benchmarks/quality-baselines/refs/
```

Either way: after the download, verify against the manifest with
`./scripts/fetch-eval-refs.sh` (re-running it idempotently is the
intended pattern — files already present + valid SHA are skipped).

## Layout

```
benchmarks/quality-baselines/
├── README.md                    # this file
├── slice/                       # frozen prompt bytes
│   ├── README.md
│   ├── make_slice.sh            # generator from wikitext-2 train (uses .venv/bin/python3)
│   ├── slice.md5                # checksum tripwire
│   └── wikitext2-1024s-2048ctx.txt   # 10.5 MB committed fixture, md5 83b0205a…
├── harness/                     # the actual harness scripts + format readers
│   ├── README.md                # how-to-add-quant
│   ├── manifest.json            # SHA-pinned reference index
│   ├── kld_reduce.py            # bootstrap CI + result-table emitter (incl. PPL)
│   ├── kldref_format.py         # HFKLDR + HFKSEQ-v2 reader/writer
│   ├── tokenizer_parity.py      # Step 1.5 tokenizer-parity check
│   └── canary.md                # 11-seq fixture (expected KLDs land after Step 5)
├── refs/                        # BF16 ref blobs (gitignored)
│   └── .gitignore
└── results/                     # output tables + plots
    └── README.md
```

The producer / candidate-side binaries are Rust examples in
`crates/hipfire-runtime/examples/` — `build_kld_ref.rs`,
`eval_hipfire.rs`, `eval_gguf.rs`, `tokenize_slice.rs`. The harness
reaches into them via plain `cargo run --release --example <name>`
invocations; nothing in this directory needs to know their paths.

## Workflow (overview)

1. **One-time** — generate the slice via `make_slice.sh`, dump BF16
   references on gfx1151 (via `build_kld_ref.rs`), upload to
   `hipfire-models/hipfire-eval-refs`, fill `manifest.json` with
   sha256 + `hf_repo` + producer metadata.

2. **Per quant variant** — run `eval_hipfire` (hipfire candidates)
   or `eval_gguf` (GGUF candidates) against the cached reference.
   Output: a small `<variant>__<arch>.kldseq` file under
   `results/<date>/per-seq/`.

3. **Aggregate** — `kld_reduce.py` reads per-sequence-KLD files,
   bootstraps 95% CIs, emits the result table (markdown + JSON) with
   columns: variant, arch, n_chunks, mean KLD ± CI, p99 KLD, PPL.

## Status (2026-05-11)

- PRD: `docs/plans/issue-113-quant-quality-eval.md` (scoring-mode in §5).
- Steps 0–4 (harness skeleton, format readers, BF16 ref dumps): **done**.
- Step 5 (hipfire candidate scoring): **per-token done** for 9B MQ3/MQ4/
  MQ3-Lloyd/MQ6 on gfx1100. **Prefill canonical** since 2026-05-11
  (MQ4 gfx1100 prefill committed; per-token rows retained as historical).
- Step 6 part-A (27B BF16 ref): **done** + uploaded to HF.
- Step 6 part-B (27B hipfire matrix): **deferred** per the 2026-05-11
  Pivot (MQ format is structurally noisier than community K-quants;
  focus shifts to HFP4G32 / MFP4G32 once .hfq files exist).
- Step 7.A (`eval_gguf.rs`): **done**.
- Step 7.B (9B GGUF anchor candidate runs on gfx1151): **done** —
  all 7 anchors (Q8_0, Q6_K, Q4_K_M, UD-Q3/Q4/Q5/Q6_K_XL).
- Steps 8 (DFlash τ) and 9 (write-up): **deferred** per Pivot.

## References

- llama.cpp pinned commit: `9dcf83552887bb898b4a98a5761361e504e31fc3`.
- HF refs repo: [`hipfire-models/qwen-kldref`](https://huggingface.co/hipfire-models/qwen-kldref)
  (both 9B and 27B BF16 references uploaded 2026-05-11).
- PRD: `docs/plans/issue-113-quant-quality-eval.md`.
