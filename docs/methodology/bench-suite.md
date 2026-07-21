# Production serve benchmark suite

Implementation map for hipfire's native Rust serve-path harness. Measurement
rules (identity, warmup, noise, and disposition) live in
[`perf-benchmarking.md`](perf-benchmarking.md); claim routing lives in
[`../VALIDATION.md`](../VALIDATION.md).

## Canonical production path

`scripts/serve_harness.py` drives the same OpenAI-compatible Rust service that
users call. It resolves a concrete sampling recipe, starts `hipfire serve` on
loopback (unless `--no-spawn` is selected), waits for `/health` to report a
loaded model, then records response semantics and daemon timings per turn.

Always inspect the resolved configuration before a run:

```bash
python3 scripts/serve_harness.py \
  --model ~/.hipfire/models/qwen3.6-35b-a3b.mq4r \
  --tag qwen3.6:35b-a3b-mq4r \
  --sampling registry \
  --show-config
```

Then run the exact same arguments with an output path:

```bash
HIPFIRE_CLI_BIN=./target/release/hipfire \
HIPFIRE_DAEMON_BIN=./target/release/examples/daemon \
python3 scripts/serve_harness.py \
  --model ~/.hipfire/models/qwen3.6-35b-a3b.mq4r \
  --tag qwen3.6:35b-a3b-mq4r \
  --sampling registry \
  --mode battery \
  --max-tokens 4096 \
  --out /tmp/serve-harness.json
```

Useful modes:

| Mode | Purpose |
|---|---|
| `battery` | Independent code/reason/factual/prose/instruction turns |
| `chain` | Growing conversation; exercises prefix reuse and cross-turn behavior |
| `session` | Exact supplied multi-turn fixture; always pass `--session PATH` |

The harness records content/reasoning previews, finish reason, empty/runaway
and attractor checks, cache accounting, TTFT, prefill/decode rates, and
speculative-decode fields when the daemon emits them. A green semantic harness
is not numerical parity or Redline certification.

## Production versus synthetic probes

| Path | Evidence class |
|---|---|
| Rust HTTP service + `serve_harness.py` | Production user-path semantics and per-turn timings |
| `hipfire bench --matrix` | Synthetic daemon `bench_prefill` / `bench_decode` probe |
| `bench_qwen35_mq4` | In-process speed floor, profile hook, or bisect lower bound |
| `scripts/probe_commits.sh` | Optional fresh-process in-process commit A/B |

Do not cite a synthetic matrix as production-path throughput. Conversely, do
not reinterpret production service numbers as a calibrated speed-gate floor
without deliberately re-baselining that gate.

The removed TypeScript sweep used to combine a fixed prefill matrix with one
resident decode sample. There is intentionally no compatibility shim: use
`hipfire bench --matrix` for exploratory shape sweeps and the native serve
harness for production-path claims, labeling each evidence class accurately.

## Identity required for kept results

The harness JSON is a measurement payload, not a complete certification
bundle. Archive these alongside every kept result:

- branch, commit, and clean/dirty state;
- native CLI and daemon paths plus hashes;
- exact model, sidecar, and draft hashes;
- prompt fixture path and byte hash;
- GPU product, gfx arch, device selection, ROCm, and driver identity;
- config, sampler, seed, KV/context/generation lengths, graph/spec flags, and
  every relevant `HIPFIRE_*` override;
- warmup policy, fresh/resident process policy, run order, run count, and UTC
  timestamp.

One resident harness invocation is one resident-process sample. Promotion or
retained A/B work needs multiple fresh service processes, with ABBA or another
declared interleave when thermal/order bias matters.

## Build and attach modes

```bash
cargo build --release -p hipfire-cli
cargo build --release -p hipfire-runtime --example daemon --features deltanet
```

By default the harness starts its own loopback service and kills only that
process group. To test an already-running loopback service:

```bash
python3 scripts/serve_harness.py \
  --model /absolute/model/path \
  --tag exact:registry-tag \
  --sampling registry \
  --no-spawn --port 11435 \
  --out /tmp/serve-harness.json
```

Serve has no authentication or TLS. Do not bind a test service beyond
loopback without explicit network exposure approval.

## Related routes

| Tool | Role | Not for |
|---|---|---|
| `scripts/speed-gate.sh` | Committed in-process floors for covered models | General production throughput |
| `dflash_spec_demo --prompts-file` | DFlash tau/perf research with pinned fixtures | AR service substitute |
| `scripts/redline_daemon_harness.py` | Capture/shadow evidence | Redline product certification by itself |
| `scripts/rocprof-wrap.sh` + `coverage-audit.py` | Device-time attribution | End-to-end service semantics |
| `hipfire-atlas` | ISA-fit measurement and analysis | Runtime promotion proof |

No harness exit code admits a model route. Admissions remain in
[`../admissions.yml`](../admissions.yml) and fail closed outside the exact
earned row.
