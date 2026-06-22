---
name: run-model
description: Run a hipfire model end-to-end via the daemon JSON-lines protocol. Use when the user wants to load a model and generate output, run a batch-prefill session, or validate a model loads and produces tokens on this machine.
---

# Run a hipfire model

## Architecture context

hipfire streams model weights from disk, batching tokens through one
layer/expert at a time to keep VRAM usage bounded. Currently:

- **MoE expert weights** are paged — only K_TOP active experts per layer are
  in VRAM at any moment (`paged_experts=true`).
- **Attention/dense layers** are loaded into VRAM at model-load time (not yet
  paged). For 397B this means ~14–16 GB of dense attention weight allocations
  via a single large `hipMalloc` call at load time.
- Full layer-streaming (one layer in VRAM at a time) is the roadmap target.

### Memory on the dev APU (Radeon 8060S — RDNA3.5)

- 512 MB **dedicated VRAM** (carved from GPU silicon)
- ~120 GiB **GTT** (system RAM shared with CPU/NPU — GPU can DMA to/from it
  via `hipMalloc` overflow when VRAM is full)
- Total system RAM: 124 GiB

GTT is accessible. The SLAB loader uses it successfully (tested: 25.79 GiB for
35B at 4.35 GiB/s in 512 MiB banks). See the 397B section for notes on a
previous deadlock that was diagnosed and fixed in `hfq.rs`.

## GPU lock

`hipfire-daemon` auto-acquires a flock(2) GPU lease before HIP init and releases
it on exit, so running models through the daemon needs no manual locking. For
non-daemon GPU binaries (cargo `--example` benches), coordinate via the native
CLI mutex:

```bash
hipfire gpu-lock acquire "run-model" --watch-pid $$   # blocks until free
# ... run stuff ...
hipfire gpu-lock release                              # or let it release on pid death
```

Check current holder: `hipfire gpu-lock status`

Check free VRAM: `rocm-smi --showmeminfo vram`

## Build

```bash
cargo build -p hipfire-runtime --example daemon
```

Release if benchmarking:
```bash
cargo build -p hipfire-runtime --example daemon --release
```

## Daemon protocol

The daemon reads JSON lines from stdin, writes JSON lines to stdout. One
daemon at a time — exclusive lock on `~/.hipfire/daemon.pid`.

Start and talk to it:
```bash
./target/debug/examples/daemon
```

Then send JSON on stdin:
```json
{"type":"load","model":"/path/to/model.hfq","params":{"max_seq":4096}}
{"type":"generate","id":"r1","prompt":"Hello","temperature":0.0,"max_tokens":32}
{"type":"unload"}
```

Or pipe a whole session:
```bash
./target/debug/examples/daemon <<'EOF'
{"type":"load","model":"/path/to/model.hfq","params":{"max_seq":4096}}
{"type":"generate","id":"r1","prompt":"Hello","temperature":0.0,"max_tokens":32}
{"type":"unload"}
EOF
```

## Stale daemon.pid

If the daemon refuses to start (`FATAL: hipfire daemon already running`):
```bash
rm -f ~/.hipfire/daemon.pid
```
Or use the `serve-restart` skill to do a full clean cycle.

## Small models (fits in dev VRAM)

The dev machine has a Radeon 8060S with 512 MB carve-out. Models that fit:

| Model | File | Notes |
|-------|------|-------|
| qwen3.5-0.8b | `~/.hipfire/models/qwen3.5-0.8b.mq6.hfq` | Good smoke target |
| qwen3.5-35b-a3b | `~/.hipfire/models/qwen3.5-35b-a3b.mq6.hfq` | Small MoE, K_TOP=8 |

## 397B paged-expert model

> **Previous deadlock (2026-06-11) — fixed in `hfq.rs`; model now runs on dev APU.**
> Root cause: `HfqFile::open` issued `mmap.advise(Sequential)` which triggered
> 291 GiB of kernel readahead. When the slab loader then opened a second
> `O_DIRECT` fd to the same inode, the readahead kworker held an inode lock the
> O_DIRECT I/Os needed → system hang requiring hard reboot. Fix: `open_at_offset`
> now delegates to `open_index_only_at_offset` for files > 64 GiB (no mmap,
> no readahead), and `drop_mmap()` calls `FADV_DONTNEED` as a belt-and-suspenders
> safeguard. Tested 2026-06-11: load=2.47s, TTFT=18.7s, decode=0.5 tok/s (debug,
> cold pread; each token pages 60×10 experts from the 291 GiB file).

The HFQM v2 repacked file with the module table is required for paged experts:
```
/home/sadara/.hipfire/models/qwen3.5-397b-a17b.mq6.hfqm2.hfq.tmp
```
(still `.tmp` pending rename; content is complete as of 2026-06-10)

```bash
HIPFIRE_QWEN35_PAGED_EXPERTS=1 \
HIPFIRE_QWEN35_EXPERT_CACHE_MB=128 \
./target/debug/examples/daemon <<'EOF'
{"type":"load","model":"/home/sadara/.hipfire/models/qwen3.5-397b-a17b.mq6.hfqm2.hfq.tmp","params":{"max_seq":512}}
{"type":"generate","id":"r1","prompt":"Hello","temperature":0.0,"max_tokens":8}
{"type":"unload"}
EOF
```

Known-working batch size from BUGS.md: B=4, 16 tokens/session, `cache128`.
B=8 was SIGKILLed at `cache128` as of 2026-06-10 — don't use until audited.
Second `generate_batch_prefill` in the same daemon exits — use fresh-process
per test case until repeated-prefill lifecycle is fixed.

### Relevant env vars

| Var | Default | Notes |
|-----|---------|-------|
| `HIPFIRE_QWEN35_PAGED_EXPERTS` | off | Set to `1` to enable expert paging |
| `HIPFIRE_QWEN35_EXPERT_CACHE_MB` | 8192 | Budget for resident expert modules |
| `HIPFIRE_QWEN35_EXPERT_CACHE_TRACE` | off | Set to `1` for eviction log |
| `HIPFIRE_PREFILL_MAX_BATCH` | 256 | Prefill scratch rows; leave unset for paged (live-row sizing) |

### Batch prefill (multi-session)

```json
{"type":"generate_batch_prefill","id":"bp1","batch_id":"b1","sessions":[
  {"id":"s1","tokens":[9906,11,220],"max_new_tokens":8},
  {"id":"s2","tokens":[9906,11,220],"max_new_tokens":8}
]}
```

## Probe the module table (397B)

Verifies HFQM v2 metadata-only load without touching expert payloads:
```bash
./scripts/probe-qwen35-expert-modules.sh \
  /home/sadara/.hipfire/models/qwen3.5-397b-a17b.mq6.hfqm2.hfq.tmp
```

## Coherence gate after kernel/dispatch changes

```bash
./scripts/coherence-gate-dflash.sh
```
Always run this after touching kernels, quant formats, or the paged-expert path.
