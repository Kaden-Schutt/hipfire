# DS4 MI300X CDNA test-failure evidence

This directory is the durable evidence bundle for the abandoned DeepSeek V4
Flash MQ2R gfx942 product-port attempt of 2026-08-01. It is intentionally
archived on branch `ds4-cdna-test-fail`; it is not production evidence and is
not a candidate for merging into the gfx1151 line.

## Contents

- `ledger.jsonl`: chronological campaign decisions, measurements, retractions,
  and closeout rows.
- `raw/`: byte-for-byte regular-file copy of
  `/mnt/scratch/hipfire-evidence/ds4-mi300x-agentmaxx/` from host `mi300x`.
- `MANIFEST.sha256`: integrity hashes for the ledger and raw evidence.

The raw bundle includes logs, JSON reports, source/patch snapshots, isolated
microbench results, and the small JIT-cache snapshots that were already stored
inside the evidence root. It excludes the 82 GB model, the 167 GB source model,
Cargo build trees, the general kernel cache, and the remote source checkout.
The non-regular FIFO `raw/a1-m0/04-daemon.fifo` was intentionally skipped by
the transfer.

## Interpretation guardrails

- The exact product artifact SHA-256 was
  `392325b5a8cd284c8f305f23f74f178007a14b88173babeb3f4784ec4fc0e511`.
- The best repo-native result was 32.1931589537 tok/s, ordinary AR, on a
  one-run 2048-prompt/32-output discovery fixture.
- That number is not a 2048/510 acceptance result and was not subjected to the
  full byte-identical correctness battery.
- Earlier custom-feeder model-level screens are diagnostic only; the ledger
  explicitly retracts them as product evidence.
- This archive documents a failed port. It does not establish supported or
  optimized DS4 execution on gfx942.

See
[`../../2026-08-01-ds4-mi300x-cdna-test-failure-postmortem.md`](../../2026-08-01-ds4-mi300x-cdna-test-failure-postmortem.md)
for the condensed analysis.
