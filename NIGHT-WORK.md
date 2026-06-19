# NIGHT-WORK.md — jobs deferred to idle / quiet-box time

Jobs that are correct to run but were deferred because they fight foreground
work — chiefly the pre-commit perf gates on this gfx1151 **UMA APU**, where a
CPU-heavy quantize job starves GPU decode of shared memory bandwidth and trips a
**false** MQ4 speed regression (see `docs/methodology/perf-benchmarking.md` and
the daemon-family-seam plan's testing note). Run these when no one is committing
and the box is otherwise idle.

Format: `- [ ] job — why deferred — exact command — done-when`.

## Pending

- [ ] **qtip3 `bf16.hfq → qtip3` regen + artifact replacement** (added 2026-06-19)
  - Why deferred: ~45 min, CPU-saturating; blocks/false-fails every commit's
    speed gate while it runs. The qtip3 *code* (commit `f10adf6d`) already
    landed and passed its own gates — this is end-to-end format validation +
    replacing the broken `qwen3.5-4b-qtip3.hfq` (the eval sweep's lone failure:
    daemon panics at decode, stale 0.2.0 artifact).
  - Command:
    ```
    ./target/release/hipfire-quantize \
      --input ~/.hipfire/models/qwen3.5-4b-bf16.hfq \
      --output ~/.hipfire/models/qwen3.5-4b-qtip3.new.hfq \
      --format qtip3
    ```
  - Done-when: the run prints a sane rotated-frame `decode max-abs-err` (≪ the
    1% damping floor), then `mv …qtip3.new.hfq …qtip3.hfq` and re-run the eval
    cell to confirm it is no longer 1-pass/22-fail:
    ```
    ./target/release/hipfire-eval --models qwen3.5-4b-qtip3 --tier medium
    ```

## How to use this file

- Add a job here instead of running it inline whenever it would contend with
  commit gates, a perf measurement, or another GPU/CPU-heavy task.
- When you run one, paste the result summary under it, check the box, and move
  it to a `## Done` section (or delete it once the outcome is committed
  elsewhere). Keep `## Pending` to genuinely-actionable items only.
