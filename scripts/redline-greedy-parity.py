#!/usr/bin/env python3
"""Production-serving greedy parity gate for Redline auto routing.

Runs identical requests through ordinary HIP and the automatic retained route.
Unlike the synthetic state shadow, this covers the real prefill -> first direct
decode -> retained takeover lifecycle and compares committed token streams.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def run_arm(
    daemon: Path,
    model: Path,
    backend: str,
    state_quant: str,
    prompt: str,
    max_tokens: int,
    max_seq: int,
    timeout: float,
) -> dict:
    env = dict(os.environ)
    env.update(
        HIPFIRE_REPLAY_BACKEND=backend,
        HIPFIRE_REPLAY_TRANSPORT="pm4",
        HIPFIRE_EMIT_TOKEN_IDS="1",
        HIPFIRE_AR_GRAPH="0",
        HIPFIRE_GRAPH="0",
    )
    messages = [
        {
            "type": "load",
            "model": str(model),
            "params": {
                "max_seq": max_seq,
                "kv_mode": "q8",
                "state_quant": state_quant,
                "dflash_mode": "off",
                "mtp_mode": "off",
                "dspark_mode": "off",
                "tp": 1,
                "pp": 1,
            },
        },
        {
            "type": "generate",
            "id": f"redline-{backend}-{state_quant}",
            "attempt_id": 1,
            "prompt": prompt,
            "temperature": 0.0,
            "top_p": 1.0,
            "repeat_penalty": 1.0,
            "max_tokens": max_tokens,
            "max_think_tokens": 1,
            "assistant_prefix": "closed_think",
        },
        {"type": "unload"},
    ]
    wire = "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in messages)
    proc = subprocess.run(
        [str(daemon)],
        input=wire,
        text=True,
        capture_output=True,
        env=env,
        timeout=timeout,
        check=False,
    )
    events = []
    for line in proc.stdout.splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    errors = [row for row in events if row.get("type") == "error"]
    done = next((row for row in reversed(events) if row.get("type") == "done"), None)
    committed = [row["tok_id"] for row in events if row.get("type") == "committed"]
    token_text = [row.get("text", "") for row in events if row.get("type") == "token"]
    stream = committed if committed else token_text
    visible = "".join(token_text)
    if proc.returncode != 0 or errors or done is None or not stream:
        raise RuntimeError(
            f"{backend}/{state_quant} failed: exit={proc.returncode} "
            f"errors={errors[:1]} done={done is not None} stream={len(stream)}\n"
            f"stderr tail:\n{proc.stderr[-4000:]}"
        )
    return {
        "backend": backend,
        "state_quant": state_quant,
        "stream": stream,
        "visible": visible,
        "done": done,
        "stderr_tail": proc.stderr[-4000:],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--daemon", type=Path, default=Path("target/release/examples/daemon")
    )
    parser.add_argument(
        "--prompt",
        default="Explain KV Cache in large-language-model inference with practical details.",
    )
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--max-seq", type=int, default=2048)
    parser.add_argument(
        "--state-quant", choices=("q8", "fp32"), action="append", dest="states"
    )
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    model = args.model.expanduser().resolve()
    daemon = args.daemon.expanduser().resolve()
    if not model.is_file() or not daemon.is_file():
        parser.error(f"missing model or daemon: model={model} daemon={daemon}")
    states = args.states or ["q8", "fp32"]
    report = {"model": str(model), "daemon": str(daemon), "rows": []}
    passed = True
    for state in states:
        hip = run_arm(
            daemon, model, "hip", state, args.prompt, args.max_tokens, args.max_seq, args.timeout
        )
        auto = run_arm(
            daemon, model, "auto", state, args.prompt, args.max_tokens, args.max_seq, args.timeout
        )
        exact = hip["stream"] == auto["stream"]
        coherent = len(auto["visible"].strip()) >= 8
        row = {
            "state_quant": state,
            "exact": exact,
            "coherent": coherent,
            "tokens": len(auto["stream"]),
            "hip_visible": hip["visible"],
            "auto_visible": auto["visible"],
        }
        report["rows"].append(row)
        passed &= exact and coherent
        print(
            f"state={state} exact={exact} coherent={coherent} tokens={len(auto['stream'])}",
            flush=True,
        )
        if not exact:
            first = next(
                (
                    i
                    for i, (left, right) in enumerate(zip(hip["stream"], auto["stream"]))
                    if left != right
                ),
                min(len(hip["stream"]), len(auto["stream"])),
            )
            print(f"first divergence={first}", flush=True)
    report["pass"] = passed
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
