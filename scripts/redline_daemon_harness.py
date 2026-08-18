#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.

"""Resident daemon phase harness for the default-off Redline graft.

Loads one model once, measures synthetic prefill and single-token decode
separately, and asks the daemon to delimit/fingerprint exactly one HIP launch
sequence for each phase. The optional DSpark verify oracle additionally lowers
one isolated fixed-B verify body and compares ordinary HIP, captured HIP, and
retained PM4 state without installing that route into serving.
"""

import argparse
import json
import os
import select
import statistics
import subprocess
import sys
import time
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent


class Daemon:
    def __init__(self, binary: Path, log_path: Path, timeout_s: float, kv_mode: str):
        self.timeout_s = timeout_s
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log = log_path.open("w")
        env = dict(os.environ)
        env.update(
            HIPFIRE_REPLAY_BACKEND="shadow",
            HIPFIRE_REPLAY_MANUAL_CAPTURE="1",
            HIPFIRE_KV_MODE=kv_mode,
            HIPFIRE_CASK_OFF="1",
            HIPFIRE_AR_GRAPH="0",
            HIPFIRE_GRAPH="0",
        )
        self.proc = subprocess.Popen(
            [str(binary)],
            cwd=REPO,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.log,
            text=True,
            bufsize=1,
            start_new_session=True,
        )

    def request(self, message):
        if self.proc.poll() is not None:
            raise RuntimeError(f"daemon exited early with code {self.proc.returncode}")
        assert self.proc.stdin is not None
        assert self.proc.stdout is not None
        self.proc.stdin.write(json.dumps(message, separators=(",", ":")) + "\n")
        self.proc.stdin.flush()
        ready, _, _ = select.select([self.proc.stdout], [], [], self.timeout_s)
        if not ready:
            raise TimeoutError(f"daemon response timed out after {self.timeout_s}s: {message['type']}")
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError(f"daemon closed while handling {message['type']}")
        response = json.loads(line)
        if response.get("type") == "error":
            raise RuntimeError(response.get("message", "daemon error"))
        return response

    def close(self):
        if self.proc.poll() is None:
            try:
                self.request({"type": "unload"})
            except Exception:
                pass
            try:
                self.proc.terminate()
                self.proc.wait(timeout=5)
            except Exception:
                self.proc.kill()
                self.proc.wait(timeout=5)
        self.log.close()


def summarize(values):
    return {
        "min": min(values),
        "median": statistics.median(values),
        "max": max(values),
    }


def capture_key(row):
    capture = row["redline_capture"]
    return (
        capture["launches"],
        capture["unique_kernels"],
        capture["sequence_hash"],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--daemon",
        default=str(REPO / "target/release/daemon"),
    )
    parser.add_argument("--out", default=str(REPO / ".redline-work/redline-daemon-phases.json"))
    parser.add_argument("--log", default=str(REPO / ".redline-work/redline-daemon-phases.log"))
    parser.add_argument("--prefill", type=int, nargs="+", default=[128, 512])
    parser.add_argument(
        "--skip-prefill",
        action="store_true",
        help="run only the decode capture, contract probe, and shadow parity gate",
    )
    parser.add_argument("--decode-context", type=int, default=128)
    parser.add_argument(
        "--kv-mode",
        choices=("q8", "fwht2", "fwht3", "fwht4"),
        default="q8",
        help="KV layout used by capture, shadow replay, and the HIP oracle",
    )
    parser.add_argument("--capture-repeats", type=int, default=2)
    parser.add_argument("--measure-repeats", type=int, default=5)
    parser.add_argument("--decode-iterations", type=int, default=100)
    parser.add_argument(
        "--shadow-iterations",
        type=int,
        default=1,
        help="consecutive token positions compared by the AQL/HIP/blob parity gate",
    )
    parser.add_argument(
        "--state-quant",
        choices=("q8", "fp32", "q4"),
        help=(
            "DeltaNet state precision for the run. Pass fp32 for any BYTE-PARITY "
            "claim: Q8 state uses stochastic rounding, which makes a bit-exact "
            "PM4/HIP path report exact=False and misattributes the failure to the "
            "lowering (see CLAUDE.md 'Byte-parity validation is meaningless under "
            "stochastic state'). Omit to keep the daemon default (q8)."
        ),
    )
    parser.add_argument("--max-seq", type=int, default=2048)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--prefix", type=int, help="compare only the first N captured launches")
    parser.add_argument(
        "--profile-prefix-step",
        type=int,
        help="profile retained-PM4 cumulative prefixes at this dispatch interval",
    )
    parser.add_argument(
        "--profile-prefix-repeats",
        type=int,
        default=3,
        help="GPU-timed repetitions per cumulative PM4 prefix (default: 3)",
    )
    parser.add_argument(
        "--profile-prefix-start",
        type=int,
        help="first cumulative dispatch prefix to profile (default: one step)",
    )
    parser.add_argument(
        "--profile-prefix-steady-state",
        action="store_true",
        help=(
            "prime the model once before retained-PM4 prefix timing instead of "
            "resetting and re-prefilling before every sample; diagnostic timing only"
        ),
    )
    parser.add_argument(
        "--pm4",
        action="store_true",
        help="lower --prefix to one retained PM4 indirect buffer",
    )
    parser.add_argument(
        "--dspark-verify-shadow",
        action="store_true",
        help=(
            "run the DSpark verify ordinary-HIP/capture-safe/blob/PM4 state oracle "
            "instead of the plain-AR phase harness"
        ),
    )
    parser.add_argument(
        "--draft",
        help="explicit DSpark sidecar path (required by --dspark-verify-shadow)",
    )
    parser.add_argument(
        "--verify-batch",
        type=int,
        default=3,
        help="fixed target verify batch for the DSpark shadow (default: 3)",
    )
    args = parser.parse_args()

    model = Path(args.model).expanduser().resolve()
    daemon_path = Path(args.daemon).expanduser().resolve()
    if not model.is_file():
        sys.exit(f"model not found: {model}")
    if not daemon_path.is_file():
        sys.exit(f"daemon not found: {daemon_path}")
    draft = Path(args.draft).expanduser().resolve() if args.draft else None
    if args.dspark_verify_shadow and (draft is None or not draft.is_file()):
        sys.exit("--dspark-verify-shadow requires an existing --draft sidecar")
    if args.dspark_verify_shadow:
        discovered_draft = model.with_name(f"{model.stem}-dspark{model.suffix}").resolve()
        if draft != discovered_draft:
            sys.exit(
                "DeepSeek4 discovers DSpark only as the sibling "
                f"{discovered_draft}; --draft resolved to {draft}"
            )

    report = {
        "model": str(model),
        "model_bytes": model.stat().st_size,
        "draft": str(draft) if draft is not None else None,
        "draft_bytes": draft.stat().st_size if draft is not None else None,
        "daemon": str(daemon_path),
        "kv_mode": args.kv_mode,
        "automatic_clocks_required": True,
        "prefill": {},
        "decode": {},
    }
    daemon = Daemon(daemon_path, Path(args.log), args.timeout, args.kv_mode)
    try:
        load_params = {
            "max_seq": args.max_seq,
            "kv_mode": args.kv_mode,
            "dflash_mode": "off",
            "dspark_mode": "on" if args.dspark_verify_shadow else "off",
        }
        # DeepSeek4 discovers `<stem>-dspark.<ext>` itself. Passing the same file
        # through params.draft would incorrectly enter the Qwen DFlash lm_head
        # eligibility gate before architecture dispatch.
        loaded = daemon.request(
            {
                "type": "load",
                "model": str(model),
                "params": {
                    **load_params,
                    **(
                        {"state_quant": args.state_quant}
                        if args.state_quant
                        else {}
                    ),
                },
            }
        )
        if loaded.get("type") != "loaded":
            raise RuntimeError(f"unexpected load response: {loaded}")
        report["loaded"] = loaded
        print(
            f"loaded arch={loaded.get('arch')} dim={loaded.get('dim')} "
            f"layers={loaded.get('layers')} vocab={loaded.get('vocab')}",
            flush=True,
        )

        if args.dspark_verify_shadow:
            shadow = daemon.request(
                {
                    "type": "redline_dspark_shadow_pm4",
                    "context_tokens": args.decode_context,
                    "verify_batch": args.verify_batch,
                    "iterations": args.shadow_iterations,
                }
            )
            report["dspark_verify_shadow"] = shadow
            report["pass"] = bool(shadow.get("bit_exact"))
            print(
                f"dspark-verify-shadow: B={args.verify_batch} "
                f"positions={args.shadow_iterations} exact={report['pass']} "
                f"launches={shadow.get('capture', {}).get('launches')} "
                f"hash={shadow.get('capture', {}).get('sequence_hash')}",
                flush=True,
            )
            output = Path(args.out)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, indent=2) + "\n")
            print(f"report={output} pass={report['pass']}", flush=True)
            if not report["pass"]:
                raise SystemExit(1)
            return

        for tokens in ([] if args.skip_prefill else args.prefill):
            captures = [
                daemon.request(
                    {
                        "type": "bench_prefill",
                        "tokens": tokens,
                        "redline_capture": True,
                    }
                )
                for _ in range(args.capture_repeats)
            ]
            measures = [
                daemon.request({"type": "bench_prefill", "tokens": tokens})
                for _ in range(args.measure_repeats)
            ]
            stable = len({capture_key(row) for row in captures}) == 1
            report["prefill"][str(tokens)] = {
                "captures": captures,
                "sequence_stable": stable,
                "measurement": {
                    "tok_s": summarize([row["tok_s"] for row in measures]),
                    "ms": summarize([row["ms"] for row in measures]),
                    "runs": measures,
                },
            }
            cap = captures[0]["redline_capture"]
            print(
                f"prefill{tokens}: stable={stable} launches={cap['launches']} "
                f"kernels={cap['unique_kernels']} hash={cap['sequence_hash']} "
                f"median={report['prefill'][str(tokens)]['measurement']['tok_s']['median']:.1f} tok/s",
                flush=True,
            )

        captures = [
            daemon.request(
                {
                    "type": "bench_decode",
                    "context_tokens": args.decode_context,
                    "iterations": 1,
                    "redline_capture": True,
                    "redline_detail": True,
                }
            )
            for _ in range(args.capture_repeats)
        ]
        measures = [
            daemon.request(
                {
                    "type": "bench_decode",
                    "context_tokens": args.decode_context,
                    "iterations": args.decode_iterations,
                }
            )
            for _ in range(args.measure_repeats)
        ]
        stable = len({capture_key(row) for row in captures}) == 1
        report["decode"] = {
            "context_tokens": args.decode_context,
            "capture_iterations": 1,
            "captures": captures,
            "sequence_stable": stable,
            "measurement_iterations": args.decode_iterations,
            "measurement": {
                "tok_s": summarize([row["tok_s"] for row in measures]),
                "us_per_token": summarize([row["us_per_token"] for row in measures]),
                "runs": measures,
            },
        }
        cap = captures[0]["redline_capture"]
        print(
            f"decode: stable={stable} launches={cap['launches']} "
            f"kernels={cap['unique_kernels']} hash={cap['sequence_hash']} "
            f"median={report['decode']['measurement']['tok_s']['median']:.1f} tok/s",
            flush=True,
        )
        report["aql_contract_probe"] = daemon.request({"type": "redline_probe_aql"})
        print(
            f"aql-contracts: kernels={report['aql_contract_probe']['kernels']}",
            flush=True,
        )
        if args.profile_prefix_step is not None:
            report["pm4_prefix_profile"] = daemon.request(
                {
                    "type": "redline_pm4_prefix_profile",
                    "context_tokens": args.decode_context,
                    "step": args.profile_prefix_step,
                    "repeats": args.profile_prefix_repeats,
                    "steady_state": args.profile_prefix_steady_state,
                    **(
                        {"start": args.profile_prefix_start}
                        if args.profile_prefix_start is not None
                        else {}
                    ),
                }
            )
            print(
                f"pm4-prefix-profile: rows={len(report['pm4_prefix_profile']['rows'])} "
                f"repeats={args.profile_prefix_repeats} "
                f"steady_state={args.profile_prefix_steady_state}",
                flush=True,
            )
        if args.prefix is None:
            report["aql_shadow"] = daemon.request(
                {
                    "type": "redline_shadow_pm4" if args.pm4 else "redline_shadow_aql",
                    "context_tokens": args.decode_context,
                    "iterations": args.shadow_iterations,
                }
            )
            shadow_pass = report["aql_shadow"]["bit_exact"]
            print(
                f"shadow: backend={'pm4_ib' if args.pm4 else 'aql_packets'} "
                f"exact={shadow_pass} "
                f"aql={report['aql_shadow']['aql_host_us']:.1f}us "
                f"hip={report['aql_shadow']['hip_host_us']:.1f}us",
                flush=True,
            )
        else:
            report["prefix_shadow"] = daemon.request(
                {
                    "type": "redline_prefix_shadow",
                    "context_tokens": args.decode_context,
                    "prefix": args.prefix,
                    "pm4": args.pm4,
                }
            )
            shadow_pass = report["prefix_shadow"]["equal"]
            print(
                f"prefix-shadow: backend={'pm4_ib' if args.pm4 else 'aql_packets'} "
                f"prefix={args.prefix} exact={shadow_pass} "
                f"differing={report['prefix_shadow']['differing']}",
                flush=True,
            )

        report["pass"] = all(
            row["sequence_stable"] for row in report["prefill"].values()
        ) and report["decode"]["sequence_stable"] \
            and report["aql_contract_probe"]["kernels"] > 0 \
            and shadow_pass
    finally:
        daemon.close()

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"report={output} pass={report['pass']}", flush=True)
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
