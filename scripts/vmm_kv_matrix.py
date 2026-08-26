#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.

"""GPU lifecycle matrix for static and adaptive Qwen VMM KV caches.

Exercises every supported static K/V combination plus all adaptive presets
through long prefill, AR graph capture/replay, reset, unload, and reload.
This is functional coverage, not a performance benchmark.
"""

import argparse
import json
import os
import queue
import re
import signal
import subprocess
import threading
import time
from pathlib import Path

STATIC_Q8 = [(mode, None) for mode in ("q8", "asym2", "asym3", "asym4", "fwht2", "fwht3", "fwht4")]
STATIC_LLOYD = [(k, v) for k in ("fwht2", "fwht3", "fwht4") for v in ("lloyd2", "lloyd3", "lloyd4")]
ADAPTIVE_EXPECTED_STEPS = {"conservative": 1, "balanced": 3, "aggressive": 4}
REPO = Path(__file__).resolve().parent.parent


class Daemon:
    def __init__(self, binary, env, stderr_path, timeout):
        self.timeout = timeout
        self.stderr_path = Path(stderr_path)
        self.stderr_handle = self.stderr_path.open("w", encoding="utf-8")
        self.proc = subprocess.Popen(
            [binary], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=self.stderr_handle, text=True, bufsize=1, env=env, cwd=REPO,
            start_new_session=True,
        )
        self.rows = []
        self.q = queue.Queue()
        self.reader = threading.Thread(target=self._read_stdout, daemon=True)
        self.reader.start()

    def _read_stdout(self):
        assert self.proc.stdout is not None
        for raw in self.proc.stdout:
            line = raw.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                self.q.put(RuntimeError(f"non-JSON daemon stdout: {line!r}: {exc}"))
                continue
            self.rows.append(row)
            self.q.put(row)
        self.q.put(EOFError(f"daemon stdout closed (rc={self.proc.poll()})"))

    def send(self, row):
        if self.proc.poll() is not None:
            raise RuntimeError(f"daemon exited before write (rc={self.proc.returncode})")
        assert self.proc.stdin is not None
        self.proc.stdin.write(json.dumps(row, separators=(",", ":")) + "\n")
        self.proc.stdin.flush()

    def wait_for(self, wanted, timeout=None):
        deadline = time.monotonic() + (timeout or self.timeout)
        seen = []
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"timed out waiting for {wanted}; seen={seen[-5:]}")
            try:
                item = self.q.get(timeout=min(remaining, 1.0))
            except queue.Empty:
                if self.proc.poll() is not None:
                    raise RuntimeError(f"daemon exited waiting for {wanted} (rc={self.proc.returncode})")
                continue
            if isinstance(item, BaseException):
                raise item
            seen.append(item)
            typ = item.get("type")
            if typ == "error":
                raise RuntimeError(f"daemon error waiting for {wanted}: {item}")
            if typ in wanted:
                return item, seen

    def load(self, model, mode, adaptive, max_seq):
        self.send({
            "type": "load", "model": model,
            "params": {
                "max_seq": max_seq, "kv_mode": mode, "kv_backend": "vmm",
                "kv_adaptive": adaptive, "dflash_mode": "off", "mtp_mode": "off",
                "tp": 1, "pp": 1,
            },
        })
        row, _ = self.wait_for({"loaded"})
        if row.get("cache_capable") is False:
            raise AssertionError(f"load did not report cache_capable: {row}")
        return row

    def generate(self, request_id, prompt, max_tokens):
        self.send({
            "type": "generate", "id": request_id, "prompt": prompt,
            "temperature": 0.0, "temp": 0.0, "max_tokens": max_tokens,
        })
        row, seen = self.wait_for({"done"}, timeout=self.timeout)
        emitted = sum(1 for event in seen if event.get("type") == "token")
        done_tokens = row.get("tokens", row.get("completion_tokens", emitted))
        if not isinstance(done_tokens, int) or done_tokens <= 0:
            raise AssertionError(f"generation emitted no tokens: done={row}, events={emitted}")
        finish = row.get("finish_reason")
        if finish == "aborted":
            raise AssertionError(f"generation aborted: {row}")
        return row, emitted

    def reset(self):
        self.send({"type": "reset"})
        row, _ = self.wait_for({"reset"})
        if row.get("seq_pos") not in (None, 0):
            raise AssertionError(f"reset did not return seq_pos=0: {row}")
        return row

    def unload(self):
        self.send({"type": "unload"})
        row, _ = self.wait_for({"unloaded"})
        return row

    def close(self):
        try:
            if self.proc.stdin:
                self.proc.stdin.close()
        except OSError:
            pass
        if self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
                self.proc.wait(timeout=5)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(self.proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        self.stderr_handle.close()

    def stderr(self):
        self.stderr_handle.flush()
        return self.stderr_path.read_text(encoding="utf-8", errors="replace")


def base_env(gpu, v_mode):
    env = os.environ.copy()
    env.update({
        "HIP_VISIBLE_DEVICES": str(gpu),
        "HIPFIRE_GRAPH": "1",
        "HIPFIRE_AR_GRAPH": "1",
        "HIPFIRE_AR_GRAPH_TRACE": "1",
        "HIPFIRE_CASK_OFF": "1",
        "HIPFIRE_DFLASH_DRAFT": "",
        "HIPFIRE_KV_ADAPTIVE": "off",
    })
    if v_mode:
        env["HIPFIRE_KV_V"] = v_mode
    else:
        env.pop("HIPFIRE_KV_V", None)
    return env


def marker_snapshot(text):
    mapped = [int(x) for x in re.findall(r"mapped_prefix=(\d+)", text)]
    static_configs = [
        (mode.lower(), v_mode.lower())
        for mode, v_mode in re.findall(
            r"KV cache:\s*(Q8|Asym2|Asym3|Asym4|Fwht2|Fwht3|Fwht4)\s+vmm\s*"
            r"\([^\n]*?\bV\s+(Q8|lloyd2|lloyd3|lloyd4)\b",
            text,
            re.I,
        )
    ]
    graph_captures = re.findall(r"^\[qwen-ar-graph\] capture\b.*$", text, re.M)
    graph_replays = re.findall(r"^\[qwen-ar-graph\] replay\b.*$", text, re.M)
    return {
        "vmm": bool(re.search(r"KV cache:.*\bvmm\b", text, re.I)),
        "static_vmm": bool(re.search(r"KV cache:\s*(?!adaptive\b)[^\n]*\bvmm\b", text, re.I)),
        "static_configs": static_configs,
        "adaptive_vmm": bool(re.search(r"KV cache:\s*adaptive vmm", text, re.I)),
        "adaptive_engaged": "[adaptive-kv] engaged:" in text,
        "downshift_lines": re.findall(r"^.*\[adaptive-kv\] downshift.*$", text, re.M),
        "mapped_prefixes": mapped,
        "graph_captures": graph_captures,
        "graph_replays": graph_replays,
        "graph_lines": graph_captures + graph_replays,
    }


def assert_ar_graph_markers(markers):
    captures = len(markers["graph_captures"])
    replays = len(markers["graph_replays"])
    if captures == 0 or replays == 0:
        raise AssertionError(
            f"missing AR graph proof: captures={captures}, replays={replays}"
        )
    return {"graph_captures": captures, "graph_replays": replays}


def assert_static_markers(markers, prompt_tokens, mode, v_mode):
    if not markers["static_vmm"]:
        raise AssertionError("missing current-process static 'KV cache: … vmm' marker")
    expected_config = (mode.lower(), (v_mode or "q8").lower())
    if expected_config not in markers["static_configs"]:
        raise AssertionError(
            f"requested KV config {expected_config} not loaded; "
            f"observed={markers['static_configs']}"
        )
    if not markers["mapped_prefixes"]:
        raise AssertionError("static VMM load did not report mapped_prefix")
    if prompt_tokens is None:
        raise AssertionError("generation did not report prefill token count")
    first = markers["mapped_prefixes"][0]
    if prompt_tokens <= first:
        raise AssertionError(f"prefill {prompt_tokens} did not cross initial mapped prefix {first}")
    graph_proof = assert_ar_graph_markers(markers)
    label = f"{mode}/{v_mode or 'q8'}"
    return {
        "label": label,
        "loaded_config": expected_config,
        "growth_boundary_proved": True,
        **graph_proof,
    }


def run_static(args, mode, v_mode, outdir):
    label = f"static-{mode}-{v_mode or 'q8'}"
    stderr_path = outdir / f"{label}.stderr.log"
    daemon = Daemon(args.daemon, base_env(args.gpu, v_mode), stderr_path, args.timeout)
    result = {"label": label, "kind": "static", "mode": mode, "v_mode": v_mode or "q8"}
    try:
        result["load"] = daemon.load(args.model, mode, "off", args.max_seq)
        long_prompt = "x " * args.static_prompt_repetitions
        done, emitted = daemon.generate(f"{label}-long", long_prompt, args.max_tokens)
        result["long_done"] = done
        result["long_emitted_events"] = emitted
        result["unload"] = daemon.unload()
        result["reload"] = daemon.load(args.model, mode, "off", args.max_seq)
        short_done, short_emitted = daemon.generate(f"{label}-reload", "Say only: VMM reload ok", 8)
        result["reload_done"] = short_done
        result["reload_emitted_events"] = short_emitted
        result["final_unload"] = daemon.unload()
        markers = marker_snapshot(daemon.stderr())
        result["markers"] = markers
        prompt_tokens = done.get("prefill_tokens", done.get("prompt_tokens"))
        result["proof"] = assert_static_markers(markers, prompt_tokens, mode, v_mode)
        result["ok"] = True
    finally:
        result["stdout_rows"] = daemon.rows
        daemon.close()
    return result


def run_adaptive(args, preset, outdir):
    label = f"adaptive-{preset}"
    stderr_path = outdir / f"{label}.stderr.log"
    daemon = Daemon(args.daemon, base_env(args.gpu, None), stderr_path, args.timeout)
    result = {"label": label, "kind": "adaptive", "mode": "fwht4", "preset": preset}
    try:
        result["load"] = daemon.load(args.model, "fwht4", preset, args.max_seq)
        long_prompt = "x " * args.adaptive_prompt_repetitions
        done, emitted = daemon.generate(f"{label}-long", long_prompt, args.max_tokens)
        result["long_done"] = done
        result["long_emitted_events"] = emitted
        result["reset"] = daemon.reset()
        reset_done, reset_emitted = daemon.generate(f"{label}-reset", "Say only: adaptive reset ok", 8)
        result["reset_done"] = reset_done
        result["reset_emitted_events"] = reset_emitted
        result["unload"] = daemon.unload()
        result["reload"] = daemon.load(args.model, "fwht4", preset, args.max_seq)
        reload_done, reload_emitted = daemon.generate(f"{label}-reload", "Say only: adaptive reload ok", 8)
        result["reload_done"] = reload_done
        result["reload_emitted_events"] = reload_emitted
        result["final_unload"] = daemon.unload()
        markers = marker_snapshot(daemon.stderr())
        result["markers"] = markers
        if not markers["adaptive_vmm"] or not markers["adaptive_engaged"]:
            raise AssertionError("missing adaptive VMM/engagement load markers")
        expected = ADAPTIVE_EXPECTED_STEPS[preset]
        if len(markers["downshift_lines"]) < expected:
            raise AssertionError(
                f"{preset} expected >= {expected} downshift steps, saw {len(markers['downshift_lines'])}"
            )
        prompt_tokens = done.get("prefill_tokens", done.get("prompt_tokens"))
        if not markers["mapped_prefixes"]:
            raise AssertionError("adaptive VMM load did not report mapped_prefix")
        if prompt_tokens is None:
            raise AssertionError("adaptive generation did not report prefill token count")
        if prompt_tokens <= markers["mapped_prefixes"][0]:
            raise AssertionError("adaptive long prefill did not cross initial mapped prefix")
        graph_proof = assert_ar_graph_markers(markers)
        result["proof"] = {
            "expected_steps": expected,
            "observed_steps": len(markers["downshift_lines"]),
            **graph_proof,
        }
        result["ok"] = True
    finally:
        result["stdout_rows"] = daemon.rows
        daemon.close()
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--daemon", default=str(REPO / "target/release/daemon"))
    ap.add_argument("--model", default=str(Path("~/.hipfire/models/qwen3.5-9b.mq4").expanduser()))
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", default=f"/tmp/vmm-kv-matrix-{int(time.time())}")
    ap.add_argument("--max-seq", type=int, default=16384)
    ap.add_argument("--static-prompt-repetitions", type=int, default=9000)
    ap.add_argument("--adaptive-prompt-repetitions", type=int, default=13000)
    ap.add_argument("--max-tokens", type=int, default=12)
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--only", action="append", help="run labels containing this substring")
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    summary_path = outdir / "summary.json"
    results = []
    failures = []

    cells = [("static", mode, v) for mode, v in STATIC_Q8 + STATIC_LLOYD]
    cells += [("adaptive", preset, None) for preset in ADAPTIVE_EXPECTED_STEPS]
    for kind, a, b in cells:
        label = f"static-{a}-{b or 'q8'}" if kind == "static" else f"adaptive-{a}"
        if args.only and not any(part in label for part in args.only):
            continue
        print(f"=== {label} ===", flush=True)
        try:
            row = run_static(args, a, b, outdir) if kind == "static" else run_adaptive(args, a, outdir)
            results.append(row)
            print(json.dumps({"label": label, "ok": True, "proof": row.get("proof")}), flush=True)
        except BaseException as exc:
            failures.append({"label": label, "error": repr(exc)})
            print(json.dumps({"label": label, "ok": False, "error": repr(exc)}), flush=True)
        summary_path.write_text(json.dumps({"results": results, "failures": failures}, indent=2), encoding="utf-8")

    print(json.dumps({"summary": str(summary_path), "passed": len(results), "failed": failures}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
