#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.

"""Offline behavioural tests for tools/benchlocal/sampling_proxy.mjs.

Stands up a stdlib echo upstream and drives the Node sampling proxy under
both THINKING_MODE values. No GPU, no hipfire runtime dependency.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
PROXY_JS = REPO / "tools" / "benchlocal" / "sampling_proxy.mjs"

# Real archived attestation row key sets.
# baseline: .../benchlocal-full-20260731T091750Z/results/a3b-mq4p/sampling-attestations.jsonl
# medium:   .../benchlocal-medium-20260731T194729Z/results/a3b-mq4p/sampling-attestations.jsonl
BASELINE_ATTESTATION_KEYS = frozenset(
    {
        "timestamp",
        "modelSlug",
        "path",
        "requestModel",
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "presence_penalty",
        "repetition_penalty",
    }
)
MEDIUM_ATTESTATION_KEYS = BASELINE_ATTESTATION_KEYS | frozenset(
    {
        "reasoning_effort",
        "enable_thinking",
        "expected_max_think_tokens",
    }
)

NODE = shutil.which("node")


def _skip_without_node() -> None:
    if NODE is None:
        raise unittest.SkipTest("node not on PATH")


class _EchoHandler(BaseHTTPRequestHandler):
    """Capture last request and echo the JSON body back."""

    last_method: str | None = None
    last_path: str | None = None
    last_body: bytes = b""
    last_headers: dict[str, str] = {}
    lock = threading.Lock()

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return

    def _read_body(self) -> bytes:
        """Read body for Content-Length or chunked Transfer-Encoding.

        The proxy strips hop-by-hop Content-Length on passthrough and may
        forward via chunked encoding when piping; the stub must accept both.
        """
        te = (self.headers.get("Transfer-Encoding") or "").lower()
        if "chunked" in te:
            chunks: list[bytes] = []
            while True:
                line = self.rfile.readline()
                if not line:
                    break
                size_str = line.strip().split(b";", 1)[0]
                try:
                    size = int(size_str, 16)
                except ValueError:
                    break
                if size == 0:
                    while True:
                        trailer = self.rfile.readline()
                        if trailer in (b"\r\n", b"\n", b""):
                            break
                    break
                data = self.rfile.read(size)
                chunks.append(data)
                self.rfile.read(2)  # CRLF after chunk
            return b"".join(chunks)
        length = int(self.headers.get("Content-Length", "0") or "0")
        return self.rfile.read(length) if length else b""

    def _capture(self) -> bytes:
        body = self._read_body()
        with self.lock:
            type(self).last_method = self.command
            type(self).last_path = self.path
            type(self).last_body = body
            type(self).last_headers = {k: v for k, v in self.headers.items()}
        return body

    def do_GET(self) -> None:  # noqa: N802
        body = self._capture()
        payload = body if body else b'{"ok":true,"echo":"get"}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self) -> None:  # noqa: N802
        body = self._capture()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_PUT(self) -> None:  # noqa: N802
        self.do_POST()


def _free_port() -> int:
    srv = HTTPServer(("127.0.0.1", 0), BaseHTTPRequestHandler)
    port = srv.server_address[1]
    srv.server_close()
    return int(port)


def _http_json(method: str, url: str, body: dict[str, Any] | None = None, timeout: float = 5.0) -> tuple[int, Any]:
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"} if data is not None else {},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            status = resp.getcode()
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        status = exc.code
    if not raw:
        return status, None
    try:
        return status, json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError:
        return status, raw.decode("utf-8", errors="replace")


def _wait_health(url: str, timeout_s: float = 8.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    last_err: Exception | None = None
    while time.monotonic() < deadline:
        try:
            status, payload = _http_json("GET", url, timeout=1.0)
            if status == 200 and isinstance(payload, dict) and payload.get("ok") is True:
                return payload
        except Exception as exc:  # noqa: BLE001 — poll until ready
            last_err = exc
        time.sleep(0.05)
    raise TimeoutError(f"proxy health not ready at {url} within {timeout_s}s; last_err={last_err!r}")


@unittest.skipUnless(NODE is not None, "node not on PATH")
class BenchlocalSamplingProxyTest(unittest.TestCase):
    def setUp(self) -> None:
        _skip_without_node()
        self.assertTrue(PROXY_JS.is_file(), f"missing proxy: {PROXY_JS}")
        self._tmpdir = tempfile.TemporaryDirectory(prefix="benchlocal-proxy-")
        self.tmp = Path(self._tmpdir.name)
        self.attestation = self.tmp / "attestations.jsonl"
        self.upstream_port = _free_port()
        self.proxy_port = _free_port()

        _EchoHandler.last_method = None
        _EchoHandler.last_path = None
        _EchoHandler.last_body = b""
        _EchoHandler.last_headers = {}

        self.upstream = HTTPServer(("127.0.0.1", self.upstream_port), _EchoHandler)
        self.upstream_thread = threading.Thread(target=self.upstream.serve_forever, daemon=True)
        self.upstream_thread.start()

        self.proxy_proc: subprocess.Popen[str] | None = None

    def tearDown(self) -> None:
        proc = self.proxy_proc
        if proc is not None:
            if proc.poll() is None:
                try:
                    proc.send_signal(signal.SIGTERM)
                    proc.wait(timeout=5)
                except Exception:  # noqa: BLE001
                    try:
                        proc.kill()
                        proc.wait(timeout=2)
                    except Exception:  # noqa: BLE001
                        pass
            for stream in (proc.stdout, proc.stderr):
                if stream is not None:
                    try:
                        stream.close()
                    except Exception:  # noqa: BLE001
                        pass
        self.proxy_proc = None
        try:
            self.upstream.shutdown()
        except Exception:  # noqa: BLE001
            pass
        try:
            self.upstream.server_close()
        except Exception:  # noqa: BLE001
            pass
        self._tmpdir.cleanup()

    def _start_proxy(self, thinking_mode: str, presence_penalty: str = "1.5") -> None:
        env = os.environ.copy()
        env.update(
            {
                "LISTEN_HOST": "127.0.0.1",
                "LISTEN_PORT": str(self.proxy_port),
                "TARGET_ORIGIN": f"http://127.0.0.1:{self.upstream_port}",
                "MODEL_SLUG": "test-route",
                "ATTESTATION_LOG": str(self.attestation),
                "PRESENCE_PENALTY": presence_penalty,
                "THINKING_MODE": thinking_mode,
            }
        )
        self.proxy_proc = subprocess.Popen(
            [NODE, str(PROXY_JS)],
            cwd=str(REPO),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            _wait_health(f"http://127.0.0.1:{self.proxy_port}/__sampling_proxy/health")
        except Exception:
            # Surface proxy stderr on failed startup.
            if self.proxy_proc.poll() is not None:
                err = self.proxy_proc.stderr.read() if self.proxy_proc.stderr else ""
                out = self.proxy_proc.stdout.read() if self.proxy_proc.stdout else ""
                self.fail(
                    f"proxy exited early rc={self.proxy_proc.returncode}\nstdout={out}\nstderr={err}"
                )
            raise

    def _post_chat(self, body: dict[str, Any]) -> tuple[int, Any]:
        return _http_json(
            "POST",
            f"http://127.0.0.1:{self.proxy_port}/v1/chat/completions",
            body=body,
        )

    def _assert_forced_sampling(self, received: dict[str, Any], presence: float = 1.5) -> None:
        self.assertEqual(received["temperature"], 1.0)
        self.assertEqual(received["top_p"], 0.95)
        self.assertEqual(received["top_k"], 20)
        self.assertEqual(received["min_p"], 0.0)
        self.assertEqual(received["repetition_penalty"], 1.0)
        self.assertEqual(received["presence_penalty"], presence)

    def _assert_attestation_row(self, thinking_mode: str) -> dict[str, Any]:
        self.assertTrue(self.attestation.is_file(), "attestation log missing")
        lines = [ln for ln in self.attestation.read_text(encoding="utf-8").splitlines() if ln.strip()]
        self.assertEqual(len(lines), 1, f"expected exactly one attestation row, got {len(lines)}")
        row = json.loads(lines[0])
        expected_keys = (
            MEDIUM_ATTESTATION_KEYS if thinking_mode == "medium" else BASELINE_ATTESTATION_KEYS
        )
        self.assertEqual(
            set(row.keys()),
            set(expected_keys),
            f"attestation keys mismatch for mode={thinking_mode}: {sorted(row.keys())}",
        )
        self.assertEqual(row["modelSlug"], "test-route")
        self.assertEqual(row["path"], "/v1/chat/completions")
        self.assertEqual(row["temperature"], 1)
        self.assertEqual(row["top_p"], 0.95)
        self.assertEqual(row["top_k"], 20)
        self.assertEqual(row["min_p"], 0)
        self.assertEqual(row["presence_penalty"], 1.5)
        self.assertEqual(row["repetition_penalty"], 1)
        self.assertIn("timestamp", row)
        if thinking_mode == "medium":
            self.assertEqual(row["reasoning_effort"], "medium")
            self.assertIs(row["enable_thinking"], True)
            self.assertEqual(row["expected_max_think_tokens"], 1024)
        return row

    def test_thinking_disabled_forces_sampling_and_no_reasoning(self) -> None:
        self._start_proxy("disabled")
        status, echoed = self._post_chat(
            {
                "model": "test-model",
                "messages": [{"role": "user", "content": "hi"}],
                "temperature": 0.3,
                "top_p": 0.5,
                # deliberately omit top_k / min_p / penalties
            }
        )
        self.assertEqual(status, 200)
        self.assertIsInstance(echoed, dict)
        # Stub echoes the rewritten upstream body.
        with _EchoHandler.lock:
            raw = _EchoHandler.last_body
            path = _EchoHandler.last_path
        self.assertEqual(path, "/v1/chat/completions")
        received = json.loads(raw.decode("utf-8"))
        self._assert_forced_sampling(received)
        ctk = received.get("chat_template_kwargs") or {}
        self.assertIsInstance(ctk, dict)
        self.assertIs(ctk.get("enable_thinking"), False)
        self.assertNotIn("reasoning_effort", received)
        self.assertNotIn("max_think_tokens", received)
        self._assert_attestation_row("disabled")
        self.assertEqual(received.get("model"), "test-model")

    def test_thinking_medium_forces_sampling_and_reasoning(self) -> None:
        self._start_proxy("medium")
        status, echoed = self._post_chat(
            {
                "model": "test-model-med",
                "messages": [{"role": "user", "content": "hi"}],
                "temperature": 0.3,
                "top_p": 0.5,
            }
        )
        self.assertEqual(status, 200)
        self.assertIsInstance(echoed, dict)
        with _EchoHandler.lock:
            raw = _EchoHandler.last_body
            path = _EchoHandler.last_path
        self.assertEqual(path, "/v1/chat/completions")
        received = json.loads(raw.decode("utf-8"))
        self._assert_forced_sampling(received)
        ctk = received.get("chat_template_kwargs") or {}
        self.assertIsInstance(ctk, dict)
        self.assertIs(ctk.get("enable_thinking"), True)
        self.assertEqual(received.get("reasoning_effort"), "medium")
        # Archive: max_think_tokens never goes on the wire body.
        self.assertNotIn("max_think_tokens", received)
        self._assert_attestation_row("medium")

    def test_non_chat_path_passthrough_unmodified(self) -> None:
        self._start_proxy("disabled")
        original = {"ping": True, "temperature": 0.3, "n": 7}
        status, echoed = _http_json(
            "POST",
            f"http://127.0.0.1:{self.proxy_port}/v1/models",
            body=original,
        )
        self.assertEqual(status, 200)
        with _EchoHandler.lock:
            raw = _EchoHandler.last_body
            path = _EchoHandler.last_path
            method = _EchoHandler.last_method
        self.assertEqual(method, "POST")
        self.assertEqual(path, "/v1/models")
        received = json.loads(raw.decode("utf-8"))
        self.assertEqual(received, original)
        # Passthrough must not write attestations.
        if self.attestation.exists():
            content = self.attestation.read_text(encoding="utf-8").strip()
            self.assertEqual(content, "")
        self.assertEqual(echoed, original)

    def test_presence_penalty_from_env(self) -> None:
        self._start_proxy("disabled", presence_penalty="1.25")
        status, _ = self._post_chat(
            {
                "model": "m",
                "messages": [{"role": "user", "content": "x"}],
                "temperature": 0.1,
            }
        )
        self.assertEqual(status, 200)
        with _EchoHandler.lock:
            received = json.loads(_EchoHandler.last_body.decode("utf-8"))
        self.assertEqual(received["presence_penalty"], 1.25)


if __name__ == "__main__":
    unittest.main()
