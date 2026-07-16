# Copyright (c) Kaden Schutt
import base64
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from urllib.error import HTTPError

import pytest
import autoresearch.ar.review.inference as inference_module

from autoresearch.ar.review.capsule import build_review_capsule
from autoresearch.ar.review.inference import (
    BoundedHttpTransport,
    HttpRequest,
    HttpResponse,
    ToollessReviewAdapter,
    ToollessInferenceError,
)
from autoresearch.ar.review.config import (
    AuthenticatedConfigSource,
    ReviewConfiguration,
    configuration_source_digest,
    load_review_configuration,
)
from autoresearch.ar.review.github import GitHubClient
from autoresearch.ar.review.models import ReviewTarget


TARGET = ReviewTarget("owner/repo", 42, "fork/repo", "head", "main", "base", "merge")
POLICY = {
    "schema": "hipfire.agentic-review.providers",
    "version": 1,
    "providers": [{
        "id": "review-adapter",
        "adapter_id": "openai-compatible",
        "adapter_version": "1",
        "endpoint": "https://provider.example.invalid/v1/review",
        "model": "review-model-v1",
        "api_key_env": "REVIEW_API_KEY",
        "max_requests": 1,
        "request_deadline_seconds": 30,
        "max_capsule_bytes": 1 << 20,
        "max_response_bytes": 1 << 20,
        "max_tokens": 128,
        "max_cost_usd": 5.0,
    }],
}
ROOT = Path(__file__).parents[3]
_CONFIGURATION = None
_LIVE_CLIENT = None
_LIVE_RUNNER = None
X_OID = hashlib.sha1(b"blob 6\0x = 1\n").hexdigest()


def protected_configuration(policy=None):
    global _CONFIGURATION, _LIVE_CLIENT, _LIVE_RUNNER
    if policy is None and _CONFIGURATION is not None:
        return _CONFIGURATION
    root = Path(tempfile.mkdtemp())
    config_dir = root / ".github" / "agentic-review"
    config_dir.mkdir(parents=True)
    (config_dir / "providers.json").write_text(json.dumps(policy or POLICY), encoding="utf-8")
    for name in ("capabilities-v1.json", "trusted-publishers.json"):
        shutil.copy(ROOT / ".github" / "agentic-review" / name, config_dir / name)
    contents = tuple((config_dir / name).read_bytes() for name in (
        "providers.json", "capabilities-v1.json", "trusted-publishers.json",
    ))
    blob_ids = [hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest() for content in contents]
    paths = (
        ".github/agentic-review/providers.json",
        ".github/agentic-review/capabilities-v1.json",
        ".github/agentic-review/trusted-publishers.json",
    )
    header = "HTTP/2 200\r\nX-OAuth-Scopes: read:user\r\n\r\n"
    responses = [
        {"id": 1, "full_name": "owner/repo", "default_branch": "main"},
        {"ref": "refs/heads/main", "object": {"sha": "c" * 40, "type": "commit"}},
        {"sha": "c" * 40, "tree": {"sha": "t" * 40}},
        {"sha": "t" * 40, "tree": [
            {"path": path, "mode": "100644", "type": "blob", "sha": oid}
            for path, oid in zip(paths, blob_ids)
        ], "truncated": False},
    ]
    responses.extend({"sha": oid, "encoding": "base64", "content": base64.b64encode(content).decode(), "size": len(content)}
                     for oid, content in zip(blob_ids, contents))

    class Runner:
        def __init__(self):
            self.responses = list(responses)

        def __call__(self, argv, input_data=None):
            payload = self.responses.pop(0)
            return subprocess.CompletedProcess(argv, 0, header + json.dumps(payload), "")

    source = GitHubClient(Runner()).authenticated_config_source(
        "owner/repo", commit_sha="c" * 40, repository_root=str(root)
    )
    loaded = load_review_configuration(root, source=source)
    if policy is None:
        _CONFIGURATION = loaded
        class LiveRunner:
            def __init__(self):
                self.head = "c" * 40

            def __call__(self, argv, input_data=None):
                path = argv[-1].split("?", 1)[0]
                if "/git/ref/heads/" in path:
                    payload = {"ref": "refs/heads/main", "object": {"sha": self.head, "type": "commit"}}
                else:
                    payload = {"id": 1, "full_name": "owner/repo", "default_branch": "main"}
                return subprocess.CompletedProcess(argv, 0, header + json.dumps(payload), "")

        _LIVE_RUNNER = LiveRunner()
        _LIVE_CLIENT = GitHubClient(_LIVE_RUNNER)
    return loaded


def capsule():
    class Client:
        def get_commit(self, repository, sha):
            tree_sha = "merge-tree" if sha == "merge" else "head-tree"
            return type("Response", (), {"data": {"sha": sha, "tree": {"sha": tree_sha}}})()

        def get_tree(self, repository, sha, *, recursive=False):
            entries = [] if sha == "merge-tree" else [{"path": "x.py", "mode": "100644", "type": "blob", "sha": X_OID}]
            return type("Response", (), {"data": {"sha": sha, "tree": entries, "truncated": False}})()

        def get_blob(self, repository, sha):
            return type("Response", (), {"data": {"sha": sha, "encoding": "base64", "content": base64.b64encode(b"x = 1\n").decode(), "size": 6}})()

    return build_review_capsule(Client(), TARGET)


class _ProviderResponse:
    def __init__(self, response):
        self.status = response.status_code
        self.headers = response.headers
        self._body = response.body
        self._read = False
        self.read_timeout = None

    def settimeout(self, timeout):
        self.read_timeout = timeout

    def read(self, size):
        if self._read:
            return b""
        self._read = True
        return self._body


class _Opener:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def open(self, request, timeout):
        self.calls.append(request)
        return _ProviderResponse(self.response)


_OPEN_OPENER = _Opener(None)


@pytest.fixture(autouse=True)
def patch_owned_transport(monkeypatch):
    global _OPEN_OPENER
    _OPEN_OPENER = _Opener(None)
    monkeypatch.setattr(inference_module, "build_opener", lambda handler: _OPEN_OPENER)


def Transport(response):
    _OPEN_OPENER.response = response
    _OPEN_OPENER.calls = []
    transport = BoundedHttpTransport()
    transport.calls = _OPEN_OPENER.calls
    return transport


def valid_response(**changes):
    content = {"verdict": "clean", "findings": []}
    value = {"choices": [{"index": 0, "message": {"role": "assistant", "content": json.dumps(content)}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}, "cost_usd": 0.01}
    if "verdict" in changes or "findings" in changes:
        content.update({key: changes.pop(key) for key in tuple(changes) if key in {"verdict", "findings"}})
        value["choices"][0]["message"]["content"] = json.dumps(content)
    value.update(changes)
    return HttpResponse(200, {"content-type": "application/json"}, json.dumps(value).encode())


def adapter(transport):
    configuration = protected_configuration()
    return ToollessReviewAdapter.from_configuration(
        configuration, "review-adapter", transport, {"REVIEW_API_KEY": "secret"}, _LIVE_CLIENT
    )


def configured_adapter(configuration, transport, environment, provider_id="review-adapter"):
    return ToollessReviewAdapter.from_configuration(
        configuration, provider_id, transport, environment, _LIVE_CLIENT
    )


def test_exactly_one_toolless_https_request_and_bound_proposal():
    transport = Transport(valid_response())
    proposal = adapter(transport).review(capsule())

    assert proposal.target == TARGET
    assert proposal.capsule_digest.startswith("sha256:")
    assert proposal.adapter_id == "openai-compatible"
    assert proposal.adapter_version == "1"
    assert proposal.model == "review-model-v1"
    assert proposal.response_digest.startswith("sha256:")
    assert len(transport.calls) == 1
    request = transport.calls[0]
    assert (request.get_method(), request.full_url) == ("POST", POLICY["providers"][0]["endpoint"])
    body = request.data.decode()
    assert '"tools":[]' in body
    assert "function" not in body.lower()
    request_json = json.loads(request.data)
    assert request_json["model"] == "review-model-v1"
    assert request_json["max_output_tokens"] == 128
    assert request_json["response_format"]["type"] == "json_schema"
    assert "x.py" in request_json["messages"][1]["content"]


def test_configuration_repository_must_match_capsule_target():
    configuration = protected_configuration()
    cross_source = replace(configuration.source, repository="other/repo")
    cross = replace(configuration, source=cross_source)
    with pytest.raises(ToollessInferenceError, match="repository|protected"):
        configured_adapter(cross, Transport(valid_response()), {"REVIEW_API_KEY": "secret"}).review(capsule())


def test_live_default_branch_advancement_invalidates_cached_configuration():
    configuration = protected_configuration()
    _LIVE_RUNNER.head = "d" * 40
    with pytest.raises(ToollessInferenceError, match="live|head|provenance"):
        configured_adapter(configuration, Transport(valid_response()), {"REVIEW_API_KEY": "secret"}).review(capsule())
    _LIVE_RUNNER.head = "c" * 40


def test_provider_selection_is_exact_and_empty_policy_fails_closed():
    with pytest.raises(ToollessInferenceError, match="provider"):
        ToollessReviewAdapter.from_configuration(ReviewConfiguration({"schema": POLICY["schema"], "version": 1, "providers": []}, {}, {}), "review-adapter", Transport(valid_response()), {"REVIEW_API_KEY": "secret"}, _LIVE_CLIENT)
    with pytest.raises(ToollessInferenceError, match="exact|configured"):
        configured_adapter(protected_configuration(), Transport(valid_response()), {"REVIEW_API_KEY": "secret"}, "review-adapter-extra")


def test_protected_configuration_is_deep_immutable_and_root_forgery_is_rejected():
    configuration = protected_configuration()
    with pytest.raises((TypeError, AttributeError)):
        configuration.providers["providers"].append({})
    with pytest.raises(TypeError):
        configuration.capabilities["capabilities"] = ()

    forged_root = Path(tempfile.mkdtemp())
    config_dir = forged_root / ".github" / "agentic-review"
    config_dir.mkdir(parents=True)
    (config_dir / "providers.json").write_text(json.dumps(POLICY), encoding="utf-8")
    for name in ("capabilities-v1.json", "trusted-publishers.json"):
        shutil.copy(ROOT / ".github" / "agentic-review" / name, config_dir / name)
    forged = load_review_configuration(forged_root, source=configuration.source)
    assert not forged.is_protected
    with pytest.raises(ToollessInferenceError, match="protected"):
        configured_adapter(forged, Transport(valid_response()), {"REVIEW_API_KEY": "secret"})


def test_caller_supplied_config_source_cannot_be_authenticated():
    source = AuthenticatedConfigSource(
        "owner/repo", "main", "c" * 40, "sha256:" + "a" * 64, "sha256:" + "b" * 64
    )
    assert not source.authenticated
    with pytest.raises(ValueError, match="GitHub boundary"):
        AuthenticatedConfigSource._from_authenticated_boundary(
            object(), "owner/repo", "main", "c" * 40, "sha256:" + "a" * 64, "/tmp"
        )


def test_provider_requires_protected_configuration_and_injected_non_github_environment():
    with pytest.raises(ToollessInferenceError, match="protected|loaded"):
        ToollessReviewAdapter.from_configuration(ReviewConfiguration(POLICY, {}, {}), "review-adapter", Transport(valid_response()), {"REVIEW_API_KEY": "secret"})
    with pytest.raises(ToollessInferenceError, match="GitHub|exactly"):
        ToollessReviewAdapter.from_configuration(
            protected_configuration(), "review-adapter", Transport(valid_response()),
            {"REVIEW_API_KEY": "secret", "GITHUB_TOKEN": "must-not-forward"}, _LIVE_CLIENT,
        )
    with pytest.raises(ToollessInferenceError, match="absent"):
        configured_adapter(protected_configuration(), Transport(valid_response()), {})
    unsupported = deepcopy(POLICY)
    unsupported["providers"][0]["adapter_id"] = "arbitrary-provider"
    with pytest.raises(ToollessInferenceError, match="supported"):
        configured_adapter(protected_configuration(unsupported), Transport(valid_response()), {"REVIEW_API_KEY": "secret"})
    unsupported["providers"][0]["adapter_id"] = "neutral-review"
    with pytest.raises(ToollessInferenceError, match="supported"):
        configured_adapter(protected_configuration(unsupported), Transport(valid_response()), {"REVIEW_API_KEY": "secret"})


@pytest.mark.parametrize(
    "response",
    [
        HttpResponse(302, {"location": "https://other.invalid"}, b""),
        HttpResponse(200, {"TrAnSfEr-EnCoDiNg": "chunked"}, b"{}"),
        HttpResponse(200, {"content-type": "application/json"}, b"{"),
        HttpResponse(200, {"content-type": "application/json"}, b'{"choices":[],"usage":{},"cost_usd":0,"extra":1}'),
    ],
)
def test_redirect_streaming_malformed_and_unknown_response_are_rejected(response):
    with pytest.raises(ToollessInferenceError):
        adapter(Transport(response)).review(capsule())


def test_transport_rejects_redirect_flag_and_enforces_response_limit_before_download():
    redirected = Transport(HttpResponse(302, {"Location": "https://other.invalid"}, b"{}"))
    with pytest.raises(ToollessInferenceError, match="redirect|status"):
        adapter(redirected).review(capsule())

    bounded = Transport(HttpResponse(200, {"Content-Length": str((1 << 20) + 1)}, b"x"))
    with pytest.raises(ToollessInferenceError, match="request failed|byte"):
        adapter(bounded).review(capsule())
    assert len(bounded.calls) == 1


def test_owned_transport_disables_redirects_streams_and_bounds_reads():
    request = HttpRequest("POST", "https://provider.example.invalid", {}, b"{}", 1, 3)
    transport = Transport(HttpResponse(200, {"Content-Length": "4"}, b"abcd"))
    with pytest.raises(ToollessInferenceError, match="byte"):
        transport.send(request)
    with pytest.raises(ToollessInferenceError, match="exactly one"):
        transport.send(request)

    redirect_opener = Transport(HttpResponse(302, {"Location": "https://other.invalid"}, b""))
    with pytest.raises(ToollessInferenceError, match="redirect"):
        redirect_opener.send(request)

    streaming = Transport(HttpResponse(200, {"Content-Type": "text/event-stream"}, b"data"))
    with pytest.raises(ToollessInferenceError, match="stream"):
        streaming.send(request)


def test_owned_transport_deadline_covers_slow_response_reads(monkeypatch):
    class SlowResponse:
        status = 200
        headers = {"Content-Length": "1"}

        def settimeout(self, timeout):
            self.timeout = timeout

        def read(self, size):
            time.sleep(0.03)
            return b"x"

    class SlowOpener:
        def open(self, request, timeout):
            return SlowResponse()

    monkeypatch.setattr(inference_module, "build_opener", lambda handler: SlowOpener())
    with pytest.raises(ToollessInferenceError, match="deadline|timed out"):
        BoundedHttpTransport().send(HttpRequest("POST", "https://provider.example.invalid", {}, b"{}", 0.005, 8))


def test_owned_transport_applies_remaining_deadline_before_near_expiry_read(monkeypatch):
    class NearExpiryResponse:
        status = 200
        headers = {"Content-Length": "1"}

        def __init__(self):
            self.read_timeout = None

        def settimeout(self, timeout):
            self.read_timeout = timeout

        def read(self, size):
            assert self.read_timeout is not None
            assert self.read_timeout < 0.1
            raise TimeoutError("socket read timed out")

    response = NearExpiryResponse()

    class NearExpiryOpener:
        def open(self, request, timeout):
            time.sleep(0.08)
            return response

    monkeypatch.setattr(inference_module, "build_opener", lambda handler: NearExpiryOpener())
    with pytest.raises(ToollessInferenceError, match="deadline|timed out"):
        BoundedHttpTransport().send(HttpRequest("POST", "https://provider.example.invalid", {}, b"{}", 0.1, 8))


def test_owned_transport_terminates_blocked_connection_setup(monkeypatch):
    class BlockingOpener:
        def open(self, request, timeout):
            time.sleep(5)

    monkeypatch.setattr(inference_module, "build_opener", lambda handler: BlockingOpener())
    started = time.monotonic()
    with pytest.raises(ToollessInferenceError, match="deadline|timed out"):
        BoundedHttpTransport().send(HttpRequest("POST", "https://provider.example.invalid", {}, b"{}", 0.05, 8))
    assert time.monotonic() - started < 1


@pytest.mark.parametrize("environment_name", ["GH_TOKEN", "GITHUB_TOKEN", "GITHUB_API_TOKEN", "GH_ENTERPRISE_TOKEN"])
def test_known_github_environment_names_are_rejected(environment_name):
    policy = deepcopy(POLICY)
    policy["providers"][0]["api_key_env"] = environment_name
    with pytest.raises(ToollessInferenceError, match="GitHub|credential"):
        configured_adapter(
            protected_configuration(policy), Transport(valid_response()),
            {environment_name: "secret"},
        )


def test_provider_environment_rejects_any_extra_secret_capability():
    with pytest.raises(ToollessInferenceError, match="exactly|capability"):
        configured_adapter(
            protected_configuration(), Transport(valid_response()),
            {"REVIEW_API_KEY": "secret", "CUSTOM_GITHUB_TOKEN": "must-not-forward"},
        )


@pytest.mark.parametrize("token", [
    "ghp_x", "github_pat_x", "gho_x", "ghu_x", "ghs_x", "ghr_x", "a" * 40,
])
def test_custom_provider_key_rejects_known_github_token_families(token):
    policy = deepcopy(POLICY)
    policy["providers"][0]["api_key_env"] = "CUSTOM_PROVIDER_KEY"
    with pytest.raises(ToollessInferenceError, match="GitHub|credential"):
        configured_adapter(
            protected_configuration(policy), Transport(valid_response()),
            {"CUSTOM_PROVIDER_KEY": token},
        )


def test_arbitrary_send_object_is_not_an_accepted_transport():
    class FakeTransport:
        def send(self, request):
            return valid_response()

    with pytest.raises(ToollessInferenceError, match="concrete|transport"):
        ToollessReviewAdapter.from_configuration(
            protected_configuration(), "review-adapter", FakeTransport(), {"REVIEW_API_KEY": "secret"}
        )


def test_input_tokens_do_not_consume_output_token_ceiling():
    response = valid_response()
    payload = json.loads(response.body)
    payload["usage"] = {"prompt_tokens": 10000, "completion_tokens": 1, "total_tokens": 10001}
    proposal = adapter(Transport(HttpResponse(200, {"content-type": "application/json"}, json.dumps(payload).encode()))).review(capsule())
    assert proposal.response_digest.startswith("sha256:")


def test_one_request_enforcement_and_no_github_credentials():
    transport = Transport(valid_response())
    review = adapter(transport)
    review.review(capsule())
    with pytest.raises(ToollessInferenceError, match="request"):
        review.review(capsule())
    request = json.loads(transport.calls[0].data)
    assert "GITHUB_TOKEN" not in json.dumps(request)
    assert "ghp_" not in json.dumps(request)


@pytest.mark.parametrize(
    "finding",
    [
        {"path": "not-changed.py", "range": [1, 1], "severity": "error", "message": "bad"},
        {"path": "x.py", "range": [2, 2], "severity": "error", "message": "bad"},
        {"path": "x.py", "range": [1, 1], "severity": "critical", "message": "bad"},
    ],
)
def test_citations_and_findings_must_be_inside_capsule(finding):
    response = valid_response(verdict="changes-requested", findings=[finding])
    with pytest.raises(ToollessInferenceError, match="finding|citation|range|path|severity"):
        adapter(Transport(response)).review(capsule())
