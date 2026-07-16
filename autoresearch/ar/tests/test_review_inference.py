# Copyright (c) Kaden Schutt
import base64
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from urllib.error import HTTPError

import pytest

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
X_OID = hashlib.sha1(b"blob 6\0x = 1\n").hexdigest()


def protected_configuration(policy=None):
    global _CONFIGURATION
    if policy is None and _CONFIGURATION is not None:
        return _CONFIGURATION
    root = Path(tempfile.mkdtemp())
    config_dir = root / ".github" / "agentic-review"
    config_dir.mkdir(parents=True)
    (config_dir / "providers.json").write_text(json.dumps(policy or POLICY), encoding="utf-8")
    for name in ("capabilities-v1.json", "trusted-publishers.json"):
        shutil.copy(ROOT / ".github" / "agentic-review" / name, config_dir / name)
    source_digest = configuration_source_digest(
        (config_dir / "providers.json").read_bytes(),
        (config_dir / "capabilities-v1.json").read_bytes(),
        (config_dir / "trusted-publishers.json").read_bytes(),
    )
    source = AuthenticatedConfigSource._from_authenticated_boundary("owner/repo", "main", "commit-sha", source_digest, root)
    loaded = load_review_configuration(root, source=source)
    if policy is None:
        _CONFIGURATION = loaded
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


class Transport(BoundedHttpTransport):
    def __init__(self, response):
        self.response = response
        self.calls = []

    def send(self, request: HttpRequest):
        self.calls.append(request)
        return self.response


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
    return ToollessReviewAdapter.from_configuration(configuration, "review-adapter", transport, {"REVIEW_API_KEY": "secret"})


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
    assert (request.method, request.url) == ("POST", POLICY["providers"][0]["endpoint"])
    assert request.timeout == 30
    assert request.max_response_bytes == 1 << 20
    assert request.headers["Authorization"] == "Bearer secret"
    assert "github" not in json.dumps(request.headers).lower()
    body = request.body.decode()
    assert '"tools":[]' in body
    assert "function" not in body.lower()
    request_json = json.loads(request.body)
    assert request_json["model"] == "review-model-v1"
    assert request_json["max_output_tokens"] == 128
    assert request_json["response_format"]["type"] == "json_schema"
    assert "x.py" in request_json["messages"][1]["content"]


def test_provider_selection_is_exact_and_empty_policy_fails_closed():
    with pytest.raises(ToollessInferenceError, match="provider"):
        ToollessReviewAdapter.from_configuration(ReviewConfiguration({"schema": POLICY["schema"], "version": 1, "providers": []}, {}, {}), "review-adapter", Transport(valid_response()), {"REVIEW_API_KEY": "secret"})
    with pytest.raises(ToollessInferenceError, match="exact|configured"):
        ToollessReviewAdapter.from_configuration(protected_configuration(), "review-adapter-extra", Transport(valid_response()), {"REVIEW_API_KEY": "secret"})


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
        ToollessReviewAdapter.from_configuration(forged, "review-adapter", Transport(valid_response()), {"REVIEW_API_KEY": "secret"})


def test_provider_requires_protected_configuration_and_injected_non_github_environment():
    with pytest.raises(ToollessInferenceError, match="protected|loaded"):
        ToollessReviewAdapter.from_configuration(ReviewConfiguration(POLICY, {}, {}), "review-adapter", Transport(valid_response()), {"REVIEW_API_KEY": "secret"})
    with pytest.raises(ToollessInferenceError, match="GitHub|exactly"):
        ToollessReviewAdapter.from_configuration(
            protected_configuration(), "review-adapter", Transport(valid_response()),
            {"REVIEW_API_KEY": "secret", "GITHUB_TOKEN": "must-not-forward"},
        )
    with pytest.raises(ToollessInferenceError, match="absent"):
        ToollessReviewAdapter.from_configuration(protected_configuration(), "review-adapter", Transport(valid_response()), {})
    unsupported = deepcopy(POLICY)
    unsupported["providers"][0]["adapter_id"] = "arbitrary-provider"
    with pytest.raises(ToollessInferenceError, match="supported"):
        ToollessReviewAdapter.from_configuration(protected_configuration(unsupported), "review-adapter", Transport(valid_response()), {"REVIEW_API_KEY": "secret"})
    unsupported["providers"][0]["adapter_id"] = "neutral-review"
    with pytest.raises(ToollessInferenceError, match="supported"):
        ToollessReviewAdapter.from_configuration(protected_configuration(unsupported), "review-adapter", Transport(valid_response()), {"REVIEW_API_KEY": "secret"})


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

    class BoundedTransport(BoundedHttpTransport):
        def __init__(self):
            self.called = False

        def send(self, request):
            self.called = True
            assert request.max_response_bytes == 1 << 20
            payload = b"x" * (request.max_response_bytes + 1)
            raise RuntimeError(f"refused before accumulating {len(payload)} bytes")

    bounded = BoundedTransport()
    with pytest.raises(ToollessInferenceError, match="request failed"):
        adapter(bounded).review(capsule())
    assert bounded.called


def test_owned_transport_disables_redirects_streams_and_bounds_reads():
    class Opener:
        def __init__(self, result):
            self.result = result
            self.calls = 0

        def open(self, request, timeout):
            self.calls += 1
            if isinstance(self.result, BaseException):
                raise self.result
            return self.result

    request = HttpRequest("POST", "https://provider.example.invalid", {}, b"{}", 1, 3)
    oversized = type("Response", (), {
        "status": 200,
        "headers": {"Content-Length": "4"},
        "read": lambda self, size: b"abcd",
    })()
    opener = Opener(oversized)
    transport = BoundedHttpTransport(opener)
    with pytest.raises(ToollessInferenceError, match="byte"):
        transport.send(request)
    with pytest.raises(ToollessInferenceError, match="exactly one"):
        transport.send(request)

    redirect_opener = Opener(HTTPError(request.url, 302, "redirect", {}, None))
    with pytest.raises(ToollessInferenceError, match="redirect"):
        BoundedHttpTransport(redirect_opener).send(request)

    streaming = type("Response", (), {
        "status": 200,
        "headers": {"Content-Type": "text/event-stream"},
        "read": lambda self, size: b"data",
    })()
    with pytest.raises(ToollessInferenceError, match="stream"):
        BoundedHttpTransport(Opener(streaming)).send(request)


@pytest.mark.parametrize("environment_name", ["GH_TOKEN", "GITHUB_TOKEN", "GITHUB_API_TOKEN", "GH_ENTERPRISE_TOKEN"])
def test_known_github_environment_names_are_rejected(environment_name):
    policy = deepcopy(POLICY)
    policy["providers"][0]["api_key_env"] = environment_name
    with pytest.raises(ToollessInferenceError, match="GitHub|credential"):
        ToollessReviewAdapter.from_configuration(
            protected_configuration(policy), "review-adapter", Transport(valid_response()),
            {environment_name: "secret"},
        )


def test_provider_environment_rejects_any_extra_secret_capability():
    with pytest.raises(ToollessInferenceError, match="exactly|capability"):
        ToollessReviewAdapter.from_configuration(
            protected_configuration(), "review-adapter", Transport(valid_response()),
            {"REVIEW_API_KEY": "secret", "CUSTOM_GITHUB_TOKEN": "must-not-forward"},
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
    request = json.loads(transport.calls[0].body)
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
