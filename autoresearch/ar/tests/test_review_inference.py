# Copyright (c) Kaden Schutt
import base64
from copy import deepcopy
import json

import pytest

from autoresearch.ar.review.capsule import build_review_capsule
from autoresearch.ar.review.inference import (
    HttpRequest,
    HttpResponse,
    ToollessReviewAdapter,
    ToollessInferenceError,
)
from autoresearch.ar.review.config import ReviewConfiguration
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


def capsule():
    class Client:
        def get_commit(self, repository, sha):
            tree_sha = "merge-tree" if sha == "merge" else "head-tree"
            return type("Response", (), {"data": {"sha": sha, "tree": {"sha": tree_sha}}})()

        def get_tree(self, repository, sha, *, recursive=False):
            entries = [] if sha == "merge-tree" else [{"path": "x.py", "mode": "100644", "type": "blob", "sha": "x"}]
            return type("Response", (), {"data": {"sha": sha, "tree": entries, "truncated": False}})()

        def get_blob(self, repository, sha):
            return type("Response", (), {"data": {"sha": sha, "encoding": "base64", "content": base64.b64encode(b"x = 1\n").decode(), "size": 6}})()

    return build_review_capsule(Client(), TARGET)


class Transport:
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
    configuration = ReviewConfiguration(POLICY, {}, {})
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
        ToollessReviewAdapter.from_configuration(ReviewConfiguration(POLICY, {}, {}), "review-adapter-extra", Transport(valid_response()), {"REVIEW_API_KEY": "secret"})


def test_provider_requires_protected_configuration_and_injected_non_github_environment():
    with pytest.raises(ToollessInferenceError, match="protected"):
        ToollessReviewAdapter.from_configuration(POLICY, "review-adapter", Transport(valid_response()), {"REVIEW_API_KEY": "secret"})
    with pytest.raises(ToollessInferenceError, match="GitHub"):
        ToollessReviewAdapter.from_configuration(
            ReviewConfiguration(POLICY, {}, {}), "review-adapter", Transport(valid_response()),
            {"REVIEW_API_KEY": "secret", "GITHUB_TOKEN": "must-not-forward"},
        )
    with pytest.raises(ToollessInferenceError, match="absent"):
        ToollessReviewAdapter.from_configuration(ReviewConfiguration(POLICY, {}, {}), "review-adapter", Transport(valid_response()), {})
    unsupported = deepcopy(POLICY)
    unsupported["providers"][0]["adapter_id"] = "arbitrary-provider"
    with pytest.raises(ToollessInferenceError, match="supported"):
        ToollessReviewAdapter.from_configuration(ReviewConfiguration(unsupported, {}, {}), "review-adapter", Transport(valid_response()), {"REVIEW_API_KEY": "secret"})


@pytest.mark.parametrize(
    "response",
    [
        HttpResponse(302, {"location": "https://other.invalid"}, b""),
        HttpResponse(200, {"transfer-encoding": "chunked"}, b"{}"),
        HttpResponse(200, {"content-type": "application/json"}, b"{"),
        HttpResponse(200, {"content-type": "application/json"}, b'{"choices":[],"usage":{},"cost_usd":0,"extra":1}'),
    ],
)
def test_redirect_streaming_malformed_and_unknown_response_are_rejected(response):
    with pytest.raises(ToollessInferenceError):
        adapter(Transport(response)).review(capsule())


def test_transport_rejects_redirect_flag_and_enforces_response_limit_before_download():
    redirected = Transport(HttpResponse(200, {}, b"{}", redirected=True))
    with pytest.raises(ToollessInferenceError, match="redirect"):
        adapter(redirected).review(capsule())

    class BoundedTransport:
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
