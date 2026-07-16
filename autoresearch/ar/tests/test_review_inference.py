# Copyright (c) Kaden Schutt
import base64
import json

import pytest

from autoresearch.ar.review.capsule import build_review_capsule
from autoresearch.ar.review.inference import (
    HttpResponse,
    ToollessReviewAdapter,
    ToollessInferenceError,
)
from autoresearch.ar.review.models import ReviewTarget


TARGET = ReviewTarget("owner/repo", 42, "owner/repo", "head", "main", "base", "merge")
POLICY = {
    "schema": "hipfire.agentic-review.providers",
    "version": 1,
    "providers": [{
        "id": "review-adapter",
        "adapter_id": "neutral-review",
        "adapter_version": "1",
        "endpoint": "https://provider.example.invalid/v1/review",
        "model": "review-model-v1",
        "api_key_env": "IGNORED",
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
            return type("Response", (), {"data": {"sha": sha, "commit": {"tree": {"sha": tree_sha}}}})()

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

    def __call__(self, method, url, *, headers, body, timeout):
        self.calls.append((method, url, headers, body, timeout))
        return self.response


def valid_response(**changes):
    value = {"verdict": "clean", "findings": [], "usage": {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5}, "cost_usd": 0.01}
    value.update(changes)
    return HttpResponse(200, {"content-type": "application/json"}, json.dumps(value).encode())


def adapter(transport):
    return ToollessReviewAdapter.from_configuration(POLICY, "review-adapter", transport, "secret")


def test_exactly_one_toolless_https_request_and_bound_proposal():
    transport = Transport(valid_response())
    proposal = adapter(transport).review(capsule())

    assert proposal.target == TARGET
    assert proposal.capsule_digest.startswith("sha256:")
    assert len(transport.calls) == 1
    method, url, headers, body, timeout = transport.calls[0]
    assert (method, url) == ("POST", POLICY["providers"][0]["endpoint"])
    assert timeout == 30
    assert headers["Authorization"] == "Bearer secret"
    assert "github" not in json.dumps(headers).lower()
    assert "tools" not in body.decode()
    assert "function" not in body.decode().lower()
    request = json.loads(body)
    assert request["model"] == "review-model-v1"
    assert request["response_format"]["type"] == "json_schema"
    assert "x.py" in request["messages"][1]["content"]


def test_provider_selection_is_exact_and_empty_policy_fails_closed():
    with pytest.raises(ToollessInferenceError, match="provider"):
        ToollessReviewAdapter.from_configuration({"schema": POLICY["schema"], "version": 1, "providers": []}, "review-adapter", Transport(valid_response()), "secret")
    with pytest.raises(ToollessInferenceError, match="exact|configured"):
        ToollessReviewAdapter.from_configuration(POLICY, "review-adapter-extra", Transport(valid_response()), "secret")


@pytest.mark.parametrize(
    "response",
    [
        HttpResponse(302, {"location": "https://other.invalid"}, b""),
        HttpResponse(200, {"transfer-encoding": "chunked"}, b"{}"),
        HttpResponse(200, {"content-type": "application/json"}, b"{"),
        HttpResponse(200, {"content-type": "application/json"}, b'{"verdict":"clean","findings":[],"usage":{},"cost_usd":0,"extra":1}'),
    ],
)
def test_redirect_streaming_malformed_and_unknown_response_are_rejected(response):
    with pytest.raises(ToollessInferenceError):
        adapter(Transport(response)).review(capsule())


def test_one_request_enforcement_and_no_github_credentials():
    transport = Transport(valid_response())
    review = adapter(transport)
    review.review(capsule())
    with pytest.raises(ToollessInferenceError, match="request"):
        review.review(capsule())
    request = json.loads(transport.calls[0][3])
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
