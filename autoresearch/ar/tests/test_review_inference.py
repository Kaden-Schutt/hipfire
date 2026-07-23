# Copyright (c) Kaden Schutt
import base64
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import multiprocessing
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
from autoresearch.ar.review.models import ReviewTarget, fixture_descriptor_digest
from autoresearch.ar.review.validation import MAX_VALIDATION_ROWS


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
_PROTECTED_VALIDATION_REQUESTS = (
    ("rdna3-smoke", "run the protected smoke fixture"),
    ("gfx1151-kernel-validation", "run the protected kernel fixture"),
    ("dflash-coherence", "run the protected coherence fixture"),
)


def protected_validation_requests(rationale_overrides=None):
    rationale_overrides = rationale_overrides or {}
    return [
        {"profile_id": profile_id, "rationale": rationale_overrides.get(profile_id, rationale)}
        for profile_id, rationale in _PROTECTED_VALIDATION_REQUESTS
    ]


def protected_configuration(policy=None, capability_policy=None):
    global _CONFIGURATION, _LIVE_CLIENT, _LIVE_RUNNER
    if policy is None and capability_policy is None and _CONFIGURATION is not None:
        return _CONFIGURATION
    root = Path(tempfile.mkdtemp())
    config_dir = root / ".github" / "agentic-review"
    config_dir.mkdir(parents=True)
    (config_dir / "providers.json").write_text(json.dumps(policy or POLICY), encoding="utf-8")
    if capability_policy is None:
        shutil.copy(ROOT / ".github" / "agentic-review" / "capabilities-v1.json", config_dir / "capabilities-v1.json")
    else:
        (config_dir / "capabilities-v1.json").write_text(json.dumps(capability_policy), encoding="utf-8")
    shutil.copy(ROOT / ".github" / "agentic-review" / "trusted-publishers.json", config_dir / "trusted-publishers.json")
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
    if policy is None and capability_policy is None:
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
    transport = BoundedHttpTransport(context=multiprocessing.get_context("fork"))
    transport.calls = _OPEN_OPENER.calls
    return transport


def valid_response(**changes):
    content = {
        "verdict": "clean",
        "findings": [],
        "validation_requests": protected_validation_requests(),
        "scope": {
            "model_architectures": ["qwen3.6-27b"],
            "hardware_architectures": ["gfx1100", "gfx1101", "gfx1102", "gfx1150", "gfx1151"],
        },
        "hardware_validation_triage": {
            "impacted_model_families": ["qwen3.6-27b"],
            "impacted_hardware": ["gfx1100", "gfx1101", "gfx1102", "gfx1150", "gfx1151"],
            "coverage_decision": "all-impacted",
            "rationale": "all model families and hardware architectures are impacted by this change",
        },
    }
    value = {"choices": [{"index": 0, "message": {"role": "assistant", "content": json.dumps(content)}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}, "cost_usd": 0.01}
    response_keys = {"verdict", "findings", "validation_requests", "scope", "hardware_validation_triage"}
    if response_keys.intersection(changes):
        content.update({key: changes.pop(key) for key in tuple(changes) if key in response_keys})
        value["choices"][0]["message"]["content"] = json.dumps(content)
    value.update(changes)
    return HttpResponse(200, {"content-type": "application/json"}, json.dumps(value).encode())


def test_provider_cannot_omit_a_protected_profile():
    response = valid_response(validation_requests=protected_validation_requests()[:-1])

    with pytest.raises(
        ToollessInferenceError,
        match="^provider validation requests must cover every protected profile$",
    ):
        adapter(Transport(response)).review(capsule())


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
    assert request_json["response_format"]["type"] == "json_object"
    assert "x.py" in request_json["messages"][1]["content"]
    assert "PROTECTED_REVIEW_MODE=non-exempt\n" in request_json["messages"][1]["content"]


@pytest.mark.parametrize("missing", ["validation_requests", "scope"])
def test_live_provider_parser_rejects_legacy_two_field_proposals(missing):
    response = valid_response()
    payload = json.loads(response.body)
    content = json.loads(payload["choices"][0]["message"]["content"])
    content.pop(missing)
    payload["choices"][0]["message"]["content"] = json.dumps(content)
    with pytest.raises(ToollessInferenceError, match="unknown or missing"):
        adapter(Transport(HttpResponse(200, response.headers, json.dumps(payload).encode()))).review(capsule())


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
        BoundedHttpTransport(context=multiprocessing.get_context("fork")).send(
            HttpRequest("POST", "https://provider.example.invalid", {}, b"{}", 0.005, 8)
        )


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
        BoundedHttpTransport(context=multiprocessing.get_context("fork")).send(
            HttpRequest("POST", "https://provider.example.invalid", {}, b"{}", 0.1, 8)
        )


def test_owned_transport_terminates_blocked_connection_setup(monkeypatch):
    class BlockingOpener:
        def open(self, request, timeout):
            time.sleep(5)

    monkeypatch.setattr(inference_module, "build_opener", lambda handler: BlockingOpener())
    started = time.monotonic()
    with pytest.raises(ToollessInferenceError, match="deadline|timed out"):
        BoundedHttpTransport(context=multiprocessing.get_context("fork")).send(
            HttpRequest("POST", "https://provider.example.invalid", {}, b"{}", 0.05, 8)
        )
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


def test_provider_request_contains_only_protected_validation_profile_catalogue():
    transport = Transport(valid_response())
    adapter(transport).review(capsule())
    request = json.loads(transport.calls[0].data)

    assert "validation_catalogue" not in request
    user_content = request["messages"][1]["content"]
    catalogue_json = user_content.split("VALIDATION_PROFILE_CATALOGUE_JSON=", 1)[1].split(
        "\nCAPSULE_JSON_STRING=", 1
    )[0]
    catalogue = json.loads(catalogue_json)
    assert [profile["id"] for profile in catalogue] == sorted(profile["id"] for profile in catalogue)
    assert catalogue
    assert all(set(profile) == {
        "id", "model_architecture", "fixture_id",
        "representative_hardware", "covered_hardware",
    } for profile in catalogue)
    assert len(catalogue_json.encode("utf-8")) <= 64 * 1024
    assert not any(field in json.dumps(catalogue) for field in ("commands", "paths", "environment", "secret", "policy"))

    assert request["response_format"] == {"type": "json_object"}


def test_trusted_instruction_requires_authoritative_mode_dependent_scope_and_requests():
    transport = Transport(valid_response())
    adapter(transport).review(capsule())
    instruction = json.loads(transport.calls[0].data)["messages"][0]["content"].lower()

    for semantic in (
        "inspect only the supplied immutable capsule",
        "validation_profile_catalogue_json",
        "the trusted protected_review_mode marker and validation_profile_catalogue_json catalogue are authoritative",
        "for protected_review_mode=non-exempt, scope must contain the complete registered model_architectures",
        "hardware_architectures inventory from the authoritative catalogue",
        "validation_requests must contain every protected profile exactly once",
        "for protected_review_mode=exempt, scope must be empty and validation_requests",
        "must be empty",
        "each item must contain only profile_id and a concise rationale",
        "the provider cannot invent profiles or scope",
        "only profile_id and a concise rationale",
        "no invented hardware, fixture, or commands",
        "required for hardware/model smoke validation",
    ):
        assert semantic in instruction

    assert "touched" not in instruction
    assert "relevant" not in instruction
    assert "coverage-based" not in instruction


def test_oversized_protected_profile_catalogue_is_rejected_before_request():
    custom_capabilities = json.loads(
        (ROOT / ".github" / "agentic-review" / "capabilities-v1.json").read_text(encoding="utf-8")
    )
    oversized_model = "x" * (64 * 1024)
    profile = custom_capabilities["profiles"][0]
    profile["model_architecture"] = oversized_model
    fixture = next(item for item in custom_capabilities["fixtures"] if item["fixture_id"] == profile["fixture_id"])
    fixture["model_architecture"] = oversized_model
    fixture["fixture_digest"] = fixture_descriptor_digest(fixture)
    profile["fixture_digest"] = fixture["fixture_digest"]
    configuration = protected_configuration(capability_policy=custom_capabilities)
    review_adapter = configured_adapter(
        configuration, Transport(valid_response()), {"REVIEW_API_KEY": "secret"}
    )

    with pytest.raises(ToollessInferenceError, match="catalogue|byte"):
        review_adapter._request_body(capsule())


def test_capability_policy_rejects_more_profiles_than_validation_rows():
    custom_capabilities = json.loads(
        (ROOT / ".github" / "agentic-review" / "capabilities-v1.json").read_text(encoding="utf-8")
    )
    profile = custom_capabilities["profiles"][0]
    custom_capabilities["profiles"].extend(
        [{**profile, "id": f"extra-profile-{index}"} for index in range(MAX_VALIDATION_ROWS)]
    )

    with pytest.raises(ValueError, match=rf"more than {MAX_VALIDATION_ROWS} profiles"):
        protected_configuration(capability_policy=custom_capabilities)


def test_provider_hardware_override_is_rejected():
    response = valid_response(validation_requests=[{
        "profile_id": "rdna3-smoke",
        "rationale": "run the protected smoke fixture",
        "hardware": "provider-selected-hardware",
    }])
    with pytest.raises(ToollessInferenceError, match="unknown|missing|validation request"):
        adapter(Transport(response)).review(capsule())


def test_validation_request_is_enriched_from_protected_profile_and_capability():
    configuration = protected_configuration()
    profile = next(item for item in configuration.capabilities["profiles"] if item["id"] == "rdna3-smoke")
    capability = next(item for item in configuration.capabilities["capabilities"] if item["id"] == profile["capability_id"])
    transport = Transport(valid_response(validation_requests=protected_validation_requests({
        "rdna3-smoke": "  inspect\n  the smoke result  ",
    }), scope={
        "model_architectures": ["qwen3.6-27b"],
        "hardware_architectures": ["gfx1100", "gfx1101", "gfx1102", "gfx1150", "gfx1151"],
    }))

    proposal = adapter(transport).review(capsule())

    assert len(proposal.validation_ledger) == 3
    row = next(row for row in proposal.validation_ledger if row.profile_snapshot["id"] == "rdna3-smoke")
    assert row.rationales == ("inspect the smoke result",)
    assert row.model_architecture == profile["model_architecture"]
    assert row.representative_hardware == profile["representative_hardware"]
    assert row.covered_hardware == tuple(profile["covered_hardware"])
    assert row.fixture_id == profile["fixture_id"]
    assert row.fixture_digest == profile["fixture_digest"]
    assert row.contract_digest == capability["contract_digest"]
    assert row.profile_snapshot == profile
    assert proposal.configuration_source_digest == configuration.source.config_digest
    assert proposal.scope.model_architectures == ("qwen3.6-27b",)
    assert proposal.scope.hardware_architectures == ("gfx1100", "gfx1101", "gfx1102", "gfx1150", "gfx1151")


def test_unapproved_scope_is_rejected_for_non_exempt_capsule():
    with pytest.raises(ToollessInferenceError, match="scope|protected"):
        adapter(Transport(valid_response(
            validation_requests=[{"profile_id": "rdna3-smoke", "rationale": "check it"}],
            scope={"model_architectures": ["qwen3.6-27b"], "hardware_architectures": ["gfx9999"]},
        ))).review(capsule())


@pytest.mark.parametrize(
    "scope",
    [
        {"model_architectures": [], "hardware_architectures": []},
        {"model_architectures": ["qwen3.6-27b"], "hardware_architectures": ["gfx1100"]},
    ],
)
def test_scope_must_exactly_match_protected_capsule_scope(scope):
    with pytest.raises(ToollessInferenceError, match="scope"):
        adapter(Transport(valid_response(
            validation_requests=[{"profile_id": "rdna3-smoke", "rationale": "check it"}],
            scope=scope,
        ))).review(capsule())


@pytest.mark.parametrize(
    "requests, message",
    [
        ([{"profile_id": "unknown-profile", "rationale": "not protected"}], "unknown"),
        ([
            {"profile_id": "rdna3-smoke", "rationale": "first"},
            {"profile_id": "rdna3-smoke", "rationale": "second"},
        ], "duplicate"),
    ],
)
def test_validation_request_profile_ids_must_be_known_and_unique(requests, message):
    with pytest.raises(ToollessInferenceError, match=message):
        adapter(Transport(valid_response(validation_requests=requests))).review(capsule())


def test_validation_rationale_is_normalized_and_bounded():
    proposal = adapter(Transport(valid_response(validation_requests=protected_validation_requests({
        "rdna3-smoke": "  first\nsecond  ",
    })))).review(capsule())
    row = next(row for row in proposal.validation_ledger if row.profile_snapshot["id"] == "rdna3-smoke")
    assert row.rationales == ("first second",)

    with pytest.raises(ToollessInferenceError, match="rationale|limit"):
        adapter(Transport(valid_response(validation_requests=protected_validation_requests({
            "rdna3-smoke": "x" * 1025,
        })))).review(capsule())

    accepted = adapter(Transport(valid_response(validation_requests=protected_validation_requests({
        "rdna3-smoke": "😀" * 256,
    })))).review(capsule())
    accepted_row = next(row for row in accepted.validation_ledger if row.profile_snapshot["id"] == "rdna3-smoke")
    assert len(accepted_row.rationales[0].encode("utf-8")) == 1024

    with pytest.raises(ToollessInferenceError, match="rationale|limit"):
        adapter(Transport(valid_response(validation_requests=protected_validation_requests({
            "rdna3-smoke": "😀" * 257,
        })))).review(capsule())


def test_empty_validation_requests_are_rejected_for_non_exempt_changes():
    with pytest.raises(
        ToollessInferenceError,
        match="^provider validation requests must cover every protected profile$",
    ):
        adapter(Transport(valid_response(validation_requests=[]))).review(capsule())


def test_reverse_ordered_validation_selections_are_serialized_by_request_id():
    profile_ids = [request["profile_id"] for request in protected_validation_requests()]
    profile_ids.sort(key=lambda profile_id: "vr-" + hashlib.sha256(profile_id.encode()).hexdigest()[:16])
    requests = [{"profile_id": profile_id, "rationale": "check it"} for profile_id in reversed(profile_ids)]
    proposal = adapter(Transport(valid_response(validation_requests=requests))).review(capsule())
    assert tuple(row.request_id for row in proposal.validation_ledger) == tuple(
        sorted(row.request_id for row in proposal.validation_ledger)
    )


@pytest.mark.parametrize(
    "content",
    [
        {"verdict": "not-a-verdict", "findings": []},
        {"verdict": "clean", "findings": [{"path": "x.py", "range": [1, 1], "severity": "error", "message": "bad"}]},
        {"verdict": "changes-requested", "findings": []},
    ],
)
def test_original_verdict_and_finding_consistency_are_validated_before_downgrade(content):
    with pytest.raises(ToollessInferenceError, match="verdict|actionable|finding"):
        adapter(Transport(valid_response(**content))).review(capsule())


def test_policy_exempt_partial_ledger_is_rejected():
    custom_capabilities = json.loads(
        (ROOT / ".github" / "agentic-review" / "capabilities-v1.json").read_text(encoding="utf-8")
    )
    custom_capabilities["exemptions"] = [{"id": "test-exempt", "path_globs": ["x.py"]}]
    configuration = protected_configuration(capability_policy=custom_capabilities)

    with pytest.raises(
        ToollessInferenceError,
        match="^provider validation requests are forbidden for exempt capsule$",
    ):
        configured_adapter(
            configuration,
            Transport(valid_response(
                validation_requests=protected_validation_requests()[:1],
                scope={"model_architectures": [], "hardware_architectures": []},
            )),
            {"REVIEW_API_KEY": "secret"},
        ).review(capsule())


def test_policy_exempt_empty_ledger_is_clean_and_binds_configuration_digest():
    custom_capabilities = json.loads(
        (ROOT / ".github" / "agentic-review" / "capabilities-v1.json").read_text(encoding="utf-8")
    )
    custom_capabilities["exemptions"] = [{"id": "test-exempt", "path_globs": ["x.py"]}]
    configuration = protected_configuration(capability_policy=custom_capabilities)
    transport = Transport(valid_response(
        validation_requests=[],
        scope={"model_architectures": [], "hardware_architectures": []},
    ))
    proposal = configured_adapter(
        configuration,
        transport,
        {"REVIEW_API_KEY": "secret"},
    ).review(capsule())

    assert proposal.verdict == "clean"
    assert proposal.validation_ledger == ()
    assert proposal.configuration_source_digest == configuration.source.config_digest
    assert proposal.exemption_ids == ("test-exempt",)
    assert proposal.exemption_paths == ("x.py",)
    request = json.loads(transport.calls[0].data)
    assert "PROTECTED_REVIEW_MODE=exempt\n" in request["messages"][1]["content"]
    with pytest.raises(ValueError, match="proposal digest"):
        replace(proposal, configuration_source_digest="sha256:" + "0" * 64)


def test_validation_request_id_collision_is_rejected(monkeypatch):
    real_row = inference_module.ValidationLedgerRow

    def colliding_row(*args, **kwargs):
        row = real_row(*args, **kwargs)
        object.__setattr__(row, "request_id", "vr-collision")
        return row

    monkeypatch.setattr(inference_module, "ValidationLedgerRow", colliding_row)
    requests = protected_validation_requests({
        "rdna3-smoke": "check smoke",
        "gfx1151-kernel-validation": "check kernel",
        "dflash-coherence": "check coherence",
    })
    with pytest.raises(ToollessInferenceError, match="collision"):
        adapter(Transport(valid_response(validation_requests=requests))).review(capsule())
