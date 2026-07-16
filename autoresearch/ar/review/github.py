# Copyright (c) Kaden Schutt
"""A fixed, typed GitHub REST boundary backed by ``gh api``."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import json
import re
import subprocess
from typing import Any, Protocol
from urllib.parse import quote

from .config import ReviewConfiguration, validate_operator_credential_manifest
from .models import GitHubEnvelope, validate_trusted_publishers_policy


class GitHubBoundaryError(RuntimeError):
    """Raised whenever GitHub data or the subprocess boundary is unsafe."""


class PreflightError(GitHubBoundaryError):
    pass


class Runner(Protocol):
    def __call__(self, argv: Sequence[str], input_data: bytes | None = None) -> Any: ...


def _subprocess_runner(argv: Sequence[str], input_data: bytes | None = None) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(argv, input=input_data, capture_output=True, check=False)


@dataclass(frozen=True)
class GitHubResponse:
    data: Any
    headers: Mapping[str, str]
    status_code: int


@dataclass(frozen=True)
class EffectivePermission:
    login: str
    principal_type: str
    permission: str


@dataclass(frozen=True)
class PreflightResult:
    login: str
    principal_type: str
    repository: Mapping[str, Any]
    scopes: tuple[str, ...]


_REPO = r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+"
_SHA = r"[A-Za-z0-9_.:-]+"
_MAX_PAGINATED_PAGES = 16
_MAX_PAGINATED_ITEMS = 4096
_MAX_RESPONSE_BYTES = 16 * 1024 * 1024
_ENDPOINTS = (
    ("GET", re.compile(rf"/user$"), False),
    ("GET", re.compile(rf"/repos/{_REPO}$"), False),
    ("GET", re.compile(rf"/repos/{_REPO}/pulls$"), True),
    ("GET", re.compile(rf"/repos/{_REPO}/pulls/[1-9][0-9]*$"), False),
    ("GET", re.compile(rf"/repos/{_REPO}/issues/[1-9][0-9]*/comments$"), True),
    ("GET", re.compile(rf"/repos/{_REPO}/pulls/[1-9][0-9]*/reviews$"), True),
    ("POST", re.compile(rf"/repos/{_REPO}/issues/[1-9][0-9]*/labels$"), False),
    ("DELETE", re.compile(rf"/repos/{_REPO}/issues/[1-9][0-9]*/labels/[^/]+$"), False),
    ("GET", re.compile(rf"/repos/{_REPO}/collaborators/[^/]+/permission$"), False),
    ("GET", re.compile(rf"/repos/{_REPO}/git/trees/{_SHA}$"), False),
    ("GET", re.compile(rf"/repos/{_REPO}/git/blobs/{_SHA}$"), False),
    ("POST", re.compile(rf"/repos/{_REPO}/issues/[1-9][0-9]*/comments$"), False),
    ("POST", re.compile(rf"/repos/{_REPO}/pulls/[1-9][0-9]*/reviews$"), False),
    ("PUT", re.compile(rf"/repos/{_REPO}/pulls/[1-9][0-9]*/reviews/[1-9][0-9]*/dismissals$"), False),
)
_LIST_PATHS = {pattern.pattern for _, pattern, paginated in _ENDPOINTS if paginated}
_PRINCIPAL_TYPES = {"User", "Bot", "Organization"}
_PERMISSIONS = ("admin", "maintain", "push", "triage", "pull")


def _decode_output(raw: str | bytes) -> tuple[list[tuple[int, dict[str, str], Any]], bool]:
    if not isinstance(raw, (str, bytes)):
        raise GitHubBoundaryError("gh response has an unsupported output type")
    if isinstance(raw, bytes) and len(raw) > _MAX_RESPONSE_BYTES:
        raise GitHubBoundaryError("gh response exceeds the fixed size bound")
    try:
        text = raw.decode() if isinstance(raw, bytes) else raw
        encoded_size = len(text.encode()) if isinstance(text, str) else 0
    except UnicodeError as exc:
        raise GitHubBoundaryError("gh response is not valid UTF-8") from exc
    if not isinstance(text, str) or encoded_size > _MAX_RESPONSE_BYTES:
        raise GitHubBoundaryError("gh response exceeds the fixed size bound")
    if not text.strip():
        raise GitHubBoundaryError("gh returned an empty response")
    if not text.lstrip().startswith("HTTP/"):
        try:
            return [(200, {}, json.loads(text, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value))))], False
        except (ValueError, json.JSONDecodeError) as exc:
            raise GitHubBoundaryError("gh returned invalid JSON") from exc
    decoder = json.JSONDecoder(parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    offset = 0
    pages: list[tuple[int, dict[str, str], Any]] = []
    while offset < len(text):
        while offset < len(text) and text[offset] in "\r\n \t":
            offset += 1
        if offset >= len(text):
            break
        line_end = text.find("\n", offset)
        if line_end < 0 or not text[offset:line_end].startswith("HTTP/"):
            raise GitHubBoundaryError("unexpected pagination response")
        status_parts = text[offset:line_end].strip().split()
        try:
            status = int(status_parts[1])
        except (IndexError, ValueError) as exc:
            raise GitHubBoundaryError("invalid GitHub response status") from exc
        offset = line_end + 1
        headers: dict[str, str] = {}
        while True:
            line_end = text.find("\n", offset)
            if line_end < 0:
                raise GitHubBoundaryError("truncated GitHub response headers")
            line = text[offset:line_end].rstrip("\r")
            offset = line_end + 1
            if not line:
                break
            if ":" not in line:
                raise GitHubBoundaryError("invalid GitHub response header")
            name, value = line.split(":", 1)
            headers[name.strip().lower()] = value.strip()
        try:
            value, consumed = decoder.raw_decode(text[offset:])
        except (ValueError, json.JSONDecodeError) as exc:
            raise GitHubBoundaryError("GitHub response contains invalid JSON") from exc
        pages.append((status, headers, value))
        if len(pages) > _MAX_PAGINATED_PAGES:
            raise GitHubBoundaryError("GitHub pagination exceeds the fixed page bound")
        offset += consumed
    if not pages:
        raise GitHubBoundaryError("unexpected pagination response")
    return pages, len(pages) > 1


def _as_result(result: Any) -> tuple[int, str, str]:
    if isinstance(result, subprocess.CompletedProcess):
        return int(result.returncode), result.stdout or "", result.stderr or ""
    if isinstance(result, Mapping):
        return int(result.get("returncode", 0)), result.get("stdout", ""), result.get("stderr", "")
    if isinstance(result, tuple) and len(result) == 3:
        return int(result[0]), result[1], result[2]
    raise GitHubBoundaryError("runner returned an unsupported result")


class GitHubClient:
    def __init__(self, runner: Runner = _subprocess_runner, *, gh_binary: str = "gh"):
        self._runner = runner
        self._gh_binary = gh_binary

    def _allowed(self, method: str, path: str) -> bool:
        return any(method == allowed_method and pattern.fullmatch(path) for allowed_method, pattern, _ in _ENDPOINTS)

    def _request(
        self,
        method: str,
        path: str,
        *,
        query: Mapping[str, str | int] | None = None,
        fields: Mapping[str, str] | None = None,
        paginate: bool = False,
    ) -> GitHubResponse:
        if not self._allowed(method, path):
            raise GitHubBoundaryError("GitHub path or method is not allowlisted")
        expected_paginated = any(
            method == allowed_method and pattern.fullmatch(path) and is_paginated
            for allowed_method, pattern, is_paginated in _ENDPOINTS
        )
        if paginate != expected_paginated:
            raise GitHubBoundaryError("unexpected pagination for endpoint")
        argv = [self._gh_binary, "api"]
        if paginate:
            argv.append("--paginate")
        argv.extend(["--include", "--method", method])
        request_path = path
        if query:
            request_path += "?" + "&".join(f"{quote(str(key))}={quote(str(value))}" for key, value in query.items())
        argv.append(request_path)
        for key, value in (fields or {}).items():
            argv.extend(["--field", f"{key}={value}"])
        try:
            result = self._runner(argv)
        except Exception as exc:
            raise GitHubBoundaryError("gh subprocess failed") from exc
        returncode, stdout, stderr = _as_result(result)
        if returncode != 0:
            raise GitHubBoundaryError(f"gh exited nonzero: {stderr}")
        try:
            pages, multiple = _decode_output(stdout)
        except GitHubBoundaryError:
            raise
        if multiple and not paginate:
            raise GitHubBoundaryError("unexpected pagination response")
        status, headers, data = pages[0]
        for _, page_headers, _ in pages[1:]:
            headers.update(page_headers)
        error_status = next(
            (page_status for page_status, _, _ in pages if page_status < 200 or page_status >= 300), None
        )
        if error_status is not None:
            raise GitHubBoundaryError(f"GitHub returned HTTP {error_status}")
        if paginate:
            if any(not isinstance(page_data, list) for _, _, page_data in pages):
                raise GitHubBoundaryError("paginated endpoint returned an incomplete page (non-list)")
            data = [item for _, _, page_data in pages for item in page_data]
            if len(data) > _MAX_PAGINATED_ITEMS:
                raise GitHubBoundaryError("GitHub pagination exceeds the fixed item bound")
        return GitHubResponse(data, headers, status)

    @staticmethod
    def _require_mapping(data: Any, name: str) -> Mapping[str, Any]:
        if not isinstance(data, Mapping):
            raise GitHubBoundaryError(f"GitHub {name} response is not an object")
        return data

    @staticmethod
    def _require(data: Mapping[str, Any], fields: Sequence[str], name: str) -> Mapping[str, Any]:
        if any(field not in data or data[field] in (None, "") for field in fields):
            raise GitHubBoundaryError(f"GitHub {name} response is missing fields")
        return data

    def get_authenticated_user(self) -> GitHubResponse:
        response = self._request("GET", "/user")
        if "x-oauth-scopes" not in response.headers:
            raise GitHubBoundaryError("visible OAuth scope header is required")
        data = self._require_mapping(response.data, "user")
        if "type" not in data or not data["type"]:
            raise GitHubBoundaryError("GitHub user principal type is missing")
        self._require(data, ("id", "login"), "user")
        if (
            isinstance(data["id"], bool)
            or not isinstance(data["id"], int)
            or data["id"] <= 0
            or not isinstance(data["login"], str)
            or not data["login"].strip()
            or not isinstance(data["type"], str)
            or data["type"] not in _PRINCIPAL_TYPES
        ):
            raise GitHubBoundaryError("GitHub user has an unsupported principal type")
        return response

    def get_repository(self, repository: str) -> GitHubResponse:
        response = self._request("GET", f"/repos/{repository}")
        data = self._require_mapping(response.data, "repository")
        self._require(data, ("id", "full_name"), "repository")
        if (
            isinstance(data["id"], bool)
            or not isinstance(data["id"], int)
            or data["id"] <= 0
            or not isinstance(data["full_name"], str)
            or not data["full_name"].strip()
        ):
            raise GitHubBoundaryError("GitHub repository response has malformed identity")
        return response

    def list_pull_requests(self, repository: str, *, pages: int = 1) -> GitHubResponse:
        if isinstance(pages, bool) or not isinstance(pages, int) or not 0 < pages <= _MAX_PAGINATED_PAGES:
            raise GitHubBoundaryError("pages must be within the fixed positive bound")
        responses = []
        for page in range(1, pages + 1):
            responses.append(self._request("GET", f"/repos/{repository}/pulls", query={"per_page": 1, "page": page}, paginate=True))
        data = []
        headers = {}
        for response in responses:
            if not isinstance(response.data, list):
                raise GitHubBoundaryError("pull request response is not a list")
            data.extend(response.data)
            headers.update(response.headers)
        for item in data:
            self._validate_pull(self._require_mapping(item, "pull request"))
        return GitHubResponse(data, headers, 200)

    @classmethod
    def _validate_pull(cls, data: Mapping[str, Any]) -> None:
        cls._require(data, ("number", "node_id", "head", "base"), "pull request")
        head = cls._require(cls._require_mapping(data["head"], "pull request head"), ("repo", "sha"), "pull request head")
        base = cls._require(cls._require_mapping(data["base"], "pull request base"), ("ref", "sha"), "pull request base")
        cls._require(cls._require_mapping(head["repo"], "head repository"), ("full_name",), "head repository")
        cls._require(base, ("ref", "sha"), "pull request base")

    def get_pull_request(self, repository: str, number: int) -> GitHubResponse:
        response = self._request("GET", f"/repos/{repository}/pulls/{number}")
        data = self._require_mapping(response.data, "pull request")
        self._validate_pull(data)
        return response

    def list_issue_comments(self, repository: str, number: int) -> GitHubResponse:
        return self._list_records("/repos/{}/issues/{}/comments".format(repository, number), "issue comments")

    def list_pull_reviews(self, repository: str, number: int) -> GitHubResponse:
        return self._list_records("/repos/{}/pulls/{}/reviews".format(repository, number), "pull reviews")

    def _list_records(self, path: str, name: str) -> GitHubResponse:
        response = self._request("GET", path, query={"per_page": 1}, paginate=True)
        if not isinstance(response.data, list):
            raise GitHubBoundaryError(f"GitHub {name} response is not a list")
        for item in response.data:
            record = self._require_mapping(item, name)
            extra = ("body",) if name == "issue comments" else ("state", "commit_id")
            self._validate_api_record(record, name, extra=extra)
            author = self._require(self._require_mapping(record["user"], f"{name} author"), ("login", "type"), f"{name} author")
            if (
                not isinstance(author["login"], str)
                or not author["login"].strip()
                or not isinstance(author["type"], str)
                or author["type"] not in _PRINCIPAL_TYPES
            ):
                raise GitHubBoundaryError(f"GitHub {name} author has an unsupported principal type")
        return response

    @staticmethod
    def _validate_api_record(
        record: Mapping[str, Any], name: str, *, extra: Sequence[str] = ()
    ) -> None:
        GitHubClient._require(record, ("id", "node_id", "user", "created_at", "updated_at", *extra), name)
        if (
            isinstance(record["id"], bool)
            or not isinstance(record["id"], int)
            or record["id"] <= 0
            or not isinstance(record["node_id"], str)
            or not record["node_id"].strip()
            or any(not isinstance(record[field], str) or not record[field].strip() for field in ("created_at", "updated_at"))
        ):
            raise GitHubBoundaryError(f"GitHub {name} response has malformed server fields")

    def collaborator_effective_permission(self, repository: str, login: str) -> EffectivePermission:
        response = self._request("GET", f"/repos/{repository}/collaborators/{quote(login, safe='')}/permission")
        data = self._require_mapping(response.data, "collaborator permission")
        self._require(data, ("user", "permissions"), "collaborator permission")
        principal = self._require(self._require_mapping(data["user"], "collaborator"), ("login", "type"), "collaborator")
        if not isinstance(principal["type"], str) or principal["type"] not in _PRINCIPAL_TYPES:
            raise GitHubBoundaryError("collaborator has an unsupported principal type")
        permissions = self._require_mapping(data["permissions"], "permissions")
        permission = next((name for name in _PERMISSIONS if permissions.get(name) is True), None)
        if permission is None:
            role_map = {"read": "pull", "write": "push", "maintain": "maintain", "triage": "triage", "admin": "admin"}
            role = data.get("role_name")
            permission = role_map.get(role) if isinstance(role, str) else None
        if permission is None:
            raise GitHubBoundaryError("effective collaborator permission is missing")
        return EffectivePermission(principal["login"], principal["type"], permission)

    def get_tree(self, repository: str, tree_sha: str, *, recursive: bool = False) -> GitHubResponse:
        query = {"recursive": "1"} if recursive else None
        response = self._request("GET", f"/repos/{repository}/git/trees/{tree_sha}", query=query)
        data = self._require_mapping(response.data, "tree")
        self._require(data, ("sha", "tree"), "tree")
        return response

    def get_blob(self, repository: str, blob_sha: str) -> GitHubResponse:
        response = self._request("GET", f"/repos/{repository}/git/blobs/{blob_sha}")
        self._require(self._require_mapping(response.data, "blob"), ("sha", "content", "encoding"), "blob")
        return response

    def add_labels(self, repository: str, number: int, labels: Sequence[str]) -> GitHubResponse:
        if not labels or any(not isinstance(label, str) or not label for label in labels):
            raise GitHubBoundaryError("labels must be non-empty strings")
        return self._request("POST", f"/repos/{repository}/issues/{number}/labels", fields={"labels": json.dumps(list(labels))})

    def remove_label(self, repository: str, number: int, label: str) -> GitHubResponse:
        if not label:
            raise GitHubBoundaryError("label must be non-empty")
        return self._request("DELETE", f"/repos/{repository}/issues/{number}/labels/{quote(label, safe='')}")

    def create_issue_comment(self, repository: str, number: int, body: str) -> GitHubResponse:
        if not body:
            raise GitHubBoundaryError("comment body must be non-empty")
        return self._request("POST", f"/repos/{repository}/issues/{number}/comments", fields={"body": body})

    def create_pull_request_review(self, repository: str, number: int, *, body: str, event: str, commit_id: str) -> GitHubResponse:
        if not commit_id or not body or event not in {"APPROVE", "REQUEST_CHANGES", "COMMENT"}:
            raise GitHubBoundaryError("review body, event, and exact commit_id are required")
        return self._request(
            "POST", f"/repos/{repository}/pulls/{number}/reviews",
            fields={"body": body, "event": event, "commit_id": commit_id},
        )

    def dismiss_workflow_review(self, repository: str, number: int, review_id: int, *, message: str) -> GitHubResponse:
        if not message:
            raise GitHubBoundaryError("dismissal message must be non-empty")
        return self._request(
            "PUT", f"/repos/{repository}/pulls/{number}/reviews/{review_id}/dismissals", fields={"message": message}
        )

    def _envelope(
        self, record: Mapping[str, Any], payload: Mapping[str, Any], *, record_name: str = "authenticated GitHub record"
    ) -> GitHubEnvelope:
        try:
            author = record["user"]
            extra = ("body",) if record_name == "issue comment" else ("state", "commit_id")
            self._validate_api_record(record, record_name, extra=extra)
            self._require(author, ("login", "type"), "authenticated author")
        except (KeyError, TypeError) as exc:
            raise GitHubBoundaryError("authenticated record is missing server fields") from exc
        if (
            not isinstance(author["login"], str)
            or not author["login"].strip()
            or not isinstance(author["type"], str)
            or author["type"] not in _PRINCIPAL_TYPES
        ):
            raise GitHubBoundaryError("authenticated author has unsupported principal type")
        if not isinstance(payload, Mapping):
            raise GitHubBoundaryError("GitHub envelope payload must be an object")
        if record["updated_at"] != record["created_at"]:
            raise GitHubBoundaryError("edited GitHub record is not admissible")
        return GitHubEnvelope(payload, record["node_id"], author["login"], record["created_at"], record["updated_at"], author["type"])

    def comment_envelope(self, repository: str, number: int, payload: Mapping[str, Any]) -> GitHubEnvelope:
        response = self.list_issue_comments(repository, number)
        if len(response.data) != 1:
            raise GitHubBoundaryError("comment envelope requires exactly one bounded response")
        return self.envelope_from_comment(response.data[0], payload)

    def review_envelope(self, repository: str, number: int, review_id: int, payload: Mapping[str, Any]) -> GitHubEnvelope:
        response = self.list_pull_reviews(repository, number)
        matches = [item for item in response.data if item.get("id") == review_id]
        if len(matches) != 1:
            raise GitHubBoundaryError("review envelope was not found in authenticated reviews")
        return self.envelope_from_review(matches[0], payload)

    def envelope_from_comment(self, record: Mapping[str, Any], payload: Mapping[str, Any]) -> GitHubEnvelope:
        """Build an envelope only from a complete authenticated comment object."""
        return self._envelope(record, payload, record_name="issue comment")

    def envelope_from_review(self, record: Mapping[str, Any], payload: Mapping[str, Any]) -> GitHubEnvelope:
        """Build an envelope only from a complete authenticated review object."""
        return self._envelope(record, payload, record_name="pull request review")


def _scope_header(response: GitHubResponse) -> tuple[str, ...]:
    raw = response.headers.get("x-oauth-scopes")
    if not isinstance(raw, str) or not raw.strip():
        raise PreflightError("visible X-OAuth-Scopes header is required")
    pieces = tuple(scope.strip() for scope in raw.split(","))
    if any(not scope or re.fullmatch(r"[A-Za-z0-9:_-]+", scope) is None for scope in pieces):
        raise PreflightError("OAuth scope header is malformed")
    scopes = tuple(pieces)
    if "repo" in scopes:
        raise PreflightError("classic repo OAuth scope is not permitted")
    return scopes


def preflight_read_only(
    client: GitHubClient,
    repository: str,
    *,
    mode: str,
    configuration: ReviewConfiguration,
    operator_manifest: Mapping[str, Any] | None = None,
    pull_number: int | None = None,
) -> PreflightResult:
    if mode not in {"discovery", "controller", "publisher"}:
        raise PreflightError("unsupported preflight mode")
    trusted = configuration.trusted_publishers
    try:
        validate_trusted_publishers_policy(trusted)
    except ValueError as exc:
        raise PreflightError(str(exc)) from exc
    try:
        user_response = client.get_authenticated_user()
        scopes = _scope_header(user_response)
        user_data = client._require_mapping(user_response.data, "user")
        repo_response = client.get_repository(repository)
        _scope_header(repo_response)
        repository_data = client._require_mapping(repo_response.data, "repository")
        pulls_response = client.list_pull_requests(repository)
        _scope_header(pulls_response)
        user_login = user_data["login"]
        principal_type = user_data["type"]
        if (
            not isinstance(user_login, str)
            or not user_login.strip()
            or principal_type not in _PRINCIPAL_TYPES
            or isinstance(user_data.get("id"), bool)
            or not isinstance(user_data.get("id"), int)
            or user_data["id"] <= 0
        ):
            raise PreflightError("token identity has no explicit principal type")
        if (
            not isinstance(repository_data.get("id"), int)
            or isinstance(repository_data.get("id"), bool)
            or repository_data["id"] <= 0
            or repository_data.get("full_name") != repository
        ):
            raise PreflightError("repository identity is malformed")
        if pull_number is not None:
            if isinstance(pull_number, bool) or not isinstance(pull_number, int) or pull_number <= 0:
                raise PreflightError("pull number must be positive")
            pull_response = client.get_pull_request(repository, pull_number)
            _scope_header(pull_response)
            comments_response = client.list_issue_comments(repository, pull_number)
            _scope_header(comments_response)
            reviews_response = client.list_pull_reviews(repository, pull_number)
            _scope_header(reviews_response)
        if mode in {"controller", "publisher"}:
            permission = client.collaborator_effective_permission(repository, user_login)
            if permission.principal_type != principal_type:
                raise PreflightError("effective permission principal type mismatch")
            # Permission is a read probe only; publisher mutation authority is
            # deliberately not inferred from it.
        if mode == "publisher":
            if operator_manifest is None:
                raise PreflightError("publisher mode requires operator credential manifest")
            validate_operator_credential_manifest(operator_manifest)
            manifest_principal = operator_manifest["principal"]
            if (
                principal_type != "Bot"
                or manifest_principal["login"] != user_login
                or manifest_principal["type"] != principal_type
                or "publish" not in operator_manifest["allowed_operations"]
            ):
                raise PreflightError("operator manifest does not match the authenticated publisher")
            apps = trusted["apps"]
            if not any(
                app["login"] == user_login
                and app["repository_id"] == repository_data["id"]
                and app["credential_attestation_digest"] == operator_manifest["credential_attestation_digest"]
                for app in apps
            ):
                raise PreflightError("publisher requires a matching trusted App manifest")
    except (GitHubBoundaryError, KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, PreflightError):
            raise
        raise PreflightError(str(exc)) from exc
    return PreflightResult(user_login, principal_type, repository_data, scopes)
