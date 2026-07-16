# Copyright (c) Kaden Schutt
"""A fixed, typed GitHub REST boundary backed by ``gh api``."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import os
import re
import selectors
import subprocess
import time
from typing import Any, Protocol
from urllib.parse import quote

from .canonical import canonical_digest, canonical_loads, metadata_digest
from .config import ReviewConfiguration, validate_operator_credential_manifest
from .models import GitHubEnvelope, IntentPayload, validate_trusted_publishers_policy


class GitHubBoundaryError(RuntimeError):
    """Raised whenever GitHub data or the subprocess boundary is unsafe."""


class PreflightError(GitHubBoundaryError):
    pass


class Runner(Protocol):
    def __call__(self, argv: Sequence[str], input_data: bytes | None = None) -> Any: ...


_SUBPROCESS_TIMEOUT_SECONDS = 30
_STREAM_CHUNK_BYTES = 64 * 1024


def _stop_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            process.kill()
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                pass
    for stream in (process.stdin, process.stdout, process.stderr):
        if stream is not None:
            stream.close()


def _subprocess_runner(argv: Sequence[str], input_data: bytes | None = None) -> subprocess.CompletedProcess[bytes]:
    """Run ``gh`` with bounded, concurrent stdout/stderr readers."""
    process = subprocess.Popen(
        argv,
        stdin=subprocess.PIPE if input_data is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    selector = selectors.DefaultSelector()
    stdout = bytearray()
    stderr = bytearray()
    streams = ((process.stdout, stdout, _MAX_RESPONSE_BYTES, "stdout"), (process.stderr, stderr, _MAX_STDERR_BYTES, "stderr"))
    try:
        for stream, buffer, limit, name in streams:
            assert stream is not None
            os.set_blocking(stream.fileno(), False)
            selector.register(stream, selectors.EVENT_READ, (buffer, limit, name))
        pending = memoryview(input_data) if input_data is not None else None
        if process.stdin is not None:
            os.set_blocking(process.stdin.fileno(), False)
            selector.register(process.stdin, selectors.EVENT_WRITE, None)
        deadline = time.monotonic() + _SUBPROCESS_TIMEOUT_SECONDS
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(argv, _SUBPROCESS_TIMEOUT_SECONDS)
            for key, _ in selector.select(remaining):
                if key.data is None:
                    if pending is None or not pending:
                        selector.unregister(key.fileobj)
                        key.fileobj.close()
                        continue
                    try:
                        written = os.write(key.fileobj.fileno(), pending)
                    except BlockingIOError:
                        continue
                    except BrokenPipeError:
                        selector.unregister(key.fileobj)
                        key.fileobj.close()
                        continue
                    pending = pending[written:]
                    if not pending:
                        selector.unregister(key.fileobj)
                        key.fileobj.close()
                    continue
                buffer, limit, name = key.data
                try:
                    chunk = os.read(key.fileobj.fileno(), _STREAM_CHUNK_BYTES)
                except BlockingIOError:
                    continue
                if not chunk:
                    selector.unregister(key.fileobj)
                    key.fileobj.close()
                    continue
                if len(buffer) + len(chunk) > limit:
                    raise GitHubBoundaryError(f"gh {name} exceeds the fixed size bound")
                buffer.extend(chunk)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise subprocess.TimeoutExpired(argv, _SUBPROCESS_TIMEOUT_SECONDS)
        returncode = process.wait(timeout=remaining)
        return subprocess.CompletedProcess(argv, returncode, bytes(stdout), bytes(stderr))
    except (GitHubBoundaryError, subprocess.TimeoutExpired):
        _stop_process(process)
        raise
    except Exception:
        _stop_process(process)
        raise
    finally:
        selector.close()


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


_REPO_SEGMENT = r"(?!\.{1,2}$)[A-Za-z0-9][A-Za-z0-9_.-]*"
_REPO = rf"{_REPO_SEGMENT}/{_REPO_SEGMENT}"
_SHA = r"[A-Za-z0-9][A-Za-z0-9_.:-]*"
_LOGIN = r"[A-Za-z0-9][A-Za-z0-9-]{0,38}(?:\[bot\])?"
_MAX_PAGINATED_PAGES = 16
_MAX_PAGINATED_ITEMS = 4096
_MAX_RESPONSE_BYTES = 16 * 1024 * 1024
_MAX_STDERR_BYTES = 1 << 20
_MAX_REQUEST_BYTES = 1 << 20
_PAGE_SIZE = 100
_PROBE_PAGE_SIZE = 1
_PROTOCOL_RECORD_TYPES = {"intent", "report", "completion", "review-metadata", "revocation"}
_PROTOCOL_FIELDS = {
    "intent": {"schema", "record_type", "record_id", "target", "target_key", "attempt_id", "canonical_digest"},
    "report": {
        "schema", "record_type", "record_id", "target", "target_key", "attempt_id", "intent_record_id",
        "head_sha", "canonical_intent_node_id", "canonical_intent_digest", "report_body", "report_body_sha256",
    },
    "review-metadata": {
        "schema", "record_type", "record_id", "target", "target_key", "attempt_id", "intent_record_id", "head_sha",
        "report_record_id", "report_node_id", "report_digest", "report_body_sha256", "canonical_intent_digest",
        "canonical_intent_node_id", "metadata_digest",
    },
    "completion": {
        "schema", "record_type", "record_id", "target", "target_key", "attempt_id", "intent_record_id", "head_sha",
        "canonical_intent_digest", "canonical_intent_node_id", "report_record_id", "report_node_id", "report_digest",
        "metadata_record_id", "metadata_digest",
    },
    "revocation": {"schema", "record_type", "record_id", "target_key", "attempt_id", "canonical_intent_digest", "reason"},
}
_ACCEPTED_PERMISSION_RE = re.compile(r"([A-Za-z0-9_-]+)\s*=\s*(read|write|admin)")
_ENDPOINTS = (
    ("GET", re.compile(rf"/user$"), False),
    ("GET", re.compile(rf"/repos/{_REPO}$"), False),
    ("GET", re.compile(rf"/repos/{_REPO}/pulls$"), True),
    ("GET", re.compile(rf"/repos/{_REPO}/pulls/[1-9][0-9]*$"), False),
    ("GET", re.compile(rf"/repos/{_REPO}/issues/[1-9][0-9]*/comments$"), True),
    ("GET", re.compile(rf"/repos/{_REPO}/issues/comments/[1-9][0-9]*$"), False),
    ("GET", re.compile(rf"/repos/{_REPO}/pulls/[1-9][0-9]*/reviews$"), True),
    ("GET", re.compile(rf"/repos/{_REPO}/pulls/[1-9][0-9]*/reviews/[1-9][0-9]*$"), False),
    ("GET", re.compile(r"/installation/repositories$"), False),
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


def _repository(value: str) -> str:
    if not isinstance(value, str) or re.fullmatch(_REPO, value) is None:
        raise GitHubBoundaryError("repository identifier is unsafe")
    return value


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise GitHubBoundaryError(f"{name} must be a positive integer")
    return value


def _identifier(value: str, name: str, pattern: str = _SHA) -> str:
    if not isinstance(value, str) or re.fullmatch(pattern, value) is None or value in {".", ".."}:
        raise GitHubBoundaryError(f"{name} identifier is unsafe")
    return value


def _login(value: str) -> str:
    return _identifier(value, "login", _LOGIN)


def _label(value: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or any(ord(char) < 0x20 or char in "/\\?#%" for char in value)
    ):
        raise GitHubBoundaryError("label identifier is unsafe")
    return value


def _safe_path(path: str) -> bool:
    return not any(segment in {".", ".."} for segment in path.split("/")) and not any(
        char in path for char in "\x00\r\n?#\\@"
    )


def _validate_protocol_payload(payload: Mapping[str, Any]) -> None:
    record_type = payload.get("record_type")
    if payload.get("schema") != "agentic-review/v1" or record_type not in _PROTOCOL_RECORD_TYPES:
        raise ValueError("protocol body has an invalid schema or record type")
    if set(payload) != _PROTOCOL_FIELDS[record_type]:
        raise ValueError("protocol body has unexpected or missing fields")
    if not isinstance(payload.get("record_id"), str) or not payload["record_id"].strip():
        raise ValueError("protocol body has no record identity")
    if record_type == "intent":
        IntentPayload.from_mapping(payload)
    elif record_type == "report":
        body = payload["report_body"]
        digest = hashlib.sha256(body.encode("utf-8")).hexdigest() if isinstance(body, str) else ""
        if payload["report_body_sha256"] not in {digest, "sha256:" + digest}:
            raise ValueError("protocol report body digest does not match")
    elif record_type == "review-metadata" and payload["metadata_digest"] != metadata_digest(payload):
        raise ValueError("protocol metadata digest does not match")
    else:
        canonical_digest(payload)


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
        except (ValueError, json.JSONDecodeError, RecursionError) as exc:
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
        if not text[offset:].strip():
            value, consumed = None, len(text) - offset
        else:
            try:
                value, consumed = decoder.raw_decode(text[offset:])
            except (ValueError, json.JSONDecodeError, RecursionError) as exc:
                raise GitHubBoundaryError("GitHub response contains invalid JSON") from exc
        pages.append((status, headers, value))
        if len(pages) > _MAX_PAGINATED_PAGES:
            raise GitHubBoundaryError("GitHub pagination exceeds the fixed page bound")
        offset += consumed
    if not pages:
        raise GitHubBoundaryError("unexpected pagination response")
    return pages, len(pages) > 1


_LINK_RE = re.compile(r'<([^<>]+)>\s*;\s*rel="([^"]+)"\s*$')


def _has_next_page(headers: Mapping[str, str]) -> bool:
    raw = headers.get("link")
    if raw is None:
        return False
    if not isinstance(raw, str) or not raw.strip():
        raise GitHubBoundaryError("GitHub Link header is malformed")
    links = [part.strip() for part in raw.split(",")]
    parsed = []
    for link in links:
        match = _LINK_RE.fullmatch(link)
        if match is None or not match.group(1).startswith(("https://", "http://")):
            raise GitHubBoundaryError("GitHub Link header is malformed")
        parsed.append(match.group(2).split())
    return any("next" in relations for relations in parsed)


def _as_result(result: Any) -> tuple[int, str, str]:
    if isinstance(result, subprocess.CompletedProcess):
        return _as_result((result.returncode, result.stdout or "", result.stderr or ""))
    if isinstance(result, Mapping):
        return _as_result((result.get("returncode", 0), result.get("stdout", ""), result.get("stderr", "")))
    if isinstance(result, tuple) and len(result) == 3:
        returncode = result[0]
        if isinstance(returncode, bool) or not isinstance(returncode, int):
            raise GitHubBoundaryError("runner returned an invalid exit status")
        stdout, stderr = result[1], result[2]
        for value, limit, name in ((stdout, _MAX_RESPONSE_BYTES, "stdout"), (stderr, _MAX_STDERR_BYTES, "stderr")):
            if not isinstance(value, (str, bytes)):
                raise GitHubBoundaryError(f"runner returned invalid {name}")
            size = len(value) if isinstance(value, bytes) else len(value.encode())
            if size > limit:
                raise GitHubBoundaryError(f"gh {name} exceeds the fixed size bound")
        return returncode, stdout, stderr
    raise GitHubBoundaryError("runner returned an unsupported result")


class GitHubClient:
    def __init__(self, runner: Runner = _subprocess_runner, *, gh_binary: str = "gh"):
        self._runner = runner
        self._gh_binary = gh_binary

    def _allowed(self, method: str, path: str) -> bool:
        return _safe_path(path) and any(
            method == allowed_method and pattern.fullmatch(path) for allowed_method, pattern, _ in _ENDPOINTS
        )

    def _request(
        self,
        method: str,
        path: str,
        *,
        query: Mapping[str, str | int] | None = None,
        fields: Mapping[str, str] | None = None,
        json_body: Any | None = None,
        paginate: bool = False,
    ) -> GitHubResponse:
        if not self._allowed(method, path):
            raise GitHubBoundaryError("GitHub path or method is not allowlisted")
        if paginate:
            raise GitHubBoundaryError("unbounded pagination is disabled; use bounded page requests")
        argv = [self._gh_binary, "api"]
        if paginate:
            argv.append("--paginate")
        argv.extend(["--include", "--method", method])
        request_path = path
        if query:
            if not isinstance(query, Mapping) or any(
                not isinstance(key, str) or not key or not isinstance(value, (str, int)) or isinstance(value, bool)
                for key, value in query.items()
            ):
                raise GitHubBoundaryError("query parameters are malformed")
            request_path += "?" + "&".join(
                f"{quote(key, safe='')}={quote(str(value), safe='')}" for key, value in query.items()
            )
        argv.append(request_path)
        input_data: bytes | None = None
        if fields is not None:
            raise GitHubBoundaryError("field encoding is disabled for untrusted API values")
        if json_body is not None:
            argv.extend(["--input", "-"])
            try:
                input_data = json.dumps(json_body, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
            except (TypeError, UnicodeError, ValueError) as exc:
                raise GitHubBoundaryError("mutation body is not strict JSON") from exc
            if len(input_data) > _MAX_REQUEST_BYTES:
                raise GitHubBoundaryError("mutation body exceeds the fixed size bound")
        try:
            result = self._runner(argv, input_data)
        except subprocess.TimeoutExpired as exc:
            raise GitHubBoundaryError("gh subprocess timed out") from exc
        except GitHubBoundaryError:
            raise
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
        _capability_signal(response)
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
        repository = _repository(repository)
        response = self._request("GET", f"/repos/{repository}")
        data = self._require_mapping(response.data, "repository")
        self._require(data, ("id", "full_name"), "repository")
        if (
            isinstance(data["id"], bool)
            or not isinstance(data["id"], int)
            or data["id"] <= 0
            or not isinstance(data["full_name"], str)
            or not data["full_name"].strip()
            or data["full_name"] != repository
        ):
            raise GitHubBoundaryError("GitHub repository response has malformed identity")
        return response

    def list_installation_repositories(self) -> GitHubResponse:
        repositories: list[Mapping[str, Any]] = []
        headers: dict[str, str] = {}
        total_count: int | None = None
        for page in range(1, _MAX_PAGINATED_PAGES + 1):
            response = self._request(
                "GET", "/installation/repositories", query={"per_page": _PAGE_SIZE, "page": page}
            )
            data = self._require_mapping(response.data, "installation repositories")
            page_total = data.get("total_count")
            if isinstance(page_total, bool) or not isinstance(page_total, int) or page_total < 0:
                raise GitHubBoundaryError("GitHub installation repositories total count is malformed")
            if total_count is None:
                total_count = page_total
            elif page_total != total_count:
                raise GitHubBoundaryError("GitHub installation repositories total count changed")
            if total_count > _MAX_PAGINATED_ITEMS:
                raise GitHubBoundaryError("GitHub installation repository pagination exceeds the fixed item bound")
            page_repositories = data.get("repositories")
            if not isinstance(page_repositories, list):
                raise GitHubBoundaryError("GitHub installation repositories response is malformed")
            if len(repositories) + len(page_repositories) > _MAX_PAGINATED_ITEMS:
                raise GitHubBoundaryError("GitHub installation repository pagination exceeds the fixed item bound")
            for repository in page_repositories:
                item = self._require_mapping(repository, "installation repository")
                if isinstance(item.get("id"), bool) or not isinstance(item.get("id"), int) or item["id"] <= 0:
                    raise GitHubBoundaryError("GitHub installation repository identity is malformed")
                repositories.append(item)
            if len(repositories) > total_count:
                raise GitHubBoundaryError("GitHub installation repositories exceed the reported total count")
            headers.update(response.headers)
            has_next = _has_next_page(response.headers)
            required_pages = (total_count + _PAGE_SIZE - 1) // _PAGE_SIZE
            if required_pages > _MAX_PAGINATED_PAGES:
                raise GitHubBoundaryError("GitHub installation repository pagination exceeds the fixed page bound")
            if page == _MAX_PAGINATED_PAGES and has_next:
                raise GitHubBoundaryError("GitHub installation repository pagination reached its fixed page bound")
            if not has_next:
                if len(repositories) < total_count:
                    raise GitHubBoundaryError("GitHub installation repositories response is incomplete")
                break
        else:
            raise GitHubBoundaryError("GitHub installation repository pagination reached its fixed page bound")
        return GitHubResponse({"total_count": total_count, "repositories": repositories}, headers, 200)

    def list_pull_requests(self, repository: str, *, max_pages: int = _MAX_PAGINATED_PAGES) -> GitHubResponse:
        return self._list_pull_requests(repository, max_pages=max_pages, page_size=_PAGE_SIZE, allow_incomplete=False)

    def _list_pull_requests(
        self, repository: str, *, max_pages: int, page_size: int, allow_incomplete: bool
    ) -> GitHubResponse:
        repository = _repository(repository)
        if isinstance(max_pages, bool) or not isinstance(max_pages, int) or not 0 < max_pages <= _MAX_PAGINATED_PAGES:
            raise GitHubBoundaryError("max_pages must be within the fixed positive bound")
        responses = []
        for page in range(1, max_pages + 1):
            response = self._request(
                "GET", f"/repos/{repository}/pulls", query={"per_page": page_size, "page": page}
            )
            if not isinstance(response.data, list):
                raise GitHubBoundaryError("pull request response is not a list")
            responses.append(response)
            if not _has_next_page(response.headers):
                break
            if len(responses) == max_pages:
                if allow_incomplete:
                    break
                raise GitHubBoundaryError("GitHub pull request scan is incomplete at the fixed page bound")
        data = []
        headers = {}
        for response in responses:
            if len(data) + len(response.data) > _MAX_PAGINATED_ITEMS:
                raise GitHubBoundaryError("GitHub pull request scan exceeds the fixed item bound")
            data.extend(response.data)
            headers.update(response.headers)
        for item in data:
            self._validate_pull(self._require_mapping(item, "pull request"), expected_repository=repository)
        return GitHubResponse(data, headers, 200)

    def sample_pull_requests(self, repository: str) -> GitHubResponse:
        """Return one bounded probe page without claiming exhaustive discovery."""
        return self._list_pull_requests(repository, max_pages=1, page_size=_PROBE_PAGE_SIZE, allow_incomplete=True)

    @classmethod
    def _validate_pull(cls, data: Mapping[str, Any], *, expected_number: int | None = None, expected_repository: str | None = None) -> None:
        cls._require(data, ("number", "node_id", "head", "base"), "pull request")
        if (
            isinstance(data["number"], bool)
            or not isinstance(data["number"], int)
            or data["number"] <= 0
            or (expected_number is not None and data["number"] != expected_number)
            or not isinstance(data["node_id"], str)
            or not data["node_id"].strip()
        ):
            raise GitHubBoundaryError("GitHub pull request number or node ID does not match")
        head = cls._require(cls._require_mapping(data["head"], "pull request head"), ("repo", "sha"), "pull request head")
        base = cls._require(cls._require_mapping(data["base"], "pull request base"), ("ref", "sha"), "pull request base")
        head_repo = cls._require(cls._require_mapping(head["repo"], "head repository"), ("full_name",), "head repository")
        cls._require(base, ("ref", "sha"), "pull request base")
        for value, name in ((head_repo["full_name"], "head repository"), (head["sha"], "head SHA"), (base["ref"], "base ref"), (base["sha"], "base SHA")):
            if not isinstance(value, str) or not value.strip():
                raise GitHubBoundaryError(f"GitHub pull request {name} is malformed")
        if expected_repository is not None:
            base_repo = data.get("base", {}).get("repo")
            if isinstance(base_repo, Mapping) and base_repo.get("full_name") != expected_repository:
                raise GitHubBoundaryError("GitHub pull request repository does not match")

    def get_pull_request(self, repository: str, number: int) -> GitHubResponse:
        repository = _repository(repository)
        number = _positive_integer(number, "pull request number")
        response = self._request("GET", f"/repos/{repository}/pulls/{number}")
        data = self._require_mapping(response.data, "pull request")
        self._validate_pull(data, expected_number=number, expected_repository=repository)
        return response

    def list_issue_comments(self, repository: str, number: int) -> GitHubResponse:
        return self._list_records(repository, number, "issue comments")

    def list_pull_reviews(self, repository: str, number: int) -> GitHubResponse:
        return self._list_records(repository, number, "pull reviews")

    def get_issue_comment(self, repository: str, comment_id: int) -> GitHubResponse:
        repository = _repository(repository)
        comment_id = _positive_integer(comment_id, "comment ID")
        response = self._request("GET", f"/repos/{repository}/issues/comments/{comment_id}")
        self._validate_record_response(
            response, "issue comment", record_kind="comment", extra=("body",), expected_id=comment_id
        )
        return response

    def get_pull_review(self, repository: str, number: int, review_id: int) -> GitHubResponse:
        repository = _repository(repository)
        number = _positive_integer(number, "pull request number")
        review_id = _positive_integer(review_id, "review ID")
        response = self._request("GET", f"/repos/{repository}/pulls/{number}/reviews/{review_id}")
        self._validate_record_response(
            response, "pull review", record_kind="review",
            extra=("state", "commit_id", "body"), expected_id=review_id, require_timestamp=False,
        )
        return response

    @classmethod
    def _validate_record_response(
        cls, response: GitHubResponse, name: str, *, record_kind: str,
        extra: Sequence[str], expected_id: int | None = None, require_timestamp: bool = True,
    ) -> None:
        record = cls._require_mapping(response.data, name)
        cls._validate_api_record(
            record, name, record_kind=record_kind, extra=extra, require_timestamp=require_timestamp
        )
        if expected_id is not None and record["id"] != expected_id:
            raise GitHubBoundaryError(f"GitHub {name} response ID does not match requested ID")
        author = cls._require(cls._require_mapping(record["user"], f"{name} author"), ("login", "type"), f"{name} author")
        if (
            not isinstance(author["login"], str)
            or not author["login"].strip()
            or not isinstance(author["type"], str)
            or author["type"] not in _PRINCIPAL_TYPES
        ):
            raise GitHubBoundaryError(f"GitHub {name} author has an unsupported principal type")

    def _list_records(self, repository: str, number: int, name: str) -> GitHubResponse:
        repository = _repository(repository)
        number = _positive_integer(number, "issue or pull request number")
        path = f"/repos/{repository}/issues/{number}/comments" if name == "issue comments" else f"/repos/{repository}/pulls/{number}/reviews"
        records: list[Any] = []
        headers: dict[str, str] = {}
        for page in range(1, _MAX_PAGINATED_PAGES + 1):
            response = self._request("GET", path, query={"per_page": _PAGE_SIZE, "page": page})
            headers.update(response.headers)
            has_next = _has_next_page(response.headers)
            if page == _MAX_PAGINATED_PAGES and has_next:
                raise GitHubBoundaryError(f"GitHub {name} pagination reached its fixed page bound")
            if not isinstance(response.data, list):
                raise GitHubBoundaryError(f"GitHub {name} response is not a list")
            if len(records) + len(response.data) > _MAX_PAGINATED_ITEMS:
                raise GitHubBoundaryError(f"GitHub {name} pagination exceeds the fixed item bound")
            for item in response.data:
                record = self._require_mapping(item, name)
                if name == "issue comments":
                    record_kind, extra = "comment", ("body",)
                else:
                    record_kind, extra = "review", ("state", "commit_id")
                self._validate_record_response(
                    GitHubResponse(record, response.headers, response.status_code), name,
                    record_kind=record_kind, extra=extra, require_timestamp=False,
                )
                records.append(record)
            if len(records) > _MAX_PAGINATED_ITEMS or not has_next:
                break
        else:
            raise GitHubBoundaryError(f"GitHub {name} pagination reached its fixed page bound")
        return GitHubResponse(records, headers, 200)

    @staticmethod
    def _validate_api_record(
        record: Mapping[str, Any], name: str, *, record_kind: str, extra: Sequence[str] = (),
        require_timestamp: bool = True,
    ) -> None:
        timestamps = ("created_at", "updated_at") if record_kind == "comment" else ("submitted_at",)
        GitHubClient._require(record, ("id", "node_id", "user", *extra), name)
        if (
            isinstance(record["id"], bool)
            or not isinstance(record["id"], int)
            or record["id"] <= 0
            or not isinstance(record["node_id"], str)
            or not record["node_id"].strip()
        ):
            raise GitHubBoundaryError(f"GitHub {name} response has malformed server fields")
        for field in timestamps:
            if (
                record_kind == "review"
                and not require_timestamp
                and record.get("state") == "PENDING"
                and record.get(field) in (None, "")
            ):
                continue
            if field not in record or record[field] in (None, ""):
                raise GitHubBoundaryError(f"GitHub {name} response has a missing {field} timestamp")
            try:
                parsed = datetime.fromisoformat(record[field].replace("Z", "+00:00"))
            except (AttributeError, TypeError, ValueError) as exc:
                raise GitHubBoundaryError(f"GitHub {name} response has an invalid {field} timestamp") from exc
            if parsed.tzinfo is None:
                raise GitHubBoundaryError(f"GitHub {name} response has an invalid {field} timestamp")

    def collaborator_effective_permission(self, repository: str, login: str) -> EffectivePermission:
        repository = _repository(repository)
        login = _login(login)
        response = self._request("GET", f"/repos/{repository}/collaborators/{quote(login, safe='')}/permission")
        _capability_signal(response, required=("metadata",))
        data = self._require_mapping(response.data, "collaborator permission")
        self._require(data, ("user", "permissions"), "collaborator permission")
        principal = self._require(self._require_mapping(data["user"], "collaborator"), ("login", "type"), "collaborator")
        if principal.get("login") != login:
            raise GitHubBoundaryError("collaborator response login does not match requested login")
        if not isinstance(principal["type"], str) or principal["type"] not in _PRINCIPAL_TYPES:
            raise GitHubBoundaryError("collaborator has an unsupported principal type")
        permissions = self._require_mapping(data["permissions"], "permissions")
        permission_map = {"admin": "admin", "maintain": "write", "push": "write", "triage": "read", "pull": "read"}
        permission = next((permission_map[name] for name in _PERMISSIONS if permissions.get(name) is True), None)
        if permission is None:
            role_map = {"read": "read", "write": "write", "push": "write", "maintain": "write", "triage": "read", "pull": "read", "admin": "admin"}
            role = data.get("role_name")
            permission = role_map.get(role) if isinstance(role, str) else None
        if permission is None:
            raise GitHubBoundaryError("effective collaborator permission is missing")
        return EffectivePermission(principal["login"], principal["type"], permission)

    def get_tree(self, repository: str, tree_sha: str, *, recursive: bool = False) -> GitHubResponse:
        repository = _repository(repository)
        tree_sha = _identifier(tree_sha, "tree SHA")
        query = {"recursive": "1"} if recursive else None
        response = self._request("GET", f"/repos/{repository}/git/trees/{tree_sha}", query=query)
        data = self._require_mapping(response.data, "tree")
        self._require(data, ("sha", "tree"), "tree")
        if data["sha"] != tree_sha or not isinstance(data["tree"], list):
            raise GitHubBoundaryError("GitHub tree sha or entries do not match request")
        for entry in data["tree"]:
            item = self._require_mapping(entry, "tree entry")
            self._require(item, ("path", "mode", "type", "sha"), "tree entry")
            if any(not isinstance(item[field], str) or not item[field].strip() for field in ("path", "mode", "type", "sha")):
                raise GitHubBoundaryError("GitHub tree entry is malformed")
        return response

    def get_blob(self, repository: str, blob_sha: str) -> GitHubResponse:
        repository = _repository(repository)
        blob_sha = _identifier(blob_sha, "blob SHA")
        response = self._request("GET", f"/repos/{repository}/git/blobs/{blob_sha}")
        data = self._require_mapping(response.data, "blob")
        self._require(data, ("sha", "content", "encoding"), "blob")
        if data["sha"] != blob_sha or not isinstance(data["content"], str) or data["encoding"] != "base64":
            raise GitHubBoundaryError("GitHub blob sha identity or encoding does not match request")
        return response

    def add_labels(self, repository: str, number: int, labels: Sequence[str]) -> GitHubResponse:
        repository = _repository(repository)
        number = _positive_integer(number, "issue number")
        if not labels or any(_label(label) != label for label in labels):
            raise GitHubBoundaryError("labels must be non-empty strings")
        response = self._request("POST", f"/repos/{repository}/issues/{number}/labels", json_body={"labels": list(labels)})
        self._validate_mutation_list(response, "labels")
        return response

    def remove_label(self, repository: str, number: int, label: str) -> GitHubResponse:
        repository = _repository(repository)
        number = _positive_integer(number, "issue number")
        label = _label(label)
        response = self._request("DELETE", f"/repos/{repository}/issues/{number}/labels/{quote(label, safe='')}")
        if response.status_code not in {200, 204}:
            raise GitHubBoundaryError("unexpected label deletion response")
        return response

    def create_issue_comment(self, repository: str, number: int, body: str) -> GitHubResponse:
        repository = _repository(repository)
        number = _positive_integer(number, "issue number")
        if not isinstance(body, str) or not body:
            raise GitHubBoundaryError("comment body must be non-empty")
        response = self._request("POST", f"/repos/{repository}/issues/{number}/comments", json_body={"body": body})
        self._validate_mutation_object(response, "comment")
        return response

    def create_pull_request_review(self, repository: str, number: int, *, body: str, event: str, commit_id: str) -> GitHubResponse:
        repository = _repository(repository)
        number = _positive_integer(number, "pull request number")
        commit_id = _identifier(commit_id, "commit")
        if not isinstance(body, str) or not body or event not in {"APPROVE", "REQUEST_CHANGES", "COMMENT"}:
            raise GitHubBoundaryError("review body, event, and exact commit_id are required")
        response = self._request(
            "POST", f"/repos/{repository}/pulls/{number}/reviews",
            json_body={"body": body, "event": event, "commit_id": commit_id},
        )
        self._validate_mutation_object(response, "review")
        return response

    def dismiss_workflow_review(self, repository: str, number: int, review_id: int, *, message: str) -> GitHubResponse:
        repository = _repository(repository)
        number = _positive_integer(number, "pull request number")
        review_id = _positive_integer(review_id, "review ID")
        if not isinstance(message, str) or not message:
            raise GitHubBoundaryError("dismissal message must be non-empty")
        response = self._request(
            "PUT", f"/repos/{repository}/pulls/{number}/reviews/{review_id}/dismissals", json_body={"message": message}
        )
        self._validate_mutation_object(response, "dismissal")
        return response

    @staticmethod
    def _validate_mutation_object(response: GitHubResponse, name: str) -> None:
        data = GitHubClient._require_mapping(response.data, name)
        if "id" not in data or "node_id" not in data:
            raise GitHubBoundaryError(f"GitHub {name} response is missing id fields")
        if (
            isinstance(data["id"], bool)
            or not isinstance(data["id"], int)
            or data["id"] <= 0
            or not isinstance(data["node_id"], str)
            or not data["node_id"].strip()
        ):
            raise GitHubBoundaryError(f"GitHub {name} response has malformed id fields")

    @staticmethod
    def _validate_mutation_list(response: GitHubResponse, name: str) -> None:
        if not isinstance(response.data, list):
            raise GitHubBoundaryError(f"GitHub {name} response is not a list")
        for item in response.data:
            GitHubClient._validate_mutation_object(GitHubResponse(item, response.headers, response.status_code), name)

    def _envelope(self, record: Mapping[str, Any], *, record_name: str = "authenticated GitHub record") -> GitHubEnvelope:
        try:
            author = record["user"]
            record_kind = "comment" if record_name == "issue comment" else "review"
            extra = ("body",) if record_kind == "comment" else ("state", "commit_id")
            self._validate_api_record(record, record_name, record_kind=record_kind, extra=extra)
            self._require(author, ("login", "type"), "authenticated author")
        except (KeyError, TypeError, RecursionError) as exc:
            raise GitHubBoundaryError("authenticated record is missing server fields") from exc
        if (
            not isinstance(author["login"], str)
            or not author["login"].strip()
            or not isinstance(author["type"], str)
            or author["type"] not in _PRINCIPAL_TYPES
        ):
            raise GitHubBoundaryError("authenticated author has unsupported principal type")
        if record_kind == "comment" and record["updated_at"] != record["created_at"]:
            raise GitHubBoundaryError("edited GitHub record is not admissible")
        if record_kind == "review" and record["state"] == "PENDING":
            raise GitHubBoundaryError("pending GitHub review is not an immutable authenticated envelope")
        body = record["body"]
        if not isinstance(body, str) or not body.strip():
            raise GitHubBoundaryError(f"GitHub {record_name} body is missing")
        try:
            payload = canonical_loads(body)
            if not isinstance(payload, Mapping):
                raise ValueError("protocol body must be an object")
            _validate_protocol_payload(payload)
        except (TypeError, ValueError, RecursionError) as exc:
            raise GitHubBoundaryError(f"GitHub {record_name} body is not a valid protocol payload") from exc
        publication_time = record["created_at"] if record_kind == "comment" else record["submitted_at"]
        return GitHubEnvelope(
            payload, record["node_id"], author["login"], publication_time, publication_time, author["type"]
        )

    def comment_envelope(self, repository: str, comment_id: int) -> GitHubEnvelope:
        response = self.get_issue_comment(repository, comment_id)
        return self._envelope(response.data, record_name="issue comment")

    def review_envelope(self, repository: str, number: int, review_id: int) -> GitHubEnvelope:
        response = self.get_pull_review(repository, number, review_id)
        return self._envelope(response.data, record_name="pull request review")


def _capability_signal(
    response: GitHubResponse, *, required: Sequence[str] = ()
) -> tuple[tuple[str, ...], Mapping[str, str]]:
    raw_scopes = response.headers.get("x-oauth-scopes")
    scopes: tuple[str, ...] = ()
    if raw_scopes is not None:
        if not isinstance(raw_scopes, str):
            raise PreflightError("OAuth scope header is malformed")
        if raw_scopes.strip():
            pieces = tuple(scope.strip() for scope in raw_scopes.split(","))
            if any(not scope or re.fullmatch(r"[A-Za-z0-9:_-]+", scope) is None for scope in pieces):
                raise PreflightError("OAuth scope header is malformed")
            scopes = pieces
            if "repo" in scopes:
                raise PreflightError("classic repo OAuth scope is not permitted")
    raw_permissions = response.headers.get("x-accepted-github-permissions")
    accepted: dict[str, str] = {}
    if raw_permissions is not None:
        if not isinstance(raw_permissions, str) or not raw_permissions.strip():
            raise PreflightError("GitHub permission header is malformed")
        for item in re.split(r"[,;]", raw_permissions):
            match = _ACCEPTED_PERMISSION_RE.fullmatch(item.strip())
            if match is None:
                raise PreflightError("GitHub permission header is malformed")
            accepted[match.group(1)] = match.group(2)
    if not scopes and not accepted:
        raise PreflightError("no usable GitHub capability signal (scope or permission header) is visible")
    rank = {"read": 1, "write": 2, "admin": 3}
    if accepted:
        missing = [permission for permission in required if permission not in accepted or rank[accepted[permission]] < rank["read"]]
        if missing:
            raise PreflightError(f"required GitHub permission is absent: {', '.join(missing)}")
    return scopes, accepted


def _scope_header(response: GitHubResponse) -> tuple[str, ...]:
    return _capability_signal(response)[0]


def _validate_app_token_repository_access(
    client: GitHubClient,
    repository: str,
    repository_id: int,
    trusted: Mapping[str, Any],
    *,
    operator_manifest: Mapping[str, Any] | None,
    mode: str,
) -> tuple[str, str, tuple[str, ...]]:
    """Validate deployment binding plus repositories visible to an App token.

    An installation token cannot prove the App bot identity here. That
    identity is checked later from the server-supplied author on the exact
    comment or review envelope.

    The configured App ID and installation ID are deployment assertions;
    installation-token API calls are intentionally not used to re-prove them.
    """
    apps = [app for app in trusted["apps"] if app["repository_id"] == repository_id]
    if len(apps) != 1:
        raise PreflightError("publisher requires exactly one trusted App binding for the repository")
    configured = apps[0]
    repositories_response = client.list_installation_repositories()
    scopes, _ = _capability_signal(repositories_response, required=("metadata",))
    repositories_data = client._require_mapping(repositories_response.data, "installation repositories")
    if not any(
        isinstance(item, Mapping) and item.get("id") == repository_id
        for item in repositories_data["repositories"]
    ):
        raise PreflightError("GitHub App installation does not include the target repository")
    if operator_manifest is not None:
        principal = operator_manifest["principal"]
        if principal["type"] != "Bot" or principal["login"] != configured["login"]:
            raise PreflightError("operator manifest does not match the trusted App")
        if operator_manifest["credential_attestation_digest"] != configured["credential_attestation_digest"]:
            raise PreflightError("operator manifest does not match the trusted App attestation")
        operation = {
            "discovery": "discover",
            "publisher": "publish",
            "dismissal": "dismiss-workflow-review",
        }[mode]
        if operation not in operator_manifest["allowed_operations"]:
            raise PreflightError("operator manifest does not attest to the requested operation")
    return configured["login"], "Bot", scopes


def preflight_read_only(
    client: GitHubClient,
    repository: str,
    *,
    mode: str,
    configuration: ReviewConfiguration,
    operator_manifest: Mapping[str, Any] | None = None,
    pull_number: int | None = None,
) -> PreflightResult:
    if mode not in {"discovery", "controller", "publisher", "dismissal"}:
        raise PreflightError("unsupported preflight mode")
    trusted = configuration.trusted_publishers
    try:
        validate_trusted_publishers_policy(trusted)
    except ValueError as exc:
        raise PreflightError(str(exc)) from exc
    write_mode = mode in {"discovery", "publisher", "dismissal"}
    operator: Mapping[str, Any] | None = None
    if write_mode and operator_manifest is None:
        raise PreflightError("mutation-capable preflight modes require an operator credential manifest")
    if write_mode:
        assert operator_manifest is not None
        try:
            validate_operator_credential_manifest(operator_manifest)
        except ValueError as exc:
            raise PreflightError(str(exc)) from exc
        operator = operator_manifest
        if operator["principal"]["type"] not in {"User", "Bot"}:
            raise PreflightError("publisher operator principal type is unsupported")
        if operator["repository"] != repository:
            raise PreflightError("operator manifest repository does not match the preflight repository")
        operation = {
            "discovery": "discover",
            "publisher": "publish",
            "dismissal": "dismiss-workflow-review",
        }[mode]
        if operation not in operator["allowed_operations"]:
            raise PreflightError("operator manifest does not attest to the requested operation")
        required_permissions = {"pull_requests"} if mode == "dismissal" else {"issues", "pull_requests"}
        if any(operator["write_permissions"].get(permission) not in {"write", "admin"} for permission in required_permissions):
            raise PreflightError("operator manifest does not declare the required intended write permissions")
    try:
        user_response = None
        user_data: Mapping[str, Any] | None = None
        scopes: tuple[str, ...] = ()
        operator_principal_type = operator["principal"]["type"] if operator is not None else ""
        use_app_token = write_mode and operator_principal_type == "Bot"
        if not use_app_token:
            user_response = client.get_authenticated_user()
            scopes, _ = _capability_signal(user_response)
            user_data = client._require_mapping(user_response.data, "user")
        repo_response = client.get_repository(repository)
        _capability_signal(repo_response, required=("metadata",))
        repository_data = client._require_mapping(repo_response.data, "repository")
        pulls_response = client.sample_pull_requests(repository)
        _capability_signal(pulls_response, required=("pull_requests",))
        if user_data is not None:
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
        else:
            user_login = principal_type = ""
        if (
            not isinstance(repository_data.get("id"), int)
            or isinstance(repository_data.get("id"), bool)
            or repository_data["id"] <= 0
            or repository_data.get("full_name") != repository
        ):
            raise PreflightError("repository identity is malformed")
        open_pulls = pulls_response.data
        if not open_pulls:
            raise PreflightError("no open pull request is available for the selected preflight")
        probe_number = pull_number if pull_number is not None else open_pulls[0]["number"]
        probe_number = _positive_integer(probe_number, "pull request number")
        pull_response = client.get_pull_request(repository, probe_number)
        _capability_signal(pull_response, required=("pull_requests",))
        pull_data = pull_response.data
        if mode in {"discovery", "publisher", "dismissal"}:
            comments_response = client.list_issue_comments(repository, probe_number)
            _capability_signal(comments_response, required=("issues",))
            reviews_response = client.list_pull_reviews(repository, probe_number)
            _capability_signal(reviews_response, required=("pull_requests",))
        if mode == "controller":
            tree_response = client.get_tree(repository, pull_data["base"]["sha"], recursive=True)
            _capability_signal(tree_response, required=("contents",))
            blob_entries = [entry for entry in tree_response.data["tree"] if entry["type"] == "blob"]
            if not blob_entries:
                raise PreflightError("probe pull request tree has no blob entry")
            blob_response = client.get_blob(repository, blob_entries[0]["sha"])
            _capability_signal(blob_response, required=("contents",))
        if mode == "discovery" and user_data is not None:
            permission = client.collaborator_effective_permission(repository, user_login)
            if permission.principal_type != principal_type:
                raise PreflightError("effective permission principal type mismatch")
        elif not write_mode:
            permission = client.collaborator_effective_permission(repository, user_login)
            if permission.principal_type != principal_type:
                raise PreflightError("effective permission principal type mismatch")
        if write_mode and user_data is None:
            user_login, principal_type, scopes = _validate_app_token_repository_access(
                client, repository, repository_data["id"], trusted,
                operator_manifest=operator, mode=mode,
            )
        if write_mode and user_data is not None:
            if principal_type != "User" or operator is None:
                raise PreflightError("mutation-capable preflight requires an attested human User credential")
            manifest_principal = operator["principal"]
            operation = {
                "discovery": "discover",
                "publisher": "publish",
                "dismissal": "dismiss-workflow-review",
            }[mode]
            if (
                manifest_principal["login"] != user_login
                or manifest_principal["type"] != principal_type
                or operation not in operator["allowed_operations"]
            ):
                raise PreflightError("operator manifest does not attest to the requested operation")
    except (GitHubBoundaryError, KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, PreflightError):
            raise
        raise PreflightError(str(exc)) from exc
    return PreflightResult(user_login, principal_type, repository_data, scopes)
