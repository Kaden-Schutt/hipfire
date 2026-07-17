# Copyright (c) Kaden Schutt
"""Bounded, fail-closed discovery of pull requests needing agentic review."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from .config import ReviewConfiguration, validate_operator_credential_manifest
from .github import GitHubBoundaryError, decode_protocol_body
from .models import GitHubEnvelope, ReviewTarget, validate_trusted_publishers_policy
from .protocol import validate_protocol
from .publisher import ReviewPublisher


_LABEL = "needs-review"
_SCHEMA = "agentic-review/v1"
_MAX_REASON = 512


@dataclass(frozen=True)
class DiscoveryItem:
    number: int
    reason: str


@dataclass(frozen=True)
class DiscoverySummary:
    reviewed: tuple[DiscoveryItem, ...] = ()
    needs_review: tuple[DiscoveryItem, ...] = ()
    labelled: tuple[DiscoveryItem, ...] = ()
    clean: tuple[DiscoveryItem, ...] = ()
    incomplete: tuple[DiscoveryItem, ...] = ()
    errors: tuple[DiscoveryItem, ...] = ()

    @property
    def complete(self) -> bool:
        return not self.incomplete


@dataclass(frozen=True)
class _Record:
    envelope: GitHubEnvelope
    is_review: bool
    server_id: int
    state: str | None = None
    commit_id: str | None = None


def _data(response: Any) -> Any:
    return response.data if hasattr(response, "data") else response


def _reason(value: Any) -> str:
    text = str(value).strip() or "review state is incomplete"
    return text[:_MAX_REASON]


def _target_fields(target: Any) -> bool:
    return isinstance(target, ReviewTarget) and target.number > 0


def _configured_app(configuration: ReviewConfiguration, login: str, repository_id: int | None) -> Mapping[str, Any] | None:
    apps = configuration.trusted_publishers.get("apps", ())
    if not isinstance(apps, Sequence) or isinstance(apps, (str, bytes)):
        return None
    matches = [
        app for app in apps
        if isinstance(app, Mapping)
        and app.get("login") == login
        and (repository_id is None or app.get("repository_id") == repository_id)
    ]
    return matches[0] if len(matches) == 1 else None


def _trust(
    client: Any,
    repository: str,
    configuration: ReviewConfiguration,
    operator_credential: Mapping[str, Any],
) -> frozenset[str]:
    try:
        validate_trusted_publishers_policy(configuration.trusted_publishers)
        validate_operator_credential_manifest(operator_credential)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid discovery provenance: {exc}") from exc
    if operator_credential["repository"] != repository:
        raise ValueError("discovery operator repository does not match target repository")
    if "discover" not in operator_credential["allowed_operations"]:
        raise ValueError("discovery operator is missing discover operation")

    principal = operator_credential["principal"]
    login = principal["login"]
    if principal["type"] == "User":
        try:
            permission = client.collaborator_effective_permission(repository, login)
        except Exception as exc:
            raise GitHubBoundaryError(f"effective permission API failure: {exc}") from exc
        if getattr(permission, "login", None) != login or getattr(permission, "permission", None) not in {"write", "admin"}:
            raise ValueError("discovery operator lacks effective write permission")
        return frozenset({login})
    if principal["type"] != "Bot":
        raise ValueError("discovery operator principal must be a User or configured App")

    repository_id: int | None = None
    getter = getattr(client, "get_repository", None)
    if not callable(getter):
        raise ValueError("GitHub repository identity is required for App trust")
    try:
        data = _data(getter(repository))
    except Exception as exc:
        raise GitHubBoundaryError(f"repository identity API failure: {exc}") from exc
    repository_id = data.get("id") if isinstance(data, Mapping) else None
    if isinstance(repository_id, bool) or not isinstance(repository_id, int) or repository_id <= 0:
        raise ValueError("GitHub repository identity is unavailable for App trust")
    app = _configured_app(configuration, login, repository_id)
    if app is None:
        raise ValueError("discovery App is not configured with validated repository provenance")
    if app.get("credential_attestation_digest") != operator_credential["credential_attestation_digest"]:
        raise ValueError("discovery App attestation does not match configured provenance")
    return frozenset({login})


def _candidate_body(body: Any) -> bool:
    return isinstance(body, str) and (
        body.lstrip().startswith("{") or "<!-- agentic-review/v1" in body
    )


def _history(client: Any, target: ReviewTarget, trusted: frozenset[str]) -> tuple[tuple[_Record, ...], str | None]:
    try:
        comments = _data(client.list_issue_comments(target.repository, target.number))
        reviews = _data(client.list_pull_reviews(target.repository, target.number))
    except Exception as exc:
        return (), _reason(f"history API failure: {exc}")
    if not isinstance(comments, list) or not isinstance(reviews, list):
        return (), "history API returned a non-list record collection"

    records: list[_Record] = []
    for raw, is_review in [*((item, False) for item in comments), *((item, True) for item in reviews)]:
        if not isinstance(raw, Mapping):
            return (), "history contains a malformed API record"
        user = raw.get("user")
        login = user.get("login") if isinstance(user, Mapping) else None
        if login not in trusted:
            continue
        body = raw.get("body")
        if not isinstance(body, str):
            return (), "trusted workflow record has a malformed body"
        if not _candidate_body(body):
            continue
        try:
            payload = decode_protocol_body(body)
        except Exception as exc:
            return (), _reason(f"trusted workflow record is malformed: {exc}")
        if payload.get("schema") != _SCHEMA:
            continue
        if payload.get("record_type") not in {"intent", "report", "review-metadata", "completion", "revocation"}:
            return (), "trusted workflow record has an invalid record type"
        if not isinstance(payload.get("record_id"), str) or not payload["record_id"].strip():
            return (), "trusted workflow record has no record identity"
        if payload.get("record_type") != "revocation" and not isinstance(payload.get("target_key"), str):
            return (), "trusted workflow record has no target binding"
        try:
            if is_review:
                exact = client.get_pull_review_record(target.repository, target.number, raw["id"])
                envelope = exact.envelope
                record = _Record(envelope, True, exact.server_id, exact.state, exact.commit_id)
            else:
                envelope = client.comment_envelope(target.repository, raw["id"])
                record = _Record(envelope, False, raw["id"])
            if envelope.author not in trusted:
                continue
        except Exception as exc:
            return (), _reason(f"trusted workflow record is deleted, edited, or unavailable: {exc}")
        records.append(record)
    return tuple(records), None


def _target_from_record(record: _Record) -> ReviewTarget | None:
    value = record.envelope.payload.get("target")
    if isinstance(value, ReviewTarget):
        return value
    if not isinstance(value, Mapping):
        return None
    try:
        return ReviewTarget(**value)
    except (TypeError, ValueError):
        return None


def _current_completion(
    records: Sequence[_Record], target: ReviewTarget, trusted: frozenset[str]
) -> tuple[_Record, _Record, GitHubEnvelope] | str:
    current = []
    for record in records:
        parsed = _target_from_record(record)
        if parsed is not None and parsed == target:
            current.append(record)
        elif record.envelope.payload.get("target_key") == target.target_key():
            current.append(record)
        elif record.envelope.payload.get("record_type") == "revocation" and record.envelope.payload.get("target_key") == target.target_key():
            current.append(record)
    if not current:
        return "no complete current-target history"
    try:
        canonical = cast(
            GitHubEnvelope,
            validate_protocol([record.envelope for record in current], expected_target=target, trusted_authors=trusted),
        )
    except Exception as exc:
        return _reason(f"current review history is incomplete or invalid: {exc}")
    attempt_id = canonical.payload.get("attempt_id")
    completion = next(
        (record for record in current
         if record.envelope.payload.get("record_type") == "completion"
         and record.envelope.payload.get("attempt_id") == attempt_id),
        None,
    )
    if completion is None:
        return "no valid canonical agentic-review completion"
    metadata_id = completion.envelope.payload.get("metadata_record_id")
    metadata = next(
        (record for record in current
         if record.envelope.payload.get("record_type") == "review-metadata"
         and record.envelope.payload.get("record_id") == metadata_id),
        None,
    )
    if metadata is None:
        return "completion metadata is missing"
    if metadata.is_review and (metadata.state != "CHANGES_REQUESTED" or metadata.commit_id != target.head_sha):
        return "active requested-change review is missing or does not match the current head"
    return completion, metadata, canonical


def _label_present(client: Any, target: ReviewTarget) -> bool:
    data = _data(client.list_issue_labels(target.repository, target.number))
    if not isinstance(data, list):
        raise GitHubBoundaryError("GitHub label state is malformed")
    return any(isinstance(item, Mapping) and item.get("name") == _LABEL for item in data)


def _ensure_label(client: Any, target: ReviewTarget) -> bool:
    before = client.get_review_target(target.repository, target.number)
    if before != target:
        raise GitHubBoundaryError("target changed before needs-review labelling")
    if _label_present(client, target):
        return False
    try:
        client.add_labels(target.repository, target.number, [_LABEL])
    finally:
        after = client.get_review_target(target.repository, target.number)
        if after != target:
            raise GitHubBoundaryError("target changed during needs-review labelling")
    if not _label_present(client, target):
        raise GitHubBoundaryError("GitHub did not confirm needs-review label")
    return True


def _recover_label(client: Any, repository: str, number: int, original: ReviewTarget | None = None) -> bool:
    """Restore the safety label against the latest complete target snapshot."""
    try:
        current = client.get_review_target(repository, number)
        if _target_fields(current) and current.repository == repository and current.number == number:
            return _ensure_label(client, current)
    except Exception:
        pass
    if original is not None:
        return _ensure_label(client, original)
    raise GitHubBoundaryError("unable to recover needs-review label")


def discover_pull_requests(
    client: Any,
    repository: str,
    *,
    configuration: ReviewConfiguration,
    operator_credential: Mapping[str, Any],
    max_pages: int = 16,
) -> DiscoverySummary:
    """Scan every open PR and reconcile the repository-owned safety label.

    ``GitHubClient.list_pull_requests`` is deliberately used instead of a
    generic pagination helper: it owns the Link-header and fixed-page-bound
    contract.  Any exception from that operation is returned as an explicit
    incomplete scan.
    """
    try:
        trusted = _trust(client, repository, configuration, operator_credential)
    except GitHubBoundaryError as exc:
        item = DiscoveryItem(0, _reason(f"incomplete scan: {exc}"))
        return DiscoverySummary(incomplete=(item,), errors=(item,))
    try:
        response = client.list_pull_requests(repository, max_pages=max_pages)
        pulls = _data(response)
        if not isinstance(pulls, list):
            raise GitHubBoundaryError("pull request scan returned a non-list")
    except Exception as exc:
        item = DiscoveryItem(0, _reason(f"incomplete scan: {exc}"))
        return DiscoverySummary(incomplete=(item,), errors=(item,))

    reviewed: list[DiscoveryItem] = []
    needs: list[DiscoveryItem] = []
    labelled: list[DiscoveryItem] = []
    clean: list[DiscoveryItem] = []
    incomplete: list[DiscoveryItem] = []
    errors: list[DiscoveryItem] = []
    publisher: ReviewPublisher | None = None

    for pull in sorted(pulls, key=lambda value: value.get("number", 0) if isinstance(value, Mapping) else 0):
        number = pull.get("number", 0) if isinstance(pull, Mapping) else 0
        if isinstance(number, bool) or not isinstance(number, int) or number <= 0:
            item = DiscoveryItem(0, "incomplete scan: malformed pull request record")
            incomplete.append(item)
            errors.append(item)
            continue
        try:
            target = client.get_review_target(repository, number)
            if not _target_fields(target) or target.repository != repository or target.number != number:
                raise GitHubBoundaryError("pull request target is incomplete or mismatched")
            records, history_error = _history(client, target, trusted)
            reason: str | None = history_error
            completion = None
            metadata = None
            canonical = None
            if reason is None:
                result = _current_completion(records, target, trusted)
                if isinstance(result, str):
                    reason = result
                else:
                    completion, metadata, canonical = result
            if reason is not None:
                item = DiscoveryItem(number, reason)
                needs.append(item)
                try:
                    if _ensure_label(client, target):
                        labelled.append(item)
                except Exception as exc:
                    try:
                        if _recover_label(client, repository, number, target):
                            labelled.append(item)
                    except Exception:
                        pass
                    error = DiscoveryItem(number, _reason(f"needs-review label mutation failed: {exc}"))
                    errors.append(error)
                    incomplete.append(error)
                if history_error is None:
                    reviewed.append(DiscoveryItem(number, "scanned; needs review"))
                else:
                    incomplete.append(item)
                if reason.startswith("current review history is incomplete or invalid"):
                    incomplete.append(item)
                continue

            assert completion is not None and metadata is not None and canonical is not None
            if publisher is None:
                publisher = ReviewPublisher(
                    client, configuration=configuration, operator_credential=operator_credential
                )
            try:
                assert publisher is not None
                publisher._remove_label(
                    target,
                    completion.envelope.payload["attempt_id"],
                    canonical.node_id,
                    metadata.envelope.node_id,
                    keep_is_review=metadata.is_review,
                )
                if client.get_review_target(repository, number) != target:
                    raise GitHubBoundaryError("target changed after clean reconciliation")
            except Exception as exc:
                try:
                    _recover_label(client, repository, number, target)
                except Exception as label_exc:
                    exc = RuntimeError(f"{exc}; label recovery failed: {label_exc}")
                item = DiscoveryItem(number, _reason(f"clean reconciliation failed: {exc}"))
                needs.append(item)
                errors.append(item)
                incomplete.append(item)
                continue
            clean.append(DiscoveryItem(number, "valid current completion"))
            reviewed.append(DiscoveryItem(number, "scanned; clean"))
        except Exception as exc:
            item = DiscoveryItem(number, _reason(f"PR discovery failed: {exc}"))
            needs.append(item)
            errors.append(item)
            incomplete.append(item)
            recovery_target: ReviewTarget | None = None
            try:
                recovery_target = client.get_review_target(repository, number)
                if isinstance(recovery_target, ReviewTarget) and _target_fields(recovery_target):
                    if _ensure_label(client, recovery_target):
                        labelled.append(item)
            except Exception as label_exc:
                try:
                    if _recover_label(client, repository, number, recovery_target):
                        labelled.append(item)
                except Exception:
                    errors.append(DiscoveryItem(number, _reason(f"label recovery failed: {label_exc}")))

    key = lambda item: (item.number, item.reason)
    return DiscoverySummary(
        reviewed=tuple(sorted(reviewed, key=key)),
        needs_review=tuple(sorted(needs, key=key)),
        labelled=tuple(sorted(labelled, key=key)),
        clean=tuple(sorted(clean, key=key)),
        incomplete=tuple(sorted(incomplete, key=key)),
        errors=tuple(sorted(errors, key=key)),
    )


discover_open_pull_requests = discover_pull_requests


__all__ = ["DiscoveryItem", "DiscoverySummary", "discover_open_pull_requests", "discover_pull_requests"]
