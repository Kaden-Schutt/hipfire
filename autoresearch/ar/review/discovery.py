# Copyright (c) Kaden Schutt
"""Bounded, fail-closed discovery of pull requests needing agentic review."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from .config import ReviewConfiguration, validate_operator_credential_manifest
from .capsule import ReviewCapsule, build_review_capsule, capsule_coverage
from .github import GitHubBoundaryError, decode_protocol_body
from .models import GitHubEnvelope, ReviewTarget, validate_trusted_publishers_policy
from .protocol import validate_protocol
from .publisher import ReviewPublisher


_LABEL = "needs-review"
_SCHEMA = "agentic-review/v1"
_SCHEMAS = {_SCHEMA}
_MAX_REASON = 512
_MAX_AUTHOR_TRUST_CHECKS = 128


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


class _TrustContext:
    def __init__(self, client: Any, repository: str, configuration: ReviewConfiguration) -> None:
        self.client = client
        self.repository = repository
        self.configuration = configuration
        self.authors: set[str] = set()
        self._human_permissions: dict[str, bool] = {}
        self._app_scope: dict[str, bool] = {}
        self._repository_id: int | None = None
        self._trust_checks = 0

    def _check_budget(self) -> None:
        self._trust_checks += 1
        if self._trust_checks > _MAX_AUTHOR_TRUST_CHECKS:
            raise GitHubBoundaryError("workflow author trust checks reached the fixed bound")

    def _repo_id(self) -> int:
        if self._repository_id is None:
            getter = getattr(self.client, "get_repository", None)
            if not callable(getter):
                raise GitHubBoundaryError("repository identity is required for App trust")
            data = _data(getter(self.repository))
            repository_id = data.get("id") if isinstance(data, Mapping) else None
            if isinstance(repository_id, bool) or not isinstance(repository_id, int) or repository_id <= 0:
                raise GitHubBoundaryError("GitHub repository identity is malformed")
            self._repository_id = repository_id
        return self._repository_id

    def _app_authorized(self, login: str) -> bool:
        if login in self._app_scope:
            return self._app_scope[login]
        self._check_budget()
        app = _configured_app(self.configuration, login, self._repo_id())
        if app is None:
            self._app_scope[login] = False
            return False
        repositories = _data(self.client.list_installation_repositories())
        visible = repositories.get("repositories") if isinstance(repositories, Mapping) else None
        authorized = isinstance(visible, list) and any(
            isinstance(item, Mapping) and item.get("id") == self._repo_id() for item in visible
        )
        self._app_scope[login] = authorized
        return authorized

    def authorize(self, login: str, principal_type: str) -> bool:
        if not isinstance(login, str) or not login.strip():
            return False
        if principal_type == "User":
            if login not in self._human_permissions:
                self._check_budget()
                permission = self.client.collaborator_effective_permission(self.repository, login)
                self._human_permissions[login] = bool(
                    getattr(permission, "login", None) == login
                    and getattr(permission, "principal_type", None) == "User"
                    and getattr(permission, "permission", None) in {"write", "admin"}
                )
            authorized = self._human_permissions[login]
        elif principal_type == "Bot":
            authorized = self._app_authorized(login)
        else:
            authorized = False
        if authorized:
            self.authors.add(login)
        return authorized

    def authorize_record(self, login: str, principal_type: str, envelope: GitHubEnvelope) -> bool:
        if not self.authorize(login, principal_type):
            return False
        if principal_type != "Bot":
            return True
        app = _configured_app(self.configuration, login, self._repo_id())
        payload = envelope.payload
        return bool(
            app is not None
            and payload.get("app_id") == app.get("app_id")
            and payload.get("installation_id") == app.get("installation_id")
            and payload.get("repository_id") == app.get("repository_id")
            and payload.get("credential_attestation_digest") == app.get("credential_attestation_digest")
        )

    def authorize_publisher(self, login: str, principal_type: str, envelope: GitHubEnvelope | None = None) -> bool:
        return self.authorize_record(login, principal_type, envelope) if envelope is not None else self.authorize(login, principal_type)


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
) -> _TrustContext:
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
    context = _TrustContext(client, repository, configuration)
    if principal["type"] == "User":
        try:
            permission = client.collaborator_effective_permission(repository, login)
        except Exception as exc:
            raise GitHubBoundaryError(f"effective permission API failure: {exc}") from exc
        if (
            getattr(permission, "login", None) != login
            or getattr(permission, "principal_type", None) != "User"
            or getattr(permission, "permission", None) not in {"write", "admin"}
        ):
            raise ValueError("discovery operator lacks effective write permission")
        context._human_permissions[login] = True
        context.authors.add(login)
        return context
    if principal["type"] != "Bot":
        raise ValueError("discovery operator principal must be a User or configured App")

    try:
        app = _configured_app(configuration, login, context._repo_id())
    except Exception as exc:
        if isinstance(exc, GitHubBoundaryError):
            raise
        raise GitHubBoundaryError(f"repository identity API failure: {exc}") from exc
    if app is None or app.get("credential_attestation_digest") != operator_credential["credential_attestation_digest"]:
        raise ValueError("discovery App attestation does not match configured provenance")
    if not context._app_authorized(login):
        raise ValueError("discovery App installation does not include the repository")
    context.authors.add(login)
    return context


def _candidate_body(body: Any) -> bool:
    return isinstance(body, str) and (
        body.lstrip().startswith("{") or "<!-- agentic-review/v1" in body
    )


def _history(client: Any, target: ReviewTarget, trust: _TrustContext) -> tuple[tuple[_Record, ...], str | None]:
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
        listed_type = user.get("type") if isinstance(user, Mapping) else None
        if not isinstance(login, str) or not isinstance(listed_type, str):
            continue
        body = raw.get("body")
        if not isinstance(body, str):
            continue
        if not _candidate_body(body):
            continue
        try:
            authorized = trust.authorize(login, listed_type)
        except Exception as exc:
            return (), _reason(f"record author trust API failure: {exc}")
        if not authorized:
            continue
        try:
            payload = decode_protocol_body(body)
        except Exception as exc:
            return (), _reason(f"trusted workflow record is malformed: {exc}")
        if payload.get("schema") not in _SCHEMAS:
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
            if not trust.authorize_record(envelope.author, envelope.author_type, envelope):
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
    records: Sequence[_Record], target: ReviewTarget, trust: _TrustContext, capsule: Any = None,
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
        if isinstance(capsule, ReviewCapsule) and capsule.complete:
            expected_coverage = capsule_coverage(capsule)
            for record in current:
                if record.envelope.payload.get("record_type") in {"report", "review-metadata", "completion"}:
                    actual_coverage = {
                        key: record.envelope.payload.get(key) for key in expected_coverage
                    }
                    if actual_coverage != expected_coverage:
                        raise ValueError("history coverage does not match authenticated capsule")
        canonical = cast(
            GitHubEnvelope,
            validate_protocol(
                [record.envelope for record in current], expected_target=target,
                trusted_authors=trust.authors, configuration=trust.configuration, capsule=capsule,
            ),
        )
    except Exception as exc:
        # A changed authenticated configuration source deliberately
        # invalidates ledger-bearing history; leave needs-review set.
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
    if completion.envelope.payload.get("schema") != _SCHEMA or completion.envelope.payload.get("coverage_complete") is not True:
        return "completion lacks complete coverage evidence"
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
        trust = _trust(client, repository, configuration, operator_credential)
        if configuration.source is None or not configuration.source.authenticated:
            raise ValueError("discovery requires an authenticated configuration source")
        client.revalidate_config_source(configuration.source)
    except Exception as exc:
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
            records, history_error = _history(client, target, trust)
            reason: str | None = history_error
            completion = None
            metadata = None
            canonical = None
            capsule = None
            if reason is None:
                # This is the acceptance boundary: history is only accepted
                # against the authenticated default-branch configuration that
                # is live at this instant.
                if configuration.source is None:
                    raise GitHubBoundaryError("authenticated configuration source is missing")
                client.revalidate_config_source(configuration.source)
                ledger_history = any(
                    record.envelope.payload.get("record_type") == "report"
                    and "validation_ledger" in record.envelope.payload
                    for record in records
                )
                if ledger_history:
                    capsule = build_review_capsule(client, target)
                    if not isinstance(capsule, ReviewCapsule) or not capsule.complete or capsule.target != target:
                        raise GitHubBoundaryError("current review capsule is incomplete or target-mismatched")
                    # Close the source race between the first history read and
                    # accepting the capsule-bound ledger history.
                    client.revalidate_config_source(configuration.source)
                result = _current_completion(records, target, trust, capsule)
                if isinstance(result, str):
                    reason = result
                else:
                    completion, metadata, canonical = result
            if reason is not None:
                item = DiscoveryItem(number, reason)
                needs.append(item)
                try:
                    labelled_now = ReviewPublisher(
                        client, configuration=configuration, operator_credential=operator_credential,
                        trusted_authors=trust.authors,
                        author_authorizer=trust.authorize_publisher,
                    ).reconcile_discovery(target, capsule=capsule)
                    if labelled_now:
                        labelled.append(item)
                except Exception as exc:
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
            try:
                ReviewPublisher(
                    client, configuration=configuration, operator_credential=operator_credential,
                    trusted_authors=trust.authors,
                    author_authorizer=trust.authorize_publisher,
                ).reconcile_discovery(
                    target,
                    attempt_id=completion.envelope.payload["attempt_id"],
                    intent_node=canonical.node_id,
                    keep_node=metadata.envelope.node_id,
                    keep_is_review=metadata.is_review,
                    capsule=capsule,
                )
                if client.get_review_target(repository, number) != target:
                    raise GitHubBoundaryError("target changed after clean reconciliation")
            except Exception as exc:
                try:
                    ReviewPublisher(
                        client, configuration=configuration, operator_credential=operator_credential,
                        trusted_authors=trust.authors,
                        author_authorizer=trust.authorize_publisher,
                    ).reconcile_discovery(target, capsule=capsule)
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
                    labelled_now = ReviewPublisher(
                        client, configuration=configuration, operator_credential=operator_credential,
                        trusted_authors=trust.authors,
                        author_authorizer=trust.authorize_publisher,
                    ).reconcile_discovery(recovery_target)
                    if labelled_now:
                        labelled.append(item)
            except Exception as label_exc:
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
