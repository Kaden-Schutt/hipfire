# Copyright (c) Kaden Schutt
"""Authenticated publication of SHA-bound agentic review records."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
import hashlib
import html
import json
from typing import Any, Callable

from .canonical import canonical_digest, canonical_json, metadata_digest
from .capsule import ReviewCapsule, build_review_capsule, capsule_coverage
from .config import (
    ReviewConfiguration,
    validate_operator_credential_manifest,
    validate_publisher_operator_credential,
)
from .github import GitHubBoundaryError, decode_protocol_body, encode_protocol_body
from .models import (
    GitHubEnvelope,
    ReviewProposal,
    ReviewTarget,
    derive_protected_review_scope,
    protected_exemption_evidence,
    validate_trusted_publishers_policy,
)
from .protocol import (
    validate_intent,
    validate_protocol,
    validate_report,
    validate_revocation,
    validate_validation_ledger,
)
from .validation import render_validation_section, validate_rendered_validation_section


class PublisherError(RuntimeError):
    """The publisher rejected an input or an authenticated protocol state."""


class LabelError(PublisherError):
    """A required label mutation failed or could not be verified."""


@dataclass(frozen=True)
class PublishResult:
    status: str
    attempt_id: str
    report_envelope: GitHubEnvelope | None = None
    review_envelope: GitHubEnvelope | None = None
    completion_envelope: GitHubEnvelope | None = None
    reason: str | None = None


@dataclass(frozen=True)
class _HistoryRecord:
    envelope: GitHubEnvelope
    is_review: bool
    server_id: int
    state: str | None = None
    commit_id: str | None = None


@dataclass(frozen=True)
class _History:
    current: tuple[_HistoryRecord, ...]
    valid: tuple[_HistoryRecord, ...]


class _StaleTarget(RuntimeError):
    pass


class _CanonicalChanged(RuntimeError):
    pass


class _UnsafeLabelSnapshot(RuntimeError):
    pass


class _PreflightRejected(PublisherError):
    """A proposal failed before any publication mutation was permitted."""


_SCHEMA = "agentic-review/v1"
_LABEL = "needs-review"
_MAX_RECONCILIATION_ROUNDS = 4
_APP_FIELDS = ("app_id", "installation_id", "repository_id", "credential_attestation_digest")
_MAX_RENDERED_REPORT_BYTES = 256 * 1024


def _target_from_payload(payload: Mapping[str, Any]) -> ReviewTarget:
    target = payload.get("target")
    if isinstance(target, ReviewTarget):
        result = target
    elif isinstance(target, Mapping):
        fields = {"repository", "number", "head_repository", "head_sha", "base_ref", "base_sha", "merge_base_sha"}
        if set(target) != fields:
            raise PublisherError("protocol record does not contain a complete ReviewTarget")
        try:
            result = ReviewTarget(**target)
        except (TypeError, ValueError) as exc:
            raise PublisherError("protocol record contains an invalid ReviewTarget") from exc
    else:
        raise PublisherError("protocol record does not contain a ReviewTarget")
    if payload.get("target_key") != result.target_key():
        raise PublisherError("protocol record target key is not bound to its target")
    return result


def _safe_html_text(value: str) -> str:
    normalized = value.replace("\r\n", "\n").replace("\r", "\n")
    return html.escape(normalized, quote=True)


def render_report(proposal: ReviewProposal) -> str:
    """Render only structured, escaped proposal fields into visible Markdown."""
    lines = ["## Agentic review", "", f"Verdict: <code>{_safe_html_text(proposal.verdict)}</code>"]
    if proposal.findings:
        lines.extend(("", "### Findings"))
        for finding in proposal.findings:
            path = _safe_html_text(finding.path)
            message = _safe_html_text(finding.message)
            severity = _safe_html_text(finding.severity)
            lines.append(f"- <code>{path}:{finding.range[0]}-{finding.range[1]}</code> ({severity}):")
            lines.append(f"  <pre><code>{message}</code></pre>")
    else:
        lines.extend(("", "No findings."))
    if proposal.hardware_validation_triage is not None:
        triage = proposal.hardware_validation_triage
        lines.extend((
            "",
            "### Hardware validation triage",
            f"- Impacted model families: {_safe_html_text(', '.join(triage.impacted_model_families))}",
            f"- Impacted hardware: {_safe_html_text(', '.join(triage.impacted_hardware))}",
            f"- Coverage decision: <code>{_safe_html_text(triage.coverage_decision)}</code>",
            f"- Rationale: {_safe_html_text(triage.rationale)}",
        ))
    if proposal.validation_ledger or proposal.exemption_ids:
        lines.extend(("", render_validation_section(
            proposal.validation_ledger, exempt=bool(proposal.exemption_ids), scope=proposal.scope,
        )))
    body = "\n".join(lines)
    if not body or body != body.strip() or len(body.encode("utf-8")) > _MAX_RENDERED_REPORT_BYTES:
        raise PublisherError("rendered report is empty, padded, or exceeds 256 KiB")
    return body

class ReviewPublisher:
    """Publish a validated proposal through the fixed GitHub boundary."""

    def __init__(
        self,
        client: Any,
        *,
        configuration: ReviewConfiguration,
        operator_credential: Mapping[str, Any],
        trusted_authors: Iterable[str] | None = None,
        author_authorizer: Callable[..., bool] | None = None,
    ) -> None:
        if not isinstance(configuration, ReviewConfiguration) or not configuration.is_protected:
            raise PublisherError("publisher requires an authenticated immutable configuration")
        if configuration.source is None or not configuration.source.authenticated:
            raise PublisherError("publisher requires an authenticated configuration source")
        try:
            validate_operator_credential_manifest(operator_credential)
        except (TypeError, ValueError) as exc:
            raise PublisherError("operator credential is not attested") from exc
        self._client = client
        self._configuration = configuration
        self._operator = deepcopy(dict(operator_credential))
        self._additional_trusted_authors = set(trusted_authors or ())
        self._author_authorizer = author_authorizer
        self._discovery_authority_enabled = False
        self._discovery_requires_dismissal = False
        self._history_capsule: Any | None = None

    @property
    def _trusted_authors(self) -> frozenset[str]:
        authors = {self._operator["principal"]["login"]}
        apps = self._configuration.trusted_publishers.get("apps", ())
        if isinstance(apps, Sequence) and not isinstance(apps, (str, bytes)):
            authors.update(
                app["login"] for app in apps
                if isinstance(app, Mapping) and isinstance(app.get("login"), str)
            )
        authors.update(self._additional_trusted_authors)
        return frozenset(authors)

    def _author_trusted(self, login: Any, principal_type: Any, envelope: GitHubEnvelope | None = None) -> bool:
        if self._author_authorizer is not None and isinstance(login, str) and isinstance(principal_type, str):
            try:
                if envelope is not None:
                    authorized = self._author_authorizer(login, principal_type, envelope)
                else:
                    authorized = self._author_authorizer(login, principal_type)
                if authorized:
                    self._additional_trusted_authors.add(login)
                    return True
                return False
            except Exception as exc:
                raise PublisherError("workflow author trust could not be revalidated") from exc
        return isinstance(login, str) and login in self._trusted_authors

    def _app_provenance_payload(self) -> dict[str, Any]:
        principal = self._operator["principal"]
        if principal["type"] != "Bot":
            return {}
        apps = [
            app for app in self._configuration.trusted_publishers.get("apps", ())
            if isinstance(app, Mapping) and app.get("login") == principal["login"]
        ]
        if len(apps) != 1:
            raise PublisherError("publisher App provenance is not uniquely configured")
        app = apps[0]
        return {field: app[field] for field in _APP_FIELDS}

    def _pull_target(self, target: ReviewTarget) -> ReviewTarget:
        getter = getattr(self._client, "get_review_target", None)
        if not callable(getter):
            raise PublisherError("GitHub client lacks the typed complete-target operation")
        current = getter(target.repository, target.number)
        if not isinstance(current, ReviewTarget):
            raise PublisherError("GitHub client returned an untyped ReviewTarget")
        return current

    def _assert_target(self, target: ReviewTarget) -> None:
        if self._pull_target(target) != target:
            raise _StaleTarget("review target changed")

    def _reapply_label(self, target: ReviewTarget, attempt_id: str | None = None) -> None:
        try:
            if attempt_id is not None:
                try:
                    self._canonical(target, attempt_id)
                except (_CanonicalChanged, PublisherError):
                    # Recovery must still restore the safety label when the
                    # attempt itself became stale; the election was performed
                    # and publication is already being aborted.
                    pass
            before = self._pull_target(target)
            self._check_discovery_authority(target, require_cleanup=False)
            self._client.add_labels(target.repository, target.number, [_LABEL])
            after = self._pull_target(target)
            if after != before:
                raise _StaleTarget("target changed while reapplying needs-review")
            if not self._label_present(target):
                raise LabelError("GitHub did not confirm needs-review after reapply")
        except _StaleTarget:
            raise
        except Exception as exc:
            raise LabelError("failed to reapply needs-review") from exc

    def _history(self, target: ReviewTarget) -> _History:
        raw_comments = self._client.list_issue_comments(target.repository, target.number).data
        raw_reviews = self._client.list_pull_reviews(target.repository, target.number).data
        records: list[_HistoryRecord] = []
        for raw, is_review in [
            *[(item, False) for item in (raw_comments or [])],
            *[(item, True) for item in (raw_reviews or [])],
        ]:
            if not isinstance(raw, Mapping):
                raise PublisherError("GitHub history contains a malformed record")
            body = raw.get("body")
            if not isinstance(body, str):
                continue
            listed_user = raw.get("user")
            listed_login = listed_user.get("login") if isinstance(listed_user, Mapping) else None
            listed_type = listed_user.get("type") if isinstance(listed_user, Mapping) else None
            if not (body.lstrip().startswith("{") or "agentic-review/v1" in body):
                continue
            if not self._author_trusted(listed_login, listed_type):
                continue
            try:
                payload = decode_protocol_body(body)
            except Exception:
                if body.lstrip().startswith("{") or "agentic-review/v1" in body:
                    raise PublisherError("a protocol record was deleted or edited")
                continue
            if payload.get("schema") not in {"agentic-review/v1", _SCHEMA}:
                continue
            try:
                if is_review:
                    exact = self._client.get_pull_review_record(target.repository, target.number, raw["id"])
                    envelope = exact.envelope
                    state = exact.state
                    commit_id = exact.commit_id
                    server_id = exact.server_id
                else:
                    envelope = self._client.comment_envelope(target.repository, raw["id"])
                    state = commit_id = None
                    server_id = raw["id"]
                if not self._author_trusted(envelope.author, envelope.author_type, envelope):
                    continue
                record = _HistoryRecord(envelope, is_review, server_id, state, commit_id)
                _target_from_payload(envelope.payload) if envelope.payload.get("record_type") != "revocation" else None
            except (KeyError, TypeError, ValueError, PublisherError) as exc:
                raise PublisherError("a protocol record was deleted, edited, or malformed") from exc
            records.append(record)

        targets: dict[str, ReviewTarget] = {}
        for record in records:
            if record.envelope.payload.get("record_type") == "revocation":
                continue
            parsed = _target_from_payload(record.envelope.payload)
            targets[parsed.target_key()] = parsed
        groups: dict[str, list[_HistoryRecord]] = {}
        for record in records:
            payload = record.envelope.payload
            key = payload.get("target_key")
            if not isinstance(key, str):
                raise PublisherError("protocol record target key is missing")
            if payload.get("record_type") == "revocation" and key not in targets:
                raise PublisherError("revocation has no complete historical target")
            groups.setdefault(key, []).append(record)

        def event_key(record: _HistoryRecord) -> tuple[datetime, str]:
            value = record.envelope.created_at
            normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
            return datetime.fromisoformat(normalized), record.envelope.node_id

        valid: list[_HistoryRecord] = []
        current: list[_HistoryRecord] = []
        for key, group in groups.items():
            expected = targets.get(key)
            if expected is None:
                continue
            intents = {
                record.envelope.payload["attempt_id"]: record
                for record in group
                if record.envelope.payload.get("record_type") == "intent"
            }
            revoked: set[str] = set()
            for record in sorted(group, key=event_key):
                payload = record.envelope.payload
                if payload.get("record_type") != "revocation":
                    continue
                intent = intents.get(payload.get("attempt_id"))
                if intent is None:
                    continue
                try:
                    validate_revocation(record.envelope, intent.envelope, trusted_authors=self._trusted_authors)
                except ValueError:
                    continue
                revoked.add(payload["attempt_id"])
            active = [record for attempt, record in intents.items() if attempt not in revoked]
            canonical_attempt = min(active, key=event_key).envelope.payload["attempt_id"] if active else None
            attempt_groups: dict[str, list[_HistoryRecord]] = {}
            for record in group:
                attempt = record.envelope.payload.get("attempt_id")
                if isinstance(attempt, str):
                    attempt_groups.setdefault(attempt, []).append(record)
            for attempt, attempt_group in attempt_groups.items():
                if attempt in revoked:
                    historical = [record for record in attempt_group if record.envelope.payload.get("record_type") != "revocation"]
                    try:
                        validate_protocol(
                            [record.envelope for record in historical],
                            expected_target=expected,
                            trusted_authors=self._trusted_authors,
                            configuration=self._configuration,
                            capsule=self._history_capsule,
                        )
                    except ValueError:
                        continue
                    valid.extend(historical)
                    continue
                try:
                    elected = validate_protocol(
                        [record.envelope for record in attempt_group],
                        expected_target=expected,
                        trusted_authors=self._trusted_authors,
                        configuration=self._configuration,
                        capsule=self._history_capsule,
                    )
                except ValueError as exc:
                    if expected == target and attempt == canonical_attempt and "no valid non-revoked intent" not in str(exc):
                        raise PublisherError(f"invalid current review history: {exc}") from exc
                    if "no valid non-revoked intent" in str(exc):
                        valid.extend(attempt_group)
                    continue
                valid.extend(attempt_group)
                if expected == target and attempt == canonical_attempt:
                    current.extend(attempt_group)
        return _History(tuple(current), tuple(valid))

    def _canonical(
        self, target: ReviewTarget, attempt_id: str, intent_node: str | None = None,
    ) -> tuple[GitHubEnvelope, _History]:
        history = self._history(target)
        try:
            elected = validate_protocol(
                [record.envelope for record in history.current],
                expected_target=target,
                trusted_authors=self._trusted_authors,
                configuration=self._configuration,
                capsule=self._history_capsule,
            )
        except ValueError as exc:
            raise _CanonicalChanged("canonical intent is no longer active") from exc
        if not isinstance(elected, GitHubEnvelope):
            raise _CanonicalChanged("canonical intent is not an authenticated envelope")
        if elected.payload.get("attempt_id") != attempt_id or (intent_node and elected.node_id != intent_node):
            raise _CanonicalChanged("canonical review attempt changed")
        return elected, history

    def _mutate(
        self,
        target: ReviewTarget,
        operation: Callable[[], Any],
        *,
        attempt_id: str | None = None,
        intent_node: str | None = None,
        before_mutation: Callable[[GitHubEnvelope, _History], None] | None = None,
        return_snapshot: bool = False,
    ) -> Any:
        self._check_discovery_authority(target)
        canonical: GitHubEnvelope | None = None
        history: _History | None = None
        if attempt_id is not None:
            canonical, history = self._canonical(target, attempt_id, intent_node)
        self._assert_target(target)
        if canonical is not None and history is not None and before_mutation is not None:
            before_mutation(canonical, history)
        value = operation()
        self._assert_target(target)
        if attempt_id is not None:
            self._canonical(target, attempt_id, intent_node)
        if return_snapshot:
            if canonical is None or history is None:
                raise PublisherError("mutation snapshot was not authenticated")
            return value, canonical, history
        return value

    def _check_discovery_authority(self, target: ReviewTarget, *, require_cleanup: bool | None = None) -> None:
        source = self._configuration.source
        if source is None or not source.authenticated or source.repository != target.repository:
            raise PublisherError("authenticated configuration source does not match target repository")
        try:
            self._client.revalidate_config_source(source)
        except Exception as exc:
            raise PublisherError("configuration provenance could not be revalidated") from exc
        if not self._discovery_authority_enabled:
            return
        try:
            validate_operator_credential_manifest(self._operator)
            if self._operator["repository"] != target.repository:
                raise PublisherError("operator manifest repository does not match target repository")
            principal = self._operator["principal"]
            if principal["type"] not in {"User", "Bot"}:
                raise PublisherError("discovery operator principal is unsupported")
            cleanup = self._discovery_requires_dismissal if require_cleanup is None else require_cleanup
            if "discover" not in self._operator["allowed_operations"]:
                raise PublisherError("discovery operator lacks discover operation")
            if cleanup and "dismiss-workflow-review" not in self._operator["allowed_operations"]:
                raise PublisherError("discovery operator lacks dismissal operation")
            if any(
                self._operator["write_permissions"].get(permission) not in {"write", "admin"}
                for permission in (("issues", "pull_requests") if cleanup else ("issues",))
            ):
                raise PublisherError("discovery operator lacks issues and pull_requests write authority")
            if principal["type"] == "User":
                permission = self._client.collaborator_effective_permission(target.repository, principal["login"])
                if (
                    permission.login != principal["login"]
                    or permission.principal_type != "User"
                    or permission.permission not in {"write", "admin"}
                ):
                    raise PublisherError("discovery operator lacks current effective write authority")
                return
            repository = self._client.get_repository(target.repository).data
            repository_id = repository.get("id") if isinstance(repository, Mapping) else None
            validate_trusted_publishers_policy(self._configuration.trusted_publishers)
            apps = [
                app for app in self._configuration.trusted_publishers["apps"]
                if app["login"] == principal["login"] and app["repository_id"] == repository_id
            ]
            if len(apps) != 1 or apps[0]["credential_attestation_digest"] != self._operator["credential_attestation_digest"]:
                raise PublisherError("discovery App provenance does not match the operator")
            installations = self._client.list_installation_repositories().data
            repositories = installations.get("repositories") if isinstance(installations, Mapping) else None
            if not isinstance(repositories, list) or not any(
                isinstance(item, Mapping) and item.get("id") == repository_id for item in repositories
            ):
                raise PublisherError("discovery App installation does not include the repository")
        except PublisherError:
            raise
        except Exception as exc:
            raise PublisherError("discovery mutation authority could not be revalidated") from exc

    def _raw_workflow_review_ids(self, target: ReviewTarget) -> list[tuple[int, str]]:
        raw_reviews = self._client.list_pull_reviews(target.repository, target.number).data
        if not isinstance(raw_reviews, list):
            raise PublisherError("GitHub review history is malformed")
        result: list[tuple[int, str]] = []
        for raw in raw_reviews:
            if not isinstance(raw, Mapping):
                raise PublisherError("GitHub review history contains a malformed record")
            user = raw.get("user")
            login = user.get("login") if isinstance(user, Mapping) else None
            user_type = user.get("type") if isinstance(user, Mapping) else None
            body = raw.get("body")
            if not isinstance(body, str) or not (body.lstrip().startswith("{") or "agentic-review/v1" in body):
                continue
            if not self._author_trusted(login, user_type):
                continue
            try:
                payload = decode_protocol_body(body)
            except Exception as exc:
                raise PublisherError("newly observed workflow review is malformed") from exc
            if payload.get("record_type") == "review-metadata" and payload.get("schema") != _SCHEMA:
                raise PublisherError("newly observed workflow review uses an unsupported schema")
            if payload.get("schema") != _SCHEMA or payload.get("record_type") != "review-metadata":
                continue
            try:
                exact = self._client.get_pull_review_record(target.repository, target.number, raw["id"])
                record_target = _target_from_payload(payload)
                if (
                    self._author_trusted(exact.envelope.author, exact.envelope.author_type, exact.envelope)
                    and exact.state == "CHANGES_REQUESTED"
                    and exact.commit_id == record_target.head_sha
                ):
                    result.append((exact.server_id, exact.envelope.node_id))
            except Exception:
                raise PublisherError("newly observed workflow review could not be authenticated")
        return sorted(set(result))

    def _remove_discovery_label(
        self, target: ReviewTarget, attempt_id: str, intent_node: str, keep_node: str, *, keep_is_review: bool,
    ) -> None:
        for _ in range(_MAX_RECONCILIATION_ROUNDS):
            canonical, history = self._canonical(target, attempt_id, intent_node)
            self._validate_keep_review(history, target, keep_node) if keep_is_review else None
            stale = [node_id for review_id, node_id in self._raw_workflow_review_ids(target) if node_id != keep_node]
            if stale:
                self._discovery_requires_dismissal = True
                self._check_discovery_authority(target, require_cleanup=True)
                for review_id in [review_id for review_id, node_id in self._raw_workflow_review_ids(target) if node_id != keep_node]:
                    self._mutate(
                        target,
                        lambda review_id=review_id: self._client.dismiss_workflow_review(
                            target.repository, target.number, review_id,
                            message="Superseded by a current agentic review",
                        ),
                        attempt_id=attempt_id,
                        intent_node=canonical.node_id,
                    )
                continue
            self._discovery_requires_dismissal = False
            if not self._label_present(target):
                self._assert_target(target)
                return
            self._mutate(
                target,
                lambda: self._client.remove_label(target.repository, target.number, _LABEL),
                attempt_id=attempt_id,
                intent_node=canonical.node_id,
            )
            self._assert_target(target)
            stable, stable_history = self._canonical(target, attempt_id, intent_node)
            if keep_is_review:
                self._validate_keep_review(stable_history, target, keep_node)
            if not self._raw_workflow_review_ids(target):
                return
        raise PublisherError("discovery label reconciliation did not stabilize")

    def _active_canonical_review_node(self, target: ReviewTarget) -> str | None:
        try:
            history = self._history(target)
            canonical = validate_protocol(
                [record.envelope for record in history.current],
                expected_target=target,
                trusted_authors=self._trusted_authors,
                configuration=self._configuration,
                capsule=self._history_capsule,
            )
            attempt_id = canonical.payload.get("attempt_id")
            for record in history.current:
                payload = record.envelope.payload
                if (
                    record.is_review
                    and payload.get("record_type") == "review-metadata"
                    and payload.get("attempt_id") == attempt_id
                    and record.state == "CHANGES_REQUESTED"
                    and record.commit_id == target.head_sha
                ):
                    return record.envelope.node_id
        except Exception:
            return None
        return None

    def reconcile_discovery(
        self,
        target: ReviewTarget,
        *,
        attempt_id: str | None = None,
        intent_node: str | None = None,
        keep_node: str | None = None,
        keep_is_review: bool = False,
        capsule: Any | None = None,
    ) -> bool:
        """Public, authority-checked discovery reconciliation operation."""
        source = self._configuration.source
        if not isinstance(target, ReviewTarget) or source is None or target.repository != source.repository:
            raise PublisherError("discovery reconciliation target is not bound to the configured repository")
        self._discovery_authority_enabled = True
        self._history_capsule = capsule
        try:
            self._discovery_requires_dismissal = False
            self._check_discovery_authority(target, require_cleanup=False)
            had_label = self._label_present(target)
            if keep_node is None:
                keep_node = self._active_canonical_review_node(target)
            for _ in range(_MAX_RECONCILIATION_ROUNDS):
                stale = [
                    review_id for review_id, node_id in self._raw_workflow_review_ids(target)
                    if node_id != keep_node
                ]
                if not stale:
                    break
                self._discovery_requires_dismissal = True
                self._check_discovery_authority(target, require_cleanup=True)
                for review_id in stale:
                    self._mutate(
                        target,
                        lambda review_id=review_id: self._client.dismiss_workflow_review(
                            target.repository, target.number, review_id,
                            message="Superseded by a current agentic review",
                        ),
                    )
            else:
                raise PublisherError("discovery workflow review reconciliation did not stabilize")
            if attempt_id is not None and intent_node is not None and keep_node is not None:
                self._remove_discovery_label(
                    target, attempt_id, intent_node, keep_node, keep_is_review=keep_is_review,
                )
            else:
                self._discovery_requires_dismissal = False
                self._reapply_label(target)
                self._assert_target(target)
                return not had_label
            return False
        except Exception:
            self._discovery_requires_dismissal = False
            try:
                self._reapply_label(target)
            except Exception:
                pass
            raise
        finally:
            self._discovery_requires_dismissal = False
            self._discovery_authority_enabled = False
            self._history_capsule = None

    def _intent_payload(self, target: ReviewTarget, attempt_id: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": _SCHEMA, "record_type": "intent", "record_id": f"intent-{attempt_id}",
            "target": target, "target_key": target.target_key(), "attempt_id": attempt_id,
            "canonical_digest": "",
            **self._app_provenance_payload(),
        }
        payload["canonical_digest"] = canonical_digest({key: value for key, value in payload.items() if key != "canonical_digest"})
        return payload

    def _report_payload(
        self, proposal: ReviewProposal, target: ReviewTarget, intent: GitHubEnvelope, capsule: Any,
    ) -> dict[str, Any]:
        body = render_report(proposal)
        payload: dict[str, Any] = {
            "schema": _SCHEMA, "record_type": "report", "record_id": f"report-{intent.payload['attempt_id']}",
            "target": target, "target_key": target.target_key(), "attempt_id": intent.payload["attempt_id"],
            "intent_record_id": intent.payload["record_id"], "canonical_intent_node_id": intent.node_id,
            "canonical_intent_digest": intent.payload["canonical_digest"], "head_sha": target.head_sha,
            "report_body": body, "report_body_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
            **capsule_coverage(capsule),
            **self._app_provenance_payload(),
        }
        payload.update({
            "capsule_digest": capsule.digest,
            "capsule_paths": [entry.path for entry in capsule.manifest],
            "capsule_target_key": capsule.target_key,
        })
        if proposal.scope is not None:
            payload["scope"] = proposal.scope.to_mapping()
        if proposal.configuration_source_digest is not None:
            payload.update({
                "validation_ledger": [row.to_mapping() for row in proposal.validation_ledger],
                "configuration_source_digest": proposal.configuration_source_digest,
            })
        if proposal.exemption_ids:
            payload.update({
                "exemption_ids": list(proposal.exemption_ids),
                "exemption_paths": list(proposal.exemption_paths),
            })
        return payload

    def _reconstruct_and_validate_capsule(
        self, proposal: ReviewProposal, target: ReviewTarget,
    ) -> Any:
        try:
            capsule = build_review_capsule(self._client, target)
            if not isinstance(capsule, ReviewCapsule) or not capsule.complete:
                raise ValueError("review capsule is incomplete")
            if capsule.digest != proposal.capsule_digest:
                raise ValueError("review capsule digest does not match proposal")
            if proposal.coverage_mapping() != capsule_coverage(capsule):
                raise ValueError("review proposal coverage does not match authenticated capsule")
            expected_scope = derive_protected_review_scope(capsule, self._configuration.capabilities)
            if proposal.scope != expected_scope:
                raise ValueError("review proposal scope does not match protected capsule scope")
            return capsule
        except (KeyError, TypeError, ValueError) as exc:
            raise _PreflightRejected("proposal capsule or protected scope could not be authenticated") from exc
        except Exception as exc:
            raise _PreflightRejected("proposal capsule could not be reconstructed") from exc

    def _validate_proposal_configuration(
        self, proposal: ReviewProposal, target: ReviewTarget, capsule: Any,
    ) -> None:
        source = self._configuration.source
        if proposal.validation_ledger or proposal.configuration_source_digest is not None:
            if source is None or not source.authenticated:
                raise PublisherError("validation proposal requires an authenticated configuration source")
            if proposal.configuration_source_digest != source.config_digest:
                raise PublisherError("proposal configuration source digest does not match publisher configuration")
            ledger_payload = {
                "validation_ledger": [row.to_mapping() for row in proposal.validation_ledger],
                "configuration_source_digest": proposal.configuration_source_digest,
                "target": target,
                "scope": proposal.scope.to_mapping() if proposal.scope is not None else None,
                "capsule_digest": capsule.digest,
                "capsule_paths": [entry.path for entry in capsule.manifest],
                "capsule_target_key": capsule.target_key,
                **capsule_coverage(capsule),
            }
            if proposal.exemption_ids:
                ledger_payload.update({
                    "exemption_ids": list(proposal.exemption_ids),
                    "exemption_paths": list(proposal.exemption_paths),
                })
            try:
                validate_validation_ledger(
                    ledger_payload, configuration=self._configuration, capsule=capsule,
                )
                if proposal.exemption_ids:
                    expected = protected_exemption_evidence(
                        self._configuration.capabilities["exemptions"], proposal.exemption_paths,
                    )
                    if expected != (proposal.exemption_ids, proposal.exemption_paths):
                        raise ValueError("exemption evidence does not match protected policy")
                    manifest_paths = tuple(item.path for item in capsule.manifest)
                    actual = protected_exemption_evidence(
                        self._configuration.capabilities["exemptions"], manifest_paths,
                    )
                    if (
                        not capsule.complete
                        or capsule.digest != proposal.capsule_digest
                        or actual != (proposal.exemption_ids, proposal.exemption_paths)
                    ):
                        raise ValueError("protected exemption capsule evidence does not match proposal")
            except (KeyError, TypeError, ValueError) as exc:
                raise _PreflightRejected("proposal validation ledger is not protected by publisher configuration") from exc

    def _require_matching_report_binding(self, report: _HistoryRecord, proposal: ReviewProposal) -> None:
        payload = report.envelope.payload
        has_binding = "validation_ledger" in payload or "configuration_source_digest" in payload
        expected_binding = proposal.configuration_source_digest is not None
        if has_binding != expected_binding:
            raise PublisherError("existing report validation binding does not match proposal")
        if expected_binding and (
            payload.get("configuration_source_digest") != proposal.configuration_source_digest
            or canonical_json(payload.get("validation_ledger"))
                != canonical_json([row.to_mapping() for row in proposal.validation_ledger])
            or tuple(payload.get("exemption_ids", ())) != proposal.exemption_ids
            or tuple(payload.get("exemption_paths", ())) != proposal.exemption_paths
            or canonical_json(payload.get("scope"))
                != canonical_json(proposal.scope.to_mapping() if proposal.scope is not None else None)
            or payload.get("capsule_digest") != proposal.capsule_digest
            or payload.get("capsule_target_key") != proposal.target.target_key()
            or (
                self._history_capsule is not None
                and tuple(payload.get("capsule_paths", ()))
                != tuple(entry.path for entry in self._history_capsule.manifest)
            )
        ):
            raise PublisherError("existing report validation ledger does not match proposal")

    def _preflight_report_comment(
        self, proposal: ReviewProposal, target: ReviewTarget, attempt_id: str, capsule: ReviewCapsule,
    ) -> None:
        """Bound the exact report comment before the intent mutation.

        GitHub node IDs are bounded at the authenticated boundary to 128 UTF-8
        bytes.  A control-escape-filled placeholder therefore gives a
        conservative upper bound for the only report field not known before
        intent creation.
        """
        # JSON control escapes are larger than UTF-8 code points, so they are
        # the conservative placeholder for a 128-byte authenticated node ID.
        placeholder_node = "\x00" * 128
        placeholder_intent = GitHubEnvelope(
            self._intent_payload(target, attempt_id), placeholder_node,
            self._operator["principal"]["login"], "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z",
            self._operator["principal"]["type"],
        )
        try:
            visible_body = render_report(proposal)
            if proposal.validation_ledger or proposal.exemption_ids:
                validate_rendered_validation_section(
                    visible_body, proposal.validation_ledger, exempt=bool(proposal.exemption_ids), scope=proposal.scope,
                )
            report_payload = self._report_payload(proposal, target, placeholder_intent, capsule)
            report_envelope = GitHubEnvelope(
                report_payload, placeholder_node, self._operator["principal"]["login"],
                "2026-01-01T00:00:01Z", "2026-01-01T00:00:01Z",
                self._operator["principal"]["type"],
            )
            validate_report(
                report_envelope, placeholder_intent, canonical_intent=placeholder_intent,
                trusted_authors={self._operator["principal"]["login"]},
                configuration=self._configuration, capsule=capsule,
            )
            encode_protocol_body(
                report_payload,
                visible_body=visible_body,
            )
        except (GitHubBoundaryError, TypeError, ValueError) as exc:
            raise _PreflightRejected("report comment exceeds the pre-publication size bound") from exc

    def _metadata_payload(self, target: ReviewTarget, intent: GitHubEnvelope, report: GitHubEnvelope) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": _SCHEMA, "record_type": "review-metadata", "record_id": f"metadata-{intent.payload['attempt_id']}",
            "target": target, "target_key": target.target_key(), "attempt_id": intent.payload["attempt_id"],
            "intent_record_id": intent.payload["record_id"], "head_sha": target.head_sha,
            "report_record_id": report.payload["record_id"], "report_node_id": report.node_id,
            "report_digest": canonical_digest(report.payload), "report_body_sha256": report.payload["report_body_sha256"],
            "canonical_intent_digest": intent.payload["canonical_digest"], "canonical_intent_node_id": intent.node_id,
            "metadata_digest": "",
            **{field: report.payload[field] for field in (
                "retrieved_file_count", "expected_file_count", "retrieved_blob_count", "expected_blob_count",
                "retrieved_content_count", "expected_content_count", "coverage_complete",
            )},
            **self._app_provenance_payload(),
        }
        payload["metadata_digest"] = metadata_digest(payload)
        return payload

    def _completion_payload(self, target: ReviewTarget, intent: GitHubEnvelope, report: GitHubEnvelope, metadata: GitHubEnvelope) -> dict[str, Any]:
        return {
            "schema": _SCHEMA, "record_type": "completion", "record_id": f"completion-{intent.payload['attempt_id']}",
            "target": target, "target_key": target.target_key(), "attempt_id": intent.payload["attempt_id"],
            "intent_record_id": intent.payload["record_id"], "head_sha": target.head_sha,
            "canonical_intent_digest": intent.payload["canonical_digest"], "canonical_intent_node_id": intent.node_id,
            "report_record_id": report.payload["record_id"], "report_node_id": report.node_id,
            "report_digest": canonical_digest(report.payload), "metadata_record_id": metadata.payload["record_id"],
            "metadata_digest": metadata.payload["metadata_digest"],
            **{field: metadata.payload[field] for field in (
                "retrieved_file_count", "expected_file_count", "retrieved_blob_count", "expected_blob_count",
                "retrieved_content_count", "expected_content_count", "coverage_complete",
            )},
            **self._app_provenance_payload(),
        }

    def _new_comment(
        self, target: ReviewTarget, payload: Mapping[str, Any], *, attempt_id: str | None = None,
        intent_node: str | None = None, visible_body: str | None = None,
    ) -> GitHubEnvelope:
        encoded_body = encode_protocol_body(payload, visible_body=visible_body)
        response = self._mutate(
            target,
            lambda: self._client.create_issue_comment(
                target.repository, target.number, encoded_body
            ),
            attempt_id=attempt_id,
            intent_node=intent_node,
        )
        record = response.data if hasattr(response, "data") else response
        if not isinstance(record, Mapping) or not isinstance(record.get("id"), int):
            raise PublisherError("GitHub comment mutation did not return a server record")
        return self._client.comment_envelope(target.repository, record["id"])

    def _new_review(self, target: ReviewTarget, payload: Mapping[str, Any], attempt_id: str, intent_node: str) -> _HistoryRecord:
        response = self._mutate(
            target,
            lambda: self._client.create_pull_request_review(
                target.repository, target.number, body=canonical_json(payload).decode("utf-8"),
                event="REQUEST_CHANGES", commit_id=target.head_sha,
            ),
            attempt_id=attempt_id,
            intent_node=intent_node,
        )
        record = response.data if hasattr(response, "data") else response
        if not isinstance(record, Mapping) or not isinstance(record.get("id"), int):
            raise PublisherError("GitHub review mutation did not return a server record")
        exact = self._client.get_pull_review_record(target.repository, target.number, record["id"])
        envelope = exact.envelope
        if exact.state != "CHANGES_REQUESTED" or exact.commit_id != target.head_sha:
            raise PublisherError("created review metadata is not an active exact-head CHANGES_REQUESTED review")
        return _HistoryRecord(envelope, True, exact.server_id, exact.state, exact.commit_id)

    def _find_record(self, history: _History, target: ReviewTarget, attempt_id: str, record_type: str) -> _HistoryRecord | None:
        return next(
            (record for record in history.current
             if record.envelope.payload.get("record_type") == record_type
             and record.envelope.payload.get("attempt_id") == attempt_id),
            None,
        )

    def _review_metadata(self, history: _History, target: ReviewTarget, attempt_id: str, verdict: str) -> _HistoryRecord | None:
        metadata = self._find_record(history, target, attempt_id, "review-metadata")
        if metadata is None:
            return None
        if verdict == "changes-requested":
            if not metadata.is_review or metadata.state != "CHANGES_REQUESTED" or metadata.commit_id != target.head_sha:
                raise PublisherError("review metadata is not an active exact-head CHANGES_REQUESTED review")
        elif metadata.is_review:
            raise PublisherError("clean verdict cannot reuse a pull request review")
        return metadata

    def _workflow_review_ids(self, history: _History, target: ReviewTarget, keep_node: str) -> list[int]:
        result: list[int] = []
        for record in history.valid:
            payload = record.envelope.payload
            record_target = _target_from_payload(payload) if payload.get("record_type") != "revocation" else None
            if (
                not record.is_review or record.envelope.node_id == keep_node
                or payload.get("record_type") != "review-metadata"
                or record.state != "CHANGES_REQUESTED" or record_target is None
                or record.commit_id != record_target.head_sha
                or record.envelope.author not in self._trusted_authors
            ):
                continue
            result.append(record.server_id)
        return result

    def _reconcile_workflow_reviews(
        self, target: ReviewTarget, attempt_id: str, intent_node: str, keep_node: str,
        *, keep_is_review: bool,
    ) -> tuple[_History, GitHubEnvelope]:
        canonical, history = self._canonical(target, attempt_id, intent_node)
        for _ in range(_MAX_RECONCILIATION_ROUNDS):
            if keep_is_review:
                self._validate_keep_review(history, target, keep_node)
            review_ids = self._workflow_review_ids(history, target, keep_node)
            if review_ids:
                for review_id in review_ids:
                    self._mutate(
                        target,
                        lambda review_id=review_id: self._client.dismiss_workflow_review(
                            target.repository, target.number, review_id,
                            message="Superseded by a current agentic review",
                        ),
                        attempt_id=attempt_id,
                        intent_node=canonical.node_id,
                    )
                    canonical, history = self._canonical(target, attempt_id, intent_node)
                continue
            # Require two consecutive no-stale snapshots. The second fetch
            # closes the window between election and the next mutation.
            stable_canonical, stable_history = self._canonical(target, attempt_id, intent_node)
            if keep_is_review:
                self._validate_keep_review(stable_history, target, keep_node)
            if not self._workflow_review_ids(stable_history, target, keep_node):
                return stable_history, stable_canonical
            history, canonical = stable_history, stable_canonical
        raise PublisherError("workflow review reconciliation did not stabilize")

    def _validate_keep_review(self, history: _History, target: ReviewTarget, keep_node: str) -> None:
        keep = next((record for record in history.current if record.envelope.node_id == keep_node), None)
        if (
            keep is None
            or not keep.is_review
            or keep.envelope.payload.get("record_type") != "review-metadata"
            or keep.state != "CHANGES_REQUESTED"
            or keep.commit_id != target.head_sha
        ):
            raise PublisherError("canonical keep review is not an active exact-head CHANGES_REQUESTED review")

    def _label_present(self, target: ReviewTarget) -> bool:
        getter = getattr(self._client, "list_issue_labels", None)
        if not callable(getter):
            raise LabelError("GitHub client lacks typed label-state retrieval")
        response = getter(target.repository, target.number)
        data = response.data if hasattr(response, "data") else response
        if not isinstance(data, list):
            raise LabelError("GitHub label state is malformed")
        return any(isinstance(item, Mapping) and item.get("name") == _LABEL for item in data)

    def _remove_label(
        self, target: ReviewTarget, attempt_id: str, intent_node: str, keep_node: str, *, keep_is_review: bool,
    ) -> None:
        self._reconcile_workflow_reviews(
            target, attempt_id, intent_node, keep_node, keep_is_review=keep_is_review,
        )
        for _ in range(_MAX_RECONCILIATION_ROUNDS):
            if not self._label_present(target):
                self._assert_target(target)
                self._reconcile_workflow_reviews(
                    target, attempt_id, intent_node, keep_node, keep_is_review=keep_is_review,
                )
                self._assert_target(target)
                return
            _, canonical = self._reconcile_workflow_reviews(
                target, attempt_id, intent_node, keep_node, keep_is_review=keep_is_review,
            )
            try:
                _, canonical, _ = self._mutate(
                    target,
                    lambda: self._client.remove_label(target.repository, target.number, _LABEL),
                    attempt_id=attempt_id,
                    intent_node=canonical.node_id,
                    before_mutation=lambda elected, history: self._validate_label_snapshot(
                        elected, history, target, keep_node, keep_is_review,
                    ),
                    return_snapshot=True,
                )
            except _UnsafeLabelSnapshot:
                self._reconcile_workflow_reviews(
                    target, attempt_id, intent_node, keep_node, keep_is_review=keep_is_review,
                )
                continue
            try:
                self._assert_target(target)
                self._reconcile_workflow_reviews(
                    target, attempt_id, intent_node, keep_node, keep_is_review=keep_is_review,
                )
                self._assert_target(target)
                return
            except Exception:
                self._reapply_label(target, attempt_id)
                raise
        raise PublisherError("label removal reconciliation did not stabilize")

    def _validate_label_snapshot(
        self, canonical: GitHubEnvelope, history: _History, target: ReviewTarget,
        keep_node: str, keep_is_review: bool,
    ) -> None:
        if keep_is_review:
            self._validate_keep_review(history, target, keep_node)
        if self._workflow_review_ids(history, target, keep_node):
            raise _UnsafeLabelSnapshot("stale workflow review appeared before label removal")

    def _recover(self, target: ReviewTarget, attempt_id: str, status: str, reason: str) -> PublishResult:
        try:
            self._reapply_label(target, attempt_id)
        except (_StaleTarget, LabelError) as exc:
            return PublishResult("error", attempt_id, reason=f"{reason}; label recovery failed: {exc}")
        return PublishResult(status, attempt_id, reason=reason)

    def publish(self, proposal: ReviewProposal, target: ReviewTarget) -> PublishResult:
        if not isinstance(proposal, ReviewProposal) or not isinstance(target, ReviewTarget):
            raise PublisherError("publish requires a validated ReviewProposal and complete ReviewTarget")
        if proposal.target != target:
            raise PublisherError("proposal and ReviewTarget do not match")
        if proposal.scope is None:
            raise PublisherError("new proposals require an explicit model/hardware scope")
        if not proposal.validation_ledger and not proposal.exemption_ids:
            raise PublisherError("new proposals require protected validation evidence or an authenticated exemption")
        if self._configuration.source is None or self._configuration.source.repository != target.repository:
            raise PublisherError("authenticated configuration source does not match target repository")
        try:
            validate_publisher_operator_credential(self._operator, target.repository)
        except (TypeError, ValueError) as exc:
            raise PublisherError(str(exc)) from exc
        attempt_id = "attempt-" + proposal.proposal_digest[7:]
        try:
            self._client.revalidate_config_source(self._configuration.source)
            self._assert_target(target)
            capsule = self._reconstruct_and_validate_capsule(proposal, target)
            self._history_capsule = capsule
            try:
                render_report(proposal)
                self._validate_proposal_configuration(proposal, target, capsule)
                # Apply verify-* labels for impacted hardware so downstream agents
                # discover validation tasks by label. Skip when triage is absent or
                # coverage_decision is "none" (no hardware validation needed).
                if proposal.hardware_validation_triage is not None and proposal.hardware_validation_triage.coverage_decision != "none":
                    verify_labels = [
                        "verify-" + arch
                        for arch in proposal.hardware_validation_triage.impacted_hardware
                    ]
                    if verify_labels:
                        self._client.add_labels(target.repository, target.number, verify_labels)
                self._preflight_report_comment(proposal, target, attempt_id, capsule)
            except _PreflightRejected:
                raise
            except PublisherError as exc:
                raise _PreflightRejected(str(exc)) from exc
            if proposal.verdict == "incomplete":
                self._reapply_label(target, attempt_id)
                return PublishResult("incomplete", attempt_id, reason="proposal verdict is incomplete")

            history = self._history(target)
            try:
                elected = validate_protocol(
                    [record.envelope for record in history.current],
                    expected_target=target, trusted_authors=self._trusted_authors,
                    configuration=self._configuration, capsule=capsule,
                ) if history.current else None
            except ValueError as exc:
                if "no valid non-revoked intent" not in str(exc):
                    raise PublisherError(f"invalid current review history: {exc}") from exc
                elected = None
            canonical = elected if isinstance(elected, GitHubEnvelope) else None
            if canonical is not None and canonical.payload.get("attempt_id") != attempt_id:
                return PublishResult("duplicate", attempt_id, reason="a different canonical attempt exists")
            intent = canonical or self._new_comment(target, self._intent_payload(target, attempt_id))
            history = self._history(target)
            canonical, history = self._canonical(target, attempt_id, intent.node_id)
            completion = self._find_record(history, target, attempt_id, "completion")
            if completion is not None:
                report = self._find_record(history, target, attempt_id, "report")
                metadata = self._review_metadata(history, target, attempt_id, proposal.verdict)
                if report is None or metadata is None:
                    raise PublisherError("completion dependencies are missing")
                self._require_matching_report_binding(report, proposal)
                self._remove_label(
                    target, attempt_id, canonical.node_id, metadata.envelope.node_id,
                    keep_is_review=metadata.is_review,
                )
                return PublishResult("duplicate", attempt_id, report.envelope, metadata.envelope, completion.envelope,
                                     "canonical attempt is already complete")

            report = self._find_record(history, target, attempt_id, "report")
            if report is not None:
                self._require_matching_report_binding(report, proposal)
            if report is None:
                report_envelope = self._new_comment(
                    target, self._report_payload(proposal, target, intent, capsule),
                    attempt_id=attempt_id, intent_node=canonical.node_id,
                    visible_body=render_report(proposal),
                )
                report = _HistoryRecord(report_envelope, False, 0)
            history = self._history(target)
            report = self._find_record(history, target, attempt_id, "report") or report
            assert report is not None
            self._require_matching_report_binding(report, proposal)

            metadata = self._review_metadata(history, target, attempt_id, proposal.verdict)
            if metadata is None:
                metadata_payload = self._metadata_payload(target, intent, report.envelope)
                if proposal.verdict == "changes-requested":
                    metadata = self._new_review(target, metadata_payload, attempt_id, canonical.node_id)
                else:
                    metadata_envelope = self._new_comment(
                        target, metadata_payload, attempt_id=attempt_id, intent_node=canonical.node_id,
                    )
                    metadata = _HistoryRecord(metadata_envelope, False, 0)

            history = self._history(target)
            canonical, history = self._canonical(target, attempt_id, intent.node_id)
            completion = self._find_record(history, target, attempt_id, "completion")
            if completion is None:
                history, canonical = self._reconcile_workflow_reviews(
                    target, attempt_id, intent.node_id, metadata.envelope.node_id,
                    keep_is_review=metadata.is_review,
                )
                completion_envelope = self._new_comment(
                    target, self._completion_payload(target, intent, report.envelope, metadata.envelope),
                    attempt_id=attempt_id, intent_node=canonical.node_id,
                )
                completion = _HistoryRecord(completion_envelope, False, 0)
            self._remove_label(
                target, attempt_id, canonical.node_id, metadata.envelope.node_id,
                keep_is_review=metadata.is_review,
            )
            return PublishResult("complete", attempt_id, report.envelope, metadata.envelope, completion.envelope)
        except _PreflightRejected as exc:
            return PublishResult("error", attempt_id, reason=str(exc))
        except _StaleTarget as exc:
            return self._recover(target, attempt_id, "stale", str(exc))
        except _CanonicalChanged as exc:
            return self._recover(target, attempt_id, "stale", str(exc))
        except PublisherError as exc:
            return self._recover(target, attempt_id, "error", str(exc))
        except Exception as exc:
            return self._recover(target, attempt_id, "incomplete", str(exc))


def publish_review(
    client: Any,
    proposal: ReviewProposal,
    target: ReviewTarget,
    *,
    configuration: ReviewConfiguration,
    operator_credential: Mapping[str, Any],
) -> PublishResult:
    return ReviewPublisher(client, configuration=configuration, operator_credential=operator_credential).publish(proposal, target)


__all__ = ["LabelError", "PublishResult", "PublisherError", "ReviewPublisher", "publish_review", "render_report"]
