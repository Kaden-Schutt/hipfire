# Copyright (c) Kaden Schutt
"""Authenticated publication of SHA-bound agentic review records."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
import hashlib
import html
import json
from typing import Any, Callable

from .canonical import canonical_digest, canonical_json, metadata_digest
from .config import (
    ReviewConfiguration,
    validate_operator_credential_manifest,
    validate_publisher_operator_credential,
)
from .github import decode_protocol_body, encode_protocol_body
from .models import GitHubEnvelope, ReviewProposal, ReviewTarget
from .protocol import validate_intent, validate_protocol, validate_revocation


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


_SCHEMA = "agentic-review/v1"
_LABEL = "needs-review"
_MAX_RECONCILIATION_ROUNDS = 4


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
    return "\n".join(lines)


class ReviewPublisher:
    """Publish a validated proposal through the fixed GitHub boundary."""

    def __init__(
        self,
        client: Any,
        *,
        configuration: ReviewConfiguration,
        operator_credential: Mapping[str, Any],
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

    @property
    def _trusted_authors(self) -> frozenset[str]:
        authors = {self._operator["principal"]["login"]}
        apps = self._configuration.trusted_publishers.get("apps", ())
        if isinstance(apps, Sequence) and not isinstance(apps, (str, bytes)):
            authors.update(
                app["login"] for app in apps
                if isinstance(app, Mapping) and isinstance(app.get("login"), str)
            )
        return frozenset(authors)

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
            try:
                payload = decode_protocol_body(body)
            except Exception:
                if body.lstrip().startswith("{") or "agentic-review/v1" in body:
                    raise PublisherError("a protocol record was deleted or edited")
                continue
            if payload.get("schema") != _SCHEMA:
                continue
            try:
                envelope = (
                    self._client.review_envelope(target.repository, target.number, raw["id"])
                    if is_review else self._client.comment_envelope(target.repository, raw["id"])
                )
                record = _HistoryRecord(
                    envelope, is_review, raw["id"],
                    raw.get("state") if is_review else None,
                    raw.get("commit_id") if is_review else None,
                )
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

    def _canonical(self, target: ReviewTarget, attempt_id: str, intent_node: str | None = None) -> GitHubEnvelope:
        history = self._history(target)
        try:
            elected = validate_protocol(
                [record.envelope for record in history.current],
                expected_target=target,
                trusted_authors=self._trusted_authors,
            )
        except ValueError as exc:
            raise _CanonicalChanged("canonical intent is no longer active") from exc
        if not isinstance(elected, GitHubEnvelope):
            raise _CanonicalChanged("canonical intent is not an authenticated envelope")
        if elected.payload.get("attempt_id") != attempt_id or (intent_node and elected.node_id != intent_node):
            raise _CanonicalChanged("canonical review attempt changed")
        return elected

    def _mutate(
        self,
        target: ReviewTarget,
        operation: Callable[[], Any],
        *,
        attempt_id: str | None = None,
        intent_node: str | None = None,
    ) -> Any:
        if attempt_id is not None:
            self._canonical(target, attempt_id, intent_node)
        self._assert_target(target)
        value = operation()
        self._assert_target(target)
        if attempt_id is not None:
            self._canonical(target, attempt_id, intent_node)
        return value

    def _intent_payload(self, target: ReviewTarget, attempt_id: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": _SCHEMA, "record_type": "intent", "record_id": f"intent-{attempt_id}",
            "target": target, "target_key": target.target_key(), "attempt_id": attempt_id,
            "canonical_digest": "",
        }
        payload["canonical_digest"] = canonical_digest({key: value for key, value in payload.items() if key != "canonical_digest"})
        return payload

    def _report_payload(self, proposal: ReviewProposal, target: ReviewTarget, intent: GitHubEnvelope) -> dict[str, Any]:
        body = render_report(proposal)
        return {
            "schema": _SCHEMA, "record_type": "report", "record_id": f"report-{intent.payload['attempt_id']}",
            "target": target, "target_key": target.target_key(), "attempt_id": intent.payload["attempt_id"],
            "intent_record_id": intent.payload["record_id"], "canonical_intent_node_id": intent.node_id,
            "canonical_intent_digest": intent.payload["canonical_digest"], "head_sha": target.head_sha,
            "report_body": body, "report_body_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
        }

    def _metadata_payload(self, target: ReviewTarget, intent: GitHubEnvelope, report: GitHubEnvelope) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": _SCHEMA, "record_type": "review-metadata", "record_id": f"metadata-{intent.payload['attempt_id']}",
            "target": target, "target_key": target.target_key(), "attempt_id": intent.payload["attempt_id"],
            "intent_record_id": intent.payload["record_id"], "head_sha": target.head_sha,
            "report_record_id": report.payload["record_id"], "report_node_id": report.node_id,
            "report_digest": canonical_digest(report.payload), "report_body_sha256": report.payload["report_body_sha256"],
            "canonical_intent_digest": intent.payload["canonical_digest"], "canonical_intent_node_id": intent.node_id,
            "metadata_digest": "",
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
        }

    def _new_comment(
        self, target: ReviewTarget, payload: Mapping[str, Any], *, attempt_id: str | None = None,
        intent_node: str | None = None, visible_body: str | None = None,
    ) -> GitHubEnvelope:
        response = self._mutate(
            target,
            lambda: self._client.create_issue_comment(
                target.repository, target.number, encode_protocol_body(payload, visible_body=visible_body)
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
        server = self._client.get_pull_review(target.repository, target.number, record["id"]).data
        envelope = self._client.review_envelope(target.repository, target.number, record["id"])
        if server.get("state") != "CHANGES_REQUESTED" or server.get("commit_id") != target.head_sha:
            raise PublisherError("created review metadata is not an active exact-head CHANGES_REQUESTED review")
        return _HistoryRecord(envelope, True, record["id"], server["state"], server["commit_id"])

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
        history = self._history(target)
        canonical = self._canonical(target, attempt_id, intent_node)
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
                    history = self._history(target)
                    canonical = self._canonical(target, attempt_id, intent_node)
                continue
            # Require two consecutive no-stale snapshots. The second fetch
            # closes the window between election and the next mutation.
            stable_history = self._history(target)
            stable_canonical = self._canonical(target, attempt_id, intent_node)
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
        if self._label_present(target):
            _, canonical = self._reconcile_workflow_reviews(
                target, attempt_id, intent_node, keep_node, keep_is_review=keep_is_review,
            )
            self._mutate(
                target,
                lambda: self._client.remove_label(target.repository, target.number, _LABEL),
                attempt_id=attempt_id,
                intent_node=canonical.node_id,
            )
            try:
                self._assert_target(target)
                self._reconcile_workflow_reviews(
                    target, attempt_id, intent_node, keep_node, keep_is_review=keep_is_review,
                )
            except Exception:
                self._reapply_label(target, attempt_id)
                raise

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
            if proposal.verdict == "incomplete":
                self._reapply_label(target, attempt_id)
                return PublishResult("incomplete", attempt_id, reason="proposal verdict is incomplete")

            history = self._history(target)
            try:
                elected = validate_protocol(
                    [record.envelope for record in history.current],
                    expected_target=target, trusted_authors=self._trusted_authors,
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
            canonical = self._canonical(target, attempt_id, intent.node_id)
            completion = self._find_record(history, target, attempt_id, "completion")
            if completion is not None:
                report = self._find_record(history, target, attempt_id, "report")
                metadata = self._review_metadata(history, target, attempt_id, proposal.verdict)
                if report is None or metadata is None:
                    raise PublisherError("completion dependencies are missing")
                self._remove_label(
                    target, attempt_id, canonical.node_id, metadata.envelope.node_id,
                    keep_is_review=metadata.is_review,
                )
                return PublishResult("duplicate", attempt_id, report.envelope, metadata.envelope, completion.envelope,
                                     "canonical attempt is already complete")

            report = self._find_record(history, target, attempt_id, "report")
            if report is None:
                report_envelope = self._new_comment(
                    target, self._report_payload(proposal, target, intent),
                    attempt_id=attempt_id, intent_node=canonical.node_id,
                    visible_body=render_report(proposal),
                )
                report = _HistoryRecord(report_envelope, False, 0)
            history = self._history(target)
            report = self._find_record(history, target, attempt_id, "report") or report

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
            canonical = self._canonical(target, attempt_id, intent.node_id)
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
