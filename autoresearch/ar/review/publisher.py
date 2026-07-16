# Copyright (c) Kaden Schutt
"""Authenticated publication of SHA-bound agentic review records."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import html
import json
import re
from typing import Any

from .canonical import canonical_digest, canonical_json, metadata_digest
from .config import ReviewConfiguration, validate_operator_credential_manifest
from .models import GitHubEnvelope, ReviewProposal, ReviewTarget
from .protocol import validate_protocol


class PublisherError(RuntimeError):
    """The publisher rejected an input or an authenticated protocol state."""


@dataclass(frozen=True)
class PublishResult:
    status: str
    attempt_id: str
    report_envelope: GitHubEnvelope | None = None
    review_envelope: GitHubEnvelope | None = None
    completion_envelope: GitHubEnvelope | None = None
    reason: str | None = None


class _StaleTarget(RuntimeError):
    pass


_SCHEMA = "agentic-review/v1"
_LABEL = "needs-review"


def _json_body(payload: Mapping[str, Any]) -> str:
    return canonical_json(payload).decode("utf-8")


def _target_from_response(data: Mapping[str, Any], expected: ReviewTarget) -> ReviewTarget:
    try:
        head = data["head"]
        base = data["base"]
        head_repo = head["repo"]["full_name"]
        repository = base.get("repo", {}).get("full_name", expected.repository)
        merge_base = data.get("merge_base_sha", expected.merge_base_sha)
        target = ReviewTarget(
            repository,
            expected.number,
            head_repo,
            head["sha"],
            base["ref"],
            base["sha"],
            merge_base,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise PublisherError("GitHub pull response is not a complete ReviewTarget") from exc
    if data.get("number") != expected.number:
        raise PublisherError("GitHub pull response number does not match ReviewTarget")
    return target


def _escape_markdown(value: str) -> str:
    escaped = html.escape(value, quote=False)
    return re.sub(r"([\\`*_\[\]#])", r"\\\1", escaped)


def render_report(proposal: ReviewProposal) -> str:
    """Render only structured, escaped proposal fields into visible Markdown."""
    lines = ["## Agentic review", "", f"Verdict: `{_escape_markdown(proposal.verdict)}`"]
    if proposal.findings:
        lines.extend(("", "### Findings"))
        for finding in proposal.findings:
            path = _escape_markdown(finding.path)
            message = _escape_markdown(finding.message)
            lines.append(f"- `{path}:{finding.range[0]}-{finding.range[1]}` ({finding.severity}): {message}")
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
        if "publish" not in operator_credential["allowed_operations"]:
            raise PublisherError("operator credential does not attest publication")
        self._client = client
        self._configuration = configuration
        self._operator = deepcopy(dict(operator_credential))

    @property
    def _trusted_authors(self) -> frozenset[str]:
        authors = {self._operator["principal"]["login"]}
        apps = self._configuration.trusted_publishers.get("apps", ())
        if isinstance(apps, Sequence) and not isinstance(apps, (str, bytes)):
            authors.update(app["login"] for app in apps if isinstance(app, Mapping) and isinstance(app.get("login"), str))
        return frozenset(authors)

    def _pull_target(self, target: ReviewTarget) -> ReviewTarget:
        response = self._client.get_pull_request(target.repository, target.number)
        data = response.data if hasattr(response, "data") else response
        if not isinstance(data, Mapping):
            raise PublisherError("GitHub pull response is not an object")
        current = _target_from_response(data, target)
        return current

    def _assert_target(self, target: ReviewTarget) -> None:
        if self._pull_target(target) != target:
            raise _StaleTarget("review target changed")

    def _reapply_label(self, target: ReviewTarget) -> bool:
        try:
            self._pull_target(target)
            self._client.add_labels(target.repository, target.number, [_LABEL])
            self._pull_target(target)
            return True
        except Exception:
            return False

    def _mutate(self, target: ReviewTarget, operation):
        self._assert_target(target)
        value = operation()
        try:
            self._assert_target(target)
        except _StaleTarget:
            self._reapply_label(target)
            raise
        return value

    @staticmethod
    def _protocolish(body: Any) -> bool:
        if not isinstance(body, str):
            return False
        try:
            value = json.loads(body)
        except (TypeError, ValueError, json.JSONDecodeError):
            return False
        return isinstance(value, Mapping) and value.get("schema") == _SCHEMA

    def _history(self, target: ReviewTarget) -> list[GitHubEnvelope]:
        records: list[GitHubEnvelope] = []
        comments = self._client.list_issue_comments(target.repository, target.number).data
        reviews = self._client.list_pull_reviews(target.repository, target.number).data
        for record in [*(comments or []), *(reviews or [])]:
            if not isinstance(record, Mapping):
                raise PublisherError("GitHub history contains a malformed record")
            if not self._protocolish(record.get("body")):
                continue
            try:
                if "submitted_at" in record:
                    envelope = self._client.review_envelope(target.repository, target.number, record["id"])
                else:
                    envelope = self._client.comment_envelope(target.repository, record["id"])
            except Exception as exc:
                raise PublisherError("a protocol record was deleted or edited") from exc
            records.append(envelope)
        try:
            validate_protocol(records, expected_target=target, trusted_authors=self._trusted_authors)
        except ValueError as exc:
            # A history containing only unrelated protocol records is still
            # meaningful; malformed records must not be silently overwritten.
            if records and "no valid non-revoked intent" not in str(exc):
                raise PublisherError(f"invalid authenticated review history: {exc}") from exc
        return records

    def _intent_payload(self, target: ReviewTarget, attempt_id: str) -> dict[str, Any]:
        payload = {
            "schema": _SCHEMA,
            "record_type": "intent",
            "record_id": f"intent-{attempt_id}",
            "target": target,
            "target_key": target.target_key(),
            "attempt_id": attempt_id,
            "canonical_digest": "",
        }
        payload["canonical_digest"] = canonical_digest(
            {key: value for key, value in payload.items() if key != "canonical_digest"}
        )
        return payload

    def _report_payload(self, proposal: ReviewProposal, target: ReviewTarget, intent: GitHubEnvelope) -> dict[str, Any]:
        body = render_report(proposal)
        payload = {
            "schema": _SCHEMA,
            "record_type": "report",
            "record_id": f"report-{intent.payload['attempt_id']}",
            "target": target,
            "target_key": target.target_key(),
            "attempt_id": intent.payload["attempt_id"],
            "intent_record_id": intent.payload["record_id"],
            "canonical_intent_node_id": intent.node_id,
            "canonical_intent_digest": intent.payload["canonical_digest"],
            "head_sha": target.head_sha,
            "report_body": body,
            "report_body_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
        }
        return payload

    def _metadata_payload(self, target: ReviewTarget, intent: GitHubEnvelope, report: GitHubEnvelope) -> dict[str, Any]:
        payload = {
            "schema": _SCHEMA,
            "record_type": "review-metadata",
            "record_id": f"metadata-{intent.payload['attempt_id']}",
            "target": target,
            "target_key": target.target_key(),
            "attempt_id": intent.payload["attempt_id"],
            "intent_record_id": intent.payload["record_id"],
            "head_sha": target.head_sha,
            "report_record_id": report.payload["record_id"],
            "report_node_id": report.node_id,
            "report_digest": canonical_digest(report.payload),
            "report_body_sha256": report.payload["report_body_sha256"],
            "canonical_intent_digest": intent.payload["canonical_digest"],
            "canonical_intent_node_id": intent.node_id,
            "metadata_digest": "",
        }
        payload["metadata_digest"] = metadata_digest(payload)
        return payload

    def _completion_payload(
        self, target: ReviewTarget, intent: GitHubEnvelope, report: GitHubEnvelope, metadata: GitHubEnvelope
    ) -> dict[str, Any]:
        return {
            "schema": _SCHEMA,
            "record_type": "completion",
            "record_id": f"completion-{intent.payload['attempt_id']}",
            "target": target,
            "target_key": target.target_key(),
            "attempt_id": intent.payload["attempt_id"],
            "intent_record_id": intent.payload["record_id"],
            "head_sha": target.head_sha,
            "canonical_intent_digest": intent.payload["canonical_digest"],
            "canonical_intent_node_id": intent.node_id,
            "report_record_id": report.payload["record_id"],
            "report_node_id": report.node_id,
            "report_digest": canonical_digest(report.payload),
            "metadata_record_id": metadata.payload["record_id"],
            "metadata_digest": metadata.payload["metadata_digest"],
        }

    def _new_comment(self, target: ReviewTarget, payload: Mapping[str, Any]) -> GitHubEnvelope:
        response = self._mutate(
            target,
            lambda: self._client.create_issue_comment(target.repository, target.number, _json_body(payload)),
        )
        record = response.data if hasattr(response, "data") else response
        if not isinstance(record, Mapping) or not isinstance(record.get("id"), int):
            raise PublisherError("GitHub comment mutation did not return a server record")
        return self._client.comment_envelope(target.repository, record["id"])

    def _workflow_reviews(self, records: Sequence[GitHubEnvelope], target: ReviewTarget, keep_node: str | None) -> list[int]:
        ids: list[int] = []
        for envelope in records:
            if envelope.node_id == keep_node or envelope.payload.get("record_type") != "review-metadata":
                continue
            if envelope.payload.get("target_key") != target.target_key() or envelope.author not in self._trusted_authors:
                continue
            # Only records that came from the review endpoint have a matching
            # server review ID in the current API listing.
            for record in self._client.list_pull_reviews(target.repository, target.number).data:
                if record.get("node_id") == envelope.node_id and isinstance(record.get("id"), int):
                    ids.append(record["id"])
        return ids

    def publish(self, proposal: ReviewProposal, target: ReviewTarget) -> PublishResult:
        if not isinstance(proposal, ReviewProposal) or not isinstance(target, ReviewTarget):
            raise PublisherError("publish requires a validated ReviewProposal and complete ReviewTarget")
        if proposal.target != target:
            raise PublisherError("proposal and ReviewTarget do not match")
        attempt_id = "attempt-" + proposal.proposal_digest[7:]
        try:
            self._client.revalidate_config_source(self._configuration.source)
            self._assert_target(target)
            records = self._history(target)
            try:
                elected = validate_protocol(
                    records, expected_target=target, trusted_authors=self._trusted_authors
                ) if records else None
            except ValueError as exc:
                if "no valid non-revoked intent" not in str(exc):
                    raise PublisherError(f"invalid authenticated review history: {exc}") from exc
                elected = None
            canonical = elected if isinstance(elected, GitHubEnvelope) else None
            if canonical is not None and canonical.payload["attempt_id"] != attempt_id:
                return PublishResult("duplicate", attempt_id, reason="a different canonical attempt exists")
            intent = canonical if canonical is not None else self._new_comment(target, self._intent_payload(target, attempt_id))
            records = self._history(target)
            elected = validate_protocol(records, expected_target=target, trusted_authors=self._trusted_authors)
            if not isinstance(elected, GitHubEnvelope):
                raise PublisherError("canonical intent is not a typed GitHub envelope")
            canonical = elected
            if canonical.node_id != intent.node_id:
                return PublishResult("duplicate", attempt_id, reason="intent is not canonical")
            existing_completion = next(
                (item for item in records
                 if item.payload.get("record_type") == "completion"
                 and item.payload.get("attempt_id") == attempt_id),
                None,
            )
            if existing_completion is not None:
                existing_report = next(
                    (item for item in records
                     if item.payload.get("record_type") == "report"
                     and item.payload.get("attempt_id") == attempt_id),
                    None,
                )
                existing_metadata = next(
                    (item for item in records
                     if item.payload.get("record_type") == "review-metadata"
                     and item.payload.get("attempt_id") == attempt_id),
                    None,
                )
                return PublishResult(
                    "duplicate", attempt_id, existing_report, existing_metadata, existing_completion,
                    "canonical attempt is already complete",
                )
            by_type = {envelope.payload["record_type"]: envelope for envelope in records
                       if envelope.payload.get("attempt_id") == attempt_id}
            report = by_type.get("report")
            if report is None:
                report = self._new_comment(target, self._report_payload(proposal, target, intent))
            records = self._history(target)
            report = next((item for item in records if item.payload.get("record_type") == "report" and item.payload.get("attempt_id") == attempt_id), report)
            metadata = next((item for item in records if item.payload.get("record_type") == "review-metadata" and item.payload.get("attempt_id") == attempt_id), None)
            if metadata is None:
                metadata_payload = self._metadata_payload(target, intent, report)
                if proposal.verdict == "changes-requested":
                    response = self._mutate(
                        target,
                        lambda: self._client.create_pull_request_review(
                            target.repository,
                            target.number,
                            body=_json_body(metadata_payload),
                            event="REQUEST_CHANGES",
                            commit_id=target.head_sha,
                        ),
                    )
                    record = response.data if hasattr(response, "data") else response
                    metadata = self._client.review_envelope(target.repository, target.number, record["id"])
                else:
                    metadata = self._new_comment(target, metadata_payload)
            records = self._history(target)
            completion = next((item for item in records if item.payload.get("record_type") == "completion" and item.payload.get("attempt_id") == attempt_id), None)
            if completion is None:
                workflow_ids = self._workflow_reviews(records, target, metadata.node_id)
                for review_id in workflow_ids:
                    self._mutate(
                        target,
                        lambda review_id=review_id: self._client.dismiss_workflow_review(
                            target.repository, target.number, review_id, message="Superseded by a current agentic review"
                        ),
                    )
                completion = self._new_comment(target, self._completion_payload(target, intent, report, metadata))
            self._mutate(target, lambda: self._client.remove_label(target.repository, target.number, _LABEL))
            return PublishResult("complete", attempt_id, report, metadata, completion)
        except _StaleTarget as exc:
            self._reapply_label(target)
            return PublishResult("stale", attempt_id, reason=str(exc))
        except PublisherError as exc:
            self._reapply_label(target)
            return PublishResult("error", attempt_id, reason=str(exc))
        except Exception as exc:
            self._reapply_label(target)
            return PublishResult("incomplete", attempt_id, reason=str(exc))


def publish_review(
    client: Any,
    proposal: ReviewProposal,
    target: ReviewTarget,
    *,
    configuration: ReviewConfiguration,
    operator_credential: Mapping[str, Any],
) -> PublishResult:
    return ReviewPublisher(
        client, configuration=configuration, operator_credential=operator_credential
    ).publish(proposal, target)


__all__ = ["PublishResult", "PublisherError", "ReviewPublisher", "publish_review", "render_report"]
