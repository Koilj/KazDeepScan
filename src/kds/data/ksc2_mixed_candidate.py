"""Build a narrow, evidence-preserving KSC2 mixed bona-fide candidate layer.

The source review contains only explicit positives.  This module turns those rows into
``ManifestRow`` objects for QA/VAD preparation; it never labels an unreviewed KSC2 row and
does not create spoof data or a binary benchmark.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Final, cast

from kds.data.assets import sha256_file
from kds.data.ksc2_mixed_review import AI_REVIEW_FIELDS
from kds.data.manifest import ManifestRow

KSC2_MIXED_REVIEW_METHOD: Final = "single_ai_transcript_semantic_review_v1"
KSC2_MIXED_REVIEWER: Final = "codex_language_review_v1"
KSC2_MIXED_SOURCE_ID: Final = "ksc2_v1"
KSC2_MIXED_SOURCE_LICENSE: Final = "CC-BY-4.0"
_HEX = frozenset("0123456789abcdef")


class Ksc2MixedCandidateError(ValueError):
    """Raised when published KSC2 evidence cannot safely become a candidate layer."""

    def __init__(self, issues: Iterable[str]) -> None:
        self.issues = tuple(issues)
        super().__init__("\n".join(self.issues))


@dataclass(frozen=True, slots=True)
class Ksc2MixedEvidenceRow:
    annotation_id: str
    component: str
    audio_relative_path: str
    audio_sha256: str
    transcript_sha256: str
    source_name: str
    source_license: str
    archive_sha256: str
    source_lock_sha256: str
    candidate_packet_sha256: str
    candidate_receipt_sha256: str
    transcript: str
    ru_evidence_token_indices: str
    ru_evidence_tokens: str
    kk_evidence_token_indices: str
    kk_evidence_tokens: str


@dataclass(frozen=True, slots=True)
class Ksc2MixedAudioInfo:
    duration_s: float
    original_sr: int
    codec: str


def _read_json_object(path: Path, label: str) -> dict[str, object]:
    if not path.is_file():
        raise Ksc2MixedCandidateError([f"{label} does not exist: {path}"])
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Ksc2MixedCandidateError([f"Cannot parse {label}: {error}"]) from error
    if not isinstance(value, dict):
        raise Ksc2MixedCandidateError([f"{label} must be a JSON object."])
    return cast(dict[str, object], value)


def _sha256(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise Ksc2MixedCandidateError([f"{label} must be a SHA-256 string."])
    digest = value.lower()
    if len(digest) != 64 or any(character not in _HEX for character in digest):
        raise Ksc2MixedCandidateError([f"{label} must be a lowercase SHA-256 digest."])
    return digest


def _positive_indices(value: str, label: str) -> None:
    try:
        indices = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise Ksc2MixedCandidateError([f"{label} has invalid token indices."]) from error
    if not indices or any(index < 1 for index in indices):
        raise Ksc2MixedCandidateError([f"{label} must contain positive token indices."])


def load_published_mixed_review(
    review_csv: Path, review_receipt: Path
) -> tuple[Ksc2MixedEvidenceRow, ...]:
    """Load the fixed review only when its receipt pins its exact CSV bytes."""

    if not review_csv.is_file():
        raise Ksc2MixedCandidateError([f"KSC2 mixed review CSV does not exist: {review_csv}"])
    try:
        with review_csv.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or tuple(reader.fieldnames) != AI_REVIEW_FIELDS:
                raise Ksc2MixedCandidateError(["KSC2 mixed review CSV has an invalid schema."])
            source_rows = [
                {field: (row.get(field) or "").strip() for field in AI_REVIEW_FIELDS}
                for row in reader
            ]
    except OSError as error:
        raise Ksc2MixedCandidateError([f"Cannot read KSC2 mixed review CSV: {error}"]) from error
    if not source_rows:
        raise Ksc2MixedCandidateError(["KSC2 mixed review CSV has no rows."])

    receipt = _read_json_object(review_receipt, "KSC2 mixed review receipt")
    review_hash = sha256_file(review_csv)
    if (
        receipt.get("schema_version") != 1
        or receipt.get("output_csv") != review_csv.as_posix()
        or _sha256(receipt.get("output_csv_sha256"), "review receipt output_csv_sha256")
        != review_hash
        or receipt.get("review_method") != KSC2_MIXED_REVIEW_METHOD
        or receipt.get("reviewer") != KSC2_MIXED_REVIEWER
        or receipt.get("confirmed_mixed_rows") != len(source_rows)
    ):
        raise Ksc2MixedCandidateError(["KSC2 mixed review receipt does not pin this CSV."])

    evidence: list[Ksc2MixedEvidenceRow] = []
    seen_ids: set[str] = set()
    component_counts: Counter[str] = Counter()
    for number, row in enumerate(source_rows, start=2):
        identifier = row["annotation_id"]
        if not identifier or identifier in seen_ids:
            raise Ksc2MixedCandidateError(
                [f"KSC2 mixed review has duplicate/empty annotation_id at row {number}."]
            )
        seen_ids.add(identifier)
        if (
            row["language"] != "mixed"
            or row["code_switch"] != "true"
            or row["review_method"] != KSC2_MIXED_REVIEW_METHOD
            or row["reviewer"] != KSC2_MIXED_REVIEWER
            or row["source_name"] != KSC2_MIXED_SOURCE_ID
            or row["source_license"] != KSC2_MIXED_SOURCE_LICENSE
            or not row["transcript"]
            or not row["ru_evidence_tokens"]
            or not row["kk_evidence_tokens"]
        ):
            raise Ksc2MixedCandidateError(
                [f"KSC2 mixed review row {number} is not explicit mixed evidence."]
            )
        _positive_indices(row["ru_evidence_token_indices"], f"KSC2 review row {number} RU")
        _positive_indices(row["kk_evidence_token_indices"], f"KSC2 review row {number} KK")
        for field in (
            "audio_sha256",
            "transcript_sha256",
            "archive_sha256",
            "source_lock_sha256",
            "candidate_packet_sha256",
            "candidate_receipt_sha256",
        ):
            _sha256(row[field], f"KSC2 review row {number} {field}")
        if not row["audio_relative_path"].startswith("raw/ksc2_v1/"):
            raise Ksc2MixedCandidateError(
                [f"KSC2 review row {number} has an unexpected audio path."]
            )
        evidence.append(
            Ksc2MixedEvidenceRow(
                annotation_id=identifier,
                component=row["component"],
                audio_relative_path=row["audio_relative_path"],
                audio_sha256=row["audio_sha256"],
                transcript_sha256=row["transcript_sha256"],
                source_name=row["source_name"],
                source_license=row["source_license"],
                archive_sha256=row["archive_sha256"],
                source_lock_sha256=row["source_lock_sha256"],
                candidate_packet_sha256=row["candidate_packet_sha256"],
                candidate_receipt_sha256=row["candidate_receipt_sha256"],
                transcript=row["transcript"],
                ru_evidence_token_indices=row["ru_evidence_token_indices"],
                ru_evidence_tokens=row["ru_evidence_tokens"],
                kk_evidence_token_indices=row["kk_evidence_token_indices"],
                kk_evidence_tokens=row["kk_evidence_tokens"],
            )
        )
        component_counts[row["component"]] += 1
    receipt_counts = receipt.get("confirmed_mixed_counts_by_component")
    if receipt_counts != dict(sorted(component_counts.items())):
        raise Ksc2MixedCandidateError(["KSC2 mixed review component counts differ from receipt."])
    return tuple(evidence)


def mixed_bonafide_rows(
    evidence: Sequence[Ksc2MixedEvidenceRow],
    audio_info_by_id: Mapping[str, Ksc2MixedAudioInfo],
    *,
    created_at: str,
) -> list[ManifestRow]:
    """Make test-candidate bona-fide rows without claiming speaker independence."""

    try:
        datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except ValueError as error:
        raise Ksc2MixedCandidateError(["created_at must be an ISO-8601 timestamp."]) from error
    if not evidence:
        raise Ksc2MixedCandidateError(["KSC2 mixed evidence is empty."])
    rows: list[ManifestRow] = []
    for item in evidence:
        info = audio_info_by_id.get(item.annotation_id)
        if info is None or info.duration_s <= 0 or info.original_sr <= 0 or not info.codec:
            raise Ksc2MixedCandidateError(
                [f"KSC2 mixed evidence has invalid audio metadata: {item.annotation_id}."]
            )
        text_id = f"ksc2_v1:transcript:{item.transcript_sha256}"
        rows.append(
            ManifestRow(
                sample_id=item.annotation_id,
                relative_path=item.audio_relative_path,
                sha256=item.audio_sha256,
                split="test",
                label="bonafide",
                language="mixed",
                code_switch="true",
                parent_group_id=f"ksc2_v1:recording:{item.annotation_id}",
                source_name=item.source_name,
                source_license=item.source_license,
                rights_basis=(
                    "KSC2 CC-BY-4.0 source recording; explicit single-AI transcript-review "
                    f"evidence; archive={item.archive_sha256}; "
                    f"source_lock={item.source_lock_sha256}"
                ),
                speaker_pseudo_id="ksc2_v1:unknown",
                text_id=text_id,
                text_hash=item.transcript_sha256,
                duration_s=info.duration_s,
                generator_family="",
                generator_name="",
                generator_version="",
                voice_id="",
                clone_consent_id="",
                device="source_recording_unknown",
                capture_route="KSC2_Test_component_recording",
                original_sr=info.original_sr,
                codec=info.codec,
                augmentation_chain="",
                augmentation_seed="",
                created_at=created_at,
            )
        )
    return rows


def select_mixed_smoke_evidence(
    evidence: Sequence[Ksc2MixedEvidenceRow], *, limit: int, seed: str
) -> tuple[Ksc2MixedEvidenceRow, ...]:
    """Select a deterministic tiny smoke set that covers every reviewed component once."""

    if limit <= 0:
        raise Ksc2MixedCandidateError(["Smoke-test limit must be positive."])
    if not seed:
        raise Ksc2MixedCandidateError(["Smoke-test seed must not be empty."])
    if len(evidence) < limit:
        raise Ksc2MixedCandidateError(
            [f"Smoke-test needs {limit} reviewed rows, found {len(evidence)}."]
        )

    def rank(item: Ksc2MixedEvidenceRow) -> bytes:
        return hashlib.sha256(f"{seed}:{item.annotation_id}".encode()).digest()

    by_component: dict[str, list[Ksc2MixedEvidenceRow]] = {}
    for item in evidence:
        by_component.setdefault(item.component, []).append(item)
    if len(by_component) > limit:
        raise Ksc2MixedCandidateError(
            ["Smoke-test limit is smaller than the reviewed component count."]
        )
    selected = [min(rows, key=rank) for _component, rows in sorted(by_component.items())]
    selected_ids = {item.annotation_id for item in selected}
    remainder = sorted(
        (item for item in evidence if item.annotation_id not in selected_ids), key=rank
    )
    return tuple(selected + remainder[: limit - len(selected)])
