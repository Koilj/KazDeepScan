"""Fail-closed acoustic language-preservation gate for a frozen mixed pair layer.

Transcript evidence says what a synthesizer was asked to say, not what its WAV contains.
This module makes that distinction explicit: it publishes an immutable listening packet and
accepts a gate only after two distinct, complete acoustic reviews for every locked asset.
It neither runs ASR nor assigns language labels from text heuristics.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final, cast

from kds.data.assets import sha256_file
from kds.data.ksc2_mixed_candidate import Ksc2MixedEvidenceRow
from kds.data.manifest import ManifestRow

MIXED_ACOUSTIC_GATE_SCHEMA_VERSION: Final = 1
MIXED_ACOUSTIC_GATE_PROTOCOL_ID: Final = "mixed-acoustic-language-preservation-v1"
MIXED_ACOUSTIC_GATE_PACKET_FIELDS: Final = (
    "protocol_id",
    "annotation_id",
    "component",
    "label",
    "sample_id",
    "relative_path",
    "audio_sha256",
    "text_hash",
    "input_transcript",
    "ru_evidence_tokens",
    "kk_evidence_tokens",
)
MIXED_ACOUSTIC_GATE_REVIEW_FIELDS: Final = (
    "protocol_id",
    "packet_sha256",
    "annotation_id",
    "label",
    "sample_id",
    "audio_sha256",
    "reviewer_pseudo_id",
    "review_status",
    "ru_evidence_audible",
    "kk_evidence_audible",
    "lexical_content_preserved",
    "notes",
)
_REVIEW_STATUSES: Final = frozenset({"pass", "fail", "inconclusive"})
_AUDIBILITY_VALUES: Final = frozenset({"yes", "no", "unknown"})
_HEX: Final = frozenset("0123456789abcdef")


class MixedAcousticGateError(ValueError):
    """Raised when an acoustic review packet or its evidence is unsafe to use."""

    def __init__(self, issues: Iterable[str]) -> None:
        self.issues = tuple(issues)
        super().__init__("\n".join(self.issues))


@dataclass(frozen=True, slots=True)
class MixedAcousticGatePacketRow:
    protocol_id: str
    annotation_id: str
    component: str
    label: str
    sample_id: str
    relative_path: str
    audio_sha256: str
    text_hash: str
    input_transcript: str
    ru_evidence_tokens: str
    kk_evidence_tokens: str


@dataclass(frozen=True, slots=True)
class MixedAcousticGateReview:
    protocol_id: str
    packet_sha256: str
    annotation_id: str
    label: str
    sample_id: str
    audio_sha256: str
    reviewer_pseudo_id: str
    review_status: str
    ru_evidence_audible: str
    kk_evidence_audible: str
    lexical_content_preserved: str
    notes: str


@dataclass(frozen=True, slots=True)
class MixedAcousticGateAssetResult:
    annotation_id: str
    label: str
    sample_id: str
    audio_sha256: str
    review_count: int
    reviewers: tuple[str, ...]
    decision: str


def build_mixed_acoustic_gate_packet(
    candidate_rows: Sequence[ManifestRow],
    evidence_rows: Sequence[Ksc2MixedEvidenceRow],
    pair_lock: Mapping[str, object],
) -> tuple[MixedAcousticGatePacketRow, ...]:
    """Bind every candidate audio to the explicit transcript evidence that selected its pair."""

    issues: list[str] = []
    if len(candidate_rows) != 60:
        issues.append(
            f"Mixed acoustic gate expects exactly 60 candidate assets, got {len(candidate_rows)}."
        )
    labels = Counter(row.label for row in candidate_rows)
    if labels != Counter({"bonafide": 30, "spoof": 30}):
        issues.append("Mixed acoustic gate needs exactly 30 bona-fide and 30 spoof assets.")
    if any(
        row.split != "test" or row.language != "mixed" or row.code_switch != "true"
        for row in candidate_rows
    ):
        issues.append("Mixed acoustic gate accepts only mixed code-switch test rows.")
    evidence_by_id = {item.annotation_id: item for item in evidence_rows}
    if len(evidence_by_id) != 32:
        issues.append("Published KSC2 evidence must contain exactly 32 unique rows.")
    _validate_pair_lock(pair_lock, candidate_rows, evidence_by_id, issues)
    if issues:
        raise MixedAcousticGateError(issues)

    rows_by_text: dict[str, list[ManifestRow]] = defaultdict(list)
    for row in candidate_rows:
        rows_by_text[row.text_hash].append(row)
    packet: list[MixedAcousticGatePacketRow] = []
    for text_hash, paired_rows in rows_by_text.items():
        if len(paired_rows) != 2 or {row.label for row in paired_rows} != {"bonafide", "spoof"}:
            raise MixedAcousticGateError(
                [f"Candidate text hash {text_hash} is not one exact pair."]
            )
        bonafide = next(row for row in paired_rows if row.label == "bonafide")
        evidence = evidence_by_id.get(bonafide.sample_id)
        if evidence is None:
            raise MixedAcousticGateError(
                [f"Candidate bona-fide sample lacks transcript evidence: {bonafide.sample_id}"]
            )
        if evidence.transcript_sha256 != text_hash:
            raise MixedAcousticGateError(
                [f"Evidence transcript hash differs from candidate pair: {bonafide.sample_id}"]
            )
        for row in paired_rows:
            packet.append(
                MixedAcousticGatePacketRow(
                    protocol_id=MIXED_ACOUSTIC_GATE_PROTOCOL_ID,
                    annotation_id=evidence.annotation_id,
                    component=evidence.component,
                    label=row.label,
                    sample_id=row.sample_id,
                    relative_path=row.relative_path,
                    audio_sha256=row.sha256,
                    text_hash=row.text_hash,
                    input_transcript=evidence.transcript,
                    ru_evidence_tokens=evidence.ru_evidence_tokens,
                    kk_evidence_tokens=evidence.kk_evidence_tokens,
                )
            )
    return tuple(sorted(packet, key=lambda item: (item.annotation_id, item.label, item.sample_id)))


def write_mixed_acoustic_gate_packet(
    path: Path, rows: Sequence[MixedAcousticGatePacketRow]
) -> None:
    """Write a new, immutable packet; reviewer decisions belong in a separate CSV."""

    if path.exists() or not path.parent.is_dir():
        raise MixedAcousticGateError(
            ["Acoustic gate packet output must be new with an existing parent."]
        )
    if len(rows) != 60:
        raise MixedAcousticGateError(["Acoustic gate packet must contain exactly 60 rows."])
    try:
        with path.open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=MIXED_ACOUSTIC_GATE_PACKET_FIELDS)
            writer.writeheader()
            writer.writerows(asdict(row) for row in rows)
    except OSError as error:
        raise MixedAcousticGateError([f"Cannot write acoustic gate packet: {error}"]) from error


def read_mixed_acoustic_gate_packet(path: Path) -> tuple[MixedAcousticGatePacketRow, ...]:
    """Read a strict packet and refuse any schema or duplicate binding drift."""

    if not path.is_file():
        raise MixedAcousticGateError([f"Acoustic gate packet does not exist: {path}"])
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if (
                reader.fieldnames is None
                or tuple(reader.fieldnames) != MIXED_ACOUSTIC_GATE_PACKET_FIELDS
            ):
                raise MixedAcousticGateError(["Acoustic gate packet has an invalid schema."])
            rows = tuple(
                MixedAcousticGatePacketRow(
                    **{
                        field: (raw.get(field) or "").strip()
                        for field in MIXED_ACOUSTIC_GATE_PACKET_FIELDS
                    }
                )
                for raw in reader
            )
    except OSError as error:
        raise MixedAcousticGateError([f"Cannot read acoustic gate packet: {error}"]) from error
    if len(rows) != 60:
        raise MixedAcousticGateError([f"Acoustic gate packet must have 60 rows, got {len(rows)}."])
    keys = {(row.sample_id, row.audio_sha256) for row in rows}
    if len(keys) != len(rows):
        raise MixedAcousticGateError(["Acoustic gate packet has duplicate asset bindings."])
    if any(
        row.protocol_id != MIXED_ACOUSTIC_GATE_PROTOCOL_ID
        or row.label not in {"bonafide", "spoof"}
        or not row.input_transcript
        or not row.ru_evidence_tokens
        or not row.kk_evidence_tokens
        or not _is_sha256(row.audio_sha256)
        or not _is_sha256(row.text_hash)
        for row in rows
    ):
        raise MixedAcousticGateError(["Acoustic gate packet contains an invalid binding."])
    return rows


def write_mixed_acoustic_gate_review_template(
    path: Path, packet_path: Path, reviewer_pseudo_id: str
) -> None:
    """Create one independent 60-row reviewer worksheet with fail-closed default decisions."""

    reviewer = reviewer_pseudo_id.strip()
    if not reviewer or any(character in reviewer for character in ",\r\n"):
        raise MixedAcousticGateError(["reviewer_pseudo_id must be non-empty and single-line."])
    if path.exists() or not path.parent.is_dir():
        raise MixedAcousticGateError(
            ["Acoustic gate review template output must be new with an existing parent."]
        )
    packet = read_mixed_acoustic_gate_packet(packet_path)
    packet_hash = sha256_file(packet_path)
    try:
        with path.open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=MIXED_ACOUSTIC_GATE_REVIEW_FIELDS)
            writer.writeheader()
            for item in packet:
                writer.writerow(
                    {
                        "protocol_id": MIXED_ACOUSTIC_GATE_PROTOCOL_ID,
                        "packet_sha256": packet_hash,
                        "annotation_id": item.annotation_id,
                        "label": item.label,
                        "sample_id": item.sample_id,
                        "audio_sha256": item.audio_sha256,
                        "reviewer_pseudo_id": reviewer,
                        "review_status": "inconclusive",
                        "ru_evidence_audible": "unknown",
                        "kk_evidence_audible": "unknown",
                        "lexical_content_preserved": "unknown",
                        "notes": "",
                    }
                )
    except OSError as error:
        raise MixedAcousticGateError(
            [f"Cannot write acoustic gate review template: {error}"]
        ) from error


def read_mixed_acoustic_gate_reviews(path: Path) -> tuple[MixedAcousticGateReview, ...]:
    """Load reviewer assertions only; their validity is established against the packet later."""

    if not path.is_file():
        raise MixedAcousticGateError([f"Acoustic gate review CSV does not exist: {path}"])
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if (
                reader.fieldnames is None
                or tuple(reader.fieldnames) != MIXED_ACOUSTIC_GATE_REVIEW_FIELDS
            ):
                raise MixedAcousticGateError(["Acoustic gate review CSV has an invalid schema."])
            rows = tuple(
                MixedAcousticGateReview(
                    **{
                        field: (raw.get(field) or "").strip()
                        for field in MIXED_ACOUSTIC_GATE_REVIEW_FIELDS
                    }
                )
                for raw in reader
            )
    except OSError as error:
        raise MixedAcousticGateError([f"Cannot read acoustic gate review CSV: {error}"]) from error
    if not rows:
        raise MixedAcousticGateError(["Acoustic gate review CSV has no rows."])
    return rows


def evaluate_mixed_acoustic_gate(
    packet_path: Path, reviews: Sequence[MixedAcousticGateReview]
) -> tuple[dict[str, object], tuple[MixedAcousticGateAssetResult, ...]]:
    """Evaluate the fixed two-review gate without thresholds or model-derived language labels."""

    packet = read_mixed_acoustic_gate_packet(packet_path)
    packet_hash = sha256_file(packet_path)
    packet_by_sample = {row.sample_id: row for row in packet}
    reviews_by_sample: dict[str, list[MixedAcousticGateReview]] = defaultdict(list)
    issues: list[str] = []
    seen_reviewer_asset: set[tuple[str, str]] = set()
    for number, review in enumerate(reviews, start=2):
        expected = packet_by_sample.get(review.sample_id)
        if expected is None:
            issues.append(f"Review row {number} references an asset outside the immutable packet.")
            continue
        if (
            review.protocol_id != MIXED_ACOUSTIC_GATE_PROTOCOL_ID
            or review.packet_sha256 != packet_hash
            or review.annotation_id != expected.annotation_id
            or review.label != expected.label
            or review.audio_sha256 != expected.audio_sha256
            or not review.reviewer_pseudo_id
            or review.review_status not in _REVIEW_STATUSES
            or review.ru_evidence_audible not in _AUDIBILITY_VALUES
            or review.kk_evidence_audible not in _AUDIBILITY_VALUES
            or review.lexical_content_preserved not in _AUDIBILITY_VALUES
        ):
            issues.append(
                f"Review row {number} does not match the immutable packet or enum contract."
            )
            continue
        reviewer_asset = (review.sample_id, review.reviewer_pseudo_id)
        if reviewer_asset in seen_reviewer_asset:
            issues.append(f"Reviewer duplicated an asset decision at review row {number}.")
            continue
        seen_reviewer_asset.add(reviewer_asset)
        reviews_by_sample[review.sample_id].append(review)
    if issues:
        raise MixedAcousticGateError(issues)

    results: list[MixedAcousticGateAssetResult] = []
    for item in packet:
        asset_reviews = reviews_by_sample[item.sample_id]
        reviewers = tuple(sorted(review.reviewer_pseudo_id for review in asset_reviews))
        passed = (
            len(asset_reviews) == 2
            and len(set(reviewers)) == 2
            and all(
                review.review_status == "pass"
                and review.ru_evidence_audible == "yes"
                and review.kk_evidence_audible == "yes"
                and review.lexical_content_preserved == "yes"
                for review in asset_reviews
            )
        )
        results.append(
            MixedAcousticGateAssetResult(
                annotation_id=item.annotation_id,
                label=item.label,
                sample_id=item.sample_id,
                audio_sha256=item.audio_sha256,
                review_count=len(asset_reviews),
                reviewers=reviewers,
                decision="pass" if passed else "not_eligible",
            )
        )
    counts = Counter(result.decision for result in results)
    report = {
        "schema_version": MIXED_ACOUSTIC_GATE_SCHEMA_VERSION,
        "protocol_id": MIXED_ACOUSTIC_GATE_PROTOCOL_ID,
        "packet": {
            "path": str(packet_path),
            "sha256": packet_hash,
            "assets": len(packet),
            "pairs": len(packet) // 2,
        },
        "review_rows": len(reviews),
        "decision_counts": dict(sorted(counts.items())),
        "all_assets_acoustically_verified": counts == Counter({"pass": len(packet)}),
        "final_or_product_eligible": False,
        "limitations": [
            "The gate establishes only audibility and lexical preservation of locked RU/KK "
            "evidence.",
            "It does not establish speaker independence, binary-source independence, calibration, "
            "or final quality.",
        ],
    }
    return report, tuple(results)


def write_mixed_acoustic_gate_report(
    path: Path, report: Mapping[str, object], results: Sequence[MixedAcousticGateAssetResult]
) -> None:
    """Publish one immutable gate receipt, including every asset decision."""

    if path.exists() or not path.parent.is_dir():
        raise MixedAcousticGateError(
            ["Acoustic gate report output must be new with an existing parent."]
        )
    payload = dict(report)
    payload["asset_results"] = [asdict(item) for item in results]
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
    except OSError as error:
        raise MixedAcousticGateError([f"Cannot write acoustic gate report: {error}"]) from error


def load_pair_lock(path: Path) -> dict[str, object]:
    """Load the exact JSON object whose bytes are pinned by a caller or receipt."""

    if not path.is_file():
        raise MixedAcousticGateError([f"Mixed pair lock does not exist: {path}"])
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise MixedAcousticGateError([f"Cannot read mixed pair lock: {error}"]) from error
    if not isinstance(value, dict):
        raise MixedAcousticGateError(["Mixed pair lock must be a JSON object."])
    return cast(dict[str, object], value)


def _validate_pair_lock(
    pair_lock: Mapping[str, object],
    candidate_rows: Sequence[ManifestRow],
    evidence_by_id: Mapping[str, Ksc2MixedEvidenceRow],
    issues: list[str],
) -> None:
    if pair_lock.get("schema_version") != 1 or pair_lock.get("pair_count") != 30:
        issues.append("Mixed pair lock must be schema v1 with exactly 30 pairs.")
    pairs = pair_lock.get("pairs")
    if not isinstance(pairs, list) or len(pairs) != 30:
        issues.append("Mixed pair lock has an invalid pairs list.")
        return
    candidate_by_text: dict[str, dict[str, ManifestRow]] = defaultdict(dict)
    for row in candidate_rows:
        candidate_by_text[row.text_hash][row.label] = row
    locked_ids: set[str] = set()
    for number, raw in enumerate(pairs, start=1):
        if not isinstance(raw, dict):
            issues.append(f"Mixed pair lock pair {number} is not an object.")
            continue
        identifier = raw.get("annotation_id")
        text_hash = raw.get("text_hash")
        if (
            not isinstance(identifier, str)
            or not isinstance(text_hash, str)
            or identifier in locked_ids
        ):
            issues.append(f"Mixed pair lock pair {number} has invalid identity fields.")
            continue
        locked_ids.add(identifier)
        evidence = evidence_by_id.get(identifier)
        paired = candidate_by_text.get(text_hash, {})
        bonafide = paired.get("bonafide")
        spoof = paired.get("spoof")
        if evidence is None or evidence.transcript_sha256 != text_hash:
            issues.append(f"Mixed pair lock pair {number} is not linked to transcript evidence.")
            continue
        if bonafide is None or spoof is None:
            issues.append(f"Mixed pair lock pair {number} lacks one candidate class.")
            continue
        required = {
            "component": evidence.component,
            "ru_evidence_tokens": evidence.ru_evidence_tokens,
            "kk_evidence_tokens": evidence.kk_evidence_tokens,
            "bonafide_audio_sha256": bonafide.sha256,
            "spoof_audio_sha256": spoof.sha256,
        }
        if any(raw.get(key) != value for key, value in required.items()):
            issues.append(f"Mixed pair lock pair {number} differs from candidate/evidence bytes.")
    expected_ids = {
        row.sample_id for row in candidate_rows if row.label == "bonafide"
    }
    if locked_ids != expected_ids:
        issues.append("Mixed pair lock does not cover exactly the candidate bona-fide identities.")


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in _HEX for character in value)
