"""Fail-closed two-review acoustic/language gate for paired VoxForge/Qwen WAVs."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

from kds.data.assets import sha256_file
from kds.data.manifest import ManifestRow
from kds.data.qwen3_tts_customvoice_candidate import QWEN3_TTS_CUSTOMVOICE_AIDEN_SOURCE_ID
from kds.data.voxforge import VOXFORGE_RU_SOURCE_ID

PROTOCOL_ID: Final = "voxforge-ru-mdc-qwen3-tts-customvoice-aiden-pre-qa-acoustic-gate-v1"
PACKET_FIELDS: Final = (
    "protocol_id",
    "text_hash",
    "label",
    "sample_id",
    "relative_path",
    "audio_sha256",
    "input_transcript",
)
REVIEW_FIELDS: Final = (
    "protocol_id",
    "packet_sha256",
    "text_hash",
    "label",
    "sample_id",
    "audio_sha256",
    "reviewer_pseudo_id",
    "review_status",
    "intelligible",
    "russian_audible",
    "lexical_content_preserved",
    "severe_artifacts_absent",
    "notes",
)
_REVIEW_STATUS: Final = frozenset({"pass", "fail", "inconclusive"})
_ANSWER: Final = frozenset({"yes", "no", "unknown"})
_HEX: Final = frozenset("0123456789abcdef")


class VoxForgeQwenAcousticGateError(ValueError):
    """Raised when a review cannot be tied to its exact immutable VoxForge/Qwen bytes."""

    def __init__(self, issues: Iterable[str] | str) -> None:
        self.issues = (issues,) if isinstance(issues, str) else tuple(issues)
        super().__init__("\n".join(self.issues))


@dataclass(frozen=True, slots=True)
class AcousticPacketRow:
    protocol_id: str
    text_hash: str
    label: str
    sample_id: str
    relative_path: str
    audio_sha256: str
    input_transcript: str


@dataclass(frozen=True, slots=True)
class AcousticReview:
    protocol_id: str
    packet_sha256: str
    text_hash: str
    label: str
    sample_id: str
    audio_sha256: str
    reviewer_pseudo_id: str
    review_status: str
    intelligible: str
    russian_audible: str
    lexical_content_preserved: str
    severe_artifacts_absent: str
    notes: str


@dataclass(frozen=True, slots=True)
class AcousticResult:
    text_hash: str
    label: str
    sample_id: str
    audio_sha256: str
    review_count: int
    reviewers: tuple[str, ...]
    decision: str


def build_packet(
    candidate_rows: Sequence[ManifestRow], transcripts: Mapping[str, str]
) -> tuple[AcousticPacketRow, ...]:
    """Bind every retained VoxForge/Qwen WAV to its exact literal archive transcript."""

    if len(candidate_rows) != 158 or Counter(row.label for row in candidate_rows) != Counter(
        {"bonafide": 79, "spoof": 79}
    ):
        raise VoxForgeQwenAcousticGateError(
            "Acoustic gate requires exactly 79 binary pairs / 158 assets."
        )
    if any(
        row.split != "test" or row.language != "ru" or row.codec != "wav"
        for row in candidate_rows
    ):
        raise VoxForgeQwenAcousticGateError(
            "Acoustic candidate must contain only RU test WAV assets."
        )
    by_hash: dict[str, list[ManifestRow]] = defaultdict(list)
    for row in candidate_rows:
        by_hash[row.text_hash].append(row)
    packet: list[AcousticPacketRow] = []
    for text_hash, pair in by_hash.items():
        if len(pair) != 2 or {row.label for row in pair} != {"bonafide", "spoof"}:
            raise VoxForgeQwenAcousticGateError(
                f"Candidate text hash {text_hash!r} is not an exact binary pair."
            )
        bona = next(row for row in pair if row.label == "bonafide")
        spoof = next(row for row in pair if row.label == "spoof")
        if (
            bona.source_name != VOXFORGE_RU_SOURCE_ID
            or spoof.source_name != QWEN3_TTS_CUSTOMVOICE_AIDEN_SOURCE_ID
            or bona.text_id != spoof.text_id
        ):
            raise VoxForgeQwenAcousticGateError(
                f"Candidate pair {text_hash!r} has unexpected source or text binding."
            )
        transcript = transcripts.get(bona.sample_id)
        if not transcript:
            raise VoxForgeQwenAcousticGateError(
                f"Literal transcript is missing for {bona.sample_id!r}."
            )
        for row in pair:
            packet.append(
                AcousticPacketRow(
                    protocol_id=PROTOCOL_ID,
                    text_hash=row.text_hash,
                    label=row.label,
                    sample_id=row.sample_id,
                    relative_path=row.relative_path,
                    audio_sha256=row.sha256,
                    input_transcript=transcript,
                )
            )
    return tuple(sorted(packet, key=lambda row: (row.text_hash, row.label)))


def write_packet(path: Path, rows: Sequence[AcousticPacketRow]) -> None:
    if path.exists() or not path.parent.is_dir() or len(rows) != 158:
        raise VoxForgeQwenAcousticGateError(
            "Acoustic packet output must be new and contain 158 rows."
        )
    _write_csv(path, PACKET_FIELDS, (asdict(row) for row in rows))


def read_packet(path: Path) -> tuple[AcousticPacketRow, ...]:
    packet = tuple(AcousticPacketRow(**row) for row in _read_csv(path, PACKET_FIELDS))
    if len(packet) != 158 or len({row.sample_id for row in packet}) != 158:
        raise VoxForgeQwenAcousticGateError(
            "Acoustic packet must contain 158 unique assets."
        )
    if (
        Counter(row.label for row in packet) != Counter({"bonafide": 79, "spoof": 79})
        or any(
            row.protocol_id != PROTOCOL_ID
            or not row.input_transcript
            or not _is_sha256(row.text_hash)
            or not _is_sha256(row.audio_sha256)
            for row in packet
        )
    ):
        raise VoxForgeQwenAcousticGateError("Acoustic packet has invalid static evidence.")
    return packet


def write_review_template(path: Path, packet_path: Path, reviewer_pseudo_id: str) -> None:
    reviewer = reviewer_pseudo_id.strip()
    if not reviewer or any(character in reviewer for character in ",\r\n"):
        raise VoxForgeQwenAcousticGateError(
            "reviewer_pseudo_id must be non-empty and single-line."
        )
    packet = read_packet(packet_path)
    if path.exists() or not path.parent.is_dir():
        raise VoxForgeQwenAcousticGateError("Acoustic review output must be new.")
    packet_sha256 = sha256_file(packet_path)
    _write_csv(
        path,
        REVIEW_FIELDS,
        (
            {
                "protocol_id": PROTOCOL_ID,
                "packet_sha256": packet_sha256,
                "text_hash": row.text_hash,
                "label": row.label,
                "sample_id": row.sample_id,
                "audio_sha256": row.audio_sha256,
                "reviewer_pseudo_id": reviewer,
                "review_status": "inconclusive",
                "intelligible": "unknown",
                "russian_audible": "unknown",
                "lexical_content_preserved": "unknown",
                "severe_artifacts_absent": "unknown",
                "notes": "",
            }
            for row in packet
        ),
    )


def read_reviews(path: Path) -> tuple[AcousticReview, ...]:
    reviews = tuple(AcousticReview(**row) for row in _read_csv(path, REVIEW_FIELDS))
    if not reviews:
        raise VoxForgeQwenAcousticGateError("Acoustic review CSV has no rows.")
    return reviews


def evaluate(
    packet_path: Path, reviews: Sequence[AcousticReview]
) -> tuple[dict[str, object], tuple[AcousticResult, ...]]:
    """Pass only when two distinct reviewers independently pass every immutable asset."""

    packet = read_packet(packet_path)
    packet_sha256 = sha256_file(packet_path)
    by_sample = {row.sample_id: row for row in packet}
    decisions_by_sample: dict[str, list[AcousticReview]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    issues: list[str] = []
    for number, review in enumerate(reviews, start=2):
        bound = by_sample.get(review.sample_id)
        if (
            bound is None
            or review.protocol_id != PROTOCOL_ID
            or review.packet_sha256 != packet_sha256
            or review.text_hash != bound.text_hash
            or review.label != bound.label
            or review.audio_sha256 != bound.audio_sha256
            or not review.reviewer_pseudo_id
            or review.review_status not in _REVIEW_STATUS
            or review.intelligible not in _ANSWER
            or review.russian_audible not in _ANSWER
            or review.lexical_content_preserved not in _ANSWER
            or review.severe_artifacts_absent not in _ANSWER
        ):
            issues.append(f"Review row {number} does not match the immutable acoustic packet.")
            continue
        key = (review.sample_id, review.reviewer_pseudo_id)
        if key in seen:
            issues.append(f"Reviewer duplicates an asset at review row {number}.")
            continue
        seen.add(key)
        decisions_by_sample[review.sample_id].append(review)
    if issues:
        raise VoxForgeQwenAcousticGateError(issues)
    results: list[AcousticResult] = []
    for item in packet:
        asset_reviews = decisions_by_sample[item.sample_id]
        reviewers = tuple(sorted(review.reviewer_pseudo_id for review in asset_reviews))
        passed = (
            len(asset_reviews) == 2
            and len(set(reviewers)) == 2
            and all(
                review.review_status == "pass"
                and review.intelligible == "yes"
                and review.russian_audible == "yes"
                and review.lexical_content_preserved == "yes"
                and review.severe_artifacts_absent == "yes"
                for review in asset_reviews
            )
        )
        results.append(
            AcousticResult(
                text_hash=item.text_hash,
                label=item.label,
                sample_id=item.sample_id,
                audio_sha256=item.audio_sha256,
                review_count=len(asset_reviews),
                reviewers=reviewers,
                decision="pass" if passed else "not_eligible",
            )
        )
    counts = Counter(result.decision for result in results)
    all_passed = counts == Counter({"pass": 158})
    report = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "packet": {"path": str(packet_path), "sha256": packet_sha256, "assets": 158, "pairs": 79},
        "review_rows": len(reviews),
        "decision_counts": dict(sorted(counts.items())),
        "all_assets_acoustically_verified": all_passed,
        "evaluation_contract_authorized": all_passed,
        "detector_inference_performed": False,
        "limitations": [
            "The gate confirms only intelligibility, Russian audibility, literal-content "
            "preservation and absence of severe artifacts for these pinned WAVs.",
            "Distinct reviewer pseudo-IDs are a technical minimum, not proof of organizational "
            "independence, architecture-family novelty, calibration or product quality.",
        ],
    }
    return report, tuple(results)


def write_report(
    path: Path, report: Mapping[str, object], results: Sequence[AcousticResult]
) -> None:
    if path.exists() or not path.parent.is_dir():
        raise VoxForgeQwenAcousticGateError("Acoustic report output must be new.")
    payload = dict(report)
    payload["asset_results"] = [asdict(result) for result in results]
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
    except OSError as error:
        raise VoxForgeQwenAcousticGateError(f"Cannot write acoustic report: {error}") from error


def _read_csv(path: Path, fields: tuple[str, ...]) -> tuple[dict[str, str], ...]:
    if not path.is_file():
        raise VoxForgeQwenAcousticGateError(f"Acoustic CSV does not exist: {path}")
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or tuple(reader.fieldnames) != fields:
                raise VoxForgeQwenAcousticGateError("Acoustic CSV has an invalid schema.")
            return tuple(
                {field: (row.get(field) or "").strip() for field in fields} for row in reader
            )
    except OSError as error:
        raise VoxForgeQwenAcousticGateError(f"Cannot read acoustic CSV: {error}") from error


def _write_csv(path: Path, fields: tuple[str, ...], rows: Iterable[Mapping[str, object]]) -> None:
    try:
        with path.open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    except OSError as error:
        raise VoxForgeQwenAcousticGateError(f"Cannot write acoustic CSV: {error}") from error


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in _HEX for character in value)
