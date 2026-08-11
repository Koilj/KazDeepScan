"""Fail-closed auditory Russian-language gate for the frozen FLEURS/eSpeak candidate."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

from kds.data.assets import sha256_file
from kds.data.fleurs_espeakng import FLEURS_RU_ESPEAKNG_SOURCE_ID, FLEURS_RU_SOURCE_ID
from kds.data.manifest import ManifestRow

FLEURS_RU_ACOUSTIC_GATE_PROTOCOL_ID: Final = "fleurs-ru-acoustic-language-preservation-v1"
FLEURS_RU_ACOUSTIC_GATE_PACKET_FIELDS: Final = (
    "protocol_id",
    "text_hash",
    "label",
    "sample_id",
    "relative_path",
    "audio_sha256",
    "input_transcript",
)
FLEURS_RU_ACOUSTIC_GATE_REVIEW_FIELDS: Final = (
    "protocol_id",
    "packet_sha256",
    "text_hash",
    "label",
    "sample_id",
    "audio_sha256",
    "reviewer_pseudo_id",
    "review_status",
    "russian_audible",
    "lexical_content_preserved",
    "notes",
)
_REVIEW_STATUS = frozenset({"pass", "fail", "inconclusive"})
_ANSWER = frozenset({"yes", "no", "unknown"})
_HEX = frozenset("0123456789abcdef")


class FleursRuAcousticGateError(ValueError):
    """Raised when a reviewer packet or decision cannot be tied to its exact candidate bytes."""

    def __init__(self, issues: Iterable[str] | str) -> None:
        self.issues = (issues,) if isinstance(issues, str) else tuple(issues)
        super().__init__("\n".join(self.issues))


@dataclass(frozen=True, slots=True)
class FleursRuAcousticPacketRow:
    protocol_id: str
    text_hash: str
    label: str
    sample_id: str
    relative_path: str
    audio_sha256: str
    input_transcript: str


@dataclass(frozen=True, slots=True)
class FleursRuAcousticReview:
    protocol_id: str
    packet_sha256: str
    text_hash: str
    label: str
    sample_id: str
    audio_sha256: str
    reviewer_pseudo_id: str
    review_status: str
    russian_audible: str
    lexical_content_preserved: str
    notes: str


@dataclass(frozen=True, slots=True)
class FleursRuAcousticResult:
    text_hash: str
    label: str
    sample_id: str
    audio_sha256: str
    review_count: int
    reviewers: tuple[str, ...]
    decision: str


def build_fleurs_ru_acoustic_packet(
    candidate_rows: Sequence[ManifestRow], transcripts: Mapping[str, str]
) -> tuple[FleursRuAcousticPacketRow, ...]:
    """Bind exactly one FLEURS transcript to both assets in each 75-pair candidate."""

    if len(candidate_rows) != 150 or Counter(row.label for row in candidate_rows) != Counter(
        {"bonafide": 75, "spoof": 75}
    ):
        raise FleursRuAcousticGateError("RU acoustic gate requires exactly 75 binary pairs.")
    if any(
        row.split != "test" or row.language != "ru" or row.code_switch != "false"
        for row in candidate_rows
    ):
        raise FleursRuAcousticGateError(
            "RU acoustic gate accepts only non-code-switched RU test rows."
        )
    by_text: dict[str, list[ManifestRow]] = defaultdict(list)
    for row in candidate_rows:
        by_text[row.text_hash].append(row)
    packet: list[FleursRuAcousticPacketRow] = []
    for text_hash, pair in by_text.items():
        if len(pair) != 2 or {row.label for row in pair} != {"bonafide", "spoof"}:
            raise FleursRuAcousticGateError(
                f"Candidate text hash {text_hash} is not one exact pair."
            )
        bona = next(row for row in pair if row.label == "bonafide")
        spoof = next(row for row in pair if row.label == "spoof")
        if (
            bona.source_name != FLEURS_RU_SOURCE_ID
            or spoof.source_name != FLEURS_RU_ESPEAKNG_SOURCE_ID
            or bona.text_id != spoof.text_id
        ):
            raise FleursRuAcousticGateError(
                f"Candidate pair {text_hash} does not bind FLEURS RU to the eSpeak RU source."
            )
        transcript = transcripts.get(bona.sample_id)
        if not transcript or bona.text_hash != text_hash:
            raise FleursRuAcousticGateError(
                f"FLEURS transcript is missing or unpinned for {bona.sample_id!r}."
            )
        for row in pair:
            packet.append(
                FleursRuAcousticPacketRow(
                    protocol_id=FLEURS_RU_ACOUSTIC_GATE_PROTOCOL_ID,
                    text_hash=text_hash,
                    label=row.label,
                    sample_id=row.sample_id,
                    relative_path=row.relative_path,
                    audio_sha256=row.sha256,
                    input_transcript=transcript,
                )
            )
    return tuple(sorted(packet, key=lambda item: (item.text_hash, item.label)))


def write_fleurs_ru_acoustic_packet(path: Path, rows: Sequence[FleursRuAcousticPacketRow]) -> None:
    if path.exists() or not path.parent.is_dir() or len(rows) != 150:
        raise FleursRuAcousticGateError(
            "RU acoustic packet output must be new and contain 150 rows."
        )
    _write_csv(path, FLEURS_RU_ACOUSTIC_GATE_PACKET_FIELDS, (asdict(row) for row in rows))


def read_fleurs_ru_acoustic_packet(path: Path) -> tuple[FleursRuAcousticPacketRow, ...]:
    packet = _read_packet_csv(path)
    if len(packet) != 150 or len({row.sample_id for row in packet}) != 150:
        raise FleursRuAcousticGateError("RU acoustic packet must have 150 unique assets.")
    if any(
        row.protocol_id != FLEURS_RU_ACOUSTIC_GATE_PROTOCOL_ID
        or row.label not in {"bonafide", "spoof"}
        or not row.input_transcript
        or not _sha256(row.text_hash)
        or not _sha256(row.audio_sha256)
        for row in packet
    ):
        raise FleursRuAcousticGateError("RU acoustic packet has invalid static evidence.")
    return packet


def write_fleurs_ru_acoustic_review_template(
    path: Path, packet_path: Path, reviewer_pseudo_id: str
) -> None:
    reviewer = reviewer_pseudo_id.strip()
    if not reviewer or any(character in reviewer for character in ",\r\n"):
        raise FleursRuAcousticGateError("reviewer_pseudo_id must be non-empty and single-line.")
    packet = read_fleurs_ru_acoustic_packet(packet_path)
    packet_hash = sha256_file(packet_path)
    if path.exists() or not path.parent.is_dir():
        raise FleursRuAcousticGateError("RU acoustic review output must be new.")
    _write_csv(
        path,
        FLEURS_RU_ACOUSTIC_GATE_REVIEW_FIELDS,
        (
            {
                "protocol_id": FLEURS_RU_ACOUSTIC_GATE_PROTOCOL_ID,
                "packet_sha256": packet_hash,
                "text_hash": row.text_hash,
                "label": row.label,
                "sample_id": row.sample_id,
                "audio_sha256": row.audio_sha256,
                "reviewer_pseudo_id": reviewer,
                "review_status": "inconclusive",
                "russian_audible": "unknown",
                "lexical_content_preserved": "unknown",
                "notes": "",
            }
            for row in packet
        ),
    )


def read_fleurs_ru_acoustic_reviews(path: Path) -> tuple[FleursRuAcousticReview, ...]:
    reviews = _read_review_csv(path)
    if not reviews:
        raise FleursRuAcousticGateError("RU acoustic review CSV has no rows.")
    return reviews


def evaluate_fleurs_ru_acoustic_gate(
    packet_path: Path, reviews: Sequence[FleursRuAcousticReview]
) -> tuple[dict[str, object], tuple[FleursRuAcousticResult, ...]]:
    packet = read_fleurs_ru_acoustic_packet(packet_path)
    packet_hash = sha256_file(packet_path)
    by_id = {row.sample_id: row for row in packet}
    review_by_id: dict[str, list[FleursRuAcousticReview]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    issues: list[str] = []
    for number, review in enumerate(reviews, start=2):
        item = by_id.get(review.sample_id)
        if (
            item is None
            or review.protocol_id != FLEURS_RU_ACOUSTIC_GATE_PROTOCOL_ID
            or review.packet_sha256 != packet_hash
            or review.text_hash != item.text_hash
            or review.label != item.label
            or review.audio_sha256 != item.audio_sha256
            or not review.reviewer_pseudo_id
            or review.review_status not in _REVIEW_STATUS
            or review.russian_audible not in _ANSWER
            or review.lexical_content_preserved not in _ANSWER
        ):
            issues.append(f"Review row {number} does not match the immutable RU packet.")
            continue
        key = (review.sample_id, review.reviewer_pseudo_id)
        if key in seen:
            issues.append(f"Reviewer duplicated asset at review row {number}.")
            continue
        seen.add(key)
        review_by_id[review.sample_id].append(review)
    if issues:
        raise FleursRuAcousticGateError(issues)
    results: list[FleursRuAcousticResult] = []
    for item in packet:
        decisions = review_by_id[item.sample_id]
        reviewers = tuple(sorted(review.reviewer_pseudo_id for review in decisions))
        passed = (
            len(decisions) == 2
            and len(set(reviewers)) == 2
            and all(
                review.review_status == "pass"
                and review.russian_audible == "yes"
                and review.lexical_content_preserved == "yes"
                for review in decisions
            )
        )
        results.append(
            FleursRuAcousticResult(
                text_hash=item.text_hash,
                label=item.label,
                sample_id=item.sample_id,
                audio_sha256=item.audio_sha256,
                review_count=len(decisions),
                reviewers=reviewers,
                decision="pass" if passed else "not_eligible",
            )
        )
    counts = Counter(result.decision for result in results)
    report = {
        "schema_version": 1,
        "protocol_id": FLEURS_RU_ACOUSTIC_GATE_PROTOCOL_ID,
        "packet": {"path": str(packet_path), "sha256": packet_hash, "assets": 150, "pairs": 75},
        "review_rows": len(reviews),
        "decision_counts": dict(sorted(counts.items())),
        "all_assets_acoustically_verified": counts == Counter({"pass": 150}),
        "final_or_product_eligible": False,
        "limitations": [
            "The gate establishes only Russian audibility and lexical preservation for locked "
            "assets.",
            "It does not establish speaker independence, external-source independence, "
            "calibration, or final quality.",
        ],
    }
    return report, tuple(results)


def write_fleurs_ru_acoustic_report(
    path: Path, report: Mapping[str, object], results: Sequence[FleursRuAcousticResult]
) -> None:
    if path.exists() or not path.parent.is_dir():
        raise FleursRuAcousticGateError("RU acoustic report output must be new.")
    payload = dict(report)
    payload["asset_results"] = [asdict(result) for result in results]
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
    except OSError as error:
        raise FleursRuAcousticGateError(f"Cannot write RU acoustic report: {error}") from error


def _read_packet_csv(path: Path) -> tuple[FleursRuAcousticPacketRow, ...]:
    raw_rows = _read_csv_rows(path, FLEURS_RU_ACOUSTIC_GATE_PACKET_FIELDS)
    return tuple(FleursRuAcousticPacketRow(**row) for row in raw_rows)


def _read_review_csv(path: Path) -> tuple[FleursRuAcousticReview, ...]:
    raw_rows = _read_csv_rows(path, FLEURS_RU_ACOUSTIC_GATE_REVIEW_FIELDS)
    return tuple(FleursRuAcousticReview(**row) for row in raw_rows)


def _read_csv_rows(path: Path, fields: tuple[str, ...]) -> tuple[dict[str, str], ...]:
    if not path.is_file():
        raise FleursRuAcousticGateError(f"Acoustic CSV does not exist: {path}")
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or tuple(reader.fieldnames) != fields:
                raise FleursRuAcousticGateError("Acoustic CSV has an invalid schema.")
            return tuple(
                {field: (raw.get(field) or "").strip() for field in fields} for raw in reader
            )
    except OSError as error:
        raise FleursRuAcousticGateError(f"Cannot read acoustic CSV: {error}") from error


def _write_csv(path: Path, fields: tuple[str, ...], rows: Iterable[Mapping[str, object]]) -> None:
    try:
        with path.open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    except OSError as error:
        raise FleursRuAcousticGateError(f"Cannot write acoustic CSV: {error}") from error


def _sha256(value: str) -> bool:
    return len(value) == 64 and all(character in _HEX for character in value)
