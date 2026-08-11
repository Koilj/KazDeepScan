"""Fail-closed auditory Russian-language gate for the locked ToneSpeak OOD candidate."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

from kds.data.assets import sha256_file
from kds.data.manifest import ManifestRow
from kds.data.tone_speak import TONE_SPEAK_SOURCE_ID

TONE_SPEAK_ACOUSTIC_GATE_PROTOCOL_ID: Final = "tone-speak-ru-acoustic-language-preservation-v1"
TONE_SPEAK_ACOUSTIC_GATE_PACKET_FIELDS: Final = (
    "protocol_id",
    "text_hash",
    "sample_id",
    "relative_path",
    "audio_sha256",
    "input_transcript",
)
TONE_SPEAK_ACOUSTIC_GATE_REVIEW_FIELDS: Final = (
    "protocol_id",
    "packet_sha256",
    "text_hash",
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


class ToneSpeakAcousticGateError(ValueError):
    """Raised when a reviewer packet or decision is not bound to exact ToneSpeak WAV bytes."""

    def __init__(self, issues: Iterable[str] | str) -> None:
        self.issues = (issues,) if isinstance(issues, str) else tuple(issues)
        super().__init__("\n".join(self.issues))


@dataclass(frozen=True, slots=True)
class ToneSpeakAcousticPacketRow:
    protocol_id: str
    text_hash: str
    sample_id: str
    relative_path: str
    audio_sha256: str
    input_transcript: str


@dataclass(frozen=True, slots=True)
class ToneSpeakAcousticReview:
    protocol_id: str
    packet_sha256: str
    text_hash: str
    sample_id: str
    audio_sha256: str
    reviewer_pseudo_id: str
    review_status: str
    russian_audible: str
    lexical_content_preserved: str
    notes: str


@dataclass(frozen=True, slots=True)
class ToneSpeakAcousticResult:
    text_hash: str
    sample_id: str
    audio_sha256: str
    review_count: int
    reviewers: tuple[str, ...]
    decision: str


def build_tone_speak_acoustic_packet(
    candidate_rows: Sequence[ManifestRow], transcripts: Mapping[str, str]
) -> tuple[ToneSpeakAcousticPacketRow, ...]:
    """Bind the 100 ready source WAVs to their revalidated source transcripts."""

    if len(candidate_rows) != 100 or any(
        row.split != "ood"
        or row.label != "spoof"
        or row.language != "ru"
        or row.code_switch != "false"
        or row.source_name != TONE_SPEAK_SOURCE_ID
        or row.codec != "wav"
        for row in candidate_rows
    ):
        raise ToneSpeakAcousticGateError(
            "ToneSpeak acoustic gate requires exactly 100 ready Russian spoof OOD WAVs."
        )
    if len({row.sample_id for row in candidate_rows}) != 100:
        raise ToneSpeakAcousticGateError("ToneSpeak acoustic candidate has duplicate sample IDs.")
    if len({row.text_hash for row in candidate_rows}) != 100:
        raise ToneSpeakAcousticGateError("ToneSpeak acoustic candidate has duplicate text groups.")
    packet: list[ToneSpeakAcousticPacketRow] = []
    for row in candidate_rows:
        transcript = transcripts.get(row.sample_id)
        if not transcript or not _sha256(row.text_hash) or row.text_hash != _text_hash(transcript):
            raise ToneSpeakAcousticGateError(
                f"ToneSpeak source transcript is missing or mismatched for {row.sample_id!r}."
            )
        packet.append(
            ToneSpeakAcousticPacketRow(
                protocol_id=TONE_SPEAK_ACOUSTIC_GATE_PROTOCOL_ID,
                text_hash=row.text_hash,
                sample_id=row.sample_id,
                relative_path=row.relative_path,
                audio_sha256=row.sha256,
                input_transcript=transcript,
            )
        )
    return tuple(sorted(packet, key=lambda item: item.sample_id))


def write_tone_speak_acoustic_packet(
    path: Path, rows: Sequence[ToneSpeakAcousticPacketRow]
) -> None:
    if path.exists() or not path.parent.is_dir() or len(rows) != 100:
        raise ToneSpeakAcousticGateError(
            "ToneSpeak acoustic packet output must be new and contain 100 rows."
        )
    _write_csv(path, TONE_SPEAK_ACOUSTIC_GATE_PACKET_FIELDS, (asdict(row) for row in rows))


def read_tone_speak_acoustic_packet(path: Path) -> tuple[ToneSpeakAcousticPacketRow, ...]:
    packet = tuple(
        ToneSpeakAcousticPacketRow(**row)
        for row in _read_csv_rows(path, TONE_SPEAK_ACOUSTIC_GATE_PACKET_FIELDS)
    )
    if len(packet) != 100 or len({row.sample_id for row in packet}) != 100:
        raise ToneSpeakAcousticGateError("ToneSpeak acoustic packet must have 100 unique assets.")
    if any(
        row.protocol_id != TONE_SPEAK_ACOUSTIC_GATE_PROTOCOL_ID
        or not row.input_transcript
        or not _sha256(row.text_hash)
        or not _sha256(row.audio_sha256)
        for row in packet
    ):
        raise ToneSpeakAcousticGateError("ToneSpeak acoustic packet has invalid static evidence.")
    return packet


def write_tone_speak_acoustic_review_template(
    path: Path, packet_path: Path, reviewer_pseudo_id: str
) -> None:
    reviewer = reviewer_pseudo_id.strip()
    if not reviewer or any(character in reviewer for character in ",\r\n"):
        raise ToneSpeakAcousticGateError("reviewer_pseudo_id must be non-empty and single-line.")
    packet = read_tone_speak_acoustic_packet(packet_path)
    packet_hash = sha256_file(packet_path)
    if path.exists() or not path.parent.is_dir():
        raise ToneSpeakAcousticGateError("ToneSpeak acoustic review output must be new.")
    _write_csv(
        path,
        TONE_SPEAK_ACOUSTIC_GATE_REVIEW_FIELDS,
        (
            {
                "protocol_id": TONE_SPEAK_ACOUSTIC_GATE_PROTOCOL_ID,
                "packet_sha256": packet_hash,
                "text_hash": row.text_hash,
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


def read_tone_speak_acoustic_reviews(path: Path) -> tuple[ToneSpeakAcousticReview, ...]:
    reviews = tuple(
        ToneSpeakAcousticReview(**row)
        for row in _read_csv_rows(path, TONE_SPEAK_ACOUSTIC_GATE_REVIEW_FIELDS)
    )
    if not reviews:
        raise ToneSpeakAcousticGateError("ToneSpeak acoustic review CSV has no rows.")
    return reviews


def evaluate_tone_speak_acoustic_gate(
    packet_path: Path, reviews: Sequence[ToneSpeakAcousticReview]
) -> tuple[dict[str, object], tuple[ToneSpeakAcousticResult, ...]]:
    packet = read_tone_speak_acoustic_packet(packet_path)
    packet_hash = sha256_file(packet_path)
    by_id = {row.sample_id: row for row in packet}
    reviews_by_id: dict[str, list[ToneSpeakAcousticReview]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    issues: list[str] = []
    for number, review in enumerate(reviews, start=2):
        item = by_id.get(review.sample_id)
        if (
            item is None
            or review.protocol_id != TONE_SPEAK_ACOUSTIC_GATE_PROTOCOL_ID
            or review.packet_sha256 != packet_hash
            or review.text_hash != item.text_hash
            or review.audio_sha256 != item.audio_sha256
            or not review.reviewer_pseudo_id
            or review.review_status not in _REVIEW_STATUS
            or review.russian_audible not in _ANSWER
            or review.lexical_content_preserved not in _ANSWER
        ):
            issues.append(f"Review row {number} does not match the immutable ToneSpeak packet.")
            continue
        key = (review.sample_id, review.reviewer_pseudo_id)
        if key in seen:
            issues.append(f"Reviewer duplicated asset at review row {number}.")
            continue
        seen.add(key)
        reviews_by_id[review.sample_id].append(review)
    if issues:
        raise ToneSpeakAcousticGateError(issues)
    results: list[ToneSpeakAcousticResult] = []
    for item in packet:
        decisions = reviews_by_id[item.sample_id]
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
            ToneSpeakAcousticResult(
                text_hash=item.text_hash,
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
        "protocol_id": TONE_SPEAK_ACOUSTIC_GATE_PROTOCOL_ID,
        "packet": {"path": str(packet_path), "sha256": packet_hash, "assets": 100},
        "review_rows": len(reviews),
        "decision_counts": dict(sorted(counts.items())),
        "all_assets_acoustically_verified": counts == Counter({"pass": 100}),
        "final_or_product_eligible": False,
        "limitations": [
            "The gate establishes only Russian audibility and lexical preservation for locked "
            "assets.",
            "It does not establish generator logging, spoof-voice-group provenance, calibration, "
            "binary final quality, or product eligibility.",
        ],
    }
    return report, tuple(results)


def write_tone_speak_acoustic_report(
    path: Path, report: Mapping[str, object], results: Sequence[ToneSpeakAcousticResult]
) -> None:
    if path.exists() or not path.parent.is_dir():
        raise ToneSpeakAcousticGateError("ToneSpeak acoustic report output must be new.")
    payload = dict(report)
    payload["asset_results"] = [asdict(result) for result in results]
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
    except OSError as error:
        raise ToneSpeakAcousticGateError(
            f"Cannot write ToneSpeak acoustic report: {error}"
        ) from error


def _read_csv_rows(path: Path, fields: tuple[str, ...]) -> tuple[dict[str, str], ...]:
    if not path.is_file():
        raise ToneSpeakAcousticGateError(f"ToneSpeak acoustic CSV does not exist: {path}")
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or tuple(reader.fieldnames) != fields:
                raise ToneSpeakAcousticGateError("ToneSpeak acoustic CSV has an invalid schema.")
            return tuple(
                {field: (raw.get(field) or "").strip() for field in fields} for raw in reader
            )
    except OSError as error:
        raise ToneSpeakAcousticGateError(f"Cannot read ToneSpeak acoustic CSV: {error}") from error


def _write_csv(path: Path, fields: tuple[str, ...], rows: Iterable[Mapping[str, object]]) -> None:
    try:
        with path.open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    except OSError as error:
        raise ToneSpeakAcousticGateError(f"Cannot write ToneSpeak acoustic CSV: {error}") from error


def _text_hash(value: str) -> str:
    return hashlib.sha256(" ".join(value.split()).encode("utf-8")).hexdigest()


def _sha256(value: str) -> bool:
    return len(value) == 64 and all(character in _HEX for character in value)
