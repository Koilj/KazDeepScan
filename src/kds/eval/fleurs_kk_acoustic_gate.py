"""Fail-closed two-review acoustic gate for the frozen FLEURS KK/Silero layer.

The detector result for this layer is already known.  This module therefore handles only
human acoustic evidence for the exact candidate bytes and never reads predictions or logits.
"""

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
from kds.data.silero_v4 import SILERO_V4_SOURCE_ID

FLEURS_KK_SOURCE_ID: Final = "google_fleurs_kk_v1"
FLEURS_KK_ACOUSTIC_GATE_SCHEMA_VERSION: Final = 1
FLEURS_KK_ACOUSTIC_GATE_PROTOCOL_ID: Final = "fleurs-kk-silero-v4-acoustic-quality-v1"
FLEURS_KK_ACOUSTIC_GATE_PACKET_FIELDS: Final = (
    "protocol_id",
    "text_hash",
    "label",
    "sample_id",
    "relative_path",
    "audio_sha256",
    "input_transcript",
)
FLEURS_KK_ACOUSTIC_GATE_REVIEW_FIELDS: Final = (
    "protocol_id",
    "packet_sha256",
    "text_hash",
    "label",
    "sample_id",
    "audio_sha256",
    "relative_path",
    "input_transcript",
    "reviewer_pseudo_id",
    "review_status",
    "audio_audible",
    "kazakh_text_matches",
    "no_obvious_defects",
    "notes",
)
_EXPECTED_ASSETS: Final = 304
_EXPECTED_PAIRS: Final = 152
_EXPECTED_REVIEWERS: Final = 2
_REVIEW_STATUSES: Final = frozenset({"pass", "fail", "inconclusive"})
_ANSWERS: Final = frozenset({"yes", "no", "unknown"})
_HEX: Final = frozenset("0123456789abcdef")


class FleursKkAcousticGateError(ValueError):
    """Raised when evidence cannot be tied strictly to the frozen KK assets."""

    def __init__(self, issues: Iterable[str] | str) -> None:
        self.issues = (issues,) if isinstance(issues, str) else tuple(issues)
        super().__init__("\n".join(self.issues))


@dataclass(frozen=True, slots=True)
class FleursKkAcousticPacketRow:
    protocol_id: str
    text_hash: str
    label: str
    sample_id: str
    relative_path: str
    audio_sha256: str
    input_transcript: str


@dataclass(frozen=True, slots=True)
class FleursKkAcousticReview:
    protocol_id: str
    packet_sha256: str
    text_hash: str
    label: str
    sample_id: str
    audio_sha256: str
    relative_path: str
    input_transcript: str
    reviewer_pseudo_id: str
    review_status: str
    audio_audible: str
    kazakh_text_matches: str
    no_obvious_defects: str
    notes: str


@dataclass(frozen=True, slots=True)
class FleursKkAcousticResult:
    text_hash: str
    label: str
    sample_id: str
    audio_sha256: str
    review_count: int
    reviewers: tuple[str, ...]
    decision: str


def build_fleurs_kk_acoustic_packet(
    candidate_rows: Sequence[ManifestRow], transcripts: Mapping[str, str]
) -> tuple[FleursKkAcousticPacketRow, ...]:
    """Bind the pinned FLEURS transcript to both WAVs in every exact KK pair."""

    if len(candidate_rows) != _EXPECTED_ASSETS or Counter(
        row.label for row in candidate_rows
    ) != Counter({"bonafide": _EXPECTED_PAIRS, "spoof": _EXPECTED_PAIRS}):
        raise FleursKkAcousticGateError(
            f"KK acoustic gate requires exactly {_EXPECTED_PAIRS} binary pairs."
        )
    if any(
        row.split != "test" or row.language != "kk" or row.code_switch != "false"
        for row in candidate_rows
    ):
        raise FleursKkAcousticGateError(
            "KK acoustic gate accepts only non-code-switched Kazakh test rows."
        )

    by_text: dict[str, list[ManifestRow]] = defaultdict(list)
    for row in candidate_rows:
        by_text[row.text_hash].append(row)
    if len(by_text) != _EXPECTED_PAIRS:
        raise FleursKkAcousticGateError(
            f"KK acoustic gate requires {_EXPECTED_PAIRS} unique text pairs."
        )

    packet: list[FleursKkAcousticPacketRow] = []
    for text_hash, pair in by_text.items():
        if len(pair) != 2 or {row.label for row in pair} != {"bonafide", "spoof"}:
            raise FleursKkAcousticGateError(
                f"Candidate text hash {text_hash} is not one exact binary pair."
            )
        bonafide = next(row for row in pair if row.label == "bonafide")
        spoof = next(row for row in pair if row.label == "spoof")
        if (
            bonafide.source_name != FLEURS_KK_SOURCE_ID
            or spoof.source_name != SILERO_V4_SOURCE_ID
            or bonafide.text_id != spoof.text_id
        ):
            raise FleursKkAcousticGateError(
                f"Candidate pair {text_hash} does not bind FLEURS KK to Silero V4."
            )
        transcript = transcripts.get(bonafide.sample_id, "").strip()
        if not transcript or _text_sha256(transcript) != text_hash:
            raise FleursKkAcousticGateError(
                f"FLEURS transcript is missing or does not match {bonafide.sample_id!r}."
            )
        for row in pair:
            packet.append(
                FleursKkAcousticPacketRow(
                    protocol_id=FLEURS_KK_ACOUSTIC_GATE_PROTOCOL_ID,
                    text_hash=text_hash,
                    label=row.label,
                    sample_id=row.sample_id,
                    relative_path=row.relative_path,
                    audio_sha256=row.sha256,
                    input_transcript=transcript,
                )
            )
    return tuple(sorted(packet, key=lambda item: (item.text_hash, item.label, item.sample_id)))


def write_fleurs_kk_acoustic_packet(
    path: Path, rows: Sequence[FleursKkAcousticPacketRow]
) -> None:
    """Publish a packet once; reviewer decisions are never stored in this file."""

    if path.exists() or not path.parent.is_dir() or len(rows) != _EXPECTED_ASSETS:
        raise FleursKkAcousticGateError(
            f"KK acoustic packet output must be new and contain {_EXPECTED_ASSETS} rows."
        )
    _write_csv(path, FLEURS_KK_ACOUSTIC_GATE_PACKET_FIELDS, (asdict(row) for row in rows))


def read_fleurs_kk_acoustic_packet(path: Path) -> tuple[FleursKkAcousticPacketRow, ...]:
    raw_rows = _read_csv_rows(path, FLEURS_KK_ACOUSTIC_GATE_PACKET_FIELDS)
    packet = tuple(FleursKkAcousticPacketRow(**row) for row in raw_rows)
    if len(packet) != _EXPECTED_ASSETS or len({row.sample_id for row in packet}) != len(packet):
        raise FleursKkAcousticGateError(
            f"KK acoustic packet must have {_EXPECTED_ASSETS} unique assets."
        )
    by_text: dict[str, list[FleursKkAcousticPacketRow]] = defaultdict(list)
    for row in packet:
        by_text[row.text_hash].append(row)
    if len(by_text) != _EXPECTED_PAIRS or any(
        len(pair) != 2
        or {row.label for row in pair} != {"bonafide", "spoof"}
        or len({row.input_transcript for row in pair}) != 1
        for pair in by_text.values()
    ):
        raise FleursKkAcousticGateError("KK acoustic packet does not contain exact text pairs.")
    if any(
        row.protocol_id != FLEURS_KK_ACOUSTIC_GATE_PROTOCOL_ID
        or row.label not in {"bonafide", "spoof"}
        or not row.relative_path.startswith("processed/")
        or not _sha256(row.text_hash)
        or not _sha256(row.audio_sha256)
        or not row.input_transcript
        or _text_sha256(row.input_transcript) != row.text_hash
        for row in packet
    ):
        raise FleursKkAcousticGateError("KK acoustic packet contains invalid static evidence.")
    return packet


def write_fleurs_kk_acoustic_review_template(
    path: Path, packet_path: Path, reviewer_pseudo_id: str
) -> None:
    """Create one new worksheet with explicit fail-closed defaults."""

    reviewer = reviewer_pseudo_id.strip()
    if not reviewer or any(character in reviewer for character in ",\r\n"):
        raise FleursKkAcousticGateError(
            "reviewer_pseudo_id must be non-empty and single-line."
        )
    packet = read_fleurs_kk_acoustic_packet(packet_path)
    packet_hash = sha256_file(packet_path)
    if path.exists() or not path.parent.is_dir():
        raise FleursKkAcousticGateError("KK acoustic review output must be new.")
    _write_csv(
        path,
        FLEURS_KK_ACOUSTIC_GATE_REVIEW_FIELDS,
        (
            {
                "protocol_id": FLEURS_KK_ACOUSTIC_GATE_PROTOCOL_ID,
                "packet_sha256": packet_hash,
                "text_hash": row.text_hash,
                "label": row.label,
                "sample_id": row.sample_id,
                "audio_sha256": row.audio_sha256,
                "relative_path": row.relative_path,
                "input_transcript": row.input_transcript,
                "reviewer_pseudo_id": reviewer,
                "review_status": "inconclusive",
                "audio_audible": "unknown",
                "kazakh_text_matches": "unknown",
                "no_obvious_defects": "unknown",
                "notes": "",
            }
            for row in packet
        ),
    )


def read_fleurs_kk_acoustic_reviews(path: Path) -> tuple[FleursKkAcousticReview, ...]:
    raw_rows = _read_csv_rows(path, FLEURS_KK_ACOUSTIC_GATE_REVIEW_FIELDS)
    reviews = tuple(FleursKkAcousticReview(**row) for row in raw_rows)
    if not reviews:
        raise FleursKkAcousticGateError("KK acoustic review CSV has no rows.")
    return reviews


def evaluate_fleurs_kk_acoustic_gate(
    packet_path: Path, reviews: Sequence[FleursKkAcousticReview]
) -> tuple[dict[str, object], tuple[FleursKkAcousticResult, ...]]:
    """Require two complete, distinct and internally consistent reviews per WAV."""

    packet = read_fleurs_kk_acoustic_packet(packet_path)
    packet_hash = sha256_file(packet_path)
    packet_by_id = {row.sample_id: row for row in packet}
    reviews_by_id: dict[str, list[FleursKkAcousticReview]] = defaultdict(list)
    reviewer_assets: dict[str, set[str]] = defaultdict(set)
    seen: set[tuple[str, str]] = set()
    issues: list[str] = []
    invalid_rows: list[int] = []
    for number, review in enumerate(reviews, start=2):
        item = packet_by_id.get(review.sample_id)
        if (
            item is None
            or review.protocol_id != FLEURS_KK_ACOUSTIC_GATE_PROTOCOL_ID
            or review.packet_sha256 != packet_hash
            or review.text_hash != item.text_hash
            or review.label != item.label
            or review.audio_sha256 != item.audio_sha256
            or review.relative_path != item.relative_path
            or review.input_transcript != item.input_transcript
            or not review.reviewer_pseudo_id
            or review.review_status not in _REVIEW_STATUSES
            or review.audio_audible not in _ANSWERS
            or review.kazakh_text_matches not in _ANSWERS
            or review.no_obvious_defects not in _ANSWERS
            or not _review_is_consistent(review)
        ):
            invalid_rows.append(number)
            continue
        key = (review.sample_id, review.reviewer_pseudo_id)
        if key in seen:
            issues.append(f"Reviewer duplicated an asset decision at review row {number}.")
            continue
        seen.add(key)
        reviewer_assets[review.reviewer_pseudo_id].add(review.sample_id)
        reviews_by_id[review.sample_id].append(review)

    if invalid_rows:
        preview = ", ".join(str(number) for number in invalid_rows[:10])
        suffix = f" and {len(invalid_rows) - 10} more" if len(invalid_rows) > 10 else ""
        issues.append(
            "Review rows "
            f"{preview}{suffix} do not match the immutable KK packet or decision contract."
        )
    if issues:
        raise FleursKkAcousticGateError(issues)

    reviewer_ids = set(reviewer_assets)
    if len(reviews) != _EXPECTED_ASSETS * _EXPECTED_REVIEWERS:
        issues.append(
            f"KK gate requires {_EXPECTED_ASSETS * _EXPECTED_REVIEWERS} review rows, "
            f"got {len(reviews)}."
        )
    if len(reviewer_ids) != _EXPECTED_REVIEWERS:
        issues.append(f"KK gate requires exactly {_EXPECTED_REVIEWERS} distinct reviewers.")
    expected_ids = set(packet_by_id)
    for reviewer_id, reviewed_ids in sorted(reviewer_assets.items()):
        if reviewed_ids != expected_ids:
            issues.append(
                f"Reviewer {reviewer_id!r} does not cover all {_EXPECTED_ASSETS} packet assets."
            )
    if any(len(asset_reviews) != _EXPECTED_REVIEWERS for asset_reviews in reviews_by_id.values()):
        issues.append("Every KK asset must have exactly two independent decisions.")
    if issues:
        raise FleursKkAcousticGateError(issues)

    results: list[FleursKkAcousticResult] = []
    for item in packet:
        asset_reviews = reviews_by_id[item.sample_id]
        reviewers = tuple(sorted(review.reviewer_pseudo_id for review in asset_reviews))
        passed = all(review.review_status == "pass" for review in asset_reviews)
        results.append(
            FleursKkAcousticResult(
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
    report = {
        "schema_version": FLEURS_KK_ACOUSTIC_GATE_SCHEMA_VERSION,
        "protocol_id": FLEURS_KK_ACOUSTIC_GATE_PROTOCOL_ID,
        "evidence_timing": "post_inference",
        "packet": {
            "path": str(packet_path),
            "sha256": packet_hash,
            "assets": _EXPECTED_ASSETS,
            "pairs": _EXPECTED_PAIRS,
        },
        "review_rows": len(reviews),
        "reviewers": sorted(reviewer_ids),
        "decision_counts": dict(sorted(counts.items())),
        "all_assets_acoustically_verified": counts == Counter({"pass": _EXPECTED_ASSETS}),
        "metric_status_changed": False,
        "blind_final_eligible": False,
        "final_or_product_eligible": False,
        "limitations": [
            "The gate establishes only audibility, Kazakh transcript correspondence and absence "
            "of obvious defects for the exact locked WAV bytes.",
            "The detector result was known before these reviews, so the evidence is post-inference "
            "and cannot become a blind final result retroactively.",
            "The gate does not establish speaker independence, source independence, calibration "
            "or product quality.",
        ],
    }
    return report, tuple(results)


def write_fleurs_kk_acoustic_report(
    path: Path, report: Mapping[str, object], results: Sequence[FleursKkAcousticResult]
) -> None:
    """Publish one immutable receipt after strict evaluation of both complete reviews."""

    if path.exists() or not path.parent.is_dir():
        raise FleursKkAcousticGateError("KK acoustic report output must be new.")
    if len(results) != _EXPECTED_ASSETS:
        raise FleursKkAcousticGateError(
            f"KK acoustic report requires {_EXPECTED_ASSETS} asset results."
        )
    payload = dict(report)
    payload["asset_results"] = [asdict(result) for result in results]
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
    except OSError as error:
        raise FleursKkAcousticGateError(f"Cannot write KK acoustic report: {error}") from error


def _review_is_consistent(review: FleursKkAcousticReview) -> bool:
    answers = (review.audio_audible, review.kazakh_text_matches, review.no_obvious_defects)
    if review.review_status == "pass":
        return all(answer == "yes" for answer in answers)
    if review.review_status == "fail":
        return "no" in answers and bool(review.notes)
    return "no" not in answers and "unknown" in answers and bool(review.notes)


def _read_csv_rows(path: Path, fields: tuple[str, ...]) -> tuple[dict[str, str], ...]:
    if not path.is_file():
        raise FleursKkAcousticGateError(f"Acoustic CSV does not exist: {path}")
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or tuple(reader.fieldnames) != fields:
                raise FleursKkAcousticGateError("Acoustic CSV has an invalid schema.")
            return tuple(
                {field: (raw.get(field) or "").strip() for field in fields} for raw in reader
            )
    except OSError as error:
        raise FleursKkAcousticGateError(f"Cannot read acoustic CSV: {error}") from error


def _write_csv(path: Path, fields: tuple[str, ...], rows: Iterable[Mapping[str, object]]) -> None:
    try:
        with path.open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    except OSError as error:
        raise FleursKkAcousticGateError(f"Cannot write acoustic CSV: {error}") from error


def _sha256(value: str) -> bool:
    return len(value) == 64 and all(character in _HEX for character in value)


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
