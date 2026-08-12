"""Two-listener language/intelligibility gate for the three KazakhTTS smoke files."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final, cast

from kds.data.assets import sha256_file

PROTOCOL_ID: Final = "fresh-suite-stage-c-kazakhtts-acoustic-gate-v1"
PACKET_FIELDS: Final = (
    "protocol_id",
    "smoke_report_sha256",
    "case_id",
    "language",
    "support_status",
    "relative_path",
    "audio_sha256",
    "text",
)
REVIEW_FIELDS: Final = (
    "protocol_id",
    "packet_sha256",
    "case_id",
    "language",
    "audio_sha256",
    "reviewer_pseudo_id",
    "review_status",
    "speech_intelligible",
    "text_preserved",
    "language_preserved",
    "severe_artifacts",
    "notes",
)


class KazakhTtsAcousticGateError(ValueError):
    def __init__(self, issues: Iterable[str]) -> None:
        self.issues = tuple(issues)
        super().__init__("\n".join(self.issues))


@dataclass(frozen=True, slots=True)
class KazakhTtsPacketRow:
    protocol_id: str
    smoke_report_sha256: str
    case_id: str
    language: str
    support_status: str
    relative_path: str
    audio_sha256: str
    text: str


@dataclass(frozen=True, slots=True)
class KazakhTtsReviewRow:
    protocol_id: str
    packet_sha256: str
    case_id: str
    language: str
    audio_sha256: str
    reviewer_pseudo_id: str
    review_status: str
    speech_intelligible: str
    text_preserved: str
    language_preserved: str
    severe_artifacts: str
    notes: str


def build_kazakhtts_acoustic_packet(report_path: Path) -> tuple[KazakhTtsPacketRow, ...]:
    """Bind the technical report to exact WAV bytes before any listening decisions."""

    try:
        value: object = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise KazakhTtsAcousticGateError(
            [f"Cannot read KazakhTTS smoke report: {error}"]
        ) from error
    if not isinstance(value, dict):
        raise KazakhTtsAcousticGateError(["KazakhTTS smoke report must be an object."])
    report = cast(dict[str, object], value)
    if (
        report.get("technical_smoke_passed") is not True
        or report.get("detector_inference_performed") is not False
        or report.get("acoustic_language_gate_passed") is not False
    ):
        raise KazakhTtsAcousticGateError(["KazakhTTS smoke report has an invalid gate state."])
    outputs = report.get("outputs")
    if not isinstance(outputs, list) or len(outputs) != 3:
        raise KazakhTtsAcousticGateError(["KazakhTTS smoke report must bind exactly three files."])
    report_hash = sha256_file(report_path)
    packet: list[KazakhTtsPacketRow] = []
    for index, value in enumerate(outputs, start=1):
        if not isinstance(value, dict):
            raise KazakhTtsAcousticGateError([f"KazakhTTS smoke output {index} is invalid."])
        output = cast(dict[str, object], value)
        strings: dict[str, str] = {}
        for field in (
            "case_id",
            "language",
            "support_status",
            "relative_path",
            "sha256",
            "text",
        ):
            item = output.get(field)
            if not isinstance(item, str) or not item.strip():
                raise KazakhTtsAcousticGateError(
                    [f"KazakhTTS smoke output {index} has invalid {field}."]
                )
            strings[field] = item.strip()
        audio_path = Path(strings["relative_path"])
        if not audio_path.is_file() or sha256_file(audio_path) != strings["sha256"]:
            raise KazakhTtsAcousticGateError(
                [f"KazakhTTS smoke output bytes do not match: {audio_path}"]
            )
        packet.append(
            KazakhTtsPacketRow(
                protocol_id=PROTOCOL_ID,
                smoke_report_sha256=report_hash,
                case_id=strings["case_id"],
                language=strings["language"],
                support_status=strings["support_status"],
                relative_path=strings["relative_path"],
                audio_sha256=strings["sha256"],
                text=strings["text"],
            )
        )
    if {row.language for row in packet} != {"ru", "kk", "mixed"}:
        raise KazakhTtsAcousticGateError(["KazakhTTS packet must contain RU, KK and mixed."])
    return tuple(sorted(packet, key=lambda row: row.case_id))


def write_kazakhtts_acoustic_packet(path: Path, rows: Sequence[KazakhTtsPacketRow]) -> None:
    _write_csv(path, PACKET_FIELDS, [asdict(row) for row in rows], "packet")


def read_kazakhtts_acoustic_packet(path: Path) -> tuple[KazakhTtsPacketRow, ...]:
    rows = _read_csv(path, PACKET_FIELDS, "packet")
    packet = tuple(KazakhTtsPacketRow(**row) for row in rows)
    if len(packet) != 3 or len({row.case_id for row in packet}) != 3:
        raise KazakhTtsAcousticGateError(["KazakhTTS packet must have three unique cases."])
    if (
        {row.language for row in packet} != {"ru", "kk", "mixed"}
        or any(
            row.protocol_id != PROTOCOL_ID
            or not _is_sha256(row.smoke_report_sha256)
            or not _is_sha256(row.audio_sha256)
            or row.support_status
            != (
                "officially_supported"
                if row.language == "kk"
                else "conditional_acoustic_smoke_only"
            )
            or not Path(row.relative_path).is_file()
            or sha256_file(Path(row.relative_path)) != row.audio_sha256
            for row in packet
        )
    ):
        raise KazakhTtsAcousticGateError(["KazakhTTS packet protocol is invalid."])
    return packet


def write_kazakhtts_review_template(
    path: Path, packet_path: Path, reviewer_pseudo_id: str
) -> None:
    reviewer = reviewer_pseudo_id.strip()
    if not reviewer or any(character in reviewer for character in ",\r\n"):
        raise KazakhTtsAcousticGateError(["Reviewer ID must be non-empty and single-line."])
    packet = read_kazakhtts_acoustic_packet(packet_path)
    packet_hash = sha256_file(packet_path)
    rows = [
        {
            "protocol_id": PROTOCOL_ID,
            "packet_sha256": packet_hash,
            "case_id": item.case_id,
            "language": item.language,
            "audio_sha256": item.audio_sha256,
            "reviewer_pseudo_id": reviewer,
            "review_status": "inconclusive",
            "speech_intelligible": "unknown",
            "text_preserved": "unknown",
            "language_preserved": "unknown",
            "severe_artifacts": "unknown",
            "notes": "",
        }
        for item in packet
    ]
    _write_csv(path, REVIEW_FIELDS, rows, "review template")


def read_kazakhtts_reviews(path: Path) -> tuple[KazakhTtsReviewRow, ...]:
    return tuple(
        KazakhTtsReviewRow(**row) for row in _read_csv(path, REVIEW_FIELDS, "review")
    )


def evaluate_kazakhtts_acoustic_gate(
    packet_path: Path, reviews: Sequence[KazakhTtsReviewRow]
) -> dict[str, object]:
    """Authorize languages only when two distinct listeners pass the exact smoke WAV."""

    packet = read_kazakhtts_acoustic_packet(packet_path)
    packet_hash = sha256_file(packet_path)
    by_case: dict[str, list[KazakhTtsReviewRow]] = defaultdict(list)
    issues: list[str] = []
    packet_by_case = {row.case_id: row for row in packet}
    seen: set[tuple[str, str]] = set()
    for number, review in enumerate(reviews, start=2):
        expected = packet_by_case.get(review.case_id)
        key = (review.case_id, review.reviewer_pseudo_id)
        if (
            expected is None
            or review.protocol_id != PROTOCOL_ID
            or review.packet_sha256 != packet_hash
            or review.language != expected.language
            or review.audio_sha256 != expected.audio_sha256
            or not review.reviewer_pseudo_id
            or "REPLACE_ME" in review.reviewer_pseudo_id
            or review.review_status not in {"pass", "fail", "inconclusive"}
            or review.speech_intelligible not in {"yes", "no", "unknown"}
            or review.text_preserved not in {"yes", "no", "unknown"}
            or review.language_preserved not in {"yes", "no", "unknown"}
            or review.severe_artifacts not in {"yes", "no", "unknown"}
            or key in seen
        ):
            issues.append(f"KazakhTTS review row {number} violates its packet contract.")
            continue
        seen.add(key)
        by_case[review.case_id].append(review)
    if issues:
        raise KazakhTtsAcousticGateError(issues)
    results: list[dict[str, object]] = []
    approved_languages: list[str] = []
    for item in packet:
        decisions = by_case[item.case_id]
        reviewers = sorted({review.reviewer_pseudo_id for review in decisions})
        passed = (
            len(decisions) == 2
            and len(reviewers) == 2
            and all(
                review.review_status == "pass"
                and review.speech_intelligible == "yes"
                and review.text_preserved == "yes"
                and review.language_preserved == "yes"
                and review.severe_artifacts == "no"
                for review in decisions
            )
        )
        if passed:
            approved_languages.append(item.language)
        results.append(
            {
                "case_id": item.case_id,
                "language": item.language,
                "reviewers": reviewers,
                "decision": "pass" if passed else "not_eligible",
            }
        )
    return {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "packet_sha256": packet_hash,
        "review_rows": len(reviews),
        "approved_input_languages": sorted(approved_languages),
        "all_languages_passed": len(approved_languages) == 3,
        "detector_inference_authorized": False,
        "results": results,
        "interpretation": (
            "A passed language may be used only to build a new pre-inference candidate; final "
            "asset selection, QA, two-review gate and immutable inference plan remain required."
        ),
    }


def _write_csv(
    path: Path, fields: tuple[str, ...], rows: Sequence[dict[str, str]], label: str
) -> None:
    if path.exists() or not path.parent.is_dir():
        raise KazakhTtsAcousticGateError(
            [f"KazakhTTS {label} output must be new with an existing parent."]
        )
    try:
        with path.open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    except OSError as error:
        raise KazakhTtsAcousticGateError([f"Cannot write KazakhTTS {label}: {error}"]) from error


def _read_csv(path: Path, fields: tuple[str, ...], label: str) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or tuple(reader.fieldnames) != fields:
                raise KazakhTtsAcousticGateError(
                    [f"KazakhTTS {label} has an invalid schema."]
                )
            return [
                {field: (raw.get(field) or "").strip() for field in fields} for raw in reader
            ]
    except OSError as error:
        raise KazakhTtsAcousticGateError([f"Cannot read KazakhTTS {label}: {error}"]) from error


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)
