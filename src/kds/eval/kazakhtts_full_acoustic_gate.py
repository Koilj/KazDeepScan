"""Fail-closed two-review acoustic gate for every QA-ready Stage-C KazakhTTS asset."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final, cast

from kds.data.assets import sha256_file
from kds.data.kazakhtts import KAZAKHTTS_SOURCE_ID
from kds.data.kazakhtts_text import KAZAKHTTS_TEXT_NORMALIZER_ID
from kds.data.manifest import ManifestRow

PROTOCOL_ID: Final = "fresh-suite-stage-c-kazakhtts-full-acoustic-gate-v1"
PACKET_FIELDS: Final = (
    "protocol_id",
    "pairing_receipt_sha256",
    "candidate_manifest_sha256",
    "normalization_plan_sha256",
    "sample_id",
    "language",
    "text_id",
    "source_text_hash",
    "audio_path",
    "audio_sha256",
    "source_text",
    "synthesis_text",
    "synthesis_text_sha256",
    "normalizer_id",
)
REVIEW_FIELDS: Final = (
    "protocol_id",
    "packet_sha256",
    "sample_id",
    "language",
    "audio_sha256",
    "reviewer_pseudo_id",
    "review_status",
    "speech_intelligible",
    "lexical_content_preserved",
    "language_preserved",
    "severe_artifacts",
    "notes",
)
_ANSWERS = frozenset({"yes", "no", "unknown"})
_REVIEW_STATUSES = frozenset({"pass", "fail", "inconclusive"})
_EXPECTED_LANGUAGES = Counter({"kk": 60, "mixed": 57, "ru": 50})


class KazakhTtsFullAcousticGateError(ValueError):
    """Raised when exact assets or reviewer decisions violate the frozen gate contract."""

    def __init__(self, issues: Iterable[str] | str) -> None:
        self.issues = (issues,) if isinstance(issues, str) else tuple(issues)
        super().__init__("\n".join(self.issues))


@dataclass(frozen=True, slots=True)
class KazakhTtsFullPacketRow:
    protocol_id: str
    pairing_receipt_sha256: str
    candidate_manifest_sha256: str
    normalization_plan_sha256: str
    sample_id: str
    language: str
    text_id: str
    source_text_hash: str
    audio_path: str
    audio_sha256: str
    source_text: str
    synthesis_text: str
    synthesis_text_sha256: str
    normalizer_id: str


@dataclass(frozen=True, slots=True)
class KazakhTtsFullReviewRow:
    protocol_id: str
    packet_sha256: str
    sample_id: str
    language: str
    audio_sha256: str
    reviewer_pseudo_id: str
    review_status: str
    speech_intelligible: str
    lexical_content_preserved: str
    language_preserved: str
    severe_artifacts: str
    notes: str


def _json_object(path: Path, label: str) -> Mapping[str, object]:
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise KazakhTtsFullAcousticGateError(f"Cannot read {label}: {error}") from error
    if not isinstance(value, dict):
        raise KazakhTtsFullAcousticGateError(f"{label} must be a JSON object.")
    return cast(Mapping[str, object], value)


def build_kazakhtts_full_acoustic_packet(
    *,
    candidate_rows: Sequence[ManifestRow],
    candidate_path: Path,
    pairing_receipt_path: Path,
    normalization_plan_path: Path,
    data_root: Path,
) -> tuple[KazakhTtsFullPacketRow, ...]:
    """Bind each QA-ready synthetic candidate WAV to its exact synthesis transcript."""

    if len(candidate_rows) != 334:
        raise KazakhTtsFullAcousticGateError(
            "Stage-C full acoustic gate requires exactly 167 binary pairs."
        )
    by_text: dict[str, list[ManifestRow]] = defaultdict(list)
    for row in candidate_rows:
        by_text[row.text_id].append(row)
    if len(by_text) != 167 or any(
        len(pair) != 2 or {row.label for row in pair} != {"bonafide", "spoof"}
        for pair in by_text.values()
    ):
        raise KazakhTtsFullAcousticGateError("Stage-C candidate does not contain exact pairs.")
    spoof = [row for row in candidate_rows if row.label == "spoof"]
    if (
        Counter(row.language for row in spoof) != _EXPECTED_LANGUAGES
        or any(
            row.source_name != KAZAKHTTS_SOURCE_ID
            or row.split != "test"
            or f"text_normalizer={KAZAKHTTS_TEXT_NORMALIZER_ID}" not in row.augmentation_chain
            for row in spoof
        )
    ):
        raise KazakhTtsFullAcousticGateError("Stage-C spoof candidate contract is invalid.")

    candidate_hash = sha256_file(candidate_path)
    pairing = _json_object(pairing_receipt_path, "Stage-C pairing receipt")
    outputs = pairing.get("outputs")
    combined = outputs.get("combined") if isinstance(outputs, dict) else None
    if (
        pairing.get("schema_version") != 1
        or pairing.get("protocol_id") != "fresh-suite-stage-c-kazakhtts-pairing-v1"
        or pairing.get("full_asset_acoustic_gate_passed") is not False
        or pairing.get("detector_inference_performed") is not False
        or pairing.get("detector_inference_authorized") is not False
        or not isinstance(combined, dict)
        or combined.get("path") != candidate_path.as_posix()
        or combined.get("sha256") != candidate_hash
        or combined.get("rows") != 334
    ):
        raise KazakhTtsFullAcousticGateError("Stage-C pairing receipt binding is invalid.")

    normalization = _json_object(normalization_plan_path, "Stage-C normalization plan")
    raw_rows = normalization.get("rows")
    if (
        normalization.get("schema_version") != 1
        or normalization.get("protocol_id")
        != "fresh-suite-stage-c-kazakhtts-normalization-v1"
        or normalization.get("normalizer_id") != KAZAKHTTS_TEXT_NORMALIZER_ID
        or normalization.get("row_count") != 168
        or not isinstance(raw_rows, list)
    ):
        raise KazakhTtsFullAcousticGateError("Stage-C normalization plan is invalid.")
    normalized_by_text: dict[str, Mapping[str, object]] = {}
    for raw in raw_rows:
        if not isinstance(raw, dict) or not isinstance(raw.get("text_id"), str):
            raise KazakhTtsFullAcousticGateError("Stage-C normalization row is invalid.")
        text_id = cast(str, raw["text_id"])
        if text_id in normalized_by_text:
            raise KazakhTtsFullAcousticGateError(
                f"Stage-C normalization repeats text ID {text_id!r}."
            )
        normalized_by_text[text_id] = cast(Mapping[str, object], raw)

    pairing_hash = sha256_file(pairing_receipt_path)
    normalization_hash = sha256_file(normalization_plan_path)
    packet: list[KazakhTtsFullPacketRow] = []
    for row in spoof:
        normalized = normalized_by_text.get(row.text_id)
        if normalized is None:
            raise KazakhTtsFullAcousticGateError(
                f"Stage-C synthesis text is missing for {row.text_id!r}."
            )
        source_text = normalized.get("source_text")
        synthesis_text = normalized.get("normalized_text")
        synthesis_hash = normalized.get("normalized_text_sha256")
        asset = data_root / row.relative_path
        if (
            normalized.get("language") != row.language
            or normalized.get("source_text_hash") != row.text_hash
            or not isinstance(source_text, str)
            or not source_text
            or not isinstance(synthesis_text, str)
            or not synthesis_text
            or not isinstance(synthesis_hash, str)
            or f"synthesis_text_sha256={synthesis_hash}" not in row.augmentation_chain
            or not asset.is_file()
            or sha256_file(asset) != row.sha256
        ):
            raise KazakhTtsFullAcousticGateError(
                f"Stage-C full acoustic evidence is invalid for {row.sample_id!r}."
            )
        packet.append(
            KazakhTtsFullPacketRow(
                protocol_id=PROTOCOL_ID,
                pairing_receipt_sha256=pairing_hash,
                candidate_manifest_sha256=candidate_hash,
                normalization_plan_sha256=normalization_hash,
                sample_id=row.sample_id,
                language=row.language,
                text_id=row.text_id,
                source_text_hash=row.text_hash,
                audio_path=asset.as_posix(),
                audio_sha256=row.sha256,
                source_text=source_text,
                synthesis_text=synthesis_text,
                synthesis_text_sha256=synthesis_hash,
                normalizer_id=KAZAKHTTS_TEXT_NORMALIZER_ID,
            )
        )
    return tuple(sorted(packet, key=lambda item: (item.language, item.text_id)))


def write_kazakhtts_full_acoustic_packet(
    path: Path, rows: Sequence[KazakhTtsFullPacketRow]
) -> None:
    if (
        path.exists()
        or not path.parent.is_dir()
        or len(rows) != 167
        or Counter(row.language for row in rows) != _EXPECTED_LANGUAGES
    ):
        raise KazakhTtsFullAcousticGateError(
            "Stage-C full acoustic packet output must be new and contain 167 assets."
        )
    _write_csv(path, PACKET_FIELDS, (asdict(row) for row in rows))


def read_kazakhtts_full_acoustic_packet(path: Path) -> tuple[KazakhTtsFullPacketRow, ...]:
    packet = tuple(
        KazakhTtsFullPacketRow(**row) for row in _read_csv(path, PACKET_FIELDS, "packet")
    )
    if (
        len(packet) != 167
        or len({row.sample_id for row in packet}) != 167
        or Counter(row.language for row in packet) != _EXPECTED_LANGUAGES
        or len({row.pairing_receipt_sha256 for row in packet}) != 1
        or len({row.candidate_manifest_sha256 for row in packet}) != 1
        or len({row.normalization_plan_sha256 for row in packet}) != 1
        or any(
            row.protocol_id != PROTOCOL_ID
            or row.normalizer_id != KAZAKHTTS_TEXT_NORMALIZER_ID
            or not row.source_text
            or not row.synthesis_text
            or not _is_sha256(row.source_text_hash)
            or not _is_sha256(row.audio_sha256)
            or not _is_sha256(row.synthesis_text_sha256)
            or not _is_sha256(row.pairing_receipt_sha256)
            or not _is_sha256(row.candidate_manifest_sha256)
            or not _is_sha256(row.normalization_plan_sha256)
            or not Path(row.audio_path).is_file()
            or sha256_file(Path(row.audio_path)) != row.audio_sha256
            for row in packet
        )
    ):
        raise KazakhTtsFullAcousticGateError(
            "Stage-C full acoustic packet violates its exact-byte contract."
        )
    return packet


def write_kazakhtts_full_review_template(
    path: Path, packet_path: Path, reviewer_pseudo_id: str
) -> None:
    reviewer = reviewer_pseudo_id.strip()
    if (
        not reviewer
        or any(character in reviewer for character in ",\r\n")
        or path.exists()
        or not path.parent.is_dir()
    ):
        raise KazakhTtsFullAcousticGateError(
            "Stage-C reviewer ID/output is invalid or already exists."
        )
    packet = read_kazakhtts_full_acoustic_packet(packet_path)
    packet_hash = sha256_file(packet_path)
    _write_csv(
        path,
        REVIEW_FIELDS,
        (
            {
                "protocol_id": PROTOCOL_ID,
                "packet_sha256": packet_hash,
                "sample_id": row.sample_id,
                "language": row.language,
                "audio_sha256": row.audio_sha256,
                "reviewer_pseudo_id": reviewer,
                "review_status": "inconclusive",
                "speech_intelligible": "unknown",
                "lexical_content_preserved": "unknown",
                "language_preserved": "unknown",
                "severe_artifacts": "unknown",
                "notes": "",
            }
            for row in packet
        ),
    )


def read_kazakhtts_full_reviews(path: Path) -> tuple[KazakhTtsFullReviewRow, ...]:
    reviews = tuple(
        KazakhTtsFullReviewRow(**row) for row in _read_csv(path, REVIEW_FIELDS, "review")
    )
    if not reviews:
        raise KazakhTtsFullAcousticGateError("Stage-C full acoustic review has no rows.")
    return reviews


def evaluate_kazakhtts_full_acoustic_gate(
    packet_path: Path, reviews: Sequence[KazakhTtsFullReviewRow]
) -> dict[str, object]:
    """Pass only when two distinct listeners approve every exact synthetic asset."""

    packet = read_kazakhtts_full_acoustic_packet(packet_path)
    packet_hash = sha256_file(packet_path)
    packet_by_id = {row.sample_id: row for row in packet}
    by_id: dict[str, list[KazakhTtsFullReviewRow]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    issues: list[str] = []
    for number, review in enumerate(reviews, start=2):
        expected = packet_by_id.get(review.sample_id)
        key = (review.sample_id, review.reviewer_pseudo_id)
        if (
            expected is None
            or review.protocol_id != PROTOCOL_ID
            or review.packet_sha256 != packet_hash
            or review.language != expected.language
            or review.audio_sha256 != expected.audio_sha256
            or not review.reviewer_pseudo_id
            or "REPLACE_ME" in review.reviewer_pseudo_id
            or review.review_status not in _REVIEW_STATUSES
            or review.speech_intelligible not in _ANSWERS
            or review.lexical_content_preserved not in _ANSWERS
            or review.language_preserved not in _ANSWERS
            or review.severe_artifacts not in _ANSWERS
            or key in seen
        ):
            issues.append(f"Stage-C full acoustic review row {number} is invalid.")
            continue
        seen.add(key)
        by_id[review.sample_id].append(review)
    if issues:
        raise KazakhTtsFullAcousticGateError(issues)

    results: list[dict[str, object]] = []
    passed_by_language: Counter[str] = Counter()
    for item in packet:
        decisions = by_id[item.sample_id]
        reviewers = sorted({review.reviewer_pseudo_id for review in decisions})
        passed = (
            len(decisions) == 2
            and len(reviewers) == 2
            and all(
                review.review_status == "pass"
                and review.speech_intelligible == "yes"
                and review.lexical_content_preserved == "yes"
                and review.language_preserved == "yes"
                and review.severe_artifacts == "no"
                for review in decisions
            )
        )
        if passed:
            passed_by_language[item.language] += 1
        results.append(
            {
                "sample_id": item.sample_id,
                "language": item.language,
                "audio_sha256": item.audio_sha256,
                "reviewers": reviewers,
                "decision": "pass" if passed else "not_eligible",
            }
        )
    all_passed = passed_by_language == _EXPECTED_LANGUAGES
    return {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "packet": {"path": packet_path.as_posix(), "sha256": packet_hash},
        "review_rows": len(reviews),
        "asset_count": len(packet),
        "passed_by_language": dict(sorted(passed_by_language.items())),
        "all_assets_acoustically_verified": all_passed,
        "detector_inference_performed": False,
        "detector_inference_authorized": False,
        "immutable_inference_plan_authorized": all_passed,
        "results": results,
        "interpretation": (
            "A full pass permits exposure audit and an immutable one-run inference plan; "
            "it does not itself authorize detector inference or product claims."
        ),
    }


def write_kazakhtts_full_acoustic_report(path: Path, report: Mapping[str, object]) -> None:
    if path.exists() or not path.parent.is_dir():
        raise KazakhTtsFullAcousticGateError(
            "Stage-C full acoustic report output must be new."
        )
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(
    path: Path, fields: Sequence[str], rows: Iterable[Mapping[str, object]]
) -> None:
    try:
        with path.open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    except OSError as error:
        raise KazakhTtsFullAcousticGateError(f"Cannot write Stage-C CSV: {error}") from error


def _read_csv(path: Path, fields: Sequence[str], label: str) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != tuple(fields):
                raise KazakhTtsFullAcousticGateError(
                    f"Stage-C {label} CSV columns are invalid."
                )
            rows = [dict(row) for row in reader]
    except OSError as error:
        raise KazakhTtsFullAcousticGateError(
            f"Cannot read Stage-C {label} CSV: {error}"
        ) from error
    return rows


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)
