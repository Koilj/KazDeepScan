"""Reproducible single-AI transcript review for a narrow KSC2 mixed evidence layer.

The immutable KSC2 candidate packet has no per-row code-switch labels.  This module
does not infer a language from a component path, Cyrillic letters, ASR, or a language
identifier.  It materialises only an explicitly curated, high-precision positive list
from one AI semantic review; its exact Russian and Kazakh token positions are kept in
source control.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

from kds.data.ksc2 import KSC2_MIXED_ANNOTATION_COMPONENTS

KSC2_MIXED_CANDIDATE_FIELDS: Final[tuple[str, ...]] = (
    "annotation_id",
    "component",
    "audio_relative_path",
    "audio_sha256",
    "archive_audio_member",
    "archive_transcript_member",
    "transcript",
    "transcript_sha256",
    "source_name",
    "source_license",
    "archive_sha256",
    "source_lock_sha256",
    "annotation_state",
    "language",
    "code_switch",
)
AI_REVIEW_FIELDS: Final[tuple[str, ...]] = (
    "annotation_id",
    "component",
    "audio_relative_path",
    "audio_sha256",
    "archive_audio_member",
    "archive_transcript_member",
    "transcript",
    "transcript_sha256",
    "source_name",
    "source_license",
    "archive_sha256",
    "source_lock_sha256",
    "candidate_packet_sha256",
    "candidate_receipt_sha256",
    "language",
    "code_switch",
    "review_method",
    "reviewer",
    "ru_evidence_token_indices",
    "ru_evidence_tokens",
    "kk_evidence_token_indices",
    "kk_evidence_tokens",
    "reviewed_at",
)

_HEX = frozenset("0123456789abcdef")
_KAZAKH_SPECIFIC = frozenset("әғқңөұүһіӘҒҚҢӨҰҮҺІ")
_WORD = re.compile(r"[^\W_]+(?:[’'-][^\W_]+)*", flags=re.UNICODE)


class Ksc2MixedReviewError(ValueError):
    """Raised when the candidate packet or explicit review evidence is invalid."""

    def __init__(self, issues: Iterable[str]) -> None:
        self.issues = tuple(issues)
        super().__init__("\n".join(self.issues))


@dataclass(frozen=True, slots=True)
class CandidatePacket:
    rows: tuple[dict[str, str], ...]
    rows_by_id: Mapping[str, dict[str, str]]
    packet_sha256: str
    receipt_sha256: str
    lock_sha256: str


@dataclass(frozen=True, slots=True)
class ReviewDecision:
    """One explicitly reviewed positive; positions are 1-based transcript tokens."""

    annotation_id: str
    ru_indices: tuple[int, ...]
    kk_indices: tuple[int, ...]


# These are deliberate semantic-review decisions, not a lexical detector.  The source text and
# evidence positions are emitted in every derived row for later audit.
CURATED_MIXED_DECISIONS: Final[tuple[ReviewDecision, ...]] = (
    ReviewDecision("ksc2_v1:Test/podcasts/09_01_093", (18,), (2,)),
    ReviewDecision("ksc2_v1:Test/podcasts/09_01_095", (9, 12), (1,)),
    ReviewDecision("ksc2_v1:Test/podcasts/09_01_149", (11,), (2,)),
    ReviewDecision("ksc2_v1:Test/podcasts/09_01_150", (11, 17), (1,)),
    ReviewDecision("ksc2_v1:Test/podcasts/09_01_151", (7, 8, 14), (2,)),
    ReviewDecision("ksc2_v1:Test/podcasts/09_01_216", (9, 20), (1,)),
    ReviewDecision("ksc2_v1:Test/podcasts/09_01_251", (1, 2), (3,)),
    ReviewDecision("ksc2_v1:Test/podcasts/09_01_292", (5,), (1,)),
    ReviewDecision("ksc2_v1:Test/podcasts/09_01_296", (4,), (2,)),
    ReviewDecision("ksc2_v1:Test/podcasts/09_03_020", (14,), (1,)),
    ReviewDecision("ksc2_v1:Test/podcasts/09_03_204", (5,), (2,)),
    ReviewDecision("ksc2_v1:Test/podcasts/09_03_220", (7, 14, 15), (1,)),
    ReviewDecision("ksc2_v1:Test/podcasts/09_03_251", (2, 3), (1,)),
    ReviewDecision("ksc2_v1:Test/podcasts/09_03_287", (12, 13, 14, 15, 16, 17, 18, 19, 20), (3,)),
    ReviewDecision("ksc2_v1:Test/podcasts/09_03_289", (3, 4), (6,)),
    ReviewDecision("ksc2_v1:Test/podcasts/09_03_368", (1, 2, 4), (3,)),
    ReviewDecision("ksc2_v1:Test/podcasts/09_04_024", (10,), (1,)),
    ReviewDecision("ksc2_v1:Test/podcasts/09_04_051", (2, 3, 4), (5,)),
    ReviewDecision("ksc2_v1:Test/podcasts/09_04_072", (13, 14, 15), (6,)),
    ReviewDecision("ksc2_v1:Test/podcasts/09_04_074", (3,), (4,)),
    ReviewDecision("ksc2_v1:Test/podcasts/09_04_099", (8, 9), (1,)),
    ReviewDecision("ksc2_v1:Test/podcasts/09_04_212", (10,), (2,)),
    ReviewDecision("ksc2_v1:Test/radio/01_04_072", (4,), (3,)),
    ReviewDecision("ksc2_v1:Test/radio/01_05_157", (7,), (2,)),
    ReviewDecision("ksc2_v1:Test/radio/01_05_194", (4, 5), (1,)),
    ReviewDecision("ksc2_v1:Test/radio/01_05_274", (8, 9), (4,)),
    ReviewDecision("ksc2_v1:Test/talkshow/01_02_067", (13, 14), (2,)),
    ReviewDecision("ksc2_v1:Test/talkshow/01_02_093", (6,), (1,)),
    ReviewDecision("ksc2_v1:Test/talkshow/01_02_109", (3,), (2,)),
    ReviewDecision("ksc2_v1:Test/talkshow/01_02_110", (5, 6, 11, 12, 18, 19), (1,)),
    ReviewDecision("ksc2_v1:Test/talkshow/01_02_274", (10,), (1,)),
    ReviewDecision("ksc2_v1:Test/talkshow/01_02_277", (8,), (2,)),
)

# A second, disjoint review pass used for Stage C.  Candidate ranking used conservative Russian
# discourse/phrase cues only to make manual semantic reading tractable; every row below is an
# explicit decision with stored token positions.  The four ambiguous ranking hits based only on
# ``проблема``, ``так`` or a weak ``там`` were deliberately left unknown.
CURATED_MIXED_DECISIONS_V2_DELTA: Final[tuple[ReviewDecision, ...]] = (
    ReviewDecision("ksc2_v1:Test/talkshow/01_02_088", (18, 19), (1,)),
    ReviewDecision("ksc2_v1:Test/podcasts/09_03_005", (2,), (3,)),
    ReviewDecision("ksc2_v1:Test/podcasts/09_03_366", (1,), (2,)),
    ReviewDecision("ksc2_v1:Test/radio/01_04_142", (2,), (3,)),
    ReviewDecision("ksc2_v1:Test/radio/01_05_180", (1,), (2,)),
    ReviewDecision("ksc2_v1:Test/radio/01_05_128", (4,), (5,)),
    ReviewDecision("ksc2_v1:Test/podcasts/09_01_243", (4,), (1,)),
    ReviewDecision("ksc2_v1:Test/podcasts/09_04_011", (1,), (3,)),
    ReviewDecision("ksc2_v1:Test/radio/01_05_057", (2,), (3,)),
    ReviewDecision("ksc2_v1:Test/radio/01_05_226", (4,), (3,)),
    ReviewDecision("ksc2_v1:Test/radio/01_05_287", (4,), (2,)),
    ReviewDecision("ksc2_v1:Test/podcasts/09_01_048", (11,), (2,)),
    ReviewDecision("ksc2_v1:Test/podcasts/09_02_062", (8,), (2,)),
    ReviewDecision("ksc2_v1:Test/podcasts/09_00_388", (8,), (2,)),
    ReviewDecision("ksc2_v1:Test/podcasts/09_01_304", (11,), (1,)),
    ReviewDecision("ksc2_v1:Test/podcasts/09_03_044", (10,), (1,)),
    ReviewDecision("ksc2_v1:Test/podcasts/09_04_226", (1,), (3,)),
    ReviewDecision("ksc2_v1:Test/radio/01_04_152", (3,), (1,)),
    ReviewDecision("ksc2_v1:Test/podcasts/09_03_384", (2,), (3,)),
    ReviewDecision("ksc2_v1:Test/podcasts/09_04_030", (1,), (2,)),
    ReviewDecision("ksc2_v1:Test/podcasts/09_04_061", (7, 8), (1,)),
    ReviewDecision("ksc2_v1:Test/radio/01_04_116", (1,), (3,)),
    ReviewDecision("ksc2_v1:Test/podcasts/09_03_414", (2,), (3,)),
    ReviewDecision("ksc2_v1:Test/radio/01_05_304", (5, 9), (2,)),
    ReviewDecision("ksc2_v1:Test/talkshow/01_02_312", (7,), (1,)),
    ReviewDecision("ksc2_v1:Test/podcasts/09_01_172", (14,), (2,)),
    ReviewDecision("ksc2_v1:Test/podcasts/09_03_121", (14,), (4,)),
    ReviewDecision("ksc2_v1:Test/podcasts/09_03_129", (6,), (2,)),
    ReviewDecision("ksc2_v1:Test/podcasts/09_03_419", (11,), (1,)),
    ReviewDecision("ksc2_v1:Test/talkshow/01_02_329", (2,), (3,)),
    ReviewDecision("ksc2_v1:Test/podcasts/09_01_017", (7,), (1,)),
    ReviewDecision("ksc2_v1:Test/podcasts/09_04_149", (10, 11), (1,)),
    ReviewDecision("ksc2_v1:Test/podcasts/09_04_253", (1,), (2,)),
    ReviewDecision("ksc2_v1:Test/podcasts/09_01_098", (15,), (3,)),
    ReviewDecision("ksc2_v1:Test/podcasts/09_03_219", (3,), (1,)),
    ReviewDecision("ksc2_v1:Test/podcasts/09_03_165", (15,), (2,)),
    ReviewDecision("ksc2_v1:Test/podcasts/09_03_221", (1,), (2,)),
    ReviewDecision("ksc2_v1:Test/podcasts/09_04_125", (10, 11, 12), (1,)),
    ReviewDecision("ksc2_v1:Test/radio/01_04_117", (4,), (1,)),
    ReviewDecision("ksc2_v1:Test/radio/01_05_272", (1, 4), (2,)),
    ReviewDecision("ksc2_v1:Test/podcasts/09_01_078", (14,), (2,)),
    ReviewDecision("ksc2_v1:Test/radio/01_04_144", (5,), (6,)),
    ReviewDecision("ksc2_v1:Test/podcasts/09_00_160", (20,), (2,)),
    ReviewDecision("ksc2_v1:Test/podcasts/09_01_083", (6,), (1,)),
    ReviewDecision("ksc2_v1:Test/radio/01_05_013", (13,), (1,)),
    ReviewDecision("ksc2_v1:Test/radio/01_05_109", (6,), (3,)),
    ReviewDecision("ksc2_v1:Test/podcasts/09_03_410", (9,), (3,)),
    ReviewDecision("ksc2_v1:Test/podcasts/09_04_153", (19,), (2,)),
    ReviewDecision("ksc2_v1:Test/podcasts/09_04_110", (18,), (2,)),
    ReviewDecision("ksc2_v1:Test/podcasts/09_03_418", (6,), (4,)),
    ReviewDecision("ksc2_v1:Test/talkshow/01_02_348", (22,), (2,)),
    ReviewDecision("ksc2_v1:Test/podcasts/09_04_102", (10,), (2,)),
    ReviewDecision("ksc2_v1:Test/podcasts/09_04_255", (2,), (5,)),
    ReviewDecision("ksc2_v1:Test/talkshow/01_02_074", (6,), (8,)),
    ReviewDecision("ksc2_v1:Test/podcasts/09_03_347", (14, 16, 22), (1,)),
    ReviewDecision("ksc2_v1:Test/podcasts/09_04_123", (27,), (2,)),
    ReviewDecision("ksc2_v1:Test/podcasts/09_04_245", (2,), (4,)),
    ReviewDecision("ksc2_v1:Test/podcasts/09_00_234", (7,), (3,)),
    ReviewDecision("ksc2_v1:Test/podcasts/09_03_085", (27,), (6,)),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tokenize_transcript(transcript: str) -> tuple[str, ...]:
    return tuple(_WORD.findall(transcript))


def _read_csv(path: Path, fields: Sequence[str], label: str) -> list[dict[str, str]]:
    if not path.is_file():
        raise Ksc2MixedReviewError([f"{label} does not exist: {path}"])
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or tuple(reader.fieldnames) != tuple(fields):
                raise Ksc2MixedReviewError([f"{label} header does not match its strict schema."])
            rows: list[dict[str, str]] = []
            for row_number, raw in enumerate(reader, start=2):
                if None in raw:
                    raise Ksc2MixedReviewError([f"{label} row {row_number} has extra values."])
                rows.append({field: (raw[field] or "").strip() for field in fields})
    except OSError as error:
        raise Ksc2MixedReviewError([f"Cannot read {label}: {error}"]) from error
    if not rows:
        raise Ksc2MixedReviewError([f"{label} has no data rows."])
    return rows


def _json_object(path: Path, label: str) -> dict[str, object]:
    if not path.is_file():
        raise Ksc2MixedReviewError([f"{label} does not exist: {path}"])
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Ksc2MixedReviewError([f"Cannot parse {label}: {error}"]) from error
    if not isinstance(value, dict):
        raise Ksc2MixedReviewError([f"{label} must be a JSON object."])
    return cast(dict[str, object], value)


def _sha256(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise Ksc2MixedReviewError([f"{label} must be a SHA-256 string."])
    digest = value.lower()
    if len(digest) != 64 or any(character not in _HEX for character in digest):
        raise Ksc2MixedReviewError([f"{label} must be a 64-character lowercase SHA-256."])
    return digest


def _counts(value: object, label: str) -> dict[str, int]:
    if not isinstance(value, dict):
        raise Ksc2MixedReviewError([f"{label} must be an object."])
    result: dict[str, int] = {}
    for name, count in value.items():
        if not isinstance(name, str) or not isinstance(count, int) or count < 0:
            raise Ksc2MixedReviewError([f"{label} must contain non-negative component counts."])
        result[name] = count
    return result


def load_candidate_packet(packet: Path, receipt: Path, lock: Path) -> CandidatePacket:
    """Load the original candidate packet if its packet/receipt/source locks still agree."""

    rows = _read_csv(packet, KSC2_MIXED_CANDIDATE_FIELDS, "KSC2 mixed candidate packet")
    packet_hash = sha256_file(packet)
    receipt_hash = sha256_file(receipt)
    lock_hash = sha256_file(lock)
    lock_value = _json_object(lock, "KSC2 mixed candidate packet lock")
    required_lock = {
        "schema_version",
        "packet_path",
        "packet_sha256",
        "packet_receipt_path",
        "packet_receipt_sha256",
        "candidate_count",
        "candidate_counts_by_component",
        "priority_components",
        "archive_sha256",
        "source_lock_sha256",
    }
    if set(lock_value) != required_lock or lock_value.get("schema_version") != 1:
        raise Ksc2MixedReviewError(["KSC2 mixed candidate packet lock has an invalid schema."])
    if _sha256(lock_value.get("packet_sha256"), "packet lock packet_sha256") != packet_hash:
        raise Ksc2MixedReviewError(["KSC2 mixed candidate packet SHA-256 differs from its lock."])
    if (
        _sha256(lock_value.get("packet_receipt_sha256"), "packet lock receipt SHA-256")
        != receipt_hash
    ):
        raise Ksc2MixedReviewError(["KSC2 mixed candidate receipt SHA-256 differs from its lock."])
    if lock_value.get("packet_path") != packet.as_posix():
        raise Ksc2MixedReviewError(["KSC2 mixed candidate packet path differs from its lock."])
    if lock_value.get("packet_receipt_path") != receipt.as_posix():
        raise Ksc2MixedReviewError(["KSC2 mixed candidate receipt path differs from its lock."])

    receipt_value = _json_object(receipt, "KSC2 mixed candidate packet receipt")
    if (
        receipt_value.get("schema_version") != 1
        or receipt_value.get("annotation_state") != "pending"
        or receipt_value.get("language") != "unknown"
        or receipt_value.get("code_switch") != "unknown"
    ):
        raise Ksc2MixedReviewError(
            ["KSC2 mixed candidate receipt must remain pending and unlabelled."]
        )

    rows_by_id: dict[str, dict[str, str]] = {}
    component_counts: Counter[str] = Counter()
    issues: list[str] = []
    for row_number, row in enumerate(rows, start=2):
        identifier = row["annotation_id"]
        if not identifier or identifier in rows_by_id:
            issues.append(
                f"KSC2 mixed candidate packet has duplicate/empty ID at row {row_number}."
            )
            continue
        if (
            row["component"] not in KSC2_MIXED_ANNOTATION_COMPONENTS
            or row["source_name"] != "ksc2_v1"
            or row["annotation_state"] != "pending"
            or row["language"] != "unknown"
            or row["code_switch"] != "unknown"
        ):
            issues.append(
                f"KSC2 mixed candidate packet row {row_number} is not an immutable candidate."
            )
        rows_by_id[identifier] = row
        component_counts[row["component"]] += 1
    if issues:
        raise Ksc2MixedReviewError(issues)
    expected_counts = dict(sorted(component_counts.items()))
    if receipt_value.get("candidate_count") != len(rows) or lock_value.get(
        "candidate_count"
    ) != len(rows):
        raise Ksc2MixedReviewError(
            ["KSC2 mixed candidate packet count differs from a receipt/lock."]
        )
    if (
        _counts(receipt_value.get("candidate_counts_by_component"), "packet receipt counts")
        != expected_counts
    ):
        raise Ksc2MixedReviewError(["KSC2 mixed candidate packet counts differ from its receipt."])
    if (
        _counts(lock_value.get("candidate_counts_by_component"), "packet lock counts")
        != expected_counts
    ):
        raise Ksc2MixedReviewError(["KSC2 mixed candidate packet counts differ from its lock."])
    expected_components = sorted(KSC2_MIXED_ANNOTATION_COMPONENTS)
    if (
        receipt_value.get("priority_components") != expected_components
        or lock_value.get("priority_components") != expected_components
    ):
        raise Ksc2MixedReviewError(
            ["KSC2 mixed candidate priority components differ from the lock."]
        )
    return CandidatePacket(
        rows=tuple(rows),
        rows_by_id=rows_by_id,
        packet_sha256=packet_hash,
        receipt_sha256=receipt_hash,
        lock_sha256=lock_hash,
    )


def curated_mixed_rows(
    packet: CandidatePacket,
    reviewed_at: str,
    *,
    decisions: Sequence[ReviewDecision] = CURATED_MIXED_DECISIONS,
    review_method: str = "single_ai_transcript_semantic_review_v1",
    reviewer: str = "codex_language_review_v1",
) -> list[dict[str, str]]:
    """Materialise only the explicit positive review list; unlisted rows stay unknown."""

    seen: set[str] = set()
    result: list[dict[str, str]] = []
    if not review_method or not reviewer:
        raise Ksc2MixedReviewError(["Review method and reviewer must not be empty."])
    for decision in decisions:
        if decision.annotation_id in seen:
            raise Ksc2MixedReviewError([f"Duplicate curated decision: {decision.annotation_id}."])
        seen.add(decision.annotation_id)
        source = packet.rows_by_id.get(decision.annotation_id)
        if source is None:
            raise Ksc2MixedReviewError(
                [f"Curated decision is absent from the locked packet: {decision.annotation_id}."]
            )
        tokens = tokenize_transcript(source["transcript"])
        if not decision.ru_indices or not decision.kk_indices:
            raise Ksc2MixedReviewError(
                [f"Curated decision lacks both language evidence: {decision.annotation_id}."]
            )
        positions = decision.ru_indices + decision.kk_indices
        if min(positions) < 1 or max(positions) > len(tokens):
            raise Ksc2MixedReviewError(
                [f"Curated decision has an out-of-range token index: {decision.annotation_id}."]
            )
        if set(decision.ru_indices).intersection(decision.kk_indices):
            raise Ksc2MixedReviewError(
                [
                    "Curated decision overlaps Russian/Kazakh token indices: "
                    f"{decision.annotation_id}."
                ]
            )
        ru_tokens = tuple(tokens[index - 1] for index in decision.ru_indices)
        kk_tokens = tuple(tokens[index - 1] for index in decision.kk_indices)
        if not all(
            any(character in _KAZAKH_SPECIFIC for character in token) for token in kk_tokens
        ):
            raise Ksc2MixedReviewError(
                [
                    "Curated Kazakh evidence lacks a Kazakh-specific token: "
                    f"{decision.annotation_id}."
                ]
            )
        result.append(
            {
                "annotation_id": source["annotation_id"],
                "component": source["component"],
                "audio_relative_path": source["audio_relative_path"],
                "audio_sha256": source["audio_sha256"],
                "archive_audio_member": source["archive_audio_member"],
                "archive_transcript_member": source["archive_transcript_member"],
                "transcript": source["transcript"],
                "transcript_sha256": source["transcript_sha256"],
                "source_name": source["source_name"],
                "source_license": source["source_license"],
                "archive_sha256": source["archive_sha256"],
                "source_lock_sha256": source["source_lock_sha256"],
                "candidate_packet_sha256": packet.packet_sha256,
                "candidate_receipt_sha256": packet.receipt_sha256,
                "language": "mixed",
                "code_switch": "true",
                "review_method": review_method,
                "reviewer": reviewer,
                "ru_evidence_token_indices": ",".join(str(index) for index in decision.ru_indices),
                "ru_evidence_tokens": " ".join(ru_tokens),
                "kk_evidence_token_indices": ",".join(str(index) for index in decision.kk_indices),
                "kk_evidence_tokens": " ".join(kk_tokens),
                "reviewed_at": reviewed_at,
            }
        )
    if not result:
        raise Ksc2MixedReviewError(["The curated KSC2 mixed review has no positive rows."])
    return result


def write_csv_once(path: Path, fields: Sequence[str], rows: Iterable[Mapping[str, str]]) -> str:
    if path.exists():
        raise Ksc2MixedReviewError([f"Refusing to overwrite output: {path}"])
    if not path.parent.is_dir():
        raise Ksc2MixedReviewError([f"Output parent does not exist: {path.parent}"])
    try:
        with path.open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(fields), extrasaction="raise", lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(rows)
    except OSError as error:
        raise Ksc2MixedReviewError([f"Cannot write output {path}: {error}"]) from error
    return sha256_file(path)


def write_json_once(path: Path, payload: Mapping[str, object]) -> str:
    if path.exists():
        raise Ksc2MixedReviewError([f"Refusing to overwrite receipt: {path}"])
    if not path.parent.is_dir():
        raise Ksc2MixedReviewError([f"Receipt parent does not exist: {path.parent}"])
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    except OSError as error:
        raise Ksc2MixedReviewError([f"Cannot write receipt {path}: {error}"]) from error
    return sha256_file(path)
