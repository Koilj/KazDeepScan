"""Frozen candidate and provenance contracts for XLS-R+SLS model v4 KK synthesis."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import cast

from kds.data.assets import sha256_file
from kds.data.kazemotts import KAZEMOTTS_SOURCE_LICENSE
from kds.data.ksc_derived_kk import KSC_DERIVED_KK_SOURCE_LICENSE
from kds.data.manifest import ManifestRow
from kds.data.sparktts import SPARKTTS_SOURCE_LICENSE
from kds.data.v4_selection import V4_CANDIDATE_FIELDS

V4_KK_ROUTE_FAMILIES = {
    "kk-piper-issai-high-v1": "piper_neural_tts",
    "kk-mms-kaz-v1": "mms_vits_tts",
    "kk-kazemotts-v1": "gradtts_hifigan_emotional_tts",
    "kk-sparktts-v1": "llm_bicodec_controlled_tts",
}

V4_KK_ROUTE_LEDGER_SOURCES = {
    "kk-piper-issai-high-v1": "ksc_derived_kk_v1",
    "kk-mms-kaz-v1": "ksc_derived_kk_v1",
    "kk-kazemotts-v1": "ksc_derived_kk_v2_kazemotts",
    "kk-sparktts-v1": "ksc_derived_kk_v3_sparktts",
}

V4_KK_ROUTE_LICENSES = {
    "kk-piper-issai-high-v1": KSC_DERIVED_KK_SOURCE_LICENSE,
    "kk-mms-kaz-v1": KSC_DERIVED_KK_SOURCE_LICENSE,
    "kk-kazemotts-v1": KAZEMOTTS_SOURCE_LICENSE,
    "kk-sparktts-v1": SPARKTTS_SOURCE_LICENSE,
}


class V4SynthesisError(ValueError):
    """Raised when a planned v4 synthesis asset is not fully contract-bound."""


@dataclass(frozen=True, slots=True)
class V4KkSpoofCandidate:
    selection_rank: int
    target_state: str
    pair_id: str
    candidate_id: str
    source_id: str
    source_lineage_id: str
    source_component: str
    transcript_member: str
    text_hash: str
    canonical_text_hash: str
    parent_group_id: str
    generator_route_id: str
    generator_family: str


def load_v4_kk_spoof_candidates(
    candidate_csv: Path,
    governance_receipt: Path,
    source_decode_receipt: Path,
) -> tuple[V4KkSpoofCandidate, ...]:
    """Load all 7,200 frozen KK spoof candidates after the source gate authorizes synthesis."""

    governance = _json_object(governance_receipt, "v4 selection governance")
    canonical = _mapping(governance.get("canonical_packet"), "canonical_packet")
    csv_binding = _mapping(canonical.get("candidate_csv"), "candidate_csv")
    decode_receipt = _json_object(source_decode_receipt, "v4 source decode receipt")
    decode_claims = _mapping(decode_receipt.get("claims"), "source decode claims")
    balanced = _mapping(decode_receipt.get("balanced_train_decision"), "balanced train decision")
    if (
        governance.get("status") != "canonical_metadata_selection_v2_audio_gate_pending"
        or canonical.get("version") != 2
        or csv_binding.get("path") != candidate_csv.as_posix()
        or csv_binding.get("rows") != 28_800
        or sha256_file(candidate_csv) != csv_binding.get("sha256")
        or decode_receipt.get("state")
        != "source_train_frozen_15000_kk_spoof_synthesis_authorized"
        or decode_claims.get("kk_spoof_synthesis_authorized") is not True
        or decode_claims.get("training_authorized") is not False
        or balanced.get("decision") != "proceed_20k_balanced"
        or balanced.get("pending_kk_spoof_target") != 5_000
    ):
        raise V4SynthesisError("v4 selection/source authorization binding is invalid.")
    try:
        with candidate_csv.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or tuple(reader.fieldnames) != V4_CANDIDATE_FIELDS:
                raise V4SynthesisError("v4 candidate CSV schema is invalid.")
            mappings = list(reader)
    except OSError as error:
        raise V4SynthesisError("Cannot read v4 candidate CSV.") from error
    rows: list[V4KkSpoofCandidate] = []
    for row_number, mapping in enumerate(mappings, start=2):
        if mapping.get("language") != "kk" or mapping.get("label") != "spoof":
            continue
        route = (mapping.get("generator_route_id") or "").strip()
        family = (mapping.get("generator_family") or "").strip()
        transcript = (mapping.get("archive_transcript_member") or "").strip()
        path = PurePosixPath(transcript)
        try:
            rank = int(mapping.get("selection_rank") or "")
        except ValueError as error:
            raise V4SynthesisError(f"Invalid v4 KK candidate rank at row {row_number}.") from error
        text_hash = _sha256(mapping.get("text_hash"), "text hash")
        canonical_text_hash = _sha256(
            mapping.get("canonical_text_hash"), "canonical text hash"
        )
        expected_state = "target" if rank <= 6_000 else "reserve"
        if (
            mapping.get("role") != "train"
            or route not in V4_KK_ROUTE_FAMILIES
            or family != V4_KK_ROUTE_FAMILIES[route]
            or mapping.get("target_state") != expected_state
            or rank not in range(1, 7_201)
            or mapping.get("source_id") != f"xlsr_sls_model_v4_kk_spoof:{route}"
            or mapping.get("source_lineage_id")
            != f"ksc2_v1:nonlegacy_train:text_only:{route}:train_only"
            or mapping.get("parent_group_id") != f"xlsr_sls_model_v4_kk_spoof:route:{route}"
            or path.is_absolute()
            or ".." in path.parts
            or len(path.parts) < 4
            or path.parts[0] != "ISSAI_KSC2"
            or path.parts[1] != "Train"
            or path.suffix.lower() != ".txt"
            or text_hash != canonical_text_hash
            or mapping.get("raw_audio_sha256") != ""
            or mapping.get("decoded_audio_sha256") != ""
            or mapping.get("asset_state") != "synthesis_planned_not_authorized"
        ):
            raise V4SynthesisError(
                f"v4 KK spoof candidate row {row_number} violates the frozen contract."
            )
        rows.append(
            V4KkSpoofCandidate(
                selection_rank=rank,
                target_state=expected_state,
                pair_id=(mapping.get("pair_id") or "").strip(),
                candidate_id=(mapping.get("candidate_id") or "").strip(),
                source_id=(mapping.get("source_id") or "").strip(),
                source_lineage_id=(mapping.get("source_lineage_id") or "").strip(),
                source_component=(mapping.get("source_component") or "").strip(),
                transcript_member=transcript,
                text_hash=text_hash,
                canonical_text_hash=canonical_text_hash,
                parent_group_id=(mapping.get("parent_group_id") or "").strip(),
                generator_route_id=route,
                generator_family=family,
            )
        )
    route_counts = Counter(row.generator_route_id for row in rows)
    target_counts = Counter(
        row.generator_route_id for row in rows if row.target_state == "target"
    )
    if (
        len(rows) != 7_200
        or route_counts != Counter({route: 1_800 for route in V4_KK_ROUTE_FAMILIES})
        or target_counts != Counter({route: 1_500 for route in V4_KK_ROUTE_FAMILIES})
        or len({row.candidate_id for row in rows}) != len(rows)
        or len({row.transcript_member for row in rows}) != len(rows)
        or len({row.canonical_text_hash for row in rows}) != len(rows)
    ):
        raise V4SynthesisError("v4 KK spoof route quotas or uniqueness changed.")
    return tuple(rows)


def load_verified_v4_transcript(candidate: V4KkSpoofCandidate, transcript_root: Path) -> str:
    path = PurePosixPath(candidate.transcript_member).relative_to("ISSAI_KSC2")
    root = transcript_root.resolve(strict=True)
    transcript_path = (root / Path(*path.parts)).resolve(strict=True)
    try:
        transcript_path.relative_to(root)
        text = " ".join(transcript_path.read_text(encoding="utf-8").split())
    except (OSError, UnicodeDecodeError, ValueError) as error:
        raise V4SynthesisError(
            f"Cannot read exact v4 transcript for {candidate.candidate_id!r}."
        ) from error
    if not text or hashlib.sha256(text.encode("utf-8")).hexdigest() != candidate.text_hash:
        raise V4SynthesisError(
            f"v4 transcript content changed for {candidate.candidate_id!r}."
        )
    return text


def v4_kk_spoof_manifest_row(
    *,
    candidate: V4KkSpoofCandidate,
    relative_path: str,
    sha256: str,
    duration_s: float,
    original_sr: int,
    generator_name: str,
    generator_version: str,
    voice_id: str,
    device: str,
    seed: int,
    created_at: str,
) -> ManifestRow:
    route = candidate.generator_route_id
    return ManifestRow(
        sample_id=candidate.candidate_id,
        relative_path=relative_path,
        sha256=sha256,
        split="train",
        label="spoof",
        language="kk",
        code_switch="unknown",
        parent_group_id=candidate.parent_group_id,
        source_name=V4_KK_ROUTE_LEDGER_SOURCES[route],
        source_license=V4_KK_ROUTE_LICENSES[route],
        rights_basis=(
            "Frozen KSC2 CC-BY-4.0 text plus hash-pinned local text-only TTS route; "
            "personal research; no reference audio or cloning"
        ),
        speaker_pseudo_id=f"xlsr-sls-model-v4:{route}:voice:{voice_id}",
        text_id=f"xlsr-sls-model-v4:text:{candidate.canonical_text_hash}",
        text_hash=candidate.canonical_text_hash,
        duration_s=duration_s,
        generator_family=candidate.generator_family,
        generator_name=generator_name,
        generator_version=generator_version,
        voice_id=voice_id,
        clone_consent_id="",
        device=device,
        capture_route=f"v4_offline_text_only_tts:{route}",
        original_sr=original_sr,
        codec="wav",
        augmentation_chain="",
        augmentation_seed=str(seed),
        created_at=created_at,
    )


def _json_object(path: Path, label: str) -> dict[str, object]:
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise V4SynthesisError(f"Cannot read {label}: {path}") from error
    return _mapping(value, label)


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise V4SynthesisError(f"{label} must be a JSON object.")
    return cast(dict[str, object], value)


def _sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise V4SynthesisError(f"v4 candidate {label} is invalid.")
    return value
