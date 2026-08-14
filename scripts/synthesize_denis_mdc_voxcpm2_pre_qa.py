"""Attempt one offline official VoxCPM2 WAV for every frozen Denis ready text."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, cast

import numpy as np
import soundfile as sf  # type: ignore[import-untyped]

from kds.data.assets import require_valid_assets, sha256_file
from kds.data.denis import inspect_denis_archive
from kds.data.denis_voxcpm2_candidate import (
    DENIS_VOXCPM2_SOURCE_ID,
    DENIS_VOXCPM2_SYNTHESIS_PROTOCOL_ID,
    DENIS_VOXCPM2_TEXT_BINDING_PROTOCOL_ID,
    DenisVoxCPM2CandidateError,
    denis_voxcpm2_spoof_row,
)
from kds.data.licenses import LicenseLedgerError, load_license_ledger, validate_manifest_licenses
from kds.data.manifest import (
    ManifestError,
    ManifestRow,
    load_manifest,
    validate_manifest,
    write_manifest,
)
from kds.data.research_tts import ResearchTtsError, load_research_tts_model_lock
from kds.data.voxcpm2 import (
    VOXCPM2_MODEL_REVISION,
    VOXCPM2_SOURCE_REVISION,
    audit_voxcpm2_artifacts,
)
from kds.data.voxcpm2_text_only import (
    VOXCPM2_FIXED_SEED,
    BoundText,
    bind_text,
    local_model_load_kwargs,
    offline_environment,
    synthesize_text_only,
)
from kds.eval.voxcpm2_smoke import (
    audit_waveform,
    install_python_network_guard,
    installed_distribution_audit,
)

_MODEL_ID = "openbmb_voxcpm2_official_text_only"
_BOUND_ROWS = 64
_HEX = frozenset("0123456789abcdef")
_EXPECTED_UV_LOCK_SHA256 = "fc066d21d09656c5060892baad096c53af6774c0947fad5bf6c676ea73c47c9b"
_EXPECTED_WRAPPER_SHA256 = "3dcc290594a6af2670203b1dfd9ff500b96dbaf425b5ebe21011abfe57f12cbd"
_EXPECTED_DISTRIBUTION_FINGERPRINT = (
    "60158bb4e2dd9dbef6a0defdf517b98b3c5df21811af13e0b0a48c25de1e5779"
)
_EXPECTED_KEY_DISTRIBUTIONS = {
    "torch": "2.10.0",
    "torchaudio": "2.10.0",
    "torchcodec": "0.10.0",
    "transformers": "5.3.0",
    "voxcpm": "2.0.3.post23+gee8161e9e",
}


class DenisVoxCPM2SynthesisError(ValueError):
    """Raised when candidate synthesis leaves the committed one-attempt contract."""


class CandidateCallAudit:
    """Bind one expected text per call and prohibit a second attempt for any row."""

    def __init__(self, model: Any, max_calls: int) -> None:
        self.model = model
        self.max_calls = max_calls
        self.calls = 0
        self.expected_text_hash: str | None = None
        self.sanitized_kwargs: dict[str, object] | None = None

    def expect(self, text_hash: str) -> None:
        if self.expected_text_hash is not None:
            raise DenisVoxCPM2SynthesisError("Prior candidate call did not finish cleanly.")
        self.expected_text_hash = text_hash

    def generate(self, **kwargs: object) -> Any:
        expected = self.expected_text_hash
        self.expected_text_hash = None
        self.calls += 1
        if expected is None or self.calls > self.max_calls:
            raise DenisVoxCPM2SynthesisError("Unexpected or excess VoxCPM2 generation call.")
        text = kwargs.get("text")
        if (
            not isinstance(text, str)
            or hashlib.sha256(text.encode("utf-8")).hexdigest() != expected
        ):
            raise DenisVoxCPM2SynthesisError("Generation text differs from the active binding.")
        sanitized = dict(kwargs)
        sanitized["text"] = {
            "sha256": expected,
            "utf8_bytes": len(text.encode("utf-8")),
            "plaintext_persisted": False,
        }
        if self.sanitized_kwargs is None:
            self.sanitized_kwargs = sanitized
        else:
            prior = dict(self.sanitized_kwargs)
            current = dict(sanitized)
            prior.pop("text")
            current.pop("text")
            if current != prior:
                raise DenisVoxCPM2SynthesisError("Generation kwargs changed between bound rows.")
        return self.model.generate(**kwargs)


def _object(path: Path, label: str) -> Mapping[str, object]:
    try:
        payload: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DenisVoxCPM2SynthesisError(f"Cannot read {label}: {path}.") from error
    if not isinstance(payload, dict):
        raise DenisVoxCPM2SynthesisError(f"{label} must be a JSON object.")
    return cast(Mapping[str, object], payload)


def _sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or set(value).difference(_HEX):
        raise DenisVoxCPM2SynthesisError(f"{label} must be a lowercase SHA-256 digest.")
    return value


def _git_output(source_root: Path, *arguments: str) -> str:
    process = subprocess.run(
        ["git", "-C", str(source_root), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    if process.returncode != 0:
        raise DenisVoxCPM2SynthesisError(
            f"Cannot verify runtime source checkout: {process.stderr.strip()}"
        )
    return process.stdout.strip()


def require_runtime(source_root: Path, project_root: Path) -> dict[str, object]:
    """Require the exact isolated runtime proven by the one permitted smoke."""

    if sys.version_info[:2] != (3, 12):
        raise DenisVoxCPM2SynthesisError(
            f"Candidate synthesis requires isolated CPython 3.12, got {sys.version}."
        )
    if os.environ.get("KDS_NETWORK_NAMESPACE") != "bwrap_unshare_net":
        raise DenisVoxCPM2SynthesisError(
            "Candidate synthesis must be launched inside bwrap --unshare-net."
        )
    if _git_output(source_root, "rev-parse", "HEAD") != VOXCPM2_SOURCE_REVISION:
        raise DenisVoxCPM2SynthesisError("Runtime source checkout is not the pinned commit.")
    if _git_output(source_root, "status", "--short"):
        raise DenisVoxCPM2SynthesisError("Runtime source checkout is dirty.")
    if sha256_file(source_root / "uv.lock") != _EXPECTED_UV_LOCK_SHA256:
        raise DenisVoxCPM2SynthesisError("Runtime uv.lock hash mismatch.")
    if sha256_file(project_root / "src/kds/data/voxcpm2_text_only.py") != (
        _EXPECTED_WRAPPER_SHA256
    ):
        raise DenisVoxCPM2SynthesisError("Frozen text-only wrapper hash mismatch.")
    distributions = installed_distribution_audit()
    versions = dict(distributions.distributions)
    if distributions.fingerprint != _EXPECTED_DISTRIBUTION_FINGERPRINT or any(
        versions.get(name) != expected for name, expected in _EXPECTED_KEY_DISTRIBUTIONS.items()
    ):
        raise DenisVoxCPM2SynthesisError("Installed runtime distributions changed.")
    return {
        "python_version": sys.version,
        "uv_lock_sha256": _EXPECTED_UV_LOCK_SHA256,
        "installed_distribution_count": len(distributions.distributions),
        "installed_distribution_fingerprint": distributions.fingerprint,
        "key_distributions": _EXPECTED_KEY_DISTRIBUTIONS,
    }


def _safe_member(value: str) -> None:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or "\\" in value:
        raise DenisVoxCPM2SynthesisError("Denis TAR contains an unsafe member path.")


def _read_texts(archive_path: Path, selected: Sequence[Mapping[str, object]]) -> dict[str, str]:
    wanted: dict[str, str] = {}
    for index, row in enumerate(selected, start=1):
        member_stem = row.get("member_stem")
        sample_id = row.get("sample_id")
        if not isinstance(member_stem, str) or not isinstance(sample_id, str):
            raise DenisVoxCPM2SynthesisError(f"Binding row {index} lacks member identity.")
        wanted[f"{member_stem}.txt"] = sample_id
    texts: dict[str, str] = {}
    try:
        with tarfile.open(archive_path, "r:gz") as archive:
            for member in archive:
                _safe_member(member.name)
                sample_id = wanted.get(member.name)
                if sample_id is None:
                    continue
                handle = archive.extractfile(member)
                if not member.isfile() or handle is None:
                    raise DenisVoxCPM2SynthesisError(
                        f"Selected transcript is not a regular file: {member.name}."
                    )
                texts[sample_id] = handle.read().decode("utf-8")
    except (OSError, UnicodeDecodeError, tarfile.TarError) as error:
        raise DenisVoxCPM2SynthesisError("Cannot recover exact bound Denis text.") from error
    if set(texts) != set(wanted.values()):
        raise DenisVoxCPM2SynthesisError("Pinned Denis archive lacks a bound transcript.")
    return texts


def _binding_digest(rows: Sequence[Mapping[str, object]]) -> str:
    fields = (
        "selection_rank",
        "sample_id",
        "member_stem",
        "text_id",
        "literal_text_sha256",
        "collapse_whitespace_text_sha256",
        "nfkc_collapse_whitespace_text_sha256",
        "literal_text_utf8_bytes",
        "collapse_whitespace_text_utf8_bytes",
        "ready_audio_sha256",
    )
    return hashlib.sha256(
        "\n".join("\t".join(str(row[field]) for field in fields) for row in rows).encode("utf-8")
    ).hexdigest()


def require_text_binding(
    path: Path,
    *,
    project_root: Path,
    base_manifest: Path,
    selection_csv: Path,
    selection_receipt: Path,
    model_lock: Path,
    output_directory: Path,
    output_manifest: Path,
    output_receipt: Path,
) -> tuple[Mapping[str, object], ...]:
    """Accept only the committed 64-row binding for these exact future outputs."""

    binding = _object(path, "Denis VoxCPM2 text binding")
    inputs = binding.get("inputs")
    programs = binding.get("frozen_programs")
    contract = binding.get("synthesis_contract")
    outputs = binding.get("output_contract")
    claims = binding.get("claims")
    rows = binding.get("rows")
    if not all(
        isinstance(value, Mapping) for value in (inputs, programs, contract, outputs, claims)
    ) or not isinstance(rows, list):
        raise DenisVoxCPM2SynthesisError("Text-binding receipt structure is invalid.")
    inputs = cast(Mapping[str, object], inputs)
    programs = cast(Mapping[str, object], programs)
    contract = cast(Mapping[str, object], contract)
    outputs = cast(Mapping[str, object], outputs)
    claims = cast(Mapping[str, object], claims)
    required_inputs = {
        "selection_csv": (selection_csv, 79),
        "selection_receipt": (selection_receipt, None),
        "ready_manifest": (base_manifest, _BOUND_ROWS),
        "model_lock": (model_lock, None),
    }
    for name, (expected_path, expected_rows) in required_inputs.items():
        value = inputs.get(name)
        if (
            not isinstance(value, Mapping)
            or value.get("path") != expected_path.as_posix()
            or _sha256(value.get("sha256"), f"{name} SHA-256") != sha256_file(expected_path)
            or (expected_rows is not None and value.get("rows") != expected_rows)
        ):
            raise DenisVoxCPM2SynthesisError(f"Text binding input changed: {name}.")
    runner = programs.get("synthesis_runner")
    runner_path = Path(__file__).resolve()
    runner_project_path = runner_path.relative_to(project_root).as_posix()
    if (
        binding.get("schema_version") != 1
        or binding.get("protocol_id") != DENIS_VOXCPM2_TEXT_BINDING_PROTOCOL_ID
        or not isinstance(runner, Mapping)
        or runner.get("path") != runner_project_path
        or _sha256(runner.get("sha256"), "Synthesis runner SHA-256") != sha256_file(runner_path)
        or outputs.get("raw_directory") != output_directory.as_posix()
        or outputs.get("raw_manifest") != output_manifest.as_posix()
        or outputs.get("synthesis_receipt") != output_receipt.as_posix()
        or contract.get("bound_rows") != _BOUND_ROWS
        or contract.get("model_loads") != 1
        or contract.get("attempts_per_bound_text") != 1
        or contract.get("total_generation_calls_required") != _BOUND_ROWS
        or contract.get("fixed_seed_per_call") != VOXCPM2_FIXED_SEED
        or contract.get("pass_collapse_whitespace_text_only") is not True
        or contract.get("external_text_normalizer_or_stress_model") != "forbidden"
        or contract.get("reference_audio") is not None
        or contract.get("prompt_audio") is not None
        or contract.get("prompt_text") is not None
        or contract.get("voice_cloning") is not False
        or contract.get("lora") is not None
        or contract.get("load_denoiser") is not False
        or contract.get("normalize") is not False
        or contract.get("denoise") is not False
        or contract.get("retry_badcase") is not False
        or contract.get("resynthesis_after_failure") != "forbidden"
        or contract.get("replacement_reselection_or_backfill") != "forbidden"
        or claims.get("text_binding_frozen") is not True
        or claims.get("synthetic_audio_generated") is not False
        or claims.get("detector_inference_performed") is not False
        or claims.get("detector_inference_authorized") is not False
        or claims.get("training_data_overlap_unverified") is not True
        or claims.get("single_bonafide_speaker") is not True
        or claims.get("speaker_independent") is not False
        or len(rows) != _BOUND_ROWS
    ):
        raise DenisVoxCPM2SynthesisError("Text binding does not authorize candidate synthesis.")
    typed_rows: list[Mapping[str, object]] = []
    sample_ids: set[str] = set()
    text_hashes: set[str] = set()
    for index, value in enumerate(rows, start=1):
        if not isinstance(value, Mapping):
            raise DenisVoxCPM2SynthesisError(f"Text binding row {index} is invalid.")
        sample_id = value.get("sample_id")
        text_hash = value.get("collapse_whitespace_text_sha256")
        if (
            not isinstance(sample_id, str)
            or sample_id in sample_ids
            or _sha256(text_hash, f"Binding row {index} canonical hash") != text_hash
            or text_hash in text_hashes
            or not isinstance(value.get("selection_rank"), int)
            or not isinstance(value.get("member_stem"), str)
            or not value.get("member_stem")
            or not isinstance(value.get("literal_text_utf8_bytes"), int)
            or not isinstance(value.get("collapse_whitespace_text_utf8_bytes"), int)
            or not 1 <= cast(int, value["collapse_whitespace_text_utf8_bytes"]) <= 4096
        ):
            raise DenisVoxCPM2SynthesisError(
                f"Text binding row {index} violates the one-to-one contract."
            )
        typed_rows.append(value)
        sample_ids.add(sample_id)
        text_hashes.add(text_hash)
    if binding.get("text_binding_sha256") != _binding_digest(typed_rows):
        raise DenisVoxCPM2SynthesisError("Text-binding digest differs from its rows.")
    return tuple(typed_rows)


def _relative_to_data_root(data_root: Path, path: Path) -> str:
    try:
        return path.resolve(strict=True).relative_to(data_root.resolve(strict=True)).as_posix()
    except ValueError as error:
        raise DenisVoxCPM2SynthesisError("Synthetic output path escapes data root.") from error


def _validate_base_rows(rows: Sequence[ManifestRow]) -> None:
    if (
        len(rows) != _BOUND_ROWS
        or len({row.sample_id for row in rows}) != _BOUND_ROWS
        or len({row.text_hash for row in rows}) != _BOUND_ROWS
        or any(
            row.split != "ood"
            or row.label != "bonafide"
            or row.language != "ru"
            or row.source_name != "denis_1_0_mdc"
            for row in rows
        )
    ):
        raise DenisVoxCPM2SynthesisError("Synthesis base is not the 64-row Denis ready set.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--runtime-source-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--denis-archive", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--license-ledger", type=Path, required=True)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--selection-csv", type=Path, required=True)
    parser.add_argument("--selection-receipt", type=Path, required=True)
    parser.add_argument("--model-lock", type=Path, required=True)
    parser.add_argument("--text-binding", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-receipt", type=Path, required=True)
    parser.add_argument("--created-at", required=True)
    arguments = parser.parse_args()
    try:
        datetime.fromisoformat(arguments.created_at.replace("Z", "+00:00"))
        if (
            arguments.output_directory.exists()
            or arguments.output_manifest.exists()
            or arguments.output_receipt.exists()
            or arguments.output_manifest == arguments.output_receipt
            or not arguments.output_manifest.parent.is_dir()
            or not arguments.output_receipt.parent.is_dir()
        ):
            raise DenisVoxCPM2SynthesisError(
                "Synthesis directory, manifest and receipt must all be distinct and new."
            )
        project_root = arguments.project_root.resolve(strict=True)
        source_root = arguments.runtime_source_root.resolve(strict=True)
        data_root = arguments.data_root.resolve(strict=True)
        runtime_audit = require_runtime(source_root, project_root)
        for name, value in offline_environment().items():
            os.environ[name] = value
        network_attempts = install_python_network_guard()
        base_rows = tuple(load_manifest(arguments.base_manifest))
        validate_manifest(base_rows)
        _validate_base_rows(base_rows)
        ledger = load_license_ledger(arguments.license_ledger)
        validate_manifest_licenses(base_rows, ledger)
        require_valid_assets(base_rows, data_root)
        if DENIS_VOXCPM2_SOURCE_ID not in ledger:
            raise LicenseLedgerError(["Official VoxCPM2 source is absent from the ledger."])
        binding_rows = require_text_binding(
            arguments.text_binding,
            project_root=project_root,
            base_manifest=arguments.base_manifest,
            selection_csv=arguments.selection_csv,
            selection_receipt=arguments.selection_receipt,
            model_lock=arguments.model_lock,
            output_directory=arguments.output_directory,
            output_manifest=arguments.output_manifest,
            output_receipt=arguments.output_receipt,
        )
        binding_by_id = {cast(str, row["sample_id"]): row for row in binding_rows}
        if set(binding_by_id) != {row.sample_id for row in base_rows}:
            raise DenisVoxCPM2SynthesisError("Binding does not cover exactly the base rows.")
        inspect_denis_archive(arguments.denis_archive.resolve(strict=True))
        selected_ready = tuple(
            sorted(binding_rows, key=lambda row: cast(int, row["selection_rank"]))
        )
        texts = _read_texts(arguments.denis_archive, selected_ready)
        base_by_id = {row.sample_id: row for row in base_rows}
        for row in selected_ready:
            sample_id = cast(str, row["sample_id"])
            text = texts[sample_id]
            actual = bind_text(text)
            bound = binding_by_id[sample_id]
            base = base_by_id[sample_id]
            if (
                actual.literal_sha256 != bound.get("literal_text_sha256")
                or actual.collapse_whitespace_sha256 != bound.get("collapse_whitespace_text_sha256")
                or len(text.encode("utf-8")) != bound.get("literal_text_utf8_bytes")
                or base.text_id != bound.get("text_id")
                or base.text_hash != actual.collapse_whitespace_sha256
                or base.sha256 != bound.get("ready_audio_sha256")
            ):
                raise DenisVoxCPM2SynthesisError(
                    f"Bound source text or ready row changed: {sample_id!r}."
                )
        lock = load_research_tts_model_lock(arguments.model_lock)
        if len(lock.models) != 1 or lock.models[0].model_id != _MODEL_ID:
            raise ResearchTtsError("Synthesis requires the one official VoxCPM2 model.")
        model_metadata = lock.models[0]
        artifact_started = time.monotonic()
        artifact_audit = audit_voxcpm2_artifacts(
            arguments.model_root.resolve(strict=True),
            arguments.source_archive.resolve(strict=True),
        )
        artifact_seconds = time.monotonic() - artifact_started

        import torch
        from voxcpm import VoxCPM  # type: ignore[import-not-found]

        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise DenisVoxCPM2SynthesisError("Pinned runtime has no CUDA/BF16 device.")
        torch.cuda.reset_peak_memory_stats()
        load_kwargs = local_model_load_kwargs(arguments.model_root)
        load_started = time.monotonic()
        model = VoxCPM.from_pretrained(**load_kwargs)
        load_seconds = time.monotonic() - load_started
        audited_model = CandidateCallAudit(model, _BOUND_ROWS)
        arguments.output_directory.parent.mkdir(parents=True, exist_ok=True)
        with (
            tempfile.TemporaryDirectory(
                prefix=".kds-denis-voxcpm2-assets-", dir=arguments.output_directory.parent
            ) as stage_name,
            tempfile.TemporaryDirectory(
                prefix="kds-denis-voxcpm2-metadata-", dir=arguments.output_receipt.parent
            ) as metadata_name,
        ):
            stage = Path(stage_name)
            stage_assets = stage / "assets"
            stage_assets.mkdir()
            rows: list[ManifestRow] = []
            staged_rows: list[ManifestRow] = []
            generated: list[dict[str, object]] = []
            failed: list[dict[str, object]] = []
            generation_total_seconds = 0.0
            for index, selected in enumerate(selected_ready, start=1):
                selected_sample_id = cast(str, selected["sample_id"])
                selection_rank = cast(int, selected["selection_rank"])
                base = base_by_id[selected_sample_id]
                text = texts[selected_sample_id]
                binding = BoundText(
                    literal_sha256=cast(
                        str,
                        binding_by_id[selected_sample_id]["literal_text_sha256"],
                    ),
                    collapse_whitespace_sha256=cast(
                        str,
                        binding_by_id[selected_sample_id]["collapse_whitespace_text_sha256"],
                    ),
                )
                file_key = hashlib.sha256(base.sample_id.encode("utf-8")).hexdigest()[:20]
                filename = f"{file_key}-{base.text_hash[:12]}.wav"
                staged_output = stage_assets / filename
                call_started = time.monotonic()
                try:
                    canonical_hash = cast(
                        str,
                        binding_by_id[selected_sample_id]["collapse_whitespace_text_sha256"],
                    )
                    audited_model.expect(canonical_hash)
                    waveform_value = synthesize_text_only(audited_model, text, binding)
                    call_seconds = time.monotonic() - call_started
                    generation_total_seconds += call_seconds
                    waveform = np.asarray(waveform_value, dtype=np.float32)
                    waveform_audit = audit_waveform(waveform, artifact_audit.output_sample_rate_hz)
                    sf.write(
                        staged_output,
                        waveform,
                        artifact_audit.output_sample_rate_hz,
                        subtype="PCM_16",
                    )
                    info = sf.info(staged_output)
                    if (
                        info.samplerate != 48_000
                        or info.channels != 1
                        or info.frames != waveform_audit.frames
                        or info.format != "WAV"
                        or info.subtype != "PCM_16"
                    ):
                        raise DenisVoxCPM2SynthesisError(
                            "Stored candidate is not the expected mono PCM-16 48-kHz WAV."
                        )
                    final_output = arguments.output_directory / filename
                    final_row = denis_voxcpm2_spoof_row(
                        base_row=base,
                        model=model_metadata,
                        binding=binding,
                        relative_path=(final_output.resolve().relative_to(data_root).as_posix()),
                        sha256=sha256_file(staged_output),
                        duration_s=float(info.duration),
                        created_at=arguments.created_at,
                    )
                    rows.append(final_row)
                    staged_rows.append(
                        ManifestRow(
                            **{
                                **asdict(final_row),
                                "relative_path": _relative_to_data_root(data_root, staged_output),
                            }
                        )
                    )
                    generated.append(
                        {
                            "selection_rank": selection_rank,
                            "base_sample_id": base.sample_id,
                            "spoof_sample_id": final_row.sample_id,
                            "text_id": final_row.text_id,
                            "text_hash": final_row.text_hash,
                            "relative_path": final_row.relative_path,
                            "audio_sha256": final_row.sha256,
                            "duration_s": final_row.duration_s,
                            "frames": waveform_audit.frames,
                            "generation_seconds": f"{call_seconds:.6f}",
                            "fixed_seed": VOXCPM2_FIXED_SEED,
                        }
                    )
                except (OSError, RuntimeError, TypeError, ValueError) as error:
                    generation_total_seconds += time.monotonic() - call_started
                    staged_output.unlink(missing_ok=True)
                    failed.append(
                        {
                            "selection_rank": selection_rank,
                            "base_sample_id": base.sample_id,
                            "text_id": base.text_id,
                            "text_hash": base.text_hash,
                            "error_type": type(error).__name__,
                            "detail": str(error)[-1200:],
                        }
                    )
                print(
                    json.dumps(
                        {
                            "status": "running",
                            "attempted_rows": index,
                            "generated_rows": len(rows),
                            "failed_rows": len(failed),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            if audited_model.calls != _BOUND_ROWS or audited_model.expected_text_hash is not None:
                raise DenisVoxCPM2SynthesisError(
                    "Candidate process did not make exactly one call per bound text."
                )
            if not rows or len(rows) + len(failed) != _BOUND_ROWS:
                raise DenisVoxCPM2SynthesisError(
                    "Candidate attempts do not account for all 64 bound texts."
                )
            if network_attempts:
                raise DenisVoxCPM2SynthesisError(
                    f"Upstream attempted network access: {network_attempts}"
                )
            validate_manifest(rows)
            validate_manifest_licenses(rows, ledger)
            validate_manifest(staged_rows)
            validate_manifest_licenses(staged_rows, ledger)
            require_valid_assets(staged_rows, data_root)
            metadata_stage = Path(metadata_name)
            staged_manifest = metadata_stage / arguments.output_manifest.name
            staged_receipt = metadata_stage / arguments.output_receipt.name
            write_manifest(staged_manifest, rows)
            receipt = {
                "schema_version": 1,
                "protocol_id": DENIS_VOXCPM2_SYNTHESIS_PROTOCOL_ID,
                "created_at": arguments.created_at,
                "base_manifest": {
                    "path": arguments.base_manifest.as_posix(),
                    "sha256": sha256_file(arguments.base_manifest),
                    "rows": len(base_rows),
                },
                "text_binding": {
                    "path": arguments.text_binding.as_posix(),
                    "sha256": sha256_file(arguments.text_binding),
                    "rows": len(binding_rows),
                },
                "model_lock": {
                    "path": arguments.model_lock.as_posix(),
                    "sha256": sha256_file(arguments.model_lock),
                    "model_id": model_metadata.model_id,
                    "model_revision": VOXCPM2_MODEL_REVISION,
                    "source_revision": VOXCPM2_SOURCE_REVISION,
                },
                "runtime": runtime_audit,
                "network_policy": {
                    "outer_namespace": "bwrap --unshare-net",
                    "offline_environment": dict(offline_environment()),
                    "python_socket_guard_installed_before_upstream_import": True,
                    "observed_upstream_network_attempts": len(network_attempts),
                },
                "artifact_revalidation": {
                    "seconds": f"{artifact_seconds:.6f}",
                    "model_files": artifact_audit.model_files,
                    "model_bytes": artifact_audit.model_bytes,
                    "model_inventory_sha256": artifact_audit.model_inventory_sha256,
                    "source_archive_files": artifact_audit.source_archive_files,
                    "output_sample_rate_hz": artifact_audit.output_sample_rate_hz,
                },
                "model_load": {
                    "count": 1,
                    "kwargs": load_kwargs,
                    "seconds": f"{load_seconds:.6f}",
                    "cuda_device_name": torch.cuda.get_device_name(0),
                    "cuda_runtime": torch.version.cuda,
                    "torch_version": torch.__version__,
                    "bf16_supported": torch.cuda.is_bf16_supported(),
                },
                "output_manifest": {
                    "path": arguments.output_manifest.as_posix(),
                    "sha256": sha256_file(staged_manifest),
                    "rows": len(rows),
                },
                "generation_policy": {
                    "device": "cuda:0",
                    "fixed_default_voice_identity": "unknown_not_claimed",
                    "model_loads": 1,
                    "bound_rows": _BOUND_ROWS,
                    "attempted_rows": audited_model.calls,
                    "successful_rows": len(rows),
                    "failed_attempt_rows": len(failed),
                    "exactly_one_attempt_per_bound_text": True,
                    "fixed_seed_per_call": VOXCPM2_FIXED_SEED,
                    "generation_kwargs_first_call_sanitized": audited_model.sanitized_kwargs,
                    "generation_total_seconds": f"{generation_total_seconds:.6f}",
                    "max_cuda_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "max_cuda_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
                    "reference_or_prompt_audio_used": False,
                    "voice_cloning_or_lora_used": False,
                    "semantic_normalizer_used": False,
                    "denoiser_used": False,
                    "retry_or_resynthesis_used": False,
                    "post_selection_replacement_or_backfill": False,
                    "resynthesis_after_failure": "forbidden",
                },
                "claims": {
                    "synthetic_audio_generated": True,
                    "technical_decode_qa_vad_performed": False,
                    "acoustic_review_performed": False,
                    "binary_pairing_performed": False,
                    "detector_inference_performed": False,
                    "detector_inference_authorized": False,
                    "external_source_and_generator_family_holdout": True,
                    "training_data_overlap_unverified": True,
                    "single_bonafide_speaker": True,
                    "speaker_independent": False,
                    "speaker_robust": False,
                },
                "generated": generated,
                "failed_attempts": failed,
            }
            staged_receipt.write_text(
                json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            if (
                arguments.output_directory.exists()
                or arguments.output_manifest.exists()
                or arguments.output_receipt.exists()
            ):
                raise DenisVoxCPM2SynthesisError("A synthesis output appeared during staging.")
            stage_assets.replace(arguments.output_directory)
            require_valid_assets(rows, data_root)
            staged_manifest.replace(arguments.output_manifest)
            staged_receipt.replace(arguments.output_receipt)
    except (
        DenisVoxCPM2CandidateError,
        LicenseLedgerError,
        ManifestError,
        OSError,
        ResearchTtsError,
        subprocess.SubprocessError,
        ValueError,
    ) as error:
        issues = (
            list(error.issues)
            if isinstance(error, (LicenseLedgerError, ManifestError))
            else [str(error)]
        )
        print(json.dumps({"status": "error", "issues": issues}, ensure_ascii=False))
        return 2
    print(
        json.dumps(
            {
                "status": "ok",
                "attempted_rows": audited_model.calls,
                "generated_rows": len(rows),
                "failed_rows": len(failed),
                "manifest": str(arguments.output_manifest),
                "receipt": str(arguments.output_receipt),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
