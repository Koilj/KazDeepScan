"""Run one isolated, resumable, local model-v4 KK spoof TTS route."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

# These must be set before any route adapter imports Transformers.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("DISABLE_TELEMETRY", "1")

import soundfile as sf  # type: ignore[import-untyped]
import torch

from kds.data.assets import require_valid_assets, sha256_file
from kds.data.licenses import LicenseLedgerError, load_license_ledger, validate_manifest_licenses
from kds.data.manifest import ManifestError, ManifestRow, validate_manifest, write_manifest
from kds.data.research_tts import (
    ResearchTtsError,
    ResearchTtsModel,
    load_research_tts_model_lock,
    verify_research_tts_model_lock,
)
from kds.data.v4_synthesis import (
    V4_KK_ROUTE_LEDGER_SOURCES,
    V4SynthesisError,
    load_v4_kk_spoof_candidates,
    load_verified_v4_transcript,
    v4_kk_spoof_manifest_row,
)
from kds.data.v4_synthesis_run import (
    V4_KK_SYNTHESIS_PROTOCOL_ID,
    V4KkSynthesisPlan,
    V4KkSynthesisRoute,
    V4KkSynthesisTask,
    V4SynthesisJournalRecord,
    V4SynthesisJournalState,
    V4SynthesisProfile,
    append_v4_synthesis_journal,
    build_v4_kk_synthesis_tasks,
    load_v4_kk_synthesis_plan,
    load_v4_synthesis_journal,
    start_v4_synthesis_record,
    terminal_v4_synthesis_record,
    v4_route_runtime_binding_path,
    v4_route_slug,
)

V4_TEXT_INVENTORY_FIELDS = (
    "selection_rank",
    "target_state",
    "candidate_id",
    "pair_id",
    "generator_route_id",
    "generator_family",
    "source_component",
    "archive_transcript_member",
    "transcript_relative_path",
    "transcript_file_sha256",
    "transcript_size_bytes",
    "text_hash",
    "canonical_text_hash",
    "normalized_utf8_bytes",
    "normalized_characters",
    "status",
)

V4_ROUTE_INVENTORY_FIELDS = (
    "selection_rank",
    "target_state",
    "candidate_id",
    "pair_id",
    "generator_route_id",
    "generator_family",
    "model_id",
    "assigned_voice_id",
    "assigned_speaker_id",
    "assigned_emotion_id",
    "terminal_state",
    "actual_voice_id",
    "actual_speaker_id",
    "actual_emotion_id",
    "actual_seed",
    "generation_attempts",
    "retry_errors",
    "raw_relative_path",
    "raw_audio_sha256",
    "duration_s",
    "original_sr",
    "device",
    "error",
)


@dataclass(frozen=True, slots=True)
class _Generation:
    profile: V4SynthesisProfile
    seed: int
    attempts: int
    retry_errors: tuple[str, ...]
    device: str


@dataclass(slots=True)
class _RouteEngine:
    model: ResearchTtsModel
    sample_rate: int
    profiles: tuple[V4SynthesisProfile, ...]
    generate: Callable[[V4KkSynthesisTask, str, Path], _Generation]


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _json_object(path: Path, label: str) -> dict[str, object]:
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise V4SynthesisError(f"Cannot read {label}: {path}") from error
    if not isinstance(value, dict):
        raise V4SynthesisError(f"{label} must be a JSON object.")
    return cast(dict[str, object], value)


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise V4SynthesisError(f"{label} must be a JSON object.")
    return cast(dict[str, object], value)


def _verify_text_packet(
    *,
    plan: V4KkSynthesisPlan,
    candidates: Sequence[Any],
    project_root: Path,
    data_root: Path,
) -> Path:
    """Bind every local transcript file to the plan before a model sees its text."""

    receipt_path = project_root / plan.inputs["text_receipt"].path
    receipt = _json_object(receipt_path, "v4 KK text receipt")
    outputs = _mapping(receipt.get("outputs"), "v4 KK text receipt outputs")
    inventory = _mapping(outputs.get("inventory"), "v4 KK text receipt inventory")
    if (
        receipt.get("protocol_id") != "xlsr-sls-model-v4-kk-spoof-text-materialization-v1"
        or receipt.get("state") != "kk_spoof_texts_verified_synthesis_pending"
        or inventory.get("path") != plan.inputs["text_inventory"].path
        or inventory.get("sha256") != plan.inputs["text_inventory"].sha256
        or inventory.get("rows") != len(candidates)
    ):
        raise V4SynthesisError("v4 KK text receipt does not bind this synthesis input.")
    inventory_path = project_root / plan.inputs["text_inventory"].path
    try:
        with inventory_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or tuple(reader.fieldnames) != V4_TEXT_INVENTORY_FIELDS:
                raise V4SynthesisError("v4 KK text inventory schema is invalid.")
            rows = list(reader)
    except OSError as error:
        raise V4SynthesisError("Cannot read v4 KK text inventory.") from error
    candidate_by_id = {candidate.candidate_id: candidate for candidate in candidates}
    if len(rows) != len(candidates) or len(candidate_by_id) != len(candidates):
        raise V4SynthesisError("v4 KK text inventory row count changed.")
    expected_root = data_root / "raw" / "v4" / "xlsr_sls_model_v4_kk_spoof_texts_v1"
    for row in rows:
        candidate = candidate_by_id.get(row.get("candidate_id", ""))
        if candidate is None:
            raise V4SynthesisError("v4 KK text inventory contains an unknown candidate.")
        expected_relative = (
            Path("raw")
            / "v4"
            / "xlsr_sls_model_v4_kk_spoof_texts_v1"
            / Path(*Path(candidate.transcript_member).relative_to("ISSAI_KSC2").parts)
        ).as_posix()
        if (
            row.get("selection_rank") != str(candidate.selection_rank)
            or row.get("target_state") != candidate.target_state
            or row.get("generator_route_id") != candidate.generator_route_id
            or row.get("generator_family") != candidate.generator_family
            or row.get("archive_transcript_member") != candidate.transcript_member
            or row.get("transcript_relative_path") != expected_relative
            or row.get("text_hash") != candidate.text_hash
            or row.get("canonical_text_hash") != candidate.canonical_text_hash
            or row.get("status") != "verified_for_synthesis"
        ):
            raise V4SynthesisError("v4 KK text inventory candidate binding changed.")
    if not expected_root.is_dir():
        raise V4SynthesisError("v4 KK transcript directory is unavailable.")
    return expected_root


def _select_route_model(
    route: V4KkSynthesisRoute,
    project_root: Path,
) -> tuple[ResearchTtsModel, dict[str, Path], Path]:
    lock_path = project_root / route.model_lock.path
    lock = load_research_tts_model_lock(lock_path)
    matches = [model for model in lock.models if model.model_id == route.model_id]
    if len(matches) != 1:
        raise ResearchTtsError(f"v4 route {route.route_id!r} model lock is inconsistent.")
    model = matches[0]
    if model.generator_family != route.generator_family:
        raise ResearchTtsError(f"v4 route {route.route_id!r} generator family changed.")
    verified = verify_research_tts_model_lock(project_root / route.model_root, lock)
    return model, verified[model.model_id], project_root / route.model_root / model.destination


def _route_profiles(
    route: V4KkSynthesisRoute, project_root: Path
) -> tuple[V4SynthesisProfile, ...]:
    """Read declared controls without importing or loading a model runtime."""

    lock = load_research_tts_model_lock(project_root / route.model_lock.path)
    matches = [model for model in lock.models if model.model_id == route.model_id]
    if len(matches) != 1 or matches[0].generator_family != route.generator_family:
        raise ResearchTtsError(f"v4 route {route.route_id!r} model lock is inconsistent.")
    model = matches[0]
    if route.adapter in {"piper", "mms"}:
        from kds.data.ksc_derived_kk import synthesis_profiles

        return tuple(
            V4SynthesisProfile(voice_id=profile.voice_id, speaker_id=profile.speaker_id)
            for profile in synthesis_profiles(lock)
            if profile.model.model_id == model.model_id
        )
    if route.adapter == "kazemotts":
        from kds.data.kazemotts import load_kazemotts_runtime

        return tuple(
            V4SynthesisProfile(
                voice_id=profile.voice_id,
                speaker_id=profile.speaker_id,
                emotion_id=profile.emotion_id,
            )
            for profile in load_kazemotts_runtime(model).profiles
        )
    if route.adapter == "sparktts":
        from kds.data.sparktts import load_sparktts_runtime

        return tuple(
            V4SynthesisProfile(voice_id=profile.voice_id)
            for profile in load_sparktts_runtime(model).profiles
        )
    raise V4SynthesisError(f"Unsupported v4 synthesis adapter: {route.adapter!r}.")


def _load_engine(
    *,
    route: V4KkSynthesisRoute,
    project_root: Path,
    device_value: str,
    piper_binary: Path,
    stack: ExitStack,
) -> _RouteEngine:
    """Import exactly one historical low-level adapter in a fresh route process."""

    model, verified_paths, model_directory = _select_route_model(route, project_root)
    if route.adapter in {"piper", "mms"}:
        from synthesize_ksc_derived_kk import (
            _load_mms_model,
            _mms_device,
            _synthesize_mms,
            _synthesize_piper,
        )

        from kds.data.ksc_derived_kk import synthesis_profiles

        profiles = tuple(
            V4SynthesisProfile(
                voice_id=profile.voice_id,
                speaker_id=profile.speaker_id,
            )
            for profile in synthesis_profiles(
                load_research_tts_model_lock(project_root / route.model_lock.path)
            )
            if profile.model.model_id == model.model_id
        )
        if not profiles:
            raise ResearchTtsError("v4 Piper/MMS model has no locked synthesis profile.")
        if route.adapter == "piper":
            if not piper_binary.is_file() or not os.access(piper_binary, os.X_OK):
                raise V4SynthesisError(f"Pinned Piper binary is unavailable: {piper_binary}")

            def generate(task: V4KkSynthesisTask, text: str, output: Path) -> _Generation:
                if task.profile.speaker_id is None:
                    raise V4SynthesisError("v4 Piper task has no locked speaker ID.")
                _synthesize_piper(
                    str(piper_binary),
                    model,
                    verified_paths,
                    task.profile.speaker_id,
                    text,
                    output,
                )
                return _Generation(
                    profile=task.profile,
                    seed=task.base_seed,
                    attempts=1,
                    retry_errors=(),
                    device="local_cpu_piper",
                )

            return _RouteEngine(
                model=model,
                sample_rate=22_050,
                profiles=profiles,
                generate=generate,
            )
        mms_device = _mms_device(device_value)
        tokenizer, mms_model = _load_mms_model(model_directory, mms_device)
        sample_rate = int(mms_model.config.sampling_rate)

        def generate(task: V4KkSynthesisTask, text: str, output: Path) -> _Generation:
            _synthesize_mms(tokenizer, mms_model, text, task.base_seed, output)
            return _Generation(
                profile=task.profile,
                seed=task.base_seed,
                attempts=1,
                retry_errors=(),
                device=f"local_{mms_device.type}_mms",
            )

        return _RouteEngine(
            model=model, sample_rate=sample_rate, profiles=profiles, generate=generate
        )
    if route.adapter == "kazemotts":
        from synthesize_ksc_kazemotts import (
            _device as kazemotts_device,
        )
        from synthesize_ksc_kazemotts import (
            _load_kazemotts_models as load_kazemotts_models,
        )
        from synthesize_ksc_kazemotts import (
            _synthesize as synthesize_kazemotts,
        )

        from kds.data.kazemotts import load_kazemotts_runtime

        kazemotts_runtime = load_kazemotts_runtime(model)
        kazemotts_device_value = kazemotts_device(device_value)
        workspace = Path(
            stack.enter_context(
                tempfile.TemporaryDirectory(
                    prefix="kds-v4-kazemotts-", dir=project_root / "artifacts"
                )
            )
        )
        tts, vocoder, convert_text = load_kazemotts_models(
            model=model,
            runtime=kazemotts_runtime,
            verified_paths=verified_paths,
            workspace=workspace,
            device=kazemotts_device_value,
        )
        profiles = tuple(
            V4SynthesisProfile(
                voice_id=profile.voice_id,
                speaker_id=profile.speaker_id,
                emotion_id=profile.emotion_id,
            )
            for profile in kazemotts_runtime.profiles
        )

        def generate(task: V4KkSynthesisTask, text: str, output: Path) -> _Generation:
            if task.profile.speaker_id is None or task.profile.emotion_id is None:
                raise V4SynthesisError("v4 KazEmoTTS task lacks locked profile IDs.")
            synthesize_kazemotts(
                tts=tts,
                vocoder=vocoder,
                convert_text=convert_text,
                text=text,
                speaker_id=task.profile.speaker_id,
                emotion_id=task.profile.emotion_id,
                seed=task.base_seed,
                runtime=kazemotts_runtime,
                device=kazemotts_device_value,
                output=output,
            )
            return _Generation(
                profile=task.profile,
                seed=task.base_seed,
                attempts=1,
                retry_errors=(),
                device=(f"local_{kazemotts_device_value.type}_kazemotts_gradtts_hifigan"),
            )

        return _RouteEngine(
            model=model,
            sample_rate=kazemotts_runtime.sample_rate,
            profiles=profiles,
            generate=generate,
        )
    if route.adapter == "sparktts":
        from synthesize_ksc_sparktts import (
            _RETRYABLE_OUTPUT_ERRORS,
        )
        from synthesize_ksc_sparktts import (
            _attempt_seed as spark_attempt_seed,
        )
        from synthesize_ksc_sparktts import (
            _device as spark_device,
        )
        from synthesize_ksc_sparktts import (
            _load_sparktts_models as load_sparktts_models,
        )
        from synthesize_ksc_sparktts import (
            _synthesize as synthesize_sparktts,
        )

        from kds.data.sparktts import load_sparktts_runtime

        spark_runtime = load_sparktts_runtime(model)
        spark_device_value = spark_device(device_value)
        workspace = Path(
            stack.enter_context(
                tempfile.TemporaryDirectory(
                    prefix="kds-v4-sparktts-", dir=project_root / "artifacts"
                )
            )
        )
        tokenizer, language_model, codec = load_sparktts_models(
            model=model,
            runtime=spark_runtime,
            verified_paths=verified_paths,
            workspace=workspace,
            device=spark_device_value,
        )
        source_profiles = tuple(spark_runtime.profiles)
        profiles = tuple(
            V4SynthesisProfile(voice_id=profile.voice_id) for profile in source_profiles
        )
        profile_by_voice = {profile.voice_id: profile for profile in source_profiles}

        def generate(task: V4KkSynthesisTask, text: str, output: Path) -> _Generation:
            assigned = profile_by_voice[task.profile.voice_id]
            profile_candidates = (
                assigned,
                *(item for item in source_profiles if item.voice_id != assigned.voice_id),
            )[: spark_runtime.profile_attempts]
            profile_errors: list[str] = []
            total_attempts = 0
            for profile in profile_candidates:
                retry_errors: list[str] = []
                for attempt in range(spark_runtime.generation_attempts):
                    total_attempts += 1
                    seed = spark_attempt_seed(task.base_seed, attempt)
                    try:
                        synthesize_sparktts(
                            tokenizer=tokenizer,
                            language_model=language_model,
                            codec=codec,
                            text=text,
                            profile=profile,
                            seed=seed,
                            runtime=spark_runtime,
                            device=spark_device_value,
                            output=output,
                        )
                    except RuntimeError as error:
                        if not str(error).startswith(_RETRYABLE_OUTPUT_ERRORS):
                            raise
                        retry_errors.append(str(error))
                        continue
                    return _Generation(
                        profile=V4SynthesisProfile(voice_id=profile.voice_id),
                        seed=seed,
                        attempts=total_attempts,
                        retry_errors=tuple(profile_errors + retry_errors),
                        device=(f"local_{spark_device_value.type}_sparktts_controlled_bicodec"),
                    )
                profile_errors.append(f"{profile.voice_id}: " + " | ".join(retry_errors))
            raise RuntimeError(
                "Spark-TTS produced no structurally valid output after frozen profile/seed "
                "attempts: " + " || ".join(profile_errors)
            )

        return _RouteEngine(
            model=model,
            sample_rate=spark_runtime.sample_rate,
            profiles=profiles,
            generate=generate,
        )
    raise V4SynthesisError(f"Unsupported v4 synthesis adapter: {route.adapter!r}.")


def _validate_generated_wav(path: Path, expected_sample_rate: int) -> tuple[str, float, int]:
    try:
        info = sf.info(str(path))
    except RuntimeError as error:
        raise RuntimeError(f"Generated output is not readable audio: {error}") from error
    if (
        not path.is_file()
        or str(info.format).lower() != "wav"
        or info.duration <= 0
        or info.samplerate != expected_sample_rate
    ):
        raise RuntimeError(f"Generator produced an invalid WAV: {path}")
    return sha256_file(path), float(info.duration), int(info.samplerate)


def _fatal_runtime_error(error: RuntimeError) -> bool:
    detail = str(error).lower()
    return any(
        token in detail
        for token in (
            "cuda out of memory",
            "cudnn",
            "no cuda",
            "cuda error",
            "device-side",
            "no kernel image",
            "driver version",
        )
    )


def _publish_asset(temp_output: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise V4SynthesisError(f"Refusing to overwrite v4 synthesis output: {destination}")
    try:
        os.link(temp_output, destination)
    except OSError as error:
        raise V4SynthesisError(f"Cannot atomically publish v4 synthesis WAV: {error}") from error


def _runtime_binding(plan: V4KkSynthesisPlan, route: V4KkSynthesisRoute) -> dict[str, object]:
    return {
        "schema_version": 1,
        "protocol_id": "xlsr-sls-model-v4-kk-spoof-synthesis-runtime-v1",
        "plan": {"path": plan.path, "sha256": plan.sha256},
        "route": {
            "route_id": route.route_id,
            "model_lock": asdict(route.model_lock),
            "adapter_source": asdict(route.adapter_source),
        },
        "offline_environment": {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "reference_audio": "prohibited",
            "detector_or_logit_feedback": "prohibited",
        },
    }


def _ensure_runtime_binding(path: Path, binding: Mapping[str, object]) -> None:
    serialized = json.dumps(binding, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != serialized:
            raise V4SynthesisError("Existing v4 synthesis runtime binding differs.")
        return
    path.write_text(serialized, encoding="utf-8")


def _acquire_route_lock(path: Path) -> int:
    try:
        return os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise V4SynthesisError(f"v4 route lock already exists: {path}") from error


def _run_route(
    *,
    plan: V4KkSynthesisPlan,
    route: V4KkSynthesisRoute,
    tasks: Sequence[V4KkSynthesisTask],
    transcript_root: Path,
    project_root: Path,
    data_root: Path,
    device: str,
    preflight_only: bool,
    max_new_items: int | None,
) -> dict[str, object]:
    runtime_root = project_root / plan.runtime_root
    runtime_root.mkdir(parents=True, exist_ok=True)
    binding_path = v4_route_runtime_binding_path(runtime_root, route.route_id)
    _ensure_runtime_binding(binding_path, _runtime_binding(plan, route))
    journal_path = runtime_root / f"{v4_route_slug(route.route_id)}.jsonl"
    lock_path = runtime_root / f"{v4_route_slug(route.route_id)}.lock"
    lock = _acquire_route_lock(lock_path)
    try:
        states = load_v4_synthesis_journal(journal_path, plan, tasks, data_root)
        pending = [state.task for state in states.values() if state.terminal is None]
        pending.sort(key=lambda task: (task.candidate.selection_rank, task.candidate.candidate_id))
        if max_new_items is not None:
            pending = pending[:max_new_items]
        piper_binary = (project_root / plan.piper_binary).resolve(strict=True)
        with ExitStack() as stack:
            engine = _load_engine(
                route=route,
                project_root=project_root,
                device_value=device,
                piper_binary=piper_binary,
                stack=stack,
            )
            if preflight_only:
                return {
                    "status": "preflight_ok",
                    "route": route.route_id,
                    "model_id": engine.model.model_id,
                    "profiles": len(engine.profiles),
                    "cuda_available": torch.cuda.is_available(),
                    "pending": len(pending),
                }
            for completed, task in enumerate(pending, start=1):
                state = states[task.candidate.candidate_id]
                if not state.started:
                    append_v4_synthesis_journal(
                        journal_path, start_v4_synthesis_record(plan, task, _now())
                    )
                output_path = data_root / task.output_relative_path
                temp_directory = output_path.parent
                temp_directory.mkdir(parents=True, exist_ok=True)
                descriptor, temp_name = tempfile.mkstemp(
                    prefix=f".{task.candidate.canonical_text_hash[:12]}-",
                    suffix=".partial.wav",
                    dir=temp_directory,
                )
                os.close(descriptor)
                temp_path = Path(temp_name)
                try:
                    text = load_verified_v4_transcript(task.candidate, transcript_root)
                    generation = engine.generate(task, text, temp_path)
                    digest, duration_s, sample_rate = _validate_generated_wav(
                        temp_path, engine.sample_rate
                    )
                    _publish_asset(temp_path, output_path)
                    record = terminal_v4_synthesis_record(
                        plan=plan,
                        task=task,
                        event="succeeded",
                        timestamp=_now(),
                        actual_profile=generation.profile,
                        actual_seed=generation.seed,
                        generation_attempts=generation.attempts,
                        retry_errors=generation.retry_errors,
                        output_sha256=digest,
                        duration_s=duration_s,
                        original_sr=sample_rate,
                        device=generation.device,
                    )
                    append_v4_synthesis_journal(journal_path, record)
                    states[task.candidate.candidate_id] = V4SynthesisJournalState(
                        task=task, started=True, terminal=record
                    )
                except RuntimeError as error:
                    if _fatal_runtime_error(error):
                        raise
                    record = terminal_v4_synthesis_record(
                        plan=plan,
                        task=task,
                        event="rejected_runtime",
                        timestamp=_now(),
                        actual_profile=None,
                        actual_seed=None,
                        generation_attempts=None,
                        retry_errors=(),
                        error=str(error),
                    )
                    append_v4_synthesis_journal(journal_path, record)
                    states[task.candidate.candidate_id] = V4SynthesisJournalState(
                        task=task, started=True, terminal=record
                    )
                finally:
                    try:
                        temp_path.unlink(missing_ok=True)
                    except OSError:
                        pass
                if completed % 20 == 0 or completed == len(pending):
                    terminal_count = sum(state.terminal is not None for state in states.values())
                    print(
                        json.dumps(
                            {
                                "status": "progress",
                                "route": route.route_id,
                                "completed_this_call": completed,
                                "terminal": terminal_count,
                                "total": len(tasks),
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                        file=sys.stderr,
                        flush=True,
                    )
        terminal_records = [
            state.terminal for state in states.values() if state.terminal is not None
        ]
        if len(terminal_records) != len(tasks):
            return {
                "status": "partial",
                "route": route.route_id,
                "terminal": len(terminal_records),
                "total": len(tasks),
                "journal": journal_path.as_posix(),
            }
        return _publish_route(
            plan=plan,
            route=route,
            tasks=tasks,
            terminals=tuple(terminal_records),
            journal_path=journal_path,
            project_root=project_root,
            data_root=data_root,
        )
    finally:
        os.close(lock)
        try:
            lock_path.unlink()
        except OSError:
            pass


def _publish_route(
    *,
    plan: V4KkSynthesisPlan,
    route: V4KkSynthesisRoute,
    tasks: Sequence[V4KkSynthesisTask],
    terminals: Sequence[V4SynthesisJournalRecord],
    journal_path: Path,
    project_root: Path,
    data_root: Path,
) -> dict[str, object]:
    slug = v4_route_slug(route.route_id)
    output_manifest = (
        project_root / plan.manifest_directory / (f"xlsr_sls_model_v4_kk_spoof_{slug}_raw_v1.csv")
    )
    output_inventory = (
        project_root
        / plan.manifest_directory
        / (f"xlsr_sls_model_v4_kk_spoof_{slug}_synthesis_inventory_v1.csv")
    )
    output_receipt = (
        project_root
        / plan.receipt_directory
        / (f"xlsr_sls_model_v4_kk_spoof_{slug}_synthesis_v1.json")
    )
    outputs = (output_manifest, output_inventory, output_receipt)
    if any(path.exists() for path in outputs) or any(not path.parent.is_dir() for path in outputs):
        raise V4SynthesisError("v4 route publication outputs already exist or have no parent.")
    terminal_by_id = {record.candidate_id: record for record in terminals}
    if len(terminal_by_id) != len(tasks):
        raise V4SynthesisError("v4 route synthesis accounting is incomplete.")
    model, _verified, _directory = _select_route_model(route, project_root)
    rows: list[ManifestRow] = []
    for task in tasks:
        record = terminal_by_id[task.candidate.candidate_id]
        if record.event != "succeeded":
            continue
        row = v4_kk_spoof_manifest_row(
            candidate=task.candidate,
            relative_path=record.output_relative_path,
            sha256=record.output_sha256,
            duration_s=cast(float, record.duration_s),
            original_sr=cast(int, record.original_sr),
            generator_name=model.generator_name,
            generator_version=model.generator_version,
            voice_id=record.actual_voice_id,
            device=record.device,
            seed=cast(int, record.actual_seed),
            created_at=record.timestamp,
        )
        rows.append(row)
    if not rows:
        raise V4SynthesisError("v4 route generated no WAVs; publication is blocked.")
    validate_manifest(rows)
    ledger = load_license_ledger(project_root / plan.inputs["license_ledger"].path)
    validate_manifest_licenses(rows, ledger)
    require_valid_assets(rows, data_root)
    succeeded = [record for record in terminals if record.event == "succeeded"]
    rejected = [record for record in terminals if record.event == "rejected_runtime"]
    receipt: dict[str, object] = {
        "schema_version": 1,
        "protocol_id": V4_KK_SYNTHESIS_PROTOCOL_ID,
        "created_at": _now(),
        "state": "route_synthesis_complete_qa_pending",
        "route": {
            "route_id": route.route_id,
            "generator_family": route.generator_family,
            "model_id": route.model_id,
            "model_lock": asdict(route.model_lock),
            "adapter_source": asdict(route.adapter_source),
        },
        "bindings": {
            "plan": {"path": plan.path, "sha256": plan.sha256},
            **{name: asdict(binding) for name, binding in sorted(plan.inputs.items())},
            "runtime_journal": {
                "path": journal_path.relative_to(project_root).as_posix(),
                "sha256": sha256_file(journal_path),
                "rows": len(terminals) * 2,
            },
        },
        "outputs": {
            "raw_manifest": {
                "path": output_manifest.relative_to(project_root).as_posix(),
                "rows": len(rows),
            },
            "synthesis_inventory": {
                "path": output_inventory.relative_to(project_root).as_posix(),
                "rows": len(tasks),
            },
        },
        "accounting": {
            "attempted": len(tasks),
            "succeeded": len(succeeded),
            "rejected_runtime": len(rejected),
            "target_succeeded": sum(
                task.candidate.target_state == "target"
                and terminal_by_id[task.candidate.candidate_id].event == "succeeded"
                for task in tasks
            ),
            "reserve_succeeded": sum(
                task.candidate.target_state == "reserve"
                and terminal_by_id[task.candidate.candidate_id].event == "succeeded"
                for task in tasks
            ),
            "voice_counts": dict(
                sorted(Counter(record.actual_voice_id for record in succeeded).items())
            ),
            "rejection_reasons": dict(sorted(Counter(record.error for record in rejected).items())),
        },
        "claims": {
            "local_offline_text_only_tts": True,
            "reference_audio_or_voice_cloning": False,
            "detector_or_logit_feedback": False,
            "output_overwrite": False,
            "audio_qa_completed": False,
            "training_authorized": False,
            "new_dataset_search_performed": False,
        },
        "next_gate": (
            "canonical decode, QA/VAD and historical/current audio leakage screening; "
            "then frozen target/reserve selection without detector feedback"
        ),
    }
    published: list[tuple[Path, Path]] = []
    try:
        with tempfile.TemporaryDirectory(
            prefix="kds-v4-synthesis-publish-", dir=project_root
        ) as stage_name:
            stage = Path(stage_name)
            staged_manifest = stage / output_manifest.name
            staged_inventory = stage / output_inventory.name
            staged_receipt = stage / output_receipt.name
            write_manifest(staged_manifest, rows)
            with staged_inventory.open("x", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=V4_ROUTE_INVENTORY_FIELDS, lineterminator="\n"
                )
                writer.writeheader()
                for task in tasks:
                    record = terminal_by_id[task.candidate.candidate_id]
                    writer.writerow(
                        {
                            "selection_rank": task.candidate.selection_rank,
                            "target_state": task.candidate.target_state,
                            "candidate_id": task.candidate.candidate_id,
                            "pair_id": task.candidate.pair_id,
                            "generator_route_id": task.route_id,
                            "generator_family": task.candidate.generator_family,
                            "model_id": task.model_id,
                            "assigned_voice_id": task.profile.voice_id,
                            "assigned_speaker_id": task.profile.speaker_id or "",
                            "assigned_emotion_id": task.profile.emotion_id or "",
                            "terminal_state": record.event,
                            "actual_voice_id": record.actual_voice_id,
                            "actual_speaker_id": record.actual_speaker_id or "",
                            "actual_emotion_id": record.actual_emotion_id or "",
                            "actual_seed": record.actual_seed or "",
                            "generation_attempts": record.generation_attempts or "",
                            "retry_errors": json.dumps(record.retry_errors, ensure_ascii=False),
                            "raw_relative_path": (
                                record.output_relative_path if record.event == "succeeded" else ""
                            ),
                            "raw_audio_sha256": record.output_sha256,
                            "duration_s": record.duration_s or "",
                            "original_sr": record.original_sr or "",
                            "device": record.device,
                            "error": record.error,
                        }
                    )
            receipt["outputs"] = {
                **cast(dict[str, object], receipt["outputs"]),
                "raw_manifest": {
                    "path": output_manifest.relative_to(project_root).as_posix(),
                    "sha256": sha256_file(staged_manifest),
                    "rows": len(rows),
                },
                "synthesis_inventory": {
                    "path": output_inventory.relative_to(project_root).as_posix(),
                    "sha256": sha256_file(staged_inventory),
                    "rows": len(tasks),
                },
            }
            staged_receipt.write_text(
                json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            for staged, output in (
                (staged_manifest, output_manifest),
                (staged_inventory, output_inventory),
                (staged_receipt, output_receipt),
            ):
                os.link(staged, output)
                published.append((output, staged))
    except (OSError, ManifestError, LicenseLedgerError) as error:
        for output, staged in reversed(published):
            try:
                if output.samefile(staged):
                    output.unlink()
            except OSError:
                pass
        raise V4SynthesisError(f"Cannot publish v4 route synthesis packet: {error}") from error
    return {
        "status": "complete",
        "route": route.route_id,
        "attempted": len(tasks),
        "succeeded": len(succeeded),
        "rejected_runtime": len(rejected),
        "manifest": output_manifest.relative_to(project_root).as_posix(),
        "receipt": output_receipt.relative_to(project_root).as_posix(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path("configs/research/v4/xlsr_sls_model_v4_kk_spoof_synthesis_v1.json"),
    )
    parser.add_argument("--route", required=True)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--max-new-items",
        type=int,
        help="Operational cap for an ordered resumable invocation; does not change frozen quotas.",
    )
    arguments = parser.parse_args()
    try:
        if arguments.max_new_items is not None and arguments.max_new_items <= 0:
            raise V4SynthesisError("max-new-items must be positive when supplied.")
        project_root = arguments.project_root.resolve(strict=True)
        data_root = arguments.data_root.resolve(strict=True)
        plan_path = arguments.plan.resolve(strict=True)
        plan_path.relative_to(project_root)
        plan = load_v4_kk_synthesis_plan(plan_path, project_root)
        route = plan.route(arguments.route)
        candidates = load_v4_kk_spoof_candidates(
            Path(plan.inputs["candidate_csv"].path),
            Path(plan.inputs["selection_governance"].path),
            Path(plan.inputs["source_decode_receipt"].path),
        )
        transcript_root = _verify_text_packet(
            plan=plan,
            candidates=candidates,
            project_root=project_root,
            data_root=data_root,
        )
        ledger = load_license_ledger(project_root / plan.inputs["license_ledger"].path)
        source_id = V4_KK_ROUTE_LEDGER_SOURCES[route.route_id]
        if source_id not in ledger:
            raise LicenseLedgerError([f"v4 route source is absent from ledger: {source_id}"])
        tasks = build_v4_kk_synthesis_tasks(
            candidates, plan, route, _route_profiles(route, project_root)
        )
        result = _run_route(
            plan=plan,
            route=route,
            tasks=tasks,
            transcript_root=transcript_root,
            project_root=project_root,
            data_root=data_root,
            device=arguments.device,
            preflight_only=arguments.preflight_only,
            max_new_items=arguments.max_new_items,
        )
    except (
        LicenseLedgerError,
        ManifestError,
        ResearchTtsError,
        RuntimeError,
        OSError,
        ValueError,
        V4SynthesisError,
    ) as error:
        issues = list(error.issues) if isinstance(error, LicenseLedgerError) else [str(error)]
        print(json.dumps({"status": "error", "issues": issues}, ensure_ascii=False))
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
