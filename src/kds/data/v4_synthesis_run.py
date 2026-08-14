"""Strict plan, task and local-journal primitives for model-v4 KK TTS synthesis."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import cast

from kds.data.assets import sha256_file
from kds.data.v4_synthesis import (
    V4_KK_ROUTE_FAMILIES,
    V4KkSpoofCandidate,
    V4SynthesisError,
)

V4_KK_SYNTHESIS_PROTOCOL_ID = "xlsr-sls-model-v4-kk-spoof-synthesis-v1"
V4_KK_SYNTHESIS_PLAN_SCHEMA_VERSION = 1
V4_KK_SYNTHESIS_JOURNAL_SCHEMA_VERSION = 1
V4_KK_SYNTHESIS_OUTPUT_ROOT = "raw/v4/xlsr_sls_model_v4_kk_spoof_raw_v1"
V4_KK_SYNTHESIS_RUNTIME_ROOT = "artifacts/v4/xlsr_sls_model_v4_kk_spoof_synthesis_v1"
V4_KK_SYNTHESIS_MANIFEST_DIRECTORY = "data/manifests/v4"
V4_KK_SYNTHESIS_RECEIPT_DIRECTORY = "docs/artifacts/v4"


@dataclass(frozen=True, slots=True)
class V4SynthesisBinding:
    """A hash-pinned project-relative input."""

    path: str
    sha256: str
    rows: int | None = None


@dataclass(frozen=True, slots=True)
class V4KkSynthesisRoute:
    route_id: str
    generator_family: str
    adapter: str
    model_lock: V4SynthesisBinding
    model_root: str
    model_id: str
    adapter_source: V4SynthesisBinding


@dataclass(frozen=True, slots=True)
class V4KkSynthesisPlan:
    path: str
    sha256: str
    created_at: str
    inputs: Mapping[str, V4SynthesisBinding]
    output_root: str
    runtime_root: str
    manifest_directory: str
    receipt_directory: str
    base_seed: str
    attempted_per_route: int
    target_candidates_per_route: int
    reserve_candidates_per_route: int
    frozen_ready_per_route: int
    piper_binary: str
    routes: tuple[V4KkSynthesisRoute, ...]

    def route(self, route_id: str) -> V4KkSynthesisRoute:
        for route in self.routes:
            if route.route_id == route_id:
                return route
        raise V4SynthesisError(f"Unknown v4 synthesis route: {route_id!r}.")


@dataclass(frozen=True, slots=True)
class V4SynthesisProfile:
    """A fixed local TTS control, never an asserted human identity."""

    voice_id: str
    speaker_id: int | None = None
    emotion_id: int | None = None


@dataclass(frozen=True, slots=True)
class V4KkSynthesisTask:
    candidate: V4KkSpoofCandidate
    route_id: str
    model_id: str
    profile: V4SynthesisProfile
    base_seed: int
    output_relative_path: str


@dataclass(frozen=True, slots=True)
class V4SynthesisJournalRecord:
    """One append-only event; terminal events make a candidate immutable."""

    schema_version: int
    protocol_id: str
    plan_sha256: str
    route_id: str
    event: str
    candidate_id: str
    text_hash: str
    model_id: str
    assigned_voice_id: str
    assigned_speaker_id: int | None
    assigned_emotion_id: int | None
    base_seed: int
    output_relative_path: str
    timestamp: str
    actual_voice_id: str
    actual_speaker_id: int | None
    actual_emotion_id: int | None
    actual_seed: int | None
    generation_attempts: int | None
    retry_errors: tuple[str, ...]
    output_sha256: str
    duration_s: float | None
    original_sr: int | None
    device: str
    error: str

    def as_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True, slots=True)
class V4SynthesisJournalState:
    task: V4KkSynthesisTask
    started: bool
    terminal: V4SynthesisJournalRecord | None


def load_v4_kk_synthesis_plan(path: Path, project_root: Path) -> V4KkSynthesisPlan:
    """Load a no-network v4 synthesis contract and verify all tracked input hashes."""

    raw = _json_object(path, "v4 KK synthesis plan")
    _expect_exact_keys(
        raw,
        {
            "schema_version",
            "protocol_id",
            "created_at",
            "state",
            "inputs",
            "outputs",
            "generation",
            "routes",
            "prohibitions",
        },
        "v4 KK synthesis plan",
    )
    if (
        raw["schema_version"] != V4_KK_SYNTHESIS_PLAN_SCHEMA_VERSION
        or raw["protocol_id"] != V4_KK_SYNTHESIS_PROTOCOL_ID
        or raw["state"] != "frozen_pre_synthesis"
    ):
        raise V4SynthesisError("v4 KK synthesis plan version or state is invalid.")
    created_at = _timestamp(raw["created_at"], "v4 KK synthesis plan created_at")
    inputs_raw = _mapping(raw["inputs"], "v4 KK synthesis plan inputs")
    input_names = {
        "candidate_csv",
        "selection_governance",
        "source_decode_receipt",
        "text_inventory",
        "text_receipt",
        "license_ledger",
        "synthesis_contract_module",
        "synthesis_runtime_module",
        "runner_script",
    }
    _expect_exact_keys(inputs_raw, input_names, "v4 KK synthesis plan inputs")
    inputs = {
        name: _load_binding(inputs_raw[name], f"v4 synthesis input {name}", project_root)
        for name in sorted(input_names)
    }
    outputs = _mapping(raw["outputs"], "v4 KK synthesis plan outputs")
    _expect_exact_keys(
        outputs,
        {"output_root", "runtime_root", "manifest_directory", "receipt_directory"},
        "v4 KK synthesis plan outputs",
    )
    output_root = _safe_project_relative(outputs["output_root"], "v4 output_root")
    runtime_root = _safe_project_relative(outputs["runtime_root"], "v4 runtime_root")
    manifest_directory = _safe_project_relative(
        outputs["manifest_directory"], "v4 manifest_directory"
    )
    receipt_directory = _safe_project_relative(outputs["receipt_directory"], "v4 receipt_directory")
    if (
        output_root != V4_KK_SYNTHESIS_OUTPUT_ROOT
        or runtime_root != V4_KK_SYNTHESIS_RUNTIME_ROOT
        or manifest_directory != V4_KK_SYNTHESIS_MANIFEST_DIRECTORY
        or receipt_directory != V4_KK_SYNTHESIS_RECEIPT_DIRECTORY
    ):
        raise V4SynthesisError("v4 synthesis output namespaces must use the frozen v1 paths.")
    generation = _mapping(raw["generation"], "v4 KK synthesis plan generation")
    _expect_exact_keys(
        generation,
        {
            "base_seed",
            "attempted_per_route",
            "target_candidates_per_route",
            "reserve_candidates_per_route",
            "frozen_ready_per_route",
            "piper_binary",
        },
        "v4 KK synthesis plan generation",
    )
    base_seed = _nonempty_string(generation["base_seed"], "v4 base seed")
    attempted_per_route = _positive_int(generation["attempted_per_route"], "v4 attempted_per_route")
    target_candidates_per_route = _positive_int(
        generation["target_candidates_per_route"], "v4 target candidates"
    )
    reserve_candidates_per_route = _positive_int(
        generation["reserve_candidates_per_route"], "v4 reserve candidates"
    )
    frozen_ready_per_route = _positive_int(
        generation["frozen_ready_per_route"], "v4 frozen ready quota"
    )
    if (
        attempted_per_route != target_candidates_per_route + reserve_candidates_per_route
        or frozen_ready_per_route > target_candidates_per_route
    ):
        raise V4SynthesisError("v4 route quotas are inconsistent.")
    piper_binary = _safe_project_relative(generation["piper_binary"], "v4 piper binary")
    routes_raw = raw["routes"]
    if not isinstance(routes_raw, list):
        raise V4SynthesisError("v4 synthesis routes must be a list.")
    routes = tuple(
        _load_route(value, position, project_root)
        for position, value in enumerate(routes_raw, start=1)
    )
    if (
        len(routes) != len(V4_KK_ROUTE_FAMILIES)
        or {route.route_id for route in routes} != set(V4_KK_ROUTE_FAMILIES)
        or len({route.model_id for route in routes}) != len(routes)
    ):
        raise V4SynthesisError("v4 synthesis route set is incomplete or ambiguous.")
    _expect_exact_keys(
        _mapping(raw["prohibitions"], "v4 KK synthesis plan prohibitions"),
        {
            "network",
            "reference_audio",
            "voice_cloning",
            "detector_or_logit_feedback",
            "output_overwrite",
            "new_dataset_search",
        },
        "v4 KK synthesis plan prohibitions",
    )
    plan_relative_path = path.resolve(strict=True).relative_to(project_root).as_posix()
    return V4KkSynthesisPlan(
        path=_safe_project_relative(plan_relative_path, "plan"),
        sha256=sha256_file(path),
        created_at=created_at,
        inputs=inputs,
        output_root=output_root,
        runtime_root=runtime_root,
        manifest_directory=manifest_directory,
        receipt_directory=receipt_directory,
        base_seed=base_seed,
        attempted_per_route=attempted_per_route,
        target_candidates_per_route=target_candidates_per_route,
        reserve_candidates_per_route=reserve_candidates_per_route,
        frozen_ready_per_route=frozen_ready_per_route,
        piper_binary=piper_binary,
        routes=routes,
    )


def build_v4_kk_synthesis_tasks(
    candidates: Sequence[V4KkSpoofCandidate],
    plan: V4KkSynthesisPlan,
    route: V4KkSynthesisRoute,
    profiles: Sequence[V4SynthesisProfile],
) -> tuple[V4KkSynthesisTask, ...]:
    """Assign profiles in frozen rank order and derive one unique content-addressed WAV path."""

    if not profiles or len({profile.voice_id for profile in profiles}) != len(profiles):
        raise V4SynthesisError("v4 synthesis profiles must have unique non-empty voice IDs.")
    route_candidates = sorted(
        (candidate for candidate in candidates if candidate.generator_route_id == route.route_id),
        key=lambda candidate: (candidate.selection_rank, candidate.candidate_id),
    )
    if (
        len(route_candidates) != plan.attempted_per_route
        or sum(candidate.target_state == "target" for candidate in route_candidates)
        != plan.target_candidates_per_route
        or any(
            candidate.generator_family != route.generator_family for candidate in route_candidates
        )
    ):
        raise V4SynthesisError("v4 route candidates do not match the frozen synthesis plan.")
    tasks: list[V4KkSynthesisTask] = []
    seen_paths: set[str] = set()
    for index, candidate in enumerate(route_candidates):
        profile = profiles[index % len(profiles)]
        relative_path = v4_synthesis_output_relative_path(candidate)
        seed = v4_synthesis_seed(plan.base_seed, candidate, route.model_id, profile.voice_id)
        if relative_path in seen_paths:
            raise V4SynthesisError("v4 synthesis output path collision.")
        seen_paths.add(relative_path)
        tasks.append(
            V4KkSynthesisTask(
                candidate=candidate,
                route_id=route.route_id,
                model_id=route.model_id,
                profile=profile,
                base_seed=seed,
                output_relative_path=relative_path,
            )
        )
    return tuple(tasks)


def v4_synthesis_seed(
    base_seed: str, candidate: V4KkSpoofCandidate, model_id: str, voice_id: str
) -> int:
    """Use only frozen input identity and explicit controls for every stochastic generator."""

    digest = hashlib.sha256(
        f"{base_seed}:{candidate.candidate_id}:{model_id}:{voice_id}".encode()
    ).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


def v4_synthesis_output_relative_path(candidate: V4KkSpoofCandidate) -> str:
    """Return a route-local immutable WAV path independent of unsafe source filenames."""

    digest = candidate.canonical_text_hash
    _require_sha256(digest, "candidate canonical text hash")
    if candidate.generator_route_id not in V4_KK_ROUTE_FAMILIES:
        raise V4SynthesisError("Unknown v4 route for output path.")
    return f"{V4_KK_SYNTHESIS_OUTPUT_ROOT}/{candidate.generator_route_id}/{digest[:2]}/{digest}.wav"


def start_v4_synthesis_record(
    plan: V4KkSynthesisPlan, task: V4KkSynthesisTask, timestamp: str
) -> V4SynthesisJournalRecord:
    return _journal_record(
        plan,
        task,
        event="started",
        timestamp=timestamp,
        actual_profile=None,
        actual_seed=None,
        generation_attempts=None,
        retry_errors=(),
        output_sha256="",
        duration_s=None,
        original_sr=None,
        device="",
        error="",
    )


def terminal_v4_synthesis_record(
    *,
    plan: V4KkSynthesisPlan,
    task: V4KkSynthesisTask,
    event: str,
    timestamp: str,
    actual_profile: V4SynthesisProfile | None,
    actual_seed: int | None,
    generation_attempts: int | None,
    retry_errors: Sequence[str],
    output_sha256: str = "",
    duration_s: float | None = None,
    original_sr: int | None = None,
    device: str = "",
    error: str = "",
) -> V4SynthesisJournalRecord:
    if event not in {"succeeded", "rejected_runtime"}:
        raise V4SynthesisError("v4 synthesis terminal event is invalid.")
    if event == "succeeded":
        _require_sha256(output_sha256, "v4 synthesis output hash")
        if duration_s is None or duration_s <= 0 or original_sr is None or original_sr <= 0:
            raise V4SynthesisError("v4 successful synthesis record lacks audio metrics.")
        if actual_profile is None or actual_seed is None or not device:
            raise V4SynthesisError("v4 successful synthesis record lacks provenance.")
    elif output_sha256 or duration_s is not None or original_sr is not None:
        raise V4SynthesisError("v4 rejected synthesis record may not bind an output asset.")
    return _journal_record(
        plan,
        task,
        event=event,
        timestamp=timestamp,
        actual_profile=actual_profile,
        actual_seed=actual_seed,
        generation_attempts=generation_attempts,
        retry_errors=tuple(retry_errors),
        output_sha256=output_sha256,
        duration_s=duration_s,
        original_sr=original_sr,
        device=device,
        error=error,
    )


def load_v4_synthesis_journal(
    path: Path,
    plan: V4KkSynthesisPlan,
    tasks: Sequence[V4KkSynthesisTask],
    data_root: Path,
) -> dict[str, V4SynthesisJournalState]:
    """Load an append-only route journal and fail closed on any task or asset mismatch."""

    task_by_id = {task.candidate.candidate_id: task for task in tasks}
    if len(task_by_id) != len(tasks):
        raise V4SynthesisError("v4 synthesis task IDs are not unique.")
    states = {
        candidate_id: V4SynthesisJournalState(task=task, started=False, terminal=None)
        for candidate_id, task in task_by_id.items()
    }
    if not path.exists():
        return states
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise V4SynthesisError(f"Cannot read v4 synthesis journal: {path}") from error
    for line_number, line in enumerate(lines, start=1):
        record = _parse_journal_record(line, line_number)
        task = task_by_id.get(record.candidate_id)
        if task is None or not _record_matches_task(record, plan, task):
            raise V4SynthesisError(
                f"v4 synthesis journal row {line_number} does not match this frozen task."
            )
        state = states[record.candidate_id]
        if record.event == "started":
            if state.started or state.terminal is not None:
                raise V4SynthesisError(
                    f"v4 synthesis journal has duplicate start for {record.candidate_id!r}."
                )
            states[record.candidate_id] = V4SynthesisJournalState(
                task=task, started=True, terminal=None
            )
            continue
        if not state.started or state.terminal is not None:
            raise V4SynthesisError(
                f"v4 synthesis journal terminal event ordering is invalid at row {line_number}."
            )
        _validate_terminal_record(record, data_root)
        states[record.candidate_id] = V4SynthesisJournalState(
            task=task, started=True, terminal=record
        )
    for state in states.values():
        asset = data_root / state.task.output_relative_path
        if state.terminal is None and asset.exists():
            raise V4SynthesisError(
                "v4 synthesis found an output without a terminal journal record; "
                "manual review is required instead of regeneration."
            )
    return states


def append_v4_synthesis_journal(path: Path, record: V4SynthesisJournalRecord) -> None:
    """Append and fsync one immutable local journal event."""

    if not path.parent.is_dir():
        raise V4SynthesisError("v4 synthesis journal parent does not exist.")
    try:
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(record.as_json() + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as error:
        raise V4SynthesisError("Cannot append v4 synthesis journal.") from error


def v4_route_slug(route_id: str) -> str:
    if route_id not in V4_KK_ROUTE_FAMILIES:
        raise V4SynthesisError("Cannot create output name for unknown v4 route.")
    return route_id.replace("-", "_")


def v4_route_runtime_binding_path(runtime_root: Path, route_id: str) -> Path:
    """Return the immutable binding path isolated to one frozen TTS route."""

    return runtime_root / f"{v4_route_slug(route_id)}.binding.json"


def _load_route(value: object, position: int, project_root: Path) -> V4KkSynthesisRoute:
    raw = _mapping(value, f"v4 synthesis route {position}")
    _expect_exact_keys(
        raw,
        {
            "route_id",
            "generator_family",
            "adapter",
            "model_lock",
            "model_root",
            "model_id",
            "adapter_source",
        },
        f"v4 synthesis route {position}",
    )
    route_id = _nonempty_string(raw["route_id"], f"v4 synthesis route {position} id")
    family = _nonempty_string(raw["generator_family"], f"v4 synthesis route {route_id} family")
    adapter = _nonempty_string(raw["adapter"], f"v4 synthesis route {route_id} adapter")
    expected_family = V4_KK_ROUTE_FAMILIES.get(route_id)
    if expected_family != family or adapter not in {
        "piper",
        "mms",
        "kazemotts",
        "sparktts",
    }:
        raise V4SynthesisError(f"v4 synthesis route {route_id!r} is not recognized.")
    return V4KkSynthesisRoute(
        route_id=route_id,
        generator_family=family,
        adapter=adapter,
        model_lock=_load_binding(
            raw["model_lock"], f"v4 route {route_id} model lock", project_root
        ),
        model_root=_safe_project_relative(raw["model_root"], f"v4 route {route_id} model root"),
        model_id=_nonempty_string(raw["model_id"], f"v4 route {route_id} model ID"),
        adapter_source=_load_binding(
            raw["adapter_source"], f"v4 route {route_id} adapter source", project_root
        ),
    )


def _load_binding(value: object, label: str, project_root: Path) -> V4SynthesisBinding:
    raw = _mapping(value, label)
    _expect_exact_keys(raw, {"path", "sha256", "rows"}, label)
    path = _safe_project_relative(raw["path"], f"{label} path")
    digest = _sha256(raw["sha256"], f"{label} SHA-256")
    rows = raw["rows"]
    if rows is not None and (not isinstance(rows, int) or isinstance(rows, bool) or rows <= 0):
        raise V4SynthesisError(f"{label} rows must be null or a positive integer.")
    target = (project_root / path).resolve(strict=True)
    try:
        target.relative_to(project_root)
    except ValueError as error:
        raise V4SynthesisError(f"{label} escapes the project root.") from error
    if sha256_file(target) != digest:
        raise V4SynthesisError(f"{label} SHA-256 no longer matches the frozen plan.")
    return V4SynthesisBinding(path=path, sha256=digest, rows=rows)


def _journal_record(
    plan: V4KkSynthesisPlan,
    task: V4KkSynthesisTask,
    *,
    event: str,
    timestamp: str,
    actual_profile: V4SynthesisProfile | None,
    actual_seed: int | None,
    generation_attempts: int | None,
    retry_errors: Sequence[str],
    output_sha256: str,
    duration_s: float | None,
    original_sr: int | None,
    device: str,
    error: str,
) -> V4SynthesisJournalRecord:
    return V4SynthesisJournalRecord(
        schema_version=V4_KK_SYNTHESIS_JOURNAL_SCHEMA_VERSION,
        protocol_id=V4_KK_SYNTHESIS_PROTOCOL_ID,
        plan_sha256=plan.sha256,
        route_id=task.route_id,
        event=event,
        candidate_id=task.candidate.candidate_id,
        text_hash=task.candidate.canonical_text_hash,
        model_id=task.model_id,
        assigned_voice_id=task.profile.voice_id,
        assigned_speaker_id=task.profile.speaker_id,
        assigned_emotion_id=task.profile.emotion_id,
        base_seed=task.base_seed,
        output_relative_path=task.output_relative_path,
        timestamp=_timestamp(timestamp, "v4 synthesis journal timestamp"),
        actual_voice_id="" if actual_profile is None else actual_profile.voice_id,
        actual_speaker_id=None if actual_profile is None else actual_profile.speaker_id,
        actual_emotion_id=None if actual_profile is None else actual_profile.emotion_id,
        actual_seed=actual_seed,
        generation_attempts=generation_attempts,
        retry_errors=tuple(retry_errors),
        output_sha256=output_sha256,
        duration_s=duration_s,
        original_sr=original_sr,
        device=device,
        error=error,
    )


def _parse_journal_record(line: str, line_number: int) -> V4SynthesisJournalRecord:
    try:
        value: object = json.loads(line)
        raw = _mapping(value, f"v4 synthesis journal row {line_number}")
        _expect_exact_keys(
            raw,
            set(V4SynthesisJournalRecord.__dataclass_fields__),
            f"v4 synthesis journal row {line_number}",
        )
        retries = raw["retry_errors"]
        if not isinstance(retries, list) or any(not isinstance(item, str) for item in retries):
            raise TypeError
        return V4SynthesisJournalRecord(
            schema_version=cast(int, raw["schema_version"]),
            protocol_id=cast(str, raw["protocol_id"]),
            plan_sha256=cast(str, raw["plan_sha256"]),
            route_id=cast(str, raw["route_id"]),
            event=cast(str, raw["event"]),
            candidate_id=cast(str, raw["candidate_id"]),
            text_hash=cast(str, raw["text_hash"]),
            model_id=cast(str, raw["model_id"]),
            assigned_voice_id=cast(str, raw["assigned_voice_id"]),
            assigned_speaker_id=cast(int | None, raw["assigned_speaker_id"]),
            assigned_emotion_id=cast(int | None, raw["assigned_emotion_id"]),
            base_seed=cast(int, raw["base_seed"]),
            output_relative_path=cast(str, raw["output_relative_path"]),
            timestamp=cast(str, raw["timestamp"]),
            actual_voice_id=cast(str, raw["actual_voice_id"]),
            actual_speaker_id=cast(int | None, raw["actual_speaker_id"]),
            actual_emotion_id=cast(int | None, raw["actual_emotion_id"]),
            actual_seed=cast(int | None, raw["actual_seed"]),
            generation_attempts=cast(int | None, raw["generation_attempts"]),
            retry_errors=tuple(retries),
            output_sha256=cast(str, raw["output_sha256"]),
            duration_s=cast(float | None, raw["duration_s"]),
            original_sr=cast(int | None, raw["original_sr"]),
            device=cast(str, raw["device"]),
            error=cast(str, raw["error"]),
        )
    except (json.JSONDecodeError, TypeError, V4SynthesisError) as error:
        raise V4SynthesisError(f"Invalid v4 synthesis journal row {line_number}.") from error


def _record_matches_task(
    record: V4SynthesisJournalRecord, plan: V4KkSynthesisPlan, task: V4KkSynthesisTask
) -> bool:
    if (
        record.schema_version != V4_KK_SYNTHESIS_JOURNAL_SCHEMA_VERSION
        or record.protocol_id != V4_KK_SYNTHESIS_PROTOCOL_ID
        or record.plan_sha256 != plan.sha256
        or record.route_id != task.route_id
        or record.text_hash != task.candidate.canonical_text_hash
        or record.model_id != task.model_id
        or record.assigned_voice_id != task.profile.voice_id
        or record.assigned_speaker_id != task.profile.speaker_id
        or record.assigned_emotion_id != task.profile.emotion_id
        or record.base_seed != task.base_seed
        or record.output_relative_path != task.output_relative_path
        or record.event not in {"started", "succeeded", "rejected_runtime"}
    ):
        return False
    try:
        _timestamp(record.timestamp, "v4 synthesis journal timestamp")
    except V4SynthesisError:
        return False
    return True


def _validate_terminal_record(record: V4SynthesisJournalRecord, data_root: Path) -> None:
    target = data_root / record.output_relative_path
    if record.event == "succeeded":
        try:
            _require_sha256(record.output_sha256, "v4 synthesis journal output hash")
        except V4SynthesisError as error:
            raise V4SynthesisError("v4 synthesis success output hash is invalid.") from error
        if (
            not record.actual_voice_id
            or record.actual_seed is None
            or record.generation_attempts is None
            or record.generation_attempts <= 0
            or record.duration_s is None
            or record.duration_s <= 0
            or record.original_sr is None
            or record.original_sr <= 0
            or not record.device
            or not target.is_file()
            or sha256_file(target) != record.output_sha256
        ):
            raise V4SynthesisError("v4 synthesis success journal asset binding failed.")
        return
    if (
        record.output_sha256
        or record.duration_s is not None
        or record.original_sr is not None
        or not record.error
        or target.exists()
    ):
        raise V4SynthesisError("v4 synthesis rejected journal record is inconsistent.")


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


def _expect_exact_keys(raw: Mapping[str, object], expected: set[str], label: str) -> None:
    missing = sorted(expected.difference(raw))
    unknown = sorted(set(raw).difference(expected))
    if missing or unknown:
        details: list[str] = []
        if missing:
            details.append("missing: " + ", ".join(missing))
        if unknown:
            details.append("unknown: " + ", ".join(unknown))
        raise V4SynthesisError(f"{label} has " + "; ".join(details) + ".")


def _safe_project_relative(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise V4SynthesisError(f"{label} must be a non-empty project-relative path.")
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts or "\\" in value or value in {".", ""}:
        raise V4SynthesisError(f"{label} is not a safe project-relative path.")
    return pure.as_posix()


def _nonempty_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise V4SynthesisError(f"{label} must be a non-empty string.")
    return value.strip()


def _positive_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise V4SynthesisError(f"{label} must be a positive integer.")
    return value


def _sha256(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise V4SynthesisError(f"{label} is not a SHA-256 string.")
    _require_sha256(value, label)
    return value


def _require_sha256(value: str, label: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise V4SynthesisError(f"{label} is not a lowercase SHA-256 digest.")


def _timestamp(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise V4SynthesisError(f"{label} must be an ISO-8601 timestamp.")
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise V4SynthesisError(f"{label} must be an ISO-8601 timestamp.") from error
    return value
