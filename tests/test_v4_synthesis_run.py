from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from kds.data.assets import sha256_file
from kds.data.v4_synthesis import V4KkSpoofCandidate, load_v4_kk_spoof_candidates
from kds.data.v4_synthesis_run import (
    V4KkSynthesisPlan,
    V4SynthesisProfile,
    append_v4_synthesis_journal,
    build_v4_kk_synthesis_tasks,
    load_v4_kk_synthesis_plan,
    load_v4_synthesis_journal,
    start_v4_synthesis_record,
    terminal_v4_synthesis_record,
    v4_route_runtime_binding_path,
)


def _repository_plan() -> tuple[V4KkSynthesisPlan, tuple[V4KkSpoofCandidate, ...]]:
    plan = load_v4_kk_synthesis_plan(
        Path("configs/research/v4/xlsr_sls_model_v4_kk_spoof_synthesis_v1.json"),
        Path(".").resolve(),
    )
    candidates = load_v4_kk_spoof_candidates(
        Path("data/manifests/v4/xlsr_sls_model_v4_train_candidates_v2.csv"),
        Path("docs/artifacts/v4/xlsr_sls_model_v4_train_candidate_selection_governance_v1.json"),
        Path("docs/artifacts/v4/xlsr_sls_model_v4_source_decode_qa_v1.json"),
    )
    return plan, candidates


def test_v4_synthesis_plan_assigns_frozen_piper_quota() -> None:
    plan, candidates = _repository_plan()
    route = plan.route("kk-piper-issai-high-v1")
    profiles = tuple(V4SynthesisProfile(f"voice-{index}", speaker_id=index) for index in range(6))

    tasks = build_v4_kk_synthesis_tasks(candidates, plan, route, profiles)

    assert len(tasks) == 1800
    assert sum(task.candidate.target_state == "target" for task in tasks) == 1500
    assert sum(task.candidate.target_state == "reserve" for task in tasks) == 300
    assert Counter(task.profile.voice_id for task in tasks) == {
        f"voice-{index}": 300 for index in range(6)
    }
    assert len({task.output_relative_path for task in tasks}) == len(tasks)
    assert all(task.output_relative_path.startswith("raw/v4/") for task in tasks)


def test_v4_synthesis_route_bindings_are_isolated(tmp_path: Path) -> None:
    plan, _candidates = _repository_plan()

    paths = {v4_route_runtime_binding_path(tmp_path, route.route_id) for route in plan.routes}

    assert len(paths) == 4
    assert all(path.parent == tmp_path for path in paths)
    assert all(path.name.endswith(".binding.json") for path in paths)


def test_v4_synthesis_journal_hash_binds_successful_output(tmp_path: Path) -> None:
    plan, candidates = _repository_plan()
    route = plan.route("kk-mms-kaz-v1")
    task = build_v4_kk_synthesis_tasks(
        candidates,
        plan,
        route,
        (V4SynthesisProfile("mms_tts_kaz_default"),),
    )[0]
    data_root = tmp_path / "data"
    output = data_root / task.output_relative_path
    output.parent.mkdir(parents=True)
    output.write_bytes(b"v4 synthetic test payload")
    timestamp = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    journal = tmp_path / "route.jsonl"

    append_v4_synthesis_journal(journal, start_v4_synthesis_record(plan, task, timestamp))
    append_v4_synthesis_journal(
        journal,
        terminal_v4_synthesis_record(
            plan=plan,
            task=task,
            event="succeeded",
            timestamp=timestamp,
            actual_profile=task.profile,
            actual_seed=task.base_seed,
            generation_attempts=1,
            retry_errors=(),
            output_sha256=sha256_file(output),
            duration_s=1.0,
            original_sr=16_000,
            device="local_cpu_mms",
        ),
    )

    states = load_v4_synthesis_journal(journal, plan, (task,), data_root)

    assert states[task.candidate.candidate_id].terminal is not None
