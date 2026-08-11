from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from kds.data.licenses import load_license_ledger
from kds.eval.xlsr_exploratory import (
    ExploratoryInferenceConfig,
    ExploratoryMixedCandidate,
    ExploratoryOutputs,
    FrozenStageBCheckpoint,
    PinnedXlsrEncoder,
    XlsrExploratoryMixedPlan,
    XlsrExploratoryMixedPlanError,
    XlsrSlsHead,
    load_xlsr_exploratory_mixed_plan,
    validate_exploratory_mixed_inputs,
    xlsr_exploratory_mixed_plan_record,
)
from kds.training.xlsr_stage_a_plan import PinnedFile


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_plan(root: Path) -> Path:
    checkpoint = root / "stage-b.pt"
    checkpoint.write_bytes(b"frozen-stage-b")
    report = root / "stage-b-report.json"
    report.write_text(
        json.dumps(
            {
                "status": "ok",
                "checkpoint_scope": "sls_head_and_final_xlsr_blocks",
                "frozen_final_evaluation_performed": False,
                "calibrated": False,
                "selected_trainable_state_sha256": "a" * 64,
            }
        ),
        encoding="utf-8",
    )
    ledger = root / "ledger.csv"
    ledger.write_text("ledger\n", encoding="utf-8")
    candidate = root / "candidate.csv"
    candidate.write_text("candidate\n", encoding="utf-8")
    pair_lock = root / "pair-lock.json"
    pair_lock.write_text("{}\n", encoding="utf-8")
    encoder_dir = root / "encoder"
    encoder_dir.mkdir()
    config = encoder_dir / "config.json"
    config.write_text("{}\n", encoding="utf-8")
    weights = encoder_dir / "weights.bin"
    weights.write_bytes(b"encoder")
    runner = root / "runner.py"
    runner.write_text("# pinned implementation\n", encoding="utf-8")
    module = root / "module.py"
    module.write_text("# pinned module\n", encoding="utf-8")
    plan = root / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": "xlsr-mixed-exploratory-test-v1",
                "purpose": "research",
                "protocol": {
                    "kind": "exploratory_mixed_stress_test",
                    "quality_claim": "not_final_quality",
                    "training": "prohibited",
                    "calibration": "prohibited",
                    "threshold_selection": "prohibited",
                    "acoustic_language_preservation": "not_performed",
                },
                "license_ledger": {"path": ledger.name, "sha256": _sha256(ledger)},
                "checkpoint": {
                    "path": checkpoint.name,
                    "sha256": _sha256(checkpoint),
                    "stage_b_report": {"path": report.name, "sha256": _sha256(report)},
                    "selected_trainable_state_sha256": "a" * 64,
                },
                "encoder": {
                    "checkpoint_dir": encoder_dir.name,
                    "revision": "b" * 64,
                    "config": {"path": "encoder/config.json", "sha256": _sha256(config)},
                    "weights": {"path": "encoder/weights.bin", "sha256": _sha256(weights)},
                },
                "head": {"attention_size": 128, "classifier_size": 256, "dropout": 0.2},
                "candidate": {
                    "manifest": {"path": candidate.name, "sha256": _sha256(candidate)},
                    "pair_lock": {"path": pair_lock.name, "sha256": _sha256(pair_lock)},
                    "expected_pairs": 30,
                    "expected_source_ids": ["base-source", "synthetic-source"],
                },
                "implementation": [
                    {"path": runner.name, "sha256": _sha256(runner)},
                    {"path": module.name, "sha256": _sha256(module)},
                ],
                "inference": {
                    "sample_rate": 16000,
                    "window_samples": 64600,
                    "batch_size": 4,
                    "num_workers": 0,
                    "device": "cuda",
                    "precision": "bf16",
                    "raw_logit_decision_boundary": 0.0,
                },
                "outputs": {"execution_lock": "execution.json", "report": "report.json"},
            }
        ),
        encoding="utf-8",
    )
    return plan


def test_plan_pins_frozen_checkpoint_candidate_and_implementation(tmp_path: Path) -> None:
    plan_path = _write_plan(tmp_path)

    plan = load_xlsr_exploratory_mixed_plan(plan_path)
    record = xlsr_exploratory_mixed_plan_record(plan)

    assert plan.inference.raw_logit_decision_boundary == 0.0
    assert plan.candidate.expected_pairs == 30
    assert record["plan_sha256"] == _sha256(plan_path)
    assert record["protocol"]["quality_claim"] == "not_final_quality"


def test_plan_rejects_implementation_changed_after_pinning(tmp_path: Path) -> None:
    plan_path = _write_plan(tmp_path)
    (tmp_path / "runner.py").write_text("# changed\n", encoding="utf-8")

    with pytest.raises(
        XlsrExploratoryMixedPlanError,
        match="Pinned implementation SHA-256 mismatch",
    ):
        load_xlsr_exploratory_mixed_plan(plan_path)


def _project_candidate_plan(pair_lock: Path) -> XlsrExploratoryMixedPlan:
    root = Path.cwd()
    manifest = root / "data/manifests/ksc2_mixed_v1_silero_v4_candidate_30.csv"
    return XlsrExploratoryMixedPlan(
        run_id="test",
        plan_path=root / "unused.json",
        plan_sha256="a" * 64,
        protocol={},
        license_ledger=PinnedFile(root / "data/licenses/license_ledger.csv", "a" * 64),
        checkpoint=FrozenStageBCheckpoint(
            checkpoint=PinnedFile(root / "models/xlsr-sls-stage-b-v1.pt", "a" * 64),
            report=PinnedFile(root / "models/xlsr-sls-stage-b-v1-report.json", "a" * 64),
            selected_trainable_state_sha256="a" * 64,
        ),
        encoder=PinnedXlsrEncoder(
            checkpoint_dir=root / "models/xlsr-300m",
            revision="a" * 64,
            config=PinnedFile(root / "models/xlsr-300m/config.json", "a" * 64),
            weights=PinnedFile(root / "models/xlsr-300m/pytorch_model.bin", "a" * 64),
        ),
        head=XlsrSlsHead(attention_size=128, classifier_size=256, dropout=0.2),
        candidate=ExploratoryMixedCandidate(
            manifest=PinnedFile(manifest, _sha256(manifest)),
            pair_lock=PinnedFile(pair_lock, _sha256(pair_lock)),
            expected_pairs=30,
            expected_source_ids=("ksc2_mixed_v1_silero_v4", "ksc2_v1"),
        ),
        implementation=(),
        inference=ExploratoryInferenceConfig(
            sample_rate=16000,
            window_samples=64600,
            batch_size=4,
            num_workers=0,
            device="cuda",
            precision="bf16",
            raw_logit_decision_boundary=0.0,
        ),
        outputs=ExploratoryOutputs(execution_lock=root / "unused-a", report=root / "unused-b"),
    )


def test_candidate_validation_requires_exact_pair_lock_content(tmp_path: Path) -> None:
    original = Path("data/licenses/ksc2_mixed_v1_silero_v4_pair_lock.json")
    pair_lock = tmp_path / "pair-lock.json"
    payload = json.loads(original.read_text(encoding="utf-8"))
    payload["pairs"][0]["spoof_audio_sha256"] = "0" * 64
    pair_lock.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(XlsrExploratoryMixedPlanError, match="content mismatches candidate pair"):
        validate_exploratory_mixed_inputs(
            _project_candidate_plan(pair_lock),
            load_license_ledger(Path("data/licenses/license_ledger.csv")),
        )
