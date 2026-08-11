from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from kds.data.licenses import load_license_ledger
from kds.eval.xlsr_tone_speak_ood import (
    ToneSpeakFrozenCheckpoint,
    ToneSpeakOodCandidate,
    ToneSpeakOodInferenceConfig,
    ToneSpeakOodOutputs,
    ToneSpeakOodPlan,
    ToneSpeakOodPlanError,
    ToneSpeakPinnedXlsrEncoder,
    ToneSpeakXlsrSlsHead,
    load_tone_speak_ood_plan,
    tone_speak_ood_plan_record,
    validate_tone_speak_ood_inputs,
)
from kds.training.xlsr_stage_a_plan import PinnedFile


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_plan(root: Path) -> Path:
    checkpoint = root / "stage-b.pt"
    checkpoint.write_bytes(b"frozen-stage-b")
    stage_b_report = root / "stage-b-report.json"
    stage_b_report.write_text(
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
    encoder_dir = root / "encoder"
    encoder_dir.mkdir()
    config = encoder_dir / "config.json"
    config.write_text("{}\n", encoding="utf-8")
    weights = encoder_dir / "weights.bin"
    weights.write_bytes(b"encoder")
    files = {
        "ledger.csv": b"ledger\n",
        "candidate.csv": b"candidate\n",
        "ready.json": json.dumps(
            {
                "source_id": "tone_speak_ru_v1",
                "ready_manifest_sha256": "b" * 64,
                "ready_rows": 100,
                "raw_rows": 100,
                "rejected_rows": 0,
                "final_or_product_eligible": False,
            }
        ).encode(),
        "packet.csv": b"packet\n",
        "gate.json": b"{}\n",
        "audit.json": b"{}\n",
        "lock.json": b"{}\n",
        "runner.py": b"# pinned implementation\n",
    }
    for name, content in files.items():
        (root / name).write_bytes(content)
    plan = root / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": "xlsr-tone-speak-ood-test-v1",
                "purpose": "research",
                "protocol": {
                    "kind": "exploratory_russian_spoof_only_ood_evaluation",
                    "quality_claim": "not_final_quality",
                    "training": "prohibited",
                    "calibration": "prohibited",
                    "threshold_selection": "prohibited",
                    "binary_metrics": "unavailable_spoof_only",
                    "acoustic_language_preservation": "verified_for_pinned_assets_only",
                },
                "license_ledger": {"path": "ledger.csv", "sha256": _sha256(root / "ledger.csv")},
                "checkpoint": {
                    "path": "stage-b.pt",
                    "sha256": _sha256(checkpoint),
                    "stage_b_report": {
                        "path": "stage-b-report.json",
                        "sha256": _sha256(stage_b_report),
                    },
                    "selected_trainable_state_sha256": "a" * 64,
                },
                "encoder": {
                    "checkpoint_dir": "encoder",
                    "revision": "b" * 64,
                    "config": {"path": "encoder/config.json", "sha256": _sha256(config)},
                    "weights": {"path": "encoder/weights.bin", "sha256": _sha256(weights)},
                },
                "head": {"attention_size": 128, "classifier_size": 256, "dropout": 0.2},
                "candidate": {
                    "manifest": {
                        "path": "candidate.csv",
                        "sha256": _sha256(root / "candidate.csv"),
                    },
                    "ready_receipt": {
                        "path": "ready.json",
                        "sha256": _sha256(root / "ready.json"),
                    },
                    "acoustic_gate_packet": {
                        "path": "packet.csv",
                        "sha256": _sha256(root / "packet.csv"),
                    },
                    "acoustic_gate_report": {
                        "path": "gate.json",
                        "sha256": _sha256(root / "gate.json"),
                    },
                    "source_audit_receipt": {
                        "path": "audit.json",
                        "sha256": _sha256(root / "audit.json"),
                    },
                    "source_artifact_lock": {
                        "path": "lock.json",
                        "sha256": _sha256(root / "lock.json"),
                    },
                    "expected_rows": 100,
                    "expected_source_id": "tone_speak_ru_v1",
                    "expected_voice_ids": [
                        f"tone_speak_ru_v1:voice:{name}"
                        for name in (
                            "alloy",
                            "ash",
                            "ballad",
                            "coral",
                            "echo",
                            "fable",
                            "nova",
                            "onyx",
                            "sage",
                            "shimmer",
                        )
                    ],
                },
                "implementation": [{"path": "runner.py", "sha256": _sha256(root / "runner.py")}],
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


def test_tone_speak_plan_pins_every_static_input(tmp_path: Path) -> None:
    plan_path = _write_plan(tmp_path)
    plan = load_tone_speak_ood_plan(plan_path)

    record = tone_speak_ood_plan_record(plan)
    protocol = record["protocol"]

    assert plan.candidate.expected_rows == 100
    assert record["plan_sha256"] == _sha256(plan_path)
    assert isinstance(protocol, dict)
    assert protocol["binary_metrics"] == "unavailable_spoof_only"


def test_tone_speak_plan_rejects_changed_implementation(tmp_path: Path) -> None:
    plan_path = _write_plan(tmp_path)
    (tmp_path / "runner.py").write_text("# changed\n", encoding="utf-8")

    with pytest.raises(ToneSpeakOodPlanError, match="Pinned implementation SHA-256 mismatch"):
        load_tone_speak_ood_plan(plan_path)


def _project_plan() -> ToneSpeakOodPlan:
    root = Path.cwd()
    candidate = root / "data/manifests/tone_speak_ru_v1_ood_ready_100.csv"
    return ToneSpeakOodPlan(
        run_id="test",
        plan_path=root / "unused.json",
        plan_sha256="a" * 64,
        protocol={},
        license_ledger=PinnedFile(
            root / "data/licenses/license_ledger.csv",
            _sha256(root / "data/licenses/license_ledger.csv"),
        ),
        checkpoint=ToneSpeakFrozenCheckpoint(
            checkpoint=PinnedFile(root / "models/xlsr-sls-stage-b-v1.pt", "a" * 64),
            report=PinnedFile(root / "models/xlsr-sls-stage-b-v1-report.json", "a" * 64),
            selected_trainable_state_sha256="a" * 64,
        ),
        encoder=ToneSpeakPinnedXlsrEncoder(
            checkpoint_dir=root / "models/xlsr-300m",
            revision="a" * 64,
            config=PinnedFile(root / "models/xlsr-300m/config.json", "a" * 64),
            weights=PinnedFile(root / "models/xlsr-300m/pytorch_model.bin", "a" * 64),
        ),
        head=ToneSpeakXlsrSlsHead(attention_size=128, classifier_size=256, dropout=0.2),
        candidate=ToneSpeakOodCandidate(
            manifest=PinnedFile(candidate, _sha256(candidate)),
            ready_receipt=PinnedFile(
                root / "data/manifests/tone_speak_ru_v1_ood_ready_100_receipt.json",
                _sha256(root / "data/manifests/tone_speak_ru_v1_ood_ready_100_receipt.json"),
            ),
            acoustic_gate_packet=PinnedFile(
                root / "data/manifests/tone_speak_ru_v1_ood_acoustic_gate_packet.csv",
                _sha256(root / "data/manifests/tone_speak_ru_v1_ood_acoustic_gate_packet.csv"),
            ),
            acoustic_gate_report=PinnedFile(
                root / "data/manifests/tone_speak_ru_v1_ood_acoustic_gate_report_v2.json",
                _sha256(root / "data/manifests/tone_speak_ru_v1_ood_acoustic_gate_report_v2.json"),
            ),
            source_audit_receipt=PinnedFile(
                root / "data/licenses/tone_speak_ru_v1_artifact_audit_receipt.json",
                _sha256(root / "data/licenses/tone_speak_ru_v1_artifact_audit_receipt.json"),
            ),
            source_artifact_lock=PinnedFile(
                root / "data/licenses/tone_speak_ru_v1_artifact_lock.json",
                _sha256(root / "data/licenses/tone_speak_ru_v1_artifact_lock.json"),
            ),
            expected_rows=100,
            expected_source_id="tone_speak_ru_v1",
            expected_voice_ids=tuple(
                f"tone_speak_ru_v1:voice:{name}"
                for name in (
                    "alloy",
                    "ash",
                    "ballad",
                    "coral",
                    "echo",
                    "fable",
                    "nova",
                    "onyx",
                    "sage",
                    "shimmer",
                )
            ),
        ),
        implementation=(),
        inference=ToneSpeakOodInferenceConfig(
            sample_rate=16000,
            window_samples=64600,
            batch_size=4,
            num_workers=0,
            device="cuda",
            precision="bf16",
            raw_logit_decision_boundary=0.0,
        ),
        outputs=ToneSpeakOodOutputs(execution_lock=root / "unused-a", report=root / "unused-b"),
    )


def test_project_candidate_requires_reviewed_spoof_only_evidence() -> None:
    rows = validate_tone_speak_ood_inputs(
        _project_plan(), load_license_ledger(Path("data/licenses/license_ledger.csv"))
    )

    assert len(rows) == 100
    assert {row.label for row in rows} == {"spoof"}
