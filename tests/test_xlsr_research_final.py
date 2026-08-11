from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from kds.eval.xlsr_research_final import (
    XlsrResearchFinalPlanError,
    final_plan_record,
    load_xlsr_research_final_plan,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pin(path: Path, root: Path) -> dict[str, str]:
    return {"path": str(path.relative_to(root)), "sha256": _sha256(path)}


def _write_plan(root: Path) -> Path:
    encoder = root / "encoder"
    encoder.mkdir()
    files = {
        "checkpoint.pt": b"checkpoint",
        "ledger.csv": b"ledger\n",
        "train.csv": b"train\n",
        "stage-a-dev.csv": b"stage-a-dev\n",
        "stage-b-dev.csv": b"stage-b-dev\n",
        "calibration.csv": b"calibration\n",
        "ru.csv": b"ru\n",
        "kk.csv": b"kk\n",
        "mixed.csv": b"mixed\n",
        "ru-gate.json": b"{}\n",
        "mixed-gate.json": b"{}\n",
        "old-mixed-report.json": b"{}\n",
        "runner.py": b"# pinned runner\n",
    }
    for name, content in files.items():
        (root / name).write_bytes(content)
    config = encoder / "config.json"
    config.write_text("{}\n", encoding="utf-8")
    weights = encoder / "weights.bin"
    weights.write_bytes(b"weights")
    stage_b_report = root / "stage-b-report.json"
    stage_b_report.write_text(
        json.dumps(
            {
                "status": "ok",
                "checkpoint_scope": "sls_head_and_final_xlsr_blocks",
                "selected_trainable_state_sha256": "a" * 64,
                "frozen_final_evaluation_performed": False,
                "calibrated": False,
            }
        ),
        encoding="utf-8",
    )
    def role(name: str, split: str, rows: int) -> dict[str, object]:
        return {
            "manifest": _pin(root / name, root),
            "selected_split": split,
            "expected_rows": rows,
        }

    def layer(
        name: str,
        rows: int,
        pairs: int,
        evidence: str,
        report: str | None,
        exposure: str,
        receipt: str | None,
    ) -> dict[str, object]:
        return {
            "name": name,
            "language": name,
            "manifest": _pin(root / f"{name}.csv", root),
            "expected_rows": rows,
            "expected_pairs": pairs,
            "evidence_kind": evidence,
            "evidence_report": _pin(root / report, root) if report else None,
            "project_exposure": exposure,
            "exposure_receipt": _pin(root / receipt, root) if receipt else None,
        }
    plan = root / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": "xlsr-stage-b-v2-research-final-v1",
                "purpose": "research",
                "protocol": {
                    "kind": "confirmatory_multilingual_research_evaluation",
                    "quality_claim": "research_only_not_product_quality",
                    "test_novelty": "model_version_holdout_not_project_level_blind",
                    "calibration": "temperature_only_on_pinned_pyara_role",
                    "decision_boundary": "fixed_calibrated_probability_0.5",
                    "pooled_language_metric": "prohibited",
                },
                "license_ledger": _pin(root / "ledger.csv", root),
                "checkpoint": {
                    **_pin(root / "checkpoint.pt", root),
                    "stage_b_report": _pin(stage_b_report, root),
                    "selected_trainable_state_sha256": "a" * 64,
                },
                "encoder": {
                    "checkpoint_dir": "encoder",
                    "revision": "b" * 40,
                    "config": _pin(config, root),
                    "weights": _pin(weights, root),
                },
                "head": {"attention_size": 128, "classifier_size": 256, "dropout": 0.2},
                "roles": {
                    "train": role("train.csv", "train", 10),
                    "stage_a_dev": role("stage-a-dev.csv", "dev", 4),
                    "stage_b_dev": role("stage-b-dev.csv", "dev", 6),
                    "calibration": role("calibration.csv", "dev", 8),
                    "final_layers": [
                        layer(
                            "ru",
                            10,
                            5,
                            "two_review_acoustic_gate",
                            "ru-gate.json",
                            "never_inferred",
                            None,
                        ),
                        layer(
                            "kk",
                            12,
                            6,
                            "source_transcript_only_no_acoustic_review",
                            None,
                            "never_inferred",
                            None,
                        ),
                        layer(
                            "mixed",
                            6,
                            3,
                            "two_review_acoustic_gate",
                            "mixed-gate.json",
                            "previously_inferred_with_older_model",
                            "old-mixed-report.json",
                        ),
                    ],
                },
                "implementation": [_pin(root / "runner.py", root)],
                "inference": {
                    "sample_rate": 16000,
                    "window_samples": 64600,
                    "batch_size": 4,
                    "num_workers": 0,
                    "device": "cuda",
                    "precision": "bf16",
                    "calibrated_probability_boundary": 0.5,
                    "temperature_max_iter": 50,
                },
                "outputs": {"execution_lock": "execution.json", "report": "report.json"},
            }
        ),
        encoding="utf-8",
    )
    return plan


def test_research_final_plan_discloses_prior_mixed_exposure(tmp_path: Path) -> None:
    path = _write_plan(tmp_path)

    plan = load_xlsr_research_final_plan(path)
    record = final_plan_record(plan)

    assert plan.plan_sha256 == _sha256(path)
    assert plan.final_layers[2].project_exposure == "previously_inferred_with_older_model"
    assert record["protocol"]["pooled_language_metric"] == "prohibited"  # type: ignore[index]


def test_research_final_plan_rejects_changed_pinned_implementation(tmp_path: Path) -> None:
    path = _write_plan(tmp_path)
    (tmp_path / "runner.py").write_text("# changed\n", encoding="utf-8")

    with pytest.raises(XlsrResearchFinalPlanError, match="missing or changed"):
        load_xlsr_research_final_plan(path)
