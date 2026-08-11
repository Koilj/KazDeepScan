from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from kds.data.licenses import load_license_ledger
from kds.data.manifest import ManifestRow, write_manifest
from kds.training.xlsr_stage_a_plan import (
    XlsrStageAPlanError,
    load_xlsr_stage_a_plan,
    validate_and_select_xlsr_stage_a,
    xlsr_stage_a_plan_record,
)
from tests.factories import manifest_mapping


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(source: str, split: str) -> list[ManifestRow]:
    def row_sha256(label: str) -> str:
        return hashlib.sha256(f"{source}:{label}".encode()).hexdigest()

    return [
        ManifestRow.from_mapping(
            manifest_mapping(
                sample_id=f"{source}-{label}",
                relative_path=f"{source}-{label}.wav",
                sha256=row_sha256(label),
                source_name=source,
                split=split,
                label=label,
                language="ru",
                parent_group_id=f"parent-{source}-{label}",
                speaker_pseudo_id=f"speaker-{source}-{label}",
                text_id=f"text-{source}-{label}",
                text_hash=("c" if label == "bonafide" else "d") * 63
                + ("1" if source == "train-source" else "2"),
                generator_family="test-tts" if label == "spoof" else "",
                generator_name="test-generator" if label == "spoof" else "",
                generator_version="1" if label == "spoof" else "",
                voice_id="test-voice" if label == "spoof" else "",
            ),
            row_number=2,
        )
        for label in ("bonafide", "spoof")
    ]


def _write_ledger(path: Path, sources: tuple[str, ...]) -> None:
    header = (
        "source_id,usage_scope,train_dev_test_use,ood_evaluation_use,"
        "bonafide_group_provenance,spoof_voice_group_provenance,license,source_url,"
        "artifact_name,expected_size_bytes,last_modified_utc,sha256,rights_basis,status,notes"
    )
    lines = [header]
    for source in sources:
        lines.append(
            f"{source},personal_research,research_only,prohibited,unknown,unknown,"
            "CC-BY-4.0,https://example.test/source,archive.zip,100,"
            "2026-08-10T00:00:00Z," + "e" * 64 + ",research,verified,test"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _prepare_plan(root: Path, *, dev_source: str = "dev-source") -> Path:
    train_manifest = root / "train.csv"
    dev_manifest = root / "dev.csv"
    write_manifest(train_manifest, _rows("train-source", "train"))
    write_manifest(dev_manifest, _rows(dev_source, "dev"))
    ledger = root / "ledger.csv"
    _write_ledger(ledger, tuple(sorted({"train-source", dev_source})))
    encoder = root / "encoder"
    encoder.mkdir()
    config = encoder / "config.json"
    weights = encoder / "pytorch_model.bin"
    config.write_text('{"model_type": "wav2vec2"}\n', encoding="utf-8")
    weights.write_bytes(b"test-weights")
    plan = root / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": "xlsr-stage-a-test",
                "purpose": "research",
                "license_ledger": {"path": "ledger.csv", "sha256": _sha256(ledger)},
                "train": {
                    "manifest": {"path": "train.csv", "sha256": _sha256(train_manifest)},
                    "source_split": "train",
                    "expected_source_ids": ["train-source"],
                    "expected_languages": ["ru"],
                },
                "dev": {
                    "manifest": {"path": "dev.csv", "sha256": _sha256(dev_manifest)},
                    "source_split": "dev",
                    "expected_source_ids": [dev_source],
                    "expected_languages": ["ru"],
                },
                "encoder": {
                    "checkpoint_dir": "encoder",
                    "revision": "test-revision",
                    "config": {"filename": "config.json", "sha256": _sha256(config)},
                    "weights": {
                        "filename": "pytorch_model.bin",
                        "sha256": _sha256(weights),
                    },
                },
                "head": {"attention_size": 4, "classifier_size": 8, "dropout": 0.2},
                "training": {
                    "seed": 20260818,
                    "epochs": 3,
                    "batch_size": 2,
                    "gradient_accumulation_steps": 2,
                    "window_samples": 1600,
                    "sample_rate": 16000,
                    "learning_rate": 0.0001,
                    "weight_decay": 0.0001,
                    "gradient_clip_norm": 1.0,
                    "num_workers": 0,
                    "pin_memory": True,
                    "device": "cuda",
                    "precision": "bf16",
                    "freeze_encoder": True,
                    "encoder_eval_mode": True,
                    "selection_metric": "dev_loss",
                },
                "outputs": {"checkpoint": "result.pt", "report": "result.json"},
            }
        ),
        encoding="utf-8",
    )
    return plan


def test_stage_a_plan_pins_inputs_and_selects_only_train_dev(tmp_path: Path) -> None:
    plan_path = _prepare_plan(tmp_path)

    plan = load_xlsr_stage_a_plan(plan_path)
    report, selected = validate_and_select_xlsr_stage_a(
        plan, load_license_ledger(plan.license_ledger.path)
    )
    record = xlsr_stage_a_plan_record(plan)

    assert len(selected.train) == 2
    assert len(selected.dev) == 2
    assert [role.role for role in report.roles] == ["train", "dev"]
    assert record["plan_sha256"] == _sha256(plan_path)
    assert plan.training.freeze_encoder
    assert plan.training.selection_metric == "dev_loss"


def test_stage_a_plan_rejects_changed_encoder_weights(tmp_path: Path) -> None:
    plan_path = _prepare_plan(tmp_path)
    (tmp_path / "encoder" / "pytorch_model.bin").write_bytes(b"changed")

    with pytest.raises(XlsrStageAPlanError, match="XLS-R weights SHA-256 mismatch"):
        load_xlsr_stage_a_plan(plan_path)


def test_stage_a_plan_rejects_source_leakage_between_train_and_dev(tmp_path: Path) -> None:
    plan_path = _prepare_plan(tmp_path, dev_source="train-source")
    plan = load_xlsr_stage_a_plan(plan_path)

    with pytest.raises(XlsrStageAPlanError, match="Source leakage between train/dev"):
        validate_and_select_xlsr_stage_a(plan, load_license_ledger(plan.license_ledger.path))


def test_stage_a_plan_rejects_non_cuda_training(tmp_path: Path) -> None:
    plan_path = _prepare_plan(tmp_path)
    raw = json.loads(plan_path.read_text(encoding="utf-8"))
    raw["training"]["device"] = "cpu"
    plan_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(XlsrStageAPlanError, match="device must be exactly 'cuda'"):
        load_xlsr_stage_a_plan(plan_path)
