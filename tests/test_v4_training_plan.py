from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from kds.data.licenses import load_license_ledger
from kds.data.manifest import load_manifest
from kds.training.v4_training_plan import (
    V4BalancedCellBatchSampler,
    load_v4_training_plan,
    validate_and_select_v4_training,
)


def _plan_path() -> Path:
    return Path("configs/research/v4/xlsr_sls_model_v4_training_v1.json")


def test_v4_training_contract_binds_complete_balanced_train_and_isolated_bilingual_dev() -> None:
    plan = load_v4_training_plan(_plan_path())
    report, selected = validate_and_select_v4_training(
        plan, load_license_ledger(plan.license_ledger.path)
    )

    assert plan.training.selection_metric == "macro_language_dev_loss_ru_kk"
    assert plan.training.sampler == "balanced_language_label_without_padding"
    assert [(role.role, role.rows) for role in report.roles] == [("train", 20_000), ("dev", 1_917)]
    assert report.train_dev_overlap == {
        "sample_id": 0,
        "asset_sha256": 0,
        "text_hash": 0,
        "parent_group_id": 0,
    }
    assert len(selected.dev_ru) == 969
    assert len(selected.dev_kk) == 948


def test_v4_balanced_sampler_uses_each_train_row_once_without_padding() -> None:
    rows = load_manifest(Path("data/manifests/v4/xlsr_sls_model_v4_train_frozen_v1.csv"))
    sampler = V4BalancedCellBatchSampler(rows, batch_size=4, seed=20260815)
    sampler.set_epoch(2)
    batches = list(sampler)

    flattened = [index for batch in batches for index in batch]
    assert len(batches) == 5_000
    assert len(flattened) == len(rows)
    assert len(set(flattened)) == len(rows)
    for batch in batches[:20]:
        assert Counter(f"{rows[index].language}/{rows[index].label}" for index in batch) == {
            "kk/bonafide": 1,
            "kk/spoof": 1,
            "ru/bonafide": 1,
            "ru/spoof": 1,
        }


def test_v4_balanced_sampler_rejects_duplicate_padding_shape() -> None:
    rows = load_manifest(Path("data/manifests/v4/xlsr_sls_model_v4_train_frozen_v1.csv"))

    with pytest.raises(ValueError, match="positive multiple of four"):
        V4BalancedCellBatchSampler(rows, batch_size=6, seed=1)
