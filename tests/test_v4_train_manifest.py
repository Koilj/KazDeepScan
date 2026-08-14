from __future__ import annotations

import runpy
from pathlib import Path
from typing import Any

import pytest

from kds.data.manifest import load_manifest
from kds.data.v4_train_manifest import V4TrainManifestError


def _module() -> dict[str, Any]:
    return runpy.run_path("scripts/freeze_v4_combined_train_manifest.py")


def test_v4_combined_train_plan_binds_frozen_inputs() -> None:
    module = _module()

    plan = module["load_plan"](
        Path("configs/research/v4/xlsr_sls_model_v4_train_manifest_v1.json"), Path(".").resolve()
    )

    assert plan.expected_combined_cells == {
        "kk/bonafide": 5_000,
        "kk/spoof": 5_000,
        "ru/bonafide": 5_000,
        "ru/spoof": 5_000,
    }
    assert plan.expected_shared_text_hashes == 4_604


def test_v4_combined_train_assembly_preserves_balanced_cells() -> None:
    module = _module()
    root = Path(".").resolve()
    plan = module["load_plan"](
        Path("configs/research/v4/xlsr_sls_model_v4_train_manifest_v1.json"), root
    )
    source_rows = load_manifest(root / plan.inputs["source_frozen_manifest"].path)
    spoof_rows = load_manifest(root / plan.inputs["kk_spoof_frozen_manifest"].path)

    combined, report = module["build_v4_combined_train_manifest"](
        source_rows,
        spoof_rows,
        expected_source_cells=plan.expected_source_cells,
        expected_spoof_cells=plan.expected_spoof_cells,
        expected_combined_cells=plan.expected_combined_cells,
        expected_shared_text_hashes=plan.expected_shared_text_hashes,
    )

    assert len(combined) == 20_000
    assert report.shared_text_hashes == 4_604
    assert report.combined_cell_counts == plan.expected_combined_cells


def test_v4_combined_train_assembly_rejects_unpinned_text_overlap() -> None:
    module = _module()
    root = Path(".").resolve()
    plan = module["load_plan"](
        Path("configs/research/v4/xlsr_sls_model_v4_train_manifest_v1.json"), root
    )
    source_rows = load_manifest(root / plan.inputs["source_frozen_manifest"].path)
    spoof_rows = load_manifest(root / plan.inputs["kk_spoof_frozen_manifest"].path)

    with pytest.raises(V4TrainManifestError, match="text overlap"):
        module["build_v4_combined_train_manifest"](
            source_rows,
            spoof_rows,
            expected_source_cells=plan.expected_source_cells,
            expected_spoof_cells=plan.expected_spoof_cells,
            expected_combined_cells=plan.expected_combined_cells,
            expected_shared_text_hashes=0,
        )
