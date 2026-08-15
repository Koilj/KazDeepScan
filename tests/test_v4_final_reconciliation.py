from __future__ import annotations

from pathlib import Path

from kds.data.v4_final_reconciliation import OUTPUTS, _load_selection, load_plan


def test_reconciliation_contract_is_publication_only_for_the_frozen_salvage_selection() -> None:
    root = Path(__file__).resolve().parents[1]
    plan = load_plan(
        root / "configs/research/v4/xlsr_sls_model_v4_final_reconciliation_v1.json", root
    )

    selected = _load_selection(plan, root)

    assert len(selected) == 997
    assert {row.selection_rank for row in selected if row.language == "kk"}.isdisjoint({272, 310})
    assert all("final_reconciliation" in path for path in OUTPUTS.values())
