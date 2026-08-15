from __future__ import annotations

from pathlib import Path

from kds.data.v4_final_salvage_materialization import (
    KK_SOURCE_ID,
    KK_SPOOF_ID,
    OUTPUTS,
    RU_SOURCE_ID,
    RU_SPOOF_ID,
    _load_rejects,
    _load_selection,
    load_plan,
)


def test_salvage_contract_is_finite_and_excludes_only_the_two_locked_token_rejects() -> None:
    root = Path(__file__).resolve().parents[1]
    plan = load_plan(
        root / "configs/research/v4/xlsr_sls_model_v4_final_salvage_materialization_v1.json",
        root,
    )
    selected = _load_selection(plan, root)
    rejects = _load_rejects(plan, root)

    assert len(selected) == 997
    assert sum(row.language == "ru" for row in selected) == 499
    assert sum(row.language == "kk" for row in selected) == 498
    assert {row.selection_rank for row in rejects} == {272, 310}
    assert all(row.sample_id not in {item.sample_id for item in selected} for row in rejects)


def test_salvage_outputs_and_source_ids_are_new() -> None:
    assert all("final_salvage" in path for path in OUTPUTS.values())
    assert len({RU_SOURCE_ID, KK_SOURCE_ID, RU_SPOOF_ID, KK_SPOOF_ID}) == 4
