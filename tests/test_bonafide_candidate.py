from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from kds.data.manifest import write_manifest
from kds.eval.bonafide_candidate import select_unexposed_bonafide_candidate
from tests.test_candidate_exposure import _row


def test_bonafide_selection_excludes_only_configured_role_overlaps(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    unexposed = _row(sample_id="unexposed", split="test", sha256="a" * 64, text_hash="b" * 64)
    exposed = _row(sample_id="exposed", split="test", sha256="c" * 64, text_hash="d" * 64)
    write_manifest(source, [unexposed, exposed])
    config_root = tmp_path / "configs"
    config_root.mkdir()
    prior = tmp_path / "prior.csv"
    write_manifest(
        prior,
        [_row(sample_id="prior", split="dev", sha256="e" * 64, text_hash="d" * 64)],
    )
    (config_root / "run.json").write_text(
        json.dumps({"role": {"manifest": "../prior.csv"}}), encoding="utf-8"
    )

    selection = select_unexposed_bonafide_candidate(
        source_manifest=source,
        source_split="test",
        project_root=tmp_path,
        config_root=config_root,
        created_at="2026-08-13T00:00:00Z",
    )

    assert [row.sample_id for row in selection.rows] == ["unexposed"]
    assert selection.receipt["exclusion_counts"] == {
        "sample_id": 0,
        "sha256": 0,
        "text_hash": 1,
        "parent_group_id": 0,
        "speaker_pseudo_id": 0,
    }


def test_bonafide_selection_excludes_the_entire_group_of_an_exposed_text(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.csv"
    exposed = _row(sample_id="exposed", split="test", sha256="a" * 64, text_hash="b" * 64)
    same_group = replace(
        _row(sample_id="same-group", split="test", sha256="c" * 64, text_hash="d" * 64),
        parent_group_id=exposed.parent_group_id,
        speaker_pseudo_id=exposed.speaker_pseudo_id,
    )
    unexposed = _row(sample_id="unexposed", split="test", sha256="e" * 64, text_hash="f" * 64)
    write_manifest(source, [exposed, same_group, unexposed])
    config_root = tmp_path / "configs"
    config_root.mkdir()
    prior = tmp_path / "prior.csv"
    write_manifest(
        prior,
        [_row(sample_id="prior", split="dev", sha256="1" * 64, text_hash="b" * 64)],
    )
    (config_root / "run.json").write_text(
        json.dumps({"role": {"manifest": "../prior.csv"}}), encoding="utf-8"
    )

    selection = select_unexposed_bonafide_candidate(
        source_manifest=source,
        source_split="test",
        project_root=tmp_path,
        config_root=config_root,
        created_at="2026-08-13T00:00:00Z",
    )

    assert [row.sample_id for row in selection.rows] == ["unexposed"]
    assert selection.receipt["group_tainted_rows"] == 2
