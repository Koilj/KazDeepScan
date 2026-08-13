from __future__ import annotations

import hashlib
import json
from pathlib import Path

from kds.data.common_voice import CommonVoiceRecord
from kds.data.manifest import ManifestRow, write_manifest
from kds.eval.common_voice_metadata_screen import screen_common_voice_ru_test_metadata
from tests.factories import manifest_mapping


def _record(clip_name: str, client_id: str, sentence: str) -> CommonVoiceRecord:
    return CommonVoiceRecord(
        clip_name=clip_name,
        split="test",
        client_id=client_id,
        sentence_id=f"sentence-{clip_name}",
        sentence=sentence,
    )


def _manifest_row(*, sample_id: str, client_id: str, text_hash: str) -> ManifestRow:
    return ManifestRow.from_mapping(
        manifest_mapping(
            sample_id=sample_id,
            source_name="common_voice_ru_v24",
            source_license="CC0-1.0",
            parent_group_id=f"common_voice_ru_v24:client:{client_id}",
            speaker_pseudo_id=f"common_voice_ru_v24:client:{client_id}",
            text_hash=text_hash,
        ),
        2,
    )


def test_metadata_screen_taints_entire_client_group_on_historical_overlap(tmp_path: Path) -> None:
    project = tmp_path / "project"
    config_root = project / "configs" / "research"
    manifest_root = project / "data" / "manifests"
    config_root.mkdir(parents=True)
    manifest_root.mkdir(parents=True)
    records = (
        _record("clean-a.mp3", "clean", "чистый текст"),
        _record("clean-b.mp3", "clean", "ещё чистый текст"),
        _record("tainted-a.mp3", "tainted", "известный текст"),
        _record("tainted-b.mp3", "tainted", "другая строка той же группы"),
    )
    known_text_hash = hashlib.sha256("известный текст".encode()).hexdigest()
    configured = manifest_root / "configured.csv"
    write_manifest(
        configured,
        [
            _manifest_row(
                sample_id="other_source:historical", client_id="other", text_hash=known_text_hash
            )
        ],
    )
    (config_root / "role.json").write_text(
        json.dumps({"manifest": "../../data/manifests/configured.csv"}), encoding="utf-8"
    )
    inventory = manifest_root / "inventory.csv"
    write_manifest(
        inventory,
        [
            _manifest_row(
                sample_id="common_voice_ru_v24:unrelated", client_id="outside", text_hash="f" * 64
            )
        ],
    )

    screen = screen_common_voice_ru_test_metadata(
        records=records,
        project_root=project,
        config_root=config_root,
        manifest_root=manifest_root,
        created_at="2026-08-13T00:00:00Z",
    )

    assert [identity.clip_name for identity in screen.surviving] == [
        "clean-a.mp3",
        "clean-b.mp3",
    ]
    strict_group_exclusion = screen.receipt["strict_group_exclusion"]
    claims = screen.receipt["claims"]
    assert isinstance(strict_group_exclusion, dict)
    assert isinstance(claims, dict)
    assert strict_group_exclusion["tainted_client_groups"] == 1
    assert strict_group_exclusion["surviving_records"] == 2
    assert claims["selection_frozen"] is False
