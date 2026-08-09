from __future__ import annotations

from kds.data.manifest import ManifestRow
from kds.data.split import GroupSplitter, SplitConfig
from tests.factories import manifest_mapping


def test_group_splitter_keeps_one_parent_group_together() -> None:
    first = ManifestRow.from_mapping(manifest_mapping(sample_id="sample-1"), row_number=2)
    second = ManifestRow.from_mapping(
        manifest_mapping(
            sample_id="sample-2",
            sha256="b" * 64,
            relative_path="processed/ru/sample-2.wav",
            parent_group_id="parent-1",
            speaker_pseudo_id="speaker-2",
            text_id="text-2",
            text_hash="text-hash-2",
        ),
        row_number=3,
    )

    assigned = GroupSplitter(SplitConfig(seed="fixed")).assign_rows([first, second])

    assert assigned[0].split == assigned[1].split


def test_group_splitter_preserves_preassigned_ood_rows() -> None:
    ood = ManifestRow.from_mapping(
        manifest_mapping(split="ood", parent_group_id="ood-parent"), row_number=2
    )

    assigned = GroupSplitter().assign_rows([ood])

    assert assigned[0].split == "ood"


def test_group_splitter_keeps_shared_text_in_one_split() -> None:
    first = ManifestRow.from_mapping(
        manifest_mapping(parent_group_id="parent-1", text_hash="shared-text"), row_number=2
    )
    second = ManifestRow.from_mapping(
        manifest_mapping(
            sample_id="sample-2",
            sha256="b" * 64,
            relative_path="processed/ru/sample-2.wav",
            parent_group_id="parent-2",
            speaker_pseudo_id="speaker-2",
            text_id="text-2",
            text_hash="shared-text",
        ),
        row_number=3,
    )

    assigned = GroupSplitter(SplitConfig(seed="fixed")).assign_rows([first, second])

    assert assigned[0].split == assigned[1].split
