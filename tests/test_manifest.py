from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from kds.data.manifest import (
    ManifestError,
    ManifestRow,
    load_manifest,
    validate_manifest,
    write_manifest,
)
from tests.factories import manifest_mapping


class ManifestTests(unittest.TestCase):
    def test_valid_bonafide_manifest_passes(self) -> None:
        row = ManifestRow.from_mapping(manifest_mapping(), row_number=2)

        validate_manifest([row])

    def test_spoof_requires_reproducible_generator_provenance(self) -> None:
        mapping = manifest_mapping(label="spoof")

        with self.assertRaisesRegex(ManifestError, "spoof requires provenance"):
            ManifestRow.from_mapping(mapping, row_number=2)

    def test_relative_path_cannot_escape_audio_root(self) -> None:
        mapping = manifest_mapping(relative_path="../outside.wav")

        with self.assertRaisesRegex(ManifestError, "relative_path"):
            ManifestRow.from_mapping(mapping, row_number=2)

    def test_write_manifest_is_reloaded_with_stable_field_order(self) -> None:
        row = ManifestRow.from_mapping(manifest_mapping(), row_number=2)
        with TemporaryDirectory() as directory:
            output = Path(directory) / "slice.csv"
            write_manifest(output, [row])
            reloaded = load_manifest(output)

            self.assertEqual(reloaded, [row])
            with self.assertRaisesRegex(ManifestError, "Refusing to overwrite"):
                write_manifest(output, [row])

    def test_parent_group_leakage_is_rejected(self) -> None:
        train = ManifestRow.from_mapping(manifest_mapping(), row_number=2)
        dev = ManifestRow.from_mapping(
            manifest_mapping(
                sample_id="sample-2",
                sha256="b" * 64,
                split="dev",
                speaker_pseudo_id="speaker-2",
                text_id="text-2",
                text_hash="text-hash-2",
            ),
            row_number=3,
        )

        with self.assertRaisesRegex(ManifestError, "parent_group_id"):
            validate_manifest([train, dev])

    def test_ood_generator_must_be_unseen_elsewhere(self) -> None:
        train = ManifestRow.from_mapping(
            manifest_mapping(
                label="spoof",
                generator_family="piper",
                generator_name="piper-ru",
                generator_version="1.0",
                voice_id="voice-a",
            ),
            row_number=2,
        )
        ood = ManifestRow.from_mapping(
            manifest_mapping(
                sample_id="sample-2",
                sha256="b" * 64,
                split="ood",
                label="spoof",
                parent_group_id="parent-2",
                speaker_pseudo_id="speaker-2",
                text_id="text-2",
                text_hash="text-hash-2",
                generator_family="piper",
                generator_name="piper-ru",
                generator_version="1.0",
                voice_id="voice-b",
            ),
            row_number=3,
        )

        with self.assertRaisesRegex(ManifestError, "OOD generator families"):
            validate_manifest([train, ood], require_ood_generator=True)

    def test_other_language_is_limited_to_cross_lingual_ood(self) -> None:
        with self.assertRaisesRegex(ManifestError, "language=other"):
            ManifestRow.from_mapping(manifest_mapping(language="other"), row_number=2)

        ood = ManifestRow.from_mapping(
            manifest_mapping(language="other", split="ood"), row_number=2
        )

        validate_manifest([ood])


if __name__ == "__main__":
    unittest.main()
