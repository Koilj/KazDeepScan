from __future__ import annotations


def manifest_mapping(**overrides: str) -> dict[str, str]:
    mapping = {
        "sample_id": "sample-1",
        "relative_path": "processed/ru/sample-1.wav",
        "sha256": "a" * 64,
        "split": "train",
        "label": "bonafide",
        "language": "ru",
        "code_switch": "false",
        "parent_group_id": "parent-1",
        "source_name": "consented",
        "source_license": "consent-v1",
        "rights_basis": "consent-001",
        "speaker_pseudo_id": "speaker-1",
        "text_id": "text-1",
        "text_hash": "text-hash-1",
        "duration_s": "3.0",
        "generator_family": "",
        "generator_name": "",
        "generator_version": "",
        "voice_id": "",
        "clone_consent_id": "",
        "device": "android",
        "capture_route": "local",
        "original_sr": "48000",
        "codec": "wav",
        "augmentation_chain": "",
        "augmentation_seed": "",
        "created_at": "2026-08-08T00:00:00Z",
    }
    mapping.update(overrides)
    return mapping
