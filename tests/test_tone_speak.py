from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import kds.data.tone_speak as tone_speak

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")


def _artifact(
    relative_path: str, split: str | None, rows: int | None, path: Path
) -> tone_speak.ToneSpeakExpectedArtifact:
    return tone_speak.ToneSpeakExpectedArtifact(
        relative_path=relative_path,
        split=split,
        expected_rows=rows,
        size_bytes=path.stat().st_size,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def _write_parquet(path: Path, *, audio_number: str, voice: str, text: str) -> None:
    schema = pa.schema(
        [
            pa.field(
                "audio",
                pa.struct([pa.field("bytes", pa.binary()), pa.field("path", pa.string())]),
            ),
            pa.field("text", pa.string()),
            pa.field("text_description", pa.string()),
            pa.field("voice_name", pa.string()),
        ],
        metadata={
            b"huggingface": json.dumps(
                {"info": {"features": {"audio": {"sampling_rate": 24_000}}}}
            ).encode("utf-8")
        },
    )
    table = pa.Table.from_pylist(
        [
            {
                "audio": {"bytes": b"an-mp3-payload", "path": f"{audio_number}_{voice}.mp3"},
                "text": text,
                "text_description": "Russian synthetic speech.",
                "voice_name": voice,
            }
        ],
        schema=schema,
    )
    pq.write_table(table, path)


def _fixture_release(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    train_voice: str = "alloy",
    validation_voice: str = "alloy",
    train_text: str = "Русский текст один.",
    validation_text: str = "Русский текст два.",
) -> Path:
    root = tmp_path / "release"
    data_root = root / "data"
    data_root.mkdir(parents=True)
    (root / ".gitattributes").write_text("*.parquet filter=lfs\n", encoding="utf-8")
    (root / "README.md").write_text("# ToneSpeak\n", encoding="utf-8")
    train = data_root / "train-00000-of-00001.parquet"
    validation = data_root / "validation-00000-of-00001.parquet"
    _write_parquet(train, audio_number="00001", voice=train_voice, text=train_text)
    _write_parquet(validation, audio_number="00002", voice=validation_voice, text=validation_text)
    monkeypatch.setattr(
        tone_speak,
        "TONE_SPEAK_EXPECTED_ARTIFACTS",
        (
            _artifact(".gitattributes", None, None, root / ".gitattributes"),
            _artifact("README.md", None, None, root / "README.md"),
            _artifact("data/train-00000-of-00001.parquet", "train", 1, train),
            _artifact("data/validation-00000-of-00001.parquet", "validation", 1, validation),
        ),
    )
    monkeypatch.setattr(tone_speak, "TONE_SPEAK_EXPECTED_VOICES", frozenset({"alloy"}))
    monkeypatch.setattr(tone_speak, "_inspect_audio", lambda _payload, context: (24_000, 24_000))
    return root


def test_audit_tone_speak_release_checks_complete_embedded_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _fixture_release(tmp_path, monkeypatch)

    audit = tone_speak.audit_tone_speak_release(root)

    assert audit.rows_by_split == {"train": 1, "validation": 1}
    assert audit.voice_counts_by_split == {
        "train": {"alloy": 1},
        "validation": {"alloy": 1},
    }
    assert audit.audio_records == 2
    assert audit.duplicate_audio_payloads == 1
    assert audit.cross_split_normalized_texts == 0
    assert audit.records_without_cyrillic == 0


def test_audit_tone_speak_release_rejects_unknown_voice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _fixture_release(tmp_path, monkeypatch, train_voice="other")

    with pytest.raises(tone_speak.ToneSpeakAuditError, match="unsupported voice_name"):
        tone_speak.audit_tone_speak_release(root)


def test_audit_tone_speak_release_rejects_cross_split_text_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _fixture_release(
        tmp_path,
        monkeypatch,
        train_text="Русский текст один.",
        validation_text="Русский текст один.",
    )

    with pytest.raises(tone_speak.ToneSpeakAuditError, match="shared by train and validation"):
        tone_speak.audit_tone_speak_release(root)


def _record(voice_name: str, number: int) -> tone_speak.ToneSpeakRecord:
    text = f"Русский текст {voice_name} {number}."
    return tone_speak.ToneSpeakRecord(
        source_split="validation",
        parquet_path="data/validation-00000-of-00001.parquet",
        embedded_path=f"{number:05d}_{voice_name}.mp3",
        text=text,
        text_hash=__import__("hashlib").sha256(text.encode("utf-8")).hexdigest(),
        voice_name=voice_name,
    )


def test_select_tone_speak_validation_records_balances_voices_and_excludes_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tone_speak, "TONE_SPEAK_EXPECTED_VOICES", frozenset({"alloy", "coral"}))
    records = [_record("alloy", 1), _record("alloy", 2), _record("coral", 3), _record("coral", 4)]

    selected = tone_speak.select_tone_speak_validation_records(
        records,
        per_voice=1,
        seed="seed",
        excluded_text_hashes={records[0].text_hash},
    )

    assert len(selected) == 2
    assert {record.voice_name for record in selected} == {"alloy", "coral"}
    assert records[0].text_hash not in {record.text_hash for record in selected}


def test_tone_speak_ood_manifest_rows_keep_per_row_voice_provenance() -> None:
    records = [_record("alloy", 1), _record("coral", 2)]
    assets = {
        record.embedded_path: tone_speak.ToneSpeakExtractedAsset(
            embedded_path=record.embedded_path,
            relative_path=f"raw/tone_speak_ru_v1/slices/ood/{record.embedded_path}",
            sha256=f"{index:064x}",
            duration_s=3.0,
            original_sr=24_000,
            codec="mp3",
        )
        for index, record in enumerate(records, start=1)
    }

    rows = tone_speak.tone_speak_ood_manifest_rows(
        records, assets, created_at="2026-08-11T00:00:00Z"
    )

    assert {row.split for row in rows} == {"ood"}
    assert {row.label for row in rows} == {"spoof"}
    assert {row.generator_name for row in rows} == {"openai_gpt_4o_mini_tts"}
    assert {row.voice_id for row in rows} == {
        "tone_speak_ru_v1:voice:alloy",
        "tone_speak_ru_v1:voice:coral",
    }
