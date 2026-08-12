from __future__ import annotations

from kds.data.kazakhtts_text import (
    LATIN_TOKEN_MAP,
    normalize_kazakhtts_stage_c_text,
    number_to_kazakh,
    number_to_russian,
)


def test_locale_number_expansion_is_deterministic() -> None:
    assert number_to_kazakh(2017) == "екі мың он жеті"
    assert number_to_kazakh(10_000) == "он мың"
    assert number_to_russian(1920) == "одна тысяча девятьсот двадцать"
    assert number_to_russian(800_000) == "восемьсот тысяч"


def test_kazakh_normalization_maps_numbers_latin_and_punctuation() -> None:
    result = normalize_kazakhtts_stage_c_text(
        "METI және Apple 2017 жылы «ауыр емес» деді", "kk"
    )

    assert result.normalized == "мети және эппл екі мың он жеті жылы ауыр емес деді"
    assert "explicit_latin_token_map" in result.operations
    assert "locale_cardinal_number_expansion" in result.operations


def test_russian_normalization_handles_decade_and_grouped_number() -> None:
    result = normalize_kazakhtts_stage_c_text(
        "В 1920-е годы было 40 000 жителей — это факт", "ru"
    )

    assert result.normalized == (
        "в тысяча девятьсот двадцатые годы было сорока тысяч жителей - это факт"
    )


def test_latin_mapping_covers_the_frozen_explicit_vocabulary() -> None:
    assert len(LATIN_TOKEN_MAP) == 29
    result = normalize_kazakhtts_stage_c_text("TMZ, USGS және pН және h", "kk")
    assert result.normalized == "ти эм зи, ю эс джи эс және пи эйч және эйч"


def test_adjacent_latin_acronym_and_kazakh_suffix_are_joined() -> None:
    result = normalize_kazakhtts_stage_c_text("f1 және 70-тен 100-ге дейін", "kk")
    assert result.normalized == "эф бір және жетпістен жүзге дейін"
