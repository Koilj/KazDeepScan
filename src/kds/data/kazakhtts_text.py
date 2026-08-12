"""Auditable text normalization for the fixed Stage-C KazakhTTS character inventory."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Final

KAZAKHTTS_TEXT_NORMALIZER_ID: Final = "stage_c_kazakhtts_text_v1"

LATIN_TOKEN_MAP: Final[dict[str, str]] = {
    "aerosmith": "аэросмит",
    "apple": "эппл",
    "asus": "асус",
    "broadsides": "бродсайдс",
    "civilis": "цивилис",
    "civis": "цивис",
    "civitas": "цивитас",
    "conirostris": "конирострис",
    "dunlap": "данлэп",
    "eee": "и и и",
    "f": "эф",
    "fkp": "эф кей пи",
    "ftir": "эф ти ай ар",
    "gbp": "джи би пи",
    "geospiza": "геоспиза",
    "h": "эйч",
    "m": "эм",
    "mdt": "эм ди ти",
    "meti": "мети",
    "p": "пи",
    "radio": "радио",
    "sanparks": "санпаркс",
    "scotturb": "скоттурб",
    "swapo": "свапо",
    "tmz": "ти эм зи",
    "toginet": "тоджинет",
    "usgs": "ю эс джи эс",
    "usoc": "ю эс о си",
    "xdr": "икс ди ар",
}

_LATIN = re.compile(r"[a-z]+")
_NUMBER = re.compile(r"\d{1,3}(?: \d{3})+|\d+")
_CURRENCY = re.compile(r"\$(\d{1,3}(?: \d{3})*|\d+)")
_SPACES = re.compile(r"\s+")
_SPACE_BEFORE_PUNCTUATION = re.compile(r"\s+([,.:;?!])")
_KK_SUFFIX = re.compile(r"-(ға|ге|қа|ке|да|де|та|те|дан|ден|тан|тен|нан|нен)\b")

RU_EXACT_PHRASE_MAP: Final[dict[str, str]] = {
    "40 000 жителей": "сорока тысяч жителей",
    "в 1920-е годы": "в тысяча девятьсот двадцатые годы",
    "35 мм": "тридцать пять миллиметров",
    "70 километров": "семидесяти километров",
    "120 километрах": "ста двадцати километрах",
    "53-летний": "пятидесятитрёхлетний",
    "800 000 солдат": "восьмисот тысяч солдат",
    "10 000 лет": "десяти тысяч лет",
    "более чем 70 острейшими": "более чем семьюдесятью острейшими",
    "около 70 км": "около семидесяти километров",
    "100 км": "ста километров",
}

_KK_ONES = ("", "бір", "екі", "үш", "төрт", "бес", "алты", "жеті", "сегіз", "тоғыз")
_KK_TENS = ("", "он", "жиырма", "отыз", "қырық", "елу", "алпыс", "жетпіс", "сексен", "тоқсан")
_RU_ONES = ("", "один", "два", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять")
_RU_TEENS = (
    "десять",
    "одиннадцать",
    "двенадцать",
    "тринадцать",
    "четырнадцать",
    "пятнадцать",
    "шестнадцать",
    "семнадцать",
    "восемнадцать",
    "девятнадцать",
)
_RU_TENS = (
    "",
    "",
    "двадцать",
    "тридцать",
    "сорок",
    "пятьдесят",
    "шестьдесят",
    "семьдесят",
    "восемьдесят",
    "девяносто",
)
_RU_HUNDREDS = (
    "",
    "сто",
    "двести",
    "триста",
    "четыреста",
    "пятьсот",
    "шестьсот",
    "семьсот",
    "восемьсот",
    "девятьсот",
)


class KazakhTtsTextError(ValueError):
    """Raised when a selected text cannot be normalized without an unknown mapping."""


@dataclass(frozen=True, slots=True)
class KazakhTtsNormalizedText:
    original: str
    normalized: str
    normalized_sha256: str
    operations: tuple[str, ...]


def _under_thousand_kk(value: int) -> str:
    parts: list[str] = []
    hundreds, remainder = divmod(value, 100)
    if hundreds:
        if hundreds > 1:
            parts.append(_KK_ONES[hundreds])
        parts.append("жүз")
    tens, ones = divmod(remainder, 10)
    if tens:
        parts.append(_KK_TENS[tens])
    if ones:
        parts.append(_KK_ONES[ones])
    return " ".join(parts)


def number_to_kazakh(value: int) -> str:
    if value < 0 or value >= 1_000_000:
        raise KazakhTtsTextError(f"Kazakh number is outside supported range: {value}.")
    if value == 0:
        return "нөл"
    thousands, remainder = divmod(value, 1000)
    parts: list[str] = []
    if thousands:
        parts.extend((_under_thousand_kk(thousands), "мың"))
    if remainder:
        parts.append(_under_thousand_kk(remainder))
    return " ".join(parts)


def _under_thousand_ru(value: int) -> str:
    parts: list[str] = []
    hundreds, remainder = divmod(value, 100)
    if hundreds:
        parts.append(_RU_HUNDREDS[hundreds])
    if 10 <= remainder <= 19:
        parts.append(_RU_TEENS[remainder - 10])
    else:
        tens, ones = divmod(remainder, 10)
        if tens:
            parts.append(_RU_TENS[tens])
        if ones:
            parts.append(_RU_ONES[ones])
    return " ".join(parts)


def number_to_russian(value: int) -> str:
    if value < 0 or value >= 1_000_000:
        raise KazakhTtsTextError(f"Russian number is outside supported range: {value}.")
    if value == 0:
        return "ноль"
    thousands, remainder = divmod(value, 1000)
    parts: list[str] = []
    if thousands:
        if thousands == 1:
            parts.append("одна тысяча")
        elif thousands == 2:
            parts.append("две тысячи")
        else:
            suffix = "тысячи" if thousands in {3, 4} else "тысяч"
            parts.append(f"{_under_thousand_ru(thousands)} {suffix}")
    if remainder:
        parts.append(_under_thousand_ru(remainder))
    return " ".join(parts)


def normalize_kazakhtts_stage_c_text(text: str, language: str) -> KazakhTtsNormalizedText:
    """Normalize only the frozen RU/KK/mixed surface forms with explicit transformations."""

    if language not in {"ru", "kk", "mixed"}:
        raise KazakhTtsTextError(f"Unsupported Stage-C language: {language!r}.")
    normalized = _SPACES.sub(" ", text.lower().strip())
    if not normalized:
        raise KazakhTtsTextError("Stage-C input text is empty.")
    operations: list[str] = ["unicode_lower_and_space_canonicalization"]
    if language == "ru":
        changed = False
        for source, replacement in RU_EXACT_PHRASE_MAP.items():
            if source in normalized:
                normalized = normalized.replace(source, replacement)
                changed = True
        if changed:
            operations.append("ru_exact_inflected_quantity_map")
    else:
        replacements = {
            "км/сағ": "километр сағатына",
            "б.з.": "біздің заманымыздың",
        }
        changed = False
        for source, replacement in replacements.items():
            if source in normalized:
                normalized = normalized.replace(source, replacement)
                changed = True
        normalized, millimeter_count = re.subn(
            r"(?<!\w)мм(?!\w)", "миллиметр", normalized
        )
        changed = changed or bool(millimeter_count)
        if changed:
            operations.append("kk_abbreviation_expansion")

    number_renderer = number_to_russian if language == "ru" else number_to_kazakh

    def currency(match: re.Match[str]) -> str:
        raw = match.group(1).replace(" ", "")
        return f"{number_renderer(int(raw))} доллар"

    if _CURRENCY.search(normalized):
        normalized = _CURRENCY.sub(currency, normalized)
        operations.append("usd_amount_expansion")

    if "pн" in normalized:
        normalized = normalized.replace("pн", "пи эйч")
        operations.append("mixed_script_ph_expansion")
    normalized, ph_count = re.subn(r"(?<!\w)рн(?!\w)", "пи эйч", normalized)
    if ph_count:
        operations.append("cyrillic_ph_expansion")

    unknown_latin = sorted(set(_LATIN.findall(normalized)).difference(LATIN_TOKEN_MAP))
    if unknown_latin:
        raise KazakhTtsTextError(
            "Stage-C text contains unmapped Latin tokens: " + " ".join(unknown_latin)
        )
    if _LATIN.search(normalized):
        normalized = _LATIN.sub(
            lambda match: f" {LATIN_TOKEN_MAP[match.group(0)]} ", normalized
        )
        operations.append("explicit_latin_token_map")

    if _NUMBER.search(normalized):
        def number(match: re.Match[str]) -> str:
            raw = match.group(0)
            compact = raw.replace(" ", "")
            if len(compact) > 1 and compact.startswith("0"):
                zero = "ноль" if language == "ru" else "нөл"
                digits = _RU_ONES if language == "ru" else _KK_ONES
                return " ".join(zero if digit == "0" else digits[int(digit)] for digit in compact)
            return number_renderer(int(compact))

        normalized = _NUMBER.sub(number, normalized)
        operations.append("locale_cardinal_number_expansion")
    if language != "ru":
        normalized, suffix_count = _KK_SUFFIX.subn(r"\1", normalized)
        if suffix_count:
            operations.append("kk_numeric_suffix_join")

    punctuation_map = str.maketrans(
        {
            "«": " ",
            "»": " ",
            '"': " ",
            "[": " ",
            "]": " ",
            "—": "-",
            "/": " ",
        }
    )
    translated = normalized.translate(punctuation_map)
    if translated != normalized:
        normalized = translated
        operations.append("unsupported_punctuation_canonicalization")
    normalized = _SPACES.sub(" ", normalized).strip()
    normalized = _SPACE_BEFORE_PUNCTUATION.sub(r"\1", normalized)
    if not normalized:
        raise KazakhTtsTextError("Stage-C text became empty after normalization.")
    return KazakhTtsNormalizedText(
        original=text,
        normalized=normalized,
        normalized_sha256=hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
        operations=tuple(dict.fromkeys(operations)),
    )
