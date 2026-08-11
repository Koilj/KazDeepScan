"""Validation for a frozen personal-research unseen-generator evaluation suite.

Several final tests may share an audited bona-fide corpus name without sharing a recording,
asset or text. They cannot therefore be expressed as separate source-matrix roles: that would
incorrectly call the corpus-level provenance field leakage. This contract keeps train/dev source
isolation, requires each held-out spoof family to be absent from train/dev, and proves that the
frozen final tests have no sample, audio or text overlap with one another.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from kds.data.licenses import LicenseLedgerEntry, LicenseLedgerError, validate_manifest_licenses
from kds.data.manifest import ManifestError, ManifestRow, load_manifest, validate_manifest

UNSEEN_GENERATOR_SUITE_SCHEMA_VERSION = 1


class UnseenGeneratorSuiteError(ValueError):
    def __init__(self, issues: Iterable[str]) -> None:
        self.issues = tuple(issues)
        super().__init__("\n".join(self.issues))


@dataclass(frozen=True, slots=True)
class SuiteSelection:
    manifest_path: Path
    source_split: str
    expected_source_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FrozenFinalTest:
    test_id: str
    selection: SuiteSelection
    expected_generator_families: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class UnseenGeneratorSuite:
    protocol_id: str
    train: SuiteSelection
    dev: SuiteSelection
    shared_final_source_ids: tuple[str, ...]
    final_tests: tuple[FrozenFinalTest, ...]


@dataclass(frozen=True, slots=True)
class FinalTestReport:
    test_id: str
    manifest_path: str
    rows: int
    source_ids: tuple[str, ...]
    generator_families: tuple[str, ...]
    label_counts: dict[str, int]


@dataclass(frozen=True, slots=True)
class UnseenGeneratorSuiteReport:
    protocol_id: str
    purpose: str
    train_rows: int
    dev_rows: int
    train_sources: tuple[str, ...]
    dev_sources: tuple[str, ...]
    train_dev_generator_families: tuple[str, ...]
    final_tests: tuple[FinalTestReport, ...]


@dataclass(frozen=True, slots=True)
class SelectedFinalTestRows:
    test_id: str
    rows: tuple[ManifestRow, ...]


@dataclass(frozen=True, slots=True)
class SelectedUnseenGeneratorSuiteRows:
    train: tuple[ManifestRow, ...]
    dev: tuple[ManifestRow, ...]
    final_tests: tuple[SelectedFinalTestRows, ...]


def load_unseen_generator_suite(path: Path) -> UnseenGeneratorSuite:
    """Load a strict versioned suite contract without accepting extra test configuration."""

    if not path.is_file():
        raise UnseenGeneratorSuiteError([f"Unseen-generator suite does not exist: {path}"])
    try:
        raw_value: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise UnseenGeneratorSuiteError(
            [f"Cannot read unseen-generator suite {path}: {error}"]
        ) from error
    if not isinstance(raw_value, dict):
        raise UnseenGeneratorSuiteError(["Unseen-generator suite root must be a JSON object."])
    raw = cast(dict[str, object], raw_value)
    _expect_exact_keys(
        raw,
        {
            "schema_version",
            "protocol_id",
            "purpose",
            "train",
            "dev",
            "shared_final_source_ids",
            "final_tests",
        },
        "Unseen-generator suite",
    )
    if raw["schema_version"] != UNSEEN_GENERATOR_SUITE_SCHEMA_VERSION:
        raise UnseenGeneratorSuiteError(
            [
                "Unseen-generator suite schema_version must be "
                f"{UNSEEN_GENERATOR_SUITE_SCHEMA_VERSION!r}, got {raw['schema_version']!r}."
            ]
        )
    if _required_string(raw, "purpose", "Unseen-generator suite") != "research":
        raise UnseenGeneratorSuiteError(
            ["Unseen-generator suite supports purpose='research' only."]
        )
    shared_sources = _string_list(raw["shared_final_source_ids"], "shared_final_source_ids")
    train = _parse_selection(raw["train"], "train", path.parent, allowed_split="train")
    dev = _parse_selection(raw["dev"], "dev", path.parent, allowed_split="dev")
    final_values = raw["final_tests"]
    if not isinstance(final_values, list) or len(final_values) < 3:
        raise UnseenGeneratorSuiteError(
            ["Unseen-generator suite must define at least three frozen final tests."]
        )
    final_tests = tuple(
        _parse_final_test(value, index, path.parent)
        for index, value in enumerate(final_values, start=1)
    )
    test_ids = [test.test_id for test in final_tests]
    if len(test_ids) != len(set(test_ids)):
        raise UnseenGeneratorSuiteError(["Unseen-generator suite has duplicate final test IDs."])
    return UnseenGeneratorSuite(
        protocol_id=_required_string(raw, "protocol_id", "Unseen-generator suite"),
        train=train,
        dev=dev,
        shared_final_source_ids=shared_sources,
        final_tests=final_tests,
    )


def validate_unseen_generator_suite(
    suite: UnseenGeneratorSuite, ledger: Mapping[str, LicenseLedgerEntry]
) -> UnseenGeneratorSuiteReport:
    """Validate source, family, rights and cross-final-test isolation without model fitting."""

    report, _rows = validate_and_select_unseen_generator_suite(suite, ledger)
    return report


def validate_and_select_unseen_generator_suite(
    suite: UnseenGeneratorSuite, ledger: Mapping[str, LicenseLedgerEntry]
) -> tuple[UnseenGeneratorSuiteReport, SelectedUnseenGeneratorSuiteRows]:
    """Validate a suite and return the exact rows validated in the same filesystem pass."""

    issues: list[str] = []
    train_rows = _load_selection(suite.train, "train", ledger, issues)
    dev_rows = _load_selection(suite.dev, "dev", ledger, issues)
    train_sources = _source_ids(train_rows)
    dev_sources = _source_ids(dev_rows)
    shared_train_dev_sources = sorted(set(train_sources).intersection(dev_sources))
    if shared_train_dev_sources:
        issues.append(
            "Source leakage between train/dev: " + ", ".join(shared_train_dev_sources) + "."
        )

    train_dev_rows = [*train_rows, *dev_rows]
    try:
        validate_manifest(train_dev_rows)
    except ManifestError as error:
        issues.extend(error.issues)
    seen_generator_families = _spoof_generator_families(train_dev_rows)
    final_selections: list[tuple[FrozenFinalTest, list[ManifestRow]]] = []
    reports: list[FinalTestReport] = []

    for final_test in suite.final_tests:
        rows = _load_selection(
            final_test.selection, f"final test {final_test.test_id!r}", ledger, issues
        )
        final_selections.append((final_test, rows))
        actual_families = _spoof_generator_families(rows)
        if actual_families != final_test.expected_generator_families:
            issues.append(
                f"Final test {final_test.test_id!r} expected generator families "
                f"{list(final_test.expected_generator_families)!r}, "
                f"found {list(actual_families)!r}."
            )
        overlap = sorted(set(actual_families).intersection(seen_generator_families))
        if overlap:
            issues.append(
                f"Final test {final_test.test_id!r} is not unseen to train/dev: "
                + ", ".join(overlap)
                + "."
            )
        final_sources = _source_ids(rows)
        train_dev_overlap = sorted(set(final_sources).intersection((*train_sources, *dev_sources)))
        if train_dev_overlap:
            issues.append(
                f"Final test {final_test.test_id!r} reuses train/dev source_name: "
                + ", ".join(train_dev_overlap)
                + "."
            )
        _validate_exact_pairs(final_test.test_id, rows, issues)
        reports.append(
            FinalTestReport(
                test_id=final_test.test_id,
                manifest_path=str(final_test.selection.manifest_path),
                rows=len(rows),
                source_ids=final_sources,
                generator_families=actual_families,
                label_counts={
                    label: sum(row.label == label for row in rows)
                    for label in ("bonafide", "spoof")
                },
            )
        )

    _validate_final_test_isolation(final_selections, suite.shared_final_source_ids, issues)
    combined_rows = [*train_dev_rows, *(row for _test, rows in final_selections for row in rows)]
    try:
        validate_manifest(combined_rows)
    except ManifestError as error:
        issues.extend(error.issues)
    if issues:
        raise UnseenGeneratorSuiteError(issues)
    report = UnseenGeneratorSuiteReport(
        protocol_id=suite.protocol_id,
        purpose="research",
        train_rows=len(train_rows),
        dev_rows=len(dev_rows),
        train_sources=train_sources,
        dev_sources=dev_sources,
        train_dev_generator_families=seen_generator_families,
        final_tests=tuple(reports),
    )
    selections = SelectedUnseenGeneratorSuiteRows(
        train=tuple(train_rows),
        dev=tuple(dev_rows),
        final_tests=tuple(
            SelectedFinalTestRows(test_id=final_test.test_id, rows=tuple(rows))
            for final_test, rows in final_selections
        ),
    )
    return report, selections


def _load_selection(
    selection: SuiteSelection,
    label: str,
    ledger: Mapping[str, LicenseLedgerEntry],
    issues: list[str],
) -> list[ManifestRow]:
    try:
        all_rows = load_manifest(selection.manifest_path)
        validate_manifest(all_rows)
    except ManifestError as error:
        issues.extend(error.issues)
        return []
    rows = [row for row in all_rows if row.split == selection.source_split]
    if not rows:
        issues.append(f"{label} selects no rows with split={selection.source_split!r}.")
        return []
    actual_sources = _source_ids(rows)
    if actual_sources != selection.expected_source_ids:
        issues.append(
            f"{label} expected sources {list(selection.expected_source_ids)!r}, "
            f"found {list(actual_sources)!r}."
        )
    if {row.label for row in rows} != {"bonafide", "spoof"}:
        issues.append(f"{label} must include both bonafide and spoof rows.")
    try:
        validate_manifest_licenses(rows, ledger)
    except LicenseLedgerError as error:
        issues.extend(error.issues)
    return rows


def _validate_exact_pairs(test_id: str, rows: Iterable[ManifestRow], issues: list[str]) -> None:
    by_text: dict[str, list[ManifestRow]] = defaultdict(list)
    for row in rows:
        by_text[row.text_hash].append(row)
    invalid = [
        text_hash
        for text_hash, pair in by_text.items()
        if len(pair) != 2 or {row.label for row in pair} != {"bonafide", "spoof"}
    ]
    if invalid:
        issues.append(
            f"Final test {test_id!r} must have one bona-fide/spoof pair per text_hash; "
            f"invalid groups={len(invalid)}."
        )


def _validate_final_test_isolation(
    selections: Iterable[tuple[FrozenFinalTest, list[ManifestRow]]],
    shared_source_ids: tuple[str, ...],
    issues: list[str],
) -> None:
    seen_samples: dict[str, str] = {}
    seen_assets: dict[str, str] = {}
    seen_texts: dict[str, str] = {}
    seen_nonshared_sources: dict[str, str] = {}
    for final_test, rows in selections:
        sources = set(_source_ids(rows))
        missing_shared = sorted(set(shared_source_ids).difference(sources))
        if missing_shared:
            issues.append(
                f"Final test {final_test.test_id!r} is missing shared bona-fide source IDs: "
                + ", ".join(missing_shared)
                + "."
            )
        for source_id in sorted(sources.difference(shared_source_ids)):
            previous = seen_nonshared_sources.setdefault(source_id, final_test.test_id)
            if previous != final_test.test_id:
                issues.append(
                    f"Final tests {previous!r}/{final_test.test_id!r} reuse "
                    f"synthetic source_name={source_id!r}."
                )
        for row in rows:
            _record_cross_test_value(
                seen_samples, row.sample_id, final_test.test_id, "sample_id", issues
            )
            _record_cross_test_value(seen_assets, row.sha256, final_test.test_id, "sha256", issues)
            _record_cross_test_value(
                seen_texts, row.text_hash, final_test.test_id, "text_hash", issues
            )


def _record_cross_test_value(
    seen: dict[str, str], value: str, test_id: str, field: str, issues: list[str]
) -> None:
    previous = seen.setdefault(value, test_id)
    if previous != test_id:
        issues.append(f"Final tests {previous!r}/{test_id!r} overlap by {field}={value!r}.")


def _parse_selection(
    value: object, label: str, base_directory: Path, *, allowed_split: str
) -> SuiteSelection:
    if not isinstance(value, dict):
        raise UnseenGeneratorSuiteError([f"{label} selection must be a JSON object."])
    raw = cast(dict[str, object], value)
    _expect_exact_keys(
        raw, {"manifest", "source_split", "expected_source_ids"}, f"{label} selection"
    )
    source_split = _required_string(raw, "source_split", f"{label} selection")
    if source_split != allowed_split:
        raise UnseenGeneratorSuiteError(
            [f"{label} selection source_split must be {allowed_split!r}."]
        )
    manifest = Path(_required_string(raw, "manifest", f"{label} selection"))
    if manifest.is_absolute():
        raise UnseenGeneratorSuiteError([f"{label} selection manifest must be relative."])
    return SuiteSelection(
        manifest_path=(base_directory / manifest).resolve(),
        source_split=source_split,
        expected_source_ids=_string_list(
            raw["expected_source_ids"], f"{label} expected_source_ids"
        ),
    )


def _parse_final_test(value: object, index: int, base_directory: Path) -> FrozenFinalTest:
    label = f"Final test {index}"
    if not isinstance(value, dict):
        raise UnseenGeneratorSuiteError([f"{label} must be a JSON object."])
    raw = cast(dict[str, object], value)
    _expect_exact_keys(
        raw,
        {"id", "manifest", "source_split", "expected_source_ids", "expected_generator_families"},
        label,
    )
    selection = _parse_selection(
        {
            "manifest": raw["manifest"],
            "source_split": raw["source_split"],
            "expected_source_ids": raw["expected_source_ids"],
        },
        label,
        base_directory,
        allowed_split="test",
    )
    return FrozenFinalTest(
        test_id=_required_string(raw, "id", label),
        selection=selection,
        expected_generator_families=_string_list(
            raw["expected_generator_families"], f"{label} expected_generator_families"
        ),
    )


def _source_ids(rows: Iterable[ManifestRow]) -> tuple[str, ...]:
    return tuple(sorted({row.source_name for row in rows}))


def _spoof_generator_families(rows: Iterable[ManifestRow]) -> tuple[str, ...]:
    return tuple(sorted({row.generator_family for row in rows if row.label == "spoof"}))


def _expect_exact_keys(raw: Mapping[str, object], expected: set[str], label: str) -> None:
    unknown = sorted(set(raw).difference(expected))
    missing = sorted(expected.difference(raw))
    if unknown or missing:
        issues: list[str] = []
        if missing:
            issues.append("missing fields: " + ", ".join(missing))
        if unknown:
            issues.append("unknown fields: " + ", ".join(unknown))
        raise UnseenGeneratorSuiteError([f"{label} has " + "; ".join(issues) + "."])


def _required_string(raw: Mapping[str, object], name: str, label: str) -> str:
    value = raw.get(name)
    if not isinstance(value, str) or not value.strip():
        raise UnseenGeneratorSuiteError([f"{label} field {name!r} must be a non-empty string."])
    return value.strip()


def _string_list(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise UnseenGeneratorSuiteError([f"{label} must be a non-empty JSON array."])
    values: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise UnseenGeneratorSuiteError([f"{label} must contain non-empty strings."])
        values.append(item.strip())
    if len(values) != len(set(values)):
        raise UnseenGeneratorSuiteError([f"{label} must not contain duplicates."])
    return tuple(sorted(values))
