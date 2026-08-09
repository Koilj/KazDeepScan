from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, replace
from typing import Literal

from kds.data.manifest import ManifestRow

SplitName = Literal["train", "dev", "test"]


@dataclass(frozen=True, slots=True)
class SplitConfig:
    train_ratio: float = 0.8
    dev_ratio: float = 0.1
    test_ratio: float = 0.1
    seed: str = "20260808"
    preserve_ood: bool = True

    def __post_init__(self) -> None:
        ratios = (self.train_ratio, self.dev_ratio, self.test_ratio)
        if any(ratio <= 0.0 for ratio in ratios):
            raise ValueError("Every split ratio must be positive.")
        if not math.isclose(sum(ratios), 1.0, abs_tol=1e-9):
            raise ValueError("train_ratio + dev_ratio + test_ratio must equal 1.0.")
        if not self.seed:
            raise ValueError("seed must not be empty.")


class GroupSplitter:
    """Assign each connected group/speaker/text component to one train/dev/test split."""

    def __init__(self, config: SplitConfig | None = None) -> None:
        self.config = config or SplitConfig()

    def assign_group(self, parent_group_id: str) -> SplitName:
        if not parent_group_id:
            raise ValueError("parent_group_id must not be empty.")
        digest = hashlib.sha256(f"{self.config.seed}:{parent_group_id}".encode()).digest()
        value = int.from_bytes(digest[:8], byteorder="big") / 2**64
        if value < self.config.train_ratio:
            return "train"
        if value < self.config.train_ratio + self.config.dev_ratio:
            return "dev"
        return "test"

    def assign_rows(self, rows: list[ManifestRow]) -> list[ManifestRow]:
        assignable_indices = [
            index
            for index, row in enumerate(rows)
            if not (self.config.preserve_ood and row.split == "ood")
        ]
        ood_indices = set(range(len(rows))).difference(assignable_indices)
        self._reject_ood_leakage(rows, assignable_indices, ood_indices)

        parent: dict[int, int] = {index: index for index in assignable_indices}

        def root(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def union(first: int, second: int) -> None:
            first_root = root(first)
            second_root = root(second)
            if first_root != second_root:
                parent[second_root] = first_root

        first_seen: dict[tuple[str, str], int] = {}
        for index in assignable_indices:
            row = rows[index]
            for field in ("parent_group_id", "speaker_pseudo_id", "text_hash"):
                key = (field, getattr(row, field))
                previous = first_seen.setdefault(key, index)
                union(previous, index)

        component_keys: dict[int, list[str]] = {}
        for index in assignable_indices:
            component_keys.setdefault(root(index), []).extend(
                f"{field}:{getattr(rows[index], field)}"
                for field in ("parent_group_id", "speaker_pseudo_id", "text_hash")
            )
        assignments = {
            component_root: self.assign_group(min(keys))
            for component_root, keys in component_keys.items()
        }
        return [
            row
            if index in ood_indices
            else replace(row, split=assignments[root(index)])
            for index, row in enumerate(rows)
        ]

    @staticmethod
    def _reject_ood_leakage(
        rows: list[ManifestRow],
        assignable_indices: list[int],
        ood_indices: set[int],
    ) -> None:
        if not ood_indices:
            return
        ood_values = {
            (field, getattr(rows[index], field))
            for index in ood_indices
            for field in ("parent_group_id", "speaker_pseudo_id", "text_hash")
        }
        overlaps = sorted(
            (field, value)
            for index in assignable_indices
            for field in ("parent_group_id", "speaker_pseudo_id", "text_hash")
            for value in (getattr(rows[index], field),)
            if (field, value) in ood_values
        )
        if overlaps:
            field, value = overlaps[0]
            raise ValueError(
                "Cannot preserve an ood row that leaks into train/dev/test: "
                f"{field}={value!r}."
            )
