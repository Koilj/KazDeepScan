from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any

from kds.data.manifest import load_manifest


def _audit_script() -> Any:
    spec = spec_from_file_location(
        "kds_common_voice_silero_v5_5_exposure",
        "scripts/audit_common_voice_ru_v24_silero_v5_5_candidate_exposure.py",
    )
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_completed_v5_5_candidate_and_technical_gate_are_bound() -> None:
    script = _audit_script()
    candidate = load_manifest(
        Path("data/manifests/common_voice_ru_v24_silero_v5_5_eugene_pre_qa_pairs_v1.csv")
    )

    script._require_candidate(candidate)
    script._require_pairing_receipt(
        Path("data/manifests/common_voice_ru_v24_silero_v5_5_eugene_pre_qa_pairing_v1.json"),
        Path("data/manifests/common_voice_ru_v24_silero_v5_5_eugene_pre_qa_pairs_v1.csv"),
    )
    script._require_route_audit(
        Path("data/manifests/silero_v5_5_ru_eugene_exact_route_audit_v1.json")
    )
    script._require_acoustic_gate(
        Path("data/manifests/common_voice_ru_v24_silero_v5_5_eugene_pre_qa_acoustic_gate_report_v1.json"),
        candidate,
    )
