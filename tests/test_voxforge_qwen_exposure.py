from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any

from kds.data.manifest import load_manifest


def _audit_script() -> Any:
    spec = spec_from_file_location(
        "kds_voxforge_qwen_exposure",
        "scripts/audit_voxforge_ru_mdc_qwen3_tts_customvoice_candidate_exposure.py",
    )
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_completed_voxforge_qwen_candidate_and_gates_are_bound() -> None:
    script = _audit_script()
    candidate_path = Path(
        "data/manifests/"
        "voxforge_ru_mdc_qwen3_tts_customvoice_aiden_pre_qa_pairs_v1.csv"
    )
    candidate = load_manifest(candidate_path)

    script._require_candidate(candidate)
    script._require_pairing_receipt(
        Path(
            "data/manifests/"
            "voxforge_ru_mdc_qwen3_tts_customvoice_aiden_pre_qa_pairing_v1.json"
        ),
        candidate_path,
    )
    script._require_route_audit(
        Path(
            "data/manifests/"
            "voxforge_ru_mdc_qwen3_tts_customvoice_aiden_exact_route_audit_v1.json"
        )
    )
    script._require_acoustic_gate(
        Path(
            "data/manifests/"
            "voxforge_ru_mdc_qwen3_tts_customvoice_aiden_pre_qa_acoustic_gate_report_v1.json"
        ),
        candidate,
    )
