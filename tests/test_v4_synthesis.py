from pathlib import Path

from kds.data.v4_synthesis import V4_KK_ROUTE_FAMILIES, load_v4_kk_spoof_candidates


def test_repository_v4_kk_spoof_packet_is_frozen_and_authorized() -> None:
    rows = load_v4_kk_spoof_candidates(
        Path("data/manifests/v4/xlsr_sls_model_v4_train_candidates_v2.csv"),
        Path(
            "docs/artifacts/v4/"
            "xlsr_sls_model_v4_train_candidate_selection_governance_v1.json"
        ),
        Path("docs/artifacts/v4/xlsr_sls_model_v4_source_decode_qa_v1.json"),
    )

    assert len(rows) == 7_200
    assert {row.generator_route_id for row in rows} == set(V4_KK_ROUTE_FAMILIES)
    assert sum(row.target_state == "target" for row in rows) == 6_000
