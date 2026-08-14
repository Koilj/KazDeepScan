"""Publish the write-once local-capacity and project-history Gate A receipt for v4."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kds.data.assets import sha256_file
from kds.data.licenses import LicenseLedgerError
from kds.data.research_tts import ResearchTtsError
from kds.data.v4_capacity import (
    V4CapacityError,
    audit_v4_local_capacity,
    write_v4_capacity_receipt,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--license-ledger", type=Path, required=True)
    parser.add_argument("--ruasd-audit", type=Path, required=True)
    parser.add_argument("--ksc2-audit", type=Path, required=True)
    parser.add_argument("--common-voice-screen", type=Path, required=True)
    parser.add_argument("--audited-at", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        receipt = audit_v4_local_capacity(
            config_path=arguments.config,
            project_root=arguments.project_root,
            license_ledger_path=arguments.license_ledger,
            ruasd_audit_path=arguments.ruasd_audit,
            ksc2_audit_path=arguments.ksc2_audit,
            common_voice_screen_path=arguments.common_voice_screen,
            audited_at=arguments.audited_at,
        )
        write_v4_capacity_receipt(arguments.output, receipt)
    except (V4CapacityError, LicenseLedgerError, ResearchTtsError, OSError) as error:
        print(json.dumps({"status": "error", "issues": [str(error)]}, ensure_ascii=False))
        return 2
    print(
        json.dumps(
            {
                "status": "ok",
                "decision": receipt["decision"],
                "output": str(arguments.output),
                "output_sha256": sha256_file(arguments.output),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
