"""Create the v4 reconciliation pair lock from two hash-bound review forms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kds.data.assets import sha256_file
from kds.data.v4_final_pair_lock_reconciliation import V4FinalPairLockError, finalize_pair_lock


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--created-at", required=True)
    args = parser.parse_args()
    try:
        plan = finalize_pair_lock(
            authorization_path=args.authorization,
            project_root=args.project_root,
            data_root=args.data_root,
            created_at=args.created_at,
        )
    except (OSError, RuntimeError, ValueError, V4FinalPairLockError) as error:
        print(json.dumps({"status": "error", "issues": [str(error)]}, ensure_ascii=False))
        return 2
    receipt = Path(args.project_root) / plan.outputs["pair_lock_receipt"]
    print(
        json.dumps(
            {
                "status": "ok",
                "receipt": plan.outputs["pair_lock_receipt"],
                "receipt_sha256": sha256_file(receipt),
                "final_inference_performed": False,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
