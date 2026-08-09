from __future__ import annotations

import argparse
import json
from pathlib import Path

from kds.data.research_tts import (
    DEFAULT_DOWNLOAD_LIMIT_BYTES,
    ResearchTtsError,
    download_research_tts_model_lock,
    load_research_tts_model_lock,
    verify_research_tts_model_lock,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download one pinned personal-research TTS model lock with SHA-256 checks."
    )
    parser.add_argument("--model-lock", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--max-download-bytes", type=int, default=DEFAULT_DOWNLOAD_LIMIT_BYTES)
    arguments = parser.parse_args()

    try:
        lock = load_research_tts_model_lock(arguments.model_lock)
        download_research_tts_model_lock(
            arguments.model_root, lock, max_download_bytes=arguments.max_download_bytes
        )
        verified = verify_research_tts_model_lock(arguments.model_root, lock)
    except ResearchTtsError as error:
        print(json.dumps({"status": "error", "issues": [str(error)]}, ensure_ascii=False))
        return 2

    print(
        json.dumps(
            {
                "status": "ok",
                "protocol_id": lock.protocol_id,
                "model_root": str(arguments.model_root),
                "models": {model_id: sorted(paths) for model_id, paths in verified.items()},
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
