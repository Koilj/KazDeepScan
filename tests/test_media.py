from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from kds.audio.contracts import AudioLimits
from kds.audio.media import FFmpegClient, validate_declared_mime


class MediaTests(unittest.TestCase):
    def test_binary_paths_can_be_configured_without_modifying_path(self) -> None:
        with patch.dict(
            os.environ,
            {
                "KDS_FFMPEG_BINARY": "/opt/kds/ffmpeg",
                "KDS_FFPROBE_BINARY": "/opt/kds/ffprobe",
            },
            clear=False,
        ):
            client = FFmpegClient()

        self.assertEqual(client._ffmpeg_binary, "/opt/kds/ffmpeg")
        self.assertEqual(client._ffprobe_binary, "/opt/kds/ffprobe")

    def test_probe_uses_ffprobe_metadata_and_accepts_m4a_container(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".m4a") as file_handle:
            payload = {
                "format": {"format_name": "mov,mp4,m4a,3gp,3g2,mj2", "duration": "12.5"},
                "streams": [{"codec_type": "audio"}],
            }
            completed = subprocess.CompletedProcess([], 0, json.dumps(payload), "")
            with patch("kds.audio.media.subprocess.run", return_value=completed) as run:
                info = FFmpegClient().probe(Path(file_handle.name), AudioLimits())

        self.assertEqual(info.duration_seconds, 12.5)
        self.assertIn("m4a", info.container_names)
        command = run.call_args.args[0]
        self.assertIsInstance(command, list)
        self.assertIn("-of", command)
        self.assertIn("json", command)

    def test_mime_is_only_allowed_from_known_audio_set(self) -> None:
        validate_declared_mime("audio/wav; charset=binary")
        with self.assertRaisesRegex(Exception, "Unsupported declared MIME"):
            validate_declared_mime("application/zip")


if __name__ == "__main__":
    unittest.main()
