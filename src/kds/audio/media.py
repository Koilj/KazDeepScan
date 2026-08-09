from __future__ import annotations

import json
import math
import os
import subprocess
from pathlib import Path
from typing import Final

from kds.audio.contracts import AudioErrorCode, AudioLimits, AudioPipelineError, MediaInfo

SUPPORTED_MIME_TYPES: Final[frozenset[str]] = frozenset(
    {
        "audio/flac",
        "audio/m4a",
        "audio/mp3",
        "audio/mp4",
        "audio/mpeg",
        "audio/ogg",
        "audio/vorbis",
        "audio/wav",
        "audio/wave",
        "audio/x-flac",
        "audio/x-m4a",
        "audio/x-wav",
    }
)

SUPPORTED_CONTAINERS: Final[frozenset[str]] = frozenset(
    {"flac", "m4a", "mov", "mp3", "mp4", "ogg", "wav"}
)


def validate_declared_mime(mime_type: str | None) -> None:
    """Validate a declared MIME type without treating it as proof of file type."""

    if mime_type is None:
        return
    normalized = mime_type.split(";", maxsplit=1)[0].strip().lower()
    if normalized not in SUPPORTED_MIME_TYPES:
        raise AudioPipelineError(
            AudioErrorCode.UNSUPPORTED_MIME,
            f"Unsupported declared MIME type: {normalized or '<empty>'}.",
        )


class FFmpegClient:
    """Thin, shell-free wrapper around ffprobe and ffmpeg."""

    def __init__(
        self, ffprobe_binary: str | None = None, ffmpeg_binary: str | None = None
    ) -> None:
        self._ffprobe_binary = ffprobe_binary or os.environ.get("KDS_FFPROBE_BINARY", "ffprobe")
        self._ffmpeg_binary = ffmpeg_binary or os.environ.get("KDS_FFMPEG_BINARY", "ffmpeg")

    @staticmethod
    def _ensure_regular_file(path: Path) -> None:
        if not path.is_file():
            raise AudioPipelineError(AudioErrorCode.FILE_NOT_FOUND, f"Audio file not found: {path}")

    def validate_file_size(self, path: Path, limits: AudioLimits) -> None:
        self._ensure_regular_file(path)
        size = path.stat().st_size
        if size > limits.max_upload_bytes:
            raise AudioPipelineError(
                AudioErrorCode.FILE_TOO_LARGE,
                f"File is {size} bytes; maximum is {limits.max_upload_bytes} bytes.",
            )

    def probe(self, path: Path, limits: AudioLimits) -> MediaInfo:
        self._ensure_regular_file(path)
        command = [
            self._ffprobe_binary,
            "-v",
            "error",
            "-show_entries",
            "format=format_name,duration",
            "-show_streams",
            "-of",
            "json",
            str(path),
        ]
        try:
            result = subprocess.run(
                command, check=False, capture_output=True, text=True, timeout=30
            )
        except FileNotFoundError as error:
            raise AudioPipelineError(
                AudioErrorCode.DEPENDENCY_MISSING,
                "ffprobe is not installed or is absent from PATH.",
            ) from error
        except subprocess.TimeoutExpired as error:
            raise AudioPipelineError(
                AudioErrorCode.DECODE_FAILED,
                "ffprobe exceeded the 30-second inspection limit.",
            ) from error

        if result.returncode != 0:
            detail = result.stderr.strip() or "ffprobe could not read this file."
            raise AudioPipelineError(AudioErrorCode.DECODE_FAILED, detail)

        try:
            payload = json.loads(result.stdout)
            format_payload = payload["format"]
            raw_duration = format_payload["duration"]
            raw_names = format_payload["format_name"]
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            raise AudioPipelineError(
                AudioErrorCode.DECODE_FAILED,
                "ffprobe returned malformed media metadata.",
            ) from error

        try:
            duration_seconds = float(raw_duration)
        except (TypeError, ValueError) as error:
            raise AudioPipelineError(
                AudioErrorCode.DECODE_FAILED,
                "Audio duration is missing or invalid.",
            ) from error
        if not math.isfinite(duration_seconds) or duration_seconds <= 0:
            raise AudioPipelineError(
                AudioErrorCode.DECODE_FAILED,
                "Audio duration must be a positive finite number.",
            )
        if duration_seconds > limits.max_duration_seconds:
            raise AudioPipelineError(
                AudioErrorCode.DURATION_LIMIT_EXCEEDED,
                (
                    f"Audio lasts {duration_seconds:.2f}s; "
                    f"maximum is {limits.max_duration_seconds:.2f}s."
                ),
            )

        container_names = tuple(
            name.strip().lower() for name in str(raw_names).split(",") if name.strip()
        )
        if not set(container_names).intersection(SUPPORTED_CONTAINERS):
            raise AudioPipelineError(
                AudioErrorCode.UNSUPPORTED_CONTAINER,
                f"Unsupported audio container: {', '.join(container_names) or '<unknown>'}.",
            )

        streams = payload.get("streams", [])
        audio_stream_count = sum(
            1
            for stream in streams
            if isinstance(stream, dict) and stream.get("codec_type") == "audio"
        )
        if audio_stream_count == 0:
            raise AudioPipelineError(AudioErrorCode.DECODE_FAILED, "The file has no audio stream.")

        return MediaInfo(
            path=path,
            container_names=container_names,
            duration_seconds=duration_seconds,
            audio_stream_count=audio_stream_count,
        )

    def normalize_to_wav(self, source: Path, destination: Path, sample_rate: int) -> None:
        """Create a mono PCM S16LE WAV without allowing an existing destination to be replaced."""

        self._ensure_regular_file(source)
        if destination.exists():
            raise AudioPipelineError(
                AudioErrorCode.INVALID_INPUT,
                f"Refusing to overwrite normalized output: {destination}",
            )
        if not destination.parent.is_dir():
            raise AudioPipelineError(
                AudioErrorCode.INVALID_INPUT,
                f"Normalized output directory does not exist: {destination.parent}",
            )

        command = [
            self._ffmpeg_binary,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(source),
            "-map",
            "0:a:0",
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-c:a",
            "pcm_s16le",
            "-f",
            "wav",
            "-n",
            str(destination),
        ]
        try:
            result = subprocess.run(
                command, check=False, capture_output=True, text=True, timeout=90
            )
        except FileNotFoundError as error:
            raise AudioPipelineError(
                AudioErrorCode.DEPENDENCY_MISSING,
                "ffmpeg is not installed or is absent from PATH.",
            ) from error
        except subprocess.TimeoutExpired as error:
            raise AudioPipelineError(
                AudioErrorCode.DECODE_FAILED,
                "ffmpeg exceeded the 90-second decode limit.",
            ) from error

        if result.returncode != 0 or not destination.is_file():
            detail = result.stderr.strip() or "ffmpeg could not normalize this file."
            raise AudioPipelineError(AudioErrorCode.DECODE_FAILED, detail)
