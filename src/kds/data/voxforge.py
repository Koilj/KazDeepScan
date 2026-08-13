"""Fail-closed intake audit for the pinned Mozilla Data Collective VoxForge RU release.

The audit reads the compressed TAR stream without extracting raw WAV files.  It
pins the exact archive, rejects unsafe TAR members, verifies the GPLv3 text,
and checks every WAV against the submission's two transcript files.  It is a
source-level receipt only: it creates no project audio, candidate selection,
synthetic derivative, model output, or detector inference.
"""

from __future__ import annotations

import io
import tarfile
import wave
from collections import Counter
from dataclasses import asdict, dataclass
from decimal import Decimal
from pathlib import Path, PurePosixPath

from kds.data.assets import sha256_file

VOXFORGE_RU_SOURCE_ID = "voxforge_ru_mdc_2026_05"
VOXFORGE_RU_ARCHIVE_NAME = "voxforge-russian-9a8495d3.tar.gz"
VOXFORGE_RU_ARCHIVE_ROOT = "voxforge-ru"
VOXFORGE_RU_ARCHIVE_EXPECTED_SIZE_BYTES = 3_795_197_539
VOXFORGE_RU_ARCHIVE_EXPECTED_SHA256 = (
    "7372c6f8d067b8d1651995ad8306b673acaf2cde705ee51295152b96c93de557"
)
VOXFORGE_RU_SOURCE_URL = "https://mozilladatacollective.com/datasets/cmp2h1zvg00n0mp07wrjxow3l"
VOXFORGE_RU_LICENSE = "GPL-3.0-or-later"
_REQUIRED_ETC_FILES = frozenset({"GPL_license.txt", "PROMPTS", "README", "prompts-original"})
_OPTIONAL_ETC_FILES = frozenset({"audiofile_details"})
_GPL_V3_MARKERS = (
    "GNU GENERAL PUBLIC LICENSE",
    "Version 3, 29 June 2007",
)


class VoxForgeRuAuditError(ValueError):
    """Raised when a local VoxForge RU archive cannot safely enter the project."""


@dataclass(frozen=True, slots=True)
class VoxForgeRuArchiveAudit:
    """Source-level inventory derived from one byte-pinned VoxForge RU archive."""

    source_id: str
    archive_name: str
    archive_size_bytes: int
    archive_sha256: str
    archive_root: str
    archive_members: int
    regular_files: int
    directories: int
    submission_license_members: int
    submissions: int
    source_provided_contributor_groups: int
    wav_files: int
    prompt_rows: int
    original_prompt_rows: int
    supplemental_audiofile_detail_files: int
    canonical_prompt_texts: int
    duplicated_prompt_rows: int
    total_duration_seconds: str
    sample_rates_hz: dict[str, int]
    channel_counts: dict[str, int]
    sample_width_bytes: dict[str, int]
    intake_status: str
    extraction_performed: bool
    detector_inference_performed: bool
    candidate_selection_performed: bool

    def receipt(self, *, audited_at: str) -> dict[str, object]:
        payload = asdict(self)
        payload.update(
            {
                "schema_version": 1,
                "audited_at": audited_at,
                "source_url": VOXFORGE_RU_SOURCE_URL,
                "license": VOXFORGE_RU_LICENSE,
                "rights_scope": (
                    "personal_research_only; no product, training, calibration, "
                    "or re-hosting authorization"
                ),
                "group_provenance": (
                    "source_provided contributor/submission identifiers; "
                    "not verified speaker identities"
                ),
                "limitations": [
                    "The audit establishes archive-level provenance and transcript bindings only.",
                    "A contributor identifier is a conservative group key, not a "
                    "verified human speaker identity.",
                    "No raw WAV is extracted or added to Git by this audit.",
                    "A future final candidate still needs project-exposure screening, "
                    "frozen selection, QA, a separate spoof route, and an immutable "
                    "one-run evaluation contract.",
                ],
            }
        )
        return payload


@dataclass(frozen=True, slots=True)
class _WavInfo:
    frames: int
    sample_rate_hz: int
    channels: int
    sample_width_bytes: int


def _validate_archive_identity(archive: Path) -> None:
    if not archive.is_file():
        raise VoxForgeRuAuditError(f"VoxForge RU archive does not exist: {archive}")
    actual_size = archive.stat().st_size
    if actual_size != VOXFORGE_RU_ARCHIVE_EXPECTED_SIZE_BYTES:
        raise VoxForgeRuAuditError(
            "VoxForge RU archive size mismatch: "
            f"expected {VOXFORGE_RU_ARCHIVE_EXPECTED_SIZE_BYTES}, got {actual_size}."
        )
    actual_sha256 = sha256_file(archive)
    if actual_sha256 != VOXFORGE_RU_ARCHIVE_EXPECTED_SHA256:
        raise VoxForgeRuAuditError(
            "VoxForge RU archive SHA-256 mismatch: "
            f"expected {VOXFORGE_RU_ARCHIVE_EXPECTED_SHA256}, got {actual_sha256}."
        )


def _safe_member_parts(member_name: str) -> tuple[str, ...]:
    path = PurePosixPath(member_name)
    if (
        not member_name
        or path.is_absolute()
        or ".." in path.parts
        or "\\" in member_name
        or any(not part for part in path.parts)
    ):
        raise VoxForgeRuAuditError(f"Unsafe TAR member path: {member_name!r}.")
    return path.parts


def _read_member_text(archive: tarfile.TarFile, member: tarfile.TarInfo) -> str:
    file_object = archive.extractfile(member)
    if file_object is None:
        raise VoxForgeRuAuditError(f"Cannot read TAR member: {member.name!r}.")
    try:
        return io.TextIOWrapper(file_object, encoding="utf-8", newline="").read()
    except UnicodeDecodeError as error:
        raise VoxForgeRuAuditError(f"TAR member is not UTF-8 text: {member.name!r}.") from error
    finally:
        file_object.close()


def _read_wav_info(archive: tarfile.TarFile, member: tarfile.TarInfo) -> _WavInfo:
    file_object = archive.extractfile(member)
    if file_object is None:
        raise VoxForgeRuAuditError(f"Cannot read WAV TAR member: {member.name!r}.")
    try:
        with wave.open(file_object, "rb") as audio:
            frames = audio.getnframes()
            sample_rate_hz = audio.getframerate()
            channels = audio.getnchannels()
            sample_width_bytes = audio.getsampwidth()
    except (EOFError, OSError, wave.Error) as error:
        raise VoxForgeRuAuditError(f"Invalid WAV member: {member.name!r}.") from error
    finally:
        file_object.close()
    if frames <= 0 or sample_rate_hz <= 0 or channels <= 0 or sample_width_bytes <= 0:
        raise VoxForgeRuAuditError(f"WAV member has invalid audio metadata: {member.name!r}.")
    return _WavInfo(
        frames=frames,
        sample_rate_hz=sample_rate_hz,
        channels=channels,
        sample_width_bytes=sample_width_bytes,
    )


def _parse_prompts(content: str, submission: str, member_name: str) -> dict[str, str]:
    prompts: dict[str, str] = {}
    for row_number, line in enumerate(content.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            prompt_path, text = line.split(maxsplit=1)
        except ValueError as error:
            raise VoxForgeRuAuditError(
                f"{member_name}:{row_number}: expected '<path> <text>'."
            ) from error
        path = PurePosixPath(prompt_path)
        if (
            len(path.parts) != 3
            or path.parts[0] != submission
            or path.parts[1] != "mfc"
            or not path.name
            or path.suffix
            or ".." in path.parts
            or "\\" in prompt_path
        ):
            raise VoxForgeRuAuditError(
                f"{member_name}:{row_number}: unsafe or unexpected prompt path {prompt_path!r}."
            )
        canonical_text = " ".join(text.split())
        if not canonical_text:
            raise VoxForgeRuAuditError(f"{member_name}:{row_number}: blank prompt text.")
        if path.name in prompts:
            raise VoxForgeRuAuditError(
                f"{member_name}:{row_number}: duplicate prompt id {path.name!r}."
            )
        prompts[path.name] = canonical_text
    if not prompts:
        raise VoxForgeRuAuditError(f"{member_name}: no prompt rows.")
    return prompts


def _parse_original_prompts(content: str, member_name: str) -> dict[str, str]:
    prompts: dict[str, str] = {}
    for row_number, line in enumerate(content.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            prompt_id, text = line.split(maxsplit=1)
        except ValueError as error:
            raise VoxForgeRuAuditError(
                f"{member_name}:{row_number}: expected '<id> <text>'."
            ) from error
        path = PurePosixPath(prompt_id)
        if (
            len(path.parts) != 1
            or not path.name
            or path.suffix
            or ".." in path.parts
            or "\\" in prompt_id
        ):
            raise VoxForgeRuAuditError(
                f"{member_name}:{row_number}: unsafe or unexpected prompt id {prompt_id!r}."
            )
        canonical_text = " ".join(text.split())
        if not canonical_text:
            raise VoxForgeRuAuditError(f"{member_name}:{row_number}: blank original prompt text.")
        if path.name in prompts:
            raise VoxForgeRuAuditError(
                f"{member_name}:{row_number}: duplicate original prompt id {path.name!r}."
            )
        prompts[path.name] = canonical_text
    if not prompts:
        raise VoxForgeRuAuditError(f"{member_name}: no original prompt rows.")
    return prompts


def _parse_contributor(content: str, member_name: str) -> str:
    names = [
        line.partition(":")[2].strip()
        for line in content.splitlines()
        if line.startswith("User Name:")
    ]
    if len(names) != 1 or not names[0]:
        raise VoxForgeRuAuditError(
            f"{member_name}: expected exactly one non-empty 'User Name:' field."
        )
    return names[0].casefold()


def _is_gpl_v3(content: str) -> bool:
    return all(marker in content for marker in _GPL_V3_MARKERS)


def _is_gpl_v3_or_later_notice(content: str) -> bool:
    normalized = content.casefold()
    return "gnu general public license" in normalized and (
        "either version 3 of the license" in normalized or "version 3" in normalized
    )


def _validate_submission(
    submission: str,
    wavs: dict[str, _WavInfo],
    etc_files: set[str],
    gpl_v3_verified: bool,
    submission_license_v3_verified: bool,
    prompts: dict[str, str] | None,
    original_prompts: dict[str, str] | None,
    contributor: str | None,
) -> tuple[str, dict[str, str], dict[str, str]]:
    missing = sorted(_REQUIRED_ETC_FILES.difference(etc_files))
    if missing:
        raise VoxForgeRuAuditError(
            f"Submission {submission!r} is missing required etc files: {', '.join(missing)}."
        )
    if not gpl_v3_verified:
        raise VoxForgeRuAuditError(f"Submission {submission!r} does not contain GNU GPL v3 text.")
    if not submission_license_v3_verified:
        raise VoxForgeRuAuditError(
            f"Submission {submission!r} does not contain a GNU GPL v3 LICENSE file."
        )
    if prompts is None or original_prompts is None or contributor is None:
        raise VoxForgeRuAuditError(f"Submission {submission!r} has incomplete parsed metadata.")
    wav_ids = set(wavs)
    if wav_ids != set(prompts):
        raise VoxForgeRuAuditError(
            f"Submission {submission!r} WAV/PROMPTS mismatch: "
            f"missing_prompts={sorted(wav_ids.difference(prompts))[:5]}, "
            f"orphan_prompts={sorted(set(prompts).difference(wav_ids))[:5]}."
        )
    if wav_ids != set(original_prompts):
        raise VoxForgeRuAuditError(
            f"Submission {submission!r} WAV/prompts-original mismatch: "
            f"missing_original={sorted(wav_ids.difference(original_prompts))[:5]}, "
            f"orphan_original={sorted(set(original_prompts).difference(wav_ids))[:5]}."
        )
    return contributor, prompts, original_prompts


def audit_voxforge_ru_archive(archive_path: Path) -> VoxForgeRuArchiveAudit:
    """Verify the full pinned archive without extracting raw audio to disk."""

    _validate_archive_identity(archive_path)
    member_paths: set[str] = set()
    submission_wavs: dict[str, dict[str, _WavInfo]] = {}
    submission_etc_files: dict[str, set[str]] = {}
    submission_prompts: dict[str, dict[str, str]] = {}
    submission_original_prompts: dict[str, dict[str, str]] = {}
    submission_contributors: dict[str, str] = {}
    submission_gpl_v3_verified: set[str] = set()
    submission_license_v3_verified: set[str] = set()
    submission_license_members = 0
    supplemental_audiofile_detail_files = 0
    regular_files = 0
    directories = 0
    archive_members = 0

    try:
        with tarfile.open(archive_path, mode="r:gz") as archive:
            for member in archive:
                archive_members += 1
                parts = _safe_member_parts(member.name)
                if parts[0] != VOXFORGE_RU_ARCHIVE_ROOT:
                    raise VoxForgeRuAuditError(
                        f"Unexpected archive root in member: {member.name!r}."
                    )
                if not (member.isdir() or member.isfile()):
                    raise VoxForgeRuAuditError(
                        f"Unsafe TAR member type for {member.name!r}: {member.type!r}."
                    )
                if member.isfile():
                    regular_files += 1
                if member.name in member_paths:
                    raise VoxForgeRuAuditError(f"Duplicate TAR member path: {member.name!r}.")
                member_paths.add(member.name)
                if member.isdir():
                    directories += 1
                    continue
                if len(parts) == 3 and parts[2] == "LICENSE":
                    submission = parts[1]
                    if not _is_gpl_v3_or_later_notice(_read_member_text(archive, member)):
                        raise VoxForgeRuAuditError(
                            f"Submission license is not GPL-3.0-or-later: {member.name!r}."
                        )
                    submission_license_v3_verified.add(submission)
                    submission_license_members += 1
                    continue
                if len(parts) != 4 or parts[0] != VOXFORGE_RU_ARCHIVE_ROOT:
                    raise VoxForgeRuAuditError(
                        f"Unexpected regular TAR member path: {member.name!r}."
                    )
                submission, directory, filename = parts[1:]
                if directory == "wav":
                    path = PurePosixPath(filename)
                    if path.suffix.lower() != ".wav" or not path.stem:
                        raise VoxForgeRuAuditError(f"Unexpected audio member: {member.name!r}.")
                    wavs = submission_wavs.setdefault(submission, {})
                    if path.stem in wavs:
                        raise VoxForgeRuAuditError(
                            f"Duplicate WAV id in submission {submission!r}: {path.stem!r}."
                        )
                    wavs[path.stem] = _read_wav_info(archive, member)
                    continue
                if directory == "etc" and filename in _REQUIRED_ETC_FILES.union(
                    _OPTIONAL_ETC_FILES
                ):
                    etc_files = submission_etc_files.setdefault(submission, set())
                    if filename in etc_files:
                        raise VoxForgeRuAuditError(
                            f"Duplicate {filename!r} in submission {submission!r}."
                        )
                    etc_files.add(filename)
                    content = _read_member_text(archive, member)
                    if filename == "GPL_license.txt":
                        if not _is_gpl_v3(content):
                            raise VoxForgeRuAuditError(
                                f"Submission {submission!r} does not contain GNU GPL v3 text."
                            )
                        submission_gpl_v3_verified.add(submission)
                    elif filename == "PROMPTS":
                        submission_prompts[submission] = _parse_prompts(
                            content, submission, member.name
                        )
                    elif filename == "prompts-original":
                        submission_original_prompts[submission] = _parse_original_prompts(
                            content, member.name
                        )
                    elif filename == "README":
                        submission_contributors[submission] = _parse_contributor(
                            content, member.name
                        )
                    elif not content.strip():
                        raise VoxForgeRuAuditError(
                            f"Supplemental metadata is blank: {member.name!r}."
                        )
                    else:
                        supplemental_audiofile_detail_files += 1
                    continue
                raise VoxForgeRuAuditError(f"Unexpected regular TAR member path: {member.name!r}.")
            submissions = set(submission_wavs).union(submission_etc_files)
            if not submissions:
                raise VoxForgeRuAuditError("Archive contains no VoxForge submissions.")
            contributors: set[str] = set()
            prompt_texts: list[str] = []
            original_prompt_rows = 0
            duration_seconds = Decimal(0)
            sample_rates: Counter[int] = Counter()
            channel_counts: Counter[int] = Counter()
            sample_widths: Counter[int] = Counter()
            for submission in sorted(submissions):
                submission_wav_entries = submission_wavs.get(submission)
                submission_etc_entries = submission_etc_files.get(submission)
                if submission_wav_entries is None or submission_etc_entries is None:
                    raise VoxForgeRuAuditError(
                        f"Submission {submission!r} lacks either WAV or etc members."
                    )
                contributor, prompts, original_prompts = _validate_submission(
                    submission,
                    submission_wav_entries,
                    submission_etc_entries,
                    submission in submission_gpl_v3_verified,
                    submission in submission_license_v3_verified,
                    submission_prompts.get(submission),
                    submission_original_prompts.get(submission),
                    submission_contributors.get(submission),
                )
                contributors.add(contributor)
                prompt_texts.extend(prompts.values())
                original_prompt_rows += len(original_prompts)
                for audio in submission_wav_entries.values():
                    duration_seconds += Decimal(audio.frames) / Decimal(audio.sample_rate_hz)
                    sample_rates[audio.sample_rate_hz] += 1
                    channel_counts[audio.channels] += 1
                    sample_widths[audio.sample_width_bytes] += 1
    except (OSError, tarfile.TarError) as error:
        raise VoxForgeRuAuditError(
            f"Cannot read VoxForge RU TAR archive: {archive_path}."
        ) from error

    canonical_prompt_texts = len(set(prompt_texts))
    return VoxForgeRuArchiveAudit(
        source_id=VOXFORGE_RU_SOURCE_ID,
        archive_name=VOXFORGE_RU_ARCHIVE_NAME,
        archive_size_bytes=VOXFORGE_RU_ARCHIVE_EXPECTED_SIZE_BYTES,
        archive_sha256=VOXFORGE_RU_ARCHIVE_EXPECTED_SHA256,
        archive_root=VOXFORGE_RU_ARCHIVE_ROOT,
        archive_members=archive_members,
        regular_files=regular_files,
        directories=directories,
        submission_license_members=submission_license_members,
        submissions=len(submissions),
        source_provided_contributor_groups=len(contributors),
        wav_files=len(prompt_texts),
        prompt_rows=len(prompt_texts),
        original_prompt_rows=original_prompt_rows,
        supplemental_audiofile_detail_files=supplemental_audiofile_detail_files,
        canonical_prompt_texts=canonical_prompt_texts,
        duplicated_prompt_rows=len(prompt_texts) - canonical_prompt_texts,
        total_duration_seconds=f"{duration_seconds:.6f}",
        sample_rates_hz={str(key): sample_rates[key] for key in sorted(sample_rates)},
        channel_counts={str(key): channel_counts[key] for key in sorted(channel_counts)},
        sample_width_bytes={str(key): sample_widths[key] for key in sorted(sample_widths)},
        intake_status="accepted_source_level_only",
        extraction_performed=False,
        detector_inference_performed=False,
        candidate_selection_performed=False,
    )
