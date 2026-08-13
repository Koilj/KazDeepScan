"""Fail-closed artifact and source audit for official OpenBMB VoxCPM2.

This module intentionally does not import or instantiate VoxCPM.  It verifies the exact local
Hugging Face snapshot, the exact official source archive, both checkpoint containers, and the
small amount of upstream code that defines the permitted local text-only route.
"""

from __future__ import annotations

import ast
import gzip
import hashlib
import json
import struct
import tarfile
import tomllib
import zipfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from kds.data.assets import sha256_file

VOXCPM2_MODEL_ID = "openbmb/VoxCPM2"
VOXCPM2_MODEL_REVISION = "bffb3df5a29440629464e5e839f4d214c8714c3d"
VOXCPM2_SOURCE_REVISION = "ee8161e9e1b7b082cb5721a3a9980da4204401e6"
VOXCPM2_SOURCE_ARCHIVE_SIZE_BYTES = 4_107_908
VOXCPM2_SOURCE_ARCHIVE_SHA256 = (
    "5af8b4def8dc200c3a0b660b63d4a08a1cf5cadcb5c5371c0b89d5f0f58c0674"
)
VOXCPM2_SOURCE_ROOT = f"VoxCPM-{VOXCPM2_SOURCE_REVISION}"
VOXCPM2_MODEL_FILES: dict[str, tuple[int, str]] = {
    ".gitattributes": (
        1_519,
        "11ad7efa24975ee4b0c3c3a38ed18737f0658a5f75a0a96787b576a78a023361",
    ),
    "README.md": (
        7_776,
        "7384fad93ce2d98f47d5c3170597f3b31d414c12c92e7fdf3121fa90f19fe29d",
    ),
    "audiovae.pth": (
        376_951_122,
        "94b5d51e107e0507d4acc976cfdadb64edd6fd06d1f751dadbf2fd1594274bf1",
    ),
    "config.json": (
        4_336,
        "405f0dcd92f7feba6011ed4eac5c8d4f74cba9712f07fd5cfa3063bbdd95402c",
    ),
    "model.safetensors": (
        4_580_080_592,
        "f7f964cfa9da23653baec6e6f7750719977ad944ed9f95fe52fe3a620506891d",
    ),
    "special_tokens_map.json": (
        1_632,
        "068594063e37662c02b21acf42ebb334ef6a74fb810e68a2368f88f08351de76",
    ),
    "tokenization_voxcpm2.py": (
        2_895,
        "84489ea32b6ee0cae22ed5480cacb6df85c46624c3119be9a2021c3649a12729",
    ),
    "tokenizer.json": (
        3_676_772,
        "f8984687e4a92a3503d521396d454b7d68e9fdaab2a0288eb3536c7c1aa4bc20",
    ),
    "tokenizer_config.json": (
        5_059,
        "e78a3ebb48a0b9437efd1823b6b726c823da89e49dd8bcc90c02419d9baa772b",
    ),
}
VOXCPM2_SOURCE_FILES: dict[str, tuple[int, str]] = {
    "LICENSE": (
        11_298,
        "4f10acc209addacfad28293315c74c4cd648f771ee1263a748f1781d1e0265e4",
    ),
    "pyproject.toml": (
        2_142,
        "6d4b8d4f6e4d96525611718ba686bc4025db9b027a53abeeb0bc4721cb97bf78",
    ),
    "src/voxcpm/core.py": (
        16_078,
        "5116a96ed19e2e2b86a9bffabfb322e276dd33dc0bf4d0b189a1e4fd5d2cc109",
    ),
    "src/voxcpm/model/voxcpm2.py": (
        54_715,
        "31ad554bc15a4da18cf820c6fda97a5b2c56ac8e9a5e66031a264d363661575c",
    ),
    "tests/test_torch_load_safety.py": (
        3_122,
        "84793db4a25234894e97a032ebaa2319ac57d844b05a6e5883d33f08c3ee3602",
    ),
}


class VoxCPM2AuditError(ValueError):
    """Raised when the exact VoxCPM2 route fails closed."""


@dataclass(frozen=True, slots=True)
class VoxCPM2ArtifactAudit:
    model_revision: str
    source_revision: str
    model_files: int
    model_bytes: int
    model_inventory_sha256: str
    architecture: str
    output_sample_rate_hz: int
    safetensors_header_bytes: int
    safetensors_tensors: int
    safetensors_dtype_counts: dict[str, int]
    safetensors_payload_bytes: int
    safetensors_offsets_contiguous: bool
    audiovae_zip_members: int
    audiovae_zip_crc_verified: bool
    audiovae_pickle_globals: tuple[str, ...]
    audiovae_weights_only_loaded: bool
    audiovae_state_tensors: int
    audiovae_state_dtype_counts: dict[str, int]
    audiovae_state_elements: int
    audiovae_sample_rate_hz: int
    tokenizer_python_imports: tuple[str, ...]
    tokenizer_python_forbidden_calls: tuple[str, ...]
    source_archive_members: int
    source_archive_files: int
    source_archive_directories: int
    source_archive_regular_bytes: int
    source_gzip_crc_verified: bool
    source_torch_load_calls: int
    source_torch_load_all_weights_only: bool
    source_license: str
    source_requires_python: str
    local_path_loader_verified: bool
    denoiser_can_be_disabled: bool
    semantic_normalizer_can_be_disabled: bool
    retry_can_be_disabled: bool
    upstream_whitespace_collapse_unconditional: bool
    tts_model_loaded: bool
    synthesis_performed: bool
    detector_inference_performed: bool

    def receipt(self, *, audited_at: str) -> dict[str, object]:
        payload = asdict(self)
        payload.update(
            {
                "schema_version": 1,
                "protocol_id": "voxcpm2-official-text-only-artifact-source-gate-v1",
                "audited_at": audited_at,
                "model_id": VOXCPM2_MODEL_ID,
                "model_url": (
                    f"https://huggingface.co/{VOXCPM2_MODEL_ID}/tree/"
                    f"{VOXCPM2_MODEL_REVISION}"
                ),
                "source_url": (
                    "https://github.com/OpenBMB/VoxCPM/tree/"
                    f"{VOXCPM2_SOURCE_REVISION}"
                ),
                "model_files_expected": {
                    name: {"size_bytes": size, "sha256": digest}
                    for name, (size, digest) in VOXCPM2_MODEL_FILES.items()
                },
                "source_archive": {
                    "name": "source.tar.gz",
                    "size_bytes": VOXCPM2_SOURCE_ARCHIVE_SIZE_BYTES,
                    "sha256": VOXCPM2_SOURCE_ARCHIVE_SHA256,
                    "verified_inner_files": {
                        name: {"size_bytes": size, "sha256": digest}
                        for name, (size, digest) in VOXCPM2_SOURCE_FILES.items()
                    },
                },
                "checkpoint_policy": {
                    "model_safetensors": "header and complete contiguous tensor payload audited",
                    "audiovae_pth": (
                        "ZIP CRC and pickle GLOBAL allow-list audited; loaded only with "
                        "torch.load(weights_only=True, map_location='cpu', mmap=True)"
                    ),
                },
                "admitted_runtime_contract": {
                    "status": "specified_not_executed",
                    "python": (
                        "isolated CPython 3.12 preferred; upstream exact source requires >=3.10"
                    ),
                    "network": "outer network namespace/firewall plus offline environment required",
                    "model_path": "exact local snapshot only",
                    "reference_wav_path": None,
                    "prompt_wav_path": None,
                    "prompt_text": None,
                    "lora_weights_path": None,
                    "load_denoiser": False,
                    "normalize": False,
                    "denoise": False,
                    "retry_badcase": False,
                    "streaming": False,
                    "seed": 20260814,
                    "cfg_value": 2.0,
                    "inference_timesteps": 10,
                    "min_len": 2,
                    "max_len": 4096,
                    "text_policy": (
                        "bind literal SHA-256 and predeclared collapse-whitespace SHA-256; "
                        "pass only the latter; no semantic rewrite or external normalizer"
                    ),
                },
                "claims": {
                    "artifact_and_source_gate_complete": True,
                    "runtime_environment_materialized": False,
                    "cuda_load_verified": False,
                    "text_only_smoke_verified": False,
                    "new_generator_family_claim": "requires separate project-history receipt",
                    "training_data_overlap": "unverified",
                    "default_voice_identity": "unknown_not_claimed",
                },
                "limitations": [
                    "The model card discloses aggregate training scale but not a complete "
                    "training-source list.",
                    "Apache-2.0 covers the published model/code route; it does not prove "
                    "training-data disjointness.",
                    "The upstream public API supports cloning, but the admitted project "
                    "wrapper exposes text only.",
                    "Upstream always collapses whitespace even with normalize=False; the "
                    "project makes that transform explicit and hash-bound.",
                    "No model load, synthesis, candidate selection, detector inference, or "
                    "post-result tuning occurred in this gate.",
                ],
            }
        )
        return payload


def _digest_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _inventory_digest(inventory: dict[str, tuple[int, str]]) -> str:
    material = "".join(
        f"{name}\0{size}\0{digest}\n"
        for name, (size, digest) in sorted(inventory.items())
    ).encode()
    return _digest_bytes(material)


def _verify_model_files(model_root: Path) -> None:
    if not model_root.is_dir():
        raise VoxCPM2AuditError(f"Model snapshot is not a directory: {model_root}")
    actual = {path.name for path in model_root.iterdir() if path.name != ".cache"}
    if actual != set(VOXCPM2_MODEL_FILES):
        raise VoxCPM2AuditError(
            f"Model snapshot top-level inventory mismatch: expected {sorted(VOXCPM2_MODEL_FILES)}, "
            f"got {sorted(actual)}."
        )
    for name, (expected_size, expected_hash) in VOXCPM2_MODEL_FILES.items():
        path = model_root / name
        if not path.is_file() or path.is_symlink():
            raise VoxCPM2AuditError(f"Model member must be a regular non-symlink file: {name}")
        if path.stat().st_size != expected_size or sha256_file(path) != expected_hash:
            raise VoxCPM2AuditError(f"Model member identity mismatch: {name}")


def _audit_safetensors(path: Path) -> tuple[int, int, dict[str, int], int]:
    with path.open("rb") as handle:
        header_length_bytes = handle.read(8)
        if len(header_length_bytes) != 8:
            raise VoxCPM2AuditError("Safetensors header length is truncated.")
        header_length = struct.unpack("<Q", header_length_bytes)[0]
        try:
            header = json.loads(handle.read(header_length))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise VoxCPM2AuditError("Safetensors header is invalid JSON.") from error
    tensors = [(name, value) for name, value in header.items() if name != "__metadata__"]
    spans: list[tuple[int, int]] = []
    dtypes: Counter[str] = Counter()
    for name, metadata in tensors:
        if not isinstance(metadata, dict):
            raise VoxCPM2AuditError(f"Safetensors metadata is invalid for {name!r}.")
        offsets = metadata.get("data_offsets")
        dtype = metadata.get("dtype")
        shape = metadata.get("shape")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(isinstance(value, int) for value in offsets)
            or not isinstance(dtype, str)
            or not isinstance(shape, list)
            or not all(isinstance(value, int) and value >= 0 for value in shape)
        ):
            raise VoxCPM2AuditError(f"Safetensors tensor metadata is malformed: {name!r}.")
        spans.append((offsets[0], offsets[1]))
        dtypes[dtype] += 1
    spans.sort()
    cursor = 0
    for start, end in spans:
        if start != cursor or end < start:
            raise VoxCPM2AuditError("Safetensors payload has gaps, overlap, or reversed offsets.")
        cursor = end
    expected_payload = path.stat().st_size - 8 - header_length
    if cursor != expected_payload:
        raise VoxCPM2AuditError(
            f"Safetensors payload mismatch: header spans {cursor}, file has {expected_payload}."
        )
    return header_length, len(tensors), dict(sorted(dtypes.items())), cursor


def _pickle_globals(pickle_payload: bytes) -> tuple[str, ...]:
    import pickletools

    globals_found: set[str] = set()
    stack: list[str] = []
    for opcode, argument, _ in pickletools.genops(pickle_payload):
        if opcode.name in {"SHORT_BINUNICODE", "BINUNICODE", "UNICODE"}:
            stack.append(str(argument))
        elif opcode.name == "GLOBAL":
            module, name = str(argument).split(" ", 1)
            globals_found.add(f"{module}.{name}")
        elif opcode.name == "STACK_GLOBAL":
            if len(stack) < 2:
                raise VoxCPM2AuditError("Cannot resolve STACK_GLOBAL in AudioVAE pickle.")
            globals_found.add(f"{stack[-2]}.{stack[-1]}")
    return tuple(sorted(globals_found))


def _audit_audiovae(path: Path) -> tuple[int, tuple[str, ...]]:
    try:
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            if any(
                PurePosixPath(member.filename).is_absolute()
                or ".." in PurePosixPath(member.filename).parts
                for member in members
            ):
                raise VoxCPM2AuditError("AudioVAE ZIP contains an unsafe path.")
            bad_member = archive.testzip()
            if bad_member is not None:
                raise VoxCPM2AuditError(f"AudioVAE ZIP CRC failed at {bad_member!r}.")
            pickle_names = [
                member.filename for member in members if member.filename.endswith("/data.pkl")
            ]
            if pickle_names != ["vae/data.pkl"]:
                raise VoxCPM2AuditError(f"Unexpected AudioVAE pickle inventory: {pickle_names}")
            globals_found = _pickle_globals(archive.read(pickle_names[0]))
    except zipfile.BadZipFile as error:
        raise VoxCPM2AuditError("AudioVAE checkpoint is not a valid ZIP container.") from error
    expected_globals = (
        "collections.OrderedDict",
        "torch.FloatStorage",
        "torch.IntStorage",
        "torch._utils._rebuild_tensor_v2",
    )
    if globals_found != expected_globals:
        raise VoxCPM2AuditError(f"AudioVAE pickle GLOBAL allow-list mismatch: {globals_found}")
    return len(members), globals_found


def _weights_only_audiovae(path: Path) -> tuple[int, dict[str, int], int, int]:
    try:
        import torch
    except ImportError as error:
        raise VoxCPM2AuditError(
            "PyTorch is required for the weights-only AudioVAE audit."
        ) from error
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except Exception as error:
        raise VoxCPM2AuditError("AudioVAE failed torch.load(weights_only=True).") from error
    if not isinstance(checkpoint, dict) or set(checkpoint) != {"metadata", "state_dict"}:
        raise VoxCPM2AuditError("AudioVAE weights-only root shape is unexpected.")
    metadata = checkpoint["metadata"]
    if metadata != {"kwargs": {"sample_rate": 16_000}}:
        raise VoxCPM2AuditError(f"AudioVAE metadata mismatch: {metadata!r}")
    state = checkpoint["state_dict"]
    if not isinstance(state, dict) or not state:
        raise VoxCPM2AuditError("AudioVAE state_dict is missing or empty.")
    dtypes: Counter[str] = Counter()
    elements = 0
    for name, tensor in state.items():
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise VoxCPM2AuditError("AudioVAE state_dict is not string-to-tensor only.")
        if tensor.device.type != "cpu":
            raise VoxCPM2AuditError("AudioVAE weights-only audit loaded a non-CPU tensor.")
        dtypes[str(tensor.dtype).removeprefix("torch.")] += 1
        elements += tensor.numel()
    return len(state), dict(sorted(dtypes.items())), elements, 16_000


def _safe_tar_parts(name: str) -> tuple[str, ...]:
    path = PurePosixPath(name)
    if not name or path.is_absolute() or ".." in path.parts or "\\" in name:
        raise VoxCPM2AuditError(f"Unsafe source TAR member path: {name!r}")
    return path.parts


def _read_tar_member(archive: tarfile.TarFile, member: tarfile.TarInfo) -> bytes:
    handle = archive.extractfile(member)
    if handle is None:
        raise VoxCPM2AuditError(f"Cannot read source member: {member.name!r}")
    payload = handle.read()
    if len(payload) != member.size:
        raise VoxCPM2AuditError(f"Short source member read: {member.name!r}")
    return payload


def _torch_load_policy(python_files: dict[str, bytes]) -> tuple[int, bool]:
    calls = 0
    all_weights_only = True
    for name, payload in python_files.items():
        try:
            tree = ast.parse(payload.decode("utf-8"), filename=name)
        except (UnicodeDecodeError, SyntaxError) as error:
            raise VoxCPM2AuditError(f"Cannot parse pinned Python source {name!r}.") from error
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if (
                node.func.attr == "load"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "torch"
            ):
                calls += 1
                weights_only = next(
                    (keyword.value for keyword in node.keywords if keyword.arg == "weights_only"),
                    None,
                )
                all_weights_only &= (
                    isinstance(weights_only, ast.Constant) and weights_only.value is True
                )
    return calls, all_weights_only


def _tokenizer_python_policy(payload: bytes) -> tuple[tuple[str, ...], tuple[str, ...]]:
    tree = ast.parse(payload.decode("utf-8"), filename="tokenization_voxcpm2.py")
    imports: set[str] = set()
    forbidden: set[str] = set()
    forbidden_names = {"eval", "exec", "compile", "open", "__import__"}
    forbidden_roots = {"os", "pathlib", "socket", "subprocess", "urllib", "requests", "httpx"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.add(node.module or "")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in forbidden_names:
                forbidden.add(node.func.id)
            if (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in forbidden_roots
            ):
                forbidden.add(f"{node.func.value.id}.{node.func.attr}")
    return tuple(sorted(imports)), tuple(sorted(forbidden))


def _audit_source_archive(source_archive: Path) -> dict[str, Any]:
    if not source_archive.is_file():
        raise VoxCPM2AuditError(f"Source archive does not exist: {source_archive}")
    if (
        source_archive.stat().st_size != VOXCPM2_SOURCE_ARCHIVE_SIZE_BYTES
        or sha256_file(source_archive) != VOXCPM2_SOURCE_ARCHIVE_SHA256
    ):
        raise VoxCPM2AuditError("Official source archive identity mismatch.")
    try:
        with gzip.open(source_archive, "rb") as compressed:
            while compressed.read(1024 * 1024):
                pass
    except (OSError, gzip.BadGzipFile) as error:
        raise VoxCPM2AuditError("Official source archive failed complete gzip CRC read.") from error
    selected: dict[str, bytes] = {}
    python_files: dict[str, bytes] = {}
    files = directories = regular_bytes = 0
    with tarfile.open(source_archive, "r:gz") as archive:
        members = archive.getmembers()
        for member in members:
            parts = _safe_tar_parts(member.name)
            if not parts or parts[0] != VOXCPM2_SOURCE_ROOT:
                raise VoxCPM2AuditError(f"Unexpected source TAR root: {member.name!r}")
            if member.isdir():
                directories += 1
                continue
            if not member.isfile() or member.issym() or member.islnk():
                raise VoxCPM2AuditError(f"Unsupported source TAR member type: {member.name!r}")
            files += 1
            regular_bytes += member.size
            relative = PurePosixPath(*parts[1:]).as_posix()
            if relative in VOXCPM2_SOURCE_FILES or relative.endswith(".py"):
                payload = _read_tar_member(archive, member)
                if relative in VOXCPM2_SOURCE_FILES:
                    selected[relative] = payload
                if relative.endswith(".py"):
                    python_files[relative] = payload
    if len(members) != 98 or files != 75 or directories != 23 or regular_bytes != 5_941_036:
        raise VoxCPM2AuditError("Official source TAR aggregate inventory mismatch.")
    for name, (size, digest) in VOXCPM2_SOURCE_FILES.items():
        selected_payload = selected.get(name)
        if (
            selected_payload is None
            or len(selected_payload) != size
            or _digest_bytes(selected_payload) != digest
        ):
            raise VoxCPM2AuditError(f"Pinned source member identity mismatch: {name}")
    project = tomllib.loads(selected["pyproject.toml"].decode("utf-8"))["project"]
    calls, all_safe = _torch_load_policy(python_files)
    core = selected["src/voxcpm/core.py"].decode("utf-8")
    return {
        "members": len(members),
        "files": files,
        "directories": directories,
        "regular_bytes": regular_bytes,
        "torch_load_calls": calls,
        "torch_load_all_weights_only": all_safe,
        "license": project["license"],
        "requires_python": project["requires-python"],
        "local_path_loader_verified": "if os.path.isdir(repo_id):" in core,
        "denoiser_can_be_disabled": (
            "zipenhancer_model_path=zipenhancer_model_id if load_denoiser else None" in core
        ),
        "semantic_normalizer_can_be_disabled": "if normalize:" in core,
        "retry_can_be_disabled": "retry_badcase=retry_badcase" in core,
        "upstream_whitespace_collapse_unconditional": (
            'text = text.replace("\\n", " ")' in core
            and 'text = re.sub(r"\\s+", " ", text)' in core
        ),
    }


def audit_voxcpm2_artifacts(
    model_root: Path, source_archive: Path
) -> VoxCPM2ArtifactAudit:
    """Audit exact local artifacts without importing VoxCPM or running inference."""

    _verify_model_files(model_root)
    config = json.loads((model_root / "config.json").read_text(encoding="utf-8"))
    if config.get("architecture") != "voxcpm2":
        raise VoxCPM2AuditError("Pinned config is not architecture=voxcpm2.")
    header_bytes, tensor_count, dtype_counts, payload_bytes = _audit_safetensors(
        model_root / "model.safetensors"
    )
    zip_members, pickle_globals = _audit_audiovae(model_root / "audiovae.pth")
    state_tensors, state_dtypes, state_elements, vae_sample_rate = _weights_only_audiovae(
        model_root / "audiovae.pth"
    )
    tokenizer_imports, tokenizer_forbidden = _tokenizer_python_policy(
        (model_root / "tokenization_voxcpm2.py").read_bytes()
    )
    source = _audit_source_archive(source_archive)
    required_truths = (
        source["torch_load_calls"] == 11,
        source["torch_load_all_weights_only"] is True,
        source["license"] == "Apache-2.0",
        source["requires_python"] == ">=3.10",
        source["local_path_loader_verified"] is True,
        source["denoiser_can_be_disabled"] is True,
        source["semantic_normalizer_can_be_disabled"] is True,
        source["retry_can_be_disabled"] is True,
        source["upstream_whitespace_collapse_unconditional"] is True,
        tokenizer_imports == ("transformers",),
        not tokenizer_forbidden,
    )
    if not all(required_truths):
        raise VoxCPM2AuditError("Pinned source/runtime policy differs from the admitted route.")
    return VoxCPM2ArtifactAudit(
        model_revision=VOXCPM2_MODEL_REVISION,
        source_revision=VOXCPM2_SOURCE_REVISION,
        model_files=len(VOXCPM2_MODEL_FILES),
        model_bytes=sum(size for size, _ in VOXCPM2_MODEL_FILES.values()),
        model_inventory_sha256=_inventory_digest(VOXCPM2_MODEL_FILES),
        architecture=config["architecture"],
        output_sample_rate_hz=int(config["audio_vae_config"]["out_sample_rate"]),
        safetensors_header_bytes=header_bytes,
        safetensors_tensors=tensor_count,
        safetensors_dtype_counts=dtype_counts,
        safetensors_payload_bytes=payload_bytes,
        safetensors_offsets_contiguous=True,
        audiovae_zip_members=zip_members,
        audiovae_zip_crc_verified=True,
        audiovae_pickle_globals=pickle_globals,
        audiovae_weights_only_loaded=True,
        audiovae_state_tensors=state_tensors,
        audiovae_state_dtype_counts=state_dtypes,
        audiovae_state_elements=state_elements,
        audiovae_sample_rate_hz=vae_sample_rate,
        tokenizer_python_imports=tokenizer_imports,
        tokenizer_python_forbidden_calls=tokenizer_forbidden,
        source_archive_members=source["members"],
        source_archive_files=source["files"],
        source_archive_directories=source["directories"],
        source_archive_regular_bytes=source["regular_bytes"],
        source_gzip_crc_verified=True,
        source_torch_load_calls=source["torch_load_calls"],
        source_torch_load_all_weights_only=source["torch_load_all_weights_only"],
        source_license=source["license"],
        source_requires_python=source["requires_python"],
        local_path_loader_verified=source["local_path_loader_verified"],
        denoiser_can_be_disabled=source["denoiser_can_be_disabled"],
        semantic_normalizer_can_be_disabled=source["semantic_normalizer_can_be_disabled"],
        retry_can_be_disabled=source["retry_can_be_disabled"],
        upstream_whitespace_collapse_unconditional=source[
            "upstream_whitespace_collapse_unconditional"
        ],
        tts_model_loaded=False,
        synthesis_performed=False,
        detector_inference_performed=False,
    )
