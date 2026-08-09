#!/usr/bin/env bash
set -euo pipefail

archive_dir="${1:-data/raw/ksc_slr102}"
archive_name="ISSAI_KSC_335RS_v1.1_flac.tar.gz"
archive_url="https://openslr.trmal.net/resources/102/${archive_name}"
expected_size_bytes=19092377812
archive_path="${archive_dir}/${archive_name}"

mkdir -p "${archive_dir}"
if [[ -e "${archive_path}" ]]; then
  actual_size_bytes="$(stat --format='%s' "${archive_path}")"
  if (( actual_size_bytes > expected_size_bytes )); then
    echo "Archive is oversized: expected ${expected_size_bytes} bytes, got ${actual_size_bytes}." >&2
    echo "Refusing to modify it. Keep it for investigation and acquire a clean archive." >&2
    exit 2
  fi
else
  actual_size_bytes=0
fi

if (( actual_size_bytes < expected_size_bytes )); then
  curl --fail --location --continue-at - --output "${archive_path}" "${archive_url}"
fi

actual_size_bytes="$(stat --format='%s' "${archive_path}")"
if (( actual_size_bytes != expected_size_bytes )); then
  echo "Download size mismatch: expected ${expected_size_bytes} bytes, got ${actual_size_bytes}." >&2
  exit 2
fi

gzip --test "${archive_path}"
sha256sum "${archive_path}"
