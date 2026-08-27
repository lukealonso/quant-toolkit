#!/usr/bin/env bash
set -euo pipefail

# Mirror Brandon's sealed GLM-5.3 BF16 teacher-logit panel without following
# mutable Hub refs. The logit payload is large, so every transfer is resumable
# and is promoted from .incomplete only after size and SHA-256 verification.

readonly REPO="brandonmusic/GLM-5.3-Flash-BF16-Teacher-Logits"
readonly REVISION="17ffa06d2ee899b6043c3307bc572bc944ed5ea5"
readonly DEFAULT_DESTINATION="/volume1/models/hf/datasets--brandonmusic--GLM-5.3-Flash-BF16-Teacher-Logits/snapshots/${REVISION}"

destination="${1:-${GLM53_TEACHER_DESTINATION:-${DEFAULT_DESTINATION}}}"
parallel_downloads="${GLM53_TEACHER_DOWNLOAD_JOBS:-4}"
base_url="https://huggingface.co/datasets/${REPO}/resolve/${REVISION}"

case "${parallel_downloads}" in
  ''|*[!0-9]*)
    echo "GLM53_TEACHER_DOWNLOAD_JOBS must be a positive integer" >&2
    exit 2
    ;;
esac
if (( parallel_downloads < 1 )); then
  echo "GLM53_TEACHER_DOWNLOAD_JOBS must be a positive integer" >&2
  exit 2
fi

for command_name in curl jq sha256sum flock; do
  if ! command -v "${command_name}" >/dev/null 2>&1; then
    echo "required command is unavailable: ${command_name}" >&2
    exit 2
  fi
done

mkdir -p "${destination}/logits"
exec 9>"${destination}/.mirror.lock"
if ! flock -n 9; then
  echo "another mirror process holds ${destination}/.mirror.lock" >&2
  exit 3
fi

fetch_file() {
  local relative_path="$1"
  local expected_sha256="${2:-}"
  local expected_bytes="${3:-}"
  local final_path="${destination}/${relative_path}"
  local temporary_path="${final_path}.incomplete"
  local actual_sha256 actual_bytes quarantine_suffix

  mkdir -p "$(dirname "${final_path}")"

  if [[ -f "${final_path}" ]]; then
    if [[ -z "${expected_sha256}" ]]; then
      echo "present ${relative_path}"
      return 0
    fi
    actual_bytes="$(stat -c %s "${final_path}")"
    actual_sha256="$(sha256sum "${final_path}" | awk '{print $1}')"
    if [[ "${actual_bytes}" == "${expected_bytes}" && "${actual_sha256}" == "${expected_sha256}" ]]; then
      echo "verified ${relative_path}"
      return 0
    fi
    quarantine_suffix="$(date -u +%Y%m%dT%H%M%SZ)"
    mv "${final_path}" "${final_path}.corrupt-${quarantine_suffix}"
  fi

  if [[ -f "${temporary_path}" && -n "${expected_bytes}" ]]; then
    actual_bytes="$(stat -c %s "${temporary_path}")"
    if (( actual_bytes > expected_bytes )); then
      quarantine_suffix="$(date -u +%Y%m%dT%H%M%SZ)"
      mv "${temporary_path}" "${temporary_path}.oversize-${quarantine_suffix}"
    elif (( actual_bytes == expected_bytes )); then
      actual_sha256="$(sha256sum "${temporary_path}" | awk '{print $1}')"
      if [[ "${actual_sha256}" == "${expected_sha256}" ]]; then
        mv "${temporary_path}" "${final_path}"
        echo "verified ${relative_path}"
        return 0
      fi
      quarantine_suffix="$(date -u +%Y%m%dT%H%M%SZ)"
      mv "${temporary_path}" "${temporary_path}.corrupt-${quarantine_suffix}"
    fi
  fi

  echo "downloading ${relative_path}"
  curl \
    --fail \
    --location \
    --continue-at - \
    --retry 12 \
    --retry-delay 3 \
    --connect-timeout 30 \
    --speed-time 120 \
    --speed-limit 1024 \
    --output "${temporary_path}" \
    "${base_url}/${relative_path}"

  if [[ -n "${expected_bytes}" ]]; then
    actual_bytes="$(stat -c %s "${temporary_path}")"
    if [[ "${actual_bytes}" != "${expected_bytes}" ]]; then
      echo "size mismatch for ${relative_path}: expected ${expected_bytes}, got ${actual_bytes}" >&2
      return 4
    fi
  fi
  if [[ -n "${expected_sha256}" ]]; then
    actual_sha256="$(sha256sum "${temporary_path}" | awk '{print $1}')"
    if [[ "${actual_sha256}" != "${expected_sha256}" ]]; then
      echo "SHA-256 mismatch for ${relative_path}: expected ${expected_sha256}, got ${actual_sha256}" >&2
      return 5
    fi
  fi
  mv "${temporary_path}" "${final_path}"
  echo "verified ${relative_path}"
}

export destination base_url
export -f fetch_file

readonly metadata_files=(
  README.md
  backend.json
  calibration/panel-v1/calibration.sealed-corpus.json
  calibration/panel-v1/corpus.receipt.json
  calibration/panel-v1/panel.json
  calibration/panel-v1/panel.receipt.json
  calibration/panel-v1/tokenizer.receipt.json
  capture-receipt.json
  dataset-manifest.json
  plan.json
  source-inventory.json
  token-panel-receipt.json
)

for metadata_file in "${metadata_files[@]}"; do
  fetch_file "${metadata_file}"
done

readonly panel_dir="${destination}/calibration/panel-v1"
if [[ "$(jq -r '.schema' "${panel_dir}/panel.json")" != "quant-pipeline.glm53-token-panel.v1" ]]; then
  echo "unexpected token panel schema" >&2
  exit 6
fi

# Mirror every sealed calibration role so quantization never needs to retokenize
# Brandon's corpus. The final 25 still remain qualification-only.
fetch_file \
  "calibration/panel-v1/arrays/causal-mask-2048.npy" \
  "$(jq -r '.windows[].attention_mask_sha256' "${panel_dir}/panel.json" | sort -u)" \
  2176
# The positional parameters intentionally expand inside each child shell.
# shellcheck disable=SC2016
jq -r '.windows[] | ["calibration/panel-v1/arrays/" + .window_id + ".tokens.npy", .token_ids_sha256, "8320"] | @tsv' \
  "${panel_dir}/panel.json" \
  | xargs -P "${parallel_downloads}" -n 3 bash -c 'fetch_file "$1" "$2" "$3"' _

verified_token_windows=0
verified_token_bytes=0
while IFS=$'\t' read -r relative_path expected_sha256 expected_bytes; do
  final_path="${destination}/${relative_path}"
  actual_bytes="$(stat -c %s "${final_path}")"
  actual_sha256="$(sha256sum "${final_path}" | awk '{print $1}')"
  if [[ "${actual_bytes}" != "${expected_bytes}" || "${actual_sha256}" != "${expected_sha256}" ]]; then
    echo "post-download verification failed for ${relative_path}" >&2
    exit 7
  fi
  verified_token_windows=$((verified_token_windows + 1))
  verified_token_bytes=$((verified_token_bytes + actual_bytes))
done < <(jq -r '.windows[] | ["calibration/panel-v1/arrays/" + .window_id + ".tokens.npy", .token_ids_sha256, "8320"] | @tsv' "${panel_dir}/panel.json")

if [[ "$(jq -r '.schema' "${destination}/dataset-manifest.json")" != "quant-pipeline.glm53-bf16-teacher-logits-dataset.v1" ]]; then
  echo "unexpected dataset manifest schema" >&2
  exit 6
fi
if [[ "$(jq -r '.logit_files | length' "${destination}/dataset-manifest.json")" != "25" ]]; then
  echo "expected exactly 25 sealed logit windows" >&2
  exit 6
fi

# The positional parameters intentionally expand inside each child shell.
# shellcheck disable=SC2016
jq -r '.logit_files[] | [.path, .sha256, (.bytes | tostring)] | @tsv' \
  "${destination}/dataset-manifest.json" \
  | xargs -P "${parallel_downloads}" -n 3 bash -c 'fetch_file "$1" "$2" "$3"' _

verified_windows=0
verified_bytes=0
while IFS=$'\t' read -r relative_path expected_sha256 expected_bytes; do
  final_path="${destination}/${relative_path}"
  actual_bytes="$(stat -c %s "${final_path}")"
  actual_sha256="$(sha256sum "${final_path}" | awk '{print $1}')"
  if [[ "${actual_bytes}" != "${expected_bytes}" || "${actual_sha256}" != "${expected_sha256}" ]]; then
    echo "post-download verification failed for ${relative_path}" >&2
    exit 7
  fi
  verified_windows=$((verified_windows + 1))
  verified_bytes=$((verified_bytes + actual_bytes))
done < <(jq -r '.logit_files[] | [.path, .sha256, (.bytes | tostring)] | @tsv' "${destination}/dataset-manifest.json")

receipt_temporary="${destination}/.mirror-receipt.json.incomplete"
receipt_final="${destination}/mirror-receipt.json"
jq -n \
  --arg schema "quant-toolkit.hf-dataset-mirror-receipt.v1" \
  --arg repo "${REPO}" \
  --arg revision "${REVISION}" \
  --arg mirrored_at "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --arg destination "${destination}" \
  --arg dataset_manifest_sha256 "$(sha256sum "${destination}/dataset-manifest.json" | awk '{print $1}')" \
  --arg capture_receipt_sha256 "$(sha256sum "${destination}/capture-receipt.json" | awk '{print $1}')" \
  --arg token_panel_receipt_sha256 "$(sha256sum "${destination}/token-panel-receipt.json" | awk '{print $1}')" \
  --arg panel_sha256 "$(sha256sum "${panel_dir}/panel.json" | awk '{print $1}')" \
  --argjson verified_token_windows "${verified_token_windows}" \
  --argjson verified_token_bytes "${verified_token_bytes}" \
  --argjson verified_windows "${verified_windows}" \
  --argjson verified_bytes "${verified_bytes}" \
  '{
    schema: $schema,
    repo: $repo,
    revision: $revision,
    mirrored_at: $mirrored_at,
    destination: $destination,
    dataset_manifest_sha256: $dataset_manifest_sha256,
    capture_receipt_sha256: $capture_receipt_sha256,
    token_panel_receipt_sha256: $token_panel_receipt_sha256,
    panel_sha256: $panel_sha256,
    verified_token_windows: $verified_token_windows,
    verified_token_bytes: $verified_token_bytes,
    verified_windows: $verified_windows,
    verified_bytes: $verified_bytes
  }' >"${receipt_temporary}"
mv "${receipt_temporary}" "${receipt_final}"

echo "mirror complete: ${verified_windows} windows, ${verified_bytes} bytes"
echo "receipt: ${receipt_final}"
