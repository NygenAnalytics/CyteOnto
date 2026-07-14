#!/usr/bin/env bash

set -euo pipefail

CYTEONTO_URL="${CYTEONTO_URL:-https://cyteonto.nygen.io}"
POLL_INTERVAL="${POLL_INTERVAL:-5}"

command -v curl >/dev/null || {
  echo "curl is required" >&2
  exit 1
}
command -v jq >/dev/null || {
  echo "jq is required" >&2
  exit 1
}

response="$(
  curl -fsS -X POST "${CYTEONTO_URL}/compare" \
    -H 'Content-Type: application/json' \
    --data-binary @- <<'JSON'
{
  "authorLabels": [
    "animal stem cell",
    "BFU-E",
    "neutrophilic granuloblast"
  ],
  "algorithms": {
    "method_a": [
      "stem cell",
      "blast forming unit erythroid",
      "spermatogonium"
    ],
    "method_b": [
      "neuronal receptor cell",
      "stem cell",
      "ovum"
    ]
  },
  "metric": "cosine_kernel"
}
JSON
)"

run_id="$(jq -er '.runId' <<<"$response")"
echo "Submitted ${run_id}"

while true; do
  status="$(curl -fsS "${CYTEONTO_URL}/status/${run_id}")"
  state="$(jq -er '.state' <<<"$status")"
  echo "State: ${state}"

  case "$state" in
    completed)
      curl -fsS "${CYTEONTO_URL}/result/${run_id}?format=json" | jq
      break
      ;;
    failed)
      jq >&2 <<<"$status"
      exit 1
      ;;
    queued|running)
      sleep "$POLL_INTERVAL"
      ;;
    *)
      echo "Unexpected state: ${state}" >&2
      jq >&2 <<<"$status"
      exit 1
      ;;
  esac
done
