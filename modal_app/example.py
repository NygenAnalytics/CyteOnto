import os
import sys
import time
from pathlib import Path

import requests

BASE_URL = os.environ.get("CYTEONTO_URL", "https://cyteonto.nygen.io")

LLM_API_KEY = None  # or  os.environ.get("LLM_API_KEY")
EMBEDDING_API_KEY = None  # or  os.environ.get("EMBEDDING_MODEL_API_KEY")

PAYLOAD: dict = {
    "authorLabels": [
        "alveolar macrophage",
        "regulatory T cell",
        "CD8-positive, alpha-beta T cell",
    ],
    "algorithms": {
        "algo1": ["lung macrophage", "Treg", "CD8 T cell"],
        "algo2": ["alveolar mac", "T regulatory cell", "cytotoxic T cell"],
    },
    "maxDescriptionConcurrency": 50,
    "embeddingMaxConcurrent": 50,
    # Using Default Providers and modal secrets API keys.
    # Check out README for more details.
}

POLL_INTERVAL_S = 10
POLL_TIMEOUT_S = 60 * 60
RESULT_FORMAT = "csv"
OUTPUT_DIR = Path("./cyteonto_results")

def submit(payload: dict) -> str:
    body = dict(payload)
    if LLM_API_KEY:
        body["llmApiKey"] = LLM_API_KEY
    if EMBEDDING_API_KEY:
        body["embeddingApiKey"] = EMBEDDING_API_KEY
    resp = requests.post(f"{BASE_URL}/compare", json=body, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    print(f"[submit] runId={data['runId']} state={data['state']}")
    return data["runId"]


def poll(run_id: str, interval_s: int, timeout_s: int) -> dict:
    deadline = time.time() + timeout_s
    last_state = None
    while time.time() < deadline:
        resp = requests.get(f"{BASE_URL}/status/{run_id}", timeout=30)
        resp.raise_for_status()
        status = resp.json()
        if status["state"] != last_state:
            print(f"[status] {status['state']}")
            last_state = status["state"]
        if status["state"] in ("completed", "failed"):
            return status
        time.sleep(interval_s)
    raise TimeoutError(f"Run {run_id} did not finish within {timeout_s}s")


def fetch_result(run_id: str, fmt: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    resp = requests.get(
        f"{BASE_URL}/result/{run_id}",
        params={"format": fmt},
        timeout=60,
    )
    resp.raise_for_status()
    suffix = "csv" if fmt == "csv" else "json"
    out_path = out_dir / f"{run_id}.{suffix}"
    out_path.write_bytes(resp.content)
    return out_path


def main() -> int:
    run_id = submit(PAYLOAD)
    status = poll(run_id, POLL_INTERVAL_S, POLL_TIMEOUT_S)

    if status["state"] == "failed":
        print(f"[failed] {status.get('error')}", file=sys.stderr)
        return 1

    path = fetch_result(run_id, RESULT_FORMAT, OUTPUT_DIR)
    print(f"[done] {status['numRows']} rows saved to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
