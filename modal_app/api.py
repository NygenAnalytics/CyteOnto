"""FastAPI HTTP layer for the CyteOnto Modal app."""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse

from .config import AppConfig
from .models import CompareRequest, CompareResponse, StatusResponse

app_config = AppConfig()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _status_path(run_id: str) -> Path:
    return Path(app_config.REMOTE_USER_DIR) / run_id / "status.json"


def _read_status(run_id: str, volume) -> dict[str, Any] | None:
    volume.reload()
    path = _status_path(run_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _write_status(run_id: str, data: dict[str, Any], volume) -> None:
    path = _status_path(run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))
    volume.commit()


def create_app(volume, run_compare_fn) -> FastAPI:
    """Build the FastAPI app bound to the given Modal volume and worker function."""
    app = FastAPI(title="CyteOnto API", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, bool]:
        return {"ok": True}

    @app.post("/compare", response_model=CompareResponse)
    def submit_compare(req: CompareRequest) -> CompareResponse:
        if not req.authorLabels:
            raise HTTPException(400, "authorLabels must be non-empty")
        if not req.algorithms:
            raise HTTPException(400, "algorithms must contain at least one entry")
        for name, labels in req.algorithms.items():
            if len(labels) != len(req.authorLabels):
                raise HTTPException(
                    400,
                    f"Algorithm '{name}' has {len(labels)} labels; "
                    f"expected {len(req.authorLabels)}",
                )

        run_id = f"run-{uuid.uuid4()}"
        status = {
            "runId": run_id,
            "state": "queued",
            "createdAt": _utc_now(),
            "startedAt": None,
            "completedAt": None,
            "error": None,
            "numAuthorLabels": len(req.authorLabels),
            "numAlgorithms": len(req.algorithms),
            "numRows": None,
            "resultCsvPath": None,
            "resultJsonPath": None,
        }
        _write_status(run_id, status, volume)

        run_compare_fn.spawn(
            run_id=run_id,
            payload=req.model_dump(),
        )
        return CompareResponse(runId=run_id, state="queued")

    @app.get("/status/{run_id}", response_model=StatusResponse)
    def get_status(run_id: str) -> StatusResponse:
        status = _read_status(run_id, volume)
        if status is None:
            raise HTTPException(404, f"run_id not found: {run_id}")
        return StatusResponse(**status)

    @app.get("/result/{run_id}")
    def get_result(run_id: str, format: str = "json"):
        if format not in ("json", "csv"):
            raise HTTPException(400, "format must be 'json' or 'csv'")

        status = _read_status(run_id, volume)
        if status is None:
            raise HTTPException(404, f"run_id not found: {run_id}")
        if status["state"] != "completed":
            raise HTTPException(
                409,
                f"Run is '{status['state']}', not completed",
            )

        rel = status["resultJsonPath"] if format == "json" else status["resultCsvPath"]
        if not rel:
            raise HTTPException(500, f"No {format} result recorded for run {run_id}")
        full = Path(app_config.REMOTE_DATA_DIR) / rel
        if not full.exists():
            raise HTTPException(500, f"Result file missing on disk: {full}")

        if format == "json":
            return JSONResponse(json.loads(full.read_text()))
        return FileResponse(
            full,
            media_type="text/csv",
            filename=f"{run_id}.csv",
        )

    return app
