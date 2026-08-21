"""FastAPI HTTP layer for the CyteOnto Modal app."""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Annotated, Any

from fastapi import FastAPI, HTTPException, Query
from fastapi import Path as ApiPath
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from cyteonto import __version__ as cyteonto_version
from cyteonto.config import Config as CyteConfig

from .config import AppConfig
from .models import (
    CompareRequest,
    CompareResponse,
    ErrorResponse,
    HealthResponse,
    ResultRow,
    StatusResponse,
)

app_config = AppConfig()
cyte_config = CyteConfig()
_volume_lock = Lock()

RunId = Annotated[
    str,
    ApiPath(description="Run identifier returned by POST /compare."),
]
ResultFormat = Annotated[
    str,
    Query(description="Response representation. Supported values are json and csv."),
]

_OPENAPI_TAGS = [
    {
        "name": "Service",
        "description": "Service availability checks.",
    },
    {
        "name": "Comparisons",
        "description": "Submit comparison jobs, monitor progress, and fetch results.",
    },
]

_LANDING_PAGE = """\
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CyteOnto API</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link
    href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400&display=swap"
    rel="stylesheet"
  >
  <style>
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      display: grid;
      place-items: center;
      padding: 2rem;
      background: #ffffff;
      color: #000000;
      font-family: "Inter", sans-serif;
      line-height: 1.2;
    }
    main {
      width: min(58rem, 100%);
    }
    .eyebrow {
      margin: 0 0 1rem;
      color: #b4b4b4;
      font-size: 0.875rem;
      font-weight: 400;
      letter-spacing: -0.04em;
      text-transform: uppercase;
    }
    h1 {
      margin: 0;
      color: #000000;
      font-size: clamp(3rem, 10vw, 6rem);
      font-weight: 400;
      letter-spacing: 0;
      line-height: 1.2;
    }
    .subheadline {
      max-width: 42rem;
      margin: 1rem 0 2.5rem;
      color: #000000;
      font-size: clamp(1.25rem, 3vw, 1.75rem);
      font-weight: 300;
      letter-spacing: -0.04em;
      line-height: 1.2;
    }
    nav {
      display: flex;
      flex-wrap: wrap;
      gap: 0.75rem;
    }
    a {
      padding: 0.75rem 1rem;
      border-radius: 999px;
      background: #0077fc;
      color: #ffffff;
      font-weight: 400;
      letter-spacing: -0.04em;
      text-decoration: none;
    }
    a + a {
      background: #ffffff;
      color: #0077fc;
      box-shadow: inset 0 0 0 1px #0077fc;
    }
    a:hover { text-decoration: underline; }
    a:focus-visible {
      outline: 3px solid #000000;
      outline-offset: 3px;
    }
  </style>
</head>
<body>
  <main>
    <p class="eyebrow">API service</p>
    <h1>CyteOnto</h1>
    <p class="subheadline">
      Compare reference cell type annotations with labels produced by one or
      more algorithms.
    </p>
    <nav aria-label="CyteOnto links">
      <a href="/docs">View API documentation</a>
      <a href="https://nygen.io/">Visit Nygen</a>
    </nav>
  </main>
</body>
</html>
"""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _status_path(run_id: str) -> Path:
    return Path(app_config.REMOTE_USER_DIR) / run_id / "status.json"


def _read_status(run_id: str, volume) -> dict[str, Any] | None:
    with _volume_lock:
        volume.reload()
        path = _status_path(run_id)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return None


def _write_status(run_id: str, data: dict[str, Any], volume) -> None:
    with _volume_lock:
        path = _status_path(run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_name(f".{path.name}.tmp")
        temporary_path.write_text(json.dumps(data, indent=2))
        temporary_path.replace(path)
        volume.commit()


def create_app(volume, run_compare_fn) -> FastAPI:
    """Build the FastAPI app bound to the given Modal volume and worker function."""
    app = FastAPI(
        title="CyteOnto API",
        summary="Compare cell type annotations with the Cell Ontology.",
        description=(
            "CyteOnto compares reference cell labels with labels from one or more "
            "algorithms. Jobs run asynchronously: submit a comparison, poll its "
            "status, then download the result as JSON or CSV."
        ),
        version=cyteonto_version,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        openapi_tags=_OPENAPI_TAGS,
    )

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def landing_page() -> HTMLResponse:
        return HTMLResponse(_LANDING_PAGE)

    @app.get(
        "/health",
        response_model=HealthResponse,
        tags=["Service"],
        summary="Check service health",
        description="Return whether the API process is available.",
        response_description="Current service availability.",
    )
    def health() -> HealthResponse:
        return HealthResponse(ok=True)

    @app.post(
        "/compare",
        response_model=CompareResponse,
        tags=["Comparisons"],
        summary="Submit a comparison job",
        description=(
            "Queue an asynchronous comparison. Every algorithm must provide one "
            "label for each entry in authorLabels."
        ),
        response_description="The queued job identifier and initial state.",
        responses={
            400: {
                "model": ErrorResponse,
                "description": "The request is inconsistent or lacks a required key.",
            }
        },
    )
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

        if (
            req.llmApiKey is None
            and req.llmProvider not in cyte_config.PROVIDER_API_KEY_ENV
        ):
            raise HTTPException(
                400,
                (
                    f"llmApiKey is required when llmProvider='{req.llmProvider}'. "
                    f"Omit llmApiKey only for hosted providers: "
                    f"{sorted(cyte_config.PROVIDER_API_KEY_ENV)}."
                ),
            )
        if (
            req.embeddingApiKey is None
            and req.embeddingProvider != "ollama"
            and req.embeddingProvider not in cyte_config.PROVIDER_API_KEY_ENV
        ):
            raise HTTPException(
                400,
                (
                    f"embeddingApiKey is required when "
                    f"embeddingProvider='{req.embeddingProvider}'. "
                    f"Omit embeddingApiKey only for hosted providers: "
                    f"{sorted(cyte_config.PROVIDER_API_KEY_ENV)} (or ollama)."
                ),
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

    @app.get(
        "/status/{run_id}",
        response_model=StatusResponse,
        tags=["Comparisons"],
        summary="Get comparison status",
        description="Return the latest status recorded for a comparison job.",
        response_description="The current job status.",
        responses={
            404: {
                "model": ErrorResponse,
                "description": "No job exists for the supplied run identifier.",
            }
        },
    )
    def get_status(run_id: RunId) -> StatusResponse:
        status = _read_status(run_id, volume)
        if status is None:
            raise HTTPException(404, f"run_id not found: {run_id}")
        return StatusResponse(**status)

    @app.get(
        "/result/{run_id}",
        response_model=list[ResultRow],
        tags=["Comparisons"],
        summary="Download comparison results",
        description=(
            "Return one row per algorithm and label pair for a completed job. JSON "
            "is returned by default. Set format=csv to download a CSV file."
        ),
        responses={
            200: {
                "description": "Comparison rows represented as JSON or CSV.",
                "content": {
                    "text/csv": {
                        "schema": {
                            "type": "string",
                            "format": "binary",
                        }
                    }
                },
            },
            400: {
                "model": ErrorResponse,
                "description": "The requested result format is unsupported.",
            },
            404: {
                "model": ErrorResponse,
                "description": "No job exists for the supplied run identifier.",
            },
            409: {
                "model": ErrorResponse,
                "description": "The job has not completed successfully.",
            },
            500: {
                "model": ErrorResponse,
                "description": "The recorded result cannot be read.",
            },
        },
    )
    def get_result(run_id: RunId, format: ResultFormat = "json"):
        if format not in ("json", "csv"):
            raise HTTPException(400, "format must be 'json' or 'csv'")

        status = _read_status(run_id, volume)
        if status is None:
            raise HTTPException(404, f"run_id not found: {run_id}")
        if status["state"] != "completed":
            raise HTTPException(
                409,
                f"Job is '{status['state']}', not completed",
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
