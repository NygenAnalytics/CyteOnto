"""Compare-job worker body. Runs inside the worker Modal container."""

import json
import os
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic_ai import Agent

from .config import AppConfig

app_config = AppConfig()


def _resolve_api_keys(payload: dict[str, Any]) -> dict[str, Any]:
    """Fill missing LLM/embedding keys from the attached secret when the provider matches."""
    resolved = dict(payload)

    if not resolved.get("llmApiKey"):
        if resolved.get("llmProvider") == app_config.LLM_SECRET_PROVIDER:
            secret_key = os.environ.get(app_config.LLM_SECRET_ENV_VAR)
            if not secret_key:
                raise RuntimeError(
                    f"{app_config.LLM_SECRET_ENV_VAR} is not set on the worker. "
                    f"Attach the '{app_config.SECRET_NAME}' Modal secret or "
                    f"pass llmApiKey in the request."
                )
            resolved["llmApiKey"] = secret_key
        else:
            raise RuntimeError(
                f"llmApiKey missing and llmProvider='{resolved.get('llmProvider')}' "
                f"is not covered by the hosted secret."
            )

    if not resolved.get("embeddingApiKey"):
        if resolved.get("embeddingProvider") == app_config.EMBEDDING_SECRET_PROVIDER:
            secret_key = os.environ.get(app_config.EMBEDDING_SECRET_ENV_VAR)
            if not secret_key:
                raise RuntimeError(
                    f"{app_config.EMBEDDING_SECRET_ENV_VAR} is not set on the worker. "
                    f"Attach the '{app_config.SECRET_NAME}' Modal secret or "
                    f"pass embeddingApiKey in the request."
                )
            resolved["embeddingApiKey"] = secret_key
        elif resolved.get("embeddingProvider") != "ollama":
            raise RuntimeError(
                f"embeddingApiKey missing and "
                f"embeddingProvider='{resolved.get('embeddingProvider')}' "
                f"is not covered by the hosted secret."
            )

    return resolved


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _status_path(run_id: str) -> Path:
    return Path(app_config.REMOTE_USER_DIR) / run_id / "status.json"


def _result_paths(run_id: str) -> tuple[Path, Path]:
    base = Path(app_config.REMOTE_USER_DIR) / run_id
    return base / "result.csv", base / "result.json"


def _read_status(run_id: str) -> dict[str, Any]:
    path = _status_path(run_id)
    if not path.exists():
        return {"runId": run_id}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {"runId": run_id}


def _write_status(run_id: str, status: dict[str, Any]) -> None:
    path = _status_path(run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(status, indent=2))


_LLM_BASE_URLS: dict[str, str | None] = {
    "openrouter": "https://openrouter.ai/api/v1/",
    "together": "https://api.together.xyz/v1/",
    "openai": None,
}


def build_agent(provider: str, model: str, api_key: str) -> Agent:
    """Build a pydantic-ai Agent for an OpenAI-compatible provider.

    Uses ``OpenAIChatModel`` with a provider-specific ``base_url`` so the same code
    path works for OpenRouter, Together, and OpenAI without depending on
    version-specific pydantic-ai submodules.
    """
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

    if provider not in _LLM_BASE_URLS:
        raise ValueError(f"Unsupported llm provider: {provider}")

    base_url = _LLM_BASE_URLS[provider]
    provider_kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url is not None:
        provider_kwargs["base_url"] = base_url

    return Agent(OpenAIChatModel(model, provider=OpenAIProvider(**provider_kwargs)))


async def run_compare_job(run_id: str, payload: dict[str, Any], volume) -> None:
    """Execute one compare request end-to-end and update `status.json` at each stage."""
    from cyteonto_new import CyteOnto
    from cyteonto_new.logger import logger
    from cyteonto_new.models import EmbdConfig

    status = _read_status(run_id)
    status.update({"state": "running", "startedAt": _utc_now()})
    _write_status(run_id, status)
    await volume.commit.aio()

    csv_path, json_path = _result_paths(run_id)

    try:
        payload = _resolve_api_keys(payload)

        agent = build_agent(
            payload["llmProvider"], payload["llmModel"], payload["llmApiKey"]
        )
        embedding = EmbdConfig(
            provider=payload["embeddingProvider"],
            model=payload["embeddingModel"],
            apiKey=payload["embeddingApiKey"],
            modelSettings=payload.get("embeddingModelSettings"),
            maxConcurrent=payload["embeddingMaxConcurrent"],
        )

        cyto = await CyteOnto.from_config(
            agent=agent,
            embedding=embedding,
            data_dir=app_config.REMOTE_DATA_DIR,
            user_dir=app_config.REMOTE_USER_DIR,
            max_description_concurrency=payload["maxDescriptionConcurrency"],
            use_pubmed_tool=payload["usePubmedTool"],
            reasoning=payload["reasoning"],
        )

        df = await cyto.compare(
            author_labels=payload["authorLabels"],
            algorithms=payload["algorithms"],
            run_id=run_id,
            metric=payload["metric"],
            metric_params=payload.get("metricParams"),
            min_match_similarity=payload["minMatchSimilarity"],
            use_cache=payload["useCache"],
        )

        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        df.to_json(json_path, orient="records", indent=2)

        status.update(
            {
                "state": "completed",
                "completedAt": _utc_now(),
                "numRows": int(len(df)),
                "resultCsvPath": str(csv_path.relative_to(app_config.REMOTE_DATA_DIR)),
                "resultJsonPath": str(
                    json_path.relative_to(app_config.REMOTE_DATA_DIR)
                ),
                "error": None,
            }
        )
        logger.info(f"[{run_id}] compare completed ({len(df)} rows)")

    except Exception as exc:
        tb = traceback.format_exc()
        logger.error(f"[{run_id}] compare failed: {type(exc).__name__}: {exc}\n{tb}")
        status.update(
            {
                "state": "failed",
                "completedAt": _utc_now(),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )

    finally:
        _write_status(run_id, status)
        await volume.commit.aio()
