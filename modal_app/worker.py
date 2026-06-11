"""Compare-job worker body. Runs inside the worker Modal container."""

import json
import os
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic_ai import Agent

from cyteonto.config import Config as CyteConfig

from .config import AppConfig

app_config = AppConfig()
cyte_config = CyteConfig()


def _env_api_key(provider: str) -> str:
    env_var = cyte_config.PROVIDER_API_KEY_ENV.get(provider)
    if not env_var:
        return ""
    return os.environ.get(env_var, "")


def _resolve_api_keys(payload: dict[str, Any]) -> dict[str, Any]:
    """Fill missing LLM/embedding keys from environment for known providers."""
    resolved = dict(payload)

    if not resolved.get("llmApiKey"):
        key = _env_api_key(resolved.get("llmProvider", ""))
        if not key:
            raise RuntimeError(
                f"llmApiKey missing and no env var for llmProvider="
                f"'{resolved.get('llmProvider')}'"
            )
        resolved["llmApiKey"] = key

    if not resolved.get("embeddingApiKey"):
        if resolved.get("embeddingProvider") == "ollama":
            pass
        else:
            key = _env_api_key(resolved.get("embeddingProvider", ""))
            if not key:
                key = cyte_config.EMBEDDING_API_KEY
            if not key:
                raise RuntimeError(
                    f"embeddingApiKey missing and no env var for "
                    f"embeddingProvider='{resolved.get('embeddingProvider')}'"
                )
            resolved["embeddingApiKey"] = key

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
    "nebius": "https://api.tokenfactory.nebius.com/v1/",
    "fireworks": "https://api.fireworks.ai/inference/v1",
    "openai": None,
}


def build_agent(provider: str, model: str, api_key: str) -> Agent:
    """Build a pydantic-ai Agent for an OpenAI-compatible provider."""
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

    if provider not in _LLM_BASE_URLS:
        raise ValueError(f"Unsupported llm provider: {provider}")

    base_url = _LLM_BASE_URLS[provider]
    provider_kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url is not None:
        provider_kwargs["base_url"] = base_url

    return Agent(OpenAIChatModel(model, provider=OpenAIProvider(**provider_kwargs)))


def _fallback_configs() -> tuple[Agent, Any, Any]:
    from cyteonto.models import EmbdConfig, LlmConfig

    fb_llm = LlmConfig(
        provider=cyte_config.FALLBACK_LLM_PROVIDER,
        model=cyte_config.FALLBACK_LLM_MODEL,
    )
    fb_key = _env_api_key(cyte_config.FALLBACK_LLM_PROVIDER)
    fb_agent = build_agent(
        cyte_config.FALLBACK_LLM_PROVIDER,
        cyte_config.FALLBACK_LLM_MODEL,
        fb_key,
    )
    fb_embedding = EmbdConfig(
        provider=cyte_config.FALLBACK_EMBEDDING_PROVIDER,  # type: ignore[arg-type]
        model=cyte_config.FALLBACK_EMBEDDING_MODEL,
        apiKey=_env_api_key(cyte_config.FALLBACK_EMBEDDING_PROVIDER)
        or cyte_config.EMBEDDING_API_KEY,
    )
    return fb_agent, fb_embedding, fb_llm


async def run_compare_job(run_id: str, payload: dict[str, Any], volume) -> None:
    """Execute one compare request end-to-end and update `status.json` at each stage."""
    from cyteonto import CyteOnto
    from cyteonto.logger import logger
    from cyteonto.models import EmbdConfig, LlmConfig

    status = _read_status(run_id)
    status.update({"state": "running", "startedAt": _utc_now()})
    _write_status(run_id, status)
    await volume.commit.aio()

    csv_path, json_path = _result_paths(run_id)
    cyto = None

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
        fb_agent, fb_embedding, fb_llm = _fallback_configs()

        cyto = await CyteOnto.from_config(
            agent=agent,
            embedding=embedding,
            llm=LlmConfig(
                provider=payload["llmProvider"],
                model=payload["llmModel"],
            ),
            data_dir=app_config.REMOTE_DATA_DIR,
            user_dir=app_config.REMOTE_USER_DIR,
            max_description_concurrency=payload["maxDescriptionConcurrency"],
            use_pubmed_tool=payload["usePubmedTool"],
            reasoning=payload["reasoning"],
            fallback_agent=fb_agent,
            fallback_embedding=fb_embedding,
            fallback_llm=fb_llm,
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
                "modelPairUsage": cyto.model_pair_usage.to_status_dict(),
            }
        )
        logger.info(f"[{run_id}] compare completed ({len(df)} rows)")

    except Exception as exc:
        tb = traceback.format_exc()
        logger.error(f"[{run_id}] compare failed: {type(exc).__name__}: {exc}\n{tb}")
        fail_update: dict[str, Any] = {
            "state": "failed",
            "completedAt": _utc_now(),
            "error": f"{type(exc).__name__}: {exc}",
        }
        if cyto is not None:
            fail_update["modelPairUsage"] = cyto.model_pair_usage.to_status_dict()
        status.update(fail_update)

    finally:
        _write_status(run_id, status)
        await volume.commit.aio()
