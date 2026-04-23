"""Async HTTP embedding generation for supported providers."""

import asyncio
from typing import Any

import aiohttp
import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential

from .config import Config
from .logger import logger
from .models import EmbdConfig

config = Config()


def _headers(cfg: EmbdConfig) -> dict[str, str]:
    key = cfg.apiKey or ""
    if cfg.provider == "ollama":
        return {}
    if cfg.provider == "google":
        return {"x-goog-api-key": key, "Content-Type": "application/json"}
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def _build_payload(text: str, cfg: EmbdConfig) -> dict[str, Any]:
    if cfg.provider == "google":
        return {"model": cfg.model, "content": {"parts": [{"text": text}]}}
    payload: dict[str, Any] = {
        "model": cfg.model,
        "input": [text],
        "encoding_format": "float",
    }
    if cfg.provider == "openrouter" and cfg.modelSettings:
        payload.update(cfg.modelSettings)
    return payload


def _extract_embedding(result: dict[str, Any], cfg: EmbdConfig) -> list[float]:
    if cfg.provider == "google":
        embd = result.get("embeddings") or result.get("embedding")
        if isinstance(embd, dict) and "values" in embd:
            return embd["values"]
        if isinstance(embd, list) and embd:
            first = embd[0]
            return first.get("values", first) if isinstance(first, dict) else first
        return []
    data = result.get("data")
    if data:
        return data[0].get("embedding", [])
    return []


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
async def _request_one(
    session: aiohttp.ClientSession,
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
) -> list[float]:
    async with session.post(url, json=payload, headers=headers) as resp:
        if resp.status != 200:
            body = await resp.text()
            raise RuntimeError(
                f"Embedding request failed ({resp.status}): {body[:200]}"
            )
        return await resp.json()  # type: ignore[return-value]


async def embed_texts(texts: list[str], cfg: EmbdConfig) -> np.ndarray | None:
    """Embed a list of texts, one request per text, with bounded concurrency."""
    if not texts:
        return np.zeros((0, 0))
    if cfg.provider != "ollama" and not cfg.apiKey:
        logger.error("Embedding API key missing")
        return None

    if cfg.provider not in config._PROVIDER_URL:
        logger.error(f"Unsupported embedding provider: {cfg.provider}")
        return None

    url = config._PROVIDER_URL[cfg.provider]
    headers = _headers(cfg)
    total = len(texts)
    sem = asyncio.Semaphore(cfg.maxConcurrent)
    results: list[list[float] | None] = [None] * total
    done = 0
    done_lock = asyncio.Lock()
    logger.info(
        f"Embedding {total} texts (provider={cfg.provider}, "
        f"model={cfg.model}, concurrency={cfg.maxConcurrent})"
    )

    async def run_one(i: int, text: str) -> None:
        nonlocal done
        async with sem:
            try:
                async with aiohttp.ClientSession() as session:
                    raw = await _request_one(
                        session, url, headers, _build_payload(text, cfg)
                    )
                results[i] = _extract_embedding(raw, cfg)  # type: ignore[arg-type]
            except Exception as e:
                logger.error(f"[{i + 1}/{total}] Embedding failed: {e}")
                results[i] = None
        async with done_lock:
            done += 1
            if done == total or done % max(1, total // 20) == 0:
                logger.info(f"Embedding progress: {done}/{total}")

    await asyncio.gather(*(run_one(i, t) for i, t in enumerate(texts)))

    missing = [i for i, r in enumerate(results) if not r]
    if missing:
        logger.error(f"{len(missing)} embeddings failed out of {len(texts)}")
        return None

    return np.asarray(results, dtype=np.float32)
