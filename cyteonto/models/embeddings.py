# models/embedding.py

import asyncio

import aiohttp
import numpy as np
from tenacity import (
    RetryCallState,
    retry,
    stop_after_attempt,
    wait_exponential,
)

from ..llm_config import (
    EMBDModelConfig,
    get_embd_param,
    get_embd_results,
)
from ..logger_config import logger


def _retry_error_callback(retry_state: RetryCallState) -> None:
    raw_exc = retry_state.outcome.exception() if retry_state.outcome else None
    exc: Exception | None = raw_exc if isinstance(raw_exc, Exception) else None
    logger.error(exc)
    return None


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(min=1, max=10),
    retry_error_callback=_retry_error_callback,
)
async def query_embd_model(
    query: str, embd_model_config: EMBDModelConfig
) -> list[float] | None:
    params = get_embd_param(embd_model_config)
    if embd_model_config.provider == "google":
        payload = {
            "model": embd_model_config.model,
            "content": {"parts": [{"text": query}]},
        }
    else:
        payload = {
            "model": embd_model_config.model,
            "input": [query],
            "encoding_format": "float",
        }
    async with aiohttp.ClientSession() as session:
        async with session.post(
            params["url"], json=payload, headers=params["headers"]
        ) as response:
            if response.status == 200:
                result = await response.json()
                results: list[float] = get_embd_results(result, embd_model_config)
                return results
            else:
                logger.error(f"Failed response with status code: {response.status}")
                return None


async def generate_embeddings(
    texts: list[str], embd_model_config: EMBDModelConfig
) -> np.ndarray | None:
    """
    Generate embeddings using DeepInfra API with correct format.
    Sends one sentence per request to avoid embedding bias.

    Args:
        texts: list of texts to embed

    Returns:
        NumPy array of embeddings or None if failed
    """
    if not embd_model_config.apiKey:
        logger.error("EMBEDDING_MODEL_API_KEY not configured")
        return None

    # Process one sentence per request with concurrency control
    semaphore = asyncio.Semaphore(embd_model_config.maxConcEmbed)

    async def generate_single_embedding(
        text: str, index: int
    ) -> tuple[int, list[float] | None] | None:
        """Generate embedding for a single text"""
        async with semaphore:
            embedding = await query_embd_model(text, embd_model_config)
            if not embedding:
                logger.error(
                    f"[{index + 1}] Failed to generate embedding for text: {text}"
                )
            return (index, embedding)

    # Create concurrent tasks for all texts
    tasks = []
    for i, text in enumerate(texts):
        task = generate_single_embedding(text, i)
        tasks.append(task)

    # Execute all tasks concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Process results and maintain order
    embeddings: list[list[float]] = [[] for _ in range(len(texts))]
    for result in results:
        if isinstance(result, Exception):
            logger.error(f"Task failed with exception: {result}")
            return None
        elif result is not None:
            index, embedding = result  # type:ignore
            embeddings[index] = embedding  # type:ignore
        else:
            logger.error("Failed to generate embedding for one or more texts")
            return None

    # Check if all embeddings were generated
    if any(emb is None for emb in embeddings):
        logger.error("Some embeddings failed to generate")
        return None

    return np.array(embeddings)
