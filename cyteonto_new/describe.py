"""LLM-driven description generation for cell types."""

import asyncio
from textwrap import dedent
from typing import Any
from xml.etree import ElementTree as ET

import requests
from pydantic_ai import (
    Agent,
    ModelHTTPError,
    UsageLimitExceeded,
)
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.usage import UsageLimits
from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .config import Config
from .logger import logger
from .models import AgentUsage, CellDescription

config = Config()


def _format_exception(exc: BaseException) -> str:
    """Return a short, useful string for an exception.

    Unwraps ``tenacity.RetryError`` to the underlying failure, and surfaces
    HTTP status codes for ``ModelHTTPError``.
    """
    root = exc
    if isinstance(root, RetryError) and root.last_attempt is not None:
        inner = root.last_attempt.exception()
        if inner is not None:
            root = inner
    if isinstance(root, ModelHTTPError):
        return f"ModelHTTPError(status={root.status_code}): {root.message}"
    return f"{type(root).__name__}: {root}"


def _label_from_prompt(prompt: str) -> str:
    """Extract the input label from a descriptor prompt for logging."""
    marker = "Input label:\n"
    idx = prompt.find(marker)
    if idx == -1:
        return "?"
    start = idx + len(marker)
    end = prompt.find("\n\n", start)
    raw = prompt[start:end] if end != -1 else prompt[start : start + 80]
    return raw.strip() or "?"


def _log_before_sleep(retry_state: Any) -> None:
    """Tenacity ``before_sleep`` callback that logs each failed attempt."""
    prompt = ""
    if len(retry_state.args) >= 2:
        prompt = retry_state.args[1]
    elif "prompt" in retry_state.kwargs:
        prompt = retry_state.kwargs["prompt"]
    label = _label_from_prompt(prompt)

    exc: BaseException | None = None
    if retry_state.outcome is not None:
        exc = retry_state.outcome.exception()
    sleep_s = (
        retry_state.next_action.sleep if retry_state.next_action is not None else 0.0
    )
    attempt = retry_state.attempt_number
    logger.warning(
        f"[{label}] attempt {attempt}/{config.RETRY_ATTEMPTS} failed: "
        f"{_format_exception(exc) if exc else 'unknown error'}. "
        f"Retrying in {sleep_s:.1f}s"
    )


def get_pubmed_abstracts(query: str, max_results: int = 5) -> list[str]:
    """Fetch up to ``max_results`` PubMed abstracts for ``query``.

    Returns an empty list on any failure so the LLM can still proceed.
    """
    try:
        search = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            params={
                "db": "pubmed",
                "term": query,
                "retmax": str(max_results),
                "retmode": "json",
                "api_key": config.NCBI_API_KEY,
            },
            timeout=10,
        )
        search.raise_for_status()
        ids = search.json().get("esearchresult", {}).get("idlist", [])
        if not ids:
            return []

        fetch = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
            params={
                "db": "pubmed",
                "id": ",".join(ids),
                "retmode": "xml",
                "api_key": config.NCBI_API_KEY,
            },
            timeout=15,
        )
        fetch.raise_for_status()
        root = ET.fromstring(fetch.text)
        return [
            str(el.text)
            for el in root.findall(".//Abstract/AbstractText")
            if el is not None and el.text
        ]
    except Exception as e:
        logger.error(f"PubMed unavailable ({type(e).__name__}: {e})")
        return []


def _build_prompt(label: str, use_pubmed: bool) -> str:
    tool_block = (
        f"You may call `get_pubmed_abstracts` at most {config.MAX_PUBMED_CALLS} times "
        "to retrieve literature evidence. If the tool returns an empty list, "
        "continue without it."
        if use_pubmed
        else "Do not call any tools."
    )
    return dedent(
        f"""
        Task: produce a structured description of one cell type.

        Input label:
        {label}

        The input label may be a single name, an abbreviation, or multiple
        synonyms separated by semicolons. Treat them as referring to the same
        cell type. If the label is ambiguous, choose the most likely
        interpretation in a standard mammalian reference context and commit
        to it throughout the response.

        {tool_block}

        Fill every field below. Write each field in third person, factual,
        present tense. Do not add text outside the structured object. Do not
        repeat the label text verbatim inside the other fields unless it is
        the only correct wording.

        initialLabel
            Copy the input label verbatim, including punctuation and
            semicolons. Do not normalise case, expand abbreviations, or
            strip synonyms.

        descriptiveName
            One noun phrase, at most 12 words. A precise name that identifies
            the cell type without relying on the initial label. Prefer lineage
            and location when available (for example "tissue-resident alveolar
            macrophage").

        function
            One to three sentences, at most 60 words total. State the primary
            biological role of the cell. Mention key effector mechanisms only
            when they are defining for this cell type.

        diseaseRelevance
            One to three sentences, at most 60 words total. State conditions
            in which this cell type is pathologically involved, depleted,
            expanded, or is a therapeutic target. If the cell type has no
            well-established disease relevance, write "Not established".

        developmentalStage
            One sentence, at most 30 words. State the lineage, progenitor, or
            developmental window the cell belongs to. Use "Terminally
            differentiated" when applicable. Use "Not established" when the
            stage is unknown.

        Do not include marker genes, tissue lists, species ranges, or
        citations inside any field. Do not hedge with phrases such as
        "it is believed that" or "some studies suggest". State facts directly.
        """
    ).strip()


def _build_descriptor_agent(
    base_agent: Agent, use_pubmed: bool, reasoning: bool = False
) -> Agent:
    """Build a descriptor agent from the base agent's model.

    When ``reasoning=False`` we disable thinking for the underlying provider.
    """
    model_settings: dict[str, Any] = {}
    if not reasoning:
        model_settings = {
            "thinking": False,
            "extra_body": {
                "chat_template_kwargs": {"thinking": False},  # for together-ai
                "reasoning": {"enabled": False},  # for openrouter
            },
        }

    agent: Agent = Agent(
        base_agent.model,
        output_type=CellDescription,  # type:ignore
        name="CellDescriptionAgent",
        model_settings=model_settings,
        system_prompt="You are an expert in single-cell genomics specializing in understanding the cellular identity.",
    )
    if use_pubmed:
        agent.tool_plain(get_pubmed_abstracts, retries=2)  # type:ignore
    return agent


@retry(
    stop=stop_after_attempt(config.RETRY_ATTEMPTS),
    wait=wait_exponential(min=config.RETRY_WAIT_MIN, max=config.RETRY_WAIT_MAX),
    retry=retry_if_exception_type(config._RETRYABLE_EXCEPTIONS),
    before_sleep=_log_before_sleep,
    reraise=False,
)
async def _run_once(
    agent: Agent,
    prompt: str,
    usage_limits: UsageLimits,
) -> tuple[CellDescription, dict[str, int], int, int, int, int]:
    result = await asyncio.wait_for(
        agent.run(prompt, usage_limits=usage_limits),
        timeout=config.PER_ATTEMPT_TIMEOUT,
    )
    tool_counts: dict[str, int] = {}
    for msg in result.all_messages():
        for part in getattr(msg, "parts", []):
            if isinstance(part, ToolCallPart):
                tool_counts[part.tool_name] = tool_counts.get(part.tool_name, 0) + 1
    usage = result.usage()
    return (  # type: ignore[return-value]
        result.output,
        tool_counts,
        usage.requests,
        usage.input_tokens or 0,
        usage.output_tokens or 0,
        usage.total_tokens or 0,
    )


async def describe_cell(
    base_agent: Agent,
    label: str,
    use_pubmed: bool = True,
    usage_limits: UsageLimits = config.DEFAULT_USAGE_LIMITS,
    reasoning: bool = False,
) -> tuple[CellDescription, AgentUsage]:
    """Generate a ``CellDescription`` for a single label and return the usage tally."""
    agent = _build_descriptor_agent(base_agent, use_pubmed, reasoning)
    usage = AgentUsage(agentName=agent.name or "CellDescriptionAgent")
    try:
        output, tool_counts, req, in_tok, out_tok, total_tok = await _run_once(
            agent, _build_prompt(label, use_pubmed), usage_limits
        )
    except UsageLimitExceeded as e:
        logger.error(f"Usage limit exceeded while describing '{label}': {e}")
        return CellDescription.blank(label), usage
    except Exception as e:
        logger.error(f"Description failed for '{label}': {_format_exception(e)}")
        return CellDescription.blank(label), usage

    usage.modelName = str(agent.model.model_name)  # type: ignore
    usage.requests = req
    usage.inputTokens = in_tok
    usage.outputTokens = out_tok
    usage.totalTokens = total_tok
    usage.toolUsage = tool_counts
    return output or CellDescription.blank(label), usage


async def describe_cells(
    base_agent: Agent,
    labels: list[str],
    use_pubmed: bool = True,
    max_concurrent: int = config.MAX_CONCURRENT_DESCRIPTIONS,
    reasoning: bool = False,
    second_pass_wait_seconds: float = 5.0,
) -> tuple[list[CellDescription], AgentUsage]:
    """Generate descriptions for many labels concurrently.

    After the first pass completes, any labels whose description came back
    blank are re-attempted once more after a short sleep. This absorbs
    transient provider outages without requiring the caller to re-invoke.

    The final result always has one ``CellDescription`` per input label in the
    original order. When a label cannot be described even after the second
    pass, the slot holds ``CellDescription.blank(label)``; upstream caching
    should skip persisting blanks so they are retried on the next run.
    """
    if not labels:
        return [], AgentUsage(agentName="CellDescriptionAgent")

    total = len(labels)
    sem = asyncio.Semaphore(max_concurrent)
    total_usage = AgentUsage(agentName="CellDescriptionAgent")
    results: list[CellDescription] = [CellDescription.blank(lbl) for lbl in labels]

    async def run_batch(indices: list[int], pass_label: str) -> None:
        done = 0
        done_lock = asyncio.Lock()
        batch_size = len(indices)

        async def run_one(i: int) -> tuple[int, CellDescription, AgentUsage]:
            nonlocal done
            lbl = labels[i]
            async with sem:
                desc, usage = await describe_cell(
                    base_agent, lbl, use_pubmed=use_pubmed, reasoning=reasoning
                )
            async with done_lock:
                done += 1
                if done == batch_size or done % max(1, batch_size // 20) == 0:
                    logger.info(
                        f"Description progress ({pass_label}): {done}/{batch_size}"
                    )
            return i, desc, usage

        tasks = [run_one(i) for i in indices]
        for coro in asyncio.as_completed(tasks):
            try:
                i, desc, usage = await coro
                results[i] = desc
                total_usage.merge(usage)
            except Exception as e:
                logger.error(f"Description task crashed: {_format_exception(e)}")

    logger.info(f"Describing {total} cell labels (pass 1)")
    await run_batch(list(range(total)), "pass 1")

    blank_indices = [i for i, d in enumerate(results) if d.is_blank()]
    if blank_indices:
        logger.warning(
            f"Pass 1 left {len(blank_indices)}/{total} labels blank. "
            f"Sleeping {second_pass_wait_seconds:.0f}s before a second pass."
        )
        await asyncio.sleep(second_pass_wait_seconds)
        await run_batch(blank_indices, "pass 2")

    final_blanks = [labels[i] for i, d in enumerate(results) if d.is_blank()]
    if final_blanks:
        preview = ", ".join(repr(lbl) for lbl in final_blanks[:10])
        suffix = f" (+{len(final_blanks) - 10} more)" if len(final_blanks) > 10 else ""
        logger.warning(
            f"describe_cells summary: {total - len(final_blanks)}/{total} ok, "
            f"{len(final_blanks)} blank: [{preview}{suffix}]. "
            "Upstream caching should skip these so they are retried next run."
        )
    else:
        logger.info(f"describe_cells summary: {total}/{total} ok")

    return results, total_usage
