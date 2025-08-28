# models/descriptor.py

import asyncio
from textwrap import dedent

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage

from ..llm_config import AGENT_CONFIG, AgentUsage, agent_run
from ..logger_config import logger
from ..model import CellDescription
from .tools.pubmed import get_pubmed_abstracts


def get_descriptor_agent(base_agent: Agent) -> Agent:
    """
    Create a descriptor agent for cell types and add relevant tools.
    Args:
        base_agent: The base agent to modify.
    Returns:
        The created agent.
    """
    agent = Agent(
        base_agent.model,
        deps_type=None,
        output_type=CellDescription,
        name="CellDescriptionAgent",
    )  # type: ignore

    agent.tool_plain(get_pubmed_abstracts, **AGENT_CONFIG.TOOL_DEFAULT_SETTINGS)

    return agent


def get_description_prompt(data: str) -> str:
    prompt = (
        dedent("""
        # INSTRUCTIONS
        You are an expert in disease biology and single-cell genomics specializing in understanding the cellular microenvironment. You are a cell type expert. 

        # TOOL USE
        You can use the get_pubmed_abstracts tool to search for relevant PubMed abstracts.
        DO NOT use the get_pubmed_abstracts tool more than {max_calls_get_pubmed_abstracts} times

        # TASK
        Given a user-supplied label for a cell type, produce a structured CellDescription as JSON. 
        You can optionally use PubMed abstracts to help you understand the label, but if PubMed is unavailable, generate comprehensive descriptions based on your knowledge of cell biology.

        # OUTPUT FORMAT
        Return a JSON object with exactly these fields:
        {{
            "initialLabel": "string - the original cell type label",
            "descriptiveName": "string - a detailed descriptive name",
            "function": "string - the primary function of this cell type",
            "markerGenes": ["array", "of", "marker", "gene", "names"],
            "diseaseRelevance": "string - relevance to diseases",
            "developmentalStage": "string - developmental stage information"
        }}

        # EXAMPLE OUTPUT
        {{
            "initialLabel": "T cell",
            "descriptiveName": "CD4+ helper T lymphocyte",
            "function": "Coordinates immune responses by secreting cytokines and activating other immune cells",
            "markerGenes": ["CD4", "CD3", "TCR", "CD28"],
            "diseaseRelevance": "Critical in autoimmune diseases, immunodeficiency, and cancer immunotherapy",
            "developmentalStage": "Mature adaptive immune cell derived from thymic precursors"
        }}

        # IMPORTANT NOTES
        - Return ONLY the JSON object, no additional text
        - Ensure all fields are filled with relevant information based on your cell biology knowledge
        - If PubMed abstracts are unavailable, use your expertise to provide comprehensive details
        - If information is uncertain, provide reasonable scientific placeholders
        - markerGenes should be an array of strings with known marker genes for this cell type
        
        # INPUT DATA
        {data}
        """)
        .strip()
        .format(
            max_calls_get_pubmed_abstracts=AGENT_CONFIG.MAX_TOOL_CALLS,
            data=data,
        )
    )
    return prompt


async def generate_descriptions(
    base_agent: Agent,
    terms: list[str],
    agent_usage: AgentUsage,
    messages: list[ModelMessage],
) -> list[CellDescription]:
    """
    Generate descriptions using a semaphore and describe_cell_type() function.
    Similar to generate_embeddings but for LLM description generation.

    Args:
        base_agent: The base agent to use for generation
        terms: list of initial labels
        agent_usage: AgentUsage object to track LLM usage
        messages: A list of messages to provide context

    Returns:
        list of CellDescription objects
    """
    if not terms:
        logger.warning("No terms provided for description generation")
        return []

    # Process descriptions with concurrency control
    semaphore = asyncio.Semaphore(AGENT_CONFIG.MAX_CONCURRENT_DESCRIPTIONS)
    completed_count = 0
    total_count = len(terms)
    progress_lock = asyncio.Lock()

    async def generate_single_description(
        index: int, label: str
    ) -> tuple[int, CellDescription]:
        """Generate description for a single cell type"""
        nonlocal completed_count

        async with semaphore:
            description = await describe_cell_type(
                base_agent=base_agent,
                agent_usage=agent_usage,
                initial_label=label,
                messages=messages,
            )
            if not description:
                logger.error(
                    f"[{index + 1}] Failed to generate description for: {label}"
                )
                description = CellDescription.get_blank()  # Fallback

            # Update progress safely
            async with progress_lock:
                completed_count += 1
                logger.info(
                    f"Description progress: {completed_count}/{total_count} completed ({completed_count / total_count * 100:.1f}%) - '{label}'"
                )

            return (index, description)

    # Create concurrent tasks for all terms
    tasks = []
    for i, label in enumerate(terms):
        task = generate_single_description(i, label)
        tasks.append(task)

    logger.info(
        f"Starting generation of {total_count} descriptions with max concurrency: {AGENT_CONFIG.MAX_CONCURRENT_DESCRIPTIONS}"
    )

    # Execute all tasks concurrently with progress tracking
    descriptions: list[CellDescription] = [
        CellDescription.get_blank() for _ in range(len(terms))
    ]

    for coro in asyncio.as_completed(tasks):
        try:
            index, description = await coro
            descriptions[index] = description
        except Exception as e:
            logger.error(f"Description task failed with exception: {e}")

    # Filter out blank placeholders and return only successful ones
    successful_descriptions = [
        desc for desc in descriptions if desc != CellDescription.get_blank()
    ]

    logger.info(
        f"Generated {len(successful_descriptions)} descriptions out of {len(terms)} terms"
    )
    return successful_descriptions


async def describe_cell_type(
    base_agent: Agent,
    agent_usage: AgentUsage,
    initial_label: str,
    messages: list[ModelMessage],
) -> CellDescription:
    """
    A function to describe a cell type using a language model.
    Args:
        agent_factory: A callable function that returns an Agent instance
        agent_usage: AgentUsage object to track LLM usage
        initial_label: The initial label for the cell type
        messages: A list of messages to provide context for the description
    """
    try:
        prompt_data = (
            dedent("""
            # INITIAL LABEL
            {initial_label}
            """)
            .strip()
            .format(initial_label=initial_label)
        )
        description_result = await agent_run(
            agent=get_descriptor_agent(
                base_agent=base_agent,
            ),
            user_prompt=get_description_prompt(prompt_data),
            deps_data=None,
            agent_usage=agent_usage,
            messages=messages,
        )
        return description_result  # type: ignore
    except Exception as e:
        logger.error(f"Description Error: {e}")
        return CellDescription.get_blank()
