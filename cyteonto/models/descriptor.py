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
        name="CellDescriptorAgent",
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
        Given a user-supplied label for a cell type, produce a structured CellDescription object. You can use PubMed abstracts to help you understand the label.

        # IMPORTANT NOTE
        Provide a detailed descriptive name of the cell type including its function, marker genes, disease relevance and developmental stage.
        
        # INPUT DATA
        {data}
        """)
        .strip()
        .format(
            max_calls_get_pubmed_abstracts=AGENT_CONFIG.MAX_TOOL_CALLS,
            input_data=data,
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

    async def generate_single_description(index: int, label: str) -> CellDescription:
        """Generate description for a single cell type"""
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
            return description

    # Create concurrent tasks for all terms
    tasks = []
    for i, label in enumerate(terms):
        task = generate_single_description(i, label)
        tasks.append(task)

    # Execute all tasks concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Process results and maintain order
    descriptions: list[CellDescription] = [
        CellDescription.get_blank() for _ in range(len(terms))
    ]
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(f"Task {i} failed with exception: {result}")
            # Keep the blank placeholder for this index
        elif isinstance(result, CellDescription):
            descriptions[i] = result
        else:
            logger.error(f"Task {i} returned unexpected result format: {type(result)}")

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
