# cyteonto/pipeline/description_gen.py

from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage

from ..llm_config import AgentUsage
from ..logger_config import logger
from ..model import CellDescription
from ..models.descriptor import generate_descriptions
from ..storage import VectorStore


class DescriptionGenerator:
    """Generates descriptions for ontology terms using LLM."""

    def __init__(self, vector_store: VectorStore):
        """
        Initialize description generator.

        Args:
            vector_store: VectorStore instance for saving descriptions
        """
        self.vector_store = vector_store

    async def generate_descriptions_for_terms(
        self,
        base_agent: Agent,
        terms: list[str],
        descriptions_file_path: Path | None = None,
        ontology_ids: list[str] | None = None,
    ) -> dict[str, CellDescription]:
        """
        Generate descriptions for multiple ontology terms using concurrent processing.

        Args:
            base_agent: Base agent for LLM calls
            terms: list of labels
            descriptions_file_path: Path to save descriptions
            ontology_ids: list of ontology ids. Optional, only used if generating descriptions for ontology terms.

        Returns:
            Dictionary mapping ontology_id to description
        """
        logger.info(f"Starting description generation for {len(terms)} terms")

        # Try to load existing descriptions first
        if descriptions_file_path:
            existing_descriptions = self.vector_store.load_descriptions(
                descriptions_file_path
            )
        else:
            existing_descriptions = None

        if existing_descriptions:
            logger.info(
                f"Loaded {len(existing_descriptions)} existing descriptions. No need to generate descriptions."
            )
            return existing_descriptions

        if ontology_ids is None:
            ontology_ids = [f"term:{idx}" for idx in range(len(terms))]

        logger.info(f"Generating descriptions for {len(terms)} terms")

        # Use concurrent processing with semaphore (similar to embedding generation)
        agent_usage = AgentUsage(agent_name="DescriptionGenerator")
        messages: list[ModelMessage] = []

        # Generate all descriptions concurrently using semaphore
        try:
            description_results = await generate_descriptions(
                base_agent=base_agent,
                terms=terms,
                agent_usage=agent_usage,
                messages=messages,
            )

            # Process results and convert to string format
            all_descriptions: dict[str, CellDescription] = {}
            for i, cell_description in enumerate(description_results):
                all_descriptions[ontology_ids[i]] = cell_description

            # Save all descriptions
            success = self.vector_store.save_descriptions(
                all_descriptions,
                descriptions_file_path,  # type: ignore
            )
            if not success:
                logger.error("Failed to save descriptions")

        except Exception as e:
            logger.error(f"Failed to generate descriptions concurrently: {e}")
            all_descriptions = {}

        logger.info(f"Generated descriptions for {len(all_descriptions)} total terms")
        logger.info(
            f"Agent usage: {agent_usage.requests} requests, {agent_usage.total_tokens} tokens"
        )

        return all_descriptions
