# cyteonto/__init__.py

from pathlib import Path

from pydantic_ai import Agent

from .llm_config import EMBDModelConfig
from .main import CyteOnto
from .matcher import CyteOntoMatcher
from .setup import CyteOntoSetup

__version__ = "0.1.0"
__all__ = [
    "CyteOnto",
    "CyteOntoMatcher",
    "CyteOntoSetup",
    "EMBDModelConfig",
    "setup",
    "quick_setup",
    "cleanup_cache",
]


async def setup(
    base_agent: Agent,
    embedding_model: str,
    embedding_provider: str,
    base_data_path: str | None = None,
    embeddings_path: str | None = None,
    force_regenerate: bool = False,
) -> bool:
    """
    Setup CyteOnto with ontology embeddings and descriptions.

    Args:
        base_agent: Agent for text generation
        embedding_model: Embedding model name
        embedding_provider: Embedding provider
        base_data_path: Base path for data files
        embeddings_path: Custom path for embeddings file
        force_regenerate: Force regeneration even if files exist

    Returns:
        True if setup successful, False otherwise
    """
    setup_instance = CyteOntoSetup(
        base_data_path=base_data_path or str(Path(__file__).parent / "data"),
        base_agent=base_agent,
        embedding_model=embedding_model,
        embedding_provider=embedding_provider,
    )

    custom_embedding_path = Path(embeddings_path) if embeddings_path else None

    return await setup_instance.setup_embeddings(
        base_agent=base_agent,
        generate_embeddings=True,
        custom_embedding_path=custom_embedding_path,
        force_regenerate=force_regenerate,
    )


async def quick_setup(
    base_agent: Agent,
    embedding_model: str,
    embedding_provider: str,
    base_data_path: str | None = None,
    embeddings_path: str | None = None,
) -> bool:
    """
    Quick setup that only generates embeddings if missing.

    Args:
        base_agent: Agent for text generation
        embedding_model: Embedding model name
        embedding_provider: Embedding provider
        base_data_path: Base path for data files
        embeddings_path: Custom path for embeddings file

    Returns:
        True if setup successful, False otherwise
    """
    return await setup(
        base_agent=base_agent,
        embedding_model=embedding_model,
        embedding_provider=embedding_provider,
        base_data_path=base_data_path,
        embeddings_path=embeddings_path,
        force_regenerate=False,
    )


def cleanup_cache(user_data_path: str | None = None) -> int:
    """
    Clean up invalid cache files in user data directory.

    Args:
        user_data_path: Path to user data directory

    Returns:
        Number of files cleaned up
    """
    from .path_config import PathConfig
    from .storage import CacheManager, VectorStore

    path_config = PathConfig(user_data_path=user_data_path)
    vector_store = VectorStore()
    cache_manager = CacheManager(vector_store)

    user_embeddings_dir = path_config.user_data_path / "embeddings"
    cleaned = 0

    for category_dir in user_embeddings_dir.glob("*"):
        if category_dir.is_dir():
            cleaned += cache_manager.cleanup_invalid_caches(category_dir)

    return cleaned
