import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import numpy as np
import pandas as pd  # type: ignore
import pytest
from pydantic_ai import Agent

from cyteonto.model import CellDescription


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield Path(tmpdirname)


@pytest.fixture
def sample_cell_description():
    """Create a sample CellDescription for testing."""
    return CellDescription(
        initialLabel="T cell",
        descriptiveName="CD4+ helper T lymphocyte",
        function="Coordinates immune responses by secreting cytokines",
        markerGenes=["CD4", "CD3", "TCR"],
        diseaseRelevance="Critical in autoimmune diseases",
        developmentalStage="Mature adaptive immune cell",
    )


@pytest.fixture
def sample_ontology_mapping_df():
    """Create a sample ontology mapping DataFrame for testing."""
    data = {
        "ontology_id": ["CL:0000001", "CL:0000002", "CL:0000003"],
        "label": ["T cell", "B cell", "NK cell"],
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_ontology_csv_file(temp_dir, sample_ontology_mapping_df):
    """Create a sample ontology CSV file for testing."""
    csv_path = temp_dir / "cell_to_cell_ontology.csv"
    sample_ontology_mapping_df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def sample_embeddings():
    """Create sample embeddings for testing."""
    return np.random.rand(5, 768).astype(np.float32)


@pytest.fixture
def sample_ontology_ids():
    """Create sample ontology IDs for testing."""
    return ["CL:0000001", "CL:0000002", "CL:0000003", "CL:0000004", "CL:0000005"]


@pytest.fixture
def mock_base_agent():
    """Create a mock base agent for testing."""
    agent = Mock(spec=Agent)
    agent.model = Mock()
    agent.model.model_name = "test-model"
    return agent


@pytest.fixture
def mock_async_base_agent():
    """Create a mock async base agent for testing."""
    agent = AsyncMock(spec=Agent)
    agent.model = Mock()
    agent.model.model_name = "test-model"
    return agent


@pytest.fixture
def sample_embd_model_config():
    """Create a sample embedding model configuration for testing."""
    from cyteonto.llm_config import EMBDModelConfig

    return EMBDModelConfig(
        provider="deepinfra",
        model="test-embedding-model",
        apiKey="test-api-key",
        maxConcEmbed=5,
    )


@pytest.fixture
def sample_user_labels():
    """Create sample user labels for testing."""
    return [
        "T helper cell",
        "B memory cell",
        "Natural killer cell",
        "Monocyte",
        "Macrophage",
    ]


@pytest.fixture
def sample_descriptions():
    """Create sample descriptions for testing."""
    descriptions = {}
    for i, label in enumerate(
        ["T cell", "B cell", "NK cell", "Monocyte", "Macrophage"]
    ):
        descriptions[f"CL:000000{i + 1}"] = CellDescription(
            initialLabel=label,
            descriptiveName=f"Descriptive {label}",
            function=f"Function of {label}",
            markerGenes=[f"MARKER{i + 1}A", f"MARKER{i + 1}B"],
            diseaseRelevance=f"Disease relevance for {label}",
            developmentalStage=f"Developmental stage for {label}",
        )
    return descriptions


@pytest.fixture
def mock_env_vars():
    """Mock environment variables for testing."""
    env_vars = {
        "NCBI_API_KEY": "test_ncbi_key",
        "EMBEDDING_MODEL_API_KEY": "test_embedding_key",
    }
    for key, value in env_vars.items():
        os.environ[key] = value

    yield env_vars

    # Cleanup
    for key in env_vars.keys():
        os.environ.pop(key, None)


@pytest.fixture
def mock_pubmed_response():
    """Mock PubMed API response for testing."""
    return {
        "search_response": {"esearchresult": {"idlist": ["12345", "67890"]}},
        "fetch_response": """<?xml version="1.0"?>
        <PubmedArticleSet>
            <PubmedArticle>
                <MedlineCitation>
                    <Article>
                        <Abstract>
                            <AbstractText>This is a test abstract about T cells.</AbstractText>
                        </Abstract>
                    </Article>
                </MedlineCitation>
            </PubmedArticle>
        </PubmedArticleSet>""",
    }


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset any singleton instances between tests."""
    # Add any singleton cleanup here if needed
    yield
