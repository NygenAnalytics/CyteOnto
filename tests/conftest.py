import os
import tempfile
from pathlib import Path
from unittest.mock import Mock

import pandas as pd  # type: ignore
import pytest
from pydantic_ai import Agent

from cyteonto.models import CellDescription


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
def mock_base_agent():
    """Create a mock base agent for testing."""
    agent = Mock(spec=Agent)
    agent.model = Mock()
    agent.model.model_name = "test-model"
    return agent


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
