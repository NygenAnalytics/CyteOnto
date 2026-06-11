# CyteOnto Tests

> Note: These tests are AI generated.

Unit tests for the `cyteonto` package. They run offline; all network and LLM
calls are mocked.

## Layout

```
tests/
├── conftest.py            # Shared fixtures
├── test_config.py         # Config defaults
├── test_models.py         # Pydantic models (descriptions, configs, usage)
├── test_path_config.py    # PathConfig path resolution
├── test_artifact_keys.py  # Artifact keys, paths, storage round trips
├── test_ontology.py       # OntologyMapping + OntologySimilarity
├── test_embed.py          # Embedding HTTP helpers and orchestration
├── test_describe.py       # Prompt building and error formatting
├── test_pubmed.py         # PubMed abstract retrieval
└── test_cyteonto.py       # Orchestrator pure units (matching, key resolution)
```

## Running

```bash
# All tests
uv run pytest

# Verbose
uv run pytest -v

# Single test
uv run pytest tests/test_models.py::TestCellDescription::test_to_sentence

# Drop into debugger on failure
uv run pytest --pdb
```

`asyncio_mode = "auto"` is set in `pyproject.toml`, so async tests run without
explicit markers.

## Fixtures

Defined in `conftest.py`:

- `temp_dir`: temporary directory
- `sample_cell_description`: populated `CellDescription`
- `sample_ontology_mapping_df` / `sample_ontology_csv_file`: ontology mapping data
- `mock_base_agent`: mock pydantic-ai agent
- `mock_env_vars`: temporary `NCBI_API_KEY` / `EMBEDDING_MODEL_API_KEY`
- `mock_pubmed_response`: canned PubMed search and fetch payloads
