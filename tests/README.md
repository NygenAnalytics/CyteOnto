# CyteOnto Tests

This directory contains comprehensive tests for the CyteOnto package.

## Test Structure

```
tests/
├── __init__.py                     # Test package init
├── conftest.py                     # Pytest fixtures and configuration
├── README.md                       # This file
├── test_config.py                  # Config module tests
├── test_model.py                   # Data model tests
├── test_logger_config.py           # Logger configuration tests
├── test_ontology_extractor.py      # Ontology extraction tests
├── test_ontology_similarity.py     # Ontology similarity tests
├── test_vector_store.py            # Vector storage tests
├── test_cache_manager.py           # Cache management tests
├── test_file_utils.py              # File utilities tests
├── test_pubmed.py                  # PubMed API tests
├── test_matcher.py                 # Matcher functionality tests
├── test_path_config.py             # Path configuration tests
├── test_embeddings.py              # Embedding generation tests
└── test_llm_config.py              # LLM configuration tests
```

## Running Tests

### Run All Tests

```bash
uv run pytest
```

### Run with Coverage

```bash
uv run pytest --cov=cyteonto --cov-report=html
```

### Run Tests by Marker

```bash
# Run only unit tests
uv run pytest -m unit

# Skip slow tests
uv run pytest -m "not slow"
```

## Test Categories

### Unit Tests
- Test individual functions and classes in isolation
- Use mocks for external dependencies
- Fast execution (< 1 second each)

### Integration Tests
- Test interactions between components
- May use real file systems or databases
- Slower execution but more comprehensive

## Test Fixtures

The `conftest.py` file provides common test fixtures:

- `temp_dir`: Temporary directory for file operations
- `sample_cell_description`: Sample CellDescription object
- `sample_ontology_mapping_df`: Sample ontology mapping data
- `sample_embeddings`: Sample embedding arrays
- `mock_base_agent`: Mock pydantic-ai agent
- `sample_embd_model_config`: Sample embedding model configuration

## Writing New Tests

When adding new tests:

1. Follow the naming convention `test_*.py`
2. Use descriptive test method names starting with `test_`
3. Group related tests in classes using `Test*` prefix
4. Use appropriate fixtures from `conftest.py`
5. Mock external dependencies (APIs, file systems, etc.)
6. Include both success and failure cases
7. Test edge cases and error conditions

### Example Test Structure

```python
class TestYourModule:
    """Test YourModule functionality."""

    def test_function_success(self, fixture_name):
        """Test successful function execution."""
        # Arrange
        input_data = "test input"
        
        # Act
        result = your_function(input_data)
        
        # Assert
        assert result == expected_output

    def test_function_failure(self):
        """Test function failure handling."""
        with pytest.raises(ExpectedException):
            your_function(invalid_input)
```
<!-- 
## Mocking Guidelines

- Mock external APIs and services
- Mock file I/O operations when testing logic
- Use `unittest.mock.Mock` and `unittest.mock.patch`
- For async functions, use `unittest.mock.AsyncMock`
- Be explicit about what you're mocking and why

## Test Data

- Use fixtures for reusable test data
- Keep test data small and focused
- Use realistic but simplified data structures
- Avoid hardcoded paths or environment-specific data

## Continuous Integration

These tests are designed to run in CI environments:

- No external API dependencies (all mocked)
- Temporary directories for file operations
- Deterministic results
- Fast execution times

## Coverage Goals

Aim for high test coverage:

- Functions: >90% coverage
- Branches: >85% coverage
- Critical paths: 100% coverage
- Error handling: Well covered -->

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure the package is installed in development mode
2. **Async Test Failures**: Use `pytest-asyncio` and `@pytest.mark.asyncio`
3. **File Permission Errors**: Use the `temp_dir` fixture for file operations
4. **Mock Issues**: Ensure mocks are applied to the correct module path

### Debugging Tests

```bash
# Run with verbose output
uv run pytest -v

# Run single test with output
uv run pytest tests/test_model.py::TestCellDescription::test_creation -s

# Drop into debugger on failure
uv run pytest --pdb
```
