import pytest
from fastapi.testclient import TestClient

from cyteonto import __version__
from modal_app.api import create_app


@pytest.fixture
def client():
    with TestClient(create_app(object(), object())) as test_client:
        yield test_client


def test_swagger_docs_are_public(client: TestClient):
    response = client.get("/docs")

    assert response.status_code == 200
    assert "Swagger UI" in response.text
    assert "/openapi.json" in response.text


def test_landing_page_links_to_docs_and_nygen(client: TestClient):
    response = client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert 'href="/docs"' in response.text
    assert 'href="https://nygen.io/"' in response.text
    assert "family=Inter:wght@300;400" in response.text
    assert "#0077fc" in response.text


def test_openapi_schema_describes_the_http_contract(client: TestClient):
    response = client.get("/openapi.json")

    assert response.status_code == 200
    schema = response.json()

    assert schema["info"]["title"] == "CyteOnto API"
    assert schema["info"]["version"] == __version__
    assert schema["info"]["summary"]
    assert set(schema["paths"]) == {
        "/health",
        "/compare",
        "/status/{run_id}",
        "/result/{run_id}",
    }

    for path, method in (
        ("/health", "get"),
        ("/compare", "post"),
        ("/status/{run_id}", "get"),
        ("/result/{run_id}", "get"),
    ):
        operation = schema["paths"][path][method]
        assert operation["summary"]
        assert operation["description"]
        assert operation["tags"]

    request_schema = schema["components"]["schemas"]["CompareRequest"]
    assert request_schema["examples"]
    assert request_schema["properties"]["authorLabels"]["description"]
    assert request_schema["properties"]["llmApiKey"]["writeOnly"] is True
    assert request_schema["properties"]["embeddingApiKey"]["writeOnly"] is True

    result_schema = schema["components"]["schemas"]["ResultRow"]
    assert "run_id" in result_schema["properties"]
    assert "pair_index" in result_schema["properties"]
    assert "runId" not in result_schema["properties"]
    assert "pairIndex" not in result_schema["properties"]

    result_content = schema["paths"]["/result/{run_id}"]["get"]["responses"]["200"][
        "content"
    ]
    assert result_content["application/json"]["schema"]["type"] == "array"
    assert result_content["application/json"]["schema"]["items"] == {
        "$ref": "#/components/schemas/ResultRow"
    }
    assert result_content["text/csv"]["schema"] == {
        "type": "string",
        "format": "binary",
    }
