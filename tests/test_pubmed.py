from unittest.mock import Mock, patch

import requests

from cyteonto.models.tools.pubmed import get_pubmed_abstracts


class TestPubMedTools:
    """Test PubMed API interaction tools."""

    @patch("requests.get")
    def test_get_pubmed_abstracts_success(self, mock_get, mock_pubmed_response):
        """Test successful PubMed abstract retrieval."""
        # Mock search response
        search_response = Mock()
        search_response.status_code = 200
        search_response.json.return_value = mock_pubmed_response["search_response"]

        # Mock fetch response
        fetch_response = Mock()
        fetch_response.status_code = 200
        fetch_response.text = mock_pubmed_response["fetch_response"]

        # Configure mock to return different responses for different calls
        mock_get.side_effect = [search_response, fetch_response]

        result = get_pubmed_abstracts("T cell function", max_results=2)

        assert isinstance(result, list)
        assert len(result) == 1
        assert "This is a test abstract about T cells." in result[0]

        # Verify API calls
        assert mock_get.call_count == 2

    @patch("requests.get")
    def test_get_pubmed_abstracts_no_results(self, mock_get):
        """Test PubMed query with no results."""
        # Mock search response with empty results
        search_response = Mock()
        search_response.status_code = 200
        search_response.json.return_value = {"esearchresult": {"idlist": []}}

        mock_get.return_value = search_response

        result = get_pubmed_abstracts("nonexistent query")

        assert result == []
        assert mock_get.call_count == 1

    @patch("requests.get")
    def test_get_pubmed_abstracts_search_failure(self, mock_get):
        """Test PubMed query with search API failure."""
        mock_get.side_effect = requests.exceptions.RequestException("API Error")

        result = get_pubmed_abstracts("T cell")

        assert result == []

    @patch("requests.get")
    def test_get_pubmed_abstracts_search_timeout(self, mock_get):
        """Test PubMed query with timeout."""
        mock_get.side_effect = requests.exceptions.Timeout("Timeout")

        result = get_pubmed_abstracts("T cell")

        assert result == []

    @patch("requests.get")
    def test_get_pubmed_abstracts_http_error(self, mock_get):
        """Test PubMed query with HTTP error status."""
        error_response = Mock()
        error_response.status_code = 500
        error_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            "Server Error"
        )

        mock_get.return_value = error_response

        result = get_pubmed_abstracts("T cell")

        assert result == []

    @patch("requests.get")
    def test_get_pubmed_abstracts_invalid_json(self, mock_get):
        """Test PubMed query with invalid JSON response."""
        search_response = Mock()
        search_response.status_code = 200
        search_response.json.side_effect = ValueError("Invalid JSON")

        mock_get.return_value = search_response

        result = get_pubmed_abstracts("T cell")

        assert result == []

    @patch("requests.get")
    def test_get_pubmed_abstracts_fetch_failure(self, mock_get):
        """Test PubMed query with fetch API failure."""
        # Mock successful search
        search_response = Mock()
        search_response.status_code = 200
        search_response.json.return_value = {"esearchresult": {"idlist": ["12345"]}}

        # Mock failed fetch
        fetch_response = Mock()
        fetch_response.side_effect = requests.exceptions.RequestException(
            "Fetch failed"
        )

        mock_get.side_effect = [search_response, fetch_response]

        result = get_pubmed_abstracts("T cell")

        assert result == []

    @patch("requests.get")
    def test_get_pubmed_abstracts_invalid_xml(self, mock_get):
        """Test PubMed query with invalid XML response."""
        # Mock successful search
        search_response = Mock()
        search_response.status_code = 200
        search_response.json.return_value = {"esearchresult": {"idlist": ["12345"]}}

        # Mock fetch with invalid XML
        fetch_response = Mock()
        fetch_response.status_code = 200
        fetch_response.text = "invalid xml content <unclosed"

        mock_get.side_effect = [search_response, fetch_response]

        result = get_pubmed_abstracts("T cell")

        assert result == []

    @patch("requests.get")
    def test_get_pubmed_abstracts_empty_abstracts(self, mock_get):
        """Test PubMed query with empty abstracts in XML."""
        # Mock successful search
        search_response = Mock()
        search_response.status_code = 200
        search_response.json.return_value = {"esearchresult": {"idlist": ["12345"]}}

        # Mock fetch with XML containing no abstracts
        xml_content = """<?xml version="1.0"?>
        <PubmedArticleSet>
            <PubmedArticle>
                <MedlineCitation>
                    <Article>
                        <Title>Test Article</Title>
                    </Article>
                </MedlineCitation>
            </PubmedArticle>
        </PubmedArticleSet>"""

        fetch_response = Mock()
        fetch_response.status_code = 200
        fetch_response.text = xml_content

        mock_get.side_effect = [search_response, fetch_response]

        result = get_pubmed_abstracts("T cell")

        assert result == []

    @patch("requests.get")
    def test_get_pubmed_abstracts_multiple_abstracts(self, mock_get):
        """Test PubMed query with multiple abstracts."""
        # Mock successful search
        search_response = Mock()
        search_response.status_code = 200
        search_response.json.return_value = {
            "esearchresult": {"idlist": ["12345", "67890"]}
        }

        # Mock fetch with multiple abstracts
        xml_content = """<?xml version="1.0"?>
        <PubmedArticleSet>
            <PubmedArticle>
                <MedlineCitation>
                    <Article>
                        <Abstract>
                            <AbstractText>First abstract about T cells.</AbstractText>
                        </Abstract>
                    </Article>
                </MedlineCitation>
            </PubmedArticle>
            <PubmedArticle>
                <MedlineCitation>
                    <Article>
                        <Abstract>
                            <AbstractText>Second abstract about B cells.</AbstractText>
                        </Abstract>
                    </Article>
                </MedlineCitation>
            </PubmedArticle>
        </PubmedArticleSet>"""

        fetch_response = Mock()
        fetch_response.status_code = 200
        fetch_response.text = xml_content

        mock_get.side_effect = [search_response, fetch_response]

        result = get_pubmed_abstracts("immune cells", max_results=5)

        assert len(result) == 2
        assert "First abstract about T cells." in result[0]
        assert "Second abstract about B cells." in result[1]

    @patch("requests.get")
    def test_get_pubmed_abstracts_with_api_key(self, mock_get, mock_env_vars):
        """Test PubMed query includes API key from config."""
        # Mock successful search
        search_response = Mock()
        search_response.status_code = 200
        search_response.json.return_value = {"esearchresult": {"idlist": ["12345"]}}

        # Mock successful fetch
        fetch_response = Mock()
        fetch_response.status_code = 200
        fetch_response.text = """<?xml version="1.0"?>
        <PubmedArticleSet>
            <PubmedArticle>
                <MedlineCitation>
                    <Article>
                        <Abstract>
                            <AbstractText>Test abstract.</AbstractText>
                        </Abstract>
                    </Article>
                </MedlineCitation>
            </PubmedArticle>
        </PubmedArticleSet>"""

        mock_get.side_effect = [search_response, fetch_response]

        _ = get_pubmed_abstracts("T cell")

        # Check that API key was included in requests
        search_call_args = mock_get.call_args_list[0]
        search_params = search_call_args[1]["params"]
        assert "api_key" in search_params

        fetch_call_args = mock_get.call_args_list[1]
        fetch_params = fetch_call_args[1]["params"]
        assert "api_key" in fetch_params

    @patch("requests.get")
    def test_get_pubmed_abstracts_default_parameters(self, mock_get):
        """Test PubMed query with default parameters."""
        # Mock successful search
        search_response = Mock()
        search_response.status_code = 200
        search_response.json.return_value = {
            "esearchresult": {"idlist": ["1", "2", "3", "4", "5"]}
        }

        # Mock successful fetch
        fetch_response = Mock()
        fetch_response.status_code = 200
        fetch_response.text = """<?xml version="1.0"?>
        <PubmedArticleSet></PubmedArticleSet>"""

        mock_get.side_effect = [search_response, fetch_response]

        _ = get_pubmed_abstracts("T cell")  # Default max_results=5

        # Check that default max_results was used
        search_call_args = mock_get.call_args_list[0]
        search_params = search_call_args[1]["params"]
        assert search_params["retmax"] == "5"

    @patch("requests.get")
    def test_get_pubmed_abstracts_custom_max_results(self, mock_get):
        """Test PubMed query with custom max_results."""
        # Mock successful search
        search_response = Mock()
        search_response.status_code = 200
        search_response.json.return_value = {
            "esearchresult": {"idlist": ["1", "2", "3"]}
        }

        # Mock successful fetch
        fetch_response = Mock()
        fetch_response.status_code = 200
        fetch_response.text = """<?xml version="1.0"?>
        <PubmedArticleSet></PubmedArticleSet>"""

        mock_get.side_effect = [search_response, fetch_response]

        _ = get_pubmed_abstracts("T cell", max_results=3)

        # Check that custom max_results was used
        search_call_args = mock_get.call_args_list[0]
        search_params = search_call_args[1]["params"]
        assert search_params["retmax"] == "3"

    def test_get_pubmed_abstracts_connection_error(self):
        """Test PubMed query with connection error."""
        with patch(
            "requests.get",
            side_effect=requests.exceptions.ConnectionError("No internet"),
        ):
            result = get_pubmed_abstracts("T cell")
            assert result == []

    @patch("requests.get")
    def test_get_pubmed_abstracts_missing_idlist_key(self, mock_get):
        """Test PubMed query with missing idlist key in response."""
        # Mock search response without idlist
        search_response = Mock()
        search_response.status_code = 200
        search_response.json.return_value = {
            "esearchresult": {}  # Missing idlist
        }

        mock_get.return_value = search_response

        result = get_pubmed_abstracts("T cell")

        assert result == []
