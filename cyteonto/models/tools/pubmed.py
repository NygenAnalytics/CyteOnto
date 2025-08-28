# cyteonto/models/tools/pubmed.py

from xml.etree import ElementTree as ET

import requests

from ...config import CONFIG
from ...logger_config import logger


def get_pubmed_abstracts(query: str, max_results: int = 5) -> list[str]:
    """
    Fetches abstracts from PubMed based on a search query.
    Returns empty list if there are network or API issues.

    Args:
        query (str): The search query.
        max_results (int): The maximum number of results to return.
    Returns:
        list[str]: A list of abstracts from the search results, or empty list if failed.
    """
    try:
        SEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        search_params: dict[str, str] = {
            "db": "pubmed",
            "term": query,
            "retmax": str(max_results),
            "retmode": "json",
            "api_key": CONFIG.NCBI_API_KEY,
        }

        # Add timeout and error handling for search request
        search_response = requests.get(
            SEARCH_URL,
            params=search_params,
            timeout=10,  # 10 second timeout
        )
        search_response.raise_for_status()

        search_data = search_response.json()
        if not search_data:
            return []
        else:
            ids = search_data["esearchresult"]["idlist"]

        if not ids:
            return []

        FETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        fetch_params: dict[str, str] = {
            "db": "pubmed",
            "id": ",".join(ids),
            "retmode": "xml",  # Use XML mode for reliable parsing
            "api_key": CONFIG.NCBI_API_KEY,
        }

        # Add timeout and error handling for fetch request
        fetch_response = requests.get(
            FETCH_URL,
            params=fetch_params,
            timeout=15,  # 15 second timeout
        )
        fetch_response.raise_for_status()

        xml = fetch_response.text
        root = ET.fromstring(xml)

        abstracts = []
        for el in root.findall(".//Abstract/AbstractText"):
            if el is not None and el.text:
                abstracts.append(str(el.text))

        return abstracts

    except (
        requests.exceptions.RequestException,
        requests.exceptions.Timeout,
        requests.exceptions.ConnectionError,
        ET.ParseError,
        KeyError,
        Exception,
    ) as e:
        # Log the error but don't raise it - return empty list instead
        logger.error(
            f"PubMed API unavailable ({type(e).__name__}: {str(e)}). Continuing without PubMed abstracts."
        )
        return []
