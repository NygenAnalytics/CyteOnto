# cyteonto/tools/pubmed.py

from xml.etree import ElementTree as ET

import requests

from ...config import CONFIG


def get_pubmed_abstracts(query: str, max_results: int = 5) -> list[str]:
    """
    Fetches abstracts from PubMed based on a search query.
    Args:
        query (str): The search query.
        max_results (int): The maximum number of results to return.
    Returns:
        list[str]: A list of abstracts from the search results.
    """

    SEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    search_params: dict[str, str] = {
        "db": "pubmed",
        "term": query,
        "retmax": str(max_results),
        "retmode": "json",
        "api_key": CONFIG.NCBI_API_KEY,
    }
    ids = requests.get(
        SEARCH_URL,
        params=search_params,
    ).json()["esearchresult"]["idlist"]

    if not ids:
        return []
    FETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    fetch_params: dict[str, str] = {
        "db": "pubmed",
        "id": ",".join(ids),
        "retmode": "json",
        "api_key": CONFIG.NCBI_API_KEY,
    }
    xml = requests.get(
        FETCH_URL,
        params=fetch_params,
    ).text
    root = ET.fromstring(xml)
    return [
        str(el.text)
        for el in root.findall(".//Abstract/AbstractText")
        if el is not None
    ]
