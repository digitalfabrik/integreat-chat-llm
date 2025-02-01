"""
A service to search for documents
"""
from django.conf import settings

from .opensearch import OpenSearch
from ..utils.search_request import SearchRequest
from ..utils.search_response import SearchResponse, Document

class SearchService:
    """
    Service class that enables searching for Integreat content
    """
    def __init__(self, search_request: SearchRequest, deduplicate_results: bool) -> None:
        self.search_request = search_request
        self.language = search_request.use_language
        self.original_language = search_request.gui_language
        self.region = search_request.region
        self.os = OpenSearch(password=settings.OPENSEARCH_PASSWORD)
        self.deduplicate_results = deduplicate_results

    def search_documents(
            self,
            max_results: int = settings.SEARCH_MAX_DOCUMENTS,
            include_text: bool = False
        ) -> list:
        """
        Create summary answer for question
        """
        results = self.os.reduce_search_result(
            response = self.os.search(
                self.region,
                self.language,
                self.search_request.translated_message
            ),
            deduplicate = self.deduplicate_results,
            max_results = max_results
        )
        documents = []
        for result in results:
            documents.append(Document(
                result["url"],
                result["chunk_text"],
                result["score"],
                include_text,
                self.search_request.gui_language
            ))
        return SearchResponse(self.search_request, documents)
