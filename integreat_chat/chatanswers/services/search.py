"""
A service to search for documents
"""

import urllib.request
import urllib.parse
import json

from sentence_transformers import SentenceTransformer
from pymilvus import (
    connections,
    Collection,
)

from django.conf import settings

class SearchService:
    """
    Service class that enables searching for Integreat content
    """
    def __init__(self, region: str, language: str) -> None:
        self.language = language
        self.region = region
        self.vdb_host = settings.VDB_HOST
        self.vdb_port = settings.VDB_PORT
        self.vdb_collection_name = f"collection_ig_{region}_{language}"
        self.embedding_model = 'sentence-transformers/all-MiniLM-L6-v2'

    def doc_details(self, results: dict, include_text: bool = False) -> list:
        """
        convert result into sources dict
        """
        sources = []
        for source in results:
            if include_text:
                sources.append({
                    "source": source.entity.get('source'),
                    "text": source.entity.get('text'),
                    "score": source.distance
                })
            else:
                sources.append({
                    "source": source.entity.get('source'),
                    "score": source.distance
                })
        sources = sorted(sources, key=lambda x: x["score"])
        return sources

    def get_embeddings(self, question: str):
        """
        Get embedding for question string
        """
        embedding_model = SentenceTransformer(self.embedding_model)
        return embedding_model.encode([question])

    def load_collection(self) -> Collection:
        """
        Connect to Milvus and load collection
        """
        connections.connect("default", host=self.vdb_host, port=self.vdb_port)
        collection = Collection(self.vdb_collection_name)
        collection.load()
        return collection

    def search_documents(
            self,
            question: str,
            limit_results: int = settings.SEARCH_MAX_DOCUMENTS,
            include_text: bool = False
        ) -> list:
        """
        Create summary answer for question
        """
        results = self.load_collection().search(
            data=self.get_embeddings(question),
            anns_field="vector",
            param={"metric_type": "L2", "params": {"nprobe": 10}},
            limit=limit_results,
            expr=None,
            consistency_level="Strong",
            output_fields=(["source", "text"] if include_text else ["source"])
        )[0]
        return self.doc_details(results, include_text)

    def deduplicate_pages(self, sources: list[dict], max_pages: int = settings.SEARCH_MAX_PAGES):
        """
        Get N unique pages from the sources retrieved from the retriever
        """
        unique_sources = []
        for source in sources:
            if source['source'] not in [source['source'] for source in unique_sources]:
                unique_sources.append(source)
            if len(unique_sources) == max_pages:
                break
        return unique_sources

    def retrieve_pages(self, sources: list[dict]) -> list[dict]:
        """
        Retrieve page content from Integreat CMS
        """
        top_pages = []
        for source in sources:
            top_pages.append(
                    {
                        "source": source["source"],
                        "text": self.fetch_page_from_cms(source["source"]),
                        "score": source["score"]
                    })
        return top_pages

    def retrieve_unique_pages(self, sources: list, max_pages: int) -> list:
        return self.retrieve_pages(self.deduplicate_pages(sources, max_pages))

    def fetch_page_from_cms(self, page_url: str) -> str:
        """
        get data from Integreat cms using the children endpoint
        """
        pages_url = (
            f"https://{settings.INTEGREAT_CMS_DOMAIN}/api/v3/{self.region}/"
            f"{self.language}/children/?url={page_url}&depth=0"
        )
        encoded_url = urllib.parse.quote(pages_url, safe=':/=?&')
        response = urllib.request.urlopen(encoded_url)
        page = json.loads(response.read())[0]
        if page["excerpt"]:
            return page["excerpt"]
        else:
            return ""
