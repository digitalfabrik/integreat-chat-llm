"""
A service to search for documents
"""
from sentence_transformers import SentenceTransformer
from pymilvus import (
    connections,
    Collection,
)

from django.conf import settings
import urllib.request
import urllib.parse
import json

class SearchService:
    """
    Service class that enables searching for Integreat content
    """
    def __init__(self, region, language):
        self.language = language
        self.region = region
        self.vdb_host = settings.VDB_HOST
        self.vdb_port = settings.VDB_PORT
        self.vdb_collection_name = f"collection_ig_{region}_{language}"
        self.embedding_model = 'sentence-transformers/all-MiniLM-L6-v2'

    def doc_details(self, results, include_text=False):
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

    def get_embeddings(self, question):
        """
        Get embedding for question string
        """
        embedding_model = SentenceTransformer(self.embedding_model)
        return embedding_model.encode([question])

    def load_collection(self):
        """
        Connect to Milvus and load collection
        """
        connections.connect("default", host=self.vdb_host, port=self.vdb_port)
        collection = Collection(self.vdb_collection_name)
        collection.load()
        return collection

    def search_documents(
            self,
            question,
            limit_results=settings.SEARCH_MAX_DOCUMENTS,
            include_text=False
        ):
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
        sources = self.doc_details(results, include_text)
        return self.retrieve_unique_pages(sources)

    def retrieve_unique_pages(self, sources):
        """
        Get 3 unique pages from the sources retrieved from the retriever
        """
        unique_sources = []
        top_pages = []
        for source in sources:
            if source['source'] not in unique_sources:
                unique_sources.append(source['source'])
            if len(unique_sources) == 3:
                break
        print(unique_sources)
        for source in unique_sources:
            top_pages.append(
                    {
                        "source": source,
                        "text": self.fetch_page_from_cms(source)
                    })
        return top_pages

    def fetch_page_from_cms(self, page_url):
        """
        get data from Integreat cms using the children endpoint
        """
        pages_url = f"https://cms-test.integreat-app.de/api/v3/{self.region}/{self.language}/children/?url={page_url}&depth=0"
        encoded_url = urllib.parse.quote(pages_url, safe=':/=?&')
        print(f"URL - {encoded_url}\n")
        response = urllib.request.urlopen(encoded_url)
        page = json.loads(response.read())[0]
        if page["excerpt"]:
            return page["excerpt"]
        else:
            return ""
