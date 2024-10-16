"""
A service to search for documents
"""
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
    def __init__(self, region, language):
        self.language = language
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
        return self.doc_details(results, include_text)
