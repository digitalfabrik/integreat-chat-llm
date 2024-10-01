"""
A service to search for documents
"""
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
from langchain_milvus.vectorstores import Milvus

#from langchain.runnables.base import RunnableLambda
from langchain_core.runnables import RunnableLambda

from django.conf import settings


class SearchService:
    """
    Service class that enables searching for Integreat content
    """
    _instance = None

    @staticmethod
    def get_instance(region, language):
        if SearchService._instance is None:
            SearchService._instance = SearchService(region, language)
        return SearchService._instance

    def __init__(self, region, language):
        self.language = language
        self.llm_model_name = settings.RAG_MODEL

        self.vdb_host = settings.VDB_HOST
        self.vdb_port = settings.VDB_PORT
        self.vdb_collection = f"collection_ig_{region}_{language}"
        self.vdb = self.load_vdb(self.vdb_host, self.vdb_port,
                                 self.vdb_collection, settings.EMBEDDINGS)

    def load_vdb(self, URI, port, collection, embedding_model):
        vdb = Milvus(
                embedding_model,
                connection_args={"host": URI, "port": port},
                collection_name=collection)
        return vdb

    def doc_details(self, results):
        """
        convert result into sources dict
        """
        sources = []
        for source in results:
            sources.append({"source": source[0].metadata["source"], "score": source[1]})
        return sources

    def search_documents(self, question):
        """
        Create summary answer for question
        """
        results = [
            result for result in self.vdb.similarity_search_with_score(
                question, k=settings.SEARCH_MAX_DOCUMENTS
            ) if result[1] < settings.SEARCH_DISTANCE_THRESHOLD
        ]
        return {
            "documents": self.doc_details(results)
        }
