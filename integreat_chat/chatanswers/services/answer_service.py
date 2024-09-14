"""
Retrieving matching documents for question an create summary text
"""
import json

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain import hub
from langchain_community.llms import Ollama
from langchain_milvus.vectorstores import Milvus

from django.conf import settings

class AnswerService:
    _instance = None

    @staticmethod
    def get_instance(region, language):
        if AnswerService._instance is None:
            AnswerService._instance = AnswerService(region, language)
        return AnswerService._instance

    def __init__(self, region, language):
        self.language = language
        self.llm_model_name = settings.MODEL_LLM
        self.embedding_model_name = settings.MODEL_EMBEDDINGS
        self.embedding_model = self.load_embeddings(self.embedding_model_name)

        self.vdb_host = settings.VDB_HOST
        self.vdb_port = settings.VDB_PORT
        self.vdb_collection = f"collection_ig_{region}_{language}"
        self.vdb = self.load_vdb(self.vdb_host, self.vdb_port,
                                 self.vdb_collection, self.embedding_model)

        self.llm = self.load_llm(self.llm_model_name)

    def load_config(self, config_path):
        with open(config_path) as f:
            config = json.load(f)
        return config

    def load_llm(self, llm_model_name):
        llm = Ollama(model=llm_model_name, base_url=settings.OLLAMA_BASE_PATH)
        return llm

    def load_embeddings(self, embedding_model_name):
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name, show_progress=False)
        return embeddings

    def load_vdb(self, URI, port, collection, embedding_model):
        vdb = Milvus(
                embedding_model,
                connection_args={"host": URI, "port": port},
                collection_name=collection)
        return vdb

    def extract_answer(self, question):
        prompt = hub.pull("rlm/rag-prompt")
        retriever = self.vdb.as_retriever()
        results = retriever.get_relevant_documents(question)
        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
                | prompt
                | self.llm
                | StrOutputParser()
        )
        answer = rag_chain.invoke(question)
        return {"answer": answer, "sources": {result.metadata["source"] for result in results[:3]}}
