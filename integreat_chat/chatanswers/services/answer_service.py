from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain import hub
from langchain_community.llms import Ollama
from langchain_milvus.vectorstores import Milvus
import os
import json
from django.conf import settings

class AnswerService:
    _instance = None

    @staticmethod
    def get_instance(language):
        if AnswerService._instance is None:
            AnswerService._instance = AnswerService(language)
        return AnswerService._instance

    def __init__(self, language):
        self.language = language
        config_path = os.path.join(settings.BASE_DIR, "chatanswers", "services", "config.json")
        
        # Load models
        #self.config = self.load_config(config_path)
        """
        self.llm_model_name = self.config["model_llm"]
        self.embedding_model_name = self.config["model_embeddings"]
        self.embedding_model = self.load_embeddings(self.embedding_model_name)
        """
        self.llm_model_name = "wizardlm2:latest"
        self.embedding_model_name = "all-MiniLM-L6-v2"
        self.embedding_model = self.load_embeddings(self.embedding_model_name)
        # Load vdb collection
        """
        self.vdb_URI = self.config["vdb_URI"]
        self.vdb_port = self.config["vdb_port"]
        self.vdb_collection = self.config["vdb_collection"]
        """

        self.vdb_URI = "127.0.0.1"
        self.vdb_port = "19530"
        self.vdb_collection = "collection_1"
        self.vdb = self.load_vdb(self.vdb_URI, self.vdb_port, self.vdb_collection, self.embedding_model)

        # Load LLM
        self.llm = self.load_llm(self.llm_model_name)
    
    def load_config(self, config_path):
        with open(config_path) as f:
            config = json.load(f)
        return config
    
    def load_llm(self, llm_model_name):
        print("Loading LLM model")
        llm = Ollama(model=llm_model_name)
        return llm

    def load_embeddings(self, embedding_model_name):
        print("Loading embeddings")
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name, show_progress=True)
        return embeddings

    def load_vdb(self, URI, port, collection, embeddings):
        print("Loading vdb")
        vdb = Milvus(
                embeddings,
                connection_args={"host": URI, "port": port},
                collection_name=collection)
        return vdb

    def extract_answer(self, question):
        print("Extracting answer")
        # Get prompt
        prompt = hub.pull("rlm/rag-prompt")
        # Get retriever
        retriever = self.vdb.as_retriever()
        # Get answer
        
        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
                | prompt
                | self.llm
                | StrOutputParser()
        )
        print("Invoking rag chain")
        answer = rag_chain.invoke(question)
        return answer
