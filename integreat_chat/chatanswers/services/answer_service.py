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

class AnswerService:
    def __init__(self, language):
        self.language = language
        config_path = os.path.join(os.path.dirname(__file__), "config.json")
        
        # Load models
        self.config = self.load_config(config_path)
        self.llm_model_name = self.config["model_llm"]
        self.embedding_model_name = self.config["model_embeddings"]
        self.embedding_model = self.load_embeddings(self.embedding_model_name)
        
        # Load vdb collection
        self.vdb_URI = self.config["vdb_URI"]
        self.vdb_port = self.config["vdb_port"]
        self.vdb_collection = self.config["vdb_collection"]
        self.vdb = self.load_vdb(self.vdb_URI, self.vdb_port, self.vdb_collection, self.embedding_model)
    
    def load_config(self):
        with open(self.config_path) as f:
            config = json.load(f)
        return config
    
    def load_llm(self, llm_model_name):
        pass

    def load_embeddings(self, embedding_model_name):
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name, show_progress=True)
        return embeddings

    def load_vdb(self, URI, port, collection, embeddings):
        vdb = Milvus(
                embeddings,
                connection_args={"host": URI, "port": port},
                collection_name=collection)
        return vdb

    def extract_answer(self, question):
        
        return []
