#!/usr/bin/env python3

from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Weaviate

DATA_PATH = "../data/muenchen_en/"
WEAVIATE_URL = "http://localhost:8080"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

loader = DirectoryLoader(DATA_PATH, show_progress=True)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
data_split = text_splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, show_progress=True)

vdb = Weaviate.from_documents(data_split, embeddings, weaviate_url=WEAVIATE_URL)

sample_query = "How can I learn German?"

result = vdb.similarity_search(sample_query)
print(result)
