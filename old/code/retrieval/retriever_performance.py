#!/usr/bin/env python3

from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
import json
import time

DATA_PATH = "../../data/muenchen_en/"
EMBEDDING_MODELS_PATH = "embedding_models.json"
QUERIES_PATH = "retriever_queries.json"
SAVE_PATH = "results/"

def load_json_container(path):
    with open(path, "r") as f:
        json_array = json.load(f)
    return json_array

def pretty_save(results, model, query, number, save_path):
    output_text = ""
    for i, d in enumerate(results):
        output_text += f"Model: {model}\nQuery: {query}\n"

        output_text += f"\n{'-' * 100}\n"
        output_text += f"Document {i+1}:\n\n{d.page_content}\nMetadata: {d.metadata}\n"

    file_name = f"{save_path}{model}_{number}.txt"
    with open(file_name, "w", encoding="utf-8") as f:
        f.write(output_text)

# Load data from directory
t1 = time.time()
loader = DirectoryLoader(DATA_PATH, show_progress=True)
docs = loader.load()
t2 = time.time()
td = t2 - t1
print(f"\nDataset load successfull -{td:.2f} seconds")

# Define a text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
data_split = text_splitter.split_documents(docs)
t3 = time.time()
td = t3 - t2
print(f"\nData Split- {td:.2f} seconds")

# Define embeddings
embedding_models = load_json_container(EMBEDDING_MODELS_PATH)
for model in embedding_models:
    tm1 = time.time()
    model_name = f"{model['model_base']}/{model['model_name']}"
    print(f"\nModel: {model_name}")
    embeddings = HuggingFaceEmbeddings(model_name=model_name, show_progress=True)

    # Vector Database
    db = FAISS.from_documents(data_split, embeddings)
    retriever = db.as_retriever()
    queries = load_json_container(QUERIES_PATH)
    tm2 = time.time()
    print(f"Model:{model_name} load in db successfull -{(tm2 - tm1):.2f} seconds")
    for i, query in enumerate(queries):
        query = query["query"]
        results = retriever.get_relevant_documents(query)
        pretty_save(results, model["model_name"], query, i, SAVE_PATH)
    tm3 = time.time()
    print(f"Model:{model_name} query results saved - {(tm3 - tm2):.2f} seconds")
