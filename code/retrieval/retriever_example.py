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
SAVE_PATH = "results/chunk/"

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
    #with open(file_name, "w", encoding="utf-8") as f:
    #    f.write(output_text)
    print(output_text)

def results(query):
    global retriever
    results = retriever.get_relevant_documents(query)
    output_text = ""
    for i, d in enumerate(results):
        #output_text += f"Model: {model}\nQuery: {query}\n"
        output_text += f"\n{'-' * 100}\n"
        output_text += f"Document {i+1}:\n\n{d.page_content}\nMetadata: {d.metadata}\n"
    filename = "example_retriever.txt"
    with open(filename, 'w', encoding="utf-8") as f:
        f.write(output_text)
    print(output_text)

# Load data from directory
t1 = time.time()
loader = DirectoryLoader(DATA_PATH, show_progress=True)
docs = loader.load()
t2 = time.time()
td = t2 - t1
#print(f"\nDataset load successfull -{td:.2f} seconds")

# Define a text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=500)
data_split = text_splitter.split_documents(docs)
t3 = time.time()
td = t3 - t2
#print(f"\nData Split- {td:.2f} seconds")

model_name = "mixedbread-ai/mxbai-embed-large-v1"
embeddings = HuggingFaceEmbeddings(model_name=model_name, show_progress=True)

db = FAISS.from_documents(data_split, embeddings)
retriever = db.as_retriever()
#query = "How can I learn german in munich?"
#results = retriever.get_relevant_documents(query)
#pretty_save(results, model_name, query, 1, SAVE_PATH)
