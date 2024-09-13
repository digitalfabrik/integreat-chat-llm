"""
Update Milvus vector database with new data
"""

import urllib
import json

from langchain_text_splitters import HTMLHeaderTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Milvus

class UpdateMilvus:
    """
    Update Milvus vdb
    """

    def __init__(self, region="testumgebung", language="en"):
        self.region = region
        self.language = language
        self.milvus_host = "127.0.0.1"
        self.milvus_port = "19530"
        self.embedding_model = "all-MiniLM-L6-v2"
        self.milvus_collection = f"collection_ig_{region}_{language}"

    def fetch_pages_from_cms(self):
        """
        get data from Integreat cms
        """
        pages_url = f"https://cms-test.integreat-app.de/api/v3/{self.region}/{self.language}/pages"
        response = urllib.request.urlopen(pages_url)
        pages = json.loads(response.read())
        return pages

    def split_page(self, page):
        """
        split pages at headlines
        """
        if page["content"] == "":
            return []
        headers_to_split_on = [
            ("h2", "Header 2"),
        ]
        html_splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        documents = html_splitter.split_text(page['content'])
        for doc in documents:
            doc.metadata['source'] = page['path']
        return documents

    def create_embeddings(self, text_chunks):
        """
        create embeddings and save to database
        """
        embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model, show_progress=False)
        Milvus.from_documents(text_chunks, embeddings, collection_name=self.milvus_collection, drop_old=True,
                              connection_args={"host": self.milvus_host, "port": self.milvus_port})
