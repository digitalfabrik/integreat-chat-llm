"""
Update Milvus vector database with new data
"""

import json
import hashlib
import logging
import urllib

from django.conf import settings
from langchain_text_splitters import HTMLHeaderTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Milvus


LOGGER = logging.getLogger('django')

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
        pages_url = (
            f"https://{settings.INTEGREAT_CMS_DOMAIN}/api/v3/{self.region}/{self.language}/pages"
        )
        response = urllib.request.urlopen(pages_url)
        pages = json.loads(response.read())
        return pages

    def split_page(self, page):
        """
        split pages at headlines
        """
        if page["content"] == "":
            return [], []
        headers_to_split_on = [
            ("h1", "headline"),
            ("h2", "headline"),
        ]
        html_splitter = HTMLHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
        )
        documents = html_splitter.split_text(page['content'])
        texts = []
        paths = []
        for doc in documents:
            texts.append(doc.page_content)
            paths.append({"source": page['path']})
        return texts, paths

    def deduplicate_documents(self, texts, paths):
        """
        Guarantees that each text chunk is unique.
        """
        hashes = []
        result_texts = []
        result_paths = []
        num_dedups = 0
        for document in zip(texts, paths):
            text, path = document
            text_hash = hashlib.md5(text.encode(encoding="utf-8")).digest()
            if text_hash in hashes:
                num_dedups += 1
                continue
            hashes.append(text_hash)
            result_texts.append(text)
            result_paths.append(path)
        return result_texts, result_paths, num_dedups

    def create_embeddings(self, texts, paths):
        """
        create embeddings and save to database
        """
        embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model, show_progress=False)

        Milvus.from_texts(
            texts,
            embeddings,
            metadatas=paths,
            collection_name=self.milvus_collection,
            connection_args={"host": self.milvus_host, "port": self.milvus_port},
            consistency_level='Session',
            index_params={
                        "metric_type": "L2",
                        "index_type": "FLAT",
                    },
            drop_old=True
        )
