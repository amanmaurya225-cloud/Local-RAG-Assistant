"""
Vector Store Module
-------------------
Handles ChromaDB vector storage.
"""

import chromadb
import os


class VectorStoreManager:

    def __init__(
        self,
        persist_directory="vector_store",
        collection_name="rag_collection"
    ):

        self.persist_directory = persist_directory
        self.collection_name = collection_name

        self.client = None
        self.collection = None

        self._initialize_store()

    def _initialize_store(self):

        os.makedirs(self.persist_directory, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=self.persist_directory
        )

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={
                "description": "Local RAG Assistant Vector Store"
            }
        )

        print("Vector Store Initialized")
        print(f"Documents in collection: {self.collection.count()}")

    def add_documents(self, chunks, embedding_model):

        for idx, chunk in enumerate(chunks):

            embedding = embedding_model.generate_embedding(
                chunk.page_content
            )

            self.collection.add(
                ids=[str(idx)],
                embeddings=[embedding],
                documents=[chunk.page_content],
                metadatas=[chunk.metadata]
            )

        print("Documents added successfully")
