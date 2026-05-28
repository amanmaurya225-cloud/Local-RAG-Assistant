"""
Main RAG Pipeline
-----------------
Runs the complete local RAG assistant.
"""

from src.ingestion import DocumentIngestion
from src.embeddings import EmbeddingModel
from src.vector_store import VectorStoreManager
from src.retriever import Retriever
from src.llm import LocalLLM


PDF_PATH = "data/sample.pdf"


def main():

    # =========================
    # INGESTION
    # =========================

    ingestion = DocumentIngestion()

    documents = ingestion.load_pdf(PDF_PATH)

    chunks = ingestion.split_documents(documents)

    # =========================
    # EMBEDDINGS
    # =========================

    embedding_model = EmbeddingModel()

    # =========================
    # VECTOR STORE
    # =========================

    vector_store = VectorStoreManager()

    vector_store.add_documents(chunks, embedding_model)

    # =========================
    # RETRIEVER
    # =========================

    retriever = Retriever(vector_store.collection)

    # =========================
    # LOCAL LLM
    # =========================

    llm = LocalLLM(model_name="qwen3:8b")

    print("\n===== Local RAG Assistant =====")

    while True:

        query = input("\nAsk a question (or type 'exit'): ")

        if query.lower() == "exit":
            break

        query_embedding = embedding_model.generate_embedding(query)

        retrieved_docs = retriever.retrieve(
            query_embedding=query_embedding,
            top_k=3
        )

        context = "\n".join(
            [doc["document"] for doc in retrieved_docs]
        )

        response = llm.generate_response(
            query=query,
            context=context
        )

        print("\nAssistant Response:\n")
        print(response)


if __name__ == "__main__":
    main()
