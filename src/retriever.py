"""
Retriever Module
----------------
Performs semantic similarity search.
"""


class Retriever:

    def __init__(self, collection):

        self.collection = collection

    def retrieve(self, query_embedding, top_k=3):

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        retrieved_docs = []

        ids = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for i, (doc_id, document, metadata, distance) in enumerate(
            zip(ids, documents, metadatas, distances)
        ):

            similarity_score = 1 - distance

            retrieved_docs.append({
                "id": doc_id,
                "document": document,
                "metadata": metadata,
                "similarity_score": similarity_score,
                "rank": i + 1
            })

        return retrieved_docs
