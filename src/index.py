import pinecone
import uuid
from llama_index.vector_stores import VectorStore
from typing import List, Tuple, Optional

class RAGIndex(VectorStore):
    def __init__(self, api_key):
        """
        Initialize the RAGIndex with Pinecone and make it compatible as a vector store for LlamaIndex.

        :param api_key: API key for Pinecone.
        :param index_name: Name of the Pinecone index.
        """
        # Initialize Pinecone
        self.client = Pinecone(api_key=api_key)

    def create_index(self, index_name: str, dimension: int)
        # Connect to Pinecone index
        if index_name not in client.list_indexes().names():
            print(f"Create index: {index_name}")
            self.client.create_index(name=
                index_name, 
                dimension=1536, 
                metric="dotproduct",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
        else:
            print(f"Index with name {index_name} already exists.")

        self.index = self.client.Index(pinecone_index_name)


    def add(self, documents: List):
        entries = []

        for doc in documents:
            vector = Settings.embed_model.get_text_embedding(doc["text"])
            entries.append(
                {"id": }
            )
        

        node_parser = SentenceSplitter(
            chunk_size=512, 
            chunk_overlap=50
        )


        vectors = [(doc_id, vector, metadata or {}) for doc_id, vector, metadata in embeddings]
        self.pinecone_index.upsert(vectors)

    def delete(self, doc_ids: List[str]):
        """
        Delete vectors from the vector store.

        :param doc_ids: A list of document IDs to delete.
        """
        self.pinecone_index.delete(ids=doc_ids)

    def query(self, query_embedding: List[float], top_k: int) -> List[Tuple[str, float, dict]]:
        """
        Query the vector store for similar embeddings.

        :param query_embedding: The query embedding vector.
        :param top_k: Number of top results to return.
        :return: A list of tuples containing (id, similarity_score, metadata).
        """
        results = self.pinecone_index.query(query_embedding, top_k=top_k, include_metadata=True)
        return [
            (match["id"], match["score"], match.get("metadata", {}))
            for match in results.get("matches", [])
        ]
