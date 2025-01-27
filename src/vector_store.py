from pinecone import Pinecone, ServerlessSpec
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings
from typing import List

class VectorStore:
    def __init__(self, api_key: str, index_name: str, dimension: int):
        self.pc = Pinecone(api_key=api_key)
        self.index_name = index_name
        self.dimension = dimension

        self.index = self.get_index(index_name=index_name, dimension=dimension)
        self.vector_store = PineconeVectorStore(index=self.index)

    def get_index(self, index_name: str, dimension: int, delete_index: bool = False):
        """Get index.."""
        if index_name not in self.pc.list_indexes().names():
            print(f"Create index: {index_name}")
            self.pc.create_index(
                index_name,
                dimension=dimension,
                metric="dotproduct",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
        
        if delete_index:
            print(f"Delete index: {index_name}")
            self.pc.delete_index(index_name)

        
        return self.pc.Index(index_name)
    

    def add(self, documents: List, chunk_size: int, chunk_overlap: int):
        """
        Add nodes to vector store.
        """
        text_parser = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        nodes = text_parser.get_nodes_from_documents(documents=documents)

        # get embeddings for nodes
        for node in nodes:
            node_embedding = Settings.embed_model.get_text_embedding(
                node.get_content(metadata_mode="all")
            )
            node.embedding = node_embedding

        self.vector_store.add(nodes=nodes)