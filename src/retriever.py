from llama_index.core import QueryBundle, Settings, VectorStoreIndex
from llama_index.core.schema import NodeWithScore
from llama_index.core.retrievers import BaseRetriever
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.postprocessor.sbert_rerank import SentenceTransformerRerank
from llama_index.core.vector_stores import VectorStoreQuery
from typing import Any, List, Optional


class Retriever():
    def __init__(
        self,
        vector_store: PineconeVectorStore,
        rerank: str = "colbert",
        top_k: int = 10,
        top_n: int = 5,
        alpha: float = 0.5
    ) -> None:
        """Init params."""
        self.vector_store = vector_store
        self.embed_model = Settings.embed_model
        self.rerank = rerank
        self.top_k = top_k
        self.top_n = top_n
        self.alpha = alpha

        self.reranker = SentenceTransformerRerank(
            model="cross-encoder/ms-marco-MiniLM-L-2-v2", 
            top_n=self.top_n,
            device="cpu"
        )

        print("Init retriever.")

    def _get_reranker(self):
        if self.rerank.lower() == "colbert":
            return ColbertRerank(
                top_n=5,
                model="colbert-ir/colvertv2.0",
                tokenizer="colbert-ir/colvertv2.0",
                keep_retrieval_score=True
            )
        else:
            return None

    def rerank_nodes(self, nodes: List[NodeWithScore], retrieval_str: str) -> List[NodeWithScore]:
        """
        Reranks a list of nodes based on a retrieval string.

        Args:
            nodes (List[NodeWithScore]): A list of nodes with associated scores.
            retrieval_str (str): The retrieval string used for reranking the nodes.
        Returns:
            List[NodeWithScore]: A list of reranked nodes.
        """
        reranked_nodes = self.reranker.postprocess_nodes(
            nodes=nodes,
            query_bundle=QueryBundle(query_str=retrieval_str)
        )

        return reranked_nodes
    

    def retrieve(self, retrieval_str: str) -> List[NodeWithScore]:
        """
        Retrieve and rerank nodes based on the given retrieval string.

        Args:
            retrieval_str (str): The string used to perform the retrieval.
        Returns:
            List[NodeWithScore]: A list of reranked nodes with their respective scores.
        """
        # create vector store index
        index = VectorStoreIndex.from_vector_store(
            vector_store=self.vector_store,
            embed_model=Settings.embed_model
        )

        # create retriever query engine
        query_engine = index.as_query_engine(
            vector_store_query_mode="hybrid",
            similarity_top_k=self.top_k,
            alpha=self.alpha
        )

        # retrieve nodes
        retrieved_nodes = query_engine.retrieve(
            query_bundle=QueryBundle(query_str=retrieval_str)
        )

        # rerank nodes
        reranked_nodes = self.rerank_nodes(
            nodes=retrieved_nodes, 
            retrieval_str=retrieval_str
        )

        return reranked_nodes