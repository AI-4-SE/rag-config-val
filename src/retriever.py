from llama_index.core import QueryBundle, Settings, VectorStoreIndex
from llama_index.core.schema import NodeWithScore
from llama_index.postprocessor.sbert_rerank import SentenceTransformerRerank
from llama_index.postprocessor.colbert_rerank import ColbertRerank
from typing import Any, List


class Retriever():
    def __init__(
        self,
        rerank: str,
        top_k: int,
        top_n: int,
        alpha: float
    ) -> None:
        """Init params."""
        self.rerank = rerank
        self.top_n = top_n
        self.top_k = top_k
        self.alpha = alpha

        print("Init retriever.")

    def _get_reranker(self):
        if self.rerank.lower() == "colbert":
            print("Init ColbertRerank.")
            return ColbertRerank(
                top_n=self.top_n,
                model="colbert-ir/colbertv2.0",
                tokenizer="colbert-ir/colbertv2.0",
                keep_retrieval_score=True
            )
        if self.rerank.lower() == "sentence":
            print("Init SentenceTransformerRerank.")
            return SentenceTransformerRerank(
            model="cross-encoder/ms-marco-MiniLM-L-2-v2", 
            top_n=self.top_n,
            device="cpu"
        )
        else:
            print("No reranker specified.")
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
        reranker = self._get_reranker()
        reranked_nodes = reranker.postprocess_nodes(
            nodes=nodes,
            query_bundle=QueryBundle(query_str=retrieval_str)
        )
        print(f"Reranked {len(nodes)} in {len(reranked_nodes)} nodes.")
        return reranked_nodes
    
    def retrieve(self, vector_store, retrieval_str: str) -> List[NodeWithScore]:
        """
        Retrieve and rerank nodes based on the given retrieval string.

        Args:
            retrieval_str (str): The string used to perform the retrieval.
        Returns:
            List[NodeWithScore]: A list of reranked nodes with their respective scores.
        """
        # create vector store index
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=Settings.embed_model
        )

        # create retriever query engine
        query_engine = index.as_query_engine(
            vector_store_query_mode="hybrid",
            similarity_top_k=self.top_k,
            alpha=self.alpha
        )

        retrieved_nodes = []

        while not retrieved_nodes:
            # retrieve nodes
            retrieved_nodes = query_engine.retrieve(
                query_bundle=QueryBundle(query_str=retrieval_str)
            )

        print("Len retrieved nodes: ", len(retrieved_nodes))

        return retrieved_nodes