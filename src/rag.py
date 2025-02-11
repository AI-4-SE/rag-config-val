from typing import Dict, List
from src.ingestion import get_documents_from_web, add_nodes
from llama_index.vector_stores.pinecone import PineconeVectorStore
from src.retriever import Retriever
from src.generator import Generator
from src.prompts import Prompts
from src.utils import get_projet_description, get_most_similar_shots, load_shots
from src.data import Dependency
from rich.logging import RichHandler
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler()],
)

class RAG:
    def __init__(
        self, 
        static_index,
        dynamic_index ,
        retriever: Retriever, 
        generators: List[Generator]
    ) -> None:
        self.vector_store_static = PineconeVectorStore(
            pinecone_index=static_index,
            add_sparse_vector=True
        )
        self.vector_store_dynamic = PineconeVectorStore(
            pinecone_index=dynamic_index,
            add_sparse_vector=True
        )
        self.prompts = Prompts
        self.retriever = retriever
        self.generators = generators
        logging.info("Initialize RAG.")
  
    def retrieve(self, dependency: Dependency, with_websearch: bool) -> str:
        """
        Retrieve relevant context for given dependency
        
        Args:
            dependency: Dependency
                Dependency to retrieve context for.
            with_websearch: bool
                Whether to enable websearch.

        Returns:
            Retrieved context.
        """
        retrieved_dynamic_context = []
        retrieved_static_context = []

        # get retrieval prompt
        retrieval_str = self.prompts.get_retrieval_prompt(dependency=dependency)

        # scrape the web
        if with_websearch:
            logging.info(f"Start Web scraping.")
            web_documents = get_documents_from_web(
                    query_str=retrieval_str,
                    num_websites=3
                )
            
            logging.info(f"Add {len(web_documents)} web documents to vector store.")
            # add nodes to vector store and store their ids
            if web_documents:
                web_node_ids = add_nodes(
                    documents=web_documents,
                    vector_store=self.vector_store_dynamic
                )

                # retrieve relavant nodes from dynamic vector store
                retrieved_dynamic_context = self.retriever.retrieve(
                    vector_store=self.vector_store_dynamic,
                    retrieval_str=retrieval_str
                )

                # delete nodes from web index
                self.vector_store_dynamic.delete_nodes(node_ids=web_node_ids)
            else:
                logging.info("No web documents found. There is nothing to add nor delete.")

        # retrieve relavant nodes
        retrieved_static_context = self.retriever.retrieve(
            retrieval_str=retrieval_str
        )

        # retrieve relavant nodes from statix vector store
        retrieved_static_context = self.retriever.retrieve(
            vector_store=self.vector_store_static,
            retrieval_str=retrieval_str
        )

        # merge static and dynamic context and rerank
        reranked_nodes = self.retriever.rerank_nodes(
            nodes=retrieved_static_context + retrieved_dynamic_context,
            retrieval_str=retrieval_str
        )

        # define context string
        context_str = "\n\n".join([node.get_content() for node in reranked_nodes])
        
        return context_str
    
    def generate(self, dependency: Dependency, with_context: bool, with_websearch: bool, advanced: bool = False) -> Dict:
        """
        Validate a given dependency with our without context.

        Args:
            dependency: Dependency
                Dependency to validate.
            with_context: bool
                Whether to start validation with or without context.
            with_websearch: bool
                Whether to enable websearch or not.
            advanced: bool
                Whether to use advanced prompts or not.
        
        Returns:
            List of the generation results.
        """
        if advanced:
            shots = load_shots()
            project_info = get_projet_description(project_name=dependency.project)
            shot_str = "\n\n".join([shot for shot in get_most_similar_shots(shots, dependency)])

        system_str = self.prompts.get_system_str(
            project_name=dependency.project,
            project_info=project_info,
            advanced=advanced
        )

        task_str = self.prompts.get_task_str(dependency=dependency)
        context_str = self.retrieve(dependency=dependency, with_websearch=with_websearch)
        format_str = self.prompts.get_format_prompt()
            

        # create final query
        if with_context:
            query_str = self.prompts.get_query_str(
                context_str=context_str,
                task_str=task_str,
                shot_str=shot_str,
                advanced=advanced
            )
        else:
            query_str = f"{task_str}\n\n{format_str}"

        messages = [
            {"role": "system", "content": system_str},
            {"role": "user", "content": query_str }
        ]

        generations = {}
        for generator in self.generators:
            response = generator.generate(messages=messages)
            generations.update({generator.model_name: response})

        return generations