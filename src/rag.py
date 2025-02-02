from typing import List
from llama_index.core import Document, Settings
from llama_index.vector_stores.pinecone import PineconeVectorStore
from src.util import load_config, set_embedding, set_llm
from src.ingestion import get_documents_from_web, add_nodes_to_vector_store, delete_nodes_by_ids
from src.data import transform
from src.retriever import Retriever
from src.prompts import CfgNetPromptSettings
import pandas as pd
from tqdm import tqdm


class RAG:
    def __init__(self, vector_store, retriever, generators) -> None:
        self.vector_store = vector_store
        self.prompts = CfgNetPromptSettings
        self.retriever = retriever
        self.generators = generators

        print("Init RAG.")
  
    def retrieve(self, dataset: pd.DataFrame, enable_websearch: bool) -> List:
        """
        Retrieve relevant context for each sample in the dataset and return the retrieval results
        
        Args:
            dataset: pd.Dataframe
                The dataset to retrieve the data from.
            enable_websearch: bool
                Whether to enable websearch.

        Returns:
            List of the retrieval results.
        """
        retrieval_results = []

        for index, row in tqdm(dataset.iterrows(), total=len(dataset), desc="Processing dependencies"):

            # turn row data into dict
            row_dict = row.to_dict()

            # transform row data into a dependency
            dependency = transform(row)

            # get retrieval prompt
            retrieval_str = self.prompts.get_retrieval_prompt(dependency=dependency)

            # scrape the web
            if enable_websearch:
                print(f"Start scraping the web for dependency {index}")
                web_documents = get_documents_from_web(
                    query_str=retrieval_str,
                    num_websites=3
                )

                # add nodes to vector store and store their ids
                web_node_ids = add_nodes_to_vector_store(
                    documents=web_documents,
                    vector_store=self.vector_store
                )

            # retrieve relavant nodes
            retrieved_nodes = self.retriever.retrieve(
                retrieval_str=retrieval_str
            )

            # defines context to append to row dict
            context = [
                {
                    "text": node.get_content(),
                    "score": str(node.get_score()),
                    "source": node.metadata["source"],
                    "id": node.node_id
                } for node in retrieved_nodes
            ]

            row_dict.update({"context": context})

            retrieval_results.append(row_dict)

            # delete nodes from web search
            delete_nodes_by_ids(
                vector_store=self.vector_store,
                ids=web_node_ids
            )

        return retrieval_results
    
    def generate(self, dataset: List, with_context: bool) -> List:
        """
        Generate answers for sample in the dataset and return generation results.

        Args:
            dataset: List
                The dataset to generate the answers for.
            with_context: bool
                Whether to start generation with or without context.
        
        Returns:
            List of the generation results.
        """
        generation_results = []

        for sample in tqdm(dataset, total=len(dataset), desc="Processing dependencies"):
            # transform data into dependency
            dependency = transform(row=sample)

            # create prompt parts
            system_str = self.prompts.get_system_str(dependency=dependency)
            task_str = self.prompts.get_task_str(dependency=dependency)
            context_str = "\n\n".join([x["text"] for x in sample["context"]])
            format_str = self.prompts.get_format_prompt()
            

            # create final query
            if with_context:
                query_str = self.prompts.query_prompt.format(
                    context_str=context_str,
                    task_str=task_str,
                    format_str=format_str
                )
            else:
                query_str = f"{task_str}\n\n{format_str}"

            messages = [
                {"role": "system", "content": system_str},
                {"role": "user", "content": query_str }
            ]

            generations = []
            for generator in self.generators:
                response = generator.generate(messages=messages)
                generations.append({generator.model_name: response})

            sample.update({"generations": generations})
            generation_results.append(sample)

            print("Generations: ", generations)

        return generation_results
    
    def compute_evaluation_metrics(self, dataset: List) -> dict:
        """
        Compute the evaluation metrics, including precision, recall and F1 score.

        Args:
            dataset: List
                The dataset to compute the evaluation metrics for.

        Returns:
            List of the evaluation results.
        """
        pass