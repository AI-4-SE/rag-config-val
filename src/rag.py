from data import Dependency, Response
from typing import List
from util import load_config, set_embedding, set_llm, get_embedding_dimension
import backoff
import os
import logging
import pandas as pd


class RAG:
    def __init__(self, retriever, generators: List, prompts: dict) -> None:
        self.retriever = retriever
        self.generators = generators
        self.prompts = prompts
        print("RAG system initialized.")
  
    def retrieve(self, index, enable_websearch: bool, dataset: pd.Dataframe) -> List:
        """
        Retrieve relevant context for each sample in the dataset and return the retrieval results
        
        Args:  
            index: str
                The index to retrieve the data from.
            enable_websearch: bool
                Whether to enable websearch.
            dataset: pd.Dataframe
                The dataset to retrieve the data from.

        Returns:
            List of the retrieval results.
        """
        retrieval_results = self.retriever.retrieve(
            index=index,
            enable_websearch=enable_websearch,
            dataset=dataset,
            retrieval_prompt=self.prompts["retrieval_prompt"]
        )

        return retrieval_results

    def generate(self, dataset: List) -> List:
        """
        Generate answers for sample in the dataset and return generation results.

        Args:
            dataset: List
                The dataset to generate the answers for.
        
        Returns:
            List of the generation results.
        """

        for generator in self.generators:
            genration_results = generator.generate(
                dataset=dataset,
                generation_prompt=self.prompts["generation_prompt"]
            )

            #Update the dataset with the generation results
            #dataset.update(genration_results)

        return dataset
    
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