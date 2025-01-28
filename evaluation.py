import toml
import argparse
import pandas as pd
import os
import json
from pinecone import Pinecone
from dotenv import load_dotenv
from src.rag import RAG
from typing import List
from src.retriever import Retriever
from src.util import load_config, set_embedding, set_llm
from llama_index.vector_stores.pinecone import PineconeVectorStore


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default="config.toml")
    parser.add_argument("--env_file", type=str, default=".env")
    return parser.parse_args()

def get_index(index_name: str):
    pinecone_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    if index_name in pinecone_client.list_indexes().names():
        pinecone_client.Index(index_name)
    else:
        print(f"Index '{index_name}' does not exist")
        return None

    # get pinecone index
    pinecone_index = pinecone_client.Index(index_name)

    return pinecone_index

def run_evaluation():
    # parse args
    args = parse_args()

    # load env variables
    load_dotenv(dotenv_path=args.env_file)

    # load config
    config = load_config(config_file=args.config_file)

    # set inference and embedding moddels
    set_llm(inference_model_name=config["generation"]["inference_model"])
    set_embedding(embed_model_name=config["indexing"]["embedding_model"])

    # get index
    index = get_index(index_name=config["indexing"]["index_name"])

    # get vector store
    vector_store = PineconeVectorStore(
        pinecone_index=index,
        add_sparse_vector=True
    )

    # get retriever
    retriever = Retriever(
        vector_store=vector_store,
        rerank=config["retrieval"]["rerank"],
        top_k=config["retrieval"]["top_k"],
        top_n=config["retrieval"]["top_n"],
        alpha=config["retrieval"]["alpha"]
    )

    # TODO: define genrators

    # init rag
    rag = RAG(
        vector_store=vector_store,
        retriever=retriever,
        generators=[]
    )

    #retrieval_results = rag.retrieve(
    #    dataset=pd.read_csv(config["evaluation"]["data_file"]),
    #    enable_websearch=True
    #)
    
    #with open(config["evaluation"]["output_file"], "w", encoding="utf-8") as dest:
    #    json.dump(retrieval_results, dest, indent=2)

    with open("data/evaluation/test_dependencies_retrieval.json", "r", encoding="utf-8") as src:
        data = json.load(src)

    generation_results = rag.generate(dataset=data, with_context=True)


if __name__ == "__main__":
    run_evaluation()