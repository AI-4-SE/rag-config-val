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
from src.generator import GeneratorFactory
from src.util import load_config, set_embedding, set_llm, compute_evaluation_metrics
from llama_index.vector_stores.pinecone import PineconeVectorStore


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str)
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
    config_name = os.path.basename(args.config_file).split(".")[0]

    # set inference and embedding moddels
    set_llm(inference_model_name=config["inference_models"][0])
    set_embedding(embed_model_name=config["embedding_model"])

    # get index
    index = get_index(index_name=config["index_name"])

    # get vector store
    vector_store = PineconeVectorStore(
        pinecone_index=index,
        add_sparse_vector=True
    )

    # get retriever
    retriever = Retriever(
        vector_store=vector_store,
        rerank=config["rerank"],
        top_n=config["top_n"]
    )

    # TODO: define genrators
    generators = [
        GeneratorFactory().get_generator(model_name=inference_model_name, temperature=config["temperature"]) for inference_model_name in config["inference_models"]
    ]

    # init rag
    rag = RAG(
        vector_store=vector_store,
        retriever=retriever,
        generators=generators
    )

    print("Start retrieving context information.")
    retrieval_results = rag.retrieve(
        dataset=pd.read_csv(config["data_file"]),
        enable_websearch=config["web_search_enabled"]
    )

    with open(config["retrieval_output_file"], "w", encoding="utf-8") as dest:
        json.dump(retrieval_results, dest, indent=2)

    print("Start generating responses.")
    generation_results = rag.generate(dataset=retrieval_results, with_context=config["with_context"])

    with open(config["generation_output_file"], "w", encoding="utf-8") as dest:
        json.dump(generation_results, dest, indent=2)

    metrics = compute_evaluation_metrics(dataset=generation_results)

    df_results = pd.DataFrame(metrics)
    df_results.to_csv(f"data/evaluation/validation_effectiveness/{config_name}_metrics.csv", index=False)


if __name__ == "__main__":
    run_evaluation()