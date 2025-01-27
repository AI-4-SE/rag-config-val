import toml
import argparse
import pandas as pd
from typing import List
from index_builder import create_index, index_documents
from src.util import get_documents_from_dir, get_documents_from_urls, get_documents_from_github
from llama_index.vector_stores.pinecone import PineconeVectorStore

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default="config.toml")
    return parser.parse_args()


def load_config(config_file: str) -> dict:
    """Load config from TOML file."""
    with open(config_file, "r") as f:
        config = toml.load(f)
    return config


def build_index(index_name: str, documents: list, dimension: int, chunk_size: int, chunk_overlap: int):
    # build index if index not exist
    index = create_index(index_name=index_name, dimension=dimension)

    vector_store = PineconeVectorStore(index=index)

    index_documents(
        vector_store=vector_store,
        documents=documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    # return index if index exists
    pass


def init_rag(index, retriever, generators):
    # init rag
    return None


def run_evaluation():
    # parse args
    args = parse_args()

    # load config
    config = load_config(config_file=args.config_file)

    documents = []
    documents += get_documents_from_dir(data_dir=config["indexing"]["data_dir"])

    # build index
    index = build_index(
        index_name=config["indexing"]["index_name"], 
        documents=documents
    )

    # init retriever
    retriever = None

    # init generator
    generators = ""

    # init rag
    rag = init_rag(
        index=index,
        retriever=retriever,
        generators=generators
    )

    # load dataset
    data_file_path = config["evaluation"]["data_file"]
    print(f"Loading dataset from: {data_file_path}")
    dataset = pd.read_csv(data_file_path)

    # evaluate

    

if __name__ == "__main__":
    run_evaluation()