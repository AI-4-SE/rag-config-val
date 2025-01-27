import toml
import argparse
import pandas as pd
from typing import List

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default="config.toml")
    return parser.parse_args()

def load_config(config_file: str) -> dict:
    """Load config from TOML file."""
    with open("config.toml", "r") as f:
        config = toml.load(f)
    return config

def build_index(index_name: str, documents: list):
    # build index if index not exists
    ## index documents
    # return index if index exists
    pass

def init_retriever():
    # init retriever
    return None

def init_rag():
    # init rag
    return None

def init_generators(generator_names: List[str]):
    # init generator
    return None

def init_rag(index, retriever, generators):
    # init rag
    return None


def run_evaluation():
    # parse args
    args = parse_args()

    # load config
    config = load_config(config_file=args.config_file)

    # build index
    index = build_index(
        index_name=config["indexing"]["index_name"], 
        documents=[
            config["indexing"]["urls"], 
            config["indexing"]["data_dir"], 
            config["indexing"]["github"]
        ]
    )

    # init retriever
    retriever = init_retriever()

    # init generator
    generators = init_generators()

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