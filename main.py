from src.rag import RAG
from src.retriever import Retriever
from src.generator import GeneratorFactory
from src.ingestion import build_static_index, build_dynamic_index
from src.utils import load_config, transform, set_embedding, set_llm
from dotenv import load_dotenv
from rich.logging import RichHandler
import pandas as pd
import argparse
import logging


logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler()],
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--env_file", type=str, default=".env")
    parser.add_argument("--ingestion_config_file", type=str)
    parser.add_argument("--rag_config_file", type=str)
    return parser.parse_args()


def main():
    # parse args
    args = parse_args()

    # load env variables
    load_dotenv(args.env_file)

    # load config
    ingestion_config = load_config(args.ingestion_config_file)
    config = load_config(args.rag_config_file)

    # set inference and embedding moddels
    set_llm(inference_model_name=config["inference_models"][0])
    set_embedding(embed_model_name=config["embedding_model"])

    # build static and dynamic index
    static_index = build_static_index(ingestion_config=ingestion_config)
    dynamic_index = build_dynamic_index(ingestion_config=ingestion_config)
   
    # get retriever
    retriever = Retriever(
        rerank=config["rerank"],
        top_n=config["top_n"],
        top_k=config["top_k"],
        alpha=config["alpha"]
    )

    # define genrators
    generators = [
        GeneratorFactory().get_generator(model_name=inference_model_name, temperature=config["temperature"]) 
        for inference_model_name in config["inference_models"]
    ]

    # init rag
    rag = RAG(
        static_index=static_index,
        dynamic_index=dynamic_index,
        retriever=retriever,
        generators=generators,
    )  

    # load data
    data = pd.read_csv(args.data_file)

    generations = []

    # validate dependencies and generate responses
    for _ , row in data.iterrows():
        dependency = transform(entry=row)
        response = rag.generate(
            dependency=dependency, 
            with_context=config["with_context"],
            enable_websearch=config["web_search_enabled"],
            advanced=config["advanced"],
        )

        generations.append(response)

    print(generations)


if __name__ ==  "__main__":
    main()