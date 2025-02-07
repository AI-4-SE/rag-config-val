
import argparse
import pandas as pd
import os
import json
from pinecone import Pinecone
from dotenv import load_dotenv
from src.retriever import Retriever
from src.utils import load_config, set_embedding, set_llm, transform
from llama_index.vector_stores.pinecone import PineconeVectorStore
from tqdm import tqdm
from src.utils_ingestion import get_documents_from_web, add_nodes_to_vector_store
from src.prompts import CfgNetPromptSettings
import time



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


def wait_for_vector_count_increase(pinecone_index, initial_count, timeout=60, interval=5):
    """
    Wait for the vector count to increase.

    Args:
        pinecone_index: The Pinecone index instance.
        initial_count: The initial vector count.
        timeout: Maximum time to wait in seconds.
        interval: Time to wait between checks in seconds.
    """
    start_time = time.time()
    while time.time() - start_time < timeout:
        current_count = pinecone_index.describe_index_stats().get("total_vector_count", 0)
        if current_count > initial_count:
            print("Vector count increased.")
            return 
        time.sleep(interval)
    raise TimeoutError("Timeout waiting for vector count to increase")

def run_retrieval():
    print("Run retrieval.")
    # parse args
    args = parse_args()

    # load env variables
    load_dotenv(dotenv_path=args.env_file)

    # load config
    config = load_config(config_file=args.config_file)

    prompts = CfgNetPromptSettings

    # set inference and embedding moddels
    set_llm(inference_model_name=config["inference_models"][0])
    set_embedding(embed_model_name=config["embedding_model"])

    # get index
    pinecone_index = get_index(index_name=config["index_name"])
    vector_count = pinecone_index.describe_index_stats().get("total_vector_count", 0)
    print(f"Current vector count: {vector_count}")

    # get vector store
    vector_store = PineconeVectorStore(
        pinecone_index=pinecone_index,
        add_sparse_vector=True
    )

    # get retriever
    retriever = Retriever(
        vector_store=vector_store,
        rerank=config["rerank"],
        top_n=config["top_n"]
    )

    dataset=pd.read_csv(config["data_file"])[:10]

    retrieval_results = []

    try:
        for index, row in tqdm(dataset.iterrows(), total=len(dataset), desc="Processing dependencies"):

            # turn row data into dict
            row_dict = row.to_dict()

            # transform row data into a dependency
            dependency = transform(row)

            # get retrieval prompt
            retrieval_str = prompts.get_retrieval_prompt(dependency=dependency)

            # scrape the web
            if config["web_search_enabled"]:
                print(f"Start scraping the web for dependency {index}")
                web_documents = get_documents_from_web(
                    query_str=retrieval_str,
                    num_websites=3
                )

                print(f"Add {len(web_documents)} web documents to vector store.")
                # add nodes to vector store and store their ids
                web_node_ids = add_nodes_to_vector_store(
                    documents=web_documents,
                    vector_store=vector_store
                )

                # Wait for the vector count to increase
                wait_for_vector_count_increase(pinecone_index, current_vector_count)

            # retrieve relavant nodes
            retrieved_nodes = retriever.retrieve(
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

            # delete nodes from websearch if enabled
            if config["web_search_enabled"]:
                vector_store.delete_nodes(node_ids=web_node_ids)

    except Exception as e:
        print(f"An error occurred: {e}")
        with open(config["retrieval_output_file"], "w", encoding="utf-8") as dest:
            json.dump(retrieval_results, dest, indent=2)
    
    with open(config["retrieval_output_file"], "w", encoding="utf-8") as dest:
        json.dump(retrieval_results, dest, indent=2)

    final_vector_count = pinecone_index.describe_index_stats().get("total_vector_count", 0)
    print(f"Final vector count: {final_vector_count}")

if __name__ == "__main__":
    run_retrieval()