
import argparse
import pandas as pd
import os
import json
from pinecone import Pinecone
from dotenv import load_dotenv
from src.retriever import Retriever
from src.utils import load_config, set_embedding, set_llm, transform
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import Settings, VectorStoreIndex
from tqdm import tqdm
from src.utils_ingestion import get_documents_from_web, add_nodes
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


def run_retrieval():
    print("Run retrieval.")
    # parse args
    args = parse_args()

    # load env variables
    load_dotenv(dotenv_path=args.env_file)

    # load config
    config = load_config(config_file=args.config_file)

    prompts = CfgNetPromptSettings
    pinecone_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    # set inference and embedding moddels
    set_llm(inference_model_name=config["inference_models"][0])
    set_embedding(embed_model_name=config["embedding_model"])

    # get index
    pinecone_index_static = pinecone_client.Index(config["index_name"])
    pinecone_index_dynamic = pinecone_client.Index(f"{config['index_name']}-web")
    pinecone_index_dynamic.delete(delete_all=True)

    # get static vector store
    vector_store_static = PineconeVectorStore(
        pinecone_index=pinecone_index_static,
        add_sparse_vector=True
    )
    
    # get dynamic vector store
    vector_store_dynamic = PineconeVectorStore(
        pinecone_index=pinecone_index_dynamic,
        add_sparse_vector=True
    )

    # get retriever
    retriever = Retriever(
        rerank=config["rerank"],
        top_k=config["top_k"],
        top_n=config["top_n"],
        alpha=config["alpha"]
    )

    dataset=pd.read_csv(config["data_file"])[:5]

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
                    num_websites=config["num_websites"]
                )

                print(f"Add {len(web_documents)} web documents to vector store.")
                # add nodes to vector store and store their ids
                web_node_ids = add_nodes(
                    documents=web_documents,
                    vector_store=vector_store_dynamic
                )

                # retrieve relavant nodes from web index
                retrieved_dynamic_context = retriever.retrieve(
                    vector_store=vector_store_dynamic,
                    retrieval_str=retrieval_str
                )

                # delete nodes from web index
                vector_store_dynamic.delete_nodes(node_ids=web_node_ids)

            # retrieve relavant nodes
            retrieved_static_context = retriever.retrieve(
                vector_store=vector_store_static,
                retrieval_str=retrieval_str
            )

            # merge static and dynamic context and rerank
            reranked_nodes = retriever.rerank_nodes(
                nodes=retrieved_static_context + retrieved_dynamic_context,
                retrieval_str=retrieval_str
            )

            # defines context to append to row dict
            context = [
                {
                    "text": node.get_content(),
                    "score": str(node.get_score()),
                    "source": node.metadata["source"],
                    "id": node.node_id
                } for node in reranked_nodes
            ]

            row_dict.update({"context": context})

            retrieval_results.append(row_dict)
                
    except Exception as e:
        print(f"An error occurred: {e}")
        with open(config["retrieval_output_file"], "w", encoding="utf-8") as dest:
            json.dump(retrieval_results, dest, indent=2)
    
    with open(config["retrieval_output_file"], "w", encoding="utf-8") as dest:
        json.dump(retrieval_results, dest, indent=2)


if __name__ == "__main__":
    run_retrieval()