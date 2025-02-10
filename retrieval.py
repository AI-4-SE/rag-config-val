
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
import backoff



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str)
    parser.add_argument("--env_file", type=str, default=".env")
    return parser.parse_args()


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

    stats = pinecone_index_dynamic.describe_index_stats()
    vector_count = stats.get("total_vector_count", 0)
    if vector_count > 0:
        print("Delete all vectors from dynamic index.")
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

    dataset=pd.read_csv(config["data_file"])

    retrieval_results = []

    with open("data/evaluation/failed.json", "r", encoding="utf-8") as src:
        failed = json.load(src)

    entries_failed = []
    
    for index, row in tqdm(dataset.iterrows(), total=len(dataset), desc="Processing dependencies"):
        try:

            # turn row data into dict
            row_dict = row.to_dict()

            #if not row_dict["index"] in failed:
            #    print(f"Skip sample {row_dict['index']}")
            #    continue
            
            print(f"Process sample {row_dict['index']}")

            # transform row data into a dependency
            dependency = transform(row)

            # get retrieval prompt
            retrieval_str = prompts.get_retrieval_prompt(dependency=dependency)

            retrieved_dynamic_context = []
            retrieved_static_context = []

            # scrape the web
            if config["web_search_enabled"]:
                print(f"Start scraping the web for dependency {index}")
                web_documents = get_documents_from_web(
                    query_str=retrieval_str,
                    num_websites=config["num_websites"]
                )

                print(f"Add {len(web_documents)} web documents to vector store.")
                # add nodes to vector store and store their ids
                if web_documents:
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
                else:
                    print("No web documents found.")

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
            print(f"An error occurred for sample: {index}")
            print(f"Error: {e}")
            retrieval_results.append(row_dict)
            entries_failed.append(row_dict["index"])
            continue

    with open(config["retrieval_output_file"], "w", encoding="utf-8") as dest:
        json.dump(retrieval_results, dest, indent=2)

    with open("data/evaluation/failed.json", "w", encoding="utf-8") as dest:
        json.dump(entries_failed, dest, indent=2)


if __name__ == "__main__":
    run_retrieval()