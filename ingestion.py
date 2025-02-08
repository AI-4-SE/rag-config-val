from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from src.utils import load_config, set_embedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.settings import Settings
from src.utils_ingestion import get_documents_from_github, get_documents_from_dir, get_documents_from_urls
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default="configs/ingestion.toml")
    parser.add_argument("--env_file", type=str, default="../.env")
    return parser.parse_args()


def run_ingestion():
    # parse args 
    args = parse_args()
    
    # load env variables
    load_dotenv(dotenv_path=args.env_file)

    # load config
    config = load_config(config_file=args.config_file)

    urls = config["data"]["urls"]
    data_dir = config["data"]["data_dir"]  
    github_project_names = config["data"]["github"]

    # Create index for static context data
    for index_name, values in config["indices"].items():
        print(f"Run ingestion for index: {index_name}")
        
        index_name = values["index_name"]
        dimension = values["dimension"]
        embed_model_name = values["embedding_model"]

        # set embedding model
        set_embedding(embed_model_name=embed_model_name)

        # create Pinecone client
        pinecone_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

        # create index if not exist
        if index_name not in pinecone_client.list_indexes().names():
            print(f"Create index: {index_name}")
            pinecone_client.create_index(
                name=index_name,
                dimension=dimension,
                metric="dotproduct",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
        else:
            print(f"Index: '{index_name}' already exists. Continue with next index.")
            continue

        # get pinecone index
        pinecone_index = pinecone_client.Index(index_name)

        # check if index contains data
        stats = pinecone_index.describe_index_stats()
        vector_count = stats.get("total_vector_count", 0)

        # delete all vectors in index
        if vector_count > 0:
            pinecone_index.delete(delete_all=True)
            print(f"Vectors in index '{index_name}' has been deleted.")
        else:
            print(f"Index '{index_name}' is empty.")

        # create vector store
        vector_store = PineconeVectorStore(
            pinecone_index=pinecone_index,
            add_sparse_vector=True
        )

        # store documents from different sources
        documents = []

        # get documents from a directory
        documents += get_documents_from_dir(data_dir=data_dir)

        # get documents from github repositories
        for project_name in github_project_names:
            documents += get_documents_from_github(project_name=project_name)

        # get documents from urls
        documents += get_documents_from_urls(urls=urls)

        # create text parser
        text_parser = SentenceSplitter(
            chunk_size=256,
            chunk_overlap=10
        )

        # build list of transformations
        transformations = [text_parser, Settings.embed_model]

        # create ingestion pipeline
        pipeline = IngestionPipeline(
            transformations=transformations,
            vector_store=vector_store
        )

        # run ingestion pipeline
        pipeline.run(
            documents=documents,
            show_progress=True
        )

    # Create index for dynamic context data
    for index_name, values in config["indices"].items():
        print(f"Create index for dynamic data: {index_name}")
        
        index_name = f"{values["index_name"]}-web"
        dimension = values["dimension"]
        embed_model_name = values["embedding_model"]

        # set embedding model
        set_embedding(embed_model_name=embed_model_name)

        # create Pinecone client
        pinecone_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

        # create index if not exist
        if index_name not in pinecone_client.list_indexes().names():
            print(f"Create index: {index_name}")
            pinecone_client.create_index(
                name=index_name,
                dimension=dimension,
                metric="dotproduct",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
        else:
            print(f"Index: '{index_name}' already exists. Continue with next index.")
            continue
    

if __name__ == "__main__":
    run_ingestion()