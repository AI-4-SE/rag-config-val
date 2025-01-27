from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from src.util import load_config
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.github import GithubRepositoryReader, GithubClient
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.readers.web import SimpleWebPageReader
from llama_index.core import Settings, Document, SimpleDirectoryReader
from typing import List
import argparse
import os
import requests
import traceback


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default="config.toml")
    parser.add_argument("--env_file", type=str, default=".env")
    return parser.parse_args()


def get_documents_from_github(project_names: List) -> List[Document]:
    """
    Get documents from GitHub repositories.
    """
    documents = []
    for project_name in project_names:
        print(f"Start scraping the repository of {project_name}.")
        response = requests.get(f"https://api.github.com/search/repositories?q={project_name}")
        response.raise_for_status()

        data = response.json()

        if data['total_count'] > 0:
            owner = data["items"][0]["owner"]["login"]
            branch = data["items"][0]["default_branch"]
            repo_name = data['items'][0]["name"]
        else:
            return []
                        
        try:
            github_client = GithubClient(
                github_token=os.getenv(key="GITHUB_TOKEN"), 
                verbose=True
            )

            docs = GithubRepositoryReader(
                github_client=github_client,
                owner=owner,
                repo=repo_name,
                use_parser=False,
                verbose=False,
                filter_file_extensions=(
                    [
                        ".xml",
                        ".properties",
                        ".yml",
                        "Dockerfile",
                        ".json",
                        ".ini",
                        ".cnf",
                        ".toml",
                        ".conf",
                        ".md"
                    ],
                    GithubRepositoryReader.FilterType.INCLUDE,
                ),
            ).load_data(branch=branch)    
            for doc in docs:
                doc.metadata["index_name"] = "github"
            documents += docs
        except Exception:
            print.info("Error occurred while scraping Github.")
            print(traceback.format_exc)


def get_documents_from_dir(data_dir: str) -> List[Document]:
    """
    Get documents from data directory.
    """
    documents = SimpleDirectoryReader(input_dir=data_dir, recursive=True).load_data()
    for doc in documents:
        doc.metadata["index_name"] = "so-posts"
    return documents


def get_documents_from_urls(self, urls: List[str]) -> List[Document]:
    """
    Get documents from urls.
    """
    documents = SimpleWebPageReader(html_to_text=True).load_data(urls)
    for doc in documents:
        doc.metadata["index_name"] = "tech-docs"
    return documents


def run_ingestion():
    # parse args 
    args = parse_args()
    
    # load env variables
    load_dotenv(dotenv_path=args.env_file)

    # load config
    config = load_config(config_file=args.config_file)
    index_name = config["indexing"]["index_name"]
    dimension = config["indexing"]["dimension"]
    chunk_size = config["indexing"]["chunk_size"]
    chunk_overlap = config["indexing"]["chunk_overlap"]

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
        print(f"Index: '{index_name}' already exists")

    pinecone_index = pinecone_client.Index(index_name)

    # create vector store
    vector_store = PineconeVectorStore(
        pinecone_index=pinecone_index,
        add_sparse_vector=True
    )

    # get documents
    documents = []
    documents += get_documents_from_dir(data_dir=config["indexing"]["data_dir"])
    #documents += get_documents_from_github(project_names=config["indexing"]["github"])
    #documents += get_documents_from_urls(urls=config["indexing"]["urls"])

    # create text parser
    text_parser = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    embed_model = embed_model = HuggingFaceEmbedding(
        "Alibaba-NLP/gte-Qwen2-7B-instruct", 
        trust_remote_code=True
    )

    # build list of transformations
    transformations = [text_parser, embed_model]

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
    

    # nodes = text_parser.get_nodes_from_documents(documents=documents)

    # # get embeddings for nodes
    # for node in nodes:
    #     node_embedding = Settings.embed_model.get_text_embedding(
    #         node.get_content(metadata_mode="all")
    #     )
    #     node.embedding = node_embedding

    # vector_store.add(nodes=nodes)

if __name__ == "__main__":
    run_ingestion()