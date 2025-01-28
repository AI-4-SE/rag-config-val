from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from src.util import load_config, set_embedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.github import GithubRepositoryReader, GithubClient
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.readers.web import SimpleWebPageReader
from llama_index.core import Settings, Document, SimpleDirectoryReader
from duckduckgo_search import DDGS
from typing import List
import argparse
import os
import requests
import traceback
import backoff


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default="../config.toml")
    parser.add_argument("--env_file", type=str, default="../.env")
    return parser.parse_args()


@backoff.on_exception(
    backoff.expo,
    Exception,
    max_tries=5
)
def get_documents_from_web(query_str: str, num_websites: int = 3) -> List[Document]:
    """Get documents from web and transform them into documents."""
    results = DDGS().text(query_str, max_results=num_websites)
    urls = []
    for result in results:
        url = result["href"]
        urls.append(url)

    print(f"Number of urls to scrape: {len(urls)}")

    documents = SimpleWebPageReader(html_to_text=True).load_data(urls)

    for doc in documents:
        doc.metadata["source"] = "web"

    return documents

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
                doc.metadata["source"] = "github"
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
        doc.metadata["source"] = "so-posts"
    return documents


def get_documents_from_urls(self, urls: List[str]) -> List[Document]:
    """
    Get documents from urls.
    """
    documents = SimpleWebPageReader(html_to_text=True).load_data(urls)
    for doc in documents:
        doc.metadata["source"] = "tech-docs"
    return documents


def add_nodes_to_vector_store(documents: List[Document], vector_store: PineconeVectorStore) -> List:
    """
    Parse documents into nodes and add them to a vector store.

    Args:
        documents (list): List of documents to add to the vector store
        vector_store (PineconeVectorStore): The initialized Pinecone vector store.
        embed:model: Embedding model to create the embeddings.
        text_parser: Technique to splite documents into smaller chunks.

    Return:
        List of node ids.
    """

    # Create text parser
    text_parser = SentenceSplitter(
        chunk_size=256,
        chunk_overlap=10
    )

    # Parse documents into nodes
    nodes = text_parser.get_nodes_from_documents(documents=documents)

    # Embed nodes and add them to the vector store
    for node in nodes:
        # Generate embedding for the node's content
        node_embedding = Settings.embed_model.get_text_embedding(node.get_content(metadata_mode="all"))
        node.embedding = node_embedding

    vector_store.add(nodes)
    print(f"Nodes successfully added to vector store.")
    return [node.node_id for node in nodes]

def delete_nodes_by_metadata(vector_store: PineconeVectorStore, metadata_filter: dict):
    """
    Delete nodes from the vector store using a metadata filter.

    Args:
        vector_store (PineconeVectorStore): The initialized Pinecone vector store.
        metadata_filter (dict): Metadata filter to specify which nodes to delete.
    """
    # Delete nodes using metadata filter
    vector_store._pinecone_index.delete(filter=metadata_filter, namespaces="")
    print(f"Successfully deleted nodes with metadata: {metadata_filter}")

def delete_nodes_by_ids(vector_store: PineconeVectorStore, ids: List):
    """
    Delete nodes from the vector store via ids.

    Args:
        vector_store (PineconeVectorStore): The initialized Pinecone vector store.
        ids (list): list of node ids.
    """
    # Delete nodes using metadata filter
    vector_store._pinecone_index.delete(ids=ids)
    print("Successfully deleted nodes via ids.")


def run_ingestion():
    # parse args 
    args = parse_args()
    
    # load env variables
    load_dotenv(dotenv_path=args.env_file)

    # load config
    config = load_config(config_file=args.config_file)
    index_name = config["indexing"]["index_name"]
    dimension = config["indexing"]["dimension"]
    embed_model_name = config["indexing"]["embedding_model"]
    
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
        print(f"Index: '{index_name}' already exists")

    # get pinecone index
    pinecone_index = pinecone_client.Index(index_name)

    # check if index contains data
    stats = pinecone_index.describe_index_stats()
    vector_count = stats.get("total_vector_count", 0)

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

    # get documents
    documents = []
    documents += get_documents_from_dir(data_dir=config["indexing"]["data_dir"])
    #documents += get_documents_from_github(project_names=config["indexing"]["github"])
    #documents += get_documents_from_urls(urls=config["indexing"]["urls"])

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
    

if __name__ == "__main__":
    run_ingestion()