from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.readers.github import GithubRepositoryReader, GithubClient
from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.web import SimpleWebPageReader
from llama_index.core import Settings, Document, SimpleDirectoryReader
from llama_index.core.ingestion import IngestionPipeline
from src.utils import get_documents_from_dir, get_documents_from_github, get_documents_from_urls
from pinecone import Pinecone, ServerlessSpec
from duckduckgo_search import DDGS
from typing import List, Dict
from rich.logging import RichHandler
import os
import requests
import traceback
import backoff
import logging


logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler()],
)


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

@backoff.on_exception(
    backoff.expo,
    Exception,
    max_tries=5
)
def get_documents_from_github(project_name: str) -> List[Document]:
    """
    Get documents from GitHub repositories.
    """
    documents = []
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
        return documents
    except Exception:
        print(f"Error occurred while scraping the project {project_name}")
        print(traceback.format_exc())


def get_documents_from_dir(data_dir: str) -> List[Document]:
    """
    Get documents from data directory.
    """
    print(f"Get data from directory: {data_dir}")
    documents = SimpleDirectoryReader(input_dir=data_dir, recursive=True).load_data()
    for doc in documents:
        doc.metadata["source"] = "so-posts"

    if documents:
        print("Data from directory successfully loaded.")
        return documents
    return []


def get_documents_from_urls(urls: List[str]) -> List[Document]:
    """
    Get documents from urls.
    """
    print("Get data from urls.")
    documents = SimpleWebPageReader(html_to_text=True).load_data(urls)
    for doc in documents:
        doc.metadata["source"] = "tech-docs"
    
    if documents:
        print("Data from urls successfully loaded.")
        return documents
    return []


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
    return [f"{node.ref_doc_id}#{node.node_id}" for node in nodes]


def add_nodes(documents: List[Document], vector_store: PineconeVectorStore) -> List:
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
    ids = [f"{node.ref_doc_id}#{node.node_id}" for node in nodes]

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
    
    return ids


def build_static_index(ingestion_config: Dict):
    """Build static index for context data."""
    # define data sources
    urls = ingestion_config["data"]["urls"]
    data_dir = ingestion_config["data"]["data_dir"]  
    github_project_names = ingestion_config["data"]["github"]

    # Create index for static context data
    for index_name, values in ingestion_config["indices"].items():
        logging.info(f"Run ingestion for index: {index_name}")
        
        index_name = values["index_name"]
        dimension = values["dimension"]

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

        # get pinecone index
        pinecone_index = pinecone_client.Index(index_name)

        # check if index contains data
        stats = pinecone_index.describe_index_stats()
        vector_count = stats.get("total_vector_count", 0)

        if vector_count > 0:
            # delete all vectors in index
            if ingestion_config["delete_all"]:
                pinecone_index.delete(delete_all=True)
                logging.info(f"Vectors in index '{index_name}' has been deleted.")
            else:
                return pinecone_client.Index(index_name)
        else:
            logging(f"Index '{index_name}' is empty. Start ingestion.")

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
            chunk_size=ingestion_config["chunk_size"],
            chunk_overlap=ingestion_config["chunk_overlap"]
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

        return pinecone_client.Index(index_name)


def build_dynamic_index(ingestion_config: Dict):
    # Create index for dynamic context data
    for index_name, values in ingestion_config["indices"].items():
        logging.info(f"Create index for dynamic data: {index_name}")
        
        index_name = f"{values['index_name']}-web"
        dimension = values["dimension"]

        # create Pinecone client
        pinecone_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

        # create index if not exist
        if index_name not in pinecone_client.list_indexes().names():
            logging.info(f"Create index: {index_name}")
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
            logging.info(f"Index: '{index_name}' already exists. Continue with next index.")
    
    return pinecone_client.Index(index_name)