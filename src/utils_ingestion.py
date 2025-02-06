from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.readers.github import GithubRepositoryReader, GithubClient
from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.web import SimpleWebPageReader
from llama_index.core import Settings, Document, SimpleDirectoryReader
from duckduckgo_search import DDGS
from typing import List
import os
import requests
import traceback
import backoff


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
            print(f"Error occurred while scraping the project {project_name}")
            print(traceback.format_exc())
            continue

    if documents:
        print("Data from github repositories successfully loaded.")
        return documents
    return []


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