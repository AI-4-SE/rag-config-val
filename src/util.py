from llama_index.embeddings.openai import OpenAIEmbedding, OpenAIEmbeddingModelType
from llama_index.embeddings.ollama import OllamaEmbedding 
from llama_index.embeddings.huggingface import HuggingFaceEmbedding 
from llama_index.readers.web import SimpleWebPageReader
from llama_index.readers.github import GithubRepositoryReader, GithubClient
from llama_index.core import Settings, SimpleDirectoryReader, Document
from sentence_transformers import SentenceTransformer
from llama_index.llms.openai import OpenAI
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.prompts import CfgNetPromptSettings
from src.data import Dependency
from typing import Dict, Optional, List
import numpy as np
import requests
import traceback
import os
import toml


DIMENSION = {
    "openai": 1536,
    "qwen": 3584,
    "sfr": 4096
}

def get_embedding_dimension(embed_model_name: str) -> int:
    dimension = DIMENSION[embed_model_name]

    if not isinstance(dimension, int):
        raise Exception(f"Dimension has to be an integer and not of type {type(dimension)}")

    return dimension


def set_embedding(embed_model_name: str) -> None:
    """
    Set embedding model.
    """
    if embed_model_name == "openai":
        print(f"Set OpenAI Embedding with key: {os.getenv(key='OPENAI_KEY')}")
        Settings.embed_model = OpenAIEmbedding(
            api_key=os.getenv(key="OPENAI_KEY"),
            model=OpenAIEmbeddingModelType.TEXT_EMBED_ADA_002
        )
    elif embed_model_name == "ollama":
        print("Set Ollama Embedding.")
        Settings.embed_model = OllamaEmbedding(
            model_name=embed_model_name
        )
    elif embed_model_name == "qwen":
        print("Set Qwen Embedding.")
        Settings.embed_model = HuggingFaceEmbedding(
            "Alibaba-NLP/gte-Qwen2-7B-instruct", 
            trust_remote_code=True
        )
    else:
        raise Exception("Embedding model has to be set.")


def set_llm(inference_model_name: Optional[str]) -> None: 
    """
    Set inference model.
    """
    if not inference_model_name:
        Settings.llm = None
        return
    elif inference_model_name.startswith("gpt"):
        Settings.llm = OpenAI(
            model=inference_model_name, 
            api_key=os.getenv(key="OPENAI_KEY"),
            api_base=os.getenv(key="BASE_URL")
        )
    elif inference_model_name.startswith("llama"):
        Settings.llm = Ollama(
            model=inference_model_name,
            request_timeout=90.0
    )
    else:
        raise Exception("Embedding model has to be set.")


def load_config(config_file: str) -> Dict:
    """
    Load config from TOML file.
    """
    if not config_file.endswith(".toml"):
            raise Exception("Config file has to be a TOML file.")
        
    with open(config_file, "r", encoding="utf-8") as f:
        config = toml.load(f)
        
    return config

    
def get_projet_description(project_name: str) -> str:
    """
    Read and return project-specific information.
    """
    with open(f"../data/project_info/{project_name}.txt", "r", encoding="utf-8") as src:
        content = src.read().strip()

    return content


def load_shots() -> List[str]:
    """
    Load shots from the shot pool.
    """
    shot_pool_path = "../data/shot_pool/"
    shot_files = [shot_pool_path + x for x in os.listdir(shot_pool_path) if ".csv" not in x]
    shots = []
    for shot_file in shot_files:
        with open(shot_file, "r", encoding="utf-8") as src:
            shot_content = src.read()
            shots.append(shot_content.strip())

    return shots


def transform(entry) -> Dependency:
    dependency = Dependency(
        project=entry["project"],
        option_name=entry["option_name"],
        option_value=entry["option_value"],
        option_type=entry["option_type"].split(".")[-1],
        option_file=entry["option_file"],
        option_technology=entry["option_technology"],
        dependent_option_name=entry["dependent_option_name"],
        dependent_option_value=entry["dependent_option_value"],
        dependent_option_type=entry["dependent_option_type"].split(".")[-1],
        dependent_option_file=entry["dependent_option_file"],
        dependent_option_technology=entry["dependent_option_technology"]
    )
    return dependency




def get_most_similar_shot(shots: List[str], dependency: Dependency) -> str:
    """
    Return most similar shot based on the given dependency.
    """
    task_str = CfgNetPromptSettings.get_task_str(dependency=dependency)
    
    all = shots + [task_str]

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all)

    cosine_sim_matrix = cosine_similarity(tfidf_matrix)

    task_similarities = cosine_sim_matrix[-1, :-1]
    most_similar_index = np.argmax(task_similarities)
    most_similar_shot = shots[most_similar_index]

    return most_similar_shot


def get_most_similar_shots(shots: List[str], dependency: Dependency) -> str:
    """
    Return most similar shot based on the given dependency.
    """
    task_str = CfgNetPromptSettings.get_task_str(dependency=dependency)
    
    all = shots + [task_str]

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all)

    cosine_sim_matrix = cosine_similarity(tfidf_matrix)

    task_similarities = cosine_sim_matrix[-1, :-1]
    top_two_indices = np.argsort(task_similarities)[-2:][::-1]
    most_similar_string1 = shots[top_two_indices[0]]
    most_similar_string2 = shots[top_two_indices[1]]


    return (most_similar_string1, most_similar_string2)



def get_documents_from_github(project_name: str) -> List[Document]:
    """
    Get documents from GitHub repository.
    """
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

        documents = GithubRepositoryReader(
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

        for d in documents:
            d.metadata["index_name"] = "github"

        return documents
    except Exception:
        print(traceback.format_exc)
        return []

def get_documents_from_dir( data_dir: str) -> List[Document]:
    """
    Get documents from data directory.
    """
    documents = SimpleDirectoryReader(input_dir=data_dir, recursive=True).load_data()
    
    for doc in documents:
        doc.metadata["index_name"] = "so-posts"
    
    return documents

def get_documents_from_urls(urls: List[str]) -> List[Document]:
    """
    Get documents from urls.
    """
    documents = SimpleWebPageReader(html_to_text=True).load_data(urls)
    
    for doc in documents:
        doc.metadata["index_name"] = "tech-docs"

    return documents