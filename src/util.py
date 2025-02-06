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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from collections import Counter
import random
import numpy as np
import requests
import traceback
import os
import json
import toml


def set_embedding(embed_model_name: str) -> None:
    """
    Set embedding model.
    """
    if embed_model_name == "openai":
        print(f"Set OpenAI Embedding")
        Settings.embed_model = OpenAIEmbedding(
            api_key=os.getenv(key="OPENAI_KEY_API"),
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

def get_dominant_element(elements: List) -> str:
    # Calculate the TF-IDF scores
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(elements)
    
    # Cluster the reasons using DBSCAN
    # These parameters can be tuned based on the nature of the data
    dbscan = DBSCAN(eps=0.3, min_samples=2, metric='cosine').fit(X)
    labels = dbscan.labels_
    # If all reasons are considered as noise by DBSCAN, just return a random reason
    if len(set(labels)) == 1 and -1 in labels:
        return random.choice(elements)
    
    # Find the dominant cluster
    counter = Counter(labels)
    if -1 in counter:  # Removing the noise label
        del counter[-1]
    dominant_cluster_label = counter.most_common(1)[0][0]
    
    # Get a random reason from the dominant cluster
    dominant_cluster_reasons = [reason for idx, reason in enumerate(elements) if labels[idx] == dominant_cluster_label]
    return random.choice(dominant_cluster_reasons)

def get_dominat_response(responses: List[Dict]) -> List:
    # Get dominant responses
    votes = [response["isDependency"] for response in responses]
    votes_counter = Counter(votes)
    dominant_vote = votes_counter.most_common(1)[0][0]

    # Filter dominant responses as dictionaries
    dominant_responses = [response for response in responses if response["isDependency"] == dominant_vote]

    # Extract textual reasons for clustering
    text_reasons = [str(response) for response in dominant_responses]

    # Calculate the TF-IDF scores
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(text_reasons)

    # Cluster the reasons using DBSCAN
    dbscan = DBSCAN(eps=0.3, min_samples=2, metric='cosine').fit(X)
    labels = dbscan.labels_

    # If all reasons are noise, return a random dictionary from dominant responses
    if len(set(labels)) == 1 and -1 in labels:
        return random.choice(dominant_responses)

    # Find the dominant cluster
    counter = Counter(labels)
    if -1 in counter:  # Removing the noise label
        del counter[-1]
    dominant_cluster_label = counter.most_common(1)[0][0]

    # Get a random reason from the dominant cluster
    dominant_cluster_reasons = [
        dominant_responses[idx] for idx, reason in enumerate(text_reasons) if labels[idx] == dominant_cluster_label
    ]

    return random.choice(dominant_cluster_reasons)

def compute_evaluation_metrics(dataset: List) -> list:
    """
    Compute the evaluation metrics, including precision, recall and F1 score.

    Args:
        dataset: List
            The dataset to compute the evaluation metrics for.

    Returns:
        List of the evaluation results.
    """
    models = list(dataset[0]["generations"].keys())

    metrics = []

    for model in models:

        print("Model: ", model)

        true_positives = []
        true_negatives = []
        false_positives = []
        false_negatives = []
        accuracy_count = []
        skipped = 0

        for entry in dataset:
            final_rating = entry["final_rating"]
            model_response = entry["generations"][model]
            isDependency = model_response["isDependency"]

            if isinstance(isDependency, str) and isDependency == "None":
                skipped += 1
                continue

            # TP: The LLM validates a dependency as correct and the dependency is correct
            if isDependency and final_rating:
                accuracy_count.append(1)
                true_positives.append(1)
                
            # FP: The LLM validates a dependency as correct, but the dependency is actually incorrect
            if isDependency and not final_rating:
                accuracy_count.append(0)
                false_positives.append(1)

            # TN: The LLM validates a dependency as incorrect and the dependency is incorrect
            if not isDependency and not final_rating:
                accuracy_count.append(1)
                true_negatives.append(1)

            # FN: The LLM validates a dependency as incorrect, but the dependency is actually correct
            if not isDependency and final_rating:
                accuracy_count.append(0)
                false_negatives.append(1)

        tp = sum(true_positives)
        fp = sum(false_positives)
        fn = sum(false_negatives)
        tn = sum(true_negatives)
        accuracy = sum(accuracy_count)/len(accuracy_count)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        print("TP: ", tp)
        print("FP: ", fp)
        print("FN: ", fn)
        print("TN: ", tn)

        metrics.append({
            "model": model,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "accuracy": accuracy,
            "skipped": skipped
        })

    return metrics
