import argparse
import tomllib
import mlflow
from typing import Dict
import os
from dotenv import load_dotenv
from src.utils import load_config as load_utils_config, set_embedding, set_llm, transform, load_shots, get_projet_description, get_most_similar_shots
from src.rag import RAG
from src.generator import GeneratorFactory
from src.prompts import Prompts
from tqdm import tqdm
import json

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default="configs/sensitivity.toml")
    parser.add_argument("--env_file", type=str, default=".env")
    return parser.parse_args()

def load_config(config_file: str):
    with open(config_file, "rb") as src:
        config = tomllib.load(src)

    return config

def run_sensitivity_analysis(config: Dict):
    # Load the data
    with open(config["data_file"], "r", encoding="utf-8") as src:
        data = json.load(src)

    print("Length of data:", len(data))

    shots = load_shots() if config["advanced"] else None


    for temperature in config["temperature"]:
        print(f"\nRunning analysis with temperature: {temperature}")
        for model_name in config["inference_models"]:
            # Create generator with current temperature
            generator = GeneratorFactory().get_generator(
                model_name=model_name, 
                temperature=temperature
            )

            entries_failed = []
            counter = 0

            for entry in tqdm(data, total=len(data), desc=f"Generating responses (model={model_name}, temp={temperature})"):
                try:
                    # transform row data into a dependency
                    dependency = transform(entry)

                    project_info = get_projet_description(project_name=dependency.project) if config["advanced"] and config["with_context"] else None
                    task_str = Prompts.get_task_str(dependency=dependency)
                    context_str = "\n\n".join([x["text"] for x in entry["context"]])
                    shot_str = "\n\n".join([shot for shot in get_most_similar_shots(shots, dependency)])

                    system_prompt = Prompts.get_system_str(
                        project_name=dependency.project,
                        project_info=project_info,
                        advanced=config["advanced"]
                    )

                    query_prompt = Prompts.get_query_str(
                        context_str=context_str,
                        task_str=task_str,
                        shot_str=shot_str,
                        advanced=config["advanced"]
                    )

                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": query_prompt}
                    ]

                    if "generations" not in entry:
                        entry["generations"] = {}

                    # Store generations with temperature as part of the key
                    temp_key = f"{model_name}_temp_{temperature}"
                    if not temp_key in entry["generations"]:
                        print(f"Generate response for entry {entry['index']} with model {model_name} and temperature {temperature}")
                        response, ratings= generator.generate_with_ratings(messages=messages)
                        # Add temperature to the response
                        response["temperature"] = temperature
                        response["ratings"] = ratings
                        entry["generations"][temp_key] = response
                    else:
                        print(f"Generation for entry {entry['index']} of model {model_name} with temperature {temperature} already exists. Skipping generation.")

                    counter += 1

                except Exception as e:
                    print(f"An error occurred for sample: {entry['index']}")
                    print(f"Error: {e}")
                    entries_failed.append(entry["index"])
                    continue

            # Save final reslts for this temperature
            with open(f"data/evaluation/sensitivity/test_dependencies_temp_{temperature}.json", "w", encoding="utf-8") as dest:
                json.dump(data, dest, indent=2)

            # Log results to MLflow
            mlflow.log_artifact(f"data/evaluation/sensitivity/test_dependencies_temp_{temperature}.json")

    # Save final results for this temperature
    #with open(f"data/evaluation/sensitivity/test_dependencies_temp_all.json", "w", encoding="utf-8") as dest:
    #    json.dump(data, dest, indent=2)

        



def main():
    # parse args
    args = parse_args()

    # load env variable
    load_dotenv(dotenv_path=args.env_file)

    # load config
    config = load_config(config_file=args.config_file)
    config_name = os.path.basename(args.config_file).split(".")[0]

    mlflow.set_experiment(experiment_name="sensitivity_analysis")
    
    with mlflow.start_run(run_name=f"sensitivity_{config_name}"): 
        mlflow.log_params(config)
        mlflow.log_artifact(local_path=args.env_file)
        run_sensitivity_analysis(config=config)
        mlflow.log_artifact(local_path=config["data_file"])


if __name__ == "__main__":
    main()