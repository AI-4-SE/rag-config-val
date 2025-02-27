from src.utils import load_config, set_embedding, set_llm, transform, load_shots, get_projet_description, get_most_similar_shots
from src.rag import RAG
from src.generator import GeneratorFactory
from src.prompts import Prompts
from dotenv import load_dotenv
from tqdm import tqdm
from typing import Dict
import pandas as pd
import argparse
import os
import json
import mlflow


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str)
    parser.add_argument("--env_file", type=str, default=".env")
    return parser.parse_args()

def run_generation(config: Dict, config_name: str):

    # set inference and embedding moddels
    set_llm(inference_model_name=config["inference_models"][0])
    set_embedding(embed_model_name=config["embedding_model"])

    # define generators
    generators = [
        GeneratorFactory().get_generator(model_name=inference_model_name, temperature=config["temperature"]) 
        for inference_model_name in config["inference_models"]
    ]

    print(f"{len(generators)} generators initialized.")

    with open(config["generation_file"], "r", encoding="utf-8") as src:
        data = json.load(src)

    print("Length of data:", len(data))

    if config["with_context"] and not config["advanced"]:
        print("Start generation with context.")

    if config["advanced"] and config["with_context"]:
        print("Start advanced generation with context.")

    if config["advanced"] and not config["with_context"]:
        print("Start advanced generation without context.")
    
    if not config["advanced"] and not config["with_context"]:
        print("Start generation without context.")

    shots = load_shots() if config["advanced"] else None

    for generator in generators:

        # reset generation results and failed entries
        #generation_results = []
        entries_failed = []
        counter = 0

        for entry in tqdm(data, total=len(data), desc="Generating responses"):
            try:
            # transform row data into a dependency
                dependency = transform(entry)

                project_info = get_projet_description(project_name=dependency.project) if config["advanced"] and config["with_context"] else None
                task_str = Prompts.get_task_str(dependency=dependency)
                context_str = "\n\n".join([x["text"] for x in entry["context"]])
                shot_str = "\n\n".join([shot for shot in get_most_similar_shots(shots, dependency)])
                format_str = Prompts.get_format_prompt()

                system_prompt = Prompts.get_system_str(
                    project_name=dependency.project,
                    project_info=project_info,
                    advanced=config["advanced"]
                )

                # create final query
                if config["with_context"]:
                    print("Start generation with context.")
                    query_prompt = Prompts.get_query_str(
                        context_str=context_str,
                        task_str=task_str,
                        shot_str=shot_str,
                        advanced=config["advanced"]
                    )
                else:
                    print("Start generation without context.")
                    query_prompt = f"{task_str}\n\n{format_str}"


                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query_prompt }
                ]

                if "generations" not in entry:
                    entry["generations"] = {}

                # check if generation for model already exists
                if not generator.model_name in entry["generations"]:
                    print(f"Generate response for entry {entry['index']} with model {generator.model_name}.")
                    response = generator.generate(messages=messages)
                    entry["generations"][generator.model_name] = response
                else:
                    print(f"Generation for entry {entry['index']} of model {generator.model_name} already exists. Skipping generation.")
                
                #generation_results.append(entry)

                counter += 1

                # Save results every 100 entries
                if counter % 50 == 0:
                    with open(f"data/evaluation/generation_results/train_dependencies_{config_name}_{counter}.json", "w", encoding="utf-8") as dest:
                        json.dump(data, dest, indent=2)

            except Exception as e:
                print(f"An error occurred for sample: {entry['index']}")
                print(f"Error: {e}")
                #generation_results.append(entry)
                entries_failed.append(entry["index"])
                continue

    with open(f"data/evaluation/generation_results/train_dependencies_{config_name}_{counter}.json", "w", encoding="utf-8") as dest:
        json.dump(data, dest, indent=2)

    with open(f"data/evaluation/failed_train_{config_name}_{counter}.json.json", "w", encoding="utf-8") as dest:
        json.dump(entries_failed, dest, indent=2)


def main():
    # parse args
    args = parse_args()

    # load env variable
    load_dotenv(dotenv_path=args.env_file)

    # load config
    config = load_config(config_file=args.config_file)
    config_name = os.path.basename(args.config_file).split(".")[0]

    mlflow.set_experiment(experiment_name="generation")
    
    with mlflow.start_run(run_name=f"generation_{config_name}"): 

        mlflow.log_params(config)
        mlflow.log_artifact(local_path=args.env_file)

        run_generation(config=config, config_name=config_name)

        mlflow.log_artifact(local_path=config["generation_file"])



if __name__ == "__main__":
    main()