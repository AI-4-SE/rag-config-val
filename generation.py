from src.utils import load_config, set_embedding, set_llm, transform
from src.rag import RAG
from src.generator import GeneratorFactory
from src.prompts import Prompts
from dotenv import load_dotenv
from tqdm import tqdm
import pandas as pd
import argparse
import os
import json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str)
    parser.add_argument("--with_context", type=bool, default=False)
    parser.add_argument("--env_file", type=str, default=".env")
    return parser.parse_args()

def run_generation():
    print("Start generation.")

    # parse args
    args = parse_args()

    # load env
    load_dotenv(args.env_file)

    # load config
    config = load_config(args.config_file)
    config_name = os.path.basename(args.config_file).split(".")[0]

    # set inference and embedding moddels
    set_llm(inference_model_name=config["inference_models"][0])
    set_embedding(embed_model_name=config["embedding_model"])

    # define generators
    generators = [
        GeneratorFactory().get_generator(model_name=inference_model_name, temperature=config["temperature"]) 
        for inference_model_name in config["inference_models"]
    ]

    print(f"{len(generators)} generators initialized.")

    with open(config["retrieval_output_file"], "r", encoding="utf-8") as src:
        data = json.load(src)

    generation_results = []
    entries_failed = []
    counter = 0

    for generator in generators:

        for entry in tqdm(data, total=len(data), desc="Generating responses..."):
            try:
                # transform row data into a dependency
                dependency = transform(entry)

                # get prompts
                system_prompt = Prompts.get_system_str(
                    project_name=dependency.project,
                    project_info=None,
                    advanced=config["advanced"]
                )

                task_str = Prompts.get_task_str(dependency=dependency)
                context_str = "\n\n".join([x["text"] for x in entry["context"]])
                format_str = Prompts.get_format_prompt()
                

                # create final query
                if args.with_context:
                    query_prompt = Prompts.get_query_str(
                        context_str=context_str,
                        task_str=task_str,
                        shot_str="",
                        advanced=config["advanced"]
                    )
                else:
                    query_prompt = f"{task_str}\n\n{format_str}"


                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query_prompt }
                ]


                response = generator.generate(messages=messages)    
                if "generations" not in entry:
                    entry["generations"] = {}
                entry["generations"][generator.model_name] = response
                generation_results.append(entry)

                counter += 1

                # Save results every 100 entries
                if counter % 50 == 0:
                    with open(f"data/evaluation/generation_results/all_dependencies_{config_name}_{counter}.json", "w", encoding="utf-8") as dest:
                        json.dump(generation_results, dest, indent=2)
                    with open(f"data/evaluation/failed_{config_name}_{counter}.json.json", "w", encoding="utf-8") as dest:
                        json.dump(entries_failed, dest, indent=2)

            except Exception as e:
                print(f"An error occurred for sample: {entry['index']}")
                print(f"Error: {e}")
                generation_results.append(entry)
                entries_failed.append(entry["index"])
                continue


    with open(config["generation_output_file"], "w", encoding="utf-8") as dest:
        json.dump(generation_results, dest, indent=2)

    with open("data/evaluation/generation_failed.json", "w", encoding="utf-8") as dest:
        json.dump(entries_failed, dest, indent=2)


if __name__ == "__main__":
    run_generation()