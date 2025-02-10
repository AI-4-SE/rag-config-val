import argparse
import json
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation_file", type=str)
    parser.add_argument("--output_file", type=str)
    return parser.parse_args()

def get_validation_failures():
    # parse args
    args = parse_args()

    # load generation data
    with open(args.generation_file, "r", encoding="utf-8") as src:
        generation_data = json.load(src)

    models = list(generation_data[0]["generations"].keys())

    failures = []

    for model in models:
        print("Get failures for model: ", model)

        for entry in generation_data:
            final_rating = entry["final_rating"]
            model_response = entry["generations"][model]
            isDependency = model_response["isDependency"]

            data = {
                "model": model,
                "human_classification": final_rating,
                "llm_classification": isDependency,
                "plan": model_response["plan"],
                "rationale": model_response["rationale"],
                "context": "\n\n".join([context["text"] for context in entry["context"]]),
                "project": entry["project"],
                "option_name": entry["option_name"],
                "option_value": entry["option_value"],
                "option_type": entry["option_type"],
                "option_file": entry["option_file"],
                "option_technology": entry["option_technology"],
                "dependend_option_name": entry["dependent_option_name"],
                "dependent_option_value": entry["dependent_option_value"],
                "dependent_option_type": entry["dependent_option_type"],
                "dependent_option_file": entry["dependent_option_file"],
                "dependent_option_technology": entry["dependent_option_technology"],
                "final_category": entry["final_category"],
                "sub_category": entry["sub_category"]
            }

            if isinstance(isDependency, str) and isDependency == "None":
                continue

            # TP: The LLM validates a dependency as correct and the dependency is correct
            if isDependency and final_rating:
                continue
                
            # FP: The LLM validates a dependency as correct, but the dependency is actually incorrect
            if isDependency and not final_rating:
                failures.append(data)

            # TN: The LLM validates a dependency as incorrect and the dependency is incorrect
            if not isDependency and not final_rating:
                continue

            # FN: The LLM validates a dependency as incorrect, but the dependency is actually correct
            if not isDependency and final_rating:
                failures.append(data)
            
    df_failures = pd.DataFrame(failures)
    df_failures.to_csv(args.output_file, index=False)

if __name__ == "__main__":
    get_validation_failures()