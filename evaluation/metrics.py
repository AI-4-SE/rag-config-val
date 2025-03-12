import argparse
import json
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation_file", type=str)
    return parser.parse_args()

def compute_evaluation_metrics():
    """
    Compute the evaluation metrics, including precision, recall and F1 score.
    """
    # parse args
    args = parse_args()

    config_name = args.generation_file.split("_")[-1].split(".")[0]

    # load generation data
    with open(args.generation_file, "r", encoding="utf-8") as src:
        generation_data = json.load(src)

    models = list(generation_data[0]["generations"].keys())

    metrics = []

    for model in models:

        print("Model: ", model)

        true_positives = []
        true_negatives = []
        false_positives = []
        false_negatives = []
        accuracy_count = []
        skipped = 0

        for entry in generation_data:
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
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "failures": fp + fn,
            "precision": round(precision, 2),
            "recall": round(recall, 2),
            "f1_score": round(f1_score, 2),
            "accuracy": round(accuracy, 2),
            "skipped": skipped
        })

    output_file = f"../data/evaluation/validation_effectiveness/{config_name}.csv"

    print("Output file: ", output_file)

    df = pd.DataFrame(data=metrics)
    df.to_csv(output_file, index=False)

    print(df)

if __name__ == "__main__":
    compute_evaluation_metrics()