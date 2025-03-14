import glob
import json
import numpy as np
import pandas as pd


def extract_context_scores():
    retrieval_results_dir = "../data/evaluation/retrieval_results/**"

    for file_path in glob.glob(retrieval_results_dir):
        config_name = file_path.split("/")[-1].split(".")[0].split("_")[-1]

        with open(file_path, "r", encoding="utf-8") as src:
            data = json.load(src)

        context_sources = []

        for entry in data:
            context = entry["context"]
            sources = [x["score"] for x in context]

            context_sources.append(sources)

        with open(f"../data/evaluation/context_scores/context_scores_{config_name}.json", "w", encoding="utf-8") as dest:
            json.dump(context_sources, dest, indent=2)


def compute_average_score():
    file_paths = [
        "../data/evaluation/context_scores/context_scores_config1.json",
        "../data/evaluation/context_scores/context_scores_config2.json",
        "../data/evaluation/context_scores/context_scores_config3.json",
        "../data/evaluation/context_scores/context_scores_config4.json",
        "../data/evaluation/context_scores/context_scores_config5.json",
        "../data/evaluation/context_scores/context_scores_config6.json",
        "../data/evaluation/context_scores/context_scores_config7.json",
        "../data/evaluation/context_scores/context_scores_config8.json",
    ]

    # Dictionary to store average scores for each file
    avg_scores_dict = {}

    for file_path in file_paths:
        # Load the JSON file
        with open(file_path, "r") as file:
            data = json.load(file)

        # Process data: Ensure all rows have length 5
        processed_data = []
        for row in data:
            if len(row) == 1:
                # Expand single values to match length 5 (repeat the value)
                processed_data.append([float(row[0])] * 5)
            elif len(row) == 5:
                # Keep valid rows as they are
                processed_data.append([float(x) for x in row])
            else:
                # Handle unexpected lengths by padding with NaNs
                processed_data.append([float(x) for x in row] + [np.nan] * (5 - len(row)))

        # Convert to NumPy array
        data_array = np.array(processed_data, dtype=np.float64)

        # Compute the average score for each context slot (column-wise mean, ignoring NaNs)
        average_scores = np.nanmean(data_array, axis=0)

        # Round the values to 4 decimal places
        avg_scores_dict[file_path] = np.round(average_scores, 2)

    # Convert the dictionary into a DataFrame
    df_avg_scores = pd.DataFrame(avg_scores_dict)

    # Rename columns with file identifiers for clarity
    df_avg_scores.columns = [f"Config {i+1}" for i in range(len(file_paths))]

    # Add a column for context slot index
    df_avg_scores.insert(0, "Context Slot", range(1, df_avg_scores.shape[0] + 1))

    df_avg_scores.to_csv("../data/evaluation/context_scores/average_scores.csv", index=False)



def main():
    extract_context_scores()
    compute_average_score()


if __name__ == "__main__":
    main()