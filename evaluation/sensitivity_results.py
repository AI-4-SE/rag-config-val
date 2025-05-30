import json
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_rag_results(json_file_path: str, output_csv_path: str = None):
    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = []
    for entry in data:
        true_label = entry.get("final_rating", None)
        if true_label is None:
            continue

        generations = entry.get("generations", {})
        for model_temp_key, gen in generations.items():
            prediction = gen.get("isDependency", None)
            temp = gen.get("temperature", None)
            if prediction in (None, "None") or temp is None:
                continue

            model = model_temp_key.rsplit("_temp_", 1)[0]
            records.append({
                "model": model,
                "temperature": temp,
                "predicted": prediction,
                "true": true_label
            })

    df = pd.DataFrame(records)

    # Berechne Metriken je Modell und Temperatur
    grouped = df.groupby(["model", "temperature"]).apply(
        lambda g: pd.Series({
            "precision": precision_score(g["true"], g["predicted"], zero_division=0),
            "recall": recall_score(g["true"], g["predicted"], zero_division=0),
            "f1": f1_score(g["true"], g["predicted"], zero_division=0)
        })
    ).reset_index()

    if output_csv_path:
        grouped.to_csv(output_csv_path, index=False)

    return grouped




def plot_linechart(df: pd.DataFrame):
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='temperature', y='f1', hue='model', marker='o')
    plt.title('F1-Score über Temperature je Modell')
    plt.xlabel('Temperature')
    plt.ylabel('F1-Score')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_instance_f1_boxplot(json_file_path: str):
    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    instance_rows = []
    for entry in data:
        true_label = entry.get("final_rating")
        if true_label is None:
            continue

        generations = entry.get("generations", {})
        for model_temp_key, gen in generations.items():
            pred_label = gen.get("isDependency", None)
            temp = gen.get("temperature", None)

            if pred_label in (None, "None") or temp is None:
                continue

            model = model_temp_key.rsplit("_temp_", 1)[0]
            instance_f1 = f1_score([true_label], [pred_label], zero_division=0)

            instance_rows.append({
                "model": model,
                "temperature": float(temp),
                "instance_f1": instance_f1
            })

    df_instances = pd.DataFrame(instance_rows)

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df_instances, x="temperature", y="instance_f1", hue="model")
    plt.title("Instance-Level F1 Distribution per Temperature and Model")
    plt.xlabel("Temperature")
    plt.ylabel("Instance-Level F1")
    plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()



# Beispielnutzung
if __name__ == "__main__":
    result = evaluate_rag_results("../data/evaluation/sensitivity/test_dependencies_temp_all.json", "model_temp_scores.csv")
    print(result)
    #plot_linechart(result)
    #plot_instance_f1_boxplot("../data/evaluation/sensitivity/test_dependencies_temp_all.json")