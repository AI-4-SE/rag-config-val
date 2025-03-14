import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import seaborn as sns
import numpy as np
import colorsys
import glob
import json


def extract_context_sources():
    retrieval_results_dir = "../data/evaluation/retrieval_results/**"

    for file_path in glob.glob(retrieval_results_dir):
        config_name = file_path.split("/")[-1].split(".")[0].split("_")[-1]

        with open(file_path, "r", encoding="utf-8") as src:
            data = json.load(src)

        context_sources = []

        for entry in data:
            context = entry["context"]
            sources = [x["source"] for x in context]

            context_sources.append(sources)

        with open(f"../data/evaluation/context_sources/context_sources_{config_name}.json", "w", encoding="utf-8") as dest:
            json.dump(context_sources, dest, indent=2)


# Define the target color
target_color = "#e59604"

# Function to adjust saturation toward white
def adjust_saturation_to_white(hex_color, saturation_factor):
    hex_color = hex_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))

    h, l, s = colorsys.rgb_to_hls(r, g, b)

    new_s = min(1, saturation_factor)

    blend_factor = 1 - new_s
    r_new = r + blend_factor * (1 - r)
    g_new = g + blend_factor * (1 - g)
    b_new = b + blend_factor * (1 - b)

    return (int(r_new * 255), int(g_new * 255), int(b_new * 255))

saturation_factors = np.linspace(0, 1, 50)**1.14
palette_white_shift = [adjust_saturation_to_white(target_color, 0.05 + factor**0.5) for factor in saturation_factors]
palette_white_shift_hex = ['#%02x%02x%02x' % rgb for rgb in palette_white_shift]

saturated_colors = palette_white_shift_hex
diverging_colors = LinearSegmentedColormap.from_list("diverging", ["#3E7F94", "#FFFFFF", "#C4583D"][::-1], N=100)

def create_heatmap(dataframe, cmap, title, xlabel, ylabel, file_name, center=None, show_ylabel=True, show_yticks=True,
                   show_cbar_label=True, ratio=4.3 / 7, scale_adjust=1.0, max_labels=10):
    with sns.plotting_context("poster"):
        scaling = 1.25 * scale_adjust
        width = 7 * scaling

        # Adjust height dynamically based on the number of y-axis labels
        num_labels = len(dataframe)
        height = max(1, num_labels / max_labels) * (width * ratio)  # Scale height based on row count
        
        plt.figure(figsize=(width, height))
        ax = sns.heatmap(dataframe, annot=True, cmap=cmap, fmt=".0%",
                         cbar_kws={'label': 'Proportion' if show_cbar_label else None}, center=center)

        plt.title("")
        plt.xlabel(xlabel)

        if show_ylabel:
            plt.ylabel(ylabel)
        else:
            plt.ylabel("")

        if not show_yticks:
            ax.set_yticks([])

        plt.tight_layout()
        plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
        plt.show()



# Function to process JSON files and normalize values
def process_file(file_path, all_sources=None):
    df = pd.read_json(file_path)
    result = pd.DataFrame({
        slot: df[slot].value_counts(normalize=True) for slot in df.columns
    }).fillna(0)  # Fill missing values with 0

    # Ensure consistent y-axis ordering
    if all_sources is not None:
        result = result.reindex(all_sources, fill_value=0)
    
    return result


# Function to generate heatmaps with aligned labels
def process_and_generate_heatmaps(file_paths):
    all_results = [process_file(fp) for fp in file_paths]

    # Get all unique context sources
    all_sources = sorted(set().union(*[df.index for df in all_results]))

    # Reprocess with consistent y-axis order
    all_results = [process_file(fp, all_sources) for fp in file_paths]

    # Determine max label count for scaling
    max_labels = max(len(df) for df in all_results)

    # Generate heatmaps
    for i, result in enumerate(all_results):
        config_name = file_paths[i].split("_")[-1].split(".")[0]

        if i == 0:
            create_heatmap(
            result, saturated_colors, "Proportion of Context Sources per Slot", "Top N Slot",
            "Context Source" if i == 0 else "",
            f"../data/evaluation/figures/context_{config_name}.pdf",
            show_ylabel=True,
            show_yticks=True,
            show_cbar_label=False,
            max_labels=max_labels  # Pass max labels to scale heights correctly
        )
        else:

            create_heatmap(
                result, saturated_colors, "Proportion of Context Sources per Slot", "Top N Slot",
                "",
                f"../data/evaluation/figures/context_{config_name}.pdf",
                show_ylabel=False,
                show_yticks=False,
                show_cbar_label=False,
                ratio=5.5 / 7,
                scale_adjust=0.79
            )



def main():
    extract_context_sources()

    # Paths to JSON files
    # file_paths = [
    #     '../data/evaluation/context_sources/context_sources_config1.json',
    #     '../data/evaluation/context_sources/context_sources_config2.json',
    #     '../data/evaluation/context_sources/context_sources_config3.json',
    #     '../data/evaluation/context_sources/context_sources_config4.json'
    # ]

    file_paths = [
        '../data/evaluation/context_sources/context_sources_config5.json',
        '../data/evaluation/context_sources/context_sources_config6.json',
        '../data/evaluation/context_sources/context_sources_config7.json',
        '../data/evaluation/context_sources/context_sources_config8.json'
    ]

    # Run processing and heatmap generation
    process_and_generate_heatmaps(file_paths)


if __name__ == "__main__":
    main()
