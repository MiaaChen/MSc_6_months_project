# %%
from pathlib import Path
import re
import sys

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


# %%
# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = BASE_DIR / "data" / "pocket_model_iterations"
OUTPUT_DIR = BASE_DIR / "data" / "graph"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# %%
def extract_gene_count(folder_name):
    """Extract the leading number from names such as '10_gene_related_disease'."""
    match = re.match(r"^(\d+)_gene_related_disease$", folder_name)
    if not match:
        return None
    return int(match.group(1))


def load_pocket_summary(input_dir):
    summary_files = sorted(input_dir.glob("*_gene_related_disease/100_change_iterations_combined_summary.csv"))
    if not summary_files:
        raise FileNotFoundError(f"No combined summary CSV files found under: {input_dir}")

    frames = []
    for csv_path in summary_files:
        folder_name = csv_path.parent.name
        gene_count = extract_gene_count(folder_name)
        if gene_count is None:
            continue

        df = pd.read_csv(csv_path)
        df["gene_related_count"] = gene_count
        df["gene_related_group"] = folder_name
        frames.append(df)

    if not frames:
        raise ValueError("No valid '*_gene_related_disease' summary folders were loaded.")

    data = pd.concat(frames, ignore_index=True)

    required_columns = [
        "disease_id",
        "causal_gene_similarity",
        "mean_final_score",
        "max_final_score",
        "gene_related_count",
        "gene_related_group",
    ]
    missing = [col for col in required_columns if col not in data.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}")

    numeric_columns = ["causal_gene_similarity", "mean_final_score", "max_final_score", "gene_related_count"]
    for col in numeric_columns:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.dropna(subset=numeric_columns).copy()
    data = data.sort_values(["gene_related_count", "disease_id"]).reset_index(drop=True)
    return data


plot_data = load_pocket_summary(INPUT_DIR)
plot_data.to_csv(OUTPUT_DIR / "pocket_similarity_plot_data.csv", index=False)

print(f"Loaded {len(plot_data)} disease rows from {plot_data['gene_related_group'].nunique()} gene-related groups.")
print(f"Saved combined plot data to: {OUTPUT_DIR / 'pocket_similarity_plot_data.csv'}")
try:
    display(plot_data.head())
except NameError:
    print(plot_data.head())


# %%
def draw_similarity_scatter(data, y_col, y_label, output_name):
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, ax = plt.subplots(figsize=(10, 7), dpi=150)

    norm = Normalize(
        vmin=data["gene_related_count"].min(),
        vmax=data["gene_related_count"].max(),
    )
    cmap = plt.get_cmap("viridis")

    scatter = ax.scatter(
        data["causal_gene_similarity"],
        data[y_col],
        c=data["gene_related_count"],
        cmap=cmap,
        norm=norm,
        s=42,
        alpha=0.78,
        linewidth=0.35,
        edgecolor="white",
    )

    ax.set_xlabel("Causal gene similarity")
    ax.set_ylabel(y_label)
    ax.set_title(f"{y_label} vs causal gene similarity")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, max(1, data[y_col].max() * 1.05))

    colorbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax)
    colorbar.set_label("Number in gene-related disease folder")

    ax.grid(True, linewidth=0.5, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    output_path = OUTPUT_DIR / output_name
    fig.savefig(output_path, bbox_inches="tight")
    if "ipykernel" in sys.modules:
        plt.show()
    else:
        plt.close(fig)
    print(f"Saved graph to: {output_path}")


# %%
# Graph 1:
# y-axis = distribution mean, calculated as the mean final score from the 100 repeat similarity runs.
# x-axis = causal gene similarity.
draw_similarity_scatter(
    plot_data,
    y_col="mean_final_score",
    y_label="Distribution mean",
    output_name="distribution_mean_vs_causal_gene_similarity.png",
)


# %%
# Graph 2:
# y-axis = distribution pocket max, calculated as the max final score from the 100 repeat similarity runs.
# x-axis = causal gene similarity.
draw_similarity_scatter(
    plot_data,
    y_col="max_final_score",
    y_label="Distribution pocket max",
    output_name="distribution_pocket_max_vs_causal_gene_similarity.png",
)
