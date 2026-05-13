import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

# =========================================================
# LOAD TSV
# =========================================================

df = pd.read_csv("final_system_scores.tsv", sep="\t")

sns.set_style("whitegrid")

# =========================================================
# HELPER FUNCTIONS
# =========================================================

def get_method(name):
    name = name.lower()

    if "unstructured" in name:
        return "Unstructured"

    elif "semi" in name:
        return "Semi-Structured"

    elif "structured" in name:
        return "Structured"

    return "Unknown"


def get_sparsity(name):
    name = name.lower()

    if "25" in name:
        return 25

    elif "50" in name:
        return 50

    elif "75" in name:
        return 75

    return 0


def get_model_type(name):
    """
    Detect Base vs Fine-Tuned
    """

    name = name.lower()

    if "_ft_" in name or "fine" in name:
        return "Fine-Tuned"

    return "Base"


# =========================================================
# PARSE COLUMNS
# =========================================================

df["Method"] = df["File_Name"].apply(get_method)
df["Sparsity"] = df["File_Name"].apply(get_sparsity)
df["ModelType"] = df["File_Name"].apply(get_model_type)

# =========================================================
# METRICS TO PLOT
# =========================================================

metrics = [
    "BLEU",
    "chrF++",
    "BERTScore_F1",
    "COMET",
    "COMTAIL"
]

# =========================================================
# CREATE PLOTS
# =========================================================

for metric in metrics:

    plt.figure(figsize=(10, 6))

    # -----------------------------------------------------
    # Plot each method separately
    # -----------------------------------------------------

    for method in ["Structured", "Unstructured", "Semi-Structured"]:

        subset = df[df["Method"] == method]

        # Base Model
        base_subset = subset[subset["ModelType"] == "Base"]

        if len(base_subset) > 0:

            sns.lineplot(
                data=base_subset,
                x="Sparsity",
                y=metric,
                marker="o",
                linewidth=2,
                label=f"{method} - Base"
            )

        # Fine-Tuned Model
        ft_subset = subset[subset["ModelType"] == "Fine-Tuned"]

        if len(ft_subset) > 0:

            sns.lineplot(
                data=ft_subset,
                x="Sparsity",
                y=metric,
                marker="o",
                linewidth=2,
                linestyle="--",
                label=f"{method} - Fine-Tuned"
            )

    # -----------------------------------------------------
    # Formatting
    # -----------------------------------------------------

    plt.title(f"{metric} vs Sparsity")
    plt.xlabel("Sparsity (%)")
    plt.ylabel(metric)

    plt.xticks([25, 50, 75])

    plt.legend()

    plt.tight_layout()

    plt.savefig(f"{metric}_comparison_plot.png")

    plt.close()

print("All metric comparison plots generated successfully!")