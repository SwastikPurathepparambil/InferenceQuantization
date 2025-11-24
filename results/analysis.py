import pandas as pd
import glob
import os
import matplotlib.pyplot as plt


def load_all_benchmark_csvs(folder="results"):
    rows = []

    for path in glob.glob(os.path.join(folder, "*.csv")):
        filename = os.path.basename(path)

        # Parse GPU name (prefix before first underscore)
        gpu = filename.split("_")[0]

        # Parse model: everything after GPU prefix and before .csv
        # e.g. "2080_phi2.csv" → "phi2"
        model = filename[len(gpu)+1:].replace(".csv", "")

        # Load CSV
        df = pd.read_csv(path)

        # Add GPU + model columns
        df["gpu"] = gpu
        df["model"] = model
        df["file"] = filename

        rows.append(df)

    # Combine all results
    full_df = pd.concat(rows, ignore_index=True)
    return full_df

df = load_all_benchmark_csvs("./csvfiles/")



PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

# Human-friendly titles
METRICS = {
    "latency_sec": "Latency (s)",
    "tokens_per_sec": "Throughput (tokens/s)",
    "peak_mem_gb": "Peak Memory (GB)",
}

# Consistent ordering for bar charts
PRECISIONS = ["fp16", "int8", "4bit"]

# Drop rows with errors (optional)
clean_df = df[(df["error"].isna()) | (df["error"] == "")]



# Generate bar charts
for (gpu, model), group in clean_df.groupby(["gpu", "model"]):

    # Sort rows by precision order
    group = group.set_index("precision")
    group = group.reindex(PRECISIONS).dropna(subset=["latency_sec", "tokens_per_sec", "peak_mem_gb"])
    group = group.reset_index()

    if group.empty:
        continue

    for metric, title in METRICS.items():
        plt.figure(figsize=(6, 4))
        plt.bar(group["precision"], group[metric], color="steelblue")
        plt.title(f"{title} — {model} on {gpu}")
        plt.xlabel("Precision")
        plt.ylabel(title)
        plt.grid(axis="y", alpha=0.3)

        filename = f"{model}_{gpu}_{metric}.png".replace("/", "_")
        filepath = os.path.join(PLOTS_DIR, filename)

        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close()

print(f"Saved all charts into: {os.path.abspath(PLOTS_DIR)}")




