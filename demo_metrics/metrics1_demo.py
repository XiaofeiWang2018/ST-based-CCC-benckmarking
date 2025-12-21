"""
Goal
-----
This script reproduces the "commonly identified interactions" evaluation workflow on a demo dataset:
1) Read standardized prediction edges (Pred) from multiple tools and the shared edges (Shared).
2) For each (dataset, sample, tool), compute TP / FP / FN against Shared.
3) Compute Precision / Recall / F1 per (dataset, sample, tool).
4) Visualize Precision / Recall / F1 distributions across samples using boxplots.

Inputs
------
1) pred_edges.csv  (tool predictions; long-format table)
   Required columns:
     - dataset_id: str
     - sample_id:  str
     - tool:       str
     - sender_ct:  str
     - receiver_ct:str
     - ligand:     str
     - receptor:   str

   Conceptual format / dimension:
     For each (dataset_id, sample_id, tool), this file defines a SET of standardized edges:
       Pred[(dataset, sample, tool)] = {
           (sender_ct, receiver_ct, ligand, receptor),
           ...
       }
     i.e., "cell-type-pair × ligand–receptor" edges (flattened into a set).

2) shared_edges.csv  (alternative positive set; long-format table)
   Required columns:
     - dataset_id: str
     - sample_id:  str
     - sender_ct:  str
     - receiver_ct:str
     - ligand:     str
     - receptor:   str

   Conceptual format / dimension:
     For each (dataset_id, sample_id), this file defines a SET of standardized edges:
       Shared[(dataset, sample)] = {
           (sender_ct, receiver_ct, ligand, receptor),
           ...
       }

   Note:
     In the original paper, "Shared" is usually computed from multiple tools' outputs
     (e.g., edges appearing in >=3 tools). In this demo, Shared is provided as an input file
     to keep the evaluation pipeline simple and reproducible.

Outputs
-------
1) metrics (pandas DataFrame) with one row per (dataset_id, sample_id, tool):
   Columns:
     - TP, FP, FN: int
     - precision, recall, f1: float
     - n_pred: int        (#edges predicted by the tool for that sample)
     - n_shared: int      (#edges in Shared for that sample)

   Definitions (per dataset/sample/tool):
     - TP = |Pred ∩ Shared|
     - FP = |Pred \ Shared|
     - FN = |Shared \ Pred|
     - Precision = TP / (TP + FP)
     - Recall    = TP / (TP + FN)
     - F1        = 2 * Precision * Recall / (Precision + Recall)

   Dimensionality:
     metrics is a table of size (#datasets × #samples_per_dataset × #tools) rows.

2) demo_cci_metrics.csv
   Saved copy of the metrics table.

3) Boxplots
   For each dataset, three separate figures are produced:
     - precision across samples (grouped by tool)
     - recall across samples (grouped by tool)
     - f1 across samples (grouped by tool)
"""
#%%
import pandas as pd
import matplotlib.pyplot as plt

PRED_CSV = "/Users/taochenyang/Downloads/demo_cci_data/pred_edges.csv"
SHARED_CSV = "/Users/taochenyang/Downloads/demo_cci_data/shared_edges.csv"

pred_df = pd.read_csv(PRED_CSV)
shared_df = pd.read_csv(SHARED_CSV)

# ---- 1) Build sets:
# Pred[(dataset,sample,tool)] = set(edges), Shared[(dataset,sample)] = set(edges)
def edge_tuple(df: pd.DataFrame):
    """Convert long-format edge table into a list of standardized edge tuples."""
    return list(zip(df["sender_ct"], df["receiver_ct"], df["ligand"], df["receptor"]))

Pred = {}
for (ds, s, tool), g in pred_df.groupby(["dataset_id", "sample_id", "tool"], sort=False):
    Pred[(ds, s, tool)] = set(edge_tuple(g))

Shared = {}
for (ds, s), g in shared_df.groupby(["dataset_id", "sample_id"], sort=False):
    Shared[(ds, s)] = set(edge_tuple(g))

# ---- 2) Compute TP / FP / FN + Precision / Recall / F1
def safe_div(a: float, b: float) -> float:
    return a / b if b != 0 else 0.0

tools = sorted(pred_df["tool"].unique().tolist())
datasets = sorted(pred_df["dataset_id"].unique().tolist())
samples_by_dataset = {
    ds: sorted(pred_df.loc[pred_df["dataset_id"] == ds, "sample_id"].unique().tolist())
    for ds in datasets
}

rows = []
for ds in datasets:
    for s in samples_by_dataset[ds]:
        shared_set = Shared[(ds, s)]
        for tool in tools:
            pred_set = Pred.get((ds, s, tool), set())

            tp = len(pred_set & shared_set)
            fp = len(pred_set - shared_set)
            fn = len(shared_set - pred_set)

            precision = safe_div(tp, tp + fp)
            recall = safe_div(tp, tp + fn)
            f1 = safe_div(2 * precision * recall, precision + recall) if (precision + recall) > 0 else 0.0

            rows.append({
                "dataset_id": ds,
                "sample_id": s,
                "tool": tool,
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "n_pred": len(pred_set),
                "n_shared": len(shared_set),
            })

metrics = pd.DataFrame(rows)
print(metrics.head(20).to_string(index=False))

# Save metrics table
metrics.to_csv("/Users/taochenyang/Downloads/demo_cci_data/demo_cci_metrics.csv", index=False)
print("\nSaved metrics to ./demo_cci_metrics.csv")

# ---- 3) Boxplots: per dataset, one figure per metric (no subplots)
def boxplot_metric_for_dataset(df: pd.DataFrame, dataset_id: str, metric_name: str, tools_order):
    sub = df[df["dataset_id"] == dataset_id]
    data = [sub.loc[sub["tool"] == t, metric_name].values for t in tools_order]

    plt.figure(figsize=(8, 4.5))
    plt.boxplot(data, labels=tools_order, showmeans=True)
    plt.ylabel(metric_name)
    plt.title(f"{metric_name} across samples ({dataset_id})")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.show()

for ds in datasets:
    boxplot_metric_for_dataset(metrics, ds, "precision", tools)
    boxplot_metric_for_dataset(metrics, ds, "recall", tools)
    boxplot_metric_for_dataset(metrics, ds, "f1", tools)
