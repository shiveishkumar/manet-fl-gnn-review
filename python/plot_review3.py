from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path("results")
g = pd.read_csv(R / "gnn_review3_results.csv")
s = pd.read_csv(R / "scalability_review3_runs.csv")

# 1. model comparison (150 nodes, test set, mean +/- std over 5 runs)
order = ["GraphSAGE", "EvolveGCN-H (SAGE output)", "GAT",
         "EvolveGCN-H (GCN output)", "Static-GCN"]
m = g.groupby("model")[["test_mal_precision", "test_mal_recall", "test_mal_f1"]]
mean, std = m.mean().loc[order], m.std().loc[order]
ax = mean.plot.bar(yerr=std, capsize=3, figsize=(9, 5))
ax.set_ylabel("Score (test set)")
ax.set_title("GNN comparison, 150 nodes, mean ± std over 5 runs")
ax.legend(["Precision", "Recall", "F1"])
plt.xticks(rotation=25, ha="right"); plt.tight_layout()
plt.savefig(R / "gnn_model_comparison.png", dpi=150); plt.close()

# 2. static vs temporal pairs (not a perfectly controlled comparison)
pairs = ["Static-GCN", "EvolveGCN-H (GCN output)",
         "GraphSAGE", "EvolveGCN-H (SAGE output)"]
mm = g.groupby("model")[["test_mal_recall", "test_mal_f1"]]
mean, std = mm.mean().loc[pairs], mm.std().loc[pairs]
ax = mean.plot.bar(yerr=std, capsize=3, figsize=(9, 5))
ax.set_ylabel("Score (test set)")
ax.set_title("Static vs temporal (architectures also differ)")
ax.legend(["Recall", "F1"])
plt.xticks(rotation=20, ha="right"); plt.tight_layout()
plt.savefig(R / "temporal_vs_static.png", dpi=150); plt.close()

# 3-5. scalability (mean +/- std over 3 seeds, one simulation per size)
def line(col, ylabel, fname, title):
    plt.figure(figsize=(8, 5))
    for model, d in s.groupby("model"):
        a = d.groupby("nodes")[col].agg(["mean", "std"]).fillna(0)
        plt.errorbar(a.index, a["mean"], yerr=a["std"], marker="o",
                     capsize=3, label=model)
    plt.xlabel("Number of nodes"); plt.ylabel(ylabel); plt.title(title)
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(R / fname, dpi=150); plt.close()

line("test_accuracy", "Accuracy", "scalability_accuracy.png",
     "Scalability: accuracy (temporal test split)")
line("test_mal_f1", "Malicious F1", "scalability_f1.png",
     "Scalability: malicious F1")
line("infer_s", "Inference time (s)", "scalability_inference_time.png",
     "Scalability: inference time")
print("saved 5 plots")
