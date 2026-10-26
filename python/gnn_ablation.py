import pandas as pd
from pathlib import Path

R = Path("results")
SRC = {
    "Static-GCN": R / "tuning_others/tuning_Static-GCN.csv",
    "GraphSAGE": R / "tuning_others/tuning_GraphSAGE.csv",
    "GAT": R / "tuning_others/tuning_GAT.csv",
    "EvolveGCN-H (GCN output)": R / "evolvegcn_tuning.csv",
    "EvolveGCN-H (SAGE output)": R / "evolvegcn_sage/evolvegcn_tuning.csv",
}

rows = []
for model, path in SRC.items():
    d = pd.read_csv(path)
    for factor in ["weights", "hidden"]:
        per_run = (d.groupby([factor, "run"])[["val_mal_f1", "val_pred_share"]]
                     .mean().reset_index())
        s = per_run.groupby(factor).agg(
            val_f1_mean=("val_mal_f1", "mean"),
            val_f1_std=("val_mal_f1", "std"),
            flagged_share=("val_pred_share", "mean")).reset_index()
        for _, r in s.iterrows():
            rows.append({"model": model, "factor": factor,
                         "level": r[factor], "val_f1_mean": r["val_f1_mean"],
                         "val_f1_std": r["val_f1_std"],
                         "flagged_share": r["flagged_share"]})

out = pd.DataFrame(rows)
out.to_csv(R / "gnn_ablation.csv", index=False)
pd.set_option("display.width", 200)
print(out.round(3).to_string(index=False))
