import sys
import time
from pathlib import Path

import pandas as pd
import torch
from torch_geometric.nn import SAGEConv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_gnn_models import (
    StaticGCN, GraphSAGE, GAT, EvolveGCNH, fit_scaler, prepare_snapshots,
    calculate_metrics, evaluate_static_model, evaluate_evolvegcn,
    split_temporal,
)
from train_gnn_models_review3 import set_seed, train_model
from train_evolvegcn_review3 import weights_for


class EvolveGCNHSage(EvolveGCNH):
    """EvolveGCN-H-style model with a SAGEConv output layer."""
    def __init__(self, in_channels=4, hidden_channels=16, out_channels=2):
        super().__init__(in_channels, hidden_channels, out_channels)
        self.output_conv = SAGEConv(hidden_channels, out_channels)


R = Path("results")
VARIANTS = {   # label: (class, tuning summary file, training kind)
    "Static-GCN":   (StaticGCN, R / "tuning_others/tuning_Static-GCN_summary.csv", "static"),
    "GraphSAGE":    (GraphSAGE, R / "tuning_others/tuning_GraphSAGE_summary.csv", "static"),
    "GAT":          (GAT, R / "tuning_others/tuning_GAT_summary.csv", "static"),
    "EvolveGCN-H (GCN output)":  (EvolveGCNH, R / "evolvegcn_tuning_summary.csv", "EvolveGCN-H"),
    "EvolveGCN-H (SAGE output)": (EvolveGCNHSage, R / "evolvegcn_sage/evolvegcn_tuning_summary.csv", "EvolveGCN-H"),
}
EPOCHS = 50


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nodes = pd.read_csv("data/processed/nodes_dynamic.csv")
    edges = pd.read_csv("data/processed/edges_dynamic.csv")

    rows, preds = [], []
    for label, (cls, path, kind) in VARIANTS.items():
        best = pd.read_csv(path).iloc[0]       # top row = best validation F1
        wmode, hid, lr = str(best["weights"]), int(best["hidden"]), float(best["lr"])
        print(f"{label}: weights={wmode} hidden={hid} lr={lr}", flush=True)
        for run in sorted(nodes["run"].unique()):
            rn = nodes[nodes["run"] == run].copy()
            re_ = edges[edges["run"] == run].copy()
            times = sorted(rn["time"].unique())
            train_all, test_times = split_temporal(times)   # unchanged 70/30

            set_seed(int(run))
            scaler = fit_scaler(rn, train_all)              # train only
            snaps = prepare_snapshots(rn, re_, times, scaler, device)
            model = cls(hidden_channels=hid).to(device)
            n_params = sum(p.numel() for p in model.parameters())
            w = weights_for(wmode, rn, train_all, device)

            t0 = time.perf_counter()
            train_model(model, kind, snaps, train_all, w, EPOCHS, lr)
            train_s = time.perf_counter() - t0

            t0 = time.perf_counter()
            if kind == "EvolveGCN-H":
                ev = evaluate_evolvegcn(model, snaps, train_all, test_times)
            else:
                ev = evaluate_static_model(model, snaps, test_times)
            infer_s = time.perf_counter() - t0

            m = calculate_metrics(ev[0], ev[1])
            rows.append({"run": run, "model": label, "weights": wmode,
                         "hidden": hid, "lr": lr, "params": n_params,
                         "train_s": train_s, "infer_s": infer_s,
                         **{f"test_{k}": v for k, v in m.items()}})
            preds.append(pd.DataFrame({"run": run, "model": label,
                "time": ev[3], "node_id": ev[2], "label": ev[0], "prediction": ev[1]}))
            print(f"  run {run} f1={m['mal_f1']:.3f} recall={m['mal_recall']:.3f} "
                  f"precision={m['mal_precision']:.3f}", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(R / "gnn_review3_results.csv", index=False)
    pd.concat(preds).to_csv(R / "gnn_review3_predictions.csv", index=False)
    cols = ["test_accuracy", "test_mal_precision", "test_mal_recall",
            "test_mal_f1", "train_s", "infer_s", "params"]
    summ = res.groupby("model")[cols].agg(["mean", "std"])
    summ.to_csv(R / "gnn_review3_summary.csv")
    print(summ.round(3).to_string())


if __name__ == "__main__":
    main()
