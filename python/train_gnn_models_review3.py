import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_gnn_models import (  # Review 2 code, reused unchanged
    StaticGCN, GraphSAGE, GAT, EvolveGCNH,
    fit_scaler, prepare_snapshots, calculate_metrics,
    evaluate_static_model, evaluate_evolvegcn, split_temporal,
)

MODEL_CLASSES = {
    "Static-GCN": StaticGCN,
    "GraphSAGE": GraphSAGE,
    "GAT": GAT,
    "EvolveGCN-H": EvolveGCNH,
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def class_weights(nodes_df, fit_times, device):
    labels = nodes_df[nodes_df["time"].isin(fit_times)]["label"]
    counts = labels.value_counts().reindex([0, 1]).fillna(1).values
    w = len(labels) / (2.0 * counts)          # inverse frequency
    return torch.tensor(w, dtype=torch.float32, device=device)


def train_model(model, name, snaps, fit_times, weights, epochs, lr, chunk=7):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    crit = torch.nn.CrossEntropyLoss(weight=weights)
    for _ in range(epochs):
        model.train()
        if name == "EvolveGCN-H":
            hidden, acc, n = None, 0.0, 0
            for i, t in enumerate(fit_times):          # chronological
                x, ei, y, _ = snaps[t]
                out, hidden = model(x, ei, hidden)
                acc = acc + crit(out, y)
                n += 1
                if n == chunk or i == len(fit_times) - 1:
                    opt.zero_grad()
                    (acc / n).backward()
                    opt.step()
                    hidden = hidden.detach()
                    acc, n = 0.0, 0
        else:
            for t in fit_times:
                x, ei, y, _ = snaps[t]
                loss = crit(model(x, ei), y)
                opt.zero_grad()
                loss.backward()
                opt.step()
    return model


def run_stage(name, run_nodes, run_edges, fit_times, eval_times,
              device, epochs, lr, hidden, use_weights=True):
    """Train on fit_times, evaluate on eval_times. Scaler fit on fit_times only."""
    scaler = fit_scaler(run_nodes, fit_times)
    snaps = prepare_snapshots(run_nodes, run_edges,
                              list(fit_times) + list(eval_times), scaler, device)
    model = MODEL_CLASSES[name](hidden_channels=hidden).to(device)
    w = class_weights(run_nodes, fit_times, device) if use_weights else None
    train_model(model, name, snaps, fit_times, w, epochs, lr)
    if name == "EvolveGCN-H":
        ev = evaluate_evolvegcn(model, snaps, fit_times, eval_times)
    else:
        ev = evaluate_static_model(model, snaps, eval_times)
    return calculate_metrics(ev[0], ev[1]), ev


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/processed")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--hidden", type=int, default=16)
    a = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    out_dir = Path(a.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    nodes = pd.read_csv(Path(a.data_dir) / "nodes_dynamic.csv")
    edges = pd.read_csv(Path(a.data_dir) / "edges_dynamic.csv")

    rows, preds = [], []
    for run in sorted(nodes["run"].unique()):
        rn = nodes[nodes["run"] == run].copy()
        re_ = edges[edges["run"] == run].copy()
        times = sorted(rn["time"].unique())
        train_all, test_times = split_temporal(times)         # unchanged 70/30
        cut = int(len(train_all) * 0.80)
        train_fit, val_times = train_all[:cut], train_all[cut:]

        for name in MODEL_CLASSES:
            set_seed(int(run))
            vm, _ = run_stage(name, rn, re_, train_fit, val_times,
                              device, a.epochs, a.lr, a.hidden)
            set_seed(int(run))
            tm, ev = run_stage(name, rn, re_, train_all, test_times,
                               device, a.epochs, a.lr, a.hidden)
            row = {"run": run, "model": name}
            row.update({f"val_{k}": v for k, v in vm.items()})
            row.update({f"test_{k}": v for k, v in tm.items()})
            rows.append(row)
            preds.append(pd.DataFrame({
                "run": run, "model": name, "time": ev[3],
                "node_id": ev[2], "label": ev[0], "prediction": ev[1]}))
            print(f"run {run} {name:12s} val_f1={vm['mal_f1']:.3f} "
                  f"test_recall={tm['mal_recall']:.3f} test_f1={tm['mal_f1']:.3f}")

    res = pd.DataFrame(rows)
    res.to_csv(out_dir / "gnn_review3_results.csv", index=False)
    pd.concat(preds).to_csv(out_dir / "gnn_review3_predictions.csv", index=False)
    cols = [c for c in res.columns if c.startswith(("val_", "test_"))]
    summ = res.groupby("model")[cols].agg(["mean", "std"])
    summ.to_csv(out_dir / "gnn_review3_summary.csv")
    print(summ[[c for c in summ.columns if c[0].startswith("test_")]].round(3))


if __name__ == "__main__":
    main()
