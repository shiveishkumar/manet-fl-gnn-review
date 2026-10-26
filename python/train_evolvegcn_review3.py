import argparse
import itertools
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_gnn_models import (
    fit_scaler, prepare_snapshots, calculate_metrics,
    evaluate_static_model, evaluate_evolvegcn, split_temporal,
)
from train_gnn_models_review3 import MODEL_CLASSES, set_seed, train_model


def weights_for(mode, nodes_df, fit_times, device):
    if mode == "none":
        return None
    labels = nodes_df[nodes_df["time"].isin(fit_times)]["label"]
    counts = labels.value_counts().reindex([0, 1]).fillna(1).values
    w = len(labels) / (2.0 * counts)
    if mode == "sqrt":
        w = w ** 0.5
    return torch.tensor(w, dtype=torch.float32, device=device)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="EvolveGCN-H")
    p.add_argument("--data-dir", default="data/processed")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--epochs", type=int, default=50)
    a = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(a.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    nodes = pd.read_csv(Path(a.data_dir) / "nodes_dynamic.csv")
    edges = pd.read_csv(Path(a.data_dir) / "edges_dynamic.csv")

    grid = list(itertools.product(["none", "sqrt", "full"], [16, 32],
                                  [0.01, 0.005, 0.001]))
    rows = []
    for run in sorted(nodes["run"].unique()):
        rn = nodes[nodes["run"] == run].copy()
        re_ = edges[edges["run"] == run].copy()
        times = sorted(rn["time"].unique())
        train_all, _test = split_temporal(times)       # test is never touched
        cut = int(len(train_all) * 0.80)
        train_fit, val_times = train_all[:cut], train_all[cut:]

        scaler = fit_scaler(rn, train_fit)             # train_fit only
        snaps = prepare_snapshots(rn, re_, list(train_fit) + list(val_times),
                                  scaler, device)

        for wmode, hid, lr in grid:
            set_seed(int(run))
            model = MODEL_CLASSES[a.model](hidden_channels=hid).to(device)
            w = weights_for(wmode, rn, train_fit, device)
            train_model(model, a.model, snaps, train_fit, w, a.epochs, lr)
            if a.model == "EvolveGCN-H":
                ev = evaluate_evolvegcn(model, snaps, train_fit, val_times)
            else:
                ev = evaluate_static_model(model, snaps, val_times)
            m = calculate_metrics(ev[0], ev[1])
            rows.append({"run": run, "weights": wmode, "hidden": hid,
                         "lr": lr, **{f"val_{k}": v for k, v in m.items()},
                         "val_pred_share": float(ev[1].mean())})
            print(f"run {run} w={wmode:4s} h={hid:2d} lr={lr:<6} "
                  f"val_f1={m['mal_f1']:.3f} flagged={ev[1].mean():.2f}",
                  flush=True)

    df = pd.DataFrame(rows)
    stem = "evolvegcn_tuning" if a.model == "EvolveGCN-H" \
        else f"tuning_{a.model}"
    df.to_csv(out_dir / f"{stem}.csv", index=False)
    summ = (df.groupby(["weights", "hidden", "lr"])
              [["val_mal_f1", "val_mal_precision", "val_mal_recall",
                "val_pred_share"]].mean().round(3)
              .sort_values("val_mal_f1", ascending=False))
    summ.to_csv(out_dir / f"{stem}_summary.csv")
    print("\nTop 6 configs by mean VALIDATION malicious F1:")
    print(summ.head(6).to_string())
    print("\nBest:", summ.index[0])


if __name__ == "__main__":
    main()
