import sys
import time
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_gnn_models import (
    StaticGCN, GraphSAGE, fit_scaler, prepare_snapshots, calculate_metrics,
    evaluate_static_model, evaluate_evolvegcn, split_temporal,
)
from train_gnn_models_review3 import set_seed, train_model
from train_evolvegcn_review3 import weights_for
from final_gnn_review3 import EvolveGCNHSage

SIZES = [100, 200, 300, 400, 500]
SEEDS = [1, 2, 3]
EPOCHS = 50
# label: (class, kind, weights, hidden, lr)  -- best validation configs
MODELS = {
    "GraphSAGE": (GraphSAGE, "static", "none", 16, 0.001),
    "EvolveGCN-H (SAGE output)": (EvolveGCNHSage, "EvolveGCN-H", "sqrt", 32, 0.01),
    "Static-GCN": (StaticGCN, "static", "sqrt", 16, 0.001),
}
R = Path("results")


def main():
    device = torch.device("cpu")
    rows = []
    for size in SIZES:
        folder = Path("data/processed") / str(size)
        nodes = pd.read_csv(folder / "nodes.csv")
        edges = pd.read_csv(folder / "edges.csv")
        run = nodes["run"].min()
        nodes = nodes[nodes["run"] == run].copy()
        edges = edges[edges["run"] == run].copy()

        times = sorted(nodes["time"].unique())
        train_times, test_times = split_temporal(times)       # same 70/30
        rate_t0 = nodes[nodes["time"] == times[0]]["label"].mean()
        rate_tr = nodes[nodes["time"].isin(train_times)]["label"].mean()
        rate_te = nodes[nodes["time"].isin(test_times)]["label"].mean()
        n_edges = len(edges[edges["time"] == test_times[0]])
        print(f"\n== {size} nodes | edges/snapshot~{n_edges} | malicious rate: "
              f"t0={rate_t0:.3f} train={rate_tr:.3f} test={rate_te:.3f}", flush=True)

        scaler = fit_scaler(nodes, train_times)               # train only
        snaps = prepare_snapshots(nodes, edges, times, scaler, device)

        for label, (cls, kind, wmode, hid, lr) in MODELS.items():
            for seed in SEEDS:
                set_seed(seed)
                model = cls(hidden_channels=hid).to(device)
                n_params = sum(p.numel() for p in model.parameters())
                w = weights_for(wmode, nodes, train_times, device)

                t0 = time.perf_counter()
                train_model(model, kind, snaps, train_times, w, EPOCHS, lr)
                train_s = time.perf_counter() - t0

                t0 = time.perf_counter()
                if kind == "EvolveGCN-H":
                    ev = evaluate_evolvegcn(model, snaps, train_times, test_times)
                else:
                    ev = evaluate_static_model(model, snaps, test_times)
                infer_s = time.perf_counter() - t0

                m = calculate_metrics(ev[0], ev[1])
                rows.append({"nodes": size, "model": label, "seed": seed,
                             "params": n_params, "train_s": train_s,
                             "infer_s": infer_s,
                             "test_malicious_rate": rate_te,
                             **{f"test_{k}": v for k, v in m.items()}})
                print(f"  {label:26s} seed {seed} f1={m['mal_f1']:.3f} "
                      f"recall={m['mal_recall']:.3f} train={train_s:.1f}s "
                      f"infer={infer_s:.2f}s", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(R / "scalability_review3_runs.csv", index=False)
    cols = ["test_accuracy", "test_mal_recall", "test_mal_f1",
            "train_s", "infer_s"]
    summ = df.groupby(["model", "nodes"])[cols].agg(["mean", "std"])
    summ.to_csv(R / "scalability_review3.csv")
    pd.set_option("display.width", 220)
    print("\n", summ.round(3).to_string())


if __name__ == "__main__":
    main()
