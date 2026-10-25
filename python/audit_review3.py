import pandas as pd
from pathlib import Path

nodes = pd.read_csv(Path("data/processed/nodes_dynamic.csv"))

times = sorted(nodes["time"].unique())
cut = int(len(times) * 0.70)
train_all, test = times[:cut], times[cut:]
val_cut = int(len(train_all) * 0.80)
train_fit, val = train_all[:val_cut], train_all[val_cut:]

assert times == sorted(times), "times not ordered"
assert max(train_fit) < min(val) < max(val) < min(test), "overlap"
print("train_fit:", train_fit[0], "-", train_fit[-1], f"({len(train_fit)} snaps)")
print("val      :", val[0], "-", val[-1], f"({len(val)} snaps)")
print("test     :", test[0], "-", test[-1], f"({len(test)} snaps)")

for name, ts in [("train_fit", train_fit), ("val", val), ("test", test)]:
    d = nodes[nodes["time"].isin(ts)]
    print(f"{name:9s} malicious rate = {d['label'].mean():.4f}")
    print(d.groupby("attack").size().to_dict())
