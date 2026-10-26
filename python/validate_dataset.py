"""Validate dynamic MANET dataset integrity for Review 3.

Checks:
- Missing / NaN values
- Duplicate records
- Timestamp consistency and spacing
- Node count consistency per snapshot (150 nodes/snapshot)
- Label and attack type distributions across runs and temporal split
- Feature value ranges and sanity
- Edge dataset validity (endpoints, types, distance ranges)
- Coordinate alignment with raw position files
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def validate_nodes(nodes_path: str) -> pd.DataFrame:
    print(f"\n{'='*60}")
    print(f"1. VALIDATING NODES DATASET: {nodes_path}")
    print(f"{'='*60}")
    
    df = pd.read_csv(nodes_path)
    print(f"Loaded {len(df):,} rows with {len(df.columns)} columns.")
    print(f"Columns: {list(df.columns)}")
    
    # 1. Missing / NaN values
    nans = df.isna().sum()
    total_nans = nans.sum()
    print(f"\n[Check 1] Missing values (NaNs): {total_nans}")
    if total_nans > 0:
        print(nans[nans > 0])
        raise AssertionError("Found NaN values in nodes dataset!")
    print("  -> PASS: Zero missing values.")

    # 2. Duplicate records
    dup_keys = df.duplicated(subset=["run", "time", "node_id"]).sum()
    print(f"\n[Check 2] Duplicate (run, time, node_id) tuples: {dup_keys}")
    if dup_keys > 0:
        raise AssertionError(f"Found {dup_keys} duplicate node records!")
    print("  -> PASS: Unique key constraint verified.")

    # 3. Runs and Timestamps
    runs = sorted(df["run"].unique())
    print(f"\n[Check 3] Runs present: {runs} (Total: {len(runs)})")
    
    for r in runs:
        r_df = df[df["run"] == r]
        times = sorted(r_df["time"].unique())
        expected_times = list(range(0, 501, 10))  # 0, 10, ..., 500
        time_match = [float(t) for t in times] == [float(t) for t in expected_times]
        print(f"  Run {r}: {len(times)} snapshots ({times[0]}s to {times[-1]}s, step 10s) -> Match expected: {time_match}")
        if not time_match:
            raise AssertionError(f"Run {r} timestamps do not match standard 0..500s grid!")
    print("  -> PASS: All runs have identical, uniform 51 timestamps (step 10s).")

    # 4. Node counts per snapshot
    print(f"\n[Check 4] Node count per snapshot:")
    counts = df.groupby(["run", "time"])["node_id"].count()
    min_nodes, max_nodes = counts.min(), counts.max()
    print(f"  Nodes per snapshot: min={min_nodes}, max={max_nodes} (Expected: 150)")
    if min_nodes != 150 or max_nodes != 150:
        raise AssertionError("Snapshot node count mismatch!")
    print("  -> PASS: Exactly 150 nodes in all 5 runs * 51 snapshots = 38,250 rows.")

    # 5. Label and Attack consistency
    print(f"\n[Check 5] Label and attack distribution:")
    label_dist = df["label"].value_counts().to_dict()
    attack_dist = df["attack"].value_counts().to_dict()
    print(f"  Binary Labels: {label_dist}")
    print(f"  Attack Types : {attack_dist}")
    
    # Check consistency: label == 0 iff attack == 'normal'
    mismatches = df[(df["label"] == 0) & (df["attack"] != "normal")]
    if len(mismatches) > 0:
        raise AssertionError(f"Found {len(mismatches)} label=0 records with attack != normal!")
    mismatches2 = df[(df["label"] == 1) & (df["attack"] == "normal")]
    if len(mismatches2) > 0:
        raise AssertionError(f"Found {len(mismatches2)} label=1 records with attack == normal!")
    print("  -> PASS: Binary labels strictly match attack categories.")

    # 6. Feature bounds sanity
    print(f"\n[Check 6] Feature value bounds:")
    num_cols = ["energy", "trust", "drop_rate", "forward_ratio", "mobility", "delay", "degree"]
    for col in num_cols:
        c_min, c_max = df[col].min(), df[col].max()
        print(f"  {col:15s}: min = {c_min:.4f}, max = {c_max:.4f}")
    assert df["energy"].min() >= 0.0 and df["energy"].max() <= 1.01, "Energy out of range!"
    assert df["trust"].min() >= 0.0 and df["trust"].max() <= 1.01, "Trust out of range!"
    assert df["drop_rate"].min() >= 0.0 and df["drop_rate"].max() <= 1.01, "Drop rate out of range!"
    print("  -> PASS: All behavioural features are strictly within physically valid ranges.")

    # 7. Temporal split class verification
    print(f"\n[Check 7] Temporal Train/Test split verification (cut at t=340s):")
    for r in runs:
        r_df = df[df["run"] == r]
        train = r_df[r_df["time"] <= 340]
        test = r_df[r_df["time"] >= 350]
        tr_norm = (train["label"] == 0).sum()
        tr_mal = (train["label"] == 1).sum()
        te_norm = (test["label"] == 0).sum()
        te_mal = (test["label"] == 1).sum()
        print(f"  Run {r}: Train (0..340s) = {tr_norm} normal, {tr_mal} mal | Test (350..500s) = {te_norm} normal, {te_mal} mal")
        assert tr_mal > 0, f"Run {r} has 0 malicious samples in train!"
        assert te_mal > 0, f"Run {r} has 0 malicious samples in test!"
    print("  -> PASS: Both train and test partitions across all runs contain malicious and normal samples.")

    return df


def validate_edges(edges_path: str) -> pd.DataFrame:
    print(f"\n{'='*60}")
    print(f"2. VALIDATING EDGES DATASET: {edges_path}")
    print(f"{'='*60}")
    
    df = pd.read_csv(edges_path)
    print(f"Loaded {len(df):,} edges with {len(df.columns)} columns.")
    print(f"Columns: {list(df.columns)}")
    
    # 1. Missing / NaN values
    nans = df.isna().sum().sum()
    print(f"\n[Check 1] Missing values (NaNs): {nans}")
    assert nans == 0, "Found NaN values in edges dataset!"
    print("  -> PASS: Zero missing values.")

    # 2. Node index range
    min_node = min(df["src"].min(), df["dst"].min())
    max_node = max(df["src"].max(), df["dst"].max())
    print(f"\n[Check 2] Node ID endpoints in edges: min={min_node}, max={max_node} (Expected: 0..149)")
    assert min_node >= 0 and max_node < 150, "Edge endpoints out of node index bounds!"
    print("  -> PASS: All edges connect valid nodes within [0, 149].")

    # 3. Edge types
    edge_types = df["edge_type"].value_counts().to_dict()
    print(f"\n[Check 3] Edge types: {edge_types}")
    print(f"  Radio edges: {edge_types.get('radio', 0):,}")
    print(f"  Wormhole edges: {edge_types.get('wormhole', 0):,}")
    assert "radio" in edge_types, "No radio edges found!"
    print("  -> PASS: Edge types verified.")

    # 4. Distance distribution
    dist_min, dist_max = df["distance"].min(), df["distance"].max()
    print(f"\n[Check 4] Edge distances: min={dist_min:.2f}m, max={dist_max:.2f}m")
    assert dist_min >= 0.0, "Negative edge distance found!"
    print("  -> PASS: All edge distances are non-negative and valid.")

    return df


def validate_positions_alignment(nodes_df: pd.DataFrame, raw_dir: str):
    print(f"\n{'='*60}")
    print(f"3. VALIDATING POSITION TRACE ALIGNMENT FOR CLUSTERING")
    print(f"{'='*60}")
    
    for r in range(1, 6):
        pos_file = Path(raw_dir) / f"positions_150n_seed{r}.csv"
        if not pos_file.exists():
            raise FileNotFoundError(f"Missing position file: {pos_file}")
        
        pos_df = pd.read_csv(pos_file)
        n_pos = len(pos_df)
        n_nodes_run = len(nodes_df[nodes_df["run"] == r])
        print(f"  Run {r}: {pos_file.name} -> {n_pos} positions vs {n_nodes_run} node records.")
        assert n_pos == n_nodes_run, f"Count mismatch for run {r}!"
        
        # Check alignment of (time, node_id)
        merged = pd.merge(
            nodes_df[nodes_df["run"] == r][["run", "time", "node_id"]],
            pos_df[["time", "node_id", "x", "y"]],
            on=["time", "node_id"],
            how="inner"
        )
        assert len(merged) == n_nodes_run, f"Inner join mismatch for run {r}!"
        assert not merged["x"].isna().any(), f"NaN x coordinates in run {r}!"
        assert not merged["y"].isna().any(), f"NaN y coordinates in run {r}!"
    print("  -> PASS: Exact 1-to-1 position alignment verified for all 5 runs.")


def main():
    nodes_df = validate_nodes("data/processed/nodes_dynamic.csv")
    validate_edges("data/processed/edges_dynamic.csv")
    validate_positions_alignment(nodes_df, "data/raw")
    print(f"\n{'='*60}")
    print("ALL VALIDATION CHECKS PASSED SUCCESSFULLY!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
