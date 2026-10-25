"""Cluster MANET nodes independently at each graph snapshot.

Review 3 - Raghav Bajaj
Responsibility: Dataset, Clustering, Cluster-Head Selection and Baselines

This script performs snapshot-by-snapshot spatial clustering of mobile nodes.
At each time snapshot (t = 0, 10, ..., 500s) for each simulation run, nodes
are partitioned into K geographic clusters using K-Means clustering on the
(x, y) coordinates from the verified NS-3 mobility traces.

Input:
- data/processed/nodes_dynamic.csv
- data/raw/positions_150n_seed*.csv

Output:
- data/processed_clustered/nodes_with_clusters.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def cluster_snapshot(
    snap_df: pd.DataFrame,
    n_clusters: int,
    seed: int
) -> pd.DataFrame:
    """Assign cluster_id to nodes in a single snapshot using spatial coordinates."""
    coords = snap_df[["x", "y"]].to_numpy(dtype=float)
    
    # If number of nodes is less than n_clusters, reduce K
    k = min(n_clusters, len(snap_df))
    
    kmeans = KMeans(
        n_clusters=k,
        init="k-means++",
        n_init=10,
        random_state=seed
    )
    cluster_labels = kmeans.fit_predict(coords)
    
    result = snap_df.copy()
    result["cluster_id"] = cluster_labels
    return result


def process_clustering(
    nodes_path: str,
    raw_dir: str,
    n_clusters: int = 6,
    out_path: str = "data/processed_clustered/nodes_with_clusters.csv"
) -> pd.DataFrame:
    print(f"Loading dynamic nodes dataset from: {nodes_path}")
    nodes_df = pd.read_csv(nodes_path)
    
    raw_path = Path(raw_dir)
    clustered_runs = []
    
    runs = sorted(nodes_df["run"].unique())
    print(f"Processing {len(runs)} simulation runs with K={n_clusters} clusters per snapshot...")
    
    for r in runs:
        pos_file = raw_path / f"positions_150n_seed{int(r)}.csv"
        if not pos_file.exists():
            raise FileNotFoundError(f"Positions file not found: {pos_file}")
            
        pos_df = pd.read_csv(pos_file)
        
        # Merge node features with (x, y) coordinates on time and node_id
        run_nodes = nodes_df[nodes_df["run"] == r].copy()
        merged = pd.merge(
            run_nodes,
            pos_df[["time", "node_id", "x", "y"]],
            on=["time", "node_id"],
            how="inner"
        )
        
        if len(merged) != len(run_nodes):
            raise ValueError(
                f"Run {r}: Merged row count ({len(merged)}) != node count ({len(run_nodes)})"
            )
            
        snapshot_dfs = []
        timestamps = sorted(merged["time"].unique())
        
        for t in timestamps:
            snap = merged[merged["time"] == t]
            snap_seed = int(int(r) * 10000 + int(t))
            snap_clustered = cluster_snapshot(snap, n_clusters=n_clusters, seed=snap_seed)
            snapshot_dfs.append(snap_clustered)
            
        run_clustered = pd.concat(snapshot_dfs, ignore_index=True)
        clustered_runs.append(run_clustered)
        print(f"  Run {r} completed: {len(run_clustered):,} records clustered across {len(timestamps)} snapshots.")
        
    final_df = pd.concat(clustered_runs, ignore_index=True)
    
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(out, index=False)
    print(f"\nSaved clustered nodes dataset to: {out_path}")
    print(f"Shape: {final_df.shape}")
    print(f"Cluster IDs distribution:\n{final_df['cluster_id'].value_counts().sort_index()}")
    
    return final_df


def main():
    parser = argparse.ArgumentParser(description="Snapshot-level clustering for MANET nodes.")
    parser.add_argument("--nodes", default="data/processed/nodes_dynamic.csv", help="Path to nodes_dynamic.csv")
    parser.add_argument("--raw-dir", default="data/raw", help="Directory with positions_150n_seed*.csv")
    parser.add_argument("--n-clusters", type=int, default=6, help="Number of clusters per snapshot (default: 6)")
    parser.add_argument("--out", default="data/processed_clustered/nodes_with_clusters.csv", help="Output path")
    args = parser.parse_args()
    
    process_clustering(
        nodes_path=args.nodes,
        raw_dir=args.raw_dir,
        n_clusters=args.n_clusters,
        out_path=args.out
    )


if __name__ == "__main__":
    main()
