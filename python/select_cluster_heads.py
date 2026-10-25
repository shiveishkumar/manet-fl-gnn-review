"""Select cluster heads and construct the Review 3 cluster-aware MANET dataset.

Review 3 - Raghav Bajaj
Responsibility: Dataset, Clustering, Cluster-Head Selection and Baselines

This script evaluates candidate nodes in every cluster at each time snapshot using
the transparent multi-criteria formula specified in the Review 3 Work Plan:

  CH_score = 0.30 * norm_energy + 0.30 * norm_trust + 0.20 * norm_degree + 0.20 * inv_distance

Metrics added to dataset:
- cluster_id: group ID of the node
- is_cluster_head: binary flag (1 for elected CH, 0 for member nodes)
- cluster_size: total number of nodes in this cluster
- distance_to_ch: Euclidean distance from the node to its elected cluster head
- ch_score: computed multi-criteria suitability score (for verification)

Outputs:
- data/processed_clustered/nodes_clustered.csv
- data/processed_clustered/edges_clustered.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def normalize_series(s: pd.Series) -> pd.Series:
    """Min-max normalize a series to [0, 1]. If all values equal, returns 1.0."""
    s_min = s.min()
    s_max = s.max()
    if s_max > s_min:
        return (s - s_min) / (s_max - s_min)
    return pd.Series(1.0, index=s.index)


def elect_cluster_heads_for_snapshot(snap_df: pd.DataFrame) -> pd.DataFrame:
    """Elect one cluster head per cluster in a snapshot and calculate distances."""
    updated_clusters = []
    
    for c_id, c_df in snap_df.groupby("cluster_id"):
        c_df = c_df.copy()
        c_size = len(c_df)
        c_df["cluster_size"] = c_size
        
        # 1. Normalized Energy
        norm_energy = normalize_series(c_df["energy"])
        
        # 2. Normalized Trust
        norm_trust = normalize_series(c_df["trust"])
        
        # 3. Normalized Degree
        norm_degree = normalize_series(c_df["degree"])
        
        # 4. Average Distance to other nodes in cluster & Normalized Inverse Distance
        coords = c_df[["x", "y"]].to_numpy(dtype=float)
        inv_distances = []
        
        for i in range(c_size):
            if c_size > 1:
                diffs = coords - coords[i]
                dists = np.sqrt(np.sum(diffs ** 2, axis=1))
                # Average distance to other nodes (excluding self at index i)
                avg_dist = float(np.sum(dists) / (c_size - 1))
            else:
                avg_dist = 0.0
            inv_distances.append(1.0 / (1.0 + avg_dist))
            
        inv_dist_series = pd.Series(inv_distances, index=c_df.index)
        norm_inv_dist = normalize_series(inv_dist_series)
        
        # Multi-criteria score computation
        ch_score = (
            0.30 * norm_energy
            + 0.30 * norm_trust
            + 0.20 * norm_degree
            + 0.20 * norm_inv_dist
        )
        c_df["ch_score"] = ch_score
        
        # Select one Cluster Head: node with highest CH_score
        # Tie-breaker: higher energy, then higher trust
        ch_idx = c_df.sort_values(
            by=["ch_score", "energy", "trust"],
            ascending=[False, False, False]
        ).index[0]
        
        c_df["is_cluster_head"] = 0
        c_df.loc[ch_idx, "is_cluster_head"] = 1
        
        # Compute distance to elected CH for all nodes in cluster
        ch_x = c_df.loc[ch_idx, "x"]
        ch_y = c_df.loc[ch_idx, "y"]
        c_df["distance_to_ch"] = np.sqrt(
            (c_df["x"] - ch_x) ** 2 + (c_df["y"] - ch_y) ** 2
        )
        
        updated_clusters.append(c_df)
        
    return pd.concat(updated_clusters, ignore_index=True)


def process_cluster_heads(
    nodes_clustered_input: str,
    edges_input: str,
    nodes_output: str = "data/processed_clustered/nodes_clustered.csv",
    edges_output: str = "data/processed_clustered/edges_clustered.csv"
):
    print(f"Reading clustered nodes from: {nodes_clustered_input}")
    nodes_df = pd.read_csv(nodes_clustered_input)
    
    runs = sorted(nodes_df["run"].unique())
    processed_snapshots = []
    
    print(f"Selecting cluster heads across {len(runs)} runs and all snapshots...")
    for r in runs:
        r_df = nodes_df[nodes_df["run"] == r]
        timestamps = sorted(r_df["time"].unique())
        
        for t in timestamps:
            snap = r_df[r_df["time"] == t]
            processed_snap = elect_cluster_heads_for_snapshot(snap)
            processed_snapshots.append(processed_snap)
            
    final_nodes_df = pd.concat(processed_snapshots, ignore_index=True)
    
    # Sort for clean, deterministic layout
    final_nodes_df = final_nodes_df.sort_values(by=["run", "time", "node_id"]).reset_index(drop=True)
    
    out_nodes = Path(nodes_output)
    out_nodes.parent.mkdir(parents=True, exist_ok=True)
    final_nodes_df.to_csv(out_nodes, index=False)
    print(f"Saved cluster-aware nodes dataset to: {nodes_output}")
    print(f"  Shape: {final_nodes_df.shape}")
    print(f"  Total Cluster Heads elected: {final_nodes_df['is_cluster_head'].sum():,}")
    print(f"  CH count per snapshot: exactly {final_nodes_df.groupby(['run', 'time'])['is_cluster_head'].sum().unique().tolist()} per snapshot.")
    
    # Process edges: annotate with cluster information
    print(f"\nProcessing edges from: {edges_input}...")
    edges_df = pd.read_csv(edges_input)
    
    # Build lookup dictionary: (run, time, node_id) -> cluster_id
    cluster_map = final_nodes_df.set_index(["run", "time", "node_id"])["cluster_id"].to_dict()
    
    src_tuples = list(zip(edges_df["run"], edges_df["time"], edges_df["src"]))
    dst_tuples = list(zip(edges_df["run"], edges_df["time"], edges_df["dst"]))
    
    edges_df["src_cluster"] = [cluster_map.get(t, -1) for t in src_tuples]
    edges_df["dst_cluster"] = [cluster_map.get(t, -1) for t in dst_tuples]
    edges_df["is_intra_cluster"] = (edges_df["src_cluster"] == edges_df["dst_cluster"]).astype(int)
    
    out_edges = Path(edges_output)
    edges_df.to_csv(out_edges, index=False)
    print(f"Saved cluster-aware edges dataset to: {edges_output}")
    print(f"  Shape: {edges_df.shape}")
    print(f"  Intra-cluster edges: {(edges_df['is_intra_cluster'] == 1).sum():,} ({100*(edges_df['is_intra_cluster'] == 1).mean():.1f}%)")
    print(f"  Inter-cluster edges: {(edges_df['is_intra_cluster'] == 0).sum():,} ({100*(edges_df['is_intra_cluster'] == 0).mean():.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Multi-criteria Cluster Head selection.")
    parser.add_argument(
        "--nodes",
        default="data/processed_clustered/nodes_with_clusters.csv",
        help="Input nodes with cluster_id"
    )
    parser.add_argument(
        "--edges",
        default="data/processed/edges_dynamic.csv",
        help="Input dynamic edges"
    )
    parser.add_argument(
        "--out-nodes",
        default="data/processed_clustered/nodes_clustered.csv",
        help="Output path for nodes_clustered.csv"
    )
    parser.add_argument(
        "--out-edges",
        default="data/processed_clustered/edges_clustered.csv",
        help="Output path for edges_clustered.csv"
    )
    args = parser.parse_args()
    
    process_cluster_heads(
        nodes_clustered_input=args.nodes,
        edges_input=args.edges,
        nodes_output=args.out_nodes,
        edges_output=args.out_edges
    )


if __name__ == "__main__":
    main()
