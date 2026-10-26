"""Generate high-quality cluster and cluster-head visualizations for Review 3.

Review 3 - Raghav Bajaj
Responsibility: Dataset, Clustering, Cluster-Head Selection and Baselines

This script visualizes the MANET topology with cluster memberships, elected cluster heads,
communication links, and active attack nodes at key timestamps:
- t = 0s (initial deployment, 100% normal, clear spatial clustering)
- t = 350s (held-out test evaluation phase with active attacks and cluster heads)

Outputs:
- results/cluster_visualization_t0.png
- results/cluster_visualization_t350.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull


def plot_cluster_snapshot(
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    run_id: int,
    time_s: float,
    output_path: str
):
    # Filter for the specific snapshot
    snap_nodes = nodes_df[
        (nodes_df["run"] == run_id) & (nodes_df["time"] == time_s)
    ].sort_values("node_id").copy()
    
    snap_edges = edges_df[
        (edges_df["run"] == run_id) & (edges_df["time"] == time_s)
    ].copy()
    
    if snap_nodes.empty:
        print(f"Warning: No node data for run={run_id}, time={time_s}")
        return

    fig, ax = plt.subplots(figsize=(10, 9), dpi=300)
    
    # Color palette for 6 clusters
    cluster_colors = [
        "#1f77b4",  # Blue
        "#2ca02c",  # Green
        "#9467bd",  # Purple
        "#8c564b",  # Brown
        "#e377c2",  # Pink
        "#17becf",  # Cyan
    ]
    
    node_coords = snap_nodes.set_index("node_id")[["x", "y"]].to_dict(orient="index")
    
    # 1. Draw intra-cluster and inter-cluster edges
    for _, edge in snap_edges.iterrows():
        u, v = int(edge["src"]), int(edge["dst"])
        if u in node_coords and v in node_coords:
            p1 = (node_coords[u]["x"], node_coords[u]["y"])
            p2 = (node_coords[v]["x"], node_coords[v]["y"])
            
            if edge.get("edge_type") == "wormhole":
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="#d62728", linestyle="--", linewidth=1.8, alpha=0.85, zorder=2)
            elif edge.get("is_intra_cluster", 0) == 1:
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="#bcbcbc", linestyle="-", linewidth=0.6, alpha=0.45, zorder=1)
            else:
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="#dcdcdc", linestyle=":", linewidth=0.4, alpha=0.3, zorder=1)

    # 2. Draw convex hulls around each cluster
    for c_id, c_nodes in snap_nodes.groupby("cluster_id"):
        color = cluster_colors[int(c_id) % len(cluster_colors)]
        pts = c_nodes[["x", "y"]].to_numpy(dtype=float)
        
        if len(pts) >= 3:
            try:
                hull = ConvexHull(pts)
                hull_pts = pts[hull.vertices]
                ax.fill(hull_pts[:, 0], hull_pts[:, 1], color=color, alpha=0.12, zorder=0)
                ax.plot(
                    np.append(hull_pts[:, 0], hull_pts[0, 0]),
                    np.append(hull_pts[:, 1], hull_pts[0, 1]),
                    color=color, linestyle="--", linewidth=1.2, alpha=0.5, zorder=0
                )
            except Exception:
                pass

    # 3. Plot member nodes
    normal_nodes = snap_nodes[(snap_nodes["is_cluster_head"] == 0) & (snap_nodes["label"] == 0)]
    malicious_nodes = snap_nodes[(snap_nodes["is_cluster_head"] == 0) & (snap_nodes["label"] == 1)]
    cluster_heads = snap_nodes[snap_nodes["is_cluster_head"] == 1]
    
    # Plot normal member nodes colored by cluster
    for c_id, c_nodes in normal_nodes.groupby("cluster_id"):
        color = cluster_colors[int(c_id) % len(cluster_colors)]
        ax.scatter(
            c_nodes["x"], c_nodes["y"],
            c=color, s=55, marker="o", edgecolors="black", linewidths=0.6,
            alpha=0.85, zorder=3, label=f"Cluster {c_id}" if time_s == 0 else None
        )
        
    # Plot malicious nodes if any
    if not malicious_nodes.empty:
        for atk_type, atk_nodes in malicious_nodes.groupby("attack"):
            marker = "X" if atk_type == "blackhole" else ("^" if atk_type == "greyhole" else "s")
            atk_color = "#d62728" if atk_type == "blackhole" else ("#ff7f0e" if atk_type == "greyhole" else "#9467bd")
            ax.scatter(
                atk_nodes["x"], atk_nodes["y"],
                c=atk_color, s=110, marker=marker, edgecolors="black", linewidths=1.2,
                alpha=0.95, zorder=4, label=f"Malicious: {atk_type.capitalize()}"
            )

    # 4. Plot elected Cluster Heads prominently as stars
    for _, ch in cluster_heads.iterrows():
        c_id = int(ch["cluster_id"])
        ch_color = cluster_colors[c_id % len(cluster_colors)]
        is_mal = ch["label"] == 1
        
        ax.scatter(
            ch["x"], ch["y"],
            c=ch_color, s=320, marker="*", edgecolors="gold" if not is_mal else "red",
            linewidths=2.0, alpha=1.0, zorder=5
        )
        # Label the CH
        ax.annotate(
            f"CH-{c_id}\n(N{int(ch['node_id'])})",
            (ch["x"], ch["y"]),
            textcoords="offset points", xytext=(0, 12),
            ha="center", fontsize=8, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gold", alpha=0.9),
            zorder=6
        )

    # Dummy scatter for Cluster Head legend entry
    ax.scatter([], [], c="black", s=200, marker="*", edgecolors="gold", linewidths=1.5, label="Elected Cluster Head (CH)")

    # Title and axes
    mal_count = snap_nodes["label"].sum()
    ax.set_title(
        f"Dynamic MANET Cluster-Aware Topology — Run {run_id}, t = {int(time_s)}s\n"
        f"[Total Nodes: {len(snap_nodes)} | Clusters: {snap_nodes['cluster_id'].nunique()} | "
        f"Cluster Heads: {len(cluster_heads)} | Malicious: {mal_count}]",
        fontsize=13, fontweight="bold", pad=12
    )
    ax.set_xlabel("X Coordinate (meters)", fontsize=11)
    ax.set_ylabel("Y Coordinate (meters)", fontsize=11)
    ax.set_xlim(-20, 1020)
    ax.set_ylim(-20, 1020)
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9)

    plt.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Generated cluster topology plot: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate cluster topology visualizations.")
    parser.add_argument("--nodes", default="data/processed_clustered/nodes_clustered.csv")
    parser.add_argument("--edges", default="data/processed_clustered/edges_clustered.csv")
    parser.add_argument("--run", type=int, default=1)
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()
    
    print(f"Loading data for visualization...")
    nodes_df = pd.read_csv(args.nodes)
    edges_df = pd.read_csv(args.edges)
    
    out_dir = Path(args.out_dir)
    
    # 1. Snapshot at t = 0s
    plot_cluster_snapshot(
        nodes_df=nodes_df,
        edges_df=edges_df,
        run_id=args.run,
        time_s=0.0,
        output_path=str(out_dir / "cluster_visualization_t0.png")
    )
    
    # 2. Snapshot at t = 350s
    plot_cluster_snapshot(
        nodes_df=nodes_df,
        edges_df=edges_df,
        run_id=args.run,
        time_s=350.0,
        output_path=str(out_dir / "cluster_visualization_t350.png")
    )


if __name__ == "__main__":
    main()
