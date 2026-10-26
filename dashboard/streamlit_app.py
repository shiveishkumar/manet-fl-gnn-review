import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(
    page_title="MANET Intrusion Detection - Review 3",
    layout="wide"
)

st.title("Dynamic MANET Intrusion Detection — Review 3")

st.write(
    "Interactive visualization of the dynamic MANET dataset with "
    "cluster-aware analysis, centralized GNN results, lightweight "
    "federated GCN evaluation, scalability, and communication-cost analysis."
)


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def load_data():

    nodes = pd.read_csv("data/processed/nodes_dynamic.csv")
    edges = pd.read_csv("data/processed/edges_dynamic.csv")

    # ---------------- FL-GCN Review 3 ----------------

    fl = pd.read_csv(
        "results/fl_gcn_review3_summary.csv"
    )

    thresholds = pd.read_csv(
        "results/fl_threshold_comparison.csv"
    )

    communication = pd.read_csv(
        "results/communication_cost.csv"
    )

    # ---------------- GNN Review 3 ----------------
    # Files have two-level CSV headers, so parse manually.

    gnn = pd.read_csv(
        "results/gnn_review3_summary.csv",
        header=[0, 1],
        index_col=0
    )

    gnn.columns = [
        "Accuracy Mean",
        "Accuracy Std",
        "Malicious Precision Mean",
        "Malicious Precision Std",
        "Malicious Recall Mean",
        "Malicious Recall Std",
        "Malicious F1 Mean",
        "Malicious F1 Std",
        "Training Time Mean",
        "Training Time Std",
        "Inference Time Mean",
        "Inference Time Std",
        "Parameters Mean",
        "Parameters Std"
    ]

    gnn = (
        gnn
        .reset_index()
        .rename(columns={"model": "Model"})
    )

    # ---------------- Cluster Baselines ----------------

    cluster = pd.read_csv(
        "results/cluster_baseline_summary.csv",
        header=[0, 1],
        index_col=[0, 1]
    )

    cluster.columns = [
        "Accuracy Mean",
        "Accuracy Std",
        "Malicious Precision Mean",
        "Malicious Precision Std",
        "Malicious Recall Mean",
        "Malicious Recall Std",
        "Malicious F1 Mean",
        "Malicious F1 Std"
    ]

    cluster = (
        cluster
        .reset_index()
        .rename(
            columns={
                "feature_set": "Feature Set",
                "model": "Model"
            }
        )
    )

    # ---------------- Scalability ----------------

    scalability = pd.read_csv(
        "results/scalability_review3.csv",
        header=[0, 1],
        index_col=[0, 1]
    )

    scalability.columns = [
        "Accuracy Mean",
        "Accuracy Std",
        "Malicious Recall Mean",
        "Malicious Recall Std",
        "Malicious F1 Mean",
        "Malicious F1 Std",
        "Training Time Mean",
        "Training Time Std",
        "Inference Time Mean",
        "Inference Time Std"
    ]

    scalability = (
        scalability
        .reset_index()
        .rename(
            columns={
                "model": "Model",
                "nodes": "Nodes"
            }
        )
    )

    return (
        nodes,
        edges,
        fl,
        thresholds,
        communication,
        gnn,
        cluster,
        scalability
    )


(
    nodes,
    edges,
    fl_summary,
    threshold_summary,
    communication_summary,
    gnn_summary,
    cluster_summary,
    scalability_summary
) = load_data()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Simulation Controls")

runs = sorted(nodes["run"].unique())

selected_run = st.sidebar.selectbox(
    "Select simulation run",
    runs
)

run_nodes = nodes[
    nodes["run"] == selected_run
]

times = sorted(run_nodes["time"].unique())

selected_time = st.sidebar.select_slider(
    "Select time (seconds)",
    options=times
)


# ============================================================
# SNAPSHOT DATA
# ============================================================

snapshot_nodes = nodes[
    (nodes["run"] == selected_run) &
    (nodes["time"] == selected_time)
].copy()

snapshot_edges = edges[
    (edges["run"] == selected_run) &
    (edges["time"] == selected_time)
].copy()

normal_count = (
    snapshot_nodes["label"] == 0
).sum()

malicious_count = (
    snapshot_nodes["label"] == 1
).sum()


# ============================================================
# TOP SUMMARY METRICS
# ============================================================

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Run",
    int(selected_run)
)

col2.metric(
    "Time",
    f"{selected_time} s"
)

col3.metric(
    "Normal Nodes",
    int(normal_count)
)

col4.metric(
    "Malicious Nodes",
    int(malicious_count)
)


# ============================================================
# POSITION COLUMN DETECTION
# ============================================================

x_col = None
y_col = None

for name in ["x", "x_pos", "pos_x", "x_position"]:
    if name in snapshot_nodes.columns:
        x_col = name
        break

for name in ["y", "y_pos", "pos_y", "y_position"]:
    if name in snapshot_nodes.columns:
        y_col = name
        break

if x_col is None or y_col is None:

    snapshot_nodes["plot_x"] = (
        snapshot_nodes["node_id"] % 15
    )

    snapshot_nodes["plot_y"] = (
        snapshot_nodes["node_id"] // 15
    )

    x_col = "plot_x"
    y_col = "plot_y"


# ============================================================
# NETWORK GRAPH
# ============================================================

fig = go.Figure()

positions = {}

for _, row in snapshot_nodes.iterrows():

    positions[int(row["node_id"])] = (
        row[x_col],
        row[y_col]
    )


for _, edge in snapshot_edges.iterrows():

    src = int(edge["src"])
    dst = int(edge["dst"])

    if src in positions and dst in positions:

        x0, y0 = positions[src]
        x1, y1 = positions[dst]

        fig.add_trace(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines",
                line=dict(
                    width=0.5,
                    color="rgba(150,150,150,0.20)"
                ),
                hoverinfo="skip",
                showlegend=False
            )
        )


# Normal nodes

normal = snapshot_nodes[
    snapshot_nodes["label"] == 0
]

fig.add_trace(
    go.Scatter(
        x=normal[x_col],
        y=normal[y_col],
        mode="markers",
        name="Normal",
        marker=dict(
            size=9
        ),
        text=[
            f"Node {node_id}"
            for node_id in normal["node_id"]
        ],
        hovertemplate="%{text}<extra></extra>"
    )
)


# Malicious nodes

malicious = snapshot_nodes[
    snapshot_nodes["label"] == 1
]

fig.add_trace(
    go.Scatter(
        x=malicious[x_col],
        y=malicious[y_col],
        mode="markers",
        name="Malicious",
        marker=dict(
            size=13,
            symbol="x"
        ),
        text=[
            f"Node {node_id}<br>Attack: {attack}"
            for node_id, attack in zip(
                malicious["node_id"],
                malicious["attack"]
            )
        ],
        hovertemplate="%{text}<extra></extra>"
    )
)


fig.update_layout(
    title=(
        f"Dynamic MANET — Run {selected_run}, "
        f"Time {selected_time}s"
    ),
    height=650,
    xaxis_title="X Position",
    yaxis_title="Y Position",
    showlegend=True
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# ============================================================
# ATTACK DISTRIBUTION
# ============================================================

st.subheader("Attack Distribution")

if malicious_count == 0:

    st.info(
        "No malicious nodes are active at this timestamp."
    )

else:

    attack_counts = (
        malicious["attack"]
        .value_counts()
        .reset_index()
    )

    attack_counts.columns = [
        "Attack",
        "Nodes"
    ]

    st.dataframe(
        attack_counts,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# REVIEW 3 OVERVIEW
# ============================================================

st.divider()

st.header("Review 3 Improvements")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "NS-3 Validation",
    "10 / 10 Passed"
)

c2.metric(
    "Clusters per Snapshot",
    "6"
)

c3.metric(
    "FL-GCN Parameters",
    "114"
)

c4.metric(
    "FL Communication",
    "22.27 KB"
)

st.write(
    """
    Review 3 improves the project through four major additions:

    - validated NS-3 mobility and scalability datasets,
    - cluster-aware MANET representation with cluster-head election,
    - improved centralized and temporal GNN evaluation,
    - validation-based FL-GCN threshold analysis and communication-cost measurement.
    """
)


# ============================================================
# CLUSTER-AWARE BASELINES
# ============================================================

st.divider()

st.header("Cluster-Aware Classical Baselines")

st.write(
    "Review 3 adds cluster-level topology information such as "
    "cluster membership, cluster-head status, cluster size, and "
    "distance to the elected cluster head."
)

st.dataframe(
    cluster_summary,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# CENTRALIZED / TEMPORAL GNN RESULTS
# ============================================================

st.divider()

st.header("Centralized and Temporal GNN Results")

st.dataframe(
    gnn_summary,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# FL-GCN REVIEW 3
# ============================================================

st.divider()

st.header("Federated Lightweight GCN")

st.write(
    """
    The federated model retains the lightweight **4 → 16 → 2**
    architecture with only **114 trainable parameters**.
    Five logical clients train locally and are aggregated using FedAvg.
    """
)

st.subheader("Final 5-Run FL-GCN Results")

st.dataframe(
    fl_summary,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# THRESHOLD TUNING
# ============================================================

st.subheader("Validation-Based Threshold Tuning")

st.dataframe(
    threshold_summary,
    use_container_width=True,
    hide_index=True
)

best_row = threshold_summary.loc[
    threshold_summary["Malicious F1"].idxmax()
]

st.success(
    f"Selected threshold: "
    f"{best_row['Threshold']:.1f} "
    f"(Validation Malicious F1 = "
    f"{best_row['Malicious F1']:.4f})"
)


# ============================================================
# COMMUNICATION COST
# ============================================================

st.subheader("Federated Communication Cost")

st.dataframe(
    communication_summary,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# SCALABILITY
# ============================================================

st.divider()

st.header("Scalability Evaluation")

st.write(
    "Centralized and temporal graph models were evaluated from "
    "100 to 500 MANET nodes."
)

st.dataframe(
    scalability_summary,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# REVIEW 2 VS REVIEW 3
# ============================================================

st.divider()

st.header("Review 2 → Review 3 Progress")

comparison = pd.DataFrame({
    "Review 2 Limitation": [
        "Flat node representation",
        "Unvalidated FL decision threshold",
        "Weak temporal GNN malicious detection",
        "Limited scalability evidence",
        "No FL communication-cost analysis"
    ],

    "Review 3 Improvement": [
        "Cluster-aware hierarchy + cluster-head election",
        "Validation threshold search from 0.3 to 0.6",
        "Audited and tuned EvolveGCN-H",
        "Evaluation from 100 to 500 nodes",
        "Quantified 22.27 KB federated communication"
    ]
})

st.dataframe(
    comparison,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# FOOTER
# ============================================================

st.caption(
    "Review 3 dashboard — visualization of saved simulation, "
    "clustering, GNN, federated-learning, and scalability results. "
    "The dashboard replays saved experiment outputs and does not "
    "run NS-3 or model training live."
)
