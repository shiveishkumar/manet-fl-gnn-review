from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support
)
from sklearn.preprocessing import StandardScaler

from torch_geometric.nn import (
    GCNConv,
    SAGEConv,
    GATConv
)


# ============================================================
# CONFIGURATION
# ============================================================

FEATURES = [
    "drop_rate",
    "trust",
    "forward_ratio",
    "energy"
]

HIDDEN_CHANNELS = 16
NUM_CLASSES = 2


# ============================================================
# METRICS
# ============================================================

def metrics(y, pred):

    p, r, f1, _ = precision_recall_fscore_support(
        y,
        pred,
        labels=[1],
        average=None,
        zero_division=0
    )

    return (
        accuracy_score(y, pred),
        float(p[0]),
        float(r[0]),
        float(f1[0])
    )


# ============================================================
# TEMPORAL SPLIT
# ============================================================

def split_temporal(run_df):

    times = sorted(run_df.time.unique())

    cut = int(len(times) * 0.70)

    train_times = times[:cut]
    test_times = times[cut:]

    train = run_df[
        run_df.time.isin(train_times)
    ]

    test = run_df[
        run_df.time.isin(test_times)
    ]

    if test.label.nunique() < 2:
        raise ValueError(
            "Temporal test set must contain "
            "normal and malicious nodes"
        )

    return train, test, train_times, test_times


# ============================================================
# GRAPH CONSTRUCTION
# ============================================================

def build_edge_index(edge_df):

    if edge_df.empty:
        return torch.empty(
            (2, 0),
            dtype=torch.long
        )

    src = edge_df.src.to_numpy(
        dtype=np.int64
    )

    dst = edge_df.dst.to_numpy(
        dtype=np.int64
    )

    # MANET links are treated as undirected.
    src_all = np.concatenate([src, dst])
    dst_all = np.concatenate([dst, src])

    return torch.tensor(
        np.vstack([src_all, dst_all]),
        dtype=torch.long
    )


# ============================================================
# STATIC GCN
# ============================================================

class StaticGCN(torch.nn.Module):

    def __init__(
        self,
        in_channels=4,
        hidden_channels=16,
        out_channels=2
    ):

        super().__init__()

        self.conv1 = GCNConv(
            in_channels,
            hidden_channels
        )

        self.conv2 = GCNConv(
            hidden_channels,
            out_channels
        )

    def forward(
        self,
        x,
        edge_index
    ):

        x = self.conv1(
            x,
            edge_index
        )

        x = torch.relu(x)

        x = self.conv2(
            x,
            edge_index
        )

        return x


# ============================================================
# GRAPH SAGE
# ============================================================

class GraphSAGE(torch.nn.Module):

    def __init__(
        self,
        in_channels=4,
        hidden_channels=16,
        out_channels=2
    ):

        super().__init__()

        self.conv1 = SAGEConv(
            in_channels,
            hidden_channels
        )

        self.conv2 = SAGEConv(
            hidden_channels,
            out_channels
        )

    def forward(
        self,
        x,
        edge_index
    ):

        x = self.conv1(
            x,
            edge_index
        )

        x = torch.relu(x)

        x = self.conv2(
            x,
            edge_index
        )

        return x


# ============================================================
# GAT
# ============================================================

class GAT(torch.nn.Module):

    def __init__(
        self,
        in_channels=4,
        hidden_channels=16,
        out_channels=2
    ):

        super().__init__()

        self.conv1 = GATConv(
            in_channels,
            hidden_channels,
            heads=1
        )

        self.conv2 = GATConv(
            hidden_channels,
            out_channels,
            heads=1
        )

    def forward(
        self,
        x,
        edge_index
    ):

        x = self.conv1(
            x,
            edge_index
        )

        x = torch.relu(x)

        x = self.conv2(
            x,
            edge_index
        )

        return x


# ============================================================
# EVOLVEGCN-H STYLE MODEL
# ============================================================
#
# The official torch_geometric_temporal EvolveGCNH package
# could not be installed in the current Python 3.13 environment
# because its torch-sparse dependency failed to build.
#
# This implementation therefore keeps the EvolveGCN-H idea:
# the model maintains a hidden temporal state which evolves
# from one graph snapshot to the next.
#
# ============================================================

class EvolveGCNH(torch.nn.Module):

    def __init__(
        self,
        in_channels=4,
        hidden_channels=16,
        out_channels=2
    ):

        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels

        # Initial GCN weight state.
        self.initial_weight = torch.nn.Parameter(
            torch.empty(
                in_channels,
                hidden_channels
            )
        )

        torch.nn.init.xavier_uniform_(
            self.initial_weight
        )

        # Convert graph-level information into
        # the same size as the flattened GCN weight.
        weight_size = (
            in_channels * hidden_channels
        )

        self.graph_encoder = torch.nn.Linear(
            hidden_channels,
            weight_size
        )

        # GRU evolves the GCN weight state.
        self.weight_gru = torch.nn.GRUCell(
            weight_size,
            weight_size
        )

        # Bias for the evolving first layer.
        self.initial_bias = torch.nn.Parameter(
            torch.zeros(hidden_channels)
        )

        self.output_conv = GCNConv(
            hidden_channels,
            out_channels
        )

    def forward(
        self,
        x,
        edge_index,
        hidden
    ):

        # Graph representation.
        graph_x = torch.relu(
            torch.matmul(
                x,
                self.initial_weight
                if hidden is None
                else hidden.view(
                    self.in_channels,
                    self.hidden_channels
                )
            )
        )

        pooled = graph_x.mean(
            dim=0
        )

        # Create input for the recurrent weight update.
        gru_input = self.graph_encoder(
            pooled
        )

        if hidden is None:
            hidden = self.initial_weight.reshape(-1)

        # Evolve the GCN weight state.
        hidden = self.weight_gru(
            gru_input.unsqueeze(0),
            hidden.unsqueeze(0)
        ).squeeze(0)

        evolving_weight = hidden.reshape(
            self.in_channels,
            self.hidden_channels
        )

        # First graph convolution using evolved weight.
        x = torch.matmul(
            x,
            evolving_weight
        )

        x = x + self.initial_bias

        x = torch.relu(x)

        # Second GCN layer.
        x = self.output_conv(
            x,
            edge_index
        )

        return x, hidden


# ============================================================
# STATIC MODEL TRAINING
# ============================================================

def train_static_model(
    model,
    snapshots,
    train_times,
    epochs=50,
    lr=0.01
):

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr
    )

    criterion = torch.nn.CrossEntropyLoss()

    model.train()

    for epoch in range(epochs):

        total_loss = 0.0

        for time_s in train_times:

            x, edge_index, y, _ = snapshots[
                time_s
            ]

            optimizer.zero_grad()

            out = model(
                x,
                edge_index
            )

            loss = criterion(
                out,
                y
            )

            loss.backward()

            optimizer.step()

            total_loss += float(
                loss.item()
            )

    return model


# ============================================================
# EVOLVEGCN-H TRAINING
# ============================================================

def train_evolvegcn(
    model,
    snapshots,
    train_times,
    device,
    epochs=50,
    lr=0.01
):

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr
    )

    criterion = torch.nn.CrossEntropyLoss()

    model.train()

    for epoch in range(epochs):

        hidden = None

        for time_s in train_times:

            x, edge_index, y, _ = snapshots[
                time_s
            ]

            optimizer.zero_grad()

            out, hidden = model(
                x,
                edge_index,
                hidden
            )

            loss = criterion(
                out,
                y
            )

            loss.backward()

            optimizer.step()

            # Detach temporal state so that the graph from
            # previous snapshots is not retained indefinitely.
            hidden = hidden.detach()

    return model


# ============================================================
# STATIC MODEL EVALUATION
# ============================================================

def evaluate_static_model(
    model,
    snapshots,
    test_times
):

    model.eval()

    all_labels = []
    all_predictions = []
    all_nodes = []
    all_times = []

    with torch.no_grad():

        for time_s in test_times:

            x, edge_index, y, node_ids = snapshots[
                time_s
            ]

            out = model(
                x,
                edge_index
            )

            pred = out.argmax(
                dim=1
            )

            all_labels.extend(
                y.cpu().numpy()
            )

            all_predictions.extend(
                pred.cpu().numpy()
            )

            all_nodes.extend(
                node_ids
            )

            all_times.extend(
                [time_s] * len(node_ids)
            )

    return (
        np.asarray(all_labels),
        np.asarray(all_predictions),
        np.asarray(all_nodes),
        np.asarray(all_times)
    )


# ============================================================
# EVOLVEGCN-H EVALUATION
# ============================================================

def evaluate_evolvegcn(
    model,
    snapshots,
    train_times,
    test_times
):

    model.eval()

    hidden = None

    with torch.no_grad():

        # Process training sequence first so that the temporal
        # state entering the test period reflects the past.
        for time_s in train_times:

            x, edge_index, _, _ = snapshots[
                time_s
            ]

            _, hidden = model(
                x,
                edge_index,
                hidden
            )

            hidden = hidden.detach()

        all_labels = []
        all_predictions = []
        all_nodes = []
        all_times = []

        for time_s in test_times:

            x, edge_index, y, node_ids = snapshots[
                time_s
            ]

            out, hidden = model(
                x,
                edge_index,
                hidden
            )

            hidden = hidden.detach()

            pred = out.argmax(
                dim=1
            )

            all_labels.extend(
                y.cpu().numpy()
            )

            all_predictions.extend(
                pred.cpu().numpy()
            )

            all_nodes.extend(
                node_ids
            )

            all_times.extend(
                [time_s] * len(node_ids)
            )

    return (
        np.asarray(all_labels),
        np.asarray(all_predictions),
        np.asarray(all_nodes),
        np.asarray(all_times)
    )


# ============================================================
# SNAPSHOT PREPARATION
# ============================================================

def prepare_snapshots(
    run_nodes,
    run_edges,
    scaler
):

    snapshots = {}

    for time_s in sorted(
        run_nodes.time.unique()
    ):

        node_df = (
            run_nodes[
                run_nodes.time == time_s
            ]
            .sort_values("node_id")
        )

        edge_df = run_edges[
            run_edges.time == time_s
        ]

        X = scaler.transform(
            node_df[FEATURES].to_numpy(
                dtype=float
            )
        )

        y = node_df.label.to_numpy(
            dtype=np.int64
        )

        node_ids = node_df.node_id.to_numpy(
            dtype=np.int64
        )

        x = torch.tensor(
            X,
            dtype=torch.float32
        )

        edge_index = build_edge_index(
            edge_df
        )

        labels = torch.tensor(
            y,
            dtype=torch.long
        )

        snapshots[time_s] = (
            x,
            edge_index,
            labels,
            node_ids
        )

    return snapshots


# ============================================================
# SCALER
# ============================================================

def fit_scaler(train_df):

    scaler = StandardScaler()

    scaler.fit(
        train_df[FEATURES].to_numpy(
            dtype=float
        )
    )

    return scaler


# ============================================================
# MODEL FACTORY
# ============================================================

def create_model(name):

    if name == "Static-GCN":
        return StaticGCN()

    if name == "GraphSAGE":
        return GraphSAGE()

    if name == "GAT":
        return GAT()

    if name == "EvolveGCN-H":
        return EvolveGCNH()

    raise ValueError(
        f"Unknown model: {name}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--nodes",
        default="data/processed/nodes_dynamic.csv"
    )

    ap.add_argument(
        "--edges",
        default="data/processed/edges_dynamic.csv"
    )

    ap.add_argument(
        "--out",
        default="results/gnn_results.csv"
    )

    ap.add_argument(
        "--epochs",
        type=int,
        default=50
    )

    args = ap.parse_args()

    nodes = pd.read_csv(
        args.nodes
    )

    edges = pd.read_csv(
        args.edges
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    if torch.cuda.is_available():

        device = torch.device("cuda")

    elif torch.backends.mps.is_available():

        device = torch.device("mps")

    else:

        device = torch.device("cpu")

    print(
        f"Using device: {device}"
    )

    # --------------------------------------------------------
    # Output containers
    # --------------------------------------------------------

    rows = []

    prediction_rows = []

    temporal_rows = []

    models = [
        "Static-GCN",
        "GraphSAGE",
        "GAT",
        "EvolveGCN-H"
    ]

    # --------------------------------------------------------
    # Process every run
    # --------------------------------------------------------

    for run_id, run_nodes in nodes.groupby(
        "run"
    ):

        run_id = int(run_id)

        print(
            f"\n========== RUN {run_id} =========="
        )

        run_edges = edges[
            edges.run == run_id
        ]

        # ----------------------------------------------------
        # 70/30 temporal split
        # ----------------------------------------------------

        (
            train_df,
            test_df,
            train_times,
            test_times
        ) = split_temporal(
            run_nodes
        )

        print(
            f"Train snapshots: {len(train_times)}"
        )

        print(
            f"Test snapshots: {len(test_times)}"
        )

        # ----------------------------------------------------
        # Fit scaler ONLY on training data
        # ----------------------------------------------------

        scaler = fit_scaler(
            train_df
        )

        snapshots = prepare_snapshots(
            run_nodes,
            run_edges,
            scaler
        )

        # Move snapshots to device.
        device_snapshots = {}

        for time_s, (
            x,
            edge_index,
            y,
            node_ids
        ) in snapshots.items():

            device_snapshots[time_s] = (
                x.to(device),
                edge_index.to(device),
                y.to(device),
                node_ids
            )

        static_predictions = None
        static_labels = None
        static_nodes = None
        static_times = None

        evolve_predictions = None
        evolve_labels = None
        evolve_nodes = None
        evolve_times = None

        # ----------------------------------------------------
        # Train all four models
        # ----------------------------------------------------

        for name in models:

            torch.manual_seed(
                run_id
            )

            np.random.seed(
                run_id
            )

            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(
                    run_id
                )

            model = create_model(
                name
            ).to(device)

            if name == "EvolveGCN-H":

                train_evolvegcn(
                    model,
                    device_snapshots,
                    train_times,
                    device,
                    epochs=args.epochs
                )

                (
                    y_true,
                    pred,
                    node_ids,
                    time_ids
                ) = evaluate_evolvegcn(
                    model,
                    device_snapshots,
                    train_times,
                    test_times
                )

            else:

                train_static_model(
                    model,
                    device_snapshots,
                    train_times,
                    epochs=args.epochs
                )

                (
                    y_true,
                    pred,
                    node_ids,
                    time_ids
                ) = evaluate_static_model(
                    model,
                    device_snapshots,
                    test_times
                )

            # ------------------------------------------------
            # Metrics
            # ------------------------------------------------

            (
                acc,
                precision,
                recall,
                f1
            ) = metrics(
                y_true,
                pred
            )

            rows.append([
                run_id,
                name,
                acc,
                precision,
                recall,
                f1
            ])

            print(
                run_id,
                name,
                f"acc={acc:.4f}",
                f"mal_precision={precision:.4f}",
                f"mal_recall={recall:.4f}",
                f"mal_f1={f1:.4f}"
            )

            # ------------------------------------------------
            # Detailed predictions
            # ------------------------------------------------

            for node_id, time_id, true_label, predicted_label in zip(
                node_ids,
                time_ids,
                y_true,
                pred
            ):

                prediction_rows.append([
                    run_id,
                    name,
                    float(time_id),
                    int(node_id),
                    int(true_label),
                    int(predicted_label)
                ])

            # ------------------------------------------------
            # Save Static-GCN predictions for comparison
            # ------------------------------------------------

            if name == "Static-GCN":

                static_predictions = pred.copy()
                static_labels = y_true.copy()
                static_nodes = node_ids.copy()
                static_times = time_ids.copy()

            # ------------------------------------------------
            # Save EvolveGCN-H predictions
            # ------------------------------------------------

            if name == "EvolveGCN-H":

                evolve_predictions = pred.copy()
                evolve_labels = y_true.copy()
                evolve_nodes = node_ids.copy()
                evolve_times = time_ids.copy()

        # ====================================================
        # TEMPORAL ADVANTAGE
        # ====================================================
        #
        # Count SAME node-time cases where:
        #
        #   actual label = malicious
        #   Static-GCN = normal
        #   EvolveGCN-H = malicious
        #
        # ====================================================

        static_df = pd.DataFrame({
            "time": static_times,
            "node_id": static_nodes,
            "label": static_labels,
            "static_prediction": static_predictions
        })

        evolve_df = pd.DataFrame({
            "time": evolve_times,
            "node_id": evolve_nodes,
            "label": evolve_labels,
            "evolve_prediction": evolve_predictions
        })

        comparison = static_df.merge(
            evolve_df[
                [
                    "time",
                    "node_id",
                    "evolve_prediction"
                ]
            ],
            on=[
                "time",
                "node_id"
            ],
            how="inner"
        )

        temporal_advantage = comparison[
            (comparison["label"] == 1)
            &
            (comparison["static_prediction"] == 0)
            &
            (comparison["evolve_prediction"] == 1)
        ]

        static_missed = comparison[
            (comparison["label"] == 1)
            &
            (comparison["static_prediction"] == 0)
        ]

        evolve_caught = comparison[
            (comparison["label"] == 1)
            &
            (comparison["evolve_prediction"] == 1)
        ]

        temporal_rows.append([
            run_id,
            int(len(static_missed)),
            int(len(evolve_caught)),
            int(len(temporal_advantage))
        ])

        print(
            f"Temporal advantage cases: "
            f"{len(temporal_advantage)}"
        )

    # ========================================================
    # SAVE MAIN RESULTS
    # ========================================================

    out = pd.DataFrame(
        rows,
        columns=[
            "run",
            "model",
            "accuracy",
            "mal_precision",
            "mal_recall",
            "mal_f1"
        ]
    )

    out_path = Path(
        args.out
    )

    out_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    out.to_csv(
        out_path,
        index=False
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    summary = (
        out.groupby("model")
        [
            [
                "accuracy",
                "mal_precision",
                "mal_recall",
                "mal_f1"
            ]
        ]
        .agg(["mean", "std"])
    )

    summary_path = out_path.with_name(
        "gnn_summary.csv"
    )

    summary.to_csv(
        summary_path
    )

    # ========================================================
    # DETAILED PREDICTIONS
    # ========================================================

    prediction_df = pd.DataFrame(
        prediction_rows,
        columns=[
            "run",
            "model",
            "time",
            "node_id",
            "label",
            "prediction"
        ]
    )

    for run_id in sorted(
        prediction_df.run.unique()
    ):

        prediction_df[
            prediction_df.run == run_id
        ].to_csv(
            out_path.with_name(
                f"prediction_detail_run{run_id}.csv"
            ),
            index=False
        )

    # ========================================================
    # TEMPORAL ADVANTAGE OUTPUT
    # ========================================================

    temporal_df = pd.DataFrame(
        temporal_rows,
        columns=[
            "run",
            "static_gcn_missed_malicious",
            "evolvegcn_caught_malicious",
            "same_node_time_caught_by_evolvegcn"
        ]
    )

    temporal_path = out_path.with_name(
        "temporal_advantage.csv"
    )

    temporal_df.to_csv(
        temporal_path,
        index=False
    )

    # ========================================================
    # FINAL DISPLAY
    # ========================================================

    print(
        "\nMEAN +/- STD OVER RUNS\n"
    )

    print(summary)

    print(
        "\nSaved:",
        out_path
    )

    print(
        "Saved:",
        summary_path
    )

    print(
        "Saved prediction detail files"
    )

    print(
        "Saved:",
        temporal_path
    )


if __name__ == "__main__":
    main()
