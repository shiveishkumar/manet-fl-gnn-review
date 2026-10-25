# Review 3: Dataset, Clustering, Cluster-Head Selection, and Classical Baselines

**Author:** Raghav Bajaj (24BCE2068)  
**Project:** Dynamic Federated Graph Neural Network for Intrusion Detection in Mobile Ad Hoc Networks  
**Scope:** Review 3 Assigned Module — Dataset Integrity, MANET Clustering, Multi-Criteria Cluster-Head Election, and Baseline Comparison  

---

## 1. Executive Summary

In **Review 2**, the MANET dataset treated nodes as isolated, flat devices characterized only by local packet/traffic counters. While classical ML baselines (Random Forest, SVM, XGBoost) achieved baseline detection, they lacked awareness of local topology, node hierarchy, or cluster structure.

In **Review 3**, I extended the dynamic dataset into a **cluster-aware hierarchical MANET representation**:
1. Conducted an exhaustive dataset integrity audit on `nodes_dynamic.csv` and `edges_dynamic.csv` verifying zero missing values, zero duplicate keys, and strict temporal alignment.
2. Implemented independent snapshot clustering (`python/cluster_nodes.py`) partitioning the 1000m × 1000m area into explainable topological clusters at every 10-second timestep.
3. Designed and implemented transparent multi-criteria Cluster Head (CH) election (`python/select_cluster_heads.py`) using residual energy, trust, degree, and distance centrality.
4. Generated the Review 3 cluster-aware dataset (`data/processed_clustered/nodes_clustered.csv` and `edges_clustered.csv`), preserving Review 2 files intact.
5. Re-trained Random Forest, SVM, and XGBoost under the exact chronological split (0–340 s train, 350–500 s test) comparing Review 2 vs Review 3 performance.
6. Produced high-resolution cluster topology plots at $t = 0\text{ s}$ and $t = 350\text{ s}$ (`results/cluster_visualization_t0.png` and `results/cluster_visualization_t350.png`).

---

## 2. Methodology & Multi-Criteria Cluster-Head Election

### 2.1 Snapshot-by-Snapshot Spatial Clustering
At each time snapshot $t \in \{0, 10, 20, \dots, 500\text{ s}\}$ for each simulation run, nodes move according to NS-3 Random Walk mobility. Nodes are partitioned into $K=6$ geographic clusters using spatial coordinates $(x, y)$. This groups nodes into local communication neighborhoods of approximately 18–35 nodes per cluster, matching typical radio coverage ($250\text{ m}$).

### 2.2 Transparent Multi-Criteria Cluster Head Score
Within each cluster $C$ at snapshot $(run, time)$, candidate nodes are evaluated using the weighted scoring formula:

$$\text{CH\_score}_i = 0.30 \times \text{norm\_energy}_i + 0.30 \times \text{norm\_trust}_i + 0.20 \times \text{norm\_degree}_i + 0.20 \times \text{norm\_inv\_dist}_i$$

Where:
* **$\text{norm\_energy}_i$**: Normalized residual battery level $\in [0, 1]$. Nodes with higher remaining energy can handle coordination duties.
* **$\text{norm\_trust}_i$**: Normalized trust score $\in [0, 1]$. Prevents compromised or misbehaving nodes from becoming cluster heads.
* **$\text{norm\_degree}_i$**: Normalized node connectivity (number of radio neighbors). Ensures the CH can communicate directly with many peers.
* **$\text{norm\_inv\_dist}_i$**: Normalized inverse average distance to all other members in the cluster:
  $$\bar{d}_i = \frac{1}{|C|-1} \sum_{j \in C, j \neq i} \text{dist}(P_i, P_j), \quad \text{inv\_dist}_i = \frac{1}{1 + \bar{d}_i}$$
  Nodes positioned near the cluster centroid receive higher centrality scores.

The node with the maximum $\text{CH\_score}$ is elected as the Cluster Head (`is_cluster_head = 1`).

### 2.3 New Cluster-Aware Features
* `cluster_id`: Assigned local cluster partition index $\{0, 1, \dots, 5\}$.
* `is_cluster_head`: Binary indicator ($1$ for elected leader, $0$ for ordinary cluster member).
* `cluster_size`: Total count of nodes in the cluster $|C|$.
* `distance_to_ch`: Euclidean distance in meters from the node to its elected cluster head ($0.0\text{ m}$ for the CH itself).
* `is_intra_cluster`: Edge attribute denoting whether a communication link is within the same cluster ($1$) or across clusters ($0$).

---

## 3. Experimental Results: Review 2 vs Review 3 Comparison

All models were evaluated across **5 independent simulation runs** using the identical strict chronological split:
* **Training Partition**: Timestamps $0\le t \le 340\text{ s}$ (70% past history).
* **Testing Partition**: Timestamps $350\le t \le 500\text{ s}$ (30% future horizon containing active Blackhole, Greyhole, and Wormhole attacks).

### Summary Results Table (Mean ± Standard Deviation over 5 Runs)

| Feature Set | Model | Accuracy | Malicious Precision | Malicious Recall | Malicious F1 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Review 2 (Unclustered)** | **Random Forest** | 96.79% ± 0.29% | 89.47% ± 2.37% | **79.30% ± 0.92%** | 84.07% ± 1.29% |
| | **SVM** | 96.05% ± 1.42% | 81.83% ± 12.09% | 83.75% ± 3.60% | 82.23% ± 4.89% |
| | **XGBoost** | 97.75% ± 0.32% | 99.80% ± 0.28% | 79.06% ± 2.83% | 88.21% ± 1.81% |
| **Review 3 (Clustered)** | **Random Forest** | **97.56% ± 0.07%** | **97.96% ± 0.70%** | 78.75% ± 0.59% | **87.31% ± 0.37%** |
| | **SVM** | 94.54% ± 2.32% | 73.26% ± 14.83% | **84.22% ± 3.90%** | 77.41% ± 6.79% |
| | **XGBoost** | **97.78% ± 0.32%** | 99.71% ± 0.27% | **79.38% ± 3.02%** | **88.36% ± 1.83%** |

### Key Research Findings
1. **Random Forest Precision and F1 Surge**:
   Adding cluster structure (`cluster_size`, `distance_to_ch`, `is_cluster_head`) reduced false positives significantly:
   * **Malicious Precision** increased from **89.47% to 97.96%** (+8.49 percentage points).
   * **Malicious F1** improved from **84.07% to 87.31%** (+3.24 percentage points).
   * **Accuracy** increased from **96.79% to 97.56%** with ultra-low variance (std drops from 0.29% to 0.07%).
2. **XGBoost Detection Peak**:
   XGBoost maintained near-perfect precision (**99.71%**) while slightly improving recall to **79.38%**, delivering the highest overall F1 score of **88.36%**.
3. **Hierarchy Matters in MANETs**:
   Attack nodes (such as greyholes selectively dropping packets or wormholes funneling traffic) exhibit abnormal behavioral deviations relative to their position from the cluster head. Incorporating `distance_to_ch` and `cluster_size` gives classifiers spatial-topological context that raw packet counters alone cannot provide.

---

## 4. Repository Structure & Deliverables

```
manet/
├── data/
│   ├── processed/
│   │   ├── nodes_dynamic.csv            # Original Review 2 dataset (preserved)
│   │   └── edges_dynamic.csv            # Original Review 2 edges (preserved)
│   └── processed_clustered/
│       ├── nodes_clustered.csv          # Review 3 cluster-aware dataset (38,250 rows, 23 cols)
│       ├── edges_clustered.csv          # Review 3 cluster-aware edges (612,971 rows, 9 cols)
│       └── nodes_with_clusters.csv      # Intermediate snapshot cluster mappings
├── python/
│   ├── validate_dataset.py              # Automated data audit and integrity check
│   ├── cluster_nodes.py                 # Snapshot spatial clustering
│   ├── select_cluster_heads.py          # Multi-criteria CH election and feature generation
│   ├── train_cluster_baselines.py       # Comparative baseline training and evaluation
│   └── visualize_clusters.py            # High-resolution topology visualization generator
├── results/
│   ├── cluster_baseline_results.csv     # Per-run detailed evaluation metrics
│   ├── cluster_baseline_summary.csv     # Mean +/- Std summary comparison table
│   ├── cluster_visualization_t0.png     # Topology snapshot at t=0s with clusters & CHs
│   └── cluster_visualization_t350.png   # Topology snapshot at t=350s with active attacks
└── README_RAGHAV_REVIEW3.md             # This document
```

---

## 5. How to Reproduce All Review 3 Results

Run each step sequentially from the root repository directory:

```bash
# Step 1: Validate dataset integrity
python python/validate_dataset.py

# Step 2: Generate snapshot clusters
python python/cluster_nodes.py --n-clusters 6

# Step 3: Elect cluster heads and build clustered dataset
python python/select_cluster_heads.py

# Step 4: Train and compare baselines (Review 2 vs Review 3)
python python/train_cluster_baselines.py

# Step 5: Generate cluster visualization figures
python python/visualize_clusters.py
```
