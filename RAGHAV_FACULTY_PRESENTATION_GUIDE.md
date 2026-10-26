# Raghav Bajaj — Review 2 vs Review 3 Faculty Presentation Guide

**Project Title:** Dynamic Federated Graph Neural Network for Intrusion Detection in Mobile Ad Hoc Networks  
**Student Name:** Raghav Bajaj (24BCE2068)  
**Assigned Responsibility:** Dynamic Dataset Construction, MANET Clustering, Multi-Criteria Cluster-Head Election, and Machine Learning Baselines  

---

## Part 1: How to Introduce Yourself & Your Role (Say this in 30 seconds)

> *"Good morning/afternoon, professors. My name is Raghav Bajaj. In this project, my core responsibility is the **data foundation and classical machine learning baselines**.*
> 
> *In **Review 2**, I built the dynamic graph dataset from NS-3 mobility simulations, simulated network attacks, and established our reference machine learning models.*
> 
> *In **Review 3**, I addressed the biggest limitation of our earlier dataset: our nodes were treated as flat, isolated devices. I introduced **MANET clustering and multi-criteria cluster-head election**, creating a structured, hierarchical network representation. This single improvement boosted our Random Forest malicious precision by over **8.5%**, dramatically reducing false alarms without touching or breaking anyone else's code."*

---

## Part 2: What I Did Till Review 2 (The Foundation)

### In Simple Words:
Before our team could train any Graph Neural Network (GNN) or Federated Learning model, we needed a realistic, moving network with simulated attacks. That was my job.

### Key Contributions in Review 2:
1. **Turned Raw Movement into Network Graphs:**
   * Our simulator (NS-3) gave raw node coordinates $(x, y)$.
   * I wrote `build_dynamic_dataset.py`, which connected any two mobile nodes with a radio link if they were within 250 meters of each other.
2. **Simulated Realistic Network Traffic & Attacks:**
   * Simulated packet forwarding across the shortest path between random nodes.
   * Injected three real-world MANET attacks over time:
     * **Blackhole (from 100 s):** Malicious node silently drops 95% of packets it receives.
     * **Greyhole (from 200 s):** Sneakier attacker that drops 50% of packets at random.
     * **Wormhole (from 300 s):** Attacker creates a deceptive fast link between distant nodes to eavesdrop on traffic.
3. **Calculated Behavioral Features for Every Node:**
   * At every 10-second snapshot, calculated battery energy, trust score, packet drop rate, and forwarding ratio.
   * Produced `nodes_dynamic.csv` (38,250 records across 5 simulation runs) and `edges_dynamic.csv` (612,971 links).
4. **Built Leakage-Free Classical Baselines:**
   * Used a strict **chronological split**: first 70% of time (0–340 s) for training, and remaining 30% of time (350–500 s) for testing. (Crucial: no future data leaked into training).
   * Trained Random Forest, SVM, and XGBoost on 4 behavioral features to establish benchmark accuracy and F1 scores.

---

## Part 3: What Was Missing in Review 2 & Why Review 3 Was Needed

### The Problem:
* In Review 2, our dataset was **flat**: each node only knew its own packet stats.
* But in real Mobile Ad Hoc Networks, nodes move constantly. To stay organized, nodes naturally form **local clusters** (neighborhood groups) and choose one trusted leader called a **Cluster Head (CH)**.
* Without this cluster information, our models had a lot of false alarms because they couldn't tell whether a node's behavior was normal for its local neighborhood.

---

## Part 4: What I Did in Review 3 (The Solution)

I implemented a 5-step engineering and research workflow:

### Step 1: Complete Data Integrity Audit (`python/validate_dataset.py`)
* Wrote an automated check that verified all 38,250 rows:
  * Zero missing values (0 NaNs).
  * Zero duplicate records.
  * Exactly 150 nodes at every single timestamp.
  * Confirmed both normal and malicious nodes exist in both train and test splits.

### Step 2: Snapshot-by-Snapshot Spatial Clustering (`python/cluster_nodes.py`)
* At every 10-second timestep, nodes move. 
* I grouped the 150 nodes into **6 geographical clusters** based on their physical $(x, y)$ coordinates.
* Every node now receives a `cluster_id` (0 to 5) showing which neighborhood it belongs to.

### Step 3: Multi-Criteria Cluster Head Election (`python/select_cluster_heads.py`)
* In each cluster, we need to elect the best node to be the leader (Cluster Head).
* Instead of guessing, I used a transparent, weighted formula:
  $$\text{CH Score} = 0.30 \times \text{Energy} + 0.30 \times \text{Trust} + 0.20 \times \text{Degree} + 0.20 \times \text{Centricity}$$
  * **Energy (30%):** A dead battery cannot lead; only high-energy nodes are chosen.
  * **Trust (30%):** Attackers or suspicious nodes are disqualified.
  * **Degree (20%):** Node must have many neighbors to communicate easily.
  * **Centricity / Inverse Distance (20%):** Node must be located in the middle of its cluster, not at the edge.
* Added 4 new features to the dataset:
  1. `cluster_id`: Which group the node belongs to.
  2. `is_cluster_head`: 1 if this node is the leader, 0 if it's a regular member.
  3. `cluster_size`: How many nodes are in its cluster.
  4. `distance_to_ch`: How many meters the node is from its cluster head.
* Saved this into `data/processed_clustered/nodes_clustered.csv`, preserving our original Review 2 dataset completely untouched.

### Step 4: Re-trained ML Baselines & Evaluated Impact (`python/train_cluster_baselines.py`)
* Re-trained Random Forest, SVM, and XGBoost using the **exact same 70/30 chronological split** across all 5 runs.
* Evaluated whether adding cluster hierarchy improved detection.

### Step 5: Topology Visualizations (`python/visualize_clusters.py`)
* Generated high-resolution plots at $t = 0\text{ s}$ and $t = 350\text{ s}$ showing:
  * Shaded cluster boundaries (convex hulls).
  * Elected Cluster Heads highlighted with **gold stars**.
  * Active Blackhole, Greyhole, and Wormhole attackers marked clearly in red/orange.

---

## Part 5: The Exact Results (Show this Table to Faculty!)

All numbers are **Mean ± Standard Deviation across 5 independent simulation seeds**:

| Model | Feature Set | Accuracy | Malicious Precision (Fewer False Alarms) | Malicious Recall (Detection Rate) | Malicious F1 Score |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Random Forest** | Review 2 (Flat / Unclustered) | 96.79% ± 0.29% | 89.47% ± 2.37% | 79.30% ± 0.92% | 84.07% ± 1.29% |
| | **Review 3 (Cluster-Aware)** | **97.56% ± 0.07%** | **97.96% ± 0.70%** *(+8.5% Surge!)* | 78.75% ± 0.59% | **87.31% ± 0.37%** *(+3.2% Better)* |
| **XGBoost** | Review 2 (Flat / Unclustered) | 97.75% ± 0.32% | **99.80% ± 0.28%** | 79.06% ± 2.83% | 88.21% ± 1.81% |
| | **Review 3 (Cluster-Aware)** | **97.78% ± 0.32%** | 99.71% ± 0.27% | **79.38% ± 3.02%** | **88.36% ± 1.83%** *(Best Overall)* |
| **SVM** | Review 2 (Flat / Unclustered) | 96.05% ± 1.42% | 81.83% ± 12.09% | 83.75% ± 3.60% | 82.23% ± 4.89% |
| | **Review 3 (Cluster-Aware)** | 94.54% ± 2.32% | 73.26% ± 14.83% | **84.22% ± 3.90%** | 77.41% ± 6.79% |

### Key Takeaway for Faculty:
* In Random Forest, **Malicious Precision jumped from 89.47% to 97.96%**. That means false alarms dropped dramatically!
* **Why did this happen?** Because malicious nodes (like wormholes or greyholes) behave abnormally compared to where they sit relative to the cluster head. Knowing `distance_to_ch` and `cluster_size` gives the tree models spatial context that raw packet counters alone could not provide.

---

## Part 6: Expected Questions by Faculty & How Raghav Should Answer

### Q1: "Why did you use K-Means for clustering instead of letting GNN do it?"
> **Answer:** *"In MANET protocols (like LEACH or CBRP), clustering is a networking function based on physical radio proximity, not a learning task. By using K-Means on physical $(x, y)$ coordinates, our clusters represent actual physical radio neighborhoods (around 200m radius), which matches our 250m radio transmission range. This keeps the clustering explainable and fully independent of Vivek's GNN and Shiveish's FL-GCN."*

### Q2: "Why does the Cluster Head formula give 30% to energy and 30% to trust?"
> **Answer:** *"In a mobile ad hoc network, if a cluster head runs out of battery, the whole cluster loses its coordinator. If a compromised or malicious node becomes cluster head, it can tamper with routing. That is why residual energy (30%) and trust (30%) are given the highest weights, while degree connectivity (20%) and central position (20%) ensure efficient communication coverage."*

### Q3: "Did adding cluster features cause any data leakage between train and test sets?"
> **Answer:** *"No, professors. Clustering and cluster-head selection are performed snapshot-by-snapshot at each individual timestamp. No information from future timestamps (350–500 s) is ever used during earlier timestamps (0–340 s). Furthermore, model training strictly uses only snapshots 0 to 340 s, and testing strictly uses 350 to 500 s."*

### Q4: "Where are your outputs saved?"
> **Answer:** *"All my code and outputs are pushed to our GitHub repository on my branch `raghav-review3`:*
> * *Code: `python/cluster_nodes.py`, `python/select_cluster_heads.py`, `python/train_cluster_baselines.py`, `python/visualize_clusters.py`.*
> * *Data: `data/processed_clustered/nodes_clustered.csv` and `edges_clustered.csv`.*
> * *Results: `results/cluster_baseline_results.csv`, `results/cluster_baseline_summary.csv`.*
> * *Figures: `results/cluster_visualization_t0.png` and `results/cluster_visualization_t350.png`.*
> * *Documentation: `README_RAGHAV_REVIEW3.md`."*
