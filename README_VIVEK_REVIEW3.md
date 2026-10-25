# Review 3 - Vivek Kashyap: Temporal GNN, Model Comparison, Ablation and Scalability

Project: *Dynamic Federated Graph Neural Network for Intrusion Detection in Mobile Ad Hoc Networks*

Every number below comes from a file in `results/`. Items marked **[TO CONFIRM]** have to be checked against the code before the review.

## 1. What I was asked to do

Audit the Review 2 GNN code, fix or tune EvolveGCN-H, re-run Static-GCN, GraphSAGE, GAT and EvolveGCN-H on the same temporal split with 5 seeds, run ablations, and re-run scalability at 100-500 nodes.

## 2. Problems found in Review 2

1. **EvolveGCN-H was weak.** Static-GCN, GAT and EvolveGCN-H had very low malicious recall.
2. **The Review 2 scalability result was not a real measurement.** It showed 100% accuracy at every node count. The old `python/scalability.py` takes only the first snapshot (t = 0), trains and evaluates on that same snapshot (no train/test split), and at t = 0 the malicious rate is 0.000 (see `results/scalability_review3.log`, line `malicious rate: t0=0.000`). Every label is benign, so predicting "benign" for everything gives 100% accuracy. The old plot is kept as `results/scalability_accuracy_review2.png` and the old CSV is unchanged for comparison.

## 3. Method

- Same temporal train/test split as before. Malicious rate: train about 4.7-5.1%, test about 8.8-12% depending on network size.
- Model settings were chosen on **validation malicious F1** (`results/evolvegcn_tuning.csv`, `results/tuning_others/`), not on test results.
- Tuning grid: class weights (full / sqrt / none), hidden size (16 / 32), learning rate (0.01 / 0.005 / 0.001). Dropout was **not** tuned.
- Final comparison: 5 independent runs per model, mean +/- std (`results/gnn_review3_results.csv`, `results/gnn_review3_summary.csv`).
- Per-node predictions are saved in `results/gnn_review3_predictions.csv`.
- Two EvolveGCN-H variants are reported:
  - **GCN output**: the original design.
  - **SAGE output**: a variant I added. **[TO CONFIRM: describe the exact change from `python/evolvegcn_sage_review3.py`]**
- Audit checks (chronological snapshot order, no future timestamps in train/validation, scaler fitted on training data only): **[TO CONFIRM from `python/audit_review3.py`]**

Selected settings (from `results/final_review3.log`):

| Model | Weights | Hidden | LR |
|---|---|---|---|
| Static-GCN | sqrt | 16 | 0.001 |
| GraphSAGE | none | 16 | 0.001 |
| GAT | sqrt | 16 | 0.01 |
| EvolveGCN-H (GCN output) | full | 32 | 0.01 |
| EvolveGCN-H (SAGE output) | sqrt | 32 | 0.01 |

## 4. Results: 5-run test comparison (mean +/- std)

| Model | Accuracy | Mal. Precision | Mal. Recall | Mal. F1 | Params |
|---|---|---|---|---|---|
| Static-GCN | 0.877 +/- 0.013 | 0.348 +/- 0.104 | 0.130 +/- 0.095 | 0.167 +/- 0.093 | 114 |
| GraphSAGE | 0.978 +/- 0.003 | 0.997 +/- 0.003 | 0.794 +/- 0.026 | **0.884 +/- 0.016** | 210 |
| GAT | 0.866 +/- 0.112 | 0.562 +/- 0.262 | 0.723 +/- 0.074 | 0.592 +/- 0.165 | 150 |
| EvolveGCN-H (GCN output) | 0.216 +/- 0.102 | 0.117 +/- 0.009 | 0.960 +/- 0.053 | 0.208 +/- 0.013 | 103,522 |
| EvolveGCN-H (SAGE output) | 0.954 +/- 0.017 | 0.787 +/- 0.146 | 0.833 +/- 0.051 | 0.800 +/- 0.054 | 103,586 |

Findings:

- **GraphSAGE is the best model** (F1 0.884, precision 0.997, recall 0.794).
- **The original EvolveGCN-H (GCN output) was not fixed.** It flags almost every node as malicious (recall 0.96, precision 0.117, accuracy 0.216). Its best validation F1 over the whole tuning grid was only 0.203.
- **EvolveGCN-H with SAGE output works** (F1 0.800), but it does not beat GraphSAGE. It has slightly higher recall (0.833 vs 0.794) and clearly lower precision (0.787 vs 0.997). So these results do not show a benefit from temporal modelling over the static GraphSAGE baseline.
- Static-GCN and GAT remain weak or unstable.

## 5. Ablation (validation malicious F1, mean +/- std; `results/gnn_ablation.csv`)

Class weighting:

| Model | none | sqrt | full |
|---|---|---|---|
| Static-GCN | 0.000 | 0.206 | 0.204 |
| GraphSAGE | **0.815** | 0.765 | 0.530 |
| GAT | 0.269 | **0.491** | 0.266 |
| EvolveGCN-H (GCN output) | 0.018 | 0.140 | **0.195** |
| EvolveGCN-H (SAGE output) | 0.759 | 0.758 | 0.548 |

Hidden size:

| Model | 16 | 32 |
|---|---|---|
| Static-GCN | 0.136 | 0.137 |
| GraphSAGE | 0.706 | 0.701 |
| GAT | 0.339 | 0.344 |
| EvolveGCN-H (GCN output) | 0.112 | 0.124 |
| EvolveGCN-H (SAGE output) | 0.689 | 0.688 |

- The best class-weighting choice depends on the model. Heavy weighting helps Static-GCN and EvolveGCN-H (GCN output) but hurts GraphSAGE and EvolveGCN-H (SAGE output).
- Hidden size 16 vs 32 makes almost no difference.

## 6. Scalability (`results/scalability_review3.csv`, 3 seeds)

Test metrics at 100-500 nodes (edges per snapshot grow from about 934 to about 26,895):

| Nodes | Model | Accuracy | Mal. Recall | Mal. F1 | Train (s) | Inference (s) |
|---|---|---|---|---|---|---|
| 100 | GraphSAGE | 0.994 | 0.946 | 0.972 | 1.24 | 0.006 |
| 100 | EvolveGCN-H (SAGE) | 0.993 | 0.946 | 0.968 | 1.33 | 0.013 |
| 100 | Static-GCN | 0.838 | 0.264 | 0.280 | 1.48 | 0.007 |
| 200 | GraphSAGE | 0.967 | 0.667 | 0.800 | 1.86 | 0.009 |
| 200 | EvolveGCN-H (SAGE) | 0.963 | 0.728 | 0.800 | 1.63 | 0.017 |
| 200 | Static-GCN | 0.897 | 0.031 | 0.057 | 2.09 | 0.011 |
| 300 | GraphSAGE | 0.959 | 0.562 | 0.719 | 2.74 | 0.015 |
| 300 | EvolveGCN-H (SAGE) | 0.954 | 0.588 | 0.705 | 2.32 | 0.027 |
| 300 | Static-GCN | 0.907 | 0.000 | 0.000 | 3.38 | 0.019 |
| 400 | GraphSAGE | 0.952 | 0.465 | 0.635 | 3.63 | 0.021 |
| 400 | EvolveGCN-H (SAGE) | 0.939 | 0.483 | 0.589 | 2.97 | 0.036 |
| 400 | Static-GCN | 0.902 | 0.023 | 0.034 | 4.99 | 0.028 |
| 500 | GraphSAGE | 0.946 | 0.386 | 0.556 | 4.81 | 0.033 |
| 500 | EvolveGCN-H (SAGE) | 0.930 | 0.443 | 0.530 | 3.80 | 0.050 |
| 500 | Static-GCN | 0.904 | 0.024 | 0.036 | 6.92 | 0.039 |

Findings:

- Accuracy stays high (0.93-0.99 for the two stronger models), but **malicious recall and F1 drop clearly as the network grows**. Accuracy alone hides this because only about 9-12% of test nodes are malicious.
- GraphSAGE is equal to or better than EvolveGCN-H (SAGE) in F1 at every size (500 nodes: 0.556 vs 0.530).
- Static-GCN essentially stops detecting malicious nodes from 200 nodes onward (F1 0.000 at 300 nodes).
- Training time grows with size; EvolveGCN-H (SAGE) trains slightly faster than GraphSAGE at 200-500 nodes, but inference is slower.

## 7. Limitations

- Scalability uses **3 seeds**, not 5, and covers only GraphSAGE, EvolveGCN-H (SAGE output) and Static-GCN (not GAT or EvolveGCN-H GCN output).
- Dropout was not part of the tuning grid.
- Memory usage was not measured.
- With 3 seeds, small F1 differences between GraphSAGE and EvolveGCN-H (SAGE) at a single size should not be over-interpreted.

## 8. Files

| Purpose | File |
|---|---|
| Audit | `python/audit_review3.py` |
| Main training and comparison | `python/train_gnn_models_review3.py`, `python/final_gnn_review3.py` |
| EvolveGCN-H | `python/train_evolvegcn_review3.py`, `python/evolvegcn_sage_review3.py` |
| Ablation | `python/gnn_ablation.py` |
| Scalability | `python/scalability_review3.py` |
| Plots | `python/plot_review3.py` |
| Result files | `results/gnn_review3_*.csv`, `results/evolvegcn_tuning*.csv`, `results/gnn_ablation.csv`, `results/scalability_review3*.csv`, `results/tuning_others/` |
| Figures | `results/gnn_model_comparison.png`, `results/temporal_vs_static.png`, `results/scalability_accuracy.png`, `results/scalability_f1.png`, `results/scalability_inference_time.png` |

## 9. Summary for the review

"In Review 2, EvolveGCN-H showed weak detection and the scalability result of 100% accuracy came from evaluating on a single t = 0 snapshot where every node is benign. In Review 3 I re-ran all models on the temporal split, chose settings by validation malicious F1, and compared them over 5 seeds. GraphSAGE is the strongest model (F1 0.884). The original EvolveGCN-H did not recover, while a SAGE-output variant reached F1 0.800 but did not beat GraphSAGE. Scalability shows that malicious recall and F1 fall as the network grows even though accuracy stays high."
