import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from torch_geometric.nn import SAGEConv
from train_gnn_models import EvolveGCNH
import train_gnn_models_review3 as t3


class EvolveGCNHSage(EvolveGCNH):
    """EvolveGCN-H-style model whose output layer keeps a self-weight."""
    def __init__(self, in_channels=4, hidden_channels=16, out_channels=2):
        super().__init__(in_channels, hidden_channels, out_channels)
        self.output_conv = SAGEConv(hidden_channels, out_channels)


t3.MODEL_CLASSES["EvolveGCN-H"] = EvolveGCNHSage

import train_evolvegcn_review3 as tune
tune.main()
