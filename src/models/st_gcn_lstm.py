"""Models. Every model takes ONE tensor  seq: [B, L, 5]  (4 risk indices + Total_Cost)
and returns node risks [B, 4] (or a scalar TRI [B] for the paper-style model).

* STGCNLSTM            - proposed: GCN at EVERY time step, then a shared LSTM PER NODE
* LSTMBaseline         - same LSTM + head, no graph (ablation: does the graph help at all?)
* PaperHybridOverall   - re-implementation of the paper-style fusion: GCN -> mean pool, || LSTM -> scalar
"""
import copy
from typing import Optional
import torch
import torch.nn as nn

from src.graph_builder import build_graphs
from src.models.graph_layers import GraphConv


def _register_graphs(module: nn.Module):
    for k, v in build_graphs().items():
        module.register_buffer(k, v)          # moves with .to(device), saved in state_dict

    def get():
        return {k: getattr(module, k) for k in ("A_hat", "A_down", "A_up")}
    return get


def _head(in_dim, hidden, dropout, residual):
    last = nn.Linear(hidden, 1)
    if residual:                               # start exactly at the persistence forecast
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout), last)


class STGCNLSTM(nn.Module):
    def __init__(self, cfg=None, mode: Optional[str] = None, residual: Optional[bool] = None):
        super().__init__()
        if cfg is None:
            from src.config import Config
            cfg = Config()
        elif isinstance(cfg, type):
            cfg = type("ConfigInstance", (cfg,), {})()
        else:
            cfg = copy.copy(cfg)

        if mode is not None:
            cfg.GRAPH_MODE = mode
        if residual is not None:
            cfg.RESIDUAL = residual
            cfg.RESIDUAL_CONNECTION = residual

        self.cfg = cfg
        self.residual = getattr(cfg, "RESIDUAL_CONNECTION", getattr(cfg, "RESIDUAL", True))
        self.graphs = _register_graphs(self)
        d = cfg.GCN_HIDDEN_DIM
        self.gcn1 = GraphConv(cfg.NODE_FEAT_DIM, d, cfg.GRAPH_MODE)
        self.gcn2 = GraphConv(d, d, cfg.GRAPH_MODE)       # 2 layers -> 2-hop reach (S can influence D)
        self.echelon_emb = nn.Parameter(torch.randn(1, cfg.NUM_NODES, d) * 0.05)
        self.lstm = nn.LSTM(d, cfg.LSTM_HIDDEN_DIM, cfg.LSTM_NUM_LAYERS, batch_first=True,
                            dropout=cfg.LSTM_DROPOUT if cfg.LSTM_NUM_LAYERS > 1 else 0.0)
        self.head = _head(cfg.LSTM_HIDDEN_DIM, cfg.HEAD_HIDDEN, cfg.FC_DROPOUT, self.residual)

    def forward(self, seq: torch.Tensor, g: Optional[dict] = None) -> torch.Tensor:
        B, L, _ = seq.shape
        N = self.cfg.NUM_NODES
        ri = seq[..., :N]                                          # [B, L, 4]
        cost = seq[..., N:N + 1].expand(-1, -1, N)                 # [B, L, 4] (global feature, shared)
        x = torch.stack([ri, cost], dim=-1).reshape(B * L, N, 2)   # per-step node features

        graphs = g if g is not None else self.graphs()
        h = self.gcn2(self.gcn1(x, graphs), graphs) + self.echelon_emb       # [B*L, 4, d]  node identity kept
        h = h.reshape(B, L, N, -1).permute(0, 2, 1, 3).reshape(B * N, L, -1)

        _, (h_n, _) = self.lstm(h)                                 # shared LSTM over each node's own series
        z = h_n[-1].reshape(B, N, -1)                              # [B, 4, 64]  (distinct per node)
        out = self.head(z).squeeze(-1)                             # [B, 4]
        return seq[:, -1, :N] + out if self.residual else out


class LSTMBaseline(nn.Module):
    def __init__(self, cfg=None, residual: Optional[bool] = None):
        super().__init__()
        if cfg is None:
            from src.config import Config
            cfg = Config()
        elif isinstance(cfg, type):
            cfg = type("ConfigInstance", (cfg,), {})()
        else:
            cfg = copy.copy(cfg)

        if residual is not None:
            cfg.RESIDUAL = residual
            cfg.RESIDUAL_CONNECTION = residual

        self.cfg = cfg
        self.residual = getattr(cfg, "RESIDUAL_CONNECTION", getattr(cfg, "RESIDUAL", True))
        self.lstm = nn.LSTM(cfg.NUM_INPUT_FEATURES, cfg.LSTM_HIDDEN_DIM, cfg.LSTM_NUM_LAYERS, batch_first=True,
                            dropout=cfg.LSTM_DROPOUT if cfg.LSTM_NUM_LAYERS > 1 else 0.0)
        self.head = nn.Sequential(nn.Linear(cfg.LSTM_HIDDEN_DIM, cfg.HEAD_HIDDEN), nn.ReLU(),
                                  nn.Dropout(cfg.FC_DROPOUT), nn.Linear(cfg.HEAD_HIDDEN, cfg.NUM_NODES))
        if self.residual:
            nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(seq)
        out = self.head(h_n[-1])
        return seq[:, -1, :self.cfg.NUM_NODES] + out if self.residual else out


class PaperHybridOverall(nn.Module):
    """Paper-style: 1 GCN layer -> global mean pool (node identity destroyed) || LSTM -> scalar TRI."""

    def __init__(self, cfg=None):
        super().__init__()
        if cfg is None:
            from src.config import Config
            cfg = Config()
        elif isinstance(cfg, type):
            cfg = type("ConfigInstance", (cfg,), {})()
        else:
            cfg = copy.copy(cfg)

        self.cfg = cfg
        self.graphs = _register_graphs(self)
        self.gcn = GraphConv(1, cfg.GCN_HIDDEN_DIM, "symmetric")
        self.lstm = nn.LSTM(cfg.NUM_INPUT_FEATURES, cfg.LSTM_HIDDEN_DIM, cfg.LSTM_NUM_LAYERS, batch_first=True,
                            dropout=cfg.LSTM_DROPOUT if cfg.LSTM_NUM_LAYERS > 1 else 0.0)
        self.fc = nn.Sequential(nn.Linear(cfg.GCN_HIDDEN_DIM + cfg.LSTM_HIDDEN_DIM, 64), nn.ReLU(),
                                nn.Dropout(cfg.FC_DROPOUT), nn.Linear(64, 1))

    def forward(self, seq: torch.Tensor, g: Optional[dict] = None) -> torch.Tensor:
        gx = seq[:, -1, :self.cfg.NUM_NODES].unsqueeze(-1)         # [B, 4, 1]
        graphs = g if g is not None else self.graphs()
        g_emb = self.gcn(gx, graphs).mean(dim=1)            # [B, 32]  global mean pooling
        _, (h_n, _) = self.lstm(seq)
        return self.fc(torch.cat([g_emb, h_n[-1]], dim=-1)).squeeze(-1)


MODELS = {"st_gcn_lstm": STGCNLSTM, "lstm": LSTMBaseline, "paper_overall": PaperHybridOverall}


def build_model(name: str, cfg=None) -> nn.Module:
    if cfg is None:
        from src.config import Config
        cfg = Config()
    
    if name.startswith("st_gcn_lstm_"):
        mode = name.split("_")[-1]
        cfg = copy.copy(cfg)
        cfg.GRAPH_MODE = "symmetric" if mode == "sym" else "directed"
        return STGCNLSTM(cfg)
        
    return MODELS[name](cfg)
