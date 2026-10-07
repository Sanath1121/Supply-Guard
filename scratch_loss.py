import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from src.config import Config
from src.dataset import build_datasets
from src.models.st_gcn_lstm import build_model
from training.train import ckpt_path

cfg = Config()
_, va_ds, _, _, _ = build_datasets(cfg, save_scaler=False)
va_loader = DataLoader(va_ds, batch_size=512, shuffle=False)
model_file = ckpt_path(cfg, "paper_overall", 43)
model = build_model("paper_overall", cfg)
model.load_state_dict(torch.load(model_file, map_location="cpu"))
model.eval()
total_loss, total_count = 0.0, 0
with torch.no_grad():
    for batch in va_loader:
        seq = batch["sequence"]
        pred = model(seq)
        tgt = batch["tri_target"]
        loss = F.mse_loss(pred, tgt)
        n = len(seq)
        total_loss += loss.item() * n
        total_count += n
recomputed_loss = total_loss / max(total_count, 1)
print(f"Exact loss: {recomputed_loss}")
