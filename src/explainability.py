"""Integrated Gradients for node-risk forecasts.

Why IG instead of Input x Gradient
----------------------------------
* a meaningful reference point: the baseline is the TRAINING-MEAN window, not "scaled zero"
* completeness: sum(attributions) = f(x) - f(baseline); we return the residual so you can show it
* signed: positive = pushes predicted risk up, negative = pulls it down
* single input tensor: node-level information enters only via seq[..., :4]; there is no separate
  graph input to double-count. "Which echelon drove this?" = attribution on that echelon's RI column.

Attribution answers "how sensitive is this forecast to each input?", NOT "what caused the risk".
"""
import numpy as np
import torch

FEATURE_NAMES = ["Supplier RI", "Manufacturer RI", "Distributor RI", "Retailer RI", "Total Cost"]
NODE_NAMES = ["Supplier", "Manufacturer", "Distributor", "Retailer"]


class RiskExplainer:
    def __init__(self, model: torch.nn.Module, baseline: torch.Tensor, steps: int = 64):
        """baseline: [F] (training-mean feature vector) or [L, F]."""
        self.model = model.eval()
        self.baseline = baseline.float()
        self.steps = steps

    def _baseline_like(self, seq):
        b = self.baseline
        return b.expand_as(seq).clone() if b.dim() == 1 else b.clone()

    def explain(self, seq: torch.Tensor, target_node: int, residual_delta: bool = False) -> dict:
        """seq: [L, F] one window. Returns signed attributions [L, F] plus summaries."""
        with torch.enable_grad():
            x = seq.detach().float()
            base = self._baseline_like(x)
            alphas = torch.linspace(0.0, 1.0, self.steps + 1)[1:].view(-1, 1, 1)   # right Riemann sum
            path = (base + alphas * (x - base)).requires_grad_(True)               # [S, L, F]
            out = self.model(path)[:, target_node]
            if residual_delta:
                out = out - path[:, -1, target_node]
            out = out.sum()
            grads, = torch.autograd.grad(out, path)
            attr = (x - base) * grads.mean(dim=0)                                  # [L, F]

        with torch.no_grad():
            f_x = float(self.model(x.unsqueeze(0))[0, target_node])
            f_b = float(self.model(base.unsqueeze(0))[0, target_node])
            if residual_delta:
                f_x -= float(x[-1, target_node])
                f_b -= float(base[-1, target_node])

        a = attr.numpy()
        abs_a = np.abs(a)
        total = abs_a.sum() + 1e-12
        feat_abs, time_abs = abs_a.sum(axis=0), abs_a.sum(axis=1)
        return {
            "target_node": target_node,
            "predicted_risk": f_x,
            "baseline_risk": f_b,
            "completeness_gap": float(a.sum() - (f_x - f_b)),     # should be ~0; report it
            "attribution": a,                                     # [L, F] signed
            "feature_importance": (feat_abs / total).tolist(),    # share of |attribution|
            "feature_signed": a.sum(axis=0).tolist(),             # net push up (+) / down (-)
            "time_importance": (time_abs / total).tolist(),        # index 0 = oldest, L-1 = t
            "total_abs_attribution": float(total),                # magnitude, not just shares
        }

    def explain_all(self, seq: torch.Tensor, residual_delta: bool = False) -> list:
        return [self.explain(seq, i, residual_delta=residual_delta) for i in range(len(NODE_NAMES))]


def upstream_share(result: dict) -> float:
    """Share of |attribution| coming from echelons upstream of the target (S->M->D->R order)."""
    t, fi = result["target_node"], result["feature_importance"]
    return float(sum(fi[:t]))


def narrate(result: dict, seq_len: int = 10) -> str:
    """Plain-English summary built ONLY from computed numbers (nothing hard-coded)."""
    fi, ft = result["feature_importance"], result["time_importance"]
    k = int(np.argmax(fi))
    sign = "raises" if result["feature_signed"][k] >= 0 else "lowers"
    j = int(np.argmax(ft))
    steps_ago = seq_len - 1 - j
    when = "the most recent step (t)" if steps_ago == 0 else f"{steps_ago} step(s) before t"
    node = NODE_NAMES[result["target_node"]]
    return (f"Forecast {node} risk = {result['predicted_risk']:.2f} "
            f"(window-average baseline {result['baseline_risk']:.2f}). "
            f"Most influential input: {FEATURE_NAMES[k]} ({fi[k]:.0%} of attribution), which {sign} the forecast. "
            f"Most influential time step: {when} ({ft[j]:.0%}). "
            f"Upstream echelons contribute {upstream_share(result):.0%}.")
