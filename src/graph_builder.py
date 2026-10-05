"""4-echelon supply-chain graph: S -> M -> D -> R.

Two formulations are provided:

* symmetric  : Kipf & Welling renormalised adjacency  A_hat = D~^-1/2 (A+I) D~^-1/2
               (NOT the Laplacian; the normalised Laplacian is L = I - D^-1/2 A D^-1/2)
* directed   : separate row-normalised matrices for "messages from upstream parents"
               (A_down) and "messages from downstream children" (A_up), so the model can
               tell supply-flow propagation from demand/bullwhip propagation.
"""
import torch

EDGES = [(0, 1), (1, 2), (2, 3)]  # (source, target): goods flow S->M->D->R


def _row_normalise(a: torch.Tensor) -> torch.Tensor:
    deg = a.sum(dim=1, keepdim=True).clamp(min=1.0)   # isolated rows stay all-zero
    return a / deg


def build_graphs(n: int = 4) -> dict:
    a_dir = torch.zeros(n, n)
    for s, t in EDGES:
        a_dir[s, t] = 1.0                      # a_dir[i, j] = 1  <=>  edge i -> j

    # ---- symmetric (Kipf & Welling) ----
    a_sym = ((a_dir + a_dir.T) > 0).float()
    assert torch.equal(a_sym, a_sym.T)
    a_tilde = a_sym + torch.eye(n)
    d_inv_sqrt = a_tilde.sum(dim=1).pow(-0.5)
    a_hat = d_inv_sqrt[:, None] * a_tilde * d_inv_sqrt[None, :]

    # ---- directed ----
    # (A_down @ H)[i] = mean of H over parents j with edge j -> i   (upstream messages)
    # (A_up   @ H)[i] = mean of H over children j with edge i -> j  (downstream messages)
    a_down = _row_normalise(a_dir.T.contiguous())
    a_up = _row_normalise(a_dir)

    return {"A_hat": a_hat, "A_down": a_down, "A_up": a_up, "A_dir": a_dir}


def normalised_laplacian(n: int = 4) -> torch.Tensor:
    """L = I - D^-1/2 A D^-1/2 (symmetric, no self-loops) - for the viva / sanity checks."""
    a = build_graphs(n)["A_dir"]
    a = ((a + a.T) > 0).float()
    d = a.sum(dim=1).pow(-0.5)
    return torch.eye(n) - d[:, None] * a * d[None, :]
