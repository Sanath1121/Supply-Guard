import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConv(nn.Module):
    """Batched graph convolution on x: [B, N, in]; no PyG needed.

    mode="symmetric": H' = ReLU( A_hat H W + b )                        (Kipf & Welling 2017)
    mode="directed" : H' = ReLU( H W_self + A_down H W_down + A_up H W_up + b )
                      A_down aggregates upstream parents, A_up aggregates downstream children
                      (a relational / directed variant, row-mean normalised).
    """

    def __init__(self, in_dim: int, out_dim: int, mode: str = "directed"):
        super().__init__()
        assert mode in ("symmetric", "directed")
        self.mode = mode
        if mode == "symmetric":
            self.lin = nn.Linear(in_dim, out_dim)
        else:
            self.self_lin = nn.Linear(in_dim, out_dim)               # carries the bias
            self.down_lin = nn.Linear(in_dim, out_dim, bias=False)
            self.up_lin = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor, g: dict) -> torch.Tensor:
        if self.mode == "symmetric":
            return F.relu(self.lin(torch.matmul(g["A_hat"], x)))      # [N,N] @ [B,N,in]
        out = (self.self_lin(x)
               + self.down_lin(torch.matmul(g["A_down"], x))
               + self.up_lin(torch.matmul(g["A_up"], x)))
        return F.relu(out)
