import torch
import torch.nn as nn
from einops import rearrange, repeat


def Flatten(inp: torch.Tensor) -> torch.Tensor:
    # spatiotemporal flattening of (batch, length, dim) into (batch, length x dim)
    out = rearrange(inp, "batch len dy -> batch (dy len) 1")
    return out


class Normalization(nn.Module):
    def __init__(self, method, d_model=None):
        super().__init__()
        assert method in ["layer", "scale", "batch", "power", "none"]
        if method == "layer":
            assert d_model
            self.norm = nn.LayerNorm(d_model)
        # elif method == "scale":
        #     self.norm = ScaleNorm(d_model)
        # elif method == "power":
        #     self.norm = MaskPowerNorm(d_model, warmup_iters=1000)
        elif method == "none":
            self.norm = lambda x: x
        else:
            assert d_model
            self.norm = nn.BatchNorm1d(d_model)
        self.method = method

    def forward(self, x):
        if self.method == "batch":
            return self.norm(x.transpose(-1, 1)).transpose(-1, 1)
        return self.norm(x)


def Localize(inp: torch.Tensor, variables: int) -> torch.Tensor:
    # split spatiotemporal into individual vars and fold into batch dim
    return rearrange(
        inp,
        "batch (variables len) dim -> (variables batch) len dim",
        variables=variables,
    )


def ReverseLocalize(inp: torch.Tensor, variables: int) -> torch.Tensor:
    return rearrange(
        inp,
        "(variables batch) len dim -> batch (variables len) dim",
        variables=variables,
    )


def FoldForPred(inp: torch.Tensor, dy: int) -> torch.Tensor:
    out = rearrange(inp, "batch (dy len) dim -> dim batch len dy", dy=dy)
    out = out.squeeze(0)
    return out
