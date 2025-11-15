# import torch
# import torch.nn as nn
# from e3nn.o3 import FullyConnectedTensorProduct, Irreps

# class SE3Block(nn.Module):
#     def __init__(self, in_channels=16, out_channels=16):
#         super().__init__()
#         self.tp = FullyConnectedTensorProduct(
#             Irreps(f"{in_channels}x0e"),
#             Irreps(f"{in_channels}x0e"),
#             Irreps(f"{out_channels}x0e")
#         )
#         self.lin = nn.Linear(in_channels, out_channels)

#     def forward(self, x, context):
#         # x: (N_lig, C), context: (N_pocket, C)
#         # simplified interaction: mean over pocket features
#         ctx = context.mean(0, keepdim=True)
#         t = self.tp(x.unsqueeze(0), ctx.unsqueeze(0))[0]
#         return self.lin(t)


import torch
import torch.nn as nn
from e3nn.o3 import FullyConnectedTensorProduct, Irreps

class SE3Block(nn.Module):
    def __init__(self, in_channels=16, out_channels=3):
        super().__init__()

        self.tp = FullyConnectedTensorProduct(
            Irreps(f"{in_channels}x0e"),
            Irreps(f"{in_channels}x0e"),
            Irreps(f"{in_channels}x0e")       # TP → 16 dims
        )

        self.lin = nn.Linear(in_channels, out_channels)  # → 3 coordinates

    def forward(self, x_feat, ctx_feat):
        if x_feat.dim() == 3:
            x_feat = x_feat.squeeze(0)
        if ctx_feat.dim() == 3:
            ctx_feat = ctx_feat.squeeze(0)

        ctx = ctx_feat.mean(0, keepdim=True)
        ctx_rep = ctx.expand(x_feat.shape[0], -1)

        x_tp = x_feat.unsqueeze(0)
        ctx_tp = ctx_rep.unsqueeze(0)

        t = self.tp(x_tp, ctx_tp)[0]     # (N, 16)

        return self.lin(t)               # (N, 3)
