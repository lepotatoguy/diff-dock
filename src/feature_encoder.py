import torch
import torch.nn as nn

class FeatureEncoder(nn.Module):
    def __init__(self, out_channels=16):
        super().__init__()
        self.atom_emb = nn.Embedding(100, 8)    # atomic number → 8d
        self.lin = nn.Linear(8 + 3, out_channels)

    def forward(self, coords, atom_types):
        # coords: (N, 3)
        # atom_types: (N,)
        e = self.atom_emb(atom_types)          # (N, 8)
        x = torch.cat([coords, e], dim=-1)     # (N, 3+8=11)
        return self.lin(x)                     # (N, out_channels)
