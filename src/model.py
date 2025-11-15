import torch
import torch.nn as nn

class SimpleGeomNet(nn.Module):
    def __init__(self, hidden_dim=32):
        super().__init__()
        self.emb_lig = nn.Embedding(100, hidden_dim)
        self.emb_poc = nn.Embedding(100, hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim*2 + 3, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

    def forward(self, lig_pos, lig_type, poc_pos, poc_type):
        lig_f = self.emb_lig(lig_type)     # (N_lig,H)

        poc_f = self.emb_poc(poc_type)     # (N_poc,H)
        poc_f = poc_f.mean(dim=0)          # (H,)
        poc_f = poc_f.unsqueeze(0)         # (1,H)
        poc_f = poc_f.expand(lig_f.size(0), -1)  # (N_lig,H)

        x = torch.cat([lig_f, poc_f, lig_pos], dim=-1)

        return self.mlp(x)
