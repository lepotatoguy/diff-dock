import torch
import torch.nn as nn
import torch.nn.functional as F

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.lin = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, t):
        # t: scalar tensor (1,)
        half = self.dim // 2
        freqs = torch.exp(
            torch.arange(half, device=t.device) * -(torch.log(torch.tensor(10000.0)) / half)
        )
        arg = t * freqs  # (half,)
        sinusoid = torch.cat([torch.sin(arg), torch.cos(arg)], dim=-1)  # (dim,)
        return self.lin(sinusoid)


class SimpleEGNN(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()

        self.emb_lig = nn.Embedding(100, hidden_dim)
        self.emb_poc = nn.Embedding(100, hidden_dim)

        self.time_mlp = TimeEmbedding(hidden_dim)

        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)

        self.update = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )

    def forward(self, lig_pos, lig_type, poc_pos, poc_type, t):

        lig_f = self.emb_lig(lig_type)
        poc_f = self.emb_poc(poc_type)

        # t_embed = self.time_mlp(t).unsqueeze(0)
        t_embed = self.time_mlp(t).unsqueeze(0)     # (1, H)
        lig_f = lig_f + t_embed
        poc_f = poc_f + t_embed

        lig_attn, _ = self.cross_attn(lig_f.unsqueeze(0),
                                      poc_f.unsqueeze(0),
                                      poc_f.unsqueeze(0))
        lig_attn = lig_attn.squeeze(0)

        Nl = lig_pos.size(0)
        Np = poc_pos.size(0)

        lig_exp = lig_pos.unsqueeze(1).expand(Nl, Np, 3)
        poc_exp = poc_pos.unsqueeze(0).expand(Nl, Np, 3)

        dists = ((lig_exp - poc_exp) ** 2).sum(-1, keepdim=True)

        poc_msg = poc_f.unsqueeze(0).expand(Nl, -1, -1)

        edge_feat = torch.cat([lig_attn.unsqueeze(1).expand(-1, Np, -1),
                               poc_msg,
                               dists], dim=-1)

        edge_feat = edge_feat.mean(dim=1)

        return self.update(edge_feat)
