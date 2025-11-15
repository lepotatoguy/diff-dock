# import torch
# import torch.nn as nn

# class DiffusionWrapper(nn.Module):
#     def __init__(self, model, timesteps=1000):
#         super().__init__()
#         self.model = model
#         self.timesteps = timesteps

#     def q_sample(self, x0, t, noise):
#         return x0 + noise * (t / self.timesteps)

#     def p_mean(self, x_t, context):
#         return self.model(x_t, context)

#     def loss(self, lig_pos, poc_pos):
#         t = torch.randint(1, self.timesteps, (1,), device=lig_pos.device).float()
#         noise = torch.randn_like(lig_pos)
#         x_t = self.q_sample(lig_pos, t, noise)
#         pred_noise = self.p_mean(x_t, poc_pos)
#         return ((noise - pred_noise)**2).mean()

import torch
import torch.nn as nn

class DiffusionWrapper(nn.Module):
    def __init__(self, model, encoder, timesteps=1000):
        super().__init__()
        self.model = model
        self.encoder = encoder
        self.timesteps = timesteps

    def q_sample(self, x0, t, noise):
        return x0 + noise * (t / self.timesteps)

    def p_mean(self, x_t, x_feat, ctx_feat):
        return self.model(x_feat, ctx_feat)

    def loss(self, lig_pos, lig_type, poc_pos, poc_type):
        t = torch.randint(1, self.timesteps, (1,), device=lig_pos.device).float()
        noise = torch.randn_like(lig_pos)
        x_t = self.q_sample(lig_pos, t, noise)

        # encode to SE3 features
        x_feat = self.encoder(x_t, lig_type)
        ctx_feat = self.encoder(poc_pos, poc_type)

        pred_noise = self.p_mean(x_t, x_feat, ctx_feat)
        return ((noise - pred_noise)**2).mean()
