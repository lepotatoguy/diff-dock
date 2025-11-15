import torch
import torch.nn as nn

class DiffusionWrapper(nn.Module):
    def __init__(self, model, timesteps=1000):
        super().__init__()
        self.model = model
        self.timesteps = timesteps

    def q_sample(self, x0, t, noise):
        return x0 + noise * (t / self.timesteps)

    def loss(self, lig_pos, lig_type, poc_pos, poc_type):

        t = torch.randint(1, self.timesteps, (1,), device=lig_pos.device).float()
        noise = torch.randn_like(lig_pos)

        x_t = self.q_sample(lig_pos, t, noise)

        pred_noise = self.model(x_t, lig_type, poc_pos, poc_type, t)

        return ((noise - pred_noise)**2).mean()
