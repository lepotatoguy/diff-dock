import torch

def generate_pose(diff, poc_pos, n_steps=1000):
    x = torch.randn((poc_pos.shape[0]//2, 3), device=poc_pos.device)
    for t in reversed(range(1, n_steps)):
        pred_noise = diff.p_mean(x, poc_pos)
        x = x - pred_noise * (1.0 / n_steps)
    return x
