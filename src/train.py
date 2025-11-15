import torch
from torch.utils.data import DataLoader
from dataset import DockingDataset
from model import SimpleEGNN
from diffusion import DiffusionWrapper

dataset = DockingDataset("data")
loader = DataLoader(dataset, batch_size=1, shuffle=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = SimpleEGNN(hidden_dim=64).to(device)
diff = DiffusionWrapper(model).to(device)

opt = torch.optim.Adam(diff.parameters(), lr=2e-4)

for epoch in range(50):
    epoch_loss = 0.0
    count = 0

    for lig_pos, lig_type, poc_pos, poc_type in loader:
        # remove batch dim
        lig_pos = lig_pos.squeeze(0).to(device)
        lig_type = lig_type.squeeze(0).to(device)
        poc_pos = poc_pos.squeeze(0).to(device)
        poc_type = poc_type.squeeze(0).to(device)

        loss = diff.loss(lig_pos, lig_type, poc_pos, poc_type)

        opt.zero_grad()
        loss.backward()
        opt.step()

        # epoch_loss += float(loss)
        epoch_loss += loss.detach().item()
        count += 1

    print(f"epoch {epoch+1}: {epoch_loss / count:.4f}")

