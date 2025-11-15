import torch
from torch.utils.data import DataLoader
from dataset import DockingDataset
from model import SimpleGeomNet
from diffusion import DiffusionWrapper

dataset = DockingDataset("data")
loader = DataLoader(dataset, batch_size=1, shuffle=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = SimpleGeomNet().to(device)
diff = DiffusionWrapper(model).to(device)

opt = torch.optim.Adam(diff.parameters(), lr=1e-4)

for epoch in range(50):
    for lig_pos, lig_type, poc_pos, poc_type in loader:
        # remove batch dim
        lig_pos = lig_pos.squeeze(0)
        lig_type = lig_type.squeeze(0)
        poc_pos = poc_pos.squeeze(0)
        poc_type = poc_type.squeeze(0)

        lig_pos = lig_pos.to(device)
        lig_type = lig_type.to(device)
        poc_pos = poc_pos.to(device)
        poc_type = poc_type.to(device)

        loss = diff.loss(lig_pos, lig_type, poc_pos, poc_type)

        opt.zero_grad()
        loss.backward()
        opt.step()

        print("loss:", float(loss))

    print("epoch:", epoch)
