# import torch
# from torch.utils.data import DataLoader
# from dataset import DockingDataset
# from model import SE3Block
# from diffusion import DiffusionWrapper

# dataset = DockingDataset("data")
# loader = DataLoader(dataset, batch_size=1, shuffle=True)

# device = "cuda" if torch.cuda.is_available() else "cpu"

# model = SE3Block().to(device)
# diff = DiffusionWrapper(model).to(device)

# opt = torch.optim.Adam(diff.parameters(), lr=1e-4)

# for epoch in range(50):
#     for lig_pos, lig_type, poc_pos, poc_type in loader:
#         lig_pos = lig_pos.to(device)
#         poc_pos = poc_pos.to(device)

#         loss = diff.loss(lig_pos, poc_pos)

#         opt.zero_grad()
#         loss.backward()
#         opt.step()


import torch
from torch.utils.data import DataLoader
from dataset import DockingDataset
from model import SE3Block
from feature_encoder import FeatureEncoder
from diffusion import DiffusionWrapper

dataset = DockingDataset("data")
loader = DataLoader(dataset, batch_size=1, shuffle=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = SE3Block().to(device)
encoder = FeatureEncoder().to(device)
diff = DiffusionWrapper(model, encoder).to(device)

opt = torch.optim.Adam(diff.parameters(), lr=1e-4)

for epoch in range(50):
    for lig_pos, lig_type, poc_pos, poc_type in loader:
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
