import numpy as np
from torch.utils.data import Dataset
import torch
import os

class DockingDataset(Dataset):
    def __init__(self, root):
        self.root = root
        self.ids = [x.split(".")[0] for x in os.listdir(root + "/ligands")]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        key = self.ids[idx]
        lig = np.load(f"{self.root}/ligands/{key}.npy", allow_pickle=True).item()
        poc = np.load(f"{self.root}/pockets/{key}.npy", allow_pickle=True).item()

        lig_pos = torch.tensor(lig["pos"], dtype=torch.float32)
        lig_type = torch.tensor(lig["atom_type"], dtype=torch.long)

        poc_pos = torch.tensor(poc["pos"], dtype=torch.float32)
        poc_type = torch.tensor(poc["atom_type"], dtype=torch.long)

        return lig_pos, lig_type, poc_pos, poc_type
