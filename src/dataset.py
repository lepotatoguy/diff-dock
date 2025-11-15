import os
import numpy as np
import torch
from torch.utils.data import Dataset

class DockingDataset(Dataset):
    def __init__(self, root):
        self.root = root
        self.ids = sorted(os.listdir(os.path.join(root, "ligands")))

    def __len__(self):
        return len(self.ids)

    def fix_pos(self, arr):
        arr = np.array(arr)
        arr = np.squeeze(arr)

        if arr.ndim == 1:
            arr = arr.reshape(1, 3)

        if arr.shape[-1] != 3:
            if arr.shape[0] == 3:
                arr = arr.T
            else:
                raise RuntimeError("Invalid coordinate shape")

        return arr.astype(np.float32)

    def __getitem__(self, idx):
        pdbid = self.ids[idx]

        lig = np.load(os.path.join(self.root, "ligands", pdbid), allow_pickle=True).item()
        poc = np.load(os.path.join(self.root, "pockets", pdbid), allow_pickle=True).item()

        lig_pos = self.fix_pos(lig["pos"])
        poc_pos = self.fix_pos(poc["pos"])

        lig_type = lig["atom_type"].astype(np.int64)
        poc_type = poc["atom_type"].astype(np.int64)

        return (
            torch.tensor(lig_pos),
            torch.tensor(lig_type),
            torch.tensor(poc_pos),
            torch.tensor(poc_type),
        )
