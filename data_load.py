import os

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class LiDARnpzDataset(Dataset):
    def __init__(self, dataset_root, transform=None):
        self.files = [os.path.join(dataset_root, f)
                      for f in os.listdir(dataset_root) if f.endswith(".npz")]
        self.files.sort()
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        x = torch.tensor(data["x"], dtype=torch.float32)  # [max_points, 3]
        y = torch.tensor(data["y"], dtype=torch.float32)  # [max_points, 3]
        x_mask = torch.tensor(data["x_mask"], dtype=torch.float32)  # [max_points]
        y_mask = torch.tensor(data["y_mask"], dtype=torch.float32)  # [max_points]
        rgb = torch.tensor(data["rgb"], dtype=torch.float32)  # [max_points]
        if self.transform:
            rgb = self.transform(rgb.permute(2, 0, 1))

        return {"x": x, "y": y, "x_mask": x_mask, "y_mask": y_mask, "rgb": rgb}
