import pandas as pd
import numpy as np
import psutil
from utils import SAVE_DIR, REPO_DIR, DATA_DIR, read_image

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

num_workers = psutil.cpu_count()
print(f"Available workers: {num_workers}")


class DataFrameSet(Dataset):
    """
    Dataset class that can take a pandas.DataFrame as input.
    """

    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __gettitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.df.loc[idx]
        fname = sample["filename"]
        label = sample["label"]

        img = read_image(fname, plot=False)
        sample = {"x": img, "y": label, "path": str(fname), "idx": idx}

        if self.transform:
            sample=self.transform(sample)

        return sample["x"], sample["y"], sample["path"], sample["idx"]
