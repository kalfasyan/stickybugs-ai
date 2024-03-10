import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import torch
import torchvision
import torchvision.transforms.functional as fn
from PIL import Image
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from utils import (basic_df_columns, extract_filename_info, get_files,
                   read_image)

num_workers = psutil.cpu_count()
print(f"Available workers: {num_workers}")
SEED = 42


class InsectImgDataset(Dataset):
    """
    Dataset class that can take a dataset directory as input.
    It creates a dataframe with all relevant insect info such as: sticky plate name, year, date etc.
    """

    def __init__(self, df=pd.DataFrame(), directory='', ext='.png', system="fuji", img_dim=150, transform=None):
        self.system = system
        self.directory = str(directory) if len(str(directory)) else ''
        self.ext = ext
        self.df = df

        if not self.directory:
            assert len(self.df), "You chose to use a pre-made dataframe."
        else:
            assert len(self.directory)
            assert self.df.empty, "You chose to use a directory and load its filenames into a dataframe."

            self.files = get_files(self.directory, ext=self.ext)
            assert len(self.files), f"Couldn't find any files in:\n{self.directory}\n1. Is this path correct?\n2. Is the extension {self.ext} correct?"

            self.df = pd.DataFrame(self.files, columns=['filename'])
            self.df = self.df.astype(str).reset_index(drop=True)

        self.img_dim = img_dim
        self.transform = transform

    def extract_df_info(self, fix_cols=False):
        info = []
        for row in tqdm(self.df.itertuples(), total=len(self.df), desc="Extracting info from filenames.."):
            info.append(extract_filename_info(row.filename, system=self.system))
        self.df = pd.DataFrame(info, columns=basic_df_columns)
        if fix_cols:
            self.df.columns = [' '.join(col).strip() for col in self.df.columns.values]
        if not len(self.df):
            raise ValueError("Dataframe was not loaded.")
        self.df.year = self.df.year.astype('int64')
        self.df.plate_idx = self.df.plate_idx.astype('int64')
        self.info_extracted = True

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if len(self.df.columns) > 1:
            sample = self.df.loc[idx]
        else:
            print("Run extract_df_info to read image data.")
            sample = extract_filename_info(str(self.df.loc[idx].filename), system=self.system)
            return sample

        fname = sample["filename"]
        label = sample["label"]
        imgname = sample["imgname"]
        platename = sample["platename"]
        year = sample["year"]
        location = sample["location"]
        date = sample["date"]
        xtra = sample["xtra"]
        plate_idx = sample["plate_idx"]

        # Read an image from a file as a tensor.
        img = Image.open(fname)
        tensor_img = torchvision.transforms.functional.to_tensor(img)
        tensor_img = fn.resize(tensor_img, size=(150, 150), antialias=True)

        _, width, height = tensor_img.shape  # tensor_img.size()

        sample = {"tensor_img": tensor_img,
                  "label": label,
                  "imgname": imgname,
                  "platename": platename,
                  "filename": str(fname),
                  "plate_idx": plate_idx,
                  "location": location,
                  "date": date,
                  "year": year,
                  "xtra": xtra,
                  "width": width,
                  "height": height}

        if self.transform:
            sample["tensor_img"] = self.transform(sample["tensor_img"])

        return tuple(sample.values())

    def plot_samples(self, df=pd.DataFrame(), noaxis=True, title='label'):
        if not len(df):
            df = self.df.sample(20, replace=False, random_state=SEED).reset_index(drop=True)
        else:
            df = df.sample(20, replace=False, random_state=SEED).reset_index(drop=True)

        plt.figure(figsize=(20, 12))

        for i in tqdm(range(20)):
            plt.subplot(4, 5, i+1)
            img = read_image(df.loc[i].filename)
            plt.imshow(img)
            if title == 'label':
                plt.title(df.loc[i].label)
            if noaxis:
                plt.axis('off')

# def worker_init_fn(worker_id):
#     np.random.SEED(np.random.get_state()[1][0] + worker_id)

def worker_init_fn(worker_id):
    """ Fixing the seed for each worker to ensure reproducibility """
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)
    return
