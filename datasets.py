import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import torch
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as fn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from utils import (DATA_DIR, REPO_DIR, SAVE_DIR, basic_df_columns,
                   extract_filename_info, get_files, read_image)

num_workers = psutil.cpu_count()
print(f"Available workers: {num_workers}")
seed = 42

class InsectImgDataset(Dataset):
    """
    Dataset class that can take a dataset directory as input.
    It creates a dataframe with all relevant insect info such as: sticky plate name, year, date etc.
    """

    def __init__(self, directory=DATA_DIR, ext='.png', setting="fuji", img_dim=150, transform=None):
        self.directory = directory
        self.ext = ext
        self.files = get_files(directory, ext=self.ext)
        
        self.df = pd.DataFrame(self.files, columns=['filename'])
        self.df = self.df.astype(str).reset_index(drop=True)
        
        self.setting = setting
        self.img_dim = img_dim
        self.transform = transform

    def extract_df_info(self):
        info = []
        for row in tqdm(self.df.itertuples(), total=len(self.df), desc="Extracting info from filenames.."):
            info.append(extract_filename_info(row.filename, setting=self.setting))
        self.df = pd.DataFrame(info, columns=[basic_df_columns])
        self.df.columns = [' '.join(col).strip() for col in self.df.columns.values]
        if not len(self.df):
            raise ValueError("Dataframe was not loaded.")
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
            sample = extract_filename_info(str(self.df.loc[idx].filename), setting=self.setting)
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

        tensor_img = torchvision.io.read_image(fname)
        tensor_img = fn.center_crop(tensor_img, output_size=[self.img_dim])
        tensor_img = fn.resize(tensor_img, size=[self.img_dim])

        _, width, height = tensor_img.size()

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
            sample=self.transform(sample)

        return tuple(sample.values())


    def plot_samples(self, df=pd.DataFrame(), noaxis=True, title='label'):
        if not len(df):
            df = self.df.sample(20, replace=False, random_state=seed).reset_index(drop=True)
        else:
            df = df.sample(20, replace=False, random_state=seed).reset_index(drop=True)

        plt.figure(figsize=(20,12))

        for i in tqdm(range(20)):
            plt.subplot(4,5,i+1)
            img = read_image(df.loc[i].filename)
            plt.imshow(img);
            if title == 'label':
                plt.title(df.loc[i].label)
            if noaxis:
                plt.axis('off')
