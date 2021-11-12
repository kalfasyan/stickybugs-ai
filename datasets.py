import numpy as np
import pandas as pd
import psutil
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from utils import (DATA_DIR, REPO_DIR, SAVE_DIR, extract_filename_info, get_files,
                   read_image, dataframe_columns)

num_workers = psutil.cpu_count()
print(f"Available workers: {num_workers}")


class DataFrameSet(Dataset):
    """
    Dataset class that can take a pandas.DataFrame as input.
    """

    def __init__(self, directory=DATA_DIR, setting="fuji", transform=None):
        self.directory = directory
        self.files = get_files(directory)
        
        self.df = pd.DataFrame(self.files, columns=['filename'])
        self.df = self.df.reset_index(drop=True)
        
        self.setting = setting
        self.transform = transform        

    def extract_df_info(self):
        tmp = self.df.filename.apply(lambda x: extract_filename_info(str(x), setting=self.setting))
        self.df = pd.DataFrame.from_records(tmp, columns=dataframe_columns)        

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

        img = read_image(fname, plot=False)
        sample = {"img": img, 
                "label": label, 
                "imgname": imgname,
                "platename": platename,
                "filename": str(fname), 
                "plate_idx": plate_idx, 
                "location": location, 
                "date": date, 
                "year": year,
                "xtra": xtra}

        if self.transform:
            sample=self.transform(sample)

        return tuple(sample.values())
