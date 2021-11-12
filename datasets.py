import numpy as np
import pandas as pd
import psutil
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as T

from utils import (DATA_DIR, REPO_DIR, SAVE_DIR, extract_filename_info, get_files,
                   read_image, dataframe_columns)

num_workers = psutil.cpu_count()
print(f"Available workers: {num_workers}")
seed = 42

class InsectImgDataset(Dataset):
    """
    Dataset class that can take a dataset directory as input.
    It creates a dataframe with all relevant insect info such as: sticky plate name, year, date etc.
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

        pil_img = read_image(fname, plot=False)

        wsize = 150
        hsize = 150
        pil_img = pil_img.resize((wsize,hsize), Image.ANTIALIAS)
        
        width, height = pil_img.size
        tensor_img = transforms.ToTensor()(pil_img).unsqueeze_(0)
        sample = {"tensor_img": tensor_img,
                "label": label, 
                # "pil_img": pil_img,
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
