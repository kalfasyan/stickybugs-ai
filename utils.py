import pandas as pd
import numpy as np
from pathlib import Path
from configparser import ConfigParser
from PIL import Image

cfg = ConfigParser()
cfg.read('config.ini')

DATA_DIR = Path(cfg.get('base', 'data_dir'))
REPO_DIR = Path(cfg.get('base', 'repo_dir'))
SAVE_DIR = Path(cfg.get('base', 'save_dir'))

def read_image(filename, plot=False):
    img = Image.open(filename)
    return img

def get_files(directory, ext='.jpg'):
    return pd.Series(Path.rglob(directory, f"**/*{ext}"))

def extract_filename_info(filename: str, setting='fuji') -> str:
    if not isinstance(filename, str):
        raise TypeError("Provide the filename as a string.")
    
    path = Path(filename)
    datadir_len = len(DATA_DIR.parts)
    parts = path.parts

    if setting == 'fuji':
        label = parts[datadir_len]
        name = parts[datadir_len+1]

        name_split_parts = name.split('_')

        year = name_split_parts[0]
        location = name_split_parts[1]
        if location.startswith("UNDISTORTED"):
            location = name_split_parts[2]
            date = name_split_parts[3]
            xtra = name_split_parts[4]
            idx = name_split_parts[-1]
        else:
            date = name_split_parts[2]
            xtra = name_split_parts[3]
            idx = name_split_parts[-1]

    elif setting == 'photobox':
        raise NotImplementedError()


    return filename, label, name[:-4], year, location, date, xtra, idx[:-4]
    