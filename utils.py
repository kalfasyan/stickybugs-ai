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


date_mapping = {
    "1926719": "w30",
    "1219719": "w29",
    "02090819": "w32",
    "262719" : "w31",
    "512719": "w28",
    "09160819": "w33",
    "2128619": "w26",
    "2856719": "w27",
    "30719": "w30",
    "8719": "w27",
    "15": "w28",
    "w24": "w24",
    "w25": "w25",
    "w26": "w26",
    "w27": "w27",
    "w28": "w28",
    "w29": "w29",
    "w30": "w30",
    "w31": "w31",
    "w32": "w32",
    "w33": "w33",
    "w34": "w34",
    "w35": "w35",
    "w36": "w36",
    "w37": "w37",
    "w38": "w38",
    "w39": "w39",
    "w40": "w40",
    "w41": "w41",
}

# Creating the location mapping to fix location names from plates
location_mapping = {
    "herentval1": "herent",
    "herentval2": "herent",
    "herentval3": "herent",
    "herentcontrole": "herent",
    "merchtem": "merchtem",
    "mollem": "mollem",
    "landen": "landen",
    "herent": "herent",
    "her": "herent",
    "kampen": "kampenhout",
    "braine": "brainelalleud",
    "brainelal": "brainelalleud",
    "brainlal": "brainelalleud",
    "beauvech": "beauvechain",
    "beauv": "beauvechain",
    "beavech" : "beauvechain",
    "Racour" : "racour",
    "racour": "racour",
    "Merchtem": "merchtem",
    "wortel": "wortel",
}

dataframe_columns = ['filename', 'label','imgname','platename','year','location','date','xtra','plate_idx']

def read_image(filename, plot=False):
    img = Image.open(filename)
    return img

def get_files(directory, ext='.jpg'):
    return pd.Series(Path.rglob(directory, f"**/*{ext}"))

def to_weeknr(date=''):
    """
    Transforms a date strings YYYYMMDD to the corresponding week nr (e.g. 20200713 becomes w29)
    """
    week_nr = pd.to_datetime(date).to_pydatetime().isocalendar()[1]
    return f"w{week_nr}"

def format_date(date: str) -> str:
    try:
        date = date_mapping[date.lower()]
    except:
        pass
    try:
        date = to_weeknr(date)
    except:
        pass
    return date

def format_location(location: str) -> str:
    return location_mapping[location.lower()]

def extract_filename_info(filename: str, setting='fuji') -> str:
    if not isinstance(filename, str):
        raise TypeError("Provide the filename as a string.")
    
    path = Path(filename)
    datadir_len = len(DATA_DIR.parts)
    parts = path.parts

    if setting == 'fuji':
        label = parts[datadir_len]
        imgname = parts[datadir_len+1]
        platename = "_".join(imgname.split('_')[1:-1])

        name_split_parts = imgname.split('_')

        year = name_split_parts[0]
        location = name_split_parts[1]
        if location.startswith("UNDISTORTED"):
            location = name_split_parts[2]
            date = name_split_parts[3]
            xtra = name_split_parts[4] if not name_split_parts[4].startswith("3daysold") else name_split_parts[5]
            plate_idx = name_split_parts[-1]
        else:
            date = name_split_parts[2]
            xtra = name_split_parts[3]  if not name_split_parts[3].startswith("3daysold") else name_split_parts[4]
            plate_idx = name_split_parts[-1]

    elif setting == 'photobox':
        raise NotImplementedError()
    else:
        raise ValueError()


    return filename, label, imgname[:-4], platename, year, format_location(location), format_date(date), xtra, plate_idx[:-4]
    