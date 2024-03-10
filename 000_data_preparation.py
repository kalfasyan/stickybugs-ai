#!/usr/bin/env python
# coding: utf-8

import sys
import warnings

import pandas as pd
import psutil
from torch.utils.data import DataLoader

from datasets import InsectImgDataset
from utils import *

warnings.filterwarnings("ignore", category=UserWarning, module="torch")
import yaml
from tqdm.auto import tqdm

from settings import Settings

with open("config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

settings = Settings(**config)

print("Current environment:", sys.prefix)
print(f"System: {settings.system}")

dfs = InsectImgDataset(directory=settings.data_dir.as_posix(), ext='.png', system=settings.system)
dfs.extract_df_info();
dfs.df.label = dfs.df.label.astype(str);
insect_classes = list(settings.insect_labels_map[settings.system].keys())
other_categs = ['gv_gaasvlieg','sl','ONBEKEND','w','lhb','zv','unknown','mot','psylloidea','sv','vl','gaasvlieg','wants',"?","."]
dfs.df.label = dfs.df.label.apply(lambda x: 'other' if x in other_categs else x)
dfs.df.label = dfs.df.label.apply(lambda x: 'not_insect' if x in ['st','vuil'] else x)
dfs.df = dfs.df[dfs.df.label.isin(insect_classes)].copy()
dfs.df.reset_index(drop=True, inplace=True)
dfs.df.label.value_counts()


# # Extra feature collection
dloader = DataLoader(dfs, batch_size=settings.batch_size, shuffle=False, num_workers=settings.num_workers, pin_memory=True)

l_blur_factors = [0]*len(dfs)
l_meansRGB = [0]*len(dfs)
l_stdsRGB = [0]*len(dfs)
l_nb_contours, l_mean_cnt_area, l_mean_cnt_perimeter, l_std_cnt_area, l_std_cnt_perimeter = [
    0]*len(dfs), [0]*len(dfs), [0]*len(dfs), [0]*len(dfs), [0]*len(dfs)

print(f"Memory usage: {psutil.virtual_memory().percent}%")
c = 0
for x, l, imgname, platename, filename, plate_idx, location, date, year, xtra, width, height \
    in tqdm(dloader, total=len(dfs)//settings.batch_size, desc='Collecting all data from the dataloader..'):
    
    for i, f in enumerate(filename):
        meanRGB, stdRGB = calc_mean_RGB_vals(f)
        nb_contours, mean_cnt_area, mean_cnt_perimeter, std_cnt_area, std_cnt_perimeter = calc_contour_features(
            f)
        l_blur_factors[c] = calc_variance_of_laplacian(f)
        l_meansRGB[c] = meanRGB
        l_stdsRGB[c] = stdRGB
        l_nb_contours[c] = nb_contours
        l_mean_cnt_area[c] = mean_cnt_area
        l_mean_cnt_perimeter[c] = mean_cnt_perimeter
        l_std_cnt_area[c] = std_cnt_area
        l_std_cnt_perimeter[c] = std_cnt_perimeter
        c += 1
print(f"Memory usage: {psutil.virtual_memory().percent}%")

df_rgb = pd.DataFrame(l_meansRGB, columns=['R', 'G', 'B'])
df_feats = pd.DataFrame({'blur': l_blur_factors,
                        'nb_contours': l_nb_contours,
                         'mean_cnt_area': l_mean_cnt_area,
                         'mean_cnt_perimeter': l_mean_cnt_perimeter,
                         'std_cnt_area': l_std_cnt_area,
                         'std_cnt_perimeter': l_std_cnt_perimeter})
df_feats = pd.concat([df_feats, df_rgb], axis=1)
feature_columns = df_feats.columns


df = pd.concat([dfs.df, df_feats], axis=1)
# df = pd.concat([df, df_modelfeats], axis=1)
df.sort_values(by='label', inplace=True)

# # Outlier detection

# ### Performed per insect class
l_outlier_features = ['blur', 'R','G','B', 'nb_contours', 'std_cnt_area']#, 'mean_cnt_area', 'mean_cnt_perimeter', 'std_cnt_area', 'std_cnt_perimeter']

def insect_category_outliers(df, features, insect='bl'):
    df = df[df.label==insect]
    print(f"Found {len(df)} images for insect {insect}")
    if len(df) < 100:
        print(f"Too few images for {insect}. Setting all outlier scores to 0.")
        # Return numpy arrays of zeros
        return np.zeros(len(df)), np.zeros(len(df))
    else:
        outlier, outlier_score = detect_outliers(df[features].fillna(0).values, algorithm='KNN')    
        return outlier, outlier_score

df['knn_outlier'], df['knn_outlier_score'] = 0,0
outliers, scores = [],[]
for ins in tqdm(insect_classes, total=len(insect_classes)):
    out, scr = insect_category_outliers(df, l_outlier_features, insect=ins)
    assert len(out) == df[df.label==ins].shape[0]
    outliers.extend(out)
    scores.extend(scr)
    
df['knn_outlier'], df['knn_outlier_score'] = outliers, scores


# # SAVING DATAFRAME

df.to_parquet(f'{settings.exports_dir}/df_preparation_{settings.system}.parquet');
# Save also as csv
df.to_csv(f'{settings.exports_dir}/df_preparation_{settings.system}.csv', index=False)
print("Saved.", end=f"\n{'-'*100}\n")

# If the time is between 7pm and 8am then send a Teams message to say the data has been prepared
from datetime import datetime

if datetime.now().hour >= 17 and datetime.now().hour <= 8:
    from utils import send_teams_message
    send_teams_message(f"Data prepared for {settings.system}")

