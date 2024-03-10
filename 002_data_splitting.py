#!/usr/bin/env python
# coding: utf-8

# # Script description: Splitting the data into train, validation and test sets
# Author: Yannis Kalfas (ioannis.kalfas@kuleuven.be; kalfasyan@gmail.com)  
# 
# In this script, we split the data into train, validation and test sets. 
# To do so, we create a feature called `platename_uniq` which is a unique identifier for each plate. 
# We then use this feature to split the data into train, validation and test sets.

import yaml
from settings import Settings
from category_encoders import OrdinalEncoder
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import check_if_outlier, train_test_split_stickyplate
from sklearn.model_selection import train_test_split

# Helper functions
# \----------------
def restart_program():
    """
    Restarts the current program, with file objects and descriptors cleanup.
    """
    import sys
    import os

    python = sys.executable
    os.execl(python, python, *sys.argv)

def check_condition(df, N, classes):
    """
    We check if there are at least N images per class in the df per system
    """
    return all([len(df[(df['system'] == system) & (df['txt_label'] == class_)]) >= N
            for system in settings.multi_system_training
            for class_ in classes])
# ----------------/

# Load the settings
with open("config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

settings = Settings(**config)

if len(settings.multi_system_training):
    systems: list = []
    for system in settings.multi_system_training:
        sub = pd.read_parquet(f"{settings.exports_dir}/df_preparation_{system}.parquet")
        sub['system'] = system
        systems.append(sub)
    df = pd.concat(systems)
else:
    df = pd.read_parquet(f"{settings.exports_dir}/df_preparation_{settings.system}.parquet")
    df['system'] = settings.system

df.reset_index(drop=True, inplace=True)
df.label = df.label.astype(str)

# print(f"Original df columns: \n{df.columns}", end=f"\n{100*'-'}\n")
# print(f"Platenames (unique): \n{df.platename.unique()}", end=f"\n{100*'-'}\n")

oe = OrdinalEncoder(cols=['label'], mapping=[{'col': 'label', 'mapping': settings.insect_labels_map[settings.system]}])
oe_class_names = list(settings.insect_labels_map[settings.system].keys())
tmp = oe.fit_transform(df['label'])
df.rename(columns={"label": "txt_label"}, inplace=True)

df['label'] = tmp.copy()
print(f"Number of classes: {len(df.label.unique())}", end=f"\n{100*'-'}\n")
print(f"Number of images per class: \n{df.txt_label.value_counts()}", end=f"\n{100*'-'}\n")

# Add a number on top of each bar
for i, v in enumerate(df.txt_label.value_counts()):
    plt.text(i-0.3, v+10, str(v), color='black', fontweight='bold')
if len(settings.multi_system_training):
    plt.title(f"Number of images per class in the {settings.multi_system_config} dataset")
else:
    plt.title(f"Number of images per class in the {settings.system} dataset")

df['platename_uniq'] = df.year.astype(str) + '_' + df.location.astype(str) + '_' + df.xtra.str.lower() + '_' + df.date.astype(str) + '_' + df.system.astype(str) 
# print(f"Platenames (unique): \n{df.platename_uniq.unique()}", end=f"\n{100*'-'}\n")

print(f"There are {len(df.platename_uniq.unique())} unique plates in the dataset")
if "photobox" in settings.multi_system_training or (len(settings.multi_system_training) == 0 and (settings.system == "photobox")):
    assert len(df[(df.txt_label == "wmv") & (df.system == "photobox")]) > 0, "There are no wmv images in the photobox dataset"

# Remove files that have been marked as outliers in the script 000_data_preparation
idx = check_if_outlier(df)
if len(idx):
    df = df.drop(idx)
    print(f"Removed {len(idx)} files that were marked as outliers.")

if "photobox" in df.system.unique(): assert len(df[(df.txt_label == "wmv") & (df.system == "photobox")]) > 0, "There are no wmv images in the photobox dataset"
if "photobox" in df.system.unique(): assert len(df[(df.txt_label == "wswl") & (df.system == "photobox")]) > 0, "There are no wswl images in the photobox dataset"
if "fuji" in df.system.unique(): assert len(df[(df.txt_label == "wmv") & (df.system == "fuji")]) > 0, "There are no wmv images in the fuji dataset"

if len(settings.classes_to_remove):
    # Remove some classes from the data
    classes_to_remove = settings.classes_to_remove
    df = df[~df['txt_label'].isin(classes_to_remove)]

# Drop any rows that contain 'test' or 'Test' in the xtra column since people ignored the instructions to use ⚠️TEST⚠️ as location when testing the app...
df = df[~df.xtra.str.contains('test', case=False)]

# Let's see how our df looks like. Specifically the columns : system, txt_label, platename_uniq, date
print(f"Sample: \n{df[['system', 'txt_label', 'platename_uniq', 'date']].sample(5)}", end=f"\n{100*'-'}\n")

# let's now have a look per system
print(f"Number of images per system: \n{df.system.value_counts()}", end=f"\n{100*'-'}\n")

# Let's find the weeks present (ordered by index) and number of images per week
weeks_present = df.date.value_counts().index
weeks_present = sorted(weeks_present)
print(f"Weeks present: {weeks_present}", end=f"\n{100*'-'}\n")
print(f"Number of images per week (ordered by week): \n{df.date.value_counts()[weeks_present]}", end=f"\n{100*'-'}\n")

# NOTE
# We select data from the first W weeks for all our data (-1 means all weeks)
W = settings.weeks

df = df[df.date.isin(weeks_present[:W])]
# Let's print again the weeks present (ordered by index) and number of insects total
weeks_present = df.date.value_counts().index
weeks_present = sorted(weeks_present)
print(f"Weeks present: {weeks_present}", end=f"\n{100*'-'}\n")
print(f"Value counts of whole dataframe (before splitting): \n{df.txt_label.value_counts()}", end=f"\n{100*'-'}\n")
# We save all classes present in the df now and verify they all exist in df_train, df_val and df_test
all_classes_present = df.txt_label.unique()

if settings.system == "fuji" or "fuji" in settings.multi_system_training:
    print(f"The not_insect class has {df[df.txt_label=='not_insect'].shape[0]} images so we will downsample it to match the number of images in the second smallest largest class")
    # Downsample the not_insect class to match the number of images in the third largest class
    third_largest_class = df.txt_label.value_counts().index[2]
    number_of_images_in_third_largest_class = df[df.txt_label==third_largest_class].shape[0]
    not_insect = df[df.txt_label=='not_insect'].sample(number_of_images_in_third_largest_class, random_state=42)
    df = df[df.txt_label!='not_insect']
    df = pd.concat([df, not_insect])

df = df.reset_index(drop=True)

# We first check if there are at least N images per class in the df per system. If not, we split the
# data into train, val and test sets again. We do this until there are at least N images per class in the df per system.
# We do this for a selection of classes as defined below. This was necessary when we didn't have enough data yet (September 2023).
N = 1
classes = ['wmv','wrl'] #'k','c','t'] # 'wrl'

# We need a first split to get the train, val and test sets
df_train, df_test = train_test_split_stickyplate(df, test_size=0.18, shuffle=True)
df_train, df_val = train_test_split_stickyplate(df_train, test_size=0.18, shuffle=True)


# Let's see all weeks present in df_train, df_val and df_test
print(f"TRAIN weeks: {df_train.date.value_counts().index}")
print(f"VAL weeks: {df_val.date.value_counts().index}")
print(f"TEST: {df_test.date.value_counts().index}")

# Let's set a limit of 100 iterations
iterations = 200
i = 0
while not check_condition(df_train, N, classes) or not check_condition(df_val, N, classes) or not check_condition(df_test, N, classes):

    # NOTE: fuji has all wrl images in the same week so it's hard to split. This if condition is necessary to handle this case.
    # Make sure to add 'wrl' in the classes list! Otherwise the first split might exclude wrl from one of the sets.
    # if there is fuji or photobox in the multi_system_training
    if "fuji" in settings.multi_system_training or "photobox" in settings.multi_system_training \
        or "fuji" in settings.system or "photobox" in settings.system:
        # Make a new df where insect is wrl and system is fuji or photobox
        df_wrl_fuji_photobox = df[(df.txt_label == 'wrl') & (df.system.isin(['fuji', 'photobox']))]
        # Remove this part from df
        df = df[~df.index.isin(df_wrl_fuji_photobox.index)]
        # Create a df_wrl_fuji_photobox_train, df_wrl_fuji_photobox_val and df_wrl_fuji_photobox_test
        # We use the function train_test_split instead of train_test_split_stickyplate to randomly split the data
        df_wrl_fuji_photobox_train, df_wrl_fuji_photobox_test = train_test_split(df_wrl_fuji_photobox, test_size=0.18, shuffle=True, random_state=42)
        df_wrl_fuji_photobox_train, df_wrl_fuji_photobox_val = train_test_split(df_wrl_fuji_photobox_train, test_size=0.18, shuffle=True, random_state=42)

    df_train, df_test = train_test_split_stickyplate(df, test_size=0.2, shuffle=True)
    df_train, df_val = train_test_split_stickyplate(df_train, test_size=0.2, shuffle=True)

    if "fuji" in settings.multi_system_training or "photobox" in settings.multi_system_training \
        or "fuji" in settings.system or "photobox" in settings.system:
        # Add the wrl back to the dfs
        df_train = pd.concat([df_train, df_wrl_fuji_photobox_train])
        df_val = pd.concat([df_val, df_wrl_fuji_photobox_val])
        df_test = pd.concat([df_test, df_wrl_fuji_photobox_test])

    print(f"The sizes of df: train: {len(df_train)}, val: {len(df_val)}, test: {len(df_test)}")
    i+=1
    if i > iterations: 
        print(f"The condition is not met after {iterations} iterations. Please check the data.")
        restart_program()

# We make sure that all classes are present in the train, val and test sets
checks = []
for ins_cls in all_classes_present:
    if ins_cls not in df_train.txt_label.unique():
        print(f"Class {ins_cls} is not present in the train set")
        checks.append(False)
    if ins_cls not in df_val.txt_label.unique():
        print(f"Class {ins_cls} is not present in the val set")
        checks.append(False)
    if ins_cls not in df_test.txt_label.unique():
        print(f"Class {ins_cls} is not present in the test set")
        checks.append(False)

# If there's at least one False in checks, then we raise an exception
if False in checks:
    print(f"Not all classes are present in the train, val and test sets")
    restart_program()

print(f"FINAL: Size of train set: {len(df_train)}, size of val set: {len(df_val)}, size of test set: {len(df_test)}", end=f"\n{100*'-'}\n")

excluding_randomly = False
# If this is True, we will exclude some plates from the train set and downsample the wmv class
if excluding_randomly:
    SAMPLE_PLATES_PCT = 0.5
    SAMPLE_WMV_PCT = 0.2

    print(200*"-")
    print("Selecting a smaller number of sticky plates for the train set")
    print(200*"-")

    # We exclude plates that contain wswl, t, k 
    plates_to_exclude = df_train[(df_train.txt_label == 'wswl') | (df_train.txt_label == 't') | (df_train.txt_label == 'k')].platename_uniq.unique()
    # plates_to_exclude = df_train[df_train.txt_label == 'wswl'].platename_uniq.unique()

    train_plates = df_train.platename_uniq.unique()
    print(f"Number of plates to exclude: {len(plates_to_exclude)}")
    print(f"Number of plates in train set: {len(train_plates)}")
    # Now let's exclude the plates that contain either wswl, t or k
    train_plates = [plate for plate in train_plates if plate not in plates_to_exclude]
    print(f"Number of plates in train set after excluding plates of some classes: {len(train_plates)}")
    # Select a smaller number of plates for the train set
    # We select some of the plates for the train set
    train_plates = np.random.choice(train_plates, size=int(len(train_plates)*SAMPLE_PLATES_PCT), replace=False)
    print(f"Number of plates in train set after selecting some plates: {len(train_plates)}")
    # Put the excluded plates back in the train set
    train_plates = np.concatenate([train_plates, plates_to_exclude])
    print(f"Number of plates in train set after putting excluded back: {len(train_plates)}")

    df_train = df_train[df_train.platename_uniq.isin(train_plates)]
    print(f"FINAL: Size of train set: {len(df_train)}, size of val set: {len(df_val)}, size of test set: {len(df_test)}")

    # Now let's downsample the wmv class specifically in the train set
    # We first check how many wmv images there are in the train set
    print(f"Number of wmv images in train set: {len(df_train[df_train.txt_label == 'wmv'])}")
    # We downsample the wmv class by selecting 20% of the wmv images but we keep the other classes as they are
    wmv = df_train[df_train.txt_label == 'wmv'].sample(frac=SAMPLE_WMV_PCT, random_state=42)
    df_train = df_train[df_train.txt_label != 'wmv']
    df_train = pd.concat([df_train, wmv])
    print(f"Number of wmv images in train set after removing: {len(df_train[df_train.txt_label == 'wmv'])}")

    # Let's remove duplicate rows from df_train
    print(f"Number of duplicate rows in df_train: {df_train.duplicated().sum()}")
    df_train = df_train.drop_duplicates()
    print(f"Number of duplicate rows in df_train after removing: {df_train.duplicated().sum()}")

def plots_per_system(df, split='TEST'):
    # Get the counts of txt_label for each system and plot them

    sns.set_palette('colorblind')
    sns.set_context('paper', font_scale=1.5)
    # Always sns.despine() after plotting to remove top and right spines
    # df_test.groupby(['system','txt_label']).count()['imgname'].unstack().plot(kind='bar', stacked=True, figsize=(15,5)); sns.despine()
    # Plot the value counts of txt_label for each system in separate plots but order the txt_labels by their name
    plt.figure(figsize=(15,10))
    if len(settings.multi_system_training):
        for i, system in enumerate(df['system'].unique()):
            plt.subplot(len(settings.multi_system_training), 1, i+1)
            sns.countplot(x='txt_label', data=df[df['system']==system], order=sorted(df['txt_label'].unique()))
            sns.despine()
            # print the numbers on top of the bars as integers
            for p in plt.gca().patches:
                plt.gca().annotate(int(p.get_height()), (p.get_x()+p.get_width()/2., p.get_height()),
                                ha='center', va='center', xytext=(0, 10), textcoords='offset points')
            plt.title(f"{split}_{system}", y=1.05, fontsize=20, loc='left')
            plt.xlabel('Insect class')
            plt.ylabel('Count')
            plt.tight_layout()
            plt.savefig(f"{settings.exports_dir}/{split}_class_distribution.png")
    else:
        sns.countplot(x='txt_label', data=df, order=sorted(df['txt_label'].unique()))
        sns.despine()
        # print the numbers on top of the bars as integers
        for p in plt.gca().patches:
            plt.gca().annotate(int(p.get_height()), (p.get_x()+p.get_width()/2., p.get_height()),
                            ha='center', va='center', xytext=(0, 10), textcoords='offset points')
        plt.title(f"{split}_{settings.system}", y=1.05, fontsize=20, loc='left')
        plt.xlabel('Insect class')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(f"{settings.exports_dir}/{split}_class_distribution.png")

plt.figure()
df_train['txt_label'].value_counts().plot(kind='bar', figsize=(15,5), title="Train"); sns.despine()
for i, v in enumerate(df_train['txt_label'].value_counts()):
    plt.text(i, v, str(v), ha='center', va='bottom')
plt.savefig(f"{settings.exports_dir}/TRAIN_class_distribution_all_systems.png")
plt.figure()
df_val['txt_label'].value_counts().plot(kind='bar', figsize=(15,5), title="Val"); sns.despine()
for i, v in enumerate(df_val['txt_label'].value_counts()):
    plt.text(i, v, str(v), ha='center', va='bottom')
plt.savefig(f"{settings.exports_dir}/VAL_class_distribution_all_systems.png")
plt.figure()
df_test['txt_label'].value_counts().plot(kind='bar', figsize=(15,5), title="Test"); sns.despine()
for i, v in enumerate(df_test['txt_label'].value_counts()):
    plt.text(i, v, str(v), ha='center', va='bottom')
plt.savefig(f"{settings.exports_dir}/TEST_class_distribution_all_systems.png")
plt.close('all')

plots_per_system(df_train, split='TRAIN')
plots_per_system(df_val, split='VAL')
plots_per_system(df_test, split='TEST')
plt.close('all')

# Save the splits in settings.exports_dir as a parquet file
# The parquet file will be used in the next script

assert len(set(df_train.filename.tolist()).intersection(df_test.filename.tolist())) == 0
assert len(set(df_train.filename.tolist()).intersection(df_val.filename.tolist())) == 0

# Let's have a look at the insect labels value counts for df_train, df_val, df_test
print(f"Train set insect labels value counts: \n{df_train['txt_label'].value_counts()}", end=f"\n{'-'*100}\n")
print(f"Val set insect labels value counts: \n{df_val['txt_label'].value_counts()}", end=f"\n{'-'*100}\n")
print(f"Test set insect labels value counts: \n{df_test['txt_label'].value_counts()}", end=f"\n{'-'*100}\n")


df_train.to_parquet(f"{settings.exports_dir}/df_train_fixed.parquet")
df_val.to_parquet(f"{settings.exports_dir}/df_val_fixed.parquet")
df_test.to_parquet(f"{settings.exports_dir}/df_test_fixed.parquet")

# If the time is after 7pm then send a Teams message to say the data has been split
from datetime import datetime

if datetime.now().hour >= 20 or datetime.now().hour <= 8:
    from utils import send_teams_message
    send_teams_message("Data splitting complete.")

