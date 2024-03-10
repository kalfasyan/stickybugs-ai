#!/usr/bin/env python
# coding: utf-8
import matplotlib.pyplot as plt
import pandas as pd

pd.set_option('display.max_colwidth', None)

import shutil
import warnings
from pathlib import Path

import seaborn as sns

sns.set(font_scale=1.2)

import timm
import torch
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')

import torchvision.transforms as T
import yaml
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader

from datasets import *
from settings import Settings
from utils import *
from utils_test import get_gt_and_preds

with open("config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

settings = Settings(**config)

use_gpu = True if settings.device == 'cuda' else False
print(f'Using gpu: {use_gpu}')  
torch.backends.cudnn.benchmark = True

if len(settings.multi_system_training):
    print(f"Fetching results for systems: {settings.multi_system_training}")

warnings.filterwarnings("ignore", category=UserWarning, module="torch")


# # Loading train,val,test sets
df_train = pd.read_parquet(settings.exports_dir / f"df_train_fixed.parquet")
df_val = pd.read_parquet(settings.exports_dir / f"df_val_fixed.parquet")
df_test = pd.read_parquet(settings.exports_dir / f"df_test_fixed.parquet")

# Select only the systems that we want to test on
df_test = df_test[df_test['system'].isin(settings.multi_system_training)]

if len(settings.classes_to_remove):
    # Remove some classes from the df_train, df_val, df_test
    classes_to_remove = settings.classes_to_remove
    df_train = df_train[~df_train['txt_label'].isin(classes_to_remove)]
    df_val = df_val[~df_val['txt_label'].isin(classes_to_remove)]
    df_test = df_test[~df_test['txt_label'].isin(classes_to_remove)]

assert len(set(df_train.filename.tolist()).intersection(df_test.filename.tolist())) == 0
assert len(set(df_train.filename.tolist()).intersection(df_val.filename.tolist())) == 0
assert len(set(df_test.filename.tolist()).intersection(df_val.filename.tolist())) == 0

df_train = df_train[df_train['system'].isin(settings.multi_system_training)]
df_val = df_val[df_val['system'].isin(settings.multi_system_training)]
df_test = df_test[df_test['system'].isin(settings.multi_system_training)] # comment out if we want to test with all systems* need to remap labels

if len(settings.multi_system_training) > 1: label_map_key = settings.multi_system_config
else: label_map_key = settings.system

oe_class_names = list(settings.insect_labels_map[label_map_key].keys())
print(f"Class mapping used for training: \n{settings.insect_labels_map[label_map_key]}")
df_train['label'] = df_train['txt_label'].map(settings.insect_labels_map[label_map_key])
df_val['label'] = df_val['txt_label'].map(settings.insect_labels_map[label_map_key])
df_test['label'] = df_test['txt_label'].map(settings.insect_labels_map[label_map_key]) # comment out if we want to test with all systems

if len(settings.classes_to_remove):
    # Remove some classes from the df_train, df_val, df_test
    classes_to_remove = settings.classes_to_remove
    df_train = df_train[~df_train['txt_label'].isin(classes_to_remove)]
    df_val = df_val[~df_val['txt_label'].isin(classes_to_remove)]
    df_test = df_test[~df_test['txt_label'].isin(classes_to_remove)]
    # We need to edit oe_class_names and account for the removed classes
    oe_class_names = [label for label in oe_class_names if label not in classes_to_remove]
    # Also, we need to reset the settings.insect_labels_map
    # It should have the oe_class_names as keys and the values should be the indices of the oe_class_names
    settings.insect_labels_map[label_map_key] = {label: i for i, label in enumerate(oe_class_names)}
    print(f"Class mapping adjusted after class removal: \n{settings.insect_labels_map[label_map_key]}", end=f"\n{'-'*100}\n")
    # Now we should remap the labels in df_train, df_val, df_test
    df_train['label'] = df_train['txt_label'].map(settings.insect_labels_map[label_map_key])
    df_val['label'] = df_val['txt_label'].map(settings.insect_labels_map[label_map_key])
    df_test['label'] = df_test['txt_label'].map(settings.insect_labels_map[label_map_key])

# # Creating Pytorch Datasets and Dataloaders
transforms_list_train = [
    #     A.SmallestMaxSize(max_size=150),
    T.ToPILImage(),
    T.Resize(size=(150, 150), antialias=True),
    T.RandomVerticalFlip(p=0.5),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomAutocontrast(p=0.5),
    T.RandomAdjustSharpness(sharpness_factor=1.5, p=0.5),
    T.RandomRotation(degrees=(-5, 5)),
    T.RandomPosterize(bits=7, p=0.1),
    # T.RandomEqualize(p=0.5),
    T.ToTensor(),
]

transforms_list_test = [
    T.ToPILImage(),
    # T.Resize(size=(150, 150), antialias=True),
    T.ToTensor(),
]

train_dataset = InsectImgDataset(df=df_train.reset_index(drop=True), transform=T.Compose(transforms_list_train))
valid_dataset = InsectImgDataset(df=df_val.reset_index(drop=True), transform=T.Compose(transforms_list_test))
test_dataset = InsectImgDataset(df=df_test.reset_index(drop=True), transform=T.Compose(transforms_list_test))

train_dataloader = DataLoader(train_dataset, batch_size=settings.batch_size_val//2, shuffle=True, num_workers=settings.num_workers, pin_memory=False, worker_init_fn=worker_init_fn)
valid_dataloader = DataLoader(valid_dataset, batch_size=settings.batch_size_val//2, shuffle=False, num_workers=settings.num_workers, pin_memory=False, worker_init_fn=worker_init_fn)
test_dataloader = DataLoader(test_dataset, batch_size=settings.batch_size_test//2, shuffle=False, num_workers=settings.num_workers, pin_memory=False, worker_init_fn=worker_init_fn)


# # Defining the model and training parameters
model = timm.create_model(
    settings.modelname, 
    pretrained=True, 
    num_classes=df_train['label'].unique().shape[0]
)

# Load model from the pth.tar file
model.load_state_dict(torch.load(f"{settings.exports_dir}/{settings.modelname}_{settings.multi_system_config}_best.pth.tar")['state_dict'])

model = model.to('cuda', dtype=torch.float)
model.eval();

# Move the model to the cpu to measure inference time
model = model.to('cpu', dtype=torch.float)
import time

# Measure inference time
start_time = time.time()
for i in range(100):
    model(torch.randn(1, 3, 150, 150))
t_avg_inf = (time.time() - start_time) / 100
print(f"Average inference time: {t_avg_inf:.4f} seconds")

# Create a dataframe to store the results and append the results
# Append these results to the results dataframe
results = {'model': settings.modelname,
            'system': settings.multi_system_config,
            'loss': settings.loss,
            'batchsize': settings.batch_size,
            'epochs': settings.num_epochs,
            'inference_time': t_avg_inf}

df_results = pd.DataFrame(columns=['model', 'system', 'loss', 'batchsize', 'epochs', 'train_time', 'inference_time'])
results_df = pd.DataFrame.from_dict(results, orient='index').T
df_results = pd.concat([df_results, results_df], ignore_index=True)
df_results.to_csv(f"{settings.results_dir}/results_{settings.modelname}.csv", index=False)

y_true, y_pred = get_gt_and_preds(model.to('cuda'), test_dataloader, device='cuda')
df_test['y_pred'] = y_pred
df_test['y_true'] = y_true


# # Results for all systems
# Plot accuracy scores for each txt_label in df_test

def plot_accuracy_scores(df, system):
    df = df[df.system == system]
    for insect_class in df.txt_label.unique():
        acc = accuracy_score(df[df.txt_label == insect_class].y_true, df[df.txt_label == insect_class].y_pred)
        plt.bar(insect_class, acc)
        # Add text on top of each bar
        plt.text(insect_class, acc+0.01, round(acc,2), ha='center')
        plt.title(f"Accuracy scores for {system}")
        plt.ylim(0,1)
        plt.xticks(rotation=90);

def plot_system_insect_counts(df, system):
    df = df[df.system == system]
    df.txt_label.value_counts().plot(kind='bar', title=f'Insect counts for {system}')
    for i, v in enumerate(df.txt_label.value_counts()):
        plt.text(i-0.3, v+10, str(v), color='black')
    plt.xticks(rotation=90);

def plot_confusion_matrix(df, system):
    from sklearn.metrics import confusion_matrix

    df = df[df.system == system]
    cm = confusion_matrix(df.y_true, df.y_pred, normalize='true')
    # cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(10,10))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', xticklabels=oe_class_names, yticklabels=oe_class_names)
    plt.title(f"Confusion matrix for {system}")
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)

# We plot the accuracies and counts in one plot and the confusion matrix in another plot
for system in settings.multi_system_training:
    plt.figure(figsize=(20,5))
    plt.suptitle(f"{system}", fontsize=20, fontweight='bold', y=1.1)
    plt.subplot(1, 2, 1)
    plot_accuracy_scores(df_test, system)
    plt.subplot(1, 2, 2)
    plot_system_insect_counts(df_test, system)
    plt.tight_layout()
    plt.savefig(f"{settings.results_dir}/{system}_{settings.modelname}_accuracies_and_counts.png", dpi=300, bbox_inches='tight')
    plt.figure(figsize=(10,10))
    plot_confusion_matrix(df_test, system)
    plt.savefig(f"{settings.results_dir}/{system}_{settings.modelname}_confusion_matrix.png", dpi=300, bbox_inches='tight')


inverset_insect_labels_map = {v: k for k, v in settings.insect_labels_map[label_map_key].items()}

if len(settings.multi_system_training):
    # Let's save the misclassified images in a new folder (subfolder of the results folder) for each system
    for system in settings.multi_system_training:
        misclassified_folder = f"{settings.results_dir}/{system}_misclassified"
        if not Path(misclassified_folder).exists(): Path(misclassified_folder).mkdir(parents=True, exist_ok=True)
        
        df = df_test[df_test.system == system]
        df.to_csv(f"{settings.results_dir}/{system}_data.csv", index=False)
        df_misclassified = df[df.y_true != df.y_pred]
        df_misclassified.to_csv(f"{settings.results_dir}/{system}_misclassified.csv", index=False)
        for i, row in df_misclassified.iterrows():
            # Image path is in the filename column
            img_path = row['filename']
            # Copy the image to the misclassified folder but rename it with the pattern f"true{true_label}_pred{pred_label}_{filename}"
            # The true label and the pred label is mapped to the text label using the inverse insect_labels_map
            true_label = inverset_insect_labels_map[row['y_true']]
            pred_label = inverset_insect_labels_map[row['y_pred']]
            filename = img_path.split('/')[-1]
            shutil.copy(img_path, f"{misclassified_folder}/TRUE{true_label}_PRED{pred_label}_{filename}")

# We save the accuracies and counts in a csv file
df_accuracies = pd.DataFrame(columns=['system', 'weeks', 'txt_label', 'accuracy'])
df_counts = pd.DataFrame(columns=['system',  'weeks', 'txt_label', 'count'])

for system in settings.multi_system_training:
    df = df_test[df_test.system == system]
    # Do not use concat or append to save the accuracies and counts in the same dataframe
    for insect_class in df.txt_label.unique():
        acc = accuracy_score(df[df.txt_label == insect_class].y_true, df[df.txt_label == insect_class].y_pred)
        # Use loc to append the new row to the dataframe
        # Make sure that the labels are in the same order as in the settings.insect_labels_map
        df_accuracies.loc[len(df_accuracies)] = [system, settings.weeks, insect_class, acc]
        df_counts.loc[len(df_counts)] = [system, settings.weeks, insect_class, df[df.txt_label == insect_class].shape[0]]
        
    
accuracies_csv_fname = f"{settings.exports_dir}/{settings.multi_system_config}_{settings.modelname}_WEEKS{settings.weeks}_accuracies.csv" 
counts_csv_fname = f"{settings.exports_dir}/{settings.multi_system_config}_{settings.modelname}_WEEKS{settings.weeks}_counts.csv"
# If they already exist, make another accuracy2 column and count2 column and append the new accuracies and counts to them
if Path(accuracies_csv_fname).exists():
    # Read the existing accuracies and counts csv files
    df_accuracies_old = pd.read_csv(accuracies_csv_fname)
    df_counts_old = pd.read_csv(counts_csv_fname)
    # Delete the existing accuracies and counts csv files
    Path(accuracies_csv_fname).unlink()
    Path(counts_csv_fname).unlink()
    # Read the new accuracies and counts
    df_accuracies_new = df_accuracies
    df_counts_new = df_counts
    # Merge the old and new accuracies and counts on system, weeks, txt_label
    # We will add a random integer as suffix to the old accuracies and counts columns
    rdm_int = np.random.randint(1000)
    df_accuracies = df_accuracies_old.merge(df_accuracies_new, on=['system', 'weeks', 'txt_label'], how='outer', suffixes=('', f'_{rdm_int}'))
    df_counts = df_counts_old.merge(df_counts_new, on=['system', 'weeks', 'txt_label'], how='outer', suffixes=('', f'_{rdm_int}'))

print(f"df_accuracies shape: {df_accuracies.shape}")
print(f"df_counts shape: {df_counts.shape}")    

df_accuracies.to_csv(accuracies_csv_fname, index=False)
df_counts.to_csv(counts_csv_fname, index=False)

# Let's save a list of all plate names that belong to the test set
test_plate_names = df_test.platename_uniq.unique().tolist()
with open(f"{settings.results_dir}/test_plate_names.txt", 'w') as f:
    for item in test_plate_names:
        f.write("%s\n" % item)