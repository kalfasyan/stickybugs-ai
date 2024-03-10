#!/usr/bin/env python
# coding: utf-8
import os

import matplotlib

matplotlib.use('Agg')
import shutil
import sys

import matplotlib.pyplot as plt
import numpy as np
np.random.seed(42)

import pandas as pd
import seaborn as sns
import timm
import torch
torch.manual_seed(42)

import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import yaml
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from torch.optim.lr_scheduler import CyclicLR
from torch.utils.data import DataLoader

torch.backends.cudnn.benchmark = True

from tqdm.auto import tqdm

from datasets import *
from settings import Settings
from utils import *
from utils_test import *


# Helper functions
# \----------------------------------------------------------
def adjust_weight_for_class(weight, class_name, factor):
    """Adjust the weight of a class by multiplying it with a factor"""
    assert class_name in settings.insect_labels_map[label_map_key], f"The class {class_name} is not recognized"
    assert factor > 0, "The factor should be greater than 0"
    if class_name in settings.insect_labels_map[label_map_key]:
        class_idx = settings.insect_labels_map[label_map_key][class_name]
        weight[class_idx] = weight[class_idx] * factor
    return weight
# ----------------------------------------------------------/

with open("config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

settings = Settings(**config)

# Clean past contents of the results dir and create the dir
if Path(settings.results_dir).exists():
    # Delete any subdirectories and files in the results dir
    try:
        for subdir in Path(settings.results_dir).glob("*"):
            # Check if it's a directory
            if subdir.is_dir():
                # Delete the directory
                shutil.rmtree(subdir)
            # If it's a file
            else:
                # Delete the file
                subdir.unlink()
    except Exception as e:
        print(f"Error while deleting past results: {e}")
Path(settings.results_dir).mkdir(exist_ok=True, parents=True)

# Choosing whether to train on a gpu
train_on_gpu = torch.cuda.is_available()
print(f'Train on gpu: {train_on_gpu}, Model: {settings.modelname}')

if len(settings.multi_system_training):
    print(f"Training for systems: {settings.multi_system_training} (config: {settings.multi_system_config}), using model: {settings.modelname}", end=f"\n{'-'*100}\n")

if settings.wandb_log:
    from datetime import datetime

    import wandb
    start_time = datetime.now().strftime(format="%Y/%m/%d-%H")
    wandb.login(relogin=True)
    wandb_name = f"{start_time}-{settings.modelname}-{settings.multi_system_config}"
    print(f"length of wandb_name: {len(wandb_name)}")
    print(100*"-")
    wandb.init(wandb_name)
    wandb.run.name = f"R_{wandb_name}"

# # Loading train,val,test sets
df_train = pd.read_parquet(settings.exports_dir / f"df_train_fixed.parquet")
df_val = pd.read_parquet(settings.exports_dir / f"df_val_fixed.parquet")
df_test = pd.read_parquet(settings.exports_dir / f"df_test_fixed.parquet")

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
    print(f"Classes chosen to remove: {settings.classes_to_remove}")
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
    # Save the new df_train, df_val, df_test to parquet files
    df_train.to_parquet(settings.exports_dir / f"df_train_fixed.parquet")
    df_val.to_parquet(settings.exports_dir / f"df_val_fixed.parquet")
    df_test.to_parquet(settings.exports_dir / f"df_test_fixed.parquet")


# Check if there are any invalid values in the labels of df_train, df_val, df_test
assert df_train['label'].isnull().sum() == 0, f"There are null values in the labels of df_train. Some examples: \n{df_train[df_train['label'].isnull()].head()}"
assert df_val['label'].isnull().sum() == 0, f"There are null values in the labels of df_val. Some examples: \n{df_val[df_val['label'].isnull()].head()}"
assert df_test['label'].isnull().sum() == 0, f"There are null values in the labels of df_test. Some examples: \n{df_test[df_test['label'].isnull()].head()}"
# Check also if there are negative values in the labels of df_train, df_val, df_test
assert df_train['label'].min() >= 0, "There are negative values in the labels of df_train. Did you run data splitting?"
assert df_val['label'].min() >= 0, "There are negative values in the labels of df_val. Did you run data splitting?"
assert df_test['label'].min() >= 0, "There are negative values in the labels of df_test. Did you run data splitting?"
# Check that all labels exist in df_train, df_val, df_test
assert df_train['label'].unique().shape[0] == len(oe_class_names), f"Not all labels exist in df_train. Missing labels: {set(oe_class_names) - set(df_train['txt_label'].unique())}, Extra labels: {set(df_train['txt_label'].unique()) - set(oe_class_names)}"
assert df_val['label'].unique().shape[0] == len(oe_class_names), f"Not all labels exist in df_val. Missing labels: {set(oe_class_names) - set(df_val['txt_label'].unique())}, Extra labels: {set(df_val['txt_label'].unique()) - set(oe_class_names)}"
# Check that all labels exist in df_test
if len(set(oe_class_names) - set(df_test['txt_label'].unique())) == 0 and len(set(df_test['txt_label'].unique()) - set(oe_class_names)) == 0:
    pass
else:
    print(f"⚠️ Not all labels exist in df_test or there are extra labels. \nMissing labels: {set(oe_class_names) - set(df_test['txt_label'].unique())}, Extra labels: {set(df_test['txt_label'].unique()) - set(oe_class_names)}")

# Let's have a look at the insect labels value counts for df_train, df_val, df_test
print(f"Train set insect labels value counts: \n{df_train['txt_label'].value_counts()}", end=f"\n{'-'*100}\n")
print(f"Val set insect labels value counts: \n{df_val['txt_label'].value_counts()}", end=f"\n{'-'*100}\n")
print(f"Test set insect labels value counts: \n{df_test['txt_label'].value_counts()}", end=f"\n{'-'*100}\n")
# Assert that the maximum label is smaller than the number of classes
assert df_train['label'].max() < len(oe_class_names), f"The maximum label in df_train is {df_train['label'].max()} but the number of classes is {len(oe_class_names)}"

# Let's save the settings.insect_labels_map to a file
with open(settings.results_dir / "insect_labels_map.txt", "w") as f:
    f.write(str(settings.insect_labels_map[label_map_key]))

# # Creating Pytorch Datasets and Dataloaders

transforms_list_train = [
    T.ToPILImage(),
    T.Resize(size=(150, 150), antialias=True),
    T.RandomVerticalFlip(p=0.5),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomAutocontrast(p=0.7),
    T.RandomAdjustSharpness(sharpness_factor=1.5, p=0.5),
    T.RandomRotation(degrees=(-25, 25)),
    T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
    # T.RandomPosterize(bits=7, p=0.1),
    T.ToTensor(),
]

transforms_list_test = [
    T.ToPILImage(),
    T.ToTensor(),
]

train_dataset = InsectImgDataset(df=df_train.reset_index(drop=True), transform=T.Compose(transforms_list_train))
valid_dataset = InsectImgDataset(df=df_val.reset_index(drop=True), transform=T.Compose(transforms_list_test))
test_dataset = InsectImgDataset(df=df_test.reset_index(drop=True), transform=T.Compose(transforms_list_test))

train_dataloader = DataLoader(train_dataset, batch_size=settings.batch_size, shuffle=True, num_workers=settings.num_workers, pin_memory=False, worker_init_fn=worker_init_fn)
valid_dataloader = DataLoader(valid_dataset, batch_size=settings.batch_size_val, shuffle=False, num_workers=settings.num_workers, pin_memory=False, worker_init_fn=worker_init_fn)
test_dataloader = DataLoader(test_dataset, batch_size=settings.batch_size_test, shuffle=False, num_workers=settings.num_workers, pin_memory=False, worker_init_fn=worker_init_fn)


# #### Defining the model and training parameters


if settings.pretrained:
    # Ask for user input on the number of classes used for the pretrained model
    # input_classes = input("Enter the number of classes used for the pretrained model: ")
    # assert input_classes.isdigit(), "The input should be a number"
    input_classes = 12

    # Create the backbone network architecture from a pretrained model using timm library
    try:
        model = timm.create_model(settings.modelname, num_classes=int(input_classes))
        model.load_state_dict(torch.load(settings.pretrained_modelpath, map_location=torch.device(settings.device))['state_dict'])
        print(f"Pretrained model loaded from: \n{settings.pretrained_modelpath}")
        print(100*"-")
    except Exception as e:
        print(f"Error while loading the pretrained model: {e}")
        print(f"Pretrained model path: {settings.pretrained_modelpath}")
        print(f"Please make sure that the modelname and the number of classes are correct")
        sys.exit(1) # We stop execution of the script which stops the bash script as well

    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features=in_features, out_features=df_train['label'].unique().shape[0])
    
    if not settings.pretrained_finetune_all:
        # Freeze the weights of the first layers of the model
        for param in model.parameters():
            param.requires_grad = False
        for param in model.classifier.parameters():
            param.requires_grad = True
else:
    model = timm.create_model(settings.modelname, 
                            pretrained=True, 
                            num_classes=df_train['label'].unique().shape[0])


if settings.wandb_log:
    wandb.watch(model)

model = model.to('cuda', dtype=torch.float32)

# Let's calculate the class weights
classes, class_counts  = np.unique(df_train['label'], return_counts=True)
# Since we have thousands of samples for some classes, we will use smoothing
epsilon = 0.1
# Calculate the class weights with smoothing
nb_labels = len(df_train['label'].unique())
# The total number of samples is the sum of the class counts plus the number of classes times the smoothing factor
total_samples = nb_labels + epsilon * len(classes)
# To calculate the class weights we use the following formula:
weight = (1 - epsilon) * total_samples / class_counts
# What this formula does is that it gives more weight to the classes with fewer samples
# and less weight to the classes with more samples
# We can further adjust the weights by multiplying them with a factor.
print("Adjusting the weights for some key classes...")

# TIP: These weight factors could be loaded from a file or dict.
weight = adjust_weight_for_class(weight, 'wmv', 15) if 'wmv' in settings.insect_labels_map[label_map_key] else weight
weight = adjust_weight_for_class(weight, 'wswl', 2) if 'wswl' in settings.insect_labels_map[label_map_key] else weight
weight = adjust_weight_for_class(weight, 'other', 0.1) if 'other' in settings.insect_labels_map[label_map_key] else weight
weight = adjust_weight_for_class(weight, 'not_insect', 0.1) if 'not_insect' in settings.insect_labels_map[label_map_key] else weight
weight = adjust_weight_for_class(weight, 't', 0.1) if 't' in settings.insect_labels_map[label_map_key] else weight
weight = adjust_weight_for_class(weight, 'kv', 10) if 'kv' in settings.insect_labels_map[label_map_key] else weight
weight = adjust_weight_for_class(weight, 'wrl', 10) if 'wrl' in settings.insect_labels_map[label_map_key] else weight

class_weights_df = pd.DataFrame({'class': classes})
class_weights_df['class_name'] = class_weights_df['class'].map({v: k for k, v in settings.insect_labels_map[label_map_key].items()})
class_weights_df.set_index('class', inplace=True)
class_weights_df['count'] = class_counts
class_weights_df['weight'] = weight
print(f"The following dataframe shows the class weights and the counts of the classes: \n{class_weights_df}\n")
print(100*"-")

# Set the loss function
if settings.loss == "huber": criterion = nn.HuberLoss(reduction='mean', delta=1.0)
elif settings.loss == "crossentropy": criterion = nn.CrossEntropyLoss(label_smoothing=.15, weight=torch.Tensor(weight).cuda())
elif settings.loss == 'SCE': 
    from losses import SCELoss
    criterion = SCELoss(alpha=6., beta=1., num_classes=df_train['label'].unique().shape[0])    
else:
    raise ValueError(f"Loss {settings.loss} not recognized. Use one of the following: huber, crossentropy, SCE")

# Defining the optimizer
optimizer = optim.Adam(model.parameters())

# Set up the cyclical learning rate scheduler
cycles = settings.num_epochs // 2  # Half the number of epochs since there are two phases per cycle
step_size_up = len(train_dataloader) * cycles  # Number of update steps per cycle
scheduler = CyclicLR(optimizer, base_lr=0.001, max_lr=0.01, step_size_up=step_size_up, cycle_momentum=False)

# Training
best_valacc = 0
best_valloss = 100
val_losses = []

for epoch in tqdm(range(settings.num_epochs), desc="Total progress", total=settings.num_epochs):
    # Training
    train_accuracy, loss = train_epoch(model, train_dataloader, criterion, optimizer, disable=True)

    # Validation
    valid_accuracy, val_loss = validate_epoch(model, valid_dataloader, criterion, disable=True)
    val_losses.append(val_loss)

    # Learning rate scheduler step
    scheduler.step()

    # Printing results and saving 
    print(f"Epoch {epoch}: train_acc: {train_accuracy:.1f}% loss: {loss:.7f},  val_loss: {val_loss:.7f} val_acc: {valid_accuracy:.1f}%")

    # Check if the validation accuracy improved
    valid_accuracy_improved = valid_accuracy > best_valacc

    # Check if the validation loss improved or if it is in the 10% percentile of the validation losses and the validation accuracy improved
    model_improved = (val_loss < best_valloss) or (val_loss < np.percentile(val_losses, 20) and valid_accuracy_improved)
    if model_improved :
        print(f"Validation loss: {best_valloss:.7f}->{val_loss:.7f}. Validation accuracy: {best_valacc:.1f}->{valid_accuracy:.1f}. Saving model...") 
        best_valloss = val_loss
    if valid_accuracy_improved: best_valacc = valid_accuracy

    # Save the model if the validation loss improved
    results_dict = {'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'best_valacc': best_valacc,
                    'loss': loss,
                    'val_loss': val_loss,
                    'train_accuracy': train_accuracy,
                    'valid_accuracy': valid_accuracy,
                    'optimizer': optimizer.state_dict(),}

    # Whether to save the model and under what name. Note that the save_checkpoint function will still check the model_improved flag
    if len(settings.multi_system_training):
        model_save_name = f"{settings.modelname}_{settings.multi_system_config}"
    else:
        model_save_name = f"{settings.modelname}_{settings.system}"

    if model_improved:
        save_checkpoint_improved(results_dict, settings.exports_dir, model_save_name)

    # Log results to wandb
    if settings.wandb_log or model_improved:
        y_gt, y_pr = get_gt_and_preds(model, test_dataloader, device=settings.device, disable=True)

        # We need to make sure that we use all labels in the test set to calculate the balanced accuracy
        # and to plot the confusion matrix
        # The labels from 0 to len(oe_class_names)-1 are the ones that exist in the training set
        # Any other label is an extra label that does not exist in the training set so we need to map it
        # to new labels starting from len(oe_class_names) and ending at len(oe_class_names) + len(extra_labels)
        # First let's check if it's necessary to map any labels
        if len(set(y_gt) - set(range(len(oe_class_names)))) == 0:
            # No need to map any labels
            pass
        else:
            extra_labels = set(y_gt) - set(range(len(oe_class_names)))
            extra_labels_map = {label: len(oe_class_names) + i for i, label in enumerate(extra_labels)}
            y_gt = [extra_labels_map[label] if label in extra_labels else label for label in y_gt]
            y_pr = [extra_labels_map[label] if label in extra_labels else label for label in y_pr]
            # Let's give them txt names
            extra_labels = [f"extra_{label}" for label in extra_labels]
            oe_class_names = oe_class_names + list(extra_labels)

        plt.figure()
        sns.heatmap(confusion_matrix(y_true=y_gt, y_pred=y_pr, normalize='true'), 
                    annot=True, 
                    fmt='.0%',
                    yticklabels=oe_class_names,
                    xticklabels=oe_class_names,
                    cbar=False,)
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.title(f"Balanced accuracy: {balanced_accuracy_score(y_true=y_gt, y_pred=y_pr):.2f}")
        plt.tight_layout()

        if settings.wandb_log:
            plt.savefig(os.path.join("wandb", f"confmat_{epoch}.jpg"))        
            log_dict = without_keys(results_dict, {'state_dict','optimizer'})
            log_dict["Confusion Matrix (test-set)"] = wandb.Image(os.path.join("wandb", f"confmat_{epoch}.jpg"))
            wandb.log(log_dict)
        plt.savefig(f"{settings.results_dir}/confmat_testset_{epoch}.jpg")
        plt.close()

        # Let's make another confusion matrix where we show the counts
        plt.figure()
        sns.heatmap(confusion_matrix(y_true=y_gt, y_pred=y_pr), 
                    annot=True, 
                    fmt='d',
                    yticklabels=oe_class_names,
                    xticklabels=oe_class_names,
                    cbar=False,)
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.title(f"Balanced accuracy: {balanced_accuracy_score(y_true=y_gt, y_pred=y_pr):.2f}")
        plt.tight_layout()
        plt.savefig(f"{settings.results_dir}/confmat_testset_counts_{epoch}.jpg")
        plt.close()
        
        # Let's also create a classification report and save it
        from sklearn.metrics import classification_report
        report = classification_report(y_true=y_gt, y_pred=y_pr, target_names=oe_class_names, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv(f"{settings.results_dir}/classification_report_{epoch}.csv")


if settings.wandb_log:
    os.system("rm wandb/*confmat*")

# Find any txt_label that is in df_test but not in df_train
for txtlbl in df_test['txt_label'].unique():
    if txtlbl not in df_train['txt_label'].unique():
        print(txtlbl)

df_train.label.value_counts()
