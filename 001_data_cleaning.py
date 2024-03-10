#!/usr/bin/env python
# coding: utf-8

# # Script content
# 
# In this script, we will clean the data and prepare it for modeling using CNNs in the next script (003_model_training). To do this we will:
# - Split the data into a number of folds (defined in the config file)
#     - The folds are as balanced as possible in terms of the number of total images
#     - One of the folds serves as validation data, the rest as training data
# - We train a model for each split and in the end, calculate the individual model loss for each validation image
# - Given a threshold (defined in the config file), we copy the images with a loss above the threshold to a separate folder (outlier_dir in the config file)

# Import libraries
import yaml
from settings import Settings
import timm
import torch
import pandas as pd
from datasets import InsectImgDataset
from torch.utils.data import DataLoader
from torchvision import transforms as T
from datasets import worker_init_fn
from tqdm.auto import tqdm
import numpy as np
from category_encoders import OrdinalEncoder
import shutil
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import sys

with open("config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

settings = Settings(**config)

print("Current environment:", sys.prefix)
print(f"System: {settings.system}")

# Read and prepare data
df = pd.read_parquet(f'{settings.exports_dir}/df_preparation_{settings.system}.parquet');
df.reset_index(drop=True, inplace=True)
df.label = df.label.astype(str)
# Making sure that the labels are encoded as intended
oe = OrdinalEncoder(cols=['label'], mapping=[{'col': 'label', 'mapping': settings.insect_labels_map[settings.system]}])
oe_class_names = list(settings.insect_labels_map[settings.system].keys())
tmp = oe.fit_transform(df['label'])
df.rename(columns={"label": "txt_label"}, inplace=True)
df['label'] = tmp.copy()

if settings.system == "fuji":
    plt.figure()
    df.txt_label.value_counts().plot(kind='bar', figsize=(10, 5), title='Number of images per class');
    print(f"The not_insect class has {df[df.txt_label=='not_insect'].shape[0]} images so we will downsample it to match the number of images in the second smallest largest class")
    # Downsample the not_insect class to match the number of images in the second smallest largest class
    second_largest_class = df.txt_label.value_counts().index[1]
    number_of_images_in_second_largest_class = df[df.txt_label==second_largest_class].shape[0]
    not_insect = df[df.txt_label=='not_insect'].sample(number_of_images_in_second_largest_class, random_state=42)
    df = df[df.txt_label!='not_insect']
    df = pd.concat([df, not_insect])
    plt.figure()
    df.txt_label.value_counts().plot(kind='bar', figsize=(10, 5), title='Number of images per class');
    sns.despine()


# Define number of epochs to train for
epochs = settings.num_epochs_cleaning
# Define number of folds to split data into
N = settings.num_folds_cleaning
# Loss threshold for outlier detection
loss_threshold = settings.loss_thresh_cleaning

# Split the df into N folds
df['fold'] = 0
for i in range(N):
    df.loc[df.index % N == i, 'fold'] = i

# Assert that the folds are balanced in terms of number of images 
for i in range(N):
    print(f'Fold {i}: {len(df[df.fold==i])} images')
assert len(df[df.fold==0].index.intersection(df[df.fold==1].index)) == 0

# Set up GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using {device} to train the model: {settings.modelname_cleaning}')

# In a loop, create a train and validation set for each fold
# where each train set contains all images except those in the validation set
fold_valid_losses = {}
for i in tqdm(range(N), 'Folds'):
    print(f'Fold {i}')
    df_train = df[df.fold != i]
    df_val = df[df.fold == i]
    print(f"Train set: {len(df_train)} images, Validation set: {len(df_val)} images")

    # Create torch datasets and dataloaders for train and validation sets
    # Define transforms
    transforms_list_train = [
        T.ToPILImage(),
        T.Resize(size=(settings.img_size, settings.img_size)),
        T.RandomVerticalFlip(p=0.5),
        T.RandomHorizontalFlip(p=0.5),
        # T.RandomAutocontrast(p=0.5),
        # T.RandomAdjustSharpness(sharpness_factor=1.5, p=0.5),
        # T.RandomRotation(degrees=(-5, 5)),
        # T.RandomPosterize(bits=7, p=0.1),
        T.ToTensor()]
    transforms_list_val = [
        T.ToPILImage(),
        T.Resize(size=(settings.img_size, settings.img_size)),
        T.ToTensor()]


    # Create datasets
    train_dataset = InsectImgDataset(df=df_train.reset_index(drop=True), transform=T.Compose(transforms_list_train))
    valid_dataset = InsectImgDataset(df=df_val.reset_index(drop=True), transform=T.Compose(transforms_list_val))
    
    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=settings.batch_size, shuffle=True, num_workers=settings.num_workers, pin_memory=False, worker_init_fn=worker_init_fn)
    valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=settings.num_workers, pin_memory=False, worker_init_fn=worker_init_fn)


    torch.backends.cudnn.benchmark = True

    # Define model
    model = timm.create_model(settings.modelname_cleaning, pretrained=True, num_classes=df['label'].nunique())
    model.to(device)

    # Define loss function
    from losses import SCELoss
    criterion = SCELoss(alpha=6., beta=1., num_classes=df_train['label'].unique().shape[0])
    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters())
    # Set up the cyclical learning rate scheduler
    from torch.optim.lr_scheduler import CyclicLR
    cycles = settings.num_epochs // 2  # Half the number of epochs since there are two phases per cycle
    step_size_up = len(train_dataloader) * cycles  # Number of update steps per cycle
    scheduler = CyclicLR(optimizer, base_lr=0.001, max_lr=0.01, step_size_up=step_size_up, cycle_momentum=False)
        
    # Model training
    best_valacc = 0
    best_valloss = 100
    val_losses = np.zeros((len(valid_dataset),))
    
    for epoch in tqdm(range(epochs), "Total Progress"):
        # Train model
        model.train()
        
        train_loss = 0
        train_correct = 0
        for x_batch, y_batch, imgname, platename, filename, plate_idx, location, date, year, xtra, width, height in train_dataloader:
            y_batch = torch.as_tensor(y_batch)
            x_batch, y_batch = x_batch.float().to(device), y_batch.long().to(device)
            for param in model.parameters():
                param.grad = None
            pred = model(x_batch)
            
            y_batch = y_batch.type(torch.LongTensor).to(device)
            train_correct += (pred.argmax(dim=1) == y_batch).sum().item()
            
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
        train_accuracy = train_correct / len(train_dataset)*100.
        torch.cuda.empty_cache()

        scheduler.step()

        # Print progress
        print(f'Epoch {epoch+1}/{epochs}: Train loss: {loss.item():.4f}, Train accuracy: {train_accuracy:.2f}%')

    # Going through validation set once in the end of all epochs
    correct_valid = 0
    model.eval()
    with torch.no_grad():
        for jj, (x_batch, y_batch, imgname, platename, filename, plate_idx, location, date, year, xtra, width, height) in enumerate(valid_dataloader):
            y_batch = torch.as_tensor(y_batch)
            x_batch, y_batch = x_batch.float().to(device), y_batch.long().to(device)
            pred = model(x_batch)
            y_batch = y_batch.type(torch.LongTensor).to(device)
            correct_valid += (pred.argmax(dim=1) == y_batch).sum().item()
            # Get the loss of each image in the batch


            # Get the loss of the whole batch                    
            val_loss = criterion(pred, y_batch)
            # Save the val_loss.item() (after you move it to cpu) in the val_losses array
            val_losses[jj] = val_loss.item()
    df_val['val_loss'] = val_losses
    
    #  <> <> <> <> <> <> <>
    #  | Handle outliers  |
    #  <> <> <> <> <> <> <>
    # Define outliers as images with validation loss above a certain threshold
    df_val_outliers = df_val[df_val.val_loss > loss_threshold]
    outliers = df_val_outliers['filename'].tolist()
    print(f'Number of outliers: {len(outliers)}')

    # Copy a list of files (full path) found in outliers variable, in a folder defined in settings.outlier_dir
    for file in outliers:
        dest = settings.outlier_dir / Path(file).parent.stem
        Path(dest).mkdir(parents=True, exist_ok=True)
        shutil.copy(file, dest)

    # Save the fold's validation losses
    fold_valid_losses[i] = val_losses


# If the time is after 7pm then send a Teams message to say the outliers have been detected
from datetime import datetime

if datetime.now().hour >= 17 and datetime.now().hour <= 8:
    from utils import send_teams_message
    send_teams_message(f"Outliers have been detected for {settings.system}")

