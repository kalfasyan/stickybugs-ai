import os
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import cv2
import numpy as np

basic_df_columns = ['filename', 'label', 'imgname',
                    'platename', 'year', 'location', 'date', 'xtra', 'plate_idx']

def read_image(filename, plot=False):
    """ Read an image from a file. """
    img = Image.open(filename)
    return img

def get_files(directory, ext='.jpg'):
    """ Get all files in a directory. """
    return pd.Series(Path(directory).rglob(f"**/*{ext}"))

def to_weeknr(date=''):
    """
    Transforms a date strings YYYYMMDD to the corresponding week nr (e.g. 20200713 becomes w29)
    """
    week_nr = pd.to_datetime(date).to_pydatetime().isocalendar()[1]
    return f"w{week_nr}"

def extract_filename_info(filename: str, system='fuji') -> str:
    """ Extracts information from a filename. """
    import os
    import yaml
    from settings import Settings

    with open("config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    settings = Settings(**config)

    if not isinstance(filename, str):
        raise TypeError("Provide the filename as a string.")

    path = Path(filename)
    datadir_len = len(Path(os.path.join(settings.data_dir, system)).parts)
    parts = path.parts
    label = parts[datadir_len-1]
    imgname = path.stem

    if system in ['fuji', 'photobox', 'phoneboxS20FE']:
        platename = "_".join(imgname.split('_')[0:-2])
        name_split_parts = imgname.split('_')
        year = name_split_parts[0]
        location = name_split_parts[1]
        date = name_split_parts[2]
        xtra = name_split_parts[3]
        plate_idx = name_split_parts[-1].split('.')[0]
    elif system == 'phoneboxS22Ultra':
        platename = "_".join(imgname.split('_')[0:-3])
        name_split_parts = imgname.split('_')
        year = name_split_parts[0]
        location = name_split_parts[1]
        date = name_split_parts[2]
        # crop = name_split_parts[3]
        xtra = name_split_parts[4]
        plate_idx = name_split_parts[5]
    else:
        raise NotImplementedError(f"System {system} not implemented.")

    return filename, label, imgname[:-4], platename, year, location, date, xtra, plate_idx

def plot_torch_img(x, idx):
    """ Plot a torch image. """
    import matplotlib.pyplot as plt
    plt.imshow(x[idx].permute(1, 2, 0))

def detect_outliers(X_train, algorithm='KNN'):
    """ Detect outliers in the training data. """
    from pyod.models.knn import KNN  # kNN detector
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    if algorithm == 'KNN':
        # train kNN detector
        clf = KNN()
        clf.fit(X_train)

        # get the prediction label and outlier scores of the training data
        # binary labels (0: inliers, 1: outliers)
        return clf.labels_, clf.decision_scores_
    else:
        raise NotImplementedError()

def save_checkpoint_improved(state, save_dir, filename=''):
    """ Save a checkpoint. """
    import torch
    assert isinstance(filename, str), "Filename should be a string."
    assert not filename.endswith('pth.tar'), "Filename should NOT end with pth.tar"

    filename = f'{save_dir}/{filename}_best.pth.tar'
    torch.save(state, filename)

def load_checkpoint(filename, model, optimizer):
    """ Load a checkpoint. """
    import torch
    assert isinstance(filename, str) and filename.endswith(
        'pth.tar'), "Only works with a pth.tar file."
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer

def copy_files(filelist, destination):
    """ Copy files to a destination. """
    from shutil import copy2
    for f in tqdm(filelist, total=len(filelist), desc="Copying files.."):
        copy2(f, destination)

@torch.no_grad()
def get_all_preds(model, loader, dataframe=False, final_nodes=2):
    all_preds = torch.tensor([]).cuda()
    all_labels = torch.tensor([]).cuda()
    all_paths = []
    all_idx = torch.tensor([]).cuda()
    for x_batch, y_batch, path_batch, idx_batch in tqdm(loader):

        preds = model(x_batch.cuda())
        all_preds = torch.cat((all_preds, preds), dim=0)
        all_labels = torch.cat((all_labels, y_batch.cuda()), dim=0)
        all_paths.extend(path_batch)
        all_idx = torch.cat((all_idx, idx_batch.cuda()), dim=0)

    out = all_preds, all_labels, all_paths, all_idx

    if not dataframe:
        return out
    else:
        if final_nodes == 1:
            df_out = pd.DataFrame(out[0], columns=['pred'])
        else:
            df_out = pd.DataFrame(
                out[0], columns=[f'pred{i}' for i in range(final_nodes)])
        df_out['y'] = out[1].cpu()
        df_out['fnames'] = out[2]
        df_out['idx'] = out[3].cpu()
        df_out['softmax'] = torch.argmax(
            F.softmax(out[0], dim=1), dim=1).detach().cpu()
        return df_out

# ----------------------------------------------------------
# ------------ HANDCRAFTED FEATURE CALCULATIONS ------------
# ----------------------------------------------------------
def calc_variance_of_laplacian(image_fname):
    """ Compute the focus measure of the given image using the variance of the Laplacian. """
    ## Credits: Pyimagesearch
    # compute the Laplacian of the image and then return the focus
    # measure, which is simply the variance of the Laplacian
    image = cv2.imread(image_fname, 0)
    return cv2.Laplacian(image, cv2.CV_64F).var()

def calc_mean_RGB_vals(image_fname):
    """ Compute the mean RGB values of an image. """
    image = cv2.imread(image_fname)
    (means, stds) = cv2.meanStdDev(image)
    return np.array([(means[2], means[1], means[0])]).flatten(), \
        np.array([(stds[2], stds[1], stds[0])]).flatten()

def calc_contour_features(image_fname):
    """ Compute the contour features of an image. """
    img = cv2.imread(image_fname)
    img = 255 - img
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # define a thresh
    thresh = 110
    # get threshold image
    ret, thresh_img = cv2.threshold(img_grey, thresh, 255, cv2.THRESH_BINARY)
    # find contours
    contours, hierarchy = cv2.findContours(
        thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # calculate features
    nb_contours = len(contours)
    cnt_areas = [cv2.contourArea(cnt) for cnt in contours]
    cnt_perimeters = [cv2.arcLength(cnt, True) for cnt in contours]
    mean_cnt_area = np.mean(cnt_areas)
    mean_cnt_perimeter = np.mean(cnt_perimeters)
    std_cnt_area = np.std(cnt_areas)
    std_cnt_perimeter = np.std(cnt_perimeters)

    return nb_contours, mean_cnt_area, mean_cnt_perimeter, std_cnt_area, std_cnt_perimeter

def check_if_file_exists(df):
    """	
    This function goes through each row in the dataframe 'df' 
    and checks the filename column if the file exists
    It then adds the index of absent files in a list and returns the list
    """	
    idx_list = []
    for idx, row in df.iterrows():
        if not os.path.isfile(row['filename']):
            idx_list.append(idx)
    return idx_list

def check_if_outlier(df):
    """
    This function goes through each row in the dataframe 'df'
    and checks the filename column if the file exists in the outlier directory
    It then adds the index of absent files in a list and returns the list
    """
    idx = []
    for i, row in df.iterrows():
        outlier_path = row.filename.replace(f"{row.system}_tile_exports", f"{row.system}_tile_exports_outliers")
        outlier_path = Path(outlier_path).parent
        filename = Path(row.filename).name
        full_path = os.path.join(outlier_path, filename)
        if os.path.isfile(full_path):
            idx.append(i)
        else:
            continue
    return idx

def identify_system(string):
    # '/home/u0159868/data/INSECTS/All_sticky_plate_images/created_data/photobox_tile_exports/bl/2021_brainelalleud_w27_a_4056x3040_22634.png'
    # Find one of ['fuji', 'photobox', 'phoneboxS20FE', 'phoneboxS22Ultra']	in the string
    # Return the first one found
    for system in ['fuji', 'photobox', 'phoneboxS20FE', 'phoneboxS22Ultra']:
        if system in string:
            return system

def without_keys(d, keys):
    """Return a new dictionary with the given keys removed."""
    return {x: d[x] for x in d if x not in keys}

def plot_system_insect_counts(df_train, df_val, df_test, system):
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(20,5))
    plt.subplot(1,3,1)
    df_train[df_train.system == system]['txt_label'].value_counts().plot(kind='bar', title='train')
    for i, v in enumerate(df_train[df_train.system == system]['txt_label'].value_counts()):
        plt.text(i-0.3, v+10, str(v), color='black', fontweight='bold')
    plt.subplot(1,3,2)
    df_val[df_val.system == system]['txt_label'].value_counts().plot(kind='bar', title='val')
    for i, v in enumerate(df_val[df_val.system == system]['txt_label'].value_counts()):
        plt.text(i-0.3, v+10, str(v), color='black', fontweight='bold')
    plt.subplot(1,3,3)
    df_test[df_test.system == system]['txt_label'].value_counts().plot(kind='bar', title='test'); sns.despine()
    for i, v in enumerate(df_test[df_test.system == system]['txt_label'].value_counts()):
        plt.text(i-0.3, v+10, str(v), color='black', fontweight='bold')
    plt.suptitle(f"{system} insect counts", fontsize=20, fontweight='bold');

def send_teams_message(message):
    import json

    import requests
    
    webhook_url = "https://kuleuven.webhook.office.com/webhookb2/626b255b-8871-4cec-9dd6-486811dd7dfb@3973589b-9e40-4eb5-800e-b0b6383d1621/IncomingWebhook/124d01d8b14347ed8b9c32bf3d8c2434/c3512653-3fdb-4779-861d-e5c4c07729a2"
    headers = {"Content-Type": "application/json"}
    data = {"text": message}

    response = requests.post(webhook_url, headers=headers, data=json.dumps(data))
    if response.status_code != 200:
        print("Failed to send message to Teams. Status code:", response.status_code)
    else:
        print("Message sent to Teams successfully!")

def train_epoch(model, train_dataloader, criterion, optimizer, disable=False):
    correct_train = 0
    total_loss = 0

    model.train()
    for x_batch, y_batch, _, _, _, _, _, _, _, _, _, _ in tqdm(train_dataloader, 
                                                               desc='Training..\t', 
                                                               total=len(train_dataloader),
                                                               disable=disable):
        y_batch = torch.as_tensor(y_batch)
        x_batch, y_batch = x_batch.float().cuda(), y_batch.cuda()

        optimizer.zero_grad()
        pred = model(x_batch)

        y_batch = y_batch.type(torch.LongTensor).cuda()
        correct_train += (pred.argmax(axis=1) == y_batch).float().sum().item()

        loss = criterion(pred, y_batch)
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    train_accuracy = correct_train / len(train_dataloader.dataset) * 100.0
    avg_loss = total_loss / len(train_dataloader)

    return train_accuracy, avg_loss

def validate_epoch(model, valid_dataloader, criterion, disable=True):
    correct_valid = 0
    total_loss = 0

    model.eval()
    with torch.no_grad():
        for x_batch, y_batch, _, _, _, _, _, _, _, _, _, _ in tqdm(valid_dataloader, 
                                                                   desc='Validating..\t', 
                                                                   total=len(valid_dataloader),
                                                                   disable=disable):
            y_batch = torch.as_tensor(y_batch)
            x_batch, y_batch = x_batch.float().cuda(), y_batch.cuda()

            pred = model(x_batch)

            y_batch = y_batch.type(torch.LongTensor).cuda()
            correct_valid += (pred.argmax(axis=1) == y_batch).float().sum().item()

            loss = criterion(pred, y_batch)
            total_loss += loss.item()

    valid_accuracy = correct_valid / len(valid_dataloader.dataset) * 100.0
    avg_loss = total_loss / len(valid_dataloader)

    return valid_accuracy, avg_loss

def train_test_split_stickyplate(df, test_size=0.2, shuffle=True):
    """	
    Split the data into train and test sets based on the sticky plate.
    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe containing the data
    test_size : float, optional
        The size of the test set, by default 0.2
    shuffle : bool, optional
        Whether to shuffle the data before splitting, by default True
    Returns 
    -------
    df_train : pandas.DataFrame
        The dataframe containing the training data
    df_test : pandas.DataFrame
        The dataframe containing the test data
    """
    # First we split a list of sticky plates into train and test sets
    train_plates, test_plates = train_test_split(df.platename_uniq.unique().tolist(), test_size=test_size, shuffle=shuffle)
    assert not len(set(train_plates).intersection(set(test_plates))), 'Train and test plates overlap!'
    # Assign the train and test sets to the df based on the sticky plate
    df_train = df[df.platename_uniq.isin(train_plates)]
    df_test = df[df.platename_uniq.isin(test_plates)]
    # Reset the index
    df_train.reset_index(drop=True, inplace=True)
    df_test.reset_index(drop=True, inplace=True)
    return df_train, df_test