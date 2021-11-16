import pandas as pd
import numpy as np
from pathlib import Path
from configparser import ConfigParser
from PIL import Image
from tqdm import tqdm

cfg = ConfigParser()
cfg.read('config.ini')

working_at = 'home'
DATA_DIR = Path(cfg.get(working_at, 'data_dir'))
REPO_DIR = Path(cfg.get(working_at, 'repo_dir'))
SAVE_DIR = Path(cfg.get(working_at, 'save_dir'))


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
    "arc": "arc",
    "kortemark": "kortemark",
    "pecq": "pecq",
    "framez": "frasnes",
    "braneall": "brainelalleud",
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

basic_df_columns = ['filename', 'label','imgname','platename','year','location','date','xtra','plate_idx']

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
    label = parts[datadir_len]
    imgname = parts[datadir_len+1]

    if setting == 'fuji' or setting == 'photobox':

        if setting=='photobox':
            platename = "_".join(imgname.split('_')[1:-2])
        else:
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
        if date[0].lower() in ['a','b','c','d','e','f','g']:
            date, xtra = xtra.lower(), date.lower()
        if xtra.lower().startswith('w'):
            xtra, date = date.lower(), date.lower()
    else:
        raise ValueError()


    return filename, label, imgname[:-4], platename, year, format_location(location), format_date(date), xtra, plate_idx[:-4]
    
def calc_variance_of_laplacian(image_fname):
    ## Credits: Pyimagesearch
    import cv2
    # compute the Laplacian of the image and then return the focus
    # measure, which is simply the variance of the Laplacian
    image = cv2.imread(image_fname,0)
    return cv2.Laplacian(image, cv2.CV_64F).var()

def calc_mean_RGB_vals(image_fname):
    import cv2
    image = cv2.imread(image_fname)
    (means, stds) = cv2.meanStdDev(image)
    return np.array([(means[2], means[1], means[0])]).flatten(), \
            np.array([(stds[2], stds[1], stds[0])]).flatten()

def calc_contour_features(image_fname):
    import cv2

    img = cv2.imread(image_fname)
    img = 255 - img
    img_grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # define a thresh
    thresh = 110
    # get threshold image
    ret,thresh_img = cv2.threshold(img_grey, thresh, 255, cv2.THRESH_BINARY)
    # find contours
    contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # calculate features
    nb_contours = len(contours)
    cnt_areas = [cv2.contourArea(cnt) for cnt in contours]
    cnt_perimeters = [cv2.arcLength(cnt,True) for cnt in contours]
    mean_cnt_area  = np.mean(cnt_areas)
    mean_cnt_perimeter = np.mean(cnt_perimeters)
    std_cnt_area = np.std(cnt_areas)
    std_cnt_perimeter = np.std(cnt_perimeters)
    
    return nb_contours, mean_cnt_area, mean_cnt_perimeter, std_cnt_area, std_cnt_perimeter

def plot_torch_img(x, idx):
    import matplotlib.pyplot as plt
    plt.imshow(x[idx].permute(1,2,0))

# def copy_list_of_files(files):

def detect_outliers(X_train, algorithm='KNN'):
    from pyod.models.knn import KNN   # kNN detector
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    if algorithm == 'KNN':
        # train kNN detector
        clf = KNN()
        clf.fit(X_train)

        # get the prediction label and outlier scores of the training data
        return clf.labels_, clf.decision_scores_  # binary labels (0: inliers, 1: outliers)
    else:
        raise NotImplementedError()

def worker_init_fn(worker_id):                                                          
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def model_selector(modelname, pretrained=False):
    if modelname == 'densenet121':
        from torchvision.models import densenet121
        return densenet121(pretrained=pretrained)
    else: 
        raise ValueError("No model returned")

def save_checkpoint(state, is_best, filename=''):
    import torch
    from shutil import copyfile
    filename = f'{SAVE_DIR}/{filename}.pth.tar'
    torch.save(state, filename)
    if is_best:
        copyfile(filename, f"{filename.split('.')[0]}_best.pth.tar")

def load_checkpoint(filename, model, optimizer):
    import torch
    assert isinstance(filename, str) and filename.endswith('pth.tar'), "Only works with a pth.tar file."
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer

def copy_files(filelist, destination):
    from shutil import copy2
    for f in tqdm(filelist, total=len(filelist), desc="Copying files.."):
        copy2(f, destination)