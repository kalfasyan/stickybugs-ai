import psutil
from pathlib import Path
from dataclasses import dataclass, field
import torch
@dataclass
class Settings():
    """	
    Class to hold all the settings for the training process. 
    Check config.yaml for the default values.
    Use python script edit_config_file.py to edit the config.yaml file default values
    and script read_config_file.py to read the config.yaml file values
    """
    available_systems: list = field(default_factory=list) 
    system: str = "" # Set the system (e.g. "fuji" or "photobox" etc.)
    multi_system_training: list = field(default_factory=list) # Set the systems to train on (e.g. ["fuji", "photobox"])
    base_dir: str = Path("/home/u0159868/data/INSECTS/All_sticky_plate_images/created_data")
    data_dir: str = base_dir / Path(f"{system}_tile_exports/")
    outlier_dir: str = base_dir / Path(f"{system}_tile_exports_outliers/")
    exports_dir: str = base_dir / "exports/"
    results_dir: str = base_dir / "results/"
    crossvalidate_system: list = field(default_factory=list) # Set which system to perform cv for its data
    modelname: str = "tf_efficientnet_b4" # Models are fetched from the timm library
    modelname_cleaning: str = "mobilenetv3_large_100"
    img_size: int = 150
    num_workers: int = 0
    batch_size: int = 64
    batch_size_val: int = 64
    batch_size_test: int = 64
    num_epochs: int = 150
    num_epochs_cleaning: int = 10
    num_folds_cleaning: int = 3
    loss_thresh_cleaning: float = 2.
    loss: str = "SCE"
    wandb_log: str = "False"
    insect_labels_map: dict = field(default_factory=dict)
    pretrained: str = "False"
    pretrained_on: list = field(default_factory=list)
    pretrained_finetune_all: str = "True"
    device: str = ""
    classes_to_remove: list = field(default_factory=list)
    weeks: int = -1

    def __post_init__(self):
        # Convert boolean strings to boolean
        self.wandb_log = True if self.wandb_log in ["True", "true", "1", "yes", "y"] else False
        self.pretrained = True if self.pretrained in ["True", "true", "1", "yes", "y"] else False
        self.pretrained_finetune_all = True if self.pretrained_finetune_all == "True" else False
        
        if isinstance(self.classes_to_remove, str): self.classes_to_remove = [self.classes_to_remove]
        
        assert self.loss in ["SCE", "CE"], f"loss: {self.loss} is not recognized. Use one of the following: SCE, CE"
        assert self.weeks >= -1 and self.weeks <= 30, f"weeks: {self.weeks} is not recognized. Use a value between -1 and 30"
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Check if the given system is recognized
        assert self.system in ["fuji",
                               "photobox",
                               "phoneboxS20FE",
                               "phoneboxS22Ultra"], \
                                   f"system: {self.system} is not recognized. \
                                       Use one of the following: fuji, photobox, phoneboxS20FE, phoneboxS22Ultra"

        # Create directories for reading and saving data and exporting results
        self.base_dir = Path(self.base_dir)
        self.data_dir = Path(self.base_dir / f"{self.system}_tile_exports/")
        Path(self.data_dir).mkdir(exist_ok=True, parents=True)
        self.outlier_dir = Path(self.base_dir / f"{self.system}_tile_exports_outliers/")
        Path(self.outlier_dir).mkdir(exist_ok=True, parents=True)
        self.exports_dir = Path(self.base_dir / "exports/")
        Path(self.exports_dir).mkdir(exist_ok=True, parents=True)
        self.results_dir = Path(self.base_dir / "results/")
        Path(self.results_dir).mkdir(exist_ok=True, parents=True)

        # Select systems for multi-system training
        self.multi_system_training = [self.system] if len(self.multi_system_training) == 0 else self.multi_system_training
        self.multi_system_codes = {"canon": "0", "fuji": "1", "photobox": "2", "phoneboxS20FE": "3", "phoneboxS22Ultra": "4"}
        if len(self.multi_system_training):
            current_system_code = [self.multi_system_codes[system] for system in self.multi_system_training]
            current_system_code.sort()
            self.multi_system_config = "multi" + "".join(current_system_code)
        else:
            self.multi_system_config = ""

        # Rename the results dir to include the system name
        if len(self.multi_system_training):
 
            if self.pretrained: 

                if not len(self.pretrained_on): 
                    print("⚠️ pretrained is True, but no pretraining systems specified ⚠️")

                self.pretrained_system_config = "multi" + "".join([self.multi_system_codes[system] for system in self.pretrained_on])
                # example pretrained_system_config: "multi014"
                self.config_PTconfig = self.multi_system_config + "_PT" + self.pretrained_system_config
                # example config_PTconfig: "multi014_PTmulti014"
                self.pretrained_modelpath = Path(self.exports_dir / f"{self.modelname}_{self.multi_system_config}_best.pth.tar")
                # example: /home/u0159868/data/INSECTS/All_sticky_plate_images/created_data/exports/tf_efficientnet_b4_multi014_PTmulti014_best.pth.tar

                self.results_dir = Path(self.results_dir / f"{self.config_PTconfig}/{self.modelname}/")
                # example: /home/u0159868/data/INSECTS/All_sticky_plate_images/created_data/results/multi014_PTmulti014/tf_efficientnet_b4/
            else:
                self.results_dir = Path(self.results_dir / f"{self.multi_system_config}/{self.modelname}/")
                # example: /home/u0159868/data/INSECTS/All_sticky_plate_images/created_data/results/multi014/tf_efficientnet_b4/
        else: 
            self.results_dir = Path(self.results_dir / f"{self.system}/{self.modelname}/")
            # example: /home/u0159868/data/INSECTS/All_sticky_plate_images/created_data/results/fuji/tf_efficientnet_b4/

        # Set the number of workers
        self.num_workers = psutil.cpu_count(logical=False) if self.num_workers == -1 else self.num_workers

        # Set the image size
        self.img_size = self.img_size if self.img_size > 0 else 150

        # Set the labels for the different systems
        self.insect_labels_map = {}
        self.insect_labels_map['fuji'] ={
            'bl': 0,
            'wswl': 1,
            'sp': 2,
            't': 3,
            'sw': 4,
            'k': 5,
            'm': 6,
            'c': 7,
            'v': 8,
            'wmv': 9,
            'wrl': 10,
            # 'other': 11,
            'not_insect': 11
        }

        self.insect_labels_map['photobox'] = {
            'bl': 0,
            'wswl': 1,
            'sp': 2,
            't': 3,
            'sw': 4,
            'k': 5,
            'm': 6,
            'c': 7,
            'v': 8,
            'wmv': 9,
            'grv': 10,
            'wrl': 11,
            # 'other': 12,
            'not_insect': 12
        }

        self.insect_labels_map['phoneboxS20FE'] = {
            'bl': 0,
            'wswl': 1,
            'sp': 2,
            't': 3,
            'sw': 4,
            'k': 5,
            'm': 6,
            'c': 7,
            'v': 8,
            'wmv': 9,
            'grv': 10,
            'other': 11,
            'not_insect': 12
        }

        self.insect_labels_map['phoneboxS22Ultra'] = {
            'bl': 0,
            'wswl': 1,
            'sp': 2,
            't': 3,
            'sw': 4,
            'k': 5,
            'm': 6,
            'c': 7,
            'v': 8,
            'kv': 9,
            'wmv': 10,
            # 'grv': 11,
            'wrl': 11,
            # 'other': 13,
            'not_insect': 12
        }

        # If we want to train on multiple systems, we need to make sure that the labels are the same
        # for all systems. We do that by concatenating the labels of all systems in self.multi_system_training.
        # If the labels are not the same, we raise an error.
        if len(self.multi_system_training) > 1:
            # The labels are in the keys of the insect_labels_map dict. The values we can recreate as a range from 
            # 0 to the number of classes in all systems in multi_system_training.
            labels = []
            for system in self.multi_system_training:
                labels.extend(list(self.insect_labels_map[system].keys()))
            labels = list(set(labels))
            # We short the labels based on the last character of the label name. This is because the last character
            # typically shows the insect family (e.g. v and wmv are both vliegen/flies)
            labels.sort(key=lambda x: x[-1])
            labels = {label: idx for idx, label in enumerate(labels)}
            self.insect_labels_map[self.multi_system_config] = labels
