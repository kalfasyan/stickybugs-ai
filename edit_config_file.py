"""
Python script to edit the config.yaml file

Example usage:
1) python edit_config_file.py -s photobox
This will change the "system" section of the config.yaml file to "photobox"
2) python edit_config_file.py -m tf_efficientnet_b4
This will change the "modelname" section of the config.yaml file to "tf_efficientnet_b4"
3) python edit_config_file.py -ne 100 -w True
This will change the "num_epochs" section of the config.yaml file to 100 and the "wandb_log" section to True
"""

import yaml
from pathlib import Path
import sys
import argparse

# Get the path to the config.yaml file
config_file = Path(__file__).parent / "config.yaml"

# Load the config.yaml file
with open(config_file, "r") as f:
    config = yaml.safe_load(f)
    available_systems = config["available_systems"]

# Create the parser
parser = argparse.ArgumentParser(description="Edit the config.yaml file")

# Add the arguments
parser.add_argument("-b", "--base_dir", type=str, help="The base directory to use", default="/home/u0159868/data/INSECTS/All_sticky_plate_images/created_data")
parser.add_argument("-bs", "--batch_size", type=int, help="The batch size to use", default=32)
parser.add_argument("-bsv", "--batch_size_val", type=int, help="The batch size to use for validation", default=64)
parser.add_argument("-bst", "--batch_size_test", type=int, help="The batch size to use for testing", default=64)
parser.add_argument("-ims", "--img_size", type=int, help="The image size to use", default=150)
parser.add_argument("-l", "--loss", type=str, help="The loss function to use", default="SCE")
parser.add_argument("-ls", "--list_systems", nargs="+", type=str, help="List of systems to use", default=[])
parser.add_argument("-ltc", "--loss_thresh_cleaning", type=float, help="The loss threshold to use for cleaning", default=5.)
parser.add_argument("-m", "--modelname", type=str, help="The modelname to use", default="tf_efficientnet_b4")
parser.add_argument("-mc", "--modelname_cleaning", type=str, help="The modelname to use for cleaning", default="mobilenetv3_large_100.miil_in21k_ft_in1k")
parser.add_argument("-ne", "--num_epochs", type=int, help="The number of epochs to use", default=150)
parser.add_argument("-nec", "--num_epochs_cleaning", type=int, help="The number of epochs to use for cleaning", default=10)
parser.add_argument("-nfc", "--num_folds_cleaning", type=int, help="The number of folds to use for cleaning", default=3)
parser.add_argument("-nw", "--num_workers", type=int, help="The number of workers to use", default=-1)
parser.add_argument("-s", "--system", type=str, help="The system to use", default="phoneboxS22Ultra", choices=available_systems)
parser.add_argument("-wdb", "--wandb_log", type=str, help="Whether to use wandb logging", default="False")
parser.add_argument("-pt", "--pretrained", type=str, help="Whether to use pretrained model", default="False")
parser.add_argument("-po", "--pretrained_on", nargs="+", type=str, help="The systems used for pretraining the model", default=[])
parser.add_argument("-pfa", "--pretrained_finetune_all", type=str, help="Whether to finetune all layers of the pretrained model", default="True")
parser.add_argument("-crm", "--classes_to_remove", nargs="+", type=str, help="The classes to remove from the dataset", default=[])
parser.add_argument("-wks", "--weeks", type=int, help="The number of weeks to use", default=-1)

# Parse the arguments
args, unknown = parser.parse_known_args()

# Change the config 
config["base_dir"] = args.base_dir
config["batch_size"] = args.batch_size
config["batch_size_val"] = args.batch_size_val
config["batch_size_test"] = args.batch_size_test
config["img_size"] = args.img_size
config["loss"] = args.loss
config["loss_thresh_cleaning"] = args.loss_thresh_cleaning
config["modelname"] = args.modelname
config["modelname_cleaning"] = args.modelname_cleaning
config["multi_system_training"] = args.list_systems
config["num_epochs"] = args.num_epochs
config["num_epochs_cleaning"] = args.num_epochs_cleaning
config["num_folds_cleaning"] = args.num_folds_cleaning
config["num_workers"] = args.num_workers
config["system"] = args.system
config["wandb_log"] = args.wandb_log
config["pretrained"] = args.pretrained
config["pretrained_on"] = args.pretrained_on
config["pretrained_finetune_all"] = args.pretrained_finetune_all
config["classes_to_remove"] = args.classes_to_remove
config["weeks"] = args.weeks

# Save the changes to the config.yaml file
with open(config_file, "w") as f:
    yaml.dump(config, f)

# Print the new system
print(f"System changed to {args.system} in the config.yaml file")

