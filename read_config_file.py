"""
Python script to read the config.yaml file

Example usage:
1) python read_config_file.py -r base_dir
This will read the "base_dir" section of the config.yaml file
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

# Create the parser
parser = argparse.ArgumentParser(description="Read the config.yaml file")

# Add the arguments
parser.add_argument("-r", "--read", type=str, help="The section to read", default="system", choices=config.keys())

# Parse the arguments
args = parser.parse_args()

# Print the section
print(config[args.read])

