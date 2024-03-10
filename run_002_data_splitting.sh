#!/bin/bash
# Activate the bugai mamba environment
source /home/kalfasyan/miniforge3/envs/bugai/bin/activate
echo "$(date) - Activated the bugai mamba environment"

systems=(phoneboxS22Ultra)
# fuji photobox phoneboxS20FE phoneboxS22Ultra)

classes_to_remove=(wswl grv other) # (wswl other grv)
weeks=-1 # -1 means all weeks

# First let's edit the config.yaml file using the python script `edit_config_file.py`
# We now care about the list of systems to run the data splitting for
# We use the -ls argument to pass the list of systems to run the data splitting for
# We also set the system (-s) to phoneboxS22Ultra to load its label mappping.
# (the label mapping of phoneboxS22Ultra contains the new class -koolvlieg-)
python edit_config_file.py -ls "${systems[@]}" -s "${systems[0]}" -wks "$weeks" -crm "${classes_to_remove[@]}" 
echo "$(date) - Running Data splitting"
python 002_data_splitting.py
echo "$(date) - Finished data splitting."
