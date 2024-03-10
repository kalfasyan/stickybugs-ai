#!/bin/bash
# Activate the bugai mamba environment
source /home/kalfasyan/miniforge3/envs/bugai/bin/activate
echo "$(date) - Activated the bugai mamba environment"

# All systems
systems=(phoneboxS22Ultra)
# systems=(fuji photobox phoneboxS20FE)

# Run a loop to run the data preparation for all systems
for system in "${systems[@]}"
do
    echo "$(date) - Running data preparation for $system"
    python edit_config_file.py -s $system -b /home/kalfasyan/data/INSECTS/All_sticky_plate_images/created_data
    python 000_data_preparation.py
    echo "$(date) - Finished."
done
