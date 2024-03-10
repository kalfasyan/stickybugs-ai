#!/bin/bash
set -e
# List of all available systems 
# systems=(canon fuji photobox phoneboxS20FE phoneboxS22Ultra)

# Select which systems to run the model training for
systems=(phoneboxS22Ultra) # ⚠️ NOTE: Make sure you run data splitting for the systems you want to run the model training for

pretrained="False"
pretrained_on=(fuji photobox)
wandb_logging="False"

modelname="mobilenetv3_large_100.miil_in21k_ft_in1k" #"tf_efficientnetv2_m.in21k_ft_in1k" # "tf_efficientnet_b4" # "mobilenetv3_large_100" # "mobilenetv3_large_100.miil_in21k_ft_in1k"
nb_epochs=21
loss="SCE"
batch_size=64

classes_to_remove=(wswl other)

# Activate the bugai mamba environment
source /home/kalfasyan/miniforge3/envs/bugai/bin/activate
echo "$(date) - Activated the bugai mamba environment"
# 9 10 11 13 15 17 19 21 
# Loop over some weeks (13, 15, -1)
for weeks in -1 # ⚠️ Define which weeks you want to run the model training for
do
    printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
    echo "$(date) - Running model training for $weeks weeks"
    printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
    for tt in {1..5} # ⚠️ Define how many trials you want to run
    do
        echo "$(date) - Running trial $tt"
        python edit_config_file.py -ls "${systems[@]}" -s "${systems[0]}" -crm "${classes_to_remove[@]}" -wks "$weeks"
        echo "$(date) - Running Data splitting"
        python 002_data_splitting.py
        echo "$(date) - Finished data splitting."

        # First let's edit the config.yaml file using the python script `edit_config_file.py`
        python edit_config_file.py -m "$modelname" -ls "${systems[@]}" \
            -ne "$nb_epochs" -l "$loss" -bs "$batch_size" -s "${systems[0]}" \
            -pt "$pretrained" -po "${pretrained_on[@]}" -wdb "$wandb_logging" \
            -crm "${classes_to_remove[@]}" -wks "$weeks"

        echo "$(date) - Running model training"
        python 003_model_training.py
        echo "$(date) - Finished model training."

        echo "$(date) - Running model results"
        python 004_model_results.py
        echo "$(date) - Finished model results."
    done
done