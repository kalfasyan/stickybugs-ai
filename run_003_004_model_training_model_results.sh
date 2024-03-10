#!/bin/bash
set -e
# List of all available systems 
# systems=(canon fuji photobox phoneboxS20FE phoneboxS22Ultra)

# Select which systems to run the model training for
systems=(phoneboxS22Ultra) # ⚠️ NOTE: Make sure you run data splitting for the systems you want to run the model training for

pretrained="True"
pretrained_on=(fuji photobox)
wandb_logging="False"

modelname="mobilenetv3_large_100.miil_in21k_ft_in1k" #"tf_efficientnetv2_m.in21k_ft_in1k" # "tf_efficientnet_b4" # "mobilenetv3_large_100" # "mobilenetv3_large_100.miil_in21k_ft_in1k"
nb_epochs=36
loss="SCE"
batch_size=32

classes_to_remove=(wswl grv other)
weeks=-1 # -1 means all weeks

# Activate the bugai mamba environment
source /home/kalfasyan/miniforge3/envs/bugai/bin/activate
echo "$(date) - Activated the bugai mamba environment"

# First let's edit the config.yaml file using the python script `edit_config_file.py`
python edit_config_file.py \
                        -m "$modelname" \
                        -ls "${systems[@]}" \
                        -ne "$nb_epochs" \
                        -l "$loss" \
                        -bs "$batch_size" \
                        -s "${systems[0]}" \
                        -wdb "$wandb_logging" \
                        -crm "${classes_to_remove[@]}" \
                        -wks "$weeks" \
                        -pt "$pretrained" \
                        -po "${pretrained_on[@]}" \

echo "$(date) - Running model training"
python 003_model_training.py
echo "$(date) - Finished model training."

echo "$(date) - Running model results"
python 004_model_results.py
echo "$(date) - Finished model results."