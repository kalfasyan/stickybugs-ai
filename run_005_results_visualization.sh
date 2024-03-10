#!/bin/bash
set -e
# List of all available systems 
# systems=(canon fuji photobox phoneboxS20FE phoneboxS22Ultra)

# Select which systems to run the model training for
systems=(phoneboxS22Ultra) 

pretrained="True"
pretrained_on=(fuji photobox)

modelname="mobilenetv3_large_100.miil_in21k_ft_in1k" #"tf_efficientnetv2_m.in21k_ft_in1k" # "tf_efficientnet_b4" # "mobilenetv3_large_100" # "mobilenetv3_large_100.miil_in21k_ft_in1k"
nb_epochs=21
loss="SCE"
batch_size=64

classes_to_remove=(wswl other)

# Activate the bugai mamba environment
source /home/kalfasyan/miniforge3/envs/bugai/bin/activate
echo "$(date) - Activated the bugai mamba environment"

for weeks in 9 10 11 13 15 17 19 21 -1
do
    python edit_config_file.py -m "$modelname" -ls "${systems[@]}" \
                -ne "$nb_epochs" -l "$loss" -bs "$batch_size" -s "${systems[0]}" \
                -pt "$pretrained" -po "${pretrained_on[@]}" -wks "$weeks" 

    # Run the python script 005_results_visualization.py
    echo "$(date) - Running results visualization"
    python 005_results_visualizations.py
    echo "$(date) - Finished results visualization."
    printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
done