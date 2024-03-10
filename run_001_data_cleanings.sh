#!/bin/bash
# Activate the bugai mamba environment
source /home/kalfasyan/miniforge3/envs/bugai/bin/activate
echo "$(date) - Activated the bugai mamba environment"

# All systems
# systems=(fuji photobox phoneboxS22Ultra)
systems=(phoneboxS22Ultra)

# Print systems to run the data cleaning for
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
echo "$(date) - Running data cleaning for the following systems:"
for stm in "${systems[@]}"
do
    echo "$stm"
done
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -

# First read the "base_dir" variable from config.yaml
base_dir=$(python read_config_file.py -r base_dir)

# Add user input to verify that they want to delete the folders
read -p "Are you sure you want to delete the folders in $base_dir that end in '_tile_exports_outliers'? (y/n) " -n 1 -r
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo ""
else
    echo "Exiting..."
    exit 1
fi

# Find and delete folders in base_dir that end in the system name plus "_tile_exports_outliers" 
echo "$(date) - Find and delete folders that end in '_tile_exports_outliers' in the base directory"
for stm in "${systems[@]}"
do
    stm_tile_exports_outliers="${stm}_tile_exports_outliers"
    echo "Deleting $stm_tile_exports_outliers"
    find "$base_dir" -type d -name "$stm_tile_exports_outliers" -print -exec rm -rf {} \;
    printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
done

# ⚠️ IMPORTANT NOTE: Whatever you don't specifically pass in edit_config_file, will be set as the default value (see edit_config_file.py)

# Run a loop to run the data preparation for all systems
for system in "${systems[@]}"
do
    printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
    echo "$(date) - Running data cleaning for $system"

    # if the system is fuji or photobox, we change the "num_folds_cleaning" variable to 5 
    # and set the "num_epochs_cleaning" variable to 35
    # if the system is phoneboxS20FE or phoneboxS22Ultra, we change the "num_folds_cleaning" variable to 5
    # and set the "num_epochs_cleaning" variable to 50
    if [ "$system" == "fuji" ] || [ "$system" == "photobox" ]
    then
        python edit_config_file.py -nfc 3 -nec 16 -s "$system" -mc "mobilenetv3_large_100.miil_in21k_ft_in1k" -ltc 5.0 
    elif [ "$system" == "phoneboxS20FE" ] || [ "$system" == "phoneboxS22Ultra" ] 
    then
        python edit_config_file.py -nfc 5 -nec 26 -s "$system" -mc "mobilenetv3_large_100.miil_in21k_ft_in1k" -ltc 5.0 
    fi
    python 001_data_cleaning.py
    echo "$(date) - Finished."
    printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
done
