import pandas as pd
import yaml

from settings import Settings
from utils import *

with open("config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

settings = Settings(**config)

def plot_boxplot(accuracies_dict_fname, class_to_visualize, system_to_visualize):
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set(font_scale=1.5, style="whitegrid")
    accuracies_df = pd.read_csv(accuracies_dict_fname)
    # Keep only the columns that start with "accuracy"
    accuracies_df = accuracies_df.filter(regex='^accuracy')
    # Order the columns by the number after accuracy_
    accuracies_df = accuracies_df.reindex(sorted(accuracies_df.columns, key=lambda x: int(x.split("_")[1])), axis=1)
    # Place accuracy_-1 at the end (if it exists)
    if "accuracy_-1" in accuracies_df.columns:
        accuracies_df = accuracies_df[[col for col in accuracies_df.columns if col != "accuracy_-1"] + ["accuracy_-1"]]
    else:
        accuracies_df = accuracies_df[[col for col in accuracies_df.columns if col != "accuracy_-1"]]
    # Now rename the columns to be the only the number after accuracy_ and use "ALL" for accuracy_-1
    accuracies_df.columns = [col.split("_")[1] if col != "accuracy_-1" else "ALL" for col in accuracies_df.columns]

    # Plot the boxplot
    accuracies_df.boxplot()
    plt.title(f"Boxplot of accuracies for {class_to_visualize} on {system_to_visualize}")
    plt.ylabel("Accuracy for class")
    plt.xlabel("Number of first X weeks used")
    sns.despine()
    plt.ylim(-.05, 1.05)

    plt.savefig(f"{settings.exports_dir}/{class_to_visualize}_{system_to_visualize}_accuracies.png", dpi=300, bbox_inches='tight', pad_inches=0)
    # Save a pdf version
    plt.savefig(f"{settings.exports_dir}/{class_to_visualize}_{system_to_visualize}_accuracies.pdf", dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    # Plot the boxplot for the counts
    counts_dict_fname = accuracies_dict_fname.replace("accuracies", "counts")
    counts_df = pd.read_csv(counts_dict_fname)
    # Keep only the columns that start with "count"
    counts_df = counts_df.filter(regex='^count')
    # Order the columns by the number after count_
    counts_df = counts_df.reindex(sorted(counts_df.columns, key=lambda x: int(x.split("_")[1])), axis=1)
    # Place count_-1 at the end
    if "count_-1" in counts_df.columns:
        counts_df = counts_df[[col for col in counts_df.columns if col != "count_-1"] + ["count_-1"]]
    else:
        counts_df = counts_df[[col for col in counts_df.columns if col != "count_-1"]]
    # Now rename the columns to be the only the number after count_ and use "ALL" for count_-1
    counts_df.columns = [col.split("_")[1] if col != "count_-1" else "ALL" for col in counts_df.columns]

    # Plot the boxplot
    counts_df.boxplot()
    plt.title(f"Boxplot of counts for {class_to_visualize} on {system_to_visualize}")
    plt.ylabel("Count of samples")
    plt.xlabel("Number of first X weeks used")
    sns.despine()
    plt.savefig(f"{settings.exports_dir}/{class_to_visualize}_{system_to_visualize}_counts.png", dpi=300, bbox_inches='tight', pad_inches=0)
    # Save a pdf version
    plt.savefig(f"{settings.exports_dir}/{class_to_visualize}_{system_to_visualize}_counts.pdf", dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

def get_accuracies_for_given_run(settings, system_to_visualize="phoneboxS22Ultra", class_to_visualize="wmv"):
    # Let's load the csv files that were saved in the 004_model_results.py script like this:
    accuracies_csv_fname = f"{settings.exports_dir}/{settings.multi_system_config}_{settings.modelname}_WEEKS{settings.weeks}_accuracies.csv" 
    counts_csv_fname = f"{settings.exports_dir}/{settings.multi_system_config}_{settings.modelname}_WEEKS{settings.weeks}_counts.csv"

    accuracies_df = pd.read_csv(accuracies_csv_fname)
    counts_df = pd.read_csv(counts_csv_fname)

    # The accuracies_df dataframe looks like this:
    #    system    weeks  txt_label  accuracy  accuracy_123  accuracy_312  accuracy_532 ...
    #    phonebox  15     wmv        0.95      0.85          0.85          0.95
    #    phonebox  15     sp         0.85      0.95          0.95          0.85
    #    ...

    # We select the row that corresponds to txt_label == class_to_visualize and system == system_to_visualize
    # and we keep only the columns that start with "accuracy" in their name
    accuracies_df = accuracies_df[(accuracies_df.txt_label == class_to_visualize) & (accuracies_df.system == system_to_visualize)].filter(regex='^accuracy')
    # We return a list of accuracies
    accuracies = accuracies_df.values[0]
    
    # Let's do the same for the counts 
    # The counts_df dataframe looks like this:
    #    system    weeks  txt_label  count  count_123  count_312  count_532 ...
    #    phonebox  15     wmv        100    200        300        400
    #    phonebox  15     sp         200    300        400        500
    #    ...
    
    # We select the row that corresponds to txt_label == class_to_visualize and system == system_to_visualize
    # and we keep only the columns that start with "count" in their name
    counts_df = counts_df[(counts_df.txt_label == class_to_visualize) & (counts_df.system == system_to_visualize)].filter(regex='^count')
    # We return a list of counts
    counts = counts_df.values[0]

    return accuracies, counts

# TODO: Make this read from the bash script
class_to_visualize = "wmv"
system_to_visualize = "phoneboxS22Ultra"
accuracies, counts = get_accuracies_for_given_run(settings, 
                                          system_to_visualize=system_to_visualize, 
                                          class_to_visualize=class_to_visualize)

# If the accuracies are more than 5, we will only keep the last 5
nwks =5
if len(accuracies) >= nwks:
    accuracies = accuracies[-nwks:]
if len(counts) >= nwks:
    counts = counts[-nwks:]

# Let's create the dictionary
accuracies_dict = {}
accuracies_dict[settings.weeks] = accuracies
print(f"Weeks: {settings.weeks}, accuracies: {accuracies_dict}")

counts_dict = {}
counts_dict[settings.weeks] = counts
print(f"Weeks: {settings.weeks}, counts: {counts_dict}")

# Let's save the dictionary as a csv file
accuracies_dict_fname = f"{settings.exports_dir}/{settings.multi_system_config}_{settings.modelname}_{class_to_visualize}_{system_to_visualize}_accuracies.csv"
counts_dict_fname = f"{settings.exports_dir}/{settings.multi_system_config}_{settings.modelname}_{class_to_visualize}_{system_to_visualize}_counts.csv"

# We will put all results in a dataframe with header being the settings.weeks and the values being the accuracies
# If the file already exists, we will load it and add the new results as a new column
if Path(accuracies_dict_fname).is_file() and Path(counts_dict_fname).is_file():
    accuracies_df = pd.read_csv(accuracies_dict_fname, index_col=0)
    counts_df = pd.read_csv(counts_dict_fname, index_col=0)
    # Let's check if the accuracy_-1 column exists
    if f"accuracy_{settings.weeks}" in accuracies_df.columns:
        print(f"accuracy_{settings.weeks} already exists in {accuracies_dict_fname}. We will not add it again.")
        plot_boxplot(accuracies_dict_fname, class_to_visualize, system_to_visualize)       
        exit()
    accuracies_df[f"accuracy_{settings.weeks}"] = accuracies
    
    if f"count_{settings.weeks}" in counts_df.columns:
        print(f"count_{settings.weeks} already exists in {counts_dict_fname}. We will not add it again.")
        exit()
    counts_df[f"count_{settings.weeks}"] = counts
else:
    accuracies_df = pd.DataFrame(accuracies_dict)
    counts_df = pd.DataFrame(counts_dict)

# We save the dataframe
accuracies_df.to_csv(accuracies_dict_fname)
counts_df.to_csv(counts_dict_fname)