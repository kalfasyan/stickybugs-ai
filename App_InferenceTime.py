"""
**Author**: `Ioannis Kalfas (mailto:ioannis.kalfas@kuleuven.be)

Streamlit App for measuring inference time of timm library models
"""

import streamlit as st
import torch
import timm
from pathlib import Path
import time
import numpy as np
import pandas as pd
from settings import Settings
import yaml
import psutil
from torch.utils.data import DataLoader
from datasets import InsectImgDataset, worker_init_fn
from torchvision import transforms as T
from sklearn.metrics import balanced_accuracy_score
from tqdm.auto import tqdm

# make pytorch faster
torch.backends.cudnn.benchmark = True

def get_nb_params(model):
    """Function to get the number of parameters of a model"""
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    return nb_params

def get_model_metrics(model, device, dataloader, nb_repetitions=10):
    """Function to estimate the inference time of a model"""
    # Put the model on the device
    model = model.to(device)
    # Set the model to eval mode
    model.eval()
    # Create a list to hold the inference times
    inference_times, y_pred, y_true = [], [], []
    # Loop over the dataloader
    for i, sample in tqdm(enumerate(dataloader), total=len(dataloader), desc="Estimating inference time"):
        x = sample[0]
        y = sample[1]
        # Put the data on the device
        x = x.to(device)
        y = torch.as_tensor(y).type(torch.LongTensor).to(device)
        pred = model(x)
        _, preds = torch.max(pred, 1)
        y_pred.extend(preds.detach().cpu().numpy())
        y_true.extend(y.detach().cpu().numpy())

        # Loop over the number of repetitions
        for _ in range(nb_repetitions):
            # Start the timer
            start = time.time()
            # Do a forward pass
            with torch.no_grad():
                _ = model(x)
            # Stop the timer
            end = time.time()
            # Append the inference time to the list
            inference_times.append(end-start)
    # Calculate the mean inference time and the mean inference time per image
    mean_inf_time = np.mean(inference_times) 
    mean_inf_time_im = np.mean(inference_times)/len(dataloader.dataset)
    # Calculate the balanced accuracy
    balanced_acc = balanced_accuracy_score(y_true, y_pred, adjusted=True)
    return mean_inf_time, mean_inf_time_im, balanced_acc

# Available workers
nb_workers = psutil.cpu_count(logical=False)

with open("config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

settings = Settings(**config)

# Load the data
df_test = pd.read_parquet(f"{settings.exports_dir}/df_test_{settings.system}.parquet")
transforms_list_test = [
    T.ToPILImage(),
    # T.Resize(size=(150, 150), antialias=True),
    T.ToTensor(),
] 
test_dataset = InsectImgDataset(df=df_test.reset_index(drop=True), transform=T.Compose(transforms_list_test))
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=settings.num_workers, pin_memory=False, worker_init_fn=worker_init_fn)
im_size = test_dataset[0][0].shape[1]

# Create a form to select the device, system, model, image size, number of images to test, number of repetitions and number of workers
form = st.form(key="my_form")

with form:
    # Select the device using streamlit
    device = st.selectbox("Select device", ["cpu", "cuda"])
    # Check if the device is cuda and if cuda is available
    if device == "cuda" and not torch.cuda.is_available():
        st.error("CUDA is not available. Please select device as CPU")
        st.stop()

    # Select the system using streamlit
    system = st.selectbox("Select system", ["fuji", "photobox", "samsungS20FE", "samsungS22Ultra"], index=1)

    # Get all models from the exports directory
    list_models = [str(x) for x in Path(settings.exports_dir).glob(f"*{system}_best.pth.tar")]

    # Select number of images to test from the test set
    nb_ims = st.slider("Select number of images to test", 10, len(test_dataset), 100, 10)
    
    # Sample the test set
    test_dataset_sample = torch.utils.data.Subset(test_dataset, np.random.choice(len(test_dataset), nb_ims, replace=False))
    # Create a dataloader for the test set
    test_dataloader_sample = DataLoader(test_dataset_sample, batch_size=1, shuffle=False, num_workers=settings.num_workers, pin_memory=False, worker_init_fn=worker_init_fn)

    # Select number of repetitions
    nb_reps = st.slider("Select number of repetitions", 1, 100, 10, 1)

    # Select number of workers
    nb_workers = st.slider("Select number of workers", 0, nb_workers, nb_workers, 1)

    # Create a button to run a selection of models
    model_selection = st.multiselect("Select models", [Path(x).name for x in list_models]) 

    start_inference_test = st.form_submit_button("Start")

    if start_inference_test:
        # Create a dataframe to store the results
        df = pd.DataFrame(columns=["Model", "Average inference time", "Average inference time per image", "Image size", "Number of images", "Number of repetitions"])
        
        # Make a list of the selected models
        list_selected_models = [str(Path(settings.exports_dir) / x) for x in model_selection]
        
        # Loop over all models
        for model_selected in list_selected_models:

            # Strip the _system_best.pth.tar from the model name
            model_timm_name = '_'.join(Path(model_selected).name.split("_")[:-2])
            with st.spinner(f"Estimating inference time for {model_timm_name}..."):
                # Load the model
                model = timm.create_model(model_timm_name, pretrained=False, num_classes=14)
                model.load_state_dict(torch.load(model_selected)["state_dict"])

                # Estimate the average inference time for the selected model and inference time per image
                avg_inference_time, avg_inference_time_per_image, balanced_acc = get_model_metrics(model=model, 
                                                                                                   dataloader=test_dataloader_sample, 
                                                                                                   device=device, 
                                                                                                   nb_repetitions=nb_reps)

                st.markdown(f"**{model_timm_name}** inference time: {avg_inference_time:.4f}s, {avg_inference_time_per_image:.5f}s/image, balanced accuracy: {balanced_acc:.2f}")
                
            # Add the results to the dataframe
            df = df.append({"Model": model_timm_name, 
                            "Image size": im_size, 
                            "Number of images": nb_ims, 
                            "Number of repetitions": nb_reps, 
                            "Average inference time": avg_inference_time, 
                            "Average inference time per image": avg_inference_time_per_image,
                            "Balanced accuracy": balanced_acc,
                            }, ignore_index=True)
        
        # Show the dataframe
        st.header("Results")
        st.dataframe(df)
        
        # Show a bar plot for the average inference time per model
        import plotly.express as px
        fig = px.bar(df, x="Model", y="Average inference time", color="Model", title="Average inference time per model (seconds) - Smaller is better")
        st.plotly_chart(fig, use_container_width=True)
        # Make x-axis labels vertical
        st.markdown("""<style>
        .plotly-graph-div > .plot-container > .svg-container > svg {
            display: block;
        }
        </style>""", unsafe_allow_html=True)
        
        # Show a scatter plot for the average inference time vs balanced accuracy
        fig = px.scatter(df, x="Balanced accuracy", y="Average inference time", color="Model", title="Average inference time vs Balanced Accuracy")
        st.plotly_chart(fig, use_container_width=True)
        # Make x-axis labels vertical
        st.markdown("""<style>
        .plotly-graph-div > .plot-container > .svg-container > svg {
            display: block;
        }
        </style>""", unsafe_allow_html=True)

            