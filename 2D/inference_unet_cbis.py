# ----------------- Imports -----------------
import os
import glob
import random
import numpy as np
import pandas as pd
import json
import argparse

import monai
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from monai.transforms import (
    LoadImaged, Compose, EnsureTyped, EnsureChannelFirstd,
    Lambdad, AsDiscrete, Activations
)
from monai.data import CacheDataset
from monai.data.utils import decollate_batch
from monai.networks.nets import UNet
from monai.networks.layers import Norm

import seg_metrics.seg_metrics as sg
from sklearn.model_selection import KFold
from transform_utils import transform_lib
from tqdm import tqdm

# ----------------- Reproducibility -----------------
monai.utils.set_determinism(seed=0)
np.random.seed(0)
random.seed(0)

# ----------------- Device Setup -----------------
if torch.cuda.is_available():
    device = torch.device("cuda:0")  # Change GPU index if needed
else:
    device = torch.device("cpu")

# ----------------- Data Loading -----------------
# Load image-segmentation path list
img_seg_path = pd.read_csv(
    "/workspace/nas_data/cbis_ddsm/manifest-ZkhPvrLo5216730872708713142/filtered_image_segmentation.csv", 
    delimiter=";"
)

# Load best model info
best_models_file = pd.read_csv(
    "/workspace/code/results/cbis_best_val_mean_perfold.csv", 
    delimiter=",", 
    encoding='windows-1254'
)

# Load config file for transformation pipelines
conf_file = json.load(open("/workspace/code/config_unet_cbis_raw.json", "r"))

# ----------------- Model Hyperparameters -----------------
unet_norm = Norm.BATCH
lr = 0.0001
channels = (16, 32, 64, 128)
strides = (2, 2, 2)

# ----------------- Data Splitting -----------------
kf = KFold(n_splits=5, shuffle=True, random_state=42)
patient_ids = img_seg_path["patient_id"].unique()

# Prepare validation cases
for fold, (train_idx, val_idx) in enumerate(kf.split(patient_ids)):
    val_ids = patient_ids[val_idx]
    val_cases = []

    for _, row in img_seg_path.iterrows():
        if row["patient_id"] in val_ids:
            val_cases.append({"image": row["img_nii"], "segmentation": row["seg_nii"]})

# ----------------- Model Initialization -----------------
model = UNet(
    spatial_dims=2,
    in_channels=1,
    out_channels=1,
    channels=channels,
    strides=strides,
    num_res_units=2,
    norm=unet_norm
).to(device)

optimizer = Adam(model.parameters(), lr=lr)
post_trans = Compose([
    Activations(sigmoid=True),
    AsDiscrete(threshold=0.5)
])

# ----------------- Inference -----------------
for idx, line in best_models_file.iterrows():
    # Extract run-specific parameters
    run_name = line["run_name"]
    model_path = line["checkpoint_path"]
    fold_nr = int(line["fold"].split("fold")[-1]) - 1
    conf = conf_file[run_name]

    print(f"→ Running {run_name} | Fold {fold_nr + 1} | #Patients: {len(val_cases)} | Model: {os.path.basename(model_path)}")

    # Compose transform pipeline
    transf_pipeline = [
        LoadImaged(keys=["image", "segmentation"]),
        EnsureTyped(keys=["image", "segmentation"], device=device, track_meta=True),
        EnsureChannelFirstd(keys=["image", "segmentation"], channel_dim="no_channel")
    ]

    # Add user-defined transforms from config
    for transf in conf["transforms"]:
        transf_pipeline.append(transform_lib[transf["name"]](**transf["params"]))

    # Final composed transform
    preproc = Compose(transf_pipeline)

    # Cache dataset to speed up inference
    dataset_val = CacheDataset(
        data=val_cases,
        transform=preproc,
        cache_rate=1.0,
        num_workers=5,
        copy_cache=False
    )

    # Create DataLoader
    val_loader = DataLoader(dataset_val, num_workers=0, shuffle=False)

    # Load model from checkpoint and set to eval mode
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint.get("model", checkpoint))
    model.eval()

    # ----------------- Batch-wise Inference -----------------
    csv_file = 'metrics.csv' # To save metrics

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc=f"Fold {fold_nr + 1} Inference")):
            # Get inputs
            img = batch["image"].to(device)
            label = batch["segmentation"].to(device)
            output = model(img)
            # Compute evaluation metrics
            metrics_list = sg.write_metrics(
                labels=[1.], # Exclude background
                gdth_img=label,
                pred_img=output,
                csv_file=csv_file,
                metrics=['dice', 'hd']
            )
            metrics = metrics_list[0]
            
            print(metrics)
          
            # Save prediction and label arrays (optional)
            np.save(
                f"/workspace/code/results/val_preds_cbis_inference/pred_{preds.meta['filename_or_obj'].split('/')[-4].split('_')[-3]}_{run_name}_{model_path.split('/')[-3]}.npy",
                preds.cpu().numpy()
            )
            np.save(
                f"/workspace/code/results/val_preds_cbis_inference/labels_{labels.meta['filename_or_obj'].split('/')[-4].split('_')[-4]}_{run_name}_{model_path.split('/')[-3]}.npy",
                labels.cpu().numpy()
            )
