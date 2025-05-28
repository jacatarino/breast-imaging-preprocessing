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
    Lambdad
)
from monai.data import CacheDataset
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
# Load image and label paths
img_list = sorted(glob.glob("/workspace/duke/images/*"))
seg_list = sorted(glob.glob("/workspace/duke/labels/*"))

# Load best model info
best_models_file = pd.read_csv("/workspace/code/results/duke_best_val_mean_perfold.csv", delimiter=",", encoding='windows-1254')

# Load config file for transformation pipelines
conf_file = json.load(open("/workspace/code/config_unet_duke_raw.json", "r"))

# ----------------- Model Hyperparameters -----------------
unet_norm = Norm.BATCH
lr = 0.001
channels = (16, 32, 64, 128)
strides = (2, 2, 2)

# ----------------- Data Splitting -----------------
data_dict = [{"image": img, "segmentation": seg} for img, seg in zip(img_list, seg_list)]
patient_ids = np.array([x["image"] for x in data_dict])

# 5-fold split using fixed seed
kf = KFold(n_splits=5, shuffle=True, random_state=42)
idx_folds = list(kf.split(patient_ids))

# ----------------- Model Initialization -----------------
model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=4,
    channels=channels,
    strides=strides,
    num_res_units=2,
    norm=unet_norm
).to(device)

optimizer = Adam(model.parameters(), lr=lr)

# ----------------- Inference -----------------
for idx, line in best_models_file.iterrows():
    run_name = line["run_name"]
    model_path = line["checkpoint_path"]
    fold_nr = int(line["fold"].split("fold")[-1]) - 1
    
    val_cases = [data_dict[i] for i in idx_folds[fold_nr][1]]
    conf = conf_file[run_name]

    print(f"→ Running {run_name} | Fold {fold_nr + 1} | #Patients: {len(val_cases)} | Model: {os.path.basename(model_path)}")

    # Compose transform pipeline
    transf_pipeline = [
        LoadImaged(keys=["image", "segmentation"]),
        EnsureTyped(keys=["image", "segmentation"], device=device, track_meta=True),
        EnsureChannelFirstd(keys=["image", "segmentation"]),
        Lambdad(keys=["image", "segmentation"], func=lambda x: x[:, :, :, 25:89])
    ]

    for transf in conf["transforms"]:
        transf_pipeline.append(transform_lib[transf["name"]](**transf["params"]))

    preproc = Compose(transf_pipeline)
    dataset_val = CacheDataset(data=val_cases, transform=preproc, cache_rate=1.0, num_workers=5, copy_cache=False)
    val_loader = DataLoader(dataset_val, num_workers=0, shuffle=False)

    # Load model checkpoint
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint.get("model", checkpoint))  # More flexible load
    model.eval()

    # ----------------- Batch-wise Inference -----------------
    csv_file = 'metrics.csv' # To save metrics
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc=f"Fold {fold_nr + 1} Inference")):
            img = batch["image"].to(device)
            labels = batch["segmentation"].to(device)
            outputs = model(img)
            preds = torch.argmax(outputs, dim=1)

            # Compute evaluation metrics
            metrics_list = sg.write_metrics(
                labels=[1., 2., 3.], # Exclude background
                gdth_img=label,
                pred_img=output,
                csv_file=csv_file,
                metrics=['dice', 'hd']
            )
            metrics = metrics_list[0]
            
            print(metrics)

            # Save predictions and labels as .npy files
            np.save(f"/workspace/code/results/val_preds_duke_inference/pred_{preds.meta['filename_or_obj'].split('.')[0].split('_')[-1]}_{run_name}_{model_path.split('/')[-3]}.npy", preds.cpu().numpy())
            np.save(f"/workspace/code/results/val_preds_duke_inference/labels_{labels.meta['filename_or_obj'].split('.')[0].split('_')[-1]}_{run_name}_{model_path.split('/')[-3]}.npy", labels.cpu().numpy())

