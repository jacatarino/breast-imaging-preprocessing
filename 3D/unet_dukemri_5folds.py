# ---------------------------- IMPORTS ----------------------------
import os
import glob
import random
import argparse
import numpy as np
import pandas as pd
import json
from matplotlib import pyplot as plt

import monai
import wandb
import torch
from torchvision.utils import save_image

import SimpleITK as itk
import nibabel as nib

from sklearn.model_selection import train_test_split, KFold

from monai.visualize import blend_images
from monai.networks.layers import Norm
from monai.data.utils import decollate_batch
from monai.transforms import (
    LoadImage, LoadImaged, Compose, AsDiscrete, Activations, 
    EnsureTyped, EnsureChannelFirstd, Lambdad
)
from monai.data import MetaTensor, DataLoader, ImageDataset, CacheDataset

from monai.networks.nets import UNet
from monai.losses import DiceLoss
from torch.optim import Adam
from monai.metrics import DiceMetric
from monai.handlers import LrScheduleHandler
from monai.config import print_config

from transform_utils import transform_lib  # Custom transform library

# ------------------------ CONFIGURATION --------------------------
print_config()

# For reproducibility
monai.utils.set_determinism(seed=0)
np.random.seed(0)
random.seed(0)

# ----------------------------- MAIN ------------------------------
def main(args):
    # -------------------- DEVICE CONFIGURATION --------------------
    gpu = args.gpu
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    print("I'll use:", device)

    transf_pip = args.transf_pip

    # --------------------- DATASET PREPARATION --------------------
    img_list = sorted(glob.glob("/workspace/yasna_duke/images/*"))
    seg_list = sorted(glob.glob("/workspace/yasna_duke/three_ labels/*"))

    conf = json.load(open("/workspace/code/config_unet_duke.json", "r"))[transf_pip]

    # --------------------- HYPERPARAMETERS ------------------------
    wandb_proj = "UNet-Seg_Duke_Preprocess"
    dataset = "DukeMRI"
    transf_parameters = conf["label"]
    unet_norm = Norm.BATCH
    batch_size = 4
    lr = 0.001
    mode = "max"
    factor = 0.7
    patience = 100
    threshold = 0.001
    max_epochs = 600
    channels = (16, 32, 64, 128)
    strides = (2, 2, 2)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    data_dict = [{"image": i, "segmentation": s} for i, s in zip(img_list, seg_list)]
    patient_ids = np.array([x["image"] for x in data_dict])

    # --------------------- 5-FOLD CROSS-VALID ---------------------
    for fold, (train_idx, val_idx) in enumerate(kf.split(patient_ids)):
        print(f"Starting Fold {fold + 1}/5")
        train_cases = [data_dict[i] for i in train_idx]
        val_cases = [data_dict[i] for i in val_idx]

        # ---------------------- TRANSFORM PIPELINE ------------------
        transf_pipeline = [
            LoadImaged(keys=["image", "segmentation"]),
            EnsureTyped(keys=["image", "segmentation"], device=device, track_meta=True),
            EnsureChannelFirstd(keys=["image", "segmentation"], channel_dim="no_channel"),
            Lambdad(keys=["image", "segmentation"], func=lambda x: x[:, :, :, 25:89])
        ]
        transf_pipeline += [transform_lib[t["name"]](**t["params"]) for t in conf["transforms"]]
        preproc = Compose(transf_pipeline)

        dataset_train = CacheDataset(train_cases, preproc, cache_rate=1.0, num_workers=8, copy_cache=False)
        dataset_val = CacheDataset(val_cases, preproc, cache_rate=1.0, num_workers=5, copy_cache=False)

        train_loader = DataLoader(dataset_train, num_workers=0, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(dataset_val, num_workers=0, shuffle=False)

        # ------------------------- MODEL ----------------------------
        model = UNet(spatial_dims=3, in_channels=1, out_channels=4, channels=channels, strides=strides, num_res_units=2, norm=unet_norm).to(device)
        loss_function = DiceLoss(to_onehot_y=True, softmax=True)
        dice_metric = DiceMetric(include_background=True, reduction="none", get_not_nans=False, num_classes=4)
        optimizer = Adam(model.parameters(), lr=lr)
        lr_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=mode, factor=factor, patience=patience, threshold=threshold)
        post_trans = AsDiscrete(argmax=True, to_onehot=4)

        # ------------------------- WANDB ----------------------------
        run = wandb.init(
            project=wandb_proj,
            name=transf_pip,
            config={
                "dataset": dataset,
                "architecture": "CNN",
                "transforms": transf_parameters,
                "unet_norm": unet_norm,
                "batch_size": batch_size,
                "scheduler": "ReduceOnPlateau",
                "initial_lr": lr,
                "lr_mode": mode,
                "lr_factor": factor,
                "lr_patience": patience,
                "lr_threshold": threshold,
                "channels": channels,
                "strides": strides,
                "epochs": max_epochs,
                "fold": f"{transf_pip}_fold{fold + 1}"
            }
        )

        # ----------------------- TRAINING LOOP ----------------------
        epoch_loss_values, train_metric_values, val_metric_values = [], [], []
        train_bg_values, train_b_values, train_v_values, train_fgt_values = [], [], [], []
        val_bg_values, val_b_values, val_v_values, val_fgt_values = [], [], [], []

        for epoch in range(max_epochs):
            print(f"Fold {fold + 1}, Epoch {epoch + 1}/{max_epochs}")

            # Training step
            model.train()
            epoch_loss, step = 0, 0
            for batch_data in train_loader:
                step += 1
                img, seg = batch_data["image"], batch_data["segmentation"]
                optimizer.zero_grad()
                pred = model(img)
                loss = loss_function(pred, seg)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                train_outputs = [post_trans(i) for i in decollate_batch(pred)]
                dice_metric(y_pred=train_outputs, y=seg)

            epoch_loss /= step
            epoch_loss_values.append(epoch_loss)

            train_total = np.mean(dice_metric.aggregate("mean").item())
            train_bg, train_b, train_v, train_fgt = dice_metric.aggregate("mean_batch")
            train_metric_values.append(train_total)
            train_bg_values.append(train_bg.item())
            train_b_values.append(train_b.item())
            train_v_values.append(train_v.item())
            train_fgt_values.append(train_fgt.item())
            dice_metric.reset()

            # Validation step
            model.eval()
            val_step = 0
            for val_batch_data in val_loader:
                val_step += 1
                val_img, val_seg = val_batch_data["image"], val_batch_data["segmentation"]
                with torch.no_grad():
                    val_pred = model(val_img)
                    val_outputs = [post_trans(i) for i in decollate_batch(val_pred)]
                    dice_metric(y_pred=val_outputs, y=val_seg)

            val_total = np.mean(dice_metric.aggregate("mean").item())
            val_bg, val_b, val_v, val_fgt = dice_metric.aggregate("mean_batch")
            val_metric_values.append(val_total)
            val_bg_values.append(val_bg.item())
            val_b_values.append(val_b.item())
            val_v_values.append(val_v.item())
            val_fgt_values.append(val_fgt.item())
            dice_metric.reset()

            lr_plateau.step(val_total)

            if (epoch + 1) % 10 == 0:
                print(f"epoch {epoch + 1} average training loss: {epoch_loss:.4f}")
                print(f"average training dice total: {train_total:.4f}")
                print(f"average validation total: {val_total:.4f}")
                wandb.log({
                    "epoch": epoch + 1, "epoch_loss": epoch_loss, 
                    "train_mean_total": train_total, "val_mean_total": val_total,
                    "train_dice_breast": train_b.item(), "train_dice_vessels": train_v.item(), "train_dice_fgt": train_fgt.item(),
                    "val_dice_breast": val_b.item(), "val_dice_vessels": val_v.item(), "val_dice_fgt": val_fgt.item(),
                    "lr_plateau": lr_plateau.get_last_lr()[0]
                })

                model_save_path = f"workspace/unet_checkpoints/proc_dukemri/{run.name}/fold{fold + 1}/models"
                os.makedirs(model_save_path, exist_ok=True)
                torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, f"{model_save_path}/model_epoch_{epoch+1}.pt")

        wandb.finish()

    # -------------------------- CLEANUP ---------------------------
    transf_file = json.load(open(f"/workspace/code/config_unet_duke.json", "r"))
    transf_file[run.name]["status"] = "done"
    json.dump(transf_file, open(f"/workspace/code/config_unet_duke.json", "w"), indent=2)

    cuda_json = json.load(open(f"/workspace/code/cuda.json", "r"))
    cuda_json[gpu] = True
    json.dump(cuda_json, open(f"/workspace/code/cuda.json", "w"), indent=2)

# ----------------------------- ENTRY POINT ------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UNet Transformation Pipeline for Duke MRI Dataset")
    parser.add_argument("-t", "--transf_pip", type=str, default=None, help="Transformation pipeline to choose")
    parser.add_argument("-gpu", "--gpu", type=str, default=None, help="Choose the gpu")
    args = parser.parse_args()
    main(args)
