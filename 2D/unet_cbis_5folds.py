# ============================ #
#         IMPORTS             #
# ============================ #
import os
import glob
import random
import numpy as np
import pandas as pd
import json
import argparse
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
    LoadImage, LoadImaged, Compose,
    AsDiscrete, Activations, EnsureTyped, EnsureChannelFirstd
)
from monai.data import MetaTensor, DataLoader, ImageDataset, CacheDataset
from monai.networks.nets import UNet
from monai.losses import DiceLoss
from torch.optim import Adam
from monai.metrics import DiceMetric
from monai.handlers import LrScheduleHandler  # Previously used, now replaced
from monai.config import print_config

from transform_utils import transform_lib  # Custom library of MONAI transform functions

# ============================ #
#     SETUP & DETERMINISM     #
# ============================ #
print_config()
monai.utils.set_determinism(seed=0, additional_settings=None)
np.random.seed(0)
random.seed(0)

# ============================ #
#             MAIN            #
# ============================ #
def main(args):
    # Setup GPU or CPU
    gpu = args.gpu
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    print("I'll use:", device)

    transf_pip = args.transf_pip

    # -----------------------------
    # LOAD CONFIGURATION & PARAMS
    # -----------------------------
    img_seg_path = pd.read_csv("/workspace/nas_data/cbis_ddsm/manifest-ZkhPvrLo5216730872708713142/filtered_image_segmentation.csv", delimiter=";")
    conf_file = json.load(open("/workspace/code/config_unet_cbis.json", "r"))
    conf = conf_file[transf_pip]

    wandb_proj_name = "UNet-Seg_CBIS_Preprocess"
    dataset = "CBIS_DDSM"
    transf_parameters = conf["label"]
    unet_norm = Norm.BATCH
    batch_size = 4
    lr = 0.0001
    mode = "max"
    factor = 0.7
    patience = 100
    threshold = 0.0001
    max_epochs = 700
    channels = (16, 32, 64, 128)
    strides = (2, 2, 2)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # ============================ #
    #   5-FOLD CROSS VALIDATION   #
    # ============================ #
    patient_ids = img_seg_path["patient_id"].unique()

    for fold, (train_idx, val_idx) in enumerate(kf.split(patient_ids)):
        print(f"Starting Fold {fold + 1}/5")
        train_ids = patient_ids[train_idx]
        val_ids = patient_ids[val_idx]

        train_dict, val_dict = [], []

        for _, row in img_seg_path.iterrows():
            if row["patient_id"] in train_ids:
                train_dict.append({"image": row["img_nii"], "segmentation": row["seg_nii"]})
            else:
                val_dict.append({"image": row["img_nii"], "segmentation": row["seg_nii"]})

        # -----------------------------
        #     TRANSFORMATION SETUP
        # -----------------------------
        transf_pipeline = [
            LoadImaged(keys=["image", "segmentation"]),
            EnsureTyped(keys=["image", "segmentation"], device=device, track_meta=True),
            EnsureChannelFirstd(keys=["image", "segmentation"], channel_dim="no_channel")
        ]

        for transf in conf["transforms"]:
            transf_pipeline.append(transform_lib[transf["name"]](**transf["params"]))

        preproc = Compose(transf_pipeline)

        # -----------------------------
        #      DATASET & LOADER
        # -----------------------------
        cbis_dataset_train = CacheDataset(data=train_dict, transform=preproc, cache_rate=1.0, num_workers=8, copy_cache=False)
        cbis_dataset_val = CacheDataset(data=val_dict, transform=preproc, cache_rate=1.0, num_workers=4, copy_cache=False)

        train_loader = DataLoader(cbis_dataset_train, num_workers=0, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(cbis_dataset_val, num_workers=0, shuffle=False)

        # -----------------------------
        #      MODEL, LOSS, OPTIM
        # -----------------------------
        model = UNet(spatial_dims=2, in_channels=1, out_channels=1, channels=channels, strides=strides, num_res_units=2, norm=unet_norm).to(device)
        loss_function = DiceLoss(sigmoid=True)
        dice_metric = DiceMetric(include_background=True, reduction="none", get_not_nans=False, num_classes=2)
        optimizer = Adam(model.parameters(), lr=lr)
        lr_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=mode, factor=factor, patience=patience, threshold=threshold)

        epoch_loss_values = []
        train_metric_values = []
        val_metric_values = []
        post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

        # -----------------------------
        #       WANDB LOGGING
        # -----------------------------
        run = wandb.init(
            project=wandb_proj_name,
            name=transf_pip,
            config={
                "dataset": dataset,
                "architecture": "CNN",
                "transf_parameters": transf_parameters,
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
                "fold": f"{transf_pip}_fold{fold + 1}",
            }
        )

        # ============================ #
        #           TRAINING          #
        # ============================ #
        for epoch in range(max_epochs):
            print(f"Fold {fold + 1}, Epoch {epoch + 1}/{max_epochs}")

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
            train_metric = dice_metric.aggregate(reduction="mean").item()
            dice_metric.reset()
            train_metric_values.append(train_metric)
            train_std = np.std(train_metric_values)

            # ============================ #
            #         VALIDATION          #
            # ============================ #
            model.eval()
            val_metric = 0
            val_step = 0

            for val_batch_data in val_loader:
                val_step += 1
                val_img, val_seg = val_batch_data["image"], val_batch_data["segmentation"]

                with torch.no_grad():
                    val_pred = model(val_img)
                    val_outputs = [post_trans(i) for i in decollate_batch(val_pred)]
                    dice_metric(y_pred=val_outputs, y=val_seg)

            val_metric = dice_metric.aggregate(reduction="mean").item()
            dice_metric.reset()
            val_metric_values.append(val_metric)
            val_std = np.std(val_metric_values)

            lr_plateau.step(val_metric)

            # ============================ #
            #       LOGGING & SAVE        #
            # ============================ #
            if (epoch + 1) % 10 == 0:
                save_img_path = f"workspace/unet_checkpoints/proc_cbis/{run.name}/fold{fold + 1}/predictions"
                os.makedirs(save_img_path, exist_ok=True)

                save_image(val_img[0].T.squeeze() / img[0].max(), f"{save_img_path}/inputs.png")
                save_image(val_outputs[0][0, :, :].T, f"{save_img_path}/predictions.png")
                save_image(val_seg[0][0, :, :].T / seg[0].max(), f"{save_img_path}/targets.png")

                print(f"epoch {epoch + 1} average training loss: {epoch_loss:.4f}, lr {lr_plateau.get_last_lr()[0]:.5f}")
                print(f"average training dice: {train_metric:.4f}, {train_std:.4f}")
                print(f"average validation dice: {val_metric:.4f}, {val_std:.4f}")

                wandb.log({
                    "epoch_loss": epoch_loss,
                    "train_dice": train_metric,
                    "train_std": train_std,
                    "val_dice": val_metric,
                    "val_std": val_std,
                    "epoch": epoch + 1,
                    "lr": lr_plateau.get_last_lr()[0]
                })

                model_save_path = f"workspace/unet_checkpoints/proc_cbis/{run.name}/fold{fold + 1}/models"
                os.makedirs(model_save_path, exist_ok=True)
                torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, f"{model_save_path}/model_epoch_{epoch+1}.pt")

        wandb.finish()

    # ============================ #
    #       CLEANUP FLAGS         #
    # ============================ #
    transf_file = json.load(open(f"/workspace/code/config_unet_cbis.json", "r"))
    transf_file[run.name]["status"] = "done"
    json.dump(transf_file, open(f"/workspace/code/config_unet_cbis.json", "w"), indent=2)

    cuda_json = json.load(open(f"/workspace/code/cuda.json", "r"))
    cuda_json[gpu] = True
    json.dump(cuda_json, open(f"/workspace/code/cuda.json", "w"), indent=2)

# ============================ #
#         ENTRY POINT         #
# ============================ #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UNet Transformation Pipeline for CBIS Dataset")
    parser.add_argument("-t", "--transf_pip", type=str, default=None, help="Transformation pipeline to choose")
    parser.add_argument("-gpu", "--gpu", type=str, default="0", help="Choose the GPU")
    args = parser.parse_args()
    main(args)
