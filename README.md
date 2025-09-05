# Breast Imaging Preprocessing & UNet Training Pipelines

This repository contains the code accompanying the paper:  
**"The Impact of Pre-processing Techniques on Deep Learning Medical Image Segmentation"**  
Submitted to [Conference/Journal Name], 2025.

We focus on evaluating the impact of different preprocessing strategies for breast imaging using two datasets:  
- **CBIS-DDSM (2D mammography)**: Available under the Creative Commons Attribution 3.0 (CC BY 3.0) license in TCIA.
- **Duke Breast Cancer MRI (3D MRI)**: Licensed under the Creative Commons Attribution-NonCommercial 4.0 (CC BY-NC 4.0) in TCIA.

## 🧠 Key Features

- Training pipelines using **MONAI** and **PyTorch**
- 5-fold cross-validation for robustness
- Easily configurable preprocessing pipelines via JSON
- Automatic GPU scheduling via `cuda.json`
- Support for image–segmentation pair CSVs
- **Dockerfile included for reproducibility**

## 📁 Dataset Preparation

### 1. CBIS-DDSM
Download from The Cancer Imaging Archive (TCIA):  
[https://www.cancerimagingarchive.net/collection/cbis-ddsm/](https://www.cancerimagingarchive.net/collection/cbis-ddsm/)

### 2. Duke Breast MRI
Also available on TCIA:  
[https://www.cancerimagingarchive.net/collection/duke-breast-cancer-mri/](https://www.cancerimagingarchive.net/collection/duke-breast-cancer-mri/)

Take advantage of using the "metadata.csv" provided from each dataset to make pairs of path image-segmentation to give to the model. 

## 🐳 Docker Configuration (Recommended)
### Step 1: Build Docker Image
docker build -t breast-imaging-pipeline .

### Step 2: Run Container
docker run --gpus all -v /path/to/data:/data breast-imaging-pipeline \
    python main.py --config config_unet_cbisd.json

### Step 3: Configure GPU Availability
Edit cuda.json to specify which GPUs are currently available. 

Example:
{
  "available_gpus": [0, 2, 3]
}

### Step 4: Run
python main.py --config config_unet_cbisd.json
**or**
python main.py --config config_unet_dukemri.json

## 🤝 Acknowledgements
- MONAI by Project MONAI
- Datasets from TCIA
